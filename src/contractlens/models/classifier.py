"""Multi-label clause classifier: TF-IDF (word + char n-grams) -> one logistic
regression per CUAD category, with per-category decision thresholds.

Why a linear model first?
  * Clause types are strongly lexical ("shall not assign", "in no event shall
    ... liability exceed"), so n-grams carry most of the signal.
  * Trains on 80K+ segments in minutes on a CPU, is fully inspectable
    (top-weighted n-grams per category = built-in explanation) and gives a
    strong, honest baseline to compare Legal-BERT against
    (``scripts/train_legalbert.py``).

Context window: each segment is classified together with the heading-like
text of its predecessor, because a sub-clause such as "(b) the foregoing
shall survive" only makes sense under its section title.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from contractlens.config import ARTIFACT_DIR, CATEGORY_NAMES

log = logging.getLogger(__name__)


def with_context(texts: list[str], doc_ids: np.ndarray | None = None, n_prev_chars: int = 120) -> list[str]:
    """Prefix each segment with the tail of the previous segment from the same document."""
    out = []
    for i, t in enumerate(texts):
        same_doc = i > 0 and (doc_ids is None or doc_ids[i - 1] == doc_ids[i])
        prev = texts[i - 1][-n_prev_chars:] if same_doc else ""
        out.append(f"{prev} ||| {t}")
    return out


class ClauseClassifier:
    def __init__(self, C: float = 4.0, max_word_features: int = 200_000,
                 max_char_features: int = 150_000, use_char: bool = True, n_jobs: int = -1,
                 categories: list[str] | None = None):
        self.categories = categories or CATEGORY_NAMES
        self.C = C
        self.use_char = use_char
        self.n_jobs = n_jobs
        self.word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.6, sublinear_tf=True,
                                    max_features=max_word_features, lowercase=True,
                                    token_pattern=r"(?u)\b\w[\w'-]*\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    sublinear_tf=True, max_features=max_char_features)
        self.models: list[LogisticRegression | None] = []
        self.priors: np.ndarray | None = None
        self.thresholds = np.full(len(self.categories), 0.5)
        self.metrics: dict = {}

    # ------------------------------------------------------------- features
    def _fit_features(self, texts: list[str]):
        parts = [self.word.fit_transform(texts)]
        if self.use_char:
            parts.append(self.char.fit_transform(texts))
        X = sparse.hstack(parts).tocsr()
        X.sum_duplicates()
        X.sort_indices()
        return X

    def _features(self, texts: list[str]):
        parts = [self.word.transform(texts)]
        if self.use_char:
            parts.append(self.char.transform(texts))
        return sparse.hstack(parts).tocsr()

    # ------------------------------------------------------------------ fit
    def fit(self, texts: list[str], Y: np.ndarray, doc_ids: np.ndarray | None = None,
            val: tuple[list[str], np.ndarray, np.ndarray] | None = None) -> ClauseClassifier:
        t0 = time.time()
        X = self._fit_features(with_context(texts, doc_ids))
        self.priors = Y.mean(0)
        # liblinear releases the GIL, so threads parallelise the 41 independent fits
        self.models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_one)(X, Y[:, k]) for k in range(len(self.categories)))
        log.info("Trained %d category models on %s in %.0fs", sum(m is not None for m in self.models),
                 X.shape, time.time() - t0)
        if val is not None:
            self.tune_thresholds(*val)
        return self

    def _fit_one(self, X, y: np.ndarray) -> LogisticRegression | None:
        if y.sum() < 3:
            return None
        # moderate re-weighting: full "balanced" weights over-predict rare clauses
        w_pos = min(np.sqrt((len(y) - y.sum()) / y.sum()), 30.0)
        return LogisticRegression(C=self.C, solver="liblinear", max_iter=200,
                                  class_weight={0: 1.0, 1: w_pos}).fit(X, y)

    def predict_proba(self, texts: list[str], doc_ids: np.ndarray | None = None) -> np.ndarray:
        X = self._features(with_context(texts, doc_ids))
        P = np.zeros((X.shape[0], len(self.categories)), dtype=np.float32)
        for k, m in enumerate(self.models):
            if m is not None:
                P[:, k] = m.predict_proba(X)[:, 1]
        return P

    def tune_thresholds(self, texts, Y, doc_ids, min_threshold: float = 0.2) -> np.ndarray:
        """Per-category threshold maximising segment-level F1 on held-out contracts."""
        P = self.predict_proba(texts, doc_ids)
        for k in range(len(self.categories)):
            if Y[:, k].sum() == 0 or self.models[k] is None:
                continue
            prec, rec, thr = precision_recall_curve(Y[:, k], P[:, k])
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            self.thresholds[k] = max(float(thr[np.nanargmax(f1[:-1])]), min_threshold) if len(thr) else 0.5
        return self.thresholds

    # ------------------------------------------------------------ explain
    def top_terms(self, category: str, k: int = 15) -> list[tuple[str, float]]:
        m = self.models[self.categories.index(category)]
        if m is None:
            return []
        names = self.word.get_feature_names_out()
        coef = m.coef_.ravel()
        word_n = len(self.word.vocabulary_)
        order = np.argsort(-coef[:word_n])[:k]  # word n-grams are human-readable
        return [(names[i], round(float(coef[i]), 3)) for i in order]

    # --------------------------------------------------------- persistence
    def save(self, path: Path | str = ARTIFACT_DIR / "clause_classifier.joblib") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        (path.parent / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        return path

    @staticmethod
    def load(path: Path | str = ARTIFACT_DIR / "clause_classifier.joblib") -> ClauseClassifier:
        return joblib.load(path)


# ------------------------------------------------------------------ metrics
def precision_at_recall(y: np.ndarray, p: np.ndarray, recall: float) -> float:
    """CUAD's headline metric family: best precision achievable at >= recall."""
    if y.sum() == 0:
        return float("nan")
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec[rec >= recall]
    return float(ok.max()) if len(ok) else 0.0


def evaluate(clf: ClauseClassifier, texts, Y, doc_ids, Ydoc: np.ndarray) -> dict:
    P = clf.predict_proba(texts, doc_ids)
    pred = (P >= clf.thresholds).astype(int)
    per_cat = {}
    n_docs = int(doc_ids.max()) + 1
    Pdoc = np.zeros((n_docs, P.shape[1]), dtype=np.float32)
    np.maximum.at(Pdoc, doc_ids, P)
    pred_doc = (Pdoc >= clf.thresholds).astype(int)
    for k, cat in enumerate(clf.categories):
        y = Y[:, k]
        per_cat[cat] = {
            "support_segments": int(y.sum()),
            "support_docs": int(Ydoc[:, k].sum()),
            "aupr": _r(average_precision_score(y, P[:, k])) if y.sum() else None,
            "p_at_80r": _r(precision_at_recall(y, P[:, k], 0.8)) if y.sum() else None,
            "f1": _r(f1_score(y, pred[:, k], zero_division=0)) if y.sum() else None,
            "doc_f1": _r(f1_score(Ydoc[:, k], pred_doc[:, k], zero_division=0)) if Ydoc[:, k].sum() else None,
            "threshold": _r(clf.thresholds[k]),
        }
    valid = [c for c in per_cat.values() if c["aupr"] is not None]
    summary = {
        "macro_aupr": _r(np.mean([c["aupr"] for c in valid])),
        "macro_p_at_80r": _r(np.mean([c["p_at_80r"] for c in valid])),
        "macro_f1": _r(np.mean([c["f1"] for c in valid])),
        "micro_f1": _r(f1_score(Y, pred, average="micro", zero_division=0)),
        "macro_doc_f1": _r(np.mean([c["doc_f1"] for c in per_cat.values() if c["doc_f1"] is not None])),
        "doc_accuracy": _r(float((pred_doc == Ydoc).mean())),
        "n_test_contracts": n_docs,
        "n_test_segments": len(texts),
    }
    return {"summary": summary, "per_category": per_cat}


def _r(x) -> float:
    return round(float(x), 4)
