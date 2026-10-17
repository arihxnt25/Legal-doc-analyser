"""Legal-BERT (or any HF encoder) fine-tuned for multi-label clause classification.

Drop-in replacement for ``ClauseClassifier``: same ``predict_proba``,
``thresholds`` and ``categories`` interface, so evaluation, the analyzer, API
and dashboard work unchanged. Requires ``torch`` and ``transformers``
(``pip install -r requirements-transformer.txt``); a GPU (e.g. Colab T4) is
strongly recommended -- see ``notebooks/colab_legalbert.ipynb``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from contractlens.config import CATEGORY_NAMES
from contractlens.models.classifier import with_context

log = logging.getLogger(__name__)
DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"


class TransformerClauseClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 256, categories=None,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.categories = list(categories or CATEGORY_NAMES)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.categories), problem_type="multi_label_classification",
            ignore_mismatched_sizes=True).to(self.device)
        self.thresholds = np.full(len(self.categories), 0.5)
        self.metrics: dict = {}

    # ---------------------------------------------------------------- train
    def fit(self, texts, Y, doc_ids=None, val=None, epochs: int = 2, batch_size: int = 16,
            lr: float = 3e-5, neg_ratio: float = 3.0, seed: int = 0) -> TransformerClauseClassifier:
        """Fine-tune with BCE loss. ~90% of segments carry no label, so negatives
        are down-sampled to ``neg_ratio`` x positives per epoch (re-drawn each
        epoch) and the loss uses a capped per-class ``pos_weight``."""
        import torch
        from torch.nn import BCEWithLogitsLoss
        from transformers import get_linear_schedule_with_warmup

        rng = np.random.default_rng(seed)
        texts = with_context(list(texts), doc_ids)
        pos = np.where(Y.any(1))[0]
        neg = np.where(~Y.any(1))[0]
        n_neg = min(len(neg), int(neg_ratio * len(pos)))
        steps_per_epoch = int(np.ceil((len(pos) + n_neg) / batch_size))
        pos_weight = torch.tensor(np.clip((len(Y) - Y.sum(0)) / np.maximum(Y.sum(0), 1), 1, 50) ** 0.5,
                                  dtype=torch.float32, device=self.device)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_per_epoch * epochs),
                                                steps_per_epoch * epochs)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.model.train()
        for ep in range(epochs):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            rng.shuffle(idx)
            t0, running = time.time(), 0.0
            for step in range(steps_per_epoch):
                b = idx[step * batch_size:(step + 1) * batch_size]
                enc = self._encode([texts[i] for i in b])
                yb = torch.tensor(Y[b], dtype=torch.float32, device=self.device)
                with torch.autocast(self.device, enabled=use_amp):
                    logits = self.model(**enc).logits
                loss = loss_fn(logits.float(), yb)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if (step + 1) % 200 == 0:
                    log.info("epoch %d step %d/%d loss %.4f (%.0fs)", ep + 1, step + 1, steps_per_epoch,
                             running / 200, time.time() - t0)
                    running = 0.0
        if val is not None:
            self.tune_thresholds(*val)
        return self

    # ------------------------------------------------------------ inference
    def _encode(self, batch):
        enc = self.tokenizer(batch, truncation=True, max_length=self.max_length, padding=True,
                             return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def predict_proba(self, texts, doc_ids=None, batch_size: int = 64) -> np.ndarray:
        import torch
        texts = with_context(list(texts), doc_ids)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                logits = self.model(**self._encode(texts[i:i + batch_size])).logits
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(self.categories)), dtype=np.float32)

    def tune_thresholds(self, texts, Y, doc_ids, min_threshold: float = 0.2) -> np.ndarray:
        P = self.predict_proba(texts, doc_ids)
        for k in range(len(self.categories)):
            if Y[:, k].sum() == 0:
                continue
            prec, rec, thr = precision_recall_curve(Y[:, k], P[:, k])
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            self.thresholds[k] = max(float(thr[np.nanargmax(f1[:-1])]), min_threshold) if len(thr) else 0.5
        return self.thresholds

    def top_terms(self, category: str, k: int = 15) -> list:
        return []  # not linear; use attention / integrated gradients for token attributions

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        (d / "contractlens.json").write_text(json.dumps({
            "categories": self.categories, "thresholds": self.thresholds.tolist(),
            "max_length": self.max_length}, indent=2))
        (d / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        return d

    @classmethod
    def load(cls, directory: Path | str, device: str | None = None) -> TransformerClauseClassifier:
        d = Path(directory)
        meta = json.loads((d / "contractlens.json").read_text())
        obj = cls(str(d), max_length=meta["max_length"], categories=meta["categories"], device=device)
        obj.thresholds = np.array(meta["thresholds"])
        if (d / "metrics.json").exists():
            obj.metrics = json.loads((d / "metrics.json").read_text())
        return obj


def load_any(path: Path | str):
    """Load a linear (``.joblib``) or transformer (directory) clause model."""
    path = Path(path)
    if path.is_dir():
        return TransformerClauseClassifier.load(path)
    from contractlens.models.classifier import ClauseClassifier
    return ClauseClassifier.load(path)
