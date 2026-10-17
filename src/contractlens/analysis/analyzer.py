"""ContractAnalyzer: one call from raw text to a full review report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from contractlens.analysis.entities import extract_entities
from contractlens.analysis.risk import assess, risk_level
from contractlens.config import CATEGORY_BY_NAME, METADATA_CATEGORIES
from contractlens.ingest.segment import segment
from contractlens.models.classifier import ClauseClassifier
from contractlens.search.retriever import BM25


@dataclass
class Report:
    title: str
    n_chars: int
    n_segments: int
    metadata: dict
    clauses: dict[str, list[dict]]
    absent_categories: list[str]
    findings: list[dict]
    risk_score: int
    risk_level: str
    summary: list[str]
    highlights: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ContractAnalyzer:
    def __init__(self, classifier: ClauseClassifier, max_clauses_per_category: int = 3):
        self.clf = classifier
        self.k = max_clauses_per_category
        self._cache: dict[int, tuple] = {}

    def _segments_and_probs(self, text: str):
        key = hash(text)
        if key not in self._cache:
            segs = segment(text)
            P = self.clf.predict_proba([s.text for s in segs]) if segs else \
                np.zeros((0, len(self.clf.categories)))
            self._cache = {key: (segs, P)}  # keep only the latest document
        return self._cache[key]

    def analyze(self, text: str, title: str = "contract") -> Report:
        t0 = time.time()
        segs, P = self._segments_and_probs(text)
        thr = self.clf.thresholds

        clauses: dict[str, list[dict]] = {}
        highlights = []
        for k, cat in enumerate(self.clf.categories):
            if not len(segs):
                break
            hits = np.where(P[:, k] >= thr[k])[0]
            if not len(hits):
                continue
            hits = hits[np.argsort(-P[hits, k])]
            # confidence: rescale so the category threshold maps to 0.5
            clauses[cat] = [{
                "text": segs[i].text, "start": segs[i].start, "end": segs[i].end,
                "probability": round(float(P[i, k]), 4),
                "confidence": round(float(_calibrate(P[i, k], thr[k])), 3),
                "segment_index": int(i),
            } for i in hits[: self.k]]
            for i in hits:
                highlights.append({"start": segs[i].start, "end": segs[i].end, "category": cat,
                                   "group": CATEGORY_BY_NAME[cat].group,
                                   "probability": round(float(P[i, k]), 4)})

        entities = extract_entities(text)
        findings, score = assess(clauses)
        metadata = {
            "parties": entities["parties"],
            "governing_law": entities["governing_law"],
            "agreement_date": _first_date(clauses.get("Agreement Date"), entities),
            "effective_date": _first_date(clauses.get("Effective Date"), None),
            "expiration": clauses["Expiration Date"][0]["text"] if "Expiration Date" in clauses else None,
            "document_name": clauses["Document Name"][0]["text"][:120] if "Document Name" in clauses else None,
            "entities": entities,
        }
        return Report(
            title=title, n_chars=len(text), n_segments=len(segs), metadata=metadata, clauses=clauses,
            absent_categories=[c for c in self.clf.categories if c not in clauses],
            findings=[f.to_dict() for f in findings], risk_score=score, risk_level=risk_level(score),
            summary=_summary(clauses, metadata, findings),
            highlights=sorted(highlights, key=lambda h: h["start"]),
            elapsed_ms=int((time.time() - t0) * 1000))

    def search(self, text: str, query: str, k: int = 5) -> list[dict]:
        segs, P = self._segments_and_probs(text)
        if not segs:
            return []
        bm = BM25([s.text for s in segs])
        out = []
        for i, score in bm.top_k(query, k):
            j = int(np.argmax(P[i] / np.maximum(self.clf.thresholds, 1e-6)))
            label = self.clf.categories[j] if P[i, j] >= self.clf.thresholds[j] else None
            out.append({"text": segs[i].text, "start": segs[i].start, "end": segs[i].end,
                        "score": score, "category": label})
        return out


def _calibrate(p: float, t: float) -> float:
    return 0.5 + 0.5 * (p - t) / (1 - t) if p >= t else 0.5 * p / max(t, 1e-6)


def _first_date(hits, entities):
    from contractlens.analysis.entities import DATE_RE
    for h in hits or []:
        m = DATE_RE.search(h["text"])
        if m:
            return m.group(0)
    if entities and entities["dates"]:
        return entities["dates"][0]
    return None


def _summary(clauses: dict, meta: dict, findings) -> list[str]:
    lines = []
    if meta["parties"]:
        lines.append("Parties: " + " and ".join(meta["parties"]))
    if meta["agreement_date"]:
        lines.append(f"Dated: {meta['agreement_date']}")
    if meta["governing_law"]:
        lines.append(f"Governing law: {meta['governing_law']}")
    found = [c for c in clauses if c not in METADATA_CATEGORIES]
    if found:
        lines.append(f"{len(found)} substantive clause types detected: " + ", ".join(found[:12])
                     + ("…" if len(found) > 12 else ""))
    high = [f.title for f in findings if f.severity == "HIGH"]
    if high:
        lines.append("Needs attention: " + "; ".join(high))
    return lines
