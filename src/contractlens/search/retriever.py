"""BM25 retrieval over a contract's segments ("where does it talk about X?").

Self-contained Okapi BM25 (k1=1.5, b=0.75) with light legal-aware query
expansion, so search works with no external service. The same interface can
be swapped for a dense / hybrid retriever (e.g. Legal-BERT embeddings) and is
the retrieval half of a RAG Q&A system.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be by for from has have in is it its of on or shall such that the this to
under was were will with any all may hereof herein hereunder thereof""".split())
SYNONYMS = {
    "terminate": ["termination", "terminated", "cancel"],
    "termination": ["terminate", "terminated"],
    "liability": ["liable", "damages", "indemnify", "indemnification"],
    "assign": ["assignment", "assigned", "transfer"],
    "assignment": ["assign", "transfer"],
    "confidential": ["confidentiality", "proprietary", "disclose"],
    "pay": ["payment", "fees", "invoice", "compensation"],
    "payment": ["pay", "fees", "invoice"],
    "renew": ["renewal", "extend", "extension"],
    "law": ["governed", "jurisdiction", "laws"],
    "ip": ["intellectual", "property", "patent", "copyright", "trademark"],
    "compete": ["competition", "competitive", "noncompete"],
    "insurance": ["insured", "coverage", "policy"],
    "audit": ["inspect", "records", "books"],
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str, expand: bool = True) -> list[float]:
        q = tokenize(query)
        weights = Counter({t: 1.0 for t in q})
        if expand:
            for t in q:
                for s in SYNONYMS.get(t, []):
                    weights[s] = max(weights[s], 0.5)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t, w in weights.items():
                f = tf.get(t)
                if f:
                    s += w * self.idf.get(t, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(i, round(sc[i], 3)) for i in order if sc[i] > 0]
