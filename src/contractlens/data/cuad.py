"""CUAD v1 (Contract Understanding Atticus Dataset) loading and segment labelling.

CUAD: 510 commercial contracts from SEC EDGAR, annotated by lawyers with
13,000+ spans across 41 clause categories (Hendrycks et al., NeurIPS 2021).
License: CC BY 4.0. https://www.atticusprojectai.org/cuad

CUAD is distributed as SQuAD-style extractive QA. We convert it to a
**multi-label segment classification** task: each contract is segmented
(``ingest.segment``) and a segment receives every category whose annotated
answer span overlaps it substantially. The official CUAD test split (102
contracts) is used for evaluation, so no contract appears in both splits.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contractlens.config import CATEGORY_NAMES, CUAD_URL, DATA_DIR
from contractlens.ingest.segment import Segment, segment

log = logging.getLogger(__name__)


@dataclass
class Contract:
    title: str
    text: str
    spans: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)  # cat -> [(start, end, text)]


def download_cuad(data_dir: Path | str = DATA_DIR, url: str = CUAD_URL) -> Path:
    data_dir = Path(data_dir)
    target = data_dir / "CUADv1.json"
    if target.exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading CUAD (~18 MB) from %s", url)
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed https URL
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in ("CUADv1.json", "test.json"):
            member = next(m for m in z.namelist() if m.endswith(name))
            (data_dir / name).write_bytes(z.read(member))
    return data_dir


def _parse(squad: dict) -> list[Contract]:
    out = []
    for doc in squad["data"]:
        p = doc["paragraphs"][0]
        c = Contract(doc["title"], p["context"])
        for q in p["qas"]:
            cat = q["id"].split("__", 1)[1]
            spans = {(a["answer_start"], a["answer_start"] + len(a["text"]), a["text"])
                     for a in q["answers"]}
            c.spans[cat] = sorted(spans)
        out.append(c)
    return out


def load_cuad(data_dir: Path | str = DATA_DIR, download: bool = True) -> tuple[list[Contract], list[Contract]]:
    """Return (train, test) contracts using the official CUAD test split."""
    data_dir = Path(data_dir)
    if download:
        download_cuad(data_dir)
    full = _parse(json.loads((data_dir / "CUADv1.json").read_text()))
    test_titles = {d["title"] for d in json.loads((data_dir / "test.json").read_text())["data"]}
    train = [c for c in full if c.title not in test_titles]
    test = [c for c in full if c.title in test_titles]
    log.info("CUAD: %d train / %d test contracts", len(train), len(test))
    return train, test


def label_segments(contract: Contract, segs: list[Segment], min_overlap: float = 0.5,
                   categories: list[str] = CATEGORY_NAMES) -> np.ndarray:
    """Multi-hot labels: category k is positive if an answer span covers at
    least ``min_overlap`` of the segment, or the segment covers at least
    ``min_overlap`` of a (short) answer span."""
    y = np.zeros((len(segs), len(categories)), dtype=np.int8)
    starts = np.array([s.start for s in segs])
    ends = np.array([s.end for s in segs])
    seg_len = np.maximum(ends - starts, 1)
    for k, cat in enumerate(categories):
        for a0, a1, _ in contract.spans.get(cat, []):
            inter = np.minimum(ends, a1) - np.maximum(starts, a0)
            hit = inter > 0
            if not hit.any():
                continue
            frac_seg = inter / seg_len
            frac_ans = inter / max(a1 - a0, 1)
            y[hit & ((frac_seg >= min_overlap) | (frac_ans >= min_overlap)), k] = 1
    return y


def to_segment_dataset(contracts: list[Contract], **seg_kwargs):
    """Flatten contracts into (texts, Y, doc_ids) for training."""
    texts, ys, docs = [], [], []
    for i, c in enumerate(contracts):
        segs = segment(c.text, **seg_kwargs)
        texts.extend(s.text for s in segs)
        ys.append(label_segments(c, segs))
        docs.extend([i] * len(segs))
    return texts, np.vstack(ys), np.array(docs)


def document_labels(contracts: list[Contract], categories: list[str] = CATEGORY_NAMES) -> np.ndarray:
    """Per-contract presence of each category (did lawyers find any span?)."""
    return np.array([[int(bool(c.spans.get(cat))) for cat in categories] for c in contracts], dtype=np.int8)
