"""Split a contract into clause-sized segments while keeping character offsets.

Contracts are not prose: headings, numbered sub-clauses and long run-on
sentences are the norm. The segmenter works in two passes:

1. Split on blank lines / line breaks into blocks.
2. Split long blocks at sentence boundaries ("; " is *not* a boundary --
   legal lists use it inside one obligation), then merge tiny fragments
   (headings like "12. TERM.") into the following segment so each segment
   carries enough context to classify.

Offsets let us map predictions back onto the original document for
highlighting and let us label CUAD segments by answer-span overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(r"\n\s*\n|\n(?=\s*(?:\(?[0-9]{1,2}(?:\.[0-9]+)*[.)]|\([a-z]{1,3}\)|[A-Z][A-Z ]{3,}:?\s))")
# sentence end candidate: . ! ? followed by space + capital/number/quote/paren
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"(A-Z0-9])")
# ...unless the period belongs to an abbreviation ("Inc. ", "No. 5", "Sec. 2", "e.g. ")
_ABBREV = re.compile(r"(?:\b(?:Inc|Ltd|Co|Corp|No|Nos|Sec|Secs|St|Mr|Ms|Mrs|Dr|vs|Art|para|e\.g|i\.e|U\.S|N\.A)"
                     r"|\b[A-Z])\.$")


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int


def _iter_blocks(text: str):
    pos = 0
    for m in _BLOCK.finditer(text):
        yield pos, m.start()
        pos = m.end()
    yield pos, len(text)


def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def segment(text: str, max_chars: int = 600, min_chars: int = 40) -> list[Segment]:
    """Return non-overlapping segments covering the meaningful text."""
    raw: list[tuple[int, int]] = []
    for bs, be in _iter_blocks(text):
        bs, be = _strip_span(text, bs, be)
        if be <= bs:
            continue
        if be - bs <= max_chars:
            raw.append((bs, be))
            continue
        cur = bs
        for m in _SENT.finditer(text, bs, be):
            if _ABBREV.search(text, max(bs, m.start() - 6), m.start()):
                continue
            if m.start() - cur >= min_chars:
                raw.append((cur, m.start()))
                cur = m.end()
        raw.append((cur, be))
        # hard-wrap pathological run-ons (tables, OCR noise) at whitespace
    wrapped: list[tuple[int, int]] = []
    for s, e in raw:
        while e - s > 2 * max_chars:
            cut = text.rfind(" ", s + max_chars // 2, s + max_chars)
            cut = cut if cut > s else s + max_chars
            wrapped.append((s, cut))
            s = cut + 1
        wrapped.append((s, e))

    merged: list[list[int]] = []
    carry: int | None = None  # start of a short fragment waiting to be merged forward
    for s, e in wrapped:
        s, e = _strip_span(text, s, e)
        if e <= s:
            continue
        if carry is not None:
            s, carry = carry, None
        if e - s < min_chars:
            carry = s
            continue
        merged.append([s, e])
    if carry is not None:
        if merged:
            merged[-1][1] = max(merged[-1][1], _strip_span(text, carry, len(text))[1])
        else:
            s, e = _strip_span(text, carry, len(text))
            if e > s:
                merged.append([s, e])
    return [Segment(re.sub(r"\s+", " ", text[s:e]), s, e) for s, e in merged]
