# Architecture & design notes

## 1. Problem framing

CUAD ships as SQuAD-style extractive QA: for each of 41 clause categories, lawyers highlighted the
spans "that should be reviewed by a lawyer". The CUAD paper fine-tunes QA models (RoBERTa/DeBERTa)
that read 512-token windows and predict start/end positions, which needs a GPU and is slow on 50-page
contracts.

ContractLens reframes this as **multi-label segment classification**:

1. Split each contract into clause-sized segments (keeping character offsets).
2. Label a segment with every category whose annotated span substantially overlaps it.
3. Train one binary classifier per category.

This keeps what a reviewer needs (*where* each clause type is, with highlightable offsets), runs on a
CPU in about 1 second per contract, and allows a transformer to be dropped in with no other changes.
The trade-off is honest: we locate clauses at segment granularity, not exact token spans, so our
numbers are **not directly comparable** to the paper's span-level AUPR.

## 2. Ingestion — `contractlens.ingest`

**`parser.py`**
- PDF via `pdfplumber` (raises a clear error on scanned PDFs with no text layer), DOCX via
  `python-docx` (paragraphs and tables), TXT/MD.
- Normalisation: NFKC Unicode, de-hyphenation across line breaks, stripping bare page numbers,
  collapsing whitespace.

**`segment.py`**
- Pass 1 splits on blank lines and on lines that start a numbered or lettered clause
  (`12.`, `(b)`, `ARTICLE`).
- Pass 2 splits long blocks at sentence ends, but not after abbreviations (`Inc.`, `No.`, `Sec.`,
  `e.g.`) and not at `;`, because legal lists use it inside one obligation.
- Run-ons longer than 1,200 characters are hard-wrapped at whitespace. Fragments under 40
  characters, usually headings, are merged forward so a heading travels with its clause.
- On CUAD this gives ~84K training segments (median 196 characters). **100% of annotated spans are
  covered** by at least one segment; a test enforces this.

## 3. Labels and split — `contractlens.data.cuad`

- **Labelling rule.** A segment is positive for category *k* if an annotated span covers ≥ 50% of
  the segment, or the segment covers ≥ 50% of the span. The second condition matters for short
  answers such as dates and party names.
- **Split.** The official CUAD test split is used: 102 test contracts and 408 training contracts. No
  contract appears in both, so there is no leakage of contract boilerplate.
- **Thresholds.** 15% of training contracts are held out to tune per-category thresholds.

## 4. Models — `contractlens.models`

### Baseline: TF-IDF + logistic regression (`classifier.py`)

- **Features**
  - Word 1–3-grams (200K features, sublinear TF, `max_df=0.6`).
  - Character 3–5-grams within word boundaries (150K features). These add robustness to OCR
    noise, defined terms and inflections, and gave +1.9 points of AUPR on a held-out category
    sample at roughly 5× the training cost.
- **Context.** Each segment is prefixed with the last 120 characters of the previous segment, so
  sub-clauses see their section heading.
- **Classifiers**
  - One logistic regression per category (liblinear, C = 4), fitted in parallel threads.
  - The positive class is up-weighted by √(neg/pos), capped at 30. Full "balanced" weighting
    over-predicts rare clauses.
- **Thresholds.** Each category's threshold maximises segment-level F1 on held-out contracts
  (floor 0.2).
- **Explainability.** The highest-weighted n-grams per category (`top_terms`, `/model/explain`)
  show what the model keys on.

### Legal-BERT (`transformer.py`, `scripts/train_legalbert.py`)

- **Base model:** `nlpaueb/legal-bert-base-uncased`, pre-trained on 12 GB of legal text (EU and
  UK legislation, court cases, US contracts from EDGAR), so its pre-training domain overlaps CUAD.
- **Task head:** multi-label head with `BCEWithLogitsLoss`, and per-class `pos_weight` set to
  √(neg/pos), capped.
- **Negative down-sampling.** About 90% of segments have no label, so each epoch uses all positives
  plus 3× as many randomly re-drawn negatives.
- **Training settings:** AdamW (lr 3e-5), linear warm-up/decay, mixed precision on GPU, gradient
  clipping, max length 256 tokens.
- **Same interface** (`predict_proba`, `thresholds`, `categories`) and same evaluation. The API and
  dashboard load either model through `load_any()`.

## 5. Metrics — `classifier.evaluate`

| Metric | Why |
|---|---|
| **AUPR** per category (macro) | Threshold-free and honest under heavy class imbalance; CUAD's headline metric family |
| **Precision @ 80% recall** | CUAD's operating point: a reviewer can tolerate some false positives but must not miss clauses |
| Segment F1 (macro and micro) | Quality at the tuned thresholds |
| **Document-level F1** | "Does this contract contain an X clause?" — the question that drives risk findings. Document score = max segment score. |

## 6. Analysis — `contractlens.analysis`

**`analyzer.py`** turns the text into a report:
1. Segment the text and predict probabilities for each segment.
2. For each category, keep the top 3 segments above threshold.
3. Rescale confidence so that the threshold maps to 0.5.
4. Emit highlight spans (offsets + category + group).

**`entities.py`** uses regexes for dates, money, durations, percentages, governing-law
jurisdiction (US states and common countries), and parties from the preamble ("by and between X
and Y"). Regexes are exact and auditable for these rigid formats, and they complement the classifier
by reading the value inside the clause the classifier found.

**`risk.py`** produces review findings:
- **Present-clause rules** (e.g. Uncapped Liability → HIGH, Exclusivity → MEDIUM).
- **Missing-clause rules** (no Cap On Liability → HIGH, no Governing Law → MEDIUM).
- **Combination rules**, for example:
  - Renewal without a notice period → "possible silent auto-renewal".
  - Uncapped liability *and* a cap → "probably a carve-out, check which claims sit outside".
  - Non-compete *with* carve-outs → severity downgraded.
- Every finding carries evidence segments.
- The 0–100 **review-priority** score sums severity points.

Risk is perspective-dependent, so findings are worded as review flags, not verdicts.

## 7. Search — `contractlens.search.retriever`

Self-contained Okapi BM25 (k1 = 1.5, b = 0.75) over a contract's segments, with a small legal
synonym map for query expansion (terminate → termination, cancel; liability → damages, indemnify;
and so on). Each hit is tagged with its predicted clause category. This is the retrieval half of a
contract Q&A (RAG) system. The next step is hybrid retrieval with Legal-BERT embeddings, plus an LLM
that answers only from the retrieved clauses and cites them.

## 8. Serving

- **FastAPI** (`api/main.py`):
  - Endpoints: `/analyze/text`, `/analyze/file` (PDF/DOCX/TXT, 15 MB limit, 415 for unsupported
    types, 422 for empty or scanned files), `/search`, `/categories`, `/model/metrics`,
    `/model/explain/{category}`.
  - Model choice through `CONTRACTLENS_MODEL`: a `.joblib` file for the linear model, a directory
    for Legal-BERT.
- **Streamlit** (`dashboard/app.py`), four tabs:
  - Review: priority gauge, findings with evidence, key terms.
  - Clauses: the full contract with colour-coded highlights by clause group, plus a per-category
    panel.
  - Search.
  - Model: per-category test AUPR and top n-grams.

## 9. Limitations

- Segment-level rather than span-level extraction.
- Regex entities cover common formats, not every jurisdiction or date style.
- Risk weights are hand-set; a learned or lawyer-calibrated weighting would be better.
- CUAD covers US commercial contracts from SEC filings. Other contract types (NDAs, employment,
  Indian-law contracts) would need fine-tuning on in-domain data.
