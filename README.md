# ⚖️ ContractLens — NLP Legal Contract Analyser

Upload a contract (PDF, DOCX or TXT) and ContractLens:
- **finds 41 types of legal clauses** (liability caps, non-competes, change of control, IP
  assignment, auto-renewal and more),
- **flags risky and missing clauses** with evidence,
- **extracts key terms** (parties, dates, governing law, amounts, durations),
- and lets you **search the contract in plain English**.

It is trained and evaluated on **CUAD**, the expert-annotated Contract Understanding Atticus Dataset
(510 real commercial contracts, 13,000+ lawyer-labelled clauses).

[![CI](https://github.com/arihxnt25/Legal-doc-analyser/actions/workflows/ci.yml/badge.svg)](../../actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> Decision support for contract review, not legal advice.

---

## Results (CUAD v1, official test split — 102 unseen contracts)

Baseline model (TF-IDF word + char n-grams → logistic regression), trained on 347 contracts, thresholds
tuned on 61 held-out training contracts, **evaluated once on the 102-contract official test set**
(18,043 segments):

| Metric | Score | What it means |
|---|---|---|
| **Document-level accuracy** | **90.0%** | "Does this contract contain clause X?" — correct for 41 types × 102 contracts |
| Document-level macro-F1 | 0.731 | Same question, averaged over categories (rare ones count equally) |
| Segment macro-AUPR | 0.549 | Ranking quality for *locating* each clause, threshold-free |
| Segment micro-F1 | 0.578 | Located clauses at the tuned thresholds, all categories pooled |
| Precision @ 80% recall (macro) | 0.287 | CUAD's strict operating point: miss at most 20% of clauses |

**Best and worst categories** (test set):

| Category | Segment AUPR | Doc F1 | Test contracts with clause |
|---|---|---|---|
| Insurance | 0.86 | 0.92 | 32 |
| Governing Law | 0.86 | 0.92 | 83 |
| Expiration Date | 0.80 | 0.90 | 78 |
| Notice Period To Terminate Renewal | 0.80 | 0.79 | 16 |
| Irrevocable Or Perpetual License | 0.79 | 0.67 | 13 |
| Cap On Liability | 0.76 | 0.94 | 44 |
| … | | | |
| Post-Termination Services | 0.33 | 0.69 | 29 |
| Change Of Control | 0.32 | 0.68 | 26 |
| Liquidated Damages | 0.31 | 0.44 | 14 |
| Unlimited/All-You-Can-Eat-License | 0.22 | 0.00 | 3 |
| Third Party Beneficiary | 0.16 | 0.31 | 6 |

Frequent, formulaic clauses (insurance, governing law, liability caps, assignment) are found
reliably. Rare or semantically subtle ones (third-party beneficiary, unlimited licence; fewer than
10 test contracts) are where a transformer should help most. Per-category numbers are in the
dashboard's **Model** tab and `GET /model/metrics`.

**Legal-BERT:** `scripts/train_legalbert.py` fine-tunes `nlpaueb/legal-bert-base-uncased` on the
identical segments, split and metrics. The pipeline is tested end-to-end; run
[`notebooks/colab_legalbert.ipynb`](notebooks/colab_legalbert.ipynb) on a free Colab GPU to fill in
this comparison:

| Model | Macro AUPR | Doc macro-F1 |
|---|---|---|
| TF-IDF + LR (this repo) | 0.549 | 0.731 |
| Legal-BERT | _run notebook_ | _run notebook_ |

> These are segment-level metrics, so they are not directly comparable to the CUAD paper's
> span-extraction AUPR.

## How it works

```
 PDF / DOCX / TXT
        │  parser: pdfplumber · python-docx · normalisation (de-hyphenation, page numbers)
        ▼
 Segmenter ── clause-sized segments with character offsets
        │     (numbered-clause aware, abbreviation-safe, headings merged forward)
        ▼
 Clause classifier ── 41 one-vs-rest models, per-category tuned thresholds
        │   • baseline: TF-IDF word 1-3 + char 3-5 grams → logistic regression
        │   • upgrade:  Legal-BERT fine-tuned (multi-label BCE)        ← same interface
        ▼
 ┌──────────────┬────────────────────┬───────────────────┬──────────────────┐
 │ Clause map   │ Risk review        │ Key terms (regex) │ BM25 search      │
 │ top hits per │ present / missing /│ parties, dates,   │ "can we terminate│
 │ category +   │ combination rules →│ governing law, $, │  early?" → ranked│
 │ highlights   │ findings + 0-100   │ durations, %      │  clauses + label │
 └──────────────┴────────────────────┴───────────────────┴──────────────────┘
        ▼
  FastAPI (/analyze/file, /analyze/text, /search, /model/*)  ·  Streamlit review console
```

### Key design decisions

| Problem | Decision |
|---|---|
| CUAD is QA-style span extraction, which needs a GPU and is slow on 50-page contracts | Reframed as **multi-label segment classification**: clause-sized segments with offsets, labelled by span overlap. Runs on a CPU in about 1 s per contract and still highlights where each clause is. |
| Contracts don't split cleanly into sentences | Custom segmenter: splits at numbered and lettered clauses, ignores `Inc.` / `No.` / `Sec.`, never splits at `;`, merges headings forward. **100% of the lawyer-annotated spans are covered** (enforced by a test). |
| Sub-clauses ("(b) the foregoing shall survive") lack context | Each segment is classified together with the tail of the previous segment. |
| Severe imbalance (about 90% of segments have no label; some clause types appear in under 3% of contracts) | √(neg/pos) class weights, **per-category thresholds** tuned on held-out contracts, and AUPR / precision-at-80%-recall instead of accuracy. |
| Contract boilerplate leaking between train and test | **Official CUAD contract-level split**; thresholds are tuned on separate held-out training contracts. |
| A risk score that can't be explained is useless to a lawyer | Rule-based findings, each with severity, explanation and **evidence clauses**. The model's top n-grams per category are exposed through `/model/explain`. |
| A missing clause is often the real risk | Absence rules (e.g. no limitation of liability) and **combination rules** (e.g. renewal with no opt-out notice; uncapped liability alongside a cap means a carve-out). |

Full details: [`docs/architecture.md`](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/arihxnt25/Legal-doc-analyser.git
cd Legal-doc-analyser
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/train.py          # downloads CUAD (~18 MB) and trains (~15 min on 2 cores)
python scripts/train.py --fast   # word n-grams only, ~3 min

streamlit run dashboard/app.py                 # http://localhost:8501  (samples included)
uvicorn api.main:app --reload --port 8000      # http://localhost:8000/docs
```

**Legal-BERT:** open [`notebooks/colab_legalbert.ipynb`](notebooks/colab_legalbert.ipynb) on a
Colab T4 GPU (about 35 min). It trains both models on identical data and prints a side-by-side
comparison. Serve the result with `CONTRACTLENS_MODEL=artifacts/legalbert`.

**Docker:** `docker compose up --build`

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/analyze/file` | Upload PDF/DOCX/TXT → full report (clauses, findings, key terms, summary) |
| `POST` | `/analyze/text` | Same, for raw text; `include_highlights=true` returns character offsets |
| `POST` | `/search` | Plain-English question → ranked clauses with predicted category |
| `GET` | `/categories` | The 41 CUAD clause types with groups and descriptions |
| `GET` | `/model/metrics` | Test-set metrics, overall and per category |
| `GET` | `/model/explain/{category}` | Top n-grams the model uses for a category |

```bash
curl -F "file=@data/samples/supply_agreement.txt" localhost:8000/analyze/file
```

Real output on the bundled supply agreement (abridged):

```json
{
  "risk_score": 53, "risk_level": "MEDIUM", "elapsed_ms": 123,
  "summary": [
    "Parties: ORGANIC PREPARATIONS INC. and AGAPE ATP INTERNATIONAL HOLDING LIMITED",
    "Governing law: England",
    "9 substantive clause types detected: Renewal Term, Notice Period To Terminate Renewal, Exclusivity, Rofr/Rofo/Rofn, Change Of Control, Anti-Assignment, Minimum Commitment, Warranty Duration, Insurance",
    "Needs attention: No limitation-of-liability clause found"
  ],
  "findings": [
    {"severity": "HIGH",   "title": "No limitation-of-liability clause found", "kind": "missing"},
    {"severity": "MEDIUM", "title": "Exclusivity commitment", "evidence": [{"text": "The Manufacturer agrees to give the Customer exclusive rights to the marketing, promotion and sales of the new products…", "confidence": 0.74}]},
    {"severity": "MEDIUM", "title": "Right of first refusal / offer / negotiation", "evidence": [{"text": "12. TRANSFER OF INTELLECTUAL PROPERTY The Manufacturer agrees to offer the Customer the first right of refusal…"}]},
    {"severity": "LOW",    "title": "Renewal term with notice period", "evidence": [{"text": "1.3 Renewal This agreement will be automatically renewed at the end of every ten (10) year term…"}]}
  ]
}
```

What the model keys on for *Cap On Liability* (`/model/explain/Cap On Liability`): `sole remedy`,
`exceed`, `not exceed`, `shall not exceed`, `liability`, `liable`, `loss of`.

## Dashboard

- **Review:** review-priority gauge, findings with evidence quotes, and key terms.
- **Clauses:** the whole contract with colour-coded highlights by clause group, and every detected
  clause with a confidence score.
- **Search:** ask "can either party terminate early?" and get the ranked clauses back.
- **Model:** test AUPR for each of the 41 categories, plus the n-grams driving each prediction.

## Project structure

```
├── api/main.py                      FastAPI service
├── dashboard/app.py                 Streamlit review console
├── scripts/
│   ├── train.py                     TF-IDF baseline: train + evaluate on CUAD
│   ├── train_legalbert.py           Legal-BERT fine-tuning (same data, same metrics)
│   └── download_cuad.py
├── src/contractlens/
│   ├── config.py                    41 CUAD categories with groups and descriptions
│   ├── data/cuad.py                 CUAD loading, official split, span→segment labelling
│   ├── ingest/parser.py             PDF / DOCX / TXT extraction and normalisation
│   ├── ingest/segment.py            offset-preserving clause segmenter
│   ├── models/classifier.py         TF-IDF + LR, thresholds, metrics (AUPR, P@80R, doc-F1)
│   ├── models/transformer.py        Legal-BERT classifier with the same interface
│   ├── analysis/analyzer.py         end-to-end contract report
│   ├── analysis/risk.py             present / missing / combination risk rules
│   ├── analysis/entities.py         parties, dates, governing law, money, durations
│   └── search/retriever.py          BM25 clause search with legal query expansion
├── data/samples/                    3 CUAD test-set contracts for the demo
├── notebooks/colab_legalbert.ipynb
├── tests/                           20 tests on a bundled 38-contract CUAD fixture
└── docs/architecture.md
```

## Testing

```bash
pytest -q        # segmentation coverage, labelling, model quality, risk rules, parser, API
ruff check .
```

CI runs the tests on Python 3.10 and 3.12, plus an end-to-end job that downloads CUAD and trains
a model.

## Limitations & next steps

- **Segment-level, not span-level.** Results are not directly comparable to the CUAD paper's
  span-extraction AUPR.
- **Rare categories** (Source Code Escrow, Price Restrictions; fewer than 20 training contracts)
  remain weak. Legal-BERT and data augmentation are the main levers.
- **Contract Q&A with an LLM:** combine the BM25 retriever with Legal-BERT embeddings (hybrid
  retrieval) and have an LLM answer only from the retrieved clauses, with citations.
- **Domain shift:** CUAD covers US commercial contracts. NDAs, employment contracts and
  Indian-law contracts would need in-domain fine-tuning.

## Data & license

Code: MIT. CUAD © The Atticus Project, licensed **CC BY 4.0**
([atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)). The sample contracts and test
fixture are excerpts of CUAD and remain under CC BY 4.0. *Hendrycks, Burns, Chen & Ball, "CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021.*
