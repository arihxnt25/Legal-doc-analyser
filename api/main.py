"""FastAPI service for contract analysis.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contractlens import __version__  # noqa: E402
from contractlens.analysis.analyzer import ContractAnalyzer  # noqa: E402
from contractlens.config import ARTIFACT_DIR, CATEGORIES  # noqa: E402
from contractlens.ingest.parser import SUPPORTED, UnsupportedFormat, extract_text  # noqa: E402
from contractlens.models.transformer import load_any  # noqa: E402

MAX_UPLOAD_MB = 15
STATE: dict = {}


def load_state(model_path: Path | str | None = None) -> None:
    # CONTRACTLENS_MODEL may point to the linear .joblib or a fine-tuned Legal-BERT directory
    path = Path(model_path or os.getenv("CONTRACTLENS_MODEL", ARTIFACT_DIR / "clause_classifier.joblib"))
    if not path.exists():
        raise RuntimeError(f"Model not found at {path}. Train first: python scripts/train.py")
    clf = load_any(path)
    STATE.update(clf=clf, analyzer=ContractAnalyzer(clf))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "clf" not in STATE:
        load_state()
    yield


app = FastAPI(title="ContractLens API", version=__version__, lifespan=lifespan,
              description="Clause extraction (41 CUAD categories), risk review and search for contracts.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TextRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=2_000_000)
    title: str = "contract"
    include_highlights: bool = False


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=2_000_000)
    query: str = Field(..., min_length=2, max_length=500)
    k: int = Field(5, ge=1, le=25)


def _report(text: str, title: str, highlights: bool) -> dict:
    r = STATE["analyzer"].analyze(text, title).to_dict()
    if not highlights:
        r.pop("highlights")
    return r


@app.get("/health")
def health():
    s = STATE["clf"].metrics.get("summary", {})
    return {"status": "ok", "version": __version__, "categories": len(STATE["clf"].categories),
            "model_type": type(STATE["clf"]).__name__,
            "model": {k: s.get(k) for k in ("macro_aupr", "macro_f1", "macro_doc_f1")}}


@app.get("/categories")
def categories():
    return [{"name": c.name, "group": c.group, "description": c.description} for c in CATEGORIES]


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    return _report(req.text, req.title, req.include_highlights)


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...), include_highlights: bool = Query(False)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        text = extract_text(data, filename=file.filename)
    except UnsupportedFormat as e:
        raise HTTPException(415, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if len(text) < 50:
        raise HTTPException(422, "Document contains too little text to analyse")
    return _report(text, file.filename or "upload", include_highlights)


@app.post("/search")
def search(req: SearchRequest):
    return STATE["analyzer"].search(req.text, req.query, req.k)


@app.get("/model/metrics")
def metrics():
    return STATE["clf"].metrics


@app.get("/model/explain/{category:path}")
def explain(category: str, k: int = Query(15, le=50)):
    if category not in STATE["clf"].categories:
        raise HTTPException(404, f"Unknown category. See /categories ({len(CATEGORIES)} available)")
    return {"category": category, "top_terms": STATE["clf"].top_terms(category, k),
            "threshold": float(STATE["clf"].thresholds[STATE["clf"].categories.index(category)])}


@app.get("/formats")
def formats():
    return sorted(SUPPORTED)
