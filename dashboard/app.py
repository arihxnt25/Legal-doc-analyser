"""Streamlit contract review console.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contractlens.analysis.analyzer import ContractAnalyzer  # noqa: E402
from contractlens.config import ARTIFACT_DIR, CATEGORY_BY_NAME  # noqa: E402
from contractlens.ingest.parser import extract_text  # noqa: E402
from contractlens.models.transformer import load_any  # noqa: E402

st.set_page_config(page_title="ContractLens", page_icon="⚖️", layout="wide")

SEV_COLOR = {"HIGH": "#c92a2a", "MEDIUM": "#e67700", "LOW": "#1971c2", "INFO": "#868e96"}
GROUP_COLOR = {"General": "#e9ecef", "Term": "#d0ebff", "Restrictive": "#ffe3e3", "Termination": "#fff3bf",
               "Financial": "#d3f9d8", "IP": "#e5dbff", "Liability": "#ffd8a8"}
SAMPLES = sorted((ROOT / "data" / "samples").glob("*.txt"))
MODEL_PATH = Path(os.getenv("CONTRACTLENS_MODEL", ARTIFACT_DIR / "clause_classifier.joblib"))


@st.cache_resource
def load():
    clf = load_any(MODEL_PATH)
    return clf, ContractAnalyzer(clf)


@st.cache_data(show_spinner="Analysing contract…")
def analyse(text: str, title: str) -> dict:
    return load()[1].analyze(text, title).to_dict()


def highlighted_html(text: str, highlights: list[dict], selected: set[str]) -> str:
    spans = [h for h in highlights if h["category"] in selected]
    # merge by segment: one segment may carry several categories
    by_seg: dict[tuple[int, int], list[str]] = {}
    for h in spans:
        by_seg.setdefault((h["start"], h["end"]), []).append(h["category"])
    out, pos = [], 0
    for (s, e), cats in sorted(by_seg.items()):
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        color = GROUP_COLOR[CATEGORY_BY_NAME[cats[0]].group]
        tag = " · ".join(cats)
        out.append(f"<mark style='background:{color};padding:1px 2px;border-radius:3px' title='{tag}'>"
                   f"{html.escape(text[s:e])}<sup style='font-size:0.65em;color:#495057'> [{tag}]</sup></mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    body = "".join(out).replace("\n", "<br>")
    return (f"<div style='height:560px;overflow-y:auto;font-family:Georgia,serif;font-size:0.92rem;"
            f"line-height:1.6;padding:12px;border:1px solid #dee2e6;border-radius:6px'>{body}</div>")


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python scripts/train.py` first.")
    st.stop()

clf, analyzer = load()
st.title("⚖️ ContractLens — Contract Review Console")
st.caption("Clause extraction across 41 CUAD categories · risk review · clause search. "
           "Decision support for lawyers, not legal advice.")

with st.sidebar:
    st.header("Document")
    up = st.file_uploader("Upload a contract", type=["pdf", "docx", "txt", "md"])
    sample = st.selectbox("…or pick a sample (CUAD test set)", ["—"] + [p.name for p in SAMPLES])
    text, title = None, None
    if up is not None:
        try:
            text, title = extract_text(up.getvalue(), filename=up.name), up.name
        except Exception as e:  # noqa: BLE001 - surface any parser error to the user
            st.error(str(e))
    elif sample != "—":
        p = next(p for p in SAMPLES if p.name == sample)
        text, title = p.read_text(), p.stem

tab_review, tab_clauses, tab_search, tab_model = st.tabs(["Review", "Clauses", "Search", "Model"])

if text:
    rep = analyse(text, title)

    with tab_review:
        c = st.columns([1, 1, 1, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=rep["risk_score"], title={"text": f"Review priority · {rep['risk_level']}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": SEV_COLOR.get(rep["risk_level"], "#495057")},
                   "steps": [{"range": [0, 30], "color": "#ebfbee"}, {"range": [30, 60], "color": "#fff9db"},
                             {"range": [60, 100], "color": "#fff5f5"}]}))
        gauge.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=0))
        c[0].plotly_chart(gauge, width="stretch")
        sev = pd.Series([f["severity"] for f in rep["findings"]]).value_counts()
        c[1].metric("High-severity findings", int(sev.get("HIGH", 0)))
        c[1].metric("Medium", int(sev.get("MEDIUM", 0)))
        c[2].metric("Clause types found", len(rep["clauses"]))
        c[2].metric("Segments analysed", rep["n_segments"])
        c[3].metric("Analysis time", f"{rep['elapsed_ms']} ms")
        c[3].metric("Characters", f"{rep['n_chars']:,}")

        st.subheader("Summary")
        for line in rep["summary"]:
            st.markdown(f"- {line}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Findings")
            for f in rep["findings"]:
                if f["severity"] == "INFO":
                    continue
                badge = (f"<span style='background:{SEV_COLOR[f['severity']]};color:white;padding:1px 7px;"
                         f"border-radius:4px;font-size:0.75em'>{f['severity']}</span>")
                with st.expander(f"{f['severity']} · {f['title']}", expanded=f["severity"] == "HIGH"):
                    st.markdown(badge + " " + f["explanation"], unsafe_allow_html=True)
                    for ev in f["evidence"]:
                        st.markdown(f"> {ev['text'][:500]}  \n<small>confidence {ev['confidence']:.2f}</small>",
                                    unsafe_allow_html=True)
                    if f["kind"] == "missing":
                        st.caption("No clause of this type was detected anywhere in the document.")
        with right:
            st.subheader("Key terms")
            m = rep["metadata"]
            ent = m["entities"]
            st.markdown(f"**Parties:** {', '.join(m['parties']) or '—'}  \n"
                        f"**Agreement date:** {m['agreement_date'] or '—'}  \n"
                        f"**Governing law:** {m['governing_law'] or '—'}")
            if m["expiration"]:
                st.markdown(f"**Term / expiry:** {m['expiration'][:300]}")
            for label, key in (("Monetary amounts", "monetary_amounts"), ("Durations", "durations"),
                               ("Percentages", "percentages")):
                if ent[key]:
                    st.markdown(f"**{label}:** " + ", ".join(ent[key][:10]))

    with tab_clauses:
        groups = sorted({CATEGORY_BY_NAME[c].group for c in rep["clauses"]})
        sel_groups = st.multiselect("Clause groups", groups, groups)
        cats = [c for c in rep["clauses"] if CATEGORY_BY_NAME[c].group in sel_groups]
        left, right = st.columns([3, 2])
        with left:
            st.markdown(highlighted_html(text, rep["highlights"], set(cats)), unsafe_allow_html=True)
        with right:
            for cat in cats:
                hits = rep["clauses"][cat]
                with st.expander(f"{cat} ({len(hits)})"):
                    st.caption(CATEGORY_BY_NAME[cat].description)
                    for h in hits:
                        st.markdown(f"> {h['text'][:600]}")
                        st.progress(min(h["confidence"], 1.0), text=f"confidence {h['confidence']:.2f}")
            with st.expander(f"Not found ({len(rep['absent_categories'])})"):
                st.write(", ".join(rep["absent_categories"]))

    with tab_search:
        q = st.text_input("Ask where the contract covers something",
                          "Can either party terminate the agreement early?")
        if q:
            for r in analyzer.search(text, q, k=6):
                tag = f" · _{r['category']}_" if r["category"] else ""
                st.markdown(f"**score {r['score']:.2f}**{tag}\n\n> {r['text'][:700]}")
else:
    for t in (tab_review, tab_clauses, tab_search):
        with t:
            st.info("Upload a contract or pick a sample in the sidebar.")

with tab_model:
    s = clf.metrics.get("summary", {})
    c = st.columns(5)
    c[0].metric("Macro AUPR", f"{s.get('macro_aupr', 0):.3f}")
    c[1].metric("Macro P@80%R", f"{s.get('macro_p_at_80r', 0):.3f}")
    c[2].metric("Segment micro-F1", f"{s.get('micro_f1', 0):.3f}")
    c[3].metric("Document macro-F1", f"{s.get('macro_doc_f1', 0):.3f}")
    c[4].metric("Test contracts", s.get("n_test_contracts", 0))
    st.caption("Evaluated on the official CUAD v1 test split (102 contracts never seen in training).")
    per = pd.DataFrame(clf.metrics.get("per_category", {})).T.reset_index(names="category")
    if len(per):
        per = per.dropna(subset=["aupr"]).sort_values("aupr")
        fig = px.bar(per, x="aupr", y="category", orientation="h", height=900,
                     color=per["category"].map(lambda c: CATEGORY_BY_NAME[c].group),
                     hover_data=["support_docs", "f1", "doc_f1"])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None, legend_title="group",
                          xaxis_title="segment-level AUPR (test)")
        st.plotly_chart(fig, width="stretch")
        cat = st.selectbox("Why does the model predict…", list(clf.categories), index=35)
        terms = clf.top_terms(cat, 20)
        if terms:
            st.write(pd.DataFrame(terms, columns=["n-gram", "weight"]))
        else:
            st.caption("Term weights are only available for the linear model.")
    with st.expander("Raw metrics JSON"):
        st.code(json.dumps(s, indent=2))
