"""Train and evaluate the clause classifier on CUAD v1.

    python scripts/train.py                 # downloads CUAD (~18 MB) on first run
    python scripts/train.py --max-train 60  # quick run on a subset (CI / laptops)

Writes artifacts/clause_classifier.joblib, metrics.json and per_category.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contractlens.config import ARTIFACT_DIR, DATA_DIR  # noqa: E402
from contractlens.data.cuad import document_labels, load_cuad, to_segment_dataset  # noqa: E402
from contractlens.models.classifier import ClauseClassifier, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    ap.add_argument("--max-train", type=int, default=None, help="limit training contracts")
    ap.add_argument("--max-test", type=int, default=None, help="limit test contracts")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="train contracts held out for thresholds")
    ap.add_argument("--fast", action="store_true", help="word n-grams only (~5x faster, ~2pt lower AUPR)")
    ap.add_argument("--refit", action="store_true",
                    help="after tuning thresholds, refit on all training contracts (2x time)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train, test = load_cuad(args.data_dir)
    train, test = train[: args.max_train], test[: args.max_test]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(train))
    n_val = max(1, int(len(train) * args.val_frac))
    fit_c, val_c = [train[i] for i in idx[n_val:]], [train[i] for i in idx[:n_val]]

    Xf, Yf, Df = to_segment_dataset(fit_c)
    Xv, Yv, Dv = to_segment_dataset(val_c)
    Xt, Yt, Dt = to_segment_dataset(test)
    logging.info("segments: fit=%d val=%d test=%d", len(Xf), len(Xv), len(Xt))

    # 1) fit on fit-split, tune thresholds on held-out contracts
    clf = ClauseClassifier(C=args.C, use_char=not args.fast).fit(Xf, Yf, Df, val=(Xv, Yv, Dv))
    n_train_segments = len(Xf)
    if args.refit:  # 2) optionally refit on all training contracts, keep the tuned thresholds
        thresholds = clf.thresholds.copy()
        Xa, Ya, Da = to_segment_dataset(train)
        clf = ClauseClassifier(C=args.C, use_char=not args.fast).fit(Xa, Ya, Da)
        clf.thresholds = thresholds
        n_train_segments = len(Xa)

    clf.metrics = evaluate(clf, Xt, Yt, Dt, document_labels(test))
    clf.metrics["config"] = {"C": args.C, "char_ngrams": not args.fast, "refit": args.refit,
                             "n_train_contracts": len(train), "n_train_segments": n_train_segments}
    out = Path(args.out)
    clf.save(out / "clause_classifier.joblib")

    with open(out / "per_category.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "support_docs", "support_segments", "aupr", "p_at_80r", "f1", "doc_f1"])
        for cat, m in clf.metrics["per_category"].items():
            w.writerow([cat, m["support_docs"], m["support_segments"], m["aupr"], m["p_at_80r"], m["f1"],
                        m["doc_f1"]])
    print("\n=== CUAD test set (official split) ===")
    print(json.dumps(clf.metrics["summary"], indent=2))
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
