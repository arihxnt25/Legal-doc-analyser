"""Fine-tune Legal-BERT on CUAD for multi-label clause classification.

    # Colab (T4 GPU): ~25-35 min for 2 epochs
    python scripts/train_legalbert.py --epochs 2

    # any HF encoder works, e.g. a stronger general model
    python scripts/train_legalbert.py --model microsoft/deberta-v3-base

Uses the exact same segments, labels, official split and metrics as
scripts/train.py, so the two models are directly comparable. Writes
artifacts/legalbert/ (HF model + tokenizer + thresholds + metrics.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contractlens.config import ARTIFACT_DIR, DATA_DIR  # noqa: E402
from contractlens.data.cuad import document_labels, load_cuad, to_segment_dataset  # noqa: E402
from contractlens.models.classifier import evaluate  # noqa: E402
from contractlens.models.transformer import DEFAULT_MODEL, TransformerClauseClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(ARTIFACT_DIR / "legalbert"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train, test = load_cuad(args.data_dir)
    train, test = train[: args.max_train], test[: args.max_test]
    idx = np.random.default_rng(0).permutation(len(train))
    n_val = max(1, int(len(train) * args.val_frac))
    fit_c, val_c = [train[i] for i in idx[n_val:]], [train[i] for i in idx[:n_val]]
    Xf, Yf, Df = to_segment_dataset(fit_c)
    Xv, Yv, Dv = to_segment_dataset(val_c)
    Xt, Yt, Dt = to_segment_dataset(test)

    clf = TransformerClauseClassifier(args.model, max_length=args.max_length)
    logging.info("device=%s fit=%d val=%d test=%d segments", clf.device, len(Xf), len(Xv), len(Xt))
    clf.fit(Xf, Yf, Df, val=(Xv, Yv, Dv), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    clf.metrics = evaluate(clf, Xt, Yt, Dt, document_labels(test))
    clf.metrics["config"] = {**vars(args), "n_fit_segments": len(Xf)}
    clf.save(args.out)
    print("\n=== CUAD test set (official split) ===")
    print(json.dumps(clf.metrics["summary"], indent=2))
    print(f"\nSaved to {args.out}. Serve it with CONTRACTLENS_MODEL={args.out}")


if __name__ == "__main__":
    main()
