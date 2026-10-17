"""Download CUAD v1 (CC BY 4.0) into data/cuad/. Training scripts call this automatically."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contractlens.config import DATA_DIR  # noqa: E402
from contractlens.data.cuad import download_cuad  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CUAD available in", download_cuad(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR))
