import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from contractlens.data.cuad import load_cuad, to_segment_dataset  # noqa: E402
from contractlens.models.classifier import ClauseClassifier  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"  # 38 real CUAD contracts (CC BY 4.0), official split kept


@pytest.fixture(scope="session")
def cuad():
    return load_cuad(FIXTURES, download=False)


@pytest.fixture(scope="session")
def trained(cuad, tmp_path_factory):
    train, test = cuad
    X, Y, D = to_segment_dataset(train)
    clf = ClauseClassifier(max_word_features=30_000, use_char=False).fit(X, Y, D)
    clf.thresholds[:] = 0.4
    path = tmp_path_factory.mktemp("model") / "clause_classifier.joblib"
    clf.save(path)
    return clf, path


@pytest.fixture(scope="session")
def sample_text():
    return (ROOT / "data" / "samples" / "master_services_agreement.txt").read_text()
