import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.main import create_app  # noqa: E402
from gerar_amostra import make_series_frame  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app = create_app(store_dir=tmp_path / "store")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def train_df():
    return make_series_frame(120, seed=1)


@pytest.fixture
def test_df():
    return make_series_frame(30, seed=2)


def to_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()
