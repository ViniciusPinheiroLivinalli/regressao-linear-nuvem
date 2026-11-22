import io

import numpy as np
import pandas as pd

from app import config
from tests.conftest import to_bytes


def post_csv(client, path, df_or_bytes):
    raw = df_or_bytes if isinstance(df_or_bytes, bytes) else to_bytes(df_or_bytes)
    return client.post(path, files={"file": ("data.csv", raw, "text/csv")})


def test_health_and_initial_status(client):
    assert client.get("/health").json() == {"status": "ok"}
    status = client.get("/status").json()
    assert status["trained"] is False and status["metrics"] is None


def test_index_page_served(client):
    assert client.get("/").status_code == 200


# --- Issue 4: treino ---
def test_train_returns_estimated_metrics(client, train_df):
    r = post_csv(client, "/train", train_df)
    assert r.status_code == 200
    body = r.json()
    for key in ("rmse", "mae", "r2"):
        assert body[f"{key}_mean"] is not None
        assert body[f"{key}_std"] is not None
    assert body["n_rows"] == len(train_df)
    status = client.get("/status").json()
    assert status["trained"] is True and status["metrics"] == body


def test_train_too_few_rows(client, train_df):
    r = post_csv(client, "/train", train_df.head(10))
    assert r.status_code == 422
    assert "minimo" in r.json()["detail"]


def test_retrain_replaces_model(client, train_df):
    post_csv(client, "/train", train_df)
    first = client.get("/status").json()["trained_at"]
    other = train_df.copy() * 3
    post_csv(client, "/train", other)
    assert client.get("/status").json()["trained_at"] != first


def test_invalid_train_keeps_existing_model(client, train_df):
    post_csv(client, "/train", train_df)
    before = client.get("/status").json()
    r = post_csv(client, "/train", b"time-5,time-4\n1,2\n")
    assert r.status_code == 422
    assert client.get("/status").json() == before


def test_train_busy_returns_409(client, train_df):
    client.app.state.store.begin_training()
    r = post_csv(client, "/train", train_df)
    assert r.status_code == 409
    client.app.state.store.end_training()
    assert post_csv(client, "/train", train_df).status_code == 200


def test_upload_too_large_413(client, monkeypatch, train_df):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    assert post_csv(client, "/train", train_df).status_code == 413


# --- Issue 5: predicao ---
def test_predict_without_model_409(client, test_df):
    assert post_csv(client, "/predict", test_df).status_code == 409


def test_predict_labeled(client, train_df, test_df):
    post_csv(client, "/train", train_df)
    r = post_csv(client, "/predict", test_df)
    assert r.status_code == 200
    body = r.json()
    assert body["labeled"] is True and body["n_rows"] == len(test_df)

    out = pd.read_csv(io.StringIO(body["predictions_csv"]))
    assert list(out.columns) == config.TRAIN_COLUMNS + ["prediction"]
    assert np.allclose(out[config.TRAIN_COLUMNS].to_numpy(), test_df.to_numpy())  # ordem preservada

    # metricas reais batem com calculo independente
    err = out["time"] - out["prediction"]
    assert abs(body["metrics"]["rmse"] - np.sqrt((err**2).mean())) < 1e-9
    assert abs(body["metrics"]["mae"] - err.abs().mean()) < 1e-9


def test_predict_unlabeled(client, train_df, test_df):
    post_csv(client, "/train", train_df)
    r = post_csv(client, "/predict", test_df.drop(columns="time"))
    body = r.json()
    assert r.status_code == 200
    assert body["labeled"] is False and body["metrics"] is None
    out = pd.read_csv(io.StringIO(body["predictions_csv"]))
    assert list(out.columns) == config.FEATURES + ["prediction"]
    assert len(out) == len(test_df)


def test_predict_matches_labeled_predictions(client, train_df, test_df):
    post_csv(client, "/train", train_df)
    a = pd.read_csv(io.StringIO(post_csv(client, "/predict", test_df).json()["predictions_csv"]))
    b = pd.read_csv(io.StringIO(
        post_csv(client, "/predict", test_df.drop(columns="time")).json()["predictions_csv"]))
    assert np.allclose(a["prediction"], b["prediction"])


def test_predict_invalid_csv_422(client, train_df):
    post_csv(client, "/train", train_df)
    assert post_csv(client, "/predict", b"a,b\n1,2\n").status_code == 422


# --- Issue 6: reset ---
def test_reset_flow(client, train_df, test_df):
    post_csv(client, "/train", train_df)
    store = client.app.state.store
    assert store.model_path.exists()

    assert client.post("/reset").json() == {"trained": False}
    assert client.get("/status").json()["trained"] is False
    assert not store.model_path.exists()
    assert post_csv(client, "/predict", test_df).status_code == 409

    # retreino com outra base gera modelo novo, sem residuos do anterior
    other = train_df.copy() * 5
    assert post_csv(client, "/train", other).status_code == 200
    assert client.get("/status").json()["trained"] is True


def test_reset_during_training_409(client):
    store = client.app.state.store
    store.begin_training()
    assert client.post("/reset").status_code == 409
    store.end_training()
    assert client.post("/reset").status_code == 200


def test_reset_without_model_is_ok(client):
    assert client.post("/reset").status_code == 200
