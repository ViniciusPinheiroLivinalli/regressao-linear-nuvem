import numpy as np
import pandas as pd
import pytest

from app import config
from app.ml.model import LinearModel, compute_metrics


def xy(df):
    return df[config.FEATURES], df[config.TARGET]


def test_scaler_fitted_only_on_train(train_df, test_df):
    X, y = xy(train_df)
    model = LinearModel().fit(X, y)
    scaler = model.pipeline.named_steps["scaler"]
    min_before = scaler.data_min_.copy()
    max_before = scaler.data_max_.copy()

    extreme = test_df.copy()
    extreme[config.FEATURES] = extreme[config.FEATURES] * 100  # fora da faixa do treino
    model.predict(extreme[config.FEATURES])

    assert np.array_equal(scaler.data_min_, min_before)
    assert np.array_equal(scaler.data_max_, max_before)
    assert np.allclose(min_before, X.min().to_numpy())
    assert np.allclose(max_before, X.max().to_numpy())


def test_linear_relation_gives_r2_near_one():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, (200, 5)), columns=config.FEATURES)
    y = pd.Series(X.to_numpy() @ np.array([1, 2, 3, 4, 5]) + 7)
    model = LinearModel().fit(X, y)
    assert model.evaluate(X, y)["r2"] > 0.999


def test_metrics_match_independent_calculation():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.0, 2.0, 5.0])
    m = compute_metrics(y, p)
    err = y.to_numpy() - p
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(err)))
    assert m["r2"] == pytest.approx(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2))


def test_r2_none_with_single_sample():
    assert compute_metrics(pd.Series([1.0]), np.array([1.0]))["r2"] is None


def test_cross_validate_is_chronological(train_df):
    X, y = xy(train_df)
    summary = LinearModel().cross_validate(X, y, n_splits=5)
    assert summary["n_splits"] == 5
    for key in ("rmse", "mae", "r2"):
        assert summary[f"{key}_mean"] is not None
        assert summary[f"{key}_std"] >= 0


def test_save_and_load_roundtrip(tmp_path, train_df, test_df):
    X, y = xy(train_df)
    model = LinearModel().fit(X, y)
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = LinearModel.load(path)
    Xt = test_df[config.FEATURES]
    assert np.allclose(model.predict(Xt), loaded.predict(Xt))
