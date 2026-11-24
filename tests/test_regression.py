"""Prova que a regressao por gradiente descendente do grupo esta correta."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression  # usado so como referencia nos testes
from sklearn.preprocessing import MinMaxScaler

from app.ml.regression import GradientDescentRegressor


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 700, size=(n, 5))  # escala parecida com a base do professor
    y = X @ np.array([-0.4, -0.3, 0.0, 0.4, 0.5]) + 100 + rng.normal(0, 50, n)
    return X, y


def test_matches_exact_least_squares_solution():
    """O problema e convexo: o gradiente descendente tem que chegar no mesmo ponto
    que a solucao exata de minimos quadrados."""
    X, y = _data()
    Xs = MinMaxScaler().fit_transform(X)
    model = GradientDescentRegressor().fit(Xs, y)
    exact, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(len(y))], y, rcond=None)
    assert model.converged_
    assert np.allclose(model.coef_, exact[:-1], atol=1e-4)
    assert np.isclose(model.intercept_, exact[-1], atol=1e-4)


def test_same_predictions_as_scikit_learn():
    X, y = _data(seed=1)
    Xs = MinMaxScaler().fit_transform(X)
    ours = GradientDescentRegressor().fit(Xs, y).predict(Xs)
    reference = LinearRegression().fit(Xs, y).predict(Xs)
    assert np.allclose(ours, reference, atol=1e-3)


def test_loss_decreases_every_epoch():
    X, y = _data(seed=2)
    model = GradientDescentRegressor().fit(MinMaxScaler().fit_transform(X), y)
    losses = np.array(model.loss_history_)
    assert len(losses) == model.n_epochs_
    assert np.all(np.diff(losses) <= 1e-9)  # nunca sobe
    assert losses[-1] < losses[0] / 2


def test_diverges_without_min_max():
    """Sem normalizar, o mesmo passo explode: e por isso que o pipeline usa min-max."""
    X, y = _data(seed=3)
    with pytest.raises(ValueError, match="divergiu"):
        GradientDescentRegressor(max_epochs=2_000).fit(X, y)


def test_stops_at_max_epochs_and_reports_not_converged():
    X, y = _data(seed=4)
    model = GradientDescentRegressor(max_epochs=10).fit(MinMaxScaler().fit_transform(X), y)
    assert model.n_epochs_ == 10 and model.converged_ is False
