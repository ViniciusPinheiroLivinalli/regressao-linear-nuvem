"""Issue 3: pipeline de ML (MinMax + regressao linear).

A interface publica (fit / predict / evaluate / save / load) nao depende da
implementacao da regressao. Para trocar a LinearRegression do scikit-learn por
uma implementacao propria, basta substituir o estimador em `_build_pipeline`
(desde que siga a API fit/predict do scikit-learn).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def _clean(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE e R2 (R2 fica None com menos de 2 amostras)."""
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None
    return {
        "rmse": _clean(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": _clean(mean_absolute_error(y_true, y_pred)),
        "r2": _clean(r2),
    }


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("regressor", LinearRegression()),
        ]
    )


class LinearModel:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline if pipeline is not None else _build_pipeline()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        # O scaler e ajustado aqui, somente com os dados de treino.
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Somente transform: o teste nunca altera min/max do scaler.
        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        return compute_metrics(y, self.predict(X))

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int) -> dict:
        """Validacao cronologica (janela expansiva). O scaler e reajustado a
        cada fold, usando apenas as linhas de treino do fold."""
        folds = []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            fold_pipeline = clone(self.pipeline)
            fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = fold_pipeline.predict(X.iloc[test_idx])
            folds.append(compute_metrics(y.iloc[test_idx], preds))

        summary: dict = {"n_splits": n_splits}
        for key in ("rmse", "mae", "r2"):
            values = [f[key] for f in folds if f[key] is not None]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_std"] = float(np.std(values)) if values else None
        return summary

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "LinearModel":
        return cls(joblib.load(path))
