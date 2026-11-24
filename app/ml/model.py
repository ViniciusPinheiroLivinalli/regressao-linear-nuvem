"""Pipeline de ML: normalizacao min-max + regressao linear por gradiente descendente.

A regressao e implementacao propria (app/ml/regression.py). Do scikit-learn usamos
o MinMaxScaler, o Pipeline, a divisao temporal e as metricas.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from app.ml.regression import GradientDescentRegressor


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
            ("regressor", GradientDescentRegressor()),
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

    def training_summary(self, max_points: int = 300) -> dict:
        """Epocas, convergencia e a curva de perda do ultimo fit (para o grafico).

        A curva vem em RMSE (raiz do MSE, na mesma unidade de `time`) e reduzida a no
        maximo `max_points` epocas, espacadas em escala log: a perda cai rapido no
        comeco e quase nao muda no fim.
        """
        reg = self.pipeline.named_steps["regressor"]
        losses = np.asarray(reg.loss_history_)
        idx = np.unique(np.geomspace(1, len(losses), num=max_points).astype(int)) - 1
        return {
            "epochs": int(reg.n_epochs_),
            "converged": bool(reg.converged_),
            "learning_rate": float(reg.learning_rate),
            "loss_curve": {
                "epoch": (idx + 1).tolist(),
                "rmse": np.sqrt(losses[idx]).round(4).tolist(),
            },
        }

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
