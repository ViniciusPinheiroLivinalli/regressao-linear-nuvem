"""Issues 4 e 5: casos de uso de treino e predicao (independentes de HTTP)."""
from dataclasses import dataclass
from datetime import datetime, timezone

from app import config
from app.errors import CsvValidationError, NoModelError
from app.ml.model import LinearModel
from app.ml.validation import parse_csv
from app.state import ModelStore, TrainedModel


@dataclass
class PredictionResult:
    labeled: bool
    n_rows: int
    metrics: dict | None
    csv: str
    expected_metrics: dict


def train(store: ModelStore, raw: bytes) -> dict:
    df, _ = parse_csv(raw, mode="train")

    min_rows = max(config.MIN_TRAIN_ROWS, 2 * (config.CV_SPLITS + 1))
    if len(df) < min_rows:
        raise CsvValidationError(
            f"Treino com {len(df)} linhas; o minimo e {min_rows}."
        )

    X, y = df[config.FEATURES], df[config.TARGET]

    store.begin_training()
    try:
        model = LinearModel()
        estimate = model.cross_validate(X, y, n_splits=config.CV_SPLITS)
        model.fit(X, y)  # modelo final com todos os dados de treino
        metrics = {**estimate, "n_rows": len(df)}
        store.commit(
            TrainedModel(
                model=model,
                metrics=metrics,
                trained_at=datetime.now(timezone.utc).isoformat(),
                n_rows=len(df),
            )
        )
        return metrics
    finally:
        store.end_training()


def predict(store: ModelStore, raw: bytes) -> PredictionResult:
    trained = store.get()
    if trained is None:
        raise NoModelError("Nenhum modelo treinado. Envie um CSV de treino primeiro.")

    df, labeled = parse_csv(raw, mode="test")
    X = df[config.FEATURES]

    out = df.copy()
    out["prediction"] = trained.model.predict(X)
    metrics = trained.model.evaluate(X, df[config.TARGET]) if labeled else None

    return PredictionResult(
        labeled=labeled,
        n_rows=len(df),
        metrics=metrics,
        csv=out.to_csv(index=False),
        expected_metrics=trained.metrics,
    )
