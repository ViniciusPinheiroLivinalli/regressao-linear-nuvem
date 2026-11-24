"""Issues 4 e 5: casos de uso de treino e predicao (independentes de HTTP)."""
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

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
    warnings: list[str]


def train(store: ModelStore, raw: bytes) -> tuple[dict, list[str]]:
    """Treina e devolve (metricas esperadas, avisos da leitura do CSV)."""
    parsed = parse_csv(raw, mode="train")
    df = parsed.data

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
        return metrics, parsed.warnings
    finally:
        store.end_training()


def predict(store: ModelStore, raw: bytes) -> PredictionResult:
    trained = store.get()
    if trained is None:
        raise NoModelError("Nenhum modelo treinado. Envie um CSV de treino primeiro.")

    parsed = parse_csv(raw, mode="test")
    df, labeled = parsed.data, parsed.labeled
    X = df[config.FEATURES]

    # Saida: colunas do arquivo na ordem original (extras como vieram) + prediction.
    out = pd.DataFrame(
        {c: (df[c] if c in df.columns else parsed.extras[c]) for c in parsed.columns}
    )
    out["prediction"] = trained.model.predict(X)
    metrics = trained.model.evaluate(X, df[config.TARGET]) if labeled else None

    return PredictionResult(
        labeled=labeled,
        n_rows=len(df),
        metrics=metrics,
        csv=out.to_csv(index=False),
        expected_metrics=trained.metrics,
        warnings=parsed.warnings,
    )
