"""Issue 2: validacao e leitura do CSV conforme o contrato de dados."""
import io

import numpy as np
import pandas as pd

from app import config
from app.errors import CsvValidationError, PayloadTooLargeError


def _header_message(columns: list[str], mode: str) -> str:
    expected = ",".join(config.TRAIN_COLUMNS)
    if mode == "test":
        expected += "  (ou, sem rotulo: " + ",".join(config.FEATURES) + ")"
    missing = [c for c in config.TRAIN_COLUMNS if c not in columns]
    extra = [c for c in columns if c not in config.TRAIN_COLUMNS]
    detail = []
    if mode == "train" and columns == config.FEATURES:
        detail.append("o treino exige a coluna 'time'")
    elif missing:
        detail.append("faltando: " + ", ".join(missing))
    if extra:
        detail.append("inesperadas: " + ", ".join(extra))
    if not detail:
        detail.append("verifique a ordem das colunas")
    return f"Cabecalho invalido ({'; '.join(detail)}). Esperado: {expected}"


def parse_csv(raw: bytes, mode: str) -> tuple[pd.DataFrame, bool]:
    """Valida e converte o CSV.

    mode: "train" (exige as 6 colunas) ou "test" (6 colunas = com rotulo,
    5 colunas = sem rotulo).

    Retorna (DataFrame numerico, tem_rotulo).
    """
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Arquivo maior que o limite de {config.MAX_UPLOAD_BYTES} bytes."
        )
    if not raw.strip():
        raise CsvValidationError("Arquivo vazio.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CsvValidationError("O arquivo deve estar codificado em UTF-8.")

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CsvValidationError(f"CSV malformado: {exc}")

    df = df.fillna("")  # linhas curtas viram valores vazios, detectados abaixo
    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)

    if columns == config.TRAIN_COLUMNS:
        labeled = True
    elif mode == "test" and columns == config.FEATURES:
        labeled = False
    else:
        raise CsvValidationError(_header_message(columns, mode))

    if df.empty:
        raise CsvValidationError("O arquivo nao tem linhas de dados.")

    numeric = pd.DataFrame(index=df.index)
    for col in columns:
        stripped = df[col].str.strip()
        blank = stripped == ""
        if blank.any():
            line = int(blank.idxmax()) + 2  # +1 cabecalho, +1 indice base 0
            raise CsvValidationError(
                f"Valor ausente na coluna '{col}' (linha {line})."
            )
        values = pd.to_numeric(stripped, errors="coerce").astype(float)
        invalid = ~np.isfinite(values)
        if invalid.any():
            pos = int(invalid.to_numpy().argmax())
            line = pos + 2
            raise CsvValidationError(
                f"Valor nao numerico na coluna '{col}' (linha {line}): "
                f"'{stripped.iloc[pos]}'."
            )
        numeric[col] = values

    return numeric, labeled
