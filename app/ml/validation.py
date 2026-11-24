"""Leitura e validacao do CSV conforme o contrato de dados.

Tolerante ao que da para corrigir sem ambiguidade:
  - colunas em qualquer ordem e cabecalho com espacos/maiusculas ("Time-1 " -> "time-1");
  - colunas extras (ex.: um indice ou "id"): ignoradas no modelo, com aviso;
  - separador ";" ou tab, alem de ",";
  - virgula decimal ("12,5") quando o separador nao e virgula.

Rigido no que nao da para adivinhar: falta de alguma feature, colunas repetidas,
celulas vazias e valores nao numericos sao recusados com a coluna e a linha.
"""
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app import config
from app.errors import CsvValidationError, PayloadTooLargeError

SEPARATORS = {",": "virgula", ";": "ponto e virgula", "\t": "tab"}


@dataclass
class ParsedCsv:
    data: pd.DataFrame  # colunas conhecidas, numericas, na ordem canonica
    labeled: bool
    extras: pd.DataFrame  # colunas extras, como texto (devolvidas no CSV de previsoes)
    columns: list[str]  # ordem original das colunas (nomes normalizados)
    warnings: list[str] = field(default_factory=list)


def _expected(mode: str) -> str:
    text = ",".join(config.TRAIN_COLUMNS)
    if mode == "test":
        text += "  (ou, sem rotulo: " + ",".join(config.FEATURES) + ")"
    return text


def _detect_separator(text: str) -> str:
    header = text.split("\n", 1)[0]
    counts = {sep: header.count(sep) for sep in SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def parse_csv(raw: bytes, mode: str) -> ParsedCsv:
    """Valida e converte o CSV.

    mode: "train" (exige as 5 features e `time`) ou "test" (com `time` = rotulado,
    sem `time` = so previsao).
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

    sep = _detect_separator(text)
    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, header=None, dtype=str,
            keep_default_na=False, index_col=False, skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CsvValidationError(f"CSV malformado: {exc}")

    df = df.fillna("")  # linhas curtas viram valores vazios, detectados abaixo
    columns = [str(c).strip().lower() for c in df.iloc[0]]
    # Coluna sem nome (ex.: indice salvo pelo pandas/Excel) ganha um nome pela posicao.
    columns = [c or f"sem_nome_{i + 1}" for i, c in enumerate(columns)]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = columns

    repeated = sorted({c for c in columns if columns.count(c) > 1})
    if repeated:
        raise CsvValidationError(
            f"Coluna repetida no cabecalho: {', '.join(repeated)}. Esperado: {_expected(mode)}"
        )

    missing = [c for c in config.FEATURES if c not in columns]
    if missing:
        raise CsvValidationError(
            f"Cabecalho invalido (faltando: {', '.join(missing)}). Esperado: {_expected(mode)}"
        )
    labeled = config.TARGET in columns
    if mode == "train" and not labeled:
        raise CsvValidationError(
            f"Cabecalho invalido (o treino exige a coluna 'time'). Esperado: {_expected(mode)}"
        )

    if df.empty:
        raise CsvValidationError("O arquivo nao tem linhas de dados.")

    known = config.TRAIN_COLUMNS if labeled else config.FEATURES
    extra_cols = [c for c in columns if c not in known]

    warnings = []
    if sep != ",":
        warnings.append(f"Separador detectado: {SEPARATORS[sep]}.")
    if extra_cols:
        warnings.append(
            "Colunas ignoradas pelo modelo (mantidas no CSV de previsoes): "
            + ", ".join(extra_cols) + "."
        )
    if [c for c in columns if c in known] != known:
        warnings.append("Colunas fora da ordem padrao; reordenadas pelo nome.")

    decimal_comma = False
    numeric = pd.DataFrame(index=df.index)
    for col in known:
        stripped = df[col].str.strip()
        blank = stripped == ""
        if blank.any():
            line = int(blank.idxmax()) + 2  # +1 cabecalho, +1 indice base 0
            raise CsvValidationError(
                f"Valor ausente na coluna '{col}' (linha {line})."
            )
        if sep != "," and stripped.str.contains(",", regex=False).any():
            stripped = stripped.str.replace(",", ".", regex=False)
            decimal_comma = True
        values = pd.to_numeric(stripped, errors="coerce").astype(float)
        invalid = ~np.isfinite(values)
        if invalid.any():
            pos = int(invalid.to_numpy().argmax())
            line = pos + 2
            raise CsvValidationError(
                f"Valor nao numerico na coluna '{col}' (linha {line}): "
                f"'{df[col].iloc[pos].strip()}'."
            )
        numeric[col] = values

    if decimal_comma:
        warnings.append("Virgula decimal convertida para ponto.")

    return ParsedCsv(
        data=numeric,
        labeled=labeled,
        extras=df[extra_cols],
        columns=columns,
        warnings=warnings,
    )
