import pytest

from app import config
from app.errors import CsvValidationError, PayloadTooLargeError
from app.ml.validation import parse_csv

HEADER = ",".join(config.TRAIN_COLUMNS)
HEADER_NO_LABEL = ",".join(config.FEATURES)


def csv(header, *rows):
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def test_train_valid():
    parsed = parse_csv(csv(HEADER, "1,2,3,4,5,6"), "train")
    assert parsed.labeled is True
    assert parsed.data.shape == (1, 6)
    assert parsed.warnings == []


def test_test_labeled_and_unlabeled_detected():
    assert parse_csv(csv(HEADER, "1,2,3,4,5,6"), "test").labeled is True
    assert parse_csv(csv(HEADER_NO_LABEL, "1,2,3,4,5"), "test").labeled is False


def test_train_rejects_unlabeled_file():
    with pytest.raises(CsvValidationError, match="exige a coluna 'time'"):
        parse_csv(csv(HEADER_NO_LABEL, "1,2,3,4,5"), "train")


def test_wrong_header_and_missing_column():
    with pytest.raises(CsvValidationError, match="faltando: time-3"):
        parse_csv(csv("time-5,time-4,time-2,time-1,time", "1,2,3,4,5"), "train")


# --- Tolerancias ---
def test_columns_in_any_order_are_reordered():
    parsed = parse_csv(csv("time,time-1,time-2,time-3,time-4,time-5", "6,5,4,3,2,1"), "train")
    assert list(parsed.data.columns) == config.TRAIN_COLUMNS
    assert parsed.data.iloc[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert any("ordem" in w for w in parsed.warnings)


def test_extra_columns_ignored_with_warning():
    parsed = parse_csv(csv("id," + HEADER + ",obs", "a1,1,2,3,4,5,6,x"), "test")
    assert list(parsed.data.columns) == config.TRAIN_COLUMNS
    assert parsed.extras.iloc[0].tolist() == ["a1", "x"]
    assert parsed.columns[0] == "id" and parsed.columns[-1] == "obs"
    assert any("id, obs" in w for w in parsed.warnings)


def test_header_spaces_and_case_normalized():
    parsed = parse_csv(csv(" Time-5 , TIME-4,time-3,time-2,time-1, Time ", "1,2,3,4,5,6"), "train")
    assert parsed.labeled is True
    assert parsed.warnings == []


@pytest.mark.parametrize("sep, name", [(";", "ponto e virgula"), ("\t", "tab")])
def test_semicolon_and_tab_separators(sep, name):
    header = sep.join(config.TRAIN_COLUMNS)
    parsed = parse_csv(csv(header, sep.join("123456")), "train")
    assert parsed.data.iloc[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert any(name in w for w in parsed.warnings)


def test_decimal_comma_with_semicolon():
    header = ";".join(config.TRAIN_COLUMNS)
    parsed = parse_csv(csv(header, "1,5;2;3;4;5;6,25"), "train")
    assert parsed.data["time-5"].iloc[0] == 1.5
    assert parsed.data["time"].iloc[0] == 6.25
    assert any("decimal" in w for w in parsed.warnings)


# --- Continua rigido ---
def test_repeated_column_rejected():
    with pytest.raises(CsvValidationError, match="repetida no cabecalho: time-1"):
        parse_csv(csv(HEADER + ",time-1", "1,2,3,4,5,6,7"), "train")


def test_decimal_comma_with_comma_separator_is_not_guessed():
    # Com virgula como separador, "1,5" vira duas celulas: a linha fica longa demais.
    with pytest.raises(CsvValidationError):
        parse_csv(csv(HEADER, "1,5,2,3,4,5,6"), "train")


def test_non_numeric_reports_column_and_line():
    raw = csv(HEADER, "1,2,3,4,5,6", "1,2,abc,4,5,6")
    with pytest.raises(CsvValidationError, match=r"'time-3' \(linha 3\)"):
        parse_csv(raw, "train")


def test_null_value_reports_column_and_line():
    raw = csv(HEADER, "1,2,3,4,5,6", "1,2,,4,5,6")
    with pytest.raises(CsvValidationError, match=r"ausente.*'time-3' \(linha 3\)"):
        parse_csv(raw, "train")


def test_short_row_is_treated_as_missing():
    raw = csv(HEADER, "1,2,3,4,5,6", "1,2,3")
    with pytest.raises(CsvValidationError, match="ausente"):
        parse_csv(raw, "train")


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_nan_and_inf_rejected(bad):
    with pytest.raises(CsvValidationError):
        parse_csv(csv(HEADER, f"1,2,{bad},4,5,6"), "train")


def test_empty_file_and_header_only():
    with pytest.raises(CsvValidationError, match="vazio"):
        parse_csv(b"", "train")
    with pytest.raises(CsvValidationError, match="nao tem linhas"):
        parse_csv(csv(HEADER), "train")


def test_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 50)
    with pytest.raises(PayloadTooLargeError):
        parse_csv(csv(HEADER, *["1,2,3,4,5,6"] * 20), "train")


def test_not_utf8():
    with pytest.raises(CsvValidationError, match="UTF-8"):
        parse_csv(b"\xff\xfe\x00bad", "train")


def test_unnamed_index_columns_get_positional_names():
    parsed = parse_csv(csv(",," + HEADER, "0,a,1,2,3,4,5,6"), "train")
    assert parsed.columns[:2] == ["sem_nome_1", "sem_nome_2"]
    assert any("sem_nome_1, sem_nome_2" in w for w in parsed.warnings)
