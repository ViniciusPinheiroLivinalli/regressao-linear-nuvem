import pytest

from app import config
from app.errors import CsvValidationError, PayloadTooLargeError
from app.ml.validation import parse_csv

HEADER = ",".join(config.TRAIN_COLUMNS)
HEADER_NO_LABEL = ",".join(config.FEATURES)


def csv(header, *rows):
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def test_train_valid():
    df, labeled = parse_csv(csv(HEADER, "1,2,3,4,5,6"), "train")
    assert labeled is True
    assert df.shape == (1, 6)


def test_test_labeled_and_unlabeled_detected():
    _, labeled = parse_csv(csv(HEADER, "1,2,3,4,5,6"), "test")
    assert labeled is True
    _, labeled = parse_csv(csv(HEADER_NO_LABEL, "1,2,3,4,5"), "test")
    assert labeled is False


def test_train_rejects_unlabeled_file():
    with pytest.raises(CsvValidationError, match="exige a coluna 'time'"):
        parse_csv(csv(HEADER_NO_LABEL, "1,2,3,4,5"), "train")


def test_wrong_header_and_missing_column():
    with pytest.raises(CsvValidationError, match="faltando: time-3"):
        parse_csv(csv("time-5,time-4,time-2,time-1,time", "1,2,3,4,5"), "train")


def test_wrong_order_and_extra_column():
    with pytest.raises(CsvValidationError, match="ordem"):
        parse_csv(csv("time-4,time-5,time-3,time-2,time-1,time", "1,2,3,4,5,6"), "train")
    with pytest.raises(CsvValidationError, match="inesperadas: foo"):
        parse_csv(csv(HEADER + ",foo", "1,2,3,4,5,6,7"), "train")


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
