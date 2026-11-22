import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from app import huffman

JS = Path(__file__).resolve().parents[1] / "app" / "web" / "huffman.js"

CASES = {
    "vazio": b"",
    "um_simbolo": b"a",
    "um_simbolo_repetido": b"zzzzzzzzzz",
    "dois_simbolos": b"abababbbba",
    "csv": b"time-5,time-4,time-3,time-2,time-1,time\n1.5,2.25,3,4,5,6\n" * 50,
    "todos_os_bytes": bytes(range(256)) * 3,
    "aleatorio": random.Random(42).randbytes(5000),
    "utf8": "previsão,ção,ñ\n".encode("utf-8") * 20,
}


@pytest.mark.parametrize("name", CASES)
def test_round_trip(name):
    data = CASES[name]
    assert huffman.descompactar(huffman.compactar(data)) == data


def test_csv_gets_smaller():
    data = CASES["csv"]
    assert len(huffman.compactar(data)) < len(data)


def test_header_format():
    packed = huffman.compactar(b"aab")
    assert huffman.tamanho_declarado(packed) == 3
    assert len(packed) >= huffman.HEADER_SIZE


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00" * 10,  # menor que o cabecalho
        b"\x00\x00\x00\x05" + bytes(256),  # tamanho > 0 sem tabela
        b"\x00\x00\x00\x05" + bytes([1, 1, 1]) + bytes(253) + b"\xff",  # prefixos conflitantes
    ],
)
def test_invalid_payloads(payload):
    with pytest.raises(ValueError):
        huffman.descompactar(payload)


def test_truncated_payload():
    packed = huffman.compactar(CASES["csv"])
    with pytest.raises(ValueError):
        huffman.descompactar(packed[:-5])


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js nao instalado")
def test_js_and_python_agree_byte_for_byte():
    """O navegador (huffman.js) e o servidor (huffman.py) produzem o mesmo payload
    e cada um descompacta o que o outro gerou."""
    script = f"""
const H = require({json.dumps(str(JS))});
const casos = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = {{}};
for (const [nome, b64] of Object.entries(casos)) {{
  const dados = new Uint8Array(Buffer.from(b64, "base64"));
  const packed = H.compactar(dados);
  const volta = H.descompactar(packed);
  out[nome] = {{
    packed: Buffer.from(packed).toString("base64"),
    ok: Buffer.from(volta).equals(Buffer.from(dados)),
  }};
}}
process.stdout.write(JSON.stringify(out));
"""
    import base64

    payload = {k: base64.b64encode(v).decode() for k, v in CASES.items()}
    proc = subprocess.run(
        ["node", "-e", script], input=json.dumps(payload), capture_output=True, text=True, check=True
    )
    result = json.loads(proc.stdout)
    for name, data in CASES.items():
        js_packed = base64.b64decode(result[name]["packed"])
        assert result[name]["ok"], name
        assert js_packed == huffman.compactar(data), name
        assert huffman.descompactar(js_packed) == data, name


@pytest.mark.parametrize("name", CASES)
def test_valid_header_accepted(name):
    data = CASES[name]
    assert huffman.validar_cabecalho(huffman.compactar(data)) == len(data)


def test_plain_csv_header_rejected():
    with pytest.raises(ValueError):
        huffman.validar_cabecalho(CASES["csv"])
