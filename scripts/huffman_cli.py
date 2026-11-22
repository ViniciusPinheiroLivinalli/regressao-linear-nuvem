"""Compacta/descompacta arquivos no formato Huffman usado pela API.

    python scripts/huffman_cli.py compactar data/treino.csv data/treino.huff
    python scripts/huffman_cli.py descompactar previsoes.huff previsoes.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import huffman  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("compactar", "descompactar"):
        sys.exit(__doc__)
    acao, origem, destino = sys.argv[1:]
    dados = Path(origem).read_bytes()
    saida = huffman.compactar(dados) if acao == "compactar" else huffman.descompactar(dados)
    Path(destino).write_bytes(saida)
    print(f"{origem} ({len(dados)} bytes) -> {destino} ({len(saida)} bytes)")


if __name__ == "__main__":
    main()
