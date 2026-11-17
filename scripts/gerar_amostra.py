"""Gera CSVs sinteticos no formato do trabalho, para testar localmente.

Uso: python scripts/gerar_amostra.py
Cria em data/: treino.csv, teste_com_rotulo.csv, teste_sem_rotulo.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["time-5", "time-4", "time-3", "time-2", "time-1", "time"]


def make_series_frame(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows + 5
    s = np.zeros(n)
    s[:5] = rng.normal(10, 1, 5)
    for t in range(5, n):
        s[t] = 0.4 * s[t - 1] + 0.3 * s[t - 2] + 0.1 * s[t - 3] + 2 + rng.normal(0, 0.2)
    rows = [s[i : i + 6] for i in range(n_rows)]
    return pd.DataFrame(rows, columns=COLS)


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    make_series_frame(200, seed=1).to_csv(out / "treino.csv", index=False)
    teste = make_series_frame(40, seed=2)
    teste.to_csv(out / "teste_com_rotulo.csv", index=False)
    teste.drop(columns="time").to_csv(out / "teste_sem_rotulo.csv", index=False)
    print("Arquivos gerados em", out.resolve())
