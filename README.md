# Regressão Linear na Nuvem — núcleo local (Issues 1 a 6)

API FastAPI que treina uma regressão linear (MinMax + `LinearRegression`) sobre uma
série temporal com colunas `time-5,time-4,time-3,time-2,time-1,time` (alvo: `time`),
estima o desempenho, prevê sobre CSVs de teste (com ou sem rótulo) e permite resetar o modelo.

> Nesta fase os arquivos trafegam em claro. Compactação + criptografia ponta-a-ponta
> entram nas Issues 7 e 8, envolvendo a entrada (`raw`) e a saída (`predictions_csv`).

## Rodar localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Docs interativas: http://localhost:8000/docs

## Docker

```bash
docker build -t regressao-linear .
docker run --rm -p 8000:8000 regressao-linear
```

## Dados de exemplo e testes

```bash
python scripts/gerar_amostra.py   # cria data/treino.csv, teste_com_rotulo.csv, teste_sem_rotulo.csv
pytest
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Verificação de vida |
| GET | `/status` | `{trained, trained_at, n_rows, metrics}` |
| POST | `/train` | Multipart `file` (CSV). Devolve RMSE/MAE/R² estimados (média e desvio) |
| POST | `/predict` | Multipart `file` (CSV). Devolve `labeled`, `metrics` (se rotulado) e `predictions_csv` |
| POST | `/reset` | Apaga o modelo; volta a "sem modelo" |

Erros: 409 (sem modelo / treino em andamento), 413 (arquivo > 1 MB), 422 (CSV inválido).

```bash
curl -F file=@data/treino.csv http://localhost:8000/train
curl -F file=@data/teste_com_rotulo.csv http://localhost:8000/predict
curl -X POST http://localhost:8000/reset
```

## Decisões

- **Desempenho esperado:** `TimeSeriesSplit(5)` (janela expansiva, sem embaralhar); modelo final reajustado com todo o treino.
- **Min-max:** `fit` só no treino; no teste só `transform`. Dentro da validação, o scaler é reajustado a cada fold.
- **Estado global único:** modelo imutável trocado atomicamente; treino simultâneo ou reset durante treino → 409. Rodar com **um único worker/instância**.
- **Sem persistência:** o diretório `model_store/` é limpo a cada start.
- **Regressão:** `LinearRegression` do scikit-learn por enquanto (decisão em aberto); a troca é isolada em `app/ml/model.py`.
- **Configuração por ambiente:** `MAX_UPLOAD_BYTES`, `MIN_TRAIN_ROWS`, `CV_SPLITS`, `MODEL_STORE_DIR`.
