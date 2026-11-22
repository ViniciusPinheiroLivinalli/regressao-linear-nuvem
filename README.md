# Regressão Linear na Nuvem — núcleo local (Issues 1 a 6)

API FastAPI que treina uma regressão linear (MinMax + `LinearRegression`) sobre uma
série temporal com colunas `time-5,time-4,time-3,time-2,time-1,time` (alvo: `time`),
estima o desempenho, prevê sobre CSVs de teste (com ou sem rótulo) e permite resetar o modelo.

**Tratamento ponta-a-ponta:** todo arquivo trafega compactado com **Huffman Canônico**,
nos dois sentidos. O navegador compacta o CSV (`app/web/huffman.js`) antes do envio; o
servidor descompacta só em memória (`app/huffman.py`), processa e devolve o CSV de previsões
compactado, que o navegador descompacta para o download. O enunciado pede "compactar ou
criptografar"; optamos só pela compactação.

Interface web em http://localhost:8000 (treino, teste, comparação esperado × real, download e reset).

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

## Deploy na Azure

Produção: **https://regressao-linear-d9527f.azurewebsites.net** (App Service Linux B1, região
`brazilsouth`, 1 instância, só HTTPS, health check em `/health`).

```bash
pip install azure-cli          # ou o instalador oficial
az login --use-device-code
BUILD=local ./scripts/deploy_azure.sh   # publica uma nova versão
./scripts/destroy_azure.sh              # apaga tudo (para a cobrança)
```

- A **Azure for Students** bloqueia o build na nuvem (ACR Tasks), por isso `BUILD=local`
  constrói a imagem com o Docker Desktop e envia para o ACR. No WSL sem integração com o
  Docker Desktop, passe `DOCKER=/mnt/c/.../docker.exe`.
- A assinatura só permite as regiões `brazilsouth`, `canadacentral`, `chilecentral`,
  `mexicocentral` e `francecentral`.
- Custo aproximado: B1 + ACR Basic ≈ US$ 20–25/mês, debitado do crédito de estudante.
- O modelo vive em memória: reiniciar ou publicar uma nova versão volta ao estado "sem modelo".

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
| POST | `/train` | Multipart `file` (CSV compactado com Huffman). Devolve RMSE/MAE/R² estimados (média e desvio) |
| POST | `/predict` | Multipart `file` (CSV compactado). Devolve `labeled`, `metrics` reais (se rotulado), `expected_metrics` e `predictions_huff` (CSV compactado, em base64) |
| POST | `/reset` | Apaga o modelo; volta a "sem modelo" |

Erros: 400 (arquivo não compactado ou corrompido), 409 (sem modelo / treino em andamento),
413 (arquivo > 1 MB, compactado ou descompactado), 422 (CSV inválido).

```bash
python scripts/huffman_cli.py compactar data/treino.csv data/treino.huff
python scripts/huffman_cli.py compactar data/teste_com_rotulo.csv data/teste.huff
curl -F file=@data/treino.huff http://localhost:8000/train
curl -s -F file=@data/teste.huff http://localhost:8000/predict \
  | python -c "import sys,json,base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['predictions_huff']))" > previsoes.huff
python scripts/huffman_cli.py descompactar previsoes.huff previsoes.csv
curl -X POST http://localhost:8000/reset
```

## Formato Huffman

Idêntico, byte a byte, no Python e no JavaScript (há teste que compara os dois):

| Bytes | Conteúdo |
|---|---|
| 0–3 | Tamanho original, uint32 big-endian |
| 4–259 | Comprimento em bits do código de cada byte 0..255 (0 = ausente) |
| 260… | Códigos canônicos concatenados (MSB primeiro), último byte completado com zeros |

Os códigos são reconstruídos pela regra canônica (ordem por comprimento e depois pelo valor
do byte), então só os comprimentos precisam ir no cabeçalho. Antes de descompactar, o servidor
confere se a tabela forma um código de prefixo completo e se o tamanho declarado cabe no limite.

## Decisões

- **Desempenho esperado:** `TimeSeriesSplit(5)` (janela expansiva, sem embaralhar); modelo final reajustado com todo o treino.
- **Min-max:** `fit` só no treino; no teste só `transform`. Dentro da validação, o scaler é reajustado a cada fold.
- **Estado global único:** modelo imutável trocado atomicamente; treino simultâneo ou reset durante treino → 409. Rodar com **um único worker/instância**.
- **Sem persistência:** o diretório `model_store/` é limpo a cada start.
- **Regressão:** `LinearRegression` do scikit-learn por enquanto (decisão em aberto); a troca é isolada em `app/ml/model.py`.
- **Configuração por ambiente:** `MAX_UPLOAD_BYTES`, `MIN_TRAIN_ROWS`, `CV_SPLITS`, `MODEL_STORE_DIR`.


https://excalidraw.com/#json=MPl7fA3qPLxtOvtaFTFKZ,yhAzFy-F5FQ58w4f1QVC9g