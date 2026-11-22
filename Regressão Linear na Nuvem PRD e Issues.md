# Regressão Linear na Nuvem: PRD e Issues

Plano de implementação para deploy na Azure com Docker. Cada issue pode ser copiada para o GitHub Issues.

## 1. Visão geral

**Objetivo:** serviço web que treina uma regressão linear a partir de um CSV de série temporal, estima o desempenho esperado, prevê sobre CSVs de teste (com ou sem rótulo) e permite resetar o modelo. Os arquivos trafegam compactados e criptografados ponta-a-ponta.

**Fora de escopo:** autenticação de usuários, múltiplos modelos simultâneos, persistência do modelo após reinício do servidor.

## 2. Decisões já tomadas

| Tema | Decisão |
| --- | --- |
| Colunas do CSV | `time-5,time-4,time-3,time-2,time-1,time` (alvo: `time`) |
| Formato do teste | Lags já montados; sem rótulo = só as 5 colunas `time-5…time-1` |
| Escopo do modelo | Global único (um modelo para todo o sistema) |
| Valores nulos | Rejeitar o arquivo com erro claro |
| Segurança | Ponta-a-ponta: compactação e cifragem no navegador, decifragem no servidor, e o caminho inverso no download |
| Chaves | RSA do servidor fixa (variável de ambiente); chave AES nova por arquivo |
| Persistência | O modelo não precisa sobreviver a reinício |
| Tamanho dos dados | Cerca de 9 KB; limite de upload de 1 MB |
| Front-end | Uma página HTML simples com JS puro, servida pelo próprio backend |
| Deploy | Docker: Azure Container Registry + App Service for Containers (Linux) |
| Stack | Python + FastAPI + scikit-learn |

**Decisão em aberto:** usar a `LinearRegression` do scikit-learn ou implementar a regressão no grupo (equação normal ou gradiente descendente). O modelo fica atrás de uma interface (`fit`, `predict`), então a troca não afeta a API.

## 3. Decisões técnicas

### 3.1 Estimativa de desempenho

- Série temporal não pode ser embaralhada, senão o futuro vaza para o treino.
- Usar `TimeSeriesSplit(n_splits=5)` (janela expansiva) e reportar média e desvio de **RMSE, MAE e R²**.
- Depois da validação, o modelo final é reajustado com todos os dados de treino.

### 3.2 Normalização

- `MinMaxScaler` e regressão ficam em um único `Pipeline`.
- O `fit` do scaler acontece só nos dados de treino; no teste usa-se apenas `transform`.
- Dentro da validação cruzada, o scaler é reajustado a cada fold.

### 3.3 Protocolo de criptografia

**Upload (cliente → servidor):**

1. O navegador busca a chave pública RSA em `GET /public-key`.
2. Gera duas chaves AES-256 novas: `k_up` (para o arquivo enviado) e `k_down` (para a resposta).
3. Compacta o CSV (gzip) e cifra com AES-256-GCM usando `k_up` e um IV aleatório de 12 bytes.
4. Envia um envelope JSON em base64: `wrapped_k_up`, `wrapped_k_down` (ambas cifradas com RSA-OAEP-SHA256), `iv` e `ciphertext`.

**Download (servidor → cliente):**

1. O servidor decifra as chaves com a chave privada e processa o arquivo.
2. Compacta o CSV de previsões (gzip) e cifra com `k_down` e um IV novo.
3. O navegador decifra, descompacta e oferece o `.csv` para download.

**Observações:**

- Nenhum arquivo em claro é gravado em disco no servidor.
- As métricas voltam como JSON simples sobre HTTPS, porque não são arquivos.
- HTTPS é obrigatório na Azure; a cifragem do arquivo é adicional.

### 3.4 Estado global

- Um objeto em memória (com cópia em `model_store/` dentro do container) guarda o pipeline, as métricas e a data do treino.
- Um lock impede treino e predição simultâneos; treino durante treino retorna 409.
- Como o estado é global e não persiste, o App Service deve rodar com **uma única instância** (sem scale-out).
- Ao subir o serviço, o estado inicial é "sem modelo treinado".

## 4. API

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | Verificação de vida |
| GET | `/public-key` | Chave pública RSA em PEM |
| GET | `/status` | `{trained: bool, trained_at, metrics}` |
| POST | `/train` | Recebe envelope cifrado, treina, devolve métricas estimadas |
| POST | `/predict` | Recebe envelope cifrado, devolve previsões cifradas e métricas reais (se houver rótulo) |
| POST | `/reset` | Apaga o modelo e volta para "sem modelo" |

Códigos de erro: 400 envelope inválido, 409 sem modelo (ou treino em andamento), 413 arquivo grande demais, 422 CSV inválido.

## 5. Issues

### Issue 1: Setup do projeto e Dockerfile

**Labels:** infra, setup · **Depende de:** nenhuma

**Tarefas**

- Criar a estrutura `app/` (`api`, `ml`, `crypto`, `web`), `tests/`, `requirements.txt` e `.gitignore`.
- Criar `Dockerfile` (python slim, usuário não-root, `uvicorn` na porta configurável).
- Implementar `GET /health`.
- Servir a pasta `web/` como arquivos estáticos.

**Critérios de aceite**

- `docker build` e `docker run` sobem o serviço localmente.
- `/health` responde 200.
- README com o comando para rodar.

### Issue 2: Validação do CSV

**Labels:** backend, dados · **Depende de:** 1

**Tarefas**

- Função `validate_csv(bytes, mode)` com `mode` em `train`, `test_labeled` e `test_unlabeled`.
- Regras: cabeçalho exato, todas as colunas numéricas, sem nulos, ao menos 1 linha (treino: mínimo configurável, sugestão 30), tamanho máximo de 1 MB.
- Detectar automaticamente se o teste tem rótulo (presença da coluna `time`).
- Erros em 422 com mensagem indicando coluna e linha.

**Critérios de aceite**

- Testes unitários cobrem: cabeçalho errado, coluna faltando, texto em campo numérico, nulo, arquivo vazio, arquivo grande.
- Teste com e sem rótulo é identificado corretamente.

### Issue 3: Pipeline de ML

**Labels:** ml · **Depende de:** 1

**Tarefas**

- Classe `LinearModel` com `fit(X, y)`, `predict(X)` e `evaluate(X, y)`.
- `Pipeline` com `MinMaxScaler` + regressão linear.
- Cálculo de RMSE, MAE e R².
- *Decisão em aberto:* sklearn ou implementação própria; manter a interface independente disso.

**Critérios de aceite**

- Teste unitário confirma que o scaler foi ajustado apenas no treino (os mínimos e máximos do teste não o alteram).
- Previsão num conjunto sintético linear tem R² próximo de 1.

### Issue 4: Treino com estimativa de desempenho

**Labels:** backend, ml · **Depende de:** 2, 3

**Tarefas**

- `POST /train`: valida o CSV, roda `TimeSeriesSplit(5)`, calcula média e desvio das métricas, reajusta com todos os dados e guarda o modelo no estado global.
- `GET /status`.
- Lock contra treinos simultâneos.
- Retornar `{rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, n_rows}`.

**Critérios de aceite**

- Treinar com a base do trabalho devolve métricas coerentes.
- Novo treino substitui o modelo anterior.
- Treino com CSV inválido não altera o modelo existente.

### Issue 5: Predição (com e sem rótulo)

**Labels:** backend, ml · **Depende de:** 4

**Tarefas**

- `POST /predict`: exige modelo treinado (senão 409).
- Com rótulo: gerar CSV com colunas originais + `prediction` e calcular RMSE, MAE e R² reais.
- Sem rótulo: gerar CSV com colunas originais + `prediction`, sem métricas.
- Preservar a ordem das linhas.

**Critérios de aceite**

- Os dois fluxos funcionam e retornam o CSV esperado.
- Predict sem modelo retorna 409.
- Teste confere que as métricas reais batem com um cálculo independente.

### Issue 6: Reset do modelo

**Labels:** backend · **Depende de:** 4

**Tarefas**

- `POST /reset`: remove o modelo, as métricas e os arquivos em `model_store/`.
- `GET /status` passa a retornar `trained: false`.
- Impedir reset durante treino em andamento (409).

**Critérios de aceite**

- Treinar, resetar, predizer retorna 409.
- Treinar com uma base diferente após o reset gera um modelo novo, sem resíduos do anterior.

### Issue 7: Criptografia no servidor

**Labels:** segurança, backend · **Depende de:** 1

**Tarefas**

- Carregar a chave privada RSA de variável de ambiente (`RSA_PRIVATE_KEY_PEM`, em base64) e expor `GET /public-key`.
- Função para abrir o envelope: RSA-OAEP-SHA256 → AES-256-GCM → gunzip.
- Função para fechar a resposta: gzip → AES-256-GCM com `k_down` e IV novo.
- Rejeitar envelope malformado ou com tag GCM inválida (400).
- Script para gerar o par de chaves.

**Critérios de aceite**

- Testes: ida e volta, envelope adulterado rejeitado, IV nunca reaproveitado.
- Nenhum arquivo em claro tocado em disco (verificar no código e nos testes).

### Issue 8: Criptografia no navegador

**Labels:** segurança, frontend · **Depende de:** 7

**Tarefas**

- Módulo JS com Web Crypto API: gerar `k_up` e `k_down`, `CompressionStream('gzip')`, AES-GCM, RSA-OAEP.
- Funções `encryptFile(file)` e `decryptResponse(envelope, k_down)`.
- Tratar falta de suporte do navegador com mensagem clara.

**Critérios de aceite**

- Um arquivo cifrado pelo navegador é aceito pelo servidor.
- A resposta do servidor é decifrada e baixada como `.csv` idêntico ao esperado.

### Issue 9: Página HTML

**Labels:** frontend · **Depende de:** 5, 6, 8

**Tarefas**

- Seções: estado do modelo, upload de treino, métricas estimadas, upload de teste, download das previsões, métricas reais e botão de reset.
- Mensagens de erro vindas da API exibidas de forma legível.
- Indicador de carregamento e botões desabilitados durante operações.

**Critérios de aceite**

- Fluxo completo pelo navegador: treinar, testar com rótulo, testar sem rótulo, baixar, resetar.
- Sem chamadas manuais pelo Postman.

### Issue 10: Deploy na Azure

**Labels:** infra, deploy · **Depende de:** 1 (e das demais para o teste final)

**Tarefas**

- Criar Resource Group, Azure Container Registry e App Service Plan Linux (tier B1 ou superior; o gratuito F1 não suporta containers).
- Criar o Web App for Containers apontando para a imagem do ACR; definir `WEBSITES_PORT`.
- Configurar `RSA_PRIVATE_KEY_PEM` como application setting.
- Fixar o número de instâncias em 1.
- Pipeline GitHub Actions: build, push para o ACR e deploy a cada merge na `main`.

**Critérios de aceite**

- A URL pública (HTTPS) executa o fluxo completo.
- Reiniciar o app volta ao estado "sem modelo" sem erro.
- Segredos não aparecem no repositório.

### Issue 11: Testes e documentação

**Labels:** qualidade, docs · **Depende de:** todas

**Tarefas**

- Teste de integração ponta-a-ponta (cifra → treino → predição → decifra → reset).
- Rodar os testes no CI antes do deploy.
- README: arquitetura, protocolo de criptografia, como rodar local e na Azure, decisões e limitações.
- Registrar a decisão sobre a implementação da regressão (sklearn ou própria).

**Critérios de aceite**

- Suíte verde no CI.
- Cada requisito do enunciado tem uma seção ou teste correspondente.

## 6. Ordem e paralelismo

**Núcleo local:** 1 → 2 → 3 → 4 → 5 → 6 **Em paralelo ao núcleo:** 7 (servidor) e 10 (infra) podem começar logo após a 1. **Depois:** 8 → 9 → deploy final → 11

## 7. Checklist de requisitos do enunciado

| Requisito | Issue |
| --- | --- |
| Upload de CSV para treino | 4, 9 |
| Expectativa de desempenho após o treino | 4 |
| Upload de CSV de teste (com e sem rótulo) | 5, 9 |
| Download das previsões e desempenho real | 5, 9 |
| Tratamento de dados (min-max) | 3 |
| Compactação e criptografia ponta-a-ponta | 7, 8 |
| Reset do modelo | 6 |
| Execução remota (Azure) | 10 |
