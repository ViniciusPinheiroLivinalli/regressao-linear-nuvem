#!/usr/bin/env bash
# Deploy na Azure: imagem construida no proprio ACR (nao precisa de Docker local)
# e publicada num Web App for Containers (App Service Linux) com UMA instancia.
#
# Uso:  az login  &&  ./scripts/deploy_azure.sh
# Azure for Students (ACR Tasks bloqueado): BUILD=local ./scripts/deploy_azure.sh
#   (no WSL com Docker Desktop sem integracao: DOCKER=/mnt/c/.../docker.exe)
# Rodar de novo publica uma nova versao (os recursos ja existentes sao reaproveitados).
#
# Variaveis opcionais: LOCATION, RG, PLAN_SKU, ACR_NAME, APP_NAME
set -euo pipefail

cd "$(dirname "$0")/.."

LOCATION="${LOCATION:-brazilsouth}"
RG="${RG:-rg-regressao-linear}"
PLAN="${PLAN:-plan-regressao-linear}"
PLAN_SKU="${PLAN_SKU:-B1}"          # F1 (gratuito) nao suporta containers
IMAGE_REPO="regressao-linear"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)-$(date +%Y%m%d%H%M%S)"

# Nomes globais precisam ser unicos: sufixo derivado da assinatura (estavel entre execucoes).
SUFFIX="$(az account show --query id -o tsv | tr -d '-' | cut -c1-6)"
ACR_NAME="${ACR_NAME:-regressaolinear${SUFFIX}}"
APP_NAME="${APP_NAME:-regressao-linear-${SUFFIX}}"

echo "Assinatura : $(az account show --query name -o tsv)"
echo "Regiao     : $LOCATION"
echo "Grupo      : $RG"
echo "Registry   : $ACR_NAME"
echo "Web App    : $APP_NAME ($PLAN_SKU)"
echo

echo "==> Registrando provedores (uma vez por assinatura)"
az provider register --namespace Microsoft.ContainerRegistry --wait -o none
az provider register --namespace Microsoft.Web --wait -o none

echo "==> Resource group"
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Azure Container Registry"
if ! az acr show -n "$ACR_NAME" -g "$RG" -o none 2>/dev/null; then
  az acr create -n "$ACR_NAME" -g "$RG" --sku Basic --admin-enabled true -o none
fi
ACR_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
IMAGE="$ACR_SERVER/$IMAGE_REPO:$TAG"
# O app roda sempre a tag "latest": e ela que a entrega continua (CD) observa.
IMAGE_LATEST="$ACR_SERVER/$IMAGE_REPO:latest"

if [[ "${BUILD:-acr}" == "local" ]]; then
  # Build com o Docker local (Docker Desktop). Necessario em assinaturas que
  # bloqueiam ACR Tasks, como a Azure for Students.
  DOCKER="${DOCKER:-docker}"
  CONTEXT="."
  if [[ "$DOCKER" == *.exe ]]; then CONTEXT="$(wslpath -w .)"; fi  # docker.exe do Windows via WSL
  echo "==> Build local da imagem ($IMAGE)"
  "$DOCKER" build --platform linux/amd64 --build-arg "APP_VERSION=$TAG" -t "$IMAGE" -t "$IMAGE_LATEST" -f "$CONTEXT/Dockerfile" "$CONTEXT"
  az acr credential show -n "$ACR_NAME" --query 'passwords[0].value' -o tsv \
    | "$DOCKER" login "$ACR_SERVER" -u "$(az acr credential show -n "$ACR_NAME" --query username -o tsv)" --password-stdin
  "$DOCKER" push "$IMAGE"
  "$DOCKER" push "$IMAGE_LATEST"
else
  echo "==> Build da imagem no ACR ($IMAGE)"
  az acr build -r "$ACR_NAME" -t "$IMAGE_REPO:$TAG" -t "$IMAGE_REPO:latest" -f Dockerfile .
fi

echo "==> App Service Plan (Linux, 1 instancia)"
if ! az appservice plan show -n "$PLAN" -g "$RG" -o none 2>/dev/null; then
  az appservice plan create -n "$PLAN" -g "$RG" -l "$LOCATION" --is-linux \
    --sku "$PLAN_SKU" --number-of-workers 1 -o none
fi

ACR_USER="$(az acr credential show -n "$ACR_NAME" --query username -o tsv)"
ACR_PASS="$(az acr credential show -n "$ACR_NAME" --query 'passwords[0].value' -o tsv)"

echo "==> Web App"
if ! az webapp show -n "$APP_NAME" -g "$RG" -o none 2>/dev/null; then
  az webapp create -n "$APP_NAME" -g "$RG" -p "$PLAN" \
    --container-image-name "$IMAGE_LATEST" \
    --container-registry-url "https://$ACR_SERVER" \
    --container-registry-user "$ACR_USER" \
    --container-registry-password "$ACR_PASS" -o none
fi
az webapp config container set -n "$APP_NAME" -g "$RG" \
  --container-image-name "$IMAGE_LATEST" \
  --container-registry-url "https://$ACR_SERVER" \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS" -o none

echo "==> Configuracao (porta, HTTPS, always-on, health check, 1 instancia)"
az webapp config appsettings set -n "$APP_NAME" -g "$RG" -o none --settings \
  WEBSITES_PORT=8000 PORT=8000 WEBSITES_ENABLE_APP_SERVICE_STORAGE=false
az webapp update -n "$APP_NAME" -g "$RG" --https-only true -o none
az webapp config set -n "$APP_NAME" -g "$RG" -o none \
  --always-on true --min-tls-version 1.2 \
  --generic-configurations '{"healthCheckPath": "/health"}'
az appservice plan update -n "$PLAN" -g "$RG" --number-of-workers 1 -o none

echo "==> Entrega continua: push de :latest no ACR reinicia o app com a imagem nova"
# O webhook do ACR chama o SCM (Kudu) com credencial basica; apps novos vem com ela desligada.
az resource update -g "$RG" --namespace Microsoft.Web --resource-type basicPublishingCredentialsPolicies \
  --parent "sites/$APP_NAME" -n scm --set properties.allow=true -o none
CD_URL="$(az webapp deployment container config -n "$APP_NAME" -g "$RG" --enable-cd true --query CI_CD_URL -o tsv)"
if az acr webhook show -r "$ACR_NAME" -n webappcd -o none 2>/dev/null; then
  az acr webhook update -r "$ACR_NAME" -n webappcd --uri "$CD_URL" --actions push \
    --scope "$IMAGE_REPO:latest" -o none
else
  az acr webhook create -r "$ACR_NAME" -n webappcd --uri "$CD_URL" --actions push \
    --scope "$IMAGE_REPO:latest" -o none
fi

az webapp restart -n "$APP_NAME" -g "$RG" -o none

URL="https://$(az webapp show -n "$APP_NAME" -g "$RG" --query defaultHostName -o tsv)"
echo "==> Aguardando o app responder em $URL/health"
for _ in $(seq 1 40); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then
    echo
    echo "Pronto: $URL"
    exit 0
  fi
  sleep 10
done
echo "O app ainda nao respondeu. Veja os logs com:"
echo "  az webapp log tail -n $APP_NAME -g $RG"
exit 1
