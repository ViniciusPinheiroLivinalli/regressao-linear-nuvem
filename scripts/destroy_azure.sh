#!/usr/bin/env bash
# Apaga TODOS os recursos do deploy (o resource group inteiro). Irreversivel.
# Use depois da apresentacao para parar a cobranca do App Service e do ACR.
set -euo pipefail
RG="${RG:-rg-regressao-linear}"
read -r -p "Apagar o resource group '$RG' e tudo dentro dele? [s/N] " ok
[[ "$ok" == "s" || "$ok" == "S" ]] || { echo "Cancelado."; exit 0; }
az group delete -n "$RG" --yes --no-wait
echo "Exclusao iniciada (leva alguns minutos)."
