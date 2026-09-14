#!/usr/bin/env bash
# Roda a aula inteira e grava RESULTADOS.md. Requer apenas: uv.
set -euo pipefail
cd "$(dirname "$0")"

SAIDA="RESULTADOS.md"

registra() {  # registra "<título>" "<comando>"
    printf '\n### %s\n\n```console\n$ %s\n' "$1" "$2" >>"$SAIDA"
    eval "$2" 2>&1 \
        | grep -viE 'uv_utils|agent\.hint|WARNING|warnings\.warn|FutureWarning|mlflow\.sklearn:|mlflow\.utils|^  warnings' \
        | sed 's/\x1b\[[0-9;]*m//g' >>"$SAIDA" || true
    printf '```\n' >>"$SAIDA"
}

rm -rf mlflow.db mlruns mlartifacts matriz.png

{
    echo "# Resultados reais — $(date -Is)"
    echo
    echo "Gerado por \`demo.sh\` em \`$(hostname)\`."
    echo "MLflow $(uv run --no-active python -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null | tail -1)," \
         "scikit-learn $(uv run --no-active python -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null | tail -1)."
    echo
    echo "As linhas de log do MLflow foram filtradas; o resto é literal."
} >"$SAIDA"

echo "## 1. Treinar cinco variações" >>"$SAIDA"
registra "just treinar" "uv run --no-active python treino.py"

echo "" >>"$SAIDA"
echo "## 2. Comparar os runs sem abrir a interface" >>"$SAIDA"
registra "just promover" "uv run --no-active python promover.py"

echo "" >>"$SAIDA"
echo "## 3. Carregar pelo alias e prever" >>"$SAIDA"
registra "just prever" "uv run --no-active python inferencia.py"

echo "" >>"$SAIDA"
echo "## 4. O mesmo modelo como API REST" >>"$SAIDA"
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run --no-active \
    mlflow models serve -m "models:/wine-arvore@champion" -p 1234 --env-manager local \
    >.servidor.log 2>&1 &
SERVIDOR=$!
for _ in $(seq 1 40); do sleep 2; curl -s -o /dev/null --max-time 2 http://localhost:1234/ping && break; done
registra "Servidor no ar?" "curl -sS -o /dev/null -w 'GET /ping -> HTTP %{http_code}\n' http://localhost:1234/ping"
registra "Três vinhos pelo endpoint /invocations" "curl -sS -X POST http://localhost:1234/invocations -H 'Content-Type: application/json' -d @exemplo.json"
printf '\nOs rótulos reais dessas três linhas são `[0, 1, 0]`.\n' >>"$SAIDA"
kill "$SERVIDOR" 2>/dev/null || true

echo "Pronto. Veja $SAIDA"
