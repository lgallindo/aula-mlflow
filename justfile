default:
    @just --list

# Instala o ambiente.
sync:
    uv sync --no-active

# Treina as cinco variações e registra tudo no MLflow.
treinar:
    uv run --no-active python treino.py

# Abre a interface do MLflow em http://localhost:5000
ui:
    uv run --no-active mlflow ui --backend-store-uri sqlite:///mlflow.db

# Registra os dois melhores e aplica os aliases champion/challenger.
promover:
    uv run --no-active python promover.py

# Carrega models:/wine-arvore@champion e prevê oito linhas.
prever:
    uv run --no-active python inferencia.py

# Sobe uma API REST a partir do modelo campeão, em http://localhost:1234
servir:
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run --no-active \
        mlflow models serve -m "models:/wine-arvore@champion" -p 1234 --env-manager local

# Com `just servir` rodando em outro terminal: manda três vinhos e lê a resposta.
curl:
    curl -sS -X POST http://localhost:1234/invocations \
        -H 'Content-Type: application/json' -d @exemplo.json

# Apaga runs, modelos e o banco. A próxima `just treinar` começa do zero.
limpar:
    rm -rf mlflow.db mlruns mlartifacts matriz.png
