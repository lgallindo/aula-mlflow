"""Registra os dois melhores runs e dá a eles os apelidos champion e challenger."""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
NOME = "wine-arvore"

# A mesma busca que a UI faz quando você ordena por uma coluna — só que em pandas.
runs = mlflow.search_runs(
    experiment_names=[NOME], order_by=["metrics.f1_cv_media DESC"], max_results=5
)
print(runs[["tags.mlflow.runName", "metrics.f1_cv_media", "metrics.f1_teste"]].to_string(index=False))

cliente = MlflowClient()
for posicao, apelido in [(0, "champion"), (1, "challenger")]:
    run = runs.iloc[posicao]
    # Registrar sob o MESMO nome não sobrescreve nada: cria uma versão nova.
    versao = mlflow.register_model(f"runs:/{run['run_id']}/model", NOME)
    cliente.set_registered_model_alias(NOME, apelido, versao.version)
    print(f"\n{apelido:11} -> versão {versao.version}  ({run['tags.mlflow.runName']})")

print(f"\nCarregue sem saber o run_id:  models:/{NOME}@champion")
