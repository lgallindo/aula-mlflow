"""Carrega o modelo campeão pelo alias e prevê. Não sabe qual run o treinou."""

import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Nenhum run_id, nenhum caminho de arquivo, nenhum "sklearn" no código.
modelo = mlflow.pyfunc.load_model("models:/wine-arvore@champion")
print(modelo)

X, y = load_wine(return_X_y=True, as_frame=True)
_, X_teste, _, y_teste = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

predicoes = modelo.predict(X_teste.head(8))
print("\nprevisto:", list(predicoes))
print("real:    ", list(y_teste.head(8)))
