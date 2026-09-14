"""Treina variações de uma árvore de decisão e registra tudo no MLflow."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Um arquivo SQLite: é o suficiente para ter tracking E Model Registry,
# sem subir servidor nenhum.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-arvore")

X, y = load_wine(return_X_y=True, as_frame=True)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

VARIACOES = [
    {"max_depth": 2, "criterion": "gini"},
    {"max_depth": 3, "criterion": "gini"},
    {"max_depth": 5, "criterion": "gini"},
    {"max_depth": 3, "criterion": "entropy"},
    {"max_depth": None, "criterion": "entropy", "min_samples_leaf": 5},
]

# Uma linha, e o MLflow passa a salvar sozinho hiperparâmetros, métricas de
# treino, o modelo e os metadados de cada run.
mlflow.sklearn.autolog()

for variacao in VARIACOES:
    nome = "-".join(f"{chave}={valor}" for chave, valor in variacao.items())
    with mlflow.start_run(run_name=nome):
        modelo = DecisionTreeClassifier(random_state=42, **variacao)

        # Validação cruzada no conjunto de treino: cinco notas em vez de uma.
        notas = cross_val_score(modelo, X_treino, y_treino, cv=5, scoring="f1_macro")
        modelo.fit(X_treino, y_treino)
        predicoes = modelo.predict(X_teste)

        # O que o autolog() não sabe que interessa: nós escrevemos.
        mlflow.log_metric("f1_cv_media", notas.mean())
        mlflow.log_metric("f1_cv_desvio", notas.std())
        mlflow.log_metric("f1_teste", f1_score(y_teste, predicoes, average="macro"))

        ConfusionMatrixDisplay.from_predictions(y_teste, predicoes)
        plt.title(nome)
        plt.savefig("matriz.png", dpi=90, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("matriz.png")

        print(f"{nome:45} f1_cv={notas.mean():.4f} (+-{notas.std():.4f})")

print("\nPronto. Veja com:  just ui")
