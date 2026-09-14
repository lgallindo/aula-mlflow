# Onde foi parar o modelo bom?

Você treinou vinte variações esta semana. Qual delas era a boa? Com que
hiperparâmetros? Deu quanto no teste? E o arquivo, está em qual pasta?

Se a resposta envolve rolar o notebook para cima, um `modelo_final_v3_agora_vai.pkl`
ou uma planilha, esta aula é sobre isso.

Esta pasta é autossuficiente: os dados vêm do `scikit-learn`, o banco é um arquivo
local, e nada aqui depende de outra pasta.

## Instalar

```
just sync
```

`uv` e `just`. O dataset (`wine`, 178 vinhos, 13 medidas, 3 castas) já vem dentro
do `scikit-learn` — não tem download.

## 1. O ciclo de vida não cabe em um `pickle`

Salvar o modelo com `joblib.dump` guarda **o objeto**. Não guarda:

- com quais hiperparâmetros ele foi treinado;
- que acurácia ele teve, e em que conjunto;
- qual versão do código e dos dados produziu aquilo;
- quem treinou, quando, e se essa é a versão que está em produção.

O MLflow é uma plataforma open-source para esse ciclo inteiro, em quatro peças:

| Peça | O que faz |
| --- | --- |
| **Tracking** | Registra parâmetros, métricas, artefatos e código de cada execução |
| **Models** | Um formato padrão para salvar modelo, independente do framework |
| **Registry** | Versões, histórico e qual versão é a oficial |
| **Projects** | Empacotamento do código para rodar em outro lugar |

Esta aula usa as três primeiras. `Projects` fica de fora de propósito: é a peça
menos usada na prática, e contêiner resolve o mesmo problema melhor.

## 2. Onde isso tudo fica salvo

Uma linha, no começo de `treino.py`:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

Um arquivo SQLite nesta pasta. Sem servidor, sem nuvem, sem conta.

> **Por que SQLite e não a pasta `./mlruns` padrão?** Porque o Model Registry —
> as seções 7 e 8 desta aula — **não funciona** com o armazenamento em arquivos
> soltos. Com um banco, funciona. É a diferença entre um comando e uma mensagem
> de erro no meio da aula.

Quando o time cresce, o mesmo MLflow vira um servidor que todo mundo acessa —
inclusive como contêiner (`ghcr.io/mlflow/mlflow`, porta 5000), com o script
apontando para `mlflow.set_tracking_uri("http://localhost:5000")`. O código de
quem treina **não muda**: muda só o endereço. *(Essa variante não foi executada
na preparação desta aula; o que está medido aqui é o caminho SQLite.)*

## 3. `autolog()` — uma linha, e o registro acontece

```python
mlflow.sklearn.autolog()
```

A partir daí, todo `fit()` salva sozinho: hiperparâmetros, métricas de
treinamento, o modelo serializado, o formato dos dados de entrada, usuário e
carimbo de tempo.

```
just treinar
```

Cinco variações de árvore de decisão, com validação cruzada de 5 folds:

```
max_depth=2-criterion=gini                          f1_cv=0.8715 (+-0.0781)
max_depth=3-criterion=gini                          f1_cv=0.9055 (+-0.0813)
max_depth=5-criterion=gini                          f1_cv=0.8989 (+-0.0744)
max_depth=3-criterion=entropy                       f1_cv=0.8917 (+-0.0655)
max_depth=None-criterion=entropy-min_samples_leaf=5 f1_cv=0.8644 (+-0.0743)
```

## 4. Tracking manual — o que o `autolog()` não sabe

O `autolog()` não conhece a **sua** métrica. A validação cruzada acima, por
exemplo: ela acontece antes do `fit`, e o `autolog` não a vê. Então escrevemos:

```python
with mlflow.start_run(run_name=nome):
    mlflow.log_metric("f1_cv_media", notas.mean())
    mlflow.log_metric("f1_cv_desvio", notas.std())
    mlflow.log_metric("f1_teste", f1_score(y_teste, predicoes, average="macro"))
    mlflow.log_artifact("matriz.png")
```

| Comando | Para quê |
| --- | --- |
| `log_param` | Um valor de entrada (o `autolog` já pega os do estimador) |
| `log_metric` | Um número de saída. Pode ser chamado várias vezes e vira uma série |
| `log_artifact` | Qualquer arquivo: gráfico, CSV, relatório |
| `log_model` | O modelo, no formato MLflow |

O nome é você quem escolhe. `f1_cv_desvio` não existe no MLflow — existe porque
escrevemos essa linha.

## 5. Comparar os runs

```
just ui
```

Abre <http://localhost:5000>. Em **Experiments**, cada run é uma linha, com
parâmetros e métricas em colunas. Dá para:

- **filtrar** com `metrics.f1_cv_media > 0.89`;
- selecionar dois ou mais runs e clicar em **Compare** — parâmetros e métricas
  lado a lado, mais gráficos de dispersão e de **coordenadas paralelas**, que
  mostram o efeito de cada hiperparâmetro na métrica;
- abrir um run e ver os artefatos (a matriz de confusão está lá), as métricas ao
  longo do tempo e o modelo salvo.

A mesma coisa sem sair do terminal, porque a UI não faz nada que a API não faça:

```python
mlflow.search_runs(experiment_names=["wine-arvore"], order_by=["metrics.f1_cv_media DESC"])
```

```
                                tags.mlflow.runName  metrics.f1_cv_media  metrics.f1_teste
                         max_depth=3-criterion=gini             0.905546          0.956469
                         max_depth=5-criterion=gini             0.898862          0.956469
                      max_depth=3-criterion=entropy             0.891708          0.914697
                         max_depth=2-criterion=gini             0.871520          0.878676
max_depth=None-criterion=entropy-min_samples_leaf=5             0.864390          0.891414
```

**Olhe as duas primeiras linhas.** No conjunto de teste as duas deram
**exatamente** `0.956469` — pelo teste, empate. Quem desempata é a validação
cruzada, que olhou cinco divisões em vez de uma: `0.9055` contra `0.8989`. A
profundidade 3 ganha, e ainda por cima é o modelo mais simples dos dois.

Sem registro, esse empate passaria despercebido e você escolheria pelo número que
estivesse na tela.

## 6. MLflow Models: `flavor` e `pyfunc`

Todo modelo salvo vira uma **pasta**, não um arquivo:

```
MLmodel            metadados: quando, com quê, que flavors, que assinatura
model.pkl          o modelo serializado
conda.yaml         as dependências, para quem usa conda
python_env.yaml    as mesmas, para quem usa virtualenv
requirements.txt
```

Abra o `MLmodel` de uma das runs: além dos flavors, ele traz a **assinatura** —
as 13 colunas com nome e tipo, e a saída como `int64`. Foi o `autolog()` que
inferiu isso do `fit`, e é essa assinatura que a API da seção 9 usa para recusar
uma entrada malformada.

O `MLmodel` declara os **flavors** — as maneiras de carregar aquele modelo.
O `scikit-learn` gera dois: `sklearn` (o objeto original, com `.feature_importances_`
e tudo mais) e **`python_function`**, o genérico.

`pyfunc` é a peça que faz o resto funcionar: **qualquer** modelo do MLflow —
sklearn, PyTorch, TensorFlow, XGBoost — carrega com a mesma função e responde ao
mesmo `.predict()`. É por isso que o time que vai consumir o modelo não precisa
saber em que framework ele foi treinado.

## 7. Model Registry: versão, não sobrescrita

Depois de 50 runs, ninguém lembra qual é o oficial. O Registry é o repositório
central do que acontece **depois** do treino:

```python
mlflow.register_model(f"runs:/{run_id}/model", "wine-arvore")
```

Registrar sob o **mesmo nome** não sobrescreve nada: cria a versão seguinte.
Cada versão guarda o histórico completo — quem registrou, quando, a partir de
qual run. Dá para fazer pela UI também, no botão **Register Model** dentro do run.

## 8. Aliases: `champion` e `challenger`

Versão é número, e número não diz qual está valendo. Para isso existe o
**alias** — um apelido fixo apontando para uma versão específica:

```
just promover
```

```
champion    -> versão 1  (max_depth=3-criterion=gini)
challenger  -> versão 2  (max_depth=5-criterion=gini)
```

```python
cliente.set_registered_model_alias("wine-arvore", "champion", versao)
```

A aplicação carrega **pelo apelido**, sem saber run nem versão:

```python
modelo = mlflow.pyfunc.load_model("models:/wine-arvore@champion")
```

```
just prever
```

```
previsto: [0, 1, 0, 0, 1, 0, 0, 1]
real:     [0, 1, 0, 0, 1, 0, 0, 1]
```

Promover um modelo novo passa a ser **mover o apelido**. Nada no código de quem
consome muda. E o `challenger` é o candidato que você está avaliando em paralelo
antes de trocar.

> Em tutoriais antigos você vai ver *stages*: `Staging`, `Production`,
> `Archived`. **Está obsoleto** — os aliases substituíram esse esquema.

## 9. O modelo como serviço

```
just servir     # em um terminal
just curl       # em outro
```

```
GET /ping -> HTTP 200
{"predictions": [0, 1, 0]}
```

Os rótulos reais dessas três linhas são `[0, 1, 0]`.

O que aconteceu: `mlflow models serve -m "models:/wine-arvore@champion" -p 1234`
subiu uma API REST a partir de um modelo registrado. **Você não escreveu uma
linha de código de servidor.** O endpoint `/invocations` recebe JSON e devolve
predições.

Isso é ótimo para testar antes de empacotar num serviço próprio (FastAPI,
BentoML) — e o mesmo modelo vira imagem de contêiner com
`mlflow models build-docker`.

O formato do corpo está em [`exemplo.json`](exemplo.json): `dataframe_split`,
com os nomes das colunas e as linhas. O MLflow valida a entrada contra a
assinatura que ele mesmo salvou — mande uma coluna a menos e ele reclama antes
de o modelo ver o dado.

## 10. Prática

Sozinho ou em dupla, **sem copiar o `treino.py`**:

1. Leia o conjunto `wine`.
2. Separe treino e teste de forma **estratificada**.
3. Treine e avalie variações de uma árvore de decisão com **validação cruzada** —
   mexa em `max_depth`, `criterion`, `min_samples_leaf`.
4. Registre parâmetros, métricas e **um artefato seu** com o MLflow.
5. Registre a melhor versão no Model Registry e dê a ela o alias `champion`.
6. Carregue pelo alias e preveja.

Entregue o `mlflow.db` ou uma captura da tela de **Compare** com as suas runs.

Quem terminar antes: troque a árvore por `RandomForestClassifier` e responda —
**o que muda no código que carrega o modelo?** (Nada. Esse é o ponto do `pyfunc`.)

## Os atalhos

| Comando | O que faz |
| --- | --- |
| `just sync` | Instala o ambiente |
| `just treinar` | Treina as cinco variações e registra tudo |
| `just ui` | Abre a interface em <http://localhost:5000> |
| `just promover` | Registra os dois melhores e aplica `champion` / `challenger` |
| `just prever` | Carrega `models:/wine-arvore@champion` e prevê |
| `just servir` | Sobe a API REST em <http://localhost:1234> |
| `just curl` | Manda três vinhos para a API |
| `just limpar` | Apaga runs, modelos e o banco |

`./demo.sh` roda tudo em sequência e grava `RESULTADOS.md` com a
saída da **sua** máquina. Os números deste README saíram de uma execução dessas,
em 13/09/2026, com MLflow 3.16.0 e scikit-learn 1.9.1.

## Sem instalar nada

O [`MLflow_na_pratica.ipynb`](MLflow_na_pratica.ipynb) roda inteiro no Google
Colab: instala o MLflow na sessão, treina, registra, promove e carrega pelo
alias. O que não dá para fazer lá é abrir a interface — por isso o notebook faz
a comparação com `search_runs`, que é a mesma consulta que a tela executa.

## O que o MLflow não resolve

- **Não avalia por você.** Ele registra a métrica que você escolheu. Métrica
  errada, registrada com todo rigor, continua errada.
- **Não versiona o dado.** O run aponta para o código e os parâmetros; o CSV que
  mudou ontem continua sendo problema seu (DVC, hash, data lake).
- **Não reproduz o ambiente sozinho.** Ele salva um `conda.yaml`; quem garante a
  mesma libc e a mesma GPU é contêiner, não ele.
- **Não é monitoramento.** Ele registra treino. O que acontece com as predições
  em produção — atraso, deriva, volume — é outra ferramenta.

## Referências

- *MLflow Tracking Quickstart* — <https://mlflow.org/docs/latest/ml/tracking/quickstart/>
- *MLflow Model Registry* — <https://mlflow.org/docs/latest/ml/model-registry/>
- *MLflow Deployment* — <https://mlflow.org/docs/latest/ml/deployment/>
