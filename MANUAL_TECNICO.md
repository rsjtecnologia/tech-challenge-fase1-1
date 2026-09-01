# Manual Técnico — Tech Challenge Fase 1

**Pós-graduação IA para Devs (FIAP) — Fase 1, Tech Challenge A**

Sistema de apoio ao diagnóstico em saúde feminina com classificação de riscos via Machine Learning sobre dados médicos estruturados, cobrindo o pipeline completo: exploração, pré-processamento, modelagem, avaliação, explicabilidade e dashboard web de demonstração.

> **Aviso permanente do projeto:** ferramenta acadêmica de apoio à decisão clínica. O profissional de saúde tem sempre a palavra final — o sistema não substitui biópsia, histopatologia nem julgamento médico.

---

## Sumário

1. [Visão geral e escopo](#1-visão-geral-e-escopo)
2. [Arquitetura geral](#2-arquitetura-geral)
3. [Stack tecnológica](#3-stack-tecnológica)
4. [Estrutura do repositório](#4-estrutura-do-repositório)
5. [Ambiente e automação (scripts PowerShell/Python)](#5-ambiente-e-automação)
6. [Metodologia de desenvolvimento dos notebooks](#6-metodologia-de-desenvolvimento-dos-notebooks)
7. [Estudo 1 — Câncer de mama (Wisconsin)](#7-estudo-1--câncer-de-mama-wisconsin)
8. [Estudo 2 — SOP (PCOS)](#8-estudo-2--sop-pcos)
9. [Relatório técnico consolidado](#9-relatório-técnico-consolidado)
   - [9.1 Discussões da análise exploratória](#91-discussões-da-análise-exploratória)
   - [9.2 Estratégias de pré-processamento](#92-estratégias-de-pré-processamento)
   - [9.3 Modelos usados e porquê](#93-modelos-usados-e-porquê)
   - [9.4 Resultados e interpretação dos dados](#94-resultados-e-interpretação-dos-dados)
10. [Contrato de dados notebooks → web](#10-contrato-de-dados-notebooks--web)
11. [Aplicação web — backend (FastAPI)](#11-aplicação-web--backend-fastapi)
12. [Aplicação web — frontend](#12-aplicação-web--frontend)
13. [Guia de execução completo](#13-guia-de-execução-completo)
14. [Limitações conhecidas e pontos de atenção](#14-limitações-conhecidas-e-pontos-de-atenção)

---

## 1. Visão geral e escopo

O projeto implementa **dois estudos independentes** de classificação binária clínica, cada um com seu próprio pipeline de notebooks e sua própria página no dashboard:

| Estudo | Dataset | Problema | Rota web |
|---|---|---|---|
| Câncer de mama | [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — 569 amostras, 30 features numéricas | Maligno (M) vs. Benigno (B) | `/` |
| SOP (Síndrome do Ovário Policístico) | [PCOS Infertility](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) — 541 amostras, 3 biomarcadores | Com PCOS vs. Sem PCOS | `/pcos` |

Em ambos os estudos a decisão central de projeto é a mesma: **o falso negativo é o erro clinicamente mais grave** (a paciente doente deixa de ser investigada). Por isso, a métrica de seleção do "melhor modelo" é sempre o **recall da classe positiva** (maligno / com PCOS), e não a acurácia.

O produto final não faz inferência online: os notebooks treinam, avaliam e **pré-computam todas as predições e métricas**, exportando um "contrato de dados" (Parquet + JSON) que o dashboard FastAPI apenas serve e visualiza.

---

## 2. Arquitetura geral

O fluxo de ponta a ponta é um pipeline em três camadas, encadeado por artefatos em disco:

```
┌──────────────────────┐     ┌──────────────────────────────┐     ┌───────────────────────┐
│ Datasets (CSV)       │     │ Notebooks (pipeline por      │     │ Dashboard web         │
│ src/db/models/       │ ──► │ estudo, encadeados por       │ ──► │ FastAPI + Chart.js    │
│                      │     │ artefatos .joblib)           │     │ src/web/              │
└──────────────────────┘     │                              │     └───────────────────────┘
                             │ Saídas por etapa em          │
                             │ src/db/outputs/<estudo>/     │            ▲
                             │  ├─ figures/  (PNG)          │            │ lê JSON + Parquet
                             │  ├─ metrics/  (JSON)         │            │ (contrato de export)
                             │  └─ artifacts/ (joblib/      │────────────┘
                             │                parquet)      │
                             └──────────────────────────────┘
```

Pontos-chave da arquitetura:

- **Encadeamento por artefatos:** cada notebook lê o artefato `.joblib` produzido pelo anterior e falha com `FileNotFoundError` e mensagem explícita ("Execute o notebook X primeiro") se ele não existir. Isso torna a ordem de execução auto-verificável.
- **Separação estrita treino/avaliação:** o notebook de modelagem treina **somente no conjunto de treino** e "empacota" o teste sem avaliá-lo; as métricas só são computadas no notebook seguinte. Imputação e padronização ficam **dentro** do `Pipeline` do scikit-learn, então o `fit` do scaler acontece só com dados de treino — sem vazamento de dados.
- **Contrato versionado:** o último notebook de cada estudo gera um conjunto fixo de arquivos (2 Parquet + 2 JSON) com `versao_contrato` explícita. O backend só conhece esse contrato — nunca os modelos.
- **Web somente leitura:** a API não carrega modelo nenhum; serve métricas, gráficos e a tabela de amostras pré-computados, com cache em memória (`lru_cache`) e endpoint para recarregar.

A pasta `src/db/outputs/` está no `.gitignore` — os artefatos são regenerados em runtime executando os notebooks.

---

## 3. Stack tecnológica

Dependências declaradas em `requirements.txt`:

| Camada | Bibliotecas | Uso |
|---|---|---|
| Dados | `pandas>=2.0`, `numpy>=1.24`, `pyarrow>=18.0` | Manipulação de dados; Parquet (engine pyarrow, compressão snappy) |
| ML | `scikit-learn>=1.3` | Pipelines, split, imputação, scaling, 5 algoritmos de classificação, métricas, calibração |
| Explicabilidade | `shap>=0.43` | SHAP global (summary plot) e local (waterfall) |
| Visualização (notebooks) | `matplotlib>=3.7`, `seaborn>=0.13` | EDA, matrizes de confusão, heatmaps |
| Execução de notebooks | `jupyter>=1.0`, `nbformat>=5.10`, `nbclient>=0.10`, `ipykernel>=6.29` | Execução programática via `run_notebook.py` |
| Web backend | `fastapi>=0.115`, `uvicorn[standard]>=0.34`, `jinja2>=3.1` | API REST + servir templates HTML |
| Web frontend | Chart.js 4.4.7 (CDN), Google Fonts (DM Sans, Manrope) | Gráficos no navegador, sem build step |

Persistência de modelos: **joblib** (`.joblib`) entre notebooks. Consumo web: **JSON + Parquet**. Nenhum `.pkl` é usado.

---

## 4. Estrutura do repositório

```
tech-challenge-fase1/
├── notebooks/
│   ├── breast-cancer-wisconsin-data/      # Pipeline do estudo 1 (02→08)
│   │   ├── 02_analise_exploratoria.ipynb
│   │   ├── 03_preprocessamento.ipynb
│   │   ├── 04_modelagem_treino.ipynb
│   │   ├── 05_avaliacao_metricas.ipynb
│   │   ├── 06_explicabilidade.ipynb
│   │   ├── 07_discussao_demo.ipynb
│   │   └── 08_export_web.ipynb
│   └── polycystic-ovary-syndrome-pcos/    # Pipeline do estudo 2 (01→05)
│       ├── 01_analise_exploratoria.ipynb
│       ├── 02_preprocessamento.ipynb
│       ├── 03_modelagem_treino.ipynb
│       ├── 04_avaliacao_metricas.ipynb
│       ├── 05_export_web.ipynb
│       └── export_pcos_web_runner.py      # Lógica do export extraída do notebook 05
├── src/
│   ├── db/
│   │   ├── models/                        # Datasets brutos (CSV/XLSX)
│   │   │   ├── breast-cancer-wisconsin-data/data.csv
│   │   │   └── polycystic-ovary-syndrome-pcos/
│   │   │       ├── PCOS_infertility.csv               # usado
│   │   │       └── PCOS_data_without_infertility.xlsx # presente, NÃO usado
│   │   └── outputs/                       # Gerado em runtime (gitignored)
│   └── web/                               # Dashboard FastAPI
│       ├── app.py                         # App, rotas de página, mount de estáticos
│       ├── api/
│       │   ├── routes.py                  # API do estudo mama  (prefixo /api)
│       │   └── pcos_routes.py             # API do estudo PCOS  (prefixo /api/pcos)
│       ├── services/
│       │   ├── data_loader.py             # Carrega contrato do estudo mama (com cache)
│       │   └── pcos_data_loader.py        # Idem para PCOS
│       ├── templates/
│       │   ├── index.html                 # Página do estudo mama
│       │   └── pcos.html                  # Página do estudo PCOS
│       └── static/
│           ├── css/app.css                # Tema único (dark/glassmorphism) p/ as 2 páginas
│           └── js/
│               ├── app.js                 # Lógica da página mama
│               └── pcos.js                # Lógica da página PCOS
├── pre_execucao.ps1                       # Prepara venv + instala dependências
├── run_notebook.py                        # Executa 1 notebook com progresso por célula
├── run_notebooks.ps1                      # Menu interativo p/ executar os pipelines
├── run_web.ps1                            # Sobe o dashboard (detecção de porta livre)
├── requirements.txt
└── README.md
```

---



## 5. Ambiente e automação

O projeto foi desenvolvido para **Windows + PowerShell**, com três scripts de automação e um executor de notebooks em Python.

### 5.1 `pre_execucao.ps1` — preparação do ambiente

Responsável por deixar o ambiente pronto de forma idempotente. O que ele faz, em ordem:

1. **Detecta a versão mais recente de Python 3** instalada na máquina: tenta o launcher `py` (varre `py -0p`, ordena as versões e valida executando `sys.version_info`), com fallback para `python3`/`python` no PATH. Falha com mensagem clara se nenhum Python 3 existir.
2. **Valida o venv existente** (`.venv/` na raiz): se o venv foi criado com uma versão de Python diferente da detectada (ou está corrompido), ele é **removido e recriado** automaticamente. Se o venv problemático estiver ativo na sessão atual, o script aborta e instrui o usuário a desativá-lo primeiro.
3. **Cria e ativa o venv** (`py -X.Y -m venv .venv` + dot-sourcing de `Activate.ps1`). Para o venv permanecer ativo na sessão do usuário, o script deve ser chamado com dot-sourcing: `. .\pre_execucao.ps1`.
4. **Confere que o interpretador ativo** bate com a versão esperada e **instala `requirements.txt`** usando o `python.exe` do venv, propagando falha do pip como erro.

### 5.2 `run_notebook.py` — executor de um notebook

Executa um `.ipynb` de ponta a ponta via `nbformat` + `nbclient`, com estas características:

- **Progresso por célula:** imprime `[n/total] <primeira linha da célula>` antes de executar cada célula de código.
- **Diretório de trabalho correto:** passa `resources={"metadata": {"path": <pasta do notebook>}}` ao `NotebookClient`, para que caminhos relativos dentro do notebook resolvam a partir da pasta do próprio notebook.
- **Workaround Windows:** força `WindowsSelectorEventLoopPolicy` no asyncio (Python < 3.14), porque o `ProactorEventLoop` padrão do Windows não implementa `add_reader`, exigido pelo ZeroMQ do kernel Jupyter.
- **Sem timeout** (`timeout=None`) — células longas (ex.: SHAP KernelExplainer) não são interrompidas.
- **Persistência dos outputs:** ao final, o notebook é salvo de volta no mesmo arquivo, com os outputs da execução (por isso os `.ipynb` do repositório contêm resultados e imagens).

Uso: `python run_notebook.py notebooks/<estudo>/<notebook>.ipynb`

### 5.3 `run_notebooks.ps1` — executor interativo dos pipelines

Menu de console em dois níveis:

1. Configura o console para UTF-8 (`chcp 65001`) para não quebrar acentos, e faz dot-sourcing de `pre_execucao.ps1` (ambiente sempre pronto antes de executar).
2. **Menu de estudos:** `[1]` Câncer de mama, `[2]` SOP (PCOS), `[T]` todos os estudos em ordem, `[S]` sair.
3. **Menu de notebooks do estudo:** lista os notebooks do pipeline na ordem correta (note que a **EDA não está na lista** — ela é opcional para reproduzir o pipeline), com `[T]` para executar todos em sequência.
4. Cada notebook é executado via `run_notebook.py` com o Python do venv; qualquer falha interrompe a sequência com mensagem em vermelho.

Listas de execução embutidas no script (ordem oficial de reprodução):
- Mama: `03 → 04 → 05 → 06 → 07 → 08`
- PCOS: `02 → 03 → 04 → 05`

### 5.4 `run_web.ps1` — subir o dashboard

1. Faz dot-sourcing de `pre_execucao.ps1` (garante venv + dependências).
2. **Seleção de porta:** usa `System.Net.NetworkInformation.IPGlobalProperties` para listar as portas TCP em uso. Sem parâmetro, escolhe a primeira porta livre entre **8010–8013**; com `-Port <n>`, valida se a porta pedida está livre (erro orientando alternativa se não estiver).
3. Sobe o servidor: `python -m uvicorn src.web.app:app --host 127.0.0.1 --port <porta> --reload` (hot reload habilitado para desenvolvimento).
4. Imprime as URLs do dashboard e da documentação automática da API (`/docs`, Swagger UI do FastAPI).

---

## 6. Metodologia de desenvolvimento dos notebooks

Os dois pipelines seguem **as mesmas convenções de engenharia**, que valem a pena entender antes do detalhe de cada estudo:

1. **Um notebook = uma etapa, encadeados por artefatos.** A EDA não exporta nada de modelagem; o pré-processamento exporta `dados_preprocessados.joblib`; a modelagem exporta `modelos_treinados.joblib`; a avaliação exporta `metricas_teste.json` e um joblib enriquecido com `nome_melhor_modelo`; o export web gera o contrato para o dashboard. Cada notebook começa validando que o artefato do anterior existe.

2. **Resolução de caminhos portável.** Todo notebook define `encontrar_raiz()`, que sobe a árvore de diretórios até achar a pasta que contém `src/db`. Assim o notebook funciona independentemente de onde o kernel foi iniciado. As saídas vão para `src/db/outputs/<estudo>/<NN-etapa>/` com a tríade `figures/` (PNG), `metrics/` (JSON) e `artifacts/` (joblib/parquet).

3. **Reprodutibilidade:** semente fixa `SEMENTE = 42` em split e em todos os modelos com componente aleatório.

4. **Prevenção de vazamento de dados:** o pré-processador (`ColumnTransformer` com `SimpleImputer(mediana) → StandardScaler`) é **instanciado** no notebook de pré-processamento, mas só é **fitado** dentro do `Pipeline([preprocessador, classificador])` no notebook de treino — portanto exclusivamente com dados de treino. O conjunto de teste "viaja" pelos artefatos sem ser tocado até o notebook de avaliação.

5. **Células de validação (`assert`).** Todo notebook termina com um bloco de asserts que verifica os artefatos gerados (existência, chaves, contagens, consistência do melhor modelo) e imprime "Validação OK". Funciona como um teste de contrato embutido no pipeline.

6. **Os mesmos 5 algoritmos nos dois estudos**, sem tuning de hiperparâmetros (valores fixos/default; não há GridSearchCV nem cross-validation de seleção — a única CV existente é a interna do `CalibratedClassifierCV(cv=5)`, usada apenas para calibrar as probabilidades do SVC, que nativamente não expõe `predict_proba`):

| Modelo | Classe sklearn | Hiperparâmetros definidos | Justificativa registrada no export |
|---|---|---|---|
| Regressão Logística | `LogisticRegression` | `max_iter=2000`, `random_state=42` (+ `class_weight="balanced"` no PCOS) | Baseline linear probabilístico, interpretável, adequado à triagem |
| Árvore de Decisão | `DecisionTreeClassifier` | `random_state=42`, sem poda (+ `class_weight="balanced"` no PCOS) | Relações não lineares + importância global dos atributos |
| SVM | `CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"), method="sigmoid", cv=5, ensemble=False)` | Calibração Platt com CV interna de 5 folds | Fronteira não linear com probabilidades calibradas |
| Gradient Boosting | `GradientBoostingClassifier` | `random_state=42` (defaults: 100 árvores, lr 0.1, depth 3) | Ensemble sequencial que corrige erros |
| Random Forest | `RandomForestClassifier` | `n_estimators=300`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` | Muitas árvores para reduzir variância |

7. **Desbalanceamento tratado por `class_weight="balanced"`** (não há SMOTE/reamostragem) e **critério clínico de seleção**: `max(recall da classe positiva)`.

---

## 7. Estudo 1 — Câncer de mama (Wisconsin)

**Dataset:** `src/db/models/breast-cancer-wisconsin-data/data.csv` — 569 linhas × 33 colunas. Cada linha é uma amostra de massa mamária com 30 medidas do núcleo celular extraídas de imagens de FNA (aspirado por agulha fina), em 3 famílias de sufixo: `_mean` (média), `_se` (erro padrão) e `_worst` (pior valor) para 10 características (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal). Alvo: `diagnosis` (M/B).

**Cadeia:** `02 (EDA) → 03 (pré-processamento) → 04 (treino) → 05 (avaliação) → 06 (explicabilidade) → 07 (discussão/demo) → 08 (export web)`. Não existe `01_*` nesta pasta — o 02 acumula visão geral do problema + inspeção + EDA.

### 7.1 `02_analise_exploratoria.ipynb` — EDA

- Carrega o CSV com `str.strip()` nos nomes de colunas; descarta `id` e `Unnamed: 32` (coluna vazia do CSV) → 30 features + alvo.
- Distribuição do alvo: **Benigno 357 (62,7%) | Maligno 212 (37,3%)** — desbalanceamento leve.
- `describe()` global e médias por classe. Destaques que separam as classes: `radius_mean` (B 12,15 vs M 17,46), `area_mean` (462,8 vs 978,4), `concave points_mean` (0,026 vs 0,088), `area_worst` (558,9 vs 1422,3).
- Figuras geradas (em `02-analise-exploratoria/figures/`): countplot do diagnóstico, histogramas 2×5 por família de sufixo, boxplots por diagnóstico e por família, e heatmap de correlação de Pearson (máscara triangular, cmap coolwarm) incluindo o alvo recodificado M=1/B=0 — recodificação declarada como "apenas exploratória".
- Não exporta artefatos de modelagem.

### 7.2 `03_preprocessamento.ipynb`

- Recarrega o CSV bruto; qualidade: **0 ausentes, 0 duplicatas**.
- **Encoding do alvo:** `{"M": 1, "B": 0}` — maligno é a classe positiva porque o foco clínico é recall/falso negativo. Não há variáveis categóricas, portanto não há one-hot.
- **Análise de colinearidade:** identifica **21 pares de features com |r| > 0,9** (ex.: `radius_mean × perimeter_mean` = 0,998). Decisão documentada: **manter todas as colunas** nesta fase (padronizar escalas sem remover atributos).
- **Split estratificado:** `train_test_split(test_size=0.2, random_state=42, stratify=rotulo)` → treino **455×30**, teste **114×30** (proporção de maligno 0,374 / 0,368).
- **Pré-processador (instanciado, não fitado):** `ColumnTransformer` → `Pipeline(SimpleImputer(strategy="median") → StandardScaler())` nas 30 colunas. Mediana por robustez a outliers; scaler necessário para a regressão logística e o SVM.
- **Exporta** `03-preprocessamento/artifacts/dados_preprocessados.joblib` com: `atributos_treino/teste`, `rotulo_treino/teste`, `colunas_atributos`, `preprocessador`, `semente`.

### 7.3 `04_modelagem_treino.ipynb`

- Carrega o joblib do 03; treina os **5 pipelines** (tabela da seção 6) com `pipeline.fit(atributos_treino, rotulo_treino)`.
- Regra explícita em markdown: **"Não avaliar o teste aqui"** — o teste é apenas re-empacotado no artefato para "avaliação honesta depois".
- **Exporta** `04-modelagem-treino/artifacts/modelos_treinados.joblib` com `modelos_treinados` (dict nome → pipeline fitado), `rotulo_teste`, `atributos_teste`, `colunas_atributos`.

### 7.4 `05_avaliacao_metricas.ipynb`

- Para cada modelo: `predict` no teste → `classification_report` → extrai `acuracia`, `recall_maligno`, `f1_maligno`, `precisao_maligno`; plota e salva a matriz de confusão (`matriz_confusao_<slug>.png`).
- **Resultados no holdout (n=114; 42 malignos / 72 benignos):**

| Modelo | Acurácia | Recall (M) | F1 (M) | Precisão (M) | FN |
|---|---|---|---|---|---|
| Regressão Logística | 0,9649 | 0,9286 | 0,9512 | 0,9750 | 3 |
| Árvore de Decisão | 0,9298 | 0,9048 | 0,9048 | 0,9048 | 4 |
| **SVM (escolhido)** | **0,9825** | **0,9762** | **0,9762** | **0,9762** | **1** |
| Gradient Boosting | 0,9649 | 0,9048 | 0,9500 | 1,0000 | 4 |
| Random Forest | 0,9737 | 0,9286 | 0,9630 | 1,0000 | 3 |

- **Seleção:** `max(recall_maligno)` → **SVM** (SVC RBF calibrado), com apenas **1 falso negativo** em 42 casos malignos. A discussão em markdown justifica o critério: FN tem custo clínico grave; FP é "erro mais recuperável" (exames complementares).
- **Exporta** `metrics/metricas_teste.json` (métricas por modelo + `_melhor_modelo`) e re-salva o joblib em `05-avaliacao-metricas/artifacts/` acrescido de `nome_melhor_modelo` — é este artefato que os notebooks 06/07/08 consomem.

### 7.5 `06_explicabilidade.ipynb` — SHAP e feature importance

- **Importância global (Gini):** extrai `feature_importances_` da **Árvore de Decisão** (por ser o modelo com importância nativa) e plota o top-15. Resultado fortemente concentrado: `perimeter_worst` domina com **0,7275**, seguido de `concave points_worst` (0,079).
- **SHAP no melhor modelo:** o código escolhe o explicador automaticamente conforme o tipo do classificador (desembrulhando o `CalibratedClassifierCV`): `TreeExplainer` para modelos de árvore, `LinearExplainer` para lineares, e **`KernelExplainer`** (agnóstico, baseado em `predict_proba`) como fallback — que é o ramo usado, pois o vencedor é o SVM. Background de 50 amostras do treino, `nsamples=100`.
- Gera `shap_resumo.png` (summary plot global, classe maligno) e ranking por |SHAP| médio: top-3 `texture_worst` (0,0457), `radius_worst` (0,0401), `perimeter_worst` (0,0351).
- **SHAP local:** waterfall plot do caso 0 do teste (`shap_exemplo_caso.png`) — real benigno, previsto benigno, P(maligno) = 0,07%.
- Discussão em markdown: "importância ≠ causalidade"; a colinearidade radius/perimeter/area distribui a importância entre variáveis correlacionadas; as visualizações servem para auditar e comunicar o modelo.
- ⚠️ Nota de manutenção: o texto markdown da seção "Interpretação" ficou de uma execução anterior (cita Regressão Logística como vencedora), enquanto o output real da execução registrada usa SVM.

### 7.6 `07_discussao_demo.ipynb`

- Consolida a tabela comparativa de métricas e faz uma **demo de inferência**: primeira amostra do teste → `predict` + `predict_proba` do melhor pipeline, com impressão do aviso legal ("Apoio à decisão. Não substitui diagnóstico médico.").
- Documenta as **limitações**: dataset pequeno (569 amostras), população de contexto institucional específico, features pré-calculadas (o pipeline não cobre erro de aquisição/segmentação da imagem), desbalanceamento leve + colinearidade afetando a interpretação de importância.
- Não exporta arquivos; é o texto-base para o relatório da entrega.

### 7.7 `08_export_web.ipynb` — contrato para o dashboard

Monta e valida o contrato de dados consumido pela rota `/` do dashboard, sem retreinar nada. Entradas: joblib e JSON do 05, joblib do 03 e o CSV bruto (para o inventário). Conteúdo computado:

- **Grid comparativo (formato largo)** — 114 linhas × 51 colunas: `id` sequencial, 30 features, `rotulo_real`, `diagnostico_real` e, por modelo, `pred_<modelo>`, `prob_<modelo>` (P(maligno), 6 casas), `acerto_<modelo>`; mais os agregados `n_modelos_acertaram`, `consenso` (todos acertaram ou todos erraram) e `amplitude_probabilidade` (máx−mín entre os 5 modelos — mede divergência entre modelos na amostra).
- **Predições em formato longo** — 570 linhas (114 amostras × 5 modelos).
- **Matrizes de confusão** por modelo decompostas em `tn/fp/fn/tp`; **ranking** dos modelos pelas 4 métricas; **hiperparâmetros reais** extraídos via `get_params()` + justificativa textual por modelo; bloco `preprocessamento` descritivo; marcação `melhor_em`/`pior_em` com `np.isclose` (empates preservados).
- **Dados de gráficos:** splits (completo/treino/teste com contagens e % por classe), histogramas de cada feature por split (16 bins compartilhados entre as classes), matriz de correlação 31×31, importância de atributos (Árvore de Decisão) e inventário de qualidade (registros brutos/usados, colunas removidas, semente, test_size, estratificação).

**Arquivos gerados** (em `src/db/outputs/breast-cancer-wisconsin-data/08-export-web/`):

| Arquivo | Pasta | Conteúdo |
|---|---|---|
| `amostras_comparativo.parquet` | `artifacts/` | Grid largo 114×51 |
| `predicoes_longas.parquet` | `artifacts/` | Formato longo 570×8 |
| `metricas_comparativo.json` | `metrics/` | `versao_contrato: 2`, `criterio_selecao: recall_maligno`, `melhor_modelo: SVM`, ranking, por_modelo, preprocessamento |
| `graficos.json` | `metrics/` | inventário, splits, histogramas, correlação, importância |

---

## 8. Estudo 2 — SOP (PCOS)

**Dataset:** `src/db/models/polycystic-ovary-syndrome-pcos/PCOS_infertility.csv` — 541 linhas × 6 colunas: `Sl. No`, `Patient File No.`, `PCOS (Y/N)` (alvo), `I beta-HCG (mIU/mL)`, `II beta-HCG (mIU/mL)`, `AMH (ng/mL)`. O arquivo `PCOS_data_without_infertility.xlsx` presente na mesma pasta **não é usado**.

**Cadeia:** `01 (EDA) → 02 (pré-processamento) → 03 (treino) → 04 (avaliação) → 05 (export web)`. Não há notebook de explicabilidade dedicado neste estudo (a importância de atributos vai direto no export).

### 8.1 `01_analise_exploratoria.ipynb` — EDA

- Renomeia colunas para nomes limpos (`beta_hcg_i`, `beta_hcg_ii`, `amh`, `pcos`, ...) e força coerção numérica dos biomarcadores.
- Distribuição do alvo: **364 sem PCOS (67,3%) | 177 com PCOS (32,7%)**.
- **Descoberta central da EDA — censura por limite de detecção:** o valor `1.99` aparece massivamente em beta-HCG (**191 ocorrências em beta-HCG I, 307 em beta-HCG II**) e representa "abaixo do limite de detecção do exame", não um valor real. Isso vira a principal decisão de pré-processamento.
- **AMH é o melhor sinal:** média 4,54 (sem PCOS) vs **7,84 (com PCOS)**; mediana 3,2 vs 5,9.
- **Limitação estrutural identificada:** **10 grupos de rótulo conflitante** — perfis bioquímicos idênticos (em geral totalmente censurados) com rótulos opostos, o que impõe um teto irredutível de erro a qualquer modelo.
- Distribuições com cauda extrema (std de beta-HCG I ≈ 3540; máx 32.460) → histogramas em escala `log1p`.
- Salva 5 figuras + `resumo_eda.json` com o diagnóstico de qualidade. Não exporta artefato de modelagem.

### 8.2 `02_preprocessamento.ipynb`

- **Censura → ausente:** valores `1.99` em beta-HCG I/II são convertidos para `NaN` (serão imputados pela mediana dentro do pipeline). Ausentes resultantes: 191 / 307 (+1 ausente original em AMH).
- **Engenharia de atributos** — de 3 biomarcadores para **9 features**:
  - `beta_hcg_max` (máx das duas medições), `beta_hcg_diff` (|I − II|), `beta_hcg_ratio` (I/II);
  - `amh_alto` (binária: `amh >= 4.5`);
  - `beta_i_censurado`, `beta_ii_censurado` (flags "o valor era 1.99" — preservam a informação de missingness, que aqui é informativa).
- Descarta identificadores (`Sl. No`, `Patient File No.`).
- **Split estratificado 80/20** com semente 42 → treino **432×9**, teste **109×9** (32,6% / 33,0% positivos).
- Mesmo pré-processador do estudo 1 (mediana + StandardScaler em `ColumnTransformer`), instanciado sem fit.
- **Exporta** `02-preprocessamento/artifacts/dados_preprocessados.joblib` (mesmas chaves do estudo 1).

### 8.3 `03_modelagem_treino.ipynb`

- Idêntico em estrutura ao estudo 1: 5 pipelines `preprocessador → classificador` fitados só no treino (tabela da seção 6; aqui a Regressão Logística e a Árvore também usam `class_weight="balanced"`).
- **Exporta** `03-modelagem-treino/artifacts/modelos_treinados.joblib`.

### 8.4 `04_avaliacao_metricas.ipynb`

- Mesmo procedimento do estudo 1. **Resultados no holdout (n=109):**

| Modelo | Acurácia | Recall PCOS | F1 PCOS |
|---|---|---|---|
| Regressão Logística | 0,6514 | 0,5278 | 0,5000 |
| Árvore de Decisão | 0,6422 | 0,4722 | 0,4658 |
| SVM (calibrado) | 0,6330 | 0,4167 | 0,4286 |
| Gradient Boosting | 0,6422 | 0,3611 | 0,4000 |
| **Random Forest** | **0,6606** | **0,5556** | **0,5195** |

- **Seleção:** `max(recall_pcos)` → **Random Forest**.
- Leitura honesta documentada: o desempenho é modesto (nenhum modelo supera em acurácia a baseline "prever sempre negativo" ≈ 0,67); com apenas 3 biomarcadores — dois deles fortemente censurados — e rótulos conflitantes na base, o ganho real está no recall da classe positiva. AMH é o principal sinal.
- **Exporta** `metrics/metricas_teste.json` + joblib enriquecido em `04-avaliacao-metricas/artifacts/`.

### 8.5 `05_export_web.ipynb` + `export_pcos_web_runner.py`

- O notebook é um invólucro fino: importa e chama `exportar()` de `export_pcos_web_runner.py` (lógica extraída para um módulo Python reutilizável e testável fora do notebook) e valida as saídas.
- O runner computa o mesmo contrato do estudo 1, com particularidades: histogramas de **12 bins** (NaNs excluídos), importância de atributos da **Árvore de Decisão**, médias de probabilidade/confiança por modelo (`media_prob_pcos`, `media_confianca` — usadas na tabela "resumo consolidado" do frontend PCOS) e um bloco `destaques_eda` (médias de AMH por classe + contagens de censura).
- Metadados: `versao_contrato: 1`, `dataset_id: "pcos"`, `criterio_selecao: "recall_pcos"`, nomes das classes positiva/negativa.

**Arquivos gerados** (em `src/db/outputs/polycystic-ovary-syndrome-pcos/05-export-web/`): `artifacts/amostras_comparativo.parquet`, `artifacts/predicoes_longas.parquet`, `metrics/metricas_web.json`, `metrics/graficos_web.json`.

---

## 9. Relatório técnico consolidado

Esta seção reúne, em formato de relatório, as quatro discussões centrais do trabalho — o que a análise exploratória revelou, como os dados foram preparados, por que cada modelo foi escolhido e o que os resultados significam — cruzando os dois estudos. O detalhe operacional (célula a célula) está nas seções 7 e 8.

### 9.1 Discussões da análise exploratória

**Câncer de mama (Wisconsin).** A base tem 569 amostras e 30 atributos numéricos, todos derivados de imagens de aspirado por agulha fina (FNA), organizados em três famílias por sufixo: `_mean` (valor médio da característica no núcleo celular), `_se` (erro padrão, isto é, incerteza da medida) e `_worst` (pior/maior valor observado). A EDA sustentou três conclusões que guiaram o resto do pipeline:

1. **O problema é bem separável.** As médias por classe mostram distâncias grandes nas features de tamanho e de irregularidade do contorno: `radius_mean` 12,15 (benigno) vs 17,46 (maligno), `area_mean` 462,8 vs 978,4, `concave points_mean` 0,026 vs 0,088, `area_worst` 558,9 vs 1422,3. Os boxplots por diagnóstico confirmam separação visual em quase todas as famílias — o que antecipa (e depois se confirma) que modelos relativamente simples atingiriam desempenho alto.
2. **O desbalanceamento é leve, mas real** (62,7% benigno / 37,3% maligno) — suficiente para justificar split estratificado e `class_weight="balanced"`, mas não reamostragem agressiva.
3. **Há colinearidade estrutural forte.** O heatmap de correlação (e depois a varredura numérica do notebook 03) mostra que raio, perímetro e área medem essencialmente a mesma grandeza física (r = 0,998 entre `radius_mean` e `perimeter_mean`; 21 pares com |r| > 0,9). A decisão foi **não remover** atributos — os modelos escolhidos toleram colinearidade em termos de predição — mas registrar que isso **dilui a interpretação de importância** (a relevância de "tamanho do tumor" se espalha entre variáveis redundantes).

Qualidade dos dados: zero ausentes e zero duplicatas; as únicas colunas descartadas foram `id` e `Unnamed: 32` (coluna vazia do CSV).

**SOP (PCOS).** A base é muito menor em informação: 541 pacientes e apenas **3 biomarcadores** (beta-HCG em duas medições e AMH), com 32,7% de casos positivos. Aqui a EDA foi decisiva, porque revelou os três problemas que definem o teto de desempenho do estudo:

1. **Censura por limite de detecção.** O valor `1.99` aparece 191 vezes em beta-HCG I e **307 vezes** (mais da metade da base) em beta-HCG II. Não é um valor real: é o código do laboratório para "abaixo do limite de detecção do exame". Tratá-lo como número contaminaria qualquer estatística — a mediana de beta-HCG II é literalmente 1,99 nas duas classes. Essa descoberta virou a principal estratégia de pré-processamento (censura → ausente + flag).
2. **AMH é o sinal dominante.** Média de 4,54 ng/mL nas pacientes sem SOP contra **7,84 ng/mL** nas com SOP (medianas 3,2 vs 5,9) — consistente com a literatura clínica, em que AMH elevado se associa à SOP. Beta-HCG I também difere (mediana 13,7 vs 70,5), mas com caudas extremas (desvio padrão ≈ 3.540, máximo 32.460), o que motivou histogramas em escala log na EDA e imputação por mediana no pré-processamento.
3. **Rótulos conflitantes.** Foram identificados **10 grupos de pacientes com o perfil bioquímico idêntico e diagnósticos opostos** — em geral perfis totalmente censurados (1,99/1,99) com o mesmo AMH. Isso significa que nenhum modelo, por melhor que seja, consegue separar esses casos: é um **erro irredutível** documentado desde a EDA, e a explicação honesta para o desempenho modesto observado depois.

### 9.2 Estratégias de pré-processamento

As estratégias comuns aos dois estudos formam a espinha dorsal metodológica:

- **Encoding do alvo com a classe positiva clínica = 1** (maligno; com SOP). A escolha não é cosmética: define qual recall o pipeline inteiro otimiza e reporta.
- **Split estratificado 80/20 com semente fixa 42** (`train_test_split(..., stratify=rotulo, random_state=42)`), preservando a proporção de classes nos dois conjuntos (mama: 455/114; PCOS: 432/109).
- **Imputação por mediana + `StandardScaler`**, empacotados num `ColumnTransformer`. A mediana foi escolhida por robustez a outliers (crítico no PCOS, com caudas de beta-HCG na casa das dezenas de milhares); a padronização é exigida pela regressão logística e pelo SVM RBF, sensíveis à escala.
- **Prevenção de vazamento como decisão de arquitetura:** o pré-processador é apenas *instanciado* no notebook de pré-processamento e só é *fitado* dentro do `Pipeline(preprocessador → classificador)` no notebook de treino — ou seja, mediana e média/desvio do scaler são aprendidos exclusivamente no treino. O conjunto de teste atravessa os artefatos sem ser tocado até o notebook de avaliação, que é o único a computar métricas.
- **Desbalanceamento tratado por ponderação de classes** (`class_weight="balanced"`), não por reamostragem (sem SMOTE/undersampling) — adequado ao grau de desbalanceamento (~33–37% de positivos) e evita amostras sintéticas em dados clínicos.

Estratégias específicas por estudo:

- **Mama:** nenhuma feature foi criada nem removida (além de `id`/`Unnamed: 32`). A colinearidade identificada foi deliberadamente mantida, com a justificativa registrada de padronizar escalas sem descartar informação nesta fase.
- **PCOS:** o pré-processamento é onde o estudo ganha (ou perderia) qualidade. Em ordem: (1) **valores censurados 1,99 → `NaN`**, para a imputação tratá-los como desconhecidos em vez de números válidos; (2) **flags `beta_i_censurado`/`beta_ii_censurado`** preservando a informação de que o valor estava abaixo do limite — a *ausência* aqui é informativa (beta-HCG indetectável é um estado clínico, não falta de dado); (3) **engenharia de atributos** expandindo 3 biomarcadores em 9 features: `beta_hcg_max`, `beta_hcg_diff` e `beta_hcg_ratio` capturam a relação entre as duas medições de beta-HCG, e `amh_alto` (corte em 4,5 ng/mL) discretiza o principal biomarcador num indicador clínico; (4) descarte dos identificadores (`Sl. No`, `Patient File No.`).

### 9.3 Modelos usados e porquê

Os **mesmos cinco algoritmos** foram treinados nos dois estudos, escolhidos para cobrir famílias diferentes de hipóteses (linear, árvore única, margem não linear, ensembles de bagging e de boosting) e permitir um comparativo honesto no dashboard:

| Modelo | Por que foi incluído |
|---|---|
| **Regressão Logística** | Baseline linear probabilístico e interpretável — em triagem clínica, fornecer P(doença) calibrável e coeficientes legíveis é valioso por si só. Todo o comparativo se ancora nela. |
| **Árvore de Decisão** | Captura relações não lineares com regras inspecionáveis e fornece **importância global de atributos** nativa (usada na explicabilidade e no dashboard). Treinada sem poda, serve também de referência de overfitting de modelo único. |
| **SVM (SVC RBF calibrado)** | Fronteira de decisão não linear com `class_weight="balanced"`. Como o SVC não expõe `predict_proba`, foi envolvido em `CalibratedClassifierCV` (calibração sigmoide/Platt, CV interna de 5 folds) — decisão necessária porque o dashboard e o critério clínico dependem de probabilidades confiáveis. |
| **Gradient Boosting** | Ensemble sequencial que corrige erros dos estágios anteriores; hipótese de capturar interações finas entre atributos (ou entre biomarcadores, no PCOS). |
| **Random Forest** | Ensemble de bagging: muitas árvores (300) para reduzir a variância da árvore única, com balanceamento de classes. Tende a ser o candidato forte "sem tuning". |

Duas decisões transversais completam o desenho: **nenhum tuning de hiperparâmetros** (valores fixos/default com semente 42 — o objetivo da fase é comparar famílias de modelos com um protocolo limpo, não espremer décimos de ponto) e **critério de seleção clínico**: o "melhor modelo" é o de maior **recall da classe positiva**, porque o falso negativo (doente classificado como saudável) atrasa diagnóstico e tratamento, enquanto o falso positivo custa exames complementares — um erro recuperável. Acurácia, F1 e precisão são reportados para transparência, não para seleção.

### 9.4 Resultados e interpretação dos dados

**Câncer de mama — resultados no holdout (114 amostras: 42 malignas, 72 benignas):**

| Modelo | Acurácia | Recall (maligno) | F1 | Precisão | Falsos negativos |
|---|---|---|---|---|---|
| Regressão Logística | 96,5% | 92,9% | 95,1% | 97,5% | 3 |
| Árvore de Decisão | 93,0% | 90,5% | 90,5% | 90,5% | 4 |
| **SVM (vencedor)** | **98,2%** | **97,6%** | **97,6%** | **97,6%** | **1** |
| Gradient Boosting | 96,5% | 90,5% | 95,0% | 100% | 4 |
| Random Forest | 97,4% | 92,9% | 96,3% | 100% | 3 |

Interpretação: todos os modelos ficam acima de 90% de recall — coerente com a separabilidade vista na EDA — mas o **SVM calibrado** se destaca ao errar **um único caso maligno em 42**, com apenas 1 falso positivo. Gradient Boosting e Random Forest atingem precisão perfeita (nenhum falso alarme), porém ao custo de 3–4 malignos perdidos — exatamente o trade-off que o critério clínico resolve em favor do SVM. A explicabilidade converge com a EDA: o SHAP do SVM aponta `texture_worst`, `radius_worst`, `perimeter_worst`, `area_worst` e `concave points_worst` como os maiores contribuintes para a predição de malignidade (tamanho + irregularidade do núcleo no pior campo observado), e a importância Gini da Árvore concentra 73% em `perimeter_worst`. A leitura registrada nos notebooks é cautelosa: **importância ≠ causalidade**, e a colinearidade radius/perimeter/area espalha o crédito entre variáveis redundantes — o SHAP serve para auditar e comunicar o modelo, não para inferência biológica. No caso local demonstrado (waterfall), uma amostra benigna recebe P(maligno) = 0,07%, ilustrando o uso da explicação individual como instrumento de auditoria caso a caso.

**SOP (PCOS) — resultados no holdout (109 amostras, 36 positivas):**

| Modelo | Acurácia | Recall (PCOS) | F1 |
|---|---|---|---|
| Regressão Logística | 65,1% | 52,8% | 50,0% |
| Árvore de Decisão | 64,2% | 47,2% | 46,6% |
| SVM (calibrado) | 63,3% | 41,7% | 42,9% |
| Gradient Boosting | 64,2% | 36,1% | 40,0% |
| **Random Forest (vencedor)** | **66,1%** | **55,6%** | **52,0%** |

Interpretação: o contraste com o estudo da mama é o resultado mais instrutivo do projeto. Nenhum modelo supera em acurácia a baseline trivial "prever sempre negativo" (~67%) — e isso **não é falha de modelagem, é limite dos dados**: com apenas 3 biomarcadores, dois deles censurados na maioria dos registros, e 10 perfis bioquímicos idênticos com rótulos opostos, o teto de separabilidade é baixo. O valor dos modelos está no recall da classe positiva: o **Random Forest** identifica 55,6% das pacientes com SOP (contra 0% da baseline), que é o que importa numa triagem. A importância de atributos (Árvore de Decisão, exportada ao dashboard) confirma o AMH e seus derivados como o sinal dominante, como a EDA antecipava. A conclusão registrada nos notebooks é a correta do ponto de vista clínico: esses biomarcadores isolados são insuficientes para diagnóstico, e o caminho de melhoria é **mais dados** (o dataset completo de PCOS tem ~40 variáveis clínicas) antes de mais engenharia de modelo.

**Leitura conjunta.** Os dois estudos, com o mesmo protocolo e os mesmos cinco algoritmos, entregam o par de lições que o comparativo do dashboard materializa: quando as features carregam sinal (mama), qualquer família de modelo funciona e a disputa se decide no erro clinicamente relevante; quando não carregam (PCOS), nenhum ensemble compensa, e a honestidade metodológica — reportar a limitação em vez de mascará-la — é o resultado. Em ambos os casos vale o aviso permanente do projeto: os modelos **apoiam** a decisão; a palavra final é do profissional de saúde.

---

## 10. Contrato de dados notebooks → web

O "contrato" é a interface entre as duas metades do projeto. Por estudo, são sempre 4 arquivos:

| Arquivo | Formato | Papel |
|---|---|---|
| `amostras_comparativo.parquet` | Parquet (largo) | 1 linha por amostra de teste: features + predição/probabilidade/acerto de cada modelo + agregados (`n_modelos_acertaram`, `consenso`, `amplitude_probabilidade`) |
| `predicoes_longas.parquet` | Parquet (longo) | 1 linha por (amostra × modelo) — usado em agregações por modelo |
| `metricas_comparativo.json` / `metricas_web.json` | JSON | Métricas por modelo, matrizes de confusão (tn/fp/fn/tp), ranking por métrica, melhor modelo + critério, hiperparâmetros e justificativas, resumo do pré-processamento, versão do contrato |
| `graficos.json` / `graficos_web.json` | JSON | Dados prontos para os gráficos do navegador: inventário do dataset, splits com % por classe, histogramas por feature/split, matriz de correlação, importância de atributos |

Decisões de design relevantes:

- **O navegador nunca acessa Parquet:** a API lê os Parquet com pandas e entrega JSON paginado/filtrado.
- **Nenhum modelo é serializado para a web** — apenas predições pré-computadas. Isso elimina risco de divergência treino/serving e dispensa dependências de ML no servidor web.
- **Histogramas com bins compartilhados** entre as classes (mesmas faixas para benigno/maligno) para comparação visual honesta.
- O JSON carrega **tudo o que o frontend precisa para se explicar**: até as justificativas textuais de cada algoritmo e os hiperparâmetros reais (via `get_params()`) são exportados, alimentando o painel "Configuração dos modelos".

---

## 11. Aplicação web — backend (FastAPI)

### 11.1 `src/web/app.py`

Cria o app FastAPI (`title="Clinical Model Lab"`, versão 1.1.0), registra os dois routers de API, monta `/static` (StaticFiles) e serve duas páginas Jinja2:

| Rota | Template | Estudo |
|---|---|---|
| `GET /` | `index.html` | Câncer de mama |
| `GET /pcos` | `pcos.html` | SOP (PCOS) |

A documentação automática (Swagger) fica em `/docs`.

### 11.2 Camada de serviços (`services/`)

`data_loader.py` (mama) e `pcos_data_loader.py` (PCOS) são espelhos um do outro:

- Resolvem os caminhos do contrato do respectivo estudo sob `src/db/outputs/...`.
- Expõem `load_metrics()`, `load_charts()`, `load_samples()`, `load_predictions()`, todos com **`@lru_cache(maxsize=1)`** — os arquivos são lidos do disco uma única vez e servidos da memória.
- Se um arquivo do contrato não existe, levantam `DataUnavailableError`/`PcosDataUnavailableError` com mensagem instrutiva ("Execute o notebook 08_export_web.ipynb"), que a camada de rotas converte em **HTTP 503**. Ou seja: o dashboard sobe mesmo sem os exports e explica ao usuário o que falta.
- `clear_cache()` invalida os quatro caches — usado pelo endpoint de recarga para refletir uma nova execução dos notebooks sem reiniciar o servidor.
- `records()` converte DataFrames em registros JSON-safe (via `to_json`/`json.loads`, tratando tipos NumPy).

### 11.3 Endpoints da API

Router do estudo mama com prefixo `/api` (`api/routes.py`); router do PCOS com prefixo `/api/pcos` (`api/pcos_routes.py`), quase idêntico. Endpoints:

| Endpoint | Método | Função |
|---|---|---|
| `/api/health` | GET | Status + versão do contrato + nº de amostras (503 se export ausente) |
| `/api/metricas` | GET | JSON de métricas completo (repassa o arquivo do contrato) |
| `/api/modelos` | GET | Melhor modelo, lista de modelos ordenada (ordenação insensível a acentos via `unicodedata`), atributos |
| `/api/graficos` | GET | JSON de gráficos completo |
| `/api/amostras` | GET | Tabela comparativa paginada e filtrável (ver abaixo) |
| `/api/amostras/{id}` | GET | Detalhe de uma amostra: predição de cada modelo (rótulo, P(positivo), confiança, acerto) + os atributos clínicos (404 se não existir) |
| `/api/predicoes/resumo` | GET | *(só mama)* Distribuição das probabilidades em 10 faixas + acertos/erros por modelo, a partir do Parquet longo |
| `/api/cache/recarregar` | POST | Limpa o `lru_cache` e força releitura dos exports |

**`GET /api/amostras` em detalhe** — é o endpoint mais rico:

- **Paginação:** `page` (≥1) e `size` (0 = todos, máx 5000).
- **Filtros:** `diagnostico` (literal "Maligno"/"Benigno" ou "Com PCOS"/"Sem PCOS"), `consenso` (bool), e — exigindo `modelo` selecionado — `acerto` (bool) e, no estudo mama, faixa de probabilidade `prob_min`/`prob_max`. Filtros dependentes de modelo sem modelo selecionado retornam **400** com mensagem explicativa; modelo desconhecido também é 400.
- **Coluna computada on-the-fly:** `confianca_<modelo>` = P(classe prevista) — `prob` se a predição foi positiva, `1 − prob` caso contrário.
- **Ordenação com allowlist:** `sort` só aceita colunas conhecidas (`id`, `diagnostico_real`, `n_modelos_acertaram`, `amplitude_probabilidade`, `prob_*`, `confianca_*`); qualquer outra coluna → 400. Evita ordenação por colunas arbitrárias do DataFrame.
- **Resposta compacta:** apenas as colunas necessárias para a tabela (as 30 features ficam de fora da listagem e só aparecem no endpoint de detalhe), + metadados de paginação, lista de modelos e de atributos.

### 11.4 Padrões do backend

- Erros de dados → 503 com instrução de qual notebook executar; erros de uso da API → 400/404 com detalhe.
- Sem banco de dados, sem autenticação, sem estado mutável (exceto o cache): o servidor é um leitor tipado do contrato.
- Host padrão 127.0.0.1 (uso local/acadêmico).

---

## 12. Aplicação web — frontend

SPA-like em **JavaScript vanilla + Chart.js 4** (via CDN), sem framework nem build step. Um arquivo JS por estudo (`app.js` para mama, `pcos.js` para PCOS — mesma arquitetura, com pequenas variações), CSS único compartilhado (`app.css`, ~840 linhas, tema dark/glassmorphism com efeitos "ambient", responsivo com sidebar colapsável).

### 12.1 Estrutura das páginas

Ambas as páginas têm a mesma anatomia (seções ancoradas na sidebar):

1. **Visão geral (hero):** contadores de amostras/modelos/atributos + card do **modelo recomendado** com as 4 métricas principais (badge deixa claro o critério: "Maior recall maligno"/"Maior recall PCOS").
2. **Modelos:** gráfico de barras das métricas por modelo com **dois eixos Y** (percentuais 80–100% à esquerda; contagem de falsos negativos à direita, em rosa — reforçando o foco clínico), ranking ordenado por recall, cards por modelo com tags "Melhor/Menor <métrica>", e painel de **configuração** exibindo algoritmo, justificativa e hiperparâmetros reais vindos do contrato. *(No PCOS, há adicionalmente uma tabela de resumo consolidado com médias de probabilidade e confiança por modelo.)*
3. **Gráficos / Biomarcadores:** cards de inventário (registros brutos/usados/treino/teste), doughnut da distribuição de classes com seletor de conjunto (completo/treino/teste), histograma comparativo por feature e conjunto (gráfico de linha com área preenchida, uma série por classe), **heatmap de correlação** renderizado como grid CSS (triângulo inferior, com reimplementação em JS do colormap `coolwarm` do Matplotlib para manter consistência visual com os notebooks), matriz de confusão por modelo (grid 2×2 com células de erro destacadas) e importância de atributos (barras horizontais, top-N configurável no estudo mama).
4. **Amostras:** tabela comparativa servida por `/api/amostras` com filtros (diagnóstico real, modelo de referência, acerto/erro — habilitado só com modelo selecionado —, consenso), ordenação clicável nos cabeçalhos (delegada ao backend), paginação, e barra de probabilidade por célula colorida por acerto/erro. A coluna **Amplitude** tem tooltip que abre a P(positivo) de cada um dos 5 modelos com destaque de máx/mín — instrumento para investigar casos onde os modelos divergem. Clicar na amostra abre um **modal de detalhe** com a predição de cada modelo e todos os atributos clínicos (via `/api/amostras/{id}`).

### 12.2 Padrões de implementação do JS

- **Estado central** (`state`): métricas, lista de modelos, página/ordenação corrente e referências das instâncias Chart.js (destruídas antes de re-renderizar para evitar leaks).
- **Carga inicial:** `Promise.all` de `/api/metricas` + `/api/graficos`, depois renderização de todas as seções e primeira página de amostras. Em erro (export ausente), toast com a mensagem do backend e dica "Execute o notebook 08".
- **Interação servidor-side:** filtros/ordenação/paginação da tabela sempre refazem a query — o frontend não re-filtra em memória.
- **Segurança básica:** todo conteúdo dinâmico interpolado em HTML passa por `escapeHtml()`.
- **Botão ↻ (atualizar):** chama `POST /api/cache/recarregar` e recarrega o dashboard — fluxo para ver novos resultados após reexecutar os notebooks com o servidor de pé.
- Formatação localizada pt-BR (`toLocaleString`, ordenação `localeCompare` com sensibilidade a acentos).

---

## 13. Guia de execução completo

Pré-requisito: Python 3 instalado no Windows (o script escolhe a versão mais recente disponível).

### Caminho rápido

```powershell
# 1. Executar os notebooks (menu interativo; "T" = todos)
.\run_notebooks.ps1

# 2. Subir o dashboard (porta livre entre 8010–8013, ou -Port <n>)
.\run_web.ps1
```

Abra `http://127.0.0.1:8010` ( `/` = mama, `/pcos` = SOP). É necessário ter executado ao menos o pipeline (com o notebook de export) de cada estudo que quiser visualizar.

### Caminho manual

```powershell
# Ambiente
. .\pre_execucao.ps1
# ou: py -3 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

# Notebooks um a um (exemplo)
python run_notebook.py notebooks/breast-cancer-wisconsin-data/03_preprocessamento.ipynb

# Dashboard
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8010 --reload
```

### Ordem obrigatória dos notebooks

- **Mama:** `03 → 04 → 05 → 06 → 07 → 08` (o `02` de EDA é opcional). O dashboard `/` depende do `08_export_web.ipynb`.
- **PCOS:** `02 → 03 → 04 → 05` (o `01` de EDA é opcional). O dashboard `/pcos` depende do `05_export_web.ipynb`.

Cada notebook valida a existência do artefato do anterior; se pular uma etapa, ele falha com a instrução de qual executar.

### Verificação

- Fim de cada notebook: célula de validação imprime "Validação OK".
- API: `GET /api/health` e `GET /api/pcos/health` respondem `{"status": "ok", ...}` quando o contrato está no lugar; 503 com instrução caso contrário.
- Após reexecutar notebooks com o servidor rodando: botão ↻ no dashboard ou `POST /api/cache/recarregar`.

---

## 14. Limitações conhecidas e pontos de atenção

Registradas nos próprios notebooks ou identificadas na análise do código:

**Metodologia de ML**
- **Sem tuning de hiperparâmetros** (nenhum GridSearchCV/RandomizedSearchCV) e **sem cross-validation de avaliação** — as métricas vêm de um único holdout de 20%; com datasets pequenos (114 e 109 amostras de teste), variam com o split. A única CV é a interna da calibração do SVM.
- **Sem ROC-AUC/PR-AUC nem ajuste de threshold** (decisão em 0,5), apesar de todos os modelos exporem probabilidades — o ajuste de threshold seria a alavanca natural dado o critério de maximizar recall.
- **Colinearidade não removida** no estudo mama (21 pares com |r| > 0,9), o que dilui a interpretação das importâncias (limitação documentada no notebook 06).
- Desbalanceamento tratado apenas com `class_weight="balanced"` (e nem em todos os modelos — o Gradient Boosting não tem esse parâmetro).

**Dados**
- Mama: dataset pequeno e de população institucional específica; o pipeline trabalha com 30 features pré-calculadas, não com as imagens.
- PCOS: apenas 3 biomarcadores, com **censura pesada** (1.99 = limite de detecção em 191/307 registros de beta-HCG) e **10 grupos de rótulo conflitante** (mesmo perfil bioquímico, rótulos opostos) — teto irredutível de erro que explica o desempenho modesto (acurácia ~0,63–0,66, abaixo da baseline majoritária; o valor do modelo está no recall da classe positiva).

**Código / manutenção**
- Markdown do notebook 06 (mama) desatualizado: cita Regressão Logística como vencedora, mas a execução registrada selecionou o SVM.
- `export_pcos_web_runner.py` conta a censura do bloco `destaques_eda` por **posição de coluna** (`iloc[:, 3]`/`iloc[:, 4]`) no CSV bruto — frágil se a ordem das colunas do arquivo mudar.
- No PCOS, `amh_alto` (binária de corte fixo 4.5) passa pelo `StandardScaler` junto com as contínuas; e as flags `beta_*_censurado` tornam os beta-HCG imputados parcialmente redundantes.
- Duplicação estrutural: routers, loaders, templates e JS do estudo mama e do PCOS são quase espelhos — o próprio notebook 07 lista como próximo passo extrair código comum para módulos em `src/`.
- Os outputs gravados nos `.ipynb` do repositório vieram de execuções em outra máquina (caminhos `E:\...`); a pasta `src/db/outputs/` é gerada localmente em runtime e não é versionada.
