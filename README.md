# Tech Challenge — Fase 1

Projeto da **Pós-graduação IA para Devs (FIAP)** — Fase 1, Tech Challenge A.

## Sobre o projeto

Tech Challenge da **Pós IA para Devs (FIAP) — Fase 1**: construir a base de um sistema de IA para suporte ao diagnóstico em saúde feminina, com classificação de riscos via Machine Learning sobre dados médicos estruturados.

O projeto cobre o pipeline completo — exploração, pré-processamento, modelagem (múltiplos algoritmos), avaliação com métricas adequadas ao problema, explicabilidade (SHAP) e dashboard web para demonstração.

Dois estudos implementados:

| Estudo | Dataset | Problema |
|--------|---------|----------|
| Câncer de mama (Wisconsin) | [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) | Maligno vs. benigno |
| SOP (PCOS) | [PCOS Infertility](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) | Com vs. sem SOP |

> Apoio à decisão clínica — o profissional de saúde tem sempre a palavra final.

Artefatos gerados em `src/db/outputs/`; o dashboard em `src/web/` consome os exports de cada estudo.

## Fase 2 — Projeto 1

A extensão da Fase 2 implementa otimização genética dos modelos, interpretação agregada por LLM e logging para execução local. A entrega contém código-fonte e documentação; não utiliza Docker, Kubernetes ou deploy em nuvem. Consulte [FASE2_PROJETO1.md](./FASE2_PROJETO1.md) para arquitetura, execução, configuração do provedor LLM e limites do protótipo.

O [Projeto 2 da Fase 2](./FASE2_PROJETO2_ROTAS.md) adiciona otimização genética de rotas para múltiplos veículos, com prioridades, capacidade, autonomia e cadeia fria, visualização geográfica sintética offline e assistente LLM opcional para instruções, relatórios e perguntas. A página fica em `/rotas`.

Execução de demonstração:

```powershell
python -m src.optimization.cli --study breast_cancer --model "Random Forest"
```

---

## Início rápido (Windows)

Pré-requisito: [Python 3](https://www.python.org/downloads/) instalado (o script usa a versão mais recente disponível).

### 1. Executar os notebooks

```powershell
.\run_notebooks.ps1
```

Menu interativo para escolher o estudo e os notebooks. Use **`T`** para executar **todos** de uma vez (estudo ou pipeline completo) e acelerar a reprodução.

### 2. Subir o dashboard

```powershell
.\run_web.ps1
```

Prepara o ambiente, instala dependências e abre o dashboard em `http://127.0.0.1:8010` (ou na primeira porta livre entre 8010–8013). Porta customizada:

```powershell
.\run_web.ps1 -Port 8020
```

| Rota | Estudo |
|------|--------|
| `/` | Câncer de mama |
| `/pcos` | SOP (PCOS) |

> Execute os notebooks de export (`08_export_web` ou `05_export_web`) antes de abrir o dashboard, para cada estudo que quiser visualizar.

---

## Passo a passo manual

### Ambiente

Na raiz do repositório:

```powershell
# Criar e ativar o venv + instalar dependências
. .\pre_execucao.ps1
```

Ou manualmente:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Notebooks — Câncer de mama

Execute na ordem, a partir de `notebooks/breast-cancer-wisconsin-data/`:

1. `02_analise_exploratoria.ipynb` — EDA (opcional para o pipeline)
2. `03_preprocessamento.ipynb`
3. `04_modelagem_treino.ipynb`
4. `05_avaliacao_metricas.ipynb`
5. `06_explicabilidade.ipynb`
6. `07_discussao_demo.ipynb`
7. `08_export_web.ipynb` — gera dados para o dashboard

Via terminal (um notebook por vez):

```powershell
python run_notebook.py notebooks/breast-cancer-wisconsin-data/03_preprocessamento.ipynb
```

### Notebooks — SOP (PCOS)

Execute na ordem, a partir de `notebooks/polycystic-ovary-syndrome-pcos/`:

1. `01_analise_exploratoria.ipynb` — EDA (opcional)
2. `02_preprocessamento.ipynb`
3. `03_modelagem_treino.ipynb`
4. `04_avaliacao_metricas.ipynb`
5. `05_export_web.ipynb` — gera dados para o dashboard

### Dashboard web

Com o venv ativo:

```powershell
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8010 --reload
```

Documentação da API: `http://127.0.0.1:8010/docs`

---

## Estrutura

```
notebooks/          # Pipelines por estudo (.ipynb)
src/db/models/      # Datasets (CSV)
src/db/outputs/     # Artefatos gerados pelos notebooks
src/web/            # Dashboard FastAPI
run_notebooks.ps1   # Executor interativo de notebooks
run_web.ps1         # Sobe o dashboard
pre_execucao.ps1    # Prepara venv e dependências
```
