# Análise do desenvolvimento atual

**Projeto:** Tech Challenge — Fase 1  
**Levantamento:** 16/10/2026  
**Objetivo:** registrar como o sistema está estruturado e funciona hoje, com detalhes suficientes para apoiar a definição de uma nova especificação. Este documento descreve o estado encontrado no repositório; não altera nem substitui o manual técnico existente.

> **Aviso de uso:** trata-se de um projeto acadêmico de análise e demonstração. Os modelos não são dispositivos médicos validados e não substituem diagnóstico, exames ou decisão de profissionais de saúde.

## 1. Resumo executivo

O repositório contém um fluxo de ciência de dados e uma aplicação web de demonstração para dois problemas de classificação binária:

1. **Câncer de mama — Wisconsin:** classificar amostras como benignas ou malignas a partir de 30 atributos numéricos derivados de imagens de aspirado por agulha fina.
2. **Síndrome do ovário policístico (SOP/PCOS):** classificar pacientes como com ou sem PCOS a partir de três biomarcadores de fertilidade, transformados em nove atributos.

Cada estudo tem notebooks para exploração, preparação, treino e avaliação. Os resultados e as predições do conjunto de teste são exportados para arquivos locais. Uma aplicação FastAPI serve duas páginas e uma API que lê esses arquivos; **não treina nem executa inferência online**.

O projeto está mais desenvolvido como pipeline reproduzível e painel comparativo do que como produto clínico: não recebe dados de pacientes, não tem cadastro/autenticação, persistência transacional, controle de acesso, monitoramento de modelos ou integração com sistemas clínicos.

## 2. Mapa do repositório

| Caminho | Responsabilidade |
|---|---|
| `README.md` | Contexto, execução rápida e estrutura resumida. |
| `MANUAL_TECNICO.md` | Descrição extensa da arquitetura, dos notebooks, dos resultados e do dashboard. |
| `requirements.txt` | Dependências Python; usa faixas mínimas de versão, sem arquivo de lock. |
| `pre_execucao.ps1` | Detecta Python, cria/valida `.venv`, ativa o ambiente e instala dependências. |
| `run_notebook.py` | Executa um notebook usando `nbclient`, exibe progresso e salva a execução no próprio `.ipynb`. |
| `run_notebooks.ps1` | Menu interativo de reprodução dos pipelines. |
| `run_web.ps1` | Prepara o ambiente e inicia Uvicorn numa porta local livre ou informada. |
| `notebooks/breast-cancer-wisconsin-data/` | EDA opcional e pipeline de preparação, treino, avaliação, explicabilidade, discussão e export web. |
| `notebooks/polycystic-ovary-syndrome-pcos/` | EDA opcional e pipeline de preparação, treino, avaliação e export web. Contém também o runner Python do export PCOS. |
| `src/db/models/` | Arquivos de dados de entrada (CSV e XLSX). |
| `src/db/outputs/` | Artefatos gerados localmente: modelos, métricas, gráficos e contratos web; ignorados pelo Git. |
| `src/web/app.py` | Criação da aplicação FastAPI, inclusão das rotas e renderização das páginas. |
| `src/web/api/` | Endpoints de métricas, gráficos, amostras, detalhes e cache. |
| `src/web/services/` | Leitura, conversão e cache dos contratos gerados pelos notebooks. |
| `src/web/templates/` | HTML das duas páginas. |
| `src/web/static/js/` | Interações e gráficos do dashboard, em JavaScript sem framework. |
| `src/web/static/css/app.css` | Estilo compartilhado pelas páginas. |

O inventário versionado não mostra uma suíte de testes automatizados, configuração de CI, migrações ou especificação OpenAPI mantida à parte. A API gera documentação interativa pelo FastAPI em `/docs`.

## 3. Dados e objetivo dos estudos

### 3.1 Câncer de mama

- **Entrada:** `src/db/models/breast-cancer-wisconsin-data/data.csv`.
- **Dimensão documentada:** 569 amostras; 30 medidas numéricas, além de identificador, alvo e coluna vazia descartados na preparação.
- **Alvo:** `diagnosis`, com `M` convertido em 1 (classe positiva) e `B` em 0.
- **Atributos:** dez medidas celulares — raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal — representadas por média, erro padrão e valor “worst”.
- **Distribuição documentada:** 357 benignas e 212 malignas. A divisão estratificada 80/20 gera 455 amostras de treino e 114 de teste.
- **Ponto de interpretação:** o pipeline recebe variáveis tabulares pré-calculadas; não processa imagens.

### 3.2 SOP/PCOS

- **Entrada usada:** `src/db/models/polycystic-ovary-syndrome-pcos/PCOS_infertility.csv`.
- **Entrada adicional presente, mas não usada no pipeline descrito:** `PCOS_data_without_infertility.xlsx`.
- **Dimensão documentada:** 541 pacientes e três biomarcadores originais: duas medições de beta-HCG e AMH. Identificadores não são usados como atributos.
- **Alvo:** PCOS (Y/N), com a classe “com PCOS” como positiva.
- **Distribuição documentada:** 177 positivas e 364 negativas. O split estratificado 80/20 gera 432 linhas de treino e 109 de teste.
- **Limitação de dados registrada:** `1.99` representa valores de beta-HCG abaixo do limite de detecção; foram contadas 191 ocorrências na primeira medição e 307 na segunda. Também foram encontrados dez grupos de perfis bioquímicos idênticos com rótulos opostos.
- **Ponto de interpretação:** o modelo usa poucos biomarcadores e não representa avaliação clínica completa da SOP.

## 4. Fluxo de processamento e artefatos

O fluxo é separado por estudo e depende de arquivos intermediários no diretório `src/db/outputs/`. A ordem de execução é relevante: cada etapa consome saídas produzidas pela anterior.

### 4.1 Câncer de mama

| Etapa | Notebook | Trabalho principal | Saída/consumidor |
|---|---|---|---|
| EDA (opcional) | `02_analise_exploratoria.ipynb` | Estatísticas, qualidade, distribuição do alvo, histogramas, boxplots e correlações. | Figuras e métricas exploratórias. |
| Preparação | `03_preprocessamento.ipynb` | Limpeza, codificação, análise de correlação, split, definição do pré-processador. | `03-preprocessamento/artifacts/dados_preprocessados.joblib`. |
| Treino | `04_modelagem_treino.ipynb` | Ajuste dos cinco modelos somente com treino. | `04-modelagem-treino/artifacts/modelos_treinados.joblib`. |
| Avaliação | `05_avaliacao_metricas.ipynb` | Predições no teste, métricas e matrizes de confusão; escolhe maior recall maligno. | `05-avaliacao-metricas/metrics/metricas_teste.json` e joblib atualizado. |
| Explicabilidade | `06_explicabilidade.ipynb` | Importância da árvore e explicações globais/locais SHAP do melhor modelo. | Figuras de importância e SHAP. |
| Discussão/demo | `07_discussao_demo.ipynb` | Consolidação e demonstração de predição de um exemplo. | Material explicativo do estudo. |
| Export web | `08_export_web.ipynb` | Montagem de predições, métricas, metadados e dados para gráficos. | Contrato consumido pela rota `/`. |

### 4.2 SOP/PCOS

| Etapa | Notebook | Trabalho principal | Saída/consumidor |
|---|---|---|---|
| EDA (opcional) | `01_analise_exploratoria.ipynb` | Qualidade e distribuição, censura dos biomarcadores, diferenças por classe e perfis com rótulos conflitantes. | Figuras e resumo exploratório. |
| Preparação | `02_preprocessamento.ipynb` | Converte censura em ausente, cria atributos, split e define o pré-processador. | `02-preprocessamento/artifacts/dados_preprocessados.joblib`. |
| Treino | `03_modelagem_treino.ipynb` | Ajuste dos mesmos cinco tipos de modelo, apenas no treino. | `03-modelagem-treino/artifacts/modelos_treinados.joblib`. |
| Avaliação | `04_avaliacao_metricas.ipynb` | Métricas e matrizes de confusão; escolhe maior recall PCOS. | `04-avaliacao-metricas/metrics/metricas_teste.json` e joblib atualizado. |
| Export web | `05_export_web.ipynb` | Invoca `export_pcos_web_runner.py` para montar o contrato do dashboard. | Contrato consumido pela rota `/pcos`. |

### 4.3 Padrões comuns

- Cada notebook resolve a raiz do repositório e cria subpastas por etapa com `figures/`, `metrics/` e `artifacts/`.
- As etapas de preparação criam um split estratificado com `random_state=42`; a EDA é opcional para a reprodução do pipeline de modelagem.
- O pré-processador combina imputação pela mediana e `StandardScaler` num `ColumnTransformer`. Ele é incluído no `Pipeline` do estimador e ajustado no conjunto de treino, evitando ajustar imputação e escala usando o conjunto de teste.
- Os cinco modelos são Regressão Logística, Árvore de Decisão, SVM com calibração sigmoide, Gradient Boosting e Random Forest. Os valores de configuração são fixos/default, sem busca sistemática de hiperparâmetros.
- As classes positivas e as métricas priorizadas são maligno/recall maligno no primeiro estudo e com PCOS/recall PCOS no segundo. A intenção declarada é reduzir falsos negativos.
- A serialização intermediária usa joblib. O consumo web usa JSON para métricas e gráficos e Parquet para amostras e predições.
- Os notebooks contêm validações com `assert`; o executor persiste resultados e outputs no próprio arquivo `.ipynb`.

## 5. Resultados registrados nos artefatos locais

Os números abaixo foram lidos dos JSONs em `src/db/outputs/` presentes no ambiente durante este levantamento. Essa pasta está no `.gitignore`, portanto esses arquivos não são a fonte versionada e podem variar após nova execução.

### 5.1 Câncer de mama — teste com 114 casos

| Modelo | Acurácia | Recall maligno | F1 maligno | Precisão maligno |
|---|---:|---:|---:|---:|
| Regressão Logística | 0,9649 | 0,9286 | 0,9512 | 0,9750 |
| Árvore de Decisão | 0,9298 | 0,9048 | 0,9048 | 0,9048 |
| SVM | 0,9825 | 0,9762 | 0,9762 | 0,9762 |
| Gradient Boosting | 0,9649 | 0,9048 | 0,9500 | 1,0000 |
| Random Forest | 0,9737 | 0,9286 | 0,9630 | 1,0000 |

O SVM tem o maior recall maligno entre os resultados observados e é selecionado como melhor modelo segundo esse critério.

### 5.2 SOP/PCOS — teste com 109 casos

| Modelo | Acurácia | Recall PCOS | F1 PCOS | Precisão PCOS |
|---|---:|---:|---:|---:|
| Regressão Logística | 0,6514 | 0,5278 | 0,5000 | 0,4750 |
| Árvore de Decisão | 0,6422 | 0,4722 | 0,4658 | 0,4595 |
| SVM | 0,6330 | 0,4167 | 0,4286 | 0,4412 |
| Gradient Boosting | 0,6422 | 0,3611 | 0,4000 | 0,4483 |
| Random Forest | 0,6606 | 0,5556 | 0,5195 | 0,4878 |

O Random Forest é selecionado pelo maior recall PCOS observado. A acurácia desse estudo fica abaixo da proporção da classe negativa (aproximadamente 0,673), o que deve ser explicitado ao interpretar o valor dos resultados.

## 6. Contrato entre notebooks e dashboard

O dashboard depende de quatro arquivos por estudo, gerados pelos notebooks de export:

| Tipo | Nome | Uso |
|---|---|---|
| Parquet largo | `amostras_comparativo.parquet` | Uma linha por amostra de teste, atributos, rótulo real, predição/probabilidade/acerto por modelo e agregados de consenso/divergência. |
| Parquet longo | `predicoes_longas.parquet` | Uma linha por combinação de amostra e modelo; usado em resumos de predição. |
| JSON de métricas | Mama: `metricas_comparativo.json`; PCOS: `metricas_web.json`. | Melhor modelo, critério, ranking, métricas, configuração e atributos. |
| JSON de gráficos | Mama: `graficos.json`; PCOS: `graficos_web.json`. | Inventário, divisões de dados, histogramas, correlação, importância de atributos e destaques da EDA. |

O notebook da mama declara contrato versão 2; o export PCOS declara versão 1. O backend atual lê os arquivos e os repassa para a aplicação; não há indicação de uma validação central de esquema/compatibilidade entre versões antes de servir todos os endpoints.

Consequência operacional: o dashboard pode iniciar sem os arquivos, mas endpoints que os leem respondem HTTP 503 e instruem a execução do notebook de export correspondente. Após regenerar os arquivos com o servidor ativo, é necessário limpar o cache usando `POST /api/cache/recarregar` ou `POST /api/pcos/cache/recarregar`.

## 7. Aplicação web

### 7.1 Backend e rotas

`src/web/app.py` cria o FastAPI, inclui os dois routers, monta `/static` e serve templates com Jinja2:

- `GET /` — página de câncer de mama.
- `GET /pcos` — página PCOS.
- `GET /docs` — documentação Swagger gerada pelo FastAPI.

Os loaders localizam os contratos sob `src/db/outputs/`, leem JSON/Parquet com pandas e mantêm cada conjunto carregado em cache `lru_cache(maxsize=1)`. Erros de contrato ausente são convertidos em 503.

| Grupo | Endpoints implementados |
|---|---|
| Mama (`/api`) | `/health`, `/metricas`, `/modelos`, `/graficos`, `/amostras`, `/amostras/{sample_id}`, `/predicoes/resumo`, `/cache/recarregar`. |
| PCOS (`/api/pcos`) | `/health`, `/metricas`, `/modelos`, `/graficos`, `/amostras`, `/amostras/{sample_id}`, `/cache/recarregar`. |

As rotas de amostras suportam paginação, filtro por diagnóstico, consenso e acerto mediante seleção de modelo, ordenação controlada por allowlist e resposta JSON compacta. A rota da mama também permite faixa de probabilidade e resumo de distribuição/acertos no Parquet longo. Entradas inválidas retornam 400, amostras inexistentes retornam 404 e dados/exports ausentes retornam 503.

### 7.2 Frontend

O frontend é HTML/Jinja, CSS compartilhado, JavaScript vanilla e Chart.js carregado por CDN; não há etapa de build de frontend. Cada estudo possui página e script próprios. As páginas oferecem:

- resumo do dataset e do modelo selecionado pelo critério de recall;
- comparação/ranking de métricas e configuração dos modelos;
- visualização de distribuição, histogramas, correlação, matrizes de confusão e importância de atributos;
- tabela comparativa de amostras com filtros, ordenação e paginação;
- modal de detalhe com atributos e previsões dos modelos;
- recarga de cache após uma nova exportação.

O backend hospeda a página localmente em `127.0.0.1`; o uso de CDN/fontes externas significa que alguns elementos visuais podem depender de acesso à rede.

## 8. Ambiente e execução

O caminho recomendado é Windows/PowerShell:

1. `.\run_notebooks.ps1` para selecionar estudo/notebooks e executar o pipeline. A ordem é Mama `03 → 04 → 05 → 06 → 07 → 08` e PCOS `02 → 03 → 04 → 05`; EDA fica fora da sequência obrigatória.
2. `.\run_web.ps1` para iniciar o dashboard; sem `-Port`, escolhe a primeira porta livre entre 8010 e 8013. A opção `-Port` solicita uma porta específica.
3. Abrir a URL exibida pelo script. O contrato de export do estudo desejado precisa existir para a página exibir dados.

`pre_execucao.ps1` detecta Python 3 disponível, compara a versão do `.venv` e instala `requirements.txt`. Pode remover e recriar o próprio `.venv` quando estiver inválido ou criado com outra versão; se o ambiente divergente estiver ativo, interrompe com instruções. `run_notebook.py` não define timeout e altera o notebook executado ao gravar outputs no mesmo arquivo.

Dependências principais: pandas, NumPy, scikit-learn, matplotlib, seaborn, SHAP, Jupyter/nbclient, pyarrow, FastAPI, Uvicorn e Jinja2. A aplicação é executada com Uvicorn e `--reload` nos scripts de desenvolvimento.

## 9. Decisões, limitações e riscos técnicos/metodológicos

Esta seção destaca aspectos que afetam decisões para a próxima especificação; não representa uma solicitação para alterar a implementação atual.

### 9.1 Metodologia e validade

- **Seleção do melhor modelo no teste:** o fluxo compara os modelos no conjunto de teste e seleciona o vencedor pelo maior recall calculado nesse mesmo conjunto. Esse conjunto deixa de ser uma avaliação independente do modelo escolhido; as métricas do vencedor podem estar otimistas por seleção. Uma nova especificação deveria definir validação para seleção (por exemplo, validação cruzada no treino) e reservar avaliação final independente.
- **Uma única divisão 80/20:** não há CV de avaliação nem estimativa de variabilidade dos resultados. Os conjuntos de teste são pequenos, sobretudo PCOS (109 casos).
- **Sem ajuste de limiar documentado:** o foco é recall, mas não há otimização de threshold; decisões de classe usam o limiar padrão dos estimadores/calibrador.
- **Acurácia isolada não serve como indicador suficiente:** no PCOS, a acurácia é inferior à baseline majoritária. Deve-se comunicar recall, precisão, F1 e custos de falsos positivos/negativos em conjunto.
- **Dados observacionais e reduzidos:** alto resultado de benchmark no Wisconsin não demonstra generalização clínica. A base PCOS com três marcadores tem informação limitada e conflitos de rótulo.
- **Explicabilidade não é causalidade:** SHAP e importância de árvore descrevem o comportamento do modelo sob os dados apresentados, não causas clínicas.

### 9.2 Engenharia e operação

- Não existe serviço de predição de novas entradas: todas as amostras exibidas são pré-calculadas a partir do teste.
- Os contratos são arquivos locais ignorados pelo Git; uma instalação limpa precisa reproduzir notebooks antes de visualizar os dados. Não há armazenamento compartilhado ou coordenação entre instâncias.
- O cache é em memória do processo. O endpoint de recarga não autentica o chamador; adequado apenas ao uso local atual, não a uma exposição pública sem controle adicional.
- Não há autenticação, autorização, auditoria de acesso, perfis de usuário, política de retenção ou fluxo de consentimento.
- `requirements.txt` usa versões mínimas, sem lockfile; execuções em datas/ambientes diferentes podem instalar combinações distintas.
- Os notebooks são executáveis e contêm código, outputs e gráficos; executar altera arquivos versionados. Isso pode misturar resultados da máquina com mudanças de código.
- Os dois estudos têm duplicação em loaders, rotas, páginas e scripts. Para nova especificação, decidir se a duplicação permanece intencional ou se haverá uma camada compartilhada.
- O `run_notebooks.ps1` contém a lista e a ordem dos notebooks manualmente; adicionar/remover uma etapa implica sincronizar o script e a documentação.

### 9.3 Pontos específicos registrados

- A EDA PCOS identifica `1.99` como valor censurado e a preparação converte ocorrências em ausentes, imputadas no pipeline, mantendo flags de censura e atributos derivados. Essa regra depende da interpretação do dataset e não deve ser generalizada sem validação da origem dos dados.
- O PCOS gera `amh_alto` com corte fixo 4.5 e padroniza as features no mesmo pré-processador; qualquer nova especificação clínica deve confirmar esse limiar e a transformação.
- O runner `export_pcos_web_runner.py` foi descrito no manual técnico como contando censura por posição de coluna (`iloc`), o que é frágil a alterações na ordem das colunas. Validar esse ponto antes de reutilizar ou estender o export.
- O manual técnico registra texto de interpretação desatualizado em uma seção do notebook de explicabilidade da mama. Outputs e comentários de notebooks devem ser reconciliados com o código e as métricas atuais.
- `joblib` é adequado para artefatos locais controlados, mas não deve ser tratado como formato seguro para carregar arquivos recebidos de usuários ou fontes não confiáveis.

## 10. Estado observado do workspace

Durante o levantamento, o Git indicou alterações locais não commitadas em dez notebooks:

- Mama: `03_preprocessamento.ipynb`, `04_modelagem_treino.ipynb`, `05_avaliacao_metricas.ipynb`, `06_explicabilidade.ipynb`, `07_discussao_demo.ipynb` e `08_export_web.ipynb`.
- PCOS: `02_preprocessamento.ipynb`, `03_modelagem_treino.ipynb`, `04_avaliacao_metricas.ipynb` e `05_export_web.ipynb`.

Essas alterações foram preservadas. O executor de notebooks salva outputs no arquivo executado; portanto, a condição local dos notebooks pode refletir trabalho em andamento e não deve ser confundida automaticamente com uma versão aprovada/estável. A pasta `src/db/outputs/` continha artefatos de ambos os estudos no momento da inspeção, mas está ignorada pelo Git.

## 11. Base recomendada para escrever a próxima especificação

Antes de iniciar outra implementação, registrar explicitamente:

1. **Escopo do novo sistema:** reutiliza os dois estudos, substitui um deles ou adiciona domínio/dataset novo? Mantém o dashboard apenas demonstrativo ou requer predição de novas entradas?
2. **Público e uso pretendido:** trabalho acadêmico, protótipo interno ou uso assistencial? Quem pode interpretar resultados e quais avisos são obrigatórios?
3. **Dados:** fonte, autorização/licença, população, esquema, qualidade, valores ausentes/censurados, classes, volume, atualização e proteção de dados.
4. **Saída clínica:** classe positiva, custos relativos de erro, métricas mínimas, limiar, calibração, baseline e formato de explicação.
5. **Validação:** separação de treino/validação/teste, seleção de modelos, validação cruzada, validação temporal/externa, incerteza e critério de aceite. Não selecionar e reportar o vencedor no mesmo teste final.
6. **Fluxo de dados/artefatos:** se haverá treino batch, inferência online, contrato JSON/Parquet, versionamento, compatibilidade, persistência e forma de atualização.
7. **Aplicação:** páginas, formulários, APIs, perfis, autenticação, acessibilidade, idiomas, dispositivos e conectividade necessária.
8. **Operação e segurança:** local ou hospedado, rede, gestão de segredos, logs, auditoria, disponibilidade, backup, controle de acesso, retenção e resposta a incidentes.
9. **Qualidade e entrega:** testes unitários/de integração, validação do contrato, dados sintéticos, lint/type-check, CI, documentação, critérios de aceite e procedimento de reprodução.
10. **Migração e compatibilidade:** quais dados, modelos, métricas, rotas e comportamento atual precisam permanecer; quais podem ser removidos ou renomeados.

## 12. Referências internas

- Guia de uso: [README.md](./README.md).
- Manual técnico detalhado existente: [MANUAL_TECNICO.md](./MANUAL_TECNICO.md).
- Código do backend: `src/web/app.py`, `src/web/api/` e `src/web/services/`.
- Código de apresentação: `src/web/templates/` e `src/web/static/`.
- Pipelines: `notebooks/breast-cancer-wisconsin-data/` e `notebooks/polycystic-ovary-syndrome-pcos/`.

