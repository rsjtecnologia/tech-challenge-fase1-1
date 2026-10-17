# Tech Challenge — Fase 2, Projeto 1

Implementação incremental sobre os estudos médicos da Fase 1: otimização genética dos cinco classificadores existentes, interpretação de métricas agregadas via LLM e logging para execução local. A entrega contém somente código-fonte e documentação; não inclui infraestrutura de nuvem, Kubernetes ou Docker.

> **Uso acadêmico:** resultados experimentais não são diagnóstico, recomendação médica ou validação clínica. A LLM recebe somente métricas agregadas, não dados individuais de pacientes. Não enviar dados identificáveis a provedores externos.

## Arquitetura

```text
CSVs da Fase 1
     │
     ▼
src.optimization.experiments
  ├─ reaplica transformação e split estratificado 80/20
  ├─ pipelines de imputação + escala + classificador
  ├─ GA otimiza no treino com validação cruzada estratificada
  └─ comparação final baseline/otimizado no holdout
     │
     ├── optimization_results.json + optimization.log
     │
     ├── GET /api/otimizacao/{study} ──► comparação agregada baseline × GA
     └── POST /api/interpretacao ──────► provedor compatível OpenAI Chat Completions
                                          contexto agregado, resposta JSON e rubrica
                       ▲
                       └── botões do dashboard exibem comparação e interpretação

Dashboard FastAPI local
  ├─ /healthz (liveness) e /readyz (arquivos disponíveis)
  └─ logging de requisições e latência
```

## Otimização genética

Executar na raiz do repositório com o ambiente da Fase 1:

```powershell
python -m src.optimization.cli
```

O comando padrão otimiza os dois estudos e os cinco modelos, e escreve:

- `src/db/outputs/phase2-genetic/optimization_results.json`
- `src/db/outputs/phase2-genetic/optimization.log`

Para limitar a demonstração a um estudo e modelo:

```powershell
python -m src.optimization.cli --study breast_cancer --model "Random Forest"
python -m src.optimization.cli --study pcos --model "Regressao Logistica"
```

O algoritmo representa cada hiperparâmetro como um gene discreto. A população inicial é aleatória com semente fixa; seleção por torneio, cruzamento uniforme, mutação independente por gene e elitismo de um indivíduo são implementados em `src/optimization/genetic.py`. Espaços de busca em `src/optimization/experiments.py` abrangem C/peso de classe (Regressão Logística), profundidade/folha/critério (Árvore), C/gamma/kernel (SVM), estimadores/taxa/profundidade/subamostragem (Gradient Boosting) e número/profundidade/folha/features (Random Forest).

### Três configurações obrigatórias

| Experimento | População | Gerações | Mutação por gene | CV |
|---|---:|---:|---:|---:|
| `exploracao_conservadora` | 6 | 3 | 0,15 | 3 folds |
| `exploracao_ampla` | 10 | 3 | 0,30 | 3 folds |
| `refinamento` | 8 | 5 | 0,10 | 3 folds |

O fitness maximiza `0,50 × recall + 0,30 × F1 + 0,20 × acurácia`, calculado sobre folds do **conjunto de treino**. A configuração vencedora é escolhida pelo fitness de CV. Um único holdout estratificado 80/20, com semente 42, é usado para comparar baseline e modelo otimizado depois da seleção. Reportam-se acurácia, recall, F1, precisão e acurácia balanceada. O SVM otimizado mantém calibração sigmoide e usa CV interna com três folds; o baseline preserva configuração original da Fase 1.

O resultado contém, por estudo/modelo: métricas baseline e otimizadas no teste, métricas CV da configuração otimizada, parâmetros, resultado individual de cada um dos três experimentos, convergência por geração, número de candidatos únicos avaliados e duração. Logging informa início/fim, fitness, avaliações e duração sem imprimir registros dos pacientes.

### Execução completa realizada

Foi executada a configuração padrão nos dois datasets e nos cinco classificadores. O artefato completo está em `src/db/outputs/phase2-genetic/optimization_results.json` (saída local ignorada pelo Git); o log correspondente é `src/db/outputs/phase2-genetic/optimization.log`.

| Estudo | Modelo | Experimento selecionado | Acurácia baseline → GA | Recall baseline → GA | F1 baseline → GA |
|---|---|---|---:|---:|---:|
| Mama | Regressão Logística | Exploração ampla | 0,9649 → 0,9912 | 0,9286 → 0,9762 | 0,9512 → 0,9880 |
| Mama | Árvore de Decisão | Exploração ampla | 0,9298 → 0,9123 | 0,9048 → 0,8571 | 0,9048 → 0,8780 |
| Mama | SVM | Exploração conservadora | 0,9825 → 0,9561 | 0,9762 → 0,8810 | 0,9762 → 0,9367 |
| Mama | Gradient Boosting | Exploração conservadora | 0,9649 → 0,9737 | 0,9048 → 0,9286 | 0,9500 → 0,9630 |
| Mama | Random Forest | Exploração ampla | 0,9737 → 0,9737 | 0,9286 → 0,9286 | 0,9630 → 0,9630 |
| PCOS | Regressão Logística | Exploração ampla | 0,6514 → 0,6330 | 0,5278 → 0,5833 | 0,5000 → 0,5122 |
| PCOS | Árvore de Decisão | Exploração ampla | 0,6422 → 0,6422 | 0,4722 → 0,5278 | 0,4658 → 0,4935 |
| PCOS | SVM | Exploração ampla | 0,6330 → 0,6422 | 0,4167 → 0,3056 | 0,4286 → 0,3607 |
| PCOS | Gradient Boosting | Refinamento | 0,6422 → 0,6697 | 0,3611 → 0,4722 | 0,4000 → 0,4857 |
| PCOS | Random Forest | Exploração ampla | 0,6606 → 0,6422 | 0,5556 → 0,5278 | 0,5195 → 0,4935 |

As métricas são do mesmo holdout da Fase 1 e não foram usadas para escolher hiperparâmetros nem o experimento vencedor. O resultado real não indica melhora universal: alguns estimadores perdem recall/F1 após o ajuste, especialmente o SVM do PCOS. Isso é reportado para comparação honesta — não se força um “ganho” nem se escolhe retrospectivamente o vencedor pelo teste. O dashboard agora apresenta o comparativo do holdout e o experimento selecionado em cada estudo.

**Observação de custo:** o comando sem filtros executa até três experimentos por modelo e estudo, com CV dentro do treino. É computacionalmente mais custoso do que a demonstração de um modelo; os filtros `--study` e `--model` permitem validar etapas menores antes da execução integral.

## Integração LLM e prompt engineering

A implementação usa apenas biblioteca padrão Python e suporta Ollama local ou qualquer endpoint compatível com OpenAI Chat Completions. Não há provedor externo obrigatório. **Para obter geração real por LLM sem nuvem, instale Ollama, baixe um modelo uma vez e configure a aplicação:**

```powershell
.\scripts\start_ollama.ps1
```

O script encontra o Podman, cria/inicia a máquina local se necessário, constrói a imagem `tech-challenge-ollama:local` usando [infra/ollama/Containerfile](./infra/ollama/Containerfile) sobre a imagem oficial `docker.io/ollama/ollama:latest`, inicia o container `tech-challenge-ollama`, baixa o modelo e verifica `http://127.0.0.1:11434`. Ele pré-carrega o modelo e define `LLM_PROVIDER`, `LLM_BASE_URL` e `LLM_MODEL` no terminal atual; inicie a aplicação nesse mesmo terminal com `.\run_web.ps1`. O modelo padrão é `llama3.2:latest`; pode escolher outro com `.\scripts\start_ollama.ps1 -Model "nome-do-modelo"`. Os arquivos do modelo permanecem no volume Podman `tech-challenge-ollama-data`, sem serem incluídos no Git ou na imagem do projeto. O modelo precisa caber na memória disponível.

No Windows, o Podman precisa de uma máquina Linux baseada em WSL 2 e de virtualização de hardware ativa (Intel VT-x/AMD-V, conforme o processador). Se ainda não configurou WSL, abra PowerShell como administrador, execute `wsl --install --no-distribution` e reinicie o computador antes de rodar o script. Se o hipervisor não estiver ativo, habilite também a virtualização no firmware UEFI/BIOS. O script verifica essa condição antes de tentar criar a máquina e mostra uma instrução específica; não ativa funcionalidades do Windows nem altera o firmware automaticamente. O download da imagem-base e do modelo exige acesso à internet.

Se o Ollama estiver selecionado mas indisponível, a API reporta o erro em vez de apresentar uma resposta local como se fosse saída de LLM. A instalação local do provedor não configura Docker, Kubernetes ou deploy da aplicação; é somente runtime de inferência local.

### Falha ao importar a máquina Podman no WSL

O erro `WSL import of guest OS failed` com saída 125 do Podman ocorre antes da execução do Ollama. Consulte primeiro `wsl --status`, `wsl --version` e `wsl --list --verbose`. Se retornarem erro ou somente a tela de ajuda, repare os pré-requisitos do Windows antes de tentar criar a máquina novamente.

Em PowerShell **como Administrador**, habilite os dois recursos, verificando o sucesso de cada comando:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --install --no-distribution
```

Reinicie o Windows. Depois execute `wsl --update` e `wsl --set-default-version 2` e confirme que `wsl --status` responde sem erro. Uma lista de distribuições vazia pode ser normal antes da primeira máquina Podman; falha ao executar o comando de listagem não é. Execute então `.\scripts\start_ollama.ps1` na raiz do projeto e inicie `.\run_web.ps1` no mesmo terminal.

Se o processador reporta `VirtualizationFirmwareEnabled=True`, não é necessário alterar a BIOS apenas porque `HypervisorPresent=False`: habilite os recursos do Windows e reinicie primeiro. Se o WSL continuar falhando, preserve a mensagem específica dessa etapa para diagnóstico. Não remova máquinas, distribuições WSL ou volumes para corrigir pré-requisitos ausentes, pois essa remoção pode apagar dados e modelos.

Alternativamente, configure credenciais por ambiente para um endpoint remoto compatível, sem gravá-las em arquivos versionados:

```powershell
$env:LLM_PROVIDER = "openai_compatible"
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_MODEL = "gpt-4o-mini"
$env:LLM_API_KEY = "<chave-fornecida-localmente>"
python -m src.optimization.cli --study breast_cancer --model "Random Forest" --interpret
```

Para outros servidores locais compatíveis, configure `LLM_BASE_URL` para seu endpoint e `LLM_MODEL`; a chave pode ser omitida para `localhost`/`127.0.0.1`. `LLM_PROVIDER=ollama` usa a **API nativa `/api/chat`**, sem chave e somente em endereço local; uma URL antiga terminada em `/v1` é normalizada para a raiz. Não usa Ollama Cloud. `openai_compatible` preserva Chat Completions e tem prazo padrão de 45 segundos.

### Configuração e desempenho do Ollama

Os padrões do cliente equivalem aos parâmetros de inferência da configuração C# fornecida:

| Ambiente Python | Valor padrão | Equivalente |
|---|---|---|
| `LLM_PROVIDER` | `ollama` quando selecionado pelo script | Enabled, UseCloud=false |
| `LLM_BASE_URL` | `http://localhost:11434` | BaseUrl |
| `LLM_MODEL` | `llama3.2:latest` | Model |
| `LLM_TIMEOUT_SECONDS` | `60` no cliente | TimeoutSeconds |
| `LLM_TEMPERATURE` | `0.1` | Temperature |
| `LLM_NUM_PREDICT` | `500` | NumPredict |

Use a URL pura, sem a sintaxe de link Markdown. Nenhuma chave é necessária para o Ollama local. O script de execução usa **180 segundos se `LLM_TIMEOUT_SECONDS` não estiver definido**, pois a CPU desta máquina não completou a interpretação médica em 60 segundos; um valor explícito é preservado. Para exigir o mesmo limite da aplicação C#, defina `$env:LLM_TIMEOUT_SECONDS = "60"` antes de iniciar. Timeout limita a espera, não acelera a inferência.

A API nativa recebe `options.temperature`, `options.num_predict`, `stream=false` e `keep_alive="30m"` para reutilizar o modelo em memória. A interpretação médica solicita JSON e um texto curto com contexto agregado compactado (quatro métricas em porcentagem, arredondadas a duas casas), mantendo a tabela de conferência com os resultados originais. Respostas cortadas pelo limite de tokens são erros explícitos, não JSON parcial apresentado como sucesso. Relatórios extensos de rotas podem exigir aumentar `LLM_NUM_PREDICT`. Os logs registram tempo, modelo e contagens de tokens, sem prompts ou dados individuais.

Medição local: a requisição médica completa levou cerca de 107 segundos após a mudança; uma execução posterior pela API retornou HTTP 200 em 92,66 segundos (559 tokens de entrada e 181 tokens gerados), contra cerca de 494 segundos na execução anterior (comparação observacional, não benchmark controlado). Testes com limite de 60 segundos ainda expiraram nesta CPU. O carregamento a frio levou aproximadamente 54 segundos; o script agora pré-carrega o modelo, mas isso desloca esse custo para a inicialização. `llama3.2:latest` e `llama3.2:3b` tinham o mesmo digest neste ambiente; mudar somente o nome não acelera o modelo. CPU/GPU, memória, prompt e quantidade de tokens tornam a latência diferente de outra aplicação, mesmo com os mesmos parâmetros.

`--interpret` grava `llm_interpretations.json` junto aos resultados. Sem configuração de provedor, a interpretação é local e baseada em regras; configuração inválida e falhas de um provedor selecionado são reportadas, não mascaradas por fallback. A interpretação continua experimental e requer revisão de factualidade: a rubrica automática não comprova que cada afirmação corresponde às métricas.

O painel tem um botão de interpretação em cada estudo. Com resultados da otimização disponíveis, o botão funciona mesmo sem provedor: nessa situação, apresenta uma interpretação determinística baseada em regras, identificada explicitamente como **local (sem LLM)**. Se Ollama ou outro endpoint estiver configurado, a resposta é identificada como gerada por LLM. Falhas de conexão ou respostas inválidas do provedor continuam explícitas e não são substituídas por uma resposta local. API:

```http
POST /api/interpretacao
Content-Type: application/json

{"estudo": "breast_cancer"}
```

`estudo` também aceita `pcos`. A API responde 503 se os resultados da otimização não estiverem disponíveis, 404 se o estudo não estiver no arquivo e 502 se o provedor LLM configurado falhar. A ausência de configuração LLM, por si só, não impede a interpretação local.

O painel compara o treino genético com o baseline. O endpoint `GET /api/otimizacao/breast_cancer` ou `GET /api/otimizacao/pcos` devolve métricas agregadas, hiperparâmetros selecionados e a configuração/resultados de CV das três experiências. Os endpoints retornam 503 se o JSON ainda não existir ou estiver inválido.

A seção de interpretação dos dois estudos também mostra uma tabela de conferência por modelo e experimento selecionado, com recall, precisão, F1 e acurácia do baseline e do otimizado lado a lado. A diferença é expressa em pontos percentuais (otimizado menos baseline), incluindo quedas. Esses números vêm diretamente dos resultados salvos do experimento, não da LLM. O título e a tabela ficam ocultos ao abrir a página e aparecem somente após clicar em **Gerar interpretação**, permanecendo disponíveis durante a geração e mesmo quando o provedor falha.

O prompt de sistema exige português, JSON com `summary`, `findings`, `limitations` e `disclaimer`, comparação baseada apenas nas métricas fornecidas, explicação de recall/precisão, declaração de limitações e aviso de que não substitui avaliação clínica. A interpretação local compara recall, precisão, F1 e acurácia no holdout, destaca quedas de recall para revisão experimental e recomenda validação independente, sem sugerir decisões clínicas. Ambas as saídas incluem limitações e disclaimer.

O contexto enviado ao provedor é montado explicitamente a partir de nome do estudo/classe, tamanhos dos conjuntos, fitness e métricas agregadas baseline/CV/teste; não inclui linhas individuais, nomes ou atributos de pacientes. Ele inclui também um marcador estruturado de preparação para dados textuais futuros: essa capacidade está **desativada** e só poderá receber resumos agregados desidentificados após revisão de privacidade e autorização explícita. Prontuários, notas clínicas brutas ou dados identificáveis não são enviados.

### Avaliação da interpretação

`evaluate_interpretation` aplica uma rubrica determinística de sete verificações: estrutura/campos, disclaimer de não substituição, ausência de alegações absolutas, limite de 4.000 caracteres, sinais de resposta em português, menção a métricas e presença de próximo passo de validação/revisão. A resposta, a origem (`local` ou `llm`) e a pontuação (0–7) são devolvidas à API e incluídas na saída CLI. É uma avaliação automatizada preliminar de segurança/forma, não uma validação por especialistas clínicos nem medida semântica completa de factualidade; uma etapa posterior deve avaliar amostras anonimizadas por revisores humanos.

## Saúde e logging local

Para executar localmente, inicie o FastAPI como na Fase 1. O endpoint `/healthz` verifica que a aplicação responde e `/readyz` verifica a presença dos exports dos dois estudos e dos resultados da Fase 2. O middleware registra método, caminho, status e latência por requisição; o job de otimização registra progresso, fitness, candidatos avaliados e duração em `optimization.log`.

Não foi configurado autoscaling, orquestração, contêiner ou deploy na nuvem: esta entrega será enviada como código-fonte e executada localmente. Assim, `/healthz`, `/readyz` e os logs servem para inspeção e monitoramento simples no processo local, não são uma infraestrutura de escalabilidade automática. A integração com LLM pode usar endpoint local; provedores externos são opcionais e dependem de configuração explícita.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src\optimization src\web\app.py tests
```

Os testes verificam determinismo e operadores básicos do GA, espaços/fábricas dos cinco modelos, reprodução das transformações e dimensões dos datasets, filtro de dados individuais no contexto LLM, interpretação local sem credenciais, propagação de falhas do provedor, rubrica, parse do endpoint compatível e endpoints de saúde/interpretação. O LLM é substituído por resposta simulada; nenhuma chamada real ou chave é necessária para a suíte.

## Limitações e próximos passos

- Os resultados são de uma divisão/split fixo e datasets pequenos; uma validação mais forte requer CV repetida/externa e nova avaliação independente.
- Fitness favorece recall, mas ainda inclui acurácia e F1 com pesos fixos. Os custos clínicos devem ser validados por especialistas.
- Não se ajusta threshold e não se reportam intervalos de confiança nem calibração/curvas ROC-PR.
- A rubrica automática da LLM não comprova correção clínica; revisão humana é necessária.
- Escalabilidade automática não faz parte desta entrega local. A API não tem autenticação, rate limiting ou controle de gastos; não exponha o serviço publicamente sem projetar essas proteções.
- A otimização é um job batch local; não se paraleliza automaticamente nem se executa em resposta a requisições do dashboard.
- A reutilização dos datasets da Fase 1 não converte o protótipo em ferramenta clínica validada.
