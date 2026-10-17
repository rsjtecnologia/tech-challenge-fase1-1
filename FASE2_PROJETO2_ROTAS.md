# Tech Challenge — Fase 2, Projeto 2

## Visão geral

Nova funcionalidade local de roteamento para distribuição simulada de medicamentos e insumos hospitalares. Implementa um problema de roteamento de veículos (VRP) a partir de uma rota gigante representada geneticamente, com múltiplos veículos e restrições logísticas. A tela `/rotas` mostra o cenário, as rotas por veículo, uma projeção geográfica offline, o desempenho do algoritmo e um assistente LLM opcional.

> **Importante:** todos os destinos, coordenadas, cargas, prazos e recursos são fictícios e apenas demonstrativos. Distâncias são aproximações Haversine em linha reta; não representam estradas, trânsito, pacientes ou uma operação real. Confirmar sempre rotas e prioridades com a coordenação. Não é necessário nem apropriado inserir dados pessoais ou clínicos no protótipo.

## Arquitetura e organização

```text
src/logistics/scenario.py          cenário sintético tipado (depósito, frota, entregas)
          │
          ├── src/logistics/genetic.py  cromossomo, decoder VRP, operadores e fitness
          │          │
          │          └── src/logistics/service.py  persistência atômica do último plano
          │                    └── route_plan_history.json  histórico agregado local
          │                    │
          │                    └── POST /api/rotas/otimizar
          │                         GET /api/rotas/resultado
          │                         GET /api/rotas/historico
          │
          ├── src/logistics/api.py  endpoints de cenário, otimização e assistente
          ├── src/logistics/llm.py  prompts fundamentados em dados da rota
          └── src/web/templates/routes.html + static/js/routes.js
                     └── GET /rotas, SVG geográfico local, quadro de rotas e LLM

src/db/outputs/phase2-logistics/latest_route_plan.json
          └── JSON gerado localmente e ignorado pelo Git
src/db/outputs/phase2-logistics/route_plan_history.json
          └── métricas agregadas das últimas 500 simulações locais; ignorado pelo Git
```

## Cenário

`DEMO_SCENARIO`, em `src/logistics/scenario.py`, contém um depósito central, cinco veículos (dois refrigerados), dezoito entregas fictícias distribuídas por diferentes regiões da cidade e configurações explícitas:

- capacidade máxima por veículo (kg);
- autonomia máxima por percurso (km), considerando ida e volta ao depósito;
- velocidade média configurada para estimativa de chegada;
- suporte a refrigeração; apenas veículos refrigerados podem transportar entregas que a exigem;
- demanda, prioridade (`critica`, `alta`, `regular`), horário-limite e duração de serviço por entrega.

O cenário abrange uma área geográfica maior, com pontos ilustrativos em regiões Norte, Sul, Leste, Oeste e Central da cidade de São Paulo. As coordenadas não identificam endereços nem unidades reais; cargas, destinos, frota e horários são fictícios. A tela e a API expõem esse aviso. O identificador do cenário foi versionado para que planos locais da versão anterior sejam regenerados automaticamente.

## Histórico para relatórios comparativos

Cada execução bem-sucedida de `POST /api/rotas/otimizar` salva, além do plano mais recente, um registro agregado em `src/db/outputs/phase2-logistics/route_plan_history.json`. O histórico local mantém no máximo 500 execuções e contém somente data/hora, cenário, semente/parâmetros do algoritmo, distâncias baseline/otimizada, minutos agregados estimados dos veículos (deslocamento em linha reta / velocidade média + atendimento), frota utilizada, carga e utilização de capacidade, entregas servidas/atrasadas, atraso ponderado, economia percentual estimada e duração da busca. Os minutos agregam o trabalho estimado de cada veículo, não o tempo de relógio do conjunto. Não guarda coordenadas individuais, nomes de destino, itens, nem dados de pacientes. Os arquivos ficam em outputs ignorados pelo Git.

`GET /api/rotas/historico` apresenta médias e execuções registradas nos últimos sete dias. A tela `/rotas` permite definir a semente antes de cada execução; use sementes diferentes para comparar variações. Um plano atual mais antigo, gerado pela versão anterior a este histórico, pode ser importado uma única vez como registro legado; sua data original é preservada. O relatório semanal usa esses registros de simulação, não um dataset de operações reais. Com menos de duas execuções, informa insuficiência de amostra e não declara tendência. Repetir a otimização com sementes/parâmetros diferentes cria comparações do comportamento do algoritmo — incluindo distância, minutos estimados de veículo, atrasos e utilização de capacidade; não transforma execuções no mesmo dia em uma semana de entregas. Para avaliar tendências operacionais, seria necessário coletar e identificar separadamente dados reais autorizados.

## Algoritmo genético e restrições

- **Cromossomo:** permutação de IDs de entregas, uma ocorrência por entrega. O decodificador transforma a ordem numa solução com várias rotas.
- **Inicialização:** inclui uma solução base determinística e permutações aleatórias.
- **Seleção:** torneio.
- **Crossover:** Order Crossover (OX), preservando um segmento de um pai e a ordem relativa do outro.
- **Mutação:** inversão de um segmento aleatório da rota.
- **Elitismo:** melhor indivíduo da geração é mantido.
- **Reprodutibilidade:** semente configurável; padrão 42.
- **Decodificação VRP:** para cada entrega, avalia posições de inserção em cada veículo elegível. Elimina candidatos que violam capacidade, cadeia fria ou autonomia e escolhe a inserção de menor custo marginal. A prioridade não é apenas um rótulo: atraso de entrega crítica pesa 8×, alta 3× e regular 1× no fitness. O baseline também tenta alocar primeiro itens refrigerados e, em seguida, maior prioridade e prazo.
- **Baseline:** alocação gulosa de referência, priorizando cadeia fria, prioridade e prazo; mesma função de avaliação e todas as entregas atendidas.
- **Fitness minimizado:** `distância total em km + soma(atraso em minutos × peso da prioridade) + 1.000.000 × entregas não atendidas`. A autonomia/capacidade são restrições duras do decoder, não penalidades negociáveis.
- **Distância:** fórmula Haversine com raio médio terrestre. Não é roteamento viário; falta matriz de trânsito/distâncias de ruas e direção de vias.
- **Chegada:** distância em linha reta dividida pela velocidade média do veículo, adicionando duração de atendimento das paradas anteriores. É uma estimativa simplificada.

Os parâmetros da busca podem ser enviados em `POST /api/rotas/otimizar`:

```json
{
  "population_size": 64,
  "generations": 160,
  "mutation_rate": 0.18,
  "tournament_size": 4,
  "seed": 42
}
```

Faixas validadas: população 4–300; gerações 1–500; mutação 0–1; torneio de 2 a população. Resposta contém configuração, convergência, solução base e otimizada, distâncias, atrasos, carga/autonomia restante, horário estimado, entregas atendidas, coordenadas e economia percentual estimada contra o baseline. O plano é salvo com escrita atômica no diretório de outputs local.

### Resultado de uma execução de referência

Uma execução local com a configuração padrão (população 64, 160 gerações, mutação 0,18, torneio 4 e semente 42) avaliou 9.730 cromossomos em aproximadamente 101,4 segundos no cenário ampliado:

| Métrica | Baseline guloso | Algoritmo genético |
|---|---:|---:|
| Distância estimada | 314,161 km | 228,881 km |
| Entregas atendidas | 18/18 | 18/18 |
| Entregas atrasadas | 0 | 0 |
| Atraso ponderado | 0 | 0 |

A diferença estimada foi de 85,280 km (27,15%) contra o baseline desta simulação. As rotas otimizadas ficaram dentro da autonomia configurada e atenderam às restrições de carga e refrigeração. A comparação entre heurísticas usa distâncias em linha reta — não representa economia realizada, garantia de desempenho em outras instâncias ou distância rodoviária. Os resultados da versão de 12 entregas não são comparáveis. Os resultados completos são gerados localmente em `src/db/outputs/phase2-logistics/latest_route_plan.json` e esse artefato não é versionado; use o endpoint ou a página `/rotas` para reproduzir a execução.

## Execução e API

Com o ambiente virtual e dependências da Fase 1 ativos:

```powershell
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8010 --reload
```

Abra `http://127.0.0.1:8010/rotas`. A página carrega o plano salvo ou executa uma otimização inicial. O botão de recarga dispara uma nova busca com a semente exibida na tela; ao variá-la, cada execução bem-sucedida entra no histórico. O menu das duas páginas de modelos também liga ao novo módulo.

| Endpoint | Método | Finalidade |
|---|---|---|
| `/api/rotas/cenario` | GET | Dados e limites do cenário ilustrativo. |
| `/api/rotas/otimizar` | POST | Executar o algoritmo genético; aceita configuração opcional. |
| `/api/rotas/resultado` | GET | Obter o último plano de rota salvo. |
| `/api/rotas/historico` | GET | Médias e registros agregados das simulações nos últimos sete dias. |
| `/api/rotas/assistente` | POST | Instruções, relatório ou pergunta com assistente local e LLM opcional. |
| `/rotas` | GET | Página de visualização do mapa/rotas e assistente. |

Exemplos de API:

```http
POST /api/rotas/otimizar
Content-Type: application/json

{"population_size": 32, "generations": 80, "mutation_rate": 0.2, "tournament_size": 4, "seed": 7}
```

```http
POST /api/rotas/assistente
Content-Type: application/json

{"action": "question", "question": "Qual veículo transporta os itens refrigerados?"}
```

Valores de `action`: `instructions`, `daily_report`, `weekly_report` e `question`. A pergunta é opcional nos demais modos e limitada a 1.000 caracteres. O relatório diário resume o plano atual. O semanal resume execuções registradas nos últimos sete dias e compara médias/variação somente quando há pelo menos duas simulações. Nenhum relatório alega entregas reais, economia realizada ou tendência operacional; as sugestões indicam próximos testes/medições baseados nos padrões observados ou na falta de dados.

### Mapa

O mapa é um SVG projetado diretamente no navegador a partir das coordenadas; não usa serviço de mapas, tiles ou chaves. Linhas coloridas indicam rotas, numeração indica ordem de paradas, marcador dourado indica depósito, legenda identifica veículos. A opção da tela alterna a solução genética e o baseline. O esquema é geográfico ilustrativo e não segue ruas.

## LLM e prompt engineering

O assistente usa a mesma configuração OpenAI-compatible Chat Completions da funcionalidade do Projeto 1. Por padrão, não há requisição externa. Configure via ambiente, sem guardar credenciais no código:

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_MODEL = "gpt-4o-mini"
$env:LLM_API_KEY = "<chave-configurada-localmente>"
```

Um servidor local compatível pode ser usado por `LLM_BASE_URL` e `LLM_MODEL`, omitindo chave em loopback. Sem configuração, as ações do assistente continuam disponíveis por um modo local determinístico: instruções e relatórios são montados a partir dos campos do plano; perguntas recebem fatos do plano e limitações conhecidas. A resposta indica se veio do modo local ou da LLM. Esse modo não é generativo nem substitui a LLM exigida para demonstrar geração por modelo; para isso, configure as variáveis acima. Falhas de conexão ou respostas inválidas do provedor configurado são informadas como HTTP 502.

O prompt compartilhado:

1. delimita que o contexto é cenário sintético acadêmico;
2. instrui a usar somente fatos fornecidos;
3. proíbe inventar trânsito, endereços, itens, custos ou fatos; lembra que a distância é em linha reta;
4. exige texto operacional claro em português, sem orientação clínica;
5. exige JSON por tarefa, validado no backend antes de retornar;
6. pede instruções por veículo/parada, relatórios com limitações explícitas ou respostas com fatos que as sustentam e informação ausente.

O contexto LLM inclui rotas, restrições, estimativas, comparativo agregado e resumo das métricas das simulações recentes, quando disponíveis. Não inclui dados pessoais, nomes/contatos de pacientes ou qualquer campo clínico. O prompt exige que a LLM diferencie execução do algoritmo de operação real, use apenas os registros fornecidos e não invente tendências/causas. No futuro, se o cenário aceitar informações reais, será necessária revisão de privacidade, acesso e política de dados antes de enviar contexto a provedor externo.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes da nova funcionalidade validam determinismo, cromossomo/permutação, execução genética, entrega de todas as encomendas, limites de carga/autonomia/refrigeração, persistência, histórico e média de execuções, filtro de sete dias, endpoints, parâmetros inválidos, instruções/relatórios/perguntas e tratamento de resposta LLM simulada. Não requerem rede ou chave LLM.

## Limitações e evolução

- Para operação real: substituir o cenário demonstrativo por dados autorizados e revisar prioridades/regras com responsáveis operacionais e clínicos.
- Usar grafo/matriz de distâncias rodoviárias, tráfego, janelas de tempo reais, horário de motorista, disponibilidade e retorno entre centros.
- Formalizar incompatibilidades de produtos, temperatura, volumes e compartimentos, jornada máxima, pausas e falhas de entrega.
- Crescer para múltiplos depósitos/turnos/veículos com benchmarking e mais métodos de comparação; a busca atual é estocástica e não garante ótimo global.
- O greedy decoder encontra uma alocação viável para este cenário, mas não enumera toda partição possível; se uma nova instância não puder ser decodificada, a otimização falha explicitamente.
- A “distância evitada” compara uma só solução genética a um baseline heurístico com a mesma demanda; não equivale a economia financeira ou operacional observada.
- O mapa offline não fornece navegação turn-by-turn.
- A LLM é opcional; conteúdo gerado deve ser revisado pela coordenação e não pode alterar automaticamente plano, prioridades ou restrições.
