# Model Lab — dashboard web

Dashboard FastAPI com duas visões independentes:

| Rota | Estudo | Export necessário |
|------|--------|-------------------|
| `/` | Câncer de mama | `notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb` |
| `/pcos` | SOP (PCOS) | `notebooks/polycystic-ovary-syndrome-pcos/05_export_web.ipynb` |
| `/rotas` | Otimização de rotas sintéticas (Fase 2 — Projeto 2) | Cenário demo definido em `src/logistics/scenario.py` |

Use o menu **Estudo** na sidebar para alternar entre as páginas.

## Executar

1. Execute os notebooks de export de cada estudo que deseja visualizar.
2. Na raiz do repositório:

   ```powershell
   .\run_web.ps1
   ```

O script prepara o ambiente virtual, instala dependências e usa a primeira porta livre entre `8010`–`8013`.

```powershell
.\run_web.ps1 -Port 8020
```

## API

- Mama: `/api/metricas`, `/api/graficos`, `/api/amostras`
- PCOS: `/api/pcos/metricas`, `/api/pcos/graficos`, `/api/pcos/amostras`
- Fase 2: `GET /api/otimizacao/breast_cancer`, `GET /api/otimizacao/pcos`
- Interpretação agregada via LLM: `POST /api/interpretacao` com `{"estudo":"breast_cancer"}` ou `{"estudo":"pcos"}`
- Rotas médicas: `GET /api/rotas/cenario`, `POST /api/rotas/otimizar`, `GET /api/rotas/resultado`, `GET /api/rotas/historico`, `POST /api/rotas/assistente`
- Verificação local: `/healthz` (aplicação ativa) e `/readyz` (exports e resultados disponíveis)

Gere os resultados genéticos antes de consultar os endpoints da Fase 2:

```powershell
python -m src.optimization.cli
```

O botão **Gerar interpretação** usa regras locais se nenhum provedor estiver configurado. Para geração real por LLM sem nuvem, execute `.\scripts\start_ollama.ps1` e depois `.\run_web.ps1` no mesmo terminal; o script constrói a imagem local `tech-challenge-ollama:local` via Podman sobre a imagem oficial Ollama, baixa e pré-carrega `llama3.2:latest` e configura o ambiente. O cliente usa a API nativa com temperatura 0,1, limite de 500 tokens e retenção em memória por 30 minutos. Somente métricas agregadas são enviadas ao provedor. Consulte [FASE2_PROJETO1.md](../../FASE2_PROJETO1.md) para prazos, medições de desempenho e limitações.

Esta entrega é executada localmente como código-fonte; não inclui Docker, Kubernetes ou configuração de nuvem.

O cenário de rotas é inteiramente sintético e usa um mapa SVG offline. Execuções ficam registradas localmente como métricas agregadas para apoiar o relatório semanal; isso não constitui histórico de entregas reais. Veja [FASE2_PROJETO2_ROTAS.md](../../FASE2_PROJETO2_ROTAS.md) para operador genético, restrições, API, histórico, LLM e limitações.
