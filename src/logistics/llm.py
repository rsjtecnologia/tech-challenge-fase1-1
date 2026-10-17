"""LLM prompts for route instructions, operations reports and route Q&A."""

from __future__ import annotations

import json
from typing import Any, Literal

from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    request_chat_completion,
)


Action = Literal["instructions", "daily_report", "weekly_report", "question"]
SYSTEM_PROMPT = """Você auxilia uma equipe de logística hospitalar num cenário
acadêmico de demonstração. Os dados recebidos descrevem uma rota sintética;
não são tráfego em tempo real nem operação/paciente real. Use somente o contexto
fornecido. Não invente trânsito, horários, destinatários, itens, quilometragem,
economia monetária ou fatos externos. Distâncias são aproximações em linha reta,
não rotas de rua. Não altere prioridades médicas nem instruções de conservação.
Quando faltar informação, diga claramente que ela não consta. Escreva em
português direto, útil para motoristas e coordenação. Não inclua conselhos
clínicos. Responda somente JSON válido no formato solicitado pelo pedido.
"""

ACTION_PROMPTS = {
    "instructions": (
        "Gere instruções detalhadas para os motoristas, uma sequência por veículo, "
        "respeitando as paradas e horários estimados do plano. Preserve itens "
        "refrigerados e prioridades sem inventar procedimentos. Responda JSON com "
        "summary (string), vehicle_instructions (lista de objetos com vehicle, "
        "ordered_stops, notes), operational_checks (lista de strings) e disclaimer "
        "(string indicando que confirmar horários e condições com a coordenação)."
    ),
    "daily_report": (
        "Produza um relatório operacional diário como proposta para este cenário, "
        "sem afirmar que a entrega ocorreu. Compare o plano base e o otimizado, "
        "distância estimada, minutos agregados dos veículos, utilização da frota, "
        "atrasos e restrições. Os dados são de uma única simulação, não série "
        "histórica; tempo e recursos são estimativas do modelo, não economia "
        "realizada. Responda JSON com period, summary, key_metrics (objeto), "
        "improvements (lista), caveats (lista) e disclaimer (string)."
    ),
    "weekly_report": (
        "Produza um relatório dos registros de simulações disponíveis na janela "
        "dos últimos sete dias. Use o histórico agregado do contexto, informe a "
        "quantidade de execuções e compare médias e variação entre execuções. "
        "Só descreva tendência quando trend_available for true; execuções do "
        "otimizador não equivalem a entregas reais nem a dias de operação. "
        "Baseie sugestões de melhoria nos padrões numéricos observados ou, se "
        "houver poucos dados, recomende quais medições coletar. Não invente "
        "histórico, economia de tempo/custo ou causas. Responda JSON com period, "
        "summary, key_metrics (objeto), improvements (lista), caveats (lista) e "
        "disclaimer (string)."
    ),
    "question": (
        "Responda à pergunta em linguagem natural usando somente os dados do plano. "
        "Não suponha dados em falta; se a pergunta exigir trânsito ou outra fonte "
        "não incluída, informe essa limitação. Responda JSON com answer (string), "
        "supporting_facts (lista de strings), missing_information (lista de "
        "strings) e disclaimer (string)."
    ),
}

REQUIRED_KEYS = {
    "instructions": {
        "summary",
        "vehicle_instructions",
        "operational_checks",
        "disclaimer",
    },
    "daily_report": {
        "period",
        "summary",
        "key_metrics",
        "improvements",
        "caveats",
        "disclaimer",
    },
    "weekly_report": {
        "period",
        "summary",
        "key_metrics",
        "improvements",
        "caveats",
        "disclaimer",
    },
    "question": {
        "answer",
        "supporting_facts",
        "missing_information",
        "disclaimer",
    },
}


def _validate_response(action: Action, parsed: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS[action].difference(parsed)
    if missing:
        raise LLMProviderError(
            f"A resposta LLM está sem os campos: {sorted(missing)}."
        )
    if not isinstance(parsed["disclaimer"], str):
        raise LLMProviderError("O aviso da resposta LLM deve ser textual.")
    list_fields = {
        "instructions": ("operational_checks",),
        "daily_report": ("improvements", "caveats"),
        "weekly_report": ("improvements", "caveats"),
        "question": ("supporting_facts", "missing_information"),
    }[action]
    for field in list_fields:
        if not isinstance(parsed[field], list) or not all(
            isinstance(item, str) for item in parsed[field]
        ):
            raise LLMProviderError(
                f"O campo {field} da resposta LLM deve ser uma lista de textos."
            )
    if action == "instructions":
        if (
            not isinstance(parsed["summary"], str)
            or not isinstance(parsed["vehicle_instructions"], list)
            or not all(
                isinstance(item, dict)
                and {"vehicle", "ordered_stops", "notes"}.issubset(item)
                and isinstance(item["vehicle"], str)
                and isinstance(item["ordered_stops"], list)
                and isinstance(item["notes"], (str, list))
                for item in parsed["vehicle_instructions"]
            )
        ):
            raise LLMProviderError(
                "As instruções LLM não correspondem ao esquema de veículos."
            )
    elif action in ("daily_report", "weekly_report"):
        if not all(isinstance(parsed[field], str) for field in ("period", "summary")):
            raise LLMProviderError("O relatório LLM deve informar período e resumo.")
        if not isinstance(parsed["key_metrics"], dict):
            raise LLMProviderError("key_metrics deve ser um objeto JSON.")
    else:
        if not isinstance(parsed["answer"], str):
            raise LLMProviderError("A resposta da pergunta deve ser textual.")


def _route_context(
    plan: dict[str, Any],
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "scenario_id": plan["scenario_id"],
        "scenario": plan["scenario_name"],
        "synthetic": plan["synthetic"],
        "algorithm": {
            "name": plan["algorithm"]["name"],
            "config": plan["algorithm"]["config"],
            "evaluated_chromosomes": plan["algorithm"]["evaluated_chromosomes"],
        },
        "objective": plan["objective"],
        "depot": plan["depot"],
        "vehicles": plan["vehicles"],
        "deliveries": plan["deliveries"],
        "baseline": plan["comparison"]["baseline"],
        "optimized": plan["comparison"]["optimized"],
        "distance_saved_km": plan["comparison"]["distance_saved_km"],
        "distance_saved_percent": plan["comparison"]["distance_saved_percent"],
        "elapsed_seconds": plan["elapsed_seconds"],
        "data_limitations": [
            "Cenário sintético, com pontos ilustrativos; não descreve entregas reais.",
            "Distâncias Haversine em linha reta; não são distâncias de ruas.",
            "Não há trânsito, clima, custos, histórico de entregas reais ou dados de pacientes.",
            "Prazos e horários são estimativas do cenário e devem ser confirmados.",
        ],
    }
    if history is not None:
        context["simulation_history_last_7_days"] = history
    return context


def _local_route_content(
    plan: dict[str, Any],
    action: Action,
    question: str | None,
    history: dict[str, Any] | None,
) -> dict[str, Any]:
    """Provide grounded route summaries when no LLM provider is configured."""
    comparison = plan["comparison"]
    optimized = comparison["optimized"]
    baseline = comparison["baseline"]
    routes = optimized["routes"]
    disclaimer = (
        "Resposta local baseada somente no plano sintético salvo; confirme "
        "rotas e condições com a coordenação. Não é navegação viária."
    )

    if action == "instructions":
        vehicle_instructions = [
            {
                "vehicle": route["vehicle_name"],
                "ordered_stops": [
                    f'{index}. {stop["name"]} — chegada estimada {stop["arrival_time"]}; '
                    f'prioridade {stop["priority"]}; carga {stop["demand_kg"]} kg'
                    for index, stop in enumerate(route["stops"], start=1)
                ],
                "notes": (
                    f'Percurso estimado de {route["distance_km"]} km; '
                    f'carga total {route["load_kg"]} de {route["capacity_kg"]} kg. '
                    "Horários e distâncias são estimativas em linha reta."
                ),
            }
            for route in routes
        ]
        content = {
            "summary": (
                f'Plano local para {optimized["served_delivery_count"]} entregas '
                f'em {len(routes)} veículo(s).'
            ),
            "vehicle_instructions": vehicle_instructions,
            "operational_checks": [
                "Confirme a sequência, os horários e os destinos com a coordenação.",
                "Verifique capacidade e condições de transporte antes da saída.",
                "As distâncias não representam vias, trânsito ou navegação.",
            ],
            "disclaimer": disclaimer,
        }
    elif action == "daily_report":
        content = {
            "period": "Resumo desta simulação diária",
            "summary": (
                f'O plano otimizado atende {optimized["served_delivery_count"]} '
                f'entrega(s), com {optimized["total_distance_km"]} km estimados, '
                f'contra {baseline["total_distance_km"]} km do baseline.'
            ),
            "key_metrics": {
                "entregas_atendidas": optimized["served_delivery_count"],
                "entregas_planejadas": optimized["delivery_count"],
                "distancia_otimizada_km": optimized["total_distance_km"],
                "distancia_baseline_km": baseline["total_distance_km"],
                "minutos_estimados_dos_veiculos": optimized[
                    "estimated_vehicle_minutes"
                ],
                "veiculos_utilizados": optimized["vehicles_used"],
                "carga_total_kg": optimized["total_load_kg"],
                "utilizacao_media_da_capacidade_percentual": optimized[
                    "fleet_utilization_percent"
                ],
                "minutos_estimados_poupados_contra_baseline": round(
                    baseline["estimated_vehicle_minutes"]
                    - optimized["estimated_vehicle_minutes"],
                    2,
                ),
                "distancia_estimadamente_evitada_km": comparison[
                    "distance_saved_km"
                ],
                "reducao_estimada_percentual": comparison[
                    "distance_saved_percent"
                ],
                "entregas_atrasadas_estimadas": optimized["late_delivery_count"],
            },
            "improvements": [
                "Substituir distâncias em linha reta por distâncias rodoviárias.",
                "Comparar minutos estimados de veículo com durações reais de rota.",
                "Registrar execuções futuras para permitir comparações históricas.",
            ],
            "caveats": [
                "O cenário é sintético; os resultados são estimativas de uma simulação.",
                "Não há dados de trânsito, custos ou histórico de operação.",
                "A redução é estimada contra um baseline; não é economia realizada.",
            ],
            "disclaimer": disclaimer,
        }
    elif action == "weekly_report":
        history = history or {}
        run_count = int(history.get("run_count", 0))
        runs = history.get("recent_runs", [])
        if run_count == 0:
            runs = [
                {
                    "optimized_distance_km": optimized["total_distance_km"],
                    "baseline_distance_km": baseline["total_distance_km"],
                    "late_delivery_count": optimized["late_delivery_count"],
                    "weighted_lateness": optimized["total_weighted_lateness"],
                }
            ]
            run_count = 1
        trend_available = run_count >= 2 and len(runs) >= 2
        improvements = []
        if trend_available:
            first_distance = float(runs[0]["optimized_distance_km"])
            last_distance = float(runs[-1]["optimized_distance_km"])
            distance_change = last_distance - first_distance
            if distance_change < -0.5:
                pattern = (
                    f"A distância estimada da primeira à última execução caiu "
                    f"{abs(distance_change):.3f} km; verificar se a diferença se "
                    "mantém em novas sementes e com dados rodoviários."
                )
            elif distance_change > 0.5:
                pattern = (
                    f"A distância estimada da primeira à última execução subiu "
                    f"{distance_change:.3f} km; comparar as configurações e "
                    "repetir os testes antes de concluir que houve regressão."
                )
            else:
                pattern = (
                    "A distância estimada permaneceu semelhante entre a primeira "
                    "e a última execução."
                )
            improvements.append(pattern)
            first_minutes = runs[0].get("optimized_estimated_vehicle_minutes")
            last_minutes = runs[-1].get("optimized_estimated_vehicle_minutes")
            if first_minutes is not None and last_minutes is not None:
                minute_change = float(last_minutes) - float(first_minutes)
                if abs(minute_change) >= 1:
                    direction = "subiu" if minute_change > 0 else "caiu"
                    improvements.append(
                        f"Os minutos estimados de veículo {direction} "
                        f"{abs(minute_change):.2f} entre a primeira e a última "
                        "simulação; validar com tempos reais antes de interpretar "
                        "como economia ou regressão."
                    )
        else:
            improvements.extend(
                [
                    "Executar novas otimizações com sementes diferentes para formar "
                    "uma amostra comparável.",
                    "Registrar métricas de operação reais separadamente antes de "
                    "inferir tendências de entregas ou recursos.",
                ]
            )
        improvements.append(
            "Comparar as distâncias estimadas com distâncias rodoviárias antes de "
            "usar os resultados para planejamento."
        )
        mean_optimized = history.get("mean_optimized_distance_km")
        mean_baseline = history.get("mean_baseline_distance_km")
        mean_saved_percent = history.get("mean_distance_saved_percent")
        mean_late = history.get("mean_late_delivery_count")
        mean_weighted = history.get("mean_weighted_lateness")
        mean_vehicle_minutes = history.get("mean_estimated_vehicle_minutes")
        mean_vehicle_minutes_saved = history.get(
            "mean_estimated_vehicle_minutes_saved"
        )
        mean_vehicles = history.get("mean_vehicles_used")
        mean_utilization = history.get("mean_fleet_utilization_percent")
        if mean_optimized is None:
            mean_optimized = optimized["total_distance_km"]
        if mean_baseline is None:
            mean_baseline = baseline["total_distance_km"]
        if mean_saved_percent is None:
            mean_saved_percent = comparison["distance_saved_percent"]
        if mean_late is None:
            mean_late = optimized["late_delivery_count"]
        if mean_weighted is None:
            mean_weighted = optimized["total_weighted_lateness"]
        if mean_vehicle_minutes is None:
            mean_vehicle_minutes = optimized["estimated_vehicle_minutes"]
        if mean_vehicle_minutes_saved is None:
            mean_vehicle_minutes_saved = round(
                baseline["estimated_vehicle_minutes"]
                - optimized["estimated_vehicle_minutes"],
                2,
            )
        if mean_vehicles is None:
            mean_vehicles = optimized["vehicles_used"]
        if mean_utilization is None:
            mean_utilization = optimized["fleet_utilization_percent"]
        run_label = "simulação" if run_count == 1 else "simulações"
        content = {
            "period": (
                f"Últimos 7 dias — {run_count} {run_label} registrada(s)"
            ),
            "summary": (
                f"Foram registradas {run_count} execução(ões) do otimizador. "
                f"A distância otimizada média foi {mean_optimized} km, contra "
                f"{mean_baseline} km do baseline. "
                + (
                    "Há mais de uma execução para comparar configurações do "
                    "algoritmo, mas isso não representa uma série de entregas reais."
                    if trend_available
                    else "Ainda não há execuções suficientes para inferir um padrão."
                )
            ),
            "key_metrics": {
                "simulacoes_registradas_ultimos_7_dias": run_count,
                "minimo_para_comparar_execucoes": 2,
                "tendencia_entre_simulacoes_disponivel": trend_available,
                "distancia_otimizada_media_km": mean_optimized,
                "distancia_baseline_media_km": mean_baseline,
                "reducao_estimada_media_percentual": mean_saved_percent,
                "minutos_estimados_de_veiculo_media": mean_vehicle_minutes,
                "minutos_estimados_poupados_contra_baseline_media": (
                    mean_vehicle_minutes_saved
                ),
                "veiculos_utilizados_media": mean_vehicles,
                "utilizacao_media_da_frota_percentual": mean_utilization,
                "entregas_atrasadas_estimadas_media": mean_late,
                "atraso_ponderado_estimado_medio": mean_weighted,
                "primeira_execucao": history.get("first_run_at"),
                "ultima_execucao": history.get("last_run_at"),
            },
            "improvements": improvements,
            "caveats": [
                "O histórico contém execuções de simulação, não operações ou entregas reais.",
                "Execuções repetidas no mesmo dia não comprovam uma tendência semanal operacional.",
                "Não são coletados tempo de viagem real, custos financeiros ou trânsito.",
                "Com menos de duas execuções, nenhuma tendência entre execuções é inferida.",
            ],
            "disclaimer": disclaimer,
        }
    else:
        normalized_question = (question or "").casefold()
        supporting_facts = [
            f'{optimized["served_delivery_count"]} de '
            f'{optimized["delivery_count"]} entregas estão atribuídas no plano.',
            f'Distância otimizada estimada: {optimized["total_distance_km"]} km.',
            f'Foram estimadas {optimized["late_delivery_count"]} entregas atrasadas.',
        ]
        if any(term in normalized_question for term in ("frio", "refriger", "gelad")):
            vehicles = {vehicle["id"]: vehicle for vehicle in plan["vehicles"]}
            refrigerated_vehicles = [
                route["vehicle_name"]
                for route in routes
                if vehicles[route["vehicle_id"]]["refrigerated"]
                and any(
                    any("refrigerad" in item.casefold() for item in stop["items"])
                    for stop in route["stops"]
                )
            ]
            supporting_facts.append(
                "Veículo(s) atribuído(s) às entregas com itens refrigerados: "
                + (
                    ", ".join(refrigerated_vehicles)
                    if refrigerated_vehicles
                    else "nenhum no plano otimizado"
                )
                + "."
            )
        if any(
            term in normalized_question
            for term in ("distância", "distancia", "quilometr", "km", "econom")
        ):
            supporting_facts.append(
                f'Baseline: {baseline["total_distance_km"]} km; '
                f'diferença estimada: {comparison["distance_saved_km"]} km '
                f'({comparison["distance_saved_percent"]}%).'
            )
        if any(term in normalized_question for term in ("veículo", "veiculo", "carro")):
            supporting_facts.extend(
                f'{route["vehicle_name"]}: {len(route["stops"])} parada(s), '
                f'{route["distance_km"]} km.'
                for route in routes
            )
        content = {
            "answer": (
                "Este assistente local não usa um modelo de linguagem. "
                "Com base no plano, "
                + " ".join(supporting_facts)
                + " Para perguntas específicas fora desses dados, configure uma LLM."
            ),
            "supporting_facts": supporting_facts,
            "missing_information": [
                "Não há trânsito em tempo real, endereços viários ou custos.",
                "As distâncias são aproximações em linha reta do cenário sintético.",
            ],
            "disclaimer": disclaimer,
        }

    return {
        "action": action,
        "content": content,
        "source": "local",
    }


def generate_route_content(
    plan: dict[str, Any],
    action: Action,
    question: str | None = None,
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if action not in ACTION_PROMPTS:
        raise ValueError(f"Ação LLM desconhecida: {action}")
    if action == "question" and not question:
        raise ValueError("Informe uma pergunta sobre o plano de rotas.")
    if question is not None and len(question) > 1000:
        raise ValueError("A pergunta não pode exceder 1000 caracteres.")

    prompt = ACTION_PROMPTS[action]
    if question:
        prompt += "\nPergunta da equipe: " + question
    try:
        response = request_chat_completion(
            SYSTEM_PROMPT,
            prompt
            + "\nContexto validado do plano:\n"
            + json.dumps(
                _route_context(plan, history),
                ensure_ascii=False,
                allow_nan=False,
            ),
            temperature=0.2,
        )
    except LLMConfigurationError:
        return _local_route_content(plan, action, question, history)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as error:
        raise LLMProviderError(
            "O provedor LLM retornou uma resposta fora do formato JSON."
        ) from error
    if not isinstance(parsed, dict):
        raise LLMProviderError("A resposta LLM precisa ser um objeto JSON.")
    _validate_response(action, parsed)
    return {"action": action, "content": parsed, "source": "llm"}
