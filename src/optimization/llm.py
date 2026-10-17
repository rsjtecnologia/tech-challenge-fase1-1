"""OpenAI-compatible LLM integration for aggregate, non-patient results."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when LLM settings are missing or invalid."""


class LLMNotConfiguredError(LLMConfigurationError):
    """Raised when no LLM model has been selected."""


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot return a valid result."""


SYSTEM_PROMPT = """Você é um assistente de interpretação de métricas de modelos de ML
em contexto acadêmico de saúde. Os dados são agregados e descrevem experimentos,
não uma pessoa. Não diagnostique, não recomende tratamento e não invente fatos.
Explique métricas em português simples, diferencie recall de precisão, mencione
limitações e incerteza, e deixe explícito que o resultado não substitui avaliação
clínica. Responda somente com um objeto JSON válido contendo:
summary (string), findings (lista de strings), limitations (lista de strings) e
disclaimer (string). Utilize exclusivamente os valores fornecidos no contexto.
"""

REQUIRED_KEYS = {"summary", "findings", "limitations", "disclaimer"}
FORBIDDEN_CERTAINTY = re.compile(
    r"\b(cura|garante|garantido|diagnóstico definitivo|certeza absoluta)\b",
    flags=re.IGNORECASE,
)


def build_interpretation_context(
    study: dict[str, Any],
) -> dict[str, Any]:
    """Keep only aggregate metrics; never include rows or individual features."""
    models: dict[str, Any] = {}
    for name, result in study["models"].items():
        models[name] = {
            "baseline_test_metrics": result["baseline"]["test_metrics"],
            "optimized_test_metrics": result["optimized"]["test_metrics"],
            "optimized_cv_metrics": result["optimized"]["cv_metrics"],
            "selected_experiment": result["optimized"]["selected_experiment"],
        }
    return {
        "study": study["label"],
        "positive_class": study["positive_class"],
        "train_samples": study["split"]["train"],
        "test_samples": study["split"]["test"],
        "models": models,
        "fitness_formula": "0.50*recall + 0.30*f1 + 0.20*accuracy",
        "future_textual_data_readiness": {
            "status": "not_enabled",
            "planned_input": "validated, de-identified aggregate text summaries only",
            "restrictions": [
                "Do not include patient records, names, identifiers, or raw clinical notes.",
                "Textual context requires privacy review and explicit authorization before use.",
                "Text must not be used to make an autonomous diagnosis or treatment decision.",
            ],
        },
    }


def evaluate_interpretation(
    interpretation: dict[str, Any],
) -> dict[str, Any]:
    """Apply a small, reproducible structural and safety rubric."""
    missing = REQUIRED_KEYS.difference(interpretation)
    if missing:
        return {
            "score": 0,
            "maximum": 5,
            "passed": False,
            "checks": {"required_sections": False},
            "missing_sections": sorted(missing),
        }

    well_formed = (
        isinstance(interpretation["summary"], str)
        and isinstance(interpretation["disclaimer"], str)
        and isinstance(interpretation["findings"], list)
        and all(isinstance(item, str) for item in interpretation["findings"])
        and isinstance(interpretation["limitations"], list)
        and all(isinstance(item, str) for item in interpretation["limitations"])
    )
    if not well_formed:
        return {
            "score": 0,
            "maximum": 5,
            "passed": False,
            "checks": {"required_sections": False},
            "missing_sections": [],
        }

    strings = [
        interpretation["summary"],
        interpretation["disclaimer"],
        *interpretation["findings"],
        *interpretation["limitations"],
    ]
    text = "\n".join(str(value) for value in strings)
    checks = {
        "required_sections": all(
            isinstance(interpretation[key], str)
            for key in ("summary", "disclaimer")
        )
        and isinstance(interpretation["findings"], list)
        and bool(interpretation["findings"])
        and isinstance(interpretation["limitations"], list)
        and bool(interpretation["limitations"]),
        "medical_disclaimer": (
            "não substitui" in interpretation["disclaimer"].casefold()
            or "nao substitui" in interpretation["disclaimer"].casefold()
        ),
        "no_absolute_claims": FORBIDDEN_CERTAINTY.search(text) is None,
        "bounded_length": len(text) <= 4000,
        "portuguese_output": any(
            marker in text.casefold()
            for marker in (" modelo", " métr", "dados", "saúde", "saude")
        ),
        "metric_grounding": any(
            metric in text.casefold()
            for metric in ("recall", "precisão", "precision", "f1", "acurácia", "accuracy")
        ),
        "actionable_validation_step": any(
            action in text.casefold()
            for action in (
                "valid",
                "revis",
                "avali",
                "investig",
                "acompanhe",
                "exam",
            )
        ),
    }
    passed = sum(checks.values())
    return {
        "score": passed,
        "maximum": len(checks),
        "passed": passed == len(checks),
        "checks": checks,
        "missing_sections": [],
    }


def _settings() -> tuple[str, str, str | None, int]:
    provider = os.getenv("LLM_PROVIDER", "").strip().casefold()
    if not provider:
        provider = (
            "openai_compatible"
            if os.getenv("LLM_MODEL", "").strip()
            else ""
        )
    if provider not in ("", "ollama", "openai_compatible"):
        raise LLMConfigurationError(
            "LLM_PROVIDER deve ser 'ollama' ou 'openai_compatible'."
        )

    default_base_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = os.getenv("LLM_BASE_URL", default_base_url).rstrip("/")
    default_model = "llama3.2:latest" if provider == "ollama" else ""
    model = os.getenv("LLM_MODEL", default_model).strip()
    api_key = os.getenv("LLM_API_KEY")
    try:
        default_timeout = "60" if provider == "ollama" else "45"
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", default_timeout))
    except ValueError as error:
        raise LLMConfigurationError(
            "LLM_TIMEOUT_SECONDS deve ser um inteiro positivo."
        ) from error
    if not model:
        if provider:
            raise LLMConfigurationError(
                "Configure LLM_MODEL para o provedor LLM selecionado."
            )
        raise LLMNotConfiguredError("Configure a variável LLM_MODEL.")
    if timeout <= 0:
        raise LLMConfigurationError("LLM_TIMEOUT_SECONDS deve ser positivo.")
    if provider == "ollama":
        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme not in ("http", "https")
            or parsed_url.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise LLMConfigurationError(
                "O provedor Ollama deve usar um endpoint local."
            )
        if parsed_url.path.rstrip("/") not in ("", "/v1"):
            raise LLMConfigurationError(
                "LLM_BASE_URL do Ollama deve apontar para a raiz, sem /api/chat."
            )
        return urlunsplit(
            (parsed_url.scheme, parsed_url.netloc, "", "", "")
        ), model, None, timeout
    if not api_key and not any(
        host in base_url.casefold() for host in ("localhost", "127.0.0.1", "[::1]")
    ):
        raise LLMConfigurationError(
            "Configure LLM_API_KEY ou use um endpoint local compatível."
        )
    return base_url, model, api_key, timeout


def _local_interpretation(context: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic, aggregate-only explanation when no LLM is configured."""
    model_metrics: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for model_name, result in context["models"].items():
        baseline = result["baseline_test_metrics"]
        optimized = result["optimized_test_metrics"]
        model_metrics.append((model_name, baseline, optimized))

    if not model_metrics:
        raise ValueError("Não há métricas de modelos para interpretar.")

    recall_best = max(
        model_metrics,
        key=lambda item: item[2].get("recall", float("-inf")),
    )
    model_count = len(model_metrics)
    test_samples = context["test_samples"]
    findings: list[str] = []

    for name, baseline, optimized in model_metrics:
        recall_delta = optimized["recall"] - baseline["recall"]
        f1_delta = optimized["f1"] - baseline["f1"]
        precision_delta = optimized["precision"] - baseline["precision"]
        accuracy_delta = optimized["accuracy"] - baseline["accuracy"]
        findings.append(
            f"{name}: recall {baseline['recall']:.1%} → {optimized['recall']:.1%} "
            f"({recall_delta:+.1%}); precisão {baseline['precision']:.1%} → "
            f"{optimized['precision']:.1%}; F1 {baseline['f1']:.1%} → "
            f"{optimized['f1']:.1%}; acurácia {baseline['accuracy']:.1%} → "
            f"{optimized['accuracy']:.1%}."
        )
        if recall_delta < -0.001:
            findings.append(
                f"Atenção para {name}: o recall caiu após a otimização. "
                "Recomenda-se revisar erros e validar novamente em dados "
                "independentes antes de considerar qualquer uso clínico."
            )
        elif recall_delta > 0.001:
            findings.append(
                f"Em {name}, o recall aumentou no holdout; confirme se o ganho "
                "se mantém em validação externa e avalie a troca com precisão "
                f"({precision_delta:+.1%}) e F1 ({f1_delta:+.1%})."
            )
        elif accuracy_delta < -0.001 or f1_delta < -0.001:
            findings.append(
                f"Em {name}, recall ficou semelhante, mas houve queda em outra "
                "métrica; examine a matriz de confusão e o equilíbrio entre "
                "métricas antes de selecionar o modelo."
            )

    findings.append(
        f"Maior recall otimizado neste conjunto: {recall_best[0]} "
        f"({recall_best[2]['recall']:.1%}). Isso é uma comparação experimental, "
        "não uma recomendação de modelo clínico."
    )
    return {
        "summary": (
            f"Foram comparados {model_count} modelos no estudo "
            f"{context['study']}, usando métricas agregadas de um holdout de "
            f"{test_samples} amostras. {recall_best[0]} obteve o maior recall "
            f"otimizado ({recall_best[2]['recall']:.1%}); os resultados variam "
            "por modelo e nenhuma melhora é universal."
        ),
        "findings": findings,
        "limitations": [
            "As métricas são agregadas de um único holdout; não mostram resultados individuais.",
            "O conjunto de teste não substitui validação externa ou avaliação clínica.",
            "Recall (sensibilidade) e precisão medem aspectos diferentes; melhorar um pode afetar o outro.",
            "A interpretação local é baseada em regras e não foi gerada por uma LLM.",
            "Dados textuais clínicos não estão conectados; integração futura exige revisão de privacidade e desidentificação.",
        ],
        "disclaimer": (
            "Conteúdo acadêmico sobre desempenho agregado; não constitui diagnóstico "
            "individual e não substitui avaliação de profissionais de saúde."
        ),
    }


def validate_configuration() -> bool:
    """Return whether a real provider is configured; missing config permits local mode."""
    try:
        _settings()
    except LLMNotConfiguredError:
        return False
    return True


def request_chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    *,
    json_mode: bool = False,
) -> str:
    """Request a bounded completion from Ollama native or an OpenAI endpoint."""
    base_url, model, api_key, timeout = _settings()
    ollama = os.getenv("LLM_PROVIDER", "").strip().casefold() == "ollama"
    try:
        default_temperature = "0.1" if ollama else str(temperature)
        temperature = float(os.getenv("LLM_TEMPERATURE", default_temperature))
        num_predict = int(os.getenv("LLM_NUM_PREDICT", "500"))
    except ValueError as error:
        raise LLMConfigurationError(
            "LLM_TEMPERATURE deve ser numérico e LLM_NUM_PREDICT inteiro positivo."
        ) from error
    if not math.isfinite(temperature) or not 0 <= temperature <= 2 or num_predict <= 0:
        raise LLMConfigurationError(
            "LLM_TEMPERATURE deve estar entre 0 e 2 e LLM_NUM_PREDICT deve ser positivo."
        )
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    endpoint = f"{base_url}/chat/completions"
    if ollama:
        endpoint = f"{base_url}/api/chat"
        del payload["temperature"]
        payload.update({
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        })
        if json_mode:
            payload["format"] = "json"
    elif json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:500]
        raise LLMProviderError(
            f"Provedor LLM respondeu HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        if os.getenv("LLM_PROVIDER", "").strip().casefold() == "ollama":
            raise LLMProviderError(
                "Não foi possível conectar ao Ollama local. Inicie o Ollama e "
                "baixe o modelo configurado (por padrão: "
                f"`ollama pull {model}`)."
            ) from error
        raise LLMProviderError(
            f"Não foi possível conectar ao provedor LLM: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise LLMProviderError(
            f"Tempo limite de {timeout}s ao consultar o provedor LLM. "
            "A inferência local em CPU pode demorar vários minutos; "
            "ajuste LLM_TIMEOUT_SECONDS se necessário."
        ) from error

    try:
        response_data = json.loads(response_body)
        if ollama:
            if response_data.get("done_reason") == "length":
                raise LLMProviderError(
                    f"Ollama atingiu o limite de {num_predict} tokens. "
                    "Reduza o pedido ou aumente LLM_NUM_PREDICT."
                )
            content = response_data["message"]["content"]
            LOGGER.info(
                "ollama_complete model=%s elapsed_seconds=%.2f prompt_tokens=%s generated_tokens=%s",
                model, time.monotonic() - started,
                response_data.get("prompt_eval_count"),
                response_data.get("eval_count"),
            )
        else:
            content = response_data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise LLMProviderError(
            "O provedor LLM retornou uma resposta fora do formato esperado."
        ) from error
    if not isinstance(content, str):
        raise LLMProviderError("O provedor LLM não retornou conteúdo textual.")
    return content


def generate_interpretation(context: dict[str, Any]) -> dict[str, Any]:
    """Use a configured LLM or a transparent aggregate-only local interpretation."""
    metric_names = ("recall", "precision", "f1", "accuracy")
    prompt_context = {
        "study": context["study"],
        "test_samples": context["test_samples"],
        "positive_class": context["positive_class"],
        "metric_order": metric_names,
        "metric_unit": "percent",
        "models": {
            name: {
                label: [
                    round(result[key][metric] * 100, 2)
                    for metric in metric_names
                ]
                for label, key in (
                    ("baseline", "baseline_test_metrics"),
                    ("optimized", "optimized_test_metrics"),
                )
            }
            for name, result in context["models"].items()
        },
    }
    user_prompt = (
        "Compare baseline/optimized no holdout. Não afirme melhora universal; "
        "destaque ganhos e quedas por modelo, sem inventar experimentos ou "
        "significância estatística. Recall mede positivos identificados; "
        "precisão mede positivos preditos corretos. Recomende validação externa. "
        "Dados textuais individuais não estão habilitados. "
        "JSON: summary (1 frase), findings (2 frases), limitations (1 frase), "
        "disclaimer ('Não substitui avaliação clínica.'). "
        "Use listas de strings em findings e limitations. Máximo 140 palavras.\n"
        + json.dumps(
            prompt_context, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
    )
    try:
        content = request_chat_completion(
            SYSTEM_PROMPT, user_prompt, json_mode=True
        )
        source = "llm"
    except LLMNotConfiguredError:
        interpretation = _local_interpretation(context)
        evaluation = evaluate_interpretation(interpretation)
        return {
            "interpretation": interpretation,
            "quality_evaluation": evaluation,
            "source": "local",
        }
    try:
        interpretation = json.loads(content)
    except json.JSONDecodeError as error:
        raise LLMProviderError(
            "O provedor LLM retornou uma resposta fora do formato JSON esperado."
        ) from error
    if not isinstance(interpretation, dict):
        raise LLMProviderError("A interpretação LLM deve ser um objeto JSON.")
    if REQUIRED_KEYS.difference(interpretation):
        raise LLMProviderError(
            "A interpretação LLM não contém todos os campos solicitados."
        )
    if (
        not isinstance(interpretation["summary"], str)
        or not isinstance(interpretation["disclaimer"], str)
        or not isinstance(interpretation["findings"], list)
        or not all(isinstance(item, str) for item in interpretation["findings"])
        or not isinstance(interpretation["limitations"], list)
        or not all(
            isinstance(item, str) for item in interpretation["limitations"]
        )
    ):
        raise LLMProviderError(
            "A interpretação LLM não corresponde ao esquema solicitado."
        )
    evaluation = evaluate_interpretation(interpretation)
    return {
        "interpretation": interpretation,
        "quality_evaluation": evaluation,
        "source": source,
    }
