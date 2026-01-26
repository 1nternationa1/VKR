import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from .prompt_loader import build_prompt


# ---------------------------
# Helpers
# ---------------------------

def _sanitize_jsonable(x: Any) -> Any:
    """
    Convert common non-JSON types to JSON-safe primitives and
    make sure NaN/Inf don't leak into outbound payloads.
    """
    if x is None or isinstance(x, (str, int, bool)):
        return x

    if isinstance(x, float):
        # JSON does not allow NaN/Infinity in strict mode
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(x, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_sanitize_jsonable(v) for v in x]

    # Fallback: stringify unknown objects (datetime, Decimal, UUID, Path, etc.)
    return str(x)


def _strict_json_dumps(obj: Any) -> str:
    """
    Strict JSON dump:
    - Converts non-serializable objects to str (via sanitize)
    - Rejects NaN/Inf by converting them to None
    """
    safe = _sanitize_jsonable(obj)
    return json.dumps(safe, ensure_ascii=False, allow_nan=False)


def _to_amvera_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Amvera /models/gpt expects: [{"role":"system|user|assistant","text":"..."}]
    Some code builds OpenAI style: {"content": "..."}.
    This converts both to Amvera format.
    """
    out: List[Dict[str, str]] = []
    for m in messages or []:
        role = str(m.get("role") or "user")
        text = m.get("text")
        if text is None:
            text = m.get("content")
        out.append({"role": role, "text": "" if text is None else str(text)})
    return out


# ---------------------------
# Provider interface
# ---------------------------

class AIProvider(ABC):
    @abstractmethod
    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        """Return the raw LLM response as a string."""


# ---------------------------
# Cloud provider (Amvera / OpenAI-compat hybrid)
# ---------------------------

class CloudProvider(AIProvider):
    def __init__(self) -> None:
        self.api_url = os.getenv("CLOUD_API_URL", "https://kong-proxy.yc.amvera.ru/api/v1")
        self.api_key = os.getenv("CLOUD_API_KEY")
        self.model = os.getenv("CLOUD_MODEL", "gpt-5")
        self.timeout = float(os.getenv("CLOUD_TIMEOUT", "20"))
        self.temperature = float(os.getenv("CLOUD_TEMPERATURE", "0.2"))

    async def _chat(self, payload: Dict[str, Any]) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("Cloud provider is not configured. Set CLOUD_API_URL and CLOUD_API_KEY.")

        headers = {
            # Some gateways accept one or both; keep both for compatibility.
            "X-Auth-Token": f"Bearer {self.api_key}",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        base_url = self.api_url.rstrip("/")
        url_candidates: List[str] = []

        # If user already provided a full endpoint, respect it.
        if "/models/" in base_url or "/chat/completions" in base_url:
            url_candidates.append(base_url)
        else:
            # Prefer Amvera documented path first; then OpenAI-compatible fallback if enabled upstream.
            inference = "gpt" if (payload.get("model") or self.model).startswith("gpt-") else "llama"
            url_candidates.append(f"{base_url}/models/{inference}")
            url_candidates.append(f"{base_url}/chat/completions")

        # Try primary model, then fallback models
        models_to_try = [payload.get("model") or self.model]
        fallback_model = os.getenv("CLOUD_MODEL_FALLBACK")
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)
        # Optional last-resort legacy, in case gateway maps it
        if self.model != "gpt-3.5-turbo-0125" and "gpt-3.5-turbo-0125" not in models_to_try:
            models_to_try.append("gpt-3.5-turbo-0125")

        last_exc: Optional[httpx.HTTPStatusError] = None

        for final_url in url_candidates:
            for model_name in models_to_try:
                # IMPORTANT: do not mutate the original payload across attempts
                attempt: Dict[str, Any] = dict(payload)
                attempt["model"] = model_name

                # Add temperature if not set (won't hurt if ignored by backend)
                attempt.setdefault("temperature", self.temperature)

                # If hitting Amvera inference endpoint, convert messages schema to {"text":...}
                if "/models/" in final_url:
                    attempt["messages"] = _to_amvera_messages(attempt.get("messages", []))

                # Ensure outbound JSON is valid (no NaN/Inf, no exotic types)
                # Also gives a clean error before network if something is wrong.
                _strict_json_dumps(attempt)

                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(final_url, json=attempt, headers=headers)
                        response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    text = (exc.response.text or "").lower()

                    # If endpoint/model is missing or schema rejected, try next candidate.
                    if (
                        exc.response.status_code in (400, 403, 404)
                        or "not found" in text
                        or "unknown" in text
                        or "invalid json" in text
                        or "schema" in text
                        or "validation" in text
                    ):
                        continue

                    raise RuntimeError(f"LLM HTTP error: {exc.response.status_code} {exc.response.text}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"LLM request failed: {exc}") from exc

                # Parse response
                body = response.json()

                raw_content: Any = None
                if isinstance(body, dict):
                    # OpenAI-like
                    raw_content = body.get("choices", [{}])[0].get("message", {}).get("content")
                    # Amvera-like
                    if not raw_content:
                        raw_content = body.get("message", {}).get("text")

                if not raw_content:
                    raw_content = json.dumps(body, ensure_ascii=False)

                if isinstance(raw_content, (dict, list)):
                    raw_content = json.dumps(raw_content, ensure_ascii=False)

                if raw_content:
                    return str(raw_content)

        if last_exc:
            raise RuntimeError(f"LLM HTTP error: {last_exc.response.status_code} {last_exc.response.text}") from last_exc
        raise RuntimeError("Empty response from provider")

    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        prompt = build_prompt(property_data)

        system_text = (
            "Ты аналитик недвижимости. Ответь строго валидным JSON по схеме: "
            '{"summary":"","recommendation":"","risk_score":0,'
            '"price_range":{"min_value":0,"max_value":0,"currency":"RUB"},'
            '"pros":[],"cons":[],"checks":[]} без лишнего текста.'
        )

        messages = [
            {"role": "system", "text": system_text},
            {"role": "user", "text": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
        }

        return await self._chat(payload)

    async def generate_comparison(self, objects: List[Dict[str, Any]]) -> str:
        # objects may contain non-serializable types or NaN; sanitize first
        formatted_objects = json.dumps(_sanitize_jsonable(objects), ensure_ascii=False, indent=2)

        prompt = (
            "Ты эксперт по недвижимости. Тебе даны 2-3 объекта. "
            "Определи, какой выгоднее, и верни строго JSON по схеме:\n"
            "{\n"
            '  \"winner_id\": \"A\",\n'
            '  \"score\": {\"A\": 82, \"B\": 74, \"C\": 61},\n'
            '  \"reasons\": [\"...\"],\n'
            '  \"risks\": [\"...\"],\n'
            '  \"checks\": [\"...\"],\n'
            '  \"summary\": \"...\"\n'
            "}\n"
            "Требования: winner_id только из входных id; score в диапазоне 0-100 (целые); 3-6 reasons, 2-4 risks, 2-4 checks. "
            "Пиши кратко на русском. Не добавляй текста вне JSON. Входные данные:\n"
            f"{formatted_objects}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "text": "Ты аналитик недвижимости. Отвечай строго валидным JSON без лишнего текста."},
                {"role": "user", "text": prompt},
            ],
        }

        return await self._chat(payload)


# ---------------------------
# Local stub provider
# ---------------------------

class LocalStubProvider(AIProvider):
    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        area = property_data.get("area") or property_data.get("square_meters")
        base_price = 1200 * float(area or 50)
        low = round(base_price * 0.9, 2)
        high = round(base_price * 1.1, 2)
        return (
            "Черновой ответ (заглушка):\n"
            "— Объект выглядит стандартно, данных мало — нужна очная проверка.\n"
            f"— Ориентир цены: {low:,.0f}–{high:,.0f} у.е. по площади {area or 'N/A'} м².\n"
            "— Сильные стороны: базовая инфраструктура, типовой уровень спроса.\n"
            "— Риски: нет информации о ремонте и документах, цену нужно сравнить с аналогами.\n"
            "Рекомендация: запросить документы, осмотреть объект и уточнить состояние инженерии."
        )


# ---------------------------
# Gemini provider
# ---------------------------

class GeminiProvider(AIProvider):
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.fallback_model = os.getenv("GEMINI_MODEL_FALLBACK") or "gemini-1.5-flash-latest"
        self.api_url = os.getenv("GEMINI_API_URL")  # optional override
        self.timeout = float(os.getenv("GEMINI_TIMEOUT", "20"))
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.25"))

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

    async def _call(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini provider is not configured. Set GEMINI_API_KEY.")

        payload = self._build_payload(prompt)
        params = {"key": self.api_key}

        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        for candidate in [
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-pro",
        ]:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        last_exc: Optional[httpx.HTTPStatusError] = None

        for idx, model_name in enumerate(models_to_try):
            for version in ["v1beta", "v1"]:
                url = (
                    self.api_url
                    if idx == 0 and version == "v1beta" and self.api_url
                    else f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent"
                )
                payload["model"] = model_name

                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(url, params=params, json=payload)
                        response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    text = (exc.response.text or "").lower()
                    if exc.response.status_code in (403, 404) and "model" in text:
                        continue
                    raise RuntimeError(f"Gemini HTTP error: {exc.response.status_code} {exc.response.text}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"Gemini request failed: {exc}") from exc

                body = response.json()
                text_out: Optional[str] = None

                try:
                    if isinstance(body, dict):
                        text_out = (
                            body.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text")
                        )
                except Exception:
                    text_out = None

                if not text_out:
                    text_out = json.dumps(body, ensure_ascii=False)

                if isinstance(text_out, (dict, list)):
                    text_out = json.dumps(text_out, ensure_ascii=False)

                if text_out:
                    return str(text_out)

        if last_exc:
            raise RuntimeError(f"Gemini HTTP error: {last_exc.response.status_code} {last_exc.response.text}") from last_exc
        raise RuntimeError("Empty response from Gemini")

    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        prompt = (
            "Ты эксперт по недвижимости. Верни строго валидный JSON по схеме: "
            '{"summary":"","recommendation":"","risk_score":0,'
            '"price_range":{"min_value":0,"max_value":0,"currency":"RUB"},'
            '"pros":[],"cons":[],"checks":[]} без лишнего текста. Пиши кратко.\n'
        )
        prompt += "Данные об объекте:\n" + build_prompt(property_data)
        return await self._call(prompt)

    async def generate_comparison(self, objects: List[Dict[str, Any]]) -> str:
        formatted_objects = json.dumps(_sanitize_jsonable(objects), ensure_ascii=False, indent=2)
        prompt = (
            "Ты эксперт по недвижимости. Даны 2-3 объекта. Верни строго JSON по схеме:\n"
            "{\n"
            '  \"winner_id\": \"A\",\n'
            '  \"score\": {\"A\": 82, \"B\": 74, \"C\": 61},\n'
            '  \"reasons\": [\"...\"],\n'
            '  \"risks\": [\"...\"],\n'
            '  \"checks\": [\"...\"],\n'
            '  \"summary\": \"...\"\n'
            "}\n"
            "Условия: winner_id только из входных id; score целые 0-100; 3-6 reasons, 2-4 risks, 2-4 checks. "
            "Кратко, на русском, без текста вне JSON. Вход:\n"
            f"{formatted_objects}"
        )
        return await self._call(prompt)


def get_provider() -> AIProvider:
    mode = os.getenv("AI_MODE", "cloud").lower()
    if mode == "stub":
        return LocalStubProvider()
    if mode == "gemini":
        return GeminiProvider()
    return CloudProvider()
