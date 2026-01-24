import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from .prompt_loader import build_prompt


class AIProvider(ABC):
    @abstractmethod
    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        """Return the raw LLM response as a string."""


class CloudProvider(AIProvider):
    def __init__(self) -> None:
        self.api_url = os.getenv("CLOUD_API_URL", "https://kong-proxy.yc.amvera.ru/api/v1")
        self.api_key = os.getenv("CLOUD_API_KEY")
        self.model = os.getenv("CLOUD_MODEL", "gpt-5")
        self.timeout = float(os.getenv("CLOUD_TIMEOUT", "20"))

    async def _chat(self, payload: Dict[str, Any]) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("Cloud provider is not configured. Set CLOUD_API_URL and CLOUD_API_KEY.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Try primary model, then fallback if configured/access denied.
        models_to_try = [payload.get("model") or self.model]
        fallback_model = os.getenv("CLOUD_MODEL_FALLBACK")
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)
        elif self.model != "gpt-3.5-turbo-0125":
            models_to_try.append("gpt-3.5-turbo-0125")

        last_exc: httpx.HTTPStatusError | None = None

        for model_name in models_to_try:
            payload["model"] = model_name
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.api_url, json=payload, headers=headers)
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                text = exc.response.text.lower()
                if exc.response.status_code in (403, 404) and "model" in text:
                    # try next model
                    continue
                raise RuntimeError(f"LLM HTTP error: {exc.response.status_code} {exc.response.text}") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"LLM request failed: {exc}") from exc

            body = response.json()
            raw_content: Any = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
                if isinstance(body, dict)
                else None
            )

            if not raw_content:
                raw_content = body.get("content") if isinstance(body, dict) else None

            if not raw_content:
                raw_content = json.dumps(body)

            if isinstance(raw_content, (dict, list)):
                raw_content = json.dumps(raw_content)

            if not raw_content:
                continue

            return str(raw_content)

        if last_exc:
            raise RuntimeError(f"LLM HTTP error: {last_exc.response.status_code} {last_exc.response.text}") from last_exc
        raise RuntimeError("Empty response from provider")

    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        prompt = build_prompt(property_data)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты аналитик недвижимости. Ответь строго валидным JSON по схеме: "
                        '{"summary":"","recommendation":"","risk_score":0,"price_range":{"min_value":0,"max_value":0,"currency":"RUB"},'
                        '"pros":[],"cons":[],"checks":[]} без лишнего текста.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        return await self._chat(payload)

    async def generate_comparison(self, objects: List[Dict[str, Any]]) -> str:
        formatted_objects = json.dumps(objects, ensure_ascii=False, indent=2)
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
                {
                    "role": "system",
                    "content": "Ты аналитик недвижимости. Отвечай строго валидным JSON без лишнего текста.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
        }
        return await self._chat(payload)


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
            "generationConfig": {
                "temperature": self.temperature
            }
        }

    async def _call(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini provider is not configured. Set GEMINI_API_KEY.")

        payload = self._build_payload(prompt)
        params = {"key": self.api_key}
        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)
        for candidate in ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        last_exc: Optional[httpx.HTTPStatusError] = None

        for idx, model_name in enumerate(models_to_try):
            versions = ["v1beta", "v1"]
            for version in versions:
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
                    text = exc.response.text.lower()
                    if exc.response.status_code in (403, 404) and "model" in text:
                        continue
                    raise RuntimeError(f"Gemini HTTP error: {exc.response.status_code} {exc.response.text}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"Gemini request failed: {exc}") from exc

                body = response.json()
                text: Optional[str] = None
                try:
                    text = (
                        body.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text")
                        if isinstance(body, dict)
                        else None
                    )
                except Exception:
                    text = None

                if not text:
                    text = json.dumps(body)

                if isinstance(text, (dict, list)):
                    text = json.dumps(text)

                if not text:
                    continue

                return str(text)

        if last_exc:
            raise RuntimeError(f"Gemini HTTP error: {last_exc.response.status_code} {last_exc.response.text}") from last_exc
        raise RuntimeError("Empty response from Gemini")

    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        prompt = (
            "Ты эксперт по недвижимости. Верни строго валидный JSON по схеме: "
            '{"summary":"","recommendation":"","risk_score":0,"price_range":{"min_value":0,"max_value":0,"currency":"RUB"},'
            '"pros":[],"cons":[],"checks":[]} без лишнего текста. Пиши кратко.\n'
        )
        prompt += "Данные об объекте:\n" + build_prompt(property_data)
        return await self._call(prompt)

    async def generate_comparison(self, objects: List[Dict[str, Any]]) -> str:
        formatted_objects = json.dumps(objects, ensure_ascii=False, indent=2)
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
