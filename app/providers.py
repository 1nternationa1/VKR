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


def _filter_property_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Оставляем только поля, введённые пользователем в форме оценки."""
    allowed_keys = {
        "type",
        "location",
        "address",
        "price",
        "area",
        "rooms",
        "floor",
        "floors_total",
        "year",
        "condition",
        "notes",
    }
    return {k: v for k, v in data.items() if k in allowed_keys and v not in (None, "", [])}


def _trim_strings(data: Dict[str, Any], limit: int = 600) -> Dict[str, Any]:
    """Limit string fields length to avoid huge prompts and upstream 400."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = v[:limit]
        else:
            out[k] = v
    return out


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


def _to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """OpenAI chat/completions expects 'content' instead of 'text'."""
    out: List[Dict[str, str]] = []
    for m in messages or []:
        role = str(m.get("role") or "user")
        content = m.get("content")
        if content is None:
            content = m.get("text")
        out.append({"role": role, "content": "" if content is None else str(content)})
    return out


def _shrink_messages(messages: List[Dict[str, Any]], limit: int = 2000) -> List[Dict[str, Any]]:
    """
    Truncate long message texts to avoid oversized/invalid JSON bodies
    even если апстрим прислал гигантское description.
    """
    trimmed: List[Dict[str, Any]] = []
    for m in messages or []:
        m_copy = dict(m)
        if "text" in m_copy and isinstance(m_copy["text"], str):
            m_copy["text"] = m_copy["text"][:limit]
        if "content" in m_copy and isinstance(m_copy.get("content"), str):
            m_copy["content"] = m_copy["content"][:limit]
        trimmed.append(m_copy)
    return trimmed


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
        self.timeout = float(os.getenv("CLOUD_TIMEOUT", "60"))
        # temp по умолчанию 1.0 для совместимости с /models/gpt Amvera (иначе 400)
        self.temperature = float(os.getenv("CLOUD_TEMPERATURE", "1"))
        self.proxies = {
            "http://": os.getenv("CLOUD_HTTP_PROXY") or os.getenv("HTTP_PROXY") or None,
            "https://": os.getenv("CLOUD_HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or None,
        }
        if not any(self.proxies.values()):
            self.proxies = None
        self.verify_ssl = os.getenv("CLOUD_VERIFY_SSL", "true").lower() not in ("0", "false", "no")
        self.httpx_timeout = httpx.Timeout(
            timeout=self.timeout,
            connect=min(self.timeout, 10.0),
            read=self.timeout,
            write=min(self.timeout, 15.0),
            pool=min(self.timeout, 15.0),
        )
        # http/2 опционален; по умолчанию выключен для совместимости с прокси Amvera
        self.use_http2 = os.getenv("HTTP2", "false").lower() in ("1", "true", "yes")

    def _build_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Cloud provider is not configured. Set CLOUD_API_URL and CLOUD_API_KEY.")
        bearer = self.api_key if str(self.api_key).startswith("Bearer ") else f"Bearer {self.api_key}"
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Auth-Token": bearer,
            # Отдаём и стандартный Authorization для совместимости с OpenAI-совместимыми шлюзами
            "Authorization": bearer,
        }

    def _candidate_urls(self) -> List[str]:
        """
        Возвращает список URL, которые нужно попробовать:
        1) Amvera /models/gpt (родной формат role/text)
        2) OpenAI-совместимый /chat/completions (на случай обратного прокси)
        3) Если указан полный путь (уже содержит /models/ или /chat/completions) — используем только его.
        """
        base = (self.api_url or "").rstrip("/")
        if "/models/" in base or "/chat/completions" in base:
            return [base]
        return [f"{base}/models/gpt", f"{base}/chat/completions"]

    def _prepare_payload(self, url: str, messages: List[Dict[str, Any]], model: str, temperature: float) -> Dict[str, Any]:
        trimmed = _shrink_messages(messages)
        if "/models/" in url and "chat/completions" not in url:
            attempt = {
                "model": model,
                "messages": _to_amvera_messages(trimmed),
            }
            # Amvera /models/gpt принимает только default temperature=1; чтобы не ловить 400, не передаём иной temp
            if temperature and abs(float(temperature) - 1.0) > 1e-6:
                pass
            else:
                attempt["temperature"] = 1
            return attempt
        return {
            "model": model,
            "messages": _to_openai_messages(trimmed),
            "temperature": temperature,
        }

    @staticmethod
    def _parse_response(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except Exception:
            return resp.text

        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or choices[0].get("delta") or {}
                if isinstance(msg, dict):
                    if msg.get("text"):
                        return str(msg["text"])
                    if msg.get("content"):
                        return str(msg["content"])
                elif isinstance(msg, str):
                    return msg
            # прямой формат {"message":{"text": "..."}}
            msg = body.get("message")
            if isinstance(msg, dict):
                if msg.get("text"):
                    return str(msg["text"])
                if msg.get("content"):
                    return str(msg["content"])
            for key in ("text", "content"):
                if key in body and body[key]:
                    return str(body[key])

        return json.dumps(body, ensure_ascii=False)

    async def _chat(self, payload: Dict[str, Any]) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("Cloud provider is not configured. Set CLOUD_API_URL and CLOUD_API_KEY.")

        headers = self._build_headers()
        models_primary = payload.get("model") or self.model
        models_to_try: List[str] = []
        for m in [models_primary, self.model, "gpt-5", "gpt"]:
            if m and m not in models_to_try:
                models_to_try.append(m)

        errors: List[str] = []
        for final_url in self._candidate_urls():
            for model_name in models_to_try:
                attempt = self._prepare_payload(
                    final_url,
                    payload.get("messages", []),
                    model_name,
                    payload.get("temperature", self.temperature),
                )
                try:
                    async with httpx.AsyncClient(
                        timeout=self.httpx_timeout,
                        http2=self.use_http2,
                        proxies=self.proxies,
                        verify=self.verify_ssl,
                    ) as client:
                        response = await client.post(final_url, json=attempt, headers=headers)
                    response.raise_for_status()
                    return self._parse_response(response)
                except httpx.HTTPStatusError as exc:
                    err_text = exc.response.text[:200] if exc.response else str(exc)
                    errors.append(f"{final_url} [{model_name}]: {exc.response.status_code} {err_text}")
                    # Если модель или эндпоинт не подходят — пробуем следующий
                    if exc.response is not None and exc.response.status_code in (400, 403, 404, 422):
                        continue
                    raise RuntimeError(f"LLM HTTP error: {err_text}") from exc
                except httpx.RequestError as exc:
                    errors.append(f"{final_url} [{model_name}]: {type(exc).__name__} {exc}")
                    continue

        raise RuntimeError(f"LLM request failed: {'; '.join(errors[:3])}")

    async def generate_report(self, property_data: Dict[str, Any]) -> str:
        # Если цена не дошла в распарсенных полях (из-за 429 на fetch_listing), попробуем достать её из description.
        def _extract_price_from_text(txt: str) -> Optional[float]:
            if not txt:
                return None
            import re
            candidates = []
            for m in re.finditer(r"(\d[\d\s]{4,})\s*₽?", txt):
                try:
                    candidates.append(float(m.group(1).replace(" ", "")))
                except Exception:
                    continue
            return max(candidates) if candidates else None

        if not property_data.get("price"):
            maybe_price = _extract_price_from_text(property_data.get("description", "")) or _extract_price_from_text(
                property_data.get("source_text", "")
            )
            if maybe_price:
                property_data = dict(property_data)
                property_data["price"] = maybe_price

        filtered = _trim_strings(_filter_property_data(property_data))
        # Человеческое описание + явный JSON с распознанными полями,
        # чтобы модель гарантированно увидела цену/метраж даже при обрезке текста.
        prompt_human = build_prompt(filtered)
        prompt_json = _strict_json_dumps(filtered)
        fallback_text_parts = []
        for key in ("source_text", "description"):
            val = property_data.get(key)
            if isinstance(val, str) and val.strip():
                fallback_text_parts.append(f"{key}:\n{val[:1500]}")
        fallback_block = ("\n\nСырые тексты:\n" + "\n---\n".join(fallback_text_parts)) if fallback_text_parts else ""
        prompt = f"{prompt_human}\n\nДанные JSON:\n{prompt_json}{fallback_block}"

        # Final safety: cap prompt size to avoid proxy 400 on oversized bodies
        max_chars = int(os.getenv("PROMPT_CHAR_LIMIT", "4000"))
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]

        system_text = (
            "Ты генератор JSON. Ответь ОДНОЙ строкой строго валидным JSON без пробела в начале и без markdown. "
            "Схема ответа фиксирована и неизменна: "
            '{"summary":"<1–2 предложения: тип, локация, метраж, состояние/год, ключевой плюс>",'
            '"recommendation":"<следующий шаг>",'
            '"risk_score":0.0,'
            '"price_range":{"min_value":0,"max_value":0,"currency":"RUB"},'
            '"pros":[],'
            '"cons":[],'
            '"checks":[]} '
            "Правила: "
            "- Не добавляй никаких других полей, текста, комментариев, markdown. "
            "- Строки обрезай до 200 символов, массивы максимум 4 элемента. "
            "- Если цена передана (price), то price_range.min_value = price_range.max_value = <входной price>, currency=\"RUB\". "
            "Если цена отсутствует — оставь 0. "
            "- Не копируй длинное description, используй только суть. "
            "- Если данных нет — ставь пустую строку или пустой массив, risk_score = 0.0. "
            "- Ответ должен быть валидным JSON без NaN/Infinity/None."
        )

        messages = [
            {"role": "system", "text": system_text},
            {"role": "user", "text": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
        }

        raw = await self._chat(payload)

        # Пост-обработка: если модель вернула JSON, принудительно проставляем цену и приводим к схеме.
        try:
            data = json.loads(raw)
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                price = filtered.get("price")
                try:
                    price_val = float(price) if price is not None else None
                except Exception:
                    price_val = None

                # price_range
                pr = data.get("price_range") or {}
                if price_val is not None:
                    pr = {"min_value": price_val, "max_value": price_val, "currency": "RUB"}
                else:
                    pr = {
                        "min_value": pr.get("min_value", 0) if isinstance(pr, dict) else 0,
                        "max_value": pr.get("max_value", 0) if isinstance(pr, dict) else 0,
                        "currency": "RUB",
                    }

                # поля строки <=200
                def _clip(v):
                    return v[:200] if isinstance(v, str) else v

                data = {
                    "summary": _clip(data.get("summary", "")),
                    "recommendation": _clip(data.get("recommendation", "")),
                    "risk_score": float(data.get("risk_score", 0) or 0),
                    "price_range": pr,
                    "pros": [_clip(x) for x in (data.get("pros") or [])][:4],
                    "cons": [_clip(x) for x in (data.get("cons") or [])][:4],
                    "checks": [_clip(x) for x in (data.get("checks") or [])][:4],
                }

                return json.dumps(_sanitize_jsonable(data), ensure_ascii=False, allow_nan=False)
        except Exception:
            pass  # если не json, отдадим как есть

        return raw

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
