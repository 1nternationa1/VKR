import asyncio
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import random
import re
import time
import html
from typing import Any, Dict, Optional, List
from urllib.parse import urljoin
from uuid import uuid4

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

from .db import fetch_history, fetch_history_item, init_db, log_history
from .providers import (
    AIProvider,
    CloudProvider,
    GeminiProvider,
    LocalStubProvider,
    get_provider,
    _filter_property_data,
)
from .proxy import get_httpx_proxies, get_playwright_proxy
from .schemas import AnalyzeRequest, CompareObject, CompareRequest, CompareResponse, ReportModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    root_logger = logging.getLogger()
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    has_file_handler = any(
        isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == LOG_FILE_PATH
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", "1048576")),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "3")),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_env_file() -> None:
    """Load environment variables from .env if they are not already set."""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        # Best-effort load; ignore failures to keep startup resilient.
        return


load_env_file()
configure_logging()

app = FastAPI(
    title="Real Estate Valuation MVP",
    description="Экспертная оценка недвижимости через LLM",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _parse_llm_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    raise ValueError(f"LLM returned non-JSON response: {raw[:200]}")


def _append_trace(trace: List[Dict[str, Any]], stage: str, message: str, **meta: Any) -> None:
    entry: Dict[str, Any] = {
        "time": time.strftime("%H:%M:%S"),
        "stage": stage,
        "message": message,
    }
    if meta:
        entry["meta"] = meta
    trace.append(entry)


def _render_report_text(report: Any) -> str:
    """Build a human-readable note from a structured report or raw text."""
    if isinstance(report, str):
        return report.strip()
    if not isinstance(report, dict):
        return str(report)

    parts: list[str] = []
    summary = report.get("summary")
    if summary:
        parts.append(str(summary).strip())

    pros = report.get("pros") or []
    if pros:
        parts.append("Сильные стороны:")
        parts.extend([f"- {item}" for item in pros])

    cons = report.get("cons") or []
    if cons:
        parts.append("Риски и проверки:")
        parts.extend([f"- {item}" for item in cons])

    checks = report.get("checks") or []
    if checks:
        parts.append("Что проверить:")
        parts.extend([f"- {item}" for item in checks])

    price_range = report.get("price_range") or {}
    if price_range:
        min_value = price_range.get("min_value")
        max_value = price_range.get("max_value")
        currency = price_range.get("currency") or ""
        if min_value is not None and max_value is not None:
            parts.append(f"Диапазон цены: {min_value}–{max_value} {currency}".strip())

    recommendation = report.get("recommendation")
    if recommendation:
        parts.append(f"Рекомендация: {recommendation}")

    raw_notes = report.get("raw_notes")
    if raw_notes:
        parts.append(f"Дополнительно: {raw_notes}")

    return "\n".join(parts).strip()


def _format_kv_text(data: Any) -> str:
    if isinstance(data, dict):
        return "\n".join(f"- {key}: {value}" for key, value in data.items())
    if isinstance(data, list):
        return "\n".join(f"- {item}" for item in data)
    return str(data)


def _estimate_price_position(property_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute price per m² and rough comparison vs baseline city medians."""
    price = property_data.get("price")
    area = property_data.get("area")
    if not price or not area:
        return {}
    try:
        price = float(price)
        area = float(area)
    except (TypeError, ValueError):
        return {}
    if area <= 0:
        return {}

    ppm = price / area
    location_text = (
        property_data.get("location")
        or property_data.get("address")
        or property_data.get("city")
        or ""
    )
    city = property_data.get("city") or _infer_city(location_text)
    baselines = {
        "Москва": 350_000,
        "Санкт-Петербург": 250_000,
        "Екатеринбург": 170_000,
        "Новосибирск": 150_000,
        "Казань": 180_000,
    }
    baseline = baselines.get(city or "", 160_000)
    delta_pct = (ppm - baseline) / baseline * 100
    if delta_pct <= -10:
        verdict = "Ниже среднего по рынку"
    elif delta_pct <= 10:
        verdict = "В диапазоне рынка"
    else:
        verdict = "Дороже среднего по рынку"
    return {
        "city": city or "—",
        "price_per_m2": round(ppm),
        "baseline_per_m2": baseline,
        "delta_percent": round(delta_pct, 1),
        "position": verdict,
    }


def _has_any(text: str, keywords: list[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _compute_risk(property_data: Dict[str, Any], price_meta: Dict[str, Any]) -> tuple[float, list[str]]:
    """
    Эвристический риск (0-1) + причины. Покрывает отсутствие данных, год/этаж, цену vs рынок,
    шум/промзону, обременения в тексте, перепланировку и состояние.
    """
    score = 0.05
    reasons: list[str] = []

    desc = "\n".join(
        str(x)
        for x in (
            property_data.get("description"),
            property_data.get("notes"),
            property_data.get("source_text"),
        )
        if x
    )

    # Полнота
    if not property_data.get("price"):
        score += 0.07
        reasons.append("Нет цены")
    if not property_data.get("area"):
        score += 0.05
        reasons.append("Нет площади")
    if not property_data.get("year"):
        score += 0.04
        reasons.append("Нет года постройки")
    if not (property_data.get("address") or property_data.get("location")):
        score += 0.05
        reasons.append("Нет адреса/локации")

    # Год и этаж
    year = property_data.get("year")
    if year:
        year = int(year)
        if year <= 1975:
            score += 0.15
            reasons.append(f"Дом {year} (старый фонд)")
        elif year <= 1990:
            score += 0.08
            reasons.append(f"Дом {year} (старше 1990)")
        elif year <= 2010:
            score += 0.04
        elif year > 2015:
            score -= 0.05

    floor = property_data.get("floor")
    floors_total = property_data.get("floors_total")
    if floor and floors_total and floor in (1, floors_total):
        score += 0.05
        reasons.append("1-й/последний этаж")

    # Материал (если придёт из парсера)
    material = (property_data.get("material") or "").lower()
    if material in ("панель", "дерево"):
        score += 0.05
        reasons.append(f"Материал: {material}")

    # Цена vs рынок
    delta = price_meta.get("delta_percent")
    if delta is not None:
        try:
            delta_f = float(delta)
            if delta_f < -15:
                score += 0.12
                reasons.append(f"Цена {delta_f:.0f}% ниже рынка")
            elif abs(delta_f) <= 15:
                score -= 0.02
            elif delta_f > 25:
                score += 0.04
                reasons.append(f"Цена {delta_f:.0f}% выше рынка")
        except Exception:
            pass

    # Транспорт/шум по ключевым словам
    if not property_data.get("metro"):
        score += 0.05
        reasons.append("Нет данных о метро/транспорте")
    if _has_any(desc, ["магистрал", "жд", "ж/д", "шум", "аэропорт", "промзона", "пром-зона"]):
        score += 0.03
        reasons.append("Шум/магистраль/ЖД рядом")

    # Юр. ключи / обременения
    if _has_any(desc, ["ипотек", "опек", "несовершеннолет", "залог", "арест", "долг", "обремен"]):
        score += 0.10
        reasons.append("В тексте: ипотека/опека/долги")
    if delta is not None:
        try:
            delta_f = float(delta)
            if delta_f < -15 and not _has_any(desc, ["ипотек", "опек", "несовершеннолет", "залог", "арест", "долг", "обремен"]):
                score += 0.05
                reasons.append("Цена сильно ниже без явных причин")
        except Exception:
            pass

    # Переплан
    if _has_any(desc, ["переплан", "узакон", "самовол"]):
        score += 0.05
        reasons.append("Есть перепланировка/узаконение")

    # Состояние
    if _has_any(desc, ["требует ремонта", "плохое состояние", "без ремонта"]):
        score += 0.05
        reasons.append("Требуется ремонт")
    if _has_any(desc, ["свежий ремонт", "евроремонт", "капремонт", "после ремонта"]):
        score -= 0.03

    # Лифт (если данные появятся)
    if floors_total and floors_total > 5 and property_data.get("has_elevator") is False:
        score += 0.03
        reasons.append("Высокий дом без лифта")

    score = max(0.0, min(1.0, score))
    return score, reasons
DOC_CHECKLIST = [
    "Выписка из ЕГРН (права, обременения, история переходов). Заказать самостоятельно через Госуслуги/МФЦ.",
    "Правоустанавливающий документ (ДКП/дарение/наследство/приватизация) — продавец в нём совпадает с ЕГРН.",
    "Согласие супруга/совладельцев; нотариальные доверенности — проверить срок и полномочия.",
    "Справка о зарегистрированных (форма №9/Выписка о прописанных) — отсутствие несовершеннолетних и недееспособных.",
    "Технический паспорт/БТИ и фактическая планировка без самовольных перепланировок.",
    "Справка об отсутствии долгов по ЖКУ, капремонту, электроэнергии; акт сверки при передаче.",
    "Если ипотека/залоги — согласие банка, закладная; расчёты только через эскроу/аккредитив.",
]

TECH_CHECKLIST = [
    "Осмотр инженерии: стояки, запорная арматура, отсутствие протечек и запахов в санузлах.",
    "Электрика: щиток, сечение кабеля, состояние розеток, УЗО/автоматы.",
    "Окна/балкон: герметичность, отсутствие конденсата и продуваний.",
    "Шум и вибрации: лифт, мусоропровод, дороги/бар/трамвай рядом.",
    "Доступность: лифт для колясок/грузовой, ширина коридоров, парковка во дворе.",
    "Тепло/вентиляция: температура в квартире, тяга в вытяжках.",
]

AREA_CHECKLIST = [
    "Транспорт: оцените время до метро/МЦД/остановок и пробки в час пик.",
    "Инфраструктура: школа/сад/поликлиника/магазины в 10–15 мин пешком.",
    "Экология и шум: есть ли магистраль, ЖД/аэропорт, промзоны или свалки рядом.",
    "Двор: благоустройство, освещение, парковка, безопасность (камеры/консьерж).",
    "Дом: год и материал, капремонт/реновация, состояние подъезда и кровли.",
]


def _build_extended_insights(property_data: Dict[str, Any], price_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сформировать расширенные подсказки: юрпроверка, техосмотр, район, шаги сделки, комментарий рынка.
    """
    steps = [
        "Соберите пакет документов от продавца и закажите независимую выписку ЕГРН.",
        "Проверьте историю объекта, перепланировки и долги, при необходимости — юрист/техэксперт.",
        "Зафиксируйте цену и состояние в предварительном договоре/авансе, используйте безопасные расчёты (аккредитив/эскроу).",
        "Подготовьте ДКП/Ипотечные документы, подпишите у нотариуса при долях/опеке/доверенности.",
        "Передайте квартиру по акту с фиксацией показаний счётчиков и состояния, подайте на регистрацию в Росреестр.",
    ]

    market_note_parts: list[str] = []
    if price_meta:
        ppm = price_meta.get("price_per_m2")
        base = price_meta.get("baseline_per_m2")
        delta = price_meta.get("delta_percent")
        city = price_meta.get("city")
        if ppm and base and delta is not None:
            trend = "ниже среднего" if delta < -5 else "в рынке" if abs(delta) <= 5 else "выше среднего"
            market_note_parts.append(
                f"Цена за м² ≈ {ppm:,.0f} ₽ против медианы города {base:,.0f} ₽ ({delta:+.1f}%, {trend})."
            )
        if city:
            market_note_parts.append(f"Локация: {city}.")

    if property_data.get("year"):
        year = int(property_data["year"])
        if year >= 2015:
            market_note_parts.append("Дом свежий (2015+), ниже риски капитального ремонта.")
        elif year < 1975:
            market_note_parts.append("Старый фонд — проверьте капремонт, коммуникации и перекрытия.")

    if property_data.get("floor") and property_data.get("floors_total"):
        floor = property_data["floor"]
        total = property_data["floors_total"]
        if floor in (1, total):
            market_note_parts.append("1-й/последний этаж — учтите тепло/шум/протечки и торгуйтесь.")

    return {
        "documents": DOC_CHECKLIST,
        "tech": TECH_CHECKLIST,
        "area": AREA_CHECKLIST,
        "steps": steps,
        "market_note": " ".join(market_note_parts) or None,
    }


def _normalize_property_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort cleanup for incoming property data to avoid validation errors."""
    cleaned = dict(data or {})
    # Defaults for type and location
    if not cleaned.get("type") and cleaned.get("property_type"):
        cleaned["type"] = cleaned["property_type"]
    cleaned.setdefault("type", "Квартира")
    location = cleaned.get("location") or cleaned.get("address") or cleaned.get("city")
    if not location:
        desc = cleaned.get("description") or cleaned.get("source_text")
        if desc:
            location = str(desc)[:80] + ("..." if len(str(desc)) > 80 else "")
        else:
            location = "Не указан"
    cleaned["location"] = location
    # Save city for heuristics
    if cleaned.get("city") and cleaned["city"] not in cleaned.get("location", ""):
        cleaned["location"] = f"{cleaned['location']} ({cleaned['city']})"
    if cleaned.get("metro") and "метро" not in cleaned.get("location", "").lower():
        cleaned["location"] = f"{cleaned['location']}, метро {cleaned['metro']}"

    # Normalize numeric-like fields
    num_fields = ["price", "area", "rooms", "floor", "floors_total", "year"]
    for key in num_fields:
        val = cleaned.get(key)
        if isinstance(val, str):
            norm = val.replace("\u00a0", " ").replace(" ", "").replace(",", ".")
            try:
                cleaned[key] = float(norm) if key in ("area",) else int(float(norm))
            except (ValueError, TypeError):
                cleaned.pop(key, None)
        elif val is None:
            cleaned.pop(key, None)
    # Trim verbose text fields to prevent "input too long" errors downstream.
    max_len = int(os.getenv("MAX_TEXT_FIELD_LEN", "2000"))
    cleaned = _trim_text_fields(cleaned, limit=max_len)
    return cleaned


def _infer_city(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    if "москва" in lower or "moscow" in lower or "мск" in lower:
        return "Москва"
    if "санкт" in lower or "питер" in lower or "spb" in lower or "sankt" in lower:
        return "Санкт-Петербург"
    if "екатеринбург" in lower:
        return "Екатеринбург"
    if "новосибир" in lower:
        return "Новосибирск"
    if "казань" in lower:
        return "Казань"
    return None


def _html_to_text(content: str) -> str:
    """Rudimentary HTML to text conversion for fetched listings."""
    # Remove scripts/styles
    content = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    content = re.sub(r"<[^>]+>", " ", content)
    # Unescape entities and normalize whitespace
    content = html.unescape(content)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def _looks_like_listing_block(content: str) -> bool:
    """Detect Avito/r.jina anti-bot pages so we don't analyze them as listings."""
    if not content:
        return False
    lower = content.lower()
    markers = (
        "доступ ограничен: проблема с ip",
        "target url returned error 429",
        "too many requests",
        "нажмите на кнопку продолжить",
        "иногда такое случается",
        "что не так с ip",
        "решения капчи",
        "captcha",
    )
    return any(marker in lower for marker in markers)


def _is_truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _is_avito_url(url: str) -> bool:
    lower = (url or "").lower()
    return "avito.ru" in lower or "m.avito.ru" in lower


def _playwright_storage_state_path() -> str:
    return os.getenv(
        "LISTING_PLAYWRIGHT_STORAGE_STATE",
        os.path.join(BASE_DIR, "data", "playwright_avito_state.json"),
    )


def _trim_text_fields(data: Dict[str, Any], limit: int = 3000) -> Dict[str, Any]:
    """Cap long free-text fields to avoid model/input limits."""
    for key in ("description", "source_text", "notes"):
        val = data.get(key)
        if isinstance(val, str) and len(val) > limit:
            data[key] = val[:limit]
    return data


def _extract_image_urls(html_content: str, base_url: str, limit: int = 6) -> List[str]:
    """Extract image URLs from HTML."""
    urls: List[str] = []
    for match in re.findall(r'<img[^>]+(?:src|data-src|data-original)="([^"]+)"', html_content, flags=re.IGNORECASE):
        if match.startswith("data:"):
            continue
        full_url = urljoin(base_url, match)
        if full_url.startswith(("http://", "https://")) and full_url not in urls:
            urls.append(full_url)
        if len(urls) >= limit:
            break
    return urls


def _parse_meta_tags(raw_html: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Extract meta tags into two lookup dicts:
    - meta_by_prop: property/itemprop -> [content...]
    - meta_by_name: name -> [content...]
    """
    meta_by_prop: dict[str, list[str]] = {}
    meta_by_name: dict[str, list[str]] = {}

    for tag in re.finditer(r"<meta\s+[^>]*?>", raw_html, flags=re.IGNORECASE):
        attrs = dict(
            (k.lower(), v)
            for k, v in re.findall(r'([a-zA-Z0-9:_-]+)\s*=\s*["\'](.*?)["\']', tag.group(0))
        )
        if not attrs:
            continue
        content = attrs.get("content") or attrs.get("value")
        if not content:
            continue
        prop = attrs.get("property") or attrs.get("itemprop")
        name = attrs.get("name")
        if prop:
            meta_by_prop.setdefault(prop.lower(), []).append(content)
        if name:
            meta_by_name.setdefault(name.lower(), []).append(content)

    return meta_by_prop, meta_by_name


def _extract_listing_from_meta(raw_html: str) -> Dict[str, Any]:
    """Best-effort extraction using only meta/link tags (works on saved Avito pages)."""
    meta_by_prop, meta_by_name = _parse_meta_tags(raw_html)

    def _get_prop(key: str) -> Optional[str]:
        vals = meta_by_prop.get(key.lower())
        return vals[0] if vals else None

    def _get_name(key: str) -> Optional[str]:
        vals = meta_by_name.get(key.lower())
        return vals[0] if vals else None

    images: List[str] = []
    for val in meta_by_prop.get("og:image", []):
        if val not in images:
            images.append(val)
    # image_src link fallback
    for match in re.findall(r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']', raw_html, flags=re.IGNORECASE):
        if match not in images:
            images.append(match)

    data: Dict[str, Any] = {
        "title": _get_prop("og:title") or _get_name("mrc__share_title") or _get_name("title"),
        "description": _get_name("description") or _get_prop("og:description"),
        "url": _get_prop("og:url"),
        "price": _get_prop("product:price:amount"),
        "currency": _get_prop("product:price:currency"),
        "seller": _get_prop("vk:seller_name") or _get_name("vk:seller_name"),
        "locale": _get_prop("og:locale"),
        "country": _get_prop("og:country-name"),
        "images": images,
        "image_alts": meta_by_prop.get("og:image:alt", []),
    }
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def _extract_structured_listing(raw_html: str, url: str) -> Dict[str, Any]:
    """Extract key fields from listing HTML using meta tags (Avito-saved pages)."""
    data: Dict[str, Any] = {}

    def _to_int(val: Any) -> Optional[int]:
        if val is None:
            return None
        try:
            if isinstance(val, (int, float)):
                return int(val)
            norm = str(val).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
            return int(float(norm))
        except Exception:
            return None

    def _to_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            if isinstance(val, (int, float)):
                return float(val)
            norm = str(val).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
            return float(norm)
        except Exception:
            return None

    # Only meta-tag driven extraction (robust for saved Avito HTML)
    meta_fields = _extract_listing_from_meta(raw_html)
    for key, value in meta_fields.items():
        if key == "price":
            num = _to_int(value)
            if num:
                data["price"] = num
            continue
        if key in ("images", "image_alts"):
            data[key] = value
            continue
        data.setdefault(key, value)

    # Fallback: price from visible text if meta price missing
    if "price" not in data:
        m_price = re.search(r"([0-9][0-9\s]{3,})\s*(?:₽|руб)", raw_html, flags=re.IGNORECASE)
        if m_price:
            num = _to_int(m_price.group(1))
            if num:
                data["price"] = num

    # Numbers from meta title/description like "1-к. квартира, 43 м², 5/9 эт."
    meta_text = " ".join(
        str(x)
        for x in (
            data.get("title"),
            data.get("description"),
        )
        if x
    )
    if meta_text:
        if "area" not in data:
            m_area = re.search(r"(\d+(?:[\.,]\d+)?)\s*м²", meta_text, flags=re.IGNORECASE)
            if m_area:
                area_val = _to_float(m_area.group(1))
                if area_val:
                    data["area"] = area_val
        if "rooms" not in data:
            m_rooms = re.search(r"(\d+)\s*[-–]?\s*к\.?\s*кв", meta_text, flags=re.IGNORECASE)
            if m_rooms:
                rooms_val = _to_int(m_rooms.group(1))
                if rooms_val is not None:
                    data["rooms"] = rooms_val
            elif re.search(r"студия", meta_text, flags=re.IGNORECASE):
                data["rooms"] = 0
        if "floor" not in data or "floors_total" not in data:
            m_floor = re.search(r"(\d+)\s*/\s*(\d+)\s*эт", meta_text, flags=re.IGNORECASE)
            if m_floor:
                floor_val = _to_int(m_floor.group(1))
                floors_total_val = _to_int(m_floor.group(2))
                if floor_val is not None:
                    data.setdefault("floor", floor_val)
                if floors_total_val is not None:
                    data.setdefault("floors_total", floors_total_val)

    # City from URL if not found
    if "city" not in data and url:
        m = re.match(r"https?://[^/]+/([^/]+)/", url)
        if m:
            slug = m.group(1)
            if slug.lower() in ("moskva", "msk", "mjk", "moscow"):
                data["city"] = "Москва"
            elif slug.lower() in ("sankt-peterburg", "spb"):
                data["city"] = "Санкт-Петербург"
            else:
                data["city"] = slug.replace("-", " ")

    return data


def _extract_address_metro_from_text(text: str) -> Dict[str, Any]:
    """
    Best-effort парсинг адреса и метро из plain-text объявления.
    Ищем паттерны вида "Москва, ул. Сергея Эйзенштейна, 6" и "Ботанический сад 16–20 мин."
    """
    if not text:
        return {}
    out: Dict[str, Any] = {}

    # Адрес: город + улица + дом
    addr_regex = re.compile(
        r"(?:г\.\s*)?([А-ЯЁ][а-яёA-Za-z\-\\s]+),\s*"
        r"(ул\.|улица|проспект|пр-кт|шоссе|ш\.|пер\.|переулок|бульвар|бул\.|пл\.|площадь|набережная|наб\.)\s*"
        r"([^,\n]+?)\s*,\s*(\d+[А-Яа-я0-9\/\-]*)",
        re.IGNORECASE,
    )
    m_addr = addr_regex.search(text)
    if m_addr:
        city = m_addr.group(1).strip()
        street_type = m_addr.group(2).strip()
        street = m_addr.group(3).strip()
        house = m_addr.group(4).strip()
        out["city"] = city
        out["address"] = f"{city}, {street_type} {street}, {house}".strip()

    # Метро и время пешком
    metro_regex = re.compile(r"([А-ЯЁA-Za-z0-9\-\s\.]+?)\s+(\d+)[–-](\d+)\s*мин", re.IGNORECASE)
    metro_matches = metro_regex.findall(text)
    if metro_matches:
        metro_list = []
        for name, t_min, t_max in metro_matches:
            metro_list.append(
                {"name": name.strip(), "time_min": int(t_min), "time_max": int(t_max)}
            )
        # Берём ближайшее
        metro_sorted = sorted(metro_list, key=lambda x: x["time_min"])
        best = metro_sorted[0]
        out.setdefault("metro", best["name"])
        out.setdefault("metro_time_min", best["time_min"])
        out.setdefault("metro_time_max", best["time_max"])
        out["metro_list"] = metro_sorted

    return out


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


def _validate_property_json(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc
    try:
        validated = AnalyzeRequest(property_data=data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return validated.property_data


async def _process_report(property_data: Dict[str, Any], provider: AIProvider, request_id: str) -> Dict[str, Any]:
    raw_response: Optional[str] = None
    parsed_report: Optional[Dict[str, Any]] = None
    report_text: str = ""
    price_meta: Dict[str, Any] = _estimate_price_position(property_data)
    insights = _build_extended_insights(property_data, price_meta)
    debug_trace: List[Dict[str, Any]] = []
    provider_name = type(provider).__name__

    logger.info("[%s] analyze started provider=%s", request_id, provider_name)
    _append_trace(
        debug_trace,
        "analyze_started",
        "Запрос принят сервером.",
        provider=provider_name,
        input_keys=sorted(property_data.keys()),
    )

    # Extra safety: drop лишние поля и длинные тексты до вызова провайдера,
    # чтобы не получить 400 "Invalid JSON" из-за объёмного body.
    safe_input = _trim_text_fields(_filter_property_data(property_data), limit=800)
    _append_trace(
        debug_trace,
        "input_normalized",
        "Данные подготовлены для анализа.",
        safe_input_keys=sorted(safe_input.keys()),
    )

    # Эвристический риск, чтобы был даже без LLM
    risk_score_heur, risk_reasons = _compute_risk(property_data, price_meta)
    _append_trace(
        debug_trace,
        "risk_precomputed",
        "Эвристический риск рассчитан.",
        risk_score=risk_score_heur,
        reasons_count=len(risk_reasons),
    )

    try:
        raw_response = await provider.generate_report(safe_input, trace=debug_trace, request_id=request_id)
        report_text = (raw_response or "").strip()
        logger.info("[%s] provider returned raw response chars=%s", request_id, len(report_text))
        try:
            parsed_candidate = _parse_llm_json(raw_response)
            if isinstance(parsed_candidate, dict):
                # Сливаем риск: консервативно берём max эвристики и LLM
                if "risk_score" in parsed_candidate:
                    try:
                        parsed_candidate["risk_score"] = max(
                            float(parsed_candidate["risk_score"] or 0), risk_score_heur
                        )
                    except Exception:
                        parsed_candidate["risk_score"] = risk_score_heur
                report_model = ReportModel.parse_obj(parsed_candidate)
                parsed_report = report_model.dict()
                report_text = _render_report_text(parsed_report)
                _append_trace(debug_trace, "report_parsed", "Ответ LLM успешно распознан как JSON-отчёт.")
        except Exception:
            # Treat response as plain text if it is not JSON-shaped.
            parsed_report = None
            _append_trace(debug_trace, "report_parse_failed", "Ответ LLM не удалось распарсить как JSON.")
    except Exception as exc:
        logger.exception("[%s] LLM response handling failed, falling back to stub", request_id)
        _append_trace(
            debug_trace,
            "provider_failed",
            "Основной LLM-запрос завершился ошибкой, включаем fallback.",
            error=str(exc),
        )
        fallback = LocalStubProvider()
        raw_response = f"Fallback after error: {exc}"
        try:
            stub_resp = await fallback.generate_report(property_data, trace=debug_trace, request_id=request_id)
            raw_response = f"{raw_response}\n{stub_resp}"
            parsed_report = None
            report_text = stub_resp
            _append_trace(debug_trace, "fallback_ready", "Fallback-ответ сформирован локально.")
        except Exception:
            _append_trace(debug_trace, "fallback_failed", "Fallback тоже завершился ошибкой.")
            raise ValueError(f"LLM response handling failed: {exc}", raw_response, debug_trace) from exc

    # Если LLM не вернул риск — подставляем эвристику
    def _fallback_price_range() -> Dict[str, Any]:
        price = property_data.get("price")
        if price:
            try:
                p = float(price)
                return {"min_value": p, "max_value": p, "currency": "RUB"}
            except Exception:
                pass
        return {"min_value": 0, "max_value": 0, "currency": "RUB"}

    if parsed_report is None:
        parsed_report = {
            "risk_score": risk_score_heur,
            "summary": report_text or "",
            "recommendation": "",
            "price_range": price_meta.get("price_range") or _fallback_price_range(),
            "pros": [],
            "cons": risk_reasons,
            "checks": [],
        }
    else:
        if parsed_report.get("risk_score") is None:
            parsed_report["risk_score"] = risk_score_heur
        if not parsed_report.get("price_range"):
            parsed_report["price_range"] = price_meta.get("price_range") or _fallback_price_range()

    parsed_report.setdefault("risk_reasons", risk_reasons)

    log_history(property_data, parsed_report or report_text, raw_response, mode=os.getenv("AI_MODE", "cloud"))
    logger.info("[%s] analyze completed", request_id)
    _append_trace(
        debug_trace,
        "analyze_completed",
        "Оценка завершена.",
        report_ready=bool(parsed_report),
        log_file=LOG_FILE_PATH,
    )

    return {
        "report": parsed_report,
        "report_text": report_text,
        "raw": raw_response,
        "price_meta": price_meta,
        "insights": insights,
    }


def _score_object(obj: CompareObject) -> float:
    """Lightweight heuristic scoring without external AI."""
    price_per_meter = obj.price / obj.area
    base_score = 100 - min(price_per_meter / 1000 * 10, 60)
    base_score += min(max(obj.area - 45, 0) / 10 * 2.5, 8)

    if obj.year:
        base_score += min((obj.year - 2000) / 5, 6)
    if obj.condition:
        base_score += 2.5
    if obj.floor is not None and obj.floors_total:
        if obj.floor in (1, obj.floors_total):
            base_score -= 2
        else:
            base_score += 1

    return max(25.0, min(95.0, base_score))


def _build_compare_response(objects: list[CompareObject]) -> CompareResponse:
    scores: Dict[str, int] = {obj.id: round(_score_object(obj)) for obj in objects}
    winner_id = max(scores, key=scores.get)
    price_per_meter = {obj.id: obj.price / obj.area for obj in objects}
    best_ppm_id = min(price_per_meter, key=price_per_meter.get)
    best_ppm_value = price_per_meter[best_ppm_id]
    largest_area_obj = max(objects, key=lambda o: o.area)

    reasons = [
        f"Лучшее соотношение цена за метр у {best_ppm_id} (~{best_ppm_value:,.0f} за м²).",
        f"{largest_area_obj.id} предлагает максимальную площадь ({largest_area_obj.area:g} м²).",
        f"{winner_id} имеет самый высокий интегральный балл ({scores[winner_id]}).",
    ]

    recent = [o for o in objects if o.year and o.year >= 2010]
    if recent:
        recent_ids = ", ".join(sorted({o.id for o in recent}))
        reasons.append(f"Свежая постройка: {recent_ids} (2010+).")

    risks = [
        "Проверьте юридическую чистоту и обременения по всем объектам.",
        "Уточните состояние коммуникаций и скрытых дефектов при осмотре.",
        "Сверьте фактическую площадь и планировку с документами.",
    ]

    checks = [
        "Запросите выписку ЕГРН и историю переходов права.",
        "Сравните рыночные аналоги в районе для верификации цены.",
        "Проверьте подъезд, двор и шумовой фон в разное время суток.",
    ]

    summary = (
        f"{winner_id} выглядит выгоднее по цене/метражу и базовым параметрам. "
        "Уточните состояние, документы и стоимость владения перед решением."
    )

    return CompareResponse(
        winner_id=winner_id,
        score=scores,
        reasons=reasons,
        risks=risks,
        checks=checks,
        summary=summary,
    )


@app.post("/analyze")
async def analyze(
    request: Request,
    property_json: Optional[str] = Form(None),
) -> Any:
    provider = get_provider()
    request_id = uuid4().hex[:8]

    is_form = property_json is not None
    property_data: Optional[Dict[str, Any]] = None

    if property_json:
        try:
            property_data = _validate_property_json(property_json)
        except HTTPException as exc:
            context = {"request": request, "result": None, "error": exc.detail, "sample_json": property_json}
            if is_form:
                return templates.TemplateResponse("index.html", context, status_code=exc.status_code)
            raise
    else:
        try:
            raw_body = await request.json()
        except Exception:
            raw_body = {}
        if isinstance(raw_body, dict):
            property_data = raw_body.get("property_data") or raw_body

    # Graceful fallback: allow empty input, but normalize to defaults.
    property_data = _normalize_property_data(property_data or {})
    try:
        property_data = AnalyzeRequest(property_data=property_data).property_data
    except Exception as exc:
        detail = str(exc)
        if is_form:
            context = {"request": request, "result": None, "error": detail, "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2)}
            return templates.TemplateResponse("index.html", context, status_code=400)
        raise HTTPException(status_code=400, detail=detail) from exc

    try:
        result = await _process_report(property_data, provider, request_id)
    except ValueError as exc:
        error_text = str(exc.args[0])
        raw_response = exc.args[1] if len(exc.args) > 1 else None
        logger.error("[%s] analyze failed with ValueError: %s", request_id, error_text)
        if is_form:
            context = {
                "request": request,
                "result": None,
                "error": error_text,
                "raw_response": raw_response,
                "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2),
            }
            return templates.TemplateResponse("index.html", context, status_code=502)
        return JSONResponse(
            {
                "error": "Не удалось выполнить оценку. Попробуйте ещё раз.",
                "raw_response": None,
            },
            status_code=502,
        )
    except Exception as exc:  # provider errors
        logger.exception("[%s] analyze failed", request_id)
        if is_form:
            context = {
                "request": request,
                "result": None,
                "error": str(exc),
                "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2),
            }
            return templates.TemplateResponse("index.html", context, status_code=502)
        return JSONResponse(
            {
                "error": "Не удалось выполнить оценку. Попробуйте ещё раз.",
            },
            status_code=502,
        )

    if is_form:
        context = {
            "request": request,
            "result": result,
            "error": None,
            "raw_response": result["raw"],
            "report_text": result["report_text"],
            "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2),
        }
        return templates.TemplateResponse("index.html", context)

    return {
        "report": result["report"],
        "report_text": result["report_text"],
        "raw_response": result["raw"],
        "price_meta": result.get("price_meta"),
        "insights": result.get("insights"),
    }


@app.post("/api/analyze", response_model=CompareResponse)
async def api_analyze(payload: CompareRequest) -> CompareResponse:
    if not payload.objects:
        raise HTTPException(status_code=400, detail="objects must include at least one item")

    provider = get_provider()

    # If cloud mode is available, call GPT/Gemini; otherwise fallback to heuristic.
    if isinstance(provider, (CloudProvider, GeminiProvider)):
        try:
            raw = await provider.generate_comparison([obj.dict() for obj in payload.objects])
            parsed = _parse_llm_json(raw)
            return CompareResponse.parse_obj(parsed)
        except Exception as exc:
            logger.exception("LLM comparison failed")
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return _build_compare_response(payload.objects)


async def fetch_listing_data(url: str, include_raw_html: bool = False) -> Dict[str, Any]:
    if not url:
        raise ValueError("URL is required")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    # Прокси включается только в fallback-ветке загрузки объявления.
    fallback_proxies = get_httpx_proxies("LISTING")

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        # Мобильный UA часто проходит защиту Avito
        "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36",
    ]
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": random.choice(
            ["ru-RU,ru;q=0.9,en;q=0.6", "ru, en;q=0.8", "ru-RU,ru;q=0.95,en-US;q=0.5"]
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.avito.ru/",
    }

    async def _fetch_direct(target: str, client_proxies: Optional[Dict[str, str]] = None) -> str:
        async with httpx.AsyncClient(
            timeout=12.0,
            proxies=client_proxies,
            headers=headers,
            trust_env=False,
        ) as client:
            response = await client.get(target, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_jina(target: str) -> str:
        proxy_url = f"https://r.jina.ai/{target}"
        async with httpx.AsyncClient(timeout=15.0, headers=headers, trust_env=False) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_jina_mobile(target: str) -> str:
        mobile = target.replace("https://www.avito.ru", "https://m.avito.ru").replace("http://", "https://")
        proxy_url = f"https://r.jina.ai/{mobile}"
        async with httpx.AsyncClient(timeout=15.0, headers=headers, trust_env=False) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_jina_double(target: str) -> str:
        proxy_url = f"https://r.jina.ai/https://r.jina.ai/{target}"
        async with httpx.AsyncClient(timeout=18.0, headers=headers, trust_env=False) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_textise(target: str) -> str:
        """Фолбэк через textise-dot-iitty / allorigins-класс (через r.jina.ai)."""
        proxy_url = f"https://r.jina.ai/https://r.jina.ai/{target}"
        async with httpx.AsyncClient(timeout=18.0, headers=headers, trust_env=False) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_playwright(target: str, use_proxy: bool = False) -> Optional[str]:
        """
        Последняя попытка: открыть страницу реальным браузером и нажать «Продолжить»
        на капче Avito. Работает только если playwright установлен и доступен Chromium.
        Не бросает исключения, чтобы основная цепочка не падала при отсутствии браузера.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception:
            return None

        headless = _is_truthy_env("LISTING_PLAYWRIGHT_HEADLESS", "false")
        captcha_wait_seconds = int(os.getenv("LISTING_CAPTCHA_WAIT_SECONDS", "90"))
        browser = None
        context = None
        page = None
        try:
            async with async_playwright() as p:
                launch_kwargs: Dict[str, Any] = {"headless": headless}
                channel = os.getenv("LISTING_PLAYWRIGHT_CHANNEL")
                if channel:
                    launch_kwargs["channel"] = channel
                playwright_proxy = get_playwright_proxy("LISTING") if use_proxy else None
                if playwright_proxy:
                    launch_kwargs["proxy"] = playwright_proxy
                browser = await p.chromium.launch(**launch_kwargs)
                context_kwargs: Dict[str, Any] = {
                    "user_agent": headers["User-Agent"],
                    "locale": "ru-RU",
                    "viewport": {"width": 1280, "height": 720},
                }
                storage_state_path = _playwright_storage_state_path()
                if os.path.exists(storage_state_path):
                    context_kwargs["storage_state"] = storage_state_path

                context = await browser.new_context(
                    **context_kwargs,
                )
                page = await context.new_page()
                await page.goto(target, wait_until="domcontentloaded", timeout=20000)

                # Если показана капча Avito, там есть кнопка «Продолжить».
                try:
                    await page.get_by_text("Продолжить").click(timeout=3000)
                    await page.wait_for_timeout(1500)
                except Exception:
                    pass

                content = await page.content()
                if _looks_like_listing_block(content) and not headless:
                    logger.warning(
                        "Avito anti-bot page detected in Playwright; waiting up to %s seconds for manual solve",
                        captcha_wait_seconds,
                    )
                    deadline = time.monotonic() + captcha_wait_seconds
                    while time.monotonic() < deadline:
                        try:
                            await page.get_by_text("Продолжить").click(timeout=1000)
                        except Exception:
                            pass
                        await page.wait_for_timeout(1000)
                        content = await page.content()
                        if not _looks_like_listing_block(content):
                            break
                if context:
                    os.makedirs(os.path.dirname(storage_state_path), exist_ok=True)
                    await context.storage_state(path=storage_state_path)
                return content
        except Exception:
            return None
        finally:
            try:
                if page:
                    await page.close()
            except Exception:
                pass
            try:
                if context:
                    await context.close()
            except Exception:
                pass
            try:
                if browser:
                    await browser.close()
            except Exception:
                pass

    raw_html: Optional[str] = None
    errors: list[str] = []
    # Основной путь идёт без прокси. Прокси включается только в fallback для прямого доступа к объявлению.
    images: List[str] = []
    parsed_fields: Dict[str, Any] = {}
    playwright_first_for_avito = _is_avito_url(url) and _is_truthy_env("LISTING_AVITO_PLAYWRIGHT_FIRST", "true")
    fetch_chain: tuple[tuple[str, Any], ...]
    if playwright_first_for_avito:
        fetch_chain = (
            ("playwright", lambda: _fetch_via_playwright(url)),
            ("jina_mobile", lambda: _fetch_via_jina_mobile(url)),
            ("jina", lambda: _fetch_via_jina(url)),
            ("jina_double", lambda: _fetch_via_jina_double(url)),
            ("textise", lambda: _fetch_via_textise(url)),
            ("direct", lambda: _fetch_direct(url)),
        )
    else:
        fetch_chain = (
            ("jina_mobile", lambda: _fetch_via_jina_mobile(url)),
            ("jina", lambda: _fetch_via_jina(url)),
            ("jina_double", lambda: _fetch_via_jina_double(url)),
            ("textise", lambda: _fetch_via_textise(url)),
            ("direct", lambda: _fetch_direct(url)),
            ("playwright", lambda: _fetch_via_playwright(url)),
        )
    if fallback_proxies:
        fetch_chain += (
            ("playwright_proxy_fallback", lambda: _fetch_via_playwright(url, use_proxy=True)),
            ("direct_proxy_fallback", lambda: _fetch_direct(url, client_proxies=fallback_proxies)),
        )

    for fetcher_name, fetcher in fetch_chain:
        try:
            raw_html = await fetcher()
            if not raw_html:
                raise ValueError("empty body")
            if _looks_like_listing_block(raw_html):
                raise ValueError("listing blocked by anti-bot / captcha")

            if not images:
                try:
                    images = _extract_image_urls(raw_html, url)
                except Exception:
                    images = []
            if not parsed_fields:
                try:
                    parsed_fields = _extract_structured_listing(raw_html, url)
                except Exception:
                    parsed_fields = {}

            break
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:200] if exc.response is not None else str(exc)
            if exc.response is not None and exc.response.status_code in (403, 429):
                body = f"{exc.response.status_code}: Avito вернул защиту (403/429). Скопируйте текст объявления вручную или сохраните страницу и загрузите её."
            errors.append(f"{fetcher_name}: {body}")
        except httpx.RequestError as exc:
            detail = str(exc).strip() or type(exc).__name__
            errors.append(f"{fetcher_name}: {type(exc).__name__} {detail}")
        except ValueError as exc:
            errors.append(f"{fetcher_name}: {exc}")
        await asyncio.sleep(0.8)

    if not raw_html:
        logger.warning("fetch_listing failed for %s: %s", url, "; ".join(errors))
        fallback_msg = (
            "Не удалось загрузить объявление автоматически. "
            "Скопируйте текст и параметры объявления вручную."
        )
        # Возвращаем мягкий ответ, чтобы UI мог продолжить работу без 502.
        result = {"text": "", "images": [], "parsed": {}, "error": fallback_msg, "fetch_errors": errors}
        if include_raw_html:
            result["raw_html"] = None
        return result

    try:
        text = _html_to_text(raw_html)
    except Exception:
        text = ""
    if _looks_like_listing_block(text):
        text = ""

    # Merge images from meta-tags (if present) with scraped <img> list
    meta_images = parsed_fields.get("images")
    if isinstance(meta_images, list) and meta_images:
        merged = meta_images + [img for img in images if img not in meta_images]
        images = merged

    images = [img for img in images[:6] if isinstance(img, str)]
    text_limit = int(os.getenv("FETCH_TEXT_LIMIT", "4000"))
    text = text[:text_limit] if text else ""

    if (not text or len(text) < 40) and parsed_fields.get("description"):
        text = str(parsed_fields["description"])[:text_limit]

    # Дополнительный разбор адреса/метро из текста
    try:
        addr_meta = _extract_address_metro_from_text(text)
        for k, v in (addr_meta or {}).items():
            parsed_fields.setdefault(k, v)
        # Если нашли адрес/город — заполним location для формы
        if addr_meta.get("address"):
            parsed_fields.setdefault("address", addr_meta["address"])
            parsed_fields.setdefault("location", addr_meta["address"])
        elif addr_meta.get("city"):
            parsed_fields.setdefault("location", addr_meta["city"])
    except Exception:
        pass

    if not text or len(text) < 40:
        fallback_msg = (
            "Не удалось надёжно извлечь текст объявления. "
            "Вероятная причина: защита Avito по IP / капча / 429. "
            "Скопируйте описание вручную или пройдите капчу в браузерном fallback."
        )
        result = {
            "text": text,
            "images": images,
            "parsed": parsed_fields,
            "error": fallback_msg,
            "fetch_errors": errors,
        }
        if include_raw_html:
            result["raw_html"] = raw_html
        return result

    result = {"text": text, "images": images, "parsed": parsed_fields, "error": None, "fetch_errors": errors}
    if include_raw_html:
        result["raw_html"] = raw_html
    return result


@app.post("/api/fetch_listing")
async def api_fetch_listing(payload: Dict[str, str]) -> Dict[str, Any]:
    url = (payload.get("url") or "").strip()
    try:
        result = await fetch_listing_data(url, include_raw_html=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result.pop("fetch_errors", None)
    return result


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, limit: int = 20) -> HTMLResponse:
    records = fetch_history(limit=limit)
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "records": records, "limit": limit},
    )


@app.get("/history/{record_id}", response_class=HTMLResponse)
async def history_detail(request: Request, record_id: int) -> HTMLResponse:
    record = fetch_history_item(record_id)
    if not record:
        return templates.TemplateResponse(
            "history_detail.html",
            {"request": request, "record": None, "error": "Запись не найдена."},
            status_code=404,
        )
    output_source = record.get("output_json") or record.get("raw_response")
    output_text = (
        _render_report_text(output_source) if isinstance(output_source, dict) else (str(output_source) if output_source else None)
    )
    input_text = _format_kv_text(record.get("input_json")) if record.get("input_json") else None
    return templates.TemplateResponse(
        "history_detail.html",
        {
            "request": request,
            "record": record,
            "error": None,
            "output_text": output_text,
            "input_text": input_text,
        },
    )


@app.get("/api/history")
async def api_history(limit: int = 20) -> Dict[str, Any]:
    return {"items": fetch_history(limit=limit)}
