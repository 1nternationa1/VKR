import asyncio
import json
import os
import logging
import random
import re
import html
from typing import Any, Dict, Optional, List
from urllib.parse import urljoin

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
from .schemas import AnalyzeRequest, CompareObject, CompareRequest, CompareResponse, ReportModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
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
    max_len = int(os.getenv("MAX_TEXT_FIELD_LEN", "3000"))
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


async def _process_report(property_data: Dict[str, Any], provider: AIProvider) -> Dict[str, Any]:
    raw_response: Optional[str] = None
    parsed_report: Optional[Dict[str, Any]] = None
    report_text: str = ""
    price_meta: Dict[str, Any] = _estimate_price_position(property_data)

    # Extra safety: drop лишние поля и длинные тексты до вызова провайдера,
    # чтобы не получить 400 "Invalid JSON" из-за объёмного body.
    safe_input = _trim_text_fields(_filter_property_data(property_data), limit=800)

    try:
        raw_response = await provider.generate_report(safe_input)
        report_text = (raw_response or "").strip()
        try:
            parsed_candidate = _parse_llm_json(raw_response)
            if isinstance(parsed_candidate, dict):
                report_model = ReportModel.parse_obj(parsed_candidate)
                parsed_report = report_model.dict()
                report_text = _render_report_text(parsed_report)
        except Exception:
            # Treat response as plain text if it is not JSON-shaped.
            parsed_report = None
    except Exception as exc:
        logger.exception("LLM response handling failed, falling back to stub")
        fallback = LocalStubProvider()
        raw_response = f"Fallback after error: {exc}"
        try:
            stub_resp = await fallback.generate_report(property_data)
            raw_response = f"{raw_response}\n{stub_resp}"
            parsed_report = None
            report_text = stub_resp
        except Exception:
            raise ValueError(f"LLM response handling failed: {exc}", raw_response) from exc

    log_history(property_data, parsed_report or report_text, raw_response, mode=os.getenv("AI_MODE", "cloud"))

    return {"report": parsed_report, "report_text": report_text, "raw": raw_response, "price_meta": price_meta}


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
        result = await _process_report(property_data, provider)
    except ValueError as exc:
        error_text = str(exc.args[0])
        raw_response = exc.args[1] if len(exc.args) > 1 else None
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
            {"error": error_text, "raw_response": raw_response},
            status_code=502,
        )
    except Exception as exc:  # provider errors
        if is_form:
            context = {
                "request": request,
                "result": None,
                "error": str(exc),
                "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2),
            }
            return templates.TemplateResponse("index.html", context, status_code=502)
        raise HTTPException(status_code=502, detail=str(exc))

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

    return {"report": result["report"], "report_text": result["report_text"], "raw_response": result["raw"]}


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


@app.post("/api/fetch_listing")
async def api_fetch_listing(payload: Dict[str, str]) -> Dict[str, Any]:
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Прокси только для загрузки объявлений (не влияет на другие запросы)
    proxies = {
        "http://": os.getenv("LISTING_HTTP_PROXY") or os.getenv("HTTP_PROXY") or None,
        "https://": os.getenv("LISTING_HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or None,
    }
    if not proxies["http://"] and not proxies["https://"]:
        proxies = None

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    ]
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.avito.ru/",
    }

    async def _fetch_direct(target: str) -> str:
        async with httpx.AsyncClient(timeout=12.0, proxies=proxies, headers=headers) as client:
            response = await client.get(target, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_jina(target: str) -> str:
        proxy_url = f"https://r.jina.ai/{target}"
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    async def _fetch_via_textise(target: str) -> str:
        """Фолбэк через textise-dot-iitty / allorigins-класс (через r.jina.ai)."""
        proxy_url = f"https://r.jina.ai/https://r.jina.ai/{target}"
        async with httpx.AsyncClient(timeout=18.0, headers=headers) as client:
            response = await client.get(proxy_url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    raw_html: Optional[str] = None
    errors: list[str] = []
    # Try proxy first (лучше для заблокированных RU сайтов), затем прямой доступ.
    images: List[str] = []
    parsed_fields: Dict[str, Any] = {}
    fetch_chain = (_fetch_via_jina, _fetch_via_textise, _fetch_direct)
    for fetcher in fetch_chain:
        try:
            raw_html = await fetcher(url)
            if raw_html and not images:
                try:
                    images = _extract_image_urls(raw_html, url)
                except Exception:
                    images = []
            if raw_html and not parsed_fields:
                try:
                    parsed_fields = _extract_structured_listing(raw_html, url)
                except Exception:
                    parsed_fields = {}
            break
        except httpx.HTTPStatusError as exc:
            errors.append(f"{exc.response.status_code}: {exc.response.text[:200]}")
        except httpx.RequestError as exc:
            errors.append(str(exc))
        await asyncio.sleep(0.5)

    if not raw_html:
        logger.warning("fetch_listing failed for %s: %s", url, "; ".join(errors))
        fallback_msg = (
            "Не удалось загрузить объявление автоматически. "
            "Скопируйте текст и параметры объявления вручную."
        )
        # Возвращаем мягкий ответ, чтобы UI мог продолжить работу без 502.
        return {"text": "", "images": [], "parsed": {}, "error": fallback_msg}

    try:
        text = _html_to_text(raw_html)
    except Exception:
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

    if not text or len(text) < 40:
        fallback_msg = (
            "Не удалось надёжно извлечь текст объявления (возможно, защита от ботов или 429). "
            "Скопируйте описание вручную и повторите."
        )
        return {"text": text, "images": images, "parsed": parsed_fields, "error": fallback_msg}

    return {"text": text, "images": images, "parsed": parsed_fields, "error": None}


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
