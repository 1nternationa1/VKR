import json
import os
import logging
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import fetch_history, fetch_history_item, init_db, log_history
from .providers import AIProvider, CloudProvider, GeminiProvider, get_provider
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
    error_message: Optional[str] = None

    try:
        raw_response = await provider.generate_report(property_data)
        parsed_report = _parse_llm_json(raw_response)
        report_model = ReportModel.parse_obj(parsed_report)
        parsed_report = report_model.dict()
    except json.JSONDecodeError as exc:
        error_message = f"LLM returned non-JSON response: {exc}"
    except Exception as exc:
        logger.exception("LLM response validation failed")
        error_message = f"LLM response validation failed: {exc}"

    log_history(property_data, parsed_report, raw_response, mode=os.getenv("AI_MODE", "cloud"))

    if error_message:
        raise ValueError(error_message, raw_response)

    return {"report": parsed_report, "raw": raw_response}


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
    analyze_request: Optional[AnalyzeRequest] = Body(None),
) -> Any:
    provider = get_provider()

    is_form = property_json is not None
    if property_json:
        try:
            property_data = _validate_property_json(property_json)
        except HTTPException as exc:
            context = {"request": request, "result": None, "error": exc.detail, "sample_json": property_json}
            if is_form:
                return templates.TemplateResponse("index.html", context, status_code=exc.status_code)
            raise
    elif analyze_request:
        property_data = analyze_request.property_data
    else:
        raise HTTPException(status_code=400, detail="Provide property_json form field or JSON body.")

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
            "sample_json": json.dumps(property_data, ensure_ascii=False, indent=2),
        }
        return templates.TemplateResponse("index.html", context)

    return {"report": result["report"], "raw_response": result["raw"]}


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
    return templates.TemplateResponse(
        "history_detail.html",
        {"request": request, "record": record, "error": None},
    )


@app.get("/api/history")
async def api_history(limit: int = 20) -> Dict[str, Any]:
    return {"items": fetch_history(limit=limit)}
