from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class AnalyzeRequest(BaseModel):
    property_data: Dict[str, Any] = Field(..., description="Raw property JSON")

    @validator("property_data")
    def validate_property_data(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("property_data must not be empty")
        if not isinstance(value, dict):
            raise ValueError("property_data must be a JSON object")

        property_type = value.get("type") or value.get("property_type")
        location = value.get("location") or value.get("address") or value.get("city")
        area = value.get("area") or value.get("square_meters") or value.get("sqft")

        if not property_type:
            raise ValueError("property_data must include 'type' or 'property_type'")
        if not location:
            raise ValueError("property_data must include 'location', 'address', or 'city'")

        if area is not None:
            try:
                numeric_area = float(area)
            except (TypeError, ValueError):
                raise ValueError("area must be numeric when provided")
            if numeric_area <= 0:
                raise ValueError("area must be > 0 when provided")

        return value


class PriceRange(BaseModel):
    min_value: float = Field(..., ge=0)
    max_value: float = Field(..., ge=0)
    currency: str = Field(..., min_length=1, max_length=10)

    @root_validator
    def check_order(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        min_value = values.get("min_value")
        max_value = values.get("max_value")
        if min_value is not None and max_value is not None and max_value < min_value:
            raise ValueError("max_value must be greater than or equal to min_value")
        return values


class ReportModel(BaseModel):
    summary: str
    recommendation: str
    risk_score: float = Field(..., ge=0, le=1, description="0 (low risk) to 1 (high risk)")
    price_range: PriceRange
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    raw_notes: Optional[str] = None

    @validator("pros", "cons", each_item=True)
    def clean_items(cls, value: str) -> str:
        return value.strip()


class HistoryRecord(BaseModel):
    id: int
    created_at: datetime
    recommendation: Optional[str]
    risk_score: Optional[float]
    report: Optional[ReportModel]
    raw_response: Optional[str]
    input_json: Dict[str, Any]


class CompareObject(BaseModel):
    id: str = Field(..., min_length=1, max_length=50, description="Уникальный идентификатор объекта")
    price: float = Field(..., gt=0)
    area: float = Field(..., gt=0)
    district: str = Field(..., min_length=1, max_length=200)
    year: Optional[int] = Field(None, ge=1800, le=2100)
    condition: Optional[str] = Field(None, max_length=200)
    floor: Optional[int] = Field(None, ge=0)
    floors_total: Optional[int] = Field(None, ge=0)
    raw_text: Optional[str] = None

    @root_validator
    def validate_floor(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        floor = values.get("floor")
        total = values.get("floors_total")
        if floor is not None and total is not None and floor > total:
            raise ValueError("floor cannot exceed floors_total")
        return values


class CompareRequest(BaseModel):
    objects: List[CompareObject] = Field(..., min_items=1, max_items=3)


class CompareResponse(BaseModel):
    winner_id: str
    score: Dict[str, int]
    reasons: List[str]
    risks: List[str]
    checks: List[str]
    summary: str
