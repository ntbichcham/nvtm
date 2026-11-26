"""Pydantic schemas cho Python API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Nội dung đánh giá tiếng Việt",
    )
    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Số sao 1..5 nếu có; không dùng làm đầu vào mô hình",
    )
    review_date: date | None = Field(
        default=None,
        description="Ngày đánh giá; chỉ là metadata",
    )
    review_id: int | None = Field(
        default=None,
        ge=1,
        description="ID review phía PHP; chỉ dùng để ghi log",
    )
    include_all_aspects: bool = Field(
        default=False,
        description="Trả thêm cả 10 khía cạnh, kể cả None",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Nội dung đánh giá không được để trống")
        return value


class AspectPrediction(BaseModel):
    code: str
    name_vi: str
    model_index: int
    sentiment: Literal["None", "Negative", "Neutral", "Positive"]
    sentiment_vi: str
    sentiment_index: int
    confidence: float
    confidence_percent: float
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    version: str
    best_epoch: int | None
    max_len: int
    device: str
    segmentation: str


class PredictResponse(BaseModel):
    success: bool = True
    request_id: str
    model: ModelInfo
    text: str
    segmented_text: str
    aspects: list[AspectPrediction]
    all_aspects: list[AspectPrediction] | None = None
    warnings: list[str] = Field(default_factory=list)
    segmentation_ms: float
    inference_ms: float
    total_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    segmenter_ready: bool
    checkpoint_found: bool
    device: str
    max_len: int
    segmenter_backend: str
    model_name: str
    model_version: str | None
    message: str