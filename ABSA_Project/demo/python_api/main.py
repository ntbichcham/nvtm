"""FastAPI app cho PhoBERT ABSA.

Chạy từ thư mục brand_health_demo:
    python -m uvicorn python_api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .config import load_settings
from .inference import (
    InferenceError,
    PhoBERTABSAPredictor,
    PredictorNotReady,
    SegmenterUnavailable,
)
from .schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse

settings = load_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

predictor = PhoBERTABSAPredictor(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Đang khởi động Python API...")
    try:
        await asyncio.to_thread(predictor.load)
    except Exception:
        # Không làm sập server: /health sẽ trả degraded và /predict trả 503.
        logger.exception("Khởi động mô hình thất bại; API vẫn chạy ở trạng thái degraded.")
    yield
    logger.info("Đang tắt Python API...")


app = FastAPI(
    title="PhoBERT ABSA API",
    description=(
        "API phân tích cảm xúc theo khía cạnh cho UIT-ViSFD. "
        "Kết quả được suy luận từ checkpoint PhoBERT đã huấn luyện."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key không hợp lệ.",
        )


def get_predictor() -> PhoBERTABSAPredictor:
    if not predictor.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình chưa sẵn sàng. Kiểm tra log của Python API.",
        )
    return predictor


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": "PhoBERT ABSA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(**predictor.health())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    payload: PredictRequest,
    request: Request,
    _: None = Depends(verify_api_key),
    model: PhoBERTABSAPredictor = Depends(get_predictor),
) -> PredictResponse:
    request_id = uuid.uuid4().hex
    logger.info(
        "predict request_id=%s review_id=%s chars=%d",
        request_id,
        payload.review_id if payload.review_id is not None else "-",
        len(payload.text),
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                model.predict,
                payload.text,
                payload.include_all_aspects,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("predict timeout request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Mô hình phản hồi quá lâu. Vui lòng thử lại.",
        )
    except PredictorNotReady as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except SegmenterUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bộ tách từ chưa sẵn sàng.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except InferenceError:
        logger.exception("predict inference error request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể phân tích đánh giá. Vui lòng thử lại.",
        )
    except Exception:
        logger.exception("predict unexpected error request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi phân tích đánh giá.",
        )

    logger.info(
        "predict done request_id=%s aspects=%d total_ms=%s",
        request_id,
        len(result["aspects"]),
        result["total_ms"],
    )

    return PredictResponse(
        success=True,
        request_id=request_id,
        model=ModelInfo(**result["model"]),
        text=result["text"],
        segmented_text=result["segmented_text"],
        aspects=result["aspects"],
        all_aspects=result["all_aspects"],
        warnings=result["warnings"],
        segmentation_ms=result["segmentation_ms"],
        inference_ms=result["inference_ms"],
        total_ms=result["total_ms"],
    )