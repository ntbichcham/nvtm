"""Cấu hình cho Python API.

Tất cả đường dẫn có thể override bằng biến môi trường hoặc file `.env`
đặt cạnh file này.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BRAND_HEALTH_DIR = BASE_DIR.parent
PROJECT_ROOT = BRAND_HEALTH_DIR.parent

DEFAULT_CHECKPOINT = PROJECT_ROOT / "ABSA_Project" / "models" / "best_phobert_absa.pt"
DEFAULT_MODEL_CODE_DIR = PROJECT_ROOT / "ABSA_Project" / "src"
DEFAULT_VNCORENLP_DIR = PROJECT_ROOT / "ABSA_Project" / "models" / "vncorenlp"

# Nạp file .env nếu có. Không bắt buộc cài python-dotenv.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except Exception:
    pass


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path
    model_code_dir: Path
    tokenizer_name: str
    max_len: int
    device: str
    segmenter_backend: str
    vncorenlp_dir: Path
    java_home: Path | None
    api_key: str | None
    request_timeout_seconds: float
    log_level: str
    local_files_only: bool
    model_version_override: str | None


def load_settings() -> Settings:
    device = os.getenv("DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE chỉ nhận: auto, cpu, cuda")

    segmenter_backend = os.getenv("SEGMENTER_BACKEND", "vncorenlp").strip().lower()
    if segmenter_backend not in {"vncorenlp", "raw"}:
        raise ValueError("SEGMENTER_BACKEND chỉ nhận: vncorenlp, raw")

    max_len = _env_int("MAX_LEN", 128)
    if not 1 <= max_len <= 256:
        raise ValueError("MAX_LEN phải nằm trong khoảng 1..256")

    timeout = _env_float("REQUEST_TIMEOUT_SECONDS", 30.0)
    if timeout <= 0:
        raise ValueError("REQUEST_TIMEOUT_SECONDS phải lớn hơn 0")

    java_home_raw = os.getenv("JAVA_HOME", "").strip()
    java_home = Path(java_home_raw).expanduser() if java_home_raw else None

    api_key = os.getenv("API_KEY", "").strip() or None

    model_version_override = os.getenv("MODEL_VERSION", "").strip() or None

    return Settings(
        checkpoint_path=_env_path("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT),
        model_code_dir=_env_path("MODEL_CODE_DIR", DEFAULT_MODEL_CODE_DIR),
        tokenizer_name=os.getenv("TOKENIZER_NAME", "vinai/phobert-base").strip(),
        max_len=max_len,
        device=device,
        segmenter_backend=segmenter_backend,
        vncorenlp_dir=_env_path("VNCORENLP_DIR", DEFAULT_VNCORENLP_DIR),
        java_home=java_home,
        api_key=api_key,
        request_timeout_seconds=timeout,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        local_files_only=_env_bool("HF_LOCAL_FILES_ONLY", True),
        model_version_override=model_version_override,
    )