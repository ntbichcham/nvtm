"""Nạp checkpoint PhoBERT thật và thực hiện inference.

File này KHÔNG tự tạo mô hình mới và KHÔNG sinh kết quả giả.
Kiến trúc được import trực tiếp từ ABSA_Project/src/model_phobert.py.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .config import Settings

logger = logging.getLogger(__name__)

# Nhãn hiển thị tiếng Việt. Khóa phải trùng ASPECT_LIST trong dataset_loader.py.
ASPECT_NAME_VI = {
    "BATTERY": "Pin",
    "CAMERA": "Camera",
    "DESIGN": "Thiết kế",
    "FEATURES": "Tính năng",
    "GENERAL": "Tổng quát",
    "PERFORMANCE": "Hiệu năng",
    "PRICE": "Giá",
    "SCREEN": "Màn hình",
    "SER&ACC": "Dịch vụ & Phụ kiện",
    "STORAGE": "Lưu trữ",
}

SENTIMENT_NAME_VI = {
    "None": "Không đề cập",
    "Negative": "Tiêu cực",
    "Neutral": "Trung tính",
    "Positive": "Tích cực",
}


class PredictorError(RuntimeError):
    """Lỗi chung của predictor."""


class PredictorNotReady(PredictorError):
    """Model hoặc segmenter chưa sẵn sàng."""


class ModelLoadError(PredictorError):
    """Không nạp được checkpoint/tokenizer/kiến trúc."""


class SegmenterUnavailable(PredictorError):
    """Không nạp được VnCoreNLP."""


class InferenceError(PredictorError):
    """Lỗi trong quá trình dự đoán."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    best_epoch: int | None
    max_len: int
    device: str
    segmentation: str


class PhoBERTABSAPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.tokenizer: Any = None
        self.segmenter: Any = None
        self.ready = False
        self.segmenter_ready = False
        self.load_error: str | None = None
        self.model_info: ModelInfo | None = None

        self._model_class: Any = None
        self._aspect_list: list[str] = []
        self._idx2polarity: dict[int, str] = {}
        self._num_classes = 0
        self._device: torch.device | None = None
        self._segmentation_name = settings.segmenter_backend
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Nạp mô hình
    # ------------------------------------------------------------------
    def load(self) -> None:
        started = time.perf_counter()
        try:
            self._configure_environment()
            self._load_model_code()
            self._load_tokenizer()
            self._load_checkpoint()
            self._load_segmenter()
            self._warmup()
            self.ready = True
            self.load_error = None
            logger.info(
                "PhoBERT ABSA ready in %.1fs | device=%s | segmentation=%s | max_len=%d",
                time.perf_counter() - started,
                self._device,
                self._segmentation_name,
                self.settings.max_len,
            )
        except Exception as exc:
            self.ready = False
            self.load_error = str(exc)
            logger.exception("Không nạp được PhoBERT ABSA: %s", exc)
            raise

    def _configure_environment(self) -> None:
        # Bắt buộc để transformers dùng cache local nếu HF_LOCAL_FILES_ONLY=true.
        if self.settings.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if self.settings.java_home:
            java_home = str(self.settings.java_home)
            os.environ["JAVA_HOME"] = java_home
            java_bin = str(self.settings.java_home / "bin")
            current_path = os.environ.get("PATH", "")
            if java_bin not in current_path.split(os.pathsep):
                os.environ["PATH"] = java_bin + os.pathsep + current_path

    def _load_model_code(self) -> None:
        code_dir = self.settings.model_code_dir
        if not code_dir.is_dir():
            raise ModelLoadError(
                f"Không tìm thấy thư mục mã nguồn mô hình: {code_dir}"
            )

        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)

        try:
            from dataset_loader import ASPECT_LIST, IDX2POLARITY, NUM_CLASSES
            from model_phobert import MultiHeadPhoBERTForABSA
        except Exception as exc:
            raise ModelLoadError(
                "Không import được model_phobert.py/dataset_loader.py. "
                "Kiểm tra MODEL_CODE_DIR."
            ) from exc

        self._aspect_list = list(ASPECT_LIST)
        self._idx2polarity = dict(IDX2POLARITY)
        self._num_classes = int(NUM_CLASSES)
        self._model_class = MultiHeadPhoBERTForABSA

        if len(self._aspect_list) != 10:
            raise ModelLoadError(
                f"ASPECT_LIST phải có 10 khía cạnh, hiện có {len(self._aspect_list)}"
            )
        if sorted(self._idx2polarity) != [0, 1, 2, 3]:
            raise ModelLoadError("IDX2POLARITY phải có đúng 4 nhãn: 0..3")
        if self._num_classes != 4:
            raise ModelLoadError(f"NUM_CLASSES phải bằng 4, hiện bằng {self._num_classes}")
        if set(ASPECT_NAME_VI) != set(self._aspect_list):
            raise ModelLoadError("ASPECT_NAME_VI không khớp ASPECT_LIST")

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.tokenizer_name,
                use_fast=False,
                local_files_only=self.settings.local_files_only,
            )
        except Exception as exc:
            raise ModelLoadError(
                "Không nạp được tokenizer PhoBERT. "
                "Kiểm tra TOKENIZER_NAME và cache Hugging Face."
            ) from exc

    def _load_checkpoint(self) -> None:
        path = self.settings.checkpoint_path
        if not path.is_file():
            raise ModelLoadError(f"Không tìm thấy checkpoint: {path}")

        logger.info("Đang nạp checkpoint: %s", path)
        try:
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,  # checkpoint có dict metadata, không chỉ state_dict
            )
        except Exception as exc:
            raise ModelLoadError("Không đọc được checkpoint .pt") from exc

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint
        else:
            state_dict = checkpoint
            metadata = {}

        args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
        if not isinstance(args, dict):
            args = {}

        dense_head = bool(args.get("dense_head", self._infer_dense_head(state_dict)))
        dropout = float(args.get("dropout", 0.1))
        pretrained_model = str(args.get("pretrained_model", self.settings.tokenizer_name))

        try:
            model = self._model_class(
                pretrained_model_name=pretrained_model,
                num_aspects=len(self._aspect_list),
                num_classes=self._num_classes,
                dropout=dropout,
                dense_head=dense_head,
            )
            load_result = model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ModelLoadError(
                "Checkpoint không khớp kiến trúc MultiHeadPhoBERTForABSA."
            ) from exc

        if load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "Checkpoint load warnings | missing=%s | unexpected=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )

        device = self._resolve_device()
        model.to(device)
        model.eval()

        best_epoch = metadata.get("epoch") if isinstance(metadata, dict) else None
        experiment_name = str(args.get("experiment_name", "phobert_absa"))
        version = self.settings.model_version_override or (
            f"{experiment_name}_ep{best_epoch if best_epoch is not None else 'unknown'}"
        )

        self.model = model
        self._device = device
        self.model_info = ModelInfo(
            name="PhoBERT-ABSA",
            version=version,
            best_epoch=int(best_epoch) if best_epoch is not None else None,
            max_len=self.settings.max_len,
            device=str(device),
            segmentation=self._segmentation_name,
        )

        # Giải phóng optimizer_state_dict và bản checkpoint gốc.
        del checkpoint
        del state_dict
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "Checkpoint loaded | version=%s | epoch=%s | parameters=%s",
            version,
            best_epoch,
            f"{sum(p.numel() for p in model.parameters()):,}",
        )

    @staticmethod
    def _infer_dense_head(state_dict: dict[str, Any]) -> bool:
        return any(key.startswith("heads.0.0.") for key in state_dict)

    def _resolve_device(self) -> torch.device:
        requested = self.settings.device
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("DEVICE=cuda nhưng torch không thấy CUDA")
        return torch.device(requested)

    def _load_segmenter(self) -> None:
        backend = self.settings.segmenter_backend

        if backend == "raw":
            self.segmenter = None
            self.segmenter_ready = True
            self._segmentation_name = "raw"
            logger.warning(
                "SEGMENTER_BACKEND=raw: văn bản KHÔNG được tách từ. "
                "Chỉ dùng để smoke test, không dùng cho kết quả chính thức."
            )
            return

        if shutil.which("java") is None:
            raise SegmenterUnavailable(
                "Không tìm thấy Java. Cài JDK 17 hoặc đặt JAVA_HOME đúng."
            )

        try:
            import py_vncorenlp

            self.settings.vncorenlp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Đang nạp VnCoreNLP: %s", self.settings.vncorenlp_dir)
            self.segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=str(self.settings.vncorenlp_dir),
            )
        except Exception as exc:
            raise SegmenterUnavailable(
                "Không nạp được VnCoreNLP. Kiểm tra Java, py_vncorenlp và VNCORENLP_DIR."
            ) from exc

        self.segmenter_ready = True
        self._segmentation_name = "vncorenlp"
        logger.info("VnCoreNLP ready")

    def _warmup(self) -> None:
        # Chạy một câu ngắn để phát hiện lỗi tokenizer/model/segmenter ngay khi khởi động.
        try:
            self._run_prediction("Pin dùng rất lâu.", include_all=False)
        except Exception as exc:
            raise InferenceError("Warm-up inference thất bại") from exc
        logger.info("Warm-up inference OK")

    # ------------------------------------------------------------------
    # Dự đoán
    # ------------------------------------------------------------------
    def segment(self, text: str) -> str:
        if self.settings.segmenter_backend == "raw":
            return text

        if self.segmenter is None:
            raise SegmenterUnavailable("VnCoreNLP chưa sẵn sàng")

        try:
            sentences = self.segmenter.word_segment(text)
        except Exception as exc:
            raise InferenceError("Lỗi tách từ VnCoreNLP") from exc

        segmented = " ".join(sentences).strip() if sentences else ""
        if not segmented:
            raise InferenceError("VnCoreNLP trả về văn bản rỗng")
        return segmented

    def predict(self, text: str, include_all: bool = False) -> dict[str, Any]:
        if not self.ready:
            raise PredictorNotReady("Mô hình chưa sẵn sàng")
        text = text.strip()
        if not text:
            raise ValueError("Nội dung đánh giá không được để trống")
        return self._run_prediction(text, include_all=include_all)

    def _run_prediction(self, text: str, include_all: bool) -> dict[str, Any]:
        total_start = time.perf_counter()

        seg_start = time.perf_counter()
        segmented_text = self.segment(text)
        segmentation_ms = (time.perf_counter() - seg_start) * 1000.0

        try:
            untruncated = self.tokenizer(
                segmented_text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_token_type_ids=False,
            )
            was_truncated = len(untruncated["input_ids"]) > self.settings.max_len

            encoding = self.tokenizer(
                segmented_text,
                truncation=True,
                max_length=self.settings.max_len,
                padding="max_length",
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise InferenceError("Lỗi tokenize văn bản") from exc

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        inference_start = time.perf_counter()
        try:
            with self._lock, torch.inference_mode():
                logits = self.model(input_ids, attention_mask)["logits"]
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        except Exception as exc:
            raise InferenceError("Lỗi forward pass của mô hình") from exc
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        all_aspects: list[dict[str, Any]] = []
        mentioned_aspects: list[dict[str, Any]] = []

        for aspect_index, aspect_code in enumerate(self._aspect_list):
            probs = probabilities[aspect_index]
            sentiment_index = int(np.argmax(probs))
            sentiment = self._idx2polarity[sentiment_index]
            confidence = float(probs[sentiment_index])

            item = {
                "code": aspect_code,
                "name_vi": ASPECT_NAME_VI[aspect_code],
                "model_index": aspect_index,
                "sentiment": sentiment,
                "sentiment_vi": SENTIMENT_NAME_VI[sentiment],
                "sentiment_index": sentiment_index,
                "confidence": round(confidence, 6),
                "confidence_percent": round(confidence * 100.0, 2),
                "probabilities": {
                    self._idx2polarity[i]: round(float(probs[i]), 6)
                    for i in range(self._num_classes)
                },
            }
            all_aspects.append(item)
            if sentiment != "None":
                mentioned_aspects.append(item)

        warnings: list[str] = []
        if self.settings.segmenter_backend == "raw":
            warnings.append(
                "SEGMENTER_BACKEND=raw: văn bản chưa được tách từ; "
                "kết quả chỉ dùng để smoke test."
            )
        if was_truncated:
            warnings.append(
                f"Văn bản đã bị cắt tại max_len={self.settings.max_len} token."
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        model_info = asdict(self.model_info) if self.model_info else {
            "name": "PhoBERT-ABSA",
            "version": "unknown",
            "best_epoch": None,
            "max_len": self.settings.max_len,
            "device": str(self._device) if self._device else self.settings.device,
            "segmentation": self._segmentation_name,
        }

        return {
            "model": model_info,
            "text": text,
            "segmented_text": segmented_text,
            "aspects": mentioned_aspects,
            "all_aspects": all_aspects if include_all else None,
            "warnings": warnings,
            "segmentation_ms": round(segmentation_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_ms": round(total_ms, 2),
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        model_loaded = self.model is not None
        ready = self.ready and model_loaded and self.segmenter_ready
        return {
            "status": "ok" if ready else "degraded",
            "model_loaded": model_loaded,
            "segmenter_ready": self.segmenter_ready,
            "checkpoint_found": self.settings.checkpoint_path.is_file(),
            "device": str(self._device) if self._device else self.settings.device,
            "max_len": self.settings.max_len,
            "segmenter_backend": self.settings.segmenter_backend,
            "model_name": self.model_info.name if self.model_info else "PhoBERT-ABSA",
            "model_version": self.model_info.version if self.model_info else None,
            "message": (
                "API sẵn sàng."
                if ready
                else "API chưa sẵn sàng. Kiểm tra log của Python API."
            ),
        }