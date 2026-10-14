"""Contratos públicos de la API (request/response). Son la fuente de verdad del esquema OpenAPI."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())


class BoundingBox(_Schema):
    """Caja en píxeles de la imagen original, formato ``xyxy``."""

    x1: float = Field(ge=0, examples=[34.5])
    y1: float = Field(ge=0, examples=[120.0])
    x2: float = Field(ge=0, examples=[310.2])
    y2: float = Field(ge=0, examples=[480.7])


class DetectionOut(_Schema):
    class_id: int = Field(ge=0, examples=[2])
    class_name: str = Field(examples=["Car"])
    confidence: float = Field(ge=0, le=1, examples=[0.91])
    bbox: BoundingBox


class ImageSize(_Schema):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Timings(_Schema):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class DetectionResponse(_Schema):
    request_id: str
    model_sha256: str
    image: ImageSize
    count: int = Field(ge=0)
    detections: list[DetectionOut]
    timings: Timings


class ModelInfo(_Schema):
    task: str
    description: str
    sha256: str
    ultralytics_version: str
    exported_at: str
    input_size: ImageSize
    end2end: bool
    providers: list[str]
    classes: dict[int, str]


class HealthResponse(_Schema):
    status: Literal["ok", "unavailable"]
    version: str


class ErrorDetail(_Schema):
    code: str = Field(examples=["unsupported_media_type"])
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


class ErrorResponse(_Schema):
    error: ErrorDetail
