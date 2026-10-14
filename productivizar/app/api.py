"""Endpoints HTTP. La lógica de inferencia vive en ``detector.py``; aquí solo validación y mapeo."""

import time
from typing import Annotated

import anyio
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings
from app.detector import DetectionResult, InvalidImageError, YoloOnnxDetector, draw_detections
from app.errors import APIError
from app.logging_config import request_id_ctx
from app.metrics import DETECTIONS, INFERENCE_LATENCY, INFERENCES_IN_FLIGHT
from app.schemas import (
    BoundingBox,
    DetectionOut,
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
    ImageSize,
    ModelInfo,
    Timings,
)

ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"})

_ERRORS = {
    status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse, "description": "Archivo demasiado grande"},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse, "description": "Tipo de archivo no soportado"},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse, "description": "Imagen o parámetros inválidos"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Modelo no disponible"},
}

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/v1", tags=["detection"])


# ----------------------------------------------------------------------------- dependencias


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_detector(request: Request) -> YoloOnnxDetector:
    detector: YoloOnnxDetector | None = getattr(request.app.state, "detector", None)
    if detector is None:
        raise APIError(status.HTTP_503_SERVICE_UNAVAILABLE, "model_unavailable", "El modelo no está cargado")
    return detector


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DetectorDep = Annotated[YoloOnnxDetector, Depends(get_detector)]
ImageFile = Annotated[UploadFile, File(description="Imagen JPEG/PNG/WebP/BMP/TIFF")]
ConfQuery = Annotated[float | None, Query(gt=0, lt=1, description="Umbral de confianza")]
IouQuery = Annotated[float | None, Query(gt=0, le=1, description="Umbral IoU para NMS")]
MaxDetQuery = Annotated[int | None, Query(ge=1, le=1000, description="Máximo de detecciones")]


async def _read_upload(request: Request, file: UploadFile, settings: Settings) -> bytes:
    limit = settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:  # margen para el envoltorio multipart
        raise APIError(status.HTTP_413_CONTENT_TOO_LARGE, "payload_too_large", f"Máximo {limit} bytes")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise APIError(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            f"Tipo '{file.content_type}' no soportado. Permitidos: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise APIError(status.HTTP_413_CONTENT_TOO_LARGE, "payload_too_large", f"Máximo {limit} bytes")
    if not data:
        raise APIError(status.HTTP_422_UNPROCESSABLE_CONTENT, "empty_file", "El archivo está vacío")
    return data


async def _run_detection(
    request: Request,
    detector: YoloOnnxDetector,
    settings: Settings,
    data: bytes,
    conf: float | None,
    iou: float | None,
    max_det: int | None,
) -> tuple[np.ndarray, DetectionResult]:
    """Decodifica + infiere en un hilo de trabajo (no bloquea el event loop) con concurrencia acotada."""

    def _work() -> tuple[np.ndarray, DetectionResult]:
        image = detector.decode_image(data)
        result = detector.predict(
            image,
            conf_threshold=conf if conf is not None else settings.conf_threshold,
            iou_threshold=iou if iou is not None else settings.iou_threshold,
            max_detections=max_det if max_det is not None else settings.max_detections,
        )
        return image, result

    INFERENCES_IN_FLIGHT.inc()
    try:
        image, result = await anyio.to_thread.run_sync(_work, limiter=request.app.state.inference_limiter)
    except InvalidImageError as exc:
        raise APIError(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_image", str(exc)) from exc
    finally:
        INFERENCES_IN_FLIGHT.dec()

    INFERENCE_LATENCY.observe(result.inference_ms / 1e3)
    for det in result.detections:
        DETECTIONS.labels(det.class_name).inc()
    return image, result


# ----------------------------------------------------------------------------- health / ops


@health_router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)


@health_router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe (modelo cargado)",
    responses={503: {"model": HealthResponse}},
)
async def readiness(request: Request, response: Response, settings: SettingsDep) -> HealthResponse:
    if getattr(request.app.state, "detector", None) is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", version=settings.app_version)
    return HealthResponse(status="ok", version=settings.app_version)


@health_router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------------------- v1


@v1_router.get("/model", response_model=ModelInfo, summary="Metadatos del modelo desplegado")
async def model_info(detector: DetectorDep) -> ModelInfo:
    m = detector.metadata
    return ModelInfo(
        task=m.task,
        description=m.description,
        sha256=m.sha256,
        ultralytics_version=m.ultralytics_version,
        exported_at=m.exported_at,
        input_size=ImageSize(width=m.input_width, height=m.input_height),
        end2end=m.end2end,
        providers=m.providers,
        classes=m.class_names,
    )


@v1_router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detecta objetos en una imagen",
    responses=_ERRORS,
)
async def detect(
    request: Request,
    file: ImageFile,
    detector: DetectorDep,
    settings: SettingsDep,
    conf: ConfQuery = None,
    iou: IouQuery = None,
    max_det: MaxDetQuery = None,
) -> DetectionResponse:
    t0 = time.perf_counter()
    data = await _read_upload(request, file, settings)
    _, result = await _run_detection(request, detector, settings, data, conf, iou, max_det)
    return DetectionResponse(
        request_id=request_id_ctx.get() or "",
        model_sha256=detector.metadata.sha256,
        image=ImageSize(width=result.image_width, height=result.image_height),
        count=len(result.detections),
        detections=[
            DetectionOut(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
            )
            for d in result.detections
        ],
        timings=Timings(
            preprocess_ms=round(result.preprocess_ms, 2),
            inference_ms=round(result.inference_ms, 2),
            postprocess_ms=round(result.postprocess_ms, 2),
            total_ms=round((time.perf_counter() - t0) * 1e3, 2),
        ),
    )


@v1_router.post(
    "/detect/annotated",
    summary="Devuelve la imagen con las detecciones dibujadas (JPEG)",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, **_ERRORS},
)
async def detect_annotated(
    request: Request,
    file: ImageFile,
    detector: DetectorDep,
    settings: SettingsDep,
    conf: ConfQuery = None,
    iou: IouQuery = None,
    max_det: MaxDetQuery = None,
) -> Response:
    data = await _read_upload(request, file, settings)
    image, result = await _run_detection(request, detector, settings, data, conf, iou, max_det)

    def _render() -> bytes:
        ok, buf = cv2.imencode(".jpg", draw_detections(image, result.detections), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise APIError(status.HTTP_500_INTERNAL_SERVER_ERROR, "encode_error", "No se pudo codificar la imagen")
        return buf.tobytes()

    jpeg = await anyio.to_thread.run_sync(_render)
    return Response(content=jpeg, media_type="image/jpeg", headers={"X-Detections-Count": str(len(result.detections))})
