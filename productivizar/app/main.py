"""Punto de entrada ASGI: ``uvicorn app.main:app``."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health_router, v1_router
from app.config import Settings, get_settings
from app.detector import YoloOnnxDetector
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.inference_limiter = anyio.CapacityLimiter(settings.max_concurrent_inferences)

    def _load() -> YoloOnnxDetector:
        detector = YoloOnnxDetector(
            model_path=settings.model_path,
            providers=settings.providers,
            intra_op_threads=settings.intra_op_threads,
            inter_op_threads=settings.inter_op_threads,
        )
        detector.warmup()
        return detector

    # Fail-fast: si el modelo no carga, el proceso no arranca y el orquestador lo reporta.
    app.state.detector = await anyio.to_thread.run_sync(_load)
    logger.info("Servicio listo", extra={"environment": settings.environment, "version": settings.app_version})
    try:
        yield
    finally:
        app.state.detector = None
        logger.info("Servicio detenido")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Detección de objetos con YOLO26 exportado a ONNX y servido con ONNX Runtime.",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.detector = None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Process-Time-Ms", "X-Detections-Count"],
        )
    app.add_middleware(RequestContextMiddleware)  # el último añadido es el más externo

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(v1_router)
    return app


app = create_app()
