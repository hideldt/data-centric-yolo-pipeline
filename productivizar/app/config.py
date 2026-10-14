"""Configuración centralizada (12-factor): todo se sobreescribe con variables de entorno ``API_*``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import __version__


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Servicio ---
    app_name: str = "YOLO26 Detection API"
    app_version: str = __version__
    environment: Literal["development", "staging", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    docs_enabled: bool = True
    cors_origins: list[str] = Field(default_factory=list)

    # --- Modelo / ONNX Runtime ---
    model_path: Path = Path("models/best.onnx")
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    intra_op_threads: int = Field(default=0, ge=0, description="0 = ONNX Runtime decide")
    inter_op_threads: int = Field(default=0, ge=0, description="0 = ONNX Runtime decide")
    max_concurrent_inferences: int = Field(default=2, ge=1)

    # --- Inferencia (valores por defecto, sobreescribibles por request) ---
    conf_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    iou_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1, le=1000)

    # --- Entrada ---
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
