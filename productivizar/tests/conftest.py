from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.onnx"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(model_path=MODEL_PATH, log_format="text", max_upload_bytes=2 * 1024 * 1024)


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:  # el context manager ejecuta el lifespan
        yield c


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(480, 720, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (400, 350), (40, 40, 200), -1)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()
