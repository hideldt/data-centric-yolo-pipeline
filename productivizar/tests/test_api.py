import numpy as np
from fastapi.testclient import TestClient

from app.detector import nms


def _upload(data: bytes, content_type: str = "image/jpeg") -> dict:
    return {"file": ("img.jpg", data, content_type)}


# ----------------------------------------------------------------------------- health / ops


def test_liveness(client: TestClient) -> None:
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["task"] == "detect"
    assert body["input_size"] == {"width": 640, "height": 640}
    assert len(body["classes"]) == 9
    assert body["classes"]["2"] == "Car"
    assert len(body["sha256"]) == 64


def test_metrics_exposed(client: TestClient) -> None:
    client.get("/health/live")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert 'http_requests_total{method="GET",route="/health/live",status="200"}' in r.text


# ----------------------------------------------------------------------------- detección


def test_detect_ok(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/v1/detect", files=_upload(jpeg_bytes), params={"conf": 0.01})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": 720, "height": 480}
    assert body["count"] == len(body["detections"])
    assert body["request_id"] == r.headers["X-Request-ID"]
    for det in body["detections"]:
        b = det["bbox"]
        assert 0 <= b["x1"] <= b["x2"] <= 720
        assert 0 <= b["y1"] <= b["y2"] <= 480
        assert 0.01 <= det["confidence"] <= 1


def test_detect_respects_max_det(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/v1/detect", files=_upload(jpeg_bytes), params={"conf": 0.001, "max_det": 3})
    assert r.status_code == 200
    assert r.json()["count"] <= 3


def test_request_id_is_propagated(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/v1/detect", files=_upload(jpeg_bytes), headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_detect_annotated_returns_jpeg(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/v1/detect/annotated", files=_upload(jpeg_bytes))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    assert "X-Detections-Count" in r.headers


# ----------------------------------------------------------------------------- errores


def test_unsupported_media_type(client: TestClient) -> None:
    r = client.post("/v1/detect", files=_upload(b"hola", "text/plain"))
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_corrupted_image(client: TestClient) -> None:
    r = client.post("/v1/detect", files=_upload(b"\xff\xd8 esto no es un jpeg"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_image"


def test_empty_file(client: TestClient) -> None:
    r = client.post("/v1/detect", files=_upload(b""))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "empty_file"


def test_payload_too_large(client: TestClient) -> None:
    r = client.post("/v1/detect", files=_upload(b"\x00" * (3 * 1024 * 1024)))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_invalid_query_params(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/v1/detect", files=_upload(jpeg_bytes), params={"conf": 1.5})
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    assert body["request_id"]


def test_missing_file(client: TestClient) -> None:
    r = client.post("/v1/detect")
    assert r.status_code == 422


def test_not_found_uses_error_envelope(client: TestClient) -> None:
    r = client.get("/no-existe")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


# ----------------------------------------------------------------------------- unidad


def test_nms_suppresses_overlaps() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5, max_keep=10)
    assert keep.tolist() == [0, 2]
