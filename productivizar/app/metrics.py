"""Métricas Prometheus. Se recomienda 1 worker por contenedor y escalar con réplicas."""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "http_requests_total", "Total de requests HTTP", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latencia de requests HTTP",
    ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Latencia de la sesión ONNX (solo session.run)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
DETECTIONS = Counter("model_detections_total", "Detecciones emitidas por clase", ["class_name"])
INFERENCES_IN_FLIGHT = Gauge("model_inferences_in_flight", "Inferencias ejecutándose o en cola")
