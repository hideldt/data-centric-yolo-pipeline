# 🚀 Productivización — YOLO26 ONNX Detection API

API REST de detección de objetos que sirve el modelo `best.onnx` (YOLO26s entrenado en
`GData/YOLO_Dataset`, 9 clases) con **FastAPI + Uvicorn + ONNX Runtime**. Sin PyTorch ni
Ultralytics en runtime → imagen ligera y arranque rápido.

## Estructura

```
productivizar/
├── app/
│   ├── main.py            # App factory + lifespan (carga y warm-up del modelo)
│   ├── api.py             # Endpoints y validación de entrada
│   ├── detector.py        # Inferencia ONNX: letterbox, NMS por clase, reescalado
│   ├── schemas.py         # Contratos Pydantic (OpenAPI)
│   ├── config.py          # Configuración por variables de entorno (API_*)
│   ├── middleware.py      # Request-ID, access log, métricas, errores no controlados
│   ├── errors.py          # Formato de error único
│   ├── metrics.py         # Métricas Prometheus
│   └── logging_config.py  # Logs JSON estructurados
├── models/best.onnx       # Modelo (ignorado por git: *.onnx)
├── tests/                 # Tests de contrato con el modelo real
├── Dockerfile             # Multi-stage: builder → test → runtime
└── requirements*.txt      # Dependencias fijadas
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/v1/detect` | Imagen (multipart `file`) → detecciones JSON |
| `POST` | `/v1/detect/annotated` | Imagen → JPEG con las cajas dibujadas |
| `GET` | `/v1/model` | Metadatos del modelo (clases, sha256, input, providers) |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (503 hasta que el modelo cargue) |
| `GET` | `/metrics` | Métricas Prometheus |
| `GET` | `/docs` | Swagger UI (desactivable con `API_DOCS_ENABLED=false`) |

Parámetros opcionales por query en `/v1/detect*`: `conf` (0–1), `iou` (0–1), `max_det` (1–1000).

## Uso con Docker

```bash
cd productivizar

# Tests dentro de la imagen (CI)
docker build --target test -t yolo26-api:test .

# Imagen de producción
docker build -t yolo26-api:1.0.0 .

# Despliegue SOLO por Tailscale (no expuesto a la LAN ni a localhost)
TSIP=$(tailscale ip -4 | head -1)
docker run -d --name yolo26-api --restart unless-stopped \
  -p ${TSIP}:8000:8000 --read-only --tmpfs /tmp yolo26-api:1.0.0

# Probar desde cualquier equipo del tailnet
curl -s http://<tu-nodo>.ts.net:8000/health/ready
curl -s -F "file=@foto.jpg" "http://<tu-nodo>.ts.net:8000/v1/detect?conf=0.3" | jq
curl -s -F "file=@foto.jpg" http://<tu-nodo>.ts.net:8000/v1/detect/annotated -o resultado.jpg
```

Swagger UI: `http://<tu-nodo>.ts.net:8000/docs` (o `http://<tu-ip-tailscale>:8000/docs`)

> [!IMPORTANT]
> Al publicar el puerto en la IP de Tailscale, tras un reinicio Docker puede arrancar antes de que
> `tailscaled` asigne la IP y el contenedor fallará al iniciar. Para evitarlo (una sola vez, requiere sudo):
>
> ```bash
> echo 'net.ipv4.ip_nonlocal_bind=1' | sudo tee /etc/sysctl.d/99-tailscale-bind.conf
> sudo sysctl --system
> ```

Para cambiar de modelo sin reconstruir la imagen:

```bash
docker run -p 8000:8000 -v $(pwd)/models/otro.onnx:/app/models/best.onnx:ro yolo26-api:1.0.0
```

## Ejecución local (sin Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
API_LOG_FORMAT=text uvicorn app.main:app --reload
```

## Configuración (variables de entorno)

| Variable | Default | Descripción |
|---|---|---|
| `API_MODEL_PATH` | `models/best.onnx` | Ruta del modelo |
| `API_PROVIDERS` | `["CPUExecutionProvider"]` | Execution providers de ONNX Runtime (JSON) |
| `API_CONF_THRESHOLD` | `0.25` | Confianza mínima por defecto |
| `API_IOU_THRESHOLD` | `0.7` | IoU del NMS por defecto |
| `API_MAX_DETECTIONS` | `300` | Máximo de detecciones |
| `API_MAX_UPLOAD_BYTES` | `10485760` | Tamaño máximo del archivo (10 MB) |
| `API_MAX_CONCURRENT_INFERENCES` | `2` | Inferencias simultáneas (el resto espera en cola) |
| `API_INTRA_OP_THREADS` | `0` | Hilos de ONNX Runtime por inferencia (0 = auto) |
| `API_LOG_LEVEL` / `API_LOG_FORMAT` | `INFO` / `json` | Logging |
| `API_DOCS_ENABLED` | `true` | Exponer `/docs` y `/openapi.json` |
| `API_CORS_ORIGINS` | `[]` | Orígenes CORS permitidos (JSON) |

## Decisiones de diseño

- **Paridad con Ultralytics**: el letterbox (pad 114, redondeo idéntico) y el NMS por clase reproducen
  el pipeline de `model.predict`, así que la API da las mismas cajas que el notebook. También soporta
  exportaciones `end2end` (NMS-free) leyendo los metadatos del ONNX.
- **Event loop libre**: la decodificación y la inferencia corren en un thread pool con un
  `CapacityLimiter`, para no saturar la CPU con hilos de ONNX Runtime compitiendo entre sí.
- **Fail-fast**: si el modelo no carga, el contenedor no arranca. `/health/ready` sólo devuelve 200
  cuando el modelo está cargado y precalentado.
- **1 worker por contenedor** (`WEB_CONCURRENCY=1`): las métricas Prometheus son coherentes y se escala
  con réplicas (Kubernetes/Compose).
- **Seguridad**: usuario no-root (UID 10001), código de sólo lectura, límite de tamaño de archivo,
  validación de tipo MIME, límite de píxeles decodificados (`OPENCV_IO_MAX_IMAGE_PIXELS`) y
  `X-Request-ID` saneado.

> [!NOTE]
> GPU: reemplaza `onnxruntime` por `onnxruntime-gpu` en `requirements.txt`, usa una imagen base con CUDA
> y define `API_PROVIDERS='["CUDAExecutionProvider","CPUExecutionProvider"]'`.

> [!WARNING]
> Los modelos YOLO de Ultralytics se distribuyen bajo licencia **AGPL-3.0**. Para un servicio comercial
> cerrado se requiere una licencia Enterprise de Ultralytics.
