# -----------------------------------------------------------------------------
#  YOLO26 ONNX Detection API
#    docker build -t yolo26-api .                 -> imagen de producción
#    docker build --target test -t yolo26-api:test .  -> corre la suite de tests
# -----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12

# ---------- 1) builder: resuelve dependencias en un venv aislado ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# ---------- 2) test: valida la imagen antes de publicarla (CI) ----------
FROM builder AS test
WORKDIR /app
COPY requirements-dev.txt requirements.txt ./
RUN pip install -r requirements-dev.txt
COPY app ./app
COPY models ./models
COPY tests ./tests
RUN python -m pytest -q

# ---------- 3) runtime: imagen mínima, sin herramientas de build, usuario no-root ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="yolo26-detection-api" \
      org.opencontainers.image.description="YOLO26 (ONNX Runtime) servido con FastAPI + Uvicorn" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="AGPL-3.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    API_MODEL_PATH=/app/models/best.onnx \
    API_ENVIRONMENT=production \
    # Protección contra "decompression bombs" (máx. ~50 MP por imagen)
    OPENCV_IO_MAX_IMAGE_PIXELS=50000000 \
    # 1 worker por contenedor; escalar horizontalmente con réplicas
    WEB_CONCURRENCY=1

RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Código y modelo propiedad de root (solo lectura para el usuario de la app)
COPY app ./app
COPY models ./models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", "--no-server-header", \
     "--timeout-graceful-shutdown", "30"]
