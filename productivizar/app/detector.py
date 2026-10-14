"""Motor de inferencia YOLO sobre ONNX Runtime, sin dependencia de PyTorch ni Ultralytics.

El preprocesamiento (letterbox, pad=114, BGR->RGB, /255) y el postprocesamiento (NMS por clase,
reescalado de cajas) replican exactamente los de Ultralytics para obtener las mismas predicciones
que en el notebook.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_PAD_COLOR = (114, 114, 114)
_MAX_WH = 7680.0  # desplazamiento por clase para hacer NMS por clase en una sola pasada


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    path: str
    sha256: str
    task: str
    description: str
    ultralytics_version: str
    exported_at: str
    input_name: str
    input_height: int
    input_width: int
    end2end: bool
    class_names: dict[int, str]
    providers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Letterbox:
    gain: float
    pad_x: float
    pad_y: float


class InvalidImageError(ValueError):
    """La imagen no pudo decodificarse."""


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_keep: int) -> np.ndarray:
    """Non-Maximum Suppression vectorizado (cajas ``xyxy``). Devuelve índices ordenados por score."""
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < max_keep:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        inter = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None) * np.clip(
            np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None
        )
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.intp)


class YoloOnnxDetector:
    """Envoltura thread-safe de una ``InferenceSession`` de ONNX Runtime para detección YOLO."""

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads

        available = ort.get_available_providers()
        requested = providers or ["CPUExecutionProvider"]
        chosen = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        if chosen != requested:
            logger.warning("Providers no disponibles ignorados", extra={"requested": requested, "using": chosen})

        self._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=chosen)

        inp = self._session.get_inputs()[0]
        meta = self._session.get_modelmeta().custom_metadata_map
        if meta.get("task", "detect") != "detect":
            raise ValueError(f"El modelo es de tarea '{meta.get('task')}', se esperaba 'detect'")

        imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        height = inp.shape[2] if isinstance(inp.shape[2], int) else int(imgsz[0])
        width = inp.shape[3] if isinstance(inp.shape[3], int) else int(imgsz[1])
        names = {int(k): str(v) for k, v in ast.literal_eval(meta.get("names", "{}")).items()}

        self._input_name = inp.name
        self._input_hw = (height, width)
        self._class_names = names
        self._end2end = meta.get("end2end", "False") == "True"
        self.metadata = ModelMetadata(
            path=str(model_path),
            sha256=_sha256(model_path),
            task=meta.get("task", "detect"),
            description=meta.get("description", ""),
            ultralytics_version=meta.get("version", "unknown"),
            exported_at=meta.get("date", "unknown"),
            input_name=inp.name,
            input_height=height,
            input_width=width,
            end2end=self._end2end,
            class_names=names,
            providers=self._session.get_providers(),
        )
        logger.info(
            "Modelo ONNX cargado",
            extra={
                "model_path": str(model_path),
                "sha256": self.metadata.sha256,
                "input": [height, width],
                "classes": len(names),
                "end2end": self._end2end,
                "providers": self.metadata.providers,
            },
        )

    # ------------------------------------------------------------------ API pública

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise InvalidImageError("No se pudo decodificar la imagen")
        return image

    def warmup(self, runs: int = 2) -> None:
        dummy = np.full((*self._input_hw, 3), _PAD_COLOR[0], dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy)

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        blob, lb = self._preprocess(image_bgr)
        t1 = time.perf_counter()
        output = self._session.run(None, {self._input_name: blob})[0]
        t2 = time.perf_counter()
        h0, w0 = image_bgr.shape[:2]
        detections = self._postprocess(output, lb, (h0, w0), conf_threshold, iou_threshold, max_detections)
        t3 = time.perf_counter()
        return DetectionResult(
            detections=detections,
            image_width=w0,
            image_height=h0,
            preprocess_ms=(t1 - t0) * 1e3,
            inference_ms=(t2 - t1) * 1e3,
            postprocess_ms=(t3 - t2) * 1e3,
        )

    # ------------------------------------------------------------------ internos

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, _Letterbox]:
        h0, w0 = image.shape[:2]
        th, tw = self._input_hw
        gain = min(th / h0, tw / w0)
        new_w, new_h = round(w0 * gain), round(h0 * gain)
        if (new_w, new_h) != (w0, h0):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = (tw - new_w) / 2, (th - new_h) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_PAD_COLOR)

        blob = image[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32)[None] / np.float32(255.0)
        return blob, _Letterbox(gain=gain, pad_x=float(left), pad_y=float(top))

    def _postprocess(
        self,
        output: np.ndarray,
        lb: _Letterbox,
        orig_hw: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        if self._end2end:
            # Salida NMS-free: (1, N, 6) -> x1, y1, x2, y2, score, class
            pred = output[0]
            pred = pred[pred[:, 4] >= conf_threshold][:max_detections]
            boxes, scores, class_ids = pred[:, :4].copy(), pred[:, 4], pred[:, 5].astype(np.intp)
        else:
            # Salida clásica: (1, 4 + nc, N) -> cx, cy, w, h, score_c0..score_cN
            pred = output[0].T
            class_scores = pred[:, 4:]
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(len(pred)), class_ids]
            mask = scores >= conf_threshold
            if not mask.any():
                return []
            xywh, scores, class_ids = pred[mask, :4], scores[mask], class_ids[mask]
            boxes = np.empty_like(xywh)
            boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
            boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
            keep = nms(boxes + class_ids[:, None] * _MAX_WH, scores, iou_threshold, max_detections)
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        # Deshacer letterbox -> coordenadas de la imagen original
        h0, w0 = orig_hw
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - lb.pad_x) / lb.gain).clip(0, w0)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - lb.pad_y) / lb.gain).clip(0, h0)

        return [
            Detection(
                class_id=int(c),
                class_name=self._class_names.get(int(c), str(int(c))),
                confidence=round(float(s), 4),
                x1=round(float(b[0]), 2),
                y1=round(float(b[1]), 2),
                x2=round(float(b[2]), 2),
                y2=round(float(b[3]), 2),
            )
            for b, s, c in zip(boxes, scores, class_ids, strict=True)
        ]


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Dibuja las cajas sobre una copia de la imagen (para el endpoint de visualización)."""
    canvas = image_bgr.copy()
    thickness = max(round(sum(canvas.shape[:2]) / 2 * 0.003), 2)
    for det in detections:
        hue = (det.class_id * 37) % 180
        color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        label = f"{det.class_name} {det.confidence:.2f}"
        font_scale = thickness / 3
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(thickness - 1, 1))
        y_text = p1[1] - 4 if p1[1] - th - baseline - 4 >= 0 else p1[1] + th + 4
        cv2.rectangle(canvas, (p1[0], y_text - th - 4), (p1[0] + tw + 4, y_text + baseline), color, -1)
        cv2.putText(
            canvas, label, (p1[0] + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), max(thickness - 1, 1), cv2.LINE_AA,
        )
    return canvas
