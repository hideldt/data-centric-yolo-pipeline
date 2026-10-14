"""Middleware ASGI puro: request-id, access log, métricas y red de seguridad ante errores no controlados."""

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_response
from app.logging_config import request_id_ctx
from app.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = logging.getLogger("app.access")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_QUIET_ROUTES = frozenset({"/health/live", "/health/ready", "/metrics"})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1e3:.2f}"
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("Error no controlado")
            if response_started:
                raise
            await error_response(500, "internal_error", "Error interno del servidor")(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "__unmatched__")
            HTTP_REQUESTS.labels(scope["method"], route, str(status_code)).inc()
            HTTP_LATENCY.labels(scope["method"], route).observe(elapsed)
            if route not in _QUIET_ROUTES:
                client = scope.get("client")
                logger.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1e3, 2),
                        "client_ip": client[0] if client else None,
                    },
                )
            request_id_ctx.reset(token)
