r"""
backend.main
============

FastAPI application factory and ASGI entrypoint.

Run with::

    uvicorn backend.main:app --reload --port 8000

Design notes
------------
* **Lifespan-managed resources.**  The telemetry hub's asyncio task is started in
  the ``lifespan`` context manager and cancelled on shutdown, so a ``--reload``
  cycle does not leak tasks.
* **Global exception handlers.**  Three handlers cover the failure modes that
  matter here: ``KeyError`` (unknown chemistry -> 422), ``RuntimeError`` (model
  / dataset failure -> 503 with the repr, which the auto-heal monitor parses),
  and the catch-all (500).  Every error body includes a ``traceback_id`` so a
  user-reported failure maps to one line in the log.
* **The inference engine is NOT started at import time.**  Loading torch and
  constructing the network costs ~1-2 s; keeping it lazy means ``/api/health``
  answers within milliseconds even while the engine is still being built, which
  is what lets the auto-heal supervisor distinguish "slow start" from "dead".
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure the repository root is importable when launched as ``python backend/main.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import (  # noqa: E402
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    LOG_DIR,
)
from backend.routers import (  # noqa: E402
    control,
    health,
    heatwave,
    misc,
    nasa,
    predict,
    simulate,
)
from backend.services.telemetry import HUB  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(LOG_DIR) / "backend.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backend.main")

DESCRIPTION = """
**Heatwave-Aware AI-Assisted Microchannel Battery Thermal Management Dashboard for EVs**

A dual-chemistry (NMC / LFP) physics-informed neural network predicts the 3-D
temperature field of a liquid-cooled microchannel battery pack; an anticipatory
controller ramps the coolant mass flow *before* the vehicle enters a
high-temperature zone.

### Surfaces

| Prefix | Purpose |
|---|---|
| `/api/predict` | PINN field + point evaluation, with physics-residual diagnostics |
| `/api/heatwave` | Anticipatory route planning (advisory) |
| `/api/simulate` | Full-route rollouts and the live WebSocket telemetry stream |
| `/api/control` | Coolant set-point commands and the fail-safe supervisor |
| `/api/nasa` | NASA run-to-failure calibration provenance |
| `/api/catalog` | Chemistry catalogue (single source of truth for the UI) |
| `/api/demo` | One-call bootstrap and the comparative-stability study |
"""

TAGS_METADATA = [
    {"name": "prediction", "description": "Dual-chemistry PINN temperature field."},
    {"name": "heatwave", "description": "Anticipatory coolant planning."},
    {"name": "simulation", "description": "Rollouts and live telemetry."},
    {"name": "control", "description": "Coolant actuator + fail-safe."},
    {"name": "nasa-prognostics", "description": "NASA battery dataset calibration."},
    {"name": "meta", "description": "Health, catalogue, model info."},
    {"name": "demo", "description": "Demo fixtures and studies."},
]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the telemetry hub around the app's lifetime."""
    logger.info("starting %s v%s", API_TITLE, API_VERSION)
    await HUB.start()
    logger.info("telemetry hub started (tick=%.2fs)", HUB.tick_s)
    try:
        yield
    finally:
        logger.info("shutting down telemetry hub")
        await HUB.stop()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the FastAPI application with routers, middleware and handlers."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "GenAI Lab -- Semester 7 Mini Project"},
        license_info={"name": "Research/Educational Use Only"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Traceback-Id"],
    )

    # -- request timing header ---------------------------------------------
    @app.middleware("http")
    async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Stamp every response with its server-side duration.

        The frontend uses this for its own end-to-end latency readout, so the
        800 ms budget in the model card is measured, not asserted.
        """
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        return response

    # -- routers ------------------------------------------------------------
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(heatwave.router)
    app.include_router(simulate.router)
    app.include_router(control.router)
    app.include_router(nasa.router)
    app.include_router(misc.router)

    # -- exception handlers -------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 with a machine-readable error list."""
        tid = uuid.uuid4().hex[:12]
        logger.warning("[%s] validation error on %s: %s", tid, request.url.path,
                       exc.errors()[:3])
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "traceback_id": tid,
                "path": request.url.path,
                "detail": _jsonable(exc.errors()),
            },
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        """Unknown chemistry -> 422 (a client error, not a server fault)."""
        tid = uuid.uuid4().hex[:12]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "unknown_chemistry",
                "traceback_id": tid,
                "detail": str(exc).strip("'"),
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        """Model/dataset runtime failure -> 503 with the repr.

        The auto-heal monitor keys off the ``error`` field being
        ``model_unavailable`` and reads ``detail`` for the traceback signature.
        """
        tid = uuid.uuid4().hex[:12]
        logger.error("[%s] RuntimeError on %s: %r", tid, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "model_unavailable",
                "traceback_id": tid,
                "path": request.url.path,
                "detail": repr(exc),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all 500 that always emits a ``traceback_id``."""
        tid = uuid.uuid4().hex[:12]
        logger.exception("[%s] unhandled %s on %s", tid, type(exc).__name__,
                         request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "traceback_id": tid,
                "path": request.url.path,
                "exception_type": type(exc).__name__,
                "detail": repr(exc),
            },
        )

    # -- root ---------------------------------------------------------------
    @app.get("/", tags=["meta"], summary="Service banner")
    async def root() -> Dict[str, Any]:
        """Human-readable banner with links to the docs and the key endpoints."""
        return {
            "title": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "frontend": "http://localhost:3000",
            "endpoints": {
                "health": "/api/health",
                "chemistries": "/api/catalog/chemistries",
                "field": "/api/predict/field",
                "residuals": "/api/predict/residuals",
                "forecast": "/api/heatwave/forecast",
                "scenario": "/api/simulate/scenario",
                "rollout": "/api/simulate/trajectory",
                "telemetry_ws": "/api/simulate/ws/telemetry?chemistry=nmc",
                "control": "/api/control/flow",
                "nasa_calibration": "/api/nasa/calibration",
                "comparison": "/api/demo/comparison",
                "bootstrap": "/api/demo/bootstrap",
            },
        }

    return app


#: Module-level ASGI application (uvicorn target).
app = create_app()


# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of pydantic error payloads to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    import uvicorn

    from backend.config import API_HOST, API_PORT

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=os.getenv("RELOAD", "0") == "1",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )