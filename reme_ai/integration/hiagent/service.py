"""FastAPI service skeleton for the HiAgent/ReMe integration (phase A0)."""

import inspect
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from reme_ai.main import ReMeApp

from .schemas import (
    API_VERSION,
    SERVICE_NAME,
    ComponentHealth,
    ErrorDetail,
    ErrorResponse,
    FinishTrialRequest,
    HealthResponse,
    RetrieveRequest,
)


Probe = Callable[[Any], bool | ComponentHealth | Awaitable[bool | ComponentHealth]]


def _load_env_file() -> None:
    """Load the local env file before model names are needed by ReMeApp."""

    env_path = Path(os.getenv("REME_HIAGENT_ENV_FILE", ".env"))
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _create_reme_app_from_env() -> ReMeApp:
    """Create ReMeApp using optional model-name overrides from the environment."""

    _load_env_file()
    overrides = []
    llm_model = os.getenv("REME_HIAGENT_LLM_MODEL", "").strip()
    embedding_model = os.getenv("REME_HIAGENT_EMBEDDING_MODEL", "").strip()
    llm_backend = os.getenv("REME_HIAGENT_LLM_BACKEND", "").strip()
    embedding_backend = os.getenv("REME_HIAGENT_EMBEDDING_BACKEND", "").strip()
    if llm_model:
        overrides.append(f"llm.default.model_name={llm_model}")
    if embedding_model:
        overrides.append(f"embedding_model.default.model_name={embedding_model}")
    if llm_backend:
        overrides.append(f"llm.default.backend={llm_backend}")
    if embedding_backend:
        overrides.append(f"embedding_model.default.backend={embedding_backend}")
    return ReMeApp(*overrides)


class HiAgentReadinessChecker:
    """Conservative readiness checks with injectable probes for real backends."""

    _COMPONENT_NAMES = {
        "llm": ("default_llm", "llm", "llms", "llm_dict"),
        "embedding": (
            "default_embedding_model",
            "embedding_model",
            "embedding_models",
            "embedding_model_dict",
        ),
        "vector_store": (
            "default_vector_store",
            "vector_store",
            "vector_stores",
            "vector_store_dict",
        ),
    }

    def __init__(self, probes: Mapping[str, Probe] | None = None):
        self.probes = dict(probes or {})

    async def check(self, reme_app: Any) -> HealthResponse:
        results = {
            name: await self._check_component(name, reme_app)
            for name in ("llm", "embedding", "vector_store")
        }
        is_ready = all(result.ready for result in results.values())
        return HealthResponse(status="ok" if is_ready else "error", **results)

    async def _check_component(self, name: str, reme_app: Any) -> ComponentHealth:
        probe = self.probes.get(name)
        if probe is not None:
            try:
                result = probe(reme_app)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, ComponentHealth):
                    return result
                return ComponentHealth(ready=bool(result))
            except Exception as exc:  # readiness must report failures, not crash
                return ComponentHealth(ready=False, detail=f"{type(exc).__name__}: {exc}")

        component = self._find_component(name, reme_app)
        if component is None:
            return ComponentHealth(ready=False, detail=f"{name} component is not available")

        if name == "embedding":
            return await self._probe_embedding(component)
        if name == "vector_store":
            return await self._probe_vector_store(component)
        return ComponentHealth(ready=True, detail="configured")

    def _find_component(self, name: str, reme_app: Any) -> Any | None:
        component = self._find_component_via_flowllm(name)
        if component is not None:
            return component

        sources = [reme_app, getattr(reme_app, "context", None)]
        try:
            from flowllm.core.context import C

            sources.append(C)
        except ImportError:
            pass

        for source in sources:
            if source is None:
                continue
            for attr_name in self._COMPONENT_NAMES[name]:
                value = getattr(source, attr_name, None)
                if isinstance(value, Mapping):
                    value = value.get("default") or next(iter(value.values()), None)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _find_component_via_flowllm(name: str) -> Any | None:
        """Use FlowLLM's public registry API before compatibility fallbacks."""

        try:
            from flowllm.core.context import C

            if name == "llm":
                backend = os.getenv("REME_HIAGENT_LLM_BACKEND", "openai_compatible")
                return C.get_llm_class(backend)

            if name == "embedding":
                backend = os.getenv("REME_HIAGENT_EMBEDDING_BACKEND", "openai_compatible")
                C.get_embedding_model_class(backend)
                vector_store = C.get_vector_store("default")
                return getattr(vector_store, "embedding_model", None)

            if name == "vector_store":
                return C.get_vector_store("default")
        except (AssertionError, KeyError, RuntimeError):
            return None

        return None

    async def _probe_embedding(self, component: Any) -> ComponentHealth:
        try:
            method = getattr(component, "aget_embeddings", None) or getattr(component, "get_embeddings", None)
            if method is None:
                return ComponentHealth(ready=False, detail="embedding probe method is unavailable")
            result = method(["health check"])
            if inspect.isawaitable(result):
                result = await result
            ready = bool(result and result[0])
            return ComponentHealth(ready=ready, detail=None if ready else "embedding probe returned no vector")
        except Exception as exc:
            return ComponentHealth(ready=False, detail=f"{type(exc).__name__}: {exc}")

    async def _probe_vector_store(self, component: Any) -> ComponentHealth:
        try:
            method = getattr(component, "async_exist_workspace", None)
            if method is None:
                return ComponentHealth(ready=False, detail="vector-store probe method is unavailable")
            result = method("__hiagent_health_check__")
            if inspect.isawaitable(result):
                await result
            return ComponentHealth(ready=True)
        except Exception as exc:
            return ComponentHealth(ready=False, detail=f"{type(exc).__name__}: {exc}")


def _error_response(status_code: int, code: str, message: str, *, details: dict[str, Any] | None = None):
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}),
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


def _serializable_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Remove exception objects from validation errors across FastAPI versions."""

    return [{key: value for key, value in error.items() if key != "ctx"} for error in exc.errors()]


def create_hiagent_api(
    *,
    reme_app_factory: Callable[[], Any] | None = None,
    readiness_checker: HiAgentReadinessChecker | None = None,
) -> FastAPI:
    """Create the phase-A0 API with one ReMeApp instance per service lifespan."""

    factory = reme_app_factory or _create_reme_app_from_env
    checker = readiness_checker or HiAgentReadinessChecker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.reme_app = None
        app.state.reme_app_started = False
        app.state.startup_error = None
        try:
            reme_app = factory()
            app.state.reme_app = reme_app
            await reme_app.async_start()
            app.state.reme_app_started = True
        except Exception as exc:  # keep health reachable for diagnostics
            app.state.startup_error = exc
        try:
            yield
        finally:
            reme_app = app.state.reme_app
            if reme_app is not None and app.state.reme_app_started:
                try:
                    await reme_app.async_stop()
                except Exception:
                    pass

    api = FastAPI(
        title=SERVICE_NAME,
        version=API_VERSION,
        lifespan=lifespan,
    )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return _error_response(
            422,
            "invalid_request",
            "Request validation failed",
            details={"errors": _serializable_validation_errors(exc)},
        )

    @api.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        return _error_response(
            500,
            "internal_error",
            "An unexpected server error occurred",
            details={"exception_type": type(exc).__name__},
        )

    @api.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request):
        startup_error = request.app.state.startup_error
        reme_app = request.app.state.reme_app
        if startup_error is not None or reme_app is None:
            detail = "ReMeApp is not initialized"
            if startup_error is not None:
                detail = f"{type(startup_error).__name__}: {startup_error}"
            response = HealthResponse(
                status="error",
                llm=ComponentHealth(ready=False, detail=detail),
                embedding=ComponentHealth(ready=False, detail=detail),
                vector_store=ComponentHealth(ready=False, detail=detail),
            )
            return JSONResponse(status_code=503, content=jsonable_encoder(response))

        response = await checker.check(reme_app)
        if response.status != "ok":
            return JSONResponse(status_code=503, content=jsonable_encoder(response))
        return response

    @api.post("/api/v1/memory/retrieve")
    async def retrieve(_: RetrieveRequest):
        return _error_response(
            501,
            "phase_not_implemented",
            "Memory retrieval is introduced in phase A1",
        )

    @api.post("/api/v1/memory/finish-trial")
    async def finish_trial(_: FinishTrialRequest):
        return _error_response(
            501,
            "phase_not_implemented",
            "Offline finish-trial processing is introduced in phase A2",
        )

    return api


app = create_hiagent_api()
