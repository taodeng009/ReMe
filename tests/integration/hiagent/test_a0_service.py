"""Local phase-A0 tests; all external ReMe dependencies are mocked."""

from fastapi.testclient import TestClient

from reme_ai.integration.hiagent.schemas import ComponentHealth
from reme_ai.integration.hiagent import service
from reme_ai.integration.hiagent.service import HiAgentReadinessChecker, create_hiagent_api


class FakeReMeApp:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    async def async_start(self):
        self.started += 1

    async def async_stop(self):
        self.stopped += 1


def ready_checker(**overrides):
    values = {"llm": True, "embedding": True, "vector_store": True}
    values.update(overrides)
    return HiAgentReadinessChecker(probes={name: (lambda _, value=value: value) for name, value in values.items()})


def test_lifespan_initializes_one_reme_app():
    instances = []

    def factory():
        instance = FakeReMeApp()
        instances.append(instance)
        return instance

    api = create_hiagent_api(reme_app_factory=factory, readiness_checker=ready_checker())
    with TestClient(api) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 200

    assert len(instances) == 1
    assert instances[0].started == 1
    assert instances[0].stopped == 1


def test_health_reports_dependency_failure():
    checker = ready_checker(embedding=ComponentHealth(ready=False, detail="embedding unavailable"))
    api = create_hiagent_api(reme_app_factory=FakeReMeApp, readiness_checker=checker)

    with TestClient(api) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["embedding"] == {"ready": False, "detail": "embedding unavailable"}


def test_health_reports_startup_failure():
    def failing_factory():
        raise RuntimeError("bad configuration")

    api = create_hiagent_api(reme_app_factory=failing_factory, readiness_checker=ready_checker())
    with TestClient(api) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["llm"]["ready"] is False
    assert "bad configuration" in response.json()["llm"]["detail"]


def test_a1_and_a2_routes_are_explicitly_unavailable():
    api = create_hiagent_api(reme_app_factory=FakeReMeApp, readiness_checker=ready_checker())
    retrieve_payload = {
        "workspace_id": "alfworld/test",
        "query": "heat an apple",
    }
    finish_payload = {
        "workspace_id": "alfworld/test",
        "request_id": "request-1",
        "retrieval_id": None,
        "trajectory": {
            "trajectory_id": "trajectory-1",
            "messages": [],
            "metadata": {"query": "heat an apple"},
        },
        "outcome": {"success": True, "score": 1.0, "progress_rate": 1.0},
    }

    with TestClient(api) as client:
        retrieve_response = client.post("/api/v1/memory/retrieve", json=retrieve_payload)
        finish_response = client.post("/api/v1/memory/finish-trial", json=finish_payload)

    assert retrieve_response.status_code == 501
    assert finish_response.status_code == 501
    assert retrieve_response.json()["error"]["code"] == "phase_not_implemented"


def test_validation_errors_use_common_error_contract():
    api = create_hiagent_api(reme_app_factory=FakeReMeApp, readiness_checker=ready_checker())
    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={"workspace_id": " ", "query": "", "top_k": 0},
        )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["details"]["errors"]


def test_default_factory_reads_model_names_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REME_HIAGENT_LLM_MODEL=test-chat-model\n"
        "REME_HIAGENT_EMBEDDING_MODEL=test-embedding-model\n"
        "REME_HIAGENT_EMBEDDING_DIMENSIONS=native\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REME_HIAGENT_ENV_FILE", str(env_file))
    monkeypatch.delenv("REME_HIAGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("REME_HIAGENT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("REME_HIAGENT_LLM_BACKEND", raising=False)
    monkeypatch.delenv("REME_HIAGENT_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("REME_HIAGENT_EMBEDDING_DIMENSIONS", raising=False)
    received_overrides = []

    class CapturingReMeApp(FakeReMeApp):
        def __init__(self, *overrides):
            super().__init__()
            received_overrides.extend(overrides)

    monkeypatch.setattr(service, "ReMeApp", CapturingReMeApp)
    api = create_hiagent_api(readiness_checker=ready_checker())
    with TestClient(api) as client:
        assert client.get("/api/v1/health").status_code == 200

    assert received_overrides == [
        "llm.default.model_name=test-chat-model",
        "embedding_model.default.model_name=test-embedding-model",
        "embedding_model.default.params={}",
    ]
