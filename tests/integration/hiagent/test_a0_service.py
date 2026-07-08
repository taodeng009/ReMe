"""Local phase-A0/A1 tests; all external ReMe dependencies are mocked."""

from types import SimpleNamespace

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


class FakeVectorStore:
    def __init__(self, nodes=None):
        self.nodes = list(nodes or [])
        self.search_calls = []

    async def async_search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.nodes[: kwargs["top_k"]]


class FakeEmbeddingModel:
    def get_embeddings(self, input_text):
        assert input_text == "goal"
        return [1.0, 0.0]


def task_node(memory_id, when_to_use, content, *, validation_score=0.8, retrieval_score=0.7, metadata=None):
    return SimpleNamespace(
        unique_id=memory_id,
        workspace_id="alfworld/test",
        content=when_to_use,
        score=retrieval_score,
        metadata={
            "memory_type": "task",
            "content": content,
            "score": validation_score,
            "time_created": "2026-01-01 00:00:00",
            "time_modified": "2026-01-01 00:00:00",
            "author": "test",
            "metadata": metadata or {},
        },
    )


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


def test_a1_empty_retrieve_and_a2_route_status():
    vector_store = FakeVectorStore()
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
    )
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

    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["memory_prompt"] == ""
    assert retrieve_response.json()["memories"] == []
    assert retrieve_response.json()["diagnostics"]["candidate_count"] == 0
    assert finish_response.status_code == 501
    assert finish_response.json()["error"]["code"] == "phase_not_implemented"


def test_a1_retrieve_preserves_raw_query_and_separates_scores():
    vector_store = FakeVectorStore(
        [
            task_node(
                "memory-1",
                "When heating an object in a microwave.",
                "Close the microwave before heating.",
                validation_score=0.91,
                retrieval_score=0.73,
                metadata={"source": "trajectory-1"},
            )
        ]
    )
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
    )

    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={"workspace_id": "alfworld/test", "query": "heat an apple", "top_k": 5},
        )

    assert response.status_code == 200
    assert vector_store.search_calls == [
        {"query": "heat an apple", "workspace_id": "alfworld/test", "top_k": 5}
    ]
    memory = response.json()["memories"][0]
    assert memory["validation_score"] == 0.91
    assert memory["retrieval_score"] == 0.73
    assert memory["metadata"] == {"source": "trajectory-1"}
    assert response.json()["memory_prompt"] == (
        "Memory 1:\n"
        " When to use: When heating an object in a microwave.\n"
        " Content: Close the microwave before heating.\n"
    )


def test_a1_context_budget_never_returns_partial_memory():
    vector_store = FakeVectorStore([task_node("memory-1", "When heating food.", "Close the microwave.")])
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
    )

    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={
                "workspace_id": "alfworld/test",
                "query": "heat an apple",
                "max_context_chars": 1,
            },
        )

    assert response.status_code == 200
    assert response.json()["memory_prompt"] == ""
    assert response.json()["memories"] == []
    assert response.json()["diagnostics"]["truncated_count"] == 1


def test_a1_filters_only_on_retrieval_score_and_deduplicates_content():
    vector_store = FakeVectorStore(
        [
            task_node("memory-low", "Condition A", "Same content", validation_score=0.99, retrieval_score=0.2),
            task_node("memory-high", "Condition B", "Useful content", validation_score=0.1, retrieval_score=0.8),
            task_node("memory-duplicate", "Condition C", "Useful content", retrieval_score=0.9),
        ]
    )
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
    )

    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={"workspace_id": "alfworld/test", "query": "goal", "min_score": 0.5},
        )

    assert response.status_code == 200
    assert [memory["memory_id"] for memory in response.json()["memories"]] == ["memory-high"]
    assert response.json()["memories"][0]["validation_score"] == 0.1


def test_a1_rejects_rerank_and_rewrite():
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=FakeVectorStore,
    )
    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={"workspace_id": "alfworld/test", "query": "goal", "rerank": True},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_option"


def test_a1_computes_cosine_when_backend_omits_score():
    node = task_node("memory-1", "Condition", "Content")
    node.score = None
    node.embedding = [0.5, 0.0]
    vector_store = FakeVectorStore([node])
    vector_store.embedding_model = FakeEmbeddingModel()
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
    )

    with TestClient(api) as client:
        response = client.post(
            "/api/v1/memory/retrieve",
            json={"workspace_id": "alfworld/test", "query": "goal"},
        )

    assert response.status_code == 200
    assert response.json()["memories"][0]["retrieval_score"] == 1.0


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
    monkeypatch.setattr(service, "_apply_native_embedding_dimensions", lambda: None)
    api = create_hiagent_api(readiness_checker=ready_checker())
    with TestClient(api) as client:
        assert client.get("/api/v1/health").status_code == 200

    assert received_overrides == [
        "llm.default.model_name=test-chat-model",
        "embedding_model.default.model_name=test-embedding-model",
    ]
