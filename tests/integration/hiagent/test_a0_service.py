"""Local phase-A0/A1/A2 tests; all external ReMe dependencies are mocked."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from reme_ai.integration.hiagent.schemas import ComponentHealth, FinishTrialResponse, LearningSummary
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


class RecomputingEmbeddingModel:
    def __init__(self):
        self.calls = []

    def get_embeddings(self, input_text):
        self.calls.append(input_text)
        if isinstance(input_text, str):
            return [1.0, 0.0]
        return [[0.6, 0.8] for _ in input_text]


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


def finish_payload(request_id="request-1", *, success=True, retrieval_id=None, query="heat an apple"):
    return {
        "workspace_id": "alfworld/test",
        "request_id": request_id,
        "retrieval_id": retrieval_id,
        "trajectory": {
            "trajectory_id": request_id,
            "messages": [],
            "metadata": {"query": query},
        },
        "outcome": {"success": success, "score": 1.0 if success else 0.0, "progress_rate": 1.0},
    }


def test_a1_empty_retrieve_and_a2_success(tmp_path):
    vector_store = FakeVectorStore()

    async def processor(_):
        return FinishTrialResponse()

    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=lambda: vector_store,
        finish_trial_processor=processor,
        request_ledger_path=tmp_path / "ledger.jsonl",
    )
    retrieve_payload = {
        "workspace_id": "alfworld/test",
        "query": "heat an apple",
    }

    with TestClient(api) as client:
        retrieve_response = client.post("/api/v1/memory/retrieve", json=retrieve_payload)
        finish_response = client.post("/api/v1/memory/finish-trial", json=finish_payload())

    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["memory_prompt"] == ""
    assert retrieve_response.json()["memories"] == []
    assert retrieve_response.json()["diagnostics"]["candidate_count"] == 0
    assert finish_response.status_code == 200
    assert finish_response.json()["learning"]["memories_committed"] == 0


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


def test_a1_uses_preserved_validation_score_when_backend_overwrites_generic_score():
    node = task_node(
        "memory-1",
        "Condition",
        "Content",
        validation_score=0.43,
        retrieval_score=0.61,
        metadata={"_hiagent_validation_score": 0.85, "source": "trajectory-1"},
    )
    vector_store = FakeVectorStore([node])
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

    memory = response.json()["memories"][0]
    assert memory["validation_score"] == 0.85
    assert memory["retrieval_score"] == 0.61
    assert memory["metadata"] == {"source": "trajectory-1"}


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


def test_a1_recomputes_node_embedding_when_score_and_embedding_are_missing():
    node = task_node("memory-1", "Condition", "Content")
    node.score = None
    embedding_model = RecomputingEmbeddingModel()
    vector_store = FakeVectorStore([node])
    vector_store.embedding_model = embedding_model
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
    assert response.json()["memories"][0]["retrieval_score"] == 0.6
    assert embedding_model.calls == ["goal", ["Condition Content"]]


def test_a2_completed_zero_memory_request_is_idempotent_across_restart(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    calls = []

    async def processor(request):
        calls.append(request.request_id)
        return FinishTrialResponse(learning=LearningSummary(candidates_generated=2, memories_committed=0))

    def make_api():
        return create_hiagent_api(
            reme_app_factory=FakeReMeApp,
            readiness_checker=ready_checker(),
            vector_store_getter=FakeVectorStore,
            finish_trial_processor=processor,
            request_ledger_path=ledger_path,
        )

    with TestClient(make_api()) as client:
        first = client.post("/api/v1/memory/finish-trial", json=finish_payload())
        duplicate = client.post("/api/v1/memory/finish-trial", json=finish_payload())

    with TestClient(make_api()) as client:
        after_restart = client.post("/api/v1/memory/finish-trial", json=finish_payload())

    assert first.status_code == duplicate.status_code == after_restart.status_code == 200
    assert calls == ["request-1"]
    assert after_restart.json() == first.json()
    assert '"status": "completed"' in ledger_path.read_text(encoding="utf-8")
    assert '"memories_committed": 0' in ledger_path.read_text(encoding="utf-8")


def test_a2_same_request_id_with_different_payload_conflicts(tmp_path):
    async def processor(_):
        return FinishTrialResponse()

    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=FakeVectorStore,
        finish_trial_processor=processor,
        request_ledger_path=tmp_path / "ledger.jsonl",
    )
    with TestClient(api) as client:
        assert client.post("/api/v1/memory/finish-trial", json=finish_payload()).status_code == 200
        conflict = client.post(
            "/api/v1/memory/finish-trial",
            json=finish_payload(query="heat a potato"),
        )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "request_conflict"


def test_a2_rejects_online_and_failure_modes_before_processing(tmp_path):
    calls = []

    async def processor(request):
        calls.append(request.request_id)
        return FinishTrialResponse()

    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        vector_store_getter=FakeVectorStore,
        finish_trial_processor=processor,
        request_ledger_path=tmp_path / "ledger.jsonl",
    )
    with TestClient(api) as client:
        failure = client.post(
            "/api/v1/memory/finish-trial",
            json=finish_payload(request_id="failure", success=False),
        )
        online = client.post(
            "/api/v1/memory/finish-trial",
            json=finish_payload(request_id="online", retrieval_id="retrieval-1"),
        )

    assert failure.status_code == online.status_code == 400
    assert failure.json()["error"]["code"] == "unsupported_mode"
    assert online.json()["error"]["code"] == "unsupported_mode"
    assert calls == []


def test_a2_requires_raw_query_in_trajectory_metadata(tmp_path):
    payload = finish_payload()
    payload["trajectory"]["metadata"] = {}
    api = create_hiagent_api(
        reme_app_factory=FakeReMeApp,
        readiness_checker=ready_checker(),
        request_ledger_path=tmp_path / "ledger.jsonl",
    )

    with TestClient(api) as client:
        response = client.post("/api/v1/memory/finish-trial", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


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
