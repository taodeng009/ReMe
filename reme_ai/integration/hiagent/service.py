"""FastAPI service for the phase-A HiAgent/ReMe integration."""

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
    FeedbackSummary,
    FinishTrialDiagnostics,
    FinishTrialRequest,
    FinishTrialResponse,
    HealthResponse,
    LearningSummary,
    MaintenanceSummary,
    RetrievedMemory,
    RetrieveDiagnostics,
    RetrieveRequest,
    RetrieveResponse,
)
from reme_ai.schema.memory import TaskMemory


Probe = Callable[[Any], bool | ComponentHealth | Awaitable[bool | ComponentHealth]]
VectorStoreGetter = Callable[[], Any]
LLMGetter = Callable[[], Any]
FinishTrialProcessor = Callable[[FinishTrialRequest], Awaitable[FinishTrialResponse]]
_VALIDATION_SCORE_METADATA_KEY = "_hiagent_validation_score"
_DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.5
_FLOWLLM_DEFAULT_LLM = object()
_MAX_RERANK_CANDIDATE_K = 100
_MEMORY_RERANK_PROMPT = """You are an expert AI analyst tasked with reranking retrieved experiences based on their relevance to a specific query.

Your task is to analyze the candidates and rank them by relevance, considering:
- DIRECT RELEVANCE: How directly applicable the experience is to the current query
- SITUATION SIMILARITY: How similar the experience context is to the current situation
- ACTIONABILITY: How actionable and specific the experience is
- QUALITY: The overall quality and clarity of the experience

# Current Query
{query}

# Candidate Experiences (Total: {num_candidates})
{candidates}

OUTPUT FORMAT:
Provide a ranked list of candidate indices (0-based) from most relevant to least relevant:
```json
{{
  "ranked_indices": [2, 0, 4, 1, 3],
  "reasoning": "Brief explanation of ranking rationale"
}}
```

Note: Include ALL candidate indices in the ranking, even if some are less relevant."""


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
    embedding_dimensions = os.getenv("REME_HIAGENT_EMBEDDING_DIMENSIONS", "").strip()
    vector_store_backend = os.getenv("REME_HIAGENT_VECTOR_STORE_BACKEND", "").strip()
    vector_store_path = os.getenv("REME_HIAGENT_VECTOR_STORE_PATH", "").strip()
    if llm_model:
        overrides.append(f"llm.default.model_name={llm_model}")
    if embedding_model:
        overrides.append(f"embedding_model.default.model_name={embedding_model}")
    if llm_backend:
        overrides.append(f"llm.default.backend={llm_backend}")
    if embedding_backend:
        overrides.append(f"embedding_model.default.backend={embedding_backend}")
    if embedding_dimensions and embedding_dimensions.lower() not in {"native", "none"}:
        dimensions = int(embedding_dimensions)
        if dimensions <= 0:
            raise ValueError("REME_HIAGENT_EMBEDDING_DIMENSIONS must be positive or 'native'")
        overrides.append(f"embedding_model.default.params={{'dimensions': {dimensions}}}")
    if vector_store_backend:
        overrides.append(f"vector_store.default.backend={vector_store_backend}")
        if vector_store_backend == "local":
            if not vector_store_path:
                raise ValueError("REME_HIAGENT_VECTOR_STORE_PATH is required for the local backend")
            storage_path = Path(vector_store_path).expanduser()
            if not storage_path.is_absolute():
                raise ValueError("REME_HIAGENT_VECTOR_STORE_PATH must be an absolute path")
            overrides.append(f"vector_store.default.params.store_dir={storage_path}")
    return ReMeApp(*overrides)


def _resolve_workspace_config(
    workspace_mode: str | None,
    workspace_id: str | None,
    *,
    use_environment: bool = True,
) -> tuple[str, str | None]:
    env_mode = os.getenv("REME_HIAGENT_WORKSPACE_MODE", "read_write") if use_environment else "read_write"
    mode = (workspace_mode or env_mode).strip().lower()
    if mode not in {"read_write", "read_only"}:
        raise ValueError("workspace mode must be 'read_write' or 'read_only'")
    env_workspace_id = os.getenv("REME_HIAGENT_WORKSPACE_ID", "") if use_environment else ""
    configured_workspace_id = (workspace_id or env_workspace_id).strip() or None
    return mode, configured_workspace_id


def _resolve_dedup_similarity_threshold() -> float:
    raw = os.getenv("REME_HIAGENT_DEDUP_SIMILARITY_THRESHOLD", "").strip()
    if not raw:
        return _DEFAULT_DEDUP_SIMILARITY_THRESHOLD
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError("REME_HIAGENT_DEDUP_SIMILARITY_THRESHOLD must be a float") from exc
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("REME_HIAGENT_DEDUP_SIMILARITY_THRESHOLD must be between -1.0 and 1.0")
    return threshold


def _resolve_rerank_candidate_k() -> int | None:
    raw = os.getenv("REME_HIAGENT_RERANK_CANDIDATE_K", "").strip()
    if not raw:
        return None
    try:
        candidate_k = int(raw)
    except ValueError as exc:
        raise ValueError("REME_HIAGENT_RERANK_CANDIDATE_K must be an integer") from exc
    if not 1 <= candidate_k <= _MAX_RERANK_CANDIDATE_K:
        raise ValueError(f"REME_HIAGENT_RERANK_CANDIDATE_K must be between 1 and {_MAX_RERANK_CANDIDATE_K}")
    return candidate_k


def _apply_native_embedding_dimensions() -> None:
    """Make the OpenAI SDK omit ``dimensions`` for non-Matryoshka models."""

    mode = os.getenv("REME_HIAGENT_EMBEDDING_DIMENSIONS", "").strip().lower()
    if mode not in {"native", "none"}:
        return

    from flowllm.core.context import C
    from openai import NOT_GIVEN

    embedding_model = C.get_vector_store("default").embedding_model
    embedding_model.dimensions = NOT_GIVEN


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


def _get_default_vector_store():
    from flowllm.core.context import C

    return C.get_vector_store("default")


def _get_default_llm(reme_app: Any | None = None):
    """Best-effort lookup for the default LLM instance used by FlowLLM ops."""

    sources = [reme_app, getattr(reme_app, "context", None)]
    try:
        from flowllm.core.context import C

        sources.append(C)
    except ImportError:
        pass

    for source in sources:
        if source is None:
            continue
        for attr_name in ("default_llm", "llm", "llms", "llm_dict"):
            value = getattr(source, attr_name, None)
            if isinstance(value, Mapping):
                value = value.get("default") or next(iter(value.values()), None)
            if value is not None and hasattr(value, "achat"):
                return value
    return None


def _score_from_search_node(node: Any) -> float | None:
    for attribute in ("retrieval_score", "similarity_score", "similarity", "score"):
        value = getattr(node, attribute, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    metadata = getattr(node, "metadata", {}) or {}
    for key in ("retrieval_score", "similarity_score", "similarity"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if left is None or right is None or len(left) == 0 or len(right) == 0 or len(left) != len(right):
        return None
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot_product / (left_norm * right_norm)


async def _fill_missing_retrieval_scores(query: str, nodes: list[Any], vector_store: Any) -> list[float | None]:
    scores = [_score_from_search_node(node) for node in nodes]
    if all(score is not None for score in scores):
        return scores

    embedding_model = getattr(vector_store, "embedding_model", None)
    if embedding_model is None:
        return scores

    method = getattr(embedding_model, "aget_embeddings", None) or getattr(embedding_model, "get_embeddings", None)
    if method is None:
        return scores

    query_embedding = method(query)
    if inspect.isawaitable(query_embedding):
        query_embedding = await query_embedding

    missing_embedding_indexes = []
    missing_embedding_texts = []
    for index, node in enumerate(nodes):
        if scores[index] is None:
            node_embedding = getattr(node, "embedding", None)
            if node_embedding:
                scores[index] = _cosine_similarity(query_embedding, node_embedding)
            else:
                metadata = getattr(node, "metadata", {}) or {}
                when_to_use = str(getattr(node, "content", "") or "")
                content = str(metadata.get("content", "") or "")
                missing_embedding_indexes.append(index)
                missing_embedding_texts.append(f"{when_to_use} {content}".strip())

    if missing_embedding_texts:
        generated_embeddings = method(missing_embedding_texts)
        if inspect.isawaitable(generated_embeddings):
            generated_embeddings = await generated_embeddings
        for index, node_embedding in zip(missing_embedding_indexes, generated_embeddings):
            scores[index] = _cosine_similarity(query_embedding, node_embedding)
    return scores


def _format_memory_block(index: int, memory: RetrievedMemory) -> str:
    return (
        f"Memory {index}:\n"
        f" When to use: {memory.when_to_use}\n"
        f" Content: {memory.content}\n"
    )


def _sort_by_retrieval_score_desc(memories: list[RetrievedMemory]) -> list[RetrievedMemory]:
    """Sort retrieved memories by score while keeping score-less entries last."""

    return sorted(
        memories,
        key=lambda memory: (
            memory.retrieval_score is not None,
            memory.retrieval_score if memory.retrieval_score is not None else -math.inf,
        ),
        reverse=True,
    )


def _format_candidates_for_llm_rerank(candidates: list[RetrievedMemory]) -> str:
    formatted_candidates = []
    for index, candidate in enumerate(candidates):
        formatted_candidates.append(
            "\n".join(
                [
                    f"Candidate {index}:",
                    f"Condition: {candidate.when_to_use}",
                    f"Experience: {candidate.content}",
                ]
            )
        )
    return "\n---\n".join(formatted_candidates)


def _parse_llm_rerank_indices(response: str, candidate_count: int) -> list[int]:
    try:
        json_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", response)
        parsed: Any | None = None
        if json_blocks:
            parsed = json.loads(json_blocks[0])
        else:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None

        if isinstance(parsed, dict) and isinstance(parsed.get("ranked_indices"), list):
            raw_indices = parsed["ranked_indices"]
        elif isinstance(parsed, list):
            raw_indices = parsed
        else:
            raw_indices = re.findall(r"\b\d+\b", response)

        ranked_indices = []
        seen = set()
        for value in raw_indices:
            if isinstance(value, bool):
                continue
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < candidate_count and index not in seen:
                ranked_indices.append(index)
                seen.add(index)
        return ranked_indices
    except Exception:
        return []


async def _llm_rerank_retrieved_memories(
    query: str,
    candidates: list[RetrievedMemory],
    llm: Any,
) -> tuple[list[RetrievedMemory], bool]:
    if not candidates or llm is None:
        return candidates, False

    if llm is _FLOWLLM_DEFAULT_LLM:
        try:
            from flowllm.core.context import FlowContext

            from reme_ai.retrieve.task.rerank_memory_op import RerankMemoryOp

            context = FlowContext(query=query)
            context.response.metadata["memory_list"] = list(candidates)
            await _execute_op(
                RerankMemoryOp(
                    enable_llm_rerank=True,
                    enable_score_filter=False,
                    top_k=len(candidates),
                ),
                context,
            )
            reranked = list(context.response.metadata.get("memory_list", candidates))
            return reranked, reranked != candidates
        except Exception:
            return candidates, False

    if not hasattr(llm, "achat"):
        return candidates, False

    prompt = _MEMORY_RERANK_PROMPT.format(
        query=query,
        candidates=_format_candidates_for_llm_rerank(candidates),
        num_candidates=len(candidates),
    )

    try:
        from flowllm.core.enumeration import Role
        from flowllm.core.schema import Message

        response = await llm.achat([Message(role=Role.USER, content=prompt)])
        response_content = getattr(response, "content", str(response))
        ranked_indices = _parse_llm_rerank_indices(response_content, len(candidates))
        if not ranked_indices:
            return candidates, False

        reranked = [candidates[index] for index in ranked_indices]
        ranked_set = set(ranked_indices)
        reranked.extend(candidate for index, candidate in enumerate(candidates) if index not in ranked_set)
        return reranked, True
    except Exception:
        return candidates, False


async def _retrieve_read_only(
    request: RetrieveRequest,
    vector_store: Any,
    llm: Any | None = None,
    *,
    rerank_candidate_k: int | None = None,
) -> RetrieveResponse:
    search_top_k = request.top_k
    if request.rerank and rerank_candidate_k is not None:
        search_top_k = max(request.top_k, rerank_candidate_k)

    nodes = list(
        await vector_store.async_search(
            query=request.query,
            workspace_id=request.workspace_id,
            top_k=search_top_k,
        )
    )
    scores = await _fill_missing_retrieval_scores(request.query, nodes, vector_store)

    candidates: list[RetrievedMemory] = []
    seen_content: set[str] = set()
    skipped_count = 0
    for node, retrieval_score in zip(nodes, scores):
        try:
            if (getattr(node, "metadata", {}) or {}).get("memory_type") != "task":
                skipped_count += 1
                continue
            memory = TaskMemory.from_vector_node(node)
        except (KeyError, TypeError, ValueError):
            skipped_count += 1
            continue

        content_key = str(memory.content)
        if content_key in seen_content:
            skipped_count += 1
            continue
        seen_content.add(content_key)

        if request.min_score is not None:
            if retrieval_score is None or retrieval_score < request.min_score:
                skipped_count += 1
                continue

        public_metadata = dict(memory.metadata)
        validation_score = public_metadata.pop(_VALIDATION_SCORE_METADATA_KEY, memory.score)
        candidates.append(
            RetrievedMemory(
                memory_id=memory.memory_id,
                when_to_use=memory.when_to_use,
                content=str(memory.content),
                validation_score=validation_score,
                retrieval_score=retrieval_score,
                metadata=public_metadata,
            )
        )

    candidates = _sort_by_retrieval_score_desc(candidates)
    reranked = False
    if request.rerank:
        candidates, reranked = await _llm_rerank_retrieved_memories(request.query, candidates, llm)

    exposed: list[RetrievedMemory] = []
    blocks: list[str] = []
    current_length = 0
    for memory in candidates:
        if len(exposed) >= request.top_k:
            break
        block = _format_memory_block(len(exposed) + 1, memory)
        separator_length = 1 if blocks else 0
        if current_length + separator_length + len(block) > request.max_context_chars:
            continue
        blocks.append(block)
        exposed.append(memory)
        current_length += separator_length + len(block)

    memory_prompt = "\n".join(blocks)
    return RetrieveResponse(
        memory_prompt=memory_prompt,
        memories=exposed,
        diagnostics=RetrieveDiagnostics(
            candidate_count=len(nodes),
            returned_count=len(exposed),
            truncated_count=len(candidates) - len(exposed),
            skipped_count=skipped_count,
            reranked=reranked,
        ),
    )


class JsonlRequestLedger:
    """Single-process phase-A2 ledger for completed request idempotency."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get(self, request_id: str) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        found = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("request_id") == request_id:
                    found = record
        return found

    def append_completed(
        self,
        request_id: str,
        payload_hash: str,
        response: FinishTrialResponse,
    ) -> None:
        record = {
            "request_id": request_id,
            "payload_hash": payload_hash,
            "status": "completed",
            **response.learning.model_dump(),
            "response": response.model_dump(mode="json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _payload_hash(request: FinishTrialRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_request_ledger_path() -> Path:
    """Return a repository-anchored path, independent of process cwd."""

    repository_root = Path(__file__).resolve().parents[3]
    run_id = os.getenv("REME_HIAGENT_RUN_ID", "default").strip() or "default"
    return repository_root / "outputs" / "reme_api" / run_id / "request_ledger.jsonl"


async def _execute_op(op: Any, context: Any) -> None:
    await op.async_call(context=context)
    response = context.response
    if hasattr(response, "success") and not response.success:
        raise RuntimeError(response.answer or f"{type(op).__name__} failed")


async def _process_offline_success_trial(request: FinishTrialRequest) -> FinishTrialResponse:
    """Explicit A2 pipeline; deliberately avoids the default summary flow."""

    from flowllm.core.context import FlowContext

    from reme_ai.summary.task.memory_deduplication_op import MemoryDeduplicationOp
    from reme_ai.summary.task.memory_validation_op import MemoryValidationOp
    from reme_ai.summary.task.success_extraction_op import SuccessExtractionOp
    from reme_ai.summary.task.trajectory_preprocess_op import TrajectoryPreprocessOp
    from reme_ai.vector_store.update_vector_store_op import UpdateVectorStoreOp

    trajectory = {
        "messages": [message.model_dump(mode="json") for message in request.trajectory.messages],
        # HiAgent's boolean outcome is authoritative at this boundary. The
        # ReMe success classifier expects a normalized score of 1.0.
        "score": 1.0,
        "metadata": dict(request.trajectory.metadata),
    }
    context = FlowContext(workspace_id=request.workspace_id, trajectories=[trajectory])

    await _execute_op(TrajectoryPreprocessOp(success_threshold=1.0), context)
    await _execute_op(SuccessExtractionOp(), context)
    candidates = list(context.get("success_task_memories", []))
    for memory in candidates:
        memory.metadata.update(
            {
                "source_request_id": request.request_id,
                "source_trajectory_id": request.trajectory.trajectory_id,
                "extractor_type": "success",
            }
        )

    await _execute_op(MemoryValidationOp(validation_threshold=0.5), context)
    validated = list(context.response.metadata.get("memory_list", []))
    for memory in validated:
        # Some vector-store search implementations reuse/overwrite the generic
        # node metadata "score". Preserve validation quality independently.
        memory.metadata[_VALIDATION_SCORE_METADATA_KEY] = memory.score

    dedup_similarity_threshold = _resolve_dedup_similarity_threshold()
    await _execute_op(MemoryDeduplicationOp(similarity_threshold=dedup_similarity_threshold), context)
    deduplicated = list(context.response.metadata.get("memory_list", []))
    dedup_decisions = list(context.response.metadata.get("dedup_decisions", []))

    await _execute_op(UpdateVectorStoreOp(), context)
    update_result = context.response.metadata.get("update_result", {})
    committed_count = int(update_result.get("inserted_count", len(deduplicated)))

    return FinishTrialResponse(
        feedback=FeedbackSummary(),
        learning=LearningSummary(
            candidates_generated=len(candidates),
            candidates_validated=len(validated),
            candidates_deduplicated=len(validated) - len(deduplicated),
            memories_committed=committed_count,
        ),
        maintenance=MaintenanceSummary(),
        diagnostics=FinishTrialDiagnostics(
            dedup_similarity_threshold=dedup_similarity_threshold,
            dedup_decisions=dedup_decisions,
        ),
    )


def create_hiagent_api(
    *,
    reme_app_factory: Callable[[], Any] | None = None,
    readiness_checker: HiAgentReadinessChecker | None = None,
    vector_store_getter: VectorStoreGetter | None = None,
    llm_getter: LLMGetter | None = None,
    rerank_candidate_k: int | None = None,
    finish_trial_processor: FinishTrialProcessor | None = None,
    request_ledger_path: str | Path | None = None,
    workspace_mode: str | None = None,
    workspace_id: str | None = None,
) -> FastAPI:
    """Create the phase-A0 API with one ReMeApp instance per service lifespan."""

    uses_default_factory = reme_app_factory is None
    if uses_default_factory:
        _load_env_file()
    factory = reme_app_factory or _create_reme_app_from_env
    checker = readiness_checker or HiAgentReadinessChecker()
    get_vector_store = vector_store_getter or _get_default_vector_store
    if llm_getter is not None:
        get_llm = llm_getter
    elif uses_default_factory:
        get_llm = lambda: _get_default_llm(getattr(api.state, "reme_app", None)) or _FLOWLLM_DEFAULT_LLM
    else:
        get_llm = lambda: None
    configured_rerank_candidate_k = rerank_candidate_k
    if configured_rerank_candidate_k is None and uses_default_factory:
        configured_rerank_candidate_k = _resolve_rerank_candidate_k()
    process_finish_trial = finish_trial_processor or _process_offline_success_trial
    configured_mode, configured_workspace_id = _resolve_workspace_config(
        workspace_mode,
        workspace_id,
        use_environment=uses_default_factory,
    )
    request_ledger_env = os.getenv("REME_HIAGENT_REQUEST_LEDGER", "").strip() if uses_default_factory else ""
    ledger = JsonlRequestLedger(
        request_ledger_path
        or request_ledger_env
        or _default_request_ledger_path()
    )
    finish_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.reme_app = None
        app.state.reme_app_started = False
        app.state.startup_error = None
        try:
            reme_app = factory()
            app.state.reme_app = reme_app
            await reme_app.async_start()
            if uses_default_factory:
                _apply_native_embedding_dimensions()
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

    @api.post("/api/v1/memory/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest):
        if configured_workspace_id is not None and request.workspace_id != configured_workspace_id:
            return _error_response(
                403,
                "workspace_not_configured",
                "The service is not configured for the requested workspace",
            )
        if request.rewrite:
            return _error_response(
                400,
                "unsupported_option",
                "Phase B supports rerank but still requires rewrite=false",
            )
        return await _retrieve_read_only(
            request,
            get_vector_store(),
            get_llm() if request.rerank else None,
            rerank_candidate_k=configured_rerank_candidate_k,
        )

    @api.post("/api/v1/memory/finish-trial", response_model=FinishTrialResponse)
    async def finish_trial(request: FinishTrialRequest):
        if configured_workspace_id is not None and request.workspace_id != configured_workspace_id:
            return _error_response(
                403,
                "workspace_not_configured",
                "The service is not configured for the requested workspace",
            )
        if configured_mode == "read_only":
            return _error_response(
                403,
                "workspace_read_only",
                "finish-trial is disabled while the workspace is read-only",
            )
        if request.retrieval_id is not None or not request.outcome.success:
            return _error_response(
                400,
                "unsupported_mode",
                "Phase A2 only supports successful offline imports with retrieval_id=null",
            )

        digest = _payload_hash(request)
        async with finish_lock:
            existing = ledger.get(request.request_id)
            if existing is not None:
                if existing.get("payload_hash") != digest:
                    return _error_response(
                        409,
                        "request_conflict",
                        "request_id was already used with a different payload",
                    )
                if existing.get("status") == "completed":
                    return FinishTrialResponse.model_validate(existing["response"])

            response = await process_finish_trial(request)
            ledger.append_completed(request.request_id, digest, response)
            return response

    return api


app = create_hiagent_api()
