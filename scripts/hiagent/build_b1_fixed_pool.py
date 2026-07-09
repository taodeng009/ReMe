"""Build a phase-B1 fixed ReMe memory pool from normalized HiAgent trajectories.

This utility intentionally stays outside the ReMe service runtime. It reads the
normalized B1 JSONL produced for HiAgent/ALFWorld, adds the target workspace_id,
and sequentially submits each successful trajectory to:

    POST /api/v1/memory/finish-trial

It can also run a lightweight retrieval smoke check against:

    POST /api/v1/memory/retrieve

The script uses only the Python standard library so it can run in a minimal
server environment without installing extra CLI dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8002"
DEFAULT_INPUT = "data/test134_b1/standard_trajectories.jsonl"


class B1InputError(ValueError):
    """Raised when a normalized B1 record is not suitable for import."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]
    elapsed_s: float


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise B1InputError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise B1InputError(f"{path}:{line_no}: each JSONL row must be an object")
            records.append(value)
    if not records:
        raise B1InputError(f"{path}: no records found")
    return records


def _resolve_input_path(input_path: str) -> Path:
    path = Path(input_path)
    if path.is_dir():
        path = path / "standard_trajectories.jsonl"
    if not path.exists():
        raise B1InputError(f"input path does not exist: {path}")
    if not path.is_file():
        raise B1InputError(f"input path is not a file: {path}")
    return path


def _validate_record(record: dict[str, Any], index: int) -> None:
    prefix = f"record[{index}]"
    request_id = record.get("request_id")
    trajectory = record.get("trajectory")
    outcome = record.get("outcome")

    if not isinstance(request_id, str) or not request_id.strip():
        raise B1InputError(f"{prefix}: request_id must be a non-empty string")
    if not isinstance(trajectory, dict):
        raise B1InputError(f"{prefix}: trajectory must be an object")
    if not isinstance(outcome, dict):
        raise B1InputError(f"{prefix}: outcome must be an object")

    trajectory_id = trajectory.get("trajectory_id")
    if trajectory_id != request_id:
        raise B1InputError(
            f"{prefix}: request_id must equal trajectory.trajectory_id "
            f"for phase A2/A3 idempotency; got {request_id!r} vs {trajectory_id!r}",
        )

    metadata = trajectory.get("metadata")
    if not isinstance(metadata, dict):
        raise B1InputError(f"{prefix}: trajectory.metadata must be an object")
    query = metadata.get("query")
    if not isinstance(query, str) or not query.strip():
        raise B1InputError(f"{prefix}: trajectory.metadata.query must be a non-empty string")

    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise B1InputError(f"{prefix}: trajectory.messages must be a non-empty list")
    for msg_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise B1InputError(f"{prefix}: message[{msg_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise B1InputError(f"{prefix}: message[{msg_index}].role is invalid: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise B1InputError(f"{prefix}: message[{msg_index}].content must be non-empty")

    if outcome.get("success") is not True:
        raise B1InputError(f"{prefix}: B1 fixed-pool import only accepts successful trajectories")
    progress_rate = outcome.get("progress_rate")
    if progress_rate is not None:
        if not isinstance(progress_rate, (int, float)) or not 0.0 <= float(progress_rate) <= 1.0:
            raise B1InputError(f"{prefix}: outcome.progress_rate must be between 0 and 1")


def _build_finish_trial_payload(record: dict[str, Any], workspace_id: str) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "request_id": record["request_id"].strip(),
        "retrieval_id": None,
        "trajectory": record["trajectory"],
        "outcome": record["outcome"],
    }


def _build_retrieve_payload(
    *,
    workspace_id: str,
    query: str,
    top_k: int,
    max_context_chars: int,
) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "query": query,
        "top_k": top_k,
        "rerank": False,
        "rewrite": False,
        "max_context_chars": max_context_chars,
    }


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout_s: float) -> HttpResult:
    url = base_url.rstrip("/") + endpoint
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            elapsed_s = time.perf_counter() - start
            return HttpResult(status=response.status, body=_decode_json(raw), elapsed_s=elapsed_s)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed_s = time.perf_counter() - start
        return HttpResult(status=exc.code, body=_decode_json(raw), elapsed_s=elapsed_s)


def _decode_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _unique_queries(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for record in records:
        query = record["trajectory"]["metadata"]["query"].strip()
        if query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


def _dedup_decision_counts(response_body: dict[str, Any]) -> dict[str, int]:
    diagnostics = response_body.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    decisions = diagnostics.get("dedup_decisions")
    if not isinstance(decisions, list):
        return {}
    counts: dict[str, int] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision", "unknown"))
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a phase-B1 fixed ReMe memory pool from normalized HiAgent trajectories.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to standard_trajectories.jsonl, or a directory containing it. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Target ReMe workspace_id. Must match REME_HIAGENT_WORKSPACE_ID if the API is fixed to one workspace.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"ReMe API base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds per request.")
    parser.add_argument("--limit", type=int, default=None, help="Import at most this many records.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without calling ReMe API.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed import/retrieve request.")
    parser.add_argument(
        "--retrieve-check",
        action="store_true",
        help="After import, run retrieve smoke checks using the trajectory queries.",
    )
    parser.add_argument("--retrieve-limit", type=int, default=5, help="Maximum number of queries for retrieve checks.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieve checks.")
    parser.add_argument("--max-context-chars", type=int, default=3000, help="max_context_chars for retrieve checks.")
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON report. Parent directories are created automatically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        input_path = _resolve_input_path(args.input)
        records = _load_jsonl(input_path)
        if args.limit is not None:
            if args.limit < 1:
                raise B1InputError("--limit must be positive")
            records = records[: args.limit]
        for index, record in enumerate(records, start=1):
            _validate_record(record, index)
    except B1InputError as exc:
        print(f"[B1][input-error] {exc}", file=sys.stderr)
        return 2

    request_ids = [record["request_id"] for record in records]
    duplicate_request_ids = sorted({rid for rid in request_ids if request_ids.count(rid) > 1})
    if duplicate_request_ids:
        print(f"[B1][input-error] duplicate request_id values: {duplicate_request_ids}", file=sys.stderr)
        return 2

    print(
        f"[B1] input={input_path} records={len(records)} "
        f"workspace_id={args.workspace_id} dry_run={args.dry_run}",
    )

    report: dict[str, Any] = {
        "input": str(input_path),
        "workspace_id": args.workspace_id,
        "base_url": args.base_url,
        "dry_run": args.dry_run,
        "record_count": len(records),
        "finish_trial": [],
        "retrieve_check": [],
    }

    if args.dry_run:
        sample_payload = _build_finish_trial_payload(records[0], args.workspace_id)
        report["sample_finish_trial_payload"] = sample_payload
        print("[B1] dry-run passed")
        print("[B1] sample finish-trial payload:")
        print(_json_dumps(sample_payload))
        if args.report:
            _write_report(Path(args.report), report)
            print(f"[B1] report written: {args.report}")
        return 0

    failures = 0
    committed_total = 0
    for index, record in enumerate(records, start=1):
        payload = _build_finish_trial_payload(record, args.workspace_id)
        result = _post_json(args.base_url, "/api/v1/memory/finish-trial", payload, args.timeout)
        learning = result.body.get("learning") if isinstance(result.body.get("learning"), dict) else {}
        memories_committed = int(learning.get("memories_committed", 0) or 0)
        candidates_deduplicated = int(learning.get("candidates_deduplicated", 0) or 0)
        dedup_counts = _dedup_decision_counts(result.body)
        committed_total += memories_committed
        ok = 200 <= result.status < 300
        if not ok:
            failures += 1
        report["finish_trial"].append(
            {
                "index": index,
                "request_id": record["request_id"],
                "status": result.status,
                "elapsed_s": round(result.elapsed_s, 3),
                "memories_committed": memories_committed,
                "candidates_deduplicated": candidates_deduplicated,
                "dedup_decision_counts": dedup_counts,
                "response": result.body,
            },
        )
        print(
            f"[B1][finish] {index}/{len(records)} request_id={record['request_id']} "
            f"status={result.status} committed={memories_committed} "
            f"deduped={candidates_deduplicated} decisions={dedup_counts} "
            f"elapsed={result.elapsed_s:.2f}s",
        )
        if not ok and args.fail_fast:
            break

    if args.retrieve_check and not (failures and args.fail_fast):
        queries = _unique_queries(records)[: args.retrieve_limit]
        for index, query in enumerate(queries, start=1):
            payload = _build_retrieve_payload(
                workspace_id=args.workspace_id,
                query=query,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
            )
            result = _post_json(args.base_url, "/api/v1/memory/retrieve", payload, args.timeout)
            diagnostics = result.body.get("diagnostics") if isinstance(result.body.get("diagnostics"), dict) else {}
            returned_count = int(diagnostics.get("returned_count", 0) or 0)
            ok = 200 <= result.status < 300
            if not ok:
                failures += 1
            report["retrieve_check"].append(
                {
                    "index": index,
                    "query": query,
                    "status": result.status,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "returned_count": returned_count,
                    "response": result.body,
                },
            )
            print(
                f"[B1][retrieve] {index}/{len(queries)} status={result.status} "
                f"returned={returned_count} query={query!r}",
            )
            if not ok and args.fail_fast:
                break

    report["summary"] = {
        "finish_requests": len(report["finish_trial"]),
        "retrieve_requests": len(report["retrieve_check"]),
        "failures": failures,
        "memories_committed_total": committed_total,
    }
    if args.report:
        _write_report(Path(args.report), report)
        print(f"[B1] report written: {args.report}")

    print(
        f"[B1] done finish_requests={len(report['finish_trial'])} "
        f"retrieve_requests={len(report['retrieve_check'])} "
        f"memories_committed_total={committed_total} failures={failures}",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
