"""Small bounded benchmark helpers for local serving endpoints.

Now includes a streaming benchmark that captures TTFT / TPOT / ITL / E2E
percentiles, matching what ``vllm bench serve`` produces (item P). The
sequential helpers below stay for backward compat and embedding endpoints.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Protocol
from urllib.request import Request, urlopen


Clock = Callable[[], float]


class PostJson(Protocol):
    def __call__(self, url: str, payload: dict, *, timeout: float) -> dict:
        ...


@dataclass
class TokenTimeline:
    """Per-request token-arrival timeline used to compute TTFT/TPOT/ITL."""

    request_started_ms: float
    first_token_ms: float | None = None
    last_token_ms: float | None = None
    token_count: int = 0
    inter_token_ms: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_ms is None:
            return None
        return self.first_token_ms - self.request_started_ms

    @property
    def tpot_ms(self) -> float | None:
        if self.token_count <= 1 or self.first_token_ms is None or self.last_token_ms is None:
            return None
        decode_span = self.last_token_ms - self.first_token_ms
        return decode_span / max(1, self.token_count - 1)

    @property
    def e2e_ms(self) -> float | None:
        if self.last_token_ms is None:
            return None
        return self.last_token_ms - self.request_started_ms


@dataclass
class BenchResult:
    """Streaming benchmark output (matches ``vllm bench serve`` schema)."""

    ttft_ms_p50: float | None
    ttft_ms_p99: float | None
    tpot_ms_p50: float | None
    tpot_ms_p99: float | None
    itl_ms_p99: float | None
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    e2e_p99_ms: float | None
    requests_completed: int
    requests_total: int
    errors: list[str]
    total_seconds: float


# Cumulative spec-decode counters vLLM exposes on /metrics (V1 engine).
# Suffixes only — the metric lines carry {engine=...,model_name=...} labels.
_SPEC_COUNTER_KEYS: dict[str, str] = {
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
}


def read_spec_decode_counters(
    base_url: str,
    *,
    timeout: float = 5.0,
    fetch: Callable[[str], str] | None = None,
) -> dict[str, float] | None:
    """Read vLLM's cumulative spec-decode counters from ``/metrics``.

    Returns ``{"drafts": ..., "draft_tokens": ..., "accepted_tokens": ...}``
    (values summed across engine/model label sets), or None when the endpoint
    is unreachable or exposes no spec-decode counters — i.e. speculative
    decoding is off, or the backend isn't vLLM. ``fetch`` injects the metrics
    text in tests.
    """
    if fetch is not None:
        try:
            text = fetch(f"{base_url}/metrics")
        except Exception:
            return None
    else:
        try:
            with urlopen(f"{base_url}/metrics", timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = _SPEC_COUNTER_KEYS.get(name)
        if key is None:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[key] = out.get(key, 0.0) + value
    return out or None


def spec_decode_stats(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict | None:
    """Acceptance stats over a bench window from two counter snapshots.

    None when there was no spec-decode activity in the window. A missing
    ``before`` snapshot (server started mid-session) degrades to lifetime
    counters rather than dropping the signal.

    ``mean_accepted_per_step`` is the accepted-draft-tokens-per-engine-step —
    the bonus (verified) token is not included, so tokens per step is
    ``1 + mean_accepted_per_step``.
    """
    if not after:
        return None
    base = before or {}
    drafts = after.get("drafts", 0.0) - base.get("drafts", 0.0)
    draft_tokens = after.get("draft_tokens", 0.0) - base.get("draft_tokens", 0.0)
    accepted = after.get("accepted_tokens", 0.0) - base.get("accepted_tokens", 0.0)
    if drafts <= 0 or draft_tokens <= 0:
        return None
    return {
        "drafts": int(drafts),
        "draft_tokens": int(draft_tokens),
        "accepted_tokens": int(accepted),
        "acceptance_rate": accepted / draft_tokens,
        "mean_accepted_per_step": accepted / drafts,
    }


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body or "{}")
    return data if isinstance(data, dict) else {}


def _p95_ms(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[idx] * 1000, 2)


def run_openai_completion_benchmark(
    base_url: str,
    *,
    model: str,
    request_count: int = 8,
    timeout_s: float = 60,
    max_tokens: int = 16,
    prompt: str = "Write one concise sentence about GPU inference tuning.",
    post_json: PostJson = _post_json,
    monotonic: Clock = time.monotonic,
) -> dict:
    """Run a sequential, bounded OpenAI-compatible completions benchmark."""
    request_count = max(1, int(request_count))
    timeout_s = max(0.1, float(timeout_s))
    max_tokens = max(1, int(max_tokens))
    started = monotonic()
    deadline = started + timeout_s
    latencies: list[float] = []
    completion_tokens = 0
    errors: list[str] = []
    url = f"{base_url.rstrip('/')}/v1/completions"

    for _ in range(request_count):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req_started = monotonic()
        try:
            data = post_json(url, payload, timeout=max(0.1, min(remaining, 30.0)))
        except Exception as exc:
            errors.append(str(exc))
            break
        elapsed = max(0.0, monotonic() - req_started)
        latencies.append(elapsed)
        usage = data.get("usage")
        if isinstance(usage, dict):
            value = usage.get("completion_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                completion_tokens += value

    total_seconds = max(0.0, monotonic() - started)
    completed = len(latencies)
    return {
        "status": "ok" if completed else "error",
        "requests_completed": completed,
        "request_count": request_count,
        "total_seconds": round(total_seconds, 3),
        "mean_latency_ms": round((sum(latencies) / completed) * 1000, 2) if completed else None,
        "p95_latency_ms": _p95_ms(latencies),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(completion_tokens / total_seconds, 2) if total_seconds > 0 else 0,
        "errors": errors,
    }


def run_openai_embedding_benchmark(
    base_url: str,
    *,
    model: str,
    request_count: int = 8,
    timeout_s: float = 60,
    text: str = "GPU inference benchmark text.",
    post_json: PostJson = _post_json,
    monotonic: Clock = time.monotonic,
) -> dict:
    """Run a sequential, bounded OpenAI-compatible embeddings benchmark."""
    request_count = max(1, int(request_count))
    timeout_s = max(0.1, float(timeout_s))
    started = monotonic()
    deadline = started + timeout_s
    latencies: list[float] = []
    errors: list[str] = []
    dimensions: int | None = None
    url = f"{base_url.rstrip('/')}/v1/embeddings"

    for _ in range(request_count):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        payload = {
            "model": model,
            "input": text,
        }
        req_started = monotonic()
        try:
            data = post_json(url, payload, timeout=max(0.1, min(remaining, 30.0)))
        except Exception as exc:
            errors.append(str(exc))
            break
        elapsed = max(0.0, monotonic() - req_started)
        latencies.append(elapsed)
        values = data.get("data")
        if isinstance(values, list) and values and isinstance(values[0], dict):
            embedding = values[0].get("embedding")
            if isinstance(embedding, list):
                dimensions = len(embedding)

    total_seconds = max(0.0, monotonic() - started)
    completed = len(latencies)
    return {
        "status": "ok" if completed else "error",
        "requests_completed": completed,
        "request_count": request_count,
        "total_seconds": round(total_seconds, 3),
        "mean_latency_ms": round((sum(latencies) / completed) * 1000, 2) if completed else None,
        "p95_latency_ms": _p95_ms(latencies),
        "embedding_dimensions": dimensions,
        "items_per_second": round(completed / total_seconds, 2) if total_seconds > 0 else 0,
        "errors": errors,
    }


def _percentile_ms(values: list[float], pct: float) -> float | None:
    """Return the ``pct``-th percentile of seconds, in milliseconds."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1)
    return round(ordered[idx] * 1000, 2)


def _stream_request(
    url: str,
    payload: dict,
    *,
    timeout_s: float,
    monotonic: Clock,
) -> TokenTimeline:
    """Issue one streaming /v1/chat/completions request and record timings.

    Parses Server-Sent Events lines (``data: <JSON>\\n\\n``) and timestamps
    each chunk that contains a token delta. Returns a TokenTimeline.
    """
    started_ms = monotonic() * 1000
    timeline = TokenTimeline(request_started_ms=started_ms)
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as response:
            for line in response:
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                payload_text = decoded[len("data:"):].strip()
                if not payload_text or payload_text == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    continue
                # vLLM 0.22 streams thinking-channel tokens as `reasoning`
                # (0.20/0.21 used `reasoning_content`, renamed in vllm#42664).
                # Thinking-default models behind a reasoning parser emit most
                # of their tokens there — count every channel, since decode
                # cadence is what the bench measures.
                token_text = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                )
                if not token_text:
                    continue
                now_ms = monotonic() * 1000
                if timeline.first_token_ms is None:
                    timeline.first_token_ms = now_ms
                else:
                    timeline.inter_token_ms.append(now_ms - (timeline.last_token_ms or now_ms))
                timeline.last_token_ms = now_ms
                timeline.token_count += 1
    except Exception as exc:
        timeline.error = str(exc)
    return timeline


def run_streaming_benchmark(
    base_url: str,
    *,
    model: str,
    concurrency: int = 1,
    duration_s: float = 60.0,
    max_tokens: int = 256,
    prompt: str = "Write one paragraph about GPU inference tuning.",
    max_latency_ms: float | None = None,
    monotonic: Clock = time.monotonic,
) -> BenchResult:
    """Drive concurrent streaming /v1/chat/completions requests for
    ``duration_s`` seconds (or until ``max_latency_ms`` is exceeded). Computes
    TTFT / TPOT / ITL percentiles and total throughput. Concurrency=1 is
    effectively sequential streaming.
    """
    concurrency = max(1, int(concurrency))
    duration_s = max(0.1, float(duration_s))
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    started = monotonic()
    deadline = started + duration_s
    timelines: list[TokenTimeline] = []
    timelines_lock = threading.Lock()
    abort = threading.Event()

    def _worker() -> None:
        while not abort.is_set() and monotonic() < deadline:
            tl = _stream_request(url, payload, timeout_s=duration_s + 30, monotonic=monotonic)
            with timelines_lock:
                timelines.append(tl)
            # Early termination on latency ceiling.
            if max_latency_ms is not None and tl.e2e_ms is not None and tl.e2e_ms > max_latency_ms:
                abort.set()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_worker) for _ in range(concurrency)]
        for _ in as_completed(futures):
            pass

    total_seconds = max(1e-6, monotonic() - started)

    ttfts = [tl.ttft_ms / 1000.0 for tl in timelines if tl.ttft_ms is not None]
    tpots = [tl.tpot_ms / 1000.0 for tl in timelines if tl.tpot_ms is not None]
    itl_all = [v / 1000.0 for tl in timelines for v in tl.inter_token_ms]
    e2es = [tl.e2e_ms / 1000.0 for tl in timelines if tl.e2e_ms is not None]
    total_tokens = sum(tl.token_count for tl in timelines)
    completed = sum(1 for tl in timelines if tl.error is None and tl.token_count > 0)
    errors = [tl.error for tl in timelines if tl.error]

    return BenchResult(
        ttft_ms_p50=_percentile_ms(ttfts, 0.5),
        ttft_ms_p99=_percentile_ms(ttfts, 0.99),
        tpot_ms_p50=_percentile_ms(tpots, 0.5),
        tpot_ms_p99=_percentile_ms(tpots, 0.99),
        itl_ms_p99=_percentile_ms(itl_all, 0.99),
        throughput_tokens_per_sec=round(total_tokens / total_seconds, 2),
        throughput_requests_per_sec=round(completed / total_seconds, 2),
        e2e_p99_ms=_percentile_ms(e2es, 0.99),
        requests_completed=completed,
        requests_total=len(timelines),
        errors=[str(e) for e in errors][:5],  # cap for log readability
        total_seconds=round(total_seconds, 3),
    )
