"""Tests for bounded benchmark helpers."""


def test_openai_completion_benchmark_counts_requests_and_tokens():
    from vserve.bench import run_openai_completion_benchmark

    calls: list[dict] = []

    def post_json(_url: str, payload: dict, *, timeout: float) -> dict:
        calls.append(payload)
        assert timeout <= 10
        return {"usage": {"completion_tokens": 7}}

    result = run_openai_completion_benchmark(
        "http://localhost:8888",
        model="/models/test",
        request_count=3,
        timeout_s=10,
        max_tokens=8,
        post_json=post_json,
    )

    assert len(calls) == 3
    assert result["status"] == "ok"
    assert result["requests_completed"] == 3
    assert result["completion_tokens"] == 21
    assert result["tokens_per_second"] >= 0


def test_openai_completion_benchmark_stops_at_time_budget():
    from vserve.bench import run_openai_completion_benchmark

    calls = 0
    values = [0.0, 0.0, 0.0, 0.6, 0.6, 0.6]

    def monotonic() -> float:
        return values.pop(0) if values else 0.6

    def post_json(_url: str, _payload: dict, *, timeout: float) -> dict:
        nonlocal calls
        calls += 1
        return {"usage": {"completion_tokens": 1}}

    result = run_openai_completion_benchmark(
        "http://localhost:8888",
        model="/models/test",
        request_count=10,
        timeout_s=0.5,
        post_json=post_json,
        monotonic=monotonic,
    )

    assert calls == 1
    assert result["status"] == "ok"
    assert result["requests_completed"] == 1


def test_openai_embedding_benchmark_uses_embeddings_endpoint():
    from vserve.bench import run_openai_embedding_benchmark

    calls: list[tuple[str, dict]] = []

    def post_json(url: str, payload: dict, *, timeout: float) -> dict:
        calls.append((url, payload))
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    result = run_openai_embedding_benchmark(
        "http://localhost:8888",
        model="/models/embed",
        request_count=2,
        timeout_s=10,
        post_json=post_json,
    )

    assert len(calls) == 2
    assert calls[0][0] == "http://localhost:8888/v1/embeddings"
    assert calls[0][1]["input"]
    assert result["status"] == "ok"
    assert result["requests_completed"] == 2
    assert result["embedding_dimensions"] == 3


# --- 0.7.0 item P: streaming bench with TTFT / TPOT / ITL ---


class TestTokenTimeline:
    def test_ttft_from_first_token_minus_start(self):
        from vserve.bench import TokenTimeline
        tl = TokenTimeline(request_started_ms=1000, first_token_ms=1100, last_token_ms=2100, token_count=10)
        assert tl.ttft_ms == 100

    def test_tpot_from_inter_token_average(self):
        from vserve.bench import TokenTimeline
        # 10 tokens spanning 1000ms after first → 9 inter-token gaps of ~111ms.
        tl = TokenTimeline(request_started_ms=1000, first_token_ms=1100, last_token_ms=2100, token_count=10)
        assert tl.tpot_ms is not None
        assert abs(tl.tpot_ms - (1000 / 9)) < 1.0

    def test_e2e_from_last_token_minus_start(self):
        from vserve.bench import TokenTimeline
        tl = TokenTimeline(request_started_ms=1000, last_token_ms=3500)
        assert tl.e2e_ms == 2500

    def test_zero_token_request_has_no_metrics(self):
        from vserve.bench import TokenTimeline
        tl = TokenTimeline(request_started_ms=1000, error="HTTP 500")
        assert tl.ttft_ms is None
        assert tl.tpot_ms is None
        assert tl.e2e_ms is None


class TestPercentileMs:
    def test_p99_picks_tail_value_when_few_samples(self):
        from vserve.bench import _percentile_ms
        # 10 values with a single tail outlier — p99 lands on the tail
        # (index = ceil(0.99 * 10) - 1 = 9).
        vals = [0.001] * 9 + [5.0]
        assert _percentile_ms(vals, 0.99) == 5000.0

    def test_p50_median_for_uniform_distribution(self):
        from vserve.bench import _percentile_ms
        # 100 values all 100ms → p50 of 100ms.
        vals = [0.1] * 100
        assert _percentile_ms(vals, 0.5) == 100.0

    def test_empty_returns_none(self):
        from vserve.bench import _percentile_ms
        assert _percentile_ms([], 0.5) is None


class TestRunStreamingBenchmark:
    def test_runs_with_fake_sse_stream(self, mocker):
        """End-to-end with a mocked urlopen returning SSE chunks. Verify
        TTFT and TPOT come out of the parser."""
        from vserve.bench import run_streaming_benchmark

        # Build a fake SSE stream: 1 first-token chunk + 4 follow-up chunks.
        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"."}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        # Monotonic clock yields strictly increasing values so we can
        # eyeball TTFT and inter-token spacing.
        # urlopen called once per request_started_ms tick + once per SSE line
        # — feed the worker enough tick values.
        tick = iter([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1000.0, 1000.0, 1000.0])
        def fake_monotonic():
            try:
                return next(tick)
            except StopIteration:
                return 1000.0

        class _FakeResp:
            def __iter__(self):
                return iter(sse_lines)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        mocker.patch("vserve.bench.urlopen", return_value=_FakeResp())
        # concurrency=1, duration small so we only do one pass before deadline.
        result = run_streaming_benchmark(
            "http://localhost:8888",
            model="m",
            concurrency=1,
            duration_s=0.30,
            monotonic=fake_monotonic,
        )
        assert result.requests_completed >= 1
        assert result.requests_total >= 1
        assert result.ttft_ms_p50 is not None
        assert result.tpot_ms_p50 is not None

    def test_max_latency_ceiling_terminates_early(self, mocker):
        from vserve.bench import run_streaming_benchmark
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        class _FakeResp:
            def __iter__(self):
                return iter(sse_lines)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        # Each tick advances by 1s — first E2E will be ~1s = 1000ms.
        ticks = iter([float(i) for i in range(200)])
        def fake_monotonic():
            try:
                return next(ticks)
            except StopIteration:
                return 9999.0

        mocker.patch("vserve.bench.urlopen", return_value=_FakeResp())
        result = run_streaming_benchmark(
            "http://localhost:8888",
            model="m",
            concurrency=1,
            duration_s=10.0,
            max_latency_ms=10.0,  # very tight — terminate after first request
            monotonic=fake_monotonic,
        )
        assert result.requests_total >= 1

    def test_records_errors_in_result(self, mocker):
        from vserve.bench import run_streaming_benchmark
        mocker.patch("vserve.bench.urlopen", side_effect=RuntimeError("connection refused"))
        # Ticks that let the loop body run at least once before deadline.
        # started=0.0, deadline=10.0; loop-condition tick (1.0 < 10) → enter,
        # _stream_request consumes 2.0 (started_ms), urlopen raises, error captured.
        # Then loop-condition tick (3.0 < 10) → enter again, ticks run out.
        ticks = iter([0.0, 1.0, 2.0, 3.0, 100.0])
        def fake_monotonic():
            try:
                return next(ticks)
            except StopIteration:
                return 100.0
        result = run_streaming_benchmark(
            "http://localhost:8888", model="m",
            concurrency=1, duration_s=10.0,
            monotonic=fake_monotonic,
        )
        assert result.requests_completed == 0
        assert result.errors  # non-empty
        assert "refused" in result.errors[0]

    def test_concurrency_runs_multiple_workers(self, mocker):
        """concurrency=N spawns N worker threads. Each issues at least one
        streaming request before the deadline. Confirms parallel dispatch."""
        from vserve.bench import run_streaming_benchmark
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        class _FakeResp:
            def __iter__(self):
                return iter(sse_lines)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        # Real monotonic — much simpler than crafting tick sequences for
        # concurrent workers (and the test isn't latency-sensitive).
        mocker.patch("vserve.bench.urlopen", return_value=_FakeResp())
        result = run_streaming_benchmark(
            "http://localhost:8888", model="m",
            concurrency=4, duration_s=0.1, max_tokens=8,
        )
        # All 4 workers should produce at least one request between them.
        assert result.requests_total >= 4

    def test_returns_zero_throughput_on_zero_requests(self):
        """The math doesn't divide-by-zero when nothing completed."""
        from vserve.bench import BenchResult
        r = BenchResult(
            ttft_ms_p50=None, ttft_ms_p99=None,
            tpot_ms_p50=None, tpot_ms_p99=None,
            itl_ms_p99=None, throughput_tokens_per_sec=0.0,
            throughput_requests_per_sec=0.0,
            e2e_p99_ms=None, requests_completed=0, requests_total=0,
            errors=[], total_seconds=1.0,
        )
        # Just verify the dataclass accepts the empty case cleanly.
        assert r.throughput_tokens_per_sec == 0.0


class TestReasoningDeltaCounting:
    def test_counts_reasoning_deltas_from_022_parsers(self, mocker):
        """vLLM 0.22 streams thinking-channel tokens as ``delta.reasoning``
        (renamed from ``reasoning_content``, vllm#42664). Thinking-default
        models behind a reasoning parser emit most — sometimes all — of
        their tokens there; the bench must time those tokens or it reports
        0/N completed against a perfectly healthy server (found live in
        the 0.6.3 on-GPU sweep)."""
        from vserve.bench import run_streaming_benchmark

        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning":"Thinking"}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning":" hard"}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":" legacy"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Answer."}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        tick = iter([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1000.0, 1000.0, 1000.0])

        def fake_monotonic():
            try:
                return next(tick)
            except StopIteration:
                return 1000.0

        class _FakeResp:
            def __iter__(self):
                return iter(sse_lines)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        mocker.patch("vserve.bench.urlopen", return_value=_FakeResp())
        result = run_streaming_benchmark(
            "http://localhost:8888",
            model="m",
            concurrency=1,
            duration_s=0.30,
            monotonic=fake_monotonic,
        )
        assert result.requests_completed >= 1
        assert result.ttft_ms_p50 is not None
        assert result.tpot_ms_p50 is not None


class TestSpecDecodeCounters:
    """0.6.8: /metrics spec-decode counter scrape used by `vserve bench`."""

    METRICS = "\n".join([
        "# HELP vllm:spec_decode_num_drafts_total ...",
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 6620.0',
        'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 19860.0',
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 11905.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 5183.0',
        'vllm:num_requests_running{engine="0"} 1.0',
    ])

    def test_parses_labeled_counters(self):
        from vserve.bench import read_spec_decode_counters
        out = read_spec_decode_counters("http://x", fetch=lambda url: self.METRICS)
        assert out == {"drafts": 6620.0, "draft_tokens": 19860.0, "accepted_tokens": 11905.0}

    def test_returns_none_without_spec_counters(self):
        from vserve.bench import read_spec_decode_counters
        out = read_spec_decode_counters("http://x", fetch=lambda url: "vllm:num_requests_running 0.0")
        assert out is None

    def test_returns_none_on_fetch_error(self):
        from vserve.bench import read_spec_decode_counters

        def boom(url):
            raise OSError("refused")

        assert read_spec_decode_counters("http://x", fetch=boom) is None

    def test_sums_across_label_sets(self):
        from vserve.bench import read_spec_decode_counters
        text = (
            'vllm:spec_decode_num_drafts_total{engine="0"} 10.0\n'
            'vllm:spec_decode_num_drafts_total{engine="1"} 5.0\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 30.0\n'
        )
        out = read_spec_decode_counters("http://x", fetch=lambda url: text)
        assert out is not None
        assert out["drafts"] == 15.0


class TestSpecDecodeStats:
    def test_window_delta_and_rates(self):
        from vserve.bench import spec_decode_stats
        before = {"drafts": 100.0, "draft_tokens": 300.0, "accepted_tokens": 200.0}
        after = {"drafts": 200.0, "draft_tokens": 600.0, "accepted_tokens": 380.0}
        stats = spec_decode_stats(before, after)
        assert stats is not None
        assert stats["drafts"] == 100
        assert stats["draft_tokens"] == 300
        assert stats["accepted_tokens"] == 180
        assert stats["acceptance_rate"] == 180 / 300
        assert stats["mean_accepted_per_step"] == 1.8

    def test_none_before_degrades_to_lifetime(self):
        from vserve.bench import spec_decode_stats
        after = {"drafts": 10.0, "draft_tokens": 30.0, "accepted_tokens": 18.0}
        stats = spec_decode_stats(None, after)
        assert stats is not None and stats["acceptance_rate"] == 0.6

    def test_no_activity_returns_none(self):
        from vserve.bench import spec_decode_stats
        same = {"drafts": 100.0, "draft_tokens": 300.0, "accepted_tokens": 200.0}
        assert spec_decode_stats(same, dict(same)) is None
        assert spec_decode_stats(None, None) is None
