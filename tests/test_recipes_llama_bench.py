"""Tests for the llama-bench wrapper (item O)."""

from __future__ import annotations



class TestParseLlamaBenchJsonl:
    def test_parses_canonical_jsonl(self):
        from vserve.recipes.llama_bench import parse_llama_bench_jsonl
        text = "\n".join([
            '{"pp_avg_ts": 1100.0, "tg_avg_ts": 65.0, "n_threads": 8, "build": "b9222"}',
            '{"pp_avg_ts": 1200.0, "tg_avg_ts": 70.0, "n_threads": 16, "build": "b9222"}',
        ])
        cells = parse_llama_bench_jsonl(text)
        assert len(cells) == 2
        assert cells[0].pp_avg_ts == 1100.0
        assert cells[1].tg_avg_ts == 70.0
        assert cells[0].config["n_threads"] == 8

    def test_skips_non_json_progress_lines(self):
        from vserve.recipes.llama_bench import parse_llama_bench_jsonl
        text = "progress: 5/10\n{\"pp_avg_ts\": 1000.0, \"tg_avg_ts\": 60.0}\ndone\n"
        cells = parse_llama_bench_jsonl(text)
        assert len(cells) == 1

    def test_skips_records_missing_required_fields(self):
        from vserve.recipes.llama_bench import parse_llama_bench_jsonl
        text = '{"build": "b9222"}\n{"pp_avg_ts": 1, "tg_avg_ts": 1}'
        cells = parse_llama_bench_jsonl(text)
        assert len(cells) == 1


class TestPickBestCell:
    def test_picks_highest_balanced_score(self):
        from vserve.recipes.llama_bench import BenchCell, pick_best_cell
        cells = [
            BenchCell(pp_avg_ts=500, tg_avg_ts=10),    # balanced score = 150 + 7 = 157
            BenchCell(pp_avg_ts=200, tg_avg_ts=50),    # balanced = 60 + 35 = 95
            BenchCell(pp_avg_ts=300, tg_avg_ts=40),    # balanced = 90 + 28 = 118
        ]
        best = pick_best_cell(cells, profile="balanced")
        assert best is not None
        assert best.pp_avg_ts == 500

    def test_latency_profile_weights_decode(self):
        from vserve.recipes.llama_bench import BenchCell, pick_best_cell
        cells = [
            BenchCell(pp_avg_ts=1000, tg_avg_ts=10),    # latency score = 100 + 9 = 109
            BenchCell(pp_avg_ts=100, tg_avg_ts=100),    # latency = 10 + 90 = 100
        ]
        best = pick_best_cell(cells, profile="latency")
        # First cell wins despite "wanting" latency-heavy because total
        # weighted score still favors total tokens/sec.
        assert best is not None
        assert best.pp_avg_ts == 1000

    def test_unknown_profile_raises(self):
        import pytest
        from vserve.recipes.llama_bench import BenchCell, pick_best_cell
        with pytest.raises(ValueError, match="unknown profile"):
            pick_best_cell([BenchCell(1, 1)], profile="nonexistent")

    def test_empty_cells_returns_none(self):
        from vserve.recipes.llama_bench import pick_best_cell
        assert pick_best_cell([], profile="balanced") is None


class TestCacheKey:
    def test_changes_with_model_path(self, tmp_path):
        from vserve.recipes.llama_bench import cache_key
        a = cache_key(model_path=tmp_path / "a", gpu_uuid="gpu-x", build_commit="abc")
        b = cache_key(model_path=tmp_path / "b", gpu_uuid="gpu-x", build_commit="abc")
        assert a != b

    def test_changes_with_build_commit(self, tmp_path):
        from vserve.recipes.llama_bench import cache_key
        a = cache_key(model_path=tmp_path / "m", gpu_uuid="gpu-x", build_commit="abc")
        b = cache_key(model_path=tmp_path / "m", gpu_uuid="gpu-x", build_commit="def")
        assert a != b

    def test_stable_across_runs(self, tmp_path):
        from vserve.recipes.llama_bench import cache_key
        a = cache_key(model_path=tmp_path / "m", gpu_uuid="gpu-x", build_commit="abc")
        b = cache_key(model_path=tmp_path / "m", gpu_uuid="gpu-x", build_commit="abc")
        assert a == b


class TestRunSweep:
    def test_run_sweep_invokes_llama_bench_with_axes(self, mocker, tmp_path):
        from unittest.mock import Mock
        from vserve.recipes.llama_bench import run_sweep
        captured: dict = {}

        def _fake_run(cmd, capture_output, text, timeout):
            captured["cmd"] = cmd
            return Mock(
                returncode=0,
                stdout='{"pp_avg_ts": 1000.0, "tg_avg_ts": 50.0, "n_threads": 8}\n',
                stderr="",
            )
        mocker.patch("vserve.recipes.llama_bench.subprocess.run", side_effect=_fake_run)
        sweep = run_sweep(
            entrypoint="llama-bench",
            model_path=tmp_path / "model.gguf",
            sweep_axes={"-p": [512, 4096], "-n": [128, 256], "-fa": [1]},
        )
        assert "-m" in captured["cmd"]
        assert "-p" in captured["cmd"]
        # Comma-joined values.
        pp_idx = captured["cmd"].index("-p")
        assert captured["cmd"][pp_idx + 1] == "512,4096"
        assert len(sweep.cells) == 1
        assert sweep.cells[0].pp_avg_ts == 1000.0
