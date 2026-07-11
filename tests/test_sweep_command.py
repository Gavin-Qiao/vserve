"""Orchestration tests for `vserve tune --sweep spec` (_run_spec_sweep).

The pure ranking core is covered by test_sweep.py; these verify the live-cycling
orchestration with the service/bench/restore seams mocked — especially that the
pre-sweep profile is ALWAYS restored (the "never leave the box down" guarantee).
"""

from __future__ import annotations

import json
from pathlib import Path


def _mtp_model(tmp_path: Path):
    from vserve.models import detect_model

    d = tmp_path / "models" / "nvidia" / "Qwen3.6-Test-NVFP4"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1,
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4, "num_attention_heads": 32,
            "hidden_size": 4096, "num_hidden_layers": 32,
        },
    }))
    (d / "model.safetensors").write_bytes(b"\0" * 64)
    return detect_model(d)


def _wire_sweep(mocker, tmp_path, *, bench_tps, boot_fail_variant=None):
    """Patch every GPU/service seam _run_spec_sweep touches. Returns
    (backend, launch_mock, restore_target)."""
    from vserve.bench import BenchResult

    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.build_config.side_effect = lambda m, ch: {"model": str(m.path), "spec": ch.get("spec")}

    restore_target = tmp_path / "configs" / "prior.yaml"
    restore_target.parent.mkdir(parents=True, exist_ok=True)
    restore_target.write_text("model: x\n")

    active = mocker.Mock()
    active.exists.return_value = True
    active.resolve.return_value = restore_target
    mocker.patch("vserve.config.active_yaml_path", return_value=active)
    mocker.patch("vserve.config.profile_path", side_effect=lambda p, n, prof: tmp_path / f"{prof}.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._resolve_probe_model_name", return_value="served")

    launch = mocker.patch("vserve.cli._launch_backend")
    if boot_fail_variant is not None:
        def _maybe_fail(bk, path, label, **kw):
            if boot_fail_variant in str(label):
                raise RuntimeError("boot failed")
        launch.side_effect = _maybe_fail

    def _fake_bench(base_url, *, model, concurrency, duration_s, max_tokens):
        return BenchResult(
            ttft_ms_p50=50.0, ttft_ms_p99=60.0, tpot_ms_p50=5.0, tpot_ms_p99=6.0,
            itl_ms_p99=7.0, throughput_tokens_per_sec=bench_tps.pop(0),
            throughput_requests_per_sec=1.0, e2e_p99_ms=100.0,
            requests_completed=5, requests_total=5, errors=[], total_seconds=float(duration_s),
        )

    # _run_spec_sweep does a local `from vserve.bench import …`, so patch the
    # source module (the local import fetches the mock at call time).
    mocker.patch("vserve.bench.run_streaming_benchmark", side_effect=_fake_bench)
    mocker.patch("vserve.bench.read_spec_decode_counters", return_value=None)
    return backend, launch, restore_target


class TestRunSpecSweep:
    def test_benches_all_variants_and_restores(self, mocker, tmp_path):
        from vserve.cli import _run_spec_sweep

        m = _mtp_model(tmp_path)
        # 4 variants (off, ngram, mtp-k1..k3 gated by n_predict=1 → k1,k2,k3) x 1 concurrency.
        # off, ngram, mtp-k1, mtp-k2, mtp-k3 = 5 variants, 1 concurrency each.
        tps = [220.0, 240.0, 135.0, 107.0, 105.0]
        backend, launch, restore = _wire_sweep(mocker, tmp_path, bench_tps=list(tps))
        mocker.patch("vserve.cli._vllm_mtp_gate_reason", return_value=None)

        rc = _run_spec_sweep(
            m, backend, concurrencies=[1], duration_s=15,
            base_choices={"context": 8192, "kv_dtype": "fp8", "slots": 8, "gpu_mem_util": 0.9, "port": 8888},
            runtime_info=None, port=8888,
        )
        assert rc == 0
        # 5 variant boots + 1 restore.
        assert launch.call_count == 6
        # Last call is the restore to the captured prior profile.
        assert launch.call_args_list[-1].args[1] == restore

    def test_restores_even_when_a_variant_boot_fails(self, mocker, tmp_path):
        from vserve.cli import _run_spec_sweep

        m = _mtp_model(tmp_path)
        # mtp-k2 boot fails; others bench fine. Provide enough tps for the ok ones.
        tps = [220.0, 240.0, 135.0, 105.0]  # off, ngram, k1, k3 (k2 fails before bench)
        backend, launch, restore = _wire_sweep(mocker, tmp_path, bench_tps=list(tps), boot_fail_variant="mtp-k2")
        mocker.patch("vserve.cli._vllm_mtp_gate_reason", return_value=None)

        rc = _run_spec_sweep(
            m, backend, concurrencies=[1], duration_s=15,
            base_choices={"context": 8192, "kv_dtype": "fp8", "slots": 8, "gpu_mem_util": 0.9, "port": 8888},
            runtime_info=None, port=8888,
        )
        assert rc == 0
        # Restore still happened (last launch call → prior profile).
        assert launch.call_args_list[-1].args[1] == restore

    def test_rejects_non_vllm_backend(self, mocker, tmp_path):
        from vserve.cli import _run_spec_sweep

        m = _mtp_model(tmp_path)
        backend = mocker.Mock()
        backend.name = "llamacpp"
        rc = _run_spec_sweep(
            m, backend, concurrencies=[1], duration_s=15,
            base_choices={}, runtime_info=None, port=8888,
        )
        assert rc == 1

    def test_skips_mtp_when_runtime_gate_blocks(self, mocker, tmp_path):
        from vserve.cli import _run_spec_sweep

        m = _mtp_model(tmp_path)
        # Only off + ngram benched (MTP gated out) → 2 boots + 1 restore.
        tps = [220.0, 240.0]
        backend, launch, restore = _wire_sweep(mocker, tmp_path, bench_tps=list(tps))
        mocker.patch("vserve.cli._vllm_mtp_gate_reason", return_value="runtime too old")

        rc = _run_spec_sweep(
            m, backend, concurrencies=[1], duration_s=15,
            base_choices={"context": 8192, "kv_dtype": "fp8", "slots": 8, "gpu_mem_util": 0.9, "port": 8888},
            runtime_info=None, port=8888,
        )
        assert rc == 0
        assert launch.call_count == 3  # off, ngram, restore
