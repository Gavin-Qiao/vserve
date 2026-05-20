# vserve architecture audit — module boundaries

Date: 2026-05-20. Scope: 547-test 0.6.x codebase. Method: read-only static analysis (Read/grep/wc + git log).

## Findings

1. **cli.py is a god-module (6,136 lines, 22 Typer commands + 81 private helpers).** Seven units exceed 200 lines: `doctor` 513 (`cli.py:5423-5934`), `status` 331 (`cli.py:4778-5108`), `_launch_backend` 283 (`cli.py:2700-2982`), `fan` 280 (`cli.py:4355-4634`), `tune` 250 (`cli.py:1591-1840`), `init` 243 (`cli.py:5169-5411`), `run` 216 (`cli.py:4044-4258`); plus `_scripted_config` 219 and `_custom_config_{vllm,llamacpp}` 172+179. **Severity: critical.**

2. **Layering violation — cli.py branches on `backend.name` 29 times** (`cli.py:298,300,1719,1775,1957,2387,2426,2466,2603,2613,2622,2627,2656,2837,2848,2993,3053,3055,3066,3290,4822,5007,5020,5081,5083,5085,5092,5854`). `_custom_config` (`cli.py:2993-3002`) dispatches to `_custom_config_vllm` vs `_custom_config_llamacpp` — the picker UX is one giant elif ladder that belongs behind the Backend protocol (`backends/protocol.py:1-133`). **Severity: high.**

3. **Reverse layering — backends import from cli's neighbors at call time.** `backends/vllm.py:391-861` and `backends/llamacpp.py:66-1475` use 18 inline `from vserve.config|serve|tools|recipes.* import …` calls inside methods. The "low-level" backend layer pulls in serve.py systemd glue (`backends/vllm.py:834-842`), config singletons, and recipes. **Severity: high.**

4. **config.py knows specific backends.** `unit_content_matches_backend` (`config.py:113-131`), `BackendConfig` defaults seeded for `"vllm"`/`"llamacpp"` (`config.py:327-340`), and `VserveConfig.vllm_*`/`llamacpp_*` fields (`config.py:213-232`) hard-code two backends the would-be neutral config layer should be agnostic about. **Severity: medium.**

5. **104 inline `from vserve.…` imports inside cli.py.** Re-importing `cfg`, `_BACKENDS`, `gpu`, `runtime` inside command bodies is classic patch-sediment — each new feature added its own import block. Hotspots: `config` 30×, `backends` 19×, `gpu` 14×. **Severity: medium.**

6. **Profile / architecture ownership is split.** `_resolve_profile_path` (`cli.py:3274-3315`) is in cli.py, YAML I/O in config.py (`config.py:600-625`). Architecture detection lives in `models.py:107-329` *and* `backends/vllm.py:129-186` (`_architecture_forces_triton_attn`, `_forced_attention_backend` keep a parallel arch→attention map). **Severity: medium.**

7. **UX helpers sit next to commands.** `_pick`, `_pick_many`, `_pick_variants`, `_has_gum`, `_restore_terminal`, `_show_model_detail`, `_refresh_banner` (`cli.py:650-1085`) plus the perf-cache picker overlay `_print_measured_cells_block` (`cli.py:2249-2326`) have no Typer dependency. **Severity: medium.**

8. **Patch markers visible.** `# Pre-0.5.8 llama.cpp cache shape` (`cli.py:2156`), `# Prefer ctx_per_slot (vserve 0.5.9+)` (`cli.py:5021`), and `# Item Q/R/AA/C` (`models.py:76,87`; `backends/vllm.py:153,712`; `backends/llamacpp.py:1102,1282`; `recipes/spec_decode.py:14`) symptomatic of release-by-release accretion. **Severity: low.**

## Recommended refactors

1. **Extract `src/vserve/cli_picker.py`**: move `_pick`, `_pick_many`, `_pick_variants`, `_has_gum`, `_restore_terminal`, `_parse_multi_select`, `_show_model_detail`, `_refresh_banner` (`cli.py:780-1085, 739-779, 650-659`) — pure UI utilities, zero backend coupling. Shrinks cli.py by ~400 lines.

2. **Extract `src/vserve/cli_config_builder.py`**: move `_custom_config`, `_custom_config_vllm`, `_custom_config_llamacpp`, `_scripted_config`, `_choose_vllm_scripted_defaults`, `_choose_llamacpp_scripted_defaults`, `_llamacpp_*_interactive_*`, `_vllm_limits_entry`, `_vllm_limit_dtype_order`, `_vllm_kv_label`, `_print_measured_cells_block`, `_print_llamacpp_moe_block` (`cli.py:2249-2363, 2983-4042, 3360-3692`) — drops ~1700 lines and decouples `backend.name` branching from `run`.

3. **Extract `src/vserve/cli_doctor.py` and `src/vserve/cli_init.py`**: `doctor` (`cli.py:5423-5934`, 513 lines) and `init` (`cli.py:5169-5411`, 243 lines) are self-contained one-shot reports that share no state with the rest of cli.py.

4. **Push backend branching behind the protocol.** Add `Backend.build_id() -> str`, `Backend.scripted_config_defaults(...)`, `Backend.custom_config_prompt(...)` to `backends/protocol.py:1-133` and let each backend own its own `cli.py:2613-2656, 2993-3066` branches. This eliminates ~25 of the 29 `backend.name ==` checks.

5. **Invert the backend↔config/serve dependency.** Move `serve.start_vllm/stop_vllm/is_vllm_running` (`serve.py:188-227`) into `backends/vllm.py` (they only exist for one caller, `backends/vllm.py:834-842`). Drop the `vserve.config.cfg` inline-imports inside backend methods by passing a `RuntimeContext` once at construction.
