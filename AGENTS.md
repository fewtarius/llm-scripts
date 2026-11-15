# AGENTS.md

Technical reference for llm-scripts. For methodology (checkpoints, handoffs,
ownership), see `.clio/instructions.md`. For C/C++ engine work, see
`llama.cpp/AGENTS.md` and `gufo/docs/AGENTS.md`.

## Purpose

llm-scripts is the runner layer above the inference engines. It detects
hardware, resolves a model to a file, solves a memory-aware profile, renders
that profile into engine-specific argv and environment, and runs the server.
It never reimplements inference and it never asks the user to export an
engine variable.

## Engine adapter contract

Every engine provides these functions, defined in `lib/engines/<name>.sh`:

| Function | Contract |
|---|---|
| `engine_name_<e>` | Display name |
| `engine_solve_<e> MODEL` | Fill the `PROFILE_*` fields, read `LLM_*` request knobs |
| `engine_render_<e>` | Fill `PROFILE_ARGS` with complete engine argv |
| `engine_env_<e>` | Activate the process environment and fill `PROFILE_ENV` |
| `engine_binary` (dispatcher) | Path to the server binary |
| `engine_ready_regex_<e>` | Log line that means the server is accepting requests |
| `engine_comm` (dispatcher) | Process name for the one-GPU-one-server check |

Adapters and libraries source what they use. A file that only works because
the caller already sourced something else is a latent break; `tests/lib.sh`
loads every one of them in a clean shell.

Rules:

- Build argv in arrays. Never build a command string and eval it.
- Translate generic request fields into engine spelling. The user-facing
  vocabulary is defined in `llm` usage, not per engine.
- Engine environment variables are an implementation detail. If an engine
  only has an environment switch (gufo's `LAGUNA_KV_CACHE_TYPE`), the adapter
  sets it from `PROFILE_KV_CACHE`.
- A missing binary is fatal for `serve`, but `LLM_DRY_RUN=1` (used by
  `llm profile`) renders the command anyway.

## Profile fields

`PROFILE_*` is the engine-neutral plan. `lib/profile.sh` owns the names and
emits text or JSON.

| Field | Meaning |
|---|---|
| `MODEL`, `MODEL_NAME`, `MODEL_BYTES` | Resolved first shard and size |
| `ENGINE`, `BACKEND` | Selected engine and backend |
| `CTX` | Context tokens, 0 = engine native |
| `SESSIONS` | Concurrent sessions |
| `KV_CACHE` | `f16`, `q8_0`, `q4_0` (llama uses `k/v` form) |
| `SPEC`, `DRAFT_MODEL`, `DRAFT_TOKENS` | Speculative policy |
| `SPARSE_ATTN` | `on`, `off`, `auto`, `n/a` |
| `CACHE_RAM`, `CACHE_DISK` | Cache budgets |
| `BIAS` | Solver bias |
| `ARGS`, `ENV`, `REASONS` | Rendered argv, environment, explanation |

`llm profile --json` is the machine-readable contract. Tests parse it.

## Libraries

- `lib/common.sh`: `LLM_ROOT`, `LLM_MODEL_DIR`, config loading, path
  resolution, `llm_truthy`.
- `lib/gpu.sh`: GPU_MAP, PCI/amd-smi detection, CPU ISA, thread count.
  Exports `GPU_*` and `SYS_*`. Overrides are `LLM_*`.
- `lib/env.sh`: ROCm/Vulkan/Metal paths, RADV tuning, driconf merge, ACPI
  platform profile, wrapper generation.
- `lib/models.sh`: first-shard filtering, sidecar filtering, byte totals,
  substring resolution, drafter discovery.
- `lib/gguf.sh`: metadata via `tools/gguf_meta.py`, PLE via `gguf_ple.py`,
  drafter validation via `gguf_validate.py`. The tool paths can be overridden
  with `LLM_GGUF_META`, `LLM_GGUF_PLE`, `LLM_GGUF_VALIDATE` for tests.
- `lib/process.sh`: port ownership, GPU exclusivity (by process name, not
  command line), device settle, retry.
- `lib/validate.sh`: one pass over every user-supplied setting after parsing.
  Rejects values the selected engine cannot use and lists all problems at
  once with exit 2.
- `lib/download.sh`: HuggingFace lookup, shard assembly, size and SHA-256
  verification.

## Solvers

`solver/llama.sh` is optimistic-first: it starts at the best-quality
configuration for the model archetype and detunes through a fixed priority
list until the allocation fits the GPU budget. It reads GGUF metadata and
writes `SOLVER_*` globals. The adapter maps those to the profile.

`solver/gufo.sh` is a policy solver. gufo plans its own memory, so this
solver selects family, speculative mode, draft count, context, sessions, KV
precision, and the sparse-attention switch. Bias is applied here: `context`
drops the draft, `quality` pins f16 and drops the draft, `speed` sets 64K,
f16, and sparse off.

Bias is a real option on both engines. It was a dead reference in the old
tree; `tests/engine-render.sh` now asserts it.

## Model naming rules

- A model is a single `.gguf` or the `-00001-of-` shard.
- Later shards, drafters (`*dflash*`, `*dspark*`, `mtp-*`), and projectors
  (`*mmproj*`) are sidecars, not models.
- Resolution accepts a path, exact basename, or unique substring. Ambiguous
  substrings warn and take the first.
- Drafter discovery uses the base name with shard, quant, and `-UD` stripped.
  A `-gufo.gguf` patched copy wins.

## Build

```
scripts/build.sh              # platform default (Vulkan on Linux, Metal on macOS)
scripts/build.sh --rocm       # ROCm llama.cpp
scripts/build.sh --both       # ROCm + Vulkan
scripts/build.sh --gufo       # gufo into build/gufo/
scripts/build.sh --sdk-only   # ROCm SDK into deps/ only
```

Build output is normalized: `build/llama-<backend>/bin/llama-*` and
`build/gufo/gufo`. The ROCm SDK is downloaded into `deps/` by `build.sh`.
`scripts/build.sh` owns the shared work (SDK download, environment activation,
wrapper generation) and dispatches to one script per target:
`scripts/build/llama-rocm.sh`, `llama-vulkan.sh`, `llama-metal.sh`, `gufo.sh`.
Each script is runnable on its own once `deps/` exists, takes `--clean`, and
passes ccache to cmake as the compiler launcher for C, C++ and HIP.
`scripts/build/gufo.sh` wipes its build directory when the gufo source commit
moves, because HIP objects carry no dependency files.

After a build, `bin/` wrappers are regenerated. They call
`llm _exec <engine>`, which activates the environment and execs the binary.

## Retired entry points

Nothing from the old `llama-scripts` tree is ported unless this rewrite has no
equivalent. The hardware-specific launchers (`gpt-oss-bot.sh`,
`qwen-35b-bot.sh`, `llama-run.sh`, `gufo-run.sh`, `launch-server.sh`) are
retired on purpose: each is a hand-rolled engine argv for one machine, and
every flag they set is an `llm` option. The bot case they encoded is

```bash
./llm serve --engine llama.cpp --sessions 4 --ctx 96000 --kv-cache q8_0
```

Do not reintroduce a fixed argv to keep a file-for-file mapping with the old
tree. A knob that is missing belongs in the adapter, where both engines get it.

## Tests

```bash
./tests/run.sh
./tests/mutation.sh          # opt-in: proves the suites detect regressions
./tests/lib.sh
./tests/engine-render.sh --verbose
./tests/solver-golden.sh --update
./tests/apply-ttm-tests.sh
```

`engine-render.sh` proves the no-hacks contract: a generic option maps to a
specific flag or environment variable. Add a case when adding an option.
`solver-golden.sh` cases are hermetic; update goldens only with an intentional
solver change and read the diff first.

## Constraints

- Never source a config file through `eval`; keys are parsed and assigned.
- Never accept engine-specific flags as the primary interface. If a knob is
  needed, add a generic option and translate it in the adapter.
- Keep one logging implementation (`lib/log.sh`) and one environment
  implementation (`lib/env.sh`).
- Diagnostics go to stderr; stdout is data. A logging call inside command
  substitution must never be able to corrupt the captured value.
- Validate user input before it reaches an engine or a path. `lib/validate.sh`
  is the single place that decides what a setting may be.
- Do not duplicate a value the code under test produces. `tests/` must call
  the real function, not a copy of its output; a duplicated string cannot
  fail when the original changes.
- `deps/`, `build/`, `models/`, `kv-cache/`, and `ai-assisted/` are
  gitignored. Never commit handoff files.
<!-- Documentation license: CC-BY-NC-SA-4.0. Source code: GPL-3.0-or-later. -->
