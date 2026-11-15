# llm-scripts

Run GGUF models on AMD APUs with one command and one configuration model.
Two engines share the same detection, model discovery, solver output, and
process handling:

- **gufo** - the gfx1151 HIP engine, the default
- **llama.cpp** - Vulkan by default, ROCm optional, Metal on macOS

The runner decides the profile from GGUF metadata and the machine's memory
budget. You pass intent (`--kv-cache q8_0`, `--bias context`), not engine
spelling. Engine-specific environment variables are applied by the adapter,
so nothing has to be exported in your shell.

## Quick start

```bash
git clone --recurse-submodules https://github.com/fewtarius/llm-scripts.git
cd llm-scripts
./scripts/install-deps.sh
./scripts/build.sh                 # llama.cpp, Vulkan backend
./scripts/build.sh --gufo          # gufo, HIP backend
./llm models
./llm serve --model Qwen3.6-35B-A3B-UD-Q4_K_XL
```

Point at a shared model store with `LLM_MODEL_DIR` or `~/.config/llm-scripts/config.env`.

## Commands

| Command | What it does |
|---|---|
| `llm serve MODEL` | Solve a profile and run the server in the foreground |
| `llm profile MODEL [--json]` | Show the resolved profile and exact command |
| `llm models` | List models, first shard only, drafters excluded |
| `llm pull REPO[:QUANT]` | Download from HuggingFace with shard and checksum verification |
| `llm build [--engine llama.cpp\|gufo\|all]` | Build llama.cpp and/or gufo; `--backend` selects Vulkan/ROCm/both/Metal |
| `llm doctor` | Detection, build state, driver state, model count |
| `llm tune [GB]` | Configure GTT kernel parameters |
| `llm env [--engine ...]` | Print the environment the engine would use |
| `llm metrics` | Read the engine's Prometheus endpoint (both engines) |

`serve` is the default command: `llm --model NAME` works. Every command that
acts on the system (`build`, `tune`, `serve`, `profile`) accepts
`--print-command` to show what it would run and change nothing.

## The command you no longer have to write

This was the old way to start gufo:

```bash
LAGUNA_KV_CACHE_TYPE=q8_0 \
/home/deck/llm-scripts/src/llama-gufo/build/gufo serve llm \
  --host 0.0.0.0 --port 9090 \
  --model /home/deck/llm-scripts/models/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf \
  --sessions 1 --speculative dflash2 \
  --dflash-model /home/deck/llm-scripts/models/Laguna-S-2.1-DFlash-BF16-gufo.gguf \
  -d 2
```

Now:

```bash
./llm serve --engine gufo --model Laguna-S-2.1-UD-Q4_K_XL \
  --host 0.0.0.0 --port 9090 --sessions 1 --kv-cache q8_0
```

The drafter is discovered from the model name, the draft count comes from
the solver, and the family environment variable is applied by the adapter.
`bin/gufo` and `bin/llama-server` are generated wrappers for direct use.

## Mapping

| Intent | gufo | llama.cpp |
|---|---|---|
| `--kv-cache q8_0` | `LAGUNA_KV_CACHE_TYPE` / `QWEN_KV_CACHE_TYPE` | `--cache-type-k/v q8_0` |
| `--sparse-attn auto` | `LAGUNA_SPARSE_ATTN=1` (Laguna) | not supported |
| `--draft-tokens 2` | `-d 2` | `--spec-draft-n-max 2` |
| `--sessions 2` | `--sessions 2` | `-np 2` |
| `--spec dflash2` | `--speculative dflash2 --dflash-model ...` | `--spec-type draft-dflash -md ...` |
| `--cache-ram off` | `--cache-ram-bytes off` | `--cache-ram 0` |
| `--backend rocm` | HIP library path | `build/llama-rocm` |

## Configuration

Precedence, lowest to highest:

1. Built-in defaults
2. Hardware detection and the solver
3. `profiles/<model>.conf` written by `llm profile --save`
4. `~/.config/llm-scripts/config.env` (`LLM_*` keys)
5. Environment variables (`LLM_*`)
6. Command-line flags

See `config/llm.conf.example`. Nothing in the config file is eval'd; values
are data.

`LLM_MODEL` may come from the environment or the config file, which is how
the shipped systemd unit names the model without a command line. Every
setting is validated once, after parsing: a typo in a flag, an environment
variable, or the config file is reported with the allowed values instead of
reaching an engine. All problems are listed in one pass.

```bash
$ llm serve --model M --sessions 0 --port 70000
[ERROR] sessions must be at least 1, got 0
[ERROR] port must be at most 65535, got 70000
[ERROR] 2 setting(s) rejected
```

Diagnostics go to stderr; stdout carries data only (resolved paths, JSON,
rendered commands). That separation is what keeps `llm profile --json`
valid when a warning fires.

## GPU memory

The solver budgets against the GPU-visible pool: VRAM carveout plus GTT. On
Strix Halo the BIOS carveout is typically small and GTT supplies the rest.
Raise GTT if the pool is too small:

```bash
sudo ./scripts/apply-ttm-kernel-params.sh 104
```

`llm doctor` prints the pool it sees.

### Leaving the OS some RAM

GTT is address space, not a reservation. You pay for it in the pages the GPU
actually touches, and on an APU those pages come out of the same RAM the
operating system needs. A quant that fills the pool at native context can leave
the kernel and the desktop only a few GiB.

gufo sizes its snapshot cache automatically at half the host RAM available when
the model loads, capped at 32 GiB. That reading happens before the K/V pages
have all faulted in, so on a nearly-full box the automatic figure is optimistic.
Cap it when the weights leave little behind:

```bash
./llm serve --engine gufo --model Laguna-S-2.1-UD-Q5_K_XL --cache-ram 2048
```

`--cache-ram` takes MiB, or `off` to disable the cache. An explicit value keeps
its own reserve; the automatic one does not.

Watch it across a long agent session, where context grows and never shrinks:

```bash
watch -n5 grep\ MemAvailable /proc/meminfo
```

The kernel's page recycling pool is a separate thing and stays small (2 GiB,
override with `LLM_TTM_POOL_GB`). Pool pages are released from GPU buffers but
not handed back to the allocator, so a pool sized to the GTT window lets the GPU
hold the entire budget in pages the host cannot see as free.

## Layout

```
llm                  dispatcher (serve, profile, models, pull, build, doctor, tune, env, metrics)
lib/
  common.sh          paths, config precedence
  log.sh             logging
  gpu.sh             GPU/APU and CPU ISA detection
  env.sh             the only runtime environment implementation
  models.sh          discovery, sharding, drafter resolution
  gguf.sh            GGUF metadata, PLE size, drafter validation
  process.sh         port ownership, GPU exclusivity, settle, retry
  validate.sh        rejects settings an engine cannot use
  profile.sh         engine-neutral profile and emission
  download.sh        HuggingFace download
  engines/llama.sh   llama.cpp adapter
  engines/gufo.sh    gufo adapter
solver/llama.sh      optimistic-first memory solver
solver/gufo.sh       gufo policy solver
scripts/
  build.sh           shared build front end (SDK download, dispatch, wrappers)
  build/             one script per target: llama-rocm, llama-vulkan, llama-metal, gufo
  install-deps.sh    system dependencies, including ccache
  apply-ttm-kernel-params.sh, metrics.sh, benchmark.sh
tools/               gguf_meta.py, gguf_ple.py, gguf_validate.py
tests/               engine contract, solver goldens, bootloader mutation
llama.cpp/ gufo/     engine submodules
build/               build output (gitignored)
models/              model store (gitignored)
```

## Testing

```bash
./tests/run.sh          # every suite; also fails if a shell diagnostic leaks
./tests/mutation.sh     # break the code on purpose, confirm a suite catches it
```

The solver goldens are hermetic: GGUF metadata and host probes are stubbed,
so results are identical on every machine. `tests/engine-render.sh` asserts
that generic options map to the correct per-engine flags and environment.
`tests/lib.sh` loads every library on its own, so a missing dependency cannot
hide behind whatever the caller happened to source first.

`tests/mutation.sh` is the answer to "how do we know the tests work": it
breaks one piece of production code at a time and fails if the suite that
should catch it still passes. It edits tracked files, so it refuses to run
against a dirty tree.

## License

Source code: GPL-3.0-or-later. Documentation: CC-BY-NC-SA-4.0.
llama.cpp and gufo are licensed under their own terms.
<!-- Documentation license: CC-BY-NC-SA-4.0. Source code: GPL-3.0-or-later. -->
