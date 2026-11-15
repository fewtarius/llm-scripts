# Solver Algorithm and Tuning

The optimistic-first solver in `solver/llama.sh` picks (ctx, KV cache type,
MoE strategy, batch, ubatch) for each (model, hardware) pair from a scored
combination space. User overrides applied last.

## Decision order

1. **Built-in defaults** (solver starts here - see "Optimistic defaults" below)
2. **System detection** (GPU memory via `lib/gpu.sh`)
3. **Solver output** (chosen from a scored combination matrix)
4. **User overrides** via env vars and CLI flags

## Bias flags (`--bias`)

The solver is optimistic-first: it assumes you want the largest context that
fits and starts from the best-quality configuration for the model archetype.
The `--bias` flag (or `LLM_BIAS` env var) steers the search toward one
dimension instead. The default is `context`:

| Bias | Effect | KV cache | Context | MTP draft |
|------|--------|----------|---------|-----------|
| `context` (default) | Maximize context: drop MTP draft to free ~1.2 GiB, one checkpoint | Archetype start | Largest that fits | off |
| `mtp` | Balanced: keep the drafter for decode speed | Archetype start | Native max | on |
| `quality` | Maximum KV fidelity: f16 KV, no draft | f16/f16 (detunes only if it cannot fit) | Large | off |
| `speed` | Fastest decode: smallest ctx, f16 KV, no draft | f16/f16 | 65536 | off |

Bias affects both the phase-1 candidate scoring (boosts penalized/preferred
combos in the ranking) and the phase-2 detune priority (e.g., `context` bias
keeps MTP off in phase 2 as well). It is applied on top of memory constraints —
if f16 KV doesn't fit even with bias=quality, the solver falls back to q8_0.

The KV column is the archetype's optimistic start, not a fixed value: the
detune walk lowers it when the allocation does not fit. Only `quality` and
`speed` pin a precision, and only `quality` resists detuning.

**Usage:**

```bash
./llm serve --model Qwen3.6-35B-A3B-UD-Q4_K_XL --bias mtp
./llm serve --model Qwen3.6-35B-A3B-UD-Q4_K_XL --bias speed
LLM_BIAS=quality ./llm serve --model Qwen3.6-35B-A3B-UD-Q4_K_XL
```

## Optimistic defaults (per archetype)

`llama-bench` sweeps across n_batch in {2048, 4096, 8192} and n_ubatch in
{256, 512, 1024, 2048, 4096} on Ayaneo Flip (7840U) and Nimo Axis (Strix
Halo) for 17 model/hardware combinations informed the per-archetype defaults
below. Decode (tg) is memory-bandwidth bound and varies <2% across the
entire (batch, ubatch) range - prefill (pp) is what tuning moves.

The table below applies on every machine. `lib/gpu.sh` detects a hardware
tier (`halo`, `standard`, `handheld`) and exports it for diagnostics, but the
solver does not read it: memory pressure is what separates a handheld from a
desktop, and that is expressed by the budget, not by a different table. The
handheld numbers in the benchmark section are observations, not a second set
of defaults.

### Per-archetype defaults

The profile name reported for a model comes from the same ordered checks:
`qwen4exp`, `ssm`, `mla`, `moe-small`/`moe-large` (split at 50 GB), `dense`.
SSM is decided before MLA because a recurrent model also reports
`head_count_kv=1`, which the MLA test matches.

| Archetype | Size | ubatch | batch | Why |
|-----------|------|--------|-------|-----|
| Dense | any | 1024 | 4096 | 5-8% pp loss at ub=2048 (qwen35 27B Q8_0: 127 vs 121 pp) |
| MoE small/medium | <60 GB | 2048 | 4096 | Benchmarks on Strix Halo show 2048/4096 (ub/b) is optimal across small (512-token) and large (15k-token) prefill prompts. The old 1024/2048 default was 8.7% slower at pp15000 due to insufficient batch size for large prefill contexts. |
| MoE large | 60-100 GB | 2048 | 8192 | 8192 batch + 2048 ubatch is the empirical sweet spot (Laguna 118B, qwen35moe 122B, gpt-oss 120B) |
| MoE huge | >=100 GB | 4096 | 8192 | 4096 ubatch needed to amortize the heavy per-token work |
| SSM / hybrid | any | 1024 | 4096 | Linear-attention layers don't benefit from larger batches |
| qwen4exp (Qwen3.8-Flash-Next) | any | 2048 | 8192 | PLE + hybrid attention. 8192/2048 is 0.4-0.9% faster than 4096/2048 on Strix Halo for both small and large prefill prompts. NOTE: despite 3 of 4 layers using recurrent/linear attention (is_recr, full_attention_interval=4), **all 48 layers allocate a full-context KV cache** — is_recr selects the attention kernel, not KV allocation (llama_hparams::has_kv defaults true; qwen4exp sets no n_layer_kv_from_start). KV budget must use the full 48-layer count, NOT ceil(48/4)=12. |
| MLA (DeepSeek-V2/V3/V4, GLM-4.x) | any | 4096 | 8192 | Latent attention has ~1/N KV cache vs non-MLA, so ubatch can carry more; Lightning Indexer fused op in llama.cpp accelerates the indexer/attention split when subgroup_size_control gates pass |

## Context size candidates

The solver reads the model's `context_length` from GGUF metadata (via
`tools/gguf_meta.py`) and uses it directly as the starting `SOLVER_CTX_SIZE`.
The model's native context_length is capped at 1M (llama.cpp hard limit)
and floored at `MIN_CTX` (65536).

Phase-1 candidate context sizes are generated dynamically by
`_opt_build_ctx_candidates(max_ctx)`: it starts at the model's native
context_length and steps down by 1024 (ctx > 128k) or 512 (ctx <= 128k)
until `MIN_CTX`, capped at 200 candidates. This replaced the older static
grid (`{1048576, 524288, 262144, 196608, 131072, 98304, 65536}`) which
missed models with non-standard context lengths (e.g. 200k, 256k, 32k,
100k). The step-down continues until the largest fitting context is found.

q4_0 KV cache is **not** a phase-1 candidate - it is reserved for phase 2
as an absolute last resort. Phase 1 enumerates ctx from largest to smallest
with KV types `{f16, q8_0}` in score order.

If no phase-1 combo fits, phase 2 starts from the most memory-conservative
fallback (`MIN_CTX` + `q4_0/q4_0` with CPU strategy for MoE models) and
applies detune steps in priority order (q8 → q4 → ubatch → ctx → draft → NGL).

## Combination scoring

For each (strategy, ctx, KV, draft, batch, ubatch) tuple, the solver computes
a score. Higher scores are tried first. The first combo that fits the GPU
budget AND system memory wins.

```
score = strategy_score * 1000      # gpu=300, cpu=250
      + ctx_score * 10             # 1048576=110, 524288=105, 262144=100, 196608=95,
                                   # 131072=90, 98304=85, 65536=80, other=ctx/1000
      + kv_score                   # f16=30, q8_0=20
      + draft_score                # enabled=5, qwen4exp=125
      + batchub_score              # opt=2, partial-match=1, other=0
```

The ctx_score is monotonically increasing with context size, so the
solver prefers the largest context that fits. The kv_score prefers f16
over q8_0. q4_0 is NOT in phase 1 - it is a phase-2 detune step only,
applied when no f16 or q8_0 configuration fits at any context size.

Phase 1 enumerates (ctx from model max down to 128k, kv in {f16, q8_0})
in score order. The (batch, ubatch) component is a small tiebreaker (max
+2) so it does not override the ctx/kv preference.

The (batch, ubatch) component is a small tiebreaker (max +2) so it does
not override the ctx/kv preference. The real decision driver is the GPU
memory check + system memory check.

## Load mode and UMA accounting

### Load mode (`_opt_decide_load_mode`)

| Condition | Load mode |
|---|---|
| Model fits in RAM (≤ 85% of MemAvailable after 4 GiB OS reserve) | `none` (eager `pread` into a contiguous host buffer) |
| Model too large for RAM | `auto` (mmap; page cache acts as L4 buffer) |

**qwen4exp** uses `none` based on the **GPU-resident** footprint (model
minus the ~40% PLE, which is offloaded to CPU via `-ot per_layer_token_embd.weight=CPU`),
since only that fraction is eagerly read from disk.

`dio` (direct I/O, `O_DIRECT`) is **not used for large models (>80 GB) on UMA**
because `O_DIRECT` reserves the full virtual address range upfront, which on
kernels with `overcommit_memory=0` exceeds the commit limit — qwen3.8 at
103 GiB reserves ~128 GiB VMA against a ~128 GiB commit limit, OOM-killing
the process during `load_model` (total-vm ≈ 128 GiB, anon-rss ≈ 3.5 MiB —
a pure VMA overcommit kill, no llama.cpp error). `none` commits real RSS via
`pread`, no speculative reservation.

For **MoE models that fit within the commit limit** (<80 GiB, e.g.
Qwen3.6-35B at 21 GiB), `dio` is used instead of `auto` (mmap) because mmap
triggers `radv/amdgpu: Not enough memory for command submission` DeviceLost
at large contexts (>=128k) — lazy page faults during graph compute overwhelm
the GEM allocator on UMA APUs. `dio` reads the model into a staging buffer
upfront, making GPU memory resident predictable.

The `residency` (cpu-moe) strategy still uses `dio` as its offload transport —
intentional, unaffected.

### UMA memory is not two pools

On integrated GPUs (Strix Halo: `vram_total ≈ 0.5 GiB` carveout, `gtt_total ≈
112 GiB` = system RAM), **VRAM+GTT is the same physical memory as system RAM**.
The solver detects this (`vram > 0 && gtt > vram`) and adjusts the system
memory check:

```
sys_budget = MemAvailable - mem_needed   (UMA, gpu strategy only)
```

This subtracts the UMA-resident GPU allocation (model weights + KV + FA
scratch) from the system budget so the two checks don't double-count the same
RAM. Without this, the solver green-lit oversubscribed configs — e.g. qwen3.8
ctx=262144 f16/f16 reported `sys_ok=1` (49 GiB PLE vs 125 GiB "available"),
but at runtime the 91 GiB GPU allocation + 41 GiB PLE = 132 GiB against 128 GiB,
firing the FA scratch gate ("exceeds host RAM; using slow path").

This is applied in both the phase-1 scoring loop and the phase-2 fit check.

## Detune steps (phase 2)

If no combination fits in phase 1, phase 2 starts from the most
memory-conservative fallback (MIN_CTX + q4_0/q4_0) and applies detune
steps in priority order. The preference is: **q4_0 KV cache only when no
choice, ctx below 128k only when no choice, reducing layers from GPU only
when no choice.**

The phase-2 detune order (first fit wins, each step applies at most once):

1. Reduce KV cache to q8_0 (only downgrades from f16 - skipped if already q8 or q4)
2. Reduce KV cache to q4_0 (only downgrades from f16/q8 - skipped if already q4)
3. Reduce ubatch by half (clamped at 512)
4. Reduce ctx (cascading values 1048576 -> 524288 -> 262144 -> 196608 -> 131072 -> 98304 -> 65536)
5. Drop speculative draft model
6. Reduce NGL by 10% per step (layers moved from GPU to CPU - absolute last resort)

## Cache-RAM adequacy detune

Phase 1/phase 2 only guarantee the *model + KV* fits the GPU budget; they do
not enforce a host-prompt-cache floor. On a GPU-budget-bound fit (common on
a 25-32 GiB box) the leftover can land right at the floor — too small to
cache the prompt. This is load-bearing for SSM/hybrid models (and
`n_parallel>1`), where llama-server's KV shifting is disabled and the host
prompt cache is the critical fallback for cache recovery on prompt
discontinuities (`n_past=0` and full prompt re-processing otherwise).

`_opt_min_cache_ram_mib` returns the model-appropriate minimum
(`--cache-ram`):

| Model class | Budget | Minimum cache-ram |
|-------------|--------|-------------------|
| MoE, >=50 GiB | any | 4096 MiB |
| MoE (low-VRAM, <32 GiB GPU) | <32 GiB | 1024 MiB (~1 GB, just enough to cache the prompt while offloading the fewest layers) |
| MoE (ample) | >=32 GiB | 3072 MiB |
| Dense / SSM / hybrid (low-VRAM) | <32 GiB | 1024 MiB |
| Dense / SSM / hybrid (ample) | >=32 GiB | 2048 MiB |

When the cache is actually used (`n_parallel>1` or `is_ssm`), a
**post-fit cache-RAM adequacy detune** runs after phase 2: it re-evaluates
`cache-ram` from the live `SOLVER_*` globals and, if below the minimum,
detunes in a cache-friendly priority order — **offload compute layers first**
(on UMA this frees GPU VRAM directly, which raises `cache-ram`, while
preserving the ctx window, KV precision and MTP draft), then shrink ctx,
then drop the draft. The fit is re-checked each step so the profile never
becomes unloadable. If the minimum cannot be reached (the model is larger
than the GPU even fully offloaded) the best-fit config is kept with a
sub-minimum cache — graceful fallback, never a crash.
`OVERRIDE_CACHE_RAM` bypasses the whole step.

> Example (Qwen3.6-35B-A3B Q4_K_XL on a 25 GiB GPU): phase 1/2 choose
> ctx=131072, KV=q8_0/q8_0, NGL=41 (full), draft=off, which leaves ~310 MiB
> of GPU headroom -> `cache-ram` 279 MiB. The adequacy detune offloads 2
> layers (NGL 41->39), lifting `cache-ram` to ~1.2 GB (>= the 1024 MiB
> minimum) while keeping ctx/KV/draft/batch/ubatch intact.

## Checkpoint memory accounting

The `--ctx-checkpoints N` feature stores KV snapshots in GPU VRAM. During
checkpoint creation (`create_checkpoint` → `state_seq_get_data` in
llama-context.cpp), llama.cpp copies the active KV window into a temporary
buffer before writing to the checkpoint ring buffer. This creates a transient
peak of ~`ckpt_min_step × kv_per_token` bytes.

The solver accounts for checkpoint storage in `_opt_gpu_memory` (passed as
the `ckpt_count` parameter) and includes it in the GPU fit check. The transient
copy buffer is NOT in the fit check (it's brief, ~50ms per checkpoint), but
after the fit is chosen, the post-fit adaptive reduction checks whether
`mem_with_ckpt + copy_peak > budget` and reduces the checkpoint count
(4 → 3 → 2 → 1) until it fits.

This was added after the initial checkpoint accounting caused
`radv/amdgpu: Not enough memory for command submission` (DeviceLost) crashes
at runtime — the solver chose configs that fit without checkpoint storage
but OOM'd when the checkpoint copy buffer was allocated during the second
inference task.

## Edge cases and known issues

### Hybrid SSM/MoE (qwen3next)

`qwen3next` is detected as `is_ssm=true` in `_scan_gguf_arch` (the GGUF has
`qwen3next.ssm.*` keys plus `qwen3next.expert_count > 0`). The SSM branch
takes priority over MoE in the optimistic defaults, picking ubatch=1024
which matches the empirical peak (qwen3next 80B Q8_0: 762 pp at ub=1024 vs
537 at ub=4096).

### Models with misleading filenames

Some MoE models (Laguna-S-2.1, Qwen3-Coder-Next, GLM-4.7-Flash) don't match
the filename MoE regex `moe|a3b|a8b|flash|expert|gpt-oss`. The architecture scan in
`lib/engines/llama.sh` reads GGUF metadata and checks for `expert_count` to
set `is_moe=true`. This
catches all known MoE architectures.

### 8192 batch for very large MoE

The data shows 8192 batch is marginally faster (0.4-0.9% pp) than 4096
for very large MoE/qwen4exp models on Strix Halo (Qwen3.8-Flash-Next
qwen4exp: 602.6 pp at 8192/2048 vs 597.2 at 4096/2048). For 60-100 GB MoE,
the solver already defaults to 8192/2048. The phase-1 candidates list
always includes 8192 as a fallback so the memory check has it on the table.

### Decode (tg) is bandwidth-bound

Across all benchmark sweeps, tg varies <2% across the entire
(batch, ubatch) range. Tuning (batch, ubatch) moves pp but not tg. So
`vram-bandwidth-limited` decode speed is a hardware characteristic, not
something the solver can tune.

## Benchmark data (llama-bench)

| Hardware | Model | Size | Peak (b/u) | Peak pp (t/s) | Notes |
|----------|-------|-----:|-----------:|--------------:|-------|
| Halo | Qwen3.6-35B-A3B Q8_K_XL (qwen35moe) | 36 GB | 4096/2048 | 1520.0 | 8.7% better at pp15000 vs 2048/1024 halo default |
| Halo | Qwen3.8-Flash-Next Q4_K_XL (qwen4exp) | 104 GB | 8192/2048 | 602.6 | 0.9% better at pp15000 vs 4096/2048 |
| Halo | Laguna-S-2.1 Q5_K_XL (MoE) | 82 GB | 8192/2048 | 619.3 | 8192/4096 is 0.07% better at pp512 but 1.2% worse at pp15000 |
| Halo | gpt-oss-120b Q8_K_XL (MoE) | 60 GB | 8192/2048 | — | Profile confirms 8192/2048 |
| Halo | GLM-4.7-Flash Q8_K_XL (MLA) | 33 GB | 8192/4096 | — | MLA branch: 4096/8192 |
| Halo | DeepSeek-V4-Flash IQ3_XXS (MLA) | 101 GB | 8192/4096 | — | MLA branch: 4096/8192 |
| 7840U | Qwen3.6-35B-A3B Q8_0 (qwen35moe) | 36 GB | 4096/2048 | — | Same default as Halo; the 780M confirms it |

Benchmarked with llama-bench on Nimo Axis N161 (Strix Halo, gfx1151,
Radeon 8060S, 124 GB RAM, RADV unified heap enabled). Vulkan backend,
llama-bench with -ngl 99, --flash-attn auto, -t 16, pp512 and pp15000
with -n 128. The Qwen3.6-35B-A3B pp15000 sweep was decisive: 8192 batch
is 5% slower than 4096 (dispatch overhead on 40 CUs with 35B params),
while 2048 batch is 8.7% slower. The 4096/2048 default is optimal across
both prompt sizes.

The old benchmark table (from earlier sessions, different llama.cpp
builds and test harnesses) is superseded by the data above. Absolute pp
values differ because the old data used different model variants and
test parameters.

## Solver accuracy

After the refactor, the solver picks within 5% of the empirical
peak for all tested models on Strix Halo (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next,
Laguna Q5_K_XL). The solver's (batch, ubatch) defaults now match the benchmark
peaks for all three primary models.

To override the solver's choice: use `--ubatch-size N` and `--batch-size N`
on the `llm` command line. The overrides win.


---

# gufo solver

`solver/gufo.sh` is a policy solver. gufo plans its own memory against the
unified pool, so this solver does not compute KV budgets. It decides:

- family (from GGUF keys, then filename)
- speculative mode and drafter (validated before use)
- draft count (2 for Laguna's DFlash2, 7 elsewhere)
- context (native by default)
- sessions
- KV precision
- sparse attention (Laguna)

Bias semantics:

| Bias | KV cache | Effect |
|---|---|---|
| `context` (default) | q8_0 | Drop the drafter; the freed headroom and the denser cache both go to context |
| `mtp` | f16 | Keep the drafter for decode speed |
| `quality` | f16 | Pin f16, drop the drafter |
| `speed` | f16 | 65536 context, sparse attention off, no drafter |

The bias supplies the default for the dimensions it steers. An explicit
`--kv-cache` or `--ctx` is the narrower request and outranks it, so
`--bias context --kv-cache f16` runs f16.

`context` is the only bias that moves KV precision, and q8_0 is why: a row
keeps 53.1% of its F16 bytes for about -3.7% prefill and decode (see the
engine's `docs/internal/models/laguna/KV-Q8.md`). That is the trade the bias
is making. Families with no cache switch (`deepseek`, `glm`, `flint`) stay on
f16 and say so in the reason list rather than requesting a format gufo cannot
run.

KV translation happens in `lib/engines/gufo.sh`:

- Laguna: `LAGUNA_KV_CACHE_TYPE=q8_0` when `--kv-cache q8_0`
- Qwen families: `QWEN_KV_CACHE_TYPE=q8_0`
- Other families: gufo has no switch; the request is reported and ignored

Sparse attention maps to `LAGUNA_SPARSE_ATTN=1` and is Laguna-only.

Speculative drafters are discovered from the model base name. A patched
`-gufo.gguf` copy is preferred, and `tools/gguf_validate.py` rejects a
drafter the loader cannot execute before the server is started.
<!-- Documentation license: CC-BY-NC-SA-4.0. Source code: GPL-3.0-or-later. -->
