---
name: gpu-memory-solver-audit
description: Use this skill when auditing and fixing a llama.cpp GPU memory solver for AMD APUs/UMA systems. It captures the complete methodology, common pitfalls, and discovered patterns from the llm-scripts project.
---

# GPU Memory Solver Audit & Fix Methodology

## Core Problem Pattern (observed across multiple systems)

The solver over-allocates GPU memory estimates, causing either:
1. **False negatives**: Solver rejects configs that actually fit, forcing suboptimal ctx/KV quality
2. **False positives**: Solver green-lights configs that OOM at runtime (DeviceLost crashes)

## Phase 1: Investigation Methodology

### 1.1 Extract GGUF Metadata
Always read the model's GGUF metadata, not just the filename. Critical fields:
- `context_length` — native context (may be 128k, 200k, 256k, 512k, 1M)
- `*..block_count` — total layers (e.g., 41 for Qwen3.6 MoE)
- `*..full_attention_interval` — KV cache uses only this many layers!
- `*..attention.head_count` — n_attn (full attention)
- `*..attention.head_count_kv` — hckv (key/value head count)
- `*..attention.key_length` — kl
- `*..attention.value_length` — vl
- `*..expert_count` / `*..expert_used_count` — MoE detection
- `*..nextn_predict_layers` — MTP layers (often missed!)

**Critical insight**: `block_count` is NOT the KV layer count. Use
`block_count / full_attention_interval` for the actual KV layers.
Example: block_count=41, full_attention_interval=4 → KV layers = 10
(the remaining 31 layers are MoE/SSM layers with no KV).

### 1.2 Measure GPU Memory Budget
On AMD UMA APUs (RADV):
- VRAM total = `mem_info_vram_total` (BIOS carveout, often tiny: 512 MiB to 8 GiB)
- GTT total = `mem_info_gtt_total` (system RAM mapped for GPU)
- GPU-visible = VRAM + GTT
- Effective budget = min(VRAM+GTT - 512 MiB reserve, total_RAM - OS_reserve)

**Critical insight**: On UMA, VRAM+GTT IS system RAM. Double-counting
this pool between the GPU budget and system budget is the #1 cause of
false negatives.

### 1.3 KV Cache Math Verification
```
kv_per_token_per_layer = hckv * (kl * kv_type_bytes + vl * kv_type_bytes)
kv_per_token = kv_per_token_per_layer * n_attn
kv_total = ctx_size * kv_per_token * n_parallel
```

Example (Qwen3.6 q8_0):
- hckv=2, kl=256, vl=256, n_attn=16, q8_0=1 byte
- kv_per_token_per_layer = 2 * (256 + 256) = 1024
- kv_per_token = 1024 * 16 = 16384 bytes
- KV @ 131072 ctx = 131072 * 16384 = 2.0 GiB

### 1.4 Runtime Memory Profiling
Launch the server manually with verbose logging and check actual GPU
memory usage:
```
cat /sys/class/drm/card0/device/mem_info_vram_used
cat /sys/class/drm/card0/device/mem_info_gtt_used
```
Compare actual peak usage against the solver's estimate to find
gaps (f16-B path, descriptor sets, transient buffers, etc.).

## Phase 2: Common Bug Patterns

### Bug Pattern 1: Static Context Size Grid
**Symptom**: Solver can't reach the model's native context_length.
**Root cause**: Using a static grid (65536, 98304, 131072, 196608, 262144...)
instead of the GGUF's native `context_length`.
**Fix**: Read `context_length` from GGUF metadata, merge with the standard
grid, and try largest-first until one fits.

### Bug Pattern 2: KV Layer Count Over-Count
**Symptom**: Solver thinks model+KV needs 34 GiB when it actually needs 24 GiB.
**Root cause**: Using `block_count` (41) instead of `block_count / full_attention_interval`
(10) for KV cache sizing.
**Fix**: Compute KV layers as `block_count / full_attention_interval`
(when `full_attention_interval` is present in metadata).

### Bug Pattern 3: UMA Double-Counting
**Symptom**: System check fails even with plenty of free RAM.
**Root cause**: On UMA, VRAM+GTT is the same physical RAM as system RAM.
Subtracting the full GPU allocation from MemAvailable makes sys_budget=0.
**Fix**: sys_budget = MemAvailable - mem_needed (GPU allocation already
covers VRAM+GTT). Accept sys_budget=0 as a pass (GPU budget check is
sufficient).

### Bug Pattern 4: In-Model MTP Double-KV
**Symptom**: Solver adds ~2-5 GiB of MTP overhead that doesn't exist.
**Root cause**: In-model MTP (nextn_predict_layers=1) uses a unified KV
cache with --kv-unified. The MTP KV is tiny (~250 MiB), not a full
duplicate.
**Fix**: MTP overhead = MTP_weight_est + (kv_per_token_per_layer * hckv
* ctx_size), NOT kv_per_token * ctx_size / n_attn * ctx_size.

### Bug Pattern 5: Checkpoint Memory Unaccounted
**Symptom**: Server loads but crashes with DeviceLost during checkpoint
creation or during initial KV cache init.
**Root cause**: llama.cpp's --ctx-checkpoints pre-allocates a KV snapshot
ring buffer in GPU VRAM. The solver doesn't include this in mem_needed.
During checkpoint creation (state_seq_get_data), a transient copy buffer
doubles the KV allocation briefly.
**Fix**: Add checkpoint storage (ckpt_min_step * kv_per_token * ckpt_count)
to _opt_gpu_memory. After the fit is chosen, verify the transient copy
peak (ckpt_min_step * kv_per_token) fits; if not, reduce checkpoint count.

### Bug Pattern 6: Conservative Budget Reserve
**Symptom**: Solver leaves 2+ GiB unused on LLM-focused systems.
**Root cause**: Generic budget formula reserves too much for mixed-use.
**Fix**: On LLM-only systems: 512 MiB reserve (not 2 GiB), 1% headroom
(not 5%).

### Bug Pattern 7: Load Mode (mmap vs dio)
**Symptom**: Server loads but crashes with DeviceLost during graph execution.
**Root cause**: mmap load mode triggers page-fault storms on UMA with
large contexts. dio (eager pread) loads the model fully into GPU-accessible
memory.
**Fix**: Use `dio` for MoE models on UMA. Use `auto` (mmap) only for
small models that fit entirely in VRAM.

## Phase 3: Verification Protocol

### Step 1: Syntax + Shellcheck
```bash
bash -n scripts/optimize.sh
shellcheck scripts/optimize.sh
```

### Step 2: Solver Output Verification
```bash
./the llama.cpp adapter --print-profile <model> 2>&1 | grep 'Solver chose'
./the llama.cpp adapter --print-profile --bias context <model>
./the llama.cpp adapter --print-profile --bias quality <model>
./the llama.cpp adapter --print-profile --bias speed <model>
```

Expected: each bias produces a distinct, sensible output:
- mtp: largest ctx, MTP draft on
- context: larger ctx, draft off
- quality: smaller ctx, f16 KV
- speed: smallest ctx (MIN_CTX), f16 KV

### Step 3: Server Launch + Multi-Inference Test
Launch with the solver's output and run 3+ inference calls:
1. First inference (initial load)
2. Second inference (triggers context shift / checkpoint creation)
3. Third inference (stress test)

All must succeed without DeviceLost crashes. Check:
- `/sys/class/drm/card*/device/mem_info_*_used` before/after
- Server log for `DeviceLost`, `not enough memory`, `ErrorDeviceLost`
- Answers must be correct

### Step 4: Memory Math Cross-Check
Compare solver's mem_needed estimate against actual runtime usage:
```
actual_usage = model_bytes + kv_total + mtp_kv + ckpt_storage + compute + system + vk_overhead
solver_estimate = _opt_gpu_memory(model, kv, ctx, draft, MTP, compute, system, ckpt)
```
The gap (vk_overhead) is ~1-2 GiB on RADV/Vulkan (f16-B path,
descriptor sets, transient buffers). If the gap exceeds the 1% headroom,
add a fixed buffer or increase the headroom.

## Phase 4: Variable Contract (DO NOT BREAK)

The solver exports these variables that the llama.cpp adapter depends on:
- `SOLVER_CTX_SIZE` — context size in tokens
- `SOLVER_K_TYPE` / `SOLVER_V_TYPE` — KV cache types (f16, q8_0, q4_0)
- `SOLVER_KV_CACHE_MIB` — KV cache size in MiB
- `SOLVER_NGL` — GPU layers
- `SOLVER_LOAD_MODE` — dio, auto, none
- `SOLVER_MOE_STRATEGY` — gpu, residency, cpu
- `SOLVER_CHECKPOINTS` / `SOLVER_CHECKPOINT_MIN` — checkpoint config
- `SOLVER_DRAFT_ENABLE` — MTP draft on/off
- `SOLVER_BIAS` — bias direction
- `SOLVER_REASONS` — debug trace of decisions

## Phase 5: Key Formulas Reference

```
kv_per_token_per_layer = hckv * (kl * k_bytes + vl * v_bytes)
kv_per_token = kv_per_token_per_layer * n_attn
kv_total = ctx * kv_per_token * n_parallel

# Model GPU footprint (gpu strategy)
# For full GPU offload: model_bytes * (ngl / n_layer) * gpu_fraction
# gpu_fraction for MoE on Vulkan = 1.0 (all experts on GPU)

# Checkpoint memory (GPU-resident)
ckpt_min_step = max(ctx / ckpt_count, 8192)
# _opt_scale_checkpoints bumps min_step to 32768
ckpt_storage = ckpt_min_step * kv_per_token * ckpt_count * n_parallel
ckpt_copy_peak = ckpt_min_step * kv_per_token * n_parallel  # transient

# GPU memory = model + KV + draft_KV + MTP + compute + system + checkpoints
# Budget = min(VRAM+GTT - 512MiB, total_RAM - OS_reserve)
# On UMA: sys_budget = MemAvailable - mem_needed (GPU alloc = VRAM+GTT = sys RAM)
```

## Phase 6: Anti-Patterns to Avoid

1. **Don't use static context grids** — read from GGUF, merge with standard grid
2. **Don't use `block_count` for KV layers** — divide by `full_attention_interval`
3. **Don't double-count VRAM+GTT on UMA** — it IS system RAM
4. **Don't add full MTP KV as overhead** — with --kv-unified, it's 250 MiB, not 2 GiB
5. **Don't use 5% headroom on LLM-only systems** — 1% is sufficient, saves ~1.25 GiB at 25 GiB
6. **Don't ignore checkpoint memory** — it's GPU-resident, not CPU-resident on Vulkan
7. **Don't use mmap on UMA for large MoE** — dio prevents page-fault storms
8. **Don't trust vendor docs for VRAM** — BIOS carveout is often 512 MiB, not 96 GiB
9. **Don't assume `sys_budget = 0` means failure** — on UMA, sys=0 is often OK if GPU budget check passes
10. **Don't hardcode model-specific values** — use GGUF metadata + hardware detection