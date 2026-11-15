#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Optimistic-First Solver for llama.cpp Configuration
# =============================================================================
# Source this file (do not execute directly):
#   source solver/llama.sh
#
# Agentic development requires a large context window; the solver will not
# reduce context below MIN_CTX (default 65536) unless absolutely impossible.
# Outputs are exposed via globals prefixed with SOLVER_*. the engine adapter reads
# these after calling solve_optimal_config() and applies user overrides last.
#
# Public entry points (the ONLY functions the llama.cpp adapter calls):
#   solve_optimal_config <model_path>
#   apply_user_overrides
#
# All _opt_* / _reduce_* functions are internal and never invoked by the
# caller; they exist solely to support solve_optimal_config.

[[ -n "${_LLM_SOLVER_LLAMA_LOADED:-}" ]] && return 0
_LLM_SOLVER_LLAMA_LOADED=1

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_OPT_GIB=1073741824
: "${MIN_CTX:=65536}"

# OS reserve used for the "does the model fit in RAM?" load-mode decision.
_OPT_OS_RESERVE_BYTES=$(( 4 * 1073741824 ))

# Solver bias: steers the optimistic-first search toward a dimension.
# mtp (default) = maximize throughput with MTP draft
# context = maximize context length (drops draft)
# quality = maximize KV quality (f16 KV, no draft)
# speed = maximize decode throughput (small ctx, large batch, no draft)
# Read at solve_optimal_config call time (not source time) so CLI flags
# parsed later by the llama.cpp adapter are picked up. The adapter defaults
# LLM_BIAS to "context" at source time, so this default only matters for
# callers that source solver/llama.sh without setting LLM_BIAS.
_OPT_BIAS="${OVERRIDE_BIAS:-${LLM_BIAS:-context}}"

# GGUF metadata reader lives in scripts/lib-gguf-meta.sh (shared with optimize-gufo.sh)
# shellcheck source=lib/gguf.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/gguf.sh"
[[ -n "${PROJECT_ROOT:-}" ]] || PROJECT_ROOT="$LLM_ROOT"

# -----------------------------------------------------------------------------
# Architecture detection
# -----------------------------------------------------------------------------
# is_qwen4exp / is_mla are derived from GGUF metadata here. is_moe, is_ssm,
# is_mtp are set by the adapter before
# solve_optimal_config is called; they are consumed as inputs below.
declare -g is_qwen4exp=false is_mla=false

_opt_is_qwen4exp() {
    local key
    for key in "${!SOLVER_GGUF[@]}"; do
        [[ "$key" == qwen4exp.* ]] && return 0
    done
    return 1
}

_opt_is_mla() {
    local hckv="${SOLVER_GGUF[head_count_kv]:-${SOLVER_GGUF[n_head_kv]:-0}}"
    local hc="${SOLVER_GGUF[head_count]:-${SOLVER_GGUF[n_head]:-0}}"
    local kl="${SOLVER_GGUF[key_length]:-0}"
    local vl="${SOLVER_GGUF[value_length]:-0}"
    # Strong signal: head_count_kv=1 (single latent head).
    [[ "${hckv:-0}" -eq 1 ]] && return 0
    # Soft signal: key_length and value_length differ significantly AND
    # head_count_kv is small. MLA has K compressed to a smaller latent.
    if [[ -n "$kl" && -n "$vl" && "$vl" -gt 0 && "$kl" -gt 0 ]]; then
        local diff_pct=$(( (kl - vl) * 100 / kl ))
        [[ $diff_pct -gt 20 && "${hckv:-0}" -le 2 ]] && return 0
    fi
    # Arch prefix fallback (covers new MLA architectures before metadata
    # is added to the detection script).
    for key in "${!SOLVER_GGUF[@]}"; do
        case "$key" in
            deepseek2.*|deepseek3.*|deepseek4.*|glm4.*) return 0 ;;
        esac
    done
    return 1
}

_opt_is_uma() {
    # Pinned by a caller that already knows the memory model, or by the solver
    # regression harness so results do not depend on the host's sysfs.
    if [[ -n "${SOLVER_IS_UMA_OVERRIDE:-}" ]]; then
        [[ "${SOLVER_IS_UMA_OVERRIDE}" == "1" ]]
        return
    fi
    [[ "${BACKEND:-}" =~ ^(vulkan|rocm|metal)$ ]] || return 1
    local _vram_total=0 _gtt_total=0
    for _c in /sys/class/drm/card[0-9]/device; do
        [[ -d "$_c" ]] || continue
        _vram_total=$(cat "$_c/mem_info_vram_total" 2>/dev/null || echo 0)
        _gtt_total=$(cat "$_c/mem_info_gtt_total" 2>/dev/null || echo 0)
        break
    done
    [[ $_vram_total -gt 0 && $_gtt_total -gt $_vram_total ]]
}

# -----------------------------------------------------------------------------
# PLE (n-gram hash embedding) size for qwen4exp models.
# The PLE (~20M rows * 2560 dims) is always offloaded to CPU via -ot.
# Cached in $_SOLVER_PLE_CACHE so the Python scan runs at most once per
# solve pass (the scoring loop calls this from $() subshells).
# -----------------------------------------------------------------------------
_SOLVER_PLE_CACHE=""

_opt_qwen4exp_ple_bytes() {
    if [[ "${is_qwen4exp:-false}" != "true" ]]; then
        echo 0
        return
    fi
    # Cached value (pre-computed in the main shell; available in subshells).
    if [[ -n "$_SOLVER_PLE_CACHE" ]]; then
        echo "$_SOLVER_PLE_CACHE"
        return
    fi
    # Heuristic fallback: ~40% of model bytes (51B PLE params / 125B total).
    # Returning 0 is fatal: the solver would assume the entire model must fit
    # GPU, causing cascading detunes and a dio load-mode that OOM-kills the
    # kernel (the original bug). So always return at least the heuristic.
    local model_bytes="${MODEL_BYTES:-0}"
    [[ $model_bytes -le 0 ]] && { echo 0; return; }
    local model_path="${MODEL_PATH:-${MODEL:-}}"
    local ple_size=0
    if [[ -n "$model_path" && -f "$model_path" ]]; then
        ple_size=$(gguf_ple_bytes "$model_path")
    fi
    if [[ -z "$ple_size" || "$ple_size" -le 0 ]]; then
        ple_size=$(awk -v m="$model_bytes" 'BEGIN { printf "%.0f", m * 0.40 }')
    fi
    echo "$ple_size"
}

# -----------------------------------------------------------------------------
# System memory detection (mirrors the llama.cpp adapter functions)
# -----------------------------------------------------------------------------
_opt_get_total_memory_bytes() {
    [[ -n "${SOLVER_TOTAL_MEMORY_BYTES:-}" ]] && { echo "$SOLVER_TOTAL_MEMORY_BYTES"; return; }
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.memsize 2>/dev/null || echo 0
    else
        awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

_opt_get_available_memory_bytes() {
    [[ -n "${SOLVER_AVAILABLE_MEMORY_BYTES:-}" ]] && { echo "$SOLVER_AVAILABLE_MEMORY_BYTES"; return; }
    if [[ "$(uname -s)" == "Darwin" ]]; then
        local ps
        ps=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
        local out
        out=$(vm_stat 2>/dev/null)
        local free
        free=$(echo "$out" | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        local inactive
        inactive=$(echo "$out" | awk '/Pages inactive/ {gsub(/\./,"",$3); print $3}')
        echo $(( (${free:-0} + ${inactive:-0}) * ps ))
    else
        awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

# -----------------------------------------------------------------------------
# Memory math: KV cache sizing
# -----------------------------------------------------------------------------
_opt_kv_type_size() {
    case "$1" in
        f16|bf16) echo 2 ;;
        f32) echo 4 ;;
        q8_0|q4_1|q5_1) echo 1 ;;
        q4_0|q5_0|iq4_nl) echo "0.5625" ;;
        *) echo 2 ;;
    esac
}

_opt_layer_kv_bytes_per_token() {
    local k_type="$1" v_type="$2" head_count_kv="$3" key_len="$4" val_len="$5"
    local k_size v_size
    k_size=$(_opt_kv_type_size "$k_type")
    v_size=$(_opt_kv_type_size "$v_type")
    awk -v k="$k_size" -v v="$v_size" -v h="$head_count_kv" -v kl="$key_len" -v vl="$val_len" \
        'BEGIN { printf "%.0f", h * (kl * k + vl * v) }'
}

_opt_mtp_overhead_bytes() {
    local model_bytes="$1"
    local mtp_bytes=$(( model_bytes * 5 / 100 ))
    [[ $mtp_bytes -lt $(( 512 * 1048576 )) ]] && mtp_bytes=$(( 512 * 1048576 ))
    [[ $mtp_bytes -gt $(( 2048 * 1048576 )) ]] && mtp_bytes=$(( 2048 * 1048576 ))
    echo "$mtp_bytes"
}

# -----------------------------------------------------------------------------
# Model GPU footprint (offloaded bytes) given NGL + strategy.
#
# llama-context.cpp sets swa_full=true by default (no CLI flag to override to
# false), so ALL layers allocate a full-context KV cache regardless of
# full_attention_interval. The memory-effective attention-layer count is
# therefore always the full layer count — there is no SWA windowing in the
# memory math. (The dead per-SWA variables that used to be computed are gone;
# this reality is documented here once.)
# -----------------------------------------------------------------------------
_opt_model_gpu_footprint() {
    local ngl="$1" n_layer="$2" total_model_bytes="$3" moe_strategy="${4:-gpu}"
    if [[ "$ngl" -le 0 ]]; then
        echo 0
        return
    fi
    local gpu_fraction=1
    case "$moe_strategy" in
        gpu)    gpu_fraction=1 ;;
        residency)
            # Partial offload: ~30% of model on GPU, ~70% in system RAM.
            # Meaningful only for MoE — dense/SSM models have no experts to
            # selectively offload, so gpu_fraction=1.
            if [[ "${is_moe:-false}" == "true" ]]; then
                gpu_fraction=0.30
            fi
            ;;
        cpu)
            # Only attn/emb/head on GPU (~6% of MoE params).
            if [[ "${is_moe:-false}" == "true" ]]; then
                gpu_fraction=0.06
            fi
            ;;
    esac

    local ple_bytes=0
    if [[ "${is_qwen4exp:-false}" == "true" ]]; then
        ple_bytes=$(_opt_qwen4exp_ple_bytes)
    fi

    if [[ "$ngl" -ge "$n_layer" ]]; then
        awk -v frac="$gpu_fraction" -v total="$total_model_bytes" -v ple="$ple_bytes" \
            'BEGIN { printf "%.0f", frac * (total - ple) }'
    else
        awk -v ngl="$ngl" -v nl="$n_layer" -v frac="$gpu_fraction" \
            -v total="$total_model_bytes" -v ple="$ple_bytes" \
            'BEGIN { printf "%.0f", (ngl / nl) * frac * (total - ple) }'
    fi
}

# -----------------------------------------------------------------------------
# GPU-side memory (model + KV + draft + MTP + compute), with 5% headroom.
# KV cache is on GPU for ALL GPU-backed strategies (Vulkan/ROCm/Metal).
# -----------------------------------------------------------------------------
_opt_gpu_memory() {
    local offloaded_bytes="$1" kv_per_token="$2" ctx_size="$3"
    local draft_bytes="$4" draft_ctx="$5" n_parallel="$6" ubatch="$7"
    local ckpt_count="${8:-0}"

    local kv_total=$(( ctx_size * kv_per_token * n_parallel ))
    local draft_kv_total=0
    if [[ $draft_ctx -gt 0 && $draft_bytes -gt 0 ]]; then
        draft_kv_total=$(( draft_ctx * kv_per_token / 4 * n_parallel ))
        local draft_kv_cap=$(( draft_bytes * 10 / 100 ))
        [[ $draft_kv_total -gt $draft_kv_cap ]] && draft_kv_total=$draft_kv_cap
    fi

    local compute_bytes=$(( 512 * 1048576 + ubatch * 256 ))
    local system_bytes=$(( 256 * 1048576 ))

    local mtp_overhead=0
    # In-model MTP (no separate draft file): llama.cpp creates a separate
    # llama_context for the MTP block (typically 1 layer), which gets its
    # OWN small KV cache. The main model's KV cache is NOT duplicated —
    # with --kv-unified, the MTP context shares the unified KV cache.
    # Only charge the MTP overhead when the draft is actually enabled
    # (draft_bytes > 0), not when SOLVER_DRAFT_ENABLE is true but the
    # phase-1 loop chose "disabled" mode. The old check on
    # SOLVER_DRAFT_ENABLE caused the solver to over-charge for MTP even
    # when drafting was disabled, pushing configs out of budget unnecessarily.
    if [[ "${is_mtp:-false}" == "true" && "${_OPT_DRAFT_ACTIVE:-0}" == "1" && -z "${SOLVER_DRAFT_PATH:-}" ]]; then
        local mtp_kv=$(( kv_per_token * ctx_size / n_attn * n_parallel ))
        local mtp_weight_est=$(_opt_mtp_overhead_bytes "${MODEL_BYTES:-0}")
        mtp_overhead=$(( mtp_weight_est + mtp_kv ))
    fi

    # Checkpoint KV overhead (GPU-resident on Vulkan): the checkpoint
    # system stores KV snapshots in GPU VRAM. During checkpoint creation
    # (create_checkpoint -> state_seq_get_data), llama.cpp copies the FULL
    # active KV state into a temporary buffer before writing to the
    # checkpoint ring buffer. This transient copy occurs at checkpoint
    # boundaries (every SOLVER_CHECKPOINT_MIN tokens) and requires peak
    # memory: active KV + checkpoint storage + copy buffer.
    # We count the STORAGE (not the transient copy buffer) in mem_needed
    # — the copy buffer is brief (~50ms per checkpoint) and is accounted
    # for via the 5% headroom. If checkpoints would exceed budget,
    # _opt_scale_checkpoints reduces the count adaptively.
    local ckpt_kv_bytes=0
    if [[ $ckpt_count -gt 0 && $kv_per_token -gt 0 ]]; then
        # Use the actual checkpoint_min_step from _opt_scale_checkpoints
        # if available (it may differ from ctx_size/ckpt_count due to
        # the 8192 and 32768 clamping applied by _opt_scale_checkpoints).
        local ckpt_min_step="${SOLVER_CHECKPOINT_MIN:-0}"
        [[ $ckpt_min_step -eq 0 ]] && ckpt_min_step=$(( ctx_size / ckpt_count ))
        [[ $ckpt_min_step -lt 8192 ]] && ckpt_min_step=8192
        ckpt_kv_bytes=$(( ckpt_min_step * kv_per_token * ckpt_count * n_parallel ))
    fi

   local dynamic=$(( draft_bytes + kv_total + draft_kv_total + compute_bytes + system_bytes + mtp_overhead + ckpt_kv_bytes + ${SOLVER_RECR_KV_BYTES:-0} ))
   raw=$(( offloaded_bytes + dynamic ))
   # 5% headroom on the DYNAMIC portion only (KV + checkpoints + compute +
   # draft + MTP + recr), NOT on static model weights. The model weights
   # are a fixed allocation measured in GiB — 5% headroom on 21 GiB weights
   # wastes ~1.1 GiB unnecessarily, pushing configs like 128K ctx out of
   # budget when they actually fit. The 5% on dynamic memory covers the
   # observed ~0.96 GiB estimation gap plus checkpoint copy peaks (~100 MiB
   # transient buffers) without penalizing the static model footprint.
   local dynamic_with_headroom=$(( dynamic * 105 / 100 ))
   echo $(( offloaded_bytes + dynamic_with_headroom ))
}

# -----------------------------------------------------------------------------
# System-side memory (CPU-resident allocations: model tail, checkpoints,
# compute buffers). On UMA the GPU budget already covers VRAM+GTT so the
# system headroom is 0%; on discrete systems it is 10%.
# -----------------------------------------------------------------------------
_opt_system_memory() {
    local offloaded_bytes="$1" model_bytes="$2" kv_per_token="$3" ctx_size="$4"
    local draft_bytes="$5" draft_ctx="$6" n_parallel="$7" ubatch="$8"
    local strategy="${9:-gpu}" load_mode="${10:-dio}"

    local kv_total=$(( ctx_size * kv_per_token * n_parallel ))
    local draft_kv_total=0
    if [[ $draft_ctx -gt 0 && $draft_bytes -gt 0 ]]; then
        draft_kv_total=$(( draft_ctx * kv_per_token / 4 * n_parallel ))
        local draft_kv_cap=$(( draft_bytes * 10 / 100 ))
        [[ $draft_kv_total -gt $draft_kv_cap ]] && draft_kv_total=$draft_kv_cap
    fi

    local compute_bytes=$(( 512 * 1048576 + ubatch * 256 ))
    local system_bytes=$(( 256 * 1048576 ))

    local model_sys_bytes=0
    case "$strategy" in
        gpu)
            # qwen4exp: PLE is offloaded to CPU via -ot, so it's the only
            # CPU-resident model portion. The rest is GPU-side (covered by the
            # GPU budget check, no double-counting on UMA).
            if [[ "${is_qwen4exp:-false}" == "true" ]]; then
                model_sys_bytes=$(_opt_qwen4exp_ple_bytes)
            fi
            ;;
        cpu)
            model_sys_bytes="$model_bytes"
            ;;
        residency)
            # 70% of model lives in system RAM (1 - gpu_fraction=0.30). On UMA
            # the GPU-pinned 30% is already covered by the GPU budget. For
            # load_mode=none (eager pread) the full 70% is resident; for
            # mmap/auto the resident working set is smaller (MoE ~20%, dense ~30%).
            if [[ "$load_mode" == "none" ]]; then
                model_sys_bytes=$(awk -v m="$model_bytes" 'BEGIN { printf "%.0f", m * 0.70 }')
            elif [[ "${is_moe:-false}" == "true" ]]; then
                model_sys_bytes=$(awk -v m="$model_bytes" 'BEGIN { printf "%.0f", m * 0.20 }')
            else
                model_sys_bytes=$(awk -v m="$model_bytes" 'BEGIN { printf "%.0f", m * 0.30 }')
            fi
            ;;
    esac

    # Checkpoint hot-set memory: max(2 GiB, n_ctx_checkpoints * 400 MiB) per
    # active slot. Scaled to the actual average checkpoint KV size.
    # NOTE: For gpu strategy on UMA, checkpoint KV is now included in
    # _opt_gpu_memory (ckpt_kv_bytes), so it is excluded here to avoid
    # double-counting the same physical RAM. Only residency/cpu strategies
    # need checkpoint memory in the system budget (they store checkpoints
    # in CPU RAM, not GPU VRAM).
    local ckpt_mem_bytes=0
    if [[ "$strategy" != "gpu" || "${SOLVER_IS_UMA:-0}" != "1" ]]; then
        local ckpt_count="${SOLVER_CHECKPOINTS:-8}"
        local ckpt_avg_kv=$(( ctx_size * kv_per_token / 2 ))
        local ckpt_budget_mib=2048
        local ckpt_est_mib=$(( ckpt_avg_kv / 1048576 * ckpt_count ))
        [[ $ckpt_est_mib -gt $ckpt_budget_mib ]] && ckpt_est_mib=$ckpt_budget_mib
        ckpt_mem_bytes=$(( ckpt_est_mib * 1048576 * n_parallel ))
    fi

    local raw
    if [[ "$strategy" == "gpu" && "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        # On UMA with full GPU offload: ALL model + KV + compute + system
        # buffers are GPU-resident in VRAM+GTT (= system RAM). They are
        # already counted in _opt_gpu_memory's mem_needed. The only
        # CPU-resident allocation is the PLE (qwen4exp), handled by
        # model_sys_bytes. So sys_mem_needed = model_sys_bytes only.
        raw="$model_sys_bytes"
    else
        raw=$(( model_sys_bytes + kv_total + draft_kv_total + compute_bytes + system_bytes + ckpt_mem_bytes ))
    fi
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        echo "$raw"
    else
        echo $(( raw * 110 / 100 ))
    fi
}

# -----------------------------------------------------------------------------
# Consolidated fit check: computes mem_needed + sys_mem_needed + sys_budget
# for a given candidate configuration. Used by BOTH phase-1 (scoring loop)
# and phase-2 (detune loop), eliminating the duplicated scaffolding that
# previously diverged (UMA-on-residency).
#
# Globals read: SOLVER_GPU_BUDGET_BYTES|GPU_BUDGET_BYTES, SOLVER_IS_UMA,
#   SOLVER_LOAD_MODE, OVERRIDE_N_PARALLEL, MODEL_BYTES, is_*
# -----------------------------------------------------------------------------
_opt_compute_memory_fit() {
    # Inputs (all positional for clarity):
    # $1 kv_per_token  $2 ctx  $3 draft_bytes  $4 batch  $5 ubatch
    # $6 offloaded_bytes  $7 strategy  $8 ckpt_count
    local kv_per_token="$1" ctx="$2" draft_bytes="$3" ubatch="$5"
    local offloaded_bytes="$6" strategy="$7"
    local ckpt_count="${8:-0}"

    local n_parallel="${OVERRIDE_N_PARALLEL:-1}"

    # GPU-side: model-offload + KV + draft + MTP + compute + checkpoint KV + recurrent cache, 5% headroom.
    local mem_needed
    mem_needed=$(_opt_gpu_memory \
        "$offloaded_bytes" "$kv_per_token" "$ctx" \
        "$draft_bytes" "$ctx" "$n_parallel" "$ubatch" "$ckpt_count")

    # System-side: CPU-resident model portion + checkpoints + compute.
    local sys_mem_needed
    sys_mem_needed=$(_opt_system_memory \
        "$offloaded_bytes" "${MODEL_BYTES:-0}" "$kv_per_token" "$ctx" \
        "$draft_bytes" "$ctx" "$n_parallel" "$ubatch" \
        "$strategy" "${SOLVER_LOAD_MODE:-auto}")

    local sys_mem_avail=$(_opt_get_available_memory_bytes)

    # On UMA with gpu strategy: VRAM+GTT IS system RAM. mem_needed (model+
    # KV+ckpt+compute) lives in VRAM+GTT = system RAM. sys_mem_needed from
    # _opt_system_memory only counts the PLE (CPU-resident). The fit check
    # must also leave room for cache-ram, which shares the same physical RAM
    # pool. NOTE: the OS reserve is NOT added here — it is already subtracted
    # from the GPU budget via system_limit = vram + total_ram - os_reserve in
    # _compute_gpu_budget_bytes(). Adding it again to sys_mem_needed double-
    # counts the reserve, causing the solver to reject every gpu-strategy
    # config on tight UMA systems (e.g. a 21 GiB model on a 27 GiB GPU budget:
    # mem_needed=24 GiB < 27 GiB budget passes the GPU check, but
    # sys_mem_needed=4 GiB > sys_budget=0 GiB fails the system check).
    # The cache-ram estimate below also subtracts _solver_os_mib (derived from
    # MemAvailable) from the system leftover, so the OS reserve is accounted
    # for exactly once in the system-budget path.
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]] && [[ "$strategy" == "gpu" ]]; then
        # Estimate cache-ram: the solver allows 90% of the effective leftover
        # (min of GPU-leftover and system-leftover after OS reserve) as
        # cache-ram. Without this estimate in the fit check, the solver
        # accepts configs where the actual cache-ram pushes the total past
        # MemAvailable at the 95% pre-flight threshold. The estimate is
        # conservative (assumes the full 90% of the system leftover becomes
        # cache-ram).
        local _solver_gpu_left=$(( solver_budget_bytes - mem_needed ))
        local _solver_avail=$(( sys_mem_avail * 95 / 100 ))
        local _solver_os_mib
        [[ $(( _solver_avail / 1048576 )) -le 16384 ]] && _solver_os_mib=$(( _solver_avail / 1048576 / 4 )) || _solver_os_mib=8192
        local _solver_sys_left=$(( _solver_avail - _solver_os_mib * 1048576 - sys_mem_needed - mem_needed ))
        local _solver_eff=$(( _solver_gpu_left < _solver_sys_left ? _solver_gpu_left : _solver_sys_left ))
        [[ $_solver_eff -lt 0 ]] && _solver_eff=0
        sys_mem_needed=$(( sys_mem_needed + _solver_eff * 90 / 100 ))
    fi

    # Budget: raw GPU bytes vs. solver/system system bytes.
    local solver_budget_bytes="${SOLVER_GPU_BUDGET_BYTES:-${GPU_BUDGET_BYTES:-0}}"

    # System budget: MemAvailable minus the UMA-resident GPU allocation
    # (on UMA, VRAM+GTT is the same physical RAM as system RAM, so the GPU
    # commitment must be subtracted to avoid double-counting the same pool).
    # For cpu strategy, reserve 4 GiB OS headroom against total RAM.
    local sys_budget=0
    case "$strategy" in
        cpu)
            # cpu strategy: model lives entirely in system RAM (outside the
            # GPU VRAM+GTT pool). Budget = total RAM minus OS reserve.
            local sys_total_bytes=$(_opt_get_total_memory_bytes)
            sys_budget=$(( sys_total_bytes - _OPT_OS_RESERVE_BYTES ))
            ;;
        *)
            sys_budget=$sys_mem_avail
            if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
                # On UMA, VRAM+GTT IS system RAM. The GPU budget check
                # already ensures model+KV+overhead fits in VRAM+GTT. To
                # avoid double-counting the same physical RAM, subtract the
                # GPU-resident allocation (mem_needed) from the system budget.
                # For residency, also subtract the CPU-resident model fraction.
                if [[ "$strategy" == "residency" ]]; then
                    local cpu_model_frac
                    cpu_model_frac=$(awk -v m="${MODEL_BYTES:-0}" -v l="${SOLVER_LOAD_MODE:-auto}" \
                        'BEGIN { if (l == "none") printf "%.0f", m * 0.70; else if (m > 0) printf "%.0f", m * 0.20; else print 0 }')
                    sys_budget=$(( sys_mem_avail - mem_needed - cpu_model_frac ))
                else
                    # gpu strategy: ALL of mem_needed (model + KV + compute +
                    # MTP) is GPU-resident in VRAM+GTT (= system RAM).
                    # Use 95% of MemAvailable to match the llama.cpp adapter's pre-flight
                    # threshold — otherwise the solver declares "fits" while the
                    # pre-flight check (same 95% threshold) later aborts, causing
                    # a confusing "solver chose X but server refuses to start"
                    # situation instead of a graceful phase-2 KV downgrade.
                    sys_budget=$(( sys_mem_avail * 95 / 100 - mem_needed ))
                fi
            fi
            ;;
    esac
    [[ $sys_budget -lt 0 ]] && sys_budget=0

    # Results emitted as a parseable "key=value" block for the caller.
    echo "mem_needed=${mem_needed}"
    echo "sys_mem_needed=${sys_mem_needed}"
    echo "sys_budget=${sys_budget}"
    echo "solver_budget_bytes=${solver_budget_bytes}"
}

# Returns 0 if the candidate fits within BOTH GPU and system budgets.
_opt_fits_budget() {
    local mem_needed="$1" sys_mem_needed="$2" sys_budget="$3" solver_budget_bytes="$4"
    if [[ $mem_needed -le $solver_budget_bytes ]] || [[ $solver_budget_bytes -le 0 ]]; then
        # System check: on UMA with gpu strategy, sys_mem_needed is 0
        # (all memory is GPU-resident), so sys_mem_needed <= sys_budget
        # passes correctly. The old `|| sys_budget <= 0` bypass was a bug
        # — it let configs through when the system had zero RAM headroom
        # (e.g. MemAvailable < mem_needed on a 25 GiB box running a 22 GiB
        # model). Only bypass when sys_budget is strictly negative (meminfo
        # unavailable).
        if [[ $sys_mem_needed -le $sys_budget ]] || [[ $sys_budget -lt 0 ]]; then
            return 0
        fi
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Cache RAM target helpers
# -----------------------------------------------------------------------------
_opt_min_cache_ram_mib() {
    local model_bytes="$1"
    local size_gb=$(( model_bytes / _OPT_GIB ))
    local low_vram=0
    [[ ${SOLVER_GPU_BUDGET_BYTES:-0} -gt 0 && ${SOLVER_GPU_BUDGET_BYTES} -lt $(( 32 * _OPT_GIB )) ]] && low_vram=1

    # Low-VRAM MoE (~1 GB): enough headroom to cache the prompt on a
    # <32 GiB GPU budget while offloading the fewest compute layers.
    # Ample-budget MoE keeps the higher 3072/4096 reservations.
    if [[ "${is_moe:-false}" == "true" && $size_gb -ge 50 ]]; then
        echo 4096
    elif [[ "${is_moe:-false}" == "true" && $low_vram -eq 1 ]]; then
        echo 1024
    elif [[ "${is_moe:-false}" == "true" ]]; then
        echo 3072
    elif [[ $low_vram -eq 1 ]]; then
        echo 1024
    else
        echo 2048
    fi
}

# Compute SOLVER_CACHE_RAM from leftover memory after the chosen config.
_opt_update_cache_ram() {
    local solver_budget_bytes="$1" offloaded_bytes="$2" kv_per_token="$3"
    local ctx_size="$4" draft_bytes="$5" n_parallel="$6" ubatch="$7"

    [[ $solver_budget_bytes -le 0 ]] && return 0

    if [[ -n "${OVERRIDE_CACHE_RAM:-}" ]]; then
        SOLVER_CACHE_RAM="${OVERRIDE_CACHE_RAM}"
        SOLVER_REASONS+=("cache-ram: ${OVERRIDE_CACHE_RAM} MiB")
        return 0
    fi

    # n_parallel=1: host prompt cache is a back-to-back no-op in server-context.cpp
    # (save+immediately-load on the same slot). Skip the allocation so the memory
    # serves the KV cache — UNLESS the model is hybrid/recurrent (SSM), in which
    # case KV shifting is disabled (MROPE models like Qwen3.6 set
    # get_can_shift()=false), making the host prompt cache the critical fallback
    # for cache recovery when prompt discontinuities shrink the LCP below the
    # checkpoint position. Without it, n_past=0 and full prompt re-processing.
    if [[ "${OVERRIDE_N_PARALLEL:-1}" -le 1 && "${is_ssm:-false}" != "true" ]]; then
        SOLVER_CACHE_RAM=0
        SOLVER_REASONS+=("cache-ram: 0 MiB (n_parallel=1, host prompt cache is a no-op)")
        return 0
    fi

    local n_parallel="${OVERRIDE_N_PARALLEL:-1}"

    local used_gpu
    used_gpu=$(_opt_gpu_memory \
        "$offloaded_bytes" "$kv_per_token" "$ctx_size" \
        "$draft_bytes" "$ctx_size" "$n_parallel" "$ubatch" "${SOLVER_CHECKPOINTS:-0}")
    local gpu_leftover=$(( solver_budget_bytes - used_gpu ))

    local sys_needed_no_cache
    sys_needed_no_cache=$(_opt_system_memory \
        "$offloaded_bytes" "${MODEL_BYTES:-0}" "$kv_per_token" "$ctx_size" \
        "$draft_bytes" "$ctx_size" "$n_parallel" "$ubatch" \
        "${SOLVER_MOE_STRATEGY:-gpu}" "${SOLVER_LOAD_MODE:-auto}")

    # System budget: on UMA, VRAM+GTT IS system RAM, so the GPU-resident
    # allocation (used_gpu) is carved from the same physical RAM pool and
    # must be subtracted. On non-UMA, the system budget is total RAM minus
    # OS reserve.
    # Use MemAvailable (not MemTotal) on UMA: MemTotal counts kernel
    # buffers/cache as "available" on paper but they aren't actually free
    # for allocation. This mismatch caused the solver to compute cache-ram
    # from a 128 GiB MemTotal pool while the pre-flight check enforces 95%
    # of MemAvailable (114 GiB), pushing the total past the threshold.
    local sys_budget_mib
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        local sys_avail_bytes
        sys_avail_bytes=$(_opt_get_available_memory_bytes)
        local sys_avail_mib=$(( sys_avail_bytes / 1048576 ))
        local os_reserve_mib
        if [[ $sys_avail_mib -le 16384 ]]; then
            os_reserve_mib=$(( sys_avail_mib / 4 ))
        else
            os_reserve_mib=8192
        fi
        sys_budget_mib=$(( sys_avail_mib - os_reserve_mib ))
    else
        local sys_total_mib=$(( $(_opt_get_total_memory_bytes) / 1048576 ))
        local os_reserve_mib
        if [[ $sys_total_mib -le 16384 ]]; then
            os_reserve_mib=$(( sys_total_mib / 4 ))
        else
            os_reserve_mib=8192
        fi
        sys_budget_mib=$(( sys_total_mib - os_reserve_mib ))
    fi
    local sys_leftover=$(( sys_budget_mib * 1048576 - sys_needed_no_cache ))

    # On UMA, subtract the GPU-resident allocation (already counted in
    # sys_budget via MemAvailable which reflects current usage).
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        sys_leftover=$(( sys_leftover - used_gpu ))
    fi

    local effective_leftover=$(( gpu_leftover < sys_leftover ? gpu_leftover : sys_leftover ))
    [[ $effective_leftover -lt 0 ]] && effective_leftover=0

    # raw cache-ram = 90% of the effective leftover (10% headroom for
    # cache-write churn). The phase-2 adequacy detune (see solve_optimal_config)
    # is the authority that lifts this to the model-appropriate minimum; here
    # we only guarantee a positive value so the server receives a valid
    # --cache-ram argument even in the infeasible/degraded fallback.
    local total_cache_mib=$(( effective_leftover * 90 / 100 / 1048576 ))
    [[ $total_cache_mib -lt 1 ]] && total_cache_mib=1

    # Cap cache-ram at 25% of system RAM (prevents pathological allocations
    # on very high-RAM systems where the GPU leftover is artificially large).
    local max_total_cache_mib
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        local _sys_avail=$(_opt_get_available_memory_bytes)
        max_total_cache_mib=$(( _sys_avail * 25 / 100 / 1048576 ))
    else
        local _sys_total=$(_opt_get_total_memory_bytes)
        max_total_cache_mib=$(( _sys_total * 25 / 100 / 1048576 ))
    fi
    [[ $total_cache_mib -gt $max_total_cache_mib ]] && total_cache_mib=$max_total_cache_mib

    SOLVER_CACHE_RAM=$total_cache_mib
    SOLVER_REASONS+=("cache-ram: ${total_cache_mib} MiB")
}

# -----------------------------------------------------------------------------
# Load-mode decision (SINGLE SOURCE OF TRUTH).
#
# llama.cpp load modes:
#   auto - mmap if supported, none otherwise
#   none - eager pread of all tensors into a host buffer at startup
#   mmap - memory-map the GGUF file; lazy page-in on access
#   dio  - direct I/O (O_DIRECT, bypasses kernel page cache)
#
# UMA (integrated GPU, VRAM+GTT == system RAM): always "auto" (mmap) for
# page-cache L4 benefit and GPU overlap during lazy page-in. "none" eager
# pread blocks the GPU for 60+ seconds on large models. mmap lazy-faults.
#   EXCEPTION: qwen4exp on UMA. The ~27 GiB PLE tensor is offloaded to CPU
#   via -ot, but with "auto" (mmap) llama.cpp page-faults the PLE into
#   GPU-accessible GTT before the -ot migration, creating contention.
#   "none" pread skips the PLE (goes straight to CPU-resident memory).
#   Measured: "none" pp512 = 554 t/s vs "auto" = 361 t/s (+53%).
# "dio" is NOT used on UMA: O_DIRECT reserves the full VMA upfront, which on
# overcommit=0 kernels exceeds the commit limit for large models -> OOM-kill.
#
# Discrete GPU with dedicated VRAM: "none" (eager pread) is faster when the
# model fits (copy targets GPU-pinned VRAM, no page faults during compute).
# When it doesn't fit, "auto" (mmap) page-caches re-accessed tensor pages.
# -----------------------------------------------------------------------------
_opt_decide_load_mode() {
    # UMA load-mode decision:
    # The dio change in commit d712eb4 used O_DIRECT for MoE models under the
    # theory that it avoids mmap page-fault storms. But O_DIRECT reserves the
    # full VMA upfront (see the documentation block above: "dio is NOT used on
    # UMA: O_DIRECT reserves the full VMA upfront, which on overcommit=0
    # kernels exceeds the commit limit for large models -> OOM-kill"). Commit
    # 8e5ff71 already reverted dio->none for this reason. The d712eb4 change
    # re-introduced dio for MoE, which is contradictory and unsafe on the
    # 25-32 GiB handheld systems where the model alone consumes 21+ GiB.
    if _opt_is_uma; then
        # qwen4exp on UMA uses "none" (PLE offloaded to CPU via -ot, avoids
        # mmap contention during the PLE page-fault). Non-qwen4exp UMA uses
        # "auto" (mmap) for page-cache L4 benefit and lazy page-in; eager
        # pread ("none") creates a transient peak of 2x model size (host
        # buffer + GPU copy) that can OOM-kill on 32-125 GiB RAM systems.
        if [[ "${is_qwen4exp:-false}" == "true" ]]; then
            echo "none"
        else
            echo "auto"
        fi
        return
    fi

    # Discrete GPU path: decide whether the model fits in RAM with margin.
    local model_bytes="${MODEL_BYTES:-0}"
    [[ $model_bytes -eq 0 ]] && { echo "auto"; return; }

    local total_ram=$(_opt_get_total_memory_bytes)
    [[ $total_ram -eq 0 ]] && { echo "auto"; return; }

    local avail=$(( total_ram - _OPT_OS_RESERVE_BYTES ))
    [[ $avail -le 0 ]] && { echo "auto"; return; }
    # 85% of available RAM = "fits" threshold (15% headroom for KV + scratch).
    local threshold=$(( avail * 85 / 100 ))

    # qwen4exp: the PLE (~40% of model bytes) is offloaded to CPU via -ot, so
    # the GPU-resident eager read is model_bytes - ple_bytes. Use "none" when
    # that fits (avoids both dio OOM-kill and mmap scratch-gate pressure).
    if [[ "${is_qwen4exp:-false}" == "true" ]]; then
        local ple_bytes=$(_opt_qwen4exp_ple_bytes)
        local gpu_resident=$(( model_bytes - ple_bytes ))
        if [[ $gpu_resident -le $threshold ]]; then
            echo "none"
            return
        fi
        # PLE oversized: fall through to the general check below.
    fi

    if [[ $model_bytes -le $threshold ]]; then
        echo "none"
    else
        echo "auto"
    fi
}

# -----------------------------------------------------------------------------
# Archetype-aware defaults: (batch, ubatch) per model architecture.
# Derived from llama-bench sweeps across 17 model/hardware combinations.
# Decode (tg) is memory-bandwidth bound (<2% variance across batch/ubatch);
# prefill (pp) is what tuning moves.
# -----------------------------------------------------------------------------
_opt_start_optimistic() {
    SOLVER_MOE_STRATEGY="gpu"
    SOLVER_LOAD_MODE="auto"

    # Derive architecture flags from GGUF metadata (single source of truth).
    if _opt_is_qwen4exp; then
        is_qwen4exp=true
    else
        is_qwen4exp=false
    fi
    if _opt_is_mla; then
        is_mla=true
    else
        is_mla=false
    fi

        local ctx_train
        ctx_train=$(gguf_get context_length 32768)
    SOLVER_CTX_SIZE="$ctx_train"
    [[ $SOLVER_CTX_SIZE -gt 1048576 ]] && SOLVER_CTX_SIZE=1048576
    [[ $SOLVER_CTX_SIZE -lt $MIN_CTX ]] && SOLVER_CTX_SIZE=$MIN_CTX

    SOLVER_K_TYPE="f16"
    SOLVER_V_TYPE="f16"

    local size_gb_start=$(( ${MODEL_BYTES:-0} / _OPT_GIB ))
    if [[ "${is_qwen4exp:-false}" == "true" ]]; then
        # Qwen3.8-Flash-Next: PLE + hybrid (recurrent/linear) attention.
        SOLVER_UBATCH=2048; SOLVER_BATCH=8192
    elif [[ "${is_ssm:-false}" == "true" ]]; then
        # Pure SSM / Mamba / RWKV - linear-time recurrence.
        SOLVER_UBATCH=1024; SOLVER_BATCH=4096
    elif [[ "${is_mla:-false}" == "true" ]]; then
        # MLA (DeepSeek-V2/V3/V4, GLM-4.x): smaller KV cache enables larger ubatch.
        SOLVER_UBATCH=4096; SOLVER_BATCH=8192
    elif [[ "${is_moe:-false}" == "true" ]]; then
        if [[ $size_gb_start -ge 100 ]]; then
            SOLVER_UBATCH=4096; SOLVER_BATCH=8192
        elif [[ $size_gb_start -ge 60 ]]; then
            SOLVER_UBATCH=2048; SOLVER_BATCH=8192
        else
            SOLVER_UBATCH=2048; SOLVER_BATCH=4096
        fi
    else
        # Dense transformer.
        SOLVER_UBATCH=1024; SOLVER_BATCH=4096
    fi
    [[ $SOLVER_UBATCH -gt $SOLVER_BATCH ]] && SOLVER_UBATCH=$SOLVER_BATCH

    # Thread count: batch uses physical cores / 2 for decode parallelism.
    local phys_cores=${PHYSICAL_CORES:-}
    if [[ -z "$phys_cores" ]]; then
        if command -v lscpu &>/dev/null; then
            phys_cores=$(lscpu -p 2>/dev/null | grep -E '^[0-9]' | cut -d',' -f2 | sort -u | wc -l)
        else
            phys_cores=$(nproc)
            if [[ -f /sys/devices/system/cpu/smt/active ]] && [[ "$(cat /sys/devices/system/cpu/smt/active)" == "1" ]]; then
                phys_cores=$((phys_cores / 2))
            fi
        fi
    fi
    [[ -z "$phys_cores" || "$phys_cores" -lt 1 ]] && phys_cores=1
    SOLVER_THREADS_BATCH="$phys_cores"
    SOLVER_THREADS="$(( phys_cores / 2 ))"
    [[ $SOLVER_THREADS_BATCH -lt 1 ]] && SOLVER_THREADS_BATCH=1
    [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1

    SOLVER_DRAFT_ENABLE=true
    SOLVER_DRAFT_N_MAX=8
    SOLVER_VK_NPS="${GGML_VK_NODES_PER_SUBMIT:-}"
    SOLVER_REASONING_BUDGET="${LLM_REASONING_BUDGET:-8192}"
    SOLVER_CHECKPOINTS="${SOLVER_CHECKPOINTS:-}"
    SOLVER_NGL="$SOLVER_N_LAYER"
}

# -----------------------------------------------------------------------------
# Phase-2 detune steps (the ONLY list; the dead twin is gone).
# Each step is idempotent via its own done-flag.
# -----------------------------------------------------------------------------
_reduce_ctx() {
    local min_ctx=$MIN_CTX
    [[ $SOLVER_CTX_SIZE -le $min_ctx ]] && return 1
    local step=256
    [[ $SOLVER_CTX_SIZE -gt 262144 ]] && step=1024
    [[ $SOLVER_CTX_SIZE -gt 65536 ]] && step=512
    local new_ctx=$(( SOLVER_CTX_SIZE - step ))
    new_ctx=$(( new_ctx / step * step ))
    [[ $new_ctx -lt $min_ctx ]] && new_ctx=$min_ctx
    [[ $new_ctx -ge $SOLVER_CTX_SIZE ]] && return 1
    SOLVER_CTX_SIZE=$new_ctx
    SOLVER_REASONS+=("ctx: ${new_ctx}")
    return 0
}

_reduce_ngl() {
    [[ "${_SOLVER_DONE_reduce_ngl:-0}" == "1" ]] && return 1
    local current="$SOLVER_NGL"
    [[ $current -le 0 ]] && { _SOLVER_DONE_reduce_ngl=1; return 1; }
    local reduction=$(( SOLVER_N_LAYER / 10 ))
    [[ $reduction -lt 1 ]] && reduction=1
    local new_ngl=$(( current - reduction ))
    [[ $new_ngl -lt 0 ]] && new_ngl=0
    SOLVER_NGL="$new_ngl"
    SOLVER_REASONS+=("ngl: $new_ngl")
    [[ $new_ngl -eq 0 ]] && _SOLVER_DONE_reduce_ngl=1
    return 0
}

_reduce_kv_q8_0() {
    [[ "${_SOLVER_DONE_kv_q8_0:-0}" == "1" ]] && return 1
    # Downgrade from f16/bf16 to q8_0. Quantize BOTH K and V together
    # rather than V-first: the mixed-precision K=f16/V=q8_0 path triggers
    # a segfault in the unstable llama.cpp Vulkan backend (MUL_MAT_ID
    # f16-B path). Using q8_0/q8_0 avoids this while saving the same
    # memory. The quality difference is negligible for most models.
    if [[ "$SOLVER_K_TYPE" == "f16" || "$SOLVER_K_TYPE" == "bf16" ]]; then
        SOLVER_K_TYPE="q8_0"
        SOLVER_V_TYPE="q8_0"
        SOLVER_REASONS+=("KV: q8_0/q8_0")
        _SOLVER_DONE_kv_q8_0=1
        return 0
    fi
    return 1
}

_reduce_kv_q4_0() {
    [[ "${_SOLVER_DONE_kv_q4_0:-0}" == "1" ]] && return 1
    [[ "$SOLVER_K_TYPE" == "q4_0" && "$SOLVER_V_TYPE" == "q4_0" ]] && return 1
    # Quantize both K and V together (see _reduce_kv_q8_0 for rationale).
    SOLVER_K_TYPE="q4_0"
    SOLVER_V_TYPE="q4_0"
    _SOLVER_DONE_kv_q4_0=1
    SOLVER_REASONS+=("KV: q4_0/q4_0")
    return 0
}

_drop_draft() {
    [[ "${_SOLVER_DONE_drop_draft:-0}" == "1" || "$SOLVER_DRAFT_ENABLE" != "true" ]] && return 1
    SOLVER_DRAFT_ENABLE=false
    _SOLVER_DONE_drop_draft=1
    SOLVER_REASONS+=("draft: dropped")
    return 0
}

_reduce_ubatch() {
    [[ $SOLVER_UBATCH -le 512 ]] && return 1
    SOLVER_UBATCH=$(( SOLVER_UBATCH / 2 ))
    [[ $SOLVER_BATCH -gt $SOLVER_UBATCH ]] && SOLVER_BATCH=$SOLVER_UBATCH
    SOLVER_REASONS+=("ubatch: ${SOLVER_UBATCH}")
    return 0
}

# Detune priority (phase 2): KV downgrades -> ubatch -> ctx -> draft -> NGL.
_opt_detune_steps() {
    local bias="${_OPT_BIAS:-context}"
    case "$bias" in
        context)
            # Maximize context: reduce ctx only after KV quality, draft,
            # and checkpoint overhead are already squeezed. Dropping ctx
            # is the last resort since this bias is specifically asking for
            # the largest possible context.
            cat <<'STEPS'
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ubatch
_drop_draft
_reduce_ctx
_reduce_ngl
STEPS
            ;;
        quality)
            # Maximize KV fidelity: resist KV quantization, drop draft first
            # (frees MTP KV overhead), then reduce ubatch, ctx, and NGL.
            # KV downgrade is last to preserve f16 precision.
            cat <<'STEPS'
_drop_draft
_reduce_ubatch
_reduce_ctx
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ngl
STEPS
            ;;
        speed)
            # Fastest decode: reduce ctx first (smaller ctx = faster decode),
            # then KV, draft, ubatch, NGL.
            cat <<'STEPS'
_reduce_ctx
_reduce_ubatch
_drop_draft
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ngl
STEPS
            ;;
        mtp|*)
            # Default (mtp): balanced — same as before.
            cat <<'STEPS'
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ubatch
_reduce_ctx
_drop_draft
_reduce_ngl
STEPS
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Checkpoint auto-scaling (single implementation; called pre-fit and post-fit
# with different caps).
# -----------------------------------------------------------------------------
_opt_scale_checkpoints() {
    # $1 = ctx_size, $2 = max_count
    local ctx="$1" max_cp="$2"
    if [[ -z "${SOLVER_CHECKPOINTS:-}" ]]; then
        local base_ctx=65536 base_cp=2 scale_per=8192
        if [[ $ctx -gt $base_ctx ]]; then
            local extra=$(( (ctx - base_ctx) / scale_per ))
            SOLVER_CHECKPOINTS=$(( base_cp + extra ))
        else
            SOLVER_CHECKPOINTS=$base_cp
        fi
        [[ $SOLVER_CHECKPOINTS -gt $max_cp ]] && SOLVER_CHECKPOINTS=$max_cp
    fi
    if [[ -z "${SOLVER_CHECKPOINT_MIN:-}" ]]; then
        SOLVER_CHECKPOINT_MIN=$(( ctx / SOLVER_CHECKPOINTS ))
        [[ $SOLVER_CHECKPOINT_MIN -lt 8192 ]] && SOLVER_CHECKPOINT_MIN=8192
        # Cap at 32768 — checkpoints larger than 32K tokens waste GPU memory
        # without improving resume quality. Previously a floor of 32768 was
        # enforced here, so a legitimately smaller min_step (smaller ctx or
        # more checkpoints) was inflated to 32768, overstating checkpoint
        # storage by up to 4x and rejecting 128K configs that actually fit.
        [[ $SOLVER_CHECKPOINT_MIN -gt 32768 ]] && SOLVER_CHECKPOINT_MIN=32768
        local ckpt_max=$(( ctx / SOLVER_CHECKPOINT_MIN ))
        [[ $SOLVER_CHECKPOINTS -gt $ckpt_max ]] && SOLVER_CHECKPOINTS=$ckpt_max
    fi
    if [[ -z "${SOLVER_CHECKPOINT_EVERY_N_TOKENS:-}" ]]; then
        SOLVER_CHECKPOINT_EVERY_N_TOKENS="-1"
    fi
}

# -----------------------------------------------------------------------------
# Profile selection (replaces the old _opt_pick_legacy_profile).
# -----------------------------------------------------------------------------
_opt_pick_profile() {
    local model_bytes="${MODEL_BYTES:-0}"
    local size_gb=$(( model_bytes / 1073741824 ))
    # Names describe the model, not the machine: the same archetype defaults
    # apply on every tier. Order matters: qwen4exp and MLA models also carry
    # MoE or recurrent layers, and the more specific family is the useful
    # label.
    # Same order as the archetype defaults in _opt_start_optimistic: a
    # recurrent model also reports head_count_kv=1, which the MLA test
    # matches, so SSM has to be decided first.
    if [[ "${is_qwen4exp:-false}" == "true" ]]; then
        echo "qwen4exp"
    elif [[ "${is_ssm:-false}" == "true" ]]; then
        echo "ssm"
    elif [[ "${is_mla:-false}" == "true" ]]; then
        echo "mla"
    elif [[ "${is_moe:-false}" == "true" ]]; then
        [[ $size_gb -gt 50 ]] && echo "moe-large" || echo "moe-small"
    else
        echo "dense"
    fi
}

# -----------------------------------------------------------------------------
# Builder: produce the scored candidate list for phase-1.
# Each entry is a parseable "combo_key" string with an attached score,
# emitted as "score:strategy:ctx:kvq:draft:batch:ubatch".
# -----------------------------------------------------------------------------
_opt_build_scored_combos() {
    local n_layer="$1" hckv="$2" kl="$3" vl="$4" model_bytes="$5"

    # Candidate ctx values: start from the model's native context_length
    # (capped at 1M). _opt_build_ctx_candidates merges the native ctx
    # with the standard grid ({262144, 196608, 131072, 98304, 65536})
    # so models with non-standard context lengths are handled correctly.
    # The phase-1 loop tries candidates largest-first and stops at first fit.
    local ctx_train
        ctx_train=$(gguf_get context_length 32768)
    [[ $ctx_train -gt 1048576 ]] && ctx_train=1048576
    [[ $ctx_train -lt $MIN_CTX ]] && ctx_train=$MIN_CTX
    local ctx_values=()
    read -ra ctx_values <<< "$(_opt_build_ctx_candidates "$ctx_train")"

    # KV cache quality candidates (preference order, scored below).
    # Order: f16/f16 -> q8_0/q8_0 -> f16/q8_0
    # For memory savings, q8_0/q8_0 (both quantized) is preferred over
    # f16/q8_0 (mixed precision). The mixed-precision K=f16/V=q8_0 path
    # triggers a segfault in the unstable llama.cpp Vulkan backend
    # (MUL_MAT_ID f16-B path). For models where q8_0/q8_0 fits, it avoids
    # the crash while saving the same memory. f16/q8_0 is the fallback.
    local kv_qualities=("f16/f16" "q8_0/q8_0" "f16/q8_0")

    # Strategies: gpu always tried; residency/cpu only for MoE.
    local strategies=("gpu")
    if [[ "${is_moe:-false}" == "true" ]]; then
        # Check CPU-MoE viability: model fits in system RAM with OS reserve.
        local sys_total_bytes=$(_opt_get_total_memory_bytes)
        local sys_avail_bytes=$(( sys_total_bytes - _OPT_OS_RESERVE_BYTES ))
        [[ ${MODEL_BYTES:-0} -le $sys_avail_bytes ]] && strategies+=("residency" "cpu")
    fi

    # (batch, ubatch) candidate pairs: optimistic defaults first, then
    # empirically-close alternatives within ~5% of peak.
    local batch_candidates=() ubatch_candidates=()
    batch_candidates+=("$SOLVER_BATCH")
    ubatch_candidates+=("$SOLVER_UBATCH")
    [[ "$SOLVER_UBATCH" -gt 1024 ]] && ubatch_candidates+=(1024)
    [[ "$SOLVER_BATCH" -lt 8192 ]] && batch_candidates+=(8192)
    [[ "$SOLVER_UBATCH" -lt 4096 ]] && ubatch_candidates+=(4096)
    [[ "$SOLVER_BATCH" -ge 4096 && "$SOLVER_BATCH" -ne 4096 ]] && batch_candidates+=(4096)
    [[ "$SOLVER_UBATCH" -lt 2048 ]] && ubatch_candidates+=(2048)
    [[ "$SOLVER_BATCH" -gt 2048 ]] && batch_candidates+=(2048)

    # Deduplicate.
    local _seen_u="" _seen_b="" _u_dedup=() _b_dedup=()
    for u in "${ubatch_candidates[@]}"; do
        if [[ " $_seen_u " != *" $u "* ]]; then
            _seen_u+=" $u"; _u_dedup+=("$u")
        fi
    done
    for b in "${batch_candidates[@]}"; do
        if [[ " $_seen_b " != *" $b "* ]]; then
            _seen_b+=" $b"; _b_dedup+=("$b")
        fi
    done
    ubatch_candidates=("${_u_dedup[@]}")
    batch_candidates=("${_b_dedup[@]}")
    unset _seen_u _seen_b _u_dedup _b_dedup

    declare -A combo_score
    for strategy in "${strategies[@]}"; do
        for ctx in "${ctx_values[@]}"; do
            for kvq in "${kv_qualities[@]}"; do
                # When the bias has explicitly disabled draft (context/quality/
                # speed/checkpoints), only consider "disabled" mode so the bias
                # is respected instead of being overridden by the +5 draft_score.
                local draft_modes_for_strategy=("disabled")
                if [[ "${SOLVER_DRAFT_ENABLE:-true}" == "true" ]]; then
                    draft_modes_for_strategy=("enabled")
                    [[ "$strategy" == "gpu" ]] && draft_modes_for_strategy=("enabled" "disabled")
                else
                    [[ "$strategy" == "gpu" ]] && draft_modes_for_strategy=("disabled" "enabled")
                    # Remove duplicates and "enabled" since bias disabled it
                    draft_modes_for_strategy=("disabled")
                fi
                for draft_mode in "${draft_modes_for_strategy[@]}"; do
                    for b in "${batch_candidates[@]}"; do
                        for u in "${ubatch_candidates[@]}"; do
                            [[ $u -gt $b ]] && continue

                            local strategy_score=0
                            [[ "$strategy" == "gpu" ]] && strategy_score=300
                            [[ "$strategy" == "residency" ]] && strategy_score=200
                            [[ "$strategy" == "cpu" ]] && strategy_score=150
                            if [[ "${OVERRIDE_MOE_STRATEGY:-}" == "residency" ]]; then
                                [[ "$strategy" == "residency" ]] && strategy_score=300
                                [[ "$strategy" == "gpu" ]] && strategy_score=299
                            fi

                            local ctx_score=0
                            case "$ctx" in
                                1048576) ctx_score=110 ;;
                                524288)  ctx_score=105 ;;
                                262144)  ctx_score=100 ;;
                                196608)  ctx_score=95  ;;
                                131072)  ctx_score=90  ;;
                                98304)   ctx_score=85  ;;
                                65536)   ctx_score=80  ;;
                                *)      ctx_score=$(( ctx / 1000 )) ;;
                            esac

                            # Bias-aware ctx scoring: speed bias prefers smaller
                            # ctx (faster decode, less KV memory); context/quality
                            # bias prefers larger ctx.
                            local bias="${_OPT_BIAS:-context}"
                            if [[ "$bias" == "speed" ]]; then
                                # Invert: smaller ctx gets higher score
                                ctx_score=$(( 200 - ctx_score ))
                            fi
                            [[ "$kvq" == "f16/f16" ]] && kv_score=30
                            [[ "$kvq" == "q8_0/f16" ]] && kv_score=25
                            [[ "$kvq" == "q8_0/q8_0" ]] && kv_score=20
                            [[ "$kvq" == "q4_0/q4_0" ]] && kv_score=10

                            local draft_score=0
                            [[ "$draft_mode" == "enabled" ]] && draft_score=5
                            if [[ "$draft_mode" == "enabled" && "${is_qwen4exp:-false}" == "true" && "${SOLVER_DRAFT_N_MAX:-0}" -gt 0 ]]; then
                                draft_score=$((draft_score + 120))
                            fi

                            # Bias-aware scoring adjustments.
                            local bias="${_OPT_BIAS:-context}"
                            case "$bias" in
                                context)
                                    # Prefer draft-off (saves MTP KV overhead for ctx)
                                    [[ "$draft_mode" == "enabled" ]] && draft_score=$((draft_score - 20))
                                    ;;
                                quality)
                                    # Prefer f16 KV + draft-off for accuracy.
                                    # +200 to kv_score ensures f16/f16 at a
                                    # smaller ctx outranks q8_0/q8_0 at a
                                    # larger ctx (ctx_score diff at ctx 131072 vs
                                    # 196608 = 10*10=100 points, plus batchub).
                                    [[ "$kvq" == "f16/f16" ]] && kv_score=$((kv_score + 200))
                                    [[ "$draft_mode" == "disabled" ]] && draft_score=$((draft_score + 10))
                                    ;;
                                speed)
                                    # Prefer smaller ctx (faster decode) + draft-off
                                    [[ "$draft_mode" == "disabled" ]] && draft_score=$((draft_score + 10))
                                    ;;
                            esac

                            local batchub_score
                            batchub_score=$(_opt_archetype_batchub_score "$b" "$u" "$SOLVER_BATCH" "$SOLVER_UBATCH")

                            combo_score["${strategy}:${ctx}:${kvq}:${draft_mode}:${b}:${u}"]=$(( strategy_score * 1000 + ctx_score * 10 + kv_score + draft_score + batchub_score ))
                        done
                    done
                done
            done
        done
    done

    local sorted_combos=()
    for combo in "${!combo_score[@]}"; do
        sorted_combos+=("${combo_score[$combo]}:$combo")
    done
    mapfile -t sorted_combos < <(printf '%s\n' "${sorted_combos[@]}" | sort -s -rn)
    printf '%s\n' "${sorted_combos[@]}"
}

_opt_archetype_batchub_score() {
    local b="$1" u="$2" opt_b="$3" opt_u="$4"
    if [[ "$b" == "$opt_b" && "$u" == "$opt_u" ]]; then
        echo 2
    elif [[ "$b" == "$opt_b" || "$u" == "$opt_u" ]]; then
        echo 1
    else
        echo 0
    fi
}

_opt_build_ctx_candidates() {
    # Generate a candidate list from the model's native context_length (capped
    # at 1M), merged with the standard context grid. Models can have
    # non-standard context lengths (e.g. 200k, 256k, 32k, 100k) that don't
    # align with the standard grid ({262144, 196608, 131072, 98304, 65536}).
    # By including both, we handle arbitrary context lengths while keeping
    # the standard grid points that models are tested against. Phase-1 tries
    # these largest-first and breaks on first fit (largest ctx wins).
    local max_ctx="$1"
    [[ $max_ctx -gt 1048576 ]] && max_ctx=1048576
    [[ $max_ctx -lt $MIN_CTX ]] && max_ctx=$MIN_CTX

    local grid=(262144 196608 131072 98304 65536)
    local candidates=()
    local seen=""
    for c in "$max_ctx" "${grid[@]}"; do
        [[ "$c" -le "$max_ctx" ]] && [[ "$c" -ge "$MIN_CTX" ]] || continue
        if [[ " $seen " != *" $c "* ]]; then
            seen+=" $c"
            candidates+=("$c")
        fi
    done
    [[ " $seen " != *" $MIN_CTX "* ]] && candidates+=("$MIN_CTX")

    local result=""
    for c in "${candidates[@]}"; do
        result="${result:+$result }$c"
    done
    echo "$result"
}

# -----------------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------------
solve_optimal_config() {
    local model_path="$1"
    MODEL_FILENAME=$(basename "$model_path")

    # Re-sync bias from OVERRIDE_BIAS (set by apply_user_overrides / CLI --bias
    # after this file was sourced). This ensures the bias flows through to
    # _opt_build_scored_combos (run in a mapfile subshell) and the phase-1/2
    # scoring adjustments.
    _OPT_BIAS="${OVERRIDE_BIAS:-${_OPT_BIAS:-context}}"

    # Reset detune step done-flags.
    _SOLVER_DONE_drop_draft=0
    _SOLVER_DONE_reduce_ngl=0
    _SOLVER_DONE_kv_q8_0=0
    _SOLVER_DONE_kv_q4_0=0

    gguf_read_meta "$model_path" || SOLVER_GGUF=()

    local n_layer=$(gguf_get block_count 32)
    SOLVER_N_LAYER="$n_layer"

    _opt_start_optimistic

    # Pre-compute PLE size once in the main shell so the scoring loop's
    # subshells see the cached value instead of spawning Python each iteration.
    if [[ "${is_qwen4exp:-false}" == "true" ]]; then
        _SOLVER_PLE_CACHE=$(_opt_qwen4exp_ple_bytes)
    fi

    local draft_path=""
    [[ -n "${prof_dspark:-}" ]] && draft_path="$prof_dspark"
    [[ -z "$draft_path" && -n "${prof_dflash:-}" ]] && draft_path="$prof_dflash"
    [[ -z "$draft_path" && -n "${prof_mtp:-}" ]] && draft_path="$prof_mtp"
    local draft_bytes=0
    if [[ -n "$draft_path" && -f "$draft_path" ]]; then
        draft_bytes=$(stat -c%s "$draft_path" 2>/dev/null || stat -f%z "$draft_path" 2>/dev/null || echo 0)
    fi

    local hckv=$(gguf_get head_count_kv 0)
    [[ $hckv -eq 0 ]] && hckv=$(gguf_get n_head_kv 0)
    [[ $hckv -eq 0 ]] && hckv=$(gguf_get head_count 0)
    [[ $hckv -eq 0 ]] && hckv=$(gguf_get n_head 0)
    if [[ $hckv -eq 0 ]]; then
        local size_gb=$(( ${MODEL_BYTES:-0} / _OPT_GIB ))
        if [[ $size_gb -gt 40 ]]; then hckv=8
        elif [[ $size_gb -gt 20 ]]; then hckv=4
        else hckv=2; fi
    fi

    local embd=$(gguf_get embedding_length 0)
    [[ $embd -eq 0 ]] && embd=$(gguf_get n_embd 0)
    local hc=$(gguf_get head_count 0)
    [[ $hc -eq 0 ]] && hc=$(gguf_get n_head 0)

    local kl=0 vl=0
    kl=$(gguf_get key_length 0)
    vl=$(gguf_get value_length 0)
    if [[ $kl -eq 0 || $vl -eq 0 ]]; then
        if [[ $embd -gt 0 && $hc -gt 0 ]]; then
            local head_dim=$(( embd / hc ))
            [[ $kl -eq 0 ]] && kl=$head_dim
            [[ $vl -eq 0 ]] && vl=$head_dim
        else
            local size_gb=$(( ${MODEL_BYTES:-0} / _OPT_GIB ))
            if [[ $size_gb -gt 40 ]]; then kl=128; vl=128
            else kl=256; vl=256; fi
        fi
    fi
    [[ $kl -eq 0 ]] && kl=256
    [[ $vl -eq 0 ]] && vl=256

    # NOTE: llama.cpp reads n_embd_head_k/n_embd_head_v directly from the GGUF
    # attention.key_length/attention.value_length metadata with NO additional
    # RoPE padding. The KV cache per layer per token is exactly:
    #   head_count_kv * (key_length * k_type_size + value_length * v_type_size)
    # We use kl/vl as-is. The old RoPE padding (adding hckv*16 to kl/vl) was
    # based on a misunderstanding — llama.cpp does NOT pad these dimensions.

    # swa_full=true always (llama.cpp default; no CLI flag), so ALL KV layers
    # allocate full-context KV. But for MoE models with full_attention_interval
    # > 1, only every Nth layer has attention/KV state — the expert-only layers
    # are "filtered" and don't allocate KV cache. llama.cpp logs this as
    # "N layers" in the KV cache init. Use n_kv_layer (not n_layer) for KV
    # sizing to avoid the 4.1x over-count that was forcing q4_0/cpu-moe.
    # (From manual-server.log: block_count=41, full_attention_interval=4,
    # n_kv_layer=40/4=10. KV = 2*(256+256)*1*10 = 10240 bytes/token @ q8_0.)
    local full_attn_interval=$(gguf_get full_attention_interval 1)
    local n_attn
    if [[ $full_attn_interval -gt 1 ]]; then
        # llama.cpp uses n_layer_all-1 (excludes the embedding/output block)
        # for the attention-layer count. block_count=41 -> 40 attn layers.
        local n_attn_layers=$(( n_layer - 1 ))
        [[ $n_attn_layers -le 0 ]] && n_attn_layers=$n_layer
        n_attn=$(( n_attn_layers / full_attn_interval ))
        [[ $n_attn -lt 1 ]] && n_attn=1
        SOLVER_REASONS+=("kv-layers: ${n_attn} (full_attention_interval=${full_attn_interval}, block_count=${n_layer})")
    else
        n_attn=$n_layer
    fi
    SOLVER_N_ATTN=$n_attn

    # Recurrent (SSM) cache for hybrid models: Qwen35MoE and similar models
    # have recurrent layers (gated delta net / linear attention) that maintain
    # a separate per-layer state cache in GPU memory. llama_memory_hybrid
    # allocates llama_memory_recurrent for these layers with F32 R and S
    # tensors, each having rs_size (=n_seq_max, usually 1) * (1+n_rs_seq)
    # rows. This is NOT part of the attention KV cache and was missing from
    # all prior solver estimates, contributing to the ~1 GiB gap between
    # estimated and actual VRAM usage.
    local recr_cache_bytes=0
    if [[ $full_attn_interval -gt 1 ]]; then
        local ssm_d_state=$(gguf_get ssm.state_size 0)
        local ssm_d_inner=$(gguf_get ssm.inner_size 0)
        local ssm_d_conv=$(gguf_get ssm.conv_kernel 0)
        local ssm_n_group=$(gguf_get ssm.group_count 0)
        if [[ $ssm_d_state -gt 0 && $ssm_d_inner -gt 0 ]]; then
            local n_recr_layers=$(( n_attn_layers - n_attn ))
            [[ $n_recr_layers -lt 0 ]] && n_recr_layers=0
            local n_embd_r=$(( (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state) ))
            local n_embd_s=$(( ssm_d_state * ssm_d_inner ))
            local n_rows=1  # rs_size=1, n_rs_seq=0 for single-sequence server
            recr_cache_bytes=$(( n_recr_layers * (n_embd_r + n_embd_s) * n_rows * 4 ))
        fi
    fi
    SOLVER_RECR_KV_BYTES=$recr_cache_bytes
    local solver_is_uma=0
    local solver_budget_bytes="${SOLVER_GPU_BUDGET_BYTES:-${GPU_BUDGET_BYTES:-0}}"
    if [[ $solver_budget_bytes -gt 0 ]] && _opt_is_uma; then
        solver_is_uma=1
    fi
    SOLVER_IS_UMA=$solver_is_uma
    [[ $solver_is_uma -eq 1 ]] && SOLVER_REASONS+=("UMA: integrated GPU detected (VRAM+GTT carve from system RAM)")

    # Checkpoint scaling: pre-fit with cap=16 so the fit-check accounts for
    # checkpoint hot-set RAM.
    _opt_scale_checkpoints "$SOLVER_CTX_SIZE" 16
    SOLVER_REASONS+=("checkpoints: ${SOLVER_CHECKPOINTS} (initial)")

    # Resolve load mode ONCE (single source of truth).
    SOLVER_LOAD_MODE="$(_opt_decide_load_mode)"
    SOLVER_REASONS+=("load-mode: ${SOLVER_LOAD_MODE}")

    SOLVER_DRAFT_PATH="$draft_path"

    # -------------------------------------------------------------------------
    # Bias application: let the user steer the solver toward a specific
    # dimension (context length, quality, speed, or MTP draft speed).
    # Defaults to "mtp" (optimistic-first: MTP draft on, q8_0 KV, large ctx).
    # -------------------------------------------------------------------------
    local bias="${_OPT_BIAS:-context}"
    case "$bias" in
        context)
            # Free MTP KV overhead (~1-2 GiB) for maximum context length.
            # Also reduce checkpoints to 1 (single-user, granular resume not needed).
            SOLVER_DRAFT_ENABLE=false
            SOLVER_CHECKPOINTS_OVERRIDE=1
            SOLVER_REASONS+=("bias=context: MTP draft disabled, ctx maximized, checkpoints=1")
            ;;
        checkpoints)
            # Maximize checkpoint granularity for session resume.
            # Keeps MTP draft off (to free memory for more/faster checkpoints).
            SOLVER_DRAFT_ENABLE=false
            SOLVER_REASONS+=("bias=checkpoints: MTP draft disabled, checkpoint count maximized")
            ;;
        quality)
            # Maximum KV quality (f16), no draft to preserve accuracy.
            SOLVER_DRAFT_ENABLE=false
            SOLVER_K_TYPE="f16"
            SOLVER_V_TYPE="f16"
            SOLVER_REASONS+=("bias=quality: f16 KV, draft disabled")
            ;;
        speed)
            # Smaller ctx, larger batch, no draft — decode throughput first.
            # Checkpoints to 1 (minimal overhead, max speed).
            SOLVER_DRAFT_ENABLE=false
            SOLVER_CTX_SIZE=$MIN_CTX
            SOLVER_CHECKPOINTS_OVERRIDE=1
            SOLVER_REASONS+=("bias=speed: ctx=${MIN_CTX}, draft disabled, checkpoints=1")
            ;;
        mtp|*)
            # Default: optimistic-first (MTP draft on, q8_0 KV, native ctx).
            # Default checkpoint count: 4 for hybrid/SSM models (Qwen3.6, etc.)
            # to provide multiple resume points across the prompt. With only 1
            # checkpoint saved at the end-of-prompt position, any prompt
            # discontinuity (e.g. dropped tool batches) drops LCP below the
            # checkpoint and forces n_past=0. 4 checkpoints at positions
            # spread across the prompt increase the probability that at least
            # one falls within the LCP prefix. The solver's memory budget
            # check confirms 4 checkpoints fit within the GPU budget for this
            # model's KV size.
            SOLVER_CHECKPOINTS_OVERRIDE=4
            SOLVER_REASONS+=("bias=mtp: optimistic-first (default)")
            ;;
    esac

    # -------------------------------------------------------------------------
    # Phase 1: optimistic-first scoring across all candidates.
    # -------------------------------------------------------------------------
    local sorted_combos=()
    mapfile -t sorted_combos < <(_opt_build_scored_combos "$n_layer" "$hckv" "$kl" "$vl" "${MODEL_BYTES:-0}")

    if [[ "${LLM_DEBUG_SOLVER:-0}" == "1" ]]; then
        log_info "Solver phase1 sorted combos (score:combo):"
        for s in "${sorted_combos[@]}"; do
            log_info "  $s"
        done
    fi

    local chosen_strategy="" chosen_ctx=0 chosen_kvq="" chosen_k_type="" chosen_v_type=""
    local chosen_draft_enable=true chosen_batch=0 chosen_ubatch=0 found=0

    for scored_combo in "${sorted_combos[@]}"; do
        local score="${scored_combo%%:*}" combo="${scored_combo#*:}"
        local strategy="${combo%%:*}" rest="${combo#*:}"
        local ctx="${rest%%:*}" rest="${rest#*:}"
        local kvq="${rest%%:*}" rest="${rest#*:}"
        local draft_mode="${rest%%:*}" rest="${rest#*:}"
        local batch="${rest%%:*}"
        local ubatch="${rest#*:}"

        # Skip impossible combos.
        [[ "$strategy" == "cpu" && $ctx -lt 8192 ]] && continue
        [[ $ctx -lt $MIN_CTX && "$strategy" != "cpu" ]] && continue

        local k_type="${kvq%%/*}" v_type="${kvq##*/}"

        local kv_per_token_per_layer
        kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$k_type" "$v_type" "$hckv" "$kl" "$vl")
        local kv_per_token
        kv_per_token=$(awk -v k="$kv_per_token_per_layer" -v n="$n_attn" 'BEGIN { printf "%.0f", k * n }')

        local eff_draft_bytes=0
        [[ "$draft_mode" == "enabled" && "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

        # Signal to _opt_gpu_memory whether draft is active for THIS combo,
        # so the MTP overhead (weight estimate + KV) is only counted when
        # draft_mode is "enabled" with SOLVER_DRAFT_ENABLE true — not for
        # "disabled" combos where SOLVER_DRAFT_ENABLE might still be true
        # (e.g. from _opt_start_optimistic before phase-1 resolution).
        _OPT_DRAFT_ACTIVE=0
        [[ "$draft_mode" == "enabled" && "$SOLVER_DRAFT_ENABLE" == "true" ]] && _OPT_DRAFT_ACTIVE=1

        local offloaded_bytes
        offloaded_bytes=$(_opt_model_gpu_footprint "$SOLVER_NGL" "$n_layer" "${MODEL_BYTES:-0}" "$strategy")

        # Compute checkpoint count for this combo's ctx so the fit-check
        # includes checkpoint KV storage. The previous code passed 0,
        # allowing configs that fit on paper but crash at runtime when
        # the checkpoint system allocates its ring buffer + copy buffer.
        # Respect SOLVER_CHECKPOINTS_OVERRIDE (set by bias) so the fit-check
        # uses the same checkpoint count as the final config, not an
        # over-scaled pre-fit count that inflates memory estimate.
        local _saved_ckpt="${SOLVER_CHECKPOINTS:-}" _saved_ckptmin="${SOLVER_CHECKPOINT_MIN:-}"
        SOLVER_CHECKPOINTS="" SOLVER_CHECKPOINT_MIN=""
        if [[ -n "${SOLVER_CHECKPOINTS_OVERRIDE:-}" ]]; then
            SOLVER_CHECKPOINTS="${SOLVER_CHECKPOINTS_OVERRIDE}"
            local _tmp_min=$(( ctx / SOLVER_CHECKPOINTS ))
            [[ $_tmp_min -lt 8192 ]] && _tmp_min=8192
            # Cap min_step at 32768 to match the post-fit cap — checkpoints
            # storing more than 32768 tokens waste GPU memory without
            # improving resume quality. Without this cap, phase-1 overestimates
            # checkpoint storage (e.g. 1.25 GiB instead of 320 MiB for 1 ckpt
            # at 128K ctx), causing the solver to reject 128K configs that
            # actually fit.
            [[ $_tmp_min -gt 32768 ]] && _tmp_min=32768
            SOLVER_CHECKPOINT_MIN=$_tmp_min
        else
            _opt_scale_checkpoints "$ctx" 8
        fi
        local combo_ckpt="${SOLVER_CHECKPOINTS:-0}"

        local mem_needed sys_mem_needed sys_budget fit_budget
        {
            read -r mem_needed
            read -r sys_mem_needed
            read -r sys_budget
            read -r fit_budget
        } < <(_opt_compute_memory_fit \
            "$kv_per_token" "$ctx" "$eff_draft_bytes" "$batch" "$ubatch" \
            "$offloaded_bytes" "$strategy" "$combo_ckpt")

        # Restore pre-fit checkpoint state.
        SOLVER_CHECKPOINTS="$_saved_ckpt" SOLVER_CHECKPOINT_MIN="$_saved_ckptmin"

        local gpu_ok=0 sys_ok=0
        if _opt_fits_budget "$mem_needed" "$sys_mem_needed" "$sys_budget" "$fit_budget"; then
            gpu_ok=1; sys_ok=1
        fi

        if [[ "${LLM_DEBUG_SOLVER:-0}" == "1" ]]; then
            local draft_str="off"
            [[ "$draft_mode" == "enabled" ]] && draft_str="on"
            local solver_budget_gib=$(( fit_budget / _OPT_GIB ))
            local mem_needed_gib=$(( mem_needed / _OPT_GIB ))
            log_info "Solver try: strategy=$strategy ctx=${ctx} KV=${kvq} batch=${batch} ubatch=${ubatch} draft=$draft_str GPU=${mem_needed_gib}GiB/${solver_budget_gib}GiB sys=$((sys_mem_needed/1048576))MiB/$((sys_budget/1048576))MiB -> gpu_ok=$gpu_ok sys_ok=$sys_ok"
        fi

        if [[ $gpu_ok -eq 1 && $sys_ok -eq 1 ]]; then
            local min_cache_mib
            min_cache_mib=$(_opt_min_cache_ram_mib "${MODEL_BYTES:-0}")
            # n_parallel=1 on non-hybrid models: no prompt cache, so no min-cache gate.
            [[ "${OVERRIDE_N_PARALLEL:-1}" -le 1 && "${is_ssm:-false}" != "true" ]] && min_cache_mib=0

            local leftover_mib=$(( (fit_budget - mem_needed) / 1048576 ))

            # When draft is enabled and leftover is tight, allow the combo if the
            # no-draft leftover is at least 50% of min_cache_mib (draft model is
            # reclaimable; KV headroom is not). Guard against negative leftover
            # from integer rounding when min_cache_mib==0.
            if [[ $fit_budget -gt 0 && $min_cache_mib -gt 0 && $leftover_mib -lt $min_cache_mib && "$draft_mode" == "enabled" ]]; then
                local no_draft_mem=$(( mem_needed - eff_draft_bytes ))
                local no_draft_leftover=$(( (fit_budget - no_draft_mem) / 1048576 ))
                local half_min=$(( min_cache_mib / 2 ))
                if [[ $no_draft_leftover -ge $half_min ]]; then
                    : # Allow.
                else
                    continue
                fi
            fi

            chosen_strategy="$strategy"
            chosen_ctx="$ctx" chosen_kvq="$kvq" chosen_k_type="$k_type" chosen_v_type="$v_type"
            chosen_batch=$batch chosen_ubatch=$ubatch
            [[ "$draft_mode" == "enabled" ]] && chosen_draft_enable=true || chosen_draft_enable=false
            found=1
            break
        fi
    done

    SOLVER_PROFILE_NAME=$(_opt_pick_profile)

    if [[ $found -eq 0 ]]; then
        # Phase 1 found nothing: degrade to the baseline (min ctx + q4_0/q4_0).
        # For MoE, prefer cpu strategy so only the on-GPU fraction + KV must
        # fit the GPU budget (NGL stays at max). Resolve load mode through the
        # single authority (_opt_decide_load_mode) rather than hardcoding.
        SOLVER_CTX_SIZE=$MIN_CTX
        SOLVER_K_TYPE="q4_0"
        SOLVER_V_TYPE="q4_0"
        if [[ "${is_moe:-false}" == "true" ]]; then
            SOLVER_MOE_STRATEGY="cpu"
        else
            SOLVER_MOE_STRATEGY="gpu"
        fi
        SOLVER_LOAD_MODE="$(_opt_decide_load_mode)"
        SOLVER_REASONS+=("phase-1: no fit; baseline ctx=${MIN_CTX} KV=q4_0/q4_0 strategy=${SOLVER_MOE_STRATEGY} load-mode=${SOLVER_LOAD_MODE}")
    else
        SOLVER_CTX_SIZE=$chosen_ctx
        SOLVER_K_TYPE=$chosen_k_type
        SOLVER_V_TYPE=$chosen_v_type
        # Re-evaluate batch/ubatch for the chosen archetype if the scoring loop
        # picked a different pair that fits the budget.
        if [[ $chosen_batch -gt 0 ]]; then SOLVER_BATCH=$chosen_batch; fi
        if [[ $chosen_ubatch -gt 0 ]]; then SOLVER_UBATCH=$chosen_ubatch; fi
        [[ $SOLVER_UBATCH -gt $SOLVER_BATCH ]] && SOLVER_UBATCH=$SOLVER_BATCH
        SOLVER_DRAFT_ENABLE=$chosen_draft_enable

        # Map chosen strategy to solver globals + load mode.
        # residency/gpu route through _opt_decide_load_mode (the single
        # authority, which avoids the unsafe "dio" on overcommit=0 kernels).
        # cpu (full MoE offload to system RAM) intentionally uses "none"
        # (eager pread): the cpu strategy is only selected when the model
        # fits in system RAM, where eager read is both faster and avoids the
        # lazy page-in contention that mmap would cause for a CPU-resident
        # workload with no GPU compute overlap.
        case "$chosen_strategy" in
            cpu)
                SOLVER_MOE_STRATEGY="cpu"
                SOLVER_LOAD_MODE="none"
                ;;
            residency)
                SOLVER_MOE_STRATEGY="residency"
                SOLVER_LOAD_MODE="$(_opt_decide_load_mode)"
                ;;
            *)
                SOLVER_MOE_STRATEGY="gpu"
                SOLVER_LOAD_MODE="$(_opt_decide_load_mode)"
                ;;
        esac

        local draft_str=""
        [[ "$chosen_draft_enable" == "true" ]] && draft_str=" draft=on" || draft_str=" draft=off"
        SOLVER_REASONS+=("phase-1 chosen: ctx=${chosen_ctx} KV=${chosen_kvq} strategy=${chosen_strategy} batch=${SOLVER_BATCH} ubatch=${SOLVER_UBATCH}${draft_str}")
    fi

    # -------------------------------------------------------------------------
    # Phase 2: fine-tune via detune steps if still over budget.
    # -------------------------------------------------------------------------
    local _phase2_fit=0
    local step_idx=0
    while [[ $step_idx -lt 50 ]]; do
        # Recompute kv_per_token inside the loop so KV downgrades take effect
        # on the memory estimate (Phase 2's KV downgrade must actually save RAM).
        local kv_per_token_per_layer
        kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
        local kv_per_token
        kv_per_token=$(awk -v k="$kv_per_token_per_layer" -v n="$n_attn" 'BEGIN { printf "%.0f", k * n }')

        local offloaded_bytes
        offloaded_bytes=$(_opt_model_gpu_footprint "$SOLVER_NGL" "$n_layer" "${MODEL_BYTES:-0}" "${SOLVER_MOE_STRATEGY:-gpu}")

        local eff_draft_bytes=0
        [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"
        _OPT_DRAFT_ACTIVE=0
        [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && _OPT_DRAFT_ACTIVE=1

        local mem_needed sys_mem_needed sys_budget fit_budget
        {
            read -r mem_needed
            read -r sys_mem_needed
            read -r sys_budget
            read -r fit_budget
        } < <(_opt_compute_memory_fit \
            "$kv_per_token" "$SOLVER_CTX_SIZE" "$eff_draft_bytes" "$SOLVER_BATCH" "$SOLVER_UBATCH" \
            "$offloaded_bytes" "${SOLVER_MOE_STRATEGY:-gpu}" 0)

        if [[ "${LLM_DEBUG_SOLVER:-0}" == "1" ]]; then
            local draft_str="off"
            [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && draft_str="on"
            local solver_budget_gib=$(( fit_budget / _OPT_GIB ))
            local mem_needed_gib=$(( mem_needed / _OPT_GIB ))
            log_info "Solver phase2 check: ctx=${SOLVER_CTX_SIZE} KV=${SOLVER_K_TYPE}/${SOLVER_V_TYPE} draft=$draft_str batch=${SOLVER_BATCH} ubatch=${SOLVER_UBATCH} strategy=${SOLVER_MOE_STRATEGY} load_mode=${SOLVER_LOAD_MODE} GPU=${mem_needed_gib}GiB/${solver_budget_gib} GiB sys=$((sys_mem_needed/1048576))MiB/$((sys_budget/1048576))MiB"
        fi

        if _opt_fits_budget "$mem_needed" "$sys_mem_needed" "$sys_budget" "$fit_budget"; then
            _phase2_fit=1
            break
        fi

        local applied=0 step_fn
        while IFS= read -r step_fn; do
            [[ -z "$step_fn" || "$step_fn" == \#* ]] && continue
            if "$step_fn" 2>/dev/null; then
                applied=1
                if [[ "${LLM_DEBUG_SOLVER:-0}" == "1" ]]; then
                    local draft_str="off"
                    [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && draft_str="on"
                    log_info "Solver detune applied: $step_fn -> ctx=${SOLVER_CTX_SIZE} KV=${SOLVER_K_TYPE}/${SOLVER_V_TYPE} draft=$draft_str ubatch=${SOLVER_UBATCH} batch=${SOLVER_BATCH}"
                fi
                break
            fi
        done < <(_opt_detune_steps)

        [[ $applied -eq 0 ]] && break
        step_idx=$(( step_idx + 1 ))
    done

    # Re-scale checkpoint config based on the FINAL ctx_size (phase 2 may have
    # detuned the context; cap widens from 16 to 32 post-fit).
    SOLVER_CHECKPOINTS=""
    SOLVER_CHECKPOINT_MIN=""

    # Final cache RAM derivation from the chosen config.
    local kv_per_token_per_layer
    kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
    local kv_per_token
    kv_per_token=$(awk -v k="$kv_per_token_per_layer" -v n="$n_attn" 'BEGIN { printf "%.0f", k * n }')

    local final_offloaded
    final_offloaded=$(_opt_model_gpu_footprint "$SOLVER_NGL" "$n_layer" "${MODEL_BYTES:-0}" "${SOLVER_MOE_STRATEGY:-gpu}")

    local final_draft_bytes=0
    [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && final_draft_bytes="$draft_bytes"
    _OPT_DRAFT_ACTIVE=0
    [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && _OPT_DRAFT_ACTIVE=1

    # If the bias requested a fixed checkpoint count, set it directly.
    if [[ -n "${SOLVER_CHECKPOINTS_OVERRIDE:-}" ]]; then
        SOLVER_CHECKPOINTS="$SOLVER_CHECKPOINTS_OVERRIDE"
        SOLVER_CHECKPOINT_MIN=$(( SOLVER_CTX_SIZE / SOLVER_CHECKPOINTS ))
        [[ $SOLVER_CHECKPOINT_MIN -lt 8192 ]] && SOLVER_CHECKPOINT_MIN=8192
        # Cap min_step at 32768 — checkpoint snapshots larger than 32768 tokens
        # waste GPU memory without improving resume quality (32k tokens = ~512
        # MiB at q8_0). This is a ceiling, not a floor: a smaller ctx or more
        # checkpoints legitimately yields a smaller min_step.
        [[ $SOLVER_CHECKPOINT_MIN -gt 32768 ]] && SOLVER_CHECKPOINT_MIN=32768
        SOLVER_REASONS+=("checkpoints: ${SOLVER_CHECKPOINTS} (override)")
    else
        _opt_scale_checkpoints "$SOLVER_CTX_SIZE" 32
    fi

    # Adaptive checkpoint reduction: the checkpoint system stores KV
    # snapshots in GPU VRAM and during checkpoint creation
    # (create_checkpoint -> state_seq_get_data) copies the active KV
    # window as a transient peak. If checkpoint storage + copy peak would
    # exceed the GPU budget, reduce the checkpoint count.
    local _solver_budget="${SOLVER_GPU_BUDGET_BYTES:-${GPU_BUDGET_BYTES:-0}}"
    if [[ $_solver_budget -gt 0 && ${SOLVER_CHECKPOINTS:-0} -gt 0 ]]; then
        local _ckpt_min_step=$(( SOLVER_CTX_SIZE / SOLVER_CHECKPOINTS ))
        [[ $_ckpt_min_step -lt 8192 ]] && _ckpt_min_step=8192
        [[ $_ckpt_min_step -gt 32768 ]] && _ckpt_min_step=32768
        local _ckpt_copy_peak=$(( _ckpt_min_step * kv_per_token * ${OVERRIDE_N_PARALLEL:-1} ))
        local _mem_with_ckpt
        _mem_with_ckpt=$(_opt_gpu_memory \
            "$final_offloaded" "$kv_per_token" "$SOLVER_CTX_SIZE" \
            "$final_draft_bytes" "$SOLVER_CTX_SIZE" "${OVERRIDE_N_PARALLEL:-1}" "$SOLVER_UBATCH" "${SOLVER_CHECKPOINTS:-0}")
        if [[ $(( _mem_with_ckpt + _ckpt_copy_peak )) -gt $_solver_budget ]]; then
            # Reduce checkpoint count until it fits. Allow reducing to 0
            # (disabling checkpoints entirely) instead of forcing a minimum
            # of 1 that would exceed budget and cause runtime OOM.
            while [[ ${SOLVER_CHECKPOINTS} -gt 0 ]]; do
                _ckpt_min_step=$(( SOLVER_CTX_SIZE / SOLVER_CHECKPOINTS ))
                [[ $_ckpt_min_step -lt 8192 ]] && _ckpt_min_step=8192
                [[ $_ckpt_min_step -gt 32768 ]] && _ckpt_min_step=32768
                _ckpt_copy_peak=$(( _ckpt_min_step * kv_per_token * ${OVERRIDE_N_PARALLEL:-1} ))
                local _ckpt_storage=$(( _ckpt_min_step * kv_per_token * SOLVER_CHECKPOINTS * ${OVERRIDE_N_PARALLEL:-1} ))
                local _mem_without_ckpt=$(( _mem_with_ckpt - _ckpt_storage ))
                [[ $(( _mem_without_ckpt + _ckpt_copy_peak )) -le $_solver_budget ]] && break
                _mem_with_ckpt=$(( _mem_without_ckpt ))
                SOLVER_CHECKPOINTS=$(( SOLVER_CHECKPOINTS - 1 ))
            done
            SOLVER_REASONS+=("checkpoints: ${SOLVER_CHECKPOINTS} (reduced for memory)")
        fi
        # Update SOLVER_CHECKPOINT_MIN to match final count. Guard against
        # division-by-zero when checkpoints were reduced to 0 (disabled).
        if [[ ${SOLVER_CHECKPOINTS:-0} -gt 0 ]]; then
            SOLVER_CHECKPOINT_MIN=$(( SOLVER_CTX_SIZE / SOLVER_CHECKPOINTS ))
            [[ $SOLVER_CHECKPOINT_MIN -lt 8192 ]] && SOLVER_CHECKPOINT_MIN=8192
            [[ $SOLVER_CHECKPOINT_MIN -gt 32768 ]] && SOLVER_CHECKPOINT_MIN=32768
        else
            SOLVER_CHECKPOINT_MIN=0
        fi
    fi

    # Cap SOLVER_CHECKPOINT_MIN at 32768 — checkpoint snapshots larger
    # than 32768 tokens waste GPU memory without improving resume quality.
    [[ ${SOLVER_CHECKPOINT_MIN:-0} -gt 32768 ]] && SOLVER_CHECKPOINT_MIN=32768

    SOLVER_REASONS+=("checkpoints: ${SOLVER_CHECKPOINTS} (rescaled)")
    SOLVER_REASONS+=("checkpoint-min-step: ${SOLVER_CHECKPOINT_MIN}")

    if [[ $found -eq 0 && $_phase2_fit -eq 0 ]]; then
        SOLVER_REASONS+=("ctx: ${MIN_CTX} KV: q4_0/q4_0 (no fit at any strategy/kv/ctx)")
    fi

    # ---------------------------------------------------------------------
    # Cache-RAM adequacy detune (DISABLED — layer offloading for cache-ram
    # is counterproductive: it sacrifices GPU compute layers for an arbitrary
    # prompt-cache floor that only matters if the user actually reuses prompts.
    # GPU layers should be prioritized over cache-ram.  The solver now leaves
    # cache-ram at whatever the phase-1/phase-2 fit produces (typically 200-800
    # MiB), which is plenty for most prompt-caching workloads.  Users who need
    # a larger prompt cache can set --cache-ram or OVERRIDE_CACHE_RAM manually.
    #
    # Previously this section offloaded compute layers to achieve a minimum
    # cache-ram target (e.g. 1024 MiB for SSM models on low-VRAM systems),
    # which degraded throughput by moving layers to CPU.  See issue #XXX.
    # ---------------------------------------------------------------------
    # DISABLED: Cache-RAM adequacy detune — see above.
    # if [[ -z "${OVERRIDE_CACHE_RAM:-}" ]]; then ...

    _opt_update_cache_ram \
        "${SOLVER_GPU_BUDGET_BYTES:-${GPU_BUDGET_BYTES:-0}}" \
        "$final_offloaded" "$kv_per_token" "$SOLVER_CTX_SIZE" \
        "$final_draft_bytes" "${OVERRIDE_N_PARALLEL:-1}" "$SOLVER_UBATCH"

    local kv_cache_bytes=$(( SOLVER_CTX_SIZE * kv_per_token ))
    SOLVER_KV_CACHE_MIB=$(( kv_cache_bytes / 1048576 + 1024 ))

    # Expose final intermediate values for the llama.cpp adapter's pre-flight memory
    # check. These are the exact figures the solver used to verify the fit,
    # so the pre-flight check sees the same numbers without re-deriving them.
    SOLVER_KV_PER_TOKEN="$kv_per_token"
    SOLVER_OFFLOADED_BYTES="$final_offloaded"
    SOLVER_DRAFT_BYTES="$final_draft_bytes"

    # Total system RAM impact of the final config. On UMA, VRAM+GTT == system
    # RAM, so the GPU-resident allocation (offloaded + KV + MTP + compute +
    # ckpt) is ALSO system RAM. On discrete, only CPU-resident portions
    # (PLE + cache-ram) are system RAM. The OS reserve (_OPT_OS_RESERVE_BYTES)
    # is NOT included: it is already subtracted in the GPU budget
    # (system_limit = vram + total_ram - os_reserve), so adding it here would
    # double-count it and cause the pre-flight check to reject valid configs.
    local _final_gpu_mem
    _final_gpu_mem=$(_opt_gpu_memory \
        "$final_offloaded" "$kv_per_token" "$SOLVER_CTX_SIZE" \
        "$final_draft_bytes" "$SOLVER_CTX_SIZE" "${OVERRIDE_N_PARALLEL:-1}" "$SOLVER_UBATCH" \
        "${SOLVER_CHECKPOINTS:-0}")
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        SOLVER_TOTAL_MEM_BYTES=$(( _final_gpu_mem + ${SOLVER_CACHE_RAM:-0} * 1048576 + $(_opt_qwen4exp_ple_bytes) ))
    else
        SOLVER_TOTAL_MEM_BYTES=$(( $(_opt_qwen4exp_ple_bytes) + ${SOLVER_CACHE_RAM:-0} * 1048576 ))
    fi
}

# -----------------------------------------------------------------------------
# User overrides (applied last, after solver output). Wins over everything.
# -----------------------------------------------------------------------------
apply_user_overrides() {
    SOLVER_OVERRIDES=()
    [[ -n "${LLM_CTX:-}" ]] && { SOLVER_CTX_SIZE="$LLM_CTX"; SOLVER_OVERRIDES+=("ctx-size"); }
    [[ -n "${LLM_KV_CACHE:-}" ]] && { SOLVER_K_TYPE="$LLM_KV_CACHE"; SOLVER_V_TYPE="$LLM_KV_CACHE"; SOLVER_OVERRIDES+=("kv-cache"); }
    [[ -n "${LLM_CACHE_TYPE_K:-}" ]] && { SOLVER_K_TYPE="$LLM_CACHE_TYPE_K"; SOLVER_OVERRIDES+=("cache-type-k"); }
    [[ -n "${LLM_CACHE_TYPE_V:-}" ]] && { SOLVER_V_TYPE="$LLM_CACHE_TYPE_V"; SOLVER_OVERRIDES+=("cache-type-v"); }

    if [[ -n "${LLM_THREADS:-}" ]]; then
        SOLVER_THREADS_BATCH="$LLM_THREADS"
        SOLVER_THREADS="$(( LLM_THREADS / 2 ))"
        [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1
        SOLVER_OVERRIDES+=("threads")
    elif [[ -n "${SYS_THREADS:-}" ]]; then
        SOLVER_THREADS_BATCH="$SYS_THREADS"
        SOLVER_THREADS="$(( SYS_THREADS / 2 ))"
        [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1
    fi

    [[ -n "${LLM_UBATCH:-}" ]] && { SOLVER_UBATCH="$LLM_UBATCH"; SOLVER_OVERRIDES+=("ubatch"); }
    [[ -n "${LLM_CACHE_RAM:-}" ]] && { SOLVER_CACHE_RAM="$LLM_CACHE_RAM"; SOLVER_OVERRIDES+=("cache-ram"); }
    if [[ -n "${LLM_NGL:-}" ]]; then
        SOLVER_NGL="$LLM_NGL"
        SOLVER_OVERRIDES+=("ngl")
    fi
    if [[ "${LLM_MOE_STRATEGY:-}" == "residency" ]]; then
        SOLVER_MOE_STRATEGY="residency"
        SOLVER_LOAD_MODE="$(_opt_decide_load_mode)"
        SOLVER_OVERRIDES+=("moe-strategy")
    fi
    [[ -n "${LLM_LOAD_MODE:-}" ]] && { SOLVER_LOAD_MODE="$LLM_LOAD_MODE"; SOLVER_OVERRIDES+=("load-mode"); }
    [[ "${LLM_FIT:-}" == "on" ]] && { SOLVER_NGL=-1; SOLVER_OVERRIDES+=("--fit on"); }
    [[ -n "${LLM_REASONING_BUDGET:-}" ]] && { SOLVER_REASONING_BUDGET="$LLM_REASONING_BUDGET"; SOLVER_OVERRIDES+=("reasoning-budget"); }

    # Recompute SOLVER_TOTAL_MEM_BYTES after overrides. User flags like
    # --ctx-size, --kv-cache-type, or --cache-ram can shrink the KV cache or
    # cache-ram, which makes the pre-flight check pass where the pre-override
    # value would have blocked a valid configuration. Conversely, --ctx-size
    # could also grow it. The recompute uses the live SOLVER_* globals.
    _recompute_total_mem
}

# Recomputes SOLVER_TOTAL_MEM_BYTES from current SOLVER_* state (post-override).
# Called by apply_user_overrides() and usable standalone.
_recompute_total_mem() {
    local _kv_per_token="${SOLVER_KV_PER_TOKEN:-0}"
    local _kv_cache_bytes=$(( ${SOLVER_CTX_SIZE:-0} * _kv_per_token ))
    # On UMA, VRAM+GTT == system RAM: GPU-resident allocation (offloaded + KV +
    # MTP + compute + ckpt) is also system RAM. On discrete, only CPU-resident
    # portions (PLE + cache-ram) matter. The OS reserve is NOT included — it
    # is already subtracted in the GPU budget (system_limit = vram + total_ram
    # - os_reserve), so adding it here double-counts it and causes the
    # pre-flight check to reject valid configs on tight UMA systems.
    if [[ "${SOLVER_IS_UMA:-0}" == "1" ]]; then
        SOLVER_TOTAL_MEM_BYTES=$(( ${SOLVER_OFFLOADED_BYTES:-0} + _kv_cache_bytes + ${SOLVER_CACHE_RAM:-0} * 1048576 + $(_opt_qwen4exp_ple_bytes) ))
    else
        SOLVER_TOTAL_MEM_BYTES=$(( $(_opt_qwen4exp_ple_bytes) + ${SOLVER_CACHE_RAM:-0} * 1048576 ))
    fi
}
