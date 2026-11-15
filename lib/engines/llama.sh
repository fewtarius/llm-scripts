#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# llama.cpp engine adapter.
#
# Owns every llama.cpp spelling: binary layout, server flags, speculative
# mode names, checkpoint probing. Nothing outside this file should need to
# know that llama-server takes --cache-type-k or that --fit is a flag.

[[ -n "${_LLM_ENGINE_LLAMA_LOADED:-}" ]] && return 0
_LLM_ENGINE_LLAMA_LOADED=1

_LLAMA_ENG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$_LLAMA_ENG_DIR/common.sh"
# shellcheck source=lib/profile.sh
source "$_LLAMA_ENG_DIR/profile.sh"
# shellcheck source=lib/models.sh
source "$_LLAMA_ENG_DIR/models.sh"
# shellcheck source=lib/gguf.sh
source "$_LLAMA_ENG_DIR/gguf.sh"
# shellcheck source=lib/env.sh
source "$_LLAMA_ENG_DIR/env.sh"
# shellcheck source=solver/llama.sh
source "$_LLAMA_ENG_DIR/../solver/llama.sh"

LLM_BUILD_DIR="${LLM_BUILD_DIR:-$LLM_ROOT/build}"
LLM_KV_CACHE_DIR="${LLM_KV_CACHE_DIR:-$LLM_ROOT/kv-cache}"

engine_name_llama() { printf 'llama.cpp'; }

llama_resolve_backend() {
    local requested="${LLM_BACKEND:-auto}"
    if [[ "$requested" != "auto" ]]; then
        printf '%s' "$requested"
        return 0
    fi
    if [[ "$(uname -s)" == "Darwin" && -x "$LLM_BUILD_DIR/llama-metal/bin/llama-server" ]]; then
        printf 'metal'
    elif [[ -x "$LLM_BUILD_DIR/llama-vulkan/bin/llama-server" ]]; then
        printf 'vulkan'
    elif [[ -x "$LLM_BUILD_DIR/llama-rocm/bin/llama-server" ]]; then
        printf 'rocm'
    else
        printf 'cpu'
    fi
}

llama_binary() {
    local cmd="${1:-server}"
    printf '%s/llama-%s/bin/llama-%s' "$LLM_BUILD_DIR" "$PROFILE_BACKEND" "$cmd"
}

engine_env_llama() {
    # A CPU-only build still runs with the Vulkan environment; it only adds
    # library paths and driver tuning that the CPU path ignores.
    case "$PROFILE_BACKEND" in
        rocm)   llm_env_activate rocm ;;
        metal)  llm_env_activate metal ;;
        *)      llm_env_activate vulkan ;;
    esac
}



engine_ready_regex_llama() { printf 'server is listening|listening on http'; }

_llama_scan_arch() {
    local model="$1" key
    is_moe=false; is_ssm=false; is_mtp=false
    gguf_read_meta "$model" || SOLVER_GGUF=()
    for key in "${!SOLVER_GGUF[@]}"; do
        case "$key" in
            *expert_count*|*expert_used_count*) is_moe=true ;;
            ssm.*|qwen3next*) is_ssm=true ;;
            *full_attention_interval*) is_ssm=true ;;
            *nextn_predict_layers*) is_mtp=true ;;
        esac
    done
    if (( $(gguf_get nextn_predict_layers 0) > 0 )); then is_mtp=true; fi
}

# GPU-visible pool = VRAM carveout + GTT, less a staging reserve. On UMA the
# pool is also capped by what the OS can actually give up.
_llama_gpu_budget_bytes() { gpu_budget_bytes; }

engine_solve_llama() {
    local model="$1"
    PROFILE_ENGINE="llama.cpp"
    PROFILE_MODEL="$model"
    PROFILE_MODEL_NAME="$(basename "$model" .gguf)"
    PROFILE_MODEL_BYTES=$(llm_model_bytes "$model")
    PROFILE_BACKEND=$(llama_resolve_backend)
    PROFILE_HOST="${LLM_HOST:-127.0.0.1}"
    PROFILE_PORT="${LLM_PORT:-9090}"
    PROFILE_SESSIONS="${LLM_SESSIONS:-1}"
    PROFILE_BIAS="${LLM_BIAS:-context}"
    PROFILE_SPARSE_ATTN="n/a"
    if [[ -n "${LLM_SPARSE_ATTN:-}" && "${LLM_SPARSE_ATTN}" != "auto" && "${LLM_SPARSE_ATTN}" != "off" ]]; then
        log_warn "llama.cpp has no sparse-attention switch; ignoring --sparse-attn ${LLM_SPARSE_ATTN}"
    fi

    MODEL_BYTES="$PROFILE_MODEL_BYTES"
    MODEL_PATH="$model"
    export MODEL_PATH
    _llama_scan_arch "$model"
    is_qwen4exp=false

    prof_dspark=$(llm_find_sidecar "$model" dspark || true)
    prof_dflash=$(llm_find_sidecar "$model" dflash || true)
    prof_mtp=$(llm_find_sidecar "$model" mtp || true)
    [[ -n "$prof_mtp" ]] && is_mtp=true

    GPU_BUDGET_BYTES=$(_llama_gpu_budget_bytes)
    SOLVER_GPU_BUDGET_BYTES="$GPU_BUDGET_BYTES"
    OVERRIDE_N_PARALLEL="$PROFILE_SESSIONS"
    export GPU_BUDGET_BYTES SOLVER_GPU_BUDGET_BYTES OVERRIDE_N_PARALLEL

    solve_optimal_config "$model"
    apply_user_overrides

    if [[ "${SOLVER_REASONS[*]}" == *"no fit at any strategy/kv/ctx"* ]]; then
        log_error "No configuration fits the memory budget."
        log_error "GPU budget: $((GPU_BUDGET_BYTES / 1073741824)) GiB, model: $((PROFILE_MODEL_BYTES / 1073741824)) GiB."
        return 1
    fi

    # Speculative policy. The solver computes whether a draft is affordable;
    # the request can force a mode or turn it off.
    local spec="${LLM_SPEC:-auto}" draft_kind="" draft_model=""
    case "$spec" in
        off) SOLVER_DRAFT_ENABLE=false ;;
        dspark)
            draft_kind=dspark; draft_model="$prof_dspark"
            [[ -n "$draft_model" ]] || { log_warn "No DSpark drafter found; speculative disabled"; SOLVER_DRAFT_ENABLE=false; }
            ;;
        dflash2)
            draft_kind=dflash; draft_model="$prof_dflash"
            [[ -n "$draft_model" ]] || { log_warn "No DFlash drafter found; speculative disabled"; SOLVER_DRAFT_ENABLE=false; }
            ;;
        mtp)
            draft_kind=mtp; draft_model="$prof_mtp"
            [[ -z "$draft_model" && "$is_mtp" != "true" ]] && { log_warn "No MTP head found; speculative disabled"; SOLVER_DRAFT_ENABLE=false; }
            ;;
    esac
    if [[ "$spec" == "auto" && "${SOLVER_DRAFT_ENABLE:-false}" == "true" ]]; then
        if [[ -n "$prof_dspark" ]]; then draft_kind=dspark; draft_model="$prof_dspark"
        elif [[ "$is_mtp" == "true" ]]; then draft_kind=mtp; draft_model="$prof_mtp"
        elif [[ -n "$prof_dflash" ]]; then draft_kind=dflash; draft_model="$prof_dflash"
        fi
    fi
    if [[ "${SOLVER_DRAFT_ENABLE:-false}" != "true" ]]; then
        draft_kind=""; draft_model=""
    fi

    PROFILE_SPEC="${draft_kind:-off}"
    case "$draft_kind" in
        dspark)  PROFILE_SPEC="dspark";  PROFILE_DRAFT_MODEL="$draft_model"
                 PROFILE_DRAFT_TOKENS="${LLM_DRAFT_TOKENS:-${LLM_SPEC_DRAFT_N_MAX_DSPARK:-3}}" ;;
        mtp)     PROFILE_SPEC="mtp";     PROFILE_DRAFT_MODEL="$draft_model"
                 PROFILE_DRAFT_TOKENS="${LLM_DRAFT_TOKENS:-${LLM_SPEC_DRAFT_N_MAX_MTP:-2}}" ;;
        dflash)  PROFILE_SPEC="dflash2"; PROFILE_DRAFT_MODEL="$draft_model"
                 PROFILE_DRAFT_TOKENS="${LLM_DRAFT_TOKENS:-${LLM_SPEC_DRAFT_N_MAX_DFLASH:-7}}" ;;
    esac
    SOLVER_DRAFT_KIND="$draft_kind"
    SOLVER_DRAFT_MODEL="$draft_model"

    PROFILE_CTX="$SOLVER_CTX_SIZE"
    PROFILE_KV_CACHE="${SOLVER_K_TYPE}/${SOLVER_V_TYPE}"
    PROFILE_FAMILY="${SOLVER_PROFILE_NAME:-dense}"
    PROFILE_ARCH=$(gguf_arch "$model")

    local r
    for r in "${SOLVER_REASONS[@]}"; do profile_reason "$r"; done
    profile_reason "load-mode: ${SOLVER_LOAD_MODE:-auto}"
    profile_reason "batch/ubatch: ${SOLVER_BATCH}/${SOLVER_UBATCH}"
    return 0
}

engine_render_llama() {
    local model="$PROFILE_MODEL"
    PROFILE_ARGS=()
    local server
    server=$(llama_binary server)
    if [[ ! -x "$server" ]]; then
        [[ "${LLM_DRY_RUN:-0}" == "1" ]] || { log_error "llama-server not built at $server"; log_info "Build it with: ./llm build --engine llama.cpp"; return 1; }
    fi

    local ctx="${PROFILE_CTX:-0}"
    (( ctx > 0 )) || ctx=65536

    local ngl="${LLM_NGL:-99}"
    if [[ "${SOLVER_NGL:-0}" -gt 0 && "${SOLVER_NGL}" -lt "${SOLVER_N_LAYER:-$SOLVER_NGL}" && "$ngl" -gt "$SOLVER_NGL" ]]; then
        log_info "Capping -ngl from $ngl to $SOLVER_NGL (solver detuned to fit)"
        ngl="$SOLVER_NGL"
    fi
    [[ -n "${SOLVER_NGL:-}" && "$SOLVER_NGL" == "-1" ]] && ngl=-1

    PROFILE_ARGS+=(-m "$model")
    [[ -n "${LLM_ALIAS:-}" ]] && PROFILE_ARGS+=(-a "$LLM_ALIAS")
    PROFILE_ARGS+=(-c "$ctx")
    PROFILE_ARGS+=(--threads "${SOLVER_THREADS}" --threads-batch "${SOLVER_THREADS_BATCH}")
    PROFILE_ARGS+=(--batch-size "${SOLVER_BATCH}" --ubatch-size "${SOLVER_UBATCH}")
    PROFILE_ARGS+=(-ngl "$ngl")
    PROFILE_ARGS+=(--cache-type-k "${SOLVER_K_TYPE}" --cache-type-v "${SOLVER_V_TYPE}")
    PROFILE_ARGS+=(--load-mode "${SOLVER_LOAD_MODE:-auto}")
    PROFILE_ARGS+=(--flash-attn "${LLM_FLASH_ATTN:-auto}")
    PROFILE_ARGS+=(--lazy-mode "${LLM_LAZY_MODE:-auto}")
    [[ -z "${LLM_MMPROJ:-}" ]] && PROFILE_ARGS+=(--no-mmproj)
    [[ -n "${LLM_MMPROJ:-}" ]] && PROFILE_ARGS+=(--mmproj "$LLM_MMPROJ")

    PROFILE_ARGS+=(--temp "${LLM_TEMP:-1.0}" --top-p "${LLM_TOP_P:-0.95}" --top-k "${LLM_TOP_K:-20}" --min-p "${LLM_MIN_P:-0.00}")
    PROFILE_ARGS+=(--repeat-penalty "${LLM_REPEAT_PENALTY:-1.0}" --presence-penalty "${LLM_PRESENCE_PENALTY:-0.0}")

    if [[ "${is_moe:-false}" == "true" ]]; then
        case "${SOLVER_MOE_STRATEGY:-gpu}" in
            cpu) PROFILE_ARGS+=(--cpu-moe) ;;
            residency) PROFILE_ARGS+=(--n-cpu-moe "$(( SOLVER_N_LAYER * 70 / 100 ))") ;;
        esac
    fi
    [[ "${is_qwen4exp:-false}" == "true" ]] && PROFILE_ARGS+=(-ot "per_layer_token_embd.weight=CPU")

    if [[ -n "$PROFILE_DRAFT_MODEL" && "$PROFILE_DRAFT_MODEL" != "embedded" ]]; then
        PROFILE_ARGS+=(-md "$PROFILE_DRAFT_MODEL")
    fi
    case "$PROFILE_SPEC" in
        dspark)
            PROFILE_ARGS+=(--spec-type draft-dspark --spec-draft-n-max "${PROFILE_DRAFT_TOKENS}" --spec-draft-p-min "${LLM_SPEC_DRAFT_P_MIN_DSPARK:-0.0}" --fit off)
            ;;
        mtp)
            PROFILE_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "${PROFILE_DRAFT_TOKENS}" --spec-draft-p-min "${LLM_SPEC_DRAFT_P_MIN_MTP:-0.0}" --fit off)
            ;;
        dflash2)
            PROFILE_ARGS+=(--spec-type draft-dflash --spec-draft-n-max "${PROFILE_DRAFT_TOKENS}" --spec-draft-p-min "${LLM_SPEC_DRAFT_P_MIN_DFLASH:-0.6}" --fit off)
            ;;
    esac

    local ckpt_min="${SOLVER_CHECKPOINT_MIN:-32768}"
    local ckpt_count="${SOLVER_CHECKPOINTS:-8}"
    local ckpt_every="${SOLVER_CHECKPOINT_EVERY_N_TOKENS:--1}"
    PROFILE_ARGS+=(--checkpoint-min-step "$ckpt_min" --ctx-checkpoints "$ckpt_count")
    if "$server" --help 2>/dev/null | grep -q -- '--no-checkpoint-near-end'; then
        PROFILE_ARGS+=(--no-checkpoint-near-end)
    fi
    if [[ "$ckpt_every" != "-1" ]] && "$server" --help 2>/dev/null | grep -q -- '--checkpoint-every-n-tokens'; then
        PROFILE_ARGS+=(--checkpoint-every-n-tokens "$ckpt_every")
    fi

    local cache_ram="${SOLVER_CACHE_RAM:-0}"
    if (( cache_ram > 0 )); then
        PROFILE_ARGS+=(--cache-ram "$cache_ram")
    elif [[ "${PROFILE_SESSIONS:-1}" -le 1 && -z "${LLM_CACHE_RAM:-}" && "${is_ssm:-false}" != "true" ]]; then
        PROFILE_ARGS+=(--cache-ram 0)
    fi

    PROFILE_ARGS+=(--host "$PROFILE_HOST" --port "$PROFILE_PORT" --jinja --metrics)
    PROFILE_ARGS+=(--reasoning "${LLM_REASONING:-on}")
    [[ -n "${SOLVER_REASONING_BUDGET:-}" && "${SOLVER_REASONING_BUDGET}" != "0" ]] && PROFILE_ARGS+=(--reasoning-budget "$SOLVER_REASONING_BUDGET")
    PROFILE_ARGS+=(-np "$PROFILE_SESSIONS" --prio "${LLM_PRIO:-3}" --prio-batch "${LLM_PRIO_BATCH:-3}")
    PROFILE_ARGS+=(-ctxcp "${LLM_CTXCP:-64}" --cache-reuse "${LLM_CACHE_REUSE:-512}")
    mkdir -p "$LLM_KV_CACHE_DIR"
    PROFILE_ARGS+=(--slot-save-path "$LLM_KV_CACHE_DIR")
    PROFILE_ARGS+=(--slot-prompt-similarity "${LLM_SLOT_PROMPT_SIMILARITY:-0.20}" --kv-unified)
    [[ "${LLM_CONTEXT_SHIFT:-1}" == "0" ]] && PROFILE_ARGS+=(--no-context-shift)
    if [[ "${LLM_PRESERVE_REASONING:-1}" != "0" ]]; then
        PROFILE_ARGS+=(--reasoning-preserve --chat-template-kwargs '{"preserve_thinking":true}')
    fi
    [[ "${LLM_FIT:-}" == "on" ]] && PROFILE_ARGS+=(--fit on)

    if [[ -n "${LLM_EXTRA_ARGS:-}" ]]; then
        local extra=()
        read -r -a extra <<< "$LLM_EXTRA_ARGS"
        PROFILE_ARGS+=("${extra[@]}")
    fi
    return 0
}

engine_argv_llama() { printf '%s\n' "$(llama_binary server)" "${PROFILE_ARGS[@]}"; }
