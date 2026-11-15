#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# gufo engine adapter.
#
# gufo exposes some performance knobs through family environment variables
# (LAGUNA_KV_CACHE_TYPE, QWEN_KV_CACHE_TYPE, LAGUNA_SPARSE_ATTN) and the rest
# through serve flags. This adapter is the only place that knows either.

[[ -n "${_LLM_ENGINE_GUFO_LOADED:-}" ]] && return 0
_LLM_ENGINE_GUFO_LOADED=1

_GUFO_ENG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$_GUFO_ENG_DIR/common.sh"
# shellcheck source=lib/profile.sh
source "$_GUFO_ENG_DIR/profile.sh"
# shellcheck source=lib/models.sh
source "$_GUFO_ENG_DIR/models.sh"
# shellcheck source=lib/gguf.sh
source "$_GUFO_ENG_DIR/gguf.sh"
# shellcheck source=lib/env.sh
source "$_GUFO_ENG_DIR/env.sh"
# shellcheck source=solver/gufo.sh
source "$_GUFO_ENG_DIR/../solver/gufo.sh"

LLM_BUILD_DIR="${LLM_BUILD_DIR:-$LLM_ROOT/build}"
LLM_KV_CACHE_DIR="${LLM_KV_CACHE_DIR:-$LLM_ROOT/kv-cache}"

engine_name_gufo() { printf 'gufo'; }

gufo_binary() { printf '%s/gufo/gufo' "$LLM_BUILD_DIR"; }

engine_env_gufo() {
    case "$SOLVER_GUFO_FAMILY" in
        laguna)
            if [[ "$PROFILE_KV_CACHE" == "q8_0" ]]; then
                export LAGUNA_KV_CACHE_TYPE=q8_0
                PROFILE_ENV+=("LAGUNA_KV_CACHE_TYPE=q8_0")
            else
                unset LAGUNA_KV_CACHE_TYPE
            fi
            if [[ "$PROFILE_SPARSE_ATTN" == "on" ]]; then
                export LAGUNA_SPARSE_ATTN=1
                PROFILE_ENV+=("LAGUNA_SPARSE_ATTN=1")
            else
                unset LAGUNA_SPARSE_ATTN
            fi
            if [[ -n "${LLM_PREFILL_TILE:-}" ]]; then
                export LAGUNA_PREFILL_TILE="$LLM_PREFILL_TILE"
                PROFILE_ENV+=("LAGUNA_PREFILL_TILE=$LLM_PREFILL_TILE")
            fi
            ;;
        qwen|qwen-flash-next)
            if [[ "$PROFILE_KV_CACHE" == "q8_0" ]]; then
                export QWEN_KV_CACHE_TYPE=q8_0
                PROFILE_ENV+=("QWEN_KV_CACHE_TYPE=q8_0")
            else
                unset QWEN_KV_CACHE_TYPE
            fi
            ;;
        *)
            if [[ -n "$PROFILE_KV_CACHE" && "$PROFILE_KV_CACHE" != "f16" ]]; then
                log_warn "gufo has no KV-cache switch for family '$SOLVER_GUFO_FAMILY'; using its default"
            fi
            ;;
    esac
    # gufo links the HIP runtime from the bundled SDK on every backend.
    llm_env_activate rocm
}



engine_ready_regex_gufo() { printf 'event=listening'; }

engine_solve_gufo() {
    local model="$1"
    PROFILE_ENGINE="gufo"
    PROFILE_MODEL="$model"
    PROFILE_MODEL_NAME="$(basename "$model" .gguf)"
    PROFILE_MODEL_BYTES=$(llm_model_bytes "$model")
    PROFILE_BACKEND="rocm"
    PROFILE_HOST="${LLM_HOST:-127.0.0.1}"
    PROFILE_PORT="${LLM_PORT:-9090}"
    PROFILE_SESSIONS="${LLM_SESSIONS:-1}"
    PROFILE_BIAS="${LLM_BIAS:-context}"

    solve_gufo_config "$model"

    PROFILE_FAMILY="$SOLVER_GUFO_FAMILY"
    PROFILE_ARCH=$(gguf_arch "$model")
    PROFILE_CTX="$SOLVER_GUFO_CTX"
    PROFILE_SESSIONS="$SOLVER_GUFO_SESSIONS"
    PROFILE_KV_CACHE="$SOLVER_GUFO_KV_CACHE"
    PROFILE_SPEC="$SOLVER_GUFO_SPEC"
    PROFILE_DRAFT_MODEL="$SOLVER_GUFO_DRAFT_MODEL"
    PROFILE_DRAFT_TOKENS="$SOLVER_GUFO_DRAFT_TOKENS"
    PROFILE_SPARSE_ATTN="$SOLVER_GUFO_SPARSE"
    PROFILE_CACHE_RAM="${LLM_CACHE_RAM:-}"
    PROFILE_CACHE_DISK="${LLM_CACHE_DISK:-}"
    if [[ "$PROFILE_FAMILY" != "laguna" && "${LLM_SPARSE_ATTN:-auto}" != "auto" ]]; then
        log_warn "sparse attention is a Laguna feature; ignoring --sparse-attn for family $PROFILE_FAMILY"
    fi

    local r
    for r in "${SOLVER_GUFO_REASONS[@]}"; do profile_reason "$r"; done
    return 0
}

engine_render_gufo() {
    PROFILE_ARGS=()
    local bin
    bin=$(gufo_binary)
    if [[ ! -x "$bin" ]]; then
        [[ "${LLM_DRY_RUN:-0}" == "1" ]] || { log_error "gufo not built at $bin"; log_info "Build it with: ./llm build --engine gufo"; return 1; }
    fi

    PROFILE_ARGS+=(serve llm --model "$PROFILE_MODEL")
    (( PROFILE_CTX > 0 )) && PROFILE_ARGS+=(--context "$PROFILE_CTX")
    PROFILE_ARGS+=(--sessions "$PROFILE_SESSIONS")

    case "$PROFILE_SPEC" in
        dflash2)
            PROFILE_ARGS+=(--speculative dflash2 --dflash-model "$PROFILE_DRAFT_MODEL" -d "$PROFILE_DRAFT_TOKENS")
            ;;
        dspark)
            PROFILE_ARGS+=(--speculative dspark --dspark-model "$PROFILE_DRAFT_MODEL" -d "$PROFILE_DRAFT_TOKENS")
            ;;
        mtp)
            PROFILE_ARGS+=(--speculative mtp --mtp-model "${PROFILE_DRAFT_MODEL:-embedded}" -d "$PROFILE_DRAFT_TOKENS")
            ;;
    esac

    if [[ -n "$PROFILE_CACHE_RAM" ]]; then
        if [[ "$PROFILE_CACHE_RAM" == "0" || "$PROFILE_CACHE_RAM" == "off" ]]; then
            PROFILE_ARGS+=(--cache-ram-bytes off)
        elif [[ "$PROFILE_CACHE_RAM" =~ ^[0-9]+$ ]]; then
            PROFILE_ARGS+=(--cache-ram-bytes "$(( PROFILE_CACHE_RAM * 1048576 ))")
        else
            PROFILE_ARGS+=(--cache-ram-bytes "$PROFILE_CACHE_RAM")
        fi
    fi
    [[ -n "$PROFILE_CACHE_DISK" ]] && PROFILE_ARGS+=(--cache-disk "$PROFILE_CACHE_DISK")

    PROFILE_ARGS+=(--host "$PROFILE_HOST" --port "$PROFILE_PORT")
    [[ "${LLM_LOG_PROGRESS:-0}" == "1" ]] && PROFILE_ARGS+=(--log-progress)

    if [[ -n "${LLM_EXTRA_ARGS:-}" ]]; then
        local extra=()
        read -r -a extra <<< "$LLM_EXTRA_ARGS"
        PROFILE_ARGS+=("${extra[@]}")
    fi
    return 0
}

engine_argv_gufo() { printf '%s\n' "$(gufo_binary)" "${PROFILE_ARGS[@]}"; }
