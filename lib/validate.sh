#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Validation for every user-supplied setting.
#
# Values arrive from three places: command-line flags, LLM_* environment
# variables, and the config file. All three end up in the same variable, so
# this runs once, after parsing, and rejects anything the engine cannot use.
# Without it a typo becomes a wrong build path or a server that fails at load
# time with an unrelated message.

[[ -n "${_LLM_VALIDATE_LOADED:-}" ]] && return 0
_LLM_VALIDATE_LOADED=1

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ggml cache types llama.cpp accepts for --cache-type-k/-v.
LLM_VALID_KV_TYPES="f32 f16 bf16 q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl iq4_xs q2_k q3_k q4_k q5_k q6_k q8_k"

_llm_errors=0

_invalid() {
    log_error "$1"
    _llm_errors=$(( _llm_errors + 1 ))
}

# validate_int <setting> <value> <min> <max>
validate_int() {
    local name="$1" value="$2" min="$3" max="$4"
    [[ -z "$value" ]] && return 0
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        _invalid "$name must be a whole number, got '$value'"
        return 1
    fi
    if (( value < min )); then
        _invalid "$name must be at least $min, got $value"
        return 1
    fi
    if [[ "$max" != "-" ]] && (( value > max )); then
        _invalid "$name must be at most $max, got $value"
        return 1
    fi
    return 0
}

# validate_enum <setting> <value> <allowed...>
validate_enum() {
    local name="$1" value="$2"
    shift 2
    [[ -z "$value" ]] && return 0
    local allowed
    for allowed in "$@"; do
        [[ "$value" == "$allowed" ]] && return 0
    done
    _invalid "$name must be one of: $*, got '$value'"
    return 1
}

# validate_settings [engine]
#   Rejects every setting the selected engine cannot use. Exits 2 with one
#   message per problem so a user fixes them in one pass.
validate_settings() {
    _llm_errors=0
    local engine="${1:-${LLM_ENGINE:-llama.cpp}}"

    case "$engine" in
        gufo) engine=gufo ;;
        *)    engine=llama.cpp ;;
    esac

    validate_enum "engine" "${LLM_ENGINE:-}" llama llama.cpp llamacpp gufo >/dev/null || true
    validate_enum "backend" "${LLM_BACKEND:-}" auto vulkan rocm metal cpu >/dev/null || true

    validate_int "context size" "${LLM_CTX:-}" 0 - >/dev/null || true
    validate_int "sessions" "${LLM_SESSIONS:-}" 1 1024 >/dev/null || true
    validate_int "port" "${LLM_PORT:-}" 1 65535 >/dev/null || true
    validate_int "threads" "${LLM_THREADS:-}" 1 4096 >/dev/null || true
    validate_int "GPU layers" "${LLM_NGL:-}" 0 1000 >/dev/null || true
    validate_int "ubatch size" "${LLM_UBATCH:-}" 1 65536 >/dev/null || true
    validate_int "draft tokens" "${LLM_DRAFT_TOKENS:-}" 1 64 >/dev/null || true
    validate_int "reasoning budget" "${LLM_REASONING_BUDGET:-}" 0 - >/dev/null || true

    if [[ -n "${LLM_CACHE_RAM:-}" && "${LLM_CACHE_RAM}" != "off" ]]; then
        validate_int "cache RAM (MiB)" "$LLM_CACHE_RAM" 0 - >/dev/null || true
    fi

    validate_enum "bias" "${LLM_BIAS:-}" mtp context quality speed >/dev/null || true
    validate_enum "speculative mode" "${LLM_SPEC:-}" auto off dflash2 mtp dspark >/dev/null || true
    validate_enum "sparse attention" "${LLM_SPARSE_ATTN:-}" auto on off >/dev/null || true
    validate_enum "reasoning" "${LLM_REASONING:-}" on off auto >/dev/null || true
    validate_enum "load mode" "${LLM_LOAD_MODE:-}" auto none mmap mlock dio >/dev/null || true
    validate_enum "fit" "${LLM_FIT:-}" on off >/dev/null || true

    if [[ "$engine" == "gufo" ]]; then
        # gufo rejects anything but f16/q8_0 at model load, per family.
        validate_enum "kv cache (gufo)" "${LLM_KV_CACHE:-}" f16 q8_0 >/dev/null || true
        if [[ -n "${LLM_CACHE_TYPE_K:-}${LLM_CACHE_TYPE_V:-}" ]]; then
            _invalid "llama.cpp cache type overrides do not apply to gufo; use --kv-cache"
        fi
        if [[ -n "${LLM_NGL:-}" ]]; then
            _invalid "GPU layers do not apply to gufo"
        fi
    else
        local t
        if [[ -n "${LLM_KV_CACHE:-}" ]]; then
            local ok=1
            for t in $LLM_VALID_KV_TYPES; do
                [[ "$LLM_KV_CACHE" == "$t" ]] && { ok=0; break; }
            done
            if (( ok )); then
                _invalid "kv cache type must be one of: $LLM_VALID_KV_TYPES, got '$LLM_KV_CACHE'"
            fi
        fi
        if [[ -n "${LLM_CACHE_TYPE_K:-}" ]]; then
            local ok=1
            for t in $LLM_VALID_KV_TYPES; do
                [[ "$LLM_CACHE_TYPE_K" == "$t" ]] && { ok=0; break; }
            done
            (( ok )) && _invalid "cache-type-k must be one of: $LLM_VALID_KV_TYPES, got '$LLM_CACHE_TYPE_K'"
        fi
        if [[ -n "${LLM_CACHE_TYPE_V:-}" ]]; then
            local ok=1
            for t in $LLM_VALID_KV_TYPES; do
                [[ "$LLM_CACHE_TYPE_V" == "$t" ]] && { ok=0; break; }
            done
            (( ok )) && _invalid "cache-type-v must be one of: $LLM_VALID_KV_TYPES, got '$LLM_CACHE_TYPE_V'"
        fi
        if [[ -n "${LLM_SPARSE_ATTN:-}" && "${LLM_SPARSE_ATTN}" != "auto" && "${LLM_SPARSE_ATTN}" != "off" ]]; then
            _invalid "sparse attention is a gufo Laguna feature; llama.cpp has no switch for it"
        fi
    fi

    if (( _llm_errors > 0 )); then
        log_error "$_llm_errors setting(s) rejected"
        return 2
    fi
    return 0
}
