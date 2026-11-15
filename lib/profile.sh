#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Profile representation and emission.
#
# A profile is the engine-neutral description of how to run one model. Engine
# adapters translate it into argv and environment; nothing outside an adapter
# needs to know an engine's flag spelling.

[[ -n "${_LLM_PROFILE_LOADED:-}" ]] && return 0
_LLM_PROFILE_LOADED=1

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

profile_reset() {
    PROFILE_ENGINE=""
    PROFILE_MODEL=""
    PROFILE_MODEL_NAME=""
    PROFILE_MODEL_BYTES=0
    PROFILE_FAMILY=""
    PROFILE_ARCH=""
    PROFILE_BACKEND=""
    PROFILE_CTX=0
    PROFILE_SESSIONS=1
    PROFILE_KV_CACHE=""
    PROFILE_SPEC="off"
    PROFILE_DRAFT_MODEL=""
    PROFILE_DRAFT_TOKENS=""
    PROFILE_SPARSE_ATTN="auto"
    PROFILE_CACHE_RAM=""
    PROFILE_CACHE_DISK=""
    PROFILE_BIAS="context"
    PROFILE_HOST="127.0.0.1"
    PROFILE_PORT=9090
    PROFILE_ARGS=()
    PROFILE_ENV=()
    PROFILE_REASONS=()
}

profile_reason() { PROFILE_REASONS+=("$1"); }

_json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

_json_str() { printf '"%s":"%s"' "$1" "$(_json_escape "$2")"; }
_json_num() { printf '"%s":%s' "$1" "${2:-0}"; }

_json_array() {
    local name="$1"; shift
    local first=1 item
    printf '"%s":[' "$name"
    for item in "$@"; do
        (( first )) || printf ','
        first=0
        printf '"%s"' "$(_json_escape "$item")"
    done
    printf ']'
}

profile_emit_json() {
    local sep=""
    printf '{'
    _json_str engine "$PROFILE_ENGINE"; printf ','
    _json_str model "$PROFILE_MODEL"; printf ','
    _json_str model_name "$PROFILE_MODEL_NAME"; printf ','
    _json_num model_bytes "$PROFILE_MODEL_BYTES"; printf ','
    _json_str family "$PROFILE_FAMILY"; printf ','
    _json_str arch "$PROFILE_ARCH"; printf ','
    _json_str backend "$PROFILE_BACKEND"; printf ','
    _json_num context "$PROFILE_CTX"; printf ','
    _json_num sessions "$PROFILE_SESSIONS"; printf ','
    _json_str kv_cache "$PROFILE_KV_CACHE"; printf ','
    _json_str speculative "$PROFILE_SPEC"; printf ','
    _json_str draft_model "$PROFILE_DRAFT_MODEL"; printf ','
    _json_str draft_tokens "$PROFILE_DRAFT_TOKENS"; printf ','
    _json_str sparse_attn "$PROFILE_SPARSE_ATTN"; printf ','
    _json_str cache_ram "$PROFILE_CACHE_RAM"; printf ','
    _json_str cache_disk "$PROFILE_CACHE_DISK"; printf ','
    _json_str bias "$PROFILE_BIAS"; printf ','
    _json_str host "$PROFILE_HOST"; printf ','
    _json_num port "$PROFILE_PORT"; printf ','
    _json_array args "${PROFILE_ARGS[@]}"; printf ','
    _json_array env "${PROFILE_ENV[@]}"; printf ','
    _json_array reasons "${PROFILE_REASONS[@]}"
    printf '}\n'
}

profile_emit_text() {
    printf '%-22s %s\n' "Engine:" "$PROFILE_ENGINE"
    printf '%-22s %s\n' "Model:" "$PROFILE_MODEL_NAME"
    printf '%-22s %s\n' "Family:" "${PROFILE_FAMILY:-unknown}"
    printf '%-22s %s\n' "Backend:" "$PROFILE_BACKEND"
    printf '%-22s %s\n' "Context:" "$( (( PROFILE_CTX > 0 )) && printf '%s' "$PROFILE_CTX" || printf 'native' )"
    printf '%-22s %s\n' "Sessions:" "$PROFILE_SESSIONS"
    printf '%-22s %s\n' "KV cache:" "${PROFILE_KV_CACHE:-engine default}"
    printf '%-22s %s\n' "Speculative:" "$PROFILE_SPEC"
    [[ -n "$PROFILE_DRAFT_MODEL" ]] && printf '%-22s %s\n' "Draft model:" "$(basename "$PROFILE_DRAFT_MODEL")"
    [[ -n "$PROFILE_DRAFT_TOKENS" ]] && printf '%-22s %s\n' "Draft tokens:" "$PROFILE_DRAFT_TOKENS"
    printf '%-22s %s\n' "Sparse attention:" "$PROFILE_SPARSE_ATTN"
    [[ -n "$PROFILE_CACHE_RAM" ]] && printf '%-22s %s\n' "Cache RAM:" "$PROFILE_CACHE_RAM"
    [[ -n "$PROFILE_CACHE_DISK" ]] && printf '%-22s %s\n' "Cache disk:" "$PROFILE_CACHE_DISK"
    printf '%-22s %s\n' "Bias:" "$PROFILE_BIAS"
    printf '%-22s %s\n' "Listen:" "$PROFILE_HOST:$PROFILE_PORT"
    if [[ ${#PROFILE_REASONS[@]} -gt 0 ]]; then
        echo ""
        echo "Reasoning:"
        local r
        for r in "${PROFILE_REASONS[@]}"; do printf '  - %s\n' "$r"; done
    fi
}

# ---------------------------------------------------------------------------
# Per-model pins
# ---------------------------------------------------------------------------
# A pin records a resolved profile so a known-good configuration is reused.
# Pins live below config, environment, and command-line flags: llm_load_pin
# only assigns a key that is still unset.
llm_pin_path() {
    local model="$1" base
    base=$(basename "$model" .gguf)
    printf '%s/%s.conf' "$LLM_PROFILE_DIR" "$base"
}

llm_load_pin() {
    local model="$1" file key val
    file=$(llm_pin_path "$model")
    [[ -f "$file" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(llm_trim "$line")
        [[ -z "$line" || "$line" == \#* ]] && continue
        [[ "$line" =~ ^(LLM_[A-Za-z0-9_]+)=(.*)$ ]] || continue
        key="${BASH_REMATCH[1]}"
        val=$(llm_trim "${BASH_REMATCH[2]}")
        [[ -n "${!key:-}" ]] && continue
        printf -v "$key" '%s' "$val"
        export "$key"
    done < "$file"
    log_info "Pinned profile: $file"
    return 0
}

llm_save_pin() {
    local model="$1" file
    file=$(llm_pin_path "$model")
    if ! mkdir -p "$LLM_PROFILE_DIR" 2>/dev/null; then
        log_error "Cannot create the profile directory: $LLM_PROFILE_DIR"
        log_error "Set LLM_PROFILE_DIR to a writable path."
        return 1
    fi
    {
        echo "# Generated by: llm profile --save"
        echo "# Model: $(basename "$model")"
        echo "LLM_ENGINE=$PROFILE_ENGINE"
        [[ "$PROFILE_BACKEND" != "cpu" ]] && echo "LLM_BACKEND=$PROFILE_BACKEND"
        (( PROFILE_CTX > 0 )) && echo "LLM_CTX=$PROFILE_CTX"
        echo "LLM_SESSIONS=$PROFILE_SESSIONS"
        if [[ "$PROFILE_KV_CACHE" == */* ]]; then
            echo "LLM_CACHE_TYPE_K=${PROFILE_KV_CACHE%%/*}"
            echo "LLM_CACHE_TYPE_V=${PROFILE_KV_CACHE##*/}"
        else
            echo "LLM_KV_CACHE=$PROFILE_KV_CACHE"
        fi
        echo "LLM_SPEC=$PROFILE_SPEC"
        [[ -n "$PROFILE_DRAFT_TOKENS" ]] && echo "LLM_DRAFT_TOKENS=$PROFILE_DRAFT_TOKENS"
        echo "LLM_BIAS=$PROFILE_BIAS"
        [[ "$PROFILE_SPARSE_ATTN" != "n/a" ]] && echo "LLM_SPARSE_ATTN=$PROFILE_SPARSE_ATTN"
        [[ -n "$PROFILE_CACHE_RAM" ]] && echo "LLM_CACHE_RAM=$PROFILE_CACHE_RAM"
        [[ -n "$PROFILE_CACHE_DISK" ]] && echo "LLM_CACHE_DISK=$PROFILE_CACHE_DISK"
    } > "$file" 2>/dev/null || {
        log_error "Cannot write the profile: $file"
        return 1
    }
    log_ok "Pinned profile: $file"
}
