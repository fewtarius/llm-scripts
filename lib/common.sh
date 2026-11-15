#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Paths, config precedence, and small helpers shared by every entry point.

[[ -n "${_LLM_COMMON_LOADED:-}" ]] && return 0
_LLM_COMMON_LOADED=1

_LLM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="${LLM_ROOT:-$(cd "$_LLM_LIB_DIR/.." && pwd)}"
export LLM_ROOT

# shellcheck source=lib/log.sh
source "$_LLM_LIB_DIR/log.sh"

# Model directory. The repository default is ./models; set LLM_MODEL_DIR to
# point at a shared model store.
LLM_MODEL_DIR="${LLM_MODEL_DIR:-$LLM_ROOT/models}"
[[ -d "$LLM_MODEL_DIR" ]] && LLM_MODEL_DIR=$(readlink -f "$LLM_MODEL_DIR")
export LLM_MODEL_DIR

# Where solver profiles are pinned. See lib/profile.sh.
LLM_PROFILE_DIR="${LLM_PROFILE_DIR:-$LLM_ROOT/profiles}"

# User config. Environment variables win over the file; command-line flags
# win over both. Only LLM_* keys are accepted.
LLM_CONFIG="${LLM_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/llm-scripts/config.env}"

LLM_STATE_DIR="${LLM_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/llm-scripts}"
LLM_CACHE_DIR="${LLM_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/llm-scripts}"

llm_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

llm_falsey() { ! llm_truthy "${1:-}"; }

llm_die() { log_error "$1"; exit "${2:-1}"; }

llm_have() { command -v "$1" >/dev/null 2>&1; }

llm_require() {
    local cmd
    for cmd in "$@"; do
        llm_have "$cmd" || llm_die "Required command not found: $cmd"
    done
}

llm_trim() {
    local s="${1-}"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Load KEY=value pairs from LLM_CONFIG into the environment. Comments and
# blank lines are ignored. A variable already set in the environment is not
# overwritten, so the precedence is env > file. Values are assigned without
# eval, so metacharacters in a value are data.
llm_load_config() {
    local file="${1:-$LLM_CONFIG}"
    [[ -f "$file" ]] || return 0
    local line key val
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(llm_trim "$line")
        [[ -z "$line" || "$line" == \#* ]] && continue
        [[ "$line" =~ ^(LLM_[A-Za-z0-9_]+)=(.*)$ ]] || continue
        key="${BASH_REMATCH[1]}"
        val=$(llm_trim "${BASH_REMATCH[2]}")
        if [[ ${#val} -ge 2 && ( ( "${val:0:1}" == '"' && "${val: -1}" == '"' ) || ( "${val:0:1}" == "'" && "${val: -1}" == "'" ) ) ]]; then
            val="${val:1:${#val}-2}"
        fi
        [[ -n "${!key:-}" ]] && continue
        printf -v "$key" '%s' "$val"
        export "$key"
    done < "$file"
}

llm_state_dir() { mkdir -p "$LLM_STATE_DIR" 2>/dev/null || true; printf '%s' "$LLM_STATE_DIR"; }
llm_cache_dir() { mkdir -p "$LLM_CACHE_DIR" 2>/dev/null || true; printf '%s' "$LLM_CACHE_DIR"; }

llm_load_config
