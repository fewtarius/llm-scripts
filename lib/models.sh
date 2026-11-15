#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Model discovery and naming.
#
# One rule set for every engine: a model is a first shard or a single file;
# drafters, projectors and later shards are sidecars, not models.

[[ -n "${_LLM_MODELS_LOADED:-}" ]] && return 0
_LLM_MODELS_LOADED=1

_LLM_MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_LLM_MODELS_DIR/common.sh"

llm_is_shard() { [[ "$(basename "$1")" =~ ^.*-[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; }
llm_is_first_shard() { [[ "$(basename "$1")" =~ ^.*-00001-of-[0-9]{5}\.gguf$ ]]; }
llm_is_later_shard() { llm_is_shard "$1" && ! llm_is_first_shard "$1"; }

# Drafters, projectors and adapter sidecars are not standalone models.
llm_is_sidecar() {
    local base
    base=$(basename "$1" | tr '[:upper:]' '[:lower:]')
    case "$base" in
        *dflash*|*dspark*|mtp-*|*-mtp*|mmproj*|*mmproj*) return 0 ;;
    esac
    return 1
}

llm_model_files() {
    local first="$1" base dir f
    if ! llm_is_shard "$first"; then
        printf '%s\n' "$first"
        return 0
    fi
    dir=$(dirname "$first")
    base=$(basename "$first")
    base="${base%-[0-9][0-9][0-9][0-9][0-9]-of-*.gguf}"
    for f in "$dir/$base"-[0-9][0-9][0-9][0-9][0-9]-of-*.gguf; do
        [[ -f "$f" ]] && printf '%s\n' "$f"
    done | sort
}

llm_model_bytes() {
    local total=0 f size
    while IFS= read -r f; do
        size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f" 2>/dev/null || echo 0)
        total=$(( total + size ))
    done < <(llm_model_files "$1")
    printf '%s' "$total"
}

# Strip the shard, quant and -UD segments from a filename.
llm_model_base_name() {
    local base
    base=$(basename "$1")
    base="${base%.gguf}"
    if [[ "$base" =~ ^(.*)-[0-9]{5}-of-[0-9]{5}$ ]]; then
        base="${BASH_REMATCH[1]}"
    fi
    base="${base%-*}"
    [[ "$base" == *-UD ]] && base="${base%-UD}"
    printf '%s' "$base"
}

# First shard or single file for every model in the model directory.
llm_model_entries() {
    local dir="${1:-$LLM_MODEL_DIR}" f
    [[ -d "$dir" ]] || return 0
    while IFS= read -r f; do
        llm_is_later_shard "$f" && continue
        llm_is_sidecar "$f" && continue
        printf '%s\n' "$f"
    done < <(find "$dir" -maxdepth 1 -name '*.gguf' -type f 2>/dev/null | sort)
}

# Resolve a path, an exact basename, or a unique substring to a model path.
llm_resolve_model() {
    local query="$1" dir="${2:-$LLM_MODEL_DIR}"
    if [[ -f "$query" ]]; then
        realpath "$query"
        return 0
    fi
    local f base matches=() entries=()
    mapfile -t entries < <(llm_model_entries "$dir")
    for f in "${entries[@]}"; do
        base=$(basename "$f")
        if [[ "$base" == "$query" || "$base" == "$query.gguf" || "$base" == "${query}-00001-of-"* ]]; then
            realpath "$f"; return 0
        fi
    done
    for f in "${entries[@]}"; do
        base=$(basename "$f")
        [[ "${base,,}" == *"${query,,}"* ]] && matches+=("$f")
    done
    case ${#matches[@]} in
        0)
            log_error "No model matching '$query' in $dir"
            return 1
            ;;
        1) realpath "${matches[0]}" ;;
        *)
            log_warn "Multiple models match '$query':"
            local m
            for m in "${matches[@]}"; do log_warn "  $(basename "$m")"; done
            log_info "Using: $(basename "${matches[0]}")"
            realpath "${matches[0]}"
            ;;
    esac
}

# Find a drafter beside the model. Suffix (Base-DFlash-Quant) and prefix
# (mtp-Base-Quant) layouts are accepted. A metadata-patched -gufo.gguf copy
# wins when present.
llm_find_sidecar() {
    local model="$1" pattern="$2"
    shift 2
    local dirs=("$@")
    [[ ${#dirs[@]} -eq 0 ]] && dirs=("$LLM_MODEL_DIR" "$LLM_MODEL_DIR/disable")
    local base="${model##*/}"
    local base_lc
    base=$(llm_model_base_name "$model")
    base_lc="${base,,}"
    [[ -z "$base_lc" ]] && return 1

    local dir f fb files=()
    for dir in "${dirs[@]}"; do
        [[ -d "$dir" ]] || continue
        mapfile -t files < <(find "$dir" -maxdepth 1 -iname "*${pattern}*.gguf" 2>/dev/null | sort)
        for f in "${files[@]}"; do
            fb=$(basename "$f")
            {
                [[ "${fb,,}" =~ ${base_lc}.*${pattern}.*\.gguf$ ]] ||
                [[ "${fb,,}" =~ ^${pattern}-${base_lc}.*\.gguf$ ]]
            } || continue
            [[ "${fb,,}" == *-gufo.gguf ]] && { printf '%s\n' "$f"; return 0; }
        done
    done
    for dir in "${dirs[@]}"; do
        [[ -d "$dir" ]] || continue
        mapfile -t files < <(find "$dir" -maxdepth 1 -iname "*${pattern}*.gguf" 2>/dev/null | sort)
        for f in "${files[@]}"; do
            fb=$(basename "$f")
            {
                [[ "${fb,,}" =~ ${base_lc}.*${pattern}.*\.gguf$ ]] ||
                [[ "${fb,,}" =~ ^${pattern}-${base_lc}.*\.gguf$ ]]
            } || continue
            printf '%s\n' "$f"
            return 0
        done
    done
    return 1
}

llm_list_models() {
    local f size_gib entries
    printf '\n%s\n\n' "${CYAN}Models in $LLM_MODEL_DIR${NC}"
    if [[ ! -d "$LLM_MODEL_DIR" ]]; then
        log_warn "Model directory does not exist: $LLM_MODEL_DIR"
        log_warn "Set LLM_MODEL_DIR, or create it and add a GGUF file."
        printf '\n'
        return 1
    fi
    if [[ ! -r "$LLM_MODEL_DIR" ]]; then
        log_error "Model directory is not readable: $LLM_MODEL_DIR"
        return 1
    fi
    entries=$(llm_model_entries)
    if [[ -z "$entries" ]]; then
        printf '  (none)\n\n'
        return 0
    fi
    while IFS= read -r f; do
        size_gib=$(llm_model_bytes "$f")
        printf '  %-68s %6.1f GiB\n' "$(basename "$f")" "$(awk -v b="$size_gib" 'BEGIN{print b/1073741824}')"
    done < <(llm_model_entries)
    printf '\n'
}
