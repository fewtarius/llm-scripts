#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# GGUF introspection shared by both solvers.

[[ -n "${_LLM_GGUF_LOADED:-}" ]] && return 0
_LLM_GGUF_LOADED=1

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

_LLM_GGUF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF_META="${LLM_GGUF_META:-${_LLM_GGUF_DIR}/tools/gguf_meta.py}"
GGUF_PLE="${LLM_GGUF_PLE:-${_LLM_GGUF_DIR}/tools/gguf_ple.py}"
GGUF_VALIDATE="${LLM_GGUF_VALIDATE:-${_LLM_GGUF_DIR}/tools/gguf_validate.py}"

declare -A SOLVER_GGUF=()

# Fill SOLVER_GGUF with numeric metadata, indexed by both the full key and
# its last dotted segment.
gguf_read_meta() {
    local path="$1"
    SOLVER_GGUF=()
    [[ -f "$path" && -r "$path" ]] || return 1
    local pairs
    pairs=$(python3 - "$GGUF_META" "$path" <<'PY'
import json, subprocess, sys
try:
    out = subprocess.check_output(["python3", sys.argv[1], sys.argv[2]],
                                  stderr=subprocess.DEVNULL, text=True, timeout=60)
    d = json.loads(out)
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            print(f"{k}={v}")
            print(f"{k.rsplit('.', 1)[-1]}={v}")
except Exception:
    sys.exit(1)
PY
    ) || return 1
    [[ -z "$pairs" ]] && return 1
    local key val
    while IFS='=' read -r key val; do
        [[ -n "$key" ]] && SOLVER_GGUF["$key"]="$val"
    done <<< "$pairs"
    return 0
}

gguf_get() {
    local key="$1" default="${2:-0}"
    local val="${SOLVER_GGUF[$key]:-$default}"
    [[ "$val" =~ ^-?[0-9]+$ ]] && printf '%s' "$val" || printf '%s' "$default"
}

gguf_string() {
    local key="$1" default="${2:-}"
    printf '%s' "${SOLVER_GGUF[$key]:-$default}"
}

gguf_arch() {
    local model="$1"
    python3 "$GGUF_META" "$model" 2>/dev/null \
        | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("general.architecture", ""))
except Exception:
    pass' 2>/dev/null || true
}

gguf_ple_bytes() {
    local model="$1"
    python3 "$GGUF_PLE" "$model" 2>/dev/null || echo 0
}

gguf_validate_drafter() {
    local family="$1" file="$2"
    python3 "$GGUF_VALIDATE" --family "$family" --file "$file" 2>&1
}
