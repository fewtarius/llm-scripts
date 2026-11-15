#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Solver regression harness
# =============================================================================
# Runs solver/llama.sh over a set of fixed model/hardware profiles and
# compares the resolved SOLVER_* configuration against recorded golden files.
#
# The solver is the most regression-prone component in this repo: most of its
# history is "fix(solver)" commits for memory-accounting errors that produced
# either DeviceLost crashes or silently detuned profiles. Each case here is
# derived from one of those failure modes, so reintroducing it fails a test
# instead of a server launch.
#
# Cases are hermetic. the GGUF metadata and PLE readers are stubbed from
# per-case fixtures, and host probes are pinned through the
# SOLVER_IS_UMA_OVERRIDE, SOLVER_TOTAL_MEMORY_BYTES,
# SOLVER_AVAILABLE_MEMORY_BYTES and PHYSICAL_CORES inputs, so results do not
# depend on the machine running them.
#
# Usage:
#   ./tests/solver-golden.sh                 # run all cases
#   ./tests/solver-golden.sh --case mla      # run cases matching a substring
#   ./tests/solver-golden.sh --update        # rewrite goldens from current code
#   ./tests/solver-golden.sh --verbose       # show full diff on failure
#   ./tests/solver-golden.sh --list
#
# Exit status is non-zero if any case fails.
# =============================================================================

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
CASES_DIR="$TESTS_DIR/solver-cases"
GOLDEN_DIR="$TESTS_DIR/golden"

# shellcheck source=lib/log.sh
source "$PROJECT_ROOT/lib/log.sh"

UPDATE=0
VERBOSE=0
FILTER=""
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)  UPDATE=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --case)    FILTER="${2:-}"; shift 2 ;;
        --list)    LIST_ONLY=1; shift ;;
        -h|--help) sed -n '5,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) log_error "Unknown argument: $1"; exit 2 ;;
    esac
done

# Solver outputs compared against the golden file. Order matters: it defines
# the golden format. Deliberately excluded are values that are informational
# only or that embed absolute paths.
OUTPUT_VARS=(
    SOLVER_PROFILE_NAME
    SOLVER_CTX_SIZE
    SOLVER_K_TYPE
    SOLVER_V_TYPE
    SOLVER_NGL
    SOLVER_UBATCH
    SOLVER_LOAD_MODE
    SOLVER_MOE_STRATEGY
    SOLVER_DRAFT_ENABLE
    SOLVER_DRAFT_N_MAX
    SOLVER_N_LAYER
    SOLVER_N_ATTN
    SOLVER_THREADS
    SOLVER_THREADS_BATCH
    SOLVER_KV_CACHE_MIB
    SOLVER_RECR_KV_BYTES
    SOLVER_CHECKPOINT_MIN
    SOLVER_CHECKPOINTS
    SOLVER_CHECKPOINT_EVERY_N_TOKENS
    SOLVER_CACHE_RAM
    SOLVER_REASONING_BUDGET
    SOLVER_IS_UMA
    OVERRIDE_BATCH_SIZE
    EXTRA_SERVER_ARGS
)

# Emit the stub GGUF reader. It prints the JSON fixture named by the case.
write_stub_reader() {
    local dest="$1"
    cat > "$dest" <<'STUB'
#!/usr/bin/env python3
"""GGUF metadata stub for the solver regression harness.

The real reader parses a GGUF file's KV section. Tests supply the metadata
directly through SOLVER_TEST_GGUF_JSON so cases stay small and readable.
"""
import json
import os
import sys

payload = os.environ.get("SOLVER_TEST_GGUF_JSON", "{}")
try:
    data = json.loads(payload)
except json.JSONDecodeError as exc:
    sys.stderr.write("stub reader: bad fixture JSON: %s\n" % exc)
    sys.exit(1)
json.dump(data, sys.stdout)
STUB
    chmod +x "$dest"
}

write_stub_ple() {
    local dest="$1"
    cat > "$dest" <<'STUB'
#!/usr/bin/env python3
"""PLE size stub for the solver regression harness."""
import os
import sys

sys.stdout.write(os.environ.get("SOLVER_TEST_PLE_BYTES", "0"))
STUB
    chmod +x "$dest"
}

# run_case <case-file> <sandbox>
# Sources the case, runs the solver, prints the snapshot to stdout.
run_case() {
    local case_file="$1" sandbox="$2"

    # Case defaults; individual cases override what they care about.
    local CASE_NAME CASE_DESC
    local SOLVER_TEST_GGUF_JSON='{}'
    local SOLVER_TEST_PLE_BYTES=0
    local MODEL_BYTES=0
    local SOLVER_GPU_BUDGET_BYTES=0
    local SOLVER_TOTAL_MEMORY_BYTES=0
    local SOLVER_AVAILABLE_MEMORY_BYTES=0
    local SOLVER_IS_UMA_OVERRIDE=0
    local SYS_TIER=halo
    local BACKEND=vulkan
    local PHYSICAL_CORES=16
    local LLM_BIAS=mtp
    local is_moe=false is_ssm=false is_mtp=false is_dflash=false is_dspark=false
    local is_qwen4exp=false
    local OVERRIDE_N_PARALLEL=1
    local SOLVER_LOAD_MODE=auto
    local prof_mtp="" prof_dflash="" prof_dspark=""
    local LLM_CACHE_TYPE_K="" LLM_CACHE_TYPE_V=""
    local LLM_UBATCH="" LLM_CTX=""
    local LLM_NGL=99 LLM_CACHE_RAM="" LLM_THREADS=""
    local LLM_FIT=""
    local SOLVER_REASONS=()

    # shellcheck disable=SC1090
    source "$case_file"

    CASE_NAME="$(basename "$case_file" .case)"

    # Bias: default to mtp (optimistic-first). A case can override by setting
    # LLM_BIAS. Exported so solve_optimal_config's
    # re-sync sees the case's value rather than inheriting the runner's env.
    LLM_BIAS="${LLM_BIAS:-mtp}"
    OVERRIDE_BIAS="$LLM_BIAS"
    _OPT_BIAS="$LLM_BIAS"

    export SOLVER_TEST_GGUF_JSON SOLVER_TEST_PLE_BYTES
    export LLM_BIAS OVERRIDE_BIAS _OPT_BIAS

    # LLM_GGUF_META / LLM_GGUF_PLE point at the sandbox stubs

    local MODEL_PATH="$sandbox/models/${CASE_NAME}.gguf"
    : > "$MODEL_PATH"

    # shellcheck disable=SC1091
    LLM_GGUF_META="$sandbox/scripts/gguf_meta.py"
    LLM_GGUF_PLE="$sandbox/scripts/gguf_ple.py"
    export LLM_GGUF_META LLM_GGUF_PLE
    source "$TESTS_DIR/../solver/llama.sh"

    MODEL_BYTES="$MODEL_BYTES"
    SOLVER_GPU_BUDGET_BYTES="$SOLVER_GPU_BUDGET_BYTES"
    GPU_BUDGET_BYTES="$SOLVER_GPU_BUDGET_BYTES"
    export SOLVER_TOTAL_MEMORY_BYTES SOLVER_AVAILABLE_MEMORY_BYTES
    export SOLVER_IS_UMA_OVERRIDE
    export SYS_TIER BACKEND PHYSICAL_CORES

    solve_optimal_config "$MODEL_PATH" >/dev/null 2>&1

    local name
    for name in "${OUTPUT_VARS[@]}"; do
        printf '%s=%s\n' "$name" "${!name:-}"
    done
}

# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
if [[ ! -d "$CASES_DIR" ]]; then
    log_error "No cases directory: $CASES_DIR"
    exit 1
fi

mapfile -t ALL_CASES < <(find "$CASES_DIR" -maxdepth 1 -name '*.case' | sort)
if [[ ${#ALL_CASES[@]} -eq 0 ]]; then
    log_error "No .case files in $CASES_DIR"
    exit 1
fi

if [[ $LIST_ONLY -eq 1 ]]; then
    for c in "${ALL_CASES[@]}"; do
        printf '  %s\n' "$(basename "$c" .case)"
    done
    exit 0
fi

CASES=()
for c in "${ALL_CASES[@]}"; do
    n="$(basename "$c" .case)"
    if [[ -z "$FILTER" || "$n" == *"$FILTER"* ]]; then
        CASES+=("$c")
    fi
done
if [[ ${#CASES[@]} -eq 0 ]]; then
    log_error "No cases matched filter: $FILTER"
    exit 1
fi

SANDBOX="$(mktemp -d)"
cleanup() { rm -rf "$SANDBOX"; }
trap cleanup EXIT
mkdir -p "$SANDBOX/scripts" "$SANDBOX/models"
write_stub_reader "$SANDBOX/scripts/gguf_meta.py"
write_stub_ple "$SANDBOX/scripts/gguf_ple.py"

mkdir -p "$GOLDEN_DIR"
pass=0 fail=0 added=0 drift=0

log_info "Solver regression: ${#CASES[@]} case(s)"
echo ""

for case_file in "${CASES[@]}"; do
    name="$(basename "$case_file" .case)"
    golden="$GOLDEN_DIR/$name.txt"

    actual="$(run_case "$case_file" "$SANDBOX" 2>/dev/null)"
    rc=$?

    if [[ -z "$actual" || $rc -ne 0 ]]; then
        log_error "  $name: solver produced no output (rc=$rc)"
        fail=$((fail + 1))
        continue
    fi

    if [[ $UPDATE -eq 1 ]]; then
        printf '%s\n' "$actual" > "$golden"
        log_ok "  $name: golden updated"
        added=$((added + 1))
        continue
    fi

    if [[ ! -f "$golden" ]]; then
        log_warn "  $name: no golden file (run --update to create)"
        drift=$((drift + 1))
        continue
    fi

    if diff -q "$golden" <(printf '%s\n' "$actual") >/dev/null 2>&1; then
        log_ok "  $name"
        pass=$((pass + 1))
    else
        log_error "  $name: configuration drifted"
        fail=$((fail + 1))
        if [[ $VERBOSE -eq 1 ]]; then
            diff -u "$golden" <(printf '%s\n' "$actual") | sed 's/^/      /'
        else
            diff -u "$golden" <(printf '%s\n' "$actual") \
                | grep -E '^[-+][A-Za-z]' | sed 's/^/      /' | head -12
        fi
    fi
done

echo ""
if [[ $UPDATE -eq 1 ]]; then
    log_ok "Recorded $added golden file(s)"
    exit 0
fi

printf '%bSummary%b  passed=%d failed=%d missing-golden=%d\n' \
    "$GREEN" "$NC" "$pass" "$fail" "$drift"
[[ $fail -eq 0 && $drift -eq 0 ]] || exit 1
