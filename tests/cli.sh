#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Command-line contract: every entry point runs under `set -u`, bare
# invocation explains itself, and mistakes exit non-zero without a stack of
# usage text. Nothing here touches the network, the GPU, or the home directory.

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$TESTS_DIR/.." && pwd)"

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT
mkdir -p "$SANDBOX/models" "$SANDBOX/state"
: > "$SANDBOX/models/Test-Model-UD-Q4_K_XL.gguf"

export LLM_MODEL_DIR="$SANDBOX/models"
export LLM_STATE_DIR="$SANDBOX/state"
export LLM_CONFIG="$SANDBOX/absent.conf"
export LLM_SKIP_UNIFIED_HEAP=1
export LLM_SKIP_GPU_PERF=1

LLM="$LLM_ROOT/llm"
pass=0 fail=0
out=""

# run <expected-exit> <description> [args...]
run() {
    local want="$1" desc="$2"
    shift 2
    out=$("$LLM" "$@" 2>&1)
    local rc=$?
    if (( rc != want )); then
        log_error "$desc: exit ${rc}, expected ${want}"
        printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
        fail=$((fail + 1))
        return 0
    fi
    log_ok "$desc"
    pass=$((pass + 1))
}

# contains <description> <text>
contains() {
    if printf '%s' "$out" | grep -qF -- "$2"; then
        log_ok "$1"
        pass=$((pass + 1))
    else
        log_error "$1: output missing '$2'"
        fail=$((fail + 1))
    fi
}

check() {
    local desc="$1" rc="$2"
    if (( rc == 0 )); then
        log_ok "$desc"
        pass=$((pass + 1))
    else
        log_error "$desc"
        fail=$((fail + 1))
    fi
}

not_contains() {
    if printf '%s' "$out" | grep -qF -- "$2"; then
        log_error "$1: output should not contain '$2'"
        fail=$((fail + 1))
    else
        log_ok "$1"
        pass=$((pass + 1))
    fi
}

run 0 "bare invocation exits 0" && {
    contains "bare invocation prints the project name" "llm-scripts"
    contains "bare invocation lists commands" "Commands"
    contains "bare invocation reports the model directory" "$SANDBOX/models"
    not_contains "bare invocation does not raise unbound variables" "unbound variable"
}

run 0 "--help exits 0" --help
contains "--help shows usage" "Usage:"

run 0 "help subcommand exits 0" help

run 0 "models runs without a profile" models
contains "models lists the sandbox model" "Test-Model-UD-Q4_K_XL"

run 0 "doctor runs" doctor
contains "doctor reports the root" "$LLM_ROOT"

run 0 "env prints an export line" env --engine llama.cpp --backend rocm
contains "env exports ROCM_PATH" "ROCM_PATH="

run 2 "serve without a model is a usage error" serve
contains "missing model explains how to list models" "llm models"
not_contains "missing model does not dump the whole usage block" "Speculative:"

run 2 "unknown command is rejected" frobnicate
contains "unknown command names the input" "frobnicate"

run 2 "unknown option is rejected" serve --nonsense X
contains "unknown option names the flag" "--nonsense"

run 2 "profile without a model is a usage error" profile

# Engine-explicit: `serve` with no --engine now defaults to gufo, and this
# assertion is about llama.cpp rendering.
run 0 "explicit serve with --print-command does not launch" \
    serve --engine llama.cpp --model Test-Model-UD-Q4_K_XL --print-command
contains "print-command emits a command line" "llama-server -m"

run 0 "profile --engine gufo is renderable" \
    profile --engine gufo --model Test-Model-UD-Q4_K_XL --print-command
contains "gufo command is a serve invocation" "gufo serve llm"

# The engine must never be left unset when no --engine is given: that is the
# bug this suite exists to catch.
run 0 "no --engine defaults to gufo" profile --model Test-Model-UD-Q4_K_XL --json
contains "profile JSON names the default engine" '"engine":"gufo"'
python3 -c 'import json,sys; json.loads(sys.stdin.read())' <<< "$out" 2>/dev/null
if (( $? == 0 )); then
    log_ok "profile JSON parses"
    pass=$((pass + 1))
else
    log_error "profile JSON does not parse"
    fail=$((fail + 1))
fi

# ---------------------------------------------------------------------------
# Every documented command must be exercised here. The list is checked against
# the dispatcher's own usage text, so documenting a command without covering
# it fails the suite.
# ---------------------------------------------------------------------------
echo ""
log_info "command coverage"

declare -A COVERED=(
    [serve]=1 [profile]=1 [models]=1 [pull]=1 [build]=1
    [doctor]=1 [tune]=1 [env]=1 [metrics]=1
)

documented=$("$LLM" --help 2>&1 | sed -nE 's/^    llm ([a-z]+).*/\1/p' | sort -u)
missing=""
while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    [[ -n "${COVERED[$cmd]:-}" ]] || missing+="$cmd "
done <<< "$documented"

if [[ -z "$missing" ]]; then
    log_ok "every documented command is covered"
    pass=$((pass + 1))
else
    log_error "documented but untested: $missing"
    fail=$((fail + 1))
fi

# ---------------------------------------------------------------------------
# Options that take a value must never reach a bare $2. Under `set -u` that is
# an unbound-variable crash instead of a usage error.
# ---------------------------------------------------------------------------
for opt in --engine -b --model --threads --ngl --ctx --sessions --kv-cache \
           --cache-type-k --cache-type-v --cache-ram --cache-disk --ubatch-size \
           --bias --spec --draft-tokens --sparse-attn --host --port --reasoning \
           --reasoning-budget --extra-args --alias; do
    out=$("$LLM" serve --model Test-Model-UD-Q4_K_XL "$opt" 2>&1)
    rc=$?
    if (( rc == 2 )) && ! grep -q 'unbound variable' <<< "$out"; then
        pass=$((pass + 1))
    else
        log_error "option without a value: $opt (exit $rc)"
        printf '%s\n' "$out" | head -1 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
done
log_ok "options reject a missing value without tripping set -u"
pass=$((pass + 0))

# ---------------------------------------------------------------------------
# Diagnostics must never contaminate captured output. A function that logs to
# stdout while its value is captured corrupts the value; this regressed once
# on ambiguous model names.
# ---------------------------------------------------------------------------
AMB="$SANDBOX/amb"
mkdir -p "$AMB"
: > "$AMB/Dup-Model-UD-Q4_K_XL.gguf"
: > "$AMB/Dup-Model-UD-Q8_K_XL.gguf"

out=$(LLM_MODEL_DIR="$AMB" "$LLM" profile Dup --json 2>/dev/null)
if python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); assert d["model"].endswith(".gguf")' <<< "$out" 2>/dev/null; then
    log_ok "ambiguous model name still yields valid JSON"
    pass=$((pass + 1))
else
    log_error "ambiguous model name corrupts the profile JSON"
    fail=$((fail + 1))
fi

out=$(LLM_MODEL_DIR="$AMB" "$LLM" profile --engine llama.cpp Dup --print-command 2>/dev/null)
if grep -qE '^  /[^ ]*llama-server -m /[^ ]*Dup-Model[^ ]*\.gguf ' <<< "$out"; then
    log_ok "ambiguous model name yields a clean -m argument"
    pass=$((pass + 1))
else
    log_error "ambiguous model name corrupts the -m argument"
    printf '%s\n' "$out" | tail -1 | cut -c1-120 | sed 's/^/      /'
    fail=$((fail + 1))
fi

# ---------------------------------------------------------------------------
# Invalid settings are rejected before an engine sees them.
# ---------------------------------------------------------------------------
reject() {
    local desc="$1" pattern="$2"
    shift 2
    out=$("$LLM" "$@" 2>&1)
    rc=$?
    if (( rc != 0 )) && grep -qi -- "$pattern" <<< "$out" && ! grep -q 'unbound variable' <<< "$out"; then
        log_ok "$desc"
        pass=$((pass + 1))
    else
        log_error "$desc (exit $rc)"
        printf '%s\n' "$out" | head -2 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
}

reject "rejects non-numeric context" "whole number" profile M --ctx abc
reject "rejects negative context" "whole number" profile M --ctx -1
reject "rejects zero sessions" "at least 1" profile M --sessions 0
reject "rejects an out-of-range port" "at most 65535" profile M --port 99999
reject "rejects an unknown kv cache type" "kv cache type" profile --engine llama.cpp M --kv-cache banana
reject "rejects an unknown speculative mode" "speculative mode" profile M --spec banana
reject "rejects an unknown bias" "bias must be one of" profile M --bias banana
reject "rejects an unknown backend" "backend must be one of" profile M --backend banana
reject "rejects an unknown engine" "Unknown engine" profile M --engine banana
reject "rejects gufo kv types it cannot run" "kv cache .gufo." profile --engine gufo M --kv-cache q4_k
reject "rejects GPU layers on gufo" "GPU layers do not apply" profile --engine gufo M --ngl 99
reject "rejects sparse attention on llama.cpp" "sparse attention" profile --engine llama.cpp M --sparse-attn on

out=$(LLM_SESSIONS=0 "$LLM" profile M 2>&1)
if (( $? != 0 )) && grep -q "at least 1" <<< "$out"; then
    log_ok "rejects invalid values from the environment directly"
    pass=$((pass + 1))
else
    log_error "invalid LLM_SESSIONS accepted"
    fail=$((fail + 1))
fi

run 2 "_exec without an engine is a usage error" _exec

# ---------------------------------------------------------------------------
# build: engine selection must reach the right build script.
# ---------------------------------------------------------------------------
out=$("$LLM" build --engine gufo --print-command 2>&1)
grep -q 'scripts/build/gufo.sh' <<< "$out" && ! grep -q 'build.sh' <<< "$out"
check "build --engine gufo plans the gufo build" $?

out=$("$LLM" build --engine llama.cpp --backend rocm --print-command 2>&1)
grep -q 'scripts/build.sh --rocm' <<< "$out"
check "build --engine llama.cpp --backend rocm plans --rocm" $?

out=$("$LLM" build --engine all --print-command 2>&1)
[[ $(grep -c . <<< "$out") -eq 2 ]]
check "build --engine all plans both scripts" $?

out=$("$LLM" build --sdk-only --print-command 2>&1)
grep -q -- '--sdk-only' <<< "$out"
check "build --sdk-only forwards the flag" $?

out=$("$LLM" build --engine gufo --clean --print-command 2>&1)
grep -q -- '--clean' <<< "$out"
check "build --clean reaches the gufo build" $?

run 2 "build rejects an unknown engine" build --engine tensorrt
contains "build names the unknown engine" "tensorrt"

run 2 "build rejects an unknown backend" build --backend cpu
contains "build explains the CPU case" "CPU"

run 2 "build rejects an unknown flag" build --nonsense
contains "build lists its own options" "llm build"

run 0 "build --help delegates to the build script" build --help
contains "build help shows build.sh usage" "build.sh [OPTIONS]"

# ---------------------------------------------------------------------------
# build scripts: one per target under scripts/build/, each standalone.
# ---------------------------------------------------------------------------
bash -n "$LLM_ROOT/scripts/build/common.sh" 2>/dev/null
check "parses: scripts/build/common.sh" $?

for script in "$LLM_ROOT"/scripts/build/llama-*.sh "$LLM_ROOT"/scripts/build/gufo.sh; do
    name="$(basename "$script")"
    bash -n "$script" 2>/dev/null
    check "parses: scripts/build/$name" $?
    "$script" --help >/dev/null 2>&1
    check "answers --help: scripts/build/$name" $?
done

for target in llama-rocm llama-vulkan llama-metal; do
    grep -q "scripts/build/$target.sh" "$LLM_ROOT/scripts/build.sh"
    check "build.sh dispatches to $target.sh" $?
done

# The compiler launcher must reach cmake for C, C++ and HIP. A stand-in ccache
# in front of PATH proves the flags are built from what resolves, not from a
# hardcoded path.
mkdir -p "$SANDBOX/bin"
printf '#!/bin/sh\nexit 0\n' > "$SANDBOX/bin/ccache"
chmod +x "$SANDBOX/bin/ccache"
out=$(PATH="$SANDBOX/bin:$PATH" bash -c '
    source "$1/scripts/build/common.sh"
    build_enable_ccache hip
    printf "%s\n" "${BUILD_CCACHE_ARGS[@]}"
' _ "$LLM_ROOT" 2>/dev/null)
grep -qF -- "-DCMAKE_C_COMPILER_LAUNCHER=$SANDBOX/bin/ccache" <<< "$out" \
    && grep -qF -- "-DCMAKE_CXX_COMPILER_LAUNCHER=$SANDBOX/bin/ccache" <<< "$out" \
    && grep -qF -- "-DCMAKE_HIP_COMPILER_LAUNCHER=$SANDBOX/bin/ccache" <<< "$out"
check "ccache becomes the C, C++ and HIP compiler launcher" $?

out=$(PATH="$SANDBOX/bin:$PATH" bash -c '
    source "$1/scripts/build/common.sh"
    build_enable_ccache
    printf "%s\n" "${BUILD_CCACHE_ARGS[@]}"
' _ "$LLM_ROOT" 2>/dev/null)
grep -qF -- "HIP_COMPILER_LAUNCHER" <<< "$out"
[[ $? -ne 0 ]]
check "non-HIP backends do not pass a HIP launcher" $?

# ---------------------------------------------------------------------------
# metrics: the summary must read the numbers the engine actually publishes.
# ---------------------------------------------------------------------------
MPORT=18391
python3 - "$MPORT" <<'PYSERVER' &
import sys, socketserver, http.server
port = int(sys.argv[1])
BODY = b"""# HELP llamacpp:prompt_tokens_total
llamacpp:prompt_tokens_total 1000
llamacpp:prompt_tokens_cached_total 400
llamacpp:tokens_predicted_total 250
llamacpp:prompt_seconds_total 2.0
llamacpp:tokens_predicted_seconds_total 10.0
llamacpp:prompt_tokens_seconds 500.0
llamacpp:predicted_tokens_seconds 25.0
llamacpp:spec_decode_num_drafts_total 8
llamacpp:spec_decode_num_draft_tokens_total 24
llamacpp:spec_decode_num_accepted_tokens_total 18
llamacpp:kv_cache_usage_ratio 0.5
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
llamacpp:n_tokens_max 4096
gufo_device_lost_total 0
"""
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Length', str(len(BODY))); self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, *a): pass
class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
S(("127.0.0.1", port), H).serve_forever()
PYSERVER
server_pid=$!
for _ in $(seq 1 40); do
    (exec 3<>/dev/tcp/127.0.0.1/$MPORT) 2>/dev/null && { exec 3<&- 3>&-; break; }
    sleep 0.1
done

out=$("$LLM" metrics --port "$MPORT" 2>&1)
contains "metrics reports prefill throughput" "500.0 tok/s"
contains "metrics reports decode throughput" "25.00 tok/s"
contains "metrics reports the cache hit rate" "28.6%"
contains "metrics reports speculative acceptance" "75.0%"
contains "metrics reports fresh and reused prompt tokens" "400 reused"

out=$("$LLM" metrics --port "$MPORT" --raw 2>&1)
contains "metrics --raw prints Prometheus text" "llamacpp:prompt_tokens_total 1000"

kill "$server_pid" 2>/dev/null
wait "$server_pid" 2>/dev/null

# ---------------------------------------------------------------------------
# The remaining commands must be runnable without a GPU, a model, or a server.
# ---------------------------------------------------------------------------
run 0 "tune plans without sudo" tune --print-command 104
contains "tune plans the kernel script" "apply-ttm-kernel-params.sh"

run 1 "metrics reports an unreachable server instead of crashing" metrics --port 1
contains "metrics explains the failure" "cannot reach"
not_contains "metrics does not trip set -u" "unbound variable"

run 2 "pull without a target is a usage error" pull
contains "pull explains the input format" "llm pull"
not_contains "pull does not trip set -u" "unbound variable"

run 1 "metrics rejects a missing port value" metrics --port
not_contains "metrics value guard is a message, not a crash" "unbound variable"

# ---------------------------------------------------------------------------
# Directory edge cases report the problem instead of showing an empty list.
# ---------------------------------------------------------------------------
emptydir="$SANDBOX/nodir"
out=$(LLM_MODEL_DIR="$emptydir" "$LLM" models 2>&1); rc=$?
if (( rc != 0 )) && grep -q "does not exist" <<< "$out"; then
    log_ok "missing model directory is reported, not shown as empty"
    pass=$((pass + 1))
else
    log_error "missing model directory silently reported as empty (exit $rc)"
    fail=$((fail + 1))
fi

locked="$SANDBOX/locked"
mkdir -p "$locked"; : > "$locked/X.gguf"; chmod 000 "$locked"
out=$(LLM_MODEL_DIR="$locked" "$LLM" models 2>&1); rc=$?
chmod 755 "$locked"
if (( rc != 0 )) && grep -qi "not readable" <<< "$out"; then
    log_ok "unreadable model directory is reported"
    pass=$((pass + 1))
else
    log_error "unreadable model directory silently reported as empty (exit $rc)"
    fail=$((fail + 1))
fi

out=$(LLM_PROFILE_DIR="/proc/nope" "$LLM" profile Test-Model --save 2>&1); rc=$?
if (( rc != 0 )) && grep -q "Cannot create the profile directory" <<< "$out" \
   && ! grep -q "cannot create directory" <<< "$out"; then
    log_ok "unwritable profile directory gives a clean error"
    pass=$((pass + 1))
else
    log_error "unwritable profile directory handled badly (exit $rc)"
    fail=$((fail + 1))
fi

# ---------------------------------------------------------------------------
# The shipped systemd unit must match what the dispatcher supports: it sets
# LLM_* in Environment= and runs `llm serve` with no model argument.
# ---------------------------------------------------------------------------
unit="$LLM_ROOT/systemd/llm-server.service"
grep -q '^ExecStart=.*/llm serve$' "$unit"
check "systemd unit runs the dispatcher with no model argument" $?

bad_env=$(grep -E '^Environment=' "$unit" | sed 's/^Environment=//; s/=.*//' | grep -v '^LLM_' || true)
[[ -z "$bad_env" ]]
check "systemd unit only sets LLM_* environment" $?

out=$(LLM_MODEL=Test-Model "$LLM" serve --print-command 2>/dev/null)
# Engine-neutral: llama.cpp spells the model -m, gufo spells it --model.
grep -qE -- '(^| )(-m|--model) .*Test-Model' <<< "$out"
check "a model named only by environment is served (as the unit does)" $?

echo ""
printf 'Summary passed=%d failed=%d\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
