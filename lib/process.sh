#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Server lifecycle: port ownership, GPU exclusivity, device settle, retry.

[[ -n "${_LLM_PROCESS_LOADED:-}" ]] && return 0
_LLM_PROCESS_LOADED=1

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# PIDs listening on a TCP port. Empty when the port is free.
llm_pids_on_port() {
    local port="$1"
    [[ -n "$port" ]] || return 0
    if llm_have lsof; then
        lsof -ti "tcp:$port" 2>/dev/null || true
    elif llm_have ss; then
        ss -ltnpH "sport = :$port" 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' | sort -u
    fi
}

# Stop only processes whose command name matches $2 and that hold $1.
llm_stop_on_port() {
    local port="$1" comm="$2" pid pname
    local pids
    pids=$(llm_pids_on_port "$port")
    [[ -n "$pids" ]] || return 0
    for pid in $pids; do
        pname=$(ps -p "$pid" -o comm= 2>/dev/null || true)
        [[ "$pname" == "$comm" ]] || continue
        log_warn "Stopping stale $comm on port $port (pid $pid)"
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in $pids; do
        pname=$(ps -p "$pid" -o comm= 2>/dev/null || true)
        [[ "$pname" == "$comm" ]] || continue
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    done
}

# Refuse to start a second engine process that would share the GPU.
#   llm_assert_gpu_exclusive <process-name>
# Matching is on the executable name, not the command line: a shell, an
# editor, or a grep that merely mentions the engine must not block a start.
llm_assert_gpu_exclusive() {
    local comm="$1"
    [[ -n "$comm" ]] || return 0
    local pids pid
    pids=$(pgrep -x "$comm" 2>/dev/null || true)
    local offenders=()
    for pid in $pids; do
        [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
        offenders+=("$pid")
    done
    (( ${#offenders[@]} == 0 )) && return 0

    log_error "Another $comm is already running (pid ${offenders[*]}):"
    for pid in "${offenders[@]}"; do
        log_error "  $(ps -p "$pid" -o args= 2>/dev/null || echo "$pid")"
    done
    log_error "Only one server can hold the GPU at a time."
    log_error "Stop it first and wait for the device memory to be released."
    return 1
}

# UMA devices need a moment after a large mapping is released. The marker
# records when the previous server exited; a cold start is not delayed.
llm_wait_device_settle() {
    local marker="$1"
    local settle="${LLM_DEVICE_SETTLE_SECONDS:-120}"
    [[ "$settle" =~ ^[0-9]+$ ]] || settle=0
    (( settle <= 0 )) && return 0
    [[ -f "$marker" ]] || return 0
    local now last age wait
    now=$(date +%s)
    last=$(cat "$marker" 2>/dev/null || echo 0)
    [[ "$last" =~ ^[0-9]+$ ]] || return 0
    age=$(( now - last )); (( age < 0 )) && age=0
    if (( age < settle )); then
        wait=$(( settle - age ))
        log_info "Waiting ${wait}s for the device to settle"
        sleep "$wait"
    fi
}

_llm_write_exit_marker() {
    date +%s > "$_LLM_EXIT_MARKER" 2>/dev/null || true
}

# Run a server in the foreground, retrying only if it never became ready.
#   llm_supervise <ready-regex> <log-file> <marker-file> <argv...>
# A process that logged readiness and later died is not restarted behind the
# user's back.
llm_supervise() {
    local ready="$1" log_file="$2" marker="$3"
    shift 3
    local attempts="${LLM_LOAD_ATTEMPTS:-3}"
    local attempt status
    # Run the server with errexit suspended so a non-zero status can be
    # inspected, then restore the caller's setting. Leaving errexit on would
    # silently change how the caller behaves after this returns.
    local caller_errexit=0
    [[ $- == *e* ]] && caller_errexit=1

    mkdir -p "$(dirname "$log_file")" 2>/dev/null || true

    # Record when the server exits so the next start can wait for the device.
    # The marker lives in a global: a single-quoted trap would expand $marker
    # at exit time, when the function's locals are gone, and fail under set -u.
    _LLM_EXIT_MARKER="$marker"
    trap '_llm_write_exit_marker' EXIT

    for (( attempt = 1; attempt <= attempts; attempt++ )); do
        set +e
        "$@" 2>&1 | tee "$log_file"
        status=${PIPESTATUS[0]}
        (( caller_errexit )) && set -e
        (( status == 0 )) && return 0
        if [[ -n "$ready" ]] && grep -Eq "$ready" "$log_file" 2>/dev/null; then
            return "$status"
        fi
        (( attempt == attempts )) && return "$status"
        log_warn "Server failed during load (exit ${status}); retrying (attempt ${attempt}/${attempts})"
        sleep "${LLM_DEVICE_SETTLE_SECONDS:-120}"
    done
    return 1
}
