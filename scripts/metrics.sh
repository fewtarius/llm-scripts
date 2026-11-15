#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Engine metrics summary
# =============================================================================
# Reads the engine's Prometheus-format /metrics endpoint and prints a readable
# summary: throughput, cache hit rate, speculative decoding and KV usage.
#
# Both engines expose the same llama-server style names (llamacpp:...) plus
# gufo_device_lost_total, so one parser serves both.
#
# Usage:
#   ./scripts/metrics.sh              # summary against 127.0.0.1:9090
#   ./scripts/metrics.sh --port 9090
#   ./scripts/metrics.sh --raw        # dump the Prometheus text as-is
#
# The server exposes the same data at /metrics and /v1/metrics. Request-level
# measurements are also in the Server-Timing header, the response's
# "timings" object, and usage.gufo on Chat Completions responses.

set -euo pipefail

usage() {
    sed -n '5,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-9090}"
RAW=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|--host)
            if (( $# < 2 )) || [[ -z "${2-}" ]]; then
                echo "option $1 needs a value" >&2
                exit 1
            fi
            [[ "$1" == "--port" ]] && PORT="$2" || HOST="$2"
            shift 2
            ;;
        --raw)  RAW=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

URL="http://${HOST}:${PORT}/metrics"
METRICS=$(curl -fsS --max-time 10 "$URL" 2>/dev/null) || {
    echo "cannot reach the metrics endpoint at ${URL}" >&2
    echo "is the server running? (./llm serve --model <name>)" >&2
    exit 1
}

if [[ "$RAW" == "1" ]]; then
    printf '%s\n' "$METRICS"
    exit 0
fi

# Pull one metric value. Prometheus lines are "name value".
metric() {
    printf '%s\n' "$METRICS" | awk -v key="$1" '
        $1 == key { print $2; found = 1 }
        END { if (!found) print "" }'
}

fmt_int() {
    local value="$1"
    [[ -z "$value" ]] && { echo "n/a"; return; }
    printf "%'d" "$value" 2>/dev/null || echo "$value"
}

fmt_float() {
    local value="$1" decimals="${2:-1}"
    [[ -z "$value" ]] && { echo "n/a"; return; }
    printf "%.*f" "$decimals" "$value" 2>/dev/null || echo "$value"
}

fmt_bytes() {
    local value="$1"
    [[ -z "$value" ]] && { echo "n/a"; return; }
    awk -v b="$value" 'BEGIN {
        if (b >= 1073741824) printf "%.2f GiB", b / 1073741824
        else if (b >= 1048576) printf "%.1f MiB", b / 1048576
        else printf "%d B", b
    }'
}

PROMPT_TOTAL=$(metric llamacpp:prompt_tokens_total)
PROMPT_CACHED=$(metric llamacpp:prompt_tokens_cached_total)
GEN_TOTAL=$(metric llamacpp:tokens_predicted_total)
PROMPT_SEC=$(metric llamacpp:prompt_seconds_total)
GEN_SEC=$(metric llamacpp:tokens_predicted_seconds_total)
PREFILL_TPS=$(metric llamacpp:prompt_tokens_seconds)
DECODE_TPS=$(metric llamacpp:predicted_tokens_seconds)
DRAFTS=$(metric llamacpp:spec_decode_num_drafts_total)
DRAFT_TOKENS=$(metric llamacpp:spec_decode_num_draft_tokens_total)
ACCEPTED=$(metric llamacpp:spec_decode_num_accepted_tokens_total)
KV_RATIO=$(metric llamacpp:kv_cache_usage_ratio)
DEVICE_LOST=$(metric gufo_device_lost_total)
N_TOKENS_MAX=$(metric llamacpp:n_tokens_max)
INFLIGHT=$(metric llamacpp:requests_processing)
DEFERRED=$(metric llamacpp:requests_deferred)

# Cache hit rate over the server's lifetime.
HIT_RATE="n/a"
if [[ -n "$PROMPT_TOTAL" && -n "$PROMPT_CACHED" ]]; then
    HIT_RATE=$(awk -v c="$PROMPT_CACHED" -v t="$PROMPT_TOTAL" '
        BEGIN { total = t + c; printf "%.1f%%", (total > 0) ? 100.0 * c / total : 0 }')
fi

# Speculative decoding acceptance rate.
ACCEPT_RATE="n/a"
if [[ -n "$DRAFT_TOKENS" && -n "$ACCEPTED" ]]; then
    ACCEPT_RATE=$(awk -v a="$ACCEPTED" -v d="$DRAFT_TOKENS" '
        BEGIN { printf "%.1f%%", (d > 0) ? 100.0 * a / d : 0 }')
fi

echo "metrics - ${URL}"
echo
echo "Throughput"
echo "  prefill        $(fmt_float "$PREFILL_TPS" 1) tok/s   (cumulative $(fmt_float "$PROMPT_SEC" 1)s over $(fmt_int "$PROMPT_TOTAL") fresh tokens)"
echo "  decode         $(fmt_float "$DECODE_TPS" 2) tok/s   (cumulative $(fmt_float "$GEN_SEC" 1)s over $(fmt_int "$GEN_TOTAL") tokens)"
echo
echo "Cache"
echo "  prompt tokens  $(fmt_int "$PROMPT_TOTAL") fresh, $(fmt_int "$PROMPT_CACHED") reused"
echo "  hit rate       ${HIT_RATE}"
echo
echo "Speculative decoding"
if [[ -z "$DRAFTS" || "$DRAFTS" == "0" ]]; then
    echo "  disabled       no speculative rounds (drafts=0, proposed=0, accepted=0)"
else
    echo "  rounds         $(fmt_int "$DRAFTS")"
    echo "  draft tokens   $(fmt_int "$DRAFT_TOKENS") proposed, $(fmt_int "$ACCEPTED") accepted (${ACCEPT_RATE})"
fi
echo
echo "Sessions"
echo "  in flight      $(fmt_int "$INFLIGHT")   deferred: $(fmt_int "$DEFERRED")"
echo "  largest prompt $(fmt_int "$N_TOKENS_MAX") tokens"
echo "  kv usage       $(fmt_float "$KV_RATIO" 3) of session context"
if [[ -n "$DEVICE_LOST" && "$DEVICE_LOST" != "0" ]]; then
    echo
    echo "  WARNING: $(fmt_int "$DEVICE_LOST") confirmed GPU context losses"
fi
