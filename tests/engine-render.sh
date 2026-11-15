#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Engine contract tests: generic options must map to the right engine flags
# and environment variables. No binaries and no models are required.

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
export LLM_ROOT

# The sandbox comes before the sources: lib/gguf.sh resolves GGUF_META from
# LLM_GGUF_META when it loads, so exporting the stub afterwards left every
# case reading the real tool. The stubs themselves are written below, which is
# fine because they are only executed later.
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT
mkdir -p "$SANDBOX/models"
export LLM_MODEL_DIR="$SANDBOX/models"
export LLM_GGUF_META="$SANDBOX/gguf_meta.py"
export LLM_GGUF_PLE="$SANDBOX/gguf_ple.py"
export LLM_DRY_RUN=1
export LLM_SKIP_GPU_PERF=1

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"
# shellcheck source=lib/common.sh
source "$LLM_ROOT/lib/common.sh"
# shellcheck source=lib/profile.sh
source "$LLM_ROOT/lib/profile.sh"
# shellcheck source=lib/engines/llama.sh
source "$LLM_ROOT/lib/engines/llama.sh"
# shellcheck source=lib/engines/gufo.sh
source "$LLM_ROOT/lib/engines/gufo.sh"

cat > "$LLM_GGUF_META" <<'PY'
import json, os, sys
json.dump(json.loads(os.environ.get("FIXTURE_JSON", "{}")), sys.stdout)
PY
cat > "$LLM_GGUF_PLE" <<'PY'
import os, sys
sys.stdout.write(os.environ.get("FIXTURE_PLE", "0"))
PY

pass=0 fail=0
check() {
    local desc="$1" ok="$2"
    if [[ "$ok" == "0" ]]; then
        log_ok "$desc"; pass=$((pass+1))
    else
        log_error "$desc"; fail=$((fail+1))
    fi
}

argv_has() {
    local want="$1" i
    for i in "${PROFILE_ARGS[@]}"; do
        [[ "$i" == "$want" ]] && return 0
    done
    return 1
}

argv_pair() {
    local key="$1" val="$2" i
    for ((i=0; i<${#PROFILE_ARGS[@]}-1; i++)); do
        [[ "${PROFILE_ARGS[$i]}" == "$key" && "${PROFILE_ARGS[$((i+1))]}" == "$val" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------- gufo ------
LAGUNA="$SANDBOX/models/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf"
DRAFTER="$SANDBOX/models/Laguna-S-2.1-DFlash-BF16-gufo.gguf"
: > "$LAGUNA"; : > "$DRAFTER"
export FIXTURE_JSON='{"general.architecture":"laguna","laguna.block_count":61,"laguna.attention.head_count":40}'
LLM_ENGINE=gufo
LLM_CTX=0 LLM_SESSIONS=1 LLM_KV_CACHE=q8_0 LLM_SPARSE_ATTN=auto LLM_BIAS=mtp
profile_reset
engine_solve_gufo "$LAGUNA" || true
engine_render_gufo || true

[[ "$PROFILE_SPEC" == "dflash2" ]]; check "gufo: finds the DFlash drafter" $?
[[ "$PROFILE_DRAFT_TOKENS" == "2" ]]; check "gufo: Laguna draft count is 2" $?
argv_has "--dflash-model"; check "gufo: renders --dflash-model" $?
argv_pair "-d" "2"; check "gufo: renders -d 2" $?
engine_env_gufo
[[ "${LAGUNA_KV_CACHE_TYPE:-}" == "q8_0" ]]; check "gufo: --kv-cache q8_0 sets LAGUNA_KV_CACHE_TYPE" $?
[[ "${LAGUNA_SPARSE_ATTN:-}" == "1" ]]; check "gufo: sparse attention is on by default for Laguna" $?

LLM_BIAS=context
profile_reset
engine_solve_gufo "$LAGUNA" || true
[[ "$PROFILE_SPEC" == "off" ]]; check "gufo: --bias context drops the draft" $?

# The context bias buys room, and q8_0 KV is how: a row keeps 53% of its f16
# bytes. An explicit --kv-cache is the narrower request and outranks it.
LLM_BIAS=context
unset LLM_KV_CACHE
profile_reset
engine_solve_gufo "$LAGUNA" || true
[[ "$PROFILE_KV_CACHE" == "q8_0" ]]; check "gufo: --bias context selects q8_0 KV" $?

LLM_BIAS=context LLM_KV_CACHE=f16
profile_reset
engine_solve_gufo "$LAGUNA" || true
[[ "$PROFILE_KV_CACHE" == "f16" ]]; check "gufo: --kv-cache outranks the bias" $?

LLM_BIAS=quality
unset LLM_KV_CACHE
profile_reset
engine_solve_gufo "$LAGUNA" || true
[[ "$PROFILE_KV_CACHE" == "f16" ]]; check "gufo: --bias quality pins f16 KV" $?

# Families with no cache switch must fall back to f16 instead of rendering a
# q8_0 request the extension cannot honor.
LLM_BIAS=context
export FIXTURE_JSON='{"general.architecture":"deepseek2","deepseek2.block_count":61}'
profile_reset
engine_solve_gufo "$LAGUNA" || true
[[ "$PROFILE_KV_CACHE" == "f16" ]]
check "gufo: q8_0 KV falls back to f16 where the family has no switch" $?

# --------------------------------------------------------------- llama ------
QWEN="$SANDBOX/models/Qwen3.6-Test-UD-Q4_K_XL.gguf"
: > "$QWEN"
export FIXTURE_JSON='{"general.architecture":"qwen3moe","qwen3moe.block_count":40,"qwen3moe.embedding_length":4096,"qwen3moe.attention.head_count":32,"qwen3moe.attention.head_count_kv":4,"qwen3moe.attention.key_length":128,"qwen3moe.attention.value_length":128,"qwen3moe.context_length":32768,"qwen3moe.expert_count":128,"qwen3moe.expert_used_count":8}'
LLM_ENGINE=llama.cpp
LLM_CTX=32768 LLM_SESSIONS=2 LLM_KV_CACHE=q8_0 LLM_BIAS=mtp LLM_SPEC=off
LLM_GPU_BUDGET_BYTES=$(( 60 * 1073741824 ))
profile_reset
engine_solve_llama "$QWEN" || true
engine_render_llama || true

[[ "$PROFILE_CTX" == "32768" ]]; check "llama: --ctx reaches the profile" $?
argv_pair "--cache-type-k" "q8_0"; check "llama: --kv-cache q8_0 maps to --cache-type-k" $?
argv_pair "--cache-type-v" "q8_0"; check "llama: --kv-cache q8_0 maps to --cache-type-v" $?
argv_pair "-np" "2"; check "llama: --sessions maps to -np" $?
argv_has "--no-mmproj"; check "llama: default run disables mmproj" $?
[[ "$PROFILE_SPEC" == "off" ]]; check "llama: --spec off disables drafting" $?

# JSON contract
json=$(profile_emit_json)
python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["engine"]=="llama.cpp"; assert d["context"]==32768; assert isinstance(d["args"], list)' <<< "$json"
check "profile: JSON is valid and carries the resolved values" $?

echo ""
printf 'Summary passed=%d failed=%d\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
