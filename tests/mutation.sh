#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
#
# Mutation testing. Each case breaks one piece of production code, runs the
# suite that is supposed to catch it, and restores the file. A mutation the
# suite still passes is a coverage hole: the test asserts something weaker
# than it appears to.
#
# Opt-in (not part of tests/run.sh) because it edits tracked files and takes
# about 30 seconds. Files are restored after every case, including on
# failure, and the suite refuses to run against a dirty tree.
#
#   ./tests/mutation.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "$(git status --porcelain -- lib llm solver scripts 2>/dev/null)" ]]; then
    echo "Refusing to run: tracked source files have uncommitted changes."
    echo "Commit or stash them first; this suite edits files in place."
    exit 1
fi

pass=0 fail=0

mutate() {
    local desc="$1" file="$2" from="$3" to="$4" suite="$5"
    cp "$file" /tmp/mut.bak
    python3 - "$file" "$from" "$to" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert old in s, f"mutation anchor not found in {p}: {old[:60]}"
open(p, 'w').write(s.replace(old, new, 1))
PY
    if [[ $? -ne 0 ]]; then
        cp /tmp/mut.bak "$file"
        echo "SKIP  $desc (anchor missing)"; fail=$((fail+1)); return
    fi
    if "$suite" >/tmp/mut.out 2>&1; then
        echo "HOLE  $desc -> $suite still passes"
        fail=$((fail+1))
    else
        echo "CAUGHT $desc -> $(grep -cE 'ERROR|FAIL|unbound' /tmp/mut.out) failure line(s)"
        pass=$((pass+1))
    fi
    cp /tmp/mut.bak "$file"
}

mutate "log_info back to stdout" lib/log.sh \
  "log_info()  { printf '%b[INFO]%b  %s\\n' \"\$BLUE\" \"\$NC\" \"\$1\" >&2; }" \
  "log_info()  { printf '%b[INFO]%b  %s\\n' \"\$BLUE\" \"\$NC\" \"\$1\"; }" \
  ./tests/cli.sh

mutate "gufo kv mapping inverted" lib/engines/gufo.sh \
  'export LAGUNA_KV_CACHE_TYPE=q8_0' \
  'export LAGUNA_KV_CACHE_TYPE=f16' \
  ./tests/engine-render.sh

mutate "gufo sparse dropped" lib/engines/gufo.sh \
  'export LAGUNA_SPARSE_ATTN=1' \
  'unset LAGUNA_SPARSE_ATTN' \
  ./tests/engine-render.sh

mutate "gufo bias=context no longer drops the draft" solver/gufo.sh \
  '        context)
            # The context bias trades fidelity for room' \
  '        contextx)
            # The context bias trades fidelity for room' \
  ./tests/engine-render.sh

mutate "gufo bias=context stops selecting q8_0 KV" solver/gufo.sh \
  '            _gufo_bias_kv q8_0 "$bias" "$kv_explicit"' \
  '            _gufo_bias_kv f16 "$bias" "$kv_explicit"' \
  ./tests/engine-render.sh

mutate "gufo ignores an explicit --kv-cache over the bias" solver/gufo.sh \
  '    (( explicit )) && return 0' \
  '    explicit=0' \
  ./tests/engine-render.sh

mutate "llama kv flag renamed" lib/engines/llama.sh \
  'PROFILE_ARGS+=(--cache-type-k "${SOLVER_K_TYPE}" --cache-type-v "${SOLVER_V_TYPE}")' \
  'PROFILE_ARGS+=(--cache-type-k "${SOLVER_K_TYPE}")' \
  ./tests/engine-render.sh

mutate "build engine mapping broken" llm \
  'gufo)           want_gufo=1 ;;' \
  'gufo)           want_llama=1 ;;' \
  ./tests/cli.sh

mutate "validation call removed" llm \
  '    validate_settings "$LLM_ENGINE"' \
  '    :' \
  ./tests/cli.sh

mutate "value guard removed" llm \
  'option_value "$1" "${2-}"; LLM_ENGINE="$2"; shift 2 ;;' \
  'LLM_ENGINE="$2"; shift 2 ;;' \
  ./tests/cli.sh

mutate "solver ubatch default shifted" solver/llama.sh \
  'SOLVER_UBATCH=1024; SOLVER_BATCH=4096' \
  'SOLVER_UBATCH=512; SOLVER_BATCH=4096' \
  ./tests/solver-golden.sh

mutate "gttsize parameter renamed" scripts/apply-ttm-kernel-params.sh \
  'KERNEL_PARAMS="amdgpu.gttsize=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${POOL_PAGES}"' \
  'KERNEL_PARAMS="amdgpu.gttsizeX=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${POOL_PAGES}"' \
  ./tests/apply-ttm-tests.sh

mutate "recycling pool sized back to the allocation limit" scripts/apply-ttm-kernel-params.sh \
  'ttm.page_pool_size=${POOL_PAGES}"' \
  'ttm.page_pool_size=${SIZE_PAGES}"' \
  ./tests/apply-ttm-tests.sh

mutate "supervisor trap loses the marker variable" lib/process.sh \
  "trap '_llm_write_exit_marker' EXIT" \
  "trap 'date +%s > \"\$marker\" 2>/dev/null || true' EXIT" \
  ./tests/lib.sh

mutate "supervisor leaves errexit on" lib/process.sh \
  '(( caller_errexit )) && set -e' \
  ':' \
  ./tests/lib.sh

echo ""
echo "mutation score: $pass caught, $fail hole(s)"
[[ $fail -eq 0 ]]
