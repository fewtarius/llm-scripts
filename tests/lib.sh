#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Library-level tests: model discovery, process guards, environment setup,
# profile emission and pinning. Each library is loaded on its own, which also
# proves it does not depend on the caller having sourced something first.

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
export LLM_ROOT

S="$(mktemp -d)"
cleanup() { rm -rf "$S"; }
trap cleanup EXIT

export LLM_MODEL_DIR="$S/models"
export LLM_CONFIG="$S/absent.conf"
export LLM_STATE_DIR="$S/state"
export LLM_PROFILE_DIR="$S/profiles"
export HOME="$S/home"
mkdir -p "$LLM_MODEL_DIR" "$HOME" "$S/disable"

pass=0 fail=0
ok()   { log_ok "$1"; pass=$((pass+1)); }
bad()  { log_error "$1"; fail=$((fail+1)); }
eq()   { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1 (expected '$3', got '$2')"; fi; }

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"

# --------------------------------------------------------------- models ----
# shellcheck source=lib/models.sh
source "$LLM_ROOT/lib/models.sh"

M="$LLM_MODEL_DIR"
: > "$M/Model-UD-Q4_K_XL.gguf"
: > "$M/Model-UD-Q4_K_XL-00001-of-00002.gguf"
: > "$M/Model-UD-Q4_K_XL-00002-of-00002.gguf"
: > "$M/Model-DFlash-Q8_0-gufo.gguf"
: > "$M/mtp-Model-Q8_0.gguf"
: > "$M/mmproj-Model.gguf"
: > "$M/With Space.gguf"
: > "$M/Unique-Alpha-UD-Q4_K_XL.gguf"

llm_is_later_shard "$M/Model-UD-Q4_K_XL-00002-of-00002.gguf" && ok "later shard recognised" || bad "later shard not recognised"
llm_is_later_shard "$M/Model-UD-Q4_K_XL-00001-of-00002.gguf" && bad "first shard treated as later" || ok "first shard not treated as later"
llm_is_sidecar "$M/Model-DFlash-Q8_0-gufo.gguf" && ok "dflash is a sidecar" || bad "dflash not a sidecar"
llm_is_sidecar "$M/mtp-Model-Q8_0.gguf" && ok "mtp is a sidecar" || bad "mtp not a sidecar"
llm_is_sidecar "$M/mmproj-Model.gguf" && ok "mmproj is a sidecar" || bad "mmproj not a sidecar"
llm_is_sidecar "$M/Model-UD-Q4_K_XL.gguf" && bad "plain model treated as sidecar" || ok "plain model is not a sidecar"

eq "base name strips shard and quant" "$(llm_model_base_name "$M/Model-UD-Q4_K_XL-00001-of-00002.gguf")" "Model"
eq "drafter discovery prefers -gufo" "$(basename "$(llm_find_sidecar "$M/Model-UD-Q4_K_XL.gguf" dflash)")" "Model-DFlash-Q8_0-gufo.gguf"
eq "prefix drafter discovery" "$(basename "$(llm_find_sidecar "$M/Model-UD-Q4_K_XL.gguf" mtp)")" "mtp-Model-Q8_0.gguf"
llm_find_sidecar "$M/Model-UD-Q4_K_XL.gguf" nosuch >/dev/null 2>&1 && bad "missing drafter reported as found" || ok "missing drafter returns failure"
eq "entries exclude shards and sidecars" "$(llm_model_entries | wc -l)" "4"
eq "resolve by unique substring" "$(basename "$(llm_resolve_model Unique-Alpha 2>/dev/null)")" "Unique-Alpha-UD-Q4_K_XL.gguf"
eq "resolve a name with a space" "$(basename "$(llm_resolve_model "With Space" 2>/dev/null)")" "With Space.gguf"
llm_resolve_model nosuchmodel >/dev/null 2>&1 && bad "unknown model resolved" || ok "unknown model fails"

# -------------------------------------------------------------- process ----
# shellcheck source=lib/process.sh
source "$LLM_ROOT/lib/process.sh"

cp /bin/sleep "$S/gufo-test-bin" 2>/dev/null || true
if [[ -x "$S/gufo-test-bin" ]]; then
    "$S/gufo-test-bin" 15 & pid=$!
    sleep 0.2
    llm_assert_gpu_exclusive gufo-test-bin >/dev/null 2>&1 && bad "running engine not detected" || ok "running engine detected by process name"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
fi
llm_assert_gpu_exclusive gufo-test-bin >/dev/null 2>&1 && ok "absent engine does not block" || bad "absent engine blocks"
eq "free port has no owner" "$(llm_pids_on_port 1 2>/dev/null | wc -l)" "0"

python3 -c "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',18211)); s.listen(1); time.sleep(6)" & lpid=$!
sleep 0.3
llm_pids_on_port 18211 | grep -q . && ok "listener detected on its port" || bad "listener not detected"
kill "$lpid" 2>/dev/null; wait "$lpid" 2>/dev/null

# ------------------------------------------------------------------ env ----
# shellcheck source=lib/env.sh
source "$LLM_ROOT/lib/env.sh"

ensure_drirc_unified_heap >/dev/null 2>&1
grep -q radv_enable_unified_heap_on_apu "$HOME/.drirc" 2>/dev/null && ok "drirc written" || bad "drirc not written"
before=$(md5sum "$HOME/.drirc" | cut -d' ' -f1)
ensure_drirc_unified_heap >/dev/null 2>&1
eq "drirc write is idempotent" "$(md5sum "$HOME/.drirc" | cut -d' ' -f1)" "$before"

cat > "$HOME/.drirc" <<'XML'
<?xml version="1.0"?>
<driconf>
  <device driver="radeonsi">
    <application name="keepme" application_name_match="keepme">
      <option name="x" value="1"/>
    </application>
  </device>
</driconf>
XML
ensure_drirc_unified_heap >/dev/null 2>&1
grep -q keepme "$HOME/.drirc" && ok "existing driconf entries preserved" || bad "existing driconf entries lost"
grep -q radv_enable_unified_heap_on_apu "$HOME/.drirc" && ok "unified heap merged into existing file" || bad "unified heap not merged"
python3 -c "import xml.etree.ElementTree as E,sys; E.parse(sys.argv[1])" "$HOME/.drirc" 2>/dev/null \
    && ok "merged drirc is valid XML" || bad "merged drirc is not valid XML"

# -------------------------------------------------------------- profile ----
# shellcheck source=lib/profile.sh
source "$LLM_ROOT/lib/profile.sh"

profile_reset
PROFILE_ENGINE="test" PROFILE_MODEL="/x/model.gguf" PROFILE_MODEL_NAME="model"
PROFILE_CTX=4096 PROFILE_ARGS=(-m /x/model.gguf --flag "with space") PROFILE_ENV=(A=1)
json=$(profile_emit_json)
python3 -c '
import json,sys
d=json.loads(sys.stdin.read())
assert d["engine"]=="test"
assert d["context"]==4096
assert d["args"]==["-m","/x/model.gguf","--flag","with space"]
assert d["env"]==["A=1"]
' <<< "$json" 2>/dev/null && ok "profile JSON round-trips arrays and spaces" || bad "profile JSON malformed"

PROFILE_MODEL_NAME="quote\"and\\backslash"
json=$(profile_emit_json)
python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
assert d["model_name"] == '"'"'quote"and\\backslash'"'"', repr(d["model_name"])' <<< "$json" 2>/dev/null \
    && ok "profile JSON escapes quotes and backslashes" || bad "profile JSON escaping broken"

PROFILE_MODEL_NAME="model"
llm_save_pin "/x/model.gguf" >/dev/null 2>&1
[[ -f "$S/profiles/model.conf" ]] && ok "pin saved" || bad "pin not saved"
unset LLM_CTX
llm_load_pin "/x/model.gguf" >/dev/null 2>&1
eq "pin restores context" "${LLM_CTX:-}" "4096"


# ---------------------------------------------------------- supervisor ----
# llm_supervise decides whether a failed server is retried. Getting this
# wrong either restarts behind the user's back or never restarts at all.

mk_stub() {
    local path="$1" body="$2"
    printf '#!/bin/bash\n%s\n' "$body" > "$path"
    chmod +x "$path"
}

# clean exit
mk_stub "$S/ok.sh" 'echo "event=listening"; exit 0'
LLM_DEVICE_SETTLE_SECONDS=0 llm_supervise "event=listening" "$S/l.ok" "$S/m.ok" "$S/ok.sh" >/dev/null 2>&1
eq "supervise: clean exit returns 0" "$?" "0"

# served, then died: never restart behind the user's back
echo 0 > "$S/c.died"
mk_stub "$S/died.sh" "n=\$(( \$(cat $S/c.died) + 1 )); echo \$n > $S/c.died; echo 'event=listening'; exit 3"
LLM_DEVICE_SETTLE_SECONDS=0 LLM_LOAD_ATTEMPTS=3 \
    llm_supervise "event=listening" "$S/l.died" "$S/m.died" "$S/died.sh" >/dev/null 2>&1
eq "supervise: a server that served is not restarted" "$(cat "$S/c.died")" "1"

# never ready: retried up to LLM_LOAD_ATTEMPTS
echo 0 > "$S/c.never"
mk_stub "$S/never.sh" "n=\$(( \$(cat $S/c.never) + 1 )); echo \$n > $S/c.never; echo boom; exit 4"
LLM_DEVICE_SETTLE_SECONDS=0 LLM_LOAD_ATTEMPTS=3 \
    llm_supervise "event=listening" "$S/l.never" "$S/m.never" "$S/never.sh" >/dev/null 2>&1
eq "supervise: a server that never started is retried" "$(cat "$S/c.never")" "3"

# The exit marker drives the device-settle wait on the next start. It is
# written by an EXIT trap, so it appears when the process exits, not when the
# function returns: check it in a subshell that exits immediately.
( LLM_DEVICE_SETTLE_SECONDS=0 llm_supervise "event=listening" "$S/l.mk" "$S/m.mk" "$S/ok.sh" >/dev/null 2>&1 ) 2>/dev/null
[[ -s "$S/m.mk" ]]
if [[ -s "$S/m.mk" ]]; then
    ok "supervise: exit marker written on exit"
else
    bad "supervise: exit marker not written"
fi

# errexit must be restored exactly as the caller had it
res=$(bash -c "set -e; source '$LLM_ROOT/lib/process.sh'; LLM_DEVICE_SETTLE_SECONDS=0 llm_supervise '' '$S/l.e' '$S/m.e' '$S/ok.sh' >/dev/null 2>&1; case \$- in *e*) echo restored;; *) echo lost;; esac" 2>/dev/null)
eq "supervise: restores errexit when the caller had it" "$res" "restored"
res=$(bash -c "set +e; source '$LLM_ROOT/lib/process.sh'; LLM_DEVICE_SETTLE_SECONDS=0 llm_supervise '' '$S/l.e2' '$S/m.e2' '$S/ok.sh' >/dev/null 2>&1; case \$- in *e*) echo added;; *) echo absent;; esac" 2>/dev/null)
eq "supervise: does not add errexit to a caller without it" "$res" "absent"

# ------------------------------------------------- self-contained loading --
# Every library and adapter must load and work on its own. A file that only
# functions when the caller happened to source something else is a latent
# break for the next consumer.
for f in lib/log.sh lib/common.sh lib/gpu.sh lib/env.sh lib/models.sh \
         lib/gguf.sh lib/process.sh lib/profile.sh lib/validate.sh \
         lib/download.sh lib/rust.sh lib/engines/llama.sh lib/engines/gufo.sh \
         solver/llama.sh solver/gufo.sh; do
    if bash -c "set -u; source '$LLM_ROOT/$f'" >/dev/null 2>&1; then
        pass=$((pass+1))
    else
        bad "$f does not load standalone"
    fi
done
ok "every library and adapter loads standalone"

# Adapter smoke: the environment mapping must work without the dispatcher.
smoke=$(bash -c "
    set -u
    source '$LLM_ROOT/lib/engines/gufo.sh'
    SOLVER_GUFO_FAMILY=laguna
    PROFILE_KV_CACHE=q8_0
    PROFILE_SPARSE_ATTN=on
    PROFILE_ENV=()
    LLM_SKIP_UNIFIED_HEAP=1 LLM_SKIP_GPU_PERF=1
    engine_env_gufo >/dev/null 2>&1
    printf '%s %s' "\${LAGUNA_KV_CACHE_TYPE:-none}" "\${LAGUNA_SPARSE_ATTN:-none}"
" 2>/dev/null)
eq "gufo adapter maps kv and sparse on its own" "$smoke" "q8_0 1"

smoke=$(bash -c "
    set -u
    source '$LLM_ROOT/lib/engines/llama.sh'
    printf '%s' \"\$(llama_resolve_backend 2>/dev/null || echo none)\"
" 2>/dev/null)
[[ -n "$smoke" ]] && ok "llama adapter resolves a backend on its own" || bad "llama adapter needs the dispatcher"

echo ""
printf 'Summary passed=%d failed=%d\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
