#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Hermetic tests for scripts/apply-ttm-kernel-params.sh
# =============================================================================
# Exercises the bootloader-config mutations against synthetic systemd-boot
# entries and GRUB defaults. Nothing on the host is read or written - every
# input file is a temp copy.
#
# Usage:
#   ./tests/apply-ttm-tests.sh
#   ./tests/apply-ttm-tests.sh --verbose
#
# Exits non-zero on the first failing check.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$LLM_ROOT"

SCRIPT="$PROJECT_ROOT/scripts/apply-ttm-kernel-params.sh"
VERBOSE=0
if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT not found"
    exit 1
fi

# Build a function library from the script: everything above the "# Main"
# section, which holds the pure functions and nothing else. It is written
# under scratch/ so PROJECT_ROOT still resolves to the repo root and the
# sourced lib/gpu.sh / lib/log.sh are found.
LIB="$PROJECT_ROOT/scratch/ttm-test-lib.$$.sh"
mkdir -p "$PROJECT_ROOT/scratch"
trap 'rm -f "$LIB"' EXIT
awk '/^# Main$/ {exit} {print}' "$SCRIPT" > "$LIB"

# shellcheck source=/dev/null
source "$LIB"
# The library carries the script's own strict mode. A test harness wants to
# keep running after a failed assertion.
set +e +u +o pipefail

FAILURES=0
CHECKS=0

pass() {
    CHECKS=$((CHECKS + 1))
    echo "PASS: $1"
}

fail() {
    CHECKS=$((CHECKS + 1))
    FAILURES=$((FAILURES + 1))
    echo "FAIL: $1"
    if (( VERBOSE )); then
        echo "      expected: $2"
        echo "      actual:   $3"
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" label="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        pass "$label"
    else
        fail "$label" "contains '$needle'" "$haystack"
    fi
}

assert_not_contains() {
    local haystack="$1" needle="$2" label="$3"
    if [[ "$haystack" != *"$needle"* ]]; then
        pass "$label"
    else
        fail "$label" "does not contain '$needle'" "$haystack"
    fi
}

assert_eq() {
    local actual="$1" expected="$2" label="$3"
    if [[ "$actual" == "$expected" ]]; then
        pass "$label"
    else
        fail "$label" "$expected" "$actual"
    fi
}

# The values a 10GB GTT request resolves to. These come from the script under
# test, not from a copy: a wrong parameter name has to fail here.
GTT_SIZE_GB=10
build_kernel_params "$GTT_SIZE_GB"

assert_contains "$KERNEL_PARAMS" "amdgpu.gttsize=${SIZE_MB}" "kernel params carry amdgpu.gttsize"
assert_contains "$KERNEL_PARAMS" "ttm.pages_limit=${SIZE_PAGES}" "kernel params carry ttm.pages_limit"
assert_contains "$KERNEL_PARAMS" "ttm.page_pool_size=${POOL_PAGES}" "kernel params carry ttm.page_pool_size"
assert_not_contains "$KERNEL_PARAMS" "ttm.page_pool_size=${SIZE_PAGES}" "recycling pool is not sized to the allocation limit"
assert_eq "$POOL_PAGES" "$(( LLM_TTM_POOL_GB * 1024 * 1024 / 4 ))" "pool pages follow LLM_TTM_POOL_GB"
assert_not_contains "$KERNEL_PARAMS" "vis_vramlimit" "kernel params never set vis_vramlimit"

TMP_FILES=()
new_tmp() {
    local f
    f=$(mktemp)
    TMP_FILES+=("$f")
    echo "$f"
}

cleanup() {
    local f
    for f in "${TMP_FILES[@]:-}"; do
        [[ -n "$f" ]] && rm -f "$f"
    done
}
trap 'cleanup; rm -f "$LIB"' EXIT

# -----------------------------------------------------------------------------
# Default GTT sizing
# -----------------------------------------------------------------------------
assert_eq "$(default_gtt_gb 32768 25811 6)" "21" \
    "handheld: GTT is sized from the RAM the kernel sees"
assert_eq "$(default_gtt_gb 131072 125000 96)" "4" \
    "large firmware carveout: GTT stays small"
assert_eq "$(default_gtt_gb 131072 128000 0)" "121" \
    "no carveout: GTT fills the visible RAM minus a reserve"
assert_eq "$(default_gtt_gb 8192 7000 1)" "4" \
    "small machine: GTT uses the smaller reserve"
assert_eq "$(default_gtt_gb 32768 0 6)" "22" \
    "no meminfo: GTT is sized from physical RAM minus the carveout"
assert_eq "$(default_gtt_gb 2048 1000 0)" "1" \
    "a size below 1GB is clamped"

# -----------------------------------------------------------------------------
# GRUB: apply
# -----------------------------------------------------------------------------
grub_file=$(new_tmp)
cat > "$grub_file" <<'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.vis_vramlimit=6144 amdgpu.gttsize=20480 ttm.pages_limit=5242880 ttm.page_pool_size=5242880 quiet splash"
GRUB_TIMEOUT=5
EOF

mutate_grub_params "$grub_file"
out=$(cat "$grub_file")
assert_contains "$out" "amdgpu.gttsize=10240" "GRUB apply writes the new gttsize"
assert_not_contains "$out" "vis_vramlimit" "GRUB apply removes vis_vramlimit"
assert_not_contains "$out" "gttsize=20480" "GRUB apply drops the old gttsize"
assert_contains "$out" "ttm.pages_limit=2621440" "GRUB apply writes the new pages limit"
assert_contains "$out" "quiet splash" "GRUB apply keeps unrelated params"
assert_contains "$out" "GRUB_TIMEOUT=5" "GRUB apply keeps unrelated lines"

# Applying twice must not leave two gttsize entries - the kernel takes the
# last one, so a duplicate silently pins the old value.
mutate_grub_params "$grub_file"
dupes=$(grep -oE "amdgpu\.gttsize=[0-9]+" "$grub_file" | wc -l)
assert_eq "$dupes" "1" "GRUB apply twice yields a single gttsize"

# -----------------------------------------------------------------------------
# GRUB: remove
# -----------------------------------------------------------------------------
mutate_grub_remove "$grub_file"
out=$(cat "$grub_file")
assert_not_contains "$out" "amdgpu." "GRUB remove drops every amdgpu param"
assert_not_contains "$out" "ttm." "GRUB remove drops every ttm param"
assert_contains "$out" "quiet splash" "GRUB remove keeps unrelated params"

grub_clean=$(new_tmp)
printf 'GRUB_CMDLINE_LINUX_DEFAULT="quiet"\n' > "$grub_clean"
mutate_grub_remove "$grub_clean"
assert_contains "$(cat "$grub_clean")" 'GRUB_CMDLINE_LINUX_DEFAULT="quiet"' \
    "GRUB remove on a clean file is a no-op"

grub_broken=$(new_tmp)
printf 'GRUB_TIMEOUT=5\n' > "$grub_broken"
if mutate_grub_params "$grub_broken" 2>/dev/null; then
    fail "GRUB apply without GRUB_CMDLINE_LINUX_DEFAULT fails" "non-zero exit" "zero exit"
else
    pass "GRUB apply without GRUB_CMDLINE_LINUX_DEFAULT fails"
fi
if mutate_grub_remove "$grub_broken" 2>/dev/null; then
    fail "GRUB remove without GRUB_CMDLINE_LINUX_DEFAULT fails" "non-zero exit" "zero exit"
else
    pass "GRUB remove without GRUB_CMDLINE_LINUX_DEFAULT fails"
fi

# -----------------------------------------------------------------------------
# systemd-boot: apply
# -----------------------------------------------------------------------------
entry_file=$(new_tmp)
cat > "$entry_file" <<'EOF'
title JELOS
linux /vmlinuz-linux
initrd /amd-ucode.img
initrd /initramfs-linux.img
options amdgpu.vis_vramlimit=6144 amdgpu.gttsize=20480 ttm.pages_limit=5242880 ttm.page_pool_size=5242880 root=/dev/nvme0n1p2 rw quiet splash
EOF

mutate_entry_params "$entry_file"
out=$(cat "$entry_file")
assert_contains "$out" "amdgpu.gttsize=10240" "entry apply writes the new gttsize"
assert_not_contains "$out" "vis_vramlimit" "entry apply removes vis_vramlimit"
assert_not_contains "$out" "gttsize=20480" "entry apply drops the old gttsize"
assert_contains "$out" "root=/dev/nvme0n1p2 rw quiet splash" "entry apply keeps unrelated params"
assert_contains "$out" "title JELOS" "entry apply keeps unrelated lines"

dupes=$(grep -oE "amdgpu\.gttsize=[0-9]+" "$entry_file" | wc -l)
assert_eq "$dupes" "1" "entry apply writes a single gttsize"

# -----------------------------------------------------------------------------
# systemd-boot: remove
# -----------------------------------------------------------------------------
strip_gpu_params_from_file "$entry_file"
out=$(cat "$entry_file")
assert_not_contains "$out" "amdgpu." "entry remove drops every amdgpu param"
assert_not_contains "$out" "ttm." "entry remove drops every ttm param"
assert_contains "$out" "root=/dev/nvme0n1p2 rw quiet splash" "entry remove keeps unrelated params"

entry_bare=$(new_tmp)
printf 'title JELOS\nlinux /vmlinuz-linux\n' > "$entry_bare"
if mutate_entry_params "$entry_bare" 2>/dev/null; then
    fail "entry apply without an options line fails" "non-zero exit" "zero exit"
else
    pass "entry apply without an options line fails"
fi

# -----------------------------------------------------------------------------
# Unit suffixes - a suffixed value must be removed whole, not leave the
# letter behind as a stray kernel argument.
# -----------------------------------------------------------------------------
entry_suffix=$(new_tmp)
printf 'options amdgpu.vis_vramlimit=6144M amdgpu.gttsize=20480M quiet\n' > "$entry_suffix"
mutate_entry_params "$entry_suffix"
out=$(cat "$entry_suffix")
assert_not_contains "$out" "6144M" "entry apply removes a suffixed vis_vramlimit"
assert_not_contains "$out" "20480M" "entry apply removes a suffixed gttsize"
assert_not_contains "$out" "M quiet" "entry apply leaves no stray unit letter"

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
verify_file=$(new_tmp)
printf 'options amdgpu.gttsize=10240 ttm.pages_limit=2621440 ttm.page_pool_size=2621440 quiet\n' > "$verify_file"
if verify_params_present "$verify_file"; then
    pass "verify accepts the requested gttsize with no vis_vramlimit"
else
    fail "verify accepts the requested gttsize with no vis_vramlimit" "accepted" "rejected"
fi

printf 'options amdgpu.gttsize=10240 amdgpu.vis_vramlimit=6144 quiet\n' > "$verify_file"
if verify_params_present "$verify_file"; then
    fail "verify rejects a file that still carries vis_vramlimit" "rejected" "accepted"
else
    pass "verify rejects a file that still carries vis_vramlimit"
fi

printf 'options amdgpu.gttsize=102400 quiet\n' > "$verify_file"
if verify_params_present "$verify_file"; then
    fail "verify rejects a gttsize that is not the requested one" "rejected" "accepted"
else
    pass "verify rejects a gttsize that is not the requested one"
fi

printf 'options amdgpu.gttsize=1024 quiet\n' > "$verify_file"
if verify_params_present "$verify_file"; then
    fail "verify does not match a gttsize prefix of the requested one" "rejected" "accepted"
else
    pass "verify does not match a gttsize prefix of the requested one"
fi

# -----------------------------------------------------------------------------
echo ""
if (( FAILURES > 0 )); then
    echo "${FAILURES} of ${CHECKS} checks failed"
    exit 1
fi
echo "all ${CHECKS} checks passed"
