#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Run every hermetic test suite.
#
# A suite's exit status is not enough. A shell diagnostic on stderr
# ("command not found", "unbound variable") means the code under test crashed
# somewhere the assertions did not look, which is a failure even when every
# assertion passed. Both conditions are checked.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0
out=$(mktemp); err=$(mktemp)
trap 'rm -f "$out" "$err"' EXIT

for t in cli.sh lib.sh engine-render.sh solver-golden.sh apply-ttm-tests.sh; do
    echo ""
    echo "=== $t ==="
    : > "$out"; : > "$err"
    if ! "$DIR/$t" >"$out" 2>"$err"; then
        fail=1
        echo "    suite reported failure"
    fi
    # Match bash's own diagnostic shape (file: line N: ...), not test output
    # that merely talks about these errors.
    pattern=':[[:space:]]*line[[:space:]]+[0-9]+:.*(command not found|unbound variable|syntax error|integer expression expected)'
    if grep -qE "$pattern" "$err"; then
        fail=1
        echo "    shell diagnostics leaked to stderr:"
        grep -E "$pattern" "$err" | head -3 | sed 's/^/      /'
    fi
done

echo ""
if (( fail )); then
    echo "FAILED"
    exit 1
fi
echo "All suites passed"
