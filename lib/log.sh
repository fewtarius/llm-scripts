#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Terminal logging. One implementation for every script in the tree.

[[ -n "${_LLM_LOG_LOADED:-}" ]] && return 0
_LLM_LOG_LOADED=1

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
MAGENTA=$'\033[0;35m'
CYAN=$'\033[0;36m'
DIM=$'\033[2m'
NC=$'\033[0m'

# Diagnostics go to stderr. stdout carries data only: resolved paths, JSON,
# rendered commands. A logging call inside command substitution must never be
# able to contaminate the value it is capturing.
log_info()  { printf '%b[INFO]%b  %s\n' "$BLUE" "$NC" "$1" >&2; }
log_ok()    { printf '%b[OK]%b    %s\n' "$GREEN" "$NC" "$1" >&2; }
log_warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1" >&2; }
log_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1" >&2; }
log_debug() { [[ "${LLM_DEBUG:-0}" == "1" ]] && printf '%b[DEBUG]%b %s\n' "$DIM" "$NC" "$1" >&2; return 0; }
