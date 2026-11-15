#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# common.sh - shared setup for the per-backend build scripts
# =============================================================================
# Source this file from scripts/build/*.sh (do not execute directly):
#
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# It resolves LLM_ROOT, loads logging and hardware detection, and provides the
# pieces every backend build needs: a job count, the ccache launcher flags, and
# a check that the ROCm SDK is actually installed.
# =============================================================================

[[ -n "${_LLM_BUILD_COMMON_LOADED:-}" ]] && return 0
_LLM_BUILD_COMMON_LOADED=1

BUILD_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$BUILD_SCRIPTS_DIR/../.." && pwd)"

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"
# shellcheck source=lib/gpu.sh
source "$LLM_ROOT/lib/gpu.sh"

BUILD_JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
BUILD_CCACHE_ARGS=()
BUILD_CPU_FLAGS=()

# build_enable_ccache [hip]
#   Fill BUILD_CCACHE_ARGS with the CMake compiler launcher flags. ccache
#   caches C and C++ compilations, plus HIP single-source compilations on
#   ccache 4.x, so the launcher is set only for the languages the backend
#   compiles.
#
#   Passing the launcher explicitly also suppresses ggml's GGML_CCACHE
#   RULE_LAUNCH_COMPILE fallback, which applies the launcher globally.
build_enable_ccache() {
    local ccache_bin
    if ! ccache_bin="$(command -v ccache 2>/dev/null)"; then
        log_warn "ccache not found; every rebuild recompiles from scratch"
        log_warn "Install it with ./scripts/install-deps.sh"
        return 0
    fi

    BUILD_CCACHE_ARGS=(
        "-DCMAKE_C_COMPILER_LAUNCHER=$ccache_bin"
        "-DCMAKE_CXX_COMPILER_LAUNCHER=$ccache_bin"
    )
    if [[ "${1:-}" == "hip" ]]; then
        BUILD_CCACHE_ARGS+=("-DCMAKE_HIP_COMPILER_LAUNCHER=$ccache_bin")
    fi
    log_info "Compiler cache: $ccache_bin"
}

# build_require_rocm
#   The per-backend scripts compile against deps/ but do not download it.
#   Fail with the command that installs the SDK instead of configuring against
#   a toolchain that is not there.
build_require_rocm() {
    if [[ -x "$LLM_ROOT/deps/lib/llvm/bin/amdclang++" ]]; then
        return 0
    fi
    log_error "ROCm SDK not found at $LLM_ROOT/deps"
    log_error "Install it once with: ./scripts/build.sh --sdk-only"
    return 1
}

# build_cpu_flags
#   Split SYS_CMAKE_CPU_FLAGS into BUILD_CPU_FLAGS so the flags reach cmake
#   without relying on unquoted expansion.
build_cpu_flags() {
    BUILD_CPU_FLAGS=()
    read -r -a BUILD_CPU_FLAGS <<< "${SYS_CMAKE_CPU_FLAGS:-}"
}
