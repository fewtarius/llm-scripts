#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Build gufo against the bundled ROCm SDK.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/build/common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=lib/rust.sh
source "$LLM_ROOT/lib/rust.sh"
# shellcheck source=lib/env.sh
source "$LLM_ROOT/lib/env.sh"

BUILD_DIR="$LLM_ROOT/build/gufo"
GUFO_DIR="$LLM_ROOT/gufo"

usage() {
    cat <<'USAGE'
Usage: build/gufo.sh [OPTIONS]

Build gufo against the bundled ROCm SDK into build/gufo/.

OPTIONS:
    --clean     Wipe the build directory first
    -h, --help  Show this help
USAGE
}

GUFO_CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) GUFO_CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage >&2; exit 1 ;;
    esac
done

build_require_rocm || exit 1

mkdir -p "$BUILD_DIR"
llm_env_activate rocm

# gufo's HIP toolchain probe needs the device bitcode at <rocm-path>/amdgcn.
if [[ ! -e "$ROCM_PATH/amdgcn" ]]; then
    ln -sfn "$ROCM_PATH/lib/llvm/amdgcn" "$ROCM_PATH/amdgcn"
fi

if [[ "$GUFO_CLEAN" == 1 ]]; then
    log_info "Cleaning gufo build directory"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# The HIP objects carry no dependency files, so a moved source tree leaves
# objects compiled against headers that no longer exist. Wipe on commit move.
stamp="$BUILD_DIR/.source-commit"
source_commit="$(git -C "$GUFO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -f "$stamp" && "$(cat "$stamp")" != "$source_commit" ]]; then
    log_info "gufo source moved; wiping the build directory"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

ensure_rust_toolchain "$LLM_ROOT" || exit 1

hip_arch="${GPU_GFX_ARCH:-gfx1151}"
log_info "HIP target architecture: $hip_arch"

build_enable_ccache hip

# A cache written before the launcher was wired in keeps compiling directly.
# Reconfigure when the launcher entry is absent so the flags take effect.
if [[ ! -f CMakeCache.txt || "$GUFO_CLEAN" == 1 ]] \
   || { (( ${#BUILD_CCACHE_ARGS[@]} )) && ! grep -q "^CMAKE_HIP_COMPILER_LAUNCHER:" CMakeCache.txt 2>/dev/null; } \
   || ! grep -qx "CMAKE_HIP_ARCHITECTURES:STRING=$hip_arch" CMakeCache.txt 2>/dev/null; then
    cmake "$GUFO_DIR" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_TESTING=OFF \
        -DENGINE_ENABLE_HIP=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_HIP_ARCHITECTURES="$hip_arch" \
        -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/amdclang++" \
        -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH" \
        -DHIPCUB_INCLUDE_DIR="$ROCM_PATH/include" \
        -DROCPRIM_INCLUDE_DIR="$ROCM_PATH/include" \
        -DROCWMMA_INCLUDE_DIR="$ROCM_PATH/include" \
        "${BUILD_CCACHE_ARGS[@]}"
fi

cmake --build . --parallel "$BUILD_JOBS"
printf '%s\n' "$source_commit" > "$stamp"

log_ok "gufo build complete: $BUILD_DIR/gufo"
log_info "Run it through: $LLM_ROOT/bin/gufo"
