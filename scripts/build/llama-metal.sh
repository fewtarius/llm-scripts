#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Build llama.cpp with the Metal backend into build/llama-metal/ (macOS only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/build/common.sh
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="$LLM_ROOT/build/llama-metal"
LLAMA_DIR="$LLM_ROOT/llama.cpp"

usage() {
    cat <<'USAGE'
Usage: build/llama-metal.sh [OPTIONS]

Build llama.cpp with the Metal backend into build/llama-metal/ (macOS only).
Requires Xcode Command Line Tools: xcode-select --install

OPTIONS:
    --clean     Wipe the build directory first
    -h, --help  Show this help
USAGE
}

CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage >&2; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    log_error "Metal backend is macOS only (detected: $(uname -s))"
    exit 1
fi

if ! command -v clang++ &>/dev/null; then
    log_error "clang++ not found. Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

mkdir -p "$BUILD_DIR"
if [[ "$CLEAN" == 1 ]]; then
    log_info "Cleaning Metal build directory"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# GGML_METAL defaults to ON on Apple platforms, but the configure step is
# self-documenting when every backend switch is explicit.
build_enable_ccache

cmake "$LLAMA_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DGGML_METAL=ON \
    -DGGML_METAL_NDEBUG=ON \
    -DGGML_HIP=OFF \
    -DGGML_HIPBLAS=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_CPU=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    "${BUILD_CCACHE_ARGS[@]}"

cmake --build . --config Release -j"$BUILD_JOBS"

log_ok "Metal build complete: $BUILD_DIR/bin/llama-server"
