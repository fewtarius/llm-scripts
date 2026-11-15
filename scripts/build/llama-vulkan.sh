#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Build llama.cpp with the Vulkan backend into build/llama-vulkan/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/build/common.sh
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="$LLM_ROOT/build/llama-vulkan"
LLAMA_DIR="$LLM_ROOT/llama.cpp"

usage() {
    cat <<'USAGE'
Usage: build/llama-vulkan.sh [OPTIONS]

Build llama.cpp with the Vulkan backend into build/llama-vulkan/.

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

mkdir -p "$BUILD_DIR"
if [[ "$CLEAN" == 1 ]]; then
    log_info "Cleaning Vulkan build directory"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# The ROCm SDK bundles the clang/lld this build prefers. It is optional for
# Vulkan: without it the system clang is used.
if [[ -d "$LLM_ROOT/deps/lib/llvm/bin" ]]; then
    export PATH="$LLM_ROOT/deps/lib/llvm/bin:$PATH"
fi

cd "$BUILD_DIR"

# -march=native enables microarchitecture-specific scheduling (Zen 3 AVX2
# dispatch, loop unrolling) that the generic -O3 build misses.
arch_flags=""
if clang -march=native -x c -c -o /dev/null /dev/null 2>/dev/null; then
    arch_flags="-march=native"
    log_info "Using -march=native for CPU-specific optimization"
fi

build_enable_ccache
build_cpu_flags

cmake "$LLAMA_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="$arch_flags" \
    -DCMAKE_CXX_FLAGS="$arch_flags" \
    -DGGML_HIP=OFF \
    -DGGML_HIPBLAS=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_VULKAN_USE_COOPMAT=ON \
    -DGGML_CPU=ON \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    "${BUILD_CPU_FLAGS[@]}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    "${BUILD_CCACHE_ARGS[@]}"

cmake --build . --config Release -j"$BUILD_JOBS"

log_ok "Vulkan build complete: $BUILD_DIR/bin/llama-server"
log_info "CPU ISA level: $SYS_CPU_ISA"
