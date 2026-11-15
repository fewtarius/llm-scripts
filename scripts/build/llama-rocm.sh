#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Build llama.cpp with the ROCm/HIP backend into build/llama-rocm/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/build/common.sh
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="$LLM_ROOT/build/llama-rocm"
LLAMA_DIR="$LLM_ROOT/llama.cpp"

usage() {
    cat <<'USAGE'
Usage: build/llama-rocm.sh [OPTIONS]

Build llama.cpp with the ROCm/HIP backend into build/llama-rocm/.
The ROCm SDK must already be installed in deps/ (./scripts/build.sh --sdk-only).

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

build_require_rocm || exit 1

mkdir -p "$BUILD_DIR"
if [[ "$CLEAN" == 1 ]]; then
    log_info "Cleaning ROCm build directory"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# shellcheck source=lib/env.sh
source "$LLM_ROOT/lib/env.sh"
llm_env_activate rocm

cd "$BUILD_DIR"

hip_arch="${GPU_GFX_ARCH:-gfx1103}"
log_info "HIP target architecture: $hip_arch"

build_enable_ccache hip
build_cpu_flags

cmake "$LLAMA_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_PLATFORM=amd \
    -DCMAKE_HIP_ARCHITECTURES="$hip_arch" \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH/lib/llvm" \
    -DGGML_HIP=ON \
    -DGGML_HIPBLAS=ON \
    -DGGML_HIP_NO_VMM=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_CPU=ON \
    -DGGML_NATIVE=OFF \
    "${BUILD_CPU_FLAGS[@]}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    "${BUILD_CCACHE_ARGS[@]}"

cmake --build . --config Release -j"$BUILD_JOBS"

log_ok "ROCm build complete: $BUILD_DIR/bin/llama-server"
log_info "CPU ISA level: $SYS_CPU_ISA"
