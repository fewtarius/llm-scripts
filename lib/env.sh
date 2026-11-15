#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# Runtime environment for every engine.
#
# This is the only place that knows where the bundled ROCm SDK lives, how to
# reach the Vulkan driver tuning, or how to merge the driconf entry. Runners
# call llm_env_activate; users are never asked to export anything.

[[ -n "${_LLM_ENV_LOADED:-}" ]] && return 0
_LLM_ENV_LOADED=1

_LLM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_LLM_ENV_DIR/common.sh"
# shellcheck source=lib/gpu.sh
source "$_LLM_ENV_DIR/gpu.sh"

LLM_DEPS_DIR="${LLM_DEPS_DIR:-$LLM_ROOT/deps}"

# ---------------------------------------------------------------- Vulkan ----
_llm_env_vulkan_tuning() {
    export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-2G}"
    export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-$HOME/.cache/mesa_shader_cache}"
    mkdir -p "$MESA_SHADER_CACHE_DIR" 2>/dev/null || true

    export RADV_PERFTEST="${RADV_PERFTEST:-gplp}"
    export GGML_VK_MAX_MB_PER_SUBMIT="${GGML_VK_MAX_MB_PER_SUBMIT:-0}"
    export GGML_VK_NODES_PER_SUBMIT="${GGML_VK_NODES_PER_SUBMIT:-100}"
    export GGML_VK_FA_WAVE32="${GGML_VK_FA_WAVE32:-1}"
}

# DRIRC: on APUs RADV exposes two thirds of VRAM+GTT as DEVICE_LOCAL unless
# the unified-heap option is set. Mesa only reads driconf per application, so
# each engine binary is listed. Existing user entries are preserved.
_LLM_DRIRC_HEAD='<?xml version="1.0"?>
<!DOCTYPE driconf SYSTEM "http://www.freedesktop.org/software/mesa/driconf.dtd">'
_LLM_DRIRC_APPS='        <application name="gufo" application_name_match="gufo">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-server" application_name_match="llama-server">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-cli" application_name_match="llama-cli">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-bench" application_name_match="llama-bench">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-batched-bench" application_name_match="llama-batched-bench">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>'
_LLM_DRIRC_BLOCK="    <device driver=\"radv\">
${_LLM_DRIRC_APPS}
    </device>"

ensure_drirc_unified_heap() {
    local drirc="${HOME}/.drirc"
    [[ -n "$drirc" && -d "$(dirname "$drirc")" ]] || {
        log_warn "Cannot write ${drirc}: home directory unavailable"
        return 0
    }
    if [[ ! -s "$drirc" ]] || ! grep -q '<driconf' "$drirc" 2>/dev/null; then
        [[ -s "$drirc" ]] && cp "$drirc" "${drirc}.bak-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
        printf '%s\n<driconf>\n%s\n</driconf>\n' "$_LLM_DRIRC_HEAD" "$_LLM_DRIRC_BLOCK" > "$drirc" || {
            log_warn "Could not write ${drirc}; RADV unified heap not enabled"
            return 0
        }
        log_info "Enabled RADV unified heap on APU in ${drirc}"
        return 0
    fi
    grep -q 'radv_enable_unified_heap_on_apu' "$drirc" 2>/dev/null && return 0
    local tmp
    tmp=$(mktemp "${drirc}.tmp.XXXXXX") || return 0
    awk -v block="$_LLM_DRIRC_BLOCK" '
        !inserted && /<\/driconf>/ { print block; print $0; inserted=1; next }
        { print }
        END { if (!inserted) { print block; print "</driconf>" } }
    ' "$drirc" > "$tmp" || { rm -f "$tmp"; return 0; }
    cp "$drirc" "${drirc}.bak-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
    if mv "$tmp" "$drirc"; then
        log_info "Added RADV unified heap entries to existing ${drirc} (backup saved)"
    else
        rm -f "$tmp"
        log_warn "Could not update ${drirc}; RADV unified heap not enabled"
    fi
}

# ------------------------------------------------------------- Platform -----
# "low-power" throttles the iGPU to ~800 MHz. Without the performance profile
# the engine looks slow and the solver's numbers are not reproducible.
ensure_platform_performance() {
    local file="/sys/firmware/acpi/platform_profile"
    [[ -f "$file" ]] || return 0
    local current
    current=$(cat "$file" 2>/dev/null || true)
    [[ "$current" == "performance" ]] && return 0
    if echo performance > "$file" 2>/dev/null; then
        log_ok "Set ACPI platform profile to performance (was: ${current:-unknown})"
        return 0
    fi
    if llm_have sudo && sudo -n bash -c "echo performance > $file" 2>/dev/null; then
        log_ok "Set ACPI platform profile to performance via sudo (was: ${current:-unknown})"
        return 0
    fi
    if [[ "$current" == "low-power" ]]; then
        log_warn "ACPI platform profile is 'low-power' and could not be changed"
        log_warn "  The iGPU stays throttled to ~800 MHz: expect 20-30% lower throughput."
        log_warn "  Fix with: echo performance | sudo tee $file"
    fi
    return 1
}

# --------------------------------------------------------------- ROCm -------
_llm_env_rocm() {
    export ROCM_PATH="${LLM_ROCM_PATH:-$LLM_DEPS_DIR}"
    export HIP_PATH="$ROCM_PATH"
    export HIP_PLATFORM=amd
    export HSA_PATH="$ROCM_PATH"
    export ROCM_DIR="$ROCM_PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/llvm/lib/clang/23/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
    [[ -n "${GPU_GFX_VERSION:-}" ]] && export HSA_OVERRIDE_GFX_VERSION="$GPU_GFX_VERSION"
    export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
    return 0
}

_llm_env_vulkan() {
    # The ROCm SDK ships libraries some builds link even in Vulkan mode.
    export LD_LIBRARY_PATH="$LLM_DEPS_DIR/lib:${LD_LIBRARY_PATH:-}"
    export GGML_BACKEND=vulkan
    if [[ -z "${LLM_SKIP_UNIFIED_HEAP:-}" ]]; then
        ensure_drirc_unified_heap
    fi
    if [[ -z "${LLM_SKIP_GPU_PERF:-}" ]]; then
        _llm_env_vulkan_tuning
    fi
    return 0
}

_llm_env_metal() {
    export GGML_METAL_DEVICE_DEBUG=0
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    return 0
}

# Activate the environment for an engine/backend pair.
#   llm_env_activate vulkan
#   llm_env_activate rocm
#   llm_env_activate metal
# gufo always needs the HIP runtime libraries, so the gufo adapter passes
# "rocm" here regardless of its display backend.
llm_env_activate() {
    local backend="${1:-vulkan}"
    case "$backend" in
        rocm)   _llm_env_rocm ;;
        vulkan) _llm_env_vulkan ;;
        metal)  _llm_env_metal ;;
        cpu)    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" ;;
        *)      log_warn "Unknown backend '$backend'; using Vulkan"; _llm_env_vulkan ;;
    esac
    if [[ -z "${LLM_SKIP_GPU_PERF:-}" ]]; then
        ensure_platform_performance || true
    fi
    return 0
}

# Wrappers so the engine binaries work without a shell preamble.
llm_env_write_wrappers() {
    local dir="$LLM_ROOT/bin"
    mkdir -p "$dir" || return 1
    cat > "$dir/gufo" <<'WRAP'
#!/bin/sh
exec "$(dirname "$(readlink -f "$0")")/../llm" _exec gufo "$@"
WRAP
    cat > "$dir/llama-server" <<'WRAP'
#!/bin/sh
exec "$(dirname "$(readlink -f "$0")")/../llm" _exec llama.cpp server "$@"
WRAP
    chmod +x "$dir/gufo" "$dir/llama-server"
}
