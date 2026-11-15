#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# GPU/APU Auto-Detection Library
# =============================================================================
# Source this file (do not execute directly):
#   source lib/gpu.sh
#
# Provides auto-detected values for:
#   GPU_GFX_VERSION   - HSA_OVERRIDE_GFX_VERSION (e.g. 11.0.3)
#   GPU_GFX_ARCH      - GFX architecture (e.g. gfx1103)
#   GPU_NAME      - Human-readable GPU name (e.g. Radeon 780M)
#   GPU_PCI_ID    - PCI device ID (e.g. 1002:15bf)
#   GPU_ROCM_VARIANT  - ROCm SDK tarball variant (e.g. gfx110X)
#   SYS_THREADS       - Recommended thread count
#   SYS_RAM_GB  - Total system RAM in GB
#   GPU_GTT_GB - Recommended GTT size in GB
#   SYS_CPU_ISA       - CPU ISA level for cmake (e.g. "avx512_bf16", "avx2")
#   SYS_CMAKE_CPU_FLAGS - CMake flags for CPU features (e.g. "-DGGML_AVX512=ON ...")
#
# User overrides via environment:
#   LLM_GFX_VERSION=11.0.3    (skip detection, use this value)
#   LLM_GFX_ARCH=gfx1103      (skip detection, use this value)
#   LLM_THREADS=16            (override auto-detected thread count)
#   LLM_GTT_SIZE=18           (override recommended GTT size)
#   LLM_CPU_ISA=avx512_bf16   (skip detection, use this ISA level)
#
# Detection priority:
#   1. Environment variable overrides (LLM_GFX_VERSION, etc.)
#   2. amd-smi runtime detection (most accurate, resolves PCI ID collisions)
#   3. PCI ID lookup in GPU_MAP (fallback for systems without amd-smi)
#
# Extend the GPU map by editing the GPU_MAP below.

# Prevent double-sourcing
[[ -n "${_LLM_GPU_LOADED:-}" ]] && return 0
_LLM_GPU_LOADED=1

# =============================================================================
# AMD APU/GPU Map
# =============================================================================
# Format: "PCI_DEVICE_ID|GFX_ARCH|GFX_VERSION|GPU_NAME|TARBALL_VARIANT"
#
# TARBALL_VARIANT: the AMD nightly tarball family (gfx900, gfx103X, gfx110X, gfx120X)
# Used by build.sh to download the correct ROCm SDK.
#
# Some PCI IDs are shared across different GPU architectures.
# For example, 1002:1638 is used by both Van Gogh (gfx1033) and Cezanne (gfx90c),
# and 1002:1681 / 1002:1680 are used by both Rembrandt (gfx103x) and Strix Point
# (gfx1151). AMD reuses these IDs in silicon, so the table cannot simply be
# deduplicated.
#
# Resolution order:
#   1. amd-smi, which reports the authoritative TARGET_GRAPHICS_VERSION
#   2. the optional 6th GPU_MAP field, the CPU family that owns the variant.
#      Rembrandt is Zen 3 (family 25), Strix Point is Zen 5 (family 26), so the
#      host CPU separates them on machines without amd-smi.
#   3. the first matching entry, which is the previously published behaviour
#
# To add your device:
#   1. Find your PCI ID: lspci -nn | grep VGA
#   2. Find your GFX arch: amd-smi static --asic (look for TARGET_GRAPHICS_VERSION)
#   3. Add a line to GPU_MAP below
#
# Field layout: PCI_ID|gfx_arch|gfx_version|gpu_name|tarball_variant[|cpu_family]
# cpu_family is the decimal "cpu family" value from /proc/cpuinfo.
#
GPU_MAP=(
    # GCN5/Vega (gfx9 family) - use gfx900 tarball
    "1002:1638|gfx90c|9.0.0|Radeon Graphics (Cezanne)|gfx900|25"   # 5800H/5700G - NOTE: 1002:1638 is also Van Gogh (Zen 2, family 2)
    "1002:1636|gfx90c|9.0.0|Radeon Graphics (Renoir)|gfx900"       # 4800H/4700U etc.
    "1002:1635|gfx90c|9.0.0|Radeon Graphics (Renoir)|gfx900"       # 4600H/4600U etc.
    "1002:168d|gfx90c|9.0.0|Radeon Graphics (Lucienne)|gfx900"     # 5600H/5500U etc.

    # RDNA2 (gfx10.3 family) - use gfx103X tarball
    "1002:1435|gfx1032|10.3.2|Radeon Graphics (Sephiroth)|gfx103X"
    "1002:1681|gfx1036|10.3.6|Radeon 680M|gfx103X|25"              # Rembrandt (6800U), Zen 3
    "1002:1680|gfx1035|10.3.5|Radeon 660M|gfx103X|25"              # Rembrandt (6600U), Zen 3
    "1002:1506|gfx1036|10.3.6|Radeon 610M|gfx103X"                 # Mendocino

    # RDNA3 (gfx11 family) - use gfx110X tarball
    "1002:15bf|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Phoenix1 (7840U)
    "1002:15be|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Phoenix (7840HS)
    "1002:15c0|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Phoenix (7940HS)
    "1002:15e0|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Phoenix (7640U)
    "1002:15e1|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Phoenix (7540U)
    "1002:15c8|gfx1103|11.0.3|Radeon 890M|gfx110X"                 # Hawk Point (8840U)
    "1002:15c9|gfx1103|11.0.3|Radeon 890M|gfx110X"                 # Hawk Point (8945HS)
    "1002:15ca|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Hawk Point (8540U)
    "1002:15cb|gfx1103|11.0.3|Radeon 780M|gfx110X"                 # Hawk Point (8640U)

    # RDNA3.5 (gfx11.5 family) - use gfx120X tarball
    "1002:1681|gfx1151|11.5.1|Radeon 890M|gfx120X|26"              # Strix Point (HX 370), Zen 5
    "1002:1682|gfx1151|11.5.1|Radeon 880M|gfx120X"                 # Strix Point (HX 375)
    "1002:1586|gfx1151|11.5.1|Radeon 8060S|gfx120X"                # Strix Halo (Ryzen AI Max+ 395)
    "1002:1660|gfx1150|11.5.0|Radeon 8060S|gfx120X"                # Strix Halo alt SKU
    "1002:1680|gfx1151|11.5.1|Radeon 890M|gfx120X|26"              # Strix Point (365), Zen 5
)

# =============================================================================
# Detection functions
# =============================================================================

# Detect Apple Silicon GPU/chip on macOS
# Sets: GPU_NAME (e.g. "Apple M4 Pro"), GPU_GFX_ARCH ("metal"),
#        SYS_RAM_GB
# On non-macOS systems, returns 1 and leaves vars unchanged.
_detect_macos_gpu() {
    [[ "$(uname -s)" != "Darwin" ]] && return 1

    # Use sysctl for the chip brand string (always present, no dependencies)
    local chip=""
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) || return 1

    # Only Apple Silicon has the brand string starting with "Apple"
    if [[ "$chip" != Apple* ]]; then
        echo "[WARN] darwin: non-Apple CPU detected ($chip); macOS GPU acceleration requires Apple Silicon" >&2
        return 1
    fi

    GPU_NAME="$chip"
    # No GFX arch in the AMD sense - Metal is the unified GPU API
    GPU_GFX_ARCH="metal"
    GPU_GFX_VERSION=""
    GPU_PCI_ID=""
    GPU_ROCM_VARIANT=""
    return 0
}

# Get the AMD GPU PCI device ID
_detect_gpu_pci_id() {
    local pci_id=""

    # Method 1: /sys/class/drm (most reliable, no dependencies)
    for card in /sys/class/drm/card[0-9]/device/uevent; do
        [[ -f "$card" ]] || continue
        local driver=""
        local id=""
        while IFS='=' read -r key val; do
            case "$key" in
                DRIVER) driver="$val" ;;
                PCI_ID) id="$val" ;;
            esac
        done < "$card"
        if [[ "$driver" == "amdgpu" && -n "$id" ]]; then
            pci_id="$id"
            break
        fi
    done

    # Method 2: lspci fallback
    if [[ -z "$pci_id" ]] && command -v lspci &>/dev/null; then
        pci_id=$(lspci -nn 2>/dev/null | grep -m1 "VGA.*AMD" | grep -oP '\[\K[0-9a-f]{4}:[0-9a-f]{4}' | head -1)
    fi

    echo "${pci_id,,}"  # lowercase
}

# Detect GFX architecture using amd-smi (most accurate method)
# Resolves PCI ID collisions where the same ID maps to different architectures
_detect_gfx_via_amd_smi() {
    local amd_smi=""

    # Method 1: bundled amd-smi in deps
    local _project_root="${_LLM_PROJECT_ROOT:-}"
    if [[ -n "$_project_root" && -x "$_project_root/deps/bin/amd-smi" ]]; then
        amd_smi="$_project_root/deps/bin/amd-smi"
    fi

    # Method 2: system amd-smi
    if [[ -z "$amd_smi" ]]; then
        amd_smi=$(command -v amd-smi 2>/dev/null || true)
    fi

    # Method 3: rocm-smi fallback (older ROCm)
    if [[ -z "$amd_smi" ]]; then
        amd_smi=$(command -v rocm-smi 2>/dev/null || true)
    fi

    [[ -z "$amd_smi" ]] && return 1

    local gfx_version=""
    gfx_version=$("$amd_smi" static --asic 2>/dev/null \
        | grep -i "TARGET_GRAPHICS_VERSION" \
        | awk '{print $2}' | head -1)

    if [[ -n "$gfx_version" ]]; then
        echo "$gfx_version"
        return 0
    fi

    return 1
}

# Map a detected gfx architecture to its ROCm SDK tarball variant
_gfx_to_rocm_variant() {
    local gfx="$1"
    case "$gfx" in
        gfx9*)   echo "gfx900" ;;
        gfx101*) echo "gfx101X" ;;
        gfx103*) echo "gfx103X" ;;
        gfx110*) echo "gfx110X" ;;
        gfx1150) echo "gfx120X" ;;
        gfx1151) echo "gfx120X" ;;
        gfx1152) echo "gfx120X" ;;
        gfx1153) echo "gfx120X" ;;
        gfx120*) echo "gfx120X" ;;
        *)       echo "gfx110X" ;;  # default fallback
    esac
}

# Map a detected gfx architecture to its HSA_OVERRIDE_GFX_VERSION
_gfx_to_hsa_version() {
    local gfx="$1"
    # Strip any suffix like :xnack+, :xnack-
    gfx="${gfx%%:*}"
    case "$gfx" in
        gfx900)  echo "9.0.0" ;;
        gfx902)  echo "9.0.2" ;;
        gfx904)  echo "9.0.4" ;;
        gfx906)  echo "9.0.6" ;;
        gfx908)  echo "9.0.8" ;;
        gfx909)  echo "9.0.9" ;;
        gfx90a)  echo "9.0.10" ;;
        gfx90c)  echo "9.0.12" ;;
        gfx1010) echo "10.1.0" ;;
        gfx1012) echo "10.1.2" ;;
        gfx1030) echo "10.3.0" ;;
        gfx1031) echo "10.3.1" ;;
        gfx1032) echo "10.3.2" ;;
        gfx1033) echo "10.3.3" ;;
        gfx1034) echo "10.3.4" ;;
        gfx1035) echo "10.3.5" ;;
        gfx1036) echo "10.3.6" ;;
        gfx1100) echo "11.0.0" ;;
        gfx1101) echo "11.0.1" ;;
        gfx1102) echo "11.0.2" ;;
        gfx1103) echo "11.0.3" ;;
        gfx1150) echo "11.5.0" ;;
        gfx1151) echo "11.5.1" ;;
        gfx1152) echo "11.5.2" ;;
        gfx1200) echo "12.0.0" ;;
        gfx1201) echo "12.0.1" ;;
        *)       echo "0.0.0" ;;
    esac
}

# Host CPU family as printed by /proc/cpuinfo ("cpu family", decimal). Empty on
# non-x86 platforms or when the file is unavailable. Used to separate GPU
# variants that share a PCI device ID.
_detect_cpu_family() {
    [[ -r /proc/cpuinfo ]] || return 0
    awk -F': *' '/^cpu family[ \t]*:/ { gsub(/[ \t]/, "", $2); print $2; exit }' /proc/cpuinfo 2>/dev/null || true
}

# Look up GPU info from the map.
#
# When several entries share a PCI ID, prefer the one whose 6th field matches
# the host CPU family. Otherwise fall back, in order, to the first entry that
# carried no family constraint and then to the first entry for the ID at all,
# so a machine with an unreadable CPU family still resolves a GPU.
_lookup_gpu() {
    local pci_id="$1"
    local cpu_family="${SYS_CPU_FAMILY:-}"
    local unpinned="" first_any="" entry family
    local -a parts

    for entry in "${GPU_MAP[@]}"; do
        IFS='|' read -ra parts <<< "$entry"
        [[ "${parts[0]}" == "$pci_id" ]] || continue

        [[ -z "$first_any" ]] && first_any="$entry"

        if [[ ${#parts[@]} -ge 6 ]]; then
            family="${parts[5]}"
            if [[ -n "$cpu_family" && "$family" == "$cpu_family" ]]; then
                echo "$entry"
                return 0
            fi
            continue
        fi

        [[ -z "$unpinned" ]] && unpinned="$entry"
    done

    if [[ -n "$unpinned" ]]; then
        echo "$unpinned"
        return 0
    fi
    if [[ -n "$first_any" ]]; then
        echo "$first_any"
        return 0
    fi
    return 1
}

# Detect the APU's dedicated VRAM in GB (vis_vram_total / 1024^3).
# Returns 0 when no dedicated VRAM is reported (discrete GPU or unallocated).
_detect_apu_vram_gb() {
    local vram_bytes=0
    for f in /sys/class/drm/card[0-9]/device/mem_info_vram_total; do
        [[ -f "$f" ]] || continue
        vram_bytes=$(cat "$f" 2>/dev/null || echo 0)
        break
    done
    [[ -z "$vram_bytes" || "$vram_bytes" -eq 0 ]] && { echo 0; return; }
    echo $(( vram_bytes / 1024 / 1024 / 1024 ))
}

# Detect Strix Halo. Strix Halo ships with 64-128 GB of unified LPDDR5X
# memory and a 40-CU RDNA3.5 iGPU. The hallmark is the Radeon 8060S/8050S
# die on a Ryzen AI Max+ 395/390/388 SoC - a unique silicon combination.
# Detection key: PCI device ID is exclusive to Strix Halo (no other APU
# shares 1002:1586 or 1002:1660). GFX arch gfx1150/1/2 alone is not
# sufficient (Strix Point HX 370/375 is also gfx1151, but caps at 32 GB
# RAM), so we cross-check with total system RAM >= 64 GB which only
# Strix Halo boards reach.
#
# BIOS caveat: some firmware ships without the 96 GB VRAM carveout,
# leaving only 512 MiB BIOS-allocated and relying on GTT for the rest.
# This function ignores VRAM entirely - the PCI ID + RAM size signal is
# definitive. The result feeds GPU_IS_STRIX_HALO which the llama.cpp adapter
# uses to select the halo preset (f16 KV, big cache-ram).
_detect_strix_halo() {
    local pci_id="${GPU_PCI_ID:-}"
    local ram_gb="${SYS_RAM_GB:-0}"
    local gfx_arch="${GPU_GFX_ARCH:-}"

    case "$pci_id" in
        1002:1586|1002:1660)
            # Strix Halo-exclusive PCI IDs
            echo 1
            return 0
            ;;
    esac

    # Secondary signal: gfx1150/1/2 family + RAM >= 64 GB. Catches future
    # boards that might share the same silicon but get new PCI IDs.
    case "$gfx_arch" in
        gfx1150|gfx1151|gfx1152)
            if [[ "$ram_gb" -ge 64 ]]; then
                echo 1
                return 0
            fi
            ;;
    esac

    echo 0
    return 0
}

# Classify the hardware into a tier that downstream scripts can branch on.
# Tier is set from the detected APU VRAM carveout (the most reliable signal
# for how much GPU memory is actually available to the iGPU). For mini-PCs
# with a tiny VRAM carveout but desktop-class system RAM (e.g. Cezanne
# 5800H with 512 MiB carveout and 30 GB RAM), we fall back to total RAM to
# avoid misclassifying them as handhelds.
#
#   handheld  - <=16GB APU VRAM AND <24GB system RAM
#                (Phoenix/Hawk Point, 780M/890M; Steam Deck class)
#   standard  - 16-64GB APU VRAM, OR mini-PC class (>=24GB system RAM
#                with sub-GB VRAM carveout - e.g. Minisforum UM580+)
#   halo      - >=64GB APU VRAM (Strix Halo with 96GB BIOS carveout)
#
# Override with SYS_TIER env var.
_detect_hardware_tier() {
    local vram_gb="$1"
    local ram_gb="${2:-0}"
    if [[ "${GPU_IS_STRIX_HALO:-0}" == "1" ]]; then
        echo "halo"
        return 0
    fi
    if [[ "$vram_gb" -ge 64 ]]; then
        echo "halo"
    elif [[ "$vram_gb" -ge 16 ]]; then
        echo "standard"
    elif [[ "$vram_gb" -lt 1 && "$ram_gb" -ge 24 ]]; then
        # Mini-PC class: tiny BIOS VRAM carveout but desktop-class RAM.
        # zaphod (Minisforum UM580+, 512 MiB carveout, 30 GB RAM) hits this
        # branch and gets the standard profile instead of being misclassified
        # as a Steam Deck. The 7840U (6 GB VRAM) is unaffected because its
        # VRAM is well above the 1 GB threshold.
        echo "standard"
    else
        echo "handheld"
    fi
}
_detect_total_ram_gb() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # macOS: hw.memsize reports physical RAM in bytes
        local ram_bytes
        ram_bytes=$(sysctl -n hw.memsize 2>/dev/null)
        [[ -z "$ram_bytes" ]] && { echo 0; return; }
        echo $(( ram_bytes / 1024 / 1024 / 1024 ))
    else
        # Linux: /proc/meminfo reports in kB
        local ram_kb
        ram_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
        echo $((ram_kb / 1024 / 1024))
    fi
}

# Recommend GTT size based on total RAM and current VRAM allocation.
#
# Strategy depends on whether VRAM has been pre-allocated (e.g. Strix Halo
# BIOS-level 96GB carveout). When the APU already has dedicated VRAM, GTT
# is only needed as a spillover and should stay small. Otherwise reserve
# 6GB for OS and use the rest as GTT.
#
# Users can override with LLM_GTT_SIZE env var.
_recommend_gtt_gb() {
    local ram_gb="$1"
    local vram_gb="$2"
    local os_reserve=6

    # If APU already has a large dedicated VRAM pool (e.g. Strix Halo with
    # 96GB BIOS carveout), GTT is only useful as overflow. Default to 4GB
    # so the GART table doesn't waste address space.
    if [[ -n "$vram_gb" && "$vram_gb" -ge 32 ]]; then
        echo 4
        return
    fi

    local gtt=$((ram_gb - os_reserve))
    # Minimum 4GB GTT for systems without dedicated VRAM
    (( gtt < 4 )) && gtt=4
    echo "$gtt"
}

# =============================================================================
# CPU ISA Detection
# =============================================================================
# Detects the highest x86 ISA level supported by the CPU and generates
# the corresponding CMake flags for ggml-cpu.
#
# ISA levels (in order of preference):
#   avx512_bf16  - AVX-512 with BF16 + VNNI (Zen 4, Sapphire Rapids, etc.)
#   avx512_vnni  - AVX-512 with VNNI but no BF16
#   avx512       - Base AVX-512 (F, CD, VL, DQ, BW)
#   avx2          - AVX2 + FMA + F16C + BMI2
#   avx           - AVX without AVX2
#   sse42         - SSE4.2 baseline
#
# The detection reads /proc/cpuinfo on Linux and uses sysctl on macOS.
# On macOS (Apple Silicon), we skip x86 detection entirely.
#
# Output variables:
#   SYS_CPU_ISA         - Highest ISA level string (e.g. "avx512_bf16")
#   SYS_CMAKE_CPU_FLAGS - CMake flags string (e.g. "-DGGML_AVX512=ON -DGGML_AVX512_BF16=ON ...")

_detect_cpu_isa() {
    # macOS Apple Silicon: no x86 SIMD detection needed
    if [[ "$SYS_PLATFORM" == "macos" ]]; then
        SYS_CPU_ISA="apple_silicon"
        SYS_CMAKE_CPU_FLAGS=""
        return 0
    fi

    # Linux: read CPU flags from /proc/cpuinfo
    local cpuflags=""
    cpuflags=$(grep -m1 "^flags" /proc/cpuinfo 2>/dev/null | cut -d: -f2)

    if [[ -z "$cpuflags" ]]; then
        # Fallback: assume AVX2 (safe for any modern x86-64 CPU)
        SYS_CPU_ISA="avx2"
        SYS_CMAKE_CPU_FLAGS="-DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_SSE42=ON"
        return 0
    fi

    # Check ISA levels from highest to lowest
    # AVX-512 BF16 requires: avx512f, avx512bw, avx512cd, avx512dq, avx512vl, avx512_bf16
    local has_avx512f=0 has_avx512bw=0 has_avx512cd=0 has_avx512dq=0 has_avx512vl=0
    local has_avx512_bf16=0 has_avx512_vnni=0 has_avx512_vbmi=0
    local has_avx2=0 has_avx=0 has_fma=0 has_f16c=0 has_bmi2=0 has_sse42=0

    # Parse flags
    for flag in $cpuflags; do
        case "$flag" in
            avx512f)      has_avx512f=1 ;;
            avx512bw)     has_avx512bw=1 ;;
            avx512cd)     has_avx512cd=1 ;;
            avx512dq)     has_avx512dq=1 ;;
            avx512vl)     has_avx512vl=1 ;;
            avx512_bf16)  has_avx512_bf16=1 ;;
            avx512_vnni)  has_avx512_vnni=1 ;;
            avx512vbmi)   has_avx512_vbmi=1 ;;
            avx2)         has_avx2=1 ;;
            avx)          has_avx=1 ;;
            fma)          has_fma=1 ;;
            f16c)         has_f16c=1 ;;
            bmi2)         has_bmi2=1 ;;
            sse4_2)       has_sse42=1 ;;
        esac
    done

    # Determine highest ISA level and build CMake flags
    local flags=""

    # Always include baseline
    if [[ "$has_sse42" -eq 1 ]]; then
        flags="-DGGML_SSE42=ON"
    fi
    if [[ "$has_avx" -eq 1 ]]; then
        flags="$flags -DGGML_AVX=ON"
    fi
    if [[ "$has_fma" -eq 1 ]]; then
        flags="$flags -DGGML_FMA=ON"
    fi
    if [[ "$has_f16c" -eq 1 ]]; then
        flags="$flags -DGGML_F16C=ON"
    fi
    if [[ "$has_bmi2" -eq 1 ]]; then
        flags="$flags -DGGML_BMI2=ON"
    fi
    if [[ "$has_avx2" -eq 1 ]]; then
        flags="$flags -DGGML_AVX2=ON"
    fi

    # AVX-512 base: requires F, CD, VL, DQ, BW
    local has_avx512_base=0
    if [[ "$has_avx512f" -eq 1 && "$has_avx512cd" -eq 1 && \
          "$has_avx512vl" -eq 1 && "$has_avx512dq" -eq 1 && \
          "$has_avx512bw" -eq 1 ]]; then
        has_avx512_base=1
        flags="$flags -DGGML_AVX512=ON"
    fi

    # AVX-512 extensions (only meaningful if base AVX-512 is present)
    if [[ "$has_avx512_base" -eq 1 ]]; then
        if [[ "$has_avx512_vbmi" -eq 1 ]]; then
            flags="$flags -DGGML_AVX512_VBMI=ON"
        fi
        if [[ "$has_avx512_vnni" -eq 1 ]]; then
            flags="$flags -DGGML_AVX512_VNNI=ON"
        fi
        if [[ "$has_avx512_bf16" -eq 1 ]]; then
            flags="$flags -DGGML_AVX512_BF16=ON"
        fi
    fi

    # Determine ISA level string (highest first)
    if [[ "$has_avx512_base" -eq 1 && "$has_avx512_bf16" -eq 1 ]]; then
        SYS_CPU_ISA="avx512_bf16"
    elif [[ "$has_avx512_base" -eq 1 && "$has_avx512_vnni" -eq 1 ]]; then
        SYS_CPU_ISA="avx512_vnni"
    elif [[ "$has_avx512_base" -eq 1 ]]; then
        SYS_CPU_ISA="avx512"
    elif [[ "$has_avx2" -eq 1 ]]; then
        SYS_CPU_ISA="avx2"
    elif [[ "$has_avx" -eq 1 ]]; then
        SYS_CPU_ISA="avx"
    else
        SYS_CPU_ISA="sse42"
    fi

    # Strip leading space
    flags="${flags# }"
    SYS_CMAKE_CPU_FLAGS="$flags"
}

# =============================================================================
# Main detection (runs on source)
# =============================================================================

# Store project root for amd-smi lookup
_LLM_PROJECT_ROOT="${_LLM_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

# Platform identifier: "macos" on Darwin, "linux" elsewhere.
# Downstream scripts branch on this rather than calling uname repeatedly.
case "$(uname -s)" in
    Darwin) SYS_PLATFORM="macos" ;;
    *)      SYS_PLATFORM="linux" ;;
esac

# macOS path: short-circuit AMD detection, set Apple Silicon identifiers
if _detect_macos_gpu; then
    : # values already set by _detect_macos_gpu
else
# Detect PCI ID
GPU_PCI_ID="$(_detect_gpu_pci_id)"

# CPU family, used to disambiguate GPU variants that share a PCI device ID
SYS_CPU_FAMILY="$(_detect_cpu_family)"
export SYS_CPU_FAMILY

# Look up in map (provides defaults, may be overridden by amd-smi)
GPU_GFX_ARCH=""
GPU_GFX_VERSION=""
GPU_NAME=""
GPU_ROCM_VARIANT=""

if [[ -n "$GPU_PCI_ID" ]]; then
    _map_entry="$(_lookup_gpu "$GPU_PCI_ID")" || true
    if [[ -n "$_map_entry" ]]; then
        # Parse: pci_id|gfx_arch|gfx_version|gpu_name|tarball_variant
        GPU_GFX_ARCH=$(echo "$_map_entry" | cut -d'|' -f2)
        GPU_GFX_VERSION=$(echo "$_map_entry" | cut -d'|' -f3)
        GPU_NAME=$(echo "$_map_entry" | cut -d'|' -f4)
        GPU_ROCM_VARIANT=$(echo "$_map_entry" | cut -d'|' -f5)
    fi
fi

# Override with amd-smi detection when available (resolves PCI ID collisions)
_detected_gfx="$(_detect_gfx_via_amd_smi)" || true
if [[ -n "$_detected_gfx" ]]; then
    # amd-smi provides the authoritative gfx architecture
    GPU_GFX_ARCH="$_detected_gfx"
    GPU_GFX_VERSION="$(_gfx_to_hsa_version "$_detected_gfx")"
    GPU_ROCM_VARIANT="$(_gfx_to_rocm_variant "$_detected_gfx")"
    # Keep GPU_NAME from map if available, otherwise derive from gfx arch
    if [[ -z "$GPU_NAME" ]]; then
        GPU_NAME="AMD GPU ($_detected_gfx)"
    fi
fi
fi  # end macOS/AMD branch

# Detect system resources
SYS_RAM_GB="$(_detect_total_ram_gb)"
GPU_VRAM_GB="$(_detect_apu_vram_gb)"
GPU_GTT_GB="$(_recommend_gtt_gb "$SYS_RAM_GB" "$GPU_VRAM_GB")"
GPU_IS_STRIX_HALO="$(_detect_strix_halo)"
SYS_TIER="$(_detect_hardware_tier "$GPU_VRAM_GB" "$SYS_RAM_GB")"

# Detect CPU ISA features (x86 SIMD level for cmake build flags)
_detect_cpu_isa

# Thread count: use nproc by default. The solver (optimize.sh
# apply_user_overrides) splits this into batch/gen (nproc for batch,
# nproc/2 for generation) - that is the only halving applied.
# Hardware tier does NOT adjust thread count: all tiers should be
# optimized for best performance. Handheld-class systems (Steam Deck,
# etc.) still benefit from full nproc for batch and nproc/2 for gen,
# which already leaves headroom for the OS.
SYS_THREADS=$(nproc 2>/dev/null || echo 4)

# =============================================================================
# Apply user overrides (environment variables take precedence)
# =============================================================================

if [[ -n "${LLM_GFX_VERSION:-}" ]]; then
    GPU_GFX_VERSION="$LLM_GFX_VERSION"
fi

if [[ -n "${LLM_GFX_ARCH:-}" ]]; then
    GPU_GFX_ARCH="$LLM_GFX_ARCH"
fi

if [[ -n "${LLM_THREADS:-}" ]]; then
    SYS_THREADS="$LLM_THREADS"
fi

if [[ -n "${LLM_GTT_SIZE:-}" ]]; then
    GPU_GTT_GB="$LLM_GTT_SIZE"
fi

if [[ -n "${LLM_HARDWARE_TIER:-}" ]]; then
    SYS_TIER="$LLM_HARDWARE_TIER"
fi
# Honor LLM_IS_STRIX_HALO for testing (e.g. Apple Silicon with
# 64+ GB unified memory, where the unified-memory heuristic should still
# pick "halo"-equivalent settings but the auto-detect sees gfx!=1150/1/2).
if [[ -n "${LLM_IS_STRIX_HALO:-}" ]]; then
    GPU_IS_STRIX_HALO="$LLM_IS_STRIX_HALO"
fi

if [[ -n "${LLM_CPU_ISA:-}" ]]; then
    SYS_CPU_ISA="$LLM_CPU_ISA"
fi

# =============================================================================
# Effective GPU memory budget
# =============================================================================
# The pool an engine can actually allocate: VRAM carveout + GTT, less a
# staging reserve, and never more than the OS can give up. The solver budgets
# against this, and `llm doctor` reports it. One formula, one place.
gpu_budget_bytes() {
    if [[ -n "${LLM_GPU_BUDGET_BYTES:-}" ]]; then
        printf '%s' "$LLM_GPU_BUDGET_BYTES"
        return 0
    fi
    local vram=0 gtt=0 c
    for c in /sys/class/drm/card[0-9]/device; do
        [[ -d "$c" ]] || continue
        vram=$(cat "$c/mem_info_vram_total" 2>/dev/null || echo 0)
        gtt=$(cat "$c/mem_info_gtt_total" 2>/dev/null || echo 0)
        break
    done
    if (( vram + gtt == 0 )); then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            printf '%s' "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) ))"
        else
            printf '0'
        fi
        return 0
    fi
    local vram_gib=$(( vram / 1073741824 ))
    local total_gib="${SYS_RAM_GB:-1}"
    (( total_gib > 0 )) || total_gib=1
    local os_reserve=4
    (( total_gib <= 16 )) && os_reserve=$(( total_gib / 4 ))
    local unified=$(( (vram + gtt - 512 * 1048576) / 1073741824 ))
    local limit=$(( vram_gib + total_gib - os_reserve ))
    (( unified > limit )) && unified=$limit
    (( unified < 1 )) && unified=1
    printf '%s' $(( unified * 1073741824 ))
}

# Actual memory the driver reports, for reporting rather than budgeting.
gpu_reported_pool_gib() {
    local vram=0 gtt=0 c
    for c in /sys/class/drm/card[0-9]/device; do
        [[ -d "$c" ]] || continue
        vram=$(cat "$c/mem_info_vram_total" 2>/dev/null || echo 0)
        gtt=$(cat "$c/mem_info_gtt_total" 2>/dev/null || echo 0)
        break
    done
    awk -v v="$vram" -v g="$gtt" 'BEGIN{ printf "%.1f", (v+g)/1073741824 }'
}

# =============================================================================
# Export for use by other scripts
# =============================================================================

export GPU_GFX_VERSION GPU_GFX_ARCH GPU_NAME GPU_PCI_ID GPU_ROCM_VARIANT
export SYS_THREADS SYS_RAM_GB GPU_VRAM_GB GPU_GTT_GB SYS_TIER
export GPU_IS_STRIX_HALO
export SYS_CPU_ISA SYS_CMAKE_CPU_FLAGS
export SYS_PLATFORM

# Unset locals
unset _map_entry _detected_gfx
