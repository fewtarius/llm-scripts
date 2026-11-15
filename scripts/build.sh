#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Rebuild Script - Downloads ROCm SDK + Builds llama.cpp
# =============================================================================
# Handles full setup from a fresh checkout:
#   Linux + AMD:  Vulkan (default) or ROCm/HIP - downloads ROCm SDK
#   macOS (Apple Silicon): Metal backend - uses Xcode toolchain
#
# Usage:
#   ./scripts/build.sh               # Default backend for this platform
#   ./scripts/build.sh --rebuild     # Full rebuild (wipe deps/, re-download SDK)
#   ./scripts/build.sh --rocm        # Build ROCm backend only (Linux)
#   ./scripts/build.sh --vulkan      # Build Vulkan backend only (Linux)
#   ./scripts/build.sh --metal       # Build Metal backend (macOS)
#   ./scripts/build.sh --both        # Build both ROCm and Vulkan (Linux)
#   ./scripts/build.sh --help        # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$LLM_ROOT"
DEPS_DIR="$LLM_ROOT/deps"

# AMD stable ROCm pip repository
ROCM_PIP_INDEX="https://stable.repo.amd.com/rocm/whl-next/"

# ROCm SDK package version (stable)
ROCM_SDK_VERSION="${ROCM_SDK_VERSION:-10.1.0}"

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"
# shellcheck source=lib/gpu.sh
source "$LLM_ROOT/lib/gpu.sh"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build llama.cpp and/or gufo with the appropriate backend for the host platform.
  - Linux + AMD: downloads ROCm SDK, builds ROCm and/or Vulkan
  - macOS (Apple Silicon): builds Metal backend

OPTIONS:
    --rocm          Build ROCm/HIP backend only (Linux + AMD)
    --vulkan        Build Vulkan backend only (Linux + AMD)
    --metal         Build Metal backend (macOS)
    --both          Build both ROCm and Vulkan (Linux)
    --gufo          Build gufo (Linux + AMD, ROCm/HIP)
    --sdk-only      Install the ROCm SDK into deps/ and stop
    --rebuild       Full rebuild (wipe deps/, re-download SDK)
    --clean         Clean build directories only (keep deps)
    -h, --help      Show this help

ENVIRONMENT:
    ROCM_SDK_VERSION  Stable ROCm SDK version (default: $ROCM_SDK_VERSION)

EXAMPLES:
    $(basename "$0")                    # Build default backend for this platform
    $(basename "$0") --rebuild          # Full rebuild from scratch
    $(basename "$0") --rocm             # Build ROCm only (Linux)
    $(basename "$0") --vulkan           # Build Vulkan only (Linux)
    $(basename "$0") --metal            # Build Metal (macOS)
    $(basename "$0") --gufo             # Build gufo (Linux + AMD, ROCm/HIP)

EOF
    exit 0
}

# =============================================================================
# Parse arguments and set platform-appropriate defaults
# =============================================================================

# Platform defaults: macOS -> Metal only; Linux -> Vulkan only.
case "$(uname -s)" in
    Darwin)
        BUILD_ROCM=false
        BUILD_VULKAN=false
        BUILD_METAL=true
        ;;
    *)
        BUILD_ROCM=false
        BUILD_VULKAN=true
        BUILD_METAL=false
        ;;
esac

CLEAN=false
REBUILD=false
BUILD_GUFO=false
SDK_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm)
            BUILD_VULKAN=false
            BUILD_METAL=false
            BUILD_ROCM=true
            shift
            ;;
        --vulkan)
            BUILD_ROCM=false
            BUILD_METAL=false
            BUILD_VULKAN=true
            shift
            ;;
        --metal)
            BUILD_ROCM=false
            BUILD_VULKAN=false
            BUILD_METAL=true
            shift
            ;;
        --both)
            BUILD_ROCM=true
            BUILD_VULKAN=true
            BUILD_METAL=false
            BUILD_GUFO=false
            shift
            ;;
        --gufo)
            BUILD_ROCM=false
            BUILD_VULKAN=false
            BUILD_METAL=false
            BUILD_GUFO=true
            shift
            ;;
        --sdk-only)
            # Install the ROCm SDK into deps/ and stop. The gufo build needs
            # the same SDK but compiles its own HIP backend, so it has no
            # llama.cpp build to trigger.
            SDK_ONLY=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Prerequisite checks
# =============================================================================

check_prereqs() {
    local missing=()

    # --- Build tools ---
    # cmake: can be system-installed, pip-installed, or bundled with ROCm LLVM
    local has_cmake=false
    for cmake_bin in cmake "$HOME/.local/bin/cmake" "$LLM_ROOT/deps/lib/llvm/bin/cmake"; do
        if command -v "$cmake_bin" &>/dev/null; then
            has_cmake=true
            break
        fi
    done
    if [[ "$has_cmake" == false ]]; then
        missing+=("cmake")
    fi

    # make: required by cmake's Makefile generator (Linux only)
    # macOS uses ninja or the default Xcode generator; ninja is checked later
    if [[ "$(uname -s)" != "Darwin" ]]; then
        if ! command -v make &>/dev/null; then
            missing+=("make")
        fi
    fi

    # curl: used for downloading ROCm SDK; git: required tooling
    for tool in curl git; do
        if ! command -v "$tool" &>/dev/null; then
            missing+=("$tool")
        fi
    done

    # ccache is optional. The backend scripts pass it to cmake as the compiler
    # launcher when it resolves, so a missing ccache costs rebuild time but
    # never fails the build.
    if ! command -v ccache &>/dev/null; then
        log_warn "ccache not found; rebuilds will recompile everything"
        log_warn "Install it with ./scripts/install-deps.sh"
    fi

    # --- C/C++ compiler ---
    # Priority: ROCm bundled clang > system clang > system gcc (Linux)
    # On macOS, prefer Apple's clang (from Xcode Command Line Tools)
    local has_compiler=false
    for compiler in clang++ g++; do
        if command -v "$compiler" &>/dev/null; then
            has_compiler=true
            break
        fi
    done
    if [[ "$has_compiler" == false ]]; then
        missing+=("clang++ or g++ (C++ compiler)")
    fi

    # --- GCC runtime libraries (Linux ROCm builds) ---
    # The ROCm SDK's bundled clang links against system libgcc/libstdc++.
    # On minimal distros (SteamOS, etc.) GCC may not be installed.
    if [[ "$(uname -s)" != "Darwin" ]]; then
        if ! command -v gcc &>/dev/null; then
            # Check if libgcc is available (some distros ship libgcc_s without gcc)
            if ! ldconfig -p 2>/dev/null | grep -q libgcc; then
                missing+=("gcc (provides libgcc/libstdc++ needed by ROCm clang linker)")
            fi
        fi
    fi

   # --- Vulkan-specific ---
    if [[ "$BUILD_VULKAN" == true ]]; then
        # Vulkan shader compilation needs glslc or glslangValidator
        if ! command -v glslc &>/dev/null && ! command -v glslangValidator &>/dev/null; then
            missing+=("glslc (vulkan-shaders)")
        fi
    fi

    # --- Metal-specific (macOS) ---
    if [[ "$BUILD_METAL" == true ]]; then
        if ! command -v xcrun &>/dev/null; then
            missing+=("xcrun (Xcode Command Line Tools)")
        fi
    fi

    # Report any missing prerequisites
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for m in "${missing[@]}"; do
            echo "  - $m"
        done
        if [[ "$(uname -s)" == "Darwin" ]]; then
            echo ""
            echo "Install Xcode Command Line Tools:"
            echo "  xcode-select --install"
        fi
        exit 1
    fi
}

# =============================================================================
# GPU detection
# =============================================================================

source "$LLM_ROOT/lib/gpu.sh"

# Show the detected platform/GPU
if [[ "$(uname -s)" == "Darwin" ]]; then
    log_info "Platform: macOS (${GPU_NAME:-unknown})"
    log_info "Metal acceleration enabled via Apple Silicon GPU"
else
    if [[ -z "$GPU_ROCM_VARIANT" ]]; then
        log_warn "GPU not in detection map. Defaulting to gfx110X."
        log_warn "Set GPU_ROCM_VARIANT in your environment if this is wrong."
        GPU_ROCM_VARIANT="gfx110X"
    fi
    log_info "GPU: ${GPU_NAME:-unknown} ($GPU_PCI_ID)"
    log_info "GFX: $GPU_GFX_ARCH ($GPU_GFX_VERSION)"
    log_info "SDK variant: $GPU_ROCM_VARIANT"
fi

# =============================================================================
# Download ROCm SDK (Linux only)
# =============================================================================

# verify_rocm_wheels <dir> <packages> <index-url>
# Check the downloaded wheels against scripts/rocm-wheels.sha256.
#
# The packages are version-pinned and fetched over HTTPS, which protects the
# transport but not the artifact: AMD re-publishing a wheel, or the index being
# compromised, both pass a version pin unnoticed. When the pin file is present
# every wheel must appear in it and match its digest, or the build stops before
# anything is extracted. When it is absent the build proceeds with a warning,
# because AMD re-rolls these wheels without notice and a hard requirement would
# break the ROCm path on their schedule rather than ours.
verify_rocm_wheels() {
    local dir="$1" pkgs="$2" index="$3"
    local checksums="$SCRIPT_DIR/rocm-wheels.sha256"

    if [[ ! -f "$checksums" ]]; then
        log_warn "No wheel pin file at $checksums; ROCm SDK integrity is unverified"
        log_warn "To pin this download, re-fetch it and record the digests:"
        log_warn "  pip3 download --no-deps -d /tmp/rocm-wheels \"$pkgs\" --index-url $index"
        log_warn "  ( cd /tmp/rocm-wheels && sha256sum *.whl ) > $checksums"
        return 0
    fi

    log_info "Verifying wheels against $(basename "$checksums")..."
    local bad=0 checked=0 wheel name want have
    for wheel in "$dir"/*.whl; do
        [[ -e "$wheel" ]] || continue
        name=$(basename "$wheel")
        want=$(awk -v n="$name" '$2 == n { print $1; exit }' "$checksums")
        if [[ -z "$want" ]]; then
            log_error "Wheel not listed in $(basename "$checksums"): $name"
            bad=$((bad + 1))
            continue
        fi
        have=$(sha256sum "$wheel" | awk '{ print $1 }')
        if [[ "$have" != "$want" ]]; then
            log_error "Checksum mismatch for $name"
            log_error "  expected $want"
            log_error "  actual   $have"
            bad=$((bad + 1))
            continue
        fi
        checked=$((checked + 1))
    done

    if [[ $bad -gt 0 ]]; then
        log_error "Refusing to extract $bad unverified wheel(s)."
        log_error "If AMD re-published this version, regenerate the pin file once you"
        log_error "have confirmed the new digests against AMD's published values."
        exit 1
    fi
    log_ok "Verified $checked wheel(s) against pin file"
}

download_rocm() {
    # The gufo build also needs the ROCm SDK (it compiles HIP kernels against
    # deps/), but --gufo sets BUILD_ROCM=false because it builds no llama.cpp
    # ROCm backend. Gating the download on BUILD_ROCM alone left
    # `build.sh --gufo` with no toolchain at all.
    if [[ "$BUILD_ROCM" != true && "$BUILD_GUFO" != true && "$SDK_ONLY" != true ]]; then
        return 0
    fi

    # Check if already extracted
    if [[ -f "$DEPS_DIR/lib/libamdhip64.so" ]] && [[ -f "$DEPS_DIR/lib/llvm/bin/clang" ]] && [[ "$REBUILD" != true ]]; then
        log_ok "ROCm SDK already installed at $DEPS_DIR"
        return 0
    fi

    # Full rebuild: wipe deps
    if [[ "$REBUILD" == true ]] && [[ -d "$DEPS_DIR" ]]; then
        log_info "Removing existing deps for full rebuild..."
        rm -rf "$DEPS_DIR"
    fi

    local rocm_ver="$ROCM_SDK_VERSION"
    local pip_repo="$ROCM_PIP_INDEX"

    # Map llm-scripts GPU detection to ROCm pip device packages.
    # The stable index ships one package per architecture (gfx1100, gfx1101,
    # gfx1102, gfx1103, gfx1150, gfx1151, ...). Older indices also carried
    # combined "-X-all" packages; those are kept as a fallback so a machine
    # whose exact architecture is not published still resolves.
    local device_pkg=""
    local device_pkg_fallback=""
    case "$GPU_GFX_ARCH" in
        gfx90*)
            device_pkg="rocm-sdk-device-gfx90a"
            ;;
        gfx101*)
            device_pkg="rocm-sdk-device-gfx1012"
            ;;
        gfx103*)
            device_pkg="rocm-sdk-device-gfx1030"
            ;;
        gfx110*|gfx115*|gfx120*|gfx125*)
            device_pkg="rocm-sdk-device-${GPU_GFX_ARCH}"
            device_pkg_fallback="rocm-sdk-device-gfx${GPU_GFX_ARCH#gfx}X-all"
            ;;
        *)
            device_pkg="rocm-sdk-device-gfx1103"
            device_pkg_fallback="rocm-sdk-device-gfx110X-all"
            ;;
    esac

    # Packages to download (core, libraries, device-specific, devel headers/cmake)
    local -a pkgs=(
        "rocm-sdk-core==${rocm_ver}"
        "rocm-sdk-libraries==${rocm_ver}"
        "${device_pkg}==${rocm_ver}"
        "rocm-sdk-devel==${rocm_ver}"
    )

    local tmp_wheel_dir
    tmp_wheel_dir=$(mktemp -d)
    # Expand the path now, while the local is in scope, and escape it so the
    # trap string survives any path. Deferring expansion to trap time would
    # see an unset variable once the function returns.
    trap "rm -rf $(printf '%q' "$tmp_wheel_dir")" RETURN

    log_info "Downloading ROCm SDK ${rocm_ver} (stable) from AMD pip repo..."
    log_info "Packages: ${pkgs[*]}"
    log_info "Device variant: $device_pkg"
    log_info "This is ~2.6 GB total (may take several minutes...)"

    # Download all wheels. The device package name is the one that varies
    # between index layouts, so it gets its own retry against the fallback.
    local download_ok=false
    if pip3 download --no-deps -d "$tmp_wheel_dir" "${pkgs[@]}" --index-url "$pip_repo" 2>&1 | tail -5; then
        download_ok=true
    elif [[ -n "$device_pkg_fallback" ]]; then
        log_warn "Device package '$device_pkg' unavailable; retrying with '$device_pkg_fallback'"
        pkgs=(
            "rocm-sdk-core==${rocm_ver}"
            "rocm-sdk-libraries==${rocm_ver}"
            "${device_pkg_fallback}==${rocm_ver}"
            "rocm-sdk-devel==${rocm_ver}"
        )
        if pip3 download --no-deps -d "$tmp_wheel_dir" "${pkgs[@]}" --index-url "$pip_repo" 2>&1 | tail -5; then
            download_ok=true
            device_pkg="$device_pkg_fallback"
        fi
    fi
    if [[ "$download_ok" != true ]]; then
        log_error "Failed to download ROCm SDK packages"
        log_error "Try: pip3 download --no-deps -d /tmp/rocm \"${pkgs[*]}\" --index-url $pip_repo"
        exit 1
    fi

    verify_rocm_wheels "$tmp_wheel_dir" "${pkgs[*]}" "$pip_repo"

    mkdir -p "$DEPS_DIR"

    # --- Extract rocm-sdk-core (biggest, contains compiler + runtime + headers) ---
    local core_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_core-*.whl 2>/dev/null | head -1)
    if [[ -z "$core_wheel" ]]; then
        log_error "rocm-sdk-core wheel not found"
        exit 1
    fi
    log_info "Extracting rocm-sdk-core..."
    python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$core_wheel')
prefix = None
for name in z.namelist():
    if name.startswith('_rocm_sdk_core/'):
        prefix = '_rocm_sdk_core/'
        break
if not prefix:
    print('ERROR: _rocm_sdk_core/ prefix not found in wheel')
    exit(1)
dest_root = '$DEPS_DIR'
count = 0
for name in z.namelist():
    if name.startswith(prefix) and not name.endswith('/'):
        relpath = name[len(prefix):]
        dest = os.path.join(dest_root, relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        count += 1
print(f'  Extracted {count} files')
"

    # --- Extract rocm-sdk-libraries (hipBLAS, rocBLAS, MIOpen, etc.) ---
    local libs_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_libraries-*.whl 2>/dev/null | head -1)
    if [[ -n "$libs_wheel" ]]; then
        log_info "Extracting rocm-sdk-libraries..."
        python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$libs_wheel')
prefix = '_rocm_sdk_libraries/'
count = 0
for name in z.namelist():
    if name.startswith(prefix) and not name.endswith('/'):
        relpath = name[len(prefix):]
        dest = os.path.join('$DEPS_DIR', relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        count += 1
print(f'  Extracted {count} files')
"
    fi

    # --- Extract device-specific code objects ---
    local device_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_device_*-$rocm_ver-*.whl 2>/dev/null | head -1)
    if [[ -n "$device_wheel" ]]; then
        log_info "Extracting device code for $GPU_ROCM_VARIANT..."
        python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$device_wheel')
# Device wheels may have different top-level prefixes
prefixes = set()
for name in z.namelist():
    if name and not name.endswith('/'):
        top = name.split('/')[0]
        if top.startswith('_'):
            prefixes.add(top + '/')

count = 0
for name in z.namelist():
    if name.endswith('/'):
        continue
    relpath = name
    for p in prefixes:
        if name.startswith(p):
            relpath = name[len(p):]
            break
    dest = os.path.join('$DEPS_DIR', relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with z.open(name) as src, open(dest, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    count += 1
print(f'  Extracted {count} files')
"
    fi

    # --- Extract devel package (cmake config files, additional headers) ---
    local devel_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_devel-*.whl 2>/dev/null | head -1)
    if [[ -n "$devel_wheel" ]]; then
        log_info "Extracting rocm-sdk-devel (cmake configs)..."
        python3 -c "
import zipfile, io, tarfile, os, shutil
z = zipfile.ZipFile('$devel_wheel')
# The devel package contains an internal _devel.tar that holds the actual files
tar_members = [n for n in z.namelist() if n.endswith('_devel.tar')]
for tar_name in tar_members:
    data = z.read(tar_name)
    t = tarfile.open(fileobj=io.BytesIO(data))
    # Determine the prefix from the first member
    sample = t.getmembers()[0] if t.getmembers() else None
    prefix = '_rocm_sdk_devel/'
    count = 0
    for m in t.getmembers():
        if not m.isfile():
            continue
        if m.name.startswith(prefix):
            relpath = m.name[len(prefix):]
        else:
            relpath = m.name
        dest = os.path.join('$DEPS_DIR', relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f = t.extractfile(m)
        if f:
            shutil.copyfileobj(f, open(dest, 'wb'))
        count += 1
    print(f'  Extracted {count} files from {tar_name}')
"
    fi

    # --- Create symlinks for versioned library names ---
    # The stable pip packages ship versioned .so files (e.g. libamdhip64.so.7)
    # but not the unversioned .so symlink. The cmake config files reference
    # specific versioned filenames. Create symlinks so all references resolve.
    log_info "Creating compatibility symlinks..."
    _create_rocm_symlinks "$DEPS_DIR/lib"

    # Clean up temporary wheel directory
    rm -rf "$tmp_wheel_dir"

    # Fix permissions on executables
    # `rocm*`, not `rocm-*`: the SDK ships rocm_agent_enumerator with
    # underscores, hipcc runs it to enumerate the device, and a wheel does not
    # preserve the exec bit. Missing it makes every HIP invocation warn
    # "Permission denied" and fall back to an auto-detected architecture.
    find "$DEPS_DIR" -type f \( -name "clang*" -o -name "hipcc" -o -name "hipconfig" \
        -o -name "rocm*" -o -name "hip*" -o -name "amd*" \) -exec chmod +x {} + 2>/dev/null || true
    find "$DEPS_DIR/lib/llvm/bin" -type f -exec chmod +x {} + 2>/dev/null || true

    # Verify
    if [[ -f "$DEPS_DIR/lib/libamdhip64.so" || -f "$DEPS_DIR/lib/libamdhip64.so.7" ]]; then
        log_ok "ROCm ${rocm_ver} SDK installed to $DEPS_DIR"
        log_info "Libraries: $(ls "$DEPS_DIR/lib/"*.so 2>/dev/null | wc -l) shared objects"
        
        # Register library paths with ldconfig so binaries can find them
        # without needing LD_LIBRARY_PATH sourced
        local ldconf="/etc/ld.so.conf.d/llm-scripts-rocm.conf"
        if sudo sh -c "cat > '$ldconf'" 2>/dev/null <<EOF
$DEPS_DIR/lib/llvm/lib
$DEPS_DIR/lib
$DEPS_DIR/lib/rocm_sysdeps/lib
EOF
        then
            sudo ldconfig 2>/dev/null || true
            log_info "Registered ROCm library paths with ldconfig"
        else
            log_warn "Could not register ROCm paths with ldconfig (non-root) - source env.sh before running"
        fi
    else
        log_error "ROCm SDK extraction failed - libamdhip64.so not found"
        exit 1
    fi

    # Verify ROCm bundled tools are accessible
    local rocm_clang="$DEPS_DIR/lib/llvm/bin/clang"
    if [[ ! -x "$rocm_clang" ]]; then
        log_error "ROCm clang not found at $rocm_clang"
        log_error "The SDK may be incomplete or corrupted. Try --rebuild."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Create symlinks so cmake config files (which reference specific versioned .so
# filenames) can find the libraries shipped by the stable ROCm packages.
# -----------------------------------------------------------------------------
_create_rocm_symlinks() {
    local lib_dir="$1"
    [[ -d "$lib_dir" ]] || return 0
    cd "$lib_dir" || return 0

    # Create unversioned .so symlinks from .so.MAJOR files (or shortest version)
    # Do this first so the Python step below can use them as fallback candidates.
    for f in *.so.*; do
        local base
        base=$(echo "$f" | sed 's/\.so\..*//')
        if [[ "$base" != "$f" ]] && [[ ! -L "${base}.so" ]] && [[ ! -f "${base}.so" ]]; then
            ln -sf "$f" "${base}.so" 2>/dev/null || true
        fi
    done

    # For each cmake targets file, find IMPORTED_LOCATION_RELEASE entries and
    # create symlinks from the referenced versioned name to the actual library
    if command -v python3 &>/dev/null; then
        python3 -c "
import os, re, glob

base = '$lib_dir'

# Collect all referenced library names from cmake targets files
referenced = set()
for cmake_path in glob.glob(os.path.join(base, 'cmake', '**', '*.cmake'), recursive=True):
    try:
        with open(cmake_path) as f:
            content = f.read()
        for m in re.finditer(r'lib([a-zA-Z0-9_-]+)\.so\.([0-9][0-9.\-]*)', content):
            referenced.add(m.group(0))
    except:
        pass

created = 0
for ref in sorted(referenced):
    if os.path.exists(ref):
        continue
    # Find the base library name (strip version entirely)
    base_name = ref.split('.so')[0] + '.so'
    # Build candidate paths by progressively stripping version components
    # e.g. for libhsa-runtime64.so.1.21.0:
    #   candidates = ['libhsa-runtime64.so', 'libhsa-runtime64.so.1',
    #                 'libhsa-runtime64.so.1.21', 'libhsa-runtime64.so.1.21.0']
    # We try longest version suffix first (most specific match)
    vparts = ref.split('.so.')[1].split('.') if '.so.' in ref else []
    candidates = [base_name]
    for i in range(1, len(vparts) + 1):
        candidates.append(base_name + '.' + '.'.join(vparts[:i]))
    # Reverse so we try longest match first
    candidates.reverse()

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break
    if found:
        os.symlink(found, ref)
        created += 1
    else:
        # Try any matching .so file with the same library prefix
        for f in os.listdir('.'):
            if f.startswith(base_name) and '.so' in f and f != ref:
                if os.path.islink(f) or os.path.isfile(f):
                    os.symlink(f, ref)
                    created += 1
                    break

print(f'  Created {created} compatibility symlinks')
" || true
    fi

    # Create unversioned .so symlinks from .so.MAJOR files (or shortest version)
    for f in *.so.*; do
        local base
        base=$(echo "$f" | sed 's/\.so\..*//')
        if [[ "$base" != "$f" ]] && [[ ! -L "${base}.so" ]] && [[ ! -f "${base}.so" ]]; then
            ln -sf "$f" "${base}.so" 2>/dev/null || true
        fi
    done

    cd - >/dev/null 2>&1 || true
}

# =============================================================================
# Build functions
# =============================================================================
# One script per backend under scripts/build/. They are runnable on their own
# once the SDK is installed; this dispatcher owns the shared work (SDK download,
# environment activation, wrapper generation) and passes --clean through.

build_rocm() {
    log_info "Building ROCm backend..."
    local -a args=()
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && args+=(--clean)
    "$LLM_ROOT/scripts/build/llama-rocm.sh" "${args[@]}"
}

build_vulkan() {
    log_info "Building Vulkan backend..."
    local -a args=()
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && args+=(--clean)
    "$LLM_ROOT/scripts/build/llama-vulkan.sh" "${args[@]}"
}

build_metal() {
    log_info "Building Metal backend (macOS)..."
    local -a args=()
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && args+=(--clean)
    "$LLM_ROOT/scripts/build/llama-metal.sh" "${args[@]}"
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo -e "${CYAN}=== Llama.cpp Build Script ===${NC}"
echo -e "${CYAN}  Platform: $(uname -s) $(uname -m)${NC}"
echo -e "${CYAN}  GPU: ${GPU_NAME:-unknown} (${GPU_GFX_ARCH:-?})${NC}"
echo -e "${CYAN}  CPU ISA: ${SYS_CPU_ISA:-unknown}${NC}"
echo -e "${CYAN}  CMake CPU flags: ${SYS_CMAKE_CPU_FLAGS:-none}${NC}"
if command -v ccache &>/dev/null; then
    echo -e "${CYAN}  ccache: $(command -v ccache)${NC}"
else
    echo -e "${YELLOW}  ccache: not found (rebuilds are full compiles)${NC}"
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${CYAN}  ROCm: ${ROCM_SDK_VERSION}${NC}"
fi
echo ""

# Step 1: Download ROCm SDK (Linux ROCm builds only)
download_rocm

# --sdk-only: the SDK is installed, nothing else to do.
if [[ "$SDK_ONLY" == true ]]; then
    echo ""
    echo -e "${GREEN}=== ROCm SDK Ready ===${NC}"
    echo "  Path: $DEPS_DIR"
    echo ""
    exit 0
fi

# Step 2: Source environment so ROCm tools (clang, etc.) are on PATH (Linux only)
if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "$BUILD_ROCM" == true || "$BUILD_VULKAN" == true ]]; then
        if [[ "$BUILD_ROCM" == true ]]; then
            source "$LLM_ROOT/lib/env.sh"
            llm_env_activate rocm
        else
            # Vulkan builds still need ROCm's bundled clang/lld for compilation
            export PATH="$LLM_ROOT/deps/lib/llvm/bin:$PATH"
        fi
    fi
fi

# Step 3: Check prerequisites
check_prereqs

# Step 4: Build requested backends
[[ "$BUILD_ROCM" == true ]]   && build_rocm
[[ "$BUILD_VULKAN" == true ]] && build_vulkan
[[ "$BUILD_METAL" == true ]]  && build_metal
if [[ "$BUILD_GUFO" == true ]]; then
    # The gufo build keeps its own build directory. --clean/--rebuild must
    # reach it, otherwise a stale object file survives and the binary is
    # silently built against headers that no longer match the sources.
    if [[ "$CLEAN" == true || "$REBUILD" == true ]]; then
        "$LLM_ROOT/scripts/build/gufo.sh" --clean
    else
        "$LLM_ROOT/scripts/build/gufo.sh"
    fi
fi

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "Binaries:"
[[ "$BUILD_ROCM" == true ]]   && echo "  ROCm:   $LLM_ROOT/build/llama-rocm/bin/llama-server"
[[ "$BUILD_VULKAN" == true ]] && echo "  Vulkan: $LLM_ROOT/build/llama-vulkan/bin/llama-server"
[[ "$BUILD_METAL" == true ]]  && echo "  Metal:  $LLM_ROOT/build/llama-metal/bin/llama-server"
[[ "$BUILD_GUFO" == true ]]   && echo "  gufo:   $LLM_ROOT/build/gufo/gufo"
echo ""
source "$LLM_ROOT/lib/env.sh"
llm_env_write_wrappers
echo "Next: drop a GGUF model in models/ and run ./llm serve --model NAME"
echo ""
