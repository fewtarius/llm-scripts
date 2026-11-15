#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# lib-rust.sh - Rust toolchain resolution for the gufo build
# =============================================================================
# gufo builds its PNG and WAV decoders from Rust, and its CMakeLists calls
# find_program(cargo REQUIRED). The tree pins the toolchain in
# gufo/rust-toolchain.toml, so rustup selects the right one once it is on PATH.
#
# rustup installs into ~/.cargo/bin and adds it to PATH from the shell profile
# only. A build driven from a non-login shell - build.sh, a CI step, a
# systemd unit, an editor build task - therefore sees no cargo even when the
# toolchain is installed and current. Look for it where it actually lives
# before concluding it is missing, and install it only when it really is.
#
# Callers: src/llama-gufo/build.sh and scripts/install-deps.sh. Both provide
# the log_* helpers; the versions here are fallbacks for standalone use.
# =============================================================================

# Minimum supported toolchain. gufo/Cargo.toml sets rust-version, and
# gufo/rust-toolchain.toml pins the channel the build is verified against.
GUFO_RUST_MIN_VERSION="${GUFO_RUST_MIN_VERSION:-1.88}"
# Set to 0 to fail instead of downloading a toolchain.
GUFO_RUST_AUTO_INSTALL="${GUFO_RUST_AUTO_INSTALL:-1}"

if ! declare -F log_info >/dev/null 2>&1; then
    log_info()  { printf '[INFO]  %s\n' "$1"; }
fi
if ! declare -F log_ok >/dev/null 2>&1; then
    log_ok()    { printf '[OK]    %s\n' "$1"; }
fi
if ! declare -F log_warn >/dev/null 2>&1; then
    log_warn()  { printf '[WARN]  %s\n' "$1" >&2; }
fi
if ! declare -F log_error >/dev/null 2>&1; then
    log_error() { printf '[ERROR] %s\n' "$1" >&2; }
fi

# _rust_version_ge <have> <want>
# True when <have> is at least <want>, comparing dotted numeric components.
_rust_version_ge() {
    local have="${1#rustc }" want="${2:-$GUFO_RUST_MIN_VERSION}"
    [[ -n "$have" ]] || return 1
    # sort -V orders 1.9.0 before 1.88.0 correctly, which a string compare
    # would not. The lowest entry is <want> only when <have> >= <want>.
    [[ "$(printf '%s\n%s\n' "$have" "$want" | sort -V | head -1)" == "$want" ]]
}

# _rust_version_in <dir>
_rust_version_in() {
    "$1/rustc" --version 2>/dev/null | awk '{print $2}'
}

# _rust_probe_dirs
# Candidate bin directories, in preference order. Emits one per line.
_rust_probe_dirs() {
    local dir
    [[ -n "${CARGO_HOME:-}" ]] && printf '%s\n' "$CARGO_HOME/bin"
    printf '%s\n' "${HOME:-/root}/.cargo/bin"
    printf '%s\n' "/usr/local/cargo/bin"
    printf '%s\n' "/opt/rust/bin"
    # A distro or hand-installed toolchain has no rustup shim, so look for the
    # compiler where the package manager puts it and under ~/.rustup.
    for dir in "${HOME:-/root}"/.rustup/toolchains/*/bin; do
        [[ -d "$dir" ]] && printf '%s\n' "$dir"
    done
    return 0
}

# _rust_prepend <dir>
# Puts <dir> on PATH unless an equivalent entry is already there.
_rust_prepend() {
    local dir="$1"
    case ":$PATH:" in
        *":$dir:"*) return 0 ;;
    esac
    PATH="$dir:$PATH"
    export PATH
}

_rust_install_hint() {
    log_error "gufo needs cargo and rustc (Rust $GUFO_RUST_MIN_VERSION or newer) to build its decoders"
    log_error "install it with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $GUFO_RUST_MIN_VERSION.0"
    log_error "or set GUFO_RUST_AUTO_INSTALL=1 (the default) and let the build do it"
}

# _rust_install <project_root>
# Installs the channel pinned by gufo/rust-toolchain.toml, else the minimum.
_rust_install() {
    local root="${1:-}" channel tmp installer="" fetcher=""

    channel="$GUFO_RUST_MIN_VERSION.0"
    if [[ -n "$root" && -f "$root/gufo/rust-toolchain.toml" ]]; then
        local pinned
        pinned="$(awk -F'"' '/^[[:space:]]*channel/ {print $2; exit}' "$root/gufo/rust-toolchain.toml")"
        [[ -n "$pinned" ]] && channel="$pinned"
    fi

    if command -v curl >/dev/null 2>&1; then
        fetcher="curl --proto =https --tlsv1.2 -sSf -o"
    elif command -v wget >/dev/null 2>&1; then
        fetcher="wget -q -O"
    else
        log_error "neither curl nor wget is available to fetch the Rust toolchain"
        _rust_install_hint
        return 1
    fi

    tmp="$(mktemp -t rustup-init.XXXXXX.sh)"
    log_info "Downloading the Rust toolchain ($channel) into ${CARGO_HOME:-${HOME:-/root}/.cargo}..."
    # shellcheck disable=SC2086  # $fetcher is a command plus its own flags
    if ! $fetcher "$tmp" https://sh.rustup.rs; then
        rm -f "$tmp"
        log_error "could not download the Rust installer from https://sh.rustup.rs"
        _rust_install_hint
        return 1
    fi

    # --no-modify-path keeps the build out of the user's shell profile; the
    # resolver above puts the toolchain on PATH for every build instead.
    # shellcheck disable=SC2086
    if ! sh "$tmp" -y --profile minimal --default-toolchain "$channel" --no-modify-path; then
        rm -f "$tmp"
        log_error "the Rust installer failed"
        _rust_install_hint
        return 1
    fi
    rm -f "$tmp"

    log_info "Rust installed; add ${CARGO_HOME:-${HOME:-/root}/.cargo}/bin to PATH for interactive use"
    return 0
}

# find_rust_toolchain [quiet]
# Resolves a usable cargo/rustc onto PATH without installing anything.
# Returns non-zero when no acceptable toolchain is present.
find_rust_toolchain() {
    local quiet="${1:-0}" dir have

    if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
        have="$(rustc --version 2>/dev/null | awk '{print $2}')"
        if _rust_version_ge "$have"; then
            [[ "$quiet" == 1 ]] || log_ok "Rust $have ($(command -v cargo))"
            return 0
        fi
        [[ "$quiet" == 1 ]] || log_warn "Rust $have on PATH is older than $GUFO_RUST_MIN_VERSION; looking for a newer one"
    fi

    while IFS= read -r dir; do
        [[ -n "$dir" ]] || continue
        [[ -x "$dir/cargo" && -x "$dir/rustc" ]] || continue
        # Skip a directory already covered by PATH.
        case ":$PATH:" in
            *":$dir:"*) continue ;;
        esac

        have="$(_rust_version_in "$dir")"
        if ! _rust_version_ge "$have"; then
            [[ "$quiet" == 1 ]] || log_warn "Rust ${have:-unknown} at $dir is older than $GUFO_RUST_MIN_VERSION"
            continue
        fi
        _rust_prepend "$dir"
        if [[ "$quiet" != 1 ]]; then
            log_info "Rust $have found at $dir (not on PATH; using it for this build)"
            log_ok "Rust toolchain resolved"
        fi
        return 0
    done < <(_rust_probe_dirs)

    return 1
}

# ensure_rust_toolchain [project_root]
# Resolves a usable cargo/rustc, installing the pinned channel when none is
# present and installation is not disabled. Returns non-zero when no toolchain
# can be provided.
ensure_rust_toolchain() {
    local root="${1:-${PROJECT_ROOT:-}}" dir have

    find_rust_toolchain && return 0

    if [[ "$GUFO_RUST_AUTO_INSTALL" == 0 ]]; then
        _rust_install_hint
        return 1
    fi

    log_warn "no usable Rust toolchain found; installing one"
    _rust_install "$root" || return 1

    while IFS= read -r dir; do
        [[ -n "$dir" ]] || continue
        [[ -x "$dir/cargo" && -x "$dir/rustc" ]] || continue
        have="$(_rust_version_in "$dir")"
        if _rust_version_ge "$have"; then
            _rust_prepend "$dir"
            log_ok "Rust $have resolved at $dir"
            return 0
        fi
    done < <(_rust_probe_dirs)

    log_error "Rust was installed but no usable toolchain could be resolved"
    _rust_install_hint
    return 1
}
