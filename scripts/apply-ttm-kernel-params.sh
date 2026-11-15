#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Configure GPU Memory (GTT) Allocation for AMD APUs
# =============================================================================
# Supports two methods:
#   1. Kernel parameters via bootloader (persisted across reboots):
#      - systemd-boot (SteamFork 3.8+): edits loader entry
#      - GRUB (SteamFork 3.7): edits /etc/default/grub + steamfork-grub-update
#   2. amd-smi set -G (runtime hint, applied AFTER bootloader writes so the
#      reboot prompt appears last; uses --no-prompt to suppress confirmation)
#
# Usage:
#   sudo ./scripts/apply-ttm-kernel-params.sh [GB]
#   Default: auto-detected based on system RAM
#   sudo ./scripts/apply-ttm-kernel-params.sh --dry-run [GB]
#   Prints the proposed kernel cmdline and changes nothing
#   sudo ./scripts/apply-ttm-kernel-params.sh --remove|--reset
#   Removes all TTM/GPU kernel params and resets amd-smi to defaults
#
# Only GTT is configured. VRAM stays at the firmware default and an existing
# amdgpu.vis_vramlimit in the kernel cmdline is removed - that parameter only
# caps how much of the BIOS carveout is visible, it never allocates VRAM.
#
# Kernel parameters set (all sizes in MB except ttm which uses 4KB pages):
#   amdgpu.gttsize       - GTT (GPU-accessible system RAM) window size
#   ttm.pages_limit       - cap on pages TTM may allocate, set to the window
#   ttm.page_pool_size    - pages held by the WC/UC/DMA recycling pool, kept far
#                           below pages_limit. Pool pages are freed from buffers
#                           but not returned to the buddy allocator, so a pool
#                           sized to the limit lets TTM hold the whole GPU
#                           budget in pages the host cannot see as free.
#                           Override with LLM_TTM_POOL_GB.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$LLM_ROOT"

source "$LLM_ROOT/lib/gpu.sh"

# Firmware VRAM carveout as reported by lib/gpu.sh. Reported for context
# only - this script never changes the VRAM configuration.
APU_VRAM_GB="${GPU_VRAM_GB:-0}"

# Pages the WC/UC/DMA pool may retain, in GiB. This is a recycling cache, not an
# allocation budget, so it stays small no matter how large the GTT window is.
LLM_TTM_POOL_GB="${LLM_TTM_POOL_GB:-2}"

# shellcheck source=lib/log.sh
source "$LLM_ROOT/lib/log.sh"

# =============================================================================
# Memory detection
# =============================================================================

# Get physical system RAM from DMI (actual hardware, not kernel-adjusted)
get_physical_ram_mb() {
    local total_mb=0

    # Method 1: dmidecode (most reliable)
    if command -v dmidecode &>/dev/null; then
        local dimm_size
        while read -r dimm_size; do
            [[ "$dimm_size" =~ Size:\ ([0-9]+)\ ([GM]B) ]] || continue
            local size="${BASH_REMATCH[1]}"
            local unit="${BASH_REMATCH[2]}"
            case "$unit" in
                GB) (( total_mb += size * 1024 )) ;;
                MB) (( total_mb += size )) ;;
            esac
        done < <(dmidecode -t memory 2>/dev/null | grep "^\s*Size:" | grep -v "No Module")
    fi

    # Method 2: fallback to /sys/class/drm + estimate
    if (( total_mb == 0 )) && [[ -d /sys/class/drm ]]; then
        local gtt_total vis_vram
        gtt_total=$(cat /sys/class/drm/card0/device/mem_info_gtt_total 2>/dev/null || echo 0)
        vis_vram=$(cat /sys/class/drm/card0/device/mem_info_vis_vram_total 2>/dev/null || echo 0)
        if (( gtt_total > 0 || vis_vram > 0 )); then
            # Sum GPU-accessible memory + buffer for OS
            total_mb=$(( (gtt_total + vis_vram) / 1024 / 1024 + 6144 ))
        fi
    fi

    echo "$total_mb"
}

# Get the RAM the kernel actually sees (MemTotal). On UMA systems the
# firmware carves VRAM out of system RAM before the kernel starts, so this
# figure is already net of the carveout and is the correct budget for
# sizing GTT.
get_system_ram_mb() {
    local total_kb=0
    if [[ -r /proc/meminfo ]]; then
        total_kb=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true)
    fi
    [[ "${total_kb:-0}" =~ ^[0-9]+$ ]] || total_kb=0
    echo $(( total_kb / 1024 ))
}

# Get current GPU memory allocation from sysfs
get_current_gpu_alloc_mb() {
    local gtt vis_vram total
    gtt=$(cat /sys/class/drm/card0/device/mem_info_gtt_total 2>/dev/null || echo 0)
    vis_vram=$(cat /sys/class/drm/card0/device/mem_info_vis_vram_total 2>/dev/null || echo 0)
    total=$((gtt + vis_vram))
    echo $(( total / 1024 / 1024 ))
}

# Compute the default GTT size in GB.
#   $1 physical RAM in MB (from DMI)
#   $2 RAM the kernel sees in MB (0 when unknown)
#   $3 firmware VRAM carveout in GB
default_gtt_gb() {
    local physical_ram_mb="$1" system_ram_mb="$2" apu_vram_gb="$3"
    local os_reserve_mb=2048
    local gtt_gb=0

    if (( physical_ram_mb > 16384 )); then
        os_reserve_mb=4096
    fi

    if (( apu_vram_gb >= 32 )); then
        # Large firmware carveout (e.g. Strix Halo with a 96 GB BIOS
        # allocation). The GPU already owns a dedicated pool, so GTT is only
        # useful as overflow - a small one is enough and a large one would
        # just waste address space.
        gtt_gb=4
    elif (( system_ram_mb > 0 )); then
        # Size against the RAM the kernel sees, which is already net of the
        # firmware VRAM carveout. Sizing off total physical RAM over-commits
        # by the carveout the firmware already took, and the kernel silently
        # clamps the result.
        gtt_gb=$(( (system_ram_mb - os_reserve_mb) / 1024 ))
    else
        gtt_gb=$(( (physical_ram_mb - apu_vram_gb * 1024 - os_reserve_mb) / 1024 ))
    fi

    if (( gtt_gb < 1 )); then
        gtt_gb=1
    fi
    echo "$gtt_gb"
}

# =============================================================================
# Boot loader detection
# =============================================================================

detect_boot_method() {
    if [[ -d /boot/loader/entries ]] || [[ -d /boot/EFI/steamfork/loader/entries ]]; then
        echo "systemd-boot"
        return 0
    fi

    if [[ -f /etc/default/grub ]] || [[ -f /boot/grub/grub.cfg ]]; then
        echo "grub"
        return 0
    fi

    if command -v steamfork-grub-update &>/dev/null; then
        echo "grub"
        return 0
    fi

    # systemd-boot runtime vars are present even when the ESP holding
    # loader/entries is not mounted (dual-ESP SteamFork installs where /boot
    # is a redundant empty ESP and the real entries live on an unmounted one).
    if [[ -e /sys/firmware/efi/efivars/LoaderEntries-4a67b082-0a4c-41cf-b6c7-440b29bb8c4f ]]; then
        echo "systemd-boot"
        return 0
    fi

    echo "unknown"
}

# =============================================================================
# amd-smi method (runtime, no reboot on ROCm 7.2+)
# =============================================================================

try_amd_smi() {
    local amd_smi_path="$PROJECT_ROOT/deps/bin/amd-smi"
    if [[ ! -x "$amd_smi_path" ]]; then
        amd_smi_path=$(command -v amd-smi 2>/dev/null || true)
    fi
    if [[ -z "$amd_smi_path" && -x /opt/rocm/bin/amd-smi ]]; then
        amd_smi_path=/opt/rocm/bin/amd-smi
    fi

    if [[ -z "$amd_smi_path" ]]; then
        log_warn "amd-smi not found"
        return 1
    fi

    log_info "Setting GTT to ${GTT_SIZE_GB}GB using amd-smi..."
    if "$amd_smi_path" set -G "$GTT_SIZE_GB" --no-prompt 2>&1; then
        log_ok "GTT hint set to ${GTT_SIZE_GB}GB via amd-smi"
        return 0
    else
        log_warn "amd-smi set failed"
        return 1
    fi
}

# =============================================================================
# amd-smi reset (revert GTT to system default)
# =============================================================================

reset_amd_smi() {
    local amd_smi_path="$PROJECT_ROOT/deps/bin/amd-smi"
    if [[ ! -x "$amd_smi_path" ]]; then
        amd_smi_path=$(command -v amd-smi 2>/dev/null || true)
    fi
    if [[ -z "$amd_smi_path" && -x /opt/rocm/bin/amd-smi ]]; then
        amd_smi_path=/opt/rocm/bin/amd-smi
    fi

    if [[ -z "$amd_smi_path" ]]; then
        log_warn "amd-smi not found, skipping reset"
        return 1
    fi

    log_info "Resetting GTT to system default using amd-smi..."
    if "$amd_smi_path" reset --gtt --no-prompt 2>&1; then
        log_ok "GTT reset to system default via amd-smi"
        return 0
    else
        log_warn "amd-smi reset failed"
        return 1
    fi
}

# =============================================================================
# systemd-boot method (SteamFork 3.8+)
# =============================================================================

apply_systemd_boot() {
    log_info "Using systemd-boot method..."

    local efi_mount="/boot"
    local efi_part
    efi_part=$(findmnt -n -o SOURCE -t vfat /boot 2>/dev/null) \
        || efi_part=$(findmnt -n -o SOURCE -t vfat /boot/efi 2>/dev/null) \
        || efi_part=$(findmnt -n -o SOURCE -t vfat /efi 2>/dev/null) \
        || true

    if [[ -z "$efi_part" ]]; then
        efi_part=$(lsblk -l -n -o NAME,FSTYPE,SIZE,MOUNTPOINT 2>/dev/null \
            | awk '$2=="vfat" && $3+0 < 1024 {print "/dev/"$1; exit}') \
            || true
    fi

    if [[ -z "$efi_part" ]]; then
        log_error "Could not auto-detect EFI partition"
        return 1
    fi

    if ! mountpoint -q "$efi_mount"; then
        mkdir -p "$efi_mount"
        mount "$efi_part" "$efi_mount"
        trap "umount $efi_mount 2>/dev/null || true" EXIT
    fi

    local loader_entry=""
    for dir in "$efi_mount/loader/entries" "$efi_mount/EFI/steamfork/loader/entries"; do
        [[ -d "$dir" ]] || continue
        for f in "$dir"/*.conf; do
            [[ -f "$f" ]] || continue
            local name=$(basename "$f")
            [[ "$name" =~ fallback|previous|verbose ]] && continue
            loader_entry="$f"
            break
        done
        [[ -n "$loader_entry" ]] && break
    done

    # Some SteamFork installs (e.g. Nimo with dual ESPs in RAID1) keep the
    # loader entries on an ESP that is NOT mounted - the mounted /boot can be
    # a redundant empty ESP. Probe the other vfat partitions for the real
    # entries and edit them in place (mount rw, apply, unmount).
    if [[ -z "$loader_entry" || ! -f "$loader_entry" ]]; then
        log_info "No loader entries on mounted ESP, probing other vfat partitions..."
        local probe_mnt="/mnt/esp-probe-ttm"
        mkdir -p "$probe_mnt"
        local probed=0
        while read -r _dev; do
            [[ -n "$_dev" ]] || continue
            [[ "/dev/$_dev" == "$efi_part" ]] && continue
            mount -o rw "/dev/$_dev" "$probe_mnt" 2>/dev/null || continue
            probed=1
            loader_entry=""
            for dir in "$probe_mnt/loader/entries" "$probe_mnt/EFI/steamfork/loader/entries"; do
                [[ -d "$dir" ]] || continue
                for f in "$dir"/*.conf; do
                    [[ -f "$f" ]] || continue
                    local name=$(basename "$f")
                    [[ "$name" =~ fallback|previous|verbose ]] && continue
                    loader_entry="$f"
                    break
                done
                [[ -n "$loader_entry" ]] && break
            done
            if [[ -n "$loader_entry" && -f "$loader_entry" ]]; then
                log_info "Found loader entries on /dev/$_dev"
                apply_params_to_entry "$loader_entry"
                umount "$probe_mnt" 2>/dev/null || true
                return 0
            fi
            umount "$probe_mnt" 2>/dev/null || true
        done < <(lsblk -l -n -o NAME,FSTYPE 2>/dev/null | awk '$2=="vfat" {print $1}')
        if [[ "$probed" -eq 0 ]]; then
            log_error "Could not mount any vfat ESP to search for loader entries"
            return 1
        fi
    fi

    if [[ -z "$loader_entry" || ! -f "$loader_entry" ]]; then
        log_error "Could not find systemd-boot loader entry"
        return 1
    fi

    apply_params_to_entry "$loader_entry"
}

# =============================================================================
# GRUB method (SteamFork 3.7 and others)
# =============================================================================

apply_grub() {
    log_info "Using GRUB method..."

    local grub_default="/etc/default/grub"

    if [[ ! -f "$grub_default" ]]; then
        log_error "/etc/default/grub not found"
        return 1
    fi

    current_cmdline=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$grub_default" | head -1 \
        | sed 's/GRUB_CMDLINE_LINUX_DEFAULT=//;s/"//g')

    log_info "Current GRUB_CMDLINE_LINUX_DEFAULT:"
    echo "  $current_cmdline"

    apply_params_to_grub "$grub_default"

    # Update GRUB
    if command -v steamfork-grub-update &>/dev/null; then
        log_info "Running steamfork-grub-update..."
        steamfork-grub-update
    elif command -v grub-mkconfig &>/dev/null; then
        log_info "Running grub-mkconfig..."
        grub-mkconfig -o /boot/grub/grub.cfg
    else
        log_warn "Could not update GRUB automatically"
        log_info "Run: sudo grub-mkconfig -o /boot/grub/grub.cfg"
    fi
}

# =============================================================================
# File mutation helpers
# =============================================================================

# Strip every GPU memory param from a systemd-boot entry file.
strip_gpu_params_from_file() {
    local file="$1"
    sed -i 's/amdgpu\.vis_vramlimit=[-0-9]*[KMGT]*//g; s/amdgpu\.gttsize=[-0-9]*[KMGT]*//g; s/ttm\.pages_limit=[-0-9]*[KMGT]*//g; s/ttm\.page_pool_size=[-0-9]*[KMGT]*//g' "$file"
    # Clean up multiple spaces
    sed -i 's/  */ /g; s/^ //; s/ $//' "$file"
}

# Replace the GPU memory params in a systemd-boot entry with KERNEL_PARAMS.
mutate_entry_params() {
    local file="$1"
    strip_gpu_params_from_file "$file"
    if ! grep -q "^options " "$file"; then
        log_error "No options line in $file - cannot set kernel params"
        return 1
    fi
    sed -i "s/options /options ${KERNEL_PARAMS} /" "$file"
}

# Replace the GPU memory params in a GRUB default file with KERNEL_PARAMS.
mutate_grub_params() {
    local file="$1"
    local current_cmdline stripped new_cmdline

    if ! grep -q "^GRUB_CMDLINE_LINUX_DEFAULT=" "$file"; then
        log_error "No GRUB_CMDLINE_LINUX_DEFAULT in $file"
        return 1
    fi

    # Read the command line here rather than reusing the caller's copy. The
    # previous version read it before this function ran, stripped the old TTM
    # params from the file, then rebuilt the line from that stale copy - which
    # put the params straight back. The result was duplicate amdgpu.gttsize
    # entries with the older value last, and the kernel takes the last
    # occurrence, so changing the size silently did nothing.
    current_cmdline=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$file" | head -1 \
        | sed 's/GRUB_CMDLINE_LINUX_DEFAULT=//;s/"//g')

    # Strip any params we set previously, then append ours last so the value we
    # want is the one the kernel resolves.
    stripped=$(printf '%s' "$current_cmdline" \
        | sed 's/amdgpu\.vis_vramlimit=[-0-9]*[KMGT]*//g; s/amdgpu\.gttsize=[-0-9]*[KMGT]*//g; s/ttm\.pages_limit=[-0-9]*[KMGT]*//g; s/ttm\.page_pool_size=[-0-9]*[KMGT]*//g' \
        | sed 's/  */ /g; s/^ //; s/ $//')

    new_cmdline="${stripped:+${stripped} }${KERNEL_PARAMS}"
    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${new_cmdline}\"|" "$file"
}

# Strip every GPU memory param from a GRUB default file.
mutate_grub_remove() {
    local file="$1"
    local current_cmdline new_cmdline

    if ! grep -q "^GRUB_CMDLINE_LINUX_DEFAULT=" "$file"; then
        log_error "No GRUB_CMDLINE_LINUX_DEFAULT in $file"
        return 1
    fi

    current_cmdline=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$file" | head -1 \
        | sed 's/GRUB_CMDLINE_LINUX_DEFAULT=//;s/"//g')

    new_cmdline=$(printf '%s' "$current_cmdline" \
        | sed 's/amdgpu\.vis_vramlimit=[-0-9]*[KMGT]*//g; s/amdgpu\.gttsize=[-0-9]*[KMGT]*//g; s/ttm\.pages_limit=[-0-9]*[KMGT]*//g; s/ttm\.page_pool_size=[-0-9]*[KMGT]*//g' \
        | sed 's/  */ /g; s/^ //; s/ $//')

    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${new_cmdline}\"|" "$file"
}

# Show what a mutation would do to a file without touching the original.
preview_mutation() {
    local file="$1" mutate="$2"
    local tmp
    tmp=$(mktemp)
    cp "$file" "$tmp"
    if ! "$mutate" "$tmp"; then
        rm -f "$tmp"
        log_error "Failed to build the proposed change for $file"
        return 1
    fi
    echo ""
    log_info "Dry run - proposed changes to ${file}:"
    diff -u "$file" "$tmp" || true
    rm -f "$tmp"
}

# =============================================================================
# Apply parameters to systemd-boot entry
# =============================================================================

apply_params_to_entry() {
    local entry="$1"

    log_info "Target entry: $entry"

    if (( DRY_RUN )); then
        preview_mutation "$entry" mutate_entry_params || return 1
        return 0
    fi

    # Backup
    local backup="${entry}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$entry" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    if ! mutate_entry_params "$entry"; then
        restore_backup "$entry" "$backup"
        return 1
    fi

    verify_and_report "$entry" "$backup"
}

# =============================================================================
# Remove parameters from systemd-boot entry (restore to stock)
# =============================================================================

remove_params_from_entry() {
    local entry="$1"

    log_info "Target entry: $entry"

    if (( DRY_RUN )); then
        preview_mutation "$entry" strip_gpu_params_from_file || return 1
        return 0
    fi

    # Backup
    local backup="${entry}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$entry" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    strip_gpu_params_from_file "$entry"

    verify_removal "$entry" "$backup"
}

# =============================================================================
# Apply parameters to GRUB default
# =============================================================================

apply_params_to_grub() {
    local file="$1"

    log_info "Target file: $file"

    if (( DRY_RUN )); then
        preview_mutation "$file" mutate_grub_params || return 1
        return 0
    fi

    # Backup
    local backup="${file}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$file" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    if ! mutate_grub_params "$file"; then
        restore_backup "$file" "$backup"
        return 1
    fi

    verify_and_report "$file" "$backup"
}

# =============================================================================
# Remove parameters from GRUB default (restore to stock)
# =============================================================================

remove_params_from_grub() {
    local file="$1"

    log_info "Target file: $file"

    if (( DRY_RUN )); then
        preview_mutation "$file" mutate_grub_remove || return 1
        return 0
    fi

    # Backup
    local backup="${file}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$file" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    if ! mutate_grub_remove "$file"; then
        restore_backup "$file" "$backup"
        return 1
    fi

    verify_removal "$file" "$backup"

    # Update GRUB
    if command -v steamfork-grub-update &>/dev/null; then
        log_info "Running steamfork-grub-update..."
        steamfork-grub-update
    elif command -v grub-mkconfig &>/dev/null; then
        log_info "Running grub-mkconfig..."
        grub-mkconfig -o /boot/grub/grub.cfg
    else
        log_warn "Could not update GRUB automatically"
        log_info "Run: sudo grub-mkconfig -o /boot/grub/grub.cfg"
    fi
}

# =============================================================================
# Verify and report
# =============================================================================

# restore_backup <file> <backup>
# Restore exactly the backup this run created. The previous form,
# `cp "${file}.bak-"* "$file"`, expanded to multiple sources once more than one
# backup existed; cp then requires a directory destination, failed, and the
# `|| true` hid it - leaving a half-edited bootloader config.
restore_backup() {
    local file="$1" backup="$2"

    if [[ -z "$backup" || ! -f "$backup" ]]; then
        log_error "No backup available to restore ${file}"
        log_error "Inspect ${file} manually before rebooting"
        return 1
    fi

    log_error "Restoring ${file} from $(basename "$backup")..."
    if cp "$backup" "$file"; then
        log_ok "Restored ${file}"
    else
        log_error "RESTORE FAILED - ${file} may be in an inconsistent state"
        log_error "Restore by hand: sudo cp '$backup' '$file'"
        return 1
    fi
}

# True when the file carries the requested gttsize and no vis_vramlimit.
verify_params_present() {
    local file="$1"
    grep -qE "amdgpu\.gttsize=${SIZE_MB}([^0-9]|$)" "$file" 2>/dev/null || return 1
    if grep -qE "amdgpu\.vis_vramlimit" "$file" 2>/dev/null; then
        return 1
    fi
    return 0
}

verify_and_report() {
    local file="$1"
    local backup="${2:-}"

    # Anchored on a non-digit so a requested gttsize of 2048 does not verify
    # against a leftover gttsize=20480.
    if verify_params_present "$file"; then
        log_ok "Parameters applied:"
        grep -oE "amdgpu.gttsize=[0-9]+|ttm.pages_limit=[0-9]+|ttm.page_pool_size=[0-9]+" "$file" | while read -r line; do
            echo "  $line"
        done
        local dups
        dups=$(grep -oE "amdgpu\.gttsize=[0-9]+" "$file" | wc -l)
        if [[ "$dups" -gt 1 ]]; then
            log_warn "${dups} amdgpu.gttsize entries present; the kernel uses the last one"
        fi
        log_ok "VRAM left at the firmware default (no vis_vramlimit set)"
        echo ""
        log_warn "Reboot required for changes to take effect"
        log_warn "Run: sudo reboot"
        return 0
    else
        log_error "Failed to apply parameters"
        if grep -qE "amdgpu\.vis_vramlimit" "$file" 2>/dev/null; then
            log_error "amdgpu.vis_vramlimit survived the rewrite in $(basename "$file")"
        fi
        restore_backup "$file" "$backup"
        return 1
    fi
}

# =============================================================================
# Verify removal (params should be gone)
# =============================================================================

verify_removal() {
    local file="$1"
    local backup="${2:-}"

    if grep -qE "amdgpu\.(vis_vramlimit|gttsize)|ttm\.(pages_limit|page_pool_size)" "$file" 2>/dev/null; then
        log_error "Failed to remove all GPU memory parameters"
        echo "  Remaining params:"
        grep -oE "amdgpu\.(vis_vramlimit|gttsize)=[0-9]+|ttm\.(pages_limit|page_pool_size)=[0-9]+" "$file" || true
        restore_backup "$file" "$backup"
        return 1
    else
        log_ok "All GPU memory parameters removed"
        echo ""
        log_warn "Reboot required for changes to take effect"
        log_warn "Run: sudo reboot"
        return 0
    fi
}

# =============================================================================
# Remove from bootloader (dispatcher)
# =============================================================================

remove_from_bootloader() {
    local boot_method
    boot_method=$(detect_boot_method)
    log_info "Detected boot method: $boot_method"

    case "$boot_method" in
        systemd-boot)
            local efi_mount="/boot"
            local efi_part
            efi_part=$(findmnt -n -o SOURCE -t vfat /boot 2>/dev/null) \
                || efi_part=$(findmnt -n -o SOURCE -t vfat /boot/efi 2>/dev/null) \
                || efi_part=$(findmnt -n -o SOURCE -t vfat /efi 2>/dev/null) \
                || true

            if [[ -z "$efi_part" ]]; then
                efi_part=$(lsblk -l -n -o NAME,FSTYPE,SIZE,MOUNTPOINT 2>/dev/null \
                    | awk '$2=="vfat" && $3+0 < 1024 {print "/dev/"$1; exit}') \
                    || true
            fi

            if [[ -z "$efi_part" ]]; then
                log_error "Could not auto-detect EFI partition"
                return 1
            fi

            if ! mountpoint -q "$efi_mount"; then
                mkdir -p "$efi_mount"
                mount "$efi_part" "$efi_mount"
                trap "umount $efi_mount 2>/dev/null || true" EXIT
            fi

            local loader_entry=""
            for dir in "$efi_mount/loader/entries" "$efi_mount/EFI/steamfork/loader/entries"; do
                [[ -d "$dir" ]] || continue
                for f in "$dir"/*.conf; do
                    [[ -f "$f" ]] || continue
                    local name=$(basename "$f")
                    [[ "$name" =~ fallback|previous|verbose ]] && continue
                    loader_entry="$f"
                    break
                done
                [[ -n "$loader_entry" ]] && break
            done

            # Same dual-ESP fallback as apply_systemd_boot: probe unmounted
            # vfat partitions for the real loader entries.
            if [[ -z "$loader_entry" || ! -f "$loader_entry" ]]; then
                log_info "No loader entries on mounted ESP, probing other vfat partitions..."
                local probe_mnt="/mnt/esp-probe-ttm"
                mkdir -p "$probe_mnt"
                local probed=0
                while read -r _dev; do
                    [[ -n "$_dev" ]] || continue
                    [[ "/dev/$_dev" == "$efi_part" ]] && continue
                    mount -o rw "/dev/$_dev" "$probe_mnt" 2>/dev/null || continue
                    probed=1
                    loader_entry=""
                    for dir in "$probe_mnt/loader/entries" "$probe_mnt/EFI/steamfork/loader/entries"; do
                        [[ -d "$dir" ]] || continue
                        for f in "$dir"/*.conf; do
                            [[ -f "$f" ]] || continue
                            local name=$(basename "$f")
                            [[ "$name" =~ fallback|previous|verbose ]] && continue
                            loader_entry="$f"
                            break
                        done
                        [[ -n "$loader_entry" ]] && break
                    done
                    if [[ -n "$loader_entry" && -f "$loader_entry" ]]; then
                        log_info "Found loader entries on /dev/$_dev"
                        remove_params_from_entry "$loader_entry"
                        umount "$probe_mnt" 2>/dev/null || true
                        return 0
                    fi
                    umount "$probe_mnt" 2>/dev/null || true
                done < <(lsblk -l -n -o NAME,FSTYPE 2>/dev/null | awk '$2=="vfat" {print $1}')
                if [[ "$probed" -eq 0 ]]; then
                    log_error "Could not mount any vfat ESP to search for loader entries"
                    return 1
                fi
            fi

            if [[ -z "$loader_entry" || ! -f "$loader_entry" ]]; then
                log_error "Could not find systemd-boot loader entry"
                return 1
            fi

            remove_params_from_entry "$loader_entry"
            ;;
        grub)
            local grub_default="/etc/default/grub"
            if [[ ! -f "$grub_default" ]]; then
                log_error "/etc/default/grub not found"
                return 1
            fi
            remove_params_from_grub "$grub_default"
            ;;
        *)
            log_error "Unknown boot method"
            log_info "Remove kernel params manually from bootloader config"
            exit 1
            ;;
    esac
}

# build_kernel_params <gtt_gb>
#   Sets SIZE_MB, SIZE_PAGES, POOL_PAGES and KERNEL_PARAMS for the requested GTT
#   size. Kept above the Main banner so the test harness, which sources only the
#   function section, exercises the real string instead of a copy of it.
#
# The recycling pool is sized independently of the window on purpose. Setting
# page_pool_size equal to pages_limit, which an earlier revision of this script
# did, lets TTM retain every page the GPU budget can reach inside its own pool.
# Those pages are no longer in the buddy allocator, so MemAvailable stops
# reporting them and the OOM killer stays quiet while an allocation blocks.
build_kernel_params() {
    local gtt_gb="$1"
    SIZE_MB=$(( gtt_gb * 1024 ))
    SIZE_PAGES=$(( gtt_gb * 1024 * 1024 / 4 ))  # 4KB pages
    POOL_PAGES=$(( LLM_TTM_POOL_GB * 1024 * 1024 / 4 ))  # 4KB pages
    KERNEL_PARAMS="amdgpu.gttsize=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${POOL_PAGES}"
}

# =============================================================================
# Main
# =============================================================================

# Help and dry-run are read-only: they must work without root. Parsing comes
# first so a preview never asks for privileges it does not use.
usage() {
     sed -n '5,35p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# --dry-run: preview the change, write nothing
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $EUID -ne 0 && $DRY_RUN -eq 0 ]]; then
    log_error "This script must be run as root (sudo)."
    log_error "Add --dry-run to preview without root."
    exit 1
fi

# --remove/--reset flag: strip all GPU memory params and reset amd-smi to defaults
if [[ "${1:-}" == "--remove" || "${1:-}" == "--reset" ]]; then
    log_info "Removing GPU memory parameters..."
    echo ""

    if (( DRY_RUN == 0 )); then
        # Reset amd-smi to default
        reset_amd_smi || true
    fi

    # Remove from bootloader
    remove_from_bootloader

    exit 0
fi

# Get requested size (or auto-detect)
GTT_SIZE_GB="${1:-}"

# Detect system RAM
PHYSICAL_RAM_MB=$(get_physical_ram_mb)
CURRENT_GPU_ALLOC_MB=$(get_current_gpu_alloc_mb)

log_info "Physical system RAM: $((PHYSICAL_RAM_MB / 1024))GB"
log_info "Current GPU allocation: ${CURRENT_GPU_ALLOC_MB}MB"

if [[ -z "$GTT_SIZE_GB" ]]; then
    if (( PHYSICAL_RAM_MB == 0 )); then
        log_error "Could not detect physical RAM. Try running as root, or install dmidecode."
        log_error "Or specify GTT size manually: sudo $0 <GB>"
        exit 1
    fi
    GTT_SIZE_GB=$(default_gtt_gb "$PHYSICAL_RAM_MB" "$(get_system_ram_mb)" "$APU_VRAM_GB")
fi

# Only GTT is configured. The firmware VRAM carveout is left alone: this
# script never writes amdgpu.vis_vramlimit, and any existing
# amdgpu.vis_vramlimit in the kernel cmdline is stripped.
build_kernel_params "$GTT_SIZE_GB"

log_info "Firmware VRAM carveout: ${APU_VRAM_GB}GB (left untouched)"
log_info "Target GPU memory: ${GTT_SIZE_GB}GB GTT"
log_info "Parameters:"
echo "  ${KERNEL_PARAMS}"
echo ""

# Always write kernel params to bootloader - this is what actually persists
BOOT_METHOD=$(detect_boot_method)
log_info "Detected boot method: $BOOT_METHOD"

case "$BOOT_METHOD" in
    systemd-boot)
        apply_systemd_boot
        ;;
    grub)
        apply_grub
        ;;
    *)
        log_error "Unknown boot method"
        log_info "Set parameters manually:"
        echo "  ${KERNEL_PARAMS}"
        exit 1
        ;;
esac

if (( DRY_RUN )); then
    log_ok "Dry run complete - nothing was changed."
    exit 0
fi

# Apply runtime GTT hint via amd-smi AFTER bootloader params are written.
# This way the reboot prompt (if any) appears last, after all persistent
# changes are complete. Pass --no-prompt to suppress the confirmation.
try_amd_smi || true

log_ok "GPU memory configuration complete."
log_warn "Reboot required for kernel parameter changes to take effect."