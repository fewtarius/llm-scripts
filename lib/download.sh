#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# HuggingFace model download with shard assembly and integrity checks.

[[ -n "${_LLM_DOWNLOAD_LOADED:-}" ]] && return 0
_LLM_DOWNLOAD_LOADED=1

_LLM_DL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_LLM_DL_DIR/common.sh"

hf_api_get() { curl -s -L --fail "https://huggingface.co/api/$1" 2>/dev/null; }
hf_search_models() { hf_api_get "models?search=${1// /+}&sort=downloads&direction=-1&limit=${2:-10}" | jq -r '.[].id // empty' 2>/dev/null; }
hf_list_files() { hf_api_get "models/$1" | jq -r '.siblings[].rfilename // empty' 2>/dev/null; }
hf_file_info() { jq -c --arg filename "$1" 'first(.siblings[]? | select(.rfilename == $filename)) // empty' 2>/dev/null; }

hf_find_gguf() {
    local repo="$1"
    local quant_pattern="${2:-Q4_K_M}"
    local files
    files=$(hf_list_files "$repo") || return 1
    local quant_base="${quant_pattern#*-}"
    local quant_family="${quant_base%%_*}"
    local quant_lc quant_base_lc quant_family_lc
    quant_lc=$(printf '%s' "$quant_pattern" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "$quant_base" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "$quant_family" | tr 'A-Z' 'a-z')
    local matches=()
    while IFS= read -r f; do
        [[ "$f" == *.gguf ]] || continue
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_family_lc}[-_] ]]; then
            matches+=("$f")
        fi
    done <<< "$files"
    if [[ ${#matches[@]} -eq 0 ]]; then
        while IFS= read -r f; do
            [[ "$f" == *.gguf ]] && matches+=("$f")
        done <<< "$files"
    fi
    printf '%s\n' "${matches[@]}" | sort -t'_' -k2 -V | tac
}

hf_list_quants() {
    local repo="$1"
    local files
    files=$(hf_list_files "$repo") 2>/dev/null || return 1
    echo "$files" | grep -oE '[Qq][0-9]+[_-]?K?_?[SMXL]?' | sort -u | head -20
}

download_model() {
    local input="$1"
    local quant="${2:-Q4_K_M}"
    local target_dir="${LLM_MODEL_DIR}"
    local repo=""

    # The HuggingFace helpers pipe through jq and swallow stderr, so a missing
    # jq otherwise presents as "no models found" rather than a missing tool.
    if ! command -v jq >/dev/null 2>&1; then
        log_error "jq is required for --download but is not installed"
        log_error "Install it with: sudo pacman -S jq   (or: ./scripts/install-deps.sh)"
        return 1
    fi

    if [[ "$input" == *"/"* ]]; then
        repo="${input%%:*}"
        if [[ "$input" == *":"* ]]; then
            quant="${input##*:}"
        fi
    else
        echo -e "${BLUE}Searching HuggingFace for: $input${NC}"
        local search_results
        search_results=$(hf_search_models "$input" 10)
        if [[ -z "$search_results" ]]; then
            echo -e "${RED}No models found matching: $input${NC}"
            return 1
        fi
        local found=false
        while IFS= read -r candidate; do
            local gguf_files
            gguf_files=$(hf_find_gguf "$candidate" 2>/dev/null)
            if [[ -n "$gguf_files" ]]; then
                repo="$candidate"
                echo -e "${GREEN}Found repo: $repo${NC}"
                found=true
                break
            fi
        done <<< "$search_results"
        if [[ "$found" != "true" ]]; then
            echo -e "${RED}No GGUF files found for any matching model${NC}"
            return 1
        fi
    fi

    # Validate the repository before printing anything: an empty listing
    # means the name is wrong or the request failed, not that it has no quants.
    local repo_files
    repo_files=$(hf_list_files "$repo" 2>/dev/null || true)
    if [[ -z "$repo_files" ]]; then
        log_error "No files found for repository: $repo"
        log_error "Expected a HuggingFace repo id that contains GGUF files,"
        log_error "for example unsloth/Qwen3-14B-GGUF:Q4_K_M"
        return 1
    fi

    echo -e "\n${BLUE}Available quantizations in $repo:${NC}"
    local quants
    quants=$(hf_list_quants "$repo")
    if [[ -n "$quants" ]]; then
        echo "$quants" | head -15 | while read -r q; do
            [[ "$q" == "$quant" ]] && echo -e "  $q (selected)" || echo "  $q"
        done
    fi

    echo -e "\n${BLUE}Finding best GGUF file for quant: $quant${NC}"
    local candidates
    candidates=$(hf_find_gguf "$repo" "$quant")
    if [[ -z "$candidates" ]]; then
        echo -e "${RED}No GGUF files found in $repo${NC}"
        return 1
    fi

    local selected=""
    local selected_base=""
    local part_count=1
    local total_parts=1
    local quant_lc quant_family_lc quant_base_lc
    quant_lc=$(printf '%s' "$quant" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "${quant%%_*}" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "${quant#*-}" | tr 'A-Z' 'a-z')

    while IFS= read -r f; do
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || [[ "$f_lc" == *"${quant_lc}"*.gguf ]]; then
            selected="$f"
            break
        fi
    done <<< "$candidates"

    if [[ -z "$selected" ]]; then
        while IFS= read -r f; do
            local f_lc
            f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
            if [[ "$f_lc" =~ [-_]${quant_family_lc}[-_] ]]; then
                selected="$f"
                break
            fi
        done <<< "$candidates"
    fi

    if [[ -z "$selected" ]]; then
        selected=$(echo "$candidates" | head -1)
    fi

    if [[ -z "$selected" ]]; then
        echo -e "${RED}Could not determine filename${NC}"
        return 1
    fi

    if [[ "$selected" =~ ^(.*?)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
        selected_base="${BASH_REMATCH[1]}"
        part_count="${BASH_REMATCH[2]}"
        total_parts="${BASH_REMATCH[3]}"
        echo -e "${BLUE}Detected multi-part file ($part_count of $total_parts)${NC}"
    fi

    local all_files=()
    all_files+=("$selected")
    if [[ -n "$selected_base" ]]; then
        while IFS= read -r f; do
            [[ " ${all_files[*]} " == *" $f "* ]] && continue
            local escaped_base
            escaped_base=$(printf '%s' "$selected_base" | tr '/' '\\/')
            if [[ "$f" =~ ^${escaped_base}-[0-9]+-of-[0-9]+\.gguf$ ]]; then
                all_files+=("$f")
            fi
        done <<< "$candidates"
    fi

    IFS=$'\n' sorted_files=($(sort -t'_' -k2 -V <<< "${all_files[*]}")); unset IFS
    mkdir -p "$target_dir"

    echo -e "\n${BLUE}Model Download Information${NC}"
    echo ""
    echo -e "  ${GREEN}Repo:${NC}     $repo"
    echo -e "  ${GREEN}Parts:${NC}    ${#sorted_files[@]} file(s) to download"
    for f in "${sorted_files[@]}"; do
        echo -e "    - $f"
    done
    echo -e "  ${GREEN}Quant:${NC}    $quant (requested)"
    echo -e "  ${GREEN}Target:${NC}   $target_dir"
    echo ""

    local files_to_download=()
    local expected_sizes=()
    local expected_shas=()
    local repo_metadata=""
    repo_metadata=$(hf_api_get "models/$repo?blobs=true" 2>/dev/null || true)

    for ((i = 0; i < ${#sorted_files[@]}; i++)); do
        local f="${sorted_files[$i]}"
        local file_info=""
        local expected_size=""
        local expected_sha=""
        if [[ -n "$repo_metadata" ]]; then
            file_info=$(printf '%s' "$repo_metadata" | hf_file_info "$f" || true)
        fi
        if [[ -n "$file_info" ]]; then
            expected_size=$(printf '%s' "$file_info" | jq -r '.size // .lfs.size // empty' 2>/dev/null || true)
            # HuggingFace stores the LFS object digest; it is the SHA-256 of the
            # file bytes, so it is a real integrity check rather than a hint.
            expected_sha=$(printf '%s' "$file_info" | jq -r '.lfs.sha256 // empty' 2>/dev/null || true)
        fi
        [[ ! "$expected_size" =~ ^[0-9]+$ ]] && expected_size=""
        [[ ! "$expected_sha" =~ ^[0-9a-fA-F]{64}$ ]] && expected_sha=""
        expected_sizes[$i]="$expected_size"
        expected_shas[$i]="$expected_sha"

        if [[ -f "$target_dir/$f" ]]; then
            local local_size
            local_size=$(stat -c %s "$target_dir/$f" 2>/dev/null || stat -f %z "$target_dir/$f")
            if [[ -n "$expected_size" ]] && [[ "$local_size" != "$expected_size" ]]; then
                echo -e "${YELLOW}Cached file has wrong size: $f${NC}"
                rm -f -- "$target_dir/$f"
                files_to_download+=("$f")
            else
                echo -e "${GREEN}Already exists: $f${NC}"
            fi
        else
            files_to_download+=("$f")
        fi
    done

    if [[ ${#files_to_download[@]} -eq 0 ]]; then
        echo -e "${GREEN}All files already cached.${NC}"
        return 0
    fi

    echo -e "${BLUE}Downloading ${#files_to_download[@]} file(s)...${NC}"
    echo ""

    local use_python=false
    local download_tool=""
    local idx=0
    if command -v hf &>/dev/null; then
        download_tool="hf"
    elif command -v huggingface-cli &>/dev/null; then
        download_tool="huggingface-cli"
    else
        use_python=true
    fi

    if [[ "$download_tool" == "hf" ]]; then
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"
            if hf download "$repo" "$f" --local-dir "$target_dir" 2>&1; then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                use_python=true
                break
            fi
        done
    elif [[ "$download_tool" == "huggingface-cli" ]]; then
        echo -e "${YELLOW}Note: huggingface-cli is deprecated. Install the 'hf' CLI instead.${NC}"
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"
            if huggingface-cli download "$repo" "$f" --local-dir "$target_dir" --local-dir-use-symlinks False 2>&1; then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                use_python=true
                break
            fi
        done
    fi

    if [[ "$use_python" == "true" ]]; then
        if ! python3 -c "import huggingface_hub" 2>/dev/null; then
            echo -e "${RED}huggingface_hub not available. Cannot download.${NC}"
            return 1
        fi
        local idx=0
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"
            if python3 << PYEOF
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id='$repo',
        filename='$f',
        local_dir='$target_dir',
        local_dir_use_symlinks=False
    )
    print(f'  Downloaded to: {path}')
except Exception as e:
    print(f'  Error: {e}')
    exit(1)
PYEOF
            then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                echo -e "${RED}  Failed: $f${NC}"
            fi
        done
    fi

    # Track which files came off the network this run. Cached files that passed
    # the size check are not re-hashed: a 40 GB shard costs real time and the
    # bytes we just pulled over the wire are the ones worth verifying.
    local -A just_downloaded=()
    local _dl
    for _dl in "${files_to_download[@]}"; do
        just_downloaded["$_dl"]=1
    done

    local invalid=0
    for ((i = 0; i < ${#sorted_files[@]}; i++)); do
        local f="${sorted_files[$i]}"
        local expected_size="${expected_sizes[$i]}"
        local expected_sha="${expected_shas[$i]:-}"
        if [[ ! -f "$target_dir/$f" ]]; then
            echo -e "${RED}Missing: $f${NC}"
            invalid=1
            continue
        fi
        if [[ -n "$expected_size" ]]; then
            local local_size
            local_size=$(stat -c %s "$target_dir/$f" 2>/dev/null || stat -f %z "$target_dir/$f")
            if [[ "$local_size" != "$expected_size" ]]; then
                echo -e "${RED}Invalid size: $f ($local_size bytes; expected $expected_size)${NC}"
                invalid=1
                continue
            fi
        fi
        if [[ -n "$expected_sha" && -n "${just_downloaded[$f]:-}" ]]; then
            local local_sha
            printf '%s' "  Verifying checksum: $f"
            local_sha=$(sha256sum "$target_dir/$f" 2>/dev/null | awk '{ print $1 }')
            if [[ "$local_sha" == "$expected_sha" ]]; then
                printf ' %bOK%b\n' "$GREEN" "$NC"
            else
                printf ' %bMISMATCH%b\n' "$RED" "$NC"
                echo -e "${RED}  Expected sha256: $expected_sha${NC}"
                echo -e "${RED}  Actual sha256:   ${local_sha:-<unreadable>}${NC}"
                echo -e "${RED}  Removing corrupt download: $f${NC}"
                rm -f -- "$target_dir/$f"
                invalid=1
            fi
        fi
    done

    if [[ $invalid -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}All files downloaded successfully!${NC}"
        return 0
    else
        return 1
    fi
}
