#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# gufo profile solver.
#
# gufo owns its memory plan, so this solver decides policy, not budget:
# family, speculative mode, draft count, context, sessions, KV precision
# and the sparse-attention switch. Engine knobs (family environment
# variables) are applied by lib/engines/gufo.sh, not here.

[[ -n "${_LLM_SOLVER_GUFO_LOADED:-}" ]] && return 0
_LLM_SOLVER_GUFO_LOADED=1

_GUFO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/gguf.sh
source "$_GUFO_DIR/../lib/gguf.sh"
# shellcheck source=lib/models.sh
source "$_GUFO_DIR/../lib/models.sh"

_gufo_detect_family() {
    local model="$1" key name
    SOLVER_GUFO_FAMILY="unknown"
    gguf_read_meta "$model" || SOLVER_GGUF=()
    for key in "${!SOLVER_GGUF[@]}"; do
        case "$key" in
            laguna.*)    SOLVER_GUFO_FAMILY="laguna"; return 0 ;;
            deepseek2.*) SOLVER_GUFO_FAMILY="deepseek"; return 0 ;;
        esac
    done
    name="${model,,}"
    case "$name" in
        *laguna*)     SOLVER_GUFO_FAMILY="laguna" ;;
        *deepseek*)   SOLVER_GUFO_FAMILY="deepseek" ;;
        *flash-next*) SOLVER_GUFO_FAMILY="qwen-flash-next" ;;
        *qwen*)       SOLVER_GUFO_FAMILY="qwen" ;;
        *glm*)        SOLVER_GUFO_FAMILY="glm" ;;
        *flint*)      SOLVER_GUFO_FAMILY="flint" ;;
        *)            SOLVER_GUFO_FAMILY="unknown" ;;
    esac
}

_gufo_drafter_usable() {
    local family="$1" file="$2" reason
    if reason=$(gguf_validate_drafter "$family" "$file"); then
        return 0
    fi
    SOLVER_GUFO_DRAFTER_REJECT="$reason"
    return 1
}

# KV precision a bias selects, unless the user already chose one. Only the
# families whose extension has a cache switch take q8_0; the rest run f16 and
# would otherwise log "no KV-cache switch" on every request.
_gufo_bias_kv() {
    local want="$1" bias="$2" explicit="${3:-0}"
    (( explicit )) && return 0
    if [[ "$want" == "q8_0" ]]; then
        case "$SOLVER_GUFO_FAMILY" in
            laguna|qwen|qwen-flash-next) ;;
            *)
                SOLVER_GUFO_REASONS+=("q8_0 KV unavailable for family $SOLVER_GUFO_FAMILY; f16 (bias=$bias)")
                return 0
                ;;
        esac
    fi
    SOLVER_GUFO_KV_CACHE="$want"
    [[ "$want" == "q8_0" ]] && SOLVER_GUFO_REASONS+=("KV cache q8_0 (bias=$bias)")
    return 0
}

_gufo_drop_spec() {
    local bias="$1"
    if [[ "$SOLVER_GUFO_SPEC" != "off" ]]; then
        SOLVER_GUFO_SPEC="off"
        SOLVER_GUFO_DRAFT_MODEL=""
        SOLVER_GUFO_DRAFT_KIND=""
        SOLVER_GUFO_REASONS+=("speculative dropped (bias=$bias)")
    fi
}

_gufo_solve_spec() {
    local model="$1"
    SOLVER_GUFO_SPEC="off"
    SOLVER_GUFO_DRAFT_MODEL=""
    SOLVER_GUFO_DRAFT_KIND=""

    if [[ "${LLM_SPEC:-}" == "off" ]]; then
        SOLVER_GUFO_REASONS+=("speculative disabled by request")
        return 0
    fi

    local draft
    case "$SOLVER_GUFO_FAMILY" in
        deepseek)
            if draft=$(llm_find_sidecar "$model" dspark); then
                SOLVER_GUFO_SPEC="dspark"
                SOLVER_GUFO_DRAFT_MODEL="$draft"
                SOLVER_GUFO_DRAFT_KIND="dspark"
                SOLVER_GUFO_REASONS+=("DSpark drafter: $(basename "$draft")")
            else
                SOLVER_GUFO_REASONS+=("no DSpark drafter found; plain decode")
            fi
            ;;
        qwen)
            if draft=$(llm_find_sidecar "$model" dflash); then
                if _gufo_drafter_usable dflash "$draft"; then
                    SOLVER_GUFO_SPEC="dflash2"
                    SOLVER_GUFO_DRAFT_MODEL="$draft"
                    SOLVER_GUFO_DRAFT_KIND="dflash"
                    SOLVER_GUFO_REASONS+=("DFlash2 drafter: $(basename "$draft")")
                else
                    SOLVER_GUFO_REASONS+=("DFlash2 drafter rejected: $SOLVER_GUFO_DRAFTER_REJECT; plain decode")
                fi
            else
                SOLVER_GUFO_REASONS+=("no DFlash2 drafter found; plain decode")
            fi
            ;;
        qwen-flash-next)
            if draft=$(llm_find_sidecar "$model" dflash) && _gufo_drafter_usable dflash "$draft"; then
                SOLVER_GUFO_SPEC="dflash2"
                SOLVER_GUFO_DRAFT_MODEL="$draft"
                SOLVER_GUFO_DRAFT_KIND="dflash"
                SOLVER_GUFO_REASONS+=("DFlash2 drafter: $(basename "$draft")")
            elif draft=$(llm_find_sidecar "$model" mtp) && _gufo_drafter_usable flash-next-mtp "$draft"; then
                SOLVER_GUFO_SPEC="mtp"
                SOLVER_GUFO_DRAFT_MODEL="$draft"
                SOLVER_GUFO_DRAFT_KIND="mtp"
                SOLVER_GUFO_REASONS+=("MTP sidecar: $(basename "$draft")")
            else
                local nextn
                nextn=$(gguf_get nextn_predict_layers 0)
                if (( nextn > 0 )); then
                    SOLVER_GUFO_SPEC="mtp"
                    SOLVER_GUFO_DRAFT_MODEL="embedded"
                    SOLVER_GUFO_DRAFT_KIND="mtp"
                    SOLVER_GUFO_REASONS+=("embedded MTP head (nextn_predict_layers=$nextn)")
                else
                    SOLVER_GUFO_REASONS+=("no DFlash2/MTP drafter found; plain decode")
                fi
            fi
            ;;
        laguna)
            if draft=$(llm_find_sidecar "$model" dflash); then
                SOLVER_GUFO_SPEC="dflash2"
                SOLVER_GUFO_DRAFT_MODEL="$draft"
                SOLVER_GUFO_DRAFT_KIND="dflash"
                SOLVER_GUFO_REASONS+=("DFlash2 drafter: $(basename "$draft")")
            else
                SOLVER_GUFO_REASONS+=("no DFlash drafter found; plain decode")
            fi
            ;;
        *)
            SOLVER_GUFO_REASONS+=("unknown family; speculative disabled")
            ;;
    esac

    if [[ -n "${LLM_SPEC:-}" && "${LLM_SPEC}" != "off" ]]; then
        SOLVER_GUFO_SPEC="$LLM_SPEC"
        SOLVER_GUFO_REASONS+=("speculative mode forced: $LLM_SPEC")
    fi
}

solve_gufo_config() {
    local model="$1"
    SOLVER_GUFO_REASONS=()
    _gufo_detect_family "$model"

    SOLVER_GUFO_CTX="${LLM_CTX:-0}"
    SOLVER_GUFO_SESSIONS="${LLM_SESSIONS:-1}"
    SOLVER_GUFO_KV_CACHE="${LLM_KV_CACHE:-f16}"
    SOLVER_GUFO_SPARSE="${LLM_SPARSE_ATTN:-auto}"

    _gufo_solve_spec "$model"

    local draft_default=7
    [[ "$SOLVER_GUFO_FAMILY" == "laguna" ]] && draft_default=2
    SOLVER_GUFO_DRAFT_TOKENS="${LLM_DRAFT_TOKENS:-$draft_default}"

    # A bias supplies the default for the dimensions it steers. An explicit
    # --kv-cache or --ctx outranks it: the narrower request wins.
    local bias="${LLM_BIAS:-context}"
    local kv_explicit=0
    [[ -n "${LLM_KV_CACHE:-}" ]] && kv_explicit=1
    case "$bias" in
        context)
            # The context bias trades fidelity for room, and q8_0 KV is how a
            # row keeps 53% of its f16 bytes for about -3.7% decode.
            _gufo_bias_kv q8_0 "$bias" "$kv_explicit"
            _gufo_drop_spec "$bias"
            ;;
        quality)
            _gufo_bias_kv f16 "$bias" "$kv_explicit"
            (( kv_explicit )) || SOLVER_GUFO_REASONS+=("KV cache pinned to f16 (bias=$bias)")
            _gufo_drop_spec "$bias"
            ;;
        speed)
            (( SOLVER_GUFO_CTX == 0 )) && SOLVER_GUFO_CTX=65536
            _gufo_bias_kv f16 "$bias" "$kv_explicit"
            SOLVER_GUFO_SPARSE="off"
            _gufo_drop_spec "$bias"
            SOLVER_GUFO_REASONS+=("context 65536, sparse attention off (bias=speed)")
            ;;
        mtp)
            # Balanced: keep the drafter and the engine's f16 cache.
            ;;
    esac

    [[ "$SOLVER_GUFO_SPARSE" == "auto" ]] && SOLVER_GUFO_SPARSE="on"
    SOLVER_GUFO_REASONS+=("context: $( ((SOLVER_GUFO_CTX>0)) && echo "$SOLVER_GUFO_CTX" || echo native )")
    return 0
}
