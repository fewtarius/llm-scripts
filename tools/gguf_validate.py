#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
"""Reject a speculative drafter the loader cannot accept.

The solver discovers drafters by filename. That is not enough: a sidecar can
be the right family and still use tensor formats or a layout the runner does
not implement. Loading it then fails the whole model, so the solver asks this
tool first and falls back to plain decode with the reason.

Exit 0 when usable; exit 1 and print the reason to stderr otherwise.

Usage: validate-gufo-drafter.py --family flash-next-mtp --file PATH
"""
import argparse
import struct
import sys

# Same GGML type IDs the loaders use.
F32, F16 = 0, 1
Q8_0, Q5_0, Q5_1 = 8, 6, 7
Q4_K, Q5_K, Q6_K = 12, 13, 14
BF16 = 30

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16",
}

# The Flash-Next loader accepts these for dense, mixer and embedding tensors.
DENSE_OK = {F32, F16, Q8_0, BF16}
# ... and these for routed expert tensors.
EXPERTS_OK = {Q4_K, Q5_K, Q6_K, Q5_1, Q8_0}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t in (10, 11, 12):
        f.read(8)
    elif t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            skip_value(f, et)
    else:
        raise ValueError(f"bad metadata type {t}")


def read_value(f, t):
    if t in (0, 7):
        return f.read(1)[0]
    if t in (1,):
        return struct.unpack("<b", f.read(1))[0]
    if t in (2,):
        return struct.unpack("<H", f.read(2))[0]
    if t in (3,):
        return struct.unpack("<h", f.read(2))[0]
    if t in (4,):
        return struct.unpack("<I", f.read(4))[0]
    if t in (5,):
        return struct.unpack("<i", f.read(4))[0]
    if t in (6,):
        return struct.unpack("<f", f.read(4))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, et) for _ in range(count)]
    if t in (10,):
        return struct.unpack("<Q", f.read(8))[0]
    if t in (11,):
        return struct.unpack("<q", f.read(8))[0]
    if t in (12,):
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"bad metadata type {t}")


def metadata(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        out = {}
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            out[key] = read_value(f, t)
        return out, n_tensors


def tensors(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            skip_value(f, t)
        out = []
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims)
            (ttype,) = struct.unpack("<I", f.read(4))
            f.read(8)
            out.append((name, ttype))
        return out


def fail(message):
    print(message, file=sys.stderr)
    return 1


def check_dflash(meta):
    """Mirror QwenDFlashWeights::LoadFromGguf's required metadata."""
    if meta.get("general.architecture") != "dflash":
        return ("drafter is not a dflash GGUF "
                f"(general.architecture={meta.get('general.architecture')!r})")
    required = (
        "embedding_length", "feed_forward_length", "attention.head_count",
        "attention.head_count_kv", "attention.key_length", "context_length",
        "block_count", "rope.freq_base",
    )
    missing = [f"dflash.{key}" for key in required
               if meta.get(f"dflash.{key}") is None]
    if meta.get("dflash.rope.dimension_count") is None:
        sections = meta.get("dflash.rope.dimension_sections")
        degenerate = (isinstance(sections, list) and len(sections) == 4
                      and sections[0] > 0 and sections[1:] == [0, 0, 0])
        if not degenerate:
            missing.append("dflash.rope.dimension_count")
    if missing:
        return "missing required metadata: " + ", ".join(missing)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    try:
        items = tensors(args.file)
    except Exception as exc:
        return fail(f"cannot read drafter: {exc}")

    names = {name for name, _ in items}

    if args.family == "flash-next-mtp":
        # Mirror what the loader binds for the nextn block: the two norms and
        # the fusion projection, plus a hyper-connection head mixer. The head
        # mixer is named nextn.hc_head_* in some exports and output_hc_* in
        # others; it is a distinct trained mixer under either name, so accept
        # both rather than reject a sidecar the loader can bind.
        for required in ("nextn.enorm.weight", "nextn.hnorm.weight",
                         "nextn.eh_proj.weight"):
            if not any(n.endswith(required) for n in names):
                return fail(f"sidecar has no {required}")
        head_names = ("nextn.hc_head_norm.weight", "output_hc_norm.weight")
        if not any(n.endswith(h) for n in names for h in head_names):
            return fail("sidecar has no hyper-connection head mixer "
                        "(nextn.hc_head_* or output_hc_*)")
        if not any(n.endswith(".hc_attn_down.weight") for n in names):
            return fail("sidecar has no hyper-connection mixer tensors")
        for name, ttype in items:
            allowed = EXPERTS_OK if name.endswith("_exps.weight") else DENSE_OK
            if ttype not in allowed:
                return fail(
                    f"{name} is {TYPE_NAMES.get(ttype, ttype)}, which the "
                    "runner cannot execute")
        return 0

    if args.family == "dflash":
        try:
            meta, _ = metadata(args.file)
        except Exception as exc:
            return fail(f"cannot read drafter metadata: {exc}")
        reason = check_dflash(meta)
        if reason:
            return fail(reason)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
