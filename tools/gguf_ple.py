#!/usr/bin/env python3
"""Read the actual size of the per_layer_token_embd.weight (PLE) tensor from
a (possibly multi-file) GGUF model.

Scans shard files for the PLE tensor without loading tensor data (header-only
parse). Uses gguf.GGUFReader if available (fast), falls back to a pure-Python
struct-based parser for systems without the gguf package.

Prints the PLE tensor's byte size, or 0 if not found.
"""
import struct
import sys
import glob
import os

# ---------------------------------------------------------------------------
# GGUF v3 constants and helpers
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
QK_K = 256  # GGML default block size for K-quant types

# GGMLQuantizationType -> (block_size, type_size_in_bytes)
# Mirrors GGML_QUANT_SIZES from llama.cpp/gguf-py constants.py
GGML_TYPE_INFO = {
    0:  (1, 4),                      # F32
    1:  (1, 2),                      # F16
    2:  (32, 2 + 16),                # Q4_0
    3:  (32, 2 + 2 + 16),            # Q4_1
    6:  (32, 2 + 4 + 16),            # Q5_0
    7:  (32, 2 + 2 + 4 + 16),        # Q5_1
    8:  (32, 2 + 32),                # Q8_0
    9:  (32, 4 + 4 + 32),            # Q8_1
    10: (QK_K, 2 + 2 + QK_K // 16 + QK_K // 4),                        # Q2_K
    11: (QK_K, 2 + QK_K // 4 + QK_K // 8 + 12),                         # Q3_K
    12: (QK_K, 2 + 2 + QK_K // 2 + 12),                                 # Q4_K
    13: (QK_K, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),                     # Q5_K
    14: (QK_K, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),                 # Q6_K
    15: (QK_K, 4 + QK_K + QK_K // 8),                                   # Q8_K
    16: (QK_K, 2 + QK_K // 4),                                          # IQ2_XXS
    17: (QK_K, 2 + QK_K // 4 + QK_K // 32),                             # IQ2_XS
    18: (QK_K, 2 + QK_K // 4 + QK_K // 8),                              # IQ3_XXS
    19: (QK_K, 2 + QK_K // 8 + QK_K // 16),                             # IQ1_S
    20: (32, 2 + 16),                                                  # IQ4_NL
    21: (QK_K, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),             # IQ3_S
    22: (QK_K, 2 + QK_K // 4 + QK_K // 16),                             # IQ2_S
    23: (QK_K, 2 + 2 + QK_K // 2 + QK_K // 64),                         # IQ4_XS
    24: (1, 1),                                                        # I8
    25: (1, 2),                                                        # I16
    26: (1, 4),                                                        # I32
    27: (1, 8),                                                        # I64
    28: (1, 8),                                                        # F64
    29: (QK_K, QK_K // 8 + QK_K // 16 + QK_K // 32),                    # IQ1_M
    30: (1, 2),                                                        # BF16
    34: (1, 5),                                                        # TQ1_0
    35: (1, 6),                                                        # TQ2_0
    39: (32, 2 + 8),                                                   # MXFP4
    40: (32, 2 + 8),                                                   # NVFP4
    41: (32, 2 + 16),                                                  # Q1_0
}

# GGUFValueType sizes for KV metadata skipping (uint64 = 8 bytes for length-prefixed strings)
KV_ELT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_shard_header(path):
    """Read the 24-byte GGUF header. Returns (n_tensors, n_kv, magic, version)."""
    with open(path, 'rb') as f:
        data = f.read(24)
    if len(data) < 24:
        return 0, 0, 0, 0
    magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', data)
    return n_tensors, n_kv, magic, version


def get_shard_paths(model_path):
    """Build sorted list of shard file paths for a (possibly multi-file) GGUF model."""
    model_dir = os.path.dirname(model_path) or '.'
    model_base = os.path.basename(model_path)
    pattern = os.path.join(model_dir, model_base.replace('-00001-of-', '-*-of-'))
    files = sorted(glob.glob(pattern))
    if not files:
        if os.path.exists(model_path):
            files = [model_path]
    return files


def ple_size_gguf(files):
    """Use gguf.GGUFReader to find the PLE tensor byte size.

    Skips metadata-only shards (n_tensors=0) to avoid the slow GGUFReader
    initialization on header-only files.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return 0

    for fpath in files:
        n_tensors, n_kv, magic, version = read_shard_header(fpath)
        if magic != GGUF_MAGIC:
            continue
        if n_tensors == 0:
            continue
        r = GGUFReader(fpath)
        for t in r.tensors:
            if t.name == 'per_layer_token_embd.weight':
                return t.n_bytes
    return 0


def _skip_kv_section(f, n_kv):
    """Skip n_kv KV metadata entries in a GGUF v3 file."""
    for _ in range(n_kv):
        # key: uint64 length + bytes
        key_len = struct.unpack('<Q', f.read(8))[0]
        f.read(key_len)
        kv_type = struct.unpack('<I', f.read(4))[0]
        if kv_type == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif kv_type == 9:  # array
            at = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            es = KV_ELT_SIZE.get(at, 8)
            if at == 8:  # array of strings
                for _ in range(alen):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
            else:
                f.read(alen * es)
        else:
            es = KV_ELT_SIZE.get(kv_type)
            if es is not None:
                f.read(es)


def ple_size_struct(files):
    """Pure-Python fallback: parse GGUF tensor directory with struct module.

    No external dependencies. Reads only the header + tensor info (not tensor
    data), so it's fast even for large files.
    """

    for fpath in files:
        n_tensors, n_kv, magic, version = read_shard_header(fpath)
        if magic != GGUF_MAGIC:
            continue
        if n_tensors == 0:
            continue

        with open(fpath, 'rb') as f:
            # Skip header (24 bytes) + KV metadata
            f.seek(24)
            _skip_kv_section(f, n_kv)

            # Read tensor info (v3 format: name_str, n_dims u32, dims u64[], type u32, offset u64)
            for _ in range(n_tensors):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8', errors='replace')
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                tensor_type = struct.unpack('<I', f.read(4))[0]
                data_offset = struct.unpack('<Q', f.read(8))[0]

                n_elements = 1
                for d in dims:
                    n_elements *= d

                blk, tsz = GGML_TYPE_INFO.get(tensor_type, (1, 4))
                tensor_bytes = n_elements * tsz // blk

                if name == 'per_layer_token_embd.weight':
                    return tensor_bytes

    return 0


def main():
    if len(sys.argv) < 2:
        print(0)
        return

    model_path = sys.argv[1]
    files = get_shard_paths(model_path)
    if not files:
        print(0)
        return

    # Prefer gguf library (most battle-tested)
    ple = ple_size_gguf(files)
    if ple > 0:
        print(ple)
        return

    # Fallback: pure struct parsing (no external deps)
    ple = ple_size_struct(files)
    if ple > 0:
        print(ple)
        return

    print(0)


if __name__ == '__main__':
    main()
