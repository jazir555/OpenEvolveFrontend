"""Naive LZ77-style compressor (baseline candidate for OpenEvolve optimization).

The functions in this file implement a simple LZ77 compression scheme:
    compress(data: bytes) -> bytes
    decompress(data: bytes) -> bytes

Encoding format:
    A stream of tokens. Each token starts with a flag byte:
        0x00 followed by one data byte => literal
        0x01 followed by a 2-byte little-endian ``offset`` and 2-byte
        little-endian ``length`` => back-reference (copy of length bytes
        starting ``offset`` bytes before the current output position).

The implementation between ``# EVOLVE-BLOCK-START`` and ``# EVOLVE-BLOCK-END``
is the part that OpenEvolve may mutate. Everything outside the block is
considered stable scaffolding (imports, decoding helpers, public entrypoints,
the eval-friendly ``run()``).

Goal for the optimizer: maximize a fitness that blends
compression ratio and compress/decompress speed while preserving exact
round-trip correctness (decompress(compress(x)) == x for every input).
A great candidate will swap the naive O(n*window) match scan for a hash
table / chain table that finds the longest match in amortized O(log n) or
O(1) time per position, and may also tweak the token format / decision
heuristics (lazy matching, hash chains, 3-byte vs 4-byte literal windows,
etc.) to trade a tiny amount of ratio for a big speed win.
"""

from __future__ import annotations

import struct
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Stable helpers (outside the EVOLVE-BLOCK)
# ---------------------------------------------------------------------------

_FLAG_LITERAL = 0x00
_FLAG_MATCH = 0x01

_MAX_OFFSET = 0xFFFF  # 2-byte offset
_MAX_LENGTH = 0xFFFF  # 2-byte length
_WINDOW_SIZE = _MAX_OFFSET  # 64 KiB sliding window
_MIN_MATCH = 3
_MAX_MATCH = 258  # classic LZ77 max match length


def _emit_literal(out: bytearray, b: int) -> None:
    out.append(_FLAG_LITERAL)
    out.append(b & 0xFF)


def _emit_match(out: bytearray, offset: int, length: int) -> None:
    out.append(_FLAG_MATCH)
    out += struct.pack("<HH", offset, length)


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------

def compress(data: bytes) -> bytes:
    """Compress ``data`` to an LZ77 token stream.

    The naive baseline below performs an O(n*window_size) forward scan for the
    longest match starting at each position. It is correct but slow on inputs
    longer than a few KiB. OpenEvolve may replace the whole body.
    """
    out = bytearray()
    n = len(data)
    pos = 0
    while pos < n:
        best_len = 0
        best_off = 0
        # Search backwards in the sliding window for the longest prefix match.
        start = max(0, pos - _WINDOW_SIZE)
        max_match_here = min(_MAX_MATCH, n - pos)
        for cand in range(start, pos):
            # Compute the run length match.
            length = 0
            while (
                length < max_match_here
                and data[cand + length] == data[pos + length]
            ):
                length += 1
            if length > best_len:
                best_len = length
                best_off = pos - cand
                if best_len == max_match_here:
                    break  # can't do better
        if best_len >= _MIN_MATCH:
            _emit_match(out, best_off, best_len)
            pos += best_len
        else:
            _emit_literal(out, data[pos])
            pos += 1
    return bytes(out)


def decompress(data: bytes) -> bytes:
    """Inverse of :func:`compress`. Reads the token stream and rebuilds bytes."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        flag = data[i]
        i += 1
        if flag == _FLAG_LITERAL:
            out.append(data[i])
            i += 1
        elif flag == _FLAG_MATCH:
            offset, length = struct.unpack_from("<HH", data, i)
            i += 4
            base = len(out) - offset
            for k in range(length):
                out.append(out[base + k])
        else:
            raise ValueError(f"unknown flag byte 0x{flag:02x} at offset {i - 1}")
    return bytes(out)


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-END
# ---------------------------------------------------------------------------


def run() -> bytes:
    """Eval-friendly entry point used by OpenEvolve's default evaluator template.

    Not used by the custom evaluator (which calls ``compress`` / ``decompress``
    directly), but kept so the default scaffold still scores sensibly.
    """
    sample = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc"
    return decompress(compress(sample))


if __name__ == "__main__":
    # Quick smoke test: round-trip on a tiny sample.
    demo = b"the quick brown fox jumps over the lazy dog. " * 16
    enc = compress(demo)
    dec = decompress(enc)
    assert dec == demo, "round-trip failed"
    print(f"baseline smoke OK: {len(demo)} -> {len(enc)} bytes "
          f"(ratio {len(enc) / len(demo):.3f})")
