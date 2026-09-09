"""Evaluator for the LZ77 compression optimization workflow.

Exposes ``evaluate(program_path) -> dict`` as required by OpenEvolve.

The candidate program must define:

    compress(data: bytes) -> bytes
    decompress(data: bytes) -> bytes

Scoring blends three objectives into a single ``combined_score`` in [0, 1]:

    * correctness  --- round-trip must be exact for every corpus sample
                      (decompress(compress(x)) == x). Any failure zeros the score.
    * ratio        --- average compressed size / original size, lower is better.
                      Mapped so the baseline lands near a fixed mid-range.
    * speed        --- wall-clock throughput on a fixed repetitive block,
                      higher is better.

The combined score is a product of a correctness gate and a weighted blend of
the ratio and speed terms. This gives the optimizer a smooth, monotonic signal
that rewards *both* smaller output and faster execution while never tolerating
a broken round-trip.
"""

from __future__ import annotations

import ast
import tempfile

import importlib.util
import time
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Benchmark corpus. Deterministic, self-contained, sized so a full evaluation
# completes in well under a second (thousands of evaluations per run).
# ---------------------------------------------------------------------------

_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "LZ77 replaces repeated byte strings with back-references into a sliding "
    "window, trading a little CPU for dramatically smaller output on text and "
    "other structured data. A good compressor finds long matches fast."
)

# Highly compressible: heavy repetition (long runs, repeated words).
_REPETITIVE = (
    b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    + _TEXT.encode("utf-8") * 16
    + bytes(range(256)) * 4
)

# Medium compressibility: natural-ish English with mild repetition.
_MEDIUM = (_TEXT * 10).encode("utf-8")

# Incompressible: pseudo-random bytes (deterministic LCG).
def _prng_block(n: int) -> bytes:
    state = 0x9E3779B9
    out = bytearray()
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out.append((state >> 16) & 0xFF)
    return bytes(out)


_RANDOM = _prng_block(512)

# The speed block: long enough to measure throughput, but not pathological.
_SPEED_BLOCK = _REPETITIVE  # ~2 KiB of highly redundant data

# Each corpus item is (name, bytes).
_CORPUS: List[Tuple[str, bytes]] = [
    ("repetitive", _REPETITIVE),
    ("medium", _MEDIUM),
    ("random", _RANDOM),
]


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Ratio anchors: the naive baseline compresses *repetitive* to ~0.03, *medium*
# to ~0.20 and leaves *random* at ~1.0. The blended average is ~0.4. We map
# ratio into [0, 1] linearly between RATIO_HIGH (worst) and RATIO_LOW (best).
_RATIO_LOW = 0.10
_RATIO_HIGH = 1.05

# Speed anchor: baseline compresses the ~8 KiB speed block in ~X ms. We score
# throughput against a generous target so there is clear room to improve.
_SPEED_REF_BYTES_PER_SEC = 300_000.0  # bytes/sec that maps to speed_term == 0.5

_WEIGHT_RATIO = 0.55
_WEIGHT_SPEED = 0.45


def _rewrite_relative_imports(source: str) -> str:
    """Rewrite relative imports in *source* to absolute imports.

    When the LLM generates code with ``from .something import ...`` patterns,
    ``spec_from_file_location`` + ``exec_module`` fails with
    ``attempted relative import beyond top-level package`` because the
    candidate is loaded as a standalone file, not as part of a package.

    This function rewrites ``from .X import Y`` → ``import X`` and
    ``from .X.Y import Z`` → ``import X.Y`` so the candidate can be
    executed without a package context.  If the rewritten import cannot be
    resolved at load time the evaluator will catch the resulting error and
    return a score of 0.0.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    rewriter = _RelativeImportRewriter(source)
    rewriter.visit(tree)

    if not rewriter.changes:
        return source

    lines = source.splitlines(keepends=True)
    for lineno, old_text, new_text in reversed(rewriter.changes):
        lines[lineno - 1] = lines[lineno - 1].replace(old_text, new_text)

    return "".join(lines)


class _RelativeImportRewriter(ast.NodeTransformer):
    """Rewrite relative ``from .X import Y`` statements to absolute imports."""

    def __init__(self, source: str = "") -> None:
        self.source = source
        self.changes: List[Tuple[int, str, str]] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.level <= 0 or node.module is None:
            return node

        # ``from .X.Y import Z`` → ``import X.Y``
        # ``from .X import Y``       → ``import X``
        abs_module = node.module.lstrip(".")
        if not abs_module:
            # ``from . import X`` – cannot resolve, leave as-is
            return node

        old_line = self._original_line(node)
        new_line = f"import {abs_module}"
        self.changes.append((node.lineno, old_line.strip(), new_line))
        return node

    def _original_line(self, node: ast.ImportFrom) -> str:
        try:
            segment = ast.get_source_segment(self.source, node)
        except (ValueError, TypeError):
            segment = None
        return segment or ""


def _load(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not build import spec for candidate program")
    module = importlib.util.module_from_spec(spec)

    # Pre-process the candidate source so that relative imports (which
    # fail when the file is loaded as a standalone module) are rewritten
    # to absolute imports before exec_module is called.
    try:
        with open(program_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        rewritten = _rewrite_relative_imports(source)
        if rewritten is not source:
            _tmp = tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w", encoding="utf-8"
            )
            _tmp.write(rewritten)
            _tmp.close()
            program_path = _tmp.name
            spec = importlib.util.spec_from_file_location("candidate", program_path)
            if spec is None or spec.loader is None:
                raise ImportError("could not rebuild import spec after rewrite")
            module = importlib.util.module_from_spec(spec)
    except Exception:  # pragma: no cover - defensive
        pass

    spec.loader.exec_module(module)
    return module


def _ratio_term(ratio: float) -> float:
    """Map a compression ratio into [0, 1]; smaller ratio -> higher score."""
    clamped = max(_RATIO_LOW, min(_RATIO_HIGH, ratio))
    return (_RATIO_HIGH - clamped) / (_RATIO_HIGH - _RATIO_LOW)


def _speed_term(bytes_per_sec: float) -> float:
    """Map throughput to [0, 1] with a soft cap."""
    v = bytes_per_sec / _SPEED_REF_BYTES_PER_SEC
    return min(1.0, v / (1.0 + v) * 2.0)  # saturates toward 1.0


def evaluate(program_path: str) -> dict:
    """Score a candidate LZ77 program. Always returns a dict of numeric metrics."""
    try:
        module = _load(program_path)
    except Exception as exc:  # unparsable / crashing candidate
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "ratio": 1.0,
            "speed_mbps": 0.0,
            "error": str(exc)[:200],
        }

    compress = getattr(module, "compress", None)
    decompress = getattr(module, "decompress", None)
    if not callable(compress) or not callable(decompress):
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "ratio": 1.0,
            "speed_mbps": 0.0,
            "error": "candidate must define compress(data) and decompress(data)",
        }

    # --- Correctness gate + ratio on the full corpus -----------------------
    ratio_sum = 0.0
    ratio_count = 0
    for _name, block in _CORPUS:
        try:
            enc = compress(block)
            dec = decompress(enc)
        except Exception as exc:
            return {
                "combined_score": 0.0,
                "correctness": 0.0,
                "ratio": 1.0,
                "speed_mbps": 0.0,
                "error": f"compress/decompress raised: {type(exc).__name__}: {str(exc)[:120]}",
            }
        if dec != block:
            return {
                "combined_score": 0.0,
                "correctness": 0.0,
                "ratio": 1.0,
                "speed_mbps": 0.0,
                "error": "round-trip mismatch (decompress(compress(x)) != x)",
            }
        if len(block) > 0:
            ratio_sum += len(enc) / len(block)
            ratio_count += 1

    avg_ratio = ratio_sum / max(1, ratio_count)

    # --- Speed on the dedicated block --------------------------------------
    # Two timed passes; keep the best to reduce scheduler noise.
    best_bps = 0.0
    for _ in range(2):
        t0 = time.perf_counter()
        compress(_SPEED_BLOCK)
        dt = time.perf_counter() - t0
        if dt > 0:
            best_bps = max(best_bps, len(_SPEED_BLOCK) / dt)

    r = _ratio_term(avg_ratio)
    s = _speed_term(best_bps)

    combined = _WEIGHT_RATIO * r + _WEIGHT_SPEED * s

    return {
        "combined_score": round(combined, 6),
        "correctness": 1.0,
        "ratio": round(avg_ratio, 6),
        "ratio_score": round(r, 6),
        "speed_bps": round(best_bps, 2),
        "speed_score": round(s, 6),
        "runs": 1.0,
    }