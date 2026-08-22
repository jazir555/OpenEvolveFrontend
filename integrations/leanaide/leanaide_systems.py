"""
LeanAIDE Systems Module

Provides core systems for Lean 4 theorem prover integration, including a
genuine structural / syntax-level proof checker. The checker never returns
success unconditionally: it parses the proof, verifies delimiter balance,
checks for incomplete proofs (``sorry`` / ``admit``), and inspects tactic
sequencing. When the Lean 4 toolchain is available it can be upgraded to a
real compiler verification via :mod:`engines.other.lean4_integration`.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# A conservative catalog of well-known Lean 4 tactics. Used by the structural
# checker to flag obviously unknown tactic tokens (a signal of a malformed
# proof rather than a guarantee of correctness).
KNOWN_TACTICS = {
    "intro", "intros", "apply", "exact", "refine", "rw", "rewrite", "simp",
    "simpa", "dsimp", "cases", "case", "induction", "assumption", "refl",
    "rfl", "tauto", "ring", "ring_nf", "linarith", "nlinarith", "norm_num",
    "field_simp", "noncomm_ring", "aesop", "contradiction", "constructor",
    "use", "have", "let", "show", "calc", "conv", "unfold", "omega", "lia",
    "decide", "done", "admit", "sorry", "by_cases", "by_contra", "split",
    "left", "right", "exists", "clear", "rename", "generalize", "revert",
    "specialize", "subst", "replace", "trans", "symm", "cc", "grind",
    "positivity", "compute", "whnf", "infer", "trace", "abort", "skip",
    "focus", "all_goals", "any_goals", "repeat", "try", "first", "solve",
    "iterate", "swap", "rotate", "rename_var",
}

DECL_RE = re.compile(r"\b(theorem|lemma|def|example|axiom|structure|class|instance)\b")
BY_RE = re.compile(r":=\s*by\b|\bby\b")
ERROR_TOKENS_RE = re.compile(r"\b(sorry|admit)\b")
BRACE_OPEN = "([{"
BRACE_CLOSE = ")]}"
BRACE_PAIRS = {")": "(", "]": "[", "}": "{"}


def _strip_lean_comments(code: str) -> str:
    """Remove Lean line (``--``) and block (``/- ... -/``) comments.

    A rough scanner that avoids stripping comment markers that appear inside
    string literals. Good enough for a structural balance check.
    """
    out: List[str] = []
    i = 0
    n = len(code)
    in_string = False
    in_block = False
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if in_block:
            if ch == "-" and nxt == "/":
                in_block = False
                i += 2
                continue
            out.append(" ")
            i += 1
            continue
        if in_string:
            out.append(ch)
            if ch == '"' and (i == 0 or code[i - 1] != "\\"):
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "-":
            in_block = True
            i += 2
            continue
        if ch == "-" and nxt == "-":
            while i < n and code[i] != "\n":
                i += 1
                break
            out.append("\n")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_tactics(code: str) -> List[str]:
    """Pull tactic tokens from a ``by`` proof block.

    Returns the first whitespace-delimited token of each indented line that
    follows a ``by`` keyword, until indentation drops back to the declaration
    level. This is a heuristic structural check, not a full tactic parser.
    """
    tactics: List[str] = []
    in_block = False
    base_indent: Optional[int] = None
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--") or stripped.startswith("/-"):
            continue
        indent = len(line) - len(line.lstrip())
        if BY_RE.search(line) and not in_block:
            in_block = True
            base_indent = indent
            continue
        if in_block:
            if base_indent is not None and indent <= base_indent and stripped:
                in_block = False
            else:
                token = re.split(r"[\s(]", stripped, 1)[0]
                if token:
                    tactics.append(token)
                continue
    return tactics


def check_lean_proof_structural(proof: str) -> Dict[str, Any]:
    """Genuine structural / syntax-level Lean proof check.

    This is the canonical structural checker used both directly by the
    LeanAIDE proof checkers and as the graceful-degradation fallback of the
    Lean 4 verification engine when the toolchain is unavailable.

    Returns a dict with at least ``valid`` (bool), ``errors`` (list),
    ``warnings`` (list) and ``method`` (``"structural"``). ``valid`` is only
    ``True`` when no hard errors were found; it is never set unconditionally.
    """
    result: Dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "method": "structural",
        "details": {},
    }

    if not isinstance(proof, str) or not proof.strip():
        result["errors"].append("Empty proof: no Lean code provided")
        return result

    code = _strip_lean_comments(proof)

    # 1. Delimiter balance (ignoring comments / strings).
    stack: List[str] = []
    for ch in code:
        if ch in BRACE_OPEN:
            stack.append(ch)
        elif ch in BRACE_CLOSE:
            if not stack or stack[-1] != BRACE_PAIRS[ch]:
                result["errors"].append(f"Unbalanced delimiter '{ch}'")
                break
            stack.pop()
    else:
        if stack:
            result["errors"].append(
                "Unbalanced opening delimiters: " + "".join(stack)
            )

    # 2. A declaration should be present.
    has_decl = bool(DECL_RE.search(code))
    if not has_decl:
        result["warnings"].append(
            "No theorem/lemma/def/example declaration found"
        )

    # 3. A proof must be assigned (`:=`) and ideally use `by`.
    if ":=" not in code:
        result["warnings"].append(
            "No ':=' assignment; declaration appears incomplete"
        )

    # 4. Incomplete proofs must not be reported as valid.
    if ERROR_TOKENS_RE.search(code):
        result["errors"].append(
            "Proof contains 'sorry'/'admit' (incomplete proof) "
            "and cannot be verified"
        )

    # 5. Tactic sequencing sanity check.
    tactics = _extract_tactics(code)
    result["details"]["tactic_count"] = len(tactics)
    unknown = [
        t for t in tactics
        if t and t not in KNOWN_TACTICS and not t.startswith("(")
    ]
    if unknown:
        result["warnings"].append(
            "Unrecognized tactic tokens: "
            + ", ".join(sorted(set(unknown))[:10])
        )

    # 6. Dependency import hint.
    if "import" not in code:
        result["warnings"].append(
            "No 'import' statement; Lean may require dependencies "
            "(e.g. Mathlib)"
        )

    if not result["errors"]:
        result["valid"] = True
    return result


class StructuralLeanChecker:
    """Object-oriented wrapper around :func:`check_lean_proof_structural`."""

    def check(self, proof: str) -> Dict[str, Any]:
        return check_lean_proof_structural(proof)


@dataclass
class LeanSystemConfig:
    """Configuration for Lean systems"""
    auto_imports: List[str] = field(default_factory=list)
    strict_mode: bool = True


class LeanSystemCore:
    """Core Lean system component"""

    def __init__(self, config: Optional[LeanSystemConfig] = None):
        self.config = config or LeanSystemConfig()
        logger.info("Lean System Core initialized")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        return {"result": "processed", "data": input_data}


class LeanProofChecker:
    """Proof checking component.

    Performs a genuine structural / syntax-level check of a Lean proof.
    It never returns ``{"valid": True}`` unconditionally; the result reflects
    what the structural analyzer actually determined.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._checker = StructuralLeanChecker()
        logger.info("Lean Proof Checker initialized")

    def check(self, proof: str) -> Dict[str, Any]:
        """Check a proof structurally.

        Returns ``{"valid": bool, "errors": [...], "warnings": [...],
        "method": "structural", "proof": <str>}``. ``valid`` is ``False``
        whenever structural errors (e.g. unbalanced delimiters, ``sorry``)
        are detected.
        """
        res = self._checker.check(proof)
        snippet = proof if isinstance(proof, str) else str(proof)
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
        return {
            "valid": res["valid"],
            "errors": res["errors"],
            "warnings": res["warnings"],
            "method": res["method"],
            "details": res.get("details", {}),
            "proof": snippet,
        }
