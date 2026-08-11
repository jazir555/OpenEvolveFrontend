"""
Shared Web3 formal evidence helpers for Z3 + Lean integration surfaces.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


def _extract_lean_spec(translation: Optional[Dict[str, Any]]) -> str:
    if not isinstance(translation, dict):
        return ""
    lean_spec = translation.get("lean_spec")
    if not isinstance(lean_spec, str):
        return ""
    return lean_spec.strip()


def _extract_theorem_name(lean_spec: str) -> Optional[str]:
    if not lean_spec:
        return None
    first_line = lean_spec.splitlines()[0].strip()
    if first_line.startswith("theorem "):
        name = first_line[len("theorem ") :].split(":", 1)[0].strip()
        return name or None
    return None


def _base_lean_evidence(translation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    lean_spec = _extract_lean_spec(translation)
    return {
        "available": False,
        "attempted": bool(lean_spec),
        "verified": False,
        "status": "pending" if lean_spec else "missing_lean_spec",
        "method": "none",
        "theorem": _extract_theorem_name(lean_spec),
        "contains_placeholder_proof": "sorry" in lean_spec,
        "errors": [],
        "warnings": [],
        "confidence": 0.0,
    }


def build_web3_formal_evidence(
    verification: Optional[Dict[str, Any]],
    witness: Optional[Dict[str, Any]],
    lean_proof_verification: Dict[str, Any],
) -> Dict[str, Any]:
    """Build consistent composite Web3 formal evidence payload."""
    witness = witness if isinstance(witness, dict) else {}
    return {
        "z3_invariant_verification": verification,
        "lean_proof_verification": lean_proof_verification,
        "symbolic_exploit_witness": {
            "satisfiable": bool(witness.get("satisfiable", False)),
            "status": witness.get("status"),
            "model_available": bool(witness.get("model")),
        },
    }


async def verify_web3_lean_proof_async(
    translation: Optional[Dict[str, Any]],
    *,
    lean_service: Optional[Any] = None,
    use_real_lean: bool = True,
) -> Dict[str, Any]:
    """
    Verify Lean theorem scaffold attached to translated Solidity invariants.

    Prefer direct Lean 4 service when available, then fallback to
    root-level LeanAIDE integration compatibility layer.
    """
    evidence = _base_lean_evidence(translation)
    lean_spec = _extract_lean_spec(translation)
    if not lean_spec:
        return evidence

    if lean_service is not None and hasattr(lean_service, "verify"):
        evidence["available"] = True
        try:
            lean_result = await lean_service.verify(lean_spec)
            status = getattr(getattr(lean_result, "status", None), "value", None)
            evidence.update(
                {
                    "verified": bool(getattr(lean_result, "success", False)),
                    "status": status or ("verified" if getattr(lean_result, "success", False) else "failed"),
                    "method": "lean4_service",
                    "errors": list(getattr(lean_result, "errors", []) or []),
                    "warnings": list(getattr(lean_result, "warnings", []) or []),
                    "confidence": 1.0 if bool(getattr(lean_result, "success", False)) else 0.0,
                }
            )
            return evidence
        except Exception as exc:  # pragma: no cover - environment-dependent
            evidence["errors"].append(f"Lean service verification failed: {exc}")

    try:
        from lean4_integration import create_lean4_service

        service = create_lean4_service()
        evidence["available"] = True
        lean_result = await service.verify(lean_spec)
        status = getattr(getattr(lean_result, "status", None), "value", None)
        evidence.update(
            {
                "verified": bool(getattr(lean_result, "success", False)),
                "status": status or ("verified" if getattr(lean_result, "success", False) else "failed"),
                "method": "lean4_integration",
                "errors": list(getattr(lean_result, "errors", []) or []),
                "warnings": list(getattr(lean_result, "warnings", []) or []),
                "confidence": 1.0 if bool(getattr(lean_result, "success", False)) else 0.0,
            }
        )
        return evidence
    except Exception as exc:  # pragma: no cover - environment-dependent
        evidence["errors"].append(f"Lean4 integration unavailable: {exc}")

    try:
        from leanaide_integration import create_integration

        def _fallback_verify() -> Dict[str, Any]:
            integration = create_integration()
            result = integration.verify_theorem(lean_spec, use_real_lean=use_real_lean)
            return {
                "available": bool(getattr(integration, "is_available", False)),
                "result": result,
            }

        fallback_payload = await asyncio.to_thread(_fallback_verify)
        fallback = fallback_payload.get("result", {})
        evidence["available"] = bool(fallback_payload.get("available", False))
        evidence.update(
            {
                "verified": bool(fallback.get("verified", False)),
                "status": "verified" if bool(fallback.get("verified", False)) else "failed",
                "method": str(fallback.get("method", "leanaide_integration")),
                "errors": list(fallback.get("errors", []) or []),
                "confidence": float(fallback.get("confidence", 0.0) or 0.0),
            }
        )
        return evidence
    except Exception as exc:  # pragma: no cover - environment-dependent
        evidence["errors"].append(f"LeanAIDE fallback unavailable: {exc}")
        evidence["status"] = "unavailable"
        evidence["method"] = "unavailable"
        return evidence


def verify_web3_lean_proof(
    translation: Optional[Dict[str, Any]],
    *,
    use_real_lean: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for Lean proof evidence generation.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            verify_web3_lean_proof_async(
                translation,
                lean_service=None,
                use_real_lean=use_real_lean,
            )
        )

    evidence = _base_lean_evidence(translation)
    if evidence["attempted"]:
        evidence["status"] = "event_loop_running"
        evidence["method"] = "deferred"
        evidence["errors"].append(
            "Synchronous Lean verification deferred because an event loop is already running."
        )
    return evidence

