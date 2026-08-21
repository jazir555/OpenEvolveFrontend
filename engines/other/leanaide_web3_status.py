"""
Shared Web3 formal-status helpers for LeanAide/Lean integrations.
"""
from __future__ import annotations


from typing import Any, Dict, List


def default_web3_formal_status() -> Dict[str, Any]:
    """Return a stable default payload for Web3 formal verification status."""
    formal_capabilities = {
        "solidity_invariant_translation": False,
        "invariant_translation_verification": False,
        "symbolic_exploit_witness": False,
        "composite_exploit_verification": False,
    }
    return {
        "web3_formal_available": False,
        "web3_formal_verification_available": False,
        "web3_formal_tools": [],
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": False,
    }


def collect_web3_formal_status() -> Dict[str, Any]:
    """
    Collect Web3 formal status from root LeanAide integration when available.
    Falls back to a stable default schema when unavailable.
    """
    default_status = default_web3_formal_status()
    try:
        from leanaide_integration import create_integration

        status = create_integration().get_web3_formal_status()
        if not isinstance(status, dict):
            return default_status

        formal_capabilities = status.get("formal_capabilities")
        if not isinstance(formal_capabilities, dict):
            formal_capabilities = default_status["formal_capabilities"]

        web3_formal_tools = status.get("web3_formal_tools")
        if not isinstance(web3_formal_tools, list):
            web3_formal_tools = []
        web3_formal_tools = sorted(set(str(tool) for tool in web3_formal_tools if tool))

        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(value) for value in formal_capabilities.values()
        )

        return {
            "web3_formal_available": bool(
                status.get("web3_formal_available", inferred_formal_available)
            ),
            "web3_formal_verification_available": bool(
                status.get("web3_formal_verification_available", inferred_formal_available)
            ),
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "audit_exploit_verification_available": bool(
                status.get("audit_exploit_verification_available")
            ),
        }
    except Exception:
        return default_status


def merge_web3_formal_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge Web3 formal status keys into an existing payload."""
    merged = dict(payload)
    merged.update(collect_web3_formal_status())
    return merged
