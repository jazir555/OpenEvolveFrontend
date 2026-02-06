"""
Root-level Z3 API compatibility facade.

This module preserves the lightweight ``Z3API`` surface while wiring in
Web3 smart-contract formal verification helpers from ``z3prover_integration``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import (
        Z3Constraint,
        Z3ConstraintType,
        Z3Variable,
        get_z3_solver_engine,
        solve_smart_contract_exploit_witness,
        translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation,
    )
    Z3_INTEGRATION_AVAILABLE = True
except ImportError:
    Z3Constraint = None
    Z3ConstraintType = None
    Z3Variable = None
    get_z3_solver_engine = None
    solve_smart_contract_exploit_witness = None
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    Z3_INTEGRATION_AVAILABLE = False

# Compatibility re-export for modules that expect knowledge API symbols here.
try:
    from knowledge_engine.integrations.z3_api import app, create_z3_knowledge_app
except Exception:
    app = None
    create_z3_knowledge_app = None


@dataclass
class Z3APIConfig:
    """Configuration for Z3 API facade."""

    host: str = "localhost"
    port: int = 5000
    timeout: float = 60.0


class Z3API:
    """Backward-compatible Z3 API facade with Web3 helpers."""

    def __init__(self, config: Optional[Z3APIConfig] = None):
        self.config = config or Z3APIConfig()
        self._last_result: Optional[Any] = None
        self._solver = get_z3_solver_engine() if get_z3_solver_engine is not None else None
        logger.info("Z3 API initialized")

    def solve(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a formula payload with optional native Z3 integration."""
        if not isinstance(formula, dict):
            return {"result": "error", "error": "formula must be a dictionary"}

        if (
            self._solver is None
            or Z3Variable is None
            or Z3Constraint is None
            or Z3ConstraintType is None
        ):
            return {"result": "sat", "formula": formula, "backend": "stub"}

        variables_payload = formula.get("variables", [])
        constraints_payload = formula.get("constraints", [])
        if not isinstance(variables_payload, list) or not isinstance(constraints_payload, list):
            return {"result": "error", "error": "variables/constraints must be lists"}

        try:
            variables: List[Any] = []
            for var in variables_payload:
                if not isinstance(var, dict):
                    continue
                type_name = str(var.get("type", "integer")).upper()
                z3_type = getattr(Z3ConstraintType, type_name, Z3ConstraintType.INTEGER)
                variables.append(Z3Variable(str(var.get("name", "x")), z3_type))

            constraints = [
                Z3Constraint(str(expr), Z3ConstraintType.BOOLEAN)
                for expr in constraints_payload
            ]

            self._last_result = self._solver.solve_constraints(variables, constraints)
            status = getattr(getattr(self._last_result, "status", None), "value", "unknown")
            return {
                "result": status,
                "satisfiable": getattr(self._last_result, "satisfiable", None),
                "model": (
                    self._last_result.model.assignments
                    if getattr(self._last_result, "model", None) is not None
                    else None
                ),
                "formula": formula,
                "backend": "z3",
            }
        except Exception as exc:
            logger.error("Z3 solve failed: %s", exc)
            return {"result": "error", "error": str(exc), "formula": formula}

    def get_model(self) -> Dict[str, Any]:
        """Return model from last solve call when available."""
        if self._last_result is None or getattr(self._last_result, "model", None) is None:
            return {"model": {}}
        return {"model": self._last_result.model.assignments}

    def get_proof(self) -> Dict[str, Any]:
        """Return proof from last solve call when available."""
        proof = getattr(self._last_result, "proof", None) if self._last_result is not None else None
        return {"proof": proof or {}}

    def get_web3_status(self) -> Dict[str, Any]:
        """Expose Web3 formal capability status from the Z3 facade."""
        formal_capabilities = {
            "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification": verify_solidity_invariant_translation is not None,
            "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
            "composite_exploit_verification": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
        }
        web3_formal_tools = []
        if formal_capabilities["solidity_invariant_translation"]:
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities["symbolic_exploit_witness"]:
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities["composite_exploit_verification"]:
            web3_formal_tools.append("z3_web3_audit_exploit_verification")
        return {
            "available": translate_solidity_assignment_to_z3 is not None
            and solve_smart_contract_exploit_witness is not None,
            "solidity_invariant_translation_available": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification_available": verify_solidity_invariant_translation is not None,
            "exploit_witness_available": solve_smart_contract_exploit_witness is not None,
            "audit_exploit_verification_available": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
        }

    def translate_solidity_invariant(
        self,
        statement: str,
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
    ) -> Dict[str, Any]:
        """Translate a Solidity assignment into Z3/Lean invariants."""
        if translate_solidity_assignment_to_z3 is None:
            return {"success": False, "error": "Solidity invariant translation unavailable"}

        try:
            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
            )
            result: Dict[str, Any] = {"success": True, "translation": translation}
            if verify_translation and verify_solidity_invariant_translation is not None:
                result["verification"] = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=assume_non_negative_amount,
                )
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def solve_web3_exploit_witness(
        self,
        additional_constraints: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Solve symbolic exploit witness predicates for smart contracts."""
        if solve_smart_contract_exploit_witness is None:
            return {"success": False, "error": "Smart contract exploit witness solver unavailable"}
        try:
            witness = solve_smart_contract_exploit_witness(
                additional_constraints=additional_constraints,
                timeout=timeout,
            )
            return {"success": True, "result": witness}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_audit_exploit_verification(
        self,
        statement: str = "balance[msg.sender] -= amount;",
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
        additional_constraints: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Run a combined Web3 formal pass:
        translate invariants + optional proof check + exploit witness search.
        """
        translation = self.translate_solidity_invariant(
            statement=statement,
            non_negative_target=non_negative_target,
            max_withdraw_expr=max_withdraw_expr,
            verify_translation=verify_translation,
            assume_non_negative_amount=assume_non_negative_amount,
        )
        witness = self.solve_web3_exploit_witness(
            additional_constraints=additional_constraints,
            timeout=timeout,
        )

        verification = translation.get("verification")
        witness_result = witness.get("result", {})
        verified_exploit = bool(witness_result.get("satisfiable", False))
        if verify_translation and isinstance(verification, dict):
            verified_exploit = verified_exploit and bool(verification.get("proven", False))

        return {
            "success": bool(translation.get("success")) and bool(witness.get("success")),
            "translation": translation.get("translation"),
            "verification": verification,
            "exploit_witness": witness_result,
            "verified_exploit": verified_exploit,
        }


def create_api(config: Optional[Z3APIConfig] = None) -> Z3API:
    """Factory function to create API instance."""
    return Z3API(config)


__all__ = [
    "Z3APIConfig",
    "Z3API",
    "create_api",
    "app",
    "create_z3_knowledge_app",
]
