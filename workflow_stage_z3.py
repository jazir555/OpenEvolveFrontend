"""
Z3 Workflow Stage Integration

Adds Z3 as a native workflow stage type in the OpenEvolve workflow engine.
Enables Z3 solving as a workflow primitive alongside decomposition/recomposition.

Integrates with:
- workflow_engine.py
- workflow_stage_functions.py
- workflow_structures.py

Author: OpenEvolve
Created: 2026-02-02
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3SolverResult, translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation, solve_smart_contract_exploit_witness
    )
    Z3_AVAILABLE = True
    WEB3_FORMAL_AVAILABLE = (
        translate_solidity_assignment_to_z3 is not None
        and solve_smart_contract_exploit_witness is not None
    )
except ImportError:
    Z3_AVAILABLE = False
    WEB3_FORMAL_AVAILABLE = False
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    solve_smart_contract_exploit_witness = None

try:
    from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.cav_nlp_integration import Z3LeanAideBridge
    from openevolve.cav_nlp_integration.adapter import MathematicalTextParser
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    Z3LeanAideBridge = None
    MathematicalTextParser = None


def _get_web3_formal_capabilities() -> Dict[str, bool]:
    """Return capability flags for Web3 formal stage operations."""
    return {
        "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
        "invariant_translation_verification": verify_solidity_invariant_translation is not None,
        "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
        "composite_exploit_verification": (
            translate_solidity_assignment_to_z3 is not None
            and solve_smart_contract_exploit_witness is not None
        ),
    }


def get_web3_formal_status() -> Dict[str, Any]:
    """Get normalized Web3 formal status for workflow-stage integrations."""
    formal_capabilities = _get_web3_formal_capabilities()
    web3_formal_tools: List[str] = []
    if formal_capabilities["solidity_invariant_translation"]:
        web3_formal_tools.append("z3_translate_solidity_invariant")
    if formal_capabilities["symbolic_exploit_witness"]:
        web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
    if formal_capabilities["composite_exploit_verification"]:
        web3_formal_tools.append("z3_web3_audit_exploit_verification")
    web3_formal_tools = sorted(set(web3_formal_tools))
    inferred_formal_available = bool(web3_formal_tools) or any(
        bool(v) for v in formal_capabilities.values()
    )
    return {
        "available": inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE),
        "web3_formal_available": inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE),
        "web3_formal_tools": web3_formal_tools,
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": bool(
            formal_capabilities.get("composite_exploit_verification")
        ),
    }


class Z3StageType(Enum):
    """Types of Z3 workflow stages."""
    SOLVE = "z3_solve"
    OPTIMIZE = "z3_optimize"
    PROVE = "z3_prove"
    VERIFY = "z3_verify"
    TRANSLATE = "z3_translate"
    WEB3_INVARIANT_TRANSLATE = "z3_web3_invariant_translate"
    WEB3_EXPLOIT_WITNESS = "z3_web3_exploit_witness"
    WEB3_AUDIT_EXPLOIT_VERIFICATION = "z3_web3_audit_exploit_verification"


@dataclass
class Z3StageConfig:
    """Configuration for a Z3 workflow stage."""
    stage_type: Z3StageType
    timeout_seconds: float = 60.0
    proof_generation: bool = True
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    objective: Optional[Dict[str, Any]] = None
    smtlib_input: Optional[str] = None
    use_cav_nlp: bool = True  # Enable CAV-NLP enhancement
    statement: Optional[str] = None
    non_negative_target: bool = True
    max_withdraw_expr: Optional[str] = None
    verify_translation: bool = True
    assume_non_negative_amount: bool = True
    additional_constraints: List[str] = field(default_factory=list)


@dataclass
class Z3StageResult:
    """Result of executing a Z3 workflow stage."""
    success: bool
    stage_type: Z3StageType
    status: str
    model: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    execution_time_ms: float = 0.0
    z3_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Z3WorkflowStage:
    """
    Z3 solver as a workflow stage.
    
    Enables constraint solving, optimization, and theorem proving
    as first-class workflow operations.
    """
    
    def __init__(self, config: Z3StageConfig):
        self.config = config
        self.z3_config = None
        if Z3_AVAILABLE:
            self.z3_config = Z3Config(
                timeout=config.timeout_seconds,
                proof_generation=config.proof_generation
            )
        self.solver = Z3SolverEngine(self.z3_config) if Z3_AVAILABLE and self.z3_config else None
        self.prover = Z3TheoremProver(self.z3_config) if Z3_AVAILABLE and self.z3_config else None
        self.advanced = Z3AdvancedSolver(self.z3_config) if Z3_ADVANCED_AVAILABLE and self.z3_config else None
        
        # CAV-NLP integration
        self.use_cav_nlp = config.use_cav_nlp and CAV_NLP_AVAILABLE
        self.cav_nlp_bridge = None
        self.math_parser = None
        if self.use_cav_nlp:
            try:
                self.cav_nlp_bridge = Z3LeanAideBridge()
                self.math_parser = MathematicalTextParser()
                logger.info("CAV-NLP integration enabled for Z3 workflow stage")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}")
                self.use_cav_nlp = False
    
    def execute(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute the Z3 workflow stage."""
        start_time = time.time()

        required_web3_capability = {
            Z3StageType.WEB3_INVARIANT_TRANSLATE: "solidity_invariant_translation",
            Z3StageType.WEB3_EXPLOIT_WITNESS: "symbolic_exploit_witness",
            Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION: "composite_exploit_verification",
        }.get(self.config.stage_type)
        if required_web3_capability:
            formal_capabilities = _get_web3_formal_capabilities()
            if not formal_capabilities.get(required_web3_capability, False):
                return Z3StageResult(
                    success=False,
                    stage_type=self.config.stage_type,
                    status="error",
                    metadata={
                        "reason": "web3_formal_unavailable",
                        "required_capability": required_web3_capability,
                        "formal_capabilities": formal_capabilities,
                    },
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        if (
            self.config.stage_type not in {
                Z3StageType.WEB3_INVARIANT_TRANSLATE,
                Z3StageType.WEB3_EXPLOIT_WITNESS,
                Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION,
            }
            and not Z3_AVAILABLE
        ):
            return Z3StageResult(
                success=False,
                stage_type=self.config.stage_type,
                status="error",
                metadata={"reason": "z3_unavailable"},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            if self.config.stage_type == Z3StageType.SOLVE:
                return self._execute_solve(context)
            elif self.config.stage_type == Z3StageType.OPTIMIZE:
                return self._execute_optimize(context)
            elif self.config.stage_type == Z3StageType.PROVE:
                return self._execute_prove(context)
            elif self.config.stage_type == Z3StageType.VERIFY:
                return self._execute_verify(context)
            elif self.config.stage_type == Z3StageType.TRANSLATE:
                return self._execute_translate(context)
            elif self.config.stage_type == Z3StageType.WEB3_INVARIANT_TRANSLATE:
                return self._execute_web3_invariant_translate(context)
            elif self.config.stage_type == Z3StageType.WEB3_EXPLOIT_WITNESS:
                return self._execute_web3_exploit_witness(context)
            elif self.config.stage_type == Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION:
                return self._execute_web3_audit_exploit_verification(context)
            else:
                return Z3StageResult(
                    success=False,
                    stage_type=self.config.stage_type,
                    status="unknown_stage_type",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        except Exception as e:
            logger.error(f"Z3 stage execution failed: {e}")
            return Z3StageResult(
                success=False,
                stage_type=self.config.stage_type,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _execute_solve(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute constraint solving stage."""
        start_time = time.time()
        
        # Get variables and constraints from config or context
        variables = self._build_variables(
            self.config.variables or context.get("variables", [])
        )
        constraints = self._build_constraints(
            self.config.constraints or context.get("constraints", [])
        )
        
        # Solve
        if self.config.smtlib_input:
            result = self.solver.solve_smtlib(self.config.smtlib_input)
        else:
            result = self.solver.solve_constraints(variables, constraints)
        
        return Z3StageResult(
            success=True,
            stage_type=Z3StageType.SOLVE,
            status=result.status.value,
            model=result.model.assignments if result.model else None,
            execution_time_ms=(time.time() - start_time) * 1000,
            z3_output=result.smtlib_output
        )
    
    def _execute_optimize(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute optimization stage."""
        start_time = time.time()
        
        if not Z3_ADVANCED_AVAILABLE or not self.advanced:
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.OPTIMIZE,
                status="advanced_not_available",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        variables = self._build_variables(self.config.variables)
        constraints = self._build_constraints(self.config.constraints)
        
        objective = self.config.objective or context.get("objective", {})
        obj_expr = objective.get("expression", "x")
        obj_type = OptimizationObjective.MINIMIZE if objective.get("direction") == "minimize" else OptimizationObjective.MAXIMIZE
        
        result = self.advanced.optimize(variables, constraints, [(obj_expr, obj_type)])
        
        return Z3StageResult(
            success=result.success,
            stage_type=Z3StageType.OPTIMIZE,
            status="optimal" if result.success else "failed",
            model=result.optimal_model.assignments if result.optimal_model else None,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _execute_prove(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute theorem proving stage."""
        start_time = time.time()
        
        theorem = self.config.smtlib_input or context.get("theorem", "")
        assumptions = context.get("assumptions", [])
        
        result = self.prover.prove_theorem(theorem, assumptions)
        
        return Z3StageResult(
            success=True,
            stage_type=Z3StageType.PROVE,
            status="proven" if result.proven else "not_proven",
            proof=result.proof,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _execute_verify(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute verification stage - verifies a specification against a model."""
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.VERIFY,
                status="z3_unavailable",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Get specification to verify
            spec = self.config.smtlib_input or context.get("specification", "")
            assumptions = context.get("assumptions", [])
            
            # Verify using prover
            result = self.prover.prove_theorem(spec, assumptions)
            
            return Z3StageResult(
                success=True,
                stage_type=Z3StageType.VERIFY,
                status="verified" if result.proven else "not_verified",
                proof=result.proof,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Verify stage failed: {e}")
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.VERIFY,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _execute_translate(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute translation stage - translates between SMT-LIB and other formats."""
        start_time = time.time()
        
        try:
            direction = context.get("direction", "smt_to_lean")
            content = self.config.smtlib_input or context.get("content", "")
            
            # Try to use Z3-LeanAIDE bridge if available
            try:
                from z3_leanaide_bridge import get_z3_leanaide_bridge_sync
                bridge = get_z3_leanaide_bridge_sync()
                
                if direction == "smt_to_lean":
                    import asyncio
                    result = asyncio.run(bridge.translate_smt_to_lean(content))
                    translated = result.translation if result.success else ""
                else:
                    import asyncio
                    result = asyncio.run(bridge.translate_lean_to_smt(content))
                    translated = result.translation if result.success else ""
                
                return Z3StageResult(
                    success=result.success,
                    stage_type=Z3StageType.TRANSLATE,
                    status="translated" if result.success else "failed",
                    model={"translation": translated, "direction": direction},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            except ImportError:
                # Bridge not available - return placeholder
                return Z3StageResult(
                    success=False,
                    stage_type=Z3StageType.TRANSLATE,
                    status="bridge_unavailable",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        except Exception as e:
            logger.error(f"Translate stage failed: {e}")
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.TRANSLATE,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _execute_web3_invariant_translate(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute Web3 Solidity invariant translation stage."""
        start_time = time.time()
        if translate_solidity_assignment_to_z3 is None:
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.WEB3_INVARIANT_TRANSLATE,
                status="translator_unavailable",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            statement = (
                self.config.statement
                or context.get("statement")
                or context.get("solidity_statement")
                or self.config.smtlib_input
                or ""
            )
            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=bool(
                    context.get("non_negative_target", self.config.non_negative_target)
                ),
                max_withdraw_expr=context.get("max_withdraw_expr", self.config.max_withdraw_expr),
            )
            metadata: Dict[str, Any] = {"translation": translation}
            if (
                context.get("verify_translation", self.config.verify_translation)
                and verify_solidity_invariant_translation is not None
            ):
                metadata["verification"] = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=bool(
                        context.get(
                            "assume_non_negative_amount",
                            self.config.assume_non_negative_amount,
                        )
                    ),
                )
            return Z3StageResult(
                success=True,
                stage_type=Z3StageType.WEB3_INVARIANT_TRANSLATE,
                status="translated",
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Web3 invariant translation stage failed: %s", exc)
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.WEB3_INVARIANT_TRANSLATE,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(exc)},
            )

    def _execute_web3_exploit_witness(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute Web3 symbolic exploit witness stage."""
        start_time = time.time()
        if solve_smart_contract_exploit_witness is None:
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.WEB3_EXPLOIT_WITNESS,
                status="solver_unavailable",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            additional_constraints = context.get(
                "additional_constraints", self.config.additional_constraints
            )
            result = solve_smart_contract_exploit_witness(
                additional_constraints=additional_constraints,
                timeout=float(context.get("timeout_seconds", self.config.timeout_seconds)),
            )
            return Z3StageResult(
                success=bool(result.get("satisfiable")),
                stage_type=Z3StageType.WEB3_EXPLOIT_WITNESS,
                status=result.get("status", "unknown"),
                model=result.get("model"),
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"result": result},
            )
        except Exception as exc:
            logger.error("Web3 exploit witness stage failed: %s", exc)
            return Z3StageResult(
                success=False,
                stage_type=Z3StageType.WEB3_EXPLOIT_WITNESS,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(exc)},
            )

    def _execute_web3_audit_exploit_verification(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute combined Web3 translation + witness exploit verification stage."""
        start_time = time.time()

        translation_result = self._execute_web3_invariant_translate(context)
        witness_result = self._execute_web3_exploit_witness(context)

        translation_metadata = translation_result.metadata if isinstance(translation_result.metadata, dict) else {}
        witness_metadata = witness_result.metadata if isinstance(witness_result.metadata, dict) else {}
        witness_payload = witness_metadata.get("result", {})
        verification = translation_metadata.get("verification")

        verified_exploit = bool(witness_payload.get("satisfiable", False))
        if context.get("verify_translation", self.config.verify_translation) and isinstance(verification, dict):
            verified_exploit = verified_exploit and bool(verification.get("proven", False))

        if not translation_result.success:
            status = "translation_failed"
        elif not witness_result.success:
            status = "witness_unsatisfied"
        else:
            status = "verified_exploit" if verified_exploit else "unverified"

        return Z3StageResult(
            success=translation_result.success and witness_result.success and verified_exploit,
            stage_type=Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION,
            status=status,
            model=witness_result.model,
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "translation": translation_metadata.get("translation"),
                "verification": verification,
                "exploit_witness": witness_payload,
                "verified_exploit": verified_exploit,
            },
        )
    
    def _build_variables(self, var_specs: List[Dict[str, Any]]) -> List[Any]:
        """Build Z3 variables from specifications."""
        variables = []
        for spec in var_specs:
            var_type = Z3ConstraintType[spec.get("type", "INTEGER")]
            variables.append(Z3Variable(spec["name"], var_type))
        return variables
    
    def _build_constraints(self, constraint_exprs: List[str]) -> List[Any]:
        """Build Z3 constraints from expressions."""
        return [
            Z3Constraint(expr, Z3ConstraintType.BOOLEAN)
            for expr in constraint_exprs
        ]
    
    def _is_natural_language(self, text: str) -> bool:
        """Check if input appears to be natural language rather than formal code."""
        if not text or not isinstance(text, str):
            return False
        # Heuristics for natural language detection
        nl_indicators = [
            ' ',  # Contains spaces (sentences)
            '.',  # Sentence endings
            '?',  # Questions
            'the ', 'a ', 'an ',  # Articles
            'is ', 'are ', 'has ', 'have ',  # Common verbs
        ]
        formal_indicators = [
            '(declare-',  # SMT-LIB declarations
            '(assert',    # SMT-LIB assertions
            '(check-sat)', # SMT-LIB commands
            '(>', '(<', '(=',  # SMT-LIB operators
        ]
        
        # Check for formal indicators first (strong signal)
        for indicator in formal_indicators:
            if indicator in text:
                return False
        
        # Check for natural language indicators
        nl_score = sum(1 for ind in nl_indicators if ind in text.lower())
        return nl_score >= 2
    
    def execute_with_cav_nlp(self, stage_input: Any, context: Dict[str, Any] = None) -> Z3StageResult:
        """
        Execute workflow stage with CAV-NLP enhancement.
        
        Automatically formalizes natural language input before execution.
        
        Args:
            stage_input: Input to the stage (can be natural language or formal)
            context: Execution context
            
        Returns:
            Z3StageResult with execution results
        """
        context = context or {}
        
        # Check if CAV-NLP is enabled and input is natural language
        if self.use_cav_nlp and self._is_natural_language(str(stage_input)):
            try:
                logger.info("CAV-NLP: Formalizing natural language input")
                start_time = time.time()
                
                # Formalize using CAV-NLP
                formalized = self.cav_nlp_bridge.formalize_text(str(stage_input))
                
                if formalized and hasattr(formalized, 'code'):
                    # Update context with formalized code
                    context['formalized_input'] = formalized.code
                    context['original_input'] = stage_input
                    context['cav_nlp_time'] = time.time() - start_time
                    
                    logger.debug(f"CAV-NLP formalization successful in {context['cav_nlp_time']:.3f}s")
                    
                    # Execute with formalized input
                    return self.execute(context)
                else:
                    logger.warning("CAV-NLP formalization returned empty result, using original input")
            except Exception as e:
                logger.error(f"CAV-NLP formalization failed: {e}")
                # Fall through to execute with original input
        
        # Default: execute normally
        if isinstance(stage_input, dict):
            context.update(stage_input)
        return self.execute(context or {})


class Z3StageRegistry:
    """Registry for Z3 workflow stage types."""
    
    def __init__(self):
        self.stage_types = {}
        self._register_default_types()
    
    def _register_default_types(self):
        """Register default Z3 stage types."""
        for stage_type in Z3StageType:
            self.register(stage_type.value, Z3WorkflowStage)
    
    def register(self, type_name: str, stage_class: type):
        """Register a Z3 stage type."""
        self.stage_types[type_name] = stage_class
    
    def create_stage(self, config: Z3StageConfig) -> Optional[Z3WorkflowStage]:
        """Create a Z3 workflow stage."""
        stage_class = self.stage_types.get(config.stage_type.value)
        if stage_class:
            return stage_class(config)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get registry status including Web3 formal stage capabilities."""
        return {
            "registered_stage_types": sorted(self.stage_types.keys()),
            "stage_count": len(self.stage_types),
            **get_web3_formal_status(),
        }


# Global registry
_registry = None

def get_z3_stage_registry() -> Z3StageRegistry:
    """Get global Z3 stage registry."""
    global _registry
    if _registry is None:
        _registry = Z3StageRegistry()
    return _registry


if __name__ == "__main__":
    print("Z3 Workflow Stage Integration initialized")
