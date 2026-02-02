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
        Z3ConstraintType, Z3Config, Z3SolverResult
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False


class Z3StageType(Enum):
    """Types of Z3 workflow stages."""
    SOLVE = "z3_solve"
    OPTIMIZE = "z3_optimize"
    PROVE = "z3_prove"
    VERIFY = "z3_verify"
    TRANSLATE = "z3_translate"


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
    
    def execute(self, context: Dict[str, Any]) -> Z3StageResult:
        """Execute the Z3 workflow stage."""
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return Z3StageResult(
                success=False,
                stage_type=self.config.stage_type,
                status="error",
                execution_time_ms=(time.time() - start_time) * 1000
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
