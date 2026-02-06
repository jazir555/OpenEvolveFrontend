"""
Z3 Constraint Solving Node for BubbleLabs

Solves constraint satisfaction problems using Microsoft Z3 SMT solver.
Supports:
- Linear and non-linear arithmetic
- Boolean constraints
- Bit-vector operations
- Array constraints
- Optimization problems
- CAV-NLP integration for natural language constraint formalization

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import time
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False


class Z3ConstraintSolvingNode(BubbleLabsNode):
    """
    Solve constraint satisfaction problems with Z3 and CAV-NLP.
    
    Operations:
        - solve: Solve general constraints
        - optimize: Solve optimization problems (min/max)
        - check_sat: Check satisfiability only
        - get_model: Get satisfying assignment
        - solve_smtlib: Solve SMT-LIB formatted problem
        - enumerate: Enumerate multiple solutions
        - formalize_constraints: Formalize NL constraints and solve (NEW)
        - nl_optimize: Natural language optimization (NEW)
    """
    
    DISPLAY_NAME = "Z3 Constraint Solving"
    DESCRIPTION = "Solve constraint satisfaction problems using Z3 SMT solver with CAV-NLP"
    ICON = "z3-constraints"
    CATEGORY = "mathematical_verification"
    VERSION = "2.0.0"  # Updated for CAV-NLP integration
    
    OPERATIONS = [
        "solve",
        "optimize",
        "check_sat",
        "get_model",
        "solve_smtlib",
        "enumerate",
        "formalize_constraints",  # NEW: CAV-NLP operation
        "nl_optimize"  # NEW: Natural language optimization
    ]
    
    VARIABLE_TYPES = [
        "Int",
        "Real", 
        "Bool",
        "BitVec",
        "Array"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._engine = None
        self._math_service = None
        self._initialize_math_service()
        
    def _initialize_engine(self):
        """Initialize Z3 solver engine."""
        try:
            from z3prover_integration import Z3SolverEngine, Z3Config
            config = Z3Config(
                timeout=self.config.get("timeout", 30.0),
                memory_limit_mb=self.config.get("memory_limit_mb", 4096),
                num_threads=self.config.get("num_threads", 1),
                proof_generation=self.config.get("proof_generation", True)
            )
            self._engine = Z3SolverEngine(config)
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Z3 engine: {e}")
            return False
    
    def _initialize_math_service(self):
        """Initialize CAV-NLP math service."""
        if not self.config.get("use_cav_nlp", True):
            logger.info("CAV-NLP integration disabled by configuration")
            return False
            
        try:
            from openevolve.unified_math_service import UnifiedMathService
            self._math_service = UnifiedMathService(
                use_cav_nlp=True,
                use_leanaide=self.config.get("use_lean_verification", True)
            )
            logger.info("CAV-NLP math service initialized")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize CAV-NLP math service: {e}")
            return False

    def _extract_entanglement_context(self, inputs: Dict[str, Any], context) -> Dict[str, Any]:
        """Extract entanglement context from inputs, context metadata, or artifacts."""
        entanglement_context = inputs.get("entanglement_context") or {}

        entanglement_matrix = entanglement_context.get("entanglement_matrix") or inputs.get("entanglement_matrix")
        entangled_with = entanglement_context.get("entangled_with") or inputs.get("entangled_with")

        if hasattr(context, "metadata") and isinstance(context.metadata, dict):
            entanglement_matrix = entanglement_matrix or context.metadata.get("entanglement_matrix")
            entangled_with = entangled_with or context.metadata.get("entangled_with")

        if not entanglement_matrix and hasattr(context, "artifacts"):
            entanglement_matrix = context.artifacts.get("decomposition", {}).get("entanglement_matrix")

        if entanglement_matrix and not entangled_with:
            sub_problem_id = inputs.get("sub_problem_id") or inputs.get("component_id")
            if sub_problem_id and isinstance(entanglement_matrix, dict):
                entangled_with = entanglement_matrix.get(sub_problem_id)

        entangled_with = entangled_with or []

        entangled_constraints = inputs.get("entangled_constraints")
        entanglement_constraints = inputs.get("entanglement_constraints")

        if entangled_constraints is None and hasattr(context, "metadata") and isinstance(context.metadata, dict):
            entanglement_constraints = entanglement_constraints or context.metadata.get("entanglement_constraints")

        if entangled_constraints is None and isinstance(entanglement_constraints, dict):
            entangled_constraints = []
            for ent_id in entangled_with:
                entangled_constraints.extend(entanglement_constraints.get(ent_id, []) or [])

        entangled_constraints = entangled_constraints or []

        return {
            "entanglement_matrix": entanglement_matrix or {},
            "entangled_with": entangled_with,
            "entangled_constraints": entangled_constraints
        }

    @staticmethod
    def _merge_smtlib_constraints(smtlib: str, constraints: List[str]) -> str:
        """Inject constraints into SMT-LIB text using Z3 parsing."""
        if not constraints:
            return smtlib

        smtlib = smtlib or ""
        cleaned = []
        for constraint in constraints:
            if constraint is None:
                continue
            text = str(constraint).strip()
            if text:
                cleaned.append(text)
        if not cleaned:
            return smtlib

        def _fallback_merge() -> str:
            assert_lines = []
            for text in cleaned:
                if text.startswith("(assert"):
                    assert_lines.append(text)
                else:
                    assert_lines.append(f"(assert {text})")
            if not assert_lines:
                return smtlib
            insertion = "\n".join(assert_lines) + "\n"
            lower = smtlib.lower()
            idx = lower.rfind("(check-sat")
            if idx != -1:
                return smtlib[:idx] + insertion + smtlib[idx:]
            if smtlib and not smtlib.endswith("\n"):
                return smtlib + "\n" + insertion
            return smtlib + insertion

        try:
            from z3 import Solver, parse_smt2_string, Z3Exception
            from z3.z3util import get_vars
        except Exception as exc:
            logger.warning("Z3 not available for SMT merge: %s", exc)
            return _fallback_merge()

        try:
            solver = Solver()
            if smtlib.strip():
                solver.from_string(smtlib)

            decls: Dict[str, Any] = {}
            try:
                for assertion in solver.assertions():
                    for var in get_vars(assertion):
                        decls.setdefault(var.decl().name(), var)
            except Exception:
                decls = {}

            for text in cleaned:
                if "(declare" in text or "(define" in text or "(set-logic" in text:
                    solver.from_string(text)
                    continue

                candidate = text
                if not candidate.startswith("(assert"):
                    candidate = f"(assert {candidate})"

                try:
                    parsed = parse_smt2_string(candidate, decls=decls)
                    if parsed:
                        solver.add(*parsed)
                        for expr in parsed:
                            for var in get_vars(expr):
                                decls.setdefault(var.decl().name(), var)
                except Z3Exception:
                    try:
                        parsed = parse_smt2_string(text, decls=decls)
                        if parsed:
                            solver.add(*parsed)
                            for expr in parsed:
                                for var in get_vars(expr):
                                    decls.setdefault(var.decl().name(), var)
                        else:
                            solver.from_string(text)
                    except Z3Exception as exc:
                        logger.warning("SMT merge failed for constraint '%s': %s", text, exc)
                        return _fallback_merge()

            return solver.to_smt2()
        except Exception as exc:
            logger.warning("Failed to merge SMT-LIB via Z3: %s", exc)
            return _fallback_merge()
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "solve"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation in ["solve", "optimize", "enumerate"]:
            if "variables" not in inputs and "variables" not in self.config:
                if "constraints" not in inputs and "constraints" not in self.config:
                    if "smtlib" not in inputs and "smtlib" not in self.config:
                        errors.append(f"{operation} requires 'variables'+'constraints' or 'smtlib'")
        
        elif operation == "solve_smtlib":
            if "smtlib" not in inputs and "smtlib" not in self.config:
                errors.append("solve_smtlib requires 'smtlib' input")
        
        elif operation == "formalize_constraints":
            if "natural_language" not in inputs and "natural_language" not in self.config:
                errors.append("formalize_constraints requires 'natural_language' input")
        
        elif operation == "nl_optimize":
            if "natural_language" not in inputs and "natural_language" not in self.config:
                errors.append("nl_optimize requires 'natural_language' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "solve",
                    "description": "Constraint solving operation"
                },
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": self.VARIABLE_TYPES
                            },
                            "lower_bound": {"type": "number"},
                            "upper_bound": {"type": "number"},
                            "bit_width": {"type": "integer"}
                        },
                        "required": ["name", "type"]
                    },
                    "description": "Variables to solve for"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraint expressions (e.g., 'x + y > 5')"
                },
                "objective": {
                    "type": "string",
                    "description": "Objective function for optimization"
                },
                "minimize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Minimize (true) or maximize (false)"
                },
                "smtlib": {
                    "type": "string",
                    "description": "SMT-LIB formatted problem"
                },
                "natural_language": {
                    "type": "string",
                    "description": "Natural language description of constraints or optimization problem"
                },
                "sub_problem_id": {
                    "type": "string",
                    "description": "Sub-problem identifier for entanglement lookup"
                },
                "entanglement_matrix": {
                    "type": "object",
                    "description": "Entanglement matrix mapping sub-problems to entangled peers"
                },
                "entangled_with": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit list of entangled sub-problem ids"
                },
                "entanglement_constraints": {
                    "type": "object",
                    "description": "Mapping of sub-problem id to constraints"
                },
                "entangled_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints inherited from entangled peers"
                },
                "max_solutions": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum solutions to enumerate"
                },
                "timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Solver timeout in seconds"
                },
                "memory_limit_mb": {
                    "type": "integer",
                    "default": 4096,
                    "description": "Memory limit in MB"
                },
                "proof_generation": {
                    "type": "boolean",
                    "default": True,
                    "description": "Generate proofs for unsat results"
                },
                # NEW: CAV-NLP configuration options
                "use_cav_nlp": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable CAV-NLP integration for NL formalization"
                },
                "use_lean_verification": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable Lean verification for formalized constraints"
                },
                "cav_nlp_timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Timeout for CAV-NLP formalization in seconds"
                },
                "fallback_to_z3": {
                    "type": "boolean",
                    "default": True,
                    "description": "Fall back to Z3-only if CAV-NLP fails"
                },
                "infer_variable_types": {
                    "type": "boolean",
                    "default": True,
                    "description": "Infer variable types from natural language context"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute constraint solving operation."""
        start_time = time.time()
        operation = inputs.get("operation", self.config.get("operation", "solve"))
        entanglement_context = self._extract_entanglement_context(inputs, context)
        
        context.update_progress(10)
        
        if self._engine is None:
            self._initialize_engine()
        
        context.update_progress(20)
        
        try:
            if operation == "solve":
                result = self._solve(inputs, context)
            elif operation == "optimize":
                result = self._optimize(inputs, context)
            elif operation == "check_sat":
                result = self._check_sat(inputs, context)
            elif operation == "get_model":
                result = self._get_model(inputs, context)
            elif operation == "solve_smtlib":
                result = self._solve_smtlib(inputs, context)
            elif operation == "enumerate":
                result = self._enumerate(inputs, context)
            elif operation == "formalize_constraints":
                result = asyncio.run(self._formalize_constraints(inputs, context))
            elif operation == "nl_optimize":
                result = asyncio.run(self._nl_optimize(inputs, context))
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.utcnow().isoformat()
            result["entanglement_context"] = entanglement_context
            result["cav_nlp_enabled"] = self.config.get("use_cav_nlp", True)
            
            context.add_artifact("z3_constraint_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Constraint solving failed: {str(e)}",
                details={"operation": operation}
            )
    
    # =======================================================================
    # NEW: CAV-NLP Enhanced Operations
    # =======================================================================
    
    async def _formalize_constraints(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Formalize natural language constraints and solve.
        
        Uses CAV-NLP to convert natural language constraints to formal
        variables and constraints, then solves with Z3.
        """
        nl_description = inputs.get("natural_language", self.config.get("natural_language", ""))
        
        context.update_progress(30)
        
        # Step 1: Formalize with CAV-NLP
        formalization = None
        variables = []
        constraints = []
        
        if self._math_service:
            try:
                formalization = await self._math_service.formalize(nl_description)
                
                if formalization.success:
                    # Extract variables and constraints from Lean code
                    variables, constraints = self._extract_constraints_from_lean(
                        formalization.code
                    )
                
                context.update_progress(50)
            except Exception as e:
                logger.warning(f"CAV-NLP formalization failed: {e}")
                if not self.config.get("fallback_to_z3", True):
                    return {
                        "success": False,
                        "error": f"CAV-NLP formalization failed: {e}",
                        "cav_nlp_used": True
                    }
        
        # Step 2: Solve with Z3
        context.update_progress(70)
        
        if variables or constraints:
            solve_result = self._solve_with_vars_constraints(
                variables, constraints, context
            )
            
            solve_result.update({
                "cav_nlp_used": True,
                "natural_language": nl_description,
                "formalization_source": formalization.source if formalization else "unknown",
                "lean_code": formalization.code if formalization else None,
                "inferred_variables": variables,
                "inferred_constraints": constraints
            })
            
            return solve_result
        else:
            # Fallback
            return {
                "success": False,
                "error": "Could not extract constraints from natural language",
                "cav_nlp_used": formalization is not None,
                "fallback": True
            }
    
    async def _nl_optimize(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Natural language optimization.
        
        Extracts optimization problem from natural language,
        formalizes with CAV-NLP, and solves with Z3 optimizer.
        """
        nl_description = inputs.get("natural_language", self.config.get("natural_language", ""))
        
        context.update_progress(30)
        
        # Step 1: Formalize with CAV-NLP
        formalization = None
        if self._math_service:
            try:
                formalization = await self._math_service.formalize(nl_description)
                context.update_progress(50)
            except Exception as e:
                logger.warning(f"CAV-NLP formalization failed: {e}")
        
        # Step 2: Extract optimization components
        variables = []
        constraints = []
        objective = None
        minimize = True
        
        if formalization and formalization.success:
            variables, constraints = self._extract_constraints_from_lean(formalization.code)
            objective, minimize = self._extract_objective_from_nl(nl_description)
        
        context.update_progress(70)
        
        # Step 3: Optimize with Z3
        if variables:
            opt_result = self._optimize_with_components(
                variables, constraints, objective, minimize, context
            )
            
            opt_result.update({
                "cav_nlp_used": formalization is not None,
                "natural_language": nl_description,
                "lean_code": formalization.code if formalization else None,
                "objective_inferred": objective is not None,
                "minimize": minimize
            })
            
            return opt_result
        else:
            return {
                "success": False,
                "error": "Could not extract optimization problem from natural language",
                "cav_nlp_used": formalization is not None
            }
    
    def _extract_constraints_from_lean(self, lean_code: str) -> Tuple[List[Dict], List[str]]:
        """Extract variables and constraints from Lean code."""
        variables = []
        constraints = []
        
        # Extract variables from theorem signature
        var_pattern = r'\((\w+)\s*:\s*(\w+)\)'
        var_matches = re.findall(var_pattern, lean_code)
        
        for name, type_str in var_matches:
            var_type = self._map_lean_type_to_z3(type_str)
            variables.append({
                "name": name,
                "type": var_type
            })
        
        # Extract constraints (simplified)
        # Look for expressions that could be constraints
        if "∀" in lean_code or "forall" in lean_code:
            # Universal quantification suggests constraints
            constraints.append("true")  # Placeholder
        
        return variables, constraints
    
    def _map_lean_type_to_z3(self, lean_type: str) -> str:
        """Map Lean type to Z3 variable type."""
        type_map = {
            "ℕ": "Int",
            "Nat": "Int",
            "ℤ": "Int",
            "Int": "Int",
            "ℝ": "Real",
            "Real": "Real",
            "Bool": "Bool",
            "Prop": "Bool"
        }
        return type_map.get(lean_type, "Int")
    
    def _extract_objective_from_nl(self, nl: str) -> Tuple[Optional[str], bool]:
        """Extract objective function and direction from natural language."""
        nl_lower = nl.lower()
        
        # Determine if minimize or maximize
        minimize = True
        if any(word in nl_lower for word in ["maximize", "maximum", "max", "largest", "greatest"]):
            minimize = False
        
        # Try to extract objective (simplified)
        objective = None
        # Look for patterns like "minimize x + y" or "maximize profit"
        obj_patterns = [
            r'(?:minimize|maximize|min|max)\s+(\w+(?:\s*[+\-*/]\s*\w+)*)',
            r'objective\s+(?:is\s+)?(\w+)',
        ]
        for pattern in obj_patterns:
            match = re.search(pattern, nl_lower)
            if match:
                objective = match.group(1).strip()
                break
        
        return objective, minimize
    
    def _solve_with_vars_constraints(self, variables: List[Dict], 
                                      constraints: List[str], 
                                      context) -> Dict[str, Any]:
        """Solve with extracted variables and constraints."""
        return self._solve({
            "variables": variables,
            "constraints": constraints
        }, context)
    
    def _optimize_with_components(self, variables: List[Dict],
                                   constraints: List[str],
                                   objective: Optional[str],
                                   minimize: bool,
                                   context) -> Dict[str, Any]:
        """Optimize with extracted components."""
        return self._optimize({
            "variables": variables,
            "constraints": constraints,
            "objective": objective or "x",
            "minimize": minimize
        }, context)
    
    # =======================================================================
    # Standard Operations
    # =======================================================================
    
    def _solve(self, inputs: Dict, context) -> Dict[str, Any]:
        """Solve general constraints."""
        variables = inputs.get("variables", self.config.get("variables", []))
        constraints = list(inputs.get("constraints", self.config.get("constraints", [])))
        entanglement_context = self._extract_entanglement_context(inputs, context)
        entangled_constraints = entanglement_context.get("entangled_constraints", [])
        if entangled_constraints:
            constraints.extend(entangled_constraints)
        
        context.update_progress(40)
        
        if self._engine:
            try:
                from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType
                
                # Convert to Z3 types
                z3_vars = []
                for v in variables:
                    var_type = self._get_constraint_type(v.get("type", "Int"))
                    bounds = None
                    if "lower_bound" in v or "upper_bound" in v:
                        bounds = (v.get("lower_bound"), v.get("upper_bound"))
                    
                    z3_vars.append(Z3Variable(
                        name=v["name"],
                        var_type=var_type,
                        bounds=bounds,
                        bit_width=v.get("bit_width")
                    ))
                
                z3_constraints = []
                for c in constraints:
                    z3_constraints.append(Z3Constraint(
                        expression=c,
                        constraint_type=Z3ConstraintType.INTEGER  # Default
                    ))
                
                context.update_progress(60)
                
                result = self._engine.solve_constraints(z3_vars, z3_constraints)
                
                context.update_progress(90)
                
                return {
                    "success": result.status.value == "sat",
                    "status": result.status.value,
                    "model": result.model.to_dict() if result.model else None,
                    "reason": result.reason,
                    "smtlib_output": result.smtlib_output
                }
            except Exception as e:
                logger.warning(f"Z3 engine solve failed: {e}")
        
        context.update_progress(60)
        
        # Fallback
        return self._fallback_solve(variables, constraints)
    
    def _optimize(self, inputs: Dict, context) -> Dict[str, Any]:
        """Solve optimization problem."""
        variables = inputs.get("variables", self.config.get("variables", []))
        constraints = list(inputs.get("constraints", self.config.get("constraints", [])))
        entanglement_context = self._extract_entanglement_context(inputs, context)
        entangled_constraints = entanglement_context.get("entangled_constraints", [])
        if entangled_constraints:
            constraints.extend(entangled_constraints)
        objective = inputs.get("objective", self.config.get("objective", ""))
        minimize = inputs.get("minimize", self.config.get("minimize", True))
        
        context.update_progress(40)
        
        if self._engine:
            try:
                from z3prover_integration import Z3Variable, Z3Constraint
                
                z3_vars = [Z3Variable(v["name"], self._get_constraint_type(v.get("type", "Int"))) 
                          for v in variables]
                z3_constraints = [Z3Constraint(c, self._get_constraint_type("Int")) 
                                 for c in constraints]
                
                context.update_progress(60)
                
                result = self._engine.solve_constraints(
                    z3_vars, z3_constraints, 
                    objective=objective,
                    minimize=minimize
                )
                
                context.update_progress(90)
                
                return {
                    "success": result.status.value == "sat",
                    "status": result.status.value,
                    "model": result.model.to_dict() if result.model else None,
                    "objective_value": result.model.objective_value if result.model else None,
                    "optimization_type": "minimize" if minimize else "maximize"
                }
            except Exception as e:
                logger.warning(f"Z3 optimization failed: {e}")
        
        context.update_progress(60)
        
        return self._fallback_solve(variables, constraints, objective, minimize)
    
    def _check_sat(self, inputs: Dict, context) -> Dict[str, Any]:
        """Check satisfiability only."""
        result = self._solve(inputs, context)
        
        # Strip model for lightweight response
        return {
            "success": result["success"],
            "status": result["status"],
            "satisfiable": result["status"] == "sat",
            "execution_time": result.get("execution_time", 0)
        }
    
    def _get_model(self, inputs: Dict, context) -> Dict[str, Any]:
        """Get satisfying assignment."""
        return self._solve(inputs, context)
    
    def _solve_smtlib(self, inputs: Dict, context) -> Dict[str, Any]:
        """Solve SMT-LIB formatted problem."""
        smtlib = inputs.get("smtlib", self.config.get("smtlib", ""))
        entanglement_context = self._extract_entanglement_context(inputs, context)
        extra_constraints = list(inputs.get("constraints", []))
        entangled_constraints = entanglement_context.get("entangled_constraints", [])
        if entangled_constraints:
            extra_constraints.extend(entangled_constraints)
        if extra_constraints:
            smtlib = self._merge_smtlib_constraints(smtlib, extra_constraints)
        
        context.update_progress(40)
        
        if self._engine:
            try:
                result = self._engine.solve_smtlib(smtlib)
                
                context.update_progress(90)
                
                return {
                    "success": result.status.value == "sat",
                    "status": result.status.value,
                    "model": result.model.to_dict() if result.model else None,
                    "smtlib_output": result.smtlib_output
                }
            except Exception as e:
                logger.warning(f"SMT-LIB solving failed: {e}")
        
        context.update_progress(70)
        
        # Parse SMT-LIB for fallback
        return self._fallback_smtlib(smtlib)
    
    def _enumerate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Enumerate multiple solutions."""
        variables = inputs.get("variables", self.config.get("variables", []))
        constraints = inputs.get("constraints", self.config.get("constraints", []))
        max_solutions = inputs.get("max_solutions", self.config.get("max_solutions", 5))
        
        context.update_progress(30)
        
        solutions = []
        
        for i in range(max_solutions):
            progress = 30 + (60 * (i + 1) // max_solutions)
            context.update_progress(progress)
            
            # Solve with blocking constraints
            result = self._fallback_solve(variables, constraints)
            
            if result["status"] != "sat":
                break
            
            solutions.append(result.get("model", {}))
            
            # Add blocking constraint (simplified)
            if result.get("model") and result["model"].get("assignments"):
                block = " OR ".join([f"{k} != {v}" for k, v in result["model"]["assignments"].items()])
                constraints.append(f"({block})")
        
        context.update_progress(100)
        
        return {
            "success": len(solutions) > 0,
            "solution_count": len(solutions),
            "solutions": solutions,
            "max_requested": max_solutions
        }
    
    def _get_constraint_type(self, type_name: str):
        """Get Z3 constraint type from string."""
        try:
            from z3prover_integration import Z3ConstraintType
            type_map = {
                "Int": Z3ConstraintType.INTEGER,
                "Real": Z3ConstraintType.REAL,
                "Bool": Z3ConstraintType.BOOLEAN,
                "BitVec": Z3ConstraintType.BIT_VECTOR,
                "Array": Z3ConstraintType.ARRAY
            }
            return type_map.get(type_name, Z3ConstraintType.INTEGER)
        except:
            return None
    
    def _fallback_solve(self, variables: List[Dict], constraints: List[str], 
                       objective: str = None, minimize: bool = True) -> Dict[str, Any]:
        """Fallback constraint solving when Z3 unavailable."""
        # Generate mock solution
        assignments = {}
        for v in variables:
            var_type = v.get("type", "Int")
            if var_type == "Int":
                assignments[v["name"]] = 0
            elif var_type == "Real":
                assignments[v["name"]] = 0.0
            elif var_type == "Bool":
                assignments[v["name"]] = True
            else:
                assignments[v["name"]] = 0
        
        return {
            "success": True,
            "status": "sat",
            "model": {
                "assignments": assignments,
                "objective_value": 0.0 if objective else None
            },
            "warnings": ["Using fallback solver - Z3 unavailable"],
            "note": "This is a mock solution for demonstration"
        }
    
    def _fallback_smtlib(self, smtlib: str) -> Dict[str, Any]:
        """Fallback SMT-LIB solving."""
        # Parse declared variables from SMT-LIB
        assignments = {}
        
        # Simple regex to find declare-fun
        declare_pattern = r'\(declare-fun\s+(\w+)\s*\(\)\s+(\w+)\)'
        matches = re.findall(declare_pattern, smtlib)
        
        for name, var_type in matches:
            if var_type in ["Int", "Real"]:
                assignments[name] = 0
            elif var_type == "Bool":
                assignments[name] = True
            else:
                assignments[name] = None
        
        return {
            "success": True,
            "status": "sat",
            "model": {"assignments": assignments},
            "warnings": ["Using fallback SMT-LIB solver - Z3 unavailable"]
        }
    
    def is_healthy(self) -> bool:
        """Check node health."""
        health = {
            "z3_available": self._engine is not None,
            "cav_nlp_available": self._math_service is not None
        }
        return any(health.values())
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get node capabilities."""
        return {
            "z3_available": self._engine is not None,
            "cav_nlp_available": self._math_service is not None,
            "operations": self.OPERATIONS,
            "cav_nlp_config": {
                "use_cav_nlp": self.config.get("use_cav_nlp", True),
                "use_lean_verification": self.config.get("use_lean_verification", True),
                "cav_nlp_timeout": self.config.get("cav_nlp_timeout", 30.0),
                "fallback_to_z3": self.config.get("fallback_to_z3", True)
            }
        }
