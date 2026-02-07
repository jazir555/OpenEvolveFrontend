"""
Z3 MCP (Model Context Protocol) Tools

Provides MCP-compatible tools for Z3 integration, enabling external AI systems
to use Z3 capabilities through a standardized protocol.

Tools Provided:
- z3_solve_constraints: Solve constraint satisfaction problems
- z3_optimize: Solve optimization problems
- z3_prove_theorem: Prove theorems
- z3_translate_smt_to_lean: Translate SMT-LIB to Lean 4
- z3_solve_incremental: Incremental constraint solving
- z3_extract_proof: Extract proofs from Z3
- z3_analyze_problem: Analyze problem characteristics
- z3_solve_portfolio: Portfolio solving with multiple strategies
- z3_formalize_constraint: Formalize natural language to Z3 using CAV-NLP
- z3_verify_hybrid: Verify using hybrid Z3 + Lean approach
- z3_canonicalize_constraint: Return canonical form using CAV-NLP
- z3_translate_solidity_invariant: Translate Solidity state updates to Z3/Lean invariants
- z3_solve_smart_contract_exploit_witness: Solve symbolic exploit witness predicates
- z3_web3_audit_exploit_verification: Combined invariant verification + exploit witness workflow

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from web3_formal_evidence import (
    build_web3_formal_evidence,
    verify_web3_lean_proof_async,
)

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, get_z3_solver_engine, get_z3_theorem_prover,
        is_z3_available, translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation, solve_smart_contract_exploit_witness
    )
    Z3_AVAILABLE = True
    WEB3_FORMAL_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    WEB3_FORMAL_AVAILABLE = False
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    solve_smart_contract_exploit_witness = None
    logger.warning("Z3 integration not available")

try:
    from z3prover_advanced import (
        Z3AdvancedSolver, OptimizationObjective, ProofFormat,
        ArrayConstraint, BitVectorConstraint,
        get_z3_advanced_solver
    )
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, TranslationDirection,
        get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_AVAILABLE = False
    logger.warning("Z3-LeanAIDE bridge not available")

# Import CAV-NLP / Unified Math Service
try:
    from openevolve.unified_math_service import (
        UnifiedMathService,
        create_unified_math_service,
        FormalizationResult,
    )
    from openevolve.cav_nlp_integration import (
        Z3LeanAideBridge as CAVNLPBridge,
        create_z3_lean_bridge,
        CanonicalizationResult,
    )
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False
    logger.warning("Unified Math Service (CAV-NLP) not available")

# Import CAV-NLP canonicalizer
try:
    from openevolve.cav_nlp_integration.z3_canonicalizer import Z3Canonicalizer
    CANONICALIZER_AVAILABLE = True
except ImportError:
    CANONICALIZER_AVAILABLE = False


# =============================================================================
# MCP Tool Decorator and Registry
# =============================================================================

class MCPTool:
    """Decorator for MCP tools."""
    
    _registry: Dict[str, Callable] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def __call__(self, func: Callable) -> Callable:
        MCPTool._registry[self.name] = func
        MCPTool._metadata[self.name] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
        
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"MCP tool {self.name} failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return wrapper
    
    @classmethod
    def get_registry(cls) -> Dict[str, Callable]:
        """Get all registered tools."""
        return cls._registry.copy()
    
    @classmethod
    def get_metadata(cls) -> List[Dict[str, Any]]:
        """Get metadata for all tools."""
        return list(cls._metadata.values())
    
    @classmethod
    def execute(cls, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = cls._registry.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}"
            }
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(tool(**params))


# =============================================================================
# MCP Tool Definitions
# =============================================================================

@MCPTool(
    name="z3_solve_constraints",
    description="Solve a constraint satisfaction problem using Z3",
    parameters={
        "variables": {
            "type": "array",
            "description": "List of variable definitions",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["BOOLEAN", "INTEGER", "REAL", "BIT_VECTOR", "STRING", "FLOATING_POINT"]},
                    "bit_width": {"type": "integer", "optional": True}
                }
            }
        },
        "constraints": {
            "type": "array",
            "description": "List of SMT-LIB constraint expressions",
            "items": {"type": "string"}
        },
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds",
            "optional": True,
            "default": 30
        }
    }
)
async def z3_solve_constraints(
    variables: List[Dict[str, Any]],
    constraints: List[str],
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Solve constraint satisfaction problem.
    
    Example:
    {
        "variables": [
            {"name": "x", "type": "INTEGER"},
            {"name": "y", "type": "INTEGER"}
        ],
        "constraints": [
            "x > 0",
            "x < 10",
            "y == x + 5"
        ]
    }
    """
    if not Z3_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 not available"
        }
    
    try:
        solver = get_z3_solver_engine(Z3Config(timeout=timeout))
        
        # Parse variables
        z3_vars = []
        for v in variables:
            var_type = Z3ConstraintType[v.get('type', 'INTEGER').upper()]
            z3_var = Z3Variable(
                name=v['name'],
                var_type=var_type,
                bit_width=v.get('bit_width')
            )
            z3_vars.append(z3_var)
        
        # Parse constraints
        z3_constraints = [
            Z3Constraint(expr, Z3ConstraintType.INTEGER)
            for expr in constraints
        ]
        
        # Solve
        result = solver.solve_constraints(z3_vars, z3_constraints)
        
        return {
            "success": True,
            "status": result.status.value,
            "satisfiable": result.is_sat(),
            "model": result.model.assignments if result.model else None,
            "execution_time": result.execution_time,
            "errors": result.errors
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_optimize",
    description="Solve an optimization problem using Z3",
    parameters={
        "variables": {
            "type": "array",
            "description": "List of variable definitions"
        },
        "constraints": {
            "type": "array",
            "description": "List of constraint expressions"
        },
        "objective": {
            "type": "object",
            "description": "Objective function",
            "properties": {
                "expression": {"type": "string"},
                "direction": {"type": "string", "enum": ["minimize", "maximize"]}
            }
        },
        "timeout": {
            "type": "number",
            "optional": True,
            "default": 30
        }
    }
)
async def z3_optimize(
    variables: List[Dict[str, Any]],
    constraints: List[str],
    objective: Dict[str, str],
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Solve optimization problem.
    
    Example:
    {
        "variables": [{"name": "x", "type": "INTEGER"}],
        "constraints": ["x >= 0", "x <= 100"],
        "objective": {"expression": "x", "direction": "maximize"}
    }
    """
    if not Z3_ADVANCED_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 advanced features not available"
        }
    
    try:
        solver = get_z3_advanced_solver(Z3Config(timeout=timeout))
        
        # Parse variables
        z3_vars = []
        for v in variables:
            var_type = Z3ConstraintType[v.get('type', 'INTEGER').upper()]
            z3_vars.append(Z3Variable(
                name=v['name'], 
                var_type=var_type,
                bit_width=v.get('bit_width')
            ))
        
        # Parse constraints
        z3_constraints = [
            Z3Constraint(expr, Z3ConstraintType.INTEGER)
            for expr in constraints
        ]
        
        # Parse objective
        obj_expr = objective['expression']
        obj_direction = OptimizationObjective.MINIMIZE if objective['direction'] == 'minimize' else OptimizationObjective.MAXIMIZE
        
        # Optimize
        result = solver.optimize(z3_vars, z3_constraints, [(obj_expr, obj_direction)])
        
        return {
            "success": result.success,
            "optimal_value": result.optimal_value,
            "model": result.optimal_model.assignments if result.optimal_model else None,
            "execution_time": result.execution_time
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_prove_theorem",
    description="Prove a theorem using Z3",
    parameters={
        "theorem": {
            "type": "string",
            "description": "Theorem statement in SMT-LIB or natural language"
        },
        "assumptions": {
            "type": "array",
            "description": "List of assumptions",
            "items": {"type": "string"},
            "optional": True
        },
        "extract_proof": {
            "type": "boolean",
            "description": "Whether to extract detailed proof",
            "optional": True,
            "default": False
        }
    }
)
async def z3_prove_theorem(
    theorem: str,
    assumptions: Optional[List[str]] = None,
    extract_proof: bool = False
) -> Dict[str, Any]:
    """
    Prove theorem using Z3.
    
    Example:
    {
        "theorem": "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(assert (not (> (+ x 1) 0)))(check-sat)"
    }
    """
    if not Z3_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 not available"
        }
    
    try:
        prover = get_z3_theorem_prover()
        
        result = prover.prove_theorem(theorem, assumptions or [])
        
        response = {
            "success": True,
            "proven": result.proven,
            "execution_time": result.execution_time,
            "tactic_used": result.tactic_used,
            "errors": result.errors
        }
        
        if result.counterexample:
            response["counterexample"] = result.counterexample
        
        if extract_proof and result.proof:
            response["proof"] = result.proof[:1000]  # Truncate long proofs
        
        return response
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_translate_smt_to_lean",
    description="Translate SMT-LIB to Lean 4 code",
    parameters={
        "smtlib": {
            "type": "string",
            "description": "SMT-LIB content to translate"
        }
    }
)
async def z3_translate_smt_to_lean(smtlib: str) -> Dict[str, Any]:
    """
    Translate SMT-LIB to Lean 4.
    
    Example:
    {
        "smtlib": "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)"
    }
    """
    if not Z3_LEANAIDE_AVAILABLE:
        return {
            "success": False,
            "error": "Z3-LeanAIDE bridge not available"
        }
    
    try:
        bridge = get_z3_leanaide_bridge_sync()
        
        result = await bridge.translate_smt_to_lean(smtlib)
        
        return {
            "success": result.success,
            "translation": result.translation,
            "execution_time": result.execution_time,
            "errors": result.errors,
            "metadata": result.metadata
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_solve_incremental",
    description="Solve constraints incrementally with push/pop",
    parameters={
        "state_id": {
            "type": "string",
            "description": "Incremental state ID (omit to create new)",
            "optional": True
        },
        "operation": {
            "type": "string",
            "description": "Operation to perform",
            "enum": ["create", "push", "pop", "add", "check"]
        },
        "variables": {
            "type": "array",
            "description": "Variables for create operation",
            "optional": True
        },
        "constraints": {
            "type": "array",
            "description": "Constraints for create/add operations",
            "optional": True
        },
        "constraint": {
            "type": "string",
            "description": "Single constraint for add operation",
            "optional": True
        }
    }
)
async def z3_solve_incremental(
    operation: str,
    state_id: Optional[str] = None,
    variables: Optional[List[Dict]] = None,
    constraints: Optional[List[str]] = None,
    constraint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Incremental constraint solving.
    
    Example workflow:
    1. {"operation": "create", "variables": [...], "constraints": [...]}
    2. {"operation": "push", "state_id": "..."}
    3. {"operation": "add", "state_id": "...", "constraint": "x < 10"}
    4. {"operation": "check", "state_id": "..."}
    5. {"operation": "pop", "state_id": "..."}
    """
    if not Z3_ADVANCED_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 advanced features not available"
        }
    
    try:
        solver = get_z3_advanced_solver()
        
        if operation == "create":
            z3_vars = [
                Z3Variable(
                    name=v['name'], 
                    var_type=Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                    bit_width=v.get('bit_width')
                )
                for v in (variables or [])
            ]
            z3_constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in (constraints or [])
            ]
            
            new_state_id = solver.create_incremental_state(z3_vars, z3_constraints, state_id)
            
            return {
                "success": True,
                "state_id": new_state_id,
                "message": "Incremental state created"
            }
        
        elif operation == "push":
            success = solver.push_scope(state_id)
            return {
                "success": success,
                "message": "Scope pushed" if success else "Failed to push scope"
            }
        
        elif operation == "pop":
            success = solver.pop_scope(state_id)
            return {
                "success": success,
                "message": "Scope popped" if success else "Failed to pop scope"
            }
        
        elif operation == "add":
            z3_constraint = Z3Constraint(constraint, Z3ConstraintType.INTEGER)
            success = solver.add_constraint_incremental(state_id, z3_constraint)
            return {
                "success": success,
                "message": "Constraint added" if success else "Failed to add constraint"
            }
        
        elif operation == "check":
            result = solver.check_incremental(state_id)
            return {
                "success": True,
                "status": result.status.value,
                "satisfiable": result.is_sat(),
                "model": result.model.assignments if result.model else None
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_extract_proof",
    description="Extract proof from Z3",
    parameters={
        "smtlib": {
            "type": "string",
            "description": "SMT-LIB problem (must be UNSAT for proof)"
        },
        "format": {
            "type": "string",
            "description": "Proof format",
            "enum": ["text", "json", "dot", "smtlib2"],
            "optional": True,
            "default": "text"
        }
    }
)
async def z3_extract_proof(
    smtlib: str,
    format: str = "text"
) -> Dict[str, Any]:
    """
    Extract proof from Z3.
    
    Example:
    {
        "smtlib": "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(assert (not (> (+ x 1) 0)))(check-sat)",
        "format": "json"
    }
    """
    if not Z3_ADVANCED_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 advanced features not available"
        }
    
    try:
        solver = get_z3_advanced_solver()
        
        format_enum = ProofFormat[format.upper()]
        result = solver.extract_proof(smtlib, format_enum)
        
        return {
            "success": result.success,
            "proof_steps": [s.to_dict() for s in result.proof_steps],
            "axioms_used": result.axioms_used,
            "tactics_used": result.tactics_used,
            "verification_status": result.verification_status,
            "raw_proof": result.raw_proof[:2000] if result.raw_proof else None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_analyze_problem",
    description="Analyze problem characteristics with optional CAV-NLP enhancement",
    parameters={
        "problem": {
            "type": "string",
            "description": "Problem description or SMT-LIB"
        },
        "use_cav_nlp": {
            "type": "boolean",
            "description": "Whether to use CAV-NLP for semantic analysis",
            "optional": True,
            "default": True
        }
    }
)
async def z3_analyze_problem(
    problem: str,
    use_cav_nlp: bool = True
) -> Dict[str, Any]:
    """
    Analyze problem to determine characteristics.
    
    When use_cav_nlp is True, uses CAV-NLP to extract semantic primitives
    and provide deeper analysis of natural language problems.
    
    Example:
    {
        "problem": "Find x and y where x + y = 10 and x > 0",
        "use_cav_nlp": true
    }
    """
    if not Z3_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 not available"
        }
    
    try:
        from z3prover_integration import Z3ProblemDetector
        
        detector = Z3ProblemDetector()
        problem_type, confidence = detector.detect_problem_type(problem)
        
        # Check if SMT-LIB
        is_smt = '(assert' in problem or '(declare' in problem
        
        # Count features
        has_arithmetic = any(op in problem for op in ['+', '-', '*', '/', '>', '<', '='])
        has_boolean = any(kw in problem.lower() for kw in ['and', 'or', 'not', 'implies'])
        has_quantifiers = any(kw in problem for kw in ['forall', 'exists', '∀', '∃'])
        
        result = {
            "success": True,
            "detected_type": problem_type,
            "confidence": confidence,
            "is_smtlib": is_smt,
            "features": {
                "arithmetic": has_arithmetic,
                "boolean": has_boolean,
                "quantifiers": has_quantifiers
            },
            "recommended_approach": "SMT solver" if is_smt or problem_type != "unknown" else "Standard solver",
            "cav_nlp_used": False
        }
        
        # Enhanced analysis with CAV-NLP
        if use_cav_nlp and UNIFIED_MATH_AVAILABLE and not is_smt:
            try:
                from openevolve.cav_nlp_integration.flexible_semantic_parsing import SemanticNormalizer
                normalizer = SemanticNormalizer()
                primitives = normalizer.normalize(problem)
                
                result["cav_nlp_used"] = True
                result["semantic_analysis"] = {
                    "primitive_count": len(primitives),
                    "primitives": [
                        {"kind": p.kind, "confidence": p.confidence, "canonical_form": p.canonical_form}
                        for p in primitives[:10]  # Limit to first 10
                    ]
                }
                
                # Try to extract dependency DAG
                try:
                    from openevolve.cav_nlp_integration.dependency_dag import PaperStructureExtractor
                    extractor = PaperStructureExtractor()
                    dag = extractor.extract_dag(f"Problem: {problem}")
                    if dag and dag.nodes:
                        result["semantic_analysis"]["statement_count"] = len(dag.nodes)
                        result["semantic_analysis"]["has_dependencies"] = len(dag.edges) > 0 if hasattr(dag, 'edges') else False
                except Exception as e:
                    logger.debug(f"DAG extraction failed: {e}")
                
                # Update recommended approach based on CAV-NLP analysis
                if primitives:
                    result["recommended_approach"] = "CAV-NLP formalization + SMT solver"
                    
            except Exception as e:
                logger.debug(f"CAV-NLP analysis failed: {e}")
                result["cav_nlp_error"] = str(e)
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_solve_portfolio",
    description="Solve using multiple strategies in parallel",
    parameters={
        "smtlib": {
            "type": "string",
            "description": "SMT-LIB problem"
        },
        "strategies": {
            "type": "array",
            "description": "List of strategies to try",
            "items": {"type": "string"},
            "optional": True
        },
        "timeout": {
            "type": "number",
            "optional": True,
            "default": 30
        }
    }
)
async def z3_solve_portfolio(
    smtlib: str,
    strategies: Optional[List[str]] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Portfolio solving with multiple strategies.
    
    Example:
    {
        "smtlib": "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)",
        "strategies": ["default", "smt", "qflia"]
    }
    """
    if not Z3_ADVANCED_AVAILABLE:
        return {
            "success": False,
            "error": "Z3 advanced features not available"
        }
    
    try:
        solver = get_z3_advanced_solver(Z3Config(timeout=timeout))
        
        result = solver.solve_portfolio(smtlib, strategies)
        
        return {
            "success": result.success,
            "winner_strategy": result.winner_strategy,
            "execution_time": result.execution_time,
            "parallel_speedup": result.parallel_speedup,
            "strategies_tried": len(result.all_results),
            "status": result.best_result.status.value if result.best_result else None,
            "model": result.best_result.model.assignments if result.best_result and result.best_result.model else None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# =============================================================================
# CAV-NLP Enhanced Tools
# =============================================================================

@MCPTool(
    name="z3_formalize_constraint",
    description="Formalize natural language constraint to Z3/SMT-LIB using CAV-NLP",
    parameters={
        "natural_language": {
            "type": "string",
            "description": "Natural language description of the constraint"
        },
        "target_format": {
            "type": "string",
            "description": "Target formalization format",
            "enum": ["z3", "lean", "smtlib"],
            "optional": True,
            "default": "lean"
        },
        "elaborate": {
            "type": "boolean",
            "description": "Whether to elaborate the generated code",
            "optional": True,
            "default": True
        }
    }
)
async def z3_formalize_constraint(
    natural_language: str,
    target_format: str = "lean",
    elaborate: bool = True
) -> Dict[str, Any]:
    """
    Formalize natural language constraint to formal code using CAV-NLP.
    
    Uses CAV-NLP (Canonical Arithmetic Verification via NLP) to convert
    natural language or LaTeX mathematical statements into formal code.
    
    Example:
    {
        "natural_language": "For all x > 0, x + 1 > 1",
        "target_format": "lean",
        "elaborate": true
    }
    
    Returns:
    {
        "success": true,
        "code": "import Mathlib\n\ntheorem ...",
        "source": "cav_nlp",
        "elaborated_code": "..."  # if elaboration requested
    }
    """
    if not UNIFIED_MATH_AVAILABLE:
        return {
            "success": False,
            "error": "CAV-NLP/Unified Math Service not available"
        }
    
    try:
        service = create_unified_math_service()
        result = await service.formalize(natural_language, elaborate=elaborate)
        
        response = {
            "success": result.success,
            "code": result.code,
            "source": result.source,
            "raw_text": result.raw_text,
            "warnings": result.warnings
        }
        
        if result.elaborated_code:
            response["elaborated_code"] = result.elaborated_code
        
        if result.documentation:
            response["documentation"] = result.documentation
        
        # Add metadata
        response["metadata"] = {
            "timestamp": result.timestamp,
            "cav_nlp_used": result.source == "cav_nlp"
        }
        
        return response
    
    except Exception as e:
        logger.error(f"CAV-NLP formalization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "error"
        }


@MCPTool(
    name="z3_verify_hybrid",
    description="Verify constraint using hybrid Z3 + Lean approach with CAV-NLP",
    parameters={
        "constraint": {
            "type": "string",
            "description": "Constraint to verify (natural language, SMT-LIB, or Lean code)"
        },
        "input_format": {
            "type": "string",
            "description": "Format of the input constraint",
            "enum": ["auto", "natural_language", "smtlib", "lean"],
            "optional": True,
            "default": "auto"
        },
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds",
            "optional": True,
            "default": 30
        }
    }
)
async def z3_verify_hybrid(
    constraint: str,
    input_format: str = "auto",
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Verify constraint using hybrid Z3 + Lean approach.
    
    Pipeline:
    1. If natural language: formalize using CAV-NLP
    2. Quick check using Z3
    3. Formal verification using Lean/CAV-NLP
    
    Example:
    {
        "constraint": "For all integers x, x + 0 = x",
        "input_format": "natural_language"
    }
    
    Returns:
    {
        "success": true,
        "verified": true,
        "z3_result": {...},
        "lean_verification": {...},
        "hybrid_confidence": 0.95
    }
    """
    result = {
        "success": True,
        "verified": False,
        "z3_result": None,
        "lean_verification": None,
        "hybrid_confidence": 0.0,
        "errors": []
    }
    
    lean_code = None
    
    # Step 1: Formalize if needed
    if input_format == "natural_language" or (input_format == "auto" and '(assert' not in constraint):
        if UNIFIED_MATH_AVAILABLE:
            try:
                service = create_unified_math_service()
                formalization = await service.formalize(constraint, elaborate=True)
                if formalization.success:
                    lean_code = formalization.elaborated_code or formalization.code
                    result["formalization"] = {
                        "success": True,
                        "source": formalization.source,
                        "code": lean_code
                    }
                else:
                    result["errors"].append("Formalization failed")
            except Exception as e:
                result["errors"].append(f"Formalization error: {str(e)}")
        else:
            result["errors"].append("CAV-NLP not available for natural language input")
    elif input_format == "lean" or '(theorem' in constraint or '(lemma' in constraint:
        lean_code = constraint
    elif input_format == "smtlib" or '(assert' in constraint:
        # Will use Z3 directly
        lean_code = None
    
    # Step 2: Quick check with Z3
    if Z3_AVAILABLE and '(assert' in constraint or '(set-logic' in constraint:
        try:
            prover = get_z3_theorem_prover()
            z3_result = prover.prove_theorem(constraint, [])
            result["z3_result"] = {
                "proven": z3_result.proven,
                "status": "verified" if z3_result.proven else "not_proven",
                "execution_time": z3_result.execution_time
            }
            if z3_result.proven:
                result["verified"] = True
                result["hybrid_confidence"] += 0.4
        except Exception as e:
            result["errors"].append(f"Z3 verification error: {str(e)}")
    
    # Step 3: Formal verification with Lean
    if lean_code and UNIFIED_MATH_AVAILABLE:
        try:
            service = create_unified_math_service()
            verification = await service.verify(lean_code)
            if verification:
                result["lean_verification"] = {
                    "success": verification.success,
                    "status": str(verification.status) if hasattr(verification, 'status') else "unknown",
                    "errors": verification.errors if hasattr(verification, 'errors') else []
                }
                if verification.success:
                    result["verified"] = True
                    result["hybrid_confidence"] += 0.6
            else:
                result["lean_verification"] = {"success": False, "error": "Verification returned None"}
        except Exception as e:
            result["errors"].append(f"Lean verification error: {str(e)}")
    
    # Normalize confidence
    result["hybrid_confidence"] = min(1.0, result["hybrid_confidence"])
    
    return result


@MCPTool(
    name="z3_canonicalize_constraint",
    description="Return canonical form of constraint using CAV-NLP",
    parameters={
        "constraint": {
            "type": "string",
            "description": "Constraint to canonicalize (natural language or SMT-LIB)"
        },
        "input_type": {
            "type": "string",
            "description": "Type of input",
            "enum": ["auto", "natural_language", "smtlib"],
            "optional": True,
            "default": "auto"
        }
    }
)
async def z3_canonicalize_constraint(
    constraint: str,
    input_type: str = "auto"
) -> Dict[str, Any]:
    """
    Return canonical form of constraint using CAV-NLP canonicalization.
    
    CAV-NLP canonicalization normalizes mathematical expressions to a
    standard form that can be compared and analyzed.
    
    Example:
    {
        "constraint": "x + y = y + x",
        "input_type": "smtlib"
    }
    
    Returns:
    {
        "success": true,
        "canonical_form": "(= (+ x y) (+ y x))",
        "semantic_primitives": [...],
        "complexity_score": 0.5
    }
    """
    if not UNIFIED_MATH_AVAILABLE:
        return {
            "success": False,
            "error": "CAV-NLP not available for canonicalization"
        }
    
    try:
        # Detect input type if auto
        if input_type == "auto":
            if '(assert' in constraint or '(declare' in constraint:
                input_type = "smtlib"
            else:
                input_type = "natural_language"
        
        # For natural language, first formalize
        smtlib_code = constraint
        if input_type == "natural_language":
            service = create_unified_math_service()
            formalization = await service.formalize(constraint, elaborate=False)
            if formalization.success:
                # Extract SMT-LIB if embedded in Lean code
                smtlib_code = formalization.code
            else:
                return {
                    "success": False,
                    "error": "Failed to formalize natural language input"
                }
        
        # Use CAV-NLP canonicalizer if available
        if CANONICALIZER_AVAILABLE:
            try:
                canonicalizer = Z3Canonicalizer()
                canonical_form = canonicalizer.canonicalize(smtlib_code)
                
                return {
                    "success": True,
                    "canonical_form": canonical_form,
                    "input_type": input_type,
                    "original": constraint[:200] + "..." if len(constraint) > 200 else constraint
                }
            except Exception as e:
                logger.warning(f"Canonicalizer failed: {e}")
        
        # Fallback: Use basic normalization
        # Extract semantic primitives if available
        semantic_primitives = []
        try:
            from openevolve.cav_nlp_integration.flexible_semantic_parsing import SemanticNormalizer
            normalizer = SemanticNormalizer()
            primitives = normalizer.normalize(constraint)
            semantic_primitives = [
                {"kind": p.kind, "canonical_form": p.canonical_form}
                for p in primitives
            ]
        except Exception as e:
            logger.debug(f"Semantic primitive extraction failed: {e}")
        
        # Return basic canonicalization result
        return {
            "success": True,
            "canonical_form": smtlib_code.strip(),
            "semantic_primitives": semantic_primitives,
            "note": "Fallback canonicalization (canonicalizer unavailable)"
        }
    
    except Exception as e:
        logger.error(f"Canonicalization failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_enhanced_prove",
    description="Prove theorem using CAV-NLP enhanced verification",
    parameters={
        "theorem": {
            "type": "string",
            "description": "Theorem statement (natural language, SMT-LIB, or Lean)"
        },
        "use_cav_nlp": {
            "type": "boolean",
            "description": "Whether to use CAV-NLP for enhanced verification",
            "optional": True,
            "default": True
        },
        "generate_proof": {
            "type": "boolean",
            "description": "Whether to generate a proof sketch",
            "optional": True,
            "default": True
        },
        "input_format": {
            "type": "string",
            "description": "Format of input theorem",
            "enum": ["auto", "natural_language", "smtlib", "lean"],
            "optional": True,
            "default": "auto"
        }
    }
)
async def z3_enhanced_prove(
    theorem: str,
    use_cav_nlp: bool = True,
    generate_proof: bool = True,
    input_format: str = "auto"
) -> Dict[str, Any]:
    """
    Prove theorem with optional CAV-NLP enhanced verification.
    
    When use_cav_nlp is True:
    - Natural language is formalized using CAV-NLP
    - Lean code is verified using CAV-NLP/LeanAide
    - Proof sketch generation uses semantic analysis
    
    Example:
    {
        "theorem": "For all natural numbers n, n + 0 = n",
        "use_cav_nlp": true,
        "generate_proof": true,
        "input_format": "natural_language"
    }
    """
    if not use_cav_nlp:
        # Fall back to standard Z3 proving
        return await z3_prove_theorem(theorem, [], extract_proof=generate_proof)
    
    if not UNIFIED_MATH_AVAILABLE:
        logger.warning("CAV-NLP not available, falling back to standard Z3 proving")
        return await z3_prove_theorem(theorem, [], extract_proof=generate_proof)
    
    try:
        service = create_unified_math_service()
        
        # Detect input format if auto
        if input_format == "auto":
            if '(theorem' in theorem or '(lemma' in theorem or 'import Mathlib' in theorem:
                input_format = "lean"
            elif '(assert' in theorem or '(set-logic' in theorem:
                input_format = "smtlib"
            else:
                input_format = "natural_language"
        
        lean_code = theorem
        formalization_result = None
        
        # Step 1: Formalize if natural language
        if input_format == "natural_language":
            formalization_result = await service.formalize(theorem, elaborate=True)
            if formalization_result.success:
                lean_code = formalization_result.elaborated_code or formalization_result.code
            else:
                return {
                    "success": False,
                    "error": "Failed to formalize natural language theorem",
                    "formalization_warnings": formalization_result.warnings
                }
        
        result = {
            "success": True,
            "proven": False,
            "input_format": input_format,
            "lean_code": lean_code,
            "source": formalization_result.source if formalization_result else "input"
        }
        
        # Step 2: Verify with Lean
        verification = await service.verify(lean_code)
        if verification:
            result["proven"] = verification.success
            result["verification"] = {
                "success": verification.success,
                "status": str(verification.status) if hasattr(verification, 'status') else "unknown"
            }
        
        # Step 3: Generate proof if requested
        if generate_proof and UNIFIED_MATH_AVAILABLE:
            try:
                proof_result = await service.prove(theorem)
                result["proof"] = {
                    "code": proof_result.proof_code if hasattr(proof_result, 'proof_code') else None,
                    "sketch": proof_result.sketch if hasattr(proof_result, 'sketch') else None,
                    "tactics": proof_result.tactics_used if hasattr(proof_result, 'tactics_used') else []
                }
            except Exception as e:
                result["proof_error"] = str(e)
        
        return result
    
    except Exception as e:
        logger.error(f"CAV-NLP enhanced proving failed: {e}")
        # Fallback to standard Z3
        return await z3_prove_theorem(theorem, [], extract_proof=generate_proof)


@MCPTool(
    name="z3_translate_solidity_invariant",
    description="Translate Solidity state transitions into Z3 constraints and Lean invariants",
    parameters={
        "statement": {
            "type": "string",
            "description": "Solidity assignment/update statement to translate"
        },
        "non_negative_target": {
            "type": "boolean",
            "description": "Add non-negative target invariant",
            "optional": True,
            "default": True
        },
        "max_withdraw_expr": {
            "type": "string",
            "description": "Optional max-withdraw invariant expression",
            "optional": True
        },
        "verify_translation": {
            "type": "boolean",
            "description": "Run Z3 validation for translated invariants",
            "optional": True,
            "default": True
        },
        "assume_non_negative_amount": {
            "type": "boolean",
            "description": "When verifying, assume amount >= 0",
            "optional": True,
            "default": True
        },
    }
)
async def z3_translate_solidity_invariant(
    statement: str,
    non_negative_target: bool = True,
    max_withdraw_expr: Optional[str] = None,
    verify_translation: bool = True,
    assume_non_negative_amount: bool = True,
) -> Dict[str, Any]:
    """Translate Solidity assignment semantics and optionally verify invariants."""
    if translate_solidity_assignment_to_z3 is None:
        return {
            "success": False,
            "error": "Solidity invariant translation is unavailable"
        }

    try:
        translation = translate_solidity_assignment_to_z3(
            statement=statement,
            non_negative_target=non_negative_target,
            max_withdraw_expr=max_withdraw_expr,
        )
        result: Dict[str, Any] = {
            "success": True,
            "translation": translation,
        }
        if verify_translation and verify_solidity_invariant_translation is not None:
            result["verification"] = verify_solidity_invariant_translation(
                translation=translation,
                assume_non_negative_amount=assume_non_negative_amount,
            )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_solve_smart_contract_exploit_witness",
    description="Solve symbolic exploit witness predicates for smart-contract balance drain conditions",
    parameters={
        "additional_constraints": {
            "type": "array",
            "description": "Optional additional SMT constraints",
            "items": {"type": "string"},
            "optional": True
        },
        "timeout": {
            "type": "number",
            "description": "Solver timeout in seconds",
            "optional": True,
            "default": 10.0
        },
    }
)
async def z3_solve_smart_contract_exploit_witness(
    additional_constraints: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Solve canonical exploit witness query for Web3 audit workflows."""
    if solve_smart_contract_exploit_witness is None:
        return {
            "success": False,
            "error": "Smart contract exploit witness solver is unavailable"
        }
    try:
        witness = solve_smart_contract_exploit_witness(
            additional_constraints=additional_constraints,
            timeout=timeout,
        )
        return {
            "success": True,
            "witness": witness,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@MCPTool(
    name="z3_web3_audit_exploit_verification",
    description="Run full Web3 formal pass: invariant translation + optional verification + exploit witness solving",
    parameters={
        "statement": {
            "type": "string",
            "description": "Solidity assignment/update statement to translate",
            "optional": True,
            "default": "balance[msg.sender] -= amount;"
        },
        "non_negative_target": {
            "type": "boolean",
            "description": "Add non-negative target invariant",
            "optional": True,
            "default": True
        },
        "max_withdraw_expr": {
            "type": "string",
            "description": "Optional withdrawal upper-bound expression",
            "optional": True
        },
        "verify_translation": {
            "type": "boolean",
            "description": "Run Z3 invariant implication check",
            "optional": True,
            "default": True
        },
        "assume_non_negative_amount": {
            "type": "boolean",
            "description": "When verifying, assume amount >= 0",
            "optional": True,
            "default": True
        },
        "additional_constraints": {
            "type": "array",
            "description": "Optional additional SMT constraints for witness search",
            "items": {"type": "string"},
            "optional": True
        },
        "timeout": {
            "type": "number",
            "description": "Solver timeout in seconds",
            "optional": True,
            "default": 10.0
        },
    }
)
async def z3_web3_audit_exploit_verification(
    statement: str = "balance[msg.sender] -= amount;",
    non_negative_target: bool = True,
    max_withdraw_expr: Optional[str] = None,
    verify_translation: bool = True,
    assume_non_negative_amount: bool = True,
    additional_constraints: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Run combined Web3 exploit verification workflow with Z3 tools."""
    if translate_solidity_assignment_to_z3 is None:
        return {"success": False, "error": "Solidity invariant translation is unavailable"}
    if solve_smart_contract_exploit_witness is None:
        return {"success": False, "error": "Smart contract exploit witness solver is unavailable"}

    try:
        translation = translate_solidity_assignment_to_z3(
            statement=statement,
            non_negative_target=non_negative_target,
            max_withdraw_expr=max_withdraw_expr,
        )
        verification: Optional[Dict[str, Any]] = None
        if verify_translation and verify_solidity_invariant_translation is not None:
            verification = verify_solidity_invariant_translation(
                translation=translation,
                assume_non_negative_amount=assume_non_negative_amount,
            )
        witness = solve_smart_contract_exploit_witness(
            additional_constraints=additional_constraints,
            timeout=timeout,
        )
        lean_proof_verification = await verify_web3_lean_proof_async(
            translation,
            use_real_lean=True,
        )
        verified_exploit = bool(witness.get("satisfiable", False))
        if verify_translation and isinstance(verification, dict):
            verified_exploit = verified_exploit and bool(verification.get("proven", False))

        return {
            "success": True,
            "translation": translation,
            "verification": verification,
            "exploit_witness": witness,
            "lean_proof_verification": lean_proof_verification,
            "formal_evidence": build_web3_formal_evidence(
                verification,
                witness,
                lean_proof_verification,
            ),
            "verified_exploit": verified_exploit,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_web3_formal_tool_inventory() -> Dict[str, Any]:
    """Return Web3 formal-verification MCP tool inventory from the Z3 service."""
    formal_capabilities = {
        "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
        "invariant_translation_verification": verify_solidity_invariant_translation is not None,
        "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
        "composite_exploit_verification": (
            translate_solidity_assignment_to_z3 is not None
            and solve_smart_contract_exploit_witness is not None
        ),
    }
    tools: List[str] = []
    if formal_capabilities["solidity_invariant_translation"]:
        tools.append("z3_translate_solidity_invariant")
    if formal_capabilities["symbolic_exploit_witness"]:
        tools.append("z3_solve_smart_contract_exploit_witness")
    if formal_capabilities["composite_exploit_verification"]:
        tools.append("z3_web3_audit_exploit_verification")
    tools = sorted(set(tools))
    available = bool(tools) or any(bool(v) for v in formal_capabilities.values())
    if not available:
        available = bool(WEB3_FORMAL_AVAILABLE)
    return {
        "available": available,
        "tools": tools,
        "web3_formal_tools": tools,
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": bool(
            formal_capabilities.get("composite_exploit_verification")
        ),
    }


# =============================================================================
# MCP Server Interface
# =============================================================================

class Z3MCPServer:
    """
    MCP Server for Z3 integration.
    
    Can be used standalone or integrated with existing MCP infrastructure.
    """
    
    def __init__(self):
        self.tools = MCPTool.get_registry()
        self.metadata = MCPTool.get_metadata()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        return self.metadata
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name."""
        return MCPTool.execute(tool_name, params)
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method")
        
        if method == "list_tools":
            return {
                "success": True,
                "tools": self.list_tools()
            }
        
        elif method == "call_tool":
            tool_name = request.get("tool")
            params = request.get("params", {})
            return self.call_tool(tool_name, params)
        
        else:
            return {
                "success": False,
                "error": f"Unknown method: {method}"
            }


# =============================================================================
# Global Server Instance
# =============================================================================

_mcp_server: Optional[Z3MCPServer] = None


def get_z3_mcp_server() -> Z3MCPServer:
    """Get global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = Z3MCPServer()
    return _mcp_server


# =============================================================================
# Example Usage
# =============================================================================

def example_mcp_usage():
    """Example: Using MCP tools directly."""
    server = get_z3_mcp_server()
    
    # List tools
    print("Available Z3 MCP Tools:")
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Call solve tool
    result = server.call_tool("z3_solve_constraints", {
        "variables": [
            {"name": "x", "type": "INTEGER"},
            {"name": "y", "type": "INTEGER"}
        ],
        "constraints": [
            "x > 0",
            "x < 10",
            "y == x + 5"
        ]
    })
    
    print("\nSolve Result:")
    print(json.dumps(result, indent=2))


async def example_cav_nlp_usage():
    """Example: Using CAV-NLP enhanced tools."""
    print("\n" + "=" * 50)
    print("CAV-NLP Enhanced Tools Examples")
    print("=" * 50)
    
    server = get_z3_mcp_server()
    
    # Example 1: Formalize natural language
    print("\n1. Formalize Natural Language Constraint:")
    result = server.call_tool("z3_formalize_constraint", {
        "natural_language": "For all natural numbers n, n + 0 = n",
        "target_format": "lean",
        "elaborate": True
    })
    print(f"   Success: {result.get('success')}")
    print(f"   Source: {result.get('source')}")
    if result.get('code'):
        print(f"   Code:\n{result['code'][:300]}...")
    
    # Example 2: Hybrid verification
    print("\n2. Hybrid Verification:")
    result = server.call_tool("z3_verify_hybrid", {
        "constraint": "For all integers x, x + 0 = x",
        "input_format": "natural_language"
    })
    print(f"   Success: {result.get('success')}")
    print(f"   Verified: {result.get('verified')}")
    print(f"   Hybrid Confidence: {result.get('hybrid_confidence')}")
    
    # Example 3: Canonicalization
    print("\n3. Canonicalize Constraint:")
    result = server.call_tool("z3_canonicalize_constraint", {
        "constraint": "x + y = y + x",
        "input_type": "smtlib"
    })
    print(f"   Success: {result.get('success')}")
    if result.get('canonical_form'):
        print(f"   Canonical Form: {result['canonical_form'][:100]}...")
    
    # Example 4: Enhanced proving
    print("\n4. Enhanced Proving with CAV-NLP:")
    result = server.call_tool("z3_enhanced_prove", {
        "theorem": "For all natural numbers n, n + 0 = n",
        "use_cav_nlp": True,
        "generate_proof": True,
        "input_format": "natural_language"
    })
    print(f"   Success: {result.get('success')}")
    print(f"   Proven: {result.get('proven')}")
    print(f"   Source: {result.get('source')}")


def example_mcp_protocol():
    """Example: MCP protocol request/response."""
    server = get_z3_mcp_server()
    
    # List tools request
    request = {"method": "list_tools"}
    response = server.handle_request(request)
    
    print("List Tools Response:")
    print(json.dumps(response, indent=2))
    
    # Call tool request
    request = {
        "method": "call_tool",
        "tool": "z3_analyze_problem",
        "params": {
            "problem": "Find x and y where x + y = 10 and x > 0"
        }
    }
    response = server.handle_request(request)
    
    print("\nAnalyze Problem Response:")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    print("Z3 MCP Tools")
    print("=" * 50)
    
    example_mcp_usage()
    print("\n" + "=" * 50)
    example_mcp_protocol()
    
    # Run CAV-NLP examples
    import asyncio
    asyncio.run(example_cav_nlp_usage())
