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

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, get_z3_solver_engine, get_z3_theorem_prover,
        is_z3_available
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
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
    logger.warning("Z3 advanced features not available")

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, TranslationDirection,
        get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_AVAILABLE = False
    logger.warning("Z3-LeanAIDE bridge not available")


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
                    "type": {"type": "string", "enum": ["BOOLEAN", "INTEGER", "REAL", "BIT_VECTOR"]},
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
            "(> x 0)",
            "(< x 10)",
            "(= y (+ x 5))"
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
            var_type = Z3ConstraintType[v.get('type', 'INTEGER')]
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
        "constraints": ["(>= x 0)", "(<= x 100)"],
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
            var_type = Z3ConstraintType[v.get('type', 'INTEGER')]
            z3_vars.append(Z3Variable(v['name'], var_type))
        
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
    3. {"operation": "add", "state_id": "...", "constraint": "(< x 10)"}
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
                Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER')])
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
    description="Analyze problem characteristics",
    parameters={
        "problem": {
            "type": "string",
            "description": "Problem description or SMT-LIB"
        }
    }
)
async def z3_analyze_problem(problem: str) -> Dict[str, Any]:
    """
    Analyze problem to determine characteristics.
    
    Example:
    {
        "problem": "Find x and y where x + y = 10 and x > 0"
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
        
        return {
            "success": True,
            "detected_type": problem_type,
            "confidence": confidence,
            "is_smtlib": is_smt,
            "features": {
                "arithmetic": has_arithmetic,
                "boolean": has_boolean,
                "quantifiers": has_quantifiers
            },
            "recommended_approach": "SMT solver" if is_smt or problem_type != "unknown" else "Standard solver"
        }
    
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
            "(> x 0)",
            "(< x 10)",
            "(= y (+ x 5))"
        ]
    })
    
    print("\nSolve Result:")
    print(json.dumps(result, indent=2))


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
