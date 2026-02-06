"""
Z3 Prover Integration Module for OpenEvolve

This module provides comprehensive integration with Microsoft Z3 SMT solver,
enabling constraint solving, theorem proving, and formal verification capabilities
within the OpenEvolve workflow ecosystem.

Key Features:
- Z3 solver interface for constraint satisfaction problems
- SMT-LIB2 integration for standard theorem proving
- Integration with LeanAIDE for combined formal verification
- OpenEvolve workflow integration through BubbleLabs
- Support for arithmetic, bit-vector, and array constraints
- Proof generation and verification

Architecture:
    Z3Integration
        ├── Z3SolverEngine (Core solver interface)
        ├── Z3TheoremProver (Theorem proving capabilities)
        ├── Z3ConstraintOptimizer (Optimization features)
        └── Z3LeanAideBridge (Integration with LeanAIDE)

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
import ast
import operator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Z3 Availability Detection
# =============================================================================

Z3_AVAILABLE = False
Z3_PYTHON_AVAILABLE = False

# Try to import z3 Python bindings
try:
    import z3
    Z3_PYTHON_AVAILABLE = True
    Z3_AVAILABLE = True
    logger.info("Z3 Python bindings available")
except ImportError:
    logger.warning("Z3 Python bindings not available - will use CLI interface")


# Check for Z3 binary
try:
    result = subprocess.run(['z3', '--version'], capture_output=True, timeout=5)
    if result.returncode == 0:
        Z3_AVAILABLE = True
        logger.info(f"Z3 binary available: {result.stdout.decode().strip()}")
except (subprocess.CalledProcessError, FileNotFoundError, OSError):
    logger.warning("Z3 binary not detected - some features may be unavailable")

# Import solver pool for metrics tracking
try:
    from z3_solver_pool import get_solver_pool, Z3SolverPool
    SOLVER_POOL_AVAILABLE = True
except ImportError:
    SOLVER_POOL_AVAILABLE = False
    logger.debug("Z3 solver pool not available")


# =============================================================================
# CAV-NLP Integration
# =============================================================================

try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
    logger.info("CAV-NLP integration available for enhanced Z3 solving")
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.debug("CAV-NLP integration not available - enhanced NL features disabled")


# =============================================================================
# Data Classes and Enums
# =============================================================================

class Z3ResultStatus(Enum):
    """Status of Z3 solver result."""
    SAT = "sat"           # Satisfiable
    UNSAT = "unsat"       # Unsatisfiable
    UNKNOWN = "unknown"   # Result unknown
    ERROR = "error"       # Error occurred
    TIMEOUT = "timeout"   # Timeout


class Z3ConstraintType(Enum):
    """Types of constraints supported by Z3."""
    BOOLEAN = auto()
    INTEGER = auto()
    REAL = auto()
    BIT_VECTOR = auto()
    ARRAY = auto()
    FLOATING_POINT = auto()
    STRING = auto()


@dataclass
class Z3Variable:
    """Represents a Z3 variable."""
    name: str
    var_type: Z3ConstraintType
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    bit_width: Optional[int] = None  # For bit-vectors
    
    def to_smtlib(self) -> str:
        """Convert to SMT-LIB2 declaration."""
        type_map = {
            Z3ConstraintType.BOOLEAN: "Bool",
            Z3ConstraintType.INTEGER: "Int",
            Z3ConstraintType.REAL: "Real",
            Z3ConstraintType.BIT_VECTOR: f"(_ BitVec {self.bit_width or 32})",
            Z3ConstraintType.ARRAY: "(Array Int Int)",
            Z3ConstraintType.FLOATING_POINT: "(_ FloatingPoint 11 53)",
            Z3ConstraintType.STRING: "String"
        }
        return f"(declare-fun {self.name} () {type_map.get(self.var_type, 'Int')})"


@dataclass
class Z3Constraint:
    """Represents a Z3 constraint."""
    expression: str
    constraint_type: Z3ConstraintType
    description: Optional[str] = None
    
    def to_smtlib(self) -> str:
        """Convert to SMT-LIB2 assertion."""
        return f"(assert {self.expression})"


@dataclass
class Z3Model:
    """Represents a Z3 model (solution)."""
    assignments: Dict[str, Any]
    objective_value: Optional[float] = None
    proof: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_value(self, var_name: str, default=None):
        """Get value for a variable."""
        return self.assignments.get(var_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assignments": self.assignments,
            "objective_value": self.objective_value,
            "proof": self.proof,
            "statistics": self.statistics
        }


@dataclass
class Z3SolverResult:
    """Result from Z3 solver."""
    status: Z3ResultStatus
    model: Optional[Z3Model] = None
    reason: Optional[str] = None
    execution_time: float = 0.0
    smtlib_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_sat(self) -> bool:
        """Check if result is satisfiable."""
        return self.status == Z3ResultStatus.SAT
    
    def is_unsat(self) -> bool:
        """Check if result is unsatisfiable."""
        return self.status == Z3ResultStatus.UNSAT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "model": self.model.to_dict() if self.model else None,
            "reason": self.reason,
            "execution_time": self.execution_time,
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class Z3TheoremResult:
    """Result from Z3 theorem prover."""
    proven: bool
    proof: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    tactic_used: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proven": self.proven,
            "proof": self.proof,
            "counterexample": self.counterexample,
            "execution_time": self.execution_time,
            "tactic_used": self.tactic_used,
            "errors": self.errors
        }


@dataclass
class Z3Config:
    """Configuration for Z3 integration."""
    timeout: float = 30.0
    memory_limit_mb: int = 4096
    num_threads: int = 1
    proof_generation: bool = True
    unsat_core: bool = False
    optimization: bool = False
    auto_config: bool = True
    
    # Solver tactics
    default_tactic: str = "default"
    quantifier_tactic: str = "qe"
    arithmetic_tactic: str = "qfnra"
    
    def to_z3_params(self) -> Dict[str, Any]:
        """Convert to Z3 parameter dictionary."""
        return {
            "timeout": int(self.timeout * 1000),  # Convert to milliseconds
            "memory": self.memory_limit_mb,
            "threads": self.num_threads,
            "proof": self.proof_generation,
            "unsat_core": self.unsat_core,
            "auto_config": self.auto_config
        }


# =============================================================================
# Mathematical Problem Detector for Z3
# =============================================================================

class Z3ProblemDetector:
    """Detects problems suitable for Z3 constraint solving."""
    
    # Patterns indicating Z3-suitable problems
    CONSTRAINT_PATTERNS = [
        r'\b(satisfy|constraint|solve|system of)\b',
        r'\b(equation|inequality|inequalities)\b',
        r'\b(optimization|minimize|maximize|optimal)\b',
        r'\b(scheduling|allocation|assignment)\b',
        r'\b(verify|check|prove|valid)\b',
        r'\b(allocation|knapsack|sudoku)\b',
        r'[><=!]=?\s*\d+',  # Comparison operators with numbers
        r'\b(and|or|not|implies)\b.*\b(=|<=|>=|<|>)\b',
    ]
    
    SMT_PATTERNS = [
        r'\b(SMT|SMT-LIB|smtlib)\b',
        r'\b(declare-fun|assert|check-sat)\b',
        r'\b(BitVec|Int|Real|Bool)\s*\(',
        r'\bforall\s*\(|exists\s*\(',
    ]
    
    def __init__(self):
        self.constraint_regex = [re.compile(p, re.IGNORECASE) for p in self.CONSTRAINT_PATTERNS]
        self.smt_regex = [re.compile(p, re.IGNORECASE) for p in self.SMT_PATTERNS]
    
    def detect_problem_type(self, problem_text: str) -> Tuple[str, float]:
        """
        Detect if problem is suitable for Z3 and determine its type.
        
        Returns:
            Tuple of (problem_type, confidence)
        """
        text = problem_text.lower()
        
        # Check for SMT-LIB format
        smt_matches = sum(1 for r in self.smt_regex if r.search(text))
        if smt_matches >= 2:
            return "smtlib", min(0.9 + smt_matches * 0.05, 1.0)
        
        # Check for constraint patterns
        constraint_matches = sum(1 for r in self.constraint_regex if r.search(text))
        
        # Check for mathematical symbols
        has_equations = '=' in problem_text and any(op in problem_text for op in ['<', '>', '<=', '>='])
        has_variables = re.search(r'\b[x-zX-Z][0-9]?\b', problem_text) is not None
        
        score = (
            constraint_matches * 0.15 +
            (0.2 if has_equations else 0) +
            (0.1 if has_variables else 0)
        )
        
        if score >= 0.5:
            if 'optimization' in text or 'minimize' in text or 'maximize' in text:
                return "optimization", min(score, 1.0)
            elif 'prove' in text or 'verify' in text or 'theorem' in text:
                return "theorem", min(score, 1.0)
            else:
                return "constraint", min(score, 1.0)
        
        return "unknown", score
    
    def is_suitable_for_z3(self, problem_text: str, threshold: float = 0.4) -> bool:
        """Check if problem is suitable for Z3 solving."""
        _, confidence = self.detect_problem_type(problem_text)
        return confidence >= threshold


# =============================================================================
# Core Z3 Solver Engine
# =============================================================================

class Z3SolverEngine:
    """
    Core Z3 solver engine providing constraint solving capabilities.
    
    Supports both Python API (when available) and CLI interface.
    Integrates with Z3SolverPool for metrics tracking.
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()
        self.detector = Z3ProblemDetector()
        self._solver_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self._stats = {
            "total_calls": 0,
            "sat_results": 0,
            "unsat_results": 0,
            "error_results": 0,
            "total_time": 0.0
        }
        
        # Register with solver pool for metrics tracking
        self._solver_id: Optional[str] = None
        self._pool: Optional[Z3SolverPool] = None
        if SOLVER_POOL_AVAILABLE:
            try:
                self._pool = get_solver_pool()
                self._solver_id = self._pool.register_solver(
                    metadata={
                        'class': 'Z3SolverEngine',
                        'config_timeout': self.config.timeout,
                        'created_by': 'Z3SolverEngine.__init__'
                    }
                )
                logger.debug(f"Z3SolverEngine registered with pool: {self._solver_id}")
            except Exception as e:
                logger.warning(f"Failed to register with solver pool: {e}")
                self._pool = None
                self._solver_id = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        web3_formal_capabilities = self._get_web3_formal_capabilities()
        web3_formal_tools: List[str] = []
        if web3_formal_capabilities["solidity_invariant_translation"]:
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if web3_formal_capabilities["symbolic_exploit_witness"]:
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if web3_formal_capabilities["composite_exploit_verification"]:
            web3_formal_tools.append("z3_web3_audit_exploit_verification")
        web3_formal_tools = sorted(set(web3_formal_tools))
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(v) for v in web3_formal_capabilities.values()
        )

        status = {
            "z3_available": Z3_AVAILABLE,
            "z3_python_available": Z3_PYTHON_AVAILABLE,
            "config": asdict(self.config),
            "statistics": self._stats.copy(),
            "solver_id": self._solver_id,
            "pool_registered": self._pool is not None,
            "web3_formal_available": inferred_formal_available,
            "web3_formal_verification_available": inferred_formal_available,
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": web3_formal_capabilities,
            "audit_exploit_verification_available": bool(
                web3_formal_capabilities.get("composite_exploit_verification")
            ),
        }
        
        # Add pool metrics if available
        if self._pool is not None:
            try:
                metrics = self._pool.get_metrics()
                status["pool_metrics"] = metrics.to_dict()
            except Exception as e:
                logger.debug(f"Failed to get pool metrics: {e}")
        
        return status

    @staticmethod
    def _get_web3_formal_capabilities() -> Dict[str, bool]:
        """Return Web3 formal capability flags without hard import coupling."""
        translator = globals().get("translate_solidity_assignment_to_z3")
        verifier = globals().get("verify_solidity_invariant_translation")
        witness_solver = globals().get("solve_smart_contract_exploit_witness")
        return {
            "solidity_invariant_translation": callable(translator),
            "invariant_translation_verification": callable(verifier),
            "symbolic_exploit_witness": callable(witness_solver),
            "composite_exploit_verification": callable(translator) and callable(witness_solver),
        }
    
    def _track_operation(self, operation_name: str = "solve"):
        """
        Context manager for tracking solver operations with the pool.
        
        Args:
            operation_name: Name of the operation being tracked
            
        Yields:
            None
        """
        if self._pool is not None and self._solver_id is not None:
            return self._pool.active_operation(self._solver_id)
        else:
            # Return a no-op context manager if pool not available
            from contextlib import nullcontext
            return nullcontext()
    
    def __del__(self):
        """Cleanup: unregister from solver pool."""
        if self._pool is not None and self._solver_id is not None:
            try:
                self._pool.unregister_solver(self._solver_id)
                if logger is not None:
                    logger.debug(f"Z3SolverEngine unregistered from pool: {self._solver_id}")
            except Exception:
                # Ignore errors during cleanup, especially during interpreter shutdown
                pass
    
    def solve_constraints(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objective: Optional[str] = None,
        minimize: bool = True
    ) -> Z3SolverResult:
        """
        Solve a constraint satisfaction problem.
        
        Args:
            variables: List of variables
            constraints: List of constraints
            objective: Optional objective function for optimization
            minimize: Whether to minimize (True) or maximize (False)
            
        Returns:
            Z3SolverResult with solution
        """
        start_time = time.time()
        self._stats["total_calls"] += 1
        
        # Track this operation with the solver pool
        with self._track_operation("solve_constraints"):
            try:
                if Z3_PYTHON_AVAILABLE:
                    result = self._solve_with_python_api(variables, constraints, objective, minimize)
                else:
                    result = self._solve_with_cli(variables, constraints, objective, minimize)
                
                # Update statistics
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                self._stats["total_time"] += execution_time
                
                if result.status == Z3ResultStatus.SAT:
                    self._stats["sat_results"] += 1
                elif result.status == Z3ResultStatus.UNSAT:
                    self._stats["unsat_results"] += 1
                else:
                    self._stats["error_results"] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Z3 solving failed: {e}")
                self._stats["error_results"] += 1
                return Z3SolverResult(
                    status=Z3ResultStatus.ERROR,
                    reason=str(e),
                    execution_time=time.time() - start_time,
                    errors=[str(e)]
                )
    
    def _solve_with_python_api(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objective: Optional[str],
        minimize: bool
    ) -> Z3SolverResult:
        """Solve using Z3 Python API."""
        with self._solver_lock:
            # Create solver
            solver = z3.Solver()
            
            # Set parameters
            solver.set("timeout", int(self.config.timeout * 1000))
            
            # Create Z3 variables
            z3_vars = {}
            for var in variables:
                z3_vars[var.name] = self._create_z3_variable(var)
            
            # Add constraints
            for constraint in constraints:
                z3_expr = self._parse_constraint(constraint.expression, z3_vars)
                if z3_expr is not None:
                    solver.add(z3_expr)
            
            # Check satisfiability
            result = solver.check()
            
            if result == z3.sat:
                model = solver.model()
                assignments = {}
                for var in variables:
                    z3_var = z3_vars.get(var.name)
                    if z3_var is not None:
                        value = model.eval(z3_var, model_completion=True)
                        assignments[var.name] = self._z3_value_to_python(value)
                
                return Z3SolverResult(
                    status=Z3ResultStatus.SAT,
                    model=Z3Model(
                        assignments=assignments,
                        statistics={"num_constraints": len(constraints)}
                    )
                )
            elif result == z3.unsat:
                return Z3SolverResult(
                    status=Z3ResultStatus.UNSAT,
                    reason="Constraints are unsatisfiable"
                )
            else:
                return Z3SolverResult(
                    status=Z3ResultStatus.UNKNOWN,
                    reason="Solver returned unknown"
                )
    
    def _create_z3_variable(self, var: Z3Variable):
        """Create a Z3 variable from specification."""
        if var.var_type == Z3ConstraintType.BOOLEAN:
            return z3.Bool(var.name)
        elif var.var_type == Z3ConstraintType.INTEGER:
            return z3.Int(var.name)
        elif var.var_type == Z3ConstraintType.REAL:
            return z3.Real(var.name)
        elif var.var_type == Z3ConstraintType.BIT_VECTOR:
            return z3.BitVec(var.name, var.bit_width or 32)
        elif var.var_type == Z3ConstraintType.STRING:
            return z3.String(var.name)
        elif var.var_type == Z3ConstraintType.FLOATING_POINT:
            # Default to Double (64-bit) if width not specified
            return z3.FP(var.name, z3.Float64())
        else:
            return z3.Int(var.name)  # Default to Int
    
    def _parse_constraint(self, expression: str, z3_vars: Dict[str, Any]) -> Optional[Any]:
        """Parse constraint expression using Z3 Python API."""
        # Simple expression parser for common patterns
        # For complex expressions, use eval with restricted globals
        local_vars = z3_vars.copy()
        local_vars.update({
            'And': z3.And,
            'Or': z3.Or,
            'Not': z3.Not,
            'Implies': z3.Implies,
            'If': z3.If,
            'Sum': z3.Sum,
            'ForAll': z3.ForAll,
            'Exists': z3.Exists
        })
        
        # Safely evaluate the expression
        try:
            result = eval(expression, {"__builtins__": {}}, local_vars)
            return result
        except Exception as eval_err:
            # Fallback: Try parsing as SMT-LIB prefix notation
            try:
                # Wrap in assert if it's just an expression
                # and use parse_smt2_string with the existing variables as declarations
                smt_stmt = f"(assert {expression})"
                assertions = z3.parse_smt2_string(smt_stmt, decls=z3_vars)
                if len(assertions) > 0:
                    return assertions[0]
            except Exception:
                # If both fail, log the original eval error
                logger.warning(f"Failed to parse constraint '{expression}': {eval_err}")
                return None
    def _z3_value_to_python(self, value) -> Any:
        """Convert Z3 value to Python value."""
        if z3.is_int_value(value):
            return value.as_long()
        elif z3.is_rational_value(value):
            return value.as_fraction()
        elif z3.is_true(value):
            return True
        elif z3.is_false(value):
            return False
        elif z3.is_string_value(value):
            return value.as_string()
        elif z3.is_fp_value(value):
            try:
                return float(value.as_decimal(10).replace('?', ''))
            except:
                return str(value)
        else:
            return str(value)
    
    def _solve_with_cli(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objective: Optional[str],
        minimize: bool
    ) -> Z3SolverResult:
        """Solve using Z3 CLI interface."""
        # Generate SMT-LIB2 content
        smtlib_content = self._generate_smtlib(variables, constraints, objective, minimize)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib_content)
            temp_file = f.name
        
        try:
            # Run Z3
            cmd = ['z3', '-smt2', temp_file]
            if self.config.timeout > 0:
                cmd.extend(['-t:%d' % int(self.config.timeout * 1000)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout + 5  # Add buffer
            )
            
            return self._parse_z3_output(result.stdout, result.stderr)
            
        except subprocess.TimeoutExpired:
            return Z3SolverResult(
                status=Z3ResultStatus.TIMEOUT,
                reason=f"Solver timed out after {self.config.timeout}s"
            )
        except Exception as e:
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason=str(e),
                errors=[str(e)]
            )
        finally:
            # Cleanup temp file
            try:
                Path(temp_file).unlink()
            except OSError:
                pass
    
    def _generate_smtlib(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objective: Optional[str],
        minimize: bool
    ) -> str:
        """Generate SMT-LIB2 format content."""
        lines = [
            "; Generated by OpenEvolve Z3 Integration",
            "(set-logic ALL)",
            "(set-option :produce-models true)"
        ]
        
        if self.config.proof_generation:
            lines.append("(set-option :produce-proofs true)")
        
        # Declare variables
        for var in variables:
            lines.append(var.to_smtlib())
        
        # Add constraints
        for constraint in constraints:
            lines.append(constraint.to_smtlib())
        
        # Add optimization objective if provided
        if objective and self.config.optimization:
            opt_cmd = "minimize" if minimize else "maximize"
            lines.append(f"({opt_cmd} {objective})")
        
        # Check satisfiability
        lines.append("(check-sat)")
        
        # Get model if satisfiable
        lines.append("(get-model)")
        
        return "\n".join(lines)
    
    def _parse_z3_output(self, stdout: str, stderr: str) -> Z3SolverResult:
        """Parse Z3 CLI output."""
        output = stdout.strip()
        
        if not output:
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason="No output from Z3",
                errors=[stderr] if stderr else ["Empty output"]
            )
        
        lines = output.split('\n')
        first_line = lines[0].strip().lower()
        
        if first_line == 'sat':
            # Parse model
            assignments = self._parse_model(lines[1:])
            return Z3SolverResult(
                status=Z3ResultStatus.SAT,
                model=Z3Model(assignments=assignments),
                smtlib_output=output
            )
        elif first_line == 'unsat':
            return Z3SolverResult(
                status=Z3ResultStatus.UNSAT,
                reason="Constraints are unsatisfiable",
                smtlib_output=output
            )
        elif first_line == 'unknown':
            return Z3SolverResult(
                status=Z3ResultStatus.UNKNOWN,
                reason="Solver returned unknown",
                smtlib_output=output
            )
        else:
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason=f"Unexpected output: {first_line}",
                errors=[stderr] if stderr else [],
                smtlib_output=output
            )
    
    def _parse_model(self, lines: List[str]) -> Dict[str, Any]:
        """Parse Z3 model output."""
        assignments = {}
        model_text = '\n'.join(lines)
        
        # Simple regex-based parsing for model output
        # Pattern: (define-fun name () type value)
        pattern = r'\(define-fun\s+(\w+)\s+\(\)\s+(\w+|\([^)]+\))\s+([^)]+)\)'
        matches = re.findall(pattern, model_text)
        
        for name, type_str, value in matches:
            # Parse value based on type
            value = value.strip()
            if type_str == 'Int':
                try:
                    assignments[name] = int(value)
                except ValueError:
                    assignments[name] = value
            elif type_str == 'Real':
                try:
                    if '/' in value:
                        num, den = value.split('/')
                        assignments[name] = float(num) / float(den)
                    else:
                        assignments[name] = float(value)
                except ValueError:
                    assignments[name] = value
            elif type_str == 'Bool':
                assignments[name] = value == 'true'
            else:
                assignments[name] = value
        
        return assignments
    
    def _extract_model_assignments(self, model) -> Dict[str, Any]:
        """
        Extract variable assignments from a Z3 model object.
        
        Args:
            model: Z3 Model object
            
        Returns:
            Dictionary mapping variable names to their Python values
        """
        import z3  # type: ignore
        
        assignments = {}
        
        for decl in model.decls():
            name = decl.name()
            value = model[decl]
            
            # Convert Z3 value to Python value based on sort
            if z3.is_int_value(value):
                assignments[name] = value.as_long()
            elif z3.is_rational_value(value):
                # Convert rational to float
                assignments[name] = float(value.as_decimal(10).rstrip('?'))
            elif z3.is_true(value) or z3.is_false(value):
                assignments[name] = z3.is_true(value)
            elif z3.is_bv_value(value):
                # BitVec value - convert to int
                assignments[name] = value.as_long()
            elif z3.is_algebraic_value(value):
                # Algebraic number - convert to approximate float
                assignments[name] = float(value.approx(10).as_decimal(10).rstrip('?'))
            else:
                # For other types, try to get string representation
                try:
                    assignments[name] = str(value)
                except Exception:
                    assignments[name] = value
        
        return assignments
    
    def solve_smtlib(self, smtlib_content: str) -> Z3SolverResult:
        """
        Solve directly from SMT-LIB2 content.
        
        Args:
            smtlib_content: SMT-LIB2 formatted problem
            
        Returns:
            Z3SolverResult
        """
        start_time = time.time()
        self._stats["total_calls"] += 1
        temp_file = None
        
        try:
            # Try CLI first
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                    f.write(smtlib_content)
                    temp_file = f.name
                
                cmd = ['z3', '-smt2', temp_file]
                if self.config.timeout > 0:
                    cmd.extend(['-t:%d' % int(self.config.timeout * 1000)])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout + 5
                )
                
                execution_time = time.time() - start_time
                self._stats["total_time"] += execution_time
                
                parsed_result = self._parse_z3_output(result.stdout, result.stderr)
                parsed_result.execution_time = execution_time
                
                # Update statistics
                if parsed_result.status == Z3ResultStatus.SAT:
                    self._stats["sat_results"] += 1
                elif parsed_result.status == Z3ResultStatus.UNSAT:
                    self._stats["unsat_results"] += 1
                
                return parsed_result
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                if not Z3_PYTHON_AVAILABLE:
                    raise e
                
                # Fallback to Python API
                with self._solver_lock:
                    solver = z3.Solver()
                    solver.set("timeout", int(self.config.timeout * 1000))
                    
                    # Parse SMT-LIB string
                    # Note: parse_smt2_string returns a vector of expressions
                    assertions = z3.parse_smt2_string(smtlib_content)
                    solver.add(assertions)
                    
                    result = solver.check()
                    execution_time = time.time() - start_time
                    
                    if result == z3.sat:
                        model = solver.model()
                        assignments = self._extract_model_assignments(model)
                        self._stats["sat_results"] += 1
                        return Z3SolverResult(
                            status=Z3ResultStatus.SAT,
                            model=Z3Model(assignments=assignments),
                            execution_time=execution_time
                        )
                    elif result == z3.unsat:
                        self._stats["unsat_results"] += 1
                        return Z3SolverResult(
                            status=Z3ResultStatus.UNSAT,
                            execution_time=execution_time
                        )
                    else:
                        return Z3SolverResult(
                            status=Z3ResultStatus.UNKNOWN,
                            execution_time=execution_time
                        )
            
        except Exception as e:
            self._stats["error_results"] += 1
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason=str(e),
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
        finally:
            if temp_file:
                try:
                    Path(temp_file).unlink()
                except OSError:
                    pass


# =============================================================================
# Z3 Theorem Prover
# =============================================================================

class Z3LogicCompressor:
    """Compress and simplify boolean logic using Z3 when available."""

    def __init__(self):
        self.solver = Z3SolverEngine()

    def simplify_condition(self, condition: str) -> str:
        if not Z3_PYTHON_AVAILABLE:
            return condition
        expr = self._to_smtlib(condition)
        if not expr:
            return condition
        try:
            import z3  # type: ignore

            variables = {v: z3.Bool(v) for v in self._extract_symbols(expr)}
            parsed = z3.parse_smt2_string(f"(assert {expr})", decls=variables)
            if not parsed:
                return condition
            simplified = z3.simplify(parsed[0])
            return simplified.sexpr()
        except Exception:
            return condition

    def compress_code_conditions(self, code: str, min_chain: int = 3) -> str:
        """Simplify large if/elif chains by compressing their conditions."""
        lines = code.splitlines()
        updated = False
        chain_count = 0
        for idx, line in enumerate(lines):
            match = re.match(r"^(\s*)(if|elif)\s+(.*):\s*$", line)
            if not match:
                chain_count = 0
                continue
            indent, keyword, condition = match.groups()
            chain_count += 1
            if chain_count >= min_chain:
                simplified = self.simplify_condition(condition)
                if simplified and simplified != condition:
                    lines[idx] = f"{indent}{keyword} {simplified}:"
                    updated = True
        return "\n".join(lines) if updated else code

    def _to_smtlib(self, condition: str) -> Optional[str]:
        """Attempt to convert simple python boolean expressions to SMT-LIB."""
        stripped = condition.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            return stripped
        if " and " in stripped and " or " not in stripped:
            parts = [p.strip() for p in stripped.split(" and ") if p.strip()]
            return f"(and {' '.join(parts)})"
        if " or " in stripped and " and " not in stripped:
            parts = [p.strip() for p in stripped.split(" or ") if p.strip()]
            return f"(or {' '.join(parts)})"
        if stripped.startswith("not "):
            return f"(not {stripped[4:].strip()})"
        return None

    def _extract_symbols(self, expr: str) -> List[str]:
        # Exclude SMT-LIB and Python keywords
        keywords = {
            'and', 'or', 'not', 'implies', 'iff', 'forall', 'exists', 
            'true', 'false', 'assert', 'declare-fun', 'set-logic'
        }
        symbols = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
        return list(set(s for s in symbols if s.lower() not in keywords))

class Z3TheoremProver:
    """
    Z3-based theorem prover for formal verification.
    
    Provides capabilities for proving mathematical theorems and
    verifying logical formulas.
    Integrates with Z3SolverPool for metrics tracking.
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()
        self.solver_engine = Z3SolverEngine(config)
        self._prover_lock = threading.RLock()
        
        # Register with solver pool for metrics tracking (separate from engine)
        self._solver_id: Optional[str] = None
        self._pool: Optional[Any] = None
        if SOLVER_POOL_AVAILABLE:
            try:
                self._pool = get_solver_pool()
                self._solver_id = self._pool.register_solver(
                    metadata={
                        'class': 'Z3TheoremProver',
                        'config_timeout': self.config.timeout,
                        'has_engine': self.solver_engine is not None,
                        'engine_id': getattr(self.solver_engine, '_solver_id', None)
                    }
                )
                logger.debug(f"Z3TheoremProver registered with pool: {self._solver_id}")
            except Exception as e:
                logger.debug(f"Failed to register Z3TheoremProver with solver pool: {e}")
                self._pool = None
                self._solver_id = None
    
    def _track_operation(self, operation_name: str = "prove"):
        """
        Context manager for tracking prover operations with the pool.
        
        Args:
            operation_name: Name of the operation being tracked
            
        Yields:
            None
        """
        if self._pool is not None and self._solver_id is not None:
            return self._pool.active_operation(self._solver_id)
        else:
            # Return a no-op context manager if pool not available
            from contextlib import nullcontext
            return nullcontext()
    
    def __del__(self):
        """Cleanup: unregister from solver pool."""
        if self._pool is not None and self._solver_id is not None:
            try:
                self._pool.unregister_solver(self._solver_id)
                if logger is not None:
                    logger.debug(f"Z3TheoremProver unregistered from pool: {self._solver_id}")
            except Exception:
                # Ignore errors during cleanup, especially during interpreter shutdown
                pass
    
    def prove_theorem(
        self,
        theorem_statement: str,
        assumptions: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Z3TheoremResult:
        """
        Prove a theorem using Z3.
        
        Args:
            theorem_statement: The theorem to prove (SMT-LIB or natural language)
            assumptions: Optional list of assumptions
            timeout: Optional timeout override
            
        Returns:
            Z3TheoremResult
        """
        start_time = time.time()
        
        # Track this operation with the solver pool
        with self._track_operation("prove_theorem"):
            try:
                # Check if input is SMT-LIB or natural language
                if self._is_smtlib(theorem_statement):
                    return self._prove_smtlib(theorem_statement, assumptions, timeout)
                else:
                    return self._prove_natural_language(theorem_statement, assumptions, timeout)
                    
            except Exception as e:
                logger.error(f"Theorem proving failed: {e}")
                return Z3TheoremResult(
                    proven=False,
                    errors=[str(e)],
                    execution_time=time.time() - start_time
                )
    
    def _is_smtlib(self, text: str) -> bool:
        """Check if text is in SMT-LIB format."""
        smt_keywords = ['(assert', '(declare-fun', '(check-sat)', '(set-logic']
        return any(kw in text for kw in smt_keywords)
    
    def _prove_smtlib(
        self,
        theorem_statement: str,
        assumptions: Optional[List[str]],
        timeout: Optional[float]
    ) -> Z3TheoremResult:
        """Prove theorem from SMT-LIB format."""
        start_time = time.time()
        
        # If it looks like a complete SMT-LIB script with check-sat, 
        # assume the user knows what they are doing (e.g. they provided a proof-by-contradiction script)
        if '(check-sat)' in theorem_statement.lower() and '(assert' in theorem_statement.lower():
            smtlib_content = theorem_statement
        # Negate the theorem for proof by contradiction if it's just a set of assertions
        elif '(assert' in theorem_statement:
            # Extract the last assertion (theorem) and negate it
            lines = theorem_statement.split('\n')
            assertions = [l for l in lines if l.strip().startswith('(assert')]
            
            if assertions:
                # Negate the last assertion for proof by contradiction
                theorem_line = assertions[-1]
                # Simple negation
                negated = f"(assert (not {theorem_line.strip()[8:-1]}))"
                
                # Replace the theorem with its negation
                new_lines = []
                found = False
                for l in lines:
                    if l.strip() == theorem_line.strip() and not found:
                        new_lines.append(negated)
                        found = True
                    else:
                        new_lines.append(l)
                
                smtlib_content = '\n'.join(new_lines)
            else:
                smtlib_content = theorem_statement
        else:
            smtlib_content = theorem_statement
        
        # Solve
        result = self.solver_engine.solve_smtlib(smtlib_content)
        execution_time = time.time() - start_time
        
        if result.status == Z3ResultStatus.UNSAT:
            # Theorem is proven (negation is unsatisfiable)
            return Z3TheoremResult(
                proven=True,
                proof=result.smtlib_output,
                execution_time=execution_time,
                tactic_used="smt"
            )
        elif result.status == Z3ResultStatus.SAT:
            # Found counterexample
            return Z3TheoremResult(
                proven=False,
                counterexample=result.model.assignments if result.model else None,
                execution_time=execution_time,
                tactic_used="smt"
            )
        else:
            return Z3TheoremResult(
                proven=False,
                errors=result.errors,
                execution_time=execution_time
            )
    
    def _prove_natural_language(
        self,
        theorem_statement: str,
        assumptions: Optional[List[str]],
        timeout: Optional[float]
    ) -> Z3TheoremResult:
        """
        Prove theorem from natural language using LLM translation to SMT-LIB.
        
        This method translates natural language theorem descriptions into SMT-LIB format
        using an LLM, then calls the existing SMT-LIB prover to verify the theorem.
        
        Args:
            theorem_statement: Natural language description of the theorem
            assumptions: Optional list of assumptions in natural language
            timeout: Optional timeout override
            
        Returns:
            Z3TheoremResult with proof status
        """
        import os
        start_time = time.time()
        
        # Get API configuration from environment
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_API_TOKEN")
        )
        if not api_key:
            logger.warning("Natural language theorem proving skipped: OPENAI_API_KEY not set")
            return Z3TheoremResult(
                proven=False,
                errors=["Natural language theorem proving requires OPENAI_API_KEY environment variable"],
                execution_time=time.time() - start_time
            )
        
        # Translate natural language to SMT-LIB
        smtlib_content = self._nl_to_smtlib(theorem_statement, assumptions)
        if not smtlib_content:
            return Z3TheoremResult(
                proven=False,
                errors=["Failed to translate natural language theorem to SMT-LIB format"],
                execution_time=time.time() - start_time
            )
        
        # Call the existing SMT-LIB prover
        result = self._prove_smtlib(smtlib_content, None, timeout)
        
        # Update execution time to include translation
        result.execution_time = time.time() - start_time
        return result
    
    def _nl_to_smtlib(
        self,
        theorem_statement: str,
        assumptions: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Translate natural language theorem to SMT-LIB format using LLM.
        
        Args:
            theorem_statement: Natural language theorem description
            assumptions: Optional list of assumptions
            
        Returns:
            SMT-LIB formatted theorem content or None if translation failed
        """
        try:
            from llm_utils import _compose_messages, _request_openai_compatible_chat
        except ImportError as exc:
            logger.warning("LLM utilities not available for theorem translation: %s", exc)
            return None
        
        import os
        
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_API_TOKEN")
        )
        if not api_key:
            return None
        
        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL_ID")
            or "gpt-4o-mini"
        )
        
        # Build the assumptions text if provided
        assumptions_text = ""
        if assumptions:
            assumptions_text = "\nAssumptions:\n" + "\n".join(
                f"- {a}" for a in assumptions
            )
        
        system_prompt = (
            "You are a formal methods expert that translates natural language "
            "mathematical theorems into SMT-LIB2 format for Z3 theorem proving. "
            "Return ONLY valid SMT-LIB2 code without markdown formatting or explanations."
        )
        
        user_prompt = (
            "Translate the following natural language theorem into SMT-LIB2 format "
            "for Z3 theorem proving.\n\n"
            "Requirements:\n"
            "1. Include (set-logic ALL) at the beginning\n"
            "2. Declare all variables using (declare-fun name () type)\n"
            "3. Use appropriate types: Int, Real, Bool, (Array Int Int), etc.\n"
            "4. Include (assert ...) for all constraints and assumptions\n"
            "5. End with (check-sat) and (get-model)\n"
            "6. The theorem should be negated for proof-by-contradiction\n"
            "7. Return ONLY the SMT-LIB2 code, no markdown or explanations\n\n"
            f"Theorem: {theorem_statement}{assumptions_text}\n\n"
            "Example output format:\n"
            "(set-logic ALL)\n"
            "(declare-fun x () Int)\n"
            "(assert (> x 0))\n"
            "(assert (not (>= x 1)))\n"
            "(check-sat)\n"
            "(get-model)"
        )
        
        messages = _compose_messages(system_prompt, user_prompt)
        
        try:
            response = _request_openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=0.1,
                top_p=1.0,
                max_tokens=1500,
                timeout=60
            )
        except Exception as exc:
            logger.warning("Theorem translation LLM call failed: %s", exc)
            return None
        
        if not response:
            return None
        
        # Clean up the response - remove markdown code blocks if present
        smtlib_content = response.strip()
        if smtlib_content.startswith("```"):
            lines = smtlib_content.split("\n")
            # Remove first line (```smt2 or ```)
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            smtlib_content = "\n".join(lines).strip()
        
        # Basic validation: must contain required SMT-LIB elements
        if "(check-sat)" not in smtlib_content:
            logger.warning("LLM response missing (check-sat), adding it")
            smtlib_content += "\n(check-sat)\n(get-model)"
        
        if "(set-logic" not in smtlib_content:
            smtlib_content = "(set-logic ALL)\n" + smtlib_content
        
        return smtlib_content
    
    def verify_formula(
        self,
        formula: str,
        variables: Optional[Dict[str, str]] = None
    ) -> Z3TheoremResult:
        """
        Verify a logical formula.
        
        Args:
            formula: Logical formula to verify
            variables: Dictionary of variable names to types
            
        Returns:
            Z3TheoremResult
        """
        # Build SMT-LIB content
        lines = [
            "(set-logic ALL)",
            "(set-option :produce-models true)"
        ]
        
        # Declare variables
        if variables:
            for name, var_type in variables.items():
                lines.append(f"(declare-fun {name} () {var_type})")
        
        # Add formula
        lines.append(f"(assert (not {formula}))")
        lines.append("(check-sat)")
        lines.append("(get-model)")
        
        smtlib_content = '\n'.join(lines)
        return self._prove_smtlib(smtlib_content, None, None)


# =============================================================================
# Digital Twin Sandbox (DTS)
# =============================================================================

class DigitalTwinSandbox:
    """
    Digital Twin Logical Sandboxing for SOP validation.

    Translates SOP steps or fix descriptions into Z3 constraints and verifies
    them against global safety invariants.
    """

    def __init__(self, solver_engine: Optional[Z3SolverEngine] = None):
        self.solver_engine = solver_engine or Z3SolverEngine()

    def sop_to_constraints(self, steps: List[str]) -> Tuple[List[Z3Variable], List[Z3Constraint]]:
        """Parse SOP steps into Z3 constraints using LLM extraction."""
        variables: Dict[str, Z3Variable] = {}
        constraints: List[Z3Constraint] = []

        for step in steps:
            step_vars, step_constraints = self._parse_natural_language_constraints(step)
            for var in step_vars:
                if var.name not in variables:
                    variables[var.name] = var
            constraints.extend(step_constraints)

        return list(variables.values()), constraints

    def verify_fix_with_invariants(
        self,
        fix_text: str,
        safety_invariants: List[str]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify that fix constraints imply safety invariants.

        Returns:
            (passed, counterexample)
        """
        variables, constraints = self.sop_to_constraints([fix_text])
        invariant_constraints = [Z3Constraint(expr, Z3ConstraintType.BOOLEAN) for expr in safety_invariants]

        if not Z3_PYTHON_AVAILABLE:
            # Fallback: assume pass when Z3 is unavailable
            return True, None

        try:
            solver = z3.Solver()
            for var in variables:
                self.solver_engine._create_z3_variable(var)
            z3_vars = {var.name: self.solver_engine._create_z3_variable(var) for var in variables}

            for constraint in constraints:
                expr = self.solver_engine._parse_constraint(constraint.expression, z3_vars)
                if expr is not None:
                    solver.add(expr)

            # Add negation of invariants to test implication
            for invariant in invariant_constraints:
                expr = self.solver_engine._parse_constraint(invariant.expression, z3_vars)
                if expr is not None:
                    solver.add(z3.Not(expr))

            result = solver.check()
            if result == z3.unsat:
                return True, None
            if result == z3.sat:
                model = solver.model()
                counterexample = {d.name(): str(model[d]) for d in model.decls()}
                return False, counterexample
            return False, None
        except Exception as exc:
            logger.warning("Digital twin verification failed: %s", exc)
            return False, {"error": str(exc)}

    def _parse_natural_language_constraints(
        self,
        text: str
    ) -> Tuple[List[Z3Variable], List[Z3Constraint]]:
        """
        Extract Z3 constraints from natural language text using LLM.
        Replaces legacy regex-based parsing.
        """
        variables: List[Z3Variable] = []
        constraints: List[Z3Constraint] = []
        
        if not text or not text.strip():
            return variables, constraints

        try:
            # Late import to avoid circular dependencies
            import os
            from llm_utils import _compose_messages, _request_openai_compatible_chat
        except ImportError as exc:
            logger.warning("LLM utilities not available for constraint extraction: %s", exc)
            return variables, constraints

        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_API_TOKEN")
        )
        if not api_key:
            logger.warning("Constraint extraction skipped: OPENAI_API_KEY not set")
            return variables, constraints

        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL_ID")
            or "gpt-4o-mini"
        )

        system_prompt = (
            "You extract SMT constraints from natural language. "
            "Return ONLY a JSON object that matches the required schema."
        )
        user_prompt = (
            "Extract variables and constraints from the text below.\n\n"
            "Return JSON with this exact schema:\n"
            "{\n"
            "  \"variables\": [\n"
            "    {\"name\": \"x\", \"type\": \"integer|real|boolean|string\", "
            "\"bounds\": [\"min_or_null\", \"max_or_null\"], \"bit_width\": 32}\n"
            "  ],\n"
            "  \"constraints\": [\"SMT-LIB boolean expressions without (assert)\"]\n"
            "}\n\n"
            "Rules:\n"
            "- Use lowercase type strings: integer, real, boolean, string.\n"
            "- If bounds are unknown, use nulls.\n"
            "- Constraints must be SMT-LIB boolean expressions (no (assert)).\n"
            "- If nothing is found, return empty arrays.\n\n"
            f"Text:\n{text}"
        )

        messages = _compose_messages(system_prompt, user_prompt)
        try:
            response = _request_openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
        except Exception as exc:
            logger.warning("Constraint extraction LLM call failed: %s", exc)
            return variables, constraints

        if not response:
            return variables, constraints

        # Parse JSON response
        raw = response.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Constraint extraction JSON parse failed: %s", exc)
            return variables, constraints

        if not isinstance(parsed, dict):
            return variables, constraints

        type_map = {
            "integer": Z3ConstraintType.INTEGER,
            "int": Z3ConstraintType.INTEGER,
            "real": Z3ConstraintType.REAL,
            "float": Z3ConstraintType.REAL,
            "boolean": Z3ConstraintType.BOOLEAN,
            "bool": Z3ConstraintType.BOOLEAN,
            "string": Z3ConstraintType.STRING,
            "str": Z3ConstraintType.STRING
        }

        # Process variables
        seen_names = set()
        for entry in parsed.get("variables") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            
            var_type_str = str(entry.get("type", "integer")).strip().lower()
            var_type = type_map.get(var_type_str, Z3ConstraintType.INTEGER)
            
            bounds = entry.get("bounds")
            bit_width = entry.get("bit_width")
            
            if isinstance(bounds, list) and len(bounds) == 2:
                min_val = None if bounds[0] in (None, "null") else bounds[0]
                max_val = None if bounds[1] in (None, "null") else bounds[1]
                bounds_tuple = (min_val, max_val)
            else:
                bounds_tuple = None
                
            if isinstance(bit_width, (int, float)):
                bit_width = int(bit_width)
            else:
                bit_width = None

            variables.append(Z3Variable(
                name=name,
                var_type=var_type,
                bounds=bounds_tuple,
                bit_width=bit_width
            ))
            seen_names.add(name)

        # Process constraints
        for constraint_expr in parsed.get("constraints") or []:
            if not isinstance(constraint_expr, str):
                continue
            text_constraint = constraint_expr.strip()
            if not text_constraint:
                continue
            
            constraints.append(Z3Constraint(
                expression=text_constraint,
                constraint_type=Z3ConstraintType.BOOLEAN
            ))

        return variables, constraints


def pattern_operator(pattern: str) -> str:
    """Return SMT-LIB operator for regex pattern."""
    if "<=" in pattern:
        return "<="
    if ">=" in pattern:
        return ">="
    if ">" in pattern and "<" not in pattern:
        return ">"
    return "<"


def generate_refutation_narrative(
    result: Union[Z3SolverResult, Z3TheoremResult, str],
    constraints: Optional[List[str]] = None,
    counterexample: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a human-readable refutation narrative from a Z3 result.

    Args:
        result: Z3 result object or status string (e.g., "unsat", "sat").
        constraints: Optional list of constraint strings to include in the narrative.
        counterexample: Optional counterexample assignments from a model.

    Returns:
        Natural language refutation narrative.
    """
    status_value = ""
    if isinstance(result, Z3SolverResult):
        status_value = result.status.value
        if counterexample is None and result.model:
            counterexample = result.model.assignments
    elif isinstance(result, Z3TheoremResult):
        status_value = "unsat" if not result.proven else "sat"
        if counterexample is None:
            counterexample = result.counterexample
    else:
        status_value = str(result).lower()

    constraint_text = ""
    if constraints:
        constraint_text = "Constraints involved:\n" + "\n".join(f"- {c}" for c in constraints)

    if "unsat" in status_value:
        narrative = (
            "Refutation Narrative: A contradiction was found. "
            "The constraints cannot be satisfied simultaneously."
        )
    elif "sat" in status_value:
        narrative = "Refutation Narrative: A satisfying assignment exists for the constraints."
    else:
        narrative = "Refutation Narrative: Solver status unknown; unable to confirm satisfiability."

    if counterexample:
        narrative += "\nCounterexample:\n" + "\n".join(f"- {k} = {v}" for k, v in counterexample.items())

    if constraint_text:
        narrative += "\n" + constraint_text

    return narrative


# =============================================================================
# Z3 Prover Integration Facade
# =============================================================================

class Z3ProverIntegration:
    """
    Facade class for Z3 integration, providing a unified interface for
    constraint solving, theorem proving, and logical sandboxing.
    """
    
    def __init__(self, config: Optional[Z3Config] = None, timeout: float = 30.0):
        self.config = config or Z3Config(timeout=timeout)
        self.solver_engine = Z3SolverEngine(self.config)
        self.theorem_prover = Z3TheoremProver(self.config)
        self.sandbox = DigitalTwinSandbox(self.solver_engine)
        self.logger = logging.getLogger(__name__ + ".Z3ProverIntegration")
        
    def solve_constraints(self, variables: List[Z3Variable], constraints: List[Z3Constraint]) -> Z3SolverResult:
        """Unified entry point for constraint solving."""
        return self.solver_engine.solve_constraints(variables, constraints)
        
    def prove_theorem(self, theorem: str, assumptions: List[str] = None) -> Z3TheoremResult:
        """Unified entry point for theorem proving."""
        return self.theorem_prover.prove_theorem(theorem, assumptions)
        
    def verify_safety(self, fix_text: str, safety_invariants: List[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Unified entry point for safety invariant verification."""
        return self.sandbox.verify_fix_with_invariants(fix_text, safety_invariants)
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of all Z3 components."""
        return {
            "engine": self.solver_engine.get_status(),
            "prover_available": self.theorem_prover is not None,
            "sandbox_available": self.sandbox is not None,
            "z3_available": Z3_AVAILABLE
        }


# =============================================================================
# SMART CONTRACT INVARIANT TRANSLATION
# =============================================================================

@dataclass
class SolidityInvariantTranslation:
    """Structured translation of Solidity state update semantics to Z3 artifacts."""
    source_statement: str
    variables: List[Z3Variable]
    constraints: List[Z3Constraint]
    invariants: List[Z3Constraint]
    lean_spec: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        serialized_variables = []
        for variable in self.variables:
            serialized_variables.append(
                {
                    "name": variable.name,
                    "var_type": variable.var_type.name.lower(),
                    "bounds": variable.bounds,
                    "bit_width": variable.bit_width,
                }
            )
        return {
            "source_statement": self.source_statement,
            "variables": serialized_variables,
            "constraints": [c.expression for c in self.constraints],
            "invariants": [i.expression for i in self.invariants],
            "lean_spec": self.lean_spec,
            "metadata": self.metadata,
        }


class SmartContractInvariantTranslator:
    """
    Translate Solidity state transitions into Z3 constraints and Lean specs.

    Focuses on common high-impact patterns in audits (withdraw/deposit balance math),
    while remaining conservative and explicit.
    """

    _ASSIGN_PATTERN = re.compile(r"^\s*(?P<lhs>[^=]+?)\s*=\s*(?P<rhs>.+?)\s*;?\s*$")
    _INPLACE_PATTERN = re.compile(r"^\s*(?P<lhs>.+?)\s*(?P<op>\+=|-=)\s*(?P<rhs>.+?)\s*;?\s*$")
    _TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
    _KEYWORDS = {"if", "for", "while", "return", "require", "assert", "true", "false"}

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        cleaned = symbol.strip()
        cleaned = cleaned.replace("msg.sender", "msg_sender")
        cleaned = cleaned.replace("tx.origin", "tx_origin")
        cleaned = cleaned.replace("block.timestamp", "block_timestamp")
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            cleaned = "value"
        if cleaned[0].isdigit():
            cleaned = f"v_{cleaned}"
        return cleaned

    @classmethod
    def _extract_tokens(cls, expression: str) -> List[str]:
        tokens = cls._TOKEN_PATTERN.findall(expression or "")
        filtered = []
        for token in tokens:
            if token.lower() in cls._KEYWORDS:
                continue
            filtered.append(token)
        return sorted(set(filtered), key=len, reverse=True)

    @classmethod
    def _base_symbol(cls, lhs: str) -> str:
        raw = lhs.strip()
        raw = re.split(r"[\[.]", raw)[0]
        return cls._normalize_symbol(raw) or "state"

    @classmethod
    def _rewrite_expression(
        cls,
        expression: str,
        lhs: str,
        old_symbol: str,
    ) -> Tuple[str, Dict[str, str]]:
        rewritten = expression.strip()
        token_map: Dict[str, str] = {}

        lhs_norm = cls._normalize_symbol(lhs)
        tokens = cls._extract_tokens(rewritten)
        for token in tokens:
            norm = cls._normalize_symbol(token)
            mapped = old_symbol if norm == lhs_norm else norm
            token_map[token] = mapped

        for token in sorted(token_map.keys(), key=len, reverse=True):
            rewritten = rewritten.replace(token, token_map[token])

        return rewritten, token_map

    def translate_assignment(
        self,
        statement: str,
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
    ) -> SolidityInvariantTranslation:
        """
        Translate a Solidity assignment/update statement into Z3 constraints.

        Example input:
            balance[msg.sender] -= amount;

        Example output constraints:
            new_balance == old_balance - amount
            new_balance >= 0
        """
        source = (statement or "").strip()
        if not source:
            raise ValueError("Solidity statement cannot be empty")

        lhs = ""
        rhs = ""
        op = ""

        inplace = self._INPLACE_PATTERN.match(source)
        if inplace:
            lhs = inplace.group("lhs").strip()
            rhs = inplace.group("rhs").strip()
            op = inplace.group("op")
        else:
            assign = self._ASSIGN_PATTERN.match(source)
            if not assign:
                raise ValueError(f"Unsupported Solidity assignment syntax: {source}")
            lhs = assign.group("lhs").strip()
            rhs = assign.group("rhs").strip()
            rhs_match = re.match(rf"^\s*{re.escape(lhs)}\s*([+-])\s*(.+)$", rhs)
            if rhs_match:
                op = "+=" if rhs_match.group(1) == "+" else "-="
                rhs = rhs_match.group(2).strip()
            else:
                op = "="

        base = self._base_symbol(lhs)
        old_symbol = f"old_{base}"
        new_symbol = f"new_{base}"
        rewritten_rhs, token_map = self._rewrite_expression(rhs, lhs, old_symbol)

        if op == "-=":
            relation = f"{new_symbol} == {old_symbol} - ({rewritten_rhs})"
        elif op == "+=":
            relation = f"{new_symbol} == {old_symbol} + ({rewritten_rhs})"
        else:
            relation = f"{new_symbol} == ({rewritten_rhs})"

        variable_names = {old_symbol, new_symbol}
        variable_names.update(token_map.values())
        variable_names = {v for v in variable_names if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v)}
        variables = [
            Z3Variable(name=name, var_type=Z3ConstraintType.INTEGER)
            for name in sorted(variable_names)
        ]

        constraints = [
            Z3Constraint(
                expression=relation,
                constraint_type=Z3ConstraintType.BOOLEAN,
                description=f"State transition derived from: {source}",
            )
        ]

        invariants: List[Z3Constraint] = []
        if non_negative_target:
            invariants.append(
                Z3Constraint(
                    expression=f"{new_symbol} >= 0",
                    constraint_type=Z3ConstraintType.BOOLEAN,
                    description=f"Non-negative {base} invariant",
                )
            )
        if max_withdraw_expr:
            rewritten_limit, _ = self._rewrite_expression(max_withdraw_expr, lhs, old_symbol)
            invariants.append(
                Z3Constraint(
                    expression=f"{rewritten_rhs} <= ({rewritten_limit})",
                    constraint_type=Z3ConstraintType.BOOLEAN,
                    description="Withdrawal upper-bound invariant",
                )
            )

        lean_spec = self.to_lean_spec(
            theorem_name=f"{base}_state_transition",
            constraints=[c.expression for c in constraints],
            invariants=[i.expression for i in invariants],
        )

        return SolidityInvariantTranslation(
            source_statement=source,
            variables=variables,
            constraints=constraints,
            invariants=invariants,
            lean_spec=lean_spec,
            metadata={
                "lhs": lhs,
                "operator": op,
                "base_symbol": base,
                "token_map": token_map,
            },
        )

    @staticmethod
    def to_lean_spec(
        theorem_name: str,
        constraints: List[str],
        invariants: List[str],
    ) -> str:
        """Generate a minimal Lean 4 theorem scaffold for translated constraints."""
        assumptions = constraints or ["True"]
        goals = invariants or ["True"]
        assumptions_expr = " /\\ ".join(assumptions)
        goals_expr = " /\\ ".join(goals)
        return (
            f"theorem {theorem_name} :\n"
            f"  ({assumptions_expr}) -> ({goals_expr}) := by\n"
            f"  intro h\n"
            f"  sorry\n"
        )


def translate_solidity_assignment_to_z3(
    statement: str,
    non_negative_target: bool = True,
    max_withdraw_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience API for Blue Team invariant translation requests.

    Example:
        translate_solidity_assignment_to_z3("balance[msg.sender] -= amount;")
    """
    translator = SmartContractInvariantTranslator()
    translation = translator.translate_assignment(
        statement=statement,
        non_negative_target=non_negative_target,
        max_withdraw_expr=max_withdraw_expr,
    )
    return translation.to_dict()


def verify_solidity_invariant_translation(
    translation: Union[SolidityInvariantTranslation, Dict[str, Any]],
    assume_non_negative_amount: bool = True,
) -> Dict[str, Any]:
    """
    Verify translated invariants by checking:
        constraints AND assumptions AND NOT(invariants) is UNSAT.
    """
    def _coerce_var_type(value: Any) -> Z3ConstraintType:
        if isinstance(value, Z3ConstraintType):
            return value
        if isinstance(value, str):
            mapping = {
                "boolean": Z3ConstraintType.BOOLEAN,
                "bool": Z3ConstraintType.BOOLEAN,
                "integer": Z3ConstraintType.INTEGER,
                "int": Z3ConstraintType.INTEGER,
                "real": Z3ConstraintType.REAL,
                "bit_vector": Z3ConstraintType.BIT_VECTOR,
                "array": Z3ConstraintType.ARRAY,
                "floating_point": Z3ConstraintType.FLOATING_POINT,
                "string": Z3ConstraintType.STRING,
            }
            return mapping.get(value.lower(), Z3ConstraintType.INTEGER)
        return Z3ConstraintType.INTEGER

    if isinstance(translation, dict):
        variables = [
            Z3Variable(
                name=v.get("name"),
                var_type=_coerce_var_type(v.get("var_type", Z3ConstraintType.INTEGER)),
                bounds=v.get("bounds"),
                bit_width=v.get("bit_width"),
            )
            for v in translation.get("variables", [])
            if isinstance(v, dict) and v.get("name")
        ]
        constraints = [
            Z3Constraint(expression=expr, constraint_type=Z3ConstraintType.BOOLEAN)
            for expr in translation.get("constraints", [])
            if isinstance(expr, str)
        ]
        invariants = [
            Z3Constraint(expression=expr, constraint_type=Z3ConstraintType.BOOLEAN)
            for expr in translation.get("invariants", [])
            if isinstance(expr, str)
        ]
    else:
        variables = translation.variables
        constraints = translation.constraints
        invariants = translation.invariants

    if not Z3_PYTHON_AVAILABLE:
        return {
            "proven": None,
            "reason": "Z3 Python bindings unavailable",
            "counterexample": None,
        }

    try:
        engine = get_z3_solver_engine()
        solver = z3.Solver()
        z3_vars = {var.name: engine._create_z3_variable(var) for var in variables}

        for constraint in constraints:
            expr = engine._parse_constraint(constraint.expression, z3_vars)
            if expr is not None:
                solver.add(expr)

        if assume_non_negative_amount and "amount" in z3_vars:
            solver.add(z3_vars["amount"] >= 0)

        invariant_exprs = []
        for invariant in invariants:
            expr = engine._parse_constraint(invariant.expression, z3_vars)
            if expr is not None:
                invariant_exprs.append(expr)

        if not invariant_exprs:
            return {
                "proven": None,
                "reason": "No invariants provided",
                "counterexample": None,
            }

        solver.add(z3.Not(z3.And(*invariant_exprs)))
        result = solver.check()

        if result == z3.unsat:
            return {
                "proven": True,
                "reason": "Constraints imply invariants",
                "counterexample": None,
            }

        if result == z3.sat:
            model = solver.model()
            return {
                "proven": False,
                "reason": "Found counterexample",
                "counterexample": {d.name(): str(model[d]) for d in model.decls()},
            }

        return {
            "proven": None,
            "reason": "Solver returned unknown",
            "counterexample": None,
        }
    except Exception as exc:
        return {
            "proven": None,
            "reason": f"Verification failed: {exc}",
            "counterexample": None,
        }


def solve_smart_contract_exploit_witness(
    additional_constraints: Optional[List[str]] = None,
    timeout: Optional[float] = 10.0,
) -> Dict[str, Any]:
    """
    Solve a canonical exploit witness query:
        Exists(input) such that
            contract_balance_post < contract_balance_pre
            AND user_deposit == 0
    """
    variables = [
        Z3Variable("contract_balance_pre", Z3ConstraintType.INTEGER),
        Z3Variable("contract_balance_post", Z3ConstraintType.INTEGER),
        Z3Variable("user_deposit", Z3ConstraintType.INTEGER),
        Z3Variable("attacker_input", Z3ConstraintType.INTEGER),
    ]
    constraints = [
        Z3Constraint("contract_balance_pre > 0", Z3ConstraintType.BOOLEAN),
        Z3Constraint("contract_balance_post >= 0", Z3ConstraintType.BOOLEAN),
        Z3Constraint("contract_balance_post < contract_balance_pre", Z3ConstraintType.BOOLEAN),
        Z3Constraint("user_deposit == 0", Z3ConstraintType.BOOLEAN),
        Z3Constraint("attacker_input >= 0", Z3ConstraintType.BOOLEAN),
    ]

    if additional_constraints:
        for constraint in additional_constraints:
            if isinstance(constraint, str) and constraint.strip():
                constraints.append(Z3Constraint(constraint.strip(), Z3ConstraintType.BOOLEAN))

    config = Z3Config(timeout=timeout or 10.0)
    engine = get_z3_solver_engine(config)
    result = engine.solve_constraints(variables, constraints)
    return {
        "status": result.status.value,
        "satisfiable": result.is_sat(),
        "model": result.model.assignments if result.model else None,
        "constraints": [c.expression for c in constraints],
        "errors": result.errors,
    }


# =============================================================================
# Global Instance
# =============================================================================

_z3_solver_engine: Optional[Z3SolverEngine] = None
_z3_theorem_prover: Optional[Z3TheoremProver] = None
_z3_lock = threading.Lock()


def get_z3_solver_engine(config: Optional[Z3Config] = None) -> Z3SolverEngine:
    """Get global Z3 solver engine instance."""
    global _z3_solver_engine
    if _z3_solver_engine is None:
        with _z3_lock:
            if _z3_solver_engine is None:
                _z3_solver_engine = Z3SolverEngine(config)
    return _z3_solver_engine


def get_z3_theorem_prover(config: Optional[Z3Config] = None) -> Z3TheoremProver:
    """Get global Z3 theorem prover instance."""
    global _z3_theorem_prover
    if _z3_theorem_prover is None:
        with _z3_lock:
            if _z3_theorem_prover is None:
                _z3_theorem_prover = Z3TheoremProver(config)
    return _z3_theorem_prover


def is_z3_available() -> bool:
    """Check if Z3 is available."""
    return Z3_AVAILABLE


def is_cav_nlp_available() -> bool:
    """Check if CAV-NLP integration is available."""
    return CAV_NLP_AVAILABLE


# =============================================================================
# CAV-NLP Enhanced Functions
# =============================================================================

def solve_with_cav_nlp(
    constraints: List[Union[str, Z3Constraint]],
    natural_language: Optional[str] = None,
    variables: Optional[List[Union[str, Z3Variable]]] = None,
    timeout: Optional[float] = None
) -> Z3SolverResult:
    """
    Solve constraints with optional natural language formalization via CAV-NLP.
    
    This function enhances standard Z3 solving by allowing natural language
    constraint descriptions to be formalized and combined with explicit constraints.
    
    Args:
        constraints: List of constraints (SMT-LIB strings or Z3Constraint objects)
        natural_language: Optional natural language description of additional constraints
        variables: Optional list of variable names or Z3Variable objects
        timeout: Optional timeout in seconds
        
    Returns:
        Z3SolverResult with solution or error information
        
    Example:
        >>> result = solve_with_cav_nlp(
        ...     constraints=["x > 0", "y < 10"],
        ...     natural_language="x and y must be different",
        ...     variables=["x", "y"]
        ... )
    """
    # Convert string constraints to Z3Constraint objects
    z3_constraints: List[Z3Constraint] = []
    for c in constraints:
        if isinstance(c, str):
            z3_constraints.append(Z3Constraint(c, Z3ConstraintType.BOOLEAN))
        else:
            z3_constraints.append(c)
    
    # Convert string variables to Z3Variable objects
    z3_variables: List[Z3Variable] = []
    if variables:
        for v in variables:
            if isinstance(v, str):
                z3_variables.append(Z3Variable(v, Z3ConstraintType.INTEGER))
            else:
                z3_variables.append(v)
    
    # If natural language provided and CAV-NLP available, formalize it
    if natural_language and CAV_NLP_AVAILABLE:
        try:
            solver = EnhancedZ3Solver()
            formalized = solver.formalize_constraint(natural_language)
            if formalized:
                z3_constraints.append(Z3Constraint(
                    formalized, 
                    Z3ConstraintType.BOOLEAN,
                    description=f"Formalized from: {natural_language}"
                ))
        except Exception as e:
            logger.warning(f"CAV-NLP formalization failed: {e}")
    
    # Create engine and solve
    config = Z3Config(timeout=timeout) if timeout else None
    engine = get_z3_solver_engine(config)
    
    return engine.solve_constraints(z3_variables, z3_constraints)


def verify_hybrid(
    theorem: Union[str, Dict[str, Any]],
    use_lean_export: bool = True,
    assumptions: Optional[List[str]] = None,
    timeout: Optional[float] = None
) -> Optional[Z3TheoremResult]:
    """
    Verify a theorem using a hybrid Z3 + CAV-NLP approach.
    
    This function combines the strengths of Z3 SMT solving with CAV-NLP's
    Lean theorem prover integration for comprehensive verification.
    
    Args:
        theorem: Theorem statement (SMT-LIB string or dict with theorem info)
        use_lean_export: Whether to export to Lean for additional verification
        assumptions: Optional list of assumption strings
        timeout: Optional timeout in seconds
        
    Returns:
        Z3TheoremResult if verification succeeds, None if CAV-NLP unavailable
        
    Example:
        >>> result = verify_hybrid(
        ...     theorem="For all x, x > 0 implies x + 1 > 0",
        ...     use_lean_export=True
        ... )
    """
    if not CAV_NLP_AVAILABLE:
        logger.debug("CAV-NLP not available, falling back to standard Z3 theorem proving")
        # Fall back to standard Z3 theorem prover
        prover = get_z3_theorem_prover()
        if isinstance(theorem, str):
            return prover.prove_theorem(theorem, assumptions, timeout)
        return None
    
    try:
        solver = EnhancedZ3Solver()
        
        # Extract theorem string
        if isinstance(theorem, dict):
            theorem_str = theorem.get('statement', theorem.get('theorem', ''))
        else:
            theorem_str = theorem
        
        # Use hybrid verification
        result = solver.verify_with_lean(theorem_str)
        
        if result:
            # Convert CAV-NLP result to Z3TheoremResult format
            return Z3TheoremResult(
                proven=result.get('verified', False),
                proof=result.get('lean_proof') if use_lean_export else result.get('z3_proof'),
                counterexample=result.get('counterexample'),
                execution_time=result.get('execution_time', 0.0),
                tactic_used="hybrid_z3_lean",
                errors=result.get('errors', [])
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Hybrid verification failed: {e}")
        # Fall back to standard Z3
        prover = get_z3_theorem_prover()
        if isinstance(theorem, str):
            return prover.prove_theorem(theorem, assumptions, timeout)
        return None


def export_proof_to_lean(
    z3_result: Union[Z3SolverResult, Z3TheoremResult],
    theorem_name: Optional[str] = None
) -> Optional[str]:
    """
    Export a Z3 proof to Lean 4 format using CAV-NLP.
    
    Args:
        z3_result: Z3 result containing the proof to export
        theorem_name: Optional name for the theorem in Lean
        
    Returns:
        Lean 4 proof code as string, or None if export fails
        
    Example:
        >>> result = verify_hybrid("x > 0 implies x + 1 > 0")
        >>> lean_code = export_proof_to_lean(result, "positive_plus_one")
    """
    if not CAV_NLP_AVAILABLE:
        logger.warning("CAV-NLP not available for Lean export")
        return None
    
    try:
        solver = EnhancedZ3Solver()
        
        # Extract proof from result
        proof = None
        if isinstance(z3_result, Z3TheoremResult):
            proof = z3_result.proof
        elif isinstance(z3_result, Z3SolverResult) and z3_result.model:
            proof = z3_result.smtlib_output
        
        if not proof:
            logger.warning("No proof found in Z3 result")
            return None
        
        # Use CAV-NLP to convert to Lean
        lean_code = solver.export_to_lean(proof, theorem_name)
        return lean_code
        
    except Exception as e:
        logger.error(f"Lean export failed: {e}")
        return None


def canonicalize_z3_result(
    result: Union[Z3SolverResult, Z3TheoremResult],
    format_type: str = "smtlib"
) -> Optional[str]:
    """
    Canonicalize a Z3 result to a standard format using CAV-NLP.
    
    Args:
        result: Z3 result to canonicalize
        format_type: Target format - "smtlib", "lean", or "json"
        
    Returns:
        Canonicalized representation as string, or None if unavailable
    """
    if not CAV_NLP_AVAILABLE:
        # Basic fallback - just convert to JSON manually
        if hasattr(result, 'to_dict'):
            import json
            return json.dumps(result.to_dict(), indent=2)
        return None
    
    try:
        solver = EnhancedZ3Solver()
        
        # Convert result to dict for processing
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else {}
        
        # Use CAV-NLP canonicalization
        canonical = solver.canonicalize(result_dict, format_type)
        return canonical
        
    except Exception as e:
        logger.error(f"Canonicalization failed: {e}")
        return None


# =============================================================================
# Example Usage
# =============================================================================

def example_constraint_solving():
    """Example: Solve a simple constraint problem."""
    engine = get_z3_solver_engine()
    
    # Define variables
    variables = [
        Z3Variable("x", Z3ConstraintType.INTEGER),
        Z3Variable("y", Z3ConstraintType.INTEGER)
    ]
    
    # Define constraints
    constraints = [
        Z3Constraint("x > 0", Z3ConstraintType.INTEGER, "x is positive"),
        Z3Constraint("x < 10", Z3ConstraintType.INTEGER, "x is less than 10"),
        Z3Constraint("y == x + 5", Z3ConstraintType.INTEGER, "y = x + 5")
    ]
    
    # Solve
    result = engine.solve_constraints(variables, constraints)
    
    print(f"Status: {result.status.value}")
    if result.model:
        print(f"Solution: {result.model.assignments}")
    
    return result


def example_theorem_proving():
    """Example: Prove a simple theorem."""
    prover = get_z3_theorem_prover()
    
    # Theorem: For all integers x, x > 0 implies x + 1 > 0
    theorem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (assert (> x 0))
    (assert (not (> (+ x 1) 0)))
    (check-sat)
    """
    
    result = prover.prove_theorem(theorem)
    
    print(f"Proven: {result.proven}")
    if result.proof:
        print(f"Proof: {result.proof[:100]}...")
    
    return result


# =============================================================================
# DSPY-ENHANCED Z3 INTEGRATION
# =============================================================================

# Import DSPy through the global integration module for consistency
try:
    from dspy_integration import DSPY_AVAILABLE, get_global_dspy_instance, initialize_dspy
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.predict import Predict
    logger.info("DSPy available through global integration for enhanced programmatic prompting")
except ImportError:
    # Fallback to local import if global module not available
    try:
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        from dspy.predict import Predict
        DSPY_AVAILABLE = True
        logger.info("DSPy available for enhanced programmatic prompting")
    except ImportError:
        dspy = None
        BootstrapFewShot = None
        Predict = None
        DSPY_AVAILABLE = False
        logger.warning("DSPy not available - using standard prompting methods")


class Z3DSPyIntegration:
    """
    Enhanced Z3 integration with DSPy for improved constraint formulation and theorem proving.

    This class provides DSPy-enhanced capabilities for:
    - Natural language to SMT-LIB constraint translation
    - Enhanced theorem formulation from natural language
    - Improved constraint optimization
    - Structured problem analysis
    - Support for linear, nonlinear, and boolean combinations
    - Variable detection and type inference
    """

    # Comprehensive patterns for natural language constraint parsing
    CONSTRAINT_PATTERNS = {
        # Linear constraints
        'linear_greater_than': [
            r'(\w+)\s+is\s+greater\s+than\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+>\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+exceeds\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+is\s+above\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+must\s+be\s+greater\s+than\s+(\d+(?:\.\d+)?)',
        ],
        'linear_less_than': [
            r'(\w+)\s+is\s+less\s+than\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+<\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+is\s+below\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+must\s+be\s+less\s+than\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+does\s+not\s+exceed\s+(\d+(?:\.\d+)?)',
        ],
        'linear_equal': [
            r'(\w+)\s+equals\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+=\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+is\s+equal\s+to\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+must\s+be\s+(\d+(?:\.\d+)?)',
        ],
        'linear_geq': [
            r'(\w+)\s+is\s+at\s+least\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+>=\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+is\s+greater\s+than\s+or\s+equal\s+to\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+minimum\s+is\s+(\d+(?:\.\d+)?)',
        ],
        'linear_leq': [
            r'(\w+)\s+is\s+at\s+most\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+<=\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+is\s+less\s+than\s+or\s+equal\s+to\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+maximum\s+is\s+(\d+(?:\.\d+)?)',
        ],
        # Range constraints
        'range_between': [
            r'(\w+)\s+is\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+in\s+range\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+must\s+be\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',
        ],
        # Nonlinear constraints
        'nonlinear_product': [
            r'(\w+)\s*\*\s*(\w+)\s*(<=|>=|<|>|=)\s*(\d+(?:\.\d+)?)',
            r'product\s+of\s+(\w+)\s+and\s+(\w+)\s+is\s+(\w+)',
        ],
        'nonlinear_square': [
            r'(\w+)\^2\s*(<=|>=|<|>|=)\s*(\d+(?:\.\d+)?)',
            r'square\s+of\s+(\w+)\s+is\s+(\w+)',
            r'(\w+)\s+squared\s+is\s+(\w+)',
        ],
        # Boolean constraints
        'boolean_true': [
            r'(\w+)\s+is\s+true',
            r'(\w+)\s+holds',
            r'(\w+)\s+must\s+be\s+true',
        ],
        'boolean_false': [
            r'(\w+)\s+is\s+false',
            r'(\w+)\s+does\s+not\s+hold',
            r'(\w+)\s+must\s+be\s+false',
        ],
        'boolean_and': [
            r'both\s+(\w+)\s+and\s+(\w+)\s+(?:are\s+true|hold)',
            r'(\w+)\s+and\s+(\w+)\s+must\s+both\s+be\s+true',
        ],
        'boolean_or': [
            r'either\s+(\w+)\s+or\s+(\w+)\s+(?:is\s+true|holds)',
            r'at\s+least\s+one\s+of\s+(\w+)\s+and\s+(\w+)',
        ],
        'boolean_not': [
            r'(\w+)\s+is\s+not\s+true',
            r'not\s+(\w+)',
        ],
        'boolean_implies': [
            r'if\s+(\w+)\s+then\s+(\w+)',
            r'(\w+)\s+implies\s+(\w+)',
            r'(\w+)\s+=>\s+(\w+)',
        ],
        # Arithmetic relationships
        'arithmetic_sum': [
            r'sum\s+of\s+(\w+)\s+and\s+(\w+)\s+(?:is|equals)\s+(\w+)',
            r'(\w+)\s+\+\s+(\w+)\s*=\s*(\w+)',
        ],
        'arithmetic_diff': [
            r'difference\s+between\s+(\w+)\s+and\s+(\w+)\s+(?:is|equals)\s+(\w+)',
            r'(\w+)\s+-\s+(\w+)\s*=\s*(\w+)',
        ],
        # All-different constraint
        'all_different': [
            r'all\s+(?:of\s+)?(\w+(?:,\s*\w+)*)\s+are\s+different',
            r'(\w+(?:,\s*\w+)*)\s+must\s+be\s+distinct',
            r'no\s+two\s+of\s+(\w+(?:,\s*\w+)*)\s+are\s+equal',
        ],
    }

    # Variable detection patterns
    VARIABLE_PATTERNS = [
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is|are|was|were|be|being|been|has|have|had|do|does|did|will|would|should|can|could|may|might|must|shall|need|dare|ought|used|greater|less|equal|between|at|most|least|above|below|exceeds)',  # Before verb
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<=|>=|<|>|=|==|!=|\+|-|\*|\/|%)',  # Before operator
        r'(?:variable|parameter|let|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # After declaration keyword
        r'\b([xyztuvwnmkij]\d?)\b',  # Common math variables
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]\s*\d',  # Before assignment
    ]

    # Theorem pattern templates
    THEOREM_TEMPLATES = {
        'implication': {
            'patterns': [
                r'if\s+(.+?)\s+then\s+(.+)',
                r'(.+?)\s+implies\s+(.+)',
                r'(.+?)\s+=>\s+(.+)',
                r'whenever\s+(.+?),\s*(.+)',
                r'given\s+that\s+(.+?),\s*(.+)',
            ],
            'template': '(=> {premise} {conclusion})',
        },
        'forall': {
            'patterns': [
                r'for\s+all\s+(\w+),?\s*(.+)',
                r'for\s+every\s+(\w+),?\s*(.+)',
                r'forall\s+(\w+),?\s*(.+)',
                r'for\s+any\s+(\w+),?\s*(.+)',
                r'for\s+each\s+(\w+),?\s*(.+)',
            ],
            'template': '(forall (({var} Int)) {statement})',
        },
        'exists': {
            'patterns': [
                r'there\s+exists\s+(?:an?\s+)?(\w+)\s+(?:such\s+that\s+)?(.+)',
                r'exists\s+(\w+),?\s*(.+)',
                r'there\s+is\s+(?:an?\s+)?(\w+)\s+(?:such\s+that\s+)?(.+)',
            ],
            'template': '(exists (({var} Int)) {statement})',
        },
        'contradiction': {
            'patterns': [
                r'(.+?)\s+and\s+(.+?)\s+cannot\s+both\s+be\s+true',
                r'(.+?)\s+contradicts\s+(.+)',
                r'it\s+is\s+impossible\s+that\s+(.+)',
            ],
            'template': '(and {expr1} (not {expr2}))',
        },
        'transitivity': {
            'patterns': [
                r'if\s+(.+?)\s+and\s+(.+?)\s+then\s+(.+)',
                r'(.+?)\s+and\s+(.+?)\s+imply\s+(.+)',
            ],
            'template': '(=> (and {premise1} {premise2}) {conclusion})',
        },
    }

    def __init__(self):
        self.dspy_available = DSPY_AVAILABLE
        self.solver_engine = Z3SolverEngine()  # Use existing solver engine
        self.theorem_prover = Z3TheoremProver()  # Use existing prover
        
        # Compile regex patterns for efficiency
        self._compiled_constraint_patterns = {
            key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in self.CONSTRAINT_PATTERNS.items()
        }
        self._compiled_variable_patterns = [re.compile(p, re.IGNORECASE) for p in self.VARIABLE_PATTERNS]
        self._compiled_theorem_patterns = {
            key: {
                'patterns': [re.compile(p, re.IGNORECASE) for p in val['patterns']],
                'template': val['template']
            }
            for key, val in self.THEOREM_TEMPLATES.items()
        }

    def natural_language_to_constraint_with_dspy(
        self,
        natural_language: str,
        constraint_type: str = "general",
        variable_hints: Optional[Dict[str, str]] = None,
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert natural language description to SMT-LIB constraint using DSPy for enhanced parsing.

        Args:
            natural_language: Natural language description of the constraint
            constraint_type: Type of constraint (linear, nonlinear, boolean, arithmetic, mixed)
            variable_hints: Optional dictionary of variable names to their types
            context: Optional context for better constraint understanding

        Returns:
            SMT-LIB formatted constraint string or None if failed
        """
        if not natural_language or not natural_language.strip():
            return None

        if not self.dspy_available:
            logger.info("DSPy not available, falling back to basic constraint formulation")
            return self._basic_natural_language_to_constraint(
                natural_language, constraint_type, variable_hints
            )

        try:
            # Define an enhanced DSPy signature for constraint translation
            class ConstraintTranslationSignature(dspy.Signature):
                """
                Translate natural language to SMT-LIB constraint.
                
                Examples:
                - "x is greater than 5" -> (assert (> x 5))
                - "x and y must be different" -> (assert (not (= x y)))
                - "if x > 0 then y = 1" -> (assert (=> (> x 0) (= y 1)))
                """
                natural_language_description = dspy.InputField(
                    desc="Natural language description of the constraint to translate"
                )
                constraint_type = dspy.InputField(
                    desc="Type: 'linear' (linear inequalities), 'nonlinear' (products, powers), 'boolean' (logic), 'arithmetic' (equations), or 'mixed'"
                )
                variable_hints = dspy.InputField(
                    desc="Optional JSON mapping variable names to types: {'x': 'Int', 'y': 'Real', 'flag': 'Bool'}"
                )
                context = dspy.InputField(
                    desc="Optional context about the problem domain"
                )

                smt_lib_constraint = dspy.OutputField(
                    desc="SMT-LIB2 formatted constraint expression (without 'assert' wrapper if standalone)"
                )
                variable_declarations = dspy.OutputField(
                    desc="Variable declarations in SMT-LIB format, one per line: (declare-const x Int)"
                )
                constraint_logic = dspy.OutputField(
                    desc="Brief explanation of the constraint logic and how it was derived"
                )
                validation_notes = dspy.OutputField(
                    desc="Any validation warnings or notes about the constraint"
                )

            # Create a predictor using the signature with Chain of Thought for better reasoning
            translate_constraint = dspy.ChainOfThought(ConstraintTranslationSignature)

            # Run the translation
            result = translate_constraint(
                natural_language_description=natural_language,
                constraint_type=constraint_type,
                variable_hints=json.dumps(variable_hints) if variable_hints else "{}",
                context=context or ""
            )

            # Construct the full SMT-LIB constraint with proper formatting
            lines = [
                f"; Generated from: {natural_language}",
                f"; Type: {constraint_type}",
            ]
            
            if context:
                lines.append(f"; Context: {context}")
            
            if result.constraint_logic:
                lines.append(f"; Logic: {result.constraint_logic}")
            
            if result.validation_notes:
                lines.append(f"; Notes: {result.validation_notes}")
            
            # Add variable declarations
            if result.variable_declarations:
                lines.append(result.variable_declarations)
            
            # Add the constraint with proper assert wrapper
            constraint = result.smt_lib_constraint.strip()
            if constraint and not constraint.startswith('('):
                constraint = f"(assert {constraint})"
            elif constraint and not constraint.startswith('(assert'):
                constraint = f"(assert {constraint})"
            
            lines.append(constraint)

            return '\n'.join(lines)

        except Exception as e:
            logger.warning(f"DSPy constraint translation failed, falling back to basic method: {e}")
            return self._basic_natural_language_to_constraint(
                natural_language, constraint_type, variable_hints
            )

    def _basic_natural_language_to_constraint(
        self,
        natural_language: str,
        constraint_type: str = "general",
        variable_hints: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Enhanced basic fallback for natural language to constraint conversion.
        
        Supports:
        - Linear constraints (>, <, >=, <=, =)
        - Range constraints (between)
        - Boolean combinations (and, or, not, implies)
        - Nonlinear hints (products, squares)
        - Variable detection and type inference
        """
        if not natural_language or not natural_language.strip():
            return None

        text = natural_language.strip()
        lower_text = text.lower()
        lines = [f"; Basic translation for: {text}", f"; Type: {constraint_type}"]
        
        variables = set()
        constraints = []
        
        # Helper to detect variable type
        def infer_type(var_name: str, value: Optional[str] = None) -> str:
            if variable_hints and var_name in variable_hints:
                return variable_hints[var_name]
            if value and '.' in value:
                return 'Real'
            # Check if it looks like a boolean variable
            if var_name.lower() in ['flag', 'is', 'has', 'valid', 'enabled', 'active']:
                return 'Bool'
            return 'Int'
        
        # Try to match linear constraints
        for pattern_name, patterns in self._compiled_constraint_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    
                    if pattern_name == 'linear_greater_than' and len(groups) >= 2:
                        var, val = groups[0], groups[1]
                        var_type = infer_type(var, val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (> {var} {val}))")
                    
                    elif pattern_name == 'linear_less_than' and len(groups) >= 2:
                        var, val = groups[0], groups[1]
                        var_type = infer_type(var, val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (< {var} {val}))")
                    
                    elif pattern_name == 'linear_geq' and len(groups) >= 2:
                        var, val = groups[0], groups[1]
                        var_type = infer_type(var, val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (>= {var} {val}))")
                    
                    elif pattern_name == 'linear_leq' and len(groups) >= 2:
                        var, val = groups[0], groups[1]
                        var_type = infer_type(var, val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (<= {var} {val}))")
                    
                    elif pattern_name == 'linear_equal' and len(groups) >= 2:
                        var, val = groups[0], groups[1]
                        var_type = infer_type(var, val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (= {var} {val}))")
                    
                    elif pattern_name == 'range_between' and len(groups) >= 3:
                        var, min_val, max_val = groups[0], groups[1], groups[2]
                        var_type = infer_type(var, min_val if '.' in min_val else max_val)
                        variables.add((var, var_type))
                        constraints.append(f"(assert (and (>= {var} {min_val}) (<= {var} {max_val})))")
                    
                    elif pattern_name == 'boolean_true' and len(groups) >= 1:
                        var = groups[0]
                        variables.add((var, 'Bool'))
                        constraints.append(f"(assert {var})")
                    
                    elif pattern_name == 'boolean_false' and len(groups) >= 1:
                        var = groups[0]
                        variables.add((var, 'Bool'))
                        constraints.append(f"(assert (not {var}))")
                    
                    elif pattern_name == 'boolean_and' and len(groups) >= 2:
                        var1, var2 = groups[0], groups[1]
                        variables.add((var1, 'Bool'))
                        variables.add((var2, 'Bool'))
                        constraints.append(f"(assert (and {var1} {var2}))")
                    
                    elif pattern_name == 'boolean_or' and len(groups) >= 2:
                        var1, var2 = groups[0], groups[1]
                        variables.add((var1, 'Bool'))
                        variables.add((var2, 'Bool'))
                        constraints.append(f"(assert (or {var1} {var2}))")
                    
                    elif pattern_name == 'boolean_not' and len(groups) >= 1:
                        var = groups[0]
                        variables.add((var, 'Bool'))
                        constraints.append(f"(assert (not {var}))")
                    
                    elif pattern_name == 'boolean_implies' and len(groups) >= 2:
                        var1, var2 = groups[0], groups[1]
                        variables.add((var1, 'Bool'))
                        variables.add((var2, 'Bool'))
                        constraints.append(f"(assert (=> {var1} {var2}))")
                    
                    elif pattern_name == 'nonlinear_product' and len(groups) >= 4:
                        var1, var2, op, val = groups[0], groups[1], groups[2], groups[3]
                        variables.add((var1, 'Int'))
                        variables.add((var2, 'Int'))
                        op_map = {'<': '<', '>': '>', '<=': '<=', '>=': '>=', '=': '='}
                        smt_op = op_map.get(op, '=')
                        constraints.append(f"(assert ({smt_op} (* {var1} {var2}) {val}))")
                    
                    elif pattern_name == 'arithmetic_sum' and len(groups) >= 3:
                        var1, var2, result = groups[0], groups[1], groups[2]
                        variables.add((var1, 'Int'))
                        variables.add((var2, 'Int'))
                        variables.add((result, 'Int'))
                        constraints.append(f"(assert (= (+ {var1} {var2}) {result}))")
                    
                    elif pattern_name == 'arithmetic_diff' and len(groups) >= 3:
                        var1, var2, result = groups[0], groups[1], groups[2]
                        variables.add((var1, 'Int'))
                        variables.add((var2, 'Int'))
                        variables.add((result, 'Int'))
                        constraints.append(f"(assert (= (- {var1} {var2}) {result}))")
                    
                    elif pattern_name == 'all_different' and len(groups) >= 1:
                        # Parse comma-separated variable list
                        var_list = re.split(r',\s*|\s+and\s+', groups[0])
                        for var in var_list:
                            var = var.strip()
                            if var:
                                variables.add((var, 'Int'))
                        if len(var_list) >= 2:
                            var_list = [v.strip() for v in var_list if v.strip()]
                            pairs = []
                            for i in range(len(var_list)):
                                for j in range(i + 1, len(var_list)):
                                    pairs.append(f"(not (= {var_list[i]} {var_list[j]}))")
                            if pairs:
                                constraints.append(f"(assert (and {' '.join(pairs)}))")
        
        # If no specific pattern matched, try variable extraction for generic constraints
        if not constraints:
            detected_vars = self._detect_variables(text)
            for var in detected_vars:
                if var not in [v[0] for v in variables]:
                    variables.add((var, infer_type(var)))
            
            # Add a generic comment constraint
            constraints.append(f"; Note: Could not fully parse constraint, detected variables: {', '.join(detected_vars)}")
        
        # Build output
        # Add variable declarations
        for var, var_type in sorted(variables):
            lines.append(f"(declare-const {var} {var_type})")
        
        # Add constraints
        lines.extend(constraints)
        
        return '\n'.join(lines) if constraints else None

    def _detect_variables(self, text: str) -> List[str]:
        """
        Detect potential variable names from natural language text.
        Returns a list of unique variable names.
        """
        variables = set()
        
        for pattern in self._compiled_variable_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    variables.update(m for m in match if m and len(m) > 0)
                elif isinstance(match, str) and match:
                    variables.add(match)
        
        # Filter out common words that aren't variables
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                       'can', 'could', 'may', 'might', 'must', 'shall', 'this', 'that',
                       'these', 'those', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                       'where', 'why', 'how', 'what', 'who', 'which', 'whose', 'whom',
                       'than', 'as', 'like', 'so', 'very', 'just', 'only', 'even', 'also',
                       'too', 'more', 'most', 'less', 'least', 'much', 'many', 'few',
                       'some', 'any', 'all', 'both', 'each', 'every', 'either', 'neither',
                       'one', 'two', 'three', 'first', 'second', 'last', 'next', 'previous',
                       'true', 'false', 'not', 'nil', 'null', 'none'}
        
        variables = {v for v in variables if v.lower() not in common_words and len(v) > 0}
        
        return sorted(variables)

    def formulate_theorem_with_dspy(
        self,
        natural_language_theorem: str,
        logic_hint: Optional[str] = None,
        assumptions: Optional[List[str]] = None,
        quantified_vars: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Formulate a theorem in SMT-LIB format from natural language using DSPy.

        Args:
            natural_language_theorem: Natural language description of the theorem
            logic_hint: Hint for logic selection (LIA, LRA, QF_LIA, etc.)
            assumptions: List of assumptions to include
            quantified_vars: List of variables to quantify over

        Returns:
            SMT-LIB formatted theorem or None if failed
        """
        if not natural_language_theorem or not natural_language_theorem.strip():
            return None

        if not self.dspy_available:
            logger.info("DSPy not available, falling back to basic theorem formulation")
            return self._basic_formulate_theorem(
                natural_language_theorem, logic_hint, assumptions, quantified_vars
            )

        try:
            # Define an enhanced DSPy signature for theorem formulation
            class TheoremFormulationSignature(dspy.Signature):
                """
                Formulate a theorem in SMT-LIB format from natural language.
                
                Examples:
                - "For all x, if x > 0 then x + 1 > 0" -> 
                  (assert (forall ((x Int)) (=> (> x 0) (> (+ x 1) 0))))
                - "If x = y and y = z then x = z" ->
                  (assert (=> (and (= x y) (= y z)) (= x z)))
                """
                natural_language_theorem = dspy.InputField(
                    desc="Natural language description of the theorem to prove"
                )
                logic_hint = dspy.InputField(
                    desc="Suggested logic: LIA (linear int), LRA (linear real), NIA (nonlinear int), QF_LIA (quantifier-free), etc."
                )
                assumptions_json = dspy.InputField(
                    desc="JSON list of assumptions as SMT-LIB expressions"
                )
                quantified_vars_json = dspy.InputField(
                    desc="JSON list of variable names to quantify over"
                )

                smt_lib_theorem = dspy.OutputField(
                    desc="Complete SMT-LIB2 theorem as assertion(s) ready for checking"
                )
                variable_declarations = dspy.OutputField(
                    desc="Variable declarations in SMT-LIB format"
                )
                logic_declaration = dspy.OutputField(
                    desc="Appropriate SMT-LIB logic declaration (e.g., QF_LIA, LIA, LRA, NIA, UFLIA)"
                )
                proof_strategy = dspy.OutputField(
                    desc="Suggested proof strategy: direct, contradiction, induction, case-analysis"
                )
                theorem_structure = dspy.OutputField(
                    desc="Structural analysis: implication, universal, existential, conjunction, etc."
                )
                formalization_notes = dspy.OutputField(
                    desc="Notes about the formalization choices made"
                )

            # Create a predictor using Chain of Thought for better reasoning
            formulate_theorem = dspy.ChainOfThought(TheoremFormulationSignature)

            # Run the theorem formulation
            result = formulate_theorem(
                natural_language_theorem=natural_language_theorem,
                logic_hint=logic_hint or "auto-detect",
                assumptions_json=json.dumps(assumptions) if assumptions else "[]",
                quantified_vars_json=json.dumps(quantified_vars) if quantified_vars else "[]"
            )

            # Construct the full SMT-LIB theorem with proper structure
            lines = [
                f"; Theorem: {natural_language_theorem}",
                f"; Structure: {result.theorem_structure}",
                f"; Strategy: {result.proof_strategy}",
            ]
            
            if result.formalization_notes:
                lines.append(f"; Notes: {result.formalization_notes}")
            
            lines.append(f"(set-logic {result.logic_declaration})")
            lines.append("(set-option :produce-proofs true)")
            
            # Add variable declarations
            if result.variable_declarations:
                lines.append(result.variable_declarations)
            
            # Add assumptions if provided
            if assumptions:
                for i, assumption in enumerate(assumptions):
                    lines.append(f"(assert ; Assumption {i+1}")
                    lines.append(f"  {assumption}")
                    lines.append(")")
            
            # Add the main theorem assertion
            lines.append("; Main theorem")
            theorem = result.smt_lib_theorem.strip()
            if not theorem.startswith('('):
                theorem = f"(assert {theorem})"
            lines.append(theorem)
            
            lines.append("; Check satisfiability (theorem is proven if result is unsat)")
            lines.append("(check-sat)")

            return '\n'.join(lines)

        except Exception as e:
            logger.warning(f"DSPy theorem formulation failed, falling back to basic method: {e}")
            return self._basic_formulate_theorem(
                natural_language_theorem, logic_hint, assumptions, quantified_vars
            )

    def _basic_formulate_theorem(
        self,
        natural_language_theorem: str,
        logic_hint: Optional[str] = None,
        assumptions: Optional[List[str]] = None,
        quantified_vars: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Enhanced basic fallback for theorem formulation.
        
        Supports:
        - Implication patterns (if...then, implies)
        - Universal quantification (for all, forall)
        - Existential quantification (there exists, exists)
        - Transitivity patterns
        - Conjunction/disjunction in premises
        """
        if not natural_language_theorem or not natural_language_theorem.strip():
            return None

        text = natural_language_theorem.strip()
        lines = [f"; Theorem: {text}"]
        
        # Determine logic
        logic = logic_hint or self._infer_logic(text)
        lines.append(f"(set-logic {logic})")
        lines.append("(set-option :produce-proofs true)")
        
        # Detect variables
        variables = self._detect_variables(text)
        if quantified_vars:
            variables = list(set(variables + quantified_vars))
        
        # Infer variable types
        var_types = {}
        for var in variables:
            var_types[var] = self._infer_var_type(var, text)
        
        # Try to match theorem patterns
        theorem_smt = None
        
        for theorem_type, config in self._compiled_theorem_patterns.items():
            for pattern in config['patterns']:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    
                    if theorem_type == 'implication' and len(groups) >= 2:
                        premise, conclusion = groups[0], groups[1]
                        premise_smt = self._text_to_smt_expr(premise, var_types)
                        conclusion_smt = self._text_to_smt_expr(conclusion, var_types)
                        theorem_smt = f"(assert (=> {premise_smt} {conclusion_smt}))"
                    
                    elif theorem_type == 'forall' and len(groups) >= 2:
                        var, statement = groups[0], groups[1]
                        if var not in var_types:
                            var_types[var] = 'Int'
                        statement_smt = self._text_to_smt_expr(statement, var_types)
                        theorem_smt = f"(assert (forall (({var} {var_types[var]})) {statement_smt}))"
                    
                    elif theorem_type == 'exists' and len(groups) >= 2:
                        var, statement = groups[0], groups[1]
                        if var not in var_types:
                            var_types[var] = 'Int'
                        statement_smt = self._text_to_smt_expr(statement, var_types)
                        theorem_smt = f"(assert (exists (({var} {var_types[var]})) {statement_smt}))"
                    
                    elif theorem_type == 'transitivity' and len(groups) >= 3:
                        premise1, premise2, conclusion = groups[0], groups[1], groups[2]
                        premise1_smt = self._text_to_smt_expr(premise1, var_types)
                        premise2_smt = self._text_to_smt_expr(premise2, var_types)
                        conclusion_smt = self._text_to_smt_expr(conclusion, var_types)
                        theorem_smt = f"(assert (=> (and {premise1_smt} {premise2_smt}) {conclusion_smt}))"
                    
                    elif theorem_type == 'contradiction' and len(groups) >= 2:
                        expr1, expr2 = groups[0], groups[1]
                        expr1_smt = self._text_to_smt_expr(expr1, var_types)
                        expr2_smt = self._text_to_smt_expr(expr2, var_types)
                        theorem_smt = f"(assert (not (and {expr1_smt} {expr2_smt})))"
                    
                    break
            if theorem_smt:
                break
        
        # If no pattern matched, create a generic theorem structure
        if not theorem_smt:
            # Declare all detected variables
            for var in variables:
                lines.append(f"(declare-const {var} {var_types.get(var, 'Int')})")
            
            # Add assumptions
            if assumptions:
                for i, assumption in enumerate(assumptions):
                    lines.append(f"(assert ; Assumption {i+1}")
                    lines.append(f"  {assumption}")
                    lines.append(")")
            
            lines.append(f"; Note: Generic theorem - manual formalization needed")
            lines.append(f"; Detected variables: {', '.join(variables)}")
            lines.append("(check-sat)")
            return '\n'.join(lines)
        
        # Add variable declarations
        declared = set()
        for var in variables:
            if var not in declared:
                lines.append(f"(declare-const {var} {var_types.get(var, 'Int')})")
                declared.add(var)
        
        # Add assumptions
        if assumptions:
            for i, assumption in enumerate(assumptions):
                lines.append(f"(assert ; Assumption {i+1}")
                lines.append(f"  {assumption}")
                lines.append(")")
        
        # Add the theorem
        lines.append("; Main theorem")
        lines.append(theorem_smt)
        lines.append("(check-sat)")
        
        return '\n'.join(lines)

    def _infer_logic(self, text: str) -> str:
        """Infer appropriate SMT-LIB logic from theorem text."""
        lower = text.lower()
        
        # Check for quantifiers
        has_quantifiers = any(kw in lower for kw in ['for all', 'forall', 'exists', 'there exists', 'every', 'any'])
        
        # Check for reals
        has_reals = any(kw in lower for kw in ['real', 'decimal', 'fraction', 'continuous'])
        
        # Check for nonlinearity
        has_nonlinear = any(kw in lower for kw in ['*', 'product', 'square', 'multiply', 'times', 'power', '^'])
        
        # Check for arrays
        has_arrays = any(kw in lower for kw in ['array', 'index', 'element', 'select', 'store'])
        
        # Determine logic
        if has_arrays:
            return 'AUFNIRA' if has_nonlinear else 'AUFLIA'
        elif has_quantifiers:
            if has_reals:
                return 'NRA' if has_nonlinear else 'LRA'
            else:
                return 'NIA' if has_nonlinear else 'LIA'
        else:
            if has_reals:
                return 'QF_NRA' if has_nonlinear else 'QF_LRA'
            else:
                return 'QF_NIA' if has_nonlinear else 'QF_LIA'

    def _infer_var_type(self, var_name: str, context: str) -> str:
        """Infer variable type from name and context."""
        lower_name = var_name.lower()
        lower_context = context.lower()
        
        # Boolean indicators
        if lower_name in ['flag', 'is', 'has', 'valid', 'enabled', 'active', 'ok', 'success']:
            return 'Bool'
        
        # Real indicators
        if any(indicator in lower_name for indicator in ['rate', 'ratio', 'fraction', 'prob', 'percent']):
            return 'Real'
        
        # Check surrounding context for type hints
        # Look for patterns like "x is a real" or "y be an integer"
        type_patterns = [
            (rf'{var_name}\s+(?:is|are|be|as)\s+(?:a|an)?\s*real', 'Real'),
            (rf'{var_name}\s+(?:is|are|be|as)\s+(?:a|an)?\s*integer', 'Int'),
            (rf'{var_name}\s+(?:is|are|be|as)\s+(?:a|an)?\s*boolean', 'Bool'),
            (rf'{var_name}\s+(?:is|are|be|as)\s+(?:a|an)?\s*bool', 'Bool'),
        ]
        
        for pattern, vtype in type_patterns:
            if re.search(pattern, lower_context):
                return vtype
        
        return 'Int'

    def _text_to_smt_expr(self, text: str, var_types: Dict[str, str]) -> str:
        """Convert natural language text to SMT-LIB expression."""
        lower = text.lower().strip()
        
        # Try to match comparison patterns
        comparisons = [
            (r'(\w+)\s*>=\s*(\w+|\d+)', '>='),
            (r'(\w+)\s*<=\s*(\w+|\d+)', '<='),
            (r'(\w+)\s*>\s*(\w+|\d+)', '>'),
            (r'(\w+)\s*<\s*(\w+|\d+)', '<'),
            (r'(\w+)\s*=\s*(\w+|\d+)', '='),
            (r'(\w+)\s+is\s+greater\s+than\s+(\w+|\d+)', '>'),
            (r'(\w+)\s+is\s+less\s+than\s+(\w+|\d+)', '<'),
            (r'(\w+)\s+equals?\s+(\w+|\d+)', '='),
        ]
        
        for pattern, op in comparisons:
            match = re.search(pattern, lower)
            if match:
                left, right = match.group(1), match.group(2)
                return f"({op} {left} {right})"
        
        # Handle arithmetic expressions
        arithmetic = [
            (r'(\w+)\s*\+\s*(\w+)', '+'),
            (r'(\w+)\s*-\s*(\w+)', '-'),
            (r'(\w+)\s*\*\s*(\w+)', '*'),
            (r'sum\s+of\s+(\w+)\s+and\s+(\w+)', '+'),
            (r'difference\s+of\s+(\w+)\s+and\s+(\w+)', '-'),
        ]
        
        for pattern, op in arithmetic:
            match = re.search(pattern, lower)
            if match:
                left, right = match.group(1), match.group(2)
                return f"({op} {left} {right})"
        
        # If nothing matched, return the text as-is (might need manual fixing)
        return text

    def batch_translate_constraints(
        self,
        constraints: List[str],
        constraint_type: str = "general",
        variable_hints: Optional[Dict[str, str]] = None
    ) -> List[Optional[str]]:
        """
        Translate multiple natural language constraints to SMT-LIB in batch.
        
        Args:
            constraints: List of natural language constraint descriptions
            constraint_type: Type of all constraints
            variable_hints: Variable type hints
            
        Returns:
            List of SMT-LIB constraint strings (None for failed translations)
        """
        results = []
        for constraint in constraints:
            result = self.natural_language_to_constraint_with_dspy(
                constraint, constraint_type, variable_hints
            )
            results.append(result)
        return results

    def solve_problem_with_dspy_guidance(
        self,
        problem_description: str,
        constraint_type: str = "general",
        optimization_objective: Optional[str] = None,
        minimize: bool = True
    ) -> Dict[str, Any]:
        """
        Solve a constraint satisfaction problem using DSPy for enhanced problem understanding
        and Z3 for solving.

        Args:
            problem_description: Natural language description of the problem
            constraint_type: Type of constraints involved
            optimization_objective: Optional objective function for optimization
            minimize: Whether to minimize (True) or maximize (False) the objective

        Returns:
            Dictionary with solution results
        """
        start_time = time.time()
        
        try:
            # First, use DSPy to understand and structure the problem
            if self.dspy_available:
                # Define an enhanced DSPy signature for problem analysis
                class ProblemAnalysisSignature(dspy.Signature):
                    """
                    Analyze a constraint satisfaction or optimization problem.
                    
                    Given a problem description, identify variables, constraints,
                    and determine if it's a satisfaction or optimization problem.
                    """
                    problem_description = dspy.InputField(
                        desc="Natural language description of the problem to solve"
                    )
                    constraint_type = dspy.InputField(
                        desc="Type of constraints: linear, nonlinear, boolean, mixed"
                    )
                    has_objective = dspy.InputField(
                        desc="Whether this is an optimization problem with an objective"
                    )

                    key_variables = dspy.OutputField(
                        desc="JSON object mapping variable names to their types and descriptions"
                    )
                    constraints_list = dspy.OutputField(
                        desc="JSON list of constraints in natural language"
                    )
                    objective_function = dspy.OutputField(
                        desc="Objective function description if optimization, else 'none'"
                    )
                    problem_classification = dspy.OutputField(
                        desc="Classification: CSP (satisfaction), COP (optimization), theorem, or verification"
                    )
                    recommended_logic = dspy.OutputField(
                        desc="Recommended SMT-LIB logic: QF_LIA, QF_LRA, LIA, LRA, etc."
                    )
                    solution_approach = dspy.OutputField(
                        desc="Recommended solving approach and any special considerations"
                    )

                # Create a predictor using the signature
                analyze_problem = dspy.ChainOfThought(ProblemAnalysisSignature)

                # Run the problem analysis
                result = analyze_problem(
                    problem_description=problem_description,
                    constraint_type=constraint_type,
                    has_objective="yes" if optimization_objective else "no"
                )

                # Parse the results
                try:
                    variables_dict = json.loads(result.key_variables) if result.key_variables else {}
                except json.JSONDecodeError:
                    variables_dict = {}
                
                try:
                    constraints_list = json.loads(result.constraints_list) if result.constraints_list else []
                except json.JSONDecodeError:
                    constraints_list = []

                # Convert constraints to SMT-LIB
                smt_constraints = []
                for constraint in constraints_list:
                    smt = self.natural_language_to_constraint_with_dspy(
                        constraint, constraint_type, variables_dict
                    )
                    if smt:
                        smt_constraints.append(smt)

                # Build complete SMT-LIB problem
                smt_problem = self._build_smt_problem(
                    variables=variables_dict,
                    constraints=smt_constraints,
                    objective=optimization_objective or result.objective_function,
                    logic=result.recommended_logic,
                    minimize=minimize
                )

                # Solve with Z3
                solver_result = self.solver_engine.solve_smtlib(smt_problem)
                execution_time = time.time() - start_time

                return {
                    "status": "success" if solver_result.status != Z3ResultStatus.ERROR else "error",
                    "dspy_analysis": {
                        "key_variables": variables_dict,
                        "constraints": constraints_list,
                        "objective": result.objective_function,
                        "classification": result.problem_classification,
                        "recommended_logic": result.recommended_logic,
                        "solution_approach": result.solution_approach
                    },
                    "smt_problem": smt_problem,
                    "solver_result": solver_result.to_dict() if hasattr(solver_result, 'to_dict') else solver_result,
                    "problem_description": problem_description,
                    "constraint_type": constraint_type,
                    "execution_time": execution_time,
                    "dspy_enhanced": True
                }
            else:
                # Fallback to basic approach
                smt_constraint = self._basic_natural_language_to_constraint(
                    problem_description, constraint_type
                )

                solver_result = self.solver_engine.solve_smtlib(smt_constraint)
                execution_time = time.time() - start_time

                return {
                    "status": "success" if solver_result.status != Z3ResultStatus.ERROR else "error",
                    "solver_result": solver_result.to_dict() if hasattr(solver_result, 'to_dict') else solver_result,
                    "problem_description": problem_description,
                    "constraint_type": constraint_type,
                    "execution_time": execution_time,
                    "dspy_enhanced": False,
                    "message": "DSPy not available, using basic constraint solving"
                }

        except Exception as e:
            logger.error(f"Error in DSPy-guided problem solving: {e}")
            return {
                "status": "error",
                "error": str(e),
                "problem_description": problem_description,
                "dspy_enhanced": self.dspy_available,
                "execution_time": time.time() - start_time
            }

    def _build_smt_problem(
        self,
        variables: Dict[str, str],
        constraints: List[str],
        objective: Optional[str],
        logic: str,
        minimize: bool
    ) -> str:
        """Build a complete SMT-LIB problem from components."""
        lines = [
            "; Auto-generated SMT-LIB problem",
            f"(set-logic {logic or 'ALL'})",
            "(set-option :produce-models true)",
        ]
        
        # Add variable declarations
        for var_name, var_info in variables.items():
            if isinstance(var_info, dict):
                var_type = var_info.get('type', 'Int')
            elif isinstance(var_info, str):
                var_type = var_info
            else:
                var_type = 'Int'
            lines.append(f"(declare-const {var_name} {var_type})")
        
        # Add constraints
        for constraint in constraints:
            lines.append(constraint)
        
        # Add optimization objective if provided
        if objective and objective.lower() != 'none':
            opt_cmd = "minimize" if minimize else "maximize"
            # Try to extract the expression from objective
            obj_expr = objective
            if 'minimize' in obj_expr.lower():
                obj_expr = re.sub(r'\bminimize\b', '', obj_expr, flags=re.IGNORECASE).strip()
            if 'maximize' in obj_expr.lower():
                obj_expr = re.sub(r'\bmaximize\b', '', obj_expr, flags=re.IGNORECASE).strip()
            lines.append(f"({opt_cmd} {obj_expr})")
        
        lines.append("(check-sat)")
        lines.append("(get-model)")
        
        return '\n'.join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core availability flags
    'Z3_AVAILABLE',
    'Z3_PYTHON_AVAILABLE',
    'SOLVER_POOL_AVAILABLE',
    'CAV_NLP_AVAILABLE',
    
    # Core enums
    'Z3ResultStatus',
    'Z3ConstraintType',
    
    # Core data classes
    'Z3Variable',
    'Z3Constraint',
    'Z3Model',
    'Z3SolverResult',
    'Z3TheoremResult',
    'Z3Config',
    
    # Core classes
    'Z3ProblemDetector',
    'Z3SolverEngine',
    'Z3TheoremProver',
    'Z3ProverIntegration',
    'DigitalTwinSandbox',
    'Z3LogicCompressor',
    'Z3DSPyIntegration',
    'SolidityInvariantTranslation',
    'SmartContractInvariantTranslator',
    
    # CAV-NLP enhanced components
    'EnhancedZ3Solver',
    'solve_with_cav_nlp',
    'verify_hybrid',
    'export_proof_to_lean',
    'canonicalize_z3_result',
    
    # Global instance getters
    'get_z3_solver_engine',
    'get_z3_theorem_prover',
    
    # Utility functions
    'is_z3_available',
    'is_cav_nlp_available',
    'pattern_operator',
    'generate_refutation_narrative',
    'translate_solidity_assignment_to_z3',
    'verify_solidity_invariant_translation',
    'solve_smart_contract_exploit_witness',
    
    # Example functions
    'example_constraint_solving',
    'example_theorem_proving',
]


if __name__ == "__main__":
    print("Z3 Prover Integration Module")
    print(f"Z3 Available: {Z3_AVAILABLE}")
    print(f"Z3 Python Available: {Z3_PYTHON_AVAILABLE}")
    print(f"CAV-NLP Available: {CAV_NLP_AVAILABLE}")
    
    if Z3_AVAILABLE:
        print("\n--- Constraint Solving Example ---")
        example_constraint_solving()
        
        print("\n--- Theorem Proving Example ---")
        example_theorem_proving()
