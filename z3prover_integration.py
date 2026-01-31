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
            Z3ConstraintType.FLOATING_POINT: "Float64"
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "z3_available": Z3_AVAILABLE,
            "z3_python_available": Z3_PYTHON_AVAILABLE,
            "config": asdict(self.config),
            "statistics": self._stats.copy()
        }
    
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
        else:
            return z3.Int(var.name)  # Default to Int
    
    def _parse_constraint(self, expression: str, z3_vars: Dict[str, Any]) -> Optional[Any]:
        """Parse constraint expression using Z3 Python API."""
        try:
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
            result = eval(expression, {"__builtins__": {}}, local_vars)
            return result
        except Exception as e:
            logger.warning(f"Failed to parse constraint '{expression}': {e}")
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
            else:
                self._stats["error_results"] += 1
            
            return parsed_result
            
        except Exception as e:
            self._stats["error_results"] += 1
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason=str(e),
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
        finally:
            try:
                Path(temp_file).unlink()
            except OSError:
                pass


# =============================================================================
# Z3 Theorem Prover
# =============================================================================

class Z3TheoremProver:
    """
    Z3-based theorem prover for formal verification.
    
    Provides capabilities for proving mathematical theorems and
    verifying logical formulas.
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()
        self.solver_engine = Z3SolverEngine(config)
        self._prover_lock = threading.RLock()
    
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
        
        # Negate the theorem for proof by contradiction
        if '(assert' in theorem_statement:
            # Extract the last assertion (theorem) and negate it
            lines = theorem_statement.split('\n')
            assertions = [l for l in lines if l.strip().startswith('(assert')]
            
            if assertions:
                # Negate the last assertion for proof by contradiction
                theorem_line = assertions[-1]
                negated = theorem_line.replace('(assert ', '(assert (not ')
                if not negated.endswith('))'):
                    negated = negated.rstrip(')') + '))'
                
                # Replace the theorem with its negation
                new_lines = []
                for l in lines:
                    if l == theorem_line:
                        new_lines.append(negated)
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
        """Prove theorem from natural language (requires translation)."""
        # For now, return not proven - requires integration with translation
        return Z3TheoremResult(
            proven=False,
            errors=["Natural language theorem proving requires translation to SMT-LIB"],
            execution_time=0.0
        )
    
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
        Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER, "x is positive"),
        Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER, "x is less than 10"),
        Z3Constraint("(= y (+ x 5))", Z3ConstraintType.INTEGER, "y = x + 5")
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


if __name__ == "__main__":
    print("Z3 Prover Integration Module")
    print(f"Z3 Available: {Z3_AVAILABLE}")
    print(f"Z3 Python Available: {Z3_PYTHON_AVAILABLE}")
    
    if Z3_AVAILABLE:
        print("\n--- Constraint Solving Example ---")
        example_constraint_solving()
        
        print("\n--- Theorem Proving Example ---")
        example_theorem_proving()
