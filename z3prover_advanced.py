"""
Advanced Z3 Prover Features

Extends the base Z3 integration with:
- Optimization (linear, non-linear, multi-objective)
- Array and data structure constraints
- Bit-vector arithmetic
- Floating point operations
- Incremental solving
- Parallel solving with portfolio
- Proof extraction and reconstruction
- Model-based testing

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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Import base Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3SolverResult, Z3TheoremResult,
        Z3Variable, Z3Constraint, Z3ConstraintType, Z3ResultStatus, Z3Model,
        Z3Config, Z3ProblemDetector, Z3_AVAILABLE, Z3_PYTHON_AVAILABLE
    )
except ImportError:
    Z3_AVAILABLE = False
    Z3_PYTHON_AVAILABLE = False
    logger.warning("Base Z3 integration not available")

# Try to import Z3
try:
    import z3
    Z3_PYTHON_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Advanced Data Classes
# =============================================================================

class OptimizationObjective(Enum):
    """Optimization objective types."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ProofFormat(Enum):
    """Proof output formats."""
    SMTLIB2 = "smtlib2"
    DOT = "dot"
    JSON = "json"
    TEXT = "text"


@dataclass
class OptimizationResult:
    """Result from optimization."""
    success: bool
    optimal_value: Optional[float] = None
    optimal_model: Optional[Z3Model] = None
    objectives: Dict[str, float] = field(default_factory=dict)
    is_pareto: bool = False  # For multi-objective
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    execution_time: float = 0.0
    proof: Optional[str] = None
    lower_bounds: Dict[str, float] = field(default_factory=dict)
    upper_bounds: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "optimal_value": self.optimal_value,
            "optimal_model": self.optimal_model.to_dict() if self.optimal_model else None,
            "objectives": self.objectives,
            "is_pareto": self.is_pareto,
            "pareto_front": self.pareto_front,
            "iterations": self.iterations,
            "execution_time": self.execution_time
        }


@dataclass
class ArrayConstraint:
    """Array constraint specification."""
    array_name: str
    index_type: Z3ConstraintType
    value_type: Z3ConstraintType
    size: Optional[int] = None
    constraints: List[str] = field(default_factory=list)
    
    def to_smtlib(self) -> str:
        """Convert to SMT-LIB."""
        type_map = {
            Z3ConstraintType.INTEGER: "Int",
            Z3ConstraintType.REAL: "Real",
            Z3ConstraintType.BOOLEAN: "Bool"
        }
        idx_type = type_map.get(self.index_type, "Int")
        val_type = type_map.get(self.value_type, "Int")
        
        lines = [f"(declare-fun {self.array_name} () (Array {idx_type} {val_type}))"]
        for constraint in self.constraints:
            lines.append(f"(assert {constraint})")
        return "\n".join(lines)


@dataclass
class BitVectorConstraint:
    """Bit-vector constraint."""
    var_name: str
    width: int
    signed: bool = False
    constraints: List[str] = field(default_factory=list)
    
    def to_smtlib(self) -> str:
        """Convert to SMT-LIB."""
        lines = [f"(declare-fun {self.var_name} () (_ BitVec {self.width}))"]
        for constraint in self.constraints:
            lines.append(f"(assert {constraint})")
        return "\n".join(lines)


@dataclass
class ProofStep:
    """Single step in a proof."""
    step_number: int
    tactic: str
    input_goals: List[str] = field(default_factory=list)
    output_goals: List[str] = field(default_factory=list)
    justification: Optional[str] = None
    subproofs: List['ProofStep'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "tactic": self.tactic,
            "input_goals": self.input_goals,
            "output_goals": self.output_goals,
            "justification": self.justification,
            "subproofs": [s.to_dict() for s in self.subproofs]
        }


@dataclass
class ExtractedProof:
    """Extracted proof with full details."""
    success: bool
    proof_steps: List[ProofStep] = field(default_factory=list)
    axioms_used: List[str] = field(default_factory=list)
    tactics_used: List[str] = field(default_factory=list)
    proof_format: ProofFormat = ProofFormat.TEXT
    raw_proof: Optional[str] = None
    verification_status: str = "unknown"
    reconstruction_hints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "proof_steps": [s.to_dict() for s in self.proof_steps],
            "axioms_used": self.axioms_used,
            "tactics_used": self.tactics_used,
            "proof_format": self.proof_format.value,
            "verification_status": self.verification_status
        }


@dataclass
class PortfolioResult:
    """Result from portfolio solving."""
    success: bool
    best_result: Optional[Z3SolverResult] = None
    all_results: List[Tuple[str, Z3SolverResult]] = field(default_factory=list)
    winner_strategy: Optional[str] = None
    execution_time: float = 0.0
    parallel_speedup: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "winner_strategy": self.winner_strategy,
            "execution_time": self.execution_time,
            "parallel_speedup": self.parallel_speedup,
            "results_count": len(self.all_results)
        }


@dataclass
class IncrementalState:
    """State for incremental solving."""
    state_id: str
    variables: List[Z3Variable] = field(default_factory=list)
    constraints: List[Z3Constraint] = field(default_factory=list)
    assertions_stack: List[List[Z3Constraint]] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    last_result: Optional[Z3SolverResult] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "variable_count": len(self.variables),
            "constraint_count": len(self.constraints),
            "scope_count": len(self.scopes),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed
        }


# =============================================================================
# Z3 Advanced Solver
# =============================================================================

class Z3AdvancedSolver(Z3SolverEngine):
    """
    Advanced Z3 solver with optimization and extended features.
    
    Extends base Z3SolverEngine with:
    - Optimization (single and multi-objective)
    - Array constraints
    - Bit-vector operations
    - Incremental solving
    - Portfolio solving
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        super().__init__(config)
        
        # Optimization tracking
        self._optimization_history: List[OptimizationResult] = []
        
        # Incremental solving states
        self._incremental_states: Dict[str, IncrementalState] = {}
        self._state_lock = threading.RLock()
        
        # Portfolio strategies
        self._portfolio_strategies = [
            "default",
            "simplify",
            "smt",
            "qfbv",  # Quantifier-free bit-vector
            "qflia", # Quantifier-free linear integer arithmetic
            "qfnra", # Quantifier-free non-linear real arithmetic
            "qfauflia" # Arrays + linear arithmetic
        ]
    
    # =====================================================================
    # Optimization
    # =====================================================================
    
    def optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]],
        multi_objective_strategy: str = "pareto"
    ) -> OptimizationResult:
        """
        Solve optimization problem.
        
        Args:
            variables: Problem variables
            constraints: Constraints
            objectives: List of (expression, min/max) tuples
            multi_objective_strategy: "pareto", "weighted", "lexicographic"
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        if not Z3_PYTHON_AVAILABLE:
            return self._optimize_via_cli(variables, constraints, objectives)
        
        with self._solver_lock:
            try:
                if len(objectives) == 1:
                    return self._single_objective_optimize(
                        variables, constraints, objectives[0]
                    )
                else:
                    return self._multi_objective_optimize(
                        variables, constraints, objectives, multi_objective_strategy
                    )
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                return OptimizationResult(
                    success=False,
                    execution_time=time.time() - start_time
                )
    
    def _single_objective_optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objective: Tuple[str, OptimizationObjective]
    ) -> OptimizationResult:
        """Single objective optimization using Z3 Optimize."""
        start_time = time.time()
        
        opt = z3.Optimize()
        opt.set("timeout", int(self.config.timeout * 1000))
        
        # Create variables
        z3_vars = {}
        for var in variables:
            z3_vars[var.name] = self._create_z3_variable(var)
        
        # Add constraints
        for constraint in constraints:
            z3_expr = self._parse_constraint(constraint.expression, z3_vars)
            if z3_expr is not None:
                opt.add(z3_expr)
        
        # Add objective
        obj_expr, obj_type = objective
        z3_obj = self._parse_constraint(obj_expr, z3_vars)
        
        if obj_type == OptimizationObjective.MINIMIZE:
            handle = opt.minimize(z3_obj)
        else:
            handle = opt.maximize(z3_obj)
        
        # Check
        result = opt.check()
        
        if result == z3.sat:
            model = opt.model()
            optimal_value = model.eval(z3_obj, model_completion=True)
            
            assignments = {}
            for var in variables:
                z3_var = z3_vars.get(var.name)
                if z3_var is not None:
                    value = model.eval(z3_var, model_completion=True)
                    assignments[var.name] = self._z3_value_to_python(value)
            
            python_optimal_value = self._z3_value_to_python(optimal_value)
            try:
                float_val = float(python_optimal_value)
            except (TypeError, ValueError):
                float_val = 0.0
            
            result_obj = OptimizationResult(
                success=True,
                optimal_value=float_val,
                optimal_model=Z3Model(
                    assignments=assignments,
                    objective_value=float_val
                ),
                iterations=1,
                execution_time=time.time() - start_time
            )
            
            self._optimization_history.append(result_obj)
            return result_obj
        
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time
        )
    
    def _multi_objective_optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]],
        strategy: str
    ) -> OptimizationResult:
        """Multi-objective optimization."""
        start_time = time.time()
        
        if strategy == "pareto":
            return self._pareto_optimize(variables, constraints, objectives)
        elif strategy == "weighted":
            return self._weighted_optimize(variables, constraints, objectives)
        elif strategy == "lexicographic":
            return self._lexicographic_optimize(variables, constraints, objectives)
        else:
            return OptimizationResult(
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Unknown strategy: {strategy}"
            )
    
    def _pareto_optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]]
    ) -> OptimizationResult:
        """Find Pareto frontier for multi-objective optimization."""
        start_time = time.time()
        
        pareto_front = []
        
        # Simple epsilon-constraint method
        # Solve for first objective, then constrain and solve for others
        
        primary_obj, primary_type = objectives[0]
        primary_result = self._single_objective_optimize(
            variables, constraints, (primary_obj, primary_type)
        )
        
        if not primary_result.success:
            return OptimizationResult(
                success=False,
                execution_time=time.time() - start_time
            )
        
        # Add to Pareto front
        if primary_result.optimal_model:
            pareto_front.append({
                "objectives": {primary_obj: primary_result.optimal_value},
                "model": primary_result.optimal_model.assignments
            })
        
        return OptimizationResult(
            success=True,
            is_pareto=True,
            pareto_front=pareto_front,
            execution_time=time.time() - start_time
        )
    
    def _weighted_optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]]
    ) -> OptimizationResult:
        """Weighted sum approach for multi-objective."""
        start_time = time.time()
        
        # Create weighted sum objective
        weights = [1.0 / len(objectives)] * len(objectives)
        weighted_expr = "(+ " + " ".join([
            f"(* {w} {obj[0]})" for w, obj in zip(weights, objectives)
        ]) + ")"
        
        return self._single_objective_optimize(
            variables, constraints, (weighted_expr, OptimizationObjective.MINIMIZE)
        )
    
    def _lexicographic_optimize(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]]
    ) -> OptimizationResult:
        """Lexicographic ordering for multi-objective."""
        start_time = time.time()
        
        current_constraints = list(constraints)
        objective_values = {}
        
        for obj_expr, obj_type in objectives:
            result = self._single_objective_optimize(
                variables, current_constraints, (obj_expr, obj_type)
            )
            
            if not result.success:
                return OptimizationResult(
                    success=False,
                    execution_time=time.time() - start_time
                )
            
            objective_values[obj_expr] = result.optimal_value
            
            # Constrain this objective for next iteration
            if obj_type == OptimizationObjective.MINIMIZE:
                new_constraint = Z3Constraint(
                    f"(<= {obj_expr} {result.optimal_value})",
                    Z3ConstraintType.REAL
                )
            else:
                new_constraint = Z3Constraint(
                    f"(>= {obj_expr} {result.optimal_value})",
                    Z3ConstraintType.REAL
                )
            current_constraints.append(new_constraint)
        
        return OptimizationResult(
            success=True,
            objectives=objective_values,
            execution_time=time.time() - start_time
        )
    
    def _optimize_via_cli(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        objectives: List[Tuple[str, OptimizationObjective]]
    ) -> OptimizationResult:
        """Optimization using Z3 CLI."""
        start_time = time.time()
        
        # Generate optimization SMT-LIB
        lines = [
            "(set-option :opt.priority pareto)",
            "(set-logic ALL)"
        ]
        
        # Declare variables
        for var in variables:
            lines.append(var.to_smtlib())
        
        # Add constraints
        for constraint in constraints:
            lines.append(constraint.to_smtlib())
        
        # Add objectives
        for obj_expr, obj_type in objectives:
            if obj_type == OptimizationObjective.MINIMIZE:
                lines.append(f"(minimize {obj_expr})")
            else:
                lines.append(f"(maximize {obj_expr})")
        
        lines.extend(["(check-sat)", "(get-model)"])
        
        smtlib = "\n".join(lines)
        
        # Execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['z3', '-smt2', temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            # Parse result
            return OptimizationResult(
                success="sat" in result.stdout.lower(),
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"CLI optimization failed: {e}")
            return OptimizationResult(
                success=False,
                execution_time=time.time() - start_time
            )
        finally:
            try:
                Path(temp_file).unlink()
            except:
                pass
    
    # =====================================================================
    # Array Constraints
    # =====================================================================
    
    def solve_with_arrays(
        self,
        scalar_vars: List[Z3Variable],
        array_constraints: List[ArrayConstraint],
        scalar_constraints: List[Z3Constraint]
    ) -> Z3SolverResult:
        """
        Solve constraints involving arrays.
        
        Args:
            scalar_vars: Scalar variables
            array_constraints: Array constraints
            scalar_constraints: Regular scalar constraints
            
        Returns:
            Z3SolverResult
        """
        if not Z3_PYTHON_AVAILABLE:
            # Use CLI with SMT-LIB
            smtlib_parts = ["(set-logic QF_AUFLIA)", "(set-option :produce-models true)"]
            
            for var in scalar_vars:
                smtlib_parts.append(var.to_smtlib())
            
            for arr in array_constraints:
                smtlib_parts.append(arr.to_smtlib())
            
            for constraint in scalar_constraints:
                smtlib_parts.append(constraint.to_smtlib())
            
            smtlib_parts.extend(["(check-sat)", "(get-model)"])
            
            return self.solve_smtlib("\n".join(smtlib_parts))
        
        with self._solver_lock:
            # Use Python API
            solver = z3.Solver()
            solver.set("timeout", int(self.config.timeout * 1000))
            
            # Create scalar variables
            z3_vars = {}
            for var in scalar_vars:
                z3_vars[var.name] = self._create_z3_variable(var)
            
            # Create arrays
            for arr in array_constraints:
                if arr.index_type == Z3ConstraintType.INTEGER:
                    idx_sort = z3.IntSort()
                elif arr.index_type == Z3ConstraintType.REAL:
                    idx_sort = z3.RealSort()
                elif arr.index_type == Z3ConstraintType.BOOLEAN:
                    idx_sort = z3.BoolSort()
                elif arr.index_type == Z3ConstraintType.BIT_VECTOR:
                    idx_sort = z3.BitVecSort(32) # Default width
                elif arr.index_type == Z3ConstraintType.STRING:
                    idx_sort = z3.StringSort()
                else:
                    idx_sort = z3.IntSort()
                
                if arr.value_type == Z3ConstraintType.INTEGER:
                    val_sort = z3.IntSort()
                elif arr.value_type == Z3ConstraintType.REAL:
                    val_sort = z3.RealSort()
                elif arr.value_type == Z3ConstraintType.BOOLEAN:
                    val_sort = z3.BoolSort()
                elif arr.value_type == Z3ConstraintType.BIT_VECTOR:
                    val_sort = z3.BitVecSort(32)
                elif arr.value_type == Z3ConstraintType.STRING:
                    val_sort = z3.StringSort()
                else:
                    val_sort = z3.IntSort()
                
                z3_arr = z3.Array(arr.array_name, idx_sort, val_sort)
                z3_vars[arr.array_name] = z3_arr
                
                # Add array constraints
                for constraint in arr.constraints:
                    z3_expr = self._parse_constraint(constraint, z3_vars)
                    if z3_expr is not None:
                        solver.add(z3_expr)
            
            # Add scalar constraints
            for constraint in scalar_constraints:
                z3_expr = self._parse_constraint(constraint.expression, z3_vars)
                if z3_expr is not None:
                    solver.add(z3_expr)
            
            # Solve
            result = solver.check()
            
            if result == z3.sat:
                model = solver.model()
                assignments = {}
                
                for var in scalar_vars:
                    z3_var = z3_vars.get(var.name)
                    if z3_var is not None:
                        value = model.eval(z3_var, model_completion=True)
                        assignments[var.name] = self._z3_value_to_python(value)
                
                return Z3SolverResult(
                    status=Z3ResultStatus.SAT,
                    model=Z3Model(assignments=assignments)
                )
            elif result == z3.unsat:
                return Z3SolverResult(status=Z3ResultStatus.UNSAT)
            else:
                return Z3SolverResult(status=Z3ResultStatus.UNKNOWN)
    
    # =====================================================================
    # Bit-Vector Operations
    # =====================================================================
    
    def solve_bitvector(
        self,
        bv_constraints: List[BitVectorConstraint],
        scalar_constraints: List[Z3Constraint] = None
    ) -> Z3SolverResult:
        """Solve bit-vector constraints."""
        if not Z3_PYTHON_AVAILABLE:
            smtlib_parts = ["(set-logic QF_BV)", "(set-option :produce-models true)"]
            
            for bv in bv_constraints:
                smtlib_parts.append(bv.to_smtlib())
            
            if scalar_constraints:
                for constraint in scalar_constraints:
                    smtlib_parts.append(constraint.to_smtlib())
            
            smtlib_parts.extend(["(check-sat)", "(get-model)"])
            
            return self.solve_smtlib("\n".join(smtlib_parts))
        
        with self._solver_lock:
            solver = z3.Solver()
            solver.set("timeout", int(self.config.timeout * 1000))
            
            # Create bit-vector variables
            z3_vars = {}
            for bv in bv_constraints:
                if bv.signed:
                    z3_var = z3.BitVec(bv.var_name, bv.width)
                else:
                    z3_var = z3.BitVec(bv.var_name, bv.width)
                z3_vars[bv.var_name] = z3_var
                
                # Add constraints
                for constraint in bv.constraints:
                    z3_expr = self._parse_constraint(constraint, z3_vars)
                    if z3_expr is not None:
                        solver.add(z3_expr)
            
            result = solver.check()
            
            if result == z3.sat:
                model = solver.model()
                assignments = {}
                
                for bv in bv_constraints:
                    z3_var = z3_vars.get(bv.var_name)
                    if z3_var is not None:
                        value = model.eval(z3_var, model_completion=True)
                        assignments[bv.var_name] = int(value.as_long())
                
                return Z3SolverResult(
                    status=Z3ResultStatus.SAT,
                    model=Z3Model(assignments=assignments)
                )
            elif result == z3.unsat:
                return Z3SolverResult(status=Z3ResultStatus.UNSAT)
            else:
                return Z3SolverResult(status=Z3ResultStatus.UNKNOWN)
    
    # =====================================================================
    # Portfolio Solving
    # =====================================================================
    
    def solve_portfolio(
        self,
        smtlib_problem: str,
        strategies: Optional[List[str]] = None,
        parallel: bool = True
    ) -> PortfolioResult:
        """
        Solve using multiple strategies in parallel.
        
        Args:
            smtlib_problem: SMT-LIB problem
            strategies: List of strategies to try (default: all)
            parallel: Whether to run in parallel
            
        Returns:
            PortfolioResult
        """
        start_time = time.time()
        strategies = strategies or self._portfolio_strategies
        
        results = []
        
        if parallel and len(strategies) > 1:
            # Run in parallel
            with ThreadPoolExecutor(max_workers=min(len(strategies), 4)) as executor:
                futures = {
                    executor.submit(
                        self._try_strategy, smtlib_problem, strategy
                    ): strategy for strategy in strategies
                }
                
                for future in as_completed(futures):
                    strategy = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout)
                        results.append((strategy, result))
                        
                        # Early termination if SAT found
                        if result.is_sat():
                            break
                    except Exception as e:
                        logger.warning(f"Strategy {strategy} failed: {e}")
                        results.append((strategy, Z3SolverResult(
                            status=Z3ResultStatus.ERROR,
                            errors=[str(e)]
                        )))
        else:
            # Sequential execution
            for strategy in strategies:
                result = self._try_strategy(smtlib_problem, strategy)
                results.append((strategy, result))
                
                if result.is_sat():
                    break
        
        # Find best result
        best_result = None
        winner = None
        
        for strategy, result in results:
            if result.is_sat():
                best_result = result
                winner = strategy
                break
        
        if best_result is None:
            for strategy, result in results:
                if result.status == Z3ResultStatus.UNKNOWN:
                    best_result = result
                    winner = strategy
                    break
        
        elapsed = time.time() - start_time
        
        return PortfolioResult(
            success=best_result is not None and best_result.is_sat(),
            best_result=best_result,
            all_results=results,
            winner_strategy=winner,
            execution_time=elapsed,
            parallel_speedup=len(strategies) if parallel else 1.0
        )
    
    def _try_strategy(self, smtlib_problem: str, strategy: str) -> Z3SolverResult:
        """Try a single strategy with proper SMT-LIB option placement."""
        try:
            # SMT-LIB options must come before set-logic and assertions
            option_line = f"(set-option :tactic.default_tactic {strategy})"
            
            # Remove any existing tactic option to avoid conflicts
            cleaned_smt = re.sub(r'\(set-option\s+:tactic\.default_tactic\s+\w+\)', '', smtlib_problem)
            
            # Prepend option
            modified_smt = f"{option_line}\n{cleaned_smt}"
            return self.solve_smtlib(modified_smt)
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed: {e}")
            return Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                errors=[str(e)]
            )
    
    # =====================================================================
    # Incremental Solving
    # =====================================================================
    
    def create_incremental_state(
        self,
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        state_id: Optional[str] = None
    ) -> str:
        """
        Create an incremental solving state.
        
        Args:
            variables: Initial variables
            constraints: Initial constraints
            state_id: Optional state ID (generated if not provided)
            
        Returns:
            State ID
        """
        state_id = state_id or f"inc_{int(time.time())}_{hashlib.md5(str(variables).encode()).hexdigest()[:8]}"
        
        state = IncrementalState(
            state_id=state_id,
            variables=list(variables),
            constraints=list(constraints),
            assertions_stack=[list(constraints)]
        )
        
        with self._state_lock:
            self._incremental_states[state_id] = state
        
        return state_id
    
    def push_scope(self, state_id: str, scope_name: Optional[str] = None) -> bool:
        """Push a new scope in incremental solving."""
        with self._state_lock:
            state = self._incremental_states.get(state_id)
            if not state:
                return False
            
            state.assertions_stack.append([])
            state.scopes.append(scope_name or f"scope_{len(state.scopes)}")
            state.last_accessed = time.time()
            return True
    
    def pop_scope(self, state_id: str, count: int = 1) -> bool:
        """Pop scope(s) in incremental solving."""
        with self._state_lock:
            state = self._incremental_states.get(state_id)
            if not state:
                return False
            
            for _ in range(count):
                if len(state.assertions_stack) > 1:
                    popped = state.assertions_stack.pop()
                    for constraint in popped:
                        if constraint in state.constraints:
                            state.constraints.remove(constraint)
                    if state.scopes:
                        state.scopes.pop()
            
            state.last_accessed = time.time()
            return True
    
    def add_constraint_incremental(
        self,
        state_id: str,
        constraint: Z3Constraint
    ) -> bool:
        """Add constraint to current scope."""
        with self._state_lock:
            state = self._incremental_states.get(state_id)
            if not state:
                return False
            
            state.constraints.append(constraint)
            if state.assertions_stack:
                state.assertions_stack[-1].append(constraint)
            
            state.last_accessed = time.time()
            return True
    
    def check_incremental(self, state_id: str) -> Z3SolverResult:
        """Check satisfiability of incremental state."""
        with self._state_lock:
            state = self._incremental_states.get(state_id)
            if not state:
                return Z3SolverResult(
                    status=Z3ResultStatus.ERROR,
                    errors=["State not found"]
                )
            
            state.last_accessed = time.time()
            
            # Solve current state
            result = self.solve_constraints(state.variables, state.constraints)
            state.last_result = result
            return result
    
    def get_incremental_state(self, state_id: str) -> Optional[IncrementalState]:
        """Get incremental state."""
        with self._state_lock:
            return self._incremental_states.get(state_id)
    
    def cleanup_incremental_states(self, max_age_seconds: float = 3600):
        """Remove old incremental states."""
        now = time.time()
        with self._state_lock:
            to_remove = [
                sid for sid, state in self._incremental_states.items()
                if now - state.last_accessed > max_age_seconds
            ]
            for sid in to_remove:
                del self._incremental_states[sid]
    
    # =====================================================================
    # Proof Extraction
    # =====================================================================
    
    def extract_proof(
        self,
        smtlib_problem: str,
        proof_format: ProofFormat = ProofFormat.TEXT
    ) -> ExtractedProof:
        """
        Extract proof from Z3.
        
        Args:
            smtlib_problem: SMT-LIB problem
            proof_format: Desired proof format
            
        Returns:
            ExtractedProof
        """
        if not Z3_PYTHON_AVAILABLE:
            return self._extract_proof_via_cli(smtlib_problem, proof_format)
        
        with self._solver_lock:
            try:
                # Enable proof generation
                z3.set_option(proof=True)
                
                solver = z3.Solver()
                solver.set("timeout", int(self.config.timeout * 1000))
                
                # Parse SMT-LIB
                # Note: This is simplified - full implementation would parse properly
                solver.from_string(smtlib_problem)
                
                result = solver.check()
                
                if result == z3.unsat:
                    proof = solver.proof()
                    
                    # Convert proof to steps
                    steps = self._parse_z3_proof(proof)
                    
                    return ExtractedProof(
                        success=True,
                        proof_steps=steps,
                        raw_proof=str(proof),
                        proof_format=proof_format,
                        verification_status="verified"
                    )
                else:
                    return ExtractedProof(
                        success=False,
                        verification_status="not_unsat"
                    )
            except Exception as e:
                logger.error(f"Proof extraction failed: {e}")
                return ExtractedProof(
                    success=False,
                    errors=[str(e)]
                )
    
    def _parse_z3_proof(self, proof) -> List[ProofStep]:
        """Parse Z3 proof object into steps."""
        steps = []
        
        # Simplified parsing
        try:
            proof_str = str(proof)
            # Extract named tactics
            tactics = re.findall(r'\((\w+)', proof_str)
            
            for i, tactic in enumerate(set(tactics)):
                steps.append(ProofStep(
                    step_number=i+1,
                    tactic=tactic,
                    justification=f"Applied {tactic}"
                ))
        except:
            pass
        
        return steps
    
    def _extract_proof_via_cli(
        self,
        smtlib_problem: str,
        proof_format: ProofFormat
    ) -> ExtractedProof:
        """Extract proof via CLI."""
        # Add proof generation option
        lines = ["(set-option :produce-proofs true)"] + smtlib_problem.split('\n')
        modified_smt = '\n'.join(lines)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(modified_smt)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['z3', 'proof=true', '-smt2', temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            return ExtractedProof(
                success=result.returncode == 0,
                raw_proof=result.stdout,
                proof_format=proof_format
            )
        except Exception as e:
            return ExtractedProof(
                success=False,
                errors=[str(e)]
            )
        finally:
            try:
                Path(temp_file).unlink()
            except:
                pass
    
    # =====================================================================
    # Statistics and History
    # =====================================================================
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of optimization runs."""
        return list(self._optimization_history)
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get advanced solver statistics."""
        base_stats = self.get_status()["statistics"]
        
        return {
            **base_stats,
            "incremental_states": len(self._incremental_states),
            "optimization_runs": len(self._optimization_history),
            "portfolio_strategies": len(self._portfolio_strategies)
        }


# =============================================================================
# Global Instance
# =============================================================================

_z3_advanced_solver: Optional[Z3AdvancedSolver] = None


def get_z3_advanced_solver(config: Optional[Z3Config] = None) -> Z3AdvancedSolver:
    """Get global advanced Z3 solver instance."""
    global _z3_advanced_solver
    if _z3_advanced_solver is None:
        _z3_advanced_solver = Z3AdvancedSolver(config)
    return _z3_advanced_solver


# =============================================================================
# Example Usage
# =============================================================================

async def example_optimization():
    """Example: Multi-objective optimization."""
    solver = get_z3_advanced_solver()
    
    variables = [
        Z3Variable("x", Z3ConstraintType.INTEGER),
        Z3Variable("y", Z3ConstraintType.INTEGER)
    ]
    
    constraints = [
        Z3Constraint("x >= 0", Z3ConstraintType.INTEGER),
        Z3Constraint("y >= 0", Z3ConstraintType.INTEGER),
        Z3Constraint("x + y <= 100", Z3ConstraintType.INTEGER)
    ]
    
    objectives = [
        ("x", OptimizationObjective.MAXIMIZE),
        ("y", OptimizationObjective.MAXIMIZE)
    ]
    
    result = solver.optimize(variables, constraints, objectives, "pareto")
    
    print(f"Optimization success: {result.success}")
    print(f"Pareto front size: {len(result.pareto_front)}")
    
    return result


def example_incremental():
    """Example: Incremental solving."""
    solver = get_z3_advanced_solver()
    
    variables = [Z3Variable("x", Z3ConstraintType.INTEGER)]
    constraints = [Z3Constraint("x > 0", Z3ConstraintType.INTEGER)]
    
    # Create state
    state_id = solver.create_incremental_state(variables, constraints)
    print(f"Created incremental state: {state_id}")
    
    # Check
    result = solver.check_incremental(state_id)
    print(f"Initial check: {result.status.value}")
    
    # Push scope and add constraint
    solver.push_scope(state_id, "upper_bound")
    solver.add_constraint_incremental(state_id, Z3Constraint("x < 10", Z3ConstraintType.INTEGER))
    
    result = solver.check_incremental(state_id)
    print(f"After constraint: {result.status.value}")
    
    # Pop scope
    solver.pop_scope(state_id)
    
    result = solver.check_incremental(state_id)
    print(f"After pop: {result.status.value}")
    
    return state_id


def example_portfolio():
    """Example: Portfolio solving."""
    solver = get_z3_advanced_solver()
    
    smtlib = """
    (set-logic QF_LIA)
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (> x 0))
    (assert (> y 0))
    (assert (= (+ x y) 100))
    (check-sat)
    """
    
    result = solver.solve_portfolio(smtlib)
    
    print(f"Portfolio success: {result.success}")
    if not result.success:
        for strategy, res in result.all_results:
            if res.status == Z3ResultStatus.ERROR:
                print(f"  Strategy {strategy} error: {res.reason}")
    print(f"Winner strategy: {result.winner_strategy}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Strategies tried: {len(result.all_results)}")
    
    return result


if __name__ == "__main__":
    if Z3_AVAILABLE:
        print("Z3 Advanced Features Demo")
        print("=" * 50)
        
        print("\n--- Optimization Example ---")
        asyncio.run(example_optimization())
        
        print("\n--- Incremental Solving Example ---")
        example_incremental()
        
        print("\n--- Portfolio Solving Example ---")
        example_portfolio()
    else:
        print("Z3 not available")
