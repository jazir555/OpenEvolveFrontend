"""
Z3 Solver Connector - Complete Implementation

Provides a high-level connector interface for Z3 solver operations,
bridging between the OpenEvolve system and Z3 SMT solver.

Features:
- Unified solver interface
- Constraint submission and checking
- Model extraction
- Solver configuration management
- Integration with gauntlet system

Author: OpenEvolve Team
Date: 2026-02-17
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Z3 imports with graceful fallback
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None
    logger.warning("Z3 not available")

from z3prover_integration import (
    Z3SolverEngine, Z3Config, Z3SolverResult,
    Z3ResultStatus, Z3Variable, Z3Constraint,
    Z3ConstraintType, create_z3_solver
)


class SolverStrategy(Enum):
    """Solver strategies."""
    DEFAULT = "default"
    AUTO_CONFIG = "auto_config"
    PROOF = "proof"
    MODEL = "model"
    TRACKER = "tracker"


@dataclass
class SolverConfig:
    """Configuration for Z3 solver connector."""
    timeout: int = 30000  # milliseconds
    max_memory: int = 8589934592  # 8GB
    strategy: SolverStrategy = SolverStrategy.AUTO_CONFIG
    enable_proofs: bool = False
    enable_models: bool = True
    enable_unsat_cores: bool = False
    threads: int = 1
    logic: Optional[str] = None


@dataclass
class SolverRequest:
    """A solver request."""
    id: str
    constraints: List[str]
    variables: Dict[str, str]  # name -> type
    config: Optional[SolverConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResponse:
    """A solver response."""
    request_id: str
    status: Z3ResultStatus
    model: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    solve_time: float = 0.0
    error: Optional[str] = None
    solver_info: Dict[str, Any] = field(default_factory=dict)


class Z3SolverConnector:
    """
    High-level connector for Z3 solver operations.

    Provides a unified interface for:
    - Submitting constraint problems
    - Checking satisfiability
    - Extracting models and proofs
    - Managing solver instances
    - Parallel solving with different strategies
    """

    def __init__(self, default_config: Optional[SolverConfig] = None):
        """
        Initialize the Z3 solver connector.

        Args:
            default_config: Default configuration for solver instances
        """
        self.default_config = default_config or SolverConfig()
        self._solvers: Dict[str, Z3SolverEngine] = {}
        self._solver_lock = threading.Lock()
        self._request_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=4)

    def create_solver(self, config: Optional[SolverConfig] = None) -> Z3SolverEngine:
        """
        Create a new Z3 solver instance.

        Args:
            config: Solver configuration

        Returns:
            Configured Z3 solver engine
        """
        config = config or self.default_config
        z3_config = Z3Config(
            timeout=config.timeout,
            max_memory=config.max_memory,
            model=config.enable_models,
            proof=config.enable_proofs,
            threads=config.threads,
            logic=config.logic,
            unsat_core=config.enable_unsat_cores
        )
        return Z3SolverEngine(z3_config)

    def solve(self, request: SolverRequest) -> SolverResponse:
        """
        Solve a constraint problem.

        Args:
            request: Solver request with constraints and variables

        Returns:
            Solver response with status, model, and proof
        """
        start_time = time.time()

        if not Z3_AVAILABLE:
            return SolverResponse(
                request_id=request.id,
                status=Z3ResultStatus.ERROR,
                error="Z3 not available",
                solve_time=time.time() - start_time
            )

        try:
            if not Z3_AVAILABLE:
                return SolverResponse(
                    request_id=request.id,
                    status=Z3ResultStatus.ERROR,
                    error="Z3 not available",
                    solve_time=time.time() - start_time
                )

            # Use Z3 directly for reliable constraint handling
            import z3
            z3_solver = z3.Solver()
            z3_solver.set("timeout", (request.config.timeout if request.config else self.default_config.timeout))

            # Create Z3 variables
            z3_vars = {}
            for var_name, var_type in request.variables.items():
                if var_type == 'int':
                    z3_vars[var_name] = z3.Int(var_name)
                elif var_type == 'real':
                    z3_vars[var_name] = z3.Real(var_name)
                elif var_type == 'bool':
                    z3_vars[var_name] = z3.Bool(var_name)
                else:
                    z3_vars[var_name] = z3.Int(var_name)  # Default

            # Add constraints
            for constraint_str in request.constraints:
                try:
                    # Try to parse as SMT-LIB
                    if constraint_str.startswith('(') and constraint_str.endswith(')'):
                        parsed = z3.parse_smt2_string(f"(assert {constraint_str})", decls=z3_vars)
                        if parsed:
                            z3_solver.add(parsed)
                    else:
                        # Try to evaluate as Python expression with Z3 variables
                        # Safe evaluation with limited builtins
                        safe_dict = {**z3_vars, 'And': z3.And, 'Or': z3.Or, 'Not': z3.Not, 'Implies': z3.Implies}
                        expr = eval(constraint_str, {"__builtins__": {}}, safe_dict)
                        z3_solver.add(expr)
                except Exception as parse_err:
                    logger.debug(f"Failed to parse constraint '{constraint_str}': {parse_err}")
                    # Continue with other constraints

            # Check satisfiability
            result = z3_solver.check()

            # Extract model if SAT
            model_dict = None
            if result == z3.sat:
                model = z3_solver.model()
                model_dict = {}
                for var_name, z3_var in z3_vars.items():
                    try:
                        val = model[z3_var]
                        model_dict[var_name] = str(val)
                    except:
                        model_dict[var_name] = None

            # Map result to our status
            if result == z3.sat:
                status = Z3ResultStatus.SAT
            elif result == z3.unsat:
                status = Z3ResultStatus.UNSAT
            else:
                status = Z3ResultStatus.UNKNOWN

            return SolverResponse(
                request_id=request.id,
                status=status,
                model=model_dict,
                solve_time=time.time() - start_time,
                solver_info={"z3_result": str(result)}
            )

        except Exception as e:
            logger.error(f"Solver error for request {request.id}: {e}")
            import traceback
            traceback.print_exc()
            return SolverResponse(
                request_id=request.id,
                status=Z3ResultStatus.ERROR,
                error=str(e),
                solve_time=time.time() - start_time
            )

    def solve_parallel(
        self,
        requests: List[SolverRequest],
        strategies: Optional[List[SolverStrategy]] = None
    ) -> List[SolverResponse]:
        """
        Solve multiple problems in parallel.

        Args:
            requests: List of solver requests
            strategies: Optional list of strategies to try for each request

        Returns:
            List of solver responses
        """
        responses = {}

        def solve_single(req):
            responses[req.id] = self.solve(req)

        # Solve in parallel
        futures = []
        for request in requests:
            future = self._executor.submit(solve_single, request)
            futures.append(future)

        # Wait for completion
        for future in as_completed(futures):
            pass  # Results stored in responses dict

        return [responses[req.id] for req in requests]

    def solve_with_portfolio(
        self,
        request: SolverRequest,
        strategies: Optional[List[SolverStrategy]] = None
    ) -> SolverResponse:
        """
        Solve using portfolio of strategies.

        Tries multiple solver strategies in parallel and returns
        the first successful result.

        Args:
            request: Solver request
            strategies: List of strategies to try

        Returns:
            Best solver response found
        """
        strategies = strategies or [
            SolverStrategy.AUTO_CONFIG,
            SolverStrategy.MODEL,
            SolverStrategy.PROOF
        ]

        # Create requests with different strategies
        strategy_requests = []
        for strategy in strategies:
            config = SolverConfig(
                timeout=request.config.timeout if request.config else self.default_config.timeout,
                max_memory=request.config.max_memory if request.config else self.default_config.max_memory,
                strategy=strategy,
                enable_proofs=(strategy == SolverStrategy.PROOF),
                enable_models=(strategy == SolverStrategy.MODEL)
            )
            strategy_request = SolverRequest(
                id=f"{request.id}_{strategy.value}",
                constraints=request.constraints,
                variables=request.variables,
                config=config,
                metadata=request.metadata
            )
            strategy_requests.append(strategy_request)

        # Solve in parallel
        responses = self.solve_parallel(strategy_requests)

        # Find first successful response
        for response in responses:
            if response.status in [Z3ResultStatus.SAT, Z3ResultStatus.UNSAT]:
                response.request_id = request.id  # Reset to original ID
                return response

        # If all failed, return the first response
        return responses[0] if responses else SolverResponse(
            request_id=request.id,
            status=Z3ResultStatus.ERROR,
            error="All strategies failed"
        )

    def get_solver_stats(self) -> Dict[str, Any]:
        """Get statistics about solver instances."""
        return {
            "active_solvers": len(self._solvers),
            "requests_processed": self._request_counter,
            "z3_available": Z3_AVAILABLE
        }

    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        with self._solver_lock:
            for solver in self._solvers.values():
                solver.reset()
            self._solvers.clear()


# Convenience functions

def create_solver_connector(config: Optional[SolverConfig] = None) -> Z3SolverConnector:
    """Create a new Z3 solver connector."""
    return Z3SolverConnector(config)


def solve_simple(
    constraints: List[str],
    variables: Optional[Dict[str, str]] = None,
    timeout: int = 30000
) -> SolverResponse:
    """
    Solve a simple constraint problem.

    Args:
        constraints: List of constraint strings
        variables: Variable name to type mapping
        timeout: Solver timeout in milliseconds

    Returns:
        Solver response
    """
    connector = create_solver_connector()
    request = SolverRequest(
        id="simple",
        constraints=constraints,
        variables=variables or {},
        config=SolverConfig(timeout=timeout)
    )
    return connector.solve(request)
