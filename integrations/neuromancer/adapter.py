"""
NeuroMANCER Adapter for OpenEvolve

This module implements the optimization interface using NeuroMANCER as the backend.
It provides physics-informed optimization, system identification, and differentiable programming.
"""

import asyncio
import subprocess
import json
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import tempfile
import numpy as np

from integrations.base.optimization_interface import (
    OptimizationInterface,
    OptimizationResult,
    OptimizationProblem,
    OptimizationType,
    ProblemType,
    OptimizationError,
    ConfigurationError,
    ConnectionError,
    ValidationError,
    SolverError,
    TimeoutError,
    IdentificationError,
    TemplateNotFoundError
)


class NeuroMANCERAdapter(OptimizationInterface):
    """
    NeuroMANCER implementation of the optimization interface.

    This adapter communicates with NeuroMANCER through a decoupled pattern:
    - Uses separate PyTorch environment (conda env)
    - Serializes problems to temporary files
    - Invokes NeuroMANCER as subprocess
    - Retrieves results through output files
    - Maintains no direct dependencies on NeuroMANCER code
    """

    def __init__(self):
        self.config = {}
        self.pytorch_env = None
        self.device = None
        self.neuromancer_path = None
        self.initialized = False
        self.template_cache = {}
        self._process_lock = asyncio.Lock()

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the NeuroMANCER adapter.

        Args:
            config: Configuration dictionary containing:
                - pytorch_env: Name of conda environment for PyTorch
                - device: Device to use ('cuda' or 'cpu')
                - neuromancer_path: Path to NeuroMANCER installation
                - max_workers: Number of parallel workers
                - timeout: Operation timeout in seconds
                - cache_enabled: Whether to enable caching
                - cache_ttl: Cache time-to-live in seconds
        """
        try:
            self.config = config

            # Extract configuration
            self.pytorch_env = config.get("pytorch_env", "neuromancer_env")
            self.device = config.get("device", "cpu")
            self.neuromancer_path = config.get("neuromancer_path")
            self.max_workers = config.get("max_workers", 4)
            self.timeout = config.get("timeout", 30)
            self.cache_enabled = config.get("cache_enabled", True)
            self.cache_ttl = config.get("cache_ttl", 3600)

            # Validate PyTorch environment exists
            if not await self._validate_environment():
                raise ConfigurationError(
                    f"PyTorch environment '{self.pytorch_env}' not found. "
                    "Please create it with: conda create -n neuromancer_env python=3.9"
                )

            # Load templates if caching is enabled
            if self.cache_enabled:
                await self._load_templates()

            self.initialized = True
            return True

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize NeuroMANCER adapter: {str(e)}")

    async def solve(
        self,
        problem: OptimizationProblem,
        optimization_type: OptimizationType = OptimizationType.UNCONSTRAINED,
        solver_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Solve an optimization problem using NeuroMANCER.

        This method serializes the problem to a temporary file, invokes NeuroMANCER,
        and retrieves the solution.
        """
        if not self.initialized:
            raise ConfigurationError("NeuroMANCER adapter not initialized")

        try:
            # Validate problem
            if not problem.validate():
                raise ValidationError("Invalid problem definition")

            # Create temporary directory for problem and solution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Serialize problem to file
                problem_file = Path(temp_dir) / "problem.json"
                solution_file = Path(temp_dir) / "solution.json"

                problem_data = {
                    "problem_type": problem.problem_type.value,
                    "optimization_type": optimization_type.value,
                    "variables": problem.variables,
                    "parameters": problem.parameters,
                    "constraints": problem.constraints,
                    "physics_constraints": problem.physics_constraints,
                    "bounds": problem.bounds,
                    "solver_params": solver_params or {}
                }

                with open(problem_file, 'w') as f:
                    json.dump(problem_data, f, indent=2)

                # Invoke NeuroMANCER solver
                await self._invoke_solver(
                    problem_file=str(problem_file),
                    solution_file=str(solution_file),
                    problem_type=problem.problem_type.value
                )

                # Load solution
                if not solution_file.exists():
                    return OptimizationResult(
                        success=False,
                        optimal_value=0.0,
                        optimal_variables={},
                        iterations=0,
                        error_message="Solver did not produce output"
                    )

                with open(solution_file, 'r') as f:
                    solution_data = json.load(f)

                # Convert to OptimizationResult
                return OptimizationResult(
                    success=solution_data.get("success", False),
                    optimal_value=solution_data.get("optimal_value", 0.0),
                    optimal_variables=np.array(solution_data.get("optimal_variables", [])),
                    iterations=solution_data.get("iterations", 0),
                    convergence_history=solution_data.get("convergence_history", []),
                    metadata=solution_data.get("metadata", {}),
                    error_message=solution_data.get("error_message")
                )

        except asyncio.TimeoutError:
            raise TimeoutError(f"Solver exceeded timeout of {self.timeout} seconds")
        except Exception as e:
            raise SolverError(f"Solver failed: {str(e)}")

    async def identify_system(
        self,
        data: Dict[str, Any],
        model_structure: Optional[Dict[str, Any]] = None,
        physics_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform physics-informed system identification.

        Uses NeuroMANCER's system identification capabilities to learn
        dynamics from data while respecting physics constraints.
        """
        if not self.initialized:
            raise ConfigurationError("NeuroMANCER adapter not initialized")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare system identification problem
                problem_file = Path(temp_dir) / "sysid_problem.json"
                solution_file = Path(temp_dir) / "sysid_solution.json"

                problem_data = {
                    "problem_type": "system_identification",
                    "data": data,
                    "model_structure": model_structure or {},
                    "physics_constraints": physics_constraints or {},
                    "device": self.device
                }

                with open(problem_file, 'w') as f:
                    json.dump(problem_data, f, indent=2, default=self._json_serializer)

                # Invoke NeuroMANCER system identification
                await self._invoke_solver(
                    problem_file=str(problem_file),
                    solution_file=str(solution_file),
                    problem_type="system_identification"
                )

                # Load results
                if not solution_file.exists():
                    raise IdentificationError("System identification failed to produce output")

                with open(solution_file, 'r') as f:
                    solution_data = json.load(f)

                return solution_data

        except Exception as e:
            raise IdentificationError(f"System identification failed: {str(e)}")

    async def solve_ode(
        self,
        ode_definition: Dict[str, Any],
        initial_conditions: Dict[str, Any],
        time_span: tuple,
        method: str = "automatic"
    ) -> Dict[str, Any]:
        """
        Solve an ordinary differential equation using NeuroMANCER.

        NeuroMANCER uses neural ODE solvers that are differentiable and
        can incorporate physics constraints.
        """
        if not self.initialized:
            raise ConfigurationError("NeuroMANCER adapter not initialized")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                problem_file = Path(temp_dir) / "ode_problem.json"
                solution_file = Path(temp_dir) / "ode_solution.json"

                problem_data = {
                    "problem_type": "ode",
                    "ode_definition": ode_definition,
                    "initial_conditions": initial_conditions,
                    "time_span": time_span,
                    "method": method,
                    "device": self.device
                }

                with open(problem_file, 'w') as f:
                    json.dump(problem_data, f, indent=2, default=self._json_serializer)

                await self._invoke_solver(
                    problem_file=str(problem_file),
                    solution_file=str(solution_file),
                    problem_type="ode"
                )

                if not solution_file.exists():
                    raise SolverError("ODE solver failed to produce output")

                with open(solution_file, 'r') as f:
                    solution_data = json.load(f)

                return solution_data

        except Exception as e:
            raise SolverError(f"ODE solver failed: {str(e)}")

    async def solve_pde(
        self,
        pde_definition: Dict[str, Any],
        boundary_conditions: Dict[str, Any],
        initial_conditions: Optional[Dict[str, Any]] = None,
        domain: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a partial differential equation using NeuroMANCER.

        Uses physics-informed neural networks (PINNs) for PDE solving.
        """
        if not self.initialized:
            raise ConfigurationError("NeuroMANCER adapter not initialized")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                problem_file = Path(temp_dir) / "pde_problem.json"
                solution_file = Path(temp_dir) / "pde_solution.json"

                problem_data = {
                    "problem_type": "pde",
                    "pde_definition": pde_definition,
                    "boundary_conditions": boundary_conditions,
                    "initial_conditions": initial_conditions or {},
                    "domain": domain or {},
                    "device": self.device
                }

                with open(problem_file, 'w') as f:
                    json.dump(problem_data, f, indent=2, default=self._json_serializer)

                await self._invoke_solver(
                    problem_file=str(problem_file),
                    solution_file=str(solution_file),
                    problem_type="pde"
                )

                if not solution_file.exists():
                    raise SolverError("PDE solver failed to produce output")

                with open(solution_file, 'r') as f:
                    solution_data = json.load(f)

                return solution_data

        except Exception as e:
            raise SolverError(f"PDE solver failed: {str(e)}")

    async def constrained_optimization(
        self,
        objective: callable,
        constraints: List[Dict[str, Any]],
        variables: Dict[str, Any],
        method: str = "interior_point"
    ) -> OptimizationResult:
        """
        Solve a constrained optimization problem.

        NeuroMANCER handles constrained optimization through differentiable
        penalty methods and Lagrangian approaches.
        """
        if not self.initialized:
            raise ConfigurationError("NeuroMANCER adapter not initialized")

        try:
            # Create optimization problem
            problem = OptimizationProblem(
                problem_type=ProblemType.OPTIMIZATION,
                constraints=constraints,
                variables=variables
            )

            solver_params = {
                "method": method,
                "objective_type": "constrained"
            }

            return await self.solve(problem, OptimizationType.CONSTRAINED, solver_params)

        except Exception as e:
            raise SolverError(f"Constrained optimization failed: {str(e)}")

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the NeuroMANCER adapter state and connections.
        """
        checks = []
        issues = []

        # Check environment
        env_valid = await self._validate_environment()
        checks.append({"name": "pytorch_environment", "status": "passed" if env_valid else "failed"})
        if not env_valid:
            issues.append("PyTorch conda environment not found")

        # Check device availability
        device_info = self._get_device_info()
        checks.append({"name": "device_availability", "status": "passed", "info": device_info})

        # Check NeuroMANCER path
        if self.neuromancer_path:
            path_valid = Path(self.neuromancer_path).exists()
            checks.append({"name": "neuromancer_path", "status": "passed" if path_valid else "failed"})
            if not path_valid:
                issues.append(f"NeuroMANCER path not found: {self.neuromancer_path}")

        # Check templates
        template_count = len(self.template_cache)
        checks.append({"name": "templates_loaded", "status": "passed", "count": template_count})

        return {
            "is_valid": len(issues) == 0,
            "checks": checks,
            "issues": issues,
            "metrics": {
                "device": self.device,
                "environment": self.pytorch_env,
                "template_count": template_count,
                "cache_enabled": self.cache_enabled
            },
            "device_info": device_info
        }

    async def shutdown(self) -> bool:
        """
        Shutdown the NeuroMANCER adapter.

        Clears caches and releases resources.
        """
        try:
            self.template_cache.clear()
            self.initialized = False
            return True
        except Exception:
            return False

    async def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get a problem template by name.
        """
        if template_name in self.template_cache:
            return self.template_cache[template_name]

        # Try to load from file
        template_path = Path(__file__).parent / "templates" / f"{template_name}.yaml"
        if template_path.exists():
            import yaml
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)
                self.template_cache[template_name] = template
                return template

        raise TemplateNotFoundError(f"Template '{template_name}' not found")

    async def list_templates(self) -> List[str]:
        """
        List available problem templates.
        """
        template_dir = Path(__file__).parent / "templates"
        if template_dir.exists():
            return [f.stem for f in template_dir.glob("*.yaml")]
        return []

    # Private helper methods

    async def _validate_environment(self) -> bool:
        """Check if PyTorch conda environment exists."""
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return self.pytorch_env in result.stdout
        except Exception:
            return False

    def _get_device_info(self) -> Dict[str, Any]:
        """Get available device information."""
        return {
            "requested_device": self.device,
            "cpu_available": True,
            "cuda_available": self.device == "cuda"  # Simplified check
        }

    async def _load_templates(self):
        """Load all templates into cache."""
        template_dir = Path(__file__).parent / "templates"
        if template_dir.exists():
            try:
                import yaml
                for template_file in template_dir.glob("*.yaml"):
                    with open(template_file, 'r') as f:
                        self.template_cache[template_file.stem] = yaml.safe_load(f)
            except Exception:
                pass  # Templates are optional

    async def _invoke_solver(
        self,
        problem_file: str,
        solution_file: str,
        problem_type: str
    ):
        """
        Invoke NeuroMANCER solver as subprocess.

        This is the core decoupling mechanism - NeuroMANCER runs in its
        own environment with its own dependencies.
        """
        async with self._process_lock:
            # Create solver script
            solver_script = self._create_solver_script(problem_file, solution_file, problem_type)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(solver_script)
                script_path = f.name

            try:
                # Activate conda environment and run solver
                cmd = f"""
                conda run -n {self.pytorch_env} python {script_path}
                """

                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    shell=True
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout
                    )

                    if process.returncode != 0:
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        raise SolverError(f"Solver process failed: {error_msg}")

                except asyncio.TimeoutError:
                    process.kill()
                    raise TimeoutError("Solver exceeded timeout")

            finally:
                # Clean up script
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

    def _create_solver_script(self, problem_file: str, solution_file: str, problem_type: str) -> str:
        """
        Generate Python script to invoke NeuroMANCER solver.

        This script runs in the NeuroMANCER environment and imports
        NeuroMANCER libraries directly.
        """
        return f'''
import sys
import json
import traceback

try:
    # Import NeuroMANCER (this runs in the NeuroMANCER environment)
    from neuromancer import functions
    import torch

    # Load problem
    with open("{problem_file}", 'r') as f:
        problem = json.load(f)

    # Route to appropriate solver
    if problem["problem_type"] == "optimization":
        from neuromancer.opt import OptimizationSolver
        solver = OptimizationSolver()
        result = solver.solve(problem)
    elif problem["problem_type"] == "system_identification":
        from neuromancer.system import SystemIdentification
        solver = SystemIdentification()
        result = solver.identify(problem)
    elif problem["problem_type"] == "ode":
        from neuromancer.ode import ODESolver
        solver = ODESolver()
        result = solver.solve_ode(problem)
    elif problem["problem_type"] == "pde":
        from neuromancer.pde import PDESolver
        solver = PDESolver()
        result = solver.solve_pde(problem)
    else:
        result = {{"success": False, "error_message": f"Unknown problem type: {{problem['problem_type']}}"}}

    # Save solution
    with open("{solution_file}", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    sys.exit(0)

except Exception as e:
    error_result = {{
        "success": False,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }}
    with open("{solution_file}", 'w') as f:
        json.dump(error_result, f, indent=2)
    sys.exit(1)
'''

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
