"""
Solution Repair

Repair transferred solutions that fail validation.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, Any, List
import random
from ..core.domain import Domain
from .validator import SolutionValidator


class SolutionRepair:
    """
    Repair solutions that fail validation using local search
    """

    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self.validator = SolutionValidator()

    def repair(
        self,
        solution: Any,
        domain: Domain,
        validation_errors: List[Dict]
    ) -> Any:
        """
        Attempt to repair solution using local search

        Args:
            solution: Invalid solution
            domain: Target domain
            validation_errors: List of constraint violations

        Returns:
            Repaired solution (or original if repair fails)
        """
        if not isinstance(solution, dict):
            return solution

        best_solution = solution.copy()
        best_score = self._evaluate_solution(best_solution, domain)

        # Extract parameters
        if 'parameters' not in best_solution:
            return solution

        for iteration in range(self.max_iterations):
            # Perturb parameters
            perturbed = self._perturb_solution(best_solution, domain)

            # Evaluate
            score = self._evaluate_solution(perturbed, domain)

            if score > best_score:
                best_solution = perturbed
                best_score = score

                # Check if valid
                validation = self.validator.validate(best_solution, domain)
                if validation['is_valid']:
                    return best_solution

        return best_solution

    def _evaluate_solution(self, solution: Dict, domain: Domain) -> float:
        """
        Evaluate solution quality

        Lower penalty = better score
        """
        validation = self.validator.validate(solution, domain)

        if validation['is_valid']:
            return 1.0
        else:
            # Penalty for each violation
            num_violations = len(validation.get('violations', []))
            return max(0.0, 1.0 - 0.2 * num_violations)

    def _perturb_solution(self, solution: Dict, domain: Domain) -> Dict:
        """
        Create perturbed version of solution

        Adjust parameters randomly
        """
        perturbed = solution.copy()

        if 'parameters' not in perturbed:
            return perturbed

        # Perturb a random parameter
        params = perturbed['parameters'].copy()

        if not params:
            return perturbed

        # Select random parameter
        param_name = random.choice(list(params.keys()))
        param_value = params[param_name]

        # Only perturb numeric parameters
        if isinstance(param_value, (int, float)):
            # Perturb by ±10%
            perturbation = random.uniform(-0.1, 0.1)
            params[param_name] = param_value * (1 + perturbation)

        perturbed['parameters'] = params

        return perturbed
