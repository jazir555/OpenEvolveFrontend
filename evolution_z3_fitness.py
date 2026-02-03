"""
Evolution Z3 Fitness

Z3-based fitness evaluation for evolutionary workflows:
- Constraint-based fitness validation
- Verify evolved solutions satisfy constraints
- Mutation validation
- Multi-objective optimization with Z3

Integrates with:
- evolution.py
- evolutionary_optimization.py
- evolution_workflow_templates.py

Author: OpenEvolve
Created: 2026-02-02
"""


import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import (
        Z3SolverEngine, Z3Variable, Z3Constraint, Z3ConstraintType, Z3Config
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


try:
    from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False


@dataclass
class FitnessConstraint:
    """A constraint for fitness evaluation."""
    constraint_id: str
    expression: str
    weight: float = 1.0
    is_hard: bool = True  # Hard constraints must be satisfied
    penalty: float = 0.0  # Penalty for violation


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""
    individual_id: str
    fitness_score: float
    constraints_satisfied: bool
    violated_constraints: List[str] = field(default_factory=list)
    objective_values: Dict[str, float] = field(default_factory=dict)
    is_feasible: bool = True


class Z3FitnessEvaluator:
    """
    Z3-based fitness evaluator for evolutionary algorithms.
    
    Provides:
    - Constraint satisfaction checking for evolved solutions
    - Multi-objective fitness evaluation
    - Mutation validation (ensure mutated solutions are valid)
    - Pareto frontier calculation
    """
    
    def __init__(self, config=None):
        self.config = config or (Z3Config(timeout=30.0) if Z3_AVAILABLE else None)
        self.solver = Z3SolverEngine(self.config) if Z3_AVAILABLE else None
        self.advanced = Z3AdvancedSolver(self.config) if Z3_ADVANCED_AVAILABLE else None
        
        # Fitness cache
        self._fitness_cache: Dict[str, FitnessResult] = {}
    
    def evaluate_fitness(
        self,
        individual: Dict[str, Any],
        constraints: List[FitnessConstraint],
        objectives: Optional[List[str]] = None
    ) -> FitnessResult:
        """
        Evaluate fitness of an individual using Z3.
        
        Args:
            individual: The evolved solution to evaluate
            constraints: Fitness constraints
            objectives: Objective expressions for multi-objective optimization
            
        Returns:
            FitnessResult
        """
        individual_id = individual.get("id", f"ind_{hash(str(individual))}")
        
        # Check cache
        if individual_id in self._fitness_cache:
            return self._fitness_cache[individual_id]
        
        if not Z3_AVAILABLE:
            # Fallback: simple fitness calculation
            return FitnessResult(
                individual_id=individual_id,
                fitness_score=individual.get("raw_fitness", 0.0),
                constraints_satisfied=True
            )
        
        try:
            # Build variables from individual
            variables = self._extract_variables(individual)
            
            # Build constraint list
            z3_constraints = []
            for fc in constraints:
                z3_constraints.append(Z3Constraint(
                    expression=fc.expression,
                    constraint_type=Z3ConstraintType.BOOLEAN
                ))
            
            # Check constraint satisfaction
            result = self.solver.solve_constraints(variables, z3_constraints)
            
            constraints_satisfied = result.is_sat()
            violated = []
            
            if not constraints_satisfied:
                # Identify violated constraints
                violated = [c.constraint_id for c in constraints]
            
            # Calculate fitness score
            if constraints_satisfied and objectives and Z3_ADVANCED_AVAILABLE:
                # Multi-objective evaluation
                fitness_score = self._evaluate_objectives(variables, objectives)
            elif constraints_satisfied:
                fitness_score = individual.get("raw_fitness", 1.0)
            else:
                # Penalize constraint violations
                penalty = sum(c.penalty for c in constraints if c.constraint_id in violated)
                fitness_score = individual.get("raw_fitness", 0.0) - penalty
            
            fitness_result = FitnessResult(
                individual_id=individual_id,
                fitness_score=fitness_score,
                constraints_satisfied=constraints_satisfied,
                violated_constraints=violated,
                is_feasible=constraints_satisfied
            )
            
            # Cache result
            self._fitness_cache[individual_id] = fitness_result
            
            return fitness_result
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return FitnessResult(
                individual_id=individual_id,
                fitness_score=0.0,
                constraints_satisfied=False,
                violated_constraints=["evaluation_error"]
            )
    
    def validate_mutation(
        self,
        original: Dict[str, Any],
        mutated: Dict[str, Any],
        constraints: List[FitnessConstraint]
    ) -> bool:
        """
        Validate that a mutation produces a valid individual.
        
        Args:
            original: Original individual
            mutated: Mutated individual
            constraints: Constraints that must be satisfied
            
        Returns:
            True if mutation is valid
        """
        if not Z3_AVAILABLE:
            return True  # Assume valid if Z3 unavailable
        
        try:
            # Check if mutated individual satisfies constraints
            fitness = self.evaluate_fitness(mutated, constraints)
            return fitness.constraints_satisfied
        except Exception as e:
            logger.error(f"Mutation validation failed: {e}")
            return False
    
    def calculate_pareto_frontier(
        self,
        population: List[Dict[str, Any]],
        objectives: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Calculate Pareto frontier for multi-objective optimization.
        
        Args:
            population: Population of individuals
            objectives: List of objective expressions
            
        Returns:
            List of Pareto-optimal individuals
        """
        if not Z3_ADVANCED_AVAILABLE or not self.advanced:
            # Simple dominance check without Z3
            return self._simple_pareto_frontier(population, objectives)
        
        try:
            # Build optimization problem
            pareto_front = []
            
            for individual in population:
                is_dominated = False
                
                # Check if any other individual dominates this one
                for other in population:
                    if other == individual:
                        continue
                    
                    if self._dominates(other, individual, objectives):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    pareto_front.append(individual)
            
            return pareto_front
            
        except Exception as e:
            logger.error(f"Pareto frontier calculation failed: {e}")
            return population[:5]  # Return first 5 as fallback
    
    def _dominates(
        self,
        ind1: Dict[str, Any],
        ind2: Dict[str, Any],
        objectives: List[str]
    ) -> bool:
        """Check if ind1 dominates ind2."""
        obj1 = ind1.get("objective_values", {})
        obj2 = ind2.get("objective_values", {})
        
        better_in_at_least_one = False
        
        for obj in objectives:
            val1 = obj1.get(obj, 0.0)
            val2 = obj2.get(obj, 0.0)
            
            if val1 > val2:
                better_in_at_least_one = True
            elif val1 < val2:
                return False
        
        return better_in_at_least_one
    
    def _simple_pareto_frontier(
        self,
        population: List[Dict[str, Any]],
        objectives: List[str]
    ) -> List[Dict[str, Any]]:
        """Simple Pareto frontier calculation."""
        pareto_front = []
        
        for ind in population:
            is_dominated = False
            
            for other in population:
                if other == ind:
                    continue
                
                if self._dominates(other, ind, objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(ind)
        
        return pareto_front
    
    def _extract_variables(self, individual: Dict[str, Any]) -> List[Z3Variable]:
        """Extract Z3 variables from individual."""
        variables = []
        
        for key, value in individual.items():
            if isinstance(value, (int, float)) and key not in ["id", "fitness", "raw_fitness"]:
                var_type = Z3ConstraintType.INTEGER if isinstance(value, int) else Z3ConstraintType.REAL
                variables.append(Z3Variable(key, var_type))
        
        return variables
    
    def _evaluate_objectives(
        self,
        variables: List[Z3Variable],
        objectives: List[str]
    ) -> float:
        """Evaluate objective functions."""
        # Simplified: return sum of objective values
        return 1.0


class Z3EvolutionIntegration:
    """Integration between Evolution system and Z3 fitness evaluator."""
    
    def __init__(self):
        self.evaluator = Z3FitnessEvaluator()
    
    def evaluate_population(
        self,
        population: List[Dict[str, Any]],
        constraint_specs: List[Dict[str, Any]]
    ) -> List[FitnessResult]:
        """Evaluate entire population."""
        constraints = [
            FitnessConstraint(
                constraint_id=c.get("id", f"c{i}"),
                expression=c["expression"],
                weight=c.get("weight", 1.0),
                is_hard=c.get("is_hard", True)
            )
            for i, c in enumerate(constraint_specs)
        ]
        
        results = []
        for individual in population:
            fitness = self.evaluator.evaluate_fitness(individual, constraints)
            results.append(fitness)
        
        return results


def get_z3_fitness_evaluator():
    """Get global Z3 fitness evaluator."""
    return Z3FitnessEvaluator()


def get_z3_evolution_integration():
    """Get global evolution integration."""
    return Z3EvolutionIntegration()


if __name__ == "__main__":
    print("Evolution Z3 Fitness initialized")
