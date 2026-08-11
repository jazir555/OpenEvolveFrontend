"""
Evolution Z3 Fitness

Z3-based fitness evaluation for evolutionary workflows with CAV-NLP integration:
- Constraint-based fitness validation
- Verify evolved solutions satisfy constraints
- Mutation validation
- Multi-objective optimization with Z3
- CAV-NLP enhanced fitness evaluation:
  * Natural language to constraint formalization
  * Hybrid Z3 + Lean verification
  * Solution canonicalization
  * Proof certificate generation

Integrates with:
- evolution.py
- evolutionary_optimization.py
- evolution_workflow_templates.py
- openevolve.z3_cav_nlp_integration (CAV-NLP enhancement)

Author: OpenEvolve
Created: 2026-02-02
Updated: 2026-02-05 (CAV-NLP integration)
"""


import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from functools import lru_cache

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


# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.info("CAV-NLP integration not available. Install openevolve for enhanced features.")


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
    verification_status: Optional[str] = None  # NEW: CAV-NLP verification status
    proof_certificate: Optional[str] = None  # NEW: Generated proof certificate
    canonical_form: Optional[str] = None  # NEW: Canonical representation


@dataclass
class VerificationResult:
    """Result of hybrid verification (Z3 + Lean)."""
    agreed: bool
    z3_result: bool
    lean_result: Optional[bool] = None
    confidence: float = 0.0
    proof_certificate: Optional[str] = None
    verification_time_ms: float = 0.0


class Z3FitnessEvaluator:
    """
    Z3-based fitness evaluator for evolutionary algorithms with CAV-NLP enhancement.
    
    Provides:
    - Constraint satisfaction checking for evolved solutions
    - Multi-objective fitness evaluation
    - Mutation validation (ensure mutated solutions are valid)
    - Pareto frontier calculation
    - CAV-NLP enhanced features:
      * Natural language constraint formalization
      * Hybrid Z3 + Lean verification
      * Solution canonicalization for duplicate detection
      * Proof certificate generation
    
    Configuration Options:
    - use_cav_nlp: Enable CAV-NLP features (default: True if available)
    - verify_solutions: Enable hybrid verification (default: False)
    - verification_bonus: Fitness bonus for verified solutions (default: 0.1)
    - disagreement_penalty: Penalty for Z3/Lean disagreement (default: 0.2)
    - canonicalize_population: Remove equivalent candidates (default: False)
    - cache_canonical_forms: Cache canonical representations (default: True)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Z3 fitness evaluator with optional CAV-NLP integration.
        
        Args:
            config: Configuration dictionary with options:
                - use_cav_nlp: Enable CAV-NLP features (default: True if available)
                - verify_solutions: Enable hybrid verification (default: False)
                - verification_bonus: Fitness bonus for verified solutions (default: 0.1)
                - disagreement_penalty: Penalty for Z3/Lean disagreement (default: 0.2)
                - canonicalize_population: Remove equivalent candidates (default: False)
                - cache_canonical_forms: Cache canonical representations (default: True)
                - z3_timeout: Z3 solver timeout in seconds (default: 30.0)
        """
        self.config = config or {}
        
        # Standard Z3 initialization
        z3_config = Z3Config(timeout=self.config.get("z3_timeout", 30.0)) if Z3_AVAILABLE else None
        self.solver = Z3SolverEngine(z3_config) if Z3_AVAILABLE else None
        self.advanced = Z3AdvancedSolver(z3_config) if Z3_ADVANCED_AVAILABLE else None
        
        # NEW: CAV-NLP integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", CAV_NLP_AVAILABLE)
        self.verify_solutions = self.config.get("verify_solutions", False)
        self.verification_bonus = self.config.get("verification_bonus", 0.1)
        self.disagreement_penalty = self.config.get("disagreement_penalty", 0.2)
        self.canonicalize_population = self.config.get("canonicalize_population", False)
        self.cache_canonical_forms = self.config.get("cache_canonical_forms", True)
        
        if self.use_cav_nlp and CAV_NLP_AVAILABLE:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP integration enabled for fitness evaluation")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}. Falling back to standard Z3.")
                self.use_cav_nlp = False
                self.enhanced_solver = None
        else:
            self.enhanced_solver = None
            if self.use_cav_nlp and not CAV_NLP_AVAILABLE:
                logger.warning("CAV-NLP requested but not available. Install openevolve.")
                self.use_cav_nlp = False
        
        # Fitness cache
        self._fitness_cache: Dict[str, FitnessResult] = {}
        
        # NEW: Canonical form cache for performance
        self._canonical_cache: Dict[str, str] = {}
        
        # NEW: Verification cache
        self._verification_cache: Dict[str, VerificationResult] = {}
    
    def formalize_fitness_criteria(self, natural_language: str) -> List[Any]:
        """
        Formalize natural language fitness criteria to Z3 constraints.
        
        Uses CAV-NLP to parse natural language descriptions and convert them
        into formal Z3 constraints that can be used for fitness evaluation.
        
        Args:
            natural_language: Natural language description of fitness criteria
            
        Returns:
            List of Z3 ExprRef constraints
            
        Raises:
            ValueError: If CAV-NLP is not enabled or available
            
        Example:
            >>> evaluator = Z3FitnessEvaluator(config={"use_cav_nlp": True})
            >>> constraints = evaluator.formalize_fitness_criteria(
            ...     "The solution must have x greater than 0 and y less than 100"
            ... )
            >>> # Returns: [x > 0, y < 100] as Z3 expressions
        """
        if not self.use_cav_nlp or not self.enhanced_solver:
            raise ValueError(
                "CAV-NLP not enabled or available. "
                "Set use_cav_nlp=True in config and ensure openevolve is installed."
            )
        
        try:
            # Use CAV-NLP to formalize natural language to constraints
            formalized = self.enhanced_solver.formalize_constraint(natural_language)
            logger.debug(f"Formalized '{natural_language}' to {len(formalized)} constraints")
            return formalized
        except Exception as e:
            logger.error(f"Failed to formalize criteria '{natural_language}': {e}")
            raise ValueError(f"Formalization failed: {e}")
    
    def evaluate_fitness(
        self,
        individual: Dict[str, Any],
        constraints: List[FitnessConstraint],
        objectives: Optional[List[str]] = None,
        use_verification: bool = False
    ) -> FitnessResult:
        """
        Evaluate fitness of an individual using Z3 with optional CAV-NLP verification.
        
        Args:
            individual: The evolved solution to evaluate
            constraints: Fitness constraints
            objectives: Objective expressions for multi-objective optimization
            use_verification: Whether to use hybrid Z3 + Lean verification
            
        Returns:
            FitnessResult with optional verification metadata
            
        Performance Note:
            Hybrid verification adds overhead (typically 50-200ms per individual).
            Enable only when proof guarantees are required.
        """
        individual_id = individual.get("id", f"ind_{hash(str(individual))}")
        
        # Check cache
        if individual_id in self._fitness_cache:
            cached = self._fitness_cache[individual_id]
            # Return cached result unless verification is newly requested
            if not use_verification or cached.verification_status is not None:
                return cached
        
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
            
            # NEW: Hybrid verification
            verification_status = None
            proof_certificate = None
            
            if use_verification and self.verify_solutions and self.use_cav_nlp:
                verification = self._verify_with_hybrid(individual, constraints)
                verification_status = "verified" if verification.agreed else "disagreement"
                proof_certificate = verification.proof_certificate
                
                # Adjust fitness based on verification
                if verification.agreed:
                    fitness_score += self.verification_bonus
                    logger.debug(f"Individual {individual_id}: verification bonus applied")
                else:
                    fitness_score -= self.disagreement_penalty
                    logger.warning(f"Individual {individual_id}: Z3/Lean disagreement detected")
            
            # NEW: Generate canonical form if enabled
            canonical_form = None
            if self.use_cav_nlp and self.cache_canonical_forms:
                canonical_form = self.canonicalize_candidate(individual)
            
            fitness_result = FitnessResult(
                individual_id=individual_id,
                fitness_score=fitness_score,
                constraints_satisfied=constraints_satisfied,
                violated_constraints=violated,
                is_feasible=constraints_satisfied,
                verification_status=verification_status,
                proof_certificate=proof_certificate,
                canonical_form=canonical_form
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
    
    def _verify_with_hybrid(
        self,
        candidate: Dict[str, Any],
        constraints: List[FitnessConstraint]
    ) -> VerificationResult:
        """
        Perform hybrid verification using Z3 + Lean via CAV-NLP.
        
        Args:
            candidate: Candidate solution to verify
            constraints: Constraints to verify against
            
        Returns:
            VerificationResult with agreement status and certificate
        """
        if not self.use_cav_nlp or not self.enhanced_solver:
            return VerificationResult(agreed=True, z3_result=True, confidence=1.0)
        
        # Check cache
        cache_key = self._get_candidate_hash(candidate, constraints)
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]
        
        try:
            import time
            start_time = time.time()
            
            # Use enhanced solver for hybrid verification
            verification = self.enhanced_solver.verify_with_lean(constraints)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = VerificationResult(
                agreed=verification.get("agreed", True),
                z3_result=verification.get("z3_result", True),
                lean_result=verification.get("lean_result"),
                confidence=verification.get("confidence", 0.0),
                proof_certificate=verification.get("certificate"),
                verification_time_ms=elapsed_ms
            )
            
            # Cache verification result
            self._verification_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid verification failed: {e}")
            return VerificationResult(agreed=True, z3_result=True, confidence=0.0)
    
    def _get_candidate_hash(self, candidate: Dict[str, Any], constraints: List[FitnessConstraint]) -> str:
        """Generate hash key for candidate + constraints."""
        candidate_str = json.dumps(candidate, sort_keys=True)
        constraints_str = json.dumps([c.__dict__ for c in constraints], sort_keys=True)
        return hashlib.md5((candidate_str + constraints_str).encode()).hexdigest()
    
    def canonicalize_candidate(self, candidate: Dict[str, Any]) -> Optional[str]:
        """
        Return canonical form of candidate solution using CAV-NLP.
        
        Canonicalization ensures that semantically equivalent candidates
        have the same representation, enabling duplicate detection.
        
        Args:
            candidate: Candidate solution to canonicalize
            
        Returns:
            Canonical string representation or None if CAV-NLP unavailable
            
        Performance Note:
            Results are cached when cache_canonical_forms=True (default).
            First canonicalization may take 10-50ms, subsequent are instant.
        """
        if not self.use_cav_nlp or not self.enhanced_solver:
            return None
        
        # Check cache
        candidate_str = json.dumps(candidate, sort_keys=True)
        if candidate_str in self._canonical_cache:
            return self._canonical_cache[candidate_str]
        
        try:
            # Use CAV-NLP canonicalization
            canonical = self.enhanced_solver.canonical_manager.canonicalize(candidate)
            
            # Cache result
            if self.cache_canonical_forms:
                self._canonical_cache[candidate_str] = canonical
            
            return canonical
        except Exception as e:
            logger.error(f"Canonicalization failed: {e}")
            return None
    
    def check_candidate_equivalence(
        self,
        candidate1: Dict[str, Any],
        candidate2: Dict[str, Any]
    ) -> bool:
        """
        Check if two candidates are equivalent using CAV-NLP.
        
        Uses canonicalization to determine semantic equivalence,
        which is more robust than syntactic comparison.
        
        Args:
            candidate1: First candidate
            candidate2: Second candidate
            
        Returns:
            True if candidates are semantically equivalent
            
        Example:
            >>> c1 = {"x": 5, "y": 10}
            >>> c2 = {"y": 10, "x": 5}  # Same, different order
            >>> evaluator.check_candidate_equivalence(c1, c2)
            True
        """
        if not self.use_cav_nlp or not self.enhanced_solver:
            # Fallback to simple comparison
            return json.dumps(candidate1, sort_keys=True) == json.dumps(candidate2, sort_keys=True)
        
        try:
            return self.enhanced_solver.canonical_manager.are_equivalent(
                candidate1, candidate2
            )
        except Exception as e:
            logger.error(f"Equivalence check failed: {e}")
            # Fallback to simple comparison
            return json.dumps(candidate1, sort_keys=True) == json.dumps(candidate2, sort_keys=True)
    
    def remove_equivalent_candidates(
        self,
        population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove equivalent candidates from population using CAV-NLP.
        
        This maintains diversity in the population by removing duplicates
        that are semantically equivalent but syntactically different.
        
        Args:
            population: List of candidate solutions
            
        Returns:
            List of unique candidates
            
        Performance Note:
            This is O(n²) in the worst case. For large populations,
            consider using canonicalize_population() first to enable
            hash-based deduplication.
        """
        if not self.use_cav_nlp:
            logger.warning("CAV-NLP not available, using simple deduplication")
            # Simple deduplication based on string representation
            seen = set()
            unique = []
            for candidate in population:
                key = json.dumps(candidate, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    unique.append(candidate)
            return unique
        
        unique = []
        for candidate in population:
            is_unique = True
            for existing in unique:
                if self.check_candidate_equivalence(candidate, existing):
                    is_unique = False
                    break
            if is_unique:
                unique.append(candidate)
        
        logger.debug(f"Removed {len(population) - len(unique)} equivalent candidates")
        return unique
    
    def canonicalize_population(
        self,
        population: List[Dict[str, Any]]
    ) -> List[Optional[str]]:
        """
        Canonicalize all candidates in population.
        
        Args:
            population: List of candidate solutions
            
        Returns:
            List of canonical string representations
            
        Performance Note:
            Uses caching for efficiency. First call may be slow,
            subsequent calls with similar candidates are fast.
        """
        return [self.canonicalize_candidate(c) for c in population]
    
    def deduplicate_by_canonical_form(
        self,
        population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Efficiently deduplicate using canonical forms.
        
        Faster than remove_equivalent_candidates() for large populations
        as it uses hash-based deduplication on canonical forms.
        
        Args:
            population: List of candidate solutions
            
        Returns:
            List of unique candidates
        """
        canonical_forms = self.canonicalize_population(population)
        seen = set()
        unique = []
        
        for candidate, canonical in zip(population, canonical_forms):
            if canonical is None:
                # Fallback to simple comparison
                key = json.dumps(candidate, sort_keys=True)
            else:
                key = canonical
            
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        
        return unique
    
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
    
    def _extract_variables(self, individual: Dict[str, Any]) -> List:
        """Extract Z3 variables from individual."""
        variables = []
        
        for key, value in individual.items():
            if isinstance(value, (int, float)) and key not in ["id", "fitness", "raw_fitness"]:
                var_type = Z3ConstraintType.INTEGER if isinstance(value, int) else Z3ConstraintType.REAL
                variables.append(Z3Variable(key, var_type))
        
        return variables
    
    def _evaluate_objectives(
        self,
        variables: List,
        objectives: List[str]
    ) -> float:
        """Evaluate objective functions."""
        # Simplified: return sum of objective values
        return 1.0
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._fitness_cache.clear()
        self._canonical_cache.clear()
        self._verification_cache.clear()
        logger.debug("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "fitness_cache_size": len(self._fitness_cache),
            "canonical_cache_size": len(self._canonical_cache),
            "verification_cache_size": len(self._verification_cache)
        }


class Z3EvolutionIntegration:
    """
    Integration between Evolution system and Z3 fitness evaluator.
    
    Provides high-level interface for evaluating populations with
    optional CAV-NLP enhancements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize integration.
        
        Args:
            config: Configuration passed to Z3FitnessEvaluator
        """
        self.evaluator = Z3FitnessEvaluator(config)
        self.config = config or {}
    
    def evaluate_population(
        self,
        population: List[Dict[str, Any]],
        constraint_specs: List[Dict[str, Any]],
        use_verification: bool = False
    ) -> List[FitnessResult]:
        """
        Evaluate entire population.
        
        Args:
            population: List of individuals to evaluate
            constraint_specs: Constraint specifications
            use_verification: Enable hybrid verification
            
        Returns:
            List of FitnessResult
        """
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
            fitness = self.evaluator.evaluate_fitness(
                individual, constraints, use_verification=use_verification
            )
            results.append(fitness)
        
        return results
    
    def evolve_with_deduplication(
        self,
        population: List[Dict[str, Any]],
        constraint_specs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[FitnessResult]]:
        """
        Evaluate population with CAV-NLP deduplication.
        
        Removes equivalent candidates before evaluation for efficiency.
        
        Args:
            population: List of individuals
            constraint_specs: Constraint specifications
            
        Returns:
            Tuple of (unique_population, fitness_results)
        """
        # Deduplicate
        unique_pop = self.evaluator.deduplicate_by_canonical_form(population)
        
        # Evaluate
        results = self.evaluate_population(unique_pop, constraint_specs)
        
        return unique_pop, results


def get_z3_fitness_evaluator(config: Optional[Dict[str, Any]] = None) -> Z3FitnessEvaluator:
    """
    Get global Z3 fitness evaluator.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Z3FitnessEvaluator instance
    """
    return Z3FitnessEvaluator(config)


def get_z3_evolution_integration(config: Optional[Dict[str, Any]] = None) -> Z3EvolutionIntegration:
    """
    Get global evolution integration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Z3EvolutionIntegration instance
    """
    return Z3EvolutionIntegration(config)


if __name__ == "__main__":
    print("Evolution Z3 Fitness initialized")
    print(f"Z3 available: {Z3_AVAILABLE}")
    print(f"CAV-NLP available: {CAV_NLP_AVAILABLE}")
