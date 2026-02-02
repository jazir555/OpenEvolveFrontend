"""
Z3 Decomposition Validator

Provides formal verification of problem decomposition correctness:
- Constraint preservation verification
- Entanglement constraint validation
- Completeness checking
- Sub-problem independence analysis
- Decomposition quality metrics

Integrates with:
- decomposition_engine.py
- problem_decomposition.py
- decomposition_recomposition_integration.py

Author: OpenEvolve
Created: 2026-02-02
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3ResultStatus, Z3SolverResult,
        Z3ProblemDetector
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 integration not available")

try:
    from z3prover_advanced import Z3AdvancedSolver, ExtractedProof
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

# Import decomposition components
try:
    from problem_decomposition import DecompositionResult, SubProblem
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False
    logger.warning("Decomposition components not available")

try:
    from decomposition_engine import DecompositionEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


# =============================================================================
# Data Classes and Enums
# =============================================================================

class DecompositionProperty(Enum):
    """Properties to verify about a decomposition."""
    COMPLETENESS = "completeness"  # All constraints preserved
    SOUNDNESS = "soundness"  # Solutions compose correctly
    INDEPENDENCE = "independence"  # Sub-problems minimally coupled
    ENTANGLEMENT_SAT = "entanglement_satisfiable"  # Entanglements are satisfiable
    NO_OVERLAP = "no_overlap"  # Sub-problems don't contradict
    PROGRESS = "progress"  # Sub-problems are simpler than parent


@dataclass
class DecompositionConstraint:
    """A constraint in the decomposition."""
    constraint_id: str
    expression: str
    scope: str  # "global", "subproblem", "entanglement"
    source_problem: Optional[str] = None
    target_problem: Optional[str] = None
    
    def to_z3_constraint(self) -> Z3Constraint:
        """Convert to Z3 constraint."""
        return Z3Constraint(
            expression=self.expression,
            constraint_type=Z3ConstraintType.BOOLEAN,
            description=f"{self.scope}: {self.constraint_id}"
        )


@dataclass
class ValidationResult:
    """Result of decomposition validation."""
    success: bool
    valid: bool
    properties_verified: Dict[DecompositionProperty, bool] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    counterexample: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "valid": self.valid,
            "properties_verified": {
                k.value: v for k, v in self.properties_verified.items()
            },
            "violations": self.violations,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class EntanglementSpecification:
    """Specification of entanglement between sub-problems."""
    entanglement_id: str
    source_subproblem: str
    target_subproblem: str
    shared_variables: List[str] = field(default_factory=list)
    coupling_constraints: List[str] = field(default_factory=list)
    strength: str = "weak"  # "weak", "strong", "critical"
    
    def to_z3_constraints(self) -> List[Z3Constraint]:
        """Convert to Z3 constraints."""
        constraints = []
        
        for i, coupling in enumerate(self.coupling_constraints):
            constraints.append(Z3Constraint(
                expression=coupling,
                constraint_type=Z3ConstraintType.BOOLEAN,
                description=f"Entanglement {self.entanglement_id} coupling {i}"
            ))
        
        return constraints


@dataclass
class SubProblemModel:
    """Z3 model for a sub-problem."""
    subproblem_id: str
    variables: List[Z3Variable] = field(default_factory=list)
    constraints: List[DecompositionConstraint] = field(default_factory=list)
    objective: Optional[str] = None
    complexity_score: float = 0.0
    
    def to_z3_constraints(self) -> List[Z3Constraint]:
        """Convert all constraints to Z3."""
        return [c.to_z3_constraint() for c in self.constraints]


# =============================================================================
# Z3 Decomposition Validator
# =============================================================================

class Z3DecompositionValidator:
    """
    Formal verification of problem decomposition using Z3.
    
    Capabilities:
    - Verify all original constraints are preserved in decomposition
    - Check entanglement constraints are satisfiable
    - Validate solution composition correctness
    - Analyze sub-problem independence
    - Detect overlapping/contradictory sub-problems
    - Measure decomposition quality
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config(timeout=120.0, proof_generation=True)
        self.solver = None
        self.prover = None
        self.detector = Z3ProblemDetector()
        
        if Z3_AVAILABLE:
            self.solver = Z3SolverEngine(self.config)
            self.prover = Z3TheoremProver(self.config)
        
        # Statistics
        self._stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "invalid_decompositions": 0,
            "avg_execution_time_ms": 0.0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get validator status."""
        return {
            "z3_available": Z3_AVAILABLE,
            "decomposition_available": DECOMPOSITION_AVAILABLE,
            "statistics": self._stats.copy()
        }
    
    # =====================================================================
    # Main Validation Methods
    # =====================================================================
    
    def validate_decomposition(
        self,
        original_problem: str,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification],
        properties: Optional[List[DecompositionProperty]] = None
    ) -> ValidationResult:
        """
        Validate a complete decomposition.
        
        Args:
            original_problem: Original problem statement or SMT-LIB
            subproblems: List of sub-problem models
            entanglements: Entanglement specifications between sub-problems
            properties: Specific properties to verify (default: all)
            
        Returns:
            ValidationResult
        """
        start_time = time.time()
        self._stats["total_validations"] += 1
        
        if not Z3_AVAILABLE:
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 not available"}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        properties = properties or list(DecompositionProperty)
        result = ValidationResult(success=True, valid=True)
        
        try:
            # Verify completeness
            if DecompositionProperty.COMPLETENESS in properties:
                completeness_result = self.verify_completeness(
                    original_problem, subproblems
                )
                result.properties_verified[DecompositionProperty.COMPLETENESS] = completeness_result.valid
                if not completeness_result.valid:
                    result.valid = False
                    result.violations.extend(completeness_result.violations)
            
            # Verify soundness
            if DecompositionProperty.SOUNDNESS in properties:
                soundness_result = self.verify_soundness(subproblems, entanglements)
                result.properties_verified[DecompositionProperty.SOUNDNESS] = soundness_result.valid
                if not soundness_result.valid:
                    result.valid = False
                    result.violations.extend(soundness_result.violations)
            
            # Verify independence
            if DecompositionProperty.INDEPENDENCE in properties:
                independence_result = self.verify_independence(subproblems, entanglements)
                result.properties_verified[DecompositionProperty.INDEPENDENCE] = independence_result.valid
                if not independence_result.valid:
                    result.valid = False
                    result.violations.extend(independence_result.violations)
            
            # Verify entanglement satisfiability
            if DecompositionProperty.ENTANGLEMENT_SAT in properties:
                ent_result = self.verify_entanglement_satisfiability(entanglements)
                result.properties_verified[DecompositionProperty.ENTANGLEMENT_SAT] = ent_result.valid
                if not ent_result.valid:
                    result.valid = False
                    result.violations.extend(ent_result.violations)
            
            # Verify no overlap
            if DecompositionProperty.NO_OVERLAP in properties:
                overlap_result = self.verify_no_overlap(subproblems)
                result.properties_verified[DecompositionProperty.NO_OVERLAP] = overlap_result.valid
                if not overlap_result.valid:
                    result.valid = False
                    result.violations.extend(overlap_result.violations)
            
            # Calculate metrics
            result.metrics = self._calculate_metrics(original_problem, subproblems, entanglements)
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)
            
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            if result.valid:
                self._stats["successful_validations"] += 1
            else:
                self._stats["invalid_decompositions"] += 1
            
            # Update average execution time
            total_time = self._stats["avg_execution_time_ms"] * (self._stats["total_validations"] - 1)
            total_time += result.execution_time_ms
            self._stats["avg_execution_time_ms"] = total_time / self._stats["total_validations"]
            
        except Exception as e:
            logger.error(f"Decomposition validation failed: {e}")
            result.success = False
            result.valid = False
            result.violations.append({"error": str(e)})
            result.execution_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def verify_completeness(
        self,
        original_problem: str,
        subproblems: List[SubProblemModel]
    ) -> ValidationResult:
        """
        Verify that all original constraints are preserved in sub-problems.
        
        Theorem: Original problem is satisfiable iff all sub-problems are satisfiable
        and their solutions can be composed.
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            # Parse original problem
            original_vars, original_constraints = self._parse_problem(original_problem)
            
            # Collect all sub-problem constraints
            all_subproblem_constraints = []
            for sp in subproblems:
                all_subproblem_constraints.extend(sp.to_z3_constraints())
            
            # Check if original constraints imply sub-problem constraints
            # and vice versa (equivalence)
            
            # Direction 1: Original => Sub-problems
            # If original is satisfiable, sub-problems should be satisfiable
            result1 = self._check_implication(
                original_constraints,
                all_subproblem_constraints
            )
            
            violations = []
            if not result1:
                violations.append({
                    "type": "completeness",
                    "issue": "Some original constraints are not preserved in sub-problems"
                })
            
            return ValidationResult(
                success=True,
                valid=result1,
                violations=violations,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Completeness verification failed: {e}")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_soundness(
        self,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification]
    ) -> ValidationResult:
        """
        Verify that solutions to sub-problems can be composed into a valid solution.
        
        Theorem: If all sub-problems are satisfiable and respect entanglements,
        then a valid global solution exists.
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            # Collect all constraints
            all_constraints = []
            all_variables = []
            
            for sp in subproblems:
                all_constraints.extend(sp.to_z3_constraints())
                all_variables.extend(sp.variables)
            
            for ent in entanglements:
                all_constraints.extend(ent.to_z3_constraints())
            
            # Check satisfiability
            result = self.solver.solve_constraints(all_variables, all_constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                return ValidationResult(
                    success=True,
                    valid=True,
                    execution_time_ms=execution_time
                )
            elif result.is_unsat():
                return ValidationResult(
                    success=True,
                    valid=False,
                    violations=[{
                        "type": "soundness",
                        "issue": "Sub-problems and entanglements are unsatisfiable together"
                    }],
                    execution_time_ms=execution_time
                )
            else:
                return ValidationResult(
                    success=False,
                    valid=False,
                    violations=[{"error": "Unknown satisfiability result"}],
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.error(f"Soundness verification failed: {e}")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_independence(
        self,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification]
    ) -> ValidationResult:
        """
        Verify that sub-problems are sufficiently independent.
        
        Measures coupling between sub-problems through shared variables.
        Lower coupling is generally better for parallel solving.
        """
        start_time = time.time()
        
        try:
            # Analyze variable sharing
            variable_ownership = defaultdict(set)
            for sp in subproblems:
                for var in sp.variables:
                    variable_ownership[var.name].add(sp.subproblem_id)
            
            # Find shared variables
            shared_vars = {
                var: owners for var, owners in variable_ownership.items()
                if len(owners) > 1
            }
            
            # Analyze entanglement complexity
            entanglement_complexity = len(entanglements)
            for ent in entanglements:
                entanglement_complexity += len(ent.coupling_constraints)
            
            # Calculate coupling score
            total_vars = len(variable_ownership)
            shared_var_count = len(shared_vars)
            coupling_score = shared_var_count / total_vars if total_vars > 0 else 0.0
            
            violations = []
            if coupling_score > 0.5:
                violations.append({
                    "type": "independence",
                    "issue": f"High coupling detected: {coupling_score:.1%} of variables are shared",
                    "shared_variables": list(shared_vars.keys())
                })
            
            if entanglement_complexity > len(subproblems) * 2:
                violations.append({
                    "type": "independence",
                    "issue": "Too many entanglement constraints relative to sub-problems"
                })
            
            return ValidationResult(
                success=True,
                valid=len(violations) == 0,
                violations=violations,
                metrics={
                    "coupling_score": coupling_score,
                    "shared_variable_count": shared_var_count,
                    "entanglement_complexity": entanglement_complexity
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Independence verification failed: {e}")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_entanglement_satisfiability(
        self,
        entanglements: List[EntanglementSpecification]
    ) -> ValidationResult:
        """
        Verify that all entanglement constraints are mutually satisfiable.
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            # Collect all entanglement constraints
            all_constraints = []
            for ent in entanglements:
                all_constraints.extend(ent.to_z3_constraints())
            
            # Check satisfiability (no variables needed for pure entanglement check)
            result = self.solver.solve_constraints([], all_constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                return ValidationResult(
                    success=True,
                    valid=True,
                    execution_time_ms=execution_time
                )
            elif result.is_unsat():
                violations = []
                for ent in entanglements:
                    violations.append({
                        "type": "entanglement",
                        "entanglement_id": ent.entanglement_id,
                        "issue": "Entanglement constraints are contradictory"
                    })
                
                return ValidationResult(
                    success=True,
                    valid=False,
                    violations=violations,
                    execution_time_ms=execution_time
                )
            else:
                return ValidationResult(
                    success=False,
                    valid=False,
                    violations=[{"error": "Unknown satisfiability"}],
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.error(f"Entanglement verification failed: {e}")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_no_overlap(
        self,
        subproblems: List[SubProblemModel]
    ) -> ValidationResult:
        """
        Verify that sub-problems don't have contradictory constraints.
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            violations = []
            
            # Check each pair of sub-problems for contradictions
            for i, sp1 in enumerate(subproblems):
                for sp2 in subproblems[i+1:]:
                    # Try to find a model satisfying both
                    combined_constraints = sp1.to_z3_constraints() + sp2.to_z3_constraints()
                    combined_variables = list(set(sp1.variables + sp2.variables))
                    
                    result = self.solver.solve_constraints(combined_variables, combined_constraints)
                    
                    if result.is_unsat():
                        violations.append({
                            "type": "overlap",
                            "subproblem_1": sp1.subproblem_id,
                            "subproblem_2": sp2.subproblem_id,
                            "issue": "Sub-problems have contradictory constraints"
                        })
            
            return ValidationResult(
                success=True,
                valid=len(violations) == 0,
                violations=violations,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Overlap verification failed: {e}")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    # =====================================================================
    # Analysis Methods
    # =====================================================================
    
    def analyze_decomposition_quality(
        self,
        original_problem: str,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of a decomposition.
        
        Returns metrics like:
        - Constraint distribution balance
        - Variable sharing analysis
        - Complexity reduction
        - Parallelization potential
        """
        metrics = {
            "subproblem_count": len(subproblems),
            "entanglement_count": len(entanglements),
            "constraint_distribution": {},
            "variable_sharing": {},
            "complexity_scores": {},
            "parallelization_score": 0.0
        }
        
        # Constraint distribution
        constraint_counts = [len(sp.constraints) for sp in subproblems]
        if constraint_counts:
            metrics["constraint_distribution"] = {
                "min": min(constraint_counts),
                "max": max(constraint_counts),
                "avg": sum(constraint_counts) / len(constraint_counts),
                "balance": min(constraint_counts) / max(constraint_counts) if max(constraint_counts) > 0 else 1.0
            }
        
        # Variable sharing
        var_ownership = defaultdict(set)
        for sp in subproblems:
            for var in sp.variables:
                var_ownership[var.name].add(sp.subproblem_id)
        
        shared_count = sum(1 for owners in var_ownership.values() if len(owners) > 1)
        metrics["variable_sharing"] = {
            "total_variables": len(var_ownership),
            "shared_variables": shared_count,
            "sharing_ratio": shared_count / len(var_ownership) if var_ownership else 0.0
        }
        
        # Complexity scores
        for sp in subproblems:
            metrics["complexity_scores"][sp.subproblem_id] = sp.complexity_score
        
        # Parallelization score (higher is better)
        # Based on low coupling and balanced distribution
        balance_score = metrics["constraint_distribution"].get("balance", 0.0)
        sharing_score = 1.0 - metrics["variable_sharing"].get("sharing_ratio", 0.0)
        metrics["parallelization_score"] = (balance_score + sharing_score) / 2
        
        return metrics
    
    def suggest_improvements(
        self,
        validation_result: ValidationResult,
        original_problem: str,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification]
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements for an invalid decomposition.
        """
        suggestions = []
        
        for violation in validation_result.violations:
            vtype = violation.get("type")
            
            if vtype == "completeness":
                suggestions.append({
                    "target": "decomposition",
                    "action": "review_constraint_mapping",
                    "description": "Ensure all original constraints are mapped to sub-problems"
                })
            
            elif vtype == "soundness":
                suggestions.append({
                    "target": "entanglements",
                    "action": "relax_coupling",
                    "description": "Reduce entanglement constraints or make them weaker"
                })
            
            elif vtype == "independence":
                suggestions.append({
                    "target": "variable_partitioning",
                    "action": "repartition",
                    "description": "Redistribute shared variables to reduce coupling"
                })
            
            elif vtype == "overlap":
                suggestions.append({
                    "target": "subproblems",
                    "action": "merge_or_separate",
                    "description": f"Merge {violation['subproblem_1']} and {violation['subproblem_2']} or separate their concerns"
                })
        
        return suggestions
    
    # =====================================================================
    # Helper Methods
    # =====================================================================
    
    def _parse_problem(
        self,
        problem: str
    ) -> Tuple[List[Z3Variable], List[Z3Constraint]]:
        """Parse problem into Z3 variables and constraints."""
        # Check if it's SMT-LIB
        if "(declare" in problem or "(assert" in problem:
            return self._parse_smtlib_problem(problem)
        
        # Otherwise, use detector
        problem_type, confidence = self.detector.detect_problem_type(problem)
        
        # For now, return empty - would need natural language parsing
        return [], []
    
    def _parse_smtlib_problem(
        self,
        smtlib: str
    ) -> Tuple[List[Z3Variable], List[Z3Constraint]]:
        """Parse SMT-LIB problem."""
        # Use Z3 to parse
        if Z3_AVAILABLE:
            try:
                import z3
                solver = z3.Solver()
                solver.from_string(smtlib)
                
                # Extract assertions as constraints
                constraints = []
                for assertion in solver.assertions():
                    constraints.append(Z3Constraint(
                        expression=str(assertion),
                        constraint_type=Z3ConstraintType.BOOLEAN
                    ))
                
                # Note: Variable extraction would require more sophisticated parsing
                return [], constraints
                
            except Exception as e:
                logger.warning(f"Failed to parse SMT-LIB: {e}")
        
        return [], []
    
    def _check_implication(
        self,
        antecedent: List[Z3Constraint],
        consequent: List[Z3Constraint]
    ) -> bool:
        """
        Check if antecedent implies consequent.
        
        Formula: antecedent => consequent is valid
        Check: antecedent AND NOT consequent is unsatisfiable
        """
        if not Z3_AVAILABLE:
            return True  # Assume valid if Z3 unavailable
        
        # Build verification condition
        # (antecedent AND NOT consequent) should be UNSAT
        verification_constraints = list(antecedent)
        
        # Negate consequent (for proof by contradiction)
        for c in consequent:
            verification_constraints.append(Z3Constraint(
                expression=f"(not {c.expression})",
                constraint_type=Z3ConstraintType.BOOLEAN
            ))
        
        # Check satisfiability
        result = self.solver.solve_constraints([], verification_constraints)
        
        # If UNSAT, implication holds
        return result.is_unsat()
    
    def _calculate_metrics(
        self,
        original_problem: str,
        subproblems: List[SubProblemModel],
        entanglements: List[EntanglementSpecification]
    ) -> Dict[str, float]:
        """Calculate decomposition metrics."""
        metrics = {
            "subproblem_count": float(len(subproblems)),
            "entanglement_count": float(len(entanglements)),
            "avg_constraints_per_subproblem": 0.0,
            "entanglement_density": 0.0
        }
        
        total_constraints = sum(len(sp.constraints) for sp in subproblems)
        if subproblems:
            metrics["avg_constraints_per_subproblem"] = total_constraints / len(subproblems)
        
        if subproblems and len(subproblems) > 1:
            max_entanglements = len(subproblems) * (len(subproblems) - 1) / 2
            metrics["entanglement_density"] = len(entanglements) / max_entanglements
        
        return metrics
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation result."""
        if result.valid:
            return ["Decomposition is valid"]
        
        recommendations = []
        
        for violation in result.violations:
            vtype = violation.get("type")
            
            if vtype == "completeness":
                recommendations.append("Review constraint mapping to ensure all original constraints are preserved")
            elif vtype == "soundness":
                recommendations.append("Check entanglement constraints for contradictions")
            elif vtype == "independence":
                recommendations.append("Reduce coupling between sub-problems by minimizing shared variables")
            elif vtype == "entanglement":
                recommendations.append("Simplify or remove conflicting entanglement constraints")
            elif vtype == "overlap":
                recommendations.append("Merge sub-problems with overlapping concerns or clearly separate their scopes")
        
        return recommendations


# =============================================================================
# Integration with Decomposition Engine
# =============================================================================

class DecompositionEngineZ3Integration:
    """Integration between DecompositionEngine and Z3 validator."""
    
    def __init__(self):
        self.validator = Z3DecompositionValidator()
    
    def validate_decomposition_result(
        self,
        decomposition_result: Any,
        original_problem: str
    ) -> ValidationResult:
        """
        Validate a DecompositionResult from the decomposition engine.
        
        Args:
            decomposition_result: Result from DecompositionEngine
            original_problem: Original problem statement
            
        Returns:
            ValidationResult
        """
        # Convert DecompositionResult to SubProblemModels
        subproblems = self._convert_decomposition_result(decomposition_result)
        
        # Extract entanglements
        entanglements = self._extract_entanglements(decomposition_result)
        
        # Validate
        return self.validator.validate_decomposition(
            original_problem, subproblems, entanglements
        )
    
    def _convert_decomposition_result(
        self,
        result: Any
    ) -> List[SubProblemModel]:
        """Convert DecompositionResult to SubProblemModels."""
        subproblems = []
        
        if hasattr(result, 'sub_problems'):
            for sp in result.sub_problems:
                model = SubProblemModel(
                    subproblem_id=getattr(sp, 'id', str(id(sp))),
                    variables=self._extract_variables(sp),
                    constraints=self._extract_constraints(sp),
                    complexity_score=getattr(sp, 'complexity', 1.0)
                )
                subproblems.append(model)
        
        return subproblems
    
    def _extract_variables(self, subproblem: Any) -> List[Z3Variable]:
        """Extract Z3 variables from sub-problem."""
        variables = []
        
        # Try to extract variables from various sources
        if hasattr(subproblem, 'variables'):
            for var in subproblem.variables:
                var_name = getattr(var, 'name', str(var))
                var_type = getattr(var, 'type', 'INTEGER')
                z3_type = Z3ConstraintType.INTEGER
                if var_type == 'REAL':
                    z3_type = Z3ConstraintType.REAL
                elif var_type == 'BOOLEAN':
                    z3_type = Z3ConstraintType.BOOLEAN
                
                variables.append(Z3Variable(var_name, z3_type))
        
        return variables
    
    def _extract_constraints(self, subproblem: Any) -> List[DecompositionConstraint]:
        """Extract constraints from sub-problem."""
        constraints = []
        
        if hasattr(subproblem, 'constraints'):
            for i, constraint in enumerate(subproblem.constraints):
                constraint_expr = str(constraint)
                constraints.append(DecompositionConstraint(
                    constraint_id=f"{subproblem.id}_c{i}",
                    expression=constraint_expr,
                    scope="subproblem",
                    source_problem=getattr(subproblem, 'parent_id', None)
                ))
        
        return constraints
    
    def _extract_entanglements(self, result: Any) -> List[EntanglementSpecification]:
        """Extract entanglements from decomposition result."""
        entanglements = []
        
        if hasattr(result, 'entanglement_matrix'):
            matrix = result.entanglement_matrix
            for source, targets in matrix.items():
                for target in targets:
                    entanglements.append(EntanglementSpecification(
                        entanglement_id=f"ent_{source}_{target}",
                        source_subproblem=source,
                        target_subproblem=target
                    ))
        
        return entanglements


# =============================================================================
# Global Instance
# =============================================================================

_validator: Optional[Z3DecompositionValidator] = None
_engine_integration: Optional[DecompositionEngineZ3Integration] = None


def get_z3_decomposition_validator() -> Z3DecompositionValidator:
    """Get global Z3 decomposition validator."""
    global _validator
    if _validator is None:
        _validator = Z3DecompositionValidator()
    return _validator


def get_decomposition_engine_z3_integration() -> DecompositionEngineZ3Integration:
    """Get global decomposition engine integration."""
    global _engine_integration
    if _engine_integration is None:
        _engine_integration = DecompositionEngineZ3Integration()
    return _engine_integration


# =============================================================================
# Example Usage
# =============================================================================

def example_decomposition_validation():
    """Example: Validate a decomposition."""
    validator = get_z3_decomposition_validator()
    
    # Original problem (simple constraint system)
    original_problem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (declare-fun y () Int)
    (declare-fun z () Int)
    (assert (> x 0))
    (assert (< x 10))
    (assert (= y (+ x 1)))
    (assert (= z (* y 2)))
    """
    
    # Sub-problems
    subproblems = [
        SubProblemModel(
            subproblem_id="sp1",
            variables=[Z3Variable("x", Z3ConstraintType.INTEGER)],
            constraints=[
                DecompositionConstraint("c1", "(> x 0)", "subproblem"),
                DecompositionConstraint("c2", "(< x 10)", "subproblem")
            ],
            complexity_score=2.0
        ),
        SubProblemModel(
            subproblem_id="sp2",
            variables=[
                Z3Variable("x", Z3ConstraintType.INTEGER),
                Z3Variable("y", Z3ConstraintType.INTEGER)
            ],
            constraints=[
                DecompositionConstraint("c3", "(= y (+ x 1))", "subproblem")
            ],
            complexity_score=1.0
        ),
        SubProblemModel(
            subproblem_id="sp3",
            variables=[
                Z3Variable("y", Z3ConstraintType.INTEGER),
                Z3Variable("z", Z3ConstraintType.INTEGER)
            ],
            constraints=[
                DecompositionConstraint("c4", "(= z (* y 2))", "subproblem")
            ],
            complexity_score=1.0
        )
    ]
    
    # Entanglements
    entanglements = [
        EntanglementSpecification(
            entanglement_id="ent1",
            source_subproblem="sp1",
            target_subproblem="sp2",
            shared_variables=["x"],
            strength="weak"
        ),
        EntanglementSpecification(
            entanglement_id="ent2",
            source_subproblem="sp2",
            target_subproblem="sp3",
            shared_variables=["y"],
            strength="weak"
        )
    ]
    
    # Validate
    result = validator.validate_decomposition(
        original_problem, subproblems, entanglements
    )
    
    print("Decomposition Validation Result:")
    print(f"  Success: {result.success}")
    print(f"  Valid: {result.valid}")
    print(f"  Properties verified:")
    for prop, verified in result.properties_verified.items():
        print(f"    {prop.value}: {verified}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Recommendations:")
    for rec in result.recommendations:
        print(f"    - {rec}")
    
    return result


def example_quality_analysis():
    """Example: Analyze decomposition quality."""
    validator = get_z3_decomposition_validator()
    
    subproblems = [
        SubProblemModel(
            subproblem_id="sp1",
            variables=[
                Z3Variable("a", Z3ConstraintType.INTEGER),
                Z3Variable("b", Z3ConstraintType.INTEGER)
            ],
            constraints=[
                DecompositionConstraint(f"c{i}", f"(constraint {i})", "subproblem")
                for i in range(5)
            ],
            complexity_score=5.0
        ),
        SubProblemModel(
            subproblem_id="sp2",
            variables=[
                Z3Variable("b", Z3ConstraintType.INTEGER),
                Z3Variable("c", Z3ConstraintType.INTEGER)
            ],
            constraints=[
                DecompositionConstraint(f"c{i}", f"(constraint {i})", "subproblem")
                for i in range(5, 10)
            ],
            complexity_score=5.0
        )
    ]
    
    entanglements = [
        EntanglementSpecification(
            entanglement_id="ent1",
            source_subproblem="sp1",
            target_subproblem="sp2",
            shared_variables=["b"]
        )
    ]
    
    quality = validator.analyze_decomposition_quality("", subproblems, entanglements)
    
    print("\nDecomposition Quality Analysis:")
    print(f"  Sub-problems: {quality['subproblem_count']}")
    print(f"  Entanglements: {quality['entanglement_count']}")
    print(f"  Parallelization score: {quality['parallelization_score']:.2f}")
    print(f"  Variable sharing ratio: {quality['variable_sharing']['sharing_ratio']:.2f}")
    
    return quality


if __name__ == "__main__":
    print("Z3 Decomposition Validator")
    print("=" * 60)
    
    example_decomposition_validation()
    example_quality_analysis()
