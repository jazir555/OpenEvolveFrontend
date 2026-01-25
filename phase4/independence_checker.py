"""
Independence Verification for Δ₃
=================================

Implements independence checks to ensure non-circular validation.

This module provides:
- Data leakage detection
- Holdout integrity verification
- Circularity detection
- Go/no-go decision logic

Author: Agent E3 (Δ₃ Specialist)
Date: 2025-12-31
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Any
import random

from .types import Problem, RESESolution
from .aci_reduction_validator import (


    ConstraintPartition,
    Delta3Config,
    IndependenceCheckResult
)


# =============================================================================
# INDEPENDENCE CHECKER
# =============================================================================

class IndependenceChecker:
    """
    Verify independence of validation (non-circular validation).

    Implements multiple checks to ensure validation is independent
    and not subject to circular reasoning.
    """

    def __init__(self, config: Delta3Config):
        """
        Initialize independence checker.

        Args:
            config: Δ₃ configuration
        """
        self.config = config

    def verify_independence(
        self,
        partition: ConstraintPartition,
        rese_solution: RESESolution,
        problem: Problem
    ) -> IndependenceCheckResult:
        """
        Verify independence (non-circular validation).

        Performs 4 checks:
        1. Data leakage detection
        2. Holdout integrity verification
        3. Circularity detection
        4. Solution independence verification

        Args:
            partition: Constraint partition
            rese_solution: RESE solution
            problem: Original problem

        Returns:
            IndependenceCheckResult
        """
        issues = []
        is_independent = True

        # Check 1: Data leakage
        data_leakage = self._check_data_leakage(partition, rese_solution)
        if data_leakage['leaked']:
            is_independent = False
            issues.extend(data_leakage['issues'])

        # Check 2: Holdout integrity
        holdout_integrity = self._check_holdout_integrity(partition)
        if not holdout_integrity:
            is_independent = False
            issues.append("Holdout integrity compromised")

        # Check 3: Circularity
        circularity = self._check_circularity(rese_solution, problem)
        if circularity['is_circular']:
            is_independent = False
            issues.extend(circularity['issues'])

        # Check 4: Solution independence
        solution_independent = self._check_solution_independence(
            rese_solution, partition.holdout_constraints
        )
        if not solution_independent:
            is_independent = False
            issues.append("Solution depends on holdout constraints")

        return IndependenceCheckResult(
            is_independent=is_independent,
            data_leakage_detected=data_leakage['leaked'],
            holdout_integrity=holdout_integrity,
            circularity_detected=circularity['is_circular'],
            issues=issues
        )

    def create_partition(
        self,
        problem: Problem,
        seed: int = None
    ) -> ConstraintPartition:
        """
        Partition constraints into training and holdout sets.

        Strategy: Stratified random sampling by constraint type

        Args:
            problem: Problem with constraints
            seed: Random seed for reproducibility

        Returns:
            ConstraintPartition
        """
        if seed is not None:
            random.seed(seed)

        constraints = problem.constraints

        # Group by type
        by_type = self._group_by_type(constraints)

        # Calculate holdout sizes
        training = []
        holdout = []

        for constraint_type, type_constraints in by_type.items():
            n_holdout = int(len(type_constraints) * self.config.holdout_ratio)

            # Shuffle
            shuffled = type_constraints.copy()
            random.shuffle(shuffled)

            # Split
            type_holdout = shuffled[:n_holdout]
            type_training = shuffled[n_holdout:]

            holdout.extend(type_holdout)
            training.extend(type_training)

        return ConstraintPartition(
            training_constraints=training,
            holdout_constraints=holdout,
            partition_method=self.config.holdout_method,
            stratification={
                "by_type": self.config.stratify_by_type,
                "holdout_ratio": self.config.holdout_ratio
            },
            seed=seed
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_data_leakage(
        self,
        partition: ConstraintPartition,
        rese_solution: RESESolution
    ) -> Dict[str, Any]:
        """
        Check if holdout data leaked into solution.

        Args:
            partition: Constraint partition
            rese_solution: RESE solution

        Returns:
            Dict with 'leaked' (bool) and 'issues' (list)
        """
        issues = []
        leaked = False

        # Check if solution references holdout constraints
        holdout_ids = set()
        for constraint in partition.holdout_constraints:
            if hasattr(constraint, 'id'):
                holdout_ids.add(constraint.id)
            elif hasattr(constraint, 'name'):
                holdout_ids.add(constraint.name)

        solution_str = str(rese_solution.solution).lower()
        metadata_str = str(rese_solution.metadata).lower()

        for holdout_id in holdout_ids:
            if str(holdout_id).lower() in solution_str:
                leaked = True
                issues.append(f"Solution mentions holdout constraint {holdout_id}")

        # Check metadata for holdout references
        if "holdout" in metadata_str:
            leaked = True
            issues.append("Metadata contains holdout reference")

        # Check if ACI history is suspicious (perfect monotonic decrease)
        if hasattr(rese_solution, 'aci_history') and len(rese_solution.aci_history) > 2:
            if self._is_suspiciously_perfect(rese_solution.aci_history):
                leaked = True
                issues.append("ACI history shows suspiciously perfect decrease (possible overfitting)")

        return {'leaked': leaked, 'issues': issues}

    def _check_holdout_integrity(self, partition: ConstraintPartition) -> bool:
        """
        Check if holdout integrity maintained.

        Verifies no overlap between training and holdout sets.

        Args:
            partition: Constraint partition

        Returns:
            True if holdout integrity maintained
        """
        # Get constraint IDs
        training_ids = set()
        for constraint in partition.training_constraints:
            if hasattr(constraint, 'id'):
                training_ids.add(constraint.id)
            elif hasattr(constraint, 'name'):
                training_ids.add(constraint.name)

        holdout_ids = set()
        for constraint in partition.holdout_constraints:
            if hasattr(constraint, 'id'):
                holdout_ids.add(constraint.id)
            elif hasattr(constraint, 'name'):
                holdout_ids.add(constraint.name)

        # Check for overlap
        overlap = training_ids & holdout_ids
        return len(overlap) == 0

    def _check_circularity(
        self,
        rese_solution: RESESolution,
        problem: Problem
    ) -> Dict[str, Any]:
        """
        Check for circular reasoning in validation.

        Args:
            rese_solution: RESE solution
            problem: Original problem

        Returns:
            Dict with 'is_circular' (bool) and 'issues' (list)
        """
        issues = []
        is_circular = False

        # Check 1: Self-validation in metadata
        metadata_str = str(rese_solution.metadata).lower()
        if "validation" in metadata_str and "self" in metadata_str:
            is_circular = True
            issues.append("Self-validation detected in metadata")

        # Check 2: Circular metric reference
        if hasattr(rese_solution, 'validation_metrics'):
            if hasattr(rese_solution, 'solution_metrics'):
                if rese_solution.validation_metrics == rese_solution.solution_metrics:
                    is_circular = True
                    issues.append("Circular metric reference")

        # Check 3: Begging the question in solution
        solution_str = str(rese_solution.solution).lower()
        problem_str = str(problem.description).lower()

        # Check if solution assumes what it's trying to prove
        circular_patterns = [
            "correct because",
            "valid since",
            "true as",
            "obviously",
            "clearly"
        ]

        for pattern in circular_patterns:
            if pattern in solution_str:
                is_circular = True
                issues.append(f"Potential circular reasoning: '{pattern}' found")
                break

        # Check 4: Solution restates problem
        if solution_str == problem_str or solution_str in problem_str:
            is_circular = True
            issues.append("Solution restates problem (no real invention)")

        return {'is_circular': is_circular, 'issues': issues}

    def _check_solution_independence(
        self,
        rese_solution: RESESolution,
        holdout_constraints: List[Any]
    ) -> bool:
        """
        Check if solution is independent of holdout constraints.

        Args:
            rese_solution: RESE solution
            holdout_constraints: Holdout constraints

        Returns:
            True if solution is independent
        """
        # Get holdout constraint identifiers
        holdout_ids = set()
        for constraint in holdout_constraints:
            if hasattr(constraint, 'id'):
                holdout_ids.add(constraint.id)
            elif hasattr(constraint, 'name'):
                holdout_ids.add(constraint.name)
            elif hasattr(constraint, '__str__'):
                holdout_ids.add(str(constraint))

        # Check if solution references holdout
        solution_str = str(rese_solution.solution)

        for holdout_id in holdout_ids:
            if str(holdout_id) in solution_str:
                return False

        # Check metadata
        metadata_str = str(rese_solution.metadata)
        for holdout_id in holdout_ids:
            if str(holdout_id) in metadata_str:
                return False

        return True

    def _group_by_type(self, constraints: List[Any]) -> Dict[str, List[Any]]:
        """
        Group constraints by type.

        Args:
            constraints: List of constraints

        Returns:
            Dict mapping type to list of constraints
        """
        groups = {}
        for constraint in constraints:
            ctype = 'unknown'
            if hasattr(constraint, 'type'):
                ctype = constraint.type
            elif hasattr(constraint, 'constraint_type'):
                ctype = constraint.constraint_type
            elif isinstance(constraint, dict):
                ctype = constraint.get('type', 'unknown')

            if ctype not in groups:
                groups[ctype] = []
            groups[ctype].append(constraint)

        return groups

    def _is_suspiciously_perfect(self, aci_history: List[float], threshold: float = 0.99) -> bool:
        """
        Check if ACI history is suspiciously perfect (possible overfitting).

        Args:
            aci_history: ACI values through stages
            threshold: Correlation threshold

        Returns:
            True if suspiciously perfect monotonic decrease
        """
        if len(aci_history) < 3:
            return False

        # Check if perfectly monotonically decreasing
        is_monotonic = all(aci_history[i] > aci_history[i+1] for i in range(len(aci_history)-1))

        if is_monotonic:
            # Check correlation with perfect decrease
            perfect_decrease = list(range(len(aci_history), 0, -1))
            correlation = np.corrcoef(aci_history, perfect_decrease)[0, 1]

            if correlation > threshold:
                return True  # Suspiciously perfect

        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

import numpy as np


def detect_overfitting(
    training_aci: float,
    holdout_aci: float,
    threshold: float = 0.3
) -> bool:
    """
    Detect overfitting by comparing training and holdout ACI.

    Args:
        training_aci: ACI on training set
        holdout_aci: ACI on holdout set
        threshold: Maximum allowed difference

    Returns:
        True if overfitting detected
    """
    training_reduction = training_aci  # Normalized to baseline
    holdout_reduction = holdout_aci

    difference = abs(training_reduction - holdout_reduction)

    return difference > threshold


def check_data_leakage_by_hash(
    training_data: Any,
    test_data: Any
) -> bool:
    """
    Check for data leakage by comparing data hashes.

    Args:
        training_data: Training data
        test_data: Test data

    Returns:
        True if data leakage detected
    """
    import hashlib
    import json

    # Serialize and hash
    def get_hash(data):
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    return get_hash(training_data) == get_hash(test_data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'IndependenceChecker',
    'detect_overfitting',
    'check_data_leakage_by_hash',
]
