"""
ACI Analyzer - Algorithmic Complexity Index Analysis

Provides tools for calculating and analyzing algorithmic complexity
of problem solutions and constraint systems.

Author: Agent D1 (Γ₁ Specialist)
Created: 2025-12-31
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import math
import numpy as np


class ComplexityType(Enum):
    """Types of algorithmic complexity"""
    KOLMOGOROV = "kolmogorov"  # Descriptional complexity
    COMPUTATIONAL = "computational"  # Time/space complexity
    STRUCTURAL = "structural"  # Graph/structural complexity
    CONSTRAINT = "constraint"  # Constraint satisfaction complexity


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics"""
    kolmogorov_complexity: float = 0.0
    computational_time: float = 0.0
    computational_space: float = 0.0
    structural_entropy: float = 0.0
    constraint_density: float = 0.0
    description_length: float = 0.0  # Lempel-Ziv complexity

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'kolmogorov_complexity': self.kolmogorov_complexity,
            'computational_time': self.computational_time,
            'computational_space': self.computational_space,
            'structural_entropy': self.structural_entropy,
            'constraint_density': self.constraint_density,
            'description_length': self.description_length
        }


@dataclass
class ACIResult:
    """Result of ACI analysis"""
    aci_value: float  # Main ACI score (0-1)
    complexity_type: ComplexityType
    metrics: ComplexityMetrics
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'aci_value': self.aci_value,
            'complexity_type': self.complexity_type.value,
            'metrics': self.metrics.to_dict(),
            'confidence': self.confidence,
            'details': self.details
        }


class ACIAnalyzer:
    """
    Algorithmic Complexity Index Analyzer

    Calculates and analyzes algorithmic complexity of problem solutions,
    constraint systems, and architectural structures.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ACI Analyzer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.weights = self.config.get('weights', {
            'kolmogorov': 0.3,
            'computational': 0.25,
            'structural': 0.25,
            'constraint': 0.2
        })

    def calculate(
        self,
        solution: Dict[str, Any],
        constraints: Optional[List[Dict]] = None,
        **kwargs
    ) -> ACIResult:
        """
        Calculate ACI for a solution.

        Args:
            solution: Solution dictionary
            constraints: Optional list of constraints
            **kwargs: Additional parameters

        Returns:
            ACIResult with complexity analysis
        """
        # Extract metrics
        metrics = self._calculate_metrics(solution, constraints)

        # Calculate weighted ACI value
        aci_value = self._calculate_aci_value(metrics)

        # Determine primary complexity type
        complexity_type = self._determine_complexity_type(metrics)

        # Calculate confidence
        confidence = self._calculate_confidence(metrics, solution)

        return ACIResult(
            aci_value=aci_value,
            complexity_type=complexity_type,
            metrics=metrics,
            confidence=confidence,
            details={
                'num_constraints': len(constraints) if constraints else 0,
                'solution_size': len(str(solution)),
                'analysis_method': 'weighted_aggregation'
            }
        )

    def _calculate_metrics(
        self,
        solution: Dict[str, Any],
        constraints: Optional[List[Dict]]
    ) -> ComplexityMetrics:
        """Calculate detailed complexity metrics"""
        # Kolmogorov complexity (approximated by description length)
        kolmogorov = self._estimate_kolmogorov_complexity(solution)

        # Computational complexity
        time_complexity, space_complexity = self._estimate_computational_complexity(
            solution, constraints
        )

        # Structural entropy
        structural = self._calculate_structural_entropy(solution)

        # Constraint density
        constraint_density = self._calculate_constraint_density(solution, constraints)

        # Description length (Lempel-Ziv approximation)
        description_length = self._calculate_description_length(solution)

        return ComplexityMetrics(
            kolmogorov_complexity=kolmogorov,
            computational_time=time_complexity,
            computational_space=space_complexity,
            structural_entropy=structural,
            constraint_density=constraint_density,
            description_length=description_length
        )

    def _estimate_kolmogorov_complexity(self, solution: Dict[str, Any]) -> float:
        """
        Estimate Kolmogorov complexity (normalized to 0-1).

        Uses compression-based approximation.
        """
        solution_str = str(solution)
        length = len(solution_str)

        # Simple approximation: normalized entropy
        if length == 0:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in solution_str:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)

        # Normalize by max entropy (log of alphabet size)
        max_entropy = math.log2(len(char_counts)) if len(char_counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _estimate_computational_complexity(
        self,
        solution: Dict[str, Any],
        constraints: Optional[List[Dict]]
    ) -> Tuple[float, float]:
        """
        Estimate computational complexity (time and space).

        Returns normalized values (0-1).
        """
        # Heuristic: based on solution structure
        num_variables = len(solution.get('variables', {}))
        num_constraints = len(constraints) if constraints else 0
        num_operations = len(solution.get('operations', []))

        # Time complexity: function of constraints and operations
        time_complexity = min(1.0, (num_constraints * num_operations) / 1000.0)

        # Space complexity: function of variables
        space_complexity = min(1.0, num_variables / 100.0)

        return time_complexity, space_complexity

    def _calculate_structural_entropy(self, solution: Dict[str, Any]) -> float:
        """
        Calculate structural entropy of solution.

        Measures organization and pattern complexity.
        """
        # Extract structural information
        structure = solution.get('structure', {})

        if not structure:
            return 0.0

        # Count nodes, edges, dependencies
        num_nodes = len(structure.get('nodes', []))
        num_edges = len(structure.get('edges', []))

        if num_nodes == 0:
            return 0.0

        # Calculate graph entropy
        max_edges = num_nodes * (num_nodes - 1) / 2
        edge_density = num_edges / max_edges if max_edges > 0 else 0

        # Entropy peaks at moderate densities
        entropy = -edge_density * math.log2(edge_density + 1e-10) \
                  - (1 - edge_density) * math.log2(1 - edge_density + 1e-10)

        return min(1.0, entropy)

    def _calculate_constraint_density(
        self,
        solution: Dict[str, Any],
        constraints: Optional[List[Dict]]
    ) -> float:
        """Calculate constraint density (0-1)"""
        if not constraints:
            return 0.0

        num_variables = len(solution.get('variables', {}))
        if num_variables == 0:
            return 0.0

        # Constraint density: constraints per variable
        density = len(constraints) / num_variables

        # Normalize (typical range 0-10 constraints per variable)
        return min(1.0, density / 10.0)

    def _calculate_description_length(self, solution: Dict[str, Any]) -> float:
        """
        Calculate Lempel-Ziv description length.

        Approximates algorithmic complexity via compression.
        """
        solution_str = str(solution)

        if len(solution_str) == 0:
            return 0.0

        # Simple Lempel-Ziv-like measure
        dictionary = set()
        compressed_size = 0
        current_phrase = ""

        for char in solution_str:
            current_phrase += char
            if current_phrase not in dictionary:
                dictionary.add(current_phrase)
                compressed_size += 1
                current_phrase = ""

        # Normalize
        return min(1.0, compressed_size / len(solution_str))

    def _calculate_aci_value(self, metrics: ComplexityMetrics) -> float:
        """Calculate weighted ACI value from metrics"""
        aci = (
            self.weights['kolmogorov'] * metrics.kolmogorov_complexity +
            self.weights['computational'] * (
                (metrics.computational_time + metrics.computational_space) / 2
            ) +
            self.weights['structural'] * metrics.structural_entropy +
            self.weights['constraint'] * metrics.constraint_density
        )

        return min(1.0, max(0.0, aci))

    def _determine_complexity_type(self, metrics: ComplexityMetrics) -> ComplexityType:
        """Determine primary complexity type"""
        # Find maximum weighted component
        scores = {
            ComplexityType.KOLMOGOROV: metrics.kolmogorov_complexity,
            ComplexityType.COMPUTATIONAL: (
                (metrics.computational_time + metrics.computational_space) / 2
            ),
            ComplexityType.STRUCTURAL: metrics.structural_entropy,
            ComplexityType.CONSTRAINT: metrics.constraint_density
        }

        return max(scores, key=scores.get)

    def _calculate_confidence(
        self,
        metrics: ComplexityMetrics,
        solution: Dict[str, Any]
    ) -> float:
        """Calculate confidence in ACI estimate"""
        # Confidence based on solution size and completeness
        size_factor = min(1.0, len(str(solution)) / 1000.0)

        # Penalize missing data
        has_constraints = metrics.constraint_density > 0
        has_structure = metrics.structural_entropy > 0

        completeness = 0.5 + 0.3 * has_constraints + 0.2 * has_structure

        return size_factor * completeness

    def compare_complexity(
        self,
        solution1: Dict[str, Any],
        solution2: Dict[str, Any],
        constraints: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Compare complexity of two solutions.

        Args:
            solution1: First solution
            solution2: Second solution
            constraints: Shared constraints

        Returns:
            Comparison results
        """
        aci1 = self.calculate(solution1, constraints)
        aci2 = self.calculate(solution2, constraints)

        return {
            'solution1_aci': aci1.aci_value,
            'solution2_aci': aci2.aci_value,
            'difference': aci2.aci_value - aci1.aci_value,
            'relative_improvement': (
                (aci1.aci_value - aci2.aci_value) / aci1.aci_value
                if aci1.aci_value > 0 else 0.0
            ),
            'simpler_solution': 1 if aci1.aci_value < aci2.aci_value else 2,
            'confidence': (aci1.confidence + aci2.confidence) / 2
        }


# Convenience functions

def calculate_aci(
    solution: Dict[str, Any],
    constraints: Optional[List[Dict]] = None,
    config: Optional[Dict] = None
) -> ACIResult:
    """
    Convenience function to calculate ACI.

    Args:
        solution: Solution to analyze
        constraints: Optional constraints
        config: Optional analyzer configuration

    Returns:
        ACIResult
    """
    analyzer = ACIAnalyzer(config)
    return analyzer.calculate(solution, constraints)


def compare_solutions(
    solution1: Dict[str, Any],
    solution2: Dict[str, Any],
    constraints: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Convenience function to compare two solutions.

    Args:
        solution1: First solution
        solution2: Second solution
        constraints: Optional shared constraints

    Returns:
        Comparison dictionary
    """
    analyzer = ACIAnalyzer()
    return analyzer.compare_complexity(solution1, solution2, constraints)
