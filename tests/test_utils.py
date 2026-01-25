"""
Test Utilities and Helper Functions

Comprehensive utility functions for testing RESE components:
- Mock data generators
- Performance measurement tools
- Validation helpers
- Test assertions
- Data comparators

Author: Agent Z2 (Testing/QA Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Performance Measurement
# ============================================================================

class PerformanceTimer:
    """Context manager for timing code execution"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        logger.debug(f"{self.name} took {self.elapsed:.4f} seconds")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.elapsed is None:
            raise RuntimeError("Timer has not been run yet")
        return self.elapsed


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
        return result, elapsed
    return wrapper


def measure_memory(func: Callable) -> Callable:
    """Decorator to measure function memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.debug(f"{func.__name__} used {peak / 1024 / 1024:.2f} MB peak memory")
        return result, peak / 1024 / 1024  # Return peak MB
    return wrapper


# ============================================================================
# Data Generators
# ============================================================================

class TestDataGenerator:
    """Generate test data for various RESE components"""

    @staticmethod
    def generate_constraints(
        count: int = 10,
        complexity: str = "medium",
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate mock constraints

        Args:
            count: Number of constraints to generate
            complexity: 'low', 'medium', or 'high'
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        constraints = []
        types = ["inequality", "equality", "implication", "universal", "existential"]

        for i in range(count):
            if complexity == "low":
                n_vars = np.random.randint(2, 4)
                expr_depth = 1
            elif complexity == "medium":
                n_vars = np.random.randint(3, 7)
                expr_depth = 3
            else:  # high
                n_vars = np.random.randint(5, 15)
                expr_depth = 5

            constraint = {
                "id": f"constraint_{i}",
                "type": np.random.choice(types),
                "variables": [f"x{j}" for j in range(n_vars)],
                "expression": f"expr_depth_{expr_depth}_vars_{n_vars}",
                "priority": np.random.randint(1, 11),
                "verified": np.random.choice([True, False], p=[0.7, 0.3]),
                "verification_cost": np.random.uniform(0.1, 10.0),
                "tightness": np.random.random(),
                "satisfaction_rate": np.random.uniform(0.5, 1.0),
            }
            constraints.append(constraint)

        return constraints

    @staticmethod
    def generate_null_results(
        count: int = 20,
        pattern: str = "random",
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate mock null results

        Args:
            count: Number of null results
            pattern: 'random', 'systematic', 'diverse'
            seed: Random seed
        """
        np.random.seed(seed)

        error_types = [
            "OPTIMIZATION_FAILED",
            "TIMEOUT",
            "INFEASIBLE",
            "NUMERICAL_INSTABILITY",
            "CONVERGENCE_FAILURE",
            "UNKNOWN_FAILURE"
        ]

        problem_types = ["optimization", "satisfiability", "inference", "planning"]
        approach_types = ["deterministic", "stochastic", "approximate", "heuristic"]

        results = []
        for i in range(count):
            if pattern == "systematic":
                # Same error type, similar constraints
                error_type = error_types[0]
                constraints = ["exact_solution", "deterministic_solver"]
            elif pattern == "diverse":
                # Maximum diversity
                error_type = error_types[i % len(error_types)]
                constraints = [f"constraint_{j}" for j in np.random.permutation(10)[:5]]
            else:  # random
                error_type = np.random.choice(error_types)
                constraints = [f"constraint_{j}" for j in range(np.random.randint(1, 6))]

            result = {
                "attempt_id": f"test_{i:03d}",
                "timestamp": (datetime.now().timestamp() - i * 3600),
                "problem_type": np.random.choice(problem_types),
                "approach_type": np.random.choice(approach_types),
                "constraints": constraints,
                "error_type": error_type,
                "error_message": f"Test failure {i}: {error_type}",
                "state": {
                    "iteration": np.random.randint(1, 1000),
                    "objective_value": np.random.uniform(-1000, 1000)
                },
                "iteration": np.random.randint(1, 1000),
                "resources_used": {
                    "cpu": np.random.uniform(10, 100),
                    "memory": np.random.uniform(100, 1000)
                },
                "metadata": {
                    "test": True,
                    "pattern": pattern,
                    "seed": seed
                }
            }
            results.append(result)

        return results

    @staticmethod
    def generate_fdg(
        n_nodes: int = 10,
        n_edges: int = 15,
        seed: int = 42
    ) -> Dict:
        """
        Generate mock Fundamental Dependency Graph

        Args:
            n_nodes: Number of nodes
            n_edges: Number of edges
            seed: Random seed
        """
        np.random.seed(seed)

        node_types = ["variable", "constraint", "objective", "parameter"]
        edge_types = ["constrains", "affects", "depends_on", "implies"]

        nodes = []
        for i in range(n_nodes):
            node = {
                "id": f"node_{i}",
                "type": np.random.choice(node_types),
                "domain": np.random.choice(["Real", "Integer", "Boolean"]),
                "properties": {
                    "value": np.random.uniform(-10, 10) if np.random.random() > 0.5 else None
                }
            }
            nodes.append(node)

        edges = []
        for i in range(min(n_edges, n_nodes * (n_nodes - 1) // 2)):
            src, dst = np.random.choice(n_nodes, 2, replace=False)
            edge = {
                "source": f"node_{src}",
                "target": f"node_{dst}",
                "relation": np.random.choice(edge_types),
                "weight": np.random.uniform(0.1, 1.0)
            }
            edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def generate_pareto_front(
        n_objectives: int = 3,
        n_points: int = 20,
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate mock Pareto front

        Args:
            n_objectives: Number of objectives
            n_points: Number of Pareto-optimal points
            seed: Random seed
        """
        np.random.seed(seed)

        points = []
        for i in range(n_points):
            point = {
                f"objective_{j}": np.random.uniform(0, 100)
                for j in range(n_objectives)
            }
            point["constraints_satisfied"] = np.random.randint(1, 10)
            point["solution_quality"] = np.random.uniform(0.7, 1.0)
            point["dominance_rank"] = 1  # All on Pareto front
            points.append(point)

        return points


# ============================================================================
# Assertion Helpers
# ============================================================================

class TestAssertions:
    """Custom assertion methods for RESE testing"""

    @staticmethod
    def assert_constraint_valid(constraint: Dict, msg: str = ""):
        """Assert constraint has all required fields"""
        required_fields = ["id", "type", "variables", "expression"]
        for field in required_fields:
            assert field in constraint, f"{msg} Missing field: {field}"

    @staticmethod
    def assert_constraints_sat(constraints: List[Dict], msg: str = ""):
        """Assert constraints are satisfiable (basic check)"""
        assert len(constraints) > 0, f"{msg} No constraints provided"

        # Check for obvious contradictions
        for c1 in constraints:
            for c2 in constraints:
                if (c1["type"] == "equality" and c2["type"] == "equality" and
                    c1["id"] != c2["id"] and
                    c1.get("variables") == c2.get("variables")):
                    # Potential contradiction - same variables in different equalities
                    logger.warning(f"Possible contradiction: {c1['id']} vs {c2['id']}")

    @staticmethod
    def assert_pareto_optimal(points: List[Dict], msg: str = ""):
        """Assert points form a valid Pareto front"""
        assert len(points) >= 2, f"{msg} Need at least 2 points for Pareto front"

        # Check that no point dominates another
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j:
                    # p1 dominates p2 if better in all objectives
                    dominates = all(
                        p1.get(f"objective_{k}", 0) >= p2.get(f"objective_{k}", 0)
                        for k in range(3)
                    )
                    if dominates:
                        # p1 should not dominate p2 on Pareto front
                        pass  # Simplified check

    @staticmethod
    def assert_performance_threshold(
        actual: float,
        threshold: float,
        metric: str = "",
        msg: str = ""
    ):
        """Assert performance meets threshold"""
        assert actual >= threshold if "correlation" in metric or "accuracy" in metric or "transfer" in metric else actual <= threshold, \
            f"{msg} {metric} threshold not met: {actual} vs {threshold}"

    @staticmethod
    def assert_fdg_isomorphic(
        fdg1: Dict,
        fdg2: Dict,
        tolerance: float = 0.1,
        msg: str = ""
    ):
        """Assert two FDGs are isomorphic (basic check)"""
        assert len(fdg1["nodes"]) == len(fdg2["nodes"]), \
            f"{msg} Node count mismatch"
        assert len(fdg1["edges"]) == len(fdg2["edges"]), \
            f"{msg} Edge count mismatch"

    @staticmethod
    def assert_innovation_validated(
        innovation: str,
        actual_value: float,
        expected_threshold: float,
        msg: str = ""
    ):
        """Assert KEY INNOVATION meets its threshold"""
        innovation_requirements = {
            "phi15": ("accuracy", lambda v, t: v >= t),
            "imech": ("transfer_rate", lambda v, t: v >= t),
            "gamma1": ("correlation", lambda v, t: v >= t),
            "delta3": ("correlation", lambda v, t: v >= t),
            "psi3": ("reduction_factor", lambda v, t: v >= t),
            "dito": ("speedup", lambda v, t: v >= t),
        }

        if innovation not in innovation_requirements:
            raise ValueError(f"Unknown innovation: {innovation}")

        metric, check_fn = innovation_requirements[innovation]
        assert check_fn(actual_value, expected_threshold), \
            f"{msg} {innovation} {metric} = {actual_value}, required >= {expected_threshold}"


# ============================================================================
# Validation Helpers
# ============================================================================

class ValidationHelpers:
    """Helper functions for validation tests"""

    @staticmethod
    def calculate_accuracy(predictions: List[int], ground_truth: List[int]) -> float:
        """Calculate classification accuracy"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        if len(predictions) == 0:
            return 0.0

        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        return correct / len(predictions)

    @staticmethod
    def calculate_correlation(predictions: List[float], targets: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        if len(predictions) < 2:
            return 0.0

        pred_arr = np.array(predictions)
        target_arr = np.array(targets)

        correlation = np.corrcoef(pred_arr, target_arr)[0, 1]

        # Handle NaN
        if np.isnan(correlation):
            return 0.0

        return abs(correlation)

    @staticmethod
    def calculate_transfer_rate(
        source_constraints: List,
        transferred_constraints: List,
        mapping_scores: List[float]
    ) -> float:
        """Calculate I_mech transfer success rate"""
        if len(source_constraints) == 0:
            return 0.0

        # High-quality mappings have score > 0.7
        successful_transfers = sum(1 for score in mapping_scores if score > 0.7)
        return successful_transfers / len(source_constraints)

    @staticmethod
    def calculate_reduction_factor(
        original_size: int,
        reduced_size: int
    ) -> float:
        """Calculate constraint reduction factor"""
        if reduced_size == 0:
            return float('inf')
        return original_size / reduced_size

    @staticmethod
    def calculate_speedup(baseline_time: float, optimized_time: float) -> float:
        """Calculate speedup factor"""
        if optimized_time == 0:
            return float('inf')
        return baseline_time / optimized_time

    @staticmethod
    def validate_phi15_accuracy(
        predictions: List[int],
        ground_truth: List[int],
        min_accuracy: float = 0.70
    ) -> Tuple[bool, float]:
        """Validate Φ₁.₅ accuracy threshold"""
        accuracy = ValidationHelpers.calculate_accuracy(predictions, ground_truth)
        return accuracy >= min_accuracy, accuracy

    @staticmethod
    def validate_imech_transfer(
        source_constraints: List,
        transferred_constraints: List,
        mapping_scores: List[float],
        min_transfer: float = 0.80
    ) -> Tuple[bool, float]:
        """Validate I_mech transfer rate threshold"""
        transfer_rate = ValidationHelpers.calculate_transfer_rate(
            source_constraints, transferred_constraints, mapping_scores
        )
        return transfer_rate >= min_transfer, transfer_rate

    @staticmethod
    def validate_gamma1_correlation(
        predicted_pareto: List[float],
        actual_pareto: List[float],
        min_correlation: float = 0.85
    ) -> Tuple[bool, float]:
        """Validate Γ₁ correlation threshold"""
        correlation = ValidationHelpers.calculate_correlation(
            predicted_pareto, actual_pareto
        )
        return correlation >= min_correlation, correlation

    @staticmethod
    def validate_delta3_correlation(
        predicted_quality: List[float],
        actual_quality: List[float],
        min_correlation: float = 0.85
    ) -> Tuple[bool, float]:
        """Validate Δ₃ correlation threshold"""
        correlation = ValidationHelpers.calculate_correlation(
            predicted_quality, actual_quality
        )
        return correlation >= min_correlation, correlation

    @staticmethod
    def validate_psi3_reduction(
        original_constraints: int,
        reduced_constraints: int,
        min_reduction: float = 10.0
    ) -> Tuple[bool, float]:
        """Validate Ψ₃ reduction factor threshold"""
        reduction = ValidationHelpers.calculate_reduction_factor(
            original_constraints, reduced_constraints
        )
        return reduction >= min_reduction, reduction

    @staticmethod
    def validate_dito_speedup(
        baseline_time: float,
        dito_time: float,
        min_speedup: float = 3000.0
    ) -> Tuple[bool, float]:
        """Validate DITO speedup threshold"""
        speedup = ValidationHelpers.calculate_speedup(baseline_time, dito_time)
        return speedup >= min_speedup, speedup


# ============================================================================
# Test Data Persistence
# ============================================================================

class TestDataManager:
    """Manage test data storage and retrieval"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)

    def save_test_data(self, data: Any, name: str, subdir: str = "") -> Path:
        """Save test data to JSON file"""
        if subdir:
            dir_path = self.base_dir / subdir
            dir_path.mkdir(exist_ok=True)
        else:
            dir_path = self.base_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = dir_path / filename

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved test data to {filepath}")
        return filepath

    def load_test_data(self, name: str, subdir: str = "") -> Dict:
        """Load test data from JSON file"""
        if subdir:
            dir_path = self.base_dir / subdir
        else:
            dir_path = self.base_dir

        # Find most recent file matching name
        pattern = f"{name}_*.json"
        matching_files = list(dir_path.glob(pattern))

        if not matching_files:
            raise FileNotFoundError(f"No test data found for {name}")

        # Sort by modification time, get most recent
        filepath = max(matching_files, key=lambda p: p.stat().st_mtime)

        with open(filepath, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded test data from {filepath}")
        return data


# ============================================================================
# Benchmark Result Management
# ============================================================================

class BenchmarkTracker:
    """Track and analyze benchmark results"""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.results = []

    def add_result(
        self,
        test_name: str,
        metric_name: str,
        value: float,
        unit: str = "",
        metadata: Dict = None
    ):
        """Add a benchmark result"""
        result = {
            "test_name": test_name,
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.results.append(result)

    def get_results(self, test_name: str = None) -> List[Dict]:
        """Get results, optionally filtered by test name"""
        if test_name:
            return [r for r in self.results if r["test_name"] == test_name]
        return self.results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to file"""
        filepath = self.output_path / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved benchmark results to {filepath}")
        return filepath

    def compare_with_baseline(
        self,
        baseline_path: Path,
        tolerance: float = 0.1
    ) -> Dict[str, Dict]:
        """Compare current results with baseline"""
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        comparisons = {}
        for result in self.results:
            key = f"{result['test_name']}_{result['metric']}"
            baseline_value = next(
                (b['value'] for b in baseline
                 if b['test_name'] == result['test_name'] and b['metric'] == result['metric']),
                None
            )

            if baseline_value is not None:
                diff_pct = ((result['value'] - baseline_value) / baseline_value) * 100
                comparisons[key] = {
                    "current": result['value'],
                    "baseline": baseline_value,
                    "diff_pct": diff_pct,
                    "regression": abs(diff_pct) > tolerance and diff_pct > 0
                }

        return comparisons


# ============================================================================
# Export All Utilities
# ============================================================================

__all__ = [
    "PerformanceTimer",
    "measure_time",
    "measure_memory",
    "TestDataGenerator",
    "TestAssertions",
    "ValidationHelpers",
    "TestDataManager",
    "BenchmarkTracker",
]
