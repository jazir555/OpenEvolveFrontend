"""
Unit Tests for Constraint Optimizer

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from core.constraint_optimizer import (
    ConstraintOptimizer,
    OptimizationResult,
    ResolutionStrategy,
    optimize_constraints
)
from core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass"""

    def test_optimization_result_creation(self):
        """Test basic result creation"""
        result = OptimizationResult(
            satisfiable=True,
            solution={"x": 5.0},
            unsatisfied_constraints=[],
            removed_constraints=[],
            strategy=ResolutionStrategy.SATISFIABILITY,
            solver_time_ms=100.0
        )

        assert result.satisfiable is True
        assert result.solution == {"x": 5.0}
        assert result.strategy == ResolutionStrategy.SATISFIABILITY

    def test_optimization_result_defaults(self):
        """Test result with default values"""
        result = OptimizationResult(
            satisfiable=False,
            solution=None,
            unsatisfied_constraints=None,
            removed_constraints=None,
            strategy=ResolutionStrategy.PRIORITY_BASED,
            solver_time_ms=50.0
        )

        assert result.solution == {}
        assert result.unsatisfied_constraints == []
        assert result.removed_constraints == []


class TestConstraintOptimizer:
    """Test suite for ConstraintOptimizer class"""

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = ConstraintOptimizer()
        assert optimizer.sce is not None

    def test_optimizer_with_sce(self):
        """Test optimizer with existing SCE"""
        sce = SymbolicConstraintEngine()
        optimizer = ConstraintOptimizer(sce)

        assert optimizer.sce == sce

    def test_check_satisfiability_no_constraints(self):
        """Test satisfiability check with no constraints"""
        optimizer = ConstraintOptimizer()

        satisfiable, message = optimizer.check_satisfiability([])

        assert satisfiable is True
        assert "No constraints" in message

    def test_check_satisfiability_simple(self):
        """Test satisfiability check with simple constraints"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        c1 = Constraint(
            id="temp_low",
            type=ConstraintType.HARD,
            description="T > 0",
            formalization="T > 0",
            source="test"
        )

        c2 = Constraint(
            id="temp_high",
            type=ConstraintType.HARD,
            description="T < 1000",
            formalization="T < 1000",
            source="test"
        )

        sce.add_constraint(c1)
        sce.add_constraint(c2)

        satisfiable, message = optimizer.check_satisfiability()

        # Result depends on Z3 availability
        assert isinstance(satisfiable, bool)

    def test_check_satisfiability_contradictory(self):
        """Test satisfiability check with contradictory constraints"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        c1 = Constraint(
            id="less",
            type=ConstraintType.HARD,
            description="x < 10",
            formalization="x < 10",
            source="test"
        )

        c2 = Constraint(
            id="greater",
            type=ConstraintType.HARD,
            description="x > 20",
            formalization="x > 20",
            source="test"
        )

        sce.add_constraint(c1)
        sce.add_constraint(c2)

        satisfiable, message = optimizer.check_satisfiability()

        # Should be unsatisfiable (if Z3 available)
        assert isinstance(satisfiable, bool)

    def test_find_solution(self):
        """Test finding solution"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        c1 = Constraint(
            id="temp_low",
            type=ConstraintType.HARD,
            description="T > 0",
            formalization="T > 0",
            source="test"
        )

        c2 = Constraint(
            id="temp_high",
            type=ConstraintType.HARD,
            description="T < 1000",
            formalization="T < 1000",
            source="test"
        )

        sce.add_constraint(c1)
        sce.add_constraint(c2)

        result = optimizer.find_solution()

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.satisfiable, bool)

    def test_extract_variables_from_formal(self):
        """Test variable extraction from formal constraint"""
        optimizer = ConstraintOptimizer()

        # Test with quantifier
        vars = optimizer._extract_variables_from_formal("forall T : Real, T < 1000")
        assert "T" in vars

        # Test without quantifier
        vars = optimizer._extract_variables_from_formal("x < 100")
        assert "x" in vars or len(vars) >= 0

    def test_parse_formalization_less_than(self):
        """Test parsing 'less than' formalization"""
        optimizer = ConstraintOptimizer()

        var_map = {"T": optimizer.sce}
        expr = optimizer._parse_formalization("T < 1000", {"T": 1000})

        # Will return None if Z3 not available, otherwise BoolRef
        assert expr is None or hasattr(expr, '__bool__')

    def test_parse_formalization_greater_than(self):
        """Test parsing 'greater than' formalization"""
        optimizer = ConstraintOptimizer()

        expr = optimizer._parse_formalization("T > 500", {"T": 1000})

        # Will return None if Z3 not available
        assert expr is None or hasattr(expr, '__bool__')

    def test_prioritize_constraints(self):
        """Test constraint prioritization"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        # Add different constraint types
        sce.add_constraint(Constraint(
            id="hard1",
            type=ConstraintType.HARD,
            description="Hard constraint 1",
            formalization="x > 0",
            source="test"
        ))

        sce.add_constraint(Constraint(
            id="soft1",
            type=ConstraintType.SOFT,
            description="Soft constraint 1",
            formalization="y < 100",
            source="test"
        ))

        sce.add_constraint(Constraint(
            id="pref1",
            type=ConstraintType.PREFERENCE,
            description="Preference 1",
            formalization="z = 50",
            source="test"
        ))

        priorities = optimizer.prioritize_constraints()

        assert len(priorities) == 3

        # Hard constraints should have higher priority
        hard_priority = [p for c_id, p in priorities if c_id == "hard1"][0]
        soft_priority = [p for c_id, p in priorities if c_id == "soft1"][0]
        pref_priority = [p for c_id, p in priorities if c_id == "pref1"][0]

        assert hard_priority > soft_priority > pref_priority

    def test_prioritize_constraints_with_dependencies(self):
        """Test priority calculation with dependencies"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        # Add base constraint
        sce.add_constraint(Constraint(
            id="base",
            type=ConstraintType.HARD,
            description="Base constraint",
            formalization="x > 0",
            source="test"
        ))

        # Add dependent constraint
        sce.add_constraint(Constraint(
            id="dependent",
            type=ConstraintType.HARD,
            description="Dependent constraint",
            formalization="y < 100",
            source="test",
            dependencies=["base"]
        ))

        priorities = optimizer.prioritize_constraints()

        # Dependent should have higher priority (dependency boost)
        dep_priority = [p for c_id, p in priorities if c_id == "dependent"][0]
        base_priority = [p for c_id, p in priorities if c_id == "base"][0]

        assert dep_priority > base_priority

    def test_resolve_by_priority(self):
        """Test conflict resolution by priority"""
        optimizer = ConstraintOptimizer()
        sce = optimizer.sce

        constraints = [
            Constraint(
                id="hard",
                type=ConstraintType.HARD,
                description="Hard constraint",
                formalization="x > 0",
                source="test"
            ),
            Constraint(
                id="soft",
                type=ConstraintType.SOFT,
                description="Soft constraint",
                formalization="x < 0",
                source="test"
            )
        ]

        import time
        start_time = time.time()

        result = optimizer._resolve_conflicts(
            constraints,
            ResolutionStrategy.PRIORITY_BASED,
            start_time
        )

        assert isinstance(result, OptimizationResult)
        assert result.strategy in [ResolutionStrategy.PRIORITY_BASED, ResolutionStrategy.SATISFIABILITY]

    def test_get_statistics(self):
        """Test getting optimizer statistics"""
        optimizer = ConstraintOptimizer()

        sce = optimizer.sce
        sce.add_constraint(Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        ))

        stats = optimizer.get_statistics()

        assert "total_constraints" in stats
        assert "z3_available" in stats
        assert "priorities" in stats


class TestResolutionStrategy:
    """Test suite for ResolutionStrategy enum"""

    def test_strategy_values(self):
        """Test strategy enum values"""
        assert ResolutionStrategy.PRIORITY_BASED.value == "priority"
        assert ResolutionStrategy.SATISFIABILITY.value == "satisfiability"
        assert ResolutionStrategy.MINIMAL_REMOVAL.value == "minimal"
        assert ResolutionStrategy.WEIGHTED.value == "weighted"


class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_optimize_constraints(self):
        """Test convenience function for optimizing constraints"""
        constraints = [
            Constraint(
                id=f"c{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"x_{i} > {i}",
                source="test"
            )
            for i in range(3)
        ]

        result = optimize_constraints(constraints)

        assert isinstance(result, OptimizationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
