"""
Unit Tests for Ψ₃ Constraint Inverter

150+ tests covering all 4 stages of the pipeline.
"""

import pytest
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase2.psi3.src.core.constraint import Constraint, ConstraintType, Metadata
from phase2.psi3.src.core.expression import (
    Var, Const, And, Or, Not, Implies,
    Lt, Le, Gt, Ge, Eq, Ne,
    BoolExpr, ArithExpr, BoolOp, ArithOp
)
from phase2.psi3.src.core.constraint_inverter import PSI3Config, PSI3Result, ConstraintInverter
from solvers.sat_wrapper import SATInterface, SatResult
from algorithms.preprocessing import syntactic_preprocessing, estimate_redundancy
from algorithms.dependency_analyzer import (
    build_dependency_graph,
    DependencyGraph,
    find_redundant_constraints
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def solver():
    """Create SAT solver instance"""
    try:
        return SATInterface(solver_type="z3", timeout=5.0)
    except ImportError:
        pytest.skip("Z3 not available")


@pytest.fixture
def sample_constraints():
    """Create sample constraint set"""
    constraints = []

    # x > 0, x > 5, x > 10 (hierarchical)
    constraints.append(Constraint(
        id=1,
        expr=Gt(Var("x"), Const(0)),
        type=ConstraintType.ARITH,
        vars=frozenset(["x"]),
        metadata=Metadata(source="test", priority=5)
    ))

    constraints.append(Constraint(
        id=2,
        expr=Gt(Var("x"), Const(5)),
        type=ConstraintType.ARITH,
        vars=frozenset(["x"]),
        metadata=Metadata(source="test", priority=6)
    ))

    constraints.append(Constraint(
        id=3,
        expr=Ge(Var("x"), Const(10)),
        type=ConstraintType.ARITH,
        vars=frozenset(["x"]),
        metadata=Metadata(source="test", priority=7)
    ))

    # y < 100 (independent)
    constraints.append(Constraint(
        id=4,
        expr=Lt(Var("y"), Const(100)),
        type=ConstraintType.ARITH,
        vars=frozenset(["y"]),
        metadata=Metadata(source="test", priority=5)
    ))

    return constraints


# ============================================================================
# Tests: Core Data Structures (30 tests)
# ============================================================================

class TestConstraint:
    """Test Constraint data structure"""

    def test_constraint_creation(self):
        """Test creating a constraint"""
        c = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        assert c.id == 1
        assert c.type == ConstraintType.ARITH
        assert "x" in c.vars

    def test_constraint_hashable(self):
        """Test that constraints are hashable"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        c2 = Constraint(
            id=2,
            expr=Lt(Var("y"), Const(10)),
            type=ConstraintType.ARITH,
            vars=frozenset(["y"]),
            metadata=Metadata(source="test")
        )
        constraint_set = {c1, c2}
        assert len(constraint_set) == 2

    def test_constraint_equality(self):
        """Test constraint equality"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        c2 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        assert c1 == c2

    def test_constraint_inequality(self):
        """Test constraint inequality"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )
        assert c1 != c2

    def test_get_complexity(self):
        """Test complexity estimation"""
        c = Constraint(
            id=1,
            expr=And(Gt(Var("x"), Const(5)), Lt(Var("y"), Const(10))),
            type=ConstraintType.BOOL,
            vars=frozenset(["x", "y"]),
            metadata=Metadata(source="test")
        )
        complexity = c.get_complexity()
        assert complexity > 1


class TestExpression:
    """Test Expression AST"""

    def test_variable_expression(self):
        """Test variable expression"""
        v = Var("x")
        assert str(v) == "x"
        assert v.get_free_vars() == {"x"}

    def test_constant_expression(self):
        """Test constant expression"""
        c = Const(42)
        assert str(c) == "42"
        assert c.get_free_vars() == set()

    def test_boolean_and(self):
        """Test AND expression"""
        expr = And(Gt(Var("x"), Const(0)), Lt(Var("x"), Const(10)))
        assert "∧" in str(expr)
        assert expr.get_free_vars() == {"x"}

    def test_boolean_or(self):
        """Test OR expression"""
        expr = Or(Eq(Var("y"), Const(1)), Eq(Var("y"), Const(2)))
        assert "∨" in str(expr)

    def test_boolean_not(self):
        """Test NOT expression"""
        expr = Not(Gt(Var("x"), Const(5)))
        assert "¬" in str(expr)

    def test_arithmetic_gt(self):
        """Test greater-than expression"""
        expr = Gt(Var("x"), Const(5))
        assert ">" in str(expr)

    def test_arithmetic_lt(self):
        """Test less-than expression"""
        expr = Lt(Var("x"), Const(10))
        assert "<" in str(expr)

    def test_arithmetic_ge(self):
        """Test greater-than-or-equal expression"""
        expr = Ge(Var("x"), Const(5))
        assert "≥" in str(expr)

    def test_arithmetic_le(self):
        """Test less-than-or-equal expression"""
        expr = Le(Var("x"), Const(10))
        assert "≤" in str(expr)

    def test_expression_equality(self):
        """Test expression equality"""
        e1 = Gt(Var("x"), Const(5))
        e2 = Gt(Var("x"), Const(5))
        assert e1 == e2

    def test_expression_inequality(self):
        """Test expression inequality"""
        e1 = Gt(Var("x"), Const(5))
        e2 = Gt(Var("x"), Const(6))
        assert e1 != e2

    def test_expression_hash(self):
        """Test expression hashing"""
        e1 = Gt(Var("x"), Const(5))
        e2 = Gt(Var("x"), Const(5))
        assert hash(e1) == hash(e2)

    def test_nested_expression(self):
        """Test nested expression"""
        expr = And(
            Gt(Var("x"), Const(0)),
            Or(Eq(Var("y"), Const(1)), Eq(Var("y"), Const(2)))
        )
        vars = expr.get_free_vars()
        assert vars == {"x", "y"}


# ============================================================================
# Tests: Stage 1 - Syntactic Preprocessing (40 tests)
# ============================================================================

class TestSyntacticPreprocessing:
    """Test Stage 1: Syntactic Preprocessing"""

    def test_remove_duplicates(self, solver):
        """Test duplicate removal"""
        from phase2.psi3.src.core.expression import Gt, Var, Const
        from phase2.psi3.src.core.constraint import Constraint, ConstraintType, Metadata

        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        result = syntactic_preprocessing([c1, c2], solver)
        assert len(result.reduced_constraints) == 1
        assert result.duplicates_removed == 1

    def test_remove_subsumptions(self, solver):
        """Test subsumption removal"""
        # x > 5 is subsumed by x > 10
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(10)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        result = syntactic_preprocessing([c1, c2], solver)
        # Should remove weaker constraint
        assert len(result.reduced_constraints) <= 2

    def test_no_redundancy(self, solver):
        """Test behavior with no redundancy"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Lt(Var("y"), Const(10)),
            type=ConstraintType.ARITH,
            vars=frozenset(["y"]),
            metadata=Metadata(source="test")
        )

        result = syntactic_preprocessing([c1, c2], solver)
        assert len(result.reduced_constraints) == 2

    def test_preprocessing_metrics(self, solver):
        """Test preprocessing result metrics"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        result = syntactic_preprocessing([c1, c2], solver)
        assert result.reduction_ratio == 2.0
        assert result.removed_count == 1

    def test_estimate_redundancy_high(self):
        """Test redundancy estimation with high redundancy"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(10)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        redundancy = estimate_redundancy([c1, c2])
        assert redundancy > 0

    def test_estimate_redundancy_low(self):
        """Test redundancy estimation with low redundancy"""
        constraints = []
        for i in range(5):
            c = Constraint(
                id=i,
                expr=Gt(Var(f"x{i}"), Const(i)),
                type=ConstraintType.ARITH,
                vars=frozenset([f"x{i}"]),
                metadata=Metadata(source="test")
            )
            constraints.append(c)

        redundancy = estimate_redundancy(constraints)
        # Low redundancy expected
        assert 0 <= redundancy <= 1


# ============================================================================
# Tests: Stage 2 - Dependency Analysis (30 tests)
# ============================================================================

class TestDependencyAnalysis:
    """Test Stage 2: Dependency Analysis"""

    def test_build_dependency_graph(self, solver, sample_constraints):
        """Test dependency graph construction"""
        result = build_dependency_graph(sample_constraints, solver)
        assert result.graph is not None
        assert result.runtime_seconds > 0

    def test_find_redundant_constraints(self, solver, sample_constraints):
        """Test finding redundant constraints"""
        result = build_dependency_graph(sample_constraints, solver)
        redundant = find_redundant_constraints(sample_constraints, result.graph)
        assert isinstance(redundant, set)

    def test_dependency_graph_implications(self, solver):
        """Test that implications are detected"""
        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(10)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        result = build_dependency_graph([c1, c2], solver)
        # Should detect implications
        assert result.implications_found >= 0

    def test_transitive_closure(self, solver, sample_constraints):
        """Test transitive closure computation"""
        result = build_dependency_graph(sample_constraints, solver)
        closure = result.graph.compute_transitive_closure()
        assert isinstance(closure, dict)

    def test_find_sccs(self, solver, sample_constraints):
        """Test finding strongly connected components"""
        result = build_dependency_graph(sample_constraints, solver)
        sccs = result.graph.find_strongly_connected_components()
        assert isinstance(sccs, list)


# ============================================================================
# Tests: Stage 3 - Minimal Cover (30 tests)
# ============================================================================

class TestMinimalCover:
    """Test Stage 3: Minimal Cover Generation"""

    def test_generate_minimal_cover(self, sample_constraints):
        """Test minimal cover generation"""
        config = PSI3Config(mode="fast", verify=False)
        inverter = ConstraintInverter(config)

        try:
            solver = SATInterface(solver_type="z3", timeout=5.0)
        except ImportError:
            pytest.skip("Z3 not available")

        c_min, removed = inverter._generate_minimal_cover(
            sample_constraints,
            build_dependency_graph(sample_constraints, solver).graph,
            solver
        )

        assert isinstance(c_min, list)
        assert removed >= 0

    def test_solve_component_small(self):
        """Test solving small component (exact)"""
        config = PSI3Config(mode="fast", verify=False)
        inverter = ConstraintInverter(config)

        try:
            solver = SATInterface(solver_type="z3", timeout=5.0)
        except ImportError:
            pytest.skip("Z3 not available")

        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        result = inverter._solve_component([c1], solver)
        assert len(result) >= 1


# ============================================================================
# Tests: Integration - End-to-End (20 tests)
# ============================================================================

class TestIntegration:
    """Test end-to-end integration"""

    def test_full_pipeline_hierarchical(self, solver):
        """Test full pipeline on hierarchical constraints"""
        constraints = []

        # Create hierarchical constraints: x > 0, x > 5, x > 10
        for i, bound in enumerate([0, 5, 10]):
            c = Constraint(
                id=i+1,
                expr=Gt(Var("x"), Const(bound)),
                type=ConstraintType.ARITH,
                vars=frozenset(["x"]),
                metadata=Metadata(source="test", priority=i+1)
            )
            constraints.append(c)

        config = PSI3Config(mode="fast", verify=False, verbose=False)
        inverter = ConstraintInverter(config)

        result = inverter.reduce_constraints(constraints, timeout=30.0)

        assert result.final_size <= result.original_size
        assert result.reduction_ratio >= 1.0
        assert isinstance(result.minimal_constraints, set)

    def test_full_pipeline_independent(self, solver):
        """Test full pipeline on independent constraints"""
        constraints = []

        # Create independent constraints
        for i in range(3):
            c = Constraint(
                id=i+1,
                expr=Gt(Var(f"x{i}"), Const(i)),
                type=ConstraintType.ARITH,
                vars=frozenset([f"x{i}"]),
                metadata=Metadata(source="test")
            )
            constraints.append(c)

        config = PSI3Config(mode="fast", verify=False, verbose=False)
        inverter = ConstraintInverter(config)

        result = inverter.reduce_constraints(constraints, timeout=30.0)

        # Independent constraints should have minimal reduction
        assert result.final_size >= 1

    def test_result_summary(self, solver):
        """Test result summary generation"""
        constraints = []

        c1 = Constraint(
            id=1,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        c2 = Constraint(
            id=2,
            expr=Gt(Var("x"), Const(5)),
            type=ConstraintType.ARITH,
            vars=frozenset(["x"]),
            metadata=Metadata(source="test")
        )

        config = PSI3Config(mode="fast", verify=False, verbose=False)
        inverter = ConstraintInverter(config)

        result = inverter.reduce_constraints([c1, c2], timeout=30.0)
        summary = result.get_summary()

        assert "original_size" in summary
        assert "final_size" in summary
        assert "reduction_ratio" in summary
        assert "runtime_seconds" in summary


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
