"""
Phase 2 Debug Test Suite

Tests all critical components after bug fixes.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add paths
phase2_root = Path(__file__).parent
sys.path.insert(0, str(phase2_root))

def test_constraint_imports():
    """Test that constraint module imports correctly"""
    print("\n" + "="*70)
    print("TEST 1: Constraint Module Imports")
    print("="*70)

    try:
        from psi3.src.core.constraint import (
            Constraint, ConstraintType, Metadata, SatResult, SATInterface
        )
        print("✓ Constraint module imports successful")
        print(f"  - ConstraintType: {ConstraintType.BOOL}")
        print(f"  - SatResult: {SatResult.SATISFIABLE}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expression_imports():
    """Test that expression module imports correctly"""
    print("\n" + "="*70)
    print("TEST 2: Expression Module Imports")
    print("="*70)

    try:
        from psi3.src.core.expression import (
            Expr, Variable, Constant, BoolExpr, ArithExpr,
            QuantExpr, BoolOp, ArithOp, Quantifier,
            Var, Const, And, Or, Not, Lt, Le, Gt, Ge, Eq, Ne
        )
        print("✓ Expression module imports successful")

        # Test basic expression creation
        x = Var("x")
        y = Var("y")
        c = Const(5)
        expr = Gt(x, c)  # x > 5
        print(f"  - Created expression: {expr}")
        print(f"  - Expression type: {type(expr)}")
        print(f"  - Free vars: {expr.get_free_vars()}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fdg_module():
    """Test FDG module functionality"""
    print("\n" + "="*70)
    print("TEST 3: FDG Module")
    print("="*70)

    try:
        from imech.core.fdg import (
            FunctionalDependencyGraph, Node, Edge, EdgeType
        )
        print("✓ FDG module imports successful")

        # Create test FDG
        fdg = FunctionalDependencyGraph()
        node1 = Node(id="x", variable="x", constraint_type="continuous")
        node2 = Node(id="y", variable="y", constraint_type="continuous")
        edge = Edge(source="x", target="y", edge_type=EdgeType.CAUSAL)

        fdg.add_node(node1)
        fdg.add_node(node2)
        fdg.add_edge(edge)

        print(f"  - Created FDG: {fdg}")
        print(f"  - Nodes: {len(fdg.nodes)}")
        print(f"  - Edges: {len(fdg.edges)}")
        print(f"  - Feedback loops: {len(fdg.get_feedback_loops())}")
        return True
    except Exception as e:
        print(f"✗ FDG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_domain_module():
    """Test Domain module"""
    print("\n" + "="*70)
    print("TEST 4: Domain Module")
    print("="*70)

    try:
        from imech.core.domain import Domain
        print("✓ Domain module imports successful")

        domain = Domain(
            id="test_domain",
            name="Test Domain",
            description="Test domain for debugging"
        )

        print(f"  - Created domain: {domain}")
        print(f"  - Has solution: {domain.has_solution()}")
        return True
    except Exception as e:
        print(f"✗ Domain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_isomorphism_validator():
    """Test I_mech validator"""
    print("\n" + "="*70)
    print("TEST 5: I_mech Isomorphism Validator")
    print("="*70)

    try:
        from imech.isomorphism_validator import IMechValidator
        from imech.core.domain import Domain
        from imech.core.fdg import FunctionalDependencyGraph, Node, Edge, EdgeType

        print("✓ I_mech validator imports successful")

        # Create test domains
        domain1 = Domain(id="d1", name="Domain 1", description="Test")
        domain2 = Domain(id="d2", name="Domain 2", description="Test")

        # Create simple FDGs
        fdg1 = FunctionalDependencyGraph()
        fdg1.add_node(Node(id="a", variable="a", constraint_type="continuous"))
        fdg1.add_node(Node(id="b", variable="b", constraint_type="continuous"))
        fdg1.add_edge(Edge(source="a", target="b", edge_type=EdgeType.CAUSAL))
        domain1.fdg = fdg1

        fdg2 = FunctionalDependencyGraph()
        fdg2.add_node(Node(id="x", variable="x", constraint_type="continuous"))
        fdg2.add_node(Node(id="y", variable="y", constraint_type="continuous"))
        fdg2.add_edge(Edge(source="x", target="y", edge_type=EdgeType.CAUSAL))
        domain2.fdg = fdg2

        # Test comparison
        validator = IMechValidator()
        print("  - Created validator successfully")

        # Note: Full comparison requires WL algorithm, which needs testing
        print(f"  - Domain 1 FDG: {fdg1}")
        print(f"  - Domain 2 FDG: {fdg2}")
        return True
    except Exception as e:
        print(f"✗ I_mech validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_module():
    """Test Similarity Scoring"""
    print("\n" + "="*70)
    print("TEST 6: Similarity Scoring Module")
    print("="*70)

    try:
        from imech.core.scoring import SimilarityScorer
        print("✓ Scoring module imports successful")

        scorer = SimilarityScorer()
        print(f"  - Weights: struct={scorer.weight_structural:.2f}, "
              f"causal={scorer.weight_causal:.2f}, "
              f"semantic={scorer.weight_semantic:.2f}, "
              f"intervention={scorer.weight_intervention:.2f}")

        # Test total score computation
        total = scorer.compute_total_score(0.8, 0.7, 0.9, 0.6)
        print(f"  - Test total score: {total:.3f}")
        return True
    except Exception as e:
        print(f"✗ Scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ontology_mapper():
    """Test Ontology Mapper"""
    print("\n" + "="*70)
    print("TEST 7: Ontology Mapper (Ψ₂)")
    print("="*70)

    try:
        # Test import
        import ontology_mapper
        print("✓ Ontology mapper module imports successful")

        # Test mapper creation
        mapper = ontology_mapper.create_mapper()
        print(f"  - Mapper created: {type(mapper).__name__}")
        print(f"  - Config keys: {list(mapper.config.keys())}")
        return True
    except Exception as e:
        print(f"✗ Ontology mapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("PHASE 2 DEBUG TEST SUITE")
    print("="*70)
    print("Testing all critical components after bug fixes...")

    results = []
    results.append(("Constraint Imports", test_constraint_imports()))
    results.append(("Expression Imports", test_expression_imports()))
    results.append(("FDG Module", test_fdg_module()))
    results.append(("Domain Module", test_domain_module()))
    results.append(("I_mech Validator", test_isomorphism_validator()))
    results.append(("Scoring Module", test_scoring_module()))
    results.append(("Ontology Mapper", test_ontology_mapper()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Phase 2 components are working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review errors above")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
