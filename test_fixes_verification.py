#!/usr/bin/env python3
"""
Test script to verify the 5 fixes in problem_fractal_pipeline.py
"""

import sys

def test_imports():
    """Test Fix #1: import uuid at line 26"""
    print("\n" + "="*60)
    print("FIX #1: Testing 'import uuid' presence")
    print("="*60)

    try:
        # Read the file and check line 26
        with open("problem_fractal_pipeline.py", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check line 26 (0-indexed, so line 26 is index 25)
        line_26 = lines[25].strip() if len(lines) > 25 else ""

        print(f"Line 26 content: '{line_26}'")

        if "import uuid" in line_26:
            print("[OK] CONFIRMED: 'import uuid' found at line 26")
            return True
        else:
            # Check if it appears anywhere in first 30 lines
            found = False
            for i, line in enumerate(lines[:30], 1):
                if "import uuid" in line.strip():
                    print(f"[WARN] FOUND BUT NOT AT LINE 26: Found at line {i}")
                    found = True
                    break
            if not found:
                print("[FAIL] NOT FOUND: 'import uuid' not found in first 30 lines")
            return found

    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def test_sub_problem_type():
    """Test Fix #2: SubProblemType enum values"""
    print("\n" + "="*60)
    print("FIX #2: Testing SubProblemType enum values")
    print("="*60)

    try:
        from problem_fractal_pipeline import SubProblemType

        # Test accessing the enum values
        impl = SubProblemType.IMPLEMENTATION
        analysis = SubProblemType.ANALYSIS
        validation = SubProblemType.VALIDATION

        print(f"SubProblemType.IMPLEMENTATION = {impl}")
        print(f"SubProblemType.ANALYSIS = {analysis}")
        print(f"SubProblemType.VALIDATION = {validation}")

        # Verify they have the correct values
        assert impl == "IMPLEMENTATION", "IMPLEMENTATION has wrong value"
        assert analysis == "ANALYSIS", "ANALYSIS has wrong value"
        assert validation == "VALIDATION", "VALIDATION has wrong value"

        print("[OK] CONFIRMED: All three SubProblemType enum values exist and work")
        return True

    except ImportError as e:
        print(f"[FAIL] IMPORT ERROR: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] ASSERTION ERROR: {e}")
        return False
    except AttributeError as e:
        print(f"[FAIL] ATTRIBUTE ERROR: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def test_complexity_score():
    """Test Fix #3: ComplexityScore.overall_complexity field"""
    print("\n" + "="*60)
    print("FIX #3: Testing ComplexityScore.overall_complexity field")
    print("="*60)

    try:
        from problem_fractal_pipeline import ComplexityScore

        # Create an instance
        score = ComplexityScore(
            explanation="Test",
            cognitive_complexity=1.0,
            computational_complexity=2.0,
            domain_complexity=3.0,
            integration_complexity=4.0,
            overall_complexity=5.0
        )

        print(f"Created ComplexityScore with overall_complexity = {score.overall_complexity}")

        # Verify the field exists and has the correct value
        assert hasattr(score, "overall_complexity"), "overall_complexity field missing"
        assert score.overall_complexity == 5.0, "overall_complexity has wrong value"

        print("[OK] CONFIRMED: ComplexityScore has overall_complexity field")
        return True

    except ImportError as e:
        print(f"[FAIL] IMPORT ERROR: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] ASSERTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def test_dependency_graph():
    """Test Fix #4: DependencyGraph.execution_order field"""
    print("\n" + "="*60)
    print("FIX #4: Testing DependencyGraph.execution_order field")
    print("="*60)

    try:
        from problem_fractal_pipeline import DependencyGraph

        # Create an instance without execution_order (should use default)
        graph1 = DependencyGraph(
            nodes={},
            edges={}
        )

        print(f"Created DependencyGraph with default execution_order = {graph1.execution_order}")

        # Verify the field exists and default works
        assert hasattr(graph1, "execution_order"), "execution_order field missing"
        assert graph1.execution_order == [], "Default execution_order should be empty list"

        # Create with explicit execution_order
        graph2 = DependencyGraph(
            nodes={},
            edges={},
            execution_order=["a", "b", "c"]
        )

        print(f"Created DependencyGraph with explicit execution_order = {graph2.execution_order}")

        assert graph2.execution_order == ["a", "b", "c"], "execution_order has wrong value"

        print("[OK] CONFIRMED: DependencyGraph has execution_order field with default_factory=list")
        return True

    except ImportError as e:
        print(f"[FAIL] IMPORT ERROR: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] ASSERTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def test_sovereign_decomposition_strategy():
    """Test Fix #5: SovereignDecompositionStrategy class"""
    print("\n" + "="*60)
    print("FIX #5: Testing SovereignDecompositionStrategy class")
    print("="*60)

    try:
        from problem_fractal_pipeline import SovereignDecompositionStrategy

        # Test accessing the class attributes
        hybrid = SovereignDecompositionStrategy.HYBRID
        roma = SovereignDecompositionStrategy.ROMA
        semantic = SovereignDecompositionStrategy.SEMANTIC

        print(f"SovereignDecompositionStrategy.HYBRID = {hybrid}")
        print(f"SovereignDecompositionStrategy.ROMA = {roma}")
        print(f"SovereignDecompositionStrategy.SEMANTIC = {semantic}")

        # Verify they have the correct values
        assert hybrid == "HYBRID", "HYBRID has wrong value"
        assert roma == "ROMA", "ROMA has wrong value"
        assert semantic == "SEMANTIC", "SEMANTIC has wrong value"

        print("[OK] CONFIRMED: SovereignDecompositionStrategy class exists with all three attributes")
        return True

    except ImportError as e:
        print(f"[FAIL] IMPORT ERROR: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] ASSERTION ERROR: {e}")
        return False
    except AttributeError as e:
        print(f"[FAIL] ATTRIBUTE ERROR: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def test_integration():
    """Test that all fixes work together without regressions"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Testing all fixes work together")
    print("="*60)

    try:
        from problem_fractal_pipeline import (
            SubProblemType,
            ComplexityScore,
            DependencyGraph,
            SovereignDecompositionStrategy
        )
        import uuid

        # Use all components together
        problem_type = SubProblemType.IMPLEMENTATION
        complexity = ComplexityScore(
            explanation="Integration test",
            cognitive_complexity=1.0,
            computational_complexity=2.0,
            domain_complexity=3.0,
            integration_complexity=4.0,
            overall_complexity=5.0
        )
        dep_graph = DependencyGraph(
            nodes={"test": {}},
            edges={"test": []},
            execution_order=["test"]
        )
        strategy = SovereignDecompositionStrategy.HYBRID

        # Generate a UUID
        test_id = str(uuid.uuid4())

        print(f"Integration test successful!")
        print(f"  - Problem type: {problem_type}")
        print(f"  - Complexity: {complexity.overall_complexity}")
        print(f"  - Execution order: {dep_graph.execution_order}")
        print(f"  - Strategy: {strategy}")
        print(f"  - Generated UUID: {test_id}")

        print("[OK] CONFIRMED: All fixes work together without regressions")
        return True

    except Exception as e:
        print(f"[FAIL] INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and generate report"""
    print("="*60)
    print("VERIFICATION OF 5 FIXES IN problem_fractal_pipeline.py")
    print("="*60)

    results = {
        "Fix #1 (import uuid)": test_imports(),
        "Fix #2 (SubProblemType)": test_sub_problem_type(),
        "Fix #3 (ComplexityScore)": test_complexity_score(),
        "Fix #4 (DependencyGraph)": test_dependency_graph(),
        "Fix #5 (SovereignDecompositionStrategy)": test_sovereign_decomposition_strategy(),
        "Integration Test": test_integration(),
    }

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    for fix_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {fix_name}")

    # Overall assessment
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("OVERALL ASSESSMENT: [OK] PASS - All fixes verified!")
    else:
        print("OVERALL ASSESSMENT: [FAIL] FAIL - Some fixes are missing or broken")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
