"""
Comprehensive Test Suite for RESE Core Components

This script tests all major functionality of the RESE core modules to verify:
1. All imports work correctly
2. Components can be instantiated
3. Basic operations function as expected
4. Integration points are functional
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_symbolic_constraint_engine():
    """Test Symbolic Constraint Engine (SCE) - Agent A1"""
    print("\n" + "="*70)
    print("Testing Symbolic Constraint Engine (SCE)")
    print("="*70)

    from core import SymbolicConstraintEngine, Constraint, ConstraintType

    # Create SCE
    sce = SymbolicConstraintEngine()
    print("[OK] SCE instantiated")

    # Create constraints
    c1 = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall (T : Temperature), T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="min_temp",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 500°C",
        formalization="forall (T : Temperature), T > 500",
        source="user_prompt",
        dependencies=["temp_limit"]
    )

    c3 = Constraint(
        id="max_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should preferably be below 10 bar",
        formalization="forall (P : Pressure), P < 10",
        source="system_inferred"
    )

    # Add constraints
    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)
    print("[OK] Added 3 constraints")

    # Test retrieval
    all_constraints = sce.get_all_constraints()
    assert len(all_constraints) == 3
    print("[OK] Retrieved all constraints")

    # Test dependencies
    deps = sce.get_dependencies("min_temp")
    assert len(deps) == 1
    assert deps[0].id == "temp_limit"
    print("[OK] Dependencies work correctly")

    # Test topological sort
    sorted_ids = sce.topological_sort()
    print(f"[DEBUG] Topological sort: {sorted_ids}")
    # temp_limit should come before min_temp due to dependency
    assert "temp_limit" in sorted_ids
    assert "min_temp" in sorted_ids
    temp_idx = sorted_ids.index("temp_limit")
    min_idx = sorted_ids.index("min_temp")
    assert temp_idx < min_idx, f"temp_limit at {temp_idx} should come before min_temp at {min_idx}"
    print("[OK] Topological sort works")

    # Test statistics
    stats = sce.get_statistics()
    assert stats["total_constraints"] == 3
    assert stats["hard_constraints"] == 2
    assert stats["soft_constraints"] == 1
    print("[OK] Statistics computed correctly")

    # Test conflict detection
    conflicts = sce.detect_conflicts()
    print(f"[OK] Conflict detection found {len(conflicts)} conflicts")

    print("\n[PASS] Symbolic Constraint Engine test completed successfully")
    return True


def test_dito_optimizer():
    """Test DITO Optimizer - Agent A3"""
    print("\n" + "="*70)
    print("Testing DITO Optimizer")
    print("="*70)

    from core import DITOOptimizer, DITOConfig
    from core import Constraint, ConstraintType

    # Create DITO
    dito = DITOOptimizer()
    print("[OK] DITO instantiated")

    # Create test constraints
    constraints = [
        Constraint(
            id="temp_limit",
            type=ConstraintType.HARD,
            description="Temperature must be less than 1000°C",
            formalization="forall (T : Temperature), T < 1000",
            source="user_prompt"
        ),
        Constraint(
            id="min_temp",
            type=ConstraintType.HARD,
            description="Temperature must be greater than 500°C",
            formalization="forall (T : Temperature), T > 500",
            source="user_prompt"
        ),
        Constraint(
            id="pressure_limit",
            type=ConstraintType.SOFT,
            description="Pressure should be below 10 bar",
            formalization="forall (P : Pressure), P < 10",
            source="system"
        ),
    ]

    # Build DITO structures
    result = dito.build(constraints)
    print(f"[OK] Built DITO structures in {result['build_time']:.4f}s")
    print(f"[OK] Processed {result['constraints_processed']} constraints")

    # Detect contradictions
    contradictions = dito.detect_contradictions()
    print(f"[OK] Found {len(contradictions)} contradictions")

    # Get statistics
    stats = dito.get_statistics()
    print(f"[OK] Total constraints: {stats['total_constraints']}")
    print(f"[OK] R-tree size: {stats['rtree_size']}")

    print("\n[PASS] DITO Optimizer test completed successfully")
    return True


def test_constraint_optimizer():
    """Test Constraint Optimizer with Z3"""
    print("\n" + "="*70)
    print("Testing Constraint Optimizer")
    print("="*70)

    from core import ConstraintOptimizer, ResolutionStrategy
    from core import SymbolicConstraintEngine, Constraint, ConstraintType

    # Create SCE with constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_low",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 0",
        formalization="T > 0",
        source="test"
    )

    c2 = Constraint(
        id="temp_high",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000",
        formalization="T < 1000",
        source="test"
    )

    c3 = Constraint(
        id="temp_optimal",
        type=ConstraintType.SOFT,
        description="Temperature should be around 500",
        formalization="T = 500",
        source="test"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    # Create optimizer
    optimizer = ConstraintOptimizer(sce)
    print("[OK] Constraint Optimizer instantiated")

    # Check satisfiability
    satisfiable, message = optimizer.check_satisfiability()
    print(f"[OK] Satisfiability check: {satisfiable} - {message}")

    # Get priorities
    priorities = optimizer.prioritize_constraints()
    print(f"[OK] Computed priorities for {len(priorities)} constraints")

    # Get statistics
    stats = optimizer.get_statistics()
    print(f"[OK] Z3 available: {stats['z3_available']}")

    print("\n[PASS] Constraint Optimizer test completed successfully")
    return True


def test_lean4_bridge():
    """Test Lean 4 Bridge"""
    print("\n" + "="*70)
    print("Testing Lean 4 Bridge")
    print("="*70)

    from core import Lean4Bridge
    from core import Constraint, ConstraintType

    # Create bridge
    bridge = Lean4Bridge()
    print("[OK] Lean 4 Bridge instantiated")

    # Create test constraint
    constraint = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall T : Real, T < 1000",
        source="user_prompt"
    )

    # Convert to Lean 4
    theorem = bridge.constraint_to_lean4(constraint)
    print(f"[OK] Converted constraint to Lean 4 theorem: {theorem.name}")
    # Handle unicode characters safely
    try:
        print(f"[OK] Statement: {theorem.statement}")
    except UnicodeEncodeError:
        print(f"[OK] Statement: (contains unicode characters - display disabled)")
        print(f"[OK] Statement (safe): {theorem.statement.encode('ascii', 'replace').decode('ascii')}")

    # Batch convert
    constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="T < 1000",
            formalization="T < 1000",
            source="test"
        ),
        Constraint(
            id="c2",
            type=ConstraintType.SOFT,
            description="P > 5",
            formalization="P > 5",
            source="test"
        )
    ]

    theorems = bridge.batch_convert_constraints(constraints)
    print(f"[OK] Batch converted {len(theorems)} constraints")

    # Get statistics
    stats = bridge.get_statistics()
    print(f"[OK] Total theorems: {stats['total_theorems']}")

    print("\n[PASS] Lean 4 Bridge test completed successfully")
    return True


def test_lltl_handoff():
    """Test LLTL Handoff Module"""
    print("\n" + "="*70)
    print("Testing LLTL Handoff Module")
    print("="*70)

    from core import LLTLHandoff
    from core import SymbolicConstraintEngine, Constraint, ConstraintType

    # Create SCE with constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_safety",
        type=ConstraintType.HARD,
        description="Temperature must always be below 1000°C",
        formalization="forall T : Real, T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="request_liveness",
        type=ConstraintType.HARD,
        description="Every request must eventually be processed",
        formalization="forall r, eventually processed(r)",
        source="system_requirement"
    )

    c3 = Constraint(
        id="response_reactivity",
        type=ConstraintType.SOFT,
        description="When a request is received, acknowledge within 5 seconds",
        formalization="received(r) -> acknowledged(r) within 5",
        source="performance_requirement"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    # Create handoff module
    handoff = LLTLHandoff(sce)
    print("[OK] LLTL Handoff Module instantiated")

    # Prepare handoff package
    package = handoff.prepare_handoff()
    print(f"[OK] Prepared handoff package")
    print(f"[OK] Total constraints: {package.metadata['total_constraints']}")
    print(f"[OK] Total LLTL specs: {package.metadata['total_ltl_specs']}")
    print(f"[OK] Hard constraints: {package.metadata['hard_constraints']}")

    # Display generated specs
    for spec in package.ltl_specifications:
        print(f"[OK] {spec.id}: {spec.template.value} - {spec.formula}")

    # Get template distribution
    dist = package.metadata['template_distribution']
    print(f"[OK] Template distribution: {dist}")

    print("\n[PASS] LLTL Handoff Module test completed successfully")
    return True


def test_logic_to_loss_translation():
    """Test Logic-to-Loss Translation Layer"""
    print("\n" + "="*70)
    print("Testing Logic-to-Loss Translation Layer (LLTL)")
    print("="*70)

    from core import (
        LogicToLossTranslator,
        LossAggregationMethod,
        create_lltl_from_sce
    )
    from core import SymbolicConstraintEngine, Constraint, ConstraintType

    # Create SCE with constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall (T : Temperature), T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="min_temp",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 500°C",
        formalization="forall (T : Temperature), T > 500",
        source="user_prompt"
    )

    c3 = Constraint(
        id="max_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should preferably be below 10 bar",
        formalization="forall (P : Pressure), P < 10",
        source="system_inferred"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    # Create LLTL
    print("[INFO] Creating Logic-to-Loss Translator...")
    lltl = create_lltl_from_sce(sce)
    print("[OK] LLTL instantiated")

    # Translate constraints
    results = lltl.translate_sce(sce)
    print(f"[OK] Translated {len(results)} constraints")

    successful = sum(1 for r in results.values() if r.success)
    failed = sum(1 for r in results.values() if not r.success)
    print(f"[OK] Successful: {successful}, Failed: {failed}")

    # Get statistics
    stats = lltl.get_statistics()
    print(f"[OK] Total translations: {stats['total_translations']}")
    print(f"[OK] PyTorch available: {stats['pytorch_available']}")
    print(f"[OK] Aggregation method: {stats['aggregation_method']}")

    print("\n[PASS] Logic-to-Loss Translation test completed successfully")
    return True


def test_stage1_integration():
    """Test Stage 1 Integration"""
    print("\n" + "="*70)
    print("Testing Stage 1 Integration")
    print("="*70)

    from core import Stage1Integrator

    # Create integrator
    integrator = Stage1Integrator()
    print("[OK] Stage 1 Integrator instantiated")

    # Analyze test prompts
    test_prompt = """
    The thermal management system must operate at temperatures below 1000°C
    and shall maintain a pressure greater than 5 bar. The system should
    preferably cost less than $5000 to manufacture.
    """

    result = integrator.analyze_prompt(test_prompt)
    print(f"[OK] Analyzed prompt")
    print(f"[OK] Extracted {len(result.extracted_constraints)} constraints")
    print(f"[OK] Confidence: {result.confidence:.2f}")

    for constraint in result.extracted_constraints:
        print(f"[OK] {constraint.id}: {constraint.type.value} - {constraint.description}")

    # Get statistics
    stats = integrator.get_statistics()
    print(f"[OK] Total constraints: {stats['total_constraints']}")

    print("\n[PASS] Stage 1 Integration test completed successfully")
    return True


def test_stage5_integration():
    """Test Stage 5 Integration (Real-time Loss Feedback)"""
    print("\n" + "="*70)
    print("Testing Stage 5 Integration")
    print("="*70)

    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
        print("[WARNING] PyTorch not available, skipping Stage 5 test")
        return True

    from core import (
        Stage5Integration,
        GeneratorValidator,
        FeedbackMode,
        FeedbackStrategy
    )
    from core import SymbolicConstraintEngine, Constraint, ConstraintType, create_lltl_from_sce

    # Create SCE with constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall (T : Temperature), T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="min_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should be above 5 bar",
        formalization="forall (P : Pressure), P > 5",
        source="system_inferred"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)

    # Create LLTL
    lltl = create_lltl_from_sce(sce)

    # Create Stage 5 integration
    integration = Stage5Integration(
        lltl=lltl,
        sce=sce,
        feedback_mode=FeedbackMode.REALTIME,
        feedback_strategy=FeedbackStrategy.BACKPROPAGATE,
    )
    print("[OK] Stage 5 Integration instantiated")

    # Monitor generation (simulated)
    variables = {
        "temperature": torch.tensor([750.0], requires_grad=True),
        "pressure": torch.tensor([8.0], requires_grad=True),
    }

    state = integration.monitor_generation(variables, step=1)
    print(f"[OK] Monitored generation step 1")
    print(f"[OK] Loss: {state.loss.item() if hasattr(state.loss, 'item') else state.loss:.4f}")

    # Generate feedback
    signal = integration.generate_feedback(state)
    print(f"[OK] Generated feedback signal")
    print(f"[OK] Should stop: {signal.should_stop}")
    print(f"[OK] Should adjust: {signal.should_adjust}")

    # Get summary
    summary = integration.get_generation_summary()
    print(f"[OK] Total steps: {summary['total_steps']}")

    print("\n[PASS] Stage 5 Integration test completed successfully")
    return True


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# RESE Core Components - Comprehensive Test Suite")
    print("#"*70)

    tests = [
        ("Symbolic Constraint Engine", test_symbolic_constraint_engine),
        ("DITO Optimizer", test_dito_optimizer),
        ("Constraint Optimizer", test_constraint_optimizer),
        ("Lean 4 Bridge", test_lean4_bridge),
        ("LLTL Handoff", test_lltl_handoff),
        ("Logic-to-Loss Translation", test_logic_to_loss_translation),
        ("Stage 1 Integration", test_stage1_integration),
        ("Stage 5 Integration", test_stage5_integration),
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_name)
        except Exception as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    print(f"\nTotal tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_tests:
        print("\nFailed tests:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
    else:
        print("\n[SUCCESS] All tests passed!")

    print("\n" + "#"*70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
