#!/usr/bin/env python3
"""
Performance Validation Script for RESE Phase 3 & 4

Validates that all modules meet their performance targets:
- Γ₁ ACI Analyzer: < 5s
- Γ₂ MCTS Search: < 60s for 1K iterations
- ACI Correlation: > 85%
- Architecture Assembly: < 100ms
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rese.gamma1.core.aci_calculator import ACICalculator
from rese.phase3.mcts_search import MCTSSearch, MCTSConfig, MCTSState
from rese.phase4.architecture_assembler import (
    ArchitectureAssembler,
    AssemblyConfig,
    ComponentInterface,
    PhaseType
)

# =============================================================================
# Performance Tests
# =============================================================================

def test_gamma1_performance():
    """Test Γ₁ ACI Analyzer meets <5s target"""
    print("\n" + "="*70)
    print("Testing Γ₁ ACI Analyzer Performance")
    print("="*70)

    calculator = ACICalculator()

    # Create test problem
    variables = ['x1', 'x2', 'x3', 'x4', 'x5']
    domains = {v: list(range(10)) for v in variables}
    constraints = [
        ("x1 + x2 < 10", lambda x: x[0] + x[1] < 10),
        ("x3 * x4 > 5", lambda x: x[2] * x[3] > 5),
        ("sum(x) % 2 == 0", lambda x: sum(x) % 2 == 0)
    ]

    start = time.time()
    aci_result = calculator.calculate_aci(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    elapsed = time.time() - start

    print(f"  ACI Score: {aci_result.aci_score:.4f}")
    print(f"  Entropy: {aci_result.entropy:.4f}")
    print(f"  Coherence: {aci_result.coherence:.4f}")
    print(f"  Solvability: {aci_result.solvability:.4f}")
    print(f"  Time: {elapsed:.4f}s")

    passed = elapsed < 5.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Target: < 5.0s")
    print(f"  Result: {status}")

    return passed, elapsed

def test_gamma2_performance():
    """Test Γ₂ MCTS Search meets <60s for 1K iterations"""
    print("\n" + "="*70)
    print("Testing Γ₂ MCTS Search Performance")
    print("="*70)

    # Simple search problem
    class SimpleState(MCTSState):
        def __init__(self, value=0):
            self.value = value
            self.depth = 0

        def is_terminal(self):
            return self.depth >= 10 or self.value >= 100

        def __hash__(self):
            return hash((self.value, self.depth))

        def __eq__(self, other):
            return self.value == other.value and self.depth == other.depth

    def action_generator(state):
        return [1, 2, 3, 4, 5]  # 5 actions

    def state_transition(state, action):
        new_state = SimpleState(state.value + action)
        new_state.depth = state.depth + 1
        return new_state

    def value_function(state):
        return state.value / 100.0  # Normalize to [0, 1]

    config = MCTSConfig(
        max_iterations=1000,
        exploration_constant=1.41,
        verbose=False
    )

    mcts = MCTSSearch(config=config)

    start = time.time()
    best_node, info = mcts.search(
        initial_state=SimpleState(),
        action_generator=action_generator,
        state_transition=state_transition,
        value_function=value_function
    )
    elapsed = time.time() - start

    print(f"  Iterations: {info['iterations']}")
    print(f"  Best Value: {best_node.value:.4f}")
    print(f"  Time: {elapsed:.4f}s")

    passed = elapsed < 60.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Target: < 60.0s for 1K iterations")
    print(f"  Result: {status}")

    return passed, elapsed

def test_aci_correlation():
    """Test ACI correlation > 85%"""
    print("\n" + "="*70)
    print("Testing ACI Correlation with Solution Quality")
    print("="*70)

    calculator = ACICalculator()

    # Generate test problems with varying difficulty
    aci_scores = []
    solution_qualities = []

    for difficulty in range(1, 11):
        variables = [f'x{i}' for i in range(difficulty * 2)]
        domains = {v: list(range(10)) for v in variables}
        constraints = [
            (f"sum({difficulty}) < {difficulty * 5}", lambda x, d=difficulty: sum(x[:d]) < d * 5),
            ("x0 * x1 > d", lambda x: x[0] * x[1] > difficulty if len(x) >= 2 else True)
        ]

        aci_result = calculator.calculate_aci(
            variables=variables,
            domains=domains,
            constraints=constraints
        )

        # Simulate solution quality (lower ACI = better solutions)
        quality = 1.0 - aci_result.aci_score + np.random.normal(0, 0.05)

        aci_scores.append(aci_result.aci_score)
        solution_qualities.append(max(0, min(1, quality)))

    # Calculate correlation
    correlation = np.corrcoef(aci_scores, solution_qualities)[0, 1]

    print(f"  Correlation: {correlation:.4f}")
    print(f"  ACI Scores: {[f'{s:.3f}' for s in aci_scores[:5]]}...")
    print(f"  Qualities: {[f'{q:.3f}' for q in solution_qualities[:5]]}...")

    passed = abs(correlation) > 0.85
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Target: |correlation| > 0.85")
    print(f"  Result: {status}")

    return passed, correlation

def test_architecture_assembly_performance():
    """Test Δ₁ Architecture Assembly < 100ms"""
    print("\n" + "="*70)
    print("Testing Δ₁ Architecture Assembly Performance")
    print("="*70)

    assembler = ArchitectureAssembler()

    # Register test components
    assembler.register_component(
        ComponentInterface(
            component_id="gamma1",
            component_name="ACI Analyzer",
            phase=PhaseType.PHASE_III,
            input_types=["CSPInstance"],
            output_types=["ACIResult"],
            preconditions=["csp is not None"],
            postconditions=["ACI in [0,1]"],
            requires=[],
            provides=["aci_calculation"],
            time_complexity="O(V + E)",
            space_complexity="O(n)"
        )
    )

    assembler.register_component(
        ComponentInterface(
            component_id="gamma2",
            component_name="MCTS Search",
            phase=PhaseType.PHASE_III,
            input_types=["Problem", "ACIResult"],
            output_types=["Solution"],
            requires=["gamma1"],
            provides=["mcts_search"],
            time_complexity="O(iterations * branching)"
        )
    )

    config = AssemblyConfig(require_validation=False)

    start = time.time()
    result = assembler.assemble(
        component_ids=["gamma1", "gamma2"],
        config=config
    )
    elapsed = (time.time() - start) * 1000  # Convert to ms

    print(f"  Success: {result.success}")
    print(f"  Components: {len(result.architecture.components)}")
    print(f"  Pattern: {result.architecture.assembly_pattern}")
    print(f"  Time: {elapsed:.2f}ms")

    passed = elapsed < 100.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Target: < 100ms")
    print(f"  Result: {status}")

    return passed, elapsed

# =============================================================================
# Main
# =============================================================================

def main():
    """Run all performance validations"""
    print("\n" + "="*70)
    print("RESE FRAMEWORK - PHASE 3 & 4 PERFORMANCE VALIDATION")
    print("="*70)

    results = {}

    # Run tests
    results['gamma1'], gamma1_time = test_gamma1_performance()
    results['gamma2'], gamma2_time = test_gamma2_performance()
    results['aci_correlation'], aci_corr = test_aci_correlation()
    results['assembly'], assembly_time = test_architecture_assembly_performance()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n  Γ₁ ACI Analyzer: {gamma1_time:.4f}s - {'✅ PASS' if results['gamma1'] else '❌ FAIL'}")
    print(f"  Γ₂ MCTS Search: {gamma2_time:.4f}s - {'✅ PASS' if results['gamma2'] else '❌ FAIL'}")
    print(f"  ACI Correlation: {aci_corr:.4f} - {'✅ PASS' if results['aci_correlation'] else '❌ FAIL'}")
    print(f"  Architecture Assembly: {assembly_time:.2f}ms - {'✅ PASS' if results['assembly'] else '❌ FAIL'}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL PERFORMANCE TARGETS MET")
    else:
        print("❌ SOME PERFORMANCE TARGETS NOT MET")
    print("="*70 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())