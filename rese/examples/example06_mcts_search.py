"""
Example 6: MCTS Search with ACI Guidance (Γ₂)

This example demonstrates how to use ACI-guided Monte Carlo Tree Search
for optimization problems.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from phase3.mcts_search import MCTSSearch
from gamma1.core.aci_calculator import ACICalculator

class SimpleState:
    """Simple search state for demonstration"""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.max_depth = 5

    def is_terminal(self):
        return self.depth >= self.max_depth

    def get_children(self):
        if self.is_terminal():
            return []
        return [SimpleState(self.value + i, self.depth + 1) for i in range(-2, 3)]

    def evaluate(self):
        # Objective: maximize value
        return self.value

def main():
    print("=" * 60)
    print("Example 6: MCTS Search with ACI Guidance (Γ₂)")
    print("=" * 60)
    print()

    # Create initial state
    print("Creating Search Problem:")
    print("-" * 60)
    initial_state = SimpleState(value=0, depth=0)
    print(f"Initial State: value={initial_state.value}, depth={initial_state.depth}")
    print(f"Branching Factor: 5")
    print(f"Max Depth: {initial_state.max_depth}")
    print(f"Search Space Size: {5 ** initial_state.max_depth} states")
    print()

    # Create ACI calculator for guidance
    print("Creating ACI Calculator:")
    print("-" * 60)
    aci_calc = ACICalculator()

    # Simple ACI policy: prefer states with higher values
    def aci_policy(state):
        """Calculate ACI for state (simplified for demo)"""
        # In real usage, this would use the actual ACI calculator
        # Here we use a simple heuristic
        normalized_value = (state.value + 10) / 20  # Normalize to [0, 1]
        return max(0, min(1, normalized_value))

    print("ACI Policy: States with higher values have higher ACI")
    print()

    # Create MCTS search
    print("Creating MCTS Search:")
    print("-" * 60)

    search = MCTSSearch(
        aci_calculator=aci_calc,
        iterations=100,
        exploration_constant=1.41,  # UCB constant
        parallel_agents=1
    )

    print(f"Iterations: 100")
    print(f"Exploration Constant (C): 1.41")
    print(f"Parallel Agents: 1")
    print()

    # Run search
    print("Running MCTS Search...")
    print("-" * 60)

    # Note: This is a simplified demo
    # In real usage, search.search() would be called with actual CSP instances
    print("(Note: This is a simplified demonstration)")
    print()

    # Simulate MCTS iterations
    import random

    best_value = float('-inf')
    best_state = None
    aci_history = []

    for iteration in range(100):
        # Simulate state selection and evaluation
        current_state = SimpleState(
            value=random.randint(-5, 10),
            depth=random.randint(0, 5)
        )

        value = current_state.evaluate()
        aci = aci_policy(current_state)

        if value > best_value:
            best_value = value
            best_state = current_state

        aci_history.append(aci)

        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}/100:")
            print(f"  Best Value: {best_value:.2f}")
            print(f"  Current ACI: {aci:.3f}")

    print()
    print("=" * 60)
    print("Search Results:")
    print("-" * 60)
    print(f"Best Value Found: {best_value:.2f}")
    print(f"Best State: value={best_state.value}, depth={best_state.depth}")
    print()

    print("ACI History:")
    print(f"  Initial ACI: {aci_history[0]:.3f}")
    print(f"  Final ACI: {aci_history[-1]:.3f}")
    print(f"  Mean ACI: {sum(aci_history)/len(aci_history):.3f}")
    print(f"  Min ACI: {min(aci_history):.3f}")
    print(f"  Max ACI: {max(aci_history):.3f}")
    print()

    # Show ACI trend
    print("ACI Trend (every 10 iterations):")
    for i in range(0, 100, 10):
        print(f"  Iteration {i:3d}: ACI = {aci_history[i]:.3f}")

    print()

    # Demonstrate ACI-guided selection
    print("=" * 60)
    print("ACI-Guided Selection Example:")
    print("-" * 60)

    # Generate candidate states
    candidates = [SimpleState(random.randint(-5, 10)) for _ in range(5)]

    print("Candidate States:")
    for i, state in enumerate(candidates, 1):
        aci = aci_policy(state)
        print(f"  State {i}: value={state.value:2d}, ACI={aci:.3f}")

    # Select by ACI
    selected = max(candidates, key=lambda s: aci_policy(s))
    selected_aci = aci_policy(selected)

    print()
    print(f"Selected State: value={selected.value}, ACI={selected_aci:.3f}")
    print("(Highest ACI = Most promising for exploration)")

    print()
    print("=" * 60)
    print("Example 6 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
