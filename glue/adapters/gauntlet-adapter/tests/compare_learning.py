#!/usr/bin/env python3
"""
Comparison script showing the difference between random gradients and proper backpropagation.

This demonstrates why the fix was critical.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_learner import AdvancedAdaptiveLearner, LearningAlgorithm


def buggy_gradient_update(q_network, learning_rate, loss):
    """
    Simulate the OLD BUGGY behavior with random gradients.
    """
    learning_factor = learning_rate * 0.01
    for key in q_network:
        gradient = np.random.randn(*q_network[key].shape) * loss * learning_factor
        q_network[key] -= gradient


def create_simple_environment():
    """
    Create a simple deterministic environment for testing.
    """
    # State 0: [1, 0, 0, 0] -> action 0 gives reward +1
    # State 1: [0, 1, 0, 0] -> action 1 gives reward +1
    # State 2: [0, 0, 1, 0] -> action 2 gives reward +1
    # State 3: [0, 0, 0, 1] -> action 3 gives reward +1

    states = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ]

    actions = [0, 1, 2, 3]
    rewards = [1.0, 1.0, 1.0, 1.0]

    return states, actions, rewards


def compare_learning_methods():
    """
    Compare learning with random gradients vs proper backpropagation.
    """
    print("=" * 70)
    print("COMPARISON: Random Gradients vs Proper Backpropagation")
    print("=" * 70)

    # Environment
    states, actions, rewards = create_simple_environment()
    n_states = len(states)

    # Test parameters
    n_episodes = 50
    steps_per_episode = 4

    # ============================================================================
    # METHOD 1: BUGGY (Random Gradients)
    # ============================================================================
    print("\n" + "-" * 70)
    print("METHOD 1: BUGGY - Random Gradients (Old Behavior)")
    print("-" * 70)

    learner_buggy = AdvancedAdaptiveLearner(
        algorithm=LearningAlgorithm.DQN,
        state_size=4,
        action_size=4,
        learning_rate=0.01,
        gamma=0.9,
        batch_size=4
    )

    losses_buggy = []
    accuracies_buggy = []

    for episode in range(n_episodes):
        # Store experiences
        for i in range(n_states):
            next_state = np.zeros(4, dtype=np.float32)
            learner_buggy.remember(states[i], actions[i], rewards[i], next_state, True)

        # Simulate old buggy behavior
        if len(learner_buggy.memory) >= learner_buggy.batch_size:
            # Sample batch
            minibatch = np.random.choice(len(learner_buggy.memory), learner_buggy.batch_size, replace=False)

            states_batch = np.array([learner_buggy.memory[i].state for i in minibatch])
            actions_batch = np.array([learner_buggy.memory[i].action for i in minibatch])
            rewards_batch = np.array([learner_buggy.memory[i].reward for i in minibatch])

            # Forward pass
            current_q_values, _, _ = learner_buggy._forward(learner_buggy.q_network, states_batch)

            # Compute targets
            target_q = current_q_values.copy()
            for i in range(learner_buggy.batch_size):
                target_q[i, actions_batch[i]] = rewards_batch[i]

            # Calculate loss
            loss = np.mean((current_q_values - target_q) ** 2)
            losses_buggy.append(loss)

            # OLD BUGGY UPDATE: Random gradients
            buggy_gradient_update(learner_buggy.q_network, learner_buggy.learning_rate, loss)

        # Test accuracy
        correct = 0
        for i in range(n_states):
            q_values, _, _ = learner_buggy._forward(learner_buggy.q_network, states[i])
            if np.argmax(q_values) == actions[i]:
                correct += 1
        accuracies_buggy.append(correct / n_states)

        if episode % 10 == 0:
            print(f"Episode {episode:3d}: Loss = {losses_buggy[-1]:.4f}, Accuracy = {accuracies_buggy[-1]:.2f}")

    # ============================================================================
    # METHOD 2: FIXED (Proper Backpropagation)
    # ============================================================================
    print("\n" + "-" * 70)
    print("METHOD 2: FIXED - Proper Backpropagation (New Behavior)")
    print("-" * 70)

    learner_fixed = AdvancedAdaptiveLearner(
        algorithm=LearningAlgorithm.DQN,
        state_size=4,
        action_size=4,
        learning_rate=0.01,
        gamma=0.9,
        batch_size=4
    )

    losses_fixed = []
    accuracies_fixed = []

    for episode in range(n_episodes):
        # Store experiences
        for i in range(n_states):
            next_state = np.zeros(4, dtype=np.float32)
            learner_fixed.remember(states[i], actions[i], rewards[i], next_state, True)

        # Train with proper backpropagation
        if len(learner_fixed.memory) >= learner_fixed.batch_size:
            metrics = learner_fixed.replay()
            losses_fixed.append(metrics['loss'])

        # Test accuracy
        correct = 0
        for i in range(n_states):
            q_values, _, _ = learner_fixed._forward(learner_fixed.q_network, states[i])
            if np.argmax(q_values) == actions[i]:
                correct += 1
        accuracies_fixed.append(correct / n_states)

        if episode % 10 == 0:
            print(f"Episode {episode:3d}: Loss = {losses_fixed[-1]:.4f}, Accuracy = {accuracies_fixed[-1]:.2f}")

    # ============================================================================
    # RESULTS COMPARISON
    # ============================================================================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\nMethod 1 (Random Gradients):")
    print(f"  Final Loss:     {losses_buggy[-1]:.4f}")
    print(f"  Final Accuracy: {accuracies_buggy[-1]:.2f}")
    print(f"  Avg Loss:       {np.mean(losses_buggy):.4f}")
    print(f"  Avg Accuracy:   {np.mean(accuracies_buggy):.2f}")

    print(f"\nMethod 2 (Backpropagation):")
    print(f"  Final Loss:     {losses_fixed[-1]:.4f}")
    print(f"  Final Accuracy: {accuracies_fixed[-1]:.2f}")
    print(f"  Avg Loss:       {np.mean(losses_fixed):.4f}")
    print(f"  Avg Accuracy:   {np.mean(accuracies_fixed):.2f}")

    print(f"\nImprovement:")
    loss_improvement = (np.mean(losses_buggy) - np.mean(losses_fixed)) / np.mean(losses_buggy) * 100
    if np.mean(accuracies_buggy) > 0:
        acc_improvement = (np.mean(accuracies_fixed) - np.mean(accuracies_buggy)) / np.mean(accuracies_buggy) * 100
        print(f"  Loss Reduction:     {loss_improvement:+.1f}%")
        print(f"  Accuracy Increase:  {acc_improvement:+.1f}%")
    else:
        print(f"  Loss Reduction:     {loss_improvement:+.1f}%")
        print(f"  Accuracy Increase:  +inf% (from 0.00 to {np.mean(accuracies_fixed):.2f})")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss over time
    axes[0].plot(losses_buggy, label='Random Gradients (Buggy)', alpha=0.7, color='red')
    axes[0].plot(losses_fixed, label='Backpropagation (Fixed)', alpha=0.7, color='green')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Loss Comparison: Lower is Better')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy over time
    axes[1].plot(accuracies_buggy, label='Random Gradients (Buggy)', alpha=0.7, color='red')
    axes[1].plot(accuracies_fixed, label='Backpropagation (Fixed)', alpha=0.7, color='green')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison: Higher is Better')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'learning_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if np.mean(accuracies_fixed) > np.mean(accuracies_buggy):
        print("\n[SUCCESS] Proper backpropagation significantly outperforms random gradients")
        print("  The fix is CRITICAL and enables actual learning.")
    else:
        print("\n[WARNING] Unexpected result - may need more training episodes")

    print("\nKey Insight:")
    print("  Random gradients = Random weight updates = No learning")
    print("  Proper backprop = Computed gradients = Actual learning")

    return accuracies_fixed[-1] > accuracies_buggy[-1]


if __name__ == "__main__":
    success = compare_learning_methods()
    sys.exit(0 if success else 1)
