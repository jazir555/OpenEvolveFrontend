# Adaptive Learner - User Guide

Complete guide for using the Advanced Adaptive Learner component.

## Overview

The Advanced Adaptive Learner uses deep reinforcement learning to continuously learn from gauntlet execution results, optimizing testing strategies and difficulty levels automatically.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Learning Algorithms](#learning-algorithms)
3. [Training](#training)
4. [Adaptation API](#adaptation-api)
5. [Best Practices](#best-practices)

---

## Quick Start

### Basic Learning

```python
from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
    AdvancedAdaptiveLearner,
    LearningAlgorithm,
    create_learner
)

# Create learner
learner = create_learner(
    algorithm="dqn",
    state_size=8,
    action_size=10
)

# Learn from single execution
learner.learn_from_execution(
    state=state_vector,
    action=action_taken,
    reward=reward_received,
    next_state=new_state_vector,
    done=execution_complete
)

# Get adaptive strategy
strategy = learner.get_adaptive_strategy(current_state)
print(f"Recommended strategy: {strategy}")
```

### Training from History

```python
# Load historical execution data
history = load_execution_history()

# Train learner
metrics = learner.train_from_history(
    history=history,
    episodes=100
)

# View learning progress
for metric in metrics[-10:]:  # Last 10 episodes
    print(f"Episode {metric.episode}: reward={metric.total_reward:.2f}, loss={metric.loss:.4f}")
```

---

## Learning Algorithms

### DQN (Deep Q-Network)

Uses neural networks to approximate Q-values for complex problems.

**Best for:**
- Large state spaces
- Complex environments
- When you have lots of data

```python
learner = AdvancedAdaptiveLearner(
    algorithm=LearningAlgorithm.DQN,
    state_size=8,
    action_size=10,
    learning_rate=0.001,
    gamma=0.95,
    max_iterations=100
)
```

### PPO (Proximal Policy Optimization)

Policy gradient method with clipped objective for stable training.

**Best for:**
- Continuous action spaces
- When sample efficiency matters
- Stable training requirements

```python
learner = AdvancedAdaptiveLearner(
    algorithm=LearningAlgorithm.PPO,
    state_size=8,
    action_size=10
)
```

### A3C (Asynchronous Actor-Critic)

Multiple agents exploring environment in parallel.

**Best for:**
- Distributed training
- Complex environments
- When you need fast learning

```python
learner = AdvancedAdaptiveLearner(
    algorithm=LearningAlgorithm.A3C,
    state_size=8,
    action_size=10
)
```

### SARSA (State-Action-Reward-State-Action)

On-policy TD control method.

**Best for:**
- Simpler problems
- When on-policy learning is required
- More conservative updates

```python
learner = AdvancedAdaptiveLearner(
    algorithm=LearningAlgorithm.SARSA,
    state_size=8,
    action_size=10
)
```

---

## Training

### From Historical Data

```python
# Prepare historical data
historical_data = [
    {
        "state": np.array([0.5, 0.6, 0.7, 0.5, 0.5, 3.0, 0.5, 0.5]),
        "action": 2,
        "reward": 0.8,
        "next_state": np.array([0.5, 0.6, 0.7, 0.5, 0.5, 3.0, 0.5, 0.5]),
        "done": False
    },
    # ... more records
]

# Train
metrics = learner.train_from_history(
    history=historical_data,
    episodes=100
)
```

### Online Learning

```python
# During gauntlet execution
for execution in executions:
    state = extract_state(execution)
    action = select_action(execution)

    # Execute action
    result = execute_action(execution, action)

    # Learn from result
    learner.learn_from_execution(
        state=state,
        action=action,
        reward=result.reward,
        next_state=result.next_state,
        done=result.completed
    )
```

### Continuous Learning

```python
# Enable continuous improvement
while True:
    # Get current state
    state = get_current_state()

    # Get action from learner
    action = learner.act(state)

    # Execute and learn
    result = execute(action)
    learner.learn_from_execution(
        state, action, result.reward,
        result.next_state, result.done
    )

    # Periodically save model
    if training_step % 1000 == 0:
        learner.save_model("adaptive_learner.pkl")
```

---

## Adaptation API

### Get Adaptive Strategy

```python
def get_adaptive_strategy(self, state: np.ndarray) -> Dict[str, Any]:
```

Get recommended strategy based on current state and learned policy.

**Parameters:**
- `state`: Current state vector (8-dimensional)

**Returns:**
- Dictionary with strategy parameters including thresholds and difficulty

**Example:**

```python
state = np.array([5.0, 6.0, 7.0, 5.0, 5.0, 3.0, 5.0, 5.0], dtype=np.float32)
strategy = learner.get_adaptive_strategy(state)

print(f"Round 1 Threshold: {strategy['round1_threshold']}")
print(f"Difficulty: {strategy['difficulty']}")
```

### Generate Test Case

```python
def generate_test_case(
    self,
    difficulty: str = "medium",
    domain: str = "general"
) -> Dict[str, Any]:
```

Generate a test case based on learned strategy.

**Example:**

```python
test_case = learner.generate_test_case(
    difficulty="hard",
    domain="math"
)

print(f"Domain: {test_case['domain']}")
print(f"Difficulty: {test_case['difficulty']}")
print(f"Config: {test_case['config']}")
print(f"Expected: {test_case['expected_outcome']}")
```

### Save and Load Models

```python
# Save trained model
learner.save_model("models/adaptive_learner_dqn.json")

# Load model
learner.load_model("models/adaptive_learner_dqn.json")
```

---

## Best Practices

### 1. State Representation

```python
def extract_state(execution_result):
    """Extract meaningful state from execution"""
    return np.array([
        execution_result["round1_threshold"] * 10,
        execution_result["round2_threshold"] * 10,
        execution_result["round3_threshold"] * 10,
        execution_result["solution_complexity"] * 10,
        execution_result["domain_difficulty"] * 10,
        execution_result["execution_time"] / 10,
        execution_result["score"] * 10,
        execution_result["passed"] * 10
    ], dtype=np.float32)
```

### 2. Reward Design

```python
def calculate_reward(execution_result):
    """Calculate reward for learning"""
    score = execution_result["score"]
    passed = execution_result["passed"]
    time = execution_result["execution_time"]

    # Base reward from score
    reward = score * 2 - 1  # Scale to [-1, 1]

    # Bonus for passing
    if passed:
        reward += 0.5

    # Penalty for excessive time
    if time > 60:
        reward -= 0.3
    elif time < 15:
        reward += 0.2

    return reward
```

### 3. Experience Collection

```python
# Collect diverse experiences
for domain in ["code", "math", "finance"]:
    for difficulty in ["easy", "medium", "hard"]:
        execution = execute_gauntlet(
            solution=test_solution,
            domain=domain,
            difficulty=difficulty
        )

        # Extract state and reward
        state = extract_state(execution)
        reward = calculate_reward(execution)

        # Store experience
        learner.remember(state, action, reward, next_state, done)
```

### 4. Monitoring Training

```python
# Track learning metrics
metrics = learner.train_from_history(history, episodes=100)

# Analyze convergence
import matplotlib.pyplot as plt

rewards = [m.total_reward for m in metrics]
losses = [m.loss for m in metrics if m.loss > 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(rewards)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Learning Progress')

ax2.plot(losses)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')

plt.show()
```

### 5. Hyperparameter Tuning

```python
# Experiment with different hyperparameters
learning_rates = [0.001, 0.01, 0.1]
gammas = [0.9, 0.95, 0.99]
epsilons = [0.1, 0.2, 0.3]

best_score = -float('inf')
best_config = None

for lr in learning_rates:
    for gamma in gammas:
        for epsilon in epsilons:
            learner = AdvancedAdaptiveLearner(
                learning_rate=lr,
                gamma=gamma,
                epsilon=epsilon
            )

            result = learner.train_from_history(
                history=validation_history,
                episodes=50
            )

            final_reward = result[-1].total_reward
            if final_reward > best_score:
                best_score = final_reward
                best_config = (lr, gamma, epsilon)

print(f"Best config: lr={best_config[0]}, gamma={best_config[1]}, epsilon={best_config[2]}")
```

---

## Advanced Usage

### Custom Neural Network Architecture

```python
class CustomLearner(AdvancedAdaptiveLearner):
    def _initialize_network(self):
        """Define custom network architecture"""
        # Larger hidden layer
        hidden_size = 128

        weights = {
            "W1": np.random.randn(self.state_size, hidden_size) * 0.01,
            "b1": np.zeros(hidden_size),
            "W2": np.random.randn(hidden_size, hidden_size) * 0.01,
            "b2": np.zeros(hidden_size),
            "W3": np.random.randn(hidden_size, self.action_size) * 0.01,
            "b3": np.zeros(self.action_size)
        }
        return weights
```

### Custom Reward Function

```python
def custom_reward_calculation(execution_result):
    """Custom reward based on business logic"""

    # Start with base score
    reward = execution_result["score"]

    # Add business-specific bonuses
    if execution_result["domain"] == "finance":
        # Financial domains value precision
        if execution_result["confidence"] > 0.9:
            reward += 0.2

    elif execution_result["domain"] == "healthcare":
        # Healthcare values safety
        if execution_result["safety_checks_passed"]:
            reward += 0.3

    return reward
```

### Multi-Objective Learning

```python
# Track multiple objectives
learner = AdvancedAdaptiveLearner()

# Weight different objectives
for execution in executions:
    accuracy_reward = calculate_accuracy_reward(execution)
    speed_reward = calculate_speed_reward(execution)
    cost_reward = calculate_cost_reward(execution)

    # Weighted combination
    total_reward = (
        0.5 * accuracy_reward +
        0.3 * speed_reward +
        0.2 * cost_reward
    )

    learner.learn_from_execution(
        state, action, total_reward,
        next_state, done
    )
```

---

## Troubleshooting

### Not Learning

**Issue**: The agent doesn't improve over episodes.

**Solutions**:
1. Check reward function - is it informative?
2. Increase `learning_rate` for faster learning
3. Increase `epsilon` for more exploration
4. Verify state representation captures important features
5. Check if action space is adequate

### Unstable Learning

**Issue**: Reward fluctuates wildly.

**Solutions**:
1. Decrease `learning_rate`
2. Increase `batch_size` for more stable gradients
3. Use target network updates
4. Increase `replay_buffer_size` for more diverse samples

### Overfitting

**Issue**: Works well on training data but poorly on new problems.

**Solutions**:
1. Increase exploration (`epsilon`)
2. Use regularization in network
3. Increase training data diversity
4. Early stopping based on validation performance

---

## Support

For issues or questions:
- GitHub: https://github.com/openevolve/adaptive-learner/issues
- Documentation: https://docs.openevolve.org/adaptive-learner
