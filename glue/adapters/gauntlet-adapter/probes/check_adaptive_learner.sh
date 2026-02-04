#!/bin/bash

###############################################################################
# Adaptive Learner Probe
#
# Validates adaptive learner functionality per CLAUDE.md Law 2.
#
# Tests:
# 1. Module import verification
# 2. Learner instantiation
# 3. Neural network initialization
# 4. Learning from experience
# 5. Model save/load functionality
#
# Returns: 0 on success, non-zero on failure
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

test_pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    log_info "✓ $1"
}

test_fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    log_error "✗ $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Set up Python path
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

log_info "Adaptive Learner Probe"
log_info "======================"
log_info "Project root: $PROJ_ROOT"
echo ""

###############################################################################
# Test 1: Module Import Verification
###############################################################################
log_info "Test 1: Verifying adaptive learner module import..."

TEST_PYTHON_TEST1=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner,
        LearningAlgorithm,
        Experience,
        LearningMetrics,
        AdaptationResult,
        create_learner
    )
    print("SUCCESS: All adaptive learner classes imported successfully")
    exit(0)
except ImportError as e:
    print(f"FAIL: Cannot import adaptive learner: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error during import: {e}")
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST1" > /dev/null 2>&1; then
    test_pass "Module import verification"
else
    test_fail "Module import verification"
    log_error "Failed to import adaptive learner module"
    exit 1
fi

###############################################################################
# Test 2: Learner Instantiation
###############################################################################
log_info "Test 2: Testing learner instantiation..."

TEST_PYTHON_TEST2=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner,
        LearningAlgorithm
    )
    import numpy as np

    # Test default instantiation
    learner1 = AdvancedAdaptiveLearner()
    assert learner1.algorithm == LearningAlgorithm.DQN
    assert learner1.state_size == 8
    assert learner1.action_size == 10
    assert learner1.learning_rate == 0.001

    # Test custom instantiation
    learner2 = AdvancedAdaptiveLearner(
        algorithm=LearningAlgorithm.PPO,
        state_size=16,
        action_size=20,
        learning_rate=0.01
    )
    assert learner2.algorithm == LearningAlgorithm.PPO
    assert learner2.state_size == 16
    assert learner2.action_size == 20
    assert learner2.learning_rate == 0.01

    print("SUCCESS: Learner instantiation working correctly")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST2" > /dev/null 2>&1; then
    test_pass "Learner instantiation"
else
    test_fail "Learner instantiation"
fi

###############################################################################
# Test 3: Neural Network Initialization
###############################################################################
log_info "Test 3: Testing neural network initialization..."

TEST_PYTHON_TEST3=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10)

    # Validate Q-network structure
    assert 'W1' in learner.q_network, "Missing W1 in q_network"
    assert 'b1' in learner.q_network, "Missing b1 in q_network"
    assert 'W2' in learner.q_network, "Missing W2 in q_network"
    assert 'b2' in learner.q_network, "Missing b2 in q_network"

    # Validate shapes
    assert learner.q_network['W1'].shape == (8, 64), f"W1 wrong shape: {learner.q_network['W1'].shape}"
    assert learner.q_network['b1'].shape == (64,), f"b1 wrong shape: {learner.q_network['b1'].shape}"
    assert learner.q_network['W2'].shape == (64, 10), f"W2 wrong shape: {learner.q_network['W2'].shape}"
    assert learner.q_network['b2'].shape == (10,), f"b2 wrong shape: {learner.q_network['b2'].shape}"

    # Validate target network exists
    assert learner.target_network is not None, "target_network should be initialized"

    # Validate target network is a copy
    assert learner.target_network['W1'].shape == learner.q_network['W1'].shape

    print(f"SUCCESS: Neural network initialization working - W1: {learner.q_network['W1'].shape}, W2: {learner.q_network['W2'].shape}")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST3" > /dev/null 2>&1; then
    test_pass "Neural network initialization"
else
    test_fail "Neural network initialization"
fi

###############################################################################
# Test 4: Forward Pass
###############################################################################
log_info "Test 4: Testing neural network forward pass..."

TEST_PYTHON_TEST4=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10)

    # Create test state
    state = np.random.randn(8).astype(np.float32)

    # Forward pass
    q_values, hidden_layer, hidden_pre_relu = learner._forward(learner.q_network, state)

    # Validate output shapes
    assert q_values.shape == (10,), f"Q-values wrong shape: {q_values.shape}"
    assert hidden_layer.shape == (64,), f"Hidden layer wrong shape: {hidden_layer.shape}"
    assert hidden_pre_relu.shape == (64,), f"Hidden pre-relu wrong shape: {hidden_pre_relu.shape}"

    # Validate ReLU activation (hidden layer should have no negative values)
    assert np.all(hidden_layer >= 0), "ReLU activation failed - found negative values in hidden layer"

    # Validate Q-values are finite
    assert np.all(np.isfinite(q_values)), "Q-values contain NaN or Inf"

    print(f"SUCCESS: Forward pass working - Q-values: {q_values[:3]}...")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST4" > /dev/null 2>&1; then
    test_pass "Neural network forward pass"
else
    test_fail "Neural network forward pass"
fi

###############################################################################
# Test 5: Experience Storage and Retrieval
###############################################################################
log_info "Test 5: Testing experience replay buffer..."

TEST_PYTHON_TEST5=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10)

    # Store some experiences
    for i in range(5):
        state = np.random.randn(8).astype(np.float32)
        action = i % 10
        reward = np.random.rand()
        next_state = np.random.randn(8).astype(np.float32)
        done = i == 4

        learner.remember(state, action, reward, next_state, done)

    # Validate experiences were stored
    assert len(learner.memory) == 5, f"Expected 5 experiences, got {len(learner.memory)}"

    # Validate experience structure
    exp = learner.memory[0]
    assert hasattr(exp, 'state'), "Experience missing state"
    assert hasattr(exp, 'action'), "Experience missing action"
    assert hasattr(exp, 'reward'), "Experience missing reward"
    assert hasattr(exp, 'next_state'), "Experience missing next_state"
    assert hasattr(exp, 'done'), "Experience missing done"

    print(f"SUCCESS: Experience replay buffer working - stored {len(learner.memory)} experiences")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST5" > /dev/null 2>&1; then
    test_pass "Experience replay buffer"
else
    test_fail "Experience replay buffer"
fi

###############################################################################
# Test 6: Learning from Experience
###############################################################################
log_info "Test 6: Testing learning from experience..."

TEST_PYTHON_TEST6=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10, batch_size=4)

    # Add enough experiences for a batch
    for i in range(10):
        state = np.random.randn(8).astype(np.float32)
        action = i % 10
        reward = np.random.rand()
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        learner.remember(state, action, reward, next_state, done)

    # Perform learning
    initial_weights = learner.q_network['W1'].copy()
    metrics = learner.replay()

    # Validate metrics
    assert 'loss' in metrics, "Missing loss in metrics"
    assert 'q_value' in metrics, "Missing q_value in metrics"
    assert isinstance(metrics['loss'], float), "loss should be float"
    assert isinstance(metrics['q_value'], float), "q_value should be float"

    # Validate weights changed (learning occurred)
    weights_changed = not np.allclose(initial_weights, learner.q_network['W1'])
    assert weights_changed, "Weights should have changed after learning"

    print(f"SUCCESS: Learning from experience working - loss: {metrics['loss']:.4f}, q_value: {metrics['q_value']:.4f}")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST6" > /dev/null 2>&1; then
    test_pass "Learning from experience"
else
    test_fail "Learning from experience"
fi

###############################################################################
# Test 7: Action Selection (Epsilon-Greedy)
###############################################################################
log_info "Test 7: Testing action selection..."

TEST_PYTHON_TEST7=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve\Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10, epsilon=1.0)
    state = np.random.randn(8).astype(np.float32)

    # Test exploration (epsilon=1.0, should be random)
    actions_explore = [learner.act(state, use_epsilon=True) for _ in range(100)]
    # With epsilon=1.0, should get various actions
    unique_actions = len(set(actions_explore))
    assert unique_actions > 1, "Exploration should produce different actions"

    # Test exploitation (epsilon=0.0, should be deterministic)
    learner.epsilon = 0.0
    actions_exploit = [learner.act(state, use_epsilon=True) for _ in range(10)]
    # With epsilon=0.0, should always get same action
    assert len(set(actions_exploit)) == 1, "Exploitation should produce same action"

    # Test evaluation mode (use_epsilon=False)
    actions_eval = [learner.act(state, use_epsilon=False) for _ in range(10)]
    assert len(set(actions_eval)) == 1, "Evaluation mode should be deterministic"

    print(f"SUCCESS: Action selection working - exploration produced {unique_actions} unique actions")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST7" > /dev/null 2>&1; then
    test_pass "Action selection (epsilon-greedy)"
else
    test_fail "Action selection (epsilon-greedy)"
fi

###############################################################################
# Test 8: Model Save/Load
###############################################################################
log_info "Test 8: Testing model save and load..."

TEST_PYTHON_TEST8=$(cat <<'EOF'
import sys
import os
import tempfile
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    # Create learner and train a bit
    learner1 = AdvancedAdaptiveLearner(state_size=8, action_size=10)

    # Add some experiences and train
    for i in range(10):
        state = np.random.randn(8).astype(np.float32)
        action = i % 10
        reward = np.random.rand()
        next_state = np.random.randn(8).astype(np.float32)
        learner1.remember(state, action, reward, next_state, False)

    initial_epsilon = learner1.epsilon
    initial_step = learner1.training_step

    # Save model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        save_path = f.name

    try:
        learner1.save_model(save_path)

        # Create new learner and load
        learner2 = AdvancedAdaptiveLearner(state_size=8, action_size=10)
        learner2.load_model(save_path)

        # Validate loaded parameters
        assert learner2.epsilon == initial_epsilon, f"Epsilon mismatch: {learner2.epsilon} != {initial_epsilon}"
        assert learner2.training_step == initial_step, f"Training step mismatch"

        # Validate weights were restored
        assert np.allclose(learner1.q_network['W1'], learner2.q_network['W1']), "W1 weights not restored correctly"
        assert np.allclose(learner1.q_network['b1'], learner2.q_network['b1']), "b1 weights not restored correctly"

        print("SUCCESS: Model save/load working correctly")
        exit(0)
    finally:
        # Clean up temp file
        if os.path.exists(save_path):
            os.remove(save_path)

except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST8" > /dev/null 2>&1; then
    test_pass "Model save/load"
else
    test_fail "Model save/load"
fi

###############################################################################
# Test 9: Training from History
###############################################################################
log_info "Test 9: Testing training from historical data..."

TEST_PYTHON_TEST9=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        AdvancedAdaptiveLearner
    )
    import numpy as np

    learner = AdvancedAdaptiveLearner(state_size=8, action_size=10)

    # Create mock historical data
    history = []
    for i in range(20):
        record = {
            'round1_threshold': 0.5,
            'round2_threshold': 0.6,
            'round3_threshold': 0.7,
            'solution_complexity': 0.5,
            'domain_difficulty': 0.5,
            'execution_time': 30.0,
            'score': np.random.rand(),
            'passed': np.random.rand() > 0.5,
            'done': i % 5 == 0  # Episode ends every 5 records
        }
        history.append(record)

    # Train from history
    metrics_list = learner.train_from_history(history, episodes=5)

    # Validate metrics
    assert len(metrics_list) == 5, f"Expected 5 episodes, got {len(metrics_list)}"

    # Validate first episode metrics
    metrics = metrics_list[0]
    assert hasattr(metrics, 'episode'), "Missing episode"
    assert hasattr(metrics, 'total_reward'), "Missing total_reward"
    assert hasattr(metrics, 'episode_length'), "Missing episode_length"

    print(f"SUCCESS: Training from history working - episodes: {len(metrics_list)}, final_reward: {metrics_list[-1].total_reward:.2f}")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST9" > /dev/null 2>&1; then
    test_pass "Training from history"
else
    test_fail "Training from history"
fi

###############################################################################
# Test 10: Factory Function
###############################################################################
log_info "Test 10: Testing factory function..."

TEST_PYTHON_TEST10=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve\Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
        create_learner,
        LearningAlgorithm
    )

    # Test factory with different algorithms
    learner1 = create_learner(algorithm="dqn")
    assert learner1.algorithm == LearningAlgorithm.DQN

    learner2 = create_learner(algorithm="ppo", state_size=16, action_size=20)
    assert learner2.algorithm == LearningAlgorithm.PPO
    assert learner2.state_size == 16
    assert learner2.action_size == 20

    learner3 = create_learner(algorithm="a3c")
    assert learner3.algorithm == LearningAlgorithm.A3C

    learner4 = create_learner(algorithm="sarsa")
    assert learner4.algorithm == LearningAlgorithm.SARSA

    print("SUCCESS: Factory function working correctly")
    exit(0)
except AssertionError as e:
    print(f"FAIL: Assertion failed: {e}")
    exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
)

if python3 -c "$TEST_PYTHON_TEST10" > /dev/null 2>&1; then
    test_pass "Factory function"
else
    test_fail "Factory function"
fi

###############################################################################
# Summary
###############################################################################
echo ""
log_info "Test Summary"
log_info "============"
echo -e "Total tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
    exit 1
else
    echo -e "${GREEN}Failed: ${TESTS_FAILED}${NC}"
    echo ""
    log_info "✓ All adaptive learner tests passed!"
    exit 0
fi
