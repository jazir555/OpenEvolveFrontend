#!/bin/bash

###############################################################################
# ML Optimizer Probe
#
# Validates ML-based gauntlet optimizer functionality per CLAUDE.md Law 2.
#
# Tests:
# 1. Module import verification
# 2. Optimizer instantiation
# 3. Basic optimization functionality
# 4. API response validation
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

log_info "ML Optimizer Probe"
log_info "===================="
log_info "Project root: $PROJ_ROOT"
echo ""

###############################################################################
# Test 1: Module Import Verification
###############################################################################
log_info "Test 1: Verifying ML optimizer module import..."

TEST_PYTHON_TEST1=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
        MLBasedGauntletOptimizer,
        OptimizationStrategy,
        Objective,
        GauntletState,
        OptimizationResult,
        create_optimizer
    )
    print("SUCCESS: All ML optimizer classes imported successfully")
    exit(0)
except ImportError as e:
    print(f"FAIL: Cannot import ML optimizer: {e}")
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
    log_error "Failed to import ML optimizer module"
    exit 1
fi

###############################################################################
# Test 2: Optimizer Instantiation
###############################################################################
log_info "Test 2: Testing optimizer instantiation..."

TEST_PYTHON_TEST2=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
        MLBasedGauntletOptimizer,
        OptimizationStrategy,
        Objective
    )

    # Test default instantiation
    optimizer1 = MLBasedGauntletOptimizer()
    assert optimizer1.strategy == OptimizationStrategy.Q_LEARNING
    assert optimizer1.learning_rate == 0.1
    assert optimizer1.max_iterations == 100

    # Test custom instantiation
    optimizer2 = MLBasedGauntletOptimizer(
        strategy=OptimizationStrategy.DQN,
        learning_rate=0.01,
        max_iterations=200
    )
    assert optimizer2.strategy == OptimizationStrategy.DQN
    assert optimizer2.learning_rate == 0.01
    assert optimizer2.max_iterations == 200

    print("SUCCESS: Optimizer instantiation working correctly")
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
    test_pass "Optimizer instantiation"
else
    test_fail "Optimizer instantiation"
fi

###############################################################################
# Test 3: Gauntlet State Operations
###############################################################################
log_info "Test 3: Testing GauntletState operations..."

TEST_PYTHON_TEST3=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import GauntletState

    # Test state creation
    state = GauntletState(
        round1_threshold=0.5,
        round2_threshold=0.6,
        round3_threshold=0.7,
        round1_weight=0.2,
        round2_weight=0.3,
        round3_weight=0.5,
        max_evaluations_round1=50,
        enable_parallel=False
    )

    # Test to_dict conversion
    state_dict = state.to_dict()
    assert state_dict['round1_threshold'] == 0.5
    assert state_dict['round2_threshold'] == 0.6
    assert state_dict['enable_parallel'] == False

    # Test from_dict restoration
    restored_state = GauntletState.from_dict(state_dict)
    assert restored_state.round1_threshold == state.round1_threshold
    assert restored_state.enable_parallel == state.enable_parallel

    # Test to_tuple conversion
    state_tuple = state.to_tuple()
    assert len(state_tuple) == 7
    assert isinstance(state_tuple[0], int)  # Should be integer

    print("SUCCESS: GauntletState operations working correctly")
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
    test_pass "GauntletState operations"
else
    test_fail "GauntletState operations"
fi

###############################################################################
# Test 4: Basic Optimization Functionality
###############################################################################
log_info "Test 4: Testing basic optimization functionality..."

TEST_PYTHON_TEST4=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
        MLBasedGauntletOptimizer,
        Objective
    )

    # Create optimizer
    optimizer = MLBasedGauntletOptimizer(max_iterations=10)

    # Test optimization with simple parameters
    result = optimizer.optimize(
        domain="code",
        objective=Objective.BALANCED,
        historical_data=None
    )

    # Validate result structure
    assert hasattr(result, 'best_state'), "Result missing best_state"
    assert hasattr(result, 'best_score'), "Result missing best_score"
    assert hasattr(result, 'iterations'), "Result missing iterations"
    assert hasattr(result, 'convergence_history'), "Result missing convergence_history"
    assert hasattr(result, 'improvement_percent'), "Result missing improvement_percent"

    # Validate result types
    assert isinstance(result.best_score, float), "best_score should be float"
    assert isinstance(result.iterations, int), "iterations should be int"
    assert isinstance(result.convergence_history, list), "convergence_history should be list"
    assert 0.0 <= result.best_score <= 1.0, "best_score should be between 0 and 1"

    # Validate convergence history
    assert len(result.convergence_history) > 0, "convergence_history should not be empty"

    print(f"SUCCESS: Optimization completed - score: {result.best_score:.3f}, iterations: {result.iterations}")
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
    test_pass "Basic optimization functionality"
else
    test_fail "Basic optimization functionality"
fi

###############################################################################
# Test 5: Optimization Strategy Variants
###############################################################################
log_info "Test 5: Testing different optimization strategies..."

TEST_PYTHON_TEST5=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
        MLBasedGauntletOptimizer,
        OptimizationStrategy,
        Objective
    )

    strategies = [
        OptimizationStrategy.Q_LEARNING,
        OptimizationStrategy.DQN,
        OptimizationStrategy.GENETIC_ALGORITHM,
        OptimizationStrategy.BAYESIAN_OPTIMIZATION
    ]

    for strategy in strategies:
        optimizer = MLBasedGauntletOptimizer(
            strategy=strategy,
            max_iterations=5  # Small number for testing
        )

        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        assert result.best_score >= 0.0, f"{strategy.value} produced invalid score"
        print(f"  {strategy.value}: score={result.best_score:.3f}")

    print("SUCCESS: All optimization strategies working")
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
    test_pass "Optimization strategy variants"
else
    test_fail "Optimization strategy variants"
fi

###############################################################################
# Test 6: API Response Validation (to_dict)
###############################################################################
log_info "Test 6: Testing API response serialization..."

TEST_PYTHON_TEST6=$(cat <<'EOF'
import sys
import json
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
        MLBasedGauntletOptimizer,
        Objective
    )

    optimizer = MLBasedGauntletOptimizer(max_iterations=10)
    result = optimizer.optimize(
        domain="code",
        objective=Objective.MAXIMIZE_ACCURACY
    )

    # Test to_dict conversion
    result_dict = result.to_dict()

    # Validate dictionary structure
    assert 'best_state' in result_dict, "Missing best_state in dict"
    assert 'best_score' in result_dict, "Missing best_score in dict"
    assert 'iterations' in result_dict, "Missing iterations in dict"
    assert 'convergence_history' in result_dict, "Missing convergence_history in dict"
    assert 'improvement_percent' in result_dict, "Missing improvement_percent in dict"
    assert 'recommendation' in result_dict, "Missing recommendation in dict"
    assert 'timestamp' in result_dict, "Missing timestamp in dict"

    # Validate nested best_state
    best_state = result_dict['best_state']
    assert 'round1_threshold' in best_state, "Missing round1_threshold in best_state"
    assert 'round2_threshold' in best_state, "Missing round2_threshold in best_state"
    assert 'round3_threshold' in best_state, "Missing round3_threshold in best_state"

    # Test JSON serialization
    json_str = json.dumps(result_dict)
    assert len(json_str) > 0, "Failed to serialize to JSON"

    print(f"SUCCESS: API response serialization working - JSON length: {len(json_str)} bytes")
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
    test_pass "API response serialization"
else
    test_fail "API response serialization"
fi

###############################################################################
# Test 7: Factory Function
###############################################################################
log_info "Test 7: Testing factory function..."

TEST_PYTHON_TEST7=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import create_optimizer
    from glue.adapters.gauntlet_adapter.src.ml_optimizer import OptimizationStrategy

    # Test factory with different strategies
    optimizer1 = create_optimizer(strategy="q_learning")
    assert optimizer1.strategy == OptimizationStrategy.Q_LEARNING

    optimizer2 = create_optimizer(strategy="dqn", learning_rate=0.05)
    assert optimizer2.strategy == OptimizationStrategy.DQN
    assert optimizer2.learning_rate == 0.05

    optimizer3 = create_optimizer(strategy="genetic", max_iterations=50)
    assert optimizer3.strategy == OptimizationStrategy.GENETIC_ALGORITHM
    assert optimizer3.max_iterations == 50

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

if python3 -c "$TEST_PYTHON_TEST7" > /dev/null 2>&1; then
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
    log_info "✓ All ML optimizer tests passed!"
    exit 0
fi
