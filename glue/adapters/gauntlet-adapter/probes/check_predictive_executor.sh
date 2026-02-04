#!/bin/bash

###############################################################################
# Predictive Executor Probe
#
# Validates predictive gauntlet executor functionality per CLAUDE.md Law 2.
#
# Tests:
# 1. Module import verification
# 2. Executor instantiation
# 3. Success prediction API
# 4. Execution planning
# 5. Prediction validation
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

log_info "Predictive Executor Probe"
log_info "=========================="
log_info "Project root: $PROJ_ROOT"
echo ""

###############################################################################
# Test 1: Module Import Verification
###############################################################################
log_info "Test 1: Verifying predictive executor module import..."

TEST_PYTHON_TEST1=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor,
        ExecutionDecision,
        PredictionResult,
        ExecutionPlan,
        ExecutionResult
    )
    print("SUCCESS: All predictive executor classes imported successfully")
    exit(0)
except ImportError as e:
    print(f"FAIL: Cannot import predictive executor: {e}")
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
    log_error "Failed to import predictive executor module"
    exit 1
fi

###############################################################################
# Test 2: Executor Instantiation
###############################################################################
log_info "Test 2: Testing executor instantiation..."

TEST_PYTHON_TEST2=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    # Test default instantiation
    executor1 = PredictiveGauntletExecutor()
    assert executor1.success_threshold == 0.3
    assert executor1.confidence_threshold == 0.6
    assert executor1.cost_threshold == 100.0

    # Test custom instantiation
    executor2 = PredictiveGauntletExecutor(
        success_threshold=0.5,
        confidence_threshold=0.7,
        cost_threshold=50.0
    )
    assert executor2.success_threshold == 0.5
    assert executor2.confidence_threshold == 0.7
    assert executor2.cost_threshold == 50.0

    print("SUCCESS: Executor instantiation working correctly")
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
    test_pass "Executor instantiation"
else
    test_fail "Executor instantiation"
fi

###############################################################################
# Test 3: Success Prediction API
###############################################################################
log_info "Test 3: Testing success prediction API..."

TEST_PYTHON_TEST3=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    executor = PredictiveGauntletExecutor()

    # Test basic prediction
    prediction = executor.predict_success(
        solution="def solve(): return 42",
        problem="Find the answer to life",
        domain="general"
    )

    # Validate prediction structure
    assert hasattr(prediction, 'success_probability'), "Missing success_probability"
    assert hasattr(prediction, 'confidence'), "Missing confidence"
    assert hasattr(prediction, 'risk_factors'), "Missing risk_factors"
    assert hasattr(prediction, 'recommended_difficulty'), "Missing recommended_difficulty"
    assert hasattr(prediction, 'estimated_time'), "Missing estimated_time"
    assert hasattr(prediction, 'estimated_cost'), "Missing estimated_cost"

    # Validate data types and ranges
    assert isinstance(prediction.success_probability, float), "success_probability should be float"
    assert isinstance(prediction.confidence, float), "confidence should be float"
    assert isinstance(prediction.risk_factors, list), "risk_factors should be list"
    assert 0.0 <= prediction.success_probability <= 1.0, "success_probability out of range"
    assert 0.0 <= prediction.confidence <= 1.0, "confidence out of range"
    assert prediction.estimated_time > 0, "estimated_time should be positive"
    assert prediction.estimated_cost > 0, "estimated_cost should be positive"

    print(f"SUCCESS: Prediction API working - prob: {prediction.success_probability:.2f}, confidence: {prediction.confidence:.2f}")
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
    test_pass "Success prediction API"
else
    test_fail "Success prediction API"
fi

###############################################################################
# Test 4: Execution Planning
###############################################################################
log_info "Test 4: Testing execution planning..."

TEST_PYTHON_TEST4=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor,
        ExecutionDecision
    )

    executor = PredictiveGauntletExecutor()

    # Test with high probability prediction
    prediction_high = executor.predict_success(
        solution="def solve(): return optimal_solution",
        problem="Optimize portfolio",
        domain="finance"
    )

    plan_high = executor.create_execution_plan(prediction_high)

    # Validate plan structure
    assert hasattr(plan_high, 'decision'), "Missing decision"
    assert hasattr(plan_high, 'adjusted_config'), "Missing adjusted_config"
    assert hasattr(plan_high, 'reasoning'), "Missing reasoning"
    assert hasattr(plan_high, 'expected_outcome'), "Missing expected_outcome"
    assert hasattr(plan_high, 'resource_allocation'), "Missing resource_allocation"

    # Validate decision is valid enum
    assert isinstance(plan_high.decision, ExecutionDecision), "decision should be ExecutionDecision"

    # Test with low probability (simulate by using very short solution)
    prediction_low = executor.predict_success(
        solution="x",
        problem="Solve everything",
        domain="math"
    )

    plan_low = executor.create_execution_plan(prediction_low)

    # Should recommend skip or adjust for low probability
    assert plan_low.decision in [
        ExecutionDecision.SKIP_LOW_PROBABILITY,
        ExecutionDecision.ADJUST_DIFFICULTY
    ], "Low probability should trigger skip or adjust"

    print(f"SUCCESS: Execution planning working - high_prob_decision: {plan_high.decision.value}, low_prob_decision: {plan_low.decision.value}")
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
    test_pass "Execution planning"
else
    test_fail "Execution planning"
fi

###############################################################################
# Test 5: Prediction Result Serialization
###############################################################################
log_info "Test 5: Testing prediction result serialization..."

TEST_PYTHON_TEST5=$(cat <<'EOF'
import sys
import json
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    executor = PredictiveGauntletExecutor()
    prediction = executor.predict_success(
        solution="def solve(): return 42",
        problem="Test problem",
        domain="general"
    )

    # Test to_dict conversion
    prediction_dict = prediction.to_dict()

    # Validate dictionary structure
    assert 'success_probability' in prediction_dict, "Missing success_probability"
    assert 'confidence' in prediction_dict, "Missing confidence"
    assert 'risk_factors' in prediction_dict, "Missing risk_factors"
    assert 'recommended_difficulty' in prediction_dict, "Missing recommended_difficulty"
    assert 'estimated_time' in prediction_dict, "Missing estimated_time"
    assert 'estimated_cost' in prediction_dict, "Missing estimated_cost"

    # Validate types
    assert isinstance(prediction_dict['risk_factors'], list), "risk_factors should be list"

    # Test JSON serialization
    json_str = json.dumps(prediction_dict)
    assert len(json_str) > 0, "Failed to serialize to JSON"

    print(f"SUCCESS: Prediction serialization working - JSON length: {len(json_str)} bytes")
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
    test_pass "Prediction result serialization"
else
    test_fail "Prediction result serialization"
fi

###############################################################################
# Test 6: Execution Plan Serialization
###############################################################################
log_info "Test 6: Testing execution plan serialization..."

TEST_PYTHON_TEST6=$(cat <<'EOF'
import sys
import json
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    executor = PredictiveGauntletExecutor()
    prediction = executor.predict_success(
        solution="def solve(): return optimal",
        problem="Optimize",
        domain="algorithm"
    )

    plan = executor.create_execution_plan(prediction)

    # Test to_dict conversion
    plan_dict = plan.to_dict()

    # Validate dictionary structure
    assert 'decision' in plan_dict, "Missing decision"
    assert 'adjusted_config' in plan_dict, "Missing adjusted_config"
    assert 'reasoning' in plan_dict, "Missing reasoning"
    assert 'expected_outcome' in plan_dict, "Missing expected_outcome"
    assert 'resource_allocation' in plan_dict, "Missing resource_allocation"

    # Validate decision value
    assert isinstance(plan_dict['decision'], str), "decision should be string"

    # Test JSON serialization
    json_str = json.dumps(plan_dict)
    assert len(json_str) > 0, "Failed to serialize to JSON"

    print(f"SUCCESS: Execution plan serialization working - decision: {plan_dict['decision']}")
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
    test_pass "Execution plan serialization"
else
    test_fail "Execution plan serialization"
fi

###############################################################################
# Test 7: Full Execution Flow
###############################################################################
log_info "Test 7: Testing full execution with prediction..."

TEST_PYTHON_TEST7=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    executor = PredictiveGauntletExecutor()

    # Test full execution flow
    result = executor.execute_with_prediction(
        solution="def solve(): return 42",
        problem="Find answer",
        domain="general"
    )

    # Validate result structure
    assert hasattr(result, 'prediction'), "Missing prediction"
    assert hasattr(result, 'actual_outcome'), "Missing actual_outcome"
    assert hasattr(result, 'prediction_accuracy'), "Missing prediction_accuracy"
    assert hasattr(result, 'execution_time'), "Missing execution_time"
    assert hasattr(result, 'cost_savings'), "Missing cost_savings"
    assert hasattr(result, 'learning_data'), "Missing learning_data"

    # Validate types
    assert isinstance(result.prediction_accuracy, float), "prediction_accuracy should be float"
    assert isinstance(result.execution_time, float), "execution_time should be float"
    assert 0.0 <= result.prediction_accuracy <= 1.0, "prediction_accuracy out of range"
    assert result.execution_time >= 0, "execution_time should be non-negative"

    # Validate learning data
    assert 'timestamp' in result.learning_data, "Missing timestamp in learning_data"
    assert 'domain' in result.learning_data, "Missing domain in learning_data"

    print(f"SUCCESS: Full execution flow working - accuracy: {result.prediction_accuracy:.2f}, time: {result.execution_time:.2f}s")
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
    test_pass "Full execution flow"
else
    test_fail "Full execution flow"
fi

###############################################################################
# Test 8: Prediction Accuracy Statistics
###############################################################################
log_info "Test 8: Testing prediction accuracy tracking..."

TEST_PYTHON_TEST8=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
        PredictiveGauntletExecutor
    )

    executor = PredictiveGauntletExecutor()

    # Run a few predictions to build history
    for i in range(3):
        executor.execute_with_prediction(
            solution=f"def solve_{i}(): return {i}",
            problem=f"Problem {i}",
            domain="general"
        )

    # Get statistics
    stats = executor.get_prediction_accuracy_stats()

    # Validate stats structure
    assert 'mean_accuracy' in stats, "Missing mean_accuracy"
    assert 'std_accuracy' in stats, "Missing std_accuracy"
    assert 'min_accuracy' in stats, "Missing min_accuracy"
    assert 'max_accuracy' in stats, "Missing max_accuracy"
    assert 'total_predictions' in stats, "Missing total_predictions"

    # Validate values
    assert stats['total_predictions'] == 3, "Should have 3 predictions"
    assert isinstance(stats['mean_accuracy'], float), "mean_accuracy should be float"

    print(f"SUCCESS: Prediction accuracy tracking working - total: {stats['total_predictions']}, mean_accuracy: {stats['mean_accuracy']:.3f}")
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

if python3 -c "$TEST_PYTHON_TEST8" > /dev/null 2>&1; then
    test_pass "Prediction accuracy tracking"
else
    test_fail "Prediction accuracy tracking"
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
    log_info "✓ All predictive executor tests passed!"
    exit 0
fi
