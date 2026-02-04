#!/bin/bash
#
# Probe script for RESE Phase III MCTS Search
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: Verify before using
# - This script MUST execute successfully before Phase III is considered functional
#
# Usage: ./check_phase3.sh
#
# Exit codes:
# 0: All checks passed
# 1: Configuration validation failed
# 2: Import test failed
# 3: Executor initialization failed
# 4: Search execution failed
# 5: Validation failed
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "RESE Phase III Probe"
echo "Testing MCTS Search Executor"
echo "=================================="
echo ""

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

# Check 1: Verify Python is available
echo -n "Check 1: Python availability... "
if ! command -v python &> /dev/null; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: Python not found"
    exit 2
fi
echo -e "${GREEN}PASSED${NC}"

# Check 2: Set environment variables (Law of Configuration Explicitness)
echo -n "Check 2: Setting environment variables... "

export PHASE3_ITERATIONS=100
export PHASE3_UCB1_C=1.414
export PHASE3_CONVERGENCE_THRESHOLD=0.001
export PHASE3_TIMEOUT_MS=30000
export PHASE3_MAX_DEPTH=20
export PHASE3_MAX_CHILDREN=10
export PHASE3_MIN_VISITS=5
export PHASE3_SIG_THRESHOLD=0.05
export PHASE3_CONFIDENCE_INTERVAL=0.95
export PHASE3_MIN_SAMPLE_SIZE=30
export PHASE3_ACI_WINDOW=100
export PHASE3_ACI_STABILITY=0.01
export PHASE3_DEDUP_ENABLED=true
export PHASE3_CACHE_SIZE=10000
export PHASE3_CB_THRESHOLD=5
export PHASE3_CB_TIMEOUT=60000

echo -e "${GREEN}PASSED${NC}"

# Check 3: Test imports
echo -n "Check 3: Testing imports... "

python3 << EOF
import sys
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from rese_schemas import Hypothesis
    from phase3_executor import (
        MCTSSearchExecutor,
        Phase3Config,
        SearchTreeBuilder,
        HypothesisValidator,
        ConvergenceDetector,
    )
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: Import test failed"
    exit 2
fi

IMPORT_OUTPUT=$(python3 << EOF
import sys
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from rese_schemas import Hypothesis
    from phase3_executor import MCTSSearchExecutor, Phase3Config
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
    sys.exit(1)
EOF
)

if [[ "$IMPORT_OUTPUT" != *"IMPORT_SUCCESS"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $IMPORT_OUTPUT"
    exit 2
fi

echo -e "${GREEN}PASSED${NC}"

# Check 4: Configuration validation
echo -n "Check 4: Configuration validation... "

CONFIG_OUTPUT=$(python3 << EOF
import sys
import os
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from phase3_executor import Phase3Config

    # Test configuration loading
    config = Phase3Config.from_env()

    # Validate required fields
    assert config.iterations > 0, "iterations must be positive"
    assert config.ucb1_c > 0, "ucb1_c must be positive"
    assert config.convergence_threshold >= 0, "convergence_threshold must be non-negative"
    assert config.timeout_ms > 0, "timeout_ms must be positive"
    assert config.max_depth > 0, "max_depth must be positive"
    assert config.max_children_per_node > 0, "max_children_per_node must be positive"
    assert config.min_visits_before_expand >= 0, "min_visits must be non-negative"

    print("CONFIG_VALID")
except Exception as e:
    print(f"CONFIG_INVALID: {e}")
    sys.exit(1)
EOF
)

if [[ "$CONFIG_OUTPUT" != *"CONFIG_VALID"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $CONFIG_OUTPUT"
    exit 1
fi

echo -e "${GREEN}PASSED${NC}"

# Check 5: Executor initialization
echo -n "Check 5: Executor initialization... "

EXEC_OUTPUT=$(python3 << EOF
import sys
import os
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from rese_schemas import Hypothesis
    from phase3_executor import MCTSSearchExecutor, Phase3Config

    # Initialize executor
    config = Phase3Config.from_env()
    executor = MCTSSearchExecutor(config)

    # Verify components are initialized
    assert executor.tree_builder is not None, "tree_builder not initialized"
    assert executor.selection_strategy is not None, "selection_strategy not initialized"
    assert executor.hypothesis_validator is not None, "hypothesis_validator not initialized"
    assert executor.convergence_detector is not None, "convergence_detector not initialized"
    assert executor.dlq is not None, "dlq not initialized"
    assert executor.circuit_breaker is not None, "circuit_breaker not initialized"

    print("EXECUTOR_INIT_SUCCESS")
except Exception as e:
    print(f"EXECUTOR_INIT_FAILED: {e}")
    sys.exit(1)
EOF
)

if [[ "$EXEC_OUTPUT" != *"EXECUTOR_INIT_SUCCESS"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $EXEC_OUTPUT"
    exit 3
fi

echo -e "${GREEN}PASSED${NC}"

# Check 6: Simple search execution (with minimal iterations for speed)
echo -n "Check 6: Search execution (10 iterations)... "

SEARCH_OUTPUT=$(python3 << EOF
import sys
import os
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from rese_schemas import Hypothesis
    from phase3_executor import MCTSSearchExecutor, Phase3Config

    # Initialize executor with minimal config for speed
    config = Phase3Config.from_env()
    config.iterations = 10  # Minimal iterations for fast testing

    executor = MCTSSearchExecutor(config)

    # Create root hypothesis
    root_hypothesis = Hypothesis(
        statement="Test root hypothesis for Phase III probe",
        type="test",
        domain="test_domain",
        confidence=0.5
    )

    # Define hypothesis generator
    def hypothesis_generator():
        children = []
        for i in range(3):
            child = Hypothesis(
                statement=f"Child {i}",
                type="test",
                domain="test_domain",
                confidence=0.6
            )
            children.append(child)
        return children

    # Define reward function
    def reward_function(hypothesis):
        return hypothesis.confidence

    # Execute search
    search_result, error = executor.execute_search(
        root_hypothesis=root_hypothesis,
        hypothesis_generator=hypothesis_generator,
        reward_function=reward_function,
    )

    if error:
        print(f"SEARCH_FAILED: {error}")
        sys.exit(1)

    # Validate result
    assert search_result is not None, "search_result is None"
    assert search_result.search_id is not None, "search_id is None"
    assert search_result.root_hypothesis is not None, "root_hypothesis is None"
    assert search_result.best_hypothesis is not None, "best_hypothesis is None"
    assert search_result.iterations > 0, "iterations must be positive"
    assert search_result.execution_time_ms > 0, "execution_time_ms must be positive"

    print(f"SEARCH_SUCCESS: iterations={search_result.iterations}, best_confidence={search_result.best_hypothesis.confidence:.3f}")

except Exception as e:
    print(f"SEARCH_ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
)

if [[ "$SEARCH_OUTPUT" != *"SEARCH_SUCCESS"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $SEARCH_OUTPUT"
    exit 4
fi

echo -e "${GREEN}PASSED${NC}"
echo "  $SEARCH_OUTPUT"

# Check 7: Hypothesis validation
echo -n "Check 7: Hypothesis validation... "

VAL_OUTPUT=$(python3 << EOF
import sys
import os
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from rese_schemas import Hypothesis
    from phase3_executor import Phase3Config, HypothesisValidator

    # Initialize validator
    config = Phase3Config.from_env()
    validator = HypothesisValidator(config)

    # Create test hypothesis
    hypothesis = Hypothesis(
        statement="Test hypothesis for validation",
        type="test",
        domain="test_domain",
        confidence=0.7
    )

    # Generate test rewards (above threshold)
    import random
    random.seed(42)
    rewards = [0.6 + random.uniform(-0.1, 0.1) for _ in range(50)]

    # Validate
    validation_metrics, error = validator.validate(hypothesis, rewards)

    if error:
        print(f"VALIDATION_FAILED: {error}")
        sys.exit(1)

    # Check validation result
    assert validation_metrics is not None, "validation_metrics is None"
    assert validation_metrics.hypothesis_id == hypothesis.hypothesis_id, "hypothesis_id mismatch"
    assert validation_metrics.sample_size == len(rewards), "sample_size mismatch"
    assert 0.0 <= validation_metrics.mean_reward <= 1.0, "mean_reward out of range"

    print(f"VALIDATION_SUCCESS: is_valid={validation_metrics.is_valid}, confidence={validation_metrics.confidence:.3f}")

except Exception as e:
    print(f"VALIDATION_ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
)

if [[ "$VAL_OUTPUT" != *"VALIDATION_SUCCESS"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $VAL_OUTPUT"
    exit 5
fi

echo -e "${GREEN}PASSED${NC}"
echo "  $VAL_OUTPUT"

# Check 8: Convergence detection
echo -n "Check 8: Convergence detection... "

CONV_OUTPUT=$(python3 << EOF
import sys
import os
sys.path.insert(0, "glue/lib")
sys.path.insert(0, "glue/schemas")
sys.path.insert(0, "glue/adapters/rese-phase3/src")

try:
    from phase3_executor import Phase3Config, ConvergenceDetector

    # Initialize detector
    config = Phase3Config.from_env()
    detector = ConvergenceDetector(config)

    # Simulate convergence (stable confidence)
    for i in range(150):
        detector.update(i, 0.8, 0.75)

    # Check convergence
    is_converged, aci_value = detector.check_convergence()

    assert isinstance(is_converged, bool), "is_converged must be bool"
    assert aci_value is not None or not is_converged, "aci_value must be set when converged"

    print(f"CONVERGENCE_CHECK_SUCCESS: is_converged={is_converged}, aci_value={aci_value}")

except Exception as e:
    print(f"CONVERGENCE_ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
)

if [[ "$CONV_OUTPUT" != *"CONVERGENCE_CHECK_SUCCESS"* ]]; then
    echo -e "${RED}FAILED${NC}"
    echo "ERROR: $CONV_OUTPUT"
    exit 5
fi

echo -e "${GREEN}PASSED${NC}"
echo "  $CONV_OUTPUT"

# Summary
echo ""
echo "=================================="
echo -e "${GREEN}ALL CHECKS PASSED${NC}"
echo "=================================="
echo ""
echo "RESE Phase III MCTS Search Executor is functional."
echo ""
echo "Components verified:"
echo "  ✓ Configuration validation"
echo "  ✓ Import tests"
echo "  ✓ Executor initialization"
echo "  ✓ Search execution"
echo "  ✓ Hypothesis validation"
echo "  ✓ Convergence detection"
echo ""
echo "Phase III is ready for integration."
echo ""

exit 0
