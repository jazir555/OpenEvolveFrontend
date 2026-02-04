#!/bin/bash

###############################################################################
# Intelligent Orchestrator Probe
#
# Validates intelligent gauntlet orchestrator functionality per CLAUDE.md Law 2.
#
# Tests:
# 1. Module import verification
# 2. Orchestrator instantiation
# 3. Orchestration planning
# 4. Strategy selection
# 5. Resource allocation
# 6. Execution flow
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

log_info "Intelligent Orchestrator Probe"
log_info "==============================="
log_info "Project root: $PROJ_ROOT"
echo ""

###############################################################################
# Test 1: Module Import Verification
###############################################################################
log_info "Test 1: Verifying intelligent orchestrator module import..."

TEST_PYTHON_TEST1=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator,
        OptimizationObjective,
        OrchestrationStrategy,
        OrchestrationPlan,
        OrchestrationResult
    )
    print("SUCCESS: All intelligent orchestrator classes imported successfully")
    exit(0)
except ImportError as e:
    print(f"FAIL: Cannot import intelligent orchestrator: {e}")
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
    log_error "Failed to import intelligent orchestrator module"
    exit 1
fi

###############################################################################
# Test 2: Orchestrator Instantiation
###############################################################################
log_info "Test 2: Testing orchestrator instantiation..."

TEST_PYTHON_TEST2=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator,
        OptimizationObjective
    )

    # Test default instantiation
    orchestrator1 = IntelligentGauntletOrchestrator()
    assert orchestrator1.objective == OptimizationObjective.BALANCED
    assert orchestrator1.max_parallelism == 4
    assert orchestrator1.enable_prediction == True
    assert orchestrator1.enable_optimization == True

    # Test custom instantiation
    orchestrator2 = IntelligentGauntletOrchestrator(
        objective=OptimizationObjective.MAXIMIZE_ACCURACY,
        max_parallelism=8,
        enable_prediction=False,
        enable_optimization=False
    )
    assert orchestrator2.objective == OptimizationObjective.MAXIMIZE_ACCURACY
    assert orchestrator2.max_parallelism == 8
    assert orchestrator2.enable_prediction == False
    assert orchestrator2.enable_optimization == False

    print("SUCCESS: Orchestrator instantiation working correctly")
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
    test_pass "Orchestrator instantiation"
else
    test_fail "Orchestrator instantiation"
fi

###############################################################################
# Test 3: Orchestration Plan Creation
###############################################################################
log_info "Test 3: Testing orchestration plan creation..."

TEST_PYTHON_TEST3=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator,
        OrchestrationStrategy
    )

    orchestrator = IntelligentGauntletOrchestrator()

    # Create plan
    plan = orchestrator.create_orchestration_plan(
        solution="def solve(): return optimal",
        problem="Optimize portfolio",
        domain="finance"
    )

    # Validate plan structure
    assert hasattr(plan, 'strategy'), "Missing strategy"
    assert hasattr(plan, 'execution_order'), "Missing execution_order"
    assert hasattr(plan, 'resource_allocation'), "Missing resource_allocation"
    assert hasattr(plan, 'stopping_conditions'), "Missing stopping_conditions"
    assert hasattr(plan, 'fallback_plans'), "Missing fallback_plans"
    assert hasattr(plan, 'estimated_time'), "Missing estimated_time"
    assert hasattr(plan, 'estimated_cost'), "Missing estimated_cost"

    # Validate strategy is valid enum
    assert isinstance(plan.strategy, OrchestrationStrategy), "strategy should be OrchestrationStrategy"

    # Validate execution order is not empty
    assert len(plan.execution_order) > 0, "execution_order should not be empty"

    # Validate estimates are positive
    assert plan.estimated_time > 0, "estimated_time should be positive"
    assert plan.estimated_cost > 0, "estimated_cost should be positive"

    print(f"SUCCESS: Plan creation working - strategy: {plan.strategy.value}, time: {plan.estimated_time:.1f}s")
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
    test_pass "Orchestration plan creation"
else
    test_fail "Orchestration plan creation"
fi

###############################################################################
# Test 4: Strategy Selection
###############################################################################
log_info "Test 4: Testing strategy selection logic..."

TEST_PYTHON_TEST4=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator,
        OrchestrationStrategy,
        OptimizationObjective
    )

    # Test with high complexity solution
    orchestrator1 = IntelligentGauntletOrchestrator(
        objective=OptimizationObjective.BALANCED
    )

    plan1 = orchestrator1.create_orchestration_plan(
        solution="class ComplexSolution:\n" + "    def method(self):\n" * 20,
        problem="Solve complex problem",
        domain="math"
    )

    # High complexity should likely trigger adaptive strategy
    print(f"  High complexity -> strategy: {plan1.strategy.value}")

    # Test with low complexity solution
    orchestrator2 = IntelligentGauntletOrchestrator(
        objective=OptimizationObjective.MINIMIZE_TIME
    )

    plan2 = orchestrator2.create_orchestration_plan(
        solution="x = 1",
        problem="Simple problem",
        domain="general"
    )

    # Low complexity with time objective might use parallel
    print(f"  Low complexity -> strategy: {plan2.strategy.value}")

    # Test with medium complexity
    plan3 = orchestrator1.create_orchestration_plan(
        solution="def solve(): return 42",
        problem="Medium problem",
        domain="code"
    )

    print(f"  Medium complexity -> strategy: {plan3.strategy.value}")

    print("SUCCESS: Strategy selection working correctly")
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
    test_pass "Strategy selection logic"
else
    test_fail "Strategy selection logic"
fi

###############################################################################
# Test 5: Resource Allocation
###############################################################################
log_info "Test 5: Testing resource allocation..."

TEST_PYTHON_TEST5=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator
    )

    orchestrator = IntelligentGauntletOrchestrator(max_parallelism=4)

    plan = orchestrator.create_orchestration_plan(
        solution="def solve(): return solution",
        problem="Test problem",
        domain="algorithm"
    )

    # Validate resource allocation
    assert len(plan.resource_allocation) > 0, "Should have resource allocations"

    # Check each round has allocations
    for round_name, allocation in plan.resource_allocation.items():
        assert isinstance(allocation, dict), f"Allocation for {round_name} should be dict"
        # Allocations should have some configuration
        assert len(allocation) > 0, f"Allocation for {round_name} should not be empty"
        print(f"  {round_name}: {allocation}")

    print("SUCCESS: Resource allocation working correctly")
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
    test_pass "Resource allocation"
else
    test_fail "Resource allocation"
fi

###############################################################################
# Test 6: Stopping Conditions
###############################################################################
log_info "Test 6: Testing stopping conditions..."

TEST_PYTHON_TEST6=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator
    )

    orchestrator = IntelligentGauntletOrchestrator()

    plan = orchestrator.create_orchestration_plan(
        solution="def solve(): return result",
        problem="Test problem",
        domain="general"
    )

    # Validate stopping conditions exist
    assert len(plan.stopping_conditions) > 0, "Should have stopping conditions"

    # Validate each condition structure
    for condition in plan.stopping_conditions:
        assert 'condition' in condition, "Missing condition name"
        assert 'threshold' in condition, "Missing threshold"
        assert 'action' in condition, "Missing action"
        print(f"  Condition: {condition['condition']} -> {condition['action']}")

    print("SUCCESS: Stopping conditions configured correctly")
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
    test_pass "Stopping conditions"
else
    test_fail "Stopping conditions"
fi

###############################################################################
# Test 7: Plan Serialization
###############################################################################
log_info "Test 7: Testing plan serialization..."

TEST_PYTHON_TEST7=$(cat <<'EOF'
import sys
import json
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator
    )

    orchestrator = IntelligentGauntletOrchestrator()

    plan = orchestrator.create_orchestration_plan(
        solution="def solve(): return optimal",
        problem="Test",
        domain="code"
    )

    # Test to_dict conversion
    plan_dict = plan.to_dict()

    # Validate dictionary structure
    assert 'strategy' in plan_dict, "Missing strategy"
    assert 'execution_order' in plan_dict, "Missing execution_order"
    assert 'resource_allocation' in plan_dict, "Missing resource_allocation"
    assert 'stopping_conditions' in plan_dict, "Missing stopping_conditions"
    assert 'fallback_plans' in plan_dict, "Missing fallback_plans"
    assert 'estimated_time' in plan_dict, "Missing estimated_time"
    assert 'estimated_cost' in plan_dict, "Missing estimated_cost"

    # Validate strategy is string
    assert isinstance(plan_dict['strategy'], str), "strategy should be string in dict"

    # Test JSON serialization
    json_str = json.dumps(plan_dict)
    assert len(json_str) > 0, "Failed to serialize to JSON"

    print(f"SUCCESS: Plan serialization working - JSON length: {len(json_str)} bytes")
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
    test_pass "Plan serialization"
else
    test_fail "Plan serialization"
fi

###############################################################################
# Test 8: Async Execution
###############################################################################
log_info "Test 8: Testing async execution flow..."

TEST_PYTHON_TEST8=$(cat <<'EOF'
import sys
import asyncio
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator
    )

    async def test_execution():
        orchestrator = IntelligentGauntletOrchestrator()

        # Create plan
        plan = orchestrator.create_orchestration_plan(
            solution="def solve(): return 42",
            problem="Test problem",
            domain="general"
        )

        # Execute orchestration
        result = await orchestrator.execute_orchestration(
            solution="def solve(): return 42",
            problem="Test problem",
            domain="general",
            plan=plan
        )

        # Validate result structure
        assert hasattr(result, 'passed'), "Missing passed"
        assert hasattr(result, 'final_score'), "Missing final_score"
        assert hasattr(result, 'rounds_completed'), "Missing rounds_completed"
        assert hasattr(result, 'execution_time'), "Missing execution_time"
        assert hasattr(result, 'actual_cost'), "Missing actual_cost"
        assert hasattr(result, 'adaptations_made'), "Missing adaptations_made"
        assert hasattr(result, 'recommendations'), "Missing recommendations"

        # Validate types
        assert isinstance(result.passed, bool), "passed should be bool"
        assert isinstance(result.final_score, float), "final_score should be float"
        assert isinstance(result.rounds_completed, int), "rounds_completed should be int"
        assert isinstance(result.execution_time, float), "execution_time should be float"
        assert result.execution_time >= 0, "execution_time should be non-negative"

        return result

    result = asyncio.run(test_execution())

    print(f"SUCCESS: Async execution working - passed: {result.passed}, score: {result.final_score:.3f}, rounds: {result.rounds_completed}")
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
    test_pass "Async execution flow"
else
    test_fail "Async execution flow"
fi

###############################################################################
# Test 9: Different Objectives
###############################################################################
log_info "Test 9: Testing different optimization objectives..."

TEST_PYTHON_TEST9=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator,
        OptimizationObjective
    )

    objectives = [
        OptimizationObjective.MAXIMIZE_ACCURACY,
        OptimizationObjective.MINIMIZE_TIME,
        OptimizationObjective.MINIMIZE_COST,
        OptimizationObjective.MAXIMIZE_THROUGHPUT,
        OptimizationObjective.BALANCED
    ]

    for objective in objectives:
        orchestrator = IntelligentGauntletOrchestrator(objective=objective)

        plan = orchestrator.create_orchestration_plan(
            solution="def solve(): return result",
            problem="Test",
            domain="general"
        )

        # Each objective should produce a valid plan
        assert plan.estimated_time > 0, f"{objective.value} produced invalid time estimate"
        assert plan.estimated_cost > 0, f"{objective.value} produced invalid cost estimate"

        print(f"  {objective.value}: time={plan.estimated_time:.1f}s, cost={plan.estimated_cost:.2f}")

    print("SUCCESS: All optimization objectives working")
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
    test_pass "Different optimization objectives"
else
    test_fail "Different optimization objectives"
fi

###############################################################################
# Test 10: Statistics Tracking
###############################################################################
log_info "Test 10: Testing statistics tracking..."

TEST_PYTHON_TEST10=$(cat <<'EOF'
import sys
import asyncio
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
        IntelligentGauntletOrchestrator
    )

    async def test_stats():
        orchestrator = IntelligentGauntletOrchestrator()

        # Run a few executions
        for i in range(3):
            await orchestrator.execute_orchestration(
                solution=f"def solve_{i}(): return {i}",
                problem=f"Problem {i}",
                domain="general"
            )

        # Get statistics
        stats = orchestrator.get_orchestration_stats()

        # Validate stats
        assert 'total_executions' in stats, "Missing total_executions"
        assert 'pass_rate' in stats, "Missing pass_rate"
        assert 'average_score' in stats, "Missing average_score"
        assert 'average_time' in stats, "Missing average_time"
        assert 'strategy_distribution' in stats, "Missing strategy_distribution"

        # Validate values
        assert stats['total_executions'] == 3, "Should have 3 executions"
        assert isinstance(stats['pass_rate'], float), "pass_rate should be float"
        assert isinstance(stats['strategy_distribution'], dict), "strategy_distribution should be dict"

        return stats

    stats = asyncio.run(test_stats())

    print(f"SUCCESS: Statistics tracking working - executions: {stats['total_executions']}, pass_rate: {stats['pass_rate']:.2f}")
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
    test_pass "Statistics tracking"
else
    test_fail "Statistics tracking"
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
    log_info "✓ All intelligent orchestrator tests passed!"
    exit 0
fi
