#!/bin/bash
###############################################################################
# PROBE: Adaptive MDAP API Verification
#
# Federation Constitution - Law 2: Runtime Truth
# "We trust execution, not documentation."
#
# This probe verifies that the Adaptive MDAP module is properly installed
# and all expected APIs are accessible and functional.
#
# Usage: ./probes/check_adaptive_mdap_api.sh
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Log functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test result function
test_result() {
    local test_name="$1"
    local result="$2"
    local details="${3:-}"

    if [ "$result" = "PASS" ]; then
        log_info "✓ PASS: $test_name"
        ((TESTS_PASSED++))
        if [ -n "$details" ]; then
            echo "  └─ $details"
        fi
    else
        log_error "✗ FAIL: $test_name"
        ((TESTS_FAILED++))
        if [ -n "$details" ]; then
            echo "  └─ $details"
        fi
    fi
}

# Create a temporary Python script for testing
create_test_script() {
    local test_code="$1"
    echo "$test_code" > /tmp/adaptive_mdap_test_$$.py
}

# Cleanup function
cleanup() {
    rm -f /tmp/adaptive_mdap_test_$$.py
}

trap cleanup EXIT

###############################################################################
# TEST SUITE: Adaptive MDAP Module Availability
###############################################################################

echo "=========================================="
echo "PROBE: Adaptive MDAP API Verification"
echo "=========================================="
echo ""

###############################################################################
# Test 1: Module Import
###############################################################################
test_name="Module Import: adaptive_mdap"
test_code="
import sys
try:
    from adaptive_mdap import (
        TaskComplexityClassifier,
        AdaptiveMDAPAllocator,
        AdaptiveExecutionController,
        get_health_checker,
        ComplexityScore,
        SubProblem,
        AdaptiveWorkflowIntegration,
        AdaptiveWorkflowConfig,
        get_adaptive_workflow
    )
    print('PASS')
    sys.exit(0)
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "All core classes importable"
else
    test_result "$test_name" "FAIL" "Module import failed"
fi

###############################################################################
# Test 2: TaskComplexityClassifier Instantiation
###############################################################################
test_name="TaskComplexityClassifier: Instantiation"
test_code="
import sys
try:
    from adaptive_mdap import TaskComplexityClassifier, SubProblem
    classifier = TaskComplexityClassifier()
    subproblem = SubProblem(
        id='test-001',
        description='Test problem for complexity analysis',
        domain='general',
        depth=2,
        dependencies=['dep-001']
    )
    score = classifier.compute_complexity(subproblem)
    assert hasattr(score, 'overall_score'), 'Missing overall_score attribute'
    assert isinstance(score.overall_score, float), 'overall_score must be float'
    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Classifier works correctly"
else
    test_result "$test_name" "FAIL" "Classifier failed"
fi

###############################################################################
# Test 3: AdaptiveMDAPAllocator Resource Allocation
###############################################################################
test_name="AdaptiveMDAPAllocator: Resource Allocation"
test_code="
import sys
try:
    from adaptive_mdap import AdaptiveMDAPAllocator
    allocator = AdaptiveMDAPAllocator()

    # Test low complexity allocation
    strategy_low = allocator.allocate_resources(0.2)
    assert strategy_low.n_agents >= 1, 'Low complexity should have at least 1 agent'

    # Test medium complexity allocation
    strategy_med = allocator.allocate_resources(0.5)
    assert strategy_med.n_agents >= 1, 'Medium complexity should have at least 1 agent'

    # Test high complexity allocation
    strategy_high = allocator.allocate_resources(0.9)
    assert strategy_high.n_agents >= 1, 'High complexity should have at least 1 agent'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Resource allocation works for all complexity levels"
else
    test_result "$test_name" "FAIL" "Resource allocation failed"
fi

###############################################################################
# Test 4: AdaptiveExecutionController Metrics Tracking
###############################################################################
test_name="AdaptiveExecutionController: Metrics Tracking"
test_code="
import sys
import time
try:
    from adaptive_mdap import AdaptiveExecutionController
    controller = AdaptiveExecutionController()

    # Record some metrics
    controller.record_execution('task-001', True, 1.5)
    controller.record_execution('task-002', False, 2.3)

    assert len(controller.metrics) == 2, 'Should have 2 recorded metrics'
    assert controller.metrics[0]['success'] == True, 'First task should be marked as success'
    assert controller.metrics[1]['success'] == False, 'Second task should be marked as failure'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Controller tracks metrics correctly"
else
    test_result "$test_name" "FAIL" "Controller metrics failed"
fi

###############################################################################
# Test 5: Health Checker Availability
###############################################################################
test_name="Health Checker: Availability and Response"
test_code="
import sys
try:
    from adaptive_mdap import get_health_checker
    checker = get_health_checker()
    result = checker.check()
    assert isinstance(result, dict), 'Health check should return dict'
    assert 'status' in result, 'Health check should include status'
    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Health checker functional"
else
    test_result "$test_name" "FAIL" "Health check failed"
fi

###############################################################################
# Test 6: Workflow Integration
###############################################################################
test_name="AdaptiveWorkflowIntegration: Get Solver Config"
test_code="
import sys
try:
    from adaptive_mdap import get_adaptive_workflow, SubProblem
    workflow = get_adaptive_workflow()

    subproblem = SubProblem(
        id='workflow-test-001',
        description='Test workflow integration',
        domain='ml',
        depth=3
    )

    config = workflow.get_solver_config(subproblem)
    assert 'complexity_score' in config, 'Config should include complexity_score'
    assert 'strategy' in config, 'Config should include strategy'
    assert 'n_agents' in config, 'Config should include n_agents'
    assert 'adaptive' in config, 'Config should include adaptive flag'
    assert config['adaptive'] == True, 'Adaptive flag should be True'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Workflow integration functional"
else
    test_result "$test_name" "FAIL" "Workflow integration failed"
fi

###############################################################################
# Test 7: Complexity Score Structure
###############################################################################
test_name="ComplexityScore: Data Structure Validation"
test_code="
import sys
try:
    from adaptive_mdap import ComplexityScore
    score = ComplexityScore(
        overall_score=0.75,
        text_length_score=0.6,
        dependency_score=0.8,
        depth_score=0.9,
        feature_weights={'text': 0.4, 'dep': 0.3}
    )
    assert score.overall_score == 0.75, 'overall_score should be 0.75'
    assert score.text_length_score == 0.6, 'text_length_score should be 0.6'
    assert len(score.feature_weights) == 2, 'Should have 2 feature weights'
    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "ComplexityScore structure valid"
else
    test_result "$test_name" "FAIL" "ComplexityScore validation failed"
fi

###############################################################################
# Test 8: SubProblem Data Structure
###############################################################################
test_name="SubProblem: Data Structure Validation"
test_code="
import sys
try:
    from adaptive_mdap import SubProblem
    sub = SubProblem(
        id='sub-001',
        description='Test subproblem',
        domain='math',
        depth=2,
        dependencies=['sub-000'],
        metadata={'key': 'value'}
    )
    assert sub.id == 'sub-001', 'ID should be sub-001'
    assert sub.domain == 'math', 'Domain should be math'
    assert sub.depth == 2, 'Depth should be 2'
    assert len(sub.dependencies) == 1, 'Should have 1 dependency'
    assert sub.metadata['key'] == 'value', 'Metadata should be preserved'
    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/adaptive_mdap_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "SubProblem structure valid"
else
    test_result "$test_name" "FAIL" "SubProblem validation failed"
fi

###############################################################################
# SUMMARY
###############################################################################

echo ""
echo "=========================================="
echo "PROBE SUMMARY: Adaptive MDAP API"
echo "=========================================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_info "✓ ALL TESTS PASSED - Adaptive MDAP API is functional"
    echo ""
    echo "Probe Result: SUCCESS (200 OK)"
    exit 0
else
    log_error "✗ SOME TESTS FAILED - Adaptive MDAP API has issues"
    echo ""
    echo "Probe Result: FAILURE (503 Service Unavailable)"
    exit 1
fi
