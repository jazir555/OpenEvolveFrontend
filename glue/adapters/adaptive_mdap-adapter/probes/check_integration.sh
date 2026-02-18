#!/bin/bash
###############################################################################
# PROBE: MDAP/MAKER Integration Verification
#
# Federation Constitution - Law 2: Runtime Truth
# "We trust execution, not documentation."
#
# This probe verifies that the MDAP/MAKER integration module is properly
# installed and all expected integration points are functional.
#
# Usage: ./probes/check_integration.sh
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
    echo "$test_code" > /tmp/integration_test_$$.py
}

# Cleanup function
cleanup() {
    rm -f /tmp/integration_test_$$.py
}

trap cleanup EXIT

###############################################################################
# TEST SUITE: MDAP/MAKER Integration
###############################################################################

echo "=========================================="
echo "PROBE: MDAP/MAKER Integration Verification"
echo "=========================================="
echo ""

###############################################################################
# Test 1: Integration Module Import
###############################################################################
test_name="Module Import: mdap_maker_gauntlet_integration"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import (
        MDAPMakerGauntletMode,
        MDAPMakerGauntletConfig,
        MDAPMakerGauntletResult,
        MDAPMakerGauntletIntegration,
        create_mdap_maker_integration,
        execute_gauntlet_with_mdap
    )
    print('PASS')
    sys.exit(0)
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Integration module imports successfully"
else
    test_result "$test_name" "FAIL" "Integration module import failed"
fi

###############################################################################
# Test 2: MDAPMakerGauntletMode Enum
###############################################################################
test_name="MDAPMakerGauntletMode: Enum Values"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletMode

    modes = [
        MDAPMakerGauntletMode.MDAP_ADAPTIVE,
        MDAPMakerGauntletMode.MAKER_VOTING,
        MDAPMakerGauntletMode.HYBRID,
        MDAPMakerGauntletMode.CONSENSUS
    ]

    assert len(modes) == 4, 'Should have 4 modes'
    assert modes[0].value == 'mdap_adaptive', 'MDAP_ADAPTIVE value correct'
    assert modes[1].value == 'maker_voting', 'MAKER_VOTING value correct'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "All mode enums available"
else
    test_result "$test_name" "FAIL" "Mode enum check failed"
fi

###############################################################################
# Test 3: MDAPMakerGauntletConfig Configuration
###############################################################################
test_name="MDAPMakerGauntletConfig: Configuration"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import (
        MDAPMakerGauntletConfig,
        MDAPMakerGauntletMode
    )

    config = MDAPMakerGauntletConfig(
        mode=MDAPMakerGauntletMode.HYBRID,
        use_complexity_adaptation=True,
        use_maker_voting=True,
        use_red_flagging=True,
        min_complexity_threshold=0.3,
        max_complexity_threshold=0.8,
        maker_k_min=2,
        maker_k_max=5,
        maker_max_votes=30
    )

    assert config.mode == MDAPMakerGauntletMode.HYBRID, 'Mode should be HYBRID'
    assert config.use_complexity_adaptation == True, 'Complexity adaptation enabled'
    assert config.maker_k_min == 2, 'k_min should be 2'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Configuration object works"
else
    test_result "$test_name" "FAIL" "Configuration failed"
fi

###############################################################################
# Test 4: MDAPMakerGauntletIntegration Instantiation
###############################################################################
test_name="MDAPMakerGauntletIntegration: Instantiation (All Modes)"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import (
        MDAPMakerGauntletIntegration,
        MDAPMakerGauntletMode,
        MDAPMakerGauntletConfig
    )

    modes = [
        MDAPMakerGauntletMode.MDAP_ADAPTIVE,
        MDAPMakerGauntletMode.MAKER_VOTING,
        MDAPMakerGauntletMode.HYBRID,
        MDAPMakerGauntletMode.CONSENSUS
    ]

    for mode in modes:
        config = MDAPMakerGauntletConfig(mode=mode)
        integration = MDAPMakerGauntletIntegration(config=config)
        assert integration.config.mode == mode, f'Mode {mode} should be set'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "All 4 modes instantiate correctly"
else
    test_result "$test_name" "FAIL" "Instantiation failed"
fi

###############################################################################
# Test 5: Complexity Analysis
###############################################################################
test_name="Integration: Complexity Analysis"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration

    integration = MDAPMakerGauntletIntegration()

    complexity_score = integration._analyze_complexity(
        problem_description='Implement a secure authentication system with OAuth2 support',
        solution={'code': 'def auth(): pass'},
        context={'domain': 'security'}
    )

    assert hasattr(complexity_score, 'overall_score'), 'Should have overall_score'
    assert 0 <= complexity_score.overall_score <= 1, 'Score should be between 0 and 1'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Complexity analysis functional"
else
    test_result "$test_name" "FAIL" "Complexity analysis failed"
fi

###############################################################################
# Test 6: Gauntlet Adaptation
###############################################################################
test_name="Integration: Gauntlet Adaptation"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
    from adaptive_mdap import ComplexityScore

    integration = MDAPMakerGauntletIntegration()

    # Test low complexity adaptation
    low_complexity = ComplexityScore(overall_score=0.2)
    low_config = integration._adapt_gauntlet_config(low_complexity)
    assert low_config is not None, 'Low complexity config should exist'

    # Test medium complexity adaptation
    med_complexity = ComplexityScore(overall_score=0.5)
    med_config = integration._adapt_gauntlet_config(med_complexity)
    assert med_config is not None, 'Medium complexity config should exist'

    # Test high complexity adaptation
    high_complexity = ComplexityScore(overall_score=0.9)
    high_config = integration._adapt_gauntlet_config(high_complexity)
    assert high_config is not None, 'High complexity config should exist'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Gauntlet adaptation works for all complexity levels"
else
    test_result "$test_name" "FAIL" "Gauntlet adaptation failed"
fi

###############################################################################
# Test 7: Consensus Calculation
###############################################################################
test_name="Integration: Consensus Calculation"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration

    integration = MDAPMakerGauntletIntegration()

    # Test high consensus
    agent_votes_high = [
        {'score': 0.85, 'justification': 'Excellent'},
        {'score': 0.87, 'justification': 'Very good'},
        {'score': 0.83, 'justification': 'Good'}
    ]
    consensus_reached, consensus_score = integration._calculate_consensus(
        agent_votes_high,
        {'passed': True}
    )
    assert consensus_reached == True, 'High consensus should be reached'
    assert consensus_score > 0.8, 'Consensus score should be high'

    # Test low consensus
    agent_votes_low = [
        {'score': 0.3, 'justification': 'Poor'},
        {'score': 0.9, 'justification': 'Excellent'},
        {'score': 0.4, 'justification': 'Fair'}
    ]
    consensus_reached, consensus_score = integration._calculate_consensus(
        agent_votes_low,
        {'passed': False}
    )
    assert consensus_reached == False, 'Low consensus should not be reached'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Consensus calculation works correctly"
else
    test_result "$test_name" "FAIL" "Consensus calculation failed"
fi

###############################################################################
# Test 8: Convenience Functions
###############################################################################
test_name="Integration: Convenience Functions"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import (
        create_mdap_maker_integration,
        execute_gauntlet_with_mdap,
        MDAPMakerGauntletMode
    )

    # Test create function
    integration = create_mdap_maker_integration(
        mode=MDAPMakerGauntletMode.HYBRID,
        use_complexity_adaptation=True,
        use_maker_voting=True
    )
    assert integration is not None, 'Integration should be created'
    assert integration.config.mode == MDAPMakerGauntletMode.HYBRID, 'Mode should be HYBRID'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Convenience functions work"
else
    test_result "$test_name" "FAIL" "Convenience functions failed"
fi

###############################################################################
# Test 9: MDAP-Adaptive Gauntlet Creation
###############################################################################
test_name="Integration: MDAP-Adaptive Gauntlet Creation"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration

    integration = MDAPMakerGauntletIntegration()

    # Test with simple problem (should select simple gauntlet)
    gauntlet_simple, result_simple = integration.create_mdap_adaptive_gauntlet(
        problem_description='Simple addition function',
        solution={'code': 'def add(a, b): return a + b'},
        context={'domain': 'math'}
    )
    assert gauntlet_simple is not None, 'Simple gauntlet should be created'
    assert result_simple is not None, 'Result should exist'

    # Test with complex problem (should select advanced gauntlet)
    gauntlet_complex, result_complex = integration.create_mdap_adaptive_gauntlet(
        problem_description='Implement a distributed consensus algorithm for blockchain',
        solution={'code': 'class Consensus: ...'},
        context={'domain': 'distributed_systems'}
    )
    assert gauntlet_complex is not None, 'Complex gauntlet should be created'
    assert result_complex is not None, 'Result should exist'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Adaptive gauntlet creation works"
else
    test_result "$test_name" "FAIL" "Adaptive gauntlet creation failed"
fi

###############################################################################
# Test 10: Result Structure
###############################################################################
test_name="Integration: Result Data Structure"
test_code="
import sys
try:
    from mdap_maker_gauntlet_integration import MDAPMakerGauntletResult
    from adaptive_mdap import ComplexityScore
    from gauntlet_types import GauntletResult

    result = MDAPMakerGauntletResult(
        gauntlet_result=GauntletResult(
            gauntlet_type='adversarial',
            passed=True,
            score=0.85,
            details={}
        ),
        complexity_score=ComplexityScore(overall_score=0.7),
        agent_votes=[],
        red_flags=[],
        consensus_reached=True,
        consensus_score=0.88,
        mdap_strategy='MAKER_ULTRA',
        execution_time_ms=1500
    )

    assert result.gauntlet_result.passed == True, 'Gauntlet result should reflect pass'
    assert result.complexity_score.overall_score == 0.7, 'Complexity score should be preserved'
    assert result.consensus_reached == True, 'Consensus reached should be True'
    assert result.mdap_strategy == 'MAKER_ULTRA', 'Strategy should be preserved'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/integration_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "Result structure valid"
else
    test_result "$test_name" "FAIL" "Result structure validation failed"
fi

###############################################################################
# SUMMARY
###############################################################################

echo ""
echo "=========================================="
echo "PROBE SUMMARY: MDAP/MAKER Integration"
echo "=========================================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_info "✓ ALL TESTS PASSED - MDAP/MAKER Integration is functional"
    echo ""
    echo "Integration Status:"
    echo "  - MDAP Adaptive: ✓ Operational"
    echo "  - MAKER Voting: ✓ Operational"
    echo "  - Consensus: ✓ Operational"
    echo "  - Gauntlet Adaptation: ✓ Operational"
    echo ""
    echo "Probe Result: SUCCESS (200 OK)"
    exit 0
else
    log_error "✗ SOME TESTS FAILED - MDAP/MAKER Integration has issues"
    echo ""
    echo "Probe Result: FAILURE (503 Service Unavailable)"
    exit 1
fi
