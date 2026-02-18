#!/bin/bash
###############################################################################
# PROBE: MAKER Engine API Verification
#
# Federation Constitution - Law 2: Runtime Truth
# "We trust execution, not documentation."
#
# This probe verifies that the MAKER Engine module is properly installed
# and all expected APIs are accessible and functional.
#
# Usage: ./probes/check_maker_api.sh
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
    echo "$test_code" > /tmp/maker_api_test_$$.py
}

# Cleanup function
cleanup() {
    rm -f /tmp/maker_api_test_$$.py
}

trap cleanup EXIT

###############################################################################
# TEST SUITE: MAKER Engine Module Availability
###############################################################################

echo "=========================================="
echo "PROBE: MAKER Engine API Verification"
echo "=========================================="
echo ""

###############################################################################
# Test 1: Module Import
###############################################################################
test_name="Module Import: maker_engine"
test_code="
import sys
try:
    from maker_engine import (
        MakerEngine,
        MakerConfig,
        MakerState,
        MakerStep,
        MakerRunResult,
        CheckpointStore,
        FileCheckpointStore
    )
    print('PASS')
    sys.exit(0)
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "All core classes importable"
else
    test_result "$test_name" "FAIL" "Module import failed"
fi

###############################################################################
# Test 2: MDAP Engine Import (RedFlagger dependency)
###############################################################################
test_name="Module Import: mdap_engine (RedFlagger)"
test_code="
import sys
try:
    from mdap_engine import (
        RedFlagRules,
        RedFlagger,
        canonicalize_candidate
    )
    print('PASS')
    sys.exit(0)
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "MDAP Engine (RedFlagger) importable"
else
    test_result "$test_name" "FAIL" "MDAP Engine import failed"
fi

###############################################################################
# Test 3: MakerConfig Configuration
###############################################################################
test_name="MakerConfig: Configuration Validation"
test_code="
import sys
try:
    from maker_engine import MakerConfig
    from mdap_engine import RedFlagRules

    config = MakerConfig(
        k_min=2,
        k_max=7,
        max_votes_per_step=50,
        max_steps=100,
        timeout_seconds=60,
        checkpoint_interval=20,
        red_flag_rules=RedFlagRules()
    )

    assert config.k_min == 2, 'k_min should be 2'
    assert config.k_max == 7, 'k_max should be 7'
    assert config.max_votes_per_step == 50, 'max_votes_per_step should be 50'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "MakerConfig configuration works"
else
    test_result "$test_name" "FAIL" "MakerConfig failed"
fi

###############################################################################
# Test 4: MakerState Management
###############################################################################
test_name="MakerState: State Management"
test_code="
import sys
try:
    from maker_engine import MakerState

    state = MakerState(
        step_index=0,
        current_state={'value': 42},
        history=[],
        last_action=None
    )

    assert state.step_index == 0, 'Initial step_index should be 0'
    assert state.current_state['value'] == 42, 'Current state should be preserved'
    assert len(state.history) == 0, 'Initial history should be empty'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "MakerState management works"
else
    test_result "$test_name" "FAIL" "MakerState failed"
fi

###############################################################################
# Test 5: MakerStep Structure
###############################################################################
test_name="MakerStep: Step Structure"
test_code="
import sys
try:
    from maker_engine import MakerStep

    step = MakerStep(
        step_id='step-001',
        prompt_template='Process this: {state}',
        expected_schema={'type': 'object'},
        task_type='general',
        priority=1,
        system_prompt='You are a helpful assistant',
        stop_sequences=None,
        metadata={'key': 'value'}
    )

    assert step.step_id == 'step-001', 'step_id should be preserved'
    assert step.task_type == 'general', 'task_type should be general'
    assert step.priority == 1, 'priority should be 1'

    # Test prompt rendering
    rendered = step.render_prompt({'test': 'data'}, [])
    assert 'test' in rendered, 'Rendered prompt should contain state data'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "MakerStep structure valid"
else
    test_result "$test_name" "FAIL" "MakerStep failed"
fi

###############################################################################
# Test 6: FileCheckpointStore Persistence
###############################################################################
test_name="FileCheckpointStore: Persistence"
test_code="
import sys
import os
import tempfile
try:
    from maker_engine import MakerState, FileCheckpointStore

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        # Create state and save
        state = MakerState(
            step_index=5,
            current_state={'checkpointed': True},
            history=[{'action': 'test'}],
            last_action='test_action'
        )

        store = FileCheckpointStore(temp_path)
        store.save(state)

        # Load and verify
        loaded_state = store.load()
        assert loaded_state is not None, 'State should be loadable'
        assert loaded_state.step_index == 5, 'step_index should be preserved'
        assert loaded_state.current_state['checkpointed'] == True, 'State should be preserved'

        print('PASS')
        sys.exit(0)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "FileCheckpointStore persistence works"
else
    test_result "$test_name" "FAIL" "FileCheckpointStore failed"
fi

###############################################################################
# Test 7: RedFlagger Functionality
###############################################################################
test_name="RedFlagger: Red Flag Detection"
test_code="
import sys
try:
    from mdap_engine import RedFlagger, RedFlagRules

    rules = RedFlagRules()
    flagger = RedFlagger(rules)

    # Test with safe content
    safe_text = 'This is a safe response with actionable content.'
    safe_candidate = {'action': 'continue', 'reason': 'Safe to proceed'}
    is_flagged, reason = flagger.is_flagged(safe_text, safe_candidate, None)

    # Note: This test verifies the API is callable, actual flagging behavior depends on rules
    assert isinstance(is_flagged, bool), 'is_flagged should return boolean'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "RedFlagger API functional"
else
    test_result "$test_name" "FAIL" "RedFlagger failed"
fi

###############################################################################
# Test 8: Canonical Candidate Function
###############################################################################
test_name="canonicalize_candidate: Utility Function"
test_code="
import sys
try:
    from mdap_engine import canonicalize_candidate

    # Test with dict candidate
    dict_candidate = {'action': 'test', 'value': 42}
    canonical = canonicalize_candidate(dict_candidate)
    assert isinstance(canonical, str), 'Should return string key'

    # Test with string candidate
    str_candidate = 'simple_action'
    canonical = canonicalize_candidate(str_candidate)
    assert canonical == 'simple_action', 'String candidate should be preserved'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "canonicalize_candidate works"
else
    test_result "$test_name" "FAIL" "canonicalize_candidate failed"
fi

###############################################################################
# Test 9: MakerEngine Initialization
###############################################################################
test_name="MakerEngine: Initialization"
test_code="
import sys
try:
    from maker_engine import MakerEngine, MakerConfig
    from mdap_engine import RedFlagRules
    from workflow_structures import Team, ModelConfig

    # Create minimal team
    team = Team(
        name='test_team',
        members=[
            ModelConfig(
                member_id='test-001',
                api_key='test-key',
                model_id='gpt-4'
            )
        ]
    )

    config = MakerConfig(
        k_min=2,
        k_max=5,
        max_votes_per_step=30
    )

    engine = MakerEngine(team=team, config=config)
    assert engine.team.name == 'test_team', 'Team should be set'
    assert engine.config.k_min == 2, 'Config should be set'
    assert len(engine.metrics) == 5, 'Should have 5 metrics'

    print('PASS')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

create_test_script "$test_code"
if python /tmp/maker_api_test_$$.py 2>/dev/null; then
    test_result "$test_name" "PASS" "MakerEngine initialization works"
else
    test_result "$test_name" "FAIL" "MakerEngine initialization failed"
fi

###############################################################################
# SUMMARY
###############################################################################

echo ""
echo "=========================================="
echo "PROBE SUMMARY: MAKER Engine API"
echo "=========================================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_info "✓ ALL TESTS PASSED - MAKER Engine API is functional"
    echo ""
    echo "Probe Result: SUCCESS (200 OK)"
    exit 0
else
    log_error "✗ SOME TESTS FAILED - MAKER Engine API has issues"
    echo ""
    echo "Probe Result: FAILURE (503 Service Unavailable)"
    exit 1
fi
