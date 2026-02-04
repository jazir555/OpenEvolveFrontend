#!/bin/bash

###############################################################################
# WebSocket API Probe
#
# Validates WebSocket API functionality per CLAUDE.md Law 2.
#
# Tests:
# 1. Module import verification
# 2. Event creation and serialization
# 3. Connection manager functionality
# 4. WebSocket server instantiation
# 5. Event broadcasting
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

log_info "WebSocket API Probe"
log_info "==================="
log_info "Project root: $PROJ_ROOT"
echo ""

###############################################################################
# Test 1: Module Import Verification
###############################################################################
log_info "Test 1: Verifying WebSocket API module import..."

TEST_PYTHON_TEST1=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType,
        ConnectionManager,
        GauntletWebSocketServer
    )
    print("SUCCESS: All WebSocket API classes imported successfully")
    exit(0)
except ImportError as e:
    print(f"FAIL: Cannot import WebSocket API: {e}")
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
    log_error "Failed to import WebSocket API module"
    exit 1
fi

###############################################################################
# Test 2: Event Creation
###############################################################################
log_info "Test 2: Testing WebSocket event creation..."

TEST_PYTHON_TEST2=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType
    )

    # Test event creation with different types
    event1 = WebSocketEvent(
        event_type=EventType.EXECUTION_STARTED,
        data={"execution_id": "exec_123"},
        execution_id="exec_123"
    )

    assert event1.event_type == EventType.EXECUTION_STARTED
    assert event1.data["execution_id"] == "exec_123"
    assert event1.execution_id == "exec_123"

    # Test with different event type
    event2 = WebSocketEvent(
        event_type=EventType.PROGRESS_UPDATE,
        data={"progress": 0.5, "round": 1}
    )

    assert event2.event_type == EventType.PROGRESS_UPDATE
    assert event2.data["progress"] == 0.5

    # Test error event
    event3 = WebSocketEvent(
        event_type=EventType.ERROR,
        data={"error": "Test error"}
    )

    assert event3.event_type == EventType.ERROR

    print("SUCCESS: WebSocket event creation working correctly")
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
    test_pass "WebSocket event creation"
else
    test_fail "WebSocket event creation"
fi

###############################################################################
# Test 3: Event Serialization
###############################################################################
log_info "Test 3: Testing event serialization to JSON..."

TEST_PYTHON_TEST3=$(cat <<'EOF'
import sys
import json
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType
    )

    # Create event
    event = WebSocketEvent(
        event_type=EventType.ROUND_COMPLETED,
        data={"round_number": 1, "score": 0.85},
        execution_id="exec_123"
    )

    # Test to_json conversion
    json_str = event.to_json()

    # Validate JSON structure
    data = json.loads(json_str)
    assert data["event_type"] == "round_completed"
    assert data["data"]["round_number"] == 1
    assert data["data"]["score"] == 0.85
    assert data["execution_id"] == "exec_123"
    assert "timestamp" in data

    # Test from_json restoration
    restored_event = WebSocketEvent.from_json(json_str)
    assert restored_event.event_type == EventType.ROUND_COMPLETED
    assert restored_event.data["score"] == 0.85
    assert restored_event.execution_id == "exec_123"

    print(f"SUCCESS: Event serialization working - JSON: {len(json_str)} bytes")
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
    test_pass "Event serialization"
else
    test_fail "Event serialization"
fi

###############################################################################
# Test 4: Connection Manager Initialization
###############################################################################
log_info "Test 4: Testing connection manager initialization..."

TEST_PYTHON_TEST4=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import ConnectionManager

    # Create manager
    manager = ConnectionManager()

    # Validate initial state
    assert hasattr(manager, 'active_connections'), "Missing active_connections"
    assert hasattr(manager, 'connection_auth'), "Missing connection_auth"
    assert hasattr(manager, 'execution_subscriptions'), "Missing execution_subscriptions"

    # Validate empty state
    assert len(manager.active_connections) == 0, "Should start with no connections"
    assert len(manager.connection_auth) == 0, "Should start with no auth tokens"
    assert len(manager.execution_subscriptions) == 0, "Should start with no subscriptions"

    # Test get_connection_count
    count = manager.get_connection_count()
    assert count == 0, f"Initial connection count should be 0, got {count}"
    assert isinstance(count, int), "Connection count should be int"

    print("SUCCESS: Connection manager initialization working correctly")
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
    test_pass "Connection manager initialization"
else
    test_fail "Connection manager initialization"
fi

###############################################################################
# Test 5: Connection Subscription Management
###############################################################################
log_info "Test 5: Testing connection subscription management..."

TEST_PYTHON_TEST5=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve\Frontend')

try:
    from api.gauntlets_websocket import ConnectionManager

    manager = ConnectionManager()

    # Simulate subscriptions
    connection_id = "conn_123"
    execution_id = "exec_456"

    # Subscribe
    manager.subscribe_to_execution(connection_id, execution_id)

    # Validate subscription
    assert execution_id in manager.execution_subscriptions, "Execution ID not in subscriptions"
    assert connection_id in manager.execution_subscriptions[execution_id], "Connection not subscribed"

    # Test multiple connections to same execution
    manager.subscribe_to_execution("conn_789", execution_id)
    assert len(manager.execution_subscriptions[execution_id]) == 2, "Should have 2 subscribers"

    # Unsubscribe
    manager.unsubscribe_from_execution(connection_id, execution_id)
    assert connection_id not in manager.execution_subscriptions[execution_id], "Connection should be unsubscribed"
    assert len(manager.execution_subscriptions[execution_id]) == 1, "Should have 1 subscriber left"

    print("SUCCESS: Connection subscription management working correctly")
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
    test_pass "Connection subscription management"
else
    test_fail "Connection subscription management"
fi

###############################################################################
# Test 6: WebSocket Server Initialization
###############################################################################
log_info "Test 6: Testing WebSocket server initialization..."

TEST_PYTHON_TEST6=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import GauntletWebSocketServer

    # Create server
    server = GauntletWebSocketServer(
        host="localhost",
        port=8765,
        ping_interval=30.0,
        ping_timeout=10.0
    )

    # Validate server attributes
    assert server.host == "localhost", "Host not set correctly"
    assert server.port == 8765, "Port not set correctly"
    assert server.ping_interval == 30.0, "Ping interval not set correctly"
    assert server.ping_timeout == 10.0, "Ping timeout not set correctly"

    # Validate manager exists
    assert server.manager is not None, "Connection manager not initialized"

    print(f"SUCCESS: WebSocket server initialization working - {server.host}:{server.port}")
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
    test_pass "WebSocket server initialization"
else
    test_fail "WebSocket server initialization"
fi

###############################################################################
# Test 7: Event Broadcasting Methods Exist
###############################################################################
log_info "Test 7: Testing event broadcasting methods..."

TEST_PYTHON_TEST7=$(cat <<'EOF'
import sys
import asyncio
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import GauntletWebSocketServer

    server = GauntletWebSocketServer(host="localhost", port=8765)

    # Check that broadcast methods exist and are callable
    assert hasattr(server, 'broadcast_execution_progress'), "Missing broadcast_execution_progress method"
    assert callable(server.broadcast_execution_progress), "broadcast_execution_progress not callable"

    assert hasattr(server, 'broadcast_round_completed'), "Missing broadcast_round_completed method"
    assert callable(server.broadcast_round_completed), "broadcast_round_completed not callable"

    assert hasattr(server, 'broadcast_execution_completed'), "Missing broadcast_execution_completed method"
    assert callable(server.broadcast_execution_completed), "broadcast_execution_completed not callable"

    assert hasattr(server, 'broadcast_error'), "Missing broadcast_error method"
    assert callable(server.broadcast_error), "broadcast_error not callable"

    # Test method signatures (async check)
    import inspect
    assert inspect.iscoroutinefunction(server.broadcast_execution_progress), "broadcast_execution_progress should be async"
    assert inspect.iscoroutinefunction(server.broadcast_round_completed), "broadcast_round_completed should be async"
    assert inspect.iscoroutinefunction(server.broadcast_execution_completed), "broadcast_execution_completed should be async"
    assert inspect.iscoroutinefunction(server.broadcast_error), "broadcast_error should be async"

    print("SUCCESS: Event broadcasting methods exist and are async")
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
    test_pass "Event broadcasting methods"
else
    test_fail "Event broadcasting methods"
fi

###############################################################################
# Test 8: All Event Types
###############################################################################
log_info "Test 8: Testing all event types..."

TEST_PYTHON_TEST8=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve\Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType
    )

    # Test all event types
    event_types = [
        EventType.EXECUTION_STARTED,
        EventType.ROUND_STARTED,
        EventType.ROUND_COMPLETED,
        EventType.PROGRESS_UPDATE,
        EventType.EXECUTION_COMPLETED,
        EventType.ERROR,
        EventType.CONNECTION_ACK,
        EventType.PING,
        EventType.PONG
    ]

    for event_type in event_types:
        event = WebSocketEvent(
            event_type=event_type,
            data={"test": "data"},
            execution_id="exec_test"
        )

        # Validate event type
        assert event.event_type == event_type, f"Event type mismatch for {event_type.value}"

        # Test serialization
        json_str = event.to_json()
        assert len(json_str) > 0, f"Failed to serialize {event_type.value}"

        # Test deserialization
        restored = WebSocketEvent.from_json(json_str)
        assert restored.event_type == event_type, f"Failed to deserialize {event_type.value}"

    print(f"SUCCESS: All {len(event_types)} event types working correctly")
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
    test_pass "All event types"
else
    test_fail "All event types"
fi

###############################################################################
# Test 9: Event Round-Trip Serialization
###############################################################################
log_info "Test 9: Testing event round-trip serialization..."

TEST_PYTHON_TEST9=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType
    )
    import time

    # Create event with complex data
    original_event = WebSocketEvent(
        event_type=EventType.PROGRESS_UPDATE,
        data={
            "round_number": 2,
            "progress": 0.75,
            "status": "Running",
            "metrics": {
                "accuracy": 0.85,
                "time_elapsed": 45.2,
                "evaluations_remaining": 10
            }
        },
        execution_id="exec_test_123"
    )

    # Serialize
    json_str = original_event.to_json()

    # Deserialize
    restored_event = WebSocketEvent.from_json(json_str)

    # Validate all fields match
    assert restored_event.event_type == original_event.event_type, "Event type mismatch"
    assert restored_event.execution_id == original_event.execution_id, "Execution ID mismatch"

    # Validate complex nested data
    assert restored_event.data["round_number"] == original_event.data["round_number"], "round_number mismatch"
    assert restored_event.data["progress"] == original_event.data["progress"], "progress mismatch"
    assert restored_event.data["metrics"]["accuracy"] == original_event.data["metrics"]["accuracy"], "metrics.accuracy mismatch"

    print("SUCCESS: Event round-trip serialization preserving all data")
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
    test_pass "Event round-trip serialization"
else
    test_fail "Event round-trip serialization"
fi

###############################################################################
# Test 10: Performance - Event Serialization Speed
###############################################################################
log_info "Test 10: Testing event serialization performance..."

TEST_PYTHON_TEST10=$(cat <<'EOF'
import sys
import time
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType
    )

    # Create event with typical data
    event = WebSocketEvent(
        event_type=EventType.PROGRESS_UPDATE,
        data={
            "round_number": 1,
            "progress": 0.5,
            "status": "Running",
            "metrics": {f"metric_{i}": i * 0.1 for i in range(50)}
        }
    )

    # Measure serialization speed
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        json_str = event.to_json()
    elapsed = time.time() - start

    # Should be fast (less than 1 second for 1000 iterations)
    assert elapsed < 1.0, f"Serialization too slow: {elapsed}s for {iterations} iterations"

    avg_time = (elapsed / iterations) * 1000  # Convert to milliseconds

    print(f"SUCCESS: Event serialization performance - {avg_time:.3f}ms per event ({iterations} events in {elapsed:.3f}s)")
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
    test_pass "Event serialization performance"
else
    test_fail "Event serialization performance"
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
    log_info "✓ All WebSocket API tests passed!"
    exit 0
fi
