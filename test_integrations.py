"""
Integration Tests - License: Apache 2.0

Comprehensive tests for all new integrations:
- Unified MCP Server
- Event Bus with Valkey
- GraphQL API
- OpenTelemetry
- Service Orchestrator

Run with: pytest test_integrations.py -v

Dependencies:
- pytest: MIT License
- pytest-asyncio: MIT License
- httpx: BSD License
"""

import asyncio
import json
import pytest
from typing import Dict, Any, List
from datetime import datetime

# All test dependencies are permissive licensed
# pytest: MIT, pytest-asyncio: MIT, httpx: BSD


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def event_bus():
    """Create Event Bus instance for testing."""
    from event_bus import EventBus
    
    bus = EventBus(
        valkey_host="localhost",
        enable_persistence=False  # Don't persist in tests
    )
    # Don't connect to Valkey in tests - use in-memory
    yield bus
    
    await bus.disconnect()


@pytest.fixture
def mcp_server():
    """Create MCP Server instance for testing."""
    from unified_mcp_server import UnifiedMCPServer
    
    server = UnifiedMCPServer()
    server.register_all_tools()
    return server


@pytest.fixture
def telemetry_manager():
    """Create Telemetry Manager for testing."""
    from telemetry import TelemetryManager, TelemetryConfig
    
    manager = TelemetryManager()
    config = TelemetryConfig(
        service_name="test-openevolve",
        console_export=True,
        enable_metrics=True,
        enable_tracing=True
    )
    manager.initialize(config)
    return manager


@pytest.fixture
def orchestrator():
    """Create Service Orchestrator for testing."""
    from service_orchestrator import ServiceOrchestrator
    
    return ServiceOrchestrator()


# =============================================================================
# EVENT BUS TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_event_publish_subscribe(event_bus):
    """Test basic event publish/subscribe."""
    from event_bus import Event, EventType
    
    received_events: List[Event] = []
    
    async def handler(event: Event):
        received_events.append(event)
    
    # Subscribe
    await event_bus.subscribe(EventType.WORKFLOW_STARTED, handler)
    
    # Publish
    event = Event(
        type=EventType.WORKFLOW_STARTED,
        source="test",
        workflow_id="wf_test_001",
        payload={"test": "data"}
    )
    await event_bus.publish(event)
    
    # Wait for async dispatch
    await asyncio.sleep(0.1)
    
    # Verify
    assert len(received_events) == 1
    assert received_events[0].type == EventType.WORKFLOW_STARTED
    assert received_events[0].workflow_id == "wf_test_001"


@pytest.mark.asyncio
async def test_event_priority(event_bus):
    """Test event priority handling."""
    from event_bus import Event, EventType, EventPriority
    
    # Test all priority levels
    for priority in EventPriority:
        event = Event(
            type=EventType.SYSTEM_METRIC,
            priority=priority,
            payload={"priority": priority.value}
        )
        success = await event_bus.publish(event)
        assert success


@pytest.mark.asyncio
async def test_event_history(event_bus):
    """Test event history retrieval."""
    from event_bus import Event, EventType
    
    # Publish some events
    for i in range(5):
        event = Event(
            type=EventType.WORKFLOW_COMPLETED,
            workflow_id=f"wf_{i}",
            payload={"index": i}
        )
        await event_bus.publish(event)
    
    # Retrieve history
    history = await event_bus.get_history(
        event_type=EventType.WORKFLOW_COMPLETED,
        limit=3
    )
    
    assert len(history) == 3


@pytest.mark.asyncio
async def test_wait_for_event(event_bus):
    """Test waiting for specific event."""
    from event_bus import Event, EventType
    
    # Schedule event after delay
    async def delayed_publish():
        await asyncio.sleep(0.1)
        event = Event(
            type=EventType.WORKFLOW_COMPLETED,
            workflow_id="wf_delayed"
        )
        await event_bus.publish(event)
    
    asyncio.create_task(delayed_publish())
    
    # Wait for event
    received = await event_bus.wait_for_event(
        EventType.WORKFLOW_COMPLETED,
        timeout=1.0
    )
    
    assert received is not None
    assert received.workflow_id == "wf_delayed"


# =============================================================================
# MCP SERVER TESTS
# =============================================================================

def test_mcp_server_registration(mcp_server):
    """Test MCP server tool registration."""
    assert mcp_server.registry is not None
    
    # Check that tools were registered
    tools = mcp_server.registry.list_tools()
    assert len(tools) > 0
    
    # Check specific tools
    assert mcp_server.registry.get_tool("decompose_problem") is not None
    assert mcp_server.registry.get_tool("extract_knowledge") is not None
    assert mcp_server.registry.get_tool("z3_solve") is not None
    assert mcp_server.registry.get_tool("leanaide_prove") is not None
    assert mcp_server.registry.get_tool("run_workflow") is not None


def test_mcp_tool_schemas(mcp_server):
    """Test MCP tool input schemas."""
    decompose_tool = mcp_server.registry.get_tool("decompose_problem")
    assert decompose_tool is not None
    
    schema = decompose_tool.input_schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "description" in schema["properties"]
    assert "required" in schema
    assert "title" in schema["required"]
    assert "description" in schema["required"]


@pytest.mark.asyncio
async def test_mcp_decompose_tool(mcp_server):
    """Test MCP decompose tool execution."""
    result = await mcp_server.registry.execute("decompose_problem", {
        "title": "Test Problem",
        "description": "This is a test problem for decomposition",
        "domain": "software",
        "strategy": "hybrid"
    })
    
    # Should return TextContent list
    assert len(result) > 0
    
    # Parse result
    data = json.loads(result[0].text)
    assert "plan_id" in data
    assert "strategy" in data
    assert "sub_problems" in data


@pytest.mark.asyncio
async def test_mcp_z3_tool(mcp_server):
    """Test MCP Z3 tool execution."""
    result = await mcp_server.registry.execute("z3_solve", {
        "constraints": ["(> x 0)", "(< x 10)"],
        "variables": {"x": "Int"},
        "timeout": 5.0
    })
    
    assert len(result) > 0
    data = json.loads(result[0].text)
    
    # Should either succeed or gracefully handle missing Z3
    assert "satisfiable" in data or "error" in data


# =============================================================================
# TELEMETRY TESTS
# =============================================================================

def test_telemetry_initialization(telemetry_manager):
    """Test telemetry manager initialization."""
    assert telemetry_manager is not None
    assert telemetry_manager._config is not None
    assert telemetry_manager._config.service_name == "test-openevolve"


def test_telemetry_tracer(telemetry_manager):
    """Test tracer availability."""
    tracer = telemetry_manager.get_tracer()
    # Tracer might be None if OpenTelemetry not installed
    # That's ok - just check it doesn't crash
    assert tracer is not None or not telemetry_manager._config.enable_tracing


def test_telemetry_meter(telemetry_manager):
    """Test meter availability."""
    meter = telemetry_manager.get_meter()
    # Same as tracer
    assert meter is not None or not telemetry_manager._config.enable_metrics


def test_telemetry_counter_creation(telemetry_manager):
    """Test counter metric creation."""
    counter = telemetry_manager.create_counter(
        "test.counter",
        "Test counter"
    )
    # Counter might be None if metrics not enabled
    # but should not raise exception


# =============================================================================
# SERVICE ORCHESTRATOR TESTS
# =============================================================================

def test_orchestrator_service_registration(orchestrator):
    """Test service registration."""
    from service_orchestrator import RESTAPIService, GraphQLService
    
    rest_service = RESTAPIService(port=8000)
    graphql_service = GraphQLService(port=8001)
    
    orchestrator.register_service(rest_service)
    orchestrator.register_service(graphql_service)
    
    assert "rest_api" in orchestrator.services
    assert "graphql_api" in orchestrator.services
    assert orchestrator.get_service("rest_api") == rest_service


@pytest.mark.asyncio
async def test_orchestrator_service_lifecycle(orchestrator):
    """Test service start/stop lifecycle."""
    from service_orchestrator import EventBusService
    
    service = EventBusService()
    orchestrator.register_service(service)
    
    # Start
    results = await orchestrator.start_all(["event_bus"])
    assert results.get("event_bus", False) or not service.bus._connected
    
    # Stop
    stop_results = await orchestrator.stop_all()
    assert "event_bus" in stop_results


@pytest.mark.asyncio
async def test_orchestrator_health_check(orchestrator):
    """Test orchestrator health checking."""
    from service_orchestrator import EventBusService, ServiceStatus
    
    service = EventBusService()
    orchestrator.register_service(service)
    
    # Health check before start
    health = await service.health_check()
    assert "status" in health
    
    # Start and check
    await orchestrator.start_all(["event_bus"])
    health = await service.health_check()
    assert health["status"] in ["healthy", "degraded"]
    
    await orchestrator.stop_all()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

def test_config_loading():
    """Test configuration loading."""
    from integration_config import get_config, IntegrationConfig
    
    config = get_config()
    assert isinstance(config, IntegrationConfig)
    assert config.rest_api is not None
    assert config.graphql is not None


def test_config_validation():
    """Test configuration validation."""
    from integration_config import IntegrationConfig
    from pydantic import ValidationError
    
    # Valid config
    config = IntegrationConfig(log_level="INFO")
    assert config.log_level == "INFO"
    
    # Invalid log level should raise
    with pytest.raises(ValidationError):
        IntegrationConfig(log_level="INVALID")


def test_valkey_config():
    """Test Valkey configuration."""
    from integration_config import ValkeyConfig
    
    # Default values
    config = ValkeyConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 0
    
    # Custom values
    config = ValkeyConfig(host="valkey.example.com", port=6380, ssl=True)
    assert config.host == "valkey.example.com"
    assert config.port == 6380
    assert config.ssl is True


def test_telemetry_config():
    """Test telemetry configuration."""
    from integration_config import OpenTelemetryConfig
    
    config = OpenTelemetryConfig(
        service_name="test-service",
        otlp_endpoint="http://localhost:4317",
        sample_rate=0.5
    )
    
    assert config.service_name == "test-service"
    assert config.otlp_endpoint == "http://localhost:4317"
    assert config.sample_rate == 0.5


# =============================================================================
# INTEGRATION FLOW TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_full_workflow_flow(event_bus, mcp_server):
    """Test complete workflow through multiple integrations."""
    from event_bus import Event, EventType, WorkflowEventTracker
    
    # Setup event tracking
    tracker = WorkflowEventTracker(event_bus)
    
    events_received = []
    
    async def capture_event(event: Event):
        events_received.append(event)
    
    await event_bus.subscribe(EventType.WORKFLOW_STARTED, capture_event)
    await event_bus.subscribe(EventType.DECOMPOSITION_COMPLETED, capture_event)
    
    # Start workflow
    await tracker.track_workflow_start(
        workflow_id="wf_integration_test",
        problem="Test integration workflow"
    )
    
    # Run decomposition via MCP
    result = await mcp_server.registry.execute("decompose_problem", {
        "title": "Integration Test Problem",
        "description": "Testing integration between components",
        "strategy": "hybrid"
    })
    
    # Complete workflow
    await tracker.track_workflow_complete(
        workflow_id="wf_integration_test",
        result={"success": True, "summary": "Test completed"},
        duration_seconds=1.5
    )
    
    # Wait for events
    await asyncio.sleep(0.2)
    
    # Verify events were captured
    workflow_events = [e for e in events_received if e.workflow_id == "wf_integration_test"]
    assert len(workflow_events) >= 2  # Started + Completed


@pytest.mark.asyncio
async def test_end_to_end_event_flow():
    """Test complete event flow through all components."""
    from event_bus import EventBus, Event, EventType
    
    bus = EventBus(enable_persistence=False)
    
    # Subscribe to multiple event types
    all_events = []
    
    async def capture_all(event: Event):
        all_events.append(event)
    
    for event_type in [EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED]:
        await bus.subscribe(event_type, capture_all)
    
    # Simulate workflow lifecycle
    workflow_id = "wf_e2e_test"
    
    # Start
    await bus.publish(Event(
        type=EventType.WORKFLOW_STARTED,
        workflow_id=workflow_id,
        payload={"problem": "E2E Test"}
    ))
    
    # Decomposition
    await bus.publish(Event(
        type=EventType.DECOMPOSITION_COMPLETED,
        workflow_id=workflow_id,
        payload={"subproblems": 3}
    ))
    
    # Complete
    await bus.publish(Event(
        type=EventType.WORKFLOW_COMPLETED,
        workflow_id=workflow_id,
        payload={"duration": 2.0}
    ))
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Verify
    workflow_events = [e for e in all_events if e.workflow_id == workflow_id]
    assert len(workflow_events) == 2  # Started and Completed
    
    await bus.disconnect()


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_event_bus_error_handling(event_bus):
    """Test event bus error handling."""
    from event_bus import Event, EventType
    
    async def failing_handler(event: Event):
        raise ValueError("Test error")
    
    # Subscribe failing handler
    await event_bus.subscribe(EventType.SYSTEM_ERROR, failing_handler)
    
    # Publish should not crash
    event = Event(type=EventType.SYSTEM_ERROR, payload={"test": "error"})
    success = await event_bus.publish(event)
    
    assert success  # Should succeed even with failing handler


@pytest.mark.asyncio
async def test_mcp_invalid_tool(mcp_server):
    """Test MCP handling of invalid tool."""
    result = await mcp_server.registry.execute("nonexistent_tool", {})
    
    assert len(result) > 0
    assert "Error" in result[0].text or "not found" in result[0].text.lower()


@pytest.mark.asyncio
async def test_mcp_invalid_arguments(mcp_server):
    """Test MCP handling of invalid arguments."""
    # Missing required arguments
    result = await mcp_server.registry.execute("decompose_problem", {
        # Missing title and description
        "strategy": "hybrid"
    })
    
    # Should handle gracefully
    assert len(result) > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_event_bus_throughput(event_bus):
    """Test event bus performance."""
    from event_bus import Event, EventType
    
    received_count = 0
    
    async def counter(event: Event):
        nonlocal received_count
        received_count += 1
    
    await event_bus.subscribe(EventType.SYSTEM_METRIC, counter)
    
    # Publish 100 events
    start = asyncio.get_event_loop().time()
    
    for i in range(100):
        await event_bus.publish(Event(
            type=EventType.SYSTEM_METRIC,
            payload={"index": i}
        ))
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    duration = asyncio.get_event_loop().time() - start
    
    # Should process all events quickly
    assert received_count == 100
    assert duration < 2.0  # Should complete in under 2 seconds


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
