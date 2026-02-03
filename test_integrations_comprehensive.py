"""
Comprehensive Integration Tests - License: Apache 2.0

Complete test suite for all OpenEvolve integration components.
Tests REST API, GraphQL, Event Bus, MCP Server, Telemetry, and Stage 6.

Run with: pytest test_integrations_comprehensive.py -v
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Test all integration components
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import strawberry
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for tests."""
    return {
        "workflow_id": "wf_test_001",
        "problem_description": "Test optimization problem",
        "strategy": "hybrid",
        "stages": ["decomposition", "evolution", "assembly"]
    }


@pytest.fixture
def sample_execution_trace():
    """Sample execution trace."""
    from stage6_knowledge_extraction import ExecutionTrace
    return ExecutionTrace(
        trace_id="trace_001",
        workflow_id="wf_001",
        problem_description="Optimize neural network",
        stages=[
            {"stage_name": "decomposition", "parameters": {"strategy": "hybrid"}},
            {"stage_name": "evolution", "parameters": {"generations": 50}}
        ],
        final_result={"accuracy": 0.95},
        execution_time_ms=1000.0,
        timestamp=datetime.now()
    )


# =============================================================================
# STAGE 6 KNOWLEDGE EXTRACTION TESTS
# =============================================================================

class TestStage6KnowledgeExtraction:
    """Test Stage 6 Knowledge Extraction Engine."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_dir):
        """Test engine initialization."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        
        assert engine.storage_path == temp_dir
        assert temp_dir.exists()
        assert len(engine.patterns) == 0
        assert len(engine.artifacts) == 0
    
    @pytest.mark.asyncio
    async def test_process_single_trace(self, temp_dir, sample_execution_trace):
        """Test processing a single trace."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        result = await engine.process_trace(sample_execution_trace)
        
        assert isinstance(result, dict)
        assert 'patterns_extracted' in result
        assert 'artifacts_generated' in result
        assert len(engine.traces) == 1
    
    @pytest.mark.asyncio
    async def test_process_multiple_traces(self, temp_dir, sample_execution_trace):
        """Test processing multiple traces."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        
        # Process 5 traces (minimum for pattern extraction)
        for i in range(5):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                workflow_id=f"wf_{i}",
                problem_description=f"Problem {i}",
                stages=[
                    {"stage_name": "decomposition", "parameters": {"strategy": "hybrid"}},
                    {"stage_name": "evolution", "parameters": {"generations": 50}}
                ],
                final_result={"accuracy": 0.9 + i * 0.01},
                execution_time_ms=1000.0,
                timestamp=datetime.now()
            )
            await engine.process_trace(trace)
        
        stats = engine.get_statistics()
        assert stats['traces_processed'] == 5
    
    def test_get_applicable_artifacts(self, temp_dir):
        """Test artifact retrieval."""
        from stage6_knowledge_extraction import (
            Stage6KnowledgeExtraction, KnowledgeArtifact
        )
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        
        # Add test artifact
        artifact = KnowledgeArtifact(
            artifact_id="test_001",
            name="Neural Network Strategy",
            artifact_type="strategy",
            content={},
            source_workflows=["wf_001"],
            extraction_date=datetime.now(),
            validity_score=0.9,
            tags=["neural-network", "optimization"]
        )
        engine.artifacts["test_001"] = artifact
        
        # Query
        results = engine.get_applicable_artifacts("neural network optimization")
        assert len(results) > 0
        assert results[0].artifact_id == "test_001"
    
    def test_statistics(self, temp_dir):
        """Test statistics calculation."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        stats = engine.get_statistics()
        
        assert 'traces_processed' in stats
        assert 'patterns_extracted' in stats
        assert 'artifacts_generated' in stats
        assert 'avg_pattern_confidence' in stats


# =============================================================================
# PATTERN EXTRACTOR TESTS
# =============================================================================

class TestPatternExtractor:
    """Test pattern extraction components."""
    
    def test_sequence_pattern_extraction(self):
        """Test sequence pattern extraction."""
        from stage6_knowledge_extraction import (
            PatternExtractor, ExecutionTrace
        )
        
        extractor = PatternExtractor(min_confidence=0.5)
        
        traces = [
            ExecutionTrace(
                trace_id=f"t{i}",
                workflow_id=f"w{i}",
                problem_description=f"Problem {i}",
                stages=[
                    {"stage_name": "stage_a", "parameters": {}},
                    {"stage_name": "stage_b", "parameters": {}},
                    {"stage_name": "stage_c", "parameters": {}}
                ],
                final_result={},
                execution_time_ms=100.0,
                timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        patterns = extractor.extract_sequence_patterns(traces)
        
        # Should find the common sequence
        assert len(patterns) > 0
        assert patterns[0].pattern_type == 'sequence'
        assert patterns[0].occurrences == 5
    
    def test_parametric_pattern_extraction(self):
        """Test parametric pattern extraction."""
        from stage6_knowledge_extraction import (
            PatternExtractor, ExecutionTrace
        )
        
        extractor = PatternExtractor(min_confidence=0.5)
        
        traces = [
            ExecutionTrace(
                trace_id=f"t{i}",
                workflow_id=f"w{i}",
                problem_description="Test",
                stages=[
                    {"stage_name": "test", "parameters": {"param": "value_a"}}
                ],
                final_result={},
                execution_time_ms=100.0,
                timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        patterns = extractor.extract_parametric_patterns(traces)
        
        # Should find the common parameter value
        assert len(patterns) > 0
        assert patterns[0].pattern_type == 'parametric'


# =============================================================================
# SERVICE ORCHESTRATOR TESTS
# =============================================================================

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestServiceOrchestrator:
    """Test service orchestration."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        from service_orchestrator import ServiceOrchestrator
        
        orchestrator = ServiceOrchestrator()
        assert orchestrator is not None
        assert len(orchestrator.services) == 0
    
    @pytest.mark.asyncio
    async def test_service_registration(self):
        """Test service registration."""
        from service_orchestrator import ServiceOrchestrator
        
        orchestrator = ServiceOrchestrator()
        
        mock_service = Mock()
        mock_service.start = AsyncMock(return_value=True)
        mock_service.stop = AsyncMock(return_value=True)
        
        orchestrator.register_service(
            name="test_service",
            start_func=mock_service.start,
            stop_func=mock_service.stop,
            dependencies=[]
        )
        
        assert "test_service" in orchestrator.services


# =============================================================================
# EVENT BUS TESTS
# =============================================================================

class TestEventBus:
    """Test event bus functionality."""
    
    @pytest.mark.asyncio
    async def test_event_creation(self):
        """Test event creation."""
        from event_bus import WorkflowEvent, EventType
        
        event = WorkflowEvent(
            id="evt_001",
            type=EventType.WORKFLOW_STARTED,
            payload={"workflow_id": "wf_001"},
            timestamp=datetime.now(),
            priority=1
        )
        
        assert event.id == "evt_001"
        assert event.type == EventType.WORKFLOW_STARTED
    
    @pytest.mark.asyncio
    async def test_in_memory_event_bus(self):
        """Test in-memory event bus."""
        from event_bus import InMemoryEventBus, WorkflowEvent, EventType
        
        bus = InMemoryEventBus()
        await bus.connect()
        
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        await bus.subscribe("test_channel", handler)
        
        event = WorkflowEvent(
            id="evt_001",
            type=EventType.WORKFLOW_STARTED,
            payload={"test": "data"},
            timestamp=datetime.now(),
            priority=1
        )
        
        await bus.publish("test_channel", event)
        
        # Give async handler time to process
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].id == "evt_001"
        
        await bus.disconnect()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration."""
        from integration_config import IntegrationConfig
        
        config = IntegrationConfig()
        
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert config.orchestrator_port > 0
        assert isinstance(config.services, dict)
    
    def test_config_validation(self):
        """Test configuration validation."""
        from integration_config import IntegrationConfig, MCPConfig
        
        # Valid config
        config = IntegrationConfig(
            mcp=MCPConfig(transport="stdio")
        )
        assert config.mcp.transport == "stdio"


# =============================================================================
# PLUGIN REGISTRY TESTS
# =============================================================================

class TestPluginRegistry:
    """Test plugin registry."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        from plugin_registry import PluginRegistry
        
        registry = PluginRegistry()
        assert registry is not None
        assert len(registry._plugins) == 0
    
    def test_plugin_registration(self):
        """Test plugin registration."""
        from plugin_registry import PluginRegistry, PluginMetadata, PluginType
        
        registry = PluginRegistry()
        
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test",
            license="Apache-2.0",
            plugin_type=PluginType.MCP_TOOL,
            capabilities=[]
        )
        
        # Create mock plugin
        mock_plugin = Mock()
        mock_plugin.metadata = metadata
        
        registry._plugins["test_plugin"] = mock_plugin
        
        assert "test_plugin" in registry._plugins


# =============================================================================
# API GATEWAY TESTS
# =============================================================================

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIGateway:
    """Test API Gateway."""
    
    def test_gateway_creation(self):
        """Test gateway creation."""
        from api_gateway import APIGateway, GatewayConfig
        
        config = GatewayConfig(
            host="127.0.0.1",
            port=8080
        )
        
        gateway = APIGateway(config)
        
        assert gateway.config == config
        assert gateway.app is not None
    
    def test_gateway_routes(self):
        """Test gateway routes."""
        from api_gateway import APIGateway
        
        gateway = APIGateway()
        
        # Test with TestClient
        client = TestClient(gateway.app)
        
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "OpenEvolve" in data["service"]
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        from api_gateway import APIGateway
        
        gateway = APIGateway()
        client = TestClient(gateway.app)
        
        response = client.get("/health")
        assert response.status_code in [200, 503]  # 503 if services down
        
        data = response.json()
        assert "status" in data
        assert "services" in data


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Test rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        from api_gateway import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=2, burst_size=1)
        
        # First two requests should be allowed
        assert await limiter.is_allowed("client_1") is True
        assert await limiter.is_allowed("client_1") is True
        
        # Third request should be blocked (rate limited)
        assert await limiter.is_allowed("client_1") is False
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_client(self):
        """Test rate limiting is per-client."""
        from api_gateway import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=2, burst_size=1)
        
        # Each client has own limit
        assert await limiter.is_allowed("client_1") is True
        assert await limiter.is_allowed("client_2") is True
        assert await limiter.is_allowed("client_1") is True
        assert await limiter.is_allowed("client_2") is True


# =============================================================================
# TELEMETRY TESTS
# =============================================================================

class TestTelemetry:
    """Test telemetry components."""
    
    def test_telemetry_config(self):
        """Test telemetry configuration."""
        from telemetry import TelemetryConfig
        
        config = TelemetryConfig(
            service_name="test_service",
            enabled=True
        )
        
        assert config.service_name == "test_service"
        assert config.enabled is True
    
    def test_workflow_tracer(self):
        """Test workflow tracer."""
        from telemetry import WorkflowTracer, TelemetryConfig
        
        config = TelemetryConfig(enabled=False)  # Disabled for test
        tracer = WorkflowTracer(config)
        
        assert tracer is not None
        assert tracer.config == config


# =============================================================================
# INTEGRATION END-TO-END TESTS
# =============================================================================

@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_lifecycle(self, temp_dir, sample_execution_trace):
        """Test complete workflow lifecycle."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        from event_bus import InMemoryEventBus
        
        # Setup components
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        event_bus = InMemoryEventBus()
        await event_bus.connect()
        
        # Process workflow
        result = await engine.process_trace(sample_execution_trace)
        
        # Verify results
        assert result['traces_processed'] == 1
        
        stats = engine.get_statistics()
        assert stats['traces_processed'] == 1
        
        await event_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_workflows_pattern_extraction(self, temp_dir):
        """Test pattern extraction across multiple workflows."""
        from stage6_knowledge_extraction import (
            Stage6KnowledgeExtraction, ExecutionTrace
        )
        
        engine = Stage6KnowledgeExtraction(storage_path=temp_dir)
        
        # Create 10 similar workflows
        for i in range(10):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                workflow_id=f"wf_{i}",
                problem_description=f"Optimization problem type A",
                stages=[
                    {"stage_name": "decompose", "parameters": {"strategy": "hybrid"}},
                    {"stage_name": "evolve", "parameters": {"generations": 100}},
                    {"stage_name": "assemble", "parameters": {}}
                ],
                final_result={"fitness": 0.95},
                execution_time_ms=5000.0,
                timestamp=datetime.now()
            )
            await engine.process_trace(trace)
        
        # Verify patterns were extracted
        stats = engine.get_statistics()
        assert stats['traces_processed'] == 10
        
        # Should have extracted sequence patterns
        assert stats['patterns_extracted'] > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.slow
    def test_pattern_extraction_performance(self):
        """Test pattern extraction performance."""
        import time
        from stage6_knowledge_extraction import (
            PatternExtractor, ExecutionTrace
        )
        
        extractor = PatternExtractor()
        
        # Create 100 traces
        traces = [
            ExecutionTrace(
                trace_id=f"t{i}",
                workflow_id=f"w{i}",
                problem_description=f"Problem {i % 10}",
                stages=[{"stage_name": f"stage_{i % 5}", "parameters": {}}],
                final_result={"result": i},
                execution_time_ms=100.0,
                timestamp=datetime.now()
            )
            for i in range(100)
        ]
        
        start = time.time()
        patterns = extractor.extract_sequence_patterns(traces)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds
        assert len(patterns) > 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
