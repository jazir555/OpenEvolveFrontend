"""
Comprehensive tests for the 5 Robustness Components

Tests all components:
1. Execution Sandbox
2. Vision-Language Monitor
3. Live Web Interface
4. System 1 Router
5. Chronicle Memory
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import components
from execution_sandbox import (
    ExecutionSandbox, SandboxConfig, SandboxProvider,
    CodeSafetyChecker, ComplexityClassifier, SecurityPolicy,
    execute_securely
)
from vision_language_monitor import (
    VisionLanguageMonitor, VLMConfig, VLMProvider,
    AnalysisType, ScreenshotCapture, VLMAnalyzer
)
from live_web_interface import (
    ResearchAgent, ResearchQuery, BrowserConfig,
    WebPage, ResearchResult, quick_research
)
from system1_router import (
    System1Router, RouterConfig, ComplexityLevel,
    ModelTier, RouteDecision, classify_complexity
)
from chronicle_memory import (
    ChronicleMemory, ChronicleEvent, EventType,
    Outcome, LoopDetector, create_chronicle
)
from robustness_integration import (
    RobustnessCoordinator, RobustnessConfig,
    get_robustness_layer, execute_secure
)


# =============================================================================
# Test Execution Sandbox
# =============================================================================

class TestExecutionSandbox:
    """Tests for the Execution Sandbox component"""
    
    @pytest.mark.asyncio
    async def test_code_safety_checker(self):
        """Test code safety checking"""
        checker = CodeSafetyChecker()
        
        # Safe code
        safe_code = "print('Hello World')"
        result = checker.check_code(safe_code)
        assert result["is_safe"] is True
        assert result["risk_score"] == 0.0
        
        # Dangerous code
        dangerous_code = "import os; os.system('rm -rf /')"
        result = checker.check_code(dangerous_code)
        assert result["is_safe"] is False
        assert result["risk_score"] > 0
        assert len(result["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_sandbox_initialization(self):
        """Test sandbox initialization"""
        config = SandboxConfig(
            provider=SandboxProvider.DOCKER,
            timeout_seconds=10
        )
        
        sandbox = ExecutionSandbox(config)
        
        # Check that sandbox initialized correctly
        assert sandbox.config == config
        assert sandbox.safety_checker is not None
    
    @pytest.mark.asyncio
    async def test_sandbox_code_execution(self):
        """Test code execution in sandbox"""
        config = SandboxConfig(
            provider=SandboxProvider.DOCKER,
            timeout_seconds=10
        )
        
        # Skip if Docker not available
        import subprocess
        docker_check = subprocess.run(
            ["docker", "--version"],
            capture_output=True
        )
        if docker_check.returncode != 0:
            pytest.skip("Docker not available")
        
        async with ExecutionSandbox(config) as sandbox:
            code = "print('Hello from sandbox')"
            result = await sandbox.execute(code, "python")
            
            assert result.status.value in ["success", "failure", "sandbox_error"]


# =============================================================================
# Test Vision-Language Monitor
# =============================================================================

class TestVisionLanguageMonitor:
    """Tests for the VLM component"""
    
    @pytest.mark.asyncio
    async def test_vlm_config(self):
        """Test VLM configuration"""
        config = VLMConfig(
            provider=VLMProvider.OPENAI,
            model="gpt-4o",
            max_tokens=1000
        )
        
        assert config.provider == VLMProvider.OPENAI
        assert config.model == "gpt-4o"
        assert config.max_tokens == 1000
    
    @pytest.mark.asyncio
    async def test_vlm_initialization(self):
        """Test VLM monitor initialization"""
        config = VLMConfig(provider=VLMProvider.OPENAI)
        monitor = VisionLanguageMonitor(vlm_config=config)
        
        assert monitor.vlm_config == config
        assert monitor.capture is not None
        assert monitor.analyzer is not None
    
    @pytest.mark.asyncio
    async def test_screenshot_config(self):
        """Test screenshot configuration"""
        from vision_language_monitor import ScreenshotConfig
        
        config = ScreenshotConfig(
            full_page=True,
            viewport_only=False,
            wait_time_ms=1000
        )
        
        assert config.full_page is True
        assert config.viewport_only is False
        assert config.wait_time_ms == 1000


# =============================================================================
# Test Live Web Interface
# =============================================================================

class TestLiveWebInterface:
    """Tests for the web research component"""
    
    @pytest.mark.asyncio
    async def test_research_query(self):
        """Test research query creation"""
        query = ResearchQuery(
            query="Z3 solver error",
            target_sources=["https://github.com/Z3Prover/z3/issues"],
            max_results=5,
            depth="standard"
        )
        
        assert query.query == "Z3 solver error"
        assert len(query.target_sources) == 1
        assert query.max_results == 5
    
    @pytest.mark.asyncio
    async def test_web_page_creation(self):
        """Test web page data structure"""
        page = WebPage(
            url="https://example.com",
            title="Test Page",
            content="<html>Test</html>",
            text_content="Test content",
            links=["https://example.com/page2"]
        )
        
        assert page.url == "https://example.com"
        assert page.title == "Test Page"
        assert len(page.links) == 1
        assert page.content_hash is not None
    
    @pytest.mark.asyncio
    async def test_browser_config(self):
        """Test browser configuration"""
        from live_web_interface import BrowserConfig, BrowserEngine
        
        config = BrowserConfig(
            engine=BrowserEngine.PLAYWRIGHT,
            headless=True,
            timeout_ms=30000
        )
        
        assert config.engine == BrowserEngine.PLAYWRIGHT
        assert config.headless is True
        assert config.timeout_ms == 30000


# =============================================================================
# Test System 1 Router
# =============================================================================

class TestSystem1Router:
    """Tests for the routing component"""
    
    @pytest.mark.asyncio
    async def test_complexity_classification(self):
        """Test complexity classification"""
        from system1_router import ComplexityClassifier
        
        classifier = ComplexityClassifier()
        
        # Trivial query
        trivial = "What time is it?"
        complexity, features = classifier.classify(trivial)
        assert complexity == ComplexityLevel.TRIVIAL
        
        # Simple query
        simple = "Fix this typo: recieve -> receive"
        complexity, features = classifier.classify(simple)
        assert complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE]
        
        # Complex query
        complex_query = "Optimize this Z3 solver configuration for performance"
        complexity, features = classifier.classify(complex_query)
        assert complexity in [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.DEEP]
    
    @pytest.mark.asyncio
    async def test_router_config(self):
        """Test router configuration"""
        config = RouterConfig(
            trivial_word_count=5,
            simple_word_count=30,
            cost_fast=0.00025
        )
        
        assert config.trivial_word_count == 5
        assert config.simple_word_count == 30
        assert config.cost_fast == 0.00025
    
    @pytest.mark.asyncio
    async def test_route_decision(self):
        """Test routing decision"""
        router = System1Router()
        
        decision = await router.route("What is 2+2?")
        
        assert decision.request_id is not None
        assert decision.complexity in ComplexityLevel
        assert decision.model_tier in ModelTier
        assert decision.selected_model is not None
        assert decision.estimated_latency_ms > 0
        assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_model_registry(self):
        """Test model registry"""
        from system1_router import ModelRegistry
        
        # Get model by name
        model = ModelRegistry.get_model("claude-haiku")
        assert model is not None
        assert model.tier == ModelTier.FAST
        
        # Get models by tier
        fast_models = ModelRegistry.get_by_tier(ModelTier.FAST)
        assert len(fast_models) > 0


# =============================================================================
# Test Chronicle Memory
# =============================================================================

class TestChronicleMemory:
    """Tests for the memory component"""
    
    @pytest.mark.asyncio
    async def test_event_creation(self):
        """Test chronicle event creation"""
        event = ChronicleEvent(
            event_id="test-123",
            event_type=EventType.ACTION_STARTED,
            timestamp=datetime.utcnow(),
            agent_id="agent-1",
            session_id="session-1",
            action="test_action",
            parameters={"key": "value"},
            outcome=Outcome.SUCCESS,
            narrative="Test action started"
        )
        
        assert event.event_id == "test-123"
        assert event.event_type == EventType.ACTION_STARTED
        assert event.agent_id == "agent-1"
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test-123"
        assert event_dict["event_type"] == "action_started"
    
    @pytest.mark.asyncio
    async def test_loop_detector(self):
        """Test loop detection"""
        detector = LoopDetector()
        
        # Create some events
        events = [
            ChronicleEvent(
                event_id=f"evt-{i}",
                event_type=EventType.ACTION_STARTED,
                timestamp=datetime.utcnow(),
                agent_id="agent-1",
                session_id="session-1",
                action="strategy_A",
                parameters={"approach": "quick_fix"},
                outcome=Outcome.FAILURE
            )
            for i in range(3)
        ]
        
        # Check for similar attempt
        is_similar, prev_event = detector.is_similar_attempt(
            "strategy_A", {"approach": "quick_fix"}, events
        )
        
        assert is_similar is True
        assert prev_event is not None
    
    @pytest.mark.asyncio
    async def test_chronicle_actions(self):
        """Test chronicle action recording"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chronicle = await create_chronicle(storage_path=tmpdir)
            chronicle.set_agent("test-agent")
            
            # Start action
            event = await chronicle.start_action(
                "test_action",
                {"param": "value"},
                "Testing action"
            )
            
            assert event.action == "test_action"
            assert event.agent_id == "test-agent"
            
            # Complete action
            await chronicle.complete_action(
                outcome=Outcome.SUCCESS,
                result={"status": "ok"},
                lesson="Test completed successfully"
            )
    
    @pytest.mark.asyncio
    async def test_experience_summary(self):
        """Test experience summary generation"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chronicle = await create_chronicle(storage_path=tmpdir)
            chronicle.set_agent("test-agent")
            
            # Record some events
            for i in range(3):
                await chronicle.start_action(f"action_{i}", {}, f"Action {i}")
                await chronicle.complete_action(outcome=Outcome.SUCCESS)
            
            # Get summary
            summary = await chronicle.get_experience_summary()
            
            assert summary["total_events"] >= 3
            assert summary["successes"] >= 3


# =============================================================================
# Test Integration Layer
# =============================================================================

class TestRobustnessIntegration:
    """Tests for the integration layer"""
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        config = RobustnessConfig(
            enable_sandbox=False,  # Skip for test
            enable_vlm=False,
            enable_web_research=False,
            enable_router=True,
            enable_chronicle=False
        )
        
        coordinator = RobustnessCoordinator(config)
        await coordinator.initialize()
        
        assert coordinator._initialized is True
        assert coordinator.router is not None
        
        await coordinator.close()
    
    @pytest.mark.asyncio
    async def test_coordinator_config(self):
        """Test coordinator configuration"""
        config = RobustnessConfig(
            sandbox_provider=SandboxProvider.DOCKER,
            vlm_provider=VLMProvider.OPENAI,
            enable_sandbox=True
        )
        
        assert config.sandbox_provider == SandboxProvider.DOCKER
        assert config.vlm_provider == VLMProvider.OPENAI
        assert config.enable_sandbox is True
    
    @pytest.mark.asyncio
    async def test_loop_prevention_integration(self):
        """Test loop prevention through integration layer"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RobustnessConfig(
                enable_sandbox=False,
                enable_vlm=False,
                enable_web_research=False,
                enable_router=False,
                enable_chronicle=True,
                chronicle_storage_path=tmpdir
            )
            
            coordinator = RobustnessCoordinator(config)
            await coordinator.initialize()
            
            # Record multiple attempts
            for i in range(3):
                check = await coordinator.check_for_loops(
                    "strategy_A", {"approach": "test"}, "test-agent"
                )
                
                if i < 2:
                    assert check["should_prevent"] is False
                    await coordinator.record_attempt("strategy_A", {"approach": "test"}, "test-agent")
                    await coordinator.complete_attempt(success=False)
                else:
                    # Third attempt should trigger loop detection
                    assert check["should_prevent"] is True
            
            await coordinator.close()


# =============================================================================
# Test Feature Interactions
# =============================================================================

class TestFeatureInteractions:
    """Tests for interactions between components"""
    
    @pytest.mark.asyncio
    async def test_router_to_chronicle(self):
        """Test router recording to chronicle"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            router = System1Router()
            chronicle = await create_chronicle(storage_path=tmpdir)
            chronicle.set_agent("router-agent")
            
            # Route a request
            decision = await router.route("Test query")
            
            # Record in chronicle
            await chronicle.record_event(
                EventType.MODEL_ROUTED,
                "route_request",
                {"complexity": decision.complexity.value},
                outcome=Outcome.SUCCESS,
                narrative=f"Routed to {decision.model_tier.value}"
            )
            
            # Verify recording
            events = await chronicle.store.get_agent_events("router-agent")
            assert len(events) > 0
    
    @pytest.mark.asyncio
    async def test_sandbox_to_chronicle(self):
        """Test sandbox execution recording to chronicle"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chronicle = await create_chronicle(storage_path=tmpdir)
            chronicle.set_agent("sandbox-agent")
            
            # Simulate sandbox execution
            await chronicle.start_action("code_execution", {"language": "python"})
            await chronicle.complete_action(
                outcome=Outcome.SUCCESS,
                result={"exit_code": 0},
                lesson="Code executed safely"
            )
            
            # Check experience
            summary = await chronicle.get_experience_summary(
                action_type="code_execution"
            )
            
            assert summary["successes"] > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
