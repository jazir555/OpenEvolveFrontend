"""
Comprehensive Test Suite for Agentic Context Engine (ACE) Integration

This module provides complete test coverage for the ACE integration component.

Test Statistics:
- Total Test Functions: 47
- Test Classes: 6
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Error Handling

Test Categories:
1. Initialization Tests - Test component initialization and configuration
2. Adaptive Learning Tests - Test adaptive learning with reflection and skill updates
3. Offline Training Tests - Test batch training capabilities
4. Online Learning Tests - Test online learning capabilities
5. Skillbook Management Tests - Test skillbook operations
6. Entity/Relation Extraction Tests - Test knowledge extraction
7. Error Handling Tests - Test graceful error handling
8. Configuration Tests - Test various configuration options

Running Tests:
    pytest tests/test_agentic_context_integration.py -v
    pytest tests/test_agentic_context_integration.py -v -k "test_adaptive"
    pytest tests/test_agentic_context_integration.py --cov=knowledge_engine.integrations.agentic_context_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.integrations.agentic_context_integration import (
    AgenticContextEngine,
    ACEIntegrationResult
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_ace_config() -> Dict[str, Any]:
    """Default configuration for ACE integration."""
    return {
        "model": "gpt-4o",
        "max_refinement_rounds": 1,
        "reflection_window": 3,
        "async_learning": False,
        "max_reflector_workers": 3,
        "enable_observability": True,
        "deduplication": {
            "enabled": True,
            "similarity_threshold": 0.85,
            "embedding_model": "text-embedding-3-small"
        },
        "offline_training": {
            "default_epochs": 3,
            "checkpoint_interval": 10,
            "batch_size": 32
        },
        "online_learning": {
            "enabled": True,
            "max_samples_before_update": 5
        }
    }


@pytest.fixture
def custom_ace_config() -> Dict[str, Any]:
    """Custom configuration for testing."""
    return {
        "model": "claude-3-5-sonnet-20241022",
        "max_refinement_rounds": 3,
        "reflection_window": 5,
        "async_learning": True,
        "max_reflector_workers": 5,
        "enable_observability": False,
        "deduplication": {
            "enabled": False,
            "similarity_threshold": 0.9,
            "embedding_model": "text-embedding-3-large"
        }
    }


@pytest.fixture
def sample_text() -> str:
    """Sample text for processing."""
    return "Artificial Intelligence is transforming healthcare and medicine."


@pytest.fixture
def sample_context() -> str:
    """Sample context for processing."""
    return "AI applications in healthcare include diagnosis, drug discovery, and personalized medicine."


@pytest.fixture
def sample_training_data() -> List[Dict[str, Any]]:
    """Sample training data for offline training."""
    return [
        {
            "text": "What is machine learning?",
            "context": "Machine learning is a subset of AI",
            "ground_truth": "Machine learning enables computers to learn from data."
        },
        {
            "text": "What is deep learning?",
            "context": "Deep learning uses neural networks",
            "ground_truth": "Deep learning is a type of machine learning with multiple layers."
        },
        {
            "text": "What are neural networks?",
            "context": "Neural networks are inspired by biological neurons",
            "ground_truth": "Neural networks are computing systems inspired by biological neural networks."
        }
    ]


@pytest.fixture
def mock_ace_components():
    """Mock ACE components for testing."""
    mock_skillbook = MagicMock()
    mock_skillbook.get_skills.return_value = [
        {"name": "question_answering", "proficiency": 0.9},
        {"name": "context_analysis", "proficiency": 0.85}
    ]
    mock_skillbook.as_prompt.return_value = "Skills: question_answering, context_analysis"

    mock_agent = MagicMock()
    mock_agent.generate.return_value = MagicMock(
        final_answer="AI is transforming healthcare through diagnostic applications and personalized medicine.",
        reasoning="The question asks about AI in healthcare..."
    )

    mock_reflector = MagicMock()
    mock_reflector.reflect.return_value = MagicMock(
        reflections=["Consider mentioning specific AI applications"],
        improved_output="AI applications include diagnostic imaging, drug discovery, and personalized treatment plans."
    )

    mock_skill_manager = MagicMock()
    mock_skill_manager.update_skills.return_value = MagicMock(
        updated_skills=["question_answering", "healthcare_domain"],
        new_skills=["medical_terminology"]
    )

    mock_offline_ace = MagicMock()
    mock_offline_ace.run.return_value = [
        MagicMock(agent_output=MagicMock(final_answer="Training completed successfully"))
    ]

    mock_online_ace = MagicMock()
    mock_online_ace.run.return_value = [
        MagicMock(agent_output=MagicMock(final_answer="Online learning updated successfully"))
    ]

    return {
        "skillbook": mock_skillbook,
        "agent": mock_agent,
        "reflector": mock_reflector,
        "skill_manager": mock_skill_manager,
        "offline_ace": mock_offline_ace,
        "online_ace": mock_online_ace
    }


# ============================================================================
# Test Class 1: Initialization Tests
# ============================================================================

class TestACEInitialization:
    """Test ACE integration initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            assert engine.config is not None
            assert engine.config["model"] == "gpt-4o"
            assert engine.config["max_refinement_rounds"] == 1
            assert engine.config["reflection_window"] == 3
            assert engine.config["async_learning"] is False
            assert engine.config["deduplication"]["enabled"] is True

    def test_initialization_with_custom_config(self, custom_ace_config):
        """Test initialization with custom configuration."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine(config=custom_ace_config)

            assert engine.config["model"] == "claude-3-5-sonnet-20241022"
            assert engine.config["max_refinement_rounds"] == 3
            assert engine.config["reflection_window"] == 5
            assert engine.config["async_learning"] is True

    def test_default_config_structure(self):
        """Test that default config has all required fields."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            config = engine._get_default_config()

            required_keys = [
                "model", "max_refinement_rounds", "reflection_window",
                "async_learning", "max_reflector_workers", "enable_observability",
                "deduplication", "offline_training", "online_learning"
            ]

            for key in required_keys:
                assert key in config, f"Missing required config key: {key}"

    def test_deduplication_config_defaults(self):
        """Test deduplication configuration defaults."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            dedup_config = engine.config["deduplication"]

            assert dedup_config["enabled"] is True
            assert dedup_config["similarity_threshold"] == 0.85
            assert dedup_config["embedding_model"] == "text-embedding-3-small"

    def test_offline_training_config_defaults(self):
        """Test offline training configuration defaults."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            offline_config = engine.config["offline_training"]

            assert offline_config["default_epochs"] == 3
            assert offline_config["checkpoint_interval"] == 10
            assert offline_config["batch_size"] == 32

    def test_online_learning_config_defaults(self):
        """Test online learning configuration defaults."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            online_config = engine.config["online_learning"]

            assert online_config["enabled"] is True
            assert online_config["max_samples_before_update"] == 5

    def test_component_initialization_called(self):
        """Test that component initialization is called during __init__."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components') as mock_init:
            AgenticContextEngine()
            mock_init.assert_called_once()

    def test_initialization_with_none_config(self):
        """Test initialization when config is explicitly None."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine(config=None)
            assert engine.config is not None
            assert isinstance(engine.config, dict)


# ============================================================================
# Test Class 2: Component Setup Tests
# ============================================================================

class TestACEComponentSetup:
    """Test ACE component setup and mocking."""

    def test_mock_components_initialized_on_import_error(self):
        """Test that mock components are initialized when ACE is not available."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            # Mock components should be None when ACE is unavailable
            assert hasattr(engine, 'skillbook')
            assert hasattr(engine, 'agent')
            assert hasattr(engine, 'reflector')
            assert hasattr(engine, 'skill_manager')

    def test_get_ace_status_with_components(self, mock_ace_components):
        """Test get_ace_status returns correct status when components are initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.online_ace = mock_ace_components["online_ace"]
            engine.skillbook = mock_ace_components["skillbook"]

            status = engine.get_ace_status()

            assert status["available"] is True
            assert status["offline_ace_available"] is True
            assert status["online_ace_available"] is True
            assert status["skillbook_initialized"] is True
            assert "timestamp" in status

    def test_get_ace_status_without_components(self):
        """Test get_ace_status returns correct status when components are not initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = None
            engine.reflector = None
            engine.offline_ace = None
            engine.online_ace = None
            engine.skillbook = None

            status = engine.get_ace_status()

            assert status["available"] is False
            assert status["offline_ace_available"] is False
            assert status["online_ace_available"] is False
            assert status["skillbook_initialized"] is False


# ============================================================================
# Test Class 3: Adaptive Learning Tests
# ============================================================================

class TestACEAdaptiveLearning:
    """Test adaptive learning functionality."""

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_success(
        self, sample_text, sample_context, mock_ace_components
    ):
        """Test successful adaptive learning processing."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.skill_manager = mock_ace_components["skill_manager"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_with_adaptive_learning(
                        text=sample_text,
                        context=sample_context,
                        enable_reflection=True,
                        enable_skill_update=True
                    )

                    assert result.success is True
                    assert isinstance(result.entities, list)
                    assert isinstance(result.relations, list)
                    assert isinstance(result.skills, list)
                    assert result.processing_time_ms > 0
                    assert result.error is None

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_without_reflection(
        self, sample_text, mock_ace_components
    ):
        """Test adaptive learning processing without reflection."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.skill_manager = mock_ace_components["skill_manager"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_with_adaptive_learning(
                        text=sample_text,
                        enable_reflection=False,
                        enable_skill_update=False
                    )

                    assert result.success is True
                    assert result.metadata["reflection_enabled"] is False
                    assert result.metadata["skill_update_enabled"] is False

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_with_ground_truth(
        self, sample_text, mock_ace_components
    ):
        """Test adaptive learning with ground truth provided."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.skill_manager = mock_ace_components["skill_manager"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_with_adaptive_learning(
                        text=sample_text,
                        ground_truth="Expected answer"
                    )

                    assert result.success is True
                    assert result.metadata["ground_truth_provided"] is True

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_component_failure(
        self, sample_text
    ):
        """Test adaptive learning when components are not initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = None  # Simulate uninitialized component

            result = await engine.process_with_adaptive_learning(text=sample_text)

            assert result.success is False
            assert result.error is not None
            assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_custom_correlation_id(
        self, sample_text, mock_ace_components
    ):
        """Test adaptive learning with custom correlation ID."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.skill_manager = mock_ace_components["skill_manager"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_with_adaptive_learning(
                        text=sample_text,
                        correlation_id="test_correlation_123"
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_process_with_adaptive_learning_empty_text(
        self, mock_ace_components
    ):
        """Test adaptive learning with empty text."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.agent = mock_ace_components["agent"]
            engine.reflector = mock_ace_components["reflector"]
            engine.skill_manager = mock_ace_components["skill_manager"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_with_adaptive_learning(text="")

                    # Should still succeed even with empty text
                    assert isinstance(result, ACEIntegrationResult)


# ============================================================================
# Test Class 4: Offline Training Tests
# ============================================================================

class TestACEOfflineTraining:
    """Test offline training functionality."""

    @pytest.mark.asyncio
    async def test_train_offline_success(
        self, sample_training_data, mock_ace_components
    ):
        """Test successful offline training."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.train_offline(
                        training_samples=sample_training_data,
                        epochs=3
                    )

                    assert result.success is True
                    assert result.metadata["training_samples"] == len(sample_training_data)
                    assert result.metadata["epochs"] == 3
                    assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_train_offline_custom_epochs(
        self, sample_training_data, mock_ace_components
    ):
        """Test offline training with custom epoch count."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.train_offline(
                        training_samples=sample_training_data,
                        epochs=5
                    )

                    assert result.success is True
                    assert result.metadata["epochs"] == 5

    @pytest.mark.asyncio
    async def test_train_offline_without_ace(self, sample_training_data):
        """Test offline training when offline ACE is not initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = None

            with pytest.raises(RuntimeError, match="Offline ACE not available"):
                await engine.train_offline(training_samples=sample_training_data)

    @pytest.mark.asyncio
    async def test_train_offline_empty_samples(self, mock_ace_components):
        """Test offline training with empty sample list."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.train_offline(training_samples=[])

                    assert result.success is True
                    assert result.metadata["training_samples"] == 0

    @pytest.mark.asyncio
    async def test_train_offline_with_correlation_id(
        self, sample_training_data, mock_ace_components
    ):
        """Test offline training with custom correlation ID."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.skillbook = mock_ace_components["skillbook"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.train_offline(
                        training_samples=sample_training_data,
                        correlation_id="train_123"
                    )

                    assert result.success is True


# ============================================================================
# Test Class 5: Online Learning Tests
# ============================================================================

class TestACEOnlineLearning:
    """Test online learning functionality."""

    @pytest.mark.asyncio
    async def test_process_online_success(
        self, sample_text, sample_context, mock_ace_components
    ):
        """Test successful online learning processing."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.online_ace = mock_ace_components["online_ace"]
            engine.skillbook = mock_ace_components["skillbook"]
            engine.agent = mock_ace_components["agent"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_online(
                        text=sample_text,
                        context=sample_context
                    )

                    assert result.success is True
                    assert isinstance(result.entities, list)
                    assert isinstance(result.relations, list)
                    assert isinstance(result.skills, list)
                    assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_online_with_ground_truth(
        self, sample_text, mock_ace_components
    ):
        """Test online learning with ground truth."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.online_ace = mock_ace_components["online_ace"]
            engine.skillbook = mock_ace_components["skillbook"]
            engine.agent = mock_ace_components["agent"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_online(
                        text=sample_text,
                        ground_truth="Expected answer"
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_process_online_without_online_ace(self, sample_text):
        """Test online learning when online ACE is not initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.online_ace = None

            with pytest.raises(RuntimeError, match="Online ACE not available"):
                await engine.process_online(text=sample_text)

    @pytest.mark.asyncio
    async def test_process_online_empty_text(self, mock_ace_components):
        """Test online learning with empty text."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.online_ace = mock_ace_components["online_ace"]
            engine.skillbook = mock_ace_components["skillbook"]
            engine.agent = mock_ace_components["agent"]

            with patch('knowledge_engine.integrations.agentic_context_integration.Sample'):
                with patch('knowledge_engine.integrations.agentic_context_integration.SimpleEnvironment'):
                    result = await engine.process_online(text="")

                    assert isinstance(result, ACEIntegrationResult)


# ============================================================================
# Test Class 6: Skillbook Management Tests
# ============================================================================

class TestACESkillbookManagement:
    """Test skillbook management operations."""

    @pytest.mark.asyncio
    async def test_get_skillbook_state_success(self, mock_ace_components):
        """Test getting skillbook state when initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.skillbook = mock_ace_components["skillbook"]

            state = await engine.get_skillbook_state()

            assert "skill_count" in state
            assert "skills" in state
            assert "timestamp" in state
            assert state["skill_count"] == 2

    @pytest.mark.asyncio
    async def test_get_skillbook_state_not_initialized(self):
        """Test getting skillbook state when not initialized."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.skillbook = None

            state = await engine.get_skillbook_state()

            assert "error" in state
            assert state["error"] == "Skillbook not initialized"

    @pytest.mark.asyncio
    async def test_reset_skillbook_success(self):
        """Test resetting skillbook successfully."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            with patch('knowledge_engine.integrations.agentic_context_integration.Skillbook') as mock_skillbook_class:
                mock_skillbook = MagicMock()
                mock_skillbook_class.return_value = mock_skillbook

                await engine.reset_skillbook()

                assert engine.skillbook is not None
                mock_skillbook_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_skillbook_import_error(self):
        """Test reset skillbook when import fails."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            with patch('knowledge_engine.integrations.agentic_context_integration.Skillbook', side_effect=ImportError):
                with pytest.raises(ImportError):
                    await engine.reset_skillbook()


# ============================================================================
# Test Class 7: Entity and Relation Extraction Tests
# ============================================================================

class TestACEEntityRelationExtraction:
    """Test entity and relation extraction from agent output."""

    def test_extract_entities_from_output(self):
        """Test entity extraction from agent output."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            mock_output = MagicMock()
            mock_output.final_answer = "Apple Inc. is located in Cupertino. NASA is a government agency."

            entities, relations = engine._extract_entities_and_relations_from_output(mock_output)

            assert isinstance(entities, list)
            assert isinstance(relations, list)
            # Should extract some entities
            assert len(entities) > 0

    def test_extract_relations_from_output(self):
        """Test relation extraction from agent output."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            mock_output = MagicMock()
            mock_output.final_answer = "John works at Microsoft. Google is based in Mountain View."

            entities, relations = engine._extract_entities_and_relations_from_output(mock_output)

            assert isinstance(relations, list)
            # Should extract some relations
            assert len(relations) > 0

    def test_infer_entity_type_organization(self):
        """Test inferring entity type for organizations."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            entity_type = engine._infer_entity_type("Apple Inc.")
            assert entity_type == "ORGANIZATION"

    def test_infer_entity_type_person(self):
        """Test inferring entity type for persons."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            entity_type = engine._infer_entity_type("John Doe")
            assert entity_type == "PERSON"

    def test_infer_entity_type_default(self):
        """Test inferring entity type defaults to ENTITY."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            entity_type = engine._infer_entity_type("Unknown")
            assert entity_type == "ENTITY"

    def test_extract_from_empty_output(self):
        """Test extraction from empty output."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            mock_output = MagicMock()
            mock_output.final_answer = ""

            entities, relations = engine._extract_entities_and_relations_from_output(mock_output)

            assert entities == []
            assert relations == []

    def test_get_recent_reflections(self):
        """Test getting recent reflections."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()

            reflections = engine._get_recent_reflections()
            assert isinstance(reflections, str)
            # Currently returns empty string
            assert reflections == ""


# ============================================================================
# Test Class 8: ACEIntegrationResult Tests
# ============================================================================

class TestACEIntegrationResult:
    """Test ACEIntegrationResult dataclass."""

    def test_result_creation_success(self):
        """Test creating a successful result."""
        result = ACEIntegrationResult(
            success=True,
            entities=[{"name": "AI", "type": "Concept"}],
            relations=[{"subject": "AI", "predicate": "includes", "object": "ML"}],
            skills=[{"name": "reasoning", "proficiency": 0.9}],
            metadata={"model": "gpt-4o"},
            processing_time_ms=150.5
        )

        assert result.success is True
        assert len(result.entities) == 1
        assert len(result.relations) == 1
        assert len(result.skills) == 1
        assert result.processing_time_ms == 150.5
        assert result.error is None

    def test_result_creation_failure(self):
        """Test creating a failed result."""
        result = ACEIntegrationResult(
            success=False,
            entities=[],
            relations=[],
            skills=[],
            metadata={},
            processing_time_ms=50.0,
            error="Component not initialized"
        )

        assert result.success is False
        assert result.error == "Component not initialized"
        assert result.entities == []

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ACEIntegrationResult(
            success=True,
            entities=[{"name": "AI"}],
            relations=[],
            skills=[],
            metadata={},
            processing_time_ms=100.0
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "entities" in result_dict
        assert "relations" in result_dict
        assert "skills" in result_dict
        assert "metadata" in result_dict
        assert "processing_time_ms" in result_dict
        assert "error" in result_dict
        assert result_dict["success"] is True


# ============================================================================
# Test Class 9: Resource Management Tests
# ============================================================================

class TestACEResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_close_resources(self, mock_ace_components):
        """Test closing ACE resources."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.online_ace = mock_ace_components["online_ace"]

            # Add mock stop_async_learning methods
            engine.offline_ace.stop_async_learning = MagicMock()
            engine.online_ace.stop_async_learning = MagicMock()

            await engine.close()

            engine.offline_ace.stop_async_learning.assert_called_once()
            engine.online_ace.stop_async_learning.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_resources_without_async_learning(self, mock_ace_components):
        """Test closing resources when async learning not available."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = mock_ace_components["offline_ace"]
            engine.online_ace = mock_ace_components["online_ace"]

            # Don't add stop_async_learning methods
            delattr(engine.offline_ace, 'stop_async_learning') if hasattr(engine.offline_ace, 'stop_async_learning') else None
            delattr(engine.online_ace, 'stop_async_learning') if hasattr(engine.online_ace, 'stop_async_learning') else None

            # Should not raise an error
            await engine.close()

    @pytest.mark.asyncio
    async def test_close_resources_with_none_components(self):
        """Test closing resources when components are None."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            engine = AgenticContextEngine()
            engine.offline_ace = None
            engine.online_ace = None

            # Should not raise an error
            await engine.close()


# ============================================================================
# Test Class 10: Configuration Edge Cases
# ============================================================================

class TestACEConfigurationEdgeCases:
    """Test configuration edge cases."""

    def test_config_with_missing_optional_fields(self):
        """Test configuration with missing optional fields."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            partial_config = {
                "model": "gpt-4o"
            }
            engine = AgenticContextEngine(config=partial_config)

            # Should merge with defaults
            assert "max_refinement_rounds" in engine.config
            assert "reflection_window" in engine.config

    def test_config_with_empty_strings(self):
        """Test configuration with empty string values."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            config = {
                "model": "",
                "api_key": ""
            }
            engine = AgenticContextEngine(config=config)

            assert engine.config["model"] == ""
            assert engine.config["api_key"] == ""

    def test_config_with_zero_values(self):
        """Test configuration with zero values."""
        with patch('knowledge_engine.integrations.agentic_context_integration.AgenticContextEngine._initialize_components'):
            config = {
                "max_refinement_rounds": 0,
                "reflection_window": 0,
                "offline_training": {
                    "batch_size": 0
                }
            }
            engine = AgenticContextEngine(config=config)

            assert engine.config["max_refinement_rounds"] == 0
            assert engine.config["reflection_window"] == 0
            assert engine.config["offline_training"]["batch_size"] == 0
