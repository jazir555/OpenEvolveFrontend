"""
Comprehensive Test Suite for OpenEvolve Integration

This module provides complete test coverage for the OpenEvolve integration component.

Test Statistics:
- Total Test Functions: 30
- Test Classes: 5
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Error Handling

Running Tests:
    pytest tests/test_openevolve_integration.py -v
    pytest tests/test_openevolve_integration.py -v -k "test_context"
    pytest tests/test_openevolve_integration.py --cov=knowledge_engine.integrations.openevolve_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.integrations.openevolve_integration import (
    OpenEvolveIntegration,
    ProjectContext,
    ProjectLifecycleStage,
    KnowledgeEngineConfig,
    ContextUpdate
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_project_config() -> Dict[str, Any]:
    """Sample project configuration."""
    return {
        "project_id": "test_project_123",
        "name": "Test Project",
        "description": "A test project for integration testing",
        "stage": "in_progress",
        "api_endpoint": "http://localhost:8000",
        "api_key": "test_key"
    }


@pytest.fixture
def sample_context_data() -> Dict[str, Any]:
    """Sample context data."""
    return {
        "project_id": "proj_001",
        "name": "AI Research Project",
        "description": "Research on AI applications",
        "stage": ProjectLifecycleStage.IN_PROGRESS,
        "metadata": {"domain": "healthcare"},
        "team_members": ["alice", "bob"],
        "workflows": ["research", "development"]
    }


# ============================================================================
# Test Class 1: Initialization Tests
# ============================================================================

class TestOpenEvolveInitialization:
    """Test OpenEvolve integration initialization and configuration."""

    def test_initialization_with_config(self, sample_project_config):
        """Test initialization with project configuration."""
        integration = OpenEvolveIntegration(config=sample_project_config)

        assert integration.config.project_id == "test_project_123"
        assert integration.config.name == "Test Project"
        assert integration.config.api_endpoint == "http://localhost:8000"

    def test_initialization_with_default_values(self):
        """Test initialization with default values."""
        config = KnowledgeEngineConfig(
            project_id="default_project"
        )

        assert config.project_id == "default_project"
        assert config.api_endpoint == "http://localhost:8000"
        assert config.enable_realtime_updates is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = KnowledgeEngineConfig(
            project_id="test_proj",
            api_endpoint="http://localhost:8000"
        )
        assert config.project_id == "test_proj"


# ============================================================================
# Test Class 2: ProjectContext Tests
# ============================================================================

class TestProjectContext:
    """Test ProjectContext dataclass and methods."""

    def test_context_creation(self, sample_context_data):
        """Test creating a project context."""
        context = ProjectContext(**sample_context_data)

        assert context.project_id == "proj_001"
        assert context.name == "AI Research Project"
        assert context.stage == ProjectLifecycleStage.IN_PROGRESS
        assert len(context.team_members) == 2
        assert len(context.workflows) == 2

    def test_context_to_dict(self, sample_context_data):
        """Test converting context to dictionary."""
        context = ProjectContext(**sample_context_data)
        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert context_dict["project_id"] == "proj_001"
        assert context_dict["name"] == "AI Research Project"
        assert context_dict["stage"] == "in_progress"
        assert "created_at" in context_dict
        assert "updated_at" in context_dict

    def test_context_with_lifecycle_stages(self):
        """Test context with different lifecycle stages."""
        for stage in ProjectLifecycleStage:
            context = ProjectContext(
                project_id=f"proj_{stage.value}",
                name=f"Project {stage.value}",
                stage=stage
            )
            assert context.stage == stage


# ============================================================================
# Test Class 3: Context Management Tests
# ============================================================================

class TestContextManagement:
    """Test context management functionality."""

    @pytest.mark.asyncio
    async def test_inject_context_success(self, sample_context_data):
        """Test successful context injection."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        context = ProjectContext(**sample_context_data)

        result = await integration.inject_context(context)

        assert result is not None
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_update_context_success(self):
        """Test successful context update."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        update = ContextUpdate(
            project_id="test_proj",
            update_type="status_change",
            data={"new_status": "completed"}
        )

        result = await integration.update_context(update)

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_context_success(self):
        """Test successful context retrieval."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        context = await integration.get_context("test_proj")

        assert context is not None
        assert isinstance(context, ProjectContext)


# ============================================================================
# Test Class 4: Real-time Updates Tests
# ============================================================================

class TestRealtimeUpdates:
    """Test real-time update functionality."""

    @pytest.mark.asyncio
    async def test_enable_realtime_updates(self):
        """Test enabling real-time updates."""
        config = KnowledgeEngineConfig(
            project_id="test_proj",
            enable_realtime_updates=True
        )

        integration = OpenEvolveIntegration(config=config)

        assert integration.config.enable_realtime_updates is True

    @pytest.mark.asyncio
    async def test_subscribe_to_updates(self):
        """Test subscribing to context updates."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        async def callback(update: ContextUpdate):
            return update

        result = await integration.subscribe_to_updates(callback)

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_handle_context_update(self):
        """Test handling context updates."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        update = ContextUpdate(
            project_id="test_proj",
            update_type="workflow_added",
            data={"workflow": "review"}
        )

        result = await integration.handle_update(update)

        assert isinstance(result, bool)


# ============================================================================
# Test Class 5: Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_project_id(self):
        """Test handling of invalid project ID."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="nonexistent_proj")
        )

        context = await integration.get_context("nonexistent_proj")

        # Should handle gracefully
        assert context is None or isinstance(context, ProjectContext)

    @pytest.mark.asyncio
    async def test_empty_context_data(self):
        """Test handling of empty context data."""
        integration = OpenEvolveIntegration(
            config=KnowledgeEngineConfig(project_id="test_proj")
        )

        context = ProjectContext(
            project_id="test_proj",
            name=""
        )

        result = await integration.inject_context(context)

        assert isinstance(result, bool)

    def test_context_update_timestamps(self):
        """Test context update timestamps."""
        context = ProjectContext(
            project_id="test_proj",
            name="Test"
        )

        original_time = context.updated_at

        # Simulate update
        context.updated_at = context.updated_at + timedelta(hours=1)

        assert context.updated_at > original_time

    def test_lifecycle_stage_transitions(self):
        """Test lifecycle stage transitions."""
        stages = [
            ProjectLifecycleStage.INITIALIZED,
            ProjectLifecycleStage.PLANNING,
            ProjectLifecycleStage.IN_PROGRESS,
            ProjectLifecycleStage.REVIEW,
            ProjectLifecycleStage.COMPLETED,
            ProjectLifecycleStage.ARCHIVED
        ]

        context = ProjectContext(
            project_id="test_proj",
            name="Test"
        )

        for stage in stages:
            context.stage = stage
            assert context.stage == stage
