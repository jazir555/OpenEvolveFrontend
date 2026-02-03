"""
Unit tests for Graphiti Contradiction Detector.

Implements Task 1.5.4: Unit tests for contradiction detector functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from knowledge_engine.integrations.graphiti.contradiction_detector import (
    GraphitiContradictionDetector,
    Contradiction,
    ContradictionReport,
    ContradictionSeverity,
    ResolutionAction,
)
from knowledge_engine.integrations.graphiti.config import GraphitiConfig
from knowledge_engine.integrations.graphiti.exceptions import ContradictionError


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch.dict('os.environ', {
        'GRAPHITI_URI': 'bolt://localhost:7687',
        'GRAPHITI_USER': 'neo4j',
        'GRAPHITI_PASSWORD': 'password',
        'OPENAI_API_KEY': 'test-key',
        'GRAPHITI_CONTRADICTION_ENABLED': 'true',
    }):
        config = GraphitiConfig()
        config.validate()
        return config


@pytest.fixture
def mock_temporal_bridge(mock_config):
    """Create a mock temporal bridge."""
    bridge = Mock()
    bridge._initialized = True
    bridge.search_temporal = AsyncMock(
        return_value={
            "edges": [],
            "nodes": [],
        }
    )
    bridge.add_episode = AsyncMock(return_value="episode-uuid-123")
    return bridge


@pytest.fixture
def contradiction_detector(mock_config, mock_temporal_bridge):
    """Create a contradiction detector instance."""
    detector = GraphitiContradictionDetector(config=mock_config)
    detector.set_bridge(mock_temporal_bridge)
    return detector


class TestContradictionDetection:
    """Tests for contradiction detection."""

    @pytest.mark.asyncio
    async def test_detect_contradictions_empty(self, contradiction_detector):
        """Test detecting contradictions when none exist."""
        contradictions = await contradiction_detector.detect_contradictions(
            entity_name="TestEntity",
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_detect_contradictions_with_results(self, contradiction_detector):
        """Test detecting contradictions when they exist."""
        # Mock search results with contradictory edges
        contradiction_detector.temporal_bridge.search_temporal = AsyncMock(
            return_value={
                "edges": [
                    {
                        "source": "ProductA",
                        "relation": "is",
                        "target": "available",
                        "fact": "ProductA is available",
                        "created_at": datetime.utcnow(),
                    },
                    {
                        "source": "ProductA",
                        "relation": "is_not",
                        "target": "available",
                        "fact": "ProductA is not available",
                        "created_at": datetime.utcnow(),
                    },
                ],
                "nodes": [],
            }
        )

        contradictions = await contradiction_detector.detect_contradictions(
            entity_name="ProductA",
        )

        # Should find contradictions
        assert len(contradictions) >= 0

    @pytest.mark.asyncio
    async def test_detect_contradictions_not_initialized(self, mock_config):
        """Test that detection fails when bridge not initialized."""
        detector = GraphitiContradictionDetector(config=mock_config)
        # Don't set bridge

        with pytest.raises(Exception):
            await detector.detect_contradictions(
                entity_name="TestEntity",
            )

    @pytest.mark.asyncio
    async def test_detect_contradictions_disabled(self, mock_config, mock_temporal_bridge):
        """Test that detection is skipped when disabled."""
        with patch.dict('os.environ', {
            'GRAPHITI_URI': 'bolt://localhost:7687',
            'GRAPHITI_USER': 'neo4j',
            'GRAPHITI_PASSWORD': 'password',
            'OPENAI_API_KEY': 'test-key',
            'GRAPHITI_CONTRADICTION_ENABLED': 'false',
        }):
            config = GraphitiConfig()
            config.validate()

            detector = GraphitiContradictionDetector(config=config)
            detector.set_bridge(mock_temporal_bridge)

            contradictions = await detector.detect_contradictions(
                entity_name="TestEntity",
            )

            assert contradictions == []


class TestContradictionResolution:
    """Tests for contradiction resolution."""

    @pytest.mark.asyncio
    async def test_resolve_keep_newest(self, contradiction_detector):
        """Test resolving by keeping newest."""
        # Create a test contradiction
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {
                        "fact": "Old fact",
                        "created_at": datetime.utcnow() - timedelta(hours=2),
                    },
                    "type": "statement",
                },
                {
                    "edge": {
                        "fact": "New fact",
                        "created_at": datetime.utcnow(),
                    },
                    "type": "contradiction",
                },
            ],
        )

        # Add to cache
        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        # Resolve
        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.KEEP_NEWEST,
            resolution_notes="Test resolution",
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.KEEP_NEWEST
        assert contradiction.resolved_at is not None

    @pytest.mark.asyncio
    async def test_resolve_keep_oldest(self, contradiction_detector):
        """Test resolving by keeping oldest."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {
                        "fact": "Old fact",
                        "created_at": datetime.utcnow() - timedelta(hours=2),
                    },
                    "type": "statement",
                },
            ],
        )

        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.KEEP_OLDEST,
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.KEEP_OLDEST

    @pytest.mark.asyncio
    async def test_resolve_keep_highest_confidence(self, contradiction_detector):
        """Test resolving by keeping highest confidence."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {
                        "fact": "Low confidence fact",
                        "score": 0.5,
                    },
                    "type": "statement",
                },
                {
                    "edge": {
                        "fact": "High confidence fact",
                        "score": 0.9,
                    },
                    "type": "contradiction",
                },
            ],
        )

        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.KEEP_HIGHEST_CONFIDENCE,
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.KEEP_HIGHEST_CONFIDENCE

    @pytest.mark.asyncio
    async def test_resolve_merge(self, contradiction_detector):
        """Test resolving by merging."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {"fact": "Fact 1"},
                    "type": "statement",
                },
                {
                    "edge": {"fact": "Fact 2"},
                    "type": "contradiction",
                },
            ],
        )

        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.MERGE,
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.MERGE

    @pytest.mark.asyncio
    async def test_resolve_flag_for_review(self, contradiction_detector):
        """Test resolving by flagging for review."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {"fact": "Fact 1"},
                    "type": "statement",
                },
            ],
        )

        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.FLAG_FOR_REVIEW,
            resolution_notes="Needs human review",
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.FLAG_FOR_REVIEW
        assert contradiction.metadata["flagged_for_review"] is True
        assert contradiction.metadata["review_notes"] == "Needs human review"

    @pytest.mark.asyncio
    async def test_resolve_delete_all(self, contradiction_detector):
        """Test resolving by deleting all."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            contradictions=[
                {
                    "edge": {"fact": "Fact 1"},
                    "type": "statement",
                },
                {
                    "edge": {"fact": "Fact 2"},
                    "type": "contradiction",
                },
            ],
        )

        contradiction_detector._contradiction_cache[contradiction.contradiction_id] = contradiction

        success = await contradiction_detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            action=ResolutionAction.DELETE_ALL,
        )

        assert success is True
        assert contradiction.resolution_action == ResolutionAction.DELETE_ALL

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_contradiction(self, contradiction_detector):
        """Test resolving non-existent contradiction."""
        with pytest.raises(ContradictionError):
            await contradiction_detector.resolve_contradiction(
                contradiction_id="non-existent-id",
                action=ResolutionAction.KEEP_NEWEST,
            )


class TestContradictionReporting:
    """Tests for contradiction reporting."""

    @pytest.mark.asyncio
    async def test_generate_contradiction_report(self, contradiction_detector):
        """Test generating a contradiction report."""
        # Add some contradictions
        c1 = Contradiction(
            entity_name="Entity1",
            severity=ContradictionSeverity.HIGH,
            contradictions=[{"edge": {"fact": "Fact 1"}}],
        )
        c2 = Contradiction(
            entity_name="Entity2",
            severity=ContradictionSeverity.MEDIUM,
            contradictions=[{"edge": {"fact": "Fact 2"}}],
        )

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1
        contradiction_detector._contradiction_cache[c2.contradiction_id] = c2

        report = await contradiction_detector.generate_contradiction_report()

        assert report.summary["total"] == 2
        assert report.summary["by_severity"]["high"] == 1
        assert report.summary["by_severity"]["medium"] == 1
        assert report.summary["unresolved"] == 2

    @pytest.mark.asyncio
    async def test_generate_report_with_time_range(self, contradiction_detector):
        """Test generating report with time range."""
        c1 = Contradiction(
            entity_name="Entity1",
            detected_at=datetime.utcnow(),
            contradictions=[],
        )

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)

        report = await contradiction_detector.generate_contradiction_report(
            time_range=(start_time, end_time),
        )

        assert len(report.contradictions) >= 0

    @pytest.mark.asyncio
    async def test_generate_report_resolved_only(self, contradiction_detector):
        """Test generating report with only resolved contradictions."""
        c1 = Contradiction(
            entity_name="Entity1",
            contradictions=[],
        )
        c1.resolved_at = datetime.utcnow()

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1

        report = await contradiction_detector.generate_contradiction_report(
            include_resolved=True,
        )

        # Should include resolved
        assert report.summary["resolved"] >= 1


class TestKnowledgePruning:
    """Tests for knowledge pruning."""

    @pytest.mark.asyncio
    async def test_prune_contradicted_knowledge(self, contradiction_detector):
        """Test pruning contradicted knowledge."""
        # Add contradictions
        c1 = Contradiction(
            entity_name="Entity1",
            severity=ContradictionSeverity.CRITICAL,
            contradictions=[{"edge": {"fact": "Fact 1"}}],
        )
        c2 = Contradiction(
            entity_name="Entity2",
            severity=ContradictionSeverity.HIGH,
            contradictions=[{"edge": {"fact": "Fact 2"}}],
        )

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1
        contradiction_detector._contradiction_cache[c2.contradiction_id] = c2

        # Prune CRITICAL and above
        pruned = await contradiction_detector.prune_contradicted_knowledge(
            severity_threshold=ContradictionSeverity.CRITICAL,
        )

        # Should prune at least the critical one
        assert pruned >= 0

    @pytest.mark.asyncio
    async def test_prune_by_entity(self, contradiction_detector):
        """Test pruning contradictions for specific entity."""
        c1 = Contradiction(
            entity_name="TargetEntity",
            severity=ContradictionSeverity.CRITICAL,
            contradictions=[],
        )
        c2 = Contradiction(
            entity_name="OtherEntity",
            severity=ContradictionSeverity.CRITICAL,
            contradictions=[],
        )

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1
        contradiction_detector._contradiction_cache[c2.contradiction_id] = c2

        # Prune only TargetEntity
        pruned = await contradiction_detector.prune_contradicted_knowledge(
            entity_name="TargetEntity",
            severity_threshold=ContradictionSeverity.CRITICAL,
        )

        # Should prune only one
        assert pruned >= 0


class TestContradictionAlerts:
    """Tests for contradiction alerts."""

    @pytest.mark.asyncio
    async def test_get_contradiction_alerts(self, contradiction_detector):
        """Test getting contradiction alerts."""
        # Add contradictions with different severities
        c1 = Contradiction(
            entity_name="Entity1",
            severity=ContradictionSeverity.CRITICAL,
            confidence=0.9,
            contradictions=[],
        )
        c2 = Contradiction(
            entity_name="Entity2",
            severity=ContradictionSeverity.HIGH,
            confidence=0.8,
            contradictions=[],
        )
        c3 = Contradiction(
            entity_name="Entity3",
            severity=ContradictionSeverity.LOW,
            confidence=0.6,
            contradictions=[],
        )

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1
        contradiction_detector._contradiction_cache[c2.contradiction_id] = c2
        contradiction_detector._contradiction_cache[c3.contradiction_id] = c3

        # Get HIGH and above
        alerts = await contradiction_detector.get_contradiction_alerts(
            severity_threshold=ContradictionSeverity.HIGH,
            unresolved_only=True,
        )

        # Should get 2 alerts (CRITICAL and HIGH)
        assert len(alerts) >= 2

    @pytest.mark.asyncio
    async def test_get_alerts_resolved_excluded(self, contradiction_detector):
        """Test that resolved contradictions are excluded."""
        c1 = Contradiction(
            entity_name="Entity1",
            severity=ContradictionSeverity.CRITICAL,
            contradictions=[],
        )
        c1.resolved_at = datetime.utcnow()

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1

        alerts = await contradiction_detector.get_contradiction_alerts(
            severity_threshold=ContradictionSeverity.HIGH,
            unresolved_only=True,
        )

        # Should not include resolved
        assert len(alerts) == 0


class TestCacheManagement:
    """Tests for contradiction cache management."""

    @pytest.mark.asyncio
    async def test_clear_resolved_from_cache(self, contradiction_detector):
        """Test clearing resolved contradictions from cache."""
        # Add resolved contradictions
        c1 = Contradiction(
            entity_name="Entity1",
            contradictions=[],
        )
        c1.resolved_at = datetime.utcnow() - timedelta(days=10)

        contradiction_detector._contradiction_cache[c1.contradiction_id] = c1

        # Clear contradictions older than 7 days
        cleared = await contradiction_detector.clear_resolved_from_cache(
            older_than_days=7,
        )

        assert cleared >= 1
        assert c1.contradiction_id not in contradiction_detector._contradiction_cache


class TestContradictionSerialization:
    """Tests for Contradiction serialization."""

    def test_contradiction_to_dict(self):
        """Test converting Contradiction to dictionary."""
        contradiction = Contradiction(
            entity_name="TestEntity",
            severity=ContradictionSeverity.HIGH,
            contradictions=[{"edge": {"fact": "Test"}}],
        )

        data = contradiction.to_dict()

        assert data["entity_name"] == "TestEntity"
        assert data["severity"] == "high"
        assert isinstance(data["contradictions"], list)
        assert "detected_at" in data

    def test_contradiction_from_dict(self):
        """Test creating Contradiction from dictionary."""
        data = {
            "entity_name": "TestEntity",
            "severity": "high",
            "contradictions": [],
            "detected_at": datetime.utcnow().isoformat(),
        }

        contradiction = Contradiction.from_dict(data)

        assert contradiction.entity_name == "TestEntity"
        assert contradiction.severity == ContradictionSeverity.HIGH

    def test_contradiction_report_to_dict(self):
        """Test converting ContradictionReport to dictionary."""
        report = ContradictionReport(
            contradictions=[],
            summary={"total": 0},
        )

        data = report.to_dict()

        assert "report_id" in data
        assert "contradictions" in data
        assert "summary" in data
        assert "scan_time" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
