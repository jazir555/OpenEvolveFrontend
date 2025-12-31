#!/usr/bin/env python
"""
Comprehensive Tests for Phase 5: UI/CLI Integration

Tests for CLI tools, review interface, monitoring dashboard,
and knowledge explorer.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any


# ============================================================================
# CLI Tests
# ============================================================================

class TestRAGBitsCLI:
    """Test CLI tool"""

    def test_cli_creation(self):
        """Test CLI can be instantiated"""
        from ragbits_integration.ui_cli.cli import RAGBitsCLI

        cli = RAGBitsCLI()
        assert cli is not None
        assert cli.parser is not None
        assert len(cli.commands) == 8

    def test_cli_parser(self):
        """Test CLI argument parser"""
        from ragbits_integration.ui_cli.cli import RAGBitsCLI

        cli = RAGBitsCLI()

        # Test extract command
        args = cli.parser.parse_args(["extract", "--file", "test.md", "--type", "solution"])
        assert args.command == "extract"
        assert args.file == "test.md"
        assert args.type == "solution"

        # Test score command
        args = cli.parser.parse_args(["score", "--artifact", "art_123", "--details"])
        assert args.command == "score"
        assert args.artifact == "art_123"
        assert args.details is True

        # Test explore command
        args = cli.parser.parse_args(["explore", "--query", "authentication", "--limit", "5"])
        assert args.command == "explore"
        assert args.query == "authentication"
        assert args.limit == 5

    def test_cli_help(self):
        """Test CLI help output"""
        from ragbits_integration.ui_cli.cli import RAGBitsCLI

        cli = RAGBitsCLI()
        # Should not raise
        try:
            cli.parser.print_help()
        except Exception:
            pytest.fail("print_help() raised exception")


# ============================================================================
# Review Interface Tests
# ============================================================================

class TestReviewInterface:
    """Test review interface"""

    @pytest.fixture
    def review_interface(self):
        """Create review interface fixture"""
        from ragbits_integration.ui_cli.interfaces import ReviewInterface, ReviewStatus, CommentType
        # Store enums on instance for test convenience
        ri = ReviewInterface()
        ri.ReviewStatus = ReviewStatus
        ri.CommentType = CommentType
        return ri

    @pytest.fixture
    def sample_artifact(self):
        """Sample artifact content"""
        return """
# User Authentication System

## Overview
This document describes a user authentication system using JWT tokens.

## Components
- User registration
- Login/logout
- Password reset
- Token refresh

## Security Considerations
- Password hashing with bcrypt
- Token expiration handling
- Rate limiting
"""

    def test_create_review_session(self, review_interface, sample_artifact):
        """Test creating review session"""
        session = asyncio.run(review_interface.create_review_session(
            artifact_id="art_123",
            artifact_content=sample_artifact,
            artifact_type="solution",
            reviewers=["user1", "user2", "user3"]
        ))

        assert session is not None
        assert session.artifact_id == "art_123"
        assert session.artifact_type == "solution"
        assert len(session.reviewers) == 3
        assert session.overall_status.value == "pending"

    def test_add_comment(self, review_interface, sample_artifact):
        """Test adding comments"""
        session = asyncio.run(review_interface.create_review_session(
            artifact_id="art_123",
            artifact_content=sample_artifact,
            artifact_type="solution",
            reviewers=["user1"]
        ))

        comment = asyncio.run(review_interface.add_comment(
            review_id=session.review_id,
            author="user1",
            content="Consider adding multi-factor authentication",
            comment_type=review_interface.CommentType.SUGGESTION,
            section="Security Considerations"
        ))

        assert comment is not None
        assert comment.author == "user1"
        assert comment.comment_type.value == "suggestion"
        assert comment.resolved is False

    def test_submit_decision(self, review_interface, sample_artifact):
        """Test submitting review decision"""
        session = asyncio.run(review_interface.create_review_session(
            artifact_id="art_123",
            artifact_content=sample_artifact,
            artifact_type="solution",
            reviewers=["user1"]
        ))

        decision = asyncio.run(review_interface.submit_decision(
            review_id=session.review_id,
            status=review_interface.ReviewStatus.APPROVED,
            reviewer="user1",
            summary="Good solution, approved with conditions",
            conditions=["Add MFA support", "Add rate limiting tests"]
        ))

        assert decision is not None
        assert decision.status.value == "approved"
        assert decision.reviewer == "user1"
        assert len(decision.conditions) == 2

    def test_review_summary(self, review_interface, sample_artifact):
        """Test getting review summary"""
        session = asyncio.run(review_interface.create_review_session(
            artifact_id="art_123",
            artifact_content=sample_artifact,
            artifact_type="solution",
            reviewers=["user1", "user2"]
        ))

        # Add some comments
        asyncio.run(review_interface.add_comment(
            review_id=session.review_id,
            author="user1",
            content="Comment 1"
        ))
        asyncio.run(review_interface.add_comment(
            review_id=session.review_id,
            author="user2",
            content="Comment 2"
        ))

        summary = asyncio.run(review_interface.get_review_summary(session.review_id))

        assert summary is not None
        assert summary["artifact_id"] == "art_123"
        assert summary["total_comments"] == 2
        assert summary["reviewers"] == ["user1", "user2"]


# ============================================================================
# Monitoring Dashboard Tests
# ============================================================================

class TestMonitoringDashboard:
    """Test monitoring dashboard"""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard fixture"""
        from ragbits_integration.ui_cli.monitoring import MonitoringDashboard, MetricType, AlertSeverity
        # Store enums on instance for test convenience
        db = MonitoringDashboard()
        db.MetricType = MetricType
        db.AlertSeverity = AlertSeverity
        return db

    def test_dashboard_creation(self, dashboard):
        """Test dashboard creation"""
        assert dashboard is not None
        assert len(dashboard._metrics) > 0

    def test_register_metric(self, dashboard):
        """Test registering new metric"""
        metric = dashboard.register_metric(
            name="test_metric",
            metric_type=dashboard.MetricType.GAUGE,
            description="Test metric",
            unit="tests"
        )

        assert metric is not None
        assert metric.name == "test_metric"
        assert metric.unit == "tests"

    def test_record_metric(self, dashboard):
        """Test recording metric values"""
        # Record some values
        dashboard.record_metric("artifacts_stored_total", 10.0)
        dashboard.record_metric("artifacts_stored_total", 15.0)
        dashboard.record_metric("artifacts_stored_total", 20.0)

        metric = dashboard.get_metric("artifacts_stored_total")
        assert metric is not None
        assert len(metric.data_points) == 3
        assert metric.get_current_value() == 20.0

    def test_metric_average(self, dashboard):
        """Test metric average calculation"""
        # Register metric first
        dashboard.register_metric(
            name="test_counter",
            metric_type=dashboard.MetricType.COUNTER,
            description="Test counter"
        )

        # Record values
        for i in range(10):
            dashboard.record_metric("test_counter", float(i))

        metric = dashboard.get_metric("test_counter")
        avg = metric.get_average(duration_minutes=60)

        assert avg is not None
        assert avg == 4.5  # Average of 0-9

    def test_alert_definition(self, dashboard):
        """Test defining alerts"""
        alert = dashboard.define_alert(
            alert_id="test_alert",
            name="Test Alert",
            description="Test alert description",
            metric_name="test_metric",
            condition="> 100",
            severity=dashboard.AlertSeverity.WARNING
        )

        assert alert is not None
        assert alert.alert_id == "test_alert"
        assert alert.severity.value == "warning"
        assert alert.triggered is False

    def test_alert_triggering(self, dashboard):
        """Test alert triggering"""
        # Define alert
        dashboard.define_alert(
            alert_id="high_latency",
            name="High Latency",
            description="Query latency too high",
            metric_name="query_latency_ms",
            condition="> 1000",
            severity=dashboard.AlertSeverity.ERROR
        )

        # Record low value (no alert)
        dashboard.record_metric("query_latency_ms", 500.0)
        active_alerts = dashboard.get_active_alerts()
        assert len(active_alerts) == 0

        # Record high value (should trigger)
        dashboard.record_metric("query_latency_ms", 1500.0)
        active_alerts = dashboard.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "high_latency"

    def test_system_health_update(self, dashboard):
        """Test system health update"""
        # Record some metrics
        dashboard.record_metric("vector_index_size", 1000.0)
        dashboard.record_metric("cache_hit_rate", 75.0)
        dashboard.record_metric("query_latency_ms", 500.0)

        # Update health
        health = asyncio.run(dashboard.update_system_health())

        assert health is not None
        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert len(health.components) > 0

    def test_export_metrics_json(self, dashboard):
        """Test exporting metrics as JSON"""
        # Register and record some data
        dashboard.register_metric(
            name="test_metric",
            metric_type=dashboard.MetricType.GAUGE,
            description="Test metric"
        )
        dashboard.record_metric("test_metric", 42.0)

        # Export
        json_str = dashboard.export_metrics_json(duration_minutes=60)

        assert json_str is not None
        assert "test_metric" in json_str
        assert "42.0" in json_str


# ============================================================================
# Knowledge Explorer Tests
# ============================================================================

class TestKnowledgeExplorer:
    """Test knowledge explorer"""

    @pytest.fixture
    def explorer(self):
        """Create explorer fixture"""
        from ragbits_integration.ui_cli.exploration import KnowledgeExplorer, SearchStrategy, SortOrder, EntityType
        # Store enums on instance for test convenience
        ex = KnowledgeExplorer()
        ex.SearchStrategy = SearchStrategy
        ex.SortOrder = SortOrder
        ex.EntityType = EntityType
        return ex

    def test_explorer_creation(self, explorer):
        """Test explorer creation"""
        assert explorer is not None
        assert explorer._search_history == []

    def test_search(self, explorer):
        """Test knowledge search"""
        # Note: This will return empty results without RAG engine configured
        results, metadata = asyncio.run(explorer.search(
            query="authentication patterns",
            strategy=explorer.SearchStrategy.HYBRID,
            limit=10
        ))

        assert results is not None
        assert metadata is not None
        assert "total_count" in metadata
        assert "limit" in metadata

    def test_search_filter_creation(self, explorer):
        """Test creating search filters"""
        from ragbits_integration.ui_cli.exploration import SearchFilter

        filter = SearchFilter(
            entity_types=[explorer.EntityType.SOLUTION_PATTERN, explorer.EntityType.BEST_PRACTICE],
            min_quality_score=0.7,
            tags=["security", "authentication"]
        )

        assert filter.entity_types is not None
        assert len(filter.entity_types) == 2
        assert filter.min_quality_score == 0.7

    def test_search_result_creation(self, explorer):
        """Test creating search results"""
        from ragbits_integration.ui_cli.exploration import SearchResult

        result = SearchResult(
            entity_id="entity_123",
            entity_type=explorer.EntityType.SOLUTION_PATTERN,
            content="Solution pattern content",
            metadata={"artifact_type": "solution"},
            relevance_score=0.95,
            quality_score=0.85
        )

        assert result.entity_id == "entity_123"
        assert result.relevance_score == 0.95
        assert result.quality_score == 0.85

    def test_highlight_generation(self, explorer):
        """Test highlight generation"""
        content = "Authentication is important for security. Authentication should use JWT."
        highlights = explorer._generate_highlights("authentication", content)

        assert len(highlights) > 0
        assert any("auth" in h.lower() for h in highlights)

    def test_facets(self, explorer):
        """Test getting facets"""
        facets = asyncio.run(explorer.get_facets())

        assert facets is not None
        assert "entity_type" in facets
        assert "artifact_type" in facets

    def test_export_json(self, explorer):
        """Test exporting results as JSON"""
        from ragbits_integration.ui_cli.exploration import SearchResult

        results = [
            SearchResult(
                entity_id="entity_1",
                entity_type=explorer.EntityType.SOLUTION_PATTERN,
                content="Content 1",
                metadata={},
                relevance_score=0.9,
                quality_score=0.8,
                highlights=["highlight 1"]
            )
        ]

        metadata = {"total_count": 1, "offset": 0, "limit": 10}

        json_str = explorer.export_search_results(
            results,
            metadata,
            format="json"
        )

        assert "entity_1" in json_str
        assert "Content 1" in json_str

    def test_export_markdown(self, explorer):
        """Test exporting results as Markdown"""
        from ragbits_integration.ui_cli.exploration import SearchResult

        results = [
            SearchResult(
                entity_id="entity_1",
                entity_type=explorer.EntityType.SOLUTION_PATTERN,
                content="Content 1",
                metadata={},
                relevance_score=0.9,
                quality_score=0.8
            )
        ]

        metadata = {"total_count": 1, "offset": 0, "limit": 10}

        md_str = explorer.export_search_results(
            results,
            metadata,
            format="markdown"
        )

        assert "# Knowledge Search Results" in md_str
        assert "entity_1" in md_str

    def test_export_csv(self, explorer):
        """Test exporting results as CSV"""
        from ragbits_integration.ui_cli.exploration import SearchResult

        results = [
            SearchResult(
                entity_id="entity_1",
                entity_type=explorer.EntityType.SOLUTION_PATTERN,
                content="Content 1",
                metadata={},
                relevance_score=0.9,
                quality_score=0.8
            )
        ]

        csv_str = explorer.export_search_results(
            results,
            {},
            format="csv"
        )

        assert "entity_id,entity_type" in csv_str
        assert "entity_1" in csv_str

    def test_search_history(self, explorer):
        """Test search history tracking"""
        # Perform searches
        asyncio.run(explorer.search("query 1"))
        asyncio.run(explorer.search("query 2"))
        asyncio.run(explorer.search("query 3"))

        history = explorer.get_search_history()

        assert len(history) == 3
        assert history[0]["query"] == "query 1"
        assert history[1]["query"] == "query 2"
        assert history[2]["query"] == "query 3"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 components"""

    def test_review_to_monitoring_integration(self):
        """Test review interface integrates with monitoring"""
        from ragbits_integration.ui_cli.interfaces import ReviewInterface
        from ragbits_integration.ui_cli.monitoring import MonitoringDashboard

        # Create components
        review = ReviewInterface()
        dashboard = MonitoringDashboard()

        # Create review session
        session = asyncio.run(review.create_review_session(
            artifact_id="art_123",
            artifact_content="Test content",
            artifact_type="solution",
            reviewers=["user1"]
        ))

        # Record metrics
        dashboard.record_metric("active_review_sessions", 1.0)
        dashboard.record_metric("total_reviews", 1.0)

        # Verify metrics recorded
        metric = dashboard.get_metric("active_review_sessions")
        assert metric.get_current_value() == 1.0

    def test_explorer_to_review_integration(self):
        """Test knowledge explorer integrates with review"""
        from ragbits_integration.ui_cli.interfaces import ReviewInterface
        from ragbits_integration.ui_cli.exploration import KnowledgeExplorer

        # Create components
        review = ReviewInterface()
        explorer = KnowledgeExplorer()

        # Create review
        session = asyncio.run(review.create_review_session(
            artifact_id="art_123",
            artifact_content="Authentication solution",
            artifact_type="solution",
            reviewers=["user1"]
        ))

        # Use explorer to find similar solutions
        # Note: Will return empty without RAG engine
        results, _ = asyncio.run(explorer.search(
            query="authentication",
            limit=5
        ))

        # Verify search executed
        assert results is not None
        assert len(explorer.get_search_history()) == 1

    def test_dashboard_html_generation(self):
        """Test dashboard HTML generation"""
        from ragbits_integration.ui_cli.monitoring import MonitoringDashboard

        dashboard = MonitoringDashboard()

        # Record some data
        dashboard.record_metric("artifacts_stored_total", 100.0)
        dashboard.record_metric("queries_total", 500.0)
        dashboard.record_metric("query_latency_ms", 250.0)

        # Generate HTML
        html = asyncio.run(dashboard.generate_dashboard_html())

        assert html is not None
        assert "<!DOCTYPE html>" in html
        assert "RAGBits Monitoring Dashboard" in html
        # HTML uses title case with spaces
        assert "Artifacts Stored Total" in html or "artifacts_stored_total" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
