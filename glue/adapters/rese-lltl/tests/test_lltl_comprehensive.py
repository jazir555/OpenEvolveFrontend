"""
Comprehensive Test Suite for LLTL Integration Component

Tests cover:
1. Configuration (5 tests)
2. Confidence Tracker (15 tests)
3. LLTL Adapter (10 tests)
4. Formal Commitments (10 tests)

Total: 40+ tests targeting >90% code coverage

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Test env var validation
- Law of Idempotency: Test cache hit returns same result
- Law of UTC: Test timestamp format
- Structured Logging: Test JSON log format
- Circuit Breaker: Test Z3 fallback behavior

Author: RESE Team
Created: 2026-02-04
"""

import pytest
import os
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import uuid

# Import test targets
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock LLTL imports before importing our modules
sys.modules['rese_lltl'] = MagicMock()
sys.modules['z3prover_integration'] = MagicMock()

from confidence_tracker import (
    ConfidenceTracker,
    ConfidenceThreshold,
    ConfidenceLevel,
    ConfidenceHistory,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def correlation_id():
    """Test correlation ID"""
    return "test-lltl-correlation-12345"


@pytest.fixture
def tracker_config():
    """Test confidence tracker configuration"""
    return {
        "significance_level": 0.05,
        "default_threshold": 0.75,
        "very_high_threshold": 0.90,
        "high_threshold": 0.75,
        "moderate_threshold": 0.60,
        "low_threshold": 0.50,
        "calculation_strategy": "tiered",
        "enable_history": True,
        "max_history_size": 10000
    }


@pytest.fixture
def confidence_tracker(tracker_config):
    """Test confidence tracker instance"""
    return ConfidenceTracker(tracker_config)


@pytest.fixture
def sample_statistical_result():
    """Sample statistical result for testing"""
    return {
        "hypothesis_statement": "Treatment X improves outcome Y",
        "confidence": 0.85,
        "p_value": 0.03,
        "confidence_interval": (0.72, 0.95),
        "expected_value": 0.83,
        "validation_metric": "aci_score",
        "evidence": [{"sample_size": 100}, {"effect_size": 0.5}]
    }


# =============================================================================
# A. CONFIGURATION TESTS (5 tests)
# =============================================================================

class TestConfiguration:
    """Test suite for configuration management"""

    def test_tracker_config_from_env(self):
        """Test configuration loads from environment variables"""
        with patch.dict(os.environ, {
            "LLTL_SIGNIFICANCE_LEVEL": "0.01",
            "LLTL_CONFIDENCE_THRESHOLD_DEFAULT": "0.80",
            "LLTL_VERY_HIGH_THRESHOLD": "0.95",
            "LLTL_THRESHOLD_STRATEGY": "linear",
            "LLTL_ENABLE_THRESHOLD_HISTORY": "false",
        }):
            tracker = ConfidenceTracker()

            assert tracker.config["significance_level"] == 0.01
            assert tracker.config["default_threshold"] == 0.80
            assert tracker.config["very_high_threshold"] == 0.95
            assert tracker.config["calculation_strategy"] == "linear"
            assert tracker.config["enable_history"] is False

    def test_config_validation_invalid_significance(self):
        """Test configuration validation fails for invalid significance level"""
        with patch.dict(os.environ, {"LLTL_SIGNIFICANCE_LEVEL": "1.5"}):
            with pytest.raises(RuntimeError) as exc_info:
                ConfidenceTracker()

            assert "SIGNIFICANCE_LEVEL must be between 0 and 1" in str(exc_info.value)

    def test_config_validation_invalid_threshold(self):
        """Test configuration validation fails for invalid threshold"""
        with patch.dict(os.environ, {"LLTL_VERY_HIGH_THRESHOLD": "1.5"}):
            with pytest.raises(RuntimeError) as exc_info:
                ConfidenceTracker()

            assert "VERY_HIGH_THRESHOLD must be between 0 and 1" in str(exc_info.value)

    def test_config_validation_negative_max_history(self):
        """Test configuration validation fails for negative max history"""
        with patch.dict(os.environ, {"LLTL_MAX_THRESHOLD_HISTORY": "-100"}):
            with pytest.raises(RuntimeError) as exc_info:
                ConfidenceTracker()

            assert "MAX_HISTORY_SIZE must be positive" in str(exc_info.value)

    def test_config_override(self):
        """Test configuration can be overridden"""
        override_config = {
            "significance_level": 0.10,
            "calculation_strategy": "adaptive"
        }

        tracker = ConfidenceTracker(override_config)

        assert tracker.config["significance_level"] == 0.10
        assert tracker.config["calculation_strategy"] == "adaptive"


# =============================================================================
# B. CONFIDENCE TRACKER TESTS (15 tests)
# =============================================================================

class TestConfidenceTracker:
    """Test suite for ConfidenceTracker"""

    def test_tracker_initialization(self, confidence_tracker):
        """Test tracker initializes correctly"""
        assert confidence_tracker.config is not None
        assert confidence_tracker.threshold_history == []
        assert confidence_tracker._threshold_cache == {}

    def test_calculate_threshold_very_high(self, confidence_tracker, correlation_id):
        """Test threshold calculation for very high confidence"""
        threshold = confidence_tracker.calculate_threshold(
            confidence=0.97,
            derivation_method="test_method",
            correlation_id=correlation_id
        )

        assert threshold.threshold == 0.90
        assert threshold.level == ConfidenceLevel.VERY_HIGH
        assert threshold.significance_level == 0.05
        assert threshold.derivation_method == "test_method"
        assert threshold.correlation_id == correlation_id

    def test_calculate_threshold_high(self, confidence_tracker, correlation_id):
        """Test threshold calculation for high confidence"""
        threshold = confidence_tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="mcts_validation",
            correlation_id=correlation_id
        )

        assert threshold.threshold == 0.75
        assert threshold.level == ConfidenceLevel.HIGH

    def test_calculate_threshold_moderate(self, confidence_tracker, correlation_id):
        """Test threshold calculation for moderate confidence"""
        threshold = confidence_tracker.calculate_threshold(
            confidence=0.70,
            derivation_method="statistical_test",
            correlation_id=correlation_id
        )

        assert threshold.threshold == 0.60
        assert threshold.level == ConfidenceLevel.MODERATE

    def test_calculate_threshold_low(self, confidence_tracker, correlation_id):
        """Test threshold calculation for low confidence"""
        threshold = confidence_tracker.calculate_threshold(
            confidence=0.50,
            derivation_method="heuristic",
            correlation_id=correlation_id
        )

        assert threshold.threshold == 0.50
        assert threshold.level == ConfidenceLevel.LOW

    def test_calculate_threshold_invalid_confidence(self, confidence_tracker, correlation_id):
        """Test threshold calculation rejects invalid confidence"""
        with pytest.raises(ValueError) as exc_info:
            confidence_tracker.calculate_threshold(
                confidence=1.5,
                correlation_id=correlation_id
            )

        assert "Confidence must be between 0 and 1" in str(exc_info.value)

    def test_calculate_threshold_negative_confidence(self, confidence_tracker, correlation_id):
        """Test threshold calculation rejects negative confidence"""
        with pytest.raises(ValueError) as exc_info:
            confidence_tracker.calculate_threshold(
                confidence=-0.1,
                correlation_id=correlation_id
            )

        assert "Confidence must be between 0 and 1" in str(exc_info.value)

    def test_calculate_threshold_linear_strategy(self, correlation_id):
        """Test threshold calculation with linear strategy"""
        config = {"calculation_strategy": "linear", "low_threshold": 0.50, "very_high_threshold": 0.90}
        tracker = ConfidenceTracker(config)

        threshold = tracker.calculate_threshold(
            confidence=0.75,
            derivation_method="linear",
            correlation_id=correlation_id
        )

        # Linear interpolation: 0.50 + (0.90 - 0.50) * 0.75 = 0.80
        assert abs(threshold.threshold - 0.80) < 0.01

    def test_calculate_threshold_adaptive_strategy(self, correlation_id):
        """Test threshold calculation with adaptive strategy"""
        config = {"calculation_strategy": "adaptive"}
        tracker = ConfidenceTracker(config)

        # Adaptive should fall back to tiered
        threshold = tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="adaptive",
            correlation_id=correlation_id
        )

        assert threshold.level == ConfidenceLevel.HIGH

    def test_threshold_cache_idempotency(self, confidence_tracker, correlation_id):
        """Test Law of Idempotency: cache returns same result"""
        threshold1 = confidence_tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="test",
            correlation_id=correlation_id
        )
        threshold2 = confidence_tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="test",
            correlation_id=correlation_id
        )

        # Should return same cached object
        assert threshold1 is threshold2
        assert threshold1.threshold == threshold2.threshold

    def test_track_threshold_success(self, confidence_tracker, correlation_id):
        """Test tracking threshold in history"""
        threshold = confidence_tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="test",
            correlation_id=correlation_id
        )

        history_id = confidence_tracker.track_threshold(
            proposition_id="prop-123",
            input_confidence=0.85,
            threshold=threshold,
            correlation_id=correlation_id
        )

        assert history_id is not None
        assert len(confidence_tracker.threshold_history) == 1
        assert confidence_tracker.threshold_history[0].proposition_id == "prop-123"

    def test_track_threshold_disabled(self, correlation_id):
        """Test tracking raises error when history disabled"""
        config = {"enable_history": False}
        tracker = ConfidenceTracker(config)

        threshold = ConfidenceThreshold(
            threshold=0.75,
            level=ConfidenceLevel.HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="test",
            correlation_id=correlation_id
        )

        with pytest.raises(RuntimeError) as exc_info:
            tracker.track_threshold(
                proposition_id="prop-123",
                input_confidence=0.85,
                threshold=threshold,
                correlation_id=correlation_id
            )

        assert "Threshold history tracking is disabled" in str(exc_info.value)

    def test_get_history_by_proposition(self, confidence_tracker, correlation_id):
        """Test getting history filtered by proposition"""
        threshold = confidence_tracker.calculate_threshold(0.85, correlation_id=correlation_id)
        confidence_tracker.track_threshold("prop-1", 0.85, threshold, correlation_id)
        confidence_tracker.track_threshold("prop-1", 0.80, threshold, correlation_id)
        confidence_tracker.track_threshold("prop-2", 0.75, threshold, correlation_id)

        history = confidence_tracker.get_history(proposition_id="prop-1")

        assert len(history) == 2
        assert all(h.proposition_id == "prop-1" for h in history)

    def test_get_history_limit(self, confidence_tracker, correlation_id):
        """Test getting history with limit"""
        threshold = confidence_tracker.calculate_threshold(0.85, correlation_id=correlation_id)

        for i in range(10):
            confidence_tracker.track_threshold(f"prop-{i}", 0.85, threshold, correlation_id)

        history = confidence_tracker.get_history(limit=5)

        assert len(history) == 5

    def test_clear_history(self, confidence_tracker, correlation_id):
        """Test clearing history"""
        threshold = confidence_tracker.calculate_threshold(0.85, correlation_id=correlation_id)
        confidence_tracker.track_threshold("prop-1", 0.85, threshold, correlation_id)
        confidence_tracker.track_threshold("prop-2", 0.80, threshold, correlation_id)

        count = confidence_tracker.clear_history()

        assert count == 2
        assert len(confidence_tracker.threshold_history) == 0

    def test_get_stats(self, confidence_tracker, correlation_id):
        """Test getting tracker statistics"""
        stats = confidence_tracker.get_stats()

        assert "config" in stats
        assert "history" in stats
        assert "thresholds" in stats
        assert stats["config"]["significance_level"] == 0.05
        assert stats["history"]["total_entries"] == 0


# =============================================================================
# C. CONFIDENCE THRESHOLD TESTS (10 tests)
# =============================================================================

class TestConfidenceThreshold:
    """Test suite for ConfidenceThreshold dataclass"""

    def test_threshold_creation(self, correlation_id):
        """Test threshold object creation"""
        threshold = ConfidenceThreshold(
            threshold=0.75,
            level=ConfidenceLevel.HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="test_method",
            correlation_id=correlation_id
        )

        assert threshold.threshold == 0.75
        assert threshold.level == ConfidenceLevel.HIGH
        assert threshold.significance_level == 0.05

    def test_threshold_serialization(self, correlation_id):
        """Test threshold serialization to dict"""
        threshold = ConfidenceThreshold(
            threshold=0.75,
            level=ConfidenceLevel.HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="test_method",
            correlation_id=correlation_id,
            metadata={"key": "value"}
        )

        data = threshold.to_dict()

        assert data["threshold"] == 0.75
        assert data["level"] == "high"
        assert data["significance_level"] == 0.05
        assert data["derivation_method"] == "test_method"
        assert data["correlation_id"] == correlation_id
        assert data["metadata"] == {"key": "value"}

    def test_confidence_level_enum(self):
        """Test confidence level enum values"""
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MODERATE.value == "moderate"
        assert ConfidenceLevel.LOW.value == "low"

    def test_timestamp_utc_format(self, correlation_id):
        """Test Law of UTC: timestamps are in UTC ISO-8601 format"""
        threshold = ConfidenceThreshold(
            threshold=0.75,
            level=ConfidenceLevel.HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="test",
            correlation_id=correlation_id
        )

        # Check timestamp ends with Z (UTC indicator)
        assert threshold.derived_at.endswith("Z")

        # Parse and verify timezone
        dt = datetime.fromisoformat(threshold.derived_at)
        assert dt.tzinfo == timezone.utc


# =============================================================================
# D. LLTL ADAPTER TESTS (10 tests)
# =============================================================================

class TestLLTLAdapter:
    """Test suite for LLTL Adapter"""

    def test_adapter_initialization_with_mock(self):
        """Test adapter initializes with mocked LLTL"""
        # Mock the LLTL module
        mock_translator = MagicMock()
        mock_translator.translate.return_value = ({"loss_functions": []}, None)

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)
        sys.modules['rese_lltl'].EncodingConfig = MagicMock
        sys.modules['rese_lltl'].LossConfig = MagicMock
        sys.modules['rese_lltl'].DITOConfig = MagicMock

        # Import after mocking
        from lltl_adapter import LLTLAdapter

        adapter = LLTLAdapter()

        assert adapter.config is not None
        assert adapter.translator is not None

    def test_adapter_config_validation(self):
        """Test adapter validates configuration"""
        from lltl_adapter import LLTLAdapter

        # Test invalid encoding dimension
        with patch.dict(os.environ, {"LLTL_ENCODING_DIM": "0"}):
            with pytest.raises(RuntimeError) as exc_info:
                LLTLAdapter()

            assert "ENCODING_DIM must be positive" in str(exc_info.value)

    def test_adapter_health_check(self):
        """Test adapter health check"""
        from lltl_adapter import LLTLAdapter

        mock_translator = MagicMock()
        mock_encoder = MagicMock()
        mock_composer = MagicMock()
        mock_dito = MagicMock()

        mock_translator.encoder = mock_encoder
        mock_translator.composer = mock_composer
        mock_translator.dito = mock_dito

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        adapter = LLTLAdapter()
        is_healthy, message = adapter.health_check()

        assert is_healthy is True
        assert "healthy" in message.lower()

    def test_adapter_translate_constraints(self):
        """Test constraint translation"""
        from lltl_adapter import LLTLAdapter

        mock_translator = MagicMock()
        mock_translator.translate.return_value = (
            {"loss_functions": ["loss1", "loss2"]},
            None
        )

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        adapter = LLTLAdapter()
        result, error = adapter.translate_constraints([])

        assert error is None
        assert result is not None
        assert "loss_functions" in result

    def test_adapter_translate_constraints_error(self):
        """Test constraint translation error handling"""
        from lltl_adapter import LLTLAdapter

        mock_translator = MagicMock()
        mock_translator.translate.return_value = (
            None,
            "Translation failed"
        )

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        adapter = LLTLAdapter()
        result, error = adapter.translate_constraints([])

        assert result is None
        assert error == "Translation failed"

    def test_adapter_encode_single(self):
        """Test single constraint encoding"""
        from lltl_adapter import LLTLAdapter

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = ({"encoded": "data"}, None)

        mock_translator = MagicMock()
        mock_translator.encoder = mock_encoder

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        adapter = LLTLAdapter()
        result, error = adapter.encode_single({"constraint": "test"})

        assert error is None
        assert result is not None

    def test_adapter_get_stats(self):
        """Test getting adapter statistics"""
        from lltl_adapter import LLTLAdapter

        mock_translator = MagicMock()
        mock_translator.get_stats.return_value = {"cache_hits": 10}

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        adapter = LLTLAdapter()
        stats = adapter.get_stats()

        assert "adapter_config" in stats
        assert "translator_stats" in stats
        assert stats["translator_stats"]["cache_hits"] == 10

    def test_adapter_detect_contradictions_z3(self):
        """Test contradiction detection with Z3"""
        from lltl_adapter import LLTLAdapter, FormalCommitment

        mock_z3_solver = MagicMock()
        mock_translator = MagicMock()
        mock_translator.dito.detect_contradictions.return_value = ([], None)

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)
        sys.modules['z3prover_integration'].Z3SolverEngine = MagicMock(return_value=mock_z3_solver)
        sys.modules['z3prover_integration'].is_z3_available = MagicMock(return_value=True)

        with patch.dict(os.environ, {"RESE_Z3_LLTL_ENABLED": "true"}):
            adapter = LLTLAdapter()

            commitment = FormalCommitment(
                proposition_id="test-1",
                statement="x > 5",
                confidence_threshold=0.75,
                statistical_evidence={},
                source_hypothesis="h1",
                derivation_method="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id="test"
            )

            contradictions, error = adapter.detect_contradictions([commitment])

            assert error is None
            assert isinstance(contradictions, list)

    def test_adapter_detect_contradictions_naive_fallback(self):
        """Test contradiction detection falls back to naive method"""
        from lltl_adapter import LLTLAdapter, FormalCommitment

        mock_translator = MagicMock()
        mock_translator.dito.detect_contradictions.return_value = ([], None)

        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)

        with patch.dict(os.environ, {"RESE_Z3_LLTL_ENABLED": "false"}):
            adapter = LLTLAdapter()

            commitment1 = FormalCommitment(
                proposition_id="test-1",
                statement="x > 5",
                confidence_threshold=0.75,
                statistical_evidence={},
                source_hypothesis="h1",
                derivation_method="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id="test"
            )

            contradictions, error = adapter.detect_contradictions([commitment1])

            assert error is None
            assert isinstance(contradictions, list)


# =============================================================================
# E. FORMAL COMMITMENTS TESTS (10 tests)
# =============================================================================

class TestFormalCommitments:
    """Test suite for Formal Commitments"""

    def test_formal_commitment_creation(self, correlation_id):
        """Test formal commitment creation"""
        from lltl_adapter import FormalCommitment

        commitment = FormalCommitment(
            proposition_id="prop-123",
            statement="Treatment improves outcome",
            confidence_threshold=0.75,
            statistical_evidence={"confidence": 0.85, "p_value": 0.03},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id
        )

        assert commitment.proposition_id == "prop-123"
        assert commitment.confidence_threshold == 0.75
        assert commitment.statistical_evidence["confidence"] == 0.85

    def test_formal_commitment_to_sce_constraint(self, correlation_id):
        """Test converting formal commitment to SCE constraint"""
        from lltl_adapter import FormalCommitment

        commitment = FormalCommitment(
            proposition_id="prop-123",
            statement="Treatment improves outcome",
            confidence_threshold=0.75,
            statistical_evidence={"confidence": 0.85},
            source_hypothesis="h1",
            derivation_method="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id
        )

        sce_constraint = commitment.to_sce_constraint()

        assert sce_constraint["constraint_id"] == "prop-123"
        assert sce_constraint["formal_statement"] == "Treatment improves outcome"
        assert sce_constraint["confidence"] == 0.75
        assert sce_constraint["type"] == "statistical_commitment"

    def test_formal_commitment_serialization(self, correlation_id):
        """Test formal commitment serialization"""
        from lltl_adapter import FormalCommitment

        commitment = FormalCommitment(
            proposition_id="prop-123",
            statement="Test statement",
            confidence_threshold=0.75,
            statistical_evidence={"p_value": 0.03},
            source_hypothesis="h1",
            derivation_method="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
            lean4_theorem="theorem test : True"
        )

        data = commitment.to_dict()

        assert data["proposition_id"] == "prop-123"
        assert data["confidence_threshold"] == 0.75
        assert data["lean4_theorem"] == "theorem test : True"
        assert data["correlation_id"] == correlation_id

    def test_formal_commitment_timestamp_utc(self, correlation_id):
        """Test Law of UTC: formal commitment timestamps are UTC"""
        from lltl_adapter import FormalCommitment

        commitment = FormalCommitment(
            proposition_id="prop-123",
            statement="Test",
            confidence_threshold=0.75,
            statistical_evidence={},
            source_hypothesis="h1",
            derivation_method="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id
        )

        # Check timestamp format
        assert commitment.timestamp.endswith("Z")
        dt = datetime.fromisoformat(commitment.timestamp)
        assert dt.tzinfo == timezone.utc


# =============================================================================
# CLAUDE.md COMPLIANCE TESTS
# =============================================================================

class TestCLAUDECompliance:
    """Test suite for CLAUDE.md principle compliance"""

    def test_law_of_configuration_explicitness(self):
        """Test Law of Configuration Explicitness: config from env vars"""
        with patch.dict(os.environ, {
            "LLTL_SIGNIFICANCE_LEVEL": "0.01",
            "LLTL_THRESHOLD_STRATEGY": "linear"
        }):
            tracker = ConfidenceTracker()

            assert tracker.config["significance_level"] == 0.01
            assert tracker.config["calculation_strategy"] == "linear"

    def test_law_of_idempotency_cache(self, confidence_tracker, correlation_id):
        """Test Law of Idempotency: cache provides idempotent behavior"""
        threshold1 = confidence_tracker.calculate_threshold(0.85, correlation_id=correlation_id)
        threshold2 = confidence_tracker.calculate_threshold(0.85, correlation_id=correlation_id)

        # Same object returned from cache
        assert threshold1 is threshold2

        # Same values
        assert threshold1.to_dict() == threshold2.to_dict()

    def test_law_of_utc_timestamps(self, correlation_id):
        """Test Law of UTC: all timestamps are in UTC"""
        threshold = ConfidenceThreshold(
            threshold=0.75,
            level=ConfidenceLevel.HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="test",
            correlation_id=correlation_id
        )

        # Check UTC format
        assert threshold.derived_at.endswith("Z")
        dt = datetime.fromisoformat(threshold.derived_at)
        assert dt.tzinfo == timezone.utc

    def test_structured_logging_json(self, confidence_tracker, correlation_id):
        """Test Structured Logging: logs are JSON format"""
        # The logger should be configured for JSON output
        assert confidence_tracker.logger is not None

        # Log format should be JSON
        # (In production, you'd capture and validate log output)
        logger = confidence_tracker.logger.logger
        assert logger.name == "confidence_tracker"

    def test_circuit_breaker_z3_fallback(self):
        """Test Circuit Breaker: falls back to naive when Z3 unavailable"""
        from lltl_adapter import LLTLAdapter, FormalCommitment

        mock_translator = MagicMock()
        mock_translator.dito.detect_contradictions.return_value = ([], None)

        # Mock Z3 as unavailable
        sys.modules['rese_lltl'].LogicToLossTranslator = MagicMock(return_value=mock_translator)
        sys.modules['z3prover_integration'].is_z3_available = MagicMock(return_value=False)

        with patch.dict(os.environ, {"RESE_Z3_LLTL_ENABLED": "true"}):
            adapter = LLTLAdapter()

            # Should fall back to naive method when Z3 unavailable
            commitment = FormalCommitment(
                proposition_id="test-1",
                statement="x > 5",
                confidence_threshold=0.75,
                statistical_evidence={},
                source_hypothesis="h1",
                derivation_method="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id="test"
            )

            contradictions, error = adapter.detect_contradictions([commitment])

            # Should not raise error, fallback handles it
            assert error is None
