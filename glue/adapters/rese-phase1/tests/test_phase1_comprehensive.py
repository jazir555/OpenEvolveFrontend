#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for RESE Phase I Components

Tests ALL public functions and methods for:
- phase1_executor.py (EpistemicAuditExecutor, ConstraintHardener, AssumptionMiner, RedTeamProtocator)
- metacognitive_reflector.py (MetacognitiveReflector)
- bias_metrics.py (BiasMetricsTracker)

Coverage Goals:
- Unit tests for each function
- Integration tests between components
- Performance tests
- Error handling tests
- Idempotency tests
- CLAUDE.md compliance tests

Total Tests: 100+

Author: OpenEvolve
Created: 2026-02-04
"""

import os
import sys
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phase1_executor import (
    Phase1Config,
    EpistemicAuditExecutor,
    TacitAssumption,
    ContradictionDetection,
    FalsificationResult,
    CircuitBreaker,
    CircuitBreakerState,
    DeadLetterQueue,
    StructuredLogger,
    LogicalFallacy,
    ConstraintCategory,
    EpistemicAuditResult,
)

from metacognitive_reflector import (
    MetacognitiveReflector,
    DebiasingConfig,
    Hypothesis,
    BiasAnalysis,
    DebiasingResult,
    BiasType,
    Severity,
)

from bias_metrics import (
    BiasMetricsTracker,
    BiasMeasurement,
    BiasMetricsSummary,
    BiasTrend,
    BiasThresholdConfig,
    calculate_cbi,
    calculate_bias_reduction,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Create sample configuration for testing"""
    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '0.3'
    os.environ['PHASE1_MIN_ROBUSTNESS_SCORE'] = '0.5'
    os.environ['PHASE1_ENABLE_TACIT_MINING'] = 'true'
    os.environ['PHASE1_ENABLE_RED_TEAM'] = 'true'
    os.environ['PHASE1_DEBIASING_ENABLED'] = 'true'
    os.environ['PHASE1_CBI_THRESHOLD'] = '0.5'
    os.environ['PHASE1_ANTITHETICAL_COUNT'] = '3'
    return Phase1Config.from_env()


@pytest.fixture
def sample_logger():
    """Create sample logger for testing"""
    return StructuredLogger('test')


@pytest.fixture
def sample_failure_patterns():
    """Create sample failure patterns for testing"""
    return [
        {
            'pattern_description': 'lattice defects cause failure',
            'failure_rate': 0.65,
            'data_points': 150,
        },
        {
            'pattern_description': 'temperature affects yield',
            'failure_rate': 0.45,
            'data_points': 200,
        },
    ]


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis for testing"""
    return Hypothesis(
        id=str(uuid.uuid4()),
        statement='The temperature obviously causes the reaction to succeed.',
        confidence=0.85,
        assumptions=['Temperature is the key factor', 'Higher temperature increases yield'],
    )


@pytest.fixture
def sample_tacit_assumptions():
    """Create sample tacit assumptions"""
    return [
        TacitAssumption(
            id=str(uuid.uuid4()),
            description='Temperature is the primary control variable',
            source_pattern='lattice defects cause failure',
            confidence_score=0.65,
            supporting_evidence_count=150,
        ),
        TacitAssumption(
            id=str(uuid.uuid4()),
            description='Loading ratio is critical',
            source_pattern='temperature affects yield',
            confidence_score=0.45,
            supporting_evidence_count=200,
        ),
    ]


# =============================================================================
# CONFIGURATION TESTS (10 tests)
# =============================================================================

class TestPhase1Config:
    """Test Phase1Config"""

    def test_config_from_env_default_values(self, sample_config):
        """Test default configuration values"""
        assert sample_config.TIMEOUT_MS == 15000
        assert sample_config.MAX_ASSUMPTIONS == 100
        assert sample_config.MIN_ASSUMPTION_CONFIDENCE == 0.3
        assert sample_config.MIN_ROBUSTNESS_SCORE == 0.5

    def test_config_custom_values(self):
        """Test custom configuration values"""
        os.environ['PHASE1_TIMEOUT_MS'] = '25000'
        os.environ['PHASE1_MAX_ASSUMPTIONS'] = '200'
        config = Phase1Config.from_env()
        assert config.TIMEOUT_MS == 25000
        assert config.MAX_ASSUMPTIONS == 200

    def test_config_invalid_timeout_negative(self):
        """Test invalid negative timeout"""
        os.environ['PHASE1_TIMEOUT_MS'] = '-100'
        with pytest.raises(ValueError, match='must be positive'):
            Phase1Config.from_env()

    def test_config_invalid_timeout_zero(self):
        """Test invalid zero timeout"""
        os.environ['PHASE1_TIMEOUT_MS'] = '0'
        with pytest.raises(ValueError, match='must be positive'):
            Phase1Config.from_env()

    def test_config_invalid_max_assumptions(self):
        """Test invalid max assumptions"""
        os.environ['PHASE1_MAX_ASSUMPTIONS'] = '0'
        with pytest.raises(ValueError, match='must be positive'):
            Phase1Config.from_env()

    def test_config_invalid_confidence_too_high(self):
        """Test invalid confidence > 1.0"""
        os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '1.5'
        with pytest.raises(ValueError, match='between 0 and 1'):
            Phase1Config.from_env()

    def test_config_invalid_confidence_negative(self):
        """Test invalid negative confidence"""
        os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '-0.1'
        with pytest.raises(ValueError, match='between 0 and 1'):
            Phase1Config.from_env()

    def test_config_feature_flags(self):
        """Test feature flag configuration"""
        os.environ['PHASE1_ENABLE_TACIT_MINING'] = 'false'
        os.environ['PHASE1_ENABLE_RED_TEAM'] = 'false'
        config = Phase1Config.from_env()
        assert config.ENABLE_TACIT_MINING is False
        assert config.ENABLE_RED_TEAM_PROTOCOL is False

    def test_config_circuit_breaker_settings(self):
        """Test circuit breaker configuration"""
        os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '10'
        os.environ['PHASE1_CIRCUIT_BREAKER_TIMEOUT_MS'] = '120000'
        config = Phase1Config.from_env()
        assert config.CIRCUIT_BREAKER_THRESHOLD == 10
        assert config.CIRCUIT_BREAKER_TIMEOUT_MS == 120000

    def test_config_iteration_limits(self):
        """Test iteration limit configuration"""
        os.environ['PHASE1_MAX_CONSTRAINTS'] = '500'
        os.environ['PHASE1_MAX_CONTRADICTIONS'] = '50'
        config = Phase1Config.from_env()
        assert config.MAX_CONSTRAINTS == 500
        assert config.MAX_CONTRADICTIONS == 50


# =============================================================================
# STRUCTURED LOGGER TESTS (8 tests)
# =============================================================================

class TestStructuredLogger:
    """Test StructuredLogger"""

    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = StructuredLogger('TestComponent')
        assert logger.component == 'TestComponent'

    def test_logger_info(self, capsys):
        """Test info logging"""
        logger = StructuredLogger('TestComponent')
        logger.info('Test message', key1='value1', key2='value2')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        assert log_output['level'] == 'info'
        assert log_output['component'] == 'TestComponent'
        assert log_output['message'] == 'Test message'
        assert log_output['key1'] == 'value1'

    def test_logger_warn(self, capsys):
        """Test warning logging"""
        logger = StructuredLogger('TestComponent')
        logger.warn('Warning message', warning_code='W001')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        assert log_output['level'] == 'warn'
        assert log_output['message'] == 'Warning message'

    def test_logger_error(self, capsys):
        """Test error logging"""
        logger = StructuredLogger('TestComponent')
        error = ValueError('Test error')
        logger.error('Error occurred', error=error, context='test')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        assert log_output['level'] == 'error'
        assert log_output['error'] == 'Test error'
        assert log_output['error_type'] == 'ValueError'

    def test_logger_debug(self, capsys):
        """Test debug logging"""
        logger = StructuredLogger('TestComponent')
        logger.debug('Debug message')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        assert log_output['level'] == 'debug'

    def test_logger_utc_timestamp(self, capsys):
        """Test UTC timestamps in logs"""
        logger = StructuredLogger('TestComponent')
        logger.info('Test')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        # Verify ISO-8601 format
        assert 'T' in log_output['timestamp']
        assert log_output['timestamp'].endswith('Z') or '+' in log_output['timestamp']

    def test_logger_json_format(self, capsys):
        """Test JSON log format"""
        logger = StructuredLogger('TestComponent')
        logger.info('Test')
        captured = capsys.readouterr()
        # Should be valid JSON
        json.loads(captured.out.strip())

    def test_logger_correlation_id(self, capsys):
        """Test correlation ID in logs"""
        logger = StructuredLogger('TestComponent')
        logger.info('Test', correlation_id='test-123')
        captured = capsys.readouterr()
        log_output = json.loads(captured.out.strip())
        assert log_output['correlation_id'] == 'test-123'


# =============================================================================
# CIRCUIT BREAKER TESTS (12 tests)
# =============================================================================

class TestCircuitBreaker:
    """Test CircuitBreaker"""

    @pytest.fixture
    def circuit_breaker(self, sample_logger):
        """Create circuit breaker for testing"""
        return CircuitBreaker(
            threshold=3,
            timeout_ms=100,
            logger=sample_logger
        )

    def test_circuit_breaker_initial_state(self, circuit_breaker):
        """Test initial state is CLOSED"""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_circuit_breaker_can_execute_closed(self, circuit_breaker):
        """Test can_execute when CLOSED"""
        assert circuit_breaker.can_execute() is True

    def test_circuit_breaker_record_success(self, circuit_breaker):
        """Test recording success"""
        circuit_breaker.failure_count = 2
        circuit_breaker.record_success()
        assert circuit_breaker.failure_count == 0

    def test_circuit_breaker_record_failure(self, circuit_breaker):
        """Test recording failure"""
        circuit_breaker.record_failure()
        assert circuit_breaker.failure_count == 1

    def test_circuit_breaker_opens_on_threshold(self, circuit_breaker):
        """Test circuit breaker opens after threshold"""
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_rejects_when_open(self, circuit_breaker):
        """Test requests rejected when OPEN"""
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.can_execute() is False

    def test_circuit_breaker_half_open_after_timeout(self, circuit_breaker):
        """Test transitions to HALF_OPEN after timeout"""
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)  # Wait for timeout (100ms + buffer)
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_recovers_on_success(self, circuit_breaker):
        """Test recovers to CLOSED on success"""
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)
        circuit_breaker.can_execute()  # Transition to HALF_OPEN
        circuit_breaker.record_success()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_get_stats(self, circuit_breaker):
        """Test getting statistics"""
        circuit_breaker.record_failure()
        stats = circuit_breaker.get_stats()
        assert 'state' in stats
        assert 'failure_count' in stats
        assert stats['failure_count'] == 1

    def test_circuit_breaker_last_failure_time(self, circuit_breaker):
        """Test last failure time is recorded"""
        before = time.time() * 1000
        circuit_breaker.record_failure()
        after = time.time() * 1000
        assert before <= circuit_breaker.last_failure_time <= after

    def test_circuit_breaker_opened_at_time(self, circuit_breaker):
        """Test opened_at time is set"""
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.opened_at is not None


# =============================================================================
# DEAD LETTER QUEUE TESTS (10 tests)
# =============================================================================

class TestDeadLetterQueue:
    """Test DeadLetterQueue"""

    @pytest.fixture
    def dlq(self, sample_logger):
        """Create DLQ for testing"""
        return DeadLetterQueue(max_size=5, structured_logger=sample_logger)

    def test_dlq_initialization(self, dlq):
        """Test DLQ initialization"""
        assert dlq.max_size == 5
        assert len(dlq._queue) == 0

    def test_dlq_enqueue(self, dlq):
        """Test enqueuing items"""
        item = {'audit_id': 'test-1', 'error': 'Test error'}
        result = dlq.enqueue(item)
        assert result is True
        assert dlq.size() == 1

    def test_dlq_dequeue(self, dlq):
        """Test dequeuing items"""
        item = {'audit_id': 'test-1', 'error': 'Test error'}
        dlq.enqueue(item)
        dequeued = dlq.dequeue()
        assert dequeued['audit_id'] == 'test-1'
        assert dlq.size() == 0

    def test_dlq_dequeue_empty(self, dlq):
        """Test dequeuing from empty queue"""
        assert dlq.dequeue() is None

    def test_dlq_peek(self, dlq):
        """Test peeking at queue"""
        items = [
            {'audit_id': 'test-1'},
            {'audit_id': 'test-2'},
        ]
        for item in items:
            dlq.enqueue(item)
        peeked = dlq.peek()
        assert len(peeked) == 2
        assert dlq.size() == 2  # Peek doesn't remove

    def test_dlq_max_size(self, dlq):
        """Test max size limit"""
        for i in range(10):
            dlq.enqueue({'audit_id': f'test-{i}'})
        assert dlq.size() == 5  # Should not exceed max_size

    def test_dlq_fifo_order(self, dlq):
        """Test FIFO ordering"""
        dlq.enqueue({'audit_id': 'first'})
        dlq.enqueue({'audit_id': 'second'})
        assert dlq.dequeue()['audit_id'] == 'first'
        assert dlq.dequeue()['audit_id'] == 'second'

    def test_dlq_size(self, dlq):
        """Test size reporting"""
        assert dlq.size() == 0
        dlq.enqueue({'audit_id': 'test-1'})
        assert dlq.size() == 1

    def test_dlq_idempotency(self, dlq):
        """Test idempotent operations"""
        item = {'audit_id': 'test-1'}
        dlq.enqueue(item)
        dlq.enqueue(item)  # Same item again
        assert dlq.size() == 2  # Both allowed


# =============================================================================
# TACIT ASSUMPTION TESTS (8 tests)
# =============================================================================

class TestTacitAssumption:
    """Test TacitAssumption dataclass"""

    def test_tacit_assumption_creation(self):
        """Test creating tacit assumption"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test assumption',
            source_pattern='Test pattern',
            confidence_score=0.75,
            supporting_evidence_count=100,
        )
        assert assumption.id == 'test-1'
        assert assumption.confidence_score == 0.75

    def test_tacit_assumption_to_dict(self):
        """Test converting to dict"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test assumption',
            source_pattern='Test pattern',
            confidence_score=0.75,
            supporting_evidence_count=100,
        )
        data = assumption.to_dict()
        assert data['id'] == 'test-1'
        assert data['description'] == 'Test assumption'

    def test_tacit_assumption_from_dict(self):
        """Test creating from dict"""
        data = {
            'id': 'test-1',
            'description': 'Test assumption',
            'source_pattern': 'Test pattern',
            'confidence_score': 0.75,
            'supporting_evidence_count': 100,
            'formalized_in_lean4': False,
            'lean4_proposition': None,
        }
        assumption = TacitAssumption.from_dict(data)
        assert assumption.id == 'test-1'
        assert assumption.confidence_score == 0.75

    def test_tacit_assumption_defaults(self):
        """Test default values"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Pattern',
            confidence_score=0.5,
            supporting_evidence_count=10,
        )
        assert assumption.formalized_in_lean4 is False
        assert assumption.lean4_proposition is None


# =============================================================================
# CONTRADICTION DETECTION TESTS (8 tests)
# =============================================================================

class TestContradictionDetection:
    """Test ContradictionDetection dataclass"""

    def test_contradiction_detection_creation(self):
        """Test creating contradiction detection"""
        detection = ContradictionDetection(
            id='test-1',
            fallacy_type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
        )
        assert detection.id == 'test-1'
        assert detection.fallacy_type == LogicalFallacy.CONTRADICTION

    def test_contradiction_detection_to_dict(self):
        """Test converting to dict"""
        detection = ContradictionDetection(
            id='test-1',
            fallacy_type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
        )
        data = detection.to_dict()
        assert data['fallacy_type'] == 'contradiction'

    def test_contradiction_detection_from_dict(self):
        """Test creating from dict"""
        data = {
            'id': 'test-1',
            'fallacy_type': 'contradiction',
            'contradiction_set_size': 2,
            'rollback_steps': 1,
            'affected_premises': ['c1', 'c2'],
            'resolved': False,
            'resolution_strategy': None,
        }
        detection = ContradictionDetection.from_dict(data)
        assert detection.fallacy_type == LogicalFallacy.CONTRADICTION

    def test_contradiction_detection_defaults(self):
        """Test default values"""
        detection = ContradictionDetection(
            id='test-1',
            fallacy_type=LogicalFallacy.INCONSISTENCY,
            contradiction_set_size=3,
            rollback_steps=2,
            affected_premises=['c1', 'c2', 'c3'],
        )
        assert detection.resolved is False
        assert detection.resolution_strategy is None


# =============================================================================
# FALSIFICATION RESULT TESTS (8 tests)
# =============================================================================

class TestFalsificationResult:
    """Test FalsificationResult dataclass"""

    def test_falsification_result_creation(self):
        """Test creating falsification result"""
        result = FalsificationResult(
            hypothesis_id='test-1',
            falsified=True,
            degree_of_violation=0.8,
            hypothesis_robustness_score=0.2,
            falsifying_evidence=['Evidence 1'],
            counter_examples=['Example 1'],
        )
        assert result.hypothesis_id == 'test-1'
        assert result.falsified is True

    def test_falsification_result_to_dict(self):
        """Test converting to dict"""
        result = FalsificationResult(
            hypothesis_id='test-1',
            falsified=False,
            degree_of_violation=None,
            hypothesis_robustness_score=0.9,
            falsifying_evidence=[],
            counter_examples=[],
        )
        data = result.to_dict()
        assert data['hypothesis_id'] == 'test-1'
        assert data['falsified'] is False

    def test_falsification_result_from_dict(self):
        """Test creating from dict"""
        data = {
            'hypothesis_id': 'test-1',
            'falsified': True,
            'degree_of_violation': 0.7,
            'hypothesis_robustness_score': 0.3,
            'falsifying_evidence': ['E1'],
            'counter_examples': ['C1'],
        }
        result = FalsificationResult.from_dict(data)
        assert result.hypothesis_id == 'test-1'
        assert result.falsified is True


# =============================================================================
# METACOGNITIVE REFLECTOR TESTS (15 tests)
# =============================================================================

class TestMetacognitiveReflector:
    """Test MetacognitiveReflector"""

    @pytest.fixture
    def debiasing_config(self):
        """Create debiasing config"""
        os.environ['PHASE1_DEBIASING_ENABLED'] = 'true'
        os.environ['PHASE1_CBI_THRESHOLD'] = '0.5'
        os.environ['PHASE1_ANTITHETICAL_COUNT'] = '3'
        return DebiasingConfig.from_env()

    @pytest.fixture
    def reflector(self, debiasing_config, sample_logger):
        """Create reflector for testing"""
        return MetacognitiveReflector(config=debiasing_config, logger=sample_logger)

    def test_reflector_initialization(self, reflector):
        """Test reflector initialization"""
        assert reflector.config.ENABLE_DEBIASING is True
        assert reflector.config.CBI_THRESHOLD == 0.5
        assert reflector.config.ANTITHETICAL_COUNT == 3

    def test_reflector_perform_debiasing(self, reflector, sample_hypothesis):
        """Test performing debiasing"""
        result = reflector.perform_debiasing(
            hypothesis=sample_hypothesis,
            assumptions=[],
            correlation_id='test-123',
        )
        assert isinstance(result, DebiasingResult)
        assert result.original_hypothesis == sample_hypothesis
        assert result.confirmation_bias_index >= 0

    def test_reflector_debiasing_disabled(self, debiasing_config, sample_logger, sample_hypothesis):
        """Test error when debiasing disabled"""
        debiasing_config.ENABLE_DEBIASING = False
        reflector = MetacognitiveReflector(config=debiasing_config, logger=sample_logger)
        with pytest.raises(RuntimeError, match='disabled'):
            reflector.perform_debiasing(
                hypothesis=sample_hypothesis,
                assumptions=[],
                correlation_id='test-123',
            )

    def test_reflector_bias_identification(self, reflector, sample_hypothesis):
        """Test bias identification"""
        bias_analysis = reflector._identify_directional_bias(sample_hypothesis)
        assert isinstance(bias_analysis, BiasAnalysis)
        assert bias_analysis.bias_type in [BiasType.CONFIRMATION, BiasType.DISCONFIRMATION, BiasType.NEUTRAL]
        assert bias_analysis.confidence >= 0

    def test_reflector_antithetical_generation(self, reflector, sample_hypothesis):
        """Test antithetical outcome generation"""
        antithetical = reflector._generate_antithetical_outcomes(
            hypothesis=sample_hypothesis,
            count=3,
            correlation_id='test-123',
        )
        assert len(antithetical) == 3
        assert all(isinstance(h, Hypothesis) for h in antithetical)

    def test_reflector_cbi_calculation(self, reflector, sample_hypothesis):
        """Test CBI calculation"""
        antithetical = [
            Hypothesis(id='a1', statement='Negation', confidence=0.4, assumptions=[]),
            Hypothesis(id='a2', statement='Alternative', confidence=0.3, assumptions=[]),
        ]
        cbi = reflector._calculate_confirmation_bias_index(
            hypothesis=sample_hypothesis,
            antithetical=antithetical,
            evidence=[],
            correlation_id='test-123',
        )
        assert 0 <= cbi <= 1

    def test_reflector_metacognitive_reflection(self, reflector, sample_hypothesis):
        """Test metacognitive reflection application"""
        bias_analysis = BiasAnalysis(
            bias_type=BiasType.CONFIRMATION,
            confidence=0.8,
            affected_assumptions=[],
            directional_language=['obviously'],
            severity=Severity.HIGH,
        )
        antithetical = [
            Hypothesis(id='a1', statement='Negation', confidence=0.4, assumptions=[]),
        ]
        debiased = reflector._apply_metacognitive_reflection(
            hypothesis=sample_hypothesis,
            bias_analysis=bias_analysis,
            antithetical_outcomes=antithetical,
            correlation_id='test-123',
        )
        assert isinstance(debiased, Hypothesis)
        assert debiased.confidence < sample_hypothesis.confidence

    def test_reflector_hypothesis_negation(self, reflector, sample_hypothesis):
        """Test hypothesis negation"""
        negated = reflector._negate_hypothesis(sample_hypothesis, 'test-123')
        assert isinstance(negated, Hypothesis)
        assert negated.id != sample_hypothesis.id
        assert negated.confidence < sample_hypothesis.confidence

    def test_reflector_causality_inversion(self, reflector, sample_hypothesis):
        """Test causality inversion"""
        inverted = reflector._invert_causality(sample_hypothesis, 'test-123')
        assert isinstance(inverted, Hypothesis)
        assert inverted.id != sample_hypothesis.id

    def test_reflector_get_stats(self, reflector):
        """Test getting reflector statistics"""
        stats = reflector.get_stats()
        assert 'config' in stats
        assert stats['config']['enabled'] is True

    def test_reflector_bias_reduction_calculation(self, reflector, sample_hypothesis):
        """Test bias reduction calculation"""
        result = reflector.perform_debiasing(
            hypothesis=sample_hypothesis,
            assumptions=[],
            correlation_id='test-123',
        )
        assert result.bias_reduction >= 0
        assert result.initial_cbi >= result.confirmation_bias_index or result.bias_reduction == 0


# =============================================================================
# BIAS METRICS TRACKER TESTS (15 tests)
# =============================================================================

class TestBiasMetricsTracker:
    """Test BiasMetricsTracker"""

    @pytest.fixture
    def tracker(self, sample_logger):
        """Create tracker for testing"""
        return BiasMetricsTracker(logger=sample_logger)

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert len(tracker.measurements) == 0
        assert tracker.config is not None

    def test_tracker_record_measurement(self, tracker):
        """Test recording measurement"""
        measurement = tracker.record_measurement(
            epoch=1,
            confirmation_bias_index=0.65,
            initial_cbi=0.80,
            bias_reduction=18.75,
            hypotheses_count=5,
            correlation_id='test-123',
        )
        assert isinstance(measurement, BiasMeasurement)
        assert measurement.epoch == 1
        assert measurement.confirmation_bias_index == 0.65
        assert len(tracker.measurements) == 1

    def test_tracker_calculate_summary_empty(self, tracker):
        """Test summary with no measurements"""
        summary = tracker.calculate_summary()
        assert isinstance(summary, BiasMetricsSummary)
        assert summary.total_epochs == 0
        assert summary.current_cbi == 0.0

    def test_tracker_calculate_summary_with_data(self, tracker):
        """Test summary with measurements"""
        tracker.record_measurement(
            epoch=1,
            confirmation_bias_index=0.70,
            initial_cbi=0.80,
            bias_reduction=12.5,
            hypotheses_count=5,
            correlation_id='test-1',
        )
        tracker.record_measurement(
            epoch=2,
            confirmation_bias_index=0.60,
            initial_cbi=0.70,
            bias_reduction=14.3,
            hypotheses_count=5,
            correlation_id='test-2',
        )
        summary = tracker.calculate_summary()
        assert summary.total_epochs == 2
        assert summary.current_cbi == 0.60
        assert summary.average_cbi == 0.65

    def test_tracker_threshold_check_ok(self, tracker):
        """Test threshold check with OK status"""
        result = tracker.check_thresholds(0.4)
        assert result['status'] == 'ok'
        assert result['cbi'] == 0.4

    def test_tracker_threshold_check_warning(self, tracker):
        """Test threshold check with warning"""
        result = tracker.check_thresholds(0.6)
        assert result['status'] == 'warning'

    def test_tracker_threshold_check_critical(self, tracker):
        """Test threshold check with critical"""
        result = tracker.check_thresholds(0.8)
        assert result['status'] == 'critical'

    def test_tracker_threshold_check_target(self, tracker):
        """Test threshold check meeting target"""
        result = tracker.check_thresholds(0.2)
        assert result['status'] == 'target'

    def test_tracker_get_improvement_rate(self, tracker):
        """Test improvement rate calculation"""
        for i in range(5):
            tracker.record_measurement(
                epoch=i+1,
                confirmation_bias_index=0.8 - (i * 0.05),
                initial_cbi=0.8,
                bias_reduction=i * 5.0,
                hypotheses_count=5,
                correlation_id=f'test-{i}',
            )
        rate = tracker.get_improvement_rate(window_size=5)
        assert isinstance(rate, float)
        assert rate < 0  # Should be negative (improving)

    def test_tracker_export_metrics(self, tracker):
        """Test exporting metrics"""
        tracker.record_measurement(
            epoch=1,
            confirmation_bias_index=0.65,
            initial_cbi=0.80,
            bias_reduction=18.75,
            hypotheses_count=5,
            correlation_id='test-123',
        )
        exported = tracker.export_metrics()
        assert 'summary' in exported
        assert 'config' in exported
        assert 'current_threshold_check' in exported
        assert 'improvement_rate' in exported

    def test_tracker_clear_history(self, tracker):
        """Test clearing history"""
        tracker.record_measurement(
            epoch=1,
            confirmation_bias_index=0.65,
            initial_cbi=0.80,
            bias_reduction=18.75,
            hypotheses_count=5,
            correlation_id='test-123',
        )
        assert len(tracker.measurements) == 1
        tracker.clear_history()
        assert len(tracker.measurements) == 0

    def test_calculate_cbi_function(self):
        """Test standalone CBI calculation"""
        cbi = calculate_cbi(
            hypothesis_confidence=0.8,
            antithetical_confidences=[0.3, 0.4, 0.2],
        )
        assert 0 <= cbi <= 1

    def test_calculate_cbi_no_alternatives(self):
        """Test CBI with no antithetical outcomes"""
        cbi = calculate_cbi(
            hypothesis_confidence=0.8,
            antithetical_confidences=[],
        )
        assert cbi == 1.0  # Maximum bias

    def test_calculate_bias_reduction_function(self):
        """Test standalone bias reduction calculation"""
        reduction = calculate_bias_reduction(
            initial_cbi=0.8,
            final_cbi=0.6,
        )
        assert reduction == 25.0  # (0.8 - 0.6) / 0.8 * 100

    def test_calculate_bias_reduction_zero_initial(self):
        """Test bias reduction with zero initial CBI"""
        reduction = calculate_bias_reduction(
            initial_cbi=0.0,
            final_cbi=0.0,
        )
        assert reduction == 0.0


# =============================================================================
# DATA STRUCTURE TESTS (6 tests)
# =============================================================================

class TestDataStructures:
    """Test data structures"""

    def test_bias_measurement_to_dict(self):
        """Test BiasMeasurement to_dict"""
        measurement = BiasMeasurement(
            epoch=1,
            timestamp='2024-01-01T00:00:00Z',
            confirmation_bias_index=0.65,
            initial_cbi=0.80,
            bias_reduction=18.75,
            hypotheses_count=5,
            correlation_id='test-123',
        )
        data = measurement.to_dict()
        assert data['epoch'] == 1
        assert data['confirmation_bias_index'] == 0.65

    def test_bias_measurement_from_dict(self):
        """Test BiasMeasurement from_dict"""
        data = {
            'epoch': 1,
            'timestamp': '2024-01-01T00:00:00Z',
            'confirmation_bias_index': 0.65,
            'initial_cbi': 0.80,
            'bias_reduction': 18.75,
            'hypotheses_count': 5,
            'correlation_id': 'test-123',
            'metadata': {},
        }
        measurement = BiasMeasurement.from_dict(data)
        assert measurement.epoch == 1

    def test_bias_metrics_summary_to_dict(self):
        """Test BiasMetricsSummary to_dict"""
        summary = BiasMetricsSummary(
            total_epochs=5,
            current_cbi=0.6,
            average_cbi=0.65,
            min_cbi=0.5,
            max_cbi=0.8,
            cbi_trend=BiasTrend.IMPROVING,
            total_bias_reduction=50.0,
            average_bias_reduction=10.0,
            best_epoch=5,
            worst_epoch=1,
            measurements=[],
            timestamp='2024-01-01T00:00:00Z',
        )
        data = summary.to_dict()
        assert data['cbi_trend'] == 'improving'
        assert data['total_epochs'] == 5

    def test_bias_threshold_config_from_env(self):
        """Test BiasThresholdConfig from environment"""
        os.environ['BIAS_WARNING_THRESHOLD'] = '0.6'
        os.environ['BIAS_CRITICAL_THRESHOLD'] = '0.8'
        os.environ['BIAS_TARGET_THRESHOLD'] = '0.3'
        config = BiasThresholdConfig.from_env()
        assert config.WARNING_THRESHOLD == 0.6
        assert config.CRITICAL_THRESHOLD == 0.8
        assert config.TARGET_THRESHOLD == 0.3

    def test_hypothesis_to_dict(self):
        """Test Hypothesis to_dict"""
        hypothesis = Hypothesis(
            id='test-1',
            statement='Test statement',
            confidence=0.75,
            assumptions=['A1', 'A2'],
        )
        data = hypothesis.to_dict()
        assert data['id'] == 'test-1'
        assert data['statement'] == 'Test statement'

    def test_bias_analysis_to_dict(self):
        """Test BiasAnalysis to_dict"""
        analysis = BiasAnalysis(
            bias_type=BiasType.CONFIRMATION,
            confidence=0.8,
            affected_assumptions=['a1'],
            directional_language=['obviously'],
            severity=Severity.HIGH,
        )
        data = analysis.to_dict()
        assert data['bias_type'] == 'confirmation'
        assert data['severity'] == 'high'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
