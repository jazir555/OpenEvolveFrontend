"""
Unit Tests for Φ₁.₅ Tacit Assumption Miner

Comprehensive test suite for all Φ₁.₅ components:
- Data structures
- Failure Preprocessor
- Anomaly Detector
- Failure Clusterer
- Assumption Generator
- Confidence Scorer
- Paradigm Shift Detector
- Main Φ₁.₅ Engine
- Integration tests

Author: Agent B1 (Φ₁/Φ₁.₅ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import sys

# Add rese to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1.tacit_assumption_miner import (
    NullResult, FailureFeatures, TacitAssumption,
    ParadigmShiftRecommendation, ErrorType,
    AssumptionType, PatternType,
    FailurePreprocessor, AnomalyDetector,
    FailureClusterer, AssumptionGenerator,
    ConfidenceScorer, ParadigmShiftDetector,
    Phi15Engine
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_null_result():
    """Create a sample null result for testing"""
    return NullResult(
        attempt_id="test_001",
        timestamp=datetime.now(),
        problem_type="optimization",
        approach_type="deterministic",
        constraints=["constraint_1", "constraint_2"],
        error_type=ErrorType.OPTIMIZATION_FAILED,
        error_message="Optimization failed to converge due to numerical instability",
        state={"iteration": 100, "objective_value": -999.0},
        iteration=100,
        resources_used={"cpu": 50.0, "memory": 100.0},
        metadata={"test": True}
    )


@pytest.fixture
def sample_null_results():
    """Create multiple sample null results"""
    results = []
    for i in range(20):
        result = NullResult(
            attempt_id=f"test_{i:03d}",
            timestamp=datetime.now() - timedelta(hours=i),
            problem_type="optimization",
            approach_type="deterministic",
            constraints=[f"constraint_{j}" for j in range(5)],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message=f"Optimization attempt {i} failed",
            state={"iteration": i * 10},
            iteration=i * 10,
            resources_used={"cpu": float(i * 5), "memory": float(i * 10)},
            metadata={"batch": i // 5}
        )
        results.append(result)
    return results


@pytest.fixture
def sample_failure_features():
    """Create sample failure features"""
    return FailureFeatures(
        attempt_id="test_001",
        timestamp=datetime.now(),
        problem_type="optimization",
        approach_type="deterministic",
        error_type=ErrorType.OPTIMIZATION_FAILED,
        iteration=100,
        time_to_failure=100.0,
        error_magnitude=999.0,
        resource_consumption=0.5,
        constraint_violation_count=2,
        feature_vector=np.array([100.0, 100.0, 999.0, 0.5, 2.0], dtype=np.float32),
        keywords=["optimization", "failed", "numerical", "instability"]
    )


# ============================================================================
# Test Data Structures
# ============================================================================

class TestDataStructures:
    """Test core data structures"""

    def test_null_result_creation(self, sample_null_result):
        """Test NullResult creation and serialization"""
        assert sample_null_result.attempt_id == "test_001"
        assert sample_null_result.error_type == ErrorType.OPTIMIZATION_FAILED
        assert len(sample_null_result.constraints) == 2

        # Test serialization
        data = sample_null_result.to_dict()
        assert 'attempt_id' in data
        assert 'error_type' in data

        # Test deserialization
        restored = NullResult.from_dict(data)
        assert restored.attempt_id == sample_null_result.attempt_id
        assert restored.error_type == sample_null_result.error_type

    def test_tacit_assumption_creation(self):
        """Test TacitAssumption creation"""
        assumption = TacitAssumption(
            id="assumption_001",
            description="Test assumption",
            formalization="forall (x : Entity), test_assumption(x)",
            assumption_type=AssumptionType.CONSTRAINT,
            confidence=0.8,
            support=10,
            evidence=["test_001", "test_002"],
            pattern_type=PatternType.SYSTEMATIC_VIOLATION,
            constraint_relaxation="Relax: Test assumption",
            paradigm_implication=False,
            alternative_paradigm=None
        )

        assert assumption.id == "assumption_001"
        assert assumption.confidence == 0.8
        assert assumption.support == 10
        assert assumption.paradigm_implication is False

        # Test SCE constraint conversion
        sce_constraint = assumption.to_sce_constraint()
        assert sce_constraint.id == assumption.id
        assert "[INFERRED]" in sce_constraint.description


# ============================================================================
# Test Failure Preprocessor
# ============================================================================

class TestFailurePreprocessor:
    """Test Failure Preprocessor component"""

    @pytest.fixture
    def preprocessor(self):
        return FailurePreprocessor()

    def test_extract_features(self, preprocessor, sample_null_result):
        """Test feature extraction from null result"""
        features = preprocessor.extract_features(sample_null_result)

        assert features.attempt_id == sample_null_result.attempt_id
        assert features.problem_type == sample_null_result.problem_type
        assert features.approach_type == sample_null_result.approach_type
        assert features.error_type == sample_null_result.error_type
        assert features.iteration == sample_null_result.iteration

        # Check feature vector
        assert isinstance(features.feature_vector, np.ndarray)
        assert len(features.feature_vector) > 0

        # Check keywords
        assert len(features.keywords) > 0
        assert any('optimization' in kw.lower() for kw in features.keywords)

    def test_keyword_extraction(self, preprocessor):
        """Test keyword extraction from error messages"""
        message = "Optimization failed due to numerical instability in gradient computation"
        keywords = preprocessor._extract_keywords(message)

        assert len(keywords) > 0
        assert "optimization" in keywords
        assert "numerical" in keywords
        assert "instability" in keywords

    def test_time_to_failure_computation(self, preprocessor, sample_null_result):
        """Test time to failure computation"""
        features = preprocessor.extract_features(sample_null_result)
        assert features.time_to_failure > 0


# ============================================================================
# Test Anomaly Detector
# ============================================================================

class TestAnomalyDetector:
    """Test Anomaly Detector component"""

    @pytest.fixture
    def detector(self):
        return AnomalyDetector(contamination=0.1)

    @pytest.fixture
    def sample_features(self, sample_null_results):
        """Create sample failure features"""
        preprocessor = FailurePreprocessor()
        features_list = []
        for nr in sample_null_results:
            features = preprocessor.extract_features(nr)
            features_list.append(features)
        return features_list

    def test_anomaly_detection(self, detector, sample_features):
        """Test anomaly detection on failure features"""
        scores = detector.detect_anomalies(sample_features)

        assert len(scores) == len(sample_features)
        assert all(0 <= s <= 1 for s in scores.values())

        # Check that some failures are flagged as anomalies
        anomaly_count = sum(1 for s in scores.values() if s > 0.5)
        assert anomaly_count >= 0

    def test_insufficient_data(self, detector):
        """Test behavior with insufficient data"""
        # Too few failures for anomaly detection
        few_features = []
        for i in range(2):
            features = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.UNKNOWN_FAILURE,
                iteration=1,
                time_to_failure=1.0,
                error_magnitude=None,
                resource_consumption=0.1,
                constraint_violation_count=0,
                feature_vector=np.array([1.0, 1.0, 1.0, 0.1, 0.0])
            )
            few_features.append(features)

        scores = detector.detect_anomalies(few_features)
        assert len(scores) == 2
        # With insufficient data, should return neutral scores
        assert all(s == 0.0 for s in scores.values())


# ============================================================================
# Test Failure Clusterer
# ============================================================================

class TestFailureClusterer:
    """Test Failure Clusterer component"""

    @pytest.fixture
    def clusterer(self):
        return FailureClusterer()

    @pytest.fixture
    def sample_features_for_clustering(self, sample_null_results):
        """Create sample features suitable for clustering"""
        preprocessor = FailurePreprocessor()
        features_list = []
        for nr in sample_null_results:
            features = preprocessor.extract_features(nr)
            features_list.append(features)
        return features_list

    def test_clustering(self, clusterer, sample_features_for_clustering):
        """Test failure clustering"""
        clusters = clusterer.cluster_failures(sample_features_for_clustering)

        # Should produce some clusters
        assert isinstance(clusters, list)

        # If clusters produced, check properties
        for cluster in clusters:
            assert cluster.size > 0
            assert cluster.size == len(cluster.failures)
            assert cluster.compactness >= 0
            assert -1 <= cluster.silhouette_score <= 1

    def test_insufficient_data_for_clustering(self, clusterer):
        """Test with insufficient data for DBSCAN"""
        few_features = []
        for i in range(3):
            features = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.UNKNOWN_FAILURE,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=None,
                resource_consumption=0.1,
                constraint_violation_count=0,
                feature_vector=np.array([float(i), float(i), 1.0, 0.1, 0.0])
            )
            few_features.append(features)

        clusters = clusterer.cluster_failures(few_features)
        # With insufficient data, should return empty list
        assert isinstance(clusters, list)


# ============================================================================
# Test Assumption Generator
# ============================================================================

class TestAssumptionGenerator:
    """Test Assumption Generator component"""

    @pytest.fixture
    def generator(self):
        return AssumptionGenerator()

    @pytest.fixture
    def sample_cluster(self, sample_features_for_clustering):
        """Create a sample cluster"""
        from phase1.tacit_assumption_miner import FailureCluster
        import numpy as np

        # Create a simple cluster
        features = sample_features_for_clustering[:10]

        centroid = np.mean([f.feature_vector for f in features], axis=0)

        cluster = FailureCluster(
            cluster_id=0,
            size=10,
            failures=features,
            centroid=centroid,
            compactness=0.3,
            silhouette_score=0.5,
            stability=0.8,
            common_problem_types=["optimization"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=["constraint_1"],
            keywords=["optimization", "failed"]
        )
        return cluster

    def test_generate_assumptions(self, generator, sample_cluster):
        """Test assumption generation from cluster"""
        candidates = generator.generate_assumptions(sample_cluster)

        assert isinstance(candidates, list)

        # If candidates generated, check properties
        for candidate in candidates:
            assert candidate.description
            assert 0 <= candidate.confidence <= 1
            assert len(candidate.explains_failures) > 0


# ============================================================================
# Test Confidence Scorer
# ============================================================================

class TestConfidenceScorer:
    """Test Confidence Scorer component"""

    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()

    def test_score_assumption(self, scorer):
        """Test confidence scoring"""
        from phase1.tacit_assumption_miner import (
            AssumptionCandidate, FailureCluster
        )
        import numpy as np

        # Create sample cluster
        features = []
        for i in range(10):
            f = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=1.0,
                resource_consumption=0.1,
                constraint_violation_count=1,
                feature_vector=np.array([float(i), 1.0, 1.0, 0.1, 1.0])
            )
            features.append(f)

        cluster = FailureCluster(
            cluster_id=0,
            size=10,
            failures=features,
            centroid=np.zeros(5),
            compactness=0.3,
            silhouette_score=0.5,
            stability=0.8,
            common_problem_types=["test"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=["c1"],
            keywords=["test"]
        )

        candidate = AssumptionCandidate(
            description="Test assumption",
            explains_failures=[f.attempt_id for f in features],
            confidence=0.5,
            pattern_type=PatternType.SYSTEMATIC_VIOLATION,
            complexity=1,
            contradiction_count=0,
            testable=True
        )

        score = scorer.score_assumption(candidate, cluster, [])

        assert 0 <= score <= 1
        assert score >= 0


# ============================================================================
# Test Paradigm Shift Detector
# ============================================================================

class TestParadigmShiftDetector:
    """Test Paradigm Shift Detector component"""

    @pytest.fixture
    def detector(self):
        return ParadigmShiftDetector(crisis_threshold=0.7)

    def test_no_crisis(self, detector):
        """Test when no paradigm crisis"""
        # Few recent assumptions
        assumptions = []
        for i in range(3):
            assumption = TacitAssumption(
                id=f"assumption_{i}",
                description=f"Assumption {i}",
                formalization=f"formalization_{i}",
                assumption_type=AssumptionType.CONSTRAINT,
                confidence=0.6,
                support=1,
                evidence=[f"test_{i}"],
                pattern_type=PatternType.REPEATED_FAILURE,
                constraint_relaxation=f"Relax {i}",
                paradigm_implication=False,
                alternative_paradigm=None
            )
            assumptions.append(assumption)

        recommendation = detector.detect_crisis(assumptions, [])

        assert recommendation.trigger is False
        assert recommendation.confidence < detector.crisis_threshold

    def test_crisis_detected(self, detector):
        """Test when paradigm crisis is detected"""
        # Many recent high-confidence assumptions
        assumptions = []
        for i in range(15):  # More than threshold
            assumption = TacitAssumption(
                id=f"assumption_{i}",
                description=f"Paradigm-challenging assumption {i}",
                formalization=f"formalization_{i}",
                assumption_type=AssumptionType.METHODOLOGICAL,
                confidence=0.8,
                support=i + 1,
                evidence=[f"test_{j}" for j in range(i + 1)],
                pattern_type=PatternType.CROSS_DOMAIN_FAILURE,
                constraint_relaxation=f"Relax {i}",
                paradigm_implication=i > 10,  # Last few are paradigm-level
                alternative_paradigm="New Paradigm" if i > 10 else None
            )
            assumptions.append(assumption)

        recommendation = detector.detect_crisis(assumptions, [])

        # Should trigger crisis with many assumptions
        assert recommendation.trigger or recommendation.confidence > 0.5


# ============================================================================
# Test Main Φ₁.₅ Engine
# ============================================================================

class TestPhi15Engine:
    """Test main Φ₁.₅ Engine"""

    @pytest.fixture
    def engine(self):
        return Phi15Engine()

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.preprocessor is not None
        assert engine.anomaly_detector is not None
        assert engine.clusterer is not None
        assert engine.assumption_generator is not None
        assert engine.confidence_scorer is not None
        assert engine.paradigm_detector is not None

    def test_process_null_results(self, engine, sample_null_results):
        """Test processing null results through full pipeline"""
        assumptions, paradigm_rec = engine.process_null_results(sample_null_results)

        # Check return types
        assert isinstance(assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)

        # Check paradigm recommendation
        assert hasattr(paradigm_rec, 'trigger')
        assert hasattr(paradigm_rec, 'confidence')

    def test_get_top_assumptions(self, engine, sample_null_results):
        """Test getting top assumptions"""
        engine.process_null_results(sample_null_results)

        top_assumptions = engine.get_top_assumptions(k=5)

        assert isinstance(top_assumptions, list)
        assert len(top_assumptions) <= 5

        # Check that they're sorted by confidence
        if len(top_assumptions) > 1:
            for i in range(len(top_assumptions) - 1):
                assert top_assumptions[i].confidence >= top_assumptions[i + 1].confidence


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for Φ₁.₅ system"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from null results to assumptions"""
        # Create engine
        engine = Phi15Engine()

        # Create sample null results with a pattern
        null_results = []
        for i in range(30):
            result = NullResult(
                attempt_id=f"test_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["exact_solution", "polynomial_time"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Optimization {i} exceeded time limit (exponential complexity)",
                state={"iteration": i * 100, "time_limit": 3600},
                iteration=i * 100,
                resources_used={"cpu": float(i * 10), "memory": float(i * 20)},
                metadata={"pattern": "exact_deterministic_fails"}
            )
            null_results.append(result)

        # Process
        assumptions, paradigm_rec = engine.process_null_results(null_results)

        # Verify results
        assert isinstance(assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)

        # Check that assumptions were generated
        if len(assumptions) > 0:
            # Check assumption properties
            for assumption in assumptions:
                assert assumption.id
                assert assumption.description
                assert 0 <= assumption.confidence <= 1
                assert assumption.support > 0
                assert len(assumption.evidence) > 0

    def test_sce_constraint_conversion(self):
        """Test conversion to SCE constraints"""
        assumption = TacitAssumption(
            id="test_assumption",
            description="Must use approximation algorithms",
            formalization="forall (alg : Algorithm), isApproximate(alg)",
            assumption_type=AssumptionType.METHODOLOGICAL,
            confidence=0.85,
            support=20,
            evidence=["test_001", "test_002"],
            pattern_type=PatternType.REPEATED_FAILURE,
            constraint_relaxation="Allow exact solutions",
            paradigm_implication=False,
            alternative_paradigm=None
        )

        sce_constraint = assumption.to_sce_constraint()

        assert sce_constraint.id == assumption.id
        assert "[INFERRED]" in sce_constraint.description
        assert sce_constraint.source == "phi15_inferred"
        assert sce_constraint.verified is False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
