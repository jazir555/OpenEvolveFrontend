"""
Unit Tests for Φ₁.₅ Tacit Assumption Miner

Tests all components of the tacit assumption mining system.

Author: Agent B1 (Φ₁/Φ₁.₅ Specialist)
Created: 2025-12-31
Status: 🟢 Active
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from tacit_assumption_miner import (
    NullResult,
    ErrorType,
    AssumptionType,
    PatternType,
    FailureFeatures,
    FailureCluster,
    AssumptionCandidate,
    TacitAssumption,
    ParadigmShiftRecommendation,
    FailurePreprocessor,
    AnomalyDetector,
    FailureClusterer,
    AssumptionGenerator,
    ConfidenceScorer,
    ParadigmShiftDetector,
    Phi15Engine,
    create_phi15_engine
)


class TestDataClasses:
    """Test data structures"""

    def test_null_result_creation(self):
        """Test creating NullResult"""
        nr = NullResult(
            attempt_id="test_001",
            timestamp=datetime.now(),
            problem_type="optimization",
            approach_type="gradient_descent",
            constraints=["x > 0"],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message="Failed to converge",
            state={"iteration": 100},
            iteration=100,
            resources_used={"cpu": 50.0, "memory": 100.0}
        )

        assert nr.attempt_id == "test_001"
        assert nr.error_type == ErrorType.OPTIMIZATION_FAILED

    def test_null_result_to_dict(self):
        """Test NullResult serialization"""
        nr = NullResult(
            attempt_id="test_002",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.DIVERGENCE,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )

        d = nr.to_dict()

        assert d['attempt_id'] == "test_002"
        assert d['error_type'] == "divergence"
        assert isinstance(d['timestamp'], str)

    def test_null_result_from_dict(self):
        """Test NullResult deserialization"""
        data = {
            'attempt_id': 'test_003',
            'timestamp': datetime.now().isoformat(),
            'problem_type': 'test',
            'approach_type': 'test',
            'constraints': [],
            'error_type': 'constraint_violation',
            'error_message': 'Test',
            'state': {},
            'iteration': 1,
            'resources_used': {}
        }

        nr = NullResult.from_dict(data)

        assert nr.attempt_id == 'test_003'
        assert nr.error_type == ErrorType.CONSTRAINT_VIOLATION

    def test_failure_features(self):
        """Test FailureFeatures creation"""
        features = FailureFeatures(
            attempt_id="feat_001",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            iteration=50,
            time_to_failure=100.0,
            error_magnitude=0.5,
            resource_consumption=0.7,
            constraint_violation_count=2,
            feature_vector=np.array([1.0, 2.0, 3.0]),
            keywords=["convergence", "instability"]
        )

        assert features.attempt_id == "feat_001"
        assert len(features.keywords) == 2

    def test_failure_features_to_dict(self):
        """Test FailureFeatures serialization"""
        features = FailureFeatures(
            attempt_id="feat_002",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            error_type=ErrorType.TIMEOUT,
            iteration=10,
            time_to_failure=50.0,
            error_magnitude=None,
            resource_consumption=0.5,
            constraint_violation_count=1,
            feature_vector=np.array([1.0, 2.0])
        )

        d = features.to_dict()

        assert d['attempt_id'] == "feat_002"
        assert isinstance(d['feature_vector'], list)
        assert d['error_magnitude'] is None

    def test_tacit_assumption_creation(self):
        """Test TacitAssumption creation"""
        assumption = TacitAssumption(
            id="assump_001",
            description="Hidden constraint discovered",
            formalization="forall x, hidden_constraint(x)",
            assumption_type=AssumptionType.CONSTRAINT,
            confidence=0.8,
            support=10,
            evidence=["fail_1", "fail_2"],
            pattern_type=PatternType.SYSTEMATIC_VIOLATION,
            constraint_relaxation="Relax: remove constraint",
            paradigm_implication=False,
            alternative_paradigm=None
        )

        assert assumption.id == "assump_001"
        assert assumption.confidence == 0.8
        assert assumption.paradigm_implication is False

    def test_tacit_assumption_to_sce_constraint(self):
        """Test converting to SCE constraint"""
        assumption = TacitAssumption(
            id="assump_002",
            description="Test assumption",
            formalization="test_formalization",
            assumption_type=AssumptionType.METHODOLOGICAL,
            confidence=0.7,
            support=5,
            evidence=[],
            pattern_type=PatternType.REPEATED_FAILURE,
            constraint_relaxation="Relax test",
            paradigm_implication=False,
            alternative_paradigm=None
        )

        constraint = assumption.to_sce_constraint()

        assert constraint.id == "assump_002"
        assert "[INFERRED]" in constraint.description
        assert constraint.source == "phi15_inferred"

    def test_paradigm_shift_recommendation(self):
        """Test ParadigmShiftRecommendation"""
        rec = ParadigmShiftRecommendation(
            trigger=True,
            confidence=0.85,
            primary_assumptions=[],
            suggested_alternatives=["Alternative paradigm"],
            explanation="Crisis detected"
        )

        assert rec.trigger is True
        assert rec.confidence == 0.85
        assert len(rec.suggested_alternatives) == 1


class TestFailurePreprocessor:
    """Test failure preprocessing"""

    @pytest.fixture
    def preprocessor(self):
        return FailurePreprocessor()

    @pytest.fixture
    def sample_null_result(self):
        return NullResult(
            attempt_id="preprocess_001",
            timestamp=datetime.now(),
            problem_type="optimization",
            approach_type="newton_method",
            constraints=["x > 0", "y < 10"],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message="Optimization failed due to numerical instability in gradient computation",
            state={"iteration": 50, "error_magnitude": 0.05},
            iteration=50,
            resources_used={"cpu": 80.0, "memory": 200.0}
        )

    def test_extract_features(self, preprocessor, sample_null_result):
        """Test feature extraction"""
        features = preprocessor.extract_features(sample_null_result)

        assert features.attempt_id == "preprocess_001"
        assert features.problem_type == "optimization"
        assert features.approach_type == "newton_method"
        assert features.iteration == 50
        assert isinstance(features.feature_vector, np.ndarray)
        assert len(features.keywords) > 0

    def test_extract_keywords(self, preprocessor):
        """Test keyword extraction"""
        error_msg = "Optimization failed due to numerical instability in gradient computation convergence"

        keywords = preprocessor._extract_keywords(error_msg, top_k=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(isinstance(k, str) for k in keywords)

    def test_compute_time_to_failure(self, preprocessor):
        """Test time to failure computation"""
        nr = NullResult(
            attempt_id="test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=100,
            resources_used={}
        )

        time = preprocessor._compute_time_to_failure(nr)

        assert time == 100.0

    def test_compute_error_magnitude(self, preprocessor):
        """Test error magnitude computation"""
        nr1 = NullResult(
            attempt_id="test1",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            error_message="Test",
            state={"error_magnitude": 0.5},
            iteration=1,
            resources_used={}
        )

        mag1 = preprocessor._compute_error_magnitude(nr1)
        assert mag1 == 0.5

        nr2 = NullResult(
            attempt_id="test2",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.DIVERGENCE,
            error_message="Test",
            state={"objective_value": -10.5},
            iteration=1,
            resources_used={}
        )

        mag2 = preprocessor._compute_error_magnitude(nr2)
        assert mag2 == 10.5

    def test_compute_resource_usage(self, preprocessor):
        """Test resource usage computation"""
        nr = NullResult(
            attempt_id="test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={"cpu": 100.0, "memory": 100.0}
        )

        usage = preprocessor._compute_resource_usage(nr)

        assert usage == 1.0  # (100 + 100) / 200 = 1.0


class TestAnomalyDetector:
    """Test anomaly detection"""

    @pytest.fixture
    def detector(self):
        return AnomalyDetector(contamination=0.1)

    @pytest.fixture
    def sample_failures(self):
        """Create sample failure features"""
        failures = []
        for i in range(20):
            features = FailureFeatures(
                attempt_id=f"anomaly_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=0.1,
                resource_consumption=0.5,
                constraint_violation_count=1,
                feature_vector=np.random.randn(5)
            )
            failures.append(features)

        # Add one anomaly
        anomaly = FailureFeatures(
            attempt_id="anomaly_outlier",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            iteration=1000,
            time_to_failure=10000.0,
            error_magnitude=10.0,
            resource_consumption=1.0,
            constraint_violation_count=10,
            feature_vector=np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        )
        failures.append(anomaly)

        return failures

    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = AnomalyDetector(contamination=0.15)
        assert detector.contamination == 0.15

    def test_detect_anomalies(self, detector, sample_failures):
        """Test anomaly detection"""
        scores = detector.detect_anomalies(sample_failures)

        assert isinstance(scores, dict)
        assert len(scores) == len(sample_failures)
        assert all(isinstance(score, float) for score in scores.values())
        assert all(0 <= score <= 1 for score in scores.values())

    def test_detect_anomalies_insufficient_data(self, detector):
        """Test anomaly detection with insufficient data"""
        failures = [
            FailureFeatures(
                attempt_id="test",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.TIMEOUT,
                iteration=1,
                time_to_failure=1.0,
                error_magnitude=None,
                resource_consumption=0.1,
                constraint_violation_count=0,
                feature_vector=np.array([1.0])
            )
        ]

        scores = detector.detect_anomalies(failures)

        # Should return default scores
        assert len(scores) == 1
        assert scores["test"] == 0.0


class TestFailureClusterer:
    """Test failure clustering"""

    @pytest.fixture
    def clusterer(self):
        return FailureClusterer(
            n_clusters_range=(2, 5),
            dbscan_eps=0.5,
            dbscan_min_samples=3
        )

    @pytest.fixture
    def sample_failures_for_clustering(self):
        """Create failures suitable for clustering"""
        failures = []

        # Cluster 1: Similar failures
        for i in range(10):
            features = FailureFeatures(
                attempt_id=f"cluster1_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="gradient_descent",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=50 + i,
                time_to_failure=100.0,
                error_magnitude=0.5,
                resource_consumption=0.6,
                constraint_violation_count=2,
                feature_vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0]) + np.random.randn(5) * 0.1
            )
            failures.append(features)

        # Cluster 2: Different failures
        for i in range(8):
            features = FailureFeatures(
                attempt_id=f"cluster2_{i}",
                timestamp=datetime.now(),
                problem_type="constraint",
                approach_type="linear_programming",
                error_type=ErrorType.INFEASIBILITY,
                iteration=30 + i,
                time_to_failure=50.0,
                error_magnitude=1.0,
                resource_consumption=0.4,
                constraint_violation_count=5,
                feature_vector=np.array([0.0, 1.0, 0.0, 0.0, 0.0]) + np.random.randn(5) * 0.1
            )
            failures.append(features)

        return failures

    def test_clusterer_initialization(self):
        """Test clusterer initialization"""
        clusterer = FailureClusterer(n_clusters_range=(3, 8))
        assert clusterer.n_clusters_range == (3, 8)

    def test_cluster_failures(self, clusterer, sample_failures_for_clustering):
        """Test failure clustering"""
        clusters = clusterer.cluster_failures(sample_failures_for_clustering)

        assert isinstance(clusters, list)
        # Should have found at least one cluster
        assert len(clusters) >= 1

        if len(clusters) > 0:
            cluster = clusters[0]
            assert hasattr(cluster, 'cluster_id')
            assert hasattr(cluster, 'size')
            assert hasattr(cluster, 'centroid')
            assert cluster.size > 0

    def test_cluster_failures_insufficient_data(self, clusterer):
        """Test clustering with insufficient data"""
        failures = [
            FailureFeatures(
                attempt_id="test",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.TIMEOUT,
                iteration=1,
                time_to_failure=1.0,
                error_magnitude=None,
                resource_consumption=0.1,
                constraint_violation_count=0,
                feature_vector=np.array([1.0])
            )
        ]

        clusters = clusterer.cluster_failures(failures)

        # Should return empty list
        assert clusters == []

    def test_cluster_quality_checks(self):
        """Test cluster quality metrics"""
        # Create a good cluster
        failures = []
        for i in range(10):
            features = FailureFeatures(
                attempt_id=f"quality_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=50,
                time_to_failure=100.0,
                error_magnitude=0.5,
                resource_consumption=0.6,
                constraint_violation_count=2,
                feature_vector=np.array([1.0, 0.0, 0.0]) + np.random.randn(3) * 0.05
            )
            failures.append(features)

        cluster = FailureCluster(
            cluster_id=1,
            size=10,
            failures=failures,
            centroid=np.mean([f.feature_vector for f in failures], axis=0),
            compactness=0.3,
            silhouette_score=0.7,
            stability=0.8,
            common_problem_types=["test"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=[],
            keywords=["test"]
        )

        # Check if it's a good candidate
        is_candidate = cluster.is_candidate_for_assumption_mining(
            min_size=5,
            max_compactness=0.5,
            min_stability=0.7
        )

        assert is_candidate is True

        # Check with stricter criteria
        is_candidate_strict = cluster.is_candidate_for_assumption_mining(
            min_size=15,
            max_compactness=0.2,
            min_stability=0.9
        )

        assert is_candidate_strict is False


class TestAssumptionGenerator:
    """Test assumption generation"""

    @pytest.fixture
    def generator(self):
        return AssumptionGenerator()

    def test_generate_assumptions_from_cluster(self, generator):
        """Test generating assumptions from a cluster"""
        # Create a cluster
        failures = []
        for i in range(10):
            features = FailureFeatures(
                attempt_id=f"gen_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="test",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=50,
                time_to_failure=100.0,
                error_magnitude=0.5,
                resource_consumption=0.6,
                constraint_violation_count=2,
                feature_vector=np.array([1.0, 0.0])
            )
            failures.append(features)

        cluster = FailureCluster(
            cluster_id=1,
            size=10,
            failures=failures,
            centroid=np.array([0.5]),
            compactness=0.3,
            silhouette_score=0.7,
            stability=0.8,
            common_problem_types=["optimization"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=[],
            keywords=["test"]
        )

        candidates = generator.generate_assumptions(cluster)

        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert isinstance(candidates[0], AssumptionCandidate)
            assert candidates[0].pattern_type == PatternType.SYSTEMATIC_VIOLATION

    def test_generate_assumptions_small_cluster(self, generator):
        """Test generating assumptions from small cluster"""
        failures = [
            FailureFeatures(
                attempt_id="small",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.TIMEOUT,
                iteration=1,
                time_to_failure=1.0,
                error_magnitude=None,
                resource_consumption=0.1,
                constraint_violation_count=0,
                feature_vector=np.array([1.0])
            )
        ]

        cluster = FailureCluster(
            cluster_id=1,
            size=1,
            failures=failures,
            centroid=np.array([1.0]),
            compactness=0.0,
            silhouette_score=0.0,
            stability=0.0,
            common_problem_types=["test"],
            common_error_types=[ErrorType.TIMEOUT],
            common_constraints=[],
            keywords=[]
        )

        candidates = generator.generate_assumptions(cluster)

        # Should not generate from small cluster
        assert len(candidates) == 0


class TestConfidenceScorer:
    """Test confidence scoring"""

    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()

    def test_score_assumption(self, scorer):
        """Test scoring an assumption"""
        # Create cluster
        failures = []
        for i in range(10):
            features = FailureFeatures(
                attempt_id=f"score_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=50,
                time_to_failure=100.0,
                error_magnitude=0.5,
                resource_consumption=0.6,
                constraint_violation_count=2,
                feature_vector=np.array([1.0, 0.0])
            )
            failures.append(features)

        cluster = FailureCluster(
            cluster_id=1,
            size=10,
            failures=failures,
            centroid=np.array([0.5]),
            compactness=0.3,
            silhouette_score=0.7,
            stability=0.8,
            common_problem_types=["test"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=[],
            keywords=[]
        )

        # Create candidate
        candidate = AssumptionCandidate(
            description="Test assumption",
            explains_failures=[f.attempt_id for f in failures[:7]],
            confidence=0.0,
            pattern_type=PatternType.SYSTEMATIC_VIOLATION,
            complexity=1,
            contradiction_count=0,
            testable=True
        )

        score = scorer.score_assumption(candidate, cluster, [])

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_score_weights(self, scorer):
        """Test scorer weights"""
        assert scorer.weights['support'] == 0.25
        assert scorer.weights['pattern'] == 0.20
        assert sum(scorer.weights.values()) == pytest.approx(1.0, rel=0.01)


class TestParadigmShiftDetector:
    """Test paradigm shift detection"""

    @pytest.fixture
    def detector(self):
        return ParadigmShiftDetector(crisis_threshold=0.7)

    def test_detect_no_crisis(self, detector):
        """Test when no crisis is detected"""
        assumptions = [
            TacitAssumption(
                id=f"assump_{i}",
                description=f"Minor assumption {i}",
                formalization=f"minor_{i}",
                assumption_type=AssumptionType.METHODOLOGICAL,
                confidence=0.5,
                support=2,
                evidence=[],
                pattern_type=PatternType.REPEATED_FAILURE,
                constraint_relaxation=f"Relax {i}",
                paradigm_implication=False,
                alternative_paradigm=None
            )
            for i in range(5)
        ]

        recommendation = detector.detect_crisis(assumptions, [])

        assert recommendation.trigger is False
        assert recommendation.confidence < 0.7
        assert len(recommendation.primary_assumptions) == 0

    def test_detect_crisis(self, detector):
        """Test when crisis is detected"""
        # Create many high-confidence paradigm-implication assumptions
        assumptions = []
        for i in range(15):
            assumption = TacitAssumption(
                id=f"crisis_{i}",
                description=f"Major paradigm challenge {i}",
                formalization=f"paradigm_{i}",
                assumption_type=AssumptionType.ONTOLOGICAL,
                confidence=0.8,
                support=10,
                evidence=[],
                pattern_type=PatternType.CROSS_DOMAIN_FAILURE,
                constraint_relaxation=f"Relax {i}",
                paradigm_implication=True,
                alternative_paradigm=f"Alternative {i}"
            )
            # Set recent timestamp
            assumption.timestamp = datetime.now()
            assumptions.append(assumption)

        recommendation = detector.detect_crisis(assumptions, [])

        assert recommendation.trigger is True
        assert recommendation.confidence >= 0.7
        assert len(recommendation.primary_assumptions) > 0


class TestPhi15Engine:
    """Test main Φ₁.₅ engine"""

    @pytest.fixture
    def engine(self):
        return Phi15Engine()

    @pytest.fixture
    def sample_null_results(self):
        """Create sample null results"""
        results = []
        for i in range(20):
            nr = NullResult(
                attempt_id=f"engine_test_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="gradient_descent",
                constraints=[f"x{i} > 0"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Failure {i}",
                state={"iteration": i * 10},
                iteration=i * 10,
                resources_used={"cpu": float(i * 10), "memory": float(i * 20)}
            )
            results.append(nr)
        return results

    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = Phi15Engine()

        assert engine.preprocessor is not None
        assert engine.anomaly_detector is not None
        assert engine.clusterer is not None
        assert len(engine.failures) == 0
        assert len(engine.assumptions) == 0

    def test_engine_custom_config(self):
        """Test engine with custom config"""
        config = {
            'confidence_threshold': 0.8,
            'crisis_threshold': 0.9,
            'min_failures_for_clustering': 20,
            'anomaly_contamination': 0.15
        }

        engine = Phi15Engine(config)

        assert engine.config['confidence_threshold'] == 0.8
        assert engine.config['crisis_threshold'] == 0.9

    def test_process_null_results(self, engine, sample_null_results):
        """Test processing null results"""
        assumptions, paradigm_rec = engine.process_null_results(sample_null_results)

        assert isinstance(assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)

        # Check failures were added
        assert len(engine.failures) == len(sample_null_results)

    def test_get_top_assumptions(self, engine, sample_null_results):
        """Test getting top assumptions"""
        assumptions, _ = engine.process_null_results(sample_null_results)

        top_k = engine.get_top_assumptions(k=5)

        assert isinstance(top_k, list)
        assert len(top_k) <= 5

        # Check sorted by confidence
        if len(top_k) > 1:
            for i in range(len(top_k) - 1):
                assert top_k[i].confidence >= top_k[i + 1].confidence

    def test_save_and_load_state(self, engine, sample_null_results):
        """Test saving and loading engine state"""
        # Process some data
        assumptions, paradigm_rec = engine.process_null_results(sample_null_results)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            engine.save_state(temp_path)

            # Load new engine
            engine2 = Phi15Engine()
            engine2.load_state(temp_path)

            # Check assumptions restored
            assert len(engine2.assumptions) == len(engine.assumptions)
            assert len(engine2.paradigm_history) == len(engine.paradigm_history)

        finally:
            # Cleanup
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_classify_assumption_type(self, engine):
        """Test assumption type classification"""
        # Constraint type
        type1 = engine._classify_assumption_type("The system must satisfy x > 0")
        assert type1 == AssumptionType.CONSTRAINT

        # Methodological type (use 'method' without 'should' to avoid constraint classification)
        type2 = engine._classify_assumption_type("The method uses gradient descent")
        assert type2 == AssumptionType.METHODOLOGICAL

        # Representational type
        type3 = engine._classify_assumption_type("We model this as a linear system")
        assert type3 == AssumptionType.REPRESENTATIONAL

        # Ontological type (default)
        type4 = engine._classify_assumption_type("Some general statement")
        assert type4 == AssumptionType.ONTOLOGICAL

    def test_formalize_assumption(self, engine):
        """Test assumption formalization"""
        description = "The system must maintain x > 0"

        formalization = engine._formalize_assumption(description)

        assert isinstance(formalization, str)
        assert "forall" in formalization

    def test_generate_assumption_id(self, engine):
        """Test assumption ID generation"""
        id1 = engine._generate_assumption_id("Test description 1")
        id2 = engine._generate_assumption_id("Test description 2")

        assert id1.startswith("assumption_")
        assert id2.startswith("assumption_")
        assert id1 != id2
        assert len(id1) == len("assumption_") + 8  # 8 char hash


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_phi15_engine(self):
        """Test create_phi15_engine function"""
        engine = create_phi15_engine()

        assert isinstance(engine, Phi15Engine)
        assert engine.config is not None


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_null_results(self):
        """Test processing empty list"""
        engine = Phi15Engine()

        assumptions, paradigm_rec = engine.process_null_results([])

        assert assumptions == []
        assert paradigm_rec.trigger is False

    def test_single_null_result(self):
        """Test processing single null result"""
        engine = Phi15Engine()

        nr = NullResult(
            attempt_id="single",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )

        assumptions, paradigm_rec = engine.process_null_results([nr])

        # Should not crash
        assert isinstance(assumptions, list)

    def test_large_dataset(self):
        """Test processing larger dataset"""
        engine = Phi15Engine()

        results = []
        for i in range(100):
            nr = NullResult(
                attempt_id=f"large_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="test",
                constraints=[],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Error {i}",
                state={},
                iteration=i,
                resources_used={"cpu": float(i), "memory": float(i * 2)}
            )
            results.append(nr)

        assumptions, paradigm_rec = engine.process_null_results(results)

        # Should handle large dataset
        assert len(engine.failures) == 100

    def test_none_values(self):
        """Test handling of None values"""
        nr = NullResult(
            attempt_id="none_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.UNKNOWN_FAILURE,
            error_message="Unknown error",
            state={},
            iteration=1,
            resources_used={}
        )

        preprocessor = FailurePreprocessor()
        features = preprocessor.extract_features(nr)

        # Should handle None values gracefully
        assert features.error_magnitude is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
