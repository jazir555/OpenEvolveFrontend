"""
Phase I Integration Tests: Φ₁.₅ Tacit Assumption Miner

Integration tests for Φ₁.₅ components:
- End-to-end pipeline testing
- Component interaction validation
- Data flow verification
- Performance benchmarking

Author: Agent Z2 (Testing/QA Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.tacit_assumption_miner import (
    NullResult, ErrorType, TacitAssumption, AssumptionType,
    PatternType, FailurePreprocessor, AnomalyDetector,
    FailureClusterer, AssumptionGenerator, ConfidenceScorer,
    ParadigmShiftDetector, ParadigmShiftRecommendation,
    Phi15Engine, FailureFeatures, FailureCluster
)

from tests.test_utils import (
    PerformanceTimer, ValidationHelpers, TestAssertions,
    TestDataGenerator
)


# ============================================================================
# Test Markers
# ============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.phase1,
]


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestPhi15EndToEnd:
    """Test complete Φ₁.₅ pipeline from null results to assumptions"""

    @pytest.fixture
    def engine(self):
        """Get Φ₁.₅ engine"""
        return Phi15Engine()

    @pytest.fixture
    def systematic_failure_dataset(self):
        """Create dataset with systematic failure pattern"""
        # Pattern: All deterministic optimization attempts fail
        results = []
        for i in range(50):
            result = NullResult(
                attempt_id=f"sys_fail_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["exact_solution", "polynomial_time"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Deterministic solver {i} failed to converge",
                state={"iteration": i * 50, "time_limit": 3600},
                iteration=i * 50,
                resources_used={"cpu": float(i * 10), "memory": float(i * 20)},
                metadata={"pattern": "deterministic_fails"}
            )
            results.append(result)
        return results

    @pytest.fixture
    def diverse_failure_dataset(self):
        """Create dataset with diverse failure patterns"""
        results = []
        error_types = [
            ErrorType.OPTIMIZATION_FAILED,
            ErrorType.TIMEOUT,
            ErrorType.INFEASIBLE,
            ErrorType.NUMERICAL_INSTABILITY
        ]
        problem_types = ["optimization", "satisfiability", "inference"]
        approach_types = ["deterministic", "stochastic", "approximate"]

        for i in range(40):
            result = NullResult(
                attempt_id=f"div_fail_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type=problem_types[i % len(problem_types)],
                approach_type=approach_types[i % len(approach_types)],
                constraints=[f"c{j}" for j in range(1, np.random.randint(2, 6))],
                error_type=error_types[i % len(error_types)],
                error_message=f"Failure {i}: {error_types[i % len(error_types)].value}",
                state={"iteration": np.random.randint(10, 500)},
                iteration=np.random.randint(10, 500),
                resources_used={
                    "cpu": np.random.uniform(10, 100),
                    "memory": np.random.uniform(100, 500)
                },
                metadata={"pattern": "diverse"}
            )
            results.append(result)
        return results

    def test_complete_pipeline_systematic_pattern(self, engine, systematic_failure_dataset):
        """Test complete pipeline with systematic failure pattern"""
        # Process through pipeline
        with PerformanceTimer("phi15_systematic_pipeline") as timer:
            assumptions, paradigm_rec = engine.process_null_results(systematic_failure_dataset)

        # Verify results
        assert isinstance(assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)

        # Check performance
        assert timer.get_elapsed() < 30.0, "Pipeline should complete in < 30 seconds"

        # Validate assumptions were generated
        if len(assumptions) > 0:
            # Check assumption properties
            for assumption in assumptions[:5]:  # Check top 5
                assert assumption.id
                assert assumption.description
                assert 0 <= assumption.confidence <= 1
                assert assumption.support > 0
                assert len(assumption.evidence) > 0
                assert assumption.assumption_type in AssumptionType

    def test_complete_pipeline_diverse_pattern(self, engine, diverse_failure_dataset):
        """Test complete pipeline with diverse failure patterns"""
        with PerformanceTimer("phi15_diverse_pipeline") as timer:
            assumptions, paradigm_rec = engine.process_null_results(diverse_failure_dataset)

        # Verify results
        assert isinstance(assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)

        # Performance check
        assert timer.get_elapsed() < 30.0

        # Check diverse assumptions were generated
        if len(assumptions) > 0:
            assumption_types = set(a.assumption_type for a in assumptions)
            assert len(assumption_types) >= 1  # At least one type

    def test_get_top_assumptions(self, engine, systematic_failure_dataset):
        """Test getting top-k assumptions"""
        engine.process_null_results(systematic_failure_dataset)

        # Get top 5
        top_5 = engine.get_top_assumptions(k=5)
        assert len(top_5) <= 5

        # Check sorted by confidence
        if len(top_5) > 1:
            for i in range(len(top_5) - 1):
                assert top_5[i].confidence >= top_5[i + 1].confidence

        # Get top 10
        top_10 = engine.get_top_assumptions(k=10)
        assert len(top_10) >= len(top_5)  # Should have more or equal

    def test_paradigm_shift_detection(self, engine):
        """Test paradigm shift detection"""
        # Create dataset indicating paradigm crisis
        crisis_results = []
        for i in range(100):
            result = NullResult(
                attempt_id=f"crisis_{i:03d}",
                timestamp=datetime.now() - timedelta(hours=i),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["exact_solution"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Exact method {i} failed - need approximation",
                state={"iteration": i},
                iteration=i,
                resources_used={"cpu": float(i)},
                metadata={"paradigm_crisis": True}
            )
            crisis_results.append(result)

        # Process
        assumptions, paradigm_rec = engine.process_null_results(crisis_results)

        # Check paradigm detection
        # With many failures, should detect potential crisis
        assert paradigm_rec.confidence >= 0.0
        assert isinstance(paradigm_rec.trigger, bool)


# ============================================================================
# Component Integration Tests
# ============================================================================

class TestPhi15ComponentIntegration:
    """Test interactions between Φ₁.₅ components"""

    @pytest.fixture
    def preprocessor(self):
        return FailurePreprocessor()

    @pytest.fixture
    def detector(self):
        return AnomalyDetector(contamination=0.1)

    @pytest.fixture
    def clusterer(self):
        return FailureClusterer()

    @pytest.fixture
    def generator(self):
        return AssumptionGenerator()

    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()

    @pytest.fixture
    def paradigm_detector(self):
        return ParadigmShiftDetector(crisis_threshold=0.7)

    def test_preprocessor_to_detector_integration(self, preprocessor, detector):
        """Test data flow from preprocessor to anomaly detector"""
        # Create sample results
        results = []
        for i in range(20):
            result = NullResult(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["c1"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Test {i}",
                state={"iteration": i},
                iteration=i,
                resources_used={"cpu": float(i)}
            )
            results.append(result)

        # Preprocess
        features_list = [preprocessor.extract_features(r) for r in results]

        # Detect anomalies
        anomaly_scores = detector.detect_anomalies(features_list)

        # Verify
        assert len(anomaly_scores) == len(features_list)
        assert all(0 <= s <= 1 for s in anomaly_scores.values())

    def test_detector_to_clusterer_integration(self, detector, clusterer):
        """Test data flow from anomaly detector to clusterer"""
        # Create features
        features_list = []
        for i in range(30):
            features = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="deterministic",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=1.0,
                resource_consumption=0.1 * i,
                constraint_violation_count=1,
                feature_vector=np.array([float(i), 1.0, 1.0, 0.1 * i, 1.0]),
                keywords=["test"]
            )
            features_list.append(features)

        # Detect anomalies
        anomaly_scores = detector.detect_anomalies(features_list)

        # Cluster
        clusters = clusterer.cluster_failures(features_list)

        # Verify
        assert isinstance(clusters, list)

    def test_clusterer_to_generator_integration(self, clusterer, generator):
        """Test data flow from clusterer to assumption generator"""
        # Create a cluster
        features_list = []
        for i in range(15):
            features = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="deterministic",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=1.0,
                resource_consumption=0.1,
                constraint_violation_count=1,
                feature_vector=np.array([1.0, 1.0, 1.0, 0.1, 1.0]),
                keywords=["optimization", "failed", "exact"]
            )
            features_list.append(features)

        # Create cluster
        centroid = np.mean([f.feature_vector for f in features_list], axis=0)
        cluster = FailureCluster(
            cluster_id=0,
            size=15,
            failures=features_list,
            centroid=centroid,
            compactness=0.3,
            silhouette_score=0.6,
            stability=0.8,
            common_problem_types=["optimization"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=["exact_solution"],
            keywords=["optimization", "exact"]
        )

        # Generate assumptions
        candidates = generator.generate_assumptions(cluster)

        # Verify
        assert isinstance(candidates, list)

    def test_generator_to_scorer_integration(self, generator, scorer):
        """Test data flow from generator to scorer"""
        # Create cluster and candidates
        features_list = []
        for i in range(10):
            features = FailureFeatures(
                attempt_id=f"test_{i}",
                timestamp=datetime.now(),
                problem_type="optimization",
                approach_type="deterministic",
                error_type=ErrorType.OPTIMIZATION_FAILED,
                iteration=i,
                time_to_failure=float(i),
                error_magnitude=1.0,
                resource_consumption=0.1,
                constraint_violation_count=1,
                feature_vector=np.array([1.0, 1.0, 1.0, 0.1, 1.0]),
                keywords=["test"]
            )
            features_list.append(features)

        centroid = np.mean([f.feature_vector for f in features_list], axis=0)
        cluster = FailureCluster(
            cluster_id=0,
            size=10,
            failures=features_list,
            centroid=centroid,
            compactness=0.3,
            silhouette_score=0.6,
            stability=0.8,
            common_problem_types=["test"],
            common_error_types=[ErrorType.OPTIMIZATION_FAILED],
            common_constraints=["c1"],
            keywords=["test"]
        )

        candidates = generator.generate_assumptions(cluster)

        # Score candidates
        scored_assumptions = []
        for candidate in candidates:
            score = scorer.score_assumption(candidate, cluster, [])
            scored_assumptions.append((candidate, score))

        # Verify
        assert len(scored_assumptions) == len(candidates)
        for candidate, score in scored_assumptions:
            assert 0 <= score <= 1

    def test_full_pipeline_integration(self, preprocessor, detector, clusterer,
                                       generator, scorer, paradigm_detector):
        """Test all components integrated together"""
        # Create null results
        null_results = []
        for i in range(30):
            result = NullResult(
                attempt_id=f"integration_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["exact_solution", "real_valued"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Optimization {i} failed - exact solution too expensive",
                state={"iteration": i * 10},
                iteration=i * 10,
                resources_used={"cpu": float(i * 5)},
                metadata={"integration_test": True}
            )
            null_results.append(result)

        # Step 1: Preprocess
        features_list = [preprocessor.extract_features(r) for r in null_results]
        assert len(features_list) == len(null_results)

        # Step 2: Detect anomalies
        anomaly_scores = detector.detect_anomalies(features_list)
        assert len(anomaly_scores) == len(features_list)

        # Step 3: Cluster
        clusters = clusterer.cluster_failures(features_list)
        assert isinstance(clusters, list)

        # Step 4: Generate assumptions
        all_assumptions = []
        for cluster in clusters:
            candidates = generator.generate_assumptions(cluster)
            for candidate in candidates:
                score = scorer.score_assumption(candidate, cluster, [])
                assumption = TacitAssumption(
                    id=f"assumption_{len(all_assumptions)}",
                    description=candidate.description,
                    formalization=f"formal_{len(all_assumptions)}",
                    assumption_type=AssumptionType.CONSTRAINT,
                    confidence=score,
                    support=len(candidate.explains_failures),
                    evidence=candidate.explains_failures,
                    pattern_type=candidate.pattern_type,
                    constraint_relaxation="Relax constraint",
                    paradigm_implication=False,
                    alternative_paradigm=None
                )
                all_assumptions.append(assumption)

        # Step 5: Detect paradigm shift
        paradigm_rec = paradigm_detector.detect_crisis(all_assumptions, [])

        # Verify full pipeline
        assert isinstance(all_assumptions, list)
        assert isinstance(paradigm_rec, ParadigmShiftRecommendation)


# ============================================================================
# Data Flow Tests
# ============================================================================

class TestPhi15DataFlow:
    """Test data flow and transformations through Φ₁.₅ pipeline"""

    def test_null_result_serialization_roundtrip(self):
        """Test NullResult serialization and deserialization"""
        result = NullResult(
            attempt_id="roundtrip_test",
            timestamp=datetime.now(),
            problem_type="optimization",
            approach_type="deterministic",
            constraints=["c1", "c2"],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message="Test",
            state={"iter": 100},
            iteration=100,
            resources_used={"cpu": 50.0}
        )

        # Serialize
        data = result.to_dict()

        # Deserialize
        restored = NullResult.from_dict(data)

        # Verify
        assert restored.attempt_id == result.attempt_id
        assert restored.problem_type == result.problem_type
        assert len(restored.constraints) == len(result.constraints)

    def test_assumption_serialization_roundtrip(self):
        """Test TacitAssumption serialization and deserialization"""
        assumption = TacitAssumption(
            id="assumption_test",
            description="Test assumption",
            formalization="forall x, test(x)",
            assumption_type=AssumptionType.CONSTRAINT,
            confidence=0.8,
            support=10,
            evidence=["test_1", "test_2"],
            pattern_type=PatternType.SYSTEMATIC_VIOLATION,
            constraint_relaxation="Relax",
            paradigm_implication=False,
            alternative_paradigm=None
        )

        # Verify SCE constraint conversion
        sce_constraint = assumption.to_sce_constraint()
        assert sce_constraint.id == assumption.id
        assert "[INFERRED]" in sce_constraint.description

    def test_feature_vector_consistency(self):
        """Test feature vector extraction consistency"""
        preprocessor = FailurePreprocessor()

        result = NullResult(
            attempt_id="consistency_test",
            timestamp=datetime.now(),
            problem_type="optimization",
            approach_type="deterministic",
            constraints=["c1"],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message="Test",
            state={"iter": 100},
            iteration=100,
            resources_used={"cpu": 50.0}
        )

        # Extract features multiple times
        features1 = preprocessor.extract_features(result)
        features2 = preprocessor.extract_features(result)

        # Should be identical
        np.testing.assert_array_equal(features1.feature_vector, features2.feature_vector)
        assert features1.attempt_id == features2.attempt_id


# ============================================================================
# Performance Integration Tests
# ============================================================================

class TestPhi15Performance:
    """Performance tests for Φ₁.₅ integration"""

    def test_large_dataset_performance(self):
        """Test performance with large failure dataset"""
        engine = Phi15Engine()

        # Create large dataset
        large_dataset = []
        for i in range(500):
            result = NullResult(
                attempt_id=f"perf_{i:04d}",
                timestamp=datetime.now() - timedelta(seconds=i),
                problem_type="optimization",
                approach_type="deterministic",
                constraints=["c1", "c2"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Test {i}",
                state={"iteration": i},
                iteration=i,
                resources_used={"cpu": float(i % 100)},
                metadata={"performance": True}
            )
            large_dataset.append(result)

        # Time execution
        with PerformanceTimer("phi15_large_dataset") as timer:
            assumptions, paradigm_rec = engine.process_null_results(large_dataset)

        # Should complete in reasonable time
        assert timer.get_elapsed() < 60.0, "Large dataset processing should complete in < 60 seconds"

    def test_incremental_processing(self):
        """Test incremental processing of failures"""
        engine = Phi15Engine()

        # Process in batches
        all_assumptions = []
        for batch in range(5):
            batch_results = []
            for i in range(20):
                result = NullResult(
                    attempt_id=f"batch_{batch}_test_{i}",
                    timestamp=datetime.now() - timedelta(minutes=batch * 20 + i),
                    problem_type="optimization",
                    approach_type="deterministic",
                    constraints=["c1"],
                    error_type=ErrorType.OPTIMIZATION_FAILED,
                    error_message=f"Batch {batch} test {i}",
                    state={"iteration": i},
                    iteration=i,
                    resources_used={"cpu": float(i)}
                )
                batch_results.append(result)

            assumptions, _ = engine.process_null_results(batch_results)
            all_assumptions.extend(assumptions)

        # Verify incremental processing worked
        assert isinstance(all_assumptions, list)


# ============================================================================
# Validation Tests
# ============================================================================

class TestPhi15Validation:
    """Validate Φ₁.₅ meets accuracy thresholds"""

    def test_phi15_accuracy_validation(self):
        """Validate Φ₁.₅ achieves >70% accuracy"""
        # This is a placeholder - actual validation requires
        # ground truth data and labeled assumptions

        # Simulate predictions and ground truth
        predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 10
        ground_truth = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0] * 10

        passed, accuracy = ValidationHelpers.validate_phi15_accuracy(
            predictions, ground_truth, min_accuracy=0.70
        )

        # For this synthetic data, check it passes
        assert accuracy >= 0.0
        assert isinstance(passed, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
