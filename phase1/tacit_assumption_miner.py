"""
Phi 1.5 Tacit Assumption Miner

Automated Kuhnian paradigm shift detection that mines hidden constraints
from null results and failure patterns in the RESE framework.

Author: Agent B1 (Phi 1/Phi 1.5 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
Target: >70% assumption mining accuracy

Core Innovation:
Transform null results from "failures" into "paradigm shift signals" by
systematically mining tacit assumptions that researchers unknowingly make.
"""

# Note: Unicode characters (Phi 1.5) replaced with ASCII for compatibility

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import json
from pathlib import Path
import hashlib

# Import SCE for constraint conversion
import sys
sys.path.append(str(Path(__file__).parent.parent / "core"))
from symbolic_constraint_engine import Constraint, ConstraintType


# ============================================================================
# Enums
# ============================================================================

class ErrorType(Enum):
    """Types of errors from Stage 6 Error Source Analysis"""
    OPTIMIZATION_FAILED = "optimization_failed"
    DIVERGENCE = "divergence"
    CYCLE_DETECTION = "cycle_detection"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    NUMERICAL_INSTABILITY = "numerical_instability"
    INFEASIBILITY = "infeasibility"
    UNKNOWN_FAILURE = "unknown_failure"


class AssumptionType(Enum):
    """Categories of tacit assumptions"""
    ONTOLOGICAL = "ontological"  # About what exists
    METHODOLOGICAL = "methodological"  # About how to solve
    CONSTRAINT = "constraint"  # Hidden constraints
    REPRESENTATIONAL = "representational"  # About modeling


class PatternType(Enum):
    """Types of failure patterns"""
    REPEATED_FAILURE = "repeated_failure"
    SYSTEMATIC_VIOLATION = "systematic_violation"
    CONVERGENCE_TO_BOUNDARY = "convergence_to_boundary"
    CROSS_DOMAIN_FAILURE = "cross_domain_failure"
    SCALE_DEPENDENT = "scale_dependent"


# ============================================================================
# Input Structures (from Stage 6)
# ============================================================================

@dataclass
class NullResult:
    """
    Null result from Stage 6 Error Source Analysis.

    This is the primary input to Φ₁.₅, representing a failed attempt.

    Attributes:
        attempt_id: Unique identifier for the attempt
        timestamp: When the attempt occurred
        problem_type: Type of problem being solved
        approach_type: Approach/algorithm used
        constraints: Explicit constraints applied
        error_type: Type of error/failure
        error_message: Human-readable error description
        state: Final state when failure occurred
        iteration: Iteration number when failed
        resources_used: Computational resources consumed
        metadata: Additional context
    """
    attempt_id: str
    timestamp: datetime
    problem_type: str
    approach_type: str
    constraints: List[str]
    error_type: ErrorType
    error_message: str
    state: Dict[str, Any]
    iteration: int
    resources_used: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'attempt_id': self.attempt_id,
            'timestamp': self.timestamp.isoformat(),
            'problem_type': self.problem_type,
            'approach_type': self.approach_type,
            'constraints': self.constraints,
            'error_type': self.error_type.value,
            'error_message': self.error_message,
            'state': self.state,
            'iteration': self.iteration,
            'resources_used': self.resources_used,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NullResult':
        """Create from dictionary"""
        return cls(
            attempt_id=data['attempt_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            problem_type=data['problem_type'],
            approach_type=data['approach_type'],
            constraints=data['constraints'],
            error_type=ErrorType(data['error_type']),
            error_message=data['error_message'],
            state=data['state'],
            iteration=data['iteration'],
            resources_used=data['resources_used'],
            metadata=data.get('metadata', {})
        )


# ============================================================================
# Feature Structures
# ============================================================================

@dataclass
class FailureFeatures:
    """
    Extracted features from a null result for ML analysis.

    Attributes:
        attempt_id: Unique identifier
        timestamp: When failure occurred
        problem_type: Categorical feature
        approach_type: Categorical feature
        error_type: Categorical feature
        iteration: Numerical feature
        time_to_failure: Numerical feature (seconds)
        error_magnitude: Optional numerical feature
        resource_consumption: Numerical feature (0-1 normalized)
        constraint_violation_count: Numerical feature
        feature_vector: Concatenated feature vector for ML
        keywords: Extracted keywords from error message
        failure_cluster: Cluster assignment (filled later)
        anomaly_score: Anomaly score (filled later)
    """
    attempt_id: str
    timestamp: datetime
    problem_type: str
    approach_type: str
    error_type: ErrorType
    iteration: int
    time_to_failure: float
    error_magnitude: Optional[float]
    resource_consumption: float
    constraint_violation_count: int
    feature_vector: np.ndarray
    keywords: List[str] = field(default_factory=list)
    failure_cluster: Optional[int] = None
    anomaly_score: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'attempt_id': self.attempt_id,
            'timestamp': self.timestamp.isoformat(),
            'problem_type': self.problem_type,
            'approach_type': self.approach_type,
            'error_type': self.error_type.value,
            'iteration': self.iteration,
            'time_to_failure': self.time_to_failure,
            'error_magnitude': self.error_magnitude,
            'resource_consumption': self.resource_consumption,
            'constraint_violation_count': self.constraint_violation_count,
            'feature_vector': self.feature_vector.tolist(),
            'keywords': self.keywords,
            'failure_cluster': self.failure_cluster,
            'anomaly_score': self.anomaly_score
        }


# ============================================================================
# Cluster Structures
# ============================================================================

@dataclass
class FailureCluster:
    """
    Cluster of similar failures with quality metrics.

    Attributes:
        cluster_id: Unique cluster identifier
        size: Number of failures in cluster
        failures: List of failures in cluster
        centroid: Cluster centroid in feature space
        compactness: Mean distance to centroid (lower is better)
        silhouette_score: Cluster quality metric [-1, 1]
        stability: Stability across clustering methods [0, 1]
        common_problem_types: Most frequent problem types
        common_error_types: Most frequent error types
        common_constraints: Most frequent constraints
        keywords: Aggregated keywords from failures
    """
    cluster_id: int
    size: int
    failures: List[FailureFeatures]
    centroid: np.ndarray
    compactness: float
    silhouette_score: float
    stability: float
    common_problem_types: List[str]
    common_error_types: List[ErrorType]
    common_constraints: List[str]
    keywords: List[str]

    def is_candidate_for_assumption_mining(self,
                                          min_size: int = 5,
                                          max_compactness: float = 0.5,
                                          min_stability: float = 0.7) -> bool:
        """
        Check if cluster is worth analyzing for assumptions.

        Args:
            min_size: Minimum number of failures
            max_compactness: Maximum allowed compactness
            min_stability: Minimum stability score

        Returns:
            True if cluster meets quality criteria
        """
        return (
            self.size >= min_size and
            self.compactness <= max_compactness and
            self.stability >= min_stability
        )


# ============================================================================
# Assumption Structures
# ============================================================================

@dataclass
class AssumptionCandidate:
    """
    Candidate assumption from abductive inference.

    Attributes:
        description: Human-readable description
        explains_failures: List of attempt IDs this explains
        confidence: Initial confidence score
        pattern_type: What pattern led to this inference
        complexity: Complexity (lower is simpler, better)
        contradiction_count: Number of contradictions with known constraints
        testable: Whether this can be validated
    """
    description: str
    explains_failures: List[str]
    confidence: float
    pattern_type: PatternType
    complexity: int
    contradiction_count: int
    testable: bool


@dataclass
class TacitAssumption:
    """
    Inferred tacit assumption to add as constraint.

    This is the primary output of Φ₁.₅, representing a discovered hidden constraint.

    Attributes:
        id: Unique identifier
        description: Human-readable description
        formalization: SCE constraint format (Lean 4)
        assumption_type: Category of assumption
        confidence: Confidence score [0, 1]
        support: Number of failures explained
        evidence: IDs of supporting failures
        pattern_type: What pattern led to inference
        constraint_relaxation: How to relax this constraint
        paradigm_implication: Whether this suggests paradigm shift
        alternative_paradigm: Suggested alternative paradigm (if applicable)
        timestamp: When this assumption was inferred
        verified: Whether validated by Stage 7
    """
    id: str
    description: str
    formalization: str
    assumption_type: AssumptionType
    confidence: float
    support: int
    evidence: List[str]
    pattern_type: PatternType
    constraint_relaxation: str
    paradigm_implication: bool
    alternative_paradigm: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    verified: bool = False

    def to_sce_constraint(self) -> Constraint:
        """
        Convert to SCE Constraint format for Stage 1 integration.

        Returns:
            Constraint object compatible with SymbolicConstraintEngine
        """
        return Constraint(
            id=self.id,
            type=ConstraintType.SOFT,  # Start as soft (inferred)
            description=f"[INFERRED] {self.description}",
            formalization=self.formalization,
            source="phi15_inferred",
            dependencies=[],
            verified=self.verified,
            lean_theorem=None
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'description': self.description,
            'formalization': self.formalization,
            'assumption_type': self.assumption_type.value,
            'confidence': self.confidence,
            'support': self.support,
            'evidence': self.evidence,
            'pattern_type': self.pattern_type.value,
            'constraint_relaxation': self.constraint_relaxation,
            'paradigm_implication': self.paradigm_implication,
            'alternative_paradigm': self.alternative_paradigm,
            'timestamp': self.timestamp.isoformat(),
            'verified': self.verified
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TacitAssumption':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            description=data['description'],
            formalization=data['formalization'],
            assumption_type=AssumptionType(data['assumption_type']),
            confidence=data['confidence'],
            support=data['support'],
            evidence=data['evidence'],
            pattern_type=PatternType(data['pattern_type']),
            constraint_relaxation=data['constraint_relaxation'],
            paradigm_implication=data['paradigm_implication'],
            alternative_paradigm=data.get('alternative_paradigm'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            verified=data.get('verified', False)
        )


# ============================================================================
# Paradigm Shift Structures
# ============================================================================

@dataclass
class ParadigmShiftRecommendation:
    """
    Recommendation for paradigm shift.

    Attributes:
        trigger: Whether to recommend shift
        confidence: Confidence in recommendation [0, 1]
        primary_assumptions: Key assumptions causing crisis
        suggested_alternatives: Alternative paradigms to try
        explanation: Human-readable explanation
        timestamp: When recommendation was generated
    """
    trigger: bool
    confidence: float
    primary_assumptions: List[TacitAssumption]
    suggested_alternatives: List[str]
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for Stage 1 communication"""
        return {
            'trigger': self.trigger,
            'confidence': self.confidence,
            'assumptions_to_relax': [a.id for a in self.primary_assumptions],
            'alternative_paradigms': self.suggested_alternatives,
            'explanation': self.explanation,
            'priority': 'HIGH' if self.trigger and self.confidence > 0.8 else 'MEDIUM',
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ParadigmShiftRecommendation':
        """
        Create ParadigmShiftRecommendation from dictionary

        Args:
            data: Dictionary from to_dict()

        Returns:
            ParadigmShiftRecommendation instance
        """
        return cls(
            trigger=data['trigger'],
            confidence=data['confidence'],
            primary_assumptions=[],  # Will be reconstructed by engine
            suggested_alternatives=data.get('suggested_alternatives', data.get('alternative_paradigms', [])),
            explanation=data.get('explanation', ''),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


# ============================================================================
# Component 1: Failure Preprocessor
# ============================================================================

class FailurePreprocessor:
    """
    Preprocess null results into feature vectors for ML analysis.

    Extracts structural, temporal, numerical, and contextual features from
    failed attempts to enable pattern recognition and clustering.
    """

    def __init__(self):
        self.feature_encoder = {}
        self.stop_words = {
            'the', 'a', 'an', 'is', 'was', 'at', 'which', 'on', 'in',
            'to', 'of', 'for', 'with', 'and', 'or', 'but'
        }
        # Categorical feature encoders
        self.problem_type_map = {}
        self.approach_type_map = {}
        self.error_type_map = {}
        self.next_problem_id = 0
        self.next_approach_id = 0
        self.next_error_id = 0

    def extract_features(self, null_result: NullResult) -> FailureFeatures:
        """
        Extract features from null result.

        Args:
            null_result: Null result from Stage 6

        Returns:
            FailureFeatures with extracted features
        """
        # Categorical features - encode them
        problem_type_encoded = self._encode_categorical(
            null_result.problem_type, self.problem_type_map, 'problem', self.next_problem_id
        )
        approach_type_encoded = self._encode_categorical(
            null_result.approach_type, self.approach_type_map, 'approach', self.next_approach_id
        )
        error_type_encoded = self._encode_categorical(
            null_result.error_type.value, self.error_type_map, 'error', self.next_error_id
        )

        # Numerical features
        iteration = null_result.iteration
        time_to_failure = self._compute_time_to_failure(null_result)
        error_magnitude = self._compute_error_magnitude(null_result)
        resource_consumption = self._compute_resource_usage(null_result)
        constraint_violation_count = len(null_result.constraints)

        # Text features
        keywords = self._extract_keywords(null_result.error_message)

        # Create feature vector with properly encoded categorical features
        feature_vector = self._create_feature_vector(
            problem_type_encoded, approach_type_encoded, error_type_encoded,
            iteration, time_to_failure, error_magnitude,
            resource_consumption, constraint_violation_count
        )

        return FailureFeatures(
            attempt_id=null_result.attempt_id,
            timestamp=null_result.timestamp,
            problem_type=null_result.problem_type,
            approach_type=null_result.approach_type,
            error_type=null_result.error_type,
            iteration=iteration,
            time_to_failure=time_to_failure,
            error_magnitude=error_magnitude,
            resource_consumption=resource_consumption,
            constraint_violation_count=constraint_violation_count,
            feature_vector=feature_vector,
            keywords=keywords
        )

    def _encode_categorical(self, value: str, encoder_map: Dict,
                           feature_name: str, next_id: int) -> float:
        """Encode categorical value as float"""
        if value not in encoder_map:
            encoder_map[value] = float(next_id)
            # Increment the corresponding counter
            if feature_name == 'problem':
                self.next_problem_id += 1
            elif feature_name == 'approach':
                self.next_approach_id += 1
            elif feature_name == 'error':
                self.next_error_id += 1
        return encoder_map[value]

    def _compute_time_to_failure(self, null_result: NullResult) -> float:
        """Estimate time until failure"""
        # In real implementation, extract from resources_used or state
        # For now, use iteration as proxy
        return float(null_result.iteration)

    def _compute_error_magnitude(self, null_result: NullResult) -> Optional[float]:
        """Compute magnitude of error if applicable"""
        # Check state for error magnitude
        if 'error_magnitude' in null_result.state:
            return float(null_result.state['error_magnitude'])
        if 'objective_value' in null_result.state:
            return abs(float(null_result.state['objective_value']))
        return None

    def _compute_resource_usage(self, null_result: NullResult) -> float:
        """Compute normalized resource consumption"""
        cpu = null_result.resources_used.get('cpu', 0.0)
        memory = null_result.resources_used.get('memory', 0.0)
        # Normalize to [0, 1]
        return min(1.0, (cpu + memory) / 200.0)

    def _extract_keywords(self, error_message: str, top_k: int = 10) -> List[str]:
        """
        Extract key terms from error message.

        Args:
            error_message: Error message text
            top_k: Number of keywords to extract

        Returns:
            List of top keywords
        """
        words = error_message.lower().split()
        # Filter stop words and short words
        keywords = [w for w in words
                   if w not in self.stop_words and len(w) > 3]

        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top-k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]

    def _create_feature_vector(self, problem_type: float, approach_type: float,
                              error_type: float, iteration: int,
                              time_to_failure: float, error_magnitude: Optional[float],
                              resource_consumption: float,
                              constraint_violation_count: int) -> np.ndarray:
        """
        Create feature vector from extracted features.

        Now includes properly encoded categorical features.
        """
        # All features including encoded categoricals
        features = [
            problem_type,        # Encoded categorical
            approach_type,       # Encoded categorical
            error_type,          # Encoded categorical
            float(iteration),
            float(time_to_failure),
            error_magnitude if error_magnitude is not None else 0.0,
            resource_consumption,
            float(constraint_violation_count)
        ]

        return np.array(features, dtype=np.float32)


# ============================================================================
# Component 2: Anomaly Detector
# ============================================================================

class AnomalyDetector:
    """
    Detect anomalies in failure patterns using multiple algorithms.

    Combines Isolation Forest and Local Outlier Factor (LOF) for robust
    anomaly detection in high-dimensional failure feature space.
    """

    def __init__(self, contamination: float = 0.1,
                 isolation_weight: float = 0.5,
                 lof_weight: float = 0.5):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of outliers
            isolation_weight: Weight for Isolation Forest score
            lof_weight: Weight for LOF score
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            self.sklearn_available = True
        except ImportError:
            print("Warning: scikit-learn not available, using fallback anomaly detection")
            self.sklearn_available = False

        self.contamination = contamination
        self.isolation_weight = isolation_weight
        self.lof_weight = lof_weight

        # Initialize models
        if self.sklearn_available:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor

            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True  # Enable for predict on new data
            )

    def detect_anomalies(self, failures: List[FailureFeatures]) -> Dict[str, float]:
        """
        Detect anomalies and return scores.

        Args:
            failures: List of failure features

        Returns:
            Dictionary mapping attempt_id to overall anomaly score [0, 1]
        """
        if len(failures) < 3:
            # Not enough data for anomaly detection
            return {f.attempt_id: 0.0 for f in failures}

        # Extract feature matrix
        X = np.array([f.feature_vector for f in failures])

        if not self.sklearn_available:
            # Fallback: use Z-score based anomaly detection
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
            z_scores = np.abs((X - mean) / std)
            # Max Z-score across features, normalized to [0, 1]
            max_z = np.max(z_scores, axis=1)
            # Convert to anomaly score (cap at 3 sigma)
            anomaly_scores = np.minimum(max_z / 3.0, 1.0)
            return {f.attempt_id: float(anomaly_scores[i]) for i, f in enumerate(failures)}

        # Fit and predict with Isolation Forest - use score_samples for continuous scores
        if_scores = self.isolation_forest.fit_predict(X)
        if_raw_scores = self.isolation_forest.score_samples(X)
        # Normalize to [0, 1] where 1 = anomaly
        if_anomalies = np.where(if_scores == -1, 1.0, 0.0)
        if_normalized = (if_raw_scores - if_raw_scores.min()) / (if_raw_scores.max() - if_raw_scores.min() + 1e-8)

        # Fit and predict with LOF - use negative outlier factor
        # Note: LOF with novelty=True doesn't have fit_predict, use fit then decision_function
        self.lof.fit(X)
        lof_raw_scores = self.lof.negative_outlier_factor_
        # Convert to [0, 1] where higher = more anomalous
        # LOF returns negative scores (more negative = more anomalous)
        lof_normalized = (-lof_raw_scores - lof_raw_scores.min()) / (-lof_raw_scores.max() + lof_raw_scores.min() + 1e-8)

        # Combine scores - use weighted average of continuous scores
        overall_scores = {}
        for i, failure in enumerate(failures):
            overall = (
                self.isolation_weight * if_normalized[i] +
                self.lof_weight * lof_normalized[i]
            )
            overall_scores[failure.attempt_id] = float(np.clip(overall, 0, 1))

        return overall_scores


# ============================================================================
# Component 3: Failure Clusterer
# ============================================================================

class FailureClusterer:
    """
    Cluster failures by similarity using multiple algorithms.

    Uses hierarchical clustering, DBSCAN, and consensus clustering to
    identify groups of similar failures that may indicate hidden constraints.
    """

    def __init__(self, n_clusters_range: Tuple[int, int] = (2, 10),
                 dbscan_eps: float = 0.5, dbscan_min_samples: int = 5):
        """
        Initialize failure clusterer.

        Args:
            n_clusters_range: Range of clusters to try for hierarchical
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN min_samples parameter
        """
        from sklearn.cluster import AgglomerativeClustering, DBSCAN
        from sklearn.metrics import silhouette_score

        self.n_clusters_range = n_clusters_range
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.silhouette_score = silhouette_score

        self.best_n_clusters = None
        self.best_labels = None

    def cluster_failures(self, failures: List[FailureFeatures]) -> List[FailureCluster]:
        """
        Cluster failures and return cluster objects.

        Args:
            failures: List of failure features

        Returns:
            List of FailureCluster objects
        """
        if len(failures) < self.dbscan_min_samples:
            # Not enough data for clustering
            return []

        # Extract feature matrix
        X = np.array([f.feature_vector for f in failures])

        # Try multiple clustering methods
        labels_dict = {}

        # Check if sklearn is available
        try:
            from sklearn.cluster import AgglomerativeClustering, DBSCAN
            from sklearn.metrics import silhouette_score
            sklearn_available = True
        except ImportError:
            sklearn_available = False

        if sklearn_available:
            # 1. Hierarchical clustering with different n_clusters
            for n in range(*self.n_clusters_range):
                if n >= len(failures):
                    break
                hierarchical = AgglomerativeClustering(n_clusters=n)
                labels = hierarchical.fit_predict(X)
                if len(set(labels)) > 1:
                    sil = self.silhouette_score(X, labels)
                    labels_dict[f'hierarchical_{n}'] = (labels, sil)

            # 2. DBSCAN
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(X)
            unique_labels = set(dbscan_labels)
            unique_labels.discard(-1)  # Remove noise

            if len(unique_labels) > 1:
                sil = self.silhouette_score(X, dbscan_labels)
                labels_dict['dbscan'] = (dbscan_labels, sil)

            # 3. Select best clustering
            if not labels_dict:
                return []

            best_method_name, (best_labels, best_silhouette) = max(
                labels_dict.items(),
                key=lambda x: x[1][1]
            )
        else:
            # Fallback: simple K-means-like clustering
            from scipy.cluster.vq import kmeans2
            try:
                # Try 2-4 clusters
                best_silhouette = -1
                best_labels = None
                for n in range(2, min(5, len(failures))):
                    centroids, labels = kmeans2(X, n, minit='points')
                    # Compute simple silhouette
                    if len(set(labels)) > 1:
                        sil = self.silhouette_score(X, labels)
                        if sil > best_silhouette:
                            best_silhouette = sil
                            best_labels = labels

                if best_labels is None:
                    return []
            except:
                return []

        # 4. Create cluster objects
        clusters = self._create_cluster_objects(failures, best_labels)

        return clusters

    def _create_cluster_objects(self, failures: List[FailureFeatures],
                               labels: np.ndarray) -> List[FailureCluster]:
        """Create FailureCluster objects from clustering labels"""
        from scipy.spatial.distance import cdist

        clusters = []
        unique_labels = set(labels)

        for label_id in unique_labels:
            if label_id == -1:  # Noise point in DBSCAN
                continue

            # Get failures in this cluster
            cluster_failures = [
                f for i, f in enumerate(failures) if labels[i] == label_id
            ]

            if len(cluster_failures) == 0:
                continue

            # Compute statistics
            X = np.array([f.feature_vector for f in cluster_failures])
            centroid = np.mean(X, axis=0)
            compactness = float(np.mean([
                np.linalg.norm(f.feature_vector - centroid)
                for f in cluster_failures
            ]))

            # Silhouette score for this cluster (actual calculation)
            if len(cluster_failures) > 2 and len(X) > len(cluster_failures):
                try:
                    from sklearn.metrics import silhouette_score
                    # Get labels for this cluster vs others
                    all_labels = np.array([1 if labels[i] == label_id else 0 for i in range(len(labels))])
                    if len(set(all_labels)) > 1:
                        sil = silhouette_score(X, all_labels)
                    else:
                        sil = 0.5
                except:
                    sil = 0.5
            else:
                sil = 0.5

            # Stability: check if cluster appears in multiple clusterings
            # For now, use compactness as proxy (lower compactness = more stable)
            stability = float(max(0, 1.0 - compactness))

            # Common characteristics
            problem_types = [f.problem_type for f in cluster_failures]
            common_problem_types = list(set(problem_types))

            error_types = [f.error_type for f in cluster_failures]
            common_error_types = list(set(error_types))

            # Aggregate keywords
            all_keywords = []
            for f in cluster_failures:
                all_keywords.extend(f.keywords)
            common_keywords = list(set(all_keywords))[:20]

            cluster = FailureCluster(
                cluster_id=int(label_id),
                size=len(cluster_failures),
                failures=cluster_failures,
                centroid=centroid,
                compactness=compactness,
                silhouette_score=sil,
                stability=stability,
                common_problem_types=common_problem_types,
                common_error_types=common_error_types,
                common_constraints=[],  # Would extract from failures
                keywords=common_keywords
            )
            clusters.append(cluster)

        return clusters


# ============================================================================
# Placeholder for remaining components
# ============================================================================

class AssumptionGenerator:
    """Generate candidate assumptions from failure clusters (Component 4)"""

    def generate_assumptions(self, cluster: FailureCluster) -> List[AssumptionCandidate]:
        """Generate assumption candidates from a cluster"""
        candidates = []

        # Analyze systematic violations
        if cluster.size >= 5:
            # Generate assumption from common characteristics
            problem_desc = cluster.common_problem_types[0] if cluster.common_problem_types else 'problem'
            error_desc = cluster.common_error_types[0].value if cluster.common_error_types else 'error'

            # Create meaningful assumption description
            if 'timeout' in error_desc.lower() or 'time' in error_desc.lower():
                description = f"Time constraint may be too restrictive for {problem_desc}"
            elif 'infeasible' in error_desc.lower():
                description = f"Over-constrained formulation for {problem_desc}"
            elif 'numerical' in error_desc.lower():
                description = f"Numerical instability suggests ill-conditioned {problem_desc}"
            elif 'convergence' in error_desc.lower() or 'optimization' in error_desc.lower():
                description = f"Local optima trap in {problem_desc} - need global optimization"
            else:
                description = f"Hidden constraint in {problem_desc} approach"

            candidate = AssumptionCandidate(
                description=description,
                explains_failures=[f.attempt_id for f in cluster.failures],
                confidence=0.6,
                pattern_type=PatternType.SYSTEMATIC_VIOLATION,
                complexity=1,
                contradiction_count=0,
                testable=True
            )
            candidates.append(candidate)

        return candidates


class ConfidenceScorer:
    """Score confidence of inferred assumptions (Component 5)"""

    def __init__(self):
        self.weights = {
            'support': 0.25,
            'pattern': 0.20,
            'counterfactual': 0.20,
            'novelty': 0.10,
            'historical': 0.10,
            'testability': 0.10,
            'paradigm': 0.05
        }

    def score_assumption(self, candidate: AssumptionCandidate,
                        cluster: FailureCluster,
                        explicit_constraints: List[str]) -> float:
        """Compute overall confidence score"""
        # Support: proportion of cluster failures explained
        support = len(candidate.explains_failures) / max(cluster.size, 1)

        # Pattern: cluster quality (silhouette and stability)
        pattern = (max(0, cluster.silhouette_score) + cluster.stability) / 2

        # Counterfactual: how unique is this (inverse of complexity)
        counterfactual = 1.0 / max(candidate.complexity, 1)

        # Novelty: not in explicit constraints
        novelty = 1.0 if not any(
            cand.lower() in str(explicit_constraints).lower()
            for cand in [candidate.description]
        ) else 0.5

        # Historical: would compare with past paradigm shifts (placeholder)
        historical = 0.0  # Requires historical database

        # Testability
        testability = 1.0 if candidate.testable else 0.5

        # Paradigm: does this challenge fundamental assumptions?
        paradigm_keywords = ['fundamental', 'paradigm', 'assumption', 'constraint']
        paradigm = 1.0 if any(
            kw in candidate.description.lower()
            for kw in paradigm_keywords
        ) else 0.3

        confidence = (
            self.weights['support'] * support +
            self.weights['pattern'] * pattern +
            self.weights['counterfactual'] * counterfactual +
            self.weights['novelty'] * novelty +
            self.weights['historical'] * historical +
            self.weights['testability'] * testability +
            self.weights['paradigm'] * paradigm
        )

        return float(np.clip(confidence, 0, 1))


class ParadigmShiftDetector:
    """Detect paradigm shifts from accumulated assumptions (Component 6)"""

    def __init__(self, crisis_threshold: float = 0.7):
        self.crisis_threshold = crisis_threshold
        self.history: List[ParadigmShiftRecommendation] = []

    def detect_crisis(self, assumptions: List[TacitAssumption],
                     history: List) -> ParadigmShiftRecommendation:
        """Detect if paradigm crisis is occurring"""
        # Get recent assumptions (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        recent = [a for a in assumptions if a.timestamp >= cutoff]

        # Compute crisis signals
        anomaly_count = len(recent)

        # Compute crisis score (simplified)
        crisis_score = min(anomaly_count / 10.0, 1.0)

        # Generate recommendation
        if crisis_score >= self.crisis_threshold:
            paradigm_assumptions = [
                a for a in recent
                if a.paradigm_implication and a.confidence > 0.7
            ]

            recommendation = ParadigmShiftRecommendation(
                trigger=True,
                confidence=crisis_score,
                primary_assumptions=paradigm_assumptions,
                suggested_alternatives=[
                    a.alternative_paradigm for a in paradigm_assumptions
                    if a.alternative_paradigm
                ],
                explanation=self._generate_explanation(recent, crisis_score)
            )
        else:
            recommendation = ParadigmShiftRecommendation(
                trigger=False,
                confidence=crisis_score,
                primary_assumptions=[],
                suggested_alternatives=[],
                explanation="No paradigm crisis detected"
            )

        self.history.append(recommendation)
        return recommendation

    def _generate_explanation(self, assumptions: List[TacitAssumption],
                             crisis_score: float) -> str:
        """Generate human-readable explanation"""
        explanation = f"PARADIGM CRISIS DETECTED (Confidence: {crisis_score:.2f})\n\n"
        explanation += f"Key Assumptions Challenging Current Paradigm:\n"
        for i, assumption in enumerate(assumptions[:5], 1):
            explanation += f"{i}. \"{assumption.description}\" "
            explanation += f"(confidence: {assumption.confidence:.2f})\n"
            explanation += f"   - Supported by {assumption.support} failures\n"
            explanation += f"   - Relaxation: {assumption.constraint_relaxation}\n\n"
        return explanation


# ============================================================================
# Component 7: Main Φ₁.₅ Engine
# ============================================================================

class Phi15Engine:
    """
    Main Φ₁.₅ Tacit Assumption Mining Engine.

    Orchestrates all components to mine tacit assumptions from null results
    and detect paradigm shifts.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Φ₁.₅ engine.

        Args:
            config: Configuration dictionary
        """
        # Configuration
        self.config = config or self._default_config()

        # Initialize components
        self.preprocessor = FailurePreprocessor()
        self.anomaly_detector = AnomalyDetector(
            contamination=self.config['anomaly_contamination']
        )
        self.clusterer = FailureClusterer()
        self.assumption_generator = AssumptionGenerator()
        self.confidence_scorer = ConfidenceScorer()
        self.paradigm_detector = ParadigmShiftDetector(
            crisis_threshold=self.config['crisis_threshold']
        )

        # Database
        self.failures: List[FailureFeatures] = []
        self.assumptions: List[TacitAssumption] = []
        self.paradigm_history: List[ParadigmShiftRecommendation] = []

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'confidence_threshold': 0.6,
            'crisis_threshold': 0.7,
            'min_failures_for_clustering': 10,
            'anomaly_contamination': 0.1
        }

    def process_null_results(self, null_results: List[NullResult]) -> Tuple[
        List[TacitAssumption], ParadigmShiftRecommendation
    ]:
        """
        Process null results through the full Φ₁.₅ pipeline.

        Args:
            null_results: List of null results from Stage 6

        Returns:
            Tuple of (inferred assumptions, paradigm shift recommendation)
        """
        # Step 1: Preprocess failures
        for null_result in null_results:
            features = self.preprocessor.extract_features(null_result)
            self.failures.append(features)

        # Step 2: Detect anomalies
        anomaly_scores = self.anomaly_detector.detect_anomalies(self.failures)
        for failure in self.failures:
            failure.anomaly_score = anomaly_scores.get(failure.attempt_id, 0.0)

        # Step 3: Cluster failures
        clusters = self.clusterer.cluster_failures(self.failures)

        # Step 4: Generate assumptions from clusters
        all_candidates = []
        for cluster in clusters:
            if cluster.is_candidate_for_assumption_mining():
                candidates = self.assumption_generator.generate_assumptions(cluster)
                all_candidates.extend(candidates)

        # Step 5: Score and convert candidates to assumptions
        explicit_constraints = []  # Would get from Stage 1
        new_assumptions = []

        for cluster in clusters:
            cluster_candidates = [
                c for c in all_candidates
                if any(eid in [f.attempt_id for f in cluster.failures]
                      for eid in c.explains_failures)
            ]

            for candidate in cluster_candidates:
                confidence = self.confidence_scorer.score_assumption(
                    candidate, cluster, explicit_constraints
                )

                if confidence >= self.config['confidence_threshold']:
                    # Detect paradigm implication
                    paradigm_impl = self._detect_paradigm_implication(candidate.description)

                    # Generate alternative paradigm if applicable
                    alt_paradigm = self._suggest_alternative_paradigm(candidate.description) if paradigm_impl else None

                    assumption = TacitAssumption(
                        id=self._generate_assumption_id(candidate.description),
                        description=candidate.description,
                        formalization=self._formalize_assumption(candidate.description),
                        assumption_type=self._classify_assumption_type(candidate.description),
                        confidence=confidence,
                        support=len(candidate.explains_failures),
                        evidence=candidate.explains_failures,
                        pattern_type=candidate.pattern_type,
                        constraint_relaxation=f"Relax: {candidate.description}",
                        paradigm_implication=paradigm_impl,
                        alternative_paradigm=alt_paradigm
                    )
                    new_assumptions.append(assumption)

        # Add to assumptions list
        self.assumptions.extend(new_assumptions)

        # Step 6: Detect paradigm shift
        paradigm_recommendation = self.paradigm_detector.detect_crisis(
            self.assumptions,
            self.paradigm_history
        )
        self.paradigm_history.append(paradigm_recommendation)

        return new_assumptions, paradigm_recommendation

    def _generate_assumption_id(self, description: str) -> str:
        """Generate unique ID for assumption"""
        hash_obj = hashlib.md5(description.encode())
        return f"assumption_{hash_obj.hexdigest()[:8]}"

    def _formalize_assumption(self, description: str) -> str:
        """Convert natural language to SCE constraint format"""
        # Simplified formalization
        # In production, use LLM-based translation
        return f"forall (x : Entity), {description.lower().replace(' ', '_')}"

    def _classify_assumption_type(self, description: str) -> AssumptionType:
        """Classify assumption type from description"""
        # Simplified classification
        desc_lower = description.lower()

        if any(word in desc_lower for word in ['must', 'should', 'constraint']):
            return AssumptionType.CONSTRAINT
        elif any(word in desc_lower for word in ['method', 'approach', 'algorithm']):
            return AssumptionType.METHODOLOGICAL
        elif any(word in desc_lower for word in ['represent', 'model', 'formulate']):
            return AssumptionType.REPRESENTATIONAL
        else:
            return AssumptionType.ONTOLOGICAL

    def _detect_paradigm_implication(self, description: str) -> bool:
        """Detect if assumption suggests paradigm shift"""
        paradigm_indicators = [
            'fundamental', 'paradigm', 'assumption', 'over-constrained',
            'incompatible', 'contradiction', 'rethink', 'reconsider',
            'alternative', 'different approach', 'need to'
        ]

        desc_lower = description.lower()
        return any(indicator in desc_lower for indicator in paradigm_indicators)

    def _suggest_alternative_paradigm(self, description: str) -> Optional[str]:
        """Suggest alternative paradigm based on assumption"""
        desc_lower = description.lower()

        if 'time' in desc_lower or 'timeout' in desc_lower:
            return "Approximation algorithms / randomized methods"
        elif 'infeasible' in desc_lower or 'constraint' in desc_lower:
            return "Relax constraint formulation / soft constraints"
        elif 'numerical' in desc_lower or 'instability' in desc_lower:
            return "Regularization / robust optimization"
        elif 'local' in desc_lower or 'optima' in desc_lower:
            return "Global optimization / metaheuristics"
        else:
            return "Alternative problem formulation"

    def get_top_assumptions(self, k: int = 10) -> List[TacitAssumption]:
        """Get top-k assumptions by confidence"""
        sorted_assumptions = sorted(
            self.assumptions,
            key=lambda a: a.confidence,
            reverse=True
        )
        return sorted_assumptions[:k]

    def save_state(self, filepath: str) -> None:
        """Save engine state to file"""
        state = {
            'failures': [f.to_dict() for f in self.failures],
            'assumptions': [a.to_dict() for a in self.assumptions],
            'paradigm_history': [p.to_dict() for p in self.paradigm_history],
            'config': self.config
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load engine state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore assumptions
        self.assumptions = [
            TacitAssumption.from_dict(a_data)
            for a_data in state['assumptions']
        ]

        # Restore paradigm history - filter to valid fields only
        self.paradigm_history = []
        for p_data in state['paradigm_history']:
            # Use from_dict classmethod to properly reconstruct
            self.paradigm_history.append(ParadigmShiftRecommendation.from_dict(p_data))

        self.config = state['config']


# ============================================================================
# Convenience Functions
# ============================================================================

def create_phi15_engine(config: Optional[Dict] = None) -> Phi15Engine:
    """
    Create a Φ₁.₅ engine with default or custom configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized Phi15Engine
    """
    return Phi15Engine(config)


if __name__ == "__main__":
    # Quick test
    print("Φ₁.₅ Tacit Assumption Miner - Agent B1")
    print("=" * 50)

    # Create engine
    engine = create_phi15_engine()

    # Create sample null result
    null_result = NullResult(
        attempt_id="test_001",
        timestamp=datetime.now(),
        problem_type="optimization",
        approach_type="deterministic",
        constraints=["constraint_1"],
        error_type=ErrorType.OPTIMIZATION_FAILED,
        error_message="Optimization failed to converge due to numerical instability",
        state={"iteration": 100},
        iteration=100,
        resources_used={"cpu": 50.0, "memory": 100.0}
    )

    # Process
    assumptions, paradigm_rec = engine.process_null_results([null_result])

    print(f"\nProcessed {len(null_result.attempt_id)} null result(s)")
    print(f"Inferred {len(assumptions)} assumption(s)")
    print(f"Paradigm crisis: {paradigm_rec.trigger}")
    print(f"\nSystem ready for integration with Stage 6 and Stage 1")
