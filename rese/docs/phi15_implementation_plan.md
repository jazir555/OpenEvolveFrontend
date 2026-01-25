# Φ₁.₅ Implementation Plan

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: KEY INNOVATION - Implementation Plan
**Timeline**: Week 11-16 (6 weeks)
**Target**: >70% assumption mining accuracy

---

## Executive Summary

This document provides a comprehensive implementation plan for Φ₁.₅ (Tacit Assumption Mining), covering data structures, component implementation, integration points, testing strategies, and deployment considerations. The plan is designed for a 6-week implementation period (Week 11-16) with clear milestones and deliverables.

---

## Table of Contents

1. [Implementation Overview](#implementation-overview)
2. [Data Structures](#data-structures)
3. [Component Implementation Plan](#component-implementation-plan)
4. [Integration with Existing Systems](#integration-with-existing-systems)
5. [Testing Strategy](#testing-strategy)
6. [Deployment Plan](#deployment-plan)
7. [Risk Mitigation](#risk-mitigation)
8. [Performance Optimization](#performance-optimization)
9. [Documentation Plan](#documentation-plan)
10. [Milestone Timeline](#milestone-timeline)

---

## 1. Implementation Overview

### 1.1 Implementation Scope

**In Scope**:
- Core Φ₁.₅ assumption mining pipeline
- Integration with Stage 6 (Error Source Analysis)
- Integration with Stage 1 (Prompt Analysis) for feedback loop
- Integration with Stage 7 (Validation) for assumption verification
- Failure database and persistence layer
- Machine learning models for anomaly detection and clustering
- Abductive inference engine
- Confidence scoring system
- Paradigm shift detection

**Out of Scope** (Future Work):
- Automated Lean 4 verification of assumptions (handled by Agent O1)
- UI for visualizing paradigm shifts (handled by Agent Z1)
- Integration with external scientific databases
- Real-time collaborative assumption mining

### 1.2 Technology Stack

**Core Language**: Python 3.11+

**Key Libraries**:
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn, PyTorch (optional)
- **Clustering**: Scikit-learn (cluster module)
- **Anomaly Detection**: Scikit-learn, PyOD (optional)
- **Natural Language**: Transformers (for semantic similarity)
- **Graph Processing**: NetworkX (for dependency graphs)
- **Persistence**: SQLite (development), PostgreSQL (production)
- **Serialization**: JSON, Pickle
- **Testing**: Pytest, pytest-cov
- **Documentation**: Sphinx, MkDocs

**Integration with RESE**:
- SCE (Symbolic Constraint Engine) by Agent A1
- Stage 6 (Error Source Analysis)
- Stage 1 (Prompt Analysis)
- Stage 7 (Validation)

### 1.3 File Structure

```
rese/
├── phase1/
│   └── tacit_assumption_miner.py          # Main Φ₁.₅ module
├── data/
│   ├── failures.json                      # Failure database
│   ├── assumptions.json                   # Inferred assumptions
│   └── paradigm_shifts.json               # Historical paradigm shifts
├── models/                                 # Trained ML models
│   ├── anomaly_detector.pkl
│   ├── failure_clusterer.pkl
│   └── confidence_scorer.pkl
├── tests/
│   ├── test_phi15.py                      # Main test suite
│   ├── test_preprocessing.py
│   ├── test_anomaly_detection.py
│   ├── test_clustering.py
│   ├── test_abduction.py
│   └── test_integration.py
├── docs/
│   ├── phi15_assumption_mining_research.md    # ✅ Created
│   ├── phi15_algorithm_design.md              # ✅ Created
│   ├── phi15_implementation_plan.md           # ✅ This file
│   ├── phi15_validation_strategy.md           # ⏳ Next
│   └── phi15_api.md                           # API documentation
└── lean4/
    └── Phi15.lean                         # Lean 4 formalization (Agent O1)
```

---

## 2. Data Structures

### 2.1 Core Data Classes

**File**: `rese/phase1/tacit_assumption_miner.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

# ============================================================================
# Enums
# ============================================================================

class ErrorType(Enum):
    """Types of errors from Stage 6"""
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
    ONTOLOGICAL = "ontological"          # About what exists
    METHODOLOGICAL = "methodological"    # About how to solve
    CONSTRAINT = "constraint"            # Hidden constraints
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

    This is the primary input to Φ₁.₅.
    """
    attempt_id: str
    timestamp: datetime
    problem_type: str
    approach_type: str
    constraints: List[str]
    error_type: ErrorType
    error_message: str
    state: Dict
    iteration: int
    resources_used: Dict
    metadata: Dict = field(default_factory=dict)

@dataclass
class FailureFeatures:
    """
    Extracted features from a null result.
    """
    attempt_id: str
    timestamp: datetime

    # Categorical features (one-hot encoded later)
    problem_type: str
    approach_type: str
    error_type: ErrorType

    # Numerical features
    iteration: int
    time_to_failure: float
    error_magnitude: Optional[float]
    resource_consumption: float
    constraint_violation_count: int

    # Feature vector (concatenated)
    feature_vector: np.ndarray

    # Analysis results (filled later)
    failure_cluster: Optional[int] = None
    anomaly_score: Optional[float] = None

# ============================================================================
# Intermediate Structures
# ============================================================================

@dataclass
class FailureCluster:
    """
    Cluster of similar failures.
    """
    cluster_id: int
    size: int
    failures: List[FailureFeatures]

    # Cluster statistics
    centroid: np.ndarray
    compactness: float
    silhouette_score: float
    stability: float

    # Cluster characteristics
    common_problem_types: List[str]
    common_error_types: List[ErrorType]
    common_constraints: List[str]
    keywords: List[str]

    def is_candidate_for_assumption_mining(self, min_size: int = 5,
                                          max_compactness: float = 0.5,
                                          min_stability: float = 0.7) -> bool:
        """Check if cluster is worth analyzing"""
        return (
            self.size >= min_size and
            self.compactness <= max_compactness and
            self.stability >= min_stability
        )

@dataclass
class AssumptionCandidate:
    """
    Candidate assumption from abductive inference.
    """
    description: str
    explains_failures: List[str]  # Attempt IDs
    confidence: float
    pattern_type: PatternType
    complexity: int  # Simplicity score
    contradiction_count: int
    testable: bool

# ============================================================================
# Output Structures (to Stage 1)
# ============================================================================

@dataclass
class TacitAssumption:
    """
    Inferred tacit assumption to add as constraint.

    This is the primary output of Φ₁.₅.
    """
    id: str
    description: str
    formalization: str  # SCE constraint format
    assumption_type: AssumptionType
    confidence: float
    support: int  # Number of failures explained
    evidence: List[str]  # Attempt IDs supporting this
    pattern_type: PatternType
    constraint_relaxation: str
    paradigm_implication: bool
    alternative_paradigm: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_sce_constraint(self):
        """Convert to SCE Constraint format"""
        from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

        return Constraint(
            id=self.id,
            type=ConstraintType.SOFT,  # Start as soft (inferred)
            description=f"[INFERRED] {self.description}",
            formalization=self.formalization,
            source="phi15_inferred",
            dependencies=[],
            verified=False,
            lean_theorem=None
        )

@dataclass
class ParadigmShiftRecommendation:
    """
    Recommendation for paradigm shift.
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
            "trigger": self.trigger,
            "confidence": self.confidence,
            "assumptions_to_relax": [a.id for a in self.primary_assumptions],
            "alternative_paradigms": self.suggested_alternatives,
            "explanation": self.explanation,
            "priority": "HIGH" if self.trigger and self.confidence > 0.8 else "MEDIUM",
            "timestamp": self.timestamp.isoformat()
        }

# ============================================================================
# Database Structures
# ============================================================================

@dataclass
class FailureDatabase:
    """
    Database for storing failures and metadata.
    """
    failures: List[FailureFeatures] = field(default_factory=list)
    clusters: List[FailureCluster] = field(default_factory=list)
    assumptions: List[TacitAssumption] = field(default_factory=list)
    paradigm_history: List[ParadigmShiftRecommendation] = field(default_factory=list)

    def add_failure(self, failure: FailureFeatures) -> None:
        """Add a failure to the database"""
        self.failures.append(failure)

    def get_failures_since(self, timestamp: datetime) -> List[FailureFeatures]:
        """Get failures since a given timestamp"""
        return [f for f in self.failures if f.timestamp >= timestamp]

    def get_unprocessed_failures(self) -> List[FailureFeatures]:
        """Get failures that haven't been clustered yet"""
        return [f for f in self.failures if f.failure_cluster is None]
```

---

## 3. Component Implementation Plan

### 3.1 Week 11: Core Infrastructure (Days 1-7)

**Tasks**:

1. **Day 1-2: Data Structures**
   - Implement all dataclasses above
   - Add serialization methods (to/from JSON)
   - Write basic tests for data structures
   - **Deliverable**: `rese/phase1/tacit_assumption_miner.py` with data structures

2. **Day 3-4: Failure Preprocessor**
   - Implement feature extraction
   - Implement normalization
   - Add keyword extraction from error messages
   - **Deliverable**: `FailurePreprocessor` class

3. **Day 5-7: Database Layer**
   - Implement `FailureDatabase` class
   - Add persistence (JSON initially, upgrade to SQL later)
   - Implement CRUD operations
   - Write tests for database operations
   - **Deliverable**: Persistence layer working

**Code Skeleton**:

```python
class FailurePreprocessor:
    """Preprocess null results into feature vectors"""

    def __init__(self):
        self.feature_encoder = None  # Learn from data
        self.keyword_extractor = None

    def extract_features(self, null_result: NullResult) -> FailureFeatures:
        """Extract features from null result"""
        # 1. Extract categorical features
        problem_type = null_result.problem_type
        approach_type = null_result.approach_type
        error_type = null_result.error_type

        # 2. Extract numerical features
        iteration = null_result.iteration
        time_to_failure = self._compute_time_to_failure(null_result)
        error_magnitude = self._compute_error_magnitude(null_result)
        resource_consumption = self._compute_resource_usage(null_result)
        constraint_violation_count = len(null_result.constraints)

        # 3. Extract keywords
        keywords = self._extract_keywords(null_result.error_message)

        # 4. Create feature vector
        feature_vector = self._create_feature_vector(
            problem_type, approach_type, error_type,
            iteration, time_to_failure, error_magnitude,
            resource_consumption, constraint_violation_count
        )

        return FailureFeatures(
            attempt_id=null_result.attempt_id,
            timestamp=null_result.timestamp,
            problem_type=problem_type,
            approach_type=approach_type,
            error_type=error_type,
            iteration=iteration,
            time_to_failure=time_to_failure,
            error_magnitude=error_magnitude,
            resource_consumption=resource_consumption,
            constraint_violation_count=constraint_violation_count,
            feature_vector=feature_vector,
            keywords=keywords
        )

    def _extract_keywords(self, error_message: str) -> List[str]:
        """Extract key terms from error message"""
        # Use TF-IDF or RAKE for keyword extraction
        # Placeholder: simple word frequency
        words = error_message.lower().split()
        stop_words = {'the', 'a', 'an', 'is', 'was', 'at', 'which', 'on'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(keywords))[:10]  # Top 10 unique keywords
```

### 3.2 Week 12: Anomaly Detection & Clustering (Days 8-14)

**Tasks**:

1. **Day 8-10: Anomaly Detector**
   - Implement Isolation Forest
   - Implement LOF (Local Outlier Factor)
   - Implement temporal anomaly detection (CUSUM)
   - Combine multiple anomaly scores
   - **Deliverable**: `AnomalyDetector` class

2. **Day 11-14: Failure Clusterer**
   - Implement hierarchical clustering
   - Implement DBSCAN
   - Implement consensus clustering
   - Add cluster quality metrics
   - **Deliverable**: `FailureClusterer` class

**Code Skeleton**:

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

class AnomalyDetector:
    """Detect anomalies in failure patterns"""

    def __init__(self, contamination=0.1):
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    def detect_anomalies(self, failures: List[FailureFeatures]) -> Dict[str, float]:
        """
        Detect anomalies and return scores.

        Returns:
            Dict mapping attempt_id to overall anomaly score
        """
        # Extract feature matrix
        X = np.array([f.feature_vector for f in failures])

        # Point anomalies (Isolation Forest)
        if_scores = self.isolation_forest.fit_predict(X)
        if_anomalies = (if_scores == -1).astype(float)

        # Local outliers (LOF)
        lof_scores = self.lof.fit_predict(X)
        lof_anomalies = (lof_scores == -1).astype(float)

        # Combine
        overall_scores = {}
        for i, failure in enumerate(failures):
            overall = 0.5 * if_anomalies[i] + 0.5 * lof_anomalies[i]
            overall_scores[failure.attempt_id] = overall

        return overall_scores

class FailureClusterer:
    """Cluster failures by similarity"""

    def __init__(self, n_clusters_range=(2, 10)):
        self.n_clusters_range = n_clusters_range
        self.best_n_clusters = None
        self.best_labels = None

    def cluster_failures(self, failures: List[FailureFeatures]) -> List[FailureCluster]:
        """Cluster failures and return cluster objects"""
        X = np.array([f.feature_vector for f in failures])

        # Try multiple clustering methods and find consensus
        labels_dict = {}

        # 1. Hierarchical clustering
        for n in range(*self.n_clusters_range):
            hierarchical = AgglomerativeClustering(n_clusters=n)
            labels = hierarchical.fit_predict(X)
            silhouette = silhouette_score(X, labels)
            labels_dict[f'hierarchical_{n}'] = (labels, silhouette)

        # 2. DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        if len(set(dbscan_labels)) > 1:  # Not all noise
            dbscan_silhouette = silhouette_score(X, dbscan_labels)
            labels_dict['dbscan'] = (dbscan_labels, dbscan_silhouette)

        # 3. Select best clustering
        best_method = max(labels_dict.items(), key=lambda x: x[1][1])
        best_labels = best_method[1][0]

        # 4. Create cluster objects
        clusters = self._create_cluster_objects(failures, best_labels)

        return clusters

    def _create_cluster_objects(self, failures: List[FailureFeatures],
                                labels: np.ndarray) -> List[FailureCluster]:
        """Create FailureCluster objects from labels"""
        clusters = []
        unique_labels = set(labels)

        for label_id in unique_labels:
            if label_id == -1:  # Noise point in DBSCAN
                continue

            # Get failures in this cluster
            cluster_failures = [f for i, f in enumerate(failures) if labels[i] == label_id]

            # Compute statistics
            X = np.array([f.feature_vector for f in cluster_failures])
            centroid = np.mean(X, axis=0)
            compactness = np.mean([np.linalg.norm(f.feature_vector - centroid)
                                  for f in cluster_failures])

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
                silhouette_score=0.0,  # Compute if needed
                stability=0.0,  # Compute if needed
                common_problem_types=common_problem_types,
                common_error_types=common_error_types,
                common_constraints=[],  # Extract if needed
                keywords=common_keywords
            )
            clusters.append(cluster)

        return clusters
```

### 3.3 Week 13: Abductive Inference (Days 15-21)

**Tasks**:

1. **Day 15-17: Assumption Generator**
   - Implement constraint violation analysis
   - Implement counterfactual reasoning
   - Implement boundary analysis
   - **Deliverable**: `AssumptionGenerator` class

2. **Day 18-21: Pattern Matching**
   - Build historical paradigm shift database
   - Implement similarity matching
   - Add counterfactual simulation
   - **Deliverable**: `PatternMatcher` class

**Code Skeleton**:

```python
class AssumptionGenerator:
    """Generate candidate assumptions from failure clusters"""

    def __init__(self):
        self.constraint_templates = self._load_constraint_templates()

    def generate_assumptions(self, cluster: FailureCluster) -> List[AssumptionCandidate]:
        """Generate assumption candidates from a cluster"""
        candidates = []

        # Method 1: Constraint violation analysis
        candidates.extend(self._analyze_constraint_violations(cluster))

        # Method 2: Boundary analysis
        candidates.extend(self._analyze_boundaries(cluster))

        # Method 3: Pattern-based inference
        candidates.extend(self._infer_from_patterns(cluster))

        return candidates

    def _analyze_constraint_violations(self, cluster: FailureCluster) -> List[AssumptionCandidate]:
        """Analyze systematic constraint violations"""
        candidates = []

        # Find constraints commonly violated in this cluster
        violation_counts = {}
        for failure in cluster.failures:
            # This would need actual constraint data from null result
            # Placeholder: count error types
            error_type = failure.error_type
            violation_counts[error_type] = violation_counts.get(error_type, 0) + 1

        # Generate candidates for frequently violated constraints
        for error_type, count in violation_counts.items():
            if count >= cluster.size * 0.5:  # 50% of failures
                description = f"Assumption: {error_type.value} must not occur"
                candidate = AssumptionCandidate(
                    description=description,
                    explains_failures=[f.attempt_id for f in cluster.failures],
                    confidence=count / cluster.size,
                    pattern_type=PatternType.SYSTEMATIC_VIOLATION,
                    complexity=1,
                    contradiction_count=0,
                    testable=True
                )
                candidates.append(candidate)

        return candidates

    def _analyze_boundaries(self, cluster: FailureCluster) -> List[AssumptionCandidate]:
        """Analyze convergence to boundaries"""
        # Check if failures converge to same value/limit
        # Placeholder implementation
        return []

    def _infer_from_patterns(self, cluster: FailureCluster) -> List[AssumptionCandidate]:
        """Infer assumptions from failure patterns"""
        # Use pattern matching to historical paradigm shifts
        # Placeholder implementation
        return []

class PatternMatcher:
    """Match failure patterns to historical paradigm shifts"""

    def __init__(self, historical_database_path: str):
        self.historical_database = self._load_historical_data(historical_database_path)

    def find_similar_paradigm_shifts(self, cluster: FailureCluster) -> List[Dict]:
        """Find similar paradigm shifts in history"""
        # Compute similarity between cluster and historical cases
        # Return top matches with their tacit assumptions
        # Placeholder: return empty list
        return []
```

### 3.4 Week 14: Confidence Scoring (Days 22-28)

**Tasks**:

1. **Day 22-25: Confidence Scorer**
   - Implement multi-factor confidence scoring
   - Add support, pattern, counterfactual factors
   - Implement weight learning
   - **Deliverable**: `ConfidenceScorer` class

2. **Day 26-28: Assumption Manager**
   - Implement assumption deduplication
   - Add SCE constraint conversion
   - Implement filtering and ranking
   - **Deliverable**: `AssumptionManager` class

**Code Skeleton**:

```python
class ConfidenceScorer:
    """Score confidence of inferred assumptions"""

    def __init__(self, weights: Optional[Dict] = None):
        # Default weights (can be learned from data)
        self.weights = weights or {
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

        # 1. Support score
        support = len(candidate.explains_failures) / cluster.size

        # 2. Pattern score (cluster quality)
        pattern = cluster.silhouette_score if hasattr(cluster, 'silhouette_score') else 0.5

        # 3. Counterfactual score (simulate relaxation)
        counterfactual = self._simulate_relaxation(candidate, cluster)

        # 4. Novelty score (difference from explicit constraints)
        novelty = self._compute_novelty(candidate.description, explicit_constraints)

        # 5. Historical score (match to paradigm shifts)
        historical = 0.0  # Would need PatternMatcher

        # 6. Testability
        testability = 1.0 if candidate.testable else 0.5

        # 7. Paradigm score (if paradigm-level)
        paradigm = 0.0  # Would need deeper analysis

        # Combine
        confidence = (
            self.weights['support'] * support +
            self.weights['pattern'] * pattern +
            self.weights['counterfactual'] * counterfactual +
            self.weights['novelty'] * novelty +
            self.weights['historical'] * historical +
            self.weights['testability'] * testability +
            self.weights['paradigm'] * paradigm
        )

        return np.clip(confidence, 0, 1)

    def _simulate_relaxation(self, candidate: AssumptionCandidate,
                            cluster: FailureCluster) -> float:
        """Simulate what happens if constraint is relaxed"""
        # Placeholder: would need to run experiments
        # Return expected improvement probability
        return 0.5

    def _compute_novelty(self, description: str, explicit_constraints: List[str]) -> float:
        """Compute how different this is from explicit constraints"""
        # Use semantic similarity (e.g., sentence transformers)
        # Placeholder: return high novelty
        return 0.8

class AssumptionManager:
    """Manage inferred assumptions and convert to SCE constraints"""

    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.assumptions: List[TacitAssumption] = []

    def add_candidates(self, candidates: List[AssumptionCandidate],
                      cluster: FailureCluster,
                      explicit_constraints: List[str]) -> None:
        """Add candidate assumptions after scoring"""
        scorer = ConfidenceScorer()

        for candidate in candidates:
            confidence = scorer.score_assumption(candidate, cluster, explicit_constraints)

            if confidence >= self.confidence_threshold:
                # Convert to TacitAssumption
                assumption = TacitAssumption(
                    id=f"assumption_{len(self.assumptions)}",
                    description=candidate.description,
                    formalization=self._formalize(candidate.description),
                    assumption_type=self._classify_type(candidate.description),
                    confidence=confidence,
                    support=len(candidate.explains_failures),
                    evidence=candidate.explains_failures,
                    pattern_type=candidate.pattern_type,
                    constraint_relaxation=f"Relax: {candidate.description}",
                    paradigm_implication=False,  # Would need deeper analysis
                    alternative_paradigm=None
                )
                self.assumptions.append(assumption)

    def deduplicate(self) -> None:
        """Remove duplicate assumptions"""
        # Use semantic similarity to find duplicates
        # Placeholder: implement simple deduplication
        unique = []
        seen = set()

        for assumption in self.assumptions:
            key = assumption.description.lower()
            if key not in seen:
                seen.add(key)
                unique.append(assumption)

        self.assumptions = unique

    def get_top_assumptions(self, k: int = 10) -> List[TacitAssumption]:
        """Get top-k assumptions by confidence"""
        sorted_assumptions = sorted(self.assumptions, key=lambda a: a.confidence, reverse=True)
        return sorted_assumptions[:k]

    def _formalize(self, description: str) -> str:
        """Convert natural language to SCE constraint format"""
        # Use LLM or templates for conversion
        # Placeholder: simple conversion
        return f"forall (x : Entity), {description}"

    def _classify_type(self, description: str) -> AssumptionType:
        """Classify assumption type"""
        # Use keyword matching or classifier
        # Placeholder: return default
        return AssumptionType.CONSTRAINT
```

### 3.5 Week 15: Integration & Feedback (Days 29-35)

**Tasks**:

1. **Day 29-31: Stage 6 Integration**
   - Implement interface to receive null results
   - Add error source parsing
   - Implement incremental processing
   - **Deliverable**: `Phi15Stage6Interface` class

2. **Day 32-35: Stage 1 & 7 Integration**
   - Implement output to Stage 1 (constraint addition)
   - Implement feedback loop from Stage 7 (validation)
   - Add confidence updates from validation
   - **Deliverable**: `Phi15Stage1Interface`, `Phi15Stage7Interface` classes

**Code Skeleton**:

```python
class Phi15Stage6Interface:
    """Interface for receiving null results from Stage 6"""

    def __init__(self, phi15_engine: 'Phi15Engine'):
        self.phi15 = phi15_engine

    def receive_null_result(self, result: NullResult) -> None:
        """Receive a single null result"""
        # Preprocess
        features = self.phi15.preprocessor.extract_features(result)

        # Add to database
        self.phi15.database.add_failure(features)

        # Check if should process
        if self.phi15.should_process():
            self.phi15.process_incremental()

    def receive_batch(self, results: List[NullResult]) -> None:
        """Receive batch of null results"""
        for result in results:
            self.receive_null_result(result)

        # Force full processing
        self.phi15.process_full()

class Phi15Stage1Interface:
    """Interface for sending assumptions to Stage 1"""

    def __init__(self, phi15_engine: 'Phi15Engine'):
        self.phi15 = phi15_engine

    def send_assumptions(self, assumptions: List[TacitAssumption]) -> None:
        """Send inferred assumptions to Stage 1"""
        # Filter by confidence
        high_confidence = [a for a in assumptions if a.confidence >= 0.6]

        # Send to Stage 1
        for assumption in high_confidence:
            # Convert to SCE constraint
            sce_constraint = assumption.to_sce_constraint()

            # Add to Stage 1's SCE
            # This would call Stage 1's API
            print(f"Sending to Stage 1: {assumption.description}")

class Phi15Stage7Interface:
    """Interface for receiving validation results from Stage 7"""

    def __init__(self, phi15_engine: 'Phi15Engine'):
        self.phi15 = phi15_engine

    def receive_validation_result(self, assumption_id: str,
                                  validation_success: bool,
                                  improvement_score: float) -> None:
        """Receive validation result for an assumption"""
        # Find assumption
        assumption = next((a for a in self.phi15.assumptions if a.id == assumption_id), None)

        if assumption:
            # Update confidence based on validation
            if validation_success:
                assumption.confidence = min(1.0, assumption.confidence * 1.2)
            else:
                assumption.confidence = max(0.0, assumption.confidence * 0.7)
```

### 3.6 Week 16: Paradigm Shift Detection (Days 36-42)

**Tasks**:

1. **Day 36-39: Paradigm Shift Detector**
   - Implement Kuhnian crisis signal detection
   - Add historical pattern matching for paradigm shifts
   - Implement paradigm shift recommendation generation
   - **Deliverable**: `ParadigmShiftDetector` class

2. **Day 40-42: Main Φ₁.₅ Engine**
   - Assemble all components
   - Implement main processing loop
   - Add configuration management
   - **Deliverable**: `Phi15Engine` class

**Code Skeleton**:

```python
class ParadigmShiftDetector:
    """Detect paradigm shifts from accumulated assumptions"""

    def __init__(self, crisis_threshold=0.7):
        self.crisis_threshold = crisis_threshold
        self.history: List[ParadigmShiftRecommendation] = []

    def detect_crisis(self, assumptions: List[TacitAssumption],
                     history: List) -> ParadigmShiftRecommendation:
        """Detect if paradigm crisis is occurring"""
        # Get recent assumptions
        recent = [a for a in assumptions
                 if (datetime.now() - a.timestamp).days < 30]

        # Compute crisis signals
        anomaly_count = len(recent)
        rate_ratio = self._compute_rate_ratio(recent, history)
        paradigm_count = len([a for a in recent if a.paradigm_implication])

        # Compute crisis score
        crisis_score = (
            0.25 * min(anomaly_count / 10, 1.0) +
            0.25 * min(rate_ratio / 3.0, 1.0) +
            0.25 * min(paradigm_count / 5.0, 1.0) +
            0.25 * 0.0  # Cross-domain signal
        )

        # Generate recommendation
        if crisis_score >= self.crisis_threshold:
            paradigm_assumptions = [a for a in recent
                                   if a.paradigm_implication and a.confidence > 0.7]

            recommendation = ParadigmShiftRecommendation(
                trigger=True,
                confidence=crisis_score,
                primary_assumptions=paradigm_assumptions,
                suggested_alternatives=[a.alternative_paradigm for a in paradigm_assumptions
                                       if a.alternative_paradigm],
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

    def _compute_rate_ratio(self, recent: List[TacitAssumption],
                           history: List) -> float:
        """Compute ratio of current assumption rate to historical rate"""
        # Placeholder: simple implementation
        if not history:
            return 1.0
        return len(recent) / max(len(history), 1)

    def _generate_explanation(self, assumptions: List[TacitAssumption],
                             crisis_score: float) -> str:
        """Generate human-readable explanation"""
        explanation = f"PARADIGM CRISIS DETECTED (Confidence: {crisis_score:.2f})\n\n"
        explanation += f"Key Assumptions Challenging Current Paradigm:\n"
        for i, assumption in enumerate(assumptions[:5], 1):
            explanation += f"{i}. \"{assumption.description}\" (confidence: {assumption.confidence:.2f})\n"
            explanation += f"   - Supported by {assumption.support} failures\n"
            explanation += f"   - Relaxation: {assumption.constraint_relaxation}\n\n"
        return explanation

class Phi15Engine:
    """Main Φ₁.₅ engine"""

    def __init__(self, config: Optional[Dict] = None):
        # Configuration
        self.config = config or self._default_config()

        # Components
        self.preprocessor = FailurePreprocessor()
        self.anomaly_detector = AnomalyDetector()
        self.clusterer = FailureClusterer()
        self.assumption_generator = AssumptionGenerator()
        self.confidence_scorer = ConfidenceScorer()
        self.assumption_manager = AssumptionManager(
            confidence_threshold=self.config['confidence_threshold']
        )
        self.paradigm_detector = ParadigmShiftDetector(
            crisis_threshold=self.config['crisis_threshold']
        )

        # Interfaces
        self.stage6_interface = Phi15Stage6Interface(self)
        self.stage1_interface = Phi15Stage1Interface(self)
        self.stage7_interface = Phi15Stage7Interface(self)

        # Database
        self.database = FailureDatabase()
        self.assumptions: List[TacitAssumption] = []

    def process_full(self) -> Tuple[List[TacitAssumption], ParadigmShiftRecommendation]:
        """Run full Φ₁.₅ pipeline"""
        # 1. Get unprocessed failures
        unprocessed = self.database.get_unprocessed_failures()

        if not unprocessed:
            return [], self.paradigm_detector.detect_crisis([], [])

        # 2. Detect anomalies
        anomaly_scores = self.anomaly_detector.detect_anomalies(unprocessed)

        # 3. Cluster failures
        clusters = self.clusterer.cluster_failures(unprocessed)

        # 4. Generate assumptions from clusters
        all_candidates = []
        for cluster in clusters:
            if cluster.is_candidate_for_assumption_mining():
                candidates = self.assumption_generator.generate_assumptions(cluster)
                all_candidates.extend(candidates)

        # 5. Score and filter assumptions
        explicit_constraints = []  # Would get from Stage 1
        for cluster in clusters:
            if cluster.is_candidate_for_assumption_mining():
                self.assumption_manager.add_candidates(
                    [c for c in all_candidates if c in [cand for cand in ...]],  # Match candidates to cluster
                    cluster,
                    explicit_constraints
                )

        # 6. Deduplicate
        self.assumption_manager.deduplicate()
        self.assumptions = self.assumption_manager.assumptions

        # 7. Detect paradigm shift
        paradigm_recommendation = self.paradigm_detector.detect_crisis(
            self.assumptions,
            self.database.paradigm_history
        )

        # 8. Send to Stage 1
        self.stage1_interface.send_assumptions(self.assumptions)

        return self.assumptions, paradigm_recommendation

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'confidence_threshold': 0.6,
            'crisis_threshold': 0.7,
            'min_failures_for_clustering': 10,
            'anomaly_contamination': 0.1
        }
```

---

## 4. Integration with Existing Systems

### 4.1 Integration Points

**Stage 6 → Φ₁.₅** (Input):
- Null results with error classification
- Failure context and metadata
- Integration method: Direct API calls or message queue

**Φ₁.₅ → Stage 1** (Output):
- Inferred assumptions as SCE constraints
- Paradigm shift recommendations
- Integration method: Direct SCE API calls

**Φ₁.₅ → Stage 7** (Validation):
- Assumption validation requests
- Integration method: Validation API

**Stage 7 → Φ₁.₅** (Feedback):
- Validation results (success/failure)
- Confidence updates
- Integration method: Callback API

### 4.2 API Design

**REST API (Optional)**:

```python
# FastAPI example
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Φ₁.₅ Tacit Assumption Mining API")

class NullResultInput(BaseModel):
    attempt_id: str
    timestamp: datetime
    problem_type: str
    approach_type: str
    constraints: List[str]
    error_type: str
    error_message: str
    state: Dict
    iteration: int
    resources_used: Dict
    metadata: Dict = {}

@app.post("/api/v1/null-results")
async def receive_null_result(result: NullResultInput):
    """Receive null result from Stage 6"""
    null_result = NullResult(**result.dict())
    phi15_engine.stage6_interface.receive_null_result(null_result)
    return {"status": "received", "attempt_id": result.attempt_id}

@app.get("/api/v1/assumptions")
async def get_assumptions(confidence_min: float = 0.6):
    """Get inferred assumptions"""
    assumptions = phi15_engine.assumption_manager.get_top_assumptions(k=100)
    filtered = [a for a in assumptions if a.confidence >= confidence_min]
    return {
        "assumptions": [a.to_dict() for a in filtered],
        "count": len(filtered)
    }

@app.post("/api/v1/validate/{assumption_id}")
async def validate_assumption(assumption_id: str, result: ValidationResult):
    """Receive validation result from Stage 7"""
    phi15_engine.stage7_interface.receive_validation_result(
        assumption_id,
        result.success,
        result.improvement_score
    )
    return {"status": "updated"}
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Target**: >80% code coverage

**Test Files**:
- `test_preprocessing.py`: Test feature extraction
- `test_anomaly_detection.py`: Test anomaly detection
- `test_clustering.py`: Test clustering algorithms
- `test_abduction.py`: Test assumption generation
- `test_confidence.py`: Test confidence scoring
- `test_integration.py`: Test integration with Stages

**Example Tests**:

```python
# test_preprocessing.py
import pytest
from rese.phase1.tacit_assumption_miner import FailurePreprocessor, NullResult, ErrorType

def test_extract_features():
    preprocessor = FailurePreprocessor()

    null_result = NullResult(
        attempt_id="test_001",
        timestamp=datetime.now(),
        problem_type="optimization",
        approach_type="deterministic",
        constraints=["constraint_1"],
        error_type=ErrorType.OPTIMIZATION_FAILED,
        error_message="Optimization failed to converge",
        state={"x": 1.0},
        iteration=100,
        resources_used={"cpu": 0.5, "memory": 100}
    )

    features = preprocessor.extract_features(null_result)

    assert features.attempt_id == "test_001"
    assert features.problem_type == "optimization"
    assert features.approach_type == "deterministic"
    assert features.error_type == ErrorType.OPTIMIZATION_FAILED
    assert features.iteration == 100

def test_keyword_extraction():
    preprocessor = FailurePreprocessor()

    message = "Optimization failed due to numerical instability in gradient computation"
    keywords = preprocessor._extract_keywords(message)

    assert "optimization" in keywords
    assert "failed" in keywords
    assert "numerical" in keywords
```

### 5.2 Integration Tests

**Scenarios**:
1. End-to-end: Stage 6 → Φ₁.₅ → Stage 1
2. Feedback loop: Stage 7 → Φ₁.₅ (confidence update)
3. Paradigm shift detection

**Example Integration Test**:

```python
# test_integration.py
import pytest
from rese.phase1.tacit_assumption_miner import Phi15Engine, NullResult, ErrorType

def test_end_to_end_pipeline():
    engine = Phi15Engine()

    # Simulate Stage 6 sending null results
    null_results = []
    for i in range(20):
        result = NullResult(
            attempt_id=f"test_{i}",
            timestamp=datetime.now(),
            problem_type="optimization",
            approach_type="deterministic",
            constraints=[f"constraint_{j}" for j in range(5)],
            error_type=ErrorType.OPTIMIZATION_FAILED,
            error_message=f"Optimization attempt {i} failed",
            state={"iteration": i},
            iteration=i,
            resources_used={"cpu": i * 0.1}
        )
        null_results.append(result)
        engine.stage6_interface.receive_null_result(result)

    # Process
    assumptions, paradigm_rec = engine.process_full()

    # Verify
    assert len(assumptions) > 0
    assert all(a.confidence >= 0 for a in assumptions)
    assert paradigm_rec is not None
```

### 5.3 Validation Tests

**Test Cases with Known Assumptions**:

Create synthetic problems where we know the hidden constraint:
1. Problem: "Find exact solution to NP-hard problem"
   - Hidden constraint: "Must solve exactly"
   - Expected assumption: "Approximation is acceptable"

2. Problem: "Solve using only deterministic methods"
   - Hidden constraint: "Determinism required"
   - Expected assumption: "Randomization can help"

Measure if Φ₁.₅ infers these correctly (>70% accuracy target).

---

## 6. Deployment Plan

### 6.1 Development Environment

**Setup**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest rese/tests/test_phi15.py

# Run demonstration
python -m rese.phase1.tacit_assumption_miner
```

**requirements.txt**:
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
networkx>=3.1
torch>=2.0.0  # Optional, for deep learning
transformers>=4.30.0  # Optional, for semantic similarity
fastapi>=0.100.0  # Optional, for REST API
uvicorn>=0.23.0  # Optional, for REST API
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
```

### 6.2 Production Deployment

**Options**:
1. **Local Deployment**: Run as part of RESE engine
2. **Container Deployment**: Docker container
3. **Microservice**: Deploy as standalone service

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rese/ ./rese/

EXPOSE 8000

CMD ["uvicorn", "rese.phase1.tacit_assumption_miner:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Risk Mitigation

### 7.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Poor clustering quality | High | Medium | Use ensemble methods, validate with synthetic data |
| Low assumption accuracy | High | Medium | Iterative refinement, human-in-the-loop |
| Performance bottlenecks | Medium | Low | Incremental processing, caching |
| Integration issues | Medium | Low | Clear API contracts, thorough testing |

### 7.2 Validation Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Cannot achieve >70% accuracy | High | Low | Start with simpler cases, iterate |
| Overfitting to historical cases | Medium | Medium | Cross-validation, holdout test sets |
| Paradigm shift false positives | Medium | Medium | Conservative thresholds, human review |

---

## 8. Performance Optimization

### 8.1 Performance Targets

- **Latency**: <10 seconds for processing 100 failures
- **Throughput**: >1000 failures/hour
- **Memory**: <2GB for 10,000 failures in database
- **Storage**: <100MB for 10,000 failures

### 8.2 Optimization Strategies

1. **Incremental Processing**:
   - Update clusters online instead of full re-clustering
   - Process in batches

2. **Caching**:
   - Cache anomaly detection results
   - Cache cluster assignments
   - Cache historical matches

3. **Parallelization**:
   - Parallelize feature extraction
   - Parallelize clustering
   - Use multiprocessing for confidence scoring

4. **Approximation**:
   - Use approximate clustering (MiniBatchKMeans)
   - Limit candidate assumptions per cluster
   - Use sampling for large datasets

---

## 9. Documentation Plan

### 9.1 Code Documentation

- **Docstrings**: All public methods
- **Type Hints**: All function signatures
- **Comments**: Complex algorithms
- **README**: Setup and usage

### 9.2 API Documentation

**File**: `rese/docs/phi15_api.md`

Sections:
- Overview
- Installation
- Quick Start
- API Reference
- Examples
- Configuration

### 9.3 User Documentation

**File**: `rese/docs/phi15_user_guide.md`

Sections:
- What is Φ₁.₅?
- How to use
- Interpreting results
- Best practices
- Troubleshooting

---

## 10. Milestone Timeline

### Week 11 (Days 1-7): Core Infrastructure
- [ ] Implement data structures
- [ ] Implement FailurePreprocessor
- [ ] Implement FailureDatabase
- [ ] Write unit tests for data structures
- [ ] **Deliverable**: Core infrastructure working

### Week 12 (Days 8-14): Anomaly Detection & Clustering
- [ ] Implement AnomalyDetector
- [ ] Implement FailureClusterer
- [ ] Write unit tests
- [ ] Test on synthetic data
- [ ] **Deliverable**: Anomaly detection and clustering working

### Week 13 (Days 15-21): Abductive Inference
- [ ] Implement AssumptionGenerator
- [ ] Implement PatternMatcher
- [ ] Build historical paradigm shift database
- [ ] Write unit tests
- [ ] **Deliverable**: Assumption generation working

### Week 14 (Days 22-28): Confidence Scoring
- [ ] Implement ConfidenceScorer
- [ ] Implement AssumptionManager
- [ ] Implement deduplication
- [ ] Write unit tests
- [ ] **Deliverable**: Confidence scoring working

### Week 15 (Days 29-35): Integration
- [ ] Implement Stage 6 interface
- [ ] Implement Stage 1 interface
- [ ] Implement Stage 7 interface
- [ ] Write integration tests
- [ ] **Deliverable**: Full integration working

### Week 16 (Days 36-42): Paradigm Shift & Assembly
- [ ] Implement ParadigmShiftDetector
- [ ] Implement Phi15Engine
- [ ] End-to-end testing
- [ ] Documentation
- [ ] **Deliverable**: Complete Φ₁.₅ system

---

## Summary

**Implementation Plan Complete**:

✅ **Data Structures**: All classes defined with type hints
✅ **Component Implementation**: 6-week plan with daily tasks
✅ **Integration**: Stage 6, 1, 7 interfaces specified
✅ **Testing**: Unit, integration, validation tests planned
✅ **Deployment**: Development and production deployment strategies
✅ **Risk Mitigation**: Technical and validation risks addressed
✅ **Performance**: Optimization strategies and targets defined
✅ **Documentation**: Code, API, and user documentation planned
✅ **Timeline**: Week-by-week milestones

**Next Steps**:
1. Validation Strategy (phi15_validation_strategy.md)
2. Begin implementation (Week 11)

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: Implementation plan complete, ready for validation strategy
