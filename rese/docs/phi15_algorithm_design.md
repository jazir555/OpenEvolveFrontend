# Φ₁.₅ Algorithm Design: Automated Kuhnian Paradigm Shift System

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: KEY INNOVATION - Algorithm Design
**Target**: >70% assumption mining accuracy

---

## Executive Summary

This document presents the detailed algorithm design for Φ₁.₅ - an automated system that infers tacit (hidden) constraints from null results and failure patterns. The design integrates statistical anomaly detection, machine learning pattern recognition, abductive inference, and counterfactual reasoning to systematically uncover paradigm-level assumptions.

**Core Innovation**: Transform null results from "failures" into "paradigm shift signals" by:
1. Detecting accumulating anomalies
2. Clustering failure patterns
3. Inferring hidden constraints via abduction
4. Suggesting paradigm shifts with confidence scores

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Input/Output Specifications](#inputoutput-specifications)
3. [Core Algorithm Components](#core-algorithm-components)
4. [Stage 6 Integration](#stage-6-integration)
5. [Stage 1 Feedback Loop](#stage-1-feedback-loop)
6. [Confidence Scoring](#confidence-scoring)
7. [Paradigm Shift Detection](#paradigm-shift-detection)
8. [Algorithm Complexity](#algorithm-complexity)
9. [Pseudocode](#pseudocode)
10. [Integration with SCE](#integration-with-sce)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Φ₁.₅ TACIT ASSUMPTION MINER            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐                     │
│  │  Stage 6     │────▶│  Failure     │                     │
│  │  (Null       │     │  Collector   │                     │
│  │   Results)   │     └──────┬───────┘                     │
│  └──────────────┘            │                               │
│                             ▼                               │
│                    ┌───────────────┐                        │
│                    │  Failure      │                        │
│                    │  Preprocessor │                        │
│                    └───────┬───────┘                        │
│                            │                                 │
│                            ▼                                 │
│                    ┌───────────────┐                        │
│                    │  Pattern      │                        │
│                    │  Analyzer     │                        │
│                    └───────┬───────┘                        │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐             │
│         ▼                  ▼                  ▼             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │Anomaly   │      │Cluster   │      │Abductive │          │
│  │Detector  │      │Analyzer  │      │Inference│          │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘          │
│       │                 │                 │                 │
│       └─────────────────┼─────────────────┘                 │
│                         ▼                                   │
│                 ┌───────────────┐                            │
│                 │  Assumption   │                            │
│                 │  Generator    │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 ┌───────────────┐                            │
│                 │  Confidence   │                            │
│                 │  Scorer       │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 ┌───────────────┐                            │
│                 │  Paradigm     │                            │
│                 │  Shift        │                            │
│                 │  Detector     │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│        ┌────────────────┴────────────────┐                  │
│        ▼                                 ▼                  │
│  ┌─────────────┐                 ┌─────────────┐           │
│  │  Stage 1    │                 │  Stage 7    │           │
│  │  (Add       │                 │  (Validate) │           │
│  │   Constraints)│               └─────────────┘           │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

**1. Failure Collector** (Input Interface)
- Receives null results from Stage 6
- Extracts structured failure information
- Maintains failure database

**2. Failure Preprocessor**
- Feature extraction from failures
- Normalization and encoding
- Temporal ordering

**3. Pattern Analyzer**
- Detects anomalies in failure distribution
- Clusters similar failures
- Identifies systematic patterns

**4. Anomaly Detector**
- Statistical outlier detection
- Change point detection
- Temporal anomaly identification

**5. Cluster Analyzer**
- Groups failures by similarity
- Identifies archetypal failure modes
- Extracts cluster features

**6. Abductive Inference Engine**
- Generates candidate explanations
- Uses counterfactual reasoning
- Applies abduction rules

**7. Assumption Generator**
- Converts explanations to constraint format
- Generates formal representations
- Links to SCE constraint structure

**8. Confidence Scorer**
- Scores assumptions by multiple factors
- Ranks candidates
- Filters low-confidence results

**9. Paradigm Shift Detector**
- Monitors assumption accumulation
- Detects paradigm crisis signals
- Triggers paradigm shift recommendations

---

## 2. Input/Output Specifications

### 2.1 Input: From Stage 6 (Error Source Analysis)

**Data Structure**:

```python
@dataclass
class NullResult:
    """
    Null result from Stage 6 Error Source Analysis.

    Attributes:
        attempt_id: Unique identifier for the attempt
        timestamp: When the attempt occurred
        problem: The problem being solved
        approach: Approach/algorithm used
        constraints: Explicit constraints applied
        error_type: Type of error/failure
        error_message: Error description
        state: Final state when failure occurred
        iteration: Iteration number when failed
        resources_used: Computational resources consumed
        metadata: Additional context
    """
    attempt_id: str
    timestamp: datetime
    problem: ProblemRepresentation
    approach: ApproachType
    constraints: List[ConstraintID]
    error_type: ErrorType  # e.g., OPTIMIZATION_FAILED, CONSTRAINT_VIOLATION
    error_message: str
    state: StateRepresentation
    iteration: int
    resources_used: ResourceUsage
    metadata: Dict[str, Any]
```

**Error Types** (Enum):
- `OPTIMIZATION_FAILED`: Could not find feasible solution
- `DIVERGENCE`: Algorithm diverged
- `CYCLE_DETECTION`: Entered infinite loop
- `CONSTRAINT_VIOLATION`: Violated explicit constraint
- `TIMEOUT`: Exceeded time limit
- `NUMERICAL_INSTABILITY`: Numerical errors
- `INFEASIBILITY`: Problem proven infeasible
- `UNKNOWN_FAILURE`: Unclassified failure

### 2.2 Output: To Stage 1 (Prompt Analysis)

**Data Structure**:

```python
@dataclass
class TacitAssumption:
    """
    Inferred tacit assumption to add as constraint.

    Attributes:
        id: Unique identifier
        description: Human-readable description
        formalization: SCE constraint formalization
        assumption_type: Category of assumption
        confidence: Confidence score (0-1)
        support: Number of failures explained
        evidence: IDs of supporting failures
        pattern_type: What pattern led to inference
        constraint_relaxation: How to relax this constraint
        paradigm_implication: If true, indicates paradigm shift needed
        alternative_paradigm: Suggested alternative paradigm (if paradigm_implication)
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

class AssumptionType(Enum):
    """Categories of tacit assumptions"""
    ONTOLOGICAL = "ontological"  # Assumptions about what exists
    METHODOLOGICAL = "methodological"  # Assumptions about how to solve
    CONSTRAINT = "constraint"  # Hidden constraints
    REPRESENTATIONAL = "representational"  # Assumptions about modeling

class PatternType(Enum):
    """Types of failure patterns"""
    REPEATED_FAILURE = "repeated_failure"  # Same approach fails repeatedly
    SYSTEMATIC_VIOLATION = "systematic_violation"  # Constraint always violated
    CONVERGENCE_TO_BOUNDARY = "convergence_to_boundary"  # Always hits same limit
    CROSS_DOMAIN_FAILURE = "cross_domain_failure"  # Fails across different approaches
    SCALE_DEPENDENT = "scale_dependent"  # Fails only at certain scales
```

### 2.3 Output: To Stage 7 (Validation)

**Validation Request**:

```python
@dataclass
class AssumptionValidationRequest:
    """
    Request to validate an inferred assumption.

    Attributes:
        assumption: The assumption to validate
        validation_type: How to validate (simulation, experiment, relaxation)
        prediction: What should happen if assumption is correct
        test_protocol: How to test the assumption
    """
    assumption: TacitAssumption
    validation_type: ValidationType
    prediction: str
    test_protocol: str

class ValidationType(Enum):
    SIMULATION = "simulation"  # Test via simulation
    RELAXATION = "relaxation"  # Relax constraint and re-solve
    COUNTEREXAMPLE = "counterexample"  # Find counterexample
    HISTORICAL_MATCH = "historical_match"  # Match to historical paradigm shifts
```

---

## 3. Core Algorithm Components

### 3.1 Component 1: Failure Preprocessing

**Algorithm**:

```
Algorithm: PreprocessFailures(null_results: List<NullResult>)
    Output: PreprocessedFailures

    For each null_result in null_results:
        1. Extract features:
           - Structural: problem type, approach type, constraint types
           - Temporal: timestamp, iteration, time_to_failure
           - Numerical: error magnitude, resource usage
           - Contextual: metadata keywords

        2. Normalize features:
           - Scale numerical features to [0, 1]
           - One-hot encode categorical features
           - Embed text descriptions (e.g., using sentence transformers)

        3. Create feature vector:
           feature_vector = concatenate(structural, temporal, numerical, contextual)

        4. Store in failure database with:
           - Preprocessed features
           - Original null result
           - Processing timestamp

    Return failure database
```

**Feature Extraction Details**:

```python
@dataclass
class FailureFeatures:
    """Extracted features from a null result"""
    # Structural features
    problem_type: str  # e.g., "optimization", "constraint_satisfaction"
    approach_type: str  # e.g., "deterministic", "randomized", "approximation"
    constraint_categories: List[str]  # e.g., ["linear", "continuous"]

    # Temporal features
    timestamp: datetime
    iteration: int
    time_to_failure: float  # seconds

    # Numerical features
    error_magnitude: Optional[float]  # if applicable
    resource_consumption: float  # CPU/memory usage
    constraint_violation_count: int

    # Representation
    feature_vector: np.ndarray  # Concatenated vector for ML

    # Metadata
    keywords: List[str]  # Extracted from error_message
    failure_cluster: Optional[int] = None  # Assigned later
    anomaly_score: Optional[float] = None  # Computed later
```

### 3.2 Component 2: Anomaly Detection

**Algorithm (Multi-Level Anomaly Detection)**:

```
Algorithm: DetectAnomalies(failure_database: FailureDatabase)
    Output: AnomalyScores

    // Level 1: Point Anomalies (individual failures)
    For each failure in failure_database:
        1. Compute isolation forest score:
           anomaly_score = isolation_forest.score(failure.feature_vector)

        2. Compute LOF (Local Outlier Factor):
           lof_score = LOF.score(failure.feature_vector)

        3. Combine scores:
           point_anomaly = α * isolation_score + β * lof_score

        4. Store in failure record

    // Level 2: Contextual Anomalies (unusual in context)
    For each problem_type:
        1. Group failures by problem_type

        2. Compute distribution of features within this problem_type

        3. For each failure in this group:
           contextual_anomaly = z_score(failure.feature_vector,
                                        group_mean, group_std)

    // Level 3: Temporal Anomalies (change over time)
    1. Order failures by timestamp

    2. Compute sliding window statistics:
       For window of size W:
           window_mean = mean(features in window)
           window_std = std(features in window)

    3. Detect change points using CUSUM:
       For each time point t:
           cusum[t] = max(0, cusum[t-1] + (|x[t] - μ| - threshold))

       If cusum[t] > detection_threshold:
           Flag temporal anomaly at t

    // Level 4: Collective Anomalies (groups of failures)
    1. Detect unusual clusters:
       For each cluster (from clustering):
           If cluster_size < expected_size AND cluster_compactness > threshold:
               Flag as collective anomaly
               (Small but tight cluster = specific failure mode)

    Return combined anomaly scores
```

**Anomaly Scoring Formula**:

```python
def compute_overall_anomaly_score(
    point_anomaly: float,
    contextual_anomaly: float,
    temporal_anomaly: float,
    collective_anomaly: float
) -> float:
    """
    Combine multiple anomaly signals into overall score.

    Formula:
        overall = w1*point + w2*contextual + w3*temporal + w4*collective

    Weights tuned empirically (typical: w1=0.3, w2=0.3, w3=0.2, w4=0.2)
    """
    weights = {
        'point': 0.3,
        'contextual': 0.3,
        'temporal': 0.2,
        'collective': 0.2
    }

    overall = (
        weights['point'] * point_anomaly +
        weights['contextual'] * contextual_anomaly +
        weights['temporal'] * temporal_anomaly +
        weights['collective'] * collective_anomaly
    )

    return np.clip(overall, 0, 1)  # Ensure [0, 1]
```

### 3.3 Component 3: Failure Clustering

**Algorithm (Multi-Stage Clustering)**:

```
Algorithm: ClusterFailures(failure_database: FailureDatabase)
    Output: FailureClusters

    // Stage 1: Hierarchical clustering for taxonomy
    1. Compute pairwise distance matrix:
       D[i,j] = distance(failure[i].feature_vector, failure[j].feature_vector)

    2. Apply agglomerative clustering:
       clusters = agglomerative_clustering(D, n_clusters=None, distance_threshold=T)

    3. Build dendrogram for hierarchical taxonomy

    // Stage 2: DBSCAN for dense clusters
    1. Apply DBSCAN:
       clusters = DBSCAN(eps=ε, min_samples=minPts).fit(feature_vectors)

    2. Identify:
       - Core points: dense regions
       - Border points: cluster edges
       - Noise points: outliers

    // Stage 3: Spectral clustering for non-convex clusters
    1. Build similarity graph:
       For each pair (i, j):
           similarity[i,j] = exp(-||x[i] - x[j]||^2 / (2σ^2))

    2. Compute graph Laplacian: L = D - W

    3. Compute eigenvectors of L

    4. Cluster in eigenspace using k-means

    // Stage 4: Consensus clustering
    1. Combine results from all three methods using consensus:
       For each pair (i, j):
           consensus_affinity[i,j] = (
               hierarchical_same_cluster(i,j) +
               dbscan_same_cluster(i,j) +
               spectral_same_cluster(i,j)
           ) / 3

    2. Apply final clustering on consensus affinity

    // Stage 5: Cluster characterization
    For each cluster:
        1. Compute centroid: mean of feature vectors
        2. Compute compactness: mean distance to centroid
        3. Compute size: number of failures in cluster
        4. Extract common features:
           - Most frequent problem types
           - Most frequent error types
           - Most frequent constraint categories
           - Keyword extraction from error messages

    Return labeled clusters with characterizations
```

**Cluster Quality Metrics**:

```python
@dataclass
class ClusterQuality:
    """Quality metrics for a failure cluster"""
    cluster_id: int
    size: int
    compactness: float  # Lower is better
    separation: float  # Distance to nearest other cluster
    silhouette_score: float  # [-1, 1], higher is better
    stability: float  # Across clustering methods

    def is_candidate_for_assumption_mining(self) -> bool:
        """
        Determine if cluster is worth analyzing for assumptions.

        Criteria:
        - Sufficient size (>= min_failures)
        - High compactness (failures are similar)
        - High stability (robust across methods)
        """
        MIN_SIZE = 5
        MAX_COMPACTNESS = 0.5
        MIN_STABILITY = 0.7

        return (
            self.size >= MIN_SIZE and
            self.compactness <= MAX_COMPACTNESS and
            self.stability >= MIN_STABILITY
        )
```

### 3.4 Component 4: Abductive Inference

**Algorithm (Generate Explanations via Abduction)**:

```
Algorithm: AbduceAssumptions(cluster: FailureCluster)
    Output: List<AssumptionCandidate>

    // Abduction: Inference to best explanation
    // Given: These failures occurred
    // Find: Assumptions that best explain them

    candidates = []

    // Step 1: Generate candidate assumptions

    // Method 1: Constraint Violation Analysis
    For each failure in cluster:
        For each constraint in failure.constraints:
            1. Analyze how constraint was violated:
               - Was it always violated?
               - Was it barely violated (close to threshold)?
               - Was it fundamentally incompatible?

            2. Generate candidate:
               "Assumption: {constraint} must hold"
               confidence = fraction_of_failures_with_violation

            3. Add to candidates

    // Method 2: Counterfactual Reasoning
    1. Ask: "What if we relax this constraint?"
       counterfactual_scenario = relax_constraint(constraint)

    2. Simulate: Would failures disappear?
       If yes:
           candidate = "This constraint is blocking progress"
           confidence = predicted_improvement

    3. Add to candidates

    // Method 3: Pattern Matching to Historical Paradigm Shifts
    1. Represent cluster as feature vector

    2. Compare to database of historical paradigm shifts:
       For each historical_shift in paradigm_database:
           similarity = cosine_similarity(cluster.features, historical_shift.features)

           If similarity > threshold:
               candidate = historical_shift.tacit_assumption
               confidence = similarity
               Add to candidates

    // Method 4: Boundary Analysis
    1. Identify constraints that failures "push against":
       For each constraint:
           boundary_violations = count(failures at constraint boundary)

           If boundary_violations > threshold:
               candidate = "Assumption: {constraint} is a hard limit"
               confidence = boundary_violations / cluster.size

    2. Add to candidates

    // Step 2: Deduplicate candidates
    unique_candidates = merge_similar_candidates(candidates)

    // Step 3: Score candidates
    For each candidate in unique_candidates:
        score_abductive_confidence(candidate, cluster)

    Return sorted candidates by confidence
```

**Abductive Confidence Scoring**:

```python
def score_abductive_confidence(
    candidate: AssumptionCandidate,
    cluster: FailureCluster
) -> float:
    """
    Score how well this candidate explains the cluster.

    Factors:
    1. Support: How many failures does it explain?
    2. Simplicity: Parsimonious explanation (Occam's razor)
    3. Coherence: Internally consistent
    4. Novelty: Different from explicit constraints
    5. Testability: Can be validated
    """

    # 1. Support
    support = len(candidate.explains_failures) / cluster.size

    # 2. Simplicity (inverse of complexity)
    simplicity = 1 / (1 + candidate.complexity)

    # 3. Coherence (consistency with known constraints)
    coherence = 1 - candidate.contradiction_count / total_constraints

    # 4. Novelty (how different from explicit constraints)
    novelty = 1 - max_similarity(candidate, explicit_constraints)

    # 5. Testability
    testability = 1 if candidate.testable else 0.5

    # Combine (weights tuned empirically)
    confidence = (
        0.3 * support +
        0.2 * simplicity +
        0.2 * coherence +
        0.15 * novelty +
        0.15 * testability
    )

    return np.clip(confidence, 0, 1)
```

### 3.5 Component 5: Assumption Generation

**Algorithm (Convert Explanations to Constraints)**:

```
Algorithm: GenerateAssumptions(candidates: List<AssumptionCandidate>)
    Output: List<TacitAssumption>

    For each candidate in candidates:

        1. Formalize as SCE constraint:
           constraint = convert_to_sce_format(candidate.description)

        2. Determine assumption type:
           If candidate.ontology_related:
               type = ONTOLOGICAL
           Else if candidate.methodology_related:
               type = METHODOLOGICAL
           Else if candidate.constraint_related:
               type = CONSTRAINT
           Else:
               type = REPRESENTATIONAL

        3. Generate constraint relaxation:
           relaxation = "Relax: {candidate.constraint_description}"
           relaxation += " → Allow: {candidate.alternative}"

        4. Check paradigm implication:
           If candidate.changes_fundamental_model:
               paradigm_implication = True
               alternative_paradigm = suggest_alternative(candidate)
           Else:
               paradigm_implication = False
               alternative_paradigm = None

        5. Create TacitAssumption object:
           assumption = TacitAssumption(
               id = generate_id(),
               description = candidate.description,
               formalization = constraint,
               assumption_type = type,
               confidence = candidate.confidence,
               support = len(candidate.explains_failures),
               evidence = [f.id for f in candidate.explains_failures],
               pattern_type = candidate.pattern_type,
               constraint_relaxation = relaxation,
               paradigm_implication = paradigm_implication,
               alternative_paradigm = alternative_paradigm
           )

        6. Add to assumptions list

    Return assumptions sorted by confidence
```

**SCE Constraint Conversion**:

```python
def convert_to_sce_format(description: str) -> str:
    """
    Convert natural language assumption to SCE constraint format.

    Examples:
    - "Must use exact algorithms" → "forall (alg : Algorithm), isExact(alg)"
    - "Time must be polynomial" → "forall (t : Time), t ∈ Polynomial"
    - "Determinism required" → "forall (p : Process), isDeterministic(p)"
    """

    # Use LLM-based translation with templates
    templates = {
        "exactness": "forall (x : Entity), isExact(x)",
        "determinism": "forall (p : Process), isDeterministic(p)",
        "continuity": "forall (x : Space), isContinuous(x)",
        "linearity": "forall (f : Function), isLinear(f)",
        # ... more templates
    }

    # Match description to template (using semantic similarity)
    best_match = find_best_template(description, templates)
    formalization = templates[best_match]

    return formalization
```

### 3.6 Component 6: Confidence Scoring

**Algorithm (Multi-Factor Confidence Scoring)**:

```
Algorithm: ScoreConfidence(assumption: TacitAssumption,
                            cluster: FailureCluster,
                            database: AssumptionDatabase)
    Output: ConfidenceScore (0-1)

    // Factor 1: Support (how many failures explained?)
    support_score = assumption.support / cluster.size

    // Factor 2: Pattern strength (how strong is the pattern?)
    pattern_score = cluster.silhouette_score  // Compact, well-separated

    // Factor 3: Counterfactual validation (would relaxing it fix problems?)
    counterfactual_score = simulate_relaxation(assumption, cluster)

    // Factor 4: Novelty (is this different from explicit constraints?)
    explicit_constraints = get_explicit_constraints()
    novelty_score = 1 - max_similarity(assumption, explicit_constraints)

    // Factor 5: Historical precedence (has this appeared before?)
    historical_matches = database.find_similar_assumptions(assumption)
    historical_score = max(historical.confidence for historical in historical_matches)

    // Factor 6: Testability (can we validate this?)
    testability_score = 1.0 if is_testable(assumption) else 0.5

    // Factor 7: Paradigm plausibility (does alternative make sense?)
    if assumption.paradigm_implication:
        paradigm_score = evaluate_paradigm_coherence(assumption.alternative_paradigm)
    else:
        paradigm_score = 1.0  // N/A

    // Combine with learned weights
    weights = {
        'support': 0.25,
        'pattern': 0.20,
        'counterfactual': 0.20,
        'novelty': 0.10,
        'historical': 0.10,
        'testability': 0.10,
        'paradigm': 0.05
    }

    confidence = (
        weights['support'] * support_score +
        weights['pattern'] * pattern_score +
        weights['counterfactual'] * counterfactual_score +
        weights['novelty'] * novelty_score +
        weights['historical'] * historical_score +
        weights['testability'] * testability_score +
        weights['paradigm'] * paradigm_score
    )

    Return confidence
```

### 3.7 Component 7: Paradigm Shift Detection

**Algorithm (Detect Paradigm Crisis)**:

```
Algorithm: DetectParadigmShift(assumptions: List<TacitAssumption>,
                                history: AssumptionHistory)
    Output: ParadigmShiftRecommendation

    // Kuhnian crisis signals:

    // Signal 1: Accumulation of anomalies
    recent_assumptions = [a for a in assumptions if a.timestamp > window_start]
    anomaly_count = len(recent_assumptions)

    // Signal 2: Increasing frequency of tacit assumptions
    historical_rate = history.average_assumptions_per_week()
    current_rate = len(recent_assumptions) / weeks_in_window
    frequency_ratio = current_rate / historical_rate

    // Signal 3: High-confidence paradigm-level assumptions
    paradigm_assumptions = [a for a in assumptions
                            if a.paradigm_implication and a.confidence > threshold]

    // Signal 4: Cross-domain failure patterns
    // (Failures across different problem types suggesting fundamental issue)
    problem_types = set(a.problem_type for a in assumptions)
    cross_domain_signal = len(problem_types) >= 3

    // Signal 5: Historical paradigm shift pattern match
    // Compare current pattern to historical paradigm shifts
    historical_similarity = compare_to_paradigm_shifts(assumptions, history)

    // Combine signals
    crisis_score = (
        0.25 * min(anomaly_count / 10, 1.0) +  // Cap at 10 anomalies
        0.25 * min(frequency_ratio / 3.0, 1.0) +  // Cap at 3x rate
        0.25 * len(paradigm_assumptions) / 5.0 +  // Cap at 5 assumptions
        0.15 * (1.0 if cross_domain_signal else 0.0) +
        0.10 * historical_similarity
    )

    // Decision
    If crisis_score > CRISIS_THRESHOLD:  // e.g., 0.7
        Return ParadigmShiftRecommendation(
            trigger = True,
            confidence = crisis_score,
            primary_assumptions = paradigm_assumptions,
            suggested_alternatives = [a.alternative_paradigm for a in paradigm_assumptions],
            explanation = generate_crisis_explanation(assumptions, history)
        )
    Else:
        Return ParadigmShiftRecommendation(
            trigger = False,
            confidence = crisis_score,
            primary_assumptions = [],
            suggested_alternatives = [],
            explanation = "No paradigm crisis detected"
        )
```

**Paradigm Shift Recommendation Structure**:

```python
@dataclass
class ParadigmShiftRecommendation:
    """Recommendation for paradigm shift"""
    trigger: bool  # Whether to recommend shift
    confidence: float  # Confidence in recommendation
    primary_assumptions: List[TacitAssumption]  # Key assumptions causing crisis
    suggested_alternatives: List[str]  # Alternative paradigms to try
    explanation: str  # Human-readable explanation

    def to_dict(self) -> Dict:
        """Convert to dictionary for Stage 1 communication"""
        return {
            "trigger": self.trigger,
            "confidence": self.confidence,
            "assumptions_to_relax": [a.id for a in self.primary_assumptions],
            "alternative_paradigms": self.suggested_alternatives,
            "explanation": self.explanation,
            "priority": "HIGH" if self.trigger and self.confidence > 0.8 else "MEDIUM"
        }
```

---

## 4. Stage 6 Integration

### 4.1 Input Interface from Stage 6

**Integration Point**: Φ₁.₅ receives null results from Stage 6 (Error Source Analysis)

**Protocol**:

```python
class Phi15Stage6Interface:
    """Interface for receiving null results from Stage 6"""

    def receive_null_result(self, result: NullResult) -> None:
        """
        Receive a single null result from Stage 6.

        Args:
            result: Null result with failure information
        """
        # Add to failure database
        self.failure_database.add(result)

        # Trigger incremental processing
        if self.should_process_incrementally():
            self.process_incremental()

    def receive_batch_null_results(self, results: List[NullResult]) -> None:
        """
        Receive batch of null results from Stage 6.

        Args:
            results: List of null results
        """
        # Add all to database
        for result in results:
            self.failure_database.add(result)

        # Trigger full processing
        self.process_full()

    def should_process_incrementally(self) -> bool:
        """
        Decide whether to process incrementally.

        Criteria:
        - Enough new results accumulated (e.g., >10)
        - Time since last processing (e.g., >1 hour)
        - High anomaly rate detected
        """
        new_results = self.failure_database.count_unprocessed()

        return (
            new_results >= self.config.INCREMENTAL_THRESHOLD or
            self.time_since_last_processing() >= self.config.INCREMENTAL_TIME or
            self.current_anomaly_rate() >= self.config.ANOMALY_RATE_THRESHOLD
        )
```

### 4.2 Error Source Analysis Synergy

**How Φ₁.₅ uses Stage 6 error classification**:

1. **Error Type Analysis**
   - Stage 6 classifies error type (e.g., CONSTRAINT_VIOLATION)
   - Φ₁.₅ uses this to guide assumption mining
   - Example: If all errors are CONSTRAINT_VIOLATION on same constraint → that constraint is problematic

2. **Error Context Extraction**
   - Stage 6 provides context (state, iteration, resources)
   - Φ₁.₅ uses this for feature extraction
   - Example: Always failing at same iteration → systematic constraint issue

3. **Error Source Attribution**
   - Stage 6 identifies source (e.g., numerical instability)
   - Φ₁.₅ combines with pattern analysis
   - Example: Numerical issues across different approaches → fundamental representation problem

---

## 5. Stage 1 Feedback Loop

### 5.1 Output Interface to Stage 1

**Integration Point**: Φ₁.₅ sends inferred assumptions to Stage 1 (Prompt Analysis)

**Protocol**:

```python
class Phi15Stage1Interface:
    """Interface for sending inferred assumptions to Stage 1"""

    def send_assumptions(self, assumptions: List[TacitAssumption]) -> None:
        """
        Send inferred assumptions to Stage 1.

        Args:
            assumptions: List of inferred tacit assumptions

        Action:
            Stage 1 will:
            - Add assumptions as new constraints in SCE
            - Reformulate problem with relaxed constraints
            - Trigger new solving attempts
        """
        # Filter by confidence
        high_confidence = [a for a in assumptions if a.confidence >= self.config.CONFIDENCE_THRESHOLD]

        # Sort by confidence
        high_confidence.sort(key=lambda a: a.confidence, reverse=True)

        # Send to Stage 1
        for assumption in high_confidence:
            self.stage1_client.add_inferred_constraint(
                constraint_id=assumption.id,
                description=assumption.description,
                formalization=assumption.formalization,
                source="phi15_inferred",
                confidence=assumption.confidence,
                relaxation=assumption.constraint_relaxation
            )

    def send_paradigm_shift_recommendation(self, recommendation: ParadigmShiftRecommendation) -> None:
        """
        Send paradigm shift recommendation to Stage 1.

        Args:
            recommendation: Paradigm shift recommendation

        Action:
            Stage 1 will:
            - Flag high-priority paradigm issue
            - Present alternatives to user
            - Request guidance on paradigm selection
        """
        if recommendation.trigger:
            self.stage1_client.alert_paradigm_crisis(
                confidence=recommendation.confidence,
                assumptions_to_relax=[a.id for a in recommendation.primary_assumptions],
                alternative_paradigms=recommendation.suggested_alternatives,
                explanation=recommendation.explanation,
                priority="HIGH" if recommendation.confidence > 0.8 else "MEDIUM"
            )
```

### 5.2 Constraint Addition Workflow

**How Stage 1 integrates Φ₁.₅ assumptions**:

1. **Receive Assumption**
   - Stage 1 receives `TacitAssumption` from Φ₁.₅
   - Parse SCE formalization

2. **Add to SCE**
   ```python
   # In Stage 1
   new_constraint = Constraint(
       id=assumption.id,
       type=ConstraintType.SOFT,  # Inferred, so start as soft
       description=assumption.description,
       formalization=assumption.formalization,
       source="phi15_inferred",
       confidence=assumption.confidence
   )

   sce.add_constraint(new_constraint)
   ```

3. **Reformulate Problem**
   - Stage 1 reformulates with new constraint
   - Example: "Find solution WITHOUT assumption X"

4. **Trigger New Attempts**
   - Send reformulated problem to solving stages
   - Track if failures disappear

5. **Feedback to Φ₁.₅**
   - If failures disappear → validates assumption
   - If failures persist → lowers confidence

---

## 6. Confidence Scoring (Detailed)

### 6.1 Mathematical Model

**Overall Confidence Score**:

```
confidence(assumption) = Σᵢ wᵢ * scoreᵢ(assumption)

Where:
- wᵢ are learned weights (Σᵢ wᵢ = 1)
- scoreᵢ(assumption) are individual factor scores in [0, 1]
```

**Individual Factors**:

1. **Support Score**:
   ```
   support = |Fₐ| / |F|
   Where:
   - Fₐ = Failures explained by assumption
   - F = All failures in cluster
   ```

2. **Pattern Score**:
   ```
   pattern = silhouette_score(cluster) ∈ [-1, 1]
   Normalize to [0, 1]: pattern' = (pattern + 1) / 2
   ```

3. **Counterfactual Score**:
   ```
   counterfactual = P(success | relax(assumption))
   Estimated via simulation on subset of failures
   ```

4. **Novelty Score**:
   ```
   novelty = 1 - max(sim(assumption, c) for c in explicit_constraints)
   Where sim() is semantic similarity
   ```

5. **Historical Score**:
   ```
   historical = max(confidence(h) for h in similar_historical_assumptions(assumption))
   ```

### 6.2 Weight Learning

**Approach**: Learn weights from historical paradigm shift cases

**Algorithm**:

```
Algorithm: LearnWeights(historical_cases: List<ParadigmShiftCase>)
    Output: OptimalWeights

    // Each case has:
    // - Assumptions that were actually true
    // - Features for each assumption
    // - Binary label (correct/incorrect)

    // Optimize weights to maximize accuracy
    objective(weights) = Σ_case log P(correct_assumption | weights, features)

    // Use gradient descent or Bayesian optimization
    optimal_weights = optimize(objective, initial_weights, constraints)

    Return optimal_weights
```

**Constraints**: Σ wᵢ = 1, wᵢ ≥ 0

---

## 7. Paradigm Shift Detection (Detailed)

### 7.1 Kuhnian Crisis Indicators

**Quantified Kuhnian Crisis Signals**:

1. **Anomaly Accumulation**:
   ```
   anomaly_score = min(N_anomalies / 10, 1.0)
   ```
   Threshold: >5 anomalies in recent window

2. **Anomaly Rate Increase**:
   ```
   rate_ratio = current_anomaly_rate / historical_baseline_rate
   rate_score = min(rate_ratio / 3.0, 1.0)
   ```
   Threshold: >2x historical rate

3. **Paradigm-Level Assumptions**:
   ```
   paradigm_score = N_paradigm_assumptions / 5.0
   ```
   Threshold: >3 paradigm-level assumptions

4. **Cross-Domain Failures**:
   ```
   domain_diversity = |unique_problem_types|
   domain_score = 1.0 if domain_diversity >= 3 else 0.5
   ```

5. **Historical Pattern Match**:
   ```
   historical_score = max_similarity(current_pattern, historical_paradigm_shifts)
   ```

### 7.2 Crisis Decision Function

```
IsParadigmCrisis():
    score = (
        0.25 * anomaly_score +
        0.25 * rate_score +
        0.25 * paradigm_score +
        0.15 * domain_score +
        0.10 * historical_score
    )

    Return score > CRISIS_THRESHOLD  // e.g., 0.7
```

### 7.3 Paradigm Shift Recommendation Structure

**Human-Readable Output**:

```python
def generate_paradigm_shift_report(
    assumptions: List[TacitAssumption],
    crisis_score: float
) -> str:
    """
    Generate human-readable paradigm shift recommendation.

    Example output:
    """
    PARADIGM CRISIS DETECTED (Confidence: 0.85)

    Key Assumptions Challenging Current Paradigm:
    1. "Determinism is required" (confidence: 0.92)
       - Supported by 23 failures
       - Pattern: All deterministic approaches fail
       - Relaxation: Allow randomized algorithms

    2. "Exact solutions are necessary" (confidence: 0.87)
       - Supported by 18 failures
       - Pattern: Exact methods hit exponential wall
       - Relaxation: Use approximation schemes

    3. "Problem must be solved in polynomial time" (confidence: 0.81)
       - Supported by 15 failures
       - Pattern: All polynomial attempts fail
       - Relaxation: Allow exponential with pruning

    Suggested Alternative Paradigm:
    "Randomized Approximation" (confidence: 0.78)
    - Use randomness to break symmetries
    - Accept approximate solutions with guarantees
    - Expected improvement: 3-5x success rate

    Historical Precedent:
    Similar pattern to:
    - Miller's randomized algorithm for primality testing (1976)
    - Karp's randomized algorithms for NP-hard problems (1980s)
    ```

    return report
```

---

## 8. Algorithm Complexity

### 8.1 Complexity Analysis

**Component-wise Complexity**:

| Component | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Failure Preprocessing | O(N · F) | O(N · F) | N = failures, F = features |
| Anomaly Detection | O(N log N) | O(N) | Isolation forest |
| Clustering | O(N²) | O(N²) | Hierarchical, can optimize |
| Abductive Inference | O(N · C) | O(C) | C = candidates per failure |
| Confidence Scoring | O(A) | O(A) | A = assumptions |
| Paradigm Detection | O(A + H) | O(H) | H = historical cases |

**Overall**:
- **Worst-case**: O(N²) due to clustering
- **Typical**: O(N log N) with optimizations (e.g., approximate clustering)
- **Incremental updates**: O(log N) per new failure

### 8.2 Optimization Strategies

1. **Incremental Processing**
   - Update clusters online (e.g., online k-means)
   - Maintain anomaly scores incrementally
   - Process in batches

2. **Approximation Algorithms**
   - Use approximate clustering (MiniBatchKMeans)
   - Sample failures for abductive inference
   - Limit candidate assumptions

3. **Parallelization**
   - Parallelize feature extraction
   - Parallelize clustering (e.g., distributed k-means)
   - Parallelize confidence scoring

4. **Caching**
   - Cache anomaly detection results
   - Cache cluster assignments
   - Cache historical matches

---

## 9. Pseudocode

### 9.1 Complete Φ₁.₅ Algorithm

```python
# High-level pseudocode for complete Φ₁.₅ system

def PHI15(null_results: List[NullResult]) -> Tuple[List[TacitAssumption], ParadigmShiftRecommendation]:
    """
    Main Φ₁.₅ algorithm: Mine tacit assumptions from null results.

    Args:
        null_results: Failed attempts from Stage 6

    Returns:
        assumptions: Inferred tacit assumptions
        paradigm_recommendation: Whether to trigger paradigm shift
    """

    # Step 1: Preprocess failures
    failures = preprocess_failures(null_results)
    # Output: Feature vectors, normalized

    # Step 2: Detect anomalies
    anomaly_scores = detect_anomalies(failures)
    # Output: Point, contextual, temporal, collective anomalies

    # Step 3: Cluster failures
    clusters = cluster_failures(failures, anomaly_scores)
    # Output: Labeled clusters with quality metrics

    # Step 4: Filter high-quality clusters
    candidate_clusters = [c for c in clusters if c.is_candidate_for_assumption_mining()]

    # Step 5: Abduce assumptions from each cluster
    all_assumptions = []
    for cluster in candidate_clusters:
        candidates = abduce_assumptions(cluster)
        all_assumptions.extend(candidates)

    # Step 6: Deduplicate assumptions
    unique_assumptions = deduplicate_assumptions(all_assumptions)

    # Step 7: Score confidence for each assumption
    for assumption in unique_assumptions:
        assumption.confidence = score_confidence(assumption, cluster, database)

    # Step 8: Sort by confidence
    unique_assumptions.sort(key=lambda a: a.confidence, reverse=True)

    # Step 9: Detect paradigm crisis
    paradigm_recommendation = detect_paradigm_shift(unique_assumptions, history)

    # Step 10: Filter and return
    high_confidence = [a for a in unique_assumptions if a.confidence >= CONFIDENCE_THRESHOLD]

    return high_confidence, paradigm_recommendation
```

### 9.2 Incremental Update Algorithm

```python
def PHI15_INCREMENTAL(new_null_results: List[NullResult], state: Phi15State):
    """
    Incremental update for Φ₁.₅ (called periodically).

    Args:
        new_null_results: New failures since last update
        state: Current Φ₁.₅ state (database, models, etc.)

    Returns:
        Updated state, new assumptions (if any)
    """

    # Step 1: Preprocess new failures
    new_failures = preprocess_failures(new_null_results)

    # Step 2: Add to database
    state.failures.extend(new_failures)

    # Step 3: Update anomaly scores (incremental)
    update_anomaly_scores(new_failures, state)

    # Step 4: Update clusters (incremental if possible)
    if should_recluster(state):
        state.clusters = recluster(state.failures)  # Full re-clustering
    else:
        update_clusters_incremental(new_failures, state)  # Online update

    # Step 5: Check for new patterns
    new_patterns = detect_new_patterns(new_failures, state)

    # Step 6: Generate assumptions from new patterns
    new_assumptions = []
    for pattern in new_patterns:
        candidates = abduce_assumptions(pattern.cluster)
        for candidate in candidates:
            candidate.confidence = score_confidence(candidate, pattern.cluster, state.database)
            new_assumptions.append(candidate)

    # Step 7: Check for paradigm shift
    paradigm_recommendation = detect_paradigm_shift(new_assumptions, state.history)

    # Step 8: Update state
    state.assumptions.extend(new_assumptions)
    state.history.update(new_failures, new_assumptions)

    return state, new_assumptions, paradigm_recommendation
```

---

## 10. Integration with SCE

### 10.1 Constraint Representation Compatibility

**Φ₁.₅ outputs must be compatible with SCE `Constraint` structure**:

```python
# In SCE (from Agent A1)
@dataclass
class Constraint:
    id: str
    type: ConstraintType  # HARD, SOFT, PREFERENCE
    description: str
    formalization: str  # Lean 4 representation
    source: str  # "phi15_inferred"
    dependencies: List[str]
    verified: bool
    lean_theorem: Optional[str]

# In Φ₁.₅
@dataclass
class TacitAssumption:
    # ... (other fields)

    def to_sce_constraint(self) -> Constraint:
        """Convert to SCE Constraint format"""
        return Constraint(
            id=self.id,
            type=ConstraintType.SOFT,  # Start as soft (inferred)
            description=f"[INFERRED] {self.description}",
            formalization=self.formalization,
            source="phi15_inferred",
            dependencies=[],  # Initially none
            verified=False,  # Not verified yet
            lean_theorem=None
        )
```

### 10.2 Feedback Loop: Validation Updates Confidence

**When Stage 7 validates an assumption**:

```python
# In Φ₁.₅
def update_assumption_from_validation(
    assumption: TacitAssumption,
    validation_result: ValidationResult
) -> TacitAssumption:
    """
    Update assumption confidence based on validation.

    Args:
        assumption: Original assumption
        validation_result: Result from Stage 7 validation

    Returns:
        Updated assumption with adjusted confidence
    """

    if validation_result.success:
        # Assumption was correct - boost confidence
        assumption.confidence = min(1.0, assumption.confidence * 1.2)
        assumption.verified = True

        # Convert to HARD constraint if confidence high enough
        if assumption.confidence > 0.9:
            assumption.constraint_type = ConstraintType.HARD
    else:
        # Assumption was incorrect - reduce confidence
        assumption.confidence = max(0.0, assumption.confidence * 0.7)

        # If confidence drops too low, mark for removal
        if assumption.confidence < 0.3:
            assumption.active = False

    return assumption
```

---

## Summary

**Φ₁.₅ Algorithm Design Complete**:

✅ **System Architecture**: 7-component pipeline from null results to paradigm shifts
✅ **Input/Output**: Integration with Stage 6 (input) and Stage 1 (output)
✅ **Core Components**: Anomaly detection, clustering, abduction, confidence scoring
✅ **Integration**: Stage 6 error analysis, Stage 1 constraint addition, Stage 7 validation
✅ **Confidence Scoring**: Multi-factor model with >70% accuracy target
✅ **Paradigm Detection**: Kuhnian crisis signals with quantitative triggers
✅ **Complexity**: O(N log N) typical, O(N²) worst-case (optimized)
✅ **SCE Integration**: Compatible constraint representation, feedback loop

**Next Steps**:
1. Implementation Plan (phi15_implementation_plan.md)
2. Validation Strategy (phi15_validation_strategy.md)

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: Algorithm design complete, ready for implementation planning
