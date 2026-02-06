# Phase 1: Stage 6 Knowledge Extraction - Complete Implementation Tasks

**Priority**: P0 (HIGHEST PRIORITY)
**Estimated Effort**: 12-15 weeks
**Status**: READY TO START
**Source**: FRM Integration Analysis Recommendation

---

## Executive Summary

**Why This is Priority**: The Integration Architecture document identifies Stage 6 Knowledge Extraction as 75% complete. This is the **highest priority gap** in the entire Decomposition Workflow. FRM does NOT address these missing components.

**Value Proposition**:
- Enables system to learn from every workflow execution
- Improves future decomposition quality through pattern recognition
- Tracks team and gauntlet performance for optimization
- Builds comprehensive knowledge base
- No new external dependencies required

---

## Component 1: KnowledgeArtifact Schema Implementation

**Effort**: 2 weeks
**Priority**: P0 (blocks all other Stage 6 work)
**Dependencies**: None

### Tasks

#### 1.1 Define KnowledgeArtifact Data Model
**File**: `workflow_structures.py`

```python
@dataclass
class KnowledgeArtifact:
    """Base class for all knowledge artifacts extracted from workflows"""
    artifact_id: str
    artifact_type: Literal["solution_pattern", "team_performance", "gauntlet_effectiveness",
                          "decomposition_strategy", "critique_insight", "verification_method"]
    source_workflow_id: str
    source_stage: Literal[0, 1, 2, 3, 4, 5, 6]
    timestamp: datetime
    confidence: float  # 0.0 to 1.0

    # Content
    title: str
    description: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]

    # Relationships
    related_artifacts: List[str]  # artifact_ids
    citations: List[str]
    tags: List[str]

    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
```

#### 1.2 Implement Specialized Artifact Types
**File**: `workflow_structures.py`

```python
@dataclass
class SolutionPatternArtifact(KnowledgeArtifact):
    """Extracted solution approaches that work across similar problems"""
    artifact_type: Literal["solution_pattern"] = "solution_pattern"
    pattern_category: str  # e.g., "dynamic_programming", "divide_and_conquer"
    problem_domains: List[str]
    approach_signature: Dict[str, Any]  # Signature for similarity matching
    success_rate: float
    avg_execution_time: float

@dataclass
class TeamPerformanceArtifact(KnowledgeArtifact):
    """Team effectiveness metrics and insights"""
    artifact_id: str
    artifact_type: Literal["team_performance"] = "team_performance"
    team_id: str
    team_composition: Dict[str, Any]  # Models, roles, etc.
    performance_metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    optimal_problem_types: List[str]

@dataclass
class GauntletEffectivenessArtifact(KnowledgeArtifact):
    """Gauntlet rule effectiveness analysis"""
    artifact_type: Literal["gauntlet_effectiveness"] = "gauntlet_effectiveness"
    gauntlet_id: str
    rule_effectiveness: Dict[str, float]  # rule_id -> effectiveness_score
    catch_rate: float  # % of issues caught
    false_positive_rate: float
    optimal_contexts: List[str]
```

#### 1.3 Add Validation Methods
**File**: `workflow_structures.py`

```python
class KnowledgeArtifact:
    def validate(self) -> bool:
        """Validate artifact completeness and consistency"""
        # Check required fields
        # Check confidence range
        # Check content schema
        # Validate relationships exist
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage"""
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """Deserialize from storage"""
        pass
```

**Deliverable**: Complete KnowledgeArtifact schema in `workflow_structures.py`

---

## Component 2: WorkflowKnowledgeExtractor

**Effort**: 3 weeks
**Priority**: P0 (core extraction logic)
**Dependencies**: Component 1 (KnowledgeArtifact schema)

### Tasks

#### 2.1 Implement Base Extractor
**File**: `workflow_knowledge_extractor.py` (NEW)

```python
class WorkflowKnowledgeExtractor:
    """Extract knowledge artifacts from completed workflows"""

    def __init__(self, knowledge_engine, ace_client):
        self.knowledge_engine = knowledge_engine
        self.ace_client = ace_client

    async def extract_from_workflow(
        self,
        workflow_state: WorkflowState
    ) -> List[KnowledgeArtifact]:
        """Extract all knowledge artifacts from a completed workflow"""
        artifacts = []

        # Extract from each stage
        artifacts.extend(await self._extract_from_stage_0(workflow_state))
        artifacts.extend(await self._extract_from_stage_1(workflow_state))
        artifacts.extend(await self._extract_from_stage_3(workflow_state))
        artifacts.extend(await self._extract_from_stage_5(workflow_state))
        artifacts.extend(await self._extract_from_stage_6(workflow_state))

        return artifacts

    async def _extract_from_stage_0(self, workflow: WorkflowState) -> List[KnowledgeArtifact]:
        """Extract domain detection patterns, complexity assessment patterns"""
        pass

    async def _extract_from_stage_1(self, workflow: WorkflowState) -> List[KnowledgeArtifact]:
        """Extract decomposition strategies, dependency patterns"""
        pass

    async def _extract_from_stage_3(self, workflow: WorkflowState) -> List[KnowledgeArtifact]:
        """Extract solution patterns, critique insights, verification methods"""
        pass

    async def _extract_from_stage_5(self, workflow: WorkflowState) -> List[KnowledgeArtifact]:
        """Extract self-healing patterns, verification effectiveness"""
        pass

    async def _extract_from_stage_6(self, workflow: WorkflowState) -> List[KnowledgeArtifact]:
        """Extract learning patterns, process optimizations"""
        pass
```

#### 2.2 Implement Solution Pattern Extraction
**File**: `workflow_knowledge_extractor.py`

```python
class SolutionPatternExtractor:
    """Extract reusable solution patterns from successful solutions"""

    async def extract_patterns(
        self,
        solutions: List[SolutionAttempt],
        context: DecompositionPlan
    ) -> List[SolutionPatternArtifact]:
        """
        Identify patterns that could be reused:
        - Algorithmic patterns (DP, greedy, divide-and-conquer)
        - Architectural patterns (MVC, microservices, event-driven)
        - Integration patterns (API, database, messaging)
        """
        patterns = []

        for solution in solutions:
            if solution.verification_report.overall_score > 0.8:  # High quality
                pattern = await self._analyze_solution_pattern(solution, context)
                if pattern:
                    patterns.append(pattern)

        return patterns

    async def _analyze_solution_pattern(
        self,
        solution: SolutionAttempt,
        context: DecompositionPlan
    ) -> Optional[SolutionPatternArtifact]:
        """Use LLM to identify pattern type and extract signature"""
        prompt = f"""
        Analyze this solution and identify the reusable pattern:

        Problem: {context.problem_statement}
        Solution: {solution.solution_code}
        Approach: {solution.approach_description}

        Identify:
        1. Pattern category (algorithmic, architectural, integration)
        2. Pattern name (e.g., dynamic_programming, MVC, REST_API)
        3. Key characteristics
        4. Applicable domains
        5. Signature for similarity matching
        """
        # Call LLM and parse response
        pass
```

#### 2.3 Implement Decomposition Strategy Extraction
**File**: `workflow_knowledge_extractor.py`

```python
class DecompositionStrategyExtractor:
    """Extract effective decomposition strategies"""

    async def extract_strategies(
        self,
        plan: DecompositionPlan,
        execution_results: Dict[str, Any]
    ) -> List[KnowledgeArtifact]:
        """
        Extract decomposition strategies that worked well:
        - Granularity level (number of sub-problems)
        - Dependency structure
        - Team assignments
        - Gauntlet assignments
        """
        strategies = []

        # Analyze what made this decomposition successful
        if execution_results.get("success_rate", 0) > 0.8:
            strategy = await self._analyze_decomposition_strategy(plan, execution_results)
            strategies.append(strategy)

        return strategies
```

**Deliverable**: Complete `workflow_knowledge_extractor.py` with all extractors

---

## Component 3: SolutionPatternMiner with ML Clustering

**Effort**: 4 weeks
**Priority**: P0 (advanced analytics)
**Dependencies**: Component 1, Component 2

### Tasks

#### 3.1 Implement Pattern Vectorization
**File**: `solution_pattern_miner.py` (NEW)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class SolutionPatternMiner:
    """Mine and cluster solution patterns using ML"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.clustering_model = None
        self.pattern_embeddings = {}

    async def vectorize_pattern(self, pattern: SolutionPatternArtifact) -> np.ndarray:
        """
        Convert pattern to vector representation:
        - Text description (TF-IDF)
        - Code structure features
        - Domain indicators
        - Performance metrics
        """
        # Text features
        text_features = self.vectorizer.fit_transform([pattern.description])

        # Structural features
        struct_features = self._extract_structural_features(pattern)

        # Performance features
        perf_features = np.array([pattern.success_rate, pattern.avg_execution_time])

        # Combine
        combined = np.hstack([text_features.toarray()[0], struct_features, perf_features])
        return combined

    def _extract_structural_features(self, pattern: SolutionPatternArtifact) -> np.ndarray:
        """Extract structural features from pattern content"""
        # Number of components
        # Complexity metrics
        # Dependency depth
        # etc.
        pass
```

#### 3.2 Implement Clustering Algorithm
**File**: `solution_pattern_miner.py`

```python
class SolutionPatternMiner:
    async def cluster_patterns(
        self,
        patterns: List[SolutionPatternArtifact],
        algorithm: str = "dbscan"
    ) -> Dict[str, List[SolutionPatternArtifact]]:
        """
        Cluster similar patterns together:
        - DBSCAN for density-based clustering (finds natural clusters)
        - K-Means for fixed number of clusters
        - Hierarchical for dendrogram visualization
        """
        # Vectorize all patterns
        vectors = np.array([await self.vectorize_pattern(p) for p in patterns])

        # Cluster
        if algorithm == "dbscan":
            clusters = self._dbscan_cluster(vectors)
        elif algorithm == "kmeans":
            clusters = self._kmeans_cluster(vectors)
        else:
            clusters = self._hierarchical_cluster(vectors)

        # Group patterns by cluster
        clustered_patterns = {}
        for pattern, cluster_id in zip(patterns, clusters):
            if cluster_id not in clustered_patterns:
                clustered_patterns[cluster_id] = []
            clustered_patterns[cluster_id].append(pattern)

        return clustered_patterns

    def _dbscan_cluster(self, vectors: np.ndarray) -> List[int]:
        """DBSCAN clustering - finds natural clusters"""
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors)
        return clustering.labels_.tolist()
```

#### 3.3 Implement Pattern Similarity Search
**File**: `solution_pattern_miner.py`

```python
class SolutionPatternMiner:
    async def find_similar_patterns(
        self,
        query_pattern: SolutionPatternArtifact,
        pattern_database: List[SolutionPatternArtifact],
        top_k: int = 10
    ) -> List[Tuple[SolutionPatternArtifact, float]]:
        """
        Find patterns similar to query using cosine similarity
        """
        query_vector = await self.vectorize_pattern(query_pattern)
        pattern_vectors = np.array([await self.vectorize_pattern(p) for p in pattern_database])

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_vector], pattern_vectors)[0]

        # Top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(pattern_database[i], similarities[i]) for i in top_indices]

        return results
```

**Deliverable**: Complete `solution_pattern_miner.py` with ML clustering

---

## Component 4: TeamPerformanceTracker

**Effort**: 2 weeks
**Priority**: P0 (team optimization)
**Dependencies**: Component 1

### Tasks

#### 4.1 Implement Team Performance Tracking
**File**: `team_performance_tracker.py` (NEW)

```python
@dataclass
class TeamPerformanceMetrics:
    """Metrics for team performance tracking"""
    team_id: str
    total_workflows: int
    success_rate: float
    avg_quality_score: float
    avg_execution_time: float
    avg_cost_per_workflow: float

    # By problem type
    performance_by_domain: Dict[str, float]
    performance_by_complexity: Dict[str, float]

    # Historical
    performance_trend: List[Tuple[datetime, float]]  # (timestamp, score)

class TeamPerformanceTracker:
    """Track and analyze team performance across workflows"""

    def __init__(self, persistence_layer):
        self.persistence = persistence_layer

    async def track_workflow_execution(
        self,
        team: Team,
        workflow: WorkflowState,
        results: Dict[str, Any]
    ) -> TeamPerformanceArtifact:
        """
        Track team performance for this workflow execution
        """
        metrics = await self._calculate_metrics(team, workflow, results)
        artifact = await self._create_performance_artifact(team, metrics)
        await self.persistence.save_artifact(artifact)
        return artifact

    async def _calculate_metrics(
        self,
        team: Team,
        workflow: WorkflowState,
        results: Dict[str, Any]
    ) -> TeamPerformanceMetrics:
        """
        Calculate performance metrics:
        - Success rate (solutions passed verification)
        - Quality score (average verification score)
        - Execution time
        - Cost (token usage)
        - By problem domain
        - By complexity level
        """
        pass

    async def get_team_recommendations(
        self,
        problem_domain: str,
        complexity: str
    ) -> List[str]:
        """
        Recommend best teams for given problem type
        """
        # Query historical performance
        # Return top performing teams
        pass
```

**Deliverable**: Complete `team_performance_tracker.py`

---

## Component 5: GauntletEffectivenessAnalyzer

**Effort**: 2 weeks
**Priority**: P0 (gauntlet optimization)
**Dependencies**: Component 1

### Tasks

#### 5.1 Implement Gauntlet Effectiveness Tracking
**File**: `gauntlet_effectiveness_analyzer.py` (NEW)

```python
@dataclass
class GauntletEffectivenessMetrics:
    """Metrics for gauntlet effectiveness"""
    gauntlet_id: str
    total_runs: int
    catch_rate: float  # % of issues caught
    false_positive_rate: float
    avg_execution_time: float

    # By rule
    rule_effectiveness: Dict[str, float]  # rule_id -> effectiveness_score

    # By problem type
    effectiveness_by_domain: Dict[str, float]

class GauntletEffectivenessAnalyzer:
    """Analyze gauntlet effectiveness across workflows"""

    def __init__(self, persistence_layer):
        self.persistence = persistence_layer

    async def analyze_gauntlet_run(
        self,
        gauntlet: GauntletDefinition,
        inputs: List[SolutionAttempt],
        outputs: List[CritiqueReport]
    ) -> GauntletEffectivenessArtifact:
        """
        Analyze gauntlet effectiveness for this run
        """
        metrics = await self._calculate_metrics(gauntlet, inputs, outputs)
        artifact = await self._create_effectiveness_artifact(gauntlet, metrics)
        await self.persistence.save_artifact(artifact)
        return artifact

    async def _calculate_metrics(
        self,
        gauntlet: GauntletDefinition,
        inputs: List[SolutionAttempt],
        outputs: List[CritiqueReport]
    ) -> GauntletEffectivenessMetrics:
        """
        Calculate:
        - Catch rate: % of actual issues caught
        - False positive rate: % of false positives
        - Rule effectiveness: which rules work best
        - By problem domain
        """
        pass

    async def recommend_gauntlet_rules(
        self,
        problem_domain: str,
        issue_types: List[str]
    ) -> List[GauntletRoundRule]:
        """
        Recommend most effective gauntlet rules for given problem
        """
        # Query historical effectiveness
        # Return top performing rules
        pass
```

**Deliverable**: Complete `gauntlet_effectiveness_analyzer.py`

---

## Component 6: KnowledgeGraphVisualizer

**Effort**: 2 weeks
**Priority**: P1 (nice to have)
**Dependencies**: Component 1, Component 2

### Tasks

#### 6.1 Implement Knowledge Graph Structure
**File**: `knowledge_graph_visualizer.py` (NEW)

```python
import networkx as nx
from typing import Dict, List, Tuple

class KnowledgeGraphVisualizer:
    """Visualize knowledge artifact relationships"""

    def __init__(self):
        self.graph = nx.DiGraph()

    async def build_graph(
        self,
        artifacts: List[KnowledgeArtifact]
    ) -> nx.DiGraph:
        """
        Build knowledge graph from artifacts:
        - Nodes: artifacts
        - Edges: relationships (related_artifacts, citations)
        - Attributes: artifact_type, confidence, usage_count
        """
        self.graph = nx.DiGraph()

        # Add nodes
        for artifact in artifacts:
            self.graph.add_node(
                artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                title=artifact.title,
                confidence=artifact.confidence,
                usage_count=artifact.usage_count
            )

        # Add edges
        for artifact in artifacts:
            for related_id in artifact.related_artifacts:
                self.graph.add_edge(artifact.artifact_id, related_id)

        return self.graph

    async def export_graphviz(
        self,
        output_format: str = "dot"
    ) -> str:
        """
        Export graph for visualization with Graphviz, D3.js, or Cytoscape
        """
        pass
```

#### 6.2 Implement BubbleLab UI Visualization
**File**: `ui_components.py`

```python
def render_knowledge_graph_viz():
    """Render knowledge graph in BubbleLab UI UI"""
    import BubbleLab UI as st
    import graphviz

    # Load graph
    visualizer = KnowledgeGraphVisualizer()
    graph = await visualizer.build_graph(artifacts)

    # Render with graphviz
    st.graphviz_chart(visualizer.export_graphviz())
```

**Deliverable**: Complete `knowledge_graph_visualizer.py` with UI integration

---

## Integration Tasks

### Task 7.1: Update Workflow Engine
**File**: `workflow_engine.py`

```python
class WorkflowEngine:
    async def complete_workflow(self, workflow_id: str):
        """Complete workflow and extract knowledge"""
        # ... existing completion logic ...

        # NEW: Extract knowledge artifacts
        extractor = WorkflowKnowledgeExtractor(self.knowledge_engine, self.ace_client)
        artifacts = await extractor.extract_from_workflow(workflow_state)

        # Store artifacts
        for artifact in artifacts:
            await self.persistence.save_artifact(artifact)

        # Update knowledge base
        await self.knowledge_engine.add_artifacts(artifacts)
```

### Task 7.2: Update Knowledge Base Interface
**File**: `ui_components.py`

```python
def render_knowledge_base_interface():
    """Enhanced knowledge base UI"""
    import BubbleLab UI as st

    # Artifact browser
    st.subheader("Knowledge Artifacts")
    artifact_type = st.selectbox("Filter by type", ["All", "solution_pattern", "team_performance", "gauntlet_effectiveness"])

    # Pattern similarity search
    st.subheader("Find Similar Patterns")
    query = st.text_area("Describe your problem")
    if st.button("Search"):
        similar = await pattern_miner.find_similar_patterns(query, artifacts)
        st.write("Similar patterns:", similar)

    # Knowledge graph visualization
    st.subheader("Knowledge Graph")
    render_knowledge_graph_viz()
```

---

## Testing Tasks

### Task 8.1: Unit Tests
**File**: `tests/test_stage6_components.py`

```python
def test_knowledge_artifact_validation():
    """Test KnowledgeArtifact validation"""
    pass

def test_pattern_extractor():
    """Test SolutionPatternExtractor"""
    pass

def test_pattern_clustering():
    """Test SolutionPatternMiner clustering"""
    pass

def test_team_tracking():
    """Test TeamPerformanceTracker"""
    pass

def test_gauntlet_analysis():
    """Test GauntletEffectivenessAnalyzer"""
    pass
```

### Task 8.2: Integration Tests
**File**: `tests/test_stage6_integration.py`

```python
async def test_end_to_end_knowledge_extraction():
    """Test complete knowledge extraction pipeline"""
    # Run workflow
    # Extract artifacts
    # Verify storage
    # Verify retrieval
    pass
```

---

## Documentation Tasks

### Task 9.1: API Documentation
**File**: `docs/STAGE6_API.md`

Document all new APIs:
- KnowledgeArtifact schema
- WorkflowKnowledgeExtractor methods
- SolutionPatternMiner methods
- TeamPerformanceTracker methods
- GauntletEffectivenessAnalyzer methods
- KnowledgeGraphVisualizer methods

### Task 9.2: User Guide
**File**: `docs/STAGE6_USER_GUIDE.md`

Guide for:
- Using knowledge artifacts
- Searching for similar patterns
- Recommending teams and gauntlets
- Visualizing knowledge graph

---

## Success Criteria

Phase 1 is complete when:

- [ ] All 6 components implemented and tested
- [ ] KnowledgeArtifact schema in production use
- [ ] WorkflowKnowledgeExtractor extracting artifacts from all stages
- [ ] SolutionPatternMiner clustering patterns with ML
- [ ] TeamPerformanceTracker tracking team effectiveness
- [ ] GauntletEffectivenessAnalyzer analyzing gauntlet performance
- [ ] KnowledgeGraphVisualizer displaying relationships
- [ ] All unit and integration tests passing
- [ ] API documentation complete
- [ ] User guide complete
- [ ] Stage 6 is 100% complete (up from 75%)

---

## Timeline

| Week | Component | Status |
|------|-----------|--------|
| 1-2 | KnowledgeArtifact Schema | Pending |
| 3-5 | WorkflowKnowledgeExtractor | Pending |
| 6-9 | SolutionPatternMiner (ML) | Pending |
| 10-11 | TeamPerformanceTracker | Pending |
| 12-13 | GauntletEffectivenessAnalyzer | Pending |
| 14-15 | KnowledgeGraphVisualizer | Pending |
| 15 | Integration & Testing | Pending |

---

## Dependencies

**Required Python Packages**:
```txt
scikit-learn>=1.3.0
networkx>=3.0
numpy>=1.24.0
```

**External Services**: None (all local)

---

## Notes

- This is the **HIGHEST PRIORITY** gap in the Decomposition Workflow
- No new external dependencies required
- Builds on existing ACE, Knowledge Engine, RAGbits integrations
- Enables continuous learning and improvement

---

**Task File Created**: 2025-12-31
**Source**: FRM Integration Analysis Recommendation
**Status**: READY FOR IMPLEMENTATION

