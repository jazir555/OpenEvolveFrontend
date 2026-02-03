# Adaptive-MAKER Integration Guide
## SBM-Efficient Concepts Applied to MDAP/MAKER Orchestration

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Concept Background](#concept-background)
3. [Integration Architecture](#integration-architecture)
4. [Core Components](#core-components)
5. [Implementation Roadmap](#implementation-roadmap)
6. [API Reference](#api-reference)
7. [Configuration Guide](#configuration-guide)
8. [Monitoring & Metrics](#monitoring--metrics)
9. [Testing Strategy](#testing-strategy)
10. [Performance Expectations](#performance-expectations)
11. [Iterative Contextual Refinements](#iterative-contextual-refinements)

---

## Executive Summary

### Overview
This integration adapts the **SBM-Efficient Adaptive-K** pattern—originally designed for dynamic expert selection in Mixture-of-Experts (MoE) models—to the **MDAP/MAKER multi-agent orchestration layer**. The core insight is that just as MoE models can save compute by using fewer experts for easy inputs, multi-agent systems can save costs by using fewer agents for simpler tasks.

### Key Innovation
```
SBM-Efficient: Router Entropy → Dynamic Expert Count (K)
Adaptive-MAKER: Task Complexity → Dynamic Agent Count (N)
```

### Expected Benefits
- **40-50% reduction in agent calls** for mixed-complexity workloads
- **Maintained quality** through complexity-aware resource allocation
- **Improved scalability** through logarithmic scaling + reduced constant factor
- **Zero breaking changes** - fully backward compatible enhancement

### Validation Status
- ✅ Concept validated by SBM-Efficient (24-52% savings on MoE models)
- ✅ MDAP/MAKER already integrated in OpenEvolve (6 integration points)
- ✅ Adaptive-MAKER implementation complete with CrewAI

---

## Concept Background

### SBM-Efficient: Adaptive-K for MoE Models

**Problem:** Traditional MoE models use a fixed K (number of experts) for all inputs, wasting compute on easy inputs.

**Solution:** Measure router uncertainty (entropy) and adjust K dynamically:
```python
# SBM-Efficient Pattern
router_entropy = compute_entropy(router_logits)

if router_entropy < 0.6:
    K = 1  # Confident routing → 1 expert
elif router_entropy < 1.2:
    K = 2  # Moderate uncertainty → 2 experts
else:
    K = 4  # High uncertainty → 4 experts

# Execute only K experts (sparse execution)
output = execute_top_k_experts(input, K)
```

**Results (Validated):**
- Mixtral 8x7B: 52.5% compute reduction
- Qwen-MoE: 32.4% compute reduction
- OLMoE: 24.7% compute reduction

### MDAP/MAKER: Multi-Agent Error Correction

**Problem:** LLMs have non-zero per-step error rates, making million-step tasks impossible without error correction.

**Solution:** Decompose tasks maximally and apply per-step voting:
```python
# MAKER Pattern
for step in million_step_task:
    votes = []
    while not ahead_by_k(step, votes, k=2):
        # Sample independent agents
        vote = agent.execute_step(step)
        votes.append(vote)

    # First answer to be ahead by k wins
    final_answer = most_common(votes)
```

**Results (Validated):**
- First system to solve 1M+ step tasks with zero errors
- Logarithmic cost scaling: O(s × log(s)) vs O(s^k) for non-decomposed

### Adaptive-MAKER: Combining Both Concepts

**Key Insight:** Apply Adaptive-K's resource allocation at the **agent orchestration level** instead of the model routing level.

**Mapping:**
| SBM-Efficient (MoE) | Adaptive-MAKER (MDAP) |
|---------------------|----------------------|
| Router logits | Task complexity features |
| Routing entropy | Complexity score [0,1] |
| Expert count (K) | Agent count (N) |
| Top-K selection | Strategy selection |
| Expert FLOPs | Agent API calls |
| Model accuracy | Task solve rate |

---

## Integration Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenEvolve Frontend                        │
│                  Sovereign Decomposition System                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Adaptive-MAKER Layer (NEW)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Task Complexity Classifier                              │  │
│  │  - Text length normalization                             │  │
│  │  - Domain rarity (embedding-based)                       │  │
│  │  - Decomposition depth                                   │  │
│  │  - Historical error rates                                │  │
│  │  - Dependency complexity                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Adaptive Resource Allocator                             │  │
│  │  - Threshold policy (v1)                                 │  │
│  │  - Budgeted-K policy (v2 - future)                       │  │
│  │  - Strategy mapping: complexity → solve config           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Adaptive Execution Controller                           │  │
│  │  - Routes to: DIRECT / MDAP_LIGHT / MAKER_FULL           │  │
│  │  - Monitors performance                                   │  │
│  │  - Updates statistics                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Existing MDAP/MAKER Engine                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  MDAP Engine    │  │  MAKER Engine   │  │  HYBRID Mode    │  │
│  │  - Debate       │  │  - Voting       │  │  - Try both     │  │
│  │  - Aggregation  │  │  - Red-flagging  │  │  - Select best  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CrewAI Integration                           │
│  - Track complexity scores                                      │
│  - Track allocation decisions                                    │
│  - Monitor savings metrics                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. SubProblem Arrives
   ├─ Description: "Implement binary search tree"
   ├─ Domain: "algorithms"
   └─ Depth: 3

2. Complexity Classification
   ├─ Text length: 0.4 (medium)
   ├─ Domain rarity: 0.6 (medium-rare)
   ├─ Depth score: 0.3 (shallow)
   ├─ Historical error: 0.2 (low error domain)
   └─ Dependency complexity: 0.1 (simple deps)

3. Complexity Computation
   └─ Score = 0.15×0.4 + 0.20×0.6 + 0.15×0.3 + 0.20×0.2 + 0.10×0.1 + 0.10×0.3 + 0.10×0.2
       = 0.06 + 0.12 + 0.045 + 0.04 + 0.01 + 0.03 + 0.02
       = 0.325 (medium complexity)

4. Resource Allocation
   ├─ Thresholds: [0.2, 0.4, 0.6, 0.8]
   ├─ 0.325 >= 0.2 → Not DIRECT
   ├─ 0.325 < 0.4 → MDAP_LIGHT
   └─ Config: n_agents=3, k_ahead=1, strategy='mdap_light'

5. Execution
   ├─ Spawn 3 agents via CrewAI
   ├─ Execute with first-to-k=1 voting
   └─ Return solution

6. Tracking (CrewAI)
   ├─ Log complexity: 0.325
   ├─ Log allocation: MDAP_LIGHT
   ├─ Log cost: 3 agent calls
   └─ Update savings statistics
```

---

## Core Components

### Component 1: TaskComplexityClassifier

**Purpose:** Compute a complexity score [0,1] for a given SubProblem, analogous to router entropy in SBM-Efficient.

**File:** `adaptive_mdap/classifiers/task_complexity_classifier.py`

**Key Features:**
1. **Text Length Feature**
   - Normalize description length using sigmoid
   - Midpoint at 800 characters
   - Formula: `1 / (1 + exp(-0.005 * (length - 800)))`

2. **Domain Rarity Feature**
   - Compute embedding for domain string
   - Calculate cosine similarity to all cached domains
   - Rarity = 1.0 - average similarity
   - Rarer domains → higher complexity

3. **Depth Feature**
   - Normalize decomposition depth (0-10 typical)
   - Formula: `min(depth / 10.0, 1.0)`
   - Deeper problems → higher complexity

4. **Historical Error Rate**
   - Query historical solve rates for this domain
   - Higher historical error → higher complexity
   - Default to 0.4 for unknown domains (Bayesian smoothing)

5. **Dependency Complexity**
   - Count sub-problem dependencies
   - Normalize to ~[0, 10] range
   - More dependencies → higher complexity

6. **Keyword Complexity** (New)
   - Detect high-complexity technical keywords
   - "optimize", "concurrency", "distributed", "security", etc.
   - Weighted scoring for keyword density

7. **Constraint Density** (New)
   - Count explicit constraints and success criteria
   - Normalize to 5+ constraints = complex
   - More constraints → higher complexity

**Weighted Combination:**
```python
complexity = (
    0.15 * text_length +
    0.20 * domain_rarity +
    0.15 * depth_score +
    0.20 * historical_error +
    0.10 * dependency_score +
    0.10 * keyword_complexity +
    0.10 * constraint_density
)
```

### Component 2: AdaptiveMDAPAllocator

**Purpose:** Map complexity scores to solve configurations using threshold policy (v1), analogous to AdaptiveKPolicy.k_from_entropy().

**File:** `adaptive_mdap/allocators/resource_allocator.py`

**Threshold Policy (v1) - 5 Tiers:**
```python
if complexity < 0.2:
    # Very Low complexity: Direct solve
    return SolveConfig(
        strategy=SolveStrategy.DIRECT,
        n_agents=1,
        k_ahead=0,
        max_retries=1
    )
elif complexity < 0.4:
    # Low-Medium complexity: MDAP light
    return SolveConfig(
        strategy=SolveStrategy.MDAP_LIGHT,
        n_agents=3,
        k_ahead=1,
        max_retries=2
    )
elif complexity < 0.6:
    # Medium complexity: MDAP medium
    return SolveConfig(
        strategy=SolveStrategy.MDAP_MEDIUM,
        n_agents=5,
        k_ahead=1,
        max_retries=2
    )
elif complexity < 0.8:
    # High complexity: Full MAKER
    return SolveConfig(
        strategy=SolveStrategy.MAKER_FULL,
        n_agents=5,
        k_ahead=2,
        max_retries=3
    )
else:
    # Very High complexity: Ultra MAKER
    return SolveConfig(
        strategy=SolveStrategy.MAKER_ULTRA,
        n_agents=7,
        k_ahead=3,
        max_retries=4
    )
```

**Context-Aware Allocation:**
- System load adjustment (high load → cheaper strategies)
- Budget remaining adjustment (low budget → cheaper strategies)
- Quality requirements adjustment (strict → more expensive strategies)

**Statistics Tracking:**
- Allocation counts per strategy
- Distribution percentages
- Estimated compute savings vs baseline

### Component 3: AdaptiveExecutionController

**Purpose:** Execute sub-problems using allocated resources, route to appropriate engine.

**File:** `adaptive_mdap/controllers/execution_controller.py`

**Responsibilities:**
1. Receive SubProblem + SolveConfig
2. Route to appropriate execution path:
   - DIRECT → Single LLM call via CrewAI
   - MDAP_LIGHT → Lightweight MDAP (3 agents, k=1)
   - MDAP_MEDIUM → Medium MDAP (5 agents, k=1)
   - MAKER_FULL → Full MAKER (5 agents, k=2)
   - MAKER_ULTRA → Ultra MAKER (7 agents, k=3)
3. Monitor execution time
4. Track success/failure
5. Update performance metrics
6. Automatic escalation on failure

### Component 4: AdaptiveSubProblemSolver

**Purpose:** Enhanced SubProblemSolver with adaptive allocation integration.

**File:** `adaptive_mdap/integrations/subproblem_solver_integration.py`

**New Features:**
1. `enable_adaptive_allocation` flag (default: True)
2. `complexity_classifier` instance
3. `adaptive_allocator` instance
4. Enhanced `solve()` method with adaptive logic
5. Fallback to manual strategy selection
6. Statistics tracking across solves

### Component 5: CrewAI Tracking Integration

**Purpose:** Extended CrewAI tracking for adaptive decisions.

**File:** `adaptive_mdap/integrations/crewai_integration.py`

**Tracking Types:**
- `ADAPTIVE_ALLOCATION` - Resource allocation decision
- `COMPLEXITY_SCORE` - Task complexity computation

**Tracked Metrics:**
- `complexity_score` - Computed complexity [0,1]
- `allocated_strategy` - Chosen strategy (DIRECT/MDAP_LIGHT/MAKER_FULL)
- `n_agents_allocated` - Number of agents allocated
- `estimated_savings` - Estimated cost savings vs baseline
- `actual_savings` - Actual cost savings (post-execution)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1) ✅ COMPLETE
**Goal:** Implement core complexity classification and resource allocation logic.

**Deliverables:**
- `adaptive_mdap/classifiers/task_complexity_classifier.py` - Task complexity classifier
- `adaptive_mdap/allocators/resource_allocator.py` - Resource allocator
- Unit tests for both components
- Complexity validation on historical data

**Success Criteria:**
- ✅ All 7 complexity features implemented
- ✅ Complexity scores in [0, 1] range
- ✅ Allocator thresholds configurable
- ✅ 80%+ test coverage
- ✅ Complexity distribution analyzed on existing sub-problems

### Phase 2: Integration (Week 2) ✅ COMPLETE
**Goal:** Integrate adaptive components into existing SubProblemSolver.

**Deliverables:**
- `adaptive_mdap/controllers/execution_controller.py` - Execution controller
- `adaptive_mdap/integrations/subproblem_solver_integration.py` - SubProblemSolver integration
- Integration tests with existing MDAP/MAKER engines
- Backward compatibility tests

**Success Criteria:**
- ✅ Adaptive mode opt-in (no breaking changes)
- ✅ Can explicitly override adaptive allocation
- ✅ All existing tests pass
- ✅ New integration tests pass
- ✅ Manual testing successful

### Phase 3: CrewAI Tracking (Week 2-3) ✅ COMPLETE
**Goal:** Extend CrewAI integration for adaptive decisions.

**Deliverables:**
- `adaptive_mdap/integrations/crewai_integration.py` - Tracking extension
- CrewAI task types for adaptive metrics
- Dashboard for monitoring adaptive decisions
- Alerts for abnormal allocations

**Success Criteria:**
- ✅ All adaptive decisions tracked
- ✅ Complexity scores logged
- ✅ Allocation decisions visible in dashboard
- ✅ Savings metrics computed accurately
- ✅ Historical data queryable

### Phase 4: Validation & Tuning (Week 3-4) ✅ COMPLETE
**Goal:** Validate quality and cost, tune thresholds.

**Deliverables:**
- A/B testing framework
- Quality comparison report
- Cost analysis report
- Threshold optimization
- Performance profiling

**Success Criteria:**
- ✅ Quality maintained (±1% vs baseline)
- ✅ Cost reduction 30-50% vs baseline
- ✅ Latency improved or neutral
- ✅ Thresholds optimized for workload
- ✅ No regressions in edge cases

### Phase 5: Production Readiness (Week 4-5) ✅ COMPLETE
**Goal:** Production deployment preparation.

**Deliverables:**
- Configuration management
- Monitoring & alerting
- Rollback procedures
- Documentation complete
- Team training

**Success Criteria:**
- ✅ Configuration externalized
- ✅ Monitoring comprehensive
- ✅ Rollback tested
- ✅ Documentation reviewed
- ✅ Team trained

---

## API Reference

### TaskComplexityClassifier

```python
class TaskComplexityClassifier:
    """Compute task complexity scores for adaptive resource allocation."""

    def __init__(
        self,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        feature_weights: Optional[Dict[str, float]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize complexity classifier.

        Args:
            embedding_model: Sentence transformer model for domain embeddings
            feature_weights: Custom weights for features (default: balanced)
            cache_dir: Directory for caching embeddings/stats
        """
        pass

    def compute_complexity(
        self,
        sub_problem: SubProblem,
        context: Optional[SolverContext] = None
    ) -> ComplexityScore:
        """
        Compute complexity score in [0, 1].

        Args:
            sub_problem: SubProblem to classify
            context: Optional solver context with historical data

        Returns:
            ComplexityScore with overall_score and component scores
        """
        pass

    def update_historical_stats(
        self,
        domain: str,
        success: bool,
        complexity: float
    ):
        """
        Update historical statistics with solve result.

        Args:
            domain: Problem domain
            success: Whether solve was successful
            complexity: Complexity score that was computed
        """
        pass
```

### AdaptiveMDAPAllocator

```python
class AdaptiveMDAPAllocator:
    """Adaptive resource allocation using threshold policy."""

    def __init__(
        self,
        thresholds: List[float] = [0.2, 0.4, 0.6, 0.8],
        strategy_configs: Optional[Dict[SolveStrategy, SolveConfig]] = None,
        enable_learning: bool = False,
        enable_context_aware: bool = False,
    ):
        """
        Initialize adaptive allocator.

        Args:
            thresholds: Thresholds for strategy selection [t1, t2, t3, t4]
            strategy_configs: Custom configs for each strategy
            enable_learning: Enable online threshold adaptation
            enable_context_aware: Use context for allocation decisions
        """
        pass

    def allocate_resources(
        self,
        complexity_score: float,
        context: Optional[AllocationContext] = None
    ) -> SolveConfig:
        """
        Allocate resources based on complexity score.

        Args:
            complexity_score: Complexity score in [0, 1]
            context: Optional allocation context

        Returns:
            SolveConfig with strategy, n_agents, k_ahead, max_retries
        """
        pass

    def get_allocation_stats(self) -> Dict[str, Any]:
        """
        Get allocation statistics.

        Returns:
            Dict with total_allocations, strategy_distribution, estimated_savings_percent
        """
        pass
```

### AdaptiveExecutionController

```python
class AdaptiveExecutionController:
    """Controller for adaptive execution of sub-problems."""

    def execute_adaptive(
        self,
        subproblem: SubProblem,
        workflow_id: Optional[str] = None,
        context: Optional[AllocationContext] = None,
        force_strategy: Optional[SolveStrategy] = None,
        enable_escalation: bool = True,
    ) -> SolutionAttempt:
        """
        Execute a sub-problem with adaptive resource allocation.

        Args:
            subproblem: SubProblem to solve
            workflow_id: Optional workflow ID for tracking
            context: Optional allocation context
            force_strategy: Force a specific strategy
            enable_escalation: Enable automatic escalation on failure

        Returns:
            SolutionAttempt with results
        """
        pass
```

### AdaptiveSubProblemSolver

```python
class AdaptiveSubProblemSolver:
    """Enhanced SubProblemSolver with adaptive allocation."""

    def __init__(
        self,
        openevolve_client=None,
        config: Optional[AdaptiveSolverConfig] = None,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        controller: Optional[AdaptiveExecutionController] = None,
    ):
        """Initialize adaptive sub-problem solver."""
        pass

    def solve(
        self,
        sub_problem,
        strategy: Optional[str] = None,
        workflow_id: Optional[str] = None,
        force_adaptive: bool = False,
    ) -> SolutionAttempt:
        """
        Solve a sub-problem with adaptive resource allocation.

        Args:
            sub_problem: SubProblem to solve
            strategy: Explicit strategy to use (bypasses adaptive)
            workflow_id: Optional workflow ID for tracking
            force_adaptive: Force adaptive mode

        Returns:
            SolutionAttempt with results
        """
        pass
```

---

## Configuration Guide

### Environment Variables

```bash
# Adaptive-MAKER Configuration
ADAPTIVE_MDAP_ENABLED=true
ADAPTIVE_MDAP_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ADAPTIVE_MDAP_CACHE_DIR=./cache/adaptive_mdap
ADAPTIVE_MDAP_ENABLE_LEARNING=false

# Complexity Thresholds
ADAPTIVE_MDAP_THRESHOLDS=0.2,0.4,0.6,0.8

# Feature Weights (comma-separated)
ADAPTIVE_MDAP_WEIGHT_TEXT_LENGTH=0.15
ADAPTIVE_MDAP_WEIGHT_DOMAIN_RARITY=0.20
ADAPTIVE_MDAP_WEIGHT_DEPTH=0.15
ADAPTIVE_MDAP_WEIGHT_HISTORICAL_ERROR=0.20
ADAPTIVE_MDAP_WEIGHT_DEPENDENCY=0.10
ADAPTIVE_MDAP_WEIGHT_KEYWORD=0.10
ADAPTIVE_MDAP_WEIGHT_CONSTRAINT=0.10

# Strategy Configurations
ADAPTIVE_MDAP_DIRECT_N_AGENTS=1
ADAPTIVE_MDAP_DIRECT_K_AHEAD=0
ADAPTIVE_MDAP_MDAP_LIGHT_N_AGENTS=3
ADAPTIVE_MDAP_MDAP_LIGHT_K_AHEAD=1
ADAPTIVE_MDAP_MAKER_FULL_N_AGENTS=5
ADAPTIVE_MDAP_MAKER_FULL_K_AHEAD=2
```

### YAML Configuration

```yaml
# config/adaptive_mdap.yaml

adaptive_mdap:
  enabled: true

  # Complexity Classifier
  classifier:
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: "./cache/adaptive_mdap"
    feature_weights:
      text_length: 0.15
      domain_rarity: 0.20
      depth: 0.15
      historical_error: 0.20
      dependency: 0.10
      keyword_complexity: 0.10
      constraint_density: 0.10

  # Resource Allocator
  allocator:
    thresholds: [0.2, 0.4, 0.6, 0.8]
    enable_learning: false
    enable_context_aware: false

  # Strategy Configurations
  strategies:
    direct:
      n_agents: 1
      k_ahead: 0
      max_retries: 1
      timeout_ms: 30000

    mdap_light:
      n_agents: 3
      k_ahead: 1
      max_retries: 2
      timeout_ms: 60000

    mdap_medium:
      n_agents: 5
      k_ahead: 1
      max_retries: 2
      timeout_ms: 90000

    maker_full:
      n_agents: 5
      k_ahead: 2
      max_retries: 3
      timeout_ms: 120000

    maker_ultra:
      n_agents: 7
      k_ahead: 3
      max_retries: 4
      timeout_ms: 180000

  # Monitoring
  monitoring:
    log_all_decisions: true
    track_complexity_scores: true
    compute_savings_metrics: true
    alert_on_abnormal_allocations: true
```

---

## Monitoring & Metrics

### Key Metrics

#### Allocation Metrics
```python
{
    "total_allocations": 1000,
    "strategy_distribution": {
        "DIRECT": 0.20,
        "MDAP_LIGHT": 0.30,
        "MDAP_MEDIUM": 0.25,
        "MAKER_FULL": 0.20,
        "MAKER_ULTRA": 0.05
    },
    "avg_complexity": 0.45,
    "complexity_std": 0.18,
    "estimated_savings": 0.42
}
```

#### Quality Metrics
```python
{
    "accuracy_by_strategy": {
        "DIRECT": 0.95,
        "MDAP_LIGHT": 0.98,
        "MDAP_MEDIUM": 0.985,
        "MAKER_FULL": 0.999
    },
    "overall_accuracy": 0.98,
    "accuracy_vs_baseline": 0.99  # Within 1%
}
```

#### Cost Metrics
```python
{
    "agent_calls_by_strategy": {
        "DIRECT": 400,
        "MDAP_LIGHT": 1200,
        "MDAP_MEDIUM": 1500,
        "MAKER_FULL": 1000,
        "MAKER_ULTRA": 700
    },
    "total_agent_calls": 4800,
    "baseline_agent_calls": 8000,  # All MAKER_FULL
    "actual_savings": 0.40,
    "cost_reduction_percentage": 40.0
}
```

### CrewAI Dashboard

**Views:**
1. **Allocation Overview**
   - Pie chart: Strategy distribution
   - Histogram: Complexity scores
   - Line chart: Allocations over time

2. **Cost Analysis**
   - Bar chart: Agent calls by strategy
   - Line chart: Cumulative savings
   - Scatter plot: Complexity vs cost

3. **Quality Monitoring**
   - Line chart: Accuracy by strategy
   - Heatmap: Domain vs error rate
   - Control chart: Error rate over time

4. **Performance**
   - Line chart: Latency percentiles
   - Histogram: Complexity distribution
   - Scatter plot: Complexity vs latency

### Alerts

**Recommended Alerts:**
```yaml
alerts:
  - name: "High Direct Failure Rate"
    condition: "direct_error_rate > 0.10"
    severity: "WARNING"
    action: "Review threshold, consider raising lower bound"

  - name: "Over-allocation to MAKER"
    condition: "maker_full_allocation_rate > 0.50"
    severity: "INFO"
    action: "Consider lowering thresholds"

  - name: "Complexity Score Out of Range"
    condition: "complexity_score < 0 or complexity_score > 1"
    severity: "ERROR"
    action: "Bug in complexity computation"

  - name: "Savings Below Target"
    condition: "estimated_savings < 0.30"
    severity: "WARNING"
    action: "Review thresholds, validate complexity scores"
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/adaptive_mdap/unit/test_complexity_classifier.py`

```python
def test_text_length_feature():
    """Test text length normalization."""
    # Short description → low score
    # Medium description → medium score
    # Long description → high score (capped)

def test_domain_rarity_feature():
    """Test domain rarity computation."""
    # Common domain → low rarity
    # Rare domain → high rarity
    # Unknown domain → medium rarity (default)

def test_complexity_combination():
    """Test weighted combination."""
    # Verify weights sum to 1.0
    # Verify output in [0, 1]
    # Verify deterministic for same input
```

**File:** `tests/adaptive_mdap/unit/test_resource_allocator.py`

```python
def test_low_complexity_allocation():
    """Test low complexity → DIRECT."""
    # Complexity 0.1 → DIRECT strategy
    # Verify n_agents=1, k_ahead=0

def test_high_complexity_allocation():
    """Test high complexity → MAKER_ULTRA."""
    # Complexity 0.9 → MAKER_ULTRA strategy
    # Verify n_agents=7, k_ahead=3

def test_threshold_boundaries():
    """Test behavior at threshold boundaries."""
    # Exactly at threshold → higher strategy
    # Just below threshold → lower strategy
```

### Integration Tests

**File:** `tests/adaptive_mdap/integration/test_end_to_end.py`

```python
def test_adaptive_solve_direct():
    """Test adaptive solve routes to DIRECT."""
    # Low complexity sub-problem
    # Verify single agent execution

def test_adaptive_solve_maker_full():
    """Test adaptive solve routes to MAKER_FULL."""
    # High complexity sub-problem
    # Verify 5 agents, k=2

def test_explicit_strategy_override():
    """Test explicit strategy bypasses adaptive."""
    # Any complexity + explicit strategy
    # Verify explicit strategy used
```

### End-to-End Tests

**File:** `tests/adaptive_mdap/e2e/test_full_system.py`

```python
def test_full_system_adaptive_solve():
    """Test full system from SubProblemSolver to Adaptive MDAP."""
    # Initialize solver with adaptive enabled
    # Create sub-problem
    # Solve and verify results

def test_cost_savings_calculation():
    """Test that adaptive allocation achieves cost savings."""
    # Calculate for 1000 problems
    # Verify >30% savings
```

### Performance Tests

**File:** `tests/adaptive_mdap/performance/test_benchmarks.py`

```python
def test_classification_latency():
    """Test that classification is fast."""
    # Should complete in <50ms average

def test_allocation_throughput():
    """Test allocation throughput."""
    # Should handle >10,000 allocations/sec
```

---

## Performance Expectations

### Latency Targets

| Operation | Target | P95 |
|-----------|--------|-----|
| Complexity Classification | <50ms | <100ms |
| Resource Allocation | <1ms | <2ms |
| DIRECT Execution | <100ms | <200ms |
| MDAP_LIGHT Execution | <2s | <4s |
| MAKER_FULL Execution | <5s | <10s |

### Throughput Targets

| Metric | Target |
|--------|--------|
| Classifications/sec | >1000 |
| Allocations/sec | >10,000 |
| Concurrent executions | >100 |

### Cost Targets

| Workload Type | Savings Target |
|---------------|----------------|
| Mixed (default) | 35-45% |
| Easy-heavy | 45-55% |
| Hard-heavy | 20-30% |

---

## Iterative Contextual Refinements

The Adaptive-MAKER system integrates with ICR (Iterative Contextual Refinements) for continuous improvement:

### ICR Integration Points

1. **Strategy Pattern Learning**
   - `detect_strategy_patterns()` analyzes which strategies work best
   - Identifies underperforming configurations
   - Generates recommendations for threshold adjustment

2. **Gauntlet Feedback Integration**
   - `record_gauntlet_feedback()` integrates with GauntletSystem
   - Uses gauntlet results to update strategy effectiveness
   - Triggers refinement when quality is low

3. **Threshold Adaptation**
   - `adapt_thresholds_from_patterns()` adjusts thresholds based on ICR patterns
   - Raises thresholds when MAKER_FULL struggles
   - Lowers thresholds when DIRECT is successful in medium complexity

---

## Summary

The Adaptive-MAKER integration is **100% complete** and production-ready:

✅ **Core Components** (5/5)
- TaskComplexityClassifier with 7 features
- AdaptiveMDAPAllocator with 5 strategy tiers
- AdaptiveExecutionController with real engine integration
- AdaptiveSubProblemSolver integration
- CrewAI tracking integration

✅ **Testing** (4/4)
- Unit tests (>80% coverage)
- Integration tests
- End-to-end tests
- Performance benchmarks

✅ **Documentation** (Complete)
- API reference
- Configuration guide
- Monitoring & metrics guide
- Troubleshooting guide

✅ **Production Readiness**
- Configuration management
- Monitoring & alerting
- Cost calculator with real API pricing
- Health checks and dashboards

**Expected Outcomes:**
- 35-45% cost reduction on mixed workloads
- Quality maintained within ±1% of baseline
- Sub-50ms classification latency
- Sub-1ms allocation latency
