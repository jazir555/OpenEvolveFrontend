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
- 🔄 Adaptive-MAKER implementation pending

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
│                    Hephaestus Integration                       │
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
   └─ Score = 0.2×0.4 + 0.3×0.6 + 0.2×0.3 + 0.2×0.2 + 0.1×0.1
       = 0.08 + 0.18 + 0.06 + 0.04 + 0.01
       = 0.37 (medium complexity)

4. Resource Allocation
   ├─ Thresholds: [0.3, 0.7]
   ├─ 0.37 ≥ 0.3 → Not DIRECT
   ├─ 0.37 < 0.7 → MDAP_LIGHT
   └─ Config: n_agents=3, k_ahead=1, strategy='mdap_light'

5. Execution
   ├─ Spawn 3 agents
   ├─ Execute with first-to-k=1 voting
   └─ Return solution

6. Tracking (Hephaestus)
   ├─ Log complexity: 0.37
   ├─ Log allocation: MDAP_LIGHT
   ├─ Log cost: 3 agent calls
   └─ Update savings statistics
```

---

## Core Components

### Component 1: TaskComplexityClassifier

**Purpose:** Compute a complexity score [0,1] for a given SubProblem, analogous to router entropy in SBM-Efficient.

**File:** `Frontend/adaptive_mdap_complexity.py`

**Key Features:**
1. **Text Length Feature**
   - Normalize description length
   - Cap at 5000 chars (very long problems)
   - Formula: `min(len(description) / 5000.0, 1.0)`

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
   - Default to 0.5 for unknown domains

5. **Dependency Complexity**
   - Count sub-problem dependencies
   - Normalize to ~[0, 10] range
   - More dependencies → higher complexity

**Weighted Combination:**
```python
complexity = (
    0.20 * text_length +
    0.30 * domain_rarity +
    0.20 * depth_score +
    0.20 * historical_error +
    0.10 * dependency_score
)
```

### Component 2: AdaptiveMDAPAllocator

**Purpose:** Map complexity scores to solve configurations using threshold policy (v1), analogous to AdaptiveKPolicy.k_from_entropy().

**File:** `Frontend/adaptive_mdap_allocator.py`

**Threshold Policy (v1):**
```python
if complexity < 0.3:
    # Low complexity: Direct solve
    return SolveConfig(
        strategy=SolveStrategy.DIRECT,
        n_agents=1,
        k_ahead=0,
        max_retries=1
    )
elif complexity < 0.7:
    # Medium complexity: MDAP light
    return SolveConfig(
        strategy=SolveStrategy.MDAP_LIGHT,
        n_agents=3,
        k_ahead=1,
        max_retries=2
    )
else:
    # High complexity: Full MAKER
    return SolveConfig(
        strategy=SolveStrategy.MAKER_FULL,
        n_agents=5,
        k_ahead=2,
        max_retries=3
    )
```

**Statistics Tracking:**
- Allocation counts per strategy
- Distribution percentages
- Estimated compute savings vs baseline

### Component 3: AdaptiveExecutionController

**Purpose:** Execute sub-problems using allocated resources, route to appropriate engine.

**File:** `Frontend/adaptive_mdap_controller.py`

**Responsibilities:**
1. Receive SubProblem + SolveConfig
2. Route to appropriate execution path:
   - DIRECT → Standard LLM call
   - MDAP_LIGHT → Lightweight MDAP (3 agents, k=1)
   - MAKER_FULL → Full MAKER (5 agents, k=2)
3. Monitor execution time
4. Track success/failure
5. Update performance metrics

### Component 4: AdaptiveSubProblemSolver

**Purpose:** Enhanced SubProblemSolver with adaptive allocation integration.

**File:** `Frontend/sub_problem_solver.py` (extension)

**New Features:**
1. `enable_adaptive_allocation` flag (default: True)
2. `complexity_classifier` instance
3. `adaptive_allocator` instance
4. Enhanced `solve()` method with adaptive logic
5. Fallback to manual strategy selection

### Component 5: AdaptiveHephaestusIntegration

**Purpose:** Extended Hephaestus tracking for adaptive decisions.

**File:** `Frontend/adaptive_mdap_hephaestus.py`

**New Ticket Types:**
- `ADAPTIVE_ALLOCATION` - Resource allocation decision
- `COMPLEXITY_SCORE` - Task complexity computation

**New Metrics:**
- `complexity_score` - Computed complexity [0,1]
- `allocated_strategy` - Chosen strategy (DIRECT/MDAP_LIGHT/MAKER_FULL)
- `n_agents_allocated` - Number of agents allocated
- `estimated_savings` - Estimated cost savings vs baseline
- `actual_savings` - Actual cost savings (post-execution)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Goal:** Implement core complexity classification and resource allocation logic.

**Deliverables:**
- `adaptive_mdap_complexity.py` - Task complexity classifier
- `adaptive_mdap_allocator.py` - Resource allocator
- Unit tests for both components
- Complexity validation on historical data

**Success Criteria:**
- ✅ All 5 complexity features implemented
- ✅ Complexity scores in [0, 1] range
- ✅ Allocator thresholds configurable
- ✅ 80%+ test coverage
- ✅ Complexity distribution analyzed on existing sub-problems

### Phase 2: Integration (Week 2)
**Goal:** Integrate adaptive components into existing SubProblemSolver.

**Deliverables:**
- `adaptive_mdap_controller.py` - Execution controller
- Enhanced `sub_problem_solver.py` with adaptive mode
- Integration tests with existing MDAP/MAKER engines
- Backward compatibility tests

**Success Criteria:**
- ✅ Adaptive mode opt-in (no breaking changes)
- ✅ Can explicitly override adaptive allocation
- ✅ All existing tests pass
- ✅ New integration tests pass
- ✅ Manual testing successful

### Phase 3: Hephaestus Tracking (Week 2-3)
**Goal:** Extend Hephaestus integration for adaptive decisions.

**Deliverables:**
- `adaptive_mdap_hephaestus.py` - Tracking extension
- Hephaestus ticket types for adaptive metrics
- Dashboard for monitoring adaptive decisions
- Alerts for abnormal allocations

**Success Criteria:**
- ✅ All adaptive decisions tracked
- ✅ Complexity scores logged
- ✅ Allocation decisions visible in dashboard
- ✅ Savings metrics computed accurately
- ✅ Historical data queryable

### Phase 4: Validation & Tuning (Week 3-4)
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

### Phase 5: Production Readiness (Week 4-5)
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

### Phase 6: Future Enhancements (Post-Launch)
**Potential v2 Features:**
- Budgeted-K policy (compute budget constraints)
- Online threshold adaptation
- Per-domain complexity models
- Ensemble of classifiers
- Multi-arm bandit for strategy selection

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
    ) -> float:
        """
        Compute complexity score in [0, 1].

        Args:
            sub_problem: SubProblem to classify
            context: Optional solver context with historical data

        Returns:
            Complexity score in [0, 1] where:
            - 0.0 = trivial (direct solve recommended)
            - 0.5 = moderate (light voting recommended)
            - 1.0 = complex (full MAKER recommended)
        """
        pass

    def compute_feature_vector(
        self,
        sub_problem: SubProblem
    ) -> Dict[str, float]:
        """
        Compute individual feature scores.

        Returns:
            Dict with keys:
            - text_length: Normalized text length [0, 1]
            - domain_rarity: Domain rarity score [0, 1]
            - depth_score: Decomposition depth [0, 1]
            - historical_error: Historical error rate [0, 1]
            - dependency_score: Dependency complexity [0, 1]
        """
        pass

    def get_domain_rarity(
        self,
        domain: str
    ) -> float:
        """
        Get domain rarity score.

        Args:
            domain: Domain string (e.g., "algorithms", "ui-design")

        Returns:
            Rarity score in [0, 1] where 1.0 = very rare (high complexity)
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
        complexity_thresholds: List[float] = [0.3, 0.7],
        strategy_configs: Optional[Dict[SolveStrategy, SolveConfig]] = None,
        enable_learning: bool = False
    ):
        """
        Initialize adaptive allocator.

        Args:
            complexity_thresholds: Thresholds for strategy selection
                - len(thresholds) = 2 for 3 strategies
                - Example: [0.3, 0.7] means:
                    - < 0.3 → DIRECT
                    - 0.3-0.7 → MDAP_LIGHT
                    - ≥ 0.7 → MAKER_FULL
            strategy_configs: Custom configs for each strategy
            enable_learning: Enable online threshold adaptation (future)
        """
        pass

    def allocate_resources(
        self,
        complexity: float,
        context: Optional[AllocationContext] = None
    ) -> SolveConfig:
        """
        Allocate resources based on complexity score.

        Args:
            complexity: Complexity score in [0, 1]
            context: Optional allocation context

        Returns:
            SolveConfig with strategy, n_agents, k_ahead, max_retries
        """
        pass

    def get_allocation_stats(
        self
    ) -> Dict[str, Any]:
        """
        Get allocation statistics.

        Returns:
            Dict with:
            - total_allocations: Total number of allocations
            - strategy_distribution: Dict mapping strategy → percentage
            - avg_complexity_savings: Estimated cost savings
            - strategy_counts: Raw counts per strategy
        """
        pass

    def reset_stats(self):
        """Reset allocation statistics."""
        pass

    def update_thresholds(
        self,
        new_thresholds: List[float]
    ):
        """
        Update complexity thresholds.

        Args:
            new_thresholds: New threshold values
        """
        pass
```

### AdaptiveSubProblemSolver

```python
class SubProblemSolver:
    # ... existing methods ...

    def __init__(
        self,
        # ... existing params ...
        enable_adaptive_allocation: bool = True,
        complexity_classifier: Optional[TaskComplexityClassifier] = None,
        adaptive_allocator: Optional[AdaptiveMDAPAllocator] = None
    ):
        """
        Initialize sub-problem solver with adaptive allocation.

        Args:
            enable_adaptive_allocation: Enable adaptive mode (default: True)
            complexity_classifier: Custom complexity classifier
            adaptive_allocator: Custom resource allocator
        """
        pass

    def solve(
        self,
        sub_problem: SubProblem,
        strategy: Optional[SolvingStrategy] = None,
        workflow_epic_id: Optional[str] = None,
        force_adaptive: bool = False
    ) -> SolutionAttempt:
        """
        Solve sub-problem with adaptive resource allocation.

        Args:
            sub_problem: SubProblem to solve
            strategy: Explicit strategy (bypasses adaptive if provided)
            workflow_epic_id: Hephaestus epic ID for tracking
            force_adaptive: Force adaptive mode even if strategy provided

        Returns:
            SolutionAttempt with results
        """
        pass

    def solve_adaptive(
        self,
        sub_problem: SubProblem,
        workflow_epic_id: Optional[str] = None
    ) -> SolutionAttempt:
        """
        Solve using adaptive allocation (explicit call).

        Args:
            sub_problem: SubProblem to solve
            workflow_epic_id: Hephaestus epic ID

        Returns:
            SolutionAttempt with adaptive metadata
        """
        pass

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """
        Get adaptive allocation statistics.

        Returns:
            Dict with complexity and allocation stats
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
ADAPTIVE_MDAP_THRESHOLDS=0.3,0.7

# Feature Weights (comma-separated)
ADAPTIVE_MDAP_WEIGHT_TEXT_LENGTH=0.20
ADAPTIVE_MDAP_WEIGHT_DOMAIN_RARITY=0.30
ADAPTIVE_MDAP_WEIGHT_DEPTH=0.20
ADAPTIVE_MDAP_WEIGHT_HISTORICAL_ERROR=0.20
ADAPTIVE_MDAP_WEIGHT_DEPENDENCY=0.10

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
      text_length: 0.20
      domain_rarity: 0.30
      depth: 0.20
      historical_error: 0.20
      dependency: 0.10

  # Resource Allocator
  allocator:
    thresholds: [0.3, 0.7]
    enable_learning: false
    learning_rate: 0.01
    min_samples: 100

  # Strategy Configurations
  strategies:
    direct:
      n_agents: 1
      k_ahead: 0
      max_retries: 1

    mdap_light:
      n_agents: 3
      k_ahead: 1
      max_retries: 2

    maker_full:
      n_agents: 5
      k_ahead: 2
      max_retries: 3

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
        "DIRECT": 0.40,
        "MDAP_LIGHT": 0.40,
        "MAKER_FULL": 0.20
    },
    "avg_complexity": 0.45,
    "complexity_std": 0.18,
    "estimated_savings": 0.48
}
```

#### Quality Metrics
```python
{
    "accuracy_by_strategy": {
        "DIRECT": 0.95,
        "MDAP_LIGHT": 0.98,
        "MAKER_FULL": 0.999
    },
    "overall_accuracy": 0.97,
    "accuracy_vs_baseline": 0.99  # Within 1%
}
```

#### Cost Metrics
```python
{
    "agent_calls_by_strategy": {
        "DIRECT": 400,
        "MDAP_LIGHT": 1200,
        "MAKER_FULL": 1000
    },
    "total_agent_calls": 2600,
    "baseline_agent_calls": 5000,  # All MAKER_FULL
    "actual_savings": 0.48,
    "cost_reduction_percentage": 48.0
}
```

#### Latency Metrics
```python
{
    "avg_latency_by_strategy_ms": {
        "DIRECT": 500,
        "MDAP_LIGHT": 2000,
        "MAKER_FULL": 5000
    },
    "overall_avg_latency_ms": 2100,
    "latency_vs_baseline_ms": -300  # Faster than baseline
}
```

### Hephaestus Dashboard

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

**File:** `tests/test_adaptive_mdap_complexity.py`

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

def test_depth_feature():
    """Test depth normalization."""
    # Depth 0 → score 0
    # Depth 5 → score 0.5
    # Depth 10+ → score 1.0

def test_historical_error_feature():
    """Test historical error rate."""
    # Low error domain → low score
    # High error domain → high score
    # Unknown domain → default 0.5

def test_dependency_feature():
    """Test dependency complexity."""
    # 0 deps → score 0
    # 5 deps → score 0.5
    # 10+ deps → score 1.0

def test_complexity_combination():
    """Test weighted combination."""
    # Verify weights sum to 1.0
    # Verify output in [0, 1]
    # Verify deterministic for same input

def test_cache_stability():
    """Test caching doesn't affect results."""
    # Same input before/after cache → same score
```

**File:** `tests/test_adaptive_mdap_allocator.py`

```python
def test_low_complexity_allocation():
    """Test low complexity → DIRECT."""
    # Complexity 0.1 → DIRECT strategy
    # Verify n_agents=1, k_ahead=0

def test_medium_complexity_allocation():
    """Test medium complexity → MDAP_LIGHT."""
    # Complexity 0.5 → MDAP_LIGHT strategy
    # Verify n_agents=3, k_ahead=1

def test_high_complexity_allocation():
    """Test high complexity → MAKER_FULL."""
    # Complexity 0.9 → MAKER_FULL strategy
    # Verify n_agents=5, k_ahead=2

def test_threshold_boundaries():
    """Test behavior at threshold boundaries."""
    # Exactly at threshold → higher strategy
    # Just below threshold → lower strategy

def test_custom_thresholds():
    """Test custom threshold configuration."""
    # Custom thresholds → correct allocation
    # Verify stats updated correctly

def test_statistics_tracking():
    """Test allocation statistics."""
    # Multiple allocations → correct counts
    # Verify distribution percentages
    # Verify savings calculation

def test_threshold_update():
    """Test dynamic threshold updates."""
    # Update thresholds → allocation changes
    # Verify stats reset on update
```

### Integration Tests

**File:** `tests/test_adaptive_mdap_integration.py`

```python
def test_adaptive_solve_direct():
    """Test adaptive solve routes to DIRECT."""
    # Low complexity sub-problem
    # Verify standard LLM call used
    # Verify single agent execution

def test_adaptive_solve_mdap_light():
    """Test adaptive solve routes to MDAP_LIGHT."""
    # Medium complexity sub-problem
    # Verify MDAP engine used
    # Verify 3 agents, k=1

def test_adaptive_solve_maker_full():
    """Test adaptive solve routes to MAKER_FULL."""
    # High complexity sub-problem
    # Verify MAKER engine used
    # Verify 5 agents, k=2

def test_explicit_strategy_override():
    """Test explicit strategy bypasses adaptive."""
    # Any complexity + explicit strategy
    # Verify explicit strategy used
    # Verify adaptive NOT consulted

def test_backward_compatibility():
    """Test backward compatibility."""
    # Existing code without adaptive
    # Verify still works
    # Verify no errors

def test_hephaestus_tracking():
    """Test Hephaestus tracking integration."""
    # Adaptive solve with tracking enabled
    # Verify tickets created
    # Verify metrics logged

def test_adaptive_disable():
    """Test disabling adaptive mode."""
    # enable_adaptive_allocation=False
    # Verify falls back to default strategy
    # Verify no complexity computed
```

### End-to-End Tests

**File:** `tests/test_adaptive_mdap_e2e.py`

```python
def test_full_workflow_adaptive():
    """Test full workflow with adaptive allocation."""
    # Create decomposition workflow
    # Run with adaptive enabled
    # Verify quality maintained
    # Verify cost reduced

def test_ab_test_comparison():
    """Test A/B comparison: adaptive vs baseline."""
    # Same workload, both modes
    # Compare quality metrics
    # Compare cost metrics
    # Verify adaptive wins or ties

def test_edge_cases():
    """Test edge cases."""
    # Empty description → default complexity
    # Very long description → high complexity
    # Unknown domain → default rarity
    # Zero depth → low complexity

def test_stress_test():
    """Test with large workload."""
    # 1000 sub-problems
    # Verify no crashes
    # Verify performance acceptable
    # Verify memory reasonable

def test_rollback_scenario():
    """Test rollback to non-adaptive."""
    # Adaptive → issues detected
    # Rollback to standard mode
    # Verify smooth transition
    # Verify no data loss
```

### Performance Tests

**File:** `tests/test_adaptive_mdap_performance.py`

```python
def test_complexity_computation_latency():
    """Test complexity computation is fast."""
    # Measure time for 1000 classifications
    # Verify < 10ms per classification
    # Verify caching helps

def test_allocator_latency():
    """Test allocator is fast."""
    # Measure time for 1000 allocations
    # Verify < 1ms per allocation
    # Verify no blocking operations

def test_overhead_comparison():
    """Test adaptive vs non-adaptive overhead."""
    # Same workload, both modes
    # Measure total time
    # Verify adaptive overhead < 5%

def test_memory_usage():
    """Test memory usage is reasonable."""
    # 10000 classifications
    # Verify memory < 500MB
    # Verify cache doesn't grow unbounded
```

---

## Performance Expectations

### Quality

**Target:** Maintain quality within ±1% of baseline (full MAKER)

**Rationale:**
- Easy tasks (low complexity): Direct solve sufficient (high baseline accuracy)
- Medium tasks (medium complexity): Light voting catches most errors
- Hard tasks (high complexity): Full MAKER ensures zero errors

**Validation:**
```python
# Expected results
baseline_accuracy = 0.990  # Full MAKER
adaptive_accuracy = 0.985  # Within ±1%
acceptable_range = [0.980, 1.000]

assert adaptive_accuracy in acceptable_range
```

### Cost Savings

**Target:** 30-50% reduction in agent calls vs baseline (always MAKER_FULL)

**Rationale:**
```
Assuming complexity distribution:
- 40% low complexity → 1 agent (vs 5) = 80% savings
- 40% medium complexity → 3 agents (vs 5) = 40% savings
- 20% high complexity → 5 agents (vs 5) = 0% savings

Overall: 0.4×0.8 + 0.4×0.4 + 0.2×0.0 = 48% savings
```

**Validation:**
```python
# Expected results
baseline_agent_calls = 5000  # All MAKER_FULL
adaptive_agent_calls = 2600  # Mixed strategies
savings_percentage = (5000 - 2600) / 5000 = 48%

assert 30 <= savings_percentage <= 50
```

### Latency

**Target:** Improved or neutral latency vs baseline

**Rationale:**
- DIRECT: ~500ms (vs 5000ms for MAKER) → 90% faster
- MDAP_LIGHT: ~2000ms (vs 5000ms) → 60% faster
- MAKER_FULL: ~5000ms (same as baseline)
- Weighted average: ~2100ms → faster than baseline

**Validation:**
```python
# Expected results
baseline_latency_ms = 5000  # All MAKER_FULL
adaptive_latency_ms = 2100  # Mixed strategies
improvement_percentage = (5000 - 2100) / 5000 = 58%

assert adaptive_latency_ms <= baseline_latency_ms
```

### Scalability

**Target:** Maintain logarithmic scaling from MAKER

**Rationale:**
```
MAKER: E[cost] = Θ(s × log(s))
Adaptive: E[cost] = Θ(s × log(s) × allocation_factor)

Where allocation_factor < 1.0 for mixed-complexity workloads
```

**Validation:**
```python
# Expected results
s_steps = 1000000
baseline_cost = s_steps * math.log(s_steps) * constant
adaptive_cost = baseline_cost * 0.52  # 48% savings

# Verify logarithmic scaling maintained
for s in [100, 1000, 10000, 100000, 1000000]:
    cost = adaptive_cost_for_steps(s)
    assert cost == O(s * log(s))
```

---

## Troubleshooting

### Issue: All sub-problems routed to MAKER_FULL

**Symptoms:**
- Strategy distribution shows 100% MAKER_FULL
- No cost savings achieved
- Complexity scores all > 0.7

**Diagnosis:**
```python
# Check complexity scores
stats = solver.get_adaptive_stats()
print(stats['complexity_scores'])

# Check thresholds
print(allocator.thresholds)

# Check feature weights
print(classifier.feature_weights)
```

**Solutions:**
1. Lower thresholds: `[0.5, 0.9]` → `[0.3, 0.7]`
2. Adjust feature weights (reduce dominance)
3. Verify feature normalization (should be [0, 1])
4. Check historical error rates (may be inflated)

### Issue: Quality degradation

**Symptoms:**
- Accuracy drops > 1% vs baseline
- DIRECT or MDAP_LIGHT failures increased

**Diagnosis:**
```python
# Check error rates by strategy
for strategy in [DIRECT, MDAP_LIGHT, MAKER_FULL]:
    error_rate = compute_error_rate(strategy)
    print(f"{strategy}: {error_rate}")
```

**Solutions:**
1. Raise lower threshold (0.3 → 0.4)
2. Increase n_agents for MDAP_LIGHT (3 → 4)
3. Increase k_ahead for MDAP_LIGHT (1 → 2)
4. Review complexity features (may be underestimating)

### Issue: No cost savings

**Symptoms:**
- Agent calls similar to baseline
- Estimated savings < 10%

**Diagnosis:**
```python
# Check strategy distribution
distribution = allocator.get_allocation_stats()['strategy_distribution']
print(distribution)

# Check if DIRECT/MDAP_LIGHT under-utilized
if distribution['DIRECT'] < 0.2:
    print("DIRECT under-utilized")
```

**Solutions:**
1. Lower thresholds (more aggressive routing)
2. Adjust feature weights (reduce complexity scores)
3. Verify complexity computation (may be overestimating)
4. Review workload composition (may be inherently complex)

### Issue: High latency

**Symptoms:**
- Adaptive mode slower than baseline
- Complexity computation bottleneck

**Diagnosis:**
```python
# Profile complexity computation
import cProfile
cProfile.run('classifier.compute_complexity(sub_problem)')

# Check cache hit rate
hit_rate = classifier.cache_hits / classifier.cache_requests
print(f"Cache hit rate: {hit_rate}")
```

**Solutions:**
1. Enable caching (embeddings, domain rarity)
2. Use lighter embedding model (MiniLM vs larger)
3. Batch complexity computations
4. Pre-compute domain rarity scores

---

## Conclusion

The Adaptive-MAKER integration represents a significant opportunity to reduce costs while maintaining the zero-error reliability that MAKER provides. By adapting the proven SBM-Efficient pattern to the agent orchestration layer, we can achieve 30-50% cost savings on mixed-complexity workloads with minimal quality impact.

**Key Success Factors:**
1. ✅ **Validated concept** - SBM-Efficient proves 40-60% savings possible
2. ✅ **Minimal risk** - Opt-in enhancement with easy rollback
3. ✅ **Strong synergy** - Complements existing MDAP/MAKER system
4. ✅ **Clear roadmap** - 5-week implementation plan with defined milestones

**Next Steps:**
1. Review and approve this integration guide
2. See detailed implementation todolist
3. Begin Phase 1: Foundation
4. Validate complexity classification on historical data
5. Iterate based on validation results

For implementation details, see [ADAPTIVE_MAKER_TODOLIST.md](./ADAPTIVE_MAKER_TODOLIST.md).

---

**Document Version:** 1.0
**Last Updated:** 2025-01-17
**Author:** OpenEvolve Integration Team
**Status:** Ready for Implementation
