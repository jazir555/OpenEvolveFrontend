# OpenEvolve vs LoongFlow Strategy Selection Systems: Comprehensive Analysis

## Executive Summary

This analysis compares the strategy selection and adaptive systems in OpenEvolve and LoongFlow, identifying their strengths, weaknesses, and opportunities for integration.

---

## 1. System Overview

### OpenEvolve Strategy System

| Component | Location | Purpose |
|-----------|----------|---------|
| `adaptive_strategy_selector.py` | Root | Performance tracking and weight calculation |
| `strategy_templates.py` | Root | Predefined strategy templates (domain, priority, complexity, team-based) |
| `knowledge_engine/core/strategy_recommender.py` | knowledge_engine/ | Ensemble strategy selection with ML |
| `EvolutionConfiguration` | evolution.py | 272+ parameter configuration class |

**Key Files:**
- `adaptive_strategy_selector.py`: Contains `StrategyPerformanceTracker`, `AdaptiveWeightCalculator`
- `strategy_templates.py`: Contains 7 templates (domain_specific, priority_based, complexity_based, team_based, agile_sprint, research_phased, microservices)
- `strategy_recommender.py`: Contains `EnsembleStrategySelector`, `OnlineLearningTracker`

### LoongFlow Strategy System

| Component | Location | Purpose |
|-----------|----------|---------|
| `PESAgent` | framework/pes/pes_agent.py | Main PES orchestrator |
| `Planner` | framework/pes/planner/planner.py | Planning phase worker |
| `GeneralPlanAgent` | agents/general_agent/planner.py | Claude Code-based planner |
| `EvolveChainConfig` | framework/pes/context/config.py | Configuration with cost tracking |

**Key Files:**
- `pes_agent.py`: Plan-Execute-Summarize loop with concurrency control
- `general_agent/planner.py`: LLM-based planning with database integration
- `context/config.py`: Configuration with token cost tracking

---

## 2. Detailed Comparison Table

### Strategy Selection Mechanisms

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Primary Selection Method** | Rule-based + Ensemble (4 methods) | LLM-based planning + PES loop | Tie - different paradigms |
| **Ensemble Methods** | Rule-based, Similarity, Trend, ML | N/A (single PES approach) | **OpenEvolve** |
| **Learning from History** | ✅ Online learning tracker with adaptive weights | ⚠️ Limited (via database sampling) | **OpenEvolve** |
| **Confidence Intervals** | ✅ Bootstrap-based confidence intervals | ❌ Not implemented | **OpenEvolve** |
| **Cold Start Handling** | ✅ Domain heuristics + rule-based fallback | ⚠️ Initial solution evaluation | **OpenEvolve** |

### Cost-Aware Decision Making

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Cost Categories** | cheap/moderate/expensive/very_expensive | Token price tracking per request | **OpenEvolve** (higher-level) |
| **Evaluation Cost Triggers** | ✅ Primary decision factor (Rule 1: expensive → PES) | ⚠️ Tracked but not decision driver | **OpenEvolve** |
| **Budget Management** | cost_limit_usd parameter | completion_token_price + prompt_token_price | **LoongFlow** (more granular) |
| **Cost-Benefit Analysis** | ⚠️ Implicit via sample efficiency | ❌ Not implemented | **OpenEvolve** |
| **API Cost Tracking** | ❌ Not implemented | ✅ Per-request token counting | **LoongFlow** |

### Trigger Mechanisms for Strategy Changes

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Trigger Type** | Performance-based + Rule-based | Iteration-based (PES cycle) | **OpenEvolve** |
| **Performance Thresholds** | success_rate < 0.5 triggers alert | target_score comparison | **OpenEvolve** |
| **Automatic Adaptation** | ✅ Weight adjustment every 10 attempts | ❌ Manual configuration | **OpenEvolve** |
| **Strategy Degradation Detection** | ✅ Alert when quality drops | ❌ Not implemented | **OpenEvolve** |
| **Real-time Switching** | ✅ Can switch between systems | ❌ Fixed at start | **OpenEvolve** |

### Directed vs Random Search Decisions

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Exploration/Exploitation** | exploration_rate (0.2), exploitation_ratio (0.7) | exploration_rate in database config | Tie |
| **Boltzmann Sampling** | ⚠️ Not in main config | ✅ boltzmann_temperature parameter | **LoongFlow** |
| **Diversity Maintenance** | ✅ diversity_maintenance, diversity_weight | ⚠️ feature_dimensions for MAP-Elites | **OpenEvolve** |
| **Adaptive Parameters** | ✅ adaptive_parameters flag | ❌ Not implemented | **OpenEvolve** |
| **Random Ratio** | ✅ random_ratio (0.2) | ❌ Not explicit | **OpenEvolve** |

### Planning and Strategy Templates

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Template System** | ✅ 7 predefined templates | ❌ No template system | **OpenEvolve** |
| **Domain-Specific Prompts** | ✅ software, research, data_science, business | ⚠️ Via Claude system prompts | **OpenEvolve** |
| **Planning Phase** | ❌ Not implemented | ✅ Core PES planner with LLM | **LoongFlow** |
| **Plan Refinement** | ❌ Not implemented | ✅ max_refinement_iterations | **LoongFlow** |
| **Context-Aware Planning** | ⚠️ Via domain heuristics | ✅ Full context with parent sampling | **LoongFlow** |

### Knowledge Integration

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Knowledge Engine Integration** | ✅ Optional integration for strategy storage | ⚠️ Database-based memory | **OpenEvolve** |
| **Historical Performance Query** | ✅ Query for recommendations | ⚠️ Sample from database | **OpenEvolve** |
| **Pattern Recognition** | ✅ ML-based prediction | ❌ Not implemented | **OpenEvolve** |
| **Cross-Session Learning** | ✅ Via knowledge artifacts | ⚠️ Via checkpoint loading | **OpenEvolve** |
| **Alerting Integration** | ✅ Strategy degradation alerts | ❌ Not implemented | **OpenEvolve** |

### Execution and Concurrency

| Aspect | OpenEvolve | LoongFlow | Winner |
|--------|-----------|-----------|--------|
| **Concurrency Model** | parallel_evaluations parameter | asyncio-based with max_workers | **LoongFlow** |
| **Island Model** | ✅ num_islands, migration | ✅ num_islands, migration_interval | Tie |
| **Checkpointing** | checkpoint_interval | checkpoint-iter-{id}-{count} format | **LoongFlow** (more detailed) |
| **Task Management** | ❌ Not implemented | ✅ Running task tracking with cleanup | **LoongFlow** |
| **Interruption Handling** | ❌ Not implemented | ✅ Graceful interruption with _stop_event | **LoongFlow** |

---

## 3. What LoongFlow Does Better

### 3.1 Plan-Execute-Summarize Architecture
```python
# LoongFlow's core strength: Structured reasoning loop
async def _evolution_cycle(self, iteration_id: int):
    # 1. Planner Step - LLM-based strategic planning
    planner_result = await planner.run(context, None)
    
    # 2. Executor Step - Execute the plan
    executor_result = await executor.run(context, planner_result)
    
    # 3. Summary Step - Learn from execution
    summary_result = await summary.run(context, executor_result)
```

**Advantages:**
- **Explicit reasoning**: Each phase has clear inputs/outputs
- **Context preservation**: Database tracks parent-child relationships
- **Plan refinement**: Can iterate on plans based on feedback

### 3.2 Granular Cost Tracking
```python
# LoongFlow tracks actual API costs per request
prompt_cost = (prompt_tokens / 1000) * self.config.llm_config.prompt_token_price
completion_cost = (completion_tokens / 1000) * self.config.llm_config.completion_token_price
```

**Advantages:**
- Real-time budget monitoring
- Per-iteration cost attribution
- Token-level granularity

### 3.3 Robust Task Management
```python
# LoongFlow features:
- asyncio.create_task() with tracking
- _stop_event for graceful interruption
- _completion_lock for thread safety
- Automatic task cleanup on failure
```

### 3.4 Checkpoint Format
```
checkpoint-iter-{iteration_id}-{completion_count}
```
Enables precise state restoration with completion tracking.

---

## 4. What OpenEvolve Does Better

### 4.1 Ensemble Strategy Selection
```python
# OpenEvolve combines 4 prediction methods
methods = [
    ('rule_based', 0.25),
    ('similarity', 0.35),
    ('trend', 0.25),
    ('ml', 0.15)
]
# Weights adapt based on accuracy
```

**Advantages:**
- Multiple perspectives on strategy selection
- Weighted voting reduces single-point failure
- Online learning adapts weights over time

### 4.2 Cost-Aware Strategy Selection
```python
# Rule 1 in OpenEvolve: Expensive evaluations → PES
if problem_chars.evaluation_cost in ["expensive", "very_expensive"]:
    system = EvolutionSystem.LOONGFLOW  # Use PES for expensive evals
    mode = EvolutionMode.PES
    confidence = 0.85
    reasoning.append("Expensive evaluations favor PES (60% fewer evaluations)")
```

**Advantages:**
- Makes cost a primary decision factor
- Explicit trade-off between evaluation cost and strategy choice
- Can justify decisions with cost savings estimates

### 4.3 Strategy Templates
```python
# 7 predefined templates for different scenarios
templates = [
    "domain_specific",    # software, research, data_science, business
    "priority_based",     # Business value focus
    "complexity_based",   # Cognitive load balancing
    "team_based",         # Expertise matching
    "agile_sprint",       # Sprint-sized problems
    "research_phased",    # Scientific methodology
    "microservices"       # Architecture decomposition
]
```

### 4.4 Domain-Specific Heuristics
```python
# Pre-configured for 7 domains
domain_heuristics = {
    "finance": {"preferred_modes": ["pes", "mo", "standard"], "typical_iterations": 50},
    "trading": {"preferred_modes": ["qd", "pes", "adversarial"], "typical_iterations": 100},
    "science": {"preferred_modes": ["pes", "qd", "standard"], "typical_iterations": 30},
    # ... more domains
}
```

---

## 5. Specific Algorithms and Patterns to Extract

### From OpenEvolve:

#### 5.1 Adaptive Weight Calculation Algorithm
```python
class AdaptiveWeightCalculator:
    def calculate_adaptive_adjustment(self, strategy: str, problem_type: Optional[str] = None) -> float:
        data = self.tracker.get_strategy_data(strategy)
        if not data or data.total_attempts < 3:
            return 1.0  # Not enough data
        
        # Combine factors (success is more important)
        success_factor = data.success_rate
        quality_factor = data.average_quality / 100.0
        adjustment = success_factor * 0.7 + quality_factor * 0.3
        
        # Map to adjustment range: 0.5 (reduce) → 1.0 (no change) → 2.0 (increase)
        if adjustment < 0.4:
            return 0.5
        elif adjustment > 0.8:
            return 2.0
        else:
            return 0.5 + (adjustment - 0.4) / 0.4
```

**Key Pattern:** Success-rate-weighted adjustment with thresholds

#### 5.2 Ensemble Weighted Voting
```python
def _weighted_voting(self, predictions: List[MethodPrediction], weights: Dict[str, float]):
    votes = defaultdict(float)
    for pred in predictions:
        weight = weights.get(pred.method, 0.25)
        key = (pred.system, pred.mode)
        votes[key] += weight * pred.confidence
    
    winner = max(votes.items(), key=lambda x: x[1])
    
    # Calculate agreement using entropy
    total_votes = sum(votes.values())
    normalized_votes = {k: v / total_votes for k, v in votes.items()}
    entropy = -sum(p * math.log(p) for p in normalized_votes.values() if p > 0)
    max_entropy = math.log(len(votes))
    agreement = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
    
    return winner[0], agreement
```

**Key Pattern:** Entropy-based agreement calculation for ensemble confidence

#### 5.3 Bootstrap Confidence Intervals
```python
async def _calculate_confidence_interval(self, strategy, problem_chars, history, confidence_level=0.95):
    # Bootstrap sampling
    n_samples = 1000
    bootstrap_scores = []
    
    for _ in range(n_samples):
        sample = random.choices(relevant_runs, k=len(relevant_runs))
        mean_score = sum(r.final_score for r in sample) / len(sample)
        bootstrap_scores.append(mean_score)
    
    # Calculate percentiles
    alpha = 1.0 - confidence_level
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    
    return point_estimate, (lower, upper)
```

**Key Pattern:** Non-parametric confidence intervals without distributional assumptions

### From LoongFlow:

#### 5.4 PES Loop Pattern
```python
async def _evolution_cycle(self, iteration_id: int):
    trace_id = str(uuid.uuid4().hex[:12])
    
    # Phase 1: Plan
    planner = get_worker(planner_name, PLANNER, config=planner_config, db=self.database)
    planner_result = await planner.run(context, None)
    
    # Phase 2: Execute
    executor = get_worker(executor_name, EXECUTOR, config=executor_config, evaluator=evaluator, db=self.database)
    executor_result = await executor.run(context, planner_result)
    
    # Phase 3: Summarize
    summary = get_worker(summary_name, SUMMARY, config=summary_config, db=self.database)
    summary_result = await summary.run(context, executor_result)
    
    # Track costs
    prompt_cost = (prompt_tokens / 1000) * self.config.llm_config.prompt_token_price
    completion_cost = (completion_tokens / 1000) * self.config.llm_config.completion_token_price
```

**Key Pattern:** Three-phase structured evolution with explicit data flow

#### 5.5 Worker Registration Pattern
```python
# Clean separation via registry
_REGISTRY: Dict[str, Any] = {}

def register_worker(name: str, phase: str, worker_class: type):
    _REGISTRY[f"{phase}:{name}"] = worker_class

def get_worker(name: str, phase: str, **kwargs):
    key = f"{phase}:{name}"
    worker_class = _REGISTRY.get(key)
    return worker_class(**kwargs)
```

**Key Pattern:** Phase-prefixed registry for clean component organization

---

## 6. Integration Recommendations

### 6.1 Immediate Wins (Low Effort, High Impact)

#### Add LoongFlow's Cost Tracking to OpenEvolve
```python
# Add to EvolutionConfiguration
token_price_per_1k: Dict[str, float] = Field(default_factory=dict)
total_tokens_used: int = Field(default=0)
total_cost_usd: float = Field(default=0.0)

def track_request(self, prompt_tokens: int, completion_tokens: int, model: str):
    price = self.token_price_per_1k.get(model, 0.01)
    self.total_tokens_used += prompt_tokens + completion_tokens
    self.total_cost_usd += (prompt_tokens + completion_tokens) / 1000 * price
```

#### Add OpenEvolve's Strategy Selection to LoongFlow
```python
# In PESAgent.__init__(), add strategy selector
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

self.strategy_selector = EnsembleStrategySelector(
    enable_loongflow=True,
    learning_enabled=True
)

# Before starting evolution, select optimal strategy
async def run(self):
    recommendation = await self.strategy_selector.recommend_with_ensemble(
        problem_description=self.config.evolve.task,
        domain=self.config.metadata.get("domain", "general"),
        constraints=self.config.metadata.get("constraints", {})
    )
    # Use recommendation to configure planner/executor
```

### 6.2 Medium-Term Integrations

#### Unified Configuration Schema
```python
class UnifiedEvolutionConfig(BaseModel):
    # OpenEvolve-style strategy selection
    strategy_selection: StrategySelectionConfig
    
    # LoongFlow-style PES configuration
    pes_config: PESConfig
    
    # Unified cost tracking
    cost_tracking: CostTrackingConfig  # From LoongFlow
    
    # OpenEvolve-style templates
    strategy_templates: Dict[str, StrategyTemplate]
    
    # LoongFlow-style worker registry
    worker_registry: WorkerRegistryConfig
```

#### Hybrid Strategy Selection
```python
async def select_strategy(problem, constraints, history):
    # Step 1: Use OpenEvolve's ensemble for high-level selection
    ensemble_pred = await openevolve_selector.recommend_with_ensemble(
        problem, constraints
    )
    
    # Step 2: If PES selected, use LoongFlow's planner for detailed planning
    if ensemble_pred.strategy[1] == "pes":
        detailed_plan = await loongflow_planner.create_plan(
            problem, 
            parent_sampling=True,
            budget_constraints=constraints.get("budget")
        )
        return {
            "system": "loongflow",
            "mode": "pes",
            "high_level": ensemble_pred,
            "detailed_plan": detailed_plan
        }
    
    # Step 3: Otherwise use OpenEvolve's execution
    return {
        "system": "openevolve",
        "mode": ensemble_pred.strategy[1],
        "config_overrides": ensemble_pred.config_overrides
    }
```

### 6.3 Long-Term Vision

#### Unified Evolution Engine
```python
class UnifiedEvolutionEngine:
    """
    Combines OpenEvolve's strategy intelligence with LoongFlow's execution
    """
    
    def __init__(self):
        self.strategy_selector = EnsembleStrategySelector()  # OpenEvolve
        self.pes_orchestrator = PESOrchestrator()  # LoongFlow
        self.openevolve_executor = OpenEvolveExecutor()  # OpenEvolve
        self.cost_tracker = GranularCostTracker()  # LoongFlow-style
    
    async def evolve(self, problem, config):
        # Phase 1: Strategy selection (OpenEvolve intelligence)
        strategy = await self.strategy_selector.select(problem, config)
        
        # Phase 2: Planning (LoongFlow if PES, OpenEvolve templates otherwise)
        if strategy.mode == "pes":
            plan = await self.pes_orchestrator.plan(problem, strategy)
            result = await self.pes_orchestrator.execute(plan)
        else:
            template = self.get_template(strategy.template_name)
            result = await self.openevolve_executor.run(problem, template, strategy)
        
        # Phase 3: Learning (both systems contribute)
        await self.strategy_selector.learn_from_run(result)
        await self.pes_orchestrator.update_memory(result)
        
        return result
```

---

## 7. Code Patterns to Port

### Port FROM OpenEvolve TO LoongFlow:

1. **Ensemble strategy selection** (`strategy_recommender.py` lines 770-900)
2. **Online learning tracker** (`strategy_recommender.py` lines 227-444)
3. **Bootstrap confidence intervals** (`strategy_recommender.py` lines 1770-1821)
4. **Strategy templates** (`strategy_templates.py` full file)
5. **Domain heuristics** (`strategy_recommender.py` lines 468-520)
6. **Cost-aware decision rules** (`strategy_recommender.py` lines 1363-1428)

### Port FROM LoongFlow TO OpenEvolve:

1. **Token-level cost tracking** (`pes_agent.py` lines 294-312)
2. **Async task management** (`pes_agent.py` lines 377-398, 574-587)
3. **Checkpoint naming convention** (`pes_agent.py` lines 348-376)
4. **Worker registration pattern** (`register.py` full file)
5. **PES loop structure** (`pes_agent.py` lines 174-328)
6. **Parent sampling from database** (`general_agent/planner.py` lines 119-126)

---

## 8. Summary Table

| Capability | OpenEvolve | LoongFlow | Recommended Approach |
|------------|-----------|-----------|---------------------|
| Strategy Selection | ⭐⭐⭐ Ensemble with ML | ⭐⭐ Rule-based PES | **Combine**: Ensemble selects, PES executes |
| Cost Awareness | ⭐⭐⭐ High-level categories | ⭐⭐⭐ Token-level tracking | **Combine**: Categories for selection, tokens for monitoring |
| Planning | ⭐ Templates only | ⭐⭐⭐ LLM-based planning | **LoongFlow**: PES planning |
| Execution | ⭐⭐⭐ Parameter-rich | ⭐⭐ Async with tracking | **OpenEvolve**: 272+ parameters |
| Learning | ⭐⭐⭐ Online adaptive | ⭐⭐ Database memory | **OpenEvolve**: Weight adaptation |
| Concurrency | ⭐⭐ Parallel evals | ⭐⭐⭐ Async task mgmt | **LoongFlow**: Task tracking |
| Interruption | ⭐ Basic | ⭐⭐⭐ Graceful stop | **LoongFlow**: _stop_event pattern |
| Templates | ⭐⭐⭐ 7 templates | ⭐ None | **OpenEvolve**: Domain templates |
| Confidence | ⭐⭐⭐ Bootstrap CIs | ⭐ None | **OpenEvolve**: Uncertainty quantification |
| Cold Start | ⭐⭐⭐ Domain defaults | ⭐ Initial eval | **OpenEvolve**: Heuristics |

---

## 9. Conclusion

**Key Insight**: OpenEvolve excels at *deciding what to do* (strategy selection, cost-aware decisions, learning), while LoongFlow excels at *doing it* (structured execution, granular tracking, robust task management).

**Recommended Architecture**:
1. Use OpenEvolve's `EnsembleStrategySelector` for high-level strategy decisions
2. Use LoongFlow's `PESAgent` for PES-mode execution
3. Use OpenEvolve's `EvolutionConfiguration` for parameter management
4. Use LoongFlow's cost tracking for budget monitoring
5. Combine both learning systems for continuous improvement

**Files to Create**:
- `unified_strategy_orchestrator.py` - Combines selection from OpenEvolve with execution from LoongFlow
- `hybrid_cost_tracker.py` - OpenEvolve's cost categories + LoongFlow's token tracking
- `adaptive_pes_selector.py` - Uses OpenEvolve's ensemble to decide when to use PES vs other modes
