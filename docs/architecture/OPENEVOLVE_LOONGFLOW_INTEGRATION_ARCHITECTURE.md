# OpenEvolve + LoongFlow PES Integration Architecture

## Executive Summary

This document defines a unified integration architecture that combines **OpenEvolve's** powerful evolutionary optimization engine (MAP-Elites, NSGA-II, Lean 4/Z3 verification, language-agnostic code evolution) with **LoongFlow's** Plan-Execute-Summarize (PES) paradigm for cost-aware, directed search strategies.

**Core Philosophy**: PES enhances OpenEvolve evolution rather than replacing it. The integration creates a layered architecture where PES provides intelligent planning and cost optimization, while OpenEvolve provides the robust evolutionary machinery.

---

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED EVOLUTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STRATEGY ORCHESTRATION LAYER                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   PES Mode  │  │    QD Mode  │  │    MO Mode  │  │Standard Mode│ │   │
│  │  │  (Directed) │  │(MAP-Elites) │  │  (NSGA-II)  │  │  (Blind)    │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │   │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │   │
│  │                              │                                       │   │
│  │                    ┌─────────┴─────────┐                             │   │
│  │                    │  StrategySelector │                             │   │
│  │                    │  (Auto/Manual)    │                             │   │
│  │                    └─────────┬─────────┘                             │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │                    PES LAYER   │                                      │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │      PLAN PHASE           │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │ ProblemAnalyzer │      │      │  CostEstimator  │◄────────┼──┤   │
│  │  │  │                 │      │      │                 │         │  │   │
│  │  │  │ • Decomposition │──────┼─────►│ • Token Budget  │         │  │   │
│  │  │  │ • Strategy Rec  │      │      │ • Time Budget   │         │  │   │
│  │  │  │ • ConstraintMap │      │      │ • API Budget    │         │  │   │
│  │  │  └─────────────────┘      │      └─────────────────┘         │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │      EXECUTE PHASE        │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │ PESExecutor     │      │      │ BudgetMonitor   │◄────────┼──┤   │
│  │  │  │                 │◄─────┼──────│                 │         │  │   │
│  │  │  │ • Orchestrates  │      │      │ • Tracks spend  │         │  │   │
│  │  │  │ • Adapts params │──────┼─────►│ • Triggers stop │         │  │   │
│  │  │  │ • Manages flow  │      │      │ • Reports usage │         │  │   │
│  │  │  └────────┬────────┘      │      └─────────────────┘         │  │   │
│  │  └───────────┼───────────────┼───────────────────────────────────┘  │   │
│  │              │               │                                       │   │
│  │  ┌───────────┼───────────────┼───────────────────────────────────┐  │   │
│  │  │      SUMMARIZE PHASE      │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │ ResultSummarizer│      │      │ KnowledgeExtractor         │  │   │
│  │  │  │                 │      │      │                 │         │  │   │
│  │  │  │ • Patterns      │──────┼─────►│ • Store patterns│         │  │   │
│  │  │  │ • Insights      │      │      │ • Update index  │         │  │   │
│  │  │  │ • Next actions  │      │      │ • Guide future  │         │  │   │
│  │  │  └─────────────────┘      │      └─────────────────┘         │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │              OPENEVOLVE CORE ENGINE LAYER                          │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │      EVOLUTION ENGINE     │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │  MAP-Elites     │◄─────┼──────│ ArchiveManager  │         │  │   │
│  │  │  │  (QD Mode)      │      │      │                 │         │  │   │
│  │  │  └─────────────────┘      │      │ • Feature space │         │  │   │
│  │  │  ┌─────────────────┐      │      │ • Elites grid   │         │  │   │
│  │  │  │  NSGA-II        │◄─────┼──────│ • Novelty track │         │  │   │
│  │  │  │  (MO Mode)      │      │      └─────────────────┘         │  │   │
│  │  │  └─────────────────┘      │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │  LanguageAgnostic      │◄─────│ CodeEvolution   │         │  │   │
│  │  │  │  Evolution      │      │      │                 │         │  │   │
│  │  │  │                 │      │      │ • Python/JS/PHP │         │  │   │
│  │  │  │ • Multi-lang    │      │      │ • Java/C++/Go   │         │  │   │
│  │  │  │ • Auto-detect   │      │      │ • Lean 4        │         │  │   │
│  │  │  │ • Fix gen       │      │      └─────────────────┘         │  │   │
│  │  │  └─────────────────┘      │                                   │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │      VERIFICATION LAYER   │                                   │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │  Lean4Verifier  │◄─────┼──────│ TheoremProving  │         │  │   │
│  │  │  │                 │      │      │                 │         │  │   │
│  │  │  │ • Formal proofs │      │      │ • Z3 Solver     │         │  │   │
│  │  │  │ • Tactics       │      │      │ • SMT checking  │         │  │   │
│  │  │  └─────────────────┘      │      └─────────────────┘         │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │      TEAM SYSTEM (Red/Blue/Gold)                              │  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │   RedTeam       │◄─────┼──────│ AttackGenerator │         │  │   │
│  │  │  │   (Adversarial) │      │      │                 │         │  │   │
│  │  │  └─────────────────┘      │      │ • Adversarial   │         │  │   │
│  │  │  ┌─────────────────┐      │      │ • Robustness    │         │  │   │
│  │  │  │   BlueTeam      │◄─────┼──────│ FixGenerator    │         │  │   │
│  │  │  │   (Defense)     │      │      │                 │         │  │   │
│  │  │  └─────────────────┘      │      │ • Auto-fix      │         │  │   │
│  │  │  ┌─────────────────┐      │      │ • Patch apply   │         │  │   │
│  │  │  │   EvaluatorTeam │◄─────┼──────│ ConsensusEval   │         │  │   │
│  │  │  │   (Consensus)   │      │      │                 │         │  │   │
│  │  │  └─────────────────┘      │      │ • 3-Round Gauntlet         │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │              KNOWLEDGE & FEEDBACK LAYER                            │   │
│  │  ┌───────────────────────────┼───────────────────────────────────┐  │   │
│  │  │  ┌─────────────────┐      │      ┌─────────────────┐         │  │   │
│  │  │  │ KnowledgeGraph  │◄─────┼──────│ TemporalStore   │         │  │   │
│  │  │  │                 │      │      │                 │         │  │   │
│  │  │  │ • Patterns      │      │      │ • History       │         │  │   │
│  │  │  │ • Solutions     │      │      │ • Evolution     │         │  │   │
│  │  │  │ • Relationships │      │      │ • Trends        │         │  │   │
│  │  │  └─────────────────┘      │      └─────────────────┘         │  │   │
│  │  └───────────────────────────┼───────────────────────────────────┘  │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    CONFIGURATION SYSTEM   │
                    │  ┌─────────────────────┐  │
                    │  │ UnifiedParameters   │  │
                    │  │ • 272+ parameters   │  │
                    │  │ • PES-specific      │  │
                    │  │ • Cost controls     │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

---

## 2. Key Classes and Responsibilities

### 2.1 Strategy Orchestration Layer

#### `StrategyOrchestrator`
**Purpose**: Central coordinator that selects between PES-guided and standard evolution modes.

```python
class StrategyOrchestrator:
    """
    Orchestrates the selection and execution of evolution strategies.
    
    Responsibilities:
    - Analyze problem characteristics
    - Select optimal strategy (PES/QD/MO/Standard)
    - Coordinate between PES layer and OpenEvolve core
    - Monitor execution and adapt strategy
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.pes_adapter = PESOpenEvolveAdapter(config)
        self.strategy_selector = StrategySelector(config)
        self.cost_tracker = CostTracker(config)
    
    async def evolve(self, problem: EvolutionProblem) -> EvolutionResult:
        # 1. Analyze problem to determine strategy
        strategy = self.strategy_selector.select_strategy(problem)
        
        # 2. Route to appropriate execution path
        if strategy == StrategyType.PES_ENHANCED:
            return await self._run_pes_enhanced_evolution(problem)
        elif strategy == StrategyType.QD_STANDARD:
            return await self._run_standard_qd(problem)
        elif strategy == StrategyType.MO_STANDARD:
            return await self._run_standard_mo(problem)
        else:
            return await self._run_standard_evolution(problem)
```

#### `StrategySelector`
**Purpose**: Determines optimal strategy based on problem characteristics and cost constraints.

```python
@dataclass
class StrategyCharacteristics:
    problem_complexity: float  # 0.0 - 1.0
    budget_constraints: BudgetConstraints
    required_quality: float    # 0.0 - 1.0
    time_sensitivity: float    # 0.0 - 1.0
    knowledge_available: bool

class StrategySelector:
    """
    Selects optimal evolution strategy using decision tree and learned patterns.
    
    Rules:
    - High complexity + Tight budget → PES mode (directed search)
    - Exploration focus → QD mode (MAP-Elites)
    - Multi-objective → MO mode (NSGA-II)
    - Simple problems → Standard mode
    """
    
    def select_strategy(self, problem: EvolutionProblem) -> StrategyType:
        characteristics = self._analyze_problem(problem)
        
        # Decision logic
        if characteristics.budget_constraints.is_tight():
            # PES mode for cost-aware directed search
            return StrategyType.PES_ENHANCED
        elif problem.objectives and len(problem.objectives) > 1:
            return StrategyType.MO_STANDARD
        elif problem.exploration_focus:
            return StrategyType.QD_STANDARD
        else:
            return StrategyType.STANDARD
```

### 2.2 PES Layer

#### `PESPlanner`
**Purpose**: Creates intelligent evolution plans based on problem analysis.

```python
@dataclass
class EvolutionPlan:
    """PES Plan for OpenEvolve integration."""
    # Strategy selection
    recommended_mode: EvolutionMode  # standard, qd, mo, adversarial
    
    # Parameter recommendations
    suggested_parameters: Dict[str, Any]
    parameter_reasoning: str
    
    # Budget allocation
    budget_allocation: BudgetAllocation
    
    # Phase configuration
    phases: List[PlanPhase]
    
    # Early stopping criteria
    convergence_triggers: List[ConvergenceTrigger]
    
    # Expected outcomes
    expected_iterations: int
    success_probability: float

class PESPlanner:
    """
    Plans evolution strategy based on problem analysis and historical data.
    
    Responsibilities:
    - Analyze problem structure
    - Recommend evolution parameters
    - Allocate budget across phases
    - Set convergence criteria
    """
    
    def __init__(self, knowledge_engine: Optional[KnowledgeEngine] = None):
        self.knowledge_engine = knowledge_engine
        self.problem_analyzer = ProblemAnalyzer()
        self.cost_estimator = CostEstimator()
    
    async def create_plan(self, problem: EvolutionProblem) -> EvolutionPlan:
        # 1. Analyze problem structure
        analysis = await self.problem_analyzer.analyze(problem)
        
        # 2. Query knowledge for similar problems
        historical_patterns = None
        if self.knowledge_engine:
            historical_patterns = await self._query_knowledge(analysis)
        
        # 3. Estimate costs for different strategies
        cost_estimates = self.cost_estimator.estimate(analysis)
        
        # 4. Build optimized plan
        return self._build_plan(analysis, historical_patterns, cost_estimates)
```

#### `CostEstimator`
**Purpose**: Estimates and tracks evolution costs to enable budget-aware execution.

```python
@dataclass
class CostEstimate:
    """Cost estimate for evolution run."""
    # Token costs
    estimated_input_tokens: int
    estimated_output_tokens: int
    token_cost_usd: float
    
    # API costs
    estimated_api_calls: int
    api_cost_usd: float
    
    # Time costs
    estimated_duration_seconds: float
    
    # Total
    total_estimated_cost: float
    confidence: float  # 0.0 - 1.0

@dataclass
class BudgetAllocation:
    """Budget allocation across evolution phases."""
    planning_budget: CostBudget      # For PES planning
    evolution_budget: CostBudget     # For OpenEvolve execution
    verification_budget: CostBudget  # For Lean/Z3 verification
    contingency_reserve: float       # 0.0 - 1.0 (percentage)

class CostEstimator:
    """
    Estimates and tracks costs throughout evolution.
    
    Cost Model:
    - LLM tokens (input/output)
    - API calls
    - Verification operations (Lean/Z3)
    - Compute time
    """
    
    def estimate(self, analysis: ProblemAnalysis) -> CostEstimate:
        # Base costs from problem complexity
        base_tokens = self._estimate_base_tokens(analysis)
        
        # Add costs for verification if needed
        if analysis.requires_formal_verification:
            base_tokens += self._verification_cost(analysis)
        
        # Adjust for strategy
        if analysis.recommended_strategy == StrategyType.PES_ENHANCED:
            # PES adds planning overhead but reduces iterations
            base_tokens = int(base_tokens * 0.7)  # 30% reduction
        
        return CostEstimate(
            estimated_input_tokens=base_tokens,
            estimated_output_tokens=int(base_tokens * 0.3),
            token_cost_usd=self._calculate_token_cost(base_tokens),
            estimated_api_calls=self._estimate_api_calls(analysis),
            api_cost_usd=self._calculate_api_cost(analysis),
            estimated_duration_seconds=self._estimate_time(analysis),
            total_estimated_cost=self._calculate_total(analysis),
            confidence=0.8
        )
```

#### `PESExecutor`
**Purpose**: Executes the PES plan using OpenEvolve core engine.

```python
class PESExecutor:
    """
    Executes evolution according to PES plan.
    
    Responsibilities:
    - Translate PES plan to OpenEvolve parameters
    - Monitor execution against budget
    - Adapt parameters based on intermediate results
    - Coordinate with OpenEvolve evolution engine
    """
    
    def __init__(
        self,
        evolution_engine: OpenEvolveEngine,
        budget_monitor: BudgetMonitor,
        adaptation_engine: AdaptationEngine
    ):
        self.evolution_engine = evolution_engine
        self.budget_monitor = budget_monitor
        self.adaptation_engine = adaptation_engine
    
    async def execute_plan(
        self,
        plan: EvolutionPlan,
        problem: EvolutionProblem
    ) -> ExecutionResult:
        # 1. Configure OpenEvolve from plan
        config = self._translate_plan_to_config(plan)
        
        # 2. Set up budget monitoring
        self.budget_monitor.initialize(plan.budget_allocation)
        
        # 3. Execute with adaptation
        execution_state = ExecutionState()
        
        for phase in plan.phases:
            # Check budget
            if not self.budget_monitor.can_continue():
                execution_state.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                break
            
            # Execute phase
            phase_result = await self._execute_phase(
                phase, problem, config, execution_state
            )
            
            # Adapt if needed
            if phase_result.requires_adaptation:
                config = self.adaptation_engine.adapt(config, phase_result)
            
            execution_state.phase_results.append(phase_result)
        
        return ExecutionResult(
            state=execution_state,
            final_solution=execution_state.best_solution,
            cost_summary=self.budget_monitor.get_summary()
        )
```

#### `ResultSummarizer`
**Purpose**: Summarizes evolution results and extracts actionable insights.

```python
@dataclass
class EvolutionSummary:
    """Summary of evolution execution."""
    success: bool
    final_fitness: float
    iterations_completed: int
    budget_used: CostSummary
    
    # Insights
    key_insights: List[str]
    failure_modes: List[str]
    success_factors: List[str]
    
    # Recommendations
    recommended_next_actions: List[str]
    suggested_parameter_adjustments: Dict[str, Any]
    
    # Knowledge extracted
    patterns_discovered: List[Pattern]
    reusable_strategies: List[Strategy]

class ResultSummarizer:
    """
    Summarizes evolution results for knowledge extraction.
    
    Responsibilities:
    - Analyze execution trace
    - Extract patterns and insights
    - Generate recommendations
    - Update knowledge base
    """
    
    def summarize(self, execution: ExecutionResult) -> EvolutionSummary:
        return EvolutionSummary(
            success=execution.state.best_fitness > 0.9,
            final_fitness=execution.state.best_fitness,
            iterations_completed=len(execution.state.phase_results),
            budget_used=execution.cost_summary,
            key_insights=self._extract_insights(execution),
            failure_modes=self._identify_failures(execution),
            success_factors=self._identify_successes(execution),
            recommended_next_actions=self._recommend_actions(execution),
            patterns_discovered=self._extract_patterns(execution)
        )
```

### 2.3 OpenEvolve Core Integration

#### `PESOpenEvolveAdapter`
**Purpose**: Adapter that integrates PES layer with OpenEvolve core engine.

```python
class PESOpenEvolveAdapter:
    """
    Adapter between PES planning layer and OpenEvolve evolution engine.
    
    Responsibilities:
    - Translate PES plans to OpenEvolve configurations
    - Route evolution calls to appropriate engines
    - Inject PES guidance into evolution process
    - Collect results for PES summarization
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.map_elites_engine = MapElitesEngine()
        self.nsga2_engine = NSGA2Engine()
        self.standard_engine = StandardEvolutionEngine()
        self.language_agnostic_engine = LanguageAgnosticEngine()
    
    def translate_plan_to_config(
        self,
        plan: EvolutionPlan
    ) -> EvolutionConfiguration:
        """Translate PES plan to OpenEvolve configuration."""
        config = EvolutionConfiguration()
        
        # Apply suggested parameters
        for param_name, value in plan.suggested_parameters.items():
            if hasattr(config, param_name):
                setattr(config, param_name, value)
        
        # Set mode
        config.evolution_mode = plan.recommended_mode.value
        
        # Set budget constraints
        config.evaluation_budget = plan.budget_allocation.evolution_budget.max_tokens
        config.cost_limit_usd = plan.budget_allocation.evolution_budget.max_cost
        
        # Enable PES callbacks for adaptation
        config.enable_pes_callbacks = True
        
        return config
    
    async def run_evolution(
        self,
        problem: EvolutionProblem,
        config: EvolutionConfiguration,
        pes_callbacks: PESCallbacks
    ) -> OpenEvolveResult:
        """Run evolution with PES guidance."""
        
        # Select engine based on mode
        if config.evolution_mode == "qd":
            engine = self.map_elites_engine
        elif config.evolution_mode == "mo":
            engine = self.nsga2_engine
        elif problem.language != "python":
            engine = self.language_agnostic_engine
        else:
            engine = self.standard_engine
        
        # Run with PES callbacks
        return await engine.evolve(
            problem=problem,
            config=config,
            callbacks=pes_callbacks
        )
```

#### `DirectedEvolutionStrategy`
**Purpose**: Injects PES guidance into OpenEvolve's mutation/selection strategies.

```python
class DirectedEvolutionStrategy:
    """
    Directs evolution using PES planning insights.
    
    Instead of blind mutation, uses plan guidance to:
    - Direct mutations toward promising areas
    - Skip unpromising search regions
    - Adapt mutation rates based on progress
    """
    
    def __init__(self, evolution_plan: EvolutionPlan):
        self.plan = evolution_plan
        self.guidance_map = self._build_guidance_map()
    
    def guided_mutation(
        self,
        individual: Individual,
        fitness_history: List[float]
    ) -> Individual:
        """Apply mutation guided by PES plan."""
        
        # Get guidance for current state
        guidance = self._get_guidance(individual, fitness_history)
        
        if guidance.mutation_type == MutationType.TARGETED:
            # Targeted mutation based on plan
            return self._apply_targeted_mutation(individual, guidance)
        elif guidance.mutation_type == MutationType.EXPLORATORY:
            # Exploratory but constrained
            return self._apply_constrained_exploration(individual, guidance)
        else:
            # Standard mutation
            return self._apply_standard_mutation(individual)
    
    def guided_selection(
        self,
        population: List[Individual],
        plan_phase: PlanPhase
    ) -> List[Individual]:
        """Select individuals guided by plan phase objectives."""
        
        # Weight by alignment with plan objectives
        scored = [
            (ind, self._alignment_score(ind, plan_phase.objectives))
            for ind in population
        ]
        
        # Select top by weighted score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in scored[:self.plan.suggested_parameters['population_size']]]
```

---

## 3. Data Flow Between OpenEvolve and PES Layers

### 3.1 Planning Phase Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         PLANNING PHASE                          │
└─────────────────────────────────────────────────────────────────┘

User Problem
      │
      ▼
┌──────────────────┐
│ ProblemAnalyzer  │───► Decomposition, complexity analysis
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CostEstimator    │───► Budget estimation, cost modeling
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ KnowledgeQuery   │───► Historical patterns, similar solutions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PlanBuilder      │───► EvolutionPlan with parameters, budget,
└────────┬─────────┘     phases, convergence criteria
         │
         ▼
┌──────────────────┐
│ ConfigTranslator │───► OpenEvolve EvolutionConfiguration
└──────────────────┘
```

### 3.2 Execution Phase Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXECUTION PHASE                          │
└─────────────────────────────────────────────────────────────────┘

EvolutionConfiguration
         │
         ▼
┌──────────────────┐
│ BudgetMonitor    │◄──── Tracks spending vs allocation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ OpenEvolveEngine │◄──── Core evolution (MAP-Elites/NSGA-II/Standard)
└────────┬─────────┘
         │
         │  ┌─────────────────────────────────────────────┐
         │  │  Iteration Callback                         │
         │  │  • Check budget                             │
         │  │  • Check convergence                        │
         │  │  • Adapt parameters                         │
         │  └─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ AdaptationEngine │◄──── Adjust parameters based on progress
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PhaseResults     │───► Fitness progression, budget consumed
└──────────────────┘
```

### 3.3 Summarization Phase Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      SUMMARIZATION PHASE                        │
└─────────────────────────────────────────────────────────────────┘

ExecutionResults
         │
         ▼
┌──────────────────┐
│ ResultAnalyzer   │───► Success/failure analysis
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PatternExtractor │───► Reusable patterns, strategies
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ InsightGenerator │───► Key insights, failure modes
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ KnowledgeStore   │───► Update knowledge graph
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ EvolutionSummary │───► Actionable recommendations
└──────────────────┘
```

---

## 4. Integration Points

### 4.1 Integration Point 1: Strategy Selection

**Location**: `StrategyOrchestrator.select_strategy()`

**Integration**: PES layer provides cost-aware strategy selection that overrides OpenEvolve's default mode selection.

```python
class StrategyOrchestrator:
    def select_strategy(self, problem: EvolutionProblem) -> StrategyType:
        # PES-enhanced selection
        if self.config.enable_pes_planning:
            return self.pes_planner.recommend_strategy(problem)
        
        # Fall back to OpenEvolve default
        return self._default_strategy_selection(problem)
```

### 4.2 Integration Point 2: Parameter Configuration

**Location**: `PESOpenEvolveAdapter.translate_plan_to_config()`

**Integration**: PES plan parameters flow into OpenEvolve's 272+ parameter system.

```python
def translate_plan_to_config(self, plan: EvolutionPlan) -> EvolutionConfiguration:
    config = EvolutionConfiguration()
    
    # PES-guided parameter setting
    config.max_iterations = plan.suggested_parameters.get('iterations', 100)
    config.population_size = plan.suggested_parameters.get('population', 50)
    config.mutation_rate = plan.suggested_parameters.get('mutation_rate', 0.1)
    
    # Cost controls from PES budget
    config.cost_limit_usd = plan.budget_allocation.evolution_budget.max_cost
    config.token_limit = plan.budget_allocation.evolution_budget.max_tokens
    
    # Mode selection
    config.evolution_mode = plan.recommended_mode.value
    
    return config
```

### 4.3 Integration Point 3: Evolution Callbacks

**Location**: OpenEvolve evolution loop → `PESCallbacks`

**Integration**: PES layer receives callbacks during evolution for monitoring and adaptation.

```python
class PESCallbacks:
    """Callbacks from OpenEvolve to PES layer."""
    
    def on_iteration_complete(
        self,
        iteration: int,
        population: List[Individual],
        best_fitness: float,
        budget_consumed: CostSummary
    ) -> Optional[ParameterAdjustment]:
        """Called after each evolution iteration."""
        
        # Check budget
        if not self.budget_monitor.can_continue(budget_consumed):
            return ParameterAdjustment(should_stop=True)
        
        # Check convergence
        if self._convergence_detected(best_fitness):
            return ParameterAdjustment(should_stop=True)
        
        # Adapt parameters
        if iteration % self.adaptation_interval == 0:
            return self.adaptation_engine.suggest_adjustment(
                population, best_fitness, iteration
            )
        
        return None
```

### 4.4 Integration Point 4: Mutation/Selection Override

**Location**: OpenEvolve variation operators → `DirectedEvolutionStrategy`

**Integration**: PES-guided operators replace or augment standard operators.

```python
class GuidedVariationOperators:
    """OpenEvolve variation operators with PES guidance."""
    
    def __init__(self, directed_strategy: DirectedEvolutionStrategy):
        self.directed = directed_strategy
        self.standard = StandardVariationOperators()
    
    def mutate(self, individual: Individual) -> Individual:
        # Use PES-guided mutation when available
        if self.directed and self.directed.has_guidance_for(individual):
            return self.directed.guided_mutation(individual, self.fitness_history)
        
        # Fall back to standard mutation
        return self.standard.mutate(individual)
```

### 4.5 Integration Point 5: Verification Coordination

**Location**: Lean 4 / Z3 verification → `PESVerificationPlanner`

**Integration**: PES layer plans verification strategy, OpenEvolve executes.

```python
class PESVerificationPlanner:
    """Plans formal verification as part of evolution."""
    
    def plan_verification(
        self,
        candidate_solutions: List[Individual],
        budget: CostBudget
    ) -> VerificationPlan:
        """Decide which candidates to verify and how."""
        
        # Rank by confidence
        ranked = self._rank_by_verification_potential(candidate_solutions)
        
        # Select within budget
        to_verify = []
        remaining_budget = budget
        
        for candidate in ranked:
            cost = self._estimate_verification_cost(candidate)
            if cost <= remaining_budget:
                to_verify.append(candidate)
                remaining_budget -= cost
        
        return VerificationPlan(candidates=to_verify, budget=remaining_budget)
```

---

## 5. Configuration System for Strategy Selection

### 5.1 Unified Configuration Schema

```python
@dataclass
class UnifiedEvolutionConfig:
    """
    Unified configuration combining OpenEvolve's 272+ parameters
    with PES-specific configuration.
    """
    
    # ============================================================
    # OpenEvolve Core Parameters (272 existing parameters)
    # ============================================================
    
    # Evolution mode (from OpenEvolve)
    evolution_mode: str = "standard"  # standard, qd, mo, adversarial
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    # ... (269 more parameters)
    
    # ============================================================
    # PES Enhancement Parameters (New)
    # ============================================================
    
    # Strategy selection
    strategy_selection_mode: StrategySelectionMode = StrategySelectionMode.AUTO
    """How to select strategy: AUTO, MANUAL, or PES_ONLY"""
    
    # PES planning
    enable_pes_planning: bool = True
    """Enable PES planning phase"""
    
    planning_depth: int = 3
    """How many planning iterations to perform"""
    
    use_historical_patterns: bool = True
    """Query knowledge base for similar problems"""
    
    # Cost optimization
    enable_cost_optimization: bool = True
    """Enable budget-aware execution"""
    
    cost_model: CostModel = CostModel.TOKEN_BASED
    """Cost model: TOKEN_BASED, TIME_BASED, or HYBRID"""
    
    # Budget constraints
    max_cost_usd: Optional[float] = None
    """Maximum total cost in USD"""
    
    max_tokens: Optional[int] = None
    """Maximum LLM tokens"""
    
    max_api_calls: Optional[int] = None
    """Maximum API calls"""
    
    max_duration_seconds: Optional[float] = None
    """Maximum execution time"""
    
    # Budget allocation
    planning_budget_ratio: float = 0.05
    """Ratio of budget allocated to planning (0.0 - 1.0)"""
    
    evolution_budget_ratio: float = 0.85
    """Ratio of budget allocated to evolution (0.0 - 1.0)"""
    
    verification_budget_ratio: float = 0.10
    """Ratio of budget allocated to verification (0.0 - 1.0)"""
    
    contingency_reserve_ratio: float = 0.10
    """Reserve for unexpected costs (0.0 - 1.0)"""
    
    # Adaptive execution
    enable_adaptive_execution: bool = True
    """Enable parameter adaptation during evolution"""
    
    adaptation_interval: int = 10
    """Iterations between adaptation checks"""
    
    adaptation_trigger: AdaptationTrigger = AdaptationTrigger.PLATEAU
    """What triggers adaptation: PLATEAU, BUDGET, or SCHEDULED"""
    
    # Early stopping
    pes_early_stopping: bool = True
    """Enable PES-guided early stopping"""
    
    convergence_sensitivity: float = 0.01
    """Fitness change threshold for convergence"""
    
    min_improvement_window: int = 5
    """Iterations to wait for improvement before stopping"""
    
    # Knowledge integration
    knowledge_integration_mode: KnowledgeMode = KnowledgeMode.ACTIVE
    """How to use knowledge: ACTIVE, PASSIVE, or DISABLED"""
    
    knowledge_similarity_threshold: float = 0.7
    """Minimum similarity to use historical patterns"""
    
    # Directed search
    enable_directed_mutation: bool = True
    """Use PES guidance for mutations"""
    
    directed_mutation_weight: float = 0.7
    """Weight of directed vs random mutation (0.0 - 1.0)"""
    
    # Summarization
    enable_summarization: bool = True
    """Enable PES summarization phase"""
    
    extract_patterns: bool = True
    """Extract reusable patterns from results"""
    
    update_knowledge_base: bool = True
    """Update knowledge base with results"""
```

### 5.2 Strategy Selection Configuration

```python
@dataclass
class StrategySelectionRules:
    """Rules for automatic strategy selection."""
    
    # Complexity thresholds
    simple_complexity_threshold: float = 0.3
    complex_complexity_threshold: float = 0.7
    
    # Budget thresholds
    tight_budget_threshold_usd: float = 1.0
    moderate_budget_threshold_usd: float = 5.0
    
    # Time thresholds
    urgent_time_seconds: float = 60.0
    relaxed_time_seconds: float = 600.0
    
    def select_strategy(
        self,
        complexity: float,
        budget: float,
        time_available: float,
        objectives: int
    ) -> StrategyType:
        """Select strategy based on problem characteristics."""
        
        # PES mode for tight budgets or high complexity
        if budget < self.tight_budget_threshold_usd or complexity > self.complex_complexity_threshold:
            return StrategyType.PES_ENHANCED
        
        # Multi-objective mode
        if objectives > 1:
            return StrategyType.MO_STANDARD
        
        # Quick evolution for simple problems
        if complexity < self.simple_complexity_threshold and time_available < self.urgent_time_seconds:
            return StrategyType.STANDARD
        
        # Default to PES for directed search
        return StrategyType.PES_ENHANCED
```

### 5.3 Configuration Loading

```python
class UnifiedConfigLoader:
    """Loads unified configuration from multiple sources."""
    
    def load(self, sources: ConfigSources) -> UnifiedEvolutionConfig:
        config = UnifiedEvolutionConfig()
        
        # 1. Load OpenEvolve parameters from parameter_manager
        openevolve_params = self._load_openevolve_params(sources.parameter_manager)
        self._apply_openevolve_params(config, openevolve_params)
        
        # 2. Load PES configuration from YAML/JSON
        pes_config = self._load_pes_config(sources.config_file)
        self._apply_pes_config(config, pes_config)
        
        # 3. Override from environment variables
        env_overrides = self._load_env_overrides()
        self._apply_env_overrides(config, env_overrides)
        
        # 4. Validate unified configuration
        self._validate_config(config)
        
        return config
```

---

## 6. Cost Optimization Flow

### 6.1 Cost Model

```python
@dataclass
class CostModel:
    """
    Comprehensive cost model for evolution execution.
    
    Costs include:
    - LLM tokens (input/output)
    - API calls
    - Verification operations
    - Compute time
    """
    
    # Token costs (per 1K tokens)
    input_token_cost: float = 0.01  # $0.01 per 1K input tokens
    output_token_cost: float = 0.03  # $0.03 per 1K output tokens
    
    # API costs
    api_call_overhead: float = 0.001  # Base cost per API call
    
    # Verification costs
    lean_verification_cost: float = 0.05  # Per verification attempt
    z3_verification_cost: float = 0.01   # Per verification attempt
    
    # Time costs (optional)
    compute_cost_per_hour: float = 0.50  # Cloud compute cost
    
    def estimate_evolution_cost(
        self,
        iterations: int,
        population_size: int,
        avg_tokens_per_eval: int,
        verification_rate: float
    ) -> CostEstimate:
        """Estimate total evolution cost."""
        
        # Token costs
        total_evaluations = iterations * population_size
        input_tokens = total_evaluations * avg_tokens_per_eval * 0.7  # 70% input
        output_tokens = total_evaluations * avg_tokens_per_eval * 0.3  # 30% output
        
        token_cost = (
            (input_tokens / 1000) * self.input_token_cost +
            (output_tokens / 1000) * self.output_token_cost
        )
        
        # API costs
        api_cost = total_evaluations * self.api_call_overhead
        
        # Verification costs
        verifications = int(total_evaluations * verification_rate)
        verification_cost = verifications * self.lean_verification_cost
        
        return CostEstimate(
            token_cost=token_cost,
            api_cost=api_cost,
            verification_cost=verification_cost,
            total_cost=token_cost + api_cost + verification_cost
        )
```

### 6.2 Budget Allocation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     BUDGET ALLOCATION FLOW                      │
└─────────────────────────────────────────────────────────────────┘

Total Budget (USD/Tokens/Time)
              │
              ▼
    ┌─────────────────┐
    │ Reserve Check   │───► Contingency Reserve (10%)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Planning Budget │───► 5% ──► Problem Analysis
    └────────┬────────┘      ──► Strategy Selection
             │                ──► Cost Estimation
             ▼
    ┌─────────────────┐
    │Evolution Budget │───► 85% ──► Core Evolution
    └────────┬────────┘      ──► MAP-Elites/NSGA-II
             │                ──► Language-Agnostic Evolution
             ▼
    ┌─────────────────┐
    │ Verification    │───► 10% ──► Lean 4 Proofs
    │ Budget          │      ──► Z3 SMT Checking
    └─────────────────┘      ──► Quality Gates
```

### 6.3 Budget Monitoring

```python
class BudgetMonitor:
    """
    Monitors budget consumption during evolution.
    
    Provides real-time tracking and triggers
    when budgets are exhausted.
    """
    
    def __init__(self, allocation: BudgetAllocation):
        self.allocation = allocation
        self.consumed = CostSummary()
        self.start_time = time.time()
    
    def record_spending(self, cost: CostBreakdown):
        """Record a cost occurrence."""
        self.consumed.add(cost)
        
        # Check thresholds
        if self._is_critical_threshold_reached():
            self._trigger_critical_alert()
        elif self._is_warning_threshold_reached():
            self._trigger_warning_alert()
    
    def can_continue(self) -> bool:
        """Check if evolution can continue within budget."""
        
        # Check cost budget
        if self.consumed.total_cost >= self.allocation.total_budget:
            return False
        
        # Check token budget
        if self.consumed.tokens >= self.allocation.token_budget:
            return False
        
        # Check time budget
        elapsed = time.time() - self.start_time
        if elapsed >= self.allocation.time_budget_seconds:
            return False
        
        # Check API budget
        if self.consumed.api_calls >= self.allocation.api_call_budget:
            return False
        
        return True
    
    def get_remaining_budget(self) -> BudgetAllocation:
        """Get remaining budget for each category."""
        return BudgetAllocation(
            total_budget=self.allocation.total_budget - self.consumed.total_cost,
            token_budget=self.allocation.token_budget - self.consumed.tokens,
            time_budget_seconds=self.allocation.time_budget_seconds - (time.time() - self.start_time),
            api_call_budget=self.allocation.api_call_budget - self.consumed.api_calls
        )
    
    def suggest_optimization(self) -> List[OptimizationSuggestion]:
        """Suggest optimizations based on spending patterns."""
        suggestions = []
        
        # If burning through tokens too fast
        token_rate = self.consumed.tokens / (time.time() - self.start_time)
        projected_tokens = token_rate * self.allocation.time_budget_seconds
        
        if projected_tokens > self.allocation.token_budget * 0.9:
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.REDUCE_POPULATION,
                description="Reduce population size by 20%",
                estimated_savings=self._calculate_population_savings(0.2)
            ))
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.EARLY_STOPPING,
                description="Enable aggressive early stopping",
                estimated_savings=self._calculate_early_stop_savings()
            ))
        
        return suggestions
```

### 6.4 Cost-Aware Adaptation

```python
class CostAwareAdaptationEngine:
    """
    Adapts evolution parameters based on cost constraints.
    
    When budget is tight, reduces:
    - Population size
    - Number of iterations
    - Verification frequency
    
    When budget is ample, increases:
    - Exploration (diversity maintenance)
    - Verification thoroughness
    """
    
    def adapt_for_budget(
        self,
        config: EvolutionConfiguration,
        remaining_budget: BudgetAllocation,
        progress: EvolutionProgress
    ) -> EvolutionConfiguration:
        """Adapt configuration based on remaining budget."""
        
        adapted = copy.deepcopy(config)
        budget_ratio = remaining_budget.total_budget / self.initial_budget.total_budget
        
        if budget_ratio < 0.2:
            # Critical: aggressive reduction
            adapted.population_size = int(config.population_size * 0.5)
            adapted.max_iterations = int(config.max_iterations * 0.6)
            adapted.cascade_evaluation = True  # Use cascade to save evaluations
            
        elif budget_ratio < 0.5:
            # Warning: moderate reduction
            adapted.population_size = int(config.population_size * 0.7)
            adapted.max_iterations = int(config.max_iterations * 0.8)
            
        elif budget_ratio > 0.8 and progress.improvement_rate < 0.1:
            # Plenty of budget but stagnating: increase exploration
            adapted.mutation_rate = min(config.mutation_rate * 1.3, 0.5)
            adapted.diversity_maintenance = True
            adapted.exploration_bonus = 0.2
        
        return adapted
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. Create `UnifiedEvolutionConfig` dataclass
2. Implement `StrategyOrchestrator` shell
3. Create `PESOpenEvolveAdapter` basic structure
4. Add PES parameters to existing configuration

### Phase 2: PES Core (Week 3-4)
1. Implement `PESPlanner` with problem analysis
2. Build `CostEstimator` with token/cost models
3. Create `BudgetMonitor` for tracking
4. Implement basic `StrategySelector`

### Phase 3: Integration (Week 5-6)
1. Connect PES callbacks to OpenEvolve engine
2. Implement `DirectedEvolutionStrategy`
3. Create adaptation engine
4. Add configuration loading/unification

### Phase 4: Advanced Features (Week 7-8)
1. Knowledge integration for historical patterns
2. Advanced cost optimization
3. Result summarization
4. Verification planning

### Phase 5: Testing & Optimization (Week 9-10)
1. Integration testing
2. Cost model calibration
3. Performance benchmarking
4. Documentation

---

## 8. Key Design Decisions

### Decision 1: PES as Enhancement Layer
**Decision**: PES operates as an enhancement layer on top of OpenEvolve, not a replacement.

**Rationale**:
- Preserves all OpenEvolve capabilities
- Allows gradual adoption
- Maintains backward compatibility
- Enables hybrid strategies

### Decision 2: Budget-First Resource Management
**Decision**: All resource decisions are made through the lens of budget constraints.

**Rationale**:
- Cost-aware AI is essential for production
- Enables predictable spending
- Drives optimization
- Aligns with business constraints

### Decision 3: Strategy Selection at Runtime
**Decision**: Strategy (PES/QD/MO/Standard) is selected at runtime based on problem characteristics.

**Rationale**:
- No single strategy is best for all problems
- Enables automatic optimization
- Adapts to available budget
- Maximizes success probability

### Decision 4: Unified Configuration
**Decision**: Single configuration system combining OpenEvolve + PES parameters.

**Rationale**:
- Simplifies user experience
- Enables parameter interactions
- Supports validation
- Maintains consistency

---

## 9. Success Metrics

### Cost Efficiency
- **30% reduction** in average evolution cost for equivalent quality
- **90% of runs** stay within budget
- **Predictable costs** with <20% variance from estimates

### Quality Improvements
- **Directed search** finds solutions 40% faster
- **Higher success rate** on complex problems
- **Better convergence** with fewer iterations

### System Integration
- **Zero breaking changes** to existing OpenEvolve API
- **<5% overhead** for PES layer
- **Seamless fallback** to standard mode

---

## 10. Conclusion

This integration architecture provides a powerful combination:

- **OpenEvolve's robustness**: Battle-tested evolution algorithms, verification, multi-language support
- **LoongFlow's intelligence**: Cost-aware planning, directed search, adaptive execution

The result is an evolution system that:
1. Delivers higher quality solutions
2. Uses resources more efficiently
3. Adapts to problem characteristics
4. Provides predictable costs
5. Maintains all existing capabilities

**Next Steps**: Begin Phase 1 implementation with `UnifiedEvolutionConfig` and `StrategyOrchestrator`.
