# Adaptive MDAP API Reference

## Core Types

### SubProblem

```python
@dataclass
class SubProblem:
    id: str                          # Unique identifier
    description: str                 # Problem description
    domain: str                      # Problem domain
    depth: int                       # Depth in decomposition tree
    dependencies: List[str]          # Dependency IDs
    metadata: Dict[str, Any]         # Additional metadata
```

### ComplexityScore

```python
@dataclass
class ComplexityScore:
    overall_score: float            # [0.0, 1.0] weighted combination
    text_length_score: float        # [0.0, 1.0]
    domain_rarity_score: float      # [0.0, 1.0]
    depth_score: float              # [0.0, 1.0]
    historical_error_score: float   # [0.0, 1.0]
    dependency_score: float         # [0.0, 1.0]
    feature_weights: Dict[str, float]
```

### SolveStrategy

```python
class SolveStrategy(Enum):
    DIRECT = "direct"           # 1 agent, k=0
    MDAP_LIGHT = "mdap_light"   # 3 agents, k=1
    MAKER_FULL = "maker_full"   # 5+ agents, k=2+
```

### SolveConfig

```python
@dataclass
class SolveConfig:
    strategy: SolveStrategy
    n_agents: int
    k_ahead: int
    max_retries: int
    timeout_ms: Optional[int]
```

## Classifiers

### TaskComplexityClassifier

```python
class TaskComplexityClassifier:
    def __init__(self, config: Optional[ClassifierConfig] = None)
    
    def compute_complexity(self, subproblem: SubProblem) -> ComplexityScore
    # Computes weighted complexity score from multiple features
    
    def compute_text_length_feature(self, subproblem: SubProblem) -> float
    # Normalizes description length to [0, 1]
    
    def compute_domain_rarity_feature(self, subproblem: SubProblem) -> float
    # Uses embeddings to compute domain rarity
    
    def compute_depth_feature(self, subproblem: SubProblem) -> float
    # Normalizes depth to [0, 1]
    
    def compute_historical_error_feature(self, subproblem: SubProblem) -> float
    # Returns historical error rate for domain
    
    def compute_dependency_feature(self, subproblem: SubProblem) -> float
    # Normalizes dependency count to [0, 1]
    
    def update_historical_stats(self, domain: str, success: bool, complexity: float)
    # Updates historical statistics for a domain
    
    def get_cache_stats(self) -> Dict[str, Any]
    # Returns cache statistics
```

## Allocators

### AdaptiveMDAPAllocator

```python
class AdaptiveMDAPAllocator:
    def __init__(
        self,
        thresholds: Optional[List[float]] = None,  # [low, high]
        strategy_configs: Optional[Dict[SolveStrategy, SolveConfig]] = None,
        enable_learning: bool = False,
        enable_context_aware: bool = False,
    )
    
    def allocate_resources(
        self,
        complexity_score: float,
        context: Optional[AllocationContext] = None,
    ) -> SolveConfig
    # Allocates resources based on complexity
    
    def update_thresholds(
        self,
        thresholds: List[float],
        reason: str = "manual",
        reset_stats: bool = False,
    )
    # Updates allocation thresholds
    
    def get_allocation_stats(self) -> Dict[str, Any]
    # Returns allocation statistics
    
    def reset_stats(self)
    # Resets allocation statistics
    
    def record_outcome(
        self,
        complexity_score: float,
        strategy: SolveStrategy,
        success: bool,
        cost: float,
        quality: float,
    )
    # Records outcome for learning
    
    def allocate_resources_batch(
        self,
        complexity_scores: List[float],
        context: Optional[AllocationContext] = None,
    ) -> List[SolveConfig]
    # Batch allocation for efficiency
```

### AllocationContext

```python
@dataclass
class AllocationContext:
    time_of_day: Optional[str] = None        # "business_hours" or "off_hours"
    system_load: Optional[str] = None        # "high", "medium", "low"
    budget_remaining: Optional[float] = None # Percentage remaining
    quality_requirements: Optional[str] = None # "strict", "normal", "lenient"
```

## Controllers

### AdaptiveExecutionController

```python
class AdaptiveExecutionController:
    def __init__(
        self,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        crewai_integration: Optional[CrewAIIntegration] = None,
        solver_factory: Optional[Callable[[SolveConfig], Any]] = None,
    )
    
    def execute_adaptive(
        self,
        subproblem: SubProblem,
        workflow_epic_id: Optional[str] = None,
        context: Optional[AllocationContext] = None,
        force_strategy: Optional[SolveStrategy] = None,
    ) -> SolutionAttempt
    # Main execution method
    
    def get_execution_stats(self) -> Dict[str, Any]
    # Returns execution statistics
    
    def get_attempt(self, attempt_id: str) -> Optional[SolutionAttempt]
    # Gets specific attempt by ID
```

## Tools

### CostCalculator

```python
class CostCalculator:
    def __init__(
        self,
        pricing: Optional[APIPricing] = None,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    )
    
    def calculate_single_call_cost(self, token_usage: Optional[TokenUsage] = None) -> float
    # Cost for single API call
    
    def calculate_strategy_cost(
        self,
        strategy: SolveStrategy,
        num_problems: int = 1,
    ) -> StrategyCost
    # Cost for a strategy
    
    def calculate_baseline_cost(
        self,
        num_problems: int,
        baseline_strategy: SolveStrategy = SolveStrategy.MAKER_FULL,
    ) -> float
    # Baseline cost (all same strategy)
    
    def calculate_adaptive_cost(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]
    # Cost with adaptive allocation
    
    def calculate_savings(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, float]
    # Savings from adaptive allocation
    
    def generate_report(
        self,
        num_problems: int,
        num_days: int = 30,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]
    # Comprehensive cost report
    
    def compare_models(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]
    # Compare costs across models
```

## Utilities

### Logger

```python
def get_logger(name: str) -> logging.Logger
# Get logger for module

def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[str] = None
)
# Set up logging configuration

def set_correlation_id(correlation_id: str)
# Set correlation ID for current thread

def get_correlation_id() -> Optional[str]
# Get correlation ID for current thread
```

### Cache

```python
class EmbeddingCache:
    def get_embedding(self, text: str) -> Optional[List[float]]
    def set_embedding(self, text: str, embedding: List[float])
    def get_stats(self) -> Dict[str, Any]

class FeatureCache:
    def get_features(self, subproblem_id: str) -> Optional[Dict[str, float]]
    def set_features(self, subproblem_id: str, features: Dict[str, float])
    def get_complexity(self, subproblem_id: str) -> Optional[float]
    def set_complexity(self, subproblem_id: str, score: float)
```

### Metrics

```python
class MetricsCollector:
    def counter(self, name: str) -> Counter
    def histogram(self, name: str) -> Histogram
    def gauge(self, name: str) -> Gauge
    def timer(self, name: str) -> Timer
    
    def record_classification(self, duration_ms: float, success: bool)
    def record_allocation(self, strategy: str, complexity_score: float, duration_ms: float)
    def record_execution(self, strategy: str, success: bool, duration_ms: float, cost: float)
    
    def get_all_metrics(self) -> Dict[str, Any]
    def export_prometheus(self) -> str
    def reset()

def get_metrics() -> MetricsCollector
# Get global metrics collector
```

## Configuration

### ConfigLoader

```python
class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None)
    def load(self) -> AdaptiveMDAPConfig
    def reload(self) -> AdaptiveMDAPConfig
```

### AdaptiveMDAPConfig

```python
@dataclass
class AdaptiveMDAPConfig:
    classifier: ClassifierConfig
    allocator: AllocatorConfig
    strategies: StrategyConfig
    monitoring: MonitoringConfig
```

## Error Handling

### Exception Hierarchy

```python
AdaptiveMDAPError(Exception)
├── ClassificationError
├── AllocationError
├── ConfigurationError
├── CacheError
└── ExecutionError
```

Each error includes:
- `message`: Error description
- `details`: Additional context dictionary
