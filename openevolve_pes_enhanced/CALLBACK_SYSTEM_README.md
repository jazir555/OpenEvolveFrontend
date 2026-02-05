# Evolution Callback System

The Evolution Callback System provides a mechanism to monitor and intervene during evolution iterations in the PES Enhanced system. This solves the problem of `AgnosticPESEngine.evolve()` running as a black box with no visibility or control between iterations.

## Problem Statement

The original code could not monitor or intervene during evolution:

```python
async def _run_with_existing_pes(self, code, tests, language, max_iterations, **kwargs):
    engine = AgnosticPESEngine(max_iterations=max_iterations, **kwargs)
    result = await engine.evolve(code, tests, language or "general")  # Black box!
    return result
```

## Solution

The callback system provides:

1. **Pre/Post iteration hooks** - Monitor each iteration
2. **Budget enforcement** - Stop when budget exceeded
3. **Convergence detection** - Stop when solution converged
4. **Custom metrics collection** - Track any metric you want

## Quick Start

```python
from openevolve_pes_enhanced import (
    MonitoredAgnosticPES,
    BudgetAwareCallback,
    MonitoringCallback,
)

# Create callbacks
callbacks = [
    BudgetAwareCallback(max_cost_usd=5.0),
    MonitoringCallback(patience=3),
]

# Create monitored engine
engine = MonitoredAgnosticPES(
    max_iterations=10,
    callbacks=callbacks
)

# Run evolution with monitoring
result = await engine.evolve(code, tests, language)

# Check if stopped early
if result.stopped_early:
    print(f"Stopped: {result.stop_reason}")
```

## Core Components

### 1. EvolutionCallback (Abstract Base Class)

All callbacks extend this class:

```python
from openevolve_pes_enhanced import EvolutionCallback

class MyCallback(EvolutionCallback):
    async def on_iteration_start(self, iteration: int, context: EvolutionContext):
        pass
    
    async def on_iteration_end(self, iteration: int, metrics: IterationMetrics, context: EvolutionContext):
        pass
    
    async def should_stop(self, context: EvolutionContext, metrics: IterationMetrics) -> Tuple[bool, str]:
        return False, ""  # (should_stop, reason)
```

### 2. Built-in Callbacks

#### BudgetAwareCallback

Stops evolution when budget is exceeded:

```python
from openevolve_pes_enhanced import BudgetAwareCallback

callback = BudgetAwareCallback(
    max_cost_usd=10.0,          # Maximum cost in USD
    max_tokens=100000,          # Maximum tokens (optional)
    max_time_seconds=300        # Maximum time (optional)
)
```

#### MonitoringCallback

Stops evolution when convergence detected:

```python
from openevolve_pes_enhanced import MonitoringCallback

callback = MonitoringCallback(
    patience=3,                   # Iterations without improvement before stopping
    min_improvement=0.01,         # Minimum improvement to reset patience
    convergence_threshold=0.95    # Fitness threshold for convergence
)
```

#### LoggingCallback

Logs evolution progress:

```python
from openevolve_pes_enhanced import LoggingCallback

callback = LoggingCallback(
    log_level=logging.INFO,
    log_every_n_iterations=1
)
```

#### CompositeCallback

Combines multiple callbacks:

```python
from openevolve_pes_enhanced import CompositeCallback

composite = CompositeCallback([
    BudgetAwareCallback(max_cost_usd=5.0),
    MonitoringCallback(patience=3),
    LoggingCallback(),
])
```

### 3. MonitoredAgnosticPES

The wrapper engine that injects callbacks:

```python
from openevolve_pes_enhanced import MonitoredAgnosticPES

engine = MonitoredAgnosticPES(
    max_iterations=10,
    callbacks=[callback1, callback2],
    estimate_cost_per_iteration=lambda: 0.001  # Optional cost estimator
)

result = await engine.evolve(code, tests, language)
```

Returns `MonitoredEvolutionResult` with additional fields:

- `stopped_early` - Whether evolution stopped before max iterations
- `stop_reason` - Why evolution stopped
- `actual_iterations` - Number of iterations actually run
- `metrics_history` - List of IterationMetrics for each iteration
- `total_cost_usd` - Total estimated cost

## Data Structures

### IterationMetrics

Metrics collected at each iteration:

```python
@dataclass
class IterationMetrics:
    iteration: int
    total_iterations: int
    
    # Fitness metrics
    best_fitness: float
    avg_fitness: float
    
    # Test results
    tests_passed: int
    tests_total: int
    failing_tests: List[str]
    
    # Cost metrics
    cost_this_iteration: float
    total_cost: float
    tokens_used: int
    
    # Code evolution
    fixes_applied_this_iteration: List[str]
    
    # Timing
    iteration_duration_ms: int
    total_duration_ms: int
```

### EvolutionContext

Context passed to callbacks:

```python
@dataclass
class EvolutionContext:
    state: EvolutionState  # INITIALIZED, RUNNING, COMPLETED, etc.
    current_iteration: int
    max_iterations: int
    iteration_history: List[IterationMetrics]
    stop_requested: bool
    stop_reason: Optional[str]
    problem_type: str
    language: str
```

## Integration with PESIntegrationWrapper

The `PESIntegrationWrapper` now automatically uses callbacks when configured:

```python
from openevolve_pes_enhanced import PESIntegrationWrapper, PESEnhancedConfig

config = PESEnhancedConfig.cost_aware(max_cost_usd=5.0)
wrapper = PESIntegrationWrapper(config)

# This will automatically use callbacks for budget enforcement
result = await wrapper.enhance_with_planning(
    code=code,
    problem_description="Optimize sorting",
    tests=tests
)
```

Or pass explicit callbacks:

```python
from openevolve_pes_enhanced import (
    PESIntegrationWrapper,
    BudgetAwareCallback,
    MonitoringCallback,
)

wrapper = PESIntegrationWrapper()
callbacks = [
    BudgetAwareCallback(max_cost_usd=10.0),
    MonitoringCallback(patience=3),
]

result = await wrapper._run_with_existing_pes(
    code=code,
    tests=tests,
    language="python",
    max_iterations=10,
    callbacks=callbacks
)
```

## Creating Custom Callbacks

Example: Slack notification callback

```python
from openevolve_pes_enhanced import EvolutionCallback, EvolutionContext, IterationMetrics

class SlackNotificationCallback(EvolutionCallback):
    def __init__(self, webhook_url: str, notify_on_convergence: bool = True):
        super().__init__("SlackNotifier")
        self.webhook_url = webhook_url
        self.notify_on_convergence = notify_on_convergence
    
    async def on_evolution_start(self, context, initial_code, tests):
        await self._send_message(f"🚀 Evolution started: {len(tests)} tests")
    
    async def on_iteration_end(self, iteration, metrics, context):
        if metrics.best_fitness >= 1.0:
            await self._send_message(f"✅ All tests passing at iteration {iteration}!")
    
    async def on_evolution_end(self, context, final_metrics, result):
        await self._send_message(
            f"Evolution complete: {final_metrics.best_fitness:.1%} fitness"
        )
    
    async def _send_message(self, text: str):
        # Send to Slack webhook
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json={"text": text})
```

Example: Database logging callback

```python
class DatabaseLoggingCallback(EvolutionCallback):
    def __init__(self, db_connection):
        super().__init__("DBLogger")
        self.db = db_connection
        self.run_id = None
    
    async def on_evolution_start(self, context, initial_code, tests):
        # Create run record
        self.run_id = await self.db.execute(
            "INSERT INTO evolution_runs (start_time, max_iterations) VALUES (NOW(), %s) RETURNING id",
            (context.max_iterations,)
        )
    
    async def on_iteration_end(self, iteration, metrics, context):
        # Log iteration metrics
        await self.db.execute(
            """INSERT INTO evolution_iterations 
               (run_id, iteration, fitness, tests_passed, cost) 
               VALUES (%s, %s, %s, %s, %s)""",
            (self.run_id, iteration, metrics.best_fitness, 
             metrics.tests_passed, metrics.cost_this_iteration)
        )
```

## Factory Functions

Convenience functions for creating common callback configurations:

```python
from openevolve_pes_enhanced import (
    create_budget_callback,
    create_monitoring_callback,
    create_logging_callback,
    create_standard_callbacks,
    create_monitored_engine,
)

# Budget-only
callback = create_budget_callback(max_cost_usd=5.0)

# Monitoring-only  
callback = create_monitoring_callback(patience=3)

# Logging-only
callback = create_logging_callback(log_every_n_iterations=2)

# Standard set (budget + monitoring + logging)
composite = create_standard_callbacks(
    max_cost_usd=10.0,
    patience=3,
    enable_logging=True
)

# Monitored engine with standard callbacks
engine = create_monitored_engine(
    max_iterations=10,
    max_cost_usd=5.0,
    patience=3
)
```

## Running the Demo

```bash
cd c:\Users\mmeadow\Documents\OpenEvolve\Frontend
python -m openevolve_pes_enhanced.demo_callbacks
```

This demonstrates:
1. Basic callbacks with budget and monitoring
2. Budget enforcement stopping evolution
3. Custom progress reporter callback
4. Composite callbacks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PESIntegrationWrapper                      │
│                     (enhanced)                              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              _run_with_existing_pes()                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ If callbacks provided:                              │   │
│  │   Use MonitoredAgnosticPES                          │   │
│  │   (true iteration-level hooks)                      │   │
│  │ Else:                                               │   │
│  │   Use legacy budget enforcement or original engine  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              MonitoredAgnosticPES                           │
│                                                             │
│  for iteration in range(max_iterations):                   │
│      1. Call on_iteration_start() callbacks                 │
│      2. Run evolution iteration                             │
│      3. Collect metrics                                     │
│      4. Call on_iteration_end() callbacks                   │
│      5. Check should_stop() on all callbacks                │
│      6. Stop if any callback requests stop                  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Callbacks                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │BudgetAware   │  │Monitoring    │  │Logging       │      │
│  │Callback      │  │Callback      │  │Callback      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │Custom        │  │Composite     │                        │
│  │Callbacks     │  │Callback      │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

1. **Fine-grained control** - Stop evolution at any iteration
2. **Budget safety** - Never exceed cost limits
3. **Convergence detection** - Stop when solution is good enough
4. **Observability** - Monitor progress in real-time
5. **Extensibility** - Easy to add custom callbacks
6. **Backward compatible** - Falls back to original engine if no callbacks

## Files Added

- `evolution_callbacks.py` - Callback base classes and built-in implementations
- `monitored_engine.py` - MonitoredAgnosticPES wrapper engine
- `demo_callbacks.py` - Demonstration and examples
- `CALLBACK_SYSTEM_README.md` - This documentation

## Files Modified

- `integration_wrapper.py` - Updated to use callback system
- `__init__.py` - Exported new classes
