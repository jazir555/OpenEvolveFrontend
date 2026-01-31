# Fallback Mechanism Documentation

## Overview

The fallback mechanism provides graceful degradation when LoongFlow is unavailable or fails. It ensures the system continues to function with OpenEvolve-only mode, maintaining productivity and reliability.

## Architecture

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Evolution API                    │
│                  (openevolve/unified/api.py)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Selector                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Check enable_loongflow                            │  │
│  │  2. Check LOONGFLOW_AVAILABLE                         │  │
│  │  3. Check require_loongflow                           │  │
│  │  4. Check loongflow_fallback_enabled                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│   Use LoongFlow      │        │   Use OpenEvolve     │
│   - PES mode         │        │   - QD mode          │
│   - Directed search  │        │   - MO mode          │
│   - Planning         │        │   - Adversarial      │
│   - Memory           │        │   - Standard         │
└──────────────────────┘        └──────────────────────┘
```

## Decision Flow

### Complete Decision Tree

```
START
  │
  ▼
Is enable_loongflow == True?
  │
  ├─ NO → USE_OPENEVOLVE_ONLY
  │         │
  │         └─ Select OpenEvolve mode
  │           - QD (Quality-Diversity)
  │           - MO (Multi-Objective)
  │           - ADVERSARIAL (Robustness)
  │           - STANDARD (Basic GA)
  │
  └─ YES → Check LOONGFLOW_AVAILABLE
           │
           ├─ AVAILABLE → USE_LOONGFLOW
           │               │
           │               └─ Select PES mode
           │                 - Planning phase
           │                 - Memory system
           │                 - Directed search
           │
           └─ NOT_AVAILABLE → Check require_loongflow
                             │
                             ├─ TRUE → RAISE_ERROR
                             │            │
                             │            └─ LoongFlowRequiredError
                             │              "LoongFlow is required but not available"
                             │
                             └─ FALSE → Check loongflow_fallback_enabled
                                         │
                                         ├─ TRUE → USE_OPENEVOLVE_FALLBACK
                                         │            │
                                         │            ├─ Log warning
                                         │            └─ Select OpenEvolve mode
                                         │
                                         └─ FALSE → RAISE_ERROR
                                                       │
                                                       └─ LoongFlowUnavailableError
                                                         "LoongFlow unavailable and fallback disabled"
```

## Implementation Details

### 1. LoongFlow Availability Check

```python
# File: openevolve/unified/__init__.py

try:
    from loongflow import LoongFlowEvolve
    from loongflow.config import PESConfig
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False
    logger.warning("LoongFlow not available. Install with: pip install loongflow")
```

**Behavior:**
- Checked at import time
- Cached in module-level variable
- Fast check (no network calls)
- Safe to check multiple times

### 2. Strategy Selection with Fallback

```python
# File: openevolve/unified/unified_evolution_api.py

async def _select_strategy(
    self,
    problem: str,
    domain: str,
    config: UnifiedEvolutionConfig
) -> SystemMode:
    """
    Select optimal strategy with graceful fallback to OpenEvolve
    """

    # Case 1: LoongFlow explicitly disabled
    if not config.enable_loongflow:
        logger.info("LoongFlow disabled, using OpenEvolve-only mode")
        return await self._select_openevolve_strategy(
            problem, domain, config
        )

    # Case 2: Check LoongFlow availability
    if not LOONGFLOW_AVAILABLE:
        # Case 2a: Strict requirement
        if config.require_loongflow:
            raise LoongFlowRequiredError(
                "LoongFlow is required but not available. "
                "Install with: pip install loongflow"
            )

        # Case 2b: Graceful fallback
        if config.loongflow_fallback_enabled:
            logger.warning(
                "LoongFlow unavailable, falling back to OpenEvolve. "
                "For best performance, install LoongFlow: "
                "pip install loongflow"
            )
            return await self._select_openevolve_strategy(
                problem, domain, config
            )

        # Case 2c: Fallback disabled
        raise LoongFlowUnavailableError(
            "LoongFlow unavailable and fallback disabled. "
            "Install LoongFlow or set loongflow_fallback_enabled=True"
        )

    # Case 3: LoongFlow available - optimal strategy selection
    return await self._select_optimal_strategy(
        problem, domain, config
    )
```

### 3. OpenEvolve Strategy Selection

```python
async def _select_openevolve_strategy(
    self,
    problem: str,
    domain: str,
    config: UnifiedEvolutionConfig
) -> SystemMode:
    """
    Select OpenEvolve-only strategy (QD, MO, Adversarial, Standard)
    """

    # Get problem characteristics
    characteristics = await self._analyze_problem(problem, domain)

    # Scoring system
    scores = {
        EvolutionMode.QD: 0.0,
        EvolutionMode.MO: 0.0,
        EvolutionMode.ADVERSARIAL: 0.0,
        EvolutionMode.STANDARD: 0.0
    }

    # Factor 1: Multiple objectives (25 points)
    if characteristics.get('multiple_objectives', False):
        scores[EvolutionMode.MO] += 25

    # Factor 2: Diversity need (20 points)
    if characteristics.get('needs_diversity', False):
        scores[EvolutionMode.QD] += 20

    # Factor 3: Robustness need (15 points)
    if characteristics.get('needs_robustness', False):
        scores[EvolutionMode.ADVERSARIAL] += 15

    # Factor 4: Domain heuristics (20 points)
    domain_scores = self._get_domain_scores(domain, openevolve_only=True)
    for mode, score in domain_scores.items():
        scores[mode] += score

    # Factor 5: Configuration override (20 points)
    if config.qd.enabled:
        scores[EvolutionMode.QD] += 20
    if config.mo.enabled:
        scores[EvolutionMode.MO] += 20
    if config.adversarial.enabled:
        scores[EvolutionMode.ADVERSARIAL] += 20

    # Select best mode
    best_mode = max(scores, key=scores.get)

    # Map to system
    return SystemMode(
        system='openevolve',
        mode=best_mode.value,
        confidence=scores[best_mode] / 100.0,
        reasoning=f"Selected {best_mode.value} for OpenEvolve-only execution"
    )
```

### 4. Domain-Specific Scoring (OpenEvolve-only)

```python
def _get_domain_scores(
    self,
    domain: str,
    openevolve_only: bool = False
) -> Dict[EvolutionMode, float]:
    """
    Get domain-specific mode scores

    When openevolve_only=True, excludes PES from consideration
    """

    if domain == 'trading':
        # Trading: Adversarial for robustness
        return {
            EvolutionMode.ADVERSARIAL: 20,
            EvolutionMode.QD: 10,
            EvolutionMode.MO: 15,
            EvolutionMode.STANDARD: 5
        }

    elif domain == 'science':
        # Science: QD for diverse experiments
        return {
            EvolutionMode.QD: 20,
            EvolutionMode.MO: 15,
            EvolutionMode.ADVERSARIAL: 10,
            EvolutionMode.STANDARD: 5
        }

    elif domain == 'engineering':
        # Engineering: MO for competing objectives
        return {
            EvolutionMode.MO: 20,
            EvolutionMode.ADVERSARIAL: 15,
            EvolutionMode.QD: 10,
            EvolutionMode.STANDARD: 5
        }

    elif domain == 'pharma':
        # Pharma: QD for chemical space exploration
        return {
            EvolutionMode.QD: 20,
            EvolutionMode.MO: 15,
            EvolutionMode.STANDARD: 10,
            EvolutionMode.ADVERSARIAL: 5
        }

    elif domain == 'web':
        # Web: Standard is sufficient
        return {
            EvolutionMode.STANDARD: 20,
            EvolutionMode.QD: 10,
            EvolutionMode.MO: 5,
            EvolutionMode.ADVERSARIAL: 5
        }

    elif domain == 'finance':
        # Finance: MO for return/risk/liquidity
        return {
            EvolutionMode.MO: 20,
            EvolutionMode.QD: 15,
            EvolutionMode.STANDARD: 10,
            EvolutionMode.ADVERSARIAL: 5
        }

    else:  # general
        return {
            EvolutionMode.STANDARD: 15,
            EvolutionMode.QD: 15,
            EvolutionMode.MO: 10,
            EvolutionMode.ADVERSARIAL: 10
        }
```

### 5. Execution with Fallback

```python
async def _execute_evolution(
    self,
    problem: str,
    domain: str,
    strategy: SystemMode,
    config: UnifiedEvolutionConfig,
    callback: Optional[Callable]
) -> Tuple[str, float, Dict]:
    """
    Execute evolution with automatic fallback on failure
    """

    try:
        # Try LoongFlow if selected
        if strategy.system == 'loongflow' and LOONGFLOW_AVAILABLE:
            logger.info("Executing with LoongFlow PES")

            adapter = LoongFlowAdapter(config)
            solution, score, artifacts = await adapter.evolve(
                problem=problem,
                domain=domain
            )

            return solution, score, artifacts

        # Use OpenEvolve
        elif strategy.system == 'openevolve':
            logger.info(f"Executing with OpenEvolve {strategy.mode}")

            executor = OpenEvolveExecutor(config)
            solution, score, artifacts = await executor.evolve(
                problem=problem,
                domain=domain,
                mode=strategy.mode
            )

            return solution, score, artifacts

    except Exception as e:
        logger.error(f"Evolution failed: {e}")

        # Attempt fallback if configured
        if strategy.system == 'loongflow' and config.loongflow_fallback_enabled:
            logger.warning(f"LoongFlow execution failed, falling back to OpenEvolve: {e}")

            # Switch to OpenEvolve
            strategy = await self._select_openevolve_strategy(
                problem, domain, config
            )

            executor = OpenEvolveExecutor(config)
            solution, score, artifacts = await executor.evolve(
                problem=problem,
                domain=domain,
                mode=strategy.mode
            )

            return solution, score, artifacts

        # No fallback available, re-raise
        raise
```

## Error Handling

### Custom Exceptions

```python
# File: openevolve/unified/exceptions.py

class LoongFlowError(Exception):
    """Base exception for LoongFlow-related errors"""
    pass


class LoongFlowRequiredError(LoongFlowError):
    """
    Raised when LoongFlow is required but not available

    Example:
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True
        )
        # LoongFlow not installed → LoongFlowRequiredError
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.install_hint = "pip install loongflow"


class LoongFlowUnavailableError(LoongFlowError):
    """
    Raised when LoongFlow unavailable and fallback disabled

    Example:
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=False
        )
        # LoongFlow not installed → LoongFlowUnavailableError
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.fallback_hint = "Set loongflow_fallback_enabled=True"


class LoongFlowExecutionError(LoongFlowError):
    """
    Raised when LoongFlow execution fails and fallback disabled

    Example:
        # LoongFlow crashes during execution
        # loongflow_fallback_enabled=False → LoongFlowExecutionError
    """
    def __init__(self, message: str, original_error: Exception):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
```

### Error Recovery Strategies

#### Strategy 1: Silent Fallback (Default)

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Silent fallback
    require_loongflow=False
)

# Behavior:
# - If LoongFlow unavailable → Log warning, use OpenEvolve
# - If LoongFlow fails → Log error, use OpenEvolve
# - No exceptions raised
```

#### Strategy 2: Fail Fast (Strict)

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=False,  # No fallback
    require_loongflow=True  # Strict requirement
)

# Behavior:
# - If LoongFlow unavailable → Raise LoongFlowRequiredError
# - If LoongFlow fails → Raise LoongFlowExecutionError
# - Fail immediately, don't attempt fallback
```

#### Strategy 3: Hybrid (Partial Fallback)

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Allow fallback
    require_loongflow=False  # Not required
)

# Behavior:
# - If LoongFlow unavailable at start → Log warning, use OpenEvolve
# - If LoongFlow fails during execution → Log error, use OpenEvolve
# - Graceful degradation with warnings
```

## Logging and Monitoring

### Log Messages

```python
# LoongFlow disabled
logger.info("LoongFlow disabled, using OpenEvolve-only mode")

# LoongFlow available and enabled
logger.info("Executing with LoongFlow PES mode")

# LoongFlow unavailable, fallback enabled
logger.warning(
    "LoongFlow unavailable, falling back to OpenEvolve. "
    "For best performance, install LoongFlow: pip install loongflow"
)

# LoongFlow unavailable, fallback disabled
logger.error(
    "LoongFlow unavailable and fallback disabled. "
    "Install LoongFlow or set loongflow_fallback_enabled=True"
)

# LoongFlow execution failed, fallback enabled
logger.error(
    f"LoongFlow execution failed, falling back to OpenEvolve: {e}"
)

# Strategy selected
logger.info(
    f"Selected strategy: system={strategy.system}, "
    f"mode={strategy.mode}, confidence={strategy.confidence:.2%}"
)
```

### Metrics to Track

```python
# Execution metrics
metrics = {
    'system_used': result.strategy_used.system,  # 'loongflow' or 'openevolve'
    'mode_used': result.strategy_used.mode,  # 'pes', 'qd', 'mo', etc.
    'loongflow_available': LOONGFLOW_AVAILABLE,
    'fallback_triggered': False,  # Did fallback occur?
    'evaluations': result.evaluations,
    'execution_time': result.total_time,
    'final_score': result.final_score
}

# Monitor these metrics to:
# - Detect when LoongFlow is unavailable
# - Track fallback frequency
# - Compare performance (LoongFlow vs OpenEvolve)
# - Identify configuration issues
```

### Monitoring Example

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
loongflow_available = Gauge(
    'evolution_loongflow_available',
    'Whether LoongFlow is available'
)

fallback_triggered = Counter(
    'evolution_fallback_total',
    'Total number of fallbacks to OpenEvolve',
    ['reason']  # 'unavailable', 'execution_failed'
)

system_used = Counter(
    'evolution_system_used_total',
    'Total executions per system',
    ['system']  # 'loongflow', 'openevolve'
)

execution_time = Histogram(
    'evolution_execution_seconds',
    'Evolution execution time',
    ['system', 'mode']
)

evaluation_count = Histogram(
    'evolution_evaluations_total',
    'Number of evaluations per run',
    ['system', 'mode']
)

# Update metrics
loongflow_available.set(1 if LOONGFLOW_AVAILABLE else 0)

if fallback_occurred:
    fallback_triggered.labels(reason='unavailable').inc()

system_used.labels(system=result.strategy_used.system).inc()
execution_time.labels(
    system=result.strategy_used.system,
    mode=result.strategy_used.mode
).observe(result.total_time)
evaluation_count.labels(
    system=result.strategy_used.system,
    mode=result.strategy_used.mode
).observe(result.evaluations)
```

## Testing

### Unit Tests

```python
# File: tests/unified/test_fallback_mechanism.py

import pytest
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.exceptions import (
    LoongFlowRequiredError,
    LoongFlowUnavailableError
)

class TestFallbackMechanism:
    """Test graceful fallback behavior"""

    def test_openevolve_only_when_disabled(self):
        """Test OpenEvolve-only mode when LoongFlow disabled"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=False
        )

        result = await evolve(
            problem="Test problem",
            domain="general",
            config=config
        )

        assert result.strategy_used.system == 'openevolve'
        assert result.strategy_used.mode != 'pes'

    def test_fallback_when_unavailable(self, monkeypatch):
        """Test fallback when LoongFlow unavailable"""
        # Mock LoongFlow as unavailable
        monkeypatch.setattr(
            'openevolve.unified.LOONGFLOW_AVAILABLE',
            False
        )

        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True
        )

        result = await evolve(
            problem="Test problem",
            domain="general",
            config=config
        )

        assert result.strategy_used.system == 'openevolve'

    def test_error_when_required_and_unavailable(self, monkeypatch):
        """Test error when LoongFlow required but unavailable"""
        # Mock LoongFlow as unavailable
        monkeypatch.setattr(
            'openevolve.unified.LOONGFLOW_AVAILABLE',
            False
        )

        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            loongflow_fallback_enabled=False
        )

        with pytest.raises(LoongFlowRequiredError):
            await evolve(
                problem="Test problem",
                domain="general",
                config=config
            )

    def test_error_when_fallback_disabled(self, monkeypatch):
        """Test error when LoongFlow unavailable and fallback disabled"""
        # Mock LoongFlow as unavailable
        monkeypatch.setattr(
            'openevolve.unified.LOONGFLOW_AVAILABLE',
            False
        )

        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=False
        )

        with pytest.raises(LoongFlowUnavailableError):
            await evolve(
                problem="Test problem",
                domain="general",
                config=config
            )

    @pytest.mark.skipif(
        not LOONGFLOW_AVAILABLE,
        reason="LoongFlow not available"
    )
    def test_use_loongflow_when_available(self):
        """Test use LoongFlow when available"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True
        )

        result = await evolve(
            problem="Test problem",
            domain="science",
            config=config
        )

        assert result.strategy_used.system == 'loongflow'
        assert result.strategy_used.mode == 'pes'
```

### Integration Tests

```python
class TestFallbackIntegration:
    """Test fallback in realistic scenarios"""

    async def test_fallback_during_execution(self):
        """Test fallback when LoongFlow fails during execution"""
        # Mock LoongFlow to raise error during execution
        with patch('openevolve.unified.LoongFlowAdapter.evolve') as mock_evolve:
            mock_evolve.side_effect = Exception("LoongFlow crashed")

            config = UnifiedEvolutionConfig(
                enable_loongflow=True,
                loongflow_fallback_enabled=True
            )

            result = await evolve(
                problem="Test problem",
                domain="science",
                config=config
            )

            # Should fall back to OpenEvolve
            assert result.strategy_used.system == 'openevolve'
            assert result is not None

    async def test_no_fallback_on_success(self):
        """Test no fallback when LoongFlow succeeds"""
        if not LOONGFLOW_AVAILABLE:
            pytest.skip("LoongFlow not available")

        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True
        )

        result = await evolve(
            problem="Test problem",
            domain="science",
            config=config
        )

        # Should use LoongFlow, no fallback
        assert result.strategy_used.system == 'loongflow'
```

### Performance Tests

```python
class TestFallbackPerformance:
    """Test performance impact of fallback"""

    async def test_fallback_overhead(self):
        """Measure overhead of fallback check"""
        import time

        # With LoongFlow available
        start = time.time()
        await evolve(
            problem="Test problem",
            domain="general",
            use_loongflow=True
        )
        time_with_lf = time.time() - start

        # Without LoongFlow (fallback)
        start = time.time()
        await evolve(
            problem="Test problem",
            domain="general",
            use_loongflow=False
        )
        time_without_lf = time.time() - start

        # Fallback check should add < 1ms overhead
        assert abs(time_with_lf - time_without_lf) < 0.001

    async def test_fallback_latency(self):
        """Measure latency from LoongFlow failure to OpenEvolve start"""
        # Mock LoongFlow to fail immediately
        with patch('openevolve.unified.LoongFlowAdapter.evolve') as mock_evolve:
            mock_evolve.side_effect = Exception("Immediate failure")

            config = UnifiedEvolutionConfig(
                enable_loongflow=True,
                loongflow_fallback_enabled=True
            )

            import time
            start = time.time()

            result = await evolve(
                problem="Test problem",
                domain="science",
                config=config
            )

            fallback_latency = time.time() - start

            # Fallback should happen within 100ms
            assert fallback_latency < 0.1
            assert result.strategy_used.system == 'openevolve'
```

## Configuration Matrix

### All Combinations

| `enable_loongflow` | `require_loongflow` | `loongflow_fallback_enabled` | LoongFlow Available | Result |
|--------------------|---------------------|------------------------------|---------------------|--------|
| `false` | `false` | `true` | No | Use OpenEvolve |
| `false` | `false` | `true` | Yes | Use OpenEvolve (disabled) |
| `false` | `false` | `false` | No | Use OpenEvolve |
| `false` | `true` | `true` | No | Use OpenEvolve (disabled overrides require) |
| `true` | `false` | `true` | No | Use OpenEvolve (fallback) |
| `true` | `false` | `true` | Yes | Use LoongFlow |
| `true` | `false` | `false` | No | Raise error |
| `true` | `false` | `false` | Yes | Use LoongFlow |
| `true` | `true` | `true` | No | Raise error (require overrides fallback) |
| `true` | `true` | `true` | Yes | Use LoongFlow |
| `true` | `true` | `false` | No | Raise error |
| `true` | `true` | `false` | Yes | Use LoongFlow |

### Recommended Configurations

#### Development: Fast Iteration
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=False,  # Skip LoongFlow overhead
    max_iterations=20,
    verbose=True
)
```

#### Production: Graceful Degradation
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Fallback if unavailable
    require_loongflow=False
)
```

#### Production: Strict Requirement
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True,  # Must have LoongFlow
    loongflow_fallback_enabled=False
)
```

#### Testing: Compare Both
```python
# Test with LoongFlow
config_lf = UnifiedEvolutionConfig(
    enable_loongflow=True,
    evolution_mode=EvolutionMode.PES
)

# Test without LoongFlow
config_oe = UnifiedEvolutionConfig(
    enable_loongflow=False,
    evolution_mode=EvolutionMode.QD
)
```

## Best Practices

### 1. Always Enable Fallback in Production

```python
# ✅ GOOD: Graceful degradation
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True  # Allow fallback
)

# ❌ BAD: Brittle system
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=False  # No fallback
)
```

**Reasoning:** Production systems should be resilient. Fallback ensures continued operation even if LoongFlow becomes unavailable.

### 2. Log Fallback Events

```python
# ✅ GOOD: Monitor fallbacks
logger.warning(
    "LoongFlow unavailable, falling back to OpenEvolve",
    extra={
        'domain': domain,
        'problem_type': characteristics.get('type'),
        'fallback_reason': 'unavailable'
    }
)

# ❌ BAD: Silent fallback
# No logging when fallback occurs
```

**Reasoning:** Monitoring fallback frequency helps identify issues and plan infrastructure improvements.

### 3. Test Both Modes

```python
# ✅ GOOD: Test both configurations
async def test_with_both_configs(problem: str):
    # Test with LoongFlow
    result_lf = await evolve(
        problem=problem,
        use_loongflow=True
    )

    # Test with OpenEvolve only
    result_oe = await evolve(
        problem=problem,
        use_loongflow=False
    )

    # Both should succeed
    assert result_lf.final_score > 0
    assert result_oe.final_score > 0

# ❌ BAD: Only test one mode
result = await evolve(problem, use_loongflow=True)
# Don't test OpenEvolve-only mode
```

**Reasoning:** Ensures both configurations work and detects regressions early.

### 4. Document Fallback Behavior

```python
# ✅ GOOD: Document configuration
"""
Configuration Notes:
- enable_loongflow=True for expensive evaluations
- loongflow_fallback_enabled=True for graceful degradation
- Falls back to OpenEvolve QD mode if LoongFlow unavailable
"""

config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)

# ❌ BAD: No documentation
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)
# No explanation of why these settings were chosen
```

**Reasoning:** Future maintainers need to understand configuration decisions.

### 5. Use Feature Flags

```python
# ✅ GOOD: Feature flag control
enable_loongflow = os.getenv('EVOLVE_ENABLE_LOONGFLOW', 'true').lower() == 'true'

config = UnifiedEvolutionConfig(
    enable_loongflow=enable_loongflow,
    loongflow_fallback_enabled=True
)

# ❌ BAD: Hard-coded configuration
config = UnifiedEvolutionConfig(
    enable_loongflow=True,  # Hard-coded
    loongflow_fallback_enabled=True
)
```

**Reasoning:** Feature flags allow runtime configuration without code changes.

## Troubleshooting

### Issue: Unexpected Fallback to OpenEvolve

**Symptoms:**
- Expected LoongFlow but got OpenEvolve
- Log shows "LoongFlow unavailable"

**Diagnosis:**
```python
# Check LoongFlow availability
from openevolve.unified import LOONGFLOW_AVAILABLE

print(f"LoongFlow available: {LOONGFLOW_AVAILABLE}")

# Check configuration
config = UnifiedEvolutionConfig(enable_loongflow=True)
print(f"enable_loongflow: {config.enable_loongflow}")
print(f"loongflow_fallback_enabled: {config.loongflow_fallback_enabled}")
```

**Solutions:**
1. Install LoongFlow: `pip install loongflow`
2. Verify installation: `python -c "import loongflow; print('OK')"`
3. Check configuration isn't disabled

### Issue: Fallback Not Triggering When Expected

**Symptoms:**
- LoongFlow fails but no fallback occurs
- Exception raised instead

**Diagnosis:**
```python
# Check fallback is enabled
assert config.loongflow_fallback_enabled == True

# Check not in strict mode
assert config.require_loongflow == False
```

**Solutions:**
1. Enable fallback: `loongflow_fallback_enabled=True`
2. Disable strict mode: `require_loongflow=False`

### Issue: Poor Performance After Fallback

**Symptoms:**
- More evaluations than expected
- Lower solution quality

**Diagnosis:**
```python
# Check which mode was used
print(f"System: {result.strategy_used.system}")
print(f"Mode: {result.strategy_used.mode}")

# Compare with LoongFlow run
result_lf = await evolve(problem, domain, use_loongflow=True)
result_oe = await evolve(problem, domain, use_loongflow=False)

print(f"LoongFlow evals: {result_lf.evaluations}")
print(f"OpenEvolve evals: {result_oe.evaluations}")
```

**Solutions:**
1. Increase iterations in OpenEvolve mode
2. Use QD mode for better diversity
3. Enable island model for parallelism

## Summary

### Key Principles

1. **Graceful Degradation**: System continues working even if LoongFlow unavailable
2. **Configuration Control**: Fine-grained control over fallback behavior
3. **Clear Error Messages**: Helpful error messages for all failure modes
4. **Performance Monitoring**: Track fallback frequency and impact
5. **Testing Coverage**: Test all fallback scenarios

### Decision Matrix

| Situation | Configuration | Expected Behavior |
|-----------|---------------|-------------------|
| **Production with LoongFlow** | `enable_loongflow=True`, `loongflow_fallback_enabled=True` | Use LoongFlow, fallback to OpenEvolve if unavailable |
| **Production without LoongFlow** | `enable_loongflow=False` | Use OpenEvolve only |
| **Development** | `enable_loongflow=False` | Use OpenEvolve for faster iteration |
| **Testing** | Test both configurations | Validate both systems work |
| **Strict LoongFlow requirement** | `enable_loongflow=True`, `require_loongflow=True` | Fail if LoongFlow unavailable |
| **Expensive evaluations** | `enable_loongflow=True` | Use LoongFlow PES for 60% reduction |
| **Cheap evaluations** | `enable_loongflow=False` | Use OpenEvolve, no benefit from LoongFlow |

### Next Steps

1. **Monitor fallback metrics**: Track how often fallback occurs
2. **Test both configurations**: Ensure both LoongFlow and OpenEvolve work
3. **Optimize configuration**: Tune settings based on observed performance
4. **Document decisions**: Record why specific configurations were chosen
5. **Plan infrastructure**: Decide whether to deploy LoongFlow in production

For more information, see:
- [Optional LoongFlow Usage Guide](./OPTIONAL_LOONGFLOW_GUIDE.md)
- [Configuration Options Reference](./CONFIGURATION_OPTIONS.md)
- [Unified Evolution API](./UNIFIED_EVOLUTION_API.md)
