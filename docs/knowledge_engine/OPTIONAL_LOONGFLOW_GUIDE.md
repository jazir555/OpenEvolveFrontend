# Optional LoongFlow Usage Guide

## Overview

LoongFlow can now be completely optional in the evolution workflow. The system will gracefully fall back to OpenEvolve-only mode when LoongFlow is disabled or unavailable.

## Why Make LoongFlow Optional?

### Valid Use Cases for Disabling LoongFlow:

1. **Dependency Management**: Reduce dependencies in production
   - Fewer packages to maintain
   - Simpler deployment
   - Smaller Docker images

2. **Cost Optimization**: LoongFlow may require API credits
   - LLM API costs for planning
   - Memory system storage costs
   - Reduced operational overhead

3. **Simplicity**: Use familiar OpenEvolve-only workflow
   - Well-tested evolutionary algorithms
   - Predictable behavior
   - Easier debugging

4. **Testing**: Compare LoongFlow vs OpenEvolve performance
   - A/B testing strategies
   - Benchmarking
   - Validation studies

5. **Compliance**: Restrict to specific tools
   - Organizational constraints
   - Approved software lists
   - Audit requirements

6. **Debugging**: Isolate issues to one system
   - Troubleshoot problems
   - Understand behavior
   - Root cause analysis

## How to Disable LoongFlow

### Method 1: Configuration Parameter

```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    config=UnifiedEvolutionConfig(
        enable_loongflow=False
    )
)
```

### Method 2: Runtime Override

```python
result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    use_loongflow=False  # Runtime override
)
```

### Method 3: Global Configuration File

```yaml
# evolve.config.yaml
enable_loongflow: false
loongflow_fallback_enabled: true
require_loongflow: false
```

### Method 4: Convenience Function

```python
from openevolve.unified import evolve_openevolve_only

result = await evolve_openevolve_only(
    problem="Optimize portfolio",
    domain="finance"
)
```

### Method 5: Environment Variable

```bash
# Set environment variable
export EVOLVE_ENABLE_LOONGFLOW=false

# Or in .env file
echo "EVOLVE_ENABLE_LOONGFLOW=false" >> .env
```

## Configuration Options

### `enable_loongflow` (default: `true`)

Enable or disable LoongFlow PES system globally.

**Type:** `bool`
**Default:** `true`
**Impact:** When `false`, strategy selection only considers OpenEvolve modes

**Example:**
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=False,
    evolution_mode=EvolutionMode.QD  # Will use OpenEvolve QD
)
```

### `loongflow_fallback_enabled` (default: `true`)

Allow fallback to OpenEvolve if LoongFlow is unavailable.

**Type:** `bool`
**Default:** `true`
**Impact:** When `true`, gracefully degrades if LoongFlow not installed or fails

**Example:**
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True  # Fall back if LoongFlow unavailable
)
```

### `require_loongflow` (default: `false`)

Require LoongFlow to be available. Raise error if not installed.

**Type:** `bool`
**Default:** `false`
**Impact:** When `true`, fails fast if LoongFlow not available

**Example:**
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True,  # Strict requirement
    loongflow_fallback_enabled=False
)
```

### `use_loongflow` (runtime parameter)

Override config for a specific run.

**Type:** `bool`
**Default:** `None` (use config value)
**Impact:** Runtime override for single evolution call

**Example:**
```python
# Use LoongFlow for this run only
result = await evolve(
    problem="Expensive optimization",
    domain="science",
    use_loongflow=True
)

# Use OpenEvolve-only for this run only
result = await evolve(
    problem="Quick test",
    domain="web",
    use_loongflow=False
)
```

## How Fallback Works

### Decision Tree

```
Is enable_loongflow=True?
├─ No → Use OpenEvolve-only mode
└─ Yes → Check LoongFlow available
    ├─ Available → Use LoongFlow
    └─ Not Available → Check require_loongflow
        ├─ True → Raise error (fail fast)
        └─ False → Check loongflow_fallback_enabled
            ├─ True → Use OpenEvolve-only mode (graceful fallback)
            └─ False → Raise error (fallback disabled)
```

### Implementation Details

The fallback logic is implemented in the unified evolution API:

```python
async def _select_strategy(self, problem: str, domain: str,
                          config: UnifiedEvolutionConfig) -> SystemMode:
    """Select optimal strategy with graceful fallback"""

    # If LoongFlow explicitly disabled, use OpenEvolve only
    if not config.enable_loongflow:
        return self._select_openevolve_strategy(problem, domain, config)

    # Check if LoongFlow is available
    loongflow_available = LOONGFLOW_AVAILABLE

    if not loongflow_available:
        if config.require_loongflow:
            raise LoongFlowRequiredError(
                "LoongFlow is required but not available. "
                "Install with: pip install loongflow"
            )

        if config.loongflow_fallback_enabled:
            logger.warning("LoongFlow unavailable, falling back to OpenEvolve")
            return self._select_openevolve_strategy(problem, domain, config)
        else:
            raise LoongFlowUnavailableError(
                "LoongFlow unavailable and fallback disabled"
            )

    # LoongFlow available - can consider PES mode
    return await self._select_optimal_strategy(problem, domain, config)
```

### What Happens in Fallback

When using OpenEvolve-only mode:

1. **Strategy Selection**: Only OpenEvolve modes considered
   - Standard evolutionary optimization
   - QD (Quality-Diversity)
   - MO (Multi-Objective)
   - Adversarial (robustness testing)

2. **Execution**: Uses OpenEvolve executor
   - All evolutionary operators work
   - Island model parallelism
   - MAP-Elites archive
   - NSGA-II for multi-objective

3. **Knowledge Extraction**: OpenEvolve-specific artifacts
   - Population archives
   - Fitness history
   - Elite solutions
   - Lineage tracking

4. **Gauntlet**: Still runs 3-round evaluation
   - Round 1: OpenEvolve-based evaluation (instead of LoongFlow)
   - Round 2: Red Team (unchanged)
   - Round 3: Gold Team (unchanged)

## Capabilities Comparison

| Feature | With LoongFlow | OpenEvolve Only |
|---------|---------------|----------------|
| **Directed search (PES)** | ✅ Yes | ❌ No |
| **60% fewer evaluations** | ✅ Yes | ❌ No |
| **Planning strategies** | ✅ Yes | ❌ No |
| **Quality Diversity (QD)** | ✅ Yes | ✅ Yes |
| **Multi-Objective (MO)** | ✅ Yes | ✅ Yes |
| **Adversarial testing** | ✅ Yes | ✅ Yes |
| **MAP-Elites archive** | ✅ Yes | ✅ Yes |
| **Island model** | ✅ Yes | ✅ Yes |
| **Standard evolution** | ✅ Yes | ✅ Yes |
| **3-round gauntlet** | ✅ Yes | ✅ Yes |
| **Knowledge extraction** | ✅ Yes | ✅ Yes |
| **Sample efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Solution quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Execution speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Performance Impact

**Finance Domain (Expensive Backtests):**
- **With LoongFlow**: ~30 evaluations × $10 = $300
- **OpenEvolve Only**: ~75 evaluations × $10 = $750
- **Difference**: 2.5× more evaluations

**Science Domain (Very Expensive Experiments):**
- **With LoongFlow**: ~12 experiments × $5,000 = $60,000
- **OpenEvolve Only**: ~30 experiments × $5,000 = $150,000
- **Difference**: $90,000 additional cost

**Web Domain (Cheap Evaluations):**
- **With LoongFlow**: ~50 evaluations (unnecessary overhead)
- **OpenEvolve Only**: ~200 evaluations (no noticeable difference)
- **Difference**: Minimal impact

## OpenEvolve-Only Recommendations

### When LoongFlow is Disabled:

| Domain | Recommended Mode | Rationale |
|--------|----------------|-----------|
| **Finance** | Standard or MO | OpenEvolve works well for portfolio optimization. Use MO for return/risk/liquidity tradeoffs. |
| **Trading** | Adversarial | OpenEvolve adversarial is excellent for robustness testing against market regime changes. |
| **Science** | QD | QD explores experimental space well with MAP-Elites. Good for diverse experimental designs. |
| **Engineering** | Standard or MO | Multi-objective optimization works well for design problems (weight/strength/cost). |
| **Pharma** | QD | Chemical space exploration with MAP-Elites finds diverse molecular structures. |
| **Web** | Standard | Fast evaluations, standard GA is sufficient. No need for complex modes. |
| **General** | Standard or QD | Start with standard, use QD if diversity needed. |

### Compensating for Lack of LoongFlow

When using OpenEvolve-only mode, you can compensate for reduced sample efficiency:

1. **Increase iterations**
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       max_iterations=200  # Increase from default 100
   )
   ```

2. **Use QD mode for diversity**
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       evolution_mode=EvolutionMode.QD,
       qd=QDConfig(
           enabled=True,
           archive_size=2000  # Larger archive for more diversity
       )
   )
   ```

3. **Enable island model**
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       database=DatabaseConfig(
           enable_island_model=True,
           num_islands=4  # Parallel evolution
       )
   )
   ```

4. **Use adversarial mode for robustness**
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       evolution_mode=EvolutionMode.ADVERSARIAL,
       adversarial=AdversarialConfig(
           enabled=True,
           num_adversarial_rounds=3
       )
   )
   ```

## Examples

### Example 1: Explicit Disable

```python
from openevolve.unified import evolve

result = await evolve(
    problem="Maximize f(x) = x^2",
    domain="general",
    use_loongflow=False  # Explicitly disable
)

print(f"System used: {result.strategy_used.system}")  # "openevolve"
print(f"Mode: {result.strategy_used.mode}")  # "standard" or "qd"
print(f"Score: {result.final_score}")
```

### Example 2: Configuration File

```yaml
# config.yaml
enable_loongflow: false
domain: trading
evolution_mode: adversarial
max_iterations: 150

adversarial:
  enabled: true
  num_adversarial_rounds: 3
  robustness_threshold: 0.7
```

```python
# Load config
import yaml

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)
    config = UnifiedEvolutionConfig(**config_dict)

result = await evolve(
    problem="Develop trading strategy robust to regime changes",
    domain="trading",
    config=config
)
```

### Example 3: Convenience Function

```python
from openevolve.unified import evolve_openevolve_only

# OpenEvolve-only mode
result = await evolve_openevolve_only(
    problem="Develop trading strategy",
    domain="trading"
)

print(f"Strategy: {result.strategy_used.mode}")  # OpenEvolve mode only
```

### Example 4: Require LoongFlow (Error If Not Available)

```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

try:
    result = await evolve(
        problem="Expensive optimization",
        domain="science",
        config=UnifiedEvolutionConfig(
            require_loongflow=True,  # Strict requirement
            loongflow_fallback_enabled=False
        )
    )
except ImportError as e:
    print(f"LoongFlow required but not installed: {e}")
    print("Install with: pip install loongflow")
```

### Example 5: Graceful Fallback

```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

# Will use LoongFlow if available, fall back to OpenEvolve if not
result = await evolve(
    problem="Optimize function",
    domain="general",
    config=UnifiedEvolutionConfig(
        enable_loongflow=True,
        loongflow_fallback_enabled=True,  # Graceful fallback
        require_loongflow=False
    )
)

print(f"System used: {result.strategy_used.system}")
# If LoongFlow installed: "loongflow"
# If not: "openevolve" (with warning logged)
```

### Example 6: Dynamic Selection Based on Cost

```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

def optimize(problem: str, domain: str, evaluation_cost: str):
    """Select mode based on evaluation cost"""

    if evaluation_cost == "very_expensive":
        # Require LoongFlow for very expensive evaluations
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            evolution_mode=EvolutionMode.PES
        )
    elif evaluation_cost == "expensive":
        # Prefer LoongFlow but allow fallback
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True
        )
    else:
        # Cheap evaluations, use OpenEvolve only
        config = UnifiedEvolutionConfig(
            enable_loongflow=False,
            max_iterations=200  # Can do more iterations
        )

    return await evolve(problem, domain, config=config)

# Usage
result = optimize(
    problem="Optimize portfolio",
    domain="finance",
    evaluation_cost="expensive"
)
```

### Example 7: A/B Testing

```python
from openevolve.unified import evolve

async def compare_strategies(problem: str, domain: str):
    """Compare LoongFlow vs OpenEvolve"""

    # Run with LoongFlow
    result_lf = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=True,
        run_gauntlet=False  # Skip for speed
    )

    # Run with OpenEvolve only
    result_oe = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=False,
        run_gauntlet=False
    )

    # Compare results
    print(f"LoongFlow:")
    print(f"  Score: {result_lf.final_score:.4f}")
    print(f"  Evaluations: {result_lf.evaluations}")
    print(f"  Time: {result_lf.total_time:.2f}s")

    print(f"\nOpenEvolve:")
    print(f"  Score: {result_oe.final_score:.4f}")
    print(f"  Evaluations: {result_oe.evaluations}")
    print(f"  Time: {result_oe.total_time:.2f}s")

    print(f"\nImprovement:")
    print(f"  Score: {(result_lf.final_score / result_oe.final_score - 1) * 100:.1f}%")
    print(f"  Evaluations: {(result_oe.evaluations / result_lf.evaluations - 1) * 100:.1f}%")

    return result_lf, result_oe

# Usage
result_lf, result_oe = await compare_strategies(
    problem="Maximize Sharpe ratio",
    domain="finance"
)
```

## Troubleshooting

### Issue: "LoongFlow not available" Warning

**Cause:** LoongFlow package not installed but required

**Symptoms:**
```
WARNING: LoongFlow unavailable, falling back to OpenEvolve
```

**Solutions:**

1. **Install LoongFlow** (if you want PES benefits):
   ```bash
   pip install loongflow
   # or
   pip install git+https://github.com/baidu-baige/LoongFlow.git
   ```

2. **Disable requirement** (if OpenEvolve-only is acceptable):
   ```python
   config = UnifiedEvolutionConfig(
       require_loongflow=False,
       loongflow_fallback_enabled=True
   )
   ```

3. **Allow fallback** (already default):
   ```python
   config = UnifiedEvolutionConfig(
       loongflow_fallback_enabled=True
   )
   ```

### Issue: Poor Performance in OpenEvolve-Only Mode

**Cause:** Problem may benefit from LoongFlow's directed search

**Symptoms:**
- More evaluations than expected
- Lower solution quality
- Slower convergence

**Solutions:**

1. **Enable LoongFlow**:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=True,
       evolution_mode=EvolutionMode.PES
   )
   ```

2. **Increase iterations** to compensate:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       max_iterations=200  # Increase from 100
   )
   ```

3. **Use QD mode** for diversity benefits:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       evolution_mode=EvolutionMode.QD,
       qd=QDConfig(
           enabled=True,
           archive_size=2000,
           grid_resolution=[20, 20]
       )
   )
   ```

4. **Enable island model** for parallel exploration:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       database=DatabaseConfig(
           enable_island_model=True,
           num_islands=4,
           migration_interval=20
       )
   )
   ```

### Issue: Strategy Selection Ignores Context

**Cause:** Historical data may be from LoongFlow runs, not applicable to OpenEvolve-only

**Symptoms:**
- Recommends PES mode when LoongFlow disabled
- Suboptimal mode selection
- Poor performance

**Solutions:**

1. **Clear historical data** or regenerate with OpenEvolve-only:
   ```python
   # Clear strategy recommender cache
   await strategy_recommender.clear_history()
   ```

2. **Use domain-specific defaults**:
   ```python
   result = await evolve(
       problem="...",
       domain="trading",
       config=UnifiedEvolutionConfig(
           enable_loongflow=False,
           evolution_mode=EvolutionMode.ADVERSARIAL  # Explicit mode
       )
   )
   ```

3. **Manually specify mode** instead of relying on auto-selection:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       evolution_mode=EvolutionMode.MO  # Force specific mode
   )
   ```

### Issue: Import Error When Using PES Mode

**Cause:** Trying to use PES mode without LoongFlow installed

**Symptoms:**
```
ImportError: LoongFlow is required for PES mode
```

**Solutions:**

1. **Install LoongFlow**:
   ```bash
   pip install loongflow
   ```

2. **Use OpenEvolve mode instead**:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=False,
       evolution_mode=EvolutionMode.QD  # Use QD instead of PES
   )
   ```

3. **Allow fallback** to QD mode:
   ```python
   config = UnifiedEvolutionConfig(
       enable_loongflow=True,
       evolution_mode=EvolutionMode.AUTO,  # Will select QD if PES unavailable
       loongflow_fallback_enabled=True
   )
   ```

## Best Practices

### 1. Development Environment

```python
# Use OpenEvolve-only for faster iteration
config = UnifiedEvolutionConfig(
    enable_loongflow=False,
    max_iterations=20,  # Quick iterations
    verbose=True
)

result = await evolve(
    problem="Prototype optimization",
    domain="web",
    config=config
)
```

**Benefits:**
- Faster iterations
- No LLM API costs
- Simpler debugging
- No external dependencies

### 2. Production Environment

```python
# Use LoongFlow for expensive evaluations
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Graceful degradation
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True
    )
)

result = await evolve(
    problem="Optimize portfolio allocation",
    domain="finance",
    config=config
)
```

**Benefits:**
- 60% fewer evaluations
- Reduced API costs
- Better solution quality
- Graceful fallback if LoongFlow unavailable

### 3. Testing Environment

```python
# Test both modes for comparison
async def test_both_modes(problem: str, domain: str):
    # Test with LoongFlow
    result_lf = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=True,
        run_gauntlet=False
    )

    # Test with OpenEvolve only
    result_oe = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=False,
        run_gauntlet=False
    )

    # Validate both produce reasonable results
    assert result_lf.final_score > 0.8
    assert result_oe.final_score > 0.7

    # Log comparison
    logger.info(f"LoongFlow: {result_lf.final_score:.3f} in {result_lf.evaluations} evals")
    logger.info(f"OpenEvolve: {result_oe.final_score:.3f} in {result_oe.evaluations} evals")

    return result_lf, result_oe
```

**Benefits:**
- Validate both systems work
- Compare performance
- Detect regressions
- Build confidence

### 4. Deployment Strategy

```bash
# Set environment variable for deployment
export EVOLVE_ENABLE_LOONGFLOW=true

# Or in Dockerfile
ENV EVOLVE_ENABLE_LOONGFLOW=true

# Or in Kubernetes config
env:
  - name: EVOLVE_ENABLE_LOONGFLOW
    value: "true"
```

**Benefits:**
- Easy configuration
- No code changes
- Environment-specific settings
- Feature flag control

### 5. Cost-Optimized Configuration

```python
def get_config_for_budget(budget: float, domain: str) -> UnifiedEvolutionConfig:
    """Select optimal config based on budget"""

    if domain == "science":
        eval_cost = 5000  # $5K per experiment
    elif domain == "finance":
        eval_cost = 10  # $10 per backtest
    else:
        eval_cost = 1  # Cheap

    max_evals = budget / eval_cost

    if max_evals < 50:
        # Very limited budget, require LoongFlow
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(
                enabled=True,
                max_rounds=3
            )
        )
    elif max_evals < 200:
        # Moderate budget, prefer LoongFlow
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True,
            max_iterations=50
        )
    else:
        # Generous budget, OpenEvolve is fine
        return UnifiedEvolutionConfig(
            enable_loongflow=False,
            max_iterations=200
        )

# Usage
config = get_config_for_budget(budget=100000, domain="science")
result = await evolve(
    problem="Optimize reaction conditions",
    domain="science",
    config=config
)
```

## Migration Guide

### From LoongFlow-Dependent to OpenEvolve-Only

**Before** (requires LoongFlow):
```python
from loongflow import LoongFlowEvolve
from loongflow.config import PESConfig

config = PESConfig(
    enable_planning=True,
    enable_memory=True,
    max_rounds=5
)

evolver = LoongFlowEvolve(config=config)
result = evolver.evolve(problem="Optimize portfolio")
```

**After** (OpenEvolve-only):
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    config=UnifiedEvolutionConfig(
        enable_loongflow=False,
        evolution_mode=EvolutionMode.QD  # Use QD instead of PES
    )
)
```

**No other changes needed!** The unified API handles everything else.

### From Pure OpenEvolve to Hybrid (with optional LoongFlow)

**Before** (OpenEvolve only):
```python
from openevolve import OpenEvolve
from openevolve.config import QDConfig

config = QDConfig(
    archive_size=1000,
    grid_resolution=[20, 20]
)

evolver = OpenEvolve(config=config)
result = evolver.evolve(problem="Optimize portfolio")
```

**After** (Hybrid with optional LoongFlow):
```python
from openevolve.unified import evolve

result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    use_loongflow=True,  # Will use LoongFlow if available
    loongflow_fallback_enabled=True  # Fall back to OpenEvolve if not
)
```

**Benefits:**
- Automatic strategy selection
- Graceful fallback
- Better performance when LoongFlow available
- No breaking changes

### Migration Checklist

- [ ] Install unified evolution API: `pip install openevolve[unified]`
- [ ] Update imports: Use `openevolve.unified.evolve`
- [ ] Convert config to `UnifiedEvolutionConfig`
- [ ] Set `enable_loongflow` based on requirements
- [ ] Configure `loongflow_fallback_enabled` for graceful degradation
- [ ] Set `require_loongflow` if strict requirement
- [ ] Test with both `use_loongflow=True` and `use_loongflow=False`
- [ ] Update CI/CD pipelines to test both modes
- [ ] Update documentation
- [ ] Monitor performance and costs

## FAQ

**Q: Will I lose functionality if I disable LoongFlow?**

A: You'll lose PES-directed search and 60% evaluation efficiency gains, but all OpenEvolve features (QD, MO, Adversarial) remain available. For cheap evaluations, the difference is minimal.

**Q: Can I switch back and forth between LoongFlow and OpenEvolve?**

A: Yes! Use `use_loongflow=True` for one run, `use_loongflow=False` for the next. No reconfiguration needed. The system handles mode selection automatically.

**Q: What happens to my knowledge extraction when LoongFlow is disabled?**

A: Knowledge extraction still works perfectly, just uses OpenEvolve-specific artifacts instead of LoongFlow artifacts. The strategy recommender still learns from OpenEvolve runs.

**Q: Will my gauntlet evaluation change in OpenEvolve-only mode?**

A: Round 1 will use OpenEvolve-based evaluation instead of LoongFlow AI. Rounds 2 (Red Team) and 3 (Gold Team) remain unchanged. The gauntlet still provides comprehensive quality assurance.

**Q: Can I deploy without LoongFlow installed?**

A: Yes! If `enable_loongflow=False` and `loongflow_fallback_enabled=True`, the system will work perfectly with just OpenEvolve. No LoongFlow dependency required.

**Q: How do I know if LoongFlow is being used?**

A: Check the result:
```python
result = await evolve(...)
print(f"System: {result.strategy_used.system}")  # "loongflow" or "openevolve"
print(f"Mode: {result.strategy_used.mode}")  # "pes", "qd", "mo", etc.
```

**Q: What's the performance difference between LoongFlow and OpenEvolve?**

A: For expensive evaluations (science, finance), LoongFlow PES can reduce evaluations by 60%. For cheap evaluations (web, general), the difference is minimal. Use LoongFlow for expensive problems, OpenEvolve for cheap ones.

**Q: Can I use PES mode without LoongFlow installed?**

A: No. PES mode requires LoongFlow. If you try to use PES mode without LoongFlow, the system will either:
- Fall back to QD mode (if `loongflow_fallback_enabled=True`)
- Raise an error (if `require_loongflow=True`)

**Q: How do I enable LoongFlow in production?**

A: Three options:
1. Set `enable_loongflow=True` in config
2. Use `use_loongflow=True` at runtime
3. Set environment variable: `export EVOLVE_ENABLE_LOONGFLOW=true`

**Q: What happens if LoongFlow fails during execution?**

A: The system will catch the error and fall back to OpenEvolve if `loongflow_fallback_enabled=True`. If `require_loongflow=True`, it will raise an error instead.

**Q: Should I use LoongFlow for web optimization?**

A: Probably not. Web evaluations are cheap (Lighthouse tests take seconds). LoongFlow's benefits are most pronounced for expensive evaluations. Use OpenEvolve QD or Standard mode instead.

**Q: Can I disable LoongFlow after it's been enabled?**

A: Yes, anytime. Just set `use_loongflow=False` for a specific run or update your config file. The change takes effect immediately.

**Q: How do I monitor LoongFlow usage?**

A: Check logs and metrics:
```python
result = await evolve(...)
print(f"System: {result.strategy_used.system}")
print(f"Evaluations: {result.evaluations}")
print(f"Time: {result.total_time:.2f}s")
```

Look for log messages like `"Using LoongFlow PES"` or `"Falling back to OpenEvolve"`.

## Summary

### Key Takeaways

1. **LoongFlow is now optional** - Use it when beneficial, skip when not
2. **Graceful fallback** - System automatically degrades if LoongFlow unavailable
3. **Configuration options** - Fine-grained control over LoongFlow usage
4. **OpenEvolve-only mode** - Fully functional without LoongFlow
5. **Easy to switch** - Runtime overrides, environment variables, config files
6. **Performance awareness** - Use LoongFlow for expensive evaluations, OpenEvolve for cheap

### Decision Matrix

| Scenario | Recommendation | Configuration |
|----------|---------------|---------------|
| Expensive evaluations (science, finance) | Use LoongFlow | `enable_loongflow=True` |
| Cheap evaluations (web, general) | OpenEvolve-only | `enable_loongflow=False` |
| Development/Testing | OpenEvolve-only | `enable_loongflow=False` |
| Production with budget | Use LoongFlow with fallback | `enable_loongflow=True, loongflow_fallback_enabled=True` |
| Compliance restrictions | OpenEvolve-only | `enable_loongflow=False` |
| Performance comparison | Test both modes | Run with `use_loongflow=True` and `False` |
| Unknown LoongFlow availability | Graceful fallback | `loongflow_fallback_enabled=True` |
| Strict LoongFlow requirement | Require LoongFlow | `require_loongflow=True` |

### Next Steps

1. **Evaluate your use case**: Are evaluations expensive? Does sample efficiency matter?
2. **Configure appropriately**: Set `enable_loongflow` based on your needs
3. **Test both modes**: Compare performance for your specific problems
4. **Monitor results**: Track evaluations, solution quality, execution time
5. **Adjust as needed**: Fine-tune configuration based on observed performance

For more information, see:
- [Configuration Options Guide](./CONFIGURATION_OPTIONS.md)
- [Fallback Mechanism Documentation](./FALLBACK_DOCUMENTATION.md)
- [Unified Evolution API](./UNIFIED_EVOLUTION_API.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
