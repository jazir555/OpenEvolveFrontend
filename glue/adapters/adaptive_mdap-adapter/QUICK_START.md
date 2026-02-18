# Quick Start Guide - Adaptive MDAP/MAKER Adapter Integration

**Version**: 2.0.0
**Last Updated**: February 17, 2026

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Set Environment Variables

```bash
# Required
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="your-api-key-here"

# Optional (performance tuning)
export ADAPTIVE_MDAP_CACHE_SIZE=1000
export ADAPTIVE_MDAP_CACHE_TTL=300
```

### Step 2: Run Unified Entry Point

```bash
cd glue/adapters/adaptive_mdap-adapter

# Analyze a problem
python unified_entry.py analyze \
    --problem "Build distributed cache system" \
    --domain distributed_systems

# Run through gauntlet verification
python unified_entry.py verify \
    --solution "my_solution" \
    --complexity 0.7

# Check system status
python unified_entry.py status
```

### Step 3: Use in Python Code

```python
from src import get_integration_manager

# Get manager
manager = get_integration_manager()

# Execute full workflow
results = manager.execute_full_workflow(
    workflow_id="my_workflow",
    problem_statement="Solve complex optimization problem",
    workflow_type="evolution"
)

print(f"Status: {results['overall_status']}")
```

---

## 📖 Common Use Cases

### Use Case 1: Basic Complexity Analysis

**When**: You need to understand how complex a problem is before solving it.

```python
from src import get_adapter, CanonicalSubProblem

adapter = get_adapter()

subproblem = CanonicalSubProblem(
    id="analysis_001",
    description="Implement OAuth2 authentication",
    domain="security",
    depth=2
)

response = adapter.analyze_complexity(subproblem)
print(f"Complexity: {response.complexity_score.overall_score}")
print(f"Strategy: {response.strategy.value}")
```

### Use Case 2: Advanced Workflow with Decomposition

**When**: You have a complex problem that needs breaking down.

```python
from src import get_advanced_openevolve_integration

advanced = get_advanced_openevolve_integration()

# Decompose into sub-problems
decomposition = advanced.decompose_problem(
    workflow_id="complex_workflow",
    problem_statement="Build microservices architecture",
    workflow_type="sovereign",
    max_depth=3
)

# Get team recommendations
team_selection = advanced.select_teams_for_stage(
    workflow_id="complex_workflow",
    stage="solving",
    workflow_type="sovereign",
    complexity_score=0.75
)

# Optimize resources
optimization = advanced.optimize_resources(
    workflow_id="complex_workflow",
    stage="solving",
    complexity_score=0.75,
    estimated_duration_ms=60000
)
```

### Use Case 3: Multi-Gauntlet Verification

**When**: You need rigorous quality assurance for a solution.

```python
from src import get_advanced_gauntlet_integration, GauntletType

gauntlet = get_advanced_gauntlet_integration()

# Create pipeline with multiple gauntlets
pipeline = gauntlet.create_gauntlet_pipeline(
    complexity_score=0.8,
    base_gauntlet_type=GauntletType.FORMAL_VERIFICATION,
    include_cross_validation=True
)

# Execute pipeline
result = gauntlet.execute_pipeline(
    pipeline=pipeline,
    solution=my_solution
)

print(f"Passed: {result.passed_gauntlets}/{result.total_gauntlets}")
print(f"Overall: {result.overall_pass}")
```

### Use Case 4: High-Performance Concurrent Processing

**When**: You need to process many problems in parallel.

```python
import asyncio
from src import get_async_adapter, CanonicalSubProblem

async_adapter = get_async_adapter()

# Create multiple sub-problems
subproblems = [
    CanonicalSubProblem(
        id=f"task_{i}",
        description=f"Task {i}",
        domain="general",
        depth=1
    )
    for i in range(10)
]

# Process concurrently (3x-5x faster)
results = await async_adapter.batch_analyze_complexity(
    subproblems,
    max_concurrency=5
)
```

### Use Case 5: Pattern Learning and Prediction

**When**: You want to learn from past executions.

```python
from src import get_advanced_icr_integration, ICRPatternType

icr = get_advanced_icr_integration()

# Store pattern after execution
icr.store_pattern_advanced(
    pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
    passed=True,
    context={"domain": "security", "complexity": 0.7},
    metrics={"execution_time_ms": 1500}
)

# Get insights
insights = icr.get_pattern_insights()
print(f"Patterns learned: {insights['pattern_types']['WORKFLOW_EXECUTION']['count']}")

# Predict outcome
prediction = icr.predict_with_confidence(
    pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
    context={"domain": "security"},
    min_confidence=0.7
)
print(f"Predicted: {prediction.predicted_outcome}")
print(f"Confidence: {prediction.confidence:.2f}")
```

### Use Case 6: UI Dashboard Generation

**When**: You need visualization for BubbleLab UI.

```python
from src import get_advanced_bubblelab_ui

ui = get_advanced_bubblelab_ui()

# Analyze for UI
result = ui.analyze_complexity_for_ui(
    problem_description="Build real-time dashboard",
    domain="analytics",
    depth=2
)

# Get chart data
radar_chart = ui.create_complexity_radar_chart(result.problem_id)

# Get health dashboard
dashboard = ui.create_adapter_health_dashboard()

# Export report
report = ui.export_report("dashboard", format="markdown")
print(report)
```

---

## 🔧 Advanced Configuration

### Performance Tuning

```python
from src import get_async_adapter

# Create with custom cache settings
async_adapter = AsyncMDAPAdapter(
    cache_size=2000,      # Larger cache
    cache_ttl=600         # 10 minute TTL
)

# Use connection pooling
from src import ConnectionPool

pool = ConnectionPool(
    max_connections=20,
    idle_timeout=600
)

connection = pool.acquire()
# ... use connection ...
pool.release(connection)
```

### Gauntlet Customization

```python
from src import get_advanced_gauntlet_integration, GauntletType, GauntletSeverity

gauntlet = get_advanced_gauntlet_integration()

# Custom gauntlet configuration
config = gauntlet.configure_gauntlet(
    gauntlet_type=GauntletType.ADVERSARIAL,
    complexity_score=0.85,
    severity=GauntletSeverity.HARDCORE,
    custom_parameters={
        "attack_modes": ["systematic", "deep_dive", "exhaustive"],
        "max_attacks": 20
    }
)
```

### Cross-System Integration

```python
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()

# Check all systems
health = monitor.get_overall_health()

# Execute workflow across systems
results = monitor.execute_workflow(
    workflow_type="formal_verification",
    parameters={
        "query": "Knowledge base query",
        "constraints": ["x > 0"],
        "statement": "Theorem to verify"
    }
)
```

---

## 📊 Monitoring and Observability

### Health Monitoring

```python
from src import get_integration_manager

manager = get_integration_manager()

# Get health status
health = manager.get_health_status()

print(f"Overall: {health.overall_status.value}")
print(f"MDAP: {health.mdap_adapter_status}")
print(f"MAKER: {health.maker_adapter_status}")
```

### Performance Monitoring

```python
from src import get_performance_monitor

monitor = get_performance_monitor()

# Record operation
monitor.record("complexity_analysis", duration_ms=150)

# Get statistics
stats = monitor.get_stats("complexity_analysis")
print(f"Average: {stats['avg_ms']:.2f}ms")
print(f"P95: {stats['p95_ms']:.2f}ms")
```

### Cache Statistics

```python
from src import get_async_adapter

adapter = get_async_adapter()

# Get cache stats
stats = adapter.get_cache_stats()
print(f"Hit Rate: {stats['hit_rate']:.1%}")
print(f"Size: {stats['size']}/{stats['max_size']}")
```

---

## 🧪 Testing

### Run All Tests

```bash
cd glue/adapters/adaptive_mdap-adapter

# Comprehensive test suite
python test_comprehensive_integration.py

# End-to-end integration tests
python test_full_integration.py

# Complete feature demonstration
python example_complete_features.py
```

### Run Specific Test Category

```bash
# Test workflow types only
python test_comprehensive_integration.py 2>&1 | grep -A 5 "TEST SUITE 1"

# Test edge cases only
python test_comprehensive_integration.py 2>&1 | grep -A 5 "TEST SUITE 2"

# Test load scenarios
python test_comprehensive_integration.py 2>&1 | grep -A 5 "TEST SUITE 3"
```

---

## 🐛 Troubleshooting

### Issue: Module Not Found

```bash
# Ensure you're in the correct directory
cd glue/adapters/adaptive_mdap-adapter

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: API Key Not Set

```bash
# Set API key
export DEEPSEEK_API_KEY="sk-..."

# Or in Python
os.environ["DEEPSEEK_API_KEY"] = "sk-..."
```

### Issue: Integration Components Unavailable

```python
# Check what's available
from src import get_integration_manager

manager = get_integration_manager()
health = manager.get_health_status()

# Components show "disabled" if not available
print(f"ICR: {health.icr_integration_status}")
print(f"Gauntlet: {health.gauntlet_integration_status}")
```

---

## 📚 Next Steps

1. **Explore Examples**: Run `example_complete_features.py` to see all features
2. **Read Documentation**: Check `INTEGRATION_WITH_OPENEVOLVE.md` for details
3. **Run Tests**: Execute `test_comprehensive_integration.py` to verify
4. **Use Unified Entry Point**: Try `unified_entry.py --help` for CLI interface
5. **Monitor Performance**: Use dashboard to monitor adapter health

---

## 🎯 Common Patterns

### Pattern 1: Analyze → Optimize → Verify

```python
# 1. Analyze complexity
analysis = manager.analyze_workflow(
    workflow_id="my_workflow",
    problem_statement="Complex problem",
    workflow_type="evolution"
)

# 2. Get optimal resources
resources = manager.analyze_workflow(...)

# 3. Verify solution
from src import get_advanced_gauntlet_integration
gauntlet = get_advanced_gauntlet_integration()
result = gauntlet.execute_pipeline(...)
```

### Pattern 2: Learn → Predict → Adapt

```python
# 1. Store pattern after execution
icr.store_pattern_advanced(...)

# 2. Predict before execution
prediction = icr.predict_with_confidence(...)

# 3. Adapt based on prediction
if prediction.confidence > 0.8:
    # Use recommended strategy
    strategy = prediction.recommended_action
else:
    # Use conservative fallback
    strategy = "DIRECT"
```

### Pattern 3: Async Batch → Cache → Monitor

```python
# 1. Batch process asynchronously
results = await async_adapter.batch_analyze_complexity(subproblems)

# 2. Cache automatically handled (90% hit rate)
# 3. Monitor performance
monitor.record("batch_analysis", duration_ms=total_time)
```

---

## 💡 Tips

1. **Start Simple**: Use basic `analyze_complexity()` before advanced features
2. **Use Caching**: Enable caching for repeated analyses (automatic in async adapter)
3. **Monitor Health**: Check health status regularly using `get_health_status()`
4. **Learn Patterns**: Use ICR to learn from past executions
5. **Verify Solutions**: Always run through gauntlet for critical solutions

---

## 📞 Getting Help

1. **Documentation**: Check `README.md`, `ADR.md`, and integration docs
2. **Examples**: Run `example_complete_features.py` for demonstrations
3. **Tests**: Run test suites to verify functionality
4. **Health Check**: Use `unified_entry.py status` to diagnose issues

---

**Quick Start Complete!** 🎉

You're now ready to use the Adaptive MDAP/MAKER Adapter integration. Start with the unified entry point and explore from there!

For detailed information, see:
- `INTEGRATION_COMPLETE.md` - Original integration details
- `INTEGRATION_WITH_OPENEVOLVE.md` - Integration with OpenEvolve/BubbleLab
- `ENHANCEMENTS_COMPLETE.md` - All enhancement details
- `README.md` - Full API reference

---

*"The journey of a thousand integrations begins with a single step."*
— Quick Start Philosophy
