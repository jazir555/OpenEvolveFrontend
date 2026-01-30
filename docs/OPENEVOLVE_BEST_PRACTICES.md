# OpenEvolve Best Practices Guide

**Optimize Your OpenEvolve Integration for Maximum Performance and Quality**

---

## Overview

This guide provides proven strategies and best practices for using OpenEvolve effectively. Follow these recommendations to achieve better results, improve performance, and avoid common pitfalls.

---

## 🚀 Getting Started Right

### 1. Choose the Right Evolution Mode

**Standard Evolution** - Best for:
- General problem solving
- Single-objective optimization
- Quick prototyping
- Learning OpenEvolve basics

```python
config = {
    "evolution_mode": "standard",
    "max_iterations": 20,
    "population_size": 30,
    "temperature": 0.7
}
```

**Quality Diversity** - Best for:
- Exploring diverse solutions
- Creative problem solving
- When you need multiple approaches
- Research and experimentation

```python
config = {
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["complexity", "novelty", "efficiency"],
    "archive_size": 100,
    "feature_bins": 10,
    "max_iterations": 50
}
```

**Multi-Objective** - Best for:
- Balancing competing goals
- Trade-off analysis
- Complex optimization problems
- Business decision making

```python
config = {
    "evolution_mode": "multi_objective",
    "objectives": ["accuracy", "speed", "cost"],
    "pareto_front_size": 50,
    "max_iterations": 100
}
```

**Adversarial** - Best for:
- Security testing
- Robustness evaluation
- Red team exercises
- Stress testing solutions

```python
config = {
    "evolution_mode": "adversarial",
    "adversarial_rounds": 5,
    "attack_model_config": {"model_id": "gpt-4"},
    "defense_model_config": {"model_id": "claude-3"}
}
```

### 2. Start with Presets

Always begin with a preset and customize as needed:

```python
from parameter_manager import ParameterManager

pm = ParameterManager()

# For quick testing
config = pm.get_preset("fast")

# For production use
config = pm.get_preset("balanced")

# For research/experimentation
config = pm.get_preset("thorough")

# Add your API key
config["api_key"] = "your_api_key"

# Customize specific parameters
config["max_iterations"] = 30
config["temperature"] = 0.8
```

---

## ⚡ Performance Optimization

### 1. Speed Optimization Strategies

**Use Parallel Processing**
```python
config = {
    "parallel_evaluations": 8,  # Match your CPU cores
    "concurrent_requests": 5,   # Balance with API limits
    "num_workers": 4           # For distributed processing
}
```

**Enable Caching**
```python
config = {
    "cache_evaluations": True,
    "cache_size": 2000,        # Increase for better hit rate
    "incremental_eval": True   # Reuse partial evaluations
}
```

**Use Cascade Evaluation**
```python
config = {
    "cascade_evaluation": True,
    "cascade_thresholds": [0.3, 0.6, 0.8],  # Filter poor solutions early
    "evaluation_timeout": 30   # Prevent hanging evaluations
}
```

**Optimize Population Parameters**
```python
# For speed (good for prototyping)
fast_config = {
    "max_iterations": 10,
    "population_size": 15,
    "early_stopping": True,
    "early_stopping_patience": 5
}

# For balanced performance
balanced_config = {
    "max_iterations": 25,
    "population_size": 30,
    "early_stopping": True,
    "early_stopping_patience": 10
}
```

### 2. Resource Management

**Set Appropriate Limits**
```python
config = {
    "memory_limit_mb": 4096,    # Match your system
    "cpu_limit": 0.8,           # Leave room for other processes
    "cost_limit_usd": 10.0,     # Control API costs
    "api_call_limit": 1000,     # Prevent runaway usage
    "max_time": 1800           # 30 minutes max
}
```

**Monitor Resource Usage**
```python
config = {
    "resource_monitoring": True,
    "profiling_enabled": True,   # For development
    "trace_enabled": True,       # For debugging
    "trace_level": "basic"       # Avoid "full" in production
}
```

### 3. API Optimization

**Configure Timeouts and Retries**
```python
config = {
    "timeout": 60,              # Reasonable timeout
    "max_retries": 3,           # Don't retry too much
    "retry_delay": 2.0,         # Exponential backoff
    "rate_limit": 50            # Respect API limits
}
```

**Use Model Rotation**
```python
config = {
    "model_id": "gpt-4",
    "backup_models": ["gpt-3.5-turbo", "claude-3-sonnet"],
    "model_rotation": True      # Distribute load
}
```

---

## 🎯 Quality Optimization

### 1. Parameter Tuning for Better Results

**Increase Exploration**
```python
config = {
    "temperature": 0.8,         # Higher = more creative
    "exploration_ratio": 0.4,   # More exploration
    "mutation_rate": 0.15,      # Higher mutation
    "diversity_maintenance": True
}
```

**Balance Exploration vs Exploitation**
```python
config = {
    "elite_ratio": 0.15,        # Preserve good solutions
    "exploration_ratio": 0.3,   # Explore new areas
    "exploitation_ratio": 0.45, # Refine good solutions
    "random_ratio": 0.1         # Maintain diversity
}
```

**Use Quality Diversity for Innovation**
```python
config = {
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["complexity", "novelty", "practicality"],
    "archive_size": 200,        # Larger archive for more diversity
    "novelty_threshold": 0.1,   # Lower = more diverse
    "quality_threshold": 0.3    # Minimum quality bar
}
```

### 2. Advanced Quality Techniques

**Enable Meta-Learning**
```python
config = {
    "meta_learning": True,
    "transfer_learning": True,
    "few_shot_adaptation": True,
    "continual_learning": True
}
```

**Use Ensemble Methods**
```python
config = {
    "prompt_ensembling": True,
    "self_consistency": True,
    "ensemble_defense": True    # For adversarial mode
}
```

**Improve Prompt Engineering**
```python
config = {
    "prompt_optimization": True,
    "template_stochasticity": True,
    "meta_prompting": True,
    "chain_of_thought": True,
    "few_shot_examples": 5
}
```

---

## 🛡️ Reliability and Robustness

### 1. Error Handling and Fallbacks

**Configure Robust Fallbacks**
```python
from fallback_handler import FallbackHandler

# Initialize with fallback support
client = OpenEvolveClient(config={
    "api_key": "your_key",
    "backup_models": ["gpt-3.5-turbo", "claude-3-haiku"],
    "max_retries": 3,
    "fallback_enabled": True
})

# Fallback will be used automatically if primary fails
```

**Handle Errors Gracefully**
```python
try:
    result = client.evolve(
        content="Your content here",
        **config
    )
    
    if not result.success:
        print(f"Evolution failed: {result.error}")
        # Handle failure case
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error and use fallback
```

### 2. Monitoring and Alerting

**Track Key Metrics**
```python
from metrics_collector import MetricsCollector

mc = MetricsCollector()

# Start operation tracking
op_id = mc.start_operation(
    operation_id="my_evolution",
    evolution_mode="standard"
)

# Monitor progress
mc.update_operation(op_id.operation_id, 
    iteration=10, 
    best_fitness=0.75
)

# Complete and analyze
mc.end_operation(op_id.operation_id)
metrics = mc.aggregate_metrics()

# Check for issues
if metrics.success_rate < 0.8:
    print("⚠️ Low success rate detected")
if metrics.avg_duration > 300:
    print("⚠️ Operations taking too long")
```

**Set Up Resource Alerts**
```python
from resource_manager import ResourceManager, ResourceLimits

limits = ResourceLimits(
    max_api_calls=500,
    max_cost=5.0,
    max_execution_time_seconds=1800
)

rm = ResourceManager(limits=limits)

# Check limits regularly
within_limits, violations = rm.check_limits()
if not within_limits:
    print(f"⚠️ Resource limits exceeded: {violations}")
```

---

## 🔧 Configuration Management

### 1. Environment-Specific Configurations

**Development Configuration**
```python
dev_config = {
    "max_iterations": 5,        # Fast iterations
    "population_size": 10,      # Small population
    "debug_mode": True,         # Detailed logging
    "trace_enabled": True,      # Full tracing
    "cost_limit_usd": 1.0,     # Low cost limit
    "cache_evaluations": True   # Speed up testing
}
```

**Production Configuration**
```python
prod_config = {
    "max_iterations": 50,       # Thorough search
    "population_size": 40,      # Larger population
    "debug_mode": False,        # Minimal logging
    "trace_enabled": False,     # No tracing overhead
    "cost_limit_usd": 50.0,    # Higher budget
    "resource_monitoring": True, # Monitor resources
    "backup_models": ["gpt-3.5-turbo"]  # Fallbacks
}
```

**Research Configuration**
```python
research_config = {
    "max_iterations": 100,      # Extensive search
    "population_size": 100,     # Large population
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["novelty", "complexity", "feasibility"],
    "archive_size": 500,        # Large archive
    "experimental_features": True,
    "trace_enabled": True,      # Full analysis
    "trace_level": "detailed"
}
```

### 2. Configuration Validation

**Always Validate Before Use**
```python
from parameter_manager import ParameterManager

pm = ParameterManager()

def validate_and_run(config):
    # Add required parameters
    if "api_key" not in config:
        config["api_key"] = os.getenv("OPENEVOLVE_API_KEY")
    
    # Validate configuration
    is_valid, errors = pm.validate_parameters(config)
    
    if not is_valid:
        print(f"❌ Configuration errors: {errors}")
        return None
    
    print("✅ Configuration valid")
    return config

# Use validation
config = validate_and_run(my_config)
if config:
    result = client.evolve(content="...", **config)
```

---

## 📊 Monitoring and Analytics

### 1. Track Evolution Progress

**Real-Time Monitoring**
```python
def monitor_evolution(client, config, content):
    # Start with monitoring
    mc = MetricsCollector()
    op_id = mc.start_operation(
        operation_id=f"evolution_{int(time.time())}",
        evolution_mode=config["evolution_mode"]
    )
    
    try:
        result = client.evolve(content=content, **config)
        
        # Log results
        mc.update_operation(op_id.operation_id,
            best_fitness=result.best_score,
            iterations_completed=result.iterations_completed
        )
        
        return result
        
    finally:
        mc.end_operation(op_id.operation_id)
        
        # Analyze performance
        metrics = mc.aggregate_metrics()
        print(f"Success rate: {metrics.success_rate:.2%}")
        print(f"Average fitness: {metrics.avg_fitness:.3f}")
```

### 2. Performance Analysis

**Export and Analyze Metrics**
```python
# Export metrics for analysis
mc.export_json("evolution_metrics.json")
mc.export_csv("evolution_metrics.csv")

# Analyze trends
metrics = mc.aggregate_metrics()

print(f"""
📊 Evolution Performance Summary:
- Total Operations: {metrics.total_operations}
- Success Rate: {metrics.success_rate:.2%}
- Average Fitness: {metrics.avg_fitness:.3f}
- Average Duration: {metrics.avg_duration:.1f}s
- Total Cost: ${metrics.total_cost_usd:.2f}
""")

# Check for performance issues
if metrics.avg_duration > 120:
    print("⚠️ Consider optimizing for speed")
if metrics.success_rate < 0.9:
    print("⚠️ Consider improving error handling")
```

---

## 🎨 Use Case Specific Best Practices

### 1. Code Generation

```python
code_config = {
    "evolution_mode": "standard",
    "max_iterations": 30,
    "temperature": 0.6,         # Balanced creativity
    "max_tokens": 4000,         # Longer code blocks
    "enable_artifacts": True,   # Code artifacts
    "artifact_validation": True, # Validate syntax
    "prompt_template": "code_generation",
    "few_shot_examples": 3
}
```

### 2. Creative Writing

```python
creative_config = {
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["creativity", "coherence", "style"],
    "temperature": 0.9,         # High creativity
    "archive_size": 150,        # Diverse outputs
    "novelty_threshold": 0.2,   # Encourage novelty
    "max_iterations": 40
}
```

### 3. Problem Solving

```python
problem_config = {
    "evolution_mode": "multi_objective",
    "objectives": ["correctness", "efficiency", "clarity"],
    "max_iterations": 50,
    "population_size": 40,
    "chain_of_thought": True,   # Reasoning steps
    "self_consistency": True    # Multiple approaches
}
```

### 4. Security Analysis

```python
security_config = {
    "evolution_mode": "adversarial",
    "adversarial_rounds": 7,
    "attack_strength": 1.5,
    "defense_strength": 1.2,
    "robustness_metric": "security_score",
    "max_iterations": 25
}
```

---

## 🚨 Common Pitfalls to Avoid

### 1. Configuration Mistakes

❌ **Don't:**
```python
# Missing API key
config = {"evolution_mode": "standard"}

# Ratios that sum > 1.0
config = {
    "elite_ratio": 0.3,
    "exploration_ratio": 0.5,
    "exploitation_ratio": 0.4  # Total = 1.2 > 1.0
}

# Unrealistic resource limits
config = {
    "memory_limit_mb": 100,     # Too low
    "max_time": 10             # Too short
}
```

✅ **Do:**
```python
# Complete configuration
config = {
    "evolution_mode": "standard",
    "api_key": "your_key",
    "elite_ratio": 0.2,
    "exploration_ratio": 0.3,
    "exploitation_ratio": 0.4,
    "random_ratio": 0.1,        # Total = 1.0
    "memory_limit_mb": 4096,    # Realistic
    "max_time": 1800           # 30 minutes
}
```

### 2. Performance Mistakes

❌ **Don't:**
```python
# Disable caching in production
config = {"cache_evaluations": False}

# Use too many concurrent requests
config = {"concurrent_requests": 50}  # Will hit rate limits

# Set unrealistic population sizes
config = {
    "population_size": 1000,    # Too large for most cases
    "max_iterations": 1000     # Unnecessarily long
}
```

✅ **Do:**
```python
# Optimize for your use case
config = {
    "cache_evaluations": True,
    "concurrent_requests": 5,   # Reasonable
    "population_size": 30,      # Balanced
    "max_iterations": 50,       # Sufficient
    "early_stopping": True      # Stop when converged
}
```

### 3. Monitoring Mistakes

❌ **Don't:**
```python
# Ignore errors
result = client.evolve(content="...", **config)
# No error checking

# Skip resource monitoring
# No limits or monitoring set up
```

✅ **Do:**
```python
# Proper error handling and monitoring
try:
    result = client.evolve(content="...", **config)
    
    if not result.success:
        logger.error(f"Evolution failed: {result.error}")
        # Handle gracefully
        
    # Check resource usage
    within_limits, violations = rm.check_limits()
    if not within_limits:
        logger.warning(f"Resource violations: {violations}")
        
except Exception as e:
    logger.exception("Unexpected error in evolution")
    # Use fallback or retry
```

---

## 📈 Scaling Best Practices

### 1. Horizontal Scaling

```python
# Distributed processing configuration
distributed_config = {
    "distributed": True,
    "num_workers": 8,
    "load_balancing": "least_loaded",
    "fault_tolerance": True,
    "communication_backend": "redis",  # For multi-machine
    "heartbeat_interval": 10
}
```

### 2. Vertical Scaling

```python
# High-performance single machine
high_perf_config = {
    "parallel_evaluations": 16,    # Use all cores
    "memory_limit_mb": 16384,      # 16GB
    "cache_size": 5000,            # Large cache
    "batch_size": 2000,            # Large batches
    "concurrent_requests": 10      # More concurrent
}
```

### 3. Cost Optimization at Scale

```python
# Cost-effective scaling
cost_config = {
    "model_id": "gpt-3.5-turbo",   # Cheaper model
    "max_tokens": 1024,            # Limit tokens
    "cache_evaluations": True,     # Reduce API calls
    "early_stopping": True,        # Stop early
    "cost_limit_usd": 100.0,      # Hard limit
    "api_call_limit": 5000        # Call limit
}
```

---

## 🎓 Learning and Experimentation

### 1. Start Small and Iterate

```python
# Phase 1: Basic functionality
basic_config = pm.get_preset("fast")
basic_config["api_key"] = "your_key"

# Phase 2: Add complexity
enhanced_config = basic_config.copy()
enhanced_config.update({
    "max_iterations": 20,
    "temperature": 0.8
})

# Phase 3: Advanced features
advanced_config = enhanced_config.copy()
advanced_config.update({
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["novelty", "quality"]
})
```

### 2. Experiment with Different Modes

```python
# Test different evolution modes on the same problem
modes_to_test = [
    ("standard", {"max_iterations": 20}),
    ("quality_diversity", {"feature_dimensions": ["complexity", "novelty"]}),
    ("multi_objective", {"objectives": ["quality", "efficiency"]})
]

results = {}
for mode, extra_config in modes_to_test:
    config = pm.get_preset("balanced")
    config["evolution_mode"] = mode
    config.update(extra_config)
    config["api_key"] = "your_key"
    
    result = client.evolve(content="test problem", **config)
    results[mode] = result
    
# Compare results
for mode, result in results.items():
    print(f"{mode}: fitness={result.best_score:.3f}")
```

### 3. A/B Testing

```python
def ab_test_configs(content, config_a, config_b, runs=5):
    """Compare two configurations"""
    results_a = []
    results_b = []
    
    for i in range(runs):
        result_a = client.evolve(content=content, **config_a)
        result_b = client.evolve(content=content, **config_b)
        
        results_a.append(result_a.best_score)
        results_b.append(result_b.best_score)
    
    avg_a = sum(results_a) / len(results_a)
    avg_b = sum(results_b) / len(results_b)
    
    print(f"Config A average: {avg_a:.3f}")
    print(f"Config B average: {avg_b:.3f}")
    print(f"Winner: {'A' if avg_a > avg_b else 'B'}")
    
    return avg_a, avg_b

# Test different temperatures
config_low_temp = pm.get_preset("balanced")
config_low_temp["temperature"] = 0.5

config_high_temp = pm.get_preset("balanced") 
config_high_temp["temperature"] = 0.9

ab_test_configs("test problem", config_low_temp, config_high_temp)
```

---

## 📋 Checklist for Production Deployment

### Pre-Deployment
- [ ] Configuration validated with realistic parameters
- [ ] Resource limits set appropriately
- [ ] Fallback mechanisms configured
- [ ] Error handling implemented
- [ ] Monitoring and alerting set up
- [ ] Cost controls in place
- [ ] Integration tests passing
- [ ] Performance benchmarks met

### Deployment
- [ ] API keys securely configured
- [ ] Resource monitoring active
- [ ] Logging configured appropriately
- [ ] Backup models configured
- [ ] Rate limiting respected
- [ ] Health checks implemented

### Post-Deployment
- [ ] Monitor success rates
- [ ] Track performance metrics
- [ ] Review cost usage
- [ ] Analyze quality trends
- [ ] Optimize based on real usage
- [ ] Update configurations as needed

---

Following these best practices will help you get the most out of OpenEvolve while avoiding common pitfalls and ensuring reliable, high-quality results.