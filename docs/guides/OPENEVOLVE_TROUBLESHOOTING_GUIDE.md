# OpenEvolve Troubleshooting Guide

**Common Issues and Solutions for OpenEvolve Integration**

---

## Quick Diagnostics

### Check System Status
```python
from openevolve_client import OpenEvolveClient
from parameter_manager import ParameterManager
from metrics_collector import MetricsCollector

# Test basic imports
try:
    client = OpenEvolveClient(config={"api_key": "test"})
    pm = ParameterManager()
    mc = MetricsCollector()
    print("✅ All core components imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")

# Test parameter validation
config = pm.get_preset("balanced")
config["api_key"] = "your_api_key"
is_valid, errors = pm.validate_parameters(config)
print(f"Parameter validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
if errors:
    print(f"Errors: {errors}")
```

---

## Common Issues

### 1. Import Errors

#### Problem: `ImportError: attempted relative import with no known parent package`
**Cause**: Relative imports in team files when running tests directly

**Solution**:
```python
# Instead of running tests directly, use pytest
python -m pytest test_integration_openevolve.py -v

# Or add current directory to Python path
import sys
import os
sys.path.insert(0, os.getcwd())
```

#### Problem: `ImportError: cannot import name 'WorkflowEngine'`
**Cause**: WorkflowEngine class doesn't exist, only functions available

**Solution**:
```python
# Use available functions instead
from workflow_engine import generate_solution_for_sub_problem
# Instead of trying to import WorkflowEngine class
```

#### Problem: `NameError: name 'datetime' is not defined`
**Cause**: Missing datetime import in ui_components.py

**Solution**: Already fixed - datetime import added to ui_components.py

---

### 2. Parameter Validation Errors

#### Problem: `Required parameter 'api_key' is missing`
**Cause**: API key not provided in configuration

**Solution**:
```python
config = pm.get_preset("balanced")
config["api_key"] = "your_actual_api_key"  # Add this line
is_valid, errors = pm.validate_parameters(config)
```

#### Problem: `Parameter 'evolution_mode' must be one of [...]`
**Cause**: Invalid evolution mode specified

**Solution**:
```python
# Use valid evolution modes
valid_modes = ["standard", "quality_diversity", "multi_objective", "adversarial", "problem_decomposition"]
config["evolution_mode"] = "standard"  # Choose from valid modes
```

#### Problem: `Ratio sum exceeds 1.0`
**Cause**: Selection ratios (elite_ratio + exploration_ratio + exploitation_ratio + random_ratio) > 1.0

**Solution**:
```python
# Ensure ratios sum to ≤ 1.0
config = {
    "elite_ratio": 0.1,
    "exploration_ratio": 0.3,
    "exploitation_ratio": 0.5,
    "random_ratio": 0.1  # Total = 1.0
}
```

---

### 3. API and Connection Issues

#### Problem: `OpenEvolve not available - fallback mode enabled`
**Cause**: OpenEvolve backend not installed or not accessible

**Solution**:
```bash
# Install OpenEvolve backend
pip install openevolve

# Or check if service is running
curl http://localhost:8000/health  # Check if backend is running

# Verify API key and endpoint
export OPENEVOLVE_API_KEY="your_key"
export OPENEVOLVE_API_BASE="https://api.openevolve.com/v1"
```

#### Problem: `API timeout` or `Connection refused`
**Cause**: Network issues or backend unavailable

**Solution**:
```python
# Increase timeout
config = {
    "timeout": 60,  # Increase from default 30s
    "max_retries": 5,  # Increase retries
    "retry_delay": 2.0  # Increase delay between retries
}

# Use fallback configuration
client = OpenEvolveClient(config=config)
# Fallback will be used automatically if backend unavailable
```

#### Problem: `Rate limit exceeded`
**Cause**: Too many API requests

**Solution**:
```python
# Reduce concurrent requests
config = {
    "concurrent_requests": 2,  # Reduce from default 5
    "rate_limit": 30,  # Reduce from default 60
    "request_delay": 1.0  # Add delay between requests
}
```

---

### 4. Memory and Resource Issues

#### Problem: `Memory limit exceeded` or `Out of memory`
**Cause**: Insufficient memory allocation

**Solution**:
```python
# Increase memory limits
config = {
    "memory_limit_mb": 8192,  # Increase from default 4096
    "population_size": 20,  # Reduce if needed
    "max_iterations": 50  # Reduce if needed
}

# Enable memory monitoring
config["resource_monitoring"] = True
```

#### Problem: `CPU limit exceeded`
**Cause**: High CPU usage

**Solution**:
```python
# Adjust CPU settings
config = {
    "cpu_limit": 0.6,  # Reduce from default 0.8
    "parallel_evaluations": 2,  # Reduce parallelism
    "num_workers": 2  # Reduce worker count
}
```

#### Problem: `Cost limit exceeded`
**Cause**: API costs too high

**Solution**:
```python
# Set cost controls
config = {
    "cost_limit_usd": 5.0,  # Set appropriate limit
    "api_call_limit": 500,  # Limit API calls
    "max_tokens": 1024,  # Reduce token usage
    "cache_evaluations": True  # Enable caching
}
```

---

### 5. Evolution Performance Issues

#### Problem: Evolution not converging or poor results
**Cause**: Suboptimal parameters

**Solution**:
```python
# For better convergence
config = {
    "max_iterations": 100,  # Increase iterations
    "population_size": 50,  # Increase population
    "elite_ratio": 0.2,  # Preserve more elites
    "temperature": 0.8,  # Increase exploration
    "early_stopping": True,
    "early_stopping_patience": 20
}

# For quality diversity
config = {
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["complexity", "novelty", "quality"],
    "archive_size": 200,  # Increase archive size
    "novelty_threshold": 0.1
}
```

#### Problem: Evolution too slow
**Cause**: Inefficient configuration

**Solution**:
```python
# Speed optimization
config = {
    "parallel_evaluations": 8,  # Increase parallelism
    "cache_evaluations": True,  # Enable caching
    "cascade_evaluation": True,  # Use cascade evaluation
    "max_iterations": 20,  # Reduce if acceptable
    "population_size": 20  # Reduce if acceptable
}
```

#### Problem: Low diversity in results
**Cause**: Insufficient exploration

**Solution**:
```python
# Increase diversity
config = {
    "temperature": 0.9,  # Higher temperature
    "exploration_ratio": 0.4,  # More exploration
    "diversity_maintenance": True,
    "novelty_search": True,
    "mutation_rate": 0.2  # Higher mutation
}
```

---

### 6. Testing Issues

#### Problem: `TypeError: ModelConfig.__init__() missing required argument 'api_key'`
**Cause**: ModelConfig constructor requires api_key

**Solution**:
```python
# Include api_key in ModelConfig
team = Team(
    name="test_team",
    role="Blue",
    members=[ModelConfig(model_id="gpt-4", temperature=0.7, api_key="test_key")]
)
```

#### Problem: `TypeError: Team.__init__() got unexpected keyword argument 'system_prompt'`
**Cause**: Team class doesn't have system_prompt parameter

**Solution**:
```python
# Use correct Team constructor
team = Team(
    name="test_team",
    role="Blue",  # Required field
    members=[ModelConfig(...)]
    # Don't use system_prompt - use role-specific prompts instead
)
```

#### Problem: `AttributeError: 'MetricsCollector' object has no attribute 'complete_operation'`
**Cause**: Method name changed to end_operation

**Solution**:
```python
# Use correct method name
op_id = mc.start_operation(operation_id="test_op", evolution_mode="standard")
mc.update_operation(op_id.operation_id, iteration=5, best_fitness=0.8)
mc.end_operation(op_id.operation_id)  # Use end_operation, not complete_operation
```

---

### 7. File and Storage Issues

#### Problem: `PermissionError: cannot access file`
**Cause**: File permissions or file in use

**Solution**:
```python
# Use proper file handling
import tempfile
import os

# Create temporary files properly
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    filepath = f.name

try:
    mc.export_json(filepath)
    # Process file
finally:
    if os.path.exists(filepath):
        os.unlink(filepath)  # Clean up
```

#### Problem: Database connection issues
**Cause**: Database not accessible or configuration incorrect

**Solution**:
```python
# Use in-memory database for testing
config = {
    "db_type": "sqlite",
    "db_path": ":memory:",  # In-memory database
    "connection_timeout": 30
}

# Or use file-based database with proper path
config = {
    "db_path": "./openevolve_metrics.db",
    "max_connections": 5
}
```

---

### 8. UI and Visualization Issues

#### Problem: BubbleLab UI warnings in tests
**Cause**: BubbleLab UI components used outside BubbleLab UI app

**Solution**:
```python
# Mock BubbleLab UI for testing
from unittest.mock import patch

with patch('BubbleLab UI.session_state', {}):
    # Run your test code here
    pass

# Or skip UI tests when not in BubbleLab UI environment
import BubbleLab UI as st
if hasattr(st, 'session_state'):
    # Run UI code
    pass
else:
    # Skip or mock UI code
    pass
```

#### Problem: Visualization not displaying
**Cause**: Missing dependencies or configuration

**Solution**:
```bash
# Install visualization dependencies
pip install plotly matplotlib seaborn

# Enable visualization in config
config = {
    "enable_visualization": True,
    "plot_frequency": 10,
    "interactive_plots": True
}
```

---

## Debugging Strategies

### 1. Enable Debug Mode
```python
config = {
    "debug_mode": True,
    "trace_enabled": True,
    "trace_level": "detailed",
    "profiling_enabled": True
}
```

### 2. Check Logs
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# OpenEvolve components will log detailed information
```

### 3. Validate Step by Step
```python
# Test each component individually
pm = ParameterManager()
print("✅ ParameterManager initialized")

config = pm.get_preset("fast")
print("✅ Preset loaded")

config["api_key"] = "test_key"
is_valid, errors = pm.validate_parameters(config)
print(f"✅ Validation: {is_valid}, Errors: {errors}")

mc = MetricsCollector()
print("✅ MetricsCollector initialized")

client = OpenEvolveClient(config=config)
print("✅ OpenEvolveClient initialized")
```

### 4. Use Minimal Configuration
```python
# Start with minimal config and add parameters gradually
minimal_config = {
    "evolution_mode": "standard",
    "max_iterations": 5,
    "population_size": 10,
    "api_key": "test_key"
}

# Test with minimal config first
is_valid, errors = pm.validate_parameters(minimal_config)
print(f"Minimal config valid: {is_valid}")
```

---

## Performance Optimization

### 1. Speed Optimization
```python
fast_config = {
    "max_iterations": 10,  # Reduce iterations
    "population_size": 15,  # Smaller population
    "parallel_evaluations": 8,  # More parallelism
    "cache_evaluations": True,  # Enable caching
    "cascade_evaluation": True,  # Faster evaluation
    "early_stopping": True,  # Stop early if converged
    "early_stopping_patience": 5
}
```

### 2. Quality Optimization
```python
quality_config = {
    "max_iterations": 100,  # More iterations
    "population_size": 50,  # Larger population
    "temperature": 0.8,  # More exploration
    "elite_ratio": 0.15,  # Preserve good solutions
    "diversity_maintenance": True,  # Maintain diversity
    "multi_criteria_eval": True  # Better evaluation
}
```

### 3. Resource Optimization
```python
resource_config = {
    "memory_limit_mb": 2048,  # Conservative memory
    "cpu_limit": 0.6,  # Conservative CPU
    "concurrent_requests": 3,  # Fewer concurrent requests
    "batch_size": 500,  # Smaller batches
    "compression": True,  # Compress data
    "cache_size": 500  # Smaller cache
}
```

---

## Getting Help

### 1. Check Documentation
- Review OPENEVOLVE_INTEGRATION_README.md for usage examples
- Check OPENEVOLVE_API_REFERENCE.md for parameter details
- Look at test files for working examples

### 2. Run Diagnostics
```python
# Run integration tests to check system health
python -m pytest test_integration_openevolve.py -v

# Run specific test for your issue
python -m pytest test_integration_openevolve.py::TestOpenEvolveEndToEndIntegration::test_parameter_combinations -v
```

### 3. Enable Verbose Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 4. Check System Requirements
- Python 3.8+
- Sufficient memory (minimum 4GB recommended)
- Network access for API calls
- Valid API keys for LLM providers

---

## Common Error Messages and Solutions

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Required parameter 'api_key' is missing` | No API key provided | Add `api_key` to config |
| `Evolution mode 'X' not supported` | Invalid evolution mode | Use valid mode from list |
| `Ratio sum exceeds 1.0` | Selection ratios too high | Adjust ratios to sum ≤ 1.0 |
| `Memory limit exceeded` | Insufficient memory | Increase `memory_limit_mb` |
| `API timeout` | Network/backend issues | Increase `timeout`, check connection |
| `Rate limit exceeded` | Too many API calls | Reduce `concurrent_requests` |
| `File permission denied` | File access issues | Check file permissions |
| `Import error` | Missing dependencies | Install required packages |

---

## Best Practices for Troubleshooting

1. **Start Simple**: Use minimal configurations first
2. **Test Incrementally**: Add complexity gradually
3. **Check Logs**: Enable debug logging for detailed information
4. **Validate Parameters**: Always validate configuration before use
5. **Monitor Resources**: Keep track of memory, CPU, and API usage
6. **Use Fallbacks**: Configure fallback mechanisms for reliability
7. **Test Regularly**: Run integration tests to catch issues early

---

This troubleshooting guide covers the most common issues. For additional help, check the example code in the test files and integration documentation.
