"""
Configuration File Example - Using YAML Config

This example demonstrates how to use a YAML configuration file
instead of passing parameters programmatically.

Problem: Optimize a mathematical function with custom settings
"""

# EVOLVE-BLOCK-START
def optimize(x, y):
    """Find maximum of f(x,y) = -(x^2 + y^2) + 4x + 6y"""
    # Starting point - not optimal
    result = -(x**2 + y**2) + 4*x + 6*y
    return result
# EVOLVE-BLOCK-END


"""
CONFIGURATION FILE (config.yaml):
--------------------------------
Create a file named config.yaml with the following content:

```yaml
# LLM Configuration
llm:
  api_base: "https://api.openai.com/v1"
  models:
    - name: "gpt-4"
      api_key: "${OPENAI_API_KEY}"  # From environment
      weight: 1.0
      temperature: 0.7
      max_tokens: 2048

# Evolution Parameters
max_iterations: 50
checkpoint_interval: 10
log_level: "INFO"

# Database Settings
database:
  population_size: 100
  num_islands: 3
  feature_dimensions: ["complexity", "diversity"]

# Early Stopping
early_stopping_patience: 10
convergence_threshold: 0.01
```

HOW TO RUN:
----------
Using CLI with config file:
```bash
openevolve optimize.py optimize_evaluator.py --config config.yaml
```

Using Python API:
```python
from openevolve import run_evolution

result = run_evolution(
    'optimize.py',
    'optimize_evaluator.py',
    config='config.yaml',  # Path to YAML file
    iterations=50
)

print(f"Best solution found:")
print(result.best_code)
print(f"Score: {result.best_score:.4f}")
```

BENEFITS OF CONFIG FILES:
------------------------
1. Reproducibility - Same settings every time
2. Version control - Track configuration changes
3. Team collaboration - Share settings easily
4. Environment-specific - Different configs for dev/prod
5. No hardcoded values - Clean, maintainable code
"""
