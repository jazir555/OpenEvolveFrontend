# OpenEvolve Integration - Complete Guide

## 📖 Overview

This document provides a comprehensive guide to the OpenEvolve integration in the Decomposition Workflow system. OpenEvolve is a powerful evolution framework that enhances content generation, problem-solving, and quality assessment through advanced evolutionary algorithms.

---

## 🚀 Quick Start

### Basic Usage

```python
from openevolve_client import OpenEvolveClient

# Initialize client
client = OpenEvolveClient(api_key="your_api_key")

# Run standard evolution
result = client.evolve(
    initial_content="Your content here",
    evolution_mode="standard",
    max_iterations=20,
    population_size=30
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best solution: {result['best_solution']}")
```

### Using Presets

```python
from parameter_manager import ParameterManager

pm = ParameterManager()

# Load a preset configuration
config = pm.get_preset("balanced")

# Run evolution with preset
result = client.evolve(
    initial_content="Your content here",
    **config
)
```

---

## 🎯 Key Features

### 1. Evolution Modes

OpenEvolve supports 5 evolution modes:

#### Standard Evolution
Basic evolutionary optimization for general use.
```python
result = client.evolve(
    content="...",
    evolution_mode="standard",
    max_iterations=20,
    population_size=30
)
```

#### Quality Diversity (MAP-Elites)
Generates diverse, high-quality solutions across behavior space.
```python
result = client.evolve(
    content="...",
    evolution_mode="quality_diversity",
    feature_dimensions=["complexity", "novelty", "quality"],
    feature_bins=10,
    archive_size=100
)
```

#### Multi-Objective Optimization
Optimizes multiple objectives simultaneously with Pareto front.
```python
result = client.evolve(
    content="...",
    evolution_mode="multi_objective",
    objectives=["quality", "efficiency", "readability"],
    pareto_front_size=50
)
```

#### Adversarial Evolution
Combines red team critique with blue team improvement.
```python
result = client.evolve(
    content="...",
    evolution_mode="adversarial",
    adversarial_rounds=5,
    red_team_models=["gpt-4", "claude-3"],
    blue_team_models=["gpt-4"]
)
```

#### Coevolution
Multiple populations evolve together with interactions.
```python
result = client.evolve(
    content="...",
    evolution_mode="coevolution",
    num_populations=3,
    interaction_frequency=5
)
```

### 2. Parameter Configuration

OpenEvolve provides 211 configurable parameters across 19 categories:

```python
from ui_config import OPENEVOLVE_PARAMS

# View all parameters
for category, params in OPENEVOLVE_PARAMS.items():
    print(f"\n{category}:")
    for param_name, param_info in params.items():
        print(f"  - {param_name}: {param_info['description']}")
```

### 3. Metrics Collection

Track comprehensive metrics during evolution:

```python
from metrics_collector import MetricsCollector

mc = MetricsCollector()

# Start tracking an operation
op_id = mc.start_operation(
    evolution_mode="standard",
    parameters={"max_iterations": 20}
)

# Update metrics during evolution
mc.update_operation(op_id, iteration=10, best_fitness=0.85)

# Complete operation
mc.complete_operation(op_id, success=True, final_fitness=0.92)

# Get aggregated metrics
metrics = mc.aggregate_metrics()
print(f"Success rate: {metrics['success_rate']}")
```

### 4. Visualization

Visualize evolution progress and results:

```python
import BubbleLab UI as st
from analytics_dashboard import (
    render_openevolve_metrics_dashboard,
    render_fitness_evolution_plot,
    render_diversity_heatmap,
    render_pareto_front
)

# Render metrics dashboard
render_openevolve_metrics_dashboard(metrics_data)

# Render fitness evolution
render_fitness_evolution_plot(evolution_history)

# Render quality diversity archive
render_diversity_heatmap(archive_data, feature_dimensions)

# Render Pareto front
render_pareto_front(solutions, objective_names)
```

---

## 📊 Configuration Guide

### Presets

Four optimized presets are available:

#### Fast (⚡ Quick Prototyping)
```python
{
    "max_iterations": 5,
    "population_size": 10,
    "temperature": 0.5,
    "elite_ratio": 0.2,
    "exploration_ratio": 0.3,
    "exploitation_ratio": 0.5
}
```

#### Balanced (⚖️ General Use)
```python
{
    "max_iterations": 20,
    "population_size": 30,
    "temperature": 0.7,
    "elite_ratio": 0.15,
    "exploration_ratio": 0.35,
    "exploitation_ratio": 0.5,
    "enable_cascade_evaluation": True
}
```

#### Thorough (🎯 Production Use)
```python
{
    "max_iterations": 50,
    "population_size": 50,
    "temperature": 0.8,
    "enable_quality_diversity": True,
    "feature_dimensions": ["complexity", "novelty", "quality"],
    "feature_bins": 10
}
```

#### Research (🔬 Maximum Exploration)
```python
{
    "max_iterations": 100,
    "population_size": 100,
    "temperature": 0.9,
    "enable_quality_diversity": True,
    "enable_island_model": True,
    "num_islands": 4,
    "enable_meta_prompting": True
}
```

### Custom Configuration

Create custom configurations:

```python
from template_manager import TemplateManager

tm = TemplateManager()

# Create custom template based on preset
custom_id = tm.create_custom_openevolve_template(
    name="My Custom Config",
    description="Optimized for my use case",
    base_preset="balanced",
    overrides={
        "max_iterations": 30,
        "temperature": 0.75,
        "enable_artifacts": True
    }
)

# Load and use custom template
config = tm.get_template(custom_id)
```

---

## 🔧 Advanced Features

### 1. Island Model Evolution

Run multiple populations in parallel with migration:

```python
result = client.evolve(
    content="...",
    evolution_mode="standard",
    enable_island_model=True,
    num_islands=4,
    migration_interval=10,
    migration_size=5,
    migration_topology="ring"
)
```

### 2. Cascade Evaluation

Evaluate solutions in stages for efficiency:

```python
result = client.evolve(
    content="...",
    enable_cascade_evaluation=True,
    cascade_thresholds=[0.5, 0.75, 0.9],
    parallel_evaluations=8
)
```

### 3. Ensemble Evaluation

Use multiple evaluators for robust assessment:

```python
from evaluator_team import EvaluatorTeam

evaluator = EvaluatorTeam(team=evaluator_team, api_key=api_key)

assessment = evaluator.evaluate_with_ensemble(
    content="...",
    criteria={"quality": "overall quality", "correctness": "correctness"},
    consensus_threshold=0.7
)
```

### 4. Quality Diversity Analysis

Analyze content from diverse perspectives:

```python
from content_analyzer import analyze_with_quality_diversity

result = analyze_with_quality_diversity(
    content="...",
    api_key=api_key,
    feature_dimensions=["technical_depth", "readability", "completeness"],
    max_iterations=20,
    archive_size=100
)

# Access diverse analyses
for dim, analyses in result['analyses_by_dimension'].items():
    print(f"\n{dim}:")
    for analysis in analyses[:3]:  # Top 3
        print(f"  Fitness: {analysis['fitness']:.3f}")
        print(f"  Analysis: {analysis['analysis'][:200]}...")
```

### 5. Resource Management

Enforce resource limits:

```python
from resource_manager import ResourceManager, ResourceLimits

limits = ResourceLimits(
    max_api_calls=1000,
    max_cost=50.0,
    max_execution_time=3600
)

rm = ResourceManager(limits=limits)

# Track resources during evolution
rm.track_openevolve_operation(
    operation_type="evolve",
    metrics=result['metrics']
)

# Check if within limits
within_limits, violations = rm.check_limits()
if not within_limits:
    print(f"Resource limits exceeded: {violations}")
```

---

## 📈 Monitoring & Analytics

### Real-Time Monitoring

```python
import BubbleLab UI as st
from ui_components import render_openevolve_progress_monitor

# Monitor active operation
render_openevolve_progress_monitor(
    operation_id=op_id,
    metrics_collector=mc,
    auto_refresh=True
)
```

### Historical Analysis

```python
from workflow_history_manager import WorkflowHistoryManager

whm = WorkflowHistoryManager()

# Get OpenEvolve metrics from history
metrics = whm.get_openevolve_metrics_history()

print(f"Total workflows: {metrics['total_workflows']}")
print(f"Workflows with OpenEvolve: {metrics['workflows_with_openevolve']}")
print(f"Average fitness improvement: {metrics['average_fitness_improvement']:.3f}")

# Query by performance
high_performers = whm.query_workflows_by_metrics(
    min_fitness=0.8,
    min_diversity=0.6
)
```

### Time-Series Analysis

```python
from analytics_data import get_openevolve_time_series

# Get metrics over last 30 days
time_series = get_openevolve_time_series(mc, days=30)

# Visualize trends
import plotly.express as px
fig = px.line(time_series, x='date', y='fitness_improvement')
st.plotly_chart(fig)
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest test_openevolve_integration.py -v

# Run specific test class
pytest test_openevolve_integration.py::TestParameterManager -v

# Run with coverage
pytest test_openevolve_integration.py --cov=. --cov-report=html
```

Example test:

```python
def test_parameter_validation():
    from parameter_manager import ParameterManager
    
    pm = ParameterManager()
    
    params = {
        'max_iterations': 10,
        'population_size': 20,
        'temperature': 0.7
    }
    
    is_valid, errors = pm.validate_parameters(params)
    assert is_valid
    assert len(errors) == 0
```

---

## 🔌 Integration Examples

### Team Integration

```python
from blue_team import BlueTeam
from workflow_structures import Team, ModelConfig

# Create team
team = Team(
    name="solver_team",
    members=[ModelConfig(model_id="gpt-4", temperature=0.7)],
    system_prompt="You are an expert problem solver."
)

blue_team = BlueTeam(team=team, api_key=api_key)

# Generate solution with OpenEvolve
solution = blue_team.generate_solution_with_openevolve(
    problem="Solve this problem...",
    evolution_mode="standard",
    max_iterations=20
)
```

### Workflow Integration

```python
from workflow_engine import WorkflowEngine

engine = WorkflowEngine()

# Run content analysis with OpenEvolve
analysis = engine.run_content_analysis_with_openevolve(
    content="Content to analyze...",
    evolution_mode="quality_diversity"
)

# Run decomposition with OpenEvolve
decomposition = engine.run_decomposition_with_openevolve(
    problem="Complex problem...",
    analyzed_context=analysis
)
```

### External Knowledge Integration

```python
from external_knowledge_integration import get_knowledge_integration_manager

# Get manager
manager = get_knowledge_integration_manager()

# Query all sources
context = {
    "query": "machine learning optimization",
    "domain": "AI",
    "limit": 10
}

results = manager.query_all_connectors(context)

for source, items in results.items():
    print(f"\n{source}: {len(items)} items")
    for item in items[:3]:
        print(f"  - {item.content[:100]}...")
```

---

## 🎨 UI Components

### Configuration Panel

```python
import BubbleLab UI as st
from ui_components import render_openevolve_config_panel

# Render configuration panel
config = render_openevolve_config_panel(session_key="my_config")

# Use configuration
result = client.evolve(content="...", **config)
```

### Progress Monitor

```python
from ui_components import render_openevolve_progress_monitor

# Monitor evolution progress
render_openevolve_progress_monitor(
    operation_id=op_id,
    metrics_collector=mc,
    auto_refresh=True
)
```

### Archive Viewer

```python
from ui_components import render_quality_diversity_archive

# View quality diversity archive
render_quality_diversity_archive(
    archive_data=result['archive'],
    feature_dimensions=["complexity", "novelty", "quality"]
)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Parameter Validation Errors

```python
# Check parameter validity
from parameter_manager import ParameterManager

pm = ParameterManager()
is_valid, errors = pm.validate_parameters(your_params)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

#### 2. Resource Limit Exceeded

```python
# Check resource usage
from resource_manager import ResourceManager

rm = ResourceManager(limits=your_limits)
within_limits, violations = rm.check_limits()

if not within_limits:
    print("Resource violations:")
    for violation in violations:
        print(f"  - {violation}")
```

#### 3. Evolution Not Converging

```python
# Adjust parameters for better convergence
config = {
    "max_iterations": 50,  # Increase iterations
    "population_size": 50,  # Increase population
    "elite_ratio": 0.2,     # Preserve more elites
    "exploration_ratio": 0.3,  # Reduce exploration
    "exploitation_ratio": 0.5  # Increase exploitation
}
```

#### 4. Low Diversity

```python
# Increase diversity
config = {
    "temperature": 0.9,  # Higher temperature
    "exploration_ratio": 0.5,  # More exploration
    "enable_quality_diversity": True,
    "feature_dimensions": ["complexity", "novelty", "quality"],
    "novelty_threshold": 0.2  # Higher novelty requirement
}
```

---

## 📚 API Reference

### OpenEvolveClient

```python
class OpenEvolveClient:
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize OpenEvolve client"""
        
    def evolve(
        self,
        initial_content: str,
        evolution_mode: str = "standard",
        max_iterations: int = 20,
        population_size: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Run evolution"""
        
    def validate_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate parameters"""
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics"""
```

### ParameterManager

```python
class ParameterManager:
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration"""
        
    def validate_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate parameters"""
        
    def save_parameters(
        self,
        parameters: Dict[str, Any],
        name: str
    ) -> None:
        """Save parameters"""
        
    def load_parameters(self, name: str) -> Dict[str, Any]:
        """Load parameters"""
```

### MetricsCollector

```python
class MetricsCollector:
    def start_operation(
        self,
        evolution_mode: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Start tracking operation"""
        
    def update_operation(
        self,
        operation_id: str,
        **metrics
    ) -> None:
        """Update operation metrics"""
        
    def complete_operation(
        self,
        operation_id: str,
        success: bool,
        **final_metrics
    ) -> None:
        """Complete operation"""
        
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        
    def export_to_json(self) -> str:
        """Export metrics to JSON"""
```

---

## 🎓 Best Practices

### 1. Start with Presets

Begin with a preset and customize as needed:

```python
config = pm.get_preset("balanced")
config["max_iterations"] = 30  # Customize
```

### 2. Monitor Resource Usage

Always track resource usage to avoid exceeding limits:

```python
rm = ResourceManager(limits=ResourceLimits(max_cost=50.0))
# Track throughout execution
```

### 3. Use Quality Diversity for Exploration

When you need diverse solutions, use quality diversity:

```python
result = client.evolve(
    content="...",
    evolution_mode="quality_diversity",
    feature_dimensions=["complexity", "novelty", "quality"]
)
```

### 4. Enable Cascade Evaluation for Efficiency

Save resources with cascade evaluation:

```python
config["enable_cascade_evaluation"] = True
config["cascade_thresholds"] = [0.5, 0.75, 0.9]
```

### 5. Validate Parameters Before Evolution

Always validate parameters to catch errors early:

```python
is_valid, errors = pm.validate_parameters(config)
if not is_valid:
    print(f"Invalid parameters: {errors}")
    return
```

---

## 📖 Additional Resources

- **Parameter Reference:** See `ui_config.py` for all 211 parameters
- **Test Examples:** See `test_openevolve_integration.py` for usage examples
- **Visualization Examples:** See `analytics_dashboard.py` for visualization code
- **Integration Examples:** See team files (`blue_team.py`, `red_team.py`, `evaluator_team.py`)

---

## 🤝 Contributing

When contributing to OpenEvolve integration:

1. **Follow Best Practices:** Type hints, docstrings, error handling
2. **Write Tests:** Add tests for new functionality
3. **Validate Parameters:** Ensure all parameters are validated
4. **Document Changes:** Update this README and relevant docs
5. **Check Diagnostics:** Ensure code passes all checks

---

## 📄 License

This OpenEvolve integration is part of the Decomposition Workflow system.

---

## 🙏 Acknowledgments

This integration implements comprehensive OpenEvolve capabilities including:
- 5 evolution modes
- 211 configurable parameters
- Comprehensive metrics collection
- Rich visualizations
- Team and workflow integration
- External knowledge integration
- Resource management
- Template management

For questions or issues, please refer to the test suite and example code provided throughout this guide.

