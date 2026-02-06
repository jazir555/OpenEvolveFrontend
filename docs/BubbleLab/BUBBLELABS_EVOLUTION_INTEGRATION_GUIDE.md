# BubbleLabs Evolution & Adversarial Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Evolution Workflows](#evolution-workflows)
7. [Adversarial Testing](#adversarial-testing)
8. [MAKER Integration](#maker-integration)
9. [API Reference](#api-reference)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The BubbleLabs Evolution & Adversarial Integration provides a comprehensive UI for interacting with OpenEvolve's evolutionary computation and adversarial testing systems through the BubbleLabs workflow visualization interface.

### Key Capabilities

- **Evolutionary Computation**: Run genetic algorithms with MAKER voting and MDAP decomposition
- **Adversarial Testing**: Red team/blue team testing with coevolution support
- **Real-time Visualization**: Monitor evolution progress and adversarial rounds live
- **Task Management**: Start, stop, pause, and resume long-running tasks
- **Analytics Dashboard**: Comprehensive metrics and performance analysis
- **Template System**: Pre-configured workflows for common scenarios

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   BubbleLabs UI Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Evolution  │  │ Adversarial  │  │  Analytics   │     │
│  │   Controls   │  │   Controls   │  │  Dashboard   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            BubbleLabs Evolution Integration                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Task Manager (Background Threads)                 │  │
│  │  - Evolution Tasks                                   │  │
│  │  - Adversarial Tasks                                 │  │
│  │  - Progress Tracking                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Visualization Engine                              │  │
│  │  - Fitness Plots                                     │  │
│  │  - Diversity Heatmaps                                │  │
│  │  - Population Statistics                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Evolution & Adversarial Engines                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Evolution  │  │  Adversarial │  │   MAKER/     │     │
│  │    Engine    │  │    Engine    │  │    MDAP      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input**: User configures parameters through BubbleLabs UI
2. **Task Creation**: Integration creates EvolutionTask with configuration
3. **Background Execution**: Task runs in separate thread to avoid blocking UI
4. **Progress Updates**: Task updates state with current metrics
5. **Visualization**: UI polls task state and renders visualizations
6. **Completion**: Results stored in history for analysis

---

## Features

### Evolution Features

#### 1. Multiple Evolution Modes

- **Standard Evolution**: Classic genetic algorithm
- **MAKER Voting**: First-to-ahead-by-k voting for zero-error evolution
- **MDAP Decomposition**: Task decomposition for complex problems
- **Hybrid**: Combined MAKER + MDAP approach

#### 2. Population Control

- Configurable population size (2-500)
- Selection methods: Tournament, Roulette, Rank, Steady State
- Elitism for preserving best individuals
- Diversity maintenance

#### 3. Genetic Operators

- **Mutation**: Configurable rate and strength
  - Point mutation
  - Gaussian mutation
  - Uniform mutation
  - Adaptive mutation
- **Crossover**: Configurable rate
  - Single-point crossover
  - Multi-point crossover
  - Uniform crossover

#### 4. Fitness Functions

- Built-in fitness evaluators
- Custom fitness function support
- Multi-objective optimization
- Fitness scaling and normalization

### Adversarial Features

#### 1. Red Team (Attack)

- Vulnerability identification
- Attack generation with MAKER voting
- Attack decomposition for complex exploits
- Multiple attack strategies

#### 2. Blue Team (Defense)

- Defense strategy generation
- MDAP-based defense decomposition
- Ensemble defense methods
- Defense layering

#### 3. Coevolution

- Simultaneous attack/defense evolution
- Adaptive attack strength
- Defense effectiveness tracking
- Multi-round testing

### Visualization Features

#### 1. Real-time Progress

- Generation-wise fitness tracking
- Population diversity monitoring
- Convergence detection
- Live updates

#### 2. Charts and Graphs

- **Fitness Over Generations**: Line plot showing best/average fitness
- **Fitness Distribution**: Histogram of population fitness
- **Diversity Heatmap**: Pairwise distance matrix
- **3D Fitness Landscape**: Surface plot of fitness evolution

#### 3. Metrics Dashboard

- Best fitness achieved
- Average population fitness
- Current generation
- Population diversity
- Convergence status

### Task Management Features

#### 1. Task Control

- Start new tasks
- Stop running tasks
- Pause/resume support
- Multi-task execution

#### 2. Task History

- Complete task log
- Results storage
- Performance comparison
- Replay capability

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install BubbleLab UI plotly pandas numpy

# OpenEvolve dependencies
pip install openai anthropic

# Optional but recommended
pip install asyncio threading
```

### Setup

1. Clone the repository or ensure files are in your Python path:

```python
# bubblelabs_evolution_integration.py
# bubblelabs_evolution_controls.py
# evolution_workflow_templates.py
```

2. Install environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

3. Run the integration:

```bash
BubbleLab UI run bubblelabs_evolution_integration.py
```

---

## Usage Guide

### Quick Start

1. **Launch the Dashboard**:
   ```bash
   BubbleLab UI run bubblelabs_evolution_integration.py
   ```

2. **Navigate to Evolution Workflows**:
   - Click "Evolution Workflows" tab
   - Select evolution type
   - Provide initial content
   - Configure parameters
   - Click "Start Evolution"

3. **Monitor Progress**:
   - Switch to "Active Tasks" tab
   - View real-time progress
   - See fitness plots update live

4. **Analyze Results**:
   - Check "Analytics & Metrics" tab
   - Compare multiple runs
   - View detailed history

### Basic Workflow

1. **Select Mode**: Choose between Evolution or Adversarial testing
2. **Configure Parameters**: Use presets or customize settings
3. **Provide Input**: Enter code, text, or prompt to evolve/test
4. **Start Task**: Launch background execution
5. **Monitor**: Watch real-time progress
6. **Review Results**: Analyze completed tasks

---

## Evolution Workflows

### Evolution Types

#### 1. Standard Evolution

Basic genetic algorithm without MAKER/MDAP enhancements.

**Configuration:**
```python
config = {
    "max_generations": 100,
    "population_size": 20,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
    "selection_method": "tournament"
}
```

**Use Cases:**
- Simple optimization problems
- Quick prototyping
- Educational purposes

#### 2. MAKER Voting Evolution

Uses first-to-ahead-by-k voting for reliable selection (arXiv:2511.09030).

**Configuration:**
```python
config = {
    "enable_maker_voting": True,
    "voting_threshold": 3,  # k for first-to-ahead-by-k
    "adaptive_voting": True,
    "population_size": 25
}
```

**Benefits:**
- Zero-error guarantees
- High-quality selection
- Statistical convergence
- Reduced error rate

**Use Cases:**
- High-stakes optimization
- Critical system evolution
- Zero-error requirements

#### 3. MDAP Decomposition Evolution

Decomposes evolution task into subtasks for efficient search.

**Configuration:**
```python
config = {
    "enable_mdap_decomposition": True,
    "decomposition_depth": 5,
    "max_subtasks": 15
}
```

**Benefits:**
- Handles complex problems
- Parallelizable subtasks
- Better exploration
- Modular optimization

**Use Cases:**
- Large refactoring projects
- Multi-objective optimization
- Complex system design

#### 4. Hybrid Evolution

Combines MAKER voting with MDAP decomposition.

**Configuration:**
```python
config = {
    "enable_maker_voting": True,
    "enable_mdap_decomposition": True,
    "voting_threshold": 3,
    "decomposition_depth": 3
}
```

**Benefits:**
- Best of both approaches
- Zero-error + decomposition
- Scalable to complex problems
- Maximum robustness

### Fitness Functions

#### Built-in Fitness Evaluators

1. **Length-based**: Prefers reasonable content length
2. **Complexity**: Measures structural complexity
3. **Readability**: Evaluates code/text clarity
4. **Performance**: Execution speed (for code)

#### Custom Fitness Functions

Define your own fitness function:

```python
def custom_fitness(content: str) -> float:
    """
    Custom fitness evaluation.

    Args:
        content: Content to evaluate

    Returns:
        Fitness score (0-1, higher is better)
    """
    score = 0.0

    # Your custom logic here
    # Example: count lines, check patterns, etc.

    return min(score, 1.0)
```

Use in UI:
1. Select "custom" fitness type
2. Paste your function code
3. System will use it for evaluation

### Population Management

#### Initialization

- **Initial Individual**: Your provided content
- **Variants**: Mutated versions of initial content
- **All Evaluated**: Every individual gets a fitness score

#### Selection

- **Tournament**: Random subset, pick best
- **Roulette**: Probability proportional to fitness
- **Rank**: Based on fitness ranking
- **Steady State**: Replace worst individuals

#### Reproduction

1. **Parent Selection**: Choose parents based on selection method
2. **Crossover**: Combine parent genomes
3. **Mutation**: Apply random changes
4. **Elitism**: Preserve top individuals

---

## Adversarial Testing

### Adversarial Modes

#### 1. Standard Adversarial

Basic red team/blue team testing.

**Configuration:**
```python
config = {
    "adversarial_rounds": 5,
    "red_team_size": 3,
    "blue_team_size": 3,
    "attack_strength": 0.5
}
```

#### 2. MAKER Red Team

Red team with voting-based attack generation.

**Configuration:**
```python
config = {
    "enable_maker_voting": True,
    "voting_threshold": 3,
    "attack_decomposition": True,
    "red_team_size": 5
}
```

**Benefits:**
- Reliable attack generation
- Zero-error vulnerability detection
- High-quality attacks
- Reduced false positives

#### 3. MDAP Blue Team

Blue team with decomposed defense strategies.

**Configuration:**
```python
config = {
    "enable_mdap_defense": True,
    "max_defenses": 15,
    "defense_layering": True,
    "blue_team_size": 5
}
```

**Benefits:**
- Comprehensive defense coverage
- Layered security
- Thorough vulnerability handling
- Structured defense generation

#### 4. Coevolution

Simultaneous attack/defense evolution.

**Configuration:**
```python
config = {
    "coevolution": True,
    "adversarial_rounds": 10,
    "attack_strength": 0.6,
    "defense_strength": 0.8
}
```

**Process:**
1. Red team generates attacks
2. Blue team generates defenses
3. Evaluate effectiveness
4. Adapt strategies based on results
5. Repeat for multiple rounds

### Adversarial Metrics

#### Attack Metrics

- **Vulnerability Count**: Number of issues found
- **Attack Success Rate**: Percentage of successful attacks
- **Attack Diversity**: Variety of attack types
- **Severity Distribution**: Critical/High/Medium/Low

#### Defense Metrics

- **Fixes Applied**: Number of defenses implemented
- **Defense Success Rate**: Effectiveness of defenses
- **Coverage**: Percentage of attacks defended
- **Robustness Improvement**: System hardening achieved

### Vulnerability Categories

1. **SECURITY_VULNERABILITY**: Security flaws
2. **LOGICAL_ERROR**: Incorrect logic
3. **PERFORMANCE_ISSUE**: Inefficiencies
4. **EDGE_CASE**: Boundary conditions
5. **INPUT_VALIDATION**: Missing checks
6. **ERROR_HANDLING**: Poor error management

---

## MAKER Integration

### What is MAKER?

MAKER (arXiv:2511.09030) is a framework for solving complex LLM tasks with zero errors through:
- **First-to-ahead-by-k Voting**: Statistical consensus
- **Recursive Decomposition**: Break down complex tasks
- **Zero-error Guarantees**: Provable correctness

### MAKER in Evolution

#### Benefits

1. **Reliable Selection**: Voting ensures best individuals selected
2. **Zero-error Evolution**: Statistical convergence guarantees
3. **Adaptive Threshold**: Adjusts based on population diversity
4. **High Confidence**: Voting provides confidence metrics

#### Configuration

```python
maker_config = {
    "enable_maker_voting": True,
    "voting_threshold": 3,  # k value
    "num_candidates": 5,    # N >= 2k-1
    "adaptive_voting": True  # Dynamic adjustment
}
```

#### Parameters

- **voting_threshold (k)**: Consensus requirement
  - Lower (1-2): Faster, less confident
  - Higher (3-5): Slower, more confident
  - Recommendation: 3 for balance

- **num_candidates (N)**: Candidates for voting
  - Must satisfy N >= 2k - 1
  - More candidates = better quality, slower

- **adaptive_voting**: Dynamic threshold adjustment
  - Increases threshold when diversity is high
  - Decreases threshold when diversity is low

### MAKER in Adversarial

#### Red Team Voting

```python
red_team_config = {
    "enable_maker_voting": True,
    "voting_threshold": 3,
    "attack_decomposition": True
}
```

**Benefits:**
- Reliable vulnerability identification
- Zero false positives (with proper k)
- High-quality attack generation

---

## API Reference

### Main Classes

#### BubbleLabsEvolutionIntegration

Main integration class.

```python
integration = BubbleLabsEvolutionIntegration()
```

**Methods:**

- `render_evolution_dashboard()`: Render main dashboard
- `_render_evolution_workflows()`: Render evolution controls
- `_render_adversarial_testing()`: Render adversarial controls
- `_render_active_tasks()`: Show running tasks
- `_render_analytics()`: Display analytics

#### EvolutionControlPanel

Evolution parameter controls.

```python
panel = EvolutionControlPanel()
state = panel.render(key_prefix="my_evo")
```

**Returns:** `EvolutionControlState` with parameters

#### PopulationVisualizer

Population visualization components.

```python
viz = PopulationVisualizer()
viz.render_fitness_distribution(fitness_values, generation)
viz.render_diversity_heatmap(population_data, generation)
```

### Data Structures

#### EvolutionTask

```python
@dataclass
class EvolutionTask:
    task_id: str
    task_type: str  # "evolution" or "adversarial"
    status: EvolutionTaskStatus
    current_generation: int
    max_generations: int
    best_fitness: float
    fitness_history: List[float]
    # ... more fields
```

#### EvolutionControlState

```python
@dataclass
class EvolutionControlState:
    population_size: int
    mutation_rate: float
    crossover_rate: float
    enable_maker_voting: bool
    # ... more fields
```

---

## Examples

### Example 1: Code Optimization

**Goal:** Optimize Python code for performance

```python
# Initial code
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
```

**Configuration:**
1. Select "Code Optimization" template
2. Set generations to 100
3. Enable MAKER voting for reliability
4. Start evolution

**Expected Result:**
- Faster implementation
- Better structure
- List comprehensions
- Type hints

### Example 2: Prompt Refinement

**Goal:** Improve prompt for LLM

**Initial:**
```
"Write code to sort a list."
```

**Configuration:**
1. Select "Prompt Refinement" template
2. Enable MDAP decomposition
3. Set population to 25
4. Run for 75 generations

**Expected Result:**
```
"Write a Python function that sorts a list of numbers in ascending order.
Include examples and handle edge cases like empty lists and duplicates."
```

### Example 3: Security Audit

**Goal:** Find vulnerabilities in code

**Target Code:**
```python
def login(username, password):
    if username == "admin" and password == "12345":
        return True
    return False
```

**Configuration:**
1. Select "Security Audit" template
2. Set 5 adversarial rounds
3. Enable MAKER red team
4. Enable MDAP blue team

**Expected Findings:**
- Hardcoded credentials
- No rate limiting
- Missing authentication
- SQL injection risk

---

## Troubleshooting

### Common Issues

#### 1. Task Not Starting

**Problem:** Clicking start does nothing

**Solutions:**
- Check that initial content is provided
- Verify API keys are set
- Check browser console for errors
- Ensure parameters are valid

#### 2. Slow Progress

**Problem:** Evolution taking too long

**Solutions:**
- Reduce population size
- Decrease max generations
- Lower voting threshold
- Disable MAKER/MDAP for faster runs

#### 3. Poor Results

**Problem:** Fitness not improving

**Solutions:**
- Increase mutation rate
- Adjust crossover rate
- Check fitness function logic
- Enable diversity maintenance
- Try different selection method

#### 4. Memory Issues

**Problem:** Out of memory errors

**Solutions:**
- Reduce population size
- Decrease max_generations
- Clear task history
- Disable fitness history tracking

#### 5. MAKER Voting Too Slow

**Problem:** MAKER voting taking too long

**Solutions:**
- Lower voting threshold (k)
- Reduce num_candidates
- Disable adaptive voting
- Use standard evolution for comparison

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs for:
- Task creation events
- Fitness evaluation calls
- MAKER voting rounds
- Error messages

### Performance Tips

1. **Start Small**: Test with small populations (10-20)
2. **Use Presets**: Start with recommended presets
3. **Monitor Progress**: Watch first few generations
4. **Adjust Parameters**: Tune based on results
5. **Save Templates**: Save good configurations

---

## Best Practices

### Evolution Best Practices

1. **Population Size**
   - Small (10-20): Quick exploration, prototyping
   - Medium (20-50): Balanced performance
   - Large (50-100): Thorough search, complex problems

2. **Mutation Rate**
   - Low (0.01-0.05): Fine-tuning, stable search
   - Medium (0.1-0.2): Balanced exploration
   - High (0.3-0.5): High exploration, disruptive

3. **Generations**
   - 30-50: Quick tests, simple problems
   - 100-150: Standard evolution
   - 200+: Complex problems, thorough search

4. **MAKER Voting**
   - Use for critical tasks
   - Start with k=3
   - Increase k for more confidence
   - Enable adaptive voting

### Adversarial Best Practices

1. **Team Sizes**
   - Small (2-3): Quick assessment
   - Medium (3-5): Balanced coverage
   - Large (5-10): Comprehensive testing

2. **Rounds**
   - 3-5: Quick audit
   - 5-7: Standard testing
   - 10+: Thorough hardening

3. **Attack Strength**
   - Low (0.3-0.5): Gentle testing
   - Medium (0.5-0.7): Realistic attacks
   - High (0.7-1.0): Aggressive testing

4. **Coevolution**
   - Use for system hardening
   - Monitor attack/defense balance
   - Adjust strengths based on results

---

## Advanced Topics

### Custom Workflow Templates

Create your own templates:

```python
from evolution_workflow_templates import WorkflowManager

manager = TemplateManager()

template = manager.create_custom_template(
    name="My Custom Evolution",
    description="Optimize for my specific use case",
    category="evolution",
    config={
        "population_size": 30,
        "max_generations": 150,
        # ... your parameters
    },
    example_content="# Your example",
    use_cases=["Use case 1", "Use case 2"]
)
```

### Integration with Custom Fitness

Define domain-specific fitness:

```python
def code_quality_fitness(code: str) -> float:
    """Evaluate code quality."""
    score = 0.0

    # Check for docstrings
    if '"""' in code or "'''" in code:
        score += 0.2

    # Check for type hints
    if ': ' in code and ' -> ' in code:
        score += 0.3

    # Check for error handling
    if 'try:' in code and 'except' in code:
        score += 0.3

    # Check length
    if 50 <= len(code) <= 1000:
        score += 0.2

    return min(score, 1.0)
```

### Batch Processing

Run multiple evolutions:

```python
configs = [
    {"population_size": 10, "max_generations": 50},
    {"population_size": 20, "max_generations": 100},
    {"population_size": 30, "max_generations": 150},
]

for i, config in enumerate(configs):
    task = create_evolution_task(
        task_id=f"batch_{i}",
        config=config
    )
    start_evolution_task(task)
```

---

## References

### Papers

- **MAKER**: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
  - https://arxiv.org/abs/2511.09030

### Related Documentation

- `evolution.py`: Core evolution engine
- `adversarial.py`: Adversarial testing system
- `evolution_maker_integration.py`: MAKER integration
- `adversarial_maker_integration.py`: MAKER adversarial integration
- `bubblelabs_ui_component.py`: BubbleLabs UI components

### Support

For issues, questions, or contributions:
1. Check troubleshooting section
2. Review example workflows
3. Enable debug logging
4. Check existing GitHub issues

---

## Changelog

### Version 1.0.0
- Initial release
- Evolution workflow support
- Adversarial testing support
- MAKER voting integration
- MDAP decomposition integration
- Real-time visualization
- Task management system
- Analytics dashboard
- Template system

---

**License:** See LICENSE file in repository

**Authors:** OpenEvolve Frontend Team

**Last Updated:** 2025-01-03

