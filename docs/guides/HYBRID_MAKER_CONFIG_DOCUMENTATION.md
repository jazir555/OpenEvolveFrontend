# Hybrid MAKER Configuration System

## Overview

The Hybrid MAKER Configuration System provides comprehensive configuration management for hybrid MAKER strategies that combine multiple problem-solving approaches including:

- **LeanAide**: Lean theorem proving and verification
- **MAKER**: Multi-agent voting and consensus
- **MCTS**: Monte Carlo Tree Search exploration
- **Evolution**: Evolutionary optimization
- **MDAP**: Multi-agent Decomposition and Assembly

## File Structure

```
hybrid_maker_config.py (1,458 lines)
├── Configuration Classes
│   ├── LeanAideConfig
│   ├── MakerConfig
│   ├── MCTSConfig
│   ├── EvolutionConfig
│   ├── MDAPConfig
│   ├── HybridStrategyProfile
│   ├── AdaptiveConfig
│   ├── PerformanceThresholds
│   └── HybridMakerConfig (main class)
│
├── Preset Configurations
│   └── HybridMakerConfigPreset
│       ├── fast()
│       ├── balanced()
│       ├── thorough()
│       ├── leanaide_focused()
│       ├── maker_focused()
│       ├── adaptive()
│       └── research()
│
└── Utility Functions
    ├── validate_config()
    ├── estimate_runtime()
    ├── estimate_resource_usage()
    ├── load_from_file()
    ├── save_to_file()
    ├── merge_configs()
    ├── compare_configs()
    └── export_config_summary()
```

## Quick Start

### Using Presets

```python
from hybrid_maker_config import HybridMakerConfigPreset

# Fast exploration (minimal computation)
config = HybridMakerConfigPreset.fast()

# Balanced configuration (recommended)
config = HybridMakerConfigPreset.balanced()

# Thorough (maximum quality)
config = HybridMakerConfigPreset.thorough()

# Focused configurations
config = HybridMakerConfigPreset.leanaide_focused()
config = HybridMakerConfigPreset.maker_focused()
config = HybridMakerConfigPreset.adaptive()
config = HybridMakerConfigPreset.research()
```

### Custom Configuration

```python
from hybrid_maker_config import HybridMakerConfig, StrategyType

# Create custom configuration
config = HybridMakerConfig(
    config_name="my_custom_config",
    description="Custom research configuration",
    default_strategy=StrategyType.MAKER,
    global_timeout=7200,  # 2 hours
)

# Customize strategy settings
config.maker_config.k_min = 4
config.maker_config.k_max = 7
config.maker_config.min_agents = 5

config.mcts_config.num_simulations = 1500
config.mcts_config.num_workers = 6

# Validate
valid, errors = config.validate()
if not valid:
    print(f"Configuration errors: {errors}")
```

### Saving and Loading

```python
# Save to YAML
config.save_to_file("config.yaml", format="yaml")

# Save to JSON
config.save_to_file("config.json", format="json")

# Load from file
loaded_config = HybridMakerConfig.load_from_file("config.yaml")
```

## Configuration Classes

### 1. LeanAideConfig

Configuration for Lean theorem proving and verification strategy.

**Parameters:**
- `server_url` (str): LeanAide server URL (default: "http://localhost:8080")
- `timeout` (int): Request timeout in seconds (default: 30)
- `max_retries` (int): Maximum retry attempts (default: 3)
- `retry_delay` (float): Delay between retries in seconds (default: 1.0)
- `verify_tactics` (bool): Enable tactic verification (default: True)
- `verify_theorems` (bool): Enable theorem verification (default: True)
- `strict_verification` (bool): Use strict verification mode (default: False)
- `parallel_verifications` (int): Number of parallel verifications (default: 1)
- `cache_verification_results` (bool): Cache verification results (default: True)

### 2. MakerConfig

Configuration for MAKER voting strategy.

**Parameters:**
- `k_min` (int): Minimum k-ahead value (default: 2)
- `k_max` (int): Maximum k-ahead value (default: 8)
- `max_votes_per_step` (int): Maximum votes to collect (default: 50)
- `max_steps` (int): Maximum number of steps (default: 1000)
- `min_agents` (int): Minimum number of agents (default: 3)
- `max_agents` (int): Maximum number of agents (default: 10)
- `timeout_seconds` (int): Timeout per step (default: 60)
- `checkpoint_interval` (int): Checkpoint frequency (default: 25)
- `min_confidence` (float): Minimum confidence threshold (default: 0.2)
- `fallback_policy` (str): Fallback behavior (default: "escalate_then_best_effort")

### 3. MCTSConfig

Configuration for Monte Carlo Tree Search strategy.

**Parameters:**
- `num_simulations` (int): Number of MCTS simulations (default: 1000)
- `exploration_constant` (float): UCB1 exploration constant (default: 1.414)
- `discount_factor` (float): Reward discount factor (default: 0.99)
- `max_tree_depth` (int): Maximum tree depth (default: 100)
- `selection_policy` (str): Node selection policy (default: "ucb1")
- `max_actions_per_node` (int): Maximum actions per node (default: 10)
- `parallel_simulations` (bool): Enable parallel simulations (default: True)
- `num_workers` (int): Number of parallel workers (default: 4)
- `rollout_depth` (int): Rollout simulation depth (default: 10)

### 4. EvolutionConfig

Configuration for evolutionary optimization strategy.

**Parameters:**
- `population_size` (int): Population size (default: 100)
- `generations` (int): Number of generations (default: 100)
- `elite_ratio` (float): Elite population ratio (default: 0.1)
- `mutation_rate` (float): Mutation probability (default: 0.1)
- `mutation_strength` (float): Mutation magnitude (default: 0.2)
- `crossover_rate` (float): Crossover probability (default: 0.7)
- `selection_method` (str): Selection method (default: "tournament")
- `num_islands` (int): Number of islands for island model (default: 1)
- `migration_interval` (int): Migration frequency (default: 20)
- `migration_rate` (float): Migration rate (default: 0.1)

### 5. MDAPConfig

Configuration for Multi-agent Decomposition and Assembly strategy.

**Parameters:**
- `decomposition_depth` (int): Decomposition depth (default: 3)
- `min_subproblem_size` (int): Minimum subproblem size (default: 5)
- `max_subproblem_size` (int): Maximum subproblem size (default: 50)
- `agent_count` (int): Number of agents (default: 5)
- `agent_specialization` (bool): Enable agent specialization (default: True)
- `max_retries_per_task` (int): Maximum task retries (default: 3)
- `task_timeout` (int): Task timeout in seconds (default: 120)
- `parallel_tasks` (int): Number of parallel tasks (default: 3)
- `assembly_strategy` (str): Assembly method (default: "hierarchical")
- `quality_threshold` (float): Quality threshold (default: 0.8)

### 6. HybridStrategyProfile

Profile configuration for individual strategies.

**Parameters:**
- `strategy_type` (StrategyType): Strategy type
- `enabled` (bool): Whether strategy is enabled (default: True)
- `performance_weight` (float): Performance weight (default: 1.0)
- `priority` (int): Execution priority (default: 0)
- `cpu_allocation` (float): CPU allocation fraction (default: 1.0)
- `memory_allocation_mb` (int): Memory allocation in MB (default: None)
- `max_time_seconds` (int): Maximum execution time (default: None)
- `parallel_instances` (int): Number of parallel instances (default: 1)

### 7. AdaptiveConfig

Configuration for adaptive strategy selection.

**Parameters:**
- `enable_adaptive_selection` (bool): Enable adaptive selection (default: True)
- `adaptation_interval` (int): Adaptation check interval (default: 10)
- `track_performance_history` (bool): Track performance (default: True)
- `performance_history_size` (int): History size (default: 100)
- `resource_aware` (bool): Enable resource-aware adaptation (default: True)
- `allow_strategy_combination` (bool): Allow combining strategies (default: True)
- `max_combined_strategies` (int): Maximum combined strategies (default: 3)

### 8. PerformanceThresholds

Performance thresholds for strategy switching.

**Parameters:**
- `fast_time_threshold` (int): Fast mode timeout in seconds (default: 60)
- `balanced_time_threshold` (int): Balanced mode timeout in seconds (default: 300)
- `thorough_time_threshold` (int): Thorough mode timeout in seconds (default: 1800)
- `fast_quality_threshold` (float): Fast mode quality target (default: 0.6)
- `balanced_quality_threshold` (float): Balanced mode quality target (default: 0.8)
- `thorough_quality_threshold` (float): Thorough mode quality target (default: 0.95)

## Preset Configurations

### FAST

Minimal computation for quick exploration.

- MAKER: k_min=2, k_max=3, max_votes=10
- MCTS: 100 simulations
- Evolution: 20 population, 10 generations
- MDAP: depth=2, 3 agents
- Timeout: 300 seconds

**Use when:**
- Quick prototyping
- Early exploration
- Limited time/resources

### BALANCED

Good trade-off between speed and quality (recommended).

- MAKER: k_min=3, k_max=6, max_votes=30
- MCTS: 500 simulations
- Evolution: 50 population, 30 generations
- MDAP: depth=3, 5 agents
- Timeout: 1800 seconds

**Use when:**
- General problem solving
- Production workloads
- Unknown problem complexity

### THOROUGH

Maximum quality with extensive computation.

- MAKER: k_min=5, k_max=10, max_votes=100
- MCTS: 5000 simulations
- Evolution: 200 population, 100 generations
- MDAP: depth=5, 10 agents
- Timeout: 14400 seconds (4 hours)

**Use when:**
- Critical problems
- Maximum quality required
- Sufficient time/resources

### LEANAIDE_FOCUSED

Emphasize Lean theorem proving verification.

- LeanAide strict verification enabled
- Only LeanAide strategy enabled
- Extensive proof traces

**Use when:**
- Mathematical proofs required
- Formal verification needed
- Lean theorem proving available

### MAKER_FOCUSED

Emphasize multi-agent voting consensus.

- High MAKER k values
- MDAP for decomposition support
- Extensive voting

**Use when:**
- Consensus critical
- Multiple perspectives valuable
- Voting-based approach preferred

### ADAPTIVE

Automatic strategy selection based on performance.

- All strategies enabled
- Adaptive selection enabled
- Resource-aware adaptation
- Strategy combination allowed

**Use when:**
- Unknown best strategy
- Dynamic problem characteristics
- Want automatic optimization

### RESEARCH

Research configuration for experimentation.

- All advanced features enabled
- Extensive logging
- High resource allocation
- Long execution time

**Use when:**
- Experimenting with strategies
- Analyzing performance
- Developing new approaches

## Utility Functions

### validate_config()

Validate configuration consistency.

```python
valid, errors = config.validate()
if not valid:
    for error in errors:
        print(f"Error: {error}")
```

### estimate_runtime()

Estimate execution time for strategies.

```python
# Estimate for all strategies
runtime = config.estimate_runtime()
for strategy, time in runtime.items():
    print(f"{strategy}: ~{time:.1f}s")

# Estimate for single strategy
runtime = config.estimate_runtime(StrategyType.MAKER)
print(f"MAKER: ~{runtime['maker']:.1f}s")
```

### estimate_resource_usage()

Estimate CPU and memory requirements.

```python
usage = config.estimate_resource_usage()
for strategy, resources in usage.items():
    print(f"{strategy}:")
    print(f"  CPU: {resources['cpu']:.1f} cores")
    print(f"  Memory: {resources['memory_mb']:.0f} MB")
```

### merge_configs()

Merge multiple configurations.

```python
from hybrid_maker_config import merge_configs

merged = merge_configs(config1, config2, config3)
# Later configs take precedence
```

### compare_configs()

Compare two configurations.

```python
from hybrid_maker_config import compare_configs

diff = compare_configs(config1, config2)
print(f"Changed: {diff['changed']}")
print(f"Added: {diff['added']}")
print(f"Removed: {diff['removed']}")
```

### export_config_summary()

Export human-readable configuration summary.

```python
from hybrid_maker_config import export_config_summary

summary = export_config_summary(config)
print(summary)
```

## Testing

Comprehensive test suite with 75 tests covering:

- Configuration validation
- Serialization/deserialization
- Preset configurations
- Runtime estimation
- Resource estimation
- File I/O
- Edge cases

Run tests:

```bash
python test_hybrid_maker_config.py
```

## Examples

See `hybrid_maker_config_example.py` for comprehensive usage examples including:

1. Using predefined presets
2. Creating custom configurations
3. Runtime and resource estimation
4. Saving and loading configurations
5. Working with strategy profiles
6. Using focused configurations
7. Configuring performance thresholds
8. Exporting configuration summaries

Run examples:

```bash
python hybrid_maker_config_example.py
```

## Configuration File Format

### YAML Example

```yaml
config_name: my_config
description: Custom configuration
default_strategy: maker
global_timeout: 7200

leanaide_config:
  server_url: http://localhost:8080
  timeout: 30
  max_retries: 3
  verify_tactics: true

maker_config:
  k_min: 4
  k_max: 7
  max_votes_per_step: 40
  min_agents: 5
  timeout_seconds: 60

mcts_config:
  num_simulations: 1500
  exploration_constant: 1.5
  num_workers: 6

evolution_config:
  population_size: 100
  generations: 50
  mutation_rate: 0.15

mdap_config:
  decomposition_depth: 3
  agent_count: 5
  parallel_tasks: 2

strategy_profiles:
  maker:
    enabled: true
    performance_weight: 1.0
    priority: 1
  mcts:
    enabled: true
    performance_weight: 0.8
    priority: 2
```

### JSON Example

```json
{
  "config_name": "my_config",
  "description": "Custom configuration",
  "default_strategy": "maker",
  "global_timeout": 7200,
  "maker_config": {
    "k_min": 4,
    "k_max": 7,
    "max_votes_per_step": 40
  },
  "strategy_profiles": {
    "maker": {
      "enabled": true,
      "performance_weight": 1.0,
      "priority": 1
    }
  }
}
```

## Best Practices

1. **Start with presets**: Use predefined presets as starting points
2. **Validate always**: Always validate configurations before use
3. **Estimate resources**: Check runtime and resource estimates for large problems
4. **Save configurations**: Save working configurations for reproducibility
5. **Use checkpoints**: Enable checkpoints for long-running tasks
6. **Monitor performance**: Use adaptive mode for unknown problem characteristics
7. **Tune gradually**: Make small adjustments and measure impact
8. **Document changes**: Keep descriptions and tags for configuration management

## Integration with Existing Code

The configuration system integrates with existing OpenEvolve components:

```python
# With MAKER engine
from maker_engine import MakerEngine
from hybrid_maker_config import HybridMakerConfigPreset

config = HybridMakerConfigPreset.balanced()
maker_cfg = config.maker_config

# Use with existing MAKER code
team = Team(members=[...])
maker = MakerEngine(team, maker_cfg)

# With MDAP engine
from mdap_engine import MDAPOrchestrator

mdap_cfg = config.mdap_config
orchestrator = MDAPOrchestrator(team, mdap_cfg)
```

## Performance Considerations

- **FAST preset**: ~60-300 seconds, 1-2 CPU cores, ~500-2000 MB
- **BALANCED preset**: ~300-1800 seconds, 2-4 CPU cores, ~2000-4000 MB
- **THOROUGH preset**: ~1800-14400 seconds, 4-8 CPU cores, ~4000-16000 MB

Actual performance depends on:
- Problem complexity
- LLM API response times
- Network latency
- System resources
- Parallel execution settings

## Future Enhancements

Potential additions:
- Dynamic configuration updates during execution
- Configuration optimization based on historical performance
- Automatic parameter tuning
- Multi-objective optimization presets
- Cloud-based configuration sharing
- Configuration versioning and rollback

## License

Part of the OpenEvolve Frontend project.

## Contact

For questions or issues, please refer to the OpenEvolve documentation.
