# OpenEvolve API Reference

**Complete Parameter Guide for OpenEvolve Integration**

---

## Overview

This document provides a comprehensive reference for all 211 OpenEvolve parameters organized into 19 categories. Each parameter includes its type, range, default value, and description.

---

## Parameter Categories

### 1. Core Evolution (15 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `evolution_mode` | string | standard, quality_diversity, multi_objective, adversarial, problem_decomposition | standard | Evolution strategy to use |
| `max_iterations` | integer | 1-1000 | 10 | Maximum number of evolution iterations |
| `population_size` | integer | 1-1000 | 20 | Size of the evolution population |
| `temperature` | float | 0.0-2.0 | 0.7 | Controls randomness in evolution |
| `max_tokens` | integer | 1-32000 | 2048 | Maximum tokens per generation |
| `seed` | integer | 0-2147483647 | null | Random seed for reproducibility |
| `early_stopping` | boolean | true/false | false | Enable early stopping |
| `convergence_threshold` | float | 0.0-1.0 | 0.001 | Threshold for convergence detection |
| `fitness_function` | string | custom | default | Fitness evaluation function |
| `selection_pressure` | float | 0.1-10.0 | 1.0 | Selection pressure intensity |
| `mutation_rate` | float | 0.0-1.0 | 0.1 | Rate of mutations |
| `crossover_rate` | float | 0.0-1.0 | 0.8 | Rate of crossover operations |
| `elitism` | boolean | true/false | true | Preserve best individuals |
| `diversity_maintenance` | boolean | true/false | true | Maintain population diversity |
| `adaptive_parameters` | boolean | true/false | false | Adapt parameters during evolution |

### 2. Model Configuration (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `api_key` | string | - | required | API key for model access |
| `api_base` | string | URL | https://api.openai.com/v1 | Base URL for API |
| `model_id` | string | - | gpt-4 | Primary model identifier |
| `backup_models` | array | - | [] | Fallback model list |
| `timeout` | integer | 1-300 | 30 | Request timeout in seconds |
| `max_retries` | integer | 0-10 | 3 | Maximum retry attempts |
| `retry_delay` | float | 0.1-10.0 | 1.0 | Delay between retries |
| `rate_limit` | integer | 1-1000 | 60 | Requests per minute |
| `concurrent_requests` | integer | 1-50 | 5 | Concurrent API requests |
| `model_rotation` | boolean | true/false | false | Rotate between models |

### 3. Quality Diversity (12 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `feature_dimensions` | array | - | [] | Behavior dimensions for QD |
| `feature_bins` | integer | 2-50 | 10 | Number of bins per dimension |
| `archive_size` | integer | 10-10000 | 100 | Maximum archive size |
| `novelty_threshold` | float | 0.0-1.0 | 0.1 | Minimum novelty for archive |
| `quality_threshold` | float | 0.0-1.0 | 0.0 | Minimum quality for archive |
| `diversity_weight` | float | 0.0-1.0 | 0.5 | Weight of diversity vs quality |
| `behavior_space` | string | - | auto | Behavior space definition |
| `distance_metric` | string | euclidean, manhattan, cosine | euclidean | Distance calculation method |
| `archive_update_freq` | integer | 1-100 | 1 | Archive update frequency |
| `exploration_bonus` | float | 0.0-2.0 | 0.1 | Bonus for exploration |
| `crowding_distance` | boolean | true/false | true | Use crowding distance |
| `pareto_layers` | integer | 1-10 | 3 | Number of Pareto layers |

### 4. Multi-Objective (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `objectives` | array | - | [] | List of objectives to optimize |
| `objective_weights` | array | - | [] | Weights for each objective |
| `pareto_front_size` | integer | 10-1000 | 50 | Maximum Pareto front size |
| `dominance_type` | string | standard, epsilon, fuzzy | standard | Dominance relation type |
| `epsilon_values` | array | - | [] | Epsilon values for epsilon-dominance |
| `reference_point` | array | - | [] | Reference point for hypervolume |
| `scalarization` | string | weighted_sum, tchebycheff, pbi | weighted_sum | Scalarization method |
| `constraint_handling` | string | penalty, repair, feasibility | penalty | Constraint handling method |
| `constraint_tolerance` | float | 0.0-1.0 | 0.01 | Tolerance for constraints |
| `hypervolume_ref` | array | - | [] | Hypervolume reference point |

### 5. Adversarial (12 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `adversarial_rounds` | integer | 1-20 | 5 | Number of adversarial rounds |
| `attack_model_config` | object | - | {} | Configuration for attack model |
| `defense_model_config` | object | - | {} | Configuration for defense model |
| `attack_strength` | float | 0.1-2.0 | 1.0 | Strength of adversarial attacks |
| `defense_strength` | float | 0.1-2.0 | 1.0 | Strength of defense mechanisms |
| `adversarial_budget` | integer | 1-1000 | 100 | Budget for adversarial operations |
| `attack_types` | array | - | [] | Types of attacks to use |
| `defense_strategies` | array | - | [] | Defense strategies to employ |
| `robustness_metric` | string | - | accuracy | Metric for robustness evaluation |
| `perturbation_bound` | float | 0.0-1.0 | 0.1 | Maximum perturbation allowed |
| `gradient_masking` | boolean | true/false | false | Use gradient masking |
| `ensemble_defense` | boolean | true/false | true | Use ensemble for defense |

### 6. Island Model (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `num_islands` | integer | 2-20 | 5 | Number of evolution islands |
| `migration_interval` | integer | 1-100 | 10 | Generations between migrations |
| `migration_size` | integer | 1-50 | 5 | Number of individuals to migrate |
| `migration_topology` | string | ring, star, fully_connected | ring | Migration topology |
| `migration_policy` | string | best, random, diverse | best | Migration selection policy |
| `replacement_policy` | string | worst, random, similar | worst | Replacement policy |
| `island_sizes` | array | - | [] | Custom sizes for each island |
| `heterogeneous_islands` | boolean | true/false | false | Use different algorithms per island |
| `synchronous_migration` | boolean | true/false | true | Synchronize migration timing |
| `adaptive_migration` | boolean | true/false | false | Adapt migration parameters |

### 7. Selection & Reproduction (12 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `selection_method` | string | tournament, roulette, rank | tournament | Selection method |
| `tournament_size` | integer | 2-20 | 3 | Tournament selection size |
| `elite_ratio` | float | 0.0-0.5 | 0.1 | Ratio of elites to preserve |
| `exploration_ratio` | float | 0.0-1.0 | 0.2 | Ratio for exploration |
| `exploitation_ratio` | float | 0.0-1.0 | 0.6 | Ratio for exploitation |
| `random_ratio` | float | 0.0-1.0 | 0.2 | Ratio for random selection |
| `parent_selection` | string | fitness, diversity, hybrid | fitness | Parent selection strategy |
| `survivor_selection` | string | generational, steady_state | generational | Survivor selection method |
| `replacement_rate` | float | 0.0-1.0 | 1.0 | Population replacement rate |
| `selection_pressure_decay` | float | 0.0-1.0 | 0.0 | Selection pressure decay rate |
| `diversity_selection` | boolean | true/false | false | Include diversity in selection |
| `age_based_selection` | boolean | true/false | false | Consider individual age |

### 8. Evaluation (15 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `parallel_evaluations` | integer | 1-50 | 4 | Number of parallel evaluations |
| `evaluation_timeout` | integer | 1-300 | 60 | Timeout per evaluation |
| `evaluation_retries` | integer | 0-5 | 2 | Retries for failed evaluations |
| `cache_evaluations` | boolean | true/false | true | Cache evaluation results |
| `cache_size` | integer | 100-10000 | 1000 | Maximum cache size |
| `evaluation_noise` | float | 0.0-0.5 | 0.0 | Noise level in evaluations |
| `fitness_scaling` | string | linear, exponential, logarithmic | linear | Fitness scaling method |
| `normalization` | boolean | true/false | true | Normalize fitness values |
| `multi_criteria_eval` | boolean | true/false | false | Multi-criteria evaluation |
| `evaluation_budget` | integer | 1-100000 | 10000 | Total evaluation budget |
| `incremental_eval` | boolean | true/false | false | Incremental evaluation |
| `surrogate_model` | boolean | true/false | false | Use surrogate model |
| `active_learning` | boolean | true/false | false | Active learning for evaluation |
| `uncertainty_sampling` | boolean | true/false | false | Sample uncertain regions |
| `cascade_evaluation` | boolean | true/false | true | Use cascade evaluation |

### 9. Prompt Engineering (12 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `prompt_template` | string | - | default | Base prompt template |
| `system_prompt` | string | - | "" | System-level prompt |
| `context_length` | integer | 100-8000 | 2000 | Maximum context length |
| `prompt_optimization` | boolean | true/false | true | Optimize prompts during evolution |
| `template_stochasticity` | boolean | true/false | true | Use stochastic templates |
| `meta_prompting` | boolean | true/false | false | Use meta-prompting techniques |
| `few_shot_examples` | integer | 0-20 | 3 | Number of few-shot examples |
| `chain_of_thought` | boolean | true/false | true | Use chain-of-thought prompting |
| `self_consistency` | boolean | true/false | false | Use self-consistency decoding |
| `prompt_ensembling` | boolean | true/false | false | Ensemble multiple prompts |
| `dynamic_prompting` | boolean | true/false | false | Dynamically adjust prompts |
| `prompt_compression` | boolean | true/false | false | Compress long prompts |

### 10. Artifact Management (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enable_artifacts` | boolean | true/false | true | Enable artifact generation |
| `artifact_types` | array | - | ["code", "text"] | Types of artifacts to generate |
| `max_artifact_size` | integer | 1024-1048576 | 20480 | Maximum artifact size in bytes |
| `artifact_validation` | boolean | true/false | true | Validate generated artifacts |
| `artifact_compression` | boolean | true/false | false | Compress artifacts |
| `artifact_versioning` | boolean | true/false | true | Version control for artifacts |
| `artifact_metadata` | boolean | true/false | true | Include metadata with artifacts |
| `artifact_cleanup` | boolean | true/false | true | Clean up old artifacts |
| `artifact_storage` | string | memory, disk, cloud | memory | Artifact storage location |
| `artifact_encryption` | boolean | true/false | false | Encrypt sensitive artifacts |

### 11. Resource Management (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `memory_limit_mb` | integer | 512-32768 | 4096 | Memory limit in MB |
| `cpu_limit` | float | 0.1-1.0 | 0.8 | CPU usage limit (fraction) |
| `max_time` | integer | 60-7200 | 1800 | Maximum execution time in seconds |
| `disk_limit_mb` | integer | 100-10240 | 1024 | Disk usage limit in MB |
| `network_limit_mbps` | integer | 1-1000 | 100 | Network bandwidth limit |
| `api_call_limit` | integer | 10-10000 | 1000 | Maximum API calls |
| `token_limit` | integer | 1000-1000000 | 100000 | Maximum tokens |
| `cost_limit_usd` | float | 0.01-1000.0 | 10.0 | Maximum cost in USD |
| `resource_monitoring` | boolean | true/false | true | Monitor resource usage |
| `auto_scaling` | boolean | true/false | false | Auto-scale resources |

### 12. Database & Storage (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `db_path` | string | - | "./openevolve.db" | Database file path |
| `db_type` | string | sqlite, postgresql, mongodb | sqlite | Database type |
| `connection_string` | string | - | "" | Database connection string |
| `max_connections` | integer | 1-100 | 10 | Maximum database connections |
| `connection_timeout` | integer | 1-60 | 30 | Connection timeout in seconds |
| `query_timeout` | integer | 1-300 | 60 | Query timeout in seconds |
| `batch_size` | integer | 1-10000 | 1000 | Batch size for operations |
| `compression` | boolean | true/false | true | Compress stored data |
| `encryption` | boolean | true/false | false | Encrypt stored data |
| `backup_enabled` | boolean | true/false | true | Enable automatic backups |

### 13. Evolution Tracing (12 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `trace_enabled` | boolean | true/false | false | Enable evolution tracing |
| `trace_level` | string | basic, detailed, full | basic | Level of tracing detail |
| `trace_format` | string | json, csv, binary | json | Trace output format |
| `trace_file` | string | - | "./trace.log" | Trace output file |
| `trace_compression` | boolean | true/false | true | Compress trace files |
| `trace_rotation` | boolean | true/false | true | Rotate trace files |
| `max_trace_size_mb` | integer | 1-1024 | 100 | Maximum trace file size |
| `trace_buffer_size` | integer | 100-10000 | 1000 | Trace buffer size |
| `real_time_tracing` | boolean | true/false | false | Real-time trace streaming |
| `trace_sampling` | float | 0.01-1.0 | 1.0 | Sampling rate for tracing |
| `include_population` | boolean | true/false | false | Include population in trace |
| `include_fitness` | boolean | true/false | true | Include fitness in trace |

### 14. Early Stopping (8 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `early_stopping_patience` | integer | 1-100 | 10 | Patience for early stopping |
| `min_improvement` | float | 0.0-1.0 | 0.001 | Minimum improvement threshold |
| `improvement_window` | integer | 1-50 | 5 | Window for improvement calculation |
| `plateau_threshold` | integer | 1-100 | 20 | Generations to consider plateau |
| `convergence_check` | boolean | true/false | true | Check for convergence |
| `diversity_threshold` | float | 0.0-1.0 | 0.01 | Minimum diversity threshold |
| `stagnation_limit` | integer | 1-100 | 50 | Maximum stagnation generations |
| `adaptive_stopping` | boolean | true/false | false | Adaptive stopping criteria |

### 15. Distributed Processing (10 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `distributed` | boolean | true/false | false | Enable distributed processing |
| `num_workers` | integer | 1-100 | 4 | Number of worker processes |
| `worker_timeout` | integer | 10-600 | 120 | Worker timeout in seconds |
| `load_balancing` | string | round_robin, least_loaded, random | round_robin | Load balancing strategy |
| `fault_tolerance` | boolean | true/false | true | Enable fault tolerance |
| `worker_restart` | boolean | true/false | true | Auto-restart failed workers |
| `communication_backend` | string | local, redis, rabbitmq | local | Communication backend |
| `message_compression` | boolean | true/false | true | Compress messages |
| `heartbeat_interval` | integer | 1-60 | 10 | Heartbeat interval in seconds |
| `cluster_scaling` | boolean | true/false | false | Auto-scale cluster |

### 16. Advanced Research (20 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `novelty_search` | boolean | true/false | false | Enable novelty search |
| `curiosity_driven` | boolean | true/false | false | Curiosity-driven exploration |
| `meta_learning` | boolean | true/false | false | Enable meta-learning |
| `transfer_learning` | boolean | true/false | false | Transfer from previous runs |
| `continual_learning` | boolean | true/false | false | Continual learning mode |
| `few_shot_adaptation` | boolean | true/false | false | Few-shot adaptation |
| `zero_shot_transfer` | boolean | true/false | false | Zero-shot transfer |
| `domain_adaptation` | boolean | true/false | false | Domain adaptation |
| `multi_task_learning` | boolean | true/false | false | Multi-task learning |
| `lifelong_learning` | boolean | true/false | false | Lifelong learning |
| `neural_architecture_search` | boolean | true/false | false | NAS integration |
| `hyperparameter_optimization` | boolean | true/false | false | HPO integration |
| `automated_ml` | boolean | true/false | false | AutoML features |
| `explainable_ai` | boolean | true/false | false | XAI integration |
| `federated_learning` | boolean | true/false | false | Federated learning |
| `differential_privacy` | boolean | true/false | false | Privacy preservation |
| `quantum_computing` | boolean | true/false | false | Quantum computing support |
| `neuromorphic_computing` | boolean | true/false | false | Neuromorphic support |
| `edge_computing` | boolean | true/false | false | Edge deployment |
| `green_ai` | boolean | true/false | false | Energy-efficient AI |

### 17. Custom Requirements (8 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `custom_fitness` | string | - | "" | Custom fitness function code |
| `custom_operators` | array | - | [] | Custom genetic operators |
| `custom_constraints` | array | - | [] | Custom constraint functions |
| `domain_knowledge` | string | - | "" | Domain-specific knowledge |
| `expert_rules` | array | - | [] | Expert-defined rules |
| `business_logic` | string | - | "" | Business logic constraints |
| `regulatory_compliance` | array | - | [] | Compliance requirements |
| `ethical_guidelines` | array | - | [] | Ethical AI guidelines |

### 18. UI & Visualization (8 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enable_visualization` | boolean | true/false | true | Enable visualizations |
| `plot_frequency` | integer | 1-100 | 10 | Plotting frequency |
| `plot_types` | array | - | ["fitness", "diversity"] | Types of plots to generate |
| `interactive_plots` | boolean | true/false | true | Interactive visualizations |
| `real_time_updates` | boolean | true/false | false | Real-time plot updates |
| `export_plots` | boolean | true/false | true | Export plots to files |
| `plot_format` | string | png, svg, pdf | png | Plot export format |
| `dashboard_enabled` | boolean | true/false | true | Enable monitoring dashboard |

### 19. Experimental (7 parameters)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `experimental_features` | boolean | true/false | false | Enable experimental features |
| `beta_algorithms` | boolean | true/false | false | Use beta algorithms |
| `research_mode` | boolean | true/false | false | Research mode settings |
| `debug_mode` | boolean | true/false | false | Debug mode |
| `profiling_enabled` | boolean | true/false | false | Performance profiling |
| `memory_profiling` | boolean | true/false | false | Memory usage profiling |
| `experimental_logging` | boolean | true/false | false | Experimental logging |

---

## Parameter Validation Rules

### Required Parameters
- `api_key`: Must be provided for API access
- `evolution_mode`: Must be one of the supported modes

### Ratio Constraints
- `elite_ratio + exploration_ratio + exploitation_ratio + random_ratio` should sum to ≤ 1.0
- Individual ratios must be between 0.0 and 1.0

### Range Validations
- All integer parameters must be within specified ranges
- All float parameters must be within specified ranges
- Array parameters must contain valid elements

### Dependency Rules
- Quality Diversity mode requires `feature_dimensions` to be specified
- Multi-Objective mode requires `objectives` to be specified
- Adversarial mode requires both attack and defense configurations
- Distributed processing requires `num_workers > 1`

---

## Usage Examples

### Basic Configuration
```python
config = {
    "evolution_mode": "standard",
    "max_iterations": 20,
    "population_size": 30,
    "temperature": 0.7,
    "api_key": "your_api_key"
}
```

### Quality Diversity Configuration
```python
config = {
    "evolution_mode": "quality_diversity",
    "feature_dimensions": ["complexity", "novelty", "quality"],
    "feature_bins": 10,
    "archive_size": 100,
    "max_iterations": 50,
    "api_key": "your_api_key"
}
```

### Multi-Objective Configuration
```python
config = {
    "evolution_mode": "multi_objective",
    "objectives": ["accuracy", "efficiency", "robustness"],
    "pareto_front_size": 50,
    "max_iterations": 100,
    "api_key": "your_api_key"
}
```

### Adversarial Configuration
```python
config = {
    "evolution_mode": "adversarial",
    "adversarial_rounds": 5,
    "attack_model_config": {"model_id": "gpt-4"},
    "defense_model_config": {"model_id": "claude-3"},
    "max_iterations": 30,
    "api_key": "your_api_key"
}
```

---

## Best Practices

### Performance Optimization
- Use `parallel_evaluations` to speed up fitness evaluation
- Enable `cache_evaluations` to avoid redundant computations
- Set appropriate `memory_limit_mb` and `cpu_limit` for your system

### Quality Improvement
- Increase `population_size` and `max_iterations` for better results
- Use `quality_diversity` mode for diverse solutions
- Enable `early_stopping` to avoid overfitting

### Resource Management
- Set `cost_limit_usd` to control API costs
- Use `api_call_limit` to manage API usage
- Monitor with `resource_monitoring` enabled

### Debugging
- Enable `debug_mode` for detailed logging
- Use `trace_enabled` for evolution tracking
- Set `profiling_enabled` for performance analysis

---

## Error Handling

### Common Validation Errors
- **Missing API Key**: Ensure `api_key` is provided
- **Invalid Evolution Mode**: Use supported evolution modes
- **Ratio Sum Exceeded**: Ensure selection ratios sum to ≤ 1.0
- **Range Violations**: Check parameter ranges

### Runtime Errors
- **API Timeout**: Increase `timeout` parameter
- **Memory Limit**: Increase `memory_limit_mb`
- **Resource Exhaustion**: Check resource limits

### Recovery Strategies
- **Fallback Models**: Configure `backup_models`
- **Retry Logic**: Set appropriate `max_retries`
- **Graceful Degradation**: Enable fallback handlers

---

## Version Compatibility

This API reference is compatible with:
- OpenEvolve Backend v2.0+
- Python 3.8+
- All supported LLM providers (OpenAI, Anthropic, etc.)

---

## Support

For additional help:
- Check the troubleshooting guide
- Review example configurations
- Consult the integration documentation