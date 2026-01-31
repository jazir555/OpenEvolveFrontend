# Complete Parameter Documentation

**Total Parameters: 322+**

This document provides complete documentation for all parameters in the unified configuration system.

## Table of Contents

1. [Common Parameters (29)](#common-parameters)
2. [LLM Configuration (26)](#llm-configuration)
3. [Database Configuration (35)](#database-configuration)
4. [Evaluator Configuration (17)](#evaluator-configuration)
5. [PES Configuration (22)](#pes-configuration)
6. [Quality Diversity Configuration (18)](#quality-diversity-configuration)
7. [Multi-Objective Configuration (15)](#multi-objective-configuration)
8. [Adversarial Configuration (12)](#adversarial-configuration)
9. [OpenEvolve Configuration (48)](#openevolve-configuration)
10. [Parameter Mapping](#parameter-mapping)

---

## Common Parameters (29)

**Used by all evolutionary modes**

### Core Evolution (3 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_iterations` | int | 100 | ≥1 | Maximum number of evolution iterations to run |
| `random_seed` | int | 42 | ≥0 or None | Random seed for reproducibility (None = random) |
| `checkpoint_interval` | int | 50 | ≥1 | Save checkpoints every N iterations |

### Logging Configuration (6 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `log_level` | str | "INFO" | DEBUG, INFO, WARNING, ERROR, CRITICAL | Logging level |
| `log_dir` | str | None | - | Custom directory for logs (default: workspace/logs) |
| `log_to_console` | bool | True | - | Enable console logging |
| `log_to_file` | bool | True | - | Enable file logging |
| `log_rotation` | str | "H" | S, M, H, D | Log rotation frequency |
| `log_backup_count` | int | 0 | ≥0 | Number of backup logs (0 = unlimited) |

### Workspace Configuration (3 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_path` | str | "./evolve_run_output" | Root directory for outputs |
| `task_name` | str | "evolution_task" | Name for logging/filing |
| `task_description` | str | None | Detailed task description |

### Concurrency Configuration (2 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `concurrency` | int | 5 | ≥1 | Concurrent evaluations |
| `timeout` | int | 300 | ≥1 | Default timeout (seconds) |

---

## LLM Configuration (26)

**Configuration for LLM ensemble models**

### Evolution Models (10 parameters per model)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `name` | str | Required | - | Model name (e.g., "gpt-4o") |
| `weight` | float | 1.0 | ≥0.0 | Weight in ensemble |
| `api_base` | str | None | - | API base URL |
| `api_key` | str | None | - | API key |
| `model_provider` | str | None | - | Provider: openai, azure, anthropic, google |
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature |
| `top_p` | float | 0.95 | 0.0-1.0 | Nucleus sampling |
| `max_tokens` | int | 4096 | ≥1 | Max tokens to generate |
| `context_length` | int | 65536 | ≥1 | Model's context window |
| `reasoning_effort` | str | None | low/medium/high | Reasoning level for supported models |

### Request Parameters (3 parameters per model)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `timeout` | int | 60 | ≥1 | Request timeout (seconds) |
| `retries` | int | 3 | ≥0 | Number of retries |
| `retry_delay` | int | 5 | ≥0 | Delay between retries (seconds) |

### Default API Configuration (3 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_api_base` | str | "https://api.openai.com/v1" | Default API URL |
| `default_api_key` | str | None | Default API key |
| `default_temperature` | float | 0.7 | Default temperature |

---

## Database Configuration (35)

**Configuration for evolutionary database/memory**

### Storage Configuration (5 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `storage_type` | str | "in_memory" | in_memory, redis, file, database | Storage backend |
| `db_path` | str | None | - | Path for file storage |
| `redis_url` | str | "redis://localhost:6379/0" | - | Redis connection URL |
| `output_path` | str | None | - | Path for checkpoints |
| `checkpoint_interval` | int | 50 | ≥1 | Save checkpoint every N iterations |

### Population Parameters (5 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `population_size` | int | 1000 | ≥10 | Max population per island |
| `elite_archive_size` | int | 100 | ≥1 | Size of elite archive |
| `num_islands` | int | 5 | ≥1 | Number of islands |
| `use_sampling_weight` | bool | True | - | Use weighted sampling |
| `sampling_weight_power` | float | 1.0 | ≥0.0 | Power for sampling weight |

### Island Migration (2 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `migration_interval` | int | 50 | ≥1 | Migrate every N iterations |
| `migration_rate` | float | 0.1 | 0.0-1.0 | Fraction to migrate |

### Selection Parameters (4 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `elite_selection_ratio` | float | 0.1 | 0.0-1.0 | Ratio of elite solutions |
| `exploration_rate` | float | 0.2 | 0.0-1.0 | Random exploration probability |
| `exploitation_ratio` | float | 0.7 | 0.0-1.0 | Exploitation ratio |
| `boltzmann_temperature` | float | 1.0 | ≥0.0 | Temperature for Boltzmann sampling |

### MAP-Elites Feature Map (5 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_dimensions` | List[str] | ["complexity", "diversity"] | Feature dimensions for grid |
| `feature_bins` | int/dict | 10 | Bins per dimension |
| `feature_scaling_method` | str | "minmax" | Scaling: minmax, standard, robust, none |
| `diversity_reference_size` | int | 20 | Reference set size |
| `diversity_metric` | str | "edit_distance" | Metric: edit_distance, feature_based, semantic |

### Logging (2 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_prompts` | bool | True | Log prompts/responses |
| `enable_artifacts` | bool | True | Store artifacts |

---

## Evaluator Configuration (17)

**Configuration for solution evaluation**

### General Settings (5 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `timeout` | int | 300 | ≥1 | Max evaluation time (seconds) |
| `max_retries` | int | 3 | ≥0 | Max retries for failures |
| `evaluate_code` | str | "" | - | Python code for evaluation |
| `evolve_target` | str | None | - | Target/goal for evolution |
| `workspace_path` | str | None | - | Evaluation workspace path |

### Resource Limits (2 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `memory_limit_mb` | int | None | ≥1 | Memory limit (MB) |
| `cpu_limit` | float | None | ≥0.0 | CPU limit (0.0-1.0 or cores) |

### Evaluation Strategies (4 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cascade_evaluation` | bool | True | Use cascade evaluation |
| `cascade_thresholds` | List[float] | [0.5, 0.75, 0.9] | Thresholds for stages |
| `parallel_evaluations` | int | 4 | Number of parallel evals |
| `distributed` | bool | False | Enable distributed evaluation |

### LLM Feedback (2 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `use_llm_feedback` | bool | False | - | Use LLM for feedback |
| `llm_feedback_weight` | float | 0.1 | 0.0-1.0 | Weight of LLM feedback |

### Artifact Handling (2 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_artifacts` | bool | True | - | Enable artifact storage |
| `max_artifact_storage` | int | 100MB | ≥0 | Max storage per program |

---

## PES Configuration (22)

**LoongFlow PES (Plan-Evolve-Summarize) specific**

### Planning Configuration (6 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `enable_planning` | bool | True | - | Enable planning phase |
| `planner_type` | str | "evolve_planner" | evolve_planner, react_planner, chat_planner | Planner type |
| `planning_iterations` | int | 1 | ≥1 | Number of planning iterations |
| `planning_temperature` | float | 0.7 | 0.0-2.0 | Temperature for planning |
| `use_refinement` | bool | True | - | Enable plan refinement |
| `max_refinement_iterations` | int | 3 | ≥0 | Max refinement iterations |

### Execution Configuration (5 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `executor_type` | str | "evolve_executor" | evolve_executor, react_executor, chat_executor | Executor type |
| `execution_mode` | str | "sequential" | sequential, parallel, adaptive | Execution mode |
| `enable_code_execution` | bool | True | - | Enable actual code execution |
| `execution_timeout` | int | 300 | ≥1 | Timeout per execution |
| `sandbox_mode` | bool | True | - | Run in sandbox |

### Summarization Configuration (5 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `enable_summary` | bool | True | - | Enable summarization |
| `summary_type` | str | "evolve_summary" | evolve_summary, react_summary, chat_summary | Summarizer type |
| `summary_detail_level` | str | "medium" | low, medium, high | Detail level |
| `include_traceback` | bool | False | - | Include traceback |
| `summary_max_length` | int | 2000 | ≥100 | Max summary length |

### Memory Configuration (3 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `enable_memory` | bool | True | - | Enable long-term memory |
| `memory_type` | str | "in_memory" | in_memory, redis, database | Memory type |
| `memory_compression` | bool | True | - | Enable compression |

### Context Management (3 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `context_window` | int | 10000 | ≥1 | Context window size |
| `context_compression_threshold` | int | 5000 | ≥1 | Compression threshold |
| `use_context_pruning` | bool | True | - | Enable context pruning |

---

## Quality Diversity Configuration (18)

**MAP-Elites specific parameters**

### Grid Configuration (6 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_map_elites` | bool | True | - | Enable MAP-Elites |
| `grid_resolution` | int | 10 | ≥2 | Resolution of grid |
| `grid_dimensions` | List[str] | ["complexity", "diversity"] | - | Feature dimensions |
| `adaptive_grid` | bool | False | - | Enable adaptive grid |
| `grid_update_interval` | int | 100 | ≥1 | Update interval |

### Archive Configuration (5 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `archive_type` | str | "map_elites" | map_elites, cvt_map_elites, submarine_map_elites | Archive type |
| `archive_size_limit` | int | None | ≥1 or None | Max archive size |
| `archive_elitism` | bool | True | - | Use elitism |
| `use_novelty` | bool | False | - | Use novelty search |
| `novelty_threshold` | float | 0.5 | 0.0-1.0 | Novelty threshold |

### Feature Calculation (4 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `feature_extraction_method` | str | "auto" | auto, manual, learned | Extraction method |
| `feature_normalization` | str | "minmax" | minmax, standard, robust, none | Normalization |
| `use_feature_learning` | bool | False | - | Enable feature learning |
| `feature_learning_rate` | float | 0.001 | ≥0.0 | Learning rate |

### QD-Specific Parameters (3 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `cvt_samples` | int | 10000 | ≥1 | Samples for CVT |
| `use_niching` | bool | True | - | Use niching |
| `niche_radius` | float | 0.1 | ≥0.0 | Niche radius |

---

## Multi-Objective Configuration (15)

**Multi-objective optimization parameters**

### Objective Configuration (4 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objectives` | List[str] | ["score"] | List of objectives |
| `objective_weights` | Dict[str, float] | None | Weights per objective |
| `optimization_direction` | Dict[str, str] | {"score": "maximize"} | Direction: maximize/minimize |
| `use_pareto` | bool | True | Use Pareto dominance |

### Pareto Front Configuration (4 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `pareto_archive_size` | int | 100 | ≥1 | Max Pareto front size |
| `pareto_pruning_method` | str | "crowding_distance" | crowding_distance, hypervolume, epsilon_indicator | Pruning method |
| `crowding_distance_metric` | str | "euclidean" | euclidean, manhattan, cosine | Distance metric |
| `use_hypervolume` | bool | False | - | Use hypervolume indicator |

### Selection Configuration (4 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `selection_method` | str | "nsga2" | nsga2, nsga3, spea2, moead | Selection method |
| `tournament_size` | int | 2 | ≥2 | Tournament size |
| `crossover_rate` | float | 0.9 | 0.0-1.0 | Crossover rate |
| `mutation_rate` | float | 0.1 | 0.0-1.0 | Mutation rate |

### Scalarization (3 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `use_scalarization` | bool | False | - | Use scalarization |
| `scalarization_method` | str | "weighted_sum" | weighted_sum, tchebycheff, achievement | Method |
| `reference_point` | Dict[str, float] | None | - | Reference point |

---

## Adversarial Configuration (12)

**Adversarial evolution parameters**

### Adversarial Setup (4 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `enable_adversarial` | bool | False | - | Enable adversarial |
| `num_adversaries` | int | 2 | ≥2 | Number of populations |
| `adversarial_mode` | str | "generator_discriminator" | generator_discriminator, predator_prey, competitive_cooperative | Mode |
| `adversarial_rounds` | int | 20 | ≥1 | Rounds per iteration |

### Generator/Discriminator (3 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generator_objective` | str | "fool_discriminator" | Generator's objective |
| `discriminator_objective` | str | "detect_fake" | Discriminator's objective |
| `balance_factor` | float | 0.5 | Balance between updates (0.0-1.0) |

### Coevolution Dynamics (5 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_coevolution` | bool | True | Enable co-evolution |
| `coevolution_frequency` | int | 5 | Update every N iterations |
| `fitness_sharing` | bool | True | Use fitness sharing |
| `fitness_sharing_sigma` | float | 0.1 | Sigma for sharing (≥0.0) |
| `use_arms_race` | bool | False | Enable arms race |

---

## OpenEvolve Configuration (48)

**OpenEvolve-specific parameters**

### Code Evolution (6 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diff_based_evolution` | bool | True | Use diff-based evolution |
| `max_code_length` | int | 10000 | Max code length (chars) |
| `language` | str | "python" | Programming language |
| `file_suffix` | str | ".py" | File suffix |
| `enable_simplification` | bool | True | Enable auto-simplification |
| `suggest_simplification_after_chars` | int | 500 | Suggest simplify if > this |

### Prompt Configuration (8 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_dir` | str | None | Custom template directory |
| `system_message` | str | "You are an expert..." | System message |
| `evaluator_system_message` | str | "You are an expert..." | Evaluator message |
| `num_top_programs` | int | 3 | Top programs to include |
| `num_diverse_programs` | int | 2 | Diverse programs to include |
| `use_template_stochasticity` | bool | True | Use random variations |
| `template_variations` | Dict | {} | Alternative phrasings |
| `include_artifacts` | bool | True | Include artifacts |

### Artifact Handling (5 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_artifact_bytes` | int | 20KB | ≥0 | Max artifact size |
| `artifact_security_filter` | bool | True | - | Apply security filter |
| `artifact_size_threshold` | int | 32KB | ≥0 | Storage threshold |
| `cleanup_old_artifacts` | bool | True | - | Auto cleanup |
| `artifact_retention_days` | int | 30 | ≥1 | Retention days |

### Program Labeling (3 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `include_changes_under_chars` | int | 100 | ≥0 | Include changes if < this |
| `concise_implementation_max_lines` | int | 10 | ≥1 | Max lines for "concise" |
| `comprehensive_implementation_min_lines` | int | 50 | ≥1 | Min lines for "comprehensive" |

### Early Stopping (4 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `early_stopping_patience` | int | None | ≥1 or None | Stop after N no improvement |
| `convergence_threshold` | float | 0.001 | ≥0.0 | Min improvement |
| `early_stopping_metric` | str | "combined_score" | - | Metric to track |
| `target_score` | float | None | 0.0-1.0 or None | Target to stop |

### Meta-Prompting (3 parameters)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `use_meta_prompting` | bool | False | - | Enable meta-prompting |
| `meta_prompt_weight` | float | 0.1 | 0.0-1.0 | Weight for meta-prompts |
| `meta_prompt_interval` | int | 10 | ≥1 | Apply every N iterations |

### Evolution Trace (6 parameters)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `evolution_trace_enabled` | bool | False | - | Enable trace logging |
| `evolution_trace_format` | str | "jsonl" | jsonl, json, hdf5 | Trace format |
| `evolution_trace_include_code` | bool | False | - | Include full code |
| `evolution_trace_include_prompts` | bool | True | - | Include prompts |
| `evolution_trace_buffer_size` | int | 10 | ≥1 | Buffer size |
| `evolution_trace_compress` | bool | False | - | Compress output |

### Advanced Features (13 parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_embedding` | bool | False | Use embeddings |
| `embedding_model` | str | "text-embedding-ada-002" | Embedding model |
| `embedding_dimension` | int | 1536 | Embedding dim |
| `enable_novelty_search` | bool | False | Enable novelty |
| `novelty_k_nearest` | int | 10 | K for novelty |
| `enable_quality_diversity` | bool | True | Enable QD |
| `use_crossover` | bool | False | Enable crossover |
| `crossover_method` | str | "single_point" | Crossover method |
| `use_mutation` | bool | True | Enable mutation |
| `mutation_rate` | float | 0.1 | Mutation probability |
| `use_selection_pressure` | bool | True | Apply pressure |
| `selection_pressure_method` | str | "tournament" | Selection method |
| `tournament_size` | int | 3 | Tournament size |

---

## Parameter Mapping

### OpenEvolve → Unified

| OpenEvolve Parameter | Unified Parameter | Location |
|---------------------|-------------------|----------|
| `max_iterations` | `max_iterations` | common |
| `llm.models` | `models` | llm |
| `database.population_size` | `population_size` | database |
| `evaluator.timeout` | `timeout` | evaluator |
| `prompt.system_message` | `system_message` | openevolve |

### LoongFlow PES → Unified

| LoongFlow Parameter | Unified Parameter | Location |
|-------------------|-------------------|----------|
| `workspace_path` | `workspace_path` | common |
| `llm_config.model` | `models[0].name` | llm |
| `database.population_size` | `population_size` | database |
| `evolve.max_iterations` | `max_iterations` | common |
| `planner.type` | `planner_type` | pes |

### Unified → Mode-Specific

The `ConfigMapper` class handles bidirectional conversion between unified config and all mode-specific formats.

---

## Summary by Category

| Category | Parameters |
|----------|-----------|
| Common | 29 |
| LLM | 26 |
| Database | 35 |
| Evaluator | 17 |
| PES | 22 |
| Quality Diversity | 18 |
| Multi-Objective | 15 |
| Adversarial | 12 |
| OpenEvolve | 48 |
| **TOTAL** | **322** |
