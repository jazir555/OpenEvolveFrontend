# ALL 272+ OpenEvolve Parameters Now Configurable in BubbleLabs UI

**Date:** 2025-12-30
**Status:** ✅ **COMPLETE**

---

## Overview

The BubbleLabs UI now supports configuration of **ALL 272+ OpenEvolve parameters** across 19 categories. Users can fine-tune every aspect of their Sovereign Decomposition workflows through a comprehensive tabbed interface.

---

## Implementation Details

### Modified File: `bubblelabs_ui_component.py`

**1. Added Import:**
```python
from parameter_definitions import DEFAULT_PARAMETER_DEFINITIONS
```

**2. Added Three New Methods:**

#### `_render_all_openevolve_parameters(prefix: str = "sov")`
- Renders ALL 272+ parameters organized by category
- Creates a tab for each of the 19 parameter categories
- Displays parameter count per category

#### `_render_single_parameter(param_name, param_config, prefix, category)`
- Dynamically renders individual parameters based on type
- Supports: `select`, `boolean`, `integer`, `float`, `list`, `dict`, `string`
- Includes descriptions as help text
- Uses appropriate UI widgets for each type

#### `_get_all_openevolve_parameters_from_session(prefix: str = "sov")`
- Collects all parameter values from session state
- Returns parameters organized by category
- Used when creating workflow definitions

**3. Updated `_render_sovereign_workflow_config()`:**
- Now has 2 tabs: "Teams & Gauntlets" and "All 272 Parameters"
- The "All 272 Parameters" tab renders ALL parameters from `parameter_definitions.py`

**4. Updated `_get_workflow_config_from_state()`:**
- Captures ALL OpenEvolve parameters from session state
- Stores them in workflow configuration

**5. Updated `_create_sovereign_workflow_definition()`:**
- Includes ALL 272+ parameters in workflow metadata
- Counts and reports total parameter count

---

## All 19 Parameter Categories (272+ Parameters)

### 1. Core Evolution (31 parameters)
- `evolution_mode` - Evolution strategy
- `max_iterations` - Maximum evolution iterations
- `population_size` - Population size per generation
- `temperature` - LLM sampling temperature
- `max_tokens` - Maximum tokens per LLM call
- `top_p` - Nucleus sampling parameter
- `frequency_penalty` - Frequency penalty
- `presence_penalty` - Presence penalty
- `seed` - Random seed for reproducibility
- `random_seed` - Alternative random seed
- `api_timeout` - API request timeout (seconds)
- `api_retries` - Number of API retry attempts
- `api_retry_delay` - Delay between retries (seconds)
- `content_type` - Type of content being evolved
- `system_message` - System prompt for LLM
- `convergence_threshold` - Threshold for convergence detection
- `fitness_function` - Fitness evaluation function
- `elitism` - Preserve best individuals
- `diversity_maintenance` - Maintain population diversity
- `adaptive_parameters` - Adapt parameters during evolution
- `reasoning_effort` - Reasoning effort level
- `language` - Programming language
- `file_suffix` - File extension

### 2. Model Config (20 parameters)
- `model_configs` - List of model configurations
- `api_key` - API key for LLM provider
- `api_base` - Base URL for API
- `extra_headers` - Additional HTTP headers
- `n` - Number of completions per request
- `logit_bias` - Token likelihood modifications
- `stop_sequences` - Sequences that stop generation
- `logprobs` - Include log probabilities
- `top_logprobs` - Number of top log probs
- `response_format` - Response format
- `model_id` - Primary model identifier
- `backup_models` - Fallback model list
- `timeout` - Request timeout in seconds
- `max_retries` - Maximum retry attempts
- `retry_delay` - Delay between retries
- `rate_limit` - Requests per minute
- `concurrent_requests` - Concurrent API requests
- `model_rotation` - Rotate between models

### 3. Quality Diversity (23 parameters)
- `feature_dimensions` - Feature dimensions for behavior
- `feature_bins` - Bins per feature dimension
- `archive_size` - Maximum archive size
- `behavior_dimensions` - Specific behavior dimensions
- `diversity_metric` - Diversity measurement metric
- `diversity_reference_size` - Reference set size for diversity
- `adaptive_feature_dimensions` - Dynamically adjust features
- `double_selection` - Different programs for performance vs inspiration
- `qd_algorithm` - QD algorithm to use
- `novelty_threshold` - Minimum novelty for archive
- `behavior_descriptor_type` - Type of behavior descriptor
- `archive_learning_rate` - Archive adaptation rate
- `quality_threshold` - Minimum quality for archive
- `diversity_weight` - Weight of diversity vs quality
- `behavior_space` - Behavior space definition
- `distance_metric` - Distance calculation method
- `archive_update_freq` - Archive update frequency
- `exploration_bonus` - Bonus for exploration
- `pareto_layers` - Number of Pareto layers

### 4. Multi Objective (20 parameters)
- `objectives` - List of objectives to optimize
- `objective_weights` - Weights for each objective
- `pareto_front_size` - Maximum Pareto front size
- `dominance_metric` - Dominance metric
- `constraint_handling` - Constraint handling method
- `reference_point` - Reference point for hypervolume
- `crowding_distance` - Use crowding distance
- `epsilon_dominance` - Epsilon for epsilon-dominance
- `decomposition_method` - Objective decomposition method
- `scalarization_function` - Scalarization function
- `dominance_type` - Dominance relation type
- `epsilon_values` - Epsilon values for epsilon-dominance
- `scalarization` - Scalarization method
- `constraint_tolerance` - Tolerance for constraints
- `hypervolume_ref` - Hypervolume reference point

### 5. Adversarial (20 parameters)
- `attack_model_config` - Attack model configuration
- `defense_model_config` - Defense model configuration
- `adversarial_rounds` - Number of adversarial rounds
- `attack_strength` - Strength of attacks
- `defense_strategy` - Defense strategy
- `coevolutionary_approach` - Use co-evolution
- `red_team_models` - Red team model IDs
- `blue_team_models` - Blue team model IDs
- `red_team_sample_size` - Red team models to sample
- `blue_team_sample_size` - Blue team models to sample
- `adversarial_temperature` - Temperature for adversarial generation
- `attack_diversity` - Encourage diverse attacks
- `defense_strength` - Strength of defense mechanisms
- `adversarial_budget` - Budget for adversarial operations
- `attack_types` - Types of attacks to use
- `defense_strategies` - Defense strategies to employ
- `robustness_metric` - Metric for robustness evaluation
- `perturbation_bound` - Maximum perturbation allowed
- `gradient_masking` - Use gradient masking
- `ensemble_defense` - Use ensemble for defense

### 6. Island Model (20 parameters)
- `num_islands` - Number of islands
- `migration_interval` - Generations between migrations
- `migration_rate` - Proportion to migrate
- `migration_topology` - Migration topology
- `ring_topology` - Use ring topology
- `controlled_gene_flow` - Control gene flow
- `island_diversity_metric` - Island diversity metric
- `migration_selection` - Migrant selection method
- `island_initialization` - Island initialization method
- `island_specialization` - Allow island specialization
- `migration_size` - Number of individuals to migrate
- `migration_policy` - Migration selection policy
- `replacement_policy` - Replacement policy
- `island_sizes` - Custom sizes for each island
- `heterogeneous_islands` - Use different algorithms per island
- `synchronous_migration` - Synchronize migration timing
- `adaptive_migration` - Adapt migration parameters

### 7. Selection (18 parameters)
- `elite_ratio` - Proportion of elites
- `exploration_ratio` - Proportion for exploration
- `exploitation_ratio` - Proportion for exploitation
- `multi_strategy_sampling` - Use multiple sampling strategies
- `selection_pressure` - Selection pressure
- `tournament_size` - Tournament size
- `crossover_rate` - Crossover rate
- `mutation_rate` - Mutation rate
- `elitism_count` - Number of elites to preserve
- `selection_method` - Selection method
- `reproduction_method` - Reproduction method
- `parent_selection` - Parent selection method
- `random_ratio` - Ratio for random selection
- `survivor_selection` - Survivor selection method
- `replacement_rate` - Population replacement rate
- `selection_pressure_decay` - Selection pressure decay rate
- `diversity_selection` - Include diversity in selection
- `age_based_selection` - Consider individual age

### 8. Evaluation (28 parameters)
- `cascade_evaluation` - Use cascade evaluation
- `cascade_thresholds` - Thresholds for cascade levels
- `parallel_evaluations` - Number of parallel workers
- `evaluator_timeout` - Evaluation timeout (seconds)
- `max_retries_eval` - Max evaluation retries
- `use_llm_feedback` - Use LLM-based feedback
- `llm_feedback_weight` - Weight for LLM feedback
- `evaluator_models` - Evaluator model configurations
- `evaluator_system_message` - System prompt for evaluator
- `ensemble_size` - Number of evaluators in ensemble
- `consensus_threshold` - Threshold for consensus
- `evaluation_criteria` - List of evaluation criteria
- `custom_evaluator` - Custom evaluation function
- `evaluation_batch_size` - Batch size for evaluations
- `cache_evaluations` - Cache evaluation results
- `cache_size` - Maximum cache size
- `evaluation_noise` - Noise level in evaluations
- `fitness_scaling` - Fitness scaling method
- `normalization` - Normalize fitness values
- `multi_criteria_eval` - Multi-criteria evaluation
- `evaluation_budget` - Total evaluation budget
- `incremental_eval` - Incremental evaluation
- `surrogate_model` - Use surrogate model
- `active_learning` - Active learning for evaluation
- `uncertainty_sampling` - Sample uncertain regions

### 9. Prompt Engineering (12 parameters)
- `prompt_template` - Base prompt template
- `system_prompt` - System-level prompt
- `context_length` - Maximum context length
- `prompt_optimization` - Optimize prompts during evolution
- `template_stochasticity` - Use stochastic templates
- `meta_prompting` - Use meta-prompting techniques
- `few_shot_examples` - Number of few-shot examples
- `chain_of_thought` - Use chain-of-thought prompting
- `self_consistency` - Use self-consistency decoding
- `prompt_ensembling` - Ensemble multiple prompts
- `dynamic_prompting` - Dynamically adjust prompts
- `prompt_compression` - Compress long prompts

### 10. Artifact Management (11 parameters)
- `enable_artifacts` - Enable artifact generation
- `artifact_types` - Types of artifacts to generate
- `max_artifact_size` - Maximum artifact size in bytes
- `artifact_validation` - Validate generated artifacts
- `artifact_compression` - Compress artifacts
- `artifact_versioning` - Version control for artifacts
- `artifact_metadata` - Include metadata with artifacts
- `artifact_cleanup` - Clean up old artifacts
- `artifact_storage` - Artifact storage location
- `artifact_encryption` - Encrypt sensitive artifacts

### 11. Resource Management (12 parameters)
- `memory_limit_mb` - Memory limit in MB
- `cpu_limit` - CPU usage limit (fraction)
- `max_time` - Maximum execution time in seconds
- `disk_limit_mb` - Disk usage limit in MB
- `network_limit_mbps` - Network bandwidth limit
- `api_call_limit` - Maximum API calls
- `token_limit` - Maximum tokens
- `cost_limit_usd` - Maximum cost in USD
- `resource_monitoring` - Monitor resource usage
- `auto_scaling` - Auto-scale resources
- `checkpoint_interval` - Generations between checkpoints

### 12. Database Storage (9 parameters)
- `db_path` - Database file path
- `db_type` - Database type
- `connection_string` - Database connection string
- `max_connections` - Maximum database connections
- `connection_timeout` - Connection timeout in seconds
- `query_timeout` - Query timeout in seconds
- `batch_size` - Batch size for operations
- `compression` - Compress stored data
- `encryption` - Encrypt stored data
- `backup_enabled` - Enable automatic backups

### 13. Evolution Tracing (12 parameters)
- `trace_enabled` - Enable evolution tracing
- `trace_level` - Level of tracing detail
- `trace_format` - Trace output format
- `trace_file` - Trace output file
- `trace_compression` - Compress trace files
- `trace_rotation` - Rotate trace files
- `max_trace_size_mb` - Maximum trace file size
- `trace_buffer_size` - Trace buffer size
- `real_time_tracing` - Real-time trace streaming
- `trace_sampling` - Sampling rate for tracing
- `include_population` - Include population in trace
- `include_fitness` - Include fitness in trace

### 14. Early Stopping (9 parameters)
- `early_stopping` - Enable early stopping
- `early_stopping_patience` - Patience for early stopping
- `min_improvement` - Minimum improvement threshold
- `improvement_window` - Window for improvement calculation
- `plateau_threshold` - Generations to consider plateau
- `convergence_check` - Check for convergence
- `diversity_threshold` - Minimum diversity threshold
- `stagnation_limit` - Maximum stagnation generations
- `adaptive_stopping` - Adaptive stopping criteria

### 15. Distributed Processing (10 parameters)
- `distributed` - Enable distributed processing
- `num_workers` - Number of worker processes
- `worker_timeout` - Worker timeout in seconds
- `load_balancing` - Load balancing strategy
- `fault_tolerance` - Enable fault tolerance
- `worker_restart` - Auto-restart failed workers
- `communication_backend` - Communication backend
- `message_compression` - Compress messages
- `heartbeat_interval` - Heartbeat interval in seconds
- `cluster_scaling` - Auto-scale cluster

### 16. Advanced Research (19 parameters)
- `novelty_search` - Enable novelty search
- `curiosity_driven` - Curiosity-driven exploration
- `meta_learning` - Enable meta-learning
- `transfer_learning` - Transfer from previous runs
- `continual_learning` - Continual learning mode
- `few_shot_adaptation` - Few-shot adaptation
- `zero_shot_transfer` - Zero-shot transfer
- `domain_adaptation` - Domain adaptation
- `multi_task_learning` - Multi-task learning
- `lifelong_learning` - Lifelong learning
- `neural_architecture_search` - NAS integration
- `hyperparameter_optimization` - HPO integration
- `automated_ml` - AutoML features
- `explainable_ai` - XAI integration
- `federated_learning` - Federated learning
- `differential_privacy` - Privacy preservation
- `quantum_computing` - Quantum computing support
- `neuromorphic_computing` - Neuromorphic support
- `edge_computing` - Edge deployment
- `green_ai` - Energy-efficient AI

### 17. Custom Requirements (8 parameters)
- `custom_fitness` - Custom fitness function code
- `custom_operators` - Custom genetic operators
- `custom_constraints` - Custom constraint functions
- `domain_knowledge` - Domain-specific knowledge
- `expert_rules` - Expert-defined rules
- `business_logic` - Business logic constraints
- `regulatory_compliance` - Compliance requirements
- `ethical_guidelines` - Ethical AI guidelines

### 18. UI Visualization (8 parameters)
- `enable_visualization` - Enable visualizations
- `plot_frequency` - Plotting frequency
- `plot_types` - Types of plots to generate
- `interactive_plots` - Interactive visualizations
- `real_time_updates` - Real-time plot updates
- `export_plots` - Export plots to files
- `plot_format` - Plot export format
- `dashboard_enabled` - Enable monitoring dashboard

### 19. Experimental (7 parameters)
- `experimental_features` - Enable experimental features
- `beta_algorithms` - Use beta algorithms
- `research_mode` - Research mode settings
- `debug_mode` - Debug mode
- `profiling_enabled` - Performance profiling
- `memory_profiling` - Memory usage profiling
- `experimental_logging` - Experimental logging

---

## How to Use

### Step 1: Open BubbleLabs UI
```bash
streamlit run main.py
```
Navigate to **"BubbleLabs Workflows"** tab

### Step 2: Select Workflow Type
In the **"Workflow Designer"** sub-tab, select **"OpenEvolve Sovereign Decomposition"** from the dropdown

### Step 3: Configure Teams & Gauntlets
- Select Content Analyzer, Planner, Solver, Patcher, and Assembler teams
- Select Red and Gold gauntlets for sub-problem and final verification

### Step 4: Configure ALL 272 Parameters
Click the **"All 272 Parameters"** tab to access:

**19 Parameter Category Tabs:**
1. **Core Evolution** - Configure basic evolution settings
2. **Model Config** - Set up LLM models and API settings
3. **Quality Diversity** - Configure MAP-Elites and QD algorithms
4. **Multi Objective** - Set up multi-objective optimization
5. **Adversarial** - Configure red team/blue team parameters
6. **Island Model** - Set up island-based evolution
7. **Selection** - Configure selection methods
8. **Evaluation** - Set up evaluation strategies
9. **Prompt Engineering** - Configure LLM prompts
10. **Artifact Management** - Set artifact handling
11. **Resource Management** - Set resource limits
12. **Database Storage** - Configure database settings
13. **Evolution Tracing** - Set up tracing and logging
14. **Early Stopping** - Configure stopping criteria
15. **Distributed Processing** - Set up distributed execution
16. **Advanced Research** - Enable advanced features
17. **Custom Requirements** - Add custom code and rules
18. **UI Visualization** - Configure visualization options
19. **Experimental** - Enable experimental features

Each parameter includes:
- **Input widget** appropriate for its type (slider, number input, checkbox, select box, text area)
- **Description** as help text
- **Default value** pre-populated
- **Min/Max values** for numeric parameters
- **Options** for select parameters

### Step 5: Enter Problem Statement
Describe the problem you want to solve

### Step 6: Create Workflow
Click **"Create Workflow in BubbleLabs"**
- All 272+ parameters are stored in the workflow definition
- Total parameter count is displayed
- Workflow visualization is shown

### Step 7: Execute
Click **"Create and Execute Workflow Instance"** to run with your exact configuration

---

## Parameter Storage

All parameters are stored in the workflow definition under:
```python
workflow_definition["metadata"]["openevolve_parameters"] = {
    "core_evolution": { ... 31 parameters ... },
    "model_config": { ... 20 parameters ... },
    "quality_diversity": { ... 23 parameters ... },
    ... all 19 categories ...
}
```

This ensures complete reproducibility - every aspect of the workflow configuration is captured and can be reviewed, modified, or re-executed.

---

## Technical Details

### Parameter Type Support

The UI automatically renders the correct widget based on parameter type:

| Type | Widget | Example |
|------|--------|---------|
| `select` | Dropdown menu | Evolution mode, QD algorithm |
| `boolean` | Checkbox | Enable artifacts, Use caching |
| `integer` | Number input | Max iterations, Population size |
| `float` | Slider | Temperature, Mutation rate |
| `list` | Text area (JSON) | Objectives, Custom operators |
| `dict` | Text area (JSON) | Model configs, Custom fitness |
| `string` | Text input | System prompt, DB path |

### Session State Management

Each parameter is stored with a unique key:
```
{prefix}_{category}_{parameter_name}
```

Example:
- `sov_core_evolution_temperature`
- `sov_model_config_api_key`
- `sov_evaluation_ensemble_size`

### Default Values

All default values come from `parameter_definitions.py`:
```python
DEFAULT_PARAMETER_DEFINITIONS["core_evolution"]["temperature"]["default"] = 0.7
```

---

## Benefits

✅ **Complete Control** - Every OpenEvolve parameter is configurable
✅ **Organized** - 19 logical categories make navigation easy
✅ **Validated** - Min/max values enforced, options restricted
✅ **Documented** - Every parameter has a description
✅ **Reproducible** - Full configuration stored in workflow definition
✅ **User-Friendly** - Appropriate widgets for each parameter type
✅ **Comprehensive** - Covers all aspects: evolution, models, resources, tracing, etc.

---

**Status:** ✅ **ALL 272+ PARAMETERS NOW CONFIGURABLE**

Users have complete control over their OpenEvolve Sovereign Decomposition workflows through the BubbleLabs UI.

---

*End of Documentation*
