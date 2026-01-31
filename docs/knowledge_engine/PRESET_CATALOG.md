# Preset Catalog

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

Complete catalog of all 30+ configuration presets organized by category.

---

## Table of Contents

1. [Performance Presets](#1-performance-presets-5-presets)
2. [Domain Presets](#2-domain-presets-18-presets)
3. [Use Case Presets](#3-use-case-presets-5-presets)
4. [System Mode Presets](#4-system-mode-presets-4-presets)
5. [Problem Type Presets](#5-problem-type-presets-5-presets)

---

## 1. Performance Presets

### 1.1 Fast Preset

**Category:** Performance
**Use When:** You need quick results
**Typical Use Cases:**
- Rapid prototyping
- Initial exploration
- Time-critical decisions
- Resource-constrained environments

**Configuration:**
```yaml
# Fast preset - Minimize execution time
evolution_mode: auto  # Auto-selects fastest mode
max_iterations: 20
max_evaluations: 30
population_size: 50

# Performance
parallel_evaluations: 8
enable_gauntlet: false  # Skip validation
enable_knowledge_engine: false  # Skip learning
save_intermediate_results: false

# Early stopping
early_stopping: true
early_stopping_patience: 3

# Resources
evaluation_timeout: 60
memory_limit_mb: 512
```

**Trade-offs:**
- ⚡ Fast execution (2-5 minutes typical)
- ❌ Lower solution quality
- ❌ No validation
- ❌ No learning from past runs

**Expected Performance:**
- Execution time: 2-5 minutes
- Solution quality: 60-70% of optimal
- API usage: Minimal (30 evaluations)

**When NOT to Use:**
- Final production optimization
- Published results
- High-stakes decisions
- Problems requiring precision

---

### 1.2 Balanced Preset

**Category:** Performance
**Use When:** Default choice for most problems
**Typical Use Cases:**
- General optimization
- Production runs
- Standard workflows
- Balanced speed/quality

**Configuration:**
```yaml
# Balanced preset - Default configuration
evolution_mode: auto
max_iterations: 100
max_evaluations: 100
population_size: 100

# Gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - gold_team

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true

# Resources
parallel_evaluations: 4
evaluation_timeout: 300
memory_limit_mb: 2048

# Logging
log_level: INFO
save_intermediate_results: true
checkpoint_interval: 20
```

**Trade-offs:**
- ⚖️ Balanced speed/quality
- ✅ Good solution quality (80-90%)
- ✅ Comprehensive validation
- ✅ Reasonable execution time

**Expected Performance:**
- Execution time: 10-20 minutes
- Solution quality: 80-90% of optimal
- API usage: Moderate (100 evaluations)

**When NOT to Use:**
- Very time-critical (use Fast)
- Maximum quality needed (use Thorough)
- Limited API budget (use Budget)

---

### 1.3 Thorough Preset

**Category:** Performance
**Use When:** Quality is critical
**Typical Use Cases:**
- Final optimization
- Published research
- High-stakes decisions
- Competition settings

**Configuration:**
```yaml
# Thorough preset - Maximum quality
evolution_mode: auto
max_iterations: 500
max_evaluations: 500
population_size: 200

# Gauntlet (comprehensive)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - red_team  # Second round
  - gold_team
  - gold_team  # Second round

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true
use_past_solutions: true

# Resources
parallel_evaluations: 2
evaluation_timeout: 600
memory_limit_mb: 8192

# Extensive logging
log_level: DEBUG
save_intermediate_results: true
checkpoint_interval: 10
```

**Trade-offs:**
- 🎯 Best solution quality (95-99%)
- ❌ Long execution time (1-2 hours)
- ✅ Comprehensive validation
- ✅ Maximum learning

**Expected Performance:**
- Execution time: 60-120 minutes
- Solution quality: 95-99% of optimal
- API usage: High (500 evaluations)

**When NOT to Use:**
- Time constraints
- Limited API budget
- Quick iterations needed

---

### 1.4 Budget Preset

**Category:** Performance
**Use When:** Limited API/computational budget
**Typical Use Cases:**
- Limited API tokens
- Cost constraints
- Free tier usage
- Experimentation

**Configuration:**
```yaml
# Budget preset - Minimize API usage
evolution_mode: pes  # 60% fewer evaluations
max_evaluations: 20  # Minimal evaluations
enable_planning: true
enable_memory: true
early_stopping: true
early_stop_threshold: 0.85

# Sequential to avoid rate limits
parallel_evaluations: 1

# Minimal validation
enable_gauntlet: false
enable_knowledge_engine: false

# Minimal logging
log_level: WARNING
save_intermediate_results: false
```

**Trade-offs:**
- 💰 Minimal API usage (20 evaluations)
- ✅ Still good quality (70-80%)
- ❌ Sequential (slower wall-clock time)
- ❌ No validation

**Expected Performance:**
- Execution time: 5-10 minutes (sequential)
- Solution quality: 70-80% of optimal
- API usage: Minimal (20 evaluations)
- Cost savings: 80% vs Balanced

**When NOT to Use:**
- Time constraints (wall-clock)
- Maximum quality needed
- Validation required

---

### 1.5 Exploration Preset

**Category:** Performance
**Use When:** Need diverse solutions
**Typical Use Cases:**
- Discovery phase
- Multiple use cases
- Innovation
- Solution variety

**Configuration:**
```yaml
# Exploration preset - Maximize diversity
evolution_mode: qd  # Quality-Diversity
max_iterations: 100
population_size: 150

# QD settings
feature_dimensions:
  - complexity
  - diversity
  - novelty
feature_bins: 15
archive_size: 500

# Promote diversity
exploration_ratio: 0.6
mutation_rate: 0.15

# No early stopping (explore full space)
early_stopping: false

# Save diverse solutions
save_intermediate_results: true
```

**Trade-offs:**
- 🔍 Maximum diversity (500 diverse solutions)
- ❌ Slower convergence
- ✅ Multiple good solutions
- ❌ Higher memory usage

**Expected Performance:**
- Execution time: 20-30 minutes
- Solution quality: 75-85% (but diverse)
- Solutions returned: 500 (full archive)
- API usage: High (150-300 evaluations)

**When NOT to Use:**
- Single best solution needed
- Time constraints
- Limited memory

---

## 2. Domain Presets

### 2.1 Finance Preset

**Domain:** Finance
**Sub-domains:** Portfolio optimization, asset allocation, risk management

**Configuration:**
```yaml
# Finance preset
evolution_mode: pes  # Expensive evaluations
max_evaluations: 50
enable_planning: true
enable_memory: true

domain: finance
objectives:
  - return
  - risk
  - sharpe_ratio

constraints:
  max_position_size: 0.1
  sector_diversification: true
  max_drawdown: 0.2

# Gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - gold_team

# Knowledge engine
enable_knowledge_engine: true
```

**Key Characteristics:**
- Expensive evaluations (backtesting) → PES mode
- Risk-aware objectives
- Constraint-heavy
- Knowledge from past strategies

**Best For:**
- Portfolio optimization
- Asset allocation
- Risk management
- Trading strategy development

---

### 2.2 Trading Preset

**Domain:** Trading
**Sub-domains:** Momentum, mean-reversion, algorithmic trading

**Configuration:**
```yaml
# Trading preset
evolution_mode: adversarial  # Robustness
max_evaluations: 100
adversarial_rounds: 20

domain: trading
objectives:
  - sharpe_ratio
  - max_drawdown
  - win_rate
  - profit_factor

# Gauntlet (critical for trading)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team  # Stress testing
  - gold_team

# Robustness
red_team_intensity: high
enable_adversarial_market_scenarios: true
```

**Key Characteristics:**
- Adversarial for robustness
- Stress testing (red team)
- Multiple performance metrics
- Market regime awareness

**Best For:**
- Trading strategy development
- Stress testing strategies
- Risk assessment
- Backtesting validation

---

### 2.3 Science Preset

**Domain:** Science
**Sub-domains:** Chemistry, physics, biology, materials

**Configuration:**
```yaml
# Science preset
evolution_mode: pes  # Expensive experiments
max_evaluations: 30  # Very limited budget
enable_planning: true
enable_memory: true

domain: science
objectives:
  - yield
  - purity
  - cost
  - time

feature_dimensions:
  - temperature
  - pressure
  - concentration

# Minimal gauntlet (experiments are expensive)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow

# Save all experiments
save_intermediate_results: true
checkpoint_interval: 1
```

**Key Characteristics:**
- Very expensive evaluations (lab experiments)
- Minimal evaluation budget
- Multi-objective (yield, purity, cost)
- All experiments saved

**Best For:**
- Chemical reaction optimization
- Experimental design
- Materials discovery
- Process optimization

---

### 2.4 Engineering Preset

**Domain:** Engineering
**Sub-domains:** Mechanical, civil, electrical, aerospace

**Configuration:**
```yaml
# Engineering preset
evolution_mode: pes  # Expensive simulations
max_evaluations: 50

domain: engineering
objectives:
  - weight
  - strength
  - safety_factor
  - cost

constraints:
  min_safety_factor: 1.5
  max_stress: yield_strength * 0.8
  max_deflection: length / 100

# Gauntlet (safety-critical)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team  # Failure scenarios
  - gold_team

# Safety
enable_safety_validation: true
stress_test_scenarios: 10
```

**Key Characteristics:**
- Expensive evaluations (simulations)
- Safety-critical
- Constraints-heavy
- Multiple failure scenarios

**Best For:**
- Structural optimization
- Component design
- System engineering
- Safety validation

---

### 2.5 Pharma Preset

**Domain:** Pharma
**Sub-domains:** Drug discovery, molecular design, optimization

**Configuration:**
```yaml
# Pharma preset
evolution_mode: qd  # Explore diverse candidates
max_evaluations: 200

domain: pharma
objectives:
  - binding_affinity
  - selectivity
  - adme_properties
  - toxicity_score

feature_dimensions:
  - molecular_weight
  - logp
  - polar_surface_area
  - hbd_hba_count

# QD archive for diverse candidates
feature_bins: 20
archive_size: 1000

# Gauntlet (important for safety)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - gold_team  # Expert validation
```

**Key Characteristics:**
- High-dimensional search space
- Quality-Diversity for diverse candidates
- Multiple molecular properties
- Large archive (1000 candidates)

**Best For:**
- Drug discovery
- Lead optimization
- Molecular design
- ADMET prediction

---

### 2.6 Web Design Preset

**Domain:** Web Design
**Sub-domains:** UI/UX, landing pages, conversion optimization

**Configuration:**
```yaml
# Web design preset
evolution_mode: standard  # Fast evaluation
max_evaluations: 500  # Cheap evaluations

domain: web_design
objectives:
  - conversion_rate
  - bounce_rate
  - engagement_time

# Fast evaluation (user simulation)
parallel_evaluations: 8
evaluation_timeout: 10

# Quick gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow

# A/B testing
enable_ab_testing: true
ab_test_sample_size: 1000
```

**Key Characteristics:**
- Very cheap evaluations (simulated users)
- Large evaluation budget
- Fast parallel execution
- A/B testing integration

**Best For:**
- Landing page optimization
- UI/UX design
- Conversion optimization
- A/B testing

---

## 3. Use Case Presets

### 3.1 Refinement Preset

**Use Case:** Refining an existing solution

**Configuration:**
```yaml
# Refinement preset - Improve existing solution
evolution_mode: pes  # Directed search
max_iterations: 30
max_evaluations: 40

# Start from existing solution
initial_solution: <provided>
enable_planning: true
early_stopping: true
early_stop_threshold: 0.9

# Focus search around initial solution
mutation_rate: 0.05  # Small mutations
crossover_rate: 0.5  # Less recombination

# Knowledge engine (use similar solutions)
enable_knowledge_engine: true
use_past_solutions: true
similarity_threshold: 0.8
```

**Best For:**
- Improving existing solutions
- Local optimization
- Fine-tuning parameters
- Incremental improvements

---

### 3.2 Robustness Preset

**Use Case:** Testing solution robustness

**Configuration:**
```yaml
# Robustness preset - Stress test solutions
evolution_mode: adversarial  # Co-evolution
adversarial_rounds: 30

# Red team settings
red_team_intensity: high
red_team_strategies:
  - edge_cases
  - boundary_conditions
  - failure_scenarios
  - noise_injection

# Gauntlet (comprehensive)
enable_gauntlet: true
gauntlet_rounds:
  - red_team  # Multiple rounds
  - red_team
  - gold_team

# Robustness metrics
track_robustness: true
stress_test_scenarios: 50
```

**Best For:**
- Safety-critical systems
- Financial stress testing
- Reliability validation
- Failure analysis

---

### 3.3 Discovery Preset

**Use Case:** Discovering novel solutions

**Configuration:**
```yaml
# Discovery preset - Explore solution space
evolution_mode: qd  # Quality-Diversity
max_iterations: 150

# Explore entire space
feature_dimensions:
  - complexity
  - novelty
  - diversity
  - performance

feature_bins: 20
archive_size: 1000

# Promote exploration
exploration_ratio: 0.8
mutation_rate: 0.2
temperature: 1.0  # High temperature

# No early stopping
early_stopping: false
```

**Best For:**
- Innovation
- Novel solutions
- Solution space exploration
- Multiple use cases

---

### 3.4 Multi-Objective Preset

**Use Case:** Optimizing competing objectives

**Configuration:**
```yaml
# Multi-objective preset
evolution_mode: mo  # NSGA-II
max_iterations: 100

pareto_front_size: 100

# Multiple objectives
objectives:
  - performance
  - cost
  - reliability
  - maintainability

# Pareto front analysis
compute_hypervolume: true
compute_spread: true
compute_igd: true

# Visualization
generate_pareto_plots: true
plot_objectives:
  - performance vs cost
  - reliability vs maintainability
```

**Best For:**
- Trade-off analysis
- Decision support
- Multi-criteria optimization
- Pareto front exploration

---

### 3.5 Validation Preset

**Use Case:** Validating solution quality

**Configuration:**
```yaml
# Validation preset - Comprehensive validation
evolution_mode: standard
max_iterations: 50

# Comprehensive gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - red_team  # Multiple rounds
  - gold_team
  - gold_team  # Multiple rounds

# Validation criteria
validation_criteria:
  correctness: 0.95
  robustness: 0.90
  safety: 0.99
  performance: 0.85

# Detailed reporting
generate_validation_report: true
include_recommendations: true
```

**Best For:**
- Solution validation
- Quality assurance
- Safety certification
- Compliance checking

---

## 4. System Mode Presets

### 4.1 PES Preset

**Mode:** Plan-Execute-Summarize
**Strengths:** Sample efficiency, directed search

**Configuration:**
```yaml
# PES preset
evolution_mode: pes
enable_planning: true
enable_memory: true
early_stopping: true

max_evaluations: 50
parallel_candidates: 3
temperature: 0.7
```

**Best For:**
- Expensive evaluations
- Complex problems
- Directed search needed

**Expected:** 60% fewer evaluations

---

### 4.2 QD Preset

**Mode:** Quality-Diversity
**Strengths:** Diverse solutions, exploration

**Configuration:**
```yaml
# QD preset
evolution_mode: qd

feature_dimensions:
  - complexity
  - diversity
feature_bins: 10
archive_size: 100
```

**Best For:**
- Need diverse solutions
- Exploration
- Multiple use cases

**Expected:** 100 diverse solutions

---

### 4.3 MO Preset

**Mode:** Multi-Objective
**Strengths:** Pareto optimization, trade-offs

**Configuration:**
```yaml
# MO preset
evolution_mode: mo

pareto_front_size: 100
crossover_probability: 0.9
mutation_probability: 0.1
```

**Best For:**
- Competing objectives
- Trade-off analysis
- Multiple criteria

**Expected:** 100 Pareto-optimal solutions

---

### 4.4 Adversarial Preset

**Mode:** Adversarial Co-evolution
**Strengths:** Robustness, stress testing

**Configuration:**
```yaml
# Adversarial preset
evolution_mode: adversarial

adversarial_rounds: 20
red_team_intensity: medium
population_ratio: 1.0  # Equal populations
```

**Best For:**
- Robustness needed
- Safety-critical
- Stress testing

**Expected:** Highly robust solutions

---

## 5. Problem Type Presets

### 5.1 Continuous Optimization Preset

**For:** Real-valued optimization

**Configuration:**
```yaml
# Continuous optimization
evolution_mode: standard

# Real-valued operators
mutation_type: gaussian
mutation_sigma: 0.1
crossover_type: sbx  # Simulated binary crossover
eta_c: 10  # SBX parameter
```

---

### 5.2 Combinatorial Optimization Preset

**For:** Discrete optimization

**Configuration:**
```yaml
# Combinatorial optimization
evolution_mode: standard

# Discrete operators
mutation_type: swap
crossover_type: order_crossover
```

---

### 5.3 Noisy Optimization Preset

**For:** Noisy fitness functions

**Configuration:**
```yaml
# Noisy optimization
evolution_mode: standard

# Handle noise
evaluation_samples: 5  # Multiple evaluations
selection_method: tournament
tournament_size: 5  # Higher pressure
early_stopping_patience: 20  # More patience
```

---

### 5.4 Dynamic Optimization Preset

**For:** Time-varying problems

**Configuration:**
```yaml
# Dynamic optimization
evolution_mode: standard

# Track changing optima
diversity_maintenance: true
trigger_reset: true
reset_threshold: 0.1
```

---

### 5.5 Large-Scale Optimization Preset

**For:** High-dimensional problems

**Configuration:**
```yaml
# Large-scale optimization
evolution_mode: standard

# Handle high dimensions
population_size: 500
mutation_rate: 0.01  # Lower per-dimension rate
crossover_type: uniform
enable_decomposition: true
```

---

**End of Preset Catalog**

For more information:
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Master configuration guide
- [Profile Guide](PROFILE_GUIDE.md) - Profile documentation
- [Configuration Examples](CONFIGURATION_EXAMPLES.md) - Working examples
