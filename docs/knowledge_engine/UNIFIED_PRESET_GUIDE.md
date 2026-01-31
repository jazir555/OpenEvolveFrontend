# Comprehensive Preset Documentation

**Version:** 2.0
**Date:** January 30, 2026
**Status:** Production Ready

Complete documentation for the OpenEvolve Unified Evolution Engine configuration preset system with 36+ presets.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Preset Categories](#preset-categories)
4. [Performance Presets (4)](#performance-presets)
5. [Domain Presets (18)](#domain-presets)
6. [Use Case Presets (5)](#use-case-presets)
7. [System Mode Presets (4)](#system-mode-presets)
8. [Problem Type Presets (5)](#problem-type-presets)
9. [Preset Manager API](#preset-manager-api)
10. [Creating Custom Presets](#creating-custom-presets)
11. [Best Practices](#best-practices)

---

## Introduction

Configuration presets provide ready-to-use settings optimized for specific scenarios. The preset system includes **36+ presets** across **5 categories**:

- **4 Performance presets**: Speed vs quality trade-offs
- **18 Domain presets**: Finance, Trading, Science, Engineering, Pharma, Web Design
- **5 Use Case presets**: Common usage scenarios
- **4 System Mode presets**: System selection (OpenEvolve, LoongFlow, Hybrid)
- **5 Problem Type presets**: Problem characteristics

### Benefits

✅ **Quick Start**: Get started immediately without parameter tuning
✅ **Best Practices**: Expert-tuned configurations
✅ **Reproducibility**: Consistent results across runs
✅ **Documentation**: Detailed when-to-use guidance
✅ **Validation**: Built-in validation and error checking

---

## Quick Start

### Basic Usage

```python
from openevolve.unified.presets import FastPreset
from openevolve.unified.config import UnifiedEvolutionConfig

# Create preset
preset = FastPreset()

# Convert to unified config
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Use in evolution
result = await evolve(problem_code, config=config)
```

### Using Preset Manager

```python
from openevolve.unified.presets import get_preset_manager

# Get manager
manager = get_preset_manager()

# List all presets
presets = manager.list_presets()
print(f"Available presets: {presets}")

# List by category
performance = manager.list_presets(category="performance")

# Get preset info
info = manager.get_preset_info("fast")
print(info.description)

# Apply preset
config = manager.apply_preset("fast")

# Compare presets
comparison = manager.compare_presets("fast", "thorough")
print(f"Differences: {comparison.differences}")
```

### Choosing a Preset

**Not sure which preset to use?** Start with these:

| Scenario | Recommended Preset |
|----------|-------------------|
| Default/General | `balanced` |
| Quick prototyping | `fast` |
| Production deployment | `production` |
| Limited budget | `budget` |
| Finance/trading | `finance_general` |
| Research | `research` |
| Safety-critical | `safety_critical` |

---

## Preset Categories

### Overview

```
openevolve/unified/presets/
├── base.py              # Base classes
├── performance.py       # 4 performance presets
├── domains.py          # 18 domain presets
├── use_cases.py        # 5 use case presets
├── systems.py          # 4 system mode presets
├── problem_types.py    # 5 problem type presets
└── manager.py          # Preset manager
```

### Category Summary

| Category | Count | Description |
|----------|-------|-------------|
| Performance | 4 | Speed/resource optimization |
| Domain | 18 | Domain-specific configurations (6 domains × 3) |
| Use Case | 5 | Common usage scenarios |
| System | 4 | Evolutionary system selection |
| Problem Type | 5 | Problem characteristic optimization |
| **Total** | **36** | All presets |

---

## Performance Presets

Optimize for different speed vs quality trade-offs.

### Fast Preset

**Maximum speed for rapid prototyping**

**When to Use:**
- Early development and prototyping
- Testing ideas quickly
- Resource-constrained environments
- Proof-of-concept work

**Key Parameters:**
- `max_iterations`: 20
- `population_size`: 100
- `concurrency`: 3
- `timeout`: 120 seconds
- `log_level`: WARNING

**Trade-offs:**
- ⚡⚡⚡ Speed: Very fast (seconds/minutes)
- ⚠️ Quality: Lower quality
- ❌ Validation: No validation
- ✅ Cost: Low cost (minimal API calls)

**Example:**
```python
from openevolve.unified.presets import FastPreset

preset = FastPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(problem_code, config=config)
```

**Related Presets:** `balanced`, `budget`

---

### Balanced Preset

**Default configuration for most use cases**

**When to Use:**
- General evolution tasks
- Production workflows
- When unsure which preset to use

**Key Parameters:**
- `max_iterations`: 100
- `population_size`: 500
- `concurrency`: 5
- `timeout`: 300 seconds

**Trade-offs:**
- ⚡⚡ Speed: Moderate speed (minutes/hours)
- ✅ Quality: Good quality (80-90%)
- ✅ Validation: Standard validation
- ⚖️ Cost: Moderate cost

**Example:**
```python
from openevolve.unified.presets import BalancedPreset

preset = BalancedPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(problem_code, config=config)
```

**Related Presets:** `fast`, `thorough`

---

### Thorough Preset

**Maximum quality regardless of time or cost**

**When to Use:**
- Production-critical systems
- Final optimization passes
- Research publications
- When quality is paramount

**Key Parameters:**
- `max_iterations`: 500
- `population_size`: 2000
- `concurrency`: 10
- `timeout`: 600 seconds
- `log_level`: DEBUG

**Trade-offs:**
- 🐌🐌 Speed: Very slow (hours/days)
- ✅✅✅ Quality: Maximum quality (95-99%)
- ✅✅✅ Validation: Comprehensive validation
- 💰💰💰 Cost: Very high cost

**Example:**
```python
from openevolve.unified.presets import ThoroughPreset

preset = ThoroughPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(critical_code, config=config)
```

**Related Presets:** `balanced`, `quality_critical`

---

### Budget Preset

**Work within strict resource limits**

**When to Use:**
- Free tier accounts
- Limited API budgets
- Rate-limited environments
- Cost-sensitive applications

**Key Parameters:**
- `max_iterations`: 10
- `population_size`: 50
- `concurrency`: 1 (sequential)
- `timeout`: 60 seconds
- `log_level`: ERROR

**Trade-offs:**
- ⚡⚡ Speed: Fast (minimal computation)
- ⚠️⚠️ Quality: Very limited quality
- ❌ Validation: No validation
- ✅✅ Cost: Minimal cost (few API calls)

**Example:**
```python
from openevolve.unified.presets import BudgetPreset

preset = BudgetPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(problem_code, config=config)
```

**Related Presets:** `fast`, `resource_constrained`

---

## Domain Presets

Domain-specific configurations for specialized optimization.

### Finance Domain (3 Presets)

#### Finance General Preset

**General finance optimization tasks**

**Evolution Mode:** PES (planning-based)
**When to Use:** Portfolio management, risk analysis, trading strategies

**Key Features:**
- PES mode for structured evolution
- Risk-aware optimization
- Reproducible results (fixed seed)
- Strict evaluation limits (5 minutes)

**Parameters:**
- `max_iterations`: 50
- `population_size`: 200
- `timeout`: 300

**Example:**
```python
from openevolve.unified.presets import FinanceGeneralPreset

preset = FinanceGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(portfolio_code, config=config)
```

**Related:** `finance_portfolio`, `finance_risk`

---

#### Finance Portfolio Preset

**Multi-objective portfolio optimization**

**Evolution Mode:** MO (multi-objective)
**When to Use:** Portfolio optimization with multiple objectives

**Key Features:**
- Multi-objective (NSGA-II)
- Objectives: return, risk, Sharpe ratio
- Pareto front output

**Parameters:**
- `max_iterations`: 100
- `population_size`: 300
- `pareto_archive_size`: 100

**Example:**
```python
from openevolve.unified.presets import FinancePortfolioPreset

preset = FinancePortfolioPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
pareto_front = await evolve(portfolio_code, config=config)
```

**Related:** `finance_general`, `finance_risk`

---

#### Finance Risk Preset

**Risk analysis and VaR optimization**

**Evolution Mode:** QD (quality diversity)
**When to Use:** Risk analysis, VaR/CVaR optimization, stress testing

**Key Features:**
- QD for diverse risk scenarios
- Metrics: VaR, CVaR, max drawdown
- Archive of risk-optimized strategies

**Parameters:**
- `max_iterations`: 75
- `population_size`: 400
- `grid_resolution`: 15

**Example:**
```python
from openevolve.unified.presets import FinanceRiskPreset

preset = FinanceRiskPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
risk_archive = await evolve(risk_code, config=config)
```

**Related:** `finance_general`, `finance_portfolio`

---

### Trading Domain (3 Presets)

#### Trading General Preset

**Trading strategy development**

**Evolution Mode:** Adversarial
**When to Use:** Trading strategy development, signal optimization

**Key Features:**
- Adversarial training for robustness
- Generator vs discriminator
- Arms race dynamics

**Parameters:**
- `max_iterations`: 40
- `population_size`: 150
- `adversarial_rounds`: 20

**Example:**
```python
from openevolve.unified.presets import TradingGeneralPreset

preset = TradingGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
strategy = await evolve(trading_code, config=config)
```

**Related:** `trading_signal`, `trading_parameter`

---

#### Trading Signal Preset

**Trading signal optimization**

**Evolution Mode:** QD
**When to Use:** Signal optimization, feature engineering

**Key Features:**
- QD for diverse signal types
- Metrics: Sharpe ratio, win rate, profit factor
- Archive of signals

**Parameters:**
- `max_iterations`: 60
- `population_size`: 300
- `grid_resolution`: 12

**Example:**
```python
from openevolve.unified.presets import TradingSignalPreset

preset = TradingSignalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
signals = await evolve(signal_code, config=config)
```

**Related:** `trading_general`, `trading_parameter`

---

#### Trading Parameter Preset

**Parameter tuning for trading strategies**

**Evolution Mode:** PES
**When to Use:** Parameter tuning, strategy calibration

**Key Features:**
- Planning-based for efficiency
- Focused parameter search
- Fast convergence

**Parameters:**
- `max_iterations`: 30
- `population_size`: 100

**Example:**
```python
from openevolve.unified.presets import TradingParameterPreset

preset = TradingParameterPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
tuned = await evolve(strategy_code, config=config)
```

**Related:** `trading_general`, `trading_signal`

---

### Science Domain (3 Presets)

#### Science General Preset

**General scientific computing optimization**

**Evolution Mode:** OpenEvolve
**When to Use:** Scientific computing, numerical optimization

**Key Features:**
- High precision numerical computation
- Reproducible results
- Standard evolution

**Parameters:**
- `max_iterations`: 80
- `population_size`: 300

**Example:**
```python
from openevolve.unified.presets import ScienceGeneralPreset

preset = ScienceGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(science_code, config=config)
```

**Related:** `science_optimization`, `science_discovery`

---

#### Science Optimization Preset

**Numerical optimization and function maximization**

**Evolution Mode:** QD
**When to Use:** Function optimization, finding global optima

**Key Features:**
- QD for multiple optima
- Explore solution landscape
- Archive of diverse solutions

**Parameters:**
- `max_iterations`: 100
- `population_size`: 500

**Example:**
```python
from openevolve.unified.presets import ScienceOptimizationPreset

preset = ScienceOptimizationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
optima = await evolve(function_code, config=config)
```

**Related:** `science_general`, `science_discovery`

---

#### Science Discovery Preset

**Novel algorithm discovery for research**

**Evolution Mode:** QD
**When to Use:** Research, novel algorithm discovery

**Key Features:**
- Novelty search enabled
- Maximum solution diversity
- Long exploration phase

**Parameters:**
- `max_iterations`: 150
- `population_size`: 800
- `grid_resolution`: 25

**Example:**
```python
from openevolve.unified.presets import ScienceDiscoveryPreset

preset = ScienceDiscoveryPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
novel = await evolve(research_code, config=config)
```

**Related:** `science_general`, `science_optimization`

---

### Engineering Domain (3 Presets)

#### Engineering General Preset

**General engineering optimization**

**Evolution Mode:** PES
**When to Use:** Engineering design, optimization problems

**Key Features:**
- Planning-based design
- Engineering constraints
- Practical solutions

**Parameters:**
- `max_iterations`: 70
- `population_size`: 250

**Example:**
```python
from openevolve.unified.presets import EngineeringGeneralPreset

preset = EngineeringGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
design = await evolve(engineering_code, config=config)
```

**Related:** `engineering_design`, `engineering_control`

---

#### Engineering Design Preset

**Multi-objective engineering design**

**Evolution Mode:** MO
**When to Use:** Multi-constraint design, Pareto optimization

**Key Features:**
- Objectives: cost, performance, reliability
- Pareto front output
- Constraint enforcement

**Parameters:**
- `max_iterations`: 120
- `population_size`: 400

**Example:**
```python
from openevolve.unified.presets import EngineeringDesignPreset

preset = EngineeringDesignPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
designs = await evolve(design_code, config=config)
```

**Related:** `engineering_general`, `engineering_control`

---

#### Engineering Control Preset

**Control systems optimization**

**Evolution Mode:** OpenEvolve
**When to Use:** PID controller tuning, control system design

**Key Features:**
- Stability and performance focus
- Standard evolution
- Optimized control parameters

**Parameters:**
- `max_iterations`: 60
- `population_size`: 200

**Example:**
```python
from openevolve.unified.presets import EngineeringControlPreset

preset = EngineeringControlPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
controller = await evolve(control_code, config=config)
```

**Related:** `engineering_general`, `engineering_design`

---

### Pharma Domain (3 Presets)

#### Pharma General Preset

**General pharmaceutical research optimization**

**Evolution Mode:** QD
**When to Use:** Drug discovery, molecular optimization

**Key Features:**
- QD for diverse candidates
- Safety constraints
- Archive of candidates

**Parameters:**
- `max_iterations`: 100
- `population_size`: 500

**Example:**
```python
from openevolve.unified.presets import PharmaGeneralPreset

preset = PharmaGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
candidates = await evolve(molecular_code, config=config)
```

**Related:** `pharma_drug_discovery`, `pharma_clinical`

---

#### Pharma Drug Discovery Preset

**Lead optimization and drug discovery**

**Evolution Mode:** QD
**When to Use:** Lead optimization, ADMET prediction

**Key Features:**
- Objectives: efficacy, safety, ADMET
- Chemical space exploration
- Ranked candidates

**Parameters:**
- `max_iterations`: 150
- `population_size`: 800
- `grid_resolution`: 20

**Example:**
```python
from openevolve.unified.presets import PharmaDrugDiscoveryPreset

preset = PharmaDrugDiscoveryPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
leads = await evolve(lead_code, config=config)
```

**Related:** `pharma_general`, `pharma_clinical`

---

#### Pharma Clinical Preset

**Clinical trial optimization and analysis**

**Evolution Mode:** PES
**When to Use:** Clinical trial design, treatment protocol optimization

**Key Features:**
- Structured trial design
- Patient safety prioritized
- Planning-based

**Parameters:**
- `max_iterations`: 80
- `population_size`: 300

**Example:**
```python
from openevolve.unified.presets import PharmaClinicalPreset

preset = PharmaClinicalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
trial = await evolve(trial_code, config=config)
```

**Related:** `pharma_general`, `pharma_drug_discovery`

---

### Web Design Domain (3 Presets)

#### Web Design General Preset

**General web design and frontend optimization**

**Evolution Mode:** OpenEvolve
**When to Use:** Frontend optimization, UX improvement

**Key Features:**
- Fast iteration
- Good quality
- Standard evolution

**Parameters:**
- `max_iterations`: 60
- `population_size`: 200

**Example:**
```python
from openevolve.unified.presets import WebDesignGeneralPreset

preset = WebDesignGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
frontend = await evolve(web_code, config=config)
```

**Related:** `web_design_ux`, `web_design_performance`

---

#### Web Design UX Preset

**User experience and accessibility optimization**

**Evolution Mode:** MO
**When to Use:** UX optimization, accessibility improvements

**Key Features:**
- Objectives: accessibility, usability, performance
- Multi-objective optimization
- Pareto-optimal UX solutions

**Parameters:**
- `max_iterations`: 80
- `population_size`: 300

**Example:**
```python
from openevolve.unified.presets import WebDesignUxPreset

preset = WebDesignUxPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
ux = await evolve(ux_code, config=config)
```

**Related:** `web_design_general`, `web_design_performance`

---

#### Web Design Performance Preset

**Web performance and Core Web Vitals optimization**

**Evolution Mode:** OpenEvolve
**When to Use:** Performance optimization, Core Web Vitals

**Key Features:**
- Focus: load time, responsiveness, stability
- Performance tuning
- Optimized frontend code

**Parameters:**
- `max_iterations`: 70
- `population_size`: 250

**Example:**
```python
from openevolve.unified.presets import WebDesignPerformancePreset

preset = WebDesignPerformancePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
performance = await evolve(perf_code, config=config)
```

**Related:** `web_design_general`, `web_design_ux`

---

## Use Case Presets

Optimized for common usage scenarios.

### Quick Prototype Preset

**Rapid prototyping with fast feedback loops**

**When to Use:**
- Early development
- Idea validation
- Proof of concept
- Results needed in seconds/minutes

**Key Parameters:**
- `max_iterations`: 10
- `population_size`: 50
- `concurrency`: 2
- `timeout`: 60
- `log_level`: ERROR

**Trade-offs:**
- ⚡⚡⚡ Speed: Very fast
- ⚠️ Quality: Low
- ❌ Validation: None
- ✅✅ Cost: Very low

**Example:**
```python
from openevolve.unified.presets import QuickPrototypePreset

preset = QuickPrototypePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
poc = await evolve(code, config=config)
```

**Related:** `fast`, `resource_constrained`

---

### Production Preset

**Production deployment with comprehensive validation**

**When to Use:**
- Production deployment
- Code quality is critical
- Final optimization before release

**Key Parameters:**
- `max_iterations`: 200
- `population_size`: 800
- `concurrency`: 8
- `timeout`: 600

**Trade-offs:**
- 🐌 Speed: Slow
- ✅✅✅ Quality: Maximum
- ✅✅✅ Validation: Comprehensive
- 💰💰 Cost: High

**Example:**
```python
from openevolve.unified.presets import ProductionPreset

preset = ProductionPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
production_code = await evolve(code, config=config)
```

**Related:** `thorough`, `quality_critical`

---

### Research Preset

**Research exploration with novelty emphasis**

**When to Use:**
- Academic research
- Novel algorithm discovery
- Exploring solution space
- Publication work

**Key Parameters:**
- `max_iterations`: 200
- `population_size`: 1000
- `log_level`: DEBUG

**Trade-offs:**
- 🔬 Focus: Novelty and diversity
- 📊 Output: Archive of diverse solutions
- ⏰ Time: Long exploration
- ✅ Quality: High with diversity

**Example:**
```python
from openevolve.unified.presets import ResearchPreset

preset = ResearchPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
novel_solutions = await evolve(research_code, config=config)
```

**Related:** `thorough`, `science_discovery`

---

### Resource Constrained Preset

**Work within strict resource limits**

**When to Use:**
- Free tier accounts
- Limited API budget
- Rate-limited environments
- Minimal compute

**Key Parameters:**
- `max_iterations`: 15
- `population_size`: 40
- `concurrency`: 1 (sequential)
- `timeout`: 60
- `log_level`: WARNING

**Trade-offs:**
- 💾 Resources: Minimal
- ⚡ Speed: Fast
- ⚠️ Quality: Basic
- ✅✅ Cost: Very low

**Example:**
```python
from openevolve.unified.presets import ResourceConstrainedPreset

preset = ResourceConstrainedPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(code, config=config)
```

**Related:** `budget`, `fast`

---

### Quality Critical Preset

**Maximum quality assurance for critical systems**

**When to Use:**
- Safety-critical systems
- Medical devices
- Aerospace
- Failures are unacceptable

**Key Parameters:**
- `max_iterations`: 300
- `population_size`: 1500
- `concurrency`: 10
- `timeout`: 900 (15 minutes)
- `log_level`: DEBUG

**Trade-offs:**
- 🐌🐌 Speed: Very slow
- ✅✅✅ Quality: Maximum
- ✅✅✅✅ Validation: Comprehensive
- 💰💰💰 Cost: Very high

**Example:**
```python
from openevolve.unified.presets import QualityCriticalPreset

preset = QualityCriticalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
verified_code = await evolve(critical_code, config=config)
```

**Related:** `thorough`, `production`, `safety_critical`

---

## System Mode Presets

Control which evolutionary system(s) to use.

### Pure OpenEvolve Preset

**Use only OpenEvolve (no LoongFlow features)**

**Evolution Mode:** OpenEvolve

**When to Use:**
- Standard code evolution
- Don't need planning
- Pure OpenEvolve workflow

**Key Features:**
- OpenEvolve only
- No planning phase
- Standard evolutionary operators
- Fast direct evolution

**Example:**
```python
from openevolve.unified.presets import PureOpenEvolvePreset

preset = PureOpenEvolvePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(code, config=config)
```

**Related:** `pure_loongflow`, `hybrid_auto`

---

### Pure LoongFlow Preset

**Use only LoongFlow PES (Plan-Evolve-Summarize)**

**Evolution Mode:** PES

**When to Use:**
- Complex problems
- Planning is beneficial
- Structured problem-solving workflow

**Key Features:**
- LoongFlow PES only
- Planning phase enabled
- Long-term memory
- Evolution summarization

**Example:**
```python
from openevolve.unified.presets import PureLoongFlowPreset

preset = PureLoongFlowPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(code, config=config)
```

**Related:** `pure_openevolve`, `hybrid_auto`

---

### Hybrid Auto Preset

**Auto-select the best evolutionary system**

**Evolution Mode:** Hybrid

**When to Use:**
- Unsure which system to use
- Want automatic selection
- Maximum flexibility

**Key Features:**
- Hybrid - auto-selection
- Maximum flexibility
- Adaptive to problem
- Slight overhead from auto-selection

**Example:**
```python
from openevolve.unified.presets import HybridAutoPreset

preset = HybridAutoPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(code, config=config)
```

**Related:** `pure_openevolve`, `pure_loongflow`

---

### Custom Preset

**User-defined custom configuration**

**When to Use:**
- Specific requirements
- None of the presets fit
- Full control needed

**Key Features:**
- Full control
- Higher complexity
- Maximum flexibility

**Example:**
```python
from openevolve.unified.presets import CustomPreset

preset = CustomPreset(
    evolution_mode="pes",
    max_iterations=150,
    population_size=600,
    concurrency=8
)
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(code, config=config)
```

**Related:** `balanced`, `pure_openevolve`

---

## Problem Type Presets

Optimized for different problem characteristics.

### Single Objective Preset

**Single objective optimization**

**Evolution Mode:** OpenEvolve

**When to Use:**
- Single metric optimization
- Clear success criteria
- Standard maximization/minimization

**Key Features:**
- 1 objective
- Standard evolution
- Best solution found
- Simple optimization

**Example:**
```python
from openevolve.unified.presets import SingleObjectivePreset

preset = SingleObjectivePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
best = await evolve(code, config=config)
```

**Related:** `multi_objective`, `balanced`

---

### Multi Objective Preset

**Multi-objective Pareto optimization**

**Evolution Mode:** MO

**When to Use:**
- Multiple objectives
- Need Pareto analysis
- Exploring trade-offs

**Key Features:**
- 2+ objectives
- NSGA-II / NSGA-III
- Pareto front output
- Higher complexity

**Example:**
```python
from openevolve.unified.presets import MultiObjectivePreset

preset = MultiObjectivePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
pareto_front = await evolve(code, config=config)
```

**Related:** `single_objective`, `finance_portfolio`

---

### Expensive Evaluation Preset

**Optimization with very expensive evaluations**

**Evolution Mode:** PES

**When to Use:**
- Very expensive evaluations
- Limited evaluation budget
- Computational simulations
- Hours per evaluation

**Key Features:**
- Minimize evaluation count
- Planning-based intelligent search
- Low parallelism
- Early stopping enabled

**Example:**
```python
from openevolve.unified.presets import ExpensiveEvaluationPreset

preset = ExpensiveEvaluationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
result = await evolve(expensive_code, config=config)
```

**Related:** `fast_evaluation`, `budget`

---

### Fast Evaluation Preset

**Optimization with very fast evaluations**

**Evolution Mode:** QD

**When to Use:**
- Very fast evaluations
- Can afford many iterations
- Want extensive exploration

**Key Features:**
- Maximize exploration
- QD - explore entire space
- High parallelism
- Efficient

**Example:**
```python
from openevolve.unified.presets import FastEvaluationPreset

preset = FastEvaluationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
archive = await evolve(fast_code, config=config)
```

**Related:** `expensive_evaluation`, `thorough`

---

### Safety Critical Preset

**Optimization for safety-critical systems**

**Evolution Mode:** Adversarial

**When to Use:**
- Safety-critical systems
- Medical devices
- Aerospace
- Failures unacceptable

**Key Features:**
- Adversarial testing
- Robust solutions
- Comprehensive validation
- High cost

**Example:**
```python
from openevolve.unified.presets import SafetyCriticalPreset

preset = SafetyCriticalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
verified = await evolve(critical_code, config=config)
```

**Related:** `quality_critical`, `production`

---

## Preset Manager API

The `PresetManager` provides a centralized interface for working with presets.

### Initialization

```python
from openevolve.unified.presets import get_preset_manager

manager = get_preset_manager()
```

### Listing Presets

```python
# List all presets
all_presets = manager.list_presets()
# ['balanced', 'fast', 'finance_general', ...]

# List by category
performance = manager.list_presets(category="performance")
# ['fast', 'balanced', 'thorough', 'budget']

# List by evolution mode
qd_presets = manager.list_presets(evolution_mode="qd")
# ['finance_risk', 'trading_signal', ...]

# Get all categories
categories = manager.list_categories()
# ['performance', 'domain', 'use_case', 'system', 'problem_type']
```

### Getting Presets

```python
# Get preset
preset = manager.get_preset("fast")
print(preset.name)  # "fast"
print(preset.max_iterations)  # 20

# Get preset info
info = manager.get_preset_info("fast")
print(info.description)
print(info.when_to_use)
print(info.trade_offs)

# Print preset summary
manager.print_preset_summary("fast")
```

### Applying Presets

```python
# Apply preset to create config
config = manager.apply_preset("fast")

# Apply to existing config
base_config = UnifiedEvolutionConfig.from_dict({...})
config = manager.apply_preset("fast", base_config=base_config)
```

### Validation

```python
# Validate preset
result = manager.validate_preset("fast")
print(result.is_valid)  # True
print(result.errors)  # []
print(result.warnings)  # []
print(result.info)  # []
```

### Comparison

```python
# Compare two presets
comparison = manager.compare_presets("fast", "thorough")
print(comparison.differences)
# {'max_iterations': (20, 500), 'population_size': (100, 2000), ...}

print(comparison.similarities)
# ['category', 'log_to_console', ...]
```

### Search

```python
# Search by keyword
results = manager.search_presets("finance")
# ['finance_general', 'finance_portfolio', 'finance_risk']
```

### Saving and Loading

```python
# Save preset
preset = manager.get_preset("fast")
manager.save_preset(preset, "fast_preset.yaml", format="yaml")

# Load preset
loaded = manager.load_preset("fast_preset.yaml")
```

### Creating Custom Presets

```python
# Create from config
custom = manager.create_preset(
    name="my_custom",
    config=my_config,
    description="My custom configuration",
    category="custom"
)
```

---

## Creating Custom Presets

### Method 1: Extend BasePreset

```python
from openevolve.unified.presets.base import BasePreset, PresetInfo, Field
from typing import Dict

class MyCustomPreset(BasePreset):
    """My custom preset for specific use case."""

    name: str = "my_custom"
    category: str = "custom"
    description: str = "My custom configuration"

    # Override parameters
    max_iterations: int = Field(default=150, description="Custom iterations")
    population_size: int = Field(default=600, description="Custom population")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="When you need custom behavior",
            trade_offs={
                "Speed": "Custom speed",
                "Quality": "Custom quality"
            },
            related_presets=["balanced"],
            example_usage="""
preset = MyCustomPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Add custom configuration
        config["openevolve"] = {
            "early_stopping_patience": 10,
            "enable_novelty_search": True,
        }
        return config
```

### Method 2: Use CustomPreset Template

```python
from openevolve.unified.presets import CustomPreset

preset = CustomPreset(
    evolution_mode="pes",
    max_iterations=150,
    population_size=600,
    concurrency=8
)

config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())
```

### Method 3: Create from Existing Config

```python
manager = get_preset_manager()

# Start with existing config
base_config = UnifiedEvolutionConfig.from_yaml_file("config.yaml")

# Create preset from it
custom = manager.create_preset(
    name="derived_from_config",
    config=base_config,
    description="Derived from my config.yaml",
    category="custom"
)

# Use it
config = manager.apply_preset("derived_from_config")
```

---

## Best Practices

### Choosing the Right Preset

1. **Start with `balanced`** - Good default for most cases
2. **For speed**: Use `fast` or `quick_prototype`
3. **For quality**: Use `thorough` or `production`
4. **For resources**: Use `budget` or `resource_constrained`
5. **For domains**: Use domain-specific presets
6. **For safety**: Use `safety_critical` or `quality_critical`

### Combining Presets

```python
# Start with a preset
manager = get_preset_manager()
config = manager.apply_preset("balanced")

# Customize specific parameters
config.common.max_iterations = 150
config.common.concurrency = 8

# Use modified config
result = await evolve(code, config=config)
```

### Validation

Always validate presets before use:

```python
manager = get_preset_manager()

# Validate preset
result = manager.validate_preset("fast")

if not result.is_valid:
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
```

### Testing

```python
# Test preset before production
preset = FastPreset()
config_dict = preset.to_unified_config()
config = UnifiedEvolutionConfig.from_dict(config_dict)

# Validate config
config.model_validate(config.model_dump())
```

---

## Summary

This documentation covers **36+ presets** across **5 categories**:

### Performance (4)
- `fast` - Maximum speed
- `balanced` - Default configuration
- `thorough` - Maximum quality
- `budget` - Resource-constrained

### Domain (18)
- Finance: `finance_general`, `finance_portfolio`, `finance_risk`
- Trading: `trading_general`, `trading_signal`, `trading_parameter`
- Science: `science_general`, `science_optimization`, `science_discovery`
- Engineering: `engineering_general`, `engineering_design`, `engineering_control`
- Pharma: `pharma_general`, `pharma_drug_discovery`, `pharma_clinical`
- Web Design: `web_design_general`, `web_design_ux`, `web_design_performance`

### Use Cases (5)
- `quick_prototype` - Rapid prototyping
- `production` - Production deployment
- `research` - Research exploration
- `resource_constrained` - Limited resources
- `quality_critical` - Critical systems

### Systems (4)
- `pure_openevolve` - OpenEvolve only
- `pure_loongflow` - LoongFlow PES only
- `hybrid_auto` - Auto-selection
- `custom` - User-defined

### Problem Types (5)
- `single_objective` - Single metric
- `multi_objective` - Pareto optimization
- `expensive_evaluation` - Limited budget
- `fast_evaluation` - Extensive exploration
- `safety_critical` - Robustness

For more information, see:
- [Unified Configuration Guide](./UNIFIED_CONFIG.md)
- [Configuration API Reference](../api/config.md)
- [Examples](../examples/presets/)
