# Domain-Specific Optimizers

## Overview

Domain-specific optimizers provide specialized configurations and best practices for 6 target domains. Each optimizer encapsulates domain knowledge, evaluation metrics, and evolutionary strategies tailored to that domain's unique characteristics.

## Domains

1. **Finance** - Portfolio optimization, risk analysis, asset allocation
2. **Trading** - Strategy development, signal optimization, parameter tuning
3. **Science** - Experimental design, data analysis, hypothesis testing
4. **Engineering** - Structural optimization, circuit design, control systems
5. **Pharma** - Molecular optimization, drug design, clinical trial design
6. **Web Design** - Landing page optimization, UX optimization, A/B testing

## Quick Start

### Basic Usage

```python
from openevolve.domain import get_optimizer, optimize_by_domain

# Option 1: Auto-detect domain
result = await optimize_by_domain(
    "Optimize portfolio allocation for maximum return"
)

# Option 2: Specify domain explicitly
optimizer = get_optimizer("finance", sub_domain="portfolio")
result = await optimizer.optimize(
    "Maximize return with minimum risk",
    constraints={"max_assets": 50}
)

# Access domain-specific metrics
print(result['domain_metrics']['sharpe_ratio'])
```

### Domain Auto-Detection

```python
from openevolve.domain import detect_domain

# Automatically detect domain from problem description
domain = detect_domain("Optimize trading strategy with entry/exit rules")
print(domain)  # 'trading'

domain = detect_domain("Design lightweight bridge that supports 50 tons")
print(domain)  # 'engineering'
```

## Finance Domain

### Overview

Specialized for financial optimization problems with expensive backtests.

### Best System

- **System**: LoongFlow (PES mode)
- **Why**: Expensive backtests (minutes per eval), needs reasoning to reduce evaluations
- **Improvement**: 60% fewer evaluations via PES

### Sub-Domains

#### General
```python
optimizer = FinanceOptimizer()
```
- **Config**: PES mode, 50 evaluations
- **Use**: General financial optimization

#### Portfolio Optimization
```python
optimizer = FinanceOptimizer(sub_domain="portfolio")
```
- **Config**: Multi-objective (return, risk, liquidity), NSGA-II
- **Use**: Portfolio allocation with multiple objectives

#### Risk Analysis
```python
optimizer = FinanceOptimizer(sub_domain="risk")
```
- **Config**: Lower temperature, 40 evaluations
- **Use**: VaR/CVaR optimization

#### Asset Allocation
```python
optimizer = FinanceOptimizer(sub_domain="asset_allocation")
```
- **Config**: Constraint-heavy optimization
- **Use**: Asset allocation with complex constraints

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `sharpe_ratio` | Risk-adjusted return | > 1.5 |
| `sortino_ratio` | Downside risk-adjusted | > 2.0 |
| `max_drawdown` | Maximum loss from peak | < 0.20 |
| `volatility` | Annualized volatility | < 0.25 |
| `var_95` | Value at risk (95%) | < 0.05 |
| `cvar_95` | Conditional VaR (95%) | < 0.08 |

### Example

```python
from openevolve.domain import FinanceOptimizer

# Initialize optimizer
optimizer = FinanceOptimizer(sub_domain="portfolio")

# Define constraints
constraints = optimizer.get_portfolio_constraints(
    max_assets=30,
    min_weight=0.01,
    max_weight=0.4,
    sectors={"Tech": 0.4, "Healthcare": 0.3}
)

# Optimize
result = await optimizer.optimize(
    "Maximize portfolio return with minimum risk",
    constraints=constraints
)

# Access results
print(f"Sharpe Ratio: {result['domain_metrics']['sharpe_ratio']}")
print(f"Max Drawdown: {result['domain_metrics']['max_drawdown']}")
print(f"Evaluations: {result['evaluations']}")

# Validate portfolio
is_valid, violations = optimizer.validate_portfolio(
    result['best_solution'],
    constraints
)
```

## Trading Domain

### Overview

Specialized for trading strategy development with adversarial testing.

### Best System

- **System**: OpenEvolve (Adversarial mode)
- **Why**: Market regimes change, need robustness against adverse conditions
- **Improvement**: Strategies that survive regime changes

### Sub-Domains

#### General
```python
optimizer = TradingOptimizer()
```
- **Config**: Adversarial mode, 20 rounds
- **Use**: General trading optimization

#### Strategy Development
```python
optimizer = TradingOptimizer(sub_domain="strategy")
```
- **Config**: 30 adversarial rounds, high temperature
- **Use**: Discover entry/exit rules

#### Signal Optimization
```python
optimizer = TradingOptimizer(sub_domain="signal")
```
- **Config**: Lower temperature, 15 rounds
- **Use**: Optimize indicator parameters

#### Parameter Tuning
```python
optimizer = TradingOptimizer(sub_domain="parameter")
```
- **Config**: Very low temperature, 100 iterations
- **Use**: Fine-tune stop loss, take profit

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `total_return` | Total return | > 0.30 |
| `sharpe_ratio` | Risk-adjusted return | > 1.8 |
| `max_drawdown` | Maximum loss | < 0.20 |
| `win_rate` | Winning trades % | > 0.50 |
| `profit_factor` | Profit/loss ratio | > 2.0 |

### Example

```python
from openevolve.domain import TradingOptimizer

# Initialize optimizer
optimizer = TradingOptimizer(sub_domain="strategy")

# Define constraints
constraints = optimizer.get_trading_constraints(
    max_drawdown=0.2,
    min_win_rate=0.5,
    min_profit_factor=1.5
)

# Optimize
result = await optimizer.optimize(
    "Develop momentum trading strategy with entry/exit rules",
    constraints=constraints
)

# Test against adversarial scenarios
scenarios = optimizer.generate_adversarial_scenarios(
    market_data,
    ["regime_change", "volatility_spike", "black_swan"]
)

# Validate strategy
is_valid, violations = optimizer.validate_strategy(
    result['domain_metrics'],
    constraints
)
```

## Science Domain

### Overview

Specialized for scientific research with very expensive experiments.

### Best System

- **System**: Hybrid (LoongFlow PES + OpenEvolve QD)
- **Why**: Very expensive experiments ($10k+ per run), need exploration
- **Improvement**: 60% fewer experiments, diverse solutions

### Sub-Domains

#### General
```python
optimizer = ScienceOptimizer()
```
- **Config**: QD mode, 20 experiments
- **Use**: General scientific optimization

#### Experimental Design
```python
optimizer = ScienceOptimizer(sub_domain="experimental_design")
```
- **Config**: MO + QD, 30 experiments
- **Use**: DOE optimization

#### Data Analysis
```python
optimizer = ScienceOptimizer(sub_domain="data_analysis")
```
- **Config**: Standard mode, 50 iterations
- **Use**: Pipeline optimization

#### Hypothesis Testing
```python
optimizer = ScienceOptimizer(sub_domain="hypothesis_testing")
```
- **Config**: PES mode, 15 experiments
- **Use**: Prioritize hypotheses

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `statistical_power` | Statistical power | > 0.80 |
| `cost_efficiency` | Cost per result | > 0.70 |
| `discovery_rate` | Novel findings | > 0.50 |
| `reproducibility` | Reproducibility | > 0.85 |

### Example

```python
from openevolve.domain import ScienceOptimizer

# Initialize optimizer
optimizer = ScienceOptimizer(sub_domain="experimental_design")

# Define constraints
constraints = optimizer.get_experiment_constraints(
    max_experiments=20,
    budget=50000,
    cost_per_experiment=2500
)

# Get DOE suggestions
doe_params = optimizer.suggest_doe_parameters(
    "Optimize chemical reaction",
    num_factors=5,
    resolution="IV"
)

# Optimize
result = await optimizer.optimize(
    "Optimize chemical reaction conditions for maximum yield",
    constraints=constraints
)

# Estimate cost
cost = optimizer.estimate_experiment_cost(
    result['best_solution'],
    constraints
)
```

## Engineering Domain

### Overview

Specialized for engineering design with FEA simulations and safety requirements.

### Best System

- **System**: Hybrid (LoongFlow PES + Adversarial)
- **Why**: Expensive FEA simulations + safety-critical
- **Improvement**: Faster convergence, verified safety

### Sub-Domains

#### General
```python
optimizer = EngineeringOptimizer()
```
- **Config**: PES + Adversarial, 100 simulations
- **Use**: General engineering optimization

#### Structural Optimization
```python
optimizer = EngineeringOptimizer(sub_domain="structural")
```
- **Config**: MO (weight, strength, cost)
- **Use**: Lightweight structures

#### Circuit Design
```python
optimizer = EngineeringOptimizer(sub_domain="circuit")
```
- **Config**: MO (power, area, performance)
- **Use**: Low-power circuits

#### Control Systems
```python
optimizer = EngineeringOptimizer(sub_domain="control")
```
- **Config**: MO (response, stability, robustness)
- **Use**: Optimal control

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `performance` | Performance metric | > 0.80 |
| `safety_margin` | Safety factor | > 2.0 |
| `cost` | Cost (normalized) | < 0.80 |
| `weight` | Weight (normalized) | < 0.70 |
| `reliability` | Reliability | > 0.95 |

### Example

```python
from openevolve.domain import EngineeringOptimizer

# Initialize optimizer
optimizer = EngineeringOptimizer(sub_domain="structural")

# Define constraints
constraints = optimizer.get_engineering_constraints(
    max_weight=1000,
    min_safety_factor=2.5,
    max_cost=50000
)

# Optimize
result = await optimizer.optimize(
    "Design lightweight bridge that supports 50 tons",
    constraints=constraints
)

# Generate safety scenarios
scenarios = optimizer.generate_safety_scenarios("structural")

# Validate design
is_valid, violations = optimizer.validate_design(
    result['domain_metrics'],
    constraints
)
```

## Pharma Domain

### Overview

Specialized for pharmaceutical problems with chemical space exploration.

### Best System

- **System**: OpenEvolve (QD mode)
- **Why**: Need to explore diverse chemical space (many local optima)
- **Improvement**: Diverse molecule discovery

### Sub-Domains

#### General
```python
optimizer = PharmaOptimizer()
```
- **Config**: QD mode, 10,000 archive
- **Use**: General pharma optimization

#### Molecular Optimization
```python
optimizer = PharmaOptimizer(sub_domain="molecular")
```
- **Config**: MO + QD, 25 resolution
- **Use**: Multi-property optimization

#### Drug Design
```python
optimizer = PharmaOptimizer(sub_domain="drug_design")
```
- **Config**: Large archive (15,000)
- **Use**: Target binding optimization

#### Clinical Trial Design
```python
optimizer = PharmaOptimizer(sub_domain="clinical_trial")
```
- **Config**: Standard mode, 80 iterations
- **Use**: Patient stratification

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `binding_affinity` | Target binding | > 0.80 |
| `solubility` | Solubility | > 0.50 |
| `toxicity` | Toxicity (lower is better) | < 0.30 |
| `synthetic_accessibility` | Ease of synthesis | > 0.70 |
| `drug_likeness` | Lipinski's Rule of 5 | > 0.80 |

### Example

```python
from openevolve.domain import PharmaOptimizer

# Initialize optimizer
optimizer = PharmaOptimizer(sub_domain="molecular")

# Define constraints
constraints = optimizer.get_pharma_constraints(
    max_toxicity=0.3,
    min_solubility=0.5,
    min_binding_affinity=0.7,
    max_molecular_weight=500
)

# Optimize
result = await optimizer.optimize(
    "Optimize molecule for high binding affinity and low toxicity",
    constraints=constraints
)

# Calculate drug-likeness
score = optimizer.calculate_drug_likeness(result['molecule_properties'])

# Validate molecule
is_valid, violations = optimizer.validate_molecule(
    result['domain_metrics'],
    constraints
)

# Get optimization suggestions
suggestions = optimizer.suggest_lead_optimization(
    result['molecule_properties'],
    {"binding_affinity": 0.9, "solubility": 0.8}
)
```

## Web Design Domain

### Overview

Specialized for web design with fast evaluations.

### Best System

- **System**: OpenEvolve (Standard mode)
- **Why**: Fast evaluations (seconds), well-understood domain
- **Improvement**: Rapid iteration

### Sub-Domains

#### General
```python
optimizer = WebDesignOptimizer()
```
- **Config**: Standard mode, 100 iterations
- **Use**: General web optimization

#### Landing Page Optimization
```python
optimizer = WebDesignOptimizer(sub_domain="landing_page")
```
- **Config**: 150 iterations, high temperature
- **Use**: Conversion rate optimization

#### UX Optimization
```python
optimizer = WebDesignOptimizer(sub_domain="ux")
```
- **Config**: MO (engagement, satisfaction, accessibility)
- **Use**: User experience optimization

#### A/B Testing
```python
optimizer = WebDesignOptimizer(sub_domain="ab_testing")
```
- **Config**: High temperature, 50 iterations
- **Use**: Variant generation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `conversion_rate` | Conversion rate | > 0.05 |
| `bounce_rate` | Bounce rate (lower is better) | < 0.40 |
| `time_on_page` | Engagement time | > 0.60 |
| `user_satisfaction` | User satisfaction | > 0.75 |

### Example

```python
from openevolve.domain import WebDesignOptimizer

# Initialize optimizer
optimizer = WebDesignOptimizer(sub_domain="landing_page")

# Define constraints
constraints = optimizer.get_web_constraints(
    max_load_time=3.0,
    mobile_first=True,
    min_accessibility=0.8,
    min_seo=0.7
)

# Optimize
result = await optimizer.optimize(
    "Optimize landing page for maximum conversion rate",
    constraints=constraints
)

# Validate design
is_valid, violations = optimizer.validate_design(
    result['domain_metrics'],
    constraints
)

# Get improvement suggestions
suggestions = optimizer.suggest_improvements(
    result['domain_metrics'],
    {"conversion_rate": 0.08, "bounce_rate": 0.30}
)

# Generate A/B test variants
variants = optimizer.generate_ab_test_variants(
    result['best_solution'],
    num_variants=10
)
```

## Multi-Domain Optimization

Compare results across multiple domains:

```python
from openevolve.domain import optimize_multi_domain

results = await optimize_multi_domain(
    "Optimize portfolio for maximum return with minimum risk",
    domains=['finance', 'trading']
)

# Compare results
print(f"Finance: {results['finance']['domain_metrics']['sharpe_ratio']}")
print(f"Trading: {results['trading']['domain_metrics']['sharpe_ratio']}")
```

## Domain Auto-Detection

Let the system auto-detect the domain:

```python
from openevolve.domain import detect_domain, optimize_by_domain

# Auto-detect from problem description
problems = [
    "Optimize portfolio allocation",
    "Develop trading strategy",
    "Design experiments",
    "Optimize bridge design",
    "Discover new drug",
    "Improve landing page"
]

for problem in problems:
    domain = detect_domain(problem)
    print(f"{problem[:40]:40} -> {domain}")

    # Optimize with detected domain
    result = await optimize_by_domain(problem)
```

## Best Practices

### 1. Choose the Right Domain

Use the domain that best matches your problem:

- **Finance**: Portfolio optimization, risk analysis, asset allocation
- **Trading**: Strategy development, signal optimization, parameter tuning
- **Science**: Experimental design, data analysis, hypothesis testing
- **Engineering**: Structural optimization, circuit design, control systems
- **Pharma**: Molecular optimization, drug design, clinical trial design
- **Web Design**: Landing page optimization, UX optimization, A/B testing

### 2. Select Appropriate Sub-Domain

Each domain has 2-4 sub-domains with specialized configurations:

```python
# Finance sub-domains
FinanceOptimizer(sub_domain="portfolio")    # Multi-objective
FinanceOptimizer(sub_domain="risk")         # Conservative
FinanceOptimizer(sub_domain="asset_allocation")  # Constraints

# Trading sub-domains
TradingOptimizer(sub_domain="strategy")     # Creative
TradingOptimizer(sub_domain="signal")       # Focused
TradingOptimizer(sub_domain="parameter")    # Fine-tuning
```

### 3. Define Constraints Clearly

Each optimizer provides domain-specific constraint helpers:

```python
# Finance
constraints = optimizer.get_portfolio_constraints(
    max_assets=30,
    min_weight=0.01
)

# Trading
constraints = optimizer.get_trading_constraints(
    max_drawdown=0.2,
    min_win_rate=0.5
)

# Science
constraints = optimizer.get_experiment_constraints(
    max_experiments=20,
    budget=50000
)
```

### 4. Validate Results

Always validate results against constraints:

```python
is_valid, violations = optimizer.validate_portfolio(
    result['best_solution'],
    constraints
)

if not is_valid:
    print("Violations:", violations)
```

### 5. Use Domain Metrics

Each domain provides specialized metrics:

```python
# Finance metrics
print(f"Sharpe: {metrics['sharpe_ratio']}")
print(f"Max DD: {metrics['max_drawdown']}")

# Trading metrics
print(f"Win Rate: {metrics['win_rate']}")
print(f"Profit Factor: {metrics['profit_factor']}")

# Pharma metrics
print(f"Binding: {metrics['binding_affinity']}")
print(f"Toxicity: {metrics['toxicity']}")
```

## Configuration Reference

### Finance Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `pes` | - | PES mode |
| `max_iterations` | 50 | 20-100 | Max evaluations |
| `plan_temperature` | 0.7 | 0.0-1.0 | Planning creativity |
| `timeout` | 300 | 60-600 | Backtest timeout (s) |

### Trading Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `adversarial` | - | Adversarial mode |
| `adversarial_rounds` | 20 | 10-50 | Adversarial rounds |
| `temperature` | 0.8 | 0.0-1.0 | Creativity |
| `population_size` | 30 | 10-100 | Population size |

### Science Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `qd` | - | QD mode |
| `max_iterations` | 20 | 10-50 | Max experiments |
| `grid_resolution` | 15 | 5-30 | QD resolution |
| `archive_size` | 500 | 100-5000 | Archive size |

### Engineering Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `pes` | - | PES + Adversarial |
| `max_iterations` | 100 | 50-200 | Max simulations |
| `timeout` | 600 | 120-1200 | Simulation timeout (s) |
| `robustness_threshold` | 0.9 | 0.8-1.0 | Safety threshold |

### Pharma Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `qd` | - | QD mode |
| `max_iterations` | 150 | 50-300 | Max evaluations |
| `grid_resolution` | 20 | 10-30 | QD resolution |
| `archive_size` | 10000 | 1000-20000 | Archive size |

### Web Design Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `evolution_mode` | `standard` | - | Standard mode |
| `max_iterations` | 100 | 50-200 | Max iterations |
| `timeout` | 30 | 10-120 | Evaluation timeout (s) |
| `population_size` | 50 | 20-100 | Population size |

## Advanced Usage

### Custom Domain Configuration

```python
from openevolve.domain import FinanceOptimizer
from openevolve.unified.config import UnifiedEvolutionConfig

# Get base config
optimizer = FinanceOptimizer(sub_domain="portfolio")
base_config = optimizer.get_default_config()

# Customize
base_config.max_iterations = 75
base_config.llm.plan_temperature = 0.8

# Use custom config
optimizer.config = base_config
result = await optimizer.optimize("Maximize return")
```

### Domain-Specific Evaluation

```python
from openevolve.domain import TradingOptimizer

optimizer = TradingOptimizer()

# Evaluate custom strategy
strategy_code = """
def entry_signal(data):
    return data['RSI'] < 30

def exit_signal(data):
    return data['RSI'] > 70
"""

metrics = optimizer.evaluate_solution(
    strategy_code,
    "Develop mean-reversion strategy"
)

print(metrics['sharpe_ratio'])
```

### Batch Optimization

```python
from openevolve.domain import get_optimizer

domains = ['finance', 'trading', 'science']
problems = [
    "Optimize portfolio",
    "Develop strategy",
    "Design experiment"
]

results = {}
for domain, problem in zip(domains, problems):
    optimizer = get_optimizer(domain)
    result = await optimizer.optimize(problem)
    results[domain] = result
```

## Troubleshooting

### Problem: Domain Detection Incorrect

**Solution**: Specify domain explicitly

```python
# Instead of auto-detection
result = await optimize_by_domain(problem)

# Specify domain
result = await optimize_by_domain(problem, domain="finance")
```

### Problem: Too Many Evaluations

**Solution**: Reduce `max_iterations` in config

```python
optimizer = FinanceOptimizer()
optimizer.config.max_iterations = 30
```

### Problem: Solutions Not Meeting Constraints

**Solution**: Tighten constraints or use sub-domain

```python
# Use stricter validation
constraints = optimizer.get_portfolio_constraints(
    max_assets=20,  # More strict
    min_weight=0.05
)

# Or use risk-focused sub-domain
optimizer = FinanceOptimizer(sub_domain="risk")
```

### Problem: Slow Convergence

**Solution**: Use appropriate mode

```python
# For expensive evaluations (finance, science)
# PES converges faster (60% fewer evaluations)
FinanceOptimizer()  # Uses PES

# For diverse solutions (pharma)
# QD explores entire space
PharmaOptimizer()  # Uses QD

# For robustness (trading, engineering)
# Adversarial tests edge cases
TradingOptimizer()  # Uses Adversarial
```

## Performance Tips

1. **Use Right Mode**: Match mode to problem characteristics
   - Expensive evals → PES
   - Need diversity → QD
   - Multiple objectives → MO
   - Safety-critical → Adversarial

2. **Leverage Sub-Domains**: Pre-configured for specific use cases

3. **Set Appropriate Iterations**:
   - Very expensive (science): 15-30
   - Expensive (finance, engineering): 50-100
   - Moderate (trading, pharma): 100-200
   - Fast (web): 100-200

4. **Use Constraints**: Guide search and validate results

5. **Parallel Evaluation**: Enable when possible
   ```python
   config.evaluator.parallel_evaluations = 10
   ```

## API Reference

### DomainOptimizer Base Class

```python
class DomainOptimizer:
    domain_name: str
    sub_domain: str

    def get_default_config(self) -> UnifiedEvolutionConfig
    def get_recommended_system(self) -> str
    def get_recommended_mode(self) -> str
    def get_domain_metrics(self) -> List[str]
    def evaluate_solution(self, solution: str, problem: str) -> Dict[str, float]
    async def optimize(self, problem: str, constraints: Dict) -> Dict[str, Any]
```

### Factory Functions

```python
def detect_domain(problem_description: str) -> str
def get_optimizer(domain: str, sub_domain: str = "general") -> DomainOptimizer
async def optimize_by_domain(problem: str, domain: Optional[str] = None) -> Dict
async def optimize_multi_domain(problem: str, domains: List[str]) -> Dict
```

## See Also

- [Unified Configuration](UNIFIED_CONFIG.md)
- [Strategy Selector](STRATEGY_SELECTOR.md)
- [Integration Roadmap](COMPREHENSIVE_INTEGRATION_ROADMAP.md)
