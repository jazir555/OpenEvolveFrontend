# Adaptive-MAKER Cloud API Deployment Guide

## Table of Contents
1. [Cloud API Overview](#cloud-api-overview)
2. [Architecture for Cloud APIs](#architecture-for-cloud-apis)
3. [Cost Calculator Tool](#cost-calculator-tool)
4. [Configuration Examples](#configuration-examples)
5. [Optimization Strategies](#optimization-strategies)
6. [Monitoring & Cost Tracking](#monitoring--cost-tracking)
7. [Best Practices](#best-practices)

---

## Cloud API Overview

### Key Difference from Local MoE Models

**SBM-Efficient (Local MoE):**
- Optimizes FLOPs at model layer
- Access to internal routing
- 24-52% compute reduction
- Requires local model deployment

**Adaptive-MAKER (Cloud APIs):**
- Optimizes API call count at orchestration layer
- No model internals needed
- 30-50% cost reduction through fewer API calls
- Works with any cloud API provider

### Supported Cloud APIs

| Provider | Models | Compatible | Notes |
|----------|--------|------------|-------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-4 | ✅ Full | Recommended |
| Anthropic | Claude 3.5 Sonnet, Opus, Haiku | ✅ Full | Good for complex tasks |
| Google | Gemini 1.5 Pro, Flash | ✅ Full | Cost-effective |
| Azure OpenAI | GPT models | ✅ Full | Enterprise support |
| AWS Bedrock | Claude, Titan, Llama | ✅ Full | Multi-model |
| Cohere | Command R/R+ | ✅ Full | Fast inference |
| OpenAI-compatible | Any | ✅ Full | Flexible integration |

---

## Architecture for Cloud APIs

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenEvolve Application                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Adaptive-MAKER Orchestration                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Task Complexity Classifier (Local)              │  │
│  │     - Runs locally, no API calls                    │  │
│  │     - Uses sentence transformers (lightweight)      │  │
│  │     - Output: complexity score [0, 1]               │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Resource Allocator (Local)                      │  │
│  │     - Decides strategy based on complexity          │  │
│  │     - Output: DIRECT / MDAP_LIGHT / MAKER_FULL      │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. Strategy Execution (Orchestrator)                │  │
│  │     - Routes to appropriate API calling pattern     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cloud API Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   DIRECT    │  │ MDAP_LIGHT  │  │ MAKER_FULL  │          │
│  │  1 API call │  │ 3 API calls │  │ 10 API calls│          │
│  │  Fast/cheap │  │ Balanced    │  │ Thorough    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                 │                 │                │
│         └─────────────────┴─────────────────┘                │
│                           │                                  │
│                           ▼                                  │
│              ┌─────────────────────────┐                     │
│              │  API Client (OpenAI/    │                     │
│              │  Anthropic/Google/etc) │                     │
│              └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

**All decision-making happens locally (no API calls):**
- Complexity classification: Runs locally with embeddings
- Resource allocation: Simple threshold comparison (microseconds)
- Only the final execution calls cloud APIs

---

## Cost Calculator Tool

### Tool Overview

The cost calculator helps you:
1. Estimate current costs (baseline)
2. Project adaptive-MAKER savings
3. Compare different configurations
4. Calculate ROI

### Cost Calculator Implementation

Create file: `Frontend/adaptive_mdap/tools/cost_calculator.py`

```python
"""
Cost Calculator for Adaptive-MAKER with Cloud APIs
Estimates savings based on workload characteristics and API pricing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

@dataclass
class APIPricing:
    """Pricing for a specific API model."""
    provider: str
    model: str
    input_price_per_1k: float  # USD per 1K input tokens
    output_price_per_1k: float  # USD per 1K output tokens

    # Example prices (as of 2025-01)
    @classmethod
    def gpt_4o_mini(cls):
        return cls(
            provider="openai",
            model="gpt-4o-mini",
            input_price_per_1k=0.00015,  # $0.15/1M
            output_price_per_1k=0.00060   # $0.60/1M
        )

    @classmethod
    def gpt_4o(cls):
        return cls(
            provider="openai",
            model="gpt-4o",
            input_price_per_1k=0.0025,   # $2.50/1M
            output_price_per_1k=0.0100   # $10.00/1M
        )

    @classmethod
    def claude_3_5_sonnet(cls):
        return cls(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_price_per_1k=0.003,    # $3/1M
            output_price_per_1k=0.015    # $15/1M
        )

    @classmethod
    def claude_haiku(cls):
        return cls(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            input_price_per_1k=0.0008,   # $0.80/1M
            output_price_per_1k=0.0014   # $1.40/1M
        )


@dataclass
class TokenUsage:
    """Average token usage per API call."""
    input_tokens: int
    output_tokens: int


@dataclass
class WorkloadDistribution:
    """Distribution of task complexities."""
    easy_percentage: float      # % of tasks that are easy
    medium_percentage: float    # % of tasks that are medium
    hard_percentage: float      # % of tasks that are hard


@dataclass
class StrategyConfig:
    """Configuration for a solve strategy."""
    name: str
    n_api_calls: int            # Number of API calls
    pricing: APIPricing          # Which model to use


class CostCalculator:
    """Calculate costs for adaptive-MAKER with cloud APIs."""

    def __init__(
        self,
        token_usage: TokenUsage,
        workload_distribution: WorkloadDistribution
    ):
        """
        Initialize calculator.

        Args:
            token_usage: Average tokens per API call
            workload_distribution: Distribution of task complexities
        """
        self.token_usage = token_usage
        self.workload = workload_distribution

    def calculate_single_call_cost(
        self,
        pricing: APIPricing
    ) -> float:
        """Calculate cost of a single API call."""
        input_cost = (self.token_usage.input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (self.token_usage.output_tokens / 1000) * pricing.output_price_per_1k
        return input_cost + output_cost

    def calculate_strategy_cost(
        self,
        strategy: StrategyConfig
    ) -> float:
        """Calculate total cost for a strategy."""
        single_call_cost = self.calculate_single_call_cost(strategy.pricing)
        return single_call_cost * strategy.n_api_calls

    def calculate_baseline_cost(
        self,
        always_strategy: StrategyConfig,
        num_sub_problems: int
    ) -> float:
        """
        Calculate cost if always using one strategy (baseline).

        Args:
            always_strategy: Strategy to always use
            num_sub_problems: Number of sub-problems to solve

        Returns:
            Total cost in USD
        """
        cost_per_problem = self.calculate_strategy_cost(always_strategy)
        return cost_per_problem * num_sub_problems

    def calculate_adaptive_cost(
        self,
        easy_strategy: StrategyConfig,
        medium_strategy: StrategyConfig,
        hard_strategy: StrategyConfig,
        num_sub_problems: int
    ) -> float:
        """
        Calculate cost with adaptive allocation.

        Args:
            easy_strategy: Strategy for easy tasks
            medium_strategy: Strategy for medium tasks
            hard_strategy: Strategy for hard tasks
            num_sub_problems: Total number of sub-problems

        Returns:
            Total cost in USD
        """
        # Calculate cost for each complexity level
        easy_cost = self.calculate_strategy_cost(easy_strategy)
        medium_cost = self.calculate_strategy_cost(medium_strategy)
        hard_cost = self.calculate_strategy_cost(hard_strategy)

        # Weight by distribution
        num_easy = num_sub_problems * (self.workload.easy_percentage / 100)
        num_medium = num_sub_problems * (self.workload.medium_percentage / 100)
        num_hard = num_sub_problems * (self.workload.hard_percentage / 100)

        total_cost = (
            easy_cost * num_easy +
            medium_cost * num_medium +
            hard_cost * num_hard
        )

        return total_cost

    def calculate_savings(
        self,
        baseline_cost: float,
        adaptive_cost: float
    ) -> Dict[str, float]:
        """Calculate savings metrics."""
        absolute_savings = baseline_cost - adaptive_cost
        relative_savings = (absolute_savings / baseline_cost) * 100

        return {
            "baseline_cost": round(baseline_cost, 2),
            "adaptive_cost": round(adaptive_cost, 2),
            "absolute_savings": round(absolute_savings, 2),
            "relative_savings_percent": round(relative_savings, 1)
        }

    def generate_report(
        self,
        baseline_strategy: StrategyConfig,
        adaptive_strategies: Tuple[StrategyConfig, StrategyConfig, StrategyConfig],
        num_sub_problems: int,
        num_days: int = 30
    ) -> Dict:
        """
        Generate comprehensive cost comparison report.

        Args:
            baseline_strategy: Strategy used in baseline (always MAKER_FULL)
            adaptive_strategies: (easy, medium, hard) strategies
            num_sub_problems: Sub-problems per day
            num_days: Number of days to project

        Returns:
            Detailed cost comparison report
        """
        total_problems = num_sub_problems * num_days

        baseline_cost = self.calculate_baseline_cost(
            baseline_strategy,
            total_problems
        )

        adaptive_cost = self.calculate_adaptive_cost(
            adaptive_strategies[0],  # easy
            adaptive_strategies[1],  # medium
            adaptive_strategies[2],  # hard
            total_problems
        )

        savings = self.calculate_savings(baseline_cost, adaptive_cost)

        # Per-day breakdown
        daily_baseline = baseline_cost / num_days
        daily_adaptive = adaptive_cost / num_days
        daily_savings = daily_baseline - daily_adaptive

        return {
            "summary": savings,
            "daily": {
                "baseline": round(daily_baseline, 2),
                "adaptive": round(daily_adaptive, 2),
                "savings": round(daily_savings, 2)
            },
            "per_problem": {
                "baseline": round(baseline_cost / total_problems, 4),
                "adaptive": round(adaptive_cost / total_problems, 4)
            },
            "breakdown": {
                "easy": {
                    "percentage": self.workload.easy_percentage,
                    "num_problems": int(total_problems * self.workload.easy_percentage / 100),
                    "strategy": adaptive_strategies[0].name,
                    "cost_per_problem": round(self.calculate_strategy_cost(adaptive_strategies[0]), 4)
                },
                "medium": {
                    "percentage": self.workload.medium_percentage,
                    "num_problems": int(total_problems * self.workload.medium_percentage / 100),
                    "strategy": adaptive_strategies[1].name,
                    "cost_per_problem": round(self.calculate_strategy_cost(adaptive_strategies[1]), 4)
                },
                "hard": {
                    "percentage": self.workload.hard_percentage,
                    "num_problems": int(total_problems * self.workload.hard_percentage / 100),
                    "strategy": adaptive_strategies[2].name,
                    "cost_per_problem": round(self.calculate_strategy_cost(adaptive_strategies[2]), 4)
                }
            },
            "assumptions": {
                "tokens_per_call": {
                    "input": self.token_usage.input_tokens,
                    "output": self.token_usage.output_tokens
                },
                "workload_distribution": {
                    "easy": self.workload.easy_percentage,
                    "medium": self.workload.medium_percentage,
                    "hard": self.workload.hard_percentage
                },
                "period": {
                    "sub_problems_per_day": num_sub_problems,
                    "days": num_days
                }
            }
        }


def demo_cost_calculator():
    """Demonstrate cost calculator with typical workload."""

    # Setup: Typical usage patterns
    token_usage = TokenUsage(
        input_tokens=800,    # Average prompt tokens
        output_tokens=1200   # Average response tokens
    )

    workload = WorkloadDistribution(
        easy_percentage=40,    # 40% easy tasks
        medium_percentage=40,  # 40% medium tasks
        hard_percentage=20     # 20% hard tasks
    )

    calculator = CostCalculator(token_usage, workload)

    # Define strategies with different models
    direct_strategy = StrategyConfig(
        name="DIRECT",
        n_api_calls=1,
        pricing=APIPricing.gpt_4o_mini()  # Cheapest model for easy tasks
    )

    mdap_light_strategy = StrategyConfig(
        name="MDAP_LIGHT",
        n_api_calls=3,
        pricing=APIPricing.gpt_4o()  # Mid-tier for medium
    )

    maker_full_strategy = StrategyConfig(
        name="MAKER_FULL",
        n_api_calls=10,  # 5 agents × 2 votes
        pricing=APIPricing.gpt_4o()  # Premium for hard
    )

    # Generate report for 100 sub-problems/day over 30 days
    report = calculator.generate_report(
        baseline_strategy=maker_full_strategy,
        adaptive_strategies=(direct_strategy, mdap_light_strategy, maker_full_strategy),
        num_sub_problems=100,
        num_days=30
    )

    return report


if __name__ == "__main__":
    import json

    report = demo_cost_calculator()
    print(json.dumps(report, indent=2))
```

### Using the Cost Calculator

```bash
# Run the demo
cd Frontend/adaptive_mdap/tools
python cost_calculator.py

# Or import and use programmatically
from adaptive_mdap.tools.cost_calculator import CostCalculator, TokenUsage, WorkloadDistribution

# Your actual workload data
calculator = CostCalculator(
    token_usage=TokenUsage(input_tokens=500, output_tokens=1000),
    workload_distribution=WorkloadDistribution(
        easy_percentage=35,
        medium_percentage=45,
        hard_percentage=20
    )
)

# Calculate your specific scenario
report = calculator.generate_report(
    baseline_strategy=your_baseline,
    adaptive_strategies=(easy, medium, hard),
    num_sub_problems=500,  # Your daily volume
    num_days=30
)

print(f"Projected monthly savings: ${report['summary']['absolute_savings']}")
```

---

## Configuration Examples

### Example 1: OpenAI with Adaptive Model Selection

```yaml
# config/adaptive_mdap_openai.yaml

adaptive_mdap:
  enabled: true

  classifier:
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: "./cache/adaptive_mdap"
    feature_weights:
      text_length: 0.20
      domain_rarity: 0.30
      depth: 0.20
      historical_error: 0.20
      dependency: 0.10

  allocator:
    thresholds: [0.3, 0.7]

  strategies:
    # Use cheapest model for easy tasks
    direct:
      n_agents: 1
      k_ahead: 0
      max_retries: 1
      model: "gpt-4o-mini"
      temperature: 0.0
      max_tokens: 1000

    # Use mid-tier for medium tasks
    mdap_light:
      n_agents: 3
      k_ahead: 1
      max_retries: 2
      model: "gpt-4o"
      temperature: 0.1
      max_tokens: 2000

    # Use premium for hard tasks
    maker_full:
      n_agents: 5
      k_ahead: 2
      max_retries: 3
      model: "gpt-4o"
      temperature: 0.2
      max_tokens: 4000

  # OpenAI-specific settings
  openai:
    api_key_env: "OPENAI_API_KEY"
    organization: null
    timeout_ms: 30000
    max_retries: 3
    retry_delay_ms: 1000
```

### Example 2: Anthropic Claude

```yaml
# config/adaptive_mdap_anthropic.yaml

adaptive_mdap:
  enabled: true

  strategies:
    direct:
      n_agents: 1
      model: "claude-3-5-haiku-20241022"  # Fastest, cheapest
      max_tokens: 1000
      temperature: 0.0

    mdap_light:
      n_agents: 3
      model: "claude-3-5-sonnet-20241022"  # Balanced
      max_tokens: 2000
      temperature: 0.1

    maker_full:
      n_agents: 5
      model: "claude-3-5-sonnet-20241022"  # Best quality
      max_tokens: 4000
      temperature: 0.2

  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
    timeout_ms: 60000  # Claude can be slower
    max_retries: 3
    version: "2023-06-01"
```

### Example 3: Multi-Provider Strategy

```yaml
# config/adaptive_mdap_multi_provider.yaml

adaptive_mdap:
  enabled: true

  strategies:
    # Use cheapest from OpenAI for easy
    direct:
      n_agents: 1
      provider: "openai"
      model: "gpt-4o-mini"

    # Use Anthropic for medium (better at reasoning)
    mdap_light:
      n_agents: 3
      provider: "anthropic"
      model: "claude-3-5-haiku-20241022"

    # Use OpenAI for hard (faster)
    maker_full:
      n_agents: 5
      provider: "openai"
      model: "gpt-4o"

  # Provider-specific configs
  providers:
    openai:
      api_key_env: "OPENAI_API_KEY"
      base_url: "https://api.openai.com/v1"

    anthropic:
      api_key_env: "ANTHROPIC_API_KEY"
      base_url: "https://api.anthropic.com"
```

### Example 4: Conservative Configuration (Higher Quality)

```yaml
# config/adaptive_mdap_conservative.yaml

# For quality-critical applications
# Higher thresholds = more use of MAKER_FULL

adaptive_mdap:
  allocator:
    thresholds: [0.5, 0.9]  # More conservative

  strategies:
    direct:
      n_agents: 1
      model: "gpt-4o"  # Use better model even for easy

    mdap_light:
      n_agents: 5  # More agents for medium
      model: "gpt-4o"
      k_ahead: 2  # More voting

    maker_full:
      n_agents: 7  # More agents for hard
      model: "gpt-4o"
      k_ahead: 3  # Stricter consensus
```

### Example 5: Aggressive Configuration (Maximum Savings)

```yaml
# config/adaptive_mdap_aggressive.yaml

# For cost-sensitive applications
# Lower thresholds = more use of DIRECT

adaptive_mdap:
  allocator:
    thresholds: [0.2, 0.5]  # More aggressive

  strategies:
    direct:
      n_agents: 1
      model: "gpt-4o-mini"  # Cheapest

    mdap_light:
      n_agents: 2  # Fewer agents
      model: "gpt-4o-mini"
      k_ahead: 1

    maker_full:
      n_agents: 3  # Fewer agents
      model: "gpt-4o"
      k_ahead: 1
```

---

## Optimization Strategies

### Strategy 1: Adaptive Model Selection

**Idea:** Use cheaper models for easier tasks, premium models for hard tasks.

```python
class AdaptiveModelSelector:
    """Select both strategy AND model based on complexity."""

    def get_strategy(self, complexity: float) -> SolveConfig:
        if complexity < 0.3:
            # Very easy → cheapest model
            return SolveConfig(
                strategy=SolveStrategy.DIRECT,
                n_agents=1,
                model="gpt-4o-mini",     # $0.15/1M input
                api_calls=1
            )
        elif complexity < 0.5:
            # Easy → budget model
            return SolveConfig(
                strategy=SolveStrategy.DIRECT,
                n_agents=1,
                model="gpt-4o",          # $2.50/1M input
                api_calls=1
            )
        elif complexity < 0.7:
            # Medium → mid-tier with light voting
            return SolveConfig(
                strategy=SolveStrategy.MDAP_LIGHT,
                n_agents=3,
                model="gpt-4o",
                api_calls=3,
                k_ahead=1
            )
        else:
            # Hard → premium with full voting
            return SolveConfig(
                strategy=SolveStrategy.MAKER_FULL,
                n_agents=5,
                model="gpt-4o",
                api_calls=10,
                k_ahead=2
            )
```

**Expected Savings:** 60-70% vs always using GPT-4o with MAKER_FULL

### Strategy 2: Request Batching

**Idea:** Batch similar complexity tasks together to optimize API calls.

```python
class BatchExecutor:
    """Batch sub-problems by complexity for efficient execution."""

    def execute_batch(self, sub_problems: List[SubProblem]) -> List[SolutionAttempt]:
        # Group by complexity
        easy_batch = [sp for sp in sub_problems if self.complexity(sp) < 0.3]
        medium_batch = [sp for sp in sub_problems if 0.3 <= self.complexity(sp) < 0.7]
        hard_batch = [sp for sp in sub_problems if self.complexity(sp) >= 0.7]

        # Execute each batch with appropriate strategy
        results = []
        results.extend(self._execute_direct(easy_batch))
        results.extend(self._execute_mdap_light(medium_batch))
        results.extend(self._execute_maker_full(hard_batch))

        return results
```

**Expected Savings:** 10-20% additional savings from batching efficiency

### Strategy 3: Token Optimization

**Idea:** Adjust max_tokens based on task complexity.

```python
class TokenOptimizer:
    """Optimize token limits based on complexity."""

    def get_max_tokens(self, complexity: float) -> int:
        """Lower token limits for easier tasks."""
        if complexity < 0.3:
            return 500   # Short answers sufficient
        elif complexity < 0.7:
            return 1500  # Medium answers
        else:
            return 4000  # Full answers for complex tasks
```

**Expected Savings:** 10-15% from reduced output token costs

### Strategy 4: Provider Arbitrage

**Idea:** Use different providers based on current pricing/performance.

```python
class ProviderSelector:
    """Select provider based on real-time pricing/performance."""

    def get_provider_config(self, complexity: float) -> ProviderConfig:
        """Check current prices and select best provider."""

        if complexity < 0.3:
            # Compare cheapest options
            return self._cheapest_provider()
        elif complexity < 0.7:
            # Compare speed/cost tradeoff
            return self._balanced_provider()
        else:
            # Use best quality regardless of cost
            return self._premium_provider()
```

**Expected Savings:** 5-10% from provider optimization

---

## Monitoring & Cost Tracking

### Real-Time Cost Monitoring

```python
class CostTracker:
    """Track actual API costs in real-time."""

    def __init__(self):
        self.total_cost = 0.0
        self.by_strategy = defaultdict(float)
        self.by_provider = defaultdict(float)
        self.call_counts = defaultdict(int)

    def track_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        strategy: str
    ):
        """Track a single API call and its cost."""
        pricing = self._get_pricing(provider, model)
        cost = self._calculate_cost(pricing, input_tokens, output_tokens)

        self.total_cost += cost
        self.by_strategy[strategy] += cost
        self.by_provider[provider] += cost
        self.call_counts[strategy] += 1

        logger.info(
            f"API call: {provider}/{model} | Strategy: {strategy} | "
            f"Tokens: {input_tokens}+{output_tokens} | Cost: ${cost:.6f}"
        )

    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            "total_cost": self.total_cost,
            "by_strategy": dict(self.by_strategy),
            "by_provider": dict(self.by_provider),
            "call_counts": dict(self.call_counts),
            "avg_cost_per_call": self.total_cost / sum(self.call_counts.values())
        }
```

### Integration with Hephaestus

```python
# Extend Hephaestus tracking for cost metrics

class AdaptiveHephaestusIntegration:
    """Track adaptive decisions and costs in Hephaestus."""

    def track_allocation(
        self,
        sub_problem_id: str,
        complexity: float,
        strategy: str,
        estimated_cost: float
    ):
        """Create ticket for allocation decision."""
        self.manager.create_ticket(
            ticket_type="ADAPTIVE_ALLOCATION",
            properties={
                "sub_problem_id": sub_problem_id,
                "complexity_score": complexity,
                "allocated_strategy": strategy,
                "estimated_cost_usd": estimated_cost,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def track_outcome(
        self,
        sub_problem_id: str,
        actual_cost: float,
        success: bool
    ):
        """Update ticket with actual outcome."""
        self.manager.update_ticket(
            ticket_id=sub_problem_id,
            updates={
                "actual_cost_usd": actual_cost,
                "success": success,
                "cost_vs_estimate": actual_cost / estimated_cost
            }
        )
```

### Cost Dashboard Metrics

Track these metrics in your dashboard:

```python
metrics = {
    # Cost metrics
    "total_api_cost_usd": tracker.total_cost,
    "cost_by_strategy": tracker.by_strategy,
    "cost_per_sub_problem": tracker.total_cost / num_problems,

    # Call metrics
    "total_api_calls": sum(tracker.call_counts.values()),
    "calls_by_strategy": tracker.call_counts,

    # Efficiency metrics
    "avg_tokens_per_call": total_tokens / total_calls,
    "estimated_savings": baseline_cost - actual_cost,
    "savings_percentage": (baseline_cost - actual_cost) / baseline_cost * 100,

    # Quality metrics
    "success_rate_by_strategy": {
        "DIRECT": direct_success_rate,
        "MDAP_LIGHT": light_success_rate,
        "MAKER_FULL": full_success_rate
    }
}
```

---

## Best Practices

### 1. Start Conservative, Then Optimize

```yaml
# Phase 1: Conservative (focus on quality)
allocator:
  thresholds: [0.5, 0.9]  # Most tasks use MAKER_FULL

# Phase 2: Balanced (after validation)
allocator:
  thresholds: [0.3, 0.7]  # Default recommendation

# Phase 3: Optimized (after quality confirmed)
allocator:
  thresholds: [0.2, 0.6]  # Maximize savings
```

### 2. Monitor Quality Metrics Closely

```python
# Set up automated quality monitoring
if adaptive_success_rate < baseline_success_rate * 0.99:
    logger.warning("Quality degraded by >1%")
    # Consider raising thresholds
    allocator.update_thresholds([0.4, 0.8])
```

### 3. Use Feature Flags for Gradual Rollout

```python
# Gradual rollout: 10% → 50% → 100%
adaptive_enabled = feature_flag.get("adaptive_mdap_enabled", default=False)
rollout_percentage = feature_flag.get("adaptive_mdap_rollout", default=0)

if adaptive_enabled and random.random() < (rollout_percentage / 100):
    return solve_adaptive(sub_problem)
else:
    return solve_baseline(sub_problem)
```

### 4. Set Up Cost Alerts

```python
# Alert if costs exceed budget
daily_budget = 100.0  # USD
if tracker.total_cost > daily_budget:
    alert.send(
        f"Daily budget exceeded: ${tracker.total_cost:.2f} > ${daily_budget:.2f}",
        severity="warning"
    )
```

### 5. Regular Threshold Tuning

```python
# Weekly: Analyze performance and tune thresholds
def weekly_threshold_tuning():
    stats = allocator.get_allocation_stats()

    # If DIRECT success rate is high, can be more aggressive
    if stats["DIRECT_success_rate"] > 0.98:
        current_thresholds = allocator.thresholds
        allocator.update_thresholds([
            current_thresholds[0] * 0.9,  # Lower by 10%
            current_thresholds[1] * 0.9
        ])

    # If MAKER_FULL is overused, raise thresholds
    if stats["MAKER_FULL_percentage"] > 0.3:
        allocator.update_thresholds([0.4, 0.8])
```

### 6. Track Token Usage Patterns

```python
# Analyze token usage to optimize further
def analyze_token_patterns():
    patterns = {
        "DIRECT": {
            "avg_input_tokens": 500,
            "avg_output_tokens": 800,
            "suggested_max_tokens": 1000
        },
        "MDAP_LIGHT": {
            "avg_input_tokens": 800,
            "avg_output_tokens": 1500,
            "suggested_max_tokens": 2000
        },
        "MAKER_FULL": {
            "avg_input_tokens": 1200,
            "avg_output_tokens": 2500,
            "suggested_max_tokens": 4000
        }
    }
    return patterns
```

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install sentence-transformers`
- [ ] Set up API keys in environment variables
- [ ] Choose base configuration (conservative/balanced/aggressive)
- [ ] Run cost calculator with your workload data
- [ ] Review projected savings
- [ ] Configure thresholds based on your quality requirements
- [ ] Set up cost tracking
- [ ] Deploy to staging environment
- [ ] Run A/B test (adaptive vs baseline)
- [ ] Monitor quality metrics for 1 week
- [ ] Analyze cost savings
- [ ] Tune thresholds if needed
- [ ] Deploy to production
- [ ] Set up cost alerts
- [ ] Schedule weekly reviews

---

## Troubleshooting

### Issue: Higher Than Expected Costs

**Diagnosis:**
```python
# Check allocation distribution
stats = allocator.get_allocation_stats()
print(stats["strategy_distribution"])

# If MAKER_FULL > 30%, thresholds too low
# If DIRECT < 20%, thresholds too high
```

**Solution:** Adjust thresholds gradually

### Issue: Quality Degradation

**Diagnosis:**
```python
# Check success rates by strategy
for strategy in ["DIRECT", "MDAP_LIGHT", "MAKER_FULL"]:
    success_rate = compute_success_rate(strategy)
    print(f"{strategy}: {success_rate:.2%}")
```

**Solution:** Raise thresholds or increase agents for problematic strategies

### Issue: API Rate Limiting

**Diagnosis:**
```bash
# Check API rate limits
# OpenAI: 10K TPM for gpt-4o-mini, 150 TPM for gpt-4o
```

**Solution:**
- Implement rate limiting at application level
- Add delays between batches
- Use multiple API keys

---

**Document Version:** 1.0
**Last Updated:** 2025-01-17
**Author:** OpenEvolve Integration Team
**Status:** Ready for Cloud API Deployment
