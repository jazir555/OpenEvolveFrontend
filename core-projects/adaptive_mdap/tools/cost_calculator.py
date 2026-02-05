"""
Cost Calculator for Adaptive MDAP.

Calculates API costs for different strategies and providers.
Based on the MAKER paper's cost analysis (Eq. 18):
E[cost of solving full task; m=1] = Θ(p⁻¹cs ln s)

Where:
- p = per-step success rate
- c = cost per sample
- s = number of steps
- k = voting threshold (grows as ln s)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from adaptive_mdap.core.types import SolveStrategy
from adaptive_mdap.utils.logger import get_logger

logger = get_logger("tools.cost_calculator")


class Provider(Enum):
    """API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class APIPricing:
    """Pricing for an API model."""
    provider: Provider
    model: str
    input_price_per_1k: float  # USD per 1000 input tokens
    output_price_per_1k: float  # USD per 1000 output tokens
    
    @classmethod
    def gpt_4o_mini(cls) -> "APIPricing":
        """OpenAI GPT-4o-mini pricing (most cost-effective)."""
        return cls(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            input_price_per_1k=0.00015,
            output_price_per_1k=0.0006,
        )
    
    @classmethod
    def gpt_4o(cls) -> "APIPricing":
        """OpenAI GPT-4o pricing."""
        return cls(
            provider=Provider.OPENAI,
            model="gpt-4o",
            input_price_per_1k=0.0025,
            output_price_per_1k=0.01,
        )
    
    @classmethod
    def gpt_4(cls) -> "APIPricing":
        """OpenAI GPT-4 pricing."""
        return cls(
            provider=Provider.OPENAI,
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
        )
    
    @classmethod
    def claude_3_5_sonnet(cls) -> "APIPricing":
        """Anthropic Claude 3.5 Sonnet pricing."""
        return cls(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-sonnet",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
        )
    
    @classmethod
    def claude_3_5_haiku(cls) -> "APIPricing":
        """Anthropic Claude 3.5 Haiku pricing."""
        return cls(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-haiku",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
        )
    
    @classmethod
    def claude_3_opus(cls) -> "APIPricing":
        """Anthropic Claude 3 Opus pricing."""
        return cls(
            provider=Provider.ANTHROPIC,
            model="claude-3-opus",
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
        )
    
    @classmethod
    def gemini_1_5_pro(cls) -> "APIPricing":
        """Google Gemini 1.5 Pro pricing."""
        return cls(
            provider=Provider.GOOGLE,
            model="gemini-1.5-pro",
            input_price_per_1k=0.00125,
            output_price_per_1k=0.005,
        )
    
    @classmethod
    def gemini_1_5_flash(cls) -> "APIPricing":
        """Google Gemini 1.5 Flash pricing."""
        return cls(
            provider=Provider.GOOGLE,
            model="gemini-1.5-flash",
            input_price_per_1k=0.000075,
            output_price_per_1k=0.0003,
        )


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    input_tokens: int
    output_tokens: int
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class WorkloadDistribution:
    """Distribution of problems by complexity."""
    easy_percentage: float  # 0.0 to 1.0
    medium_percentage: float
    hard_percentage: float
    
    def __post_init__(self):
        total = self.easy_percentage + self.medium_percentage + self.hard_percentage
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Percentages must sum to 1.0, got {total}")
    
    @classmethod
    def default(cls) -> "WorkloadDistribution":
        """Default workload: 30% easy, 40% medium, 30% hard."""
        return cls(
            easy_percentage=0.3,
            medium_percentage=0.4,
            hard_percentage=0.3,
        )
    
    @classmethod
    def compute_heavy(cls) -> "WorkloadDistribution":
        """Compute-heavy workload: 10% easy, 30% medium, 60% hard."""
        return cls(
            easy_percentage=0.1,
            medium_percentage=0.3,
            hard_percentage=0.6,
        )
    
    @classmethod
    def cost_optimized(cls) -> "WorkloadDistribution":
        """Cost-optimized workload: 60% easy, 30% medium, 10% hard."""
        return cls(
            easy_percentage=0.6,
            medium_percentage=0.3,
            hard_percentage=0.1,
        )


@dataclass
class StrategyCost:
    """Cost breakdown for a strategy."""
    strategy: SolveStrategy
    n_calls: int
    cost_per_call: float
    total_cost: float


class CostCalculator:
    """
    Calculate API costs for Adaptive MDAP.
    
    Based on the MAKER paper's cost model:
    - Cost scales log-linearly with number of steps when m=1 (MAD)
    - Different strategies have different expected costs
    - Adaptive allocation can achieve 30-50% savings
    """
    
    # Expected API calls per strategy (based on k values)
    STRATEGY_CALLS = {
        SolveStrategy.DIRECT: 1.0,
        SolveStrategy.MDAP_LIGHT: 3.5,  # 3 agents + voting overhead
        SolveStrategy.MAKER_FULL: 8.0,  # 5 agents + k=2 voting overhead
    }
    
    def __init__(
        self,
        pricing: Optional[APIPricing] = None,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    ):
        """
        Initialize cost calculator.
        
        Args:
            pricing: API pricing model
            avg_input_tokens: Average input tokens per call
            avg_output_tokens: Average output tokens per call
        """
        self.pricing = pricing or APIPricing.gpt_4o_mini()
        self.avg_input_tokens = avg_input_tokens
        self.avg_output_tokens = avg_output_tokens
    
    def calculate_single_call_cost(self, token_usage: Optional[TokenUsage] = None) -> float:
        """
        Calculate cost for a single API call.
        
        Args:
            token_usage: Token usage (uses defaults if None)
            
        Returns:
            Cost in USD
        """
        if token_usage is None:
            token_usage = TokenUsage(
                input_tokens=self.avg_input_tokens,
                output_tokens=self.avg_output_tokens,
            )
        
        input_cost = (token_usage.input_tokens / 1000) * self.pricing.input_price_per_1k
        output_cost = (token_usage.output_tokens / 1000) * self.pricing.output_price_per_1k
        
        return input_cost + output_cost
    
    def calculate_strategy_cost(
        self,
        strategy: SolveStrategy,
        num_problems: int = 1,
    ) -> StrategyCost:
        """
        Calculate cost for a strategy.
        
        Args:
            strategy: Solving strategy
            num_problems: Number of problems
            
        Returns:
            StrategyCost with breakdown
        """
        cost_per_call = self.calculate_single_call_cost()
        n_calls = self.STRATEGY_CALLS[strategy]
        total_cost = cost_per_call * n_calls * num_problems
        
        return StrategyCost(
            strategy=strategy,
            n_calls=int(n_calls * num_problems),
            cost_per_call=cost_per_call,
            total_cost=total_cost,
        )
    
    def calculate_baseline_cost(
        self,
        num_problems: int,
        baseline_strategy: SolveStrategy = SolveStrategy.MAKER_FULL,
    ) -> float:
        """
        Calculate baseline cost (all problems with same strategy).
        
        Args:
            num_problems: Number of problems
            baseline_strategy: Strategy to use for all problems
            
        Returns:
            Total cost in USD
        """
        strategy_cost = self.calculate_strategy_cost(baseline_strategy, num_problems)
        return strategy_cost.total_cost
    
    def calculate_adaptive_cost(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]:
        """
        Calculate cost with adaptive allocation.
        
        Args:
            num_problems: Number of problems
            workload: Workload distribution (uses default if None)
            
        Returns:
            Cost breakdown with savings
        """
        if workload is None:
            workload = WorkloadDistribution.default()
        
        # Calculate costs by strategy
        easy_problems = int(num_problems * workload.easy_percentage)
        medium_problems = int(num_problems * workload.medium_percentage)
        hard_problems = num_problems - easy_problems - medium_problems
        
        direct_cost = self.calculate_strategy_cost(SolveStrategy.DIRECT, easy_problems)
        mdap_cost = self.calculate_strategy_cost(SolveStrategy.MDAP_LIGHT, medium_problems)
        maker_cost = self.calculate_strategy_cost(SolveStrategy.MAKER_FULL, hard_problems)
        
        adaptive_total = direct_cost.total_cost + mdap_cost.total_cost + maker_cost.total_cost
        
        # Calculate baseline (all MAKER_FULL)
        baseline = self.calculate_baseline_cost(num_problems, SolveStrategy.MAKER_FULL)
        
        # Calculate savings
        savings = baseline - adaptive_total
        savings_percent = (savings / baseline) * 100 if baseline > 0 else 0
        
        return {
            "baseline_cost": baseline,
            "adaptive_cost": adaptive_total,
            "savings": savings,
            "savings_percent": savings_percent,
            "breakdown": {
                "direct": {
                    "problems": easy_problems,
                    "cost": direct_cost.total_cost,
                },
                "mdap_light": {
                    "problems": medium_problems,
                    "cost": mdap_cost.total_cost,
                },
                "maker_full": {
                    "problems": hard_problems,
                    "cost": maker_cost.total_cost,
                },
            },
            "pricing_model": self.pricing.model,
        }
    
    def calculate_savings(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, float]:
        """
        Calculate savings from adaptive allocation.
        
        Args:
            num_problems: Number of problems
            workload: Workload distribution
            
        Returns:
            Savings information
        """
        result = self.calculate_adaptive_cost(num_problems, workload)
        
        return {
            "absolute_savings": result["savings"],
            "percentage_savings": result["savings_percent"],
            "baseline_cost": result["baseline_cost"],
            "adaptive_cost": result["adaptive_cost"],
        }
    
    def generate_report(
        self,
        num_problems: int,
        num_days: int = 30,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cost report.
        
        Args:
            num_problems: Number of problems per day
            num_days: Number of days
            workload: Workload distribution
            
        Returns:
            Cost report
        """
        daily = self.calculate_adaptive_cost(num_problems, workload)
        
        # Scale to time period
        total_problems = num_problems * num_days
        total = self.calculate_adaptive_cost(total_problems, workload)
        
        report = {
            "summary": {
                "pricing_model": self.pricing.model,
                "provider": self.pricing.provider.value,
                "problems_per_day": num_problems,
                "days": num_days,
                "total_problems": total_problems,
            },
            "daily_costs": {
                "baseline": daily["baseline_cost"],
                "adaptive": daily["adaptive_cost"],
                "savings": daily["savings"],
                "savings_percent": daily["savings_percent"],
            },
            "total_costs": {
                "baseline": total["baseline_cost"],
                "adaptive": total["adaptive_cost"],
                "savings": total["savings"],
                "savings_percent": total["savings_percent"],
            },
            "breakdown": daily["breakdown"],
            "assumptions": {
                "avg_input_tokens": self.avg_input_tokens,
                "avg_output_tokens": self.avg_output_tokens,
                "workload": {
                    "easy": workload.easy_percentage if workload else 0.3,
                    "medium": workload.medium_percentage if workload else 0.4,
                    "hard": workload.hard_percentage if workload else 0.3,
                } if workload else "default",
            },
        }
        
        return report
    
    def compare_models(
        self,
        num_problems: int,
        workload: Optional[WorkloadDistribution] = None,
    ) -> Dict[str, Any]:
        """
        Compare costs across different models.
        
        Args:
            num_problems: Number of problems
            workload: Workload distribution
            
        Returns:
            Comparison across models
        """
        models = [
            APIPricing.gpt_4o_mini(),
            APIPricing.gpt_4o(),
            APIPricing.claude_3_5_haiku(),
            APIPricing.claude_3_5_sonnet(),
            APIPricing.gemini_1_5_flash(),
        ]
        
        comparisons = []
        for model in models:
            calc = CostCalculator(
                pricing=model,
                avg_input_tokens=self.avg_input_tokens,
                avg_output_tokens=self.avg_output_tokens,
            )
            result = calc.calculate_adaptive_cost(num_problems, workload)
            comparisons.append({
                "model": model.model,
                "provider": model.provider.value,
                "adaptive_cost": result["adaptive_cost"],
                "savings_vs_maker_full": result["savings_percent"],
            })
        
        # Sort by cost
        comparisons.sort(key=lambda x: x["adaptive_cost"])
        
        return {
            "num_problems": num_problems,
            "comparisons": comparisons,
            "cheapest": comparisons[0] if comparisons else None,
        }


def demo_cost_calculator():
    """Demonstrate cost calculator."""
    print("=" * 60)
    print("Adaptive MDAP Cost Calculator Demo")
    print("=" * 60)
    
    # Create calculator with GPT-4o-mini pricing
    calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
    
    # Scenario 1: Daily workload
    print("\n--- Daily Workload (1000 problems) ---")
    daily = calculator.calculate_adaptive_cost(1000)
    print(f"Baseline cost (all MAKER_FULL): ${daily['baseline_cost']:.2f}")
    print(f"Adaptive cost: ${daily['adaptive_cost']:.2f}")
    print(f"Daily savings: ${daily['savings']:.2f} ({daily['savings_percent']:.1f}%)")
    
    # Scenario 2: Monthly projection
    print("\n--- Monthly Projection (30 days, 1000 problems/day) ---")
    monthly = calculator.generate_report(1000, num_days=30)
    print(f"Total baseline: ${monthly['total_costs']['baseline']:.2f}")
    print(f"Total adaptive: ${monthly['total_costs']['adaptive']:.2f}")
    print(f"Total savings: ${monthly['total_costs']['savings']:.2f}")
    
    # Scenario 3: Model comparison
    print("\n--- Model Comparison (1000 problems) ---")
    comparison = calculator.compare_models(1000)
    print(f"Cheapest option: {comparison['cheapest']['model']}")
    print(f"Cost: ${comparison['cheapest']['adaptive_cost']:.2f}")
    print("\nAll models (sorted by cost):")
    for c in comparison["comparisons"]:
        print(f"  {c['model']}: ${c['adaptive_cost']:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_cost_calculator()
