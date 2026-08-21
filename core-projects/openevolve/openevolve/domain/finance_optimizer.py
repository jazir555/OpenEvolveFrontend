"""
Finance Domain Optimizer
Specialized optimizer for financial optimization problems

Problems:
- Portfolio optimization (multi-objective: return vs risk)
- Risk analysis (VaR, CVaR optimization)
- Asset allocation (constraint-heavy)

Best System: LoongFlow (PES mode)
Why: Expensive backtests (each eval = minutes), needs reasoning

Configuration:
- Evaluation cost: "very_expensive"
- Max evaluations: 50 (PES reduces by 60%)
- Early stopping: enabled
- Planning temperature: 0.7
- Timeouts: 300s per backtest

Metrics:
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Volatility

Author: AI Architecture Team
Date: 2026-01-30
"""

import math
from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType, MOConfig, PESConfig, LLMConfig, EvaluatorConfig, DatabaseConfig
from .base import DomainOptimizer
from .heuristics import (
    PERIODS_PER_YEAR,
    clamp,
    portfolio_moments,
    return_statistics,
    synthetic_returns,
)

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based backtest allocation
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class FinanceOptimizer(DomainOptimizer):
    """
    Finance domain optimizer

    Specialized for:
    - Portfolio optimization
    - Risk analysis
    - Asset allocation

    Example:
        >>> optimizer = FinanceOptimizer(sub_domain="portfolio")
        >>> result = await optimizer.optimize(
        ...     "Maximize portfolio return with minimum risk",
        ...     constraints={"max_assets": 50, "min_weight": 0.01}
        ... )
        >>> print(result['domain_metrics']['sharpe_ratio'])
    """

    domain_name = "finance"

    def __init__(self, sub_domain: str = "general", use_adaptive_mdap: bool = True):
        """
        Initialize finance optimizer

        Args:
            sub_domain: One of 'general', 'portfolio', 'risk', 'asset_allocation'
            use_adaptive_mdap: Whether to use Adaptive MDAP for complexity-based allocation
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "portfolio": self._portfolio_config(),
            "risk": self._risk_config(),
            "asset_allocation": self._asset_allocation_config()
        }

        # Set active config
        self.config = self.sub_domain_configs.get(
            sub_domain,
            self._general_config()
        )

        # Initialize Adaptive MDAP if available
        self.use_adaptive_mdap = use_adaptive_mdap and ADAPTIVE_MDAP_AVAILABLE
        if self.use_adaptive_mdap:
            self.complexity_classifier = TaskComplexityClassifier()
            self.resource_allocator = AdaptiveMDAPAllocator(enable_learning=True)
        else:
            self.complexity_classifier = None
            self.resource_allocator = None

    def get_recommended_system(self) -> str:
        """LoongFlow PES is best for expensive backtests"""
        return "loongflow"

    def get_recommended_mode(self) -> str:
        """PES mode for reasoning-guided search"""
        return "pes"

    def get_domain_metrics(self) -> List[str]:
        """Finance-specific metrics"""
        return [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "volatility",
            "annual_return",
            "var_95",
            "cvar_95"
        ]

    def classify_complexity(self, problem: str, constraints: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Classify problem complexity using Adaptive MDAP

        Args:
            problem: Problem description
            constraints: Additional constraints

        Returns:
            Complexity result with overall_score (0-1) or None if Adaptive MDAP unavailable
        """
        if not self.use_adaptive_mdap:
            return None

        subproblem = SubProblem(
            id=f"finance_{hash(problem) % 10000}",
            description=problem,
            domain="finance",
            depth=0,
            dependencies=[],
            metadata={"constraints": constraints or {}, "sub_domain": self.sub_domain}
        )
        return self.complexity_classifier.compute_complexity(subproblem)

    def get_adaptive_config(
        self,
        problem: str,
        base_config: UnifiedEvolutionConfig,
        constraints: Optional[Dict[str, Any]] = None
    ) -> UnifiedEvolutionConfig:
        """
        Get configuration adjusted for problem complexity

        Args:
            problem: Problem description
            base_config: Base configuration to adjust
            constraints: Additional constraints

        Returns:
            Adjusted configuration
        """
        if not self.use_adaptive_mdap:
            return base_config

        # Classify complexity
        complexity = self.classify_complexity(problem, constraints)
        if complexity is None:
            return base_config

        # Get allocation based on complexity
        allocation = self.resource_allocator.allocate_resources(complexity.overall_score)

        # Create adjusted config
        config = base_config.copy() if hasattr(base_config, 'copy') else base_config

        # Adjust based on complexity score
        score = complexity.overall_score

        # Adjust iterations and PES settings based on complexity and strategy
        if allocation.strategy == "DIRECT":
            # Simple problems: fewer iterations, standard mode (no PES)
            config.max_iterations = min(25, config.max_iterations)
            config.evolution_mode = EvolutionMode.STANDARD
            config.pes.enabled = False
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = False
        elif allocation.strategy == "MDAP_LIGHT":
            # Light: moderate iterations, light PES
            config.max_iterations = min(40, config.max_iterations)
            config.pes.plan_iterations = 1
            config.pes.max_rounds = 2
        elif allocation.strategy == "MDAP_MEDIUM":
            # Medium: standard PES config
            pass  # Use base config as-is
        elif allocation.strategy == "MAKER_FULL":
            # Full: more iterations, deeper PES planning, enable MO if available
            config.max_iterations = max(75, config.max_iterations)
            config.pes.plan_iterations = 2
            config.pes.max_rounds = 4
            config.pes.use_memory = True
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True
        elif allocation.strategy == "MAKER_ULTRA":
            # Ultra: max iterations, full PES with memory, enable MO if available
            config.max_iterations = max(100, config.max_iterations)
            config.pes.plan_iterations = 3
            config.pes.max_rounds = 5
            config.pes.use_memory = True
            config.pes.memory_top_k = 10
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True

        # Adjust evaluation timeout based on complexity
        if score > 0.7:
            # High complexity: longer timeouts
            config.evaluator.timeout = min(600, config.evaluator.timeout * 1.5)
        elif score < 0.3:
            # Low complexity: shorter timeouts
            config.evaluator.timeout = max(180, config.evaluator.timeout * 0.8)

        return config

    async def optimize(
        self,
        problem: str,
        constraints: Optional[Dict[str, Any]] = None,
        use_adaptive: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization with Adaptive MDAP complexity-based configuration

        Args:
            problem: Problem description
            constraints: Additional constraints
            use_adaptive: Whether to use adaptive configuration (default: True)
            **kwargs: Additional parameters

        Returns:
            Optimization result with domain-specific metrics and complexity info
        """
        # Get adaptive config if enabled
        if use_adaptive and self.use_adaptive_mdap:
            config = self.get_adaptive_config(problem, self.config, constraints)
            complexity = self.classify_complexity(problem, constraints)
        else:
            config = self.config
            complexity = None

        # Import here to avoid circular dependency
        from ..unified.api import evolve

        # Run evolution with (possibly adaptive) config
        result = await evolve(
            problem_statement=problem,
            config=config,
            constraints=constraints,
            **kwargs
        )

        # Add domain-specific evaluation
        if result.get('best_solution'):
            domain_metrics = self.evaluate_solution(
                result['best_solution'],
                problem,
                constraints
            )
            result['domain_metrics'] = domain_metrics

        # Add metadata
        result['domain'] = self.domain_name
        result['sub_domain'] = self.sub_domain
        result['recommended_system'] = self.get_recommended_system()
        result['recommended_mode'] = self.get_recommended_mode()

        # Add complexity info if available
        if complexity:
            result['complexity'] = {
                'overall_score': complexity.overall_score,
                'features': complexity.features if hasattr(complexity, 'features') else {},
                'adaptive_config_applied': use_adaptive and self.use_adaptive_mdap
            }

        return result

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default finance configuration"""
        return self._general_config()

    # ========================================================================
    # SUB-DOMAIN CONFIGURATIONS
    # ========================================================================

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General finance configuration

        Uses PES mode with reasoning to reduce expensive backtests
        """
        
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.FINANCE,

            # Evolution mode
            evolution_mode=EvolutionMode.PES,

            # PES configuration (LoongFlow)
            pes=PESConfig(
                enabled=True,
                enable_planning=True,
                max_plans=1,
                plan_iterations=1,
                max_rounds=3,
                parallel_candidates=1,
                enable_summary=True,
                use_memory=True,
                memory_top_k=5
            ),

            # LLM configuration
            llm=LLMConfig(
                temperature=0.7,
                plan_temperature=0.7,
                summary_temperature=0.7,
                timeout=300,
                retries=2
            ),

            # Evaluation (expensive backtests)
            max_iterations=50,  # Limited budget
            evaluator=EvaluatorConfig(
                timeout=300,  # 5 minutes per backtest
                max_retries=2,
                early_stopping=True,
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            ),

            # Memory for learning past strategies
            database=DatabaseConfig(
                enable_memory=True,
                adaptive_exploration=True
            )
        )

    def _portfolio_config(self) -> UnifiedEvolutionConfig:
        """
        Portfolio optimization configuration

        Multi-objective: maximize return, minimize risk
        """
        config = self._general_config()

        # Enable multi-objective optimization
        config.evolution_mode = EvolutionMode.PES  # Use PES, but will combine with MO

            # Add multi-objective settings
        config.mo = MOConfig(
            enabled=True,
            objectives=["return", "risk", "liquidity"],
            algorithm="nsga2",
            pareto_size=100
        )

        # Portfolio-specific constraints
        config.max_iterations = 100  # More for Pareto front

        return config

    def _risk_config(self) -> UnifiedEvolutionConfig:
        """
        Risk analysis configuration

        Focus on VaR/CVaR optimization
        """
        config = self._general_config()

        # Risk analysis needs even fewer evaluations
        config.max_iterations = 40

        # Lower temperature for more conservative (less risky) search
        config.llm.plan_temperature = 0.5
        config.llm.temperature = 0.5

        # Stricter early stopping (risk metrics stabilize quickly)
        config.evaluator.early_stopping_patience = 3

        return config

    def _asset_allocation_config(self) -> UnifiedEvolutionConfig:
        """
        Asset allocation configuration

        Constraint-heavy optimization
        """
        config = self._general_config()

        # Asset allocation has complex constraints
        config.max_iterations = 60

        # Higher planning iterations to explore constraint space
        config.pes.plan_iterations = 2

        return config

    # ========================================================================
    # DOMAIN-SPECIFIC EVALUATION
    # ========================================================================

    def evaluate_solution(
        self,
        solution: str,
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate financial solution

        Args:
            solution: Portfolio allocation code or weights
            problem: Problem description
            constraints: Constraints (max_assets, min_weight, etc.)

        Returns:
            Dictionary of financial metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     portfolio_code,
            ...     "Maximize return with min risk",
            ...     {"max_assets": 50}
            ... )
            >>> print(metrics['sharpe_ratio'])
        """
        # Parse solution to extract portfolio
        portfolio = self._parse_portfolio(solution)

        # Run a deterministic synthetic backtest over the parsed portfolio
        metrics = self._calculate_financial_metrics(
            portfolio,
            problem,
            constraints
        )

        return metrics

    def _parse_portfolio(self, solution: str) -> Dict[str, float]:
        """
        Parse portfolio from solution

        Args:
            solution: Solution code or text

        Returns:
            Dictionary of asset -> weight
        """
        import ast
        import re

        portfolio: Dict[str, float] = {}

        # Pattern 1: Dictionary format, e.g. {"AAPL": 0.3, "MSFT": 0.7}
        if "{" in solution and "}" in solution:
            for match in re.finditer(r'\{[^{}]+\}', solution):
                try:
                    candidate = ast.literal_eval(match.group(0))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(candidate, dict):
                    parsed = {
                        str(asset): float(weight)
                        for asset, weight in candidate.items()
                        if isinstance(weight, (int, float))
                    }
                    if parsed:
                        portfolio.update(parsed)

        # Pattern 2: Assignment format, e.g. "AAPL = 0.3"
        if not portfolio:
            assignments = re.findall(r'(\w+)\s*=\s*([0-9]*\.?[0-9]+)', solution)
            portfolio = {asset: float(weight) for asset, weight in assignments}

        return portfolio

    def _calculate_financial_metrics(
        self,
        portfolio: Dict[str, float],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate financial metrics from a deterministic synthetic backtest

        The portfolio is priced with the synthetic asset model in
        :mod:`openevolve.domain.heuristics`: every asset name maps to a stable
        expected return/volatility and every pair to a stable correlation, so
        diversification and concentration have real, reproducible effects. A
        daily return series is then generated from the resulting portfolio
        moments and the standard statistics are computed from that series.

        Args:
            portfolio: Asset weights
            problem: Problem description
            constraints: Constraints (max_assets, max_weight, ...)

        Returns:
            Dictionary of metrics (no external market data required)
        """
        if not portfolio:
            # Nothing investable was expressed: report a fully penalized result
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 1.0,
                "volatility": 1.0,
                "annual_return": 0.0,
                "var_95": 1.0,
                "cvar_95": 1.0,
            }

        expected_return, volatility = portfolio_moments(portfolio)

        # Constraint violations directly degrade the risk profile
        penalty = self._constraint_penalty(portfolio, constraints)
        expected_return *= (1.0 - 0.5 * penalty)
        volatility *= (1.0 + penalty)

        # Deterministic daily series with the portfolio's own moments
        daily_drift = expected_return / PERIODS_PER_YEAR
        daily_vol = max(1e-4, volatility / math.sqrt(PERIODS_PER_YEAR))
        seed = "|".join(f"{asset}:{weight:.6f}" for asset, weight in sorted(portfolio.items()))
        returns = synthetic_returns(
            seed, periods=PERIODS_PER_YEAR, drift=daily_drift, volatility=daily_vol
        )

        stats = return_statistics(returns)

        return {
            "sharpe_ratio": stats["sharpe_ratio"],
            "sortino_ratio": stats["sortino_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "volatility": stats["volatility"],
            "annual_return": stats["annual_return"],
            "var_95": stats["var_95"],
            "cvar_95": stats["cvar_95"],
        }

    def _constraint_penalty(
        self,
        portfolio: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Penalty in ``[0.0, 1.0]`` for violated portfolio constraints

        Args:
            portfolio: Asset weights
            constraints: Constraints dictionary

        Returns:
            Penalty factor (0.0 == fully compliant)
        """
        if not constraints or not portfolio:
            return 0.0

        total = sum(abs(w) for w in portfolio.values()) or 1.0
        weights = {asset: abs(w) / total for asset, w in portfolio.items()}

        violations = 0
        checks = 0

        if "max_assets" in constraints:
            checks += 1
            if len(weights) > int(constraints["max_assets"]):
                violations += 1

        if "min_weight" in constraints:
            checks += 1
            if any(weight < float(constraints["min_weight"]) for weight in weights.values()):
                violations += 1

        if "max_weight" in constraints:
            checks += 1
            if any(weight > float(constraints["max_weight"]) for weight in weights.values()):
                violations += 1

        if checks == 0:
            return 0.0

        return clamp(violations / checks)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_portfolio_constraints(
        self,
        max_assets: int = 50,
        min_weight: float = 0.01,
        max_weight: float = 0.4,
        sectors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get standard portfolio constraints

        Args:
            max_assets: Maximum number of assets
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            sectors: Sector exposure limits

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_portfolio_constraints(
            ...     max_assets=30,
            ...     sectors={"Tech": 0.4, "Healthcare": 0.3}
            ... )
        """
        constraints = {
            "max_assets": max_assets,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "weights_sum_to_1": True
        }

        if sectors:
            constraints["sector_limits"] = sectors

        return constraints

    def validate_portfolio(
        self,
        portfolio: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate portfolio against constraints

        Args:
            portfolio: Asset weights
            constraints: Constraints to validate

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_portfolio(
            ...     portfolio,
            ...     {"max_assets": 30}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check max assets
        if "max_assets" in constraints:
            if len(portfolio) > constraints["max_assets"]:
                violations.append(
                    f"Too many assets: {len(portfolio)} > {constraints['max_assets']}"
                )

        # Check min weight
        if "min_weight" in constraints:
            for asset, weight in portfolio.items():
                if weight < constraints["min_weight"]:
                    violations.append(
                        f"Asset {asset} below min weight: {weight} < {constraints['min_weight']}"
                    )

        # Check max weight
        if "max_weight" in constraints:
            for asset, weight in portfolio.items():
                if weight > constraints["max_weight"]:
                    violations.append(
                        f"Asset {asset} above max weight: {weight} > {constraints['max_weight']}"
                    )

        # Check weights sum to 1
        if constraints.get("weights_sum_to_1", False):
            total_weight = sum(portfolio.values())
            if abs(total_weight - 1.0) > 0.01:
                violations.append(
                    f"Weights don't sum to 1: {total_weight}"
                )

        return len(violations) == 0, violations
