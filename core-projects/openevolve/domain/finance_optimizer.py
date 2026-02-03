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

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig, MOConfig, PESConfig, LLMConfig, EvaluatorConfig, DatabaseConfig
from . import EvolutionMode, DomainType
from .base import DomainOptimizer


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

    def __init__(self, sub_domain: str = "general"):
        """
        Initialize finance optimizer

        Args:
            sub_domain: One of 'general', 'portfolio', 'risk', 'asset_allocation'
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

        # Run financial metrics (placeholder - integrate with backtesting engine)
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
        # Placeholder: Parse portfolio weights from solution
        # In production, would integrate with backtesting engine

        # Simple parsing: look for common patterns
        portfolio = {}

        # Pattern 1: Dictionary format
        if "{" in solution and "}" in solution:
            # Extract dictionary
            import re
            dict_pattern = r'\{[^}]+\}'
            match = re.search(dict_pattern, solution)
            if match:
                # Try to eval safely
                try:
                    portfolio = eval(match.group(0))
                except:
                    pass

        # Pattern 2: Assignment format
        # Look for lines like "AAPL = 0.3"
        import re
        assignments = re.findall(r'(\w+)\s*=\s*([0-9.]+)', solution)
        if assignments:
            portfolio = {asset: float(weight) for asset, weight in assignments}

        return portfolio

    def _calculate_financial_metrics(
        self,
        portfolio: Dict[str, float],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate financial metrics

        Args:
            portfolio: Asset weights
            problem: Problem description
            constraints: Constraints

        Returns:
            Dictionary of metrics

        Note: This is a placeholder. In production, integrate with:
        - Backtesting engine (e.g., Backtrader, Zipline)
        - Market data provider
        - Risk calculation library
        """
        # Placeholder metrics - replace with real calculations
        metrics = {
            "sharpe_ratio": 1.5,      # Risk-adjusted return
            "sortino_ratio": 2.0,     # Downside risk-adjusted return
            "max_drawdown": 0.15,     # Maximum loss from peak
            "volatility": 0.20,       # Annualized volatility
            "annual_return": 0.12,    # Annual return
            "var_95": 0.05,           # Value at risk at 95% confidence
            "cvar_95": 0.08           # Conditional VaR at 95%
        }

        # In production, would run:
        # 1. Backtest portfolio with historical data
        # 2. Calculate returns
        # 3. Compute metrics from returns

        # Example backtesting flow:
        # returns = backtest(portfolio, market_data, constraints)
        # metrics['sharpe_ratio'] = calculate_sharpe(returns)
        # metrics['sortino_ratio'] = calculate_sortino(returns)
        # metrics['max_drawdown'] = calculate_max_drawdown(returns)
        # metrics['volatility'] = returns.std() * np.sqrt(252)

        return metrics

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
