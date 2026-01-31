"""
Trading Domain Optimizer
Specialized optimizer for trading strategy development

Problems:
- Strategy development (entry/exit rules)
- Signal optimization (indicator parameters)
- Parameter tuning (stop loss, take profit)

Best System: OpenEvolve (Adversarial mode)
Why: Market regimes change, need robustness against adverse conditions

Configuration:
- Evaluation cost: "expensive"
- Adversarial rounds: 20
- Red team models: [gpt-4, claude-3, llama-3]
- Attack types: regime_change, volatility_spike, black_swan
- Population size: 30

Metrics:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Profit factor

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType, AdversarialConfig, LLMConfig, EvaluatorConfig, DatabaseConfig
from .base import DomainOptimizer


class TradingOptimizer(DomainOptimizer):
    """
    Trading domain optimizer

    Specialized for:
    - Strategy development
    - Signal optimization
    - Parameter tuning

    Example:
        >>> optimizer = TradingOptimizer(sub_domain="strategy")
        >>> result = await optimizer.optimize(
        ...     "Develop momentum trading strategy with entry/exit rules",
        ...     constraints={"max_drawdown": 0.2}
        ... )
        >>> print(result['domain_metrics']['sharpe_ratio'])
    """

    domain_name = "trading"

    def __init__(self, sub_domain: str = "general"):
        """
        Initialize trading optimizer

        Args:
            sub_domain: One of 'general', 'strategy', 'signal', 'parameter'
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "strategy": self._strategy_config(),
            "signal": self._signal_config(),
            "parameter": self._parameter_config()
        }

        # Set active config
        self.config = self.sub_domain_configs.get(
            sub_domain,
            self._general_config()
        )

    def get_recommended_system(self) -> str:
        """OpenEvolve adversarial for robustness testing"""
        return "openevolve"

    def get_recommended_mode(self) -> str:
        """Adversarial mode for market regime testing"""
        return "adversarial"

    def get_domain_metrics(self) -> List[str]:
        """Trading-specific metrics"""
        return [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
            "expectancy"
        ]

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default trading configuration"""
        return self._general_config()

    # ========================================================================
    # SUB-DOMAIN CONFIGURATIONS
    # ========================================================================

    from ..unified.config import AdversarialConfig, LLMConfig, EvaluatorConfig, DatabaseConfig

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General trading configuration

        Uses adversarial mode to test against market regime changes
        """
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.TRADING,

            # Evolution mode
            evolution_mode=EvolutionMode.ADVERSARIAL,

            # Adversarial configuration
            adversarial=AdversarialConfig(
                enabled=True,
                adversarial_rounds=20,
                red_team_models=["gpt-4", "claude-3-opus", "llama-3-70b"],
                blue_team_models=["gpt-4", "claude-3-opus"],
                robustness_threshold=0.8
            ),

            # LLM configuration
            llm=LLMConfig(
                temperature=0.8,  # Higher creativity for strategy discovery
                timeout=120,
                retries=3
            ),

            # Evaluation
            max_iterations=200,  # More generations for robustness
            evaluator=EvaluatorConfig(
                timeout=120,  # 2 minutes per backtest
                max_retries=2,
                early_stopping=True,
                early_stopping_patience=10,
                parallel_evaluations=4
            ),

            # Population diversity
            database=DatabaseConfig(
                population_size=30,
                archive_size=50,
                diversity_metric="edit_distance"
            )
        )

    def _strategy_config(self) -> UnifiedEvolutionConfig:
        """
        Strategy development configuration

        Focus on discovering entry/exit rules
        """
        config = self._general_config()

        # More adversarial rounds for thorough regime testing
        config.adversarial.adversarial_rounds = 30

        # Higher temperature for more creative strategies
        config.llm.temperature = 0.9

        # Larger population for strategy diversity
        config.database.population_size = 50

        return config

    def _signal_config(self) -> UnifiedEvolutionConfig:
        """
        Signal optimization configuration

        Optimize indicator parameters (RSI, MACD, etc.)
        """
        config = self._general_config()

        # Signal optimization is more structured
        config.llm.temperature = 0.6  # Lower temperature

        # Fewer rounds (converges faster)
        config.adversarial.adversarial_rounds = 15

        # Smaller population (parameter space smaller)
        config.database.population_size = 25

        return config

    def _parameter_config(self) -> UnifiedEvolutionConfig:
        """
        Parameter tuning configuration

        Fine-tune stop loss, take profit, position sizing
        """
        config = self._general_config()

        # Parameter tuning is very structured
        config.llm.temperature = 0.4  # Even lower

        # Fewer iterations
        config.max_iterations = 100

        # Smaller population
        config.database.population_size = 20

        # Focus on robustness
        config.adversarial.robustness_threshold = 0.85

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
        Evaluate trading strategy

        Args:
            solution: Trading strategy code
            problem: Problem description
            constraints: Constraints (max_drawdown, etc.)

        Returns:
            Dictionary of trading metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     strategy_code,
            ...     "Develop momentum strategy",
            ...     {"max_drawdown": 0.2}
            ... )
            >>> print(metrics['sharpe_ratio'])
        """
        # Parse strategy
        strategy = self._parse_strategy(solution)

        # Run backtest with adversarial scenarios
        metrics = self._calculate_trading_metrics(
            strategy,
            problem,
            constraints
        )

        return metrics

    def _parse_strategy(self, solution: str) -> Dict[str, Any]:
        """
        Parse trading strategy from solution

        Args:
            solution: Strategy code

        Returns:
            Strategy components
        """
        # Placeholder: Parse strategy components
        strategy = {
            "entry_rules": [],
            "exit_rules": [],
            "indicators": {},
            "parameters": {}
        }

        # Look for common patterns
        # Entry rules
        if "entry" in solution.lower():
            import re
            entry_pattern = r'entry.*?(?=\nexit|\ndef|$)'
            entry_match = re.search(entry_pattern, solution, re.IGNORECASE | re.DOTALL)
            if entry_match:
                strategy["entry_rules"] = [entry_match.group(0)]

        # Exit rules
        if "exit" in solution.lower():
            import re
            exit_pattern = r'exit.*?(?=\ndef|$)'
            exit_match = re.search(exit_pattern, solution, re.IGNORECASE | re.DOTALL)
            if exit_match:
                strategy["exit_rules"] = [exit_match.group(0)]

        # Parameters
        import re
        params = re.findall(r'(\w+)\s*=\s*([0-9.]+)', solution)
        if params:
            strategy["parameters"] = {name: float(value) for name, value in params}

        return strategy

    def _calculate_trading_metrics(
        self,
        strategy: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate trading metrics

        Args:
            strategy: Strategy components
            problem: Problem description
            constraints: Constraints

        Returns:
            Dictionary of metrics

        Note: This is a placeholder. In production, integrate with:
        - Backtesting engine
        - Market data provider
        - Adversarial scenario generator
        """
        # Placeholder metrics
        metrics = {
            "total_return": 0.45,      # 45% total return
            "sharpe_ratio": 1.8,       # Risk-adjusted return
            "sortino_ratio": 2.5,      # Downside risk-adjusted
            "max_drawdown": 0.18,      # 18% max drawdown
            "win_rate": 0.55,          # 55% win rate
            "profit_factor": 2.2,      # Profit/loss ratio
            "avg_win": 0.03,           # 3% average win
            "avg_loss": 0.015,         # 1.5% average loss
            "expectancy": 0.012        # Expected return per trade
        }

        # In production, would:
        # 1. Backtest strategy on historical data
        # 2. Test against adversarial scenarios (regime changes, etc.)
        # 3. Calculate metrics from backtest results

        return metrics

    # ========================================================================
    # ADVERSARIAL SCENARIO GENERATION
    # ========================================================================

    def generate_adversarial_scenarios(
        self,
        base_data: Any,
        scenario_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate adversarial market scenarios

        Args:
            base_data: Base market data
            scenario_types: Types of scenarios to generate

        Returns:
            List of scenarios

        Example:
            >>> scenarios = optimizer.generate_adversarial_scenarios(
            ...     market_data,
            ...     ["regime_change", "volatility_spike"]
            ... )
        """
        if scenario_types is None:
            scenario_types = ["regime_change", "volatility_spike", "black_swan"]

        scenarios = []

        for scenario_type in scenario_types:
            if scenario_type == "regime_change":
                scenarios.append(self._create_regime_change_scenario(base_data))
            elif scenario_type == "volatility_spike":
                scenarios.append(self._create_volatility_spike_scenario(base_data))
            elif scenario_type == "black_swan":
                scenarios.append(self._create_black_swan_scenario(base_data))

        return scenarios

    def _create_regime_change_scenario(self, base_data: Any) -> Dict[str, Any]:
        """Create regime change scenario"""
        return {
            "type": "regime_change",
            "description": "Market transitions from bull to bear",
            "parameters": {
                "trend_change": -0.3,  # 30% trend reversal
                "volatility_increase": 1.5
            }
        }

    def _create_volatility_spike_scenario(self, base_data: Any) -> Dict[str, Any]:
        """Create volatility spike scenario"""
        return {
            "type": "volatility_spike",
            "description": "Sudden increase in market volatility",
            "parameters": {
                "volatility_multiplier": 3.0,
                "duration_days": 10
            }
        }

    def _create_black_swan_scenario(self, base_data: Any) -> Dict[str, Any]:
        """Create black swan scenario"""
        return {
            "type": "black_swan",
            "description": "Extreme market event",
            "parameters": {
                "price_shock": -0.5,  # 50% drop
                "recovery_time_days": 90
            }
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_trading_constraints(
        self,
        max_drawdown: float = 0.2,
        min_win_rate: float = 0.4,
        min_profit_factor: float = 1.5,
        max_trades_per_day: int = 10
    ) -> Dict[str, Any]:
        """
        Get standard trading constraints

        Args:
            max_drawdown: Maximum acceptable drawdown
            min_win_rate: Minimum win rate
            min_profit_factor: Minimum profit factor
            max_trades_per_day: Maximum trades per day

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_trading_constraints(
            ...     max_drawdown=0.15,
            ...     min_win_rate=0.5
            ... )
        """
        return {
            "max_drawdown": max_drawdown,
            "min_win_rate": min_win_rate,
            "min_profit_factor": min_profit_factor,
            "max_trades_per_day": max_trades_per_day
        }

    def validate_strategy(
        self,
        metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate strategy against constraints

        Args:
            metrics: Trading metrics
            constraints: Constraints to validate

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_strategy(
            ...     metrics,
            ...     {"max_drawdown": 0.2}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check max drawdown
        if "max_drawdown" in constraints:
            if metrics.get("max_drawdown", 0) > constraints["max_drawdown"]:
                violations.append(
                    f"Max drawdown exceeded: {metrics['max_drawdown']:.2%} > {constraints['max_drawdown']:.2%}"
                )

        # Check min win rate
        if "min_win_rate" in constraints:
            if metrics.get("win_rate", 0) < constraints["min_win_rate"]:
                violations.append(
                    f"Win rate below minimum: {metrics['win_rate']:.2%} < {constraints['min_win_rate']:.2%}"
                )

        # Check min profit factor
        if "min_profit_factor" in constraints:
            if metrics.get("profit_factor", 0) < constraints["min_profit_factor"]:
                violations.append(
                    f"Profit factor below minimum: {metrics['profit_factor']:.2f} < {constraints['min_profit_factor']:.2f}"
                )

        return len(violations) == 0, violations
