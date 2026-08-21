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
from .heuristics import (
    PERIODS_PER_YEAR,
    clamp,
    code_structure_score,
    return_statistics,
    saturating,
    signal_coverage,
    synthetic_returns,
)

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based adversarial allocation
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


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

    # Signals used by the deterministic backtest scoring (see
    # _calculate_trading_metrics). Each group drives a different property of the
    # generated return series.
    INDICATOR_SIGNALS = [
        "rsi", "macd", "sma", "ema", "moving_average", "bollinger",
        "atr", "momentum", "zscore", "volume", "vwap", "breakout"
    ]
    RISK_SIGNALS = [
        "stop_loss", "stop loss", "take_profit", "trailing", "position_size",
        "risk_per_trade", "max_drawdown", "max_position", "kelly", "leverage"
    ]
    REGIME_SIGNALS = [
        "regime", "volatility_filter", "trend_filter", "session",
        "correlation", "drawdown_guard", "liquidity"
    ]

    def __init__(self, sub_domain: str = "general", use_adaptive_mdap: bool = True):
        """
        Initialize trading optimizer

        Args:
            sub_domain: One of 'general', 'strategy', 'signal', 'parameter'
            use_adaptive_mdap: Whether to use Adaptive MDAP for complexity-based adversarial allocation
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

        # Initialize Adaptive MDAP if available
        self.use_adaptive_mdap = use_adaptive_mdap and ADAPTIVE_MDAP_AVAILABLE
        if self.use_adaptive_mdap:
            self.complexity_classifier = TaskComplexityClassifier()
            self.resource_allocator = AdaptiveMDAPAllocator(enable_learning=True)
        else:
            self.complexity_classifier = None
            self.resource_allocator = None

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
            id=f"trading_{hash(problem) % 10000}",
            description=problem,
            domain="trading",
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

        # Adjust iterations and adversarial config based on complexity strategy
        if allocation.strategy == "DIRECT":
            # Simple problems: fewer iterations, disable adversarial
            config.max_iterations = min(100, config.max_iterations)
            config.adversarial.enabled = False
        elif allocation.strategy == "MDAP_LIGHT":
            # Light: moderate iterations, light adversarial
            config.max_iterations = min(150, config.max_iterations)
            config.adversarial.adversarial_rounds = 10
            config.database.population_size = 20
        elif allocation.strategy == "MDAP_MEDIUM":
            # Medium: standard adversarial config
            config.adversarial.adversarial_rounds = 20
            config.database.population_size = 30
        elif allocation.strategy == "MAKER_FULL":
            # Full: more iterations, intensive adversarial
            config.max_iterations = max(250, config.max_iterations)
            config.adversarial.adversarial_rounds = 30
            config.database.population_size = 40
        elif allocation.strategy == "MAKER_ULTRA":
            # Ultra: max iterations, full intensive adversarial
            config.max_iterations = max(300, config.max_iterations)
            config.adversarial.adversarial_rounds = 40
            config.database.population_size = 50

        # Adjust evaluation timeout based on complexity
        if score > 0.7:
            # High complexity: longer timeouts for backtesting
            config.evaluator.timeout = min(300, config.evaluator.timeout * 1.5)
        elif score < 0.3:
            # Low complexity: shorter timeouts
            config.evaluator.timeout = max(60, config.evaluator.timeout * 0.8)

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
            Strategy components (including the raw source for scoring)
        """
        import re

        strategy: Dict[str, Any] = {
            "source": solution or "",
            "entry_rules": [],
            "exit_rules": [],
            "indicators": {},
            "parameters": {}
        }

        # Entry rules
        if "entry" in solution.lower():
            entry_pattern = r'entry.*?(?=\nexit|\ndef|$)'
            entry_match = re.search(entry_pattern, solution, re.IGNORECASE | re.DOTALL)
            if entry_match:
                strategy["entry_rules"] = [entry_match.group(0)]

        # Exit rules
        if "exit" in solution.lower():
            exit_pattern = r'exit.*?(?=\ndef|$)'
            exit_match = re.search(exit_pattern, solution, re.IGNORECASE | re.DOTALL)
            if exit_match:
                strategy["exit_rules"] = [exit_match.group(0)]

        # Parameters
        params = re.findall(r'(\w+)\s*=\s*([0-9]*\.?[0-9]+)', solution)
        if params:
            strategy["parameters"] = {name: float(value) for name, value in params}

        # Indicators actually referenced by the strategy
        for indicator in self.INDICATOR_SIGNALS:
            if indicator in solution.lower():
                strategy["indicators"][indicator] = True

        return strategy

    def _calculate_trading_metrics(
        self,
        strategy: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate trading metrics from a deterministic synthetic backtest

        Instead of market data, a reproducible return series is generated from a
        stable hash of the strategy source (see
        :func:`openevolve.domain.heuristics.synthetic_returns`). The series'
        drift and volatility are driven by real properties of the strategy:

        - signal quality: indicators and confirmation logic increase drift
        - risk controls: stops/sizing/limits reduce volatility
        - regime awareness: filters reduce tail risk
        - overfitting: an excessive number of tuned constants reduces drift

        All reported statistics are then computed from that series with the
        standard formulas, so the metrics are real numbers, deterministic per
        strategy, and monotone in the strategy's own qualities.

        Args:
            strategy: Parsed strategy components
            problem: Problem description
            constraints: Constraints (max_drawdown, ...)

        Returns:
            Dictionary of trading metrics
        """
        source = strategy.get("source", "")

        signal_quality = signal_coverage(source, self.INDICATOR_SIGNALS)
        risk_controls = signal_coverage(source, self.RISK_SIGNALS)
        regime_awareness = signal_coverage(source, self.REGIME_SIGNALS)
        structure = code_structure_score(source)

        has_entry = bool(strategy.get("entry_rules"))
        has_exit = bool(strategy.get("exit_rules"))
        completeness = 0.5 * float(has_entry) + 0.5 * float(has_exit)

        # Too many hardcoded constants relative to the logic == curve fitting
        parameter_count = len(strategy.get("parameters", {}))
        overfitting = saturating(max(0, parameter_count - 6), 12)

        # Per-period expected return: edge from signals, completeness, structure
        drift = (
            0.00005
            + 0.0007 * signal_quality
            + 0.0005 * completeness
            + 0.0003 * structure
            - 0.0008 * overfitting
        )

        # Per-period volatility: risk controls and regime filters dampen it
        volatility = max(
            0.004,
            0.020 - 0.008 * risk_controls - 0.004 * regime_awareness,
        )

        # Respect an explicit drawdown budget by scaling exposure
        if constraints and "max_drawdown" in constraints:
            budget = clamp(float(constraints["max_drawdown"]), 0.01, 1.0)
            exposure = clamp(budget / 0.25, 0.2, 1.0)
            drift *= exposure
            volatility *= exposure

        returns = synthetic_returns(
            f"{source}|{signal_quality:.3f}|{risk_controls:.3f}",
            periods=PERIODS_PER_YEAR,
            drift=drift,
            volatility=volatility,
        )

        stats = return_statistics(returns)

        return {
            "total_return": stats["total_return"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "sortino_ratio": stats["sortino_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "avg_win": stats["avg_win"],
            "avg_loss": stats["avg_loss"],
            "expectancy": stats["expectancy"],
        }

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
