"""
Science Domain Optimizer
Specialized optimizer for scientific research problems

Problems:
- Experimental design (DOE optimization)
- Data analysis (pipeline optimization)
- Hypothesis testing (experiment prioritization)

Best System: LoongFlow (PES) or OpenEvolve (QD)
Why: Very expensive experiments ($10k+ per run), need exploration

Configuration:
- Evaluation cost: "very_expensive"
- System: hybrid (LoongFlow for refinement, QD for exploration)
- Max experiments: 20 (hard constraint)
- QD grid resolution: 15
- Feature dimensions: [cost, accuracy, novelty]

Metrics:
- Statistical power
- Cost efficiency
- Discovery rate
- Reproducibility

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    QDConfig,
    MOConfig,
    PESConfig,
    LLMConfig,
    EvaluatorConfig,
    DatabaseConfig,
)
from .base import DomainOptimizer
from .heuristics import clamp, code_structure_score, saturating, signal_coverage

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based experiment allocation
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class ScienceOptimizer(DomainOptimizer):
    """
    Science domain optimizer

    Specialized for:
    - Experimental design
    - Data analysis
    - Hypothesis testing

    Example:
        >>> optimizer = ScienceOptimizer(sub_domain="experimental_design")
        >>> result = await optimizer.optimize(
        ...     "Optimize chemical reaction conditions for maximum yield",
        ...     constraints={"max_experiments": 20, "budget": 50000}
        ... )
        >>> print(result['domain_metrics']['statistical_power'])
    """

    domain_name = "science"

    # Signals used by the deterministic metric calculations
    RIGOR_SIGNALS = [
        "control", "randomiz", "blind", "replicate", "baseline",
        "confound", "power analysis", "sample_size"
    ]
    REPRODUCIBILITY_SIGNALS = [
        "seed", "protocol", "version", "requirements", "checksum",
        "calibrat", "log", "documented", "deterministic"
    ]
    ANALYSIS_SIGNALS = [
        "p_value", "p-value", "anova", "regression", "confidence_interval",
        "bootstrap", "bayes", "effect_size", "significance", "t_test"
    ]
    NOVELTY_SIGNALS = [
        "novel", "hypothesis", "exploratory", "screen", "sweep",
        "design_of_experiments", "doe", "latin_hypercube", "active_learning"
    ]

    def __init__(self, sub_domain: str = "general", use_adaptive_mdap: bool = True):
        """
        Initialize science optimizer

        Args:
            sub_domain: One of 'general', 'experimental_design', 'data_analysis', 'hypothesis_testing'
            use_adaptive_mdap: Whether to use Adaptive MDAP for complexity-based allocation
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "experimental_design": self._experimental_design_config(),
            "data_analysis": self._data_analysis_config(),
            "hypothesis_testing": self._hypothesis_testing_config()
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
        """Hybrid: LoongFlow for reasoning, OpenEvolve for exploration"""
        return "hybrid"

    def get_recommended_mode(self) -> str:
        """PES for refinement, QD for exploration"""
        return "hybrid"

    def get_domain_metrics(self) -> List[str]:
        """Science-specific metrics"""
        return [
            "statistical_power",
            "cost_efficiency",
            "discovery_rate",
            "reproducibility",
            "experimental_yield",
            "confidence_level",
            "effect_size",
            "novelty_score"
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
            id=f"science_{hash(problem) % 10000}",
            description=problem,
            domain="science",
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

        # Adjust iterations based on complexity and strategy
        if allocation.strategy == "DIRECT":
            # Simple problems: fewer iterations, standard mode
            config.max_iterations = min(15, config.max_iterations)
            config.evolution_mode = EvolutionMode.STANDARD
            config.qd.enabled = False
        elif allocation.strategy == "MDAP_LIGHT":
            # Light: moderate iterations
            config.max_iterations = min(25, config.max_iterations)
        elif allocation.strategy == "MDAP_MEDIUM":
            # Medium: standard config (no change)
            pass
        elif allocation.strategy == "MAKER_FULL":
            # Full: more iterations, enable MO
            config.max_iterations = max(30, config.max_iterations)
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True
        elif allocation.strategy == "MAKER_ULTRA":
            # Ultra: maximum iterations, full QD + MO
            config.max_iterations = max(50, config.max_iterations)
            config.qd.enabled = True
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True

        # Adjust evaluation timeout based on complexity
        if score > 0.7:
            # High complexity: longer timeouts
            config.evaluator.timeout = min(900, config.evaluator.timeout * 1.5)
        elif score < 0.3:
            # Low complexity: shorter timeouts
            config.evaluator.timeout = max(300, config.evaluator.timeout * 0.8)

        return config

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default science configuration"""
        return self._general_config()

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

    # ========================================================================
    # SUB-DOMAIN CONFIGURATIONS
    # ========================================================================

    from ..unified.config import QDConfig, LLMConfig, EvaluatorConfig, DatabaseConfig

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General science configuration

        Uses QD mode for exploring diverse experimental conditions
        """
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.SCIENCE,

            # Evolution mode
            evolution_mode=EvolutionMode.QD,

            # QD configuration (for exploration)
            qd=QDConfig(
                enabled=True,
                grid_resolution=15,
                feature_dimensions=["cost", "accuracy", "novelty"],
                archive_size=500
            ),

            # LLM configuration
            llm=LLMConfig(
                temperature=0.7,
                timeout=180,
                retries=2
            ),

            # Evaluation (very expensive experiments)
            max_iterations=20,  # Hard constraint on experiments
            evaluator=EvaluatorConfig(
                timeout=600,  # 10 minutes per experiment (setup + execution)
                max_retries=1,
                early_stopping=True,
                early_stopping_patience=3
            ),

            # Memory for learning experimental patterns
            database=DatabaseConfig(
                enable_memory=True,
                adaptive_exploration=True
            )
        )

    def _experimental_design_config(self) -> UnifiedEvolutionConfig:
        """
        Experimental design configuration

        Focus on Design of Experiments (DOE) optimization
        """
        config = self._general_config()

        # Higher QD resolution for fine-grained exploration
        config.qd.grid_resolution = 20

        # Multi-objective: maximize yield, minimize cost, maximize novelty
        config.mo = MOConfig(
            enabled=True,
            objectives=["yield", "cost", "reproducibility"],
            algorithm="nsga2",
            pareto_size=50
        )

        # Slightly more experiments for Pareto front
        config.max_iterations = 30

        return config

    def _data_analysis_config(self) -> UnifiedEvolutionConfig:
        """
        Data analysis configuration

        Optimize analysis pipeline
        """
        config = self._general_config()

        # Data analysis is less expensive (no wet lab)
        config.max_iterations = 50

        # Standard mode (no QD needed)
        config.evolution_mode = EvolutionMode.STANDARD
        config.qd.enabled = False

        # Lower temperature (more systematic)
        config.llm.temperature = 0.5

        return config

    def _hypothesis_testing_config(self) -> UnifiedEvolutionConfig:
        """
        Hypothesis testing configuration

        Optimize experiment prioritization
        """
        config = self._general_config()

        # Use PES for reasoning about which hypotheses to test
        config.evolution_mode = EvolutionMode.PES
        config.pes = PESConfig(
            enabled=True,
            enable_planning=True,
            use_memory=True,
            memory_top_k=10
        )

        # Disable QD
        config.qd.enabled = False

        # Fewer experiments (prioritize best hypotheses)
        config.max_iterations = 15

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
        Evaluate scientific solution

        Args:
            solution: Experimental design or analysis pipeline
            problem: Problem description
            constraints: Constraints (budget, max_experiments, etc.)

        Returns:
            Dictionary of scientific metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     experimental_design,
            ...     "Optimize chemical reaction",
            ...     {"max_experiments": 20, "budget": 50000}
            ... )
            >>> print(metrics['statistical_power'])
        """
        # Parse experimental design
        design = self._parse_experimental_design(solution)

        # Calculate metrics
        metrics = self._calculate_science_metrics(
            design,
            problem,
            constraints
        )

        return metrics

    def _parse_experimental_design(self, solution: str) -> Dict[str, Any]:
        """
        Parse experimental design from solution

        Args:
            solution: Solution code or text

        Returns:
            Experimental design components (including the raw text for scoring)
        """
        import re

        design: Dict[str, Any] = {
            "source": solution or "",
            "parameters": {},
            "conditions": [],
            "measurements": []
        }

        # Extract parameter ranges
        params = re.findall(r'(\w+)\s*[=:]\s*\[.*?\]', solution)
        if params:
            design["parameters"] = {"ranges": params}

        # Extract conditions
        conditions = re.findall(r'(?:if|when|condition)\s*:.*?(?=\n\s*\n|\n\s*[a-z_]+\s*[=:]|$)', solution, re.IGNORECASE)
        if conditions:
            design["conditions"] = conditions[:5]

        # Sample sizes / replicate counts drive statistical power
        design["sample_sizes"] = [
            float(value)
            for value in re.findall(
                r'(?:n|sample_size|replicates|runs|trials)\s*[=:]\s*(\d+)', solution, re.IGNORECASE
            )
        ]
        design["experiment_count"] = len(
            re.findall(r'(?:experiment|run|trial|assay)\b', solution, re.IGNORECASE)
        )

        return design

    def _calculate_science_metrics(
        self,
        design: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate scientific metrics from the experimental design itself

        Deterministic heuristic scorer (no lab automation or statistics service
        required): power comes from the declared sample sizes, reproducibility
        from seeds/protocol/version pinning, cost efficiency from the experiment
        count against the declared budget, and novelty from the breadth of the
        design space that is explored.

        Args:
            design: Experimental design from :meth:`_parse_experimental_design`
            problem: Problem description
            constraints: Constraints (max_experiments, budget, ...)

        Returns:
            Dictionary of metrics
        """
        source = design.get("source", "")

        sample_sizes = design.get("sample_sizes", [])
        largest_sample = max(sample_sizes) if sample_sizes else 0.0
        experiment_count = max(design.get("experiment_count", 0), len(design.get("conditions", [])))
        factor_count = len(design.get("parameters", {}).get("ranges", []))

        rigor = signal_coverage(source, self.RIGOR_SIGNALS)
        reproducibility_signals = signal_coverage(source, self.REPRODUCIBILITY_SIGNALS)
        analysis_signals = signal_coverage(source, self.ANALYSIS_SIGNALS)
        novelty_signals = signal_coverage(source, self.NOVELTY_SIGNALS)

        # Statistical power: grows with sample size and replication, capped
        statistical_power = clamp(
            0.2 + 0.5 * saturating(largest_sample, 60) + 0.3 * rigor
        )

        # Cost efficiency: fewer experiments per declared factor is better, and
        # an explicit budget must be respected
        budgeted_experiments = None
        if constraints:
            if "max_experiments" in constraints:
                budgeted_experiments = float(constraints["max_experiments"])
            elif "budget" in constraints and "cost_per_experiment" in constraints:
                cost = float(constraints["cost_per_experiment"]) or 1.0
                budgeted_experiments = float(constraints["budget"]) / cost

        if budgeted_experiments and experiment_count:
            cost_efficiency = clamp(1.0 - saturating(experiment_count, budgeted_experiments) * 0.8)
        else:
            cost_efficiency = clamp(0.4 + 0.6 * (1.0 - saturating(experiment_count, 40)))

        effect_size = clamp(0.2 + 0.5 * analysis_signals + 0.3 * saturating(factor_count, 4))
        confidence_level = clamp(0.5 + 0.5 * statistical_power)
        reproducibility = clamp(
            0.25 + 0.5 * reproducibility_signals + 0.25 * code_structure_score(source)
        )
        discovery_rate = clamp(
            0.15 + 0.4 * novelty_signals + 0.25 * saturating(factor_count, 5)
            + 0.2 * statistical_power
        )
        experimental_yield = clamp(
            0.3 * statistical_power + 0.3 * reproducibility + 0.4 * cost_efficiency
        )
        novelty_score = clamp(0.2 + 0.6 * novelty_signals + 0.2 * saturating(factor_count, 6))

        return {
            "statistical_power": statistical_power,
            "cost_efficiency": cost_efficiency,
            "discovery_rate": discovery_rate,
            "reproducibility": reproducibility,
            "experimental_yield": experimental_yield,
            "confidence_level": confidence_level,
            "effect_size": effect_size,
            "novelty_score": novelty_score,
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_experiment_constraints(
        self,
        max_experiments: int = 20,
        budget: float = 50000,
        cost_per_experiment: float = 2500,
        min_statistical_power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Get standard experiment constraints

        Args:
            max_experiments: Maximum number of experiments
            budget: Total budget ($)
            cost_per_experiment: Cost per experiment ($)
            min_statistical_power: Minimum statistical power

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_experiment_constraints(
            ...     max_experiments=30,
            ...     budget=100000
            ... )
        """
        return {
            "max_experiments": max_experiments,
            "budget": budget,
            "cost_per_experiment": cost_per_experiment,
            "min_statistical_power": min_statistical_power
        }

    def estimate_experiment_cost(
        self,
        design: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Estimate total experiment cost

        Args:
            design: Experimental design
            constraints: Constraints

        Returns:
            Estimated total cost

        Example:
            >>> cost = optimizer.estimate_experiment_cost(
            ...     design,
            ...     {"cost_per_experiment": 2500}
            ... )
        """
        if constraints and "cost_per_experiment" in constraints:
            cost_per_exp = constraints["cost_per_experiment"]
        else:
            cost_per_exp = 2500

        if constraints and "max_experiments" in constraints:
            num_experiments = constraints["max_experiments"]
        else:
            num_experiments = 20

        return cost_per_exp * num_experiments

    def validate_experimental_design(
        self,
        design: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate experimental design

        Args:
            design: Experimental design
            constraints: Constraints

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_experimental_design(
            ...     design,
            ...     {"max_experiments": 20}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check budget
        if "budget" in constraints:
            estimated_cost = self.estimate_experiment_cost(design, constraints)
            if estimated_cost > constraints["budget"]:
                violations.append(
                    f"Budget exceeded: ${estimated_cost:,.0f} > ${constraints['budget']:,.0f}"
                )

        # Check max experiments
        if "max_experiments" in constraints:
            # Count experiments in design
            num_experiments = len(design.get("conditions", []))
            if num_experiments > constraints["max_experiments"]:
                violations.append(
                    f"Too many experiments: {num_experiments} > {constraints['max_experiments']}"
                )

        return len(violations) == 0, violations

    def suggest_doe_parameters(
        self,
        problem: str,
        num_factors: int,
        resolution: str = "IV"
    ) -> Dict[str, Any]:
        """
        Suggest Design of Experiments (DOE) parameters

        Args:
            problem: Problem description
            num_factors: Number of experimental factors
            resolution: DOE resolution (II, III, IV, V)

        Returns:
            DOE parameters

        Example:
            >>> doe_params = optimizer.suggest_doe_parameters(
            ...     "Optimize chemical reaction",
            ...     num_factors=5,
            ...     resolution="IV"
            ... )
        """
        # Suggest fractional factorial design
        if resolution == "II":
            runs = 2 ** (num_factors - 1)
        elif resolution == "III":
            runs = 2 ** (num_factors - 2)
        elif resolution == "IV":
            runs = 2 ** (num_factors - 1)
        else:  # V or full factorial
            runs = 2 ** num_factors

        return {
            "design_type": "fractional_factorial",
            "num_factors": num_factors,
            "resolution": resolution,
            "num_runs": min(runs, 32),  # Cap at 32 for practicality
            "replicates": 2,
            "randomization": True,
            "blocking": False
        }
