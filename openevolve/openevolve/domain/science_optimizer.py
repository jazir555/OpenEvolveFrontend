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
from ..unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType, QDConfig, LLMConfig, EvaluatorConfig, DatabaseConfig
from .base import DomainOptimizer


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

    def __init__(self, sub_domain: str = "general"):
        """
        Initialize science optimizer

        Args:
            sub_domain: One of 'general', 'experimental_design', 'data_analysis', 'hypothesis_testing'
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

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default science configuration"""
        return self._general_config()

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
        config.mo = MoConfig(
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
        config.pes = PesConfig(
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
            Experimental design components
        """
        # Placeholder: Parse experimental parameters
        design = {
            "parameters": {},
            "conditions": [],
            "measurements": []
        }

        # Extract parameter ranges
        import re
        params = re.findall(r'(\w+)\s*[=:]\s*\[.*?\]', solution)
        if params:
            design["parameters"] = {"ranges": params}

        # Extract conditions
        import re
        conditions = re.findall(r'(?:if|when|condition)\s*:.*?(?=\n\s*\n|\n\s*[a-z_]+\s*[=:]|$)', solution, re.IGNORECASE)
        if conditions:
            design["conditions"] = conditions[:5]

        return design

    def _calculate_science_metrics(
        self,
        design: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate scientific metrics

        Args:
            design: Experimental design
            problem: Problem description
            constraints: Constraints

        Returns:
            Dictionary of metrics

        Note: This is a placeholder. In production, integrate with:
        - Laboratory automation systems
        - Data analysis pipelines
        - Statistical analysis tools
        """
        # Placeholder metrics
        metrics = {
            "statistical_power": 0.85,    # 85% power
            "cost_efficiency": 0.75,      # 75% cost efficiency
            "discovery_rate": 0.60,       # 60% discovery rate
            "reproducibility": 0.90,      # 90% reproducibility
            "experimental_yield": 0.78,   # 78% yield
            "confidence_level": 0.95,     # 95% confidence
            "effect_size": 0.65,          # Medium effect
            "novelty_score": 0.70         # 70% novelty
        }

        # In production, would:
        # 1. Run experiments (or simulate)
        # 2. Analyze results statistically
        # 3. Calculate metrics

        return metrics

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
