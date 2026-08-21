"""
Engineering Domain Optimizer
Specialized optimizer for engineering design problems

Problems:
- Structural optimization (weight vs strength)
- Circuit design (power vs area)
- Control systems (response vs stability)

Best System: Hybrid (LoongFlow + Adversarial)
Why: Expensive FEA simulations + safety-critical

Configuration:
- Evaluation cost: "expensive"
- Primary: LoongFlow PES
- Secondary: OpenEvolve Adversarial (safety verification)
- Adversarial attacks: load_exceedance, fatigue, resonance
- Max simulations: 100

Metrics:
- Performance
- Safety margin
- Cost
- Weight
- Reliability

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    AdversarialConfig,
    LLMConfig,
    EvaluatorConfig,
    DatabaseConfig,
    MOConfig,
)
from .base import DomainOptimizer
from .heuristics import clamp, code_structure_score, saturating, signal_coverage

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based simulation allocation
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class EngineeringOptimizer(DomainOptimizer):
    """
    Engineering domain optimizer

    Specialized for:
    - Structural optimization
    - Circuit design
    - Control systems

    Example:
        >>> optimizer = EngineeringOptimizer(sub_domain="structural")
        >>> result = await optimizer.optimize(
        ...     "Design lightweight bridge that supports 50 tons",
        ...     constraints={"max_weight": 1000, "min_safety_factor": 2.0}
        ... )
        >>> print(result['domain_metrics']['safety_factor'])
    """

    domain_name = "engineering"

    # Signals used by the deterministic metric calculations
    ANALYSIS_SIGNALS = [
        "fea", "finite_element", "mesh", "stress", "strain", "modal",
        "simulation", "spice", "transfer_function", "bode", "cfd"
    ]
    ROBUSTNESS_SIGNALS = [
        "safety_factor", "margin", "fatigue", "tolerance", "derating",
        "redundan", "failure_mode", "fmea", "worst_case", "monte_carlo"
    ]
    MANUFACTURING_SIGNALS = [
        "tolerance", "machin", "weld", "assembly", "dfm", "standard_part",
        "extrusion", "casting", "additive", "fastener"
    ]
    THERMAL_SIGNALS = [
        "thermal", "heat", "conduct", "convect", "temperature",
        "cooling", "dissipat", "ambient"
    ]
    MATERIAL_SIGNALS = [
        "steel", "aluminum", "aluminium", "titanium", "composite",
        "carbon_fiber", "concrete", "polymer", "copper", "abs"
    ]

    def __init__(self, sub_domain: str = "general", use_adaptive_mdap: bool = True):
        """
        Initialize engineering optimizer

        Args:
            sub_domain: One of 'general', 'structural', 'circuit', 'control'
            use_adaptive_mdap: Whether to use Adaptive MDAP for complexity-based allocation
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "structural": self._structural_config(),
            "circuit": self._circuit_config(),
            "control": self._control_config()
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
        """Hybrid: LoongFlow PES + Adversarial"""
        return "hybrid"

    def get_recommended_mode(self) -> str:
        """PES for design, Adversarial for safety"""
        return "hybrid"

    def get_domain_metrics(self) -> List[str]:
        """Engineering-specific metrics"""
        return [
            "performance",
            "safety_margin",
            "cost",
            "weight",
            "reliability",
            "efficiency",
            "manufacturability",
            "thermal_performance"
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
            id=f"engineering_{hash(problem) % 10000}",
            description=problem,
            domain="engineering",
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

        # Get sub-domain base iterations
        base_iterations = {
            "general": 100,
            "circuit": 150,
            "control": 200
        }.get(self.sub_domain, 100)

        # Adjust iterations and adversarial rounds based on complexity and strategy
        if allocation.strategy == "DIRECT":
            # Simple problems: fewer iterations, lighter adversarial testing
            config.max_iterations = max(50, int(base_iterations * 0.5))
            config.adversarial.adversarial_rounds = max(3, int(config.adversarial.adversarial_rounds * 0.5))
            config.adversarial.robustness_threshold = 0.85  # Slightly lower threshold for simple problems
        elif allocation.strategy == "MDAP_LIGHT":
            # Light: moderate iterations, reduced adversarial rounds
            config.max_iterations = max(75, int(base_iterations * 0.75))
            config.adversarial.adversarial_rounds = max(5, int(config.adversarial.adversarial_rounds * 0.7))
        elif allocation.strategy == "MDAP_MEDIUM":
            # Medium: standard config (no change to iterations)
            pass
        elif allocation.strategy == "MAKER_FULL":
            # Full: more iterations, thorough adversarial testing
            config.max_iterations = max(base_iterations, int(base_iterations * 1.25))
            config.adversarial.adversarial_rounds = int(config.adversarial.adversarial_rounds * 1.3)
            config.adversarial.robustness_threshold = min(0.95, config.adversarial.robustness_threshold + 0.02)
        elif allocation.strategy == "MAKER_ULTRA":
            # Ultra: maximum iterations, extensive adversarial testing
            config.max_iterations = int(base_iterations * 1.5)
            config.adversarial.adversarial_rounds = int(config.adversarial.adversarial_rounds * 1.5)
            config.adversarial.robustness_threshold = min(0.98, config.adversarial.robustness_threshold + 0.05)

        # Adjust evaluation timeout based on complexity
        if score > 0.7:
            # High complexity: longer timeouts for thorough simulations
            config.evaluator.timeout = min(1200, int(config.evaluator.timeout * 1.5))
        elif score < 0.3:
            # Low complexity: shorter timeouts for faster simulations
            config.evaluator.timeout = max(120, int(config.evaluator.timeout * 0.7))

        return config

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default engineering configuration"""
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

    from ..unified.config import PESConfig, AdversarialConfig, LLMConfig, EvaluatorConfig, DatabaseConfig

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General engineering configuration

        Uses PES for design + Adversarial for safety testing
        """
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.ENGINEERING,

            # Evolution mode
            evolution_mode=EvolutionMode.PES,

            # PES configuration (for design)
            pes=PESConfig(
                enabled=True,
                enable_planning=True,
                use_memory=True
            ),

            # Adversarial configuration (for safety)
            adversarial=AdversarialConfig(
                enabled=True,
                adversarial_rounds=10,
                red_team_models=["gpt-4"],
                robustness_threshold=0.9  # High for safety
            ),

            # LLM configuration
            llm=LLMConfig(
                temperature=0.6,
                timeout=300,
                retries=2
            ),

            # Evaluation (expensive FEA simulations)
            max_iterations=100,
            evaluator=EvaluatorConfig(
                timeout=600,  # 10 minutes per simulation
                max_retries=1,
                early_stopping=True,
                early_stopping_patience=5
            ),

            # Memory for learning design patterns
            database=DatabaseConfig(
                enable_memory=True
            )
        )

    def _structural_config(self) -> UnifiedEvolutionConfig:
        """
        Structural optimization configuration

        Focus on weight vs strength tradeoff
        """
        config = self._general_config()

        # Multi-objective: weight, strength, cost
        config.mo = MOConfig(
            enabled=True,
            objectives=["weight", "strength", "cost"],
            algorithm="nsga2",
            pareto_size=50
        )

        # Safety-critical: stricter adversarial
        config.adversarial.robustness_threshold = 0.95

        return config

    def _circuit_config(self) -> UnifiedEvolutionConfig:
        """
        Circuit design configuration

        Focus on power vs area tradeoff
        """
        config = self._general_config()

        # Multi-objective: power, area, performance
        config.mo = MOConfig(
            enabled=True,
            objectives=["power", "area", "performance"],
            algorithm="nsga2",
            pareto_size=50
        )

        # Circuit simulation faster than FEA
        config.max_iterations = 150
        config.evaluator.timeout = 300

        return config

    def _control_config(self) -> UnifiedEvolutionConfig:
        """
        Control systems configuration

        Focus on response vs stability
        """
        config = self._general_config()

        # Multi-objective: response_time, stability, robustness
        config.mo = MOConfig(
            enabled=True,
            objectives=["response_time", "stability", "robustness"],
            algorithm="nsga2",
            pareto_size=50
        )

        # Control simulation fast
        config.max_iterations = 200
        config.evaluator.timeout = 120

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
        Evaluate engineering design

        Args:
            solution: Design code or specification
            problem: Problem description
            constraints: Constraints (max_weight, min_safety_factor, etc.)

        Returns:
            Dictionary of engineering metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     bridge_design,
            ...     "Design lightweight bridge",
            ...     {"max_weight": 1000}
            ... )
            >>> print(metrics['safety_factor'])
        """
        # Parse design
        design = self._parse_design(solution)

        # Calculate metrics
        metrics = self._calculate_engineering_metrics(
            design,
            problem,
            constraints
        )

        return metrics

    def _parse_design(self, solution: str) -> Dict[str, Any]:
        """
        Parse engineering design from solution

        Args:
            solution: Design code or specification

        Returns:
            Design components (including the raw text for scoring)
        """
        import re

        design: Dict[str, Any] = {
            "source": solution or "",
            "parameters": {},
            "geometry": {},
            "materials": []
        }

        # Extract parameters
        params = re.findall(r'(\w+)\s*[=:]\s*([0-9]*\.?[0-9]+)', solution)
        if params:
            design["parameters"] = {name: float(value) for name, value in params}

        # Materials referenced by the design
        design["materials"] = [
            material
            for material in self.MATERIAL_SIGNALS
            if material in solution.lower()
        ]

        return design

    def _calculate_engineering_metrics(
        self,
        design: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate engineering metrics from the design specification

        Deterministic heuristic scorer (no FEA/SPICE solver required): safety
        margin is taken from the declared safety factor (or inferred from the
        margin/tolerance signals), and the remaining metrics are derived from the
        analysis, robustness and manufacturability signals present in the design,
        checked against the supplied constraints.

        Args:
            design: Design components from :meth:`_parse_design`
            problem: Problem description
            constraints: Constraints (max_weight, min_safety_factor, max_cost, ...)

        Returns:
            Dictionary of metrics (``safety_margin`` is a factor, not a ratio)
        """
        source = design.get("source", "")
        parameters = design.get("parameters", {})

        analysis = signal_coverage(source, self.ANALYSIS_SIGNALS)
        robustness = signal_coverage(source, self.ROBUSTNESS_SIGNALS)
        manufacturing = signal_coverage(source, self.MANUFACTURING_SIGNALS)
        thermal = signal_coverage(source, self.THERMAL_SIGNALS)
        structure = code_structure_score(source)

        # Safety factor: declared explicitly, else inferred from the signals
        declared_safety = None
        for key, value in parameters.items():
            if any(token in key.lower() for token in ("safety", "sf", "margin")):
                declared_safety = float(value)
                break

        if declared_safety is None:
            safety_margin = 1.0 + 1.5 * robustness + 0.5 * analysis
        else:
            safety_margin = max(0.0, declared_safety)

        min_safety = float((constraints or {}).get("min_safety_factor", 2.0))
        safety_compliance = clamp(safety_margin / min_safety if min_safety > 0 else 1.0)

        # Weight and cost: measured against their budgets when declared
        weight = self._budget_utilization(parameters, ("weight", "mass"), (constraints or {}).get("max_weight"))
        cost = self._budget_utilization(parameters, ("cost", "price", "budget"), (constraints or {}).get("max_cost"))

        performance = clamp(
            0.35 * analysis + 0.25 * robustness + 0.2 * structure + 0.2 * safety_compliance
        )
        reliability = clamp(0.3 + 0.4 * robustness + 0.3 * safety_compliance)
        efficiency = clamp(0.25 + 0.35 * analysis + 0.2 * (1.0 - weight) + 0.2 * (1.0 - cost))
        manufacturability = clamp(
            0.25 + 0.5 * manufacturing + 0.25 * saturating(len(design.get("materials", [])), 3)
        )
        thermal_performance = clamp(0.3 + 0.5 * thermal + 0.2 * analysis)

        return {
            "performance": performance,
            "safety_margin": safety_margin,   # safety factor (x), higher is better
            "cost": cost,                     # fraction of budget used (lower is better)
            "weight": weight,                 # fraction of weight budget (lower is better)
            "reliability": reliability,
            "efficiency": efficiency,
            "manufacturability": manufacturability,
            "thermal_performance": thermal_performance,
        }

    @staticmethod
    def _budget_utilization(
        parameters: Dict[str, float],
        keys: tuple,
        budget: Optional[float]
    ) -> float:
        """
        Fraction of a budget consumed by the matching design parameter

        Args:
            parameters: Parsed numeric design parameters
            keys: Parameter name fragments to look for
            budget: Budget for the quantity (None when unconstrained)

        Returns:
            Utilization in ``[0.0, 1.0]`` (0.5 when nothing is declared)
        """
        value = None
        for name, parameter in parameters.items():
            if any(key in name.lower() for key in keys):
                value = float(parameter)
                break

        if value is None:
            return 0.5  # unknown: neutral utilization

        if not budget:
            # No budget declared: normalize on a decade scale so it stays bounded
            return clamp(value / (value + 1000.0))

        return clamp(value / float(budget))

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_engineering_constraints(
        self,
        max_weight: Optional[float] = None,
        min_safety_factor: float = 2.0,
        max_cost: Optional[float] = None,
        material_constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get standard engineering constraints

        Args:
            max_weight: Maximum weight (kg)
            min_safety_factor: Minimum safety factor
            max_cost: Maximum cost ($)
            material_constraints: Allowed materials

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_engineering_constraints(
            ...     max_weight=1000,
            ...     min_safety_factor=2.5
            ... )
        """
        constraints = {
            "min_safety_factor": min_safety_factor
        }

        if max_weight is not None:
            constraints["max_weight"] = max_weight

        if max_cost is not None:
            constraints["max_cost"] = max_cost

        if material_constraints is not None:
            constraints["allowed_materials"] = material_constraints

        return constraints

    def validate_design(
        self,
        metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate engineering design

        Args:
            metrics: Engineering metrics
            constraints: Constraints

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_design(
            ...     metrics,
            ...     {"min_safety_factor": 2.0}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check safety factor
        if "min_safety_factor" in constraints:
            if metrics.get("safety_margin", 0) < constraints["min_safety_factor"]:
                violations.append(
                    f"Safety factor below minimum: {metrics['safety_margin']:.2f} < {constraints['min_safety_factor']:.2f}"
                )

        # Check weight
        if "max_weight" in constraints:
            if metrics.get("weight", 0) > 1.0:  # Normalized
                violations.append(
                    f"Weight exceeds maximum: {metrics['weight']:.2%}"
                )

        # Check cost
        if "max_cost" in constraints:
            if metrics.get("cost", 0) > 1.0:  # Normalized
                violations.append(
                    f"Cost exceeds budget: {metrics['cost']:.2%}"
                )

        return len(violations) == 0, violations

    def generate_safety_scenarios(
        self,
        design_type: str
    ) -> List[Dict[str, Any]]:
        """
        Generate safety testing scenarios

        Args:
            design_type: Type of design (structural, circuit, control)

        Returns:
            List of scenarios

        Example:
            >>> scenarios = optimizer.generate_safety_scenarios("structural")
        """
        if design_type == "structural":
            return [
                {"type": "load_exceedance", "description": "2x rated load"},
                {"type": "fatigue", "description": "10^6 cycles at max load"},
                {"type": "resonance", "description": "Natural frequency excitation"},
                {"type": "impact", "description": "Sudden impact load"}
            ]
        elif design_type == "circuit":
            return [
                {"type": "voltage_spike", "description": "2x rated voltage"},
                {"type": "thermal_overload", "description": "High current stress"},
                {"type": "em_interference", "description": "EM noise injection"}
            ]
        elif design_type == "control":
            return [
                {"type": "setpoint_ramp", "description": "Rapid setpoint changes"},
                {"type": "disturbance", "description": "External disturbances"},
                {"type": "delay", "description": "Sensor/actuator delays"}
            ]
        else:
            return []
