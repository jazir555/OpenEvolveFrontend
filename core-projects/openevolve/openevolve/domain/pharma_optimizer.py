"""
Pharma Domain Optimizer
Specialized optimizer for pharmaceutical problems

Problems:
- Molecular optimization (property optimization)
- Drug design (target binding)
- Clinical trial design (patient stratification)

Best System: OpenEvolve (QD mode)
Why: Need to explore chemical space diversity (many local optima)

Configuration:
- Evaluation cost: "expensive"
- Mode: QD (Quality Diversity)
- QD grid resolution: 20
- Feature dimensions: [binding_affinity, solubility, toxicity, synth_score]
- Archive size: 10,000

Metrics:
- Binding affinity
- ADMET properties
- Synthetic accessibility
- Drug-likeness

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType, QDConfig, LLMConfig, EvaluatorConfig, DatabaseConfig, MOConfig
from .base import DomainOptimizer

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based molecular optimization
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class PharmaOptimizer(DomainOptimizer):
    """
    Pharma domain optimizer

    Specialized for:
    - Molecular optimization
    - Drug design
    - Clinical trial design

    Example:
        >>> optimizer = PharmaOptimizer(sub_domain="molecular")
        >>> result = await optimizer.optimize(
        ...     "Optimize molecule for high binding affinity and low toxicity",
        ...     constraints={"max_toxicity": 0.3, "min_solubility": 0.5}
        ... )
        >>> print(result['domain_metrics']['binding_affinity'])
    """

    domain_name = "pharma"

    def __init__(self, sub_domain: str = "general", use_adaptive_mdap: bool = True):
        """
        Initialize pharma optimizer

        Args:
            sub_domain: One of 'general', 'molecular', 'drug_design', 'clinical_trial'
            use_adaptive_mdap: Whether to use Adaptive MDAP for complexity-based allocation
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "molecular": self._molecular_config(),
            "drug_design": self._drug_design_config(),
            "clinical_trial": self._clinical_trial_config()
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
        """OpenEvolve QD for exploring chemical space"""
        return "openevolve"

    def get_recommended_mode(self) -> str:
        """QD mode for diverse molecule discovery"""
        return "qd"

    def get_domain_metrics(self) -> List[str]:
        """Pharma-specific metrics"""
        return [
            "binding_affinity",
            "solubility",
            "toxicity",
            "synthetic_accessibility",
            "drug_likeness",
            "bioavailability",
            "permeability",
            "metabolic_stability",
            "hERG_blockade",  # Cardiac safety
            "cyp_inhibition"  # Drug-drug interaction
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
            id=f"pharma_{hash(problem) % 10000}",
            description=problem,
            domain="pharma",
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

        # Adjust QD archive size and iterations based on complexity and strategy
        if allocation.strategy == "DIRECT":
            # Simple problems: smaller archive, fewer iterations
            config.max_iterations = min(80, config.max_iterations)
            config.qd.archive_size = min(5000, config.qd.archive_size)
            config.database.archive_size = min(5000, config.database.archive_size)
            config.evaluator.parallel_evaluations = max(4, config.evaluator.parallel_evaluations - 4)
        elif allocation.strategy == "MDAP_LIGHT":
            # Light: moderate archive and iterations
            config.max_iterations = min(120, config.max_iterations)
            config.qd.archive_size = min(7500, config.qd.archive_size)
            config.database.archive_size = min(7500, config.database.archive_size)
            config.evaluator.parallel_evaluations = max(6, config.evaluator.parallel_evaluations - 2)
        elif allocation.strategy == "MDAP_MEDIUM":
            # Medium: standard config (no change)
            pass
        elif allocation.strategy == "MAKER_FULL":
            # Full: larger archive, more iterations, enable MO
            config.max_iterations = max(180, config.max_iterations)
            config.qd.archive_size = max(12000, config.qd.archive_size)
            config.database.archive_size = max(12000, config.database.archive_size)
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True
            config.evaluator.parallel_evaluations = min(16, config.evaluator.parallel_evaluations + 4)
        elif allocation.strategy == "MAKER_ULTRA":
            # Ultra: maximum archive (up to 15000), most iterations, full QD + MO
            config.max_iterations = max(220, config.max_iterations)
            config.qd.archive_size = max(15000, config.qd.archive_size)
            config.database.archive_size = max(15000, config.database.archive_size)
            if hasattr(config, 'mo') and config.mo:
                config.mo.enabled = True
            config.evaluator.parallel_evaluations = min(20, config.evaluator.parallel_evaluations + 6)

        # Adjust evaluation timeout based on complexity
        if score > 0.7:
            # High complexity: longer timeouts for docking
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
        """Get default pharma configuration"""
        return self._general_config()

    # ========================================================================
    # SUB-DOMAIN CONFIGURATIONS
    # ========================================================================

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General pharma configuration

        Uses QD mode to explore diverse chemical space
        """
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.PHARMA,

            # Evolution mode
            evolution_mode=EvolutionMode.QD,

            # QD configuration (for diverse molecules)
            qd=QDConfig(
                enabled=True,
                grid_resolution=20,
                feature_dimensions=["binding_affinity", "solubility", "toxicity", "synth_score"],
                archive_size=10000
            ),

            # LLM configuration
            llm=LLMConfig(
                temperature=0.8,  # High creativity for novel molecules
                timeout=180,
                retries=2
            ),

            # Evaluation (expensive docking/simulations)
            max_iterations=150,
            evaluator=EvaluatorConfig(
                timeout=300,  # 5 minutes per molecule (docking + ADMET)
                max_retries=1,
                early_stopping=True,
                early_stopping_patience=8,
                parallel_evaluations=10  # Parallel docking
            ),

            # Large archive for diversity
            database=DatabaseConfig(
                population_size=200,
                archive_size=10000,
                diversity_metric="tanimoto"  # Chemical similarity
            )
        )

    def _molecular_config(self) -> UnifiedEvolutionConfig:
        """
        Molecular optimization configuration

        Focus on multi-property optimization
        """
        config = self._general_config()

        # Multi-objective: affinity, solubility, low toxicity
        config.mo = MOConfig(
            enabled=True,
            objectives=["binding_affinity", "solubility", "safety"],
            algorithm="nsga2",
            pareto_size=200
        )

        # Higher QD resolution for fine-grained exploration
        config.qd.grid_resolution = 25

        return config

    def _drug_design_config(self) -> UnifiedEvolutionConfig:
        """
        Drug design configuration

        Focus on target binding
        """
        config = self._general_config()

        # More iterations for thorough search
        config.max_iterations = 200

        # Even larger archive
        config.qd.archive_size = 15000

        # Feature dimensions focused on binding
        config.qd.feature_dimensions = [
            "binding_affinity",
            "selectivity",
            "oral_bioavailability",
            "synth_score"
        ]

        return config

    def _clinical_trial_config(self) -> UnifiedEvolutionConfig:
        """
        Clinical trial design configuration

        Focus on patient stratification
        """
        config = self._general_config()

        # Clinical trial design less about QD, more about optimization
        config.evolution_mode = EvolutionMode.STANDARD
        config.qd.enabled = False

        # Lower temperature (more systematic)
        config.llm.temperature = 0.5

        # Fewer iterations
        config.max_iterations = 80

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
        Evaluate pharma solution

        Args:
            solution: Molecule (SMILES) or clinical trial design
            problem: Problem description
            constraints: Constraints (max_toxicity, min_solubility, etc.)

        Returns:
            Dictionary of pharma metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     "CCO.CCN",  # SMILES string
            ...     "Optimize for high binding",
            ...     {"max_toxicity": 0.3}
            ... )
            >>> print(metrics['binding_affinity'])
        """
        # Parse molecule
        molecule = self._parse_molecule(solution)

        # Calculate metrics
        metrics = self._calculate_pharma_metrics(
            molecule,
            problem,
            constraints
        )

        return metrics

    def _parse_molecule(self, solution: str) -> Dict[str, Any]:
        """
        Parse molecule from solution

        Args:
            solution: SMILES string or molecular structure

        Returns:
            Molecular properties
        """
        # Placeholder: Parse SMILES
        molecule = {
            "smiles": "",
            "molecular_weight": 0,
            "logp": 0,
            "hbd": 0,  # Hydrogen bond donors
            "hba": 0   # Hydrogen bond acceptors
        }

        # Extract SMILES
        import re
        smiles_match = re.search(r'(SMILES:?\s*)?([A-Za-z0-9@+\-\[\]\(\)\\=#$]+)', solution)
        if smiles_match:
            molecule["smiles"] = smiles_match.group(2)

        return molecule

    def _calculate_pharma_metrics(
        self,
        molecule: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate pharma metrics

        Args:
            molecule: Molecular properties
            problem: Problem description
            constraints: Constraints

        Returns:
            Dictionary of metrics

        Note: This is a placeholder. In production, integrate with:
        - Molecular docking software (AutoDock, Glide)
        - ADMET predictors (QikProp, ADMET Predictor)
        - Quantum chemistry packages (Psi4, RDKit)
        """
        # Placeholder metrics (normalized 0-1, higher is better unless noted)
        metrics = {
            "binding_affinity": 0.85,         # 85% of ideal binding
            "solubility": 0.70,              # 70% solubility
            "toxicity": 0.30,                # 30% toxicity (lower is better)
            "synthetic_accessibility": 0.75, # 75% ease of synthesis
            "drug_likeness": 0.80,           # 80% drug-like
            "bioavailability": 0.65,         # 65% oral bioavailability
            "permeability": 0.72,            # 72% Caco-2 permeability
            "metabolic_stability": 0.68,     # 68% metabolic stability
            "hERG_blockade": 0.15,           # 15% hERG blockade (lower is better)
            "cyp_inhibition": 0.25           # 25% CYP inhibition (lower is better)
        }

        # In production, would run:
        # 1. Molecular docking (binding affinity)
        # 2. ADMET predictions
        # 3. Quantum chemistry calculations
        # 4. Synthetic accessibility scoring

        return metrics

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_pharma_constraints(
        self,
        max_toxicity: float = 0.3,
        min_solubility: float = 0.5,
        min_binding_affinity: float = 0.7,
        max_molecular_weight: float = 500
    ) -> Dict[str, Any]:
        """
        Get standard pharma constraints

        Args:
            max_toxicity: Maximum toxicity (0-1, lower is better)
            min_solubility: Minimum solubility (0-1)
            min_binding_affinity: Minimum binding affinity (0-1)
            max_molecular_weight: Maximum molecular weight (Da)

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_pharma_constraints(
            ...     max_toxicity=0.2,
            ...     min_binding_affinity=0.8
            ... )
        """
        return {
            "max_toxicity": max_toxicity,
            "min_solubility": min_solubility,
            "min_binding_affinity": min_binding_affinity,
            "max_molecular_weight": max_molecular_weight
        }

    def validate_molecule(
        self,
        metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate molecule against constraints

        Args:
            metrics: Pharma metrics
            constraints: Constraints

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_molecule(
            ...     metrics,
            ...     {"max_toxicity": 0.3}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check toxicity
        if "max_toxicity" in constraints:
            if metrics.get("toxicity", 0) > constraints["max_toxicity"]:
                violations.append(
                    f"Toxicity too high: {metrics['toxicity']:.2%} > {constraints['max_toxicity']:.2%}"
                )

        # Check solubility
        if "min_solubility" in constraints:
            if metrics.get("solubility", 0) < constraints["min_solubility"]:
                violations.append(
                    f"Solubility too low: {metrics['solubility']:.2%} < {constraints['min_solubility']:.2%}"
                )

        # Check binding affinity
        if "min_binding_affinity" in constraints:
            if metrics.get("binding_affinity", 0) < constraints["min_binding_affinity"]:
                violations.append(
                    f"Binding affinity too low: {metrics['binding_affinity']:.2%} < {constraints['min_binding_affinity']:.2%}"
                )

        # Lipinski's Rule of 5
        # (MW < 500, LogP < 5, HBD < 5, HBA < 10)
        # These would be checked in the molecular parsing

        return len(violations) == 0, violations

    def calculate_drug_likeness(
        self,
        molecule: Dict[str, Any]
    ) -> float:
        """
        Calculate drug-likeness score (Lipinski's Rule of 5)

        Args:
            molecule: Molecular properties

        Returns:
            Drug-likeness score (0-1)

        Example:
            >>> score = optimizer.calculate_drug_likeness(molecule)
        """
        # Lipinski's Rule of 5:
        # - Molecular weight < 500 Da
        # - LogP < 5
        # - Hydrogen bond donors < 5
        # - Hydrogen bond acceptors < 10

        score = 1.0

        mw = molecule.get("molecular_weight", 0)
        if mw > 500:
            score -= 0.25

        logp = molecule.get("logp", 0)
        if logp > 5:
            score -= 0.25

        hbd = molecule.get("hbd", 0)
        if hbd > 5:
            score -= 0.25

        hba = molecule.get("hba", 0)
        if hba > 10:
            score -= 0.25

        return max(0.0, score)

    def suggest_lead_optimization(
        self,
        current_molecule: Dict[str, Any],
        target_properties: Dict[str, float]
    ) -> List[str]:
        """
        Suggest lead optimization strategies

        Args:
            current_molecule: Current molecular properties
            target_properties: Target property values

        Returns:
            List of optimization suggestions

        Example:
            >>> suggestions = optimizer.suggest_lead_optimization(
            ...     molecule,
            ...     {"binding_affinity": 0.9, "solubility": 0.8}
            ... )
        """
        suggestions = []

        # Compare current to target
        for prop, target in target_properties.items():
            current = current_molecule.get(prop, 0)
            if current < target:
                if prop == "binding_affinity":
                    suggestions.append("Add hydrogen bond donors/acceptors to improve binding")
                elif prop == "solubility":
                    suggestions.append("Add polar groups to improve solubility")
                elif prop == "bioavailability":
                    suggestions.append("Reduce molecular weight or improve permeability")
                elif prop == "metabolic_stability":
                    suggestions.append("Block metabolic soft spots")

        return suggestions
