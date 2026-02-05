"""
PES Planner for OpenEvolve integration.

Creates intelligent evolution plans based on problem analysis,
historical patterns, and cost constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from .config import UnifiedEvolutionConfig


logger = logging.getLogger(__name__)


class EvolutionMode(Enum):
    """Evolution mode recommendations."""
    STANDARD = "standard"
    QD = "qd"
    MO = "mo"
    ADVERSARIAL = "adversarial"


@dataclass
class PlanPhase:
    """A phase in the evolution plan."""
    name: str
    description: str
    objectives: List[str]
    suggested_parameters: Dict[str, Any]
    budget_allocation: float  # Percentage of total budget
    convergence_criteria: Dict[str, Any]


@dataclass
class BudgetAllocation:
    """Budget allocation for different activities."""
    planning_budget: "CostBudget"
    evolution_budget: "CostBudget"
    verification_budget: "CostBudget"
    contingency_reserve: float  # Percentage


@dataclass
class CostBudget:
    """Budget for a specific activity."""
    max_cost: float
    max_tokens: int
    max_api_calls: int
    max_time_seconds: float


@dataclass
class EvolutionPlan:
    """
    PES Plan for OpenEvolve integration.
    
    Contains:
    - Strategy recommendations
    - Parameter suggestions
    - Budget allocation
    - Phase configuration
    - Convergence criteria
    """
    # Strategy
    recommended_mode: EvolutionMode
    reasoning: str
    
    # Parameters
    suggested_parameters: Dict[str, Any]
    parameter_reasoning: str
    
    # Budget
    budget_allocation: BudgetAllocation
    
    # Execution
    phases: List[PlanPhase]
    
    # Expected outcomes
    expected_iterations: int
    success_probability: float
    
    # Metadata
    problem_analysis: Dict[str, Any] = field(default_factory=dict)
    historical_patterns: Optional[List[Dict]] = None


@dataclass
class ProblemAnalysis:
    """Analysis of an evolution problem."""
    complexity: float  # 0.0 - 1.0
    estimated_difficulty: str  # "low", "medium", "high"
    key_challenges: List[str]
    recommended_strategies: List[str]
    estimated_tokens_per_eval: int
    requires_formal_verification: bool
    multi_objective: bool
    exploration_focused: bool


class ProblemAnalyzer:
    """Analyzes evolution problems to inform planning."""
    
    def analyze(self, problem: "EvolutionProblem") -> ProblemAnalysis:
        """Analyze problem characteristics."""
        
        # Estimate complexity
        complexity = self._estimate_complexity(problem)
        
        # Determine difficulty level
        if complexity < 0.33:
            difficulty = "low"
        elif complexity < 0.66:
            difficulty = "medium"
        else:
            difficulty = "high"
        
        # Identify challenges
        challenges = self._identify_challenges(problem)
        
        # Estimate token usage
        tokens_per_eval = self._estimate_tokens(problem)
        
        return ProblemAnalysis(
            complexity=complexity,
            estimated_difficulty=difficulty,
            key_challenges=challenges,
            recommended_strategies=self._recommend_strategies(problem, complexity),
            estimated_tokens_per_eval=tokens_per_eval,
            requires_formal_verification=self._needs_verification(problem),
            multi_objective=bool(problem.objectives and len(problem.objectives) > 1),
            exploration_focused=problem.exploration_focus
        )
    
    def _estimate_complexity(self, problem: "EvolutionProblem") -> float:
        """Estimate problem complexity."""
        complexity = 0.5
        
        # Description length
        desc_len = len(problem.description)
        if desc_len > 1000:
            complexity += 0.2
        elif desc_len > 500:
            complexity += 0.1
        
        # Code complexity
        if problem.code:
            lines = len(problem.code.split('\n'))
            complexity += min(lines / 500, 0.2)
        
        # Objectives
        if problem.objectives:
            complexity += min(len(problem.objectives) * 0.1, 0.3)
        
        # Constraints
        if problem.constraints:
            complexity += min(len(problem.constraints) * 0.05, 0.15)
        
        return min(complexity, 1.0)
    
    def _identify_challenges(self, problem: "EvolutionProblem") -> List[str]:
        """Identify key challenges for this problem."""
        challenges = []
        
        if problem.objectives and len(problem.objectives) > 1:
            challenges.append("multi_objective_tradeoffs")
        
        if problem.language not in ["python", "javascript"]:
            challenges.append("language_specific_optimizations")
        
        if problem.constraints:
            challenges.append("constraint_satisfaction")
        
        return challenges
    
    def _recommend_strategies(self, problem: "EvolutionProblem", complexity: float) -> List[str]:
        """Recommend evolution strategies."""
        strategies = []
        
        if problem.exploration_focus:
            strategies.append("qd_map_elites")
        
        if problem.objectives and len(problem.objectives) > 1:
            strategies.append("nsga2_mo")
        
        if complexity > 0.6:
            strategies.append("pes_enhanced_directed")
        
        strategies.append("standard")
        
        return strategies
    
    def _estimate_tokens(self, problem: "EvolutionProblem") -> int:
        """Estimate tokens per evaluation."""
        base_tokens = 500
        
        if problem.code:
            base_tokens += len(problem.code) // 4  # ~4 chars per token
        
        if problem.test_cases:
            base_tokens += len(problem.test_cases) * 100
        
        return base_tokens
    
    def _needs_verification(self, problem: "EvolutionProblem") -> bool:
        """Determine if formal verification is needed."""
        # Check for mathematical terms
        math_terms = ["theorem", "proof", "verify", "formal", "correctness"]
        description_lower = problem.description.lower()
        
        return any(term in description_lower for term in math_terms)


class CostEstimator:
    """Estimates evolution costs."""
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
    
    def estimate(
        self,
        analysis: ProblemAnalysis,
        strategy: EvolutionMode
    ) -> CostBudget:
        """Estimate budget requirements."""
        
        # Base calculations
        iterations = self._suggest_iterations(analysis)
        population = self._suggest_population(analysis)
        
        total_evaluations = iterations * population
        
        # Token estimates
        tokens_per_eval = analysis.estimated_tokens_per_eval
        total_tokens = total_evaluations * tokens_per_eval
        
        # Add verification cost if needed
        if analysis.requires_formal_verification:
            total_tokens += total_evaluations * 500  # Extra for verification
        
        # Cost calculation (example rates)
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)
        token_cost = (input_tokens / 1000 * 0.01) + (output_tokens / 1000 * 0.03)
        
        return CostBudget(
            max_cost=token_cost * 1.5,  # 50% buffer
            max_tokens=int(total_tokens * 1.2),
            max_api_calls=total_evaluations + 10,
            max_time_seconds=iterations * 30  # 30s per iteration
        )
    
    def _suggest_iterations(self, analysis: ProblemAnalysis) -> int:
        """Suggest number of iterations based on complexity."""
        if analysis.complexity > 0.7:
            return 150
        elif analysis.complexity > 0.4:
            return 100
        else:
            return 50
    
    def _suggest_population(self, analysis: ProblemAnalysis) -> int:
        """Suggest population size."""
        if analysis.multi_objective:
            return 100
        elif analysis.complexity > 0.6:
            return 75
        else:
            return 50


class PESPlanner:
    """
    PES Planner for OpenEvolve integration.
    
    Creates intelligent evolution plans that guide OpenEvolve
    toward optimal solutions efficiently.
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.problem_analyzer = ProblemAnalyzer()
        self.cost_estimator = CostEstimator(config)
    
    async def create_plan(
        self,
        problem: "EvolutionProblem"
    ) -> EvolutionPlan:
        """
        Create evolution plan for a problem.
        
        Steps:
        1. Analyze problem
        2. Query knowledge (if available)
        3. Estimate costs
        4. Build plan with parameters and budget
        """
        logger.info(f"Creating PES plan for problem: {problem.description[:100]}...")
        
        # 1. Analyze problem
        analysis = self.problem_analyzer.analyze(problem)
        logger.info(f"Problem complexity: {analysis.complexity:.2f}")
        
        # 2. Query knowledge (optional)
        historical_patterns = None
        if self.config.pes_config.use_historical_patterns:
            historical_patterns = await self._query_knowledge(analysis)
        
        # 3. Select strategy
        recommended_mode = self._select_mode(analysis)
        
        # 4. Estimate costs
        cost_estimate = self.cost_estimator.estimate(analysis, recommended_mode)
        
        # 5. Build budget allocation
        budget_allocation = self._allocate_budget(cost_estimate)
        
        # 6. Suggest parameters
        suggested_params = self._suggest_parameters(analysis, recommended_mode)
        
        # 7. Create phases
        phases = self._create_phases(analysis, recommended_mode)
        
        return EvolutionPlan(
            recommended_mode=recommended_mode,
            reasoning=f"Selected {recommended_mode.value} based on complexity {analysis.complexity:.2f}",
            suggested_parameters=suggested_params,
            parameter_reasoning=self._explain_parameters(suggested_params, analysis),
            budget_allocation=budget_allocation,
            phases=phases,
            expected_iterations=suggested_params.get("max_iterations", 100),
            success_probability=self._estimate_success_probability(analysis),
            problem_analysis=analysis.__dict__,
            historical_patterns=historical_patterns
        )
    
    def _select_mode(self, analysis: ProblemAnalysis) -> EvolutionMode:
        """Select evolution mode based on analysis."""
        if analysis.multi_objective:
            return EvolutionMode.MO
        elif analysis.exploration_focused:
            return EvolutionMode.QD
        elif analysis.complexity > 0.7:
            return EvolutionMode.ADVERSARIAL
        else:
            return EvolutionMode.STANDARD
    
    def _allocate_budget(self, cost_estimate: CostBudget) -> BudgetAllocation:
        """Allocate budget across activities."""
        cfg = self.config.budget_config
        
        total_cost = cost_estimate.max_cost
        
        return BudgetAllocation(
            planning_budget=CostBudget(
                max_cost=total_cost * cfg.planning_budget_ratio,
                max_tokens=int(cost_estimate.max_tokens * cfg.planning_budget_ratio),
                max_api_calls=int(cost_estimate.max_api_calls * cfg.planning_budget_ratio),
                max_time_seconds=cost_estimate.max_time_seconds * cfg.planning_budget_ratio
            ),
            evolution_budget=CostBudget(
                max_cost=total_cost * cfg.evolution_budget_ratio,
                max_tokens=int(cost_estimate.max_tokens * cfg.evolution_budget_ratio),
                max_api_calls=int(cost_estimate.max_api_calls * cfg.evolution_budget_ratio),
                max_time_seconds=cost_estimate.max_time_seconds * cfg.evolution_budget_ratio
            ),
            verification_budget=CostBudget(
                max_cost=total_cost * cfg.verification_budget_ratio,
                max_tokens=int(cost_estimate.max_tokens * cfg.verification_budget_ratio),
                max_api_calls=int(cost_estimate.max_api_calls * cfg.verification_budget_ratio),
                max_time_seconds=cost_estimate.max_time_seconds * cfg.verification_budget_ratio
            ),
            contingency_reserve=cfg.contingency_reserve_ratio
        )
    
    def _suggest_parameters(
        self,
        analysis: ProblemAnalysis,
        mode: EvolutionMode
    ) -> Dict[str, Any]:
        """Suggest evolution parameters."""
        params = {}
        
        # Iterations based on complexity
        if analysis.complexity > 0.7:
            params["max_iterations"] = 150
        elif analysis.complexity > 0.4:
            params["max_iterations"] = 100
        else:
            params["max_iterations"] = 50
        
        # Population based on mode
        if mode == EvolutionMode.MO:
            params["population_size"] = 100
        elif analysis.complexity > 0.6:
            params["population_size"] = 75
        else:
            params["population_size"] = 50
        
        # Mutation rate
        if analysis.exploration_focused:
            params["mutation_rate"] = 0.15
        else:
            params["mutation_rate"] = 0.1
        
        # Elitism
        params["elitism"] = True
        params["elite_ratio"] = 0.1
        
        # Diversity
        if mode == EvolutionMode.QD:
            params["diversity_maintenance"] = True
            params["novelty_threshold"] = 0.1
        
        return params
    
    def _explain_parameters(self, params: Dict[str, Any], analysis: ProblemAnalysis) -> str:
        """Generate explanation for parameter choices."""
        explanations = []
        
        if "max_iterations" in params:
            explanations.append(f"Iterations set to {params['max_iterations']} for {analysis.estimated_difficulty} complexity")
        
        if "population_size" in params:
            explanations.append(f"Population size {params['population_size']} balances exploration and cost")
        
        if analysis.exploration_focused:
            explanations.append("Higher mutation rate for exploration focus")
        
        return "; ".join(explanations)
    
    def _create_phases(
        self,
        analysis: ProblemAnalysis,
        mode: EvolutionMode
    ) -> List[PlanPhase]:
        """Create evolution phases."""
        phases = []
        
        # Phase 1: Initial exploration
        phases.append(PlanPhase(
            name="exploration",
            description="Initial exploration of solution space",
            objectives=["diversity", "coverage"],
            suggested_parameters={"mutation_rate": 0.15, "exploration_bonus": 0.1},
            budget_allocation=0.3,
            convergence_criteria={"min_diversity": 0.5}
        ))
        
        # Phase 2: Exploitation
        phases.append(PlanPhase(
            name="exploitation",
            description="Refine promising solutions",
            objectives=["fitness", "convergence"],
            suggested_parameters={"mutation_rate": 0.08, "selection_pressure": 1.5},
            budget_allocation=0.5,
            convergence_criteria={"fitness_improvement": 0.01}
        ))
        
        # Phase 3: Final polish (if budget allows)
        if mode != EvolutionMode.QD:
            phases.append(PlanPhase(
                name="polish",
                description="Final solution optimization",
                objectives=["local_optimization"],
                suggested_parameters={"mutation_rate": 0.05, "fine_tuning": True},
                budget_allocation=0.2,
                convergence_criteria={"plateau_detection": True}
            ))
        
        return phases
    
    def _estimate_success_probability(self, analysis: ProblemAnalysis) -> float:
        """Estimate probability of successful evolution."""
        base_prob = 0.8
        
        # Adjust for complexity
        base_prob -= analysis.complexity * 0.2
        
        # Adjust for challenges
        base_prob -= len(analysis.key_challenges) * 0.05
        
        return max(base_prob, 0.4)
    
    async def _query_knowledge(self, analysis: ProblemAnalysis) -> Optional[List[Dict]]:
        """Query knowledge base for similar problems."""
        # This would integrate with knowledge_engine
        # For now, return None
        return None
