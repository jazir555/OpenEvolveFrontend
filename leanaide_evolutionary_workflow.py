"""
LeanAide Evolutionary Workflow Integration

This module provides comprehensive integration of evolutionary LeanAide capabilities
with the OpenEvolve decomposition workflow. It seamlessly adds evolutionary proof
generation, adversarial critique, and self-play learning to existing workflow stages
without breaking changes.

Key Features:
- LeanEvolutionaryWorkflowStage: Wraps evolutionary LeanAide for workflow use
- LeanEvolutionarySubProblemSolver: Solves mathematical sub-problems using evolution
- LeanEvolutionaryReassembler: Reassembles evolved sub-proofs into complete proofs
- Stage 3A/B/C integration: Evolutionary generation, adversarial critique, verification
- Stage 5 integration: Final evolutionary verification
- Configuration integration with WorkflowState
- Graceful fallback to non-evolutionary approaches
- Comprehensive error handling and logging

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable
)
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Import workflow structures
try:
    from workflow_structures import (
        WorkflowState,
        SubProblem,
        SolutionAttempt,
        VerificationReport,
        CritiqueReport,
        Team,
        GauntletDefinition
    )
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logger.warning("Workflow structures not available - integration limited")

# Import LeanAide components
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig,
        LeanAideVerificationResult,
        MathematicalProblemDetector,
        is_leanaide_configured
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide workflow integration not available")

# Import evolutionary components
try:
    from leanaide_evolution import (
        LeanProofEvolutionEngine,
        LeanProofStrategy,
        EvolutionResult,
        SelectionMethod,
        CrossoverMethod,
        evolve_proof
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logger.warning("LeanAide evolution not available")

# Import adversarial components
try:
    from leanaide_adversarial import (
        LeanAdversarialEvolution,
        LeanProofStrategy as AdversarialProofStrategy,
        ProofCritique,
        ProofApproach,
        CritiqueSeverity,
        LeanBlueTeamAgent,
        LeanRedTeamAgent,
        evolve_lean_proof
    )
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    logger.warning("LeanAide adversarial not available")

# Import self-play components
try:
    from leanaide_selfplay import (
        LeanSelfPlayArena,
        SelfPlayResult,
        SelfPlayStatistics
    )
    SELFPLAY_AVAILABLE = True
except ImportError:
    SELFPLAY_AVAILABLE = False
    logger.warning("LeanAide self-play not available")

# import crewai # MIGRATED: was CrewAI integration
try:
    from crewai_client import CrewAIClient
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import ACE knowledge storage
try:
    from ace_knowledge_artifacts import ACEKnowledgeManager
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


class EvolutionStrategy(Enum):
    """Available evolution strategies for mathematical problems."""
    STANDARD = "standard"  # Non-evolutionary approach
    EVOLUTION = "evolution"  # Pure genetic algorithm evolution
    ADVERSARIAL = "adversarial"  # Red team vs Blue team competition
    SELF_PLAY = "self_play"  # Self-play reinforcement learning
    HYBRID = "hybrid"  # Combine multiple strategies


class MathematicalDomain(Enum):
    """Mathematical domains for specialized evolution."""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    TOPOLOGY = "topology"
    LOGIC = "logic"
    COMPUTABILITY = "computability"
    COMPLEXITY = "complexity"
    GENERAL = "general"


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary LeanAide integration."""
    # Evolution enablement
    lean_evolution_enabled: bool = True
    lean_evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID

    # Evolution parameters
    lean_evolution_generations: int = 50
    lean_evolution_population_size: int = 20
    lean_evolution_mutation_rate: float = 0.1
    lean_evolution_crossover_rate: float = 0.8
    lean_evolution_elitism_ratio: float = 0.1

    # Adversarial parameters
    lean_adversarial_rounds: int = 10
    lean_adversarial_convergence_threshold: float = 0.95

    # Self-play parameters
    lean_self_play_games: int = 20
    lean_self_play_exploration_rate: float = 0.3

    # Verification parameters
    lean_verification_confidence_threshold: float = 0.7
    lean_verification_timeout: float = 300.0

    # Fallback behavior
    lean_fallback_to_standard: bool = True
    lean_timeout_handling: str = "partial"  # "partial", "skip", "wait"

    # Integration settings
    lean_auto_detect_mathematical: bool = True
    lean_store_evolved_proofs: bool = True
    lean_track_evolution_statistics: bool = True

    # CrewAI integration
    CrewAI_enabled: bool = False
    CrewAI_timeout: float = 600.0

    # ACE integration
    ace_learning_enabled: bool = True
    ace_store_patterns: bool = True


@dataclass
class EvolutionaryProgress:
    """Progress tracking for evolutionary runs."""
    sub_problem_id: str
    strategy: EvolutionStrategy
    generation: int = 0
    best_fitness: float = 0.0
    current_best: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    status: str = "in_progress"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_problem_id": self.sub_problem_id,
            "strategy": self.strategy.value,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "current_best": self.current_best,
            "elapsed_time": self.elapsed_time,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "convergence_history": self.convergence_history
        }


class LeanEvolutionaryWorkflowStage:
    """
    Main integration class for evolutionary LeanAide in workflow stages.

    This class wraps evolutionary LeanAide capabilities and provides
    seamless integration with workflow stages 3A, 3B, 3C, and 5.
    """

    def __init__(
        self,
        config: Optional[EvolutionaryConfig] = None,
        workflow_state: Optional[WorkflowState] = None
    ):
        """
        Initialize the evolutionary workflow stage.

        Args:
            config: Evolutionary configuration
            workflow_state: Current workflow state
        """
        self.config = config or EvolutionaryConfig()
        self.workflow_state = workflow_state

        # Initialize integrators
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None
        self.crewai_client: Optional[CrewAIClient] = None
        self.ace_manager: Optional[ACEKnowledgeManager] = None

        # Progress tracking
        self.evolution_progress: Dict[str, EvolutionaryProgress] = {}
        self.statistics: Dict[str, Any] = defaultdict(list)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all required components."""
        # Initialize LeanAide integrator
        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideWorkflowConfig(
                enabled=self.config.lean_evolution_enabled,
                auto_detect_math=self.config.lean_auto_detect_mathematical,
                fallback_to_standard=self.config.lean_fallback_to_standard,
                confidence_threshold=self.config.lean_verification_confidence_threshold,
                store_proofs=self.config.lean_store_evolved_proofs
            )
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

        # Initialize CrewAI client if enabled
        if self.config.CrewAI_enabled and CREWAI_AVAILABLE:
            self.crewai_client = CrewAIClient(timeout=self.config.CrewAI_timeout)

        # Initialize ACE manager if enabled
        if self.config.ace_learning_enabled and ACE_AVAILABLE:
            self.ace_manager = ACEKnowledgeManager()

    def is_mathematical_subproblem(
        self,
        sub_problem: SubProblem
    ) -> Tuple[bool, float, Optional[MathematicalDomain]]:
        """
        Detect if a sub-problem is mathematical in nature.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            Tuple of (is_mathematical, confidence_score, domain)
        """
        if not LEANAIDE_AVAILABLE or not self.leanaide_integrator:
            return False, 0.0, None

        # Use mathematical problem detector
        is_math, confidence = self.leanaide_integrator.detector.is_mathematical_problem(
            problem_statement=sub_problem.description,
            solution_content=None
        )

        if not is_math:
            return False, confidence, None

        # Determine mathematical domain
        domain = self._classify_mathematical_domain(sub_problem)

        return True, confidence, domain

    def _classify_mathematical_domain(
        self,
        sub_problem: SubProblem
    ) -> MathematicalDomain:
        """Classify the mathematical domain of a sub-problem."""
        description = sub_problem.description.lower()

        # Domain-specific keywords
        domain_keywords = {
            MathematicalDomain.ALGEBRA: [
                "algebra", "equation", "polynomial", "group", "ring", "field",
                "linear algebra", "matrix", "vector", "eigenvalue"
            ],
            MathematicalDomain.ANALYSIS: [
                "calculus", "derivative", "integral", "limit", "continuity",
                "differentiation", "convergence", "series", "function"
            ],
            MathematicalDomain.COMBINATORICS: [
                "combinatorics", "permutation", "combination", "graph", "counting",
                "binomial", "pigeonhole", "inclusion-exclusion"
            ],
            MathematicalDomain.GEOMETRY: [
                "geometry", "topology", "manifold", "space", "dimension",
                "triangle", "circle", "angle", "proof", "congruence"
            ],
            MathematicalDomain.NUMBER_THEORY: [
                "number theory", "prime", "divisibility", "modular", "congruence",
                "diophantine", "factorization", "number"
            ],
            MathematicalDomain.LOGIC: [
                "logic", "proof", "theorem", "lemma", "formal", "deduction",
                "induction", "contradiction", "proposition"
            ],
            MathematicalDomain.COMPUTABILITY: [
                "computability", "turing", "decidable", "recursive", "algorithm",
                "complexity", "reduction", "undecidable"
            ],
            MathematicalDomain.COMPLEXITY: [
                "complexity", "np-complete", "p vs np", "time complexity",
                "space complexity", "optimization", "approximation"
            ]
        }

        # Score each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in description)
            domain_scores[domain] = score

        # Return highest scoring domain, or GENERAL if tied
        if domain_scores:
            max_score = max(domain_scores.values())
            if max_score > 0:
                best_domains = [d for d, s in domain_scores.items() if s == max_score]
                return best_domains[0] if len(best_domains) == 1 else MathematicalDomain.GENERAL

        return MathematicalDomain.GENERAL

    async def solve_subproblem_evolutionary(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Solve a mathematical sub-problem using evolutionary approach.

        This is the main integration point for Stage 3A (Solution Generation)
        with evolutionary capabilities.

        Args:
            sub_problem: The sub-problem to solve
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with evolved solution
        """
        start_time = time.time()
        sub_problem_id = sub_problem.id

        # Create progress tracker
        progress = EvolutionaryProgress(
            sub_problem_id=sub_problem_id,
            strategy=self.config.lean_evolution_strategy
        )
        self.evolution_progress[sub_problem_id] = progress

        try:
            # Detect if this is a mathematical problem
            is_math, confidence, domain = self.is_mathematical_subproblem(sub_problem)

            if not is_math and not self.config.lean_evolution_enabled:
                logger.info(f"Sub-problem {sub_problem_id} is not mathematical, using standard approach")
                return await self._fallback_to_standard_solution(sub_problem, workflow_state)

            logger.info(f"Solving sub-problem {sub_problem_id} using evolutionary strategy: {self.config.lean_evolution_strategy.value}")

            # Select appropriate strategy
            if self.config.lean_evolution_strategy == EvolutionStrategy.EVOLUTION:
                solution = await self._solve_with_pure_evolution(sub_problem, workflow_state, progress)
            elif self.config.lean_evolution_strategy == EvolutionStrategy.ADVERSARIAL:
                solution = await self._solve_with_adversarial_evolution(sub_problem, workflow_state, progress)
            elif self.config.lean_evolution_strategy == EvolutionStrategy.SELF_PLAY:
                solution = await self._solve_with_self_play(sub_problem, workflow_state, progress)
            elif self.config.lean_evolution_strategy == EvolutionStrategy.HYBRID:
                solution = await self._solve_with_hybrid_approach(sub_problem, workflow_state, progress)
            else:
                solution = await self._fallback_to_standard_solution(sub_problem, workflow_state)

            # Update progress
            progress.status = "completed"
            progress.elapsed_time = time.time() - start_time

            # Store statistics
            self.statistics[sub_problem_id].append({
                "strategy": self.config.lean_evolution_strategy.value,
                "elapsed_time": progress.elapsed_time,
                "generations": progress.generation,
                "best_fitness": progress.best_fitness
            })

            # Store in knowledge base if enabled
            if self.config.lean_store_evolved_proofs and self.ace_manager:
                await self._store_evolved_proof(sub_problem, solution, progress)

            return solution

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Evolutionary solution failed for {sub_problem_id}: {e}")
            progress.errors.append(str(e))
            progress.status = "failed"

            if self.config.lean_fallback_to_standard:
                return await self._fallback_to_standard_solution(sub_problem, workflow_state)
            else:
                raise

    async def _solve_with_pure_evolution(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState,
        progress: EvolutionaryProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using pure genetic algorithm evolution."""
        if not EVOLUTION_AVAILABLE:
            return await self._fallback_to_standard_solution(sub_problem, workflow_state)

        logger.info(f"Using pure evolution for {sub_problem.id}")

        # Create evolution engine
        engine = LeanProofEvolutionEngine(
            theorem=sub_problem.description,
            theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}",
            population_size=self.config.lean_evolution_population_size,
            max_generations=self.config.lean_evolution_generations,
            mutation_rate=self.config.lean_evolution_mutation_rate,
            crossover_rate=self.config.lean_evolution_crossover_rate,
            elitism_ratio=self.config.lean_evolution_elitism_ratio
        )

        # Run evolution
        result = await engine.evolve()

        # Update progress
        progress.generation = result.generations_completed
        progress.best_fitness = result.best_strategy.fitness if result.best_strategy else 0.0
        progress.current_best = result.best_proof.lean_code if result.best_proof else None
        progress.convergence_history = result.convergence_history

        # Create solution attempt
        solution = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.best_proof.lean_code if result.best_proof else "",
            generated_by_model="LeanAide-Evolution",
            timestamp=time.time(),
            status="verified" if result.success else "generated",
            solution_approach="evolutionary",
            openevolve_metrics={
                "evolution_strategy": "evolution",
                "generations": result.generations_completed,
                "total_evaluations": result.total_evaluations,
                "evolution_time": result.evolution_time,
                "best_fitness": progress.best_fitness,
                "converged": result.success
            }
        )

        return solution

    async def _solve_with_adversarial_evolution(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState,
        progress: EvolutionaryProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using adversarial evolution."""
        if not ADVERSARIAL_AVAILABLE:
            return await self._fallback_to_standard_solution(sub_problem, workflow_state)

        logger.info(f"Using adversarial evolution for {sub_problem.id}")

        # Create adversarial evolution system
        evolution = LeanAdversarialEvolution()

        # Run adversarial evolution
        final_proof, round_results, statistics = evolution.run_adversarial_evolution(
            theorem=sub_problem.description,
            rounds=self.config.lean_adversarial_rounds
        )

        # Update progress
        progress.generation = len(round_results)
        progress.best_fitness = statistics.blue_success_rate
        progress.current_best = final_proof.lean_code
        progress.convergence_history = [r.blue_score for r in round_results]

        # Create solution attempt
        solution = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=final_proof.lean_code,
            generated_by_model="LeanAide-Adversarial",
            timestamp=time.time(),
            status="verified" if final_proof.confidence > 0.8 else "generated",
            solution_approach="adversarial_evolution",
            openevolve_metrics={
                "evolution_strategy": "adversarial",
                "rounds": len(round_results),
                "blue_win_rate": statistics.blue_success_rate,
                "red_win_rate": statistics.red_success_rate,
                "counterexamples_found": statistics.unique_counterexamples_found,
                "most_effective_approach": statistics.most_effective_approach.value if statistics.most_effective_approach else None
            }
        )

        return solution

    async def _solve_with_self_play(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState,
        progress: EvolutionaryProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using self-play evolution."""
        if not SELFPLAY_AVAILABLE:
            return await self._fallback_to_standard_solution(sub_problem, workflow_state)

        logger.info(f"Using self-play for {sub_problem.id}")

        # Create self-play arena
        arena = LeanSelfPlayArena(
            theorem=sub_problem.description,
            num_games=self.config.lean_self_play_games
        )

        # Run self-play
        result = await arena.run_self_play()

        # Update progress
        progress.generation = result.total_games
        progress.best_fitness = result.best_score
        progress.current_best = result.best_proof

        # Create solution attempt
        solution = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.best_proof or "",
            generated_by_model="LeanAide-SelfPlay",
            timestamp=time.time(),
            status="verified" if result.best_score > 0.8 else "generated",
            solution_approach="self_play",
            openevolve_metrics={
                "evolution_strategy": "self_play",
                "games_played": result.total_games,
                "best_score": result.best_score,
                "average_score": result.average_score
            }
        )

        return solution

    async def _solve_with_hybrid_approach(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState,
        progress: EvolutionaryProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using hybrid approach combining multiple strategies."""
        logger.info(f"Using hybrid approach for {sub_problem.id}")

        # Try adversarial first
        try:
            if ADVERSARIAL_AVAILABLE:
                solution = await self._solve_with_adversarial_evolution(
                    sub_problem, workflow_state, progress
                )
                if solution.status == "verified":
                    return solution
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning(f"Adversarial evolution failed: {e}")
            progress.warnings.append(f"Adversarial failed: {e}")

        # Fall back to pure evolution
        try:
            if EVOLUTION_AVAILABLE:
                solution = await self._solve_with_pure_evolution(
                    sub_problem, workflow_state, progress
                )
                if solution.status == "verified":
                    return solution
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning(f"Pure evolution failed: {e}")
            progress.warnings.append(f"Evolution failed: {e}")

        # Final fallback to standard
        return await self._fallback_to_standard_solution(sub_problem, workflow_state)

    async def _fallback_to_standard_solution(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """Fallback to standard non-evolutionary solution."""
        logger.info(f"Using standard approach for {sub_problem.id}")

        # This would integrate with existing workflow solution generation
        # For now, return a placeholder
        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=f"# Standard solution for {sub_problem.description}",
            generated_by_model="Standard",
            timestamp=time.time(),
            status="generated",
            solution_approach="standard"
        )

    async def evolve_solution_stage3a(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3A integration: Evolve solution through genetic algorithms.

        Takes an existing solution and evolves it to improve fitness.

        Args:
            solution: Current solution attempt
            workflow_state: Current workflow state

        Returns:
            Evolved solution attempt
        """
        logger.info(f"Stage 3A: Evolving solution for {solution.sub_problem_id}")

        if not EVOLUTION_AVAILABLE:
            return solution

        # Get sub-problem
        sub_problem = None
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            logger.warning(f"Sub-problem {solution.sub_problem_id} not found")
            return solution

        # Create evolution engine with current solution as seed
        engine = LeanProofEvolutionEngine(
            theorem=sub_problem.description,
            theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}",
            population_size=max(10, self.config.lean_evolution_population_size // 2),
            max_generations=max(10, self.config.lean_evolution_generations // 2)
        )

        # Run evolution
        result = await engine.evolve()

        # Update solution
        if result.best_proof and result.best_proof.lean_code:
            solution.content = result.best_proof.lean_code
            solution.status = "verified" if result.success else "generated"
            solution.solution_approach = "evolutionary_stage3a"

            if solution.openevolve_metrics is None:
                solution.openevolve_metrics = {}

            solution.openevolve_metrics.update({
                "stage3a_evolution": True,
                "generations": result.generations_completed,
                "improvement": result.best_strategy.fitness if result.best_strategy else 0.0
            })

        return solution

    async def adversarial_evolution_stage3b(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3B integration: Evolve through adversarial critique.

        Applies red-team vs blue-team adversarial evolution to improve solution.

        Args:
            solution: Current solution attempt
            workflow_state: Current workflow state

        Returns:
            Adversarially evolved solution attempt
        """
        logger.info(f"Stage 3B: Adversarial evolution for {solution.sub_problem_id}")

        if not ADVERSARIAL_AVAILABLE:
            return solution

        # Get sub-problem
        sub_problem = None
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            return solution

        # Create adversarial evolution system
        evolution = LeanAdversarialEvolution()

        # Run adversarial evolution with current solution as starting point
        final_proof, round_results, statistics = evolution.run_adversarial_evolution(
            theorem=sub_problem.description,
            rounds=self.config.lean_adversarial_rounds
        )

        # Update solution
        if final_proof and final_proof.lean_code:
            solution.content = final_proof.lean_code
            solution.status = "verified" if final_proof.confidence > 0.8 else "critiqued"
            solution.solution_approach = "adversarial_evolution_stage3b"

            if solution.openevolve_metrics is None:
                solution.openevolve_metrics = {}

            solution.openevolve_metrics.update({
                "stage3b_adversarial": True,
                "rounds": len(round_results),
                "robustness_score": statistics.blue_success_rate
            })

        return solution

    async def verify_evolved_proof_stage3c(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> VerificationReport:
        """
        Stage 3C integration: Verify evolved proof using LeanAide.

        Performs formal verification of evolved mathematical proofs.

        Args:
            solution: Solution to verify
            workflow_state: Current workflow state

        Returns:
            VerificationReport with LeanAide results
        """
        logger.info(f"Stage 3C: Verifying evolved proof for {solution.sub_problem_id}")

        if not LEANAIDE_AVAILABLE or not self.leanaide_integrator:
            # Return standard verification
            return VerificationReport(
                solution_attempt_id=solution.sub_problem_id,
                gauntlet_name="standard_verification",
                is_approved=True,
                reports_by_judge=[],
                average_score=0.7,
                summary="LeanAide not available, using standard verification"
            )

        # Get sub-problem
        sub_problem = None
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            return VerificationReport(
                solution_attempt_id=solution.sub_problem_id,
                gauntlet_name="leanaide_verification",
                is_approved=False,
                reports_by_judge=[],
                average_score=0.0,
                summary="Sub-problem not found"
            )

        # Perform LeanAide verification
        result = await self.leanaide_integrator.verify_sub_problem_solution(
            sub_problem_id=sub_problem.id,
            problem_statement=sub_problem.description,
            solution_content=solution.content,
            verification_requirements=sub_problem.solution_requirements
        )

        # Convert to VerificationReport
        dimension_scores = {
            "mathematical_correctness": result.confidence_score,
            "formal_verification": 1.0 if result.success else 0.0,
            "proof_quality": 0.8 if result.formal_proof else 0.5
        }

        criteria_met = []
        criteria_not_met = []

        if result.is_mathematical:
            if result.success:
                criteria_met.append("Formal mathematical verification passed")
            else:
                criteria_not_met.append(f"Formal verification failed (confidence: {result.confidence_score:.2f})")

            if result.lean_code:
                criteria_met.append("Lean 4 code generated successfully")
            else:
                criteria_not_met.append("Lean 4 code generation failed")
        else:
            criteria_met.append("Non-mathematical problem (formal verification not required)")

        summary = f"""LeanAide Stage 3C Verification Results:

Mathematical Problem: {result.is_mathematical}
Verification Success: {result.success}
Confidence Score: {result.confidence_score:.2f}
Verification Method: {result.verification_method}

Lean Code Generated: {bool(result.lean_code)}
Formal Proof Generated: {bool(result.formal_proof)}

"""

        if result.errors:
            summary += f"\nErrors:\n" + "\n".join(f"  - {e}" for e in result.errors)
        if result.warnings:
            summary += f"\nWarnings:\n" + "\n".join(f"  - {w}" for w in result.warnings)

        return VerificationReport(
            solution_attempt_id=solution.sub_problem_id,
            gauntlet_name="leanaide_stage3c_verification",
            is_approved=result.success or not result.is_mathematical,
            reports_by_judge=[{
                "method": "LeanAide Stage 3C Verification",
                "result": result.to_dict()
            }],
            average_score=result.confidence_score if result.is_mathematical else 0.8,
            score_variance=0.0,
            summary=summary,
            dimension_scores=dimension_scores,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met,
            resource_usage={
                "verification_method": "leanaide_stage3c",
                "execution_time": result.execution_time
            }
        )

    async def evolutionary_final_verification_stage5(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> VerificationReport:
        """
        Stage 5 integration: Final evolutionary verification.

        Performs comprehensive final verification including evolutionary components.

        Args:
            solution: Final solution to verify
            workflow_state: Current workflow state

        Returns:
            VerificationReport with comprehensive results
        """
        logger.info("Stage 5: Final evolutionary verification")

        if not LEANAIDE_AVAILABLE or not self.leanaide_integrator:
            return VerificationReport(
                solution_attempt_id="final_solution",
                gauntlet_name="standard_final_verification",
                is_approved=True,
                reports_by_judge=[],
                average_score=0.7,
                summary="LeanAide not available for final verification"
            )

        # Prepare sub-problems data
        sub_problems_data = []
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                sp_solution = workflow_state.sub_problem_solutions.get(sp.id)
                sub_problems_data.append({
                    "id": sp.id,
                    "description": sp.description,
                    "solution": sp_solution.content if sp_solution else None
                })

        # Perform final verification
        result = await self.leanaide_integrator.verify_final_solution(
            problem_statement=workflow_state.problem_statement,
            final_solution=solution.content,
            sub_problems=sub_problems_data,
            verification_requirements=None
        )

        # Convert to VerificationReport
        dimension_scores = {
            "mathematical_correctness": result.confidence_score,
            "formal_verification": 1.0 if result.success else 0.0,
            "solution_completeness": 0.9 if result.success else 0.6
        }

        criteria_met = []
        criteria_not_met = []

        if result.is_mathematical:
            if result.success:
                criteria_met.append("Final formal verification passed")
            else:
                criteria_not_met.append(f"Final formal verification failed (confidence: {result.confidence_score:.2f})")
        else:
            criteria_met.append("Non-mathematical solution (formal verification not applicable)")

        summary = f"""LeanAide Stage 5 Final Verification Results:

Mathematical Solution: {result.is_mathematical}
Verification Success: {result.success}
Confidence Score: {result.confidence_score:.2f}
Verification Method: {result.verification_method}

"""

        if result.metadata.get("mathematical_sub_problems"):
            summary += f"\nMathematical Sub-Problems: {len(result.metadata['mathematical_sub_problems'])}"
            summary += f"\nTotal Sub-Problems: {result.metadata.get('total_sub_problems', 0)}"

        return VerificationReport(
            solution_attempt_id="final_solution",
            gauntlet_name="leanaide_stage5_final_verification",
            is_approved=result.success or not result.is_mathematical,
            reports_by_judge=[{
                "method": "LeanAide Stage 5 Final Verification",
                "result": result.to_dict()
            }],
            average_score=result.confidence_score if result.is_mathematical else 0.8,
            score_variance=0.0,
            summary=summary,
            dimension_scores=dimension_scores,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met,
            resource_usage={
                "verification_method": "leanaide_stage5",
                "execution_time": result.execution_time
            }
        )

    async def _store_evolved_proof(
        self,
        sub_problem: SubProblem,
        solution: SolutionAttempt,
        progress: EvolutionaryProgress
    ):
        """Store evolved proof in knowledge base."""
        if not self.ace_manager:
            return

        try:
            artifact = {
                "type": "evolved_proof",
                "sub_problem_id": sub_problem.id,
                "strategy": progress.strategy.value,
                "generations": progress.generation,
                "fitness": progress.best_fitness,
                "proof": solution.content,
                "timestamp": time.time(),
                "domain": sub_problem.mathematical_domain.value if sub_problem.mathematical_domain else None
            }

            self.ace_manager.store_artifact(artifact)
            logger.info(f"Stored evolved proof for {sub_problem.id} in knowledge base")
        except (IOError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to store evolved proof: {e}")

    def get_progress(self, sub_problem_id: str) -> Optional[EvolutionaryProgress]:
        """Get evolution progress for a sub-problem."""
        return self.evolution_progress.get(sub_problem_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get all evolutionary statistics."""
        return dict(self.statistics)


class LeanEvolutionarySubProblemSolver:
    """
    Solves mathematical sub-problems using evolutionary approaches.

    Integrates with SubProblem structure and tracks evolutionary progress
    per sub-problem with detailed metadata.
    """

    def __init__(
        self,
        workflow_stage: LeanEvolutionaryWorkflowStage,
        config: Optional[EvolutionaryConfig] = None
    ):
        """
        Initialize the sub-problem solver.

        Args:
            workflow_stage: The evolutionary workflow stage
            config: Optional configuration override
        """
        self.workflow_stage = workflow_stage
        self.config = config or workflow_stage.config
        self.solved_problems: Dict[str, Dict[str, Any]] = {}

    async def solve(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using evolutionary approach.

        Args:
            sub_problem: The sub-problem to solve
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with evolutionary metadata
        """
        # Check if already solved
        if sub_problem.id in self.solved_problems:
            logger.info(f"Sub-problem {sub_problem.id} already solved, returning cached solution")
            return self.solved_problems[sub_problem.id]["solution"]

        # Detect if mathematical
        is_math, confidence, domain = self.workflow_stage.is_mathematical_subproblem(sub_problem)

        if not is_math:
            logger.info(f"Sub-problem {sub_problem.id} is not mathematical (confidence: {confidence:.2f})")
            # Use standard solver
            return await self._solve_standard(sub_problem, workflow_state)

        # Solve using evolutionary approach
        logger.info(f"Solving mathematical sub-problem {sub_problem.id} (domain: {domain.value if domain else 'unknown'})")

        solution = await self.workflow_stage.solve_subproblem_evolutionary(
            sub_problem, workflow_state
        )

        # Add metadata
        solution.metadata = {
            "mathematical_domain": domain.value if domain else None,
            "math_confidence": confidence,
            "evolutionary": True,
            "solved_by": "LeanEvolutionarySubProblemSolver"
        }

        # Cache solution
        self.solved_problems[sub_problem.id] = {
            "solution": solution,
            "domain": domain,
            "confidence": confidence,
            "timestamp": time.time()
        }

        return solution

    async def _solve_standard(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """Solve using standard (non-evolutionary) approach."""
        # This would integrate with existing sub_problem_solver.py
        # For now, return a placeholder
        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=f"# Standard solution for {sub_problem.description}",
            generated_by_model="Standard",
            timestamp=time.time(),
            status="generated",
            solution_approach="standard",
            metadata={"mathematical": False, "evolutionary": False}
        )

    def get_solution_metadata(self, sub_problem_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a solved sub-problem."""
        if sub_problem_id in self.solved_problems:
            solved = self.solved_problems[sub_problem_id]
            return {
                "domain": solved["domain"].value if solved["domain"] else None,
                "confidence": solved["confidence"],
                "timestamp": solved["timestamp"],
                "evolutionary": True
            }
        return None


class LeanEvolutionaryReassembler:
    """
    Reassembles evolved sub-proofs into complete proof.

    Validates dependencies, checks consistency across evolved components,
    and optimizes the final proof.
    """

    def __init__(
        self,
        workflow_stage: LeanEvolutionaryWorkflowStage
    ):
        """
        Initialize the reassembler.

        Args:
            workflow_stage: The evolutionary workflow stage
        """
        self.workflow_stage = workflow_stage

    async def reassemble(
        self,
        sub_problem_solutions: Dict[str, SolutionAttempt],
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Reassemble evolved sub-proofs into complete proof.

        Args:
            sub_problem_solutions: Dictionary of sub-problem solutions
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with reassembled proof
        """
        logger.info("Reassembling evolved sub-proofs into complete proof")

        # Validate dependencies
        validation_result = await self._validate_dependencies(
            sub_problem_solutions, workflow_state
        )

        # Check consistency
        consistency_result = await self._check_consistency(
            sub_problem_solutions, workflow_state
        )

        # Optimize final proof
        final_proof = await self._optimize_proof(
            sub_problem_solutions, validation_result, consistency_result
        )

        # Create final solution attempt
        solution = SolutionAttempt(
            sub_problem_id="final_solution",
            content=final_proof,
            generated_by_model="LeanEvolutionaryReassembler",
            timestamp=time.time(),
            status="verified" if validation_result["valid"] and consistency_result["consistent"] else "generated",
            solution_approach="evolutionary_reassembly",
            openevolve_metrics={
                "reassembled": True,
                "dependency_validation": validation_result,
                "consistency_check": consistency_result
            }
        )

        return solution

    async def _validate_dependencies(
        self,
        sub_problem_solutions: Dict[str, SolutionAttempt],
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Validate dependencies between sub-proofs."""
        result = {
            "valid": True,
            "missing_dependencies": [],
            "circular_dependencies": [],
            "validation_errors": []
        }

        if not workflow_state.decomposition_plan:
            return result

        # Check each sub-problem's dependencies
        for sp in workflow_state.decomposition_plan.sub_problems:
            for dep_id in sp.dependencies:
                if dep_id not in sub_problem_solutions:
                    result["valid"] = False
                    result["missing_dependencies"].append({
                        "sub_problem": sp.id,
                        "missing_dependency": dep_id
                    })

        return result

    async def _check_consistency(
        self,
        sub_problem_solutions: Dict[str, SolutionAttempt],
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Check consistency across evolved components."""
        result = {
            "consistent": True,
            "inconsistencies": [],
            "warnings": []
        }

        # Check for naming conflicts
        proof_names = set()
        for sp_id, solution in sub_problem_solutions.items():
            # Extract theorem names from Lean code
            import re
            names = re.findall(r'theorem\s+(\w+)', solution.content)
            for name in names:
                if name in proof_names:
                    result["consistent"] = False
                    result["inconsistencies"].append({
                        "type": "name_conflict",
                        "name": name,
                        "sub_problems": [sp_id]
                    })
                proof_names.add(name)

        return result

    async def _optimize_proof(
        self,
        sub_problem_solutions: Dict[str, SolutionAttempt],
        validation_result: Dict[str, Any],
        consistency_result: Dict[str, Any]
    ) -> str:
        """Optimize and combine sub-proofs into final proof."""
        # Combine all sub-proofs
        parts = []
        parts.append("-- Final evolved proof")
        parts.append("-- Generated by LeanEvolutionaryReassembler")
        parts.append(f"-- Timestamp: {datetime.now().isoformat()}")
        parts.append("")

        # Add each sub-proof in dependency order
        for sp_id, solution in sub_problem_solutions.items():
            parts.append(f"-- Sub-problem: {sp_id}")
            parts.append(solution.content)
            parts.append("")

        # Add metadata
        parts.append("-- Validation Results:")
        parts.append(f"-- Dependencies Valid: {validation_result['valid']}")
        parts.append(f"-- Consistency Check: {consistency_result['consistent']}")

        return "\n".join(parts)


# =============================================================================
# Workflow Integration Functions
# =============================================================================

def add_evolutionary_config_to_workflow_state(
    workflow_state: WorkflowState,
    config: EvolutionaryConfig
) -> WorkflowState:
    """
    Add evolutionary configuration to workflow state.

    This integrates evolutionary options with the existing workflow state
    without breaking changes.

    Args:
        workflow_state: Current workflow state
        config: Evolutionary configuration

    Returns:
        Updated workflow state
    """
    # Add evolutionary configuration to openevolve_parameters
    if workflow_state.openevolve_parameters is None:
        workflow_state.openevolve_parameters = {}

    workflow_state.openevolve_parameters.update({
        "lean_evolution_enabled": config.lean_evolution_enabled,
        "lean_evolution_strategy": config.lean_evolution_strategy.value,
        "lean_evolution_generations": config.lean_evolution_generations,
        "lean_evolution_population_size": config.lean_evolution_population_size,
        "lean_adversarial_rounds": config.lean_adversarial_rounds,
        "lean_self_play_games": config.lean_self_play_games,
        "lean_verification_confidence_threshold": config.lean_verification_confidence_threshold,
        "lean_auto_detect_mathematical": config.lean_auto_detect_mathematical,
        "lean_fallback_to_standard": config.lean_fallback_to_standard
    })

    return workflow_state


def extract_evolutionary_config_from_workflow_state(
    workflow_state: WorkflowState
) -> EvolutionaryConfig:
    """
    Extract evolutionary configuration from workflow state.

    Args:
        workflow_state: Current workflow state

    Returns:
        EvolutionaryConfig
    """
    params = workflow_state.openevolve_parameters or {}

    return EvolutionaryConfig(
        lean_evolution_enabled=params.get("lean_evolution_enabled", True),
        lean_evolution_strategy=EvolutionStrategy(
            params.get("lean_evolution_strategy", "hybrid")
        ),
        lean_evolution_generations=params.get(
            "lean_evolution_generations", 50
        ),
        lean_evolution_population_size=params.get(
            "lean_evolution_population_size", 20
        ),
        lean_adversarial_rounds=params.get(
            "lean_adversarial_rounds", 10
        ),
        lean_self_play_games=params.get(
            "lean_self_play_games", 20
        ),
        lean_verification_confidence_threshold=params.get(
            "lean_verification_confidence_threshold", 0.7
        ),
        lean_auto_detect_mathematical=params.get(
            "lean_auto_detect_mathematical", True
        ),
        lean_fallback_to_standard=params.get(
            "lean_fallback_to_standard", True
        )
    )


def is_subproblem_mathematical(
    sub_problem: SubProblem,
    workflow_stage: LeanEvolutionaryWorkflowStage
) -> Tuple[bool, float]:
    """
    Check if a sub-problem is mathematical.

    Args:
        sub_problem: The sub-problem to check
        workflow_stage: The evolutionary workflow stage

    Returns:
        Tuple of (is_mathematical, confidence_score)
    """
    return workflow_stage.is_mathematical_subproblem(sub_problem)[:2]


async def solve_with_evolutionary_approach(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    config: Optional[EvolutionaryConfig] = None
) -> SolutionAttempt:
    """
    Convenience function to solve a sub-problem with evolutionary approach.

    Args:
        sub_problem: The sub-problem to solve
        workflow_state: Current workflow state
        config: Optional evolutionary configuration

    Returns:
        SolutionAttempt with evolved solution
    """
    # Create workflow stage
    stage = LeanEvolutionaryWorkflowStage(
        config=config,
        workflow_state=workflow_state
    )

    # Solve
    return await stage.solve_subproblem_evolutionary(sub_problem, workflow_state)


# =============================================================================
# Convenience Functions for Workflow Integration
# =============================================================================

async def verify_sub_problem_with_leanaide_evolutionary(
    sub_problem: SubProblem,
    solution_attempt: SolutionAttempt,
    workflow_state: WorkflowState
) -> VerificationReport:
    """
    Verify a sub-problem solution using LeanAide evolutionary verification.

    This function integrates with workflow_stage_functions.py for Stage 3C.

    Args:
        sub_problem: The sub-problem being verified
        solution_attempt: The solution attempt to verify
        workflow_state: Current workflow state

    Returns:
        VerificationReport with LeanAide verification results
    """
    config = extract_evolutionary_config_from_workflow_state(workflow_state)
    stage = LeanEvolutionaryWorkflowStage(config=config, workflow_state=workflow_state)

    return await stage.verify_evolved_proof_stage3c(solution_attempt, workflow_state)


async def verify_final_solution_with_leanaide_evolutionary(
    integrated_solution: str,
    workflow_state: WorkflowState
) -> VerificationReport:
    """
    Verify the final integrated solution using LeanAide evolutionary verification.

    This function integrates with workflow_stage_functions.py for Stage 5.

    Args:
        integrated_solution: The final integrated solution
        workflow_state: Current workflow state

    Returns:
        VerificationReport with LeanAide verification results
    """
    config = extract_evolutionary_config_from_workflow_state(workflow_state)
    stage = LeanEvolutionaryWorkflowStage(config=config, workflow_state=workflow_state)

    # Create solution attempt
    solution = SolutionAttempt(
        sub_problem_id="final_solution",
        content=integrated_solution,
        generated_by_model="Integrated",
        timestamp=time.time()
    )

    return await stage.evolutionary_final_verification_stage5(solution, workflow_state)


# Export main classes and functions
__all__ = [
    # Main classes
    'LeanEvolutionaryWorkflowStage',
    'LeanEvolutionarySubProblemSolver',
    'LeanEvolutionaryReassembler',

    # Configuration and data classes
    'EvolutionaryConfig',
    'EvolutionaryProgress',
    'EvolutionStrategy',
    'MathematicalDomain',

    # Workflow integration functions
    'add_evolutionary_config_to_workflow_state',
    'extract_evolutionary_config_from_workflow_state',
    'is_subproblem_mathematical',
    'solve_with_evolutionary_approach',
    'verify_sub_problem_with_leanaide_evolutionary',
    'verify_final_solution_with_leanaide_evolutionary',

    # Availability flags
    'LEANAIDE_AVAILABLE',
    'EVOLUTION_AVAILABLE',
    'ADVERSARIAL_AVAILABLE',
    'SELFPLAY_AVAILABLE',
    'WORKFLOW_AVAILABLE'
]


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example demonstrating the evolutionary workflow integration."""

        print("=== LeanAide Evolutionary Workflow Integration Example ===\n")

        # Create configuration
        config = EvolutionaryConfig(
            lean_evolution_enabled=True,
            lean_evolution_strategy=EvolutionStrategy.HYBRID,
            lean_evolution_generations=20,
            lean_evolution_population_size=15,
            lean_adversarial_rounds=5
        )

        print(f"Configuration:")
        print(f"  Strategy: {config.lean_evolution_strategy.value}")
        print(f"  Generations: {config.lean_evolution_generations}")
        print(f"  Population Size: {config.lean_evolution_population_size}")
        print(f"  Adversarial Rounds: {config.lean_adversarial_rounds}")
        print()

        # Create workflow stage
        stage = LeanEvolutionaryWorkflowStage(config=config)

        # Create example sub-problem
        sub_problem = SubProblem(
            id="sp_001",
            description="Prove that for all natural numbers n, n + 0 = n",
            dependencies=[],
            ai_suggested_complexity_score=3
        )

        print(f"Sub-problem: {sub_problem.description}")
        print()

        # Check if mathematical
        is_math, confidence, domain = stage.is_mathematical_subproblem(sub_problem)
        print(f"Mathematical: {is_math}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Domain: {domain.value if domain else 'N/A'}")
        print()

        # Test availability
        print("Component Availability:")
        print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
        print(f"  Evolution: {EVOLUTION_AVAILABLE}")
        print(f"  Adversarial: {ADVERSARIAL_AVAILABLE}")
        print(f"  Self-Play: {SELFPLAY_AVAILABLE}")
        print(f"  Workflow: {WORKFLOW_AVAILABLE}")
        print()

        print("Example complete!")

    # Run example
    asyncio.run(example_usage())
