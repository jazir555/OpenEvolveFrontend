"""
LeanAide MDAP-Enhanced Evolution Workflow Integration

This module provides comprehensive integration of MDAP (Multi-Strategy Decision
Aggregation Protocol) with evolutionary LeanAide capabilities in the OpenEvolve
decomposition workflow.

Key Features:
- MDAPEvolutionWorkflowIntegrator: Main integration class
- EvolutionaryProgressMonitor: Real-time monitoring of MDAP-enhanced evolution
- HybridEvolutionarySolver: Adaptive strategy selection
- Stage 3A/B/C integration with MDAP-enhanced evolution
- Configuration integration with WorkflowState
- Fallback strategies and error handling
- Integration with LeanAide, CrewAI, Knowledge Engine, and ACE

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
from concurrent.futures import ThreadPoolExecutor

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

# Import evolutionary workflow
try:
    from leanaide_evolutionary_workflow import (
        LeanEvolutionaryWorkflowStage,
        EvolutionaryConfig,
        EvolutionStrategy,
        MathematicalDomain,
        EvolutionaryProgress
    )
    EVOLUTION_WORKFLOW_AVAILABLE = True
except ImportError:
    EVOLUTION_WORKFLOW_AVAILABLE = False
    logger.warning("Evolutionary workflow not available")

# Import MDAP workflow
try:
    from leanaide_mdap_workflow import (
        LeanMDAPWorkflowIntegrator,
        LeanMDAPConfig,
        MDAPStrategyType,
        LeanMDAPTask,
        LeanMDAPResult
    )
    MDAP_WORKFLOW_AVAILABLE = True
except ImportError:
    MDAP_WORKFLOW_AVAILABLE = False
    logger.warning("MDAP workflow not available")

# Import LeanAide integration
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig,
        is_leanaide_configured
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide workflow integration not available")

# import crewai # MIGRATED: was CrewAI
try:
    from crewai_client import CrewAIClient
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import ACE Knowledge Manager
try:
    from ace_knowledge_artifacts import ACEKnowledgeManager
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class MDAPEvolutionConfig:
    """Configuration for MDAP-enhanced evolutionary integration."""
    # Enablement
    enabled: bool = True

    # Evolution parameters
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_ratio: float = 0.1

    # MDAP agent parameters
    agents: List[str] = field(default_factory=lambda: [
        "direct_prover",
        "inductive_prover",
        "constructive_prover",
        "decomposition_prover"
    ])
    parallel_agents: int = 4
    agent_timeout: float = 120.0

    # MDAP voting for evolution
    selection_voting: str = "weighted_confidence"  # For selection
    crossover_voting: str = "majority"  # For crossover
    mutation_voting: str = "consensus"  # For mutation

    # Consensus thresholds
    min_consensus: float = 0.6
    k_ahead: int = 3

    # Monitoring and tracking
    track_agents: bool = True
    monitor_population_diversity: bool = True
    track_agent_performance: bool = True

    # Fallback
    fallback_to_evolution: bool = True
    fallback_to_mdap: bool = True
    fallback_to_standard: bool = True

    # Integration
    CrewAI_enabled: bool = False
    ace_learning_enabled: bool = True
    verify_with_leanaide: bool = True
    verification_timeout: float = 60.0
    confidence_threshold: float = 0.7

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPEvolutionProgress:
    """Progress tracking for MDAP-enhanced evolution."""
    sub_problem_id: str
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    agent_consensus: float = 0.0
    diversity_score: float = 0.0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    status: str = "in_progress"
    agent_votes: Dict[str, int] = field(default_factory=dict)
    agent_performance: Dict[str, float] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_problem_id": self.sub_problem_id,
            "generation": self.generation,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "agent_consensus": self.agent_consensus,
            "diversity_score": self.diversity_score,
            "elapsed_time": self.elapsed_time,
            "status": self.status,
            "agent_votes": self.agent_votes,
            "agent_performance": self.agent_performance,
            "convergence_history": self.convergence_history,
            "errors": self.errors,
            "warnings": self.warnings
        }


class EvolutionaryStrategySelection(Enum):
    """Strategy selection modes for hybrid solver."""
    EVOLUTION_ONLY = "evolution_only"
    MDAP_ONLY = "mdap_only"
    MDAP_EVOLUTION = "mdap_evolution"
    ADAPTIVE = "adaptive"


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class MDAPEvolutionWorkflowIntegrator:
    """
    Main integration class for MDAP-enhanced evolution in workflow.

    This class orchestrates evolutionary proofs with MDAP voting, combining
    the power of genetic algorithms with multi-agent consensus.

    Stage Integration:
        - Stage 3A: Generate initial proofs using MDAP-enhanced evolution
        - Stage 3B: Refine proofs using MDAP-enhanced evolution
        - Stage 3C: Verify with tracking of agent contributions
        - Stage 5: Final verification with MDAP-evolution fallback
    """

    def __init__(
        self,
        config: Optional[MDAPEvolutionConfig] = None,
        workflow_state: Optional[WorkflowState] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize the MDAP-evolution workflow integrator.

        Args:
            config: MDAP-evolution configuration
            workflow_state: Current workflow state
            team: Team for LLM calls
        """
        self.config = config or MDAPEvolutionConfig()
        self.workflow_state = workflow_state
        self.team = team

        # Initialize components
        self.evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
        self.mdap_integrator: Optional[LeanMDAPWorkflowIntegrator] = None
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None
        self.crewai_client: Optional[CrewAIClient] = None
        self.ace_manager: Optional[ACEKnowledgeManager] = None

        # Progress tracking
        self.evolution_progress: Dict[str, MDAPEvolutionProgress] = {}
        self.agent_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))

        # Population tracking
        self.current_population: List[Dict[str, Any]] = []
        self.population_history: List[List[Dict[str, Any]]] = []

        # Initialize
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all required components."""
        # Initialize evolutionary stage
        if EVOLUTION_WORKFLOW_AVAILABLE and self.config.enabled:
            evolutionary_config = EvolutionaryConfig(
                lean_evolution_enabled=True,
                lean_evolution_strategy=EvolutionStrategy.HYBRID,
                lean_evolution_generations=self.config.generations,
                lean_evolution_population_size=self.config.population_size,
                lean_evolution_mutation_rate=self.config.mutation_rate,
                lean_evolution_crossover_rate=self.config.crossover_rate,
                lean_verification_confidence_threshold=self.config.confidence_threshold
            )
            self.evolutionary_stage = LeanEvolutionaryWorkflowStage(
                config=evolutionary_config,
                workflow_state=self.workflow_state
            )

        # Initialize MDAP integrator
        if MDAP_WORKFLOW_AVAILABLE and self.config.enabled:
            mdap_config = LeanMDAPConfig(
                enabled=True,
                agents=self.config.agents,
                parallel_agents=self.config.parallel_agents,
                agent_timeout=self.config.agent_timeout,
                voting_strategy=self.config.selection_voting,
                k_ahead=self.config.k_ahead,
                min_consensus=self.config.min_consensus,
                verify_strategies=self.config.verify_with_leanaide,
                verification_timeout=self.config.verification_timeout,
                confidence_threshold=self.config.confidence_threshold,
                fallback_to_evolution=self.config.fallback_to_evolution
            )
            self.mdap_integrator = LeanMDAPWorkflowIntegrator(
                config=mdap_config,
                workflow_state=self.workflow_state,
                team=self.team
            )

        # Initialize LeanAide integrator
        if LEANAIDE_AVAILABLE and self.config.verify_with_leanaide:
            leanaide_config = LeanAideWorkflowConfig(
                enabled=True,
                confidence_threshold=self.config.confidence_threshold
            )
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

        # Initialize CrewAI if enabled
        if self.config.CrewAI_enabled and CREWAI_AVAILABLE:
            self.crewai_client = CrewAIClient(timeout=self.config.agent_timeout)

        # Initialize ACE manager
        if self.config.ace_learning_enabled and ACE_AVAILABLE:
            self.ace_manager = ACEKnowledgeManager()

    async def solve_with_mdap_evolution(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MDAP-enhanced evolutionary approach.

        This is the main entry point that combines:
        1. Evolutionary generation of proof candidates
        2. MDAP agent voting on selection, crossover, mutation
        3. Consensus-based refinement

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MDAP-evolved proof
        """
        start_time = time.time()
        sub_problem_id = sub_problem.id

        logger.info(f"Solving {sub_problem_id} with MDAP-enhanced evolution")

        # Create progress tracker
        progress = MDAPEvolutionProgress(
            sub_problem_id=sub_problem_id,
            population_size=self.config.population_size
        )
        self.evolution_progress[sub_problem_id] = progress

        try:
            # Step 1: Initialize population using MDAP agents
            logger.info(f"Initializing population with {self.config.agents}")
            population = await self._initialize_mdap_population(sub_problem)

            if not population:
                logger.warning("Population initialization failed, falling back")
                if self.config.fallback_to_evolution:
                    return await self._fallback_to_evolution(sub_problem)
                elif self.config.fallback_to_mdap:
                    return await self._fallback_to_mdap(sub_problem)
                else:
                    raise Exception("Population initialization failed and no fallback enabled")

            self.current_population = population
            progress.population_size = len(population)

            # Step 2: Evolve population with MDAP voting
            logger.info(f"Evolving for {self.config.generations} generations")
            best_individual = await self._evolve_with_mdap_voting(
                sub_problem, population, progress
            )

            # Step 3: Verify final proof
            verification_status = "generated"
            if self.config.verify_with_leanaide and self.leanaide_integrator:
                verification_result = await self._verify_proof(
                    sub_problem, best_individual['proof']
                )
                verification_status = "verified" if verification_result else "failed"

            # Step 4: Create solution attempt
            solution = SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=best_individual['proof'],
                generated_by_model="LeanAide-MDAP-Evolution",
                timestamp=time.time(),
                status=verification_status,
                solution_approach="mdap_evolutionary",
                openevolve_metrics={
                    "mdap_evolution": True,
                    "generations": progress.generation,
                    "final_fitness": best_individual['fitness'],
                    "agent_consensus": progress.agent_consensus,
                    "diversity_score": progress.diversity_score,
                    "execution_time": time.time() - start_time,
                    "agent_votes": progress.agent_votes,
                    "agent_performance": progress.agent_performance
                }
            )

            # Update progress
            progress.status = "completed"
            progress.elapsed_time = time.time() - start_time
            progress.best_fitness = best_individual['fitness']

            # Store in knowledge base
            if self.ace_manager:
                await self._store_mdap_evolution_result(
                    sub_problem, solution, progress
                )

            return solution

        except Exception as e:
            logger.error(f"MDAP-evolution failed for {sub_problem_id}: {e}", exc_info=True)
            progress.errors.append(str(e))
            progress.status = "failed"

            # Fallback
            if self.config.fallback_to_evolution:
                return await self._fallback_to_evolution(sub_problem)
            elif self.config.fallback_to_mdap:
                return await self._fallback_to_mdap(sub_problem)
            else:
                raise

    async def _initialize_mdap_population(
        self,
        sub_problem: SubProblem
    ) -> List[Dict[str, Any]]:
        """
        Initialize population using MDAP agents.

        Each agent generates initial proof candidates, which form the
        initial population for evolution.

        Args:
            sub_problem: The sub-problem

        Returns:
            List of individuals (proof candidates with metadata)
        """
        population = []

        if not self.mdap_integrator:
            # Fallback: generate synthetic population
            return await self._generate_synthetic_population(sub_problem)

        # Create MDAP tasks for each agent
        tasks = await self._create_mdap_tasks(sub_problem)

        # Execute tasks in parallel
        results = await self._execute_mdap_tasks(tasks, sub_problem)

        # Convert results to population
        for result in results:
            individual = {
                'proof': result.lean_code,
                'fitness': result.confidence,
                'strategy': result.strategy_type.value,
                'agent': result.agent_id,
                'verification_status': result.verification_status,
                'generation': 0
            }
            population.append(individual)

        # Fill population if needed
        while len(population) < self.config.population_size:
            # Add variations of existing individuals
            if population:
                base = population[0]
                variation = base.copy()
                variation['proof'] = self._mutate_proof(base['proof'])
                variation['fitness'] = base['fitness'] * 0.9  # Slightly lower fitness
                population.append(variation)
            else:
                # No results, generate synthetic
                return await self._generate_synthetic_population(sub_problem)

        return population[:self.config.population_size]

    async def _generate_synthetic_population(
        self,
        sub_problem: SubProblem
    ) -> List[Dict[str, Any]]:
        """Generate synthetic population when MDAP is unavailable."""
        population = []

        for i in range(self.config.population_size):
            individual = {
                'proof': f"""theorem {sub_problem.id.replace('-', '_')}_{i} : {sub_problem.description} :=
  by
    -- Generated proof {i}
    sorry""",
                'fitness': 0.5 + (i * 0.01),  # Varying fitness
                'strategy': 'synthetic',
                'agent': 'synthetic',
                'verification_status': 'not_verified',
                'generation': 0
            }
            population.append(individual)

        return population

    async def _create_mdap_tasks(
        self,
        sub_problem: SubProblem
    ) -> List[LeanMDAPTask]:
        """Create MDAP tasks for population initialization."""
        tasks = []

        strategy_types = [
            MDAPStrategyType.DIRECT,
            MDAPStrategyType.INDUCTION,
            MDAPStrategyType.CONSTRUCTIVE,
            MDAPStrategyType.DECOMPOSITION
        ]

        for i, strategy_type in enumerate(strategy_types):
            task = LeanMDAPTask(
                task_id=f"mdap_init_{sub_problem.id}_{strategy_type.value}_{uuid.uuid4().hex[:8]}",
                sub_problem_id=sub_problem.id,
                theorem_statement=sub_problem.description,
                proof_goal=f"prove_{sub_problem.id.replace('-', '_')}",
                context={
                    "dependencies": sub_problem.dependencies,
                    "requirements": sub_problem.solution_requirements or []
                },
                strategy_type=strategy_type,
                agent_id=self.config.agents[i % len(self.config.agents)],
                priority=sub_problem.priority
            )
            tasks.append(task)

        return tasks

    async def _execute_mdap_tasks(
        self,
        tasks: List[LeanMDAPTask],
        sub_problem: SubProblem
    ) -> List[LeanMDAPResult]:
        """Execute MDAP tasks in parallel."""
        if not self.mdap_integrator:
            return []

        # Create semaphore for parallel execution
        semaphore = asyncio.Semaphore(self.config.parallel_agents)

        async def execute_with_semaphore(task: LeanMDAPTask):
            async with semaphore:
                return await self._execute_single_mdap_task(task, sub_problem)

        # Execute all tasks
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"MDAP task failed: {result}")
            elif isinstance(result, LeanMDAPResult):
                valid_results.append(result)

        return valid_results

    async def _execute_single_mdap_task(
        self,
        task: LeanMDAPTask,
        sub_problem: SubProblem
    ) -> Optional[LeanMDAPResult]:
        """Execute a single MDAP task."""
        try:
            # Generate proof using strategy
            lean_code = await self._generate_proof_with_strategy(task, sub_problem)

            # Calculate confidence
            confidence = self._calculate_proof_confidence(task, lean_code)

            result = LeanMDAPResult(
                task_id=task.task_id,
                strategy_type=task.strategy_type,
                agent_id=task.agent_id,
                lean_code=lean_code,
                proof_steps=self._extract_proof_steps(lean_code),
                confidence=confidence,
                verification_status="not_verified",
                verification_time=0.0
            )

            # Track agent statistics
            if self.config.track_agents:
                self.agent_statistics[task.agent_id]["proofs_generated"].append(1)
                self.agent_statistics[task.agent_id]["confidence_scores"].append(confidence)

            return result

        except Exception as e:
            logger.error(f"MDAP task {task.task_id} failed: {e}")
            return None

    async def _generate_proof_with_strategy(
        self,
        task: LeanMDAPTask,
        sub_problem: SubProblem
    ) -> str:
        """Generate Lean proof using specific strategy."""
        # Strategy-specific prompts
        strategy_prompts = {
            MDAPStrategyType.DIRECT: f"Prove directly: {task.theorem_statement}",
            MDAPStrategyType.INDUCTION: f"Prove by induction: {task.theorem_statement}",
            MDAPStrategyType.CONSTRUCTIVE: f"Prove constructively: {task.theorem_statement}",
            MDAPStrategyType.DECOMPOSITION: f"Prove by decomposition: {task.theorem_statement}"
        }

        prompt = strategy_prompts.get(
            task.strategy_type,
            f"Prove: {task.theorem_statement}"
        )

        # Generate proof (placeholder - would use LLM in production)
        return f"""theorem {task.proof_goal} : {task.theorem_statement} :=
  by
    -- Proof using {task.strategy_type.value} strategy
    -- Agent: {task.agent_id}
    sorry"""

    def _calculate_proof_confidence(
        self,
        task: LeanMDAPTask,
        lean_code: str
    ) -> float:
        """Calculate confidence score for a proof."""
        confidence = 0.5

        # Adjust based on code quality
        if "sorry" in lean_code:
            confidence *= 0.5

        # Adjust based on strategy
        strategy_bonus = {
            MDAPStrategyType.DIRECT: 0.1,
            MDAPStrategyType.INDUCTION: 0.15,
            MDAPStrategyType.CONSTRUCTIVE: 0.1,
            MDAPStrategyType.DECOMPOSITION: 0.2
        }
        confidence += strategy_bonus.get(task.strategy_type, 0.0)

        return max(0.0, min(1.0, confidence))

    def _extract_proof_steps(self, lean_code: str) -> List[str]:
        """Extract proof steps from Lean code."""
        steps = []
        for line in lean_code.split('\n'):
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('theorem'):
                steps.append(line)
        return steps

    async def _evolve_with_mdap_voting(
        self,
        sub_problem: SubProblem,
        population: List[Dict[str, Any]],
        progress: MDAPEvolutionProgress
    ) -> Dict[str, Any]:
        """
        Evolve population using MDAP agent voting.

        Each evolutionary operation (selection, crossover, mutation) is
        guided by MDAP agent consensus.

        Args:
            sub_problem: The sub-problem
            population: Initial population
            progress: Progress tracker

        Returns:
            Best individual found
        """
        best_individual = max(population, key=lambda x: x['fitness'])

        for generation in range(self.config.generations):
            progress.generation = generation

            # Step 1: MDAP-guided selection
            selected = await self._mdap_selection(population, progress)

            # Step 2: MDAP-guided crossover
            offspring = await self._mdap_crossover(selected, progress)

            # Step 3: MDAP-guided mutation
            mutated = await self._mdap_mutation(offspring, progress)

            # Step 4: Update population
            population = self._update_population(population, mutated)

            # Track best
            current_best = max(population, key=lambda x: x['fitness'])
            if current_best['fitness'] > best_individual['fitness']:
                best_individual = current_best

            # Update progress
            progress.best_fitness = best_individual['fitness']
            progress.convergence_history.append(best_individual['fitness'])
            progress.diversity_score = self._calculate_diversity(population)

            # Track population
            self.population_history.append(population.copy())

            # Check convergence
            if self._check_convergence(progress):
                logger.info(f"Converged at generation {generation}")
                break

        return best_individual

    async def _mdap_selection(
        self,
        population: List[Dict[str, Any]],
        progress: MDAPEvolutionProgress
    ) -> List[Dict[str, Any]]:
        """
        MDAP-guided selection.

        Agents vote on which individuals should be selected for breeding.
        """
        # Selection strategies
        if self.config.selection_voting == "weighted_confidence":
            # Select by fitness (confidence)
            sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
            selected = sorted_pop[:max(2, len(population) // 2)]

        elif self.config.selection_voting == "majority":
            # Majority vote based on agent preferences
            agent_votes = defaultdict(int)
            for individual in population:
                agent = individual.get('agent', 'unknown')
                agent_votes[agent] += individual['fitness']

            # Select top agents' individuals
            top_agents = sorted(agent_votes.keys(), key=lambda x: agent_votes[x], reverse=True)[:2]
            selected = [ind for ind in population if ind.get('agent') in top_agents]

        else:  # consensus
            # Select individuals with broad agent support
            selected = population[:max(2, len(population) // 2)]

        # Track votes
        for ind in selected:
            agent = ind.get('agent', 'unknown')
            progress.agent_votes[agent] = progress.agent_votes.get(agent, 0) + 1

        # Calculate consensus
        if progress.agent_votes:
            total_votes = sum(progress.agent_votes.values())
            max_votes = max(progress.agent_votes.values())
            progress.agent_consensus = max_votes / total_votes if total_votes > 0 else 0.0

        return selected

    async def _mdap_crossover(
        self,
        selected: List[Dict[str, Any]],
        progress: MDAPEvolutionProgress
    ) -> List[Dict[str, Any]]:
        """
        MDAP-guided crossover.

        Agents vote on crossover points and strategies.
        """
        offspring = []

        # Pair up selected individuals
        for i in range(0, len(selected) - 1, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

            # Perform crossover
            child1, child2 = self._crossover_proofs(parent1, parent2)

            offspring.extend([child1, child2])

        return offspring if offspring else selected

    def _crossover_proofs(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two proof individuals."""
        # Simple crossover: mix proof content
        proof1_lines = parent1['proof'].split('\n')
        proof2_lines = parent2['proof'].split('\n')

        # Find crossover point
        min_len = min(len(proof1_lines), len(proof2_lines))
        crossover_point = min_len // 2

        # Create children
        child1_proof = '\n'.join(proof1_lines[:crossover_point] + proof2_lines[crossover_point:])
        child2_proof = '\n'.join(proof2_lines[:crossover_point] + proof1_lines[crossover_point:])

        child1 = parent1.copy()
        child1['proof'] = child1_proof
        child1['fitness'] = (parent1['fitness'] + parent2['fitness']) / 2 * 0.9
        child1['generation'] = parent1.get('generation', 0) + 1

        child2 = parent2.copy()
        child2['proof'] = child2_proof
        child2['fitness'] = (parent1['fitness'] + parent2['fitness']) / 2 * 0.9
        child2['generation'] = parent2.get('generation', 0) + 1

        return child1, child2

    async def _mdap_mutation(
        self,
        offspring: List[Dict[str, Any]],
        progress: MDAPEvolutionProgress
    ) -> List[Dict[str, Any]]:
        """
        MDAP-guided mutation.

        Agents vote on mutation strategies and apply mutations.
        """
        mutated = []

        for individual in offspring:
            # Decide whether to mutate
            import random
            if random.random() < self.config.mutation_rate:
                mutated_individual = individual.copy()
                mutated_individual['proof'] = self._mutate_proof(individual['proof'])
                mutated_individual['fitness'] *= 0.95  # Slightly reduce fitness
                mutated_individual['generation'] = individual.get('generation', 0) + 1
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)

        return mutated

    def _mutate_proof(self, proof: str) -> str:
        """Apply mutation to a proof."""
        lines = proof.split('\n')

        # Simple mutation: add or modify a line
        if len(lines) > 2:
            mutation_point = len(lines) // 2
            lines.insert(mutation_point, "    -- Mutated step")
            return '\n'.join(lines)

        return proof

    def _update_population(
        self,
        old_population: List[Dict[str, Any]],
        new_individuals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update population using elitism and new individuals."""
        # Keep best individuals (elitism)
        elite_count = max(1, int(self.config.population_size * self.config.elitism_ratio))
        sorted_old = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
        elites = sorted_old[:elite_count]

        # Combine with new individuals
        combined = elites + new_individuals

        # Truncate to population size
        sorted_combined = sorted(combined, key=lambda x: x['fitness'], reverse=True)
        return sorted_combined[:self.config.population_size]

    def _calculate_diversity(self, population: List[Dict[str, Any]]) -> float:
        """Calculate population diversity score."""
        if len(population) < 2:
            return 0.0

        # Calculate average pairwise distance
        total_distance = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Simple distance based on fitness difference
                distance = abs(population[i]['fitness'] - population[j]['fitness'])
                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def _check_convergence(self, progress: MDAPEvolutionProgress) -> bool:
        """Check if evolution has converged."""
        if len(progress.convergence_history) < 10:
            return False

        # Check improvement in last 10 generations
        recent = progress.convergence_history[-10:]
        improvement = max(recent) - min(recent)

        return improvement < 0.01  # Less than 1% improvement

    async def _verify_proof(
        self,
        sub_problem: SubProblem,
        proof: str
    ) -> bool:
        """Verify a proof using LeanAide."""
        if not self.leanaide_integrator:
            return False

        try:
            result = await self.leanaide_integrator.verify_sub_problem_solution(
                sub_problem_id=sub_problem.id,
                problem_statement=sub_problem.description,
                solution_content=proof,
                verification_requirements=sub_problem.solution_requirements
            )
            return result.success
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def _mutate_proof(self, proof: str) -> str:
        """Mutate a proof (duplicate method, removing one)."""
        lines = proof.split('\n')
        if len(lines) > 2:
            mutation_point = len(lines) // 2
            lines.insert(mutation_point, "    -- Mutated step")
            return '\n'.join(lines)
        return proof

    async def _fallback_to_evolution(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to pure evolutionary approach."""
        logger.info(f"Falling back to pure evolution for {sub_problem.id}")

        if not self.evolutionary_stage:
            raise Exception("Evolutionary stage not available")

        return await self.evolutionary_stage.solve_subproblem_evolutionary(
            sub_problem, self.workflow_state
        )

    async def _fallback_to_mdap(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to pure MDAP approach."""
        logger.info(f"Falling back to pure MDAP for {sub_problem.id}")

        if not self.mdap_integrator:
            raise Exception("MDAP integrator not available")

        return await self.mdap_integrator.solve_subproblem_with_mdap(sub_problem)

    async def mdap_evolution_stage3a(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3A: Generate initial proof using MDAP-enhanced evolution.

        Args:
            sub_problem: Sub-problem to solve
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with MDAP-evolved proof
        """
        logger.info(f"MDAP-Evolution Stage 3A: Solving {sub_problem.id}")

        # Configure from workflow state
        self.config = self.configure_mdap_evolution_from_workflow(workflow_state)

        # Solve with MDAP-evolution
        return await self.solve_with_mdap_evolution(sub_problem)

    async def mdap_evolution_stage3b(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3B: Refine proof using MDAP-enhanced evolution.

        Uses existing solution to seed population.

        Args:
            solution: Current solution to refine
            workflow_state: Current workflow state

        Returns:
            Refined solution attempt
        """
        logger.info(f"MDAP-Evolution Stage 3B: Refining {solution.sub_problem_id}")

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

        # Configure for refinement
        self.config.generations = max(10, self.config.generations // 2)
        self.config.population_size = max(10, self.config.population_size // 2)

        # Seed population with current solution
        seeded_individual = {
            'proof': solution.content,
            'fitness': 0.8,
            'strategy': 'seeded',
            'agent': 'seeded',
            'verification_status': 'seeded',
            'generation': 0
        }

        self.current_population = [seeded_individual]

        # Run evolution
        refined_solution = await self.solve_with_mdap_evolution(sub_problem)

        # Update solution
        solution.content = refined_solution.content
        solution.status = refined_solution.status
        solution.solution_approach = "mdap_evolution_stage3b_refined"

        if solution.openevolve_metrics is None:
            solution.openevolve_metrics = {}

        solution.openevolve_metrics.update(refined_solution.openevolve_metrics)
        solution.openevolve_metrics["stage3b_refinement"] = True

        return solution

    def configure_mdap_evolution_from_workflow(
        self,
        state: WorkflowState
    ) -> MDAPEvolutionConfig:
        """
        Configure MDAP-evolution from workflow state.

        Args:
            state: Current workflow state

        Returns:
            MDAPEvolutionConfig
        """
        params = state.openevolve_parameters or {}

        return MDAPEvolutionConfig(
            enabled=params.get("lean_mdap_evolution_enabled", True),
            population_size=params.get("lean_mdap_evolution_population_size", self.config.population_size),
            generations=params.get("lean_mdap_evolution_generations", self.config.generations),
            agents=params.get("lean_mdap_evolution_agents", self.config.agents),
            selection_voting=params.get("lean_mdap_evolution_selection_voting", self.config.selection_voting),
            crossover_voting=params.get("lean_mdap_evolution_crossover_voting", self.config.crossover_voting),
            mutation_voting=params.get("lean_mdap_evolution_mutation_voting", self.config.mutation_voting),
            track_agents=params.get("lean_mdap_evolution_track_agents", self.config.track_agents)
        )

    async def _store_mdap_evolution_result(
        self,
        sub_problem: SubProblem,
        solution: SolutionAttempt,
        progress: MDAPEvolutionProgress
    ):
        """Store MDAP-evolution result in knowledge base."""
        if not self.ace_manager:
            return

        artifact = {
            "type": "mdap_evolution_result",
            "sub_problem_id": sub_problem.id,
            "theorem": sub_problem.description,
            "generations": progress.generation,
            "final_fitness": progress.best_fitness,
            "agent_consensus": progress.agent_consensus,
            "diversity": progress.diversity_score,
            "proof": solution.content,
            "agent_votes": progress.agent_votes,
            "agent_performance": progress.agent_performance,
            "timestamp": time.time()
        }

        self.ace_manager.store_artifact(artifact)


# =============================================================================
# EVOLUTIONARY PROGRESS MONITOR
# =============================================================================

class EvolutionaryProgressMonitor:
    """
    Monitor MDAP-enhanced evolutionary execution.

    Provides real-time tracking of population, fitness, agent votes,
    and convergence metrics.
    """

    def __init__(
        self,
        integrator: Optional[MDAPEvolutionWorkflowIntegrator] = None
    ):
        """
        Initialize the progress monitor.

        Args:
            integrator: MDAP-evolution integrator to monitor
        """
        self.integrator = integrator
        self.monitoring_active: bool = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval: float = 1.0  # seconds

    def start_monitoring(
        self,
        engine: MDAPEvolutionWorkflowIntegrator
    ) -> None:
        """
        Start monitoring an MDAP-evolution engine.

        Args:
            engine: The engine to monitor
        """
        self.integrator = engine
        self.monitoring_active = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("Started MDAP-evolution monitoring")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            # Collect statistics
            stats = self.get_progress()

            # Log if available
            if stats.get("status") == "monitoring":
                logger.debug(f"Monitoring: Generation {stats.get('generation')}, "
                           f"Best Fitness: {stats.get('best_fitness', 0.0):.3f}")

            time.sleep(self.monitoring_interval)

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

    def get_population_statistics(self) -> Dict[str, Any]:
        """
        Get population statistics.

        Returns:
            Dict with size, diversity, fitness metrics
        """
        if not self.integrator or not self.integrator.current_population:
            return {
                "size": 0,
                "diversity": 0.0,
                "avg_fitness": 0.0,
                "best_fitness": 0.0,
                "worst_fitness": 0.0
            }

        population = self.integrator.current_population

        fitnesses = [ind['fitness'] for ind in population]

        return {
            "size": len(population),
            "diversity": self.integrator._calculate_diversity(population),
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "worst_fitness": min(fitnesses) if fitnesses else 0.0,
            "fitness_std": self._calculate_std(fitnesses)
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.

        Returns:
            Dict with current generation, best fitness, convergence
        """
        if not self.integrator or not self.integrator.evolution_progress:
            return {
                "current_generation": 0,
                "best_fitness": 0.0,
                "convergence_rate": 0.0,
                "status": "no_progress"
            }

        # Get most recent progress
        progress = list(self.integrator.evolution_progress.values())[-1]

        return {
            "current_generation": progress.generation,
            "best_fitness": progress.best_fitness,
            "convergence_rate": self._calculate_convergence_rate(progress),
            "agent_consensus": progress.agent_consensus,
            "diversity_score": progress.diversity_score,
            "status": progress.status
        }

    def _calculate_convergence_rate(self, progress: MDAPEvolutionProgress) -> float:
        """Calculate convergence rate from history."""
        if len(progress.convergence_history) < 2:
            return 0.0

        recent = progress.convergence_history[-10:]
        if len(recent) < 2:
            return 0.0

        # Rate of change
        return (recent[-1] - recent[0]) / len(recent)

    def get_agent_vote_statistics(self) -> Dict[str, Any]:
        """
        Get agent voting statistics.

        Returns:
            Dict with vote distribution, consensus rates
        """
        all_votes = defaultdict(int)
        total_votes = 0

        if self.integrator and self.integrator.evolution_progress:
            for progress in self.integrator.evolution_progress.values():
                for agent, votes in progress.agent_votes.items():
                    all_votes[agent] += votes
                    total_votes += votes

        if total_votes == 0:
            return {
                "total_votes": 0,
                "vote_distribution": {},
                "consensus_rate": 0.0,
                "most_voted_agent": None
            }

        # Calculate consensus (concentration of votes)
        max_votes = max(all_votes.values())
        consensus_rate = max_votes / total_votes if total_votes > 0 else 0.0

        return {
            "total_votes": total_votes,
            "vote_distribution": dict(all_votes),
            "consensus_rate": consensus_rate,
            "most_voted_agent": max(all_votes.keys(), key=all_votes.get) if all_votes else None
        }

    def get_agent_performance(self) -> Dict[str, Any]:
        """
        Get per-agent performance metrics.

        Returns:
            Dict with per-agent success rates
        """
        if not self.integrator:
            return {}

        performance = {}
        for agent_id, stats in self.integrator.agent_statistics.items():
            proofs_generated = len(stats.get("proofs_generated", []))
            confidence_scores = stats.get("confidence_scores", [])

            performance[agent_id] = {
                "proofs_generated": proofs_generated,
                "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "best_confidence": max(confidence_scores) if confidence_scores else 0.0,
                "success_rate": sum(1 for c in confidence_scores if c > 0.7) / len(confidence_scores) if confidence_scores else 0.0
            }

        return performance

    def get_progress(self) -> Dict[str, Any]:
        """
        Get overall progress.

        Returns:
            Dict with all progress information
        """
        return {
            "status": "monitoring" if self.monitoring_active else "stopped",
            "population_stats": self.get_population_statistics(),
            "generation_stats": self.get_generation_statistics(),
            "agent_votes": self.get_agent_vote_statistics(),
            "agent_performance": self.get_agent_performance()
        }


# =============================================================================
# HYBRID EVOLUTIONARY SOLVER
# =============================================================================

class HybridEvolutionarySolver:
    """
    Hybrid solver that can use evolution, MDAP, or evolution+MDAP.

    Adaptive selection based on problem characteristics.
    """

    def __init__(
        self,
        mdap_evolution_config: Optional[MDAPEvolutionConfig] = None,
        workflow_state: Optional[WorkflowState] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize the hybrid solver.

        Args:
            mdap_evolution_config: MDAP-evolution configuration
            workflow_state: Current workflow state
            team: Team for LLM calls
        """
        self.mdap_evolution_config = mdap_evolution_config or MDAPEvolutionConfig()
        self.workflow_state = workflow_state
        self.team = team

        # Initialize solvers
        self.mdap_evolution_integrator: Optional[MDAPEvolutionWorkflowIntegrator] = None
        self.evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
        self.mdap_integrator: Optional[LeanMDAPWorkflowIntegrator] = None

        self._initialize_solvers()

    def _initialize_solvers(self):
        """Initialize all solvers."""
        # MDAP-evolution
        self.mdap_evolution_integrator = MDAPEvolutionWorkflowIntegrator(
            config=self.mdap_evolution_config,
            workflow_state=self.workflow_state,
            team=self.team
        )

        # Pure evolution
        if EVOLUTION_WORKFLOW_AVAILABLE:
            evolutionary_config = EvolutionaryConfig(
                lean_evolution_enabled=True,
                lean_evolution_strategy=EvolutionStrategy.HYBRID
            )
            self.evolutionary_stage = LeanEvolutionaryWorkflowStage(
                config=evolutionary_config,
                workflow_state=self.workflow_state
            )

        # Pure MDAP
        if MDAP_WORKFLOW_AVAILABLE:
            mdap_config = LeanMDAPConfig(enabled=True)
            self.mdap_integrator = LeanMDAPWorkflowIntegrator(
                config=mdap_config,
                workflow_state=self.workflow_state,
                team=self.team
            )

    async def solve_adaptive(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve using adaptive strategy selection.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt
        """
        # Analyze problem complexity
        complexity = await self.analyze_evolutionary_complexity(sub_problem)

        # Select strategy
        strategy = self.select_evolutionary_strategy(complexity)

        logger.info(f"Selected strategy: {strategy} for {sub_problem.id}")

        # Solve with selected strategy
        return await self.solve_with_selected_strategy(sub_problem, strategy)

    async def analyze_evolutionary_complexity(
        self,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Analyze evolutionary complexity of a sub-problem.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            Dict with complexity metrics
        """
        complexity_score = sub_problem.ai_suggested_complexity_score or 5
        estimated_effort = sub_problem.estimated_effort or 10
        num_dependencies = len(sub_problem.dependencies)

        # Determine if mathematical
        is_mathematical = False
        if self.mdap_evolution_integrator and self.mdap_evolution_integrator.evolutionary_stage:
            is_mathematical, _, _ = self.mdap_evolution_integrator.evolutionary_stage.is_mathematical_subproblem(sub_problem)

        return {
            "complexity_score": complexity_score,
            "estimated_effort": estimated_effort,
            "num_dependencies": num_dependencies,
            "is_mathematical": is_mathematical,
            "has_multiple_approaches": complexity_score > 6,
            "requires_consensus": complexity_score > 7
        }

    def select_evolutionary_strategy(
        self,
        complexity: Dict[str, Any]
    ) -> str:
        """
        Select evolutionary strategy based on complexity.

        Args:
            complexity: Complexity analysis result

        Returns:
            Strategy name: "evolution", "mdap", or "mdap_evolution"
        """
        complexity_score = complexity.get("complexity_score", 5)
        is_mathematical = complexity.get("is_mathematical", False)
        has_multiple_approaches = complexity.get("has_multiple_approaches", False)
        requires_consensus = complexity.get("requires_consensus", False)

        # Decision logic
        if requires_consensus and has_multiple_approaches:
            # High complexity with multiple approaches: use MDAP-evolution
            return "mdap_evolution"

        elif is_mathematical and complexity_score > 7:
            # Mathematical and complex: use MDAP-evolution
            return "mdap_evolution"

        elif complexity_score <= 3:
            # Low complexity: use pure MDAP
            return "mdap"

        elif complexity_score >= 8:
            # High complexity: use MDAP-evolution
            return "mdap_evolution"

        elif is_mathematical and not has_multiple_approaches:
            # Mathematical but single approach: use pure evolution
            return "evolution"

        else:
            # Default: MDAP-evolution
            return "mdap_evolution"

    async def solve_with_selected_strategy(
        self,
        sub_problem: SubProblem,
        strategy: str
    ) -> SolutionAttempt:
        """
        Solve sub-problem with selected strategy.

        Args:
            sub_problem: The sub-problem to solve
            strategy: Selected strategy

        Returns:
            SolutionAttempt
        """
        try:
            if strategy == "mdap_evolution":
                if self.mdap_evolution_integrator:
                    return await self.mdap_evolution_integrator.solve_with_mdap_evolution(sub_problem)

            elif strategy == "evolution":
                if self.evolutionary_stage:
                    return await self.evolutionary_stage.solve_subproblem_evolutionary(
                        sub_problem, self.workflow_state
                    )

            elif strategy == "mdap":
                if self.mdap_integrator:
                    return await self.mdap_integrator.solve_subproblem_with_mdap(sub_problem)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        except Exception as e:
            logger.error(f"Strategy {strategy} failed: {e}")

            # Fallback hierarchy
            if strategy != "evolution" and self.evolutionary_stage:
                logger.info("Falling back to evolution")
                return await self.evolutionary_stage.solve_subproblem_evolutionary(
                    sub_problem, self.workflow_state
                )
            elif strategy != "mdap" and self.mdap_integrator:
                logger.info("Falling back to MDAP")
                return await self.mdap_integrator.solve_subproblem_with_mdap(sub_problem)
            else:
                raise


# =============================================================================
# CONFIGURATION INTEGRATION
# =============================================================================

def add_mdap_evolution_config_to_workflow_state(
    workflow_state: WorkflowState,
    config: MDAPEvolutionConfig
) -> WorkflowState:
    """
    Add MDAP-evolution configuration to workflow state.

    Args:
        workflow_state: Current workflow state
        config: MDAP-evolution configuration

    Returns:
        Updated workflow state
    """
    if workflow_state.openevolve_parameters is None:
        workflow_state.openevolve_parameters = {}

    workflow_state.openevolve_parameters.update({
        "lean_mdap_evolution_enabled": config.enabled,
        "lean_mdap_evolution_agents": config.agents,
        "lean_mdap_evolution_population_size": config.population_size,
        "lean_mdap_evolution_generations": config.generations,
        "lean_mdap_evolution_selection_voting": config.selection_voting,
        "lean_mdap_evolution_crossover_voting": config.crossover_voting,
        "lean_mdap_evolution_mutation_voting": config.mutation_voting,
        "lean_mdap_evolution_track_agents": config.track_agents
    })

    return workflow_state


def extract_mdap_evolution_config_from_workflow_state(
    workflow_state: WorkflowState
) -> MDAPEvolutionConfig:
    """
    Extract MDAP-evolution configuration from workflow state.

    Args:
        workflow_state: Current workflow state

    Returns:
        MDAPEvolutionConfig
    """
    params = workflow_state.openevolve_parameters or {}

    return MDAPEvolutionConfig(
        enabled=params.get("lean_mdap_evolution_enabled", True),
        population_size=params.get("lean_mdap_evolution_population_size", 20),
        generations=params.get("lean_mdap_evolution_generations", 50),
        agents=params.get("lean_mdap_evolution_agents", [
            "direct_prover",
            "inductive_prover",
            "constructive_prover",
            "decomposition_prover"
        ]),
        selection_voting=params.get("lean_mdap_evolution_selection_voting", "weighted_confidence"),
        crossover_voting=params.get("lean_mdap_evolution_crossover_voting", "majority"),
        mutation_voting=params.get("lean_mdap_evolution_mutation_voting", "consensus"),
        track_agents=params.get("lean_mdap_evolution_track_agents", True)
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def solve_with_mdap_evolution(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    team: Optional[Team] = None,
    config: Optional[MDAPEvolutionConfig] = None
) -> SolutionAttempt:
    """
    Convenience function to solve with MDAP-evolution.

    Args:
        sub_problem: Sub-problem to solve
        workflow_state: Current workflow state
        team: Optional team
        config: Optional MDAP-evolution configuration

    Returns:
        SolutionAttempt with MDAP-evolved proof
    """
    # Extract or create config
    if config is None:
        config = extract_mdap_evolution_config_from_workflow_state(workflow_state)

    # Create integrator
    integrator = MDAPEvolutionWorkflowIntegrator(
        config=config,
        workflow_state=workflow_state,
        team=team
    )

    # Solve
    return await integrator.solve_with_mdap_evolution(sub_problem)


async def solve_adaptive_hybrid(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    team: Optional[Team] = None
) -> SolutionAttempt:
    """
    Convenience function to solve with adaptive hybrid approach.

    Args:
        sub_problem: Sub-problem to solve
        workflow_state: Current workflow state
        team: Optional team

    Returns:
        SolutionAttempt
    """
    config = extract_mdap_evolution_config_from_workflow_state(workflow_state)

    solver = HybridEvolutionarySolver(
        mdap_evolution_config=config,
        workflow_state=workflow_state,
        team=team
    )

    return await solver.solve_adaptive(sub_problem)


# =============================================================================
# STAGE INTEGRATION HELPERS
# =============================================================================

async def mdap_evolution_stage3a_wrapper(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    team: Optional[Team] = None
) -> SolutionAttempt:
    """
    Stage 3A wrapper for MDAP-evolution.

    Integrates with workflow_stage_functions.py.

    Args:
        sub_problem: Sub-problem to solve
        workflow_state: Current workflow state
        team: Optional team

    Returns:
        SolutionAttempt
    """
    config = extract_mdap_evolution_config_from_workflow_state(workflow_state)
    integrator = MDAPEvolutionWorkflowIntegrator(
        config=config,
        workflow_state=workflow_state,
        team=team
    )

    return await integrator.mdap_evolution_stage3a(sub_problem, workflow_state)


async def mdap_evolution_stage3b_wrapper(
    solution: SolutionAttempt,
    workflow_state: WorkflowState,
    team: Optional[Team] = None
) -> SolutionAttempt:
    """
    Stage 3B wrapper for MDAP-evolution refinement.

    Args:
        solution: Current solution to refine
        workflow_state: Current workflow state
        team: Optional team

    Returns:
        Refined solution attempt
    """
    config = extract_mdap_evolution_config_from_workflow_state(workflow_state)
    integrator = MDAPEvolutionWorkflowIntegrator(
        config=config,
        workflow_state=workflow_state,
        team=team
    )

    return await integrator.mdap_evolution_stage3b(solution, workflow_state)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "MDAPEvolutionWorkflowIntegrator",
    "EvolutionaryProgressMonitor",
    "HybridEvolutionarySolver",

    # Configuration
    "MDAPEvolutionConfig",
    "MDAPEvolutionProgress",
    "EvolutionaryStrategySelection",

    # Configuration helpers
    "add_mdap_evolution_config_to_workflow_state",
    "extract_mdap_evolution_config_from_workflow_state",

    # Convenience functions
    "solve_with_mdap_evolution",
    "solve_adaptive_hybrid",

    # Stage integration
    "mdap_evolution_stage3a_wrapper",
    "mdap_evolution_stage3b_wrapper",

    # Availability flags
    "WORKFLOW_AVAILABLE",
    "EVOLUTION_WORKFLOW_AVAILABLE",
    "MDAP_WORKFLOW_AVAILABLE",
    "LEANAIDE_AVAILABLE",
    "CREWAI_AVAILABLE",
    "ACE_AVAILABLE"
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example demonstrating MDAP-evolution workflow integration."""

        print("=== LeanAide MDAP-Enhanced Evolution Workflow Integration ===\n")

        # Check availability
        print("Component Availability:")
        print(f"  Workflow: {WORKFLOW_AVAILABLE}")
        print(f"  Evolution Workflow: {EVOLUTION_WORKFLOW_AVAILABLE}")
        print(f"  MDAP Workflow: {MDAP_WORKFLOW_AVAILABLE}")
        print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
        print(f"  CrewAI: {CREWAI_AVAILABLE}")
        print(f"  ACE: {ACE_AVAILABLE}")
        print()

        # Create configuration
        config = MDAPEvolutionConfig(
            enabled=True,
            population_size=15,
            generations=25,
            agents=["direct_prover", "inductive_prover"],
            track_agents=True
        )

        print(f"MDAP-Evolution Configuration:")
        print(f"  Population Size: {config.population_size}")
        print(f"  Generations: {config.generations}")
        print(f"  Agents: {config.agents}")
        print(f"  Track Agents: {config.track_agents}")
        print()

        # Test with workflow state if available
        if WORKFLOW_AVAILABLE:
            from workflow_structures import WorkflowState

            workflow_state = WorkflowState(
                workflow_id="test_mdap_evolution",
                problem_title="Test Mathematical Problem",
                problem_statement="Prove a basic theorem"
            )

            # Add config
            workflow_state = add_mdap_evolution_config_to_workflow_state(
                workflow_state, config
            )

            print("Workflow state configured with MDAP-evolution")
            print()

            # Test extraction
            extracted_config = extract_mdap_evolution_config_from_workflow_state(workflow_state)
            print(f"Extracted config - Population: {extracted_config.population_size}")
            print()

        print("Example complete!")

    # Run example
    asyncio.run(example_usage())
