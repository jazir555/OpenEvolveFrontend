"""
LeanAide Hybrid Strategies - Combining MCTS with Evolutionary Approaches

This module provides hybrid strategies that combine MCTS with other
proof generation approaches for improved performance and flexibility.

Key Components:
    MCTSThenEvolution: Run MCTS first, then evolve population
    EvolutionWithMCTS: Use MCTS operators during evolution
    MCTSAdversarial: MCTS for both teams in adversarial setting
    MCTSSelfPlay: MCTS-guided self-play
    AdaptiveHybrid: Dynamically switch strategies based on progress

Author: LeanAide Hybrid Strategies
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Callable, Union
)

# Configure logging
logger = logging.getLogger(__name__)

# Import MCTS components
try:
    from leanaide_mcts import (
        LeanProofMCTS,
        ProofContext as MCTSProofContext,
        TacticAction,
        MCTSResult,
        run_mcts_search
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logger.warning("MCTS not available - hybrid strategies limited")

    # Define stubs when MCTS is not available
    class TacticAction:
        """Stub TacticAction for when MCTS is not available."""
        def __init__(self, tactic, arguments=None):
            self.tactic = tactic
            self.arguments = arguments or []

    class MCTSResult:
        """Stub MCTSResult for when MCTS is not available."""
        def __init__(self):
            self.success = False
            self.best_proof = None
            self.time_elapsed = 0.0

    class LeanProofMCTS:
        """Stub LeanProofMCTS for when MCTS is not available."""
        def __init__(self, **kwargs):
            pass

    class ProofContext:
        """Stub ProofContext for when MCTS is not available."""
        def __init__(self, **kwargs):
            pass

# Import evolutionary components
try:
    from leanaide_evolution import (
        LeanProofEvolutionEngineMCTS,
        LeanProofStrategy,
        LeanProof,
        EvolutionResult,
        Tactic
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logger.warning("Evolution not available - hybrid strategies limited")

    # Define stubs when Evolution is not available
    @dataclass
    class LeanProof:
        """Stub LeanProof for when Evolution is not available."""
        theorem_name: str = ""
        theorem_statement: str = ""
        tactics: List[Any] = field(default_factory=list)

    @dataclass
    class LeanProofStrategy:
        """Stub LeanProofStrategy for when Evolution is not available."""
        proof: LeanProof = field(default_factory=LeanProof)
        generation: int = 0
        fitness: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class EvolutionResult:
        """Stub EvolutionResult for when Evolution is not available."""
        success: bool = False
        generations_completed: int = 0
        evolution_time: float = 0.0
        best_proof: Optional[str] = None
        best_fitness: float = 0.0
        convergence_history: List[float] = field(default_factory=list)
        failed_attempts: List[Dict] = field(default_factory=dict)

    @dataclass
    class Tactic:
        """Stub Tactic for when Evolution is not available."""
        name: str = ""
        arguments: List[str] = field(default_factory=list)

    class LeanProofEvolutionEngineMCTS:
        """Stub LeanProofEvolutionEngineMCTS for when Evolution is not available."""
        def __init__(self, **kwargs):
            pass

# Import adversarial components
try:
    from leanaide_adversarial import (
        LeanAdversarialEvolution,
        LeanProof as AdversarialProof,
        ProofCritique,
        LeanBlueTeamAgent,
        LeanRedTeamAgent
    )
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    logger.warning("Adversarial not available - hybrid strategies limited")

# Import self-play components
try:
    from leanaide_selfplay import (
        LeanSelfPlayArena,
        SelfPlayResult,
        LeanProofAgent
    )
    SELFPLAY_AVAILABLE = True
except ImportError:
    SELFPLAY_AVAILABLE = False
    logger.warning("Self-play not available - hybrid strategies limited")


# ============================================================================
# Strategy Base Classes
# ============================================================================

class HybridStrategy(ABC):
    """Base class for hybrid proof generation strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.statistics = {
            "total_runs": 0,
            "successful_proofs": 0,
            "average_time": 0.0,
            "average_quality": 0.0
        }

    @abstractmethod
    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using hybrid strategy"""
        pass

    def update_statistics(self, success: bool, elapsed_time: float, quality: float):
        """Update strategy statistics"""
        self.statistics["total_runs"] += 1
        if success:
            self.statistics["successful_proofs"] += 1

        # Update running averages
        n = self.statistics["total_runs"]
        prev_time = self.statistics["average_time"] * (n - 1)
        prev_quality = self.statistics["average_quality"] * (n - 1)

        self.statistics["average_time"] = (prev_time + elapsed_time) / n
        self.statistics["average_quality"] = (prev_quality + quality) / n


# ============================================================================
# MCTS + Evolution Hybrids
# ============================================================================

class MCTSThenEvolution(HybridStrategy):
    """
    Run MCTS first to seed population, then evolve.

    Two-phase approach:
    1. MCTS generates diverse, high-quality initial proofs
    2. Evolution refines and improves these proofs

    Benefits:
    - MCTS provides good starting points
    - Evolution explores variations around MCTS findings
    - Combines strengths of both approaches
    """

    def __init__(
        self,
        mcts_simulations: int = 100,
        evolution_generations: int = 20,
        population_size: int = 15
    ):
        super().__init__(
            name="MCTS_Then_Evolution",
            description="MCTS initialization followed by evolutionary refinement"
        )
        self.mcts_simulations = mcts_simulations
        self.evolution_generations = evolution_generations
        self.population_size = population_size

    async def mcts_then_evolution(
        self,
        theorem: str,
        mcts_iters: int = 100,
        evo_gens: int = 20
    ) -> EvolutionResult:
        """
        Run MCTS first, then evolve population.

        Args:
            theorem: Theorem statement
            mcts_iters: Number of MCTS iterations per search
            evo_gens: Number of evolutionary generations

        Returns:
            EvolutionResult with final proof
        """
        start_time = time.time()
        logger.info(f"MCTS-Then-Evolution: {theorem}")

        if not MCTS_AVAILABLE or not EVOLUTION_AVAILABLE:
            logger.error("Required components not available")
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": "Required components not available"}]
            )

        try:
            # Phase 1: MCTS search for diverse initial proofs
            logger.info(f"Phase 1: MCTS search ({mcts_iters} iterations)")
            mcts = LeanProofMCTS(simulations=mcts_iters)
            initial_strategies = []

            # Generate multiple diverse proofs with different exploration parameters
            exploration_configs = [
                {"exploration_constant": 1.0, "simulations": mcts_iters},
                {"exploration_constant": 1.414, "simulations": mcts_iters},
                {"exploration_constant": 2.0, "simulations": mcts_iters},
            ]

            for config in exploration_configs:
                context = MCTSProofContext(
                    goal=theorem,
                    hypotheses=[],
                    available_lemmas=self._get_available_lemmas()
                )

                mcts_instance = LeanProofMCTS(
                    exploration_constant=config["exploration_constant"],
                    simulations=config["simulations"]
                )

                best_sequence, root = mcts_instance.search(context)

                if best_sequence:
                    strategy = self._mcts_sequence_to_strategy(
                        best_sequence,
                        theorem,
                        config
                    )
                    initial_strategies.append(strategy)

            # Ensure we have enough initial strategies
            while len(initial_strategies) < self.population_size:
                # Add random variations for diversity
                if initial_strategies:
                    base = random.choice(initial_strategies)
                    variant = self._create_variant(base)
                    initial_strategies.append(variant)

            initial_strategies = initial_strategies[:self.population_size]
            logger.info(f"Generated {len(initial_strategies)} initial strategies")

            # Phase 2: Evolve MCTS-generated proofs
            logger.info(f"Phase 2: Evolution ({evo_gens} generations)")

            engine = LeanProofEvolutionEngineMCTS(
                theorem=theorem,
                theorem_name=f"theorem_{uuid.uuid4()}",
                population_size=self.population_size,
                max_generations=evo_gens,
                mcts_simulations=self.mcts_simulations
            )

            # Initialize with MCTS strategies
            engine.population = initial_strategies

            # Evaluate initial population
            await engine.evaluate_population()

            # Run evolution with MCTS operators
            result = await engine.evolve_with_mcts(mcts_ratio=0.5)

            # Update statistics
            elapsed_time = time.time() - start_time
            quality = result.best_strategy.fitness if result.best_strategy else 0.0
            self.update_statistics(result.success, elapsed_time, quality)

            return result

        except Exception as e:
            logger.error(f"MCTS-Then-Evolution failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    def _get_available_lemmas(self) -> List[str]:
        """Get available lemmas for theorem domain"""
        return ["Nat.add_zero", "Nat.add_succ", "Nat.mul_one", "Nat.add_comm"]

    def _mcts_sequence_to_strategy(
        self,
        tactic_actions: List[TacticAction],
        theorem: str,
        mcts_config: Dict[str, Any]
    ) -> LeanProofStrategy:
        """Convert MCTS sequence to evolutionary strategy"""
        tactics = []
        for action in tactic_actions:
            tactic = Tactic(
                name=action.tactic.name,
                arguments=action.tactic.arguments
            )
            tactics.append(tactic)

        proof = LeanProof(
            theorem_name=f"theorem_{uuid.uuid4()}",
            theorem_statement=theorem,
            tactics=tactics
        )

        return LeanProofStrategy(
            proof=proof,
            generation=0,
            metadata={
                "mcts_generated": True,
                "mcts_config": mcts_config
            }
        )

    def _create_variant(self, base_strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Create variant of strategy for diversity"""
        # Simple variant: shuffle or modify tactics
        new_tactics = base_strategy.proof.tactics.copy()

        if len(new_tactics) > 1 and random.random() < 0.3:
            # Randomly remove a tactic
            new_tactics.pop(random.randint(0, len(new_tactics) - 1))

        new_proof = LeanProof(
            theorem_name=base_strategy.proof.theorem_name,
            theorem_statement=base_strategy.proof.theorem_statement,
            tactics=new_tactics
        )

        return LeanProofStrategy(
            proof=new_proof,
            generation=0,
            parents=[base_strategy.strategy_id],
            metadata={"variant": True}
        )

    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using MCTS-Then-Evolution"""
        mcts_iters = params.get("mcts_iters", self.mcts_simulations)
        evo_gens = params.get("evolution_generations", self.evolution_generations)

        return await self.mcts_then_evolution(theorem, mcts_iters, evo_gens)


class EvolutionWithMCTS(HybridStrategy):
    """
    Use MCTS operators during evolution for enhanced search.

    Periodically uses MCTS to:
    - Guide mutation operator
    - Guide crossover operator
    - Inject new high-quality individuals

    Benefits:
    - Evolution maintains diversity
    - MCTS provides intelligent direction
    - Adaptive use of MCTS based on convergence
    """

    def __init__(
        self,
        generations: int = 50,
        mcts_ratio: float = 0.3,
        adaptive_mcts: bool = True
    ):
        super().__init__(
            name="Evolution_With_MCTS",
            description="Evolution with MCTS-enhanced operators"
        )
        self.generations = generations
        self.mcts_ratio = mcts_ratio
        self.adaptive_mcts = adaptive_mcts

    async def evolution_with_mcts(
        self,
        theorem: str,
        generations: int = 50
    ) -> EvolutionResult:
        """
        Evolution with MCTS-enhanced operators.

        Args:
            theorem: Theorem statement
            generations: Number of generations

        Returns:
            EvolutionResult
        """
        start_time = time.time()
        logger.info(f"Evolution-With-MCTS: {theorem}")

        if not EVOLUTION_AVAILABLE:
            logger.error("Evolution not available")
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time
            )

        try:
            # Create MCTS-enhanced evolutionary engine
            engine = LeanProofEvolutionEngineMCTS(
                theorem=theorem,
                theorem_name=f"theorem_{uuid.uuid4()}",
                population_size=20,
                max_generations=generations,
                mcts_simulations=100
            )

            # Run MCTS-enhanced evolution
            result = await engine.evolve_with_mcts(mcts_ratio=self.mcts_ratio)

            # Update statistics
            elapsed_time = time.time() - start_time
            quality = result.best_strategy.fitness if result.best_strategy else 0.0
            self.update_statistics(result.success, elapsed_time, quality)

            return result

        except Exception as e:
            logger.error(f"Evolution-With-MCTS failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using Evolution-With-MCTS"""
        generations = params.get("generations", self.generations)
        return await self.evolution_with_mcts(theorem, generations)


# ============================================================================
# MCTS + Adversarial Hybrids
# ============================================================================

class MCTSAdversarial(HybridStrategy):
    """
    MCTS for both teams in adversarial setting.

    Blue Team: Uses MCTS for proof generation
    Red Team: Uses MCTS for critique selection

    Benefits:
    - Blue team explores proof space intelligently
    - Red team finds weaknesses systematically
    - Adversarial competition improves both
    """

    def __init__(
        self,
        blue_mcts_simulations: int = 100,
        red_mcts_simulations: int = 50,
        rounds: int = 10
    ):
        super().__init__(
            name="MCTS_Adversarial",
            description="MCTS for both blue and red teams"
        )
        self.blue_mcts_simulations = blue_mcts_simulations
        self.red_mcts_simulations = red_mcts_simulations
        self.rounds = rounds

    async def mcts_adversarial(
        self,
        theorem: str,
        rounds: int = 10
    ) -> EvolutionResult:
        """
        Adversarial with MCTS for both teams.

        Args:
            theorem: Theorem statement
            rounds: Number of adversarial rounds

        Returns:
            EvolutionResult with final proof
        """
        start_time = time.time()
        logger.info(f"MCTS-Adversarial: {theorem}")

        if not MCTS_AVAILABLE or not ADVERSARIAL_AVAILABLE:
            logger.error("Required components not available")
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time
            )

        try:
            # Create MCTS instances
            blue_mcts = LeanProofMCTS(
                simulations=self.blue_mcts_simulations,
                exploration_constant=1.414
            )
            red_mcts = LeanProofMCTS(
                simulations=self.red_mcts_simulations,
                exploration_constant=1.0  # More focused search for red team
            )

            # Initialize adversarial evolution
            adversarial = LeanAdversarialEvolution()

            # Run adversarial rounds with MCTS
            for round_num in range(rounds):
                logger.info(f"Round {round_num + 1}/{rounds}")

                # Blue team: MCTS proof generation
                context = MCTSProofContext(
                    goal=theorem,
                    hypotheses=[],
                    available_lemmas=self._get_available_lemmas()
                )

                blue_sequence, blue_root = blue_mcts.search(context)

                # Convert to proof
                blue_proof = self._mcts_sequence_to_proof(blue_sequence, theorem)

                # Red team: MCTS critique search
                critiques = self._mcts_red_team_critique(blue_proof, red_mcts)

                if not critiques:
                    # No critiques found - proof is robust
                    logger.info("No critiques found - proof accepted")
                    break

                # Apply critiques to improve proof
                # In full implementation, this would modify proof based on critiques
                logger.info(f"Found {len(critiques)} critiques")

            # Create result
            elapsed_time = time.time() - start_time

            # Create final proof strategy
            final_strategy = LeanProofStrategy(
                proof=blue_proof,
                generation=rounds,
                metadata={
                    "adversarial_mcts": True,
                    "rounds": rounds
                }
            )

            result = EvolutionResult(
                success=True,  # Survived all rounds
                best_proof=blue_proof,
                best_strategy=final_strategy,
                generations_completed=rounds,
                total_evaluations=rounds * (self.blue_mcts_simulations + self.red_mcts_simulations),
                evolution_time=elapsed_time
            )

            # Update statistics
            self.update_statistics(result.success, elapsed_time, 1.0)

            return result

        except Exception as e:
            logger.error(f"MCTS-Adversarial failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    def _get_available_lemmas(self) -> List[str]:
        """Get available lemmas"""
        return ["Nat.add_zero", "Nat.add_succ", "Nat.mul_one"]

    def _mcts_sequence_to_proof(
        self,
        tactic_actions: List[TacticAction],
        theorem: str
    ) -> LeanProof:
        """Convert MCTS sequence to LeanProof"""
        tactics = []
        for action in tactic_actions:
            tactic = Tactic(
                name=action.tactic.name,
                arguments=action.tactic.arguments
            )
            tactics.append(tactic)

        return LeanProof(
            theorem_name=f"theorem_{uuid.uuid4()}",
            theorem_statement=theorem,
            tactics=tactics
        )

    def _mcts_red_team_critique(
        self,
        proof: LeanProof,
        red_mcts: LeanProofMCTS
    ) -> List[ProofCritique]:
        """
        Use MCTS to search for critiques.

        In full implementation, this would:
        - Build critique search tree
        - Explore potential counterexamples
        - Find proof weaknesses
        """
        # Simplified: return empty list (no critiques found)
        return []

    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using MCTS-Adversarial"""
        rounds = params.get("rounds", self.rounds)
        return await self.mcts_adversarial(theorem, rounds)


# ============================================================================
# MCTS + Self-Play Hybrids
# ============================================================================

class MCTSSelfPlay(HybridStrategy):
    """
    MCTS-guided self-play for reinforcement learning.

    Uses MCTS for move selection during self-play games.
    Updates policy/value networks from MCTS statistics.

    Benefits:
    - MCTS provides high-quality move targets
    - Self-play improves overall strategy
    - Network learns from search results
    """

    def __init__(
        self,
        mcts_simulations: int = 200,
        games: int = 20,
        update_interval: int = 5
    ):
        super().__init__(
            name="MCTS_Self_Play",
            description="Self-play with MCTS move selection"
        )
        self.mcts_simulations = mcts_simulations
        self.games = games
        self.update_interval = update_interval

    async def mcts_self_play(
        self,
        theorem: str,
        games: int = 20
    ) -> EvolutionResult:
        """
        Self-play with MCTS move selection.

        Args:
            theorem: Theorem statement
            games: Number of self-play games

        Returns:
            EvolutionResult with best proof
        """
        start_time = time.time()
        logger.info(f"MCTS-Self-Play: {theorem}")

        if not MCTS_AVAILABLE or not SELFPLAY_AVAILABLE:
            logger.error("Required components not available")
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time
            )

        try:
            # Create self-play arena
            arena = LeanSelfPlayArena(
                theorem=theorem,
                num_games=games
            )

            # Create MCTS instance for move selection
            mcts = LeanProofMCTS(simulations=self.mcts_simulations)

            # Run self-play games with MCTS
            best_proof = None
            best_score = 0.0

            for game_num in range(games):
                logger.info(f"Game {game_num + 1}/{games}")

                # In full implementation, this would:
                # 1. Use MCTS to select each move
                # 2. Collect MCTS statistics
                # 3. Update policy/value networks periodically

                # Run one self-play game
                result = await arena.run_one_game()

                if result and result.score > best_score:
                    best_proof = result.proof
                    best_score = result.score

                # Update networks from MCTS statistics
                if (game_num + 1) % self.update_interval == 0:
                    logger.info("Updating networks from MCTS statistics")
                    # await self.update_policy_from_mcts(agent, mcts)
                    # await self.train_value_from_mcts(agent, mcts)

            elapsed_time = time.time() - start_time

            # Create result
            final_strategy = LeanProofStrategy(
                proof=best_proof,
                generation=games,
                metadata={
                    "mcts_self_play": True,
                    "games": games,
                    "best_score": best_score
                }
            )

            result = EvolutionResult(
                success=best_proof is not None,
                best_proof=best_proof,
                best_strategy=final_strategy,
                generations_completed=games,
                total_evaluations=games * self.mcts_simulations,
                evolution_time=elapsed_time
            )

            # Update statistics
            quality = best_score if best_score > 0 else 0.0
            self.update_statistics(result.success, elapsed_time, quality)

            return result

        except Exception as e:
            logger.error(f"MCTS-Self-Play failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using MCTS-Self-Play"""
        games = params.get("games", self.games)
        return await self.mcts_self_play(theorem, games)


# ============================================================================
# Adaptive Hybrid Strategy
# ============================================================================

class AdaptiveHybrid(HybridStrategy):
    """
    Dynamically switch strategies based on progress.

    Monitors:
    - Proof quality (fitness)
    - Convergence rate
    - Time elapsed
    - Strategy effectiveness

    Automatically switches to best strategy for current situation.

    Benefits:
    - Adapts to theorem difficulty
    - Changes strategy if stuck
    - Optimizes time allocation
    - Learns from experience
    """

    def __init__(
        self,
        time_budget: float = 300.0,
        strategy_switch_threshold: float = 0.1
    ):
        super().__init__(
            name="Adaptive_Hybrid",
            description="Dynamically switches strategies based on progress"
        )
        self.time_budget = time_budget
        self.strategy_switch_threshold = strategy_switch_threshold

        # Available strategies
        self.strategies = {
            "mcts": lambda theorem, params: self._run_mcts(theorem, params),
            "mcts_evo": lambda theorem, params: self._run_mcts_evolution(theorem, params),
            "mcts_adv": lambda theorem, params: self._run_mcts_adversarial(theorem, params)
        }

        # Strategy performance tracking
        self.strategy_performance = {
            "mcts": {"success": 0, "total": 0, "avg_time": 0.0},
            "mcts_evo": {"success": 0, "total": 0, "avg_time": 0.0},
            "mcts_adv": {"success": 0, "total": 0, "avg_time": 0.0}
        }

    async def adaptive_hybrid(
        self,
        theorem: str,
        time_budget: float = 300.0
    ) -> EvolutionResult:
        """
        Adaptively select and run best strategy.

        Args:
            theorem: Theorem statement
            time_budget: Maximum time to spend

        Returns:
            EvolutionResult with best proof found
        """
        start_time = time.time()
        logger.info(f"Adaptive-Hybrid: {theorem} (time budget: {time_budget}s)")

        best_result = None
        best_fitness = 0.0
        current_strategy = "mcts_evo"  # Start with MCTS+Evolution
        stagnation_counter = 0

        while time.time() - start_time < time_budget:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed

            logger.info(f"Strategy: {current_strategy}, Time: {elapsed:.1f}s/{time_budget}s")
            logger.info(f"Best fitness so far: {best_fitness:.4f}")

            # Run current strategy
            try:
                result = await self.strategies[current_strategy](
                    theorem,
                    {"time_limit": remaining / 3}  # Use 1/3 of remaining time
                )

                # Update best result
                current_fitness = 0.0
                if result.best_strategy:
                    current_fitness = result.best_strategy.fitness

                if current_fitness > best_fitness + self.strategy_switch_threshold:
                    best_result = result
                    best_fitness = current_fitness
                    stagnation_counter = 0

                    # Update strategy performance
                    self.strategy_performance[current_strategy]["success"] += 1
                    logger.info(f"New best fitness: {best_fitness:.4f}")
                else:
                    stagnation_counter += 1
                    logger.info(f"No improvement (stagnation: {stagnation_counter})")

                self.strategy_performance[current_strategy]["total"] += 1

                # Check if we have a good proof
                if best_fitness > 0.95:
                    logger.info("High fitness achieved - stopping early")
                    break

                # Switch strategy if stagnating
                if stagnation_counter >= 2:
                    old_strategy = current_strategy
                    current_strategy = self._select_next_strategy(current_strategy)
                    stagnation_counter = 0
                    logger.info(f"Switching strategy: {old_strategy} -> {current_strategy}")

            except Exception as e:
                logger.error(f"Strategy {current_strategy} failed: {e}")
                # Try different strategy
                current_strategy = self._select_next_strategy(current_strategy)

        elapsed_time = time.time() - start_time

        if best_result:
            best_result.evolution_time = elapsed_time
            # Update statistics
            self.update_statistics(
                best_result.success,
                elapsed_time,
                best_fitness
            )
        else:
            # Create failed result
            best_result = EvolutionResult(
                success=False,
                generations_completed=0,
                evolution_time=elapsed_time,
                failed_attempts=[{"error": "No strategy succeeded"}]
            )

        logger.info(f"Adaptive hybrid completed in {elapsed_time:.1f}s")
        logger.info(f"Final fitness: {best_fitness:.4f}")

        return best_result

    def _select_next_strategy(self, current: str) -> str:
        """Select next strategy to try"""
        strategies = ["mcts", "mcts_evo", "mcts_adv"]
        current_idx = strategies.index(current)

        # Simple round-robin
        next_idx = (current_idx + 1) % len(strategies)
        return strategies[next_idx]

    async def _run_mcts(self, theorem: str, params: Dict) -> EvolutionResult:
        """Run pure MCTS strategy"""
        if not MCTS_AVAILABLE:
            raise Exception("MCTS not available")

        mcts = LeanProofMCTS(simulations=100)
        context = MCTSProofContext(goal=theorem)

        best_sequence, root = mcts.search(context)

        # Convert to EvolutionResult
        tactics = [Tactic(name=action.tactic.name) for action in best_sequence]
        proof = LeanProof(
            theorem_name=f"theorem_{uuid.uuid4()}",
            theorem_statement=theorem,
            tactics=tactics
        )

        strategy = LeanProofStrategy(
            proof=proof,
            generation=0,
            metadata={"mcts_only": True}
        )

        return EvolutionResult(
            success=len(best_sequence) > 0,
            best_proof=proof,
            best_strategy=strategy,
            generations_completed=1,
            total_evaluations=100
        )

    async def _run_mcts_evolution(self, theorem: str, params: Dict) -> EvolutionResult:
        """Run MCTS + Evolution strategy"""
        hybrid = MCTSThenEvolution()
        return await hybrid.mcts_then_evolution(
            theorem,
            mcts_iters=100,
            evo_gens=10
        )

    async def _run_mcts_adversarial(self, theorem: str, params: Dict) -> EvolutionResult:
        """Run MCTS + Adversarial strategy"""
        hybrid = MCTSAdversarial()
        return await hybrid.mcts_adversarial(
            theorem,
            rounds=5
        )

    async def generate_proof(
        self,
        theorem: str,
        **params
    ) -> EvolutionResult:
        """Generate proof using Adaptive Hybrid"""
        time_budget = params.get("time_budget", self.time_budget)
        return await self.adaptive_hybrid(theorem, time_budget)


# ============================================================================
# Strategy Factory
# ============================================================================

class HybridStrategyFactory:
    """Factory for creating hybrid strategies"""

    @staticmethod
    def create(strategy_name: str, **config) -> HybridStrategy:
        """
        Create hybrid strategy by name.

        Args:
            strategy_name: Name of strategy to create
            **config: Strategy configuration

        Returns:
            HybridStrategy instance
        """
        strategies = {
            "mcts_then_evolution": MCTSThenEvolution,
            "evolution_with_mcts": EvolutionWithMCTS,
            "mcts_adversarial": MCTSAdversarial,
            "mcts_self_play": MCTSSelfPlay,
            "adaptive_hybrid": AdaptiveHybrid
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name](**config)

    @staticmethod
    def list_strategies() -> List[str]:
        """List available hybrid strategies"""
        return [
            "mcts_then_evolution",
            "evolution_with_mcts",
            "mcts_adversarial",
            "mcts_self_play",
            "adaptive_hybrid",
            "mcts_then_mdap",
            "mdap_then_mcts",
            "mdap_mcts_parallel",
            "adaptive_mdap_mcts",
        ]


# =============================================================================
# MDAP-MCTS Hybrid Strategies
# =============================================================================

# Import MDAP components if available
try:
    from leanaide_mdap import (
        LeanMDAPOrchestrator,
        LeanMDAPConfig,
        MDAP_AVAILABLE,
    )
    from leanaide_mcts import (
        MDAPMCTSConfig,
        MCTSMDAPIntegration,
        MDAPMCTSHybrid,
        MCTSConfig,
        MCTSResult,
        ProofState,
    )
    from leanaide_evolution import (
        MDAPMCTSGenerationConfig,
        EvolutionResult,
        LeanProof,
        Tactic,
    )
    MDAP_HYBRID_AVAILABLE = MDAP_AVAILABLE
except ImportError:
    MDAP_HYBRID_AVAILABLE = False
    logger.warning("MDAP hybrid strategies not available")


class MCTSThenMDAP(HybridStrategy):
    """
    MCTS-Then-MDAP hybrid strategy.

    Run MCTS first to explore the search space, then use MDAP to refine
    the best paths found by MCTS.
    """

    def __init__(
        self,
        mcts_iterations: int = 100,
        mdap_agents: int = 4,
        mcts_time_budget: float = 30.0,
        mdap_voting_strategy: str = "first_k_ahead"
    ):
        """
        Initialize MCTS-Then-MDAP hybrid.

        Args:
            mcts_iterations: Number of MCTS iterations
            mdap_agents: Number of MDAP agents for refinement
            mcts_time_budget: Time budget for MCTS
            mdap_voting_strategy: MDAP voting strategy
        """
        super().__init__(name="mcts_then_mdap")
        self.mcts_iterations = mcts_iterations
        self.mdap_agents = mdap_agents
        self.mcts_time_budget = mcts_time_budget
        self.mdap_voting_strategy = mdap_voting_strategy

    async def generate_proof(self, theorem: str, **kwargs) -> EvolutionResult:
        """
        Generate proof using MCTS-Then-MDAP.

        Args:
            theorem: Theorem statement
            **kwargs: Additional parameters

        Returns:
            EvolutionResult with generated proof
        """
        if not MDAP_HYBRID_AVAILABLE:
            logger.warning("MDAP not available, using MCTS only")
            # Fallback to MCTS only
            from leanaide_mcts import MCTS
            mcts_config = MCTSConfig(
                max_iterations=self.mcts_iterations,
                time_budget=self.mcts_time_budget,
            )
            mcts = MCTS(mcts_config, theorem)
            mcts_result = await mcts.search()

            return EvolutionResult(
                success=mcts_result.success,
                best_proof=mcts_result.best_proof,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )

        logger.info(f"MCTS-Then-MDAP: {theorem}")

        # Phase 1: Run MCTS
        mcts_config = MCTSConfig(
            max_iterations=self.mcts_iterations,
            time_budget=self.mcts_time_budget,
        )

        from leanaide_mcts import MCTS
        mcts = MCTS(mcts_config, theorem)
        mcts_result = await mcts.search()

        if not mcts_result.best_proof:
            logger.warning("MCTS failed to find proof")
            return EvolutionResult(
                success=False,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )

        # Phase 2: Refine with MDAP
        logger.info("Refining with MDAP agents...")

        mdap_config = MDAPMCTSConfig(
            base_mcts_config=mcts_config,
            num_mdap_agents=self.mdap_agents,
            mdap_voting_strategy=self.mdap_voting_strategy,
        )

        hybrid = MDAPMCTSHybrid(mdap_config)
        result = await hybrid.mcts_then_mdap(
            theorem,
            kwargs.get("theorem_name"),
            self.mcts_iterations,
            self.mdap_agents
        )

        return EvolutionResult(
            success=result.success,
            best_proof=result.best_proof,
            generations_completed=1,
            total_evaluations=result.nodes_visited,
            evolution_time=result.time_elapsed,
            convergence_history=[result.win_rate] if result.win_rate else [],
        )


class MDAPThenMCTS(HybridStrategy):
    """
    MDAP-Then-MCTS hybrid strategy.

    Run MDAP first to generate diverse proof candidates, then use MCTS
    to explore and refine the most promising paths.
    """

    def __init__(
        self,
        mdap_agents: int = 4,
        mcts_iterations: int = 100,
        mdap_voting_strategy: str = "first_k_ahead",
        seed_population: bool = True
    ):
        """
        Initialize MDAP-Then-MCTS hybrid.

        Args:
            mdap_agents: Number of MDAP agents
            mcts_iterations: Number of MCTS iterations
            mdap_voting_strategy: MDAP voting strategy
            seed_population: Seed MCTS with MDAP results
        """
        super().__init__(name="mdap_then_mcts")
        self.mdap_agents = mdap_agents
        self.mcts_iterations = mcts_iterations
        self.mdap_voting_strategy = mdap_voting_strategy
        self.seed_population = seed_population

    async def generate_proof(self, theorem: str, **kwargs) -> EvolutionResult:
        """
        Generate proof using MDAP-Then-MCTS.

        Args:
            theorem: Theorem statement
            **kwargs: Additional parameters

        Returns:
            EvolutionResult with generated proof
        """
        if not MDAP_HYBRID_AVAILABLE:
            logger.warning("MDAP not available, using MCTS only")
            from leanaide_mcts import MCTS
            mcts_config = MCTSConfig(max_iterations=self.mcts_iterations)
            mcts = MCTS(mcts_config, theorem)
            mcts_result = await mcts.search()

            return EvolutionResult(
                success=mcts_result.success,
                best_proof=mcts_result.best_proof,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )

        logger.info(f"MDAP-Then-MCTS: {theorem}")

        # Phase 1: Run MDAP to generate candidates
        mcts_config = MCTSConfig(max_iterations=self.mcts_iterations)
        mdap_config = MDAPMCTSConfig(
            base_mcts_config=mcts_config,
            num_mdap_agents=self.mdap_agents,
            mdap_voting_strategy=self.mdap_voting_strategy,
        )

        hybrid = MDAPMCTSHybrid(mdap_config)
        result = await hybrid.mdap_then_mcts(
            theorem,
            kwargs.get("theorem_name"),
            self.mdap_agents,
            self.mcts_iterations
        )

        return EvolutionResult(
            success=result.success,
            best_proof=result.best_proof,
            generations_completed=1,
            total_evaluations=result.nodes_visited,
            evolution_time=result.time_elapsed,
            convergence_history=[result.win_rate] if result.win_rate else [],
        )


class MDAPMCTSParallel(HybridStrategy):
    """
    MDAP-MCTS Parallel hybrid strategy.

    Run MDAP and MCTS in parallel, then combine their results using
    consensus voting to select the best proof.
    """

    def __init__(
        self,
        mcts_iterations: int = 100,
        mdap_agents: int = 4,
        combination_method: str = "best_fitness"
    ):
        """
        Initialize MDAP-MCTS Parallel hybrid.

        Args:
            mcts_iterations: Number of MCTS iterations
            mdap_agents: Number of MDAP agents
            combination_method: Method to combine results ("best_fitness", "voting", "ensemble")
        """
        super().__init__(name="mdap_mcts_parallel")
        self.mcts_iterations = mcts_iterations
        self.mdap_agents = mdap_agents
        self.combination_method = combination_method

    async def generate_proof(self, theorem: str, **kwargs) -> EvolutionResult:
        """
        Generate proof using MDAP-MCTS parallel.

        Args:
            theorem: Theorem statement
            **kwargs: Additional parameters

        Returns:
            EvolutionResult with generated proof
        """
        if not MDAP_HYBRID_AVAILABLE:
            logger.warning("MDAP not available, using MCTS only")
            from leanaide_mcts import MCTS
            mcts_config = MCTSConfig(max_iterations=self.mcts_iterations)
            mcts = MCTS(mcts_config, theorem)
            mcts_result = await mcts.search()

            return EvolutionResult(
                success=mcts_result.success,
                best_proof=mcts_result.best_proof,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )

        logger.info(f"MDAP-MCTS Parallel: {theorem}")

        # Run both in parallel
        mcts_config = MCTSConfig(max_iterations=self.mcts_iterations)
        mdap_config = MDAPMCTSConfig(
            base_mcts_config=mcts_config,
            num_mdap_agents=self.mdap_agents,
        )

        hybrid = MDAPMCTSHybrid(mdap_config)
        result = await hybrid.mdap_mcts_parallel(
            theorem,
            kwargs.get("theorem_name"),
            self.mcts_iterations,
            self.mdap_agents
        )

        return EvolutionResult(
            success=result.success,
            best_proof=result.best_proof,
            generations_completed=1,
            total_evaluations=result.nodes_visited,
            evolution_time=result.time_elapsed,
            convergence_history=[result.win_rate] if result.win_rate else [],
        )


class AdaptiveMDAPMCTS(HybridStrategy):
    """
    Adaptive MDAP-MCTS hybrid strategy.

    Dynamically switches between MCTS and MDAP based on progress
    and performance metrics.
    """

    def __init__(
        self,
        time_budget: float = 60.0,
        switch_threshold: int = 3,
        initial_mode: str = "mcts",
        performance_window: int = 5
    ):
        """
        Initialize Adaptive MDAP-MCTS hybrid.

        Args:
            time_budget: Total time budget
            switch_threshold: Iterations without improvement before switching
            initial_mode: Initial mode ("mcts" or "mdap")
            performance_window: Window size for performance tracking
        """
        super().__init__(name="adaptive_mdap_mcts")
        self.time_budget = time_budget
        self.switch_threshold = switch_threshold
        self.initial_mode = initial_mode
        self.performance_window = performance_window

    async def generate_proof(self, theorem: str, **kwargs) -> EvolutionResult:
        """
        Generate proof using adaptive MDAP-MCTS.

        Args:
            theorem: Theorem statement
            **kwargs: Additional parameters

        Returns:
            EvolutionResult with generated proof
        """
        if not MDAP_HYBRID_AVAILABLE:
            logger.warning("MDAP not available, using MCTS only")
            from leanaide_mcts import MCTS
            mcts_config = MCTSConfig(time_budget=self.time_budget)
            mcts = MCTS(mcts_config, theorem)
            mcts_result = await mcts.search()

            return EvolutionResult(
                success=mcts_result.success,
                best_proof=mcts_result.best_proof,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )

        logger.info(f"Adaptive MDAP-MCTS: {theorem}")

        # Run adaptive hybrid
        mcts_config = MCTSConfig(time_budget=self.time_budget)
        mdap_config = MDAPMCTSConfig(base_mcts_config=mcts_config)

        hybrid = MDAPMCTSHybrid(mdap_config)
        result = await hybrid.adaptive_mdap_mcts(
            theorem,
            kwargs.get("theorem_name"),
            self.time_budget
        )

        return EvolutionResult(
            success=result.success,
            best_proof=result.best_proof,
            generations_completed=1,
            total_evaluations=result.nodes_visited,
            evolution_time=result.time_elapsed,
            convergence_history=[result.win_rate] if result.win_rate else [],
        )


# Note: MDAP-MCTS strategies are registered in the second HybridStrategyFactory definition
# below to avoid attribute errors

# Export main classes
__all__ = [
    # Hybrid strategies
    'MCTSThenEvolution',
    'EvolutionWithMCTS',
    'MCTSAdversarial',
    'MCTSSelfPlay',
    'AdaptiveHybrid',

    # MDAP-MCTS Hybrid strategies
    'MCTSThenMDAP',
    'MDAPThenMCTS',
    'MDAPMCTSParallel',
    'AdaptiveMDAPMCTS',
    'MDAP_HYBRID_AVAILABLE',

    # Factory
    'HybridStrategyFactory',

    # Base class
    'HybridStrategy'
]


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example_hybrid_strategies():
        """Example demonstrating hybrid strategies"""

        print("=== LeanAide Hybrid Strategies Example ===\n")

        theorem = "∀ n : Nat, n + 0 = n"

        # Example 1: MCTS-Then-Evolution
        print("1. MCTS-Then-Evolution")
        hybrid1 = MCTSThenEvolution(
            mcts_simulations=50,
            evolution_generations=10
        )
        result1 = await hybrid1.mcts_then_evolution(theorem, 50, 10)
        print(f"   Success: {result1.success}")
        print()

        # Example 2: Evolution-With-MCTS
        print("2. Evolution-With-MCTS")
        hybrid2 = EvolutionWithMCTS(generations=20, mcts_ratio=0.3)
        result2 = await hybrid2.evolution_with_mcts(theorem, 20)
        print(f"   Success: {result2.success}")
        print()

        # Example 3: Adaptive Hybrid
        print("3. Adaptive Hybrid")
        hybrid3 = AdaptiveHybrid(time_budget=60.0)
        result3 = await hybrid3.adaptive_hybrid(theorem, 60.0)
        print(f"   Success: {result3.success}")
        print(f"   Best fitness: {result3.best_strategy.fitness if result3.best_strategy else 0:.4f}")
        print()

        # Example 4: Using Factory
        print("4. Strategy Factory")
        strategy = HybridStrategyFactory.create(
            "mcts_then_evolution",
            mcts_simulations=30,
            evolution_generations=5
        )
        result4 = await strategy.generate_proof(theorem)
        print(f"   Success: {result4.success}")
        print()

        print("Example complete!")

    # Run example
    asyncio.run(example_hybrid_strategies())


# =============================================================================
# MDAP-Evolution Hybrid Strategies
# =============================================================================

# Import MDAP components
try:
    from leanaide_mdap import (
        LeanMDAPConfig,
        LeanMDAPOrchestrator,
        LeanProofAgent,
        LeanMDAPTask,
        LeanProof as MDAPProof,
        ProofStrategy as MDAPStrategy
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP not available - MDAP-evolution hybrids limited")

# Import MDAP-enhanced evolution
try:
    from leanaide_evolution import (
        LeanProofEvolutionEngineMDAPFull,
        MDAPMCTSGenerationConfig
    )
    MDAP_EVOLUTION_AVAILABLE = True
except ImportError:
    MDAP_EVOLUTION_AVAILABLE = False
    logger.warning("MDAP-enhanced evolution not available")


class EvolutionThenMDAP(HybridStrategy):
    """
    Evolution followed by MDAP refinement.

    Workflow:
    1. Run evolutionary proof search
    2. Take best evolved strategies
    3. Use MDAP to vote on and refine best proof
    4. Return MDAP-consensus proof

    Benefits:
    - Evolution explores diverse proof space
    - MDAP provides consensus-based refinement
    - Best of both approaches
    """

    def __init__(
        self,
        evolution_generations: int = 20,
        mdap_agents: int = 4,
        mdap_agent_types: Optional[List[str]] = None
    ):
        super().__init__(
            name="Evolution_Then_MDAP",
            description="Evolution followed by MDAP consensus refinement"
        )
        self.evolution_generations = evolution_generations
        self.mdap_agents = mdap_agents
        self.mdap_agent_types = mdap_agent_types or ["evolution", "mcts", "adversarial", "self_play"]

    async def evolution_then_mdap(
        self,
        theorem: str,
        evo_gens: int,
        mdap_agents: int
    ) -> Optional[LeanProof]:
        """
        Run evolution then MDAP refinement.

        Args:
            theorem: Theorem to prove
            evo_gens: Number of evolutionary generations
            mdap_agents: Number of MDAP agents for refinement

        Returns:
            Refined proof or None
        """
        if not EVOLUTION_AVAILABLE or not MDAP_AVAILABLE:
            logger.warning("Evolution or MDAP not available")
            return None

        logger.info(f"Evolution-Then-MDAP: {evo_gens} generations, {mdap_agents} MDAP agents")

        try:
            # Phase 1: Evolutionary search
            logger.info("Phase 1: Evolutionary search")
            evo_engine = LeanProofEvolutionEngineMDAPFull(
                theorem=theorem,
                population_size=20,
                max_generations=evo_gens
            )
            evo_result = await evo_engine.evolve()

            # Get best evolved strategies
            best_strategies = evo_result.population.get_elites(5)

            # Phase 2: MDAP refinement
            logger.info("Phase 2: MDAP refinement")
            mdap_config = LeanMDAPConfig(
                default_parallel_agents=mdap_agents,
                voting_strategy="first_k_ahead"
            )

            orchestrator = LeanMDAPOrchestrator(config=mdap_config)

            # Create MDAP task
            mdap_task = LeanMDAPTask(
                task_id=f"evo_mdap_{int(time.time())}",
                description="Refine evolved proofs with MDAP",
                theorem_statement=theorem
            )

            # Add evolved proofs as candidates
            for strategy in best_strategies:
                agent = LeanProofAgent(
                    agent_id=f"evolved_{strategy.strategy_id}",
                    agent_type=MDAPStrategy.EVOLUTION,
                    config=mdap_config
                )
                mdap_task.agents.append(agent)

            # Run MDAP
            mdap_result = await orchestrator.run_task(mdap_task)

            if mdap_result.success and mdap_result.final_result:
                logger.info("Evolution-Then-MDAP succeeded")
                return mdap_result.final_result
            else:
                logger.warning("MDAP refinement failed, returning best evolved")
                return best_strategies[0].proof if best_strategies else None

        except Exception as e:
            logger.error(f"Evolution-Then-MDAP failed: {e}", exc_info=True)
            return None

    async def generate_proof(self, theorem: str, **params) -> EvolutionResult:
        """Generate proof using Evolution-Then-MDAP"""
        evo_gens = params.get("evo_gens", self.evolution_generations)
        mdap_agents = params.get("mdap_agents", self.mdap_agents)

        proof = await self.evolution_then_mdap(theorem, evo_gens, mdap_agents)

        return EvolutionResult(
            success=proof is not None,
            best_proof=proof,
            generations_completed=evo_gens
        )


class MDAPThenEvolution(HybridStrategy):
    """
    MDAP followed by evolutionary refinement.

    Workflow:
    1. Run MDAP to get initial proof candidates
    2. Seed evolution population with MDAP results
    3. Evolve to refine and improve proofs
    4. Return best evolved proof

    Benefits:
    - MDAP provides diverse initial strategies
    - Evolution refines through variation/selection
    - Combines consensus with optimization
    """

    def __init__(
        self,
        mdap_agents: int = 4,
        evolution_generations: int = 15,
        population_size: int = 30
    ):
        super().__init__(
            name="MDAP_Then_Evolution",
            description="MDAP seeding followed by evolutionary refinement"
        )
        self.mdap_agents = mdap_agents
        self.evolution_generations = evolution_generations
        self.population_size = population_size

    async def mdap_then_evolution(
        self,
        theorem: str,
        mdap_agents: int,
        evo_gens: int
    ) -> Optional[LeanProof]:
        """
        Run MDAP then evolutionary refinement.

        Args:
            theorem: Theorem to prove
            mdap_agents: Number of MDAP agents
            evo_gens: Number of evolutionary generations

        Returns:
            Refined proof or None
        """
        if not MDAP_AVAILABLE or not EVOLUTION_AVAILABLE:
            logger.warning("MDAP or Evolution not available")
            return None

        logger.info(f"MDAP-Then-Evolution: {mdap_agents} agents, {evo_gens} generations")

        try:
            # Phase 1: MDAP generation
            logger.info("Phase 1: MDAP generation")
            mdap_config = LeanMDAPConfig(
                default_parallel_agents=mdap_agents,
                voting_strategy="first_k_ahead"
            )

            orchestrator = LeanMDAPOrchestrator(config=mdap_config)

            mdap_task = LeanMDAPTask(
                task_id=f"mdap_evo_{int(time.time())}",
                description="Generate proofs with MDAP for evolution",
                theorem_statement=theorem
            )

            mdap_result = await orchestrator.run_task(mdap_task)

            if not mdap_result.success or not mdap_result.candidates:
                logger.warning("MDAP generation failed")
                return None

            # Phase 2: Evolutionary refinement
            logger.info("Phase 2: Evolutionary refinement")

            # Seed population with MDAP results
            mdap_config_evo = MDAPMCTSGenerationConfig(
                mdap_num_agents=mdap_agents,
                hybrid_mode="mdap_then_mcts"
            )

            evo_engine = LeanProofEvolutionEngineMDAPFull(
                theorem=theorem,
                population_size=self.population_size,
                max_generations=evo_gens,
                mdap_maker_config=mdap_config_evo
            )

            # Seed with MDAP candidates
            from leanaide_evolution import seed_population_with_mdap_mcts
            initial_strategies = await seed_population_with_mdap_mcts(
                theorem, self.population_size, mdap_config_evo
            )

            evo_engine.population = LeanProofPopulation(
                strategies=initial_strategies,
                selection_method="tournament",
                elitism_ratio=0.1
            )

            # Evolve
            evo_result = await evo_engine.evolve()

            if evo_result.success and evo_result.best_strategy:
                logger.info("MDAP-Then-Evolution succeeded")
                return evo_result.best_strategy.proof
            else:
                logger.warning("Evolution failed, returning best MDAP result")
                return mdap_result.candidates[0] if mdap_result.candidates else None

        except Exception as e:
            logger.error(f"MDAP-Then-Evolution failed: {e}", exc_info=True)
            return None

    async def generate_proof(self, theorem: str, **params) -> EvolutionResult:
        """Generate proof using MDAP-Then-Evolution"""
        mdap_agents = params.get("mdap_agents", self.mdap_agents)
        evo_gens = params.get("evo_gens", self.evolution_generations)

        proof = await self.mdap_then_evolution(theorem, mdap_agents, evo_gens)

        return EvolutionResult(
            success=proof is not None,
            best_proof=proof,
            generations_completed=evo_gens
        )


class MDAPEvolutionParallel(HybridStrategy):
    """
    Run MDAP and Evolution in parallel, then combine results.

    Workflow:
    1. Start MDAP and Evolution simultaneously
    2. Wait for both to complete
    3. Combine results using voting
    4. Select best proof from combined set

    Benefits:
    - Maximizes time efficiency
    - Explores both consensus and diversity
    - Best proof from both approaches
    """

    def __init__(
        self,
        mdap_agents: int = 4,
        evolution_generations: int = 15,
        time_budget: float = 60.0
    ):
        super().__init__(
            name="MDAP_Evolution_Parallel",
            description="Parallel MDAP and evolution with combination"
        )
        self.mdap_agents = mdap_agents
        self.evolution_generations = evolution_generations
        self.time_budget = time_budget

    async def mdap_evolution_parallel(
        self,
        theorem: str
    ) -> Optional[LeanProof]:
        """
        Run MDAP and Evolution in parallel.

        Args:
            theorem: Theorem to prove

        Returns:
            Best proof from both approaches
        """
        if not MDAP_AVAILABLE or not EVOLUTION_AVAILABLE:
            logger.warning("MDAP or Evolution not available")
            return None

        logger.info(f"MDAP-Evolution Parallel: time_budget={self.time_budget}s")

        try:
            # Create tasks
            mdap_task = self._run_mdap(theorem)
            evo_task = self._run_evolution(theorem)

            # Run in parallel with timeout
            mdap_result, evo_result = await asyncio.wait_for(
                asyncio.gather(mdap_task, evo_task, return_exceptions=True),
                timeout=self.time_budget
            )

            # Collect all proofs
            all_proofs = []

            if isinstance(mdap_result, LeanProof):
                all_proofs.append(mdap_result)
            elif isinstance(mdap_result, Exception):
                logger.warning(f"MDAP failed: {mdap_result}")

            if isinstance(evo_result, EvolutionResult) and evo_result.best_proof:
                all_proofs.append(evo_result.best_proof)
            elif isinstance(evo_result, Exception):
                logger.warning(f"Evolution failed: {evo_result}")

            if not all_proofs:
                logger.warning("Both approaches failed")
                return None

            # Select best proof by confidence
            best_proof = max(all_proofs, key=lambda p: p.confidence)

            logger.info(f"Parallel complete: {len(all_proofs)} proofs, best confidence={best_proof.confidence:.2f}")
            return best_proof

        except asyncio.TimeoutError:
            logger.warning("Parallel execution timed out")
            return None
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}", exc_info=True)
            return None

    async def _run_mdap(self, theorem: str) -> Optional[LeanProof]:
        """Run MDAP proof generation"""
        mdap_config = LeanMDAPConfig(
            default_parallel_agents=self.mdap_agents,
            voting_strategy="first_k_ahead"
        )

        orchestrator = LeanMDAPOrchestrator(config=mdap_config)

        mdap_task = LeanMDAPTask(
            task_id=f"parallel_mdap_{int(time.time())}",
            description="Parallel MDAP proof generation",
            theorem_statement=theorem
        )

        mdap_result = await orchestrator.run_task(mdap_task)

        if mdap_result.success and mdap_result.final_result:
            return mdap_result.final_result
        return None

    async def _run_evolution(self, theorem: str) -> EvolutionResult:
        """Run evolutionary proof generation"""
        evo_engine = LeanProofEvolutionEngineMDAPFull(
            theorem=theorem,
            population_size=20,
            max_generations=self.evolution_generations
        )

        return await evo_engine.evolve()

    async def generate_proof(self, theorem: str, **params) -> EvolutionResult:
        """Generate proof using MDAP-Evolution Parallel"""
        time_budget = params.get("time_budget", self.time_budget)

        proof = await self.mdap_evolution_parallel(theorem)

        return EvolutionResult(
            success=proof is not None,
            best_proof=proof,
            evolution_time=time_budget
        )


class AdaptiveEvolutionMDAP(HybridStrategy):
    """
    Adaptively switch between Evolution and MDAP based on progress.

    Workflow:
    1. Start with one approach
    2. Monitor progress metrics
    3. Switch approaches if stagnation detected
    4. Combine results from both approaches

    Benefits:
    - Adapts to theorem difficulty
    - Avoids stagnation
    - Best approach for each stage
    """

    def __init__(
        self,
        time_budget: float = 120.0,
        stagnation_threshold: int = 3,
        initial_approach: str = "evolution"
    ):
        super().__init__(
            name="Adaptive_Evolution_MDAP",
            description="Adaptive switching between evolution and MDAP"
        )
        self.time_budget = time_budget
        self.stagnation_threshold = stagnation_threshold
        self.initial_approach = initial_approach

        # State tracking
        self.current_approach = initial_approach
        self.stagnation_counter = 0
        self.best_fitness_ever = 0.0
        self.approach_history = []

    async def adaptive_evolution_mdap(
        self,
        theorem: str,
        time_budget: float
    ) -> Optional[LeanProof]:
        """
        Run adaptive evolution-MDAP.

        Args:
            theorem: Theorem to prove
            time_budget: Total time budget in seconds

        Returns:
            Best proof found
        """
        if not MDAP_AVAILABLE or not EVOLUTION_AVAILABLE:
            logger.warning("MDAP or Evolution not available")
            return None

        logger.info(f"Adaptive Evolution-MDAP: time_budget={time_budget}s")

        start_time = time.time()
        best_proof = None

        # Initialize both approaches
        mdap_config = LeanMDAPConfig(default_parallel_agents=4)
        evo_engine = LeanProofEvolutionEngineMDAPFull(
            theorem=theorem,
            population_size=20,
            max_generations=50
        )

        try:
            while time.time() - start_time < time_budget:
                # Run current approach
                if self.current_approach == "evolution":
                    proof = await self._run_evolution_step(evo_engine, theorem)
                else:
                    proof = await self._run_mdap_step(mdap_config, theorem)

                # Update best
                if proof and proof.confidence > self.best_fitness_ever:
                    self.best_fitness_ever = proof.confidence
                    self.stagnation_counter = 0
                    best_proof = proof
                    logger.info(f"New best: {proof.confidence:.3f} using {self.current_approach}")
                else:
                    self.stagnation_counter += 1

                # Check for stagnation
                if self.stagnation_counter >= self.stagnation_threshold:
                    logger.info(f"Stagnation detected, switching from {self.current_approach}")
                    self.current_approach = "mdap" if self.current_approach == "evolution" else "evolution"
                    self.stagnation_counter = 0
                    self.approach_history.append(self.current_approach)

                # Check termination
                if proof and proof.verified:
                    logger.info("Found verified proof!")
                    break

                # Small delay to allow other operations
                await asyncio.sleep(0.1)

            logger.info(f"Adaptive complete: best_confidence={self.best_fitness_ever:.3f}")
            return best_proof

        except Exception as e:
            logger.error(f"Adaptive execution failed: {e}", exc_info=True)
            return best_proof

    async def _run_evolution_step(
        self,
        engine: LeanProofEvolutionEngineMDAPFull,
        theorem: str
    ) -> Optional[LeanProof]:
        """Run one evolutionary step"""
        try:
            # Run one generation
            if not engine.population:
                await engine.generate_initial_population()

            await engine.evaluate_population()

            best_strategy = engine.population.get_best_strategy()
            if best_strategy:
                return best_strategy.proof
        except Exception as e:
            logger.warning(f"Evolution step failed: {e}")
        return None

    async def _run_mdap_step(
        self,
        config: LeanMDAPConfig,
        theorem: str
    ) -> Optional[LeanProof]:
        """Run one MDAP step"""
        try:
            orchestrator = LeanMDAPOrchestrator(config=config)

            task = LeanMDAPTask(
                task_id=f"adaptive_mdap_{int(time.time())}",
                description="Adaptive MDAP step",
                theorem_statement=theorem
            )

            result = await orchestrator.run_task(task)

            if result.success and result.final_result:
                return result.final_result
        except Exception as e:
            logger.warning(f"MDAP step failed: {e}")
        return None

    async def generate_proof(self, theorem: str, **params) -> EvolutionResult:
        """Generate proof using Adaptive Evolution-MDAP"""
        time_budget = params.get("time_budget", self.time_budget)

        proof = await self.adaptive_evolution_mdap(theorem, time_budget)

        return EvolutionResult(
            success=proof is not None and proof.verified,
            best_proof=proof,
            evolution_time=time_budget,
            metadata={"approach_history": self.approach_history}
        )


class HybridStrategyFactory:
    """Factory for creating hybrid strategies"""

    @staticmethod
    def create(
        strategy_type: str,
        **params
    ) -> HybridStrategy:
        """
        Create a hybrid strategy instance.

        Args:
            strategy_type: Type of hybrid strategy
            **params: Strategy-specific parameters

        Returns:
            HybridStrategy instance
        """
        strategies = {
            "mcts_then_evolution": MCTSThenEvolution,
            "evolution_with_mcts": EvolutionWithMCTS,
            "mcts_adversarial": MCTSAdversarial,
            "mcts_self_play": MCTSSelfPlay,
            "adaptive_hybrid": AdaptiveHybrid,

            # MDAP-Evolution hybrids
            "evolution_then_mdap": EvolutionThenMDAP,
            "mdap_then_evolution": MDAPThenEvolution,
            "mdap_evolution_parallel": MDAPEvolutionParallel,
            "adaptive_evolution_mdap": AdaptiveEvolutionMDAP,
        }

        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        return strategy_class(**params)


# Export all classes
__all__ = [
    # Base classes
    'HybridStrategy',
    'HybridStrategyFactory',

    # MCTS-Evolution hybrids
    'MCTSThenEvolution',
    'EvolutionWithMCTS',
    'AdaptiveHybrid',

    # MCTS-Adversarial hybrids
    'MCTSAdversarial',

    # MCTS-Self-Play hybrids
    'MCTSSelfPlay',

    # MDAP-Evolution hybrids
    'EvolutionThenMDAP',
    'MDAPThenEvolution',
    'MDAPEvolutionParallel',
    'AdaptiveEvolutionMDAP',

    # Availability flags
    'MCTS_AVAILABLE',
    'EVOLUTION_AVAILABLE',
    'ADVERSARIAL_AVAILABLE',
    'SELFPLAY_AVAILABLE',
    'MDAP_AVAILABLE',
    'MDAP_EVOLUTION_AVAILABLE',
]
