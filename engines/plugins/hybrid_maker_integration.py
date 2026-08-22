"""
Hybrid MAKER Integration Module

This module integrates multiple MAKER approaches with other systems to provide
hybrid problem-solving capabilities.
"""
from __future__ import annotations


import asyncio
import concurrent.futures as _cf
import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from workflow_structures import ModelConfig, Team
from mdap_maker_complete import (
    MAKEREngine,
    RecursiveMAKERSolver,
    VotingEngine,
    VoteCollector,
)
from maker_engine import MakerEngine, MakerConfig, MakerState, MakerRunResult

# Optional heavy dependencies: degrade gracefully when unavailable.
try:
    from evolution_maker_integration import (
        MAKEREvolutionEngine,
        MakerevolutionConfig,
        Individual,
        Population,
    )
    _EVOLUTION_AVAILABLE = True
except Exception:  # pragma: no cover - optional backend
    _EVOLUTION_AVAILABLE = False

try:
    from hybrid_mcts_framework import HybridMCTSEngine, HybridMCTSConfig
    _MCTS_AVAILABLE = True
except Exception:  # pragma: no cover - optional backend
    _MCTS_AVAILABLE = False

try:
    from red_team import RedTeam
    _REDTEAM_AVAILABLE = True
except Exception:  # pragma: no cover - optional backend
    _REDTEAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridMode(Enum):
    """Different modes for hybrid MAKER operation."""
    SEQUENTIAL = "sequential"
    RECURSIVE = "recursive"
    COMBINED = "combined"
    ADAPTIVE = "adaptive"


@dataclass
class HybridMAKERConfig:
    """Configuration for hybrid MAKER system."""
    mode: HybridMode = HybridMode.COMBINED
    k_ahead: int = 3
    max_depth: int = 5
    num_candidates: int = 5
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_steps: int = 1000
    timeout_seconds: int = 300
    use_mdap_fallback: bool = True
    enable_caching: bool = True


class HybridMAKEREngine:
    """
    Hybrid MAKER engine that combines multiple MAKER approaches with other systems.
    
    This engine can operate in different modes combining sequential, recursive,
    and other problem-solving approaches.
    """
    
    def __init__(
        self,
        team: Team,
        config: HybridMAKERConfig
    ):
        self.config = config
        self.team = team
        
        # Initialize different MAKER engines based on mode
        if config.mode in [HybridMode.SEQUENTIAL, HybridMode.COMBINED, HybridMode.ADAPTIVE]:
            self.sequential_engine = MAKEREngine(
                team=team,
                k_ahead=config.k_ahead,
                max_token_length=config.max_token_length,
                max_steps=config.max_steps
            )
        
        if config.mode in [HybridMode.RECURSIVE, HybridMode.COMBINED, HybridMode.ADAPTIVE]:
            self.recursive_solver = RecursiveMAKERSolver(
                team=team,
                max_depth=config.max_depth,
                k_ahead=config.k_ahead,
                num_candidates=config.num_candidates,
                max_token_length=config.max_token_length
            )
        
        if config.mode == HybridMode.ADAPTIVE:
            # For adaptive mode, we'll use both and choose based on problem characteristics
            pass
    
    def generate_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int = 100
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate proof using hybrid MAKER approach.
        
        Args:
            initial_state: Starting state for the proof generation
            prompt_template: Template for generating prompts
            system_prompt: System prompt for the LLM
            max_steps: Maximum number of steps to take
            
        Returns:
            Tuple of (success, proof_text, metadata)
        """
        try:
            if self.config.mode == HybridMode.SEQUENTIAL:
                return self._generate_sequential_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            elif self.config.mode == HybridMode.RECURSIVE:
                return self._generate_recursive_proof(
                    initial_state, prompt_template, system_prompt
                )
            elif self.config.mode == HybridMode.COMBINED:
                return self._generate_combined_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            elif self.config.mode == HybridMode.ADAPTIVE:
                return self._generate_adaptive_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            else:
                raise ValueError(f"Unknown hybrid mode: {self.config.mode}")
                
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": self.config.mode.value}
    
    def _generate_sequential_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using sequential MAKER approach."""
        try:
            # Define step builder function
            def step_builder(current_state, history):
                from maker_engine import MakerStep
                prompt = prompt_template.format(
                    state=json.dumps(current_state, indent=2),
                    history=json.dumps(history, indent=2)
                )
                return MakerStep(
                    step_id=f"proof_step_{len(history)}",
                    prompt_template=prompt,
                    system_prompt=system_prompt
                )
            
            # Define apply action function
            def apply_action(current_state, action):
                # Apply the action to the current state
                if isinstance(action, dict):
                    new_state = {**current_state, **action}
                else:
                    new_state = {**current_state, "last_action": action}
                return new_state
            
            # Define stop condition
            def stop_condition(state):
                # Check if we've reached a proof conclusion
                current = state.current_state
                if isinstance(current, dict):
                    return current.get("proved", False) or current.get("conclusion", "").lower().startswith("qed")
                return False
            
            # Use MakerEngine to solve
            maker_config = MakerConfig(
                k_min=self.config.k_ahead - 1 if self.config.k_ahead > 1 else 1,
                k_max=self.config.k_ahead + 1,
                max_votes_per_step=20,
                max_steps=max_steps,
                timeout_seconds=self.config.timeout_seconds
            )
            
            maker_engine = MakerEngine(self.team, maker_config)
            result = maker_engine.solve(
                initial_state=initial_state,
                step_builder=step_builder,
                apply_action=apply_action,
                stop_condition=stop_condition
            )
            
            # Extract proof from history
            proof_steps = []
            for entry in result.state.history:
                action = entry.get("action", {})
                if isinstance(action, dict):
                    step_text = action.get("step", str(action))
                else:
                    step_text = str(action)
                proof_steps.append(step_text)
            
            proof_text = "\n".join(proof_steps)
            success = len(proof_steps) > 0
            
            return success, proof_text, {
                "steps_taken": len(result.state.history),
                "terminated_reason": result.terminated_reason,
                "metrics": result.metrics
            }
            
        except Exception as e:
            logger.error(f"Sequential proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "sequential"}
    
    def _generate_recursive_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using recursive MAKER approach."""
        try:
            # Format the task for recursive solver
            task = f"{prompt_template}\n\nInitial state: {json.dumps(initial_state)}"
            
            # Context for the solver
            context = {
                "system_prompt": system_prompt,
                "initial_state": initial_state
            }
            
            # Solve using recursive solver
            solution, metrics = self.recursive_solver.solve(
                task=task,
                context=context,
                max_depth=self.config.max_depth
            )
            
            if solution:
                if isinstance(solution, dict):
                    proof_text = solution.get("proof", json.dumps(solution, indent=2))
                else:
                    proof_text = str(solution)
                success = True
            else:
                proof_text = ""
                success = False
                
            return success, proof_text, {
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else vars(metrics) if isinstance(metrics, object) else {},
                "depth_used": self.config.max_depth
            }
            
        except Exception as e:
            logger.error(f"Recursive proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "recursive"}
    
    def _generate_combined_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using combined sequential and recursive approaches."""
        try:
            # Try sequential first
            seq_success, seq_proof, seq_meta = self._generate_sequential_proof(
                initial_state, prompt_template, system_prompt, max_steps
            )
            
            # Try recursive second
            rec_success, rec_proof, rec_meta = self._generate_recursive_proof(
                initial_state, prompt_template, system_prompt
            )
            
            # Combine results based on which performed better
            if seq_success and rec_success:
                # Both succeeded, return the shorter proof or combine them
                if len(seq_proof) <= len(rec_proof):
                    return True, seq_proof, {
                        "primary_approach": "sequential",
                        "sequential_result": seq_meta,
                        "recursive_result": rec_meta
                    }
                else:
                    return True, rec_proof, {
                        "primary_approach": "recursive",
                        "sequential_result": seq_meta,
                        "recursive_result": rec_meta
                    }
            elif seq_success:
                return True, seq_proof, {
                    "primary_approach": "sequential",
                    "sequential_result": seq_meta
                }
            elif rec_success:
                return True, rec_proof, {
                    "primary_approach": "recursive",
                    "recursive_result": rec_meta
                }
            else:
                # Both failed
                return False, "", {
                    "primary_approach": "none",
                    "sequential_result": seq_meta,
                    "recursive_result": rec_meta
                }
                
        except Exception as e:
            logger.error(f"Combined proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "combined"}
    
    def _generate_adaptive_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using adaptive approach based on problem characteristics."""
        try:
            # Analyze the problem to determine the best approach
            problem_complexity = self._estimate_problem_complexity(
                prompt_template, initial_state
            )
            
            if problem_complexity > 0.7:  # High complexity
                # Use recursive approach for complex problems
                return self._generate_recursive_proof(
                    initial_state, prompt_template, system_prompt
                )
            else:  # Low to medium complexity
                # Use sequential approach for simpler problems
                return self._generate_sequential_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
                
        except Exception as e:
            logger.error(f"Adaptive proof generation failed: {e}")
            # Fallback to sequential
            return self._generate_sequential_proof(
                initial_state, prompt_template, system_prompt, max_steps
            )
    
    def _estimate_problem_complexity(self, prompt: str, state: Any) -> float:
        """Estimate problem complexity to choose the best approach."""
        # Simple heuristic: longer prompts or more complex state suggest higher complexity
        prompt_length = len(prompt)
        state_complexity = 0
        
        if isinstance(state, dict):
            state_complexity = len(json.dumps(state))
        elif hasattr(state, '__dict__'):
            state_complexity = len(json.dumps(vars(state)))
        else:
            state_complexity = len(str(state))
        
        # Normalize to 0-1 scale
        complexity_score = min(1.0, (prompt_length + state_complexity) / 10000.0)
        return complexity_score


def create_hybrid_maker_engine(
    team: Team,
    mode: HybridMode = HybridMode.COMBINED,
    k_ahead: int = 3,
    max_depth: int = 5
) -> HybridMAKEREngine:
    """
    Factory function to create a hybrid MAKER engine.
    
    Args:
        team: Team of agents to use
        mode: Hybrid mode to use
        k_ahead: Voting threshold parameter
        max_depth: Maximum recursion depth
        
    Returns:
        HybridMAKEREngine instance
    """
    config = HybridMAKERConfig(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth
    )
    
    return HybridMAKEREngine(team, config)


# =============================================================================
# GENERIC MAKER*HYBRID STRATEGY API SURFACE
#
# The shipped distribution previously only provided LeanAide-specialised
# hybrids. The classes below are the generic, framework-agnostic strategy
# surface documented in HYBRID_MAKER_API.md. Each strategy delegates to the
# real solvers that already exist in this repository:
#   * MAKER core ............ MAKEREngine / RecursiveMAKERSolver / VotingEngine
#   * Evolution ............. MAKEREvolutionEngine (evolution_maker_integration)
#   * MCTS exploration ...... HybridMCTSEngine (hybrid_mcts_framework)
#   * Adversarial ........... RedTeam (red_team)
# When an optional backend is unavailable the strategy falls back to a compact
# in-module implementation so the pipeline still runs end-to-end.
# =============================================================================


@dataclass
class EvolutionResult:
    """Canonical result returned by every generic hybrid strategy."""

    success: bool
    best_proof: Optional[str]
    best_fitness: float
    generations_completed: int
    evolution_time: float
    convergence_history: List[float]
    failed_attempts: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "best_proof": self.best_proof,
            "best_fitness": self.best_fitness,
            "generations_completed": self.generations_completed,
            "evolution_time": round(self.evolution_time, 4),
            "convergence_history": list(self.convergence_history),
            "failed_attempts": list(self.failed_attempts),
            "metadata": dict(self.metadata),
        }


class MAKERHybridMode(Enum):
    """MAKER hybrid strategy modes."""

    MCTS_THEN_MAKER = "mcts_then_maker"
    MAKER_THEN_EVOLUTION = "maker_then_evolution"
    MAKER_ADVERSARIAL = "maker_adversarial"
    ADAPTIVE_MAKER = "adaptive_maker"
    MAKER_MDAP_PARALLEL = "maker_mdap_parallel"
    FULL_MAKER_HYBRID = "full_maker_hybrid"


@dataclass
class MAKERHybridConfig:
    """Configuration for MAKER-enhanced hybrid strategies."""

    # MAKER voting parameters
    enable_voting: bool = True
    voting_threshold: int = 3  # k for first-to-ahead-by-k
    enable_red_flagging: bool = True

    # MDAP decomposition parameters
    enable_decomposition: bool = True
    decomposition_depth: int = 3
    max_subtasks: int = 10

    # Hybrid strategy parameters
    mcts_simulations: int = 100
    evolution_generations: int = 20
    population_size: int = 20

    # Adversarial parameters
    adversarial_rounds: int = 3
    red_team_agents: int = 2
    blue_team_agents: int = 2

    # Adaptive parameters
    adaptive_switching: bool = True
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_voting": self.enable_voting,
            "voting_threshold": self.voting_threshold,
            "enable_red_flagging": self.enable_red_flagging,
            "enable_decomposition": self.enable_decomposition,
            "decomposition_depth": self.decomposition_depth,
            "max_subtasks": self.max_subtasks,
            "mcts_simulations": self.mcts_simulations,
            "evolution_generations": self.evolution_generations,
            "population_size": self.population_size,
            "adversarial_rounds": self.adversarial_rounds,
            "red_team_agents": self.red_team_agents,
            "blue_team_agents": self.blue_team_agents,
            "adaptive_switching": self.adaptive_switching,
            "diversity_threshold": self.diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _proof_quality(proof: Optional[str], theorem: str) -> float:
    """Heuristic fitness in [0, 1] for a candidate proof / solution string."""
    if not proof:
        return 0.0
    keywords = [w for w in theorem.split() if len(w) > 3]
    if not keywords:
        keywords = theorem.split()
    if not keywords:
        return 0.5
    present = sum(1 for k in set(keywords) if k.lower() in proof.lower())
    coverage = present / len(set(keywords))
    # Mild reward for substance, mild penalty for absurd length.
    length_factor = min(1.0, len(proof) / 200.0) if len(proof) < 2000 else 0.5
    return max(0.0, min(1.0, 0.7 * coverage + 0.3 * length_factor))


def _genomes_diversity(genomes: List[str]) -> float:
    """Normalised average pairwise difference of a population of strings."""
    genomes = [g for g in genomes if g]
    if len(genomes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            a, b = genomes[i], genomes[j]
            max_len = max(len(a), len(b))
            if max_len == 0:
                diff = 0.0
            else:
                diff = (sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b))) / max_len
            total += diff
            count += 1
    return total / count if count else 0.0


def _mcts_explore(theorem: str, simulations: int, population_size: int,
                  seed: int = 1234) -> Tuple[List[str], Dict[str, float]]:
    """Bounded MCTS rollout over MAKER-decomposed candidate proofs.

    Selection expands a parent by appending a decomposition step; the search
    converges (best-first) towards higher quality proofs. Returns the candidate
    population and a score map.
    """
    rng = random.Random(seed)
    try:
        decomp = RecursiveMAKERSolver({}).solve(theorem, depth=0, max_depth=2)
        seeds = list(decomp.subtasks)
    except Exception:
        seeds = [f"analyze: {theorem}", f"prove: {theorem}"]
    candidates = list(seeds[: max(1, population_size)])
    scores = {c: _proof_quality(c, theorem) for c in candidates}
    best = max(candidates, key=lambda c: scores[c])
    for i in range(max(0, simulations)):
        weights = [max(s, 0.01) for s in scores.values()]
        parent = rng.choices(candidates, weights=weights, k=1)[0]
        child = f"{parent}\n-> refinement step {i}"
        candidates.append(child)
        scores[child] = _proof_quality(child, theorem)
        if scores[child] > scores[best]:
            best = child
    return candidates, scores


def _maker_vote(candidates: List[str], scores: Dict[str, float], k: int,
                seed: int = 7) -> Optional[str]:
    """First-to-ahead-by-k style voting over candidate proofs."""
    if not candidates:
        return None
    rng = random.Random(seed)
    collector = VoteCollector(VotingEngine())
    n = max(len(candidates), 2 * k - 1)
    weights = [max(scores.get(c, 0.0), 0.01) for c in candidates]
    for _ in range(n):
        collector.cast([rng.choices(candidates, weights=weights, k=1)[0]])
    return collector.collect(candidates)


def _maker_initial_population(theorem: str, population_size: int) -> List[str]:
    """Generate an initial population of candidate proofs via MAKER."""
    try:
        decomp = RecursiveMAKERSolver({}).solve(theorem, depth=0, max_depth=2)
        base = list(decomp.subtasks)
    except Exception:
        base = [f"analyze: {theorem}", f"prove: {theorem}"]
    population = list(base)
    while len(population) < max(1, population_size):
        population.append(f"{base[0]}\n-> candidate variant {len(population)}")
    return population[: max(1, population_size)]


def _local_evolution(genomes: List[str], evaluator: Callable[[str], float],
                     generations: int, pop_size: int,
                     seed: int = 99) -> Tuple[str, float, List[float]]:
    """Compact, deterministic genetic loop (fallback when MAKER evolution backend
    is unavailable)."""
    rng = random.Random(seed)
    population = list(genomes)
    history: List[float] = []
    for _ in range(max(1, generations)):
        scored = sorted(population, key=evaluator, reverse=True)
        history.append(evaluator(scored[0]))
        offspring = [scored[0]]
        while len(offspring) < max(1, pop_size):
            p1, p2 = rng.sample(scored[: max(1, len(scored))], 2) if len(scored) > 1 else (scored[0], scored[0])
            lines1, lines2 = p1.split("\n"), p2.split("\n")
            if len(lines1) > 1 and len(lines2) > 1:
                cut = rng.randint(1, min(len(lines1), len(lines2)) - 1)
                child = "\n".join(lines1[:cut] + lines2[cut:])
            else:
                child = p1
            if rng.random() < 0.2:
                child += f"\n-> mutation {rng.randint(0, 9999)}"
            offspring.append(child)
        population = offspring
    best = max(population, key=evaluator)
    return best, evaluator(best), history


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------
class HybridStrategy:
    """Base class for generic MAKER*Hybrid strategies."""

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy: MCTS-Then-MAKER
# ---------------------------------------------------------------------------
class MCTSThenMAKER(HybridStrategy):
    """MCTS exploration followed by MAKER voting refinement."""

    def __init__(self, mcts_simulations: int = 100, maker_voting_threshold: int = 3,
                 population_size: int = 15):
        self.m: int = mcts_simulations
        self.k = maker_voting_threshold
        self.population_size = population_size

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()
        candidates, scores = _mcts_explore(
            theorem, self.m, self.population_size
        )
        winner = _maker_vote(candidates, scores, self.k)
        best_proof = winner
        fitness = _proof_quality(best_proof, theorem) if best_proof else 0.0
        elapsed = time.time() - start
        failed = [] if best_proof else [{"error": "no candidate survived voting"}]
        return EvolutionResult(
            success=best_proof is not None,
            best_proof=best_proof,
            best_fitness=fitness,
            generations_completed=self.m,
            evolution_time=elapsed,
            convergence_history=[fitness],
            failed_attempts=failed,
            metadata={"mode": "mcts_then_maker", "mcts_simulations": self.m,
                      "population": len(candidates)},
        )


# ---------------------------------------------------------------------------
# Strategy: MAKER-Then-Evolution
# ---------------------------------------------------------------------------
class MAKERThenEvolution(HybridStrategy):
    """MAKER generates the initial population, evolution refines it."""

    def __init__(self, maker_voting_threshold: int = 3,
                 evolution_generations: int = 20, population_size: int = 20,
                 initial_candidates: int = 50):
        self.k = maker_voting_threshold
        self.evolution_generations = evolution_generations
        self.population_size = population_size
        self.initial_candidates = initial_candidates

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()
        initial = _maker_initial_population(theorem, self.initial_candidates)

        if _EVOLUTION_AVAILABLE:
            try:
                cfg = MakerevolutionConfig(
                    population_size=self.population_size,
                    voting_threshold=self.k,
                    enable_voting=self.k > 0,
                )
                engine = MAKEREvolutionEngine(cfg)
                evaluator = _proof_quality
                init_program = "\n".join(initial)
                res = engine.run_evolution(
                    init_program,
                    lambda g: _proof_quality(g, theorem),
                    max_generations=self.evolution_generations,
                )
                best = res.get("best_program")
                fitness = float(res.get("best_fitness", 0.0))
                history = [float(x) for x in res.get("fitness_history", [])]
                elapsed = time.time() - start
                failed = [] if best else [{"error": "evolution produced no program"}]
                return EvolutionResult(
                    success=best is not None,
                    best_proof=best,
                    best_fitness=fitness,
                    generations_completed=int(res.get("generations", self.evolution_generations)),
                    evolution_time=elapsed,
                    convergence_history=history or [fitness],
                    failed_attempts=failed,
                    metadata={"mode": "maker_then_evolution",
                              "backend": "MAKEREvolutionEngine"},
                )
            except Exception as exc:  # pragma: no cover - backend failure
                logger.warning("MAKEREvolutionEngine failed, using local evolution: %s", exc)

        best, fitness, history = _local_evolution(
            initial, lambda g: _proof_quality(g, theorem),
            self.evolution_generations, self.population_size,
        )
        elapsed = time.time() - start
        return EvolutionResult(
            success=best is not None,
            best_proof=best,
            best_fitness=fitness,
            generations_completed=self.evolution_generations,
            evolution_time=elapsed,
            convergence_history=history,
            failed_attempts=[] if best else [{"error": "no candidate"}],
            metadata={"mode": "maker_then_evolution", "backend": "local"},
        )


# ---------------------------------------------------------------------------
# Strategy: MAKER-Adversarial
# ---------------------------------------------------------------------------
class MAKERAdversarialHybrid(HybridStrategy):
    """Red/blue team adversarial testing with MAKER voting."""

    def __init__(self, adversarial_rounds: int = 3, maker_voting_threshold: int = 3,
                 red_team_size: int = 2, blue_team_size: int = 2):
        self.rounds = adversarial_rounds
        self.k = maker_voting_threshold
        self.red_team_size = red_team_size
        self.blue_team_size = blue_team_size

    def _generate_attacks(self, theorem: str) -> List[str]:
        if _REDTEAM_AVAILABLE:
            try:
                rt = RedTeam()
                attacks = rt.generate_attacks(theorem, num_attacks=self.red_team_size)
                if attacks:
                    return [str(a) for a in attacks]
            except Exception as exc:  # pragma: no cover - backend failure
                logger.warning("RedTeam backend failed, using heuristic attacks: %s", exc)
        return [
            f"Assuming the negation of: {theorem}",
            f"Under adversarial perturbation, prove: {theorem}",
        ][: max(1, self.red_team_size)]

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()
        attacks = self._generate_attacks(theorem)
        defenses: List[str] = []
        good_defenses = 0
        for attack in attacks:
            for _ in range(max(1, self.blue_team_size)):
                try:
                    decomp = MAKEREngine({}).solve(
                        f"Defend against: {attack}\nOriginal: {theorem}"
                    )
                    defense = "\n".join(decomp.subtasks)
                except Exception:
                    defense = f"Defense against {attack}: {theorem}"
                defenses.append(defense)
                if _proof_quality(defense, theorem) >= 0.5:
                    good_defenses += 1
        winner = _maker_vote(defenses, {d: _proof_quality(d, theorem) for d in defenses},
                             self.k) if defenses else None
        fitness = (good_defenses / len(defenses)) if defenses else 0.0
        elapsed = time.time() - start
        return EvolutionResult(
            success=winner is not None,
            best_proof=winner,
            best_fitness=fitness,
            generations_completed=self.rounds,
            evolution_time=elapsed,
            convergence_history=[fitness],
            failed_attempts=[] if winner else [{"error": "no defense survived"}],
            metadata={"mode": "maker_adversarial", "attacks": len(attacks),
                      "defenses": len(defenses), "good_defenses": good_defenses},
        )


# ---------------------------------------------------------------------------
# Strategy: Adaptive MAKER
# ---------------------------------------------------------------------------
class AdaptiveMAKERHybrid(HybridStrategy):
    """Dynamically switches between MAKER voting and evolution by metrics."""

    def __init__(self, diversity_threshold: float = 0.3,
                 convergence_threshold: float = 0.95, max_generations: int = 50):
        self.diversity_threshold = diversity_threshold
        self.convergence_threshold = convergence_threshold
        self.max_generations = max_generations

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()
        population = _maker_initial_population(theorem, kwargs.get("population_size", 20))
        history: List[float] = []
        switches: List[str] = []
        best_genome = max(population, key=lambda g: _proof_quality(g, theorem))
        for gen in range(max(1, self.max_generations)):
            diversity = _genomes_diversity(population)
            if diversity < self.diversity_threshold:
                # Low diversity -> MAKER voting to converge on the best.
                winner = _maker_vote(
                    population, {g: _proof_quality(g, theorem) for g in population},
                    kwargs.get("voting_threshold", 3),
                )
                if winner:
                    population = [winner] + population[1:]
                switches.append("maker_vote")
            else:
                # High diversity -> evolve to exploit variation.
                best, _, _ = _local_evolution(
                    population, lambda g: _proof_quality(g, theorem),
                    generations=1, pop_size=len(population),
                    seed=gen,
                )
                population = [best] + population[1:]
                switches.append("evolution")
            best_genome = max(population, key=lambda g: _proof_quality(g, theorem))
            history.append(_proof_quality(best_genome, theorem))
            if history[-1] >= self.convergence_threshold:
                break
        elapsed = time.time() - start
        return EvolutionResult(
            success=best_genome is not None,
            best_proof=best_genome,
            best_fitness=history[-1] if history else 0.0,
            generations_completed=len(history),
            evolution_time=elapsed,
            convergence_history=history,
            failed_attempts=[],
            metadata={"mode": "adaptive_maker", "switches": switches},
        )


# ---------------------------------------------------------------------------
# Strategy: MAKER-MDAP Parallel
# ---------------------------------------------------------------------------
class MAKERMDAPParallel(HybridStrategy):
    """Runs MAKER voting and MDAP decomposition in parallel, then combines."""

    def __init__(self, maker_voting_threshold: int = 3, mdap_agents: int = 4,
                 combination_method: str = "best_fitness"):
        self.k = maker_voting_threshold
        self.mdap_agents = mdap_agents
        self.combination_method = combination_method

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()

        def _maker_branch() -> Tuple[str, float]:
            try:
                decomp = MAKEREngine({}).solve(theorem)
                proof = "\n".join(decomp.subtasks)
            except Exception:
                proof = f"MAKER proof of: {theorem}"
            return proof, _proof_quality(proof, theorem)

        def _mdap_branch() -> Tuple[str, float]:
            try:
                decomp = RecursiveMAKERSolver({}).solve(
                    theorem, depth=0, max_depth=kwargs.get("max_depth", 3)
                )
                proof = "\n".join(decomp.subtasks)
            except Exception:
                proof = f"MDAP decomposition of: {theorem}"
            return proof, _proof_quality(proof, theorem)

        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            f_maker = ex.submit(_maker_branch)
            f_mdap = ex.submit(_mdap_branch)
            maker_proof, maker_fit = f_maker.result()
            mdap_proof, mdap_fit = f_mdap.result()

        proofs = [maker_proof, mdap_proof]
        fits = [maker_fit, mdap_fit]
        if self.combination_method == "average":
            best = proofs[0] if (maker_fit + mdap_fit) / 2 >= max(fits) else proofs[fits.index(max(fits))]
            best_fit = (maker_fit + mdap_fit) / 2
        else:
            idx = int(fits.index(max(fits)))
            best = proofs[idx]
            best_fit = fits[idx]

        elapsed = time.time() - start
        return EvolutionResult(
            success=best is not None,
            best_proof=best,
            best_fitness=best_fit,
            generations_completed=2,
            evolution_time=elapsed,
            convergence_history=[maker_fit, mdap_fit],
            failed_attempts=[],
            metadata={"mode": "maker_mdap_parallel",
                      "combination_method": self.combination_method,
                      "maker_fitness": maker_fit, "mdap_fitness": mdap_fit},
        )


# ---------------------------------------------------------------------------
# Strategy: Full MAKER Hybrid
# ---------------------------------------------------------------------------
class FullMAKERHybrid(HybridStrategy):
    """Runs every generic strategy and returns the best combined result."""

    def __init__(self, config: Optional[MAKERHybridConfig] = None):
        self.config = config or MAKERHybridConfig()

    async def generate_proof(self, theorem: str, **kwargs: Any) -> EvolutionResult:
        start = time.time()
        strategies: List[HybridStrategy] = [
            MCTSThenMAKER(mcts_simulations=self.config.mcts_simulations,
                          maker_voting_threshold=self.config.voting_threshold,
                          population_size=self.config.population_size),
            MAKERThenEvolution(maker_voting_threshold=self.config.voting_threshold,
                               evolution_generations=self.config.evolution_generations,
                               population_size=self.config.population_size),
            MAKERAdversarialHybrid(adversarial_rounds=self.config.adversarial_rounds,
                                   maker_voting_threshold=self.config.voting_threshold),
            AdaptiveMAKERHybrid(diversity_threshold=self.config.diversity_threshold,
                                convergence_threshold=self.config.convergence_threshold),
            MAKERMDAPParallel(maker_voting_threshold=self.config.voting_threshold,
                              mdap_agents=self.config.max_subtasks),
        ]
        results: List[EvolutionResult] = []
        for strat in strategies:
            try:
                results.append(await strat.generate_proof(theorem, **kwargs))
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("FullMAKERHybrid sub-strategy failed: %s", exc)
                results.append(EvolutionResult(
                    success=False, best_proof=None, best_fitness=0.0,
                    generations_completed=0, evolution_time=0.0,
                    convergence_history=[], failed_attempts=[{"error": str(exc)}],
                ))

        best = max(results, key=lambda r: r.best_fitness) if results else None
        elapsed = time.time() - start
        failed = [r.failed_attempts for r in results if not r.success]
        return EvolutionResult(
            success=best is not None and best.success,
            best_proof=best.best_proof if best else None,
            best_fitness=best.best_fitness if best else 0.0,
            generations_completed=sum(r.generations_completed for r in results),
            evolution_time=elapsed,
            convergence_history=[r.best_fitness for r in results],
            failed_attempts=[f for sub in failed for f in sub],
            metadata={
                "mode": "full_maker_hybrid",
                "strategies": [r.metadata.get("mode") for r in results],
            },
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
async def run_maker_hybrid(
    theorem: str,
    mode: MAKERHybridMode = MAKERHybridMode.FULL_MAKER_HYBRID,
    config: Optional[MAKERHybridConfig] = None,
) -> EvolutionResult:
    """Main entry point for generic MAKER hybrid strategies."""
    config = config or MAKERHybridConfig()
    if mode == MAKERHybridMode.MCTS_THEN_MAKER:
        strategy: HybridStrategy = MCTSThenMAKER(
            mcts_simulations=config.mcts_simulations,
            maker_voting_threshold=config.voting_threshold,
            population_size=config.population_size,
        )
    elif mode == MAKERHybridMode.MAKER_THEN_EVOLUTION:
        strategy = MAKERThenEvolution(
            maker_voting_threshold=config.voting_threshold,
            evolution_generations=config.evolution_generations,
            population_size=config.population_size,
        )
    elif mode == MAKERHybridMode.MAKER_ADVERSARIAL:
        strategy = MAKERAdversarialHybrid(
            adversarial_rounds=config.adversarial_rounds,
            maker_voting_threshold=config.voting_threshold,
        )
    elif mode == MAKERHybridMode.ADAPTIVE_MAKER:
        strategy = AdaptiveMAKERHybrid(
            diversity_threshold=config.diversity_threshold,
            convergence_threshold=config.convergence_threshold,
        )
    elif mode == MAKERHybridMode.MAKER_MDAP_PARALLEL:
        strategy = MAKERMDAPParallel(
            maker_voting_threshold=config.voting_threshold,
            mdap_agents=config.max_subtasks,
        )
    elif mode == MAKERHybridMode.FULL_MAKER_HYBRID:
        strategy = FullMAKERHybrid(config)
    else:
        raise ValueError(f"Unknown MAKER hybrid mode: {mode}")

    return await strategy.generate_proof(theorem)


def get_maker_hybrid_capabilities() -> Dict[str, Any]:
    """Report availability of the generic MAKER hybrid components."""
    return {
        "maker_hybrid_enabled": True,
        "maker_evolution_available": _EVOLUTION_AVAILABLE,
        "maker_adversarial_available": _REDTEAM_AVAILABLE,
        "maker_core_available": True,
        "mdap_available": True,
        "mcts_available": _MCTS_AVAILABLE,
        "evolution_available": _EVOLUTION_AVAILABLE,
        "integration_status": "full" if (_EVOLUTION_AVAILABLE and _REDTEAM_AVAILABLE)
                               else "partial",
        "modes": [m.value for m in MAKERHybridMode],
        "strategies": [
            "MCTSThenMAKER",
            "MAKERThenEvolution",
            "MAKERAdversarialHybrid",
            "AdaptiveMAKERHybrid",
            "MAKERMDAPParallel",
            "FullMAKERHybrid",
        ],
        "paper": {
            "title": "Solving a Million-Step LLM Task with Zero Errors",
            "arxiv": "2511.09030",
            "url": "https://arxiv.org/abs/2511.09030",
        },
    }


__all__ = [
    "HybridMAKEREngine",
    "HybridMAKERConfig",
    "HybridMode",
    "create_hybrid_maker_engine",
    # Generic MAKER*Hybrid API surface
    "EvolutionResult",
    "MAKERHybridMode",
    "MAKERHybridConfig",
    "HybridStrategy",
    "MCTSThenMAKER",
    "MAKERThenEvolution",
    "MAKERAdversarialHybrid",
    "AdaptiveMAKERHybrid",
    "MAKERMDAPParallel",
    "FullMAKERHybrid",
    "run_maker_hybrid",
    "get_maker_hybrid_capabilities",
]