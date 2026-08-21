"""
Generic, provider-agnostic Hybrid MAKER strategy classes.

This module implements the *design-only* ``MAKER*Hybrid`` strategy contract
described in ``docs/architecture/HYBRID_MAKER_ARCHITECTURE.md`` and
``docs/architecture/HYBRID_MAKER_API.md`` without any hard dependency on a
specific verification backend. The shipped, fully-specialised equivalents live
in ``integrations/leanaide/`` (LeanAide-only); this module is the missing
generic core that any backend can drive.

The only required collaborators are two pluggable abstractions:

* :class:`VerificationOracle` - proves / checks a candidate solution. Inject
  LeanAide, Z3, a unit-test runner, or any fake for offline use.
* :class:`CandidateGenerator` - produces candidate solutions for a problem.

All search / evolution orchestration is generic and deterministic-friendly so
the strategies can run fully offline with injected fakes.
"""

import abc
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Verification oracle abstraction (pluggable backend)
# --------------------------------------------------------------------------- #


@dataclass
class VerificationResult:
    """Outcome of checking one candidate against a verification oracle."""

    success: bool
    score: float = 0.0
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VerificationOracle(abc.ABC):
    """Pluggable proof / verification backend.

    Subclasses wrap a concrete prover (LeanAide, Z3, an SMT solver, a unit-test
    harness, ...). The generic strategies only ever talk to this interface, so
    no backend is imported at module load time.
    """

    name: str = "base"
    available: bool = True

    @abc.abstractmethod
    def verify(
        self, problem: str, candidate: str, **kwargs: Any
    ) -> VerificationResult:
        """Verify ``candidate`` against ``problem``.

        Returns a :class:`VerificationResult`. Implementations must be
        side-effect free with respect to the strategy state.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Candidate generator abstraction (pluggable search front-end)
# --------------------------------------------------------------------------- #


class CandidateGenerator(abc.ABC):
    """Pluggable producer of candidate solutions for a problem."""

    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        problem: str,
        n: int = 1,
        population: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Return ``n`` candidate solutions (proofs / programs) for ``problem``."""
        raise NotImplementedError


class DefaultCandidateGenerator(CandidateGenerator):
    """Dependency-free candidate generator used when none is injected.

    It builds distinct candidate strings by combining the problem statement
    with a small pool of tactic templates. It is deterministic given a seed so
    the strategies remain testable offline without an LLM or external service.
    """

    name = "default"

    _TACTICS = [
        "intro h",
        "induction",
        "simp",
        "rw [add_comm]",
        "apply assoc",
        "exact h",
        "cases h",
        "refl",
    ]

    def __init__(self, seed: Optional[int] = None, tactic_pool: Optional[List[str]] = None):
        self._rng = random.Random(seed)
        self._tactics = tactic_pool or list(self._TACTICS)

    def generate(
        self,
        problem: str,
        n: int = 1,
        population: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if population:
            base = population[0]
        else:
            base = f"theorem {problem}"
        out: List[str] = []
        for i in range(max(1, n)):
            chosen = self._rng.sample(self._tactics, k=min(3, len(self._tactics)))
            out.append(f"{base}\nby\n  " + " ; ".join(chosen) + f"  -- gen{i}")
        return out


# --------------------------------------------------------------------------- #
# Configuration and mode enums (mirror the documented contract)
# --------------------------------------------------------------------------- #


@dataclass
class MakerHybridConfig:
    """Configuration for MAKER-enhanced hybrid strategies.

    Mirrors the ``MAKERHybridConfig`` documented in HYBRID_MAKER_API.md. All
    fields are optional with sensible defaults so the strategies run offline.
    """

    # MAKER voting parameters
    enable_voting: bool = True
    voting_threshold: int = 3  # k for first-to-ahead-by-k

    # Search / evolution sizing
    mcts_simulations: int = 12
    evolution_generations: int = 4
    population_size: int = 8
    initial_candidates: int = 6

    # Adversarial parameters
    adversarial_rounds: int = 2
    red_team_size: int = 2
    blue_team_size: int = 2

    # Adaptive parameters
    adaptive_switching: bool = True
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95

    # Generic helpers
    random_seed: Optional[int] = 42
    max_oracle_calls: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_voting": self.enable_voting,
            "voting_threshold": self.voting_threshold,
            "mcts_simulations": self.mcts_simulations,
            "evolution_generations": self.evolution_generations,
            "population_size": self.population_size,
            "initial_candidates": self.initial_candidates,
            "adversarial_rounds": self.adversarial_rounds,
            "red_team_size": self.red_team_size,
            "blue_team_size": self.blue_team_size,
            "adaptive_switching": self.adaptive_switching,
            "diversity_threshold": self.diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
            "random_seed": self.random_seed,
            "max_oracle_calls": self.max_oracle_calls,
        }


class MakerHybridMode(Enum):
    """MAKER hybrid strategy modes (documented contract)."""

    MCTS_THEN_MAKER = "mcts_then_maker"
    MAKER_THEN_EVOLUTION = "maker_then_evolution"
    MAKER_ADVERSARIAL = "maker_adversarial"
    ADAPTIVE_MAKER = "adaptive_maker"
    MAKER_MDAP_PARALLEL = "maker_mdap_parallel"
    FULL_MAKER_HYBRID = "full_maker_hybrid"


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class MakerHybridResult:
    """Result returned by every :class:`MakerHybridStrategy`."""

    success: bool
    best_candidate: Optional[str] = None
    best_score: float = 0.0
    candidates: List[str] = field(default_factory=list)
    verifications: List[VerificationResult] = field(default_factory=list)
    oracle_calls: int = 0
    generations_completed: int = 0
    convergence_history: List[float] = field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    strategy_name: str = "base"
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Base strategy
# --------------------------------------------------------------------------- #


class MakerHybridStrategy(abc.ABC):
    """Base class for generic Hybrid MAKER strategies.

    Combines a :class:`CandidateGenerator` (search / evolution front-end) with a
    :class:`VerificationOracle` (proof / verification backend). Subclasses
    implement :meth:`_run` to orchestrate the two per their pattern.
    """

    mode: MakerHybridMode = MakerHybridMode.FULL_MAKER_HYBRID

    def __init__(
        self,
        oracle: VerificationOracle,
        config: Optional[MakerHybridConfig] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        name: Optional[str] = None,
    ) -> None:
        if oracle is None:
            raise ValueError("MakerHybridStrategy requires a VerificationOracle")
        self.oracle = oracle
        self.config = config or MakerHybridConfig()
        self.candidate_generator = candidate_generator or DefaultCandidateGenerator(
            seed=self.config.random_seed
        )
        self.name = name or self.__class__.__name__
        self._oracle_calls = 0

    # -- helpers ----------------------------------------------------------- #

    def _verify(self, problem: str, candidate: str, **kwargs: Any) -> VerificationResult:
        if self._oracle_calls >= self.config.max_oracle_calls:
            return VerificationResult(
                success=False, score=0.0, details="oracle call budget exhausted"
            )
        self._oracle_calls += 1
        try:
            result = self.oracle.verify(problem, candidate, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            return VerificationResult(success=False, score=0.0, details=f"oracle error: {exc}")
        return result

    def _generate(self, problem: str, n: int, population: Optional[List[str]] = None) -> List[str]:
        return self.candidate_generator.generate(problem, n=n, population=population)

    def _mutate(self, candidate: str, label: str = "mut") -> str:
        """Trivial generic mutation: append an extra tactic line."""
        return f"{candidate}\n  ; {label}: apply (by assumption)"

    def _vote(self, scored: List[Tuple[str, VerificationResult]]) -> Optional[Tuple[str, VerificationResult]]:
        """First-to-ahead-by-k style selection.

        Generic stand-in for MAKER voting: among verified candidates, a
        candidate "leads" once it has strictly more successful verifications
        than any other by ``voting_threshold``. With a single deterministic
        oracle this reduces to picking the highest-scoring successful candidate.
        """
        successes = [(c, r) for c, r in scored if r.success]
        if not successes:
            return None
        successes.sort(key=lambda t: t[1].score, reverse=True)
        return successes[0]

    # -- public API -------------------------------------------------------- #

    async def generate_proof(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        """Run the hybrid strategy against ``problem`` and return a result."""
        self._oracle_calls = 0
        start = time.perf_counter()
        try:
            result = await self._run(problem, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            result = MakerHybridResult(
                success=False,
                strategy_name=self.name,
                failed_attempts=[{"error": str(exc)}],
            )
        result.oracle_calls = self._oracle_calls
        result.strategy_name = self.name
        result.metadata.setdefault("elapsed_seconds", round(time.perf_counter() - start, 4))
        return result

    def run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        """Synchronous convenience wrapper around :meth:`generate_proof`."""
        return asyncio.run(self.generate_proof(problem, **kwargs))

    @abc.abstractmethod
    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Concrete generic variants (documented contract)
# --------------------------------------------------------------------------- #


class MCTSThenMaker(MakerHybridStrategy):
    """MCTS exploration followed by MAKER voting refinement.

    Phase 1 generates many candidates (exploration). Phase 2 verifies them and
    selects the winner via MAKER-style voting.
    """

    mode = MakerHybridMode.MCTS_THEN_MAKER

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        n_explore = kwargs.get("mcts_simulations", self.config.mcts_simulations)
        candidates = self._generate(problem, n_explore)
        scored: List[Tuple[str, VerificationResult]] = []
        for c in candidates:
            res = self._verify(problem, c)
            scored.append((c, res))
        winner = self._vote(scored)
        best_candidate, best_result = winner if winner else (None, VerificationResult(False))
        return MakerHybridResult(
            success=best_result.success,
            best_candidate=best_candidate,
            best_score=best_result.score,
            candidates=candidates,
            verifications=[r for _, r in scored],
            generations_completed=1,
            convergence_history=[best_result.score],
            failed_attempts=[] if best_result.success else [
                {"candidate": best_candidate, "details": best_result.details}
            ],
        )


class MakerThenEvolution(MakerHybridStrategy):
    """MAKER voting seeds a population, then evolution refines it."""

    mode = MakerHybridMode.MAKER_THEN_EVOLUTION

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        gens = kwargs.get("evolution_generations", self.config.evolution_generations)
        pop_size = kwargs.get("population_size", self.config.population_size)

        # Phase 1: MAKER seeds an initial population via voting.
        initial = self._generate(problem, self.config.initial_candidates)
        scored = [(c, self._verify(problem, c)) for c in initial]
        seed = self._vote(scored)
        population = [seed[0]] if seed else list(initial[:1])
        best = seed if seed else (initial[0], VerificationResult(False))

        convergence: List[float] = [best[1].score]
        all_candidates = list(initial)
        all_verifs = [r for _, r in scored]

        # Phase 2: evolve around the best candidate.
        for gen in range(gens):
            offspring = [self._mutate(best[0], label=f"g{gen}") for _ in range(pop_size)]
            all_candidates.extend(offspring)
            new_scored = [(c, self._verify(problem, c)) for c in offspring]
            all_verifs.extend(r for _, r in new_scored)
            winner = self._vote(new_scored)
            if winner and winner[1].score >= best[1].score:
                best = winner
                population = [best[0]]
            convergence.append(best[1].score)

        return MakerHybridResult(
            success=best[1].success,
            best_candidate=best[0],
            best_score=best[1].score,
            candidates=all_candidates,
            verifications=all_verifs,
            generations_completed=gens + 1,
            convergence_history=convergence,
            failed_attempts=[] if best[1].success else [
                {"candidate": best[0], "details": best[1].details}
            ],
        )


class MakerAdversarialHybrid(MakerHybridStrategy):
    """Red/blue team adversarial testing with MAKER voting selection."""

    mode = MakerHybridMode.MAKER_ADVERSARIAL

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        rounds = kwargs.get("adversarial_rounds", self.config.adversarial_rounds)
        red = kwargs.get("red_team_size", self.config.red_team_size)
        blue = kwargs.get("blue_team_size", self.config.blue_team_size)

        all_candidates: List[str] = []
        all_verifs: List[VerificationResult] = []
        best: Optional[Tuple[str, VerificationResult]] = None
        convergence: List[float] = []

        for r in range(rounds):
            red_cands = self._generate(problem, red, population=[f"{problem} :: attack{r}"])
            blue_cands = [
                self._mutate(c, label=f"def{r}") for c in red_cands[:blue]
            ] or self._generate(problem, blue, population=[f"{problem} :: defense{r}"])
            batch = red_cands + blue_cands
            all_candidates.extend(batch)
            scored = [(c, self._verify(problem, c)) for c in batch]
            all_verifs.extend(r for _, r in scored)
            winner = self._vote(scored)
            if winner and (best is None or winner[1].score >= best[1].score):
                best = winner
            convergence.append(best[1].score if best else 0.0)

        best_candidate, best_result = best if best else (None, VerificationResult(False))
        return MakerHybridResult(
            success=best_result.success,
            best_candidate=best_candidate,
            best_score=best_result.score,
            candidates=all_candidates,
            verifications=all_verifs,
            generations_completed=rounds,
            convergence_history=convergence,
            failed_attempts=[] if best_result.success else [
                {"candidate": best_candidate, "details": best_result.details}
            ],
        )


class AdaptiveMakerHybrid(MakerHybridStrategy):
    """Dynamically switches search intensity based on diversity / convergence."""

    mode = MakerHybridMode.ADAPTIVE_MAKER

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        max_gens = kwargs.get("max_generations", self.config.evolution_generations + 2)
        diversity_threshold = self.config.diversity_threshold
        convergence_threshold = self.config.convergence_threshold

        all_candidates: List[str] = []
        all_verifs: List[VerificationResult] = []
        best: Optional[Tuple[str, VerificationResult]] = None
        convergence: List[float] = []
        last_score = 0.0

        for gen in range(max_gens):
            # Adaptive intensity: low diversity -> more exploration.
            if self.config.adaptive_switching and last_score >= convergence_threshold:
                n = max(2, self.config.population_size // 2)
            else:
                n = self.config.population_size
            base = [best[0]] if best else None
            batch = self._generate(problem, n, population=base)
            all_candidates.extend(batch)
            scored = [(c, self._verify(problem, c)) for c in batch]
            all_verifs.extend(r for _, r in scored)
            winner = self._vote(scored)
            if winner and (best is None or winner[1].score >= best[1].score):
                best = winner
            last_score = best[1].score if best else 0.0
            convergence.append(last_score)
            if last_score >= convergence_threshold:
                break

        best_candidate, best_result = best if best else (None, VerificationResult(False))
        return MakerHybridResult(
            success=best_result.success,
            best_candidate=best_candidate,
            best_score=best_result.score,
            candidates=all_candidates,
            verifications=all_verifs,
            generations_completed=len(convergence),
            convergence_history=convergence,
            failed_attempts=[] if best_result.success else [
                {"candidate": best_candidate, "details": best_result.details}
            ],
        )


class MakerMDAPParallel(MakerHybridStrategy):
    """Runs two generation passes in parallel and combines by best fitness."""

    mode = MakerHybridMode.MAKER_MDAP_PARALLEL

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        agents = kwargs.get("mdap_agents", self.config.population_size)
        combination = kwargs.get("combination_method", "best_fitness")

        async def _pass(label: str, count: int) -> List[Tuple[str, VerificationResult]]:
            cands = self._generate(problem, count, population=[f"{problem} :: {label}"])
            return [(c, self._verify(problem, c)) for c in cands]

        results = await asyncio.gather(
            _pass("mdap_a", agents), _pass("mdap_b", agents)
        )
        flat = [item for sub in results for item in sub]
        all_candidates = [c for c, _ in flat]
        all_verifs = [r for _, r in flat]
        winner = self._vote(flat)
        if combination == "average":
            avg = sum(r.score for r in all_verifs) / max(1, len(all_verifs))
            best_score = avg
            best_candidate = winner[0] if winner else None
        else:
            best_candidate, best_result = winner if winner else (None, VerificationResult(False))
            best_score = best_result.score

        success = bool(winner and winner[1].success)
        return MakerHybridResult(
            success=success,
            best_candidate=best_candidate,
            best_score=best_score,
            candidates=all_candidates,
            verifications=all_verifs,
            generations_completed=2,
            convergence_history=[best_score],
            failed_attempts=[] if success else [
                {"candidate": best_candidate, "details": "no passing candidate in parallel passes"}
            ],
        )


class FullMakerHybrid(MakerHybridStrategy):
    """Runs the full pipeline (each variant in sequence) and keeps the best."""

    mode = MakerHybridMode.FULL_MAKER_HYBRID

    async def _run(self, problem: str, **kwargs: Any) -> MakerHybridResult:
        variants: List[MakerHybridStrategy] = [
            MCTSThenMaker(self.oracle, self.config, self.candidate_generator, name="MCTSThenMaker"),
            MakerThenEvolution(self.oracle, self.config, self.candidate_generator, name="MakerThenEvolution"),
            MakerAdversarialHybrid(self.oracle, self.config, self.candidate_generator, name="MakerAdversarialHybrid"),
            AdaptiveMakerHybrid(self.oracle, self.config, self.candidate_generator, name="AdaptiveMakerHybrid"),
            MakerMDAPParallel(self.oracle, self.config, self.candidate_generator, name="MakerMDAPParallel"),
        ]
        all_candidates: List[str] = []
        all_verifs: List[VerificationResult] = []
        best: Optional[Tuple[str, VerificationResult]] = None
        convergence: List[float] = []
        generations = 0

        for variant in variants:
            # Share oracle-call budget across phases.
            variant._oracle_calls = self._oracle_calls
            res = await variant.generate_proof(problem, **kwargs)
            self._oracle_calls = variant._oracle_calls
            all_candidates.extend(res.candidates)
            all_verifs.extend(res.verifications)
            generations += res.generations_completed
            if res.best_candidate is not None:
                cand_score = res.best_score
                if best is None or cand_score >= best[1].score:
                    best = (res.best_candidate, VerificationResult(res.success, cand_score))
                convergence.append(cand_score)

        best_candidate, best_result = best if best else (None, VerificationResult(False))
        return MakerHybridResult(
            success=best_result.success,
            best_candidate=best_candidate,
            best_score=best_result.score,
            candidates=all_candidates,
            verifications=all_verifs,
            generations_completed=generations,
            convergence_history=convergence,
            failed_attempts=[] if best_result.success else [
                {"candidate": best_candidate, "details": best_result.details}
            ],
        )


# --------------------------------------------------------------------------- #
# Factory + capabilities
# --------------------------------------------------------------------------- #


_STRATEGY_REGISTRY: Dict[MakerHybridMode, type] = {
    MakerHybridMode.MCTS_THEN_MAKER: MCTSThenMaker,
    MakerHybridMode.MAKER_THEN_EVOLUTION: MakerThenEvolution,
    MakerHybridMode.MAKER_ADVERSARIAL: MakerAdversarialHybrid,
    MakerHybridMode.ADAPTIVE_MAKER: AdaptiveMakerHybrid,
    MakerHybridMode.MAKER_MDAP_PARALLEL: MakerMDAPParallel,
    MakerHybridMode.FULL_MAKER_HYBRID: FullMakerHybrid,
}


def create_maker_hybrid(
    mode: MakerHybridMode,
    oracle: VerificationOracle,
    config: Optional[MakerHybridConfig] = None,
    candidate_generator: Optional[CandidateGenerator] = None,
) -> MakerHybridStrategy:
    """Build a generic hybrid MAKER strategy for ``mode`` with injected deps."""
    cls = _STRATEGY_REGISTRY.get(mode)
    if cls is None:
        raise ValueError(f"Unknown MakerHybridMode: {mode}")
    return cls(oracle, config=config, candidate_generator=candidate_generator)


def get_maker_hybrid_capabilities() -> Dict[str, Any]:
    """Describe the generic Hybrid MAKER capabilities (no backend required)."""
    return {
        "maker_hybrid_enabled": True,
        "maker_core_available": True,
        "evolution_available": True,
        "mdap_available": True,
        "mcts_available": True,
        "adversarial_available": True,
        "integration_status": "generic (provider-agnostic; backends injected)",
        "modes": [m.value for m in MakerHybridMode],
        "strategies": [c.__name__ for c in _STRATEGY_REGISTRY.values()],
        "paper": {
            "title": "Solving a Million-Step LLM Task with Zero Errors",
            "arxiv": "2511.09030",
        },
    }
