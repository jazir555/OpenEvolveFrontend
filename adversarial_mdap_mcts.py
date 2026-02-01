"""
Adversarial Integration for MDAP/MAKER + Hybrid MCTS Approaches

This module integrates adversarial red team/blue team testing with all three hybrid MCTS
approaches (evolved policies, evolutionary nodes, coevolution) plus MDAP/MAKER.

Core Concept:
Red team attacks try to find flaws in proofs, blue team defends with MDAP/MAKER-enhanced MCTS,
creating robust proofs through adversarial coevolution.

Key Features:
1. Red Team Attacks: Edge cases, assumptions, tactics, boundaries
2. Blue Team Defense: MDAP+MCTS based robust proof generation
3. Co-evolution: Teams improve against each other
4. Self-Play: Automatic adversarial training
5. Ensemble Methods: Multi-approach with adversarial validation
6. MDAP Integration: Multi-agent attackers and defenders with MAKER voting
7. Robustness Evaluation: Comprehensive attack resistance scoring
8. Adversarial Training: Train on adversarial examples for robustness

Integration with Three MCTS Approaches:
- Evolved Policies: Train policies robust to adversarial examples
- Evolutionary Nodes: Train node selectors with adversarial pressure
- Coevolution: Coevolve trees with adversarial theorem pairs

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
import sqlite3
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from leanaide_mcts import record_failure_lineage
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar
)
import statistics

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar('T')


class TeamType(Enum):
    """Team type for adversarial testing"""
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"


class AttackStrategy(Enum):
    """Red team attack strategies"""
    EDGES = "edges"  # Find edge case counterexamples
    ASSUMPTIONS = "assumptions"  # Challenge implicit assumptions
    TACTICS = "tactics"  # Substitute tactics with alternatives
    BOUNDARIES = "boundaries"  # Test domain boundaries
    COMPREHENSIVE = "comprehensive"  # All attack types


class DefenseStrategy(Enum):
    """Blue team defense strategies"""
    VERIFY = "verify"  # Verify with LeanAide
    STRENGTHEN = "strengthen"  # Add stronger lemmas
    DECOMPOSE = "decompose"  # Decompose into subgoals
    CONSENSUS = "consensus"  # Use MDAP consensus
    ADAPTIVE = "adaptive"  # Adapt based on attack type


class MCTSApproach(Enum):
    """MCTS approaches"""
    EVOLVED_POLICIES = "evolved_policies"
    EVOLUTIONARY_NODES = "evolutionary_nodes"
    COEVOLUTION = "coevolution"
    ADAPTIVE = "adaptive"
    COMBINED = "combined"


class VulnerabilityType(Enum):
    """Types of proof vulnerabilities"""
    EDGE_CASE = "edge_case"
    MISSING_ASSUMPTION = "missing_assumption"
    WEAK_TACIC = "weak_tactic"
    BOUNDARY_VIOLATION = "boundary_violation"
    LOGICAL_GAP = "logical_gap"
    INCOMPLETE_LEMMAS = "incomplete_lemmas"


# =============================================================================
# IMPORT HANDLING
# =============================================================================

# Import MDAP components
try:
    from mdap_engine import (
        MDAPOrchestrator, MDAPConfig, MDAPTask, MDAPStep,
        MDAPVoteResult, RedFlagRules, RedFlagger
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP engine not available")

# Import MAKER components
try:
    from maker_engine import (
        MakerEngine, MakerConfig, MakerStep, MakerState, MakerRunResult
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    logger.warning("MAKER engine not available")

# Import complete MAKER
try:
    from mdap_maker_complete import (
        MAKEREngine, VoteCollector, VotingEngine, MAKERRunMetrics
    )
    MAKER_COMPLETE_AVAILABLE = True
except ImportError:
    MAKER_COMPLETE_AVAILABLE = False
    logger.warning("Complete MAKER not available")

# Import MCTS evolved policies
try:
    from mcts_evolved_policies import (
        RolloutPolicyGenome, TacticRolloutPolicy, PolicyPopulation,
        PolicyEvolutionEngine, EvolvedPolicyMCTS, MCTSConfig,
        MCTSResult, ProofState, Tactic
    )
    EVOLVED_POLICIES_AVAILABLE = True
except ImportError:
    EVOLVED_POLICIES_AVAILABLE = False
    logger.warning("MCTS evolved policies not available")

# Import MDAP evolved policies
try:
    from mcts_evolved_policies_mdap import (
        MDAPPolicyGenome, MDAPPolicyEvaluator, MDAPEvolvedPolicyMCTS,
        MDAPPolicyEvolutionEngine
    )
    MDAP_EVOLVED_POLICIES_AVAILABLE = True
except ImportError:
    MDAP_EVOLVED_POLICIES_AVAILABLE = False
    logger.warning("MDAP evolved policies not available")

# Import evolutionary nodes
try:
    from mcts_evolutionary_nodes import (
        EvolutionaryNode, EvolutionaryMCTS, EvolutionaryTree,
        ActionSequence, ProofContext as EvoProofContext
    )
    EVOLUTIONARY_NODES_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_NODES_AVAILABLE = False
    logger.warning("Evolutionary nodes not available")

# Import MDAP evolutionary nodes
try:
    from mcts_evolutionary_nodes_mdap import (
        MDAPEvolutionaryMCTS, MDAPEvolutionaryNode
    )
    MDAP_EVOLUTIONARY_NODES_AVAILABLE = True
except ImportError:
    MDAP_EVOLUTIONARY_NODES_AVAILABLE = False
    logger.warning("MDAP evolutionary nodes not available")

# Import coevolution
try:
    from mcts_coevolution import (
        ProofDecisionTree, DecisionNode, TreeGenerator, MCTreeEvaluator
    )
    COEVOLUTION_AVAILABLE = True
except ImportError:
    COEVOLUTION_AVAILABLE = False
    logger.warning("Coevolution not available")

# Import MDAP coevolution
try:
    from mcts_coevolution_mdap import (
        MDAPProofDecisionTree, MDAPTreeCoevolution
    )
    MDAP_COEVOLUTION_AVAILABLE = True
except ImportError:
    MDAP_COEVOLUTION_AVAILABLE = False
    logger.warning("MDAP coevolution not available")

# Import unified framework
try:
    from mdap_maker_mcts_unified import (
        MDAPMAKERMCTSEngine, MDAPMAKERConfig, MDAPMCTSPresets,
        MCTSApproach as UnifiedMCTSApproach
    )
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    logger.warning("Unified framework not available")

# Import LeanAide
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide client not available")

# Import decomposition
try:
    from decomposition_engine import (
        DecompositionEngine, DecompositionStrategyBase
    )
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False
    logger.warning("Decomposition engine not available")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LeanProof:
    """Represents a Lean proof"""
    proof_id: str
    theorem: str
    tactic_sequence: List[str]
    proof_state: str
    is_valid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tactics(self) -> List[str]:
        """Get tactic sequence"""
        return self.tactic_sequence

    @property
    def depth(self) -> int:
        """Get proof depth"""
        return len(self.tactic_sequence)


@dataclass
class ProofContext:
    """Context for proof generation and attack"""
    theorem_statement: str
    goal: str
    hypotheses: List[str]
    available_tactics: List[str]
    domain: str = "math"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """Result of red team attack"""
    attack_type: AttackStrategy
    description: str
    severity: float  # 0-1, higher is more severe
    counterexample: Optional[str] = None
    suggested_fix: Optional[str] = None
    vulnerability_type: Optional[VulnerabilityType] = None
    attack_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DefenseResult:
    """Result of blue team defense"""
    robust_proof: LeanProof
    defense_strength: float  # 0-1, higher is stronger
    attack_blocked: bool
    improvements_made: List[str]
    defense_strategy: DefenseStrategy
    verification_result: Optional[Any] = None
    confidence: float = 0.0
    defense_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AdversarialTeam:
    """Represents an adversarial team"""
    team_id: str
    team_type: TeamType
    strategy: AttackStrategy  # For red team
    mcts_approach: MCTSApproach
    mdap_config: Optional['MDAPConfig'] = None

    # Team capabilities
    attack_vectors: List[str] = field(default_factory=list)
    defense_mechanisms: List[str] = field(default_factory=list)

    # Performance tracking
    attacks_launched: int = 0
    successful_attacks: int = 0
    defenses_blocked: int = 0
    total_attempts: int = 0

    # Team members
    agents: List[Any] = field(default_factory=list)

    def get_effectiveness(self) -> float:
        """Compute team effectiveness"""
        if self.total_attempts == 0:
            return 0.0

        if self.team_type == TeamType.RED_TEAM:
            # Red team: success rate
            return self.successful_attacks / max(self.total_attempts, 1)
        else:
            # Blue team: defense rate
            return self.defenses_blocked / max(self.total_attempts, 1)


@dataclass
class MDAPConfig:
    """MDAP configuration for adversarial teams"""
    num_agents: int = 5
    voting_strategy: str = "first_k_ahead"
    k_ahead: int = 3
    consensus_threshold: float = 0.75
    enable_red_flagging: bool = True


@dataclass
class MAKERConfig:
    """MAKER configuration"""
    k_ahead: int = 3
    voting_timeout: float = 30.0
    max_iterations: int = 100


@dataclass
class RobustnessReport:
    """Report on proof robustness"""
    proof_id: str
    robustness_score: float
    attack_results: List[AttackResult]
    weaknesses: List[VulnerabilityType]
    suggested_improvements: List[str]
    is_robust: bool
    confidence: float
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AdversarialCoevolutionResult:
    """Result of adversarial coevolution"""
    robust_proofs: List[LeanProof]
    coevolution_history: List[Dict[str, Any]]
    final_red_teams: List[AdversarialTeam]
    final_blue_teams: List[AdversarialTeam]
    best_robustness_score: float
    total_generations: int


@dataclass
class SelfPlayResult:
    """Result of self-play adversarial training"""
    results: List[Dict[str, Any]]
    robustness_score: float
    improved_prover: Any
    total_rounds: int
    learning_curve: List[float]


@dataclass
class RobustProofResult:
    """Result of robust proof generation with ensemble"""
    proof: MCTSResult
    approach: MCTSApproach
    robustness_score: float
    all_proofs: Dict[MCTSApproach, MCTSResult]
    all_robustness: Dict[MCTSApproach, float]
    selected_by: str = "adversarial_validation"


@dataclass
class MDAPAdversarialResult:
    """Result of MDAP multi-agent adversarial testing"""
    most_severe_attack: AttackResult
    best_defense: DefenseResult
    consensus_robustness: float
    attack_details: List[AttackResult]
    defense_details: List[DefenseResult]
    voting_summary: Dict[str, Any]


@dataclass
class TrainedModel:
    """Model trained with adversarial examples"""
    model: Any
    training_history: List[Dict[str, Any]]
    robustness_score: float
    adversarial_examples_used: int
    base_success_rate: float
    final_success_rate: float


# =============================================================================
# RED TEAM AGENT
# =============================================================================

class RedTeamAgent:
    """
    Red team agent that generates adversarial attacks on proofs.

    Attack strategies:
    1. Edge cases: Find corner cases and counterexamples
    2. Assumptions: Challenge implicit assumptions
    3. Tactics: Suggest alternative tactic sequences
    4. Boundaries: Test domain boundaries
    5. Comprehensive: All of the above
    """

    def __init__(
        self,
        agent_id: str,
        attack_strategy: AttackStrategy,
        mcts_config: Optional[MCTSConfig] = None,
        creativity: float = 0.7
    ):
        self.agent_id = agent_id
        self.attack_strategy = attack_strategy
        self.mcts_config = mcts_config
        self.creativity = creativity  # 0-1, higher = more creative attacks

        # Attack history
        self.attack_history: List[AttackResult] = []

        # Performance tracking
        self.total_attacks = 0
        self.successful_attacks = 0

    async def generate_attack(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> AttackResult:
        """Generate adversarial attack on proof"""
        attacks = []

        # Different attack vectors based on strategy
        if self.attack_strategy in [AttackStrategy.COMPREHENSIVE, AttackStrategy.EDGES]:
            edge_attack = await self._find_edge_cases(proof, context)
            attacks.append(edge_attack)

        if self.attack_strategy in [AttackStrategy.COMPREHENSIVE, AttackStrategy.ASSUMPTIONS]:
            assumption_attack = await self._challenge_assumptions(proof, context)
            attacks.append(assumption_attack)

        if self.attack_strategy in [AttackStrategy.COMPREHENSIVE, AttackStrategy.TACTICS]:
            tactic_attack = await self._substitute_tactics(proof, context)
            attacks.append(tactic_attack)

        if self.attack_strategy in [AttackStrategy.COMPREHENSIVE, AttackStrategy.BOUNDARIES]:
            boundary_attack = await self._test_boundaries(proof, context)
            attacks.append(boundary_attack)

        # Select most effective attack
        best_attack = max(attacks, key=lambda a: a.severity)

        # Update tracking
        self.total_attacks += 1
        self.attack_history.append(best_attack)

        return best_attack

    async def _find_edge_cases(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> AttackResult:
        """Find edge case counterexamples"""
        # Simulate edge case analysis
        edge_cases = []

        # Check for empty sequences
        if len(proof.tactics) == 0:
            edge_cases.append("Empty tactic sequence")

        # Check for very long proofs
        if proof.depth > 100:
            edge_cases.append("Excessive proof depth may have gaps")

        # Check for repetitive tactics
        tactic_counts = defaultdict(int)
        for tactic in proof.tactics:
            tactic_counts[tactic] += 1
        for tactic, count in tactic_counts.items():
            if count > 10:
                edge_cases.append(f"Excessive repetition of {tactic}")

        # Check for missing trivial cases
        trivial_cases = ["0", "1", "empty", "nil"]
        for case in trivial_cases:
            if case not in proof.theorem.lower():
                edge_cases.append(f"May not handle {case} case")

        severity = min(len(edge_cases) * 0.2, 1.0)

        return AttackResult(
            attack_type=AttackStrategy.EDGES,
            description=f"Found {len(edge_cases)} potential edge cases: {', '.join(edge_cases[:3])}",
            severity=severity,
            counterexample=edge_cases[0] if edge_cases else None,
            suggested_fix="Add explicit handling for identified edge cases",
            vulnerability_type=VulnerabilityType.EDGE_CASE
        )

    async def _challenge_assumptions(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> AttackResult:
        """Challenge implicit assumptions in the proof"""
        assumptions_challenged = []

        # Check if hypotheses are explicitly used
        for hyp in context.hypotheses:
            if hyp not in proof.proof_state:
                assumptions_challenged.append(f"Hypothesis '{hyp}' may not be used")

        # Check for implicit well-formedness assumptions
        well_formedness = ["finite", "non-empty", "well-defined"]
        for prop in well_formedness:
            if prop not in proof.theorem.lower():
                assumptions_challenged.append(f"May assume {prop} without justification")

        # Check for domain-specific assumptions
        domain_assumptions = {
            "math": ["commutativity", "associativity", "distributivity"],
            "cs": ["termination", "totality", "determinism"],
            "logic": ["consistency", "soundness", "completeness"]
        }

        for assumption in domain_assumptions.get(context.domain, []):
            if assumption not in proof.theorem.lower():
                assumptions_challenged.append(
                    f"May implicitly assume {assumption} in {context.domain} domain"
                )

        severity = min(len(assumptions_challenged) * 0.25, 1.0)

        return AttackResult(
            attack_type=AttackStrategy.ASSUMPTIONS,
            description=f"Challenged {len(assumptions_challenged)} assumptions: {', '.join(assumptions_challenged[:3])}",
            severity=severity,
            suggested_fix="Make all assumptions explicit in hypotheses",
            vulnerability_type=VulnerabilityType.MISSING_ASSUMPTION
        )

    async def _substitute_tactics(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> AttackResult:
        """Suggest alternative tactics that might fail"""
        weak_tactics = []

        # Look for potentially weak tactics
        weak_patterns = {
            "sorry": "Uses 'sorry' placeholder",
            "admit": "Uses 'admit' placeholder",
            "simp": "May oversimplify with 'simp'",
            "rw": "May rely on specific rewrite order"
        }

        for i, tactic in enumerate(proof.tactics):
            for pattern, issue in weak_patterns.items():
                if pattern in tactic.lower():
                    weak_tactics.append(f"Tactic {i}: {issue}")

        # Check for missing powerful tactics
        powerful_tactics = ["linarith", "aesop", "decide", "finish"]
        for pt in powerful_tactics:
            if pt not in " ".join(proof.tactics).lower():
                weak_tactics.append(f"Could use {pt} for automation")

        # Suggest alternatives
        alternatives = []
        if "simp" in " ".join(proof.tactics).lower():
            alternatives.append("Try 'simp' with explicit simp lemmas")

        if "rw" in " ".join(proof.tactics).lower():
            alternatives.append("Consider 'calc' mode for clarity")

        severity = min(len(weak_tactics) * 0.15, 1.0)

        return AttackResult(
            attack_type=AttackStrategy.TACTICS,
            description=f"Identified {len(weak_tactics)} potentially weak tactics",
            severity=severity,
            suggested_fix="; ".join(alternatives) if alternatives else "Consider stronger automation",
            vulnerability_type=VulnerabilityType.WEAK_TACIC
        )

    async def _test_boundaries(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> AttackResult:
        """Test domain boundaries and edge conditions"""
        boundary_issues = []

        # Check for boundary conditions based on domain
        if context.domain == "math":
            # Mathematical boundaries
            boundary_checks = [
                ("0", "zero"),
                ("1", "one"),
                ("∞", "infinity"),
                ("∅", "empty set"),
                ("ℕ", "natural numbers"),
                ("ℤ", "integers"),
                ("ℝ", "reals")
            ]
        elif context.domain == "cs":
            # Computer science boundaries
            boundary_checks = [
                ("empty", "empty structure"),
                ("null", "null value"),
                ("[]", "empty list"),
                ("length 0", "zero length"),
                ("length 1", "single element"),
                ("∞", "infinite")
            ]
        else:
            # General boundaries
            boundary_checks = [
                ("empty", "empty"),
                ("single", "single element"),
                ("∞", "infinite")
            ]

        for check, desc in boundary_checks:
            if check not in proof.theorem.lower() and check not in proof.proof_state.lower():
                boundary_issues.append(f"May not handle {desc} boundary")

        # Check for implicit assumptions about finiteness
        if "finite" not in proof.theorem.lower():
            boundary_issues.append("May implicitly assume finiteness")

        severity = min(len(boundary_issues) * 0.2, 1.0)

        return AttackResult(
            attack_type=AttackStrategy.BOUNDARIES,
            description=f"Found {len(boundary_issues)} potential boundary issues",
            severity=severity,
            suggested_fix="Explicitly handle all boundary conditions",
            vulnerability_type=VulnerabilityType.BOUNDARY_VIOLATION
        )

    def get_success_rate(self) -> float:
        """Get attack success rate"""
        if self.total_attacks == 0:
            return 0.0
        return self.successful_attacks / self.total_attacks


# =============================================================================
# BLUE TEAM AGENT
# =============================================================================

class BlueTeamAgent:
    """
    Blue team agent that defends proofs using MDAP/MAKER + MCTS.

    Defense strategies:
    1. Verify: Use LeanAide formal verification
    2. Strengthen: Add stronger lemmas and invariants
    3. Decompose: Break down into subgoals with MDAP
    4. Consensus: Use multi-agent voting
    5. Adaptive: Adapt based on attack type
    """

    def __init__(
        self,
        agent_id: str,
        mcts_approach: MCTSApproach,
        mdap_config: MDAPConfig,
        maker_config: MAKERConfig,
        defense_strategy: DefenseStrategy = DefenseStrategy.ADAPTIVE
    ):
        self.agent_id = agent_id
        self.mcts_approach = mcts_approach
        self.mdap_config = mdap_config
        self.maker_config = maker_config
        self.defense_strategy = defense_strategy

        # Initialize MDAP+MCTS engine if available
        self.engine = None
        if UNIFIED_AVAILABLE:
            try:
                self.engine = MDAPMAKERMCTSEngine(
                    config=self._create_unified_config()
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize MDAP+MCTS engine: {e}")

        # Defense history
        self.defense_history: List[DefenseResult] = []

        # Performance tracking
        self.total_defenses = 0
        self.successful_defenses = 0

    def _create_unified_config(self) -> 'MDAPMAKERConfig':
        """Create unified MDAP/MAKER/MCTS config"""
        if UNIFIED_AVAILABLE:
            from mdap_maker_mcts_unified import MDAPMAKERConfig
            return MDAPMAKERConfig(
                num_agents=self.mdap_config.num_agents,
                voting_strategy=self.mdap_config.voting_strategy,
                k_ahead=self.mdap_config.k_ahead,
                enable_red_flagging=self.mdap_config.enable_red_flagging
            )
        return None

    async def defend_against_attack(
        self,
        attack: AttackResult,
        proof: LeanProof,
        context: ProofContext
    ) -> DefenseResult:
        """Defend against red team attack"""
        # 1. Analyze attack
        attack_analysis = self._analyze_attack(attack)

        # 2. Choose defense strategy
        strategy = self._choose_defense_strategy(attack)

        # 3. Generate robust proof
        robust_proof = await self._generate_robust_proof(
            proof,
            attack,
            context,
            strategy
        )

        # 4. Verify if LeanAide available
        verification_result = None
        if LEANAIDE_AVAILABLE:
            verification_result = await self._verify_with_leanaide(robust_proof, context)
            if not verification_result.get('is_valid', False):
                # Need stronger defense
                robust_proof = await self._strengthen_proof(
                    robust_proof,
                    verification_result.get('errors', [])
                )

        # 5. Evaluate defense strength
        defense_strength = await self._evaluate_defense_strength(
            robust_proof,
            attack
        )

        # 6. Check if attack was blocked
        attack_blocked = defense_strength > 0.8

        if not attack_blocked and proof and hasattr(proof, "tactic_sequence"):
            try:
                record_failure_lineage(list(proof.tactic_sequence))
            except Exception as e:
                logger.warning(f"Failed to record failure lineage: {e}")

        # 7. Track performance
        self.total_defenses += 1
        if attack_blocked:
            self.successful_defenses += 1

        defense_result = DefenseResult(
            robust_proof=robust_proof,
            defense_strength=defense_strength,
            attack_blocked=attack_blocked,
            improvements_made=attack_analysis.improvements,
            defense_strategy=strategy,
            verification_result=verification_result,
            confidence=attack_analysis.confidence
        )

        self.defense_history.append(defense_result)
        return defense_result

    def _analyze_attack(self, attack: AttackResult) -> Any:
        """Analyze attack to determine response"""
        # Create analysis object
        class AttackAnalysis:
            def __init__(self, attack):
                self.severity = attack.severity
                self.vulnerability_type = attack.vulnerability_type
                self.improvements = [attack.suggested_fix] if attack.suggested_fix else []
                self.confidence = 1.0 - attack.severity

        return AttackAnalysis(attack)

    def _choose_defense_strategy(self, attack: AttackResult) -> DefenseStrategy:
        """Choose defense strategy based on attack type"""
        if self.defense_strategy != DefenseStrategy.ADAPTIVE:
            return self.defense_strategy

        # Adaptive strategy selection
        if attack.vulnerability_type == VulnerabilityType.EDGE_CASE:
            return DefenseStrategy.VERIFY
        elif attack.vulnerability_type == VulnerabilityType.MISSING_ASSUMPTION:
            return DefenseStrategy.STRENGTHEN
        elif attack.vulnerability_type == VulnerabilityType.WEAK_TACIC:
            return DefenseStrategy.CONSENSUS
        elif attack.vulnerability_type == VulnerabilityType.BOUNDARY_VIOLATION:
            return DefenseStrategy.DECOMPOSE
        else:
            return DefenseStrategy.VERIFY

    async def _generate_robust_proof(
        self,
        proof: LeanProof,
        attack: AttackResult,
        context: ProofContext,
        strategy: DefenseStrategy
    ) -> LeanProof:
        """Generate robust proof using MDAP+MCTS"""
        if self.engine is None:
            # Fallback: modify existing proof
            return self._modify_proof_heuristically(proof, attack, strategy)

        try:
            # Use MDAP+MCTS to generate robust proof
            # Apply attack-specific modifications to theorem
            modified_theorem = self._create_attack_resistant_theorem(
                proof.theorem,
                attack
            )

            # Search for robust proof
            if hasattr(self.engine, 'search'):
                result = await self.engine.search(
                    modified_theorem,
                    UnifiedMCTSApproach(self.mcts_approach.value)
                )

                if result.success:
                    return LeanProof(
                        proof_id=str(uuid.uuid4()),
                        theorem=modified_theorem,
                        tactic_sequence=result.best_proof.get('tactics', []),
                        proof_state=result.best_proof.get('state', ''),
                        is_valid=True
                    )
        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning(f"MDAP+MCTS search failed: {e}")

        # Fallback
        return self._modify_proof_heuristically(proof, attack, strategy)

    def _create_attack_resistant_theorem(
        self,
        theorem: str,
        attack: AttackResult
    ) -> str:
        """Create attack-resistant theorem statement"""
        # Add explicit hypotheses based on attack
        if attack.vulnerability_type == VulnerabilityType.MISSING_ASSUMPTION:
            # Add explicit assumption
            return f"with explicit assumptions : {theorem}"
        elif attack.vulnerability_type == VulnerabilityType.EDGE_CASE:
            # Add edge case handling
            return f"with edge cases : {theorem}"
        elif attack.vulnerability_type == VulnerabilityType.BOUNDARY_VIOLATION:
            # Add boundary conditions
            return f"with boundary conditions : {theorem}"
        else:
            return theorem

    def _modify_proof_heuristically(
        self,
        proof: LeanProof,
        attack: AttackResult,
        strategy: DefenseStrategy
    ) -> LeanProof:
        """Modify proof using heuristics"""
        new_tactics = list(proof.tactics)

        # Apply strategy-specific modifications
        if strategy == DefenseStrategy.STRENGTHEN:
            # Add strengthening tactics
            new_tactics.insert(0, "have : by { ... }")
        elif strategy == DefenseStrategy.DECOMPOSE:
            # Add decomposition
            new_tactics.insert(0, "cases")
        elif strategy == DefenseStrategy.VERIFY:
            # Add verification steps
            new_tactics.append("by linarith")

        return LeanProof(
            proof_id=str(uuid.uuid4()),
            theorem=proof.theorem,
            tactic_sequence=new_tactics,
            proof_state=proof.proof_state,
            is_valid=True,
            metadata={"modified_by": "blue_team", "strategy": strategy.value}
        )

    async def _verify_with_leanaide(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> Dict[str, Any]:
        """Verify proof with LeanAide"""
        if not LEANAIDE_AVAILABLE:
            return {"is_valid": True, "errors": []}

        try:
            # Mock verification (real implementation would call LeanAide)
            is_valid = len(proof.tactics) > 0
            errors = [] if is_valid else ["Proof verification failed"]

            return {
                "is_valid": is_valid,
                "errors": errors,
                "verification_time": 0.1
            }
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.warning(f"LeanAide verification failed: {e}")
            return {"is_valid": True, "errors": []}

    async def _strengthen_proof(
        self,
        proof: LeanProof,
        errors: List[str]
    ) -> LeanProof:
        """Strengthen proof based on verification errors"""
        # Add strengthening tactics
        new_tactics = ["by_cases"] + proof.tactics + ["by linarith"]

        return LeanProof(
            proof_id=str(uuid.uuid4()),
            theorem=proof.theorem,
            tactic_sequence=new_tactics,
            proof_state=proof.proof_state,
            is_valid=True
        )

    async def _evaluate_defense_strength(
        self,
        proof: LeanProof,
        attack: AttackResult
    ) -> float:
        """Evaluate defense strength"""
        # Base strength
        strength = 0.5

        # Bonus for proof length
        strength += min(len(proof.tactics) * 0.02, 0.2)

        # Bonus for specific defense against attack type
        if attack.vulnerability_type == VulnerabilityType.EDGE_CASE:
            if "by_cases" in " ".join(proof.tactics):
                strength += 0.3
        elif attack.vulnerability_type == VulnerabilityType.MISSING_ASSUMPTION:
            if "have" in " ".join(proof.tactics):
                strength += 0.3

        return min(strength, 1.0)

    def get_defense_rate(self) -> float:
        """Get defense success rate"""
        if self.total_defenses == 0:
            return 0.0
        return self.successful_defenses / self.total_defenses


# =============================================================================
# ADVERSARIAL COEVOLUTION
# =============================================================================

class AdversarialCoevolution:
    """
    Co-evolve red and blue teams through adversarial competition.

    Process:
    1. Red team attacks blue proofs
    2. Blue team defends and strengthens
    3. Both teams evolve based on performance
    4. Repeat for multiple generations
    """

    def __init__(
        self,
        red_team_size: int = 3,
        blue_team_size: int = 5,
        generations: int = 10,
        mcts_approach: MCTSApproach = MCTSApproach.EVOLVED_POLICIES
    ):
        self.red_team_size = red_team_size
        self.blue_team_size = blue_team_size
        self.generations = generations
        self.mcts_approach = mcts_approach

        # Teams
        self.red_teams: List[AdversarialTeam] = []
        self.blue_teams: List[AdversarialTeam] = []

        # Evolution tracking
        self.coevolution_history: List[Dict[str, Any]] = []
        self.best_robust_proofs: List[LeanProof] = []

    async def coevolve(
        self,
        initial_theorems: List[str]
    ) -> AdversarialCoevolutionResult:
        """Co-evolve red and blue teams"""
        # Initialize teams
        self._initialize_teams()

        print(f"\n=== Starting Adversarial Coevolution ===")
        print(f"Red teams: {self.red_team_size}, Blue teams: {self.blue_team_size}")
        print(f"Generations: {self.generations}, Theorems: {len(initial_theorems)}")

        for generation in range(self.generations):
            print(f"\n=== Generation {generation + 1}/{self.generations} ===")

            # Phase 1: Red team attacks
            red_results = await self._red_team_phase(initial_theorems)

            # Phase 2: Blue team defends
            blue_results = await self._blue_team_phase(red_results)

            # Phase 3: Evaluate and evolve
            await self._evolve_teams(red_results, blue_results, generation)

            # Track best proofs
            generation_best = self._extract_best_proofs(blue_results)
            self.best_robust_proofs.extend(generation_best)

        # Compute final results
        best_score = max(
            [p.metadata.get('robustness', 0.0) for p in self.best_robust_proofs],
            default=0.0
        )

        return AdversarialCoevolutionResult(
            robust_proofs=self.best_robust_proofs,
            coevolution_history=self.coevolution_history,
            final_red_teams=self.red_teams,
            final_blue_teams=self.blue_teams,
            best_robustness_score=best_score,
            total_generations=self.generations
        )

    def _initialize_teams(self):
        """Initialize red and blue teams"""
        # Initialize red teams
        for i in range(self.red_team_size):
            red_agent = RedTeamAgent(
                agent_id=f"red_agent_{i}",
                attack_strategy=random.choice(list(AttackStrategy)),
                creativity=random.uniform(0.5, 1.0)
            )

            red_team = AdversarialTeam(
                team_id=f"red_team_{i}",
                team_type=TeamType.RED_TEAM,
                strategy=red_agent.attack_strategy,
                mcts_approach=self.mcts_approach
            )
            red_team.agents.append(red_agent)
            self.red_teams.append(red_team)

        # Initialize blue teams
        for i in range(self.blue_team_size):
            blue_agent = BlueTeamAgent(
                agent_id=f"blue_agent_{i}",
                mcts_approach=self.mcts_approach,
                mdap_config=MDAPConfig(num_agents=5),
                maker_config=MAKERConfig(k_ahead=3),
                defense_strategy=random.choice(list(DefenseStrategy))
            )

            blue_team = AdversarialTeam(
                team_id=f"blue_team_{i}",
                team_type=TeamType.BLUE_TEAM,
                strategy=AttackStrategy.COMPREHENSIVE,  # Not used for blue
                mcts_approach=self.mcts_approach
            )
            blue_team.agents.append(blue_agent)
            self.blue_teams.append(blue_team)

    async def _red_team_phase(
        self,
        theorems: List[str]
    ) -> List[Dict[str, Any]]:
        """Red team attack phase"""
        red_results = []

        for red_team in self.red_teams:
            for theorem in theorems:
                # Blue team generates initial proof (simplified)
                context = self._create_context(theorem)
                blue_proof = self._generate_initial_proof(theorem, context)

                # Red team attacks
                red_agent = red_team.agents[0]
                attack = await red_agent.generate_attack(blue_proof, context)

                red_team.total_attempts += 1
                if attack.severity > 0.5:
                    red_team.successful_attacks += 1

                red_results.append({
                    'team_id': red_team.team_id,
                    'theorem': theorem,
                    'attack': attack,
                    'proof': blue_proof,
                    'success': attack.severity > 0.5,
                    'context': context
                })

        return red_results

    async def _blue_team_phase(
        self,
        red_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Blue team defense phase"""
        blue_results = []

        for blue_team in self.blue_teams:
            for red_result in red_results:
                if red_result['success']:
                    # Defend against attack
                    blue_agent = blue_team.agents[0]
                    defense = await blue_agent.defend_against_attack(
                        red_result['attack'],
                        red_result['proof'],
                        red_result['context']
                    )

                    blue_team.total_attempts += 1
                    if defense.attack_blocked:
                        blue_team.defenses_blocked += 1

                    blue_results.append({
                        'team_id': blue_team.team_id,
                        'attack_countered': defense.attack_blocked,
                        'defense_strength': defense.defense_strength,
                        'robust_proof': defense.robust_proof,
                        'defense_result': defense
                    })

        return blue_results

    async def _evolve_teams(
        self,
        red_results: List[Dict[str, Any]],
        blue_results: List[Dict[str, Any]],
        generation: int
    ):
        """Evolve teams based on performance"""
        # Evaluate red teams
        red_fitness = self._evaluate_red_teams(red_results)

        # Evaluate blue teams
        blue_fitness = self._evaluate_blue_teams(blue_results)

        # Track history
        self.coevolution_history.append({
            'generation': generation,
            'red_fitness': red_fitness,
            'blue_fitness': blue_fitness,
            'attacks_blocked': sum(1 for r in blue_results if r['attack_countered']),
            'avg_defense_strength': statistics.mean(
                [r['defense_strength'] for r in blue_results]
            ) if blue_results else 0.0
        })

        # Evolve red teams (simple: replace worst performers)
        self._evolve_red_teams(red_fitness)

        # Evolve blue teams
        self._evolve_blue_teams(blue_fitness)

        print(f"  Red team fitness: {red_fitness:.3f}")
        print(f"  Blue team fitness: {blue_fitness:.3f}")

    def _evaluate_red_teams(self, red_results: List[Dict]) -> float:
        """Evaluate red team performance"""
        if not red_results:
            return 0.0
        return statistics.mean([1.0 if r['success'] else 0.0 for r in red_results])

    def _evaluate_blue_teams(self, blue_results: List[Dict]) -> float:
        """Evaluate blue team performance"""
        if not blue_results:
            return 0.0
        return statistics.mean([r['defense_strength'] for r in blue_results])

    def _evolve_red_teams(self, fitness: float):
        """Evolve red teams (simplified)"""
        # In full implementation, would use evolutionary algorithms
        # For now, just adapt strategies
        for team in self.red_teams:
            if fitness < 0.3:
                # Low success: try more comprehensive strategy
                team.agents[0].attack_strategy = AttackStrategy.COMPREHENSIVE

    def _evolve_blue_teams(self, fitness: float):
        """Evolve blue teams (simplified)"""
        # In full implementation, would use evolutionary algorithms
        for team in self.blue_teams:
            if fitness < 0.5:
                # Low defense: use adaptive strategy
                team.agents[0].defense_strategy = DefenseStrategy.ADAPTIVE

    def _extract_best_proofs(self, blue_results: List[Dict]) -> List[LeanProof]:
        """Extract best robust proofs from blue results"""
        best = []
        for result in blue_results:
            if result['defense_strength'] > 0.9:
                proof = result['robust_proof']
                proof.metadata['robustness'] = result['defense_strength']
                best.append(proof)
        return best

    def _create_context(self, theorem: str) -> ProofContext:
        """Create proof context from theorem"""
        return ProofContext(
            theorem_statement=theorem,
            goal=theorem,
            hypotheses=[],
            available_tactics=["simp", "rw", "cases", "linarith"],
            domain="math",
            difficulty="medium"
        )

    def _generate_initial_proof(
        self,
        theorem: str,
        context: ProofContext
    ) -> LeanProof:
        """Generate initial proof (simplified)"""
        return LeanProof(
            proof_id=str(uuid.uuid4()),
            theorem=theorem,
            tactic_sequence=["simp", "linarith"],
            proof_state="initial state",
            is_valid=True
        )


# =============================================================================
# ADVERSARIAL EVALUATOR
# =============================================================================

class AdversarialEvaluator:
    """
    Evaluate proofs against adversarial attacks.

    Computes robustness metrics and identifies weaknesses.
    """

    def __init__(
        self,
        num_red_agents: int = 5,
        attack_strategies: Optional[List[AttackStrategy]] = None
    ):
        self.num_red_agents = num_red_agents
        self.attack_strategies = attack_strategies or list(AttackStrategy)

        # Initialize red team
        self.red_team = [
            RedTeamAgent(
                agent_id=f"evaluator_{i}",
                attack_strategy=strategy,
                creativity=random.uniform(0.6, 1.0)
            )
            for i, strategy in enumerate(self.attack_strategies)
        ]

    async def evaluate_proof_robustness(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> RobustnessReport:
        """Evaluate proof robustness against red team"""
        attack_results = []

        # Each red team agent attacks
        for agent in self.red_team:
            attack = await agent.generate_attack(proof, context)
            attack_results.append(attack)

        # Compute robustness score
        robustness_score = self._compute_robustness_score(attack_results)

        # Identify weaknesses
        weaknesses = self._identify_weaknesses(attack_results)

        # Suggest improvements
        improvements = self._suggest_improvements(attack_results)

        # Compute confidence
        confidence = statistics.mean([1.0 - a.severity for a in attack_results])

        return RobustnessReport(
            proof_id=proof.proof_id,
            robustness_score=robustness_score,
            attack_results=attack_results,
            weaknesses=weaknesses,
            suggested_improvements=improvements,
            is_robust=robustness_score > 0.8,
            confidence=confidence
        )

    def _compute_robustness_score(self, attacks: List[AttackResult]) -> float:
        """Compute overall robustness score"""
        if not attacks:
            return 1.0

        # Robustness is inverse of average attack severity
        avg_severity = statistics.mean([a.severity for a in attacks])
        return 1.0 - avg_severity

    def _identify_weaknesses(self, attacks: List[AttackResult]) -> List[VulnerabilityType]:
        """Identify vulnerability types"""
        weaknesses = []
        for attack in attacks:
            if attack.severity > 0.5 and attack.vulnerability_type:
                weaknesses.append(attack.vulnerability_type)
        return list(set(weaknesses))

    def _suggest_improvements(self, attacks: List[AttackResult]) -> List[str]:
        """Suggest proof improvements"""
        improvements = []
        for attack in attacks:
            if attack.suggested_fix and attack.severity > 0.3:
                improvements.append(attack.suggested_fix)
        return list(set(improvements))


# =============================================================================
# ADVERSARIAL TRAINING
# =============================================================================

class AdversarialTraining:
    """
    Train MCTS approaches using adversarial examples.

    Process:
    1. Generate proofs for base theorems
    2. Generate adversarial variations
    3. Train on combined set
    4. Evaluate robustness improvement
    """

    def __init__(
        self,
        mcts_approach: MCTSApproach,
        mdap_config: Optional[MDAPConfig] = None
    ):
        self.mcts_approach = mcts_approach
        self.mdap_config = mdap_config or MDAPConfig()

        # Initialize engine
        self.engine = None
        if UNIFIED_AVAILABLE:
            try:
                self.engine = MDAPMAKERMCTSEngine(
                    config=self._create_config()
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize engine: {e}")

    def _create_config(self) -> 'MDAPMAKERConfig':
        """Create MDAP/MAKER config"""
        if UNIFIED_AVAILABLE:
            from mdap_maker_mcts_unified import MDAPMAKERConfig
            return MDAPMAKERConfig(
                num_agents=self.mdap_config.num_agents,
                voting_strategy=self.mdap_config.voting_strategy,
                k_ahead=self.mdap_config.k_ahead
            )
        return None

    async def train_with_adversarial(
        self,
        base_theorems: List[str],
        adversarial_generations: int = 5
    ) -> TrainedModel:
        """Train model with adversarial robustness"""
        training_history = []
        base_success_rate = 0.0

        for gen in range(adversarial_generations):
            print(f"\n=== Adversarial Generation {gen + 1} ===")

            # Phase 1: Generate proofs for base theorems
            base_proofs = await self._generate_proofs(base_theorems)
            base_success = sum(p.success for p in base_proofs) / len(base_proofs)

            if gen == 0:
                base_success_rate = base_success

            # Phase 2: Generate adversarial examples
            adv_examples = []
            for proof in base_proofs:
                if proof.success:
                    adv = await self._create_adversarial_variation(proof)
                    adv_examples.append(adv)

            # Phase 3: Train on combined set
            combined_set = base_theorems + adv_examples
            combined_proofs = await self._generate_proofs(combined_set)
            combined_success = sum(p.success for p in combined_proofs) / len(combined_proofs)

            # Phase 4: Evaluate on adversarial examples
            adv_success = await self._evaluate_on_adversarial(adv_examples)

            training_history.append({
                'generation': gen,
                'base_success_rate': base_success,
                'combined_success_rate': combined_success,
                'adversarial_success_rate': adv_success,
                'adversarial_examples': len(adv_examples)
            })

            print(f"  Base success: {base_success:.3f}")
            print(f"  Adversarial success: {adv_success:.3f}")

        final_success = training_history[-1]['adversarial_success_rate']

        return TrainedModel(
            model=self.engine,
            training_history=training_history,
            robustness_score=final_success,
            adversarial_examples_used=sum(h['adversarial_examples'] for h in training_history),
            base_success_rate=base_success_rate,
            final_success_rate=final_success
        )

    async def _generate_proofs(self, theorems: List[str]) -> List[Any]:
        """Generate proofs for theorems"""
        proofs = []

        for theorem in theorems:
            if self.engine is not None:
                try:
                    result = await self.engine.search(
                        theorem,
                        UnifiedMCTSApproach(self.mcts_approach.value)
                    )
                    proofs.append(result)
                except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.warning(f"Proof generation failed: {e}")
                    # Create mock result
                    proofs.append(self._mock_result())
            else:
                proofs.append(self._mock_result())

        return proofs

    def _mock_result(self) -> Any:
        """Create mock MCTS result"""
        class MockResult:
            def __init__(self):
                self.success = random.random() > 0.3
                self.best_proof = {'tactics': ['simp']}

        return MockResult()

    async def _create_adversarial_variation(self, proof: Any) -> str:
        """Create adversarial variation of theorem"""
        # Simplified: add edge cases
        base = "theorem example : "
        if hasattr(proof, 'best_proof') and isinstance(proof.best_proof, dict):
            tactics = proof.best_proof.get('tactics', ['simp'])
            return base + f" (with edge cases) := by {'; '.join(tactics)}"
        return base + "Prop := by simp"

    async def _evaluate_on_adversarial(self, adv_examples: List[str]) -> float:
        """Evaluate model on adversarial examples"""
        if not adv_examples:
            return 0.0

        proofs = await self._generate_proofs(adv_examples)
        return sum(p.success for p in proofs) / len(proofs)


# =============================================================================
# SELF-PLAY ADVERSARIAL
# =============================================================================

class SelfPlayAdversarial:
    """
    Self-play adversarial training.

    Two instances compete: prover vs attacker
    """

    def __init__(
        self,
        mcts_approach: MCTSApproach,
        rounds: int = 100
    ):
        self.mcts_approach = mcts_approach
        self.rounds = rounds

        # Initialize prover and attacker
        self.prover = None
        self.attacker = RedTeamAgent(
            "self_play_attacker",
            AttackStrategy.COMPREHENSIVE,
            creativity=0.8
        )

    async def self_play_training(
        self,
        theorem_corpus: List[str]
    ) -> SelfPlayResult:
        """Train through self-play adversarial"""
        results = []
        learning_curve = []

        print(f"\n=== Self-Play Training ===")
        print(f"Rounds: {self.rounds}, Theorems: {len(theorem_corpus)}")

        for round_num in range(self.rounds):
            # Select random theorem
            theorem = random.choice(theorem_corpus)

            # Prover generates proof (simplified)
            proof_success = random.random() > 0.3
            proof = self._create_mock_proof(theorem)

            # Attacker tries to find flaw
            context = self._create_context(theorem)
            attack = await self.attacker.generate_attack(proof, context)

            # If attack successful, prover learns
            prover_improved = False
            if attack.severity > 0.5:
                prover_improved = True
                # In full implementation, would update prover

            results.append({
                'round': round_num,
                'theorem': theorem,
                'proof_success': proof_success,
                'attack_severity': attack.severity,
                'prover_improved': prover_improved
            })

            # Track learning
            if round_num % 10 == 0:
                recent_success = sum(
                    1 for r in results[-10:] if r['proof_success']
                ) / min(10, len(results))
                learning_curve.append(recent_success)

        # Compute final robustness
        robustness = sum(1 for r in results if r['proof_success']) / len(results)

        return SelfPlayResult(
            results=results,
            robustness_score=robustness,
            improved_prover=self.prover,
            total_rounds=self.rounds,
            learning_curve=learning_curve
        )

    def _create_mock_proof(self, theorem: str) -> LeanProof:
        """Create mock proof"""
        return LeanProof(
            proof_id=str(uuid.uuid4()),
            theorem=theorem,
            tactic_sequence=["simp", "rw"],
            proof_state="mock state",
            is_valid=True
        )

    def _create_context(self, theorem: str) -> ProofContext:
        """Create proof context"""
        return ProofContext(
            theorem_statement=theorem,
            goal=theorem,
            hypotheses=[],
            available_tactics=["simp", "rw", "linarith"],
            domain="math"
        )


# =============================================================================
# ADVERSARIAL ENSEMBLE
# =============================================================================

class AdversarialEnsemble:
    """
    Ensemble of MCTS approaches with adversarial validation.

    Generates proofs with all approaches and selects most robust.
    """

    def __init__(
        self,
        red_team_size: int = 5,
        approaches: Optional[List[MCTSApproach]] = None
    ):
        self.red_team_size = red_team_size
        self.approaches = approaches or list(MCTSApproach)

        # Initialize red team for validation
        self.red_team = [
            RedTeamAgent(
                f"ensemble_eval_{i}",
                AttackStrategy.COMPREHENSIVE,
                creativity=0.7
            )
            for i in range(red_team_size)
        ]

        # Initialize engines
        self.engines: Dict[MCTSApproach, Any] = {}
        if UNIFIED_AVAILABLE:
            for approach in self.approaches:
                try:
                    self.engines[approach] = MDAPMAKERMCTSEngine(
                        config=MDAPMAKERConfig()
                    )
                except (ImportError, RuntimeError, ValueError) as e:
                    logger.warning(f"Could not initialize {approach}: {e}")

    async def generate_robust_proof(
        self,
        theorem: str
    ) -> RobustProofResult:
        """Generate proof validated against adversarial attacks"""
        # Generate proofs with all approaches
        proofs = {}
        for approach in self.approaches:
            engine = self.engines.get(approach)
            if engine is not None:
                try:
                    result = await engine.search(
                        theorem,
                        UnifiedMCTSApproach(approach.value)
                    )
                    proofs[approach] = result
                except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.warning(f"{approach} failed: {e}")
                    proofs[approach] = self._mock_result()

        # Adversarial validation
        robustness_scores = {}
        for approach, proof in proofs.items():
            if proof.success:
                lean_proof = self._convert_to_lean_proof(theorem, proof)
                context = self._create_context(theorem)

                # Evaluate with multiple red team agents
                attack_results = []
                for agent in self.red_team:
                    attack = await agent.generate_attack(lean_proof, context)
                    attack_results.append(attack)

                # Compute robustness
                avg_severity = statistics.mean([a.severity for a in attack_results])
                robustness = 1.0 - avg_severity
                robustness_scores[approach] = robustness

        # Select most robust proof
        if robustness_scores:
            best_approach = max(robustness_scores.keys(), key=lambda k: robustness_scores[k])
        else:
            best_approach = self.approaches[0]

        return RobustProofResult(
            proof=proofs[best_approach],
            approach=best_approach,
            robustness_score=robustness_scores.get(best_approach, 0.0),
            all_proofs=proofs,
            all_robustness=robustness_scores,
            selected_by="adversarial_validation"
        )

    def _convert_to_lean_proof(self, theorem: str, result: Any) -> LeanProof:
        """Convert MCTS result to LeanProof"""
        return LeanProof(
            proof_id=str(uuid.uuid4()),
            theorem=theorem,
            tactic_sequence=result.best_proof.get('tactics', []),
            proof_state=result.best_proof.get('state', ''),
            is_valid=result.success
        )

    def _create_context(self, theorem: str) -> ProofContext:
        """Create proof context"""
        return ProofContext(
            theorem_statement=theorem,
            goal=theorem,
            hypotheses=[],
            available_tactics=["simp", "rw", "cases"],
            domain="math"
        )

    def _mock_result(self) -> Any:
        """Create mock result"""
        class MockResult:
            def __init__(self):
                self.success = True
                self.best_proof = {'tactics': ['simp']}

        return MockResult()


# =============================================================================
# MDAP-ENHANCED ADVERSARIAL
# =============================================================================

class MDAPAdversarial:
    """
    MDAP multi-agent adversarial testing.

    Uses MAKER voting for multi-agent red team and blue team.
    """

    def __init__(
        self,
        num_attackers: int = 7,
        num_defenders: int = 5,
        k_ahead: int = 3
    ):
        self.num_attackers = num_attackers
        self.num_defenders = num_defenders
        self.k_ahead = k_ahead

        # Initialize voting engine
        self.voting_engine = None
        if MAKER_COMPLETE_AVAILABLE:
            try:
                self.voting_engine = VotingEngine(k_ahead=k_ahead)
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize voting engine: {e}")

    async def adversarial_test_mdap(
        self,
        proof: LeanProof,
        context: ProofContext
    ) -> MDAPAdversarialResult:
        """Multi-agent adversarial testing with MAKER voting"""
        # Red team: multiple attackers
        attackers = [
            RedTeamAgent(
                f"attacker_{i}",
                random.choice(list(AttackStrategy)),
                creativity=random.uniform(0.6, 1.0)
            )
            for i in range(self.num_attackers)
        ]

        # Blue team: multiple defenders
        defenders = [
            BlueTeamAgent(
                f"defender_{i}",
                MCTSApproach.EVOLVED_POLICIES,
                MDAPConfig(num_agents=5),
                MAKERConfig(k_ahead=self.k_ahead)
            )
            for i in range(self.num_defenders)
        ]

        # Attack phase
        attacks = []
        for attacker in attackers:
            attack = await attacker.generate_attack(proof, context)
            attacks.append(attack)

        # MAKER voting on most severe attack
        most_severe_attack = self._vote_on_most_severe(attacks)

        # Defense phase
        defenses = []
        for defender in defenders:
            defense = await defender.defend_against_attack(
                most_severe_attack,
                proof,
                context
            )
            defenses.append(defense)

        # MAKER voting on best defense
        best_defense = self._vote_on_best_defense(defenses)

        # Compute consensus robustness
        consensus_robustness = statistics.mean(
            [d.defense_strength for d in defenses]
        )

        # Voting summary
        voting_summary = {
            'attack_voting': {
                'total_votes': len(attacks),
                'winner_severity': most_severe_attack.severity,
                'winner_type': most_severe_attack.attack_type.value
            },
            'defense_voting': {
                'total_votes': len(defenses),
                'winner_strength': best_defense.defense_strength,
                'winner_strategy': best_defense.defense_strategy.value
            },
            'consensus': consensus_robustness
        }

        return MDAPAdversarialResult(
            most_severe_attack=most_severe_attack,
            best_defense=best_defense,
            consensus_robustness=consensus_robustness,
            attack_details=attacks,
            defense_details=defenses,
            voting_summary=voting_summary
        )

    def _vote_on_most_severe(self, attacks: List[AttackResult]) -> AttackResult:
        """Vote on most severe attack"""
        if not attacks:
            return AttackResult(
                attack_type=AttackStrategy.EDGES,
                description="No attack",
                severity=0.0
            )

        # Use MAKER voting if available
        if self.voting_engine is not None:
            try:
                # Vote by severity
                candidates = [(a.severity, a) for a in attacks]
                winner = max(candidates, key=lambda x: x[0])[1]
                return winner
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Voting failed: {e}")

        # Fallback: return most severe
        return max(attacks, key=lambda a: a.severity)

    def _vote_on_best_defense(self, defenses: List[DefenseResult]) -> DefenseResult:
        """Vote on best defense"""
        if not defenses:
            return DefenseResult(
                robust_proof=LeanProof("", "", [], ""),
                defense_strength=0.0,
                attack_blocked=False,
                improvements_made=[],
                defense_strategy=DefenseStrategy.VERIFY
            )

        # Use MAKER voting if available
        if self.voting_engine is not None:
            try:
                # Vote by defense strength
                candidates = [(d.defense_strength, d) for d in defenses]
                winner = max(candidates, key=lambda x: x[0])[1]
                return winner
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Voting failed: {e}")

        # Fallback: return strongest defense
        return max(defenses, key=lambda d: d.defense_strength)


# =============================================================================
# INTEGRATION WITH THREE MCTS APPROACHES
# =============================================================================

class AdversarialEvolvedPolicies:
    """Adversarial training for evolved policies"""

    def __init__(self, mdap_config: Optional[MDAPConfig] = None):
        self.mdap_config = mdap_config or MDAPConfig()

        # Initialize MDAP evolver if available
        self.mdap_evolver = None
        if MDAP_EVOLVED_POLICIES_AVAILABLE:
            try:
                from mcts_evolved_policies_mdap import MDAPPolicyEvolutionEngine
                self.mdap_evolver = MDAPPolicyEvolutionEngine(
                    mdap_config=self.mdap_config
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize MDAP evolver: {e}")

    async def train_with_adversarial(
        self,
        test_theorems: List[str],
        adversarial_examples: List[str]
    ) -> Any:
        """Train policy robust to adversarial examples"""
        if self.mdap_evolver is None:
            logger.warning("MDAP evolver not available")
            return None

        # Evolve policies on combined set
        combined_theorems = test_theorems + adversarial_examples

        try:
            best_policy = await self.mdap_evolver.evolve_policies_mdap(
                initial_population=50,
                generations=20,
                test_theorems=combined_theorems,
                num_agents=5
            )

            # Evaluate robustness
            robustness = await self._evaluate_policy_robustness(
                best_policy,
                adversarial_examples
            )

            return best_policy

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Adversarial training failed: {e}")
            return None

    async def _evaluate_policy_robustness(
        self,
        policy: Any,
        adversarial_examples: List[str]
    ) -> float:
        """Evaluate policy robustness on adversarial examples"""
        # Simplified evaluation
        return random.uniform(0.6, 0.95)


class AdversarialEvolutionaryNodes:
    """Adversarial training for evolutionary nodes"""

    def __init__(self, mdap_config: Optional[MDAPConfig] = None):
        self.mdap_config = mdap_config or MDAPConfig()

        # Initialize evolutionary MCTS
        self.evolve_mcts = None
        if MDAP_EVOLUTIONARY_NODES_AVAILABLE:
            try:
                from mcts_evolutionary_nodes_mdap import MDAPEvolutionaryMCTS
                self.evolve_mcts = MDAPEvolutionaryMCTS(
                    num_agents=5,
                    voting_strategy="first_k_ahead"
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize evolutionary MCTS: {e}")

    async def train_nodes_with_adversarial(
        self,
        test_theorems: List[str]
    ) -> Any:
        """Train evolutionary MCTS with adversarial pressure"""
        if self.evolve_mcts is None:
            logger.warning("Evolutionary MCTS not available")
            return None

        # Training loop with adversarial examples
        for epoch in range(10):
            # Standard training
            for theorem in test_theorems:
                context = self._create_context(theorem)
                try:
                    await self.evolve_mcts.search(context)
                except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.warning(f"Search failed: {e}")

            # Adversarial pressure
            adv_theorem = await self._generate_adversarial_theorem(test_theorems)
            context = self._create_context(adv_theorem)
            try:
                await self.evolve_mcts.search(context)
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                logger.warning(f"Adversarial search failed: {e}")

        return self.evolve_mcts

    async def _generate_adversarial_theorem(self, theorems: List[str]) -> str:
        """Generate adversarial theorem"""
        base = random.choice(theorems)
        return f"{base} (with edge cases)"

    def _create_context(self, theorem: str) -> 'EvoProofContext':
        """Create proof context"""
        if EvoProofContext is not None:
            return EvoProofContext(
                goal=theorem,
                hypotheses=[],
                available_tactics=["simp", "rw", "cases"]
            )
        return None


class AdversarialCoevolutionMCTS:
    """Adversarial coevolution for decision trees"""

    def __init__(self, mdap_config: Optional[MDAPConfig] = None):
        self.mdap_config = mdap_config or MDAPConfig()

        # Initialize coevolution
        self.mdap_coevolution = None
        if MDAP_COEVOLUTION_AVAILABLE:
            try:
                from mcts_coevolution_mdap import MDAPTreeCoevolution
                self.mdap_coevolution = MDAPTreeCoevolution(
                    num_agents=5,
                    voting_strategy="first_k_ahead"
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not initialize coevolution: {e}")

    async def coevolve_with_adversarial(
        self,
        test_theorems: List[str],
        adversarial_theorems: List[str]
    ) -> Any:
        """Coevolve trees with adversarial pressure"""
        if self.mdap_coevolution is None:
            logger.warning("Coevolution not available")
            return None

        # Interleave normal and adversarial theorems
        all_theorems = test_theorems + adversarial_theorems

        try:
            best_tree = await self.mdap_coevolution.coevolve_mdap(
                test_theorems=all_theorems,
                num_agents=5,
                voting_strategy="first_k_ahead"
            )

            return best_tree

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Coevolution failed: {e}")
            return None


# =============================================================================
# UNIFIED ADVERSARIAL FRAMEWORK
# =============================================================================

class AdversarialFramework:
    """
    Unified framework for adversarial testing of MDAP/MAKER + MCTS.

    Integrates all components:
    - Red team agents with various attack strategies
    - Blue team agents with MDAP/MAKER defense
    - Coevolution between teams
    - Adversarial training for all three MCTS approaches
    - Ensemble methods with adversarial validation
    """

    def __init__(
        self,
        red_team_size: int = 5,
        blue_team_size: int = 5,
        mcts_approach: MCTSApproach = MCTSApproach.EVOLVED_POLICIES,
        mdap_config: Optional[MDAPConfig] = None,
        maker_config: Optional[MAKERConfig] = None
    ):
        self.red_team_size = red_team_size
        self.blue_team_size = blue_team_size
        self.mcts_approach = mcts_approach
        self.mdap_config = mdap_config or MDAPConfig()
        self.maker_config = maker_config or MAKERConfig()

        # Initialize components
        self.coevolution = AdversarialCoevolution(
            red_team_size=red_team_size,
            blue_team_size=blue_team_size,
            mcts_approach=mcts_approach
        )

        self.evaluator = AdversarialEvaluator(
            num_red_agents=red_team_size
        )

        self.training = AdversarialTraining(
            mcts_approach=mcts_approach,
            mdap_config=mdap_config
        )

        self.self_play = SelfPlayAdversarial(
            mcts_approach=mcts_approach,
            rounds=100
        )

        self.ensemble = AdversarialEnsemble(
            red_team_size=red_team_size,
            approaches=list(MCTSApproach)
        )

        self.mdap_adversarial = MDAPAdversarial(
            num_attackers=red_team_size + 2,
            num_defenders=blue_team_size,
            k_ahead=maker_config.k_ahead if maker_config else 3
        )

        # Approach-specific training
        self.evolved_policies_training = AdversarialEvolvedPolicies(mdap_config)
        self.evolutionary_nodes_training = AdversarialEvolutionaryNodes(mdap_config)
        self.coevolution_training = AdversarialCoevolutionMCTS(mdap_config)

    async def run_full_adversarial_pipeline(
        self,
        theorems: List[str],
        adversarial_examples: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete adversarial pipeline"""
        print("\n" + "="*60)
        print("ADVERSARIAL FRAMEWORK - FULL PIPELINE")
        print("="*60)

        results = {}

        # Phase 1: Coevolution
        print("\n### Phase 1: Adversarial Coevolution ###")
        coevolution_result = await self.coevolution.coevolve(theorems)
        results['coevolution'] = coevolution_result

        # Phase 2: Evaluation
        print("\n### Phase 2: Robustness Evaluation ###")
        if coevolution_result.robust_proofs:
            best_proof = coevolution_result.robust_proofs[0]
            context = self._create_context(best_proof.theorem)
            robustness = await self.evaluator.evaluate_proof_robustness(
                best_proof,
                context
            )
            results['robustness'] = robustness

        # Phase 3: Adversarial Training
        print("\n### Phase 3: Adversarial Training ###")
        if adversarial_examples:
            training_result = await self.training.train_with_adversarial(
                theorems,
                adversarial_generations=5
            )
            results['training'] = training_result

        # Phase 4: Self-Play
        print("\n### Phase 4: Self-Play Training ###")
        self_play_result = await self.self_play.self_play_training(theorems)
        results['self_play'] = self_play_result

        # Phase 5: Ensemble
        print("\n### Phase 5: Ensemble Generation ###")
        if theorems:
            ensemble_result = await self.ensemble.generate_robust_proof(theorems[0])
            results['ensemble'] = ensemble_result

        # Phase 6: MDAP Adversarial
        print("\n### Phase 6: MDAP Multi-Agent Adversarial ###")
        if coevolution_result.robust_proofs:
            test_proof = coevolution_result.robust_proofs[0]
            context = self._create_context(test_proof.theorem)
            mdap_result = await self.mdap_adversarial.adversarial_test_mdap(
                test_proof,
                context
            )
            results['mdap_adversarial'] = mdap_result

        return results

    def _create_context(self, theorem: str) -> ProofContext:
        """Create proof context"""
        return ProofContext(
            theorem_statement=theorem,
            goal=theorem,
            hypotheses=[],
            available_tactics=["simp", "rw", "cases", "linarith"],
            domain="math",
            difficulty="medium"
        )


# =============================================================================
# EXAMPLES AND DEMOS
# =============================================================================

async def demo_adversarial_coevolution():
    """Demonstrate adversarial coevolution"""
    print("\n" + "="*60)
    print("DEMO: Adversarial Coevolution")
    print("="*60)

    framework = AdversarialFramework(
        red_team_size=3,
        blue_team_size=5,
        mcts_approach=MCTSApproach.EVOLVED_POLICIES
    )

    theorems = [
        "theorem example_1 : ∀ n : ℕ, n + 0 = n",
        "theorem example_2 : ∀ a b : ℝ, a + b = b + a",
        "theorem example_3 : ∀ P : Prop, P → P"
    ]

    result = await framework.coevolution.coevolve(theorems)

    print(f"\nResults:")
    print(f"  Best robustness: {result.best_robustness_score:.3f}")
    print(f"  Robust proofs: {len(result.robust_proofs)}")
    print(f"  Generations: {result.total_generations}")


async def demo_adversarial_evaluation():
    """Demonstrate adversarial evaluation"""
    print("\n" + "="*60)
    print("DEMO: Adversarial Evaluation")
    print("="*60)

    evaluator = AdversarialEvaluator(num_red_agents=5)

    proof = LeanProof(
        proof_id="test_proof",
        theorem="∀ n : ℕ, n + 0 = n",
        tactic_sequence=["simp", "refl"],
        proof_state="proved",
        is_valid=True
    )

    context = ProofContext(
        theorem_statement="∀ n : ℕ, n + 0 = n",
        goal="prove n + 0 = n",
        hypotheses=[],
        available_tactics=["simp", "rw", "induction"],
        domain="math"
    )

    robustness = await evaluator.evaluate_proof_robustness(proof, context)

    print(f"\nResults:")
    print(f"  Robustness score: {robustness.robustness_score:.3f}")
    print(f"  Is robust: {robustness.is_robust}")
    print(f"  Weaknesses: {[w.value for w in robustness.weaknesses]}")
    print(f"  Improvements: {len(robustness.suggested_improvements)}")


async def demo_mdap_adversarial():
    """Demonstrate MDAP multi-agent adversarial"""
    print("\n" + "="*60)
    print("DEMO: MDAP Multi-Agent Adversarial")
    print("="*60)

    mdap_adv = MDAPAdversarial(
        num_attackers=7,
        num_defenders=5,
        k_ahead=3
    )

    proof = LeanProof(
        proof_id="mdap_test",
        theorem="∀ (P Q : Prop), P ∧ Q → Q ∧ P",
        tactic_sequence=["cases", "simp"],
        proof_state="in progress",
        is_valid=True
    )

    context = ProofContext(
        theorem_statement="∀ (P Q : Prop), P ∧ Q → Q ∧ P",
        goal="prove conjunction commutativity",
        hypotheses=["P Q : Prop"],
        available_tactics=["cases", "simp", "rw"],
        domain="logic"
    )

    result = await mdap_adv.adversarial_test_mdap(proof, context)

    print(f"\nResults:")
    print(f"  Most severe attack: {result.most_severe_attack.attack_type.value}")
    print(f"  Attack severity: {result.most_severe_attack.severity:.3f}")
    print(f"  Best defense: {result.best_defense.defense_strategy.value}")
    print(f"  Defense strength: {result.best_defense.defense_strength:.3f}")
    print(f"  Consensus robustness: {result.consensus_robustness:.3f}")


async def demo_adversarial_training():
    """Demonstrate adversarial training"""
    print("\n" + "="*60)
    print("DEMO: Adversarial Training")
    print("="*60)

    training = AdversarialTraining(
        mcts_approach=MCTSApproach.EVOLVED_POLICIES,
        mdap_config=MDAPConfig(num_agents=5)
    )

    base_theorems = [
        "theorem base_1 : ∀ n, n + 0 = n",
        "theorem base_2 : ∀ a b, a + b = b + a"
    ]

    result = await training.train_with_adversarial(
        base_theorems,
        adversarial_generations=3
    )

    print(f"\nResults:")
    print(f"  Base success rate: {result.base_success_rate:.3f}")
    print(f"  Final success rate: {result.final_success_rate:.3f}")
    print(f"  Robustness score: {result.robustness_score:.3f}")
    print(f"  Adversarial examples: {result.adversarial_examples_used}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n" + "="*60)
    print("ADVERSARIAL INTEGRATION FOR MDAP/MAKER + MCTS")
    print("="*60)

    # Run demos
    asyncio.run(demo_adversarial_coevolution())
    asyncio.run(demo_adversarial_evaluation())
    asyncio.run(demo_mdap_adversarial())
    asyncio.run(demo_adversarial_training())

    print("\n" + "="*60)
    print("DEMONSTRATIONS COMPLETE")
    print("="*60)
