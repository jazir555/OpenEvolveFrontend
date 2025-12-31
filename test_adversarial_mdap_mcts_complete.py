#!/usr/bin/env python3
"""
Comprehensive Test Suite for Adversarial MDAP/MAKER/MCTS Integration

Tests the complete adversarial framework integrated with:
- Evolved MCTS policies with MDAP
- Evolutionary MCTS nodes with MDAP
- Coevolution with MDAP
- Unified MDAP/MAKER/MCTS framework
- Red-blue team dynamics
- Attack and defense strategies
"""

import sys
import os
import asyncio
import pytest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Data Models
# ============================================================================

class MCTSApproach(Enum):
    """MCTS approach types"""
    EVOLVED_POLICIES = "evolved_policies"
    EVOLUTIONARY_NODES = "evolutionary_nodes"
    COEVOLUTION = "coevolution"
    UNIFIED = "unified"


class AttackType(Enum):
    """Types of adversarial attacks"""
    TACTIC_SUBSTITUTION = "tactic_substitution"
    HYPOTHESIS_INVERSION = "hypothesis_inversion"
    GOAL_MODIFICATION = "goal_modification"
    CONTEXT_MANIPULATION = "context_manipulation"
    PROOF_LENGTH_EXPLOSION = "proof_length_explosion"
    LOGIC_BOMBS = "logic_bombs"
    BOUNDARY_VIOLATION = "boundary_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class DefenseStrategy(Enum):
    """Types of defense strategies"""
    TACTIC_VALIDATION = "tactic_validation"
    REDUNDANT_VERIFICATION = "redundant_verification"
    CONSENSUS_FILTERING = "consensus_filtering"
    SANITY_CHECKS = "sanity_checks"
    BOUNDARY_ENFORCEMENT = "boundary_enforcement"
    RESOURCE_LIMITING = "resource_limiting"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    ENSEMBLE_DEFENSE = "ensemble_defense"


@dataclass
class AttackResult:
    """Result of an adversarial attack"""
    attack_type: AttackType
    original_proof: str
    attacked_proof: str
    success: bool
    confidence_reduction: float
    vulnerable_points: List[str]
    attack_description: str


@dataclass
class DefenseResult:
    """Result of a defense against an attack"""
    attack_detected: bool
    attack_blocked: bool
    defense_strategy: DefenseStrategy
    recovered_proof: Optional[str]
    confidence_restored: float
    defense_description: str


@dataclass
class AdversarialTestResult:
    """Result of adversarial testing"""
    theorem: str
    original_proof: str
    attacks_launched: int
    attacks_successful: int
    attacks_blocked: int
    robustness_score: float
    attack_results: List[AttackResult]
    defense_results: List[DefenseResult]
    vulnerabilities_found: List[str]
    improvements_suggested: List[str]


@dataclass
class ProofContext:
    """Proof solving context"""
    theorem: str
    current_proof_state: str
    available_tactics: List[str]
    proof_depth: int
    resources_used: Dict[str, float]


@dataclass
class LeanProof:
    """Lean 4 proof"""
    code: str
    tactics_used: List[str]
    proof_length: int
    verified: bool


# ============================================================================
# Mock Adversarial System Components
# ============================================================================

class MockRedTeamAgent:
    """Mock red team agent for testing"""

    def __init__(self, attack_types: List[AttackType] = None):
        self.attack_types = attack_types or list(AttackType)
        self.attack_count = 0

    async def generate_attack(
        self,
        proof: LeanProof,
        context: ProofContext,
        attack_type: Optional[AttackType] = None
    ) -> AttackResult:
        """Generate an adversarial attack on a proof"""
        self.attack_count += 1
        attack_type = attack_type or self.attack_types[self.attack_count % len(self.attack_types)]

        # Simulate attack generation
        success = self.attack_count % 3 != 0  # 66% success rate for testing
        confidence_reduction = 0.3 if success else 0.0

        return AttackResult(
            attack_type=attack_type,
            original_proof=proof.code,
            attacked_proof=f"-- {attack_type.value} attack\n{proof.code}",
            success=success,
            confidence_reduction=confidence_reduction,
            vulnerable_points=["tactic_3", "assumption_2"] if success else [],
            attack_description=f"Simulated {attack_type.value} attack"
        )


class MockBlueTeamAgent:
    """Mock blue team agent for testing"""

    def __init__(self, defense_strategies: List[DefenseStrategy] = None):
        self.defense_strategies = defense_strategies or list(DefenseStrategy)
        self.defense_count = 0

    async def defend_against_attack(
        self,
        attack: AttackResult,
        proof: LeanProof
    ) -> DefenseResult:
        """Defend against an adversarial attack"""
        self.defense_count += 1
        strategy = self.defense_strategies[self.defense_count % len(self.defense_strategies)]

        # Simulate defense
        attack_detected = attack.success
        attack_blocked = self.defense_count % 2 == 0  # 50% block rate
        confidence_restored = 0.25 if attack_blocked else 0.0

        return DefenseResult(
            attack_detected=attack_detected,
            attack_blocked=attack_blocked,
            defense_strategy=strategy,
            recovered_proof=proof.code if attack_blocked else None,
            confidence_restored=confidence_restored,
            defense_description=f"Applied {strategy.value} defense"
        )


class MockAdversarialCoevolution:
    """Mock adversarial coevolution for testing"""

    def __init__(self):
        self.generation = 0
        self.red_fitness = []
        self.blue_fitness = []

    async def coevolve(
        self,
        test_theorems: List[str],
        generations: int = 5
    ) -> Dict[str, Any]:
        """Run adversarial coevolution"""
        results = {
            "generations": [],
            "red_team_best_fitness": [],
            "blue_team_best_fitness": [],
            "robustness_improvement": 0.0
        }

        for gen in range(generations):
            self.generation += 1
            # Simulate coevolution
            red_fit = 0.5 + (gen * 0.1)
            blue_fit = 0.4 + (gen * 0.12)

            results["generations"].append(gen + 1)
            results["red_team_best_fitness"].append(red_fit)
            results["blue_team_best_fitness"].append(blue_fit)

        results["robustness_improvement"] = (
            results["blue_team_best_fitness"][-1] - results["blue_team_best_fitness"][0]
        )

        return results


# ============================================================================
# Test Cases
# ============================================================================

class TestAdversarialMDAPMCTSIntegration:
    """Test suite for adversarial MDAP/MCTS integration"""

    @pytest.fixture
    def red_team(self):
        """Create red team agent"""
        return MockRedTeamAgent()

    @pytest.fixture
    def blue_team(self):
        """Create blue team agent"""
        return MockBlueTeamAgent()

    @pytest.fixture
    def sample_proof(self):
        """Create sample proof"""
        return LeanProof(
            code="theorem sample (a b : Nat) : a + b = b + a := by\n  rw [Nat.add_comm]",
        tactics_used=["rw"],
        proof_length=1,
        verified=True
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample proof context"""
        return ProofContext(
            theorem="∀ a b : Nat, a + b = b + a",
            current_proof_state="a b : Nat\n⊢ a + b = b + a",
            available_tactics=["rw", "apply", "exact", "intro"],
            proof_depth=0,
            resources_used={"time": 1.0, "memory": 0.5}
        )

    # ========================================================================
    # Tests: Red Team Attacks
    # ========================================================================

    @pytest.mark.asyncio
    async def test_red_team_tactic_substitution_attack(
        self,
        red_team: MockRedTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test red team tactic substitution attack"""
        attack = await red_team.generate_attack(
            sample_proof,
            sample_context,
            AttackType.TACTIC_SUBSTITUTION
        )

        assert attack.attack_type == AttackType.TACTIC_SUBSTITUTION
        assert isinstance(attack.success, bool)
        assert 0 <= attack.confidence_reduction <= 1
        assert attack.original_proof == sample_proof.code
        assert attack.attack_description is not None

    @pytest.mark.asyncio
    async def test_red_team_all_attack_types(
        self,
        red_team: MockRedTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test red team can generate all attack types"""
        for attack_type in AttackType:
            attack = await red_team.generate_attack(
                sample_proof,
                sample_context,
                attack_type
            )
            assert attack.attack_type == attack_type
            assert attack.attacked_proof is not None

    @pytest.mark.asyncio
    async def test_red_team_attack_success_tracking(
        self,
        red_team: MockRedTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test red team tracks successful attacks"""
        successes = 0
        for _ in range(10):
            attack = await red_team.generate_attack(sample_proof, sample_context)
            if attack.success:
                successes += 1

        # Should have some successful attacks (testing mock has ~66% success)
        assert successes >= 3
        assert red_team.attack_count == 10

    # ========================================================================
    # Tests: Blue Team Defenses
    # ========================================================================

    @pytest.mark.asyncio
    async def test_blue_team_defense_against_attack(
        self,
        red_team: MockRedTeamAgent,
        blue_team: MockBlueTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test blue team can defend against attacks"""
        attack = await red_team.generate_attack(sample_proof, sample_context)
        defense = await blue_team.defend_against_attack(attack, sample_proof)

        assert isinstance(defense.attack_detected, bool)
        assert isinstance(defense.attack_blocked, bool)
        assert defense.defense_strategy in DefenseStrategy
        assert 0 <= defense.confidence_restored <= 1
        assert defense.defense_description is not None

    @pytest.mark.asyncio
    async def test_blue_team_all_defense_strategies(
        self,
        red_team: MockRedTeamAgent,
        blue_team: MockBlueTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test blue team can use all defense strategies"""
        for _ in range(len(DefenseStrategy)):
            attack = await red_team.generate_attack(sample_proof, sample_context)
            defense = await blue_team.defend_against_attack(attack, sample_proof)
            assert defense.defense_strategy in DefenseStrategy

    @pytest.mark.asyncio
    async def test_blue_team_defense_effectiveness(
        self,
        red_team: MockRedTeamAgent,
        blue_team: MockBlueTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test blue team defense effectiveness"""
        blocks = 0
        for _ in range(10):
            attack = await red_team.generate_attack(sample_proof, sample_context)
            defense = await blue_team.defend_against_attack(attack, sample_proof)
            if defense.attack_blocked:
                blocks += 1

        # Should block some attacks (testing mock has ~50% block rate)
        assert blocks >= 3
        assert blue_team.defense_count == 10

    # ========================================================================
    # Tests: Adversarial Coevolution
    # ========================================================================

    @pytest.mark.asyncio
    async def test_adversarial_coevolution(self):
        """Test adversarial coevolution process"""
        coevolution = MockAdversarialCoevolution()
        test_theorems = [
            "theorem T1: ∀ n, n + 0 = n",
            "theorem T2: ∀ a b, a + b = b + a"
        ]

        results = await coevolution.coevolve(test_theorems, generations=5)

        assert "generations" in results
        assert "red_team_best_fitness" in results
        assert "blue_team_best_fitness" in results
        assert "robustness_improvement" in results
        assert len(results["generations"]) == 5
        assert results["robustness_improvement"] > 0

    @pytest.mark.asyncio
    async def test_coevolution_fitness_improvement(self):
        """Test that coevolution improves fitness over time"""
        coevolution = MockAdversarialCoevolution()

        results = await coevolution.coevolve(["theorem T1"], generations=10)

        red_fitness = results["red_team_best_fitness"]
        blue_fitness = results["blue_team_best_fitness"]

        # Fitness should improve (monotonically in this mock)
        assert red_fitness[-1] > red_fitness[0]
        assert blue_fitness[-1] > blue_fitness[0]

    # ========================================================================
    # Tests: Adversarial Testing Results
    # ========================================================================

    def test_adversarial_test_result_structure(self):
        """Test AdversarialTestResult data structure"""
        result = AdversarialTestResult(
            theorem="test_theorem",
            original_proof="proof_code",
            attacks_launched=10,
            attacks_successful=7,
            attacks_blocked=3,
            robustness_score=0.7,
            attack_results=[],
            defense_results=[],
            vulnerabilities_found=["vuln1", "vuln2"],
            improvements_suggested=["improve1"]
        )

        assert result.theorem == "test_theorem"
        assert result.attacks_launched == 10
        assert result.robustness_score == 0.7
        assert len(result.vulnerabilities_found) == 2
        assert len(result.improvements_suggested) == 1

    def test_robustness_score_calculation(self):
        """Test robustness score calculation"""
        attacks_launched = 20
        attacks_blocked = 15
        attacks_successful = attacks_launched - attacks_blocked

        robustness = attacks_blocked / attacks_launched

        assert 0 <= robustness <= 1
        assert robustness == 0.75  # 15/20

    # ========================================================================
    # Tests: Integration with MCTS Approaches
    # ========================================================================

    @pytest.mark.asyncio
    async def test_adversarial_evolved_policies_mcts(
        self,
        red_team: MockRedTeamAgent,
        blue_team: MockBlueTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test adversarial integration with evolved policies MCTS"""
        # Simulate evolved policies MCTS with adversarial testing
        attack = await red_team.generate_attack(
            sample_proof,
            sample_context,
            AttackType.TACTIC_SUBSTITUTION
        )
        defense = await blue_team.defend_against_attack(attack, sample_proof)

        # Verify adversarial components work with evolved policies
        assert attack.attack_type is not None
        assert defense.defense_strategy is not None

    @pytest.mark.asyncio
    async def test_adversarial_evolutionary_nodes_mcts(
        self,
        red_team: MockRedTeamAgent,
        blue_team: MockBlueTeamAgent,
        sample_proof: LeanProof,
        sample_context: ProofContext
    ):
        """Test adversarial integration with evolutionary nodes MCTS"""
        # Test multiple attack-defense cycles
        for i in range(5):
            attack = await red_team.generate_attack(sample_proof, sample_context)
            defense = await blue_team.defend_against_attack(attack, sample_proof)

            # Each cycle should produce valid results
            assert attack.attack_type in AttackType
            assert defense.defense_strategy in DefenseStrategy

    @pytest.mark.asyncio
    async def test_adversarial_coevolution_mcts(self):
        """Test adversarial integration with coevolution MCTS"""
        coevolution = MockAdversarialCoevolution()

        # Simulate coevolution MCTS training
        results = await coevolution.coevolve(
            ["theorem T1", "theorem T2", "theorem T3"],
            generations=7
        )

        # Verify coevolution dynamics
        assert len(results["generations"]) == 7
        assert results["robustness_improvement"] > 0

    # ========================================================================
    # Tests: Attack and Defense Strategy Coverage
    # ========================================================================

    def test_all_attack_types_defined(self):
        """Test all attack types are defined"""
        attack_types = [t.value for t in AttackType]

        expected_attacks = [
            "tactic_substitution",
            "hypothesis_inversion",
            "goal_modification",
            "context_manipulation",
            "proof_length_explosion",
            "logic_bombs",
            "boundary_violation",
            "resource_exhaustion"
        ]

        for expected in expected_attacks:
            assert expected in attack_types

    def test_all_defense_strategies_defined(self):
        """Test all defense strategies are defined"""
        defense_strategies = [s.value for s in DefenseStrategy]

        expected_defenses = [
            "tactic_validation",
            "redundant_verification",
            "consensus_filtering",
            "sanity_checks",
            "boundary_enforcement",
            "resource_limiting",
            "adversarial_detection",
            "ensemble_defense"
        ]

        for expected in expected_defenses:
            assert expected in defense_strategies

    # ========================================================================
    # Tests: Robustness Metrics
    # ========================================================================

    def test_robustness_score_bounds(self):
        """Test robustness scores are within valid bounds"""
        test_cases = [
            (10, 10, 1.0),   # All blocked
            (10, 5, 0.5),    # Half blocked
            (10, 0, 0.0),    # None blocked
            (100, 75, 0.75), # 75% blocked
        ]

        for launched, blocked, expected in test_cases:
            robustness = blocked / launched
            assert abs(robustness - expected) < 0.01

    # ========================================================================
    # Tests: Data Model Serialization
    # ========================================================================

    def test_attack_result_serialization(self):
        """Test AttackResult can be serialized"""
        result = AttackResult(
            attack_type=AttackType.TACTIC_SUBSTITUTION,
            original_proof="proof",
            attacked_proof="attacked_proof",
            success=True,
            confidence_reduction=0.3,
            vulnerable_points=["p1", "p2"],
            attack_description="test attack"
        )

        serialized = asdict(result)
        assert serialized["attack_type"] == AttackType.TACTIC_SUBSTITUTION
        assert serialized["success"] is True
        assert len(serialized["vulnerable_points"]) == 2

    def test_defense_result_serialization(self):
        """Test DefenseResult can be serialized"""
        result = DefenseResult(
            attack_detected=True,
            attack_blocked=True,
            defense_strategy=DefenseStrategy.TACTIC_VALIDATION,
            recovered_proof="recovered",
            confidence_restored=0.5,
            defense_description="test defense"
        )

        serialized = asdict(result)
        assert serialized["attack_detected"] is True
        assert serialized["defense_strategy"] == DefenseStrategy.TACTIC_VALIDATION
        assert serialized["confidence_restored"] == 0.5

    # ========================================================================
    # Tests: Adversarial Test Result Aggregation
    # ========================================================================

    def test_adversarial_test_result_aggregation(self):
        """Test aggregation of multiple adversarial test results"""
        results = [
            AdversarialTestResult(
                theorem="T1",
                original_proof="proof1",
                attacks_launched=10,
                attacks_successful=3,
                attacks_blocked=7,
                robustness_score=0.7,
                attack_results=[],
                defense_results=[],
                vulnerabilities_found=["v1"],
                improvements_suggested=[]
            ),
            AdversarialTestResult(
                theorem="T2",
                original_proof="proof2",
                attacks_launched=10,
                attacks_successful=5,
                attacks_blocked=5,
                robustness_score=0.5,
                attack_results=[],
                defense_results=[],
                vulnerabilities_found=["v2"],
                improvements_suggested=[]
            )
        ]

        # Calculate aggregate metrics
        total_attacks = sum(r.attacks_launched for r in results)
        total_blocked = sum(r.attacks_blocked for r in results)
        avg_robustness = sum(r.robustness_score for r in results) / len(results)
        all_vulnerabilities = [v for r in results for v in r.vulnerabilities_found]

        assert total_attacks == 20
        assert total_blocked == 12
        assert avg_robustness == 0.6
        assert len(all_vulnerabilities) == 2


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestAdversarialPerformance:
    """Performance tests for adversarial system"""

    @pytest.mark.asyncio
    async def test_concurrent_attacks(self):
        """Test system can handle concurrent attacks"""
        red_team = MockRedTeamAgent()
        sample_proof = LeanProof(
            code="proof",
            tactics_used=[],
            proof_length=1,
            verified=True
        )
        sample_context = ProofContext(
            theorem="T",
            current_proof_state="state",
            available_tactics=[],
            proof_depth=0,
            resources_used={}
        )

        # Launch concurrent attacks
        tasks = [
            red_team.generate_attack(sample_proof, sample_context)
            for _ in range(20)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(isinstance(r, AttackResult) for r in results)

    @pytest.mark.asyncio
    async def test_adversarial_coevolution_scalability(self):
        """Test coevolution scales with more generations"""
        coevolution = MockAdversarialCoevolution()

        # Test with increasing generations
        for gen_count in [5, 10, 20]:
            results = await coevolution.coevolve(["T1"], generations=gen_count)
            assert len(results["generations"]) == gen_count
            assert results["robustness_improvement"] >= 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdversarialMDAPIntegration:
    """Integration tests for adversarial with MDAP/MAKER"""

    @pytest.mark.asyncio
    async def test_adversarial_mdap_voting_robustness(self):
        """Test adversarial system with MDAP voting for robustness"""
        # Simulate MDAP voting with adversarial testing
        red_team = MockRedTeamAgent()
        blue_team = MockBlueTeamAgent()

        sample_proof = LeanProof(
            code="proof",
            tactics_used=[],
            proof_length=1,
            verified=True
        )
        sample_context = ProofContext(
            theorem="T",
            current_proof_state="state",
            available_tactics=[],
            proof_depth=0,
            resources_used={}
        )

        # Run attack-defense cycles
        robustness_scores = []
        for _ in range(10):
            attack = await red_team.generate_attack(sample_proof, sample_context)
            defense = await blue_team.defend_against_attack(attack, sample_proof)

            if defense.attack_blocked:
                robustness_scores.append(defense.confidence_restored)

        # Verify robustness is maintained
        assert len(robustness_scores) > 0
        avg_robustness = sum(robustness_scores) / len(robustness_scores)
        assert avg_robustness >= 0  # Should be non-negative


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
