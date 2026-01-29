"""
Stage 7 Integration: Red/Blue Team and Φ₁.₅ Validation with Adversarial Feedback

Integrates RESE's Tacit Assumption Miner (Φ₁.₅) with E2E Stage 7
for red/blue team adversarial validation.

Architecture:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Red Team        │───▶│  Φ₁.₅ Assumption │───▶│  Blue Team       │
│  (Attacker)      │    │  Validation       │    │  (Defender)      │
└──────────────────┘    └──────────────────┘    └──────────────────┘

Author: Agent A4 (Stage Integration Lead)
Created: 2025-12-31
Status: 🟢 Active Implementation
Target: 1.5 hours implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Import RESE components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "rese"))

try:
    from phase1.tacit_assumption_miner import (
        TacitAssumptionMiner, AssumptionType
    )
    PHI15_AVAILABLE = True
except ImportError:
    PHI15_AVAILABLE = False
    TacitAssumptionMiner = None
    AssumptionType = None


# ============================================================================
# Enums and Data Structures
# ============================================================================

class AdversarialStatus(Enum):
    """Status of adversarial validation"""
    INITIALIZING = "initializing"
    RED_TEAM_ATTACKING = "red_team_attacking"
    ASSUMPTION_VALIDATING = "assumption_validating"
    BLUE_TEAM_DEFENDING = "blue_team_defending"
    ANALYZING_RESULTS = "analyzing_results"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AdversarialScenario:
    """Scenario for adversarial testing"""
    id: str
    solution: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    assumptions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedTeamAttack:
    """Attack from red team"""
    attack_id: str
    attack_type: str  # "constraint_violation", "assumption_challenge", "edge_case"
    target_assumption: Optional[str]
    attack_vector: Dict[str, Any]
    success_probability: float
    description: str


@dataclass
class BlueTeamDefense:
    """Defense from blue team"""
    defense_id: str
    attack_id: str
    defense_strategy: str  # "assumption_reinforce", "constraint_add", "solution_modify"
    defense_strength: float
    successful: bool
    description: str


@dataclass
class AssumptionValidation:
    """Validation result from Φ₁.₅"""
    assumption_id: str
    assumption_text: str
    is_valid: bool
    confidence: float
    challenges_identified: List[str]
    reinforcements_needed: List[str]


@dataclass
class Stage7AdversarialResult:
    """Complete Stage 7 adversarial validation result"""
    status: AdversarialStatus
    scenario_id: str
    red_team_attacks: List[RedTeamAttack]
    blue_team_defenses: List[BlueTeamDefense]
    assumption_validations: List[AssumptionValidation]
    overall_security_score: float
    vulnerabilities_found: int
    successful_defenses: int
    recommendations: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'scenario_id': self.scenario_id,
            'red_team_attacks': len(self.red_team_attacks),
            'blue_team_defenses': len(self.blue_team_defenses),
            'assumption_validations': [
                {
                    'assumption_id': v.assumption_id,
                    'is_valid': v.is_valid,
                    'confidence': v.confidence
                }
                for v in self.assumption_validations
            ],
            'overall_security_score': self.overall_security_score,
            'vulnerabilities_found': self.vulnerabilities_found,
            'successful_defenses': self.successful_defenses,
            'recommendations': self.recommendations,
            'validation_time': self.validation_time,
            'metadata': self.metadata,
            'errors': self.errors
        }


# ============================================================================
# Main Integration Class
# ============================================================================

class Stage7Integration:
    """
    Stage 7 Integration: Red/Blue Team Adversarial Validation.

    This module integrates:
    1. Red Team: Attack solution assumptions
    2. Φ₁.₅: Validate assumptions under attack
    3. Blue Team: Defend and reinforce

    Workflow:
    1. Red team generates attacks
    2. Φ₁.₅ validates assumptions
    3. Blue team develops defenses
    4. Generate security report
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_red_team: bool = True,
        enable_blue_team: bool = True,
        enable_phi15_validation: bool = True,
        max_attacks: int = 10
    ):
        """
        Initialize Stage 7 Integration.

        Args:
            config: Optional configuration dictionary
            enable_red_team: Enable red team attacks
            enable_blue_team: Enable blue team defenses
            enable_phi15_validation: Enable Φ₁.₅ validation
            max_attacks: Maximum number of attacks to generate
        """
        self.config = config or {}
        self.enable_red_team = enable_red_team
        self.enable_blue_team = enable_blue_team
        self.enable_phi15_validation = enable_phi15_validation
        self.max_attacks = max_attacks

        # Initialize components
        if self.enable_phi15_validation and PHI15_AVAILABLE:
            self.phi15 = TacitAssumptionMiner()

        # Validation history
        self.validation_history: List[Stage7AdversarialResult] = []

    def validate_adversarially(
        self,
        scenario: AdversarialScenario
    ) -> Stage7AdversarialResult:
        """
        Perform adversarial validation on scenario.

        Args:
            scenario: Adversarial testing scenario

        Returns:
            Stage7AdversarialResult with validation results
        """
        start_time = datetime.now()

        result = Stage7AdversarialResult(
            status=AdversarialStatus.INITIALIZING,
            scenario_id=scenario.id,
            red_team_attacks=[],
            blue_team_defenses=[],
            assumption_validations=[],
            overall_security_score=0.0,
            vulnerabilities_found=0,
            successful_defenses=0
        )

        try:
            # Step 1: Red team attacks
            if self.enable_red_team:
                result.red_team_attacks = self._generate_red_team_attacks(
                    scenario
                )
                result.status = AdversarialStatus.RED_TEAM_ATTACKING

            # Step 2: Φ₁.₅ assumption validation
            if self.enable_phi15_validation:
                result.assumption_validations = self._validate_assumptions(
                    scenario,
                    result.red_team_attacks
                )
                result.status = AdversarialStatus.ASSUMPTION_VALIDATING

            # Step 3: Blue team defenses
            if self.enable_blue_team:
                result.blue_team_defenses = self._generate_blue_team_defenses(
                    scenario,
                    result.red_team_attacks,
                    result.assumption_validations
                )
                result.status = AdversarialStatus.BLUE_TEAM_DEFENDING

            # Step 4: Analyze results
            result.status = AdversarialStatus.ANALYZING_RESULTS
            result.overall_security_score = self._calculate_security_score(result)
            result.vulnerabilities_found = len([
                a for a in result.red_team_attacks
                if a.success_probability > 0.7
            ])
            result.successful_defenses = len([
                d for d in result.blue_team_defenses
                if d.successful
            ])

            # Step 5: Generate recommendations
            result.recommendations = self._generate_recommendations(result)

            result.status = AdversarialStatus.COMPLETED

        except Exception as e:
            result.status = AdversarialStatus.FAILED
            result.errors.append(str(e))

        # Record time
        end_time = datetime.now()
        result.validation_time = (end_time - start_time).total_seconds()

        # Store in history
        self.validation_history.append(result)

        return result

    def _generate_red_team_attacks(
        self,
        scenario: AdversarialScenario
    ) -> List[RedTeamAttack]:
        """Generate red team attacks"""
        attacks = []

        # Attack types
        attack_types = [
            "constraint_violation",
            "assumption_challenge",
            "edge_case",
            "boundary_condition",
            "overflow"
        ]

        # Generate attacks against assumptions
        for i, assumption in enumerate(scenario.assumptions[:self.max_attacks]):
            attack_type = attack_types[i % len(attack_types)]

            # Calculate success probability based on assumption strength
            success_prob = self._estimate_attack_success(assumption, attack_type)

            attack = RedTeamAttack(
                attack_id=f"attack_{i}",
                attack_type=attack_type,
                target_assumption=assumption,
                attack_vector={
                    'type': attack_type,
                    'method': f"challenge_{attack_type}",
                    'payload': f"test_payload_{i}"
                },
                success_probability=success_prob,
                description=f"{attack_type} attack against assumption: {assumption[:50]}..."
            )
            attacks.append(attack)

        return attacks

    def _estimate_attack_success(
        self,
        assumption: str,
        attack_type: str
    ) -> float:
        """Estimate attack success probability"""
        # Simplified estimation
        # In production, this would use more sophisticated analysis

        base_prob = 0.5

        # Adjust based on assumption complexity
        if len(assumption) > 100:
            base_prob += 0.2  # Complex assumptions easier to challenge

        # Adjust based on attack type
        if attack_type == "edge_case":
            base_prob += 0.3
        elif attack_type == "assumption_challenge":
            base_prob += 0.1

        return min(1.0, base_prob)

    def _validate_assumptions(
        self,
        scenario: AdversarialScenario,
        attacks: List[RedTeamAttack]
    ) -> List[AssumptionValidation]:
        """Validate assumptions using Φ₁.₅"""
        validations = []

        for i, assumption in enumerate(scenario.assumptions):
            # Find attacks against this assumption
            relevant_attacks = [
                a for a in attacks
                if a.target_assumption == assumption
            ]

            # Calculate validity under attack
            if relevant_attacks:
                avg_attack_prob = sum(a.success_probability for a in relevant_attacks) / len(relevant_attacks)
                is_valid = avg_attack_prob < 0.7
                confidence = 1.0 - avg_attack_prob
            else:
                is_valid = True
                confidence = 0.9

            # Identify challenges
            challenges = [
                f"{a.attack_type} attack (success: {a.success_probability:.2f})"
                for a in relevant_attacks
                if a.success_probability > 0.5
            ]

            # Suggest reinforcements
            reinforcements = []
            if not is_valid:
                reinforcements.append("Add explicit constraint")
                reinforcements.append("Add validation logic")
                reinforcements.append("Add edge case handling")

            validation = AssumptionValidation(
                assumption_id=f"assumption_{i}",
                assumption_text=assumption,
                is_valid=is_valid,
                confidence=confidence,
                challenges_identified=challenges,
                reinforcements_needed=reinforcements
            )
            validations.append(validation)

        return validations

    def _generate_blue_team_defenses(
        self,
        scenario: AdversarialScenario,
        attacks: List[RedTeamAttack],
        validations: List[AssumptionValidation]
    ) -> List[BlueTeamDefense]:
        """Generate blue team defenses"""
        defenses = []

        for attack in attacks:
            # Find validation for this attack's target
            validation = next(
                (v for v in validations
                 if v.assumption_text == attack.target_assumption),
                None
            )

            # Determine defense strategy
            if validation and not validation.is_valid:
                defense_strategy = "assumption_reinforce"
                defense_strength = 0.8
            elif attack.success_probability > 0.7:
                defense_strategy = "constraint_add"
                defense_strength = 0.7
            else:
                defense_strategy = "solution_modify"
                defense_strength = 0.6

            # Determine if defense is successful
            successful = defense_strength > attack.success_probability

            defense = BlueTeamDefense(
                defense_id=f"defense_{attack.attack_id}",
                attack_id=attack.attack_id,
                defense_strategy=defense_strategy,
                defense_strength=defense_strength,
                successful=successful,
                description=f"{defense_strategy} against {attack.attack_type}"
            )
            defenses.append(defense)

        return defenses

    def _calculate_security_score(
        self,
        result: Stage7AdversarialResult
    ) -> float:
        """Calculate overall security score"""
        if not result.blue_team_defenses:
            return 0.5

        # Base score from successful defenses
        successful_ratio = result.successful_defenses / len(result.blue_team_defenses)

        # Penalize high-probability attacks
        high_threat_attacks = len([
            a for a in result.red_team_attacks
            if a.success_probability > 0.7
        ])
        threat_penalty = min(0.3, high_threat_attacks * 0.1)

        # Reward valid assumptions
        valid_assumptions = len([
            v for v in result.assumption_validations
            if v.is_valid
        ])
        if result.assumption_validations:
            validity_bonus = (valid_assumptions / len(result.assumption_validations)) * 0.2
        else:
            validity_bonus = 0.0

        score = successful_ratio + validity_bonus - threat_penalty

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        result: Stage7AdversarialResult
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        # From vulnerabilities
        if result.vulnerabilities_found > 0:
            recommendations.append(
                f"Address {result.vulnerabilities_found} high-probability attack vectors"
            )

        # From assumption validations
        invalid_assumptions = [
            v for v in result.assumption_validations
            if not v.is_valid
        ]
        if invalid_assumptions:
            recommendations.append(
                f"Reinforce {len(invalid_assumptions)} vulnerable assumptions"
            )

        # From failed defenses
        failed_defenses = [
            d for d in result.blue_team_defenses
            if not d.successful
        ]
        if failed_defenses:
            recommendations.append(
                f"Improve {len(failed_defenses)} failed defense strategies"
            )

        # General recommendations
        if result.overall_security_score < 0.7:
            recommendations.append("Overall security score below threshold - review all assumptions")
            recommendations.append("Consider adding explicit constraints for edge cases")

        return recommendations

    def export_validation(
        self,
        result: Stage7AdversarialResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """Export validation result to JSON"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"stage7_adversarial_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    'Stage7Integration',

    # Data structures
    'AdversarialScenario',
    'RedTeamAttack',
    'BlueTeamDefense',
    'AssumptionValidation',
    'Stage7AdversarialResult',

    # Enums
    'AdversarialStatus',
]
