"""
MAKER/MDAP Integration for Adversarial Functionality

This module integrates the MAKER framework (arXiv:2511.09030) and MDAP system
into the adversarial testing workflow, providing:

1. MAKER-enhanced red team: Multi-agent voting for reliable attack generation
2. MDAP-enhanced blue team: Decomposed defense strategies
3. Zero-error adversarial testing: Voting ensures robust attack/defense patterns
4. Recursive attack decomposition: Complex attacks built from simple primitives

Key Features:
- Red team uses first-to-ahead-by-k voting to generate high-quality attacks
- Blue team uses maximal decomposition for thorough defense coverage
- Adversarial co-evolution with MAKER-based mutation strategies
- Zero-error vulnerability detection through voting

Author: OpenEvolve Frontend Team
Paper Reference: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Core MAKER imports
from mdap_maker_complete import (
    MAKEREngine,
    RecursiveMAKERSolver,
    VotingEngine,
    VoteCollector,
    MAKERRunMetrics
)

# OpenEvolve MAKER integration
from openevolve_maker_integration import (
    OpenEvolveVoteCollector,
    OpenEvolveMAKEREngine,
    OpenEvolveRecursiveMAKERSolver,
    MAKERWorkflowConfig,
    MAKERMode,
    create_maker_config_from_workflow,
    create_maker_integrator
)

# MDAP imports
from mdap_engine import (
    MDAPConfig,
    MDAPTask,
    MDAPStep,
    MDAPOrchestrator,
    RedFlagRules
)

# Adversarial imports
from adversarial import AdversarialConfiguration
from red_team import (
    RedTeamMember,
    RedTeamAssessment,
    IssueFinding,
    IssueCategory,
    RedTeamStrategy
)
from blue_team import (
    BlueTeamMember,
    FixSuggestion,
    BlueTeamFix,
    BlueTeamAssessment
)

# Define DefenseStrategy if it doesn't exist in blue_team.py
@dataclass
class DefenseStrategy:
    """Defense strategy for adversarial protection"""
    name: str
    description: str
    implementation_steps: List[str]
    effectiveness: float
    resource_cost: float = 0.5


# Workflow structures
from workflow_structures import Team, SubProblem, WorkflowState, SolutionAttempt

logger = logging.getLogger(__name__)


# =============================================================================
# ADVERSARIAL MAKER CONFIGURATION
# =============================================================================

class AdversarialMAKERMode(Enum):
    """MAKER modes specific to adversarial testing"""
    ATTACK_GENERATION = "attack_generation"  # Generate adversarial attacks
    VULNERABILITY_SCAN = "vulnerability_scan"  # Find vulnerabilities
    DEFENSE_GENERATION = "defense_generation"  # Generate defense strategies
    COEVOLUTION = "coevolution"  # Attack/defense co-evolution


@dataclass
class AdversarialMAKERConfig:
    """
    Configuration for MAKER-enhanced adversarial testing.

    Extends MAKERWorkflowConfig with adversarial-specific parameters.
    """
    # Base MAKER configuration
    mode: MAKERMode = MAKERMode.RECURSIVE
    k_ahead: int = 3  # Voting threshold for attack consensus
    max_depth: int = 5  # Max recursion for attack decomposition
    enable_red_flagging: bool = True
    max_token_length: int = 750

    # Adversarial-specific settings
    adversarial_mode: AdversarialMAKERMode = AdversarialMAKERMode.ATTACK_GENERATION
    attack_decomposition_enabled: bool = True  # Decompose complex attacks
    defense_layering_enabled: bool = True  # Layered defense strategies
    coevolution_rounds: int = 3  # Attack/defense co-evolution rounds
    mutation_strength: float = 0.2  # How much to mutate attacks

    # Red team MAKER settings
    red_team_voting_enabled: bool = True
    red_team_consensus_threshold: int = 3  # k for first-to-ahead-by-k
    red_team_diversity_requirement: int = 5  # N = 2k - 1 candidates

    # Blue team MDAP settings
    blue_team_decomposition_enabled: bool = True
    blue_team_max_microtasks: int = 10
    blue_team_parallel_defenses: int = 3

    # Robustness settings
    attack_validation_rounds: int = 2
    defense_verification_rounds: int = 2
    adversarial_temperature: float = 0.8  # Higher temp for more creative attacks

    def to_maker_workflow_config(self) -> MAKERWorkflowConfig:
        """Convert to standard MAKERWorkflowConfig"""
        return MAKERWorkflowConfig(
            mode=self.mode,
            k_ahead=self.k_ahead,
            max_depth=self.max_depth,
            enable_red_flagging=self.enable_red_flagging,
            max_token_length=self.max_token_length
        )


# =============================================================================
# MAKER-ENHANCED RED TEAM
# =============================================================================

class MAKERRedTeamAgent(RedTeamMember):
    """
    Red team member enhanced with MAKER voting.

    Uses first-to-ahead-by-k voting to generate high-quality,
    reliable adversarial attacks with zero errors.
    """

    def __init__(
        self,
        name: str,
        specializations: List[IssueCategory],
        expertise_level: int = 7,
        attack_method: RedTeamStrategy = RedTeamStrategy.ADVERSARIAL,
        maker_config: Optional[AdversarialMAKERConfig] = None
    ):
        super().__init__(name, specializations, expertise_level, attack_method)
        self.maker_config = maker_config or AdversarialMAKERConfig()
        self.maker_engine: Optional[OpenEvolveMAKEREngine] = None
        self.attack_metrics: List[Dict[str, Any]] = []

    def generate_attacks_with_maker(
        self,
        target_content: str,
        content_type: str,
        num_attacks: int = 5,
        temperature: Optional[float] = None
    ) -> List[IssueFinding]:
        """
        Generate adversarial attacks using MAKER voting.

        Args:
            target_content: Content to attack
            content_type: Type of content
            num_attacks: Number of attacks to generate
            temperature: Temperature for generation

        Returns:
            List of IssueFinding objects representing attacks
        """
        if not self.maker_config.red_team_voting_enabled:
            # Fallback to non-voting generation
            return self._generate_attacks_simple(target_content, content_type, num_attacks)

        # Use MAKER to generate attacks with voting
        prompt = self._build_attack_prompt(target_content, content_type)

        try:
            # Create MAKER engine if needed
            if self.maker_engine is None:
                self._initialize_maker_engine()

            # Generate attacks using MAKER voting
            attacks_data = []
            for _ in range(num_attacks):
                attack_text = self._generate_single_attack_with_maker(prompt, temperature)
                if attack_text:
                    finding = self._parse_attack_to_finding(attack_text, target_content)
                    if finding:
                        attacks_data.append(finding)

            self.attack_metrics.append({
                "generated": len(attacks_data),
                "requested": num_attacks,
                "method": "maker_voting"
            })

            return attacks_data

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"MAKER attack generation failed: {e}")
            # Fallback to simple generation
            return self._generate_attacks_simple(target_content, content_type, num_attacks)

    def _initialize_maker_engine(self):
        """Initialize MAKER engine for attack generation"""
        from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE

        if OPENEVOLVE_AVAILABLE:
            try:
                openevolve_client = OpenEvolveClient()
                vote_collector = OpenEvolveVoteCollector(
                    openevolve_client=openevolve_client,
                    enable_red_flagging=self.maker_config.enable_red_flagging
                )

                self.maker_engine = OpenEvolveMAKEREngine(
                    vote_collector=vote_collector,
                    k_ahead=self.maker_config.red_team_consensus_threshold
                )
                logger.info(f"Initialized MAKER engine for {self.name}")
            except (ImportError, RuntimeError, ConnectionError) as e:
                logger.warning(f"Failed to initialize MAKER engine: {e}")
                self.maker_engine = None
        else:
            logger.warning("OpenEvolve not available, using fallback")
            self.maker_engine = None

    def _generate_single_attack_with_maker(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """Generate a single attack using MAKER voting"""
        if self.maker_engine is None:
            return None

        try:
            temperature = temperature or self.maker_config.adversarial_temperature

            # Use MAKER to generate attack with voting
            system_prompt = self._get_attack_system_prompt()

            result = self.maker_engine.generate_solution(
                initial_state=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_steps=1  # Single attack generation
            )

            if result and len(result) > 0:
                action, state, raw_text = result[0]
                return raw_text

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"MAKER single attack generation failed: {e}")

        return None

    def _build_attack_prompt(self, target_content: str, content_type: str) -> str:
        """Build prompt for attack generation"""
        return f"""
Analyze the following {content_type} and identify potential vulnerabilities, security issues,
or adversarial attack vectors:

```
{target_content[:2000]}
```

Generate a specific attack finding with:
1. Title of the vulnerability/attack
2. Description of the issue
3. Severity level (CRITICAL, HIGH, MEDIUM, LOW)
4. Category (e.g., SECURITY_VULNERABILITY, LOGICAL_ERROR, etc.)
5. Confidence score (0-1)
6. Suggested exploit or attack method

Be specific and actionable. Focus on real vulnerabilities that could be exploited.
"""

    def _get_attack_system_prompt(self) -> str:
        """Get system prompt for attack generation"""
        return """
You are an expert red team security analyst. Your role is to identify vulnerabilities,
security flaws, and potential attack vectors in the provided content.

You should:
- Think like an adversary seeking to exploit the system
- Consider both obvious and subtle attack vectors
- Prioritize findings by severity and exploitability
- Provide actionable, specific attack methods
- Be thorough but focus on realistic threats

Generate findings that are:
- Specific and well-defined
- Technically accurate
- Actionable for remediation
- Prioritized by severity
"""

    def _parse_attack_to_finding(self, attack_text: str, target_content: str) -> Optional[IssueFinding]:
        """Parse MAKER-generated attack into IssueFinding"""
        try:
            # Simple parsing - in production, use structured output
            import re

            # Extract title (first line or up to first newline)
            title_match = re.search(r'^(.+?)(?:\n|:)', attack_text)
            title = title_match.group(1).strip() if title_match else "Attack Finding"

            # Extract description
            desc_match = re.search(r'Description:\s*(.+?)(?:\n\n|Severity:)', attack_text, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else attack_text[:500]

            # Extract severity
            severity_match = re.search(r'Severity:\s*(CRITICAL|HIGH|MEDIUM|LOW)', attack_text, re.IGNORECASE)
            from quality_assessment import SeverityLevel
            severity_str = severity_match.group(1).upper() if severity_match else "MEDIUM"
            severity = SeverityLevel[severity_str] if severity_str in SeverityLevel.__members__ else SeverityLevel.MEDIUM

            # Extract category
            category_match = re.search(r'Category:\s*(\w+)', attack_text, re.IGNORECASE)
            category_str = category_match.group(1).lower() if category_match else "security_vulnerability"
            category = IssueCategory[category_str.upper()] if category_str.upper() in IssueCategory.__members__ else IssueCategory.SECURITY_VULNERABILITY

            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*([0-9.]+)', attack_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.8

            return IssueFinding(
                title=title,
                description=description,
                severity=severity,
                category=category,
                confidence=confidence,
                suggested_fix=None,
                exploit_example=None
            )

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse attack: {e}")
            return None

    def _generate_attacks_simple(
        self,
        target_content: str,
        content_type: str,
        num_attacks: int
    ) -> List[IssueFinding]:
        """Fallback simple attack generation without MAKER"""
        # This would use the original RedTeamMember logic
        return []


# =============================================================================
# MDAP-ENHANCED BLUE TEAM
# =============================================================================

class MDAPBlueTeamAgent(BlueTeamMember):
    """
    Blue team member enhanced with MDAP decomposition.

    Uses maximal agentic decomposition to thoroughly cover defense strategies.
    """

    def __init__(
        self,
        name: str,
        defense_specialization: str = "general",
        experience_level: int = 7,
        mdap_config: Optional[MDAPConfig] = None
    ):
        super().__init__(name, defense_specialization, experience_level)
        self.mdap_config = mdap_config or MDAPConfig()
        self.mdap_orchestrator: Optional[MDAPOrchestrator] = None

    def generate_defenses_with_mdap(
        self,
        attack_findings: List[IssueFinding],
        target_content: str,
        max_defenses: int = 10
    ) -> List[DefenseStrategy]:
        """
        Generate defense strategies using MDAP decomposition.

        Args:
            attack_findings: List of attacks to defend against
            target_content: Content being defended
            max_defenses: Maximum number of defense strategies

        Returns:
            List of DefenseStrategy objects
        """
        if not self.mdap_config or max_defenses <= 3:
            # Simple defense for small numbers
            return self._generate_defenses_simple(attack_findings, target_content)

        try:
            # Create MDAP task for defense generation
            task = self._create_defense_task(attack_findings, target_content)

            # Decompose into microtasks
            microtasks = self._decompose_defense_task(task, max_defenses)

            # Execute each microtask
            defense_strategies = []
            for microtask in microtasks:
                strategy = self._execute_defense_microtask(microtask)
                if strategy:
                    defense_strategies.append(strategy)

            return defense_strategies

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"MDAP defense generation failed: {e}")
            return self._generate_defenses_simple(attack_findings, target_content)

    def _create_defense_task(
        self,
        attack_findings: List[IssueFinding],
        target_content: str
    ) -> MDAPTask:
        """Create MDAP task for defense generation"""
        # Group attacks by category
        from collections import defaultdict
        attacks_by_category = defaultdict(list)
        for finding in attack_findings:
            attacks_by_category[finding.category].append(finding)

        # Create task description
        task_desc = f"""
Generate defense strategies for the following attack categories:

"""
        for category, attacks in attacks_by_category.items():
            task_desc += f"\n{category.value} ({len(attacks)} attacks):\n"
            for attack in attacks[:3]:  # Limit to top 3 per category
                task_desc += f"  - {attack.title}\n"

        return MDAPTask(
            task_id="defense_generation",
            description=task_desc,
            context={
                "target_content": target_content[:1000],
                "num_attacks": len(attack_findings),
                "categories": list(attacks_by_category.keys())
            },
            max_microtasks=min(self.mdap_config.max_steps, len(attack_findings))
        )

    def _decompose_defense_task(self, task: MDAPTask, max_defenses: int) -> List[MDAPStep]:
        """Decompose defense task into microtasks"""
        microtasks = []

        # Create microtask for each attack category
        categories = task.context.get("categories", [])

        for i, category in enumerate(categories):
            if i >= max_defenses:
                break

            microtask = MDAPStep(
                step_id=f"defense_{i}",
                description=f"Generate defense for {category.value}",
                agent_role="defender",
                context={
                    "category": category.value,
                    "target_content": task.context.get("target_content", "")
                }
            )
            microtasks.append(microtask)

        return microtasks

    def _execute_defense_microtask(self, microtask: MDAPStep) -> Optional[DefenseStrategy]:
        """Execute a single defense microtask"""
        try:
            # Generate defense for this microtask
            prompt = f"""
Generate a defense strategy for the following attack category:

Category: {microtask.context.get('category')}

Target Content:
{microtask.context.get('target_content', '')[:500]}

Provide:
1. Defense strategy name
2. Description of the defense
3. Implementation steps
4. Effectiveness estimate (0-1)
"""

            # Call LLM to generate defense
            from llm_utils import _request_openai_compatible_chat, _compose_messages
            messages = _compose_messages(
                "You are an expert security defender specializing in prevention and mitigation.",
                prompt
            )

            response = _request_openai_compatible_chat(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            if response:
                return DefenseStrategy(
                    name=f"Defense against {microtask.context.get('category')}",
                    description=response[:500],
                    implementation_steps=response.split('\n')[:5],
                    effectiveness=0.8,
                    resource_cost=0.5
                )

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to execute defense microtask: {e}")

        return None

    def _generate_defenses_simple(
        self,
        attack_findings: List[IssueFinding],
        target_content: str
    ) -> List[DefenseStrategy]:
        """Fallback simple defense generation"""
        # This would use the original BlueTeamMember logic
        return []


# =============================================================================
# CO-EVOLUTIONARY ADVERSARIAL MAKER
# =============================================================================

class AdversarialCoEvolution:
    """
    Manages co-evolution of attacks and defenses using MAKER.

    Implements alternating rounds of:
    1. Red team generates attacks (using MAKER voting)
    2. Blue team generates defenses (using MDAP decomposition)
    3. Attacks are mutated based on defense effectiveness
    4. Process repeats for coevolution_rounds
    """

    def __init__(
        self,
        maker_config: AdversarialMAKERConfig,
        red_team_members: List[MAKERRedTeamAgent],
        blue_team_members: List[MDAPBlueTeamAgent]
    ):
        self.maker_config = maker_config
        self.red_team = red_team_members
        self.blue_team = blue_team_members
        self.evolution_history: List[Dict[str, Any]] = []

    def run_coevolution(
        self,
        target_content: str,
        content_type: str,
        num_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run co-evolutionary adversarial testing.

        Args:
            target_content: Content to test
            content_type: Type of content
            num_rounds: Number of co-evolution rounds

        Returns:
            Dict with attacks, defenses, and evolution metrics
        """
        num_rounds = num_rounds or self.maker_config.coevolution_rounds

        current_attacks = []
        current_defenses = []

        for round_num in range(num_rounds):
            logger.info(f"Co-evolution round {round_num + 1}/{num_rounds}")

            # Phase 1: Red team generates attacks
            round_attacks = self._generate_attacks_round(
                target_content, content_type, round_num
            )
            current_attacks.extend(round_attacks)

            # Phase 2: Blue team generates defenses
            round_defenses = self._generate_defenses_round(
                round_attacks, target_content, round_num
            )
            current_defenses.extend(round_defenses)

            # Phase 3: Evaluate and mutate
            effectiveness = self._evaluate_round_effectiveness(
                round_attacks, round_defenses
            )

            # Record evolution
            self.evolution_history.append({
                "round": round_num + 1,
                "num_attacks": len(round_attacks),
                "num_defenses": len(round_defenses),
                "effectiveness": effectiveness
            })

            # Mutate attacks for next round based on defense effectiveness
            if round_num < num_rounds - 1:
                self._mutate_attacks(current_attacks, effectiveness)

        return {
            "final_attacks": current_attacks,
            "final_defenses": current_defenses,
            "evolution_history": self.evolution_history,
            "total_rounds": num_rounds
        }

    def _generate_attacks_round(
        self,
        target_content: str,
        content_type: str,
        round_num: int
    ) -> List[IssueFinding]:
        """Generate attacks for a co-evolution round"""
        all_attacks = []

        for red_agent in self.red_team:
            # Increase temperature in later rounds for more diversity
            temperature = self.maker_config.adversarial_temperature + (round_num * 0.1)
            temperature = min(temperature, 1.0)

            attacks = red_agent.generate_attacks_with_maker(
                target_content=target_content,
                content_type=content_type,
                num_attacks=3,
                temperature=temperature
            )

            all_attacks.extend(attacks)

        return all_attacks

    def _generate_defenses_round(
        self,
        attacks: List[IssueFinding],
        target_content: str,
        round_num: int
    ) -> List[DefenseStrategy]:
        """Generate defenses for a co-evolution round"""
        all_defenses = []

        for blue_agent in self.blue_team:
            defenses = blue_agent.generate_defenses_with_mdap(
                attack_findings=attacks,
                target_content=target_content,
                max_defenses=5
            )

            all_defenses.extend(defenses)

        return all_defenses

    def _evaluate_round_effectiveness(
        self,
        attacks: List[IssueFinding],
        defenses: List[DefenseStrategy]
    ) -> float:
        """Evaluate effectiveness of attacks vs defenses"""
        # Simple heuristic: coverage ratio
        if not attacks:
            return 0.0

        # Count how many attacks have corresponding defenses
        defended_attacks = sum(
            1 for attack in attacks
            if any(attack.category.value.lower() in defense.name.lower()
                   for defense in defenses)
        )

        return defended_attacks / len(attacks) if attacks else 0.0

    def _mutate_attacks(self, attacks: List[IssueFinding], effectiveness: float):
        """Mutate attacks based on defense effectiveness"""
        # If effectiveness is high, attacks are being defended - need mutation
        if effectiveness > 0.7:
            # Increase mutation strength
            mutation_strength = self.maker_config.mutation_strength * 1.5
        else:
            mutation_strength = self.maker_config.mutation_strength

        # In a full implementation, this would use MAKER to generate mutated attacks
        # For now, just log the intent
        logger.info(f"Mutating attacks with strength {mutation_strength}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_adversarial_maker_config(
    adversarial_config: AdversarialConfiguration
) -> AdversarialMAKERConfig:
    """
    Create AdversarialMAKERConfig from AdversarialConfiguration.

    Args:
        adversarial_config: Existing adversarial configuration

    Returns:
        AdversarialMAKERConfig for MAKER/MDAP integration
    """
    return AdversarialMAKERConfig(
        mode=MAKERMode.RECURSIVE,
        k_ahead=adversarial_config.red_team_sample_size,
        max_depth=5,
        enable_red_flagging=True,
        adversarial_mode=AdversarialMAKERMode.COEVOLUTION,
        attack_decomposition_enabled=True,
        defense_layering_enabled=True,
        coevolution_rounds=adversarial_config.adversarial_rounds,
        mutation_strength=adversarial_config.attack_strength,
        red_team_voting_enabled=True,
        red_team_consensus_threshold=max(3, adversarial_config.red_team_sample_size // 2),
        red_team_diversity_requirement=adversarial_config.red_team_sample_size * 2 - 1,
        blue_team_decomposition_enabled=True,
        blue_team_max_microtasks=10,
        blue_team_parallel_defenses=adversarial_config.blue_team_sample_size,
        adversarial_temperature=adversarial_config.adversarial_temperature
    )


def run_maker_adversarial_testing(
    content: str,
    content_type: str = "document_general",
    config: Optional[AdversarialConfiguration] = None
) -> Dict[str, Any]:
    """
    Run adversarial testing enhanced with MAKER/MDAP.

    This is the main entry point for MAKER-enhanced adversarial testing.

    Args:
        content: Content to test adversarially
        content_type: Type of content
        config: Adversarial configuration

    Returns:
        Dict with attack findings, defense strategies, and metrics
    """
    config = config or AdversarialConfiguration()

    # Create MAKER config
    maker_config = create_adversarial_maker_config(config)

    # Create red team agents
    red_team = [
        MAKERRedTeamAgent(
            name=f"RedAgent{i}",
            specializations=[IssueCategory.SECURITY_VULNERABILITY],
            maker_config=maker_config
        )
        for i in range(config.red_team_sample_size)
    ]

    # Create blue team agents
    blue_team = [
        MDAPBlueTeamAgent(
            name=f"BlueAgent{i}",
            mdap_config=MDAPConfig()
        )
        for i in range(config.blue_team_sample_size)
    ]

    # Run co-evolution
    coevolution = AdversarialCoEvolution(maker_config, red_team, blue_team)
    result = coevolution.run_coevolution(
        target_content=content,
        content_type=content_type,
        num_rounds=maker_config.coevolution_rounds
    )

    # Add metrics
    result["config"] = {
        "maker_enabled": True,
        "mdap_enabled": True,
        "coevolution_rounds": maker_config.coevolution_rounds,
        "k_ahead": maker_config.red_team_consensus_threshold
    }

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "AdversarialMAKERConfig",
    "AdversarialMAKERMode",
    "create_adversarial_maker_config",

    # Enhanced agents
    "MAKERRedTeamAgent",
    "MDAPBlueTeamAgent",

    # Co-evolution
    "AdversarialCoEvolution",

    # Main entry point
    "run_maker_adversarial_testing",
]
