"""
Adversarial Adapter - Clean Interface for Adversarial Testing Module

This module provides the AdversarialAdapter class which serves as a clean
interface between UnifiedConfiguration and the adversarial testing module.

The adapter implements the Red Team / Blue Team / Evaluator Team architecture
with a simplified interface, hiding the complexity of multi-team coordination.

Usage:
    unified_config = create_unified_config({
        'evolution_mode': 'adversarial',
        'adversarial_rounds': 5,
        'attack_strength': 0.7
    })

    adapter = AdversarialAdapter(unified_config)
    result = adapter.run_adversarial_testing("Content to test")
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from unified_configuration import UnifiedConfiguration, create_unified_config

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """
    Result from adversarial testing execution.

    Attributes:
        success: Whether testing completed successfully
        final_content: The hardened/defended content
        original_content: The original input content
        total_rounds: Number of adversarial rounds completed
        vulnerabilities_found: Total vulnerabilities identified
        fixes_applied: Total fixes applied
        robustness_score: Final robustness score (0.0-1.0)
        attack_success_rate: Rate of successful attacks (0.0-1.0)
        defense_success_rate: Rate of successful defenses (0.0-1.0)
        consensus_score: Evaluator team consensus score (0.0-1.0)
        improvement_ratio: Ratio of content improvement
        duration_seconds: Total execution time
        team_results: Detailed results from each team
        vulnerabilities: List of all vulnerabilities found
        fixes: List of all fixes applied
        rounds: Detailed per-round results
        error: Error message if execution failed
    """
    success: bool
    final_content: str
    original_content: str
    total_rounds: int = 0
    vulnerabilities_found: int = 0
    fixes_applied: int = 0
    robustness_score: float = 0.0
    attack_success_rate: float = 0.0
    defense_success_rate: float = 0.0
    consensus_score: float = 0.0
    improvement_ratio: float = 0.0
    duration_seconds: float = 0.0
    team_results: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    fixes: List[Dict[str, Any]] = field(default_factory=list)
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class AdversarialAdapter:
    """
    Adapter for adversarial testing module using unified configuration.

    This class provides a clean interface for running adversarial testing
    with the Red Team / Blue Team / Evaluator Team architecture.

    Attributes:
        config: The UnifiedConfiguration instance
        _status_callback: Optional callback for status updates
    """

    def __init__(
        self,
        config: UnifiedConfiguration,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize AdversarialAdapter.

        Args:
            config: UnifiedConfiguration with all adversarial parameters
            status_callback: Optional callback for status updates
        """
        self.config = config
        self._status_callback = status_callback

        # Validate configuration
        validation = config.validate()
        if not validation.valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")

        logger.info(
            f"AdversarialAdapter initialized with "
            f"rounds={config.adversarial_rounds}, "
            f"attack_strength={config.attack_strength}"
        )

    def _update_status(self, message: str) -> None:
        """Update status if callback is provided"""
        if self._status_callback:
            self._status_callback(message)
        logger.debug(f"Adversarial status: {message}")

    def run_adversarial_testing(
        self,
        content: str,
        content_type: str = "document_general",
        use_decomposition: bool = False,
        **kwargs
    ) -> AdversarialResult:
        """
        Run comprehensive adversarial testing with Red/Blue/Evaluator teams.

        Args:
            content: Content to test adversarially
            content_type: Type of content
            use_decomposition: Whether to use problem decomposition
            **kwargs: Additional parameters to override config

        Returns:
            AdversarialResult with comprehensive testing results
        """
        start_time = time.time()

        self._update_status(f"🛡️ Starting adversarial testing...")
        self._update_status(f"🔴 Red Team vs 🔵 Blue Team vs ⚖️ Evaluator Team")

        # Merge kwargs with config
        if kwargs:
            effective_config = self.config.merge(kwargs, validate=True)
        else:
            effective_config = self.config

        try:
            # Import adversarial module
            from adversarial import run_comprehensive_adversarial_testing

            # Run adversarial testing
            self._update_status(
                f"🔄 Running {effective_config.adversarial_rounds} adversarial rounds..."
            )

            raw_result = run_comprehensive_adversarial_testing(
                current_content=content,
                content_type=content_type,
                config=effective_config.to_adversarial_config(),
                use_decomposition=use_decomposition,
                **kwargs
            )

            # Process results into AdversarialResult
            duration = time.time() - start_time

            result = AdversarialResult(
                success=raw_result.get('success', False),
                final_content=raw_result.get('final_content', content),
                original_content=raw_result.get('original_content', content),
                total_rounds=raw_result.get('metrics', {}).get('total_rounds', 0),
                vulnerabilities_found=raw_result.get('metrics', {}).get('vulnerability_count', 0),
                fixes_applied=raw_result.get('metrics', {}).get('fixes_applied', 0),
                robustness_score=raw_result.get('metrics', {}).get('robustness_score', 0.0),
                attack_success_rate=raw_result.get('metrics', {}).get('attack_success_rate', 0.0),
                defense_success_rate=raw_result.get('metrics', {}).get('defense_success_rate', 0.0),
                consensus_score=raw_result.get('metrics', {}).get('consensus_score', 0.0),
                improvement_ratio=raw_result.get('metrics', {}).get('improvement_ratio', 0.0),
                duration_seconds=duration,
                team_results=raw_result.get('team_results', {}),
                vulnerabilities=raw_result.get('vulnerabilities', []),
                fixes=raw_result.get('fixes', []),
                rounds=raw_result.get('rounds', []),
                error=raw_result.get('error')
            )

            # Log summary
            self._update_status(
                f"✅ Adversarial testing completed!\n"
                f"   Robustness Score: {result.robustness_score:.4f}\n"
                f"   Vulnerabilities Found: {result.vulnerabilities_found}\n"
                f"   Fixes Applied: {result.fixes_applied}\n"
                f"   Rounds: {result.total_rounds}\n"
                f"   Duration: {duration:.2f}s"
            )

            return result

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            duration = time.time() - start_time
            error_msg = f"Adversarial testing failed: {str(e)}"

            self._update_status(f"💥 {error_msg}")
            logger.error(error_msg, exc_info=True)

            return AdversarialResult(
                success=False,
                final_content=content,  # Return original on failure
                original_content=content,
                duration_seconds=duration,
                error=error_msg
            )

    # =========================================================================
    # CONVENIENCE METHODS FOR SPECIFIC ATTACK TYPES
    # =========================================================================

    def run_red_team_assessment(
        self,
        content: str,
        attack_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Red Team assessment only (no Blue Team fixes).

        Useful for:
        - Security auditing
        - Vulnerability scanning
        - Initial assessment before fixes

        Args:
            content: Content to assess
            attack_types: Specific attack types to use
            **kwargs: Additional parameters

        Returns:
            Red Team assessment results
        """
        kwargs['attack_types'] = attack_types or [
            'prompt_injection',
            'adversarial_examples',
            'boundary_testing'
        ]

        self._update_status(f"🔴 Running Red Team assessment only...")

        # Import Red Team directly
        try:
            from red_team import RedTeam

            red_team = RedTeam()
            assessment = red_team.assess_content(
                content=content,
                content_type=kwargs.get('content_type', 'document_general'),
                custom_requirements=f"Attack strength: {self.config.attack_strength}"
            )

            self._update_status(
                f"✅ Red Team assessment complete: {len(assessment.findings)} issues found"
            )

            return {
                'success': True,
                'assessment': assessment,
                'findings_count': len(assessment.findings) if assessment else 0
            }

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Red Team assessment failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def run_blue_team_fixes(
        self,
        content: str,
        issues: List[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Blue Team fixes for specific issues.

        Useful for:
        - Applying specific fixes
        - Testing fix effectiveness
        - Iterative improvement

        Args:
            content: Content to fix
            issues: List of issues (from Red Team or custom)
            **kwargs: Additional parameters

        Returns:
            Blue Team fix results
        """
        self._update_status(f"🔵 Running Blue Team fixes for {len(issues)} issues...")

        try:
            from blue_team import BlueTeam

            blue_team = BlueTeam()
            assessment = blue_team.apply_fixes(
                content=content,
                issues=issues,
                content_type=kwargs.get('content_type', 'document_general'),
                custom_requirements=f"Defense strength: {self.config.defense_strength}"
            )

            self._update_status(
                f"✅ Blue Team fixes complete: {len(assessment.applied_fixes)} fixes applied"
            )

            return {
                'success': True,
                'assessment': assessment,
                'fixes_applied': len(assessment.applied_fixes) if assessment else 0
            }

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Blue Team fixes failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def run_evaluator_assessment(
        self,
        content: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Evaluator Team assessment only.

        Useful for:
        - Quality assessment
        - Consensus building
        - Final validation

        Args:
            content: Content to evaluate
            **kwargs: Additional parameters

        Returns:
            Evaluator assessment results
        """
        self._update_status(f"⚖️ Running Evaluator Team assessment...")

        try:
            from evaluator_team import EvaluatorTeam

            evaluator_team = EvaluatorTeam()
            assessment = evaluator_team.evaluate_content(
                content=content,
                content_type=kwargs.get('content_type', 'document_general'),
                custom_requirements=kwargs.get('evaluation_focus', {})
            )

            score = assessment.overall_score if assessment else 0.0

            self._update_status(f"✅ Evaluator assessment complete: Score {score:.4f}")

            return {
                'success': True,
                'assessment': assessment,
                'score': score
            }

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Evaluator assessment failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_adversarial_adapter(
    parameters: Optional[Dict[str, Any]] = None,
    rounds: int = 5,
    attack_strength: float = 0.5,
    defense_strategy: str = 'reactive',
    status_callback: Optional[Callable[[str], None]] = None
) -> AdversarialAdapter:
    """
    Factory function to create AdversarialAdapter with common presets.

    Args:
        parameters: Custom parameters (uses defaults if None)
        rounds: Number of adversarial rounds
        attack_strength: Strength of attacks (0.0-1.0)
        defense_strategy: Defense strategy ('reactive', 'proactive', 'adaptive')
        status_callback: Optional status callback

    Returns:
        Configured AdversarialAdapter

    Example:
        adapter = create_adversarial_adapter(
            rounds=3,
            attack_strength=0.7,
            defense_strategy='adaptive'
        )
        result = adapter.run_adversarial_testing("My content")
    """
    base_params = {
        'evolution_mode': 'adversarial',
        'adversarial_rounds': rounds,
        'attack_strength': attack_strength,
        'defense_strategy': defense_strategy
    }

    # Merge with custom parameters
    if parameters:
        base_params.update(parameters)

    config = create_unified_config(base_params)

    return AdversarialAdapter(
        config=config,
        status_callback=status_callback
    )


# =============================================================================
# BATCH TESTING
# =============================================================================

def run_batch_adversarial_testing(
    contents: List[str],
    config: UnifiedConfiguration,
    content_type: str = "document_general",
    status_callback: Optional[Callable[[str], None]] = None
) -> List[AdversarialResult]:
    """
    Run adversarial testing on multiple contents with the same configuration.

    Args:
        contents: List of contents to test
        config: UnifiedConfiguration to use for all
        content_type: Type of all contents
        status_callback: Optional status callback

    Returns:
        List of AdversarialResult (one per input)

    Example:
        contents = ["Content 1", "Content 2", "Content 3"]
        config = create_adversarial_testing_config(rounds=3)
        results = run_batch_adversarial_testing(contents, config)

        # Find weakest content
        weakest = min(results, key=lambda r: r.robustness_score)
        print(f"Weakest content has robustness: {weakest.robustness_score}")
    """
    adapter = AdversarialAdapter(config, status_callback=status_callback)
    results = []

    for i, content in enumerate(contents, 1):
        status_callback(f"Testing item {i}/{len(contents)}...") if status_callback else None
        result = adapter.run_adversarial_testing(content, content_type)
        results.append(result)

    # Generate summary statistics
    if results:
        avg_robustness = sum(r.robustness_score for r in results) / len(results)
        total_vulnerabilities = sum(r.vulnerabilities_found for r in results)

        status_callback(
            f"\n📊 Batch Testing Summary:\n"
            f"   Average Robustness: {avg_robustness:.4f}\n"
            f"   Total Vulnerabilities: {total_vulnerabilities}\n"
        ) if status_callback else None

    return results


# =============================================================================
# TEAM-SPECIFIC PRESETS
# =============================================================================

def create_red_team_focused_config(
    rounds: int = 3,
    attack_strength: float = 0.8,
    **kwargs
) -> UnifiedConfiguration:
    """Create config focused on aggressive red teaming"""
    params = {
        'adversarial_rounds': rounds,
        'attack_strength': attack_strength,
        'defense_strength': 0.3,  # Weaker defense for more vulnerability discovery
        'attack_diversity': True,
        'adversarial_temperature': 0.9,  # Higher creativity for attacks
        **kwargs
    }
    return create_unified_config(params)


def create_blue_team_focused_config(
    rounds: int = 5,
    defense_strength: float = 0.9,
    **kwargs
) -> UnifiedConfiguration:
    """Create config focused on strong defense"""
    params = {
        'adversarial_rounds': rounds,
        'attack_strength': 0.4,  # Moderate attacks
        'defense_strength': defense_strength,
        'ensemble_defense': True,
        'defense_strategy': 'proactive',
        **kwargs
    }
    return create_unified_config(params)


def create_balanced_testing_config(
    rounds: int = 5,
    **kwargs
) -> UnifiedConfiguration:
    """Create config with balanced attack/defense"""
    params = {
        'adversarial_rounds': rounds,
        'attack_strength': 0.6,
        'defense_strength': 0.7,
        'defense_strategy': 'adaptive',
        'ensemble_defense': True,
        'attack_diversity': True,
        **kwargs
    }
    return create_unified_config(params)
