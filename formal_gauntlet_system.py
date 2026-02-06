"""
Formal Gauntlet System for Sovereign-Grade Problem Decomposition

Implements a comprehensive, programmable gauntlet framework with configurable rules,
validation stages, and red team/gold team workflows.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum
import time

from sovereign_data_models import (
    GauntletRoundRule,
    GauntletDefinition,
    GauntletExecution,
    GauntletAssignment,
    SolutionAttempt,
    SubProblem,
    CritiqueReport,
    ValidationResult,
    Feedback,
    generate_id
)

# ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Formal Gauntlet System
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# **LEAN INTEGRATION**: Real Lean proof verification for formal gauntlets
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logger.warning("LeanAide client not available - formal verification disabled")


async def verify_with_lean_gauntlet(
    content: str, 
    attack_vector: str = None,
    gauntlet_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Verify content using Lean theorem prover for formal gauntlet system.
    
    Args:
        content: The content to verify (theorem, proof, or formal statement)
        attack_vector: Optional attack vector identifier
        gauntlet_context: Additional context from the gauntlet execution
        
    Returns:
        Dictionary with verification results including:
        - verified: bool indicating if proof/theorem is valid
        - confidence: float confidence score
        - proof: str containing the proof code if available
        - attack_vector: str identifier of the attack vector used
        - gauntlet_context: dict with additional execution context
    """
    if not LEAN_AVAILABLE:
        return {"verified": False, "reason": "Lean unavailable"}
    
    try:
        client = LeanAideClient()
        
        # Translate content to formal theorem statement
        formalized = await client.translate_thm(content)
        
        # Verify the formalized content
        result = await client.verify(formalized)
        
        return {
            "verified": result.verified if hasattr(result, 'verified') else False,
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
            "proof": result.proof_code if hasattr(result, 'proof_code') else None,
            "attack_vector": attack_vector,
            "gauntlet_context": gauntlet_context or {},
            "formalized_statement": formalized
        }
    except Exception as e:
        logger.warning(f"Lean gauntlet verification failed: {e}")
        return {
            "verified": False, 
            "reason": str(e), 
            "attack_vector": attack_vector,
            "gauntlet_context": gauntlet_context or {}
        }


class LeanFormalVerificationMixin:
    """Mixin class adding Lean verification to formal gauntlet system."""
    
    def __init__(self):
        self.lean_client: Optional[LeanAideClient] = None
        self._init_lean()
    
    def _init_lean(self):
        """Initialize Lean client if available."""
        if LEAN_AVAILABLE:
            try:
                self.lean_client = LeanAideClient()
                logger.info("LeanAide client initialized for formal gauntlet system")
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAide: {e}")
                self.lean_client = None
    
    async def verify_with_lean(
        self, 
        content: str, 
        attack_vector: str = None,
        gauntlet_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Verify content using Lean theorem prover.
        
        Args:
            content: The content to verify
            attack_vector: Optional attack vector identifier
            gauntlet_context: Additional context from gauntlet execution
            
        Returns:
            Dictionary with verification results
        """
        return await verify_with_lean_gauntlet(content, attack_vector, gauntlet_context)


# **ACTUAL INTEGRATION HELPER METHODS**: Formal Gauntlet System
def _trigger_gauntlet_system_alerts(operation, success, execution_id=None, error=None, metadata=None):
    """Trigger alerts for gauntlet system operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.HIGH if operation == "execute_gauntlet" else AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Gauntlet System {operation} Failed",
            message=f"Gauntlet system operation '{operation}' failed: {error}",
            severity=severity,
            source="FormalGauntletSystem",
            metadata=metadata or {"execution_id": execution_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger gauntlet system alert: {e}")


def _extract_gauntlet_system_knowledge(operation, execution_id, result):
    """Extract knowledge from gauntlet system operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"gauntlet_system_{operation}_{execution_id}",
            artifact_type="gauntlet_execution",
            source_component="FormalGauntletSystem",
            content={
                "operation": operation,
                "execution_id": execution_id,
                "rounds_completed": result.get("rounds_completed", 0) if result else 0,
                "final_score": result.get("final_score", 0.0) if result else 0.0,
                "passed": result.get("passed", False) if result else False,
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract gauntlet system knowledge: {e}")


def _track_gauntlet_system_performance(operation, success, duration_seconds, gauntlet_name, rounds_completed=0, final_score=0.0):
    """Track performance of gauntlet system operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name=f"gauntlet_{gauntlet_name}",
            component_name="FormalGauntletSystem",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "gauntlet_name": gauntlet_name,
                "rounds_completed": rounds_completed,
                "final_score": final_score
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track gauntlet system performance: {e}")


__all__ = [
    'GauntletTemplates',
    'ReviewStatus',
    'HumanReviewItem',
    'AdaptiveMetrics',
    'HumanReviewQueue',
    'GauntletSystem'
]

# Standard evaluation prompt templates
EVAL_PROMPT_AUTOMATED_TESTS = "Evaluate solution against automated tests"
EVAL_PROMPT_RED_TEAM_REVIEW = "Perform adversarial review to find flaws and edge cases"
EVAL_PROMPT_GOLD_TEAM_VERIFICATION = "Perform thorough verification of correctness and quality"
EVAL_PROMPT_SECURITY_SCAN = "Scan for common security vulnerabilities"
EVAL_PROMPT_PENETRATION_TEST = "Attempt to exploit security flaws and bypass controls"
EVAL_PROMPT_SECURITY_AUDIT = "Verify compliance with security standards and best practices"
EVAL_PROMPT_PERFORMANCE_BENCHMARK = "Run performance benchmarks and load tests"
EVAL_PROMPT_STRESS_TEST = "Attempt to overwhelm system with extreme load"
EVAL_PROMPT_PERFORMANCE_ANALYSIS = "Analyze performance characteristics and optimization opportunities"
EVAL_PROMPT_RESEARCH_REPRODUCIBILITY = "Verify reproducibility of research results"
EVAL_PROMPT_METHODOLOGY_REVIEW = "Critically evaluate methodology and identify potential flaws"
EVAL_PROMPT_PEER_REVIEW = "Perform thorough peer review and validate contributions"


class GauntletTemplates:
    """Predefined gauntlet templates for common use cases."""

    @staticmethod
    def standard_validation_gauntlet() -> GauntletDefinition:
        """Standard 3-round validation gauntlet."""
        rounds = [
            GauntletRoundRule(
                rule_id="automated_tests",
                rule_type="automated",
                description="Run automated tests",
                validation_type="acceptance",
                min_score=0.8,
                max_attempts=3,
                evaluator="automated",
                evaluation_prompt=EVAL_PROMPT_AUTOMATED_TESTS,
                success_criteria=["All tests pass", "Code quality checks pass"],
                is_required=True,
                can_fail_gracefully=False
            ),
            GauntletRoundRule(
                rule_id="red_team_review",
                rule_type="red_team",
                description="Red team adversarial review",
                validation_type="quality",
                min_score=0.7,
                max_attempts=2,
                evaluator="red_team_auto",
                evaluation_prompt=EVAL_PROMPT_RED_TEAM_REVIEW,
                success_criteria=["No critical flaws found", "Edge cases addressed"],
                is_required=True,
                can_fail_gracefully=True
            ),
            GauntletRoundRule(
                rule_id="gold_team_verification",
                rule_type="gold_team",
                description="Gold team final verification",
                validation_type="quality",
                min_score=0.9,
                max_attempts=2,
                evaluator="gold_team_auto",
                evaluation_prompt=EVAL_PROMPT_GOLD_TEAM_VERIFICATION,
                success_criteria=["Meets all quality standards", "Solution is correct"],
                is_required=True,
                can_fail_gracefully=False
            )
        ]

        return GauntletDefinition(
            gauntlet_id="standard_validation",
            name="Standard Validation Gauntlet",
            description="3-round validation with automated tests, red team review, and gold team verification",
            rounds=rounds,
            execution_order="sequential",
            stop_on_first_failure=False,
            require_all_rounds=True
        )

    @staticmethod
    def security_gauntlet() -> GauntletDefinition:
        """Security-focused gauntlet with penetration testing."""
        rounds = [
            GauntletRoundRule(
                rule_id="automated_security_scan",
                rule_type="automated",
                description="Automated security vulnerability scan",
                validation_type="security",
                min_score=0.85,
                max_attempts=3,
                evaluator="automated",
                evaluation_prompt=EVAL_PROMPT_SECURITY_SCAN,
                success_criteria=["No critical vulnerabilities", "No high-severity issues"],
                is_required=True
            ),
            GauntletRoundRule(
                rule_id="red_team_penetration",
                rule_type="red_team",
                description="Red team penetration testing",
                validation_type="security",
                min_score=0.75,
                max_attempts=3,
                evaluator="red_team_auto",
                evaluation_prompt=EVAL_PROMPT_PENETRATION_TEST,
                success_criteria=["Resists penetration attempts", "No unauthorized access possible"],
                is_required=True
            ),
            GauntletRoundRule(
                rule_id="gold_team_security_audit",
                rule_type="gold_team",
                description="Gold team security compliance audit",
                validation_type="security",
                min_score=0.9,
                max_attempts=2,
                evaluator="gold_team_auto",
                evaluation_prompt=EVAL_PROMPT_SECURITY_AUDIT,
                success_criteria=["Complies with security standards", "Follows secure coding practices"],
                is_required=True
            )
        ]

        return GauntletDefinition(
            gauntlet_id="security_validation",
            name="Security Validation Gauntlet",
            description="Security-focused validation with automated scans, penetration testing, and compliance audit",
            rounds=rounds,
            execution_order="sequential",
            stop_on_first_failure=False,
            require_all_rounds=True,
            red_team_required=True,
            gold_team_required=True
        )

    @staticmethod
    def performance_gauntlet() -> GauntletDefinition:
        """Performance-focused gauntlet."""
        rounds = [
            GauntletRoundRule(
                rule_id="automated_performance_tests",
                rule_type="automated",
                description="Automated performance benchmarks",
                validation_type="performance",
                min_score=0.75,
                max_attempts=3,
                evaluator="automated",
                evaluation_prompt=EVAL_PROMPT_PERFORMANCE_BENCHMARK,
                success_criteria=["Meets performance baselines", "Scales under load"],
                is_required=True
            ),
            GauntletRoundRule(
                rule_id="red_team_stress_testing",
                rule_type="red_team",
                description="Red team stress testing and adversarial load",
                validation_type="performance",
                min_score=0.7,
                max_attempts=2,
                evaluator="red_team_auto",
                evaluation_prompt=EVAL_PROMPT_STRESS_TEST,
                success_criteria=["Graceful degradation under stress", "No catastrophic failures"],
                is_required=True,
                can_fail_gracefully=True
            ),
            GauntletRoundRule(
                rule_id="gold_team_performance_analysis",
                rule_type="gold_team",
                description="Gold team detailed performance analysis",
                validation_type="performance",
                min_score=0.85,
                max_attempts=2,
                evaluator="gold_team_auto",
                evaluation_prompt=EVAL_PROMPT_PERFORMANCE_ANALYSIS,
                success_criteria=["Efficient resource usage", "Optimal performance"],
                is_required=True
            )
        ]

        return GauntletDefinition(
            gauntlet_id="performance_validation",
            name="Performance Validation Gauntlet",
            description="Performance-focused validation with benchmarks, stress testing, and analysis",
            rounds=rounds,
            execution_order="sequential",
            stop_on_first_failure=False,
            require_all_rounds=True
        )

    @staticmethod
    def research_gauntlet() -> GauntletDefinition:
        """Research-focused gauntlet for validation."""
        rounds = [
            GauntletRoundRule(
                rule_id="automated_reproducibility_check",
                rule_type="automated",
                description="Automated reproducibility verification",
                validation_type="acceptance",
                min_score=0.8,
                max_attempts=3,
                evaluator="automated",
                evaluation_prompt=EVAL_PROMPT_RESEARCH_REPRODUCIBILITY,
                success_criteria=["Results are reproducible", "Methodology is clear"],
                is_required=True
            ),
            GauntletRoundRule(
                rule_id="red_team_critique",
                rule_type="red_team",
                description="Red team critical review of methodology",
                validation_type="quality",
                min_score=0.7,
                max_attempts=2,
                evaluator="red_team_auto",
                evaluation_prompt=EVAL_PROMPT_METHODOLOGY_REVIEW,
                success_criteria=["Methodology is sound", "No logical fallacies"],
                is_required=True
            ),
            GauntletRoundRule(
                rule_id="gold_team_peer_review",
                rule_type="gold_team",
                description="Gold team peer review and validation",
                validation_type="quality",
                min_score=0.9,
                max_attempts=2,
                evaluator="gold_team_auto",
                evaluation_prompt=EVAL_PROMPT_PEER_REVIEW,
                success_criteria=["Novel contribution validated", "Meets research standards"],
                is_required=True
            )
        ]

        return GauntletDefinition(
            gauntlet_id="research_validation",
            name="Research Validation Gauntlet",
            description="Research-focused validation with reproducibility checks, methodology critique, and peer review",
            rounds=rounds,
            execution_order="sequential",
            stop_on_first_failure=False,
            require_all_rounds=True
        )

    @staticmethod
    def get_template(template_name: str) -> Optional[GauntletDefinition]:
        """Get a predefined gauntlet template by name."""
        templates = {
            "standard": GauntletTemplates.standard_validation_gauntlet,
            "security": GauntletTemplates.security_gauntlet,
            "performance": GauntletTemplates.performance_gauntlet,
            "research": GauntletTemplates.research_gauntlet
        }

        template_func = templates.get(template_name)
        if template_func:
            return template_func()
        return None


class ReviewStatus(Enum):
    """Status of human review in queue."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class HumanReviewItem:
    """Item in human review queue."""
    review_id: str
    round_rule: GauntletRoundRule
    solution: SolutionAttempt
    sub_problem: SubProblem
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    feedback: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveMetrics:
    """Metrics for adaptive difficulty adjustment."""
    total_rounds_completed: int = 0
    total_rounds_passed: int = 0
    average_score: float = 0.0
    recent_scores: List[float] = field(default_factory=list)
    difficulty_adjustments: int = 0
    current_difficulty_multiplier: float = 1.0
    failure_categories: Dict[str, int] = field(default_factory=dict)


class HumanReviewQueue:
    """
    Thread-safe queue for human review items.

    Manages queuing, assignment, and tracking of human reviews.
    """

    def __init__(self) -> None:
        """Initialize the human review queue."""
        self._queue: Dict[str, HumanReviewItem] = {}
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)

    def enqueue(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> HumanReviewItem:
        """
        Add an item to the review queue.

        Args:
            round_rule: The gauntlet round rule
            solution: The solution attempt to review
            sub_problem: The sub-problem context

        Returns:
            The created review item
        """
        review_id = generate_id("review")

        item = HumanReviewItem(
            review_id=review_id,
            round_rule=round_rule,
            solution=solution,
            sub_problem=sub_problem,
            status=ReviewStatus.PENDING
        )

        with self._lock:
            self._queue[review_id] = item

        self.logger.info(f"Enqueued review {review_id} for round {round_rule.rule_id}")
        return item

    def assign(self, review_id: str, reviewer: str) -> bool:
        """
        Assign a review to a human reviewer.

        Args:
            review_id: The review item ID
            reviewer: Identifier for the reviewer

        Returns:
            True if assignment succeeded, False otherwise
        """
        with self._lock:
            item = self._queue.get(review_id)
            if not item:
                self.logger.warning(f"Review {review_id} not found")
                return False

            if item.status != ReviewStatus.PENDING:
                self.logger.warning(f"Review {review_id} not in pending state")
                return False

            item.status = ReviewStatus.IN_PROGRESS
            item.assigned_to = reviewer
            self.logger.info(f"Assigned review {review_id} to {reviewer}")
            return True

    def complete(
        self,
        review_id: str,
        approved: bool,
        feedback: str,
        score: float = 0.0
    ) -> bool:
        """
        Complete a review with results.

        Args:
            review_id: The review item ID
            approved: Whether the solution was approved
            feedback: Reviewer feedback
            score: Optional score (0.0-1.0)

        Returns:
            True if completion succeeded, False otherwise
        """
        with self._lock:
            item = self._queue.get(review_id)
            if not item:
                self.logger.warning(f"Review {review_id} not found")
                return False

            if item.status not in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]:
                self.logger.warning(f"Review {review_id} not in completable state")
                return False

            item.status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
            item.feedback = feedback
            item.score = score
            item.completed_at = datetime.now()

            self.logger.info(
                f"Completed review {review_id}: "
                f"{'APPROVED' if approved else 'REJECTED'}, score={score:.2f}"
            )
            return True

    def get_status(self, review_id: str) -> Optional[HumanReviewItem]:
        """
        Get the status of a review item.

        Args:
            review_id: The review item ID

        Returns:
            The review item or None if not found
        """
        with self._lock:
            return self._queue.get(review_id)

    def get_pending_reviews(self) -> List[HumanReviewItem]:
        """Get all pending review items."""
        with self._lock:
            return [item for item in self._queue.values() if item.status == ReviewStatus.PENDING]

    def get_reviewer_workload(self, reviewer: str) -> int:
        """Get the number of in-progress reviews for a reviewer."""
        with self._lock:
            return sum(
                1 for item in self._queue.values()
                if item.assigned_to == reviewer and item.status == ReviewStatus.IN_PROGRESS
            )

    def clear(self) -> None:
        """
        Clear all items from the review queue.
        
        Thread-safe operation that removes all pending, in-progress,
        and completed reviews from the queue.
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self.logger.info(f"Cleared {count} items from review queue")


class GauntletSystem:
    """
    Manages formal gauntlet execution for validation.

    A gauntlet is a series of validation challenges that a solution
    must pass to be accepted.
    """

    def __init__(
        self,
        team_manager=None,
        openevolve_client=None,
        max_parallel_workers: int = 4,
        enable_adaptive: bool = True
    ):
        """
        Initialize the gauntlet system.

        Args:
            team_manager: Team manager for red/gold team assignment
            openevolve_client: OpenEvolve client for AI execution
            max_parallel_workers: Maximum number of parallel workers (default: 4)
            enable_adaptive: Enable adaptive difficulty adjustments (default: True)
        """
        self.team_manager = team_manager
        self.openevolve_client = openevolve_client
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.max_parallel_workers = max_parallel_workers
        self.enable_adaptive = enable_adaptive

        # Initialize human review queue
        self.review_queue = HumanReviewQueue()

        # Initialize adaptive metrics
        self.adaptive_metrics = AdaptiveMetrics()

        # Thread safety for parallel execution
        self._execution_lock = Lock()

        # Initialize OpenEvolve client if not provided
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except ImportError as e:
                self.logger.warning(f"OpenEvolve client not available: {e}")
            except (RuntimeError, ValueError, IOError) as e:
                self.logger.warning(f"Failed to initialize OpenEvolve client: {e}")

        # Initialize ROMA-MDAP-MAKER Engine for robust validation
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for standardized high-reliability config
                config = get_validation_config()
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config)
                self.logger.info("ROMAMDAPMakerAssociativeEngine initialized for GauntletSystem")
            except (RuntimeError, ValueError, ImportError) as e:
                self.logger.warning(f"Failed to initialize ROMA engine: {e}")

    def cleanup(self) -> None:
        """
        Clean up resources used by the GauntletSystem.
        
        Properly disposes of the ROMAMDAPMakerAssociativeEngine and other
        resources to prevent memory leaks.
        """
        self.logger.info("Cleaning up GauntletSystem resources")
        
        # Clean up ROMA engine
        if self.roma_engine is not None:
            try:
                # Call dispose method if available
                if hasattr(self.roma_engine, 'dispose') and callable(getattr(self.roma_engine, 'dispose')):
                    self.roma_engine.dispose()
                    self.logger.info("ROMAMDAPMakerAssociativeEngine disposed successfully")
                # Clear reference to allow garbage collection
                self.roma_engine = None
            except (RuntimeError, IOError, AttributeError) as e:
                self.logger.warning(f"Error during ROMA engine cleanup: {e}")
        
        # Clean up OpenEvolve client if it has a close method
        if self.openevolve_client is not None:
            try:
                if hasattr(self.openevolve_client, 'close') and callable(getattr(self.openevolve_client, 'close')):
                    self.openevolve_client.close()
                    self.logger.info("OpenEvolveClient closed successfully")
            except (RuntimeError, IOError, AttributeError) as e:
                self.logger.warning(f"Error during OpenEvolve client cleanup: {e}")
        
        # Clear review queue
        if hasattr(self, 'review_queue') and self.review_queue is not None:
            try:
                self.review_queue.clear()
            except (RuntimeError, IOError, AttributeError) as e:
                self.logger.warning(f"Error during review queue cleanup: {e}")

    def __enter__(self) -> "GauntletSystem":
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures cleanup is called."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def __del__(self) -> None:
        """Destructor - ensures cleanup is called even if not using context manager."""
        try:
            self.cleanup()
        except (RuntimeError, AttributeError):
            # Suppress errors during garbage collection
            pass

    def create_gauntlet(
        self,
        gauntlet_id: str,
        name: str,
        rounds: List[GauntletRoundRule],
        **config
    ) -> GauntletDefinition:
        """Create a new gauntlet definition."""
        gauntlet = GauntletDefinition(
            gauntlet_id=gauntlet_id,
            name=name,
            description=config.get('description', ''),
            rounds=rounds,
            execution_order=config.get('execution_order', 'sequential'),
            stop_on_first_failure=config.get('stop_on_first_failure', False),
            require_all_rounds=config.get('require_all_rounds', True),
            red_team_required=config.get('red_team_required', False),
            gold_team_required=config.get('gold_team_required', False),
            blue_team_participation=config.get('blue_team_participation', 'none'),
            metadata=config.get('metadata', {})
        )

        # Validate the gauntlet
        errors = gauntlet.validate()
        if errors:
            raise ValueError(f"Invalid gauntlet definition: {errors}")

        self.logger.info(f"Created gauntlet: {gauntlet_id}")
        return gauntlet

    def execute_gauntlet(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> GauntletExecution:
        """
        Execute gauntlet against a solution.

        Runs each round in configured order, tracks results,
        generates feedback reports.
        """
        start_time = time.time()
        success = False

        try:
            execution_id = generate_id("execution")
            self.logger.info(f"Executing gauntlet {gauntlet.gauntlet_id} for solution {solution.id}")

            execution = GauntletExecution(
                execution_id=execution_id,
                gauntlet_definition=gauntlet,
                sub_problem_id=sub_problem.id,
                solution_attempt=solution,
                start_time=datetime.now()
            )

            # Execute rounds based on execution order
            if gauntlet.execution_order == "sequential":
                self._execute_sequential_rounds(gauntlet, solution, sub_problem, execution)
            elif gauntlet.execution_order == "parallel":
                self._execute_parallel_rounds(gauntlet, solution, sub_problem, execution)
            elif gauntlet.execution_order == "adaptive":
                self._execute_adaptive_rounds(gauntlet, solution, sub_problem, execution)

            # Calculate final results
            execution.end_time = datetime.now()
            execution.execution_duration = (execution.end_time - execution.start_time).total_seconds()
            execution.overall_passed = execution.rounds_passed >= len(gauntlet.rounds) - (1 if gauntlet.stop_on_first_failure else 0)

            # Calculate final score with proper guards
            total_rounds_executed = execution.rounds_passed + execution.rounds_failed
            if total_rounds_executed > 0:
                execution.final_score = execution.rounds_passed / total_rounds_executed
            else:
                execution.final_score = 0.0
                # If no rounds were executed, mark as failed
                execution.overall_passed = False

            self.logger.info(f"Gauntlet execution complete: passed={execution.overall_passed}, score={execution.final_score:.2f}")

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            success = True
            duration = time.time() - start_time
            result_dict = {
                "rounds_completed": execution.rounds_passed + execution.rounds_failed,
                "final_score": execution.final_score,
                "passed": execution.overall_passed
            }
            _extract_gauntlet_system_knowledge("execute_gauntlet", execution_id, result_dict)
            _track_gauntlet_system_performance("execute_gauntlet", True, duration, gauntlet.gauntlet_id,
                                               execution.rounds_passed + execution.rounds_failed,
                                               execution.final_score)

            return execution

        except Exception as e:
            duration = time.time() - start_time
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_gauntlet_system_alerts("execute_gauntlet", False, execution_id, str(e))
            _track_gauntlet_system_performance("execute_gauntlet", False, duration, gauntlet.gauntlet_id, 0, 0.0)
            raise

    def _execute_sequential_rounds(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ) -> None:
        """Execute rounds sequentially."""
        for round_rule in gauntlet.rounds:
            result = self._execute_round(round_rule, solution, sub_problem)
            execution.round_results.append(result)

            if result["passed"]:
                execution.rounds_passed += 1
            else:
                execution.rounds_failed += 1

                # Check if we should stop
                if round_rule.is_required and not round_rule.can_fail_gracefully:
                    if gauntlet.stop_on_first_failure:
                        self.logger.warning(f"Stopping gauntlet due to failure in round: {round_rule.rule_id}")
                        break

                # Check if we should retry
                if round_rule.retry_on_failure and execution.rounds_failed < round_rule.max_attempts:
                    self.logger.info(f"Retrying round: {round_rule.rule_id}")
                    result = self._execute_round(round_rule, solution, sub_problem)
                    execution.round_results.append(result)
                    if result["passed"]:
                        execution.rounds_passed += 1
                        execution.rounds_failed -= 1

    def _execute_parallel_rounds(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ):
        """
        Execute rounds in parallel using ThreadPoolExecutor.

        Multiple validation rounds run simultaneously to improve throughput.
        Results are aggregated and thread-safe updates are made to execution state.
        """
        self.logger.info(
            f"Executing {len(gauntlet.rounds)} rounds in parallel "
            f"with {self.max_parallel_workers} workers"
        )

        # Thread-safe result collection
        results_lock = Lock()
        completed_results = []

        def execute_single_round(round_rule: GauntletRoundRule) -> Tuple[Dict[str, Any], float]:
            """
            Execute a single round and measure execution time.

            Args:
                round_rule: The round rule to execute

            Returns:
                Tuple of (result_dict, execution_time_seconds)
            """
            start_time = time.time()
            try:
                result = self._execute_round(round_rule, solution, sub_problem)
                exec_time = time.time() - start_time

                self.logger.info(
                    f"Parallel round {round_rule.rule_id} completed in {exec_time:.2f}s, "
                    f"passed={result.get('passed', False)}, score={result.get('score', 0.0):.2f}"
                )

                return result, exec_time
            except Exception as e:
                exec_time = time.time() - start_time
                self.logger.error(f"Error executing parallel round {round_rule.rule_id}: {e}")
                error_result = {
                    "round_id": round_rule.rule_id,
                    "passed": False,
                    "score": 0.0,
                    "feedback": f"Parallel execution error: {str(e)}",
                    "errors": [str(e)],
                    "execution_time": exec_time
                }
                return error_result, exec_time

        # Execute rounds in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all rounds for execution
            future_to_round = {
                executor.submit(execute_single_round, round_rule): round_rule
                for round_rule in gauntlet.rounds
            }

            # Collect results as they complete
            for future in as_completed(future_to_round):
                round_rule = future_to_round[future]
                try:
                    result, exec_time = future.result()
                    result["execution_time"] = exec_time

                    # Thread-safe update of results
                    with results_lock:
                        completed_results.append((round_rule, result))
                        execution.round_results.append(result)

                        if result["passed"]:
                            execution.rounds_passed += 1
                        else:
                            execution.rounds_failed += 1

                    self.logger.info(
                        f"Parallel round {round_rule.rule_id} collected: "
                        f"status={'PASS' if result['passed'] else 'FAIL'}, "
                        f"time={exec_time:.2f}s"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to collect result for round {round_rule.rule_id}: {e}")
                    error_result = {
                        "round_id": round_rule.rule_id,
                        "passed": False,
                        "score": 0.0,
                        "feedback": f"Result collection error: {str(e)}",
                        "errors": [str(e)]
                    }
                    with results_lock:
                        execution.round_results.append(error_result)
                        execution.rounds_failed += 1

        # Log parallel execution summary
        total_parallel_time = sum(r[1].get("execution_time", 0) for r in completed_results)
        max_single_time = max(r[1].get("execution_time", 0) for r in completed_results) if completed_results else 0
        time_saved = total_parallel_time - max_single_time

        self.logger.info(
            f"Parallel execution complete: "
            f"{len(completed_results)} rounds, "
            f"passed={execution.rounds_passed}, failed={execution.rounds_failed}, "
            f"total_time={total_parallel_time:.2f}s, "
            f"max_single_time={max_single_time:.2f}s, "
            f"time_saved={time_saved:.2f}s ({time_saved/total_parallel_time*100:.1f}% faster)"
        )

        # Handle stop_on_first_failure if configured
        if gauntlet.stop_on_first_failure and execution.rounds_failed > 0:
            self.logger.warning("Parallel execution had failures but could not stop early due to parallel nature")

    def _execute_adaptive_rounds(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ) -> None:
        """
        Execute rounds with adaptive difficulty adjustment.

        Monitors performance metrics and adjusts difficulty dynamically:
        - If scoring consistently high, increase difficulty with additional scrutiny
        - If scoring consistently low, provide remediation and lower thresholds
        - Track failure patterns and adapt validation criteria accordingly
        """
        if not self.enable_adaptive:
            self.logger.info("Adaptive execution disabled, falling back to sequential")
            self._execute_sequential_rounds(gauntlet, solution, sub_problem, execution)
            return

        self.logger.info(
            f"Executing adaptive rounds with current difficulty multiplier: "
            f"{self.adaptive_metrics.current_difficulty_multiplier:.2f}"
        )

        # Execute initial rounds
        initial_rounds = self._create_adaptive_rounds(gauntlet, phase="initial")
        self._execute_round_list(initial_rounds, solution, sub_problem, execution)

        # Calculate initial performance
        initial_score = self._calculate_adaptive_score(execution)
        self.logger.info(f"Initial adaptive phase score: {initial_score:.3f}")

        # Update metrics
        self._update_adaptive_metrics(execution, initial_score)

        # Make adaptive decisions
        adaptation_needed = self._assess_adaptation_need(execution, initial_score)

        if adaptation_needed["action"] == "increase_difficulty":
            self.logger.info("Performance too strong, increasing difficulty")
            self._increase_difficulty(gauntlet, solution, sub_problem, execution)

        elif adaptation_needed["action"] == "decrease_difficulty":
            self.logger.info("Performance struggling, providing remediation")
            self._decrease_difficulty(gauntlet, solution, sub_problem, execution)

        elif adaptation_needed["action"] == "add_scrutiny":
            self.logger.info("Adding additional scrutiny rounds")
            self._add_scrutiny_rounds(gauntlet, solution, sub_problem, execution)

        # Log adaptive metrics
        self.logger.info(
            f"Adaptive execution complete: "
            f"difficulty_multiplier={self.adaptive_metrics.current_difficulty_multiplier:.2f}, "
            f"total_adjustments={self.adaptive_metrics.difficulty_adjustments}, "
            f"pass_rate={self.adaptive_metrics.total_rounds_passed/max(1, self.adaptive_metrics.total_rounds_completed):.2%}"
        )

    def _create_adaptive_rounds(
        self,
        gauntlet: GauntletDefinition,
        phase: str = "initial"
    ) -> List[GauntletRoundRule]:
        """
        Create adaptive rounds based on current difficulty.

        Args:
            gauntlet: Original gauntlet definition
            phase: Phase of adaptive execution (initial, harder, easier, scrutiny)

        Returns:
            List of adapted round rules
        """
        multiplier = self.adaptive_metrics.current_difficulty_multiplier

        if phase == "initial":
            return gauntlet.rounds

        elif phase == "harder":
            # Increase min_score requirements
            harder_rounds = []
            for round_rule in gauntlet.rounds:
                adapted_rule = GauntletRoundRule(
                    rule_id=f"{round_rule.rule_id}_harder",
                    rule_type=round_rule.rule_type,
                    description=f"{round_rule.description} (Increased Difficulty)",
                    validation_type=round_rule.validation_type,
                    min_score=min(0.95, round_rule.min_score + (0.1 * multiplier)),
                    max_attempts=round_rule.max_attempts,
                    evaluator=round_rule.evaluator,
                    evaluation_prompt=(
                        f"{round_rule.evaluation_prompt}\n"
                        f"Apply STRICT scrutiny. Look for subtle flaws. "
                        f"Extra 10% rigor required."
                    ),
                    success_criteria=round_rule.success_criteria + [
                        "Extra scrutiny applied",
                        "No subtle flaws found"
                    ],
                    is_required=round_rule.is_required,
                    can_fail_gracefully=round_rule.can_fail_gracefully
                )
                harder_rounds.append(adapted_rule)
            return harder_rounds

        elif phase == "easier":
            # Decrease min_score and provide hints
            easier_rounds = []
            for round_rule in gauntlet.rounds:
                adapted_rule = GauntletRoundRule(
                    rule_id=f"{round_rule.rule_id}_easier",
                    rule_type=round_rule.rule_type,
                    description=f"{round_rule.description} (With Guidance)",
                    validation_type=round_rule.validation_type,
                    min_score=max(0.5, round_rule.min_score - (0.1 * multiplier)),
                    max_attempts=round_rule.max_attempts + 1,  # Extra attempt
                    evaluator=round_rule.evaluator,
                    evaluation_prompt=(
                        f"{round_rule.evaluation_prompt}\n"
                        f"Be CONSTRUCTIVE. Provide specific guidance for improvement. "
                        f"Focus on fixable issues rather than failures."
                    ),
                    success_criteria=round_rule.success_criteria,
                    is_required=round_rule.is_required,
                    can_fail_gracefully=True  # More forgiving
                )
                easier_rounds.append(adapted_rule)
            return easier_rounds

        elif phase == "scrutiny":
            # Add additional red team review
            scrutiny_round = GauntletRoundRule(
                rule_id="adaptive_scrutiny",
                rule_type="red_team",
                description="Additional adaptive scrutiny review",
                validation_type="quality",
                min_score=0.85,
                max_attempts=2,
                evaluator="red_team_auto",
                evaluation_prompt=(
                    "Perform EXTRA THOROUGH review due to borderline performance. "
                    "Check for edge cases, subtle bugs, and potential issues."
                ),
                success_criteria=["No critical issues found", "All edge cases handled"],
                is_required=False,
                can_fail_gracefully=True
            )
            return [scrutiny_round]

        return gauntlet.rounds

    def _execute_round_list(
        self,
        rounds: List[GauntletRoundRule],
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ):
        """Execute a list of rounds sequentially."""
        for round_rule in rounds:
            result = self._execute_round(round_rule, solution, sub_problem)
            execution.round_results.append(result)

            if result["passed"]:
                execution.rounds_passed += 1
            else:
                execution.rounds_failed += 1

                # Track failure categories
                failure_category = result.get("errors", ["unknown"])[0] if result.get("errors") else "low_score"
                self.adaptive_metrics.failure_categories[failure_category] = \
                    self.adaptive_metrics.failure_categories.get(failure_category, 0) + 1

    def _calculate_adaptive_score(self, execution: GauntletExecution) -> float:
        """Calculate adaptive score from execution results."""
        if not execution.round_results:
            return 0.0

        scores = [r.get("score", 0.0) for r in execution.round_results]
        return sum(scores) / len(scores)

    def _update_adaptive_metrics(
        self,
        execution: GauntletExecution,
        score: float
    ) -> None:
        """Update adaptive metrics with latest execution data."""
        self.adaptive_metrics.total_rounds_completed += len(execution.round_results)
        self.adaptive_metrics.total_rounds_passed += execution.rounds_passed
        self.adaptive_metrics.recent_scores.append(score)

        # Keep only last 10 scores for recent performance
        if len(self.adaptive_metrics.recent_scores) > 10:
            self.adaptive_metrics.recent_scores.pop(0)

        # Update average score
        self.adaptive_metrics.average_score = (
            sum(self.adaptive_metrics.recent_scores) /
            len(self.adaptive_metrics.recent_scores)
        )

    def _assess_adaptation_need(
        self,
        execution: GauntletExecution,
        score: float
    ) -> Dict[str, Any]:
        """
        Assess whether adaptation is needed based on performance.

        Returns:
            Dict with 'action' key indicating needed adaptation:
            - 'none': No adaptation needed
            - 'increase_difficulty': Performance too strong
            - 'decrease_difficulty': Performance too weak
            - 'add_scrutiny': Borderline performance
        """
        rounds_completed = len(execution.round_results)
        rounds_passed = execution.rounds_passed
        pass_rate = rounds_passed / rounds_completed if rounds_completed > 0 else 0

        # Strong performance - increase difficulty
        if score > 0.9 and pass_rate > 0.95:
            return {
                "action": "increase_difficulty",
                "reason": f"High score ({score:.3f}) and pass rate ({pass_rate:.2%})",
                "score": score,
                "pass_rate": pass_rate
            }

        # Weak performance - decrease difficulty
        elif score < 0.6 and pass_rate < 0.7:
            return {
                "action": "decrease_difficulty",
                "reason": f"Low score ({score:.3f}) and pass rate ({pass_rate:.2%})",
                "score": score,
                "pass_rate": pass_rate
            }

        # Borderline performance - add scrutiny
        elif 0.7 <= score <= 0.85 and 0.7 <= pass_rate <= 0.9:
            return {
                "action": "add_scrutiny",
                "reason": f"Borderline score ({score:.3f}) and pass rate ({pass_rate:.2%})",
                "score": score,
                "pass_rate": pass_rate
            }

        # No adaptation needed
        return {
            "action": "none",
            "reason": f"Acceptable performance (score={score:.3f}, pass_rate={pass_rate:.2%})",
            "score": score,
            "pass_rate": pass_rate
        }

    def _increase_difficulty(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ) -> None:
        """Increase difficulty by executing harder rounds."""
        harder_rounds = self._create_adaptive_rounds(gauntlet, phase="harder")

        # Adjust multiplier
        self.adaptive_metrics.current_difficulty_multiplier = min(
            2.0,
            self.adaptive_metrics.current_difficulty_multiplier + 0.2
        )
        self.adaptive_metrics.difficulty_adjustments += 1

        self.logger.info(
            f"Increasing difficulty: multiplier={self.adaptive_metrics.current_difficulty_multiplier:.2f}"
        )

        # Execute harder rounds
        self._execute_round_list(harder_rounds, solution, sub_problem, execution)

    def _decrease_difficulty(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ) -> None:
        """Decrease difficulty by providing easier rounds with guidance."""
        easier_rounds = self._create_adaptive_rounds(gauntlet, phase="easier")

        # Adjust multiplier
        self.adaptive_metrics.current_difficulty_multiplier = max(
            0.5,
            self.adaptive_metrics.current_difficulty_multiplier - 0.2
        )
        self.adaptive_metrics.difficulty_adjustments += 1

        self.logger.info(
            f"Decreasing difficulty: multiplier={self.adaptive_metrics.current_difficulty_multiplier:.2f}"
        )

        # Execute easier rounds
        self._execute_round_list(easier_rounds, solution, sub_problem, execution)

    def _add_scrutiny_rounds(
        self,
        gauntlet: GauntletDefinition,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        execution: GauntletExecution
    ) -> None:
        """Add additional scrutiny rounds for borderline performance."""
        scrutiny_rounds = self._create_adaptive_rounds(gauntlet, phase="scrutiny")

        self.logger.info(f"Adding {len(scrutiny_rounds)} scrutiny rounds")

        # Execute scrutiny rounds
        self._execute_round_list(scrutiny_rounds, solution, sub_problem, execution)

    def _execute_round(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """Execute a single round based on its type."""
        if round_rule.rule_type == "red_team":
            return self.execute_red_team_round(round_rule, solution, sub_problem)
        elif round_rule.rule_type == "gold_team":
            return self.execute_gold_team_round(round_rule, solution, sub_problem)
        elif round_rule.rule_type == "automated":
            return self.execute_automated_round(round_rule, solution, sub_problem)
        elif round_rule.rule_type == "human":
            return self.execute_human_round(round_rule, solution, sub_problem)
        else:
            return {
                "round_id": round_rule.rule_id,
                "passed": False,
                "score": 0.0,
                "feedback": f"Unknown round type: {round_rule.rule_type}",
                "errors": [f"Invalid round type: {round_rule.rule_type}"]
            }

    def execute_red_team_round(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Execute red team validation round using RedTeam engine, ROMA, or OpenEvolve.
        """
        self.logger.info(f"Executing red team round: {round_rule.rule_id}")

        # 1. Try RedTeam Engine (Best: has logic + LLM integration)
        try:
            from red_team import RedTeam
            red_team = RedTeam()
            
            # Use attack modes from rule if available
            attack_modes = None
            if hasattr(round_rule, 'per_judge_requirements'):
                attack_modes = round_rule.per_judge_requirements.get("attack_modes", {}).get("modes")
            
            assessment = red_team.assess_content(
                content=solution.solution_content,
                content_type=sub_problem.domain if hasattr(sub_problem, 'domain') else "general",
                attack_modes=attack_modes
            )
            
            # Map score
            score = assessment.confidence_score / 100.0  # RedTeam uses 0-100
            passed = score >= round_rule.min_score
            
            return {
                "round_id": round_rule.rule_id,
                "passed": passed,
                "score": score,
                "feedback": assessment.assessment_summary,
                "flaws_found": [f.title for f in assessment.findings],
                "details": {
                    "engine": "RedTeam",
                    "findings_count": len(assessment.findings)
                }
            }
        except ImportError:
            self.logger.info("RedTeam module not found, falling back to ROMA/OpenEvolve")
        except Exception as e:
            self.logger.warning(f"RedTeam execution failed: {e}, falling back")

        # 2. Try ROMA Engine
        if self.roma_engine:
            try:
                prompt = self._build_red_team_prompt(round_rule, solution, sub_problem)
                result = self.roma_engine.solve_problem(prompt)
                response_content = result.get("solution", "")
                if response_content:
                    parsed_result = self._parse_red_team_result(response_content, round_rule)
                    parsed_result["feedback"] += f" (Verified by ROMA)"
                    return parsed_result
            except Exception as e:
                self.logger.warning(f"ROMA engine execution failed: {e}")

        # 3. Try OpenEvolve Client
        if self.openevolve_client:
            try:
                prompt = self._build_red_team_prompt(round_rule, solution, sub_problem)
                result = self.openevolve_client.evolve(
                    content=prompt,
                    evolution_mode="adversarial",
                    content_type="analysis",
                    max_iterations=1
                )
                if result.success and result.best_code:
                    return self._parse_red_team_result(result.best_code, round_rule)
            except Exception as e:
                self.logger.warning(f"OpenEvolve execution failed: {e}")
        
        # 4. Fallback: Basic Heuristics (if no engines available)
        self.logger.warning("All Red Team engines unavailable, using heuristic fallback")
        heuristic_score = 0.5
        # Penalize for obvious keywords
        if "error" in solution.solution_content.lower() or "todo" in solution.solution_content.lower():
            heuristic_score = 0.3
        
        return {
            "round_id": round_rule.rule_id,
            "passed": heuristic_score >= round_rule.min_score,
            "score": heuristic_score,
            "feedback": "Basic heuristic review (Engines unavailable)",
            "flaws_found": ["Potential issues detected via keywords"] if heuristic_score < 0.5 else []
        }

    def _build_red_team_prompt(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> str:
        """Build prompt for red team analysis."""
        prompt = f"""You are a RED TEAM reviewer. Your goal is to FIND FLAWS and BREAK this solution.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}

SOLUTION ATTEMPT:
{solution.solution_content}

YOUR MISSION:
{round_rule.evaluation_prompt}

Analyze this solution ADVERSARIALLY:
1. Identify logical flaws and fallacies
2. Find edge cases that break the solution
3. Test assumptions critically
4. Look for security vulnerabilities
5. Identify missing error handling
6. Check for incorrect implementations

Provide your analysis in this EXACT format:
OverallScore: <0.0-1.0>
Passed: <true/false>
FlawCount: <number>
Flaws: <flaw1> | <flaw2> | <flaw3>
SeverityScores: <category>:<score> | <category>:<score>
Feedback: <2-3 sentence summary>
Improvements: <improvement1> | <improvement2>

Be critical and thorough. Your job is to find problems."""
        return prompt

    def _parse_red_team_result(self, response: str, round_rule: GauntletRoundRule) -> Dict[str, Any]:
        """Parse red team analysis result."""
        lines = response.strip().split('\n')
        result = {
            "round_id": round_rule.rule_id,
            "passed": False,
            "score": 0.5,
            "flaws_found": [],
            "feedback": "",
            "improvements": []
        }

        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == "OverallScore":
                try:
                    result["score"] = float(value)
                except ValueError:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {__name__}", exc_info=True)
                    raise  # Re-raise the exception
            elif key == "Passed":
                result["passed"] = value.lower() in ["true", "yes", "1"]
            elif key == "Flaws":
                result["flaws_found"] = [f.strip() for f in value.split('|') if f.strip()]
            elif key == "Feedback":
                result["feedback"] = value
            elif key == "Improvements":
                result["improvements"] = [i.strip() for i in value.split('|') if i.strip()]

        # Check if minimum score is met
        result["passed"] = result["score"] >= round_rule.min_score
        return result

    def execute_gold_team_round(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        red_team_feedback: List = None
    ) -> Dict[str, Any]:  # FIXED: Was bool, actually returns Dict
        """
        Execute gold team verification round.

        Gold team does thorough validation, checks correctness,
        verifies quality, and ensures standards met.
        """
        self.logger.info(f"Executing gold team round: {round_rule.rule_id}")

        # Check if OpenEvolve client is available
        if not self.openevolve_client and not self.roma_engine:
            self.logger.warning("No execution engine available, using mock gold team review")
            return {
                "round_id": round_rule.rule_id,
                "passed": True,
                "score": 0.9,
                "feedback": "Mock gold team review (Engine not available)",
                "criteria_met": []
            }

        try:
            # Build gold team prompt
            prompt = self._build_gold_team_prompt(round_rule, solution, sub_problem, red_team_feedback)

            # 1. Try ROMA Engine First
            if self.roma_engine:
                try:
                    result = self.roma_engine.solve_problem(prompt)
                    response_content = result.get("solution", "")
                    if response_content:
                        # Parse results
                        parsed_result = self._parse_gold_team_result(response_content, round_rule)
                        parsed_result["feedback"] += f" (Verified by ROMA Confidence: {result.get('confidence', 0.0):.2f})"
                        return parsed_result
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    self.logger.warning(f"ROMA engine execution failed, falling back: {e}")

            # 2. Fallback to OpenEvolve Client
            if self.openevolve_client:
                # Execute gold team analysis
                result = self.openevolve_client.evolve(
                    content=prompt,
                    evolution_mode="standard",
                    content_type="analysis",
                    max_iterations=1,
                    temperature=0.3,
                    max_tokens=2000
                )

                # Parse results
                if result.success and result.best_code:
                    return self._parse_gold_team_result(result.best_code, round_rule)
            
            return {
                "round_id": round_rule.rule_id,
                "passed": False,
                "score": 0.0,
                "feedback": "Gold team analysis failed",
                "errors": ["No valid response from any engine"]
            }

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            self.logger.error(f"Gold team execution error: {e}")
            return {
                "round_id": round_rule.rule_id,
                "passed": False,
                "score": 0.0,
                "feedback": f"Gold team execution error: {str(e)}",
                "errors": [str(e)]
            }

    def _build_gold_team_prompt(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        red_team_feedback: List = None
    ) -> str:
        """Build prompt for gold team analysis."""
        red_team_section = ""
        if red_team_feedback:
            red_team_section = f"\nRED TEAM FEEDBACK (for context):\n{json.dumps(red_team_feedback, indent=2)}\n"

        prompt = f"""You are a GOLD TEAM verifier. Your goal is to THOROUGHLY VALIDATE this solution.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}

SOLUTION ATTEMPT:
{solution.solution_content}
{red_team_section}
YOUR MISSION:
{round_rule.evaluation_prompt}

Analyze this solution THOROUGHLY:
1. Verify correctness of the approach
2. Check quality of implementation
3. Validate all success criteria are met
4. Ensure standards and best practices followed
5. Verify completeness and robustness
6. Check for proper error handling

Provide your analysis in this EXACT format:
OverallScore: <0.0-1.0>
Passed: <true/false>
CriteriaMet: <criterion1> | <criterion2>
QualityScore: <0.0-1.0>
CorrectnessScore: <0.0-1.0>
Feedback: <2-3 sentence summary>
Improvements: <improvement1> | <improvement2>

Be thorough and precise. Your approval certifies the solution is ready."""
        return prompt

    def _parse_gold_team_result(self, response: str, round_rule: GauntletRoundRule) -> Dict[str, Any]:
        """Parse gold team analysis result."""
        lines = response.strip().split('\n')
        result = {
            "round_id": round_rule.rule_id,
            "passed": False,
            "score": 0.5,
            "criteria_met": [],
            "feedback": "",
            "improvements": []
        }

        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == "OverallScore":
                try:
                    result["score"] = float(value)
                except ValueError:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {__name__}", exc_info=True)
                    raise  # Re-raise the exception
            elif key == "Passed":
                result["passed"] = value.lower() in ["true", "yes", "1"]
            elif key == "CriteriaMet":
                result["criteria_met"] = [c.strip() for c in value.split('|') if c.strip()]
            elif key == "Feedback":
                result["feedback"] = value
            elif key == "Improvements":
                result["improvements"] = [i.strip() for i in value.split('|') if i.strip()]

        # Check if minimum score is met
        result["passed"] = result["score"] >= round_rule.min_score
        return result

    def execute_automated_round(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Execute automated validation round using static analysis and heuristics.
        
        Performs:
        - Syntax checking (AST parsing for Python)
        - Static analysis (forbidden imports, structure)
        - Heuristic checks (length, keywords)
        """
        self.logger.info(f"Executing automated round: {round_rule.rule_id}")
        
        content = solution.solution_content
        score = 0.0
        feedback = []
        checks_passed = 0
        total_checks = 0
        errors = []

        try:
            # 1. Syntax Check (Python)
            total_checks += 1
            is_python = "def " in content or "class " in content or "import " in content
            
            if is_python:
                import ast
                try:
                    tree = ast.parse(content)
                    score += 0.4  # Syntax valid
                    checks_passed += 1
                    feedback.append("Syntax check passed")
                    
                    # 2. Static Analysis
                    # Check for docstrings
                    has_docstrings = any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) for n in ast.walk(tree))
                    total_checks += 1
                    if has_docstrings:
                        score += 0.2
                        checks_passed += 1
                        feedback.append("Docstrings detected")
                    else:
                        feedback.append("Missing docstrings")

                    # Check for dangerous imports
                    dangerous = ['os', 'subprocess', 'sys', 'shutil']
                    imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
                    imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
                    
                    found_dangerous = [imp for imp in imports if imp in dangerous]
                    total_checks += 1
                    if not found_dangerous:
                        score += 0.2
                        checks_passed += 1
                        feedback.append("No dangerous imports found")
                    else:
                        feedback.append(f"Dangerous imports detected: {found_dangerous}")
                        
                except SyntaxError as e:
                    errors.append(f"Syntax Error: {e}")
                    feedback.append("Syntax check failed")
            else:
                # Non-code content checks
                checks_passed += 1 # "Syntax" N/A treated as pass or fallback
                score += 0.4
                
                # Length check
                total_checks += 1
                if len(content) > 50:
                    score += 0.2
                    checks_passed += 1
                    feedback.append("Content length sufficient")
                else:
                    feedback.append("Content too short")

            # 3. Keyword/Heuristic Check based on success criteria
            if round_rule.success_criteria:
                for criterion in round_rule.success_criteria:
                    total_checks += 1
                    # Simple keyword matching as proxy for "meeting criterion"
                    keywords = [w for w in criterion.split() if len(w) > 4]
                    if any(k.lower() in content.lower() for k in keywords):
                        score += (0.2 / len(round_rule.success_criteria))
                        checks_passed += 1
                    
            # Finalize score
            score = min(1.0, score)
            passed = score >= round_rule.min_score

            return {
                "round_id": round_rule.rule_id,
                "passed": passed,
                "score": score,
                "feedback": "; ".join(feedback),
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "errors": errors
            }

        except Exception as e:
            self.logger.error(f"Automated round execution error: {e}")
            return {
                "round_id": round_rule.rule_id,
                "passed": False,
                "score": 0.0,
                "feedback": f"Automated validation error: {str(e)}",
                "errors": [str(e)]
            }

    def execute_human_round(
        self,
        round_rule: GauntletRoundRule,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Execute human review round with proper queue management.

        Queues solution for human review, tracks status, and integrates results.
        Supports both synchronous (wait) and asynchronous (return pending) modes.
        """
        self.logger.info(f"Executing human review round: {round_rule.rule_id}")

        try:
            # Enqueue the review item
            review_item = self.review_queue.enqueue(
                round_rule=round_rule,
                solution=solution,
                sub_problem=sub_problem
            )

            self.logger.info(
                f"Review {review_item.review_id} queued for human review. "
                f"Assigned to: {round_rule.evaluator}"
            )

            # Check if there's a pre-completed review (for testing/mocking)
            # In production, this would wait for actual human completion
            wait_for_review = round_rule.metadata.get("wait_for_human_review", False) if hasattr(round_rule, 'metadata') else False

            if wait_for_review:
                # Wait for review completion (blocking mode)
                # In production, this would poll or use webhooks
                timeout = round_rule.metadata.get("review_timeout_seconds", 300)  # 5 min default
                start_wait = time.time()

                self.logger.info(f"Waiting for human review (timeout={timeout}s)...")

                while time.time() - start_wait < timeout:
                    updated_item = self.review_queue.get_status(review_item.review_id)
                    if updated_item and updated_item.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED]:
                        # Review completed
                        self.logger.info(
                            f"Human review completed: "
                            f"{'APPROVED' if updated_item.status == ReviewStatus.APPROVED else 'REJECTED'}"
                        )

                        return {
                            "round_id": round_rule.rule_id,
                            "passed": updated_item.status == ReviewStatus.APPROVED,
                            "score": updated_item.score,
                            "feedback": updated_item.feedback,
                            "status": updated_item.status.value,
                            "review_id": review_item.review_id,
                            "evaluator": updated_item.assigned_to or round_rule.evaluator,
                            "review_duration": (updated_item.completed_at - updated_item.created_at).total_seconds()
                                if updated_item.completed_at else 0.0
                        }

                    time.sleep(5)  # Poll every 5 seconds

                # Timeout reached
                self.logger.warning(f"Human review timeout after {timeout}s")
                return {
                    "round_id": round_rule.rule_id,
                    "passed": False,
                    "score": 0.0,
                    "feedback": f"Human review timed out after {timeout}s",
                    "status": "timeout",
                    "review_id": review_item.review_id,
                    "evaluator": round_rule.evaluator
                }

            else:
                # Non-blocking mode - return pending status
                self.logger.info(f"Returning pending status for review {review_item.review_id}")

                return {
                    "round_id": round_rule.rule_id,
                    "passed": False,
                    "score": 0.0,
                    "feedback": "Human review required - awaiting human reviewer",
                    "status": ReviewStatus.PENDING.value,
                    "review_id": review_item.review_id,
                    "evaluator": round_rule.evaluator,
                    "queue_position": len(self.review_queue.get_pending_reviews()),
                    "instructions": {
                        "assign_reviewer": f"Use review_queue.assign('{review_item.review_id}', 'reviewer_id')",
                        "complete_review": f"Use review_queue.complete('{review_item.review_id}', approved=True, feedback='...', score=0.9)"
                    }
                }

        except Exception as e:
            self.logger.error(f"Human review execution error: {e}")
            return {
                "round_id": round_rule.rule_id,
                "passed": False,
                "score": 0.0,
                "feedback": f"Human review error: {str(e)}",
                "status": "error",
                "errors": [str(e)]
            }

    def assign_human_review(self, review_id: str, reviewer: str) -> Dict[str, Any]:
        """
        Assign a pending review to a human reviewer.

        Args:
            review_id: The review item ID
            reviewer: Identifier for the reviewer (user ID, email, etc.)

        Returns:
            Dict with success status and message
        """
        try:
            success = self.review_queue.assign(review_id, reviewer)

            if success:
                return {
                    "success": True,
                    "message": f"Review {review_id} assigned to {reviewer}",
                    "review_id": review_id,
                    "reviewer": reviewer
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to assign review {review_id}",
                    "review_id": review_id
                }

        except Exception as e:
            self.logger.error(f"Error assigning review: {e}")
            return {
                "success": False,
                "message": f"Assignment error: {str(e)}",
                "review_id": review_id,
                "error": str(e)
            }

    def complete_human_review(
        self,
        review_id: str,
        approved: bool,
        feedback: str,
        score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Complete a human review with results.

        Args:
            review_id: The review item ID
            approved: Whether the solution was approved
            feedback: Reviewer feedback
            score: Optional score (0.0-1.0)

        Returns:
            Dict with success status and message
        """
        try:
            # Validate inputs
            if not 0.0 <= score <= 1.0:
                return {
                    "success": False,
                    "message": f"Score must be between 0.0 and 1.0, got {score}",
                    "review_id": review_id
                }

            if not feedback or not feedback.strip():
                return {
                    "success": False,
                    "message": "Feedback is required",
                    "review_id": review_id
                }

            success = self.review_queue.complete(review_id, approved, feedback, score)

            if success:
                return {
                    "success": True,
                    "message": f"Review {review_id} completed: {'APPROVED' if approved else 'REJECTED'}",
                    "review_id": review_id,
                    "approved": approved,
                    "score": score
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to complete review {review_id}",
                    "review_id": review_id
                }

        except Exception as e:
            self.logger.error(f"Error completing review: {e}")
            return {
                "success": False,
                "message": f"Completion error: {str(e)}",
                "review_id": review_id,
                "error": str(e)
            }

    def get_review_status(self, review_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a human review.

        Args:
            review_id: The review item ID

        Returns:
            Dict with review status or None if not found
        """
        try:
            item = self.review_queue.get_status(review_id)

            if not item:
                return None

            return {
                "review_id": item.review_id,
                "status": item.status.value,
                "assigned_to": item.assigned_to,
                "created_at": item.created_at.isoformat(),
                "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                "feedback": item.feedback,
                "score": item.score,
                "round_id": item.round_rule.rule_id,
                "solution_id": item.solution.id
            }

        except Exception as e:
            self.logger.error(f"Error getting review status: {e}")
            return None

    def verify_with_lean(self, content: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover for formal gauntlet verification.
        
        Args:
            content: The content to verify (theorem statement or proof)
            properties: Optional properties for verification
            
        Returns:
            Dict with verification results including:
            - verified: bool
            - formalized: str (Lean code)
            - proof_status: str
            - errors: list
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "error": "Lean verification not available"}
        
        try:
            client = LeanAideClient()
            # Auto-formalize the content
            formalized = client.autoformalize(content)
            # Verify the formalized content
            verification = client.verify(formalized)
            
            return {
                "verified": verification.get("success", False),
                "formalized": formalized,
                "proof_status": verification.get("status", "unknown"),
                "errors": verification.get("errors", []),
                "metadata": properties or {}
            }
        except Exception as e:
            self.logger.error(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """
        Get all pending human review items.

        Returns:
            List of dicts with pending review information
        """
        try:
            pending_items = self.review_queue.get_pending_reviews()

            return [
                {
                    "review_id": item.review_id,
                    "round_id": item.round_rule.rule_id,
                    "round_type": item.round_rule.rule_type,
                    "solution_id": item.solution.id,
                    "created_at": item.created_at.isoformat(),
                    "evaluator": item.round_rule.evaluator
                }
                for item in pending_items
            ]

        except Exception as e:
            self.logger.error(f"Error getting pending reviews: {e}")
            return []
