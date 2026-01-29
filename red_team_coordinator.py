"""
Red Team Coordinator for OpenEvolve
Enhanced orchestration system using OpenEvolve's ensemble functionality
for coordinating multi-agent parallel adversarial testing.

Architecture:
    Content → RedTeamCoordinator → LLMEnsemble (parallel)
                         ↓
                    Attack Task Distribution
                         ↓
                    Vulnerability Aggregation
                         ↓
                    Security Findings

This refactored version uses OpenEvolve's LLMEnsemble for coordination instead of
custom team member management, providing better parallelization and reliability
for adversarial testing while maintaining all Red Team security capabilities.
"""

import os
import json
import time
import asyncio
import logging
import threading
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
# import pickle  # REMOVED - security risk
import hashlib
import random

# Import existing Red Team components
try:
    from red_team import (
        RedTeam, RedTeamMember, RedTeamAssessment, IssueFinding,
        IssueCategory, RedTeamStrategy, SeverityLevel
    )
    RED_TEAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Red Team not available: {e}")
    RED_TEAM_AVAILABLE = False

# Import OpenEvolve ensemble components
try:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.config import LLMModelConfig
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenEvolve ensemble not available: {e}")
    ENSEMBLE_AVAILABLE = False

# Import ROMA-MDAP-MAKER (Robust Execution)
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

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class AttackTaskStatus(Enum):
    """Status of an attack task in the coordinator"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class AttackTaskPriority(Enum):
    """Priority levels for attack tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LoadBalancingStrategy(Enum):
    """Strategies for distributing attack tasks"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    SPECIALIZATION_BASED = "specialization_based"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

@dataclass
class AttackerMetrics:
    """Performance metrics for an attacker (or ensemble model)"""
    attacker_name: str
    attacks_completed: int = 0
    attacks_failed: int = 0
    total_time_spent: float = 0.0
    average_attack_time: float = 0.0
    current_load: int = 0
    specialization_scores: Dict[IssueCategory, float] = field(default_factory=dict)
    reliability_score: float = 1.0
    last_active: Optional[datetime] = None
    model_weight: float = 1.0  # Weight for ensemble model selection
    vulnerabilities_found: int = 0

@dataclass
class AttackCoordinationTask:
    """An attack task to be coordinated"""
    task_id: str
    target_content: str
    content_type: str
    attack_category: IssueCategory
    attack_strategy: RedTeamStrategy
    priority: AttackTaskPriority = AttackTaskPriority.MEDIUM
    assigned_attacker: Optional[str] = None
    status: AttackTaskStatus = AttackTaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[RedTeamAssessment] = None
    findings: List[IssueFinding] = field(default_factory=list)
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

@dataclass
class RedTeamCoordinationSession:
    """A coordination session for adversarial testing"""
    session_id: str
    target_content: str
    content_type: str
    attack_tasks: List[AttackCoordinationTask] = field(default_factory=list)
    status: AttackTaskStatus = AttackTaskStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    aggregated_findings: List[IssueFinding] = field(default_factory=list)
    severity_breakdown: Dict[SeverityLevel, int] = field(default_factory=dict)

@dataclass
class RedTeamCoordinatorMetrics:
    """Overall coordinator metrics"""
    total_sessions: int = 0
    total_attack_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    vulnerabilities_found: int = 0
    average_session_time: float = 0.0
    average_task_time: float = 0.0
    team_utilization: float = 0.0
    throughput_tasks_per_minute: float = 0.0
    peak_concurrent_tasks: int = 0


# =============================================================================
# RED TEAM COORDINATOR
# =============================================================================

class RedTeamCoordinator:
    """
    Enhanced orchestration system using OpenEvolve's ensemble for adversarial testing.

    Features:
    - Parallel attack execution using LLMEnsemble
    - Intelligent attack distribution via ensemble weights
    - Vulnerability aggregation with ensemble consensus
    - Progress tracking and performance monitoring
    - State persistence and recovery
    - Backward compatibility with legacy Red Team mode
    - Specialized attack strategies (adversarial, systematic, focused, etc.)

    The coordinator maintains all Red Team capabilities while leveraging ensemble
    for more efficient parallel processing of attack vectors.
    """

    def __init__(
        self,
        red_team: Optional[RedTeam] = None,
        ensemble: Optional[LLMEnsemble] = None,
        max_concurrent_attacks: int = 5,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.SPECIALIZATION_BASED,
        task_timeout: int = 300,  # 5 minutes
        enable_persistence: bool = True,
        persistence_path: str = "./red_team_coordinator_state.pkl",
        auto_scale: bool = False,
        min_attackers: int = 2,
        max_attackers: int = 10,
        use_ensemble: bool = True,  # Flag to enable ensemble mode
        attack_categories: Optional[List[IssueCategory]] = None,
        diversify_attacks: bool = True  # Use different temperatures for diverse attacks
    ):
        """
        Initialize the Red Team Coordinator.

        Args:
            red_team: RedTeam instance to coordinate (legacy mode)
            ensemble: LLMEnsemble instance for ensemble-based coordination
            max_concurrent_attacks: Maximum number of attacks to run concurrently
            load_balancing_strategy: Strategy for distributing attack tasks
            task_timeout: Timeout in seconds for individual attack tasks
            enable_persistence: Enable state persistence
            persistence_path: Path to persist state
            auto_scale: Enable auto-scaling of attackers/models
            min_attackers: Minimum number of attackers when auto-scaling
            max_attackers: Maximum number of attackers when auto-scaling
            use_ensemble: Use ensemble-based coordination (True) or legacy mode (False)
            attack_categories: Default attack categories to use
            diversify_attacks: Use diverse temperatures for varied adversarial perspectives
        """
        self.red_team = red_team or (RedTeam() if RED_TEAM_AVAILABLE else None)
        self.ensemble = ensemble
        self.max_concurrent_attacks = max_concurrent_attacks
        self.load_balancing_strategy = load_balancing_strategy
        self.task_timeout = task_timeout
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        self.auto_scale = auto_scale
        self.min_attackers = min_attackers
        self.max_attackers = max_attackers
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE
        self.attack_categories = attack_categories or [
            IssueCategory.SECURITY_VULNERABILITY,
            IssueCategory.LOGICAL_ERROR,
            IssueCategory.PERFORMANCE_PROBLEM,
            IssueCategory.COMPLIANCE_ISSUE,
            IssueCategory.EDGE_CASE
        ]
        self.diversify_attacks = diversify_attacks

        # Task management
        self.attack_queue: queue.Queue = queue.Queue()
        self.active_attacks: Dict[str, AttackCoordinationTask] = {}
        self.completed_attacks: Dict[str, AttackCoordinationTask] = {}
        self.session_history: List[RedTeamCoordinationSession] = []

        # Attacker management (legacy mode)
        self.attackers: List[RedTeamMember] = []
        self.attacker_metrics: Dict[str, AttackerMetrics] = {}
        self.attacker_lock = threading.Lock()

        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.current_session: Optional[RedTeamCoordinationSession] = None

        # Metrics
        self.metrics = RedTeamCoordinatorMetrics()

        # Initialize ROMA-MDAP-MAKER Engine for robust orchestration/recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for standardized high-reliability config
                config_roma = get_validation_config()
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for RedTeamCoordinator")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to initialize ROMA engine: {e}")

        # Thread pool for parallel execution (legacy mode)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_attacks)

        # Load state if persistence is enabled
        if enable_persistence and os.path.exists(persistence_path):
            self._load_state()

        # Initialize based on mode
        if self.use_ensemble:
            if self.ensemble:
                logger.info(f"RedTeamCoordinator initialized with ensemble ({len(self.ensemble.models)} models)")
                self._initialize_ensemble_metrics()
            else:
                logger.warning("use_ensemble=True but no ensemble provided, falling back to legacy mode")
                self.use_ensemble = False
        else:
            # Legacy mode
            if self.red_team:
                self.attackers = self.red_team.team_members.copy()
                self._initialize_attacker_metrics()
                logger.info(f"RedTeamCoordinator initialized with {len(self.attackers)} attackers (legacy mode)")

    def _initialize_ensemble_metrics(self):
        """Initialize metrics for ensemble models"""
        if not self.ensemble:
            return

        with self.attacker_lock:
            for i, model in enumerate(self.ensemble.models):
                model_name = getattr(model, 'model', f'model_{i}')
                weight = self.ensemble.weights[i] if i < len(self.ensemble.weights) else 1.0

                if model_name not in self.attacker_metrics:
                    self.attacker_metrics[model_name] = AttackerMetrics(
                        attacker_name=model_name,
                        model_weight=weight,
                        specialization_scores={
                            category: weight * random.uniform(0.8, 1.2)
                            for category in self.attack_categories
                        }
                    )

        logger.info(f"Initialized metrics for {len(self.attacker_metrics)} ensemble models")

    def _initialize_attacker_metrics(self):
        """Initialize metrics for all attackers"""
        with self.attacker_lock:
            for attacker in self.attackers:
                if attacker.name not in self.attacker_metrics:
                    self.attacker_metrics[attacker.name] = AttackerMetrics(
                        attacker_name=attacker.name,
                        specialization_scores={
                            category: 1.0 if category in attacker.specializations else 0.3
                            for category in IssueCategory
                        }
                    )

    # =========================================================================
    # ATTACK TASK MANAGEMENT
    # =========================================================================

    def coordinate_adversarial_testing(
        self,
        content: str,
        content_type: str = "general",
        attack_categories: Optional[List[IssueCategory]] = None,
        attack_strategies: Optional[List[RedTeamStrategy]] = None,
        max_attacks_per_category: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> RedTeamCoordinationSession:
        """
        Coordinate comprehensive adversarial testing on content.

        This is the main entry point for ensemble-based adversarial testing.
        Automatically distributes attack tasks across ensemble models for
        parallel vulnerability discovery.

        Args:
            content: Content to test adversarially
            content_type: Type of content (e.g., 'code_python', 'document_general')
            attack_categories: Categories to test (None for default)
            attack_strategies: Strategies to use (None for default)
            max_attacks_per_category: Maximum attacks per category
            progress_callback: Optional callback for progress updates

        Returns:
            RedTeamCoordinationSession with aggregated findings
        """
        session_id = f"red_team_{uuid.uuid4().hex[:8]}"
        session = RedTeamCoordinationSession(
            session_id=session_id,
            target_content=content,
            content_type=content_type,
            status=AttackTaskStatus.IN_PROGRESS,
            started_at=datetime.now()
        )

        self.current_session = session
        if progress_callback:
            self.progress_callbacks.append(progress_callback)

        # Use provided categories or defaults
        categories = attack_categories or self.attack_categories
        strategies = attack_strategies or [
            RedTeamStrategy.ADVERSARIAL,
            RedTeamStrategy.SYSTEMATIC,
            RedTeamStrategy.FOCUSED_ATTACK
        ]

        # Generate attack tasks
        for category in categories:
            for strategy in strategies:
                task = AttackCoordinationTask(
                    task_id=f"attack_{uuid.uuid4().hex[:8]}",
                    target_content=content,
                    content_type=content_type,
                    attack_category=category,
                    attack_strategy=strategy,
                    priority=AttackTaskPriority.HIGH if category == IssueCategory.SECURITY_VULNERABILITY else AttackTaskPriority.MEDIUM
                )
                session.attack_tasks.append(task)
                self.attack_queue.put(task)

        session.total_tasks = len(session.attack_tasks)
        logger.info(f"Created {session.total_tasks} attack tasks for session {session_id}")

        # Execute attacks based on mode
        try:
            if self.use_ensemble and self.ensemble:
                logger.info(f"Executing ensemble-based adversarial testing with {len(self.ensemble.models)} models")
                self._execute_ensemble_attacks(session)
            else:
                logger.info(f"Executing legacy adversarial testing with {len(self.attackers)} attackers")
                self._execute_legacy_attacks(session)

            # Aggregate findings
            self._aggregate_findings(session)

            session.status = AttackTaskStatus.COMPLETED
            session.completed_at = datetime.now()
            self.session_history.append(session)

            # Update metrics
            self.metrics.total_sessions += 1
            self.metrics.total_attack_tasks += session.total_tasks
            self.metrics.completed_tasks += session.completed_tasks
            self.metrics.failed_tasks += session.failed_tasks
            self.metrics.vulnerabilities_found += len(session.aggregated_findings)

            session_duration = (session.completed_at - session.started_at).total_seconds()
            self.metrics.average_session_time = (
                (self.metrics.average_session_time * (self.metrics.total_sessions - 1) + session_duration) /
                self.metrics.total_sessions
            )

            logger.info(f"Session {session_id} completed: {len(session.aggregated_findings)} vulnerabilities found")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error in adversarial testing session {session_id}: {e}", exc_info=True)
            session.status = AttackTaskStatus.FAILED
            session.error = str(e)

        finally:
            # Save state if enabled
            if self.enable_persistence:
                self._save_state()

        return session

    def _execute_ensemble_attacks(self, session: RedTeamCoordinationSession):
        """Execute attacks using ensemble for parallel processing"""
        import asyncio

        # Build attack prompts for each task
        attack_tasks = [task for task in session.attack_tasks if task.status == AttackTaskStatus.PENDING]

        if not attack_tasks:
            logger.warning("No attack tasks to execute")
            return

        # Group tasks by category for more efficient processing
        tasks_by_category = {}
        for task in attack_tasks:
            if task.attack_category not in tasks_by_category:
                tasks_by_category[task.attack_category] = []
            tasks_by_category[task.attack_category].append(task)

        logger.info(f"Executing {len(attack_tasks)} attack tasks across {len(tasks_by_category)} categories using ensemble")

        # Create asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Process each category with ensemble
            for category, tasks in tasks_by_category.items():
                category_findings = self._execute_category_attacks_with_ensemble(
                    category, tasks, loop
                )

                # Update task results with ensemble findings
                for task, findings in zip(tasks, category_findings):
                    task.status = AttackTaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.findings = findings
                    session.completed_tasks += 1

                    # Add to aggregated findings
                    session.aggregated_findings.extend(findings)

                    # Notify progress
                    self._notify_progress(f"Completed {task.attack_category.value} attack: {len(findings)} findings")

        finally:
            loop.close()

    def _execute_category_attacks_with_ensemble(
        self,
        category: IssueCategory,
        tasks: List[AttackCoordinationTask],
        loop: asyncio.AbstractEventLoop
    ) -> List[List[IssueFinding]]:
        """
        Execute attacks for a specific category using ensemble

        Returns list of findings for each task
        """
        if not tasks:
            return []

        # Use first task as template for prompt
        template_task = tasks[0]

        # Build system message for this attack category
        system_message = self._build_attack_system_message(category, template_task.content_type)

        # Build user prompt with all attacks for this category
        user_prompt = self._build_attack_prompt(template_task, category)

        messages = [{"role": "user", "content": user_prompt}]

        try:
            if self.diversify_attacks and len(self.ensemble.models) > 1:
                # Use generate_all_with_context for diverse perspectives
                all_responses = loop.run_until_complete(
                    self.ensemble.generate_all_with_context(system_message, messages)
                )

                # Aggregate findings from all ensemble members
                all_findings = []
                for response in all_responses:
                    if response:
                        findings = self._parse_ensemble_response(response, category)
                        all_findings.extend(findings)

                # Distribute findings across tasks
                findings_per_task = max(1, len(all_findings) // len(tasks))
                task_findings_list = []

                for i, task in enumerate(tasks):
                    start_idx = i * findings_per_task
                    end_idx = start_idx + findings_per_task if i < len(tasks) - 1 else len(all_findings)
                    task_findings = all_findings[start_idx:end_idx]
                    task_findings_list.append(task_findings)

                    # Update attacker metrics
                    self._update_attacker_metrics("ensemble", len(task_findings), success=True)

                return task_findings_list

            else:
                # Use single weighted sample
                response = loop.run_until_complete(
                    self.ensemble.generate_with_context(system_message, messages)
                )

                findings = self._parse_ensemble_response(response, category) if response else []

                # Distribute same findings across all tasks
                return [findings.copy() for _ in tasks]

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error executing ensemble attacks for {category.value}: {e}", exc_info=True)
            # Return empty findings for all tasks
            return [[] for _ in tasks]

    def _build_attack_system_message(self, category: IssueCategory, content_type: str) -> str:
        """Build system message for attack generation"""
        category_guidance = {
            IssueCategory.SECURITY_VULNERABILITY: "security flaws, injection attacks, authentication bypasses, authorization issues",
            IssueCategory.LOGICAL_ERROR: "logic bugs, incorrect assumptions, edge case failures, state management issues",
            IssueCategory.PERFORMANCE_PROBLEM: "performance bottlenecks, resource leaks, inefficient algorithms, scalability issues",
            IssueCategory.COMPLIANCE_ISSUE: "regulatory violations, compliance gaps, privacy concerns, legal issues",
            IssueCategory.EDGE_CASE: "edge cases, boundary conditions, error handling gaps, input validation failures"
        }

        guidance = category_guidance.get(category, "vulnerabilities and weaknesses")

        return f"""You are an expert red team security analyst specializing in {content_type} content.
Your task is to identify {guidance} by thinking like an adversary seeking to exploit the system.

Be thorough, critical, and provide specific, actionable findings with:
- Clear vulnerability descriptions
- Severity assessments
- Potential exploit scenarios
- Suggested remediation approaches"""

    def _build_attack_prompt(self, task: AttackCoordinationTask, category: IssueCategory) -> str:
        """Build user prompt for attack task"""
        strategy_guidance = {
            RedTeamStrategy.ADVERSARIAL: "Think like a malicious adversary. How would you exploit this?",
            RedTeamStrategy.SYSTEMATIC: "Systematically check each potential vulnerability category.",
            RedTeamStrategy.FOCUSED_ATTACK: "Focus deeply on the most critical, exploitable vulnerabilities.",
            RedTeamStrategy.DEEP_DIVE: "Perform a deep dive analysis, considering subtle and complex issues.",
            RedTeamStrategy.POKA_YOKE: "Look for error-proofing failures and assumptions that could be violated."
        }

        strategy_hint = strategy_guidance.get(task.attack_strategy, "")

        return f"""Analyze the following {task.content_type} content for {category.value} vulnerabilities:

```
{task.target_content[:4000]}
```

{strategy_hint}

Provide your analysis as a JSON object with:
- 'findings': Array of vulnerability objects, each with:
  - 'title': Short descriptive title
  - 'description': Detailed description of the vulnerability
  - 'severity': One of 'critical', 'high', 'medium', 'low'
  - 'category': Vulnerability category
  - 'confidence': Confidence score (0-1)
  - 'location': Specific location if applicable
  - 'exploit_example': How this could be exploited
  - 'suggested_fix': Recommendation for remediation
- 'summary': Brief summary of key issues
- 'total_findings': Total number of findings"""

    def _parse_ensemble_response(self, response: str, category: IssueCategory) -> List[IssueFinding]:
        """Parse ensemble response into IssueFinding objects"""
        findings = []

        try:
            # Try JSON parsing first
            if response.strip().startswith('{'):
                parsed = json.loads(response)

                findings_data = parsed.get('findings', [])
                if not isinstance(findings_data, list):
                    findings_data = [findings_data]

                for finding_data in findings_data:
                    if isinstance(finding_data, dict):
                        finding = IssueFinding(
                            title=finding_data.get('title', f'{category.value} Finding'),
                            description=finding_data.get('description', ''),
                            severity=self._parse_severity(finding_data.get('severity', 'medium')),
                            category=category,
                            location=finding_data.get('location'),
                            confidence=float(finding_data.get('confidence', 0.8)),
                            suggested_fix=finding_data.get('suggested_fix'),
                            exploit_example=finding_data.get('exploit_example')
                        )
                        findings.append(finding)

            else:
                # Fallback: extract findings using heuristics
                findings = self._extract_findings_from_text(response, category)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse ensemble response: {e}, using text extraction")
            findings = self._extract_findings_from_text(response, category)

        return findings

    def _parse_severity(self, severity_str: str) -> SeverityLevel:
        """Parse severity string to SeverityLevel enum"""
        severity_map = {
            'critical': SeverityLevel.CRITICAL,
            'high': SeverityLevel.HIGH,
            'medium': SeverityLevel.MEDIUM,
            'low': SeverityLevel.LOW
        }
        return severity_map.get(severity_str.lower(), SeverityLevel.MEDIUM)

    def _extract_findings_from_text(self, text: str, category: IssueCategory) -> List[IssueFinding]:
        """Extract findings from unstructured text"""
        findings = []

        # Look for severity indicators
        lines = text.split('\n')
        current_finding = {}

        for line in lines:
            line = line.strip()

            # Detect severity
            if any(sev in line.lower() for sev in ['critical', 'high', 'medium', 'low']):
                if current_finding:
                    findings.append(self._create_finding_from_dict(current_finding, category))
                    current_finding = {}

                current_finding['severity'] = line.split(':')[0] if ':' in line else 'medium'

            # Detect description
            elif len(line) > 20 and not line.startswith('-'):
                if 'description' not in current_finding:
                    current_finding['description'] = line
                else:
                    current_finding['description'] += ' ' + line

        # Add last finding
        if current_finding and 'description' in current_finding:
            findings.append(self._create_finding_from_dict(current_finding, category))

        return findings

    def _create_finding_from_dict(self, finding_dict: Dict[str, Any], category: IssueCategory) -> IssueFinding:
        """Create IssueFinding from dictionary"""
        return IssueFinding(
            title=finding_dict.get('title', f'{category.value} Finding'),
            description=finding_dict.get('description', ''),
            severity=self._parse_severity(finding_dict.get('severity', 'medium')),
            category=category,
            location=finding_dict.get('location'),
            confidence=0.7,
            suggested_fix=finding_dict.get('suggested_fix'),
            exploit_example=finding_dict.get('exploit_example')
        )

    def _execute_legacy_attacks(self, session: RedTeamCoordinationSession):
        """Execute attacks using legacy RedTeam members (backward compatibility)"""
        if not self.red_team or not self.attackers:
            logger.warning("No attackers available in legacy mode")
            return

        # Execute attacks in parallel using thread pool
        futures = []

        for task in session.attack_tasks:
            if task.status != AttackTaskStatus.PENDING:
                continue

            # Select attacker based on load balancing strategy
            attacker = self._select_attacker_for_task(task)

            if attacker:
                task.assigned_attacker = attacker.name
                task.status = AttackTaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                # Submit to executor
                future = self.executor.submit(self._execute_single_legacy_attack, task, attacker)
                futures.append((task, future))

        # Wait for all attacks to complete
        for task, future in futures:
            try:
                assessment = future.result(timeout=self.task_timeout)
                task.result = assessment
                task.findings = assessment.findings if assessment else []
                task.status = AttackTaskStatus.COMPLETED
                task.completed_at = datetime.now()
                session.completed_tasks += 1
                session.aggregated_findings.extend(task.findings)

                self._notify_progress(f"Completed {task.attack_category.value} attack by {task.assigned_attacker}")

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Legacy attack failed for task {task.task_id}: {e}")
                task.status = AttackTaskStatus.FAILED
                task.error = str(e)
                session.failed_tasks += 1

    def _execute_single_legacy_attack(
        self,
        task: AttackCoordinationTask,
        attacker: RedTeamMember
    ) -> Optional[RedTeamAssessment]:
        """Execute a single attack using a RedTeamMember"""
        try:
            # Use the RedTeam's assess_content method
            assessment = self.red_team.assess_content(
                content=task.target_content,
                content_type=task.content_type,
                strategy=task.attack_strategy,
                num_members=1  # Single member for this task
            )

            return assessment

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error executing legacy attack: {e}", exc_info=True)
            return None

    def _select_attacker_for_task(self, task: AttackCoordinationTask) -> Optional[RedTeamMember]:
        """Select attacker for task based on load balancing strategy"""
        if not self.attackers:
            return None

        if self.load_balancing_strategy == LoadBalancingStrategy.SPECIALIZATION_BASED:
            # Select attacker with best specialization match
            best_attacker = None
            best_score = 0.0

            for attacker in self.attackers:
                metrics = self.attacker_metrics.get(attacker.name)
                if metrics:
                    score = metrics.specialization_scores.get(task.attack_category, 0.5)
                    # Factor in current load
                    load_factor = 1.0 / (metrics.current_load + 1)
                    combined_score = score * load_factor

                    if combined_score > best_score:
                        best_score = combined_score
                        best_attacker = attacker

            return best_attacker or random.choice(self.attackers)

        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select least loaded attacker
            min_load = float('inf')
            selected = None

            for attacker in self.attackers:
                metrics = self.attacker_metrics.get(attacker.name)
                if metrics and metrics.current_load < min_load:
                    min_load = metrics.current_load
                    selected = attacker

            return selected or random.choice(self.attackers)

        else:
            # Random or round-robin
            return random.choice(self.attackers)

    def _aggregate_findings(self, session: RedTeamCoordinationSession):
        """Aggregate findings from all attacks"""
        # Deduplicate findings based on similarity
        seen = set()
        unique_findings = []

        for finding in session.aggregated_findings:
            # Create a signature for deduplication
            signature = hashlib.md5(
                f"{finding.title}:{finding.category.value}:{finding.description[:100]}".encode()
            ).hexdigest()

            if signature not in seen:
                seen.add(signature)
                unique_findings.append(finding)

        session.aggregated_findings = unique_findings

        # Build severity breakdown
        session.severity_breakdown = {}
        for finding in session.aggregated_findings:
            severity = finding.severity
            session.severity_breakdown[severity] = session.severity_breakdown.get(severity, 0) + 1

        logger.info(f"Aggregated {len(session.aggregated_findings)} unique findings from {session.total_tasks} attacks")

    def _update_attacker_metrics(self, attacker_name: str, num_findings: int, success: bool):
        """Update metrics for an attacker"""
        with self.attacker_lock:
            if attacker_name not in self.attacker_metrics:
                self.attacker_metrics[attacker_name] = AttackerMetrics(
                    attacker_name=attacker_name
                )

            metrics = self.attacker_metrics[attacker_name]

            if success:
                metrics.attacks_completed += 1
                metrics.vulnerabilities_found += num_findings
            else:
                metrics.attacks_failed += 1

            metrics.last_active = datetime.now()

            # Update average task time
            if metrics.attacks_completed > 0:
                metrics.average_attack_time = metrics.total_time_spent / metrics.attacks_completed

    def _notify_progress(self, message: str):
        """Notify progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(message)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Progress callback failed: {e}")

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _save_state(self):
        """Save coordinator state to disk"""
        try:
            state = {
                'metrics': asdict(self.metrics),
                'attacker_metrics': {
                    name: asdict(metrics) for name, metrics in self.attacker_metrics.items()
                },
                'session_history': [
                    asdict(session) for session in self.session_history[-10:]  # Last 10 sessions
                ]
            }

            with open(self.persistence_path, 'w') as f:
                json.dump(state, f)

            logger.debug(f"Saved coordinator state to {self.persistence_path}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load coordinator state from disk"""
        try:
            with open(self.persistence_path, 'r') as f:
                state = json.load(f)

            # Restore metrics
            if 'metrics' in state:
                for key, value in state['metrics'].items():
                    setattr(self.metrics, key, value)

            # Restore attacker metrics
            if 'attacker_metrics' in state:
                for name, metrics_dict in state['attacker_metrics'].items():
                    self.attacker_metrics[name] = AttackerMetrics(**metrics_dict)

            logger.info(f"Loaded coordinator state from {self.persistence_path}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"Failed to load state: {e}, starting with fresh state")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_metrics(self) -> RedTeamCoordinatorMetrics:
        """Get current coordinator metrics"""
        return self.metrics

    def get_session_results(self, session_id: str) -> Optional[RedTeamCoordinationSession]:
        """Get results for a specific session"""
        for session in self.session_history:
            if session.session_id == session_id:
                return session
        return None

    def clear_history(self):
        """Clear session history"""
        self.session_history.clear()
        logger.info("Cleared session history")

    def shutdown(self):
        """Shutdown the coordinator and cleanup resources"""
        logger.info("Shutting down RedTeamCoordinator")

        # Save state
        if self.enable_persistence:
            self._save_state()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("RedTeamCoordinator shutdown complete")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_red_team_coordinator(
    api_key: str,
    model_name: str = "gpt-4o",
    num_models: int = 5,
    max_concurrent_attacks: int = 5,
    use_ensemble: bool = True,
    diversify_attacks: bool = True
) -> RedTeamCoordinator:
    """
    Convenience function to create a RedTeamCoordinator with ensemble

    Args:
        api_key: API key for LLM provider
        model_name: Base model name
        num_models: Number of models in ensemble
        max_concurrent_attacks: Max concurrent attacks
        use_ensemble: Use ensemble mode
        diversify_attacks: Use diverse temperatures

    Returns:
        Configured RedTeamCoordinator
    """
    if not ENSEMBLE_AVAILABLE:
        logger.warning("Ensemble not available, creating coordinator in legacy mode")
        return RedTeamCoordinator(use_ensemble=False)

    # Create ensemble with diverse temperatures
    models_cfg = []
    base_weight = 1.0 / num_models

    for i in range(num_models):
        # Higher temperatures for more diverse adversarial thinking
        temp_var = 0.6 + (i * 0.08)

        model_cfg = LLMModelConfig(
            name=model_name,
            api_key=api_key,
            api_base="https://api.openai.com/v1",
            temperature=min(temp_var, 1.0),
            max_tokens=2048,
            weight=base_weight
        )
        models_cfg.append(model_cfg)

    ensemble = LLMEnsemble(models_cfg)

    return RedTeamCoordinator(
        ensemble=ensemble,
        max_concurrent_attacks=max_concurrent_attacks,
        use_ensemble=use_ensemble,
        diversify_attacks=diversify_attacks
    )


if __name__ == "__main__":
    # Example usage
    print("Red Team Coordinator module loaded successfully")
