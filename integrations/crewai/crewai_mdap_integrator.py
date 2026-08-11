"""
CrewAI MDAP Integrator - Port of MDAP to CrewAI Agents and Crews

This module ports the Multi-Agent Debate Protocol (MDAP) to CrewAI,
enabling collaborative problem solving through structured agent interactions,
voting, and consensus mechanisms using CrewAI's native orchestration.

Key Features:
1. Multi-agent debate protocol with CrewAI agents
2. First-to-K voting mechanism
3. Red-flagging for unreliable outputs
4. Step-by-step validation with caching
5. Integration with CrewAI flows

Architecture:
- MDAP debate participants -> CrewAI Agents
- MDAP steps -> CrewAI Tasks
- MDAP task execution -> CrewAI Crew
- Voting coordination -> CrewAI Process

License: MIT
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    Process = None

# Import state management
from crewai_state_management import (
    WorkflowState,
    WorkflowStatus,
    SolutionAttempt,
)


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RedFlagRules:
    """Configuration rules for red-flagging undesirable outputs"""
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    blocked_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.2
    require_schema_match: bool = True


@dataclass
class MDAPStep:
    """Individual step in MDAP workflow"""
    step_id: str
    prompt: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPTask:
    """Task definition for MDAP execution"""
    task_id: str
    description: str
    steps: List[MDAPStep]
    max_retries: int = 2
    target_success_rate: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPConfig:
    """Configuration for MDAP runs"""
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 50
    timeout_seconds: int = 60
    red_flag_rules: Optional[RedFlagRules] = None
    fallback_policy: str = "escalate_then_best_effort"
    cache_ttl_seconds: Optional[int] = None
    cache_max_size: int = 5000

    def __post_init__(self):
        if self.red_flag_rules is None:
            self.red_flag_rules = RedFlagRules()


@dataclass
class MDAPVoteResult:
    """Voting results from agent interactions"""
    winner: Optional[Any]
    votes: Dict[str, int]
    red_flags: int
    confidence: float
    attempts: int
    duration_seconds: float
    flagged_reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MDAPStepResult:
    """Results from individual MDAP steps"""
    step_id: str
    vote_result: MDAPVoteResult
    status: str
    retries: int


@dataclass
class MDAPRunResult:
    """Complete results from MDAP execution"""
    task_id: str
    step_results: Dict[str, MDAPStepResult]
    metrics: Dict[str, Any]


# =============================================================================
# AGENT ROLE DEFINITIONS
# =============================================================================

class MDAPAgentRole(str, Enum):
    """Roles for MDAP debate agents"""
    DEBATER = "debater"  # Participates in debates
    MODERATOR = "moderator"  # Moderates discussions
    VALIDATOR = "validator"  # Validates outputs
    SYNTHESIZER = "synthesizer"  # Synthesizes consensus
    CRITIC = "critic"  # Provides critique


# =============================================================================
# CREWAI AGENT FACTORY
# =============================================================================

class MDAPAgentFactory:
    """
    Factory for creating CrewAI agents for MDAP operations.
    """

    @staticmethod
    def create_debater_agent(
        name: str = "MDAP_Debater",
        expertise: str = "general",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for participating in debates."""
        if not CREWAI_AVAILABLE:
            return None

        backstory = f"""You are an expert in {expertise} with strong analytical and reasoning skills.
        You participate in structured debates to:
        1. Present well-reasoned arguments
        2. Evaluate alternative perspectives
        3. Identify flaws in reasoning
        4. Work toward consensus through evidence-based discussion"""

        return Agent(
            role=MDAPAgentRole.DEBATER.value,
            goal=f"Contribute valuable insights to debates on {expertise} topics",
            backstory=backstory,
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_moderator_agent(
        name: str = "MDAP_Moderator",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for moderating debates."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MDAPAgentRole.MODERATOR.value,
            goal="Facilitate productive debate and guide agents toward consensus",
            backstory="""You are an expert moderator specializing in managing multi-agent discussions.
            You ensure:
            1. All perspectives are heard
            2. Discussion stays focused
            3. Arguments are evaluated fairly
            4. Progress is made toward resolution""",
            verbose=True,
            allow_delegation=True,
            **(llm_config or {})
        )

    @staticmethod
    def create_validator_agent(
        name: str = "MDAP_Validator",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for validating outputs."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MDAPAgentRole.VALIDATOR.value,
            goal="Validate outputs for correctness, completeness, and quality",
            backstory="""You are an expert at evaluating the quality of outputs.
            You check for:
            1. Correctness and accuracy
            2. Completeness - all requirements met
            3. Clarity and coherence
            4. Compliance with constraints""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_synthesizer_agent(
        name: str = "MDAP_Synthesizer",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for synthesizing consensus."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MDAPAgentRole.SYNTHESIZER.value,
            goal="Synthesize diverse perspectives into a coherent consensus solution",
            backstory="""You are an expert at finding common ground and synthesizing solutions.
            You excel at:
            1. Identifying shared understanding
            2. Reconciling different viewpoints
            3. Creating integrated solutions
            4. Building on the best ideas from all participants""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_critic_agent(
        name: str = "MDAP_Critic",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for providing critique."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MDAPAgentRole.CRITIC.value,
            goal="Provide constructive critique to improve solution quality",
            backstory="""You are an expert critical thinker with deep analytical skills.
            You provide:
            1. Identification of potential issues
            2. Suggestions for improvement
            3. Edge case analysis
            4. Quality enhancement recommendations""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )


# =============================================================================
# RED FLAGGING
# =============================================================================

class CrewAIRedFlagger:
    """
    Red-flagging for unreliable outputs in CrewAI workflows.
    """

    def __init__(self, rules: RedFlagRules):
        self.rules = rules
        self.stats = {
            "total_validations": 0,
            "validation_failures": 0,
            "rejected_votes": 0,
        }

    def is_flagged(
        self,
        raw_text: str,
        candidate: Any,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if content is red-flagged.

        Args:
            raw_text: Raw response text
            candidate: Parsed candidate
            schema: Optional schema for validation

        Returns:
            Tuple of (is_flagged, reasons)
        """
        reasons: List[str] = []
        self.stats["total_validations"] += 1

        # Basic checks
        if raw_text is None or raw_text.strip() == "":
            reasons.append("empty_response")
            return True, reasons

        # Length checks
        if self.rules.max_characters and len(raw_text) > self.rules.max_characters:
            reasons.append("response_too_long")

        if self.rules.max_tokens:
            approx_tokens = len(raw_text) / 4
            if approx_tokens > self.rules.max_tokens:
                reasons.append("token_limit_exceeded")

        # Pattern blocking
        import re
        for pattern in self.rules.blocked_patterns:
            if re.search(pattern, raw_text, re.IGNORECASE):
                reasons.append(f"blocked_pattern:{pattern}")

        # Schema validation
        if schema is not None and self.rules.require_schema_match:
            is_valid, errors = self._validate_schema(candidate, schema)
            if not is_valid:
                reasons.extend(errors)

        # Confidence check
        confidence = self._extract_confidence(candidate)
        if confidence < self.rules.min_confidence:
            reasons.append("low_confidence")

        is_flagged = len(reasons) > 0
        if is_flagged:
            self.stats["validation_failures"] += 1
            self.stats["rejected_votes"] += 1

        return is_flagged, reasons

    def _validate_schema(self, candidate: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate candidate against schema."""
        errors: List[str] = []
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(candidate, dict):
                return False, ["candidate is not an object"]
            required = schema.get("required", [])
            for key in required:
                if key not in candidate:
                    errors.append(f"missing required key: {key}")

        elif schema_type == "array":
            if not isinstance(candidate, list):
                errors.append("candidate is not an array")

        elif schema_type and not self._matches_type(candidate, schema_type):
            errors.append(f"candidate expected type {schema_type}")

        return len(errors) == 0, errors

    def _matches_type(self, value: Any, schema_type: str) -> bool:
        """Check if value matches schema type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        expected_type = type_mapping.get(schema_type)
        return isinstance(value, expected_type) if expected_type else True

    def _extract_confidence(self, candidate: Any) -> float:
        """Extract confidence score from candidate."""
        if isinstance(candidate, dict):
            value = candidate.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return 0.5  # Default confidence

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.stats.copy()


# =============================================================================
# CACHING
# =============================================================================

class CrewAIMDAPCache:
    """
    Caching mechanism for CrewAI MDAP computations.
    """

    def __init__(self, max_size: int = 5000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        now = time.time()
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check TTL
        if now - entry["timestamp"] > self.ttl_seconds:
            self._cache.pop(key, None)
            self._access.pop(key, None)
            return None

        # Update access time
        self._access[key] = now
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        self._cache[key] = {"value": value, "timestamp": time.time()}
        self._access[key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access:
            return

        lru_key = min(self._access, key=self._access.get)
        self._cache.pop(lru_key, None)
        self._access.pop(lru_key, None)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access.clear()


# =============================================================================
# MDAP ENGINE - CrewAI Integration
# =============================================================================

class CrewAIMDAPIntegrator:
    """
    Main MDAP integrator for CrewAI.

    Coordinates multi-agent debates using CrewAI crews and processes.
    """

    def __init__(
        self,
        config: MDAPConfig,
        workflow_id: Optional[str] = None,
        use_cav_nlp: bool = True,
    ):
        """
        Initialize MDAP integrator.

        Args:
            config: MDAP configuration
            workflow_id: Optional workflow identifier
            use_cav_nlp: Enable CAV-NLP integration
        """
        self.config = config
        self.workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:12]}"

        # Initialize components
        self.red_flagger = CrewAIRedFlagger(config.red_flag_rules)
        self.cache = CrewAIMDAPCache(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds or 3600
        )

        # CAV-NLP integration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for CrewAIMDAPIntegrator")

        logger.info(f"CrewAIMDAPIntegrator initialized for workflow {self.workflow_id}")

    def integrate_with_cav_nlp(self, crewai_result, mdap_result):
        """Integrate CrewAI and MDAP results using CAV-NLP."""
        if self.use_cav_nlp:
            # Formalize both results
            crewai_formal = self.math_service.formalize(str(crewai_result))
            mdap_formal = self.math_service.formalize(str(mdap_result))
            # Verify consistency
            return self.verify_consistency(crewai_formal.code, mdap_formal.code)
        return {"verified": False, "reason": "CAV-NLP not available"}

    def verify_consistency(self, code1: str, code2: str) -> Dict[str, Any]:
        """Verify consistency between two formalized codes."""
        if not self.use_cav_nlp:
            return {"consistent": False, "reason": "CAV-NLP not available"}
        try:
            result = self.enhanced_solver.verify_with_lean(f"{code1} == {code2}")
            return {
                "consistent": result.get("verified", False),
                "confidence": result.get("confidence", 0.0),
                "method": "lean_verification"
            }
        except Exception as e:
            logger.warning(f"CAV-NLP consistency check failed: {e}")
            return {"consistent": False, "error": str(e)}

    def execute_step(
        self,
        step: MDAPStep,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
    ) -> MDAPStepResult:
        """
        Execute a single MDAP step with voting.

        Args:
            step: MDAP step to execute
            agents: List of CrewAI agents for debate
            context: Additional context

        Returns:
            MDAPStepResult with voting outcome
        """
        logger.info(f"Executing MDAP step {step.step_id}")

        start_time = time.time()
        retries = 0
        max_retries = step.metadata.get("max_retries", self.config.max_retries)

        while retries <= max_retries:
            try:
                # Perform voting
                vote_result = self._do_voting(
                    step=step,
                    agents=agents,
                    context=context,
                )

                duration = time.time() - start_time
                vote_result.duration_seconds = duration

                # Check if successful
                if vote_result.winner is not None:
                    step_result = MDAPStepResult(
                        step_id=step.step_id,
                        vote_result=vote_result,
                        status="success",
                        retries=retries,
                    )
                    logger.info(f"MDAP step {step.step_id} completed in {duration:.2f}s")
                    return step_result
                else:
                    retries += 1
                    logger.warning(f"MDAP step {step.step_id} failed, retry {retries}/{max_retries}")

            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                retries += 1
                logger.error(f"Error executing MDAP step {step.step_id}: {e}")
                if retries > max_retries:
                    break

        # All retries failed
        vote_result = MDAPVoteResult(
            winner=None,
            votes={},
            red_flags=0,
            confidence=0.0,
            attempts=retries,
            duration_seconds=time.time() - start_time,
            errors=["All retries exceeded"],
        )

        return MDAPStepResult(
            step_id=step.step_id,
            vote_result=vote_result,
            status="failed",
            retries=retries,
        )

    def _do_voting(
        self,
        step: MDAPStep,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
    ) -> MDAPVoteResult:
        """
        Perform first-to-K voting.

        Args:
            step: MDAP step
            agents: CrewAI agents for voting
            context: Additional context

        Returns:
            MDAPVoteResult with winner and metadata
        """
        votes: Dict[str, int] = {}
        red_flags = 0
        attempts = 0
        flagged_reasons: List[str] = []
        errors: List[str] = []

        # K threshold (use average of k_min and k_max)
        k_threshold = (self.config.k_min + self.config.k_max) // 2

        for attempt in range(self.config.max_votes_per_step):
            attempts += 1

            # Select agent (round-robin)
            agent = agents[attempt % len(agents)]

            try:
                # Create CrewAI task
                task = Task(
                    description=step.prompt,
                    expected_output="A well-reasoned response that addresses the prompt",
                    agent=agent,
                )

                # Create crew for single task
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )

                # Execute
                raw_text = crew.kickoff()

                # Parse candidate
                candidate = self._parse_candidate(raw_text)

                # Check red flags
                is_flagged, reasons = self.red_flagger.is_flagged(
                    raw_text=raw_text,
                    candidate=candidate,
                    schema=step.expected_schema,
                )

                if is_flagged:
                    red_flags += 1
                    flagged_reasons.extend(reasons)
                    continue

                # Canonicalize for counting
                candidate_key = self._canonicalize_candidate(candidate)
                votes[candidate_key] = votes.get(candidate_key, 0) + 1

                # Check for winner (first-to-K)
                if votes[candidate_key] >= k_threshold:
                    winner = self._decode_candidate(candidate_key)
                    total_votes = sum(votes.values())
                    confidence = votes[candidate_key] / total_votes if total_votes > 0 else 0.0

                    return MDAPVoteResult(
                        winner=winner,
                        votes=votes,
                        red_flags=red_flags,
                        confidence=confidence,
                        attempts=attempts,
                        duration_seconds=0.0,  # Will be set by caller
                        flagged_reasons=flagged_reasons,
                        errors=errors,
                    )

            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.warning(f"Voting attempt {attempt + 1} failed: {e}")
                errors.append(str(e))
                continue

        # No winner reached threshold
        if votes:
            winner_key = max(votes, key=votes.get)
            winner = self._decode_candidate(winner_key)
            total_votes = sum(votes.values())
            confidence = votes[winner_key] / total_votes if total_votes > 0 else 0.0

            return MDAPVoteResult(
                winner=winner,
                votes=votes,
                red_flags=red_flags,
                confidence=confidence,
                attempts=attempts,
                duration_seconds=0.0,
                flagged_reasons=flagged_reasons,
                errors=errors,
            )

        return MDAPVoteResult(
            winner=None,
            votes=votes,
            red_flags=red_flags,
            confidence=0.0,
            attempts=attempts,
            duration_seconds=0.0,
            flagged_reasons=flagged_reasons,
            errors=errors,
        )

    def _parse_candidate(self, raw_text: str) -> Any:
        """Parse raw text into candidate."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    def _canonicalize_candidate(self, candidate: Any) -> str:
        """Convert candidate to canonical string for voting."""
        if isinstance(candidate, (dict, list)):
            return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return str(candidate).strip()

    def _decode_candidate(self, candidate_key: str) -> Any:
        """Decode candidate key back to candidate."""
        try:
            return json.loads(candidate_key)
        except (json.JSONDecodeError, TypeError):
            return candidate_key

    def execute_task(
        self,
        task: MDAPTask,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
    ) -> MDAPRunResult:
        """
        Execute complete MDAP task with all steps.

        Args:
            task: MDAP task to execute
            agents: CrewAI agents for execution
            context: Additional context

        Returns:
            MDAPRunResult with all step results
        """
        logger.info(f"Executing MDAP task {task.task_id} with {len(task.steps)} steps")

        step_results: Dict[str, MDAPStepResult] = {}
        start_time = time.time()

        for step in task.steps:
            step_result = self.execute_step(
                step=step,
                agents=agents,
                context=context,
            )
            step_results[step.step_id] = step_result

            # Check if step failed
            if step_result.status == "failed":
                logger.error(f"MDAP step {step.step_id} failed, stopping execution")
                break

        duration = time.time() - start_time

        # Calculate metrics
        successful_steps = sum(1 for r in step_results.values() if r.status == "success")
        total_votes = sum(r.vote_result.attempts for r in step_results.values())
        total_red_flags = sum(r.vote_result.red_flags for r in step_results.values())

        metrics = {
            "total_steps": len(task.steps),
            "successful_steps": successful_steps,
            "failed_steps": len(task.steps) - successful_steps,
            "total_votes": total_votes,
            "total_red_flags": total_red_flags,
            "duration_seconds": duration,
            "avg_confidence": sum(r.vote_result.confidence for r in step_results.values()) / len(step_results) if step_results else 0.0,
        }

        return MDAPRunResult(
            task_id=task.task_id,
            step_results=step_results,
            metrics=metrics,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mdap_config(
    k_min: int = 2,
    k_max: int = 8,
    max_votes_per_step: int = 50,
    timeout_seconds: int = 60,
    enable_red_flagging: bool = True,
) -> MDAPConfig:
    """
    Factory function to create MDAP configuration.

    Args:
        k_min: Minimum K threshold for voting
        k_max: Maximum K threshold for voting
        max_votes_per_step: Maximum voting rounds per step
        timeout_seconds: Timeout per step
        enable_red_flagging: Enable red-flagging

    Returns:
        MDAPConfig instance
    """
    red_flag_rules = RedFlagRules() if enable_red_flagging else None

    return MDAPConfig(
        k_min=k_min,
        k_max=k_max,
        max_votes_per_step=max_votes_per_step,
        timeout_seconds=timeout_seconds,
        red_flag_rules=red_flag_rules,
    )


def create_mdap_integrator(
    config: Optional[MDAPConfig] = None,
    workflow_id: Optional[str] = None,
) -> CrewAIMDAPIntegrator:
    """
    Factory function to create MDAP integrator.

    Args:
        config: MDAP configuration (uses defaults if None)
        workflow_id: Optional workflow identifier

    Returns:
        CrewAIMDAPIntegrator instance
    """
    if config is None:
        config = create_mdap_config()

    return CrewAIMDAPIntegrator(
        config=config,
        workflow_id=workflow_id,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("CrewAI MDAP Integrator Example")
    print("=" * 50)

    if not CREWAI_AVAILABLE:
        print("CrewAI not available. Install with: pip install crewai")
    else:
        # Create agents
        factory = MDAPAgentFactory()

        debate_agents = [
            factory.create_debater_agent(f"Debater_{i}", expertise="problem solving")
            for i in range(3)
        ]

        # Create config
        config = create_mdap_config(k_min=2, k_max=5)

        # Create integrator
        integrator = create_mdap_integrator(config=config)

        # Create step
        step = MDAPStep(
            step_id="step_1",
            prompt="What is the best approach to solve a complex problem?",
            task_type="analysis",
        )

        print(f"Created MDAP integrator with {len(debate_agents)} agents")
        print(f"K threshold range: {config.k_min}-{config.k_max}")
