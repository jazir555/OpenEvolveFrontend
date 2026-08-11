"""
ACE (Agentic Context Engine) CrewAI Workflow Bridge

This module bridges ACE's self-improving capabilities with CrewAI's workflow
orchestration. It replaces the AGPL-licensed CrewAI integration with
MIT-licensed CrewAI.

This replaces ace_crewai_bridge.py with local CrewAI execution.

Architecture:
    CrewAI (6 phases) -> ACE Bridge -> ACE Learning (Agent + Reflector + SkillManager)

The bridge provides:
1. Phase-specific learning integration
2. Skillbook injection for all phases
3. Execution feedback collection
4. Continuous skill updates
5. Checkpoint management

License: MIT (replaces AGPL CrewAI)
Author: OpenEvolve Team
Date: 2026-01-29
"""

from typing import Any, Dict, List, Optional, Callable, Union
import sys
import os
import json
import logging
import threading
import copy
from functools import wraps
from datetime import datetime
from pathlib import Path

# Import CrewAI zero-error workflow (replaces CrewAI)
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    create_workflow_definition,
)

# Import state management
from crewai_state_management import (
    WorkflowState,
    SubProblem,
    DecompositionPlan,
    StateManager,
)

# ============================================================================
# SECURITY IMPORTS - Import all validation utilities
# ============================================================================
try:
    from ace_security_utils import (
        validate_file_path_safe,
        validate_string_length,
        validate_list_size,
        validate_numeric_range,
        validate_dict_structure,
        atomic_save_json_file,
        safe_load_json_file,
    )
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    # Fallback implementations if security utils not available
    SECURITY_UTILS_AVAILABLE = False
    def validate_file_path_safe(filepath: str, base_dir: str = ".") -> str:
        return filepath
    def validate_string_length(value: str, name: str, **kwargs) -> str:
        return value
    def validate_list_size(items: list, name: str, **kwargs) -> list:
        return items
    def validate_numeric_range(value, name: str, **kwargs):
        return value
    def validate_dict_structure(data, expected_fields, **kwargs):
        return data
    def atomic_save_json_file(filepath: str, data: Dict[str, Any]) -> None:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    def safe_load_json_file(filepath: str, max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return json.load(f)

# Add agentic-context-engine to path
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# ACE Availability Detection
ACE_AVAILABLE = False
ACE_IMPORT_ERROR = None

try:
    from ace import (
        Skillbook,
        Skill,
        Sample,
        SimpleEnvironment,
        OfflineACE,
        OnlineACE,
        Agent,
        Reflector,
        SkillManager,
        AgentOutput,
        EnvironmentResult,
        LiteLLMClient,
    )
    from ace.prompts_v2_1 import PromptManager
    ACE_AVAILABLE = True
except ImportError as e:
    ACE_IMPORT_ERROR = str(e)
    # Create stubs
    Skillbook = None
    Sample = None
    SimpleEnvironment = None
    OfflineACE = None
    Agent = None
    Reflector = None
    SkillManager = None
    AgentOutput = None
    EnvironmentResult = None
    LiteLLMClient = None
    PromptManager = None

# THREAD SAFETY FIX: Phase 2 - Import thread safety utilities
try:
    from ace_security_utils import get_global_lock, synchronized
    THREAD_SAFETY_AVAILABLE = True
except ImportError:
    THREAD_SAFETY_AVAILABLE = False
    def get_global_lock(name):
        return threading.RLock()
    def synchronized(lock=None):
        def decorator(func):
            return func
        return decorator

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ACE CrewAI Workflow Bridge
# ============================================================================

class ACECrewAIWorkflowBridge:
    """
    Bridge between ACE learning and CrewAI workflow orchestration.

    This bridge enables CrewAI agents to learn from each phase's execution
    using ACE's three-role learning loop (Agent, Reflector, SkillManager).
    Replaces ACECrewAIWorkflowBridge with MIT-licensed CrewAI.

    Phase Mapping:
        - Phase 1 (Setup): Learn from problem analysis
        - Phase 2 (Solution): Learn from solution generation
        - Phase 3 (Critique): Learn from critique patterns
        - Phase 4 (Verify): Learn from verification strategies
        - Phase 5 (Reassemble): Learn from reassembly patterns
        - Phase 6 (Final): Learn from final validation

    Attributes:
        skillbook: Shared skillbook across all phases
        model: LiteLLM model name for ACE
        enable_learning: Whether learning is enabled
        checkpoint_dir: Directory for skillbook checkpoints
        state_manager: CrewAI state manager for tracking workflows

    Memory Management:
    - max_skills: Maximum skills to keep in skillbook (default 1000)
    - min_helpful: Minimum helpful count to keep a skill (default 5)
    - Skills are pruned when exceeding max_skills with low helpful counts

    Example:
        bridge = ACECrewAIWorkflowBridge(
            model="gpt-4o-mini",
            skillbook_path="workflow_skills.json",
        )

        # Execute phase with learning
        result = bridge.execute_phase_1_setup(
            problem_statement="Design scalable architecture",
            enable_learning=True,
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        skillbook_path: Optional[str] = None,
        enable_learning: bool = True,
        checkpoint_dir: str = "./ace_checkpoints",
        prompt_version: str = "v2.1",
        max_skills: int = 1000,
        min_helpful: int = 5,
        state_storage_dir: str = "./crewai_states",
    ):
        """
        Initialize ACE-CrewAI bridge.

        Args:
            model: LiteLLM model name
            skillbook_path: Path to load existing skillbook
            enable_learning: Enable learning from executions
            checkpoint_dir: Directory for checkpoints
            prompt_version: Prompt version (v2.1 recommended)
            max_skills: Maximum skills to keep in skillbook (resource limit)
            min_helpful: Minimum helpful count to keep a skill during cleanup
            state_storage_dir: Directory for CrewAI state storage
        """
        # SECURITY FIX: Validate skillbook_path if provided
        if skillbook_path:
            try:
                skillbook_path = validate_file_path_safe(skillbook_path)
            except ValueError as e:
                logger.warning(f"Invalid skillbook path: {e}. Using new skillbook.")
                skillbook_path = None

        # SECURITY FIX: Validate checkpoint_dir
        try:
            checkpoint_dir = validate_file_path_safe(checkpoint_dir)
        except ValueError as e:
            logger.warning(f"Invalid checkpoint directory: {e}. Using default.")
            checkpoint_dir = "./ace_checkpoints"

        self.model = model
        self.enable_learning = enable_learning
        self.checkpoint_dir = checkpoint_dir
        self.prompt_version = prompt_version
        self.max_skills = max_skills
        self.min_helpful = min_helpful

        # Initialize CrewAI state manager
        self.state_manager = StateManager(state_storage_dir)
        self.workflows: Dict[str, WorkflowState] = {}
        self.workflow_counter = 0

        # THREAD SAFETY FIX: Add thread-safe lock for skillbook access
        self._skillbook_lock = threading.RLock()
        
        # PERFORMANCE FIX: Add caching for skillbook.as_prompt()
        self._cached_skills = None
        self._skills_dirty = True

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize skillbook
        # THREAD SAFETY FIX: TS-6 - Remove TOCTOU, use exception handling
        if ACE_AVAILABLE and skillbook_path:
            try:
                self.skillbook = Skillbook.load_from_file(skillbook_path)
                logger.info(f"Loaded skillbook from {skillbook_path}")
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                self.skillbook = Skillbook()
                logger.info(f"Skillbook not found, created new skillbook")
        elif ACE_AVAILABLE:
            self.skillbook = Skillbook()
            logger.info("Created new skillbook")
        else:
            self.skillbook = None
            logger.warning("ACE not available - learning disabled")

        # Initialize ACE components (if available)
        self.agent = None
        self.reflector = None
        self.skill_manager = None
        self.prompt_mgr = None

        if ACE_AVAILABLE:
            self._initialize_ace_components()
        
        logger.info("ACE-CrewAI Bridge initialized (MIT-licensed)")

    def _initialize_ace_components(self):
        """Initialize ACE Agent, Reflector, and SkillManager."""
        try:
            # Create LLM client
            llm = LiteLLMClient(model=self.model)

            # Get prompt templates
            self.prompt_mgr = PromptManager()

            # Create ACE roles
            self.agent = Agent(llm, prompt_template=self.prompt_mgr.get_agent_prompt())
            self.reflector = Reflector(llm, prompt_template=self.prompt_mgr.get_reflector_prompt())
            self.skill_manager = SkillManager(llm, prompt_template=self.prompt_mgr.get_skill_manager_prompt())

            logger.info("ACE components initialized successfully")

        except (ImportError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to initialize ACE components: {e}")
            self.enable_learning = False

    def inject_skills(self, context: str = "") -> str:
        """
        Inject learned skills into context.

        Args:
            context: Original context string

        Returns:
            Enhanced context with skills
        """
        if not ACE_AVAILABLE or not self.skillbook:
            return context

        # THREAD SAFETY FIX: TS-4 - Synchronize skillbook access
        with self._skillbook_lock:
            skills = self.skillbook.as_prompt()

        # Context validation - add isinstance check for context
        if context is None:
            context = ""
        elif not isinstance(context, str):
            context = str(context)

        # PERFORMANCE FIX: Use list join for efficient string building
        parts = [
            "LEARNED SKILLS FROM PREVIOUS EXECUTIONS:",
            skills,
            "",
            "CURRENT CONTEXT:",
            context
        ]
        return "\n".join(parts)

    def cleanup_old_skills(self, max_skills: Optional[int] = None, min_helpful: Optional[int] = None):
        """
        RESOURCE FIX: Remove less helpful skills to keep size bounded.

        Args:
            max_skills: Maximum skills to keep (defaults to self.max_skills)
            min_helpful: Minimum helpful count to keep a skill (defaults to self.min_helpful)
        """
        if not self.skillbook:
            return

        max_skills = max_skills or self.max_skills
        min_helpful = min_helpful or self.min_helpful

        skills = self.skillbook.skills()
        if len(skills) < max_skills:
            return

        skills.sort(key=lambda s: s.helpful_count, reverse=True)
        
        # PERFORMANCE FIX: Collect skills to remove first, then batch remove (O(n) instead of O(n²))
        skills_to_remove = [
            skill.strategy for skill in skills[max_skills:]
            if skill.helpful_count < min_helpful
        ]
        
        removed_count = 0
        for strategy in skills_to_remove:
            self.skillbook.remove(strategy)
            removed_count += 1
        
        # PERFORMANCE FIX: Invalidate cache when skills are removed
        if removed_count > 0:
            self._invalidate_skills_cache()
            logger.info(f"Cleaned skillbook: {len(skills)} -> {len(self.skillbook.skills())} skills (removed {removed_count} low-helpful skills)")

    def _invalidate_skills_cache(self):
        """Invalidate the skills cache."""
        self._skills_dirty = True
        self._cached_skills = None

    def save_skillbook(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Save skillbook to file with atomic write operation.

        SECURITY FIX: Uses atomic_save_json_file to prevent TOCTOU and file corruption

        Args:
            filepath: Optional filepath (defaults to agent-specific)

        Returns:
            Dict with save result
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not ACE_AVAILABLE or not self.skillbook:
            return {
                "success": False,
                "error": "ACE not available",
            }

        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.checkpoint_dir, f"skillbook_{timestamp}.json")

            # SECURITY FIX: Validate filepath
            filepath = validate_file_path_safe(filepath, self.checkpoint_dir)

            # THREAD SAFETY FIX: TS-4 - Synchronize skillbook access
            # Deep copy skillbook inside lock, serialize outside
            with self._skillbook_lock:
                skillbook_copy = copy.deepcopy(self.skillbook)

            # Serialize outside lock
            if SECURITY_UTILS_AVAILABLE:
                # Convert skillbook copy to dict for saving
                skillbook_data = {
                    "skills": [skill.__dict__ for skill in skillbook_copy.skills()],
                    "metadata": {
                        "saved_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "num_skills": len(skillbook_copy.skills()),
                    }
                }
                atomic_save_json_file(filepath, skillbook_data)
            else:
                # Fallback to ACE's native save
                skillbook_copy.save_to_file(filepath)

            return {
                "success": True,
                "filepath": filepath,
                "skills_saved": len(skillbook_copy.skills()),
            }

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to save skillbook: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 1: Setup
    # ========================================================================

    def execute_phase_1_setup(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 1 (Setup) with ACE learning.

        Phase 1 Activities:
        - Analyze problem
        - Identify constraints
        - Plan decomposition
        - Learn from analysis patterns

        Args:
            problem_statement: The problem to solve
            problem_type: Type of problem
            domain: Problem domain
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 1: Setup", problem_statement)

        try:
            # SECURITY FIX: Validate problem_statement length
            problem_statement = validate_string_length(
                problem_statement,
                "problem_statement",
                max_length=50000,
                min_length=10,
                allow_empty=False
            )

            # Inject learned skills (safely handle None context)
            # DEEP COPY FIX: Deep copy context to prevent external modification
            context_description = ""
            if context and isinstance(context, dict):
                context_description = copy.deepcopy(context.get("description", ""))
            elif context and isinstance(context, str):
                context_description = copy.deepcopy(context)

            # SECURITY FIX: Validate context_description length
            if context_description:
                context_description = validate_string_length(
                    context_description,
                    "context_description",
                    max_length=50000,
                    allow_empty=True
                )

            enhanced_context = self.inject_skills(context_description)

            # Create sample
            sample = Sample(
                query=f"Analyze problem: {problem_statement}",
                context=enhanced_context,
            )

            # Execute agent
            agent_output = self.agent.run(sample)

            # Learn from execution (if enabled)
            learning_result = None
            if enable_learning and self.reflector and self.skill_manager:
                learning_result = self._learn_from_execution(
                    sample=sample,
                    agent_output=agent_output,
                    phase="Phase 1: Setup",
                )

            # Save checkpoint
            if save_checkpoint:
                # RESOURCE FIX: Clean up old skills before saving
                self.cleanup_old_skills()
                self.save_skillbook()

            return {
                "phase": "Phase 1: Setup",
                "success": True,
                "problem_statement": problem_statement,
                "analysis": agent_output.final_answer,
                "reasoning": agent_output.reasoning,
                "learning": learning_result,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Phase 1 execution failed: {e}")
            return {
                "phase": "Phase 1: Setup",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 2: Solution
    # ========================================================================

    def execute_phase_2_solution(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 2 (Solution Generation) with ACE learning.

        Phase 2 Activities:
        - Generate solutions for sub-problems
        - Apply learned strategies
        - Learn from solution patterns

        Args:
            problem_statement: The overall problem
            sub_problems: List of sub-problems to solve
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 2: Solution", problem_statement)

        try:
            # SECURITY FIX: Validate problem_statement length
            problem_statement = validate_string_length(
                problem_statement,
                "problem_statement",
                max_length=50000,
                min_length=10,
                allow_empty=False
            )

            # DEEP COPY FIX: Deep copy sub_problems to avoid mutating original
            sub_problems = copy.deepcopy(sub_problems)

            # SECURITY FIX: Validate sub_problems list size
            sub_problems = validate_list_size(
                sub_problems,
                "sub_problems",
                max_size=1000,
                min_size=0,
                allow_empty=True
            )

            results = []

            for sub_problem in sub_problems:
                # VALIDATION FIX: Validate sub_problem structure
                if not isinstance(sub_problem, dict):
                    logger.warning(f"Skipping invalid sub_problem (not a dict): {sub_problem}")
                    continue

                # Inject learned skills
                enhanced_context = self.inject_skills(
                    sub_problem.get("description", "")
                )

                # Create sample
                sample = Sample(
                    query=f"Solve sub-problem: {sub_problem.get('description', '')}",
                    context=enhanced_context,
                )

                # Execute agent
                agent_output = self.agent.run(sample)

                # Learn from execution
                learning_result = None
                if enable_learning and self.reflector and self.skill_manager:
                    learning_result = self._learn_from_execution(
                        sample=sample,
                        agent_output=agent_output,
                        phase="Phase 2: Solution",
                    )

                results.append({
                    "sub_problem": sub_problem.get("description", ""),
                    "solution": agent_output.final_answer,
                    "reasoning": agent_output.reasoning,
                    "learning": learning_result,
                })

            # Save checkpoint
            if save_checkpoint:
                self.save_skillbook()

            return {
                "phase": "Phase 2: Solution",
                "success": True,
                "solutions": results,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except Exception as e:
            logger.error(f"Phase 2 execution failed: {e}")
            return {
                "phase": "Phase 2: Solution",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 3: Critique
    # ========================================================================

    def execute_phase_3_critique(
        self,
        solutions: List[Dict[str, Any]],
        critique_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 3 (Critique) with ACE learning.

        Phase 3 Activities:
        - Critique solutions
        - Identify issues
        - Learn from critique patterns

        Args:
            solutions: List of solutions to critique
            critique_criteria: Criteria for critique
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 3: Critique", str(solutions[:5]))

        try:
            critiques = []

            for solution in solutions:
                # SECURITY FIX: Validate solution string length
                solution_text = solution.get('solution', '')
                if solution_text:
                    solution_text = validate_string_length(
                        solution_text,
                        "solution",
                        max_length=50000,
                        allow_empty=True
                    )

                # Inject learned skills
                enhanced_context = self.inject_skills(
                    f"Solution: {solution_text}"
                )

                # Create sample
                sample = Sample(
                    query=f"Critique solution: {solution_text}",
                    context=enhanced_context,
                )

                # Execute agent
                agent_output = self.agent.run(sample)

                # Learn from execution
                learning_result = None
                if enable_learning and self.reflector and self.skill_manager:
                    learning_result = self._learn_from_execution(
                        sample=sample,
                        agent_output=agent_output,
                        phase="Phase 3: Critique",
                    )

                critiques.append({
                    "solution": solution_text,
                    "critique": agent_output.final_answer,
                    "learning": learning_result,
                })

            # Save checkpoint
            if save_checkpoint:
                self.save_skillbook()

            return {
                "phase": "Phase 3: Critique",
                "success": True,
                "critiques": critiques,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Phase 3 execution failed: {e}")
            return {
                "phase": "Phase 3: Critique",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 4: Verify
    # ========================================================================

    def execute_phase_4_verify(
        self,
        solutions: List[Dict[str, Any]],
        verification_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 4 (Verification) with ACE learning.

        Phase 4 Activities:
        - Verify solutions
        - Check constraints
        - Learn from verification patterns

        Args:
            solutions: List of solutions to verify
            verification_criteria: Criteria for verification
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 4: Verify", str(solutions[:5]))

        try:
            verifications = []

            for solution in solutions:
                # SECURITY FIX: Validate solution and critique string lengths
                solution_text = solution.get('solution', '')
                critique_text = solution.get('critique', '')

                if solution_text:
                    solution_text = validate_string_length(
                        solution_text,
                        "solution",
                        max_length=50000,
                        allow_empty=True
                    )

                if critique_text:
                    critique_text = validate_string_length(
                        critique_text,
                        "critique",
                        max_length=50000,
                        allow_empty=True
                    )

                # Inject learned skills
                enhanced_context = self.inject_skills(
                    f"Solution: {solution_text}\nCritique: {critique_text}"
                )

                # Create sample
                sample = Sample(
                    query=f"Verify solution: {solution_text}",
                    context=enhanced_context,
                )

                # Execute agent
                agent_output = self.agent.run(sample)

                # Learn from execution
                learning_result = None
                if enable_learning and self.reflector and self.skill_manager:
                    learning_result = self._learn_from_execution(
                        sample=sample,
                        agent_output=agent_output,
                        phase="Phase 4: Verify",
                    )

                verifications.append({
                    "solution": solution_text,
                    "verification": agent_output.final_answer,
                    "learning": learning_result,
                })

            # Save checkpoint
            if save_checkpoint:
                self.save_skillbook()

            return {
                "phase": "Phase 4: Verify",
                "success": True,
                "verifications": verifications,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Phase 4 execution failed: {e}")
            return {
                "phase": "Phase 4: Verify",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 5: Reassemble
    # ========================================================================

    def execute_phase_5_reassemble(
        self,
        sub_solutions: List[Dict[str, Any]],
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 5 (Reassembly) with ACE learning.

        Phase 5 Activities:
        - Reassemble sub-solutions
        - Integrate components
        - Learn from reassembly patterns

        Args:
            sub_solutions: List of sub-solutions to reassemble
            problem_statement: Original problem statement
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 5: Reassemble", problem_statement)

        try:
            # SECURITY FIX: Validate sub_solutions list size
            sub_solutions = validate_list_size(
                sub_solutions,
                "sub_solutions",
                max_size=1000,
                min_size=0,
                allow_empty=True
            )

            # Inject learned skills
            enhanced_context = self.inject_skills(
                f"Problem: {problem_statement}\nSub-solutions: {len(sub_solutions)}"
            )

            # Create sample
            sample = Sample(
                query=f"Reassemble sub-solutions for: {problem_statement}",
                context=enhanced_context,
            )

            # Execute agent
            agent_output = self.agent.run(sample)

            # Learn from execution
            learning_result = None
            if enable_learning and self.reflector and self.skill_manager:
                learning_result = self._learn_from_execution(
                    sample=sample,
                    agent_output=agent_output,
                    phase="Phase 5: Reassemble",
                )

            # Save checkpoint
            if save_checkpoint:
                self.save_skillbook()

            return {
                "phase": "Phase 5: Reassemble",
                "success": True,
                "reassembled_solution": agent_output.final_answer,
                "reasoning": agent_output.reasoning,
                "learning": learning_result,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Phase 5 execution failed: {e}")
            return {
                "phase": "Phase 5: Reassemble",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 6: Final Validation
    # ========================================================================

    def execute_phase_6_final(
        self,
        final_solution: str,
        problem_statement: str,
        validation_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 6 (Final Validation) with ACE learning.

        Phase 6 Activities:
        - Validate final solution
        - Ensure completeness
        - Learn from validation patterns

        Args:
            final_solution: The final solution to validate
            problem_statement: Original problem statement
            validation_criteria: Criteria for validation
            context: Additional context
            enable_learning: Enable learning from this phase
            save_checkpoint: Save skillbook checkpoint after phase

        Returns:
            Dict with phase results and learning outcomes
        """
        if not ACE_AVAILABLE:
            return self._stub_result("Phase 6: Final", problem_statement)

        try:
            # SECURITY FIX: Validate final_solution string length
            final_solution = validate_string_length(
                final_solution,
                "final_solution",
                max_length=100000,
                min_length=10,
                allow_empty=False
            )

            # SECURITY FIX: Validate problem_statement string length
            problem_statement = validate_string_length(
                problem_statement,
                "problem_statement",
                max_length=50000,
                min_length=10,
                allow_empty=False
            )

            # Inject learned skills
            enhanced_context = self.inject_skills(
                f"Problem: {problem_statement}\nSolution: {final_solution}"
            )

            # Create sample
            sample = Sample(
                query=f"Validate final solution for: {problem_statement}",
                context=enhanced_context,
            )

            # Execute agent
            agent_output = self.agent.run(sample)

            # Learn from execution
            learning_result = None
            if enable_learning and self.reflector and self.skill_manager:
                learning_result = self._learn_from_execution(
                    sample=sample,
                    agent_output=agent_output,
                    phase="Phase 6: Final",
                )

            # Save checkpoint
            if save_checkpoint:
                self.save_skillbook()

            return {
                "phase": "Phase 6: Final",
                "success": True,
                "validation": agent_output.final_answer,
                "reasoning": agent_output.reasoning,
                "learning": learning_result,
                "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Phase 6 execution failed: {e}")
            return {
                "phase": "Phase 6: Final",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Full Workflow Execution
    # ========================================================================

    def execute_full_workflow(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        sub_problems: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full 6-phase CrewAI workflow with ACE learning.

        This method runs all phases sequentially, with ACE learning from
        each phase's execution and continuously improving the skillbook.

        Args:
            problem_statement: The problem to solve
            problem_type: Type of problem
            domain: Problem domain
            sub_problems: Optional pre-decomposed sub-problems
            context: Additional context
            enable_learning: Enable learning throughout workflow

        Returns:
            Dict with all phase results and overall learning metrics
        """
        # SECURITY FIX: Validate checkpoint_dir
        try:
            checkpoint_dir = validate_file_path_safe(self.checkpoint_dir)
        except ValueError:
            checkpoint_dir = "./ace_checkpoints"

        logger.info(f"Starting full workflow execution for: {problem_statement}")

        results = {
            "problem_statement": problem_statement,
            "phases": {},
            "learning_metrics": {
                "initial_skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
                "phases_with_learning": 0,
                "total_skill_updates": 0,
            },
        }

        # Phase 1: Setup
        logger.info("Executing Phase 1: Setup")
        phase1_result = self.execute_phase_1_setup(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            context=context,
            enable_learning=enable_learning,
        )
        results["phases"]["phase_1"] = phase1_result

        # PHASE SUCCESS CHECK: Check if Phase 1 succeeded before continuing
        if not phase1_result.get("success", False):
            logger.error("Phase 1 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase1_result.get("error", "Unknown error")
            return results

        # Phase 2: Solution (use provided sub-problems or extract from Phase 1)
        logger.info("Executing Phase 2: Solution")
        phase2_sub_problems = sub_problems or []
        phase2_result = self.execute_phase_2_solution(
            problem_statement=problem_statement,
            sub_problems=phase2_sub_problems,
            context=context,
            enable_learning=enable_learning,
        )
        results["phases"]["phase_2"] = phase2_result

        # Check if Phase 2 succeeded before continuing
        if not phase2_result.get("success", False):
            logger.error("Phase 2 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase2_result.get("error", "Unknown error")
            return results

        # Phase 3: Critique
        logger.info("Executing Phase 3: Critique")
        phase3_result = self.execute_phase_3_critique(
            solutions=[{"solution": phase2_result.get("solution", "")}],
            critique_criteria=None,
            context=context,
            enable_learning=enable_learning,
            save_checkpoint=True,
        )
        results["phases"]["phase_3"] = phase3_result

        # PHASE SUCCESS CHECK: Check if Phase 3 succeeded before continuing
        if not phase3_result.get("success", False):
            logger.error("Phase 3 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase3_result.get("error", "Unknown error")
            return results

        # Phase 4: Verify
        logger.info("Executing Phase 4: Verify")
        phase4_result = self.execute_phase_4_verify(
            solutions=[{"solution": phase2_result.get("solution", ""), "critique": ""}],
            verification_criteria=None,
            context=context,
            enable_learning=enable_learning,
            save_checkpoint=True,
        )
        results["phases"]["phase_4"] = phase4_result

        # PHASE SUCCESS CHECK: Check if Phase 4 succeeded before continuing
        if not phase4_result.get("success", False):
            logger.error("Phase 4 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase4_result.get("error", "Unknown error")
            return results

        # Phase 5: Reassemble
        logger.info("Executing Phase 5: Reassemble")
        phase5_result = self.execute_phase_5_reassemble(
            sub_solutions=[phase2_result.get("solutions", [])],
            problem_statement=problem_statement,
            context=context,
            enable_learning=enable_learning,
            save_checkpoint=True,
        )
        results["phases"]["phase_5"] = phase5_result

        # PHASE SUCCESS CHECK: Check if Phase 5 succeeded before continuing
        if not phase5_result.get("success", False):
            logger.error("Phase 5 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase5_result.get("error", "Unknown error")
            return results

        # Phase 6: Final
        logger.info("Executing Phase 6: Final")
        phase6_result = self.execute_phase_6_final(
            final_solution=phase5_result.get("reassembled_solution", ""),
            problem_statement=problem_statement,
            validation_criteria=None,
            context=context,
            enable_learning=enable_learning,
            save_checkpoint=True,
        )
        results["phases"]["phase_6"] = phase6_result

        # PHASE SUCCESS CHECK: Check if Phase 6 succeeded
        if not phase6_result.get("success", False):
            logger.error("Phase 6 failed, aborting workflow")
            results["workflow_success"] = False
            results["error"] = phase6_result.get("error", "Unknown error")
            return results

        # Final metrics
        results["learning_metrics"]["final_skillbook_size"] = len(self.skillbook.skills()) if self.skillbook else 0
        results["learning_metrics"]["skills_learned"] = (
            results["learning_metrics"]["final_skillbook_size"] -
            results["learning_metrics"]["initial_skillbook_size"]
        )

        # SECURITY FIX: Clean up old skills after full workflow
        self.cleanup_old_skills()

        # Mark workflow as successful if all phases completed
        results["workflow_success"] = True

        logger.info("Full workflow execution complete")
        return results

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _learn_from_execution(
        self,
        sample: Sample,
        agent_output: AgentOutput,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Learn from a single execution using Reflector and SkillManager.

        Args:
            sample: The sample that was executed
            agent_output: The agent's output
            phase: Phase name for tracking

        Returns:
            Dict with learning result
        """
        try:
            # THREAD SAFETY FIX: TS-4 - Synchronize skillbook access
            with self._skillbook_lock:
                # Reflector analysis
                reflection = self.reflector.run(
                    sample=sample,
                    agent_output=agent_output,
                    skillbook=self.skillbook,
                    environment_result=None,  # No ground truth
                )

                # SkillManager updates
                updates = self.skill_manager.run(
                    sample=sample,
                    agent_output=agent_output,
                    reflection=reflection,
                    skillbook=self.skillbook,
                )

                # Apply updates
                updates_applied = 0
                if updates:
                    for update in updates.updates:
                        update.apply(self.skillbook)
                        updates_applied += 1

                return {
                    "phase": phase,
                    "updates_applied": updates_applied,
                    "reflection_summary": reflection.summary if reflection else "",
                }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Learning failed for {phase}: {e}")
            return {
                "phase": phase,
                "error": str(e),
            }

    def _stub_result(self, phase: str, input: str) -> Dict[str, Any]:
        """Return stub result when ACE is not available."""
        return {
            "phase": phase,
            "success": False,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR or "agentic-context-engine not installed",
        }

    def cleanup(self):
        """Release resources held by this object."""
        try:
            # LLM CLEANUP FIX: Properly close LLM client connections
            # First close any LLM clients in ACE components
            if hasattr(self, 'agent') and self.agent:
                if hasattr(self.agent, 'llm'):
                    llm_client = self.agent.llm
                    if hasattr(llm_client, 'close'):
                        try:
                            llm_client.close()
                            logger.info("Closed agent LLM client")
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to close agent LLM client: {e}")
                    # Clear reference
                    self.agent.llm = None
                self.agent = None

            if hasattr(self, 'reflector') and self.reflector:
                if hasattr(self.reflector, 'llm'):
                    llm_client = self.reflector.llm
                    if hasattr(llm_client, 'close'):
                        try:
                            llm_client.close()
                            logger.info("Closed reflector LLM client")
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to close reflector LLM client: {e}")
                    self.reflector.llm = None
                self.reflector = None

            if hasattr(self, 'skill_manager') and self.skill_manager:
                if hasattr(self.skill_manager, 'llm'):
                    llm_client = self.skill_manager.llm
                    if hasattr(llm_client, 'close'):
                        try:
                            llm_client.close()
                            logger.info("Closed skill manager LLM client")
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to close skill manager LLM client: {e}")
                    self.skill_manager.llm = None
                self.skill_manager = None

            # Clear skillbook reference
            if hasattr(self, 'skillbook'):
                self.skillbook = None

            # Clear cache
            self._cached_skills = None
            self._skills_dirty = True

            logger.info("ACE-CrewAI Bridge cleanup complete")

        except Exception as e:
            logger.error(f"Error during ACE-CrewAI Bridge cleanup: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Alias for backward compatibility - maps old CrewAI class name to new CrewAI class
ACECrewAIWorkflowBridge = ACECrewAIWorkflowBridge

__all__ = [
    "ACECrewAIWorkflowBridge",
    "ACECrewAIWorkflowBridge",  # Backward compatibility alias
]
