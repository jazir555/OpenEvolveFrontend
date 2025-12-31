"""
ACE (Agentic Context Engine) MCP Tools for Hephaestus Integration

This module provides Model Context Protocol (MCP) tools that enable Hephaestus
agents to leverage ACE's self-improving capabilities. ACE enables agents to learn
from their execution feedback through three specialized roles:
- Agent: Executes tasks using learned skills
- Reflector: Analyzes what worked and what didn't
- SkillManager: Updates the skillbook with new skills

Architecture: Hephaestus (Orchestrator) -> ACE (Learning Layer) -> LLM Providers
"""

from typing import Any, Dict, List, Optional, Union
import sys
import os
import json
import logging
import threading
from functools import wraps
from datetime import datetime

# SECURITY FIX: Import security utilities
from ace_security_utils import (
    validate_and_resolve_path,
    validate_file_path_safe,
    safe_load_json_file,
    atomic_save_json_file,
    validate_numeric_range,
    validate_list_size,
    validate_string_length,
    validate_model_name,
    create_safe_error,
    sanitize_for_logging,
    get_global_lock,
    DEFAULT_SKILLBOOK_DIR,
)

# THREAD SAFETY FIX: Import thread safety utilities (already imported above)
try:
    from ace_security_utils import synchronized
    THREAD_SAFETY_AVAILABLE = True
except ImportError:
    THREAD_SAFETY_AVAILABLE = False
    def synchronized(lock=None):
        def decorator(func):
            return func
        return decorator

# Add agentic-context-engine to path
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# THREAD SAFETY FIX: TS-1 - MCP Tools Registry Race
# ============================================================================
# MCP Tool Registry with thread-safe access
_MCP_TOOLS = {}
_MCP_TOOLS_LOCK = get_global_lock('mcp_tools_registry')

def mcp_tool(name: str):
    """Decorator to register MCP tools (thread-safe)."""
    def decorator(func):
        # BUG FIX #1: Use @wraps(func) instead of @wraps(name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # THREAD SAFETY FIX: TS-1 - Synchronize registry access
            with _MCP_TOOLS_LOCK:
                _MCP_TOOLS[name] = func
            return func(*args, **kwargs)
        return wrapper
    return decorator

def clear_mcp_tools():
    """
    RESOURCE FIX: Clear all registered MCP tools.

    This should be called when you want to free up memory
    by clearing the global MCP tool registry.

    Returns:
        int: Number of tools that were cleared
    """
    global _MCP_TOOLS
    with _MCP_TOOLS_LOCK:
        count = len(_MCP_TOOLS)
        _MCP_TOOLS.clear()
        logger.info(f"Cleared {count} MCP tools from global registry")
        return count

# ACE Availability Detection
ACE_AVAILABLE = False
ACE_IMPORT_ERROR = None

try:
    from ace import (
        Skillbook,
        Skill,
        UpdateOperation,
        UpdateBatch,
        Sample,
        SimpleEnvironment,
        OfflineACE,
        OnlineACE,
        Agent,
        Reflector,
        SkillManager,
        LiteLLMClient,
        AgentOutput,  # Added to avoid late import
    )
    from ace.prompts_v2_1 import PromptManager
    ACE_AVAILABLE = True
except ImportError as e:
    ACE_IMPORT_ERROR = str(e)
    # Create stubs for graceful degradation
    Skillbook = None
    Skill = None
    Sample = None
    SimpleEnvironment = None
    OfflineACE = None
    OnlineACE = None
    Agent = None
    Reflector = None
    SkillManager = None
    LiteLLMClient = None
    PromptManager = None
    AgentOutput = None  # Add stub

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP Tool 1: Initialize ACE Agent
# ============================================================================

@mcp_tool("initialize_ace_agent")
def initialize_ace_agent(
    agent_id: str,
    model: str = "gpt-4o-mini",
    skillbook_path: Optional[str] = None,
    prompt_version: str = "v2.1",
    enable_deduplication: bool = True,
    dedup_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Initialize an ACE learning agent with a skillbook.

    Args:
        agent_id: Unique identifier for the agent
        model: LiteLLM model name (supports 100+ providers)
        skillbook_path: Optional path to load existing skillbook
        prompt_version: Prompt version to use (v2.1 recommended)
        enable_deduplication: Enable skill deduplication
        dedup_threshold: Similarity threshold for deduplication (0-1)

    Returns:
        Dict with:
            - success: bool
            - agent_id: str
            - skillbook_size: int (number of skills)
            - model: str
            - message: str
            - available: bool
    """
    # VALIDATION FIX: Validate all inputs
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # VALIDATION FIX: Validate model name to prevent command injection
    try:
        model = validate_model_name(model)
    except ValueError as e:
        return create_safe_error("Invalid model name", e)

    # VALIDATION FIX: Validate dedup_threshold with NaN/Infinity check
    try:
        if enable_deduplication:
            dedup_threshold = validate_numeric_range(
                dedup_threshold, "dedup_threshold",
                min_val=0.0, max_val=1.0,
                allow_nan=False, allow_infinity=False
            )
    except ValueError as e:
        return create_safe_error("Invalid dedup_threshold", e)

    # VALIDATION FIX: Validate prompt_version string
    try:
        prompt_version = validate_string_length(prompt_version, "prompt_version", max_length=20, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid prompt_version", e)

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR or "agentic-context-engine not installed or not accessible",
            "components": {
                "skillbook": False,
                "agent": False,
                "reflector": False,
                "skill_manager": False,
            },
        }

    try:
        # BUG FIX #5: Use try-except instead of check-then-act (TOCTOU fix)
        # SECURITY FIX: CVE-1 Path Traversal - Validate skillbook_path
        if skillbook_path:
            try:
                skillbook_path = validate_file_path_safe(skillbook_path, base_dir=".")
                skillbook = Skillbook.load_from_file(skillbook_path)
                logger.info(f"Loaded skillbook from {sanitize_for_logging(skillbook_path)}")
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load skillbook: {e}")
                skillbook = Skillbook()
            except ValueError as e:
                return create_safe_error("Invalid skillbook path", e)
        else:
            skillbook = Skillbook()
            logger.info("Created new skillbook")

        # Get prompt templates
        prompt_mgr = PromptManager()

        # Create LLM client
        llm = LiteLLMClient(model=model)

        # Create ACE roles
        agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())
        reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
        skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

        return {
            "success": True,
            "agent_id": agent_id,
            "available": True,
            "model": model,
            "skillbook_size": len(skillbook.skills()),
            "prompt_version": prompt_version,
            "enable_deduplication": enable_deduplication,
            "dedup_threshold": dedup_threshold,
            "message": f"ACE agent '{agent_id}' initialized successfully",
            "components": {
                "skillbook": True,
                "agent": True,
                "reflector": True,
                "skill_manager": True,
            },
        }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to initialize ACE agent: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to initialize ACE agent", e)


# ============================================================================
# MCP Tool 2: Execute Task with ACE
# ============================================================================

@mcp_tool("execute_task_with_ace")
def execute_task_with_ace(
    agent_id: str,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    inject_skills: bool = True,
    skillbook_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a task using ACE with learned skills.

    Note: This function creates a new skillbook on each call for statelessness.
    For production use with persistence, provide skillbook_path to load existing skills.

    Args:
        agent_id: Unique identifier for the agent
        task: Task description or query
        context: Additional context for the task
        model: LiteLLM model name
        inject_skills: Whether to inject learned skills into context
        skillbook_path: Optional path to load existing skillbook (recommended for production)

    Returns:
        Dict with:
            - success: bool
            - agent_output: str (the answer/solution)
            - skills_used: int (number of skills in skillbook)
            - execution_time: float
            - message: str
    """
    # VALIDATION FIX: Validate all inputs first
    # Validate agent_id
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # Validate model name to prevent command injection
    try:
        model = validate_model_name(model)
    except ValueError as e:
        return create_safe_error("Invalid model name", e)

    # Validate task length
    try:
        task = validate_string_length(task, "task", max_length=10000, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Task description too long", e)

    # Validate context description if provided
    if context:
        try:
            context_str = str(context)
            context_str = validate_string_length(context_str, "context", max_length=50000)
        except ValueError as e:
            return create_safe_error("Context too large", e)
    else:
        context_str = ""

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR,
        }

    try:
        start_time = datetime.now()

        # BUG FIX #5: Use try-except instead of check-then-act (TOCTOU fix)
        # SECURITY FIX: CVE-1 Path Traversal - Validate skillbook_path
        if skillbook_path:
            try:
                skillbook_path = validate_file_path_safe(skillbook_path, base_dir=".")
                skillbook = Skillbook.load_from_file(skillbook_path)
                logger.info(f"Loaded skillbook from {sanitize_for_logging(skillbook_path)}")
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load skillbook: {e}")
                skillbook = Skillbook()
            except ValueError as e:
                return create_safe_error("Invalid skillbook path", e)
        else:
            skillbook = Skillbook()
            logger.info("Created new skillbook")

        # Create LLM client and agent
        llm = LiteLLMClient(model=model)
        prompt_mgr = PromptManager()
        agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())

        # Prepare input with skills
        if inject_skills and skillbook.skills():
            skills_context = skillbook.as_prompt()
        else:
            skills_context = ""

        # DEEP COPY FIX: Deep copy context to prevent external modification
        # Create sample with deep copied context
        sample = Sample(
            query=task,
            context=copy.deepcopy(skills_context) if skills_context else "",
        )

        # Execute agent
        agent_output = agent.run(sample)

        execution_time = (datetime.now() - start_time).total_seconds()

        # BUG FIX #2: Add None check before accessing agent_output attributes
        if agent_output is None:
            return create_safe_error("Agent execution returned None", ValueError("Agent output is None"))

        return {
            "success": True,
            "agent_id": agent_id,
            "available": True,
            "agent_output": agent_output.final_answer if agent_output else None,
            "reasoning": agent_output.reasoning if agent_output else None,
            "skills_used": len(skillbook.skills()),
            "execution_time": execution_time,
            "message": f"Task executed successfully",
        }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to execute task with ACE: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to execute task with ACE", e)


# ============================================================================
# MCP Tool 3: Learn from Samples
# ============================================================================

@mcp_tool("learn_from_samples_with_ace")
def learn_from_samples_with_ace(
    agent_id: str,
    samples: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    epochs: int = 1,
    checkpoint_interval: Optional[int] = None,
    checkpoint_dir: str = "./ace_checkpoints",
    async_learning: bool = False,
    max_reflector_workers: int = 3,
) -> Dict[str, Any]:
    """
    Learn from a batch of samples using ACE.

    Args:
        agent_id: Unique identifier for the agent
        samples: List of samples with 'query' and 'ground_truth' keys
        model: LiteLLM model name
        epochs: Number of epochs to train
        checkpoint_interval: Save checkpoint every N samples
        checkpoint_dir: Directory for checkpoints
        async_learning: Enable async learning mode
        max_reflector_workers: Parallel reflector workers for async mode

    Returns:
        Dict with:
            - success: bool
            - samples_processed: int
            - skills_learned: int
            - skillbook_size: int
            - training_metrics: Dict
            - message: str
    """
    # VALIDATION FIX: Validate all inputs first
    # Validate agent_id
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # Validate model name
    try:
        model = validate_model_name(model)
    except ValueError as e:
        return create_safe_error("Invalid model name", e)

    # Validate samples list size
    try:
        samples = validate_list_size(samples, "samples", max_size=10000, min_size=1, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid samples list", e)

    # Validate epochs range
    try:
        epochs = validate_numeric_range(epochs, "epochs", min_val=1, max_val=100)
    except ValueError as e:
        return create_safe_error("Invalid epochs value", e)

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR,
        }

    try:
        # Create skillbook
        skillbook = Skillbook()

        # Create LLM client
        llm = LiteLLMClient(model=model)
        prompt_mgr = PromptManager()

        # Create ACE roles
        agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())
        reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
        skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

        # Convert samples to ACE Sample objects
        # BUG FIX #3: Add validation for required keys and type checking
        # DEEP COPY FIX: Deep copy sample data to prevent external modification
        ace_samples = []
        for s in samples:
            if not isinstance(s, dict):
                logger.warning(f"Skipping non-dict sample: {type(s)}")
                continue
            if "query" not in s:
                logger.warning("Skipping sample without 'query' key")
                continue

            # Deep copy all sample fields to prevent external modification
            ace_samples.append(Sample(
                query=copy.deepcopy(s["query"]),
                ground_truth=copy.deepcopy(s.get("ground_truth")) if s.get("ground_truth") else None,
                context=copy.deepcopy(s.get("context", "")),
            ))

        # Create OfflineACE adapter
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=agent,
            reflector=reflector,
            skill_manager=skill_manager,
            async_learning=async_learning,
            max_reflector_workers=max_reflector_workers,
        )

        # Create environment
        environment = SimpleEnvironment()

        # Run learning
        results = adapter.run(
            ace_samples,
            environment,
            epochs=epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )

        # Calculate metrics
        skills_before = 0
        skills_after = len(skillbook.skills())

        training_metrics = {
            "epochs": epochs,
            "samples_processed": len(ace_samples),
            "skills_before": skills_before,
            "skills_after": skills_after,
            "new_skills": skills_after - skills_before,
            "async_learning": async_learning,
        }

        return {
            "success": True,
            "agent_id": agent_id,
            "available": True,
            "samples_processed": len(ace_samples),
            "skills_learned": skills_after - skills_before,
            "skillbook_size": skills_after,
            "training_metrics": training_metrics,
            "message": f"Learned {skills_after - skills_before} new skills from {len(ace_samples)} samples",
        }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to learn from samples: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to learn from samples", e)


# ============================================================================
# MCP Tool 4: Learn from Single Execution
# ============================================================================

@mcp_tool("learn_from_execution_with_ace")
def learn_from_execution_with_ace(
    agent_id: str,
    query: str,
    agent_output: str,
    ground_truth: Optional[str] = None,
    feedback: Optional[str] = None,
    reasoning: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Learn from a single execution (online learning).

    Args:
        agent_id: Unique identifier for the agent
        query: The original query/task
        agent_output: The agent's output
        ground_truth: Optional ground truth for evaluation
        feedback: Optional feedback string
        reasoning: Optional reasoning trace
        model: LiteLLM model name

    Returns:
        Dict with:
            - success: bool
            - updates_applied: int
            - skillbook_size: int
            - message: str
    """
    # VALIDATION FIX: Validate all inputs first
    # Validate agent_id
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # Validate model name
    try:
        model = validate_model_name(model)
    except ValueError as e:
        return create_safe_error("Invalid model name", e)

    # Validate query
    try:
        query = validate_string_length(query, "query", max_length=10000, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid query", e)

    # Validate agent_output
    try:
        agent_output = validate_string_length(agent_output, "agent_output", max_length=10000, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_output", e)

    # Validate optional fields
    if ground_truth:
        try:
            ground_truth = validate_string_length(ground_truth, "ground_truth", max_length=10000)
        except ValueError as e:
            return create_safe_error("Invalid ground_truth", e)

    if feedback:
        try:
            feedback = validate_string_length(feedback, "feedback", max_length=5000)
        except ValueError as e:
            return create_safe_error("Invalid feedback", e)

    if reasoning:
        try:
            reasoning = validate_string_length(reasoning, "reasoning", max_length=10000)
        except ValueError as e:
            return create_safe_error("Invalid reasoning", e)

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR,
        }

    try:
        # Create skillbook
        skillbook = Skillbook()

        # Create LLM client
        llm = LiteLLMClient(model=model)
        prompt_mgr = PromptManager()

        # Create Reflector and SkillManager
        reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
        skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

        # Create sample with ground truth
        # DEEP COPY FIX: Deep copy query and ground_truth to prevent external modification
        sample = Sample(
            query=copy.deepcopy(query),
            ground_truth=copy.deepcopy(ground_truth) if ground_truth else None,
            context="",
        )

        # Create agent output (AgentOutput imported at top of file)
        agent_out = AgentOutput(
            final_answer=agent_output,
            reasoning=reasoning or "",
        )

        # Evaluate
        if ground_truth:
            environment = SimpleEnvironment()
            env_result = environment.evaluate(sample, agent_out)
        else:
            env_result = None

        # Reflector analysis
        reflection = reflector.run(
            sample=sample,
            agent_output=agent_out,
            skillbook=skillbook,
            environment_result=env_result,
        )

        # SkillManager updates
        updates = skill_manager.run(
            sample=sample,
            agent_output=agent_out,
            reflection=reflection,
            skillbook=skillbook,
        )

        # Apply updates
        # BUG FIX #4: Wrap skillbook updates in lock for thread safety
        updates_applied = 0
        if updates:
            skillbook_lock = get_global_lock('skillbook_updates')
            with skillbook_lock:
                for update in updates.updates:
                    update.apply(skillbook)
                    updates_applied += 1

        return {
            "success": True,
            "agent_id": agent_id,
            "available": True,
            "updates_applied": updates_applied,
            "skillbook_size": len(skillbook.skills()),
            "reflection_summary": reflection.summary if reflection else "",
            "message": f"Applied {updates_applied} skill updates from execution",
        }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to learn from execution: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to learn from execution", e)


# ============================================================================
# MCP Tool 5: Save/Load Skillbook
# ============================================================================

@mcp_tool("manage_ace_skillbook")
def manage_ace_skillbook(
    agent_id: str,
    action: str = "list",  # BUG FIX #6: Added safe default action
    filepath: Optional[str] = None,
    format: str = "json",  # "json" or "markdown"
) -> Dict[str, Any]:
    """
    Manage ACE skillbook (save, load, list, clear).

    Args:
        agent_id: Unique identifier for the agent
        action: Action to perform ("save", "load", "list", "clear")
        filepath: File path for save/load operations
        format: Format for save/list ("json" or "markdown")

    Returns:
        Dict with action-specific result data
    """
    # VALIDATION FIX: Validate agent_id
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # VALIDATION FIX: Validate action
    valid_actions = ["save", "load", "list", "clear"]
    if action not in valid_actions:
        return create_safe_error(
            f"Invalid action: {action}",
            ValueError(f"Action must be one of {valid_actions}")
        )

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR,
        }

    try:
        skillbook = Skillbook()

        if action == "save":
            # SECURITY FIX: CVE-1 Path Traversal - Validate filepath
            if not filepath:
                filepath = f"skillbook_{agent_id}.json"
            try:
                filepath = validate_file_path_safe(filepath, base_dir=DEFAULT_SKILLBOOK_DIR)
            except ValueError as e:
                return create_safe_error("Invalid filepath for save", e)

            skillbook.save_to_file(filepath)
            logger.info(f"Saved skillbook to {sanitize_for_logging(filepath)}")
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "save",
                "filepath": filepath,
                "skills_saved": len(skillbook.skills()),
                "message": f"Saved {len(skillbook.skills())} skills to {filepath}",
            }

        elif action == "load":
            # THREAD SAFETY FIX: TS-6 - Remove TOCTOU, use exception handling
            # SECURITY FIX: CVE-1 Path Traversal - Validate filepath
            if not filepath:
                return create_safe_error(
                    "No filepath provided",
                    ValueError("Filepath is required for load action")
                )
            try:
                filepath = validate_file_path_safe(filepath, base_dir=DEFAULT_SKILLBOOK_DIR)
            except ValueError as e:
                return create_safe_error("Invalid filepath for load", e)

            try:
                skillbook = Skillbook.load_from_file(filepath)
                logger.info(f"Loaded skillbook from {sanitize_for_logging(filepath)}")
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                return create_safe_error(f"Failed to load skillbook", e)
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "load",
                "filepath": filepath,
                "skills_loaded": len(skillbook.skills()),
                "message": f"Loaded {len(skillbook.skills())} skills from {filepath}",
            }

        elif action == "list":
            skills = skillbook.skills()
            if format == "markdown":
                skills_list = str(skillbook)
            else:
                skills_list = [
                    {
                        "strategy": s.strategy,
                        "helpful_count": s.helpful_count,
                        "harmful_count": s.harmful_count,
                    }
                    for s in skills
                ]
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "list",
                "skill_count": len(skills),
                "skills": skills_list,
                "message": f"Listed {len(skills)} skills",
            }

        elif action == "clear":
            # BUG FIX #7: Load existing skillbook if filepath provided before clearing
            # SECURITY FIX: Properly iterate and remove skills
            if filepath:
                try:
                    filepath = validate_file_path_safe(filepath, base_dir=DEFAULT_SKILLBOOK_DIR)
                    skillbook = Skillbook.load_from_file(filepath)
                    logger.info(f"Loaded skillbook from {sanitize_for_logging(filepath)} before clearing")
                except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not load skillbook for clearing: {sanitize_for_logging(e)}")
                    skillbook = Skillbook()
                except ValueError as e:
                    return create_safe_error("Invalid filepath for clear", e)

            # Clear all skills from the skillbook
            skills_to_remove = list(skillbook.skills())
            skills_cleared = 0
            for skill in skills_to_remove:
                try:
                    skillbook.remove(skill.strategy)
                    skills_cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to remove skill {skill.strategy}: {sanitize_for_logging(e)}")
            logger.info(f"Cleared {skills_cleared} skills from skillbook")
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "clear",
                "skills_cleared": skills_cleared,
                "message": f"Cleared {skills_cleared} skills from skillbook",
            }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to manage skillbook: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to manage skillbook", e)


# ============================================================================
# MCP Tool 6: Get ACE Status
# ============================================================================

@mcp_tool("get_ace_status")
def get_ace_status() -> Dict[str, Any]:
    """
    Get ACE installation and component status.

    Returns:
        Dict with:
            - available: bool
            - version: str
            - components: Dict[str, bool]
            - integrations: Dict[str, bool]
            - message: str
    """
    if not ACE_AVAILABLE:
        return {
            "available": False,
            "installed": False,
            "version": None,
            "error": ACE_IMPORT_ERROR or "ACE not installed or not accessible",
            "components": {
                "skillbook": False,
                "agent": False,
                "reflector": False,
                "skill_manager": False,
                "offline_ace": False,
                "online_ace": False,
            },
            "integrations": {
                "litellm": False,
                "langchain": False,
                "browser_use": False,
                "claude_code": False,
            },
        }

    # SECURITY FIX: Safe ace.features import with try-except
    # Check integrations (with graceful degradation)
    has_litellm_val = False
    has_langchain_val = False
    has_browser_use_val = False

    try:
        # Import features module with error handling
        from ace.features import (
            has_litellm,
            has_langchain,
            has_browser_use,
        )
        has_litellm_val = has_litellm()
        has_langchain_val = has_langchain()
        has_browser_use_val = has_browser_use()
    except ImportError:
        # features module not available
        logger.debug("ace.features module not available")
    except Exception as e:
        logger.warning(f"Error loading ace.features: {sanitize_for_logging(e)}")

    return {
        "available": True,
        "installed": True,
        "version": "0.5.0",  # ACE framework version
        "message": "ACE is available and ready",
        "components": {
            "skillbook": True,
            "agent": True,
            "reflector": True,
            "skill_manager": True,
            "offline_ace": True,
            "online_ace": True,
            "async_learning": True,
            "deduplication": True,
        },
        "integrations": {
            "litellm": has_litellm_val,
            "langchain": has_langchain_val,
            "browser_use": has_browser_use_val,
            "claude_code": False,  # Not in features module
        },
        "features": {
            "checkpoint_saving": True,
            "async_mode": True,
            "observability": has_litellm_val,  # Opik integration
            "toon_format": True,  # Token-Oriented Object Notation
        },
    }


# ============================================================================
# MCP Tool 7: Inject Skills into Context
# ============================================================================

@mcp_tool("inject_ace_skills_into_context")
def inject_ace_skills_into_context(
    agent_id: str,
    context: str,
    skillbook_path: Optional[str] = None,
    max_skills: int = 50,
    format: str = "toon",  # "toon" (compact) or "markdown" (readable)
) -> Dict[str, Any]:
    """
    Inject learned skills into context for agent execution.

    Args:
        agent_id: Unique identifier for the agent
        context: Original context string
        skillbook_path: Optional path to skillbook file
        max_skills: Maximum number of skills to inject
        format: Format for skills ("toon" or "markdown")

    Returns:
        Dict with:
            - success: bool
            - enhanced_context: str (context + skills)
            - skills_injected: int
            - message: str
    """
    # VALIDATION FIX: Validate all inputs first
    # Validate agent_id
    try:
        agent_id = validate_string_length(agent_id, "agent_id", max_length=100, allow_empty=False)
    except ValueError as e:
        return create_safe_error("Invalid agent_id", e)

    # Validate max_skills range
    try:
        max_skills = validate_numeric_range(max_skills, "max_skills", min_val=1, max_val=1000)
    except ValueError as e:
        return create_safe_error("Invalid max_skills value", e)

    # Validate context
    try:
        context = validate_string_length(context, "context", max_length=50000, allow_empty=True)
    except ValueError as e:
        return create_safe_error("Invalid context", e)

    # Validate format
    valid_formats = ["toon", "markdown"]
    if format not in valid_formats:
        return create_safe_error(
            f"Invalid format: {format}",
            ValueError(f"Format must be one of {valid_formats}")
        )

    if not ACE_AVAILABLE:
        return {
            "success": False,
            "agent_id": agent_id,
            "available": False,
            "error": "ACE not available",
            "message": ACE_IMPORT_ERROR,
        }

    try:
        # BUG FIX #5: Use try-except instead of check-then-act (TOCTOU fix)
        # SECURITY FIX: CVE-1 Path Traversal - Validate skillbook_path
        if skillbook_path:
            try:
                skillbook_path = validate_file_path_safe(skillbook_path, base_dir=DEFAULT_SKILLBOOK_DIR)
                skillbook = Skillbook.load_from_file(skillbook_path)
                logger.info(f"Loaded skillbook from {sanitize_for_logging(skillbook_path)}")
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load skillbook from {sanitize_for_logging(skillbook_path)}: {sanitize_for_logging(e)}")
                skillbook = Skillbook()
            except ValueError as e:
                return create_safe_error("Invalid skillbook path", e)
        else:
            skillbook = Skillbook()

        # Get skills
        skills = skillbook.skills()[:max_skills]

        if format == "toon":
            skills_str = skillbook.as_prompt()
        else:
            skills_str = str(skillbook)

        # Inject skills into context
        enhanced_context = f"""LEARNED SKILLS:
{skills_str}

ORIGINAL CONTEXT:
{context}
"""

        return {
            "success": True,
            "agent_id": agent_id,
            "available": True,
            "enhanced_context": enhanced_context,
            "skills_injected": len(skills),
            "format": format,
            "message": f"Injected {len(skills)} skills into context",
        }

    except Exception as e:
        # SECURITY FIX: HVE-3 Information Disclosure - Use safe error messages
        logger.error(f"Failed to inject skills: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to inject skills", e)


# ============================================================================
# MCP Tool Registry Access
# ============================================================================

def get_registered_tools() -> Dict[str, Any]:
    """Get all registered MCP tools (thread-safe)."""
    # THREAD SAFETY FIX: TS-1 - Synchronize registry access
    with _MCP_TOOLS_LOCK:
        return _MCP_TOOLS.copy()

def list_mcp_tools() -> List[str]:
    """List names of all registered MCP tools (thread-safe)."""
    # THREAD SAFETY FIX: TS-1 - Synchronize registry access
    with _MCP_TOOLS_LOCK:
        return list(_MCP_TOOLS.keys())

# Export all MCP tools
__all__ = [
    # MCP Tools
    "initialize_ace_agent",
    "execute_task_with_ace",
    "learn_from_samples_with_ace",
    "learn_from_execution_with_ace",
    "manage_ace_skillbook",
    "get_ace_status",
    "inject_ace_skills_into_context",
    # Utilities
    "get_registered_tools",
    "list_mcp_tools",
    "ACE_AVAILABLE",
]

# Module initialization
if __name__ == "__main__":
    print("ACE MCP Tools Module")
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"Registered Tools: {len(_MCP_TOOLS)}")
    print("\nTools:")
    for tool_name in sorted(_MCP_TOOLS.keys()):
        print(f"  - {tool_name}")
