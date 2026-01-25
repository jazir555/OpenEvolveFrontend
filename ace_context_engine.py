"""
ACE (Agentic Context Engine) - Central Context Engine Module

This module serves as the central hub for the Agentic Context Engine integration,
providing a unified interface to all ACE functionality across the OpenEvolve platform.

The Agentic Context Engine enables self-improving agents that learn from their
executions through a three-role system:
- Agent: Executes tasks using learned skills
- Reflector: Analyzes what worked and what didn't
- SkillManager: Updates the skillbook with new skills

This module integrates with:
- CrewAI workflows (via ace_crewai_bridge.py)
- MCP tools (via ace_mcp_tools.py)
- Various domain-specific agents (blue_team, red_team, gold_team, etc.)

Author: OpenEvolve Team
License: MIT
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import threading
from functools import wraps

# Add agentic-context-engine to path
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# Import ACE core components
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
        LiteLLMClient,
        AgentOutput,
        ReflectorOutput,
        SkillManagerOutput,
        ThreadSafeSkillbook,
        AsyncLearningPipeline,
    )
    from ace.prompts_v2_1 import PromptManager
    from ace.deduplication import DeduplicationManager, DeduplicationConfig
    ACE_AVAILABLE = True
except ImportError as e:
    ACE_AVAILABLE = False
    logging.warning(f"ACE not available: {e}")
    
    # Define stub classes for graceful degradation
    class StubClass:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return None
    
    # Create stubs
    Skillbook = StubClass
    Skill = StubClass
    Sample = StubClass
    SimpleEnvironment = StubClass
    OfflineACE = StubClass
    OnlineACE = StubClass
    Agent = StubClass
    Reflector = StubClass
    SkillManager = StubClass
    LiteLLMClient = StubClass
    AgentOutput = StubClass
    ReflectorOutput = StubClass
    SkillManagerOutput = StubClass
    ThreadSafeSkillbook = StubClass
    AsyncLearningPipeline = StubClass
    PromptManager = StubClass
    DeduplicationManager = StubClass
    DeduplicationConfig = StubClass

# Import supporting modules
try:
    from ace_crewai_bridge import ACECrewAIWorkflowBridge
    from ace_mcp_tools import (
        initialize_ace_agent,
        execute_task_with_ace,
        learn_from_samples_with_ace,
        learn_from_execution_with_ace,
        manage_ace_skillbook,
        get_ace_status,
        inject_ace_skills_into_context,
    )
    from ace_integration import ACEIntegration
    from ace_security_utils import (
        validate_file_path_safe,
        validate_string_length,
        validate_numeric_range,
        validate_model_name,
        sanitize_for_logging,
    )
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logging.warning(f"Core ACE modules not available: {e}")

# Define fallback validation functions if ace_security_utils not available
if CORE_MODULES_AVAILABLE:
    # Functions already imported above
    pass
else:
    # Define fallback validation functions
    def validate_file_path_safe(filepath: str, base_dir: str = ".") -> str:
        """Fallback path validation."""
        if not filepath or not isinstance(filepath, str):
            raise ValueError("File path must be a non-empty string")
        if '..' in filepath or '|' in filepath or ';' in filepath:
            raise ValueError(f"Unsafe path: {filepath}")
        return str(Path(filepath).resolve())

    def validate_string_length(value: str, name: str, max_length: int = 10000, min_length: int = 0, allow_empty: bool = True) -> str:
        """Fallback string length validation."""
        if not allow_empty and (not value or not isinstance(value, str)):
            raise ValueError(f"{name} must be a non-empty string")
        if value and len(value) > max_length:
            raise ValueError(f"{name} exceeds maximum length of {max_length}")
        if value and len(value) < min_length:
            raise ValueError(f"{name} is below minimum length of {min_length}")
        return value

    def validate_numeric_range(value, name: str, min_val: float = float('-inf'), max_val: float = float('inf'),
                              allow_nan: bool = True, allow_infinity: bool = True) -> float:
        """Fallback numeric range validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if not allow_nan and (value != value):  # NaN check
            raise ValueError(f"{name} cannot be NaN")
        if not allow_infinity and (value == float('inf') or value == float('-inf')):
            raise ValueError(f"{name} cannot be infinity")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return value

    def validate_model_name(model: str) -> str:
        """Fallback model name validation."""
        if not model or not isinstance(model, str):
            raise ValueError("Model name must be a non-empty string")
        if any(char in model for char in [';', '&', '|', '$', '`', '\n']):
            raise ValueError(f"Model name contains invalid characters")
        return model

    def sanitize_for_logging(text: str) -> str:
        """Fallback sanitization for logging."""
        if not text:
            return str(text)
        # Basic sanitization - remove potentially harmful characters
        return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACEContextEngine:
    """
    Centralized Agentic Context Engine for OpenEvolve Platform
    
    This class provides a unified interface to all ACE functionality,
    managing skillbooks, agents, and learning across the entire platform.
    
    Key Features:
    - Centralized skillbook management
    - Multi-agent coordination
    - Learning from execution feedback
    - Context injection for enhanced reasoning
    - Thread-safe operations
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        skillbook_path: Optional[str] = None,
        checkpoint_dir: str = "./ace_checkpoints",
        enable_async_learning: bool = True,
        enable_deduplication: bool = True,
        dedup_threshold: float = 0.85,
        max_skills: int = 1000,
        min_helpful: int = 5,
    ):
        """
        Initialize the ACE Context Engine.
        
        Args:
            model: LiteLLM model name for ACE components
            skillbook_path: Path to load existing skillbook
            checkpoint_dir: Directory for skillbook checkpoints
            enable_async_learning: Enable asynchronous learning
            enable_deduplication: Enable skill deduplication
            dedup_threshold: Similarity threshold for deduplication
            max_skills: Maximum skills to keep in skillbook
            min_helpful: Minimum helpful count to keep a skill
        """
        self.model = model
        self.skillbook_path = skillbook_path
        self.checkpoint_dir = checkpoint_dir
        self.enable_async_learning = enable_async_learning
        self.enable_deduplication = enable_deduplication
        self.dedup_threshold = dedup_threshold
        self.max_skills = max_skills
        self.min_helpful = min_helpful
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize ACE components
        self.skillbook = None
        self.agent = None
        self.reflector = None
        self.skill_manager = None
        self.llm_client = None
        self.prompt_manager = None
        self.deduplication_manager = None
        
        # Initialize bridge components
        self.crewai_bridge = None
        self.integration = None
        
        # Initialize
        self._initialize_components()
        
        logger.info("ACE Context Engine initialized successfully")
    
    def _initialize_components(self):
        """Initialize all ACE components."""
        if not ACE_AVAILABLE:
            logger.warning("ACE not available - initializing with stubs")
            return

        try:
            # Create checkpoint directory
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Load or create skillbook
            if self.skillbook_path and os.path.exists(self.skillbook_path):
                try:
                    self.skillbook = Skillbook.load_from_file(self.skillbook_path)
                    logger.info(f"Loaded skillbook from {self.skillbook_path}")
                except Exception as e:
                    logger.warning(f"Could not load skillbook from {self.skillbook_path}: {e}")
                    self.skillbook = Skillbook()
            else:
                self.skillbook = Skillbook()
                logger.info("Created new skillbook")

            # Create LLM client
            try:
                self.llm_client = LiteLLMClient(model=self.model)
            except Exception as e:
                logger.warning(f"Could not create LLM client: {e}")
                self.llm_client = None

            # Get prompt templates
            try:
                self.prompt_manager = PromptManager()
            except Exception as e:
                logger.warning(f"Could not initialize prompt manager: {e}")
                self.prompt_manager = None

            # Create ACE roles
            try:
                if self.llm_client and self.prompt_manager:
                    self.agent = Agent(
                        self.llm_client,
                        prompt_template=self.prompt_manager.get_agent_prompt()
                    )
                    self.reflector = Reflector(
                        self.llm_client,
                        prompt_template=self.prompt_manager.get_reflector_prompt()
                    )
                    self.skill_manager = SkillManager(
                        self.llm_client,
                        prompt_template=self.prompt_manager.get_skill_manager_prompt()
                    )
                else:
                    logger.warning("Could not create ACE roles due to missing LLM client or prompt manager")
                    self.agent = None
                    self.reflector = None
                    self.skill_manager = None
            except Exception as e:
                logger.warning(f"Could not create ACE roles: {e}")
                self.agent = None
                self.reflector = None
                self.skill_manager = None

            # Initialize deduplication manager
            try:
                if self.enable_deduplication and self.skillbook:
                    dedup_config = DeduplicationConfig(threshold=self.dedup_threshold)
                    self.deduplication_manager = DeduplicationManager(config=dedup_config)
            except Exception as e:
                logger.warning(f"Could not initialize deduplication manager: {e}")
                self.deduplication_manager = None

            # Initialize bridge components
            try:
                if CORE_MODULES_AVAILABLE:
                    self.crewai_bridge = ACECrewAIWorkflowBridge(
                        model=self.model,
                        skillbook_path=self.skillbook_path,
                        checkpoint_dir=self.checkpoint_dir,
                        max_skills=self.max_skills,
                        min_helpful=self.min_helpful,
                    )
                else:
                    logger.warning("Could not initialize CrewAI bridge due to missing core modules")
                    self.crewai_bridge = None
            except Exception as e:
                logger.warning(f"Could not initialize CrewAI bridge: {e}")
                self.crewai_bridge = None

            # Initialize integration layer
            try:
                if CORE_MODULES_AVAILABLE:
                    self.integration = ACEIntegration()
                else:
                    logger.warning("Could not initialize ACE integration due to missing core modules")
                    self.integration = None
            except Exception as e:
                logger.warning(f"Could not initialize ACE integration: {e}")
                self.integration = None

            logger.info("ACE components initialized (some may be unavailable due to dependencies)")

        except Exception as e:
            logger.error(f"Failed to initialize ACE components: {e}")
            # Still try to continue with available components
    
    def get_context_enhanced_prompt(
        self,
        base_prompt: str,
        domain_context: Optional[Dict[str, Any]] = None,
        task_specific_context: Optional[Dict[str, Any]] = None,
        inject_skills: bool = True,
        max_skills_to_inject: int = 20,
    ) -> str:
        """
        Generate a context-enhanced prompt with learned skills and domain knowledge.
        
        Args:
            base_prompt: Original prompt to enhance
            domain_context: Domain-specific context information
            task_specific_context: Task-specific context
            inject_skills: Whether to inject learned skills
            max_skills_to_inject: Maximum number of skills to inject
            
        Returns:
            Enhanced prompt string with context
        """
        with self._lock:
            enhanced_parts = []
            
            # Add domain context if provided
            if domain_context:
                enhanced_parts.append("DOMAIN CONTEXT:")
                for key, value in domain_context.items():
                    enhanced_parts.append(f"- {key}: {value}")
                enhanced_parts.append("")
            
            # Add task-specific context if provided
            if task_specific_context:
                enhanced_parts.append("TASK-SPECIFIC CONTEXT:")
                for key, value in task_specific_context.items():
                    enhanced_parts.append(f"- {key}: {value}")
                enhanced_parts.append("")
            
            # Add learned skills if enabled
            if inject_skills and self.skillbook and self.skillbook.skills():
                skills = self.skillbook.skills()[:max_skills_to_inject]
                if skills:
                    enhanced_parts.append("LEARNED SKILLS FROM PREVIOUS EXECUTIONS:")
                    for skill in skills:
                        enhanced_parts.append(f"- {skill.strategy}")
                    enhanced_parts.append("")
            
            # Add base prompt
            enhanced_parts.append("TASK:")
            enhanced_parts.append(base_prompt)
            
            return "\n".join(enhanced_parts)
    
    def execute_with_learning(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        inject_skills: bool = True,
        enable_learning: bool = True,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with ACE learning capabilities.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            inject_skills: Whether to inject learned skills
            enable_learning: Whether to learn from this execution
            ground_truth: Ground truth for evaluation (optional)
            feedback: Feedback for learning (optional)
            
        Returns:
            Dictionary with execution results and learning outcomes
        """
        with self._lock:
            if not ACE_AVAILABLE:
                return {
                    "success": False,
                    "error": "ACE not available",
                    "result": task,  # Return original task as fallback
                    "learning_applied": False,
                }
            
            try:
                # Enhance context with skills if enabled
                enhanced_context = task
                if inject_skills and self.skillbook:
                    skills_context = self.skillbook.as_prompt()
                    if skills_context.strip():
                        enhanced_context = f"{skills_context}\n\nTASK:\n{task}"
                
                # Create sample
                sample = Sample(
                    query=task,
                    context=enhanced_context,
                    ground_truth=ground_truth,
                )
                
                # Execute agent
                agent_output = self.agent.run(sample)
                
                # Learn from execution if enabled
                learning_result = None
                if enable_learning and self.reflector and self.skill_manager:
                    # Create environment result if ground truth provided
                    env_result = None
                    if ground_truth:
                        env_result = SimpleEnvironment().evaluate(sample, agent_output)
                    
                    # Reflector analysis
                    reflection = self.reflector.run(
                        sample=sample,
                        agent_output=agent_output,
                        skillbook=self.skillbook,
                        environment_result=env_result,
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
                    
                    learning_result = {
                        "updates_applied": updates_applied,
                        "reflection_summary": reflection.summary if reflection else "",
                        "feedback_incorporated": bool(feedback),
                    }
                
                # Return results
                result = {
                    "success": True,
                    "task": task,
                    "result": agent_output.final_answer,
                    "reasoning": agent_output.reasoning,
                    "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
                    "learning_applied": bool(learning_result),
                    "learning_result": learning_result,
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Execution with learning failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "result": task,  # Return original task as fallback
                    "learning_applied": False,
                }
    
    def learn_from_batch(
        self,
        samples: List[Dict[str, Any]],
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Learn from a batch of samples using offline learning.
        
        Args:
            samples: List of samples with 'query' and optional 'ground_truth'
            epochs: Number of training epochs
            checkpoint_interval: Save checkpoint every N samples
            
        Returns:
            Dictionary with learning results
        """
        with self._lock:
            if not ACE_AVAILABLE:
                return {
                    "success": False,
                    "error": "ACE not available",
                    "samples_processed": 0,
                    "skills_learned": 0,
                }
            
            try:
                # Convert input samples to ACE Sample objects
                ace_samples = []
                for sample_data in samples:
                    if not isinstance(sample_data, dict):
                        continue
                    if "query" not in sample_data:
                        continue
                    
                    ace_samples.append(Sample(
                        query=sample_data["query"],
                        ground_truth=sample_data.get("ground_truth"),
                        context=sample_data.get("context", ""),
                    ))
                
                # Create offline ACE adapter
                adapter = OfflineACE(
                    skillbook=self.skillbook,
                    agent=self.agent,
                    reflector=self.reflector,
                    skill_manager=self.skill_manager,
                    async_learning=self.enable_async_learning,
                )
                
                # Create environment
                environment = SimpleEnvironment()
                
                # Run learning
                results = adapter.run(
                    ace_samples,
                    environment,
                    epochs=epochs,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=self.checkpoint_dir,
                )
                
                return {
                    "success": True,
                    "samples_processed": len(ace_samples),
                    "skills_before": results.get("skills_before", 0) if results else 0,
                    "skills_after": len(self.skillbook.skills()),
                    "skills_learned": len(self.skillbook.skills()) - (results.get("skills_before", 0) if results else 0),
                    "epochs_completed": epochs,
                    "results": results,
                }
                
            except Exception as e:
                logger.error(f"Batch learning failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "samples_processed": 0,
                    "skills_learned": 0,
                }
    
    def save_skillbook(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current skillbook to file.
        
        Args:
            filepath: Optional filepath (defaults to initialized path)
            
        Returns:
            Dictionary with save results
        """
        with self._lock:
            if not ACE_AVAILABLE or not self.skillbook:
                return {
                    "success": False,
                    "error": "ACE not available or no skillbook to save",
                }
            
            try:
                save_path = filepath or self.skillbook_path
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(self.checkpoint_dir, f"skillbook_{timestamp}.json")
                
                # Validate path for security
                save_path = validate_file_path_safe(save_path, self.checkpoint_dir)
                
                self.skillbook.save_to_file(save_path)
                
                return {
                    "success": True,
                    "filepath": save_path,
                    "skills_saved": len(self.skillbook.skills()),
                }
                
            except Exception as e:
                logger.error(f"Failed to save skillbook: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }
    
    def load_skillbook(self, filepath: str) -> Dict[str, Any]:
        """
        Load a skillbook from file.
        
        Args:
            filepath: Path to skillbook file
            
        Returns:
            Dictionary with load results
        """
        with self._lock:
            if not ACE_AVAILABLE:
                return {
                    "success": False,
                    "error": "ACE not available",
                }
            
            try:
                # Validate path for security
                filepath = validate_file_path_safe(filepath, self.checkpoint_dir)
                
                self.skillbook = Skillbook.load_from_file(filepath)
                
                return {
                    "success": True,
                    "filepath": filepath,
                    "skills_loaded": len(self.skillbook.skills()),
                }
                
            except Exception as e:
                logger.error(f"Failed to load skillbook: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the ACE Context Engine.
        
        Returns:
            Dictionary with status information
        """
        return {
            "available": ACE_AVAILABLE,
            "core_modules_available": CORE_MODULES_AVAILABLE,
            "skillbook_size": len(self.skillbook.skills()) if self.skillbook else 0,
            "model": self.model,
            "checkpoint_dir": self.checkpoint_dir,
            "async_learning_enabled": self.enable_async_learning,
            "deduplication_enabled": self.enable_deduplication,
            "crewai_bridge_available": self.crewai_bridge is not None,
            "integration_available": self.integration is not None,
        }
    
    def cleanup_old_skills(self):
        """
        Clean up old/low-performing skills to manage memory usage.
        """
        if not self.skillbook:
            return
            
        skills = self.skillbook.skills()
        if len(skills) <= self.max_skills:
            return  # No need to clean up
        
        # Sort by helpful count and remove lowest performing ones
        skills_sorted = sorted(skills, key=lambda s: s.helpful_count, reverse=True)
        
        # Remove skills beyond max limit that have low helpful count
        for skill in skills_sorted[self.max_skills:]:
            if skill.helpful_count < self.min_helpful:
                self.skillbook.remove(skill.strategy)
    
    def get_enhanced_agent(
        self,
        agent_type: str,
        agent_config: Dict[str, Any],
        domain: str
    ) -> Any:
        """
        Get an ACE-enhanced agent for specific use cases.
        
        Args:
            agent_type: Type of agent ("solver", "patcher", "red_team", "gold_team", etc.)
            agent_config: Base agent configuration
            domain: Domain for specialization
            
        Returns:
            ACE-enhanced agent configuration
        """
        if self.integration:
            return self.integration.enhance_agent_with_ace(
                agent_type=agent_type,
                agent_config=agent_config,
                domain=domain
            )
        else:
            # Return a basic configuration if integration not available
            return {
                "agent_type": agent_type,
                "base_config": agent_config,
                "domain": domain,
                "ace_enhanced": False,
            }


# Global instance for easy access
_ace_engine = None
_engine_lock = threading.Lock()


def get_ace_engine() -> ACEContextEngine:
    """
    Get the global ACE Context Engine instance.
    
    Returns:
        ACEContextEngine instance
    """
    global _ace_engine
    
    with _engine_lock:
        if _ace_engine is None:
            _ace_engine = ACEContextEngine()
        return _ace_engine


def with_ace_context(
    inject_skills: bool = True,
    enable_learning: bool = True,
    domain: Optional[str] = None
):
    """
    Decorator to add ACE context to any function.
    
    Args:
        inject_skills: Whether to inject learned skills
        enable_learning: Whether to enable learning from execution
        domain: Domain context for specialization
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get ACE engine
            ace_engine = get_ace_engine()
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Optionally learn from the execution
            if enable_learning and ACE_AVAILABLE:
                try:
                    # Create a task description from the function call
                    task_desc = f"Function: {func.__name__}, Args: {str(args)[:500]}, Kwargs: {str(kwargs)[:500]}"
                    
                    # Learn from this execution
                    ace_engine.execute_with_learning(
                        task=task_desc,
                        context={"function_result": str(result)[:1000]},
                        inject_skills=inject_skills,
                        enable_learning=True,
                    )
                except Exception as e:
                    logger.warning(f"Learning from execution failed: {e}")
            
            return result
        return wrapper
    return decorator


# Convenience functions for common operations
def execute_task(task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a task using the global ACE engine.
    
    Args:
        task: The task to execute
        context: Additional context
        
    Returns:
        Execution results
    """
    ace_engine = get_ace_engine()
    return ace_engine.execute_with_learning(task, context)


def get_enhanced_prompt(
    base_prompt: str,
    domain_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get a context-enhanced prompt using the global ACE engine.
    
    Args:
        base_prompt: Original prompt
        domain_context: Domain-specific context
        
    Returns:
        Enhanced prompt string
    """
    ace_engine = get_ace_engine()
    return ace_engine.get_context_enhanced_prompt(base_prompt, domain_context)


# Export commonly used items
__all__ = [
    "ACEContextEngine",
    "get_ace_engine",
    "with_ace_context",
    "execute_task",
    "get_enhanced_prompt",
    "ACE_AVAILABLE",
    "CORE_MODULES_AVAILABLE",
]


if __name__ == "__main__":
    print("ACE Context Engine Module")
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"Core Modules Available: {CORE_MODULES_AVAILABLE}")
    
    if ACE_AVAILABLE:
        print("\nInitializing ACE Context Engine...")
        engine = ACEContextEngine()
        status = engine.get_status()
        print(f"Engine Status: {status}")
    else:
        print("\nACE not available - this may be due to missing dependencies.")
        print("Install agentic-context-engine to enable full functionality:")
        print("  pip install agentic-context-engine")