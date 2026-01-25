"""
ACE + STEER Unified Integration - Enhanced Output Quality System

This module provides a unified interface that combines the Agentic Context Engine (ACE)
with the STEER reliability layer to create a closed-loop system where:
- ACE provides learning and skill acquisition
- STEER provides deterministic verification
- ACE learns from STEER verification failures to improve over time

Architecture:
    Input → ACE Context Enhancement → Agent Execution → STEER Verification → 
    → If Pass: Output + Learning
    → If Fail: STEER Feedback → ACE Learning → Retry/Improve

Author: OpenEvolve Team
License: MIT
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import threading
from functools import wraps

# Import ACE components
try:
    from ace_context_engine import (
        ACEContextEngine,
        get_ace_engine,
        with_ace_context,
        execute_task,
        get_enhanced_prompt,
        ACE_AVAILABLE,
    )
except ImportError as e:
    ACE_AVAILABLE = False
    logging.warning(f"ACE not available: {e}")

# Import STEER components
try:
    from steer_context_engine import (
        SteerContextEngine,
        get_steer_engine,
        with_steer_verification,
        verify_output,
        get_reliable_prompt,
        STEER_AVAILABLE,
    )
except ImportError as e:
    STEER_AVAILABLE = False
    logging.warning(f"STEER not available: {e}")

# Import configuration
try:
    from ace_steer_config import (
        get_ace_steer_config,
        is_steer_enabled,
        is_unified_bridge_enabled,
        get_ace_steer_status,
        validate_ace_steer_config,
        is_any_available,
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    logging.warning(f"ACE+STEER config not available: {e}")

# Import MCP tools
try:
    from ace_mcp_tools import (
        learn_from_execution_with_ace,
        inject_ace_skills_into_context,
    )
    ACE_MCP_AVAILABLE = True
except ImportError as e:
    ACE_MCP_AVAILABLE = False
    logging.warning(f"ACE MCP tools not available: {e}")

try:
    from steer_mcp_tools import (
        run_all_verifications,
        get_steer_status,
    )
    STEER_MCP_AVAILABLE = True
except ImportError as e:
    STEER_MCP_AVAILABLE = False
    logging.warning(f"STEER MCP tools not available: {e}")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AceSteerUnifiedBridge:
    """
    Unified bridge that combines ACE learning with STEER verification for enhanced outputs.
    
    This class creates a closed-loop system where:
    1. ACE enhances prompts with learned skills
    2. Agent executes with enhanced context
    3. STEER verifies the output for reliability
    4. If verification fails, ACE learns from the feedback
    5. Output is improved based on verification feedback
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        enable_ace_learning: bool = True,
        enable_steer_verification: bool = True,
        default_verifications: List[str] = None,
        max_retries_on_failure: int = 2,
        learning_from_failures: bool = True,
        inject_skills: bool = True,
        entropy_threshold: float = 3.5,
        skillbook_path: Optional[str] = None,
        checkpoint_dir: str = "./ace_checkpoints",
    ):
        """
        Initialize the unified ACE+STEER bridge.
        
        Args:
            model: Model to use for ACE components
            enable_ace_learning: Whether to enable ACE learning
            enable_steer_verification: Whether to enable STEER verification
            default_verifications: List of default verifications to run
            max_retries_on_failure: Max retries when verification fails
            learning_from_failures: Whether ACE should learn from STEER failures
            inject_skills: Whether to inject learned skills into context
            entropy_threshold: Entropy threshold for STEER slop detection
            skillbook_path: Path to load/save ACE skillbook
            checkpoint_dir: Directory for ACE checkpoints
        """
        self.model = model
        self.enable_ace_learning = enable_ace_learning
        self.enable_steer_verification = enable_steer_verification
        self.default_verifications = default_verifications or ["json", "slop"]
        self.max_retries_on_failure = max_retries_on_failure
        self.learning_from_failures = learning_from_failures
        self.inject_skills = inject_skills
        self.entropy_threshold = entropy_threshold
        self.skillbook_path = skillbook_path
        self.checkpoint_dir = checkpoint_dir
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize engines
        self.ace_engine = None
        self.steer_engine = None
        
        # Initialize
        self._initialize_engines()
        
        logger.info("ACE+STEER Unified Bridge initialized successfully")
    
    def _initialize_engines(self):
        """Initialize ACE and STEER engines."""
        if self.enable_ace_learning and ACE_AVAILABLE:
            try:
                self.ace_engine = ACEContextEngine(
                    model=self.model,
                    skillbook_path=self.skillbook_path,
                    checkpoint_dir=self.checkpoint_dir,
                )
                logger.info("ACE engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ACE engine: {e}")
                self.ace_engine = None
        
        if self.enable_steer_verification and STEER_AVAILABLE:
            try:
                self.steer_engine = SteerContextEngine(
                    default_entropy_threshold=self.entropy_threshold
                )
                logger.info("STEER engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize STEER engine: {e}")
                self.steer_engine = None
    
    def enhance_prompt_with_both_systems(
        self,
        base_prompt: str,
        domain_context: Optional[Dict[str, Any]] = None,
        task_specific_context: Optional[Dict[str, Any]] = None,
        include_steer_rules: bool = True,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Enhance a prompt using both ACE skills and STEER reliability rules.
        
        Args:
            base_prompt: Original prompt to enhance
            domain_context: Domain-specific context for ACE
            task_specific_context: Task-specific context for ACE
            include_steer_rules: Whether to include STEER reliability rules
            agent_name: Agent name for STEER-specific rules
            
        Returns:
            Enhanced prompt with both ACE skills and STEER rules
        """
        with self._lock:
            enhanced_prompt = base_prompt
            
            # First, enhance with ACE skills if available
            if self.ace_engine and self.inject_skills:
                enhanced_prompt = self.ace_engine.get_context_enhanced_prompt(
                    base_prompt=enhanced_prompt,
                    domain_context=domain_context,
                    task_specific_context=task_specific_context,
                    inject_skills=True,
                )
            
            # Then, enhance with STEER rules if requested
            if self.steer_engine and include_steer_rules:
                enhanced_prompt = self.steer_engine.get_context_enhanced_with_rules(
                    base_prompt=enhanced_prompt,
                    agent_name=agent_name,
                    include_json_rules=True,
                    include_slop_rules=True,
                    include_pii_rules=True,
                    include_citation_rules=True,
                )
            
            return enhanced_prompt
    
    def execute_with_closed_loop(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        verifications: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
        agent_func: Optional[Callable] = None,
        inject_skills: bool = True,
        enable_learning: bool = True,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with closed-loop ACE+STEER verification and learning.
        
        This method implements the full ACE+STEER cycle:
        1. Enhance prompt with ACE skills
        2. Execute task (using agent_func or ACE engine)
        3. Verify output with STEER
        4. If verification fails, learn from feedback and retry
        
        Args:
            task: The task to execute
            context: Additional context for the task
            verifications: List of verifications to run (uses defaults if None)
            max_retries: Max retries on verification failure (uses default if None)
            agent_func: Custom agent function to execute (uses ACE engine if None)
            inject_skills: Whether to inject learned skills
            enable_learning: Whether to enable ACE learning
            ground_truth: Ground truth for ACE learning
            
        Returns:
            Dict with execution results, verification results, and learning outcomes
        """
        with self._lock:
            verifications = verifications or self.default_verifications
            max_retries = max_retries or self.max_retries_on_failure
            
            # Prepare context
            domain_context = context.get('domain_context') if context else None
            task_specific_context = context.get('task_context') if context else None
            
            # Enhance prompt with both systems
            enhanced_prompt = self.enhance_prompt_with_both_systems(
                base_prompt=task,
                domain_context=domain_context,
                task_specific_context=task_specific_context,
                include_steer_rules=True,
                agent_name=context.get('agent_name') if context else 'default'
            )
            
            # Execute and verify in a loop
            execution_result = None
            verification_result = None
            retries = 0
            success = False
            
            while retries <= max_retries and not success:
                try:
                    # Execute the task
                    if agent_func:
                        execution_result = agent_func(enhanced_prompt)
                    else:
                        # Use ACE engine if no custom function provided
                        if self.ace_engine:
                            execution_result = self.ace_engine.execute_with_learning(
                                task=enhanced_prompt,
                                context=context,
                                inject_skills=inject_skills,
                                enable_learning=enable_learning and retries == 0,  # Only learn on first try
                                ground_truth=ground_truth,
                            )
                        else:
                            # Fallback: just return the enhanced prompt
                            execution_result = {
                                "success": True,
                                "result": enhanced_prompt,
                                "learning_applied": False,
                            }

                    # Ensure execution_result is a dictionary for consistent access
                    if not isinstance(execution_result, dict):
                        execution_result = {
                            "success": True,
                            "result": execution_result,
                            "learning_applied": False,
                        }
                    
                    # Verify the output
                    if self.steer_engine and self.enable_steer_verification:
                        verification_result = self.steer_engine.run_all_verifications(
                            output=execution_result.get('result', execution_result),
                            verifications=verifications,
                        )
                        
                        # Check if all verifications passed
                        success = verification_result.get('all_passed', True)
                        
                        if not success:
                            logger.warning(f"Verification failed on attempt {retries + 1}, preparing for retry/relearning")
                            
                            # If learning from failures is enabled, learn from the verification feedback
                            if self.ace_engine and self.learning_from_failures and retries < max_retries:
                                self._learn_from_verification_failure(
                                    task=enhanced_prompt,
                                    execution_result=execution_result,
                                    verification_result=verification_result,
                                    ground_truth=ground_truth
                                )
                            
                            retries += 1
                        else:
                            logger.info(f"Verification passed on attempt {retries + 1}")
                            success = True
                    else:
                        # If STEER is not available or disabled, consider it a success
                        success = True
                        verification_result = {
                            "all_passed": True,
                            "results": [],
                            "failed_verifications": [],
                            "total_verifications": 0,
                            "passed_count": 0,
                        }
                        
                except Exception as e:
                    logger.error(f"Error in ACE+STEER execution loop: {e}")
                    break
            
            # Final result
            result = {
                "task": task,
                "enhanced_prompt": enhanced_prompt,
                "execution_result": execution_result,
                "verification_result": verification_result,
                "retries_used": retries,
                "success": success,
                "final_result": execution_result.get('result', execution_result) if execution_result else None,
            }
            
            # Add learning from final outcome if enabled
            if (self.ace_engine and enable_learning and 
                ((success and retries == 0) or (not success and retries >= max_retries))):
                try:
                    # Learn from the complete execution cycle
                    learning_result = self.ace_engine.execute_with_learning(
                        task=f"Task: {task}\nEnhanced: {enhanced_prompt}",
                        context={
                            "original_task": task,
                            "enhanced_prompt": enhanced_prompt,
                            "execution_result": execution_result,
                            "verification_result": verification_result,
                            "success": success,
                            "retries_used": retries,
                        },
                        inject_skills=False,  # Already injected above
                        enable_learning=True,
                        ground_truth=ground_truth,
                    )
                    result["learning_result"] = learning_result
                except Exception as e:
                    logger.warning(f"Could not apply final learning: {e}")
            
            return result
    
    def _learn_from_verification_failure(
        self,
        task: str,
        execution_result: Dict[str, Any],
        verification_result: Dict[str, Any],
        ground_truth: Optional[str] = None,
    ):
        """
        Internal method to make ACE learn from STEER verification failures.
        
        Args:
            task: Original task
            execution_result: Result from execution
            verification_result: STEER verification results
            ground_truth: Ground truth for learning
        """
        if not self.ace_engine or not self.learning_from_failures:
            return
        
        try:
            # Construct feedback from verification failures
            failure_feedback = []
            for result_item in verification_result.get("results", []):
                if not result_item.get("passed", False):
                    reason = result_item.get("reason", "Unknown reason")
                    judge = result_item.get("judge", "Unknown")
                    fixes = result_item.get("suggested_fixes", [])
                    
                    failure_info = f"[{judge}] {reason}"
                    if fixes:
                        fix_descriptions = [f"{fix.get('title', '')}: {fix.get('description', '')}" 
                                          for fix in fixes]
                        failure_info += f" - Suggestions: {', '.join(fix_descriptions)}"
                    
                    failure_feedback.append(failure_info)
            
            feedback_text = "VERIFICATION FEEDBACK:\n" + "\n".join(failure_feedback)
            
            # Create learning task from the failure
            learning_task = f"""Original Task: {task}
Execution Result: {execution_result.get('result', execution_result)}
{feedback_text}

Learn from this failure to improve future outputs for similar tasks."""
            
            # Have ACE learn from the failure
            learning_result = self.ace_engine.execute_with_learning(
                task=learning_task,
                context={
                    "original_task": task,
                    "execution_result": execution_result,
                    "verification_feedback": feedback_text,
                    "failure_details": verification_result,
                },
                inject_skills=False,
                enable_learning=True,
                ground_truth=ground_truth,
                feedback=feedback_text,
            )
            
            logger.info(f"ACE learned from verification failure, updates applied: {learning_result.get('learning_applied', False)}")
            
        except Exception as e:
            logger.error(f"Error learning from verification failure: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of both ACE and STEER systems.
        
        Returns:
            Dict with status information for both systems
        """
        ace_status = {}
        steer_status = {}
        
        if self.ace_engine:
            ace_status = self.ace_engine.get_status()
        
        if self.steer_engine:
            steer_status = self.steer_engine.get_status()
        
        return {
            "ace_available": ACE_AVAILABLE and self.ace_engine is not None,
            "steer_available": STEER_AVAILABLE and self.steer_engine is not None,
            "ace_status": ace_status,
            "steer_status": steer_status,
            "unified_bridge_active": (ACE_AVAILABLE and STEER_AVAILABLE and 
                                    self.ace_engine is not None and 
                                    self.steer_engine is not None),
            "default_verifications": self.default_verifications,
            "max_retries": self.max_retries_on_failure,
        }


def create_ace_steer_agent(
    agent_func: Callable,
    verifications: List[str] = None,
    max_retries: int = 2,
    inject_skills: bool = True,
    enable_learning: bool = True,
    learning_from_failures: bool = True,
) -> Callable:
    """
    Create an agent function wrapped with both ACE enhancement and STEER verification.
    
    Args:
        agent_func: The base agent function to wrap
        verifications: List of verifications to run
        max_retries: Max retries on verification failure
        inject_skills: Whether to inject ACE skills
        enable_learning: Whether to enable ACE learning
        learning_from_failures: Whether to learn from verification failures
        
    Returns:
        Wrapped agent function with ACE+STEER integration
    """
    unified_bridge = AceSteerUnifiedBridge(
        default_verifications=verifications or ["json", "slop"],
        max_retries_on_failure=max_retries,
        learning_from_failures=learning_from_failures,
        inject_skills=inject_skills,
    )
    
    @wraps(agent_func)
    def wrapper(task: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        # Use the unified execution method
        result = unified_bridge.execute_with_closed_loop(
            task=task,
            context=context,
            verifications=verifications,
            max_retries=max_retries,
            agent_func=agent_func,
            inject_skills=inject_skills,
            enable_learning=enable_learning,
        )
        
        return result
    
    return wrapper


def ace_steer_capture(
    verifications: List[str] = None,
    max_retries: int = 2,
    inject_skills: bool = True,
    enable_learning: bool = True,
    learning_from_failures: bool = True,
    **ace_kwargs,
) -> Callable:
    """
    Decorator that combines ACE context enhancement with STEER verification.
    
    This decorator:
    1. Enhances the input with ACE learned skills
    2. Executes the function
    3. Verifies output with STEER
    4. If verification fails, ACE learns from the feedback
    5. Optionally retries based on verification feedback
    
    Args:
        verifications: List of verifications to run
        max_retries: Max retries on verification failure
        inject_skills: Whether to inject ACE skills
        enable_learning: Whether to enable ACE learning
        learning_from_failures: Whether to learn from verification failures
        **ace_kwargs: Additional arguments for ACE context enhancement
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Create unified bridge
        unified_bridge = AceSteerUnifiedBridge(
            default_verifications=verifications or ["json", "slop"],
            max_retries_on_failure=max_retries,
            learning_from_failures=learning_from_failures,
            inject_skills=inject_skills,
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args/kwargs to a task representation
            task_repr = f"Function: {func.__name__}, Args: {str(args)[:500]}, Kwargs: {str(kwargs)[:500]}"

            # Prepare context
            context = {
                "function_name": func.__name__,
                "args": args,
                "kwargs": kwargs,
                **ace_kwargs  # Additional ACE context
            }

            # Execute with closed-loop ACE+STEER
            result = unified_bridge.execute_with_closed_loop(
                task=task_repr,
                context=context,
                verifications=verifications,
                max_retries=max_retries,
                agent_func=lambda enhanced_task: func(*args, **kwargs),
                inject_skills=inject_skills,
                enable_learning=enable_learning,
            )

            # Return the final result or raise an error if verification failed
            if not result["success"]:
                logger.warning(f"ACE+STEER verification failed after {result['retries_used']} attempts")
                # Optionally raise an error based on configuration
                # For now, return the result with verification info attached
                return result

            # Handle both dict and non-dict results
            final_result = result["final_result"]
            if isinstance(final_result, dict) and "result" in final_result:
                return final_result["result"]
            else:
                return final_result

        return wrapper
    return decorator


# Global unified bridge instance
_unified_bridge = None
_bridge_lock = threading.Lock()


def get_unified_bridge() -> AceSteerUnifiedBridge:
    """
    Get the global unified ACE+STEER bridge instance.
    
    Returns:
        AceSteerUnifiedBridge instance
    """
    global _unified_bridge
    
    with _bridge_lock:
        if _unified_bridge is None:
            _unified_bridge = AceSteerUnifiedBridge()
        return _unified_bridge


# Convenience functions
def execute_with_ace_steer(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    verifications: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute a task with the global unified ACE+STEER bridge.
    
    Args:
        task: The task to execute
        context: Additional context
        verifications: List of verifications to run
        
    Returns:
        Execution result with verification and learning info
    """
    bridge = get_unified_bridge()
    return bridge.execute_with_closed_loop(task, context, verifications)


def enhance_with_ace_steer_rules(
    base_prompt: str,
    domain_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enhance a prompt with both ACE skills and STEER rules.
    
    Args:
        base_prompt: Original prompt
        domain_context: Domain-specific context
        
    Returns:
        Enhanced prompt
    """
    bridge = get_unified_bridge()
    return bridge.enhance_prompt_with_both_systems(
        base_prompt=base_prompt,
        domain_context=domain_context
    )


# Export commonly used items
__all__ = [
    "AceSteerUnifiedBridge",
    "get_unified_bridge",
    "create_ace_steer_agent",
    "ace_steer_capture",
    "execute_with_ace_steer",
    "enhance_with_ace_steer_rules",
    "ACE_AVAILABLE",
    "STEER_AVAILABLE",
    "CONFIG_AVAILABLE",
]


if __name__ == "__main__":
    print("ACE+STEER Unified Integration Module")
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"STEER Available: {STEER_AVAILABLE}")
    print(f"Config Available: {CONFIG_AVAILABLE}")
    
    if ACE_AVAILABLE and STEER_AVAILABLE:
        print("\nInitializing ACE+STEER Unified Bridge...")
        bridge = AceSteerUnifiedBridge()
        status = bridge.get_status()
        print(f"Unified Bridge Status: {status}")
    else:
        print("\nOne or both systems (ACE/STEER) not available.")
        print("Install both agentic-context-engine and steer to enable full functionality.")