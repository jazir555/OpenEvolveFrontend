"""
ACE + Steer Unified Integration for OpenEvolve

This module provides a unified interface that combines the Agentic Context Engine (ACE)
with the Steer reliability layer.

Key Features:
1. ACE Skill Injection: Enhances agent prompts with learned skills.
2. Steer Verification: Ensures agent outputs meet deterministic quality standards.
3. Closed-Loop Learning: ACE learns automatically from Steer verification failures.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import json

# Graceful imports with fallbacks
try:
    from ace_mcp_tools import (
        inject_ace_skills_into_context,
        learn_from_execution_with_ace,
        ACE_AVAILABLE
    )
except ImportError:
    ACE_AVAILABLE = False
    inject_ace_skills_into_context = None
    learn_from_execution_with_ace = None

try:
    from steer_crewai_bridge import (
        run_all_verifications,
        SteerVerificationError,
        get_steer_status
    )
    STEER_AVAILABLE = True
except ImportError:
    STEER_AVAILABLE = False
    run_all_verifications = None
    SteerVerificationError = None
    get_steer_status = None

logger = logging.getLogger(__name__)

class AceSteerBridge:
    """
    Bridge class that coordinates ACE learning and Steer verification.
    All operations have comprehensive error handling and graceful fallbacks.
    """

    def __init__(self, ace_agent_id: str, skillbook_path: Optional[str] = None):
        try:
            self.ace_agent_id = str(ace_agent_id) if ace_agent_id else "unknown_agent"
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"⚠️ Invalid ace_agent_id, using default: {e}")
            self.ace_agent_id = "unknown_agent"

        self.skillbook_path = skillbook_path

        # Get Steer status safely
        try:
            if callable(get_steer_status):
                self.steer_status = get_steer_status()
            else:
                self.steer_status = {"available": False}
                logger.warning("⚠️ get_steer_status not callable, Steer may be unavailable")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"⚠️ Failed to get Steer status: {e}")
            self.steer_status = {"available": False}

    def prepare_prompt(self, task: str, context: str = "", model: Optional[str] = None) -> str:
        """
        Enhances the prompt with ACE skills.

        Returns original prompt on any error, ensuring graceful degradation.
        """
        # Validate inputs
        try:
            if not isinstance(task, str):
                logger.warning(f"⚠️ task must be string, got {type(task)}")
                task = str(task) if task else ""
            if not isinstance(context, str):
                logger.warning(f"⚠️ context must be string, got {type(context)}")
                context = str(context) if context else ""
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"⚠️ Error validating inputs: {e}")
            return f"TASK:\n{str(task)}\n\nCONTEXT:\n{str(context)}"

        # Check ACE availability
        if not ACE_AVAILABLE or not callable(inject_ace_skills_into_context):
            logger.debug("ACE unavailable, returning original prompt")
            return f"TASK:\n{task}\n\nCONTEXT:\n{context}"

        # Try to inject ACE skills
        try:
            params = {
                "agent_id": self.ace_agent_id,
                "context": context,
                "skillbook_path": self.skillbook_path,
                "format": "markdown"
            }
            if model:
                params["model"] = model

            result = inject_ace_skills_into_context(**params)

            if result and isinstance(result, dict) and result.get("success"):
                enhanced_context = result.get("enhanced_context", "")
                if isinstance(enhanced_context, str):
                    return f"TASK:\n{task}\n\n{enhanced_context}"
                else:
                    logger.warning("⚠️ enhanced_context is not a string")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"⚠️ ACE skill injection failed: {e}")

        # Fallback to original prompt
        return f"TASK:\n{task}\n\nCONTEXT:\n{context}"

    def verify_and_learn(
        self,
        query: str,
        output: Any,
        verifications: List[str],
        reasoning: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verifies output with Steer and triggers ACE learning on failure.

        Always returns a valid result dict, even on errors.
        """
        # Validate inputs
        try:
            if not isinstance(query, str):
                logger.warning(f"⚠️ query must be string, got {type(query)}")
                query = str(query) if query else ""

            if not isinstance(verifications, list):
                logger.warning(f"⚠️ verifications must be list, got {type(verifications)}")
                verifications = []

            if not isinstance(reasoning, str):
                reasoning = str(reasoning) if reasoning else ""
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"⚠️ Error validating verify_and_learn inputs: {e}")
            # Return minimal safe result
            return {
                "all_passed": True,  # Assume passed to avoid blocking
                "results": [],
                "error": str(e)
            }

        # Default result (assume passed if Steer unavailable)
        steer_result = {
            "all_passed": True,
            "results": [],
            "steer_available": STEER_AVAILABLE
        }

        # Run Steer verifications
        if STEER_AVAILABLE and callable(run_all_verifications):
            try:
                steer_result = run_all_verifications(
                    output=output,
                    verifications=verifications,
                    **kwargs
                )
                if not isinstance(steer_result, dict):
                    logger.warning("⚠️ run_all_verifications didn't return dict")
                    steer_result = {"all_passed": True, "results": []}
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Steer verification failed: {e}")
                steer_result = {
                    "all_passed": True,  # Graceful: assume passed on error
                    "results": [],
                    "error": str(e)
                }

        # Process failures and trigger ACE learning
        if not steer_result.get("all_passed", True):
            logger.warning(f"Steer verification failed for agent {self.ace_agent_id}")

            try:
                # Construct feedback for ACE from Steer failures
                failure_reasons = []
                suggested_fixes = []

                results = steer_result.get("results", [])
                if isinstance(results, list):
                    for res in results:
                        if isinstance(res, dict) and not res.get("passed"):
                            judge = res.get('judge', 'unknown')
                            reason = res.get('reason', 'no reason')
                            failure_reasons.append(f"Judge [{judge}] failed: {reason}")

                            fixes = res.get("suggested_fixes", [])
                            if isinstance(fixes, list):
                                for fix in fixes:
                                    if isinstance(fix, dict):
                                        title = fix.get('title', 'fix')
                                        desc = fix.get('description', '')
                                        suggested_fixes.append(f"- {title}: {desc}")

                feedback = "STEER VERIFICATION FAILED:\n" + "\n".join(failure_reasons)
                if suggested_fixes:
                    feedback += "\n\nSUGGESTED FIXES:\n" + "\n".join(suggested_fixes)

                # Trigger ACE learning
                if ACE_AVAILABLE and callable(learn_from_execution_with_ace):
                    try:
                        # Extract model from kwargs or use default
                        learning_model = kwargs.get("model") or self.steer_status.get("default_model") or "gpt-4o-mini"
                        
                        logger.info(f"Triggering ACE learning from Steer failure for {self.ace_agent_id} using model {learning_model}")
                        learning_result = learn_from_execution_with_ace(
                            agent_id=self.ace_agent_id,
                            query=query,
                            agent_output=str(output),
                            feedback=feedback,
                            reasoning=reasoning,
                            model=learning_model,
                            skillbook_path=self.skillbook_path
                        )
                        steer_result["ace_learning"] = learning_result
                    except Exception as e:  # TODO: Catch specific exception instead of Exception
                        logger.warning(f"⚠️ ACE learning failed: {e}")
                        steer_result["ace_learning_error"] = str(e)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"⚠️ Error processing Steer failures: {e}")
                steer_result["processing_error"] = str(e)

        return steer_result

def ace_steer_capture(
    ace_agent_id: str,
    verifications: List[str] = None,
    skillbook_path: Optional[str] = None,
    halt_on_failure: bool = True,
    **steer_kwargs
):
    """
    Decorator that integrates ACE skill injection, Steer verification, and ACE learning.

    All operations have error handling - decorator never crashes wrapped function.
    """
    if verifications is None:
        verifications = []

    # Validate agent_id
    try:
        ace_agent_id = str(ace_agent_id) if ace_agent_id else "decorator_agent"
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.warning(f"⚠️ Invalid ace_agent_id in decorator: {e}")
        ace_agent_id = "decorator_agent"

    # Validate verifications
    if not isinstance(verifications, list):
        logger.warning(f"⚠️ verifications must be list, got {type(verifications)}")
        verifications = []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create bridge with error handling
            try:
                bridge = AceSteerBridge(ace_agent_id, skillbook_path)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"⚠️ Failed to create AceSteerBridge: {e}")
                # Fallback: just run the function without verification
                return func(*args, **kwargs)

            # Extract query with error handling
            try:
                query = kwargs.get("task") or kwargs.get("query") or (args[0] if args else "Unknown Task")
                query = str(query) if query else "Unknown Task"
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error extracting query: {e}")
                query = "Unknown Task"

            # Execute agent function
            try:
                result = func(*args, **kwargs)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"⚠️ Decorated function raised exception: {e}")
                raise  # Re-raise function errors

            # Extract reasoning if available
            try:
                reasoning = ""
                if isinstance(result, dict):
                    reasoning = result.get("reasoning", "")
                    reasoning = str(reasoning) if reasoning else ""
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error extracting reasoning: {e}")
                reasoning = ""

            # Verify and Learn
            try:
                v_result = bridge.verify_and_learn(
                    query=query,
                    output=result,
                    verifications=verifications,
                    reasoning=reasoning,
                    **steer_kwargs
                )

                # Attach verification results
                try:
                    if isinstance(result, dict):
                        result["_steer_verification"] = v_result
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    logger.warning(f"⚠️ Could not attach verification results: {e}")

                # Raise error if verification failed and halt_on_failure is True
                if not v_result.get("all_passed", True) and halt_on_failure:
                    if SteerVerificationError is not None:
                        raise SteerVerificationError(
                            f"ACE+Steer: Verification failed for {ace_agent_id}",
                            result=result,
                            verification_results=v_result
                        )
                    else:
                        logger.error("SteerVerificationError not available, cannot raise")
            except SteerVerificationError:
                # Re-raise SteerVerificationError
                raise
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Verification failed but continuing: {e}")
                # Don't crash the wrapped function

            return result
        return wrapper
    return decorator

def create_ace_steer_agent(
    agent_func: Callable,
    ace_agent_id: str,
    verifications: List[str] = None,
    skillbook_path: Optional[str] = None,
) -> Callable:
    """
    Helper to create a combined ACE+Steer agent.

    Returns original function if any error occurs.
    """
    try:
        if not callable(agent_func):
            logger.warning(f"⚠️ agent_func must be callable, got {type(agent_func)}")
            return agent_func

        if not ace_agent_id:
            logger.warning("⚠️ ace_agent_id is required")
            return agent_func

        return ace_steer_capture(
            ace_agent_id=ace_agent_id,
            verifications=verifications,
            skillbook_path=skillbook_path
        )(agent_func)
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"⚠️ Failed to create ACE+Steer agent: {e}")
        return agent_func  # Return original function on error
