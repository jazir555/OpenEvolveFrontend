"""
Steer - Hephaestus Bridge

This module provides the bridge between Hephaestus workflow phases and
Steer's reliability layer for AI agent outputs.

IMPORTANT: Steer is NOT a workflow or decomposition system.
Steer provides deterministic verification (Reality Locks) for probabilistic LLM outputs.

Phase Mapping:
- Phase 1-6: All phases can use Steer judges for output verification
- Pre-execution: Verify inputs before processing
- Post-execution: Verify outputs before returning to user
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from steer_mcp_tools import (
    verify_json_output,
    verify_slop_filter,
    verify_pii_safety,
    verify_citations,
    verify_sql_security,
    run_all_verifications,
    get_steer_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STEER WRAPPER DECORATORS
# =============================================================================

def steer_capture(
    verifications: List[str] = None,
    halt_on_failure: bool = True,
    **verification_kwargs,
):
    """
    Decorator that wraps Hephaestus agent functions with Steer verification.

    This is analogous to Steer's @capture decorator but designed for
    Hephaestus agent workflow integration.

    Args:
        verifications: List of verification types to run ("json", "slop", "pii", "citations", "sql")
        halt_on_failure: Whether to raise exception if verification fails
        **verification_kwargs: Additional parameters for specific verifications

    Example:
        @steer_capture(verifications=["json", "slop"])
        def my_hephaestus_agent(input_data):
            return {"result": "processed data"}
    """
    if verifications is None:
        verifications = []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Execute the original function
            result = func(*args, **kwargs)

            # Run verifications on output
            if verifications:
                logger.info(f"Running Steer verifications: {verifications}")
                verification_result = run_all_verifications(
                    output=result,
                    verifications=verifications,
                    **verification_kwargs,
                )

                # Log results
                if not verification_result["all_passed"]:
                    logger.warning(
                        f"Steer verifications failed: {verification_result['failed_verifications']}"
                    )
                    for result_item in verification_result["results"]:
                        if not result_item.get("passed", False):
                            logger.warning(
                                f"  [{result_item['judge']}] {result_item.get('reason', 'Unknown reason')}"
                            )
                            for fix in result_item.get("suggested_fixes", []):
                                logger.info(f"    Suggested: {fix['title']}")

                    if halt_on_failure:
                        raise SteerVerificationError(
                            f"Output verification failed: {', '.join(verification_result['failed_verifications'])}",
                            result=result,
                            verification_results=verification_result,
                        )
                else:
                    logger.info("All Steer verifications passed")

                # Attach verification results to output
                if isinstance(result, dict):
                    result["_steer_verification"] = verification_result

            return result

        return wrapper
    return decorator


class SteerVerificationError(Exception):
    """Exception raised when Steer verification fails"""
    def __init__(self, message, result=None, verification_results=None):
        super().__init__(message)
        self.result = result
        self.verification_results = verification_results


# =============================================================================
# PHASE VERIFICATION FUNCTIONS
# =============================================================================

def verify_phase_output(
    phase_id: int,
    output: Any,
    verifications: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Verify the output of a Hephaestus phase using Steer judges.

    This is called by Hephaestus phase execution to validate outputs.

    Args:
        phase_id: The phase number (1-6)
        output: The output to verify
        verifications: List of verification types to run
        **kwargs: Additional parameters for specific verifications

    Returns:
        Dict with verification results
    """
    logger.info(f"Verifying Phase {phase_id} output with Steer...")

    result = run_all_verifications(
        output=output,
        verifications=verifications,
        **kwargs,
    )

    result["phase_id"] = phase_id
    result["timestamp"] = None  # Could add timestamp if needed

    return result


def verify_phase_1_setup_output(
    output: Any,
    verify_json: bool = True,
    verify_slop: bool = True,
) -> Dict[str, Any]:
    """
    Verify Phase 1 (Problem Setup) output.

    Phase 1 typically produces structured analysis and decomposition plans.
    """
    verifications = []
    if verify_json:
        verifications.append("json")
    if verify_slop:
        verifications.append("slop")

    return verify_phase_output(1, output, verifications)


def verify_phase_2_solution_output(
    output: Any,
    verify_json: bool = True,
    verify_slop: bool = True,
    verify_pii: bool = False,
) -> Dict[str, Any]:
    """
    Verify Phase 2 (Solution Generation) output.

    Phase 2 produces solution code/text that should be well-structured
    and maintain brand voice quality.
    """
    verifications = []
    if verify_json:
        verifications.append("json")
    if verify_slop:
        verifications.append("slop")
    if verify_pii:
        verifications.append("pii")

    return verify_phase_output(2, output, verifications)


def verify_phase_3_critique_output(
    output: Any,
    verify_slop: bool = True,
    verify_pii: bool = False,
) -> Dict[str, Any]:
    """
    Verify Phase 3 (Adversarial Critique) output.

    Critiques should be direct and high-quality (no slop).
    """
    verifications = []
    if verify_slop:
        verifications.append("slop")
    if verify_pii:
        verifications.append("pii")

    return verify_phase_output(3, output, verifications)


def verify_phase_4_verification_output(
    output: Any,
    verify_json: bool = True,
    verify_citations: bool = True,
) -> Dict[str, Any]:
    """
    Verify Phase 4 (Verification) output.

    Verification results should be structured and properly cited.
    """
    verifications = []
    if verify_json:
        verifications.append("json")
    if verify_citations:
        verifications.append("citations")

    return verify_phase_output(4, output, verifications)


def verify_phase_5_reassembly_output(
    output: Any,
    verify_json: bool = True,
    verify_slop: bool = True,
) -> Dict[str, Any]:
    """
    Verify Phase 5 (Reassembly) output.

    Reassembled output should be structured and high-quality.
    """
    verifications = []
    if verify_json:
        verifications.append("json")
    if verify_slop:
        verifications.append("slop")

    return verify_phase_output(5, output, verifications)


def verify_phase_6_final_output(
    output: Any,
    verify_json: bool = True,
    verify_slop: bool = True,
    verify_citations: bool = True,
) -> Dict[str, Any]:
    """
    Verify Phase 6 (Final Validation) output.

    Final output should meet all quality standards.
    """
    verifications = []
    if verify_json:
        verifications.append("json")
    if verify_slop:
        verifications.append("slop")
    if verify_citations:
        verifications.append("citations")

    return verify_phase_output(6, output, verifications)


# =============================================================================
# WORKFLOW BRIDGE CLASS
# =============================================================================

class SteerHephaestusWorkflowBridge:
    """
    Main bridge class that integrates Steer verification with Hephaestus workflows.

    This class provides a high-level interface that Hephaestus can use
    to add reliability verification to agent outputs.
    """

    def __init__(self):
        self.phase_verifiers = {
            1: verify_phase_1_setup_output,
            2: verify_phase_2_solution_output,
            3: verify_phase_3_critique_output,
            4: verify_phase_4_verification_output,
            5: verify_phase_5_reassembly_output,
            6: verify_phase_6_final_output,
        }

        # Default verifications per phase
        self.default_verifications = {
            1: ["json", "slop"],
            2: ["json", "slop"],
            3: ["slop"],
            4: ["json", "citations"],
            5: ["json", "slop"],
            6: ["json", "slop", "citations"],
        }

    def verify_phase(
        self,
        phase_id: int,
        output: Any,
        verifications: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Verify output from a specific phase.

        Args:
            phase_id: Phase to verify (1-6)
            output: Output to verify
            verifications: List of verification types (uses defaults if None)
            **kwargs: Additional parameters for verifications

        Returns:
            Dict with verification results
        """
        if phase_id not in self.phase_verifiers:
            return {
                "phase_id": phase_id,
                "all_passed": False,
                "error": f"Invalid phase ID: {phase_id}",
            }

        # Use default verifications if not specified
        if verifications is None:
            verifications = self.default_verifications.get(phase_id, [])

        # Run verification
        return verify_phase_output(phase_id, output, verifications, **kwargs)

    def wrap_agent_function(
        self,
        func: Callable,
        verifications: List[str] = None,
        halt_on_failure: bool = True,
    ) -> Callable:
        """
        Wrap an agent function with Steer verification.

        Args:
            func: The agent function to wrap
            verifications: List of verification types
            halt_on_failure: Whether to raise exception on failure

        Returns:
            Wrapped function with verification
        """
        return steer_capture(
            verifications=verifications,
            halt_on_failure=halt_on_failure,
        )(func)

    def get_verification_status(self) -> Dict[str, Any]:
        """Get Steer system status"""
        return get_steer_status()

    def list_available_verifications(self) -> List[str]:
        """List available verification types"""
        status = get_steer_status()
        return status.get("available_verifications", [])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_verified_agent(
    agent_func: Callable,
    verifications: List[str] = None,
    phase_id: Optional[int] = None,
) -> Callable:
    """
    Create a verified agent function wrapped with Steer.

    Args:
        agent_func: The base agent function
        verifications: List of verification types (uses phase defaults if None)
        phase_id: If specified, uses default verifications for that phase

    Returns:
        Wrapped agent function
    """
    if phase_id is not None and verifications is None:
        bridge = SteerHephaestusWorkflowBridge()
        verifications = bridge.default_verifications.get(phase_id, [])

    return steer_capture(verifications=verifications or [])(agent_func)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_workflow_bridge():
    """Initialize the Steer-Hephaestus workflow bridge"""
    status = get_steer_status()
    logger.info(f"Steer-Hephaestus workflow bridge initialized (available: {status['available']})")


# Auto-initialize on import
initialize_workflow_bridge()
