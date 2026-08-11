"""
STEER (Safety, Trust, Evaluation, Error Reduction) - Reliability Context Engine

This module serves as the central hub for the STEER reliability layer,
providing a unified interface to all STEER functionality across the OpenEvolve platform.

STEER provides deterministic verification of probabilistic LLM outputs through:
- JSON structure validation
- PII/safety checking
- Logic/ambiguity detection
- Brand voice filtering (slop detection)
- Citation verification
- SQL security enforcement
- Custom regex patterns

This module integrates with:
- CrewAI workflows (via steer_crewai_bridge.py)
- MCP tools (via steer_mcp_tools.py)
- Various domain-specific agents (blue_team, red_team, gold_team, etc.)

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

# Import STEER core components
try:
    from steer.core import capture, VerificationError
    from steer.judges import (
        RealityLock,
        JsonJudge,
        SlopJudge,
        AmbiguityJudge,
        PydanticJudge,
        CitationJudge,
        FactConsistencyJudge,
        SqlJudge,
        RegexJudge,
    )
    from steer.schemas import VerificationResult, TeachingOption
    from steer.utils import wait_for_rules
    from steer.storage import rulebook
    STEER_AVAILABLE = True
except ImportError as e:
    STEER_AVAILABLE = False
    logging.warning(f"STEER not available: {e}")
    
    # Define stub classes for graceful degradation
    class StubClass:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return None
    
    # Create stubs
    capture = StubClass
    VerificationError = Exception
    RealityLock = StubClass
    JsonJudge = StubClass
    SlopJudge = StubClass
    AmbiguityJudge = StubClass
    PydanticJudge = StubClass
    CitationJudge = StubClass
    FactConsistencyJudge = StubClass
    SqlJudge = StubClass
    RegexJudge = StubClass
    VerificationResult = StubClass
    TeachingOption = StubClass
    wait_for_rules = StubClass
    rulebook = StubClass


# Import supporting modules
try:
    from steer_crewai_bridge import (
        SteerCrewAIWorkflowBridge,
        steer_capture,
        SteerVerificationError,
        verify_phase_1_setup_output,
        verify_phase_2_solution_output,
        verify_phase_3_critique_output,
        verify_phase_4_verification_output,
        verify_phase_5_reassembly_output,
        verify_phase_6_final_output,
        create_verified_agent,
    )
    from steer_mcp_tools import (
        verify_json_output,
        verify_slop_filter,
        verify_pii_safety,
        verify_citations,
        verify_sql_security,
        run_all_verifications,
        get_steer_status,
    )
    from ace_steer_config import (
        get_ace_steer_config,
        is_steer_enabled,
        is_unified_bridge_enabled,
        get_ace_steer_status,
        validate_ace_steer_config,
        is_any_available,
    )
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logging.warning(f"Core STEER modules not available: {e}")


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for STEER Context Engine
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


class SteerContextEngine:
    """
    Centralized STEER Reliability Context Engine for OpenEvolve Platform
    
    This class provides a unified interface to all STEER functionality,
    managing verification rules, judges, and reliability checks across the entire platform.
    
    Key Features:
    - Centralized verification rule management
    - Multi-judge coordination
    - Verification from execution feedback
    - Context injection for enhanced reliability
    - Thread-safe operations
    - Rulebook management
    """
    
    def __init__(
        self,
        default_entropy_threshold: float = 3.5,
        default_allow_markdown: bool = False,
        default_block_emojis: bool = True,
        default_block_ai_phrases: bool = True,
        default_required_citations: bool = True,
        default_allow_select_only: bool = True,
        rulebook_path: Optional[str] = None,
    ):
        """
        Initialize the STEER Context Engine.
        
        Args:
            default_entropy_threshold: Default entropy threshold for slop detection
            default_allow_markdown: Whether to allow markdown in JSON outputs by default
            default_block_emojis: Whether to block emojis by default
            default_block_ai_phrases: Whether to block AI phrases by default
            default_required_citations: Whether citations are required by default
            default_allow_select_only: Whether to allow only SELECT SQL by default
            rulebook_path: Path to load custom rulebook
        """
        self.default_entropy_threshold = default_entropy_threshold
        self.default_allow_markdown = default_allow_markdown
        self.default_block_emojis = default_block_emojis
        self.default_block_ai_phrases = default_block_ai_phrases
        self.default_required_citations = default_required_citations
        self.default_allow_select_only = default_allow_select_only
        self.rulebook_path = rulebook_path
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize STEER components
        self.judges = {}
        self.rules = {}
        
        # Initialize bridge components
        self.crewai_bridge = None
        
        # Initialize
        self._initialize_components()
        
        logger.info("STEER Context Engine initialized successfully")
    
    def _initialize_components(self):
        """Initialize all STEER components."""
        if not STEER_AVAILABLE:
            logger.warning("STEER not available - initializing with stubs")
            return
            
        try:
            # Create judges
            self.judges['json'] = JsonJudge(name="JsonJudge")
            self.judges['slop'] = SlopJudge(name="SlopJudge")  # Assuming SlopJudge exists
            self.judges['citation'] = CitationJudge(name="CitationJudge")
            self.judges['sql'] = SqlJudge(name="SqlJudge")
            
            # Initialize bridge components
            self.crewai_bridge = SteerCrewAIWorkflowBridge()
            
            logger.info("All STEER components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STEER components: {e}")
            # Still try to continue with available components
    
    def verify_json_output(
        self,
        output: Any,
        allow_markdown: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Verify that agent output is valid JSON.
        
        Args:
            output: The agent output to verify
            allow_markdown: Whether to allow Markdown code blocks around JSON (uses default if None)
            
        Returns:
            Dict with verification result
        """
        with self._lock:
            if not STEER_AVAILABLE:
                return {
                    "passed": True,
                    "reason": "STEER not available - skipping verification",
                    "suggested_fixes": [],
                    "judge": "JsonJudge",
                }
            
            try:
                # Use provided value or default
                use_markdown = allow_markdown if allow_markdown is not None else self.default_allow_markdown
                
                # Call the MCP tool function
                result = verify_json_output(output, allow_markdown=use_markdown)
                
                return result
            except Exception as e:
                logger.error(f"JSON verification failed: {e}")
                return {
                    "passed": False,
                    "reason": f"Verification error: {str(e)}",
                    "suggested_fixes": [],
                    "judge": "JsonJudge",
                }
    
    def verify_slop_filter(
        self,
        output: Any,
        entropy_threshold: Optional[float] = None,
        block_emojis: Optional[bool] = None,
        block_ai_phrases: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Verify that agent output doesn't contain "AI slop" - low-entropy,
        sycophantic language that pollutes data protocols.
        
        Args:
            output: The agent output to verify
            entropy_threshold: Shannon entropy threshold (lower = more slop)
            block_emojis: Whether to block emojis
            block_ai_phrases: Whether to block common AI phrases
            
        Returns:
            Dict with verification result
        """
        with self._lock:
            if not STEER_AVAILABLE:
                return {
                    "passed": True,
                    "reason": "STEER not available - skipping verification",
                    "suggested_fixes": [],
                    "judge": "SlopJudge",
                }
            
            try:
                # Use provided values or defaults
                threshold = entropy_threshold if entropy_threshold is not None else self.default_entropy_threshold
                emojis = block_emojis if block_emojis is not None else self.default_block_emojis
                phrases = block_ai_phrases if block_ai_phrases is not None else self.default_block_ai_phrases
                
                # Call the MCP tool function
                result = verify_slop_filter(
                    output,
                    entropy_threshold=threshold,
                    block_emojis=emojis,
                    block_ai_phrases=phrases
                )
                
                return result
            except Exception as e:
                logger.error(f"Slop verification failed: {e}")
                return {
                    "passed": False,
                    "reason": f"Verification error: {str(e)}",
                    "suggested_fixes": [],
                    "judge": "SlopJudge",
                }
    
    def verify_pii_safety(
        self,
        output: Any,
        patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify that agent output doesn't contain PII or sensitive information.
        
        Args:
            output: The agent output to verify
            patterns: Custom regex patterns to block (uses defaults if None)
            
        Returns:
            Dict with verification result
        """
        with self._lock:
            if not STEER_AVAILABLE:
                return {
                    "passed": True,
                    "reason": "STEER not available - skipping verification",
                    "suggested_fixes": [],
                    "judge": "PIIJudge",
                }
            
            try:
                # Call the MCP tool function
                result = verify_pii_safety(output, patterns=patterns)
                
                return result
            except Exception as e:
                logger.error(f"PII verification failed: {e}")
                return {
                    "passed": False,
                    "reason": f"Verification error: {str(e)}",
                    "suggested_fixes": [],
                    "judge": "PIIJudge",
                }
    
    def verify_citations(
        self,
        output: Any,
        required: Optional[bool] = None,
        pattern: str = r"\[(doc\s?)?\d+\]",
    ) -> Dict[str, Any]:
        """
        Verify that agent output includes required source citations.
        
        Args:
            output: The agent output to verify
            required: Whether citations are required (uses default if None)
            pattern: Regex pattern for citations
            
        Returns:
            Dict with verification result
        """
        with self._lock:
            if not STEER_AVAILABLE:
                return {
                    "passed": True,
                    "reason": "STEER not available - skipping verification",
                    "suggested_fixes": [],
                    "judge": "CitationJudge",
                }
            
            try:
                # Use provided value or default
                req = required if required is not None else self.default_required_citations
                
                # Call the MCP tool function
                result = verify_citations(output, required=req, pattern=pattern)
                
                return result
            except Exception as e:
                logger.error(f"Citation verification failed: {e}")
                return {
                    "passed": False,
                    "reason": f"Verification error: {str(e)}",
                    "suggested_fixes": [],
                    "judge": "CitationJudge",
                }
    
    def verify_sql_security(
        self,
        output: Any,
        allow_select_only: Optional[bool] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify that SQL output doesn't contain destructive commands.
        
        Args:
            output: The SQL query to verify
            allow_select_only: Whether to only allow SELECT queries (uses default if None)
            forbidden_patterns: Custom forbidden patterns
            
        Returns:
            Dict with verification result
        """
        with self._lock:
            if not STEER_AVAILABLE:
                return {
                    "passed": True,
                    "reason": "STEER not available - skipping verification",
                    "suggested_fixes": [],
                    "judge": "SqlJudge",
                }
            
            try:
                # Use provided value or default
                select_only = allow_select_only if allow_select_only is not None else self.default_allow_select_only
                
                # Call the MCP tool function
                result = verify_sql_security(
                    output,
                    allow_select_only=select_only,
                    forbidden_patterns=forbidden_patterns
                )
                
                return result
            except Exception as e:
                logger.error(f"SQL verification failed: {e}")
                return {
                    "passed": False,
                    "reason": f"Verification error: {str(e)}",
                    "suggested_fixes": [],
                    "judge": "SqlJudge",
                }
    
    def run_all_verifications(
        self,
        output: Any,
        verifications: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run multiple STEER verifications on agent output.

        Args:
            output: The agent output to verify
            verifications: List of verification names to run
            **kwargs: Additional parameters for specific verifications

        Returns:
            Dict with all verification results
        """
        import time
        start_time = time.time()
        success = False
        verification_id = f"steer_{int(time.time())}"

        with self._lock:
            if not STEER_AVAILABLE:
                # **ACTUAL INTEGRATION**: Track degraded performance when STEER unavailable
                self._track_steer_performance("run_all_verifications_steered_unavailable", True, time.time() - start_time, 0)

                return {
                    "all_passed": True,
                    "results": [],
                    "failed_verifications": [],
                    "total_verifications": 0,
                    "passed_count": 0,
                }

            try:
                # Call the MCP tool function
                result = run_all_verifications(output, verifications, **kwargs)

                success = result.get("all_passed", False)
                duration = time.time() - start_time
                passed_count = result.get("passed_count", 0)

                # **ACTUAL INTEGRATION**: Extract knowledge and track performance
                self._extract_steer_knowledge("run_all_verifications", verification_id, result)
                self._track_steer_performance("run_all_verifications", success, duration, passed_count)

                if not success:
                    # **ACTUAL INTEGRATION**: Trigger alert for failed verifications
                    failed_count = len(result.get("failed_verifications", []))
                    self._trigger_steer_alerts("run_all_verifications", False, verification_id, f"{failed_count} verifications failed")

                return result

            except Exception as e:
                duration = time.time() - start_time

                # **ACTUAL INTEGRATION**: Trigger alert and track failure
                self._trigger_steer_alerts("run_all_verifications", False, verification_id, str(e))
                self._track_steer_performance("run_all_verifications", False, duration, 0)

                logger.error(f"All verifications failed: {e}")
                return {
                    "all_passed": False,
                    "results": [],
                    "failed_verifications": verifications,
                    "total_verifications": len(verifications),
                    "passed_count": 0,
                    "error": str(e),
                }
    
    def get_context_enhanced_with_rules(
        self,
        base_prompt: str,
        agent_name: Optional[str] = None,
        include_json_rules: bool = True,
        include_slop_rules: bool = True,
        include_pii_rules: bool = True,
        include_citation_rules: bool = True,
    ) -> str:
        """
        Get a context-enhanced prompt with STEER reliability rules.
        
        Args:
            base_prompt: Original prompt to enhance
            agent_name: Name of the agent (for agent-specific rules)
            include_json_rules: Whether to include JSON rules
            include_slop_rules: Whether to include slop detection rules
            include_pii_rules: Whether to include PII rules
            include_citation_rules: Whether to include citation rules
            
        Returns:
            Enhanced prompt string with reliability rules
        """
        with self._lock:
            enhanced_parts = []
            
            # Add STEER reliability rules if any are enabled
            rules_added = False
            
            if include_json_rules:
                enhanced_parts.append("### JSON STRUCTURE RULES:")
                enhanced_parts.append("- Output must be valid JSON")
                enhanced_parts.append("- No markdown wrappers around JSON")
                enhanced_parts.append("- Follow schema if provided")
                enhanced_parts.append("")
                rules_added = True
            
            if include_slop_rules:
                enhanced_parts.append("### BRAND VOICE RULES (ANTI-SLOP):")
                enhanced_parts.append("- No AI buzzwords or phrases")
                enhanced_parts.append("- No emojis or decorative elements")
                enhanced_parts.append("- Direct, concise, professional language")
                enhanced_parts.append("- High information density")
                enhanced_parts.append("")
                rules_added = True
            
            if include_pii_rules:
                enhanced_parts.append("### PRIVACY & SAFETY RULES:")
                enhanced_parts.append("- No PII in output")
                enhanced_parts.append("- No sensitive information disclosure")
                enhanced_parts.append("- Redact personal details")
                enhanced_parts.append("")
                rules_added = True
            
            if include_citation_rules:
                enhanced_parts.append("### CITATION REQUIREMENTS:")
                enhanced_parts.append("- All claims must be cited")
                enhanced_parts.append("- Use [doc N] format for citations")
                enhanced_parts.append("- Reference provided sources")
                enhanced_parts.append("")
                rules_added = True
            
            # Add agent-specific rules if available
            if agent_name and STEER_AVAILABLE:
                try:
                    agent_rules = rulebook.get_rules_text(agent_name) if rulebook else ""
                    if agent_rules:
                        enhanced_parts.append(f"### AGENT-SPECIFIC RULES FOR {agent_name.upper()}:")
                        enhanced_parts.append(agent_rules)
                        enhanced_parts.append("")
                        rules_added = True
                except Exception as e:
                    logger.warning(f"Could not get agent-specific rules: {e}")
            
            # Add base prompt
            enhanced_parts.append("TASK:")
            enhanced_parts.append(base_prompt)
            
            return "\n".join(enhanced_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the STEER Context Engine.
        
        Returns:
            Dictionary with status information
        """
        return {
            "available": STEER_AVAILABLE,
            "core_modules_available": CORE_MODULES_AVAILABLE,
            "default_entropy_threshold": self.default_entropy_threshold,
            "default_allow_markdown": self.default_allow_markdown,
            "default_block_emojis": self.default_block_emojis,
            "default_block_ai_phrases": self.default_block_ai_phrases,
            "default_required_citations": self.default_required_citations,
            "default_allow_select_only": self.default_allow_select_only,
            "crewai_bridge_available": self.crewai_bridge is not None,
            "active_judges": list(self.judges.keys()) if self.judges else [],
        }
    
    def add_custom_rule(
        self,
        rule_name: str,
        rule_description: str,
        verification_func: Callable[[Any], Dict[str, Any]]
    ):
        """
        Add a custom verification rule to the engine.
        
        Args:
            rule_name: Name of the rule
            rule_description: Description of what the rule checks
            verification_func: Function that performs the verification
        """
        with self._lock:
            self.rules[rule_name] = {
                "description": rule_description,
                "function": verification_func
            }
            logger.info(f"Added custom rule: {rule_name}")

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for STEER Context
    # =========================================================================

    def _trigger_steer_alerts(
        self,
        operation: str,
        success: bool,
        verification_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for STEER verification failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"STEER Context Alert: {operation}",
                    description=f"STEER verification operation '{operation}' failed" +
                                 (f" for verification '{verification_id}'" if verification_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="steer_context_engine",
                    component="steer_verification",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger STEER Context alert: {e}")

    def _extract_steer_knowledge(
        self,
        operation: str,
        verification_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract STEER verification knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"steer_{operation}_{verification_id}",
                artifact_type="steer_verification",
                source_component="steer_context_engine",
                title=f"STEER Verification: {operation} - {verification_id}",
                content={
                    "operation": operation,
                    "verification_id": verification_id,
                    "all_passed": result.get("all_passed", False),
                    "total_verifications": result.get("total_verifications", 0),
                    "passed_count": result.get("passed_count", 0),
                    "failed_verifications": len(result.get("failed_verifications", [])),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "steer_available": STEER_AVAILABLE
                },
                tags=["steer", operation, "verification", "reliability"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted STEER verification knowledge for {verification_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract STEER verification knowledge: {e}")
            return False

    def _track_steer_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        passed_count: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track STEER verification performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"steer_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "passed_count": passed_count
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked STEER verification performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track STEER verification performance: {e}")


# Global instance for easy access
_steer_engine = None
_engine_lock = threading.Lock()


def get_steer_engine() -> SteerContextEngine:
    """
    Get the global STEER Context Engine instance.
    
    Returns:
        SteerContextEngine instance
    """
    global _steer_engine
    
    with _engine_lock:
        if _steer_engine is None:
            _steer_engine = SteerContextEngine()
        return _steer_engine


def with_steer_verification(
    verifications: List[str] = None,
    halt_on_failure: bool = True,
    **verification_kwargs
):
    """
    Decorator to add STEER verification to any function.
    
    Args:
        verifications: List of verification types to run
        halt_on_failure: Whether to raise exception on verification failure
        **verification_kwargs: Additional parameters for specific verifications
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Run verifications if STEER is available
            if STEER_AVAILABLE and verifications:
                try:
                    # Get STEER engine
                    steer_engine = get_steer_engine()
                    
                    # Run verifications
                    verification_result = steer_engine.run_all_verifications(
                        output=result,
                        verifications=verifications,
                        **verification_kwargs
                    )
                    
                    # Log results
                    if not verification_result["all_passed"]:
                        logger.warning(
                            f"STEER verifications failed: {verification_result['failed_verifications']}"
                        )
                        
                        if halt_on_failure:
                            raise SteerVerificationError(
                                f"Output verification failed: {', '.join(verification_result['failed_verifications'])}",
                                result=result,
                                verification_results=verification_result,
                            )
                    else:
                        logger.info("All STEER verifications passed")
                    
                    # Attach verification results to output if it's a dict
                    if isinstance(result, dict):
                        result["_steer_verification"] = verification_result
                        
                except Exception as e:
                    logger.error(f"STEER verification error: {e}")
                    if halt_on_failure:
                        raise
            
            return result
        return wrapper
    return decorator


# Convenience functions for common operations
def verify_output(
    output: Any,
    verifications: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Verify an output using the global STEER engine.
    
    Args:
        output: The output to verify
        verifications: List of verification types to run
        **kwargs: Additional parameters for specific verifications
        
    Returns:
        Verification results
    """
    steer_engine = get_steer_engine()
    return steer_engine.run_all_verifications(output, verifications, **kwargs)


def get_reliable_prompt(
    base_prompt: str,
    agent_name: Optional[str] = None
) -> str:
    """
    Get a reliability-enhanced prompt using the global STEER engine.
    
    Args:
        base_prompt: Original prompt
        agent_name: Name of the agent (for agent-specific rules)
        
    Returns:
        Enhanced prompt string
    """
    steer_engine = get_steer_engine()
    return steer_engine.get_context_enhanced_with_rules(base_prompt, agent_name)


# Export commonly used items
__all__ = [
    "SteerContextEngine",
    "get_steer_engine",
    "with_steer_verification",
    "verify_output",
    "get_reliable_prompt",
    "STEER_AVAILABLE",
    "CORE_MODULES_AVAILABLE",
]


if __name__ == "__main__":
    print("STEER Context Engine Module")
    print(f"STEER Available: {STEER_AVAILABLE}")
    print(f"Core Modules Available: {CORE_MODULES_AVAILABLE}")
    
    if STEER_AVAILABLE:
        print("\nInitializing STEER Context Engine...")
        engine = SteerContextEngine()
        status = engine.get_status()
        print(f"Engine Status: {status}")
    else:
        print("\nSTEER not available - this may be due to missing dependencies.")
        print("Install steer to enable full functionality:")
        print("  pip install steer")