"""
Steer MCP Tools for CREWAI Agents

This module provides Model Context Protocol (MCP) tools that CREWAI agents
can use to leverage Steer's reliability layer for AI agent outputs.

CRITICAL ARCHITECTURE:
    CREWAI (Orchestrator) → Agent Function → Steer Reality Locks → Verified Output

Steer provides deterministic verification of probabilistic LLM outputs through:
- JSON structure validation
- PII/safety checking
- Logic/ambiguity detection
- Brand voice filtering (slop detection)
- Citation verification
- SQL security enforcement
- Custom regex patterns

Architecture:
    CREWAI Agent → MCP Tool → Steer Judge → Verification Result → Block/Pass
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import json
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)

# Try to import Steer components
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
    STEER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Steer not available: {e}")
    STEER_AVAILABLE = False

    # Create stub classes for when Steer is not available
    class RealityLock:
        pass

    class VerificationResult:
        def __init__(self, Judge_name, passed, reason="", suggested_fixes=None):
            self.Judge_name = Judge_name
            self.passed = passed
            self.reason = reason
            self.suggested_fixes = suggested_fixes or []

    class TeachingOption:
        def __init__(self, title, description="", recommended=False, logic_change=""):
            self.title = title
            self.description = description
            self.recommended = recommended
            self.logic_change = logic_change


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered Steer MCP tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered Steer MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# STEER VERIFICATION TOOLS
# =============================================================================

@mcp_tool("verify_json_output")
def verify_json_output(
    output: Any,
    allow_markdown: bool = False,
) -> Dict[str, Any]:
    """
    Verify that agent output is valid JSON.

    This is used by CREWAI agents to validate structured outputs.

    Args:
        output: The agent output to verify
        allow_markdown: Whether to allow Markdown code blocks around JSON

    Returns:
        Dict with verification result:
        {
            "passed": bool,
            "reason": str,
            "suggested_fixes": List[Dict],
            "judge": str
        }
    """
    logger.info("Verifying JSON output...")

    if not STEER_AVAILABLE:
        return {
            "passed": True,
            "reason": "Steer not available - skipping verification",
            "suggested_fixes": [],
            "judge": "JsonJudge",
        }

    try:
        judge = JsonJudge(name="JsonJudge")

        # Modify behavior based on allow_markdown flag
        if isinstance(output, str) and allow_markdown:
            # Strip markdown if allowed
            output = output.strip()
            if output.startswith("```"):
                # Extract JSON from markdown
                lines = output.split('\n')
                if lines[0].startswith('```'):
                    json_start = 1 if 'json' in lines[0].lower() else 0
                    json_text = '\n'.join(lines[json_start:])
                    # Find closing backticks
                    if '```' in json_text:
                        json_text = json_text[:json_text.rindex('```')]
                    output = json_text.strip()

        result = judge.verify({}, output)

        return {
            "passed": result.passed,
            "reason": result.reason or "JSON is valid",
            "suggested_fixes": [
                {
                    "title": fix.title,
                    "description": fix.description,
                    "recommended": fix.recommended,
                    "logic_change": fix.logic_change,
                }
                for fix in (result.suggested_fixes or [])
            ],
            "judge": result.Judge_name,
        }

    except Exception as e:
        logger.error(f"JSON verification failed: {e}")
        return {
            "passed": False,
            "reason": f"Verification error: {str(e)}",
            "suggested_fixes": [],
            "judge": "JsonJudge",
        }


@mcp_tool("verify_slop_filter")
def verify_slop_filter(
    output: Any,
    entropy_threshold: float = 3.5,
    block_emojis: bool = True,
    block_ai_phrases: bool = True,
) -> Dict[str, Any]:
    """
    Verify that agent output doesn't contain "AI slop" - low-entropy,
    sycophantic language that pollutes data protocols.

    This is used by CREWAI agents to maintain brand voice quality.

    Args:
        output: The agent output to verify
        entropy_threshold: Shannon entropy threshold (lower = more slop)
        block_emojis: Whether to block emojis
        block_ai_phrases: Whether to block common AI phrases

    Returns:
        Dict with verification result
    """
    logger.info(f"Verifying slop filter (threshold={entropy_threshold})...")

    if not STEER_AVAILABLE:
        return {
            "passed": True,
            "reason": "Steer not available - skipping verification",
            "suggested_fixes": [],
            "judge": "SlopJudge",
        }

    try:
        # Create custom slop judge with configured settings
        class CustomSlopJudge(RealityLock):
            def __init__(self, threshold, block_emojis, block_ai_phrases):
                self.name = "SlopJudge"
                self.entropy_threshold = threshold
                self.slop_patterns = [
                    r"i apologize for",
                    r"as an ai",
                    r"delve into",
                    r"embark on",
                    r"it is important to note",
                    r"comprehensive guide",
                    r"revolutionary",
                    r"seamlessly",
                    r"unlock the potential",
                    r"tapestry of",
                ] if block_ai_phrases else []

            def verify(self, inputs, output):
                text_raw = str(output)
                text_lower = text_raw.lower()

                # 1. Emoji Check
                if block_emojis:
                    if any(char for char in text_raw if char in "🚀🤖🧠✨⚡️🎯💡"):
                        return VerificationResult(
                            Judge_name=self.name,
                            passed=False,
                            reason="Detected emoji slop.",
                            suggested_fixes=[
                                TeachingOption(
                                    title="Remove Emojis",
                                    description="Professional output should not contain emojis.",
                                    recommended=True,
                                    logic_change="PROTOCOL: No emojis in output."
                                )
                            ]
                        )

                # 2. Em-dash Check
                if "—" in text_raw:
                    return VerificationResult(
                        Judge_name=self.name,
                        passed=False,
                        reason="Detected em dash formatting slop.",
                        suggested_fixes=[
                            TeachingOption(
                                title="Use Standard Dashes",
                                description="Use standard hyphens instead of em dashes.",
                                recommended=True,
                                logic_change="PROTOCOL: Use standard hyphens (-) not em dashes (—)."
                            )
                        ]
                    )

                # 3. AI Phrase Patterns
                for pattern in self.slop_patterns:
                    if re.search(pattern, text_lower):
                        return VerificationResult(
                            Judge_name=self.name,
                            passed=False,
                            reason=f"Detected AI linguistic fingerprint: '{pattern}'",
                            suggested_fixes=[
                                TeachingOption(
                                    title="Remove AI Phrases",
                                    description=f"Avoid the phrase '{pattern}'",
                                    recommended=True,
                                    logic_change="PROTOCOL: Use direct, concise language. No filler phrases."
                                )
                            ]
                        )

                # 4. Shannon Entropy Check
                if len(text_raw) > 60:
                    counts = Counter(text_raw)
                    total = len(text_raw)
                    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
                    if entropy < self.entropy_threshold:
                        return VerificationResult(
                            Judge_name=self.name,
                            passed=False,
                            reason=f"Low entropy detected ({entropy:.2f}). Signal is too predictable.",
                            suggested_fixes=[
                                TeachingOption(
                                    title="Increase Entropy",
                                    description="Use more varied, human-like language.",
                                    recommended=True,
                                    logic_change="PROTOCOL: Use high-entropy, technical prose. Be concise and direct."
                                )
                            ]
                        )

                return VerificationResult(Judge_name=self.name, passed=True)

        judge = CustomSlopJudge(entropy_threshold, block_emojis, block_ai_phrases)
        result = judge.verify({}, output)

        return {
            "passed": result.passed,
            "reason": result.reason or "Output passes slop filter",
            "suggested_fixes": [
                {
                    "title": fix.title,
                    "description": fix.description,
                    "recommended": fix.recommended,
                    "logic_change": fix.logic_change,
                }
                for fix in (result.suggested_fixes or [])
            ],
            "judge": result.Judge_name,
        }

    except Exception as e:
        logger.error(f"Slop verification failed: {e}")
        return {
            "passed": False,
            "reason": f"Verification error: {str(e)}",
            "suggested_fixes": [],
            "judge": "SlopJudge",
        }


@mcp_tool("verify_pii_safety")
def verify_pii_safety(
    output: Any,
    patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify that agent output doesn't contain PII or sensitive information.

    This is used by CREWAI agents for safety compliance.

    Args:
        output: The agent output to verify
        patterns: Custom regex patterns to block (uses defaults if None)

    Returns:
        Dict with verification result
    """
    logger.info("Verifying PII safety...")

    if not STEER_AVAILABLE:
        return {
            "passed": True,
            "reason": "Steer not available - skipping verification",
            "suggested_fixes": [],
            "judge": "PIIJudge",
        }

    try:
        # Default PII patterns
        default_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S+',  # API keys/secrets
        ]

        detection_patterns = patterns if patterns else default_patterns

        text = str(output)
        detected_patterns = []

        for pattern in detection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_patterns.append(pattern)

        passed = len(detected_patterns) == 0

        if not passed:
            fixes = [
                {
                    "title": "Redact Sensitive Info",
                    "description": f"Detected {len(detected_patterns)} sensitive pattern(s)",
                    "recommended": True,
                    "logic_change": "SECURITY OVERRIDE: You must REDACT all sensitive information with '[REDACTED]'."
                }
            ]
        else:
            fixes = []

        return {
            "passed": passed,
            "reason": f"Detected {len(detected_patterns)} sensitive patterns" if not passed else "No PII detected",
            "suggested_fixes": fixes,
            "detected_patterns": detected_patterns,
            "judge": "PIIJudge",
        }

    except Exception as e:
        logger.error(f"PII verification failed: {e}")
        return {
            "passed": False,
            "reason": f"Verification error: {str(e)}",
            "suggested_fixes": [],
            "judge": "PIIJudge",
        }


@mcp_tool("verify_citations")
def verify_citations(
    output: Any,
    required: bool = True,
    pattern: str = r"\[(doc\s?)?\d+\]",
) -> Dict[str, Any]:
    """
    Verify that agent output includes required source citations.

    This is used by CREWAI agents for RAG grounding verification.

    Args:
        output: The agent output to verify
        required: Whether citations are required
        pattern: Regex pattern for citations

    Returns:
        Dict with verification result
    """
    logger.info("Verifying citations...")

    if not STEER_AVAILABLE:
        return {
            "passed": True,
            "reason": "Steer not available - skipping verification",
            "suggested_fixes": [],
            "judge": "CitationJudge",
        }

    try:
        if not required:
            return {
                "passed": True,
                "reason": "Citations not required",
                "suggested_fixes": [],
                "judge": "CitationJudge",
            }

        text = str(output)
        has_citations = bool(re.search(pattern, text))

        if has_citations:
            return {
                "passed": True,
                "reason": "Citations present",
                "suggested_fixes": [],
                "judge": "CitationJudge",
            }
        else:
            return {
                "passed": False,
                "reason": "Output missing required source citations",
                "suggested_fixes": [
                    {
                        "title": "Require Citations",
                        "description": "Every factual claim must have a citation",
                        "recommended": True,
                        "logic_change": "GROUNDING RULE: Every factual claim must be followed by a citation in brackets, e.g., [doc 1]. If the context does not contain the answer, state that you do not know."
                    }
                ],
                "judge": "CitationJudge",
            }

    except Exception as e:
        logger.error(f"Citation verification failed: {e}")
        return {
            "passed": False,
            "reason": f"Verification error: {str(e)}",
            "suggested_fixes": [],
            "judge": "CitationJudge",
        }


@mcp_tool("verify_sql_security")
def verify_sql_security(
    output: Any,
    allow_select_only: bool = True,
    forbidden_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify that SQL output doesn't contain destructive commands.

    This is used by CREWAI agents for SQL security enforcement.

    Args:
        output: The SQL query to verify
        allow_select_only: Whether to only allow SELECT queries
        forbidden_patterns: Custom forbidden patterns

    Returns:
        Dict with verification result
    """
    logger.info("Verifying SQL security...")

    if not STEER_AVAILABLE:
        return {
            "passed": True,
            "reason": "Steer not available - skipping verification",
            "suggested_fixes": [],
            "judge": "SqlJudge",
        }

    try:
        query = str(output).lower()

        if allow_select_only:
            # Check if query starts with SELECT
            if not query.strip().startswith('select'):
                return {
                    "passed": False,
                    "reason": "Only SELECT queries are allowed",
                    "suggested_fixes": [
                        {
                            "title": "Read-Only Mode",
                            "description": "Force agent to only use SELECT statements",
                            "recommended": True,
                            "logic_change": "PROTOCOL: SELECT only. Deny all other SQL commands."
                        }
                    ],
                    "judge": "SqlJudge",
                }

        # Check for forbidden patterns
        default_forbidden = [r"drop\s+table", r"delete\s+from", r"truncate", r"insert\s+into"]
        forbidden = forbidden_patterns if forbidden_patterns else default_forbidden

        for pattern in forbidden:
            if re.search(pattern, query):
                return {
                    "passed": False,
                    "reason": f"Forbidden SQL command detected: {pattern}",
                    "suggested_fixes": [
                        {
                            "title": "Read-Only Mode",
                            "description": f"Block {pattern} commands",
                            "recommended": True,
                            "logic_change": f"PROTOCOL: Never use {pattern}. SELECT only."
                        }
                    ],
                    "judge": "SqlJudge",
                }

        return {
            "passed": True,
            "reason": "SQL is safe",
            "suggested_fixes": [],
            "judge": "SqlJudge",
        }

    except Exception as e:
        logger.error(f"SQL verification failed: {e}")
        return {
            "passed": False,
            "reason": f"Verification error: {str(e)}",
            "suggested_fixes": [],
            "judge": "SqlJudge",
        }


@mcp_tool("run_all_verifications")
def run_all_verifications(
    output: Any,
    verifications: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Run multiple Steer verifications on agent output.

    This is used by CREWAI agents to run comprehensive verification checks.

    Args:
        output: The agent output to verify
        verifications: List of verification names to run
        **kwargs: Additional parameters for specific verifications

    Returns:
        Dict with all verification results:
        {
            "all_passed": bool,
            "results": List[Dict],
            "failed_verifications": List[str]
        }
    """
    logger.info(f"Running {len(verifications)} verifications...")

    verification_functions = {
        "json": verify_json_output,
        "slop": verify_slop_filter,
        "pii": verify_pii_safety,
        "citations": verify_citations,
        "sql": verify_sql_security,
    }

    results = []
    failed_verifications = []

    for verification in verifications:
        if verification not in verification_functions:
            logger.warning(f"Unknown verification: {verification}")
            continue

        func = verification_functions[verification]

        # Extract kwargs for this verification
        prefix = f"{verification}_"
        relevant_kwargs = {
            k[len(prefix):]: v
            for k, v in kwargs.items()
            if k.startswith(prefix)
        }

        try:
            result = func(output=output, **relevant_kwargs)
            results.append(result)

            if not result.get("passed", False):
                failed_verifications.append(verification)

        except Exception as e:
            logger.error(f"Verification {verification} failed: {e}")
            results.append({
                "verification": verification,
                "passed": False,
                "reason": f"Error: {str(e)}",
            })
            failed_verifications.append(verification)

    return {
        "all_passed": len(failed_verifications) == 0,
        "results": results,
        "failed_verifications": failed_verifications,
        "total_verifications": len(verifications),
        "passed_count": len(verifications) - len(failed_verifications),
    }


@mcp_tool("get_steer_status")
def get_steer_status() -> Dict[str, Any]:
    """Get the status of the Steer reliability layer"""
    return {
        "available": STEER_AVAILABLE,
        "components": {
            "capture": STEER_AVAILABLE,
            "judges": STEER_AVAILABLE,
            "verification": STEER_AVAILABLE,
        },
        "available_verifications": [
            "json",
            "slop",
            "pii",
            "citations",
            "sql",
        ] if STEER_AVAILABLE else [],
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all Steer MCP tools"""
    logger.info("Initializing Steer MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} Steer MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()
