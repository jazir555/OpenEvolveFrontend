"""
LeanAide MCP Tools for CREWAI Agents

This module provides Model Context Protocol (MCP) tools that enable CREWAI
agents to leverage LeanAide's AI-powered formal mathematics capabilities.

LeanAide provides:
    - Autoformalization: Natural language to Lean theorem translation
    - Proof generation: Automated proof creation and completion
    - Code verification: Lean code elaboration and error checking
    - Documentation: Natural language documentation for formal code
    - Math Q&A: Answering mathematical questions
    - Definition translation: Natural language to Lean definition conversion

Architecture: CREWAI (Orchestrator) -> LeanAide MCP Tools -> LeanAide Server -> Lean Theorem Prover
"""

import asyncio
import json
import logging
import os
import copy
import socket
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from functools import wraps
import urllib.request
import urllib.parse
import urllib.error

# Security imports
from ace_security_utils import (
    validate_string_length,
    validate_numeric_range,
    validate_model_name,
    create_safe_error,
    sanitize_for_logging,
    get_global_lock,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_LEANAIDE_HOST = os.environ.get("LEANAIDE_HOST", "localhost")
DEFAULT_LEANAIDE_PORT = int(os.environ.get("LEANAIDE_PORT", 7654))
DEFAULT_TIMEOUT = int(os.environ.get("LEANAIDE_TIMEOUT", 120))  # 2 minutes

# Thread-safe MCP tool registry
_MCP_TOOLS = {}
_MCP_TOOLS_LOCK = get_global_lock('leanaide_mcp_tools_registry')


# ============================================================================
# MCP Tool Registry
# ============================================================================

def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool (thread-safe)."""
    def decorator(func):
        # Register immediately when decorator is applied
        with _MCP_TOOLS_LOCK:
            _MCP_TOOLS[name] = func
        logger.info(f"Registered LeanAide MCP tool: {name}")
        return func
    return decorator


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools."""
    with _MCP_TOOLS_LOCK:
        return list(_MCP_TOOLS.keys())


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name."""
    with _MCP_TOOLS_LOCK:
        return _MCP_TOOLS.get(name)


# ============================================================================
# LeanAide Client
# ============================================================================

class LeanAideClientError(Exception):
    """Base exception for LeanAide client errors."""
    pass


class LeanAideConnectionError(LeanAideClientError):
    """Connection error to LeanAide server."""
    pass


class LeanAideTimeoutError(LeanAideClientError):
    """Timeout waiting for LeanAide response."""
    pass


class LeanAideClient:
    """
    Client for interacting with LeanAide server.

    The LeanAide server provides a JSON API for translating natural language
    mathematics to Lean code and generating proofs.
    """

    def __init__(
        self,
        host: str = DEFAULT_LEANAIDE_HOST,
        port: int = DEFAULT_LEANAIDE_PORT,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize LeanAide client.

        Args:
            host: LeanAide server hostname or IP
            port: LeanAide server port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def _send_request(
        self,
        task_data: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a request to the LeanAide server.

        Args:
            task_data: Task dictionary with 'task' field and required inputs
            timeout: Optional override timeout

        Returns:
            Response dictionary from LeanAide

        Raises:
            LeanAideConnectionError: If connection fails
            LeanAideTimeoutError: If request times out
            LeanAideClientError: For other errors
        """
        timeout = timeout or self.timeout
        url = f"{self.base_url}/"

        try:
            # Prepare request
            data = json.dumps(task_data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'},
            )

            # Send request with timeout
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')

            # Parse response
            result = json.loads(response_data)

            # Check for errors in response
            if isinstance(result, dict) and 'error' in result:
                raise LeanAideClientError(result['error'])

            return result

        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                raise LeanAideTimeoutError(
                    f"Request timed out after {timeout}s"
                ) from e
            raise LeanAideConnectionError(
                f"Failed to connect to LeanAide server at {self.base_url}: {e}"
            ) from e
        except socket.timeout:
            raise LeanAideTimeoutError(
                f"Request timed out after {timeout}s"
            )
        except json.JSONDecodeError as e:
            raise LeanAideClientError(
                f"Invalid JSON response from server: {e}"
            ) from e
        except (IOError, ConnectionError, TimeoutError) as e:
            raise LeanAideClientError(
                f"Connection error: {e}"
            ) from e

    def translate_theorem(
        self,
        theorem_text: str,
        theorem_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translate a natural language theorem to Lean code.

        Args:
            theorem_text: Natural language theorem statement
            theorem_name: Optional name for the theorem
            timeout: Request timeout

        Returns:
            Dict with translation results including Lean code
        """
        task_data = {
            "task": "translate_thm_detailed" if theorem_name else "translate_thm",
            "theorem_text": theorem_text,
        }

        if theorem_name:
            task_data["theorem_name"] = theorem_name

        return self._send_request(task_data, timeout)

    def translate_definition(
        self,
        definition_text: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translate a natural language definition to Lean code.

        Args:
            definition_text: Natural language definition
            timeout: Request timeout

        Returns:
            Dict with Lean definition code
        """
        task_data = {
            "task": "translate_def",
            "definition_text": definition_text,
        }

        return self._send_request(task_data, timeout)

    def generate_proof(
        self,
        theorem_text: str,
        theorem_code: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a proof for a theorem.

        Args:
            theorem_text: Natural language theorem statement
            theorem_code: Optional pre-translated Lean code
            timeout: Request timeout

        Returns:
            Dict with generated proof
        """
        task_data = {
            "task": "prove_for_formalization",
            "theorem_text": theorem_text,
        }

        if theorem_code:
            task_data["theorem_code"] = theorem_code

        return self._send_request(task_data, timeout)

    def elaborate_code(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Elaborate Lean code and check for errors.

        Args:
            code: Lean code to elaborate
            timeout: Request timeout

        Returns:
            Dict with elaboration results including errors and goals
        """
        task_data = {
            "task": "elaborate",
            "document_code": code,
        }

        return self._send_request(task_data, timeout)

    def math_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        n: int = 3,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a mathematical question.

        Args:
            query: Mathematical question
            history: Optional conversation history
            n: Number of answers to generate
            timeout: Request timeout

        Returns:
            Dict with generated answers
        """
        task_data = {
            "task": "math_query",
            "query": query,
            "n": n,
        }

        if history:
            task_data["history"] = history

        return self._send_request(task_data, timeout)

    def generate_documentation(
        self,
        name: str,
        code: str,
        doc_type: str = "theorem",  # "theorem" or "definition"
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate documentation for Lean code.

        Args:
            name: Name of the theorem or definition
            code: Lean code
            doc_type: Type of code ("theorem" or "definition")
            timeout: Request timeout

        Returns:
            Dict with generated documentation
        """
        if doc_type == "theorem":
            task_data = {
                "task": "theorem_doc",
                "theorem_name": name,
                "theorem_statement": code,
            }
        else:  # definition
            task_data = {
                "task": "def_doc",
                "definition_name": name,
                "definition_code": code,
            }

        return self._send_request(task_data, timeout)

    def verify_solution(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify Lean code correctness by elaborating.

        Args:
            code: Lean code to verify
            timeout: Request timeout

        Returns:
            Dict with verification results
        """
        return self.elaborate_code(code, timeout)


# Global client instance (lazy initialization)
_client: Optional[LeanAideClient] = None
_client_lock = get_global_lock('leanaide_client')


def get_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> LeanAideClient:
    """
    Get or create the global LeanAide client.

    Args:
        host: Optional host override
        port: Optional port override
        timeout: Optional timeout override

    Returns:
        LeanAideClient instance
    """
    global _client

    with _client_lock:
        if _client is None:
            _client = LeanAideClient(
                host=host or DEFAULT_LEANAIDE_HOST,
                port=port or DEFAULT_LEANAIDE_PORT,
                timeout=timeout or DEFAULT_TIMEOUT,
            )
            logger.info(
                f"Created LeanAide client: {_client.base_url} "
                f"(timeout={timeout or DEFAULT_TIMEOUT}s)"
            )
        return _client


# ============================================================================
# MCP Tools
# ============================================================================

@mcp_tool("leanaide_translate_theorem")
def leanaide_translate_theorem(
    theorem_text: str,
    theorem_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Translate a natural language theorem to Lean code.

    This tool uses LeanAide's autoformalization capabilities to convert
    informal mathematical statements into formal Lean theorem declarations.

    Args:
        theorem_text: Natural language theorem statement
                      Example: "There are infinitely many primes"
        theorem_name: Optional name for the theorem (e.g., "infinitely_many_primes")
        host: LeanAide server host (default: from LEANAIDE_HOST env var or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT env var or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - theorem_name: str (generated or provided)
            - lean_code: str (formal Lean theorem declaration)
            - elaborated_type: str (if available)
            - command_syntax: str (full Lean command)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_translate_theorem(
        ...     "The product of two even numbers is even",
        ...     theorem_name="even_product_even"
        ... )
        >>> print(result['lean_code'])
        theorem even_product_even (a b : Nat) (ha : Even a) (hb : Even b) : Even (a * b) := by sorry
    """
    # Validate inputs
    try:
        theorem_text = validate_string_length(
            theorem_text, "theorem_text",
            max_length=5000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid theorem_text", e)

    if theorem_name:
        try:
            theorem_name = validate_string_length(
                theorem_name, "theorem_name",
                max_length=200, allow_empty=False
            )
        except ValueError as e:
            return create_safe_error("Invalid theorem_name", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.translate_theorem(
            theorem_text=theorem_text,
            theorem_name=theorem_name,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "theorem_text": theorem_text,
            "theorem_name": result.get("name") or theorem_name or "unknown",
            "lean_code": result.get("code") or result.get("command", ""),
            "elaborated_type": result.get("type"),
            "command_syntax": result.get("command"),
            "raw_response": result,
            "execution_time": execution_time,
            "message": f"Theorem translated successfully in {execution_time:.2f}s",
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error translating theorem", e)


@mcp_tool("leanaide_translate_definition")
def leanaide_translate_definition(
    definition_text: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Translate a natural language definition to Lean code.

    This tool converts informal mathematical definitions into formal
    Lean definition declarations using autoformalization.

    Args:
        definition_text: Natural language definition
                        Example: "A number is cube-free if it is not divisible
                                  by the cube of any prime number"
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - definition_text: str
            - lean_code: str (formal Lean definition)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_translate_definition(
        ...     "A natural number n is prime if it has exactly two positive divisors"
        ... )
        >>> print(result['lean_code'])
    """
    # Validate inputs
    try:
        definition_text = validate_string_length(
            definition_text, "definition_text",
            max_length=5000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid definition_text", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.translate_definition(
            definition_text=definition_text,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract Lean code from result
        lean_code = ""
        if isinstance(result, dict):
            lean_code = result.get("code") or result.get("command", "")
        elif isinstance(result, str):
            lean_code = result

        return {
            "success": True,
            "definition_text": definition_text,
            "lean_code": lean_code,
            "raw_response": result,
            "execution_time": execution_time,
            "message": f"Definition translated successfully in {execution_time:.2f}s",
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error translating definition", e)


@mcp_tool("leanaide_generate_proof")
def leanaide_generate_proof(
    theorem_text: str,
    theorem_code: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a proof for a theorem.

    This tool uses LeanAide's proof generation capabilities to create
    formal proofs for mathematical theorems.

    Args:
        theorem_text: Natural language theorem statement
        theorem_code: Optional pre-translated Lean code (if not provided,
                     will be auto-translated from theorem_text)
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - theorem_text: str
            - proof_document: str (natural language proof sketch)
            - structured_proof: dict (if available)
            - lean_proof: str (generated Lean proof code, if available)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_generate_proof(
        ...     "The square root of 2 is irrational"
        ... )
        >>> print(result['proof_document'])
    """
    # Validate inputs
    try:
        theorem_text = validate_string_length(
            theorem_text, "theorem_text",
            max_length=5000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid theorem_text", e)

    if theorem_code:
        try:
            theorem_code = validate_string_length(
                theorem_code, "theorem_code",
                max_length=10000
            )
        except ValueError as e:
            return create_safe_error("Invalid theorem_code", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.generate_proof(
            theorem_text=theorem_text,
            theorem_code=theorem_code,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "theorem_text": theorem_text,
            "theorem_code": theorem_code,
            "proof_document": result.get("proof") or result.get("document", ""),
            "structured_proof": result.get("structured"),
            "lean_proof": result.get("code") or result.get("proof_code", ""),
            "raw_response": result,
            "execution_time": execution_time,
            "message": f"Proof generated in {execution_time:.2f}s",
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error generating proof", e)


@mcp_tool("leanaide_verify_solution")
def leanaide_verify_solution(
    code: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify Lean code correctness by elaborating.

    This tool checks Lean code for errors by running it through
    the Lean elaborator, which type-checks the code and identifies
    any issues.

    Args:
        code: Lean code to verify
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - is_valid: bool (True if code elaborates without errors)
            - declarations: List[str] (names of elaborated declarations)
            - logs: List[str] (log messages)
            - sorries: List[dict] (unproven obligations)
            - sorries_after_purge: List[dict] (remaining obligations after simplification)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_verify_solution('''
        ...     theorem add_comm (a b : Nat) : a + b = b + a := by
        ...       simp [add_comm]
        ... ''')
        >>> print(result['is_valid'])
        True
    """
    # Validate inputs
    try:
        code = validate_string_length(
            code, "code",
            max_length=50000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid code", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.elaborate_code(
            code=code,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract elaboration results
        is_valid = True
        declarations = []
        logs = []
        sorries = []
        sorries_after_purge = []

        if isinstance(result, dict):
            declarations = result.get("declarations", [])
            logs = result.get("logs", [])
            sorries = result.get("sorries", [])
            sorries_after_purge = result.get("sorriesAfterPurge", [])

            # Code is valid if no errors and no remaining sorries
            is_valid = (
                len(sorries_after_purge) == 0 and
                not any("error" in log.lower() for log in logs)
            )

        return {
            "success": True,
            "is_valid": is_valid,
            "declarations": declarations,
            "logs": logs,
            "sorries": sorries,
            "sorries_after_purge": sorries_after_purge,
            "unproven_count": len(sorries_after_purge),
            "raw_response": result,
            "execution_time": execution_time,
            "message": (
                f"Code verification complete in {execution_time:.2f}s. "
                f"Valid: {is_valid}, Unproven: {len(sorries_after_purge)}"
            ),
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error verifying solution", e)


@mcp_tool("leanaide_math_query")
def leanaide_math_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    n: int = 3,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Answer a mathematical question.

    This tool uses LeanAide's mathematical knowledge to answer questions
    about mathematics, proofs, and formal verification.

    Args:
        query: Mathematical question
        history: Optional conversation history (list of {role, content} dicts)
        n: Number of answers to generate (default: 3)
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - query: str
            - answers: List[str] (generated answers)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_math_query(
        ...     "What is the fundamental theorem of calculus?"
        ... )
        >>> for i, answer in enumerate(result['answers']):
        ...     print(f"Answer {i+1}: {answer}")
    """
    # Validate inputs
    try:
        query = validate_string_length(
            query, "query",
            max_length=5000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid query", e)

    try:
        n = validate_numeric_range(
            n, "n",
            min_val=1, max_val=10
        )
    except ValueError as e:
        return create_safe_error("Invalid n", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.math_query(
            query=query,
            history=history or [],
            n=n,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract answers
        answers = []
        if isinstance(result, list):
            answers = result
        elif isinstance(result, dict):
            answers = result.get("answers", result.get("results", []))

        return {
            "success": True,
            "query": query,
            "answers": answers[:n],
            "num_answers": len(answers[:n]),
            "raw_response": result,
            "execution_time": execution_time,
            "message": f"Generated {len(answers[:n])} answers in {execution_time:.2f}s",
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error processing math query", e)


@mcp_tool("leanaide_generate_documentation")
def leanaide_generate_documentation(
    name: str,
    code: str,
    doc_type: str = "theorem",
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate documentation for Lean code.

    This tool creates natural language documentation for Lean theorems
    and definitions, explaining their purpose and meaning.

    Args:
        name: Name of the theorem or definition
        code: Lean code
        doc_type: Type of code, either "theorem" or "definition"
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - name: str
            - doc_type: str
            - documentation: str (generated natural language documentation)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_generate_documentation(
        ...     name="infinitely_many_primes",
        ...     code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
        ...     doc_type="theorem"
        ... )
        >>> print(result['documentation'])
    """
    # Validate inputs
    try:
        name = validate_string_length(
            name, "name",
            max_length=200, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid name", e)

    try:
        code = validate_string_length(
            code, "code",
            max_length=10000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid code", e)

    if doc_type not in ["theorem", "definition"]:
        return create_safe_error(
            "Invalid doc_type",
            ValueError("doc_type must be 'theorem' or 'definition'")
        )

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.generate_documentation(
            name=name,
            code=code,
            doc_type=doc_type,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract documentation
        documentation = ""
        if isinstance(result, str):
            documentation = result
        elif isinstance(result, dict):
            documentation = result.get("doc") or result.get("documentation", "")

        return {
            "success": True,
            "name": name,
            "doc_type": doc_type,
            "documentation": documentation,
            "raw_response": result,
            "execution_time": execution_time,
            "message": f"Documentation generated in {execution_time:.2f}s",
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error generating documentation", e)


@mcp_tool("leanaide_elaborate_code")
def leanaide_elaborate_code(
    code: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Elaborate Lean code and check errors.

    This is an alias for leanaide_verify_solution with more descriptive output
    focused on the elaboration process and error reporting.

    Args:
        code: Lean code to elaborate
        host: LeanAide server host (default: from LEANAIDE_HOST or localhost)
        port: LeanAide server port (default: from LEANAIDE_PORT or 7654)
        timeout: Request timeout in seconds (default: from LEANAIDE_TIMEOUT or 120)

    Returns:
        Dict with:
            - success: bool
            - has_errors: bool
            - declarations: List[str]
            - errors: List[str] (error messages)
            - warnings: List[str]
            - goals: List[dict] (unsolved goals)
            - execution_time: float
            - message: str

    Example:
        >>> result = leanaide_elaborate_code('''
        ...     theorem bad_thm (n : Nat) : n = n + 1 := by rfl
        ... ''')
        >>> print(result['has_errors'])
        True
        >>> print(result['errors'])
    """
    # Validate inputs
    try:
        code = validate_string_length(
            code, "code",
            max_length=50000, allow_empty=False
        )
    except ValueError as e:
        return create_safe_error("Invalid code", e)

    if timeout:
        try:
            timeout = validate_numeric_range(
                timeout, "timeout",
                min_val=1, max_val=600
            )
        except ValueError as e:
            return create_safe_error("Invalid timeout", e)

    start_time = datetime.now()

    try:
        client = get_client(host=host, port=port, timeout=timeout)

        result = client.elaborate_code(
            code=code,
            timeout=timeout,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract elaboration results
        declarations = []
        logs = []
        sorries = []

        if isinstance(result, dict):
            declarations = result.get("declarations", [])
            logs = result.get("logs", [])
            sorries = result.get("sorries", [])

        # Classify logs into errors and warnings
        errors = [log for log in logs if "error" in log.lower()]
        warnings = [log for log in logs if "warning" in log.lower()]
        has_errors = len(errors) > 0 or len(sorries) > 0

        return {
            "success": True,
            "has_errors": has_errors,
            "declarations": declarations,
            "logs": logs,
            "errors": errors,
            "warnings": warnings,
            "unsolved_goals": sorries,
            "unsolved_goal_count": len(sorries),
            "raw_response": result,
            "execution_time": execution_time,
            "message": (
                f"Elaboration complete in {execution_time:.2f}s. "
                f"Errors: {len(errors)}, Warnings: {len(warnings)}, "
                f"Unsolved goals: {len(sorries)}"
            ),
            "server": f"{client.host}:{client.port}",
        }

    except LeanAideConnectionError as e:
        logger.error(f"LeanAide connection error: {sanitize_for_logging(e)}")
        return create_safe_error("Failed to connect to LeanAide server", e)
    except LeanAideTimeoutError as e:
        logger.error(f"LeanAide timeout: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request timed out", e)
    except LeanAideClientError as e:
        logger.error(f"LeanAide client error: {sanitize_for_logging(e)}")
        return create_safe_error("LeanAide request failed", e)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Validation error: {sanitize_for_logging(e)}")
        return create_safe_error("Validation error elaborating code", e)


@mcp_tool("get_leanaide_status")
def get_leanaide_status() -> Dict[str, Any]:
    """
    Get LeanAide server connection status.

    Returns:
        Dict with:
            - available: bool (True if server is reachable)
            - host: str
            - port: int
            - timeout: int
            - message: str
    """
    host = DEFAULT_LEANAIDE_HOST
    port = DEFAULT_LEANAIDE_PORT
    timeout = DEFAULT_TIMEOUT

    try:
        # Try to connect with a simple timeout
        client = LeanAideClient(host=host, port=port, timeout=5)

        # We can't easily check status without a valid task,
        # so we just verify the host/port are accessible
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return {
                "available": True,
                "host": host,
                "port": port,
                "timeout": timeout,
                "message": f"LeanAide server is reachable at {host}:{port}",
            }
        else:
            return {
                "available": False,
                "host": host,
                "port": port,
                "timeout": timeout,
                "message": f"LeanAide server is not responding at {host}:{port}",
            }

    except (IOError, ConnectionError, TimeoutError) as e:
        logger.error(f"Connection error checking LeanAide status: {sanitize_for_logging(e)}")
        return {
            "available": False,
            "host": host,
            "port": port,
            "timeout": timeout,
            "error": str(e),
            "message": f"Cannot reach LeanAide server at {host}:{port}",
        }


# ============================================================================
# Async Versions
# ============================================================================

async def leanaide_translate_theorem_async(
    theorem_text: str,
    theorem_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Async version of leanaide_translate_theorem."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        leanaide_translate_theorem,
        theorem_text,
        theorem_name,
        host,
        port,
        timeout,
    )


async def leanaide_translate_definition_async(
    definition_text: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Async version of leanaide_translate_definition."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        leanaide_translate_definition,
        definition_text,
        host,
        port,
        timeout,
    )


async def leanaide_generate_proof_async(
    theorem_text: str,
    theorem_code: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Async version of leanaide_generate_proof."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        leanaide_generate_proof,
        theorem_text,
        theorem_code,
        host,
        port,
        timeout,
    )


async def leanaide_verify_solution_async(
    code: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Async version of leanaide_verify_solution."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        leanaide_verify_solution,
        code,
        host,
        port,
        timeout,
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # MCP Tools
    "leanaide_translate_theorem",
    "leanaide_translate_definition",
    "leanaide_generate_proof",
    "leanaide_verify_solution",
    "leanaide_math_query",
    "leanaide_generate_documentation",
    "leanaide_elaborate_code",
    "get_leanaide_status",
    # Async versions
    "leanaide_translate_theorem_async",
    "leanaide_translate_definition_async",
    "leanaide_generate_proof_async",
    "leanaide_verify_solution_async",
    # Client
    "LeanAideClient",
    "get_client",
    "LeanAideClientError",
    "LeanAideConnectionError",
    "LeanAideTimeoutError",
    # Utilities
    "list_mcp_tools",
    "get_mcp_tool",
]


# ============================================================================
# Module Initialization
# ============================================================================

def initialize_mcp_tools() -> Dict[str, Any]:
    """Initialize all LeanAide MCP tools."""
    logger.info("Initializing LeanAide MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} LeanAide MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()


if __name__ == "__main__":
    print("LeanAide MCP Tools Module")
    print(f"Registered Tools: {len(list_mcp_tools())}")
    print("\nTools:")
    for tool_name in sorted(list_mcp_tools()):
        print(f"  - {tool_name}")

    # Check status
    print("\n" + "="*60)
    print("LeanAide Server Status:")
    print("="*60)
    status = get_leanaide_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
