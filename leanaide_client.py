"""
LeanAide Async Client

A production-ready asynchronous Python client for the LeanAide server.
This client supports all JSON API tasks with connection pooling, retries,
streaming, and comprehensive error handling.

CAV-NLP Integration: This client now supports CAV-NLP enhanced formalization
for improved mathematical text processing.

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

import aiohttp
from aiohttp import ClientTimeout, ClientSession, ClientError, ClientResponseError

# CAV-NLP Integration - Lazy import to avoid circular dependency
CAV_NLP_AVAILABLE = None  # Will be determined on first use

# Lean availability - always True when client is importable (assumes Lean server available)
# This flag exists for compatibility with the wider codebase
LEAN_AVAILABLE = True

def _get_cav_nlp_available() -> bool:
    """Check CAV-NLP availability lazily to avoid circular imports."""
    global CAV_NLP_AVAILABLE
    if CAV_NLP_AVAILABLE is None:
        try:
            from openevolve.unified_math_service import UnifiedMathService, FormalizationResult
            CAV_NLP_AVAILABLE = True
        except ImportError:
            CAV_NLP_AVAILABLE = False
            logging.getLogger(__name__).debug("CAV-NLP not available for leanaide_client")
    return CAV_NLP_AVAILABLE


# Configure logging
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of all supported LeanAide task types."""
    TRANSLATE_THM = "translate_thm"
    TRANSLATE_THM_DETAILED = "translate_thm_detailed"
    TRANSLATE_DEF = "translate_def"
    THEOREM_DOC = "theorem_doc"
    DEF_DOC = "def_doc"
    THEOREM_NAME = "theorem_name"
    PROVE_FOR_FORMALIZATION = "prove_for_formalization"
    JSON_STRUCTURED = "json_structured"
    LEAN_FROM_JSON_STRUCTURED = "lean_from_json_structured"
    ELABORATE = "elaborate"
    MATH_QUERY = "math_query"


@dataclass
class LeanAideResult:
    """Structured result from a LeanAide task execution."""
    success: bool
    task: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    response_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "task": self.task,
            "data": self.data,
            "error": self.error,
            "logs": self.logs,
            "response_time": self.response_time,
            "timestamp": self.timestamp
        }


@dataclass
class LeanAideConfig:
    """Configuration for the LeanAide client."""
    host: str = "localhost"
    port: int = 7654
    timeout: float = 6000.0  # Default server timeout
    connect_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_connections: int = 100
    enable_logging: bool = True
    verify_ssl: bool = False  # For HTTPS connections
    enable_caching: bool = False  # Enable response caching
    cache_ttl_seconds: int = 3600  # Cache TTL in seconds
    cache_dir: str = ".leanaide_cache"  # Cache directory path
    max_cache_size_mb: int = 500  # Maximum cache size in MB

    @property
    def base_url(self) -> str:
        """Get the base URL for the server."""
        protocol = "https" if self.verify_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"


class LeanAideClientError(Exception):
    """Base exception for LeanAide client errors."""
    pass


class ConnectionError(LeanAideClientError):
    """Raised when connection to server fails."""
    pass


class TimeoutError(LeanAideClientError):
    """Raised when a request times out."""
    pass


class TaskExecutionError(LeanAideClientError):
    """Raised when a task execution fails on the server."""
    pass


class ValidationError(LeanAideClientError):
    """Raised when request validation fails."""
    pass


class LeanAideClient:
    """
    Production-ready async client for LeanAide server.

    Features:
    - Connection pooling with configurable limits
    - Automatic retries with exponential backoff
    - Comprehensive error handling
    - Request/response logging
    - Health checks
    - Support for all LeanAide JSON API tasks
    - Chained task execution
    - CAV-NLP enhanced formalization (when enabled)
    """

    def __init__(
        self,
        config: Optional[LeanAideConfig] = None,
        session: Optional[ClientSession] = None,
        use_cav_nlp: bool = True
    ):
        """
        Initialize the LeanAide client.

        Args:
            config: Client configuration (uses defaults if not provided)
            session: Optional existing aiohttp session (creates new one if not provided)
            use_cav_nlp: Whether to enable CAV-NLP enhanced formalization
        """
        self.config = config or LeanAideConfig()
        self._session = session
        self._owned_session = session is None
        self._closed = False
        
        # CAV-NLP Integration
        self.use_cav_nlp = use_cav_nlp and _get_cav_nlp_available()
        self._math_service = None
        if self.use_cav_nlp:
            try:
                from openevolve.unified_math_service import UnifiedMathService
                self._math_service = UnifiedMathService(use_cav_nlp=True, use_leanaide=True)
                logger.info("CAV-NLP enhanced formalization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP math service: {e}")
                self.use_cav_nlp = False

        if self.config.enable_logging:
            # Note: We don't call logging.basicConfig here as it's a global setting
            # The application should configure logging. We just ensure our logger level is set.
            logger.setLevel(logging.INFO)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @property
    def session(self) -> ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(
                total=self.config.timeout,
                connect=self.config.connect_timeout
            )
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                ssl=self.config.verify_ssl
            )
            self._session = ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self._session

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if not self._closed:
            if self._owned_session and self._session and not self._session.closed:
                await self._session.close()
            self._closed = True
    
    # ========== CAV-NLP Enhanced Methods ==========
    
    async def formalize_with_cav_nlp(
        self,
        text: str,
        use_cav_nlp: Optional[bool] = None
    ) -> LeanAideResult:
        """
        Formalize natural language text with optional CAV-NLP enhancement.
        
        Uses CAV-NLP (Computer Algebra Verification + NLP) for enhanced
        mathematical formalization when available and enabled.
        
        Args:
            text: Natural language mathematical statement
            use_cav_nlp: Override to use CAV-NLP (None uses instance default)
            
        Returns:
            LeanAideResult with formalized Lean code
        """
        should_use_cav_nlp = use_cav_nlp if use_cav_nlp is not None else self.use_cav_nlp
        
        if should_use_cav_nlp and self._math_service:
            try:
                logger.info("Using CAV-NLP enhanced formalization")
                result = await self._math_service.formalize(text)
                
                return LeanAideResult(
                    success=result.success,
                    task="formalize_with_cav_nlp",
                    data={
                        "result": result.code,
                        "source": result.source,
                        "canonical_form": result.canonical_form,
                        "elaborated_code": result.elaborated_code,
                        "metadata": result.metadata
                    },
                    response_time=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                logger.warning(f"CAV-NLP formalization failed: {e}, falling back to standard")
        
        # Fallback to standard translate_thm
        return await self.translate_thm(text)
    
    async def verify_with_cav_nlp(
        self,
        lean_code: str,
        use_cav_nlp: Optional[bool] = None
    ) -> LeanAideResult:
        """
        Verify Lean code with optional CAV-NLP enhancement.
        
        Args:
            lean_code: Lean 4 code to verify
            use_cav_nlp: Override to use CAV-NLP verification
            
        Returns:
            LeanAideResult with verification status
        """
        should_use_cav_nlp = use_cav_nlp if use_cav_nlp is not None else self.use_cav_nlp
        
        if should_use_cav_nlp and self._math_service:
            try:
                logger.info("Using CAV-NLP enhanced verification")
                result = await self._math_service.verify(lean_code)
                
                return LeanAideResult(
                    success=result.success,
                    task="verify_with_cav_nlp",
                    data={
                        "verified": result.success,
                        "source": "cav_nlp",
                        "metadata": result.metadata if hasattr(result, 'metadata') else {}
                    },
                    response_time=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                logger.warning(f"CAV-NLP verification failed: {e}, falling back to standard")
        
        # Fallback to standard elaborate
        return await self.elaborate(lean_code)

    async def health_check(self) -> bool:
        """
        Check if the LeanAide server is healthy and responding.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            url = f"{self.config.base_url}/"
            async with self.session.get(url) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def _execute_request(
        self,
        payload: Dict[str, Any],
        endpoint: str = ""
    ) -> LeanAideResult:
        """
        Execute a request to the LeanAide server with retry logic.

        Args:
            payload: JSON payload to send
            endpoint: Optional endpoint path (default: root)

        Returns:
            LeanAideResult containing the response or error

        Raises:
            ConnectionError: If connection fails after retries
            TimeoutError: If request times out
        """
        url = f"{self.config.base_url}/{endpoint}"
        task_name = payload.get("task", "unknown")
        start_time = datetime.now(timezone.utc)

        logger.info(f"Executing task: {task_name}")

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    # Calculate response time
                    response_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()

                    # Handle different response statuses
                    if response.status == 200:
                        data = await response.json()

                        # Check if server returned an error
                        if "error" in data:
                            return LeanAideResult(
                                success=False,
                                task=task_name,
                                error=data["error"],
                                logs=data.get("logs"),
                                response_time=response_time
                            )

                        # Successful response
                        return LeanAideResult(
                            success=True,
                            task=task_name,
                            data=data,
                            logs=data.get("logs"),
                            response_time=response_time
                        )

                    elif response.status == 400:
                        error_data = await response.json()
                        raise ValidationError(
                            error_data.get("error", "Invalid request")
                        )

                    elif response.status == 504:
                        raise TimeoutError(
                            f"Server timeout after {self.config.timeout}s"
                        )

                    elif response.status == 500:
                        error_data = await response.json()
                        raise TaskExecutionError(
                            error_data.get("error", "Server error")
                        )

                    else:
                        raise TaskExecutionError(
                            f"Unexpected status code: {response.status}"
                        )

            except ValidationError as e:
                # Don't retry validation errors
                logger.error(f"Validation error for task {task_name}: {e}")
                return LeanAideResult(
                    success=False,
                    task=task_name,
                    error=str(e),
                    response_time=response_time
                )

            except (TimeoutError, TaskExecutionError) as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Task {task_name} failed after {attempt + 1} attempts: {e}"
                    )
                    return LeanAideResult(
                        success=False,
                        task=task_name,
                        error=str(e),
                        response_time=response_time
                    )
                # Retry with exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.info(
                    f"Retry {attempt + 1}/{self.config.max_retries} "
                    f"after {delay}s delay"
                )
                await asyncio.sleep(delay)

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.error(f"Unexpected error for task {task_name}: {e}")
                return LeanAideResult(
                    success=False,
                    task=task_name,
                    error=str(e),
                    response_time=response_time
                )

        return LeanAideResult(
            success=False,
            task=task_name,
            error="Max retries exceeded",
            response_time=response_time
        )

    # ========== Task-specific Methods ==========

    async def translate_thm(
        self,
        theorem_text: str
    ) -> LeanAideResult:
        """
        Translate a natural-language theorem into Lean and elaborate its type.

        Args:
            theorem_text: Natural-language statement of the theorem

        Returns:
            LeanAideResult with elaborated theorem type or errors
        """
        payload = {
            "task": TaskType.TRANSLATE_THM.value,
            "theorem_text": theorem_text
        }
        return await self._execute_request(payload)

    async def translate_thm_detailed(
        self,
        theorem_text: str,
        theorem_name: Optional[str] = None
    ) -> LeanAideResult:
        """
        Translate a theorem with optional name and produce Lean declaration.

        Args:
            theorem_text: Natural-language statement
            theorem_name: Optional name to assign to the theorem

        Returns:
            LeanAideResult with name, type, and command syntax
        """
        payload = {
            "task": TaskType.TRANSLATE_THM_DETAILED.value,
            "theorem_text": theorem_text
        }
        if theorem_name:
            payload["theorem_name"] = theorem_name
        return await self._execute_request(payload)

    async def translate_def(
        self,
        definition_text: str
    ) -> LeanAideResult:
        """
        Translate a natural-language definition into Lean code.

        Args:
            definition_text: Natural-language definition

        Returns:
            LeanAideResult with Lean definition command or errors
        """
        payload = {
            "task": TaskType.TRANSLATE_DEF.value,
            "definition_text": definition_text
        }
        return await self._execute_request(payload)

    async def theorem_doc(
        self,
        theorem_name: str,
        theorem_statement: str
    ) -> LeanAideResult:
        """
        Generate natural-language documentation for a theorem.

        Args:
            theorem_name: Name of the theorem
            theorem_statement: Lean syntax of the theorem statement

        Returns:
            LeanAideResult with documentation string
        """
        payload = {
            "task": TaskType.THEOREM_DOC.value,
            "theorem_name": theorem_name,
            "theorem_statement": theorem_statement
        }
        return await self._execute_request(payload)

    async def def_doc(
        self,
        definition_name: str,
        definition_code: str
    ) -> LeanAideResult:
        """
        Generate natural-language documentation for a definition.

        Args:
            definition_name: Name of the definition
            definition_code: Lean syntax of the definition

        Returns:
            LeanAideResult with documentation string
        """
        payload = {
            "task": TaskType.DEF_DOC.value,
            "definition_name": definition_name,
            "definition_code": definition_code
        }
        return await self._execute_request(payload)

    async def theorem_name(
        self,
        theorem_text: str
    ) -> LeanAideResult:
        """
        Generate a Lean Prover name for a theorem.

        Args:
            theorem_text: Natural-language statement

        Returns:
            LeanAideResult with generated theorem name
        """
        payload = {
            "task": TaskType.THEOREM_NAME.value,
            "theorem_text": theorem_text
        }
        return await self._execute_request(payload)

    async def prove_for_formalization(
        self,
        theorem_text: str,
        theorem_code: str,
        theorem_statement: str
    ) -> LeanAideResult:
        """
        Generate a detailed proof or proof sketch for a theorem.

        Args:
            theorem_text: Natural-language theorem
            theorem_code: Elaborated theorem type
            theorem_statement: Full Lean statement

        Returns:
            LeanAideResult with proof document
        """
        payload = {
            "task": TaskType.PROVE_FOR_FORMALIZATION.value,
            "theorem_text": theorem_text,
            "theorem_code": theorem_code,
            "theorem_statement": theorem_statement
        }
        return await self._execute_request(payload)

    async def json_structured(
        self,
        document_text: str
    ) -> LeanAideResult:
        """
        Convert natural-language document into structured JSON representation.

        Args:
            document_text: Natural-language math text

        Returns:
            LeanAideResult with structured JSON
        """
        payload = {
            "task": TaskType.JSON_STRUCTURED.value,
            "document_text": document_text
        }
        return await self._execute_request(payload)

    async def lean_from_json_structured(
        self,
        document_json: Union[str, Dict[str, Any]]
    ) -> LeanAideResult:
        """
        Generate Lean code from structured JSON.

        Args:
            document_json: Structured JSON of a document

        Returns:
            LeanAideResult with Lean command sequence
        """
        # Convert dict to JSON string if needed
        json_str = (
            json.dumps(document_json)
            if isinstance(document_json, dict)
            else document_json
        )

        payload = {
            "task": TaskType.LEAN_FROM_JSON_STRUCTURED.value,
            "document_json": json_str
        }
        return await self._execute_request(payload)

    async def elaborate(
        self,
        document_code: str
    ) -> LeanAideResult:
        """
        Elaborate Lean code and collect results, logs, and unsolved goals.

        Args:
            document_code: Lean code (as text)

        Returns:
            LeanAideResult with declarations, logs, and unsolved goals
        """
        payload = {
            "task": TaskType.ELABORATE.value,
            "document_code": document_code
        }
        return await self._execute_request(payload)

    async def math_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        n: int = 3
    ) -> LeanAideResult:
        """
        Answer a math question in natural language.

        Args:
            query: Math question
            history: Optional conversation context (list of chat pairs)
            n: Number of answers to generate (default: 3)

        Returns:
            LeanAideResult with list of candidate answers
        """
        payload = {
            "task": TaskType.MATH_QUERY.value,
            "query": query,
            "n": n
        }
        if history:
            payload["history"] = history
        return await self._execute_request(payload)

    # ========== Chained Task Execution ==========

    async def execute_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[LeanAideResult]:
        """
        Execute a chain of tasks, where output of each is merged into input of next.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of LeanAideResult for each task
        """
        payload = {"tasks": tasks}
        return await self._execute_request(payload)

    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[LeanAideResult]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of LeanAideResult for each task
        """
        coroutines = [
            self._execute_request(task) for task in tasks
        ]
        return await asyncio.gather(*coroutines)

    # ========== Streaming Support ==========

    async def execute_streaming(
        self,
        payload: Dict[str, Any],
        chunk_handler: callable
    ) -> LeanAideResult:
        """
        Execute a task with streaming response support.

        Note: This is a placeholder for future streaming support.
        The current LeanAide server doesn't support streaming responses.

        Args:
            payload: Task payload
            chunk_handler: Callback function for handling chunks

        Returns:
            LeanAideResult with complete response
        """
        logger.warning("Streaming not yet supported by LeanAide server")
        return await self._execute_request(payload)

    # ========== Batch Operations ==========

    async def batch_translate_theorems(
        self,
        theorems: List[str]
    ) -> List[LeanAideResult]:
        """
        Translate multiple theorems in parallel.

        Args:
            theorems: List of natural-language theorem statements

        Returns:
            List of LeanAideResult for each theorem
        """
        tasks = [
            {
                "task": TaskType.TRANSLATE_THM.value,
                "theorem_text": thm
            }
            for thm in theorems
        ]
        return await self.execute_parallel_tasks(tasks)

    async def batch_translate_definitions(
        self,
        definitions: List[str]
    ) -> List[LeanAideResult]:
        """
        Translate multiple definitions in parallel.

        Args:
            definitions: List of natural-language definitions

        Returns:
            List of LeanAideResult for each definition
        """
        tasks = [
            {
                "task": TaskType.TRANSLATE_DEF.value,
                "definition_text": definition
            }
            for definition in definitions
        ]
        return await self.execute_parallel_tasks(tasks)


# ========== Utility Functions ==========

async def create_client(
    host: str = "localhost",
    port: int = 7654,
    **kwargs
) -> LeanAideClient:
    """
    Factory function to create and initialize a LeanAide client.

    Args:
        host: Server host
        port: Server port
        **kwargs: Additional configuration options

    Returns:
        Initialized LeanAideClient instance
    """
    config = LeanAideConfig(host=host, port=port, **kwargs)
    return LeanAideClient(config=config)


# ========== Example Usage ==========

async def main():
    """Example usage of the LeanAide client."""
    # Create client
    async with LeanAideClient() as client:
        # Check server health
        is_healthy = await client.health_check()
        print(f"Server healthy: {is_healthy}")

        if not is_healthy:
            print("Server is not responding. Exiting.")
            return

        # Example 1: Translate a simple theorem
        print("\n=== Example 1: Translate Theorem ===")
        result = await client.translate_thm(
            "There are infinitely many prime numbers"
        )
        print(f"Success: {result.success}")
        print(f"Response time: {result.response_time:.2f}s")
        if result.data:
            print(f"Data: {json.dumps(result.data, indent=2)[:200]}...")
        if result.error:
            print(f"Error: {result.error}")

        # Example 2: Translate with detailed output
        print("\n=== Example 2: Translate Theorem Detailed ===")
        result = await client.translate_thm_detailed(
            "There are infinitely many prime numbers",
            theorem_name="infinitely_many_primes"
        )
        print(f"Success: {result.success}")
        if result.data:
            print(f"Data: {json.dumps(result.data, indent=2)[:200]}...")

        # Example 3: Generate theorem documentation
        print("\n=== Example 3: Generate Theorem Documentation ===")
        result = await client.theorem_doc(
            theorem_name="infinitely_many_primes",
            theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
        )
        print(f"Success: {result.success}")
        if result.data:
            print(f"Documentation: {result.data.get('result', 'N/A')[:200]}...")

        # Example 4: Batch translation
        print("\n=== Example 4: Batch Translation ===")
        theorems = [
            "There are infinitely many primes",
            "The square root of 2 is irrational",
            "Every natural number has a unique prime factorization"
        ]
        results = await client.batch_translate_theorems(theorems)
        print(f"Translated {len(results)} theorems")
        for i, result in enumerate(results):
            print(f"  {i+1}. Success: {result.success}, Time: {result.response_time:.2f}s")

        # Example 5: Math query
        print("\n=== Example 5: Math Query ===")
        result = await client.math_query(
            "What is the fundamental theorem of algebra?",
            n=2
        )
        print(f"Success: {result.success}")
        if result.data:
            print(f"Answers: {json.dumps(result.data, indent=2)[:300]}...")

        # Example 6: Elaborate Lean code
        print("\n=== Example 6: Elaborate Lean Code ===")
        lean_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]
"""
        result = await client.elaborate(lean_code)
        print(f"Success: {result.success}")
        if result.data:
            print(f"Elaboration result: {json.dumps(result.data, indent=2)[:200]}...")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
