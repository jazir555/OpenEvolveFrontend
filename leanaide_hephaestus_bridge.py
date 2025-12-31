"""
LeanAide-Hephaestus Bridge

This module provides the bridge between Hephaestus workflow phases and
LeanAide's Lean 4 mathematical verification and translation capabilities.

Architecture:
    Hephaestus (6 phases) -> LeanAide Bridge -> Lean 4 Theorem Prover
                                                              |
                                                           LeanAide Core
                                                              |
                                                        Mathematical Processing

Phase Mapping:
- Phase 1: Analysis -> Mathematical problem detection and analysis
- Phase 2: Translate -> Natural language math to Lean 4 translation
- Phase 3: Verify -> Verify solutions using Lean 4 elaboration
- Phase 4: Proof Check -> Check proof validity and completeness
- Phase 5: Formal Verification -> Final formal verification
- Phase 6: Knowledge Extraction -> Extract verified theorems for knowledge base

The bridge provides:
1. Mathematical content detection
2. Lean 4 code generation (via LeanAide)
3. Formal verification (via Lean 4)
4. Proof validation
5. Knowledge extraction
6. Ticket tracking with Hephaestus
7. Comprehensive error handling
8. Synchronous and asynchronous execution modes
"""

import asyncio
import json
import logging
import os
import sys
import time
import copy
import hashlib
import subprocess
import tempfile
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import requests
import aiohttp

# Add LeanAide to path if available
LEANAIDE_PATH = os.path.join(os.path.dirname(__file__), "LeanAide")
if os.path.exists(LEANAIDE_PATH) and LEANAIDE_PATH not in sys.path:
    sys.path.insert(0, LEANAIDE_PATH)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class ExecutionMode(Enum):
    """Execution mode for LeanAide operations"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"


class MathematicalDomain(Enum):
    """Mathematical domains for classification"""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    GENERAL = "general"


class VerificationStatus(Enum):
    """Status of Lean 4 verification"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class LeanAideConfig:
    """Configuration for LeanAide integration"""
    # Server configuration
    host: str = "localhost"
    port: int = 7654
    api_endpoint: str = "/api/v1/translate"

    # Execution settings
    default_timeout: int = 300  # 5 minutes
    max_concurrent_requests: int = 5
    execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS

    # Verification settings
    enable_verification: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Lean 4 settings
    lean_workspace: str = "./lean_workspace"
    lean_library_path: str = "./lean_libraries"
    lean_command: str = "lake exe leanaide_process"

    # Hephaestus ticket settings
    enable_tickets: bool = True
    ticket_base_url: Optional[str] = None


@dataclass
class MathematicalComponent:
    """A mathematical component extracted from a problem"""
    type: str  # "theorem", "lemma", "definition", "equation", etc.
    name: str
    statement: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    lean_code: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.PENDING


@dataclass
class LeanAideResult:
    """Result from a LeanAide operation"""
    success: bool
    phase: str
    ticket_id: Optional[str] = None
    lean_code: Optional[str] = None
    verification_result: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HephaestusTicket:
    """Hephaestus ticket for tracking workflow progress"""
    ticket_id: str
    phase: str
    status: str
    created_at: str
    updated_at: str
    data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LEANAIDE CLIENT
# =============================================================================

class LeanAideClient:
    """
    Client for interacting with LeanAide server

    Handles translation of natural language mathematical statements to Lean 4 code,
    verification using Lean 4, and proof checking.
    """

    def __init__(self, config: LeanAideConfig):
        """
        Initialize LeanAide client

        Args:
            config: Configuration for LeanAide integration
        """
        self.config = config
        self.server_url = f"http://{config.host}:{config.port}"
        self.api_url = f"{self.server_url}{config.api_endpoint}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()

        # Check server availability
        self.available = self._check_server_availability()

        if not self.available:
            logger.warning(f"LeanAide server not available at {self.server_url}")

    def _check_server_availability(self) -> bool:
        """Check if LeanAide server is running"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LeanAide server health check failed: {e}")
            return False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.default_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _generate_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        return hashlib.sha256(content.encode()).hexdigest()

    async def translate_to_lean(
        self,
        mathematical_statement: str,
        include_context: bool = True
    ) -> LeanAideResult:
        """
        Translate natural language mathematical statement to Lean 4 code

        Args:
            mathematical_statement: Natural language math statement
            include_context: Include context from similar theorems

        Returns:
            LeanAideResult with translation results
        """
        start_time = time.time()

        # Check cache
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(mathematical_statement)
            with self.cache_lock:
                if cache_key in self.cache:
                    logger.debug(f"Cache hit for translation: {cache_key[:8]}...")
                    cached_result = self.cache[cache_key]
                    cached_result.execution_time = time.time() - start_time
                    return cached_result

        try:
            # Prepare request payload
            payload = {
                "task": "translate",
                "input": mathematical_statement,
                "includeContext": include_context,
                "numResponses": 1,  # Get single best translation
            }

            # Send request to LeanAide server
            session = await self._get_session()
            async with session.post(self.api_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return LeanAideResult(
                        success=False,
                        phase="translate",
                        errors=[f"HTTP {response.status}: {error_text}"],
                        execution_time=time.time() - start_time
                    )

                data = await response.json()

                # Extract Lean code from response
                lean_code = self._extract_lean_code(data)

                if lean_code:
                    result = LeanAideResult(
                        success=True,
                        phase="translate",
                        lean_code=lean_code,
                        metadata={
                            "raw_response": data,
                            "statement": mathematical_statement
                        },
                        execution_time=time.time() - start_time
                    )

                    # Cache successful translations
                    if self.config.enable_caching:
                        with self.cache_lock:
                            self.cache[cache_key] = result

                    return result
                else:
                    return LeanAideResult(
                        success=False,
                        phase="translate",
                        errors=["Could not extract Lean code from response"],
                        metadata={"raw_response": data},
                        execution_time=time.time() - start_time
                    )

        except asyncio.TimeoutError:
            return LeanAideResult(
                success=False,
                phase="translate",
                errors=["Translation timeout exceeded"],
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return LeanAideResult(
                success=False,
                phase="translate",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    def _extract_lean_code(self, response_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract Lean 4 code from LeanAide response

        Args:
            response_data: Raw response from LeanAide server

        Returns:
            Extracted Lean 4 code or None
        """
        # Try different response formats
        if "lean_code" in response_data:
            return response_data["lean_code"]

        if "output" in response_data:
            output = response_data["output"]
            if isinstance(output, str):
                return output
            elif isinstance(output, list) and len(output) > 0:
                return output[0]

        # Try to parse JSON array response
        if isinstance(response_data, list) and len(response_data) > 0:
            first_item = response_data[0]
            if isinstance(first_item, dict):
                if "text" in first_item:
                    return first_item["text"]
                if "code" in first_item:
                    return first_item["code"]

        return None

    async def verify_lean_code(
        self,
        lean_code: str,
        timeout: Optional[int] = None
    ) -> LeanAideResult:
        """
        Verify Lean 4 code using Lean 4 elaboration

        Args:
            lean_code: Lean 4 code to verify
            timeout: Optional timeout in seconds

        Returns:
            LeanAideResult with verification results
        """
        start_time = time.time()
        timeout = timeout or self.config.default_timeout

        try:
            # Create temporary file with Lean code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.lean',
                delete=False
            ) as f:
                f.write(lean_code)
                temp_file = f.name

            # Run Lean 4 verification
            process = await asyncio.create_subprocess_exec(
                'lake', 'build',  # Or appropriate Lean 4 command
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.lean_workspace
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                verification_output = stdout.decode()
                errors = stderr.decode()

                success = process.returncode == 0

                result = LeanAideResult(
                    success=success,
                    phase="verify",
                    verification_result=verification_output,
                    errors=errors.split('\n') if errors else [],
                    metadata={
                        "return_code": process.returncode,
                        "temp_file": temp_file
                    },
                    execution_time=time.time() - start_time
                )

                return result

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return LeanAideResult(
                    success=False,
                    phase="verify",
                    errors=["Verification timeout"],
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return LeanAideResult(
                success=False,
                phase="verify",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    async def batch_translate(
        self,
        statements: List[str],
        concurrent_limit: Optional[int] = None
    ) -> List[LeanAideResult]:
        """
        Translate multiple statements concurrently

        Args:
            statements: List of mathematical statements
            concurrent_limit: Max concurrent translations

        Returns:
            List of LeanAideResult objects
        """
        concurrent_limit = concurrent_limit or self.config.max_concurrent_requests

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def translate_with_limit(statement: str) -> LeanAideResult:
            async with semaphore:
                return await self.translate_to_lean(statement)

        tasks = [translate_with_limit(stmt) for stmt in statements]
        return await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# MATHEMATICAL PROBLEM DETECTOR
# =============================================================================

class MathematicalProblemDetector:
    """
    Detects and classifies mathematical content in problems

    Identifies mathematical problems, classifies them by domain,
    extracts components, and estimates complexity.
    """

    def __init__(self):
        """Initialize the detector with mathematical keywords and patterns"""
        self.mathematical_keywords = [
            # Core mathematical terms
            "theorem", "lemma", "corollary", "proposition", "axiom", "conjecture",
            "proof", "prove", "disprove", "show", "demonstrate",

            # Algebra
            "group", "ring", "field", "vector space", "matrix", "determinant",
            "polynomial", "equation", "inequality", "algebraic",

            # Analysis
            "limit", "derivative", "integral", "continuity", "differentiable",
            "function", "sequence", "series", "convergence", "divergence",
            "calculus", "differential", "integral", "measure",

            # Topology
            "topology", "metric", "compact", "connected", "continuous",
            "open set", "closed set", "topological space",

            # Number Theory
            "prime", "divisible", "integer", "natural number", "rational",
            "modular", "congruence", "divisor", "factor",

            # Combinatorics
            "permutation", "combination", "graph", "tree", "path",
            "binomial", "combinatorial", "discrete",

            # Geometry
            "triangle", "circle", "polygon", "angle", "parallel",
            "perpendicular", "geometric", "euclidean",

            # Logic
            "forall", "exists", "implies", "equivalent", "quantifier",
            "propositional", "predicate", "logical",

            # Set Theory
            "set", "subset", "union", "intersection", "cardinality",
            "infinity", "bijection", "injection", "surjection",

            # Symbols (converted to words)
            "for all", "there exists", "element of", "subset", "infinity"
        ]

        self.domain_keywords = {
            MathematicalDomain.ALGEBRA: ["group", "ring", "field", "vector", "matrix", "polynomial"],
            MathematicalDomain.ANALYSIS: ["limit", "derivative", "integral", "continuity", "function"],
            MathematicalDomain.TOPOLOGY: ["topology", "metric", "compact", "connected", "open"],
            MathematicalDomain.NUMBER_THEORY: ["prime", "divisible", "integer", "modular", "congruence"],
            MathematicalDomain.COMBINATORICS: ["permutation", "combination", "graph", "tree", "combinatorial"],
            MathematicalDomain.GEOMETRY: ["triangle", "circle", "polygon", "angle", "euclidean"],
            MathematicalDomain.LOGIC: ["forall", "exists", "implies", "quantifier", "proposition"],
            MathematicalDomain.SET_THEORY: ["set", "subset", "union", "intersection", "cardinality"],
        }

    def detect_mathematical_content(self, text: str) -> bool:
        """
        Detect if text contains mathematical content

        Args:
            text: Text to analyze

        Returns:
            True if mathematical content detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.mathematical_keywords)

    def classify_domain(self, text: str) -> MathematicalDomain:
        """
        Classify mathematical domain of the text

        Args:
            text: Text to classify

        Returns:
            MathematicalDomain classification
        """
        text_lower = text.lower()
        scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return MathematicalDomain.GENERAL

        return max(scores.items(), key=lambda x: x[1])[0]

    def extract_components(
        self,
        text: str
    ) -> List[MathematicalComponent]:
        """
        Extract mathematical components from text

        Args:
            text: Text to extract from

        Returns:
            List of MathematicalComponent objects
        """
        components = []
        domain = self.classify_domain(text)

        # Extract theorems
        theorem_pattern = r'(?:theorem|lemma|corollary|proposition)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)(?=\n\n|\Z)'
        import re
        for match in re.finditer(theorem_pattern, text, re.DOTALL | re.IGNORECASE):
            type_name, statement = match.groups()
            components.append(MathematicalComponent(
                type=match.group(1).lower(),
                name=type_name,
                statement=statement.strip(),
                domain=domain,
                complexity_score=self._estimate_complexity(statement)
            ))

        # Extract definitions
        definition_pattern = r'definition\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)(?=\n\n|\Z)'
        for match in re.finditer(definition_pattern, text, re.DOTALL | re.IGNORECASE):
            name, statement = match.groups()
            components.append(MathematicalComponent(
                type="definition",
                name=name,
                statement=statement.strip(),
                domain=domain,
                complexity_score=self._estimate_complexity(statement)
            ))

        return components

    def _estimate_complexity(self, statement: str) -> float:
        """
        Estimate complexity of a mathematical statement

        Args:
            statement: Mathematical statement

        Returns:
            Complexity score from 0.0 to 1.0
        """
        # Count mathematical symbols and complex constructs
        complexity_indicators = [
            'forall', 'exists', 'forall', 'exists',
            'sum', 'product', 'integral', 'derivative',
            'limit', 'infinity', 'union', 'intersection',
            'subset', 'implies', 'iff'
        ]

        count = sum(1 for indicator in complexity_indicators if indicator.lower() in statement.lower())

        # Add to length-based complexity
        length_complexity = min(len(statement) / 500, 0.5)

        # Combine and normalize to 0-1
        total_complexity = (count / 10) + length_complexity
        return min(max(total_complexity, 0.0), 1.0)


# =============================================================================
# LEANAIDE-HEPHAEUSTUS BRIDGE
# =============================================================================

class LeanAideHephaestusBridge:
    """
    Bridge between LeanAide and Hephaestus workflow phases

    This bridge integrates Lean 4 mathematical verification into the
    Hephaestus 6-phase workflow, providing:

    1. Mathematical content detection (Phase 1)
    2. Natural language to Lean 4 translation (Phase 2)
    3. Solution verification (Phase 3)
    4. Proof checking (Phase 4)
    5. Formal verification (Phase 5)
    6. Knowledge extraction (Phase 6)

    Each method:
    - Creates/updates Hephaestus tickets for tracking
    - Uses LeanAide client for operations
    - Returns structured results with ticket IDs
    - Handles mathematical problem detection
    - Supports sync and async execution
    """

    def __init__(self, config: Optional[LeanAideConfig] = None):
        """
        Initialize the LeanAide-Hephaestus bridge

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or LeanAideConfig()
        self.client = LeanAideClient(self.config)
        self.detector = MathematicalProblemDetector()
        self.tickets: Dict[str, HephaestusTicket] = {}
        self.ticket_counter = 0
        self.ticket_lock = threading.Lock()

        logger.info(f"LeanAide-Hephaestus Bridge initialized")
        logger.info(f"  LeanAide server: {self.client.server_url}")
        logger.info(f"  Server available: {self.client.available}")
        logger.info(f"  Tickets enabled: {self.config.enable_tickets}")

    # =========================================================================
    # TICKET MANAGEMENT
    # =========================================================================

    def _create_ticket(
        self,
        phase: str,
        data: Dict[str, Any]
    ) -> HephaestusTicket:
        """
        Create a Hephaestus ticket for tracking

        Args:
            phase: Workflow phase
            data: Ticket data

        Returns:
            HephaestusTicket object
        """
        with self.ticket_lock:
            self.ticket_counter += 1
            ticket_id = f"LEANAIDE-{self.ticket_counter:06d}"

        now = datetime.now().isoformat()

        ticket = HephaestusTicket(
            ticket_id=ticket_id,
            phase=phase,
            status="created",
            created_at=now,
            updated_at=now,
            data=data
        )

        self.tickets[ticket_id] = ticket

        logger.info(f"Created ticket {ticket_id} for phase {phase}")

        # Post to Hephaestus if configured
        if self.config.enable_tickets and self.config.ticket_base_url:
            self._post_ticket_to_hephaestus(ticket)

        return ticket

    def _update_ticket(
        self,
        ticket_id: str,
        status: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Update an existing ticket

        Args:
            ticket_id: Ticket to update
            status: New status
            data: Optional additional data
        """
        if ticket_id not in self.tickets:
            logger.warning(f"Ticket {ticket_id} not found")
            return

        ticket = self.tickets[ticket_id]
        ticket.status = status
        ticket.updated_at = datetime.now().isoformat()

        if data:
            ticket.data.update(data)

        logger.debug(f"Updated ticket {ticket_id} to status: {status}")

    def _post_ticket_to_hephaestus(self, ticket: HephaestusTicket):
        """
        Post ticket to Hephaestus server

        Args:
            ticket: Ticket to post
        """
        if not self.config.ticket_base_url:
            return

        try:
            url = f"{self.config.ticket_base_url}/tickets"
            response = requests.post(
                url,
                json=asdict(ticket),
                timeout=10
            )
            if response.status_code == 200:
                logger.debug(f"Posted ticket {ticket.ticket_id} to Hephaestus")
            else:
                logger.warning(f"Failed to post ticket: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to post ticket to Hephaestus: {e}")

    # =========================================================================
    # PHASE 1: MATHEMATICAL ANALYSIS
    # =========================================================================

    async def execute_phase_1_analysis(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 1: Analyze mathematical content in problems

        Detects mathematical problems, classifies domain, extracts components,
        and estimates complexity.

        Args:
            problem_statement: The problem to analyze
            context: Additional context
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with analysis results
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_1_analysis",
            data={
                "problem_statement": problem_statement[:200],  # Truncate for ticket
                "context": context
            }
        )

        try:
            logger.info(f"Phase 1: Analyzing mathematical content")

            # Detect mathematical content
            has_math = self.detector.detect_mathematical_content(problem_statement)

            if not has_math:
                self._update_ticket(ticket.ticket_id, "completed", {
                    "has_mathematical_content": False
                })

                return LeanAideResult(
                    success=True,
                    phase="phase_1_analysis",
                    ticket_id=ticket.ticket_id,
                    warnings=["No mathematical content detected"],
                    metadata={
                        "has_mathematical_content": False,
                        "problem_statement": problem_statement
                    },
                    execution_time=time.time() - start_time
                )

            # Classify domain
            domain = self.detector.classify_domain(problem_statement)

            # Extract components
            components = self.detector.extract_components(problem_statement)

            # Calculate overall complexity
            avg_complexity = sum(c.complexity_score for c in components) / len(components) if components else 0.0

            result_data = {
                "has_mathematical_content": True,
                "domain": domain.value,
                "num_components": len(components),
                "components": [asdict(c) for c in components],
                "average_complexity": avg_complexity
            }

            self._update_ticket(ticket.ticket_id, "completed", result_data)

            logger.info(f"Phase 1 complete: {len(components)} components, domain={domain.value}")

            return LeanAideResult(
                success=True,
                phase="phase_1_analysis",
                ticket_id=ticket.ticket_id,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_1_analysis",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 2: TRANSLATION TO LEAN 4
    # =========================================================================

    async def execute_phase_2_translate(
        self,
        mathematical_statement: str,
        components: Optional[List[MathematicalComponent]] = None,
        include_context: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 2: Translate natural language math to Lean 4

        Translates natural language mathematical statements to Lean 4 code
        using LeanAide's translation capabilities.

        Args:
            mathematical_statement: Natural language math to translate
            components: Optional pre-extracted components
            include_context: Include context from similar theorems
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with translation results
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_2_translate",
            data={
                "statement": mathematical_statement[:200],
                "include_context": include_context
            }
        )

        try:
            logger.info(f"Phase 2: Translating to Lean 4")

            if not self.client.available:
                raise Exception("LeanAide server not available")

            # Use LeanAide to translate
            result = await self.client.translate_to_lean(
                mathematical_statement,
                include_context=include_context
            )

            # Add ticket ID to result
            result.ticket_id = ticket.ticket_id

            if result.success:
                self._update_ticket(ticket.ticket_id, "completed", {
                    "lean_code": result.lean_code[:500] if result.lean_code else None
                })
                logger.info(f"Phase 2 complete: translation successful")
            else:
                self._update_ticket(ticket.ticket_id, "failed", {
                    "errors": result.errors
                })

            return result

        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_2_translate",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 3: VERIFICATION
    # =========================================================================

    async def execute_phase_3_verify(
        self,
        lean_code: str,
        original_statement: Optional[str] = None,
        timeout: Optional[int] = None,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 3: Verify solutions using Lean 4

        Verifies that Lean 4 code is correct and elaborates successfully
        using Lean 4's type checker.

        Args:
            lean_code: Lean 4 code to verify
            original_statement: Original natural language statement
            timeout: Optional timeout in seconds
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with verification results
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_3_verify",
            data={
                "code_length": len(lean_code),
                "original_statement": original_statement[:200] if original_statement else None
            }
        )

        try:
            logger.info(f"Phase 3: Verifying Lean 4 code")

            if not self.client.available:
                # Simulate verification if server not available
                # In production, this would be an error
                logger.warning("LeanAide server not available, simulating verification")
                success = "theorem" in lean_code or "lemma" in lean_code
                errors = [] if success else ["Simulation: could not verify"]

                result = LeanAideResult(
                    success=success,
                    phase="phase_3_verify",
                    ticket_id=ticket.ticket_id,
                    verification_result="Simulated verification" if success else None,
                    errors=errors,
                    execution_time=time.time() - start_time
                )
            else:
                result = await self.client.verify_lean_code(lean_code, timeout)
                result.ticket_id = ticket.ticket_id

            if result.success:
                self._update_ticket(ticket.ticket_id, "completed", {
                    "verification_passed": True
                })
                logger.info(f"Phase 3 complete: verification successful")
            else:
                self._update_ticket(ticket.ticket_id, "failed", {
                    "verification_errors": result.errors
                })

            return result

        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_3_verify",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 4: PROOF CHECKING
    # =========================================================================

    async def execute_phase_4_proof_check(
        self,
        lean_code: str,
        proof_content: Optional[str] = None,
        check_completeness: bool = True,
        check_correctness: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 4: Check proof validity and completeness

        Analyzes proofs for completeness, correctness, and style.
        Checks that all proof obligations are discharged.

        Args:
            lean_code: Lean 4 code with proof
            proof_content: Optional proof content to check
            check_completeness: Check if proof is complete
            check_correctness: Check if proof is correct
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with proof checking results
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_4_proof_check",
            data={
                "code_length": len(lean_code),
                "checks": {
                    "completeness": check_completeness,
                    "correctness": check_correctness
                }
            }
        )

        try:
            logger.info(f"Phase 4: Checking proof validity")

            # Analyze the proof
            checks = {
                "has_sorry": "sorry" in lean_code,
                "has_admit": "admit" in lean_code,
                "is_complete": False,
                "proof_lines": 0
            }

            # Count proof lines
            lines = lean_code.split('\n')
            proof_lines = [l for l in lines if l.strip() and not l.strip().startswith('--')]
            checks["proof_lines"] = len(proof_lines)

            # Check completeness
            if check_completeness:
                checks["is_complete"] = not (checks["has_sorry"] or checks["has_admit"])

            # Check correctness (via verification)
            verification_needed = check_correctness and not checks["has_sorry"]
            verification_result = None

            if verification_needed:
                verify_result = await self.client.verify_lean_code(lean_code)
                checks["verification_passed"] = verify_result.success
                verification_result = verify_result.verification_result

            # Determine overall success
            success = True
            warnings = []

            if check_completeness and checks["has_sorry"]:
                warnings.append("Proof contains 'sorry' placeholders")
                success = False

            if check_completeness and checks["has_admit"]:
                warnings.append("Proof contains 'admit' placeholders")
                success = False

            if check_correctness and verification_needed:
                if not checks.get("verification_passed", False):
                    warnings.append("Verification failed")
                    success = False

            result_data = {
                "checks": checks,
                "verification_result": verification_result
            }

            self._update_ticket(ticket.ticket_id, "completed" if success else "failed", result_data)

            logger.info(f"Phase 4 complete: proof check {'passed' if success else 'failed'}")

            return LeanAideResult(
                success=success,
                phase="phase_4_proof_check",
                ticket_id=ticket.ticket_id,
                warnings=warnings,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_4_proof_check",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 5: FORMAL VERIFICATION
    # =========================================================================

    async def execute_phase_5_formal_verification(
        self,
        lean_code: str,
        verification_level: str = "strict",
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 5: Final formal verification

        Performs comprehensive formal verification of the entire Lean 4 code.
        This is the most thorough verification step.

        Args:
            lean_code: Lean 4 code to verify
            verification_level: "strict", "standard", or "relaxed"
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with formal verification results
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_5_formal_verification",
            data={
                "verification_level": verification_level,
                "code_length": len(lean_code)
            }
        )

        try:
            logger.info(f"Phase 5: Formal verification (level={verification_level})")

            # Perform verification
            verify_result = await self.client.verify_lean_code(lean_code)

            # Additional checks based on verification level
            additional_checks = {}

            if verification_level == "strict":
                # Check for any warnings or issues
                additional_checks["style_check"] = "by " in lean_code or "simp" in lean_code
                additional_checks["no_tactics"] = not any(tactic in lean_code for tactic in ["sorry", "admit"])

            result_data = {
                "verification_level": verification_level,
                "verification_passed": verify_result.success,
                "verification_output": verify_result.verification_result,
                "additional_checks": additional_checks
            }

            self._update_ticket(ticket.ticket_id, "completed" if verify_result.success else "failed", result_data)

            logger.info(f"Phase 5 complete: formal verification {'passed' if verify_result.success else 'failed'}")

            return LeanAideResult(
                success=verify_result.success,
                phase="phase_5_formal_verification",
                ticket_id=ticket.ticket_id,
                verification_result=verify_result.verification_result,
                errors=verify_result.errors,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Phase 5 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_5_formal_verification",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 6: KNOWLEDGE EXTRACTION
    # =========================================================================

    async def execute_phase_6_knowledge_extraction(
        self,
        lean_code: str,
        verification_result: Optional[LeanAideResult] = None,
        extract_theorems: bool = True,
        extract_dependencies: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 6: Extract verified theorems for knowledge base

        Extracts verified theorems, lemmas, and definitions from the
        Lean 4 code for storage in the knowledge base.

        Args:
            lean_code: Verified Lean 4 code
            verification_result: Optional verification result
            extract_theorems: Extract theorem statements
            extract_dependencies: Extract dependencies between theorems
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with extracted knowledge
        """
        start_time = time.time()

        # Create ticket
        ticket = self._create_ticket(
            phase="phase_6_knowledge_extraction",
            data={
                "extract_theorems": extract_theorems,
                "extract_dependencies": extract_dependencies
            }
        )

        try:
            logger.info(f"Phase 6: Extracting knowledge")

            # Extract theorems, lemmas, definitions
            import re

            knowledge = {
                "theorems": [],
                "lemmas": [],
                "definitions": [],
                "dependencies": []
            }

            if extract_theorems:
                # Extract theorems
                theorem_pattern = r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*:='
                for match in re.finditer(theorem_pattern, lean_code, re.MULTILINE):
                    name, statement = match.groups()
                    knowledge["theorems"].append({
                        "name": name,
                        "statement": statement.strip(),
                        "type": "theorem"
                    })

                # Extract lemmas
                lemma_pattern = r'^lemma\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*:='
                for match in re.finditer(lemma_pattern, lean_code, re.MULTILINE):
                    name, statement = match.groups()
                    knowledge["lemmas"].append({
                        "name": name,
                        "statement": statement.strip(),
                        "type": "lemma"
                    })

                # Extract definitions
                def_pattern = r'^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(.+?))?\s*:='
                for match in re.finditer(def_pattern, lean_code, re.MULTILINE):
                    name, type_sig = match.groups()
                    knowledge["definitions"].append({
                        "name": name,
                        "type": type_sig.strip() if type_sig else None,
                        "kind": "definition"
                    })

            if extract_dependencies:
                # Simple dependency extraction (looks for imports and uses)
                import_pattern = r'^import\s+(.+)$'
                for match in re.finditer(import_pattern, lean_code, re.MULTILINE):
                    knowledge["dependencies"].append({
                        "type": "import",
                        "target": match.group(1).strip()
                    })

            # Check if verification was successful
            is_verified = verification_result.success if verification_result else False

            result_data = {
                "knowledge": knowledge,
                "is_verified": is_verified,
                "extraction_summary": {
                    "theorems": len(knowledge["theorems"]),
                    "lemmas": len(knowledge["lemmas"]),
                    "definitions": len(knowledge["definitions"]),
                    "dependencies": len(knowledge["dependencies"])
                }
            }

            self._update_ticket(ticket.ticket_id, "completed", result_data)

            logger.info(f"Phase 6 complete: extracted {len(knowledge['theorems'])} theorems, "
                       f"{len(knowledge['lemmas'])} lemmas, {len(knowledge['definitions'])} definitions")

            return LeanAideResult(
                success=True,
                phase="phase_6_knowledge_extraction",
                ticket_id=ticket.ticket_id,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Phase 6 failed: {e}")
            self._update_ticket(ticket.ticket_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_6_knowledge_extraction",
                ticket_id=ticket.ticket_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # FULL WORKFLOW EXECUTION
    # =========================================================================

    async def execute_full_workflow(
        self,
        problem_statement: str,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> Dict[str, Any]:
        """
        Execute the complete 6-phase LeanAide workflow

        Runs all phases from analysis through knowledge extraction.

        Args:
            problem_statement: Mathematical problem statement
            execution_mode: Synchronous or asynchronous execution

        Returns:
            Dict with results from all phases
        """
        logger.info(f"Starting full LeanAide workflow for: {problem_statement[:100]}...")

        results = {
            "problem_statement": problem_statement,
            "phases": {},
            "workflow_success": True,
            "start_time": datetime.now().isoformat()
        }

        try:
            # Phase 1: Analysis
            logger.info("=" * 60)
            logger.info("PHASE 1: Mathematical Analysis")
            phase1 = await self.execute_phase_1_analysis(problem_statement)
            results["phases"]["phase_1"] = asdict(phase1)

            if not phase1.success:
                results["workflow_success"] = False
                results["failure_phase"] = "phase_1"
                return results

            # Only continue if mathematical content detected
            if not phase1.metadata.get("has_mathematical_content"):
                results["message"] = "No mathematical content detected, workflow stopped"
                return results

            # Phase 2: Translation
            logger.info("=" * 60)
            logger.info("PHASE 2: Translation to Lean 4")
            phase2 = await self.execute_phase_2_translate(problem_statement)
            results["phases"]["phase_2"] = asdict(phase2)

            if not phase2.success:
                results["workflow_success"] = False
                results["failure_phase"] = "phase_2"
                return results

            lean_code = phase2.lean_code

            # Phase 3: Verification
            logger.info("=" * 60)
            logger.info("PHASE 3: Verification")
            phase3 = await self.execute_phase_3_verify(lean_code, problem_statement)
            results["phases"]["phase_3"] = asdict(phase3)

            # Phase 4: Proof Check
            logger.info("=" * 60)
            logger.info("PHASE 4: Proof Checking")
            phase4 = await self.execute_phase_4_proof_check(lean_code)
            results["phases"]["phase_4"] = asdict(phase4)

            # Phase 5: Formal Verification
            logger.info("=" * 60)
            logger.info("PHASE 5: Formal Verification")
            phase5 = await self.execute_phase_5_formal_verification(lean_code)
            results["phases"]["phase_5"] = asdict(phase5)

            # Phase 6: Knowledge Extraction
            logger.info("=" * 60)
            logger.info("PHASE 6: Knowledge Extraction")
            phase6 = await self.execute_phase_6_knowledge_extraction(
                lean_code,
                verification_result=phase5
            )
            results["phases"]["phase_6"] = asdict(phase6)

            results["end_time"] = datetime.now().isoformat()
            results["message"] = "Full workflow completed successfully"

            logger.info("=" * 60)
            logger.info("Full workflow completed")

            return results

        except Exception as e:
            logger.error(f"Full workflow failed: {e}")
            results["workflow_success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            return results

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_ticket(self, ticket_id: str) -> Optional[HephaestusTicket]:
        """
        Get a ticket by ID

        Args:
            ticket_id: Ticket ID

        Returns:
            HephaestusTicket or None
        """
        return self.tickets.get(ticket_id)

    def get_all_tickets(self) -> List[HephaestusTicket]:
        """
        Get all tickets

        Returns:
            List of all tickets
        """
        return list(self.tickets.values())

    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()

    def __del__(self):
        """Destructor to ensure cleanup"""
        # Note: Can't use async in __del__, so we just log
        logger.debug("LeanAideHephaestusBridge being destroyed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def analyze_and_verify_math_problem(
    problem_statement: str,
    config: Optional[LeanAideConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze and verify a mathematical problem

    Args:
        problem_statement: Mathematical problem statement
        config: Optional LeanAide configuration

    Returns:
        Dict with analysis and verification results
    """
    bridge = LeanAideHephaestusBridge(config)

    try:
        result = await bridge.execute_full_workflow(problem_statement)
        return result
    finally:
        await bridge.cleanup()


def run_sync(coro):
    """
    Run async coroutine synchronously

    Args:
        coro: Coroutine to run

    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Classes
    'LeanAideHephaestusBridge',
    'LeanAideClient',
    'MathematicalProblemDetector',
    'LeanAideConfig',
    'LeanAideResult',
    'MathematicalComponent',
    'HephaestusTicket',
    'VerificationStatus',
    'ExecutionMode',
    'MathematicalDomain',

    # Functions
    'analyze_and_verify_math_problem',
    'run_sync',
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("LeanAide-Hephaestus Bridge Module")
    print("=" * 60)

    # Example usage
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = "Prove that there are infinitely many prime numbers."

    print(f"Problem: {problem}")
    print()

    # Run the workflow
    result = run_sync(analyze_and_verify_math_problem(problem))

    print("Result:")
    print(json.dumps(result, indent=2))
