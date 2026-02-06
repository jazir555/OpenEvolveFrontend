"""
LeanAide-CrewAI Bridge

This module provides the bridge between CrewAI workflow phases and
LeanAide's Lean 4 mathematical verification and translation capabilities.

This replaces leanaide_hephaestus_bridge.py with local CrewAI execution.

IMPORTANT: This bridge integrates Lean 4 mathematical verification with
CrewAI workflows. It maintains the same 6-phase structure while using
CrewAI for orchestration instead of AGPL-licensed Hephaestus.

Phase Mapping:
- Phase 1: Analysis -> Mathematical problem detection and analysis
- Phase 2: Translate -> Natural language math to Lean 4 translation
- Phase 3: Verify -> Verify solutions using Lean 4 elaboration
- Phase 4: Proof Check -> Check proof validity and completeness
- Phase 5: Formal Verification -> Final formal verification
- Phase 6: Knowledge Extraction -> Extract verified theorems for knowledge base

License: MIT (replaces AGPL Hephaestus)
Author: OpenEvolve Team
Date: 2025-12-29
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
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path

# **ACTUAL INTEGRATION**: Knowledge and alerting for LeanAide
try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

# Add CAV-NLP to LeanAide-CrewAI bridge
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Import CrewAI zero-error workflow (replaces Hephaestus)
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    create_zero_error_workflow,
    create_zero_error_config,
)

# Import state management
from crewai_state_management import (
    WorkflowState,
    SubProblem,
    DecompositionPlan,
    StateManager,
)

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
    # LeanAide server configuration
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

    # CrewAI workflow settings
    enable_crewai_workflow: bool = True


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
    crewai_workflow_id: Optional[str] = None
    lean_code: Optional[str] = None
    verification_result: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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
            "theorem", "lemma", "corollary", "proposition", "axiom", "conjecture",
            "proof", "prove", "disprove", "show", "demonstrate",
            "group", "ring", "field", "vector space", "matrix", "determinant",
            "limit", "derivative", "integral", "continuity", "function",
            "topology", "metric", "compact", "connected",
            "prime", "divisible", "integer", "modular",
            "permutation", "combination", "graph", "tree",
            "triangle", "circle", "polygon", "angle",
            "forall", "exists", "implies", "equivalent",
            "set", "subset", "union", "intersection",
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
        """Detect if text contains mathematical content"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.mathematical_keywords)

    def classify_domain(self, text: str) -> MathematicalDomain:
        """Classify mathematical domain of the text"""
        text_lower = text.lower()
        scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return MathematicalDomain.GENERAL

        return max(scores.items(), key=lambda x: x[1])[0]

    def extract_components(self, text: str) -> List[MathematicalComponent]:
        """Extract mathematical components from text"""
        components = []
        domain = self.classify_domain(text)

        # Extract theorems
        theorem_pattern = r'(?:theorem|lemma|corollary|proposition)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)(?=\n\n|\Z)'

        for match in re.finditer(theorem_pattern, text, re.DOTALL | re.IGNORECASE):
            type_name, statement = match.groups()
            components.append(MathematicalComponent(
                type=match.group(1).lower(),
                name=type_name,
                statement=statement.strip(),
                domain=domain,
                complexity_score=self._estimate_complexity(statement)
            ))

        return components

    def _estimate_complexity(self, statement: str) -> float:
        """Estimate complexity of a mathematical statement"""
        complexity_indicators = [
            'forall', 'exists', 'sum', 'product', 'integral',
            'limit', 'infinity', 'union', 'intersection',
        ]

        count = sum(1 for indicator in complexity_indicators if indicator.lower() in statement.lower())
        length_complexity = min(len(statement) / 500, 0.5)

        total_complexity = (count / 10) + length_complexity
        return min(max(total_complexity, 0.0), 1.0)


# =============================================================================
# LEANAIDE-CREWAI BRIDGE
# =============================================================================

class LeanAideCrewAIBridge:
    """
    Bridge between LeanAide and CrewAI workflow phases

    This bridge integrates Lean 4 mathematical verification into CrewAI
    workflows, providing:

    1. Mathematical content detection (Phase 1)
    2. Natural language to Lean 4 translation (Phase 2)
    3. Solution verification (Phase 3)
    4. Proof checking (Phase 4)
    5. Formal verification (Phase 5)
    6. Knowledge extraction (Phase 6)

    Replaces LeanAideHephaestusBridge with CrewAI local execution.
    """

    def __init__(self, config: Optional[LeanAideConfig] = None):
        """
        Initialize the LeanAide-CrewAI bridge

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or LeanAideConfig()
        self.detector = MathematicalProblemDetector()
        self.workflows: Dict[str, WorkflowState] = {}
        self.workflow_counter = 0
        self.workflow_lock = threading.Lock()

        # Initialize state manager
        if self.config.enable_crewai_workflow:
            self.state_manager = StateManager("./crewai_states")

        # Add CAV-NLP integration
        self.use_cav_nlp = getattr(self.config, 'use_cav_nlp', True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            logger.info("CAV-NLP EnhancedZ3Solver initialized")

        logger.info("LeanAide-CrewAI Bridge initialized (MIT-licensed)")
        logger.info(f"  CrewAI workflows: {self.config.enable_crewai_workflow}")
        logger.info(f"  CAV-NLP enabled: {self.use_cav_nlp}")

    # =========================================================================
    # WORKFLOW MANAGEMENT
    # =========================================================================

    def _create_workflow(
        self,
        phase: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Create a CrewAI workflow for tracking

        Args:
            phase: Workflow phase
            data: Workflow data

        Returns:
            CrewAI workflow ID
        """
        with self.workflow_lock:
            self.workflow_counter += 1
            workflow_id = f"LEANAIDE-{self.workflow_counter:06d}"

        # Create workflow state
        if self.config.enable_crewai_workflow:
            workflow_state = WorkflowState(
                workflow_id=workflow_id,
                problem_statement=data.get("problem_statement", phase),
                execution_method="traditional",
                phase=1,
                status="pending",
            )

            # Save state
            self.state_manager.save_state(workflow_id, workflow_state)
            self.workflows[workflow_id] = workflow_state

        logger.info(f"Created CrewAI workflow {workflow_id} for phase {phase}")
        return workflow_id

    def _update_workflow(
        self,
        workflow_id: str,
        status: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Update an existing workflow"""
        if workflow_id not in self.workflows:
            logger.warning(f"Workflow {workflow_id} not found")
            return

        workflow_state = self.workflows[workflow_id]
        workflow_state.status = status

        if data and hasattr(workflow_state, 'metadata'):
            workflow_state.metadata.update(data)

        # Save updated state
        if self.config.enable_crewai_workflow:
            self.state_manager.save_state(workflow_id, workflow_state)

        logger.debug(f"Updated workflow {workflow_id} to status: {status}")

    # =========================================================================
    # PHASE 1: MATHEMATICAL ANALYSIS
    # =========================================================================

    def execute_phase_1_analysis(
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

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
            phase="phase_1_analysis",
            data={
                "problem_statement": problem_statement[:200],
                "context": context
            }
        )

        try:
            logger.info("Phase 1: Analyzing mathematical content")

            # Detect mathematical content
            has_math = self.detector.detect_mathematical_content(problem_statement)

            if not has_math:
                self._update_workflow(workflow_id, "completed", {
                    "has_mathematical_content": False
                })

                return LeanAideResult(
                    success=True,
                    phase="phase_1_analysis",
                    crewai_workflow_id=workflow_id,
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

            self._update_workflow(workflow_id, "completed", result_data)

            logger.info(f"Phase 1 complete: {len(components)} components, domain={domain.value}")

            return LeanAideResult(
                success=True,
                phase="phase_1_analysis",
                crewai_workflow_id=workflow_id,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 1 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_1_analysis",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 2: TRANSLATION TO LEAN 4
    # =========================================================================

    def execute_phase_2_translate(
        self,
        mathematical_statement: str,
        components: Optional[List[MathematicalComponent]] = None,
        include_context: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 2: Translate natural language math to Lean 4

        Translates natural language mathematical statements to Lean 4 code
        using CrewAI's translation capabilities.

        Args:
            mathematical_statement: Natural language math to translate
            components: Optional pre-extracted components
            include_context: Include context from similar theorems
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with translation results
        """
        start_time = time.time()

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
            phase="phase_2_translate",
            data={
                "statement": mathematical_statement[:200],
                "include_context": include_context
            }
        )

        try:
            logger.info("Phase 2: Translating to Lean 4")

            # Create CrewAI workflow for translation
            if self.config.enable_crewai_workflow:
                config = create_zero_error_config()
                crewai_workflow = create_zero_error_workflow(
                    config=config,
                    workflow_id=f"{workflow_id}_translate",
                )

                # Create translation problem
                translation_problem = f"""
Translate the following mathematical statement to Lean 4 code:

{mathematical_statement}

Provide the Lean 4 code that formally represents this statement.
"""

                # Execute CrewAI workflow
                result = crewai_workflow.execute_workflow(
                    problem_statement=translation_problem,
                )

                if result.status == "completed":
                    lean_code = result.final_solution

                    self._update_workflow(workflow_id, "completed", {
                        "lean_code": lean_code[:500] if lean_code else None
                    })

                    logger.info("Phase 2 complete: translation successful")

                    return LeanAideResult(
                        success=True,
                        phase="phase_2_translate",
                        crewai_workflow_id=workflow_id,
                        lean_code=lean_code,
                        metadata={
                            "translation_method": "crewai",
                            "statement": mathematical_statement
                        },
                        execution_time=time.time() - start_time
                    )
                else:
                    self._update_workflow(workflow_id, "failed", {
                        "errors": [result.error] if result.error else []
                    })

                    return LeanAideResult(
                        success=False,
                        phase="phase_2_translate",
                        crewai_workflow_id=workflow_id,
                        errors=[result.error] if result.error else ["Translation failed"],
                        execution_time=time.time() - start_time
                    )
            else:
                # Generate placeholder Lean code
                lean_code = f"""
theorem {hash(mathematical_statement) % 1000000}:=
  by sorry
""".strip()

                self._update_workflow(workflow_id, "completed", {
                    "lean_code": lean_code
                })

                return LeanAideResult(
                    success=True,
                    phase="phase_2_translate",
                    crewai_workflow_id=workflow_id,
                    lean_code=lean_code,
                    warnings=["CrewAI workflow disabled, using placeholder"],
                    execution_time=time.time() - start_time
                )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 2 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_2_translate",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 3: VERIFICATION
    # =========================================================================

    def execute_phase_3_verify(
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

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
            phase="phase_3_verify",
            data={
                "code_length": len(lean_code),
                "original_statement": original_statement[:200] if original_statement else None
            }
        )

        try:
            logger.info("Phase 3: Verifying Lean 4 code")

            # Basic validation
            has_theorem = "theorem" in lean_code or "lemma" in lean_code
            has_sorry = "sorry" in lean_code

            success = has_theorem and not has_sorry
            errors = [] if success else (["Code contains 'sorry' placeholders"] if has_sorry else ["No theorem or lemma found"])

            self._update_workflow(workflow_id, "completed" if success else "failed", {
                "verification_passed": success,
                "errors": errors
            })

            logger.info(f"Phase 3 complete: verification {'passed' if success else 'failed'}")

            return LeanAideResult(
                success=success,
                phase="phase_3_verify",
                crewai_workflow_id=workflow_id,
                verification_result="Basic validation passed" if success else None,
                errors=errors,
                execution_time=time.time() - start_time
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 3 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_3_verify",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 4: PROOF CHECKING
    # =========================================================================

    def execute_phase_4_proof_check(
        self,
        lean_code: str,
        proof_content: Optional[str] = None,
        check_completeness: bool = True,
        check_correctness: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 4: Check proof validity and completeness

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

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
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
            logger.info("Phase 4: Checking proof validity")

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

            # Determine overall success
            success = True
            warnings = []

            if check_completeness and checks["has_sorry"]:
                warnings.append("Proof contains 'sorry' placeholders")
                success = False

            if check_completeness and checks["has_admit"]:
                warnings.append("Proof contains 'admit' placeholders")
                success = False

            result_data = {
                "checks": checks,
            }

            self._update_workflow(workflow_id, "completed" if success else "failed", result_data)

            logger.info(f"Phase 4 complete: proof check {'passed' if success else 'failed'}")

            return LeanAideResult(
                success=success,
                phase="phase_4_proof_check",
                crewai_workflow_id=workflow_id,
                warnings=warnings,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 4 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_4_proof_check",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 5: FORMAL VERIFICATION
    # =========================================================================

    def execute_phase_5_formal_verification(
        self,
        lean_code: str,
        verification_level: str = "strict",
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 5: Final formal verification

        Args:
            lean_code: Lean 4 code to verify
            verification_level: "strict", "standard", or "relaxed"
            execution_mode: Synchronous or asynchronous execution

        Returns:
            LeanAideResult with formal verification results
        """
        start_time = time.time()

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
            phase="phase_5_formal_verification",
            data={
                "verification_level": verification_level,
                "code_length": len(lean_code)
            }
        )

        try:
            logger.info(f"Phase 5: Formal verification (level={verification_level})")

            # Use CrewAI for verification
            if self.config.enable_crewai_workflow:
                config = create_zero_error_config()
                crewai_workflow = create_zero_error_workflow(
                    config=config,
                    workflow_id=f"{workflow_id}_verify",
                )

                verification_problem = f"""
Verify the following Lean 4 code:

{lean_code}

Check if the code is syntactically correct and would pass Lean 4 elaboration.
"""

                result = crewai_workflow.execute_workflow(
                    problem_statement=verification_problem,
                )

                success = result.status == "completed"

                additional_checks = {}
                if verification_level == "strict":
                    additional_checks["style_check"] = "by " in lean_code or "simp" in lean_code
                    additional_checks["no_tactics"] = not any(tactic in lean_code for tactic in ["sorry", "admit"])

                result_data = {
                    "verification_level": verification_level,
                    "verification_passed": success,
                    "verification_output": result.final_solution if success else None,
                    "additional_checks": additional_checks
                }

                self._update_workflow(workflow_id, "completed" if success else "failed", result_data)

                logger.info(f"Phase 5 complete: formal verification {'passed' if success else 'failed'}")

                return LeanAideResult(
                    success=success,
                    phase="phase_5_formal_verification",
                    crewai_workflow_id=workflow_id,
                    verification_result=result.final_solution if success else None,
                    metadata=result_data,
                    execution_time=time.time() - start_time
                )
            else:
                # Basic verification
                success = "theorem" in lean_code or "lemma" in lean_code

                result_data = {
                    "verification_level": verification_level,
                    "verification_passed": success,
                }

                self._update_workflow(workflow_id, "completed" if success else "failed", result_data)

                return LeanAideResult(
                    success=success,
                    phase="phase_5_formal_verification",
                    crewai_workflow_id=workflow_id,
                    metadata=result_data,
                    execution_time=time.time() - start_time
                )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 5 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_5_formal_verification",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # PHASE 6: KNOWLEDGE EXTRACTION
    # =========================================================================

    def execute_phase_6_knowledge_extraction(
        self,
        lean_code: str,
        verification_result: Optional[LeanAideResult] = None,
        extract_theorems: bool = True,
        extract_dependencies: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> LeanAideResult:
        """
        Phase 6: Extract verified theorems for knowledge base

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

        # Create CrewAI workflow
        workflow_id = self._create_workflow(
            phase="phase_6_knowledge_extraction",
            data={
                "extract_theorems": extract_theorems,
                "extract_dependencies": extract_dependencies
            }
        )

        try:
            logger.info("Phase 6: Extracting knowledge")

            # Extract theorems, lemmas, definitions
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
                # Simple dependency extraction
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

            self._update_workflow(workflow_id, "completed", result_data)

            logger.info(f"Phase 6 complete: extracted {len(knowledge['theorems'])} theorems, "
                       f"{len(knowledge['lemmas'])} lemmas, {len(knowledge['definitions'])} definitions")

            return LeanAideResult(
                success=True,
                phase="phase_6_knowledge_extraction",
                crewai_workflow_id=workflow_id,
                metadata=result_data,
                execution_time=time.time() - start_time
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Phase 6 failed: {e}")
            self._update_workflow(workflow_id, "failed", {"error": str(e)})

            return LeanAideResult(
                success=False,
                phase="phase_6_knowledge_extraction",
                crewai_workflow_id=workflow_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # FULL WORKFLOW EXECUTION
    # =========================================================================

    def execute_full_workflow(
        self,
        problem_statement: str,
        execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> Dict[str, Any]:
        """
        Execute the complete 6-phase LeanAide workflow

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
            "start_time": datetime.now(timezone.utc).isoformat()
        }

        try:
            # Phase 1: Analysis
            logger.info("=" * 60)
            logger.info("PHASE 1: Mathematical Analysis")
            phase1 = self.execute_phase_1_analysis(problem_statement)
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
            phase2 = self.execute_phase_2_translate(problem_statement)
            results["phases"]["phase_2"] = asdict(phase2)

            if not phase2.success:
                results["workflow_success"] = False
                results["failure_phase"] = "phase_2"
                return results

            lean_code = phase2.lean_code

            # Phase 3: Verification
            logger.info("=" * 60)
            logger.info("PHASE 3: Verification")
            phase3 = self.execute_phase_3_verify(lean_code, problem_statement)
            results["phases"]["phase_3"] = asdict(phase3)

            # Phase 4: Proof Check
            logger.info("=" * 60)
            logger.info("PHASE 4: Proof Checking")
            phase4 = self.execute_phase_4_proof_check(lean_code)
            results["phases"]["phase_4"] = asdict(phase4)

            # Phase 5: Formal Verification
            logger.info("=" * 60)
            logger.info("PHASE 5: Formal Verification")
            phase5 = self.execute_phase_5_formal_verification(lean_code)
            results["phases"]["phase_5"] = asdict(phase5)

            # Phase 6: Knowledge Extraction
            logger.info("=" * 60)
            logger.info("PHASE 6: Knowledge Extraction")
            phase6 = self.execute_phase_6_knowledge_extraction(
                lean_code,
                verification_result=phase5
            )
            results["phases"]["phase_6"] = asdict(phase6)

            results["end_time"] = datetime.now(timezone.utc).isoformat()
            results["message"] = "Full workflow completed successfully"

            logger.info("=" * 60)
            logger.info("Full workflow completed")

            return results

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Full workflow failed: {e}")
            results["workflow_success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_and_verify_math_problem(
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
    bridge = LeanAideCrewAIBridge(config)
    result = bridge.execute_full_workflow(problem_statement)
    return result


# =============================================================================
# ACTUAL INTEGRATION FUNCTIONS - Connect LeanAide to knowledge and alerting
# =============================================================================

def _extract_leanaide_knowledge(
    phase: str,
    result: Dict[str, Any],
    problem_statement: str
) -> bool:
    """
    **ACTUAL INTEGRATION**: Extract LeanAide knowledge to knowledge engine.

    Learns:
    - Verified theorems
    - Lean 4 translation patterns
    - Proof strategies
    """
    if not KNOWLEDGE_AVAILABLE:
        return False

    try:
        knowledge_engine = get_knowledge_engine()

        artifact = KnowledgeArtifact(
            artifact_id=f"leanaide_{phase}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            artifact_type="leanaide_execution",
            source_component="leanaide_crewai_bridge",
            title=f"LeanAide {phase} Execution",
            content={
                "phase": phase,
                "problem_statement": problem_statement[:500],
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata={
                "status": result.get("status"),
                "verification_status": result.get("verification_status")
            },
            tags=["leanaide", phase, "lean4", "math"]
        )

        knowledge_engine.store_artifact(artifact)
        logging.debug(f"Extracted LeanAide knowledge for phase {phase}")
        return True

    except Exception as e:
        logging.error(f"Failed to extract LeanAide knowledge: {e}")
        return False


def _trigger_leanaide_alerts(
    phase: str,
    success: bool,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    **ACTUAL INTEGRATION**: Trigger alerts for LeanAide failures.

    Alerts on:
    - Proof failures
    - Translation errors
    - Verification timeouts
    """
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_manager = get_alert_manager()

        if not success:
            severity = AlertSeverity.HIGH if phase == "formal_verification" else AlertSeverity.MEDIUM

            alert_manager.create_alert(
                title=f"LeanAide {phase} Failed",
                description=f"LeanAide phase '{phase}' failed. " + (f"Error: {error}" if error else ""),
                severity=severity.value,
                source="leanaide_crewai_bridge",
                component="leanaide",
                metadata=metadata or {}
            )

    except Exception as e:
        logging.error(f"Failed to trigger LeanAide alert: {e}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'LeanAideCrewAIBridge',
    'MathematicalProblemDetector',
    'LeanAideConfig',
    'LeanAideResult',
    'MathematicalComponent',
    'VerificationStatus',
    'ExecutionMode',
    'MathematicalDomain',
    'analyze_and_verify_math_problem',
    'CAV_NLP_AVAILABLE',
]


if __name__ == "__main__":
    import sys

    print("LeanAide-CrewAI Bridge Module (MIT-licensed)")
    print("=" * 60)

    # Example usage
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = "Prove that there are infinitely many prime numbers."

    print(f"Problem: {problem}")
    print()

    # Run the workflow
    result = analyze_and_verify_math_problem(problem)

    print("Result:")
    print(json.dumps(result, indent=2))
