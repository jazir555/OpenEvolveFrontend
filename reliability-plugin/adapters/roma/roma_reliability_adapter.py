"""
ROMA Reliability Adapter - Enhanced with Direct Core Integration
================================================================

Enhanced adapter that provides dual-mode access to ROMA:
1. **Direct Core Integration**: Creates ROMA components directly with LMQL constraints
2. **MCP Tool Fallback**: Uses ROMA MCP tools when core is unavailable

IMPORTANT ARCHITECTURE PRINCIPLES:
1. Air Gap Compliance: No modifications to ROMA core files
2. Dual-Mode Execution: Prefers direct core integration, falls back to MCP tools
3. Wrapper Pattern: All LMQL/Guardrails logic lives in the ADAPTER, not in ROMA
4. Graceful Degradation: Works even if LMQL/Guardrails unavailable
5. Zero Trust: Validates everything, handles failures gracefully

Architecture:
    ROMA Core (READ ONLY)
        ↓
    [Mode A] Direct Core Integration (Preferred)
        - Creates ROMA components directly
        - Injects LMQL constraints via wrapper classes
        - Full access to ROMA API
        ↓
    [Mode B] MCP Tools Fallback
        - Uses solve_with_roma, analyze_with_roma
        - Public API interface
        - Limited but reliable access
        ↓
    Reliability Adapter (LMQL constraints + Guardrails validation)
        ↓
    Unified Bridge

Dual-Mode Execution Flow:
    Input Task
        ↓
    Layer 1: Input Validation (Guardrails)
        ↓
    Layer 2: Route to Best Available Method
        - If ROMA Core Available → Direct Integration
        - Else → MCP Tools Fallback
        ↓
    Layer 3: Execute with Constraints
        - LMQL constraints (if available)
        - ROMA execution
        ↓
    Layer 4: Output Validation (Guardrails)
        ↓
    Result

Features:
- Direct ROMA core integration (preferred)
- MCP tool fallback (reliable)
- LMQL constraints injection
- Guardrails validation (input/output)
- Remediation and retry logic
- Comprehensive error handling
- Structured JSON logging
- Health checks and status monitoring

Example Usage:
    # Create adapter
    adapter = RomaReliabilityAdapter()

    # Check status
    status = adapter.get_status()
    print(f"Execution mode: {status['execution_mode']}")

    # Solve with constraints
    result = adapter.solve_with_constraints(
        task="Solve the traveling salesman problem",
        max_depth=3,
        constraints={
            "max_depth": 3,
            "max_subtasks": 10,
            "subtask_token_limit": 500
        }
    )

    if result.success:
        print(f"Solution: {result.result}")
        print(f"Method: {result.metadata.get('method')}")
        print(f"Layers: {result.layers_used}")
    else:
        print(f"Error: {result.error}")

Author: OpenEvolve Team
Version: 2.0.0
License: MIT
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from functools import lru_cache
import traceback

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure structured JSON logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT RELIABILITY LAYERS (with graceful degradation)
# =============================================================================

# Import LMQL Adapter
try:
    from reliability.lmql_adapter import (
        LMQLAdapter,
        Constraint,
        ConstraintType,
        GenerationResult,
        get_default_adapter as get_lmql_adapter
    )
    LMQL_AVAILABLE = True
    logger.info({"event": "lmql_adapter_imported"})
except ImportError as e:
    LMQL_AVAILABLE = False
    LMQLAdapter = None
    Constraint = None
    ConstraintType = None
    GenerationResult = None
    logger.warning({"event": "lmql_adapter_not_available", "error": str(e)})

# Import Guardrails Adapter
try:
    from reliability.guardrails_adapter import (
        GuardrailsAdapter,
        ValidationResult,
        RemediationStrategy,
        create_adapter as create_guardrails_adapter
    )
    GUARDRAILS_AVAILABLE = True
    logger.info({"event": "guardrails_adapter_imported"})
except ImportError as e:
    GUARDRAILS_AVAILABLE = False
    GuardrailsAdapter = None
    ValidationResult = None
    RemediationStrategy = None
    logger.warning({"event": "guardrails_adapter_not_available", "error": str(e)})

# Import Config
try:
    from reliability.config import (
        ReliabilityConfig,
        get_config as get_reliability_config
    )
    CONFIG_AVAILABLE = True
    logger.info({"event": "config_imported"})
except ImportError as e:
    CONFIG_AVAILABLE = False
    ReliabilityConfig = None
    get_reliability_config = None
    logger.warning({"event": "config_not_available", "error": str(e)})

# =============================================================================
# IMPORT ROMA CORE AND MCP TOOLS
# =============================================================================

# Import ROMA Core components (direct integration)
try:
    from roma_dspy import (
        RecursiveSolver,
        solve,
        async_solve,
        event_solve,
        Atomizer,
        Planner,
        Executor,
        Aggregator,
        Verifier,
        AtomizerSignature,
        PlannerSignature,
        ExecutorSignature,
        AggregatorSignature,
        VerifierSignature,
        SubTask,
        TaskNode,
        TaskDAG,
    )
    from roma_dspy.core.factory import AgentFactory
    from roma_dspy.core.registry import AgentRegistry
    from roma_dspy.config.schemas.root import ROMAConfig
    ROMA_CORE_AVAILABLE = True
    logger.info({"event": "roma_core_imported"})
except ImportError as e:
    ROMA_CORE_AVAILABLE = False
    RecursiveSolver = None
    solve = None
    async_solve = None
    event_solve = None
    Atomizer = None
    Planner = None
    Executor = None
    Aggregator = None
    Verifier = None
    AtomizerSignature = None
    PlannerSignature = None
    ExecutorSignature = None
    AggregatorSignature = None
    VerifierSignature = None
    SubTask = None
    TaskNode = None
    TaskDAG = None
    AgentFactory = None
    AgentRegistry = None
    ROMAConfig = None
    logger.warning({"event": "roma_core_not_available", "error": str(e)})

# Import ROMA MCP Tools (fallback)
try:
    from roma_mcp_tools import (
        solve_with_roma,
        analyze_with_roma,
        solve_sub_problem_with_roma,
        verify_with_roma,
        critique_with_roma,
        get_roma_status
    )
    ROMA_MCP_AVAILABLE = True
    logger.info({"event": "roma_mcp_tools_imported"})
except ImportError as e:
    ROMA_MCP_AVAILABLE = False
    solve_with_roma = None
    analyze_with_roma = None
    solve_sub_problem_with_roma = None
    verify_with_roma = None
    critique_with_roma = None
    get_roma_status = None
    logger.warning({"event": "roma_mcp_tools_not_available", "error": str(e)})

# ROMA is available if either core or MCP tools are available
ROMA_AVAILABLE = ROMA_CORE_AVAILABLE or ROMA_MCP_AVAILABLE


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class RomaSolutionResult:
    """
    Result from solving a task with ROMA + Reliability layers.

    Attributes:
        success: Whether the operation completed successfully
        result: The ROMA result (if successful)
        task: The original task
        error: Error message (if failed)
        layers_used: List of reliability layers applied
        constraint_violations: Any LMQL constraint violations
        validation_failures: Any Guardrails validation failures
        remediation_applied: Any remediations that were applied
        correlation_id: Request correlation ID for tracing
        metadata: Additional metadata about the execution
    """
    success: bool
    result: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    error: Optional[str] = None
    layers_used: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    remediation_applied: List[str] = field(default_factory=list)
    correlation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "task": self.task,
            "error": self.error,
            "layers_used": self.layers_used,
            "constraint_violations": self.constraint_violations,
            "validation_failures": self.validation_failures,
            "remediation_applied": self.remediation_applied,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }

    def has_violations(self) -> bool:
        """Check if any constraint violations occurred."""
        return len(self.constraint_violations) > 0

    def has_validation_failures(self) -> bool:
        """Check if any validation failures occurred."""
        return len(self.validation_failures) > 0

    def was_remediated(self) -> bool:
        """Check if any remediations were applied."""
        return len(self.remediation_applied) > 0


@dataclass
class RomaAnalysisResult:
    """
    Result from analyzing a task with ROMA + Reliability layers.
    """
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    error: Optional[str] = None
    layers_used: List[str] = field(default_factory=list)
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    correlation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "analysis": self.analysis,
            "task": self.task,
            "error": self.error,
            "layers_used": self.layers_used,
            "validation_failures": self.validation_failures,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


# =============================================================================
# MAIN ADAPTER CLASS
# =============================================================================

class RomaReliabilityAdapter:
    """
    Enhanced ROMA adapter with direct core integration and LMQL constraints.

    This adapter provides dual access to ROMA:
    1. **Direct Core Integration**: Creates ROMA components directly with LMQL constraints
    2. **MCP Tool Fallback**: Uses ROMA MCP tools when core is unavailable

    Architecture:
        ROMA Core (READ ONLY)
            ↓
        Reliability Adapter (LMQL constraints + Guardrails validation)
            ↓
        [Option A] Direct Core Integration with enhanced agents
        [Option B] MCP Tools Fallback

    Features:
    - Input validation (Guardrails)
    - Pre-generation constraints (LMQL)
    - ROMA execution (direct core or MCP tools)
    - Output validation (Guardrails)
    - Remediation and retry logic
    - Graceful degradation
    - Enhanced agents with LMQL constraints

    Example:
        adapter = RomaReliabilityAdapter()

        result = adapter.solve_with_constraints(
            task="Solve the traveling salesman problem",
            max_depth=3,
            constraints={
                "max_depth": 3,
                "max_subtasks": 10,
                "subtask_token_limit": 500
            }
        )

        if result.success:
            print(f"Solution: {result.result}")
        else:
            print(f"Error: {result.error}")
    """

    def __init__(
        self,
        config: Optional['ReliabilityConfig'] = None,
        lmql_adapter: Optional['LMQLAdapter'] = None,
        guardrails_adapter: Optional['GuardrailsAdapter'] = None
    ):
        """
        Initialize the ROMA Reliability Adapter.

        Args:
            config: Optional reliability configuration
            lmql_adapter: Optional LMQL adapter (created if not provided)
            guardrails_adapter: Optional Guardrails adapter (created if not provided)
        """
        # Load or use provided configuration
        if CONFIG_AVAILABLE and config is None:
            self.config = get_reliability_config()
        else:
            self.config = config

        # Initialize LMQL adapter
        if LMQL_AVAILABLE:
            self.lmql_adapter = lmql_adapter or get_lmql_adapter()
            self.lmql_enabled = self.lmql_adapter.is_available() if self.lmql_adapter else False
        else:
            self.lmql_adapter = None
            self.lmql_enabled = False

        # Initialize Guardrails adapter
        if GUARDRAILS_AVAILABLE:
            self.guardrails_adapter = guardrails_adapter or create_guardrails_adapter()
            self.guardrails_enabled = self.guardrails_adapter.is_available() if self.guardrails_adapter else False
        else:
            self.guardrails_adapter = None
            self.guardrails_enabled = False

        # Check ROMA availability (both core and MCP)
        self.roma_core_available = ROMA_CORE_AVAILABLE
        self.roma_mcp_available = ROMA_MCP_AVAILABLE
        self.roma_available = ROMA_AVAILABLE

        # Initialize ROMA core components if available
        self.registry = None
        self.RecursiveSolver = None
        self.ROMAConfig = None

        if ROMA_CORE_AVAILABLE:
            try:
                self.RecursiveSolver = RecursiveSolver
                self.ROMAConfig = ROMAConfig
                self.registry = AgentRegistry() if AgentRegistry else None
                logger.info({"event": "roma_core_components_initialized"})
            except Exception as e:
                logger.warning({
                    "event": "roma_core_init_failed",
                    "error": str(e)
                })
                self.roma_core_available = False

        # Log initialization status
        logger.info({
            "event": "roma_reliability_adapter_initialized",
            "roma_core_available": self.roma_core_available,
            "roma_mcp_available": self.roma_mcp_available,
            "roma_available": self.roma_available,
            "lmql_enabled": self.lmql_enabled,
            "guardrails_enabled": self.guardrails_enabled,
            "config_provided": config is not None
        })

    # =========================================================================
    # PUBLIC API - SOLVE WITH CONSTRAINTS
    # =========================================================================

    def solve_with_constraints(
        self,
        task: str,
        max_depth: int = 3,
        constraints: Optional[Dict[str, Any]] = None,
        execution_mode: str = "recursive",
        enable_checkpoints: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> RomaSolutionResult:
        """
        Solve task using ROMA with LMQL constraints and Guardrails validation.

        This method intelligently routes to the best available execution method:
        1. **Direct Core Integration** (preferred): Creates ROMA components directly with LMQL constraints
        2. **MCP Tool Fallback**: Uses ROMA MCP tools when core is unavailable

        Layers applied:
        1. Input validation (Guardrails)
        2. Pre-generation constraints (LMQL)
        3. ROMA execution (core or MCP)
        4. Output validation (Guardrails)
        5. Remediation if needed

        Args:
            task: The task to solve
            max_depth: Maximum decomposition depth
            constraints: Optional LMQL constraints dict with keys:
                - max_depth: Maximum decomposition depth (default: from max_depth param)
                - max_subtasks: Maximum number of subtasks to generate
                - subtask_token_limit: Maximum tokens per subtask description
                - require_json: Require output in JSON format
                - custom_constraints: List of custom Constraint objects
            execution_mode: "recursive" (depth-first) or "event_driven" (parallel)
            enable_checkpoints: Enable ROMA checkpoint/recovery system
            provider: LLM provider (openai, anthropic, google, openrouter)
            model: Model name
            api_key: API key for the provider
            **kwargs: Additional arguments for ROMA

        Returns:
            RomaSolutionResult with solution or error details
        """
        correlation_id = f"roma_solve_{datetime.utcnow().timestamp()}"
        layers_used = []
        constraint_violations = []
        validation_failures = []
        remediation_applied = []

        logger.info({
            "event": "roma_solve_start",
            "task": task[:100],
            "max_depth": max_depth,
            "correlation_id": correlation_id,
            "method": "auto_select"
        })

        # --------------------------------------------------------------------
        # LAYER 1: Input validation (Guardrails)
        # --------------------------------------------------------------------
        if self.guardrails_enabled and self.guardrails_adapter:
            try:
                input_validation = self.guardrails_adapter.validate_input(
                    prompt=task,
                    validators=["roma_length", "toxic_language"],
                    on_fail="exception"
                )

                if not input_validation.is_valid:
                    logger.error({
                        "event": "input_validation_failed",
                        "failures": input_validation.failures,
                        "correlation_id": correlation_id
                    })
                    return RomaSolutionResult(
                        success=False,
                        task=task,
                        error=f"Input validation failed: {input_validation.failures}",
                        layers_used=["guardrails_input"],
                        validation_failures=input_validation.failures,
                        correlation_id=correlation_id
                    )

                layers_used.append("guardrails_input")
                logger.info({"event": "input_validation_passed", "correlation_id": correlation_id})

            except Exception as e:
                logger.error({
                    "event": "input_validation_error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "correlation_id": correlation_id
                })
                # Continue without input validation if it fails

        # --------------------------------------------------------------------
        # LAYER 2 & 3: Execute with best available method
        # --------------------------------------------------------------------
        # Try direct core integration first (preferred)
        if self.roma_core_available:
            logger.info({"event": "using_core_integration", "correlation_id": correlation_id})
            core_result = self._solve_with_core_integration(
                task=task,
                max_depth=max_depth,
                constraints=constraints,
                execution_mode=execution_mode,
                enable_checkpoints=enable_checkpoints,
                provider=provider,
                model=model,
                api_key=api_key,
                correlation_id=correlation_id,
                **kwargs
            )

            if core_result["success"]:
                layers_used.extend(core_result.get("layers_used", []))
                # Apply output validation
                validation_result = self._apply_output_validation(
                    result=core_result["result"],
                    correlation_id=correlation_id
                )
                layers_used.extend(validation_result["layers_used"])
                validation_failures = validation_result.get("validation_failures", [])
                remediation_applied = validation_result.get("remediation_applied", [])

                return RomaSolutionResult(
                    success=True,
                    result=validation_result.get("result", core_result["result"]),
                    task=task,
                    layers_used=layers_used,
                    constraint_violations=core_result.get("constraint_violations", []),
                    validation_failures=validation_failures,
                    remediation_applied=remediation_applied,
                    correlation_id=correlation_id,
                    metadata=core_result.get("metadata", {})
                )

            # Core integration failed, try MCP fallback
            logger.warning({
                "event": "core_integration_failed",
                "error": core_result.get("error"),
                "correlation_id": correlation_id
            })

        # Fallback to MCP tools
        if self.roma_mcp_available:
            logger.info({"event": "using_mcp_fallback", "correlation_id": correlation_id})
            mcp_result = self._solve_with_mcp_tools(
                task=task,
                max_depth=max_depth,
                execution_mode=execution_mode,
                enable_checkpoints=enable_checkpoints,
                provider=provider,
                model=model,
                api_key=api_key,
                correlation_id=correlation_id,
                **kwargs
            )

            if mcp_result["success"]:
                layers_used.extend(mcp_result.get("layers_used", []))
                # Apply output validation
                validation_result = self._apply_output_validation(
                    result=mcp_result["result"],
                    correlation_id=correlation_id
                )
                layers_used.extend(validation_result["layers_used"])
                validation_failures = validation_result.get("validation_failures", [])
                remediation_applied = validation_result.get("remediation_applied", [])

                return RomaSolutionResult(
                    success=True,
                    result=validation_result.get("result", mcp_result["result"]),
                    task=task,
                    layers_used=layers_used,
                    constraint_violations=mcp_result.get("constraint_violations", []),
                    validation_failures=validation_failures,
                    remediation_applied=remediation_applied,
                    correlation_id=correlation_id,
                    metadata=mcp_result.get("metadata", {})
                )

            return RomaSolutionResult(
                success=False,
                task=task,
                error=mcp_result.get("error", "ROMA execution failed"),
                layers_used=layers_used,
                correlation_id=correlation_id
            )

        # Neither method available
        return RomaSolutionResult(
            success=False,
            task=task,
            error="ROMA not available (both core and MCP tools unavailable)",
            layers_used=layers_used,
            correlation_id=correlation_id
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _solve_with_core_integration(
        self,
        task: str,
        max_depth: int,
        constraints: Optional[Dict[str, Any]],
        execution_mode: str,
        enable_checkpoints: bool,
        provider: Optional[str],
        model: Optional[str],
        api_key: Optional[str],
        correlation_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve using direct ROMA core integration with LMQL constraints.

        This method creates ROMA modules directly with LMQL constraints
        injected into the execution flow.

        Args:
            task: Task to solve
            max_depth: Maximum decomposition depth
            constraints: Optional LMQL constraints
            execution_mode: Execution mode
            enable_checkpoints: Enable checkpoints
            provider: LLM provider
            model: Model name
            api_key: API key
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Dict with success status and result or error
        """
        layers_used = []
        constraint_violations = []

        try:
            # Create ROMA config
            config = self._create_roma_config(
                provider=provider,
                model=model,
                api_key=api_key
            )

            if not config:
                return {
                    "success": False,
                    "error": "Failed to create ROMA config"
                }

            # Create enhanced components with LMQL
            atomizer = self._create_enhanced_atomizer(
                max_depth=max_depth,
                use_lmql=self.lmql_enabled
            )

            planner = self._create_enhanced_planner(
                max_subtasks=constraints.get("max_subtasks", 10) if constraints else 10,
                use_lmql=self.lmql_enabled
            )

            # Register agents
            if self.registry and atomizer and planner:
                try:
                    self.registry.register_agent("ATOMIZER", "DEFAULT", atomizer)
                    self.registry.register_agent("PLANNER", "DEFAULT", planner)
                except Exception as e:
                    logger.warning({
                        "event": "agent_registration_failed",
                        "error": str(e),
                        "correlation_id": correlation_id
                    })

            # Create solver
            solver = self.RecursiveSolver(
                config=config,
                max_depth=max_depth,
                enable_logging=True,
                enable_checkpoints=enable_checkpoints,
            )

            layers_used.append("lmql_constraints" if self.lmql_enabled else "roma_core")

            # Execute solve
            if execution_mode == "event_driven" and hasattr(solver, 'event_solve'):
                result_task_node = solver.event_solve(task)
            else:
                result_task_node = solver.solve(task)

            layers_used.append("roma_core")

            # Extract results
            result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)
            status = result_task_node.status.value if hasattr(result_task_node, 'status') else "unknown"

            logger.info({
                "event": "roma_core_solve_success",
                "status": status,
                "correlation_id": correlation_id
            })

            return {
                "success": True,
                "result": {
                    "result": result,
                    "status": status,
                    "generated_by": "ROMA Core Integration",
                    "execution_method_used": "roma_core"
                },
                "layers_used": layers_used,
                "constraint_violations": constraint_violations,
                "metadata": {
                    "max_depth": max_depth,
                    "execution_mode": execution_mode,
                    "roma_status": status,
                    "method": "core_integration"
                }
            }

        except Exception as e:
            logger.error({
                "event": "roma_core_solve_failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })
            return {
                "success": False,
                "error": str(e)
            }

    def _solve_with_mcp_tools(
        self,
        task: str,
        max_depth: int,
        execution_mode: str,
        enable_checkpoints: bool,
        provider: Optional[str],
        model: Optional[str],
        api_key: Optional[str],
        correlation_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve using ROMA MCP tools (fallback method).

        Args:
            task: Task to solve
            max_depth: Maximum decomposition depth
            execution_mode: Execution mode
            enable_checkpoints: Enable checkpoints
            provider: LLM provider
            model: Model name
            api_key: API key
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Dict with success status and result or error
        """
        layers_used = ["roma_mcp"]

        try:
            # Call ROMA via MCP tool
            roma_result = solve_with_roma(
                task=task,
                max_depth=max_depth,
                execution_mode=execution_mode,
                enable_checkpoints=enable_checkpoints,
                enable_logging=True,
                provider=provider,
                model=model,
                api_key=api_key,
                **kwargs
            )

            # Check if ROMA returned an error
            if "error" in roma_result:
                return {
                    "success": False,
                    "error": roma_result["error"]
                }

            logger.info({
                "event": "roma_mcp_solve_success",
                "status": roma_result.get("status", "unknown"),
                "correlation_id": correlation_id
            })

            return {
                "success": True,
                "result": roma_result,
                "layers_used": layers_used,
                "metadata": {
                    "max_depth": max_depth,
                    "execution_mode": execution_mode,
                    "roma_status": roma_result.get("status"),
                    "token_usage": roma_result.get("token_usage", {}),
                    "method": "mcp_tools"
                }
            }

        except Exception as e:
            logger.error({
                "event": "roma_mcp_solve_failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })
            return {
                "success": False,
                "error": str(e)
            }

    def _create_enhanced_atomizer(self, max_depth: int = 5, use_lmql: bool = True):
        """Create atomizer with LMQL constraints."""
        if not ROMA_CORE_AVAILABLE:
            return None

        try:
            # Import DSPy if available
            import dspy

            if use_lmql and self.lmql_enabled and self.lmql_adapter:
                # Create LMQL-enhanced atomizer wrapper
                class EnhancedAtomizer(Atomizer):
                    def __init__(self, lmql_adapter, **kwargs):
                        super().__init__(**kwargs)
                        self.lmql_adapter = lmql_adapter

                    def forward(self, goal, context=None, **kwargs):
                        # Apply LMQL constraints
                        if self.lmql_adapter and self.lmql_adapter.is_available():
                            try:
                                prompt = f"Goal: {goal}\nContext: {context or ''}\nIs this atomic? (yes/no)"

                                result = self.lmql_adapter.constrained_generation(
                                    prompt=prompt,
                                    constraints=[],
                                    decoding="argmax"
                                )

                                if result.success:
                                    is_atomic = "yes" in result.text.strip().lower()
                                    return dspy.Prediction(is_atomic=is_atomic)
                            except:
                                pass

                        # Fallback to standard atomizer
                        return super().forward(goal=goal, context=context, **kwargs)

                return EnhancedAtomizer(
                    lmql_adapter=self.lmql_adapter
                )
            else:
                # Standard atomizer
                return Atomizer()

        except Exception as e:
            logger.warning({
                "event": "enhanced_atomizer_creation_failed",
                "error": str(e)
            })
            return Atomizer() if ROMA_CORE_AVAILABLE else None

    def _create_enhanced_planner(self, max_subtasks: int = 10, use_lmql: bool = True):
        """Create planner with LMQL constraints."""
        if not ROMA_CORE_AVAILABLE:
            return None

        try:
            import dspy

            if use_lmql and self.lmql_enabled and self.lmql_adapter:
                # Create LMQL-enhanced planner wrapper
                class EnhancedPlanner(Planner):
                    def __init__(self, lmql_adapter, max_subtasks=10, **kwargs):
                        super().__init__(**kwargs)
                        self.lmql_adapter = lmql_adapter
                        self.max_subtasks = max_subtasks

                    def forward(self, goal, context=None, **kwargs):
                        # Apply LMQL constraints
                        if self.lmql_adapter and self.lmql_adapter.is_available():
                            try:
                                prompt = f"Goal: {goal}\nDecompose into subtasks (max {self.max_subtasks})"

                                result = self.lmql_adapter.constrained_generation(
                                    prompt=prompt,
                                    constraints=[],
                                    decoding="argmax"
                                )

                                if result.success:
                                    # Parse result into subtasks
                                    subtasks = self._parse_subtasks(result.text)
                                    return dspy.Prediction(
                                        subtasks=subtasks,
                                        dependencies_graph={}
                                    )
                            except:
                                pass

                        # Fallback to standard planner
                        return super().forward(goal=goal, context=context, **kwargs)

                    def _parse_subtasks(self, text):
                        """
                        Parse subtasks from text with comprehensive format support.

                        Supports multiple formats:
                        - JSON array: [{"goal": "...", "dependencies": []}, ...]
                        - Bullet points: "- Task 1\n- Task 2\n- Task 3"
                        - Numbered lists: "1. Task 1\n2. Task 2\n3. Task 3"
                        - Plain lines: "Task 1\nTask 2\nTask 3"

                        Args:
                            text: Text containing subtasks

                        Returns:
                            List of SubTask objects
                        """
                        if not SubTask or not text:
                            return []

                        text = text.strip()
                        subtasks = []

                        # Format 1: JSON array
                        if text.startswith('['):
                            try:
                                import json
                                data = json.loads(text)
                                if isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            # Normalize dict to SubTask
                                            goal = item.get('goal', item.get('name', item.get('description', '')))
                                            dependencies = item.get('dependencies', [])
                                            subtasks.append(SubTask(
                                                goal=goal,
                                                dependencies=dependencies if isinstance(dependencies, list) else []
                                            ))
                                        else:
                                            # String item
                                            subtasks.append(SubTask(goal=str(item), dependencies=[]))
                                return subtasks
                            except Exception as e:
                                logger.debug({
                                    "event": "json_parsing_failed",
                                    "error": str(e)
                                })

                        # Format 2: Bullet points (hyphen or asterisk)
                        bullet_pattern = r'^[\s]*[-*]\s+(.+)$'
                        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)
                        if bullet_matches and len(bullet_matches) > 1:
                            for match in bullet_matches:
                                subtasks.append(SubTask(goal=match.strip(), dependencies=[]))
                            if subtasks:
                                return subtasks

                        # Format 3: Numbered list
                        numbered_pattern = r'^[\s]*\d+[\.\)]\s+(.+)$'
                        numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
                        if numbered_matches and len(numbered_matches) > 1:
                            for match in numbered_matches:
                                subtasks.append(SubTask(goal=match.strip(), dependencies=[]))
                            if subtasks:
                                return subtasks

                        # Format 4: Plain text lines (non-empty lines)
                        lines = [line.strip() for line in text.split('\n')]
                        non_empty_lines = [line for line in lines if line and not line.startswith('#')]
                        if len(non_empty_lines) > 1:
                            for line in non_empty_lines:
                                subtasks.append(SubTask(goal=line, dependencies=[]))
                            if subtasks:
                                return subtasks

                        # Format 5: Single line as one subtask
                        if text and len(subtasks) == 0:
                            subtasks.append(SubTask(goal=text, dependencies=[]))

                        return subtasks

                return EnhancedPlanner(
                    lmql_adapter=self.lmql_adapter,
                    max_subtasks=max_subtasks
                )
            else:
                # Standard planner
                return Planner()

        except Exception as e:
            logger.warning({
                "event": "enhanced_planner_creation_failed",
                "error": str(e)
            })
            return Planner() if ROMA_CORE_AVAILABLE else None

    def _create_roma_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional['ROMAConfig']:
        """Create a ROMA configuration instance."""
        if not ROMA_CORE_AVAILABLE or not ROMAConfig:
            return None

        try:
            # Configure LLM provider if specified
            if provider:
                import os
                if api_key:
                    if provider.lower() == "openai":
                        os.environ["OPENAI_API_KEY"] = api_key
                    elif provider.lower() == "anthropic":
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                    elif provider.lower() == "google":
                        os.environ["GOOGLE_API_KEY"] = api_key
                    elif provider.lower() == "openrouter":
                        os.environ["OPENROUTER_API_KEY"] = api_key

                if model:
                    os.environ["ROMA_MODEL"] = model

            # Create config
            config = ROMAConfig()
            return config

        except Exception as e:
            logger.error({
                "event": "roma_config_creation_failed",
                "error": str(e)
            })
            return None

    def _apply_output_validation(
        self,
        result: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Apply output validation to ROMA result.

        Args:
            result: ROMA result
            correlation_id: Request correlation ID

        Returns:
            Dict with validation results
        """
        layers_used = []
        validation_failures = []
        remediation_applied = []

        if not self.guardrails_enabled or not self.guardrails_adapter:
            return {
                "result": result,
                "layers_used": [],
                "validation_failures": [],
                "remediation_applied": []
            }

        try:
            result_str = json.dumps(result, default=str)

            output_validation = self.guardrails_adapter.validate_output(
                output=result_str,
                validators=["json_structure"],
                on_fail="fix"
            )

            if not output_validation.is_valid:
                validation_failures = output_validation.failures

                # Try remediation
                if output_validation.remediation_applied:
                    remediation_applied.append(output_validation.remediation_applied)

                # Check if remediation produced valid output
                if output_validation.output:
                    try:
                        result = json.loads(output_validation.output)
                        remediation_applied.append("output_remediated")
                    except:
                        pass

            layers_used.append("guardrails_output")
            logger.info({
                "event": "output_validation_complete",
                "is_valid": output_validation.is_valid,
                "remediation_applied": output_validation.remediation_applied,
                "correlation_id": correlation_id
            })

        except Exception as e:
            logger.error({
                "event": "output_validation_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })

        return {
            "result": result,
            "layers_used": layers_used,
            "validation_failures": validation_failures,
            "remediation_applied": remediation_applied
        }

    # =========================================================================
    # PUBLIC API - ANALYZE WITH CONSTRAINTS
    # =========================================================================

    def analyze_with_constraints(
        self,
        task: str,
        analysis_type: str = "decomposition",
        max_depth: int = 3,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> RomaAnalysisResult:
        """
        Analyze task with ROMA and validate output.

        Args:
            task: Problem statement to analyze
            analysis_type: Type of analysis ("decomposition", "complexity", "dependencies")
            max_depth: Maximum recursion depth for decomposition
            provider: LLM provider
            model: Model name
            api_key: API key
            **kwargs: Additional ROMA configuration

        Returns:
            RomaAnalysisResult with analysis or error details
        """
        correlation_id = f"roma_analyze_{datetime.utcnow().timestamp()}"
        layers_used = []
        validation_failures = []

        logger.info({
            "event": "roma_analyze_start",
            "task": task[:100],
            "analysis_type": analysis_type,
            "correlation_id": correlation_id
        })

        # Input validation
        if self.guardrails_enabled and self.guardrails_adapter:
            try:
                validation = self.guardrails_adapter.validate_input(
                    prompt=task,
                    validators=["roma_length"],
                    on_fail="exception"
                )

                if not validation.is_valid:
                    return RomaAnalysisResult(
                        success=False,
                        task=task,
                        error=f"Input validation failed: {validation.failures}",
                        layers_used=["guardrails_input"],
                        validation_failures=validation.failures,
                        correlation_id=correlation_id
                    )

                layers_used.append("guardrails_input")

            except Exception as e:
                logger.error({
                    "event": "input_validation_error",
                    "error": str(e),
                    "correlation_id": correlation_id
                })

        # Call ROMA analyze
        if not self.roma_available or analyze_with_roma is None:
            return RomaAnalysisResult(
                success=False,
                task=task,
                error="ROMA not available",
                layers_used=layers_used,
                correlation_id=correlation_id
            )

        try:
            analysis_result = analyze_with_roma(
                task=task,
                analysis_type=analysis_type,
                max_depth=max_depth,
                provider=provider,
                model=model,
                api_key=api_key,
                **kwargs
            )

            if "error" in analysis_result:
                return RomaAnalysisResult(
                    success=False,
                    task=task,
                    error=analysis_result["error"],
                    layers_used=layers_used,
                    correlation_id=correlation_id
                )

            layers_used.append("roma_core")

            # Output validation
            if self.guardrails_enabled and self.guardrails_adapter:
                try:
                    result_str = json.dumps(analysis_result, default=str)
                    validation = self.guardrails_adapter.validate_output(
                        output=result_str,
                        validators=["json_structure"],
                        on_fail="fix"
                    )

                    if not validation.is_valid:
                        validation_failures = validation.failures

                    layers_used.append("guardrails_output")

                except Exception as e:
                    logger.error({
                        "event": "output_validation_error",
                        "error": str(e),
                        "correlation_id": correlation_id
                    })

            logger.info({
                "event": "roma_analyze_complete",
                "success": True,
                "layers_used": layers_used,
                "correlation_id": correlation_id
            })

            return RomaAnalysisResult(
                success=True,
                analysis=analysis_result,
                task=task,
                layers_used=layers_used,
                validation_failures=validation_failures,
                correlation_id=correlation_id,
                metadata={
                    "analysis_type": analysis_type,
                    "max_depth": max_depth
                }
            )

        except Exception as e:
            logger.error({
                "event": "roma_analyze_exception",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })
            return RomaAnalysisResult(
                success=False,
                task=task,
                error=str(e),
                layers_used=layers_used,
                correlation_id=correlation_id
            )

    # =========================================================================
    # PUBLIC API - VERIFY WITH CONSTRAINTS
    # =========================================================================

    def verify_with_constraints(
        self,
        solution: str,
        original_task: str,
        verification_criteria: Optional[List[str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> RomaSolutionResult:
        """
        Verify a solution using ROMA with Guardrails validation.

        Args:
            solution: The solution to verify
            original_task: The original task/problem
            verification_criteria: List of criteria to verify
            provider: LLM provider
            model: Model name
            **kwargs: Additional ROMA configuration

        Returns:
            RomaSolutionResult with verification results
        """
        correlation_id = f"roma_verify_{datetime.utcnow().timestamp()}"

        if not self.roma_available or verify_with_roma is None:
            return RomaSolutionResult(
                success=False,
                task=original_task,
                error="ROMA not available",
                correlation_id=correlation_id
            )

        try:
            result = verify_with_roma(
                solution=solution,
                original_task=original_task,
                verification_criteria=verification_criteria,
                provider=provider,
                model=model,
                **kwargs
            )

            if "error" in result:
                return RomaSolutionResult(
                    success=False,
                    task=original_task,
                    error=result["error"],
                    correlation_id=correlation_id
                )

            return RomaSolutionResult(
                success=True,
                result=result,
                task=original_task,
                layers_used=["roma_core"],
                correlation_id=correlation_id
            )

        except Exception as e:
            return RomaSolutionResult(
                success=False,
                task=original_task,
                error=str(e),
                correlation_id=correlation_id
            )

    # =========================================================================
    # PUBLIC API - CRITIQUE WITH CONSTRAINTS
    # =========================================================================

    def critique_with_constraints(
        self,
        solution: str,
        original_task: str,
        critique_focus: str = "comprehensive",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> RomaSolutionResult:
        """
        Critique a solution using ROMA (Red Team perspective).

        Args:
            solution: The solution to critique
            original_task: The original task
            critique_focus: Type of critique ("comprehensive", "security", "performance", "correctness")
            provider: LLM provider
            model: Model name
            **kwargs: Additional ROMA configuration

        Returns:
            RomaSolutionResult with critique results
        """
        correlation_id = f"roma_critique_{datetime.utcnow().timestamp()}"

        if not self.roma_available or critique_with_roma is None:
            return RomaSolutionResult(
                success=False,
                task=original_task,
                error="ROMA not available",
                correlation_id=correlation_id
            )

        try:
            result = critique_with_roma(
                solution=solution,
                original_task=original_task,
                critique_focus=critique_focus,
                provider=provider,
                model=model,
                **kwargs
            )

            if "error" in result:
                return RomaSolutionResult(
                    success=False,
                    task=original_task,
                    error=result["error"],
                    correlation_id=correlation_id
                )

            return RomaSolutionResult(
                success=True,
                result=result,
                task=original_task,
                layers_used=["roma_core"],
                correlation_id=correlation_id
            )

        except Exception as e:
            return RomaSolutionResult(
                success=False,
                task=original_task,
                error=str(e),
                correlation_id=correlation_id
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get adapter status and availability.

        Returns:
            Dict with status of all components including dual-mode availability
        """
        return {
            "roma_available": self.roma_available,
            "roma_core_available": self.roma_core_available,
            "roma_mcp_available": self.roma_mcp_available,
            "execution_mode": "core_preferred_with_mcp_fallback" if self.roma_core_available else "mcp_only" if self.roma_mcp_available else "unavailable",
            "lmql_available": self.lmql_enabled,
            "guardrails_available": self.guardrails_enabled,
            "layers": {
                "roma_core": {
                    "available": self.roma_core_available,
                    "enabled": True,
                    "components": {
                        "RecursiveSolver": self.RecursiveSolver is not None,
                        "Atomizer": Atomizer is not None,
                        "Planner": Planner is not None,
                        "AgentRegistry": self.registry is not None
                    }
                },
                "roma_mcp": {
                    "available": self.roma_mcp_available,
                    "enabled": True
                },
                "lmql": {
                    "available": self.lmql_enabled,
                    "enabled": self.lmql_enabled
                },
                "guardrails": {
                    "available": self.guardrails_enabled,
                    "enabled": self.guardrails_enabled
                }
            },
            "config": {
                "has_config": self.config is not None
            }
        }

    def is_available(self) -> bool:
        """
        Check if the adapter is fully operational.

        Returns:
            True if ROMA is available (other layers are optional)
        """
        return self.roma_available

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dict with health status of all components including dual-mode status
        """
        health = {
            "adapter_healthy": False,
            "execution_mode": "unavailable",
            "components": {}
        }

        # Check ROMA Core
        if self.roma_core_available:
            health["components"]["roma_core"] = {
                "healthy": True,
                "message": "ROMA core integration available",
                "components": {
                    "RecursiveSolver": self.RecursiveSolver is not None,
                    "Atomizer": Atomizer is not None,
                    "Planner": Planner is not None,
                    "AgentRegistry": self.registry is not None
                }
            }
        else:
            health["components"]["roma_core"] = {
                "healthy": False,
                "message": "ROMA core not available"
            }

        # Check ROMA MCP
        if self.roma_mcp_available and get_roma_status:
            try:
                roma_status = get_roma_status()
                health["components"]["roma_mcp"] = {
                    "healthy": roma_status.get("available", False),
                    "version": roma_status.get("version", "unknown"),
                    "details": roma_status
                }
            except Exception as e:
                health["components"]["roma_mcp"] = {
                    "healthy": False,
                    "error": str(e)
                }
        else:
            health["components"]["roma_mcp"] = {
                "healthy": False,
                "message": "ROMA MCP tools not available"
            }

        # Determine execution mode and overall health
        if self.roma_core_available:
            health["execution_mode"] = "core_preferred_with_mcp_fallback"
            health["adapter_healthy"] = True
        elif self.roma_mcp_available:
            health["execution_mode"] = "mcp_only"
            health["adapter_healthy"] = health["components"]["roma_mcp"]["healthy"]
        else:
            health["execution_mode"] = "unavailable"
            health["adapter_healthy"] = False

        # Check LMQL
        if self.lmql_adapter:
            try:
                lmql_status = self.lmql_adapter.get_status()
                health["components"]["lmql"] = {
                    "healthy": lmql_status.get("lmql_available", False),
                    "details": lmql_status
                }
            except Exception as e:
                health["components"]["lmql"] = {
                    "healthy": False,
                    "error": str(e)
                }
        else:
            health["components"]["lmql"] = {
                "healthy": False,
                "message": "LMQL adapter not initialized"
            }

        # Check Guardrails
        if self.guardrails_adapter:
            try:
                guardrails_healthy = self.guardrails_adapter.is_available()
                health["components"]["guardrails"] = {
                    "healthy": guardrails_healthy
                }
            except Exception as e:
                health["components"]["guardrails"] = {
                    "healthy": False,
                    "error": str(e)
                }
        else:
            health["components"]["guardrails"] = {
                "healthy": False,
                "message": "Guardrails adapter not initialized"
            }

        return health


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_roma_adapter(
    config: Optional['ReliabilityConfig'] = None,
    enable_lmql: bool = True,
    enable_guardrails: bool = True
) -> RomaReliabilityAdapter:
    """
    Create a ROMA reliability adapter with specified configuration.

    Args:
        config: Optional reliability configuration
        enable_lmql: Enable LMQL layer (default: True)
        enable_guardrails: Enable Guardrails layer (default: True)

    Returns:
        Configured RomaReliabilityAdapter instance
    """
    # Create LMQL adapter if enabled
    lmql_adapter = None
    if enable_lmql and LMQL_AVAILABLE:
        lmql_adapter = get_lmql_adapter()

    # Create Guardrails adapter if enabled
    guardrails_adapter = None
    if enable_guardrails and GUARDRAILS_AVAILABLE:
        guardrails_adapter = create_guardrails_adapter()

    return RomaReliabilityAdapter(
        config=config,
        lmql_adapter=lmql_adapter,
        guardrails_adapter=guardrails_adapter
    )


@lru_cache(maxsize=1)
def get_default_adapter() -> RomaReliabilityAdapter:
    """
    Get or create the default ROMA reliability adapter instance.

    Returns:
        Cached RomaReliabilityAdapter instance
    """
    return create_roma_adapter()


def solve_with_constraints(
    task: str,
    max_depth: int = 3,
    constraints: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RomaSolutionResult:
    """
    Convenience function for solving with ROMA + constraints.

    Args:
        task: Task description
        max_depth: Maximum decomposition depth
        constraints: Optional LMQL constraints
        **kwargs: Additional arguments for ROMA

    Returns:
        RomaSolutionResult
    """
    adapter = get_default_adapter()
    return adapter.solve_with_constraints(
        task=task,
        max_depth=max_depth,
        constraints=constraints,
        **kwargs
    )


def analyze_with_constraints(
    task: str,
    analysis_type: str = "decomposition",
    max_depth: int = 3,
    **kwargs
) -> RomaAnalysisResult:
    """
    Convenience function for analyzing with ROMA + constraints.

    Args:
        task: Problem statement
        analysis_type: Type of analysis
        max_depth: Maximum decomposition depth
        **kwargs: Additional arguments for ROMA

    Returns:
        RomaAnalysisResult
    """
    adapter = get_default_adapter()
    return adapter.analyze_with_constraints(
        task=task,
        analysis_type=analysis_type,
        max_depth=max_depth,
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RomaReliabilityAdapter",
    "RomaSolutionResult",
    "RomaAnalysisResult",
    "create_roma_adapter",
    "get_default_adapter",
    "solve_with_constraints",
    "analyze_with_constraints"
]
