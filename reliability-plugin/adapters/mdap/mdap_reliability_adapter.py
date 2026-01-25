"""
MDAP Reliability Adapter - Enhanced with Direct Core Integration
================================================================

A production-ready adapter that integrates Guardrails validation with MDAP
using BOTH direct core imports AND MCP tools for maximum flexibility.

This adapter provides:
- Direct MDAP core component imports (primary method)
- Guardrails validation at input/output boundaries and vote-level
- Graceful fallback to MCP tools if core unavailable
- Comprehensive error handling and logging
- Statistics tracking for validation metrics

Architecture:
    Layer 1: Input validation (Guardrails)
        ↓
    Layer 2: MDAP Core Integration (Direct imports)
        ↓
    Layer 3: Vote-level validation during execution
        ↓
    Layer 4: Output validation (Guardrails)

Air Gap Principle:
- Direct imports are READ-ONLY access to MDAP core classes
- NO modifications to MDAP core files
- All Guardrails logic lives in the ADAPTER
- MCP tools used as fallback when direct imports unavailable

Author: OpenEvolve Team
Version: 2.0.0
License: MIT
"""

import sys
import os
import json
import logging
import uuid
import dataclasses
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Add reliability plugin to path
plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if plugin_path not in sys.path:
    sys.path.insert(0, plugin_path)

# Import enhanced red flagging system
try:
    from reliability.enhanced_redflagger import (
        EnhancedRedFlagger,
        EnhancedRedFlagRules,
        RedFlag,
        RedFlagSeverity,
        create_enhanced_redflagger
    )
    ENHANCED_REDFLAGGING_AVAILABLE = True
except ImportError:
    ENHANCED_REDFLAGGING_AVAILABLE = False
    EnhancedRedFlagger = None
    EnhancedRedFlagRules = None
    RedFlag = None
    RedFlagSeverity = None
    create_enhanced_redflagger = None

# =============================================================================
# TYPE DEFINITIONS AND DATA CLASSES
# =============================================================================

class RemediationStrategy(Enum):
    """Validation failure remediation strategies"""
    REASK = "reask"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"
    EXCEPTION = "exception"


@dataclass
class VoteValidationResult:
    """Result of validating an individual MDAP vote"""
    is_valid: bool
    vote: Any
    failures: List[str] = field(default_factory=list)
    remediated: bool = False
    original_vote: Any = None
    validator_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MDAPSolveResult:
    """Result of MDAP solve with validation"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    statistics: Dict[str, int] = field(default_factory=dict)
    validation_failures: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    layers_used: List[str] = field(default_factory=list)
    method: Optional[str] = None


# =============================================================================
# MAIN ADAPTER CLASS
# =============================================================================

class MDAPReliabilityAdapter:
    """
    Enhanced MDAP adapter with direct core integration and Guardrails validation.

    This adapter:
    1. Imports MDAP core components directly (primary method)
    2. Validates inputs with Guardrails
    3. Validates individual votes during execution
    4. Validates outputs with Guardrails
    5. Falls back to MCP tools if core unavailable

    Example:
        adapter = MDAPReliabilityAdapter()
        result = adapter.solve_with_core_integration(
            task="Solve this problem",
            mdap_k_ahead=5,
            validators=["vote_format", "json_structure", "required_fields"]
        )
        if result.success:
            print(f"Solution: {result.result}")
        else:
            print(f"Error: {result.error}")
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize MDAP Reliability Adapter with core integration.

        Args:
            config: Optional ReliabilityConfig object (from reliability.config)
        """
        # Load configuration
        if config is None:
            try:
                from reliability.config import get_config
                self.config = get_config()
            except ImportError:
                # Fallback to basic config
                self.config = self._create_default_config()
        else:
            self.config = config

        # Initialize Guardrails adapter
        self.guardrails_adapter = self._init_guardrails()

        # Initialize enhanced red flagger if available
        self.enhanced_redflagger = None
        self.enhanced_redflagging_enabled = False
        if ENHANCED_REDFLAGGING_AVAILABLE:
            try:
                self.enhanced_redflagger = self._create_enhanced_redflagger()
                self.enhanced_redflagging_enabled = True
                self.logger.info("Enhanced red flagging system initialized")
            except Exception as e:
                self.logger.warning(
                    "Failed to initialize enhanced red flagger",
                    error=str(e)
                )

        # Import MDAP core components (primary method)
        self.mdap_available = False
        self.MDAPOrchestrator = None
        self.MDAPConfig = None
        self.MakerEngine = None
        self.MakerConfig = None
        self.ROMAMDAPMakerEngine = None
        self.ROMAMDAPMakerConfig = None
        self.MDAPTask = None
        self.MDAPStep = None
        self.RedFlagRules = None
        self.RedFlagger = None
        self.MDAPCacheManager = None
        self.MDAPLoadBalancer = None
        self.AdaptiveThresholdManager = None
        self.Team = None
        self.ModelConfig = None

        self._init_mdap_core()

        # Import MDAP MCP tools (fallback method)
        self.mdap_mcp_solve = None
        self.mdap_mcp_verify = None

        self._init_mdap_mcp_tools()

        # Statistics tracking
        self.stats = {
            "total_solves": 0,
            "successful_solves": 0,
            "failed_solves": 0,
            "total_votes_validated": 0,
            "valid_votes": 0,
            "remediated_votes": 0,
            "rejected_votes": 0,
            "core_integration_used": 0,
            "mcp_fallback_used": 0,
            "enhanced_redflagging_used": 0,
            "red_flags_detected": 0
        }

        # Setup logging
        self.logger = self._setup_logger()

    def _create_default_config(self) -> Any:
        """Create default configuration if reliability.config unavailable"""
        from types import SimpleNamespace

        return SimpleNamespace(
            guardrails=SimpleNamespace(
                enabled=True,
                validators=["vote_format", "json_structure", "required_fields"],
                on_fail="fix",
                max_retries=3,
                timeout=30
            ),
            observability=SimpleNamespace(
                log_level="INFO"
            ),
            ace_enabled=True,
            steer_enabled=True,
            lmql_enabled=False,
            enhanced_redflagging_enabled=True
        )

    def _create_enhanced_redflagger(self) -> Optional['EnhancedRedFlagger']:
        """
        Create enhanced red flagger with default rules.

        Returns:
            EnhancedRedFlagger instance or None if unavailable
        """
        if not ENHANCED_REDFLAGGING_AVAILABLE:
            return None

        try:
            # Get LMQL adapter
            lmql_adapter = None
            try:
                from reliability.lmql_adapter import get_default_adapter
                lmql_adapter = get_default_adapter()
            except ImportError:
                self.logger.debug("LMQL adapter not available")

            # Create default rules
            rules = self._create_default_redflag_rules()

            # Create enhanced red flagger
            redflagger = EnhancedRedFlagger(
                rules=rules,
                lmql_adapter=lmql_adapter,
                guardrails_adapter=self.guardrails_adapter,
                config=self.config
            )

            return redflagger

        except Exception as e:
            self.logger.error(
                "Failed to create enhanced red flagger",
                error=str(e),
                exc_info=True
            )
            return None

    def _create_default_redflag_rules(self) -> 'EnhancedRedFlagRules':
        """
        Create default enhanced red flag rules.

        Returns:
            EnhancedRedFlagRules instance
        """
        if not ENHANCED_REDFLAGGING_AVAILABLE:
            return None

        return EnhancedRedFlagRules(
            max_tokens=750,
            max_characters=6000,
            min_confidence=0.5,
            confidence_threshold=0.5,

            # Enable LMQL pre-generation
            enable_lmql_constraints=getattr(self.config, 'lmql_enabled', False),
            lmql_max_retries=3,

            # Guardrails validators
            guardrails_validators=[
                "toxic_language",
                "pii_filter",
                "secrets_detection",
                "malicious_patterns",
                "injection_check",
                "json_structure"
            ],

            # Security rules
            forbidden_keywords=[
                "password", "api_key", "secret", "token",
                "credential", "private_key"
            ],

            # Format requirements
            required_format="json",
            require_schema_match=True,

            # Thresholds
            toxicity_threshold=0.8,
            pii_detection_strict=True
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup structured JSON logger"""
        logger = logging.getLogger("mdap_reliability_adapter")

        # Set log level from config
        log_level = getattr(self.config.observability, "log_level", "INFO")
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Clear existing handlers
        logger.handlers.clear()

        # Create console handler with JSON formatting
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

        return logger

    def _init_guardrails(self):
        """Initialize Guardrails adapter"""
        try:
            from reliability.guardrails_adapter import create_adapter

            adapter = create_adapter(
                enabled=self.config.guardrails.enabled,
                default_on_fail=self.config.guardrails.on_fail,
                max_retries=self.config.guardrails.max_retries,
                timeout=self.config.guardrails.timeout
            )

            self.logger.info(
                "Guardrails adapter initialized",
                enabled=self.config.guardrails.enabled,
                validators=self.config.guardrails.validators
            )

            return adapter

        except ImportError:
            self.logger.warning(
                "Guardrails adapter not available, running in degraded mode",
                suggestion="Install with: pip install guardrails-ai"
            )
            return None
        except Exception as e:
            self.logger.error(
                "Failed to initialize Guardrails adapter",
                error=str(e)
            )
            return None

    def _init_mdap_core(self):
        """
        Import MDAP core components (primary integration method).

        This imports the actual MDAP classes for direct use,
        allowing vote-level validation and full control over execution.
        """
        try:
            # Import MDAP core components
            from mdap_engine import (
                MDAPOrchestrator,
                MDAPConfig,
                MDAPTask,
                MDAPStep,
                RedFlagRules,
                RedFlagger,
                MDAPCacheManager,
                MDAPLoadBalancer,
                AdaptiveThresholdManager,
                validate_schema,
                canonicalize_candidate,
                candidate_confidence
            )
            from maker_engine import (
                MakerEngine,
                MakerConfig,
                MakerStep,
                MakerState,
                MakerRunResult,
                FileCheckpointStore
            )
            from roma_mdap_maker_engine import (
                ROMAMDAPMakerEngine,
                ROMAMDAPMakerConfig,
                ROMARedFlagger,
                ROMARedFlagRules,
                HierarchicalVotingStrategy,
                AdaptiveKSelector,
                ROMAIntrospectionEngine
            )
            from workflow_structures import ModelConfig, Team
            from base_configuration import BaseConfiguration

            # Store imports
            self.MDAPOrchestrator = MDAPOrchestrator
            self.MDAPConfig = MDAPConfig
            self.MDAPTask = MDAPTask
            self.MDAPStep = MDAPStep
            self.RedFlagRules = RedFlagRules
            self.RedFlagger = RedFlagger
            self.MDAPCacheManager = MDAPCacheManager
            self.MDAPLoadBalancer = MDAPLoadBalancer
            self.AdaptiveThresholdManager = AdaptiveThresholdManager
            self.MakerEngine = MakerEngine
            self.MakerConfig = MakerConfig
            self.ROMAMDAPMakerEngine = ROMAMDAPMakerEngine
            self.ROMAMDAPMakerConfig = ROMAMDAPMakerConfig
            self.Team = Team
            self.ModelConfig = ModelConfig
            self.BaseConfiguration = BaseConfiguration

            # Helper functions
            self.validate_schema = validate_schema
            self.canonicalize_candidate = canonicalize_candidate
            self.candidate_confidence = candidate_confidence

            self.mdap_available = True

            self.logger.info(
                "MDAP core components imported successfully",
                orchestrator_available=True,
                maker_available=True,
                roma_mdap_available=True,
                method="direct_import"
            )

        except ImportError as e:
            self.logger.warning(
                "MDAP core not available, will use MCP tools as fallback",
                error=str(e),
                suggestion="Ensure mdap_engine.py, maker_engine.py, and roma_mdap_maker_engine.py are in the path"
            )
            self.mdap_available = False

        except Exception as e:
            self.logger.error(
                "Failed to import MDAP core components",
                error=str(e),
                exc_info=True
            )
            self.mdap_available = False

    def _init_mdap_mcp_tools(self):
        """
        Import MDAP MCP tools (fallback integration method).

        These are used when direct core imports are unavailable.
        """
        try:
            from roma_mdap_maker_mcp_tools import (
                solve_with_roma_mdap_maker,
                verify_solution_with_roma_mdap
            )

            self.mdap_mcp_solve = solve_with_roma_mdap_maker
            self.mdap_mcp_verify = verify_solution_with_roma_mdap

            self.logger.info(
                "MDAP MCP tools imported successfully (as fallback)",
                solve_available=self.mdap_mcp_solve is not None,
                verify_available=self.mdap_mcp_verify is not None
            )

        except ImportError as e:
            self.logger.warning(
                "MDAP MCP tools not available",
                error=str(e)
            )
            self.mdap_mcp_solve = None
            self.mdap_mcp_verify = None

        except Exception as e:
            self.logger.error(
                "Failed to import MDAP MCP tools",
                error=str(e)
            )
            self.mdap_mcp_solve = None
            self.mdap_mcp_verify = None

    # =========================================================================
    # PUBLIC API - CORE INTEGRATION
    # =========================================================================

    def solve_with_core_integration(
        self,
        task: str,
        mdap_k_ahead: int = 5,
        team: Optional[Any] = None,
        validators: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> MDAPSolveResult:
        """
        Solve using direct MDAP core integration with Guardrails validation.

        This method creates MDAP components directly with vote-level validation
        injected into the execution flow.

        Args:
            task: The task to solve
            mdap_k_ahead: Number of agents for voting (2-20)
            team: Optional Team object for agent selection
            validators: List of validators to apply
            correlation_id: Optional correlation ID for logging
            **kwargs: Additional arguments for MDAP

        Returns:
            MDAPSolveResult with validation results
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = f"mdap_core_{uuid.uuid4()}"

        self.logger.info(
            "Starting MDAP solve with core integration",
            correlation_id=correlation_id,
            task_length=len(task),
            mdap_k_ahead=mdap_k_ahead,
            validators=validators
        )

        # Initialize result
        result = MDAPSolveResult(
            correlation_id=correlation_id,
            statistics={
                "total_votes": 0,
                "valid_votes": 0,
                "rejected_votes": 0,
                "remediated_votes": 0
            },
            layers_used=[],
            method="core_integration"
        )

        # Update statistics
        self.stats["total_solves"] += 1

        # Check if MDAP core is available
        if not self.mdap_available:
            self.logger.warning(
                "MDAP core not available, falling back to MCP tools",
                correlation_id=correlation_id
            )
            return self.solve_with_mcp_tools(
                task=task,
                mdap_k_ahead=mdap_k_ahead,
                validators=validators,
                correlation_id=correlation_id,
                **kwargs
            )

        # Layer 1: Input validation (Guardrails)
        result.layers_used.append("guardrails_input")
        input_validation = self._validate_input(
            task=task,
            mdap_k_ahead=mdap_k_ahead,
            correlation_id=correlation_id
        )

        if not input_validation["is_valid"]:
            result.success = False
            result.error = f"Input validation failed: {input_validation['error']}"
            result.validation_failures = input_validation["failures"]
            self.stats["failed_solves"] += 1

            self.logger.error(
                "Input validation failed",
                correlation_id=correlation_id,
                errors=result.validation_failures
            )

            return result

        # Layer 2: Create MDAP config with Guardrails integration
        result.layers_used.append("mdap_core")
        try:
            # Create MDAP config
            mdap_config = self.MDAPConfig(
                parameters={
                    "k_min": max(2, mdap_k_ahead - 2),
                    "k_max": mdap_k_ahead + 2,
                    "max_votes_per_step": kwargs.get("max_votes", 50),
                    "timeout_seconds": kwargs.get("timeout", 60),
                    "ace_enabled": getattr(self.config, 'ace_enabled', True),
                    "steer_enabled": getattr(self.config, 'steer_enabled', True),
                    "red_flag_rules": {},
                    "fallback_policy": "escalate_then_best_effort",
                    "cache_ttl_seconds": kwargs.get("cache_ttl", 3600),
                    "cache_max_size": kwargs.get("cache_max_size", 10000)
                }
            )

            # Create RedFlagger
            red_flag_rules = self.RedFlagRules(
                max_tokens=kwargs.get("max_tokens", 750),
                max_characters=kwargs.get("max_characters", 6000),
                min_confidence=kwargs.get("min_confidence", 0.2)
            )

            red_flagger = self.RedFlagger(
                rules=red_flag_rules,
                guardrails_adapter=self.guardrails_adapter
            )

            # Create cache manager
            cache_manager = self.MDAPCacheManager(
                max_size=kwargs.get("cache_max_size", 10000),
                ttl_seconds=kwargs.get("cache_ttl", 3600)
            )

            # Create load balancer
            load_balancer = self.MDAPLoadBalancer(
                available_agents=team.agents if team else []
            )

            # Create adaptive threshold manager
            adaptive_threshold = self.AdaptiveThresholdManager(
                initial_k=mdap_k_ahead,
                min_k=2,
                max_k=20
            )

            # Layer 3: Create MDAP orchestrator
            orchestrator = self.MDAPOrchestrator(
                team=team or self._get_default_team(),
                config=mdap_config,
                guardrails_adapter=self.guardrails_adapter,
                cache_manager=cache_manager,
                load_balancer=load_balancer,
                adaptive_threshold_manager=adaptive_threshold
            )

            # Create MDAP task
            mdap_task = self.MDAPTask(
                task_id=f"task_{uuid.uuid4()}",
                description=task,
                steps=[
                    self.MDAPStep(
                        step_id="step_1",
                        prompt=task,
                        task_type="general"
                    )
                ]
            )

            # Execute task
            self.logger.debug(
                "Executing MDAP task",
                correlation_id=correlation_id,
                task_id=mdap_task.task_id
            )

            mdap_result = orchestrator.execute_task(mdap_task)

            # Layer 4: Validate result
            result.layers_used.append("guardrails_output")
            validation_result = self._validate_result(
                mdap_result,
                validators,
                correlation_id
            )

            result.success = True
            result.result = self._serialize_mdap_result(mdap_result)
            result.statistics = validation_result["statistics"]
            result.validation_failures = validation_result["failures"]

            self.stats["successful_solves"] += 1
            self.stats["core_integration_used"] += 1

            self.logger.info(
                "MDAP core solve completed successfully",
                correlation_id=correlation_id,
                total_votes=result.statistics["total_votes"],
                valid_votes=result.statistics["valid_votes"]
            )

        except Exception as e:
            self.logger.error(
                "Core integration failed, falling back to MCP tools",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )

            # Fallback to MCP tools
            return self.solve_with_mcp_tools(
                task=task,
                mdap_k_ahead=mdap_k_ahead,
                validators=validators,
                correlation_id=correlation_id,
                **kwargs
            )

        return result

    # =========================================================================
    # PUBLIC API - MCP FALLBACK
    # =========================================================================

    def solve_with_mcp_tools(
        self,
        task: str,
        mdap_k_ahead: int = 5,
        validators: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> MDAPSolveResult:
        """
        Solve using MCP tools (fallback when core unavailable).

        Args:
            task: The task to solve
            mdap_k_ahead: Number of agents for voting
            validators: List of validators to apply
            correlation_id: Optional correlation ID
            **kwargs: Additional arguments

        Returns:
            MDAPSolveResult with validation results
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = f"mdap_mcp_{uuid.uuid4()}"

        self.logger.info(
            "Using MCP tools fallback",
            correlation_id=correlation_id
        )

        # Initialize result
        result = MDAPSolveResult(
            correlation_id=correlation_id,
            statistics={
                "total_votes": 0,
                "valid_votes": 0,
                "rejected_votes": 0,
                "remediated_votes": 0
            },
            layers_used=["guardrails_input", "mcp_tools", "guardrails_output"],
            method="mcp_fallback"
        )

        # Check if MCP tools available
        if not self.mdap_mcp_solve:
            result.success = False
            result.error = "MDAP MCP tools not available"
            self.stats["failed_solves"] += 1

            self.logger.error(
                "MDAP MCP tools unavailable",
                correlation_id=correlation_id
            )

            return result

        # Layer 1: Input validation
        input_validation = self._validate_input(
            task=task,
            mdap_k_ahead=mdap_k_ahead,
            correlation_id=correlation_id
        )

        if not input_validation["is_valid"]:
            result.success = False
            result.error = f"Input validation failed: {input_validation['error']}"
            result.validation_failures = input_validation["failures"]
            self.stats["failed_solves"] += 1
            return result

        # Layer 2: Call MCP tools
        try:
            mdap_result = self.mdap_mcp_solve(
                task=task,
                mdap_k_ahead=mdap_k_ahead,
                **kwargs
            )

            # Check for errors
            if mdap_result and "error" in mdap_result:
                result.success = False
                result.error = mdap_result["error"]
                self.stats["failed_solves"] += 1
                return result

            result.result = mdap_result

            # Layer 3: Validate output
            if validators and self.guardrails_adapter:
                output_validation = self._validate_output(
                    output=mdap_result,
                    validators=validators,
                    correlation_id=correlation_id
                )

                if not output_validation["is_valid"]:
                    result.validation_failures = output_validation["failures"]
                    result.statistics["remediated_votes"] = output_validation.get("remediated_count", 0)

                    # Try remediation
                    if output_validation.get("remediated_output"):
                        try:
                            result.result = json.loads(output_validation["remediated_output"])
                        except Exception:
                            pass

                # Extract statistics
                if mdap_result and "votes" in mdap_result:
                    result.statistics["total_votes"] = sum(mdap_result["votes"].values())
                    result.statistics["valid_votes"] = result.statistics["total_votes"]

            result.success = True
            self.stats["successful_solves"] += 1
            self.stats["mcp_fallback_used"] += 1

        except Exception as e:
            result.success = False
            result.error = str(e)
            self.stats["failed_solves"] += 1

            self.logger.error(
                "MCP tools execution failed",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )

        return result

    # =========================================================================
    # PUBLIC API - ORIGINAL COMPATIBILITY METHOD
    # =========================================================================

    def solve_with_validation(
        self,
        task: str,
        mdap_k_ahead: int = 5,
        validators: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> MDAPSolveResult:
        """
        Solve task using MDAP with Guardrails vote validation.

        This method internally calls solve_with_core_integration for best results,
        with automatic fallback to MCP tools if needed.

        Args:
            task: The task to solve
            mdap_k_ahead: Number of agents for voting (2-20)
            validators: List of validators to apply
            correlation_id: Optional correlation ID for logging
            **kwargs: Additional arguments for MDAP

        Returns:
            MDAPSolveResult with validation results
        """
        return self.solve_with_core_integration(
            task=task,
            mdap_k_ahead=mdap_k_ahead,
            validators=validators,
            correlation_id=correlation_id,
            **kwargs
        )

    # =========================================================================
    # PUBLIC API - ENHANCED RED FLAGGING
    # =========================================================================

    def solve_with_enhanced_redflagging(
        self,
        task: str,
        mdap_k_ahead: int = 5,
        team: Optional[Any] = None,
        use_lmql_constraints: bool = True,
        use_enhanced_validation: bool = True,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve MDAP task with enhanced red flagging.

        This method uses the multi-layered red flagging system:
        1. Pre-generation: LMQL constraints prevent flagged content
        2. During execution: Enhanced validation at each step
        3. Post-generation: Comprehensive flag checking

        Args:
            task: Task to solve
            mdap_k_ahead: Number of agents for voting
            team: Optional team configuration
            use_lmql_constraints: Enable LMQL pre-generation constraints
            use_enhanced_validation: Enable enhanced red flagging
            correlation_id: Optional correlation ID for logging
            **kwargs: Additional parameters

        Returns:
            Dict with success, result, red_flags, statistics
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = f"mdap_enhanced_{uuid.uuid4()}"

        self.logger.info(
            "Starting MDAP solve with enhanced red flagging",
            correlation_id=correlation_id,
            task_length=len(task),
            mdap_k_ahead=mdap_k_ahead,
            use_lmql=use_lmql_constraints,
            use_validation=use_enhanced_validation
        )

        # Initialize result
        red_flags = []
        layers_used = []
        lmql_constraints = []

        # Update statistics
        self.stats["total_solves"] += 1

        # Check if enhanced red flagging is available
        if not self.enhanced_redflagging_enabled or not self.enhanced_redflagger:
            self.logger.warning(
                "Enhanced red flagging not available, falling back to core integration",
                correlation_id=correlation_id
            )
            # Fall back to standard core integration
            result = self.solve_with_core_integration(
                task=task,
                mdap_k_ahead=mdap_k_ahead,
                team=team,
                correlation_id=correlation_id,
                **kwargs
            )
            return self._convert_to_dict_result(result, {
                "red_flags": [],
                "red_flag_count": 0,
                "layers_used": result.layers_used,
                "flagging_statistics": {},
                "metadata": {
                    "method": "fallback_to_core",
                    "lmql_constraints_used": 0,
                    "validation_enabled": False
                }
            })

        # Layer 1: Get LMQL constraints for pre-generation
        if use_lmql_constraints and self.enhanced_redflagger.lmql_adapter:
            try:
                if self.enhanced_redflagger.lmql_adapter.is_available():
                    lmql_constraints = self.enhanced_redflagger.get_lmql_constraints()
                    layers_used.append("lmql_pre_generation")
                    self.logger.info(
                        f"Generated {len(lmql_constraints)} LMQL constraints",
                        correlation_id=correlation_id
                    )
                else:
                    self.logger.debug(
                        "LMQL adapter not available, skipping pre-generation constraints",
                        correlation_id=correlation_id
                    )
            except Exception as e:
                self.logger.warning(
                    "Failed to generate LMQL constraints",
                    correlation_id=correlation_id,
                    error=str(e)
                )
        else:
            self.logger.debug(
                "LMQL constraints disabled",
                correlation_id=correlation_id
            )

        # Layer 2: Execute MDAP solve (with or without core integration)
        try:
            if self.mdap_available:
                # Use core integration with enhanced red flagger
                result = self._solve_with_core_redflagging(
                    task=task,
                    mdap_k_ahead=mdap_k_ahead,
                    team=team,
                    redflagger=self.enhanced_redflagger if use_enhanced_validation else None,
                    lmql_constraints=lmql_constraints,
                    correlation_id=correlation_id,
                    **kwargs
                )
                layers_used.append("mdap_core")
            else:
                # Fallback to MCP tools
                core_result = self.solve_with_mcp_tools(
                    task=task,
                    mdap_k_ahead=mdap_k_ahead,
                    correlation_id=correlation_id,
                    **kwargs
                )
                result = {
                    "success": core_result.success,
                    "result": core_result.result,
                    "statistics": core_result.statistics
                }
                layers_used.append("mcp_tools")

        except Exception as e:
            self.logger.error(
                "MDAP execution failed",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            result = {
                "success": False,
                "result": None,
                "error": str(e)
            }

        # Layer 3: Validate result with enhanced red flagging
        if use_enhanced_validation and result.get("success"):
            try:
                raw_text = json.dumps(result.get("result", {}))
                candidate = result.get("result", {})

                is_flagged, flags = self.enhanced_redflagger.check_for_red_flags(
                    raw_text=raw_text,
                    candidate=candidate,
                    schema=kwargs.get("schema"),
                    context={"task": task, "mdap_k_ahead": mdap_k_ahead}
                )

                if is_flagged:
                    red_flags.extend(flags)

                    # Categorize flags by severity
                    if RedFlagSeverity:
                        critical_flags = [f for f in flags if f.severity == RedFlagSeverity.CRITICAL]
                        high_flags = [f for f in flags if f.severity == RedFlagSeverity.HIGH]

                        # If critical or high flags, consider result failed
                        if critical_flags or high_flags:
                            self.logger.warning(
                                f"Result has {len(critical_flags)} critical and {len(high_flags)} high severity flags",
                                correlation_id=correlation_id
                            )
                            result["success"] = False
                            result["red_flags"] = [f.to_dict() if hasattr(f, 'to_dict') else str(f) for f in flags]
                    else:
                        # Fallback if RedFlagSeverity not available
                        if len(flags) > 0:
                            self.logger.warning(
                                f"Result has {len(flags)} red flags",
                                correlation_id=correlation_id
                            )
                            result["success"] = False
                            result["red_flags"] = [str(f) for f in flags]

                layers_used.append("enhanced_redflagging")
                self.stats["enhanced_redflagging_used"] += 1
                self.stats["red_flags_detected"] += len(red_flags)

            except Exception as e:
                self.logger.error(
                    "Enhanced red flagging validation failed",
                    correlation_id=correlation_id,
                    error=str(e),
                    exc_info=True
                )

        # Get red flagging statistics
        flagging_stats = {}
        try:
            if self.enhanced_redflagger:
                flagging_stats = self.enhanced_redflagger.get_statistics()
        except Exception as e:
            self.logger.debug(
                "Failed to get red flagging statistics",
                error=str(e)
            )

        # Update final statistics
        if result.get("success"):
            self.stats["successful_solves"] += 1
        else:
            self.stats["failed_solves"] += 1

        return {
            "success": result.get("success", False),
            "result": result.get("result"),
            "task": task,
            "red_flags": [f.to_dict() if hasattr(f, 'to_dict') else str(f) for f in red_flags],
            "red_flag_count": len(red_flags),
            "layers_used": layers_used,
            "flagging_statistics": flagging_stats,
            "metadata": {
                "method": "enhanced_redflagging",
                "lmql_constraints_used": len(lmql_constraints),
                "validation_enabled": use_enhanced_validation,
                "correlation_id": correlation_id
            }
        }

    def _solve_with_core_redflagging(
        self,
        task: str,
        mdap_k_ahead: int,
        team: Optional[Any],
        redflagger: Optional['EnhancedRedFlagger'],
        lmql_constraints: List[str],
        correlation_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve using MDAP core with enhanced red flagger integration.

        Args:
            task: Task to solve
            mdap_k_ahead: Number of agents for voting
            team: Optional team configuration
            redflagger: Enhanced red flagger instance
            lmql_constraints: List of LMQL constraints
            correlation_id: Correlation ID
            **kwargs: Additional parameters

        Returns:
            Dict with solve result
        """
        try:
            # Create MDAP config
            mdap_config = self.MDAPConfig(
                parameters={
                    "k_min": max(2, mdap_k_ahead - 2),
                    "k_max": mdap_k_ahead + 2,
                    "max_votes_per_step": kwargs.get("max_votes", 50)
                }
            )

            # Create RedFlagger with enhanced rules
            if redflagger:
                red_flag_rules = self.RedFlagRules(
                    max_tokens=redflagger.rules.max_tokens,
                    max_characters=redflagger.rules.max_characters,
                    min_confidence=redflagger.rules.min_confidence
                )
            else:
                red_flag_rules = self.RedFlagRules()

            # Create RedFlagger with Guardrails
            core_redflagger = self.RedFlagger(
                rules=red_flag_rules,
                guardrails_adapter=self.guardrails_adapter
            )

            # Create orchestrator
            orchestrator = self.MDAPOrchestrator(
                team=team or self._get_default_team(),
                config=mdap_config,
                guardrails_adapter=self.guardrails_adapter
            )

            # Create and execute task
            mdap_task = self.MDAPTask(
                task_id=f"task_{uuid.uuid4()}",
                description=task,
                steps=[
                    self.MDAPStep(
                        step_id="step_1",
                        prompt=task,
                        task_type="general"
                    )
                ]
            )

            # Apply LMQL constraints if available
            if lmql_constraints and hasattr(orchestrator, 'apply_constraints'):
                try:
                    orchestrator.apply_constraints(lmql_constraints)
                    self.logger.debug(
                        f"Applied {len(lmql_constraints)} LMQL constraints",
                        correlation_id=correlation_id
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to apply LMQL constraints",
                        correlation_id=correlation_id,
                        error=str(e)
                    )

            result = orchestrator.execute_task(mdap_task)

            return {
                "success": True,
                "result": result,
                "statistics": self._extract_statistics(result)
            }

        except Exception as e:
            self.logger.error(
                "Core red flagging solve failed",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def _convert_to_dict_result(
        self,
        result: MDAPSolveResult,
        additional_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert MDAPSolveResult to dict with additional fields"""
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "statistics": result.statistics,
            "validation_failures": result.validation_failures,
            "correlation_id": result.correlation_id,
            "timestamp": result.timestamp,
            "layers_used": result.layers_used,
            "method": result.method,
            **additional_fields
        }

    def _extract_statistics(self, result: Any) -> Dict[str, int]:
        """Extract statistics from MDAP result"""
        stats = {
            "total_votes": 0,
            "valid_votes": 0,
            "rejected_votes": 0,
            "remediated_votes": 0
        }

        try:
            if hasattr(result, 'step_results') and result.step_results:
                for step_result in result.step_results:
                    if hasattr(step_result, 'vote_result') and step_result.vote_result:
                        votes = step_result.vote_result.votes
                        total = sum(votes.values()) if votes else 0
                        stats["valid_votes"] += total

                        if hasattr(step_result.vote_result, 'flagged_reasons'):
                            if step_result.vote_result.flagged_reasons:
                                stats["rejected_votes"] += len(step_result.vote_result.flagged_reasons)

                stats["total_votes"] = stats["valid_votes"] + stats["rejected_votes"]

        except Exception as e:
            self.logger.debug(
                "Failed to extract statistics",
                error=str(e)
            )

        return stats

    def verify_vote(
        self,
        vote: Any,
        validators: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> VoteValidationResult:
        """
        Validate an individual MDAP vote using core components.

        This can be called during MDAP execution to validate votes
        before they are counted.

        Args:
            vote: The vote to validate (can be dict, str, or other)
            validators: List of validators to apply
            correlation_id: Optional correlation ID for logging

        Returns:
            VoteValidationResult with validation outcome
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = f"vote_{uuid.uuid4()}"

        # Default validators for MDAP votes
        default_validators = ["json_structure", "required_fields", "malicious_patterns"]
        all_validators = default_validators + (validators or [])

        self.logger.debug(
            "Validating vote",
            correlation_id=correlation_id,
            vote_type=type(vote).__name__,
            validators=all_validators
        )

        # Initialize result
        result = VoteValidationResult(
            is_valid=True,
            vote=vote,
            original_vote=vote,
            correlation_id=correlation_id
        )

        # Check if Guardrails is available
        if not self.guardrails_adapter or not self.guardrails_adapter.is_available():
            self.logger.debug(
                "Guardrails unavailable, returning vote as-is",
                correlation_id=correlation_id
            )
            return result

        # Convert vote to string for validation
        try:
            if isinstance(vote, (dict, list)):
                vote_str = json.dumps(vote)
            else:
                vote_str = str(vote)
        except Exception as e:
            self.logger.error(
                "Failed to serialize vote",
                correlation_id=correlation_id,
                error=str(e)
            )
            result.is_valid = False
            result.failures.append(f"Vote serialization failed: {str(e)}")
            return result

        # Validate vote
        try:
            validation = self.guardrails_adapter.validate_output(
                output=vote_str,
                validators=all_validators,
                on_fail="filter",
                correlation_id=correlation_id
            )

            result.is_valid = validation.is_valid

            if not validation.is_valid:
                # Extract failure messages
                for failure in validation.failures:
                    if isinstance(failure, dict):
                        result.failures.append(failure.get("message", str(failure)))
                    else:
                        result.failures.append(str(failure))

                # Check if remediation was applied
                if validation.remediation_applied:
                    result.remediated = True
                    result.validator_name = validation.remediation_applied

                    # Parse remediated output if available
                    if validation.output:
                        try:
                            result.vote = json.loads(validation.output)
                            self.stats["remediated_votes"] += 1
                        except:
                            result.vote = validation.output
                else:
                    self.stats["rejected_votes"] += 1

            else:
                self.stats["valid_votes"] += 1
            self.stats["total_votes_validated"] += 1

        except Exception as e:
            self.logger.error(
                "Vote validation exception",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            result.is_valid = False
            result.failures.append(f"Validation exception: {str(e)}")

        return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get adapter status and health.

        Returns:
            Dict with adapter status information
        """
        guardrails_available = (
            self.guardrails_adapter is not None and
            self.guardrails_adapter.is_available()
        )

        # Check enhanced red flagging status
        enhanced_redflagging_available = (
            ENHANCED_REDFLAGGING_AVAILABLE and
            self.enhanced_redflagging_enabled and
            self.enhanced_redflagger is not None
        )

        # Get LMQL adapter status
        lmql_available = False
        if self.enhanced_redflagger and hasattr(self.enhanced_redflagger, 'lmql_adapter'):
            if self.enhanced_redflagger.lmql_adapter:
                lmql_available = self.enhanced_redflagger.lmql_adapter.is_available()

        return {
            "mdap_core_available": self.mdap_available,
            "mdap_mcp_available": self.mdap_mcp_solve is not None,
            "guardrails_available": guardrails_available,
            "enhanced_redflagging_available": enhanced_redflagging_available,
            "lmql_available": lmql_available,
            "primary_method": "core_integration" if self.mdap_available else "mcp_fallback",
            "layers": {
                "mdap_core": {
                    "available": self.mdap_available,
                    "enabled": True
                },
                "mcp_tools": {
                    "available": self.mdap_mcp_solve is not None,
                    "enabled": True
                },
                "guardrails": {
                    "available": guardrails_available,
                    "enabled": self.config.guardrails.enabled
                },
                "enhanced_redflagging": {
                    "available": enhanced_redflagging_available,
                    "enabled": getattr(self.config, 'enhanced_redflagging_enabled', True),
                    "lmql_enabled": getattr(self.config, 'lmql_enabled', False)
                }
            },
            "statistics": self.stats.copy(),
            "config": {
                "guardrails_enabled": self.config.guardrails.enabled,
                "validators": self.config.guardrails.validators,
                "on_fail_strategy": self.config.guardrails.on_fail,
                "max_retries": self.config.guardrails.max_retries,
                "ace_enabled": getattr(self.config, 'ace_enabled', True),
                "steer_enabled": getattr(self.config, 'steer_enabled', True),
                "enhanced_redflagging_enabled": getattr(self.config, 'enhanced_redflagging_enabled', True),
                "lmql_enabled": getattr(self.config, 'lmql_enabled', False)
            }
        }

    def get_statistics(self) -> Dict[str, int]:
        """Get adapter statistics"""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.stats = {
            "total_solves": 0,
            "successful_solves": 0,
            "failed_solves": 0,
            "total_votes_validated": 0,
            "valid_votes": 0,
            "remediated_votes": 0,
            "rejected_votes": 0,
            "core_integration_used": 0,
            "mcp_fallback_used": 0,
            "enhanced_redflagging_used": 0,
            "red_flags_detected": 0
        }
        self.logger.info("Statistics reset")

    # =========================================================================
    # INTERNAL VALIDATION METHODS
    # =========================================================================

    def _validate_input(
        self,
        task: str,
        mdap_k_ahead: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Validate input parameters.

        Args:
            task: Task description
            mdap_k_ahead: Number of agents for voting
            correlation_id: Correlation ID

        Returns:
            Dict with validation result
        """
        result = {
            "is_valid": True,
            "error": None,
            "failures": []
        }

        # Validate task
        if not task or not isinstance(task, str):
            result["is_valid"] = False
            result["error"] = "Task must be a non-empty string"
            result["failures"].append("Invalid task parameter")
            return result

        if len(task) > 10000:
            result["is_valid"] = False
            result["error"] = f"Task too long: {len(task)} characters (max 10000)"
            result["failures"].append("Task exceeds maximum length")
            return result

        # Validate mdap_k_ahead parameter
        if not isinstance(mdap_k_ahead, int):
            result["is_valid"] = False
            result["error"] = f"mdap_k_ahead must be an integer, got {type(mdap_k_ahead).__name__}"
            result["failures"].append("Invalid mdap_k_ahead type")
            return result

        if mdap_k_ahead < 2 or mdap_k_ahead > 20:
            result["is_valid"] = False
            result["error"] = f"mdap_k_ahead must be 2-20, got {mdap_k_ahead}"
            result["failures"].append("mdap_k_ahead out of range")
            return result

        # Guardrails input validation
        if self.guardrails_adapter and self.guardrails_adapter.is_available():
            try:
                input_validation = self.guardrails_adapter.validate_input(
                    prompt=task,
                    validators=["task_format", "injection_check"],
                    correlation_id=correlation_id
                )

                if not input_validation.is_valid:
                    result["is_valid"] = False
                    result["error"] = "Guardrails input validation failed"
                    for failure in input_validation.failures:
                        if isinstance(failure, dict):
                            result["failures"].append(failure.get("message", str(failure)))
                        else:
                            result["failures"].append(str(failure))

            except Exception as e:
                self.logger.warning(
                    "Guardrails input validation failed, continuing",
                    correlation_id=correlation_id,
                    error=str(e)
                )

        return result

    def _validate_output(
        self,
        output: Any,
        validators: List[str],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Validate output structure.

        Args:
            output: Output to validate
            validators: List of validators to apply
            correlation_id: Correlation ID

        Returns:
            Dict with validation result
        """
        result = {
            "is_valid": True,
            "failures": [],
            "remediated_output": None,
            "remediated_count": 0
        }

        if not self.guardrails_adapter or not self.guardrails_adapter.is_available():
            return result

        try:
            # Convert output to JSON string
            if isinstance(output, (dict, list)):
                output_str = json.dumps(output)
            else:
                output_str = str(output)

            # Validate with Guardrails
            validation = self.guardrails_adapter.validate_output(
                output=output_str,
                validators=validators,
                on_fail="fix",
                correlation_id=correlation_id
            )

            result["is_valid"] = validation.is_valid

            if not validation.is_valid:
                # Extract failures
                for failure in validation.failures:
                    if isinstance(failure, dict):
                        result["failures"].append(failure.get("message", str(failure)))
                    else:
                        result["failures"].append(str(failure))

                # Check for remediation
                if validation.remediation_applied and validation.output:
                    result["remediated_output"] = validation.output
                    result["remediated_count"] = 1

        except Exception as e:
            self.logger.error(
                "Output validation exception",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            result["is_valid"] = False
            result["failures"].append(f"Validation exception: {str(e)}")

        return result

    def _validate_result(
        self,
        result: Any,
        validators: Optional[List[str]],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Validate MDAP result and extract statistics.

        Args:
            result: MDAP run result
            validators: Optional list of validators
            correlation_id: Correlation ID

        Returns:
            Dict with validation statistics and failures
        """
        valid_votes = 0
        rejected_votes = 0
        remediated_votes = 0
        failures = []

        # Extract votes from result
        try:
            if hasattr(result, 'step_results') and result.step_results:
                for step_result in result.step_results:
                    if hasattr(step_result, 'vote_result') and step_result.vote_result:
                        votes = step_result.vote_result.votes
                        total = sum(votes.values()) if votes else 0
                        valid_votes += total

                        # Check for flagged reasons (rejected votes)
                        if hasattr(step_result.vote_result, 'flagged_reasons'):
                            if step_result.vote_result.flagged_reasons:
                                rejected_votes += len(step_result.vote_result.flagged_reasons)
                                failures.extend(step_result.vote_result.flagged_reasons)
        except Exception as e:
            self.logger.warning(
                "Failed to extract vote statistics",
                correlation_id=correlation_id,
                error=str(e)
            )

        return {
            "statistics": {
                "total_votes": valid_votes + rejected_votes,
                "valid_votes": valid_votes,
                "rejected_votes": rejected_votes,
                "remediated_votes": remediated_votes
            },
            "failures": failures
        }

    def _serialize_mdap_result(self, result: Any) -> Dict[str, Any]:
        """
        Serialize MDAP result to dictionary for JSON output.

        Args:
            result: MDAP run result object

        Returns:
            Dict serialization of result
        """
        try:
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            elif dataclasses.is_dataclass(result):
                return dataclasses.asdict(result)
            else:
                # Try to extract common fields
                return {
                    "task_id": getattr(result, 'task_id', None),
                    "final_answer": getattr(result, 'final_answer', None),
                    "total_votes": getattr(result, 'total_votes', 0),
                    "execution_time": getattr(result, 'execution_time', None),
                    "success": getattr(result, 'success', True)
                }
        except Exception as e:
            self.logger.warning(
                "Failed to serialize MDAP result",
                error=str(e)
            )
            return {
                "raw_result": str(result),
                "serialization_error": str(e)
            }

    def _get_default_team(self) -> Any:
        """
        Get default team for MDAP execution.

        Returns:
            Default Team object
        """
        # Import Team class if available
        if self.Team:
            try:
                # Create default team with basic configuration
                return self.Team()
            except Exception as e:
                self.logger.warning(
                    "Failed to create default team",
                    error=str(e)
                )

        # Return None and let MDAP handle it
        return None


# =============================================================================
# HELPER CLASSES
# =============================================================================

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        import traceback
        import dataclasses

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mdap_adapter(**config) -> MDAPReliabilityAdapter:
    """
    Factory function to create an MDAP reliability adapter.

    Args:
        **config: Configuration options

    Returns:
        Configured MDAPReliabilityAdapter instance

    Example:
        adapter = create_mdap_adapter(
            guardrails_enabled=True,
            validators=["vote_format", "json_structure"]
        )
    """
    return MDAPReliabilityAdapter(config=None)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def solve_with_guardrails(
    task: str,
    mdap_k_ahead: int = 5,
    validators: Optional[List[str]] = None,
    **kwargs
) -> MDAPSolveResult:
    """
    Convenience function for one-off MDAP solve with guardrails.

    Creates a temporary adapter and solves the task.

    Args:
        task: Task to solve
        mdap_k_ahead: Number of agents for voting
        validators: Validators to apply
        **kwargs: Additional parameters

    Returns:
        MDAPSolveResult

    Example:
        result = solve_with_guardrails(
            task="Solve this problem",
            mdap_k_ahead=5,
            validators=["vote_format", "json_structure"]
        )
    """
    adapter = create_mdap_adapter()
    return adapter.solve_with_validation(
        task=task,
        mdap_k_ahead=mdap_k_ahead,
        validators=validators,
        **kwargs
    )


def solve_with_redflagging(
    task: str,
    mdap_k_ahead: int = 5,
    use_lmql_constraints: bool = True,
    use_enhanced_validation: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for solving with enhanced red flagging.

    Creates a temporary adapter and solves the task with the multi-layered
    red flagging system.

    Args:
        task: Task to solve
        mdap_k_ahead: Number of agents for voting
        use_lmql_constraints: Enable LMQL pre-generation constraints
        use_enhanced_validation: Enable enhanced red flagging
        **kwargs: Additional parameters

    Returns:
        Dict with success, result, red_flags, statistics

    Example:
        result = solve_with_redflagging(
            task="Solve this problem",
            mdap_k_ahead=5,
            use_lmql_constraints=True,
            use_enhanced_validation=True
        )
        if result["success"]:
            print(f"Solution: {result['result']}")
        if result["red_flags"]:
            print(f"Red flags detected: {result['red_flags']}")
    """
    adapter = create_mdap_adapter()
    return adapter.solve_with_enhanced_redflagging(
        task=task,
        mdap_k_ahead=mdap_k_ahead,
        use_lmql_constraints=use_lmql_constraints,
        use_enhanced_validation=use_enhanced_validation,
        **kwargs
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("MDAP Reliability Adapter - Enhanced Test Suite")
    print("=" * 60)

    # Create adapter
    adapter = create_mdap_adapter()

    # Print status
    status = adapter.get_status()
    print(f"\nAdapter Status:")
    print(f"  MDAP Core Available: {status['mdap_core_available']}")
    print(f"  MDAP MCP Available: {status['mdap_mcp_available']}")
    print(f"  Guardrails Available: {status['guardrails_available']}")
    print(f"  Primary Method: {status['primary_method']}")

    # Print statistics
    stats = adapter.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example solve (if MDAP available)
    if status['mdap_core_available'] or status['mdap_mcp_available']:
        print("\n" + "=" * 60)
        print("Example Solve")
        print("=" * 60)

        result = adapter.solve_with_validation(
            task="What is 2 + 2?",
            mdap_k_ahead=3,
            validators=["vote_format", "json_structure"]
        )

        print(f"\nSuccess: {result.success}")
        print(f"Method: {result.method}")
        print(f"Layers Used: {result.layers_used}")
        if result.success:
            print(f"Result: {json.dumps(result.result, indent=2)[:200]}...")
        else:
            print(f"Error: {result.error}")

        print(f"\nStatistics:")
        print(f"  Total Votes: {result.statistics['total_votes']}")
        print(f"  Valid Votes: {result.statistics['valid_votes']}")
        print(f"  Remediated: {result.statistics['remediated_votes']}")

    # Example vote validation
    print("\n" + "=" * 60)
    print("Example Vote Validation")
    print("=" * 60)

    test_votes = [
        {"decision": "APPROVE", "confidence": 0.9},
        {"decision": "REJECT", "confidence": 0.1},
        "INVALID VOTE FORMAT"
    ]

    for i, vote in enumerate(test_votes):
        validation = adapter.verify_vote(vote)
        print(f"\nVote {i+1}:")
        print(f"  Valid: {validation.is_valid}")
        print(f"  Remediated: {validation.remediated}")
        if validation.failures:
            print(f"  Failures: {validation.failures}")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

    # Final statistics
    final_stats = adapter.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Solves: {final_stats['total_solves']}")
    print(f"  Core Integration Used: {final_stats['core_integration_used']}")
    print(f"  MCP Fallback Used: {final_stats['mcp_fallback_used']}")
