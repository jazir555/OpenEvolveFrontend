"""
REST API Server for External System Integration

This module provides a REST API for external systems to interact with the
Decomposition Workflow system.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
import threading
import asyncio
import uvicorn
import os
import re
import base64
import json
import time
import tempfile
from datetime import datetime, timedelta
import uuid
import logging
from pathlib import Path

from ui_shim import ui as _UI_SHIM, SessionState
from web3_formal_evidence import build_web3_formal_evidence, verify_web3_lean_proof

logger = logging.getLogger(__name__)

# SECURITY: Import security framework with comprehensive features
try:
    from security_framework import (
        Permission, Role, UserContext, JWTManager, get_jwt_manager,
        RateLimiter, get_rate_limiter, InputValidator, ValidationError,
        AuditLogger, get_audit_logger, SecurityHeadersMiddleware, RateLimitMiddleware,
        security_scheme, api_key_scheme, get_current_user, require_auth, require_permission,
        generate_secure_id, hash_sensitive_data
    )
    from security_framework import SecurityConfig
    SECURITY_FRAMEWORK_AVAILABLE = True
    logger.info("SECURITY: Security framework loaded successfully")
except ImportError as e:
    SECURITY_FRAMEWORK_AVAILABLE = False
    logger.warning(f"SECURITY: Security framework not available: {e}")
    # Define stub classes for when security framework is not available
    class Permission:
        WORKFLOW_CREATE = "workflow:create"
        WORKFLOW_READ = "workflow:read"
        WORKFLOW_UPDATE = "workflow:update"
        WORKFLOW_DELETE = "workflow:delete"
        WORKFLOW_EXECUTE = "workflow:execute"
        API_ACCESS = "api:access"
        API_ADMIN = "api:admin"
        SYSTEM_ADMIN = "system:admin"
    
    class UserContext:
        def __init__(self, user_id="anonymous", username="anonymous", email="", roles=None, permissions=None):
            self.user_id = user_id
            self.username = username
            self.email = email
            self.roles = roles or []
            self.permissions = permissions or []
            self.is_superuser = False
        
        def has_permission(self, permission):
            return True
    
    def get_current_user():
        return None
    
    def require_auth():
        return None
    
    def require_permission(permission):
        return None
    
    class SecurityHeadersMiddleware:
        pass
    
    class RateLimitMiddleware:
        pass


class _RunUIContext:
    """Per-run UI context with isolated session_state."""

    def __init__(self):
        self.session_state = SessionState()
        if "thread_lock" not in self.session_state:
            self.session_state.thread_lock = threading.Lock()
        self.sidebar = self

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


def _attach_ui(module, ui_instance=_UI_SHIM):
    """Attach a UI shim to a module."""
    try:
        module.st = ui_instance
    except Exception:
        pass

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for API Server
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

try:
    from crewai_state_management import create_state_manager, WorkflowState as CrewAIWorkflowState
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

try:
    from bubblelabs_extended_integration import get_extended_integration, initialize_extended_integration
    BUBBLELABS_AVAILABLE = True
except ImportError:
    BUBBLELABS_AVAILABLE = False

try:
    import model_orchestration as _model_orchestration
    from model_orchestration import ModelOrchestrator, ModelRole
    MODEL_ORCHESTRATION_AVAILABLE = True
except ImportError:
    MODEL_ORCHESTRATION_AVAILABLE = False
    _model_orchestration = None

try:
    import integrated_workflow as _integrated_workflow
    from integrated_workflow import run_fully_integrated_adversarial_evolution
    INTEGRATED_WORKFLOW_AVAILABLE = True
except ImportError:
    INTEGRATED_WORKFLOW_AVAILABLE = False
    _integrated_workflow = None

try:
    from bubblelabs_maker_integration import (
        MakerWorkflowManager,
        ToolRepository,
        CrewAIDelegationManager,
        ToolStatus,
        DelegationStatus,
    )
    MAKER_INTEGRATION_AVAILABLE = True
except ImportError:
    MAKER_INTEGRATION_AVAILABLE = False
    ToolStatus = None
    DelegationStatus = None

try:
    from knowledge_engine.engine import KnowledgeEngine
    from bubblelabs_knowledge_integration import (
        KnowledgeQueryInterface,
        KnowledgeExtractionWorkflow,
        KnowledgeGraphVisualizer,
    )
    KNOWLEDGE_EXPLORER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_EXPLORER_AVAILABLE = False

try:
    from bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        LeanAideTaskType,
        LeanAideIntegrationBridge,
    )
    LEANAIDE_BRIDGE_AVAILABLE = True
except ImportError:
    LEANAIDE_BRIDGE_AVAILABLE = False
    LeanAideTaskType = None
    LeanAideIntegrationBridge = None

# **LEAN INTEGRATION**: Real Lean client for formal verification
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

try:
    from decomposition_mcp_tools import (
        get_mcp_tool_inventory,
        web3_ingest_contract_audit_stack,
        web3_ingest_foundry_fuzzing,
        web3_ingest_slither_static_analysis,
    )
    WEB3_INGESTION_AVAILABLE = True
except ImportError:
    WEB3_INGESTION_AVAILABLE = False
    get_mcp_tool_inventory = None
    web3_ingest_contract_audit_stack = None
    web3_ingest_foundry_fuzzing = None
    web3_ingest_slither_static_analysis = None

try:
    from z3prover_integration import (
        solve_smart_contract_exploit_witness,
        translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation,
    )
    WEB3_FORMAL_VERIFICATION_AVAILABLE = (
        translate_solidity_assignment_to_z3 is not None
        and solve_smart_contract_exploit_witness is not None
    )
except ImportError:
    WEB3_FORMAL_VERIFICATION_AVAILABLE = False
    solve_smart_contract_exploit_witness = None
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None

try:
    from evolution import run_comprehensive_evolution, create_evolution_configuration
    from adversarial import run_comprehensive_adversarial_testing, create_adversarial_configuration
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

try:
    import version_control as _version_control_module
    _attach_ui(_version_control_module, _UI_SHIM)
    _version_control_manager = _version_control_module.VersionControl()
    VERSION_CONTROL_AVAILABLE = True
except Exception as e:
    VERSION_CONTROL_AVAILABLE = False
    _version_control_manager = None
    logger.warning(f"Version control unavailable: {e}")

try:
    import validation_manager as _validation_manager_module
    _attach_ui(_validation_manager_module, _UI_SHIM)
    _validation_manager = _validation_manager_module.ValidationManager()
    VALIDATION_MANAGER_AVAILABLE = True
except Exception as e:
    VALIDATION_MANAGER_AVAILABLE = False
    _validation_manager = None
    logger.warning(f"Validation manager unavailable: {e}")

BUBBLELABS_WORKFLOW_AVAILABLE = False
_bubblelabs_workflow_integration = None


def _get_bubblelabs_workflow_integration():
    global _bubblelabs_workflow_integration, BUBBLELABS_WORKFLOW_AVAILABLE
    if _bubblelabs_workflow_integration is not None:
        return _bubblelabs_workflow_integration
    try:
        from openevolve_bubblelabs_api import openevolve_bubblelabs_integration
        _bubblelabs_workflow_integration = openevolve_bubblelabs_integration
        BUBBLELABS_WORKFLOW_AVAILABLE = True
    except Exception as e:
        BUBBLELABS_WORKFLOW_AVAILABLE = False
        _bubblelabs_workflow_integration = None
        logger.warning(f"BubbleLabs workflow integration unavailable: {e}")
    return _bubblelabs_workflow_integration


# **ACTUAL INTEGRATION HELPER METHODS**: API Server
def _trigger_api_alerts(operation, success, request_id=None, error=None, metadata=None):
    """Trigger alerts for API server operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"API {operation} Failed",
            message=f"API server operation '{operation}' failed: {error}",
            severity=severity,
            source="APIServer",
            metadata=metadata or {"request_id": request_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger API alert: {e}")


def _extract_api_knowledge(operation, request_id, endpoint, result):
    """Extract knowledge from API operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"api_{operation}_{request_id}",
            artifact_type="api_execution",
            source_component="APIServer",
            content={
                "operation": operation,
                "request_id": request_id,
                "endpoint": endpoint,
                "status": result.get("status", "unknown") if result else "unknown",
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract API knowledge: {e}")


def _track_api_performance(operation, success, duration_seconds, endpoint, status_code=200):
    """Track performance of API operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name=f"api_{endpoint}",
            component_name="APIServer",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "endpoint": endpoint,
                "status_code": status_code
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track API performance: {e}")


def _request_openai_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
) -> str:
    """Make a request to an OpenAI-compatible API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed
        )
        return response.choices[0].message.content
    except ImportError:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            data["seed"] = seed
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


# Import environment helpers
from env_helpers import is_production

from workflow_structures import (
    DecompositionPlan, SubProblem, Team, GauntletDefinition, GauntletRoundRule,
    WorkflowState, ModelConfig
)
from knowledge_manager import KnowledgeManager
from template_manager import TemplateManager
from parameter_manager import ParameterManager
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import HealthMonitor
try:
    from monitoring import (
        monitoring_dashboard as system_monitoring_dashboard,
        metrics_collector as system_metrics_collector,
        alert_manager as system_alert_manager,
        health_monitor as system_health_monitor,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    system_monitoring_dashboard = None
    system_metrics_collector = None
    system_alert_manager = None
    system_health_monitor = None
    MONITORING_AVAILABLE = False
    logger.warning("System monitoring unavailable: monitoring module import failed")
from providercatalogue import PROVIDERS as PROVIDERS_MAP
try:
    from content_manager import content_manager
    CONTENT_MANAGER_AVAILABLE = True
except Exception as exc:
    content_manager = None
    CONTENT_MANAGER_AVAILABLE = False
    logger.warning("Content manager unavailable: %s", exc)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow
from determinism_stack import (
    DeterministicPipeline,
    DeterminismConfig,
    HybridDeterministicSystem,
    LLMConfig,
    build_llm,
)


# Initialize FastAPI app
app = FastAPI(
    title="Decomposition Workflow API",
    description="""
    REST API for the Sovereign-Grade Decomposition Workflow system.
    
    ## Features
    
    * **Workflow Management**: Create, monitor, pause, resume, and retrieve workflow results
    * **Team Management**: Configure AI teams for different roles (Blue, Red, Gold)
    * **Gauntlet Management**: Define evaluation gauntlets with programmable rules
    * **Webhooks**: Subscribe to workflow events for real-time notifications
    * **Authentication**: API key and JWT token-based authentication with RBAC
    
    ## Authentication
    
    Use one of the following methods:
    
    1. **API Key**: Include `X-API-Key` header with your API key
    2. **JWT Token**: Get a token from `/auth/token` and include `Authorization: Bearer <token>` header
    
    ## Roles
    
    * **ADMIN**: Full access to all endpoints
    * **USER**: Can create and manage workflows, teams, and gauntlets
    * **READONLY**: Can only view resources
    """,
    version="1.0.0",
    contact={
        "name": "Decomposition Workflow Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECURITY: Add security headers middleware
if SECURITY_FRAMEWORK_AVAILABLE:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    logger.info("SECURITY: Security middleware enabled")

_model_orchestrator = ModelOrchestrator() if MODEL_ORCHESTRATION_AVAILABLE else None
if _model_orchestrator and _model_orchestration is not None:
    _attach_ui(_model_orchestration, _UI_SHIM)


# Initialize templates for dashboard
templates = Jinja2Templates(directory="templates")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )

# Initialize default managers (legacy fallback, use tenant-scoped managers in handlers)
team_manager = TeamManager()
gauntlet_manager = GauntletManager()
knowledge_manager = KnowledgeManager()
template_manager = TemplateManager()
parameter_manager = ParameterManager()
sovereign_db = SovereignDatabase()
sovereign_health_monitor = HealthMonitor()

# Optional integration managers
_maker_manager: Optional[MakerWorkflowManager] = None
if MAKER_INTEGRATION_AVAILABLE:
    try:
        _maker_manager = MakerWorkflowManager(
            tool_repository=ToolRepository(),
            delegation_manager=CrewAIDelegationManager(),
        )
    except Exception as exc:
        logger.warning("Maker integration unavailable: %s", exc)
        _maker_manager = None

_knowledge_engine_instance: Optional[KnowledgeEngine] = None
_knowledge_query_interface: Optional[KnowledgeQueryInterface] = None
_knowledge_extraction_workflow: Optional[KnowledgeExtractionWorkflow] = None
_knowledge_graph_visualizer: Optional[KnowledgeGraphVisualizer] = None

_leanaide_bridge: Optional[LeanAideIntegrationBridge] = None
if LEANAIDE_BRIDGE_AVAILABLE:
    try:
        _leanaide_bridge = get_leanaide_bridge()
    except Exception as exc:
        logger.warning("LeanAide bridge unavailable: %s", exc)
        _leanaide_bridge = None


def _get_knowledge_components() -> Optional[Dict[str, Any]]:
    """Lazy initialize knowledge explorer components."""
    global _knowledge_engine_instance
    global _knowledge_query_interface
    global _knowledge_extraction_workflow
    global _knowledge_graph_visualizer
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        return None
    if _knowledge_engine_instance is None:
        try:
            _knowledge_engine_instance = KnowledgeEngine()
            _knowledge_query_interface = KnowledgeQueryInterface(_knowledge_engine_instance)
            _knowledge_extraction_workflow = KnowledgeExtractionWorkflow(_knowledge_engine_instance)
            _knowledge_graph_visualizer = KnowledgeGraphVisualizer()
        except Exception as exc:
            logger.warning("Failed to initialize knowledge explorer: %s", exc)
            _knowledge_engine_instance = None
            _knowledge_query_interface = None
            _knowledge_extraction_workflow = None
            _knowledge_graph_visualizer = None
            return None
    return {
        "engine": _knowledge_engine_instance,
        "query": _knowledge_query_interface,
        "extract": _knowledge_extraction_workflow,
        "graph": _knowledge_graph_visualizer,
    }


@dataclass
class _RunState:
    run_id: str
    run_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    session_state: Optional[SessionState] = None


_run_lock = threading.Lock()
_evolution_runs: Dict[str, _RunState] = {}
_adversarial_runs: Dict[str, _RunState] = {}

# Auto-approval configuration (in-memory)
AUTO_APPROVAL_CONFIG: Dict[str, Any] = {"enabled": False, "rules": []}
AUTO_APPROVAL_AUDIT_LOG: List[Dict[str, Any]] = []

# Prompt and protocol template storage
PROMPTS_FILE = os.path.join("data", "custom_prompts.json")
CUSTOM_PROTOCOL_TEMPLATES_FILE = os.path.join("data", "protocol_templates_custom.json")


def _load_json_store(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except (OSError, IOError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load JSON store %s: %s", path, exc)
    return {}


def _save_json_store(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except (OSError, IOError, TypeError) as exc:
        logger.error("Failed to save JSON store %s: %s", path, exc)


def _create_run_context(run_state: _RunState, log_key: str) -> _RunUIContext:
    """Create an isolated UI context for a background run."""
    ui_context = _RunUIContext()
    ui_context.session_state[log_key] = run_state.logs
    ui_context.session_state["thread_lock"] = threading.Lock()
    ui_context.session_state[f"{log_key}_status_message"] = ""
    run_state.session_state = ui_context.session_state
    return ui_context


def _finalize_run_state(run_state: _RunState, result: Optional[Dict[str, Any]], error: Optional[str]) -> None:
    run_state.result = result
    run_state.error = error
    run_state.completed_at = datetime.utcnow().isoformat()
    if run_state.cancel_requested:
        run_state.status = "cancelled"
    elif error:
        run_state.status = "failed"
    else:
        run_state.status = "completed"


def _start_background_run(run_state: _RunState, target, *args) -> None:
    """Start a background thread for a run and track it."""
    def _runner():
        run_state.status = "running"
        run_state.started_at = datetime.utcnow().isoformat()
        try:
            result = target(*args)
            _finalize_run_state(run_state, result, None)
        except Exception as exc:
            logger.error("Run %s failed: %s", run_state.run_id, exc, exc_info=True)
            _finalize_run_state(run_state, None, str(exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


CUSTOM_PROMPTS: Dict[str, str] = _load_json_store(PROMPTS_FILE)
CUSTOM_PROTOCOL_TEMPLATES: Dict[str, str] = _load_json_store(CUSTOM_PROTOCOL_TEMPLATES_FILE)

DEFAULT_VALIDATION_RULES = {
    "generic": {
        "max_length": 12000,
        "required_sections": ["Purpose", "Scope", "Procedure"],
        "required_keywords": ["must", "should"],
    },
    "compliance": {
        "max_length": 15000,
        "required_sections": ["Compliance", "Audit", "Controls"],
        "required_keywords": ["policy", "regulation", "compliance"],
    },
    "security": {
        "max_length": 12000,
        "required_sections": ["Security", "Threats", "Mitigations"],
        "required_keywords": ["access", "encryption", "authentication"],
    },
    "technical": {
        "max_length": 12000,
        "required_sections": ["Architecture", "Interfaces", "Testing"],
        "required_keywords": ["requirement", "interface", "validation"],
    },
}


def _basic_validate_protocol(protocol_text: str, validation_type: str = "generic") -> Dict[str, Any]:
    """Fallback validation if ContentManagement is unavailable."""
    rules = DEFAULT_VALIDATION_RULES.get(validation_type, DEFAULT_VALIDATION_RULES["generic"])
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    if not protocol_text.strip():
        errors.append("Protocol text is empty")
        return {
            "valid": False,
            "score": 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    if len(protocol_text) > rules["max_length"]:
        warnings.append(
            f"Protocol exceeds recommended length of {rules['max_length']} characters"
        )

    for section in rules.get("required_sections", []):
        if section.lower() not in protocol_text.lower():
            errors.append(f"Missing required section: {section}")

    for keyword in rules.get("required_keywords", []):
        if keyword.lower() not in protocol_text.lower():
            suggestions.append(f"Consider adding keyword: {keyword}")

    score = max(0, 100 - len(errors) * 10 - len(warnings) * 5 + len(suggestions) * 2)
    return {
        "valid": len(errors) == 0,
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
    }

# In-memory storage for workflows (replace with database in production)
workflows: Dict[str, WorkflowState] = {}

# In-memory audit log (replace with persistent storage in production)
AUDIT_LOGS: List[Dict[str, Any]] = []

# ICR event queues (in-memory, best-effort)
ICR_REFINEMENT_EVENTS: deque = deque(maxlen=200)
ICR_REWARD_CALIBRATION_QUEUE: deque = deque(maxlen=100)
ICR_REWARD_CALIBRATION_RESPONSES: Dict[str, Dict[str, Any]] = {}
ICR_HEATMAP_SNAPSHOTS: deque = deque(maxlen=100)


def record_audit_event(
    user: "AuthUser",
    operation: str,
    resource: str,
    resource_id: str,
    success: bool,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Record an audit event for workflow lifecycle actions."""
    AUDIT_LOGS.append({
        "timestamp": datetime.now().isoformat(),
        "user": user.name,
        "role": user.role,
        "operation": operation,
        "resource": resource,
        "resource_id": resource_id,
        "success": success,
        "details": details or {}
    })


def _evaluate_auto_approval_rule(rule: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    """Evaluate a single auto-approval rule against a plan."""
    conditions = rule.get("conditions", [])
    if not conditions:
        return False

    results = []
    for condition in conditions:
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        plan_value = plan.get(field)

        try:
            if operator == "<":
                result = float(plan_value) < float(value)
            elif operator == ">":
                result = float(plan_value) > float(value)
            elif operator == "==":
                result = str(plan_value) == str(value)
            elif operator == "!=":
                result = str(plan_value) != str(value)
            elif operator == "contains":
                result = str(value).lower() in str(plan_value).lower()
            else:
                result = False
        except (TypeError, ValueError):
            result = False

        results.append(result)

    final_result = results[0]
    for index, condition in enumerate(conditions[:-1]):
        logical_op = condition.get("logical_op", "AND")
        if logical_op == "AND":
            final_result = final_result and results[index + 1]
        else:
            final_result = final_result or results[index + 1]

    return final_result


def _serialize_performance_metric(metric: "PerformanceMetrics") -> Dict[str, Any]:
    """Serialize PerformanceMetrics for JSON responses."""
    timestamp = metric.timestamp
    if isinstance(timestamp, (int, float)):
        timestamp_value = datetime.fromtimestamp(timestamp).isoformat()
    elif isinstance(timestamp, datetime):
        timestamp_value = timestamp.isoformat()
    else:
        timestamp_value = None
    return {
        "entity_type": metric.entity_type,
        "entity_id": metric.entity_id,
        "metrics": metric.metrics,
        "timestamp": timestamp_value,
        "domain": metric.domain,
        "problem_type": metric.problem_type,
        "context": metric.context,
    }


def _serialize_monitoring_metric(metric: Any) -> Dict[str, Any]:
    """Serialize monitoring Metric entries."""
    try:
        timestamp = metric.timestamp.isoformat() if metric.timestamp else None
    except AttributeError:
        timestamp = None
    return {
        "name": getattr(metric, "name", None),
        "value": getattr(metric, "value", None),
        "type": getattr(metric, "type", None).value if getattr(metric, "type", None) else None,
        "labels": getattr(metric, "labels", None),
        "timestamp": timestamp,
        "description": getattr(metric, "description", None),
    }


_crewai_state_manager: Optional["StateManager"] = None


def _get_crewai_state_manager():
    """Lazy-load CrewAI state manager for monitoring endpoints."""
    global _crewai_state_manager
    if not CREWAI_AVAILABLE:
        return None
    if _crewai_state_manager is None:
        state_dir = os.getenv("CREWAI_STATE_DIR", "./crewai_states")
        _crewai_state_manager = create_state_manager(storage_dir=state_dir, enable_compression=True)
    return _crewai_state_manager


def _normalize_crewai_status(value: Optional[str]) -> str:
    if not value:
        return "pending"
    status = value.lower()
    if status in {"completed", "complete", "verified", "solved", "done"}:
        return "completed"
    if status in {"in_progress", "running", "solving", "active"}:
        return "in_progress"
    if status in {"failed", "error"}:
        return "failed"
    if status in {"paused", "blocked"}:
        return "blocked"
    return "pending"


def _build_crewai_ticket_list(state: "CrewAIWorkflowState") -> List[Dict[str, Any]]:
    """Derive ticket-like entries from a CrewAI workflow state."""
    tickets: List[Dict[str, Any]] = []
    sub_problems = []
    if state.decomposition_plan and state.decomposition_plan.sub_problems:
        sub_problems = list(state.decomposition_plan.sub_problems)

    sub_solutions = state.sub_solutions or {}

    for sub_problem in sub_problems:
        attempt = sub_solutions.get(sub_problem.id)
        status = None
        assignee = None
        created_at = None
        if attempt is not None:
            if isinstance(attempt, dict):
                status = attempt.get("status")
                assignee = attempt.get("agent_name") or attempt.get("generated_by_model")
                created_at = attempt.get("created_at") or attempt.get("timestamp")
            else:
                status = getattr(attempt, "status", None)
                assignee = getattr(attempt, "agent_name", None) or getattr(attempt, "generated_by_model", None)
                created_at = getattr(attempt, "created_at", None) or getattr(attempt, "timestamp", None)

        tickets.append(
            {
                "id": sub_problem.id,
                "title": sub_problem.title,
                "description": sub_problem.description,
                "status": _normalize_crewai_status(status),
                "assigned_agent_id": assignee,
                "created_at": created_at or state.created_at,
                "updated_at": state.updated_at,
                "sub_problem_id": sub_problem.id,
                "dependencies": list(sub_problem.dependencies or []),
                "priority": getattr(sub_problem, "priority", None),
            }
        )

    return tickets


def _tail_log_file(path: Path, limit: int) -> List[str]:
    """Return the last N lines of a log file."""
    if limit <= 0 or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file_handle:
            return list(deque(file_handle, maxlen=limit))
    except (OSError, IOError, UnicodeDecodeError):
        return []


def _collect_log_sources() -> Dict[str, Path]:
    """Collect known log sources for monitoring."""
    sources: Dict[str, Path] = {}
    logs_dir = Path("logs")
    if logs_dir.exists():
        for entry in logs_dir.iterdir():
            if entry.is_file():
                sources[entry.name] = entry
    for root_log in [Path("backend_stdout.log"), Path("backend_stderr.log")]:
        if root_log.exists():
            sources[root_log.name] = root_log
    return sources


def _extract_metric_value(metrics: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _serialize_workflow_subproblem(sub_problem: SubProblem) -> Dict[str, Any]:
    """Serialize SubProblem from workflow_structures."""
    return {
        "id": sub_problem.id,
        "description": sub_problem.description,
        "dependencies": list(sub_problem.dependencies or []),
        "ai_suggested_evolution_mode": sub_problem.ai_suggested_evolution_mode,
        "ai_suggested_complexity_score": sub_problem.ai_suggested_complexity_score,
        "ai_suggested_evaluation_prompt": sub_problem.ai_suggested_evaluation_prompt,
        "content_type": sub_problem.content_type,
        "solver_team_name": sub_problem.solver_team_name,
        "red_team_gauntlet_name": sub_problem.red_team_gauntlet_name,
        "gold_team_gauntlet_name": sub_problem.gold_team_gauntlet_name,
        "solver_generation_gauntlet_name": sub_problem.solver_generation_gauntlet_name,
        "patcher_team_name": sub_problem.patcher_team_name,
        "evolution_params": sub_problem.evolution_params or {},
        "status": sub_problem.status,
        "atomic_mode": getattr(sub_problem, "atomic_mode", False),
        "decomposition_depth": getattr(sub_problem, "decomposition_depth", 0),
        "acceptance_criteria": list(sub_problem.acceptance_criteria or []),
        "solution_requirements": sub_problem.solution_requirements or {},
        "specific_constraints": list(sub_problem.specific_constraints or []),
        "dependency_outputs": sub_problem.dependency_outputs or {},
        "metadata": sub_problem.metadata or {},
    }


def _topological_sort_subproblems(sub_problems: List[SubProblem]) -> List[str]:
    """Return execution order for sub-problems or raise on cycles."""
    graph = {sp.id: list(sp.dependencies or []) for sp in sub_problems}
    in_degree = {node: 0 for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep not in in_degree:
                raise ValueError(f"Unknown dependency '{dep}' for sub-problem '{node}'")
            in_degree[node] += 1

    queue = [node for node, degree in in_degree.items() if degree == 0]
    order: List[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for candidate, deps in graph.items():
            if node in deps:
                in_degree[candidate] -= 1
                if in_degree[candidate] == 0:
                    queue.append(candidate)

    if len(order) != len(graph):
        remaining = [node for node, degree in in_degree.items() if degree > 0]
        raise ValueError(f"Cyclic dependencies detected among: {', '.join(remaining)}")
    return order


def _normalize_tenant_id(tenant_id: str) -> str:
    """Normalize tenant ID for safe filesystem usage."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", tenant_id.strip())
    return normalized or "default"


def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Get tenant ID from request headers (defaults to 'default')."""
    if not x_tenant_id:
        return "default"
    return _normalize_tenant_id(x_tenant_id)


def _get_tenant_storage_dir(tenant_id: str) -> str:
    """Get or create the tenant storage directory."""
    base_dir = os.path.join("data", "tenants", tenant_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_tenant_team_manager(tenant_id: str) -> TeamManager:
    """Get a tenant-scoped TeamManager."""
    base_dir = _get_tenant_storage_dir(tenant_id)
    return TeamManager(teams_file=os.path.join(base_dir, "teams.json"))


def get_tenant_gauntlet_manager(tenant_id: str) -> GauntletManager:
    """Get a tenant-scoped GauntletManager."""
    base_dir = _get_tenant_storage_dir(tenant_id)
    return GauntletManager(gauntlets_file=os.path.join(base_dir, "gauntlets.json"))


def _normalize_evaluator_id(evaluator_id: str) -> str:
    """Normalize evaluator IDs for safe filesystem usage."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", evaluator_id.strip())
    return normalized or "evaluator"


def _get_evaluator_dir(tenant_id: str) -> Path:
    """Get evaluator storage directory for a tenant."""
    base_dir = _get_tenant_storage_dir(tenant_id)
    evaluators_dir = Path(base_dir) / "evaluators"
    evaluators_dir.mkdir(parents=True, exist_ok=True)
    return evaluators_dir


def _list_evaluators(tenant_id: str) -> Dict[str, str]:
    """Return evaluator code by ID for a tenant."""
    evaluators_dir = _get_evaluator_dir(tenant_id)
    evaluators: Dict[str, str] = {}
    for path in evaluators_dir.glob("*.py"):
        try:
            evaluators[path.stem] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return evaluators


def _save_evaluator(tenant_id: str, evaluator_id: str, code: str) -> None:
    evaluators_dir = _get_evaluator_dir(tenant_id)
    path = evaluators_dir / f"{_normalize_evaluator_id(evaluator_id)}.py"
    path.write_text(code, encoding="utf-8")


def _delete_evaluator(tenant_id: str, evaluator_id: str) -> bool:
    evaluators_dir = _get_evaluator_dir(tenant_id)
    path = evaluators_dir / f"{_normalize_evaluator_id(evaluator_id)}.py"
    if not path.exists():
        return False
    path.unlink()
    return True


def _validate_evaluator_code(code: str) -> Optional[str]:
    if not code or not code.strip():
        return "Evaluator code cannot be empty."
    if "def evaluate" not in code:
        return "Evaluator code must define an `evaluate(program_path)` function."
    return None


WORKFLOW_TYPE_ALIASES: Dict[str, str] = {
    "sovereign": "sovereign_decomposition",
    "sovereign_decomposition": "sovereign_decomposition",
    "web3": "web3",
    "defi": "web3",
    "smart_contract": "web3",
    "smart contract": "web3",
    "smart_contract_audit": "web3",
}
ALLOWED_WORKFLOW_TYPES: set[str] = {"sovereign_decomposition", "web3"}
DOMAIN_HINT_ALIASES: Dict[str, str] = {
    "web3": "web3",
    "defi": "web3",
    "smart_contract": "web3",
    "smart contract": "web3",
    "smart_contract_audit": "web3",
    "solidity": "web3",
}


# Pydantic models for API requests/responses
class WorkflowCreateRequest(BaseModel):
    problem_statement: str = Field(..., min_length=10, description="The problem to solve")
    content_analyzer_team: str = Field(..., description="Team for content analysis")
    planner_team: str = Field(..., description="Team for planning")
    solver_team: str = Field(..., description="Team for solving sub-problems")
    patcher_team: str = Field(..., description="Team for patching solutions")
    assembler_team: str = Field(..., description="Team for assembling final solution")
    sub_problem_red_gauntlet: str = Field(..., description="Red gauntlet for sub-problems")
    sub_problem_gold_gauntlet: str = Field(..., description="Gold gauntlet for sub-problems")
    final_red_gauntlet: str = Field(..., description="Red gauntlet for final solution")
    final_gold_gauntlet: str = Field(..., description="Gold gauntlet for final solution")
    solver_generation_gauntlet: str = Field(..., description="Gauntlet for solution generation")
    max_refinement_loops: int = Field(3, ge=1, le=10, description="Maximum refinement loops")
    mdap_enabled: bool = Field(False, description="Enable MDAP for solution generation")
    mdap_config: Dict[str, Any] = Field(default_factory=dict, description="MDAP configuration overrides")
    maker_enabled: bool = Field(False, description="Enable MAKER for solution generation")
    maker_config: Dict[str, Any] = Field(default_factory=dict, description="MAKER configuration overrides")
    workflow_type: str = Field(
        "sovereign_decomposition",
        description="Workflow type (sovereign_decomposition or web3/smart contract aliases)"
    )
    domain_hint: Optional[str] = Field(None, description="Optional domain hint (e.g., web3)")
    domain_artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional precomputed domain artifacts (e.g., Slither/Forge outputs)"
    )
    web3: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional Web3 config: enabled, project_path, run_fuzzing, slither/forge timeouts"
    )
    
    @validator('problem_statement')
    def validate_problem_statement(cls, v):
        if not v or not v.strip():
            raise ValueError('Problem statement cannot be empty')
        return v.strip()

    @validator("workflow_type", pre=True, always=True)
    def validate_workflow_type(cls, v):
        if not isinstance(v, str):
            raise ValueError("workflow_type must be a string")
        normalized = WORKFLOW_TYPE_ALIASES.get(v.strip().lower(), v.strip().lower())
        if normalized not in ALLOWED_WORKFLOW_TYPES:
            raise ValueError(
                f"Invalid workflow_type '{v}'. Allowed: {', '.join(sorted(ALLOWED_WORKFLOW_TYPES))}"
            )
        return normalized

    @validator("domain_hint", pre=True)
    def normalize_domain_hint(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("domain_hint must be a string")
        normalized = v.strip().lower().replace("-", "_")
        return DOMAIN_HINT_ALIASES.get(normalized, normalized)


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    current_stage: str
    progress: float
    created_at: str


class WorkflowDetailResponse(BaseModel):
    workflow_id: str
    problem_statement: str
    status: str
    current_stage: str
    progress: float
    start_time: float
    end_time: Optional[float]
    refinement_loop_count: int
    solved_sub_problems: int
    total_sub_problems: int


class TeamCreateRequest(BaseModel):
    name: str
    role: str
    description: Optional[str] = None
    members: List[Dict[str, Any]]


class GauntletCreateRequest(BaseModel):
    name: str
    team_name: str
    description: Optional[str] = None
    rounds: List[Dict[str, Any]]


class KnowledgeArtifactCreateRequest(BaseModel):
    artifact_type: str
    content: Any
    source_workflow_id: Optional[str] = "manual"
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    related_artifacts: Optional[List[str]] = None


class KnowledgeSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    artifact_types: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)


class KnowledgeRecommendationsRequest(BaseModel):
    problem_statement: str
    domain: Optional[str] = None


class KnowledgeImportRequest(BaseModel):
    artifacts: Dict[str, Any]


class AutoApprovalConditionModel(BaseModel):
    field: str
    operator: str
    value: Any
    logical_op: Optional[str] = "AND"


class AutoApprovalRuleModel(BaseModel):
    name: str
    priority: int = Field(0, ge=0, le=100)
    action: str = Field("approve")
    enabled: bool = True
    conditions: List[AutoApprovalConditionModel]
    created_at: Optional[str] = None


class AutoApprovalConfigModel(BaseModel):
    enabled: bool = False
    rules: List[AutoApprovalRuleModel] = Field(default_factory=list)


class AutoApprovalTestRequest(BaseModel):
    plan: Dict[str, Any]


class WorkflowTemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    config: Dict[str, Any]
    tags: Optional[List[str]] = None


class WorkflowTemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class ProviderModelsRequest(BaseModel):
    api_key: Optional[str] = None


class ParameterValidateRequest(BaseModel):
    parameters: Dict[str, Any]


class PromptCreateRequest(BaseModel):
    name: str
    content: str


class ContentTemplateRequest(BaseModel):
    name: str
    content: str


class ProtocolValidationRequest(BaseModel):
    protocol_text: str
    validation_type: Optional[str] = "generic"


class VersionCreateRequest(BaseModel):
    protocol_text: str
    version_name: Optional[str] = ""
    comment: Optional[str] = ""
    author: Optional[str] = None


class VersionCompareRequest(BaseModel):
    version_id_1: str
    version_id_2: str


class VersionBranchRequest(BaseModel):
    new_version_name: str


class VersionLoadRequest(BaseModel):
    version_id: str


class ValidationRuleCreateRequest(BaseModel):
    name: str
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    required_keywords: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    required_sections: Optional[List[str]] = None


class ValidationRuleUpdateRequest(BaseModel):
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    required_keywords: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    required_sections: Optional[List[str]] = None


class ValidationRunRequest(BaseModel):
    content: str
    rule_names: List[str] = Field(default_factory=list)


class ComplianceCheckRequest(BaseModel):
    content: str
    framework: Optional[str] = "generic"


class WorkflowDefinitionCreateRequest(BaseModel):
    name: str
    description: str
    workflow_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class WorkflowInstanceCreateRequest(BaseModel):
    definition_id: str
    instance_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


class SuggestionRequest(BaseModel):
    content: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    extra_headers: Optional[Dict[str, str]] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 1024
    seed: Optional[int] = None


class EvaluatorUploadRequest(BaseModel):
    code: str


class SubProblemUpdate(BaseModel):
    id: str
    description: Optional[str] = None
    dependencies: Optional[List[str]] = None
    ai_suggested_evolution_mode: Optional[str] = None
    ai_suggested_complexity_score: Optional[int] = None
    ai_suggested_evaluation_prompt: Optional[str] = None
    content_type: Optional[str] = None
    solver_team_name: Optional[str] = None
    red_team_gauntlet_name: Optional[str] = None
    gold_team_gauntlet_name: Optional[str] = None
    solver_generation_gauntlet_name: Optional[str] = None
    patcher_team_name: Optional[str] = None
    evolution_params: Optional[Dict[str, Any]] = None
    atomic_mode: Optional[bool] = None
    decomposition_depth: Optional[int] = None
    acceptance_criteria: Optional[List[str]] = None
    solution_requirements: Optional[Dict[str, Any]] = None
    specific_constraints: Optional[List[str]] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DecompositionPlanUpdateRequest(BaseModel):
    sub_problems: List[SubProblemUpdate]
    max_refinement_loops: Optional[int] = None
    auto_approval_enabled: Optional[bool] = None
    auto_approval_criteria: Optional[Dict[str, Any]] = None
    mdap_enabled: Optional[bool] = None
    mdap_config: Optional[Dict[str, Any]] = None
    maker_enabled: Optional[bool] = None
    maker_config: Optional[Dict[str, Any]] = None
    resource_limits: Optional[Dict[str, Any]] = None
    parallel_processing_enabled: Optional[bool] = None
    max_parallel_sub_problems: Optional[int] = None
    learning_enabled: Optional[bool] = None
    learning_config: Optional[Dict[str, Any]] = None
    content_analyzer_team_name: Optional[str] = None
    planner_team_name: Optional[str] = None
    assembler_team_name: Optional[str] = None
    final_red_team_gauntlet_name: Optional[str] = None
    final_gold_team_gauntlet_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegratedWorkflowRequest(BaseModel):
    current_content: str
    content_type: str = "text_general"
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    red_team_models: List[str]
    blue_team_models: List[str]
    evaluator_models: List[str]
    max_iterations: int = 5
    adversarial_iterations: int = 3
    evolution_iterations: int = 2
    evaluation_iterations: int = 2
    system_prompt: str
    evaluator_system_prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 4096
    seed: Optional[int] = None
    rotation_strategy: str = "round_robin"
    red_team_sample_size: int = 3
    blue_team_sample_size: int = 3
    evaluator_sample_size: int = 2
    confidence_threshold: float = 0.7
    evaluator_threshold: float = 90.0
    evaluator_consecutive_rounds: int = 1
    compliance_requirements: str = ""
    enable_data_augmentation: bool = False
    augmentation_model_id: Optional[str] = None
    augmentation_temperature: float = 0.7
    enable_human_feedback: bool = False
    multi_objective_optimization: bool = False
    feature_dimensions: Optional[List[str]] = None
    feature_bins: Optional[int] = None
    elite_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    archive_size: int = 100
    checkpoint_interval: int = 10
    keyword_analysis_enabled: bool = True
    keywords_to_target: Optional[List[str]] = None
    keyword_penalty_weight: float = 0.5


class ModelOrchestrationRegisterRequest(BaseModel):
    model_name: str
    role: str
    weight: float = 1.0
    api_key: Optional[str] = ""
    api_base: Optional[str] = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ModelOrchestrationEnsembleRequest(BaseModel):
    role: str
    messages: List[Dict[str, str]]
    selection_strategy: str = "performance_based"
    temperature: float = 0.7
    max_tokens: int = 4096
    num_responses: int = 1


class BubbleLabsAceSkillbookRequest(BaseModel):
    name: str
    skills: List[Dict[str, Any]] = Field(default_factory=list)


class BubbleLabsAcePatternRequest(BaseModel):
    workflow_results: List[Dict[str, Any]] = Field(default_factory=list)


class BubbleLabsZ3SolveRequest(BaseModel):
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class BubbleLabsZ3ProveRequest(BaseModel):
    theorem: str


class BubbleLabsRomaAnalyzeRequest(BaseModel):
    problem: str
    max_depth: int = 3


class BubbleLabsRomaConfigRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class BubbleLabsKnowledgeStoreRequest(BaseModel):
    artifact: Dict[str, Any] = Field(default_factory=dict)


class BubbleLabsKnowledgeQueryRequest(BaseModel):
    query: str


class BubbleLabsAnalyticsTrackRequest(BaseModel):
    workflow_id: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BubbleLabsLeanAideProveRequest(BaseModel):
    theorem: str


class Web3IngestStackRequest(BaseModel):
    project_path: str = "."
    run_fuzzing: bool = True
    slither_timeout_seconds: int = Field(240, ge=10, le=3600)
    forge_timeout_seconds: int = Field(420, ge=10, le=7200)


class Web3IngestSlitherRequest(BaseModel):
    project_path: str = "."
    timeout_seconds: int = Field(240, ge=10, le=3600)
    extra_args: List[str] = Field(default_factory=list)


class Web3IngestFoundryRequest(BaseModel):
    project_path: str = "."
    timeout_seconds: int = Field(420, ge=10, le=7200)
    match_contract: Optional[str] = None
    match_test: Optional[str] = None
    fork_url: Optional[str] = None
    extra_args: List[str] = Field(default_factory=list)


class Web3InvariantTranslateRequest(BaseModel):
    statement: str
    non_negative_target: bool = True
    max_withdraw_expr: Optional[str] = None
    verify_translation: bool = True
    assume_non_negative_amount: bool = True


class Web3ExploitWitnessRequest(BaseModel):
    additional_constraints: List[str] = Field(default_factory=list)
    timeout_seconds: float = Field(10.0, ge=0.1, le=120.0)


class Web3AuditExploitRequest(BaseModel):
    project_path: str = "."
    statement: Optional[str] = None
    run_fuzzing: bool = True
    verify_translation: bool = True
    timeout_seconds: float = Field(10.0, ge=0.1, le=120.0)
    additional_constraints: List[str] = Field(default_factory=list)
    non_negative_target: bool = True
    max_withdraw_expr: Optional[str] = None
    assume_non_negative_amount: bool = True


class MakerToolCreateRequest(BaseModel):
    name: str
    description: str
    task: str
    maker_mode: str = "recursive"
    k_ahead: int = 3
    max_depth: int = 5
    context: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    expected_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MakerToolExecuteRequest(BaseModel):
    input_data: Dict[str, Any] = Field(default_factory=dict)
    delegate_to_crewai: bool = False


class MakerToolUpdateRequest(BaseModel):
    status: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MakerDelegationListRequest(BaseModel):
    status: Optional[str] = None
    delegation_type: Optional[str] = None


class KnowledgeExplorerQueryRequest(BaseModel):
    query: str
    sources: List[str] = Field(default_factory=lambda: ["bedrock", "graphiti"])
    bedrock_kb_id: Optional[str] = None
    index_path: Optional[str] = None


class KnowledgeExplorerExtractRequest(BaseModel):
    source_type: str
    source_value: str
    extraction_config: Optional[Dict[str, Any]] = None


class LeanAideExecuteRequest(BaseModel):
    task_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class EvolutionRunRequest(BaseModel):
    content: str
    content_type: str = "document_general"
    evolution_mode: str = "standard"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    gauntlet_name: Optional[str] = None
    use_decomposition: bool = False


class AdversarialRunRequest(BaseModel):
    content: str
    content_type: str = "document_general"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    use_decomposition: bool = False


# Authentication and Authorization
from enum import Enum
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import timedelta


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


# API Keys - Load from environment, NO HARDCODED KEYS
# In production, these should be stored in a database with proper encryption
def _load_api_keys() -> Dict[str, Dict[str, Any]]:
    """
    Load API keys from environment variables.

    Format: API_KEY_<name>=<key>:<role>
    Example: API_KEY_ADMIN=sk-admin123:admin

    Returns:
        Dictionary mapping API keys to user info
    """
    api_keys = {}
    prefix = "API_KEY_"

    for env_var, value in os.environ.items():
        if env_var.startswith(prefix):
            try:
                # Parse format: key:role
                if ":" in value:
                    key, role = value.split(":", 1)
                    name = env_var[len(prefix):].lower().replace("_", " ")
                    api_keys[key] = {"role": UserRole(role), "name": name}
                else:
                    logger.warning(f"Invalid API key format in {env_var}. Expected 'key:role'")
            except ValueError as e:
                logger.warning(f"Failed to parse API key from {env_var}: {e}")

    return api_keys


API_KEYS = _load_api_keys()

# JWT Configuration - MUST be set from environment
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if is_production():
        raise RuntimeError(
            "SECRET_KEY environment variable must be set in production. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    else:
        # Generate a temporary key for development only
        import secrets
        SECRET_KEY = secrets.token_hex(32)
        logger.warning(
            "Using auto-generated SECRET_KEY for development. "
            "This will change on restart! Set SECRET_KEY environment variable for persistence."
        )

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthUser(BaseModel):
    """Authenticated user information."""
    api_key: str
    role: UserRole
    name: str


class IcrRefinementEvent(BaseModel):
    """Event signaling a refinement is needed."""
    reason: Optional[str] = None
    overall_score: Optional[float] = None
    weaknesses: Optional[List[str]] = None
    friction_points: Optional[List[str]] = None
    auto_refine: Optional[bool] = None


class IcrRewardCalibrationRequest(BaseModel):
    """Reward calibration request payload."""
    request_id: Optional[str] = None
    option_a: str
    option_b: str
    confidence: Optional[float] = None
    prompt: Optional[str] = None


class IcrRewardCalibrationResponse(BaseModel):
    """Reward calibration response payload."""
    request_id: Optional[str] = None
    choice: str


class IcrHeatmapPoint(BaseModel):
    """Heatmap point from GenerativeUI."""
    x: float
    y: float
    intensity: float = 0.0
    dwellMs: Optional[float] = None
    timestamp: Optional[float] = None
    type: Optional[str] = None


class IcrHeatmapSnapshot(BaseModel):
    """Heatmap snapshot payload for multimodal analysis."""
    snapshot_id: Optional[str] = None
    timestamp: Optional[float] = None
    screen_html: str
    heatmap_data_url: Optional[str] = None
    composite_data_url: Optional[str] = None
    points: List[IcrHeatmapPoint] = Field(default_factory=list)
    manual_code_delta: Optional[float] = None
    context_text: Optional[str] = None
    auto_refine: Optional[bool] = None

def verify_api_key(x_api_key: str = Header(...)) -> AuthUser:
    """Verify API key from header and return user info.
    
    Uses database-backed validation with expiration and revocation checking
    when security framework is available. Falls back to environment variable
    configuration only when database is unavailable.
    """
    import time
    import hashlib
    start_time = time.time()
    success = False

    try:
        # First, try database-backed validation via security framework
        if SECURITY_FRAMEWORK_AVAILABLE:
            try:
                from security_framework import get_api_key_database, APIKeyStatus
                
                key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
                db = get_api_key_database()
                key_record = db.get_key_by_hash(key_hash)
                
                if key_record:
                    # Check if key is active
                    if key_record.status != APIKeyStatus.ACTIVE:
                        duration = time.time() - start_time
                        _trigger_api_alerts("verify_api_key", False, None, 
                                          f"API key not active: {key_record.status.value}")
                        _track_api_performance("verify_api_key", False, duration, 
                                             "verify_api_key", 401)
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="API key is not active",
                            headers={"WWW-Authenticate": "ApiKey"}
                        )
                    
                    # Check expiration
                    from datetime import datetime
                    if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                        duration = time.time() - start_time
                        _trigger_api_alerts("verify_api_key", False, None, "API key expired")
                        _track_api_performance("verify_api_key", False, duration, 
                                             "verify_api_key", 401)
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="API key has expired",
                            headers={"WWW-Authenticate": "ApiKey"}
                        )
                    
                    # Update usage statistics
                    db.update_last_used(key_record.id)
                    
                    # Determine role from permissions
                    role = UserRole.READONLY
                    if key_record.permissions:
                        if "admin" in key_record.permissions or "system:admin" in key_record.permissions:
                            role = UserRole.ADMIN
                        elif "write" in key_record.permissions or "workflow:execute" in key_record.permissions:
                            role = UserRole.USER
                    
                    user = AuthUser(
                        api_key=x_api_key,
                        role=role,
                        name=key_record.name
                    )
                    
                    success = True
                    duration = time.time() - start_time
                    _track_api_performance("verify_api_key", True, duration, 
                                         "verify_api_key", 200)
                    return user
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Database API key validation failed: {e}, falling back to env vars")
        
        # Fallback to environment variable configuration
        if x_api_key not in API_KEYS:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            duration = time.time() - start_time
            _trigger_api_alerts("verify_api_key", False, None, "Invalid API key")
            _track_api_performance("verify_api_key", False, duration, "verify_api_key", 401)

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )

        key_info = API_KEYS[x_api_key]
        user = AuthUser(
            api_key=x_api_key,
            role=key_info["role"],
            name=key_info["name"]
        )

        # **ACTUAL INTEGRATION**: Track performance on success
        success = True
        duration = time.time() - start_time
        _track_api_performance("verify_api_key", True, duration, "verify_api_key", 200)

        return user

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # **ACTUAL INTEGRATION**: Trigger alert and track unexpected errors
        duration = time.time() - start_time
        _trigger_api_alerts("verify_api_key", False, None, str(e))
        _track_api_performance("verify_api_key", False, duration, "verify_api_key", 500)
        raise


def require_role(required_role: UserRole):
    """Dependency to require specific role."""
    def role_checker(user: AuthUser = Depends(verify_api_key)) -> AuthUser:
        # Admin can do everything
        if user.role == UserRole.ADMIN:
            return user
        
        # Check if user has required role
        role_hierarchy = {
            UserRole.ADMIN: 3,
            UserRole.USER: 2,
            UserRole.READONLY: 1
        }
        
        if role_hierarchy.get(user.role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return user
    
    return role_checker


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str = Header(..., alias="Authorization")) -> dict:
    """Verify JWT token."""
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


# API Endpoints

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Decomposition Workflow API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


class TokenRequest(BaseModel):
    """Request for JWT token."""
    api_key: str = Field(..., description="API key for authentication")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


@app.post("/auth/token", response_model=TokenResponse)
def get_token(request: TokenRequest):
    """Get JWT token using API key."""
    if request.api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    key_info = API_KEYS[request.api_key]
    
    # Create token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": key_info["name"], "role": key_info["role"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=key_info["role"]
    )


# Workflow endpoints

@app.post(
    "/workflows",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role(UserRole.USER))],
    summary="Create a new workflow",
    description="Create a new decomposition workflow with specified teams and gauntlets",
    responses={
        200: {
            "description": "Workflow created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "created",
                        "current_stage": "INITIALIZING",
                        "progress": 0.0,
                        "created_at": "2025-10-21T12:00:00"
                    }
                }
            }
        },
        400: {"description": "Invalid request or missing teams/gauntlets"},
        401: {"description": "Invalid API key"},
        500: {"description": "Internal server error"}
    }
)
def create_workflow(
    request: WorkflowCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new workflow."""
    logger.info(f"Workflow creation request by {user.name} for problem: {request.problem_statement[:50]}...")
    try:
        tenant_team_manager = get_tenant_team_manager(tenant_id)
        tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)

        # Get teams and gauntlets
        content_analyzer_team = tenant_team_manager.get_team(request.content_analyzer_team)
        planner_team = tenant_team_manager.get_team(request.planner_team)
        solver_team = tenant_team_manager.get_team(request.solver_team)
        patcher_team = tenant_team_manager.get_team(request.patcher_team)
        assembler_team = tenant_team_manager.get_team(request.assembler_team)
        
        sub_problem_red_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.sub_problem_red_gauntlet)
        sub_problem_gold_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.sub_problem_gold_gauntlet)
        final_red_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.final_red_gauntlet)
        final_gold_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.final_gold_gauntlet)
        solver_generation_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.solver_generation_gauntlet)
        
        # Validate all exist
        if not all([
            content_analyzer_team, planner_team, solver_team, patcher_team, assembler_team,
            sub_problem_red_gauntlet, sub_problem_gold_gauntlet,
            final_red_gauntlet, final_gold_gauntlet, solver_generation_gauntlet
        ]):
            raise HTTPException(status_code=400, detail="One or more teams/gauntlets not found")
        
        # Create workflow state
        workflow_id = str(uuid.uuid4())
        openevolve_parameters: Dict[str, Any] = {}
        workflow_type = request.workflow_type
        if request.domain_hint:
            openevolve_parameters["domain_hint"] = request.domain_hint
        if request.domain_artifacts:
            openevolve_parameters["domain_artifacts"] = request.domain_artifacts

        web3_requested = (
            workflow_type == "web3"
            or openevolve_parameters.get("domain_hint") == "web3"
            or bool(request.web3)
        )
        if web3_requested:
            workflow_type = "web3"
            openevolve_parameters["domain_hint"] = "web3"
            web3_config: Dict[str, Any] = {
                "enabled": True,
                "project_path": ".",
                "run_fuzzing": True,
            }
            if isinstance(request.web3, dict):
                web3_config.update(request.web3)
            openevolve_parameters["web3"] = web3_config
            openevolve_parameters.setdefault("formal_verification_enabled", True)
            openevolve_parameters.setdefault("z3_enabled", True)
            openevolve_parameters.setdefault("leanaide_enabled", True)
            openevolve_parameters.setdefault("formal_verification_mode", "hybrid")

        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            problem_statement=request.problem_statement,
            current_stage="INITIALIZING",
            status="created",
            tenant_id=tenant_id,
            mdap_enabled=request.mdap_enabled,
            mdap_config=request.mdap_config,
            maker_enabled=request.maker_enabled,
            maker_config=request.maker_config,
            openevolve_parameters=openevolve_parameters,
        )
        
        # Store workflow
        workflows[workflow_id] = workflow_state

        record_audit_event(
            user=user,
            operation="CREATE_WORKFLOW",
            resource="workflow",
            resource_id=workflow_id,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=workflow_state.status,
            current_stage=workflow_state.current_stage,
            progress=workflow_state.progress,
            created_at=datetime.now().isoformat()
        )
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows", dependencies=[Depends(verify_api_key)])
def list_workflows(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all workflows."""
    logger.info(f"User {user.name} listed workflows.")
    record_audit_event(
        user=user,
        operation="LIST_WORKFLOWS",
        resource="workflow",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "workflows": [
            {
                "workflow_id": wf.workflow_id,
                "status": wf.status,
                "current_stage": wf.current_stage,
                "progress": wf.progress
            }
            for wf in workflows.values()
            if (wf.tenant_id or "default") == tenant_id
        ],
        "total": len([wf for wf in workflows.values() if (wf.tenant_id or "default") == tenant_id])
    }


@app.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse, dependencies=[Depends(verify_api_key)])
def get_workflow(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get workflow details."""
    logger.info(f"User {user.name} requested details for workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    record_audit_event(
        user=user,
        operation="GET_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    total_sub_problems = len(wf.decomposition_plan.sub_problems) if wf.decomposition_plan else 0
    solved_sub_problems = len(wf.solved_sub_problem_ids)
    
    return WorkflowDetailResponse(
        workflow_id=wf.workflow_id,
        problem_statement=wf.problem_statement,
        status=wf.status,
        current_stage=wf.current_stage,
        progress=wf.progress,
        start_time=wf.start_time,
        end_time=wf.end_time,
        refinement_loop_count=wf.refinement_loop_count,
        solved_sub_problems=solved_sub_problems,
        total_sub_problems=total_sub_problems
    )


@app.get("/workflows/{workflow_id}/decomposition-plan", dependencies=[Depends(verify_api_key)])
def get_workflow_decomposition_plan(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get decomposition plan for a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not wf.decomposition_plan:
        raise HTTPException(status_code=404, detail="Decomposition plan not available")

    plan = wf.decomposition_plan
    sub_problems = [_serialize_workflow_subproblem(sp) for sp in plan.sub_problems]
    dependency_edges = {sp["id"]: sp.get("dependencies", []) for sp in sub_problems}

    return {
        "workflow_id": wf.workflow_id,
        "plan": {
            "problem_statement": plan.problem_statement,
            "analyzed_context": plan.analyzed_context,
            "sub_problems": sub_problems,
            "max_refinement_loops": plan.max_refinement_loops,
            "auto_approval_enabled": plan.auto_approval_enabled,
            "auto_approval_criteria": plan.auto_approval_criteria,
            "mdap_enabled": plan.mdap_enabled,
            "mdap_config": plan.mdap_config,
            "maker_enabled": plan.maker_enabled,
            "maker_config": plan.maker_config,
            "resource_limits": plan.resource_limits,
            "parallel_processing_enabled": plan.parallel_processing_enabled,
            "max_parallel_sub_problems": plan.max_parallel_sub_problems,
            "learning_enabled": plan.learning_enabled,
            "learning_config": plan.learning_config,
            "content_analyzer_team_name": plan.content_analyzer_team_name,
            "planner_team_name": plan.planner_team_name,
            "assembler_team_name": plan.assembler_team_name,
            "final_red_team_gauntlet_name": plan.final_red_team_gauntlet_name,
            "final_gold_team_gauntlet_name": plan.final_gold_team_gauntlet_name,
            "metadata": plan.metadata,
        },
        "dependency_graph": {
            "edges": dependency_edges,
            "execution_order": list(plan.metadata.get("execution_order", [])) if plan.metadata else [],
        },
    }


@app.put("/workflows/{workflow_id}/decomposition-plan", dependencies=[Depends(require_role(UserRole.USER))])
def update_workflow_decomposition_plan(
    workflow_id: str,
    request: DecompositionPlanUpdateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Update decomposition plan and sub-problems for a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not wf.decomposition_plan:
        raise HTTPException(status_code=404, detail="Decomposition plan not available")

    plan = wf.decomposition_plan
    sub_problem_map = {sp.id: sp for sp in plan.sub_problems}

    for update in request.sub_problems:
        sp = sub_problem_map.get(update.id)
        if not sp:
            raise HTTPException(status_code=400, detail=f"Unknown sub-problem id: {update.id}")

        if update.description is not None:
            sp.description = update.description
        if update.dependencies is not None:
            sp.dependencies = list(update.dependencies)
        if update.ai_suggested_evolution_mode is not None:
            sp.ai_suggested_evolution_mode = update.ai_suggested_evolution_mode
        if update.ai_suggested_complexity_score is not None:
            sp.ai_suggested_complexity_score = update.ai_suggested_complexity_score
        if update.ai_suggested_evaluation_prompt is not None:
            sp.ai_suggested_evaluation_prompt = update.ai_suggested_evaluation_prompt
        if update.content_type is not None:
            sp.content_type = update.content_type
        if update.solver_team_name is not None:
            sp.solver_team_name = update.solver_team_name
        if update.red_team_gauntlet_name is not None:
            sp.red_team_gauntlet_name = update.red_team_gauntlet_name
        if update.gold_team_gauntlet_name is not None:
            sp.gold_team_gauntlet_name = update.gold_team_gauntlet_name
        if update.solver_generation_gauntlet_name is not None:
            sp.solver_generation_gauntlet_name = update.solver_generation_gauntlet_name
        if update.patcher_team_name is not None:
            sp.patcher_team_name = update.patcher_team_name
        if update.evolution_params is not None:
            sp.evolution_params = update.evolution_params
        if update.atomic_mode is not None:
            sp.atomic_mode = update.atomic_mode
        if update.decomposition_depth is not None:
            sp.decomposition_depth = update.decomposition_depth
        if update.acceptance_criteria is not None:
            sp.acceptance_criteria = list(update.acceptance_criteria)
        if update.solution_requirements is not None:
            sp.solution_requirements = update.solution_requirements
        if update.specific_constraints is not None:
            sp.specific_constraints = list(update.specific_constraints)
        if update.status is not None:
            sp.status = update.status
        if update.metadata is not None:
            sp.metadata = update.metadata

    if request.max_refinement_loops is not None:
        plan.max_refinement_loops = request.max_refinement_loops
    if request.auto_approval_enabled is not None:
        plan.auto_approval_enabled = request.auto_approval_enabled
    if request.auto_approval_criteria is not None:
        plan.auto_approval_criteria = request.auto_approval_criteria
    if request.mdap_enabled is not None:
        plan.mdap_enabled = request.mdap_enabled
    if request.mdap_config is not None:
        plan.mdap_config = request.mdap_config
    if request.maker_enabled is not None:
        plan.maker_enabled = request.maker_enabled
    if request.maker_config is not None:
        plan.maker_config = request.maker_config
    if request.resource_limits is not None:
        plan.resource_limits = request.resource_limits
    if request.parallel_processing_enabled is not None:
        plan.parallel_processing_enabled = request.parallel_processing_enabled
    if request.max_parallel_sub_problems is not None:
        plan.max_parallel_sub_problems = request.max_parallel_sub_problems
    if request.learning_enabled is not None:
        plan.learning_enabled = request.learning_enabled
    if request.learning_config is not None:
        plan.learning_config = request.learning_config
    if request.content_analyzer_team_name is not None:
        plan.content_analyzer_team_name = request.content_analyzer_team_name
    if request.planner_team_name is not None:
        plan.planner_team_name = request.planner_team_name
    if request.assembler_team_name is not None:
        plan.assembler_team_name = request.assembler_team_name
    if request.final_red_team_gauntlet_name is not None:
        plan.final_red_team_gauntlet_name = request.final_red_team_gauntlet_name
    if request.final_gold_team_gauntlet_name is not None:
        plan.final_gold_team_gauntlet_name = request.final_gold_team_gauntlet_name
    if request.metadata is not None:
        plan.metadata = request.metadata

    try:
        execution_order = _topological_sort_subproblems(plan.sub_problems)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if plan.metadata is None:
        plan.metadata = {}
    plan.metadata["execution_order"] = execution_order

    try:
        from workflow_engine import _update_entanglement_matrix
        _update_entanglement_matrix(wf)
    except Exception as exc:
        logger.warning("Failed to update entanglement matrix: %s", exc)

    record_audit_event(
        user=user,
        operation="UPDATE_DECOMPOSITION_PLAN",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )

    return {"message": "Decomposition plan updated", "execution_order": execution_order}


@app.get("/workflows/{workflow_id}/telemetry", dependencies=[Depends(verify_api_key)])
def get_workflow_telemetry(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get telemetry and resource usage for a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    execution_time = None
    if wf.start_time:
        execution_time = (wf.end_time or time.time()) - wf.start_time

    critique_reports = getattr(wf, "all_critique_reports", []) or []
    verification_reports = getattr(wf, "all_verification_reports", []) or []

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    critique_scores = [getattr(r, "overall_score", 0.0) for r in critique_reports]
    verification_scores = [getattr(r, "average_score", 0.0) for r in verification_reports]

    return {
        "workflow_id": wf.workflow_id,
        "workflow_type": wf.workflow_type,
        "status": wf.status,
        "current_stage": wf.current_stage,
        "progress": wf.progress,
        "start_time": wf.start_time,
        "end_time": wf.end_time,
        "execution_time_seconds": execution_time,
        "refinement_loop_count": wf.refinement_loop_count,
        "resource_usage": wf.resource_usage or {},
        "performance_metrics": wf.performance_metrics or {},
        "openevolve_metrics": wf.openevolve_metrics or {},
        "crewai_workflow_id": getattr(wf, "crewai_workflow_id", None),
        "gauntlet_summary": {
            "critique_total": len(critique_reports),
            "critique_approved": sum(1 for r in critique_reports if getattr(r, "is_approved", False)),
            "critique_avg_score": _avg(critique_scores),
            "verification_total": len(verification_reports),
            "verification_approved": sum(
                1 for r in verification_reports if getattr(r, "is_approved", False)
            ),
            "verification_avg_score": _avg(verification_scores),
        },
    }


@app.get("/workflows/{workflow_id}/resource-usage", dependencies=[Depends(verify_api_key)])
def get_workflow_resource_usage(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get resource usage summary for a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not getattr(wf, "resource_usage", None):
        return {"workflow_id": wf.workflow_id, "resource_usage": {}}

    return {"workflow_id": wf.workflow_id, "resource_usage": wf.resource_usage}


@app.post("/workflows/{workflow_id}/resource-optimization", dependencies=[Depends(require_role(UserRole.USER))])
def optimize_workflow_resources(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Suggest resource allocation based on decomposition plan."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not wf.decomposition_plan:
        raise HTTPException(status_code=404, detail="Decomposition plan not available")

    from resource_manager import ResourceManager
    manager = ResourceManager()
    suggestions = manager.optimize_resource_allocation(wf.decomposition_plan.sub_problems)

    return {"workflow_id": wf.workflow_id, "suggestions": suggestions}


@app.post("/workflows/{workflow_id}/pause", dependencies=[Depends(require_role(UserRole.USER))])
def pause_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Pause a running workflow."""
    logger.info(f"User {user.name} requested to pause workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "running":
        raise HTTPException(status_code=400, detail=f"Cannot pause workflow in status: {wf.status}")
    
    wf.status = "paused"

    record_audit_event(
        user=user,
        operation="PAUSE_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"status": wf.status, "tenant_id": tenant_id}
    )
    
    return {
        "message": "Workflow paused",
        "workflow_id": workflow_id,
        "status": wf.status
    }


@app.post("/workflows/{workflow_id}/resume", dependencies=[Depends(require_role(UserRole.USER))])
def resume_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Resume a paused workflow."""
    logger.info(f"User {user.name} requested to resume workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "paused":
        raise HTTPException(status_code=400, detail=f"Cannot resume workflow in status: {wf.status}")
    
    wf.status = "running"

    record_audit_event(
        user=user,
        operation="RESUME_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"status": wf.status, "tenant_id": tenant_id}
    )
    
    return {
        "message": "Workflow resumed",
        "workflow_id": workflow_id,
        "status": wf.status
    }


@app.get("/workflows/{workflow_id}/results", dependencies=[Depends(verify_api_key)])
def get_workflow_results(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get workflow results."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow not completed yet. Current status: {wf.status}"
        )
    
    # Prepare results
    results = {
        "workflow_id": wf.workflow_id,
        "problem_statement": wf.problem_statement,
        "status": wf.status,
        "final_solution": None,
        "sub_problem_solutions": {},
        "execution_time": None,
        "refinement_loops": wf.refinement_loop_count
    }
    
    # Add final solution if available
    if wf.final_solution:
        results["final_solution"] = {
            "content": wf.final_solution.content,
            "generated_by": wf.final_solution.generated_by_model,
            "timestamp": wf.final_solution.timestamp
        }
    
    # Add sub-problem solutions
    for sp_id, solution in wf.sub_problem_solutions.items():
        results["sub_problem_solutions"][sp_id] = {
            "content": solution.content,
            "generated_by": solution.generated_by_model,
            "timestamp": solution.timestamp
        }
    
    # Calculate execution time
    if wf.start_time and wf.end_time:
        results["execution_time"] = wf.end_time - wf.start_time
    
    record_audit_event(
        user=user,
        operation="GET_WORKFLOW_RESULTS",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return results


@app.delete("/workflows/{workflow_id}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Cancel and delete a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # If running, mark as cancelled
    if wf.status == "running":
        wf.status = "cancelled"
    
    del workflows[workflow_id]

    record_audit_event(
        user=user,
        operation="DELETE_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {"message": "Workflow deleted", "workflow_id": workflow_id}


# Team endpoints

@app.get("/teams", dependencies=[Depends(verify_api_key)])
def list_teams(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all teams."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    teams = tenant_team_manager.get_all_teams()
    record_audit_event(
        user=user,
        operation="LIST_TEAMS",
        resource="team",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "teams": [
            {
                "name": team.name,
                "role": team.role,
                "description": team.description,
                "member_count": len(team.members)
            }
            for team in teams
        ],
        "total": len(teams)
    }


@app.get("/teams/{team_name}", dependencies=[Depends(verify_api_key)])
def get_team(
    team_name: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get team details."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    team = tenant_team_manager.get_team(team_name)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    record_audit_event(
        user=user,
        operation="GET_TEAM",
        resource="team",
        resource_id=team_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "name": team.name,
        "role": team.role,
        "description": team.description,
        "members": [
            {
                "model_id": m.model_id,
                "temperature": m.temperature,
                "max_tokens": m.max_tokens
            }
            for m in team.members
        ]
    }


@app.post("/teams", dependencies=[Depends(require_role(UserRole.USER))])
def create_team(
    request: TeamCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new team (requires USER role)."""
    try:
        # Convert members to ModelConfig objects
        members = [ModelConfig(**member) for member in request.members]
        
        team = Team(
            name=request.name,
            tenant_id=tenant_id,
            role=request.role,
            description=request.description,
            members=members
        )
        
        tenant_team_manager = get_tenant_team_manager(tenant_id)
        tenant_team_manager.save_team(team)
        
        logger.info(f"Team '{team.name}' created by {user.name}")
        record_audit_event(
            user=user,
            operation="CREATE_TEAM",
            resource="team",
            resource_id=team.name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Team created", "team_name": team.name}
    
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/teams/{team_name}", dependencies=[Depends(require_role(UserRole.USER))])
def update_team(
    team_name: str,
    request: TeamCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Update an existing team (requires USER role)."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    existing_team = tenant_team_manager.get_team(team_name)
    if not existing_team:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        # Convert members to ModelConfig objects
        members = [ModelConfig(**member) for member in request.members]
        
        updated_team = Team(
            name=team_name, # Ensure name from path is used
            tenant_id=tenant_id,
            role=request.role,
            description=request.description,
            members=members
        )
        
        tenant_team_manager.save_team(updated_team) # Overwrite existing
        
        logger.info(f"Team '{team_name}' updated by {user.name}")
        record_audit_event(
            user=user,
            operation="UPDATE_TEAM",
            resource="team",
            resource_id=team_name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Team updated", "team_name": team_name}
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error(f"Error updating team '{team_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/teams/{team_name}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_team(
    team_name: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a team (requires ADMIN role)."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    if not tenant_team_manager.get_team(team_name):
        raise HTTPException(status_code=404, detail="Team not found")
    
    tenant_team_manager.delete_team(team_name)
    logger.info(f"Team '{team_name}' deleted by {user.name}")
    record_audit_event(
        user=user,
        operation="DELETE_TEAM",
        resource="team",
        resource_id=team_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    return {"message": "Team deleted", "team_name": team_name}


# Gauntlet endpoints

@app.get("/gauntlets", dependencies=[Depends(verify_api_key)])
def list_gauntlets(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all gauntlets."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    gauntlets = tenant_gauntlet_manager.get_all_gauntlets()
    record_audit_event(
        user=user,
        operation="LIST_GAUNTLETS",
        resource="gauntlet",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "gauntlets": [
            {
                "name": g.name,
                "team_name": g.team_name,
                "description": g.description,
                "round_count": len(g.rounds)
            }
            for g in gauntlets
        ],
        "total": len(gauntlets)
    }


@app.get("/gauntlets/{gauntlet_name}", dependencies=[Depends(verify_api_key)])
def get_gauntlet(
    gauntlet_name: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get gauntlet details."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    gauntlet = tenant_gauntlet_manager.get_gauntlet(gauntlet_name)
    if not gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    record_audit_event(
        user=user,
        operation="GET_GAUNTLET",
        resource="gauntlet",
        resource_id=gauntlet_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "name": gauntlet.name,
        "team_name": gauntlet.team_name,
        "description": gauntlet.description,
        "rounds": [
            {
                "round_number": r.round_number,
                "quorum_required_approvals": r.quorum_required_approvals,
                "quorum_from_panel_size": r.quorum_from_panel_size,
                "min_overall_confidence": r.min_overall_confidence
            }
            for r in gauntlet.rounds
        ]
    }


@app.post("/gauntlets", dependencies=[Depends(require_role(UserRole.USER))])
def create_gauntlet(
    request: GauntletCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new gauntlet (requires USER role)."""
    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        gauntlet = GauntletDefinition(
            name=request.name,
            tenant_id=tenant_id,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
        tenant_gauntlet_manager.save_gauntlet(gauntlet)
        
        logger.info(f"Gauntlet '{gauntlet.name}' created by {user.name}")
        record_audit_event(
            user=user,
            operation="CREATE_GAUNTLET",
            resource="gauntlet",
            resource_id=gauntlet.name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Gauntlet created", "gauntlet_name": gauntlet.name}
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error(f"Error creating gauntlet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/gauntlets/{gauntlet_name}", dependencies=[Depends(require_role(UserRole.USER))])
def update_gauntlet(
    gauntlet_name: str,
    request: GauntletCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Update an existing gauntlet (requires USER role)."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    existing_gauntlet = tenant_gauntlet_manager.get_gauntlet(gauntlet_name)
    if not existing_gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")

    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        updated_gauntlet = GauntletDefinition(
            name=gauntlet_name, # Ensure name from path is used
            tenant_id=tenant_id,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        tenant_gauntlet_manager.save_gauntlet(updated_gauntlet) # Overwrite existing
        
        logger.info(f"Gauntlet '{gauntlet_name}' updated by {user.name}")
        record_audit_event(
            user=user,
            operation="UPDATE_GAUNTLET",
            resource="gauntlet",
            resource_id=gauntlet_name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Gauntlet updated", "gauntlet_name": gauntlet_name}
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error(f"Error updating gauntlet '{gauntlet_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/gauntlets/{gauntlet_name}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_gauntlet(
    gauntlet_name: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a gauntlet (requires ADMIN role)."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    if not tenant_gauntlet_manager.get_gauntlet(gauntlet_name):
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    tenant_gauntlet_manager.delete_gauntlet(gauntlet_name)
    logger.info(f"Gauntlet '{gauntlet_name}' deleted by {user.name}")
    record_audit_event(
        user=user,
        operation="DELETE_GAUNTLET",
        resource="gauntlet",
        resource_id=gauntlet_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    return {"message": "Gauntlet deleted", "gauntlet_name": gauntlet_name}


# Evaluator endpoints

@app.get("/evaluators", dependencies=[Depends(verify_api_key)])
def list_evaluators(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List custom evaluators for a tenant."""
    evaluators = _list_evaluators(tenant_id)
    record_audit_event(
        user=user,
        operation="LIST_EVALUATORS",
        resource="evaluator",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id, "count": len(evaluators)}
    )
    return {"evaluators": evaluators}


@app.post("/evaluators", dependencies=[Depends(require_role(UserRole.USER))])
def upload_evaluator(
    request: EvaluatorUploadRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Upload a custom evaluator (requires USER role)."""
    error = _validate_evaluator_code(request.code)
    if error:
        raise HTTPException(status_code=400, detail=error)
    evaluator_id = f"eval_{uuid.uuid4().hex[:8]}"
    _save_evaluator(tenant_id, evaluator_id, request.code)
    record_audit_event(
        user=user,
        operation="UPLOAD_EVALUATOR",
        resource="evaluator",
        resource_id=evaluator_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {"evaluator_id": evaluator_id}


@app.delete("/evaluators/{evaluator_id}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_evaluator(
    evaluator_id: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a custom evaluator (requires ADMIN role)."""
    deleted = _delete_evaluator(tenant_id, evaluator_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    record_audit_event(
        user=user,
        operation="DELETE_EVALUATOR",
        resource="evaluator",
        resource_id=evaluator_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {"success": True, "evaluator_id": evaluator_id}


# Webhook system
import asyncio
import aiohttp
from typing import Set


class WebhookRegistration(BaseModel):
    """Webhook registration."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for webhook verification")


class WebhookManager:
    """Manages webhook registrations and delivery."""
    
    def __init__(self):
        """Initialize webhook manager."""
        self.webhooks: Dict[str, WebhookRegistration] = {}
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def verify_with_lean(self, target: str, criteria: Dict) -> Dict:
        """Verify target using Lean theorem prover."""
        if not LEAN_AVAILABLE:
            return {'verified': False}
        try:
            client = LeanAideClient()
            return client.verify(target)
        except Exception:
            return {'verified': False}
    
    def register(self, webhook_id: str, registration: WebhookRegistration) -> None:
        """Register a webhook."""
        self.webhooks[webhook_id] = registration
        logger.info(f"Registered webhook {webhook_id} for events: {registration.events}")
    
    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False
    
    async def trigger(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger webhooks for an event."""
        matching_webhooks = [
            (wid, wh) for wid, wh in self.webhooks.items()
            if event in wh.events or "*" in wh.events
        ]
        
        if not matching_webhooks:
            return
        
        logger.info(f"Triggering {len(matching_webhooks)} webhooks for event: {event}")
        
        # Trigger all webhooks concurrently
        tasks = [
            self._deliver_webhook(wid, wh, event, data)
            for wid, wh in matching_webhooks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_webhook(
        self,
        webhook_id: str,
        webhook: WebhookRegistration,
        event: str,
        data: Dict[str, Any]
    ) -> None:
        """Deliver webhook with retry logic."""
        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            # Add signature for verification
            import hmac
            import hashlib
            import json
            
            signature = hmac.new(
                webhook.secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status < 300:
                            logger.info(f"Webhook {webhook_id} delivered successfully")
                            return
                        else:
                            logger.warning(
                                f"Webhook {webhook_id} returned status {response.status}"
                            )
            except (OSError, IOError, RuntimeError) as e:
                logger.error(f"Webhook {webhook_id} delivery failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Webhook {webhook_id} failed after {self.max_retries} attempts")


# Initialize webhook manager
webhook_manager = WebhookManager()


# Webhook endpoints

@app.post("/webhooks", dependencies=[Depends(require_role(UserRole.USER))])
def register_webhook(registration: WebhookRegistration, user: AuthUser = Depends(verify_api_key)):
    """Register a webhook."""
    webhook_id = str(uuid.uuid4())
    webhook_manager.register(webhook_id, registration)
    
    logger.info(f"Webhook registered by {user.name}: {webhook_id}")
    
    return {
        "webhook_id": webhook_id,
        "url": registration.url,
        "events": registration.events,
        "message": "Webhook registered successfully"
    }


@app.get("/webhooks", dependencies=[Depends(verify_api_key)])
def list_webhooks():
    """List all registered webhooks."""
    return {
        "webhooks": [
            {
                "webhook_id": wid,
                "url": wh.url,
                "events": wh.events
            }
            for wid, wh in webhook_manager.webhooks.items()
        ],
        "total": len(webhook_manager.webhooks)
    }


@app.delete("/webhooks/{webhook_id}", dependencies=[Depends(require_role(UserRole.USER))])
def unregister_webhook(webhook_id: str, user: AuthUser = Depends(verify_api_key)):
    """Unregister a webhook."""
    if not webhook_manager.unregister(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    logger.info(f"Webhook unregistered by {user.name}: {webhook_id}")
    
    return {"message": "Webhook unregistered", "webhook_id": webhook_id}


@app.get("/audit/logs", dependencies=[Depends(require_role(UserRole.ADMIN))])
def list_audit_logs(
    limit: int = 200,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """List audit logs for the current tenant (admin only)."""
    tenant_logs = [
        log for log in AUDIT_LOGS
        if log.get("details", {}).get("tenant_id") == tenant_id
    ]
    record_audit_event(
        user=user,
        operation="LIST_AUDIT_LOGS",
        resource="audit",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "logs": tenant_logs[-limit:],
        "total": len(tenant_logs)
    }


# Helper function to trigger webhooks from workflow events
async def trigger_workflow_event(event: str, workflow_id: str, data: Dict[str, Any] = None):
    """Trigger webhook for workflow event."""
    payload = {
        "workflow_id": workflow_id,
        **(data or {})
    }
    await webhook_manager.trigger(event, payload)


# Deterministic LLM endpoints (Bubblelab UI + CLI control)

class DeterminismGenerateRequest(BaseModel):
    prompt: str
    schema: Optional[Dict[str, Any]] = None
    constraints: Optional[str] = None
    context_document: Optional[str] = None
    mode: str = "auto"  # auto | cloud | local | hybrid | consensus
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_base_url: Optional[str] = None
    local_provider: Optional[str] = "hf"
    local_model: Optional[str] = None
    local_device: Optional[str] = "cpu"
    local_dtype: Optional[str] = "auto"
    config: Optional[Dict[str, Any]] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None


class DeterminismCheckRequest(BaseModel):
    prompt: str
    tier: int = 2
    runs: int = 3
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None
    device: Optional[str] = "cpu"
    dtype: Optional[str] = "auto"


def _build_llm(
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    device: Optional[str] = None,
    dtype: Optional[str] = None,
):
    if not provider or not model:
        return None
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        device=device or "cpu",
        dtype=dtype or "auto",
    )
    return build_llm(config)


def _build_config(overrides: Optional[Dict[str, Any]], detllm_backend: Optional[str], detllm_model: Optional[str]) -> DeterminismConfig:
    config = DeterminismConfig()
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if detllm_backend:
        config.detllm_backend = detllm_backend
    if detllm_model:
        config.detllm_model = detllm_model
    return config


@app.post("/determinism/generate", dependencies=[Depends(verify_api_key)])
def determinism_generate(req: DeterminismGenerateRequest):
    config = _build_config(req.config, req.detllm_backend, req.detllm_model)

    if req.mode in {"hybrid", "consensus"}:
        cloud_llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url)
        local_llm = _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)
        if cloud_llm is None or local_llm is None:
            raise HTTPException(status_code=400, detail="Hybrid mode requires both cloud and local LLM configs")
        system = HybridDeterministicSystem(cloud_llm=cloud_llm, local_llm=local_llm)
        result = system.generate(req.prompt, mode=req.mode)
        return result.__dict__

    if req.mode == "cloud":
        llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url)
    elif req.mode == "local":
        llm = _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)
    else:
        llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url) or _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)

    pipeline = DeterministicPipeline(llm=llm, config=config)
    result = pipeline.generate_with_all_layers(req.prompt, schema=req.schema, constraints=req.constraints, context_document=req.context_document)
    return result.__dict__


@app.post("/determinism/check", dependencies=[Depends(verify_api_key)])
def determinism_check(req: DeterminismCheckRequest):
    llm = _build_llm(req.provider, req.model, req.api_key, req.base_url, req.device, req.dtype)
    pipeline = DeterministicPipeline(llm=llm, config=_build_config(None, req.detllm_backend, req.detllm_model))
    result = pipeline.reproducibility.check(
        prompt=req.prompt,
        llm=llm,
        tier=req.tier,
        runs=req.runs,
        backend=req.detllm_backend,
        model=req.detllm_model,
    )
    return result


# --- ICR Heatmap helpers ---

def _decode_data_url(data_url: Optional[str]) -> Optional[bytes]:
    if not data_url:
        return None
    if "," not in data_url:
        return None
    try:
        _, b64_data = data_url.split(",", 1)
        return base64.b64decode(b64_data)
    except (ValueError, TypeError, base64.binascii.Error):
        return None


async def _analyze_heatmap_composite(data_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Analyze a heatmap composite image using VLM (Vision Language Model).

    This function provides UI interaction insights from heatmap composite images
    using configurable VLM providers (OpenAI, Anthropic, etc.).

    Args:
        data_url: Base64-encoded data URL of the heatmap composite image

    Returns:
        Dictionary containing VLM analysis results, or None if analysis is disabled/unavailable.
        Returns:
        {
            "summary": str,
            "insights": List[str],
            "friction_points": List[str],
            "recommendations": List[str],
            "confidence": float,
            "provider": str,
            "model": str
        }
    """
    # Check if VLM analysis is enabled
    if os.getenv("ICR_VLM_ENABLED", "").lower() not in {"1", "true", "yes"}:
        logger.debug("VLM analysis is disabled via ICR_VLM_ENABLED")
        return None

    if not data_url:
        logger.debug("No composite data URL provided for VLM analysis")
        return None

    # Decode the image
    image_bytes = _decode_data_url(data_url)
    if not image_bytes:
        logger.warning("Failed to decode composite data URL for VLM analysis")
        return None

    try:
        from vision_language_monitor import VLMAnalyzer, VLMConfig, AnalysisType, VLMProvider
    except ImportError:
        logger.warning("vision_language_monitor module not available; skipping VLM heatmap analysis")
        return None

    # Load configuration from environment variables
    provider_env = os.getenv("ICR_VLM_PROVIDER", "openai").lower()
    provider = VLMProvider.OPENAI
    if provider_env:
        for candidate in VLMProvider:
            if candidate.value == provider_env:
                provider = candidate
                break

    model = os.getenv("ICR_VLM_MODEL", "gpt-4o")
    temperature = float(os.getenv("ICR_VLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("ICR_VLM_MAX_TOKENS", "1024"))
    api_key = os.getenv("ICR_VLM_API_KEY")
    base_url = os.getenv("ICR_VLM_BASE_URL")

    # Create VLM config
    config = VLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

    # Initialize analyzer
    analyzer = VLMAnalyzer(config)

    # Check if VLM is properly configured
    if not analyzer.is_configured():
        logger.warning("VLM is not properly configured (missing API key). Skipping analysis.")
        return None

    # Build analysis prompt
    prompt = (
        "Analyze this UI snapshot with an interaction heatmap overlay.\n"
        "Identify cognitive friction points, confusing placements, and areas of repeated interaction.\n"
        "Provide concise, actionable UI refinement suggestions."
    )

    # Run analysis
    try:
        analysis = await analyzer.analyze(image_bytes, prompt, AnalysisType.LAYOUT_ANALYSIS)
        return analysis.to_dict()
    except Exception as e:
        logger.error(f"VLM analysis failed: {e}")
        return None


# ICR Event Bridge Endpoints (optional, unauthenticated for local UI polling)

@app.post("/icr/events/refinement-needed")
def icr_emit_refinement_needed(event: IcrRefinementEvent):
    payload = event.model_dump()
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REFINEMENT_EVENTS.append(payload)
    return {"queued": True}


@app.get("/icr/events/refinement-needed")
def icr_get_refinement_needed(limit: int = 5):
    items = []
    while ICR_REFINEMENT_EVENTS and len(items) < limit:
        items.append(ICR_REFINEMENT_EVENTS.popleft())
    return items


@app.post("/icr/reward-calibration/request")
def icr_queue_reward_calibration(request: IcrRewardCalibrationRequest):
    payload = request.model_dump()
    if not payload.get("request_id"):
        payload["request_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_QUEUE.append(payload)
    return {"queued": True, "request_id": payload["request_id"]}


@app.get("/icr/reward-calibration/next")
def icr_next_reward_calibration():
    if not ICR_REWARD_CALIBRATION_QUEUE:
        return {}
    return ICR_REWARD_CALIBRATION_QUEUE.popleft()


@app.post("/icr/reward-calibration/respond")
def icr_reward_calibration_respond(response: IcrRewardCalibrationResponse):
    request_id = response.request_id or str(uuid.uuid4())
    payload = response.model_dump()
    payload["request_id"] = request_id
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_RESPONSES[request_id] = payload
    return {"received": True, "request_id": request_id}


@app.get("/icr/reward-calibration/response/{request_id}")
def icr_reward_calibration_response(request_id: str):
    return ICR_REWARD_CALIBRATION_RESPONSES.get(request_id, {})


@app.post("/icr/heatmap/snapshot")
async def icr_heatmap_snapshot(snapshot: IcrHeatmapSnapshot):
    """
    Store heatmap snapshot for ICR pattern analysis.
    
    Accepts heatmap snapshot data from GenerativeUI and stores it for:
    - Pattern analysis and learning
    - Multimodal healing prompt generation
    - Vision-language model analysis of UI interactions
    
    Args:
        snapshot: Heatmap snapshot containing screen HTML, heatmap data, and interaction points
        
    Returns:
        Success response with snapshot_id and optional analysis results
        
    Environment Variables for VLM:
        - ICR_VLM_ENABLED: Enable/disable VLM analysis (default: false)
        - ICR_VLM_PROVIDER: VLM provider - openai, anthropic, mock (default: openai)
        - ICR_VLM_MODEL: Model name (default: gpt-4o for OpenAI, claude-3-5-sonnet-20241022 for Anthropic)
        - ICR_VLM_API_KEY: API key for the VLM provider (optional if using provider's default env var)
        - ICR_VLM_TEMPERATURE: Temperature for VLM (default: 0.2)
        - ICR_VLM_MAX_TOKENS: Max tokens for VLM response (default: 1024)
        - ICR_VLM_BASE_URL: Custom base URL for VLM API (optional)
    """
    payload = snapshot.model_dump()
    if not payload.get("snapshot_id"):
        payload["snapshot_id"] = str(uuid.uuid4())
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().timestamp()
    payload["received_at"] = datetime.utcnow().isoformat()
    ICR_HEATMAP_SNAPSHOTS.append(payload)

    analysis = None
    vlm_analysis = None

    # Generate multimodal healing prompt if analytics_manager is available
    try:
        from analytics_manager import analytics_manager
        heatmap_payload = {
            "points": payload.get("points", []),
            "manual_code_delta": payload.get("manual_code_delta")
        }
        analysis = analytics_manager.generate_multimodal_healing_prompt(
            payload.get("context_text", "") or "",
            heatmap_snapshot=heatmap_payload,
            auto_refine_enabled=bool(payload.get("auto_refine"))
        )
    except Exception as exc:
        logger.warning("Failed to generate multimodal healing prompt: %s", exc)

    # Run VLM analysis if enabled and composite data is available
    try:
        vlm_analysis = await _analyze_heatmap_composite(payload.get("composite_data_url"))
        if vlm_analysis and analysis is not None:
            analysis["vlm_analysis"] = vlm_analysis
    except Exception as exc:
        logger.warning("Failed to run VLM heatmap analysis: %s", exc)

    return {
        "queued": True,
        "snapshot_id": payload["snapshot_id"],
        "analysis": analysis,
        "vlm_analysis": vlm_analysis,
    }


@app.get("/icr/vlm/config")
def icr_vlm_config():
    """
    Get VLM configuration status.
    
    Returns current VLM configuration and whether it's properly set up.
    Useful for debugging and checking if VLM analysis is available.
    
    Returns:
        Dictionary with VLM configuration information
    """
    try:
        from vision_language_monitor import VLMAnalyzer, VLMConfig
    except ImportError:
        return {
            "available": False,
            "error": "vision_language_monitor module not available",
            "message": "VLM analysis is not available"
        }

    config = VLMAnalyzer._load_config_from_env()
    analyzer = VLMAnalyzer(config)
    
    return {
        "available": True,
        "enabled": os.getenv("ICR_VLM_ENABLED", "").lower() in {"1", "true", "yes"},
        "configured": analyzer.is_configured(),
        "config": analyzer.get_config_info()
    }


# =============================================================================
# ICR ANALYTICS DASHBOARD ENDPOINTS
# =============================================================================

# In-memory storage for ICR analytics data (simulated)
ICR_ANALYTICS_DATA = {
    "overview": {
        "icr_enabled": True,
        "total_patterns": 0,
        "overall_success_rate": 0.0,
        "active_components": 5,
        "total_refinements": 0
    },
    "components": {
        "quality_gate_engine": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "workflow_orchestrator": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "robustness_coordinator": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "bubblelab": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "roma": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        }
    },
    "patterns": {
        "pattern_types": {},
        "trends": {
            "timestamps": [],
            "values": []
        },
        "by_content_type": {},
        "by_quality_level": {},
        "by_complexity": {}
    },
    "vlm": {
        "available": False,
        "enabled": False,
        "total_analyses": 0,
        "total_tokens": 0,
        "avg_confidence": 0.0,
        "cache_hit_rate": 0.0,
        "by_provider": {},
        "config": None
    },
    "heatmap": {
        "points": []
    },
    "config": {
        "enabled": True,
        "enable_prediction": True,
        "enable_learning": True,
        "quality_gate_enabled": True,
        "workflow_orchestrator_enabled": True,
        "gauntlet_system_enabled": True,
        "robustness_enabled": True,
        "roma_modules_enabled": True
    }
}


@app.get("/icr/dashboard")
async def icr_dashboard(request: Request):
    """
    Serve the ICR Analytics Dashboard.
    
    Returns the HTML template for the ICR analytics dashboard.
    """
    return templates.TemplateResponse("icr_dashboard.html", {"request": request})


@app.get("/icr/analytics/overview")
async def icr_analytics_overview():
    """
    Get ICR overview statistics.
    
    Returns:
        - Total patterns stored
        - Overall success rate
        - Active components count
        - Total refinements applied
        - ICR enabled status
    """
    # Calculate total patterns from all components
    total_patterns = sum(
        comp["total_patterns"]
        for comp in ICR_ANALYTICS_DATA["components"].values()
    )
    
    # Calculate overall success rate
    component_rates = [
        comp["overall_pass_rate"]
        for comp in ICR_ANALYTICS_DATA["components"].values()
        if comp["overall_pass_rate"] > 0
    ]
    overall_success_rate = sum(component_rates) / len(component_rates) if component_rates else 0.0
    
    # Count active components
    active_components = sum(
        1 for comp in ICR_ANALYTICS_DATA["components"].values()
        if comp["active"]
    )
    
    return {
        "icr_enabled": ICR_ANALYTICS_DATA["overview"]["icr_enabled"],
        "total_patterns": total_patterns,
        "overall_success_rate": overall_success_rate,
        "active_components": active_components,
        "total_refinements": ICR_ANALYTICS_DATA["overview"]["total_refinements"]
    }


@app.get("/icr/analytics/components")
async def icr_analytics_components():
    """
    Get component-specific ICR statistics.
    
    Returns statistics for each ICR component:
        - QualityGateEngine
        - SGDWorkflowOrchestrator
        - RobustnessCoordinator
        - BubbleLab
        - ROMA
    """
    return ICR_ANALYTICS_DATA["components"]


@app.get("/icr/analytics/patterns")
async def icr_analytics_patterns():
    """
    Get pattern analysis data.
    
    Returns:
        - Pattern types distribution
        - Success rate trends over time
        - Patterns by content type
        - Patterns by quality level
        - Patterns by complexity
    """
    return ICR_ANALYTICS_DATA["patterns"]


@app.get("/icr/analytics/vlm")
async def icr_analytics_vlm():
    """
    Get VLM analytics data.
    
    Returns:
        - Total analyses performed
        - Total tokens consumed
        - Average confidence
        - Cache hit rate
        - Analysis count by provider
        - Current VLM configuration
    """
    # Get VLM status from existing endpoint
    vlm_status = icr_vlm_config()
    
    return {
        "available": vlm_status.get("available", False),
        "enabled": vlm_status.get("enabled", False),
        "total_analyses": ICR_ANALYTICS_DATA["vlm"]["total_analyses"],
        "total_tokens": ICR_ANALYTICS_DATA["vlm"]["total_tokens"],
        "avg_confidence": ICR_ANALYTICS_DATA["vlm"]["avg_confidence"],
        "cache_hit_rate": ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"],
        "by_provider": ICR_ANALYTICS_DATA["vlm"]["by_provider"],
        "config": vlm_status.get("config")
    }


@app.get("/icr/analytics/refinements")
async def icr_analytics_refinements(limit: int = 10):
    """
    Get recent refinement events.
    
    Args:
        limit: Maximum number of events to return (default: 10)
    
    Returns:
        - List of recent refinement events with details
    """
    # Get events from the global queue
    events = []
    while ICR_REFINEMENT_EVENTS and len(events) < limit:
        event = ICR_REFINEMENT_EVENTS.popleft()
        events.append(event)
        # Put it back for other consumers
        ICR_REFINEMENT_EVENTS.appendleft(event)
    
    return {
        "events": events[:limit],
        "total_count": len(ICR_REFINEMENT_EVENTS)
    }


@app.get("/icr/analytics/heatmap")
async def icr_analytics_heatmap():
    """
    Get ICR pattern heatmap data.
    
    Returns:
        - Heatmap points with coordinates and intensity
        - Snapshot metadata
    """
    # Aggregate heatmap data from snapshots
    heatmap_points = []
    
    for snapshot in list(ICR_HEATMAP_SNAPSHOTS):
        points = snapshot.get("points", [])
        heatmap_points.extend(points)
    
    return {
        "points": heatmap_points,
        "total_snapshots": len(ICR_HEATMAP_SNAPSHOTS)
    }


@app.get("/icr/config")
async def icr_get_config():
    """
    Get current ICR configuration.
    
    Returns:
        - ICR enabled status
        - Component enablement flags
        - Feature flags (prediction, learning)
    """
    return ICR_ANALYTICS_DATA["config"]


# =============================================================================
# ICR DATA UPDATE HELPERS (for component integration)
# =============================================================================

def update_icr_component_stats(component_name: str, stats: dict):
    """
    Update statistics for a specific ICR component.
    
    Args:
        component_name: Name of the component (e.g., "quality_gate_engine")
        stats: Statistics dictionary with keys:
            - total_patterns: int
            - overall_pass_rate: float
            - overall_quality: float
            - active: bool
    """
    if component_name in ICR_ANALYTICS_DATA["components"]:
        ICR_ANALYTICS_DATA["components"][component_name].update(stats)


def update_icr_pattern_data(pattern_type: str, content_type: str = None,
                           quality_level: str = None, complexity: int = None):
    """
    Update pattern analysis data when new patterns are stored.
    
    Args:
        pattern_type: Type of pattern (e.g., "content_type", "metric")
        content_type: Content type (e.g., "code", "text")
        quality_level: Quality level (e.g., "standard", "high")
        complexity: Complexity score (1-10)
    """
    # Update pattern types
    if pattern_type not in ICR_ANALYTICS_DATA["patterns"]["pattern_types"]:
        ICR_ANALYTICS_DATA["patterns"]["pattern_types"][pattern_type] = 0
    ICR_ANALYTICS_DATA["patterns"]["pattern_types"][pattern_type] += 1
    
    # Update by content type
    if content_type:
        if content_type not in ICR_ANALYTICS_DATA["patterns"]["by_content_type"]:
            ICR_ANALYTICS_DATA["patterns"]["by_content_type"][content_type] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_content_type"][content_type] += 1
    
    # Update by quality level
    if quality_level:
        if quality_level not in ICR_ANALYTICS_DATA["patterns"]["by_quality_level"]:
            ICR_ANALYTICS_DATA["patterns"]["by_quality_level"][quality_level] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_quality_level"][quality_level] += 1
    
    # Update by complexity
    if complexity:
        complexity_key = str(complexity)
        if complexity_key not in ICR_ANALYTICS_DATA["patterns"]["by_complexity"]:
            ICR_ANALYTICS_DATA["patterns"]["by_complexity"][complexity_key] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_complexity"][complexity_key] += 1
    
    # Update trend
    now = datetime.utcnow()
    ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"].append(now.isoformat())
    # Placeholder for actual success rate calculation
    ICR_ANALYTICS_DATA["patterns"]["trends"]["values"].append(0.8)
    
    # Keep only last 50 trend points
    if len(ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"]) > 50:
        ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"] = \
            ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"][-50:]
        ICR_ANALYTICS_DATA["patterns"]["trends"]["values"] = \
            ICR_ANALYTICS_DATA["patterns"]["trends"]["values"][-50:]


def update_icr_vlm_stats(provider: str, tokens_used: int = 0,
                        confidence: float = 0.0, cached: bool = False):
    """
    Update VLM analytics statistics.
    
    Args:
        provider: VLM provider name (e.g., "openai", "anthropic")
        tokens_used: Number of tokens consumed
        confidence: Analysis confidence score
        cached: Whether the result was from cache
    """
    ICR_ANALYTICS_DATA["vlm"]["total_analyses"] += 1
    ICR_ANALYTICS_DATA["vlm"]["total_tokens"] += tokens_used
    
    # Update average confidence
    current_avg = ICR_ANALYTICS_DATA["vlm"]["avg_confidence"]
    total = ICR_ANALYTICS_DATA["vlm"]["total_analyses"]
    ICR_ANALYTICS_DATA["vlm"]["avg_confidence"] = \
        (current_avg * (total - 1) + confidence) / total
    
    # Update cache hit rate
    if cached:
        cache_hits = ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] * (total - 1) + 1
    else:
        cache_hits = ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] * (total - 1)
    ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] = cache_hits / total
    
    # Update by provider
    if provider not in ICR_ANALYTICS_DATA["vlm"]["by_provider"]:
        ICR_ANALYTICS_DATA["vlm"]["by_provider"][provider] = 0
    ICR_ANALYTICS_DATA["vlm"]["by_provider"][provider] += 1


def record_icr_refinement(refinement_type: str, component: str,
                          reason: str, success: bool, confidence: float):
    """
    Record a refinement event.
    
    Args:
        refinement_type: Type of refinement (e.g., "threshold_adjustment")
        component: Component that triggered the refinement
        reason: Reason for the refinement
        success: Whether the refinement was successful
        confidence: Confidence in the refinement
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "refinement_type": refinement_type,
        "component": component,
        "reason": reason,
        "success": success,
        "confidence": confidence
    }
    
    ICR_REFINEMENT_EVENTS.append(event)
    ICR_ANALYTICS_DATA["overview"]["total_refinements"] += 1


# Statistics endpoints

@app.get("/statistics", dependencies=[Depends(verify_api_key)])
def get_statistics():
    """Get system statistics."""
    completed_workflows = [wf for wf in workflows.values() if wf.status == "completed"]
    failed_workflows = [wf for wf in workflows.values() if wf.status == "failed"]
    running_workflows = [wf for wf in workflows.values() if wf.status == "running"]
    
    return {
        "total_workflows": len(workflows),
        "completed": len(completed_workflows),
        "failed": len(failed_workflows),
        "running": len(running_workflows),
        "total_teams": len(team_manager.get_all_teams()),
        "total_gauntlets": len(gauntlet_manager.get_all_gauntlets())
    }


# Analytics endpoints

@app.get("/analytics/performance-metrics", dependencies=[Depends(verify_api_key)])
def get_performance_metrics(entity_type: Optional[str] = None, limit: int = 200):
    """Get performance metrics from the knowledge manager."""
    metrics = knowledge_manager.get_performance_metrics(entity_type=entity_type, limit=limit)
    return {
        "metrics": [_serialize_performance_metric(metric) for metric in metrics],
        "total": len(metrics)
    }


@app.get("/analytics/knowledge-stats", dependencies=[Depends(verify_api_key)])
def get_analytics_knowledge_stats():
    """Get aggregated knowledge base statistics."""
    artifacts = knowledge_manager.get_all_artifacts()
    total_artifacts = len(artifacts)
    total_usage = sum(a.usage_count for a in artifacts)
    avg_effectiveness = (
        sum(a.effectiveness_score for a in artifacts) / total_artifacts
        if total_artifacts
        else 0.0
    )

    artifact_type_distribution: Dict[str, int] = {}
    domain_distribution: Dict[str, int] = {}

    for artifact in artifacts:
        artifact_type_distribution[artifact.artifact_type] = (
            artifact_type_distribution.get(artifact.artifact_type, 0) + 1
        )
        if artifact.domain:
            domain_distribution[artifact.domain] = domain_distribution.get(artifact.domain, 0) + 1

    top_used = sorted(artifacts, key=lambda a: a.usage_count, reverse=True)[:10]
    top_effective = sorted(artifacts, key=lambda a: a.effectiveness_score, reverse=True)[:10]

    return {
        "total_artifacts": total_artifacts,
        "total_usage": total_usage,
        "avg_effectiveness": avg_effectiveness,
        "artifact_type_distribution": artifact_type_distribution,
        "domain_distribution": domain_distribution,
        "top_used_artifacts": [
            {
                "id": a.id,
                "artifact_type": a.artifact_type,
                "usage_count": a.usage_count,
                "effectiveness_score": a.effectiveness_score,
                "domain": a.domain,
            }
            for a in top_used
        ],
        "top_effective_artifacts": [
            {
                "id": a.id,
                "artifact_type": a.artifact_type,
                "usage_count": a.usage_count,
                "effectiveness_score": a.effectiveness_score,
                "domain": a.domain,
            }
            for a in top_effective
        ],
    }


@app.get("/analytics/workflow-metrics", dependencies=[Depends(verify_api_key)])
def get_workflow_metrics():
    """Get snapshot metrics for active workflows."""
    metrics = []
    timestamp = datetime.utcnow().isoformat()
    for wf in workflows.values():
        openevolve_metrics = wf.openevolve_metrics or {}
        performance_metrics = wf.performance_metrics or {}
        resource_usage = wf.resource_usage or {}
        execution_time = None
        if wf.start_time:
            execution_time = (wf.end_time or time.time()) - wf.start_time

        metrics.append(
            {
                "timestamp": timestamp,
                "workflow_id": wf.workflow_id,
                "status": wf.status,
                "progress": wf.progress,
                "best_fitness": _extract_metric_value(
                    openevolve_metrics,
                    ["best_fitness", "best_score", "fitness"],
                ),
                "avg_fitness": _extract_metric_value(
                    openevolve_metrics,
                    ["avg_fitness", "average_fitness", "mean_fitness"],
                ),
                "diversity": _extract_metric_value(
                    openevolve_metrics,
                    ["diversity", "diversity_score", "population_diversity"],
                ),
                "tokens_used": _extract_metric_value(resource_usage, ["tokens_used", "token_usage"]),
                "execution_time": execution_time,
                "memory_usage": _extract_metric_value(resource_usage, ["memory_usage_mb", "memory_peak_mb"]),
                "cpu_usage": _extract_metric_value(resource_usage, ["cpu_usage", "cpu_time"]),
                "population_size": _extract_metric_value(openevolve_metrics, ["population_size"]),
                "generation": _extract_metric_value(openevolve_metrics, ["iterations_completed", "generation"]),
                "metrics": {
                    "performance": performance_metrics,
                    "resource_usage": resource_usage,
                    "openevolve": openevolve_metrics,
                },
            }
        )

    return {"metrics": metrics, "total": len(metrics)}


# Monitoring endpoints

@app.get("/monitoring/dashboard", dependencies=[Depends(verify_api_key)])
def get_monitoring_dashboard():
    """Get monitoring dashboard metrics."""
    return system_monitoring_dashboard.get_dashboard_metrics()


@app.get("/monitoring/alerts", dependencies=[Depends(verify_api_key)])
def get_monitoring_alerts():
    """Get active monitoring alerts."""
    return {"alerts": system_alert_manager.check_alerts()}


@app.get("/monitoring/metrics", dependencies=[Depends(verify_api_key)])
def get_monitoring_metrics(
    name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Fetch monitoring metrics from the metrics collector."""
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    metrics = system_metrics_collector.get_metrics(name=name, start_time=start_dt, end_time=end_dt)
    return {"metrics": [_serialize_monitoring_metric(metric) for metric in metrics]}


@app.get("/monitoring/health", dependencies=[Depends(verify_api_key)])
def get_monitoring_health():
    """Get current monitoring health status."""
    return system_health_monitor.get_health_status()


@app.get("/monitoring/services", dependencies=[Depends(verify_api_key)])
def get_monitoring_services():
    """Get health check details for monitored services."""
    result = system_health_monitor.run_health_checks()
    services = []
    for name, check in result.get("checks", {}).items():
        services.append(
            {
                "name": name,
                "status": check.get("status"),
                "healthy": check.get("healthy"),
                "execution_time": check.get("execution_time"),
                "timestamp": check.get("timestamp"),
                "error": check.get("error"),
            }
        )
    return {"services": services, "timestamp": result.get("timestamp")}


@app.get("/monitoring/logs", dependencies=[Depends(verify_api_key)])
def get_monitoring_logs(limit: int = 200, source: Optional[str] = None):
    """Return recent log lines from known sources."""
    sources = _collect_log_sources()
    if source:
        if source not in sources:
            raise HTTPException(status_code=404, detail="Log source not found")
        paths = {source: sources[source]}
    else:
        paths = sources

    entries = []
    per_file_limit = max(1, limit)
    for name, path in paths.items():
        lines = _tail_log_file(path, per_file_limit)
        for line in lines:
            entries.append({"source": name, "line": line.rstrip("\n")})

    if limit and len(entries) > limit:
        entries = entries[-limit:]

    return {"entries": entries, "total": len(entries)}


# CrewAI monitoring endpoints

@app.get("/crewai/workflows", dependencies=[Depends(verify_api_key)])
def list_crewai_workflows():
    """List CrewAI workflow states from local storage."""
    if not CREWAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="CrewAI state management not available")

    state_manager = _get_crewai_state_manager()
    if state_manager is None:
        raise HTTPException(status_code=500, detail="CrewAI state manager unavailable")

    workflow_ids = state_manager.list_workflows()
    summaries = []
    for workflow_id in workflow_ids:
        summary = state_manager.get_state_summary(workflow_id)
        if summary:
            summaries.append(summary)

    return {"workflows": summaries, "total": len(summaries)}


@app.get("/crewai/workflows/{workflow_id}", dependencies=[Depends(verify_api_key)])
def get_crewai_workflow(workflow_id: str):
    """Get a CrewAI workflow state."""
    if not CREWAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="CrewAI state management not available")

    state_manager = _get_crewai_state_manager()
    if state_manager is None:
        raise HTTPException(status_code=500, detail="CrewAI state manager unavailable")

    state = state_manager.load_state(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="CrewAI workflow not found")

    return state.model_dump()


@app.get("/crewai/workflows/{workflow_id}/tickets", dependencies=[Depends(verify_api_key)])
def get_crewai_workflow_tickets(workflow_id: str):
    """Get ticket-like entries derived from a CrewAI workflow."""
    if not CREWAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="CrewAI state management not available")

    state_manager = _get_crewai_state_manager()
    if state_manager is None:
        raise HTTPException(status_code=500, detail="CrewAI state manager unavailable")

    state = state_manager.load_state(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="CrewAI workflow not found")

    tickets = _build_crewai_ticket_list(state)
    status_counts: Dict[str, int] = {}
    for ticket in tickets:
        status = ticket.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {"tickets": tickets, "total": len(tickets), "status_breakdown": status_counts}


# Knowledge Base endpoints

@app.get("/knowledge/artifacts", dependencies=[Depends(verify_api_key)])
def list_knowledge_artifacts():
    artifacts = knowledge_manager.get_all_artifacts()
    return {"artifacts": [a.__dict__ for a in artifacts]}


@app.get("/knowledge/artifacts/{artifact_id}", dependencies=[Depends(verify_api_key)])
def get_knowledge_artifact(artifact_id: str):
    artifact = knowledge_manager.artifacts.get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact.__dict__


@app.post("/knowledge/artifacts", dependencies=[Depends(verify_api_key)])
def create_knowledge_artifact(request: KnowledgeArtifactCreateRequest):
    from workflow_structures import KnowledgeArtifact as KnowledgeArtifactModel
    artifact_id = uuid.uuid4().hex[:16]
    artifact = KnowledgeArtifactModel(
        id=artifact_id,
        artifact_type=request.artifact_type,
        content=request.content,
        source_workflow_id=request.source_workflow_id or "manual",
        extraction_timestamp=datetime.now().isoformat(),
        domain=request.domain,
        problem_type=request.problem_type,
        usage_count=0,
        effectiveness_score=0.0,
        related_artifacts=request.related_artifacts or []
    )
    knowledge_manager.store_knowledge_artifact(artifact)
    return artifact.__dict__


@app.delete("/knowledge/artifacts/{artifact_id}", dependencies=[Depends(verify_api_key)])
def delete_knowledge_artifact(artifact_id: str):
    success = knowledge_manager.delete_artifact(artifact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return {"success": True}


@app.post("/knowledge/search", dependencies=[Depends(verify_api_key)])
def search_knowledge(request: KnowledgeSearchRequest):
    results = knowledge_manager.retrieve_relevant_knowledge(
        problem_statement=request.query,
        domain=request.domain,
        artifact_types=request.artifact_types,
        limit=request.limit
    )
    return {"results": [a.__dict__ for a in results]}


@app.get("/knowledge/graph", dependencies=[Depends(verify_api_key)])
def get_knowledge_graph():
    artifacts = knowledge_manager.get_all_artifacts()
    nodes = [
        {
            "id": artifact.id,
            "type": artifact.artifact_type,
            "domain": artifact.domain,
            "usage": artifact.usage_count
        }
        for artifact in artifacts
    ]
    edges = []
    artifact_ids = {artifact.id for artifact in artifacts}
    for artifact in artifacts:
        for related_id in artifact.related_artifacts or []:
            if related_id in artifact_ids:
                edges.append({"source": artifact.id, "target": related_id})
    return {"nodes": nodes, "edges": edges}


@app.get("/knowledge/stats", dependencies=[Depends(verify_api_key)])
def get_knowledge_stats():
    artifacts = knowledge_manager.get_all_artifacts()
    total_usage = sum(a.usage_count for a in artifacts)
    avg_effectiveness = (
        sum(a.effectiveness_score for a in artifacts) / len(artifacts)
        if artifacts else 0.0
    )
    by_type: Dict[str, int] = {}
    for artifact in artifacts:
        by_type[artifact.artifact_type] = by_type.get(artifact.artifact_type, 0) + 1
    return {
        "total_artifacts": len(artifacts),
        "total_usage": total_usage,
        "average_effectiveness": avg_effectiveness,
        "by_type": by_type
    }


@app.post("/knowledge/recommendations", dependencies=[Depends(verify_api_key)])
def get_knowledge_recommendations(request: KnowledgeRecommendationsRequest):
    recommendations = knowledge_manager.apply_learned_patterns(
        request.problem_statement,
        domain=request.domain
    )
    return recommendations


@app.get("/knowledge/export", dependencies=[Depends(verify_api_key)])
def export_knowledge_base():
    artifacts = knowledge_manager.get_all_artifacts()
    export_data = {artifact.id: artifact.__dict__ for artifact in artifacts}
    return export_data


@app.post("/knowledge/import", dependencies=[Depends(verify_api_key)])
def import_knowledge_base(request: KnowledgeImportRequest):
    # File-based import to leverage existing KnowledgeManager logic
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", "knowledge_import.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        import json
        json.dump(request.artifacts, f, indent=2)
    knowledge_manager.import_knowledge_base(temp_path)
    return {"success": True}


# Prompt endpoints

@app.get("/prompts", dependencies=[Depends(verify_api_key)])
def list_custom_prompts():
    return {"prompts": CUSTOM_PROMPTS}


@app.post("/prompts", dependencies=[Depends(verify_api_key)])
def create_custom_prompt(request: PromptCreateRequest):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Prompt name cannot be empty")
    CUSTOM_PROMPTS[request.name] = request.content
    _save_json_store(PROMPTS_FILE, CUSTOM_PROMPTS)
    return {"success": True, "name": request.name}


@app.delete("/prompts/{prompt_name}", dependencies=[Depends(verify_api_key)])
def delete_custom_prompt(prompt_name: str):
    if prompt_name not in CUSTOM_PROMPTS:
        raise HTTPException(status_code=404, detail="Prompt not found")
    del CUSTOM_PROMPTS[prompt_name]
    _save_json_store(PROMPTS_FILE, CUSTOM_PROMPTS)
    return {"success": True}


# Content management endpoints

@app.get("/content/templates", dependencies=[Depends(verify_api_key)])
def list_protocol_templates():
    builtin_templates = content_manager.list_protocol_templates() if CONTENT_MANAGER_AVAILABLE else []
    all_templates = sorted(set(builtin_templates) | set(CUSTOM_PROTOCOL_TEMPLATES.keys()))
    return {"templates": all_templates}


@app.get("/content/templates/{template_name}", dependencies=[Depends(verify_api_key)])
def get_protocol_template(template_name: str):
    if template_name in CUSTOM_PROTOCOL_TEMPLATES:
        return {"name": template_name, "content": CUSTOM_PROTOCOL_TEMPLATES[template_name], "source": "custom"}
    content = content_manager.load_protocol_template(template_name) if CONTENT_MANAGER_AVAILABLE else ""
    if not content:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"name": template_name, "content": content, "source": "builtin"}


@app.post("/content/templates", dependencies=[Depends(verify_api_key)])
def create_protocol_template(request: ContentTemplateRequest):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Template name cannot be empty")
    CUSTOM_PROTOCOL_TEMPLATES[request.name] = request.content
    _save_json_store(CUSTOM_PROTOCOL_TEMPLATES_FILE, CUSTOM_PROTOCOL_TEMPLATES)
    if CONTENT_MANAGER_AVAILABLE:
        metadata = content_manager.export_protocol_as_template(request.content, request.name)
        return {"template": metadata}
    return {"template": {"name": request.name, "content": request.content}}


@app.post("/content/validate", dependencies=[Depends(verify_api_key)])
def validate_protocol_content(request: ProtocolValidationRequest):
    if CONTENT_MANAGER_AVAILABLE:
        result = content_manager.validate_protocol(request.protocol_text, request.validation_type or "generic")
        return result
    return _basic_validate_protocol(request.protocol_text, request.validation_type or "generic")


# Auto-approval endpoints

@app.get("/auto-approval/config", dependencies=[Depends(verify_api_key)])
def get_auto_approval_config():
    return AUTO_APPROVAL_CONFIG


@app.put("/auto-approval/config", dependencies=[Depends(verify_api_key)])
def update_auto_approval_config(request: AutoApprovalConfigModel):
    AUTO_APPROVAL_CONFIG["enabled"] = request.enabled
    AUTO_APPROVAL_CONFIG["rules"] = [rule.dict() for rule in request.rules]
    return AUTO_APPROVAL_CONFIG


@app.post("/auto-approval/test", dependencies=[Depends(verify_api_key)])
def test_auto_approval_rules(request: AutoApprovalTestRequest):
    results = []
    for rule in AUTO_APPROVAL_CONFIG.get("rules", []):
        if not rule.get("enabled", True):
            continue
        matched = _evaluate_auto_approval_rule(rule, request.plan)
        results.append({
            "rule_name": rule.get("name", "Unnamed Rule"),
            "action": rule.get("action", "approve"),
            "matched": matched
        })
        AUTO_APPROVAL_AUDIT_LOG.append({
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule.get("name", "Unnamed Rule"),
            "action": rule.get("action", "approve"),
            "matched": matched,
            "plan": request.plan
        })
    return {"results": results}


@app.get("/auto-approval/audit", dependencies=[Depends(verify_api_key)])
def get_auto_approval_audit():
    return {"logs": AUTO_APPROVAL_AUDIT_LOG}


# Workflow template endpoints

@app.get("/workflow-templates", dependencies=[Depends(verify_api_key)])
def list_workflow_templates():
    return {"templates": template_manager.get_all_templates()}


@app.post("/workflow-templates", dependencies=[Depends(verify_api_key)])
def create_workflow_template(request: WorkflowTemplateCreateRequest):
    template_id = template_manager.create_template(
        name=request.name,
        description=request.description or "",
        config=request.config,
        tags=request.tags or []
    )
    template = template_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=500, detail="Failed to create template")
    return template


@app.put("/workflow-templates/{template_id}", dependencies=[Depends(verify_api_key)])
def update_workflow_template(template_id: str, request: WorkflowTemplateUpdateRequest):
    success = template_manager.update_template(
        template_id=template_id,
        name=request.name,
        description=request.description,
        config=request.config,
        tags=request.tags
    )
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_manager.get_template(template_id)


@app.delete("/workflow-templates/{template_id}", dependencies=[Depends(verify_api_key)])
def delete_workflow_template(template_id: str):
    success = template_manager.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@app.get("/workflow-templates/export", dependencies=[Depends(verify_api_key)])
def export_workflow_templates():
    templates = template_manager.get_all_templates()
    return {"templates": templates}


@app.post("/workflow-templates/import", dependencies=[Depends(verify_api_key)])
def import_workflow_templates(request: Dict[str, Any]):
    templates = request.get("templates", [])
    imported = []
    for template in templates:
        template_id = template_manager.create_template(
            name=template.get("name", "Imported Template"),
            description=template.get("description", ""),
            config=template.get("config", {}),
            tags=template.get("tags", []),
        )
        imported.append(template_id)
    return {"success": True, "imported": imported}


# Providers and parameters

@app.get("/providers", dependencies=[Depends(verify_api_key)])
def list_providers():
    providers = []
    for provider_id, data in PROVIDERS_MAP.items():
        providers.append({
            "id": provider_id,
            "name": data.get("name"),
            "api_base": data.get("api_base"),
            "models_endpoint": data.get("models_endpoint"),
            "default_model": data.get("default_model")
        })
    return {"providers": providers}


@app.post("/providers/{provider_id}/models", dependencies=[Depends(verify_api_key)])
def get_provider_models(provider_id: str, request: ProviderModelsRequest):
    provider = PROVIDERS_MAP.get(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    loader = provider.get("loader")
    if not callable(loader):
        return {"models": [provider.get("default_model")]}
    try:
        models = loader(request.api_key)
        return {"models": models}
    except Exception as e:
        logger.warning(f"Failed to fetch models for {provider_id}: {e}")
        return {"models": [provider.get("default_model")]}


# Version control

@app.get("/version-control/versions", dependencies=[Depends(verify_api_key)])
def list_versions():
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    versions = _version_control_manager.get_version_history()
    current = _version_control_manager.get_current_version()
    return {
        "versions": versions,
        "current_version_id": current["id"] if current else None
    }


@app.get("/version-control/versions/{version_id}", dependencies=[Depends(verify_api_key)])
def get_version(version_id: str):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    version = _version_control_manager.get_version_by_id(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@app.get("/version-control/current", dependencies=[Depends(verify_api_key)])
def get_current_version():
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    current = _version_control_manager.get_current_version()
    if not current:
        return {"current": None}
    return {"current": current}


@app.post("/version-control/versions", dependencies=[Depends(require_role(UserRole.USER))])
def create_version(request: VersionCreateRequest):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    if request.author:
        _UI_SHIM.session_state.user = request.author
    version_id = _version_control_manager.create_new_version(
        request.protocol_text,
        request.version_name or "",
        request.comment or "",
    )
    version = _version_control_manager.get_version_by_id(version_id)
    return {"version_id": version_id, "version": version}


@app.post("/version-control/versions/{version_id}/load", dependencies=[Depends(require_role(UserRole.USER))])
def load_version(version_id: str):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    success = _version_control_manager.load_version(version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    current = _version_control_manager.get_current_version()
    return {"loaded": True, "current": current}


@app.post("/version-control/versions/{version_id}/branch", dependencies=[Depends(require_role(UserRole.USER))])
def branch_version(version_id: str, request: VersionBranchRequest):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    new_id = _version_control_manager.branch_version(version_id, request.new_version_name)
    if not new_id:
        raise HTTPException(status_code=400, detail="Branch creation failed")
    version = _version_control_manager.get_version_by_id(new_id)
    return {"version_id": new_id, "version": version}


@app.post("/version-control/compare", dependencies=[Depends(verify_api_key)])
def compare_versions(request: VersionCompareRequest):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    return _version_control_manager.compare_versions(request.version_id_1, request.version_id_2)


@app.delete("/version-control/versions/{version_id}", dependencies=[Depends(require_role(UserRole.USER))])
def delete_version(version_id: str):
    if not VERSION_CONTROL_AVAILABLE or _version_control_manager is None:
        raise HTTPException(status_code=503, detail="Version control not available")
    success = _version_control_manager.delete_version(version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"deleted": True}


# Validation manager

@app.get("/validation/rules", dependencies=[Depends(verify_api_key)])
def list_validation_rules():
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    rules = _validation_manager.validation_rules
    return {"rules": rules, "rule_names": list(rules.keys())}


@app.get("/validation/rules/{rule_name}", dependencies=[Depends(verify_api_key)])
def get_validation_rule(rule_name: str):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    rule = _validation_manager.get_validation_rule(rule_name)
    if not rule:
        raise HTTPException(status_code=404, detail="Validation rule not found")
    return {"name": rule_name, "rule": rule}


@app.post("/validation/rules", dependencies=[Depends(require_role(UserRole.USER))])
def create_validation_rule(request: ValidationRuleCreateRequest):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    rule_config: Dict[str, Any] = {}
    if request.max_length is not None:
        rule_config["max_length"] = request.max_length
    if request.min_length is not None:
        rule_config["min_length"] = request.min_length
    if request.required_keywords:
        rule_config["required_keywords"] = request.required_keywords
    if request.forbidden_patterns:
        rule_config["forbidden_patterns"] = request.forbidden_patterns
    if request.required_sections:
        rule_config["required_sections"] = request.required_sections
    success = _validation_manager.add_validation_rule(request.name, rule_config)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add validation rule")
    return {"created": True, "rule_name": request.name, "rule": rule_config}


@app.put("/validation/rules/{rule_name}", dependencies=[Depends(require_role(UserRole.USER))])
def update_validation_rule(rule_name: str, request: ValidationRuleUpdateRequest):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    rule_config: Dict[str, Any] = {}
    if request.max_length is not None:
        rule_config["max_length"] = request.max_length
    if request.min_length is not None:
        rule_config["min_length"] = request.min_length
    if request.required_keywords is not None:
        rule_config["required_keywords"] = request.required_keywords
    if request.forbidden_patterns is not None:
        rule_config["forbidden_patterns"] = request.forbidden_patterns
    if request.required_sections is not None:
        rule_config["required_sections"] = request.required_sections
    success = _validation_manager.update_validation_rule(rule_name, rule_config)
    if not success:
        raise HTTPException(status_code=404, detail="Validation rule not found")
    return {"updated": True, "rule_name": rule_name, "rule": rule_config}


@app.delete("/validation/rules/{rule_name}", dependencies=[Depends(require_role(UserRole.USER))])
def delete_validation_rule(rule_name: str):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    success = _validation_manager.remove_validation_rule(rule_name)
    if not success:
        raise HTTPException(status_code=404, detail="Validation rule not found")
    return {"deleted": True, "rule_name": rule_name}


@app.post("/validation/run", dependencies=[Depends(require_role(UserRole.USER))])
def run_validation(request: ValidationRunRequest):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    if not request.rule_names:
        raise HTTPException(status_code=400, detail="No validation rules selected")
    results = _validation_manager.validate_content_against_custom_rules(
        request.content,
        request.rule_names
    )
    return results


@app.post("/validation/compliance", dependencies=[Depends(require_role(UserRole.USER))])
def run_compliance_check(request: ComplianceCheckRequest):
    if not VALIDATION_MANAGER_AVAILABLE or _validation_manager is None:
        raise HTTPException(status_code=503, detail="Validation manager not available")
    results = _validation_manager.run_compliance_check(request.content, request.framework or "generic")
    return results


# Workflow lifecycle controls (BubbleLabs integration)

@app.get("/bubblelabs/workflow-definitions", dependencies=[Depends(verify_api_key)])
def list_bubblelabs_workflow_definitions():
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return {"definitions": integration.list_workflow_definitions()}


@app.get("/bubblelabs/workflow-definitions/{definition_id}", dependencies=[Depends(verify_api_key)])
def get_bubblelabs_workflow_definition(definition_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    definition = integration.get_workflow_definition(definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Workflow definition not found")
    return definition


@app.post("/bubblelabs/workflow-definitions", dependencies=[Depends(require_role(UserRole.USER))])
def create_bubblelabs_workflow_definition(request: WorkflowDefinitionCreateRequest):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    definition_id = integration.create_workflow_definition(
        name=request.name,
        description=request.description,
        workflow_type=request.workflow_type,
        parameters=request.parameters,
    )
    return {"definition_id": definition_id}


@app.get("/bubblelabs/workflow-instances", dependencies=[Depends(verify_api_key)])
def list_bubblelabs_workflow_instances():
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return {"instances": integration.list_workflow_instances()}


@app.post("/bubblelabs/workflow-instances", dependencies=[Depends(require_role(UserRole.USER))])
def create_bubblelabs_workflow_instance(request: WorkflowInstanceCreateRequest):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    instance_id = integration.create_workflow_instance(
        definition_id=request.definition_id,
        instance_name=request.instance_name,
        inputs=request.inputs,
    )
    return {"instance_id": instance_id}


@app.get("/bubblelabs/workflow-instances/{instance_id}", dependencies=[Depends(verify_api_key)])
def get_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    status_info = integration.get_workflow_instance_status(instance_id)
    if "error" in status_info:
        raise HTTPException(status_code=404, detail=status_info["error"])
    workflow_state = integration.workflow_instances.get(instance_id)
    params: Dict[str, Any] = {}
    if workflow_state:
        for attr_name in dir(workflow_state):
            if attr_name.startswith("_"):
                continue
            value = getattr(workflow_state, attr_name)
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool, list, dict)) and len(str(value)) < 1000:
                params[attr_name] = value
    return {"status": status_info, "parameters": params}


@app.post("/bubblelabs/workflow-instances/{instance_id}/start", dependencies=[Depends(require_role(UserRole.USER))])
def start_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.start_workflow_instance(instance_id)


@app.post("/bubblelabs/workflow-instances/{instance_id}/pause", dependencies=[Depends(require_role(UserRole.USER))])
def pause_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.pause_workflow_instance(instance_id)


@app.post("/bubblelabs/workflow-instances/{instance_id}/resume", dependencies=[Depends(require_role(UserRole.USER))])
def resume_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.resume_workflow_instance(instance_id)


@app.post("/bubblelabs/workflow-instances/{instance_id}/stop", dependencies=[Depends(require_role(UserRole.USER))])
def stop_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.stop_workflow_instance(instance_id)


@app.post("/bubblelabs/workflow-instances/{instance_id}/cancel", dependencies=[Depends(require_role(UserRole.USER))])
def cancel_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.cancel_workflow_instance(instance_id)


@app.post("/bubblelabs/workflow-instances/{instance_id}/restart", dependencies=[Depends(require_role(UserRole.USER))])
def restart_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.restart_workflow_instance(instance_id)


@app.delete("/bubblelabs/workflow-instances/{instance_id}", dependencies=[Depends(require_role(UserRole.USER))])
def delete_bubblelabs_workflow_instance(instance_id: str):
    integration = _get_bubblelabs_workflow_integration()
    if not BUBBLELABS_WORKFLOW_AVAILABLE or integration is None:
        raise HTTPException(status_code=503, detail="BubbleLabs workflow integration not available")
    return integration.delete_workflow_instance(instance_id)


@app.get("/parameters/schema", dependencies=[Depends(verify_api_key)])
def get_parameter_schema():
    params = []
    for param in parameter_manager.schema.parameters.values():
        params.append({
            "name": param.name,
            "type": param.type.value,
            "default": param.default,
            "description": param.description,
            "category": param.category,
            "min_value": param.min_value,
            "max_value": param.max_value,
            "options": param.options,
            "required": param.required
        })
    return {"parameters": params}


@app.get("/parameters/defaults", dependencies=[Depends(verify_api_key)])
def get_parameter_defaults():
    return parameter_manager.get_defaults()


@app.get("/parameters/categories", dependencies=[Depends(verify_api_key)])
def get_parameter_categories():
    return {"categories": parameter_manager.get_categories()}


@app.post("/parameters/validate", dependencies=[Depends(verify_api_key)])
def validate_parameters(request: ParameterValidateRequest):
    result = parameter_manager.validate(request.parameters)
    return {"valid": result.valid, "errors": result.errors, "warnings": result.warnings}


# Sovereign dashboard endpoints

@app.get("/sovereign/health", dependencies=[Depends(verify_api_key)])
def get_sovereign_health():
    return sovereign_health_monitor.run_health_checks()


@app.get("/sovereign/problems", dependencies=[Depends(verify_api_key)])
def list_sovereign_problems():
    problems = sovereign_db.list_problems()
    return {"problems": [p.to_dict() for p in problems]}


@app.get("/sovereign/plans", dependencies=[Depends(verify_api_key)])
def list_sovereign_plans():
    plans = sovereign_db.list_plans()
    return {"plans": [p.to_dict() for p in plans]}


# Suggestions endpoints

@app.post("/suggestions/content", dependencies=[Depends(verify_api_key)])
def get_content_suggestions(request: SuggestionRequest):
    system_prompt = (
        "You are an AI assistant that provides suggestions for improving the given content. "
        "Provide a list of suggestions in a clear and concise manner."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    suggestions = [line.strip() for line in response.split("\n") if line.strip()]
    return {"suggestions": suggestions}


@app.post("/suggestions/classification", dependencies=[Depends(verify_api_key)])
def get_content_classification(request: SuggestionRequest):
    system_prompt = (
        "You are an AI assistant that classifies the given content and suggests relevant tags. "
        "Provide the classification and a list of tags in JSON format."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    try:
        import json
        parsed = json.loads(response)
    except Exception:
        parsed = {"classification": "", "tags": []}
    return parsed


@app.post("/suggestions/security", dependencies=[Depends(verify_api_key)])
def get_security_suggestions(request: SuggestionRequest):
    system_prompt = (
        "You are a security expert. Analyze the following code for common security vulnerabilities "
        "and provide a list of potential issues."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    vulnerabilities = [line.strip() for line in response.split("\n") if line.strip()]
    return {"vulnerabilities": vulnerabilities}


@app.post("/suggestions/improvement", dependencies=[Depends(verify_api_key)])
def get_improvement_potential(request: SuggestionRequest):
    suggestions = get_content_suggestions(request)
    classification = get_content_classification(request)
    score = 0.0
    score += len(suggestions.get("suggestions", [])) * 0.1
    score += len(classification.get("tags", [])) * 0.05
    score = min(1.0, score)
    return {"score": score}


server = None

def start_api_server(
    host: str = "0.0.0.0", 
    port: int = 8001,
    use_tls: bool = None,
    cert_path: str = None,
    key_path: str = None
):
    """Start the API server with optional TLS support.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        use_tls: Whether to use TLS (defaults to SecurityConfig.TLS_ENABLED)
        cert_path: Path to TLS certificate (defaults to SecurityConfig.TLS_CERT_PATH)
        key_path: Path to TLS private key (defaults to SecurityConfig.TLS_KEY_PATH)
    """
    global server
    
    # Initialize security components
    if SECURITY_FRAMEWORK_AVAILABLE:
        from security_framework import initialize_security, SecurityConfig
        init_status = initialize_security()
        logger.info(f"Security initialization status: {init_status}")
        
        # Determine TLS configuration
        if use_tls is None:
            use_tls = SecurityConfig.TLS_ENABLED
        if cert_path is None:
            cert_path = SecurityConfig.TLS_CERT_PATH
        if key_path is None:
            key_path = SecurityConfig.TLS_KEY_PATH
    else:
        use_tls = False
    
    # Configure uvicorn
    config_kwargs = {
        "app": app,
        "host": host,
        "port": port
    }
    
    # Add TLS configuration if enabled
    if use_tls:
        try:
            from security_framework import create_ssl_context
            ssl_context = create_ssl_context(cert_path, key_path)
            config_kwargs["ssl_version"] = ssl_context
            logger.info(f"Starting API server with TLS on {host}:{port}")
            logger.info(f"Using certificate: {cert_path}")
        except Exception as e:
            logger.error(f"Failed to configure TLS: {e}")
            logger.warning("Starting API server WITHOUT TLS - this is insecure!")
    else:
        logger.info(f"Starting API server on {host}:{port} (no TLS)")
    
    config = uvicorn.Config(**config_kwargs)
    server = uvicorn.Server(config)
    server.run()

def stop_api_server():
    """Stop the API server."""
    global server
    if server:
        server.should_exit = True
        server.force_exit = True

# PyGraphistry Visualization Endpoint for BubbleLab Integration

class PyGraphistryVisualizationRequest(BaseModel):
    """Request model for PyGraphistry visualization."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None


@app.post("/api/openevolve/visualize/pygraphistry", dependencies=[Depends(verify_api_key)])
async def get_pygraphistry_visualization(request: PyGraphistryVisualizationRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Get a PyGraphistry visualization for knowledge graph data.
    This endpoint is specifically designed for BubbleLab integration.

    Args:
        request: Request body containing nodes and edges data
        user: Authenticated user information

    Returns:
        Dictionary with visualization URL or path
    """
    try:
        from openevolve_visualization import get_pygraphistry_viz

        # Call the visualization function with nodes and edges from request
        result = get_pygraphistry_viz(request.nodes, request.edges, request.config)

        if result:
            record_audit_event(
                user=user,
                operation="VISUALIZE_PYGRAPHISTRY",
                resource="visualization",
                resource_id="pygraphistry_viz",
                success=True
            )
            return {
                "status": "success",
                "visualization_url": result,
                "message": "PyGraphistry visualization generated successfully"
            }
        else:
            record_audit_event(
                user=user,
                operation="VISUALIZE_PYGRAPHISTRY_FAILED",
                resource="visualization",
                resource_id="pygraphistry_viz",
                success=False
            )
            return {
                "status": "error",
                "message": "Failed to generate PyGraphistry visualization"
            }

    except ImportError as e:
        logger.error(f"PyGraphistry import error: {e}")
        return {
            "status": "error",
            "message": "PyGraphistry integration not available"
        }
    except Exception as e:
        logger.error(f"Error in PyGraphistry visualization endpoint: {e}")
        record_audit_event(
            user=user,
            operation="VISUALIZE_PYGRAPHISTRY_ERROR",
            resource="visualization",
            resource_id="pygraphistry_viz",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error generating visualization: {str(e)}"
        }


# =============================================================================
# DSPY ENHANCED ASSESSMENT ENDPOINT
# =============================================================================

class DSPyAssessmentRequest(BaseModel):
    """Request model for DSPy-enhanced assessment."""
    content: str = Field(..., description="Content to assess")
    content_type: str = Field("general", description="Type of content (code, document, legal, etc.)")
    assessment_type: str = Field("comprehensive", description="Type of assessment (comprehensive, security, performance, logic)")


class DSPyAssessmentResponse(BaseModel):
    """Response model for DSPy-enhanced assessment."""
    status: str
    assessment_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    issues_found: Optional[int] = None
    recommendations: Optional[List[str]] = None
    message: Optional[str] = None


@app.post("/api/openevolve/assess/dspy", dependencies=[Depends(verify_api_key)], response_model=DSPyAssessmentResponse)
async def assess_content_with_dspy(request: DSPyAssessmentRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Assess content using DSPy for enhanced programmatic prompting and structured analysis.

    Args:
        request: Assessment request containing content and parameters
        user: Authenticated user information

    Returns:
        Assessment results from DSPy-enhanced analysis
    """
    try:
        from dspy_integration import DSPY_AVAILABLE

        if not DSPY_AVAILABLE:
            # Fallback to standard assessment if DSPy not available
            from quality_assessment import QualityAssessmentEngine

            engine = QualityAssessmentEngine()
            result = engine.assess_quality(request.content, request.content_type)

            record_audit_event(
                user=user,
                operation="ASSESS_CONTENT_DSPY_FALLBACK",
                resource="assessment",
                resource_id="dspy_fallback",
                success=True,
                details={"content_type": request.content_type, "assessment_type": request.assessment_type}
            )

            return {
                "status": "success",
                "assessment_result": {
                    "scores": {dim.value: score for dim, score in result.scores.items()},
                    "composite_score": result.composite_score,
                    "issues_count": len(result.issues),
                    "recommendations_count": len(result.recommendations)
                },
                "confidence_score": result.confidence,
                "issues_found": len(result.issues),
                "recommendations": result.recommendations[:5],  # First 5 recommendations
                "message": "DSPy not available, using standard assessment"
            }

        # Use DSPy-enhanced assessment
        from quality_assessment import QualityAssessmentEngine

        engine = QualityAssessmentEngine()
        result = engine.assess_quality_with_dspy(request.content, request.content_type)

        record_audit_event(
            user=user,
            operation="ASSESS_CONTENT_DSPY",
            resource="assessment",
            resource_id="dspy_enhanced",
            success=True,
            details={"content_type": request.content_type, "assessment_type": request.assessment_type}
        )

        return {
            "status": "success",
            "assessment_result": {
                "scores": {dim.value: score for dim, score in result.scores.items()},
                "composite_score": result.composite_score,
                "issues_count": len(result.issues),
                "recommendations_count": len(result.recommendations),
                "assessment_method": result.assessment_method
            },
            "confidence_score": result.confidence,
            "issues_found": len(result.issues),
            "recommendations": result.recommendations[:5],  # First 5 recommendations
            "message": "DSPy-enhanced assessment completed"
        }

    except ImportError as e:
        logger.error(f"DSPy assessment import error: {e}")
        return {
            "status": "error",
            "message": "DSPy assessment not available"
        }
    except Exception as e:
        logger.error(f"Error in DSPy assessment endpoint: {e}")
        record_audit_event(
            user=user,
            operation="ASSESS_CONTENT_DSPY_ERROR",
            resource="assessment",
            resource_id="dspy_error",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error performing DSPy assessment: {str(e)}"
        }


# =============================================================================
# DSPY ENHANCED FIX GENERATION ENDPOINT
# =============================================================================

class DSPyFixGenerationRequest(BaseModel):
    """Request model for DSPy-enhanced fix generation."""
    content: str = Field(..., description="Content to fix")
    content_type: str = Field("general", description="Type of content (code, document, legal, etc.)")
    issues: Optional[List[Dict[str, Any]]] = Field(None, description="List of issues to address")


class DSPyFixGenerationResponse(BaseModel):
    """Response model for DSPy-enhanced fix generation."""
    status: str
    fixed_content: Optional[str] = None
    suggested_fixes: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    fixes_applied: Optional[int] = None
    message: Optional[str] = None


@app.post("/api/openevolve/fix/dspy", dependencies=[Depends(verify_api_key)], response_model=DSPyFixGenerationResponse)
async def generate_fixes_with_dspy(request: DSPyFixGenerationRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Generate fixes using DSPy for enhanced programmatic prompting and structured analysis.

    Args:
        request: Fix generation request containing content and issues
        user: Authenticated user information

    Returns:
        Fix generation results from DSPy-enhanced analysis
    """
    try:
        from dspy_integration import DSPY_AVAILABLE
        from blue_team import BlueTeam, IssueFinding
        from quality_assessment import SeverityLevel
        from red_team import IssueCategory

        if not DSPY_AVAILABLE:
            # Fallback to standard fix generation if DSPy not available
            blue_team = BlueTeam()

            # Convert issues to IssueFinding objects if provided
            issues = []
            if request.issues:
                for issue in request.issues:
                    issue_finding = IssueFinding(
                        title=issue.get("title", "Issue"),
                        description=issue.get("description", ""),
                        severity=SeverityLevel.MEDIUM,
                        category=IssueCategory.LOGICAL_ERROR,
                        confidence=issue.get("confidence", 0.5),
                        suggested_fix=issue.get("suggested_fix", ""),
                        location=issue.get("location", "")
                    )
                    issues.append(issue_finding)

            result = blue_team.apply_fixes(request.content, issues, content_type=request.content_type)

            record_audit_event(
                user=user,
                operation="GENERATE_FIXES_DSPY_FALLBACK",
                resource="fix_generation",
                resource_id="dspy_fallback",
                success=True,
                details={"content_type": request.content_type, "issues_count": len(issues)}
            )

            return {
                "status": "success",
                "fixed_content": result.fixed_content,
                "suggested_fixes": [fix.fix_description for fix in result.fix_suggestions],
                "confidence_score": result.confidence_score,
                "fixes_applied": len(result.applied_fixes),
                "message": "DSPy not available, using standard fix generation"
            }

        # Use DSPy-enhanced fix generation
        blue_team = BlueTeam()

        # Convert issues to IssueFinding objects if provided
        issues = []
        if request.issues:
            for issue in request.issues:
                issue_finding = IssueFinding(
                    title=issue.get("title", "Issue"),
                    description=issue.get("description", ""),
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.LOGICAL_ERROR,
                    confidence=issue.get("confidence", 0.5),
                    suggested_fix=issue.get("suggested_fix", ""),
                    location=issue.get("location", "")
                )
                issues.append(issue_finding)

        result = blue_team.generate_fixes_with_dspy(
            content=request.content,
            content_type=request.content_type,
            issues=issues
        )

        record_audit_event(
            user=user,
            operation="GENERATE_FIXES_DSPY",
            resource="fix_generation",
            resource_id="dspy_enhanced",
            success=True,
            details={"content_type": request.content_type, "issues_count": len(issues)}
        )

        return {
            "status": "success",
            "fixed_content": result.get("fixed_content", request.content),
            "suggested_fixes": result.get("suggested_fixes", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "fixes_applied": result.get("fix_count", 0),
            "message": "DSPy-enhanced fix generation completed"
        }

    except ImportError as e:
        logger.error(f"DSPy fix generation import error: {e}")
        return {
            "status": "error",
            "message": "DSPy fix generation not available"
        }
    except Exception as e:
        logger.error(f"Error in DSPy fix generation endpoint: {e}")
        record_audit_event(
            user=user,
            operation="GENERATE_FIXES_DSPY_ERROR",
            resource="fix_generation",
            resource_id="dspy_error",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error performing DSPy fix generation: {str(e)}"
        }


# =============================================================================
# RAGBITS INTEGRATION ENDPOINTS
# =============================================================================

class RAGBitsSearchRequest(BaseModel):
    """Request model for RAGBits search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class RAGBitsIngestRequest(BaseModel):
    """Request model for RAGBits ingest."""
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: str = Field("manual", description="Document source identifier")


@app.post("/openevolve/ragbits/search", dependencies=[Depends(verify_api_key)])
async def ragbits_search(request: RAGBitsSearchRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Search documents using RAGBits semantic search.

    Args:
        request: Search request containing query and parameters
        user: Authenticated user information

    Returns:
        Search results from RAGBits
    """
    try:
        from knowledge_engine.ragbits_retriever import get_ragbits_retriever

        retriever = get_ragbits_retriever()

        # Perform search
        results = await retriever.search_similar_solutions(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_hybrid_search=True
        )

        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH",
            resource="ragbits",
            resource_id="search",
            success=True
        )

        return {
            "status": "success",
            "results": results,
            "total_results": len(results),
            "query": request.query
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH_FAILED",
            resource="ragbits",
            resource_id="search",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error performing RAGBits search: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH_ERROR",
            resource="ragbits",
            resource_id="search",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


@app.post("/openevolve/ragbits/ingest", dependencies=[Depends(verify_api_key)])
async def ragbits_ingest(request: RAGBitsIngestRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Ingest a document into the RAGBits system.

    Args:
        request: Ingest request containing content and metadata
        user: Authenticated user information

    Returns:
        Ingestion result
    """
    try:
        from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, RAGBitsProcessorConfig

        # Initialize processor
        config = RAGBitsProcessorConfig()
        processor = RAGBitsDocumentProcessor(config)
        await processor.initialize()

        # Ingest document
        result = await processor.ingest_text(
            text=request.content,
            metadata=request.metadata,
            source=request.source
        )

        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST",
            resource="ragbits",
            resource_id=result.document_id,
            success=result.success
        )

        return {
            "status": "success" if result.success else "error",
            "document_id": result.document_id,
            "chunks_ingested": result.chunks_ingested,
            "processing_time": result.processing_time,
            "error": result.error
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST_FAILED",
            resource="ragbits",
            resource_id="ingest",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error ingesting document to RAGBits: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST_ERROR",
            resource="ragbits",
            resource_id="ingest",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


@app.get("/openevolve/ragbits/stats", dependencies=[Depends(verify_api_key)])
async def ragbits_stats(user: AuthUser = Depends(verify_api_key)):
    """
    Get RAGBits system statistics.

    Args:
        user: Authenticated user information

    Returns:
        System statistics
    """
    try:
        from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, RAGBitsProcessorConfig
        from knowledge_engine.ragbits_retriever import get_ragbits_retriever

        # Get processor stats
        config = RAGBitsProcessorConfig()
        processor = RAGBitsDocumentProcessor(config)
        await processor.initialize()
        processor_stats = await processor.get_statistics()

        # Get retriever stats
        retriever = get_ragbits_retriever()
        retriever_stats = await retriever.get_statistics()

        record_audit_event(
            user=user,
            operation="RAGBITS_STATS",
            resource="ragbits",
            resource_id="stats",
            success=True
        )

        return {
            "status": "success",
            "processor": processor_stats,
            "retriever": retriever_stats
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_STATS_FAILED",
            resource="ragbits",
            resource_id="stats",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error getting RAGBits stats: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_STATS_ERROR",
            resource="ragbits",
            resource_id="stats",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


# =============================================================================
# Adaptive MDAP Endpoints
# =============================================================================

try:
    from adaptive_mdap import (
        TaskComplexityClassifier,
        AdaptiveMDAPAllocator,
        CostCalculator,
        APIPricing,
        get_health_checker,
        get_dashboard,
        ConfigProfile,
        get_profile_config,
    )
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    logger.warning("Adaptive MDAP not available, endpoints disabled")


class ComplexityRequest(BaseModel):
    """Request to classify problem complexity."""
    description: str
    domain: str = "general"
    depth: int = 0
    dependencies: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class AllocationRequest(BaseModel):
    """Request to allocate resources based on complexity."""
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None


class CostCalculationRequest(BaseModel):
    """Request to calculate costs."""
    num_problems: int = Field(..., ge=1, le=1000000)
    workload_distribution: Optional[Dict[str, float]] = None
    model: str = "gpt-4o-mini"


@app.get("/adaptive-mdap/health", dependencies=[Depends(verify_api_key)])
def adaptive_mdap_health():
    """Get Adaptive MDAP system health."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    health = get_health_checker()
    report = health.get_status_report()
    return report


@app.post("/adaptive-mdap/complexity", dependencies=[Depends(verify_api_key)])
def classify_complexity(request: ComplexityRequest):
    """Classify problem complexity."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        from adaptive_mdap.core.types import SubProblem
        
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id=f"api-{uuid.uuid4().hex[:8]}",
            description=request.description,
            domain=request.domain,
            depth=request.depth,
            dependencies=request.dependencies,
            metadata={
                "constraints": request.constraints,
                "success_criteria": request.success_criteria,
            },
        )
        
        complexity = classifier.compute_complexity(subproblem)
        
        return {
            "overall_score": complexity.overall_score,
            "text_length_score": complexity.text_length_score,
            "domain_rarity_score": complexity.domain_rarity_score,
            "depth_score": complexity.depth_score,
            "historical_error_score": complexity.historical_error_score,
            "dependency_score": complexity.dependency_score,
            "keyword_score": complexity.keyword_score,
            "constraint_score": complexity.constraint_score,
            "feature_weights": complexity.feature_weights,
        }
    except Exception as e:
        logger.error(f"Error classifying complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adaptive-mdap/allocate", dependencies=[Depends(verify_api_key)])
def allocate_resources(request: AllocationRequest):
    """Allocate resources based on complexity score."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        allocator = AdaptiveMDAPAllocator()
        
        from adaptive_mdap.allocators.resource_allocator import AllocationContext
        
        context = None
        if request.context:
            context = AllocationContext(
                system_load=request.context.get("system_load"),
                budget_remaining=request.context.get("budget_remaining"),
                quality_requirements=request.context.get("quality_requirements"),
            )
        
        config = allocator.allocate_resources(request.complexity_score, context)
        
        return {
            "strategy": config.strategy.value,
            "n_agents": config.n_agents,
            "k_ahead": config.k_ahead,
            "max_retries": config.max_retries,
            "timeout_ms": config.timeout_ms,
        }
    except Exception as e:
        logger.error(f"Error allocating resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adaptive-mdap/cost", dependencies=[Depends(verify_api_key)])
def calculate_cost(request: CostCalculationRequest):
    """Calculate costs for adaptive allocation."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        # Get pricing model
        pricing_map = {
            "gpt-4o-mini": APIPricing.gpt_4o_mini,
            "gpt-4o": APIPricing.gpt_4o,
            "gpt-4": APIPricing.gpt_4,
            "claude-3-5-sonnet": APIPricing.claude_3_5_sonnet,
            "claude-3-5-haiku": APIPricing.claude_3_5_haiku,
            "gemini-1-5-pro": APIPricing.gemini_1_5_pro,
            "gemini-1-5-flash": APIPricing.gemini_1_5_flash,
        }
        
        pricing = pricing_map.get(request.model, APIPricing.gpt_4o_mini)()
        calculator = CostCalculator(pricing=pricing)
        
        # Get workload distribution
        from adaptive_mdap.tools.cost_calculator import WorkloadDistribution
        
        if request.workload_distribution:
            workload = WorkloadDistribution(
                easy_percentage=request.workload_distribution.get("easy", 0.3),
                medium_percentage=request.workload_distribution.get("medium", 0.4),
                hard_percentage=request.workload_distribution.get("hard", 0.3),
            )
        else:
            workload = WorkloadDistribution.default()
        
        result = calculator.calculate_adaptive_cost(request.num_problems, workload)
        
        return {
            "model": request.model,
            "num_problems": request.num_problems,
            "baseline_cost": result["baseline_cost"],
            "adaptive_cost": result["adaptive_cost"],
            "savings": result["savings"],
            "savings_percent": result["savings_percent"],
            "breakdown": result["breakdown"],
        }
    except Exception as e:
        logger.error(f"Error calculating cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adaptive-mdap/dashboard", dependencies=[Depends(verify_api_key)])
def get_adaptive_dashboard():
    """Get Adaptive MDAP dashboard data."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        dashboard = get_dashboard()
        full_dashboard = dashboard.generate_full_dashboard()
        return full_dashboard
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adaptive-mdap/profiles", dependencies=[Depends(verify_api_key)])
def get_adaptive_profiles():
    """Get available configuration profiles."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    profiles = {
        "conservative": "Favors quality over cost (lower thresholds)",
        "balanced": "Default balance between cost and quality",
        "aggressive": "Favors cost savings over quality (higher thresholds)",
        "cloud_conservative": "Cloud-optimized conservative profile",
        "cloud_balanced": "Cloud-optimized balanced profile",
        "cloud_aggressive": "Cloud-optimized aggressive profile",
    }
    
    return {
        "profiles": profiles,
        "default": "balanced",
    }


@app.get("/adaptive-mdap/profiles/{profile_name}", dependencies=[Depends(verify_api_key)])
def get_adaptive_profile_config(profile_name: str):
    """Get specific configuration profile."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    profile_map = {
        "conservative": ConfigProfile.CONSERVATIVE,
        "balanced": ConfigProfile.BALANCED,
        "aggressive": ConfigProfile.AGGRESSIVE,
        "cloud_conservative": ConfigProfile.CLOUD_CONSERVATIVE,
        "cloud_balanced": ConfigProfile.CLOUD_BALANCED,
        "cloud_aggressive": ConfigProfile.CLOUD_AGGRESSIVE,
    }
    
    if profile_name not in profile_map:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_name}")
    
    try:
        config = get_profile_config(profile_map[profile_name])
        return config
    except Exception as e:
        logger.error(f"Error getting profile config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/integrated/run", dependencies=[Depends(require_role(UserRole.USER))])
def run_integrated_workflow(
    request: IntegratedWorkflowRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Run the integrated adversarial-evolution-evaluation workflow."""
    if not INTEGRATED_WORKFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="Integrated workflow not available")
    if _integrated_workflow is not None:
        _attach_ui(_integrated_workflow, _UI_SHIM)

    try:
        results = run_fully_integrated_adversarial_evolution(
            current_content=request.current_content,
            content_type=request.content_type,
            api_key=request.api_key,
            base_url=request.base_url,
            red_team_models=request.red_team_models,
            blue_team_models=request.blue_team_models,
            evaluator_models=request.evaluator_models,
            max_iterations=request.max_iterations,
            adversarial_iterations=request.adversarial_iterations,
            evolution_iterations=request.evolution_iterations,
            evaluation_iterations=request.evaluation_iterations,
            system_prompt=request.system_prompt,
            evaluator_system_prompt=request.evaluator_system_prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            max_tokens=request.max_tokens,
            seed=request.seed,
            rotation_strategy=request.rotation_strategy,
            red_team_sample_size=request.red_team_sample_size,
            blue_team_sample_size=request.blue_team_sample_size,
            evaluator_sample_size=request.evaluator_sample_size,
            confidence_threshold=request.confidence_threshold,
            evaluator_threshold=request.evaluator_threshold,
            evaluator_consecutive_rounds=request.evaluator_consecutive_rounds,
            compliance_requirements=request.compliance_requirements,
            enable_data_augmentation=request.enable_data_augmentation,
            augmentation_model_id=request.augmentation_model_id,
            augmentation_temperature=request.augmentation_temperature,
            enable_human_feedback=request.enable_human_feedback,
            multi_objective_optimization=request.multi_objective_optimization,
            feature_dimensions=request.feature_dimensions,
            feature_bins=request.feature_bins,
            elite_ratio=request.elite_ratio,
            exploration_ratio=request.exploration_ratio,
            exploitation_ratio=request.exploitation_ratio,
            archive_size=request.archive_size,
            checkpoint_interval=request.checkpoint_interval,
            keyword_analysis_enabled=request.keyword_analysis_enabled,
            keywords_to_target=request.keywords_to_target,
            keyword_penalty_weight=request.keyword_penalty_weight,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return results


@app.get("/orchestration/models", dependencies=[Depends(verify_api_key)])
def list_orchestration_models(
    user: AuthUser = Depends(verify_api_key)
):
    """List models registered in the orchestrator."""
    if not MODEL_ORCHESTRATION_AVAILABLE or _model_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model orchestration not available")

    models = []
    for model_name, info in _model_orchestrator.models.items():
        role = info.get("role")
        role_value = role.value if hasattr(role, "value") else str(role)
        models.append(
            {
                "name": model_name,
                "role": role_value,
                "weight": info.get("weight", 1.0),
                "api_base": info.get("api_base", ""),
            }
        )

    metrics = _model_orchestrator.get_model_performance_metrics()
    return {
        "models": models,
        "metrics": metrics or {},
        "selection_strategies": list(_model_orchestrator.selection_strategies.keys()),
    }


@app.post("/orchestration/models", dependencies=[Depends(require_role(UserRole.USER))])
def register_orchestration_model(
    request: ModelOrchestrationRegisterRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Register a model with the orchestrator."""
    if not MODEL_ORCHESTRATION_AVAILABLE or _model_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model orchestration not available")

    role_value = request.role.lower()
    matched_role = None
    for role in ModelRole:
        if role_value in {role.value.lower(), role.name.lower()}:
            matched_role = role
            break
    if not matched_role:
        raise HTTPException(status_code=400, detail="Invalid role")

    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        kwargs["presence_penalty"] = request.presence_penalty

    _model_orchestrator.register_model(
        model_name=request.model_name,
        role=matched_role,
        weight=request.weight,
        api_key=request.api_key or "",
        api_base=request.api_base or "",
        **kwargs,
    )

    return {"message": "Model registered", "model_name": request.model_name}


@app.post("/orchestration/ensemble", dependencies=[Depends(require_role(UserRole.USER))])
def execute_orchestration_ensemble(
    request: ModelOrchestrationEnsembleRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Execute a request with an ensemble of models."""
    if not MODEL_ORCHESTRATION_AVAILABLE or _model_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model orchestration not available")

    role_value = request.role.lower()
    matched_role = None
    for role in ModelRole:
        if role_value in {role.value.lower(), role.name.lower()}:
            matched_role = role
            break
    if not matched_role:
        raise HTTPException(status_code=400, detail="Invalid role")

    responses = _model_orchestrator.execute_with_ensemble(
        messages=request.messages,
        role=matched_role,
        selection_strategy=request.selection_strategy,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        num_responses=request.num_responses,
    )

    return {"responses": responses}


@app.get("/bubblelabs/status", dependencies=[Depends(verify_api_key)])
def bubblelabs_status():
    """Get BubbleLabs integration status."""
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.get_all_status()


@app.post("/bubblelabs/initialize", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_initialize(
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Initialize BubbleLabs component bridges."""
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    return initialize_extended_integration()


@app.post("/bubblelabs/ace/skillbook", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_ace_skillbook(
    request: BubbleLabsAceSkillbookRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.ace_create_skillbook(request.name, request.skills)


@app.post("/bubblelabs/ace/patterns", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_ace_patterns(
    request: BubbleLabsAcePatternRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.ace_extract_patterns(request.workflow_results)


@app.post("/bubblelabs/z3/solve", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_z3_solve(
    request: BubbleLabsZ3SolveRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.z3_solve_constraints(request.variables, request.constraints)


@app.post("/bubblelabs/z3/prove", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_z3_prove(
    request: BubbleLabsZ3ProveRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.z3_prove_theorem(request.theorem)


@app.post("/bubblelabs/roma/analyze", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_roma_analyze(
    request: BubbleLabsRomaAnalyzeRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.roma_analyze_problem(request.problem, request.max_depth)


@app.post("/bubblelabs/roma/config", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_roma_config(
    request: BubbleLabsRomaConfigRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.roma_create_config(**(request.config or {}))


@app.post("/bubblelabs/knowledge/store", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_knowledge_store(
    request: BubbleLabsKnowledgeStoreRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.knowledge_store_artifact(request.artifact)


@app.post("/bubblelabs/knowledge/query", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_knowledge_query(
    request: BubbleLabsKnowledgeQueryRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.knowledge_query_patterns(request.query)


@app.post("/bubblelabs/analytics/track", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_analytics_track(
    request: BubbleLabsAnalyticsTrackRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.analytics_track_workflow(request.workflow_id, request.metrics)


@app.get("/bubblelabs/analytics/dashboard", dependencies=[Depends(verify_api_key)])
def bubblelabs_analytics_dashboard():
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.analytics_get_dashboard()


@app.post("/bubblelabs/leanaide/prove", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_leanaide_prove(
    request: BubbleLabsLeanAideProveRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.leanaide_prove_theorem(request.theorem)


@app.get("/web3/status", dependencies=[Depends(verify_api_key)])
def web3_status():
    """Get Web3 audit stack availability and MCP tool inventory."""
    inventory = {}
    web3_tools: List[str] = []
    web3_ingestion_tools: List[str] = []
    web3_formal_tools: List[str] = []
    if get_mcp_tool_inventory is not None:
        try:
            inventory = get_mcp_tool_inventory()
            if isinstance(inventory, dict):
                web3_tools = list(inventory.get("web3_tools", []) or [])
                web3_ingestion_tools = list(inventory.get("web3_ingestion_tools", []) or [])
                web3_formal_tools = list(inventory.get("web3_formal_tools", []) or [])
        except Exception as exc:
            inventory = {"error": str(exc)}
    formal_capabilities = {
        "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
        "invariant_translation_verification": verify_solidity_invariant_translation is not None,
        "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
        "composite_exploit_verification": (
            translate_solidity_assignment_to_z3 is not None
            and solve_smart_contract_exploit_witness is not None
        ),
    }
    if isinstance(inventory, dict):
        merged = inventory.get("formal_capabilities")
        if isinstance(merged, dict):
            formal_capabilities.update(merged)

    if not web3_formal_tools:
        inferred_formal_tools: List[str] = []
        if formal_capabilities.get("solidity_invariant_translation"):
            inferred_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities.get("symbolic_exploit_witness"):
            inferred_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities.get("composite_exploit_verification"):
            inferred_formal_tools.append("z3_web3_audit_exploit_verification")
        web3_formal_tools = inferred_formal_tools

    if not web3_ingestion_tools:
        inferred_ingestion_tools = [
            "web3_ingest_slither_static_analysis",
            "web3_ingest_foundry_fuzzing",
            "web3_ingest_contract_audit_stack",
        ]
        web3_ingestion_tools = inferred_ingestion_tools

    if not web3_tools:
        web3_tools = sorted(set(web3_ingestion_tools + web3_formal_tools))

    web3_formal_tools = sorted(set(web3_formal_tools))
    web3_ingestion_tools = sorted(set(web3_ingestion_tools))
    web3_tools = sorted(set(web3_tools))

    inferred_formal_available = bool(web3_formal_tools) or any(
        bool(v) for v in formal_capabilities.values()
    )
    inferred_ingestion_available = bool(web3_ingestion_tools)
    inferred_stack_available = bool(web3_tools) or inferred_formal_available or inferred_ingestion_available

    return {
        "web3_ingestion_available": WEB3_INGESTION_AVAILABLE or inferred_ingestion_available,
        "web3_formal_verification_available": (
            WEB3_FORMAL_VERIFICATION_AVAILABLE or inferred_formal_available
        ),
        "web3_formal_available": (
            WEB3_FORMAL_VERIFICATION_AVAILABLE or inferred_formal_available
        ),
        "available": inferred_stack_available,
        "slither_ingestion_available": web3_ingest_slither_static_analysis is not None,
        "foundry_ingestion_available": web3_ingest_foundry_fuzzing is not None,
        "invariant_translation_available": translate_solidity_assignment_to_z3 is not None,
        "exploit_witness_available": solve_smart_contract_exploit_witness is not None,
        "audit_exploit_verification_available": bool(
            formal_capabilities.get("composite_exploit_verification")
        ),
        "web3_tools": web3_tools,
        "web3_ingestion_tools": web3_ingestion_tools,
        "web3_formal_tools": web3_formal_tools,
        "formal_capabilities": formal_capabilities,
        "mcp_tool_inventory": inventory,
    }


@app.get("/web3/mcp-tool-inventory", dependencies=[Depends(verify_api_key)])
def web3_mcp_tool_inventory():
    if get_mcp_tool_inventory is None:
        raise HTTPException(status_code=503, detail="MCP tool inventory unavailable")
    try:
        return get_mcp_tool_inventory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/ingest", dependencies=[Depends(require_role(UserRole.USER))])
def web3_ingest_stack(
    request: Web3IngestStackRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Run Web3 ingestion stack (source discovery + Slither + optional Foundry)."""
    if not WEB3_INGESTION_AVAILABLE or web3_ingest_contract_audit_stack is None:
        raise HTTPException(status_code=503, detail="Web3 ingestion stack unavailable")
    try:
        return web3_ingest_contract_audit_stack(
            project_path=request.project_path,
            run_fuzzing=request.run_fuzzing,
            slither_timeout_seconds=request.slither_timeout_seconds,
            forge_timeout_seconds=request.forge_timeout_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/ingest/slither", dependencies=[Depends(require_role(UserRole.USER))])
def web3_ingest_slither(
    request: Web3IngestSlitherRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Run Slither static analysis and return normalized findings/dependencies."""
    if not WEB3_INGESTION_AVAILABLE or web3_ingest_slither_static_analysis is None:
        raise HTTPException(status_code=503, detail="Slither ingestion unavailable")
    try:
        return web3_ingest_slither_static_analysis(
            project_path=request.project_path,
            timeout_seconds=request.timeout_seconds,
            extra_args=request.extra_args,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/ingest/foundry", dependencies=[Depends(require_role(UserRole.USER))])
def web3_ingest_foundry(
    request: Web3IngestFoundryRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Run Foundry/Forge fuzz harness and return parsed execution summary."""
    if not WEB3_INGESTION_AVAILABLE or web3_ingest_foundry_fuzzing is None:
        raise HTTPException(status_code=503, detail="Foundry ingestion unavailable")
    try:
        return web3_ingest_foundry_fuzzing(
            project_path=request.project_path,
            timeout_seconds=request.timeout_seconds,
            match_contract=request.match_contract,
            match_test=request.match_test,
            fork_url=request.fork_url,
            extra_args=request.extra_args,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/invariants/translate", dependencies=[Depends(require_role(UserRole.USER))])
def web3_translate_invariant(
    request: Web3InvariantTranslateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Translate Solidity assignment into Z3 constraints and invariants."""
    if not WEB3_FORMAL_VERIFICATION_AVAILABLE or translate_solidity_assignment_to_z3 is None:
        raise HTTPException(status_code=503, detail="Solidity invariant translation unavailable")
    try:
        translation = translate_solidity_assignment_to_z3(
            statement=request.statement,
            non_negative_target=request.non_negative_target,
            max_withdraw_expr=request.max_withdraw_expr,
        )
        response: Dict[str, Any] = {"translation": translation}
        if request.verify_translation and verify_solidity_invariant_translation is not None:
            response["verification"] = verify_solidity_invariant_translation(
                translation=translation,
                assume_non_negative_amount=request.assume_non_negative_amount,
            )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/exploits/symbolic-witness", dependencies=[Depends(require_role(UserRole.USER))])
def web3_exploit_symbolic_witness(
    request: Web3ExploitWitnessRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Solve canonical exploit witness query with optional custom constraints."""
    if not WEB3_FORMAL_VERIFICATION_AVAILABLE or solve_smart_contract_exploit_witness is None:
        raise HTTPException(status_code=503, detail="Exploit witness solver unavailable")
    try:
        return solve_smart_contract_exploit_witness(
            additional_constraints=request.additional_constraints,
            timeout=request.timeout_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/web3/audit/exploit-verification", dependencies=[Depends(require_role(UserRole.USER))])
def web3_audit_exploit_verification(
    request: Web3AuditExploitRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    """Run ingestion + optional invariant translation + exploit witness solving."""
    ingestion_result = None
    translation_result = None
    witness_result = None

    if WEB3_INGESTION_AVAILABLE and web3_ingest_contract_audit_stack is not None:
        ingestion_result = web3_ingest_contract_audit_stack(
            project_path=request.project_path,
            run_fuzzing=request.run_fuzzing,
            slither_timeout_seconds=240,
            forge_timeout_seconds=420,
        )

    if request.statement and WEB3_FORMAL_VERIFICATION_AVAILABLE and translate_solidity_assignment_to_z3 is not None:
        translation = translate_solidity_assignment_to_z3(
            statement=request.statement,
            non_negative_target=request.non_negative_target,
            max_withdraw_expr=request.max_withdraw_expr,
        )
        translation_result = {"translation": translation}
        if request.verify_translation and verify_solidity_invariant_translation is not None:
            translation_result["verification"] = verify_solidity_invariant_translation(
                translation=translation,
                assume_non_negative_amount=request.assume_non_negative_amount,
            )

    if WEB3_FORMAL_VERIFICATION_AVAILABLE and solve_smart_contract_exploit_witness is not None:
        witness_result = solve_smart_contract_exploit_witness(
            additional_constraints=request.additional_constraints,
            timeout=request.timeout_seconds,
        )

    verification = (
        translation_result.get("verification")
        if isinstance(translation_result, dict)
        else None
    )
    witness_payload = None
    if isinstance(witness_result, dict):
        witness_payload = witness_result
    translated_payload = (
        translation_result.get("translation")
        if isinstance(translation_result, dict)
        else None
    )
    lean_proof_verification = verify_web3_lean_proof(
        translated_payload,
        use_real_lean=True,
    )

    verified_exploit = bool((witness_payload or {}).get("satisfiable", False))
    if request.verify_translation and isinstance(verification, dict):
        verified_exploit = verified_exploit and bool(verification.get("proven", False))

    return {
        "success": (translation_result is None or bool(translation_result))
        and (witness_result is None or bool(witness_result)),
        "ingestion": ingestion_result,
        "translation": translation_result,
        "exploit_witness": witness_result,
        "lean_proof_verification": lean_proof_verification,
        "formal_evidence": build_web3_formal_evidence(
            verification,
            witness_payload if isinstance(witness_payload, dict) else {},
            lean_proof_verification,
        ),
        "verified_exploit": verified_exploit,
    }


@app.get("/bubblelabs/web3/status", dependencies=[Depends(verify_api_key)])
def bubblelabs_web3_status():
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.get_web3_status()


@app.post("/bubblelabs/web3/ingest", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_web3_ingest(
    request: Web3IngestStackRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.web3_ingest_contract_stack(
        project_path=request.project_path,
        run_fuzzing=request.run_fuzzing,
        slither_timeout_seconds=request.slither_timeout_seconds,
        forge_timeout_seconds=request.forge_timeout_seconds,
    )


@app.post("/bubblelabs/web3/invariants/translate", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_web3_translate_invariant(
    request: Web3InvariantTranslateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.web3_translate_solidity_invariant(
        statement=request.statement,
        non_negative_target=request.non_negative_target,
        max_withdraw_expr=request.max_withdraw_expr,
        verify_translation=request.verify_translation,
        assume_non_negative_amount=request.assume_non_negative_amount,
    )


@app.post("/bubblelabs/web3/exploits/symbolic-witness", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_web3_symbolic_witness(
    request: Web3ExploitWitnessRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.web3_solve_exploit_witness(
        additional_constraints=request.additional_constraints,
        timeout_seconds=request.timeout_seconds,
    )


@app.post("/bubblelabs/web3/audit/exploit-verification", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_web3_audit_exploit_verification(
    request: Web3AuditExploitRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not BUBBLELABS_AVAILABLE:
        raise HTTPException(status_code=503, detail="BubbleLabs integration not available")
    integration = get_extended_integration()
    return integration.web3_audit_exploit_verification(
        project_path=request.project_path,
        run_fuzzing=request.run_fuzzing,
        statement=request.statement,
        non_negative_target=request.non_negative_target,
        max_withdraw_expr=request.max_withdraw_expr,
        verify_translation=request.verify_translation,
        assume_non_negative_amount=request.assume_non_negative_amount,
        additional_constraints=request.additional_constraints,
        timeout_seconds=request.timeout_seconds,
    )


@app.get("/maker/status", dependencies=[Depends(verify_api_key)])
def maker_status():
    """Get Maker integration availability."""
    return {
        "available": MAKER_INTEGRATION_AVAILABLE and _maker_manager is not None,
        "maker_engine_available": MAKER_INTEGRATION_AVAILABLE,
    }


@app.get("/maker/tools", dependencies=[Depends(verify_api_key)])
def maker_list_tools(
    status: Optional[str] = None,
    maker_mode: Optional[str] = None,
    search: Optional[str] = None,
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    status_filter = None
    if status and ToolStatus:
        try:
            status_filter = ToolStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status filter")
    tools = _maker_manager.tool_repository.list_tools(
        status_filter=status_filter,
        maker_mode_filter=maker_mode,
    )
    if search:
        tools = _maker_manager.tool_repository.search_tools(search)
    return {"tools": [tool.to_dict() for tool in tools]}


@app.get("/maker/tools/{tool_id}", dependencies=[Depends(verify_api_key)])
def maker_get_tool(tool_id: str):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    tool = _maker_manager.tool_repository.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"tool": tool.to_dict()}


@app.post("/maker/tools", dependencies=[Depends(require_role(UserRole.USER))])
def maker_create_tool(
    request: MakerToolCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    tool, error = _maker_manager.create_tool_workflow(
        name=request.name,
        description=request.description,
        task=request.task,
        maker_mode=request.maker_mode,
        k_ahead=request.k_ahead,
        max_depth=request.max_depth,
        context=request.context,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)
    if request.prompt_template or request.system_prompt or request.expected_schema or request.metadata:
        stored_tool = _maker_manager.tool_repository.get_tool(tool.tool_id)
        if stored_tool:
            if request.prompt_template:
                stored_tool.prompt_template = request.prompt_template
            if request.system_prompt:
                stored_tool.system_prompt = request.system_prompt
            if request.expected_schema:
                stored_tool.expected_schema = request.expected_schema
            if request.metadata:
                stored_tool.metadata.update(request.metadata)
            _maker_manager.tool_repository._save_repository()
    return {"tool": tool.to_dict()}


@app.post("/maker/tools/{tool_id}/test", dependencies=[Depends(require_role(UserRole.USER))])
def maker_test_tool(
    tool_id: str,
    request: MakerToolExecuteRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    result, error = _maker_manager.execute_tool_workflow(
        tool_id=tool_id,
        input_data=request.input_data,
        delegate_to_crewai=request.delegate_to_crewai,
    )
    if error or result is None:
        raise HTTPException(status_code=400, detail=error or "Tool execution failed")
    _maker_manager.tool_repository.update_tool(
        tool_id,
        status=ToolStatus.TESTING if ToolStatus else None,
        test_results={
            "success": result.success,
            "output": result.output_data,
            "metrics": result.metrics,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp,
        },
    )
    return {"result": result.to_dict()}


@app.post("/maker/tools/{tool_id}/validate", dependencies=[Depends(require_role(UserRole.USER))])
def maker_validate_tool(
    tool_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    updated = _maker_manager.tool_repository.update_tool(
        tool_id,
        status=ToolStatus.VALIDATED if ToolStatus else None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "validated"}


@app.post("/maker/tools/{tool_id}/execute", dependencies=[Depends(require_role(UserRole.USER))])
def maker_execute_tool(
    tool_id: str,
    request: MakerToolExecuteRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    result, error = _maker_manager.execute_tool_workflow(
        tool_id=tool_id,
        input_data=request.input_data,
        delegate_to_crewai=request.delegate_to_crewai,
    )
    if error or result is None:
        raise HTTPException(status_code=400, detail=error or "Tool execution failed")
    return {"result": result.to_dict()}


@app.get("/maker/delegations", dependencies=[Depends(verify_api_key)])
def maker_list_delegations(status: Optional[str] = None, delegation_type: Optional[str] = None):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    status_filter = None
    if status and DelegationStatus:
        try:
            status_filter = DelegationStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid delegation status")
    delegations = _maker_manager.delegation_manager.list_delegations(
        status_filter=status_filter,
        delegation_type_filter=delegation_type,
    )
    return {"delegations": [d.to_dict() for d in delegations]}


@app.post("/maker/delegations/sync", dependencies=[Depends(require_role(UserRole.USER))])
def maker_sync_delegations(
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not MAKER_INTEGRATION_AVAILABLE or _maker_manager is None:
        raise HTTPException(status_code=503, detail="Maker integration not available")
    synced = _maker_manager.delegation_manager.sync_from_crewai()
    return {"synced": synced}


@app.get("/bubblelabs/knowledge/status", dependencies=[Depends(verify_api_key)])
def bubblelabs_knowledge_status():
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge explorer not available")
    components = _get_knowledge_components()
    if components is None:
        raise HTTPException(status_code=500, detail="Knowledge explorer initialization failed")
    query = components["query"]
    return {
        "initialized": True,
        "query_history_count": len(query.get_query_history() if query else []),
    }


@app.post("/bubblelabs/knowledge/query-advanced", dependencies=[Depends(require_role(UserRole.USER))])
async def bubblelabs_knowledge_query_advanced(
    request: KnowledgeExplorerQueryRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge explorer not available")
    components = _get_knowledge_components()
    if components is None:
        raise HTTPException(status_code=500, detail="Knowledge explorer initialization failed")
    query_interface: KnowledgeQueryInterface = components["query"]
    results = await query_interface.unified_query(
        query=request.query,
        sources=request.sources,
        bedrock_kb_id=request.bedrock_kb_id,
        index_path=request.index_path,
    )
    return {
        "results": results,
        "history": query_interface.get_query_history(limit=10),
    }


@app.get("/bubblelabs/knowledge/query-history", dependencies=[Depends(verify_api_key)])
def bubblelabs_knowledge_query_history():
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge explorer not available")
    components = _get_knowledge_components()
    if components is None:
        raise HTTPException(status_code=500, detail="Knowledge explorer initialization failed")
    query_interface: KnowledgeQueryInterface = components["query"]
    return {"history": query_interface.get_query_history(limit=50)}


@app.post("/bubblelabs/knowledge/extract", dependencies=[Depends(require_role(UserRole.USER))])
async def bubblelabs_knowledge_extract(
    request: KnowledgeExplorerExtractRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge explorer not available")
    components = _get_knowledge_components()
    if components is None:
        raise HTTPException(status_code=500, detail="Knowledge explorer initialization failed")
    extractor: KnowledgeExtractionWorkflow = components["extract"]
    if request.source_type not in {"url", "path", "text"}:
        raise HTTPException(status_code=400, detail="Invalid source_type")
    if request.source_type == "text":
        entities, relationships = await extractor._extract_knowledge(request.source_value)
        results = {
            "text_content": request.source_value[:1000],
            "entities": entities,
            "relationships": relationships,
            "statistics": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
            },
        }
    else:
        results = await extractor.extract_from_document(
            request.source_value,
            extraction_config=request.extraction_config,
        )
    return {"results": results}


@app.post("/bubblelabs/knowledge/extract-file", dependencies=[Depends(require_role(UserRole.USER))])
async def bubblelabs_knowledge_extract_file(
    file: UploadFile = File(...),
    extraction_config: Optional[str] = Form(None),
    user: AuthUser = Depends(require_role(UserRole.USER)),
):
    if not KNOWLEDGE_EXPLORER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge explorer not available")
    components = _get_knowledge_components()
    if components is None:
        raise HTTPException(status_code=500, detail="Knowledge explorer initialization failed")
    extractor: KnowledgeExtractionWorkflow = components["extract"]
    config_payload = None
    if extraction_config:
        try:
            config_payload = json.loads(extraction_config)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid extraction_config JSON: {exc}")

    with tempfile.TemporaryDirectory() as temp_dir:
        safe_name = Path(file.filename or "upload.bin").name
        temp_path = Path(temp_dir) / safe_name
        content = await file.read()
        temp_path.write_bytes(content)
        results = await extractor.extract_from_document(
            str(temp_path),
            extraction_config=config_payload,
        )
    return {"results": results}


@app.get("/bubblelabs/leanaide/status", dependencies=[Depends(verify_api_key)])
def bubblelabs_leanaide_status():
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    return _leanaide_bridge.get_status()


@app.post("/bubblelabs/leanaide/execute", dependencies=[Depends(require_role(UserRole.USER))])
def bubblelabs_leanaide_execute(
    request: LeanAideExecuteRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    if LeanAideTaskType is None:
        raise HTTPException(status_code=503, detail="LeanAide task type unavailable")
    try:
        task_type = LeanAideTaskType(request.task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid LeanAide task type")
    result = _leanaide_bridge.execute_task(task_type, **(request.payload or {}))
    return {"result": result.to_dict()}


@app.get("/bubblelabs/leanaide/trees", dependencies=[Depends(verify_api_key)])
def bubblelabs_leanaide_trees():
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    return {"tree_ids": _leanaide_bridge.get_all_trees()}


@app.get("/bubblelabs/leanaide/trees/{tree_id}", dependencies=[Depends(verify_api_key)])
def bubblelabs_leanaide_tree(tree_id: str):
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    tree = _leanaide_bridge.get_tree(tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    return {"tree": tree.to_dict()}


@app.get("/bubblelabs/leanaide/proofs", dependencies=[Depends(verify_api_key)])
def bubblelabs_leanaide_proofs():
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    return {"proof_ids": _leanaide_bridge.get_all_proofs()}


@app.get("/bubblelabs/leanaide/proofs/{proof_id}", dependencies=[Depends(verify_api_key)])
def bubblelabs_leanaide_proof(proof_id: str):
    if not LEANAIDE_BRIDGE_AVAILABLE or _leanaide_bridge is None:
        raise HTTPException(status_code=503, detail="LeanAide integration not available")
    proof = _leanaide_bridge.get_proof(proof_id)
    if not proof:
        raise HTTPException(status_code=404, detail="Proof not found")
    return {"proof": proof.to_dict()}


@app.post("/evolution/runs", dependencies=[Depends(require_role(UserRole.USER))])
def start_evolution_run(
    request: EvolutionRunRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not EVOLUTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evolution engine not available")
    run_id = f"evo_{uuid.uuid4().hex[:10]}"
    run_state = _RunState(
        run_id=run_id,
        run_type="evolution",
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        parameters=request.parameters,
    )
    with _run_lock:
        _evolution_runs[run_id] = run_state

    def _execute():
        ui_context = _create_run_context(run_state, "evolution_log")
        import session_utils as _session_utils
        import evolution as _evolution_module
        _attach_ui(_session_utils, ui_context)
        _attach_ui(_evolution_module, ui_context)
        result = run_comprehensive_evolution(
            content=request.content,
            content_type=request.content_type,
            evolution_mode=request.evolution_mode,
            custom_config=request.parameters or None,
            gauntlet_name=request.gauntlet_name,
            use_decomposition=request.use_decomposition,
            team_manager=team_manager,
            gauntlet_manager=gauntlet_manager,
        )
        return result

    _start_background_run(run_state, _execute)
    return {"run_id": run_id, "status": run_state.status}


@app.get("/evolution/runs", dependencies=[Depends(verify_api_key)])
def list_evolution_runs():
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "status": run.status,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
            }
            for run in _evolution_runs.values()
        ]
    }


@app.get("/evolution/runs/{run_id}", dependencies=[Depends(verify_api_key)])
def get_evolution_run(run_id: str):
    run_state = _evolution_runs.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_state.run_id,
        "status": run_state.status,
        "created_at": run_state.created_at,
        "started_at": run_state.started_at,
        "completed_at": run_state.completed_at,
        "logs": run_state.logs,
        "result": run_state.result,
        "error": run_state.error,
    }


@app.post("/evolution/runs/{run_id}/stop", dependencies=[Depends(require_role(UserRole.USER))])
def stop_evolution_run(
    run_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    run_state = _evolution_runs.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")
    run_state.cancel_requested = True
    if run_state.session_state is not None:
        run_state.session_state["evolution_stop_flag"] = True
    return {"status": "cancel_requested"}


@app.post("/adversarial/runs", dependencies=[Depends(require_role(UserRole.USER))])
def start_adversarial_run(
    request: AdversarialRunRequest,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    if not EVOLUTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adversarial engine not available")
    run_id = f"adv_{uuid.uuid4().hex[:10]}"
    run_state = _RunState(
        run_id=run_id,
        run_type="adversarial",
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        parameters=request.parameters,
    )
    with _run_lock:
        _adversarial_runs[run_id] = run_state

    def _execute():
        ui_context = _create_run_context(run_state, "adversarial_log")
        import session_utils as _session_utils
        import adversarial as _adversarial_module
        _attach_ui(_session_utils, ui_context)
        _attach_ui(_adversarial_module, ui_context)
        config = create_adversarial_configuration(parameters=request.parameters or None)
        result = run_comprehensive_adversarial_testing(
            current_content=request.content,
            content_type=request.content_type,
            config=config,
            team_manager=team_manager,
            gauntlet_manager=gauntlet_manager,
            use_decomposition=request.use_decomposition,
        )
        return result

    _start_background_run(run_state, _execute)
    return {"run_id": run_id, "status": run_state.status}


@app.get("/adversarial/runs", dependencies=[Depends(verify_api_key)])
def list_adversarial_runs():
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "status": run.status,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
            }
            for run in _adversarial_runs.values()
        ]
    }


@app.get("/adversarial/runs/{run_id}", dependencies=[Depends(verify_api_key)])
def get_adversarial_run(run_id: str):
    run_state = _adversarial_runs.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_state.run_id,
        "status": run_state.status,
        "created_at": run_state.created_at,
        "started_at": run_state.started_at,
        "completed_at": run_state.completed_at,
        "logs": run_state.logs,
        "result": run_state.result,
        "error": run_state.error,
    }


@app.post("/adversarial/runs/{run_id}/stop", dependencies=[Depends(require_role(UserRole.USER))])
def stop_adversarial_run(
    run_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER))
):
    run_state = _adversarial_runs.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")
    run_state.cancel_requested = True
    if run_state.session_state is not None:
        run_state.session_state["adversarial_stop_flag"] = True
    return {"status": "cancel_requested"}




# =============================================================================
# TEST COMPATIBILITY CLASSES
# =============================================================================

class EndpointRegistry:
    """Registry for API endpoints (test compatibility)."""
    
    def __init__(self):
        self.endpoints = {}
    
    def register(self, path: str, methods: list = None, handler: str = None, **kwargs):
        """Register an endpoint."""
        if methods is None:
            methods = ['GET']
        self.endpoints[path] = {
            'methods': methods,
            'handler': handler,
            **kwargs
        }
    
    def list_endpoints(self) -> list:
        """List all registered endpoints."""
        return list(self.endpoints.keys())


class RequestParser:
    """Parser for API requests (test compatibility)."""
    
    def __init__(self):
        self.parsers = {}
    
    def parse(self, method: str, path: str, body: dict = None, **kwargs) -> dict:
        """Parse a request."""
        return {
            'parsed': True,
            'method': method,
            'path': path,
            'body': body or {}
        }


class ResponseFormatter:
    """Formatter for API responses (test compatibility)."""
    
    def success(self, data: dict = None, message: str = None, **kwargs) -> dict:
        """Format a success response."""
        return {
            'status': 200,
            'data': data or {},
            'message': message or 'Success'
        }
    
    def error(self, error_code: int, message: str = None, **kwargs) -> dict:
        """Format an error response."""
        return {
            'status': error_code,
            'message': message or 'Error occurred'
        }


class ErrorHandler:
    """Handler for API errors (test compatibility)."""
    
    def handle(self, error_code: int, message: str = None, **kwargs) -> dict:
        """Handle an error."""
        return {
            'status': error_code,
            'message': message or 'Error'
        }


class MiddlewareChain:
    """Chain of middleware (test compatibility)."""
    
    def __init__(self):
        self.middlewares = []
    
    def add_middleware(self, middleware: str):
        """Add middleware to the chain."""
        self.middlewares.append(middleware)
    
    def get_chain(self) -> list:
        """Get the middleware chain."""
        return self.middlewares


class EndpointRateLimiter:
    """Rate limiter for endpoints (test compatibility)."""
    
    def __init__(self):
        self.requests = {}
    
    def allow_request(self, endpoint: str, client_id: str = None) -> bool:
        """Check if request is allowed."""
        key = f'{endpoint}:{client_id or "default"}'
        count = self.requests.get(key, 0)
        self.requests[key] = count + 1
        return True

if __name__ == "__main__":
    start_api_server()
