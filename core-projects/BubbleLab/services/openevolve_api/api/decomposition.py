"""
Decomposition API Routes for OpenEvolve

Provides problem analysis + decomposition planning.
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..database import get_setting
from ..models import DecompositionDefaults

logger = structlog.get_logger()
router = APIRouter()

# Ensure the decomposition modules (which live at <repo_root>/engines/other)
# are importable. The module file is located in:
#   core-projects/BubbleLab/services/openevolve-api/api/decomposition.py
# so parents[5] is the repository root (OpenEvolveFrontend) and the engines
# package lives at <repo_root>/engines/other.
_SERVICE_DIR = Path(__file__).resolve().parents[1]  # .../services/openevolve-api
_REPO_ROOT = Path(__file__).resolve().parents[5]    # OpenEvolveFrontend
_ENGINES_OTHER = _REPO_ROOT / "engines" / "other"
# Source tree of the openevolve library (its ``openevolve`` package lives here).
_OPEVOLVE_SRC = _REPO_ROOT / "core-projects" / "openevolve"

# Prepend (highest priority) so these modules shadow any other same-named
# modules on the path.
for _p in (str(_ENGINES_OTHER), str(_OPEVOLVE_SRC), str(_REPO_ROOT), str(_SERVICE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from problem_analyzer import ProblemAnalyzer
    from decomposition_engine import DecompositionEngine
except Exception as exc:  # pragma: no cover - depends on engine source tree
    logger.warning("decomposition_modules_unavailable", error=str(exc))
    ProblemAnalyzer = None
    DecompositionEngine = None

try:
    from decomposition_mcp_tools import (
        get_mcp_tool_inventory,
        web3_ingest_contract_audit_stack,
    )
    WEB3_INGESTION_AVAILABLE = True
except Exception:
    WEB3_INGESTION_AVAILABLE = False
    get_mcp_tool_inventory = None
    web3_ingest_contract_audit_stack = None

_WEB3_DOMAIN_ALIASES = {
    "web3": "web3",
    "defi": "web3",
    "smart_contract": "web3",
    "smart contract": "web3",
    "smart_contract_audit": "web3",
    "solidity": "web3",
}

_WEB3_INGESTION_TOOL_NAMES = {
    "web3_ingest_slither_static_analysis",
    "web3_ingest_foundry_fuzzing",
    "web3_ingest_contract_audit_stack",
}
_WEB3_FORMAL_TOOL_NAMES = {
    "z3_translate_solidity_invariant",
    "z3_solve_smart_contract_exploit_witness",
    "z3_web3_audit_exploit_verification",
}


class DecompositionPlanRequest(BaseModel):
    problem_statement: str
    title: Optional[str] = None
    strategy: Optional[str] = None
    domain: Optional[str] = None
    domain_hint: Optional[str] = None
    domain_artifacts: Optional[Dict[str, Any]] = None
    web3_ingestion_enabled: Optional[bool] = None
    web3_project_path: Optional[str] = None
    web3_run_fuzzing: Optional[bool] = None
    web3: Optional[Dict[str, Any]] = None
    enable_adaptive_selection: Optional[bool] = None
    maker_config: Optional[Dict[str, Any]] = None
    openevolve_client_config: Optional[Dict[str, Any]] = None


_DECOMPOSITION_DEFAULTS_KEY = "decomposition_defaults"


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Recursively convert a value into something JSON-serializable.

    The decomposition engine may attach live tool objects / callables (e.g. from
    ``get_mcp_tool_inventory``) into the analysis metadata. FastAPI cannot
    serialize those, so we downgrade callables and other exotic types to
    descriptive strings instead of crashing the endpoint with a 500.
    """
    if _depth > 50:
        return str(value)
    if callable(value) and not isinstance(value, type):
        return f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth + 1) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(), _depth + 1)
        except Exception:
            pass
    return str(value)


def _get_decomposition_defaults() -> DecompositionDefaults:
    config_data = get_setting(_DECOMPOSITION_DEFAULTS_KEY)
    if config_data:
        try:
            return DecompositionDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_decomposition_defaults")
    return DecompositionDefaults()


def _ensure_decomposition_available() -> None:
    """Verify the decomposition engine modules imported and are usable.

    Returns HTTP 501 (not implemented) with a clear message when the engine
    modules are genuinely unavailable, instead of failing later with a 500 or
    silently returning empty analysis.
    """
    missing = []
    if not callable(ProblemAnalyzer):
        missing.append("problem_analyzer.ProblemAnalyzer")
    if not callable(DecompositionEngine):
        missing.append("decomposition_engine.DecompositionEngine")
    if missing:
        logger.warning(
            "decomposition_engine_unavailable",
            missing=missing,
            engines_dir=str(_ENGINES_OTHER),
        )
        raise HTTPException(
            status_code=501,
            detail=(
                "Decomposition engine is not available. Missing: "
                + ", ".join(missing)
            ),
        )


def _normalize_domain_hint(domain_hint: Optional[str]) -> Optional[str]:
    if not domain_hint or not isinstance(domain_hint, str):
        return None
    normalized = domain_hint.strip().lower().replace("-", "_")
    if not normalized:
        return None
    return _WEB3_DOMAIN_ALIASES.get(normalized, normalized)


def _normalize_web3_tool_inventory(raw_inventory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize Web3 tool inventory to a consistent shape."""
    inventory = dict(raw_inventory or {})
    web3_tools = list(inventory.get("web3_tools", []) or [])
    web3_ingestion_tools = list(inventory.get("web3_ingestion_tools", []) or [])
    web3_formal_tools = list(inventory.get("web3_formal_tools", []) or [])

    if not web3_ingestion_tools:
        web3_ingestion_tools = sorted(
            tool for tool in web3_tools if tool in _WEB3_INGESTION_TOOL_NAMES
        )
    if not web3_formal_tools:
        web3_formal_tools = sorted(
            tool for tool in web3_tools if tool in _WEB3_FORMAL_TOOL_NAMES
        )

    formal_capabilities = {
        "solidity_invariant_translation": "z3_translate_solidity_invariant" in web3_formal_tools,
        "invariant_translation_verification": "z3_translate_solidity_invariant" in web3_formal_tools,
        "symbolic_exploit_witness": "z3_solve_smart_contract_exploit_witness" in web3_formal_tools,
        "composite_exploit_verification": "z3_web3_audit_exploit_verification" in web3_formal_tools,
    }
    existing_capabilities = inventory.get("formal_capabilities")
    if isinstance(existing_capabilities, dict):
        formal_capabilities.update(existing_capabilities)

    if not web3_formal_tools:
        if formal_capabilities.get("solidity_invariant_translation"):
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities.get("symbolic_exploit_witness"):
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities.get("composite_exploit_verification"):
            web3_formal_tools.append("z3_web3_audit_exploit_verification")
        web3_formal_tools = sorted(set(web3_formal_tools))

    merged_web3_tools = sorted(set(web3_tools + web3_ingestion_tools + web3_formal_tools))
    web3_formal_available = bool(web3_formal_tools) or any(
        bool(value) for value in formal_capabilities.values()
    )

    inventory.update(
        {
            "web3_tools": merged_web3_tools,
            "web3_ingestion_tools": web3_ingestion_tools,
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "web3_formal_available": web3_formal_available,
            "audit_exploit_verification_available": bool(
                formal_capabilities.get("composite_exploit_verification")
            ),
        }
    )
    return inventory


@router.post("/plan")
async def create_decomposition_plan(request: DecompositionPlanRequest):
    _ensure_decomposition_available()

    defaults = _get_decomposition_defaults()
    strategy = request.strategy or defaults.strategy
    enable_adaptive_selection = (
        request.enable_adaptive_selection
        if request.enable_adaptive_selection is not None
        else defaults.enable_adaptive_selection
    )
    maker_config = {**(defaults.maker_config or {}), **(request.maker_config or {})}
    client_config = {
        **(defaults.openevolve_client_config or {}),
        **(request.openevolve_client_config or {}),
    }
    domain_hint = _normalize_domain_hint(
        request.domain_hint or request.domain or defaults.default_domain_hint
    )
    domain_artifacts: Dict[str, Any] = {
        **(defaults.default_domain_artifacts or {}),
        **(request.domain_artifacts or {}),
    }
    web3_config: Dict[str, Any] = {
        "enabled": bool(defaults.web3_ingestion_enabled),
        "project_path": defaults.web3_project_path or ".",
        "run_fuzzing": bool(defaults.web3_run_fuzzing),
    }
    if isinstance(request.web3, dict):
        web3_config.update(request.web3)
    if request.web3_ingestion_enabled is not None:
        web3_config["enabled"] = bool(request.web3_ingestion_enabled)
    if request.web3_project_path:
        web3_config["project_path"] = request.web3_project_path
    if request.web3_run_fuzzing is not None:
        web3_config["run_fuzzing"] = bool(request.web3_run_fuzzing)
    if domain_hint == "web3":
        web3_config["enabled"] = True

    mcp_tool_inventory: Dict[str, Any] = {}
    if get_mcp_tool_inventory is not None:
        try:
            mcp_tool_inventory = _normalize_web3_tool_inventory(
                get_mcp_tool_inventory() or {}
            )
        except Exception as exc:
            logger.warning("failed_to_collect_mcp_tool_inventory", error=str(exc))
            mcp_tool_inventory = _normalize_web3_tool_inventory({})
    if mcp_tool_inventory:
        domain_artifacts.setdefault("mcp_tool_inventory", mcp_tool_inventory)

    web3_ingestion = None
    if bool(web3_config.get("enabled")):
        if WEB3_INGESTION_AVAILABLE and web3_ingest_contract_audit_stack is not None:
            try:
                web3_ingestion = web3_ingest_contract_audit_stack(
                    project_path=str(web3_config.get("project_path", ".")),
                    run_fuzzing=bool(web3_config.get("run_fuzzing", True)),
                    slither_timeout_seconds=int(web3_config.get("slither_timeout_seconds", 240)),
                    forge_timeout_seconds=int(web3_config.get("forge_timeout_seconds", 420)),
                )
            except Exception as exc:
                logger.warning("web3_ingestion_failed", error=str(exc))
                web3_ingestion = {"success": False, "error": str(exc)}
        else:
            web3_ingestion = {
                "success": False,
                "error": "web3_ingestion_tools_unavailable",
            }
        domain_artifacts["web3_ingestion"] = web3_ingestion
        if isinstance(web3_ingestion, dict):
            entanglement_matrix = web3_ingestion.get("entanglement_matrix")
            if isinstance(entanglement_matrix, dict):
                domain_artifacts.setdefault("entanglement_matrix", entanglement_matrix)

    analyzer = ProblemAnalyzer(openevolve_client_config=client_config)
    problem = analyzer.analyze_problem(request.problem_statement, title=request.title or "")
    if problem is None:
        raise HTTPException(status_code=500, detail="Problem analysis failed")

    if domain_hint and hasattr(problem, "domain_context") and problem.domain_context is not None:
        problem.domain_context.domain = domain_hint

    if domain_artifacts and hasattr(problem, "domain_context") and problem.domain_context is not None:
        domain_knowledge = getattr(problem.domain_context, "domain_knowledge", None)
        if not isinstance(domain_knowledge, dict):
            domain_knowledge = {}
            problem.domain_context.domain_knowledge = domain_knowledge
        domain_knowledge["domain_artifacts"] = domain_artifacts

    if hasattr(problem, "metadata"):
        if not isinstance(problem.metadata, dict):
            problem.metadata = {}
        if domain_hint:
            problem.metadata["domain_hint"] = domain_hint
        if domain_artifacts:
            problem.metadata["domain_artifacts"] = domain_artifacts
        if bool(web3_config.get("enabled")):
            problem.metadata["web3"] = web3_config

    engine = DecompositionEngine(
        problem_analyzer=analyzer,
        enable_adaptive_selection=enable_adaptive_selection,
        maker_config=maker_config,
    )

    try:
        plan = engine.decompose(problem, strategy=strategy)
    except Exception as exc:
        logger.error("decomposition_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Decomposition failed")

    existing_plan_metadata = plan.metadata if isinstance(plan.metadata, dict) else {}
    plan.metadata = {
        **existing_plan_metadata,
        "problem_statement": request.problem_statement,
        "domain": domain_hint or getattr(problem.domain_context, "domain", None) or "general",
        "domain_hint": domain_hint,
        "domain_artifacts": domain_artifacts,
        "web3": web3_config if bool(web3_config.get("enabled")) else {},
    }
    if web3_ingestion is not None:
        plan.metadata["web3_ingestion"] = web3_ingestion
    entanglement_matrix = domain_artifacts.get("entanglement_matrix", {})
    if isinstance(entanglement_matrix, dict) and entanglement_matrix:
        plan.metadata.setdefault("entanglement_matrix", entanglement_matrix)

    return {
        "problem": _json_safe(problem.to_dict()),
        "plan": _json_safe(plan.to_dict()),
    }


# ============================================================================
# DECOMPOSITION EXECUTION ENDPOINTS
# ============================================================================

# In-memory execution tracking (in production, use Redis or database)
_decomposition_executions: dict[str, dict] = {}


@router.post("/workflows/{workflow_id}/execute-decomposition", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def execute_decomposition(workflow_id: str, payload: dict) -> dict:
    """
    Execute a decomposition plan for the given workflow.

    This endpoint starts an asynchronous decomposition execution and returns
    an execution_id for tracking progress.
    """
    try:
        # Create execution record
        execution_id = f"decomp_exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        execution = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "started",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "problem_statement": payload.get("problem_statement", ""),
            "content_type": payload.get("content_type", "text_general"),
            "decomposition_method": payload.get("decomposition_method", "hierarchical"),
            "granularity": payload.get("granularity", "medium"),
            "max_depth": payload.get("max_depth", 3),
            "max_sub_problems": payload.get("max_sub_problems", 5),
            "parameters": payload.get("parameters", {}),
            "sub_problems_completed": 0,
            "sub_problems_total": 0,
            "current_sub_problem": None,
            "results": None
        }

        _decomposition_executions[execution_id] = execution

        logger.info(
            "decomposition_execution_started",
            execution_id=execution_id,
            workflow_id=workflow_id,
            decomposition_method=payload.get("decomposition_method")
        )

        # In production, this would trigger an async task
        execution["status"] = "running"

        return {
            "execution_id": execution_id,
            "status": "started",
            "workflow_id": workflow_id
        }

    except Exception as e:
        logger.error(
            "decomposition_execution_start_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start decomposition execution: {str(e)}"
        )


@router.get("/decomposition/executions/{execution_id}/status", response_model=dict)
async def get_decomposition_execution_status(execution_id: str) -> dict:
    """Get the status of a decomposition execution."""
    try:
        if execution_id not in _decomposition_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        execution = _decomposition_executions[execution_id]

        logger.debug(
            "decomposition_execution_status_retrieved",
            execution_id=execution_id,
            status=execution["status"]
        )

        return {
            "execution_id": execution_id,
            "status": execution["status"],
            "workflow_id": execution["workflow_id"],
            "sub_problems_completed": execution["sub_problems_completed"],
            "sub_problems_total": execution["sub_problems_total"],
            "current_sub_problem": execution.get("current_sub_problem"),
            "results": execution.get("results"),
            "created_at": execution["created_at"],
            "updated_at": execution["updated_at"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "decomposition_execution_status_failed",
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get execution status"
        )


@router.get("/decomposition/executions", response_model=dict)
async def list_decomposition_executions(workflow_id: str = None) -> dict:
    """List all decomposition executions, optionally filtered by workflow ID."""
    try:
        executions = list(_decomposition_executions.values())

        if workflow_id:
            executions = [e for e in executions if e["workflow_id"] == workflow_id]

        logger.debug(
            "decomposition_executions_listed",
            total=len(executions),
            workflow_filter=workflow_id
        )

        # Return only basic info for listing
        executions_summary = [
            {
                "execution_id": e["execution_id"],
                "status": e["status"],
                "workflow_id": e["workflow_id"],
                "created_at": e["created_at"]
            }
            for e in executions
        ]

        return {
            "executions": executions_summary,
            "total": len(executions)
        }

    except Exception as e:
        logger.error(
            "decomposition_executions_listing_failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list executions"
        )
