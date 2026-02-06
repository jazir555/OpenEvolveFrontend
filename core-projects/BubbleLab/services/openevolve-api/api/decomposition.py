"""
Decomposition API Routes for OpenEvolve

Provides problem analysis + decomposition planning.
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..database import get_setting
from ..models import DecompositionDefaults

logger = structlog.get_logger()
router = APIRouter()

# Ensure repo root is on sys.path so decomposition modules can be imported
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from problem_analyzer import ProblemAnalyzer
    from decomposition_engine import DecompositionEngine
except Exception as exc:
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


def _get_decomposition_defaults() -> DecompositionDefaults:
    config_data = get_setting(_DECOMPOSITION_DEFAULTS_KEY)
    if config_data:
        try:
            return DecompositionDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_decomposition_defaults")
    return DecompositionDefaults()


def _ensure_decomposition_available() -> None:
    if ProblemAnalyzer is None or DecompositionEngine is None:
        raise HTTPException(status_code=503, detail="Decomposition engine is not available")


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
        "problem": problem.to_dict(),
        "plan": plan.to_dict(),
    }
