"""
MDAP/MAKER + ROMA-MDAP-MAKER API Routes for OpenEvolve

Exposes recomposition and ROMA-MDAP-MAKER pipelines to BubbleLab.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from typing import Any, Dict, Optional, List, Callable

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..database import get_setting
from ..models import (
    LLMConfig,
    MDAPMakerDefaults,
    ROMAMDAPMakerDefaults,
)

logger = structlog.get_logger()
router = APIRouter()

# Ensure repo root is on sys.path for shared modules
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from mdap_maker_associative_integration import (
        MakerRecomposerWorkflow,
        MDAP_AVAILABLE,
        MAKER_AVAILABLE,
        ASSOCIATIVE_AVAILABLE,
        GROUND_TRUTH_AVAILABLE,
    )
except Exception as exc:
    logger.warning("mdap_maker_modules_unavailable", error=str(exc))
    MakerRecomposerWorkflow = None
    MDAP_AVAILABLE = False
    MAKER_AVAILABLE = False
    ASSOCIATIVE_AVAILABLE = False
    GROUND_TRUTH_AVAILABLE = False

try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeConfig,
        solve_with_romamdapmaker_associative,
        get_romamdapmaker_associative_status,
    )
except Exception as exc:
    logger.warning("roma_mdap_maker_modules_unavailable", error=str(exc))
    ROMAMDAPMakerAssociativeConfig = None
    solve_with_romamdapmaker_associative = None
    get_romamdapmaker_associative_status = None

try:
    from llm_utils import _compose_messages, _request_openai_compatible_chat
except Exception as exc:
    logger.warning("llm_utils_unavailable", error=str(exc))
    _compose_messages = None
    _request_openai_compatible_chat = None

_LLM_CONFIG_KEY = "llm_config"
_MDAP_MAKER_DEFAULTS_KEY = "mdap_maker_defaults"
_ROMA_MDAP_MAKER_DEFAULTS_KEY = "roma_mdap_maker_defaults"


class MDAPMakerSolveRequest(BaseModel):
    problem_statement: str = Field(..., min_length=1)
    sub_solutions: Dict[str, Any] = Field(default_factory=dict)
    conflicts: List[Any] = Field(default_factory=list)
    use_mdap: Optional[bool] = None
    use_associative: Optional[bool] = None
    num_mdap_agents: Optional[int] = Field(default=None, ge=1, le=20)
    llm_config: Optional[Dict[str, Any]] = None


class ROMAMDAPMakerSolveRequest(BaseModel):
    problem_statement: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None
    config_overrides: Optional[Dict[str, Any]] = None
    recursive: Optional[bool] = None


def _get_llm_config() -> LLMConfig:
    config_data = get_setting(_LLM_CONFIG_KEY)
    if config_data:
        try:
            return LLMConfig(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_llm_config")
    return LLMConfig()


def _get_mdap_maker_defaults() -> MDAPMakerDefaults:
    config_data = get_setting(_MDAP_MAKER_DEFAULTS_KEY)
    if config_data:
        try:
            return MDAPMakerDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_mdap_maker_defaults")
    return MDAPMakerDefaults()


def _get_roma_mdap_maker_defaults() -> ROMAMDAPMakerDefaults:
    config_data = get_setting(_ROMA_MDAP_MAKER_DEFAULTS_KEY)
    if config_data:
        try:
            return ROMAMDAPMakerDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_roma_mdap_maker_defaults")
    return ROMAMDAPMakerDefaults()


def _merge_llm_config(base: LLMConfig, overrides: Optional[Dict[str, Any]]) -> LLMConfig:
    if not overrides:
        return base
    data = base.model_dump()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    return LLMConfig(**data)


def _build_llm_call(llm_config: LLMConfig) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        api_key = llm_config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return json.dumps({
                "error": "Missing API key",
                "mock": True,
                "content": "No API key configured for MDAP/MAKER LLM call."
            })

        if _compose_messages is None or _request_openai_compatible_chat is None:
            return json.dumps({
                "error": "LLM utilities unavailable",
                "mock": True,
                "content": "llm_utils not available in this environment."
            })

        base_url = llm_config.base_url or "https://api.openai.com/v1"
        model = llm_config.model_text or llm_config.model_leanaide
        messages = _compose_messages(
            "You are a helpful assistant that performs structured validation.",
            prompt
        )

        try:
            response = _request_openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                max_tokens=llm_config.max_tokens,
                frequency_penalty=llm_config.frequency_penalty,
                presence_penalty=llm_config.presence_penalty,
            )
            return response or ""
        except Exception as exc:
            logger.error("mdap_maker_llm_call_failed", error=str(exc))
            return json.dumps({
                "error": str(exc),
                "mock": True,
                "content": "LLM call failed."
            })

    return _call


def _filter_dataclass_config(config: Dict[str, Any], config_cls: Any) -> Dict[str, Any]:
    if not config_cls or not hasattr(config_cls, "__dataclass_fields__"):
        return config
    allowed = set(config_cls.__dataclass_fields__.keys())
    return {key: value for key, value in config.items() if key in allowed}


@router.get("/mdap-maker/status")
async def mdap_maker_status() -> Dict[str, Any]:
    return {
        "mdap_available": MDAP_AVAILABLE,
        "maker_available": MAKER_AVAILABLE,
        "associative_available": ASSOCIATIVE_AVAILABLE,
        "ground_truth_available": GROUND_TRUTH_AVAILABLE,
        "full_system_available": MDAP_AVAILABLE and MAKER_AVAILABLE and ASSOCIATIVE_AVAILABLE,
    }


@router.post("/mdap-maker/solve")
async def mdap_maker_solve(request: MDAPMakerSolveRequest) -> Dict[str, Any]:
    if MakerRecomposerWorkflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MDAP/MAKER integration is not available"
        )

    defaults = _get_mdap_maker_defaults()
    use_mdap = request.use_mdap if request.use_mdap is not None else defaults.use_mdap
    use_associative = (
        request.use_associative
        if request.use_associative is not None
        else defaults.use_associative
    )
    num_mdap_agents = request.num_mdap_agents or defaults.num_mdap_agents

    llm_config = _merge_llm_config(
        _get_llm_config(),
        {**(defaults.llm_config or {}), **(request.llm_config or {})}
    )
    llm_call_fn = _build_llm_call(llm_config)

    workflow = MakerRecomposerWorkflow(
        use_mdap=use_mdap,
        use_associative=use_associative,
        num_mdap_agents=num_mdap_agents
    )

    try:
        result = workflow.run_full_workflow(
            problem_statement=request.problem_statement,
            sub_solutions=request.sub_solutions,
            conflicts=request.conflicts,
            llm_call_fn=llm_call_fn,
            mdap_agent_llm_calls=None
        )
    except Exception as exc:
        logger.error("mdap_maker_run_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MDAP/MAKER workflow failed"
        )

    return {
        "success": True,
        "result": result,
    }


@router.get("/roma-mdap-maker/status")
async def roma_mdap_maker_status() -> Dict[str, Any]:
    if get_romamdapmaker_associative_status:
        return get_romamdapmaker_associative_status()
    return {
        "roma_mdap_maker_available": False,
        "associative_available": False,
        "ground_truth_available": False,
        "full_system_available": False,
        "components": {},
    }


@router.post("/roma-mdap-maker/solve")
async def roma_mdap_maker_solve(request: ROMAMDAPMakerSolveRequest) -> Dict[str, Any]:
    if solve_with_romamdapmaker_associative is None or ROMAMDAPMakerAssociativeConfig is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ROMA-MDAP-MAKER integration is not available"
        )

    defaults = _get_roma_mdap_maker_defaults()
    recursive = request.recursive if request.recursive is not None else defaults.recursive
    config_overrides = {**(defaults.config or {}), **(request.config_overrides or {})}

    llm_config = _get_llm_config()
    if not config_overrides.get("api_key"):
        config_overrides["api_key"] = llm_config.api_key or os.environ.get("OPENAI_API_KEY")
    if not config_overrides.get("model"):
        config_overrides["model"] = llm_config.model_text or llm_config.model_leanaide
    if not config_overrides.get("temperature"):
        config_overrides["temperature"] = llm_config.temperature
    if not config_overrides.get("provider"):
        config_overrides["provider"] = llm_config.provider

    filtered_config = _filter_dataclass_config(config_overrides, ROMAMDAPMakerAssociativeConfig)
    config = ROMAMDAPMakerAssociativeConfig(**filtered_config)

    try:
        result = solve_with_romamdapmaker_associative(
            problem=request.problem_statement,
            context=request.context,
            config=config,
            llm_call_fn=None,
            recursive=recursive
        )
    except Exception as exc:
        logger.error("roma_mdap_maker_run_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ROMA-MDAP-MAKER workflow failed"
        )

    return {
        "success": True,
        "result": result,
    }
