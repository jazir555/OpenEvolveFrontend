"""
Determinism API Routes for OpenEvolve

Provides deterministic generation and reproducibility checks.
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..database import get_setting
from ..models import DeterminismDefaults

logger = structlog.get_logger()
router = APIRouter()

# Ensure repo root is on sys.path so determinism_stack can be imported
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from determinism_stack import (
        LLMConfig,
        build_llm,
        DeterminismConfig,
        DeterministicPipeline,
        HybridDeterministicSystem,
    )
except Exception as exc:
    logger.warning("determinism_stack_unavailable", error=str(exc))
    LLMConfig = None
    build_llm = None
    DeterminismConfig = None
    DeterministicPipeline = None
    HybridDeterministicSystem = None


class DeterminismGenerateRequest(BaseModel):
    prompt: str
    schema: Optional[Dict[str, Any]] = None
    constraints: Optional[str] = None
    context_document: Optional[str] = None
    mode: Optional[str] = None  # auto | cloud | local | hybrid | consensus
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_base_url: Optional[str] = None
    local_provider: Optional[str] = None
    local_model: Optional[str] = None
    local_device: Optional[str] = None
    local_dtype: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None


class DeterminismCheckRequest(BaseModel):
    prompt: str
    tier: Optional[int] = None
    runs: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None


_DETERMINISM_DEFAULTS_KEY = "determinism_defaults"


def _get_determinism_defaults() -> DeterminismDefaults:
    config_data = get_setting(_DETERMINISM_DEFAULTS_KEY)
    if config_data:
        try:
            return DeterminismDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_determinism_defaults")
    return DeterminismDefaults()


def _ensure_stack_available() -> None:
    if DeterministicPipeline is None or DeterminismConfig is None or build_llm is None:
        raise HTTPException(status_code=503, detail="Determinism stack is not available")


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


@router.post("/generate")
async def determinism_generate(req: DeterminismGenerateRequest):
    _ensure_stack_available()
    defaults = _get_determinism_defaults()
    mode = req.mode or defaults.mode
    cloud_provider = req.cloud_provider or defaults.cloud_provider
    cloud_model = req.cloud_model or defaults.cloud_model
    cloud_base_url = req.cloud_base_url or defaults.cloud_base_url
    local_provider = req.local_provider or defaults.local_provider
    local_model = req.local_model or defaults.local_model
    local_device = req.local_device or defaults.local_device
    local_dtype = req.local_dtype or defaults.local_dtype
    detllm_backend = req.detllm_backend or defaults.detllm_backend
    detllm_model = req.detllm_model or defaults.detllm_model
    merged_config = None
    if defaults.config or req.config:
        merged_config = {**(defaults.config or {}), **(req.config or {})}

    config = _build_config(merged_config, detllm_backend, detllm_model)

    if mode in {"hybrid", "consensus"}:
        cloud_llm = _build_llm(cloud_provider, cloud_model, req.cloud_api_key, cloud_base_url)
        local_llm = _build_llm(local_provider, local_model, None, None, local_device, local_dtype)
        if cloud_llm is None or local_llm is None:
            raise HTTPException(status_code=400, detail="Hybrid mode requires both cloud and local LLM configs")
        system = HybridDeterministicSystem(cloud_llm=cloud_llm, local_llm=local_llm)
        result = system.generate(req.prompt, mode=mode)
        return result.__dict__

    if mode == "cloud":
        llm = _build_llm(cloud_provider, cloud_model, req.cloud_api_key, cloud_base_url)
    elif mode == "local":
        llm = _build_llm(local_provider, local_model, None, None, local_device, local_dtype)
    else:
        llm = _build_llm(cloud_provider, cloud_model, req.cloud_api_key, cloud_base_url) or _build_llm(
            local_provider, local_model, None, None, local_device, local_dtype
        )

    pipeline = DeterministicPipeline(llm=llm, config=config)
    result = pipeline.generate_with_all_layers(
        req.prompt,
        schema=req.schema,
        constraints=req.constraints,
        context_document=req.context_document,
    )
    return result.__dict__


@router.post("/check")
async def determinism_check(req: DeterminismCheckRequest):
    _ensure_stack_available()
    defaults = _get_determinism_defaults()
    tier = req.tier if req.tier is not None else defaults.check_tier
    runs = req.runs if req.runs is not None else defaults.check_runs
    provider = req.provider or defaults.check_provider
    model = req.model or defaults.check_model
    base_url = req.base_url or defaults.check_base_url
    device = req.device or defaults.check_device
    dtype = req.dtype or defaults.check_dtype
    detllm_backend = req.detllm_backend or defaults.detllm_backend
    detllm_model = req.detllm_model or defaults.detllm_model

    llm = _build_llm(provider, model, req.api_key, base_url, device, dtype)
    pipeline = DeterministicPipeline(llm=llm, config=_build_config(None, detllm_backend, detllm_model))
    result = pipeline.reproducibility.check(
        prompt=req.prompt,
        llm=llm,
        tier=tier,
        runs=runs,
        backend=detllm_backend,
        model=detllm_model,
    )
    return result
