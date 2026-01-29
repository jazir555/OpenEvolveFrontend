"""
Settings API Routes for OpenEvolve

Provides endpoints for managing LLM configuration.
"""

import structlog
from fastapi import APIRouter, HTTPException, status

from ..models import LLMConfig, UpdateLLMConfig

logger = structlog.get_logger()
router = APIRouter()

# In-memory settings (TODO: replace with persistent storage)
_llm_config = LLMConfig()


@router.get("/llm", response_model=LLMConfig)
async def get_llm_config() -> LLMConfig:
    """Get current LLM configuration."""
    logger.debug("llm_config_retrieved")
    return _llm_config


@router.put("/llm", response_model=LLMConfig)
async def update_llm_config(update: UpdateLLMConfig) -> LLMConfig:
    """Update LLM configuration."""
    global _llm_config
    try:
        update_data = update.dict(exclude_unset=True)
        _llm_config = _llm_config.model_copy(update=update_data)

        logger.info(
            "llm_config_updated",
            updated_fields=list(update_data.keys())
        )

        return _llm_config
    except Exception as e:
        logger.error("llm_config_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update LLM configuration"
        )
