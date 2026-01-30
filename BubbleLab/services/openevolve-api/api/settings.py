"""
Settings API Routes for OpenEvolve

Provides endpoints for managing LLM configuration.
"""

import structlog
from fastapi import APIRouter, HTTPException, status

from ..models import LLMConfig, UpdateLLMConfig
from ..database import get_setting, set_setting

logger = structlog.get_logger()
router = APIRouter()

# Default configuration
_DEFAULT_LLM_CONFIG = LLMConfig()

# Settings key
_LLM_CONFIG_KEY = "llm_config"


def _get_llm_config() -> LLMConfig:
    """Get LLM config from persistent storage or default."""
    config_data = get_setting(_LLM_CONFIG_KEY)
    if config_data:
        try:
            return LLMConfig(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_llm_config")
    return _DEFAULT_LLM_CONFIG


def _save_llm_config(config: LLMConfig) -> None:
    """Save LLM config to persistent storage."""
    set_setting(_LLM_CONFIG_KEY, config.model_dump())


@router.get("/llm", response_model=LLMConfig)
async def get_llm_config() -> LLMConfig:
    """Get current LLM configuration."""
    config = _get_llm_config()
    logger.debug("llm_config_retrieved")
    return config


@router.put("/llm", response_model=LLMConfig)
async def update_llm_config(update: UpdateLLMConfig) -> LLMConfig:
    """Update LLM configuration."""
    try:
        # Get current config
        current = _get_llm_config()
        
        # Apply updates
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        # Save to persistent storage
        _save_llm_config(updated)

        logger.info(
            "llm_config_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("llm_config_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update LLM configuration"
        )
