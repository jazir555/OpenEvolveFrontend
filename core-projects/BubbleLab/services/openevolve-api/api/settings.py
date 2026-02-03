"""
Settings API Routes for OpenEvolve

Provides endpoints for managing LLM configuration.
"""

import structlog
from fastapi import APIRouter, HTTPException, status

from ..models import (
    LLMConfig,
    UpdateLLMConfig,
    ICRConfig,
    UpdateICRConfig,
    DeterminismDefaults,
    UpdateDeterminismDefaults,
    DecompositionDefaults,
    UpdateDecompositionDefaults,
    AdaptiveDecompositionDefaults,
    UpdateAdaptiveDecompositionDefaults,
    MDAPMakerDefaults,
    UpdateMDAPMakerDefaults,
    ROMAMDAPMakerDefaults,
    UpdateROMAMDAPMakerDefaults,
)
from ..database import get_setting, set_setting

logger = structlog.get_logger()
router = APIRouter()

# Default configuration
_DEFAULT_LLM_CONFIG = LLMConfig()
_DEFAULT_ICR_CONFIG = ICRConfig()
_DEFAULT_DETERMINISM_DEFAULTS = DeterminismDefaults()
_DEFAULT_DECOMPOSITION_DEFAULTS = DecompositionDefaults()
_DEFAULT_ADAPTIVE_DECOMPOSITION_DEFAULTS = AdaptiveDecompositionDefaults()
_DEFAULT_MDAP_MAKER_DEFAULTS = MDAPMakerDefaults()
_DEFAULT_ROMA_MDAP_MAKER_DEFAULTS = ROMAMDAPMakerDefaults()

# Settings key
_LLM_CONFIG_KEY = "llm_config"
_ICR_CONFIG_KEY = "icr_config"
_DETERMINISM_DEFAULTS_KEY = "determinism_defaults"
_DECOMPOSITION_DEFAULTS_KEY = "decomposition_defaults"
_ADAPTIVE_DECOMPOSITION_DEFAULTS_KEY = "adaptive_decomposition_defaults"
_MDAP_MAKER_DEFAULTS_KEY = "mdap_maker_defaults"
_ROMA_MDAP_MAKER_DEFAULTS_KEY = "roma_mdap_maker_defaults"


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


def _get_icr_config() -> ICRConfig:
    """Get ICR config from persistent storage or default."""
    config_data = get_setting(_ICR_CONFIG_KEY)
    if config_data:
        try:
            return ICRConfig(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_icr_config")
    return _DEFAULT_ICR_CONFIG


def _save_icr_config(config: ICRConfig) -> None:
    """Save ICR config to persistent storage."""
    set_setting(_ICR_CONFIG_KEY, config.model_dump())


def _get_determinism_defaults() -> DeterminismDefaults:
    """Get determinism defaults from persistent storage or default."""
    config_data = get_setting(_DETERMINISM_DEFAULTS_KEY)
    if config_data:
        try:
            return DeterminismDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_determinism_defaults")
    return _DEFAULT_DETERMINISM_DEFAULTS


def _save_determinism_defaults(config: DeterminismDefaults) -> None:
    """Save determinism defaults to persistent storage."""
    set_setting(_DETERMINISM_DEFAULTS_KEY, config.model_dump())


def _get_decomposition_defaults() -> DecompositionDefaults:
    """Get decomposition defaults from persistent storage or default."""
    config_data = get_setting(_DECOMPOSITION_DEFAULTS_KEY)
    if config_data:
        try:
            return DecompositionDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_decomposition_defaults")
    return _DEFAULT_DECOMPOSITION_DEFAULTS


def _save_decomposition_defaults(config: DecompositionDefaults) -> None:
    """Save decomposition defaults to persistent storage."""
    set_setting(_DECOMPOSITION_DEFAULTS_KEY, config.model_dump())


def _get_adaptive_decomposition_defaults() -> AdaptiveDecompositionDefaults:
    """Get adaptive decomposition defaults from persistent storage or default."""
    config_data = get_setting(_ADAPTIVE_DECOMPOSITION_DEFAULTS_KEY)
    if config_data:
        try:
            return AdaptiveDecompositionDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_adaptive_decomposition_defaults")
    return _DEFAULT_ADAPTIVE_DECOMPOSITION_DEFAULTS


def _save_adaptive_decomposition_defaults(config: AdaptiveDecompositionDefaults) -> None:
    """Save adaptive decomposition defaults to persistent storage."""
    set_setting(_ADAPTIVE_DECOMPOSITION_DEFAULTS_KEY, config.model_dump())


def _get_mdap_maker_defaults() -> MDAPMakerDefaults:
    """Get MDAP/MAKER defaults from persistent storage or default."""
    config_data = get_setting(_MDAP_MAKER_DEFAULTS_KEY)
    if config_data:
        try:
            return MDAPMakerDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_mdap_maker_defaults")
    return _DEFAULT_MDAP_MAKER_DEFAULTS


def _save_mdap_maker_defaults(config: MDAPMakerDefaults) -> None:
    """Save MDAP/MAKER defaults to persistent storage."""
    set_setting(_MDAP_MAKER_DEFAULTS_KEY, config.model_dump())


def _get_roma_mdap_maker_defaults() -> ROMAMDAPMakerDefaults:
    """Get ROMA-MDAP-MAKER defaults from persistent storage or default."""
    config_data = get_setting(_ROMA_MDAP_MAKER_DEFAULTS_KEY)
    if config_data:
        try:
            return ROMAMDAPMakerDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_roma_mdap_maker_defaults")
    return _DEFAULT_ROMA_MDAP_MAKER_DEFAULTS


def _save_roma_mdap_maker_defaults(config: ROMAMDAPMakerDefaults) -> None:
    """Save ROMA-MDAP-MAKER defaults to persistent storage."""
    set_setting(_ROMA_MDAP_MAKER_DEFAULTS_KEY, config.model_dump())


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


@router.get("/icr", response_model=ICRConfig)
async def get_icr_config() -> ICRConfig:
    """Get current ICR configuration."""
    config = _get_icr_config()
    logger.debug("icr_config_retrieved")
    return config


@router.put("/icr", response_model=ICRConfig)
async def update_icr_config(update: UpdateICRConfig) -> ICRConfig:
    """Update ICR configuration."""
    try:
        current = _get_icr_config()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_icr_config(updated)

        logger.info(
            "icr_config_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("icr_config_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update ICR configuration"
        )


@router.get("/determinism", response_model=DeterminismDefaults)
async def get_determinism_defaults() -> DeterminismDefaults:
    """Get determinism defaults."""
    config = _get_determinism_defaults()
    logger.debug("determinism_defaults_retrieved")
    return config


@router.put("/determinism", response_model=DeterminismDefaults)
async def update_determinism_defaults(update: UpdateDeterminismDefaults) -> DeterminismDefaults:
    """Update determinism defaults."""
    try:
        current = _get_determinism_defaults()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_determinism_defaults(updated)

        logger.info(
            "determinism_defaults_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("determinism_defaults_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update determinism defaults"
        )


@router.get("/decomposition", response_model=DecompositionDefaults)
async def get_decomposition_defaults() -> DecompositionDefaults:
    """Get decomposition defaults."""
    config = _get_decomposition_defaults()
    logger.debug("decomposition_defaults_retrieved")
    return config


@router.put("/decomposition", response_model=DecompositionDefaults)
async def update_decomposition_defaults(update: UpdateDecompositionDefaults) -> DecompositionDefaults:
    """Update decomposition defaults."""
    try:
        current = _get_decomposition_defaults()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_decomposition_defaults(updated)

        logger.info(
            "decomposition_defaults_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("decomposition_defaults_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update decomposition defaults"
        )


@router.get("/adaptive-decomposition", response_model=AdaptiveDecompositionDefaults)
async def get_adaptive_decomposition_defaults() -> AdaptiveDecompositionDefaults:
    """Get adaptive decomposition defaults."""
    config = _get_adaptive_decomposition_defaults()
    logger.debug("adaptive_decomposition_defaults_retrieved")
    return config


@router.put("/adaptive-decomposition", response_model=AdaptiveDecompositionDefaults)
async def update_adaptive_decomposition_defaults(
    update: UpdateAdaptiveDecompositionDefaults
) -> AdaptiveDecompositionDefaults:
    """Update adaptive decomposition defaults."""
    try:
        current = _get_adaptive_decomposition_defaults()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_adaptive_decomposition_defaults(updated)

        logger.info(
            "adaptive_decomposition_defaults_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("adaptive_decomposition_defaults_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update adaptive decomposition defaults"
        )


@router.get("/mdap-maker", response_model=MDAPMakerDefaults)
async def get_mdap_maker_defaults() -> MDAPMakerDefaults:
    """Get MDAP/MAKER defaults."""
    config = _get_mdap_maker_defaults()
    logger.debug("mdap_maker_defaults_retrieved")
    return config


@router.put("/mdap-maker", response_model=MDAPMakerDefaults)
async def update_mdap_maker_defaults(update: UpdateMDAPMakerDefaults) -> MDAPMakerDefaults:
    """Update MDAP/MAKER defaults."""
    try:
        current = _get_mdap_maker_defaults()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_mdap_maker_defaults(updated)

        logger.info(
            "mdap_maker_defaults_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("mdap_maker_defaults_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update MDAP/MAKER defaults"
        )


@router.get("/roma-mdap-maker", response_model=ROMAMDAPMakerDefaults)
async def get_roma_mdap_maker_defaults() -> ROMAMDAPMakerDefaults:
    """Get ROMA-MDAP-MAKER defaults."""
    config = _get_roma_mdap_maker_defaults()
    logger.debug("roma_mdap_maker_defaults_retrieved")
    return config


@router.put("/roma-mdap-maker", response_model=ROMAMDAPMakerDefaults)
async def update_roma_mdap_maker_defaults(
    update: UpdateROMAMDAPMakerDefaults
) -> ROMAMDAPMakerDefaults:
    """Update ROMA-MDAP-MAKER defaults."""
    try:
        current = _get_roma_mdap_maker_defaults()
        update_data = update.model_dump(exclude_unset=True)
        updated = current.model_copy(update=update_data)

        _save_roma_mdap_maker_defaults(updated)

        logger.info(
            "roma_mdap_maker_defaults_updated",
            updated_fields=list(update_data.keys())
        )

        return updated
    except Exception as e:
        logger.error("roma_mdap_maker_defaults_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update ROMA-MDAP-MAKER defaults"
        )
