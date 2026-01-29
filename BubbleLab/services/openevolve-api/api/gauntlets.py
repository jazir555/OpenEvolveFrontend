"""
Gauntlet API Routes for OpenEvolve

Gauntlet management for solution validation.
Follows CLAUDE.md principles: structured logging, CRUD operations.
"""

import structlog
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from ..models import GauntletCreate, GauntletUpdate, GauntletResponse, GauntletListResponse, GauntletRound


logger = structlog.get_logger()
router = APIRouter()

# In-memory storage (TODO: Replace with persistent storage)
_gauntlets: dict[str, GauntletResponse] = {}


def _ensure_round_ids(rounds: list[GauntletRound]) -> list[GauntletRound]:
    hydrated: list[GauntletRound] = []
    for round_def in rounds:
        round_id = round_def.id or f"round_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        hydrated.append(round_def.model_copy(update={"id": round_id}))
    return hydrated


@router.post("", response_model=GauntletResponse, status_code=status.HTTP_201_CREATED)
async def create_gauntlet(gauntlet_data: GauntletCreate) -> GauntletResponse:
    """Create a new gauntlet."""
    try:
        gauntlet_id = f"gauntlet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now(timezone.utc)

        gauntlet = GauntletResponse(
            id=gauntlet_id,
            name=gauntlet_data.name,
            description=gauntlet_data.description,
            rounds=_ensure_round_ids(gauntlet_data.rounds),
            created_at=now,
            updated_at=now,
            user_id="anonymous",
            tenant_id="default",
        )

        _gauntlets[gauntlet_id] = gauntlet

        logger.info(
            "gauntlet_created",
            gauntlet_id=gauntlet_id,
            name=gauntlet_data.name,
            rounds_count=len(gauntlet_data.rounds)
        )

        return gauntlet

    except Exception as e:
        logger.error(
            "gauntlet_creation_failed",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create gauntlet"
        )


@router.get("", response_model=GauntletListResponse)
async def list_gauntlets() -> GauntletListResponse:
    """List all gauntlets."""
    try:
        gauntlets = list(_gauntlets.values())

        logger.debug(
            "gauntlets_listed",
            total=len(gauntlets)
        )

        return GauntletListResponse(
            gauntlets=gauntlets,
            total=len(gauntlets)
        )

    except Exception as e:
        logger.error(
            "gauntlet_listing_failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list gauntlets"
        )


@router.get("/{gauntlet_id}", response_model=GauntletResponse)
async def get_gauntlet(gauntlet_id: str) -> GauntletResponse:
    """Get a specific gauntlet by ID."""
    try:
        if gauntlet_id not in _gauntlets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        return _gauntlets[gauntlet_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "gauntlet_retrieval_failed",
            gauntlet_id=gauntlet_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gauntlet"
        )


@router.put("/{gauntlet_id}", response_model=GauntletResponse)
async def update_gauntlet(gauntlet_id: str, gauntlet_data: GauntletUpdate) -> GauntletResponse:
    """Update a gauntlet."""
    try:
        if gauntlet_id not in _gauntlets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        existing = _gauntlets[gauntlet_id]
        update_data = gauntlet_data.dict(exclude_unset=True)

        if "rounds" in update_data and update_data["rounds"] is not None:
            update_data["rounds"] = _ensure_round_ids(update_data["rounds"])

        for field, value in update_data.items():
            setattr(existing, field, value)

        existing.updated_at = datetime.now(timezone.utc)

        logger.info(
            "gauntlet_updated",
            gauntlet_id=gauntlet_id,
            updated_fields=list(update_data.keys())
        )

        return existing

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "gauntlet_update_failed",
            gauntlet_id=gauntlet_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update gauntlet"
        )


@router.delete("/{gauntlet_id}", status_code=status.HTTP_200_OK)
async def delete_gauntlet(gauntlet_id: str) -> dict:
    """Delete a gauntlet."""
    try:
        if gauntlet_id not in _gauntlets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        gauntlet_name = _gauntlets[gauntlet_id].name
        del _gauntlets[gauntlet_id]

        logger.info(
            "gauntlet_deleted",
            gauntlet_id=gauntlet_id,
            name=gauntlet_name
        )

        return {"message": f"Gauntlet '{gauntlet_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "gauntlet_delete_failed",
            gauntlet_id=gauntlet_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete gauntlet"
        )
