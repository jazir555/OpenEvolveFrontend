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
from ..database import (
    save_gauntlet, get_gauntlet, get_all_gauntlets, 
    delete_gauntlet as db_delete_gauntlet
)


logger = structlog.get_logger()
router = APIRouter()

# Cache for in-memory access (backed by persistent storage)
_gauntlets_cache: dict[str, GauntletResponse] = {}


def _ensure_round_ids(rounds: list[GauntletRound]) -> list[GauntletRound]:
    hydrated: list[GauntletRound] = []
    for round_def in rounds:
        round_id = round_def.id or f"round_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        hydrated.append(round_def.model_copy(update={"id": round_id}))
    return hydrated


def _gauntlet_from_db(gauntlet_data: dict) -> GauntletResponse:
    """Convert database dict to GauntletResponse model."""
    # Convert round dicts to GauntletRound objects
    rounds = [GauntletRound(**r) for r in gauntlet_data.get("rounds", [])]
    
    return GauntletResponse(
        id=gauntlet_data["id"],
        name=gauntlet_data["name"],
        description=gauntlet_data.get("description"),
        rounds=rounds,
        created_at=datetime.fromisoformat(gauntlet_data["created_at"]),
        updated_at=datetime.fromisoformat(gauntlet_data["updated_at"]),
        user_id=gauntlet_data.get("user_id", "anonymous"),
        tenant_id=gauntlet_data.get("tenant_id", "default"),
    )


def _gauntlet_to_db(gauntlet: GauntletResponse) -> dict:
    """Convert GauntletResponse model to database dict."""
    return {
        "id": gauntlet.id,
        "name": gauntlet.name,
        "description": gauntlet.description,
        "rounds": [r.model_dump() for r in gauntlet.rounds],
        "created_at": gauntlet.created_at.isoformat(),
        "updated_at": gauntlet.updated_at.isoformat(),
        "user_id": gauntlet.user_id,
        "tenant_id": gauntlet.tenant_id,
    }


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

        # Save to persistent storage
        save_gauntlet(gauntlet_id, _gauntlet_to_db(gauntlet))
        
        # Update cache
        _gauntlets_cache[gauntlet_id] = gauntlet

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
        # Load from database
        gauntlets_data = get_all_gauntlets()
        gauntlets = [_gauntlet_from_db(g) for g in gauntlets_data]
        
        # Update cache
        _gauntlets_cache.clear()
        for gauntlet in gauntlets:
            _gauntlets_cache[gauntlet.id] = gauntlet

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
async def get_gauntlet_by_id(gauntlet_id: str) -> GauntletResponse:
    """Get a specific gauntlet by ID."""
    try:
        # Check cache first
        if gauntlet_id in _gauntlets_cache:
            return _gauntlets_cache[gauntlet_id]
        
        # Load from database
        gauntlet_data = get_gauntlet(gauntlet_id)
        if not gauntlet_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        gauntlet = _gauntlet_from_db(gauntlet_data)
        _gauntlets_cache[gauntlet_id] = gauntlet
        return gauntlet

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
        # Load from database
        gauntlet_data_db = get_gauntlet(gauntlet_id)
        if not gauntlet_data_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        existing = _gauntlet_from_db(gauntlet_data_db)
        update_data = gauntlet_data.model_dump(exclude_unset=True)

        if "rounds" in update_data and update_data["rounds"] is not None:
            update_data["rounds"] = _ensure_round_ids(
                [GauntletRound(**r) for r in update_data["rounds"]]
            )
            update_data["rounds"] = [r.model_dump() for r in update_data["rounds"]]

        # Update fields
        for field, value in update_data.items():
            if field != "rounds":  # Handle rounds separately
                setattr(existing, field, value)
        
        if "rounds" in update_data:
            existing.rounds = [GauntletRound(**r) for r in update_data["rounds"]]

        existing.updated_at = datetime.now(timezone.utc)

        # Save to persistent storage
        save_gauntlet(gauntlet_id, _gauntlet_to_db(existing))
        
        # Update cache
        _gauntlets_cache[gauntlet_id] = existing

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
        # Check if exists
        gauntlet_data = get_gauntlet(gauntlet_id)
        if not gauntlet_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_id}' not found"
            )

        gauntlet_name = gauntlet_data.get("name", "Unknown")
        
        # Delete from database
        db_delete_gauntlet(gauntlet_id)
        
        # Remove from cache
        if gauntlet_id in _gauntlets_cache:
            del _gauntlets_cache[gauntlet_id]

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
