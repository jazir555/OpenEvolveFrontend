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



# ============================================================================
# GAUNTLET EXECUTION ENDPOINTS
# ============================================================================

# In-memory execution tracking (in production, use Redis or database)
_gauntlet_executions: dict[str, dict] = {}


@router.post("/{gauntlet_name}/execute", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def execute_gauntlet(gauntlet_name: str, payload: dict) -> dict:
    """
    Execute a gauntlet with the given content.

    This endpoint starts an asynchronous gauntlet execution and returns
    an execution_id for tracking progress.
    """
    try:
        # Find gauntlet by name
        gauntlets_data = get_all_gauntlets()
        gauntlet_data = None
        for g in gauntlets_data:
            if g.get("name") == gauntlet_name:
                gauntlet_data = g
                break

        if not gauntlet_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gauntlet '{gauntlet_name}' not found"
            )

        gauntlet = _gauntlet_from_db(gauntlet_data)

        # Create execution record
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        execution = {
            "execution_id": execution_id,
            "gauntlet_name": gauntlet_name,
            "gauntlet_id": gauntlet.id,
            "status": "started",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "content": payload.get("content", ""),
            "content_type": payload.get("content_type", "text_general"),
            "evolution_mode": payload.get("evolution_mode", "standard"),
            "parameters": payload.get("parameters", {}),
            "current_round": 0,
            "rounds_completed": [],
            "final_result": None
        }

        _gauntlet_executions[execution_id] = execution

        logger.info(
            "gauntlet_execution_started",
            execution_id=execution_id,
            gauntlet_name=gauntlet_name,
            content_length=len(payload.get("content", ""))
        )

        # In production, this would trigger an async task
        # For now, we'll mark it as running immediately
        execution["status"] = "running"

        return {
            "run_id": execution_id,  # For compatibility with evolution runs
            "execution_id": execution_id,
            "status": "started",
            "gauntlet_name": gauntlet_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "gauntlet_execution_start_failed",
            gauntlet_name=gauntlet_name,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start gauntlet execution: {str(e)}"
        )


@router.get("/executions/{execution_id}/status", response_model=dict)
async def get_gauntlet_execution_status(execution_id: str) -> dict:
    """Get the status of a gauntlet execution."""
    try:
        if execution_id not in _gauntlet_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        execution = _gauntlet_executions[execution_id]

        logger.debug(
            "gauntlet_execution_status_retrieved",
            execution_id=execution_id,
            status=execution["status"]
        )

        return {
            "run_id": execution_id,
            "execution_id": execution_id,
            "status": execution["status"],
            "gauntlet_name": execution["gauntlet_name"],
            "current_round": execution["current_round"],
            "rounds_completed": execution["rounds_completed"],
            "created_at": execution["created_at"],
            "updated_at": execution["updated_at"],
            "result": execution.get("final_result")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "gauntlet_execution_status_failed",
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get execution status"
        )


@router.get("/executions", response_model=dict)
async def list_gauntlet_executions(gauntlet_name: str = None) -> dict:
    """List all gauntlet executions, optionally filtered by gauntlet name."""
    try:
        executions = list(_gauntlet_executions.values())

        if gauntlet_name:
            executions = [e for e in executions if e["gauntlet_name"] == gauntlet_name]

        logger.debug(
            "gauntlet_executions_listed",
            total=len(executions),
            gauntlet_filter=gauntlet_name
        )

        return {
            "executions": executions,
            "total": len(executions)
        }

    except Exception as e:
        logger.error(
            "gauntlet_executions_listing_failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list executions"
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


