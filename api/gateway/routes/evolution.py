"""
Evolution Engine Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from models.schemas import (
    EvolutionStart,
    EvolutionStatus,
    EvolutionListResponse,
    EvolutionListItem,
)
from middleware.auth import get_current_user
from utils.responses import success, paginated
from utils.errors import NotFoundError, ValidationError
from realtime.manager import EvolutionRoomManager
from typing import Optional
import os
import logging
from datetime import datetime
import uuid
import httpx

router = APIRouter(prefix="/evolution", tags=["Evolution"])
logger = logging.getLogger(__name__)

# Evolution room manager for WebSocket updates
evolution_room_manager = EvolutionRoomManager()

# Mock evolution storage (replace with database in production)
evolutions_db = {}


def get_owned_evolution(evolution_id: str, current_user: dict) -> dict:
    evolution = evolutions_db.get(evolution_id)
    if not evolution or evolution.get("user_id") != current_user["user_id"]:
        raise NotFoundError(resource="Evolution")
    return evolution


async def run_evolution_background(evolution_id: str, evolution_data: EvolutionStart):
    """
    Background task to run evolution
    This would call the actual evolution engine
    """
    try:
        orchestrator_url = os.getenv(
            "EVOLUTION_ORCHESTRATOR_URL",
            "http://localhost:8003/evolve",
        )

        # Kick off an orchestrator run
        max_iterations = evolution_data.parameters.max_iterations
        survival_threshold = evolution_data.parameters.survival_threshold
        criteria = None
        if evolution_data.constraints:
            criteria = evolution_data.constraints.get("criteria") or evolution_data.constraints.get("goal")

        payload = {
            "html": evolution_data.content,
            "iterations": min(max_iterations, 20),
            "populationSize": evolution_data.parameters.population_size,
            "criteria": criteria,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(orchestrator_url, json=payload)
            response.raise_for_status()
            orchestrator_result = response.json()

        root_node_id = f"{evolution_id}-seed"
        await evolution_room_manager.broadcast_descendant_created(
            evolution_id,
            {
                "node_id": root_node_id,
                "parent_id": None,
                "generation": 0,
                "status": "survived",
                "fitness": 1.0,
                "label": "Seed",
                "metadata": {
                    "mode": evolution_data.mode,
                    "constraints": evolution_data.constraints or {},
                },
            },
        )

        history = orchestrator_result.get("history", [])
        evolutions_db[evolution_id]["history"] = history
        for i, item in enumerate(history[1:], start=1):
            progress = (i / max(len(history) - 1, 1)) * 100
            await evolution_room_manager.broadcast_progress(
                evolution_id,
                progress,
                f"Generation {i}/{len(history) - 1}",
            )

            parent_id = root_node_id
            node_id = f"{evolution_id}-g{i}-{uuid.uuid4().hex[:8]}"
            fitness = item.get("score", 0)
            status = "survived" if fitness >= survival_threshold else "killed"

            await evolution_room_manager.broadcast_descendant_created(
                evolution_id,
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "generation": i,
                    "status": "evaluating",
                    "fitness": None,
                    "label": f"Variant {i}",
                    "html": item.get("html"),
                    "metadata": {
                        "changes": item.get("changes", []),
                    },
                },
            )

            await evolution_room_manager.broadcast_descendant_status(
                evolution_id,
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "generation": i,
                    "status": status,
                    "fitness": fitness,
                },
            )

            evolutions_db[evolution_id]["current_iteration"] = i
            evolutions_db[evolution_id]["best_fitness"] = max(
                evolutions_db[evolution_id].get("best_fitness") or 0.0,
                fitness,
            )

        # Mark as complete
        evolutions_db[evolution_id]["status"] = "completed"
        evolutions_db[evolution_id]["updated_at"] = datetime.utcnow()

        await evolution_room_manager.broadcast_complete(
            evolution_id,
            {"best_fitness": 0.9},
        )

    except Exception as e:
        logger.error(f"Evolution {evolution_id} failed: {e}")
        evolutions_db[evolution_id]["status"] = "error"

        await evolution_room_manager.broadcast_error(
            evolution_id,
            str(e),
        )


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_evolution(
    evolution_data: EvolutionStart,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Start evolutionary optimization

    Args:
        evolution_data: Evolution configuration
        background_tasks: FastAPI background tasks
        current_user: Authenticated user

    Returns:
        dict: Evolution ID and WebSocket URL
    """
    # Generate evolution ID
    evolution_id = str(uuid.uuid4())

    # Store evolution in database
    evolutions_db[evolution_id] = {
        "evolution_id": evolution_id,
        "user_id": current_user["user_id"],
        "status": "running",
        "mode": evolution_data.mode,
        "content": evolution_data.content,
        "parameters": evolution_data.parameters.dict(),
        "constraints": evolution_data.constraints or {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "current_iteration": 0,
        "best_fitness": None,
        "history": [],
    }

    # Start evolution in background
    background_tasks.add_task(run_evolution_background, evolution_id, evolution_data)

    logger.info(f"Evolution started: {evolution_id} by user {current_user['user_id']}")

    return {
        "evolution_id": evolution_id,
        "status": "running",
        "created_at": datetime.utcnow().isoformat(),
        "websocket_url": f"ws://localhost:8000/ws/evolution/{evolution_id}",
    }


@router.get("/{evolution_id}", response_model=EvolutionStatus)
async def get_evolution_status(
    evolution_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """
    Get evolution status and results

    Args:
        evolution_id: Evolution ID
        current_user: Authenticated user

    Returns:
        EvolutionStatus: Evolution status
    """
    evolution = get_owned_evolution(evolution_id, current_user)
    history = evolution.get("history", [])
    paginated_history = history[offset : offset + limit]

    return EvolutionStatus(
        evolution_id=evolution_id,
        status=evolution["status"],
        progress={
            "current_iteration": evolution["current_iteration"],
            "max_iterations": evolution["parameters"]["max_iterations"],
            "percentage": (evolution["current_iteration"] / evolution["parameters"]["max_iterations"]) * 100,
            "history_total": len(history),
            "history_limit": limit,
            "history_offset": offset,
        },
        population=paginated_history,
        best_individual={"fitness": evolution.get("best_fitness", 0.0)},
        metrics={"average_fitness": 0.7, "diversity_score": 0.6},
        started_at=evolution["created_at"],
        updated_at=evolution["updated_at"],
    )


@router.post("/{evolution_id}/pause")
async def pause_evolution(
    evolution_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Pause running evolution

    Args:
        evolution_id: Evolution ID
        current_user: Authenticated user

    Returns:
        dict: Updated status
    """
    evolution = get_owned_evolution(evolution_id, current_user)

    if evolution["status"] != "running":
        raise ValidationError(message="Evolution is not running")

    evolution["status"] = "paused"
    evolution["updated_at"] = datetime.utcnow()

    logger.info(f"Evolution paused: {evolution_id}")

    return {
        "evolution_id": evolution_id,
        "status": "paused",
        "paused_at": datetime.utcnow().isoformat(),
    }


@router.post("/{evolution_id}/resume")
async def resume_evolution(
    evolution_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Resume paused evolution

    Args:
        evolution_id: Evolution ID
        background_tasks: FastAPI background tasks
        current_user: Authenticated user

    Returns:
        dict: Updated status
    """
    evolution = get_owned_evolution(evolution_id, current_user)

    if evolution["status"] != "paused":
        raise ValidationError(message="Evolution is not paused")

    # Restart evolution
    evolution["status"] = "running"
    evolution["updated_at"] = datetime.utcnow()

    # Would restart background task here
    logger.info(f"Evolution resumed: {evolution_id}")

    return {
        "evolution_id": evolution_id,
        "status": "running",
        "resumed_at": datetime.utcnow().isoformat(),
    }


@router.post("/{evolution_id}/stop")
async def stop_evolution(
    evolution_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Stop evolution execution

    Args:
        evolution_id: Evolution ID
        current_user: Authenticated user

    Returns:
        dict: Final status
    """
    evolution = get_owned_evolution(evolution_id, current_user)

    evolution["status"] = "stopped"
    evolution["updated_at"] = datetime.utcnow()

    logger.info(f"Evolution stopped: {evolution_id}")

    return {
        "evolution_id": evolution_id,
        "status": "stopped",
        "stopped_at": datetime.utcnow().isoformat(),
        "final_results": {
            "best_fitness": evolution.get("best_fitness", 0.0),
            "iterations_completed": evolution["current_iteration"],
        },
    }


@router.delete("/{evolution_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evolution(
    evolution_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete evolution and associated data

    Args:
        evolution_id: Evolution ID
        current_user: Authenticated user
    """
    evolution = get_owned_evolution(evolution_id, current_user)

    del evolutions_db[evolution_id]

    logger.info(f"Evolution deleted: {evolution_id}")

    return None


@router.get("", response_model=EvolutionListResponse)
async def list_evolutions(
    status_filter: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """
    List all evolutions for current user

    Args:
        status_filter: Optional status filter
        limit: Number of results
        offset: Pagination offset
        current_user: Authenticated user

    Returns:
        EvolutionListResponse: List of evolutions
    """
    # Filter evolutions by user
    user_evolutions = [
        e for e in evolutions_db.values() if e["user_id"] == current_user["user_id"]
    ]

    # Filter by status if specified
    if status_filter:
        user_evolutions = [e for e in user_evolutions if e["status"] == status_filter]

    # Apply pagination
    total = len(user_evolutions)
    paginated_evolutions = user_evolutions[offset : offset + limit]

    evolution_items = [
        EvolutionListItem(
            evolution_id=e["evolution_id"],
            status=e["status"],
            mode=e["mode"],
            created_at=e["created_at"],
            updated_at=e["updated_at"],
            best_fitness=e.get("best_fitness"),
            iterations_completed=e["current_iteration"],
        )
        for e in paginated_evolutions
    ]

    return EvolutionListResponse(
        evolutions=evolution_items,
        total=total,
        limit=limit,
        offset=offset,
    )
