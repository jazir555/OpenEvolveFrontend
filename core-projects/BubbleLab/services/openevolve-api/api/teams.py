"""
Team API Routes for OpenEvolve

Team management for AI agent orchestration.
Follows CLAUDE.md principles: structured logging, CRUD operations.
"""

import os
import structlog
import threading
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from ..models import TeamCreate, TeamUpdate, TeamResponse, TeamListResponse, TeamMember
from ..database import save_team, get_team, get_all_teams, delete_team as db_delete_team


# Optional bridge to the REAL openevolve engine. This is the single engine
# invocation path used by /api/v1/*; the legacy routers keep their DB-backed
# self-contained behavior as the source of truth and only consult the bridge
# when it is both importable and enabled via OPENEVOLVE_BRIDGE_ENABLED. The
# bridge call is best-effort and never alters the response shape or raises.
try:
    from ..core.openevolve_bridge import OpenEvolveBridgeError, run_openevolve_workflow

    OPENEVOLVE_BRIDGE_AVAILABLE = True
    _OPENEVOLVE_IMPORT_ERROR: Optional[str] = None
except ImportError as _bridge_import_error:  # pragma: no cover - env dependent
    OpenEvolveBridgeError = RuntimeError  # type: ignore[assignment,misc]
    run_openevolve_workflow = None  # type: ignore[assignment]
    OPENEVOLVE_BRIDGE_AVAILABLE = False
    _OPENEVOLVE_IMPORT_ERROR = str(_bridge_import_error)


logger = structlog.get_logger()
router = APIRouter()

# Cache for in-memory access (backed by persistent storage)
_teams_cache: dict[str, TeamResponse] = {}

# Latest REAL openevolve engine result per team, populated only when the bridge
# is enabled and a team is created/updated. Never surfaced in the response model.
_teams_openevolve_results: dict[str, dict] = {}


def _bridge_enabled() -> bool:
    """True when the shared engine bridge should be consulted (env-gated)."""
    return os.getenv("OPENEVOLVE_BRIDGE_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _maybe_run_openevolve_bg(request: dict, key: str) -> None:
    """Spawn a best-effort, non-blocking real-engine run.

    The result (or failure) is stashed in ``_teams_openevolve_results[key]``.
    No-op and never raises when the bridge is unavailable/disabled.
    """
    if not (OPENEVOLVE_BRIDGE_AVAILABLE and _bridge_enabled()):
        return

    def _worker() -> None:
        try:
            _teams_openevolve_results[key] = run_openevolve_workflow(request)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("openevolve_bridge_team_run_failed", team_id=key, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()


def _ensure_member_ids(members: list[TeamMember]) -> list[TeamMember]:
    """Ensure each team member has a stable ID."""
    hydrated: list[TeamMember] = []
    for member in members:
        member_id = member.id or f"member_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        hydrated.append(member.model_copy(update={"id": member_id}))
    return hydrated


def _team_from_db(team_data: dict) -> TeamResponse:
    """Convert database dict to TeamResponse model."""
    # Convert member dicts to TeamMember objects
    members = [TeamMember(**m) for m in team_data.get("members", [])]
    
    return TeamResponse(
        id=team_data["id"],
        name=team_data["name"],
        description=team_data.get("description"),
        members=members,
        created_at=datetime.fromisoformat(team_data["created_at"]),
        updated_at=datetime.fromisoformat(team_data["updated_at"]),
        user_id=team_data.get("user_id", "anonymous"),
        tenant_id=team_data.get("tenant_id", "default"),
    )


def _team_to_db(team: TeamResponse) -> dict:
    """Convert TeamResponse model to database dict."""
    return {
        "id": team.id,
        "name": team.name,
        "description": team.description,
        "members": [m.model_dump() for m in team.members],
        "created_at": team.created_at.isoformat(),
        "updated_at": team.updated_at.isoformat(),
        "user_id": team.user_id,
        "tenant_id": team.tenant_id,
    }


@router.post("", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(team_data: TeamCreate) -> TeamResponse:
    """Create a new team."""
    try:
        team_id = f"team_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now(timezone.utc)

        team = TeamResponse(
            id=team_id,
            name=team_data.name,
            description=team_data.description,
            members=_ensure_member_ids(team_data.members),
            created_at=now,
            updated_at=now,
            user_id="anonymous",
            tenant_id="default",
        )

        # Save to persistent storage
        save_team(team_id, _team_to_db(team))
        
        # Update cache
        _teams_cache[team_id] = team

        # Optionally consult the REAL openevolve engine (best-effort, off-thread).
        _maybe_run_openevolve_bg(
            {
                "system": "evolutionary",
                "problem_statement": f"Optimize team composition for: {team_data.name}",
                "context": team_data.description or "",
                "parameters": {},
                "llm": {},
            },
            team_id,
        )

        logger.info(
            "team_created",
            team_id=team_id,
            name=team_data.name,
            members_count=len(team_data.members)
        )

        return team

    except Exception as e:
        logger.error(
            "team_creation_failed",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create team"
        )


@router.get("", response_model=TeamListResponse)
async def list_teams() -> TeamListResponse:
    """List all teams."""
    try:
        # Load from database
        teams_data = get_all_teams()
        teams = [_team_from_db(t) for t in teams_data]
        
        # Update cache
        _teams_cache.clear()
        for team in teams:
            _teams_cache[team.id] = team

        logger.debug(
            "teams_listed",
            total=len(teams)
        )

        return TeamListResponse(
            teams=teams,
            total=len(teams)
        )

    except Exception as e:
        logger.error(
            "team_listing_failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list teams"
        )


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team_by_id(team_id: str) -> TeamResponse:
    """Get a specific team by ID."""
    try:
        # Check cache first
        if team_id in _teams_cache:
            return _teams_cache[team_id]
        
        # Load from database
        team_data = get_team(team_id)
        if not team_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        team = _team_from_db(team_data)
        _teams_cache[team_id] = team
        return team

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "team_retrieval_failed",
            team_id=team_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve team"
        )


@router.put("/{team_id}", response_model=TeamResponse)
async def update_team(team_id: str, team_data: TeamUpdate) -> TeamResponse:
    """Update a team."""
    try:
        # Load from database
        team_data_db = get_team(team_id)
        if not team_data_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        existing = _team_from_db(team_data_db)
        update_data = team_data.model_dump(exclude_unset=True)

        if "members" in update_data and update_data["members"] is not None:
            update_data["members"] = _ensure_member_ids(
                [TeamMember(**m) for m in update_data["members"]]
            )
            update_data["members"] = [m.model_dump() for m in update_data["members"]]
        
        # Update fields
        for field, value in update_data.items():
            if field != "members":  # Handle members separately
                setattr(existing, field, value)
        
        if "members" in update_data:
            existing.members = [TeamMember(**m) for m in update_data["members"]]

        existing.updated_at = datetime.now(timezone.utc)

        # Save to persistent storage
        save_team(team_id, _team_to_db(existing))
        
        # Update cache
        _teams_cache[team_id] = existing

        # Optionally consult the REAL openevolve engine (best-effort, off-thread).
        _maybe_run_openevolve_bg(
            {
                "system": "evolutionary",
                "problem_statement": f"Optimize team composition for: {existing.name}",
                "context": existing.description or "",
                "parameters": {},
                "llm": {},
            },
            team_id,
        )

        logger.info(
            "team_updated",
            team_id=team_id,
            updated_fields=list(update_data.keys())
        )

        return existing

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "team_update_failed",
            team_id=team_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update team"
        )


@router.delete("/{team_id}", status_code=status.HTTP_200_OK)
async def delete_team(team_id: str) -> dict:
    """Delete a team."""
    try:
        # Check if exists
        team_data = get_team(team_id)
        if not team_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        team_name = team_data.get("name", "Unknown")
        
        # Delete from database
        db_delete_team(team_id)
        
        # Remove from cache
        if team_id in _teams_cache:
            del _teams_cache[team_id]

        logger.info(
            "team_deleted",
            team_id=team_id,
            name=team_name
        )

        return {"message": f"Team '{team_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "team_delete_failed",
            team_id=team_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete team"
        )
