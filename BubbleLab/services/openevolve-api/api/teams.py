"""
Team API Routes for OpenEvolve

Team management for AI agent orchestration.
Follows CLAUDE.md principles: structured logging, CRUD operations.
"""

import structlog
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from ..models import TeamCreate, TeamUpdate, TeamResponse, TeamListResponse, TeamMember


logger = structlog.get_logger()
router = APIRouter()

# In-memory storage (TODO: Replace with persistent storage)
_teams: dict[str, TeamResponse] = {}


def _ensure_member_ids(members: list[TeamMember]) -> list[TeamMember]:
    """Ensure each team member has a stable ID."""
    hydrated: list[TeamMember] = []
    for member in members:
        member_id = member.id or f"member_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        hydrated.append(member.model_copy(update={"id": member_id}))
    return hydrated


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

        _teams[team_id] = team

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
        teams = list(_teams.values())

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
async def get_team(team_id: str) -> TeamResponse:
    """Get a specific team by ID."""
    try:
        if team_id not in _teams:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        return _teams[team_id]

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
        if team_id not in _teams:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        existing = _teams[team_id]
        update_data = team_data.dict(exclude_unset=True)

        if "members" in update_data and update_data["members"] is not None:
            update_data["members"] = _ensure_member_ids(update_data["members"])

        for field, value in update_data.items():
            setattr(existing, field, value)

        existing.updated_at = datetime.now(timezone.utc)

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
        if team_id not in _teams:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{team_id}' not found"
            )

        team_name = _teams[team_id].name
        del _teams[team_id]

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
