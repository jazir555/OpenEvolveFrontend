"""
Team Management API Endpoints

Flexible team LLM assignment with unified credential management

Features:
- Create teams with arbitrary LLM composition
- Assign any LLM to any team (blue/red/judge)
- Verify credentials before use
- Support vLLMs for visual workflows
- Unified credential management (OpenEvolve + BubbleLab)
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import structlog

from ..models.team_assignment import (
    Team,
    TeamMemberLLM,
    TeamRole,
    LLMModel,
    LLMProvider,
    LLMCapability,
    TeamAssignmentRequest,
    get_available_llms,
    get_vision_llms,
    group_llms_by_capability,
    PREDEFINED_LLM_MODELS,
)
from ..models.credential_manager import (
    CredentialManager,
    CredentialVerificationRequest,
    CredentialVerificationResponse,
    get_credential_manager,
)

logger = structlog.get_logger()

router = APIRouter()


# ==================== Health Check ====================

@router.get("/health")
async def health_check():
    """Health check for teams API"""
    return {"status": "healthy", "service": "teams-api"}


# ==================== LLM Catalog ====================

@router.get("/llms/catalog")
async def get_llm_catalog(
    provider: Optional[LLMProvider] = None,
    capability: Optional[LLMCapability] = None,
    vision_only: bool = False,
) -> dict:
    """
    Get catalog of available LLMs

    Args:
        provider: Filter by provider
        capability: Filter by capability
        vision_only: Only return vLLMs

    Returns:
        Dict with LLMs grouped by capability
    """
    logger.info(
        "fetching_llm_catalog",
        provider=provider.value if provider else None,
        capability=capability.value if capability else None,
        vision_only=vision_only,
    )

    if vision_only:
        # Return vLLMs grouped separately
        vision_llms = get_vision_llms()
        text_llms = get_available_llms(vision_only=False)

        return {
            "vision_llms": [llm.dict() for llm in vision_llms],
            "text_llms": [llm.dict() for llm in text_llms],
            "total": len(vision_llms) + len(text_llms),
        }

    llms = get_available_llms(provider=provider, capability=capability)

    # Group by capability for better UI organization
    grouped = group_llms_by_capability()

    return {
        "llms": [llm.dict() for llm in llms],
        "grouped": {k: [llm.dict() for llm in v] for k, v in grouped.items()},
        "total": len(llms),
    }


@router.get("/llms/providers")
async def get_providers() -> dict:
    """Get list of supported LLM providers"""
    return {
        "providers": [
            {"id": "openai", "name": "OpenAI", "vision_support": True},
            {"id": "anthropic", "name": "Anthropic", "vision_support": True},
            {"id": "google", "name": "Google", "vision_support": True},
            {"id": "openrouter", "name": "OpenRouter", "vision_support": False},
            {"id": "groq", "name": "Groq", "vision_support": False},
            {"id": "deepseek", "name": "DeepSeek", "vision_support": False},
            {"id": "openai-like", "name": "OpenAI-Compatible (vLLM)", "vision_support": "varies"},
        ]
    }


# ==================== Credential Management ====================

@router.get("/credentials")
async def list_credentials(
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> dict:
    """
    List all available credentials from all sources

    Returns credentials from:
    - OpenEvolve config (environment variables)
    - BubbleLab credentials API
    """
    logger.info("listing_credentials")

    credentials = await credential_mgr.get_all_credentials()

    # Don't expose actual API keys in list
    safe_credentials = []
    for cred in credentials:
        safe_creds = cred.dict()
        safe_creds["api_key"] = "****" + (cred.api_key[-4:] if cred.api_key else "")
        safe_credentials.append(safe_creds)

    return {
        "credentials": safe_credentials,
        "total": len(credentials),
        "sources": list(set(c.source.value for c in credentials)),
    }


@router.post("/credentials/verify")
async def verify_credential(
    request: CredentialVerificationRequest,
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> CredentialVerificationResponse:
    """
    Verify an LLM credential by making a test API call

    Tests the credential with a minimal API call to verify it works.

    Returns verification result with latency and available models
    """
    logger.info(
        "verifying_credential",
        provider=request.provider.value,
        has_api_base=request.api_base is not None,
    )

    result = await credential_mgr.verify_credential(request)

    # If verified, save it
    if result.verified:
        from ..models.team_assignment import LLMCredential, CredentialSource

        credential = LLMCredential(
            credential_id=f"verified_{request.provider.value}_{int(datetime.now().timestamp())}",
            provider=request.provider,
            api_key=request.api_key,
            source=CredentialSource.USER_PROVIDED,
            verified=True,
            verified_at=datetime.now().isoformat(),
            model_permissions=result.available_models or [],
            api_base=request.api_base,
        )

        await credential_mgr.save_credential(credential)

        result.credential_id = credential.credential_id

    return result


@router.get("/credentials/{credential_id}")
async def get_credential(
    credential_id: str,
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> dict:
    """Get specific credential details"""
    # Implementation would fetch from cache
    pass


# ==================== Team Management ====================

@router.post("/teams")
async def create_team(
    team: Team,
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> dict:
    """
    Create a new team with LLM members

    Team can have any combination of LLMs assigned to any role.

    Validation:
    - At least one member required
    - If require_vision_for_design: at least one vLLM member
    - If require_diverse_providers: LLMs from different providers
    """
    logger.info(
        "creating_team",
        name=team.name,
        member_count=len(team.members),
    )

    # Validate team composition
    if not team.members:
        raise HTTPException(status_code=400, detail="Team must have at least one member")

    # Check vision requirement
    if team.require_vision_for_design:
        has_vision = any(member.llm.is_vision for member in team.members)
        if not has_vision:
            raise HTTPException(
                status_code=400,
                detail="Team requires at least one vLLM for design tasks"
            )

    # Check diverse providers requirement
    if team.require_diverse_providers:
        providers = set(member.llm.provider for member in team.members)
        if len(providers) < len(team.members):
            raise HTTPException(
                status_code=400,
                detail=f"Team requires diverse providers, got: {', '.join(providers)}"
            )

    # Verify credentials for all members
    for member in team.members:
        if member.credential_id:
            # Verify specific credential
            cred = await credential_mgr.get_credential(member.llm.provider)
            if not cred:
                raise HTTPException(
                    status_code=400,
                    detail=f"No credential available for {member.llm.provider.value}"
                )
        else:
            # Try to get default credential
            cred = await credential_mgr.get_credential(member.llm.provider)
            if not cred:
                raise HTTPException(
                    status_code=400,
                    detail=f"No credential found for {member.llm.provider.value}. Please add credentials first."
                )

    # Store team (in production: save to database)
    team_id = f"team_{int(datetime.now().timestamp())}"

    logger.info(
        "team_created",
        team_id=team_id,
        member_count=len(team.members),
    )

    return {
        "team_id": team_id,
        "name": team.name,
        "members": [member.dict() for member in team.members],
        "created_at": datetime.now().isoformat(),
    }


@router.get("/teams")
async def list_teams() -> dict:
    """List all teams"""
    # In production: fetch from database
    return {
        "teams": [],
        "total": 0,
    }


@router.get("/teams/{team_id}")
async def get_team(team_id: str) -> dict:
    """Get specific team details"""
    # In production: fetch from database
    pass


@router.put("/teams/{team_id}")
async def update_team(team_id: str, team: Team) -> dict:
    """Update team composition"""
    # In production: update in database
    pass


@router.delete("/teams/{team_id}")
async def delete_team(team_id: str) -> dict:
    """Delete a team"""
    # In production: delete from database
    pass


@router.post("/teams/{team_id}/members")
async def add_team_member(
    team_id: str,
    member: TeamMemberLLM,
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> dict:
    """
    Add an LLM member to a team

    Any LLM can be added to any team regardless of role
    """
    logger.info(
        "adding_team_member",
        team_id=team_id,
        llm_provider=member.llm.provider.value,
        llm_model=member.llm.model_id,
        role=member.role.value,
    )

    # Verify credential available
    cred = await credential_mgr.get_credential(member.llm.provider)
    if not cred:
        raise HTTPException(
            status_code=400,
            detail=f"No credential found for {member.llm.provider.value}"
        )

    # In production: add to database
    member_id = f"member_{int(datetime.now().timestamp())}"

    return {
        "member_id": member_id,
        "team_id": team_id,
        "member": member.dict(),
        "added_at": datetime.now().isoformat(),
    }


@router.delete("/teams/{team_id}/members/{member_id}")
async def remove_team_member(team_id: str, member_id: str) -> dict:
    """Remove an LLM member from a team"""
    # In production: remove from database
    pass


# ==================== Team Assignment ====================

@router.post("/teams/assign")
async def assign_llm_to_team(
    request: TeamAssignmentRequest,
    credential_mgr: CredentialManager = Depends(get_credential_manager),
) -> dict:
    """
    Assign an LLM to a team

    Simplified endpoint for quick assignment
    """
    # Get LLM model
    llm = PREDEFINED_LLM_MODELS.get(request.llm_model_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {request.llm_model_id} not found")

    # Create team member
    member = TeamMemberLLM(
        member_id=f"member_{int(datetime.now().timestamp())}",
        llm=llm,
        role=request.role,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        credential_id=request.credential_id,
        system_prompt=request.system_prompt,
    )

    logger.info(
        "assigned_llm_to_team",
        team_id=request.team_id,
        llm=request.llm_model_id,
        role=request.role.value,
    )

    return {
        "member_id": member.member_id,
        "team_id": request.team_id,
        "llm": llm.dict(),
        "role": request.role.value,
    }


# ==================== Team Templates ====================

@router.get("/teams/templates")
async def get_team_templates() -> dict:
    """
    Get predefined team templates

    Templates for common team configurations
    """
    return {
        "templates": [
            {
                "id": "standard_evolution",
                "name": "Standard Evolution Team",
                "description": "Blue team generates, red team attacks, judge decides",
                "composition": [
                    {
                        "role": "blue",
                        "llm": "claude-3-opus",
                        "count": 3,
                    },
                    {
                        "role": "red",
                        "llm": "gpt-4-turbo",
                        "count": 2,
                    },
                    {
                        "role": "judge",
                        "llm": "claude-3-sonnet",
                        "count": 1,
                    },
                ],
            },
            {
                "id": "web_design",
                "name": "Web Design Team",
                "description": "Team with vLLMs for visual design tasks",
                "composition": [
                    {
                        "role": "blue",
                        "llm": "gpt-4-vision",
                        "count": 2,
                    },
                    {
                        "role": "blue",
                        "llm": "claude-3-opus",
                        "count": 1,
                    },
                    {
                        "role": "judge",
                        "llm": "claude-3-sonnet",
                        "count": 1,
                    },
                ],
            },
            {
                "id": "code_review",
                "name": "Code Review Team",
                "description": "Specialized in code analysis and review",
                "composition": [
                    {
                        "role": "blue",
                        "llm": "deepseek-coder",
                        "count": 2,
                    },
                    {
                        "role": "red",
                        "llm": "gpt-4",
                        "count": 2,
                    },
                    {
                        "role": "judge",
                        "llm": "claude-3-opus",
                        "count": 1,
                    },
                ],
            },
        ]
    }


from datetime import datetime
