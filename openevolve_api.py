"""
OpenEvolve API Endpoints for CREWAI Integration

This module provides API endpoints for managing OpenEvolve teams and gauntlets
within the CREWAI framework.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import uuid
import logging

from openevolve_structures import (
    ModelConfig,
    Team,
    GauntletDefinition,
    GauntletRoundRule
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenEvolve CREWAI Integration API",
    description="API for managing OpenEvolve teams and gauntlets within CREWAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
team_manager = TeamManager()
gauntlet_manager = GauntletManager()

# Pydantic models for API requests/responses
class ModelConfigRequest(BaseModel):
    model_id: str
    api_key: str = ""  # For local setup
    api_base: str = "http://localhost:8001"
    temperature: float = 0.7
    max_tokens: int = 4096

class TeamCreateRequest(BaseModel):
    name: str
    role: str
    members: List[ModelConfigRequest]
    description: Optional[str] = None

class GauntletCreateRequest(BaseModel):
    name: str
    team_name: str
    rounds: List[Dict[str, Any]]
    description: Optional[str] = None

class TeamUpdateRequest(BaseModel):
    name: str
    role: str
    members: List[ModelConfigRequest]
    description: Optional[str] = None

class GauntletUpdateRequest(BaseModel):
    name: str
    team_name: str
    rounds: List[Dict[str, Any]]
    description: Optional[str] = None

# API Endpoints

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "OpenEvolve CREWAI Integration API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Team endpoints

@app.post("/openevolve/teams", response_model=dict)
def create_team(request: TeamCreateRequest):
    """Create a new team."""
    try:
        # Convert ModelConfigRequest to ModelConfig objects
        members = [
            ModelConfig(
                model_id=mc.model_id,
                api_key=mc.api_key,
                api_base=mc.api_base,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens
            )
            for mc in request.members
        ]
        
        team = Team(
            name=request.name,
            role=request.role,
            members=members,
            description=request.description
        )
        
        success = team_manager.create_team(team)
        if not success:
            raise HTTPException(status_code=400, detail="Team with this name already exists")
        
        logger.info(f"Team '{team.name}' created successfully")
        
        return {"message": "Team created", "team_name": team.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (TypeError, KeyError, RuntimeError) as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openevolve/teams", response_model=List[Dict[str, Any]])
def list_teams():
    """List all teams."""
    teams = team_manager.get_all_teams()
    return [
        {
            "name": team.name,
            "role": team.role,
            "description": team.description,
            "member_count": len(team.members)
        }
        for team in teams
    ]

@app.get("/openevolve/teams/{team_name}", response_model=Dict[str, Any])
def get_team(team_name: str):
    """Get team details."""
    team = team_manager.get_team(team_name)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return {
        "name": team.name,
        "role": team.role,
        "description": team.description,
        "members": [
            {
                "model_id": m.model_id,
                "temperature": m.temperature,
                "max_tokens": m.max_tokens
            }
            for m in team.members
        ]
    }

@app.put("/openevolve/teams/{team_name}", response_model=dict)
def update_team(team_name: str, request: TeamUpdateRequest):
    """Update an existing team."""
    team = team_manager.get_team(team_name)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    try:
        # Convert ModelConfigRequest to ModelConfig objects
        members = [
            ModelConfig(
                model_id=mc.model_id,
                api_key=mc.api_key,
                api_base=mc.api_base,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens
            )
            for mc in request.members
        ]
        
        updated_team = Team(
            name=team_name,
            role=request.role,
            members=members,
            description=request.description
        )
        
        success = team_manager.update_team(updated_team)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update team")
        
        logger.info(f"Team '{team_name}' updated successfully")
        
        return {"message": "Team updated", "team_name": team_name}
    except (TypeError, KeyError, RuntimeError) as e:
        logger.error(f"Error updating team '{team_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/openevolve/teams/{team_name}", response_model=dict)
def delete_team(team_name: str):
    """Delete a team."""
    success = team_manager.delete_team(team_name)
    if not success:
        raise HTTPException(status_code=404, detail="Team not found")
    
    logger.info(f"Team '{team_name}' deleted successfully")
    
    return {"message": "Team deleted", "team_name": team_name}

# Gauntlet endpoints

@app.post("/openevolve/gauntlets", response_model=dict)
def create_gauntlet(request: GauntletCreateRequest):
    """Create a new gauntlet."""
    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        gauntlet = GauntletDefinition(
            name=request.name,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        success = gauntlet_manager.create_gauntlet(gauntlet)
        if not success:
            raise HTTPException(status_code=400, detail="Gauntlet with this name already exists")
        
        logger.info(f"Gauntlet '{gauntlet.name}' created successfully")
        
        return {"message": "Gauntlet created", "gauntlet_name": gauntlet.name}
    except (TypeError, KeyError, RuntimeError) as e:
        logger.error(f"Error creating gauntlet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openevolve/gauntlets", response_model=List[Dict[str, Any]])
def list_gauntlets():
    """List all gauntlets."""
    gauntlets = gauntlet_manager.get_all_gauntlets()
    return [
        {
            "name": g.name,
            "team_name": g.team_name,
            "description": g.description,
            "round_count": len(g.rounds)
        }
        for g in gauntlets
    ]

@app.get("/openevolve/gauntlets/{gauntlet_name}", response_model=Dict[str, Any])
def get_gauntlet(gauntlet_name: str):
    """Get gauntlet details."""
    gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)
    if not gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    return {
        "name": gauntlet.name,
        "team_name": gauntlet.team_name,
        "description": gauntlet.description,
        "rounds": [
            {
                "round_number": r.round_number,
                "quorum_required_approvals": r.quorum_required_approvals,
                "quorum_from_panel_size": r.quorum_from_panel_size,
                "min_overall_confidence": r.min_overall_confidence
            }
            for r in gauntlet.rounds
        ]
    }

@app.put("/openevolve/gauntlets/{gauntlet_name}", response_model=dict)
def update_gauntlet(gauntlet_name: str, request: GauntletUpdateRequest):
    """Update an existing gauntlet."""
    gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)
    if not gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        updated_gauntlet = GauntletDefinition(
            name=gauntlet_name,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        success = gauntlet_manager.update_gauntlet(updated_gauntlet)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update gauntlet")
        
        logger.info(f"Gauntlet '{gauntlet_name}' updated successfully")
        
        return {"message": "Gauntlet updated", "gauntlet_name": gauntlet_name}
    except (TypeError, KeyError, RuntimeError) as e:
        logger.error(f"Error updating gauntlet '{gauntlet_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/openevolve/gauntlets/{gauntlet_name}", response_model=dict)
def delete_gauntlet(gauntlet_name: str):
    """Delete a gauntlet."""
    success = gauntlet_manager.delete_gauntlet(gauntlet_name)
    if not success:
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    logger.info(f"Gauntlet '{gauntlet_name}' deleted successfully")
    
    return {"message": "Gauntlet deleted", "gauntlet_name": gauntlet_name}

def start_api_server(host: str = "0.0.0.0", port: int = 8002):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server()