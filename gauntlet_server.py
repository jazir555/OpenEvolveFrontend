
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import yaml
import os

# Assuming workflow_engine.py, gauntlet_manager.py, and other necessary files are in the same directory
# or accessible via PYTHONPATH.
from workflow_engine import run_gauntlet_headless
from gauntlet_manager import GauntletManager
from team_manager import TeamManager
from workflow_structures import GauntletDefinition, Team, CritiqueReport, VerificationReport

class GauntletRequest(BaseModel):
    solution_content: str
    gauntlet_name: str
    context: Dict[str, Any]

app = FastAPI()

# In a real app, you'd initialize these properly.
# For now, let's assume they can be loaded on-demand or are singletons.
def get_gauntlet_manager():
    if not os.path.exists('gauntlets.json'):
        raise HTTPException(status_code=500, detail="gauntlets.json not found")
    return GauntletManager('gauntlets.json')

def get_team_manager():
    if not os.path.exists('teams.json'):
        # Let's try to find a default or example teams file.
        # This is a fallback for demonstration.
        if os.path.exists('teams.example.json'):
            return TeamManager('teams.example.json')
        raise HTTPException(status_code=500, detail="teams.json not found")
    return TeamManager('teams.json')


@app.get("/docs")
def read_docs():
    """
    Returns the OpenAPI documentation for the Gauntlet Runner Service.
    """

    return app.openapi()

# Adaptive MDAP not available
ADAPTIVE_MDAP_AVAILABLE = False

@app.post("/run_gauntlet")
def handle_run_gauntlet(request: GauntletRequest):
    """
    Runs a specified gauntlet on the provided solution content.
    """
    try:
        gauntlet_manager = get_gauntlet_manager()
        team_manager = get_team_manager()

        gauntlet_def = gauntlet_manager.get_gauntlet(request.gauntlet_name)
        if not gauntlet_def:
            raise HTTPException(status_code=404, detail=f"Gauntlet '{request.gauntlet_name}' not found.")

        # The gauntlet definition specifies the team to use.
        team_name = gauntlet_def.team_name
        team = team_manager.get_team(team_name)
        if not team:
            raise HTTPException(status_code=404, detail=f"Team '{team_name}' for gauntlet '{request.gauntlet_name}' not found.")

        # Call the headless function from the workflow_engine
        result = run_gauntlet_headless(
            solution_content=request.solution_content,
            gauntlet_def=gauntlet_def,
            team=team,
            context=request.context
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catching generic exception to handle any internal errors during gauntlet run
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
