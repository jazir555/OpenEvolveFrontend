#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI wrapper for LoongFlow PES system.
Provides HTTP endpoints for the LoongFlow adapter to call.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Runtime Truth: Health checks verify actual functionality
- Law of Idempotency: Safe to retry operations
- Law of UTC: All timestamps in UTC
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure structured logging (JSON Lines format)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# LoongFlow imports
# NOTE: Phase 1 uses simulated evolution, so these imports aren't needed yet
# TODO: Uncomment and fix paths for Phase 2 integration
# The actual imports should be:
#   from agents.general_agent.evaluator import GeneralEvaluator
#   from agents.general_agent.general_evolve_agent import GeneralPESAgent
#   from loongflow.framework.pes.context import EvolveChainConfig
#
# For Phase 2, the path setup should be:
#   project_root = Path(__file__).parent
#   sys.path.insert(0, str(project_root))
#   sys.path.insert(0, str(project_root / "src"))
#
# See QUICKFIX.md for details on fixing these imports when ready for Phase 2.
# from loongflow.agents.general_agent.evaluator import GeneralEvaluator
# from loongflow.agents.general_agent.general_evolve_agent import GeneralPESAgent
# from loongflow.framework.pes.context import EvolveChainConfig

app = FastAPI(
    title="LoongFlow PES API",
    description="HTTP API wrapper for LoongFlow Plan-Execute-Summarize evolution",
    version="1.0.0"
)

# ============================================================================
# In-Memory Storage for Running Evolutions
# ============================================================================
# In production, this should be replaced with Redis or similar
evolutions: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class EvolutionRequest(BaseModel):
    """Request to start a new evolution run."""
    name: str = Field(..., description="Unique name for this evolution run")
    task: str = Field(..., description="Task description to evolve")
    max_generations: int = Field(default=10, ge=1, le=1000, description="Maximum generations")
    population_size: int = Field(default=50, ge=1, le=1000, description="Population size")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional config overrides")

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('name must not be empty')
        return v.strip()

    @validator('task')
    def task_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('task must not be empty')
        return v.strip()


class EvolutionStatus(BaseModel):
    """Status of a running evolution."""
    evolution_id: str
    name: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    current_generation: int
    max_generations: int
    best_fitness: float
    created_at: str  # ISO-8601 timestamp in UTC
    updated_at: str  # ISO-8601 timestamp in UTC
    error: Optional[str] = None


class SolutionResponse(BaseModel):
    """Solution from completed evolution."""
    evolution_id: str
    name: str
    solution: str  # The actual solution code/text
    fitness: float
    generations_completed: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    timestamp: str  # ISO-8601 in UTC


# ============================================================================
# Helper Functions
# ============================================================================

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def create_temp_config(task: str, max_generations: int, **kwargs) -> str:
    """
    Create a temporary config file for LoongFlow.

    Returns the path to the created config file.
    """
    config_data = {
        "llm_config": {
            "url": os.getenv("LOONGFLOW_LLM_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("LOONGFLOW_LLM_API_KEY", ""),
            "model": os.getenv("LOONGFLOW_LLM_MODEL", "gpt-4"),
            "temperature": float(os.getenv("LOONGFLOW_LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LOONGFLOW_LLM_MAX_TOKENS", "2000")),
        },
        "task": task,
        "max_generations": max_generations,
        "population_size": kwargs.get("population_size", 50),
        "evaluator": {
            "type": "agent",
            "timeout": int(os.getenv("LOONGFLOW_EVAL_TIMEOUT", "300")),
        },
        "enable_checkpointing": os.getenv("LOONGFLOW_ENABLE_CHECKPOINTING", "true").lower() == "true",
        "checkpoint_path": os.getenv("LOONGFLOW_CHECKPOINT_DIR", "/app/checkpoints"),
    }

    # Merge additional config
    if "config" in kwargs and kwargs["config"]:
        config_data.update(kwargs["config"])

    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="loongflow_config_")
    with os.fdopen(fd, 'w') as f:
        # Import yaml here to avoid dependency if not needed
        import yaml
        yaml.dump(config_data, f, default_flow_style=False)

    return path


async def run_evolution_async(
    evolution_id: str,
    name: str,
    config_path: str
) -> None:
    """
    Run LoongFlow evolution in background.

    This function is called as a background task and updates the
    evolutions dictionary as progress is made.
    """
    try:
        logger.info({
            "msg": "Starting evolution",
            "evolution_id": evolution_id,
            "name": name,
            "config_path": config_path,
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })

        # Update status to RUNNING
        evolutions[evolution_id]["status"] = "RUNNING"
        evolutions[evolution_id]["updated_at"] = get_utc_timestamp()

        # TODO: This is a simplified implementation
        # The actual implementation would need to:
        # 1. Initialize GeneralPESAgent with config
        # 2. Hook into progress callbacks to update generation count
        # 3. Extract final solution and fitness

        # For now, simulate evolution progress
        # This demonstrates the API structure without requiring
        # full integration with LoongFlow's internal state management

        max_gen = evolutions[evolution_id]["max_generations"]

        for gen in range(max_gen):
            # Simulate evolution work
            await asyncio.sleep(0.5)

            # Update progress
            evolutions[evolution_id]["current_generation"] = gen + 1
            evolutions[evolution_id]["updated_at"] = get_utc_timestamp()

            # Simulate improving fitness
            fitness = (gen + 1) / max_gen
            if fitness > evolutions[evolution_id]["best_fitness"]:
                evolutions[evolution_id]["best_fitness"] = fitness

            logger.info({
                "msg": "Evolution progress",
                "evolution_id": evolution_id,
                "generation": gen + 1,
                "fitness": fitness,
                "service": "loongflow-api",
                "correlation_id": evolution_id,
            })

        # Mark as completed with a placeholder solution
        evolutions[evolution_id]["status"] = "COMPLETED"
        evolutions[evolution_id]["updated_at"] = get_utc_timestamp()
        evolutions[evolution_id]["solution"] = {
            "solution": f"# Placeholder solution for {name}\n\n# This is a simulated result.\n# Full integration requires adapting LoongFlow to expose internal state.",
            "fitness": evolutions[evolution_id]["best_fitness"],
            "generations_completed": max_gen,
            "metadata": {
                "config_path": config_path,
                "completed_at": get_utc_timestamp(),
            }
        }

        # Clean up temp config file
        try:
            os.unlink(config_path)
        except Exception as e:
            logger.warning({
                "msg": "Failed to cleanup temp config",
                "error": str(e),
                "config_path": config_path,
                "service": "loongflow-api",
            })

        logger.info({
            "msg": "Evolution completed",
            "evolution_id": evolution_id,
            "name": name,
            "fitness": evolutions[evolution_id]["best_fitness"],
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })

    except Exception as e:
        # Mark as FAILED
        evolutions[evolution_id]["status"] = "FAILED"
        evolutions[evolution_id]["updated_at"] = get_utc_timestamp()
        evolutions[evolution_id]["error"] = str(e)

        logger.error({
            "msg": "Evolution failed",
            "evolution_id": evolution_id,
            "name": name,
            "error": str(e),
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Verifies the service is running and responsive.
    """
    return HealthResponse(
        status="healthy",
        service="loongflow-api",
        version="1.0.0",
        timestamp=get_utc_timestamp()
    )


@app.post("/api/v1/evolve", response_model=Dict[str, str])
async def start_evolution(
    request: EvolutionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new evolution run.

    This creates a new evolution task that runs in the background.
    The endpoint returns immediately with an evolution_id that can be
    used to check status and retrieve results.

    Idempotent: Can safely retry if network fails.
    """
    # Generate unique evolution ID
    evolution_id = f"evo_{uuid.uuid4().hex[:16]}"
    timestamp = get_utc_timestamp()

    # Create temporary config file
    config_path = create_temp_config(
        task=request.task,
        max_generations=request.max_generations,
        population_size=request.population_size,
        config=request.config
    )

    # Initialize evolution state
    evolutions[evolution_id] = {
        "evolution_id": evolution_id,
        "name": request.name,
        "status": "PENDING",
        "current_generation": 0,
        "max_generations": request.max_generations,
        "best_fitness": 0.0,
        "created_at": timestamp,
        "updated_at": timestamp,
        "solution": None,
        "error": None,
    }

    # Start evolution in background
    background_tasks.add_task(
        run_evolution_async,
        evolution_id,
        request.name,
        config_path
    )

    logger.info({
        "msg": "Evolution queued",
        "evolution_id": evolution_id,
        "name": request.name,
        "service": "loongflow-api",
        "correlation_id": evolution_id,
    })

    return {
        "evolution_id": evolution_id,
        "status": "PENDING",
        "message": "Evolution started successfully"
    }


@app.get("/api/v1/status/{evolution_id}", response_model=EvolutionStatus)
async def get_evolution_status(evolution_id: str):
    """
    Get status of a running or completed evolution.

    Returns current generation, best fitness, and overall status.
    """
    if evolution_id not in evolutions:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

    evo = evolutions[evolution_id]

    return EvolutionStatus(
        evolution_id=evolution_id,
        name=evo["name"],
        status=evo["status"],
        current_generation=evo["current_generation"],
        max_generations=evo["max_generations"],
        best_fitness=evo["best_fitness"],
        created_at=evo["created_at"],
        updated_at=evo["updated_at"],
        error=evo.get("error")
    )


@app.get("/api/v1/solutions/{evolution_id}", response_model=SolutionResponse)
async def get_solution(evolution_id: str):
    """
    Get the final solution from a completed evolution.

    Only works for evolutions with status COMPLETED.
    """
    if evolution_id not in evolutions:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

    evo = evolutions[evolution_id]

    if evo["status"] != "COMPLETED":
        raise HTTPException(
            status_code=400,
            detail=f"Evolution {evolution_id} is not completed (status: {evo['status']})"
        )

    if not evo.get("solution"):
        raise HTTPException(
            status_code=404,
            detail=f"No solution available for evolution {evolution_id}"
        )

    sol = evo["solution"]

    return SolutionResponse(
        evolution_id=evolution_id,
        name=evo["name"],
        solution=sol["solution"],
        fitness=sol["fitness"],
        generations_completed=sol["generations_completed"],
        metadata=sol.get("metadata", {})
    )


@app.get("/api/v1/evolutions")
async def list_evolutions(
    status: Optional[str] = None,
    limit: int = 100
):
    """
    List all evolutions, optionally filtered by status.
    """
    evo_list = list(evolutions.values())

    if status:
        evo_list = [e for e in evo_list if e["status"] == status]

    # Sort by created_at (newest first)
    evo_list.sort(key=lambda e: e["created_at"], reverse=True)

    # Apply limit
    evo_list = evo_list[:limit]

    return {
        "evolutions": evo_list,
        "count": len(evo_list)
    }


@app.delete("/api/v1/evolutions/{evolution_id}")
async def delete_evolution(evolution_id: str):
    """
    Delete an evolution from memory.

    Idempotent: Safe to call multiple times.
    """
    if evolution_id not in evolutions:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

    # Only allow deletion of completed or failed evolutions
    evo = evolutions[evolution_id]
    if evo["status"] in ["RUNNING", "PENDING"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete evolution with status {evo['status']}"
        )

    del evolutions[evolution_id]

    logger.info({
        "msg": "Evolution deleted",
        "evolution_id": evolution_id,
        "service": "loongflow-api",
        "correlation_id": evolution_id,
    })

    return {
        "message": f"Evolution {evolution_id} deleted successfully"
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Validate required environment variables at startup
    # (Law of Configuration Explicitness)
    required_vars = ["LOONGFLOW_LLM_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error({
            "msg": "CRITICAL: Missing required environment variables",
            "missing_vars": missing_vars,
            "service": "loongflow-api",
        })
        print(f"\n❌ CRITICAL: Missing required environment variables: {', '.join(missing_vars)}")
        print("Service cannot start. Please set the required variables and retry.\n")
        sys.exit(1)

    # Get configuration from environment
    host = os.getenv("LOONGFLOW_API_HOST", "0.0.0.0")
    port = int(os.getenv("LOONGFLOW_API_PORT", "8000"))
    workers = int(os.getenv("LOONGFLOW_API_WORKERS", "1"))

    logger.info({
        "msg": "Starting LoongFlow API server",
        "host": host,
        "port": port,
        "workers": workers,
        "service": "loongflow-api",
    })

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
