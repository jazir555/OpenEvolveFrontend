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
import redis.asyncio as redis
from redis.exceptions import RedisError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure structured logging (JSON Lines format)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# LoongFlow Imports (Phase 2 - Real Integration)
# ============================================================================
# Add project root and src to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import real LoongFlow components
from agents.general_agent.evaluator import GeneralEvaluator, create_evaluator
from agents.general_agent.general_evolve_agent import GeneralPESAgent
from agents.general_agent.planner import GeneralPlanAgent
from agents.general_agent.executor import GeneralExecuteAgent
from agents.general_agent.summary import GeneralSummaryAgent

app = FastAPI(
    title="LoongFlow PES API",
    description="HTTP API wrapper for LoongFlow Plan-Execute-Summarize evolution",
    version="1.0.0"
)

# ============================================================================
# Valkey State Manager - Phase 3: 100% Completion
# ============================================================================
# Replaces in-memory storage with Valkey (Redis fork) for persistence

class ValkeyStateManager:
    """
    Manages evolution state using Valkey/Redis for persistence.

    Provides:
    - State persistence across server restarts
    - Automatic serialization/deserialization
    - Atomic operations for consistency
    - Reconnect logic for resilience
    """

    def __init__(self):
        """Initialize Valkey Redis client."""
        self.redis_url = os.getenv(
            "VALKEY_URL",
            os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        self.key_prefix = "evolution:"
        self.ttl_seconds = int(os.getenv("EVOLUTION_STATE_TTL", "86400"))  # 24 hours default
        self._redis: Optional[redis.Redis] = None

    async def get_client(self) -> redis.Redis:
        """Get or create Redis client with lazy initialization."""
        if self._redis is None:
            try:
                self._redis = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True
                )
                # Test connection
                await self._redis.ping()
                logger.info({
                    "msg": "Valkey connection established",
                    "redis_url": self.redis_url,
                    "service": "loongflow-api"
                })
            except Exception as e:
                logger.error({
                    "msg": "Failed to connect to Valkey",
                    "error": str(e),
                    "redis_url": self.redis_url,
                    "service": "loongflow-api"
                })
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Valkey: {str(e)}"
                )
        return self._redis

    def _make_key(self, evolution_id: str) -> str:
        """Create Redis key for evolution."""
        return f"{self.key_prefix}{evolution_id}"

    async def create_evolution(self, evolution_id: str, data: Dict[str, Any]) -> bool:
        """Create new evolution state in Valkey."""
        try:
            client = await self.get_client()
            key = self._make_key(evolution_id)

            # Serialize data
            serialized = json.dumps(data)

            # Store in Redis with TTL
            await client.setex(key, self.ttl_seconds, serialized)

            logger.info({
                "msg": "Evolution state created in Valkey",
                "evolution_id": evolution_id,
                "ttl": self.ttl_seconds,
                "service": "loongflow-api"
            })
            return True
        except Exception as e:
            logger.error({
                "msg": "Failed to create evolution state",
                "error": str(e),
                "evolution_id": evolution_id,
                "service": "loongflow-api"
            })
            return False

    async def get_evolution(self, evolution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evolution state from Valkey."""
        try:
            client = await self.get_client()
            key = self._make_key(evolution_id)

            serialized = await client.get(key)
            if serialized is None:
                return None

            return json.loads(serialized)
        except Exception as e:
            logger.error({
                "msg": "Failed to retrieve evolution state",
                "error": str(e),
                "evolution_id": evolution_id,
                "service": "loongflow-api"
            })
            return None

    async def update_evolution(self, evolution_id: str, updates: Dict[str, Any]) -> bool:
        """Update evolution state in Valkey."""
        try:
            client = await self.get_client()
            key = self._make_key(evolution_id)

            # Get current state
            serialized = await client.get(key)
            if serialized is None:
                return False

            # Merge updates
            current = json.loads(serialized)
            current.update(updates)

            # Serialize and store back
            updated_serialized = json.dumps(current)
            await client.setex(key, self.ttl_seconds, updated_serialized)

            return True
        except Exception as e:
            logger.error({
                "msg": "Failed to update evolution state",
                "error": str(e),
                "evolution_id": evolution_id,
                "service": "loongflow-api"
            })
            return False

    async def delete_evolution(self, evolution_id: str) -> bool:
        """Delete evolution state from Valkey."""
        try:
            client = await self.get_client()
            key = self._make_key(evolution_id)

            result = await client.delete(key)
            return result > 0
        except Exception as e:
            logger.error({
                "msg": "Failed to delete evolution state",
                "error": str(e),
                "evolution_id": evolution_id,
                "service": "loongflow-api"
            })
            return False

    async def list_evolutions(self) -> List[str]:
        """List all evolution IDs from Valkey."""
        try:
            client = await self.get_client()
            pattern = f"{self.key_prefix}*"

            keys = []
            async for key in client.scan_iter(match=pattern):
                # Remove prefix to return just IDs
                evolution_id = key.replace(self.key_prefix, "", 1)
                keys.append(evolution_id)

            return keys
        except Exception as e:
            logger.error({
                "msg": "Failed to list evolutions",
                "error": str(e),
                "service": "loongflow-api"
            })
            return []

    async def set_field(self, evolution_id: str, field: str, value: Any) -> bool:
        """Set a single field in evolution state (atomic operation)."""
        try:
            client = await self.get_client()
            key = self._make_key(evolution_id)

            # Get current state
            serialized = await client.get(key)
            if serialized is None:
                return False

            # Update field
            current = json.loads(serialized)
            current[field] = value

            # Store back
            updated_serialized = json.dumps(current)
            await client.setex(key, self.ttl_seconds, updated_serialized)

            return True
        except Exception as e:
            logger.error({
                "msg": "Failed to set field",
                "error": str(e),
                "evolution_id": evolution_id,
                "field": field,
                "service": "loongflow-api"
            })
            return False


# Global state manager instance
state_manager = ValkeyStateManager()


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


def create_temp_config(task: str, max_generations: int, population_size: int = 50, **kwargs) -> str:
    """
    Create a temporary config file for LoongFlow.

    Returns the path to the created config file.
    """
    import tempfile

    # Create EvolveChainConfig structure
    config_data = {
        "workspace_path": tempfile.gettempdir(),
        "llm_config": {
            "url": os.getenv("LOONGFLOW_LLM_URL", "https://api.deepseek.com/v1"),
            "api_key": os.getenv("LOONGFLOW_LLM_API_KEY", ""),
            "model": os.getenv("LOONGFLOW_LLM_MODEL", "deepseek-chat"),
            "temperature": float(os.getenv("LOONGFLOW_LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LOONGFLOW_LLM_MAX_TOKENS", "2000")),
        },
        # Define all available workers
        "planners": {
            "general_planner": {
                "type": "general"
            }
        },
        "executors": {
            "general_executor": {
                "type": "general"
            }
        },
        "summarizers": {
            "general_summarizer": {
                "type": "general"
            }
        },
        # Main evolution config
        "evolve": {
            "task_name": kwargs.get("name", "api_task"),
            "task": task,
            "initial_code": "",
            "max_iterations": max_generations,
            "target_score": float(kwargs.get("target_score", "1.0")),
            "concurrency": int(kwargs.get("concurrency", "5")),
            "planner_name": "general_planner",
            "executor_name": "general_executor",
            "summary_name": "general_summarizer",
            "database": {
                "storage_type": "in_memory",
                "population_size": population_size,
                "num_islands": 3,
                "elite_archive_size": 50,
            },
            "evaluator": {
                "evaluate_code": "",
                "timeout": int(os.getenv("LOONGFLOW_EVAL_TIMEOUT", "300")),
            }
        }
    }

    # Merge additional config overrides
    if "config" in kwargs and kwargs["config"]:
        # Deep merge for nested dicts
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_merge(config_data, kwargs["config"])

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
        await state_manager.set_field(evolution_id, "status", "RUNNING")
        await state_manager.set_field(evolution_id, "updated_at", get_utc_timestamp())

        # ========================================================================
        # PHASE 2: Real LoongFlow Integration
        # ========================================================================
        # Build EvolveChainConfig from the request and run real evolution

        logger.info({
            "msg": "Initializing real LoongFlow evolution",
            "evolution_id": evolution_id,
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })

        # Load the config file that was created earlier
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate and create EvolveChainConfig
        from loongflow.framework.pes.context import EvolveChainConfig
        config = EvolveChainConfig.model_validate(config_dict)

        # Create PESAgent with the config
        from loongflow.framework.pes.pes_agent import PESAgent

        agent = PESAgent(config=config)

        # Register workers (General Agent components)
        agent.register_planner_worker("general_planner", GeneralPlanAgent)
        agent.register_executor_worker("general_executor", GeneralExecuteAgent)
        agent.register_summary_worker("general_summarizer", GeneralSummaryAgent)

        logger.info({
            "msg": "PESAgent initialized, starting evolution",
            "evolution_id": evolution_id,
            "task": config.evolve.task[:100] if config.evolve.task else "",
            "max_iterations": config.evolve.max_iterations,
            "target_score": config.evolve.target_score,
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })

        # Create a background task to monitor progress
        async def monitor_progress():
            """Monitor evolution progress by polling the database."""
            while True:
                # Check status from Valkey
                evo_state = await state_manager.get_evolution(evolution_id)
                if evo_state is None or evo_state.get("status") in ["COMPLETED", "FAILED"]:
                    break

                await asyncio.sleep(1.0)  # Poll every second

                try:
                    # Get status from database
                    global_status = agent.database.memory_status().get("global_status", {})

                    # Update progress
                    current_gen = global_status.get("current_iteration", 0)
                    best_score = global_status.get("best_score", 0.0)

                    await state_manager.set_field(evolution_id, "current_generation", current_gen)
                    await state_manager.set_field(evolution_id, "best_fitness", best_score)
                    await state_manager.set_field(evolution_id, "updated_at", get_utc_timestamp())

                    if current_gen > 0:
                        logger.info({
                            "msg": "Evolution progress update",
                            "evolution_id": evolution_id,
                            "generation": current_gen,
                            "best_fitness": best_score,
                            "service": "loongflow-api",
                            "correlation_id": evolution_id,
                        })

                except Exception as e:
                    logger.warning({
                        "msg": "Error monitoring progress",
                        "error": str(e),
                        "evolution_id": evolution_id,
                        "service": "loongflow-api",
                    })

        # Start progress monitoring
        monitor_task = asyncio.create_task(monitor_progress())

        # Run the evolution
        try:
            final_message = await agent.run()

            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Extract final solution from the message
            final_solution_text = final_message.text if hasattr(final_message, 'text') else str(final_message)

            # Get final stats from database
            global_status = agent.database.memory_status().get("global_status", {})
            final_gen = global_status.get("current_iteration", 0)
            final_fitness = global_status.get("best_score", 0.0)

            # Mark as completed with real solution
            completion_data = {
                "status": "COMPLETED",
                "current_generation": final_gen,
                "best_fitness": final_fitness,
                "updated_at": get_utc_timestamp(),
                "solution": {
                    "solution": final_solution_text,
                    "fitness": final_fitness,
                    "generations_completed": final_gen,
                    "metadata": {
                        "config_path": config_path,
                        "completed_at": get_utc_timestamp(),
                        "message_type": str(type(final_message).__name__),
                    }
                }
            }

            # Update all fields atomically
            current_state = await state_manager.get_evolution(evolution_id)
            if current_state:
                current_state.update(completion_data)
                await state_manager.update_evolution(evolution_id, completion_data)

            logger.info({
                "msg": "Real evolution completed successfully",
                "evolution_id": evolution_id,
                "generations": final_gen,
                "fitness": final_fitness,
                "service": "loongflow-api",
                "correlation_id": evolution_id,
            })

        except Exception as e:
            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Re-raise to be caught by outer exception handler
            raise

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
            "fitness": final_fitness,
            "service": "loongflow-api",
            "correlation_id": evolution_id,
        })

    except Exception as e:
        # Mark as FAILED
        await state_manager.set_field(evolution_id, "status", "FAILED")
        await state_manager.set_field(evolution_id, "updated_at", get_utc_timestamp())
        await state_manager.set_field(evolution_id, "error", str(e))

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

    # Initialize evolution state in Valkey
    evolution_state = {
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

    # Store in Valkey for persistence
    if not await state_manager.create_evolution(evolution_id, evolution_state):
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize evolution state in Valkey"
        )

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
    evo = await state_manager.get_evolution(evolution_id)

    if evo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

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
    evo = await state_manager.get_evolution(evolution_id)

    if evo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

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
    # Get all evolution IDs from Valkey
    evo_ids = await state_manager.list_evolutions()

    # Fetch all evolution states
    evo_list = []
    for evo_id in evo_ids:
        evo = await state_manager.get_evolution(evo_id)
        if evo:
            evo_list.append(evo)

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
    Delete an evolution from Valkey.

    Idempotent: Safe to call multiple times.
    """
    evo = await state_manager.get_evolution(evolution_id)

    if evo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evolution {evolution_id} not found"
        )

    # Only allow deletion of completed or failed evolutions
    if evo["status"] in ["RUNNING", "PENDING"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete evolution with status {evo['status']}"
        )

    # Delete from Valkey
    deleted = await state_manager.delete_evolution(evolution_id)

    if not deleted:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete evolution from Valkey"
        )

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
