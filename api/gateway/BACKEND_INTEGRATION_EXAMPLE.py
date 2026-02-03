"""
Example: Connecting Backend Engines to API Gateway

This file demonstrates how to integrate the existing Python backend engines
with the API Gateway WITHOUT modifying the engines themselves.

Principles:
1. NO modifications to backend engines (AIR GAP principle)
2. Call engines through standard Python imports
3. Handle async/await appropriately
4. Return responses in gateway format
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Backend Integration Example
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from fastapi import BackgroundTasks
from evolution import EvolutionaryOptimizer  # Existing backend
from adversarial import AdversarialEngine  # Existing backend
from analytics import AnalyticsManager  # Existing backend
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EXAMPLE 1: Evolution Engine Integration
# =============================================================================

async def run_evolution_backend(
    content: str,
    max_iterations: int,
    population_size: int,
    temperature: float,
    evolution_id: str,
    websocket_manager
):
    """
    Run the existing EvolutionaryOptimizer in background

    Note: We DON'T modify the EvolutionaryOptimizer class.
    We just call it and translate results to WebSocket messages.
    """
    try:
        # Create instance of existing backend engine
        optimizer = EvolutionaryOptimizer()

        # Define callback for progress updates
        def on_progress(generation, best_fitness, population):
            """Callback to broadcast progress via WebSocket"""
            import asyncio
            asyncio.create_task(
                websocket_manager.broadcast_progress(
                    evolution_id,
                    (generation / max_iterations) * 100,
                    f"Generation {generation}/{max_iterations}"
                )
            )

            # Also broadcast generation data
            asyncio.create_task(
                websocket_manager.broadcast_update(
                    evolution_id,
                    "generation_complete",
                    {
                        "generation": generation,
                        "best_fitness": best_fitness,
                        "population_size": len(population)
                    }
                )
            )

        # Run the evolution (blocking call from backend)
        # The backend doesn't know about WebSockets - we handle that
        result = optimizer.run(
            initial_content=content,
            max_iterations=max_iterations,
            population_size=population_size,
            temperature=temperature,
            progress_callback=on_progress  # Pass our callback
        )

        # Broadcast completion
        await websocket_manager.broadcast_complete(
            evolution_id,
            {
                "best_content": result.best_content,
                "best_fitness": result.best_fitness,
                "iterations": result.iterations
            }
        )

        logger.info(f"Evolution {evolution_id} completed successfully")

    except Exception as e:
        logger.error(f"Evolution {evolution_id} failed: {e}")
        await websocket_manager.broadcast_error(
            evolution_id,
            str(e)
        )


# Gateway endpoint that calls the backend
async def start_evolution_gateway(
    evolution_data: dict,
    background_tasks: BackgroundTasks,
    websocket_manager,
    user_id: str
):
    """
    API Gateway endpoint that starts evolution using existing backend
    """
    evolution_id = f"evo_{user_id}_{hash(evolution_data['content'])}"

    # Start evolution in background task
    background_tasks.add_task(
        run_evolution_backend,
        content=evolution_data["content"],
        max_iterations=evolution_data["parameters"]["max_iterations"],
        population_size=evolution_data["parameters"]["population_size"],
        temperature=evolution_data["parameters"]["temperature"],
        evolution_id=evolution_id,
        websocket_manager=websocket_manager
    )

    return {
        "evolution_id": evolution_id,
        "status": "running",
        "websocket_url": f"ws://localhost:8000/ws/evolution/{evolution_id}"
    }


# =============================================================================
# EXAMPLE 2: Adversarial Testing Integration
# =============================================================================

async def run_adversarial_backend(
    content: str,
    attack_modes: list,
    num_rounds: int,
    test_id: str,
    websocket_manager
):
    """
    Run the existing AdversarialEngine in background
    """
    try:
        # Create instance of existing backend engine
        engine = AdversarialEngine()

        # Run adversarial testing
        for round_num in range(1, num_rounds + 1):
            # Red team attack
            red_result = engine.red_team_attack(
                content=content,
                attack_mode=attack_modes[round_num % len(attack_modes)]
            )

            # Broadcast attack result
            await websocket_manager.broadcast_update(
                test_id,
                "attack_generated",
                {
                    "round": round_num,
                    "attack_mode": red_result.attack_mode,
                    "vulnerability": red_result.vulnerability,
                    "payload": red_result.payload
                }
            )

            # Blue team patch
            blue_result = engine.blue_team_patch(
                content=content,
                attack=red_result
            )

            # Broadcast patch result
            await websocket_manager.broadcast_update(
                test_id,
                "patch_generated",
                {
                    "round": round_num,
                    "patch": blue_result.patch,
                    "patched_content": blue_result.patched_content
                }
            )

            # Update content with patch
            content = blue_result.patched_content

        # Broadcast completion
        await websocket_manager.broadcast_complete(
            test_id,
            {
                "final_content": content,
                "total_vulnerabilities": num_rounds
            }
        )

    except Exception as e:
        logger.error(f"Adversarial test {test_id} failed: {e}")
        await websocket_manager.broadcast_error(test_id, str(e))


# =============================================================================
# EXAMPLE 3: Analytics Integration
# =============================================================================

def get_analytics_backend(user_id: str, start_date: str, end_date: str):
    """
    Get analytics from existing AnalyticsManager
    """
    # Create instance of existing backend
    manager = AnalyticsManager()

    # Call existing analytics methods
    metrics = manager.get_user_metrics(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

    performance = manager.get_performance_stats(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

    # Transform to gateway response format
    return {
        "period": {
            "start": start_date,
            "end": end_date
        },
        "metrics": {
            "total_evolutions": metrics.total_evolutions,
            "total_adversarial_tests": metrics.total_adversarial_tests,
            "average_fitness_improvement": metrics.avg_fitness_improvement
        },
        "performance": {
            "model_performance": performance.model_stats,
            "cost_analysis": performance.cost_breakdown
        }
    }


# =============================================================================
# EXAMPLE 4: Handling Backend That Doesn't Support Async
# =============================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for blocking backend calls
thread_pool = ThreadPoolExecutor(max_workers=4)


async def call_blocking_backend(backend_function, *args, **kwargs):
    """
    Call a blocking backend function in a thread pool

    Use this for backend engines that don't support async/await
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        backend_function,
        *args,
        **kwargs
    )
    return result


# Usage example
async def get_evolution_status_blocking(evolution_id: str):
    """
    Get evolution status from backend (blocking call)
    """
    from evolution import get_evolution_state  # Existing backend function

    # Call blocking backend function in thread pool
    status = await call_blocking_backend(
        get_evolution_state,
        evolution_id
    )

    # Transform to gateway format
    return {
        "evolution_id": evolution_id,
        "status": status.status,
        "current_iteration": status.iteration,
        "best_fitness": status.best_fitness
    }


# =============================================================================
# EXAMPLE 5: Error Handling and Translation
# =============================================================================

class BackendErrorTranslator:
    """
    Translate backend errors to API Gateway error responses
    """

    @staticmethod
    def translate_evolution_error(error: Exception) -> dict:
        """Translate EvolutionaryOptimizer errors to API format"""
        error_str = str(error)

        if "invalid content" in error_str.lower():
            return {
                "code": "VALIDATION_ERROR",
                "message": "Invalid content provided",
                "details": {"error": error_str}
            }
        elif "model not found" in error_str.lower():
            return {
                "code": "MODEL_ERROR",
                "message": "LLM model not available",
                "details": {"error": error_str}
            }
        else:
            return {
                "code": "EVOLUTION_ERROR",
                "message": "Evolution engine error",
                "details": {"error": error_str}
            }

    @staticmethod
    def translate_adversarial_error(error: Exception) -> dict:
        """Translate AdversarialEngine errors to API format"""
        error_str = str(error)

        if "invalid attack mode" in error_str.lower():
            return {
                "code": "VALIDATION_ERROR",
                "message": "Invalid attack mode specified",
                "details": {"error": error_str}
            }
        else:
            return {
                "code": "ADVERSARIAL_ERROR",
                "message": "Adversarial testing error",
                "details": {"error": error_str}
            }


# =============================================================================
# EXAMPLE 6: Complete Endpoint Implementation
# =============================================================================

from fastapi import APIRouter, Depends, HTTPException
from models.schemas import EvolutionStart, EvolutionStatus

router = APIRouter(prefix="/evolution", tags=["Evolution"])


@router.post("/start")
async def evolution_start(
    evolution_data: EvolutionStart,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    websocket_manager = Depends(lambda: evolution_room_manager)
):
    """
    Complete evolution endpoint with backend integration

    This demonstrates the full pattern:
    1. Validate request with Pydantic
    2. Call existing backend engine
    3. Start background task for async execution
    4. Return WebSocket URL for real-time updates
    """
    try:
        # Call the gateway wrapper that runs backend in background
        result = await start_evolution_gateway(
            evolution_data=evolution_data.dict(),
            background_tasks=background_tasks,
            websocket_manager=websocket_manager,
            user_id=current_user["user_id"]
        )

        return result

    except Exception as e:
        # Translate backend error to API format
        error_info = BackendErrorTranslator.translate_evolution_error(e)
        raise HTTPException(
            status_code=400,
            detail=error_info
        )


@router.get("/{evolution_id}")
async def evolution_get_status(
    evolution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get evolution status from backend
    """
    try:
        # Call backend (may be blocking)
        status = await get_evolution_status_blocking(evolution_id)

        return EvolutionStatus(**status)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": "Evolution not found"}
        )
    except Exception as e:
        error_info = BackendErrorTranslator.translate_evolution_error(e)
        raise HTTPException(
            status_code=500,
            detail=error_info
        )


# =============================================================================
# SUMMARY
# =============================================================================

"""
KEY PRINCIPLES DEMONSTRATED:

1. AIR GAP COMPLIANCE
   - Backend engines are imported, not modified
   - All translation happens in gateway layer
   - Backend remains unaware of API/WebSocket existence

2. ASYNC HANDLING
   - Use BackgroundTasks for long-running operations
   - Use ThreadPoolExecutor for blocking backend calls
   - Callbacks bridge backend events to WebSocket broadcasts

3. ERROR TRANSLATION
   - Catch backend exceptions
   - Translate to standardized API error format
   - Preserve error context for debugging

4. RESPONSE FORMATTING
   - Transform backend data structures to API models
   - Use Pydantic for validation
   - Consistent response format across all endpoints

5. WEBSOCKET BRIDGING
   - Backend callbacks broadcast to WebSocket channels
   - Real-time progress updates
   - Room-based subscriptions

This pattern can be replicated for ALL backend engines:
- AdversarialEngine
- AnalyticsManager
- KnowledgeEngine
- LeanAideIntegration
- MakerEngine
- MDAPEngine
- DecompositionEngine
- InventionPlanner
- etc.
"""

if __name__ == "__main__":
    print(__doc__)
