"""PES Enhanced API routes for integration with api_server.py.

This module provides REST API endpoints for the PES Enhanced system,
allowing external systems to run cost-aware evolution with early stopping,
budget tracking, and real-time monitoring.

Integration:
    from openevolve_pes_enhanced.api_routes import router as pes_enhanced_router
    app.include_router(pes_enhanced_router)
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

# Import PES Enhanced components
try:
    from .integration_wrapper import PESIntegrationWrapper, EnhancedEvolutionResult
    from .config import PESEnhancedConfig
    from .cost_optimizer import CostAwarePlanner, BudgetStatus
    PES_ENHANCED_AVAILABLE = True
except ImportError:
    PES_ENHANCED_AVAILABLE = False
    PESIntegrationWrapper = None
    EnhancedEvolutionResult = None
    PESEnhancedConfig = None
    CostAwarePlanner = None
    BudgetStatus = None

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(prefix="/pes-enhanced", tags=["pes-enhanced"])

# Security
security = HTTPBearer(auto_error=False)


# =============================================================================
# Pydantic Models for Requests/Responses
# =============================================================================

class TestCase(BaseModel):
    """Test case definition for evolution."""
    name: str = Field(..., description="Test case name")
    input: str = Field(..., description="Test input")
    expected_output: Optional[str] = Field(None, description="Expected output")
    expected_behavior: Optional[str] = Field(None, description="Expected behavior description")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Test weight/importance")


class PESEnhancedRunRequest(BaseModel):
    """Request to start a PES Enhanced evolution run."""
    code: str = Field(..., min_length=1, description="Source code to evolve")
    problem_description: str = Field(..., min_length=10, description="Problem description")
    tests: List[TestCase] = Field(default_factory=list, description="Test cases")
    language: Optional[str] = Field("python", description="Programming language")
    
    # Cost optimization settings
    max_cost_usd: Optional[float] = Field(None, ge=0.0, description="Maximum budget in USD")
    max_tokens: Optional[int] = Field(None, ge=1000, description="Maximum token budget")
    max_time_seconds: Optional[int] = Field(1800, ge=60, description="Maximum time in seconds")
    
    # Enhancement toggles
    enable_cost_optimization: bool = Field(True, description="Enable cost tracking")
    enable_early_stopping: bool = Field(True, description="Enable early stopping")
    enable_planning: bool = Field(True, description="Enable strategy planning")
    enable_summarization: bool = Field(True, description="Enable result summarization")
    enable_adaptive_parameters: bool = Field(True, description="Enable adaptive parameters")
    
    # Evolution parameters
    max_iterations: Optional[int] = Field(None, ge=1, le=1000, description="Maximum iterations")
    population_size: Optional[int] = Field(None, ge=5, le=500, description="Population size")
    
    # Early stopping configuration
    early_stopping_patience: int = Field(5, ge=1, le=50, description="Early stopping patience")
    early_stopping_min_improvement: float = Field(0.01, ge=0.0, le=1.0, description="Min improvement threshold")
    
    # Callback/Webhook
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")
    
    @validator('code')
    def validate_code(cls, v):
        if not v or not v.strip():
            raise ValueError('Code cannot be empty')
        return v.strip()
    
    @validator('problem_description')
    def validate_problem(cls, v):
        if not v or not v.strip():
            raise ValueError('Problem description cannot be empty')
        return v.strip()


class EvolutionMetrics(BaseModel):
    """Metrics from an evolution run."""
    total_evaluations: int = Field(0, description="Total evaluations performed")
    evaluations_saved: int = Field(0, description="Evaluations saved vs baseline")
    efficiency_gain: float = Field(0.0, description="Efficiency gain (0.0 to 1.0)")
    iterations_to_best: int = Field(0, description="Iterations to reach best solution")
    convergence_rate: float = Field(0.0, description="Convergence rate")
    time_saved_ms: int = Field(0, description="Time saved in milliseconds")
    cost_saved_usd: float = Field(0.0, description="Cost saved vs baseline")


class BudgetStatusResponse(BaseModel):
    """Current budget status."""
    cost_used_usd: float = Field(0.0, description="Cost used in USD")
    cost_remaining_usd: float = Field(0.0, description="Cost remaining in USD")
    cost_pct_used: float = Field(0.0, description="Cost percentage used")
    tokens_used: int = Field(0, description="Tokens used")
    tokens_remaining: int = Field(0, description="Tokens remaining")
    tokens_pct_used: float = Field(0.0, description="Token percentage used")
    time_used_ms: int = Field(0, description="Time used in milliseconds")
    time_remaining_ms: int = Field(0, description="Time remaining in milliseconds")
    time_pct_used: float = Field(0.0, description="Time percentage used")
    status: str = Field("ok", description="Budget status: ok, warning, critical, exceeded")
    should_stop: bool = Field(False, description="Whether evolution should stop")


class PESEnhancedRunResponse(BaseModel):
    """Response from a PES Enhanced evolution run."""
    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Run status: pending, running, completed, failed, stopped")
    success: bool = Field(False, description="Whether evolution succeeded")
    
    # Result data
    best_solution: Optional[str] = Field(None, description="Best evolved code")
    best_fitness: float = Field(0.0, description="Best fitness score")
    
    # Cost data
    total_cost_usd: float = Field(0.0, description="Total cost in USD")
    efficiency_gain: float = Field(0.0, description="Efficiency gain percentage")
    
    # Execution data
    iterations: int = Field(0, description="Iterations performed")
    converged: bool = Field(False, description="Whether evolution converged")
    stopped_early: bool = Field(False, description="Whether stopped early")
    stop_reason: Optional[str] = Field(None, description="Reason for stopping")
    
    # Strategy info
    strategy_used: str = Field("unknown", description="Evolution strategy used")
    recommendations: List[str] = Field(default_factory=list, description="Post-run recommendations")
    
    # Timestamps
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    
    # Metrics
    metrics: EvolutionMetrics = Field(default_factory=EvolutionMetrics, description="Evolution metrics")
    budget_status: Optional[BudgetStatusResponse] = Field(None, description="Final budget status")


class CostEstimateRequest(BaseModel):
    """Request to estimate evolution cost."""
    iterations: int = Field(50, ge=1, le=1000, description="Number of iterations")
    population_size: int = Field(20, ge=5, le=500, description="Population size")
    problem_complexity: str = Field("medium", description="Problem complexity: low, medium, high, very_high")
    avg_tokens_per_eval: int = Field(500, ge=100, description="Average tokens per evaluation")
    
    @validator('problem_complexity')
    def validate_complexity(cls, v):
        valid = ['low', 'medium', 'high', 'very_high']
        if v not in valid:
            raise ValueError(f'Complexity must be one of: {valid}')
        return v


class CostEstimateResponse(BaseModel):
    """Cost estimation response."""
    estimated_cost_usd: float = Field(..., description="Estimated total cost in USD")
    estimated_tokens: int = Field(..., description="Estimated total tokens")
    estimated_duration_ms: int = Field(..., description="Estimated duration in milliseconds")
    recommended_strategy: str = Field(..., description="Recommended evolution strategy")
    
    # Breakdown
    prompt_tokens: int = Field(..., description="Estimated prompt tokens")
    completion_tokens: int = Field(..., description="Estimated completion tokens")
    prompt_cost_usd: float = Field(..., description="Estimated prompt cost")
    completion_cost_usd: float = Field(..., description="Estimated completion cost")
    total_evaluations: int = Field(..., description="Total evaluations")
    
    # Recommendations
    parameter_recommendations: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameters")


class BudgetStatusRequest(BaseModel):
    """Request to get budget status."""
    include_projections: bool = Field(True, description="Include budget projections")


class StopRunRequest(BaseModel):
    """Request to stop a running evolution."""
    reason: str = Field("user_request", description="Reason for stopping")
    force: bool = Field(False, description="Force stop immediately")


class StopRunResponse(BaseModel):
    """Response from stopping a run."""
    run_id: str = Field(..., description="Run ID")
    success: bool = Field(..., description="Whether stop was successful")
    previous_status: str = Field(..., description="Status before stopping")
    current_status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class RunListResponse(BaseModel):
    """List of evolution runs."""
    runs: List[Dict[str, Any]] = Field(default_factory=list, description="List of runs")
    total_count: int = Field(0, description="Total number of runs")
    running_count: int = Field(0, description="Number of running runs")
    completed_count: int = Field(0, description="Number of completed runs")


class StrategyRecommendationRequest(BaseModel):
    """Request for strategy recommendations."""
    problem_description: str = Field(..., min_length=10, description="Problem description")
    code: Optional[str] = Field(None, description="Source code (optional)")
    language: Optional[str] = Field(None, description="Programming language")
    max_cost_usd: float = Field(10.0, ge=0.0, description="Maximum budget in USD")


class StrategyRecommendationResponse(BaseModel):
    """Strategy recommendation response."""
    strategy: str = Field(..., description="Recommended strategy")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    estimated_evaluations: int = Field(..., description="Estimated number of evaluations")
    reasoning: str = Field(..., description="Reasoning for recommendation")
    recommended_parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameters")


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., description="Message type: status, progress, result, error")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp")


# =============================================================================
# Run State Management
# =============================================================================

@dataclass
class _PERunState:
    """Internal state tracking for a PES Enhanced run."""
    run_id: str
    status: str  # pending, running, completed, failed, stopped
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    request: Optional[PESEnhancedRunRequest] = None
    result: Optional[PESEnhancedRunResponse] = None
    error: Optional[str] = None
    wrapper: Optional[PESIntegrationWrapper] = None
    task: Optional[asyncio.Task] = None
    cancel_requested: bool = False
    stop_reason: Optional[str] = None
    
    # Real-time monitoring data
    current_iteration: int = 0
    current_fitness: float = 0.0
    budget_status: Optional[BudgetStatusResponse] = None
    
    # WebSocket connections
    websocket_connections: Set[WebSocket] = field(default_factory=set)


# In-memory run storage (replace with database in production)
_pe_runs: Dict[str, _PERunState] = {}


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"pes-enhanced-{uuid.uuid4().hex[:12]}"


def _create_config_from_request(request: PESEnhancedRunRequest) -> PESEnhancedConfig:
    """Create PESEnhancedConfig from request."""
    config = PESEnhancedConfig()
    
    # Set enhancement toggles
    config.enable_cost_optimization = request.enable_cost_optimization
    config.enable_early_stopping = request.enable_early_stopping
    config.enable_planning = request.enable_planning
    config.enable_summarization = request.enable_summarization
    config.enable_adaptive_parameters = request.enable_adaptive_parameters
    
    # Configure cost settings
    if request.max_cost_usd:
        config.cost.max_cost_usd = request.max_cost_usd
    if request.max_tokens:
        config.cost.max_tokens = request.max_tokens
    if request.max_time_seconds:
        config.cost.max_time_seconds = request.max_time_seconds
    
    # Configure early stopping
    config.early_stopping.patience = request.early_stopping_patience
    config.early_stopping.min_improvement = request.early_stopping_min_improvement
    
    return config


def _budget_status_to_response(status: BudgetStatus) -> BudgetStatusResponse:
    """Convert BudgetStatus to response model."""
    return BudgetStatusResponse(
        cost_used_usd=status.cost_used_usd,
        cost_remaining_usd=status.cost_remaining_usd,
        cost_pct_used=status.cost_pct_used,
        tokens_used=status.tokens_used,
        tokens_remaining=status.tokens_remaining,
        tokens_pct_used=status.tokens_pct_used,
        time_used_ms=status.time_used_ms,
        time_remaining_ms=status.time_remaining_ms,
        time_pct_used=status.time_pct_used,
        status=status.status,
        should_stop=status.should_stop
    )


def _enhanced_result_to_response(
    run_id: str,
    run_state: _PERunState,
    result: EnhancedEvolutionResult
) -> PESEnhancedRunResponse:
    """Convert EnhancedEvolutionResult to API response."""
    
    # Extract best solution
    best_solution = None
    if result.original_result:
        if hasattr(result.original_result, 'code'):
            best_solution = result.original_result.code
        elif hasattr(result.original_result, 'best_code'):
            best_solution = result.original_result.best_code
    
    # Get metrics
    metrics = EvolutionMetrics(
        total_evaluations=getattr(result.original_result, 'total_evaluations', 0) if result.original_result else 0,
        evaluations_saved=result.evaluations_saved,
        efficiency_gain=result.efficiency_gain,
        iterations_to_best=result.iterations_to_convergence or 0,
        convergence_rate=1.0 if result.converged else 0.0,
        time_saved_ms=0,  # Calculate from execution time
        cost_saved_usd=0.0  # Calculate from baseline
    )
    
    # Get budget status
    budget_response = None
    if run_state.wrapper and run_state.wrapper.cost_optimizer and run_state.wrapper.cost_optimizer.budget_tracker:
        budget_status = run_state.wrapper.cost_optimizer.budget_tracker.get_status()
        budget_response = _budget_status_to_response(budget_status)
    
    return PESEnhancedRunResponse(
        run_id=run_id,
        status=run_state.status,
        success=getattr(result.original_result, 'success', True) if result.original_result else False,
        best_solution=best_solution,
        best_fitness=getattr(result.original_result, 'best_fitness', 0.0) if result.original_result else 0.0,
        total_cost_usd=result.total_cost_usd,
        efficiency_gain=result.efficiency_gain,
        iterations=getattr(result.original_result, 'iterations', 0) if result.original_result else 0,
        converged=result.converged,
        stopped_early=result.stopped_early,
        stop_reason=result.stop_reason,
        strategy_used=result.planning_decision.strategy.value if result.planning_decision else "standard",
        recommendations=result.evolution_summary.recommendations if result.evolution_summary else [],
        created_at=run_state.created_at,
        started_at=run_state.started_at,
        completed_at=run_state.completed_at,
        metrics=metrics,
        budget_status=budget_response
    )


async def _broadcast_to_run_websockets(run_id: str, message: WebSocketMessage):
    """Broadcast a message to all WebSocket connections for a run."""
    run_state = _pe_runs.get(run_id)
    if not run_state:
        return
    
    disconnected = set()
    for ws in run_state.websocket_connections:
        try:
            await ws.send_json(message.dict())
        except Exception:
            disconnected.add(ws)
    
    # Remove disconnected clients
    run_state.websocket_connections -= disconnected


async def _execute_pes_run(run_id: str, run_state: _PERunState):
    """Execute a PES Enhanced run in the background."""
    if not run_state.request:
        run_state.status = "failed"
        run_state.error = "No request data"
        return
    
    request = run_state.request
    
    try:
        # Create wrapper with configuration
        config = _create_config_from_request(request)
        wrapper = PESIntegrationWrapper(config)
        run_state.wrapper = wrapper
        
        # Update status
        run_state.status = "running"
        run_state.started_at = datetime.utcnow().isoformat()
        
        # Broadcast start
        await _broadcast_to_run_websockets(run_id, WebSocketMessage(
            type="status",
            data={"status": "running", "message": "Evolution started"}
        ))
        
        # Convert tests to dict format
        tests_dict = [test.dict() for test in request.tests]
        
        # Run evolution
        result = await wrapper.enhance_with_planning(
            code=request.code,
            problem_description=request.problem_description,
            tests=tests_dict,
            language=request.language,
            max_cost_usd=request.max_cost_usd,
            max_iterations=request.max_iterations
        )
        
        # Check if cancelled
        if run_state.cancel_requested:
            run_state.status = "stopped"
            run_state.stop_reason = run_state.stop_reason or "user_cancelled"
        else:
            run_state.status = "completed"
        
        run_state.completed_at = datetime.utcnow().isoformat()
        
        # Build response
        response = _enhanced_result_to_response(run_id, run_state, result)
        run_state.result = response
        
        # Broadcast completion
        await _broadcast_to_run_websockets(run_id, WebSocketMessage(
            type="result",
            data={
                "run_id": run_id,
                "status": run_state.status,
                "success": response.success,
                "total_cost_usd": response.total_cost_usd,
                "efficiency_gain": response.efficiency_gain
            }
        ))
        
        # Trigger webhook if configured
        if request.webhook_url:
            await _trigger_webhook(request.webhook_url, response)
        
    except Exception as e:
        logger.error(f"PES Enhanced run {run_id} failed: {e}", exc_info=True)
        run_state.status = "failed"
        run_state.error = str(e)
        run_state.completed_at = datetime.utcnow().isoformat()
        
        # Broadcast error
        await _broadcast_to_run_websockets(run_id, WebSocketMessage(
            type="error",
            data={"error": str(e), "run_id": run_id}
        ))


async def _trigger_webhook(webhook_url: str, response: PESEnhancedRunResponse):
    """Trigger a webhook with the run results."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=response.dict(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                logger.info(f"Webhook triggered: {webhook_url}, status: {resp.status}")
    except Exception as e:
        logger.warning(f"Failed to trigger webhook {webhook_url}: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/runs",
    response_model=PESEnhancedRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start PES Enhanced evolution run",
    description="Start a cost-aware evolution run with early stopping and budget tracking."
)
async def start_pes_enhanced_run(request: PESEnhancedRunRequest):
    """Start a PES Enhanced evolution run with cost tracking.
    
    This endpoint initiates an evolution run with:
    - Cost-aware planning before evolution
    - Budget tracking during execution
    - Early stopping with convergence detection
    - Result summarization after completion
    
    The run executes asynchronously. Use the returned run_id to:
    - Check status: GET /pes-enhanced/runs/{run_id}
    - Monitor via WebSocket: /ws/pes-enhanced/monitor/{run_id}
    - Stop: POST /pes-enhanced/runs/{run_id}/stop
    """
    if not PES_ENHANCED_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PES Enhanced system is not available"
        )
    
    # Generate run ID
    run_id = _generate_run_id()
    
    # Create run state
    run_state = _PERunState(
        run_id=run_id,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        request=request
    )
    _pe_runs[run_id] = run_state
    
    # Start background execution
    task = asyncio.create_task(_execute_pes_run(run_id, run_state))
    run_state.task = task
    
    # Return initial response
    return PESEnhancedRunResponse(
        run_id=run_id,
        status="pending",
        success=False,
        created_at=run_state.created_at,
        message="Evolution run queued"
    )


@router.post(
    "/cost-estimate",
    response_model=CostEstimateResponse,
    summary="Estimate evolution cost",
    description="Estimate cost before running evolution."
)
async def estimate_evolution_cost(request: CostEstimateRequest):
    """Estimate cost before running evolution.
    
    Provides cost estimates based on:
    - Number of iterations
    - Population size
    - Problem complexity
    - Average tokens per evaluation
    
    Also returns recommended strategy and parameters.
    """
    if not PES_ENHANCED_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PES Enhanced system is not available"
        )
    
    try:
        planner = CostAwarePlanner()
        
        # Get cost estimate
        cost_estimate = planner.estimate_cost(
            iterations=request.iterations,
            population_size=request.population_size,
            avg_tokens_per_eval=request.avg_tokens_per_eval
        )
        
        # Get strategy recommendation
        strategy_rec = planner.recommend_strategy_for_budget(
            max_cost_usd=cost_estimate['total_cost_usd'],
            problem_complexity=request.problem_complexity
        )
        
        # Estimate duration (rough approximation: 1-2 seconds per eval)
        avg_time_per_eval_ms = 1500
        estimated_duration_ms = int(cost_estimate['total_evaluations'] * avg_time_per_eval_ms)
        
        return CostEstimateResponse(
            estimated_cost_usd=cost_estimate['total_cost_usd'],
            estimated_tokens=cost_estimate['total_tokens'],
            estimated_duration_ms=estimated_duration_ms,
            recommended_strategy=strategy_rec['strategy'],
            prompt_tokens=cost_estimate['prompt_tokens'],
            completion_tokens=cost_estimate['completion_tokens'],
            prompt_cost_usd=cost_estimate['prompt_cost_usd'],
            completion_cost_usd=cost_estimate['completion_cost_usd'],
            total_evaluations=cost_estimate['total_evaluations'],
            parameter_recommendations={
                "iterations": strategy_rec['iterations'],
                "population_size": strategy_rec['population_size'],
                "early_stopping": strategy_rec['early_stopping'],
                "use_cheap_model": strategy_rec['use_cheap_model']
            }
        )
        
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost estimation failed: {str(e)}"
        )


@router.get(
    "/runs",
    response_model=RunListResponse,
    summary="List PES Enhanced runs",
    description="List all PES Enhanced evolution runs."
)
async def list_pes_runs(
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List PES Enhanced evolution runs.
    
    Query Parameters:
    - status_filter: Filter by status (pending, running, completed, failed, stopped)
    - limit: Maximum number of results (default: 100)
    - offset: Offset for pagination (default: 0)
    """
    runs_list = []
    
    for run_id, run_state in _pe_runs.items():
        if status_filter and run_state.status != status_filter:
            continue
        
        runs_list.append({
            "run_id": run_id,
            "status": run_state.status,
            "created_at": run_state.created_at,
            "started_at": run_state.started_at,
            "completed_at": run_state.completed_at,
            "has_result": run_state.result is not None,
            "error": run_state.error
        })
    
    # Sort by created_at descending
    runs_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Apply pagination
    total = len(runs_list)
    runs_list = runs_list[offset:offset + limit]
    
    # Count by status
    running_count = sum(1 for r in _pe_runs.values() if r.status == "running")
    completed_count = sum(1 for r in _pe_runs.values() if r.status in ["completed", "stopped"])
    
    return RunListResponse(
        runs=runs_list,
        total_count=total,
        running_count=running_count,
        completed_count=completed_count
    )


@router.get(
    "/runs/{run_id}",
    response_model=PESEnhancedRunResponse,
    summary="Get run status and results",
    description="Get the status and results of a PES Enhanced evolution run."
)
async def get_pes_run(run_id: str):
    """Get the status and results of a PES Enhanced evolution run.
    
    Path Parameters:
    - run_id: The unique run identifier
    
    Returns the full run data including:
    - Current status (pending, running, completed, failed, stopped)
    - Results (if completed)
    - Cost and efficiency metrics
    - Budget status
    """
    run_state = _pe_runs.get(run_id)
    if not run_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    # If completed, return stored result
    if run_state.result:
        return run_state.result
    
    # Otherwise return current status
    return PESEnhancedRunResponse(
        run_id=run_id,
        status=run_state.status,
        success=False,
        created_at=run_state.created_at,
        started_at=run_state.started_at,
        completed_at=run_state.completed_at,
        error=run_state.error
    )


@router.get(
    "/runs/{run_id}/budget",
    response_model=BudgetStatusResponse,
    summary="Get budget status",
    description="Get current budget status for a run."
)
async def get_budget_status(run_id: str):
    """Get current budget status for a run.
    
    Path Parameters:
    - run_id: The unique run identifier
    
    Returns:
    - Cost used/remaining
    - Token usage
    - Time elapsed/remaining
    - Budget status (ok, warning, critical, exceeded)
    """
    run_state = _pe_runs.get(run_id)
    if not run_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    if not run_state.wrapper or not run_state.wrapper.cost_optimizer or not run_state.wrapper.cost_optimizer.budget_tracker:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cost optimization not enabled for this run"
        )
    
    budget_status = run_state.wrapper.cost_optimizer.budget_tracker.get_status()
    return _budget_status_to_response(budget_status)


@router.post(
    "/runs/{run_id}/stop",
    response_model=StopRunResponse,
    summary="Stop a running evolution",
    description="Stop a running PES Enhanced evolution run."
)
async def stop_pes_run(run_id: str, request: StopRunRequest):
    """Stop a running PES Enhanced evolution.
    
    Path Parameters:
    - run_id: The unique run identifier
    
    Request Body:
    - reason: Reason for stopping (default: "user_request")
    - force: Force immediate stop (default: false)
    
    Note: The stop is cooperative - the evolution will stop at the next
    checkpoint. Use force=true for immediate termination.
    """
    run_state = _pe_runs.get(run_id)
    if not run_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    previous_status = run_state.status
    
    if previous_status not in ["pending", "running"]:
        return StopRunResponse(
            run_id=run_id,
            success=False,
            previous_status=previous_status,
            current_status=previous_status,
            message=f"Cannot stop run with status: {previous_status}"
        )
    
    # Request cancellation
    run_state.cancel_requested = True
    run_state.stop_reason = request.reason
    
    if request.force and run_state.task:
        # Cancel the asyncio task
        run_state.task.cancel()
        run_state.status = "stopped"
    
    return StopRunResponse(
        run_id=run_id,
        success=True,
        previous_status=previous_status,
        current_status=run_state.status,
        message=f"Stop requested: {request.reason}"
    )


@router.post(
    "/recommend-strategy",
    response_model=StrategyRecommendationResponse,
    summary="Get strategy recommendation",
    description="Get strategy recommendation for a problem."
)
async def recommend_strategy(request: StrategyRecommendationRequest):
    """Get strategy recommendation for a problem.
    
    Analyzes the problem and returns:
    - Recommended evolution strategy
    - Estimated cost
    - Recommended parameters
    - Confidence score and reasoning
    """
    if not PES_ENHANCED_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PES Enhanced system is not available"
        )
    
    try:
        wrapper = PESIntegrationWrapper()
        recommendation = wrapper.recommend_parameters(
            problem_description=request.problem_description,
            max_cost_usd=request.max_cost_usd
        )
        
        return StrategyRecommendationResponse(
            strategy=recommendation['strategy'],
            confidence=recommendation['confidence'],
            estimated_cost_usd=recommendation['estimated_cost'],
            estimated_evaluations=recommendation['estimated_evaluations'],
            reasoning=recommendation['reasoning'],
            recommended_parameters=recommendation['parameters']
        )
        
    except Exception as e:
        logger.error(f"Strategy recommendation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if PES Enhanced system is available."
)
async def pes_enhanced_health():
    """Check if PES Enhanced system is available."""
    return {
        "available": PES_ENHANCED_AVAILABLE,
        "status": "healthy" if PES_ENHANCED_AVAILABLE else "unavailable",
        "active_runs": sum(1 for r in _pe_runs.values() if r.status == "running"),
        "total_runs": len(_pe_runs)
    }


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@router.websocket("/ws/monitor/{run_id}")
async def monitor_pes_run(websocket: WebSocket, run_id: str):
    """WebSocket for real-time evolution monitoring.
    
    Path Parameters:
    - run_id: The unique run identifier
    
    Message Types:
    - status: Run status updates
    - progress: Iteration progress updates
    - result: Final result
    - error: Error messages
    - budget: Budget status updates
    
    Example client (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/pes-enhanced/ws/monitor/pes-enhanced-abc123');
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log(msg.type, msg.data);
    };
    ```
    """
    await websocket.accept()
    
    run_state = _pe_runs.get(run_id)
    if not run_state:
        await websocket.send_json({
            "type": "error",
            "data": {"error": f"Run {run_id} not found"},
            "timestamp": datetime.utcnow().isoformat()
        })
        await websocket.close()
        return
    
    # Register WebSocket connection
    run_state.websocket_connections.add(websocket)
    
    try:
        # Send current status
        await websocket.send_json({
            "type": "status",
            "data": {
                "run_id": run_id,
                "status": run_state.status,
                "current_iteration": run_state.current_iteration,
                "current_fitness": run_state.current_fitness
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (with timeout)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle ping
                if message == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "data": {},
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "data": {"status": run_state.status},
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            # Check if run is complete
            if run_state.status in ["completed", "failed", "stopped"]:
                # Send final status
                if run_state.result:
                    await websocket.send_json({
                        "type": "result",
                        "data": run_state.result.dict(),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
    finally:
        # Unregister WebSocket
        run_state.websocket_connections.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================================
# Integration Helper
# =============================================================================

def get_pes_enhanced_router() -> APIRouter:
    """Get the PES Enhanced API router for inclusion in main app.
    
    Usage in api_server.py:
        from openevolve_pes_enhanced.api_routes import get_pes_enhanced_router
        app.include_router(get_pes_enhanced_router())
    """
    return router
