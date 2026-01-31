"""
Z3 Integration REST API Server

FastAPI-based REST API providing:
- Constraint solving endpoints
- Theorem proving endpoints
- Optimization endpoints
- Translation endpoints
- Real-time progress via WebSocket
- Health checks and metrics

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class FastAPI:
        pass
    class BaseModel:
        pass
    class HTTPException(Exception):
        pass

# Import Z3 integration components
try:
    from z3_config_manager import get_config_manager, IntegrationConfig
    from z3_database_models import get_database_manager, SolverResult, TheoremProof
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from z3prover_integration import get_z3_solver_engine, Z3Variable, Z3Constraint, Z3ConstraintType
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3prover_advanced import get_z3_advanced_solver, OptimizationObjective
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

try:
    from z3_leanaide_openevolve_integration import solve_with_z3_leanaide
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    from z3_performance_monitor import get_z3_performance_monitor, monitored
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class SolveRequest(BaseModel):
    """Request model for constraint solving."""
    problem: str = Field(..., description="Problem statement or SMT-LIB")
    variables: Optional[List[Dict[str, Any]]] = Field(None, description="Variable definitions")
    constraints: Optional[List[str]] = Field(None, description="Constraint expressions")
    timeout: Optional[float] = Field(60.0, description="Timeout in seconds")


class SolveResponse(BaseModel):
    """Response model for constraint solving."""
    success: bool
    result_id: str
    status: str
    satisfiable: Optional[bool]
    model: Optional[Dict[str, Any]]
    execution_time_ms: float
    solver_used: str


class OptimizeRequest(BaseModel):
    """Request model for optimization."""
    variables: List[Dict[str, Any]] = Field(..., description="Variable definitions")
    constraints: List[str] = Field(..., description="Constraint expressions")
    objective: Dict[str, str] = Field(..., description="Objective function")
    direction: str = Field("minimize", description="minimize or maximize")
    multi_objective: bool = Field(False, description="Enable multi-objective")


class ProveRequest(BaseModel):
    """Request model for theorem proving."""
    theorem: str = Field(..., description="Theorem statement or SMT-LIB")
    assumptions: Optional[List[str]] = Field(None, description="List of assumptions")
    extract_proof: bool = Field(False, description="Extract detailed proof")


class ProveResponse(BaseModel):
    """Response model for theorem proving."""
    success: bool
    result_id: str
    proven: bool
    confidence: float
    tactics_used: Optional[List[str]]
    counterexample: Optional[Dict[str, Any]]
    proof: Optional[str]
    execution_time_ms: float


class TranslateRequest(BaseModel):
    """Request model for translation."""
    content: str = Field(..., description="Content to translate")
    direction: str = Field(..., description="smt_to_lean or lean_to_smt")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "2.0.0"
    components: Dict[str, bool]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response."""
    timestamp: str
    summary: Dict[str, Any]
    operations: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.operation_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from operation subscribers
        for op_id, subscribers in self.operation_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    def subscribe_to_operation(self, websocket: WebSocket, operation_id: str):
        """Subscribe to updates for a specific operation."""
        if operation_id not in self.operation_subscribers:
            self.operation_subscribers[operation_id] = []
        self.operation_subscribers[operation_id].append(websocket)
    
    async def send_progress(self, operation_id: str, data: Dict[str, Any]):
        """Send progress update to subscribers."""
        if operation_id not in self.operation_subscribers:
            return
        
        message = {
            "type": "progress",
            "operation_id": operation_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection in self.operation_subscribers[operation_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# API Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Z3 Integration API Server...")
    
    # Initialize components
    if CONFIG_AVAILABLE:
        config = get_config_manager().config
        logger.info(f"Loaded configuration from {get_config_manager().config_path}")
    
    if MONITOR_AVAILABLE:
        monitor = get_z3_performance_monitor()
        monitor.start_monitoring()
        logger.info("Performance monitoring started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Z3 Integration API Server...")
    
    if MONITOR_AVAILABLE:
        get_z3_performance_monitor().stop_monitoring()


# =============================================================================
# Create FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required for the API server")
    
    app = FastAPI(
        title="Z3-LeanAIDE-OpenEvolve Integration API",
        description="REST API for Z3 constraint solving, theorem proving, and formal verification",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_config() -> IntegrationConfig:
    """Get configuration dependency."""
    if CONFIG_AVAILABLE:
        return get_config_manager().config
    raise HTTPException(status_code=503, detail="Configuration not available")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "Z3-LeanAIDE-OpenEvolve Integration API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start_time = datetime.utcnow()
    
    components = {
        "z3": Z3_AVAILABLE,
        "z3_advanced": Z3_ADVANCED_AVAILABLE,
        "leanaide_integration": INTEGRATION_AVAILABLE,
        "configuration": CONFIG_AVAILABLE,
        "monitoring": MONITOR_AVAILABLE,
        "database": False  # Would check DB connection
    }
    
    # Calculate uptime (simplified)
    uptime = 0  # Would track actual uptime
    
    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        components=components,
        uptime_seconds=uptime
    )


@app.post("/solve", response_model=SolveResponse)
async def solve_constraints(request: SolveRequest):
    """Solve constraint satisfaction problem."""
    if not Z3_AVAILABLE:
        raise HTTPException(status_code=503, detail="Z3 not available")
    
    import time
    start_time = time.time()
    
    try:
        solver = get_z3_solver_engine()
        
        # Check if SMT-LIB
        if '(assert' in request.problem or '(declare' in request.problem:
            result = solver.solve_smtlib(request.problem)
        else:
            # Parse variables
            variables = [
                Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER')])
                for v in (request.variables or [])
            ]
            
            constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in (request.constraints or [])
            ]
            
            result = solver.solve_constraints(variables, constraints)
        
        execution_time = (time.time() - start_time) * 1000
        
        return SolveResponse(
            success=True,
            result_id=f"result_{int(time.time())}",
            status=result.status.value,
            satisfiable=result.is_sat(),
            model=result.model.assignments if result.model else None,
            execution_time_ms=execution_time,
            solver_used="z3"
        )
    
    except Exception as e:
        logger.error(f"Solve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_problem(request: OptimizeRequest):
    """Solve optimization problem."""
    if not Z3_ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Z3 advanced features not available")
    
    import time
    start_time = time.time()
    
    try:
        solver = get_z3_advanced_solver()
        
        variables = [
            Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER')])
            for v in request.variables
        ]
        
        constraints = [
            Z3Constraint(c, Z3ConstraintType.INTEGER)
            for c in request.constraints
        ]
        
        obj_type = OptimizationObjective.MINIMIZE if request.direction == "minimize" else OptimizationObjective.MAXIMIZE
        objectives = [(request.objective.get('expression', 'x'), obj_type)]
        
        result = solver.optimize(variables, constraints, objectives)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": result.success,
            "result_id": f"opt_{int(time.time())}",
            "optimal_value": result.optimal_value,
            "model": result.optimal_model.assignments if result.optimal_model else None,
            "is_pareto": result.is_pareto,
            "pareto_front_size": len(result.pareto_front),
            "execution_time_ms": execution_time
        }
    
    except Exception as e:
        logger.error(f"Optimize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prove", response_model=ProveResponse)
async def prove_theorem(request: ProveRequest):
    """Prove theorem."""
    if not Z3_AVAILABLE:
        raise HTTPException(status_code=503, detail="Z3 not available")
    
    import time
    start_time = time.time()
    
    try:
        from z3prover_integration import get_z3_theorem_prover
        prover = get_z3_theorem_prover()
        
        result = prover.prove_theorem(
            request.theorem,
            request.assumptions or []
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return ProveResponse(
            success=True,
            result_id=f"proof_{int(time.time())}",
            proven=result.proven,
            confidence=0.95 if result.proven else 0.3,
            tactics_used=[result.tactic_used] if result.tactic_used else None,
            counterexample=result.counterexample,
            proof=result.proof[:1000] if result.proof and request.extract_proof else None,
            execution_time_ms=execution_time
        )
    
    except Exception as e:
        logger.error(f"Prove error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve-complete")
async def solve_complete(request: SolveRequest):
    """Solve using the complete integrated workflow."""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Integration not available")
    
    try:
        result = await solve_with_z3_leanaide(request.problem)
        return result
    
    except Exception as e:
        logger.error(f"Complete solve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    monitor = get_z3_performance_monitor()
    dashboard = monitor.get_dashboard_data()
    
    return MetricsResponse(
        timestamp=datetime.utcnow().isoformat(),
        summary=dashboard.get("summary", {}),
        operations=dashboard.get("operation_performance", {}),
        bottlenecks=dashboard.get("top_bottlenecks", []),
        alerts=dashboard.get("recent_alerts", [])
    )


@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    if not CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    return get_config_manager().to_dict()


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "subscribe":
                operation_id = data.get("operation_id")
                if operation_id:
                    manager.subscribe_to_operation(websocket, operation_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "operation_id": operation_id
                    })
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/progress/{operation_id}")
async def websocket_progress(websocket: WebSocket, operation_id: str):
    """WebSocket endpoint for operation progress updates."""
    await manager.connect(websocket)
    manager.subscribe_to_operation(websocket, operation_id)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "operation_id": operation_id
        })
        
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Progress WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# Background Tasks
# =============================================================================

async def cleanup_expired_cache():
    """Background task to clean up expired cache entries."""
    # Would implement periodic cleanup
    pass


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        return
    
    # Get configuration
    if CONFIG_AVAILABLE:
        config = get_config_manager().config
        host = config.server.host
        port = config.server.port
    else:
        host = "0.0.0.0"
        port = 8765
    
    print(f"Starting Z3 Integration API Server on {host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "z3_api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
