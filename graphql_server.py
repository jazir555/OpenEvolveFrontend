"""
GraphQL API Server - License: Apache 2.0

GraphQL API for OpenEvolve using Strawberry (MIT License).
Provides flexible querying for workflows, knowledge, and decomposition.

Dependencies (all permissive licenses):
- strawberry-graphql: MIT License
- fastapi: MIT License
- uvicorn: BSD License
- pydantic: MIT License

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager

# Strawberry GraphQL - MIT License
import strawberry
from strawberry.types import Info
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL
from strawberry.federation import Schema

# FastAPI - MIT License
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

@strawberry.enum
class ProblemType(Enum):
    """Problem classification types."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    DESIGN = "design"


@strawberry.enum
class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@strawberry.enum
class DecompositionStrategy(Enum):
    """Available decomposition strategies."""
    SEMANTIC = "semantic"
    DEPENDENCY = "dependency"
    COMPLEXITY = "complexity"
    HYBRID = "hybrid"
    RESEARCH = "research"


@strawberry.enum
class KnowledgeSource(Enum):
    """Source of knowledge triples."""
    DEEPKE = "deepke"
    ONEKE = "oneke"
    KG_GEN = "kg_gen"
    AI_KG = "ai_kg"
    AGENTJSON = "agentjson"
    NEURALKG = "neuralkg"
    LEANAIDE = "leanaide"
    Z3 = "z3"


# =============================================================================
# TYPES
# =============================================================================

@strawberry.type
class SubProblem:
    """Sub-problem entity."""
    id: str
    title: str
    description: str
    type: ProblemType
    priority: int
    estimated_effort: int
    dependencies: List[str]
    status: str = "pending"
    
    @strawberry.field
    async def solutions(self, info: Info) -> List['Solution']:
        """Get solutions for this sub-problem."""
        # Resolver implementation
        return []


@strawberry.type
class DecompositionPlan:
    """Decomposition plan entity."""
    id: str
    problem_id: str
    strategy: DecompositionStrategy
    created_at: datetime
    updated_at: Optional[datetime] = None
    sub_problems: List[SubProblem]
    quality_score: Optional[float] = None
    
    @strawberry.field
    def num_subproblems(self) -> int:
        """Count of sub-problems."""
        return len(self.sub_problems)
    
    @strawberry.field
    def total_estimated_effort(self) -> int:
        """Total effort estimate across all sub-problems."""
        return sum(sp.estimated_effort for sp in self.sub_problems)


@strawberry.type
class Solution:
    """Solution entity."""
    id: str
    subproblem_id: str
    content: str
    quality_score: float
    created_at: datetime


@strawberry.type
class KnowledgeTriple:
    """Knowledge triple entity."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source: KnowledgeSource
    timestamp: datetime
    
    @strawberry.field
    def display(self) -> str:
        """Human-readable representation."""
        return f"{self.subject} {self.predicate} {self.object}"


@strawberry.type
class Workflow:
    """Workflow execution entity."""
    id: str
    problem_description: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    decomposition_plan: Optional[DecompositionPlan] = None
    final_solution: Optional[str] = None
    
    @strawberry.field
    def duration_seconds(self) -> Optional[float]:
        """Calculate workflow duration."""
        if self.updated_at and self.created_at:
            return (self.updated_at - self.created_at).total_seconds()
        return None
    
    @strawberry.field
    async def progress_percentage(self, info: Info) -> int:
        """Calculate completion percentage."""
        if not self.decomposition_plan:
            return 0
        # Calculate based on completed sub-problems
        return 0  # Implementation needed


@strawberry.type
class AnalyticsMetrics:
    """System analytics metrics."""
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_duration_seconds: float
    total_knowledge_triples: int
    decomposition_success_rate: float


@strawberry.type
class Event:
    """System event entity."""
    id: str
    event_type: str
    source: str
    timestamp: datetime
    payload: str  # JSON string
    workflow_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    @strawberry.field
    def payload_dict(self) -> str:
        """Parse payload as dictionary (returns JSON string)."""
        try:
            return json.dumps(json.loads(self.payload))
        except:
            return "{}"


# =============================================================================
# INPUT TYPES
# =============================================================================

@strawberry.input
class CreateWorkflowInput:
    """Input for creating a workflow."""
    problem_description: str
    problem_title: Optional[str] = None
    domain: Optional[str] = "general"
    problem_type: Optional[ProblemType] = ProblemType.ANALYSIS


@strawberry.input
class DecomposeProblemInput:
    """Input for problem decomposition."""
    title: str
    description: str
    domain: Optional[str] = "general"
    strategy: Optional[DecompositionStrategy] = DecompositionStrategy.HYBRID
    problem_type: Optional[ProblemType] = ProblemType.ANALYSIS


@strawberry.input
class ExtractKnowledgeInput:
    """Input for knowledge extraction."""
    text: str
    extractors: Optional[List[KnowledgeSource]] = None


@strawberry.input
class WorkflowFilter:
    """Filter for workflow queries."""
    status: Optional[WorkflowStatus] = None
    since: Optional[datetime] = None
    limit: Optional[int] = 10


# =============================================================================
# QUERIES
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query type."""
    
    @strawberry.field
    async def workflow(self, id: str, info: Info) -> Optional[Workflow]:
        """Get a workflow by ID."""
        try:
            from workflow_structures import WorkflowState
            
            # This would typically query from database
            # For now, return a mock/example
            return Workflow(
                id=id,
                problem_description="Example problem",
                status=WorkflowStatus.COMPLETED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error fetching workflow: {e}")
            return None
    
    @strawberry.field
    async def workflows(
        self,
        filter: Optional[WorkflowFilter] = None,
        info: Info = None
    ) -> List[Workflow]:
        """List workflows with optional filtering."""
        # Implementation would query database
        return []
    
    @strawberry.field
    async def decomposition_plan(
        self,
        id: str,
        info: Info
    ) -> Optional[DecompositionPlan]:
        """Get a decomposition plan by ID."""
        try:
            from decomposition_engine import DecompositionEngine
            # Query logic here
            return None
        except Exception as e:
            logger.error(f"Error fetching decomposition plan: {e}")
            return None
    
    @strawberry.field
    async def knowledge(
        self,
        query: str,
        limit: Optional[int] = 10,
        info: Info = None
    ) -> List[KnowledgeTriple]:
        """Search knowledge graph."""
        try:
            # Would integrate with knowledge engine
            return []
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    @strawberry.field
    async def analytics(self, info: Info) -> AnalyticsMetrics:
        """Get system analytics."""
        # Implementation would aggregate from various sources
        return AnalyticsMetrics(
            total_workflows=0,
            completed_workflows=0,
            failed_workflows=0,
            average_duration_seconds=0.0,
            total_knowledge_triples=0,
            decomposition_success_rate=0.0
        )
    
    @strawberry.field
    async def events(
        self,
        workflow_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = 100,
        info: Info = None
    ) -> List[Event]:
        """Get system events."""
        try:
            from event_bus import get_event_bus
            
            bus = await get_event_bus()
            history = await bus.get_history(
                workflow_id=workflow_id,
                limit=limit
            )
            
            return [
                Event(
                    id=e.id,
                    event_type=e.type.value,
                    source=e.source,
                    timestamp=e.timestamp,
                    payload=json.dumps(e.payload),
                    workflow_id=e.workflow_id,
                    correlation_id=e.correlation_id
                )
                for e in history
            ]
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []


# =============================================================================
# MUTATIONS
# =============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation type."""
    
    @strawberry.mutation
    async def create_workflow(
        self,
        input: CreateWorkflowInput,
        info: Info
    ) -> Workflow:
        """Create a new workflow."""
        workflow_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Publish event
        try:
            from event_bus import publish_event, EventType
            await publish_event(
                EventType.WORKFLOW_STARTED,
                {"problem": input.problem_description},
                workflow_id=workflow_id
            )
        except Exception as e:
            logger.warning(f"Could not publish event: {e}")
        
        return Workflow(
            id=workflow_id,
            problem_description=input.problem_description,
            status=WorkflowStatus.PENDING,
            created_at=datetime.utcnow()
        )
    
    @strawberry.mutation
    async def decompose_problem(
        self,
        input: DecomposeProblemInput,
        info: Info
    ) -> DecompositionPlan:
        """Decompose a problem into sub-problems."""
        try:
            from decomposition_engine import DecompositionEngine
            from sovereign_data_models import (
                ProblemDefinition, DomainContext, ProblemType as SPProblemType,
                ComplexityScore
            )
            
            engine = DecompositionEngine()
            
            # Map GraphQL enum to internal enum
            type_map = {
                ProblemType.RESEARCH: SPProblemType.RESEARCH,
                ProblemType.ANALYSIS: SPProblemType.ANALYSIS,
                ProblemType.IMPLEMENTATION: SPProblemType.IMPLEMENTATION,
                ProblemType.VALIDATION: SPProblemType.VALIDATION,
                ProblemType.INTEGRATION: SPProblemType.INTEGRATION,
                ProblemType.OPTIMIZATION: SPProblemType.OPTIMIZATION,
                ProblemType.DESIGN: SPProblemType.DESIGN,
            }
            
            problem = ProblemDefinition(
                id=f"problem_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                title=input.title,
                description=input.description,
                problem_type=type_map.get(input.problem_type, SPProblemType.ANALYSIS),
                domain_context=DomainContext(domain=input.domain or "general"),
                complexity_score=ComplexityScore(
                    overall_complexity=5.0,
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    explanation="Auto-generated from GraphQL request"
                )
            )
            
            strategy = input.strategy.value if input.strategy else "hybrid"
            plan = engine.decompose(problem, strategy=strategy)
            
            # Convert to GraphQL type
            return DecompositionPlan(
                id=plan.id,
                problem_id=plan.problem_id,
                strategy=DecompositionStrategy(plan.strategy.value),
                created_at=datetime.utcnow(),
                sub_problems=[
                    SubProblem(
                        id=sp.id,
                        title=sp.title,
                        description=sp.description,
                        type=ProblemType(sp.type.value),
                        priority=sp.priority,
                        estimated_effort=sp.estimated_effort,
                        dependencies=sp.dependencies
                    )
                    for sp in plan.sub_problems
                ],
                quality_score=plan.quality_scores.overall_score if hasattr(plan, 'quality_scores') else None
            )
            
        except Exception as e:
            logger.error(f"Error decomposing problem: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @strawberry.mutation
    async def extract_knowledge(
        self,
        input: ExtractKnowledgeInput,
        info: Info
    ) -> List[KnowledgeTriple]:
        """Extract knowledge from text."""
        try:
            # Would integrate with knowledge engine
            # For now return empty
            return []
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SUBSCRIPTIONS
# =============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription type for real-time updates."""
    
    @strawberry.subscription
    async def workflow_updates(
        self,
        workflow_id: str,
        info: Info
    ) -> AsyncGenerator[Workflow, None]:
        """Subscribe to workflow status updates."""
        try:
            from event_bus import get_event_bus, EventType, on_event
            
            bus = await get_event_bus()
            
            # Create a queue for events
            queue: asyncio.Queue = asyncio.Queue()
            
            async def handler(event):
                if event.workflow_id == workflow_id:
                    await queue.put(event)
            
            # Subscribe to workflow events
            await bus.subscribe(EventType.WORKFLOW_STARTED, handler)
            await bus.subscribe(EventType.WORKFLOW_COMPLETED, handler)
            await bus.subscribe(EventType.WORKFLOW_FAILED, handler)
            
            try:
                while True:
                    event = await queue.get()
                    # Yield workflow update
                    yield Workflow(
                        id=workflow_id,
                        problem_description="",
                        status=WorkflowStatus(event.payload.get("status", "running")),
                        created_at=event.timestamp
                    )
            finally:
                await bus.unsubscribe(handler)
                
        except Exception as e:
            logger.error(f"Error in subscription: {e}")
            raise
    
    @strawberry.subscription
    async def system_events(
        self,
        event_types: Optional[List[str]] = None,
        info: Info = None
    ) -> AsyncGenerator[Event, None]:
        """Subscribe to system events."""
        try:
            from event_bus import get_event_bus
            
            bus = await get_event_bus()
            queue: asyncio.Queue = asyncio.Queue()
            
            async def handler(event):
                if event_types is None or event.type.value in event_types:
                    await queue.put(event)
            
            # Subscribe to all event types
            for et in EventType:
                await bus.subscribe(et, handler)
            
            try:
                while True:
                    event = await queue.get()
                    yield Event(
                        id=event.id,
                        event_type=event.type.value,
                        source=event.source,
                        timestamp=event.timestamp,
                        payload=json.dumps(event.payload),
                        workflow_id=event.workflow_id
                    )
            finally:
                for et in EventType:
                    await bus.unsubscribe(handler, et)
                    
        except Exception as e:
            logger.error(f"Error in system events subscription: {e}")
            raise


# =============================================================================
# SCHEMA & APP
# =============================================================================

# Create Strawberry schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    extensions=[]
)

# Create FastAPI app
def create_graphql_app() -> FastAPI:
    """Create the GraphQL FastAPI application."""
    
    app = FastAPI(
        title="OpenEvolve GraphQL API",
        description="GraphQL API for workflow management, knowledge extraction, and decomposition",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount GraphQL endpoint
    from strawberry.asgi import GraphQL
    
    graphql_app = GraphQL(
        schema,
        graphql_ide="apollo-sandbox",  # Modern GraphQL IDE
        allow_queries_via_get=True,
        subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL]
    )
    
    app.add_route("/graphql", graphql_app)
    app.add_websocket_route("/graphql", graphql_app)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "graphql"}
    
    @app.get("/")
    async def root():
        """Root endpoint with links."""
        return {
            "service": "OpenEvolve GraphQL API",
            "graphql_endpoint": "/graphql",
            "graphql_ide": "/graphql (Apollo Sandbox)",
            "health": "/health"
        }
    
    return app


# Global app instance
graphql_app = create_graphql_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(graphql_app, host="0.0.0.0", port=8001)
