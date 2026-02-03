"""
Production API Server for Knowledge Engine

Provides RESTful API endpoints for production deployments with:
- Health checks
- Knowledge processing
- Configuration management
- Monitoring and metrics
- Authentication (API key based)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# FastAPI imports with fallback
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, status
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Create dummy classes for type hints
    class BaseModel:
        pass
    class FastAPI:
        pass

from .health_monitor import get_health_monitor, quick_health_check
from .embedding_service import create_embedding_service
from .confidence_scorer import calculate_confidence
from .core.strategy_recommender_complete import recommend_strategy

logger = logging.getLogger(__name__)


# Request/Response Models
class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="all-MiniLM-L6-v2")


class EmbedResponse(BaseModel):
    success: bool
    embedding: List[float]
    dimensions: int
    model: str
    processing_time_ms: float


class ConfidenceRequest(BaseModel):
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(default="unknown")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ConfidenceResponse(BaseModel):
    success: bool
    confidence: float
    level: str
    factors: Dict[str, float]


class StrategyRequest(BaseModel):
    problem_description: str = Field(..., min_length=1, max_length=5000)
    domain: str = Field(default="general")


class StrategyResponse(BaseModel):
    success: bool
    strategy_name: str
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    overall_status: str
    components: List[Dict[str, Any]]
    timestamp: str
    version: str
    uptime_seconds: float


class ConfigResponse(BaseModel):
    version: str
    features: List[str]
    configuration: Dict[str, Any]


# API Key management
class APIKeyManager:
    """Simple API key manager for production use."""
    
    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment."""
        # In production, load from secure storage
        # For now, allow a default development key
        default_key = os.getenv("KE_API_KEY", "dev-key-not-for-production")
        self._keys[default_key] = {
            "name": "default",
            "rate_limit": 1000,
            "scopes": ["read", "write"]
        }
    
    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        return key in self._keys
    
    def get_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key."""
        return self._keys.get(key)


# Global instances
_api_key_manager = APIKeyManager()
_security = HTTPBearer(auto_error=False) if HAS_FASTAPI else None


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> str:
    """Verify API key from request."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    key = credentials.credentials
    if not _api_key_manager.validate_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return key


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not available. Install with: pip install fastapi uvicorn"
        )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Knowledge Engine API starting up")
        
        # Initialize services
        try:
            from .embedding_service import create_embedding_service
            app.state.embedding_service = create_embedding_service()
            logger.info("Embedding service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            app.state.embedding_service = None
        
        yield
        
        # Shutdown
        logger.info("Knowledge Engine API shutting down")
    
    app = FastAPI(
        title="OpenEvolve Knowledge Engine API",
        description="Production API for Knowledge Engine",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Health check endpoint (no auth required)
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Get system health status."""
        health = await quick_health_check()
        return health
    
    # Ready check (no auth required)
    @app.get("/ready")
    async def ready_check():
        """Check if service is ready to accept requests."""
        health = await quick_health_check()
        if health["overall_status"] == "healthy":
            return {"status": "ready"}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
    
    # Configuration endpoint
    @app.get("/config", response_model=ConfigResponse)
    async def get_config(api_key: str = Depends(verify_api_key)):
        """Get system configuration."""
        return ConfigResponse(
            version="2.0.0",
            features=[
                "embedding",
                "confidence_scoring",
                "strategy_recommendation",
                "health_monitoring",
                "cloud_storage"
            ],
            configuration={
                "embedding_models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                "storage_backends": ["memory", "postgresql", "qdrant"],
                "cloud_providers": ["s3", "gcs", "azure"]
            }
        )
    
    # Embedding endpoint
    @app.post("/embed", response_model=EmbedResponse)
    async def create_embedding(
        request: EmbedRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Generate embedding for text."""
        start_time = time.time()
        
        try:
            service = create_embedding_service(model_name=request.model)
            embedding = service.embed_text(request.text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbedResponse(
                success=True,
                embedding=embedding.tolist(),
                dimensions=len(embedding),
                model=request.model,
                processing_time_ms=processing_time
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}"
            )
    
    # Confidence scoring endpoint
    @app.post("/confidence", response_model=ConfidenceResponse)
    async def score_confidence(
        request: ConfidenceRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Calculate confidence score."""
        try:
            from .confidence_scorer import ConfidenceScorer
            
            scorer = ConfidenceScorer()
            confidence, factors = scorer.calculate_confidence(
                similarity_score=request.similarity_score,
                source=request.source,
                metadata=request.metadata or {}
            )
            
            return ConfidenceResponse(
                success=True,
                confidence=confidence,
                level=scorer.get_confidence_level(confidence),
                factors=factors.to_dict()
            )
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Confidence scoring failed: {str(e)}"
            )
    
    # Strategy recommendation endpoint
    @app.post("/strategy", response_model=StrategyResponse)
    async def recommend_processing_strategy(
        request: StrategyRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Recommend processing strategy for a problem."""
        try:
            rec = recommend_strategy(
                problem_description=request.problem_description,
                domain=request.domain
            )
            
            return StrategyResponse(
                success=True,
                strategy_name=rec.strategy_name,
                confidence=rec.confidence,
                reasoning=rec.reasoning,
                alternatives=[
                    {"name": name, "score": score}
                    for name, score in rec.alternatives
                ]
            )
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Strategy recommendation failed: {str(e)}"
            )
    
    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics(api_key: str = Depends(verify_api_key)):
        """Get system metrics (Prometheus format)."""
        health = await quick_health_check()
        
        # Convert to Prometheus format
        metrics = []
        metrics.append(f'# HELP ke_health Overall health status')
        metrics.append(f'# TYPE ke_health gauge')
        metrics.append(f'ke_health{{status="{health["overall_status"]}"}} 1')
        
        for component in health["components"]:
            name = component["name"]
            status_val = 1 if component["status"] == "healthy" else 0
            latency = component["latency_ms"]
            
            metrics.append(f'ke_component_health{{component="{name}"}} {status_val}')
            metrics.append(f'ke_component_latency_ms{{component="{name}"}} {latency}')
        
        return "\n".join(metrics)
    
    # Error handlers
    @app.exception_handler(Exception)
    async def generic_error_handler(request, exc):
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    return app


# Create app instance
app = None
try:
    if HAS_FASTAPI:
        app = create_app()
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Run the production API server."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not available. Install with: pip install fastapi uvicorn"
        )
    
    import uvicorn
    
    uvicorn.run(
        "knowledge_engine.production_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
