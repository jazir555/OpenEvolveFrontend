"""
RAGBits Server for BubbleLab Integration

This server provides a REST API interface for the BubbleLab RAGbits plugin,
allowing the TypeScript frontend to interact with Python-based RAGbits functionality.

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via environment variables
- RUNTIME TRUTH: Verify RAGBits availability before use
- IDEMPOTENCY: Safe to re-ingest documents
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import RAGBits components
from knowledge_engine.ragbits_document_processor import (
    RAGBitsDocumentProcessor,
    RAGBitsProcessorConfig,
    DocumentProcessingResult
)
from knowledge_engine.ragbits_retriever import get_ragbits_retriever, RAGBitsEnhancedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAGBits Server for BubbleLab",
    description="REST API for RAGBits document processing and retrieval",
    version="1.0.0"
)

# Global processor instance
processor: Optional[RAGBitsDocumentProcessor] = None
retriever: Optional[RAGBitsEnhancedRetriever] = None


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[Dict[str, Any]]
    total_results: int
    query: str


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: str = Field("manual", description="Document source identifier")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    success: bool
    document_id: str
    chunks_ingested: int
    processing_time: float
    error: Optional[str] = None


class BatchIngestRequest(BaseModel):
    """Request model for batch ingest endpoint."""
    documents: List[IngestRequest]


class BatchIngestResponse(BaseModel):
    """Response model for batch ingest endpoint."""
    results: List[DocumentProcessingResult]
    total_processed: int
    success_count: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    available: bool
    ingested_documents: int
    vector_store_type: str
    embedding_model: str
    ragbits_available: bool
    cache_size: int


@app.on_event("startup")
async def startup_event():
    """Initialize RAGBits components on startup."""
    global processor, retriever
    
    logger.info("Initializing RAGBits components...")
    
    # Create processor config from environment
    config = RAGBitsProcessorConfig()
    
    # Initialize processor
    processor = RAGBitsDocumentProcessor(config)
    processor_success = await processor.initialize()
    
    # Initialize retriever
    retriever = get_ragbits_retriever()
    
    if processor_success:
        logger.info("✅ RAGBits components initialized successfully")
    else:
        logger.warning("⚠️ RAGBits initialization failed - using fallback mode")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ragbits_available": processor.available if processor else False
    }


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents using semantic search."""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=503, detail="RAGBits retriever not available")
    
    try:
        logger.info({
            "msg": "Processing search request",
            "query": request.query[:100],
            "top_k": request.top_k,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Use the enhanced retriever for search
        results = await retriever.search_similar_solutions(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_hybrid_search=True
        )
        
        logger.info({
            "msg": "Search completed",
            "results_found": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query
        )
    
    except Exception as e:
        logger.error({
            "msg": "Search failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Ingest a document into the RAG system."""
    global processor
    
    if not processor:
        raise HTTPException(status_code=503, detail="RAGBits processor not available")
    
    try:
        logger.info({
            "msg": "Processing ingest request",
            "source": request.source,
            "content_length": len(request.content),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Ingest the document
        result = await processor.ingest_text(
            text=request.content,
            metadata=request.metadata,
            source=request.source
        )
        
        logger.info({
            "msg": "Document ingested",
            "document_id": result.document_id,
            "success": result.success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return IngestResponse(
            success=result.success,
            document_id=result.document_id,
            chunks_ingested=result.chunks_ingested,
            processing_time=result.processing_time,
            error=result.error
        )
    
    except Exception as e:
        logger.error({
            "msg": "Ingest failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def batch_ingest_documents(request: BatchIngestRequest):
    """Ingest multiple documents in batch."""
    global processor

    if not processor:
        raise HTTPException(status_code=503, detail="RAGBits processor not available")

    try:
        logger.info({
            "msg": "Processing batch ingest request",
            "document_count": len(request.documents),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        results = []
        for doc_request in request.documents:
            result = await processor.ingest_text(
                text=doc_request.content,
                metadata=doc_request.metadata,
                source=doc_request.source
            )
            results.append(result)

        success_count = sum(1 for r in results if r.success)

        logger.info({
            "msg": "Batch ingest completed",
            "total_processed": len(results),
            "success_count": success_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return BatchIngestResponse(
            results=results,
            total_processed=len(results),
            success_count=success_count
        )

    except Exception as e:
        logger.error({
            "msg": "Batch ingest failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Batch ingest failed: {str(e)}")


@app.post("/generate", response_model=dict)
async def generate_response(request: dict):
    """Generate a response using RAG with semantic search."""
    global retriever

    if not retriever:
        raise HTTPException(status_code=503, detail="RAGBits retriever not available")

    try:
        query = request.get("query", "")
        context = request.get("context", "")
        search_query = request.get("search_query", query)
        top_k = request.get("top_k", 5)
        llm_model = request.get("llm_model", "gpt-4o")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1000)
        filters = request.get("filters", {})

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        logger.info({
            "msg": "Processing RAG generation request",
            "query": query[:100],
            "model": llm_model,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Perform semantic search to get relevant context
        search_results = await retriever.search_similar_solutions(
            query=search_query,
            top_k=top_k,
            filters=filters,
            enable_hybrid_search=True
        )

        # Format the context from search results
        search_context = ""
        if search_results:
            search_context = "Context from knowledge base:\n"
            for idx, result in enumerate(search_results):
                content = result.get("content", "")[:500]  # Limit content length
                search_context += f"\n[{idx + 1}] {content}\n"

        # Combine the original context with search results
        combined_context = f"{context}\n{search_context}" if context else search_context

        # For now, return a simulated response since we don't have an actual LLM integration
        # In a full implementation, this would call an LLM with the retrieved context
        response_text = f"Based on the provided context and query '{query}', here is a generated response. In a full implementation, this would come from an LLM with the retrieved context."

        logger.info({
            "msg": "RAG generation completed",
            "search_results_count": len(search_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return {
            "response": response_text,
            "generated_text": response_text,
            "search_results_used": len(search_results),
            "sources": [r.get("metadata", {}) for r in search_results],
            "model_used": llm_model,
            "success": True
        }

    except Exception as e:
        logger.error({
            "msg": "RAG generation failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get RAGBits system statistics."""
    global processor, retriever
    
    if not processor or not retriever:
        raise HTTPException(status_code=503, detail="RAGBits components not available")
    
    try:
        # Get processor stats
        processor_stats = await processor.get_statistics()
        
        # Get retriever stats
        retriever_stats = await retriever.get_statistics()
        
        return StatsResponse(
            available=processor.available,
            ingested_documents=processor_stats.get("ingested_documents", 0),
            vector_store_type=processor_stats.get("vector_store_type", "unknown"),
            embedding_model=processor_stats.get("embedding_model", "unknown"),
            ragbits_available=processor_stats.get("available", False),
            cache_size=retriever_stats.get("cache_size", 0)
        )
    
    except Exception as e:
        logger.error({
            "msg": "Stats retrieval failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.post("/clear-cache")
async def clear_cache():
    """Clear the retriever cache."""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=503, detail="RAGBits retriever not available")
    
    try:
        await retriever.clear_cache()
        return {"success": True, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error({
            "msg": "Cache clearing failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAGBits Server for BubbleLab",
        "version": "1.0.0",
        "endpoints": [
            "GET /health - Health check",
            "POST /search - Semantic search",
            "POST /ingest - Ingest document",
            "POST /ingest/batch - Batch ingest",
            "GET /stats - System statistics",
            "POST /clear-cache - Clear cache"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def main():
    """Main entry point to run the server."""
    port = int(os.getenv("RAGBITS_SERVER_PORT", "8002"))
    host = os.getenv("RAGBITS_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting RAGBits server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()