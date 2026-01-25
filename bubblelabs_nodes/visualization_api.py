"""
Visualization API Server

Provides HTTP endpoints for problem hierarchy visualization with
multiple output formats (ASCII, HTML, Graphviz/DOT).
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
import logging
from datetime import datetime

from .problem_visualization import (
    VisualizationAPI,
    OutputFormat,
    visualize_problem
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Problem Hierarchy Visualization API",
    description="API for visualizing problem decomposition hierarchies",
    version="1.0.0"
)

# Visualization API instance
viz_api = VisualizationAPI()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Problem Hierarchy Visualization API",
        "version": "1.0.0",
        "endpoints": {
            "visualize": "/api/visualize",
            "tree": "/api/problems/{id}/tree",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/visualize")
async def visualize_problem_endpoint(
    problem: Dict[str, Any],
    format: str = Query("ascii", description="Output format: ascii, html, or dot"),
    show_metadata: bool = Query(True, description="Include metadata in output"),
    show_timing: bool = Query(True, description="Include timing information"),
    show_teams: bool = Query(True, description="Include team history")
):
    """
    Visualize a problem hierarchy.

    Args:
        problem: Problem definition as JSON
        format: Output format (ascii, html, dot)
        show_metadata: Include metadata
        show_timing: Include timing
        show_teams: Include teams

    Returns:
        Visualization in requested format
    """
    try:
        # Validate format
        try:
            output_format = OutputFormat(format.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format '{format}'. Must be 'ascii', 'html', or 'dot'"
            )

        # Generate visualization
        result = viz_api.visualize_problem(
            problem,
            output_format,
            show_metadata,
            show_timing,
            show_teams
        )

        # Return based on format
        if output_format == OutputFormat.HTML:
            return HTMLResponse(content=result)
        elif output_format == OutputFormat.DOT:
            return PlainTextResponse(content=result, media_type="text/vnd.graphviz")
        else:
            return PlainTextResponse(content=result)

    except Exception as e:
        logger.error(f"Error visualizing problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/problems/{problem_id}/tree")
async def get_problem_tree(
    problem_id: str,
    format: str = Query("ascii", description="Output format: ascii, html, or dot"),
    show_metadata: bool = Query(True, description="Include metadata"),
    show_timing: bool = Query(True, description="Include timing"),
    show_teams: bool = Query(True, description="Include teams")
):
    """
    Get visualization for a specific problem by ID.

    Note: This is a placeholder implementation. In production, you would
    fetch the problem from a database or other storage.
    """
    try:
        # Validate format
        try:
            output_format = OutputFormat(format.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format '{format}'. Must be 'ascii', 'html', or 'dot'"
            )

        # TODO: Fetch problem from storage
        # For now, return a mock problem
        problem = {
            "id": problem_id,
            "status": "complete",
            "score": 85.0,
            "timing_ms": 1500,
            "teams": ["Blue", "Red", "Gold"],
            "attempt_count": 1,
            "subproblems": [
                {
                    "id": f"{problem_id}_sub1",
                    "status": "complete",
                    "score": 90.0,
                    "timing_ms": 500,
                    "teams": ["Blue", "Red"],
                    "subproblems": [
                        {"id": f"{problem_id}_sub1_a", "status": "complete", "score": 95.0},
                        {"id": f"{problem_id}_sub1_b", "status": "complete", "score": 85.0}
                    ]
                },
                {
                    "id": f"{problem_id}_sub2",
                    "status": "complete",
                    "score": 80.0,
                    "timing_ms": 1000,
                    "teams": ["Blue", "Red"]
                }
            ]
        }

        # Generate visualization
        result = viz_api.visualize_problem(
            problem,
            output_format,
            show_metadata,
            show_timing,
            show_teams
        )

        # Return based on format
        if output_format == OutputFormat.HTML:
            return HTMLResponse(content=result)
        elif output_format == OutputFormat.DOT:
            return PlainTextResponse(content=result, media_type="text/vnd.graphviz")
        else:
            return PlainTextResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting problem tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/problems/{problem_id}/tree/cached")
async def get_cached_tree(
    problem_id: str,
    format: str = Query("ascii", description="Output format")
):
    """
    Get cached visualization for a problem.

    This endpoint caches generated visualizations to improve performance
    for frequently accessed problems.
    """
    # TODO: Implement caching
    # For now, delegate to get_problem_tree
    return await get_problem_tree(problem_id, format)


@app.get("/api/formats")
async def get_supported_formats():
    """Get list of supported visualization formats"""
    return {
        "formats": [
            {
                "name": "ascii",
                "description": "ASCII art with box-drawing characters",
                "media_type": "text/plain"
            },
            {
                "name": "html",
                "description": "Interactive HTML with collapsible tree",
                "media_type": "text/html"
            },
            {
                "name": "dot",
                "description": "Graphviz DOT format for rendering graphs",
                "media_type": "text/vnd.graphviz"
            }
        ]
    }


# Cache for storing generated visualizations
_visualization_cache: Dict[str, tuple] = {}


async def get_cached_visualization(
    problem_id: str,
    format: OutputFormat,
    options: Dict[str, Any]
) -> Optional[str]:
    """
    Get cached visualization if available.

    Args:
        problem_id: Problem identifier
        format: Output format
        options: Visualization options

    Returns:
        Cached visualization or None
    """
    cache_key = f"{problem_id}:{format.value}:{hash(frozenset(options.items()))}"

    if cache_key in _visualization_cache:
        viz, timestamp = _visualization_cache[cache_key]

        # Check if cache is still valid (1 hour TTL)
        age = datetime.utcnow() - timestamp
        if age.total_seconds() < 3600:
            return viz
        else:
            # Remove stale cache entry
            del _visualization_cache[cache_key]

    return None


async def cache_visualization(
    problem_id: str,
    format: OutputFormat,
    options: Dict[str, Any],
    visualization: str
):
    """
    Cache a generated visualization.

    Args:
        problem_id: Problem identifier
        format: Output format
        options: Visualization options
        visualization: Generated visualization string
    """
    cache_key = f"{problem_id}:{format.value}:{hash(frozenset(options.items()))}"
    _visualization_cache[cache_key] = (visualization, datetime.utcnow())


async def clear_visualization_cache(problem_id: Optional[str] = None):
    """
    Clear visualization cache.

    Args:
        problem_id: Specific problem ID to clear, or None to clear all
    """
    if problem_id:
        # Clear all cache entries for this problem
        keys_to_remove = [
            key for key in _visualization_cache.keys()
            if key.startswith(f"{problem_id}:")
        ]
        for key in keys_to_remove:
            del _visualization_cache[key]
    else:
        # Clear entire cache
        _visualization_cache.clear()


def start_visualization_server(host: str = "0.0.0.0", port: int = 8001):
    """
    Start the visualization API server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn

    logger.info(f"Starting visualization API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Run server directly
    start_visualization_server()
