"""
Visualization API Server

Provides HTTP endpoints for problem hierarchy visualization with
multiple output formats (ASCII, HTML, Graphviz/DOT).
"""

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
import logging
from datetime import datetime
import json
import sys
from pathlib import Path

from .problem_visualization import (
    VisualizationAPI,
    OutputFormat,
    visualize_problem
)

# Add parent directory to path for sovereign imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sovereign_database import SovereignDatabase
    from sovereign_data_models import ProblemDefinition, SubProblem
    SOVEREIGN_AVAILABLE = True
except ImportError:
    SOVEREIGN_AVAILABLE = False
    logging.warning("Sovereign database not available, using mock data")

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Problem Hierarchy Visualization API",
    description="API for visualizing problem decomposition hierarchies",
    version="1.0.0"
)

# Visualization API instance
viz_api = VisualizationAPI()

# Database instance (lazy initialization)
_db_instance: Optional[SovereignDatabase] = None


def get_database() -> Optional[SovereignDatabase]:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None and SOVEREIGN_AVAILABLE:
        try:
            _db_instance = SovereignDatabase()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    return _db_instance


def build_problem_hierarchy(problem_id: str, db: SovereignDatabase) -> Optional[Dict[str, Any]]:
    """
    Build a complete problem hierarchy from the database.
    
    Args:
        problem_id: The problem ID to fetch
        db: Database instance
        
    Returns:
        Problem hierarchy dict or None if not found
    """
    problem = db.get_problem(problem_id)
    if not problem:
        return None
    
    def build_subproblem_tree(parent_id: str) -> List[Dict[str, Any]]:
        """Recursively build subproblem tree."""
        subproblems = db.list_sub_problems_by_parent(parent_id)
        result = []
        
        for sp in subproblems:
            sp_data = {
                "id": sp.id,
                "status": sp.status.value if hasattr(sp.status, 'value') else str(sp.status),
                "score": _calculate_subproblem_score(sp),
                "timing_ms": getattr(sp, 'timing_ms', 0) or 0,
                "teams": [sp.assigned_team] if sp.assigned_team else [],
            }
            
            # Recursively get child subproblems
            children = build_subproblem_tree(sp.id)
            if children:
                sp_data["subproblems"] = children
                
            result.append(sp_data)
        
        return result
    
    # Build the problem hierarchy
    hierarchy = {
        "id": problem.id,
        "status": "complete" if all(
            sp.status.value == 'solved' if hasattr(sp.status, 'value') else str(sp.status) == 'solved'
            for sp in db.list_sub_problems_by_parent(problem.id)
        ) else "in_progress" if db.list_sub_problems_by_parent(problem.id) else "pending",
        "score": _calculate_problem_score(problem, db),
        "timing_ms": _calculate_problem_timing(problem, db),
        "teams": list(set(
            sp.assigned_team for sp in db.list_sub_problems_by_parent(problem.id)
            if sp.assigned_team
        )),
        "attempt_count": len(problem.metadata.get('solution_attempts', [])),
        "subproblems": build_subproblem_tree(problem.id)
    }
    
    return hierarchy


def _calculate_subproblem_score(subproblem: SubProblem) -> float:
    """Calculate a score for a subproblem based on its status and attempts."""
    base_scores = {
        'solved': 95.0,
        'in_progress': 70.0,
        'pending': 50.0,
        'blocked': 40.0,
        'failed': 20.0,
        'error': 10.0,
    }
    status_val = subproblem.status.value if hasattr(subproblem.status, 'value') else str(subproblem.status)
    return base_scores.get(status_val, 50.0)


def _calculate_problem_score(problem: ProblemDefinition, db: SovereignDatabase) -> float:
    """Calculate overall problem score from subproblems."""
    subproblems = db.list_sub_problems_by_parent(problem.id)
    if not subproblems:
        return 85.0
    
    scores = [_calculate_subproblem_score(sp) for sp in subproblems]
    return sum(scores) / len(scores)


def _calculate_problem_timing(problem: ProblemDefinition, db: SovereignDatabase) -> int:
    """Calculate total timing for problem based on subproblems."""
    subproblems = db.list_sub_problems_by_parent(problem.id)
    if not subproblems:
        return 1500
    
    # Sum estimated effort from all subproblems (in ms for consistency)
    total_effort = sum(getattr(sp, 'estimated_effort', 1) or 1 for sp in subproblems)
    return total_effort * 100


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

        # Fetch problem from storage if available
        problem = None
        db = get_database()
        
        if db and SOVEREIGN_AVAILABLE:
            try:
                problem = build_problem_hierarchy(problem_id, db)
            except Exception as e:
                logger.warning(f"Failed to fetch problem from database: {e}")
        
        # Fall back to mock data if database fetch failed or unavailable
        if problem is None:
            logger.info(f"Using mock data for problem {problem_id}")
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
    for frequently accessed problems. Cache entries expire after 1 hour.
    """
    try:
        # Convert format string to OutputFormat enum
        try:
            output_format = OutputFormat(format.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid format: {format}. Supported: ascii, html, dot"
            )
        
        # Prepare options for cache key
        options = {"format": format}
        
        # Try to get from cache first
        cached_result = await get_cached_visualization(problem_id, output_format, options)
        
        if cached_result is not None:
            logger.info(f"Cache hit for problem {problem_id} with format {format}")
            
            # Return appropriate response based on format
            if format.lower() == "html":
                return HTMLResponse(content=cached_result)
            elif format.lower() == "dot":
                return PlainTextResponse(
                    content=cached_result, 
                    media_type="text/vnd.graphviz"
                )
            else:
                return PlainTextResponse(content=cached_result)
        
        # Cache miss - generate the visualization
        logger.info(f"Cache miss for problem {problem_id} with format {format}")
        
        # Generate visualization using the main endpoint logic
        result = await get_problem_tree(problem_id, format)
        
        # Cache the result for future requests
        # Note: We need to extract the actual content from the response
        if isinstance(result, (HTMLResponse, PlainTextResponse)):
            visualization_content = result.body.decode('utf-8') if hasattr(result, 'body') else str(result)
        else:
            visualization_content = str(result)
        
        await cache_visualization(problem_id, output_format, options, visualization_content)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cached tree retrieval: {e}")
        # Fall back to non-cached version
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
