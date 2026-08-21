"""
BubbleLabs Ragbits Bubbles for OpenEvolve

This module provides BubbleLab workflow nodes (bubbles) specifically designed for 
RAG (Retrieval-Augmented Generation) operations using the Ragbits system.

Bubble Types Supported:
- RAGBits Ingest (Document Ingestion)
- RAGBits Search (Semantic Retrieval)
- RAGBits Generation (Context-Aware Generation)
- RAGBits Index (Knowledge Base Management)

Usage:
    from bubblelabs_ragbits_bubbles import (
        create_ragbits_ingest_bubble,
        create_ragbits_search_bubble,
        create_ragbits_generation_bubble,
        create_ragbits_index_bubble,
        create_rag_workflow_definition
    )
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# =============================================================================
# Bubble Configuration Constants
# =============================================================================

RAGBITS_NODE_POSITIONS = {
    "ingest": {"x": 0, "y": 0},
    "search": {"x": 250, "y": 0},
    "generation": {"x": 500, "y": 0},
    "index": {"x": 250, "y": 150},
}

RAGBITS_NODE_COLORS = {
    "ingest": "#A8E6CF",      # Light Green
    "search": "#D4A5A5",      # Rose
    "generation": "#FFD3B6",  # Peach
    "index": "#DCEDC1",       # Pale Green
}

RAGBITS_NODE_ICONS = {
    "ingest": "📥",
    "search": "🔍",
    "generation": "🤖",
    "index": "📚",
}

# =============================================================================
# Bubble Creation Functions
# =============================================================================

def create_ragbits_ingest_bubble(
    label: str = "Ragbits Ingest",
    source_type: str = "text",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Ragbits ingestion bubble.
    
    Args:
        label: Display label
        source_type: Type of source (text, file, url, directory)
        position: Optional position
        
    Returns:
        Node dictionary
    """
    position = position or RAGBITS_NODE_POSITIONS["ingest"]
    
    return {
        "id": f"ragbits_ingest_{uuid.uuid4().hex[:8]}",
        "type": "ragbits_ingest",
        "position": position,
        "data": {
            "label": f"{RAGBITS_NODE_ICONS['ingest']} {label}",
            "source_type": source_type,
            "description": f"Ingest knowledge from {source_type}",
            "status": "pending",
            "node_color": RAGBITS_NODE_COLORS["ingest"],
            "parameters": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "auto_index": True
            }
        }
    }

def create_ragbits_search_bubble(
    label: str = "Ragbits Search",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Ragbits semantic search bubble.
    """
    position = position or RAGBITS_NODE_POSITIONS["search"]
    
    return {
        "id": f"ragbits_search_{uuid.uuid4().hex[:8]}",
        "type": "ragbits_search",
        "position": position,
        "data": {
            "label": f"{RAGBITS_NODE_ICONS['search']} {label}",
            "description": "Semantic search across knowledge base",
            "status": "pending",
            "node_color": RAGBITS_NODE_COLORS["search"],
            "parameters": {
                "top_k": 5,
                "min_score": 0.7,
                "include_metadata": True
            }
        }
    }

def create_ragbits_generation_bubble(
    label: str = "Ragbits Generation",
    model: str = "gpt-4o",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Ragbits context-aware generation bubble.
    """
    position = position or RAGBITS_NODE_POSITIONS["generation"]
    
    return {
        "id": f"ragbits_generation_{uuid.uuid4().hex[:8]}",
        "type": "ragbits_generation",
        "position": position,
        "data": {
            "label": f"{RAGBITS_NODE_ICONS['generation']} {label}",
            "description": "Generate response using retrieved context",
            "status": "pending",
            "node_color": RAGBITS_NODE_COLORS["generation"],
            "parameters": {
                "model": model,
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant. Use the provided context to answer the user's question."
            }
        }
    }

def create_ragbits_index_bubble(
    label: str = "Ragbits Index",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Ragbits index management bubble.
    """
    position = position or RAGBITS_NODE_POSITIONS["index"]
    
    return {
        "id": f"ragbits_index_{uuid.uuid4().hex[:8]}",
        "type": "ragbits_index",
        "position": position,
        "data": {
            "label": f"{RAGBITS_NODE_ICONS['index']} {label}",
            "description": "Manage vector index and statistics",
            "status": "pending",
            "node_color": RAGBITS_NODE_COLORS["index"],
            "parameters": {
                "operation": "stats",  # stats, clear, refresh, optimize
                "vector_store": "memory"
            }
        }
    }

# =============================================================================
# Workflow Creation
# =============================================================================

def create_rag_workflow_definition(
    name: str,
    problem_statement: str,
    source_type: str = "text"
) -> Dict[str, Any]:
    """
    Create a standard RAG workflow definition.
    
    Workflow: Ingest -> Search -> Generation
    """
    nodes = []
    edges = []
    
    # Create bubbles
    ingest = create_ragbits_ingest_bubble(source_type=source_type)
    search = create_ragbits_search_bubble()
    gen = create_ragbits_generation_bubble()
    index = create_ragbits_index_bubble()
    
    nodes.extend([ingest, search, gen, index])
    
    # Create edges
    from .bubblelabs_gauntlet_bubbles import create_bubble_edge
    
    edges.append(create_bubble_edge(ingest["id"], search["id"]))
    edges.append(create_bubble_edge(search["id"], gen["id"]))
    
    # Index is standalone or connected to ingest
    edges.append(create_bubble_edge(ingest["id"], index["id"], edge_type="default"))
    
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": problem_statement,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "workflow_type": "rag_workflow",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
    }
