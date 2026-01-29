import uuid
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from .core import EntityKnowledgeGraph
except ImportError:
    from core import EntityKnowledgeGraph

logger = logging.getLogger(__name__)

class ResearchQuestEngine:
    """
    Implements the Systematic Scientific Reasoning logic (ASR-GoT).
    Manages the 8-stage research graph evolution.
    """
    def __init__(self):
        self.graphs: Dict[str, EntityKnowledgeGraph] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("ResearchQuestEngine initialized")

    async def initialize_graph(self, task_description: str, config: Dict[str, Any] = None) -> str:
        graph_id = f"rq-{uuid.uuid4().hex[:8]}"
        graph = EntityKnowledgeGraph()
        
        # Stage 1: Initialization
        await graph.add_entity("n0", {
            "label": "Task Understanding",
            "description": task_description,
            "type": "root",
            "stage": 1,
            "confidence": config.get("initial_confidence", [0.8, 0.8, 0.8, 0.8]) if config else [0.8]*4
        })
        
        self.graphs[graph_id] = graph
        self.metadata[graph_id] = {
            "task": task_description,
            "current_stage": 1,
            "stage_name": "Initialization",
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Initialized research graph {graph_id} for task: {task_description[:50]}...")
        return graph_id

    async def decompose_task(self, graph_id: str, dimensions: List[str]) -> List[str]:
        if graph_id not in self.graphs:
            logger.error(f"Attempted decomposition on non-existent graph: {graph_id}")
            return []
        
        graph = self.graphs[graph_id]
        new_nodes = []
        for i, dim in enumerate(dimensions):
            node_id = f"2.{i+1}"
            await graph.add_entity(node_id, {
                "label": dim,
                "type": "dimension",
                "stage": 2,
                "confidence": [0.7, 0.7, 0.7, 0.7]
            })
            await graph.add_relationship("n0", "decomposes_into", node_id)
            new_nodes.append(node_id)
        
        self.metadata[graph_id]["current_stage"] = 2
        self.metadata[graph_id]["stage_name"] = "Decomposition"
        logger.info(f"Decomposed task in graph {graph_id} into {len(dimensions)} dimensions")
        return new_nodes

    async def generate_hypotheses(self, graph_id: str, dimension_id: str, hypotheses: List[Dict[str, Any]]) -> List[str]:
        if graph_id not in self.graphs:
            logger.error(f"Attempted hypothesis generation on non-existent graph: {graph_id}")
            return []
        
        graph = self.graphs[graph_id]
        new_nodes = []
        for i, h in enumerate(hypotheses):
            node_id = f"{dimension_id}.{i+1}"
            await graph.add_entity(node_id, {
                "label": h.get("label", f"Hypothesis {i+1}"),
                "description": h.get("description", ""),
                "type": "hypothesis",
                "stage": 3,
                "confidence": h.get("confidence", [0.6, 0.6, 0.6, 0.6]),
                "falsification_criteria": h.get("falsification_criteria", "")
            })
            await graph.add_relationship(dimension_id, "supports_hypothesis", node_id)
            new_nodes.append(node_id)
            
        self.metadata[graph_id]["current_stage"] = 3
        self.metadata[graph_id]["stage_name"] = "Hypothesis Planning"
        logger.info(f"Generated {len(hypotheses)} hypotheses for dimension {dimension_id} in graph {graph_id}")
        return new_nodes

    async def get_summary(self, graph_id: str) -> Dict[str, Any]:
        if graph_id not in self.graphs:
            return {
                "success": False,
                "message": "Graph not found",
                "current_stage": 0,
                "stage_name": "Unknown"
            }
        
        graph = self.graphs[graph_id]
        meta = self.metadata[graph_id]
        
        # We need to await to_dict since we made it async in core.py
        graph_data = await graph.to_dict()
        
        return {
            "success": True,
            "node_id": graph_id,
            "message": f"Graph summary for {graph_id}",
            "current_stage": meta["current_stage"],
            "stage_name": meta["stage_name"],
            "graph_summary": {
                "vertices_count": len(graph_data.get("entities", {})),
                "edges_count": len(graph_data.get("relationships", [])),
                "active_parameters": ["P1.0", "P1.1", "P1.2", "P1.3"]
            }
        }

    async def export_data(self, graph_id: str, format: str = "json") -> str:
        if graph_id not in self.graphs:
            return ""
        
        graph = self.graphs[graph_id]
        graph_data = await graph.to_dict()
        
        if format == "json":
            return json.dumps({
                "metadata": self.metadata[graph_id],
                "graph": graph_data
            }, indent=2)
        return ""