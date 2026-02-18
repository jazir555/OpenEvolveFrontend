"""
AI Knowledge Graph Bridge for OpenEvolve.

Bridges the core-projects/ai-knowledge-graph visualizer.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add core-project to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ai_kg_path = os.path.join(project_root, 'core-projects', 'ai-knowledge-graph')
if ai_kg_path not in sys.path:
    sys.path.insert(0, ai_kg_path)

logger = logging.getLogger(__name__)

class AIKnowledgeGraphBridge:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._available = False
        self._initialize()

    def _initialize(self):
        try:
            # Use import from src to match core-project structure
            from src.knowledge_graph.visualization import visualize_knowledge_graph
            self._visualize = visualize_knowledge_graph
            self._available = True
            logger.info("AI Knowledge Graph bridge initialized")
        except ImportError as e:
            logger.warning(f"AI Knowledge Graph not available: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def visualize(self, graph_data: Dict[str, Any], output_path: str = "graph.html") -> str:
        if not self._available:
            return "Visualizer not available"
        
        try:
            # Convert OpenEvolve graph to AI-KG format if necessary
            # For now assume compatible or minimal mapping
            self._visualize(graph_data, output_path)
            return output_path
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return str(e)
