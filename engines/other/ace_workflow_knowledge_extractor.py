"""
ACE Workflow Knowledge Extractor Module

Extracts knowledge from workflows for ACE framework.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# ML Knowledge Extraction Integration
try:
    from ml_pattern_clustering import MLKnowledgeExtraction
    ML_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ML_KNOWLEDGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ACEWorkflowKnowledgeExtractor:
    """ACE Workflow Knowledge Extractor class"""

    def __init__(self):
        self.ml_extractor = None
        if ML_KNOWLEDGE_AVAILABLE:
            try:
                self.ml_extractor = MLKnowledgeExtraction()
                logger.info("ACE Workflow Knowledge Extractor initialized with ML capabilities")
            except Exception as e:
                logger.error(f"Failed to initialize ML extractor: {e}")
        else:
            logger.info("ACE Workflow Knowledge Extractor initialized (Basic Mode)")

    def extract(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge from workflow"""
        if not self.ml_extractor:
            return {"knowledge": {}, "workflow": workflow}
        
        try:
            # Extract combined problem/solution text
            text = f"Problem: {workflow.get('problem_statement', '')}\nSolution: {workflow.get('final_solution', '')}"
            
            # Use ML extractor
            extraction = self.ml_extractor.extract_from_text(
                text,
                domain=workflow.get('domain', 'general')
            )
            
            return {
                "knowledge": extraction,
                "workflow": workflow,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {"knowledge": {}, "workflow": workflow, "error": str(e)}


# Alias for compatibility
WorkflowKnowledgeExtractor = ACEWorkflowKnowledgeExtractor
