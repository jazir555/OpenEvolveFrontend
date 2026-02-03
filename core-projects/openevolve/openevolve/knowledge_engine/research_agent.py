"""
Deep Research Agent Integration for OpenEvolve

Enables the Knowledge Engine to perform deep, multi-agent research on complex 
topics and ingest the findings back into the Knowledge Base.
"""

import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add deep-research-agent to path
RESEARCH_AGENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deep-research-agent")
if os.path.exists(RESEARCH_AGENT_PATH) and RESEARCH_AGENT_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_AGENT_PATH)

try:
    from src.graph import run_research
    RESEARCH_AGENT_AVAILABLE = True
except ImportError:
    RESEARCH_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeepResearchIntegration:
    """
    Wraps the Deep Research Agent for use in the Knowledge Engine.
    """
    
    def __init__(self, engine: Any):
        self.engine = engine # AgenticKnowledgeEngine
        self.available = RESEARCH_AGENT_AVAILABLE
        logger.info(f"DeepResearchIntegration initialized (Available: {self.available})")

    async def perform_deep_research(self, topic: str) -> Dict[str, Any]:
        """
        Run a deep research workflow and ingest results.
        """
        if not self.available:
            return {"success": False, "message": "Deep Research Agent not available"}
            
        logger.info(f"🚀 Starting deep research on: {topic}")
        
        try:
            # 1. Run Research
            # Ensure we are in the correct directory context if needed
            final_state = await run_research(topic, verbose=True)
            
            if final_state.get("error"):
                return {"success": False, "message": final_state["error"]}
                
            # 2. Ingest Results
            report = final_state.get("final_report", "")
            findings = final_state.get("key_findings", [])
            
            # Add report as a major entity
            report_id = f"research:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.engine.base.entity_graph.add_entity(report_id, {
                "type": "ResearchReport",
                "topic": topic,
                "content": report,
                "findings": findings,
                "source": "DeepResearchAgent",
                "timestamp": datetime.now().isoformat(),
                "effectiveness_score": 1.0
            })
            
            # Add findings as sub-entities
            for i, finding in enumerate(findings):
                finding_id = f"{report_id}:finding_{i}"
                await self.engine.base.entity_graph.add_entity(finding_id, {
                    "type": "ResearchFinding",
                    "content": finding,
                    "parent_report": report_id
                })
                await self.engine.base.entity_graph.add_relationship(report_id, "contains_finding", finding_id)
                
            await self.engine.save_all()
            
            return {
                "success": True,
                "report_id": report_id,
                "findings_count": len(findings),
                "report_preview": report[:500] + "..."
            }
            
        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            return {"success": False, "message": str(e)}
