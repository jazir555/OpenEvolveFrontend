"""
Autonomous Knowledge Synthesis for OpenEvolve

This module provides recursive knowledge synthesis, transforming low-level facts
and entities into high-level structural insights and "Meta-Nodes".
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .core import EntityKnowledgeGraph
from .agentic_knowledge import AgenticKnowledgeEngine

logger = logging.getLogger(__name__)

class KnowledgeSynthesizer:
    """
    Synthesizes higher-level knowledge from clusters of low-level entities.
    """
    
    def __init__(self, engine: AgenticKnowledgeEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)

    async def synthesize_all(self) -> Dict[str, Any]:
        """
        Perform a full synthesis pass on the knowledge graph.
        """
        self.logger.info("Starting recursive knowledge synthesis...")
        
        # 1. Cluster Analysis (via Karate Club integration)
        # We use the existing analyze_knowledge_api logic
        graph_data = await self.engine.base.entity_graph.to_dict()
        formatted_graph = {
            "nodes": [{"id": k, "label": k, **v} for k, v in graph_data.get("entities", {}).items()],
            "edges": graph_data.get("relationships", [])
        }
        
        analysis = self.engine.enhanced.ai_integrator.analyze_graph_with_karateclub(
            formatted_graph, 
            {"community_detection": {"enabled": True, "algorithms": ["louvain"]}}
        )
        
        if analysis.get("status") != "success":
            return {"success": False, "message": "Clustering failed"}
            
        communities = analysis["analysis_results"].get("communities", {}).get("non_overlapping_louvain", {}).get("labels", [])
        
        if not communities:
            return {"success": False, "message": "No communities detected"}
            
        # Group node IDs by community label
        node_ids = list(graph_data["entities"].keys())
        clusters = {}
        for i, label in enumerate(communities):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node_ids[i])
            
        # 2. LLM-Based Synthesis
        meta_nodes_created = 0
        for label, ids in clusters.items():
            if len(ids) < 3: # Only synthesize clusters of significant size
                continue
                
            meta_node = await self._synthesize_cluster(ids)
            if meta_node:
                meta_nodes_created += 1
                
        await self.engine.save_all()
        
        return {
            "success": True,
            "clusters_processed": len(clusters),
            "meta_nodes_created": meta_nodes_created,
            "timestamp": datetime.now().isoformat()
        }

    async def _synthesize_cluster(self, entity_ids: List[str]) -> Optional[str]:
        """
        Synthesize a Meta-Node for a specific cluster of entities.
        Includes robust JSON parsing and retry logic.
        """
        # Collect entity data
        entities_data = []
        for eid in entity_ids:
            entity = await self.engine.base.entity_graph.get_entity(eid)
            if entity:
                # Sanitize content for LLM context length
                content_preview = str(entity.get("content", ""))[:200]
                entities_data.append({
                    "id": eid, 
                    "type": entity.get("type", "Unknown"),
                    "preview": content_preview,
                    "metadata": entity.get("metadata", {})
                })
                
        if not entities_data:
            return None
            
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Call LLM to synthesize
                prompt = f"""You are a Knowledge Architect. Analyze the following cluster of entities from a knowledge graph and synthesize a "Meta-Node" that explains their collective relationship, theme, or underlying architecture.

Entities:
{json.dumps(entities_data, indent=2)}

Return a JSON object:
{{
  "id": "A concise ID for the Meta-Node (e.g. 'auth_subsystem')",
  "label": "A descriptive name",
  "description": "A 2-3 sentence summary of what this cluster represents",
  "key_insights": ["Insight 1", "Insight 2"],
  "structural_role": "The role of this cluster in the overall system"
}}
"""
                response = await self.engine.base._call_llm(prompt, system_prompt="Synthesize architectural knowledge.")
                
                # Robust JSON cleaning
                clean_json = response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0]
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0]
                
                # Remove any leading/trailing non-json characters
                start_idx = clean_json.find('{')
                end_idx = clean_json.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    clean_json = clean_json[start_idx:end_idx+1]
                
                meta_data = json.loads(clean_json)
                
                # Validation of required keys
                required_keys = ["id", "label", "description", "key_insights", "structural_role"]
                if not all(key in meta_data for key in required_keys):
                    raise ValueError(f"LLM response missing keys: {set(required_keys) - set(meta_data.keys())}")
                
                meta_id = f"meta:{meta_data['id'].lower().replace(' ', '_')}"
                
                # Add Meta-Node to graph
                await self.engine.base.entity_graph.add_entity(meta_id, {
                    "type": "MetaNode",
                    "label": meta_data["label"],
                    "description": meta_data["description"],
                    "insights": meta_data["key_insights"],
                    "structural_role": meta_data["structural_role"],
                    "member_entities": entity_ids,
                    "effectiveness_score": 1.0,
                    "synthesized_at": datetime.now().isoformat(),
                    "source": "autonomous_synthesis"
                })
                
                # Link Meta-Node to all its members with specific edge types
                for eid in entity_ids:
                    await self.engine.base.entity_graph.add_relationship(meta_id, "abstracts", eid)
                    await self.engine.base.entity_graph.add_relationship(eid, "member_of", meta_id)
                    
                self.logger.info(f"Successfully synthesized Meta-Node: {meta_id}")
                return meta_id
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Attempt {attempt+1}: Failed to parse LLM response as JSON: {e}")
                if attempt == max_retries:
                    self.logger.error("Final synthesis attempt failed due to JSON error.")
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1}: Synthesis error: {e}")
                if attempt == max_retries:
                    break
                    
            # Small delay before retry
            await asyncio.sleep(1)
            
        return None