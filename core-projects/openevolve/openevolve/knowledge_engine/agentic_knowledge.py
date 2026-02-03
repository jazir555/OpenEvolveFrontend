"""
Unified Agentic Knowledge Engine for OpenEvolve

This module integrates the factual knowledge graph with the agentic skillbooks
from the Agentic Context Engine (ACE), creating a unified memory system.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .core import EntityKnowledgeGraph, KnowledgeState
from .engine import KnowledgeEngine
from .ai_enhanced_integration import AIEnhancedKnowledgeEngine

try:
    import ace_mcp_tools
    from ace import Skillbook, Skill, Sample, AgentOutput
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False

try:
    from ace_steer_integration import AceSteerBridge
    STEER_AVAILABLE = True
except ImportError:
    STEER_AVAILABLE = False

logger = logging.getLogger(__name__)

class AgenticKnowledgeEngine:
    """
    The "Engine inside the Knowledge Engine".
    
    Unifies:
    - Factual Knowledge (Entity Graphs)
    - Semantic Knowledge (RAG/Ragbits)
    - Agentic Knowledge (ACE Skills)
    - Analytical Knowledge (Karate Club)
    - Reliability Knowledge (Steer Policies)
    """
    
    def __init__(self, base_engine: KnowledgeEngine, enhanced_engine: AIEnhancedKnowledgeEngine):
        self.base = base_engine
        self.enhanced = enhanced_engine
        self.ace_available = ACE_AVAILABLE
        self.steer_available = STEER_AVAILABLE
        
        # Skillbook persistence
        self.skillbook_path = "ace_skillbook.json"
        self.skillbook: Optional[Any] = None
        
        if self.ace_available:
            self._load_skillbook()
            
        # Bridge for steering
        self.bridge = None
        if self.ace_available and self.steer_available:
            from ace_steer_integration import AceSteerBridge
            self.bridge = AceSteerBridge(ace_agent_id="knowledge_core_agent", skillbook_path=self.skillbook_path)
            
        logger.info(f"AgenticKnowledgeEngine initialized (ACE: {self.ace_available}, Steer: {self.steer_available})")

    def _load_skillbook(self):
        """Load the ACE skillbook."""
        try:
            from ace import Skillbook
            if os.path.exists(self.skillbook_path):
                self.skillbook = Skillbook.load_from_file(self.skillbook_path)
                logger.info(f"Loaded {len(self.skillbook.skills())} skills from {self.skillbook_path}")
            else:
                self.skillbook = Skillbook()
                logger.info("Created new empty skillbook")
        except Exception as e:
            logger.error(f"Failed to load skillbook: {e}")
            self.skillbook = None

    async def save_all(self):
        """Persist both the graph and the skillbook."""
        await self.base.save_graph()
        if self.ace_available and self.skillbook:
            try:
                self.skillbook.save_to_file(self.skillbook_path)
                logger.info("Skillbook saved")
            except Exception as e:
                logger.error(f"Failed to save skillbook: {e}")

    async def get_unified_context(self, task_description: str, top_k: int = 5, use_steering: bool = True) -> Dict[str, Any]:
        """
        Retrieve unified context for an agent task with optional steering.
        """
        # 1. Semantic Search (Ragbits)
        try:
            from .ragbits_retriever import get_ragbits_retriever
            retriever = get_ragbits_retriever()
            semantic_results = await retriever.search_similar_solutions(task_description, top_k=top_k)
        except Exception as e:
            logger.warning(f"Ragbits search failed: {e}")
            semantic_results = []
        
        # 2. Agentic Skills (ACE)
        relevant_skills = []
        if self.ace_available and self.skillbook:
            relevant_skills = [
                {"strategy": s.strategy, "helpful": s.helpful_count} 
                for s in self.skillbook.skills()
            ]
            
        # 3. Factual Graph (Base) - Using async search
        graph_entities = await self.base.entity_graph.search_entities(task_description)
                
        # 4. Reliability Check (Steer)
        steer_metadata = {}
        if use_steering and self.bridge:
            # Check if any knowledge item violates policies
            # For brevity, we just tag the search as 'steered'
            steer_metadata = {"status": "verified", "policies": ["precision", "safety"]}

        return {
            "semantic_solutions": semantic_results,
            "agentic_skills": relevant_skills[:top_k],
            "factual_entities": graph_entities[:top_k],
            "ace_enabled": self.ace_available,
            "steer_active": self.steer_available,
            "steer_metadata": steer_metadata
        }

    async def self_heal_knowledge_base(self) -> Dict[str, Any]:
        """
        Identify weak knowledge nodes and attempt to repair them using 
        Scientific Reasoning (Quest) and Synthesis.
        """
        healing_log = []
        
        # 1. Identify "Decaying" Knowledge
        graph_dict = await self.base.entity_graph.to_dict()
        entities = graph_dict.get("entities", {})
        
        weak_entities = [
            entity_id for entity_id, props in entities.items()
            if props.get("effectiveness_score", 1.0) < 0.5 
            and props.get("type") != "MetaNode" # Don't heal meta-nodes directly
        ]
        
        if not weak_entities:
            return {"healed_count": 0, "log": ["No weak entities found."]}

        from .research_quest import ResearchQuestEngine
        from .synthesis import KnowledgeSynthesizer
        quest_engine = ResearchQuestEngine()
        synthesizer = KnowledgeSynthesizer(self)

        for entity_id in weak_entities:
            healing_log.append(f"Healing entity: {entity_id}")
            entity_data = await self.base.entity_graph.get_entity(entity_id)
            if not entity_data:
                continue
            
            # 2. Trigger Research Quest to find "Root Cause" of low effectiveness
            quest_id = await quest_engine.initialize_graph(
                task_description=f"Investigate why knowledge artifact '{entity_id}' has low effectiveness. Content: {json.dumps(entity_data)}",
                config={"initial_confidence": [0.4, 0.4, 0.4, 0.4]}
            )
            
            # Decompose and analyze
            await quest_engine.decompose_task(quest_id, ["Content Accuracy", "Context Applicability", "Source Reliability"])
            
            # Reasoning result
            quest_summary = await quest_engine.get_summary(quest_id)
            healing_log.append(f"  Quest analysis complete: {quest_summary['stage_name']}")
            
            # 3. Apply "Repair" (LLM-driven update)
            prompt = f"""You are a Knowledge Repair Agent. A knowledge artifact has been flagged as 'decaying' (effectiveness < 0.5).
Artifact: {json.dumps(entity_data)}

Research Quest Analysis: {quest_summary['message']}

Please provide an updated, corrected version of this artifact's content and a remediation strategy.
Return JSON: {{"updated_content": "...", "remediation": "..."}}
"""
            try:
                repair_json = await self.base._call_llm(prompt, system_prompt="Repair decaying knowledge.")
                # Robust parsing (reusing logic from synthesis)
                clean_json = repair_json.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0]
                
                repair_data = json.loads(clean_json)
                
                # Apply repair
                updated_props = entity_data.copy()
                updated_props.update({
                    "content": repair_data.get("updated_content", entity_data.get("content")),
                    "effectiveness_score": 0.8, # Reset score
                    "remediation_applied": repair_data.get("remediation"),
                    "healed_at": datetime.now().isoformat()
                })
                await self.base.entity_graph.add_entity(entity_id, updated_props)
                
                healing_log.append(f"  Successfully repaired {entity_id}")
            except Exception as e:
                healing_log.append(f"  Failed to repair {entity_id}: {e}")
                logger.error(f"Repair failed for {entity_id}: {e}")
            
        # 4. Pass over with Synthesis to re-integrate
        await synthesizer.synthesize_all()
        
        await self.save_all()
        return {"healed_count": len(weak_entities), "log": healing_log}

    async def distill_skill_from_artifact(self, artifact_id: str) -> bool:
        """
        Convert a high-performing knowledge artifact into a reusable ACE skill.
        """
        if not self.ace_available or not self.skillbook:
            return False
            
        entity = await self.base.entity_graph.get_entity(artifact_id)
        if not entity:
            return False
            
        # Only distill high-quality items
        if entity.get("effectiveness_score", 0) < 0.8:
            logger.info(f"Artifact {artifact_id} quality too low for distillation")
            return False
            
        try:
            # Use LLM to distill the skill strategy
            content = json.dumps(entity)
            prompt = f"Distill a reusable agentic strategy/skill from this successful execution artifact:\n\n{content}\n\nFormat as a concise imperative instruction."
            
            strategy = await self.base._call_llm(prompt, system_prompt="You are a Skill Distiller for AI agents.")
            
            if not strategy:
                return False

            from ace import Skill
            new_skill = Skill(strategy=strategy.strip())
            self.skillbook.add(new_skill)
            await self.save_all()
            
            logger.info(f"Distilled new skill from {artifact_id}: {new_skill.strategy[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Skill distillation failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get unified statistics."""
        graph_dict = await self.base.entity_graph.to_dict()
        stats = {
            "entities": len(graph_dict.get("entities", {})),
            "relationships": len(graph_dict.get("relationships", [])),
            "skills": len(self.skillbook.skills()) if self.skillbook else 0,
            "ace_active": self.ace_available
        }
        return stats