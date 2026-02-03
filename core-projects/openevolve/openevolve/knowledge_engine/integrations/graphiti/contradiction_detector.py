"""
Graphiti Contradiction Detector for OpenEvolve Knowledge Engine

This module provides contradiction detection capabilities for the Graphiti temporal knowledge graph system.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .graphiti_temporal_bridge import GraphitiTemporalBridge, KnowledgeArtifact


logger = logging.getLogger(__name__)


class GraphitiContradictionDetector:
    """
    Contradiction detector for Graphiti temporal knowledge system.
    
    Provides methods to detect contradictions in knowledge across time and contexts.
    """
    
    def __init__(self, bridge: GraphitiTemporalBridge):
        """
        Initialize the contradiction detector.
        
        Args:
            bridge: GraphitiTemporalBridge instance to use for detection
        """
        self.bridge = bridge
        logger.info({
            "msg": "GraphitiContradictionDetector initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def detect_contradictions(
        self,
        entity_name: Optional[str] = None,
        time_range: Optional[tuple] = None,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the knowledge graph.
        
        Args:
            entity_name: Specific entity to check for contradictions
            time_range: Time range tuple (start, end) to limit search
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of detected contradictions
        """
        correlation_id = correlation_id or f"contra_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting contradiction detection",
            "entity": entity_name,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        contradictions = []
        
        try:
            # If entity name is provided, focus on that entity
            if entity_name:
                # Get timeline for the specific entity
                if time_range:
                    start_dt, end_dt = time_range
                    timeline = await self.bridge.get_entity_timeline(
                        entity_name=entity_name,
                        start_time=start_dt,
                        end_time=end_dt,
                        correlation_id=correlation_id
                    )
                else:
                    # Get recent timeline (last 30 days as default)
                    from datetime import timedelta
                    end_time = datetime.now(timezone.utc)
                    start_time_range = end_time - timedelta(days=30)
                    
                    timeline = await self.bridge.get_entity_timeline(
                        entity_name=entity_name,
                        start_time=start_time_range,
                        end_time=end_time,
                        correlation_id=correlation_id
                    )
                
                # Look for contradictions in the timeline
                contradictions.extend(self._analyze_timeline_for_contradictions(timeline, entity_name))
            else:
                # Check for contradictions across all entities
                # This would be a more extensive check
                contradictions.extend(await self._detect_global_contradictions(correlation_id))
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Contradiction detection completed",
                "correlation_id": correlation_id,
                "contradictions_found": len(contradictions),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return contradictions
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Contradiction detection failed",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []
    
    def _analyze_timeline_for_contraditions(self, timeline: List[Dict[str, Any]], entity_name: str) -> List[Dict[str, Any]]:
        """
        Analyze an entity timeline for contradictions.
        
        Args:
            timeline: Timeline of events for an entity
            entity_name: Name of the entity
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Sort timeline by time
        sorted_timeline = sorted(timeline, key=lambda x: x.get('timestamp', datetime.min))
        
        # Look for contradictory statements in adjacent or overlapping time periods
        for i in range(len(sorted_timeline) - 1):
            current = sorted_timeline[i]
            next_item = sorted_timeline[i + 1]
            
            # Check if there are potentially contradictory facts
            current_fact = current.get('fact', '')
            next_fact = next_item.get('fact', '')
            
            # Simple heuristic: if facts are significantly different but close in time,
            # they might be contradictory
            if self._facts_might_be_contradictory(current_fact, next_fact):
                contradiction = {
                    "entity": entity_name,
                    "type": "temporal_change",
                    "description": f"Potentially contradictory facts: '{current_fact}' vs '{next_fact}'",
                    "timestamp1": current.get('timestamp'),
                    "timestamp2": next_item.get('timestamp'),
                    "severity": "medium",
                    "confidence": 0.7
                }
                contradictions.append(contradiction)
        
        return contradictions
    
    async def _detect_global_contradictions(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Detect contradictions across the entire knowledge graph.
        
        Args:
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        try:
            # Get all entities
            entities = await self.bridge.client.get_entity_list()
            
            # For each entity, check for contradictions
            for entity in entities[:10]:  # Limit to first 10 for performance
                entity_contradictions = await self.detect_contradictions(
                    entity_name=entity.name,
                    correlation_id=correlation_id
                )
                contradictions.extend(entity_contradictions)
            
            return contradictions
        except Exception as e:
            logger.error({
                "msg": "Global contradiction detection failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []
    
    def _facts_might_be_contradictory(self, fact1: str, fact2: str) -> bool:
        """
        Heuristic to determine if two facts might be contradictory.
        
        Args:
            fact1: First fact
            fact2: Second fact
            
        Returns:
            True if facts might be contradictory
        """
        # This is a simplified heuristic - in a real implementation,
        # this would use more sophisticated NLP techniques
        if not fact1 or not fact2:
            return False
        
        # Convert to lowercase for comparison
        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()
        
        # Look for opposite keywords
        opposite_pairs = [
            ("true", "false"),
            ("yes", "no"),
            ("present", "absent"),
            ("included", "excluded"),
            ("active", "inactive"),
            ("positive", "negative")
        ]
        
        for pos, neg in opposite_pairs:
            if (pos in fact1_lower and neg in fact2_lower) or (neg in fact1_lower and pos in fact2_lower):
                return True
        
        # Additional heuristics could be added here
        return False
    
    async def detect_entity_contradictions(
        self,
        entity_name: str,
        reference_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions specifically related to an entity.
        
        Args:
            entity_name: Name of entity to check
            reference_time: Reference time for temporal context
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of entity-specific contradictions
        """
        correlation_id = correlation_id or f"entity_contra_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Detecting entity-specific contradictions",
            "entity": entity_name,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Get all facts related to this entity
            query = f"Find all facts about {entity_name}"
            artifacts = await self.bridge.query_at_point_in_time(
                query=query,
                timestamp=reference_time or datetime.now(timezone.utc),
                max_results=50,  # Get more results for contradiction analysis
                correlation_id=correlation_id
            )
            
            # Analyze artifacts for contradictions
            contradictions = self._analyze_artifacts_for_contradictions(artifacts, entity_name)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Entity contradiction detection completed",
                "entity": entity_name,
                "contradictions_found": len(contradictions),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return contradictions
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Entity contradiction detection failed",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []
    
    def _analyze_artifacts_for_contradictions(self, artifacts: List[KnowledgeArtifact], entity_name: str) -> List[Dict[str, Any]]:
        """
        Analyze a list of knowledge artifacts for contradictions.
        
        Args:
            artifacts: List of KnowledgeArtifacts to analyze
            entity_name: Name of the entity being analyzed
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Group artifacts by their content themes
        content_groups = {}
        for artifact in artifacts:
            # Simplified grouping by first few words
            key = ' '.join(artifact.content.lower().split()[:5])
            if key not in content_groups:
                content_groups[key] = []
            content_groups[key].append(artifact)
        
        # Look for groups with conflicting information
        for group_key, group_artifacts in content_groups.items():
            if len(group_artifacts) > 1:
                # Multiple artifacts with similar content - check for contradictions
                for i in range(len(group_artifacts)):
                    for j in range(i + 1, len(group_artifacts)):
                        art1 = group_artifacts[i]
                        art2 = group_artifacts[j]
                        
                        # Check if they have conflicting temporal validity
                        if self._artifacts_conflict_temporally(art1, art2):
                            contradiction = {
                                "entity": entity_name,
                                "type": "temporal_conflict",
                                "description": f"Conflicting temporal validity: '{art1.content[:50]}...' vs '{art2.content[:50]}...'",
                                "artifact1_id": art1.id,
                                "artifact2_id": art2.id,
                                "timestamp1": art1.valid_at.isoformat(),
                                "timestamp2": art2.valid_at.isoformat(),
                                "severity": "high",
                                "confidence": 0.9
                            }
                            contradictions.append(contradiction)
        
        return contradictions
    
    def _artifacts_conflict_temporally(self, art1: KnowledgeArtifact, art2: KnowledgeArtifact) -> bool:
        """
        Check if two artifacts conflict temporally.
        
        Args:
            art1: First artifact
            art2: Second artifact
            
        Returns:
            True if artifacts conflict temporally
        """
        # Check if artifacts have overlapping validity periods but conflicting content
        # This is a simplified check - real implementation would be more nuanced
        if art1.valid_at <= art2.valid_at:
            # art1 is earlier, check if art2 invalidates art1
            if art1.invalid_at and art1.invalid_at > art2.valid_at:
                # There's overlap in validity periods
                return self._contents_conflict(art1.content, art2.content)
        else:
            # art2 is earlier, check if art1 invalidates art2
            if art2.invalid_at and art2.invalid_at > art1.valid_at:
                # There's overlap in validity periods
                return self._contents_conflict(art1.content, art2.content)
        
        return False
    
    def _contents_conflict(self, content1: str, content2: str) -> bool:
        """
        Check if two content strings conflict with each other.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            True if contents conflict
        """
        # Simplified conflict detection
        # In a real implementation, this would use more sophisticated NLP
        return self._facts_might_be_contradictory(content1, content2)