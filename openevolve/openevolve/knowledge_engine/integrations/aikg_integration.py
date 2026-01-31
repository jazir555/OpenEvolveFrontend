"""
AI-Knowledge-Graph Integration for OpenEvolve Knowledge Engine

This module provides integration with AI-driven knowledge graph processing,
including entity standardization, relationship inference, and visualization.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid
import json
import os
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class AIKGResult:
    """Result of AIKG processing."""
    success: bool
    original_triple_count: int
    inferred_triple_count: int
    standardized_entity_count: int
    visualization_path: Optional[str] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StandardizationResult:
    """Result of entity standardization."""
    canonical_entities: List[str]
    variant_mappings: Dict[str, List[str]]
    processing_time_ms: float = 0.0


@dataclass
class InferenceResult:
    """Result of relationship inference."""
    original_triples: List[Tuple[str, str, str]]
    inferred_triples: List[Tuple[str, str, str]]
    processing_time_ms: float = 0.0


@dataclass
class VisualizationResult:
    """Result of knowledge graph visualization."""
    output_path: str
    community_count: int
    node_count: int
    edge_count: int
    processing_time_ms: float = 0.0


class AIKGIntegration:
    """
    AI-Knowledge-Graph integration for OpenEvolve.
    
    Provides methods for:
    - Entity standardization
    - Relationship inference
    - Knowledge graph visualization
    - Export capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AIKG integration.
        
        Args:
            config: Configuration for AIKG components
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.standardizer = EntityStandardizer(
            config=self.config.get('standardization', {})
        )
        self.inference = RelationshipInferenceEngine(
            config=self.config.get('inference', {})
        )
        self.visualizer = KnowledgeGraphVisualizer(
            config=self.config.get('visualization', {})
        )
        
        logger.info({
            "msg": "AIKGIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'standardization': {
                'enabled': True,
                'use_llm_for_entities': False,
                'similarity_threshold': 0.9
            },
            'inference': {
                'enabled': True,
                'apply_transitive': True,
                'use_llm_for_inference': False
            },
            'visualization': {
                'enabled': True,
                'output_dir': 'data/visualizations',
                'default_layout': 'force_directed',
                'include_communities': True
            },
            'llm_client': None
        }
    
    async def process_knowledge_graph(
        self,
        text: str,
        enable_standardization: bool = True,
        enable_inference: bool = True,
        generate_visualization: bool = True,
        output_path: Optional[str] = None
    ) -> AIKGResult:
        """
        Process text with complete AIKG pipeline.
        
        Args:
            text: Input text to process
            enable_standardization: Whether to apply entity standardization
            enable_inference: Whether to apply relationship inference
            generate_visualization: Whether to generate D3.js visualization
            output_path: Optional path for visualization output file
            
        Returns:
            AIKGResult with complete processing results
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting AIKG processing pipeline",
            "text_length": len(text),
            "enable_standardization": enable_standardization,
            "enable_inference": enable_inference,
            "generate_visualization": generate_visualization,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Extract basic triples from text (simplified approach)
            # In a real implementation, this would use proper NLP/ML models
            basic_triples = self._extract_basic_triples(text)
            
            original_triple_count = len(basic_triples)
            
            # Apply standardization if enabled
            standardized_triples = basic_triples
            standardized_entity_count = len(set([t[0] for t in basic_triples] + [t[2] for t in basic_triples]))
            
            if enable_standardization:
                standardization_result = await self.standardizer.standardize_entities_from_triples(basic_triples)
                standardized_triples = standardization_result['standardized_triples']
                standardized_entity_count = len(standardization_result['canonical_entities'])
            
            # Apply inference if enabled
            all_triples = standardized_triples
            inferred_triple_count = 0
            
            if enable_inference:
                inference_result = await self.inference.infer_relationships(standardized_triples)
                all_triples = standardized_triples + inference_result['inferred_triples']
                inferred_triple_count = len(inference_result['inferred_triples'])
            
            # Generate visualization if requested
            visualization_path = None
            if generate_visualization:
                vis_result = await self.visualizer.visualize_graph(
                    triples=all_triples,
                    output_path=output_path
                )
                visualization_path = vis_result.output_path
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = AIKGResult(
                success=True,
                original_triple_count=original_triple_count,
                inferred_triple_count=inferred_triple_count,
                standardized_entity_count=standardized_entity_count,
                visualization_path=visualization_path,
                processing_time_ms=processing_time_ms,
                metadata={
                    "text_length": len(text),
                    "enable_standardization": enable_standardization,
                    "enable_inference": enable_inference,
                    "generate_visualization": generate_visualization
                }
            )
            
            logger.info({
                "msg": "AIKG processing pipeline completed",
                "original_triples": original_triple_count,
                "inferred_triples": inferred_triple_count,
                "standardized_entities": standardized_entity_count,
                "visualization_path": visualization_path,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "AIKG processing pipeline failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return AIKGResult(
                success=False,
                original_triple_count=0,
                inferred_triple_count=0,
                standardized_entity_count=0,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _extract_basic_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract basic triples from text (simplified approach).
        
        In a real implementation, this would use proper NLP/ML models.
        """
        import re
        
        # Simple pattern matching for (subject, predicate, object) triples
        # This is a very simplified approach
        triples = []
        
        # Look for patterns like "Subject verb object" in sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Simple extraction patterns
            # Pattern 1: "Subject is/are a/an Adjective Noun"
            pattern1 = r'([A-Z][a-zA-Z\s]+?)\s+(?:is|are)\s+(?:a|an)?\s*([A-Z][a-zA-Z\s]*?)\s+([A-Z][a-zA-Z\s]*)'
            matches1 = re.findall(pattern1, sentence)
            for match in matches1:
                subject, adjective, obj = match
                subject, adjective, obj = subject.strip(), adjective.strip(), obj.strip()
                if subject and obj:
                    triples.append((subject, f"is_{adjective}".lower(), obj))
            
            # Pattern 2: "Subject verb object" (very simple)
            words = sentence.split()
            for i in range(len(words) - 2):
                if words[i].istitle() and words[i+2].istitle():  # Potential subject and object
                    if words[i+1].lower() in ['is', 'are', 'was', 'were', 'works', 'located', 'founded']:
                        triples.append((words[i], words[i+1].lower(), words[i+2]))
        
        return triples


class EntityStandardizer:
    """
    Entity standardizer for AIKG.
    
    Provides methods for standardizing entity names and creating canonical forms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity standardizer.
        
        Args:
            config: Configuration for standardization
        """
        self.config = config or {
            'similarity_threshold': 0.9,
            'use_llm_for_entities': False
        }
        
        # Maintain variant mappings
        self.variant_mappings: Dict[str, List[str]] = {}
        
        logger.info({
            "msg": "EntityStandardizer initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def standardize_entities_from_triples(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Standardize entities from a list of triples.
        
        Args:
            triples: List of (subject, predicate, object) triples
            
        Returns:
            Dictionary with standardized triples and mappings
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting entity standardization from triples",
            "triple_count": len(triples),
            "timestamp": start_time.isoformat()
        })
        
        # Extract all entities from triples
        entities = set()
        for subj, _, obj in triples:
            entities.add(subj)
            entities.add(obj)
        
        # Standardize entities
        standardized_entities = await self._standardize_entity_list(list(entities))
        
        # Create mapping from original to standardized
        entity_mapping = {}
        for orig, std in standardized_entities.items():
            entity_mapping[orig] = std
        
        # Create standardized triples
        standardized_triples = []
        for subj, pred, obj in triples:
            new_subj = entity_mapping.get(subj, subj)
            new_obj = entity_mapping.get(obj, obj)
            standardized_triples.append((new_subj, pred, new_obj))
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        result = {
            'standardized_triples': standardized_triples,
            'canonical_entities': list(set([t[0] for t in standardized_triples] + [t[2] for t in standardized_triples])),
            'variant_mappings': self.variant_mappings,
            'processing_time_ms': processing_time_ms
        }
        
        logger.info({
            "msg": "Entity standardization completed",
            "original_entity_count": len(entities),
            "canonical_entity_count": len(result['canonical_entities']),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
    
    async def _standardize_entity_list(self, entities: List[str]) -> Dict[str, str]:
        """
        Standardize a list of entities.
        
        Args:
            entities: List of entity names to standardize
            
        Returns:
            Dictionary mapping original entities to standardized forms
        """
        # Group similar entities together
        entity_groups = {}
        
        for entity in entities:
            # Find if this entity is similar to any existing canonical
            found_match = False
            for canonical, variants in entity_groups.items():
                if self._entities_are_similar(entity, canonical):
                    variants.append(entity)
                    found_match = True
                    break
            
            if not found_match:
                # Create new group with this entity as canonical
                entity_groups[entity] = []
        
        # Create mapping from variants to canonicals
        mapping = {}
        for canonical, variants in entity_groups.items():
            mapping[canonical] = canonical
            for variant in variants:
                mapping[variant] = canonical
                # Update variant mappings
                if canonical not in self.variant_mappings:
                    self.variant_mappings[canonical] = []
                if variant not in self.variant_mappings[canonical]:
                    self.variant_mappings[canonical].append(variant)
        
        return mapping
    
    def _entities_are_similar(self, entity1: str, entity2: str) -> bool:
        """
        Check if two entities are similar enough to be standardized together.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            True if entities are similar
        """
        # Normalize entities
        e1_norm = entity1.lower().strip()
        e2_norm = entity2.lower().strip()
        
        # Direct match
        if e1_norm == e2_norm:
            return True
        
        # One contains the other with high overlap
        if e1_norm in e2_norm or e2_norm in e1_norm:
            shorter, longer = (e1_norm, e2_norm) if len(e1_norm) < len(e2_norm) else (e2_norm, e1_norm)
            if len(shorter) / len(longer) > self.config.get('similarity_threshold', 0.9):
                return True
        
        # Check for common variations (abbreviations, etc.)
        variations_map = {
            'inc.': 'incorporated',
            'inc': 'incorporated',
            'corp.': 'corporation',
            'corp': 'corporation',
            'ltd.': 'limited',
            'ltd': 'limited',
            'co.': 'company',
            'co': 'company',
        }
        
        # Normalize by expanding common abbreviations
        norm1, norm2 = e1_norm, e2_norm
        for abbr, full in variations_map.items():
            norm1 = norm1.replace(abbr, full)
            norm2 = norm2.replace(abbr, full)
        
        if norm1 == norm2:
            return True
        
        return False


class RelationshipInferenceEngine:
    """
    Relationship inference engine for AIKG.
    
    Provides methods for inferring new relationships from existing knowledge.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relationship inference engine.
        
        Args:
            config: Configuration for inference
        """
        self.config = config or {
            'apply_transitive': True,
            'use_llm_for_inference': False
        }
        
        logger.info({
            "msg": "RelationshipInferenceEngine initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def infer_relationships(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> InferenceResult:
        """
        Infer new relationships from existing triples.
        
        Args:
            triples: List of (subject, predicate, object) triples
            
        Returns:
            InferenceResult with original and inferred triples
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting relationship inference",
            "triple_count": len(triples),
            "timestamp": start_time.isoformat()
        })
        
        inferred_triples = []
        
        if self.config.get('apply_transitive', True):
            # Apply transitive inference: if A->B and B->C then possibly A->C
            inferred_triples.extend(self._apply_transitive_inference(triples))
        
        # Add other inference rules here as needed
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        result = InferenceResult(
            original_triples=triples,
            inferred_triples=inferred_triples,
            processing_time_ms=processing_time_ms
        )
        
        logger.info({
            "msg": "Relationship inference completed",
            "original_triples": len(triples),
            "inferred_triples": len(inferred_triples),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
    
    def _apply_transitive_inference(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Apply transitive inference to triples.
        
        Args:
            triples: List of (subject, predicate, object) triples
            
        Returns:
            List of inferred triples
        """
        inferred = []
        
        # Create a mapping of subject -> (predicate, object)
        subj_to_pred_obj = {}
        for subj, pred, obj in triples:
            if subj not in subj_to_pred_obj:
                subj_to_pred_obj[subj] = []
            subj_to_pred_obj[subj].append((pred, obj))
        
        # Look for transitive patterns: A->B and B->C => A->C
        for subj1, pred_obj_list1 in subj_to_pred_obj.items():
            for pred1, obj1 in pred_obj_list1:
                # Check if obj1 is a subject in other triples
                if obj1 in subj_to_pred_obj:
                    for pred2, obj2 in subj_to_pred_obj[obj1]:
                        # Infer: subj1 -(pred1+pred2)-> obj2
                        # For simplicity, we'll use a generic "connected_to" predicate
                        inferred.append((subj1, "connected_to_via_transitivity", obj2))
        
        return inferred


class KnowledgeGraphVisualizer:
    """
    Knowledge graph visualizer for AIKG.
    
    Provides methods for generating D3.js visualizations of knowledge graphs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge graph visualizer.
        
        Args:
            config: Configuration for visualization
        """
        self.config = config or {
            'output_dir': 'data/visualizations',
            'default_layout': 'force_directed',
            'include_communities': True
        }
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.config.get('output_dir', 'data/visualizations'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info({
            "msg": "KnowledgeGraphVisualizer initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def visualize_graph(
        self,
        triples: List[Tuple[str, str, str]],
        output_path: Optional[str] = None
    ) -> VisualizationResult:
        """
        Generate D3.js visualization of knowledge graph.
        
        Args:
            triples: List of (subject, predicate, object) triples
            output_path: Optional output path for visualization
            
        Returns:
            VisualizationResult with output path and statistics
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge graph visualization",
            "triple_count": len(triples),
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create visualization data structure
            nodes_set = set()
            links = []
            
            for subj, pred, obj in triples:
                nodes_set.add(subj)
                nodes_set.add(obj)
                links.append({
                    "source": subj,
                    "target": obj,
                    "relationship": pred
                })
            
            nodes = []
            for i, node in enumerate(nodes_set):
                nodes.append({
                    "id": node,
                    "label": node,
                    "group": hash(node) % 10,  # Simple grouping
                    "degree": sum(1 for link in links if link["source"] == node or link["target"] == node)
                })
            
            # Generate output path if not provided
            if not output_path:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                output_path = str(Path(self.config.get('output_dir', 'data/visualizations')) / f"kg_viz_{timestamp}.json")
            
            # Create visualization data
            viz_data = {
                "nodes": nodes,
                "links": links,
                "meta": {
                    "node_count": len(nodes),
                    "edge_count": len(links),
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Write visualization data to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = VisualizationResult(
                output_path=output_path,
                community_count=0,  # Placeholder - would implement community detection in real version
                node_count=len(nodes),
                edge_count=len(links),
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Knowledge graph visualization completed",
                "output_path": output_path,
                "node_count": result.node_count,
                "edge_count": result.edge_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge graph visualization failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise
    
    async def export_graph_data(
        self,
        triples: List[Tuple[str, str, str]],
        format: str = "json"
    ) -> str:
        """
        Export knowledge graph data for external tools.
        
        Args:
            triples: List of (subject, predicate, object) triples
            format: Export format ('json', 'gexf', 'graphml', 'csv')
            
        Returns:
            Exported graph data as string
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting graph data export",
            "triple_count": len(triples),
            "format": format,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if format.lower() == 'json':
                # Create nodes and links
                nodes_set = set()
                links = []
                
                for subj, pred, obj in triples:
                    nodes_set.add(subj)
                    nodes_set.add(obj)
                    links.append({
                        "source": subj,
                        "target": obj,
                        "relationship": pred
                    })
                
                nodes = [{"id": node, "label": node} for node in nodes_set]
                
                export_data = {
                    "nodes": nodes,
                    "links": links,
                    "meta": {
                        "node_count": len(nodes),
                        "edge_count": len(links),
                        "export_format": "json",
                        "exported_at": datetime.now(timezone.utc).isoformat()
                    }
                }
                
                result = json.dumps(export_data, indent=2)
                
            elif format.lower() == 'csv':
                # Export as CSV with columns: subject, predicate, object
                csv_lines = ["subject,predicate,object"]
                for subj, pred, obj in triples:
                    # Escape commas and quotes in data
                    subj_csv = subj.replace('"', '""')
                    pred_csv = pred.replace('"', '""')
                    obj_csv = obj.replace('"', '""')
                    csv_lines.append(f'"{subj_csv}","{pred_csv}","{obj_csv}"')
                
                result = "\n".join(csv_lines)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Graph data export completed",
                "format": format,
                "data_size": len(result),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Graph data export failed",
                "format": format,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise