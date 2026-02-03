"""
DeepKE Pipeline for Document Processing

End-to-end pipeline for extracting knowledge from documents and
integrating into the knowledge graph.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .extractor import DeepKEExtractor, ExtractionResult
from .linking import EntityLinker, EntityDisambiguator, LinkingResult
try:
    from ..graph.models import KnowledgeNode, KnowledgeEdge, KnowledgeGraph
    from ..graph.schema import NodeType, EdgeType
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from graph.models import KnowledgeNode, KnowledgeEdge, KnowledgeGraph
    from graph.schema import NodeType, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the DeepKE pipeline"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_confidence: float = 0.5
    link_entities: bool = True
    disambiguate: bool = True
    resolve_coreference: bool = True
    include_metadata: bool = True
    
    # Extraction options
    extract_entities: bool = True
    extract_relations: bool = True
    
    # Output options
    create_knowledge_graph: bool = True
    save_intermediate: bool = False
    output_dir: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of the DeepKE pipeline"""
    document_id: str
    extraction: ExtractionResult
    linking: List[LinkingResult] = field(default_factory=list)
    knowledge_graph: Optional[KnowledgeGraph] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "extraction": self.extraction.to_dict(),
            "linking_count": len(self.linking),
            "linked_entities": sum(1 for l in self.linking if l.is_linked()),
            "knowledge_graph": {
                "node_count": len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
                "edge_count": len(self.knowledge_graph.edges) if self.knowledge_graph else 0
            } if self.knowledge_graph else None,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


class DeepKEPipeline:
    """End-to-end DeepKE processing pipeline"""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        knowledge_base=None
    ):
        self.config = config or PipelineConfig()
        self.extractor = DeepKEExtractor()
        self.linker = EntityLinker(knowledge_base)
        self.disambiguator = EntityDisambiguator()
        self.kb = knowledge_base
    
    async def process(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Process a document through the pipeline"""
        import time
        start_time = time.time()
        
        if not document_id:
            document_id = f"doc_{datetime.utcnow().timestamp()}"
        
        logger.info(f"Processing document {document_id}")
        
        # Step 1: Extract entities and relations
        extraction = self._extract(text)
        
        # Step 2: Link entities to knowledge base
        linking = []
        if self.config.link_entities:
            linking = self._link(extraction)
        
        # Step 3: Disambiguate entities
        if self.config.disambiguate and linking:
            linking = self._disambiguate(extraction, linking)
        
        # Step 4: Build knowledge graph
        kg = None
        if self.config.create_knowledge_graph:
            kg = self._build_knowledge_graph(extraction, linking, document_id)
        
        # Step 5: Save to knowledge base
        if self.kb and kg:
            await self._save_to_kb(kg)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Processed document {document_id} in {processing_time:.2f}s")
        
        return PipelineResult(
            document_id=document_id,
            extraction=extraction,
            linking=linking,
            knowledge_graph=kg,
            processing_time=processing_time,
            metadata=metadata or {}
        )
    
    async def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Process a file through the pipeline"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file
        text = path.read_text(encoding='utf-8')
        
        # Process
        doc_id = metadata.get('id', path.stem) if metadata else path.stem
        return await self.process(text, doc_id, metadata)
    
    async def process_batch(
        self,
        texts: List[str],
        document_ids: Optional[List[str]] = None
    ) -> List[PipelineResult]:
        """Process multiple documents"""
        results = []
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else f"doc_{i}"
            result = await self.process(text, doc_id)
            results.append(result)
        return results
    
    def _extract(self, text: str) -> ExtractionResult:
        """Step 1: Extract entities and relations"""
        logger.debug("Step 1: Extracting entities and relations")
        
        return self.extractor.extract_from_document(
            text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
    
    def _link(self, extraction: ExtractionResult) -> List[LinkingResult]:
        """Step 2: Link entities to knowledge base"""
        logger.debug("Step 2: Linking entities")
        
        return self.linker.link_batch(
            extraction.entities,
            context=extraction.text
        )
    
    def _disambiguate(
        self,
        extraction: ExtractionResult,
        linking: List[LinkingResult]
    ) -> List[LinkingResult]:
        """Step 3: Disambiguate entities"""
        logger.debug("Step 3: Disambiguating entities")
        
        # Build entity graph for coherence scoring
        self.disambiguator.build_entity_graph(extraction.relations)
        
        # Find ambiguous entities
        ambiguous = [l for l in linking if len(l.candidates) > 1]
        
        if ambiguous:
            resolved = self.disambiguator.disambiguate(
                ambiguous,
                extraction.entities
            )
            
            # Update linking results
            resolved_dict = {r.entity.text: r for r in resolved}
            for i, link in enumerate(linking):
                if link.entity.text in resolved_dict:
                    linking[i] = resolved_dict[link.entity.text]
        
        return linking
    
    def _build_knowledge_graph(
        self,
        extraction: ExtractionResult,
        linking: List[LinkingResult],
        document_id: str
    ) -> KnowledgeGraph:
        """Step 4: Build knowledge graph from extraction"""
        logger.debug("Step 4: Building knowledge graph")
        
        kg = KnowledgeGraph(name=f"kg_{document_id}")
        
        # Create mapping from entity text to node ID
        entity_to_node = {}
        
        # Add entities as nodes
        for entity in extraction.entities:
            # Skip low confidence entities
            if entity.confidence < self.config.min_confidence:
                continue
            
            # Check if linked to existing entity
            link_result = next(
                (l for l in linking if l.entity.text == entity.text),
                None
            )
            
            if link_result and link_result.is_linked():
                # Use existing entity ID
                entity_to_node[entity.text] = link_result.selected_candidate.entity_id
            else:
                # Create new node
                from ..graph.models import NodeProperties
                
                node_type = self._map_entity_type(entity.entity_type)
                node = KnowledgeNode(
                    node_type=node_type,
                    properties=NodeProperties(
                        name=entity.normalized_text,
                        source=document_id,
                        confidence=entity.confidence,
                        metadata={
                            "original_text": entity.text,
                            "extraction_method": "deepke",
                            **entity.metadata
                        }
                    )
                )
                
                kg.add_node(node)
                entity_to_node[entity.text] = node.id
        
        # Add relations as edges
        for relation in extraction.relations:
            subj_id = entity_to_node.get(relation.subject.text)
            obj_id = entity_to_node.get(relation.object.text)
            
            if subj_id and obj_id:
                from ..graph.models import EdgeProperties
                
                edge_type = self._map_relation_type(relation.predicate)
                edge = KnowledgeEdge(
                    edge_type=edge_type,
                    source_id=subj_id,
                    target_id=obj_id,
                    properties=EdgeProperties(
                        weight=relation.confidence,
                        confidence=relation.confidence,
                        source=document_id,
                        metadata={
                            "context": relation.sentence_context,
                            **relation.metadata
                        }
                    )
                )
                
                try:
                    kg.add_edge(edge)
                except ValueError:
                    # Nodes might not exist
                    pass
        
        return kg
    
    async def _save_to_kb(self, kg: KnowledgeGraph):
        """Step 5: Save knowledge graph to knowledge base"""
        logger.debug("Step 5: Saving to knowledge base")
        
        # Would save to actual knowledge base here
        # For now, just log
        logger.info(f"Would save {len(kg.nodes)} nodes and {len(kg.edges)} edges to KB")
    
    def _map_entity_type(self, entity_type) -> NodeType:
        """Map DeepKE entity type to Knowledge Graph node type"""
        mapping = {
            # DeepKE types
            'PERSON': NodeType.ENTITY,
            'ORG': NodeType.ENTITY,
            'LOC': NodeType.ENTITY,
            'TECH': NodeType.CONCEPT,
            'CONCEPT': NodeType.CONCEPT,
            'CODE': NodeType.CODE,
            'ALGO': NodeType.CONCEPT,
            'API': NodeType.CODE,
            'DATA_STRUCT': NodeType.CONCEPT,
            'PRODUCT': NodeType.ENTITY,
            'EVENT': NodeType.EVENT,
        }
        
        type_str = entity_type.value if hasattr(entity_type, 'value') else str(entity_type)
        mapped = mapping.get(type_str, NodeType.CONCEPT)
        return mapped
    
    def _map_relation_type(self, relation_type) -> EdgeType:
        """Map DeepKE relation type to Knowledge Graph edge type"""
        from .extractor import RelationType
        
        mapping = {
            RelationType.WORKS_FOR: EdgeType.PERFORMED_BY,
            RelationType.LOCATED_IN: EdgeType.PART_OF,
            RelationType.FOUNDED_BY: EdgeType.PERFORMED_BY,
            RelationType.PART_OF: EdgeType.PART_OF,
            RelationType.USES: EdgeType.USES,
            RelationType.IMPLEMENTS: EdgeType.IMPLEMENTS,
            RelationType.DEPENDS_ON: EdgeType.DEPENDS_ON,
            RelationType.EXTENDS: EdgeType.RELATED_TO,
            RelationType.RELATED_TO: EdgeType.RELATED_TO,
            RelationType.CALLS: EdgeType.RELATED_TO,
        }
        
        return mapping.get(relation_type, EdgeType.RELATED_TO)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "config": {
                "chunk_size": self.config.chunk_size,
                "link_entities": self.config.link_entities,
                "disambiguate": self.config.disambiguate,
            },
            "extractor_available": self.extractor.is_available(),
        }
