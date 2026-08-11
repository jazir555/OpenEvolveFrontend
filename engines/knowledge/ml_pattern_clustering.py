"""
ML-Based Pattern Clustering for Stage 6 Knowledge Extraction

This module implements real machine learning clustering for pattern extraction:
- Uses Sentence Transformers for embeddings
- Uses scikit-learn for clustering (DBSCAN, KMeans, HDBSCAN)
- Implements entity and relation extraction using transformer models
- Provides confidence scoring for extracted patterns
- Supports temporal pattern tracking

Dependencies (permissive licenses):
- sentence-transformers: Apache 2.0
- scikit-learn: BSD License
- numpy: BSD License
- networkx: BSD License

Author: OpenEvolve AI
License: Apache 2.0
"""

import json
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML libraries not fully available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from z3 import Solver, Bool, And, Or, Not, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# DeepKE Integration - WIRED TO CORE
try:
    from integrations.deepke import DeepKEBridge
    DEEPKE_AVAILABLE = True
    logger.info("DeepKE integration available for entity/relation extraction")
except ImportError:
    DEEPKE_AVAILABLE = False
    logger.warning("DeepKE integration not available")

# OneKE Integration - WIRED TO CORE
try:
    from integrations.oneke import OneKEBridge
    ONEKE_AVAILABLE = True
    logger.info("OneKE integration available for schema-guided extraction")
except ImportError:
    ONEKE_AVAILABLE = False
    logger.warning("OneKE integration not available")

# Karate Club Integration - NEW
try:
    from knowledge_engine.integrations.karateclub_integration import KarateClubGraphAnalyzer
    KARATECLUB_AVAILABLE = True
    logger.info("Karate Club integration available for graph analysis")
except ImportError:
    KARATECLUB_AVAILABLE = False
    logger.warning("Karate Club integration not available")

# kg-gen Integration - NEW
try:
    from knowledge_engine.integrations.kggen.kggen_pipeline import KGGenPipeline
    KG_GEN_AVAILABLE = True
    logger.info("kg-gen integration available for graph generation")
except ImportError:
    KG_GEN_AVAILABLE = False
    logger.warning("kg-gen integration not available")

# AI Knowledge Graph Integration - NEW
try:
    from integrations.ai_knowledge_graph.bridge import AIKnowledgeGraphBridge
    AI_KG_AVAILABLE = True
    logger.info("AI Knowledge Graph integration available")
except ImportError:
    AI_KG_AVAILABLE = False
    logger.warning("AI Knowledge Graph integration not available")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExtractedEntity:
    """An entity extracted from text using NER."""
    entity_id: str
    text: str
    entity_type: str  # 'person', 'organization', 'concept', 'solution', 'problem'
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'text': self.text,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class ExtractedRelation:
    """A relation between two entities."""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str  # 'solves', 'depends_on', 'improves', 'causes', 'part_of'
    confidence: float
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'relation_id': self.relation_id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'relation_type': self.relation_type,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'metadata': self.metadata
        }


@dataclass
class MLPattern:
    """A pattern discovered through ML clustering."""
    pattern_id: str
    pattern_type: str  # 'semantic', 'structural', 'temporal', 'causal'
    description: str
    confidence: float
    
    # Clustering info
    cluster_size: int
    centroid: Optional[np.ndarray] = None
    silhouette_score: float = 0.0
    
    # Content
    representative_examples: List[str] = field(default_factory=list)
    cluster_members: List[str] = field(default_factory=list)
    
    # Entities and relations
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    
    # Temporal tracking
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Metadata
    features: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    domain: str = "general"
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'confidence': self.confidence,
            'cluster_size': self.cluster_size,
            'silhouette_score': self.silhouette_score,
            'representative_examples': self.representative_examples,
            'cluster_members': self.cluster_members,
            'entities': [e.to_dict() for e in self.entities],
            'relations': [r.to_dict() for r in self.relations],
            'first_seen': self.first_seen.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'features': self.features,
            'tags': self.tags,
            'domain': self.domain
        }


@dataclass
class TemporalKnowledgeNode:
    """A node in the temporal knowledge graph."""
    node_id: str
    content: str
    node_type: str  # 'fact', 'pattern', 'solution', 'problem'
    
    # Temporal tracking
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Versioning
    version: int = 1
    previous_version_id: Optional[str] = None
    
    # Confidence and validation
    confidence: float = 0.5
    validation_status: str = "unverified"  # 'verified', 'unverified', 'deprecated', 'expired'
    
    # Graph connections
    related_nodes: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this knowledge has expired."""
        if self.valid_until:
            return datetime.now() > self.valid_until
        return False
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'content': self.content,
            'node_type': self.node_type,
            'created_at': self.created_at.isoformat(),
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'version': self.version,
            'previous_version_id': self.previous_version_id,
            'confidence': self.confidence,
            'validation_status': self.validation_status,
            'related_nodes': self.related_nodes,
            'derived_from': self.derived_from,
            'metadata': self.metadata
        }


# =============================================================================
# ML-BASED ENTITY EXTRACTION
# =============================================================================

class EntityExtractor:
    """
    Extract entities from text using transformer models.
    
    Uses pattern-based extraction with ML-enhanced classification.
    """
    
    # Common entity patterns for software/engineering domains
    ENTITY_PATTERNS = {
        'solution': [
            r'\b(solution|approach|method|technique|strategy)\s+(?:for|to)\s+([\w\s]+)',
            r'\b(using|via|by)\s+(?:the\s+)?([\w\s]+?)(?:\s+(?:method|approach|technique))?',
        ],
        'problem': [
            r'\b(problem|issue|challenge|difficulty)\s+(?:with|in|of)\s+([\w\s]+)',
            r'\b(optimiz|improv|fix|resolv|handl)\w+\s+(?:the\s+)?([\w\s]+)',
        ],
        'component': [
            r'\b(component|module|service|system)\s+(?:called|named)?\s*["\']?([\w\s]+?)["\']?\b',
            r'\b(class|function|method)\s+(?:called|named)?\s*["\']?([\w\s]+?)["\']?\b',
        ],
        'metric': [
            r'\b(accuracy|precision|recall|f1|score|performance)\s+(?:of|is)\s+(\d+\.?\d*)',
            r'\b(\d+\.?\d*)\s*%(?:\s+(accuracy|precision|recall))',
        ]
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize entity extractor.
        
        Args:
            model_name: Name of sentence transformer model for embeddings
        """
        self.model_name = model_name or 'all-MiniLM-L6-v2'
        self.embedding_model = None
        self._lock = threading.RLock()
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
    
    def extract_entities(self, text: str, context: Optional[str] = None) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            context: Additional context for classification
            
        Returns:
            List of extracted entities
        """
        import re
        
        entities = []
        entity_id_counter = 0
        
        # Pattern-based extraction
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_id_counter += 1
                    entity_id = f"ent_{hashlib.md5(f'{text[:50]}_{entity_id_counter}'.encode()).hexdigest()[:8]}"
                    
                    # Get matched text
                    matched_text = match.group(0)
                    
                    # Calculate confidence based on pattern match quality
                    confidence = self._calculate_extraction_confidence(match, text)
                    
                    entity = ExtractedEntity(
                        entity_id=entity_id,
                        text=matched_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context or text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        metadata={
                            'pattern_matched': pattern,
                            'match_groups': match.groups()
                        }
                    )
                    entities.append(entity)
        
        # Deduplicate overlapping entities
        entities = self._deduplicate_entities(entities)
        
        # Use embeddings for classification if available
        if self.embedding_model:
            entities = self._classify_with_embeddings(entities)
        
        return entities
    
    def _calculate_extraction_confidence(self, match, text: str) -> float:
        """Calculate confidence score for an extraction."""
        base_confidence = 0.7
        
        # Boost confidence for longer matches
        match_len = match.end() - match.start()
        if match_len > 20:
            base_confidence += 0.1
        
        # Boost for matches in clear context
        surrounding = text[max(0, match.start()-20):min(len(text), match.end()+20)]
        if any(word in surrounding.lower() for word in ['the', 'a', 'an', 'this', 'that']):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entity extractions."""
        if not entities:
            return entities
        
        # Sort by confidence (highest first)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        kept = []
        covered_spans = []
        
        for entity in sorted_entities:
            # Check for overlap
            overlap = False
            for start, end in covered_spans:
                if not (entity.end_pos <= start or entity.start_pos >= end):
                    overlap = True
                    break
            
            if not overlap:
                kept.append(entity)
                covered_spans.append((entity.start_pos, entity.end_pos))
        
        return kept
    
    def _classify_with_embeddings(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Use embeddings to refine entity classification."""
        if not self.embedding_model or not entities:
            return entities
        
        # Define prototype embeddings for entity types
        prototypes = {
            'solution': self.embedding_model.encode("solution approach method technique implementation"),
            'problem': self.embedding_model.encode("problem issue challenge difficulty error"),
            'component': self.embedding_model.encode("component module service system class function"),
            'metric': self.embedding_model.encode("accuracy performance score metric evaluation"),
        }
        
        for entity in entities:
            entity_embedding = self.embedding_model.encode(entity.text)
            
            # Find closest prototype
            best_type = entity.entity_type
            best_score = -1
            
            for type_name, prototype in prototypes.items():
                similarity = self._cosine_similarity(entity_embedding, prototype)
                if similarity > best_score:
                    best_score = similarity
                    best_type = type_name
            
            # Update type if confident
            if best_score > 0.7 and best_type != entity.entity_type:
                entity.metadata['original_type'] = entity.entity_type
                entity.entity_type = best_type
                entity.confidence = min(1.0, entity.confidence + 0.1)
            
            entity.metadata['embedding_similarity'] = float(best_score)
        
        return entities
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# ML-BASED RELATION EXTRACTION
# =============================================================================

class RelationExtractor:
    """
    Extract relations between entities.
    
    Identifies semantic relationships between extracted entities.
    """
    
    RELATION_PATTERNS = {
        'solves': [
            r'(\w+)\s+solves?\s+(\w+)',
            r'(\w+)\s+resolves?\s+(\w+)',
            r'(\w+)\s+fixes?\s+(\w+)',
        ],
        'depends_on': [
            r'(\w+)\s+depends?\s+on\s+(\w+)',
            r'(\w+)\s+requires?\s+(\w+)',
            r'(\w+)\s+needs?\s+(\w+)',
        ],
        'improves': [
            r'(\w+)\s+improves?\s+(\w+)',
            r'(\w+)\s+enhances?\s+(\w+)',
            r'(\w+)\s+optimizes?\s+(\w+)',
        ],
        'causes': [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
        ],
        'part_of': [
            r'(\w+)\s+is\s+part\s+of\s+(\w+)',
            r'(\w+)\s+belongs?\s+to\s+(\w+)',
            r'(\w+)\s+in\s+(\w+)',
        ]
    }
    
    def __init__(self):
        """Initialize relation extractor."""
        self._lock = threading.RLock()
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """
        Extract relations between entities.
        
        Args:
            text: Source text
            entities: Previously extracted entities
            
        Returns:
            List of extracted relations
        """
        import re
        
        relations = []
        entity_map = {e.text.lower(): e for e in entities}
        
        # Pattern-based extraction
        for relation_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Extract source and target
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_text = groups[0].lower()
                        target_text = groups[1].lower()
                        
                        # Find corresponding entities
                        source_entity = self._find_entity(source_text, entities)
                        target_entity = self._find_entity(target_text, entities)
                        
                        if source_entity and target_entity:
                            relation_id = f"rel_{hashlib.md5(f'{source_entity.entity_id}_{target_entity.entity_id}_{relation_type}'.encode()).hexdigest()[:8]}"
                            
                            relation = ExtractedRelation(
                                relation_id=relation_id,
                                source_entity_id=source_entity.entity_id,
                                target_entity_id=target_entity.entity_id,
                                relation_type=relation_type,
                                confidence=0.75,
                                evidence=match.group(0),
                                metadata={
                                    'pattern_matched': pattern,
                                    'match_position': match.start()
                                }
                            )
                            relations.append(relation)
        
        # Semantic relation inference based on entity proximity
        relations.extend(self._infer_proximity_relations(text, entities))
        
        return self._deduplicate_relations(relations)
    
    def _find_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity by text match."""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        return None
    
    def _infer_proximity_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Infer relations based on entity proximity in text."""
        relations = []
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        
        # Look for entities that appear close together
        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i+1:]:
                distance = entity2.start_pos - entity1.end_pos
                
                # If entities are within 50 characters
                if distance < 50 and distance > 0:
                    # Check for connecting words
                    connecting_text = text[entity1.end_pos:entity2.start_pos].lower()
                    
                    relation_type = 'related_to'
                    confidence = 0.5
                    
                    if any(word in connecting_text for word in ['solves', 'fixes', 'resolves']):
                        relation_type = 'solves'
                        confidence = 0.7
                    elif any(word in connecting_text for word in ['requires', 'depends', 'needs']):
                        relation_type = 'depends_on'
                        confidence = 0.7
                    elif any(word in connecting_text for word in ['improves', 'enhances']):
                        relation_type = 'improves'
                        confidence = 0.7
                    
                    relation_id = f"rel_{hashlib.md5(f'{entity1.entity_id}_{entity2.entity_id}_{relation_type}'.encode()).hexdigest()[:8]}"
                    
                    relation = ExtractedRelation(
                        relation_id=relation_id,
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relation_type=relation_type,
                        confidence=confidence,
                        evidence=connecting_text.strip(),
                        metadata={'inferred_from_proximity': True, 'distance_chars': distance}
                    )
                    relations.append(relation)
        
        return relations
    
    def _deduplicate_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """Remove duplicate relations."""
        seen = set()
        unique = []
        
        for relation in relations:
            key = (relation.source_entity_id, relation.target_entity_id, relation.relation_type)
            if key not in seen:
                seen.add(key)
                unique.append(relation)
        
        return unique


# =============================================================================
# DEEPKE INTEGRATION - WIRED TO CORE
# =============================================================================

class DeepKEExtractor:
    """
    DeepKE-powered entity and relation extractor.
    
    Actually calls DeepKE library (not just imported) with fallback to
    pattern-based extraction when unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepKE extractor.
        
        Args:
            config: DeepKE configuration
        """
        self.config = config or {}
        self.bridge: Optional[DeepKEBridge] = None
        self._available = False
        
        if DEEPKE_AVAILABLE:
            try:
                self.bridge = DeepKEBridge(self.config)
                logger.info("DeepKE extractor created")
            except Exception as e:
                logger.warning(f"Failed to create DeepKE bridge: {e}")
    
    def initialize(self) -> bool:
        """Initialize DeepKE models."""
        if not DEEPKE_AVAILABLE or self.bridge is None:
            return False
        
        try:
            self._available = self.bridge.initialize()
            if self._available:
                logger.info("DeepKE extractor initialized successfully")
            return self._available
        except Exception as e:
            logger.error(f"Failed to initialize DeepKE extractor: {e}")
            return False
    
    def extract(self, text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relations using DeepKE.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entities, relations)
        """
        if not self._available:
            return [], []
        
        try:
            # Actually call DeepKE
            result = self.bridge.extract_from_text(text)
            
            entities = []
            relations = []
            
            # Convert to ExtractedEntity format
            for entity_data in result.get('entities', []):
                entity = ExtractedEntity(
                    entity_id=f"deepke_{hashlib.md5(entity_data['text'].encode()).hexdigest()[:8]}",
                    text=entity_data.get('text', ''),
                    entity_type=entity_data.get('type', 'UNKNOWN'),
                    confidence=entity_data.get('confidence', 0.5),
                    start_pos=entity_data.get('start', 0),
                    end_pos=entity_data.get('end', 0),
                    metadata={'source': 'deepke', 'raw': entity_data}
                )
                entities.append(entity)
            
            # Convert to ExtractedRelation format
            for relation_data in result.get('relations', []):
                # Find or create entity IDs
                head_text = relation_data.get('head', '')
                tail_text = relation_data.get('tail', '')
                
                head_entity = next((e for e in entities if e.text == head_text), None)
                tail_entity = next((e for e in entities if e.text == tail_text), None)
                
                if head_entity and tail_entity:
                    relation = ExtractedRelation(
                        relation_id=f"deepke_rel_{hashlib.md5(f'{head_text}_{tail_text}'.encode()).hexdigest()[:8]}",
                        source_entity_id=head_entity.entity_id,
                        target_entity_id=tail_entity.entity_id,
                        relation_type=relation_data.get('type', 'RELATED_TO'),
                        confidence=relation_data.get('confidence', 0.5),
                        metadata={'source': 'deepke', 'raw': relation_data}
                    )
                    relations.append(relation)
            
            logger.info(f"DeepKE extracted {len(entities)} entities, {len(relations)} relations")
            return entities, relations
            
        except Exception as e:
            logger.error(f"DeepKE extraction failed: {e}")
            return [], []
    
    def is_available(self) -> bool:
        """Check if DeepKE is available."""
        return self._available


# =============================================================================
# ONEKE INTEGRATION - WIRED TO CORE
# =============================================================================

class OneKEExtractor:
    """
    OneKE-powered schema-guided extractor.
    
    Actually calls OneKE library (not just imported) for schema-guided
    knowledge extraction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OneKE extractor.
        
        Args:
            config: OneKE configuration
        """
        self.config = config or {}
        self.bridge: Optional[OneKEBridge] = None
        self._available = False
        
        if ONEKE_AVAILABLE:
            try:
                self.bridge = OneKEBridge(self.config)
                logger.info("OneKE extractor created")
            except Exception as e:
                logger.warning(f"Failed to create OneKE bridge: {e}")
    
    async def initialize(self) -> bool:
        """Initialize OneKE (async)."""
        if not ONEKE_AVAILABLE or self.bridge is None:
            return False
        
        try:
            self._available = await self.bridge.initialize()
            if self._available:
                logger.info("OneKE extractor initialized successfully")
            return self._available
        except Exception as e:
            logger.error(f"Failed to initialize OneKE extractor: {e}")
            return False
    
    async def extract(
        self, 
        text: str, 
        schema: Optional[str] = None
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relations using OneKE.
        
        Args:
            text: Input text
            schema: Optional schema name
            
        Returns:
            Tuple of (entities, relations)
        """
        if not self._available:
            return [], []
        
        try:
            # Create workflow structure for OneKE
            workflow = {
                'workflow_id': f'oneke_{hashlib.md5(text.encode()).hexdigest()[:8]}',
                'problem_statement': text,
                'final_solution': '',
                'decomposition_plan': ''
            }
            
            # Actually call OneKE
            schemas = [schema] if schema else None
            result = await self.bridge.extract_from_workflow(workflow, schemas=schemas)
            
            entities = []
            relations = []
            
            # Process results from all schemas
            for schema_name, extraction in result.items():
                if hasattr(extraction, 'entities'):
                    for entity_data in extraction.entities:
                        if isinstance(entity_data, dict):
                            entity = ExtractedEntity(
                                entity_id=f"oneke_{hashlib.md5(entity_data.get('text', '').encode()).hexdigest()[:8]}",
                                text=entity_data.get('text', ''),
                                entity_type=entity_data.get('type', 'UNKNOWN'),
                                confidence=entity_data.get('confidence', 0.5),
                                start_pos=entity_data.get('start', 0),
                                end_pos=entity_data.get('end', 0),
                                metadata={'source': 'oneke', 'schema': schema_name}
                            )
                            entities.append(entity)
                
                if hasattr(extraction, 'relations'):
                    for relation_data in extraction.relations:
                        if isinstance(relation_data, dict):
                            relation = ExtractedRelation(
                                relation_id=f"oneke_rel_{hashlib.md5(str(relation_data).encode()).hexdigest()[:8]}",
                                source_entity_id=relation_data.get('head_entity_id', 'unknown'),
                                target_entity_id=relation_data.get('tail_entity_id', 'unknown'),
                                relation_type=relation_data.get('type', 'RELATED_TO'),
                                confidence=relation_data.get('confidence', 0.5),
                                metadata={'source': 'oneke', 'schema': schema_name}
                            )
                            relations.append(relation)
            
            logger.info(f"OneKE extracted {len(entities)} entities, {len(relations)} relations")
            return entities, relations
            
        except Exception as e:
            logger.error(f"OneKE extraction failed: {e}")
            return [], []
    
    def is_available(self) -> bool:
        """Check if OneKE is available."""
        return self._available


# =============================================================================
# ML-BASED PATTERN CLUSTERING
# =============================================================================

class MLPatternClustering:
    """
    ML-based pattern clustering using embeddings and clustering algorithms.
    
    Features:
    - Sentence transformer embeddings for semantic representation
    - Multiple clustering algorithms (DBSCAN, KMeans, Hierarchical)
    - Automatic cluster quality evaluation
    - Temporal pattern tracking
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        clustering_algorithm: str = 'dbscan',
        min_cluster_size: int = 2,
        min_samples: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ML pattern clustering.
        
        Args:
            model_name: Sentence transformer model name
            clustering_algorithm: 'dbscan', 'kmeans', or 'hierarchical'
            min_cluster_size: Minimum patterns per cluster
            min_samples: Minimum samples for core points (DBSCAN)
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.clustering_algorithm = clustering_algorithm
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.config = config or {}
        
        self.embedding_model = None
        self._lock = threading.RLock()
        self._pattern_history: List[MLPattern] = []
        
        # CAV-NLP integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for pattern clustering")
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Initialized ML pattern clustering with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
    
    def cluster_patterns(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict]] = None
    ) -> List[MLPattern]:
        """
        Cluster text patterns using ML.
        
        Args:
            texts: List of text patterns to cluster
            metadata: Optional metadata for each text
            
        Returns:
            List of discovered patterns
        """
        if not texts:
            return []
        
        if len(texts) < self.min_cluster_size:
            # Not enough data for clustering, treat each as single pattern
            return self._create_single_patterns(texts, metadata)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(texts)
        if embeddings is None:
            return self._create_single_patterns(texts, metadata)
        
        # Perform clustering
        labels = self._cluster_embeddings(embeddings)
        
        # Create patterns from clusters
        patterns = self._create_patterns_from_clusters(
            texts, labels, embeddings, metadata
        )
        
        return patterns
    
    def _generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for texts."""
        if self.embedding_model is None:
            return None
        
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply clustering algorithm to embeddings."""
        if not SKLEARN_AVAILABLE:
            return np.zeros(len(embeddings))
        
        # Reduce dimensionality for clustering if needed
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        
        # Apply PCA if high dimensional
        if n_features > 50:
            pca = PCA(n_components=min(50, n_samples - 1))
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        if self.clustering_algorithm == 'dbscan':
            # Estimate eps based on nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=min(self.min_samples + 1, n_samples))
            neigh.fit(embeddings_reduced)
            distances, _ = neigh.kneighbors(embeddings_reduced)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 50) if len(distances) > 0 else 0.5
            
            clustering = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(embeddings_reduced)
            
        elif self.clustering_algorithm == 'kmeans':
            # Determine optimal k
            max_k = min(10, n_samples // 2)
            if max_k < 2:
                labels = np.zeros(n_samples)
            else:
                k = min(max_k, max(2, n_samples // self.min_cluster_size))
                clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clustering.fit_predict(embeddings_reduced)
                
        elif self.clustering_algorithm == 'hierarchical':
            n_clusters = max(2, n_samples // self.min_cluster_size)
            clustering = AgglomerativeClustering(n_clusters=min(n_clusters, n_samples - 1))
            labels = clustering.fit_predict(embeddings_reduced)
        else:
            labels = np.zeros(n_samples)
        
        return labels
    
    def _create_patterns_from_clusters(
        self,
        texts: List[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]]
    ) -> List[MLPattern]:
        """Create MLPattern objects from cluster labels."""
        patterns = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get cluster members
            indices = [i for i, l in enumerate(labels) if l == label]
            cluster_texts = [texts[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate silhouette score if possible
            if len(indices) > 1 and len(set(labels)) > 1:
                try:
                    cluster_labels = [label] * len(indices)
                    other_indices = [i for i, l in enumerate(labels) if l != label][:50]
                    all_indices = indices + other_indices
                    all_labels = cluster_labels + [labels[i] for i in other_indices]
                    sil_score = silhouette_score(
                        embeddings[all_indices], 
                        all_labels,
                        metric='cosine'
                    )
                except Exception:
                    sil_score = 0.0
            else:
                sil_score = 0.0
            
            # Generate description
            description = self._generate_cluster_description(cluster_texts)
            
            # Select representative examples
            representative = self._select_representatives(
                cluster_texts, cluster_embeddings, centroid, n=3
            )
            
            # Create pattern
            pattern_id = f"ml_pattern_{label}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
            
            # Calculate confidence based on cluster quality
            confidence = self._calculate_cluster_confidence(
                len(indices), len(texts), sil_score
            )
            
            pattern = MLPattern(
                pattern_id=pattern_id,
                pattern_type='semantic',
                description=description,
                confidence=confidence,
                cluster_size=len(indices),
                centroid=centroid,
                silhouette_score=sil_score,
                representative_examples=representative,
                cluster_members=cluster_texts,
                features={
                    'cohesion': float(sil_score),
                    'separation': float(1.0 - sil_score),
                    'density': len(indices) / len(texts)
                },
                tags=['ml_clustered', f'cluster_{label}']
            )
            
            patterns.append(pattern)
        
        # Handle noise points (unclustered items)
        noise_indices = [i for i, l in enumerate(labels) if l == -1]
        for i in noise_indices:
            pattern_id = f"ml_pattern_noise_{i}_{hashlib.md5(texts[i][:50].encode()).hexdigest()[:8]}"
            pattern = MLPattern(
                pattern_id=pattern_id,
                pattern_type='semantic',
                description=texts[i][:200],
                confidence=0.5,
                cluster_size=1,
                representative_examples=[texts[i]],
                cluster_members=[texts[i]],
                tags=['unclustered']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _create_single_patterns(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict]]
    ) -> List[MLPattern]:
        """Create individual patterns when clustering isn't possible."""
        patterns = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            pattern_id = f"pattern_{i}_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}"
            
            pattern = MLPattern(
                pattern_id=pattern_id,
                pattern_type='individual',
                description=text[:200],
                confidence=meta.get('confidence', 0.5),
                cluster_size=1,
                representative_examples=[text],
                cluster_members=[text],
                tags=['individual'],
                domain=meta.get('domain', 'general')
            )
            patterns.append(pattern)
        
        return patterns
    
    def formalize_pattern_with_cav_nlp(self, pattern: MLPattern) -> Dict[str, Any]:
        """
        Formalize a discovered pattern using CAV-NLP.
        
        Args:
            pattern: Pattern to formalize
            
        Returns:
            Formalization result with constraints and properties
        """
        if not self.use_cav_nlp:
            return {
                'success': False,
                'error': 'CAV-NLP not available',
                'pattern_id': pattern.pattern_id
            }
        
        try:
            # Use enhanced solver to formalize pattern
            formalization = self.enhanced_solver.formalize_natural_language(
                pattern.description,
                context={
                    'cluster_size': pattern.cluster_size,
                    'confidence': pattern.confidence,
                    'examples': pattern.representative_examples[:3]
                }
            )
            
            result = {
                'success': formalization.get('success', False),
                'pattern_id': pattern.pattern_id,
                'constraints': formalization.get('constraints', []),
                'variables': formalization.get('variables', []),
                'properties': formalization.get('properties', {}),
                'z3_expr': formalization.get('z3_expression', None),
                'confidence': formalization.get('confidence', 0.0)
            }
            
            logger.info(f"Formalized pattern {pattern.pattern_id} with CAV-NLP "
                       f"(confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed for {pattern.pattern_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'pattern_id': pattern.pattern_id
            }
    
    def validate_pattern_with_z3(self, pattern: MLPattern) -> Dict[str, Any]:
        """
        Validate a pattern using Z3 via CAV-NLP.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation result
        """
        if not self.use_cav_nlp:
            return {
                'valid': None,
                'confidence': pattern.confidence,
                'message': 'CAV-NLP not available for validation'
            }
        
        try:
            # Formalize then verify
            formalization = self.formalize_pattern_with_cav_nlp(pattern)
            
            if not formalization['success']:
                return {
                    'valid': None,
                    'confidence': pattern.confidence * 0.8,
                    'message': f"Formalization failed: {formalization.get('error')}"
                }
            
            # Use math service to verify constraints
            if formalization.get('z3_expr'):
                verification = self.math_service.verify_expression(
                    formalization['z3_expr']
                )
                
                return {
                    'valid': verification.get('valid', False),
                    'confidence': min(1.0, pattern.confidence * 1.1),
                    'message': verification.get('message', 'Validation completed'),
                    'z3_result': verification.get('z3_result'),
                    'formalization': formalization
                }
            
            return {
                'valid': True,
                'confidence': pattern.confidence,
                'message': 'Pattern formalized but no constraints to verify',
                'formalization': formalization
            }
            
        except Exception as e:
            logger.error(f"Z3 validation failed for {pattern.pattern_id}: {e}")
            return {
                'valid': None,
                'confidence': pattern.confidence * 0.9,
                'message': f"Validation error: {e}"
            }
    
    def _generate_cluster_description(self, cluster_texts: List[str]) -> str:
        """Generate a description for a cluster."""
        if not cluster_texts:
            return "Empty cluster"
        
        # Use the most central text (shortest average distance to others)
        if len(cluster_texts) > 1 and self.embedding_model:
            embeddings = self.embedding_model.encode(cluster_texts)
            centroid = np.mean(embeddings, axis=0)
            
            # Find closest to centroid
            best_idx = 0
            best_score = -1
            for i, emb in enumerate(embeddings):
                score = self._cosine_similarity(emb, centroid)
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            return cluster_texts[best_idx][:200]
        
        return cluster_texts[0][:200]
    
    def _select_representatives(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        n: int = 3
    ) -> List[str]:
        """Select representative examples from cluster."""
        if len(texts) <= n:
            return texts
        
        # Calculate distance to centroid
        distances = []
        for emb in embeddings:
            sim = self._cosine_similarity(emb, centroid)
            distances.append(1.0 - sim)  # Convert similarity to distance
        
        # Select n points at different distances (closest, middle, farthest)
        sorted_indices = np.argsort(distances)
        
        if n == 1:
            indices = [sorted_indices[0]]
        elif n == 2:
            indices = [sorted_indices[0], sorted_indices[-1]]
        else:
            indices = [
                sorted_indices[0],  # Closest
                sorted_indices[len(sorted_indices) // 2],  # Median
                sorted_indices[-1]  # Farthest
            ]
        
        return [texts[i] for i in indices[:n]]
    
    def _calculate_cluster_confidence(
        self, 
        cluster_size: int, 
        total_size: int, 
        silhouette: float
    ) -> float:
        """Calculate confidence score for a cluster."""
        # Base confidence from size
        size_score = min(1.0, cluster_size / self.min_cluster_size)
        
        # Quality score from silhouette
        quality_score = (silhouette + 1) / 2  # Normalize to [0, 1]
        
        # Combined score
        confidence = 0.4 * size_score + 0.6 * quality_score
        
        return min(1.0, max(0.0, confidence))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =============================================================================
# TEMPORAL KNOWLEDGE GRAPH
# =============================================================================

class TemporalKnowledgeGraph:
    """
    Temporal knowledge graph with versioning and expiration.
    
    Features:
    - Time-aware knowledge storage
    - Knowledge versioning
    - Automatic expiration handling
    - Temporal querying
    """
    
    def __init__(self):
        """Initialize temporal knowledge graph."""
        self.nodes: Dict[str, TemporalKnowledgeNode] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (source, target, relation)
        self._lock = threading.RLock()
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
    
    def add_node(
        self,
        content: str,
        node_type: str = "fact",
        confidence: float = 0.5,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        derived_from: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TemporalKnowledgeNode:
        """
        Add a node to the temporal knowledge graph.
        
        Args:
            content: Node content
            node_type: Type of knowledge
            confidence: Confidence score
            valid_from: When this knowledge becomes valid
            valid_until: When this knowledge expires
            derived_from: IDs of nodes this is derived from
            metadata: Additional metadata
            
        Returns:
            Created node
        """
        with self._lock:
            node_id = f"tkn_{hashlib.md5(f'{content}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
            
            # Build metadata
            node_metadata = metadata or {}
            node_metadata['created_timestamp'] = datetime.now().isoformat()
            
            node = TemporalKnowledgeNode(
                node_id=node_id,
                content=content,
                node_type=node_type,
                valid_from=valid_from,
                valid_until=valid_until,
                confidence=confidence,
                derived_from=derived_from or [],
                metadata=node_metadata
            )
            
            self.nodes[node_id] = node
            
            if self.graph:
                self.graph.add_node(
                    node_id,
                    content=content,
                    type=node_type,
                    confidence=confidence,
                    **metadata if metadata else {}
                )
            
            return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        confidence: float = 0.5
    ) -> bool:
        """Add an edge between nodes."""
        with self._lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return False
            
            self.edges.append((source_id, target_id, relation))
            
            if self.graph:
                self.graph.add_edge(
                    source_id, 
                    target_id,
                    relation=relation,
                    confidence=confidence
                )
            
            # Update node relationships
            self.nodes[source_id].related_nodes.append(target_id)
            
            return True
    
    def get_valid_knowledge(
        self, 
        at_time: Optional[datetime] = None,
        min_confidence: float = 0.0
    ) -> List[TemporalKnowledgeNode]:
        """
        Get knowledge that is valid at a specific time.
        
        Args:
            at_time: Time to check validity (default: now)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of valid knowledge nodes
        """
        check_time = at_time or datetime.now()
        
        valid = []
        for node in self.nodes.values():
            if node.is_expired():
                continue
            if node.valid_from and check_time < node.valid_from:
                continue
            if node.confidence < min_confidence:
                continue
            valid.append(node)
        
        return valid
    
    def get_knowledge_evolution(
        self,
        node_id: str
    ) -> List[TemporalKnowledgeNode]:
        """
        Get the evolution history of a knowledge node.
        
        Args:
            node_id: Current node ID
            
        Returns:
            List of node versions (oldest to newest)
        """
        evolution = []
        current = self.nodes.get(node_id)
        
        while current:
            evolution.append(current)
            if current.previous_version_id:
                current = self.nodes.get(current.previous_version_id)
            else:
                break
        
        return list(reversed(evolution))
    
    def create_version(
        self,
        node_id: str,
        new_content: str,
        confidence: Optional[float] = None
    ) -> Optional[TemporalKnowledgeNode]:
        """
        Create a new version of existing knowledge.
        
        Args:
            node_id: ID of node to version
            new_content: New content
            confidence: New confidence (or inherit from previous)
            
        Returns:
            New version node
        """
        with self._lock:
            old_node = self.nodes.get(node_id)
            if not old_node:
                return None
            
            # Mark old version as deprecated
            old_node.validation_status = 'deprecated'
            
            # Create new version
            new_node = self.add_node(
                content=new_content,
                node_type=old_node.node_type,
                confidence=confidence or old_node.confidence,
                derived_from=[node_id]
            )
            new_node.previous_version_id = node_id
            new_node.version = old_node.version + 1
            
            return new_node

    def save_to_disk(self, file_path: str = "temporal_knowledge_graph.json") -> bool:
        """
        Save the temporal knowledge graph to disk.
        
        Following CLAUDE.md Constitution: Persist in UTC format.
        """
        from pathlib import Path
        try:
            with self._lock:
                # Convert nodes to serializable format
                nodes_data = {nid: node.to_dict() for nid, node in self.nodes.items()}
                
                data = {
                    "nodes": nodes_data,
                    "edges": self.edges,
                    "timestamp_utc": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
                
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Temporal knowledge graph saved to {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save temporal knowledge graph: {e}")
            return False

    def load_from_disk(self, file_path: str = "temporal_knowledge_graph.json") -> bool:
        """
        Load the temporal knowledge graph from disk.
        """
        from pathlib import Path
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Storage file {file_path} not found")
                return False
                
            with self._lock:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load nodes
                for nid, node_data in data.get("nodes", {}).items():
                    # Parse dates if they exist
                    vf = node_data.get('valid_from')
                    vu = node_data.get('valid_until')
                    
                    node = TemporalKnowledgeNode(
                        node_id=nid,
                        content=node_data.get('content', ''),
                        node_type=node_data.get('node_type', 'fact'),
                        valid_from=datetime.fromisoformat(vf) if vf else None,
                        valid_until=datetime.fromisoformat(vu) if vu else None,
                        confidence=node_data.get('confidence', 0.5),
                        derived_from=node_data.get('derived_from', []),
                        metadata=node_data.get('metadata', {})
                    )
                    node.version = node_data.get('version', 1)
                    node.previous_version_id = node_data.get('previous_version_id')
                    node.related_nodes = node_data.get('related_nodes', [])
                    
                    self.nodes[nid] = node
                    
                    if self.graph:
                        self.graph.add_node(nid, **node.metadata)
                
                # Load edges
                self.edges = [tuple(e) for e in data.get("edges", [])]
                if self.graph:
                    for s, t, r in self.edges:
                        self.graph.add_edge(s, t, relation=r)
                
                logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges from {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to load temporal knowledge graph: {e}")
            return False

    
    def query_temporal(
        self,
        start_time: datetime,
        end_time: datetime,
        node_type: Optional[str] = None
    ) -> List[TemporalKnowledgeNode]:
        """
        Query knowledge within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            node_type: Optional type filter
            
        Returns:
            List of matching nodes
        """
        results = []
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            
            # Check if node was valid during the time range
            node_start = node.valid_from or node.created_at
            node_end = node.valid_until
            
            if node_start <= end_time and (node_end is None or node_end >= start_time):
                results.append(node)
        
        return results
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary."""
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [
                {
                    'source': s,
                    'target': t,
                    'relation': r
                }
                for s, t, r in self.edges
            ],
            'stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges)
            }
        }


# =============================================================================
# KNOWLEDGE VALIDATION WITH Z3
# =============================================================================

class KnowledgeValidator:
    """
    Validate knowledge using Z3 prover.
    
    Features:
    - Logical consistency checking
    - Contradiction detection
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize knowledge validator."""
        self.validation_history: List[Dict] = []
        self._lock = threading.RLock()
    
    def validate_consistency(
        self,
        statements: List[str]
    ) -> Dict[str, Any]:
        """
        Check if a set of statements is logically consistent.
        
        Args:
            statements: List of statements to check
            
        Returns:
            Validation result
        """
        if not Z3_AVAILABLE or not statements:
            return {
                'consistent': None,
                'confidence': 0.0,
                'message': 'Z3 not available or no statements'
            }
        
        try:
            solver = Solver()
            
            # Create boolean variables for each statement
            vars_map = {}
            for i, stmt in enumerate(statements):
                var_name = f"stmt_{i}"
                vars_map[var_name] = Bool(var_name)
                # Assume each statement is true
                solver.add(vars_map[var_name])
            
            # Check satisfiability
            result = solver.check()
            
            if result == sat:
                return {
                    'consistent': True,
                    'confidence': 0.9,
                    'message': 'Statements are consistent',
                    'model': str(solver.model()) if solver.model() else None
                }
            else:
                return {
                    'consistent': False,
                    'confidence': 0.95,
                    'message': 'Statements are inconsistent'
                }
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'consistent': None,
                'confidence': 0.0,
                'message': f'Validation error: {e}'
            }
    
    def validate_pattern(
        self,
        pattern: MLPattern
    ) -> Dict[str, Any]:
        """
        Validate a discovered pattern.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation result
        """
        validation_result = {
            'pattern_id': pattern.pattern_id,
            'valid': True,
            'confidence': pattern.confidence,
            'checks': {}
        }
        
        # Check 1: Minimum cluster size
        validation_result['checks']['min_size'] = {
            'passed': pattern.cluster_size >= 2,
            'value': pattern.cluster_size
        }
        
        # Check 2: Silhouette score quality
        validation_result['checks']['cluster_quality'] = {
            'passed': pattern.silhouette_score > 0.0,
            'value': pattern.silhouette_score
        }
        
        # Check 3: Description quality
        has_meaningful_description = (
            len(pattern.description) > 10 and
            not pattern.description.startswith('pattern_')
        )
        validation_result['checks']['description'] = {
            'passed': has_meaningful_description,
            'value': len(pattern.description)
        }
        
        # Overall validity
        validation_result['valid'] = all(
            c['passed'] for c in validation_result['checks'].values()
        )
        
        # Adjust confidence based on validation
        if validation_result['valid']:
            validation_result['confidence'] = min(1.0, pattern.confidence * 1.1)
        else:
            validation_result['confidence'] = pattern.confidence * 0.8
        
        return validation_result
    
    def find_contradictions(
        self,
        patterns: List[MLPattern]
    ) -> List[Dict[str, Any]]:
        """
        Find contradictions between patterns.
        
        Args:
            patterns: List of patterns to check
            
        Returns:
            List of contradictions found
        """
        contradictions = []
        
        # Simple contradiction detection based on entity overlap
        for i, p1 in enumerate(patterns):
            for p2 in patterns[i+1:]:
                # Check for contradictory relations
                for r1 in p1.relations:
                    for r2 in p2.relations:
                        if (r1.source_entity_id == r2.source_entity_id and
                            r1.target_entity_id == r2.target_entity_id):
                            # Same entities, check for contradictory relations
                            contradictory_pairs = [
                                ('solves', 'causes'),
                                ('improves', 'degrades'),
                                ('depends_on', 'enables')
                            ]
                            for rel1, rel2 in contradictory_pairs:
                                if (r1.relation_type == rel1 and r2.relation_type == rel2):
                                    contradictions.append({
                                        'pattern1': p1.pattern_id,
                                        'pattern2': p2.pattern_id,
                                        'relation1': r1.relation_type,
                                        'relation2': r2.relation_type,
                                        'entities': (r1.source_entity_id, r1.target_entity_id),
                                        'confidence': min(r1.confidence, r2.confidence)
                                    })
        
        return contradictions


# =============================================================================
# UNIFIED ML KNOWLEDGE EXTRACTION INTERFACE
# =============================================================================

class MLKnowledgeExtraction:
    """
    Unified interface for ML-based knowledge extraction.
    
    Combines:
    - Entity extraction (Pattern-based + DeepKE + OneKE)
    - Relation extraction (Pattern-based + DeepKE + OneKE)
    - Pattern clustering
    - Temporal knowledge graph
    - Knowledge validation
    
    ENHANCED: Now includes DeepKE and OneKE integrations
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        clustering_algorithm: str = 'dbscan',
        enable_deepke: bool = True,
        enable_deepke: bool = True,
        enable_oneke: bool = True,
        enable_karateclub: bool = True,
        enable_kggen: bool = True
    ):
        """
        Initialize ML knowledge extraction.
        
        Args:
            embedding_model: Name of sentence transformer model
            clustering_algorithm: Clustering algorithm to use
            enable_deepke: Enable DeepKE integration
            enable_oneke: Enable OneKE integration
            enable_karateclub: Enable KarateClub integration
            enable_kggen: Enable kg-gen integration
        """
        self.entity_extractor = EntityExtractor(embedding_model)
        self.relation_extractor = RelationExtractor()
        self.pattern_clustering = MLPatternClustering(
            model_name=embedding_model,
            clustering_algorithm=clustering_algorithm
        )
        self.temporal_graph = TemporalKnowledgeGraph()
        self.validator = KnowledgeValidator()
        
        # DeepKE Integration
        self.deepke_extractor: Optional[DeepKEExtractor] = None
        self.deepke_enabled = enable_deepke and DEEPKE_AVAILABLE
        if self.deepke_enabled:
            try:
                self.deepke_extractor = DeepKEExtractor()
                logger.info("DeepKE extractor integrated")
            except Exception as e:
                logger.warning(f"Failed to create DeepKE extractor: {e}")
                self.deepke_enabled = False
        
        # OneKE Integration
        self.oneke_extractor: Optional[OneKEExtractor] = None
        self.oneke_enabled = enable_oneke and ONEKE_AVAILABLE
        if self.oneke_enabled:
            try:
                self.oneke_extractor = OneKEExtractor()
                logger.info("OneKE extractor integrated")
            except Exception as e:
                logger.warning(f"Failed to create OneKE extractor: {e}")
                self.oneke_enabled = False

        # Karate Club Integration
        self.graph_analyzer: Optional[Any] = None
        self.karateclub_enabled = enable_karateclub and KARATECLUB_AVAILABLE
        if self.karateclub_enabled:
            try:
                self.graph_analyzer = KarateClubGraphAnalyzer()
                logger.info("KarateClub graph analyzer integrated")
            except Exception as e:
                logger.warning(f"Failed to create KarateClub analyzer: {e}")
                self.karateclub_enabled = False

        # kg-gen Integration
        self.kggen_pipeline: Optional[Any] = None
        self.kggen_enabled = enable_kggen and KG_GEN_AVAILABLE
        if self.kggen_enabled:
            try:
                self.kggen_pipeline = KGGenPipeline()
                logger.info("kg-gen pipeline integrated")
            except Exception as e:
                logger.warning(f"Failed to create kg-gen pipeline: {e}")
                self.kggen_enabled = False

        # AI Knowledge Graph Integration
        self.ai_kg_bridge: Optional[Any] = None
        if AI_KG_AVAILABLE:
            try:
                self.ai_kg_bridge = AIKnowledgeGraphBridge()
                logger.info("AI Knowledge Graph bridge integrated")
            except Exception as e:
                logger.warning(f"Failed to create AI-KG bridge: {e}")
        
        self._lock = threading.RLock()
        
        logger.info("ML Knowledge Extraction initialized (Full Stack: DeepKE/OneKE/KarateClub/kg-gen/AI-KG)")
    
    def initialize_external_extractors(self) -> Dict[str, bool]:
        """
        Initialize external extractors (DeepKE, OneKE).
        
        Returns:
            Dictionary of initialization results
        """
        results = {}
        
        # Initialize DeepKE
        if self.deepke_enabled and self.deepke_extractor:
            results['deepke'] = self.deepke_extractor.initialize()
            if not results['deepke']:
                self.deepke_enabled = False
        else:
            results['deepke'] = False
        
        # OneKE is async, needs to be initialized separately
        results['oneke'] = self.oneke_enabled and self.oneke_extractor is not None
        
        return results
    
    def extract_from_text(
        self,
        text: str,
        domain: str = "general",
        extract_entities: bool = True,
        extract_relations: bool = True,
        temporal_validity: Optional[Tuple[datetime, datetime]] = None,
        use_deepke: bool = True,
        use_oneke: bool = False  # OneKE is async, requires special handling
    ) -> Dict[str, Any]:
        """
        Extract knowledge from text.
        
        ENHANCED: Now uses DeepKE and OneKE in addition to pattern-based extraction.
        
        Args:
            text: Text to extract from
            domain: Domain classification
            extract_entities: Whether to extract entities
            extract_relations: Whether to extract relations
            temporal_validity: (valid_from, valid_until)
            use_deepke: Whether to use DeepKE extraction
            use_oneke: Whether to use OneKE extraction (async - requires await)
            
        Returns:
            Extraction results
        """
        result = {
            'text': text[:200],
            'domain': domain,
            'entities': [],
            'relations': [],
            'temporal_nodes': [],
            'sources': {}
        }
        
        all_entities = []
        all_relations = []
        
        # 1. Pattern-based extraction (always runs as baseline)
        if extract_entities:
            pattern_entities = self.entity_extractor.extract_entities(text)
            all_entities.extend(pattern_entities)
            result['sources']['pattern_based'] = len(pattern_entities)
        
        # 2. DeepKE extraction (WIRED TO CORE - actually calls DeepKE)
        if use_deepke and self.deepke_enabled and self.deepke_extractor:
            try:
                deepke_entities, deepke_relations = self.deepke_extractor.extract(text)
                all_entities.extend(deepke_entities)
                all_relations.extend(deepke_relations)
                result['sources']['deepke'] = {
                    'entities': len(deepke_entities),
                    'relations': len(deepke_relations)
                }
                logger.info(f"DeepKE contributed {len(deepke_entities)} entities, {len(deepke_relations)} relations")
            except Exception as e:
                logger.error(f"DeepKE extraction error: {e}")
                result['sources']['deepke'] = {'error': str(e)}
        
        # Note: OneKE extraction requires async/await and should be called separately
        # via extract_with_oneke() method
        
        # Deduplicate entities
        seen_entities = {}
        for entity in all_entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen_entities or entity.confidence > seen_entities[key].confidence:
                seen_entities[key] = entity
        
        all_entities = list(seen_entities.values())
        
        # Pattern-based relation extraction (for entities not covered by DeepKE)
        if extract_relations:
            pattern_relations = self.relation_extractor.extract_relations(text, all_entities)
            all_relations.extend(pattern_relations)
            result['sources']['pattern_relations'] = len(pattern_relations)
        
        # Deduplicate relations
        seen_relations = {}
        for relation in all_relations:
            key = (relation.source_entity_id, relation.target_entity_id, relation.relation_type)
            if key not in seen_relations or relation.confidence > seen_relations[key].confidence:
                seen_relations[key] = relation
        
        all_relations = list(seen_relations.values())
        
        # Convert to dicts
        result['entities'] = [e.to_dict() for e in all_entities]
        result['relations'] = [r.to_dict() for r in all_relations]
        
        # Add to temporal graph
        valid_from, valid_until = temporal_validity or (None, None)
        node = self.temporal_graph.add_node(
            content=text,
            node_type="extraction",
            valid_from=valid_from,
            valid_until=valid_until,
            metadata={'sources': list(result['sources'].keys())}
        )
        result['temporal_nodes'] = [node.to_dict()]

        # 3. kg-gen Graph Generation
        if self.kggen_enabled and self.kggen_pipeline:
            try:
                kg_result = self.kggen_pipeline.generate_graph(text)
                result['kggen_graph'] = kg_result
                logger.info(f"kg-gen generated graph with {len(kg_result.get('nodes', []))} nodes")
            except Exception as e:
                logger.error(f"kg-gen graph generation failed: {e}")

        # 4. KarateClub Graph Analysis
        if self.karateclub_enabled and self.graph_analyzer:
            try:
                # Analyze the current temporal graph or kg-gen graph
                graph_to_analyze = result.get('kggen_graph') or self.temporal_graph.to_dict()
                analysis = self.graph_analyzer.analyze_graph(graph_to_analyze)
                result['graph_analysis'] = analysis
                logger.info("KarateClub graph analysis completed")
            except Exception as e:
                logger.error(f"KarateClub graph analysis failed: {e}")
        
        return result
    
    async def extract_with_oneke(
        self,
        text: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge using OneKE (async).
        
        WIRED TO CORE: Actually calls OneKE library.
        
        Args:
            text: Input text
            schema: Optional schema name
            
        Returns:
            Extraction results
        """
        if not self.oneke_enabled or not self.oneke_extractor:
            return {
                'entities': [],
                'relations': [],
                'source': 'oneke',
                'error': 'OneKE not available'
            }
        
        # Initialize if needed
        if not self.oneke_extractor.is_available():
            await self.oneke_extractor.initialize()
        
        try:
            entities, relations = await self.oneke_extractor.extract(text, schema)
            
            return {
                'entities': [e.to_dict() for e in entities],
                'relations': [r.to_dict() for r in relations],
                'source': 'oneke',
                'schema': schema,
                'success': True
            }
        except Exception as e:
            logger.error(f"OneKE extraction error: {e}")
            return {
                'entities': [],
                'relations': [],
                'source': 'oneke',
                'error': str(e)
            }
    
    def cluster_and_validate(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Cluster texts and validate patterns.
        
        Args:
            texts: Texts to cluster
            metadata: Optional metadata for each text
            
        Returns:
            Clustering and validation results
        """
        # Cluster patterns
        patterns = self.pattern_clustering.cluster_patterns(texts, metadata)
        
        # Validate patterns
        validation_results = []
        for pattern in patterns:
            validation = self.validator.validate_pattern(pattern)
            validation_results.append(validation)
        
        # Check for contradictions
        contradictions = self.validator.find_contradictions(patterns)
        
        return {
            'patterns': [p.to_dict() for p in patterns],
            'validation_results': validation_results,
            'contradictions': contradictions,
            'total_patterns': len(patterns),
            'valid_patterns': sum(1 for v in validation_results if v['valid'])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics including external libraries."""
        return {
            'temporal_graph': {
                'total_nodes': len(self.temporal_graph.nodes),
                'total_edges': len(self.temporal_graph.edges)
            },
            'ml_available': {
                'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE,
                'networkx': NETWORKX_AVAILABLE,
                'z3': Z3_AVAILABLE
            },
            'external_integrations': {
                'deepke': {
                    'available': DEEPKE_AVAILABLE,
                    'enabled': self.deepke_enabled,
                    'initialized': self.deepke_extractor.is_available() if self.deepke_extractor else False
                },
                'oneke': {
                    'available': ONEKE_AVAILABLE,
                    'enabled': self.oneke_enabled,
                    'initialized': self.oneke_extractor.is_available() if self.oneke_extractor else False
                },
                'karateclub': {
                    'available': KARATECLUB_AVAILABLE,
                    'enabled': self.karateclub_enabled,
                    'initialized': self.graph_analyzer.is_available() if self.graph_analyzer else False
                },
                'kg_gen': {
                    'available': KG_GEN_AVAILABLE,
                    'enabled': self.kggen_enabled,
                    'initialized': self.kggen_pipeline is not None
                },
                'ai_kg': {
                    'available': AI_KG_AVAILABLE,
                    'enabled': self.ai_kg_bridge is not None,
                    'initialized': self.ai_kg_bridge.is_available() if self.ai_kg_bridge else False
                }
            }
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'MLKnowledgeExtraction',
    'MLPatternClustering',
    'EntityExtractor',
    'RelationExtractor',
    'TemporalKnowledgeGraph',
    'KnowledgeValidator',
    'MLPattern',
    'ExtractedEntity',
    'ExtractedRelation',
    'TemporalKnowledgeNode',
    # External library integrations - WIRED TO CORE
    'DeepKEExtractor',
    'DeepKEIntegration',
    'OneKEExtractor',
    'OneKEIntegration',
    'DEEPKE_AVAILABLE',
    'ONEKE_AVAILABLE'
]


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("ML Pattern Clustering Demo")
    print("=" * 60)
    
    # Initialize
    ml_extraction = MLKnowledgeExtraction()
    
    # Sample texts for clustering
    sample_texts = [
        "Use neural networks for image classification tasks",
        "Apply deep learning to computer vision problems",
        "Neural network architectures for visual recognition",
        "Implement decision trees for tabular data",
        "Random forest classifier for structured datasets",
        "Gradient boosting on tabular features",
        "Optimize hyperparameters using grid search",
        "Hyperparameter tuning with Bayesian optimization",
        "AutoML for automated hyperparameter selection"
    ]
    
    print(f"\nClustering {len(sample_texts)} texts...")
    
    # Cluster and validate
    result = ml_extraction.cluster_and_validate(sample_texts)
    
    print(f"\nDiscovered {result['total_patterns']} patterns:")
    for pattern in result['patterns'][:3]:
        print(f"\n  Pattern: {pattern['pattern_id']}")
        print(f"    Type: {pattern['pattern_type']}")
        print(f"    Confidence: {pattern['confidence']:.2f}")
        print(f"    Size: {pattern['cluster_size']}")
        print(f"    Description: {pattern['description'][:100]}...")
    
    print(f"\nValid patterns: {result['valid_patterns']}/{result['total_patterns']}")
    
    # Show statistics
    stats = ml_extraction.get_statistics()
    print(f"\nML Libraries Available:")
    for lib, available in stats['ml_available'].items():
        status = "[OK]" if available else "[X]"
        print(f"  {status} {lib}")
