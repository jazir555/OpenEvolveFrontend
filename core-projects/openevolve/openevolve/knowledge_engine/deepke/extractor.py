"""
DeepKE Entity and Relation Extractor

Extracts entities and relations from text using DeepKE models.

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
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import DeepKE
try:
    from deepke import *
    DEEPKE_AVAILABLE = True
except ImportError:
    DEEPKE_AVAILABLE = False
    logger.warning("DeepKE not available, using mock extraction")


class EntityType(Enum):
    """Common entity types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FAC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECH"
    CODE = "CODE"
    ALGORITHM = "ALGO"
    DATA_STRUCTURE = "DATA_STRUCT"
    API = "API"


class RelationType(Enum):
    """Common relation types"""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    FOUNDED_BY = "founded_by"
    FOUNDED_ON = "founded_on"
    PART_OF = "part_of"
    USES = "uses"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    CALLS = "calls"
    EXTENDS = "extends"
    RELATED_TO = "related_to"
    AUTHOR_OF = "author_of"
    PUBLISHED_ON = "published_on"
    MENTIONS = "mentions"


@dataclass
class ExtractedEntity:
    """An extracted entity"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.normalized_text is None:
            self.normalized_text = self.text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start_pos,
            "end": self.end_pos,
            "confidence": self.confidence,
            "normalized": self.normalized_text
        }


@dataclass
class ExtractedRelation:
    """An extracted relation between entities"""
    subject: ExtractedEntity
    predicate: RelationType
    object: ExtractedEntity
    confidence: float = 1.0
    sentence_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate.value,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
            "context": self.sentence_context
        }
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to (subject, predicate, object) triple"""
        return (
            self.subject.normalized_text,
            self.predicate.value,
            self.object.normalized_text
        )


@dataclass
class ExtractionResult:
    """Result of extraction from text"""
    text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    language: str = "en"
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "language": self.language,
            "processing_time": self.processing_time
        }
    
    def get_triples(self) -> List[Tuple[str, str, str]]:
        """Get all relations as triples"""
        return [r.to_triple() for r in self.relations]


class EntityExtractor:
    """Extract entities from text"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "default"
        self.model = None
        
        if DEEPKE_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the DeepKE entity extraction model"""
        try:
            # Mock loading - would load actual DeepKE model
            logger.info(f"Loading entity extraction model: {self.model_name}")
            self.model = {"loaded": True, "name": self.model_name}
        except Exception as e:
            logger.error(f"Failed to load entity model: {e}")
            self.model = None
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text"""
        if not DEEPKE_AVAILABLE or not self.model:
            return self._mock_extract(text)
        
        try:
            # Would call actual DeepKE model here
            return self._mock_extract(text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _mock_extract(self, text: str) -> List[ExtractedEntity]:
        """Mock entity extraction using rules and patterns"""
        entities = []
        
        # Pattern-based extraction for demonstration
        patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?) [A-Z][a-z]+\b'
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]* (?:Inc|Corp|Ltd|LLC|Company)\b',
                r'\b(?:OpenAI|Google|Microsoft|Amazon|Facebook|Apple)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:Python|JavaScript|Java|C\+\+|Rust|Go|TypeScript)\b',
                r'\b(?:Neo4j|MongoDB|PostgreSQL|MySQL|Redis)\b',
                r'\b(?:Docker|Kubernetes|AWS|Azure|GCP)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(?:machine learning|artificial intelligence|deep learning|neural network)\b',
                r'\b(?:knowledge graph|entity extraction|relation extraction)\b'
            ],
            EntityType.CODE: [
                r'`([^`]+)`',
                r'\bfunction\s+(\w+)',
                r'\bclass\s+(\w+)'
            ]
        }
        
        for entity_type, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.7
                    ))
        
        # Remove overlapping entities (keep longer ones)
        entities.sort(key=lambda e: (e.start_pos, -e.end_pos))
        filtered = []
        last_end = -1
        for e in entities:
            if e.start_pos >= last_end:
                filtered.append(e)
                last_end = e.end_pos
        
        return filtered


class RelationExtractor:
    """Extract relations between entities"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "default"
        self.model = None
        
        if DEEPKE_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the DeepKE relation extraction model"""
        try:
            logger.info(f"Loading relation extraction model: {self.model_name}")
            self.model = {"loaded": True, "name": self.model_name}
        except Exception as e:
            logger.error(f"Failed to load relation model: {e}")
            self.model = None
    
    def extract(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations between entities"""
        if not DEEPKE_AVAILABLE or not self.model:
            return self._mock_extract(text, entities)
        
        try:
            return self._mock_extract(text, entities)
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    def _mock_extract(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Mock relation extraction based on proximity and patterns"""
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Pattern-based relation detection
        relation_patterns = [
            (r'\bworks? (?:at|for)\b', RelationType.WORKS_FOR),
            (r'\blocated (?:in|at)\b', RelationType.LOCATED_IN),
            (r'\bfounded (?:by)?\b', RelationType.FOUNDED_BY),
            (r'\bpart of\b', RelationType.PART_OF),
            (r'\buses?\b', RelationType.USES),
            (r'\bimplements?\b', RelationType.IMPLEMENTS),
            (r'\bdepends? (?:on|upon)\b', RelationType.DEPENDS_ON),
            (r'\bextends?\b', RelationType.EXTENDS),
            (r'\bcalls?\b', RelationType.CALLS),
        ]
        
        for sentence in sentences:
            # Find entities in this sentence
            sent_entities = [
                e for e in entities
                if e.start_pos >= text.find(sentence) and 
                   e.end_pos <= text.find(sentence) + len(sentence)
            ]
            
            if len(sent_entities) >= 2:
                # Check for relation patterns
                for pattern, rel_type in relation_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Create relation between first two entities
                        relations.append(ExtractedRelation(
                            subject=sent_entities[0],
                            predicate=rel_type,
                            object=sent_entities[1],
                            confidence=0.6,
                            sentence_context=sentence.strip()
                        ))
                        break
                
                # If no pattern match, create RELATED_TO for close entities
                if len(sent_entities) >= 2 and not any(
                    r.sentence_context == sentence.strip() for r in relations
                ):
                    for i in range(len(sent_entities) - 1):
                        if sent_entities[i+1].start_pos - sent_entities[i].end_pos < 50:
                            relations.append(ExtractedRelation(
                                subject=sent_entities[i],
                                predicate=RelationType.RELATED_TO,
                                object=sent_entities[i+1],
                                confidence=0.5,
                                sentence_context=sentence.strip()
                            ))
        
        return relations


class DeepKEExtractor:
    """Main DeepKE extraction interface"""
    
    def __init__(
        self,
        entity_model: Optional[str] = None,
        relation_model: Optional[str] = None,
        use_gpu: bool = False
    ):
        self.entity_extractor = EntityExtractor(entity_model)
        self.relation_extractor = RelationExtractor(relation_model)
        self.use_gpu = use_gpu
        
        if not DEEPKE_AVAILABLE:
            logger.warning("DeepKE not installed, using mock extraction")
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text"""
        import time
        start_time = time.time()
        
        # Extract entities
        entities = self.entity_extractor.extract(text)
        
        # Extract relations
        relations = self.relation_extractor.extract(text, entities)
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            text=text,
            entities=entities,
            relations=relations,
            processing_time=processing_time
        )
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract from multiple texts"""
        return [self.extract(text) for text in texts]
    
    def extract_from_document(
        self,
        document: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> ExtractionResult:
        """Extract from a large document by chunking"""
        # Split document into chunks
        chunks = []
        start = 0
        while start < len(document):
            end = min(start + chunk_size, len(document))
            chunks.append(document[start:end])
            start = end - overlap if end < len(document) else end
        
        # Extract from each chunk
        all_entities = []
        all_relations = []
        total_time = 0.0
        
        for chunk in chunks:
            result = self.extract(chunk)
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
            total_time += result.processing_time
        
        # Deduplicate entities
        seen = set()
        unique_entities = []
        for e in all_entities:
            key = (e.normalized_text.lower(), e.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        # Deduplicate relations
        seen_relations = set()
        unique_relations = []
        for r in all_relations:
            key = r.to_triple()
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)
        
        return ExtractionResult(
            text=document[:500] + "...",
            entities=unique_entities,
            relations=unique_relations,
            processing_time=total_time
        )
    
    def is_available(self) -> bool:
        """Check if DeepKE is available"""
        return DEEPKE_AVAILABLE
