"""
DeepKE Adapter for OpenEvolve

Provides entity and relation extraction using DeepKE's deep learning models.
Supports NER (Named Entity Recognition) and RE (Relation Extraction).
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionTask(Enum):
    """DeepKE extraction tasks."""
    NER = "ner"
    RE = "re"
    EE = "ee"


@dataclass
class DeepKEEntity:
    """Entity extracted by DeepKE."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepKERelation:
    """Relation extracted by DeepKE."""
    head_entity: str
    tail_entity: str
    relation_type: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepKEExtractionResult:
    """Result from DeepKE extraction."""
    task: ExtractionTask
    entities: List[DeepKEEntity]
    relations: List[DeepKERelation]
    raw_output: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""


class DeepKEAdapter:
    """
    Adapter for DeepKE knowledge extraction.
    
    Features:
    - Named Entity Recognition (NER)
    - Relation Extraction (RE)
    - Support for multiple languages
    - Configurable model selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepKE adapter.
        
        Args:
            config: Configuration dictionary with options:
                - model_name: Name of the DeepKE model
                - language: Language code (en, zh, etc.)
                - device: Device to run on (cpu, cuda)
                - confidence_threshold: Minimum confidence for extraction
        """
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'deepke_ner_re')
        self.language = self.config.get('language', 'en')
        self.device = self.config.get('device', 'cpu')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        self._ner_model = None
        self._re_model = None
        self._available = False
        
        # Try to import DeepKE
        self._check_deepke()
    
    def _check_deepke(self) -> bool:
        """Check if DeepKE is available."""
        try:
            import deepke
            self._available = True
            logger.info("DeepKE is available")
            return True
        except ImportError:
            logger.warning("DeepKE not installed. Using fallback extraction.")
            self._available = False
            return False
    
    def initialize(self) -> bool:
        """
        Initialize DeepKE models.
        
        Returns:
            True if initialization successful
        """
        if not self._available:
            logger.warning("DeepKE not available, skipping initialization")
            return False
        
        try:
            # Import DeepKE modules
            from deepke import NERModel, REModel
            
            # Initialize NER model
            logger.info(f"Loading DeepKE NER model: {self.model_name}")
            self._ner_model = NERModel(
                model_name=self.model_name,
                device=self.device
            )
            
            # Initialize RE model
            logger.info(f"Loading DeepKE RE model: {self.model_name}")
            self._re_model = REModel(
                model_name=self.model_name,
                device=self.device
            )
            
            logger.info("DeepKE models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepKE: {e}")
            self._available = False
            return False
    
    def extract_entities(self, text: str) -> DeepKEExtractionResult:
        """
        Extract entities from text using DeepKE NER.
        
        Args:
            text: Input text
            
        Returns:
            DeepKEExtractionResult with entities
        """
        if not self._available or self._ner_model is None:
            return self._fallback_ner(text)
        
        try:
            # Call DeepKE NER
            raw_results = self._ner_model.predict(text)
            
            entities = []
            for result in raw_results:
                entity = DeepKEEntity(
                    text=result.get('text', ''),
                    entity_type=result.get('type', 'UNKNOWN'),
                    start_pos=result.get('start', 0),
                    end_pos=result.get('end', 0),
                    confidence=result.get('confidence', 0.0),
                    metadata=result.get('metadata', {})
                )
                if entity.confidence >= self.confidence_threshold:
                    entities.append(entity)
            
            return DeepKEExtractionResult(
                task=ExtractionTask.NER,
                entities=entities,
                relations=[],
                raw_output={'results': raw_results},
                success=True
            )
            
        except Exception as e:
            logger.error(f"DeepKE NER extraction failed: {e}")
            return DeepKEExtractionResult(
                task=ExtractionTask.NER,
                entities=[],
                relations=[],
                success=False,
                error_message=str(e)
            )
    
    def extract_relations(
        self, 
        text: str, 
        entities: Optional[List[DeepKEEntity]] = None
    ) -> DeepKEExtractionResult:
        """
        Extract relations from text using DeepKE RE.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            DeepKEExtractionResult with relations
        """
        if not self._available or self._re_model is None:
            return self._fallback_re(text, entities)
        
        try:
            # If entities not provided, extract them first
            if entities is None:
                ner_result = self.extract_entities(text)
                entities = ner_result.entities
            
            # Prepare entity pairs for relation extraction
            entity_pairs = []
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:]:
                    entity_pairs.append({
                        'head': ent1.text,
                        'tail': ent2.text,
                        'context': text
                    })
            
            # Call DeepKE RE
            raw_results = self._re_model.predict(entity_pairs)
            
            relations = []
            for result in raw_results:
                relation = DeepKERelation(
                    head_entity=result.get('head', ''),
                    tail_entity=result.get('tail', ''),
                    relation_type=result.get('relation', 'UNKNOWN'),
                    confidence=result.get('confidence', 0.0),
                    metadata=result.get('metadata', {})
                )
                if relation.confidence >= self.confidence_threshold:
                    relations.append(relation)
            
            return DeepKEExtractionResult(
                task=ExtractionTask.RE,
                entities=entities,
                relations=relations,
                raw_output={'results': raw_results},
                success=True
            )
            
        except Exception as e:
            logger.error(f"DeepKE RE extraction failed: {e}")
            return DeepKEExtractionResult(
                task=ExtractionTask.RE,
                entities=entities or [],
                relations=[],
                success=False,
                error_message=str(e)
            )
    
    def extract_entities_and_relations(self, text: str) -> DeepKEExtractionResult:
        """
        Extract both entities and relations from text.
        
        Args:
            text: Input text
            
        Returns:
            DeepKEExtractionResult with entities and relations
        """
        # First extract entities
        ner_result = self.extract_entities(text)
        
        if not ner_result.success:
            return ner_result
        
        # Then extract relations
        re_result = self.extract_relations(text, ner_result.entities)
        
        return DeepKEExtractionResult(
            task=ExtractionTask.RE,
            entities=re_result.entities,
            relations=re_result.relations,
            raw_output={
                'ner': ner_result.raw_output,
                're': re_result.raw_output
            },
            success=re_result.success,
            error_message=re_result.error_message
        )
    
    def batch_extract(
        self, 
        texts: List[str], 
        task: ExtractionTask = ExtractionTask.NER
    ) -> List[DeepKEExtractionResult]:
        """
        Batch extract from multiple texts.
        
        Args:
            texts: List of input texts
            task: Extraction task to perform
            
        Returns:
            List of extraction results
        """
        results = []
        for text in texts:
            if task == ExtractionTask.NER:
                result = self.extract_entities(text)
            elif task == ExtractionTask.RE:
                result = self.extract_relations(text)
            else:
                result = self.extract_entities_and_relations(text)
            results.append(result)
        return results
    
    def _fallback_ner(self, text: str) -> DeepKEExtractionResult:
        """
        Fallback NER using pattern matching when DeepKE is unavailable.
        
        Args:
            text: Input text
            
        Returns:
            DeepKEExtractionResult with pattern-based entities
        """
        import re
        
        entities = []
        
        # Pattern-based entity extraction
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-z]* (?:Inc|Corp|Ltd|Company)\b',
            'TECH': r'\b(?:neural network|machine learning|deep learning|AI)\b',
            'CONCEPT': r'\b(?:algorithm|model|system|framework)\b',
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(DeepKEEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.5,
                    metadata={'source': 'fallback'}
                ))
        
        return DeepKEExtractionResult(
            task=ExtractionTask.NER,
            entities=entities,
            relations=[],
            raw_output={'fallback': True},
            success=True
        )
    
    def _fallback_re(
        self, 
        text: str, 
        entities: Optional[List[DeepKEEntity]] = None
    ) -> DeepKEExtractionResult:
        """
        Fallback RE using pattern matching when DeepKE is unavailable.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            DeepKEExtractionResult with pattern-based relations
        """
        import re
        
        relations = []
        
        # Pattern-based relation extraction
        relation_patterns = {
            'USES': r'(\w+)\s+uses?\s+(\w+)',
            'IMPLEMENTS': r'(\w+)\s+implements?\s+(\w+)',
            'DEPENDS_ON': r'(\w+)\s+depends?\s+on\s+(\w+)',
        }
        
        for relation_type, pattern in relation_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                relations.append(DeepKERelation(
                    head_entity=match.group(1),
                    tail_entity=match.group(2),
                    relation_type=relation_type,
                    confidence=0.5,
                    metadata={'source': 'fallback'}
                ))
        
        return DeepKEExtractionResult(
            task=ExtractionTask.RE,
            entities=entities or [],
            relations=relations,
            raw_output={'fallback': True},
            success=True
        )
    
    def is_available(self) -> bool:
        """Check if DeepKE is available and initialized."""
        return self._available and self._ner_model is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'available': self._available,
            'model_name': self.model_name,
            'language': self.language,
            'device': self.device,
            'ner_model_loaded': self._ner_model is not None,
            're_model_loaded': self._re_model is not None,
        }
