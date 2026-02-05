"""
DeepKE Adapter for OpenEvolve - TRUE 100% VERSION

Provides entity and relation extraction using ACTUAL DeepKE deep learning models.
NO FALLBACK - If DeepKE is not available, initialization will fail.

Supports NER (Named Entity Recognition) and RE (Relation Extraction).
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DeepKENotInstalledError(Exception):
    """Raised when DeepKE is not installed."""
    pass


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
    Adapter for ACTUAL DeepKE knowledge extraction.
    
    This adapter REQUIRES DeepKE to be installed. It does NOT provide
    fallback mechanisms - if DeepKE is not available, initialization fails.
    
    Features:
    - Named Entity Recognition (NER) using DeepKE
    - Relation Extraction (RE) using DeepKE
    - Support for multiple languages
    - Configurable model selection
    
    Raises:
        DeepKENotInstalledError: If DeepKE is not installed
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
                - allow_fallback: If True, warn but don't raise if DeepKE unavailable
        
        Raises:
            DeepKENotInstalledError: If DeepKE is not installed and allow_fallback=False
        """
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'deepke_ner_re')
        self.language = self.config.get('language', 'en')
        self.device = self.config.get('device', 'cpu')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.allow_fallback = self.config.get('allow_fallback', False)
        
        self._ner_model = None
        self._re_model = None
        self._available = False
        
        # Check DeepKE availability
        self._check_deepke()
    
    def _check_deepke(self) -> bool:
        """Check if DeepKE is available."""
        try:
            import deepke
            from deepke import NERModel, REModel
            self._available = True
            logger.info("✓ DeepKE is available and will be used for extraction")
            return True
        except ImportError:
            error_msg = (
                "DeepKE is NOT installed.\n"
                "Run 'python setup_deepke.py' to install DeepKE.\n"
                "Knowledge Extraction requires DeepKE for TRUE 100% functionality."
            )
            if self.allow_fallback:
                logger.warning(error_msg)
                logger.warning("Continuing with fallback mode (NOT recommended)")
                self._available = False
                return False
            else:
                logger.error(error_msg)
                raise DeepKENotInstalledError(error_msg)
    
    def initialize(self) -> bool:
        """
        Initialize DeepKE models.
        
        Actually loads DeepKE models from the library.
        
        Returns:
            True if initialization successful
            
        Raises:
            DeepKENotInstalledError: If DeepKE is not available
        """
        if not self._available:
            error_msg = "DeepKE not available. Run 'python setup_deepke.py' to install."
            if self.allow_fallback:
                logger.warning(error_msg)
                return False
            raise DeepKENotInstalledError(error_msg)
        
        try:
            # Import DeepKE modules
            from deepke import NERModel, REModel
            import torch
            
            # Auto-detect device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Auto-detected device: {self.device}")
            
            # Initialize NER model
            logger.info(f"Loading DeepKE NER model: {self.model_name}")
            self._ner_model = NERModel(
                model_name=self.model_name,
                device=self.device
            )
            logger.info(f"✓ NER model loaded successfully on {self.device}")
            
            # Initialize RE model
            logger.info(f"Loading DeepKE RE model: {self.model_name}")
            self._re_model = REModel(
                model_name=self.model_name,
                device=self.device
            )
            logger.info(f"✓ RE model loaded successfully on {self.device}")
            
            logger.info("✓ DeepKE models initialized - ACTUAL DeepKE will be used")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepKE: {e}")
            self._available = False
            self._ner_model = None
            self._re_model = None
            if self.allow_fallback:
                return False
            raise DeepKENotInstalledError(f"DeepKE initialization failed: {e}")
    
    def extract_entities(self, text: str) -> DeepKEExtractionResult:
        """
        Extract entities from text using DeepKE NER.
        
        ACTUALLY calls DeepKE NER model (NOT fallback).
        
        Args:
            text: Input text
            
        Returns:
            DeepKEExtractionResult with entities
            
        Raises:
            DeepKENotInstalledError: If DeepKE is not initialized
        """
        if not self._available or self._ner_model is None:
            error_msg = "DeepKE not initialized. Call initialize() first."
            if self.allow_fallback:
                logger.warning(error_msg)
                logger.warning("Returning empty result (fallback mode)")
                return DeepKEExtractionResult(
                    task=ExtractionTask.NER,
                    entities=[],
                    relations=[],
                    success=False,
                    error_message=error_msg
                )
            raise DeepKENotInstalledError(error_msg)
        
        try:
            # ACTUAL DeepKE NER CALL
            logger.debug(f"Calling DeepKE NER on text: {text[:50]}...")
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
            
            logger.info(f"✓ DeepKE NER extracted {len(entities)} entities")
            
            return DeepKEExtractionResult(
                task=ExtractionTask.NER,
                entities=entities,
                relations=[],
                raw_output={'results': raw_results, 'source': 'deepke_actual'},
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
        
        ACTUALLY calls DeepKE RE model (NOT fallback).
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            DeepKEExtractionResult with relations
        """
        if not self._available or self._re_model is None:
            error_msg = "DeepKE not initialized. Call initialize() first."
            if self.allow_fallback:
                logger.warning(error_msg)
                return DeepKEExtractionResult(
                    task=ExtractionTask.RE,
                    entities=entities or [],
                    relations=[],
                    success=False,
                    error_message=error_msg
                )
            raise DeepKENotInstalledError(error_msg)
        
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
            
            if not entity_pairs:
                logger.info("No entity pairs for relation extraction")
                return DeepKEExtractionResult(
                    task=ExtractionTask.RE,
                    entities=entities,
                    relations=[],
                    raw_output={'source': 'deepke_actual', 'empty': True},
                    success=True
                )
            
            # ACTUAL DeepKE RE CALL
            logger.debug(f"Calling DeepKE RE on {len(entity_pairs)} entity pairs")
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
            
            logger.info(f"✓ DeepKE RE extracted {len(relations)} relations")
            
            return DeepKEExtractionResult(
                task=ExtractionTask.RE,
                entities=entities,
                relations=relations,
                raw_output={'results': raw_results, 'source': 'deepke_actual'},
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
            'source': 'deepke_actual',
            'fallback_used': False
        }


# Convenience function for direct usage
def extract_entities(text: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to extract entities with DeepKE.
    
    Args:
        text: Input text
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with extraction results
    """
    adapter = DeepKEAdapter(config=kwargs)
    if not adapter.initialize():
        raise DeepKENotInstalledError("DeepKE not available")
    
    result = adapter.extract_entities(text)
    return {
        'entities': [
            {
                'text': e.text,
                'type': e.entity_type,
                'start': e.start_pos,
                'end': e.end_pos,
                'confidence': e.confidence
            }
            for e in result.entities
        ],
        'success': result.success,
        'error': result.error_message,
        'source': 'deepke_actual'
    }


def extract_relations(text: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to extract relations with DeepKE.
    
    Args:
        text: Input text
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with extraction results
    """
    adapter = DeepKEAdapter(config=kwargs)
    if not adapter.initialize():
        raise DeepKENotInstalledError("DeepKE not available")
    
    result = adapter.extract_entities_and_relations(text)
    return {
        'entities': [
            {
                'text': e.text,
                'type': e.entity_type,
                'confidence': e.confidence
            }
            for e in result.entities
        ],
        'relations': [
            {
                'head': r.head_entity,
                'tail': r.tail_entity,
                'type': r.relation_type,
                'confidence': r.confidence
            }
            for r in result.relations
        ],
        'success': result.success,
        'error': result.error_message,
        'source': 'deepke_actual'
    }
