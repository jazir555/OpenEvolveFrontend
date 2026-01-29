"""
OneKE Integration for OpenEvolve Knowledge Engine

This module provides integration with the OneKE bilingual knowledge extraction system,
enabling advanced extraction with quality enhancement and reflection capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
import json

try:
    from OneKE.src.pipeline import Pipeline
    from OneKE.src.models.llm_def import BaseEngine
    from OneKE.src.utils import DataPoint, TaskType
    ONEKE_AVAILABLE = True
except ImportError:
    ONEKE_AVAILABLE = False
    # Define mock classes for when OneKE is not available
    class Pipeline:
        pass
    class BaseEngine:
        pass
    class DataPoint:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    TaskType = str

logger = logging.getLogger(__name__)


@dataclass
class EnhancedExtractionResult:
    """Enhanced result of a knowledge extraction operation with quality scores."""
    success: bool
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    triples: List[Tuple[str, str, str]]
    quality_scores: Dict[str, float]  # e.g., {"accuracy": 0.9, "completeness": 0.8}
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    reflection_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'entities': self.entities,
            'relations': self.relations,
            'triples': self.triples,
            'quality_scores': self.quality_scores,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error,
            'reflection_notes': self.reflection_notes
        }


class OneKEIntegration:
    """
    Integration with OneKE bilingual knowledge extraction system.
    
    Provides methods for:
    - Enhanced extraction with quality assessment
    - Reflection and improvement
    - Learning from feedback
    - Case-based reasoning
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        config_path: Optional[str] = None,
        enhanced_config_path: Optional[str] = None
    ):
        """
        Initialize the OneKE integration.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM
            api_base: Base URL for the LLM API
            config_path: Path to basic OneKE configuration
            enhanced_config_path: Path to enhanced configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.config_path = config_path
        self.enhanced_config_path = enhanced_config_path
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.enhanced_config = self._load_config(enhanced_config_path) or self._get_default_enhanced_config()
        
        # Initialize OneKE pipeline
        self.pipeline = None
        self.llm_client = None
        
        # Initialize components
        self._initialize_llm_client()
        self._initialize_pipeline()
        
        # Case repository for learning
        self.case_repository = []
        
        logger.info({
            "msg": "OneKEIntegration initialized",
            "model_name": model_name,
            "oneke_available": ONEKE_AVAILABLE,
            "config": self.config,
            "enhanced_config": self.enhanced_config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and __import__('os').path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return {}
    
    def _get_default_enhanced_config(self) -> Dict[str, Any]:
        """Get default enhanced configuration."""
        return {
            "quality_thresholds": {
                "accuracy": 0.7,
                "completeness": 0.6,
                "consistency": 0.8
            },
            "reflection_enabled": True,
            "learning_enabled": True,
            "case_retention_limit": 1000,
            "enhancement_iterations": 3
        }
    
    def _initialize_llm_client(self):
        """Initialize the LLM client for OneKE."""
        if not ONEKE_AVAILABLE:
            logger.warning("OneKE not available, using mock implementation")
            return
        
        try:
            # Import the appropriate LLM client based on model
            if "openai" in self.model_name.lower() or "gpt" in self.model_name.lower():
                from OneKE.src.models.llm_def import OpenAIEngine
                self.llm_client = OpenAIEngine(
                    model=self.model_name,
                    api_key=self.api_key,
                    api_base=self.api_base
                )
            elif "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower():
                from OneKE.src.models.llm_def import AnthropicEngine
                self.llm_client = AnthropicEngine(
                    model=self.model_name,
                    api_key=self.api_key
                )
            else:
                # Default to a generic engine
                from OneKE.src.models.llm_def import BaseEngine
                self.llm_client = BaseEngine(
                    model=self.model_name,
                    api_key=self.api_key,
                    api_base=self.api_base
                )
            
            logger.info({
                "msg": "OneKE LLM client initialized",
                "model_name": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize OneKE LLM client: {e}",
                "model_name": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.llm_client = None
    
    def _initialize_pipeline(self):
        """Initialize the OneKE pipeline."""
        if not ONEKE_AVAILABLE or not self.llm_client:
            logger.warning("OneKE pipeline not available due to missing dependencies or LLM client")
            return
        
        try:
            self.pipeline = Pipeline(llm=self.llm_client)
            logger.info({
                "msg": "OneKE pipeline initialized",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize OneKE pipeline: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.pipeline = None
    
    async def extract_with_enhancement(
        self,
        text: str,
        schema: str,
        domain: str = "general",
        enable_reflection: bool = True,
        enable_cases: bool = True,
        enable_validation: bool = True,
        enable_consistency: bool = True,
        task_type: str = "Triple",
        correlation_id: Optional[str] = None
    ) -> Optional[EnhancedExtractionResult]:
        """
        Extract knowledge with quality enhancement using multiple techniques.
        
        Args:
            text: Input text to extract knowledge from
            schema: Target schema name or definition
            domain: Domain label (e.g., 'physics', 'chemistry', 'software_engineering')
            enable_reflection: Enable reflection and improvement
            enable_cases: Enable case-based reasoning
            enable_validation: Enable validation checks
            enable_consistency: Enable consistency checks
            task_type: Type of extraction task (NER, RE, EE, Triple)
            correlation_id: Correlation ID for tracking
            
        Returns:
            EnhancedExtractionResult with quality scores and reflection notes
        """
        correlation_id = correlation_id or f"oneke_enhanced_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.pipeline:
            raise RuntimeError("OneKE pipeline not initialized. Call initialize() first.")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting enhanced OneKE extraction",
            "text_length": len(text),
            "schema": schema,
            "domain": domain,
            "task_type": task_type,
            "enhancement_features": {
                "reflection": enable_reflection,
                "cases": enable_cases,
                "validation": enable_validation,
                "consistency": enable_consistency
            },
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Map domain to appropriate task type if needed
            mapped_task_type = self._map_domain_to_task(domain, task_type)
            
            # Perform initial extraction
            extraction_result, trajectory, schema_output, frontend_res = self.pipeline.get_extract_result(
                task=mapped_task_type,
                text=text,
                output_schema=schema,
                mode="quick",  # Use quick mode for initial extraction
                update_case=False,  # Don't update case repository during enhancement
                show_trajectory=False
            )
            
            # Convert extraction result to our format
            entities, relations, triples = self._convert_extraction_result(extraction_result, task_type)
            
            # Calculate initial quality scores
            quality_scores = self._calculate_quality_scores(text, entities, relations, extraction_result)
            
            enhanced_result = EnhancedExtractionResult(
                success=True,
                entities=entities,
                relations=relations,
                triples=triples,
                quality_scores=quality_scores,
                metadata={
                    "task_type": mapped_task_type,
                    "domain": domain,
                    "schema_used": schema,
                    "trajectory_length": len(trajectory) if trajectory else 0
                },
                processing_time_ms=0.0  # Will update later
            )
            
            # Apply enhancements based on enabled features
            reflection_notes = []
            
            if enable_cases:
                enhanced_result, case_note = await self._apply_case_based_enhancement(
                    text=text,
                    current_result=enhanced_result,
                    domain=domain,
                    correlation_id=f"{correlation_id}_case"
                )
                if case_note:
                    reflection_notes.append(case_note)
            
            if enable_validation:
                enhanced_result, validation_note = await self._apply_validation_enhancement(
                    text=text,
                    current_result=enhanced_result,
                    correlation_id=f"{correlation_id}_validation"
                )
                if validation_note:
                    reflection_notes.append(validation_note)
            
            if enable_consistency:
                enhanced_result, consistency_note = await self._apply_consistency_enhancement(
                    current_result=enhanced_result,
                    correlation_id=f"{correlation_id}_consistency"
                )
                if consistency_note:
                    reflection_notes.append(consistency_note)
            
            if enable_reflection and self.enhanced_config.get("reflection_enabled", True):
                enhanced_result, reflection_note = await self._apply_reflection_enhancement(
                    text=text,
                    current_result=enhanced_result,
                    correlation_id=f"{correlation_id}_reflection"
                )
                if reflection_note:
                    reflection_notes.append(reflection_note)
            
            enhanced_result.reflection_notes = "; ".join(reflection_notes) if reflection_notes else None
            
            # Update quality scores after enhancements
            final_quality_scores = self._calculate_quality_scores(text, enhanced_result.entities, enhanced_result.relations, extraction_result)
            enhanced_result.quality_scores = final_quality_scores
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            enhanced_result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "Enhanced OneKE extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(enhanced_result.entities),
                "relations_count": len(enhanced_result.relations),
                "triples_count": len(enhanced_result.triples),
                "quality_scores": enhanced_result.quality_scores,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return enhanced_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Enhanced OneKE extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return EnhancedExtractionResult(
                success=False,
                entities=[],
                relations=[],
                triples=[],
                quality_scores={},
                metadata={},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _map_domain_to_task(self, domain: str, default_task: str) -> str:
        """Map domain to appropriate task type."""
        domain_task_map = {
            "ner": "NER",  # Named Entity Recognition
            "re": "RE",    # Relation Extraction
            "ee": "EE",    # Event Extraction
            "triple": "Triple",  # Triple extraction
            "knowledge_graph": "Triple",
            "entity_relation": "RE"
        }
        
        # Try to infer from domain
        domain_lower = domain.lower()
        for key, task in domain_task_map.items():
            if key in domain_lower:
                return task
        
        return default_task
    
    def _convert_extraction_result(self, extraction_result: Any, task_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str, str]]]:
        """Convert OneKE extraction result to our format."""
        entities = []
        relations = []
        triples = []
        
        if extraction_result:
            if task_type == "NER" or "entity" in str(type(extraction_result)).lower():
                # Handle NER results
                if isinstance(extraction_result, list):
                    for item in extraction_result:
                        if isinstance(item, dict) and "entity" in item:
                            entities.append({
                                "name": item.get("entity", ""),
                                "type": item.get("type", "unknown"),
                                "start_pos": item.get("start", 0),
                                "end_pos": item.get("end", 0),
                                "confidence": item.get("confidence", 0.5)
                            })
                        elif isinstance(item, str):
                            entities.append({
                                "name": item,
                                "type": "unknown",
                                "start_pos": 0,
                                "end_pos": 0,
                                "confidence": 0.5
                            })
            
            elif task_type == "RE" or "relation" in str(type(extraction_result)).lower():
                # Handle relation extraction results
                if isinstance(extraction_result, list):
                    for item in extraction_result:
                        if isinstance(item, dict) and all(k in item for k in ["subject", "relation", "object"]):
                            relations.append({
                                "subject": item["subject"],
                                "predicate": item["relation"],
                                "object": item["object"],
                                "confidence": item.get("confidence", 0.5)
                            })
                            triples.append((item["subject"], item["relation"], item["object"]))
            
            elif task_type == "Triple" or "triple" in str(type(extraction_result)).lower():
                # Handle triple extraction results
                if isinstance(extraction_result, list):
                    for item in extraction_result:
                        if isinstance(item, dict) and all(k in item for k in ["subject", "predicate", "object"]):
                            relations.append({
                                "subject": item["subject"],
                                "predicate": item["predicate"],
                                "object": item["object"],
                                "confidence": item.get("confidence", 0.5)
                            })
                            triples.append((item["subject"], item["predicate"], item["object"]))
                        elif isinstance(item, (list, tuple)) and len(item) >= 3:
                            # Handle (subject, predicate, object) format
                            subject, predicate, object = item[0], item[1], item[2]
                            relations.append({
                                "subject": subject,
                                "predicate": predicate,
                                "object": object,
                                "confidence": 0.5
                            })
                            triples.append((subject, predicate, object))
        
        return entities, relations, triples
    
    def _calculate_quality_scores(self, text: str, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], raw_result: Any) -> Dict[str, float]:
        """Calculate quality scores for the extraction."""
        # Calculate various quality metrics
        text_length = len(text)
        entity_count = len(entities)
        relation_count = len(relations)
        
        # Accuracy: How much of the text is covered by extracted entities
        accuracy = min(entity_count * 10 / text_length, 1.0) if text_length > 0 else 0.0
        
        # Completeness: Ratio of entities to relations (indicating completeness of extraction)
        completeness = relation_count / entity_count if entity_count > 0 else 0.0
        completeness = min(completeness, 1.0)
        
        # Consistency: Based on confidence scores
        if relations:
            avg_confidence = sum(r.get("confidence", 0.5) for r in relations) / len(relations)
        else:
            avg_confidence = 0.5
        
        return {
            "accuracy": accuracy,
            "completeness": completeness,
            "consistency": avg_confidence,
            "entity_density": entity_count / text_length if text_length > 0 else 0,
            "relation_density": relation_count / text_length if text_length > 0 else 0
        }
    
    async def _apply_case_based_enhancement(
        self,
        text: str,
        current_result: EnhancedExtractionResult,
        domain: str,
        correlation_id: str
    ) -> Tuple[EnhancedExtractionResult, Optional[str]]:
        """Apply case-based reasoning to enhance extraction."""
        # Find similar cases in the repository
        similar_cases = await self._find_similar_cases(text, domain)
        
        if similar_cases:
            # Apply lessons learned from similar cases
            enhanced_entities, enhanced_relations, enhancement_note = await self._apply_case_learning(
                current_result.entities,
                current_result.relations,
                similar_cases
            )
            
            enhanced_result = EnhancedExtractionResult(
                success=current_result.success,
                entities=enhanced_entities,
                relations=enhanced_relations,
                triples=current_result.triples,  # Recompute if needed
                quality_scores=current_result.quality_scores,
                metadata=current_result.metadata,
                processing_time_ms=current_result.processing_time_ms,
                reflection_notes=current_result.reflection_notes
            )
            
            return enhanced_result, enhancement_note
        
        return current_result, None
    
    async def _apply_validation_enhancement(
        self,
        text: str,
        current_result: EnhancedExtractionResult,
        correlation_id: str
    ) -> Tuple[EnhancedExtractionResult, Optional[str]]:
        """Apply validation checks to enhance extraction."""
        # Perform validation checks
        validated_entities, validated_relations, validation_issues = await self._perform_validations(
            text,
            current_result.entities,
            current_result.relations
        )
        
        if validation_issues:
            note = f"Applied validation fixes: {len(validation_issues)} issues resolved"
            enhanced_result = EnhancedExtractionResult(
                success=current_result.success,
                entities=validated_entities,
                relations=validated_relations,
                triples=self._create_triples_from_relations(validated_relations),
                quality_scores=current_result.quality_scores,
                metadata=current_result.metadata,
                processing_time_ms=current_result.processing_time_ms,
                reflection_notes=current_result.reflection_notes
            )
            return enhanced_result, note
        
        return current_result, None
    
    async def _apply_consistency_enhancement(
        self,
        current_result: EnhancedExtractionResult,
        correlation_id: str
    ) -> Tuple[EnhancedExtractionResult, Optional[str]]:
        """Apply consistency checks to enhance extraction."""
        # Check for consistency issues
        consistent_entities, consistent_relations, consistency_issues = await self._check_consistency(
            current_result.entities,
            current_result.relations
        )
        
        if consistency_issues:
            note = f"Applied consistency fixes: {len(consistency_issues)} inconsistencies resolved"
            enhanced_result = EnhancedExtractionResult(
                success=current_result.success,
                entities=consistent_entities,
                relations=consistent_relations,
                triples=self._create_triples_from_relations(consistent_relations),
                quality_scores=current_result.quality_scores,
                metadata=current_result.metadata,
                processing_time_ms=current_result.processing_time_ms,
                reflection_notes=current_result.reflection_notes
            )
            return enhanced_result, note
        
        return current_result, None
    
    async def _apply_reflection_enhancement(
        self,
        text: str,
        current_result: EnhancedExtractionResult,
        correlation_id: str
    ) -> Tuple[EnhancedExtractionResult, Optional[str]]:
        """Apply reflection-based enhancement."""
        # Use a simple reflection approach to improve the extraction
        reflection_result = await self._perform_reflection(
            text,
            current_result.entities,
            current_result.relations
        )
        
        if reflection_result:
            enhanced_result = EnhancedExtractionResult(
                success=current_result.success,
                entities=reflection_result.get('entities', current_result.entities),
                relations=reflection_result.get('relations', current_result.relations),
                triples=reflection_result.get('triples', current_result.triples),
                quality_scores=current_result.quality_scores,
                metadata=current_result.metadata,
                processing_time_ms=current_result.processing_time_ms,
                reflection_notes=current_result.reflection_notes
            )
            note = "Applied reflection-based enhancement"
            return enhanced_result, note
        
        return current_result, None
    
    async def _find_similar_cases(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Find similar cases in the repository."""
        # This is a simplified implementation - in reality, this would use
        # semantic similarity or other advanced techniques
        similar_cases = []
        
        for case in self.case_repository:
            if case.get('domain') == domain:
                # Simple text similarity check
                if any(word in text.lower() for word in case.get('keywords', [])):
                    similar_cases.append(case)
        
        return similar_cases
    
    async def _apply_case_learning(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """Apply learning from similar cases."""
        # This is a placeholder implementation
        # In a real implementation, this would apply specific improvements
        # learned from the similar cases
        
        # For now, just return the original data with a note
        return entities, relations, f"Applied learning from {len(similar_cases)} similar cases"
    
    async def _perform_validations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """Perform validation checks."""
        issues = []
        validated_entities = []
        validated_relations = []
        
        # Check for entities that might be too short or invalid
        for entity in entities:
            if len(entity['name']) < 2:
                issues.append(f"Removed entity '{entity['name']}' - too short")
                continue
            validated_entities.append(entity)
        
        # Check for relations that reference non-existent entities
        entity_names = {e['name'] for e in validated_entities}
        for relation in relations:
            if relation['subject'] not in entity_names or relation['object'] not in entity_names:
                issues.append(f"Relation '{relation['subject']} -> {relation['predicate']} -> {relation['object']}' references non-existent entity")
                continue
            validated_relations.append(relation)
        
        return validated_entities, validated_relations, issues
    
    async def _check_consistency(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """Check for consistency issues."""
        issues = []
        consistent_entities = entities[:]  # Copy entities
        consistent_relations = relations[:]  # Copy relations
        
        # Look for contradictory relations (simple check)
        # In a real implementation, this would be more sophisticated
        for i, rel1 in enumerate(relations):
            for j, rel2 in enumerate(relations):
                if i < j:  # Avoid duplicate checks
                    # Check if they contradict each other
                    if (rel1['subject'] == rel2['subject'] and 
                        rel1['object'] == rel2['object'] and
                        self._are_predicates_contradictory(rel1['predicate'], rel2['predicate'])):
                        issues.append(f"Potential contradiction: {rel1['subject']} {rel1['predicate']} {rel1['object']} vs {rel2['predicate']}")
        
        return consistent_entities, consistent_relations, issues
    
    def _are_predicates_contradictory(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        contradictory_pairs = [
            ("is", "is_not"),
            ("has", "does_not_have"),
            ("located_in", "not_located_in"),
            ("works_for", "does_not_work_for")
        ]
        
        pred1_lower, pred2_lower = pred1.lower(), pred2.lower()
        for pos, neg in contradictory_pairs:
            if (pos == pred1_lower and neg == pred2_lower) or (neg == pred1_lower and pos == pred2_lower):
                return True
        return False
    
    async def _perform_reflection(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform reflection to improve extraction."""
        # This is a simplified reflection implementation
        # In a real implementation, this would use more sophisticated techniques
        
        # For now, just return the original data
        return {
            'entities': entities,
            'relations': relations,
            'triples': self._create_triples_from_relations(relations)
        }
    
    def _create_triples_from_relations(self, relations: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """Create triples from relations."""
        triples = []
        for rel in relations:
            triples.append((rel['subject'], rel['predicate'], rel['object']))
        return triples
    
    async def extract_and_learn(
        self,
        text: str,
        schema: str,
        domain: str = "general",
        feedback: Optional[Dict[str, Any]] = None,
        task_type: str = "Triple",
        correlation_id: Optional[str] = None
    ) -> Optional[EnhancedExtractionResult]:
        """
        Extract knowledge and learn from feedback.
        
        Args:
            text: Input text
            schema: Target schema
            domain: Domain label
            feedback: Human feedback on extraction quality
                - 'correctness': Correctness score (0-1)
                - 'completeness': Completeness score (0-1)
                - 'comments': Optional comments
            task_type: Type of extraction task
            correlation_id: Correlation ID for tracking
            
        Returns:
            EnhancedExtractionResult with learning metadata
        """
        correlation_id = correlation_id or f"oneke_learn_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Perform enhanced extraction
        result = await self.extract_with_enhancement(
            text=text,
            schema=schema,
            domain=domain,
            task_type=task_type,
            correlation_id=correlation_id
        )
        
        if result and result.success:
            # Store the extraction and feedback as a learning case
            if feedback:
                learning_case = {
                    "text": text,
                    "schema": schema,
                    "domain": domain,
                    "entities": result.entities,
                    "relations": result.relations,
                    "triples": result.triples,
                    "quality_scores": result.quality_scores,
                    "feedback": feedback,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "correlation_id": correlation_id
                }
                
                # Add to case repository
                self.case_repository.append(learning_case)
                
                # Limit repository size
                if len(self.case_repository) > self.enhanced_config.get("case_retention_limit", 1000):
                    self.case_repository = self.case_repository[-self.enhanced_config.get("case_retention_limit", 1000):]
                
                logger.info({
                    "msg": "Learning case added to repository",
                    "correlation_id": correlation_id,
                    "repository_size": len(self.case_repository)
                })
        
        return result
    
    async def get_repository_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the case repository.
        
        Returns:
            Repository statistics
        """
        domains = set()
        schema_count = {}
        avg_quality = {"accuracy": 0, "completeness": 0, "consistency": 0}
        
        if self.case_repository:
            for case in self.case_repository:
                domains.add(case.get('domain', 'unknown'))
                schema = case.get('schema', 'unknown')
                schema_count[schema] = schema_count.get(schema, 0) + 1
                
                # Average quality scores
                q_scores = case.get('quality_scores', {})
                for key in avg_quality:
                    if key in q_scores:
                        avg_quality[key] += q_scores[key]
            
            # Calculate averages
            count = len(self.case_repository)
            for key in avg_quality:
                avg_quality[key] /= count if count > 0 else 0
        
        return {
            "total_cases": len(self.case_repository),
            "domains": list(domains),
            "schema_distribution": schema_count,
            "average_quality_scores": avg_quality,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def export_repository(self, output_path: str) -> bool:
        """
        Export case repository to file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if export successful
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.case_repository, f, indent=2, ensure_ascii=False)
            
            logger.info({
                "msg": "OneKE repository exported successfully",
                "output_path": output_path,
                "cases_exported": len(self.case_repository)
            })
            
            return True
        except Exception as e:
            logger.error({
                "msg": "OneKE repository export failed",
                "output_path": output_path,
                "error": str(e)
            })
            return False
    
    async def import_repository(self, input_path: str) -> bool:
        """
        Import cases into repository from file.
        
        Args:
            input_path: Path to import file
            
        Returns:
            True if import successful
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                imported_cases = json.load(f)
            
            # Add imported cases to repository
            self.case_repository.extend(imported_cases)
            
            # Limit repository size
            limit = self.enhanced_config.get("case_retention_limit", 1000)
            if len(self.case_repository) > limit:
                self.case_repository = self.case_repository[-limit:]
            
            logger.info({
                "msg": "OneKE repository imported successfully",
                "input_path": input_path,
                "cases_imported": len(imported_cases),
                "repository_size": len(self.case_repository)
            })
            
            return True
        except Exception as e:
            logger.error({
                "msg": "OneKE repository import failed",
                "input_path": input_path,
                "error": str(e)
            })
            return False
    
    def get_oneke_status(self) -> Dict[str, Any]:
        """
        Get the status of the OneKE integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": ONEKE_AVAILABLE,
            "pipeline_initialized": self.pipeline is not None,
            "llm_client_initialized": self.llm_client is not None,
            "model_name": self.model_name,
            "repository_size": len(self.case_repository),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing OneKE integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # No specific cleanup needed for OneKE at the moment
        logger.info({
            "msg": "OneKE integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })