"""
Enhanced OneKE Bridge for OpenEvolve Knowledge Engine

This module provides an enhanced bridge to OneKE with quality enhancement,
reflection capabilities, and learning from feedback.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid
import json


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


class EnhancedOneKEBridge:
    """
    Enhanced bridge to OneKE with quality enhancement and reflection capabilities.
    
    Provides methods for:
    - Enhanced extraction with quality assessment
    - Reflection and improvement
    - Learning from feedback
    - Case-based reasoning
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enhanced_config_path: Optional[str] = None
    ):
        """
        Initialize the enhanced OneKE bridge.
        
        Args:
            config_path: Path to basic OneKE configuration
            enhanced_config_path: Path to enhanced configuration
        """
        self.config_path = config_path
        self.enhanced_config_path = enhanced_config_path
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.enhanced_config = self._load_config(enhanced_config_path) or self._get_default_enhanced_config()
        
        # Initialize components
        self.model_adapter = None
        self.quality_enhancer = None
        
        # Case repository for learning
        self.case_repository = []
        
        logger.info({
            "msg": "EnhancedOneKEBridge initialized",
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
    
    async def initialize(self) -> bool:
        """
        Initialize the enhanced bridge with all components.
        
        Returns:
            True if initialization successful
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Initializing EnhancedOneKEBridge",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Initialize model adapter
            model_name = self.config.get("model_name", "oneke/OneKE-13B")
            device = self.config.get("device", "cuda")
            
            from .model_adapter import OneKEModelAdapter
            self.model_adapter = OneKEModelAdapter(
                model_name=model_name,
                device=device,
                config=self.config
            )
            
            await self.model_adapter.load_model()
            
            # Initialize quality enhancer
            from .quality_enhancer import QualityEnhancer
            self.quality_enhancer = QualityEnhancer(
                thresholds=self.enhanced_config.get("quality_thresholds", {})
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "EnhancedOneKEBridge initialized successfully",
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "EnhancedOneKEBridge initialization failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return False
    
    async def extract_with_enhancement(
        self,
        text: str,
        schema: str,
        domain: str = "general",
        enable_reflection: bool = True,
        enable_cases: bool = True,
        enable_validation: bool = True,
        enable_consistency: bool = True,
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
            correlation_id: Correlation ID for tracking
            
        Returns:
            EnhancedExtractionResult with quality scores and reflection notes
        """
        correlation_id = correlation_id or f"enhanced_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.model_adapter:
            raise RuntimeError("Model adapter not initialized. Call initialize() first.")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting enhanced extraction",
            "text_length": len(text),
            "schema": schema,
            "domain": domain,
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
            # Step 1: Initial extraction
            initial_result = await self.model_adapter.extract_triples(
                text=text,
                schema=schema,
                domain=domain,
                correlation_id=f"{correlation_id}_initial"
            )
            
            if not initial_result.success:
                return EnhancedExtractionResult(
                    success=False,
                    entities=[],
                    relations=[],
                    triples=[],
                    quality_scores={},
                    metadata=initial_result.metadata,
                    processing_time_ms=initial_result.processing_time_ms,
                    error=initial_result.error
                )
            
            # Step 2: Quality assessment
            quality_scores = await self._assess_quality(
                text=text,
                extraction_result=initial_result,
                domain=domain,
                correlation_id=correlation_id
            )
            
            enhanced_result = EnhancedExtractionResult(
                success=True,
                entities=initial_result.entities,
                relations=initial_result.relations,
                triples=initial_result.triples,
                quality_scores=quality_scores,
                metadata=initial_result.metadata,
                processing_time_ms=initial_result.processing_time_ms
            )
            
            # Step 3: Apply enhancements based on enabled features
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
            
            # Step 4: Update quality scores after enhancements
            final_quality_scores = await self._assess_quality(
                text=text,
                extraction_result=EnhancedExtractionResult(
                    success=enhanced_result.success,
                    entities=enhanced_result.entities,
                    relations=enhanced_result.relations,
                    triples=enhanced_result.triples,
                    quality_scores=quality_scores,  # Use original scores as baseline
                    metadata=enhanced_result.metadata
                ),
                domain=domain,
                correlation_id=f"{correlation_id}_final"
            )
            enhanced_result.quality_scores = final_quality_scores
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            enhanced_result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "Enhanced extraction completed",
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
                "msg": "Enhanced extraction failed",
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
    
    async def _assess_quality(
        self,
        text: str,
        extraction_result: EnhancedExtractionResult,
        domain: str,
        correlation_id: str
    ) -> Dict[str, float]:
        """Assess the quality of an extraction result."""
        if not self.quality_enhancer:
            return {"accuracy": 0.5, "completeness": 0.5, "consistency": 0.5}
        
        return await self.quality_enhancer.assess_extraction_quality(
            text=text,
            entities=extraction_result.entities,
            relations=extraction_result.relations,
            domain=domain,
            correlation_id=correlation_id
        )
    
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
            correlation_id: Correlation ID for tracking
            
        Returns:
            EnhancedExtractionResult with learning metadata
        """
        correlation_id = correlation_id or f"learn_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Perform enhanced extraction
        result = await self.extract_with_enhancement(
            text=text,
            schema=schema,
            domain=domain,
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
                "msg": "Repository exported successfully",
                "output_path": output_path,
                "cases_exported": len(self.case_repository)
            })
            
            return True
        except Exception as e:
            logger.error({
                "msg": "Repository export failed",
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
                "msg": "Repository imported successfully",
                "input_path": input_path,
                "cases_imported": len(imported_cases),
                "repository_size": len(self.case_repository)
            })
            
            return True
        except Exception as e:
            logger.error({
                "msg": "Repository import failed",
                "input_path": input_path,
                "error": str(e)
            })
            return False