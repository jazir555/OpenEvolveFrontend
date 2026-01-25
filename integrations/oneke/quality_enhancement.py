"""
OneKE Quality Enhancement System

This module implements a comprehensive quality enhancement system that
applies multiple strategies to improve knowledge extraction quality.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .case import QualityScore, EnhancedResult, ConsistencyResult
from .reflection_agent import OneKEReflectionAgent
from .case_repository import OneKECaseRepository
from .adapter import OneKEAdapter

logger = logging.getLogger(__name__)


class OneKEQualityEnhancer:
    """
    Enhances extraction quality through multiple strategies.

    Strategies:
    1. Reflection-based improvement
    2. Schema validation
    3. Case-based learning
    4. Consistency checking
    """

    def __init__(
        self,
        reflection_agent: OneKEReflectionAgent,
        validator: Optional[Any] = None,
        min_quality_threshold: float = 0.7
    ):
        """
        Initialize the quality enhancer.

        Args:
            reflection_agent: Reflection agent for improvement
            validator: Optional schema validator
            min_quality_threshold: Minimum quality for auto-acceptance
        """
        self.reflection = reflection_agent
        self.validator = validator
        self.min_quality_threshold = min_quality_threshold

        self.logger = logging.getLogger(f"{__name__}.OneKEQualityEnhancer")

    async def enhance_extraction(
        self,
        raw_extraction: dict,
        text: str,
        schema: str,
        domain: str = "general",
        strategies: Optional[List[str]] = None
    ) -> EnhancedResult:
        """
        Enhance extraction using multiple strategies.

        Args:
            raw_extraction: Initial extraction from OneKE
            text: Original text
            schema: Target schema
            domain: Domain label
            strategies: List of strategies to apply
                - 'reflection': Use reflection agent
                - 'validation': Schema validation
                - 'cases': Case-based retrieval
                - 'consistency': Self-consistency checking

        Returns:
            Enhanced extraction with quality scores
        """
        try:
            self.logger.info("Starting quality enhancement")

            # Default strategies
            if strategies is None:
                strategies = ['reflection', 'validation', 'cases', 'consistency']

            # Score original quality
            original_quality = await self.reflection.score_quality(
                raw_extraction,
                text
            )

            # Start with raw extraction
            enhanced_extraction = raw_extraction.copy()
            applied_strategies = []
            reflection_result = None
            consistency_result = None

            # Apply strategies in sequence
            for strategy in strategies:
                try:
                    if strategy == 'reflection':
                        enhanced_extraction, reflection_result = \
                            await self.apply_reflection_strategy(
                                enhanced_extraction,
                                text,
                                schema,
                                domain
                            )
                        applied_strategies.append('reflection')

                    elif strategy == 'validation':
                        validation_result = await self.apply_validation_strategy(
                            enhanced_extraction,
                            schema
                        )
                        # Apply validation fixes
                        if validation_result.get('needs_fix'):
                            enhanced_extraction = validation_result.get(
                                'fixed_extraction',
                                enhanced_extraction
                            )
                        applied_strategies.append('validation')

                    elif strategy == 'cases':
                        enhanced_extraction = await self.apply_case_strategy(
                            enhanced_extraction,
                            domain,
                            text,
                            schema
                        )
                        applied_strategies.append('cases')

                    elif strategy == 'consistency':
                        enhanced_extraction, consistency_result = \
                            await self.apply_consistency_strategy(
                                enhanced_extraction,
                                text,
                                schema
                            )
                        applied_strategies.append('consistency')

                except Exception as e:
                    self.logger.warning(
                        f"Strategy '{strategy}' failed: {e}. Continuing..."
                    )

            # Score final quality
            final_quality = await self.reflection.score_quality(
                enhanced_extraction,
                text
            )

            # Compute improvement
            quality_improvement = final_quality.overall - original_quality.overall

            # Compute metrics
            metrics = await self.compute_quality_metrics(
                enhanced_extraction,
                text,
                original_quality.overall
            )

            self.logger.info(
                f"Enhancement complete. "
                f"Quality: {original_quality.overall:.2f} -> {final_quality.overall:.2f} "
                f"({quality_improvement:+.2%})"
            )

            return EnhancedResult(
                extraction=enhanced_extraction,
                quality_score=final_quality,
                original_quality=original_quality,
                quality_improvement=quality_improvement,
                strategies_applied=applied_strategies,
                reflection_result=reflection_result,
                consistency_result=consistency_result,
                metadata=metrics
            )

        except Exception as e:
            self.logger.error(f"Quality enhancement failed: {e}")
            # Return original extraction with neutral scores
            neutral_quality = QualityScore(
                completeness=0.5,
                accuracy=0.5,
                consistency=0.5,
                confidence=0.5,
                overall=0.5
            )

            return EnhancedResult(
                extraction=raw_extraction,
                quality_score=neutral_quality,
                original_quality=neutral_quality,
                quality_improvement=0.0,
                strategies_applied=[],
                metadata={}
            )

    async def apply_reflection_strategy(
        self,
        extraction: dict,
        text: str,
        schema: str,
        domain: str
    ) -> tuple[dict, Optional[Any]]:
        """Apply reflection-based improvement."""
        try:
            self.logger.debug("Applying reflection strategy")

            reflection_result = await self.reflection.reflect_on_extraction(
                extracted_data=extraction,
                original_text=text,
                schema=schema,
                domain=domain
            )

            return reflection_result.refined_extraction, reflection_result

        except Exception as e:
            self.logger.error(f"Reflection strategy failed: {e}")
            return extraction, None

    async def apply_validation_strategy(
        self,
        extraction: dict,
        schema: str
    ) -> Dict[str, Any]:
        """Apply schema validation."""
        try:
            self.logger.debug("Applying validation strategy")

            validation_result = {
                'is_valid': True,
                'needs_fix': False,
                'errors': [],
                'warnings': [],
                'fixed_extraction': None
            }

            # Validate entities
            entities = extraction.get('entities', [])
            for i, entity in enumerate(entities):
                if not entity.get('text'):
                    validation_result['errors'].append(
                        f"Entity {i}: missing 'text' field"
                    )
                    validation_result['is_valid'] = False

                if not entity.get('type'):
                    validation_result['warnings'].append(
                        f"Entity {i}: missing 'type' field"
                    )

            # Validate relations
            relations = extraction.get('relations', [])
            for i, relation in enumerate(relations):
                if not relation.get('subject'):
                    validation_result['errors'].append(
                        f"Relation {i}: missing 'subject' field"
                    )
                    validation_result['is_valid'] = False

                if not relation.get('object'):
                    validation_result['errors'].append(
                        f"Relation {i}: missing 'object' field"
                    )
                    validation_result['is_valid'] = False

                if not relation.get('type'):
                    validation_result['warnings'].append(
                        f"Relation {i}: missing 'type' field"
                    )

            # Fix validation errors if possible
            if validation_result['errors']:
                validation_result['needs_fix'] = True
                validation_result['fixed_extraction'] = self._fix_validation_errors(
                    extraction,
                    validation_result['errors']
                )

            return validation_result

        except Exception as e:
            self.logger.error(f"Validation strategy failed: {e}")
            return {
                'is_valid': False,
                'needs_fix': False,
                'errors': [str(e)],
                'warnings': [],
                'fixed_extraction': None
            }

    async def apply_case_strategy(
        self,
        extraction: dict,
        domain: str,
        text: str,
        schema: str
    ) -> dict:
        """Apply case-based learning."""
        try:
            self.logger.debug("Applying case strategy")

            # Retrieve similar cases
            similar_cases = await self.reflection.retrieve_similar_cases(
                extraction=extraction,
                text=text,
                schema=schema,
                domain=domain,
                top_k=5
            )

            if not similar_cases:
                self.logger.debug("No similar cases found")
                return extraction

            # Learn from high-quality cases
            high_quality_cases = [
                c for c in similar_cases
                if c.quality_score >= self.min_quality_threshold
            ]

            if not high_quality_cases:
                self.logger.debug("No high-quality cases found")
                return extraction

            # Apply case learning
            enhanced = extraction.copy()

            # Add missing entities from cases
            for case in high_quality_cases:
                case_entities = case.extracted_data.get('entities', [])

                for case_entity in case_entities:
                    entity_text = case_entity.get('text', '').lower()

                    # Check if entity appears in text
                    if entity_text and entity_text in text.lower():
                        # Check if already in extraction
                        if not any(
                            e.get('text', '').lower() == entity_text
                            for e in enhanced.get('entities', [])
                        ):
                            if 'entities' not in enhanced:
                                enhanced['entities'] = []

                            # Add entity with case-based confidence boost
                            new_entity = case_entity.copy()
                            new_entity['confidence'] = min(
                                new_entity.get('confidence', 0.7) + 0.1,
                                1.0
                            )
                            new_entity['source'] = 'case_learning'
                            enhanced['entities'].append(new_entity)

            # Add missing relations from cases
            for case in high_quality_cases:
                case_relations = case.extracted_data.get('relations', [])

                for case_relation in case_relations:
                    # Check if relation already exists
                    if not any(
                        r.get('subject') == case_relation.get('subject') and
                        r.get('object') == case_relation.get('object') and
                        r.get('type') == case_relation.get('type')
                        for r in enhanced.get('relations', [])
                    ):
                        if 'relations' not in enhanced:
                            enhanced['relations'] = []

                        new_relation = case_relation.copy()
                        new_relation['source'] = 'case_learning'
                        enhanced['relations'].append(new_relation)

            self.logger.debug(
                f"Applied case learning from {len(high_quality_cases)} cases"
            )

            return enhanced

        except Exception as e:
            self.logger.error(f"Case strategy failed: {e}")
            return extraction

    async def apply_consistency_strategy(
        self,
        extraction: dict,
        text: str,
        schema: str
    ) -> tuple[dict, Optional[ConsistencyResult]]:
        """Apply self-consistency checking."""
        try:
            self.logger.debug("Applying consistency strategy")

            consistency_result = await self.reflection.check_self_consistency(
                text=text,
                schema=schema,
                reference_extraction=extraction,
                num_samples=3
            )

            # Use consensus if consistency is high
            if consistency_result.is_consistent:
                self.logger.debug(
                    f"High consistency found, using consensus extraction"
                )
                return consistency_result.consensus_extraction, consistency_result

            # Otherwise, keep original
            return extraction, consistency_result

        except Exception as e:
            self.logger.error(f"Consistency strategy failed: {e}")
            return extraction, None

    async def compute_quality_metrics(
        self,
        extraction: dict,
        text: str,
        original_quality: float
    ) -> Dict[str, Any]:
        """
        Compute quality improvement metrics.

        Returns:
            - completeness: % of required entities present
            - accuracy: % of entities matching schema
            - consistency: Absence of contradictions
            - confidence: Average confidence score
            - improvement: Quality gain from enhancement
        """
        try:
            # Get quality score
            quality_score = await self.reflection.score_quality(
                extraction,
                text
            )

            # Count entities and relations
            entities = extraction.get('entities', [])
            relations = extraction.get('relations', [])

            # Compute average confidence
            if entities:
                confidences = [e.get('confidence', 0.5) for e in entities]
                avg_confidence = sum(confidences) / len(confidences)
            else:
                avg_confidence = 0.0

            # Compute metrics
            metrics = {
                'completeness': quality_score.completeness,
                'accuracy': quality_score.accuracy,
                'consistency': quality_score.consistency,
                'confidence': avg_confidence,
                'improvement': quality_score.overall - original_quality,
                'num_entities': len(entities),
                'num_relations': len(relations),
                'high_confidence_entities': sum(
                    1 for e in entities
                    if e.get('confidence', 0.0) >= 0.8
                ),
                'low_confidence_entities': sum(
                    1 for e in entities
                    if e.get('confidence', 0.0) < 0.5
                )
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to compute quality metrics: {e}")
            return {}

    def _fix_validation_errors(
        self,
        extraction: dict,
        errors: List[str]
    ) -> dict:
        """Fix validation errors automatically."""
        fixed = extraction.copy()

        # Remove entities without text
        if 'entities' in fixed:
            fixed['entities'] = [
                e for e in fixed['entities']
                if e.get('text')
            ]

        # Remove relations without subject or object
        if 'relations' in fixed:
            fixed['relations'] = [
                r for r in fixed['relations']
                if r.get('subject') and r.get('object')
            ]

        return fixed
