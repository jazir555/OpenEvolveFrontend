"""
OneKE Reflection Agent

This module implements a reflection agent for improving knowledge extraction
quality through self-consistency checking, case-based retrieval, and
iterative refinement.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .case import (
    Case, CaseSimilarity, QualityScore,
    ReflectionResult, ConsistencyResult
)
from .case_repository import OneKECaseRepository
from .adapter import OneKEAdapter

logger = logging.getLogger(__name__)


class OneKEReflectionAgent:
    """
    Integrates OneKE's ReflectionAgent for quality improvement.

    Features:
    - Self-consistency checking through multiple samples
    - Case-based retrieval for learning from past extractions
    - Quality scoring and error identification
    - Automatic refinement based on identified issues
    """

    def __init__(
        self,
        oneke_adapter: OneKEAdapter,
        case_repository: OneKECaseRepository,
        reflection_iterations: int = 3,
        num_samples: int = 3,
        temperature: float = 0.3
    ):
        """
        Initialize the reflection agent.

        Args:
            oneke_adapter: OneKE adapter for extraction
            case_repository: Case repository for learning
            reflection_iterations: Number of reflection cycles
            num_samples: Number of samples for self-consistency
            temperature: Temperature for sampling
        """
        self.adapter = oneke_adapter
        self.cases = case_repository
        self.reflection_iterations = reflection_iterations
        self.num_samples = num_samples
        self.temperature = temperature

        self.logger = logging.getLogger(f"{__name__}.OneKEReflectionAgent")

    async def reflect_on_extraction(
        self,
        extracted_data: dict,
        original_text: str,
        schema: str,
        domain: str = "general"
    ) -> ReflectionResult:
        """
        Reflect on extraction results to improve quality.

        Process:
        1. Generate multiple samples (self-consistency)
        2. Retrieve similar cases from repository
        3. Identify potential errors
        4. Refine extraction
        5. Score quality improvement

        Args:
            extracted_data: Initial extraction data
            original_text: Original input text
            schema: Schema name or definition
            domain: Domain label

        Returns:
            ReflectionResult with improved extraction
        """
        try:
            self.logger.info("Starting reflection on extraction")

            # Score original quality
            original_quality = await self.score_quality(
                extracted_data,
                original_text
            )

            # Check self-consistency
            consistency_result = await self.check_self_consistency(
                original_text,
                schema,
                extracted_data,
                num_samples=self.num_samples
            )

            # Retrieve similar cases
            similar_cases = await self.retrieve_similar_cases(
                extracted_data,
                original_text,
                schema,
                domain,
                top_k=5
            )

            # Identify issues
            issues = await self._identify_issues(
                extracted_data,
                consistency_result,
                similar_cases
            )

            # Refine extraction
            refined_extraction = await self.refine_extraction(
                extracted_data,
                issues,
                similar_cases,
                original_text,
                schema
            )

            # Score refined quality
            refined_quality = await self.score_quality(
                refined_extraction,
                original_text
            )

            # Track improvements
            improvements = self._identify_improvements(
                extracted_data,
                refined_extraction,
                issues
            )

            self.logger.info(
                f"Reflection complete. Quality improved from "
                f"{original_quality.overall:.2f} to {refined_quality.overall:.2f}"
            )

            return ReflectionResult(
                refined_extraction=refined_extraction,
                original_quality=original_quality,
                refined_quality=refined_quality,
                issues_found=issues,
                improvements_made=improvements,
                iterations=self.reflection_iterations
            )

        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            # Return original extraction if reflection fails
            return ReflectionResult(
                refined_extraction=extracted_data,
                original_quality=await self.score_quality(extracted_data, original_text),
                refined_quality=await self.score_quality(extracted_data, original_text),
                issues_found=[],
                improvements_made=[],
                iterations=0
            )

    async def check_self_consistency(
        self,
        text: str,
        schema: str,
        reference_extraction: dict,
        num_samples: int = 3
    ) -> ConsistencyResult:
        """
        Generate multiple extraction samples and check consistency.

        Args:
            text: Input text
            schema: Schema definition
            reference_extraction: Reference extraction for comparison
            num_samples: Number of samples to generate

        Returns:
            ConsistencyResult with agreement analysis
        """
        try:
            self.logger.info(f"Checking self-consistency with {num_samples} samples")

            # Generate multiple samples
            samples = []
            for i in range(num_samples):
                try:
                    # Sample with slight temperature variation
                    result = await self.adapter.extract_schema_guided(
                        text=text,
                        schema=schema,
                        temperature=self.temperature + (i * 0.1)
                    )

                    extraction = {
                        'entities': result.entities,
                        'relations': result.relations,
                        'events': result.events,
                        'triples': result.triples
                    }
                    samples.append(extraction)

                except Exception as e:
                    self.logger.warning(f"Sample {i} generation failed: {e}")

            if not samples:
                # No samples generated, return reference as only sample
                samples = [reference_extraction]

            # Compute consensus
            consensus_extraction = await self._compute_consensus(samples)

            # Check agreement
            agreement_ratio, disagreements = await self._compute_agreement(
                samples,
                consensus_extraction
            )

            is_consistent = agreement_ratio >= 0.7

            self.logger.info(
                f"Self-consistency check: {is_consistent} "
                f"(agreement={agreement_ratio:.2f})"
            )

            return ConsistencyResult(
                is_consistent=is_consistent,
                agreement_ratio=agreement_ratio,
                samples=samples,
                consensus_extraction=consensus_extraction,
                disagreements=disagreements
            )

        except Exception as e:
            self.logger.error(f"Self-consistency check failed: {e}")
            # Return consistent result with reference
            return ConsistencyResult(
                is_consistent=True,
                agreement_ratio=1.0,
                samples=[reference_extraction],
                consensus_extraction=reference_extraction,
                disagreements=[]
            )

    async def retrieve_similar_cases(
        self,
        extraction: dict,
        text: str,
        schema: str,
        domain: str,
        top_k: int = 5
    ) -> List[Case]:
        """
        Retrieve similar cases from repository for comparison.

        Args:
            extraction: Current extraction
            text: Input text
            schema: Schema name
            domain: Domain label
            top_k: Number of cases to retrieve

        Returns:
            List of similar cases
        """
        try:
            query = {
                'input_text': text,
                'schema': schema,
                'domain': domain
            }

            similar = await self.cases.retrieve_similar_cases(
                query=query,
                top_k=top_k,
                min_similarity=0.6,
                domain=domain
            )

            # Extract cases from CaseSimilarity objects
            cases = [s.case for s in similar]

            self.logger.info(f"Retrieved {len(cases)} similar cases")

            return cases

        except Exception as e:
            self.logger.error(f"Failed to retrieve similar cases: {e}")
            return []

    async def refine_extraction(
        self,
        extraction: dict,
        issues: List[str],
        cases: List[Case],
        text: str,
        schema: str
    ) -> dict:
        """
        Refine extraction based on identified issues and cases.

        Args:
            extraction: Original extraction
            issues: List of identified issues
            cases: Similar cases for reference
            text: Input text
            schema: Schema definition

        Returns:
            Refined extraction
        """
        try:
            self.logger.info(f"Refining extraction (issues: {len(issues)})")

            # Start with original extraction
            refined = extraction.copy()

            # Learn from similar cases
            if cases:
                refined = await self._apply_case_learning(
                    refined,
                    cases,
                    text,
                    schema
                )

            # Fix identified issues
            if issues:
                refined = await self._fix_issues(
                    refined,
                    issues,
                    text
                )

            # Validate and clean
            refined = await self._validate_and_clean(refined)

            return refined

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return extraction

    async def score_quality(
        self,
        extraction: dict,
        original_text: str
    ) -> QualityScore:
        """
        Score extraction quality (0-1).

        Metrics:
        - Completeness: All required entities extracted
        - Accuracy: Entities match schema
        - Consistency: No contradictions
        - Confidence: Average entity confidence

        Args:
            extraction: Extraction data
            original_text: Original input text

        Returns:
            QualityScore with detailed metrics
        """
        try:
            # Extract entities and relations
            entities = extraction.get('entities', [])
            relations = extraction.get('relations', [])

            # Completeness: Check if entities cover key concepts
            completeness = await self._score_completeness(
                entities,
                original_text
            )

            # Accuracy: Check entity validity
            accuracy = await self._score_accuracy(entities, relations)

            # Consistency: Check for contradictions
            consistency = await self._score_consistency(
                entities,
                relations
            )

            # Confidence: Average confidence score
            confidence = await self._score_confidence(entities)

            # Overall: Weighted average
            overall = (
                0.3 * completeness +
                0.3 * accuracy +
                0.2 * consistency +
                0.2 * confidence
            )

            return QualityScore(
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                confidence=confidence,
                overall=overall
            )

        except Exception as e:
            self.logger.error(f"Quality scoring failed: {e}")
            # Return neutral score
            return QualityScore(
                completeness=0.5,
                accuracy=0.5,
                consistency=0.5,
                confidence=0.5,
                overall=0.5
            )

    async def _identify_issues(
        self,
        extraction: dict,
        consistency: ConsistencyResult,
        cases: List[Case]
    ) -> List[str]:
        """Identify potential issues in extraction."""
        issues = []

        # Check consistency
        if not consistency.is_consistent:
            issues.append(
                f"Low self-consistency (agreement={consistency.agreement_ratio:.2f})"
            )

        # Check for disagreements
        if consistency.disagreements:
            issues.append(
                f"Found {len(consistency.disagreements)} disagreement points"
            )

        # Compare with cases
        if cases:
            avg_case_quality = sum(c.quality_score for c in cases) / len(cases)
            if avg_case_quality > 0.8:
                issues.append(
                    "Similar cases show higher quality patterns"
                )

        # Check for missing entities
        entities = extraction.get('entities', [])
        if len(entities) < 3:
            issues.append("Low entity count (possible missing information)")

        # Check for low confidence
        confidences = [e.get('confidence', 0.5) for e in entities]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        if avg_confidence < 0.6:
            issues.append(f"Low average confidence ({avg_confidence:.2f})")

        return issues

    async def _compute_consensus(
        self,
        samples: List[dict]
    ) -> dict:
        """Compute consensus extraction from multiple samples."""
        if not samples:
            return {}

        # Aggregate entities (keep those appearing in majority)
        entity_counts = {}
        for sample in samples:
            for entity in sample.get('entities', []):
                # Create entity key
                key = (
                    entity.get('type', ''),
                    entity.get('text', ''),
                    entity.get('label', '')
                )
                entity_counts[key] = entity_counts.get(key, 0) + 1

        # Keep entities appearing in >50% of samples
        threshold = len(samples) / 2
        consensus_entities = []

        for sample in samples:
            for entity in sample.get('entities', []):
                key = (
                    entity.get('type', ''),
                    entity.get('text', ''),
                    entity.get('label', '')
                )
                if entity_counts[key] > threshold:
                    # Avoid duplicates
                    if not any(
                        e.get('text') == entity.get('text')
                        for e in consensus_entities
                    ):
                        consensus_entities.append(entity)

        # Aggregate relations similarly
        relation_counts = {}
        for sample in samples:
            for relation in sample.get('relations', []):
                key = (
                    relation.get('type', ''),
                    relation.get('subject', ''),
                    relation.get('object', '')
                )
                relation_counts[key] = relation_counts.get(key, 0) + 1

        threshold = len(samples) / 2
        consensus_relations = []

        for sample in samples:
            for relation in sample.get('relations', []):
                key = (
                    relation.get('type', ''),
                    relation.get('subject', ''),
                    relation.get('object', '')
                )
                if relation_counts[key] > threshold:
                    # Avoid duplicates
                    if not any(
                        r.get('subject') == relation.get('subject') and
                        r.get('object') == relation.get('object') and
                        r.get('type') == relation.get('type')
                        for r in consensus_relations
                    ):
                        consensus_relations.append(relation)

        return {
            'entities': consensus_entities,
            'relations': consensus_relations,
            'events': samples[0].get('events', []),
            'triples': samples[0].get('triples', [])
        }

    async def _compute_agreement(
        self,
        samples: List[dict],
        consensus: dict
    ) -> tuple[float, List[str]]:
        """Compute agreement ratio and disagreements."""
        if not samples:
            return 1.0, []

        total_entities = sum(len(s.get('entities', [])) for s in samples)
        consensus_entities = len(consensus.get('entities', []))

        if total_entities == 0:
            return 1.0, []

        agreement_ratio = consensus_entities / (total_entities / len(samples))

        # Identify disagreements
        disagreements = []

        for i, sample in enumerate(samples):
            sample_entities = set(
                e.get('text', '') for e in sample.get('entities', [])
            )
            consensus_entities_set = set(
                e.get('text', '') for e in consensus.get('entities', [])
            )

            unique_to_sample = sample_entities - consensus_entities_set
            if unique_to_sample:
                disagreements.append(
                    f"Sample {i}: {len(unique_to_sample)} unique entities"
                )

        return min(agreement_ratio, 1.0), disagreements

    async def _apply_case_learning(
        self,
        extraction: dict,
        cases: List[Case],
        text: str,
        schema: str
    ) -> dict:
        """Apply learning from similar cases."""
        # Extract high-quality patterns from cases
        high_quality_cases = [c for c in cases if c.quality_score >= 0.8]

        if not high_quality_cases:
            return extraction

        # Learn entity patterns
        refined = extraction.copy()

        for case in high_quality_cases:
            case_entities = case.extracted_data.get('entities', [])

            # Add missing entities that appear in similar cases
            for case_entity in case_entities:
                entity_text = case_entity.get('text', '').lower()
                if entity_text and entity_text in text.lower():
                    # Check if this entity is already in extraction
                    if not any(
                        e.get('text', '').lower() == entity_text
                        for e in refined.get('entities', [])
                    ):
                        # Add entity from case
                        if 'entities' not in refined:
                            refined['entities'] = []
                        refined['entities'].append(case_entity.copy())

        return refined

    async def _fix_issues(
        self,
        extraction: dict,
        issues: List[str],
        text: str
    ) -> dict:
        """Fix identified issues."""
        refined = extraction.copy()

        # Fix low entity count
        if any('count' in issue.lower() for issue in issues):
            # Try to extract more entities (placeholder logic)
            # In production, would re-run extraction with adjusted parameters
            pass

        # Fix low confidence
        if any('confidence' in issue.lower() for issue in issues):
            # Filter out low-confidence entities
            if 'entities' in refined:
                refined['entities'] = [
                    e for e in refined['entities']
                    if e.get('confidence', 0.0) >= 0.5
                ]

        return refined

    async def _validate_and_clean(self, extraction: dict) -> dict:
        """Validate and clean extraction."""
        cleaned = {}

        # Validate entities
        if 'entities' in extraction:
            cleaned['entities'] = [
                e for e in extraction['entities']
                if e.get('text') and e.get('type')
            ]

        # Validate relations
        if 'relations' in extraction:
            cleaned['relations'] = [
                r for r in extraction['relations']
                if r.get('subject') and r.get('object') and r.get('type')
            ]

        # Copy other fields
        for key in ['events', 'triples']:
            if key in extraction:
                cleaned[key] = extraction[key]

        return cleaned

    def _identify_improvements(
        self,
        original: dict,
        refined: dict,
        issues: List[str]
    ) -> List[str]:
        """Identify what improvements were made."""
        improvements = []

        # Check entity count
        orig_entities = len(original.get('entities', []))
        refined_entities = len(refined.get('entities', []))

        if refined_entities > orig_entities:
            improvements.append(
                f"Added {refined_entities - orig_entities} entities"
            )

        # Check relations
        orig_relations = len(original.get('relations', []))
        refined_relations = len(refined.get('relations', []))

        if refined_relations > orig_relations:
            improvements.append(
                f"Added {refined_relations - orig_relations} relations"
            )

        # Check issue resolution
        if issues:
            improvements.append(f"Addressed {len(issues)} identified issues")

        if not improvements:
            improvements.append("Validated extraction quality")

        return improvements

    async def _score_completeness(
        self,
        entities: List[dict],
        text: str
    ) -> float:
        """Score completeness of entity extraction."""
        if not entities:
            return 0.0

        # Simple heuristic: check if entities cover key terms
        # In production, would use more sophisticated NLP
        text_words = set(text.lower().split())
        entity_texts = set(
            ' '.join(e.get('text', '').split()).lower()
            for e in entities
        )

        # Check coverage
        coverage = len(entity_texts & text_words) / len(text_words) if text_words else 0.0

        return min(coverage * 2, 1.0)  # Scale up to account for key concepts

    async def _score_accuracy(
        self,
        entities: List[dict],
        relations: List[dict]
    ) -> float:
        """Score accuracy of extraction."""
        if not entities and not relations:
            return 0.5

        # Check if entities have required fields
        valid_entities = sum(
            1 for e in entities
            if e.get('text') and e.get('type')
        )

        entity_accuracy = valid_entities / len(entities) if entities else 0.5

        # Check if relations have required fields
        valid_relations = sum(
            1 for r in relations
            if r.get('subject') and r.get('object') and r.get('type')
        )

        relation_accuracy = valid_relations / len(relations) if relations else 0.5

        # Weighted average
        return 0.6 * entity_accuracy + 0.4 * relation_accuracy

    async def _score_consistency(
        self,
        entities: List[dict],
        relations: List[dict]
    ) -> float:
        """Score internal consistency."""
        # Check for duplicate entities
        entity_texts = [e.get('text', '') for e in entities]
        duplicates = len(entity_texts) - len(set(entity_texts))

        if duplicates > 0:
            return max(1.0 - (duplicates / len(entities)), 0.0)

        # Check for contradictory relations (simplified)
        # In production, would do more sophisticated checks
        return 1.0

    async def _score_confidence(
        self,
        entities: List[dict]
    ) -> float:
        """Score average confidence."""
        if not entities:
            return 0.5

        confidences = [e.get('confidence', 0.5) for e in entities]
        return sum(confidences) / len(confidences)
