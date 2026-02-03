"""
Quality Enhancer for OneKE Integration

This module provides functionality for assessing and enhancing the quality
of knowledge extraction results from OneKE.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class QualityAssessment:
    """Assessment of extraction quality."""
    accuracy: float
    completeness: float
    consistency: float
    relevance: float
    overall_score: float
    detailed_feedback: Dict[str, Any]


class QualityEnhancer:
    """
    Quality enhancer for OneKE extraction results.
    
    Provides methods for:
    - Assessing extraction quality
    - Providing detailed feedback
    - Suggesting improvements
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the quality enhancer.
        
        Args:
            thresholds: Quality thresholds for different metrics
        """
        self.thresholds = thresholds or {
            "accuracy": 0.7,
            "completeness": 0.6,
            "consistency": 0.8,
            "relevance": 0.7
        }
        
        logger.info({
            "msg": "QualityEnhancer initialized",
            "thresholds": self.thresholds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def assess_extraction_quality(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        domain: str = "general",
        correlation_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Assess the quality of an extraction result.
        
        Args:
            text: Original text that was extracted from
            entities: Extracted entities
            relations: Extracted relations
            domain: Domain of the text
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with quality scores
        """
        correlation_id = correlation_id or f"quality_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting quality assessment",
            "text_length": len(text),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "domain": domain,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Calculate individual quality metrics
            accuracy_score = await self._assess_accuracy(text, entities, relations)
            completeness_score = await self._assess_completeness(text, entities, relations)
            consistency_score = await self._assess_consistency(entities, relations)
            relevance_score = await self._assess_relevance(text, entities, relations, domain)
            
            # Calculate overall score (weighted average)
            weights = {
                "accuracy": 0.3,
                "completeness": 0.25,
                "consistency": 0.25,
                "relevance": 0.2
            }
            
            overall_score = (
                accuracy_score * weights["accuracy"] +
                completeness_score * weights["completeness"] +
                consistency_score * weights["consistency"] +
                relevance_score * weights["relevance"]
            )
            
            quality_scores = {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "consistency": consistency_score,
                "relevance": relevance_score,
                "overall": overall_score
            }
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Quality assessment completed",
                "correlation_id": correlation_id,
                "quality_scores": quality_scores,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return quality_scores
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Quality assessment failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return default low scores in case of error
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "consistency": 0.0,
                "relevance": 0.0,
                "overall": 0.0
            }
    
    async def _assess_accuracy(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> float:
        """
        Assess the accuracy of extracted entities and relations.
        
        Args:
            text: Original text
            entities: Extracted entities
            relations: Extracted relations
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not entities and not relations:
            return 0.0
        
        # Check if extracted entities actually appear in the text
        correct_entities = 0
        total_entities = len(entities)
        
        for entity in entities:
            entity_name = entity.get('name', '').lower()
            if entity_name in text.lower():
                correct_entities += 1
        
        # Check if extracted relations are plausible based on text
        plausible_relations = 0
        total_relations = len(relations)
        
        for relation in relations:
            subj = relation.get('subject', '').lower()
            obj = relation.get('object', '').lower()
            
            # Check if both subject and object appear in text
            if subj in text.lower() and obj in text.lower():
                plausible_relations += 1
        
        # Calculate accuracy as weighted combination
        entity_accuracy = correct_entities / total_entities if total_entities > 0 else 1.0
        relation_accuracy = plausible_relations / total_relations if total_relations > 0 else 1.0
        
        # Weight entities slightly more than relations
        accuracy = (entity_accuracy * 0.6) + (relation_accuracy * 0.4)
        
        return accuracy
    
    async def _assess_completeness(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> float:
        """
        Assess the completeness of the extraction.
        
        Args:
            text: Original text
            entities: Extracted entities
            relations: Extracted relations
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not text.strip():
            return 1.0 if not entities and not relations else 0.0
        
        # Estimate how much of the text content is captured by entities
        # This is a simplified approach - in reality, you'd use more sophisticated NLP
        text_words = set(text.lower().split())
        
        # Collect entity words
        entity_words = set()
        for entity in entities:
            entity_name = entity.get('name', '')
            entity_words.update(entity_name.lower().split())
        
        # Calculate coverage
        if text_words:
            coverage = len(entity_words.intersection(text_words)) / len(text_words)
        else:
            coverage = 1.0 if not entities else 0.0
        
        # Normalize based on expected density
        # For typical text, we might expect 10-20% of words to be named entities
        expected_coverage = 0.15
        completeness = min(coverage / expected_coverage, 1.0)
        
        return completeness
    
    async def _assess_consistency(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> float:
        """
        Assess the consistency of extracted entities and relations.
        
        Args:
            entities: Extracted entities
            relations: Extracted relations
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not entities and not relations:
            return 1.0
        
        # Check for duplicate entities with different types
        entity_names = {}
        for entity in entities:
            name = entity.get('name', '').lower()
            if name in entity_names:
                # If types are different, it's inconsistent
                if entity.get('type') != entity_names[name]:
                    entity_names[name] = "INCONSISTENT"
                else:
                    entity_names[name] = entity.get('type')
            else:
                entity_names[name] = entity.get('type')
        
        inconsistent_entities = sum(1 for v in entity_names.values() if v == "INCONSISTENT")
        total_entities = len(entities)
        
        # Check for contradictory relations
        relation_pairs = {}
        contradictory_relations = 0
        
        for relation in relations:
            subj = relation.get('subject', '').lower()
            obj = relation.get('object', '').lower()
            pred = relation.get('predicate', '').lower()
            
            # Check for contradictory predicates
            key = (subj, obj)
            if key in relation_pairs:
                existing_pred = relation_pairs[key]
                if self._are_predicates_contradictory(pred, existing_pred):
                    contradictory_relations += 1
            else:
                relation_pairs[key] = pred
        
        # Calculate consistency score
        entity_consistency = 1.0 - (inconsistent_entities / total_entities if total_entities > 0 else 0)
        relation_consistency = 1.0 - (contradictory_relations / len(relations) if relations else 0)
        
        # Weight both equally
        consistency = (entity_consistency + relation_consistency) / 2.0
        
        return consistency
    
    async def _assess_relevance(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        domain: str
    ) -> float:
        """
        Assess the relevance of extracted knowledge to the domain.
        
        Args:
            text: Original text
            entities: Extracted entities
            relations: Extracted relations
            domain: Domain of the text
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not entities and not relations:
            return 1.0
        
        # Define domain-specific keywords
        domain_keywords = self._get_domain_keywords(domain)
        
        # Count how many extracted entities relate to domain keywords
        relevant_entities = 0
        total_entities = len(entities)
        
        for entity in entities:
            entity_name = entity.get('name', '').lower()
            entity_type = entity.get('type', '').lower()
            
            # Check if entity name or type relates to domain
            if any(keyword in entity_name or keyword in entity_type for keyword in domain_keywords):
                relevant_entities += 1
            # Also check if entity appears near domain keywords in text
            elif self._entity_near_domain_keyword(entity_name, text, domain_keywords):
                relevant_entities += 1
        
        relevance = relevant_entities / total_entities if total_entities > 0 else 1.0
        
        return relevance
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords associated with a domain."""
        domain_keywords = {
            "general": ["the", "and", "to", "of", "in", "that", "have", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their"],
            "science": ["experiment", "research", "study", "data", "result", "method", "theory", "hypothesis", "analysis", "conclusion", "evidence", "observation", "measurement", "variable", "control", "sample", "significant", "correlation", "causation", "peer", "review"],
            "technology": ["software", "algorithm", "system", "application", "program", "computer", "network", "data", "code", "server", "client", "api", "database", "framework", "library", "platform", "interface", "protocol", "architecture", "development"],
            "business": ["company", "market", "customer", "product", "service", "revenue", "profit", "loss", "investment", "strategy", "management", "employee", "sales", "marketing", "finance", "operation", "competition", "industry", "trend", "growth"],
            "healthcare": ["patient", "doctor", "hospital", "treatment", "medicine", "disease", "symptom", "diagnosis", "therapy", "medical", "clinical", "trial", "research", "study", "health", "care", "nurse", "pharmacy", "drug", "procedure"],
            "education": ["student", "teacher", "school", "university", "course", "class", "lesson", "education", "learning", "teaching", "academic", "degree", "grade", "exam", "test", "curriculum", "instruction", "knowledge", "skill", "ability"]
        }
        
        return domain_keywords.get(domain.lower(), domain_keywords["general"])
    
    def _entity_near_domain_keyword(self, entity: str, text: str, domain_keywords: List[str]) -> bool:
        """Check if an entity appears near domain keywords in text."""
        text_lower = text.lower()
        entity_pos = text_lower.find(entity.lower())
        
        if entity_pos == -1:
            return False
        
        # Look at a window around the entity
        window_size = 100  # characters
        start = max(0, entity_pos - window_size)
        end = min(len(text_lower), entity_pos + len(entity) + window_size)
        context = text_lower[start:end]
        
        # Check if any domain keywords appear in the context
        return any(keyword in context for keyword in domain_keywords)
    
    def _are_predicates_contradictory(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        contradictory_pairs = [
            ("is", "is_not"),
            ("has", "does_not_have"),
            ("located_in", "not_located_in"),
            ("works_for", "does_not_work_for"),
            ("includes", "excludes"),
            ("contains", "does_not_contain"),
            ("supports", "opposes"),
            ("agrees", "disagrees")
        ]
        
        pred1_lower, pred2_lower = pred1.lower(), pred2.lower()
        for pos, neg in contradictory_pairs:
            if (pos == pred1_lower and neg == pred2_lower) or (neg == pred1_lower and pos == pred2_lower):
                return True
        return False
    
    async def suggest_improvements(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        domain: str = "general",
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest improvements for the extraction based on quality assessment.
        
        Args:
            text: Original text
            entities: Extracted entities
            relations: Extracted relations
            domain: Domain of the text
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with improvement suggestions
        """
        correlation_id = correlation_id or f"improve_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting improvement suggestions",
            "text_length": len(text),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "domain": domain,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Get quality assessment
            quality_scores = await self.assess_extraction_quality(
                text=text,
                entities=entities,
                relations=relations,
                domain=domain,
                correlation_id=correlation_id
            )
            
            suggestions = {
                "accuracy_improvements": [],
                "completeness_improvements": [],
                "consistency_improvements": [],
                "relevance_improvements": [],
                "overall_quality_score": quality_scores.get("overall", 0.0)
            }
            
            # Suggest accuracy improvements
            if quality_scores.get("accuracy", 0) < self.thresholds.get("accuracy", 0.7):
                suggestions["accuracy_improvements"] = await self._suggest_accuracy_improvements(
                    text, entities, relations
                )
            
            # Suggest completeness improvements
            if quality_scores.get("completeness", 0) < self.thresholds.get("completeness", 0.6):
                suggestions["completeness_improvements"] = await self._suggest_completeness_improvements(
                    text, entities, relations
                )
            
            # Suggest consistency improvements
            if quality_scores.get("consistency", 0) < self.thresholds.get("consistency", 0.8):
                suggestions["consistency_improvements"] = await self._suggest_consistency_improvements(
                    entities, relations
                )
            
            # Suggest relevance improvements
            if quality_scores.get("relevance", 0) < self.thresholds.get("relevance", 0.7):
                suggestions["relevance_improvements"] = await self._suggest_relevance_improvements(
                    text, entities, relations, domain
                )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Improvement suggestions completed",
                "correlation_id": correlation_id,
                "suggestion_categories": [k for k, v in suggestions.items() if k.endswith('_improvements') and v],
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return suggestions
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Improvement suggestions failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "accuracy_improvements": [],
                "completeness_improvements": [],
                "consistency_improvements": [],
                "relevance_improvements": [],
                "overall_quality_score": 0.0,
                "error": str(e)
            }
    
    async def _suggest_accuracy_improvements(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest improvements for accuracy."""
        suggestions = []
        
        # Identify entities not found in text
        missing_entities = []
        for entity in entities:
            entity_name = entity.get('name', '')
            if entity_name.lower() not in text.lower():
                missing_entities.append(entity_name)
        
        if missing_entities:
            suggestions.append(f"Remove entities not found in text: {', '.join(missing_entities[:5])}")  # Limit to first 5
        
        # Identify implausible relations
        implausible_relations = []
        for relation in relations:
            subj = relation.get('subject', '')
            obj = relation.get('object', '')
            if subj.lower() not in text.lower() or obj.lower() not in text.lower():
                implausible_relations.append(f"{subj} -> {relation.get('predicate')} -> {obj}")
        
        if implausible_relations:
            suggestions.append(f"Review relations with entities not in text: {', '.join(implausible_relations[:3])}")
        
        return suggestions
    
    async def _suggest_completeness_improvements(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest improvements for completeness."""
        suggestions = []
        
        # Simple suggestion: consider using more aggressive extraction parameters
        suggestions.append("Consider adjusting extraction parameters for higher recall")
        suggestions.append("Review text for additional named entities that may have been missed")
        
        return suggestions
    
    async def _suggest_consistency_improvements(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest improvements for consistency."""
        suggestions = []
        
        # Identify inconsistent entities
        entity_types = {}
        for entity in entities:
            name = entity.get('name', '').lower()
            entity_type = entity.get('type', '')
            if name in entity_types:
                if entity_types[name] != entity_type:
                    suggestions.append(f"Entity '{name}' has inconsistent types: {entity_types[name]} vs {entity_type}")
            else:
                entity_types[name] = entity_type
        
        # Identify contradictory relations
        relation_pairs = {}
        for relation in relations:
            subj = relation.get('subject', '').lower()
            obj = relation.get('object', '').lower()
            pred = relation.get('predicate', '').lower()
            
            key = (subj, obj)
            if key in relation_pairs:
                existing_pred = relation_pairs[key]
                if self._are_predicates_contradictory(pred, existing_pred):
                    suggestions.append(f"Contradictory relations: {subj} {pred} {obj} vs {subj} {existing_pred} {obj}")
            else:
                relation_pairs[key] = pred
        
        return suggestions
    
    async def _suggest_relevance_improvements(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        domain: str
    ) -> List[str]:
        """Suggest improvements for relevance."""
        suggestions = []
        
        domain_keywords = self._get_domain_keywords(domain)
        
        # Identify entities with low domain relevance
        low_relevance_entities = []
        for entity in entities:
            entity_name = entity.get('name', '').lower()
            entity_type = entity.get('type', '').lower()
            
            # Check if entity relates to domain
            is_relevant = any(keyword in entity_name or keyword in entity_type for keyword in domain_keywords)
            if not is_relevant:
                low_relevance_entities.append(entity.get('name'))
        
        if low_relevance_entities:
            suggestions.append(f"Consider filtering low-domain-relevance entities: {', '.join(low_relevance_entities[:5])}")
        
        return suggestions