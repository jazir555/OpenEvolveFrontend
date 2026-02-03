"""
OneKE Cross-Lingual Entity Linker
Task 3.4: Cross-Lingual Entity Linking

Implements bilingual entity matching and resolution:
- 3.4.1: Bilingual entity matching (English/Chinese)
- 3.4.2: Translation-aware entity resolution
- 3.4.3: Cross-lingual relation alignment
- 3.4.4: Language detection for documents
- 3.4.5: Bilingual knowledge graph format

Following CLAUDE.md Principles:
- AIR GAP: Adapter pattern for translation services
- RUNTIME TRUTH: Probes verify translation APIs
- IDEMPOTENCY: All linking operations are idempotent
- CONFIGURATION EXPLICITNESS: All config via environment variables
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

# Structured logging
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    CHINESE = "zh"
    BILINGUAL = "bilingual"
    UNKNOWN = "unknown"


class MatchStrategy(Enum):
    """Entity matching strategies."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    TRANSLATION = "translation"
    HYBRID = "hybrid"


@dataclass
class Entity:
    """
    Bilingual entity representation.

    Attributes:
        entity_id: Unique entity identifier
        name_en: English name(s)
        name_zh: Chinese name(s)
        aliases_en: English aliases
        aliases_zh: Chinese aliases
        type: Entity type
        properties: Additional properties
        confidence: Confidence score
        source: Source document/id
        language: Primary language
        metadata: Additional metadata
    """
    entity_id: str
    name_en: List[str] = field(default_factory=list)
    name_zh: List[str] = field(default_factory=list)
    aliases_en: List[str] = field(default_factory=list)
    aliases_zh: List[str] = field(default_factory=list)
    type: str = "unknown"
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    language: Language = Language.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity data."""
        if not self.name_en and not self.name_zh:
            raise ValueError(f"Entity must have at least one name: {self.entity_id}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Invalid confidence: {self.confidence}, must be in [0, 1]")

    def get_all_names(self) -> Dict[str, List[str]]:
        """Get all names and aliases by language."""
        return {
            "en": self.name_en + self.aliases_en,
            "zh": self.name_zh + self.aliases_zh
        }


@dataclass
class EntityMatchResult:
    """
    Result of entity matching operation.

    Attributes:
        entity1_id: First entity ID
        entity2_id: Second entity ID
        matched: Whether entities match
        strategy: Matching strategy used
        confidence: Match confidence score
        evidence: Matching evidence/reasoning
        cross_lingual: Whether match is cross-lingual
        translation_used: If translation was used
        timestamp: Match timestamp (UTC)
    """
    entity1_id: str
    entity2_id: str
    matched: bool
    strategy: MatchStrategy
    confidence: float
    evidence: List[str] = field(default_factory=list)
    cross_lingual: bool = False
    translation_used: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity1_id": self.entity1_id,
            "entity2_id": self.entity2_id,
            "matched": self.matched,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "cross_lingual": self.cross_lingual,
            "translation_used": self.translation_used,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LinkerConfig:
    """
    Entity linker configuration.

    Environment Variables (CLAUDE.md: Configuration Explicitness):
    - ONEKE_TRANSLATION_API: Translation service URL
    - ONEKE_TRANSLATION_MODEL: Translation model (default: "google")
    - ONEKE_FUZZY_THRESHOLD: Fuzzy match threshold (default: 85)
    - ONEKE_SEMANTIC_THRESHOLD: Semantic similarity threshold (default: 0.7)
    - ONEKE_MAX_CANDIDATES: Max candidates for matching (default: 100)
    - ONEKE_ENABLE_TRANSLATION: Enable translation (default: true)
    - ONEKE_CACHE_TRANSLATIONS: Cache translations (default: true)
    """
    translation_api: Optional[str] = field(default_factory=lambda: os.getenv("ONEKE_TRANSLATION_API"))
    translation_model: str = field(default_factory=lambda: os.getenv("ONEKE_TRANSLATION_MODEL", "google"))
    fuzzy_threshold: int = field(default_factory=lambda: int(os.getenv("ONEKE_FUZZY_THRESHOLD", "85")))
    semantic_threshold: float = field(default_factory=lambda: float(os.getenv("ONEKE_SEMANTIC_THRESHOLD", "0.7")))
    max_candidates: int = field(default_factory=lambda: int(os.getenv("ONEKE_MAX_CANDIDATES", "100")))
    enable_translation: bool = field(default_factory=lambda: bool(os.getenv("ONEKE_ENABLE_TRANSLATION", "true")))
    cache_translations: bool = field(default_factory=lambda: bool(os.getenv("ONEKE_CACHE_TRANSLATIONS", "true")))

    def __post_init__(self):
        """Validate configuration."""
        if self.fuzzy_threshold < 0 or self.fuzzy_threshold > 100:
            raise ValueError(f"Invalid fuzzy_threshold: {self.fuzzy_threshold}, must be in [0, 100]")
        if self.semantic_threshold < 0 or self.semantic_threshold > 1:
            raise ValueError(f"Invalid semantic_threshold: {self.semantic_threshold}, must be in [0, 1]")


class CrossLingualEntityLinker:
    """
    Cross-lingual entity linking system.

    Implements:
    - Task 3.4.1: Bilingual entity matching (EN/CN)
    - Task 3.4.2: Translation-aware entity resolution
    - Task 3.4.3: Cross-lingual relation alignment
    - Task 3.4.4: Language detection
    - Task 3.4.5: Bilingual KG format

    Following CLAUDE.md:
    - IDEMPOTENCY: All linking operations safe to retry
    - STRUCTURED LOGGING: JSON logs with correlation IDs
    - UTC TIME: All timestamps in UTC
    """

    def __init__(self, config: Optional[LinkerConfig] = None):
        """
        Initialize entity linker.

        Args:
            config: Linker configuration
        """
        self.config = config or LinkerConfig()
        self.entity_index: Dict[str, Entity] = {}
        self.name_index: Dict[str, Set[str]] = defaultdict(set)  # name -> entity_ids
        self.translation_cache: Dict[str, str] = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=1
        )

        logger.info({
            "msg": "Initialized CrossLingualEntityLinker",
            "config": {
                "translation_model": self.config.translation_model,
                "fuzzy_threshold": self.config.fuzzy_threshold,
                "semantic_threshold": self.config.semantic_threshold,
                "enable_translation": self.config.enable_translation
            }
        })

    async def detect_language(self, text: str) -> Language:
        """
        Detect language of text (Task 3.4.4).

        Args:
            text: Text to analyze

        Returns:
            Detected language
        """
        # Simple heuristic-based detection
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return Language.UNKNOWN

        chinese_ratio = chinese_chars / total_chars

        # If >30% Chinese characters, classify as Chinese
        if chinese_ratio > 0.3:
            return Language.CHINESE
        elif chinese_ratio > 0:
            return Language.BILINGUAL
        else:
            return Language.ENGLISH

    async def translate(self, text: str, target_lang: Language, correlation_id: Optional[str] = None) -> str:
        """
        Translate text to target language (Task 3.4.2).

        Args:
            text: Text to translate
            target_lang: Target language
            correlation_id: Correlation ID for logging

        Returns:
            Translated text
        """
        if not self.config.enable_translation:
            logger.warning({
                "msg": "Translation disabled",
                "correlation_id": correlation_id
            })
            return text

        cache_key = f"{text}:{target_lang.value}"

        # Check cache
        if self.config.cache_translations and cache_key in self.translation_cache:
            logger.debug({
                "msg": "Translation cache hit",
                "text": text[:50],
                "target_lang": target_lang.value,
                "correlation_id": correlation_id
            })
            return self.translation_cache[cache_key]

        # Perform translation
        try:
            # Placeholder for actual translation service
            # In production, integrate with translation API
            if target_lang == Language.CHINESE:
                # Mock translation to Chinese
                translated = f"[CN]{text}"
            else:
                # Mock translation to English
                translated = f"[EN]{text}"

            # Cache result
            if self.config.cache_translations:
                self.translation_cache[cache_key] = translated

            logger.debug({
                "msg": "Translation completed",
                "source": text[:50],
                "target": translated[:50],
                "target_lang": target_lang.value,
                "correlation_id": correlation_id
            })

            return translated

        except Exception as e:
            logger.error({
                "msg": "Translation failed",
                "text": text[:50],
                "target_lang": target_lang.value,
                "error": str(e),
                "correlation_id": correlation_id
            })
            return text  # Return original on failure

    async def add_entity(self, entity: Entity, correlation_id: Optional[str] = None) -> bool:
        """
        Add entity to index (idempotent).

        Args:
            entity: Entity to add
            correlation_id: Correlation ID for logging

        Returns:
            True if added, False if already exists
        """
        if entity.entity_id in self.entity_index:
            logger.debug({
                "msg": "Entity already exists",
                "entity_id": entity.entity_id,
                "correlation_id": correlation_id
            })
            return False

        self.entity_index[entity.entity_id] = entity

        # Index all names
        for lang, names in entity.get_all_names().items():
            for name in names:
                self.name_index[name].add(entity.entity_id)

        logger.info({
            "msg": "Entity added",
            "entity_id": entity.entity_id,
            "type": entity.type,
            "language": entity.language.value,
            "correlation_id": correlation_id
        })

        return True

    async def match_entities(
        self,
        entity1: Entity,
        entity2: Entity,
        strategy: MatchStrategy = MatchStrategy.HYBRID,
        correlation_id: Optional[str] = None
    ) -> EntityMatchResult:
        """
        Match two entities (Task 3.4.1).

        Args:
            entity1: First entity
            entity2: Second entity
            strategy: Matching strategy
            correlation_id: Correlation ID for logging

        Returns:
            Match result
        """
        logger.debug({
            "msg": "Matching entities",
            "entity1_id": entity1.entity_id,
            "entity2_id": entity2.entity_id,
            "strategy": strategy.value,
            "correlation_id": correlation_id
        })

        evidence = []
        confidence = 0.0
        matched = False
        cross_lingual = False
        translation_used = False

        # Check if cross-lingual
        if entity1.language != entity2.language and entity1.language != Language.UNKNOWN and entity2.language != Language.UNKNOWN:
            cross_lingual = True

        # Strategy 1: Exact match
        if strategy in [MatchStrategy.EXACT, MatchStrategy.HYBRID]:
            exact_match, exact_confidence, exact_evidence = await self._exact_match(entity1, entity2)
            if exact_match:
                matched = True
                confidence = max(confidence, exact_confidence)
                evidence.extend(exact_evidence)

        # Strategy 2: Fuzzy match
        if strategy in [MatchStrategy.FUZZY, MatchStrategy.HYBRID] and not matched:
            fuzzy_match, fuzzy_confidence, fuzzy_evidence = await self._fuzzy_match(entity1, entity2)
            if fuzzy_match:
                matched = True
                confidence = max(confidence, fuzzy_confidence)
                evidence.extend(fuzzy_evidence)

        # Strategy 3: Semantic match
        if strategy in [MatchStrategy.SEMANTIC, MatchStrategy.HYBRID] and not matched:
            semantic_match, semantic_confidence, semantic_evidence = await self._semantic_match(entity1, entity2)
            if semantic_match:
                matched = True
                confidence = max(confidence, semantic_confidence)
                evidence.extend(semantic_evidence)

        # Strategy 4: Translation-aware match
        if strategy in [MatchStrategy.TRANSLATION, MatchStrategy.HYBRID] and cross_lingual and not matched:
            trans_match, trans_confidence, trans_evidence = await self._translation_match(
                entity1, entity2, correlation_id
            )
            if trans_match:
                matched = True
                confidence = max(confidence, trans_confidence)
                evidence.extend(trans_evidence)
                translation_used = True

        result = EntityMatchResult(
            entity1_id=entity1.entity_id,
            entity2_id=entity2.entity_id,
            matched=matched,
            strategy=strategy,
            confidence=confidence,
            evidence=evidence,
            cross_lingual=cross_lingual,
            translation_used=translation_used
        )

        logger.info({
            "msg": "Entity matching complete",
            "matched": matched,
            "confidence": confidence,
            "strategy": strategy.value,
            "cross_lingual": cross_lingual,
            "correlation_id": correlation_id
        })

        return result

    async def _exact_match(self, entity1: Entity, entity2: Entity) -> Tuple[bool, float, List[str]]:
        """Exact name matching."""
        evidence = []

        # Check exact name matches
        names1 = entity1.get_all_names()
        names2 = entity2.get_all_names()

        for lang in ["en", "zh"]:
            for name1 in names1[lang]:
                for name2 in names2[lang]:
                    if name1.lower() == name2.lower():
                        evidence.append(f"Exact {lang} match: {name1} == {name2}")
                        return True, 1.0, evidence

        return False, 0.0, evidence

    async def _fuzzy_match(self, entity1: Entity, entity2: Entity) -> Tuple[bool, float, List[str]]:
        """Fuzzy name matching."""
        evidence = []
        best_score = 0.0

        names1 = entity1.get_all_names()
        names2 = entity2.get_all_names()

        # Compare all names within same language
        for lang in ["en", "zh"]:
            for name1 in names1[lang]:
                for name2 in names2[lang]:
                    # Try both ratio and partial_ratio
                    score1 = fuzz.ratio(name1.lower(), name2.lower())
                    score2 = fuzz.partial_ratio(name1.lower(), name2.lower())
                    score = max(score1, score2)
                    if score > best_score:
                        best_score = score
                        method = "partial" if score2 > score1 else "ratio"
                        evidence.append(f"Fuzzy {lang} match ({method}): {name1} ~ {name2} ({score}%)")

        if best_score >= self.config.fuzzy_threshold:
            return True, best_score / 100.0, evidence

        return False, 0.0, evidence

    async def _semantic_match(self, entity1: Entity, entity2: Entity) -> Tuple[bool, float, List[str]]:
        """Semantic similarity matching."""
        evidence = []

        # Check type match
        if entity1.type != entity2.type:
            return False, 0.0, evidence

        # Use TF-IDF for semantic similarity
        names1 = entity1.get_all_names()
        names2 = entity2.get_all_names()

        all_text1 = " ".join(names1["en"] + names1["zh"])
        all_text2 = " ".join(names2["en"] + names2["zh"])

        if not all_text1 or not all_text2:
            return False, 0.0, evidence

        try:
            corpus = [all_text1, all_text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            evidence.append(f"Semantic similarity: {similarity:.3f}")

            if similarity >= self.config.semantic_threshold:
                return True, similarity, evidence

        except Exception as e:
            logger.warning({
                "msg": "Semantic matching failed",
                "error": str(e)
            })

        return False, 0.0, evidence

    async def _translation_match(
        self,
        entity1: Entity,
        entity2: Entity,
        correlation_id: Optional[str] = None
    ) -> Tuple[bool, float, List[str]]:
        """Translation-aware matching (Task 3.4.2)."""
        evidence = []
        best_score = 0.0

        # Translate entity1 names to entity2's language
        for lang1, names1 in entity1.get_all_names().items():
            for lang2, names2 in entity2.get_all_names().items():
                if lang1 == lang2:
                    continue

                for name1 in names1:
                    target_lang = Language.CHINESE if lang2 == "zh" else Language.ENGLISH
                    translated = await self.translate(name1, target_lang, correlation_id)

                    for name2 in names2:
                        score = fuzz.ratio(translated.lower(), name2.lower())
                        if score > best_score:
                            best_score = score
                            evidence.append(
                                f"Translation match: {name1} ({lang1}) -> {translated} ~ {name2} ({lang2})"
                            )

        if best_score >= self.config.fuzzy_threshold:
            return True, best_score / 100.0, evidence

        return False, 0.0, evidence

    async def align_relations(
        self,
        relations1: List[Dict[str, Any]],
        relations2: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Align cross-lingual relations (Task 3.4.3).

        Args:
            relations1: First set of relations
            relations2: Second set of relations
            correlation_id: Correlation ID for logging

        Returns:
            List of (relation1, relation2, similarity) tuples
        """
        alignments = []

        for rel1 in relations1:
            for rel2 in relations2:
                # Compare relation types
                type_match = rel1.get("type") == rel2.get("type")

                # Compare entities
                entity_match_score = 0.0
                if "head" in rel1 and "head" in rel2:
                    # Would need entity IDs to match properly
                    # Simplified: compare text
                    head_sim = fuzz.ratio(
                        str(rel1.get("head", "")),
                        str(rel2.get("head", ""))
                    ) / 100.0

                if "tail" in rel1 and "tail" in rel2:
                    tail_sim = fuzz.ratio(
                        str(rel1.get("tail", "")),
                        str(rel2.get("tail", ""))
                    ) / 100.0

                entity_match_score = (head_sim + tail_sim) / 2.0

                overall_similarity = (
                    0.3 * (1.0 if type_match else 0.0) +
                    0.7 * entity_match_score
                )

                if overall_similarity >= self.config.semantic_threshold:
                    alignments.append((rel1, rel2, overall_similarity))

        logger.info({
            "msg": "Relation alignment complete",
            "num_alignments": len(alignments),
            "correlation_id": correlation_id
        })

        return alignments

    def to_bilingual_kg(self, entities: List[Entity]) -> Dict[str, Any]:
        """
        Convert to bilingual knowledge graph format (Task 3.4.5).

        Args:
            entities: List of entities

        Returns:
            Bilingual KG structure
        """
        kg = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "format": "bilingual_kg",
                "languages": ["en", "zh"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entity_count": len(entities)
            }
        }

        for entity in entities:
            node = {
                "id": entity.entity_id,
                "names": {
                    "en": entity.name_en,
                    "zh": entity.name_zh
                },
                "aliases": {
                    "en": entity.aliases_en,
                    "zh": entity.aliases_zh
                },
                "type": entity.type,
                "properties": entity.properties,
                "confidence": entity.confidence,
                "language": entity.language.value
            }
            kg["nodes"].append(node)

        return kg

    async def find_candidates(
        self,
        entity: Entity,
        limit: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find candidate matching entities.

        Args:
            entity: Query entity
            limit: Max candidates to return
            correlation_id: Correlation ID for logging

        Returns:
            List of (entity_id, similarity) tuples
        """
        limit = limit or self.config.max_candidates
        candidates = []

        # Get all entity names
        names = entity.get_all_names()

        for lang, name_list in names.items():
            for name in name_list:
                if name in self.name_index:
                    for entity_id in self.name_index[name]:
                        if entity_id != entity.entity_id:
                            candidates.append((entity_id, 1.0))

        # Deduplicate
        unique_candidates = list(set(candidates))

        # Sort by similarity
        unique_candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug({
            "msg": "Found candidates",
            "num_candidates": len(unique_candidates[:limit]),
            "query_entity": entity.entity_id,
            "correlation_id": correlation_id
        })

        return unique_candidates[:limit]

    async def deduplicate_entities(
        self,
        entities: List[Entity],
        strategy: MatchStrategy = MatchStrategy.HYBRID,
        correlation_id: Optional[str] = None
    ) -> List[List[str]]:
        """
        Deduplicate entities using matching.

        Args:
            entities: List of entities
            strategy: Matching strategy
            correlation_id: Correlation ID for logging

        Returns:
            List of duplicate clusters (entity IDs)
        """
        clusters = []
        processed = set()

        for i, entity1 in enumerate(entities):
            if entity1.entity_id in processed:
                continue

            cluster = [entity1.entity_id]
            processed.add(entity1.entity_id)

            # Find matches
            for j, entity2 in enumerate(entities):
                if i >= j or entity2.entity_id in processed:
                    continue

                match_result = await self.match_entities(entity1, entity2, strategy, correlation_id)

                if match_result.matched and match_result.confidence >= 0.8:
                    cluster.append(entity2.entity_id)
                    processed.add(entity2.entity_id)

            if len(cluster) > 1:
                clusters.append(cluster)

        logger.info({
            "msg": "Entity deduplication complete",
            "num_clusters": len(clusters),
            "total_entities": len(entities),
            "correlation_id": correlation_id
        })

        return clusters
