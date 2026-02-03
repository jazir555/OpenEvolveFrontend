"""
Conversation Analyzer - Production Grade

Task 2.4: Conversation Analysis
- 2.4.1: Integrate message array processing
- 2.4.2: Implement speaker entity extraction
- 2.4.3: Add speaker-concept relationship extraction
- 2.4.4: Implement conversation summarization
- 2.4.5: Add conversation-to-knowledge-graph pipeline

Following CLAUDE.md Principles:
- AIR GAP: Adapter pattern, no direct imports
- IDEMPOTENCY: Analysis safe to retry
- CONFIGURATION EXPLICITNESS: All config via env vars
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    A single message in a conversation.

    All timestamps in UTC (LAW OF UTC).
    """
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    speaker_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SpeakerEntity:
    """
    Entity extracted from a speaker's messages.
    """
    entity_id: str
    speaker_id: str
    entity_name: str
    entity_type: str  # "person", "organization", "concept", "topic"
    frequency: int = 1
    confidence: float = 1.0
    first_mentioned: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_mentioned: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SpeakerConceptRelation:
    """
    Relationship between a speaker and a concept.
    """
    relation_id: str
    speaker_id: str
    concept: str
    relation_type: str  # "mentions", "asks_about", "discusses", "expert_in"
    strength: float = 0.5  # 0.0 to 1.0
    evidence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConversationSummary:
    """
    Summary of a conversation.

    Task 2.4.4: Implement conversation summarization.
    """
    conversation_id: str
    topic: str
    participants: List[str]
    key_points: List[str]
    entities_mentioned: List[str]
    duration_seconds: float = 0.0
    message_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConversationResult:
    """
    Result of conversation analysis.

    Task 2.4.5: Add conversation-to-knowledge-graph pipeline.
    """
    correlation_id: str
    conversation_id: str

    # Extracted data
    speaker_entities: List[SpeakerEntity] = field(default_factory=list)
    speaker_concept_relations: List[SpeakerConceptRelation] = field(default_factory=list)
    summary: Optional[ConversationSummary] = None

    # Knowledge graph representation
    entities: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)

    # Metrics
    processing_time_seconds: float = 0.0
    total_speakers: int = 0
    total_entities: int = 0
    total_relations: int = 0

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "conversation_id": self.conversation_id,
            "speaker_entities": [e.to_dict() for e in self.speaker_entities],
            "speaker_concept_relations": [r.to_dict() for r in self.speaker_concept_relations],
            "summary": self.summary.to_dict() if self.summary else None,
            "entities": self.entities,
            "relationships": self.relationships,
            "processing_time_seconds": self.processing_time_seconds,
            "total_speakers": self.total_speakers,
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


@dataclass
class ConversationAnalyzerConfig:
    """
    Conversation analyzer configuration.

    LAW OF CONFIGURATION EXPLICITNESS.
    """
    # Entity extraction
    entity_model: str = field(
        default_factory=lambda: os.getenv("KGGEN_CONV_ENTITY_MODEL", "gpt-4o")
    )
    entity_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("KGGEN_CONV_ENTITY_MIN_CONFIDENCE", "0.5"))
    )

    # Summarization
    summary_model: str = field(
        default_factory=lambda: os.getenv("KGGEN_CONV_SUMMARY_MODEL", "gpt-4o")
    )
    summary_max_length: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_CONV_SUMMARY_MAX_LENGTH", "500"))
    )

    # Processing
    min_messages_for_summary: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_CONV_MIN_MESSAGES", "3"))
    )

    # Timeouts
    entity_timeout: float = field(
        default_factory=lambda: float(os.getenv("KGGEN_CONV_ENTITY_TIMEOUT", "120.0"))
    )
    summary_timeout: float = field(
        default_factory=lambda: float(os.getenv("KGGEN_CONV_SUMMARY_TIMEOUT", "180.0"))
    )

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.entity_min_confidence <= 1.0:
            raise ValueError(f"Invalid entity_min_confidence: {self.entity_min_confidence}")
        logger.info("ConversationAnalyzerConfig validated", extra={"config": asdict(self)})


class SpeakerEntityExtractor:
    """
    Extract entities from speaker messages.

    Task 2.4.2: Implement speaker entity extraction.
    """

    def __init__(self, config: ConversationAnalyzerConfig):
        """
        Initialize extractor.

        Args:
            config: Analyzer configuration
        """
        self.config = config
        self._entity_cache: Dict[str, List[str]] = {}

    async def extract_entities(
        self,
        messages: List[Message],
        speaker_id: str,
        correlation_id: str
    ) -> List[SpeakerEntity]:
        """
        Extract entities for a speaker.

        Args:
            messages: Messages from speaker
            speaker_id: Speaker identifier
            correlation_id: Correlation ID

        Returns:
            List of speaker entities
        """
        # Filter messages by speaker
        speaker_messages = [m for m in messages if m.speaker_id == speaker_id]

        if not speaker_messages:
            return []

        # Combine message content
        combined_text = "\n".join([m.content for m in speaker_messages])

        # Extract entities
        entity_names = await self._extract_entity_names(
            combined_text,
            correlation_id
        )

        # Create speaker entities
        speaker_entities = []

        for entity_name in entity_names:
            # Determine entity type
            entity_type = self._classify_entity_type(entity_name)

            entity = SpeakerEntity(
                entity_id=f"ent-{uuid.uuid4().hex[:16]}",
                speaker_id=speaker_id,
                entity_name=entity_name,
                entity_type=entity_type,
                frequency=self._count_mentions(entity_name, speaker_messages),
                confidence=self.config.entity_min_confidence
            )

            speaker_entities.append(entity)

        logger.info(
            f"Extracted {len(speaker_entities)} entities for speaker {speaker_id}",
            extra={"correlation_id": correlation_id}
        )

        return speaker_entities

    async def _extract_entity_names(
        self,
        text: str,
        correlation_id: str
    ) -> List[str]:
        """
        Extract entity names from text.

        Args:
            text: Input text
            correlation_id: Correlation ID

        Returns:
            List of entity names
        """
        try:
            # Try LLM extraction
            return await self._extract_entities_llm(text, correlation_id)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, using fallback")
            return self._extract_entities_fallback(text)

    async def _extract_entities_llm(
        self,
        text: str,
        correlation_id: str
    ) -> List[str]:
        """
        Extract entities using LLM.

        Args:
            text: Input text
            correlation_id: Correlation ID

        Returns:
            List of entity names
        """
        from knowledge_engine.llm_utils import call_llm

        prompt = f"""Extract all entities (people, organizations, concepts, topics) from the following text.

Text:
{text[:2000]}

Return entities as a JSON list of strings.
"""

        response = await call_llm(
            prompt=prompt,
            model=self.config.entity_model,
            temperature=0.0,
            max_tokens=2000,
            timeout=self.config.entity_timeout
        )

        # Parse response
        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                return [str(e) for e in entities if len(str(e)) > 2]
        except json.JSONDecodeError:
            pass

        return []

    def _extract_entities_fallback(self, text: str) -> List[str]:
        """
        Fallback entity extraction using patterns.

        Args:
            text: Input text

        Returns:
            List of entity names
        """
        import re

        # Extract capitalized phrases
        pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        matches = re.findall(pattern, text)

        # Filter common words
        stop_words = {'This', 'That', 'These', 'Those', 'The', 'A', 'An'}
        filtered = [m for m in matches if m not in stop_words and len(m) > 2]

        return list(set(filtered))[:50]

    def _classify_entity_type(self, entity_name: str) -> str:
        """
        Classify entity type.

        Args:
            entity_name: Entity name

        Returns:
            Entity type
        """
        # Simple heuristic classification
        entity_lower = entity_name.lower()

        # Check for organization indicators
        org_indicators = ['corp', 'inc', 'llc', 'company', 'organization', 'institute']
        if any(ind in entity_lower for ind in org_indicators):
            return "organization"

        # Check for person indicators (default)
        return "person"

    def _count_mentions(self, entity: str, messages: List[Message]) -> int:
        """
        Count mentions of entity in messages.

        Args:
            entity: Entity name
            messages: Messages to search

        Returns:
            Mention count
        """
        count = 0
        for message in messages:
            if entity.lower() in message.content.lower():
                count += 1
        return count


class ConversationAnalyzer:
    """
    Analyze conversations and extract knowledge.

    Task 2.4.1: Integrate message array processing.

    Following CLAUDE.md:
    - IDEMPOTENCY: Analysis safe to retry
    - STRUCTURED LOGGING: JSON with correlation_id
    """

    def __init__(self, config: Optional[ConversationAnalyzerConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or ConversationAnalyzerConfig()
        self.config.validate()

        self.entity_extractor = SpeakerEntityExtractor(self.config)

        logger.info(
            "ConversationAnalyzer initialized",
            extra={"config": asdict(self.config)}
        )

    async def analyze(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> ConversationResult:
        """
        Analyze conversation.

        Args:
            messages: List of message dictionaries
            conversation_id: Optional conversation ID
            correlation_id: Optional correlation ID

        Returns:
            Conversation analysis result
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:16]}"
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"Analyzing conversation: {conversation_id}",
            extra={
                "correlation_id": correlation_id,
                "message_count": len(messages)
            }
        )

        # Parse messages
        parsed_messages = self._parse_messages(messages)

        # Get speakers
        speakers = self._get_speakers(parsed_messages)

        # Extract speaker entities
        # Task 2.4.2: Implement speaker entity extraction
        all_speaker_entities: List[SpeakerEntity] = []

        for speaker_id in speakers:
            speaker_entities = await self.entity_extractor.extract_entities(
                parsed_messages,
                speaker_id,
                correlation_id
            )
            all_speaker_entities.extend(speaker_entities)

        # Extract speaker-concept relations
        # Task 2.4.3: Add speaker-concept relationship extraction
        relations = await self._extract_speaker_concept_relations(
            parsed_messages,
            all_speaker_entities,
            correlation_id
        )

        # Generate summary
        # Task 2.4.4: Implement conversation summarization
        summary = None
        if len(parsed_messages) >= self.config.min_messages_for_summary:
            summary = await self._generate_summary(
                parsed_messages,
                conversation_id,
                correlation_id
            )

        # Convert to knowledge graph
        # Task 2.4.5: Add conversation-to-knowledge-graph pipeline
        entities, relationships = self._to_knowledge_graph(
            all_speaker_entities,
            relations
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = ConversationResult(
            correlation_id=correlation_id,
            conversation_id=conversation_id,
            speaker_entities=all_speaker_entities,
            speaker_concept_relations=relations,
            summary=summary,
            entities=entities,
            relationships=relationships,
            processing_time_seconds=processing_time,
            total_speakers=len(speakers),
            total_entities=len(entities),
            total_relations=len(relationships),
            started_at=start_time.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat()
        )

        logger.info(
            f"Conversation analysis complete: {conversation_id}",
            extra={
                "correlation_id": correlation_id,
                "speakers": len(speakers),
                "entities": len(entities),
                "relations": len(relations)
            }
        )

        return result

    def _parse_messages(self, messages: List[Dict[str, Any]]) -> List[Message]:
        """
        Parse message dictionaries.

        Args:
            messages: Raw message dictionaries

        Returns:
            List of Message objects
        """
        parsed = []

        for msg in messages:
            message = Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", datetime.now(timezone.utc).isoformat()),
                speaker_id=msg.get("speaker_id") or msg.get("role", "user"),
                message_id=msg.get("message_id", f"msg-{uuid.uuid4().hex[:16]}")
            )
            parsed.append(message)

        return parsed

    def _get_speakers(self, messages: List[Message]) -> List[str]:
        """
        Get unique speakers.

        Args:
            messages: List of messages

        Returns:
            List of speaker IDs
        """
        speakers = set(m.speaker_id for m in messages if m.speaker_id)
        return list(speakers)

    async def _extract_speaker_concept_relations(
        self,
        messages: List[Message],
        speaker_entities: List[SpeakerEntity],
        correlation_id: str
    ) -> List[SpeakerConceptRelation]:
        """
        Extract speaker-concept relationships.

        Task 2.4.3: Add speaker-concept relationship extraction.

        Args:
            messages: Messages
            speaker_entities: Speaker entities
            correlation_id: Correlation ID

        Returns:
            List of relations
        """
        relations: List[SpeakerConceptRelation] = []

        # Group entities by speaker
        by_speaker: Dict[str, List[SpeakerEntity]] = defaultdict(list)
        for entity in speaker_entities:
            by_speaker[entity.speaker_id].append(entity)

        # Create relations
        for speaker_id, entities in by_speaker.items():
            for entity in entities:
                # Determine relation type based on frequency
                if entity.frequency >= 3:
                    relation_type = "discusses"
                elif "?" in entity.entity_name:
                    relation_type = "asks_about"
                else:
                    relation_type = "mentions"

                relation = SpeakerConceptRelation(
                    relation_id=f"rel-{uuid.uuid4().hex[:16]}",
                    speaker_id=speaker_id,
                    concept=entity.entity_name,
                    relation_type=relation_type,
                    strength=min(entity.frequency / 10.0, 1.0),
                    evidence_count=entity.frequency
                )

                relations.append(relation)

        logger.info(
            f"Extracted {len(relations)} speaker-concept relations",
            extra={"correlation_id": correlation_id}
        )

        return relations

    async def _generate_summary(
        self,
        messages: List[Message],
        conversation_id: str,
        correlation_id: str
    ) -> ConversationSummary:
        """
        Generate conversation summary.

        Task 2.4.4: Implement conversation summarization.

        Args:
            messages: Messages
            conversation_id: Conversation ID
            correlation_id: Correlation ID

        Returns:
            Conversation summary
        """
        try:
            # Combine messages
            conversation_text = "\n".join([
                f"{m.role}: {m.content}"
                for m in messages
            ])

            # Generate summary using LLM
            summary_text = await self._summarize_llm(conversation_text, correlation_id)

            # Extract participants
            participants = list(set(m.speaker_id for m in messages if m.speaker_id))

            # Calculate duration
            if len(messages) >= 2:
                start = datetime.fromisoformat(messages[0].timestamp)
                end = datetime.fromisoformat(messages[-1].timestamp)
                duration = (end - start).total_seconds()
            else:
                duration = 0.0

            summary = ConversationSummary(
                conversation_id=conversation_id,
                topic=summary_text[:100],
                participants=participants,
                key_points=[summary_text],
                entities_mentioned=[],  # Would be extracted from summary
                duration_seconds=duration,
                message_count=len(messages)
            )

            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Return minimal summary
            return ConversationSummary(
                conversation_id=conversation_id,
                topic="Conversation",
                participants=[],
                key_points=[],
                entities_mentioned=[],
                message_count=len(messages)
            )

    async def _summarize_llm(self, text: str, correlation_id: str) -> str:
        """
        Summarize text using LLM.

        Args:
            text: Input text
            correlation_id: Correlation ID

        Returns:
            Summary text
        """
        from knowledge_engine.llm_utils import call_llm

        prompt = f"""Summarize the following conversation in 1-2 sentences.

Conversation:
{text[:3000]}

Summary:"""

        response = await call_llm(
            prompt=prompt,
            model=self.config.summary_model,
            temperature=0.3,
            max_tokens=self.config.summary_max_length,
            timeout=self.config.summary_timeout
        )

        return response.strip()

    def _to_knowledge_graph(
        self,
        speaker_entities: List[SpeakerEntity],
        relations: List[SpeakerConceptRelation]
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Convert analysis to knowledge graph format.

        Task 2.4.5: Add conversation-to-knowledge-graph pipeline.

        Args:
            speaker_entities: Speaker entities
            relations: Speaker-concept relations

        Returns:
            Tuple of (entities, relationships)
        """
        # Extract entities
        entities_set = set()

        # Add speakers as entities
        for entity in speaker_entities:
            entities_set.add(entity.speaker_id)
            entities_set.add(entity.entity_name)

        # Extract relationships
        relationships = []

        for rel in relations:
            relationships.append({
                "subject": rel.speaker_id,
                "predicate": rel.relation_type,
                "object": rel.concept,
                "strength": str(rel.strength),
                "evidence_count": str(rel.evidence_count)
            })

        return list(entities_set), relationships

    async def close(self) -> None:
        """Cleanup resources."""
        logger.info("ConversationAnalyzer closed")
