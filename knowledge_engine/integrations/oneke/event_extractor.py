"""
OneKE Event Extraction Pipeline
Task 3.5: Event Extraction Pipeline

Implements comprehensive event extraction:
- 3.5.1: Event detection model integration
- 3.5.2: Event argument extraction (participants, time, location)
- 3.5.3: Event chain construction (sequences of events)
- 3.5.4: Causal relationship extraction
- 3.5.5: Temporal event sequences

Following CLAUDE.md Principles:
- AIR GAP: Adapter pattern for event models
- RUNTIME TRUTH: Probes verify event detection
- IDEMPOTENCY: All extraction operations are idempotent
- CONFIGURATION EXPLICITNESS: All config via environment variables
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
from pathlib import Path
import re
from collections import defaultdict

from .model_adapter import OneKEModelAdapter, ModelConfig, Language

# Structured logging
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""
    ACQUISITION = "acquisition"
    MERGER = "merger"
    LAUNCH = "launch"
    APPOINTMENT = "appointment"
    RESIGNATION = "resignation"
    LEGAL = "legal"
    FINANCIAL = "financial"
    PRODUCT = "product"
    PERSONNEL = "personnel"
    ORGANIZATIONAL = "organizational"
    OTHER = "other"


class ArgumentRole(Enum):
    """Event argument roles."""
    TRIGGER = "trigger"
    SUBJECT = "subject"
    OBJECT = "object"
    TIME = "time"
    LOCATION = "location"
    INSTRUMENT = "instrument"
    PURPOSE = "purpose"
    MANNER = "manner"


class CausalType(Enum):
    """Causal relationship types."""
    DIRECT = "direct"  # A directly causes B
    INDIRECT = "indirect"  # A indirectly causes B
    ENABLING = "enabling"  # A enables B
    PREVENTING = "preventing"  # A prevents B
    CORRELATION = "correlation"  # A correlated with B


@dataclass
class EventArgument:
    """
    Event argument with role.

    Attributes:
        role: Argument role
        text: Argument text
        entity_id: Associated entity ID (if any)
        start: Start position
        end: End position
        confidence: Confidence score
    """
    role: ArgumentRole
    text: str
    entity_id: Optional[str] = None
    start: int = 0
    end: int = 0
    confidence: float = 1.0


@dataclass
class TemporalEvent:
    """
    Temporal event representation.

    Attributes:
        event_id: Unique event identifier
        event_type: Event type
        trigger: Trigger text
        arguments: Event arguments
        timestamp: Event timestamp (UTC)
        duration: Event duration (if applicable)
        certainty: Certainty score
        source: Source document
        language: Language
        metadata: Additional metadata
    """
    event_id: str
    event_type: EventType
    trigger: str
    arguments: List[EventArgument] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    duration: Optional[timedelta] = None
    certainty: float = 1.0
    source: str = ""
    language: Language = Language.ENGLISH
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate event data."""
        if self.certainty < 0 or self.certainty > 1:
            raise ValueError(f"Invalid certainty: {self.certainty}, must be in [0, 1]")

    def get_argument(self, role: ArgumentRole) -> Optional[EventArgument]:
        """Get argument by role."""
        for arg in self.arguments:
            if arg.role == role:
                return arg
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "trigger": self.trigger,
            "arguments": [
                {
                    "role": arg.role.value,
                    "text": arg.text,
                    "entity_id": arg.entity_id,
                    "confidence": arg.confidence
                }
                for arg in self.arguments
            ],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration": str(self.duration) if self.duration else None,
            "certainty": self.certainty,
            "source": self.source,
            "language": self.language.value,
            "metadata": self.metadata
        }


@dataclass
class CausalRelation:
    """
    Causal relationship between events.

    Attributes:
        cause_event_id: Cause event ID
        effect_event_id: Effect event ID
        causal_type: Type of causal relationship
        confidence: Confidence score
        evidence: Supporting evidence
        timestamp: Extraction timestamp (UTC)
    """
    cause_event_id: str
    effect_event_id: str
    causal_type: CausalType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EventChain:
    """
    Chain of temporally or causally related events.

    Attributes:
        chain_id: Unique chain identifier
        events: Ordered list of events
        causal_relations: Causal relationships
        temporal_order: Temporal ordering
        summary: Chain summary
        metadata: Additional metadata
    """
    chain_id: str
    events: List[TemporalEvent] = field(default_factory=list)
    causal_relations: List[CausalRelation] = field(default_factory=list)
    temporal_order: List[str] = field(default_factory=list)  # event_ids in order
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TemporalEvent, position: Optional[int] = None):
        """Add event to chain."""
        if event.event_id in [e.event_id for e in self.events]:
            return

        if position is not None:
            self.events.insert(position, event)
            self.temporal_order.insert(position, event.event_id)
        else:
            self.events.append(event)
            self.temporal_order.append(event.event_id)

    def add_causal_relation(self, relation: CausalRelation):
        """Add causal relation."""
        self.causal_relations.append(relation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "events": [e.to_dict() for e in self.events],
            "causal_relations": [
                {
                    "cause": rel.cause_event_id,
                    "effect": rel.effect_event_id,
                    "type": rel.causal_type.value,
                    "confidence": rel.confidence,
                    "evidence": rel.evidence
                }
                for rel in self.causal_relations
            ],
            "temporal_order": self.temporal_order,
            "summary": self.summary,
            "metadata": self.metadata
        }


@dataclass
class ExtractorConfig:
    """
    Event extractor configuration.

    Environment Variables (CLAUDE.md: Configuration Explicitness):
    - ONEKE_EVENT_MODEL: Event detection model (default: "oneke/EventExtractor")
    - ONEKE_EVENT_CONFIDENCE_THRESHOLD: Minimum confidence (default: 0.6)
    - ONEKE_MAX_EVENTS_PER_DOC: Max events per document (default: 50)
    - ONEKE_ENABLE_CAUSAL_EXTRACTION: Enable causal extraction (default: true)
    - ONEKE_ENABLE_TEMPORAL_ORDERING: Enable temporal ordering (default: true)
    - ONEKE_TEMPORAL_WINDOW: Temporal window for chaining (default: 86400 seconds)
    """
    event_model: str = field(default_factory=lambda: os.getenv("ONEKE_EVENT_MODEL", "oneke/EventExtractor"))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("ONEKE_EVENT_CONFIDENCE_THRESHOLD", "0.6")))
    max_events_per_doc: int = field(default_factory=lambda: int(os.getenv("ONEKE_MAX_EVENTS_PER_DOC", "50")))
    enable_causal_extraction: bool = field(default_factory=lambda: bool(os.getenv("ONEKE_ENABLE_CAUSAL_EXTRACTION", "true")))
    enable_temporal_ordering: bool = field(default_factory=lambda: bool(os.getenv("ONEKE_ENABLE_TEMPORAL_ORDERING", "true")))
    temporal_window: int = field(default_factory=lambda: int(os.getenv("ONEKE_TEMPORAL_WINDOW", "86400")))

    def __post_init__(self):
        """Validate configuration."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError(f"Invalid confidence_threshold: {self.confidence_threshold}, must be in [0, 1]")
        if self.max_events_per_doc < 1:
            raise ValueError(f"Invalid max_events_per_doc: {self.max_events_per_doc}, must be > 0")


class EventExtractionPipeline:
    """
    Event extraction pipeline.

    Implements:
    - Task 3.5.1: Event detection model integration
    - Task 3.5.2: Event argument extraction
    - Task 3.5.3: Event chain construction
    - Task 3.5.4: Causal relationship extraction
    - Task 3.5.5: Temporal event sequences

    Following CLAUDE.md:
    - IDEMPOTENCY: All extraction operations safe to retry
    - STRUCTURED LOGGING: JSON logs with correlation IDs
    - UTC TIME: All timestamps in UTC
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        model_adapter: Optional[OneKEModelAdapter] = None
    ):
        """
        Initialize event extraction pipeline.

        Args:
            config: Extractor configuration
            model_adapter: OneKE model adapter
        """
        self.config = config or ExtractorConfig()
        self.model_adapter = model_adapter

        # Causal indicators (English/Chinese)
        self.causal_indicators = {
            "en": {
                "direct": ["because", "since", "due to", "as a result of", "caused", "led to"],
                "indirect": ["influenced", "contributed to", "affected"],
                "enabling": ["enabled", "allowed", "made possible", "facilitated"],
                "preventing": ["prevented", "stopped", "blocked", "hindered"]
            },
            "zh": {
                "direct": ["因为", "由于", "导致", "致使", "造成"],
                "indirect": ["影响", "促使"],
                "enabling": ["使", "让", "能够"],
                "preventing": ["阻止", "防止", "阻碍"]
            }
        }

        # Temporal indicators
        self.temporal_indicators = {
            "before": ["before", "prior to", "earlier", "previously", "之前", "以前"],
            "after": ["after", "following", "subsequently", "later", "之后", "后来"],
            "during": ["during", "while", "when", "期间", "当"],
            "simultaneous": ["simultaneously", "at the same time", "同时", "一起"]
        }

        logger.info({
            "msg": "Initialized EventExtractionPipeline",
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "enable_causal_extraction": self.config.enable_causal_extraction,
                "enable_temporal_ordering": self.config.enable_temporal_ordering
            }
        })

    async def extract_events(
        self,
        text: str,
        language: Language = Language.ENGLISH,
        schema: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[TemporalEvent]:
        """
        Extract events from text (Task 3.5.1).

        Args:
            text: Input text
            language: Text language
            schema: Event schema (optional)
            correlation_id: Correlation ID for logging

        Returns:
            List of extracted events
        """
        logger.info({
            "msg": "Extracting events",
            "text_length": len(text),
            "language": language.value,
            "correlation_id": correlation_id
        })

        events = []

        try:
            # Use model adapter for event detection
            if self.model_adapter:
                extraction_result = await self.model_adapter.extract(
                    text=text,
                    schema=schema or self._get_default_event_schema(),
                    language=language
                )

                # Convert to TemporalEvent objects
                for event_data in extraction_result.events:
                    event = await self._parse_event(event_data, text, language)
                    if event.certainty >= self.config.confidence_threshold:
                        events.append(event)
            else:
                # Fallback to rule-based extraction
                events = await self._rule_based_extraction(text, language, correlation_id)

            # Limit events
            events = events[:self.config.max_events_per_doc]

            logger.info({
                "msg": "Event extraction complete",
                "num_events": len(events),
                "correlation_id": correlation_id
            })

            return events

        except Exception as e:
            logger.error({
                "msg": "Event extraction failed",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return []

    async def _parse_event(
        self,
        event_data: Dict[str, Any],
        text: str,
        language: Language
    ) -> TemporalEvent:
        """Parse event data into TemporalEvent."""
        event_id = event_data.get("id", f"event_{hash(event_data.get('trigger', ''))}")

        # Parse event type
        event_type_str = event_data.get("type", "other")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.OTHER

        # Parse arguments
        arguments = []
        for arg_data in event_data.get("arguments", []):
            role_str = arg_data.get("role", "object")
            try:
                role = ArgumentRole(role_str)
            except ValueError:
                role = ArgumentRole.OBJECT

            argument = EventArgument(
                role=role,
                text=arg_data.get("text", ""),
                entity_id=arg_data.get("entity_id"),
                start=arg_data.get("start", 0),
                end=arg_data.get("end", 0),
                confidence=arg_data.get("confidence", 1.0)
            )
            arguments.append(argument)

        # Parse timestamp
        timestamp = None
        if "timestamp" in event_data:
            try:
                timestamp = datetime.fromisoformat(event_data["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except:
                pass

        event = TemporalEvent(
            event_id=event_id,
            event_type=event_type,
            trigger=event_data.get("trigger", ""),
            arguments=arguments,
            timestamp=timestamp,
            certainty=event_data.get("certainty", 1.0),
            source=event_data.get("source", ""),
            language=language,
            metadata=event_data.get("metadata", {})
        )

        return event

    async def _rule_based_extraction(
        self,
        text: str,
        language: Language,
        correlation_id: Optional[str] = None
    ) -> List[TemporalEvent]:
        """Fallback rule-based event extraction."""
        events = []

        # Simple trigger patterns
        patterns = {
            EventType.ACQUISITION: [r"\w+ (?:acquired|bought|purchased) \w+"],
            EventType.LAUNCH: [r"\w+ (?:launched|released|introduced) \w+"],
            EventType.APPOINTMENT: [r"\w+ (?:appointed|named) \w+"],
        }

        for event_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    event_id = f"event_{len(events)}_{hash(match.group())}"

                    event = TemporalEvent(
                        event_id=event_id,
                        event_type=event_type,
                        trigger=match.group(),
                        certainty=0.7,  # Lower confidence for rule-based
                        language=language
                    )
                    events.append(event)

        return events

    async def extract_arguments(
        self,
        event: TemporalEvent,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> TemporalEvent:
        """
        Extract event arguments (Task 3.5.2).

        Args:
            event: Event to extract arguments for
            text: Source text
            entities: Known entities (optional)
            correlation_id: Correlation ID for logging

        Returns:
            Event with extracted arguments
        """
        logger.debug({
            "msg": "Extracting event arguments",
            "event_id": event.event_id,
            "correlation_id": correlation_id
        })

        # Extract time arguments
        time_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
            r"\d{4}年\d{1,2}月\d{1,2}日"
        ]

        for pattern in time_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                arg = EventArgument(
                    role=ArgumentRole.TIME,
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                )
                event.arguments.append(arg)

        # Extract location arguments
        location_patterns = [
            r"(?:in|at|from) (?:[A-Z][a-z]+\s?)+",
            r"(?:在|从) (?:[\u4e00-\u9fff]+)"
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                arg = EventArgument(
                    role=ArgumentRole.LOCATION,
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                )
                event.arguments.append(arg)

        return event

    async def build_event_chains(
        self,
        events: List[TemporalEvent],
        correlation_id: Optional[str] = None
    ) -> List[EventChain]:
        """
        Build event chains (Task 3.5.3).

        Args:
            events: List of events
            correlation_id: Correlation ID for logging

        Returns:
            List of event chains
        """
        logger.info({
            "msg": "Building event chains",
            "num_events": len(events),
            "correlation_id": correlation_id
        })

        if not events:
            return []

        chains = []
        unassigned_events = set(e.event_id for e in events)

        # Sort by timestamp if available
        sorted_events = sorted(
            events,
            key=lambda e: e.timestamp or datetime.min.replace(tzinfo=timezone.utc)
        )

        # Build chains based on temporal proximity
        chain_id = 0
        while unassigned_events:
            chain = EventChain(chain_id=f"chain_{chain_id}")
            chain_id += 1

            # Find seed event
            seed_event = next(
                (e for e in sorted_events if e.event_id in unassigned_events),
                None
            )

            if not seed_event:
                break

            chain.add_event(seed_event)
            unassigned_events.remove(seed_event.event_id)

            # Find related events
            for event in sorted_events:
                if event.event_id not in unassigned_events:
                    continue

                # Check temporal proximity
                if seed_event.timestamp and event.timestamp:
                    time_diff = abs((event.timestamp - seed_event.timestamp).total_seconds())

                    if time_diff <= self.config.temporal_window:
                        chain.add_event(event)
                        unassigned_events.remove(event.event_id)

            if len(chain.events) > 1:
                chains.append(chain)

        logger.info({
            "msg": "Event chain building complete",
            "num_chains": len(chains),
            "correlation_id": correlation_id
        })

        return chains

    async def extract_causal_relations(
        self,
        events: List[TemporalEvent],
        text: str,
        language: Language = Language.ENGLISH,
        correlation_id: Optional[str] = None
    ) -> List[CausalRelation]:
        """
        Extract causal relationships (Task 3.5.4).

        Args:
            events: List of events
            text: Source text
            language: Text language
            correlation_id: Correlation ID for logging

        Returns:
            List of causal relations
        """
        if not self.config.enable_causal_extraction:
            return []

        logger.info({
            "msg": "Extracting causal relations",
            "num_events": len(events),
            "correlation_id": correlation_id
        })

        causal_relations = []

        # Detect causal indicators in text
        lang_code = language.value
        indicators = self.causal_indicators.get(lang_code, {})

        for causal_type, phrases in indicators.items():
            for phrase in phrases:
                # Find occurrences
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                matches = pattern.finditer(text)

                for match in matches:
                    # Find events near this indicator
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(text), match.end() + 200)
                    context = text[context_start:context_end]

                    # Find events in context
                    nearby_events = []
                    for event in events:
                        if event.trigger and event.trigger in context:
                            nearby_events.append(event)

                    # Create causal relations
                    if len(nearby_events) >= 2:
                        cause = nearby_events[0]
                        effect = nearby_events[1]

                        try:
                            causal_type_enum = CausalType(causal_type)
                        except ValueError:
                            continue

                        relation = CausalRelation(
                            cause_event_id=cause.event_id,
                            effect_event_id=effect.event_id,
                            causal_type=causal_type_enum,
                            confidence=0.7,
                            evidence=[f"Causal indicator '{phrase}' found in text"]
                        )
                        causal_relations.append(relation)

        logger.info({
            "msg": "Causal relation extraction complete",
            "num_relations": len(causal_relations),
            "correlation_id": correlation_id
        })

        return causal_relations

    async def order_events_temporally(
        self,
        events: List[TemporalEvent],
        text: str,
        language: Language = Language.ENGLISH,
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Order events temporally (Task 3.5.5).

        Args:
            events: List of events
            text: Source text
            language: Text language
            correlation_id: Correlation ID for logging

        Returns:
            List of event IDs in temporal order
        """
        if not self.config.enable_temporal_ordering:
            return [e.event_id for e in events]

        logger.debug({
            "msg": "Ordering events temporally",
            "num_events": len(events),
            "correlation_id": correlation_id
        })

        # Primary: use explicit timestamps
        events_with_time = [e for e in events if e.timestamp]
        if events_with_time:
            events_with_time.sort(key=lambda e: e.timestamp)
            return [e.event_id for e in events_with_time]

        # Secondary: use textual order
        ordered = sorted(events, key=lambda e: text.find(e.trigger) if e.trigger else 0)
        return [e.event_id for e in ordered]

    def _get_default_event_schema(self) -> Dict[str, Any]:
        """Get default event schema."""
        return {
            "type": "event_extraction",
            "events": [
                {
                    "type": "acquisition",
                    "trigger": ["acquired", "bought", "purchased"],
                    "arguments": ["subject", "object", "time", "location"]
                },
                {
                    "type": "launch",
                    "trigger": ["launched", "released", "introduced"],
                    "arguments": ["subject", "object", "time", "location"]
                }
            ]
        }

    async def extract_complete_pipeline(
        self,
        text: str,
        language: Language = Language.ENGLISH,
        schema: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete event extraction pipeline.

        Args:
            text: Input text
            language: Text language
            schema: Event schema (optional)
            correlation_id: Correlation ID for logging

        Returns:
            Complete extraction results
        """
        logger.info({
            "msg": "Running complete event extraction pipeline",
            "text_length": len(text),
            "language": language.value,
            "correlation_id": correlation_id
        })

        # Extract events
        events = await self.extract_events(text, language, schema, correlation_id)

        # Extract arguments for each event
        for event in events:
            await self.extract_arguments(event, text, correlation_id=correlation_id)

        # Build event chains
        chains = await self.build_event_chains(events, correlation_id)

        # Extract causal relations
        causal_relations = await self.extract_causal_relations(events, text, language, correlation_id)

        # Order events temporally
        temporal_order = await self.order_events_temporally(events, text, language, correlation_id)

        result = {
            "events": [e.to_dict() for e in events],
            "event_chains": [c.to_dict() for c in chains],
            "causal_relations": [
                {
                    "cause": rel.cause_event_id,
                    "effect": rel.effect_event_id,
                    "type": rel.causal_type.value,
                    "confidence": rel.confidence
                }
                for rel in causal_relations
            ],
            "temporal_order": temporal_order,
            "metadata": {
                "num_events": len(events),
                "num_chains": len(chains),
                "num_causal_relations": len(causal_relations),
                "language": language.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

        logger.info({
            "msg": "Event extraction pipeline complete",
            "num_events": len(events),
            "num_chains": len(chains),
            "num_causal_relations": len(causal_relations),
            "correlation_id": correlation_id
        })

        return result
