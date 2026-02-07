"""
KnowledgeState - Temporal knowledge tracking implementation

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify state changes
- IDEMPOTENCY: State operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs

Author: OpenEvolve Distinguished Engineer
Version: 2.0.0
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from threading import Lock
import uuid
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeTriple:
    """
    Represents a knowledge triple (subject, predicate, object).

    Attributes:
        subject: Subject entity
        predicate: Relationship/attribute
        object: Object entity or value
        confidence: Confidence score (0-1)
        timestamp: UTC timestamp when triple was added
        source: Source of knowledge
    """
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeTriple':
        """Create triple from dictionary."""
        return cls(
            subject=data['subject'],
            predicate=data['predicate'],
            obj=data['obj'],
            confidence=data.get('confidence', 1.0),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            source=data.get('source'),
            metadata=data.get('metadata', {})
        )

    def __hash__(self):
        """Make triple hashable."""
        return hash((self.subject, self.predicate, self.obj))

    def __eq__(self, other):
        """Check triple equality."""
        if not isinstance(other, KnowledgeTriple):
            return False
        return (self.subject == other.subject and
                self.predicate == other.predicate and
                self.obj == other.obj)


@dataclass
class StateSnapshot:
    """
    Represents a snapshot of knowledge at a point in time.

    Attributes:
        timestamp: UTC timestamp of snapshot
        triples: List of knowledge triples
        facts: List of accumulated facts
        uncertainties: List of uncertainties
        version: State version number
    """
    timestamp: str
    triples: List[KnowledgeTriple]
    facts: List[str]
    uncertainties: List[str]
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "triples": [t.to_dict() for t in self.triples],
            "facts": self.facts,
            "uncertainties": self.uncertainties,
            "version": self.version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create snapshot from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            triples=[KnowledgeTriple.from_dict(t) for t in data.get('triples', [])],
            facts=data.get('facts', []),
            uncertainties=data.get('uncertainties', []),
            version=data['version'],
            metadata=data.get('metadata', {})
        )


class KnowledgeState:
    """
    Temporal knowledge state tracker.

    Features:
    - Track knowledge over time
    - Store temporal snapshots
    - Version tracking
    - State queries
    - Thread-safe operations
    - Idempotent operations
    - Structured logging
    """

    def __init__(self, query: str, correlation_id: Optional[str] = None):
        """
        Initialize knowledge state.

        Args:
            query: Research/query context
            correlation_id: Optional correlation ID for logging
        """
        self.query = query
        self._triples: List[KnowledgeTriple] = []
        self._facts: List[str] = []
        self._uncertainties: List[str] = []
        self._snapshots: List[StateSnapshot] = []
        self._version = 0
        self._lock = Lock()
        self._async_lock = asyncio.Lock()

        # Correlation ID for structured logging
        self._correlation_id = correlation_id or str(uuid.uuid4())

        # Backward compatibility: add search_history attribute
        self._search_history: List[Dict[str, Any]] = []
        self._current_understanding: str = ""
        self._candidate_answers: List[str] = []  # Backward compatibility

        self._log("info", "KnowledgeState initialized", query=query)

    # Backward compatibility properties
    @property
    def search_history(self) -> List[Dict[str, Any]]:
        """Get search history (backward compatibility)."""
        with self._lock:
            return self._search_history.copy()

    @property
    def current_understanding(self) -> str:
        """Get current understanding (backward compatibility)."""
        with self._lock:
            return self._current_understanding

    @property
    def candidate_answers(self) -> List[str]:
        """Get candidate answers (backward compatibility)."""
        with self._lock:
            return self._candidate_answers.copy()

    @property
    def facts(self) -> List[str]:
        """Get facts list (backward compatibility - extracts from triples)."""
        with self._lock:
            return self._facts.copy()

    @property
    def uncertainties(self) -> List[str]:
        """Get uncertainties list (backward compatibility)."""
        with self._lock:
            return self._uncertainties.copy()

    def set_current_understanding(self, understanding: str) -> None:
        """
        Set current understanding (backward compatibility).

        Args:
            understanding: Current understanding text
        """
        with self._lock:
            self._current_understanding = understanding
            self._log("debug", "Current understanding updated", understanding=understanding[:100])

    def add_fact(self, fact: str) -> None:
        """
        Add a fact (backward compatibility).

        Args:
            fact: Fact text to add
        """
        with self._lock:
            if fact not in self._facts:
                self._facts.append(fact)
                self._version += 1
                self._log("debug", "Fact added", fact=fact[:100])

    def add_uncertainty(self, uncertainty: str) -> None:
        """
        Add an uncertainty (backward compatibility).

        Args:
            uncertainty: Uncertainty text to add
        """
        with self._lock:
            if uncertainty not in self._uncertainties:
                self._uncertainties.append(uncertainty)
                self._version += 1
                self._log("debug", "Uncertainty added", uncertainty=uncertainty[:100])

    def add_search_result(self, search_result: Dict[str, Any]) -> None:
        """
        Add a search result to history (backward compatibility).

        Args:
            search_result: Search result dictionary
        """
        with self._lock:
            self._search_history.append(search_result)
            self._log("debug", "Search result added", result_type=type(search_result).__name__)

    def _log(self, level: str, message: str, **kwargs):
        """
        Structured logging with correlation ID.

        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            **kwargs: Additional context
        """
        log_data = {
            "msg": message,
            "correlation_id": self._correlation_id,
            "query": self.query,
            "version": self._version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }

        log_func = getattr(logger, level, logger.info)
        log_func(json.dumps(log_data))

    def add_knowledge(
        self,
        triples: List[Tuple[str, str, str]],
        timestamp: Optional[str] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        Add knowledge triples to state (synchronous).

        IDEMPOTENT: Duplicate triples are ignored.

        Args:
            triples: List of (subject, predicate, object) tuples
            timestamp: UTC timestamp (defaults to now)
            source: Optional knowledge source

        Returns:
            True if successful, False on error
        """
        try:
            timestamp = timestamp or datetime.now(timezone.utc).isoformat()

            with self._lock:
                added_count = 0

                for subject, predicate, obj in triples:
                    # Create triple
                    triple = KnowledgeTriple(
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        timestamp=timestamp,
                        source=source
                    )

                    # Check for duplicate (idempotent)
                    if triple not in self._triples:
                        self._triples.append(triple)
                        added_count += 1

                # Create snapshot
                self._create_snapshot(timestamp)

                self._log("info", "Knowledge added",
                         added_count=added_count,
                         total_count=len(self._triples))

                return True

        except Exception as e:
            self._log("error", "Failed to add knowledge", error=str(e))
            return False

    async def add_knowledge_async(
        self,
        triples: List[Tuple[str, str, str]],
        timestamp: Optional[str] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        Add knowledge triples to state (asynchronous).

        IDEMPOTENT: Duplicate triples are ignored.

        Args:
            triples: List of (subject, predicate, object) tuples
            timestamp: UTC timestamp (defaults to now)
            source: Optional knowledge source

        Returns:
            True if successful, False on error
        """
        try:
            timestamp = timestamp or datetime.now(timezone.utc).isoformat()

            async with self._async_lock:
                added_count = 0

                for subject, predicate, obj in triples:
                    # Create triple
                    triple = KnowledgeTriple(
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        timestamp=timestamp,
                        source=source
                    )

                    # Check for duplicate (idempotent)
                    if triple not in self._triples:
                        self._triples.append(triple)
                        added_count += 1

                # Create snapshot
                await self._create_snapshot_async(timestamp)

                self._log("info", "Knowledge added",
                         added_count=added_count,
                         total_count=len(self._triples))

                return True

        except Exception as e:
            self._log("error", "Failed to add knowledge", error=str(e))
            return False

    def add_fact(self, fact: str) -> bool:
        """
        Add a fact to the state (synchronous).

        IDEMPOTENT: Duplicate facts are ignored.

        Args:
            fact: Fact statement

        Returns:
            True if added, False if duplicate or error
        """
        try:
            with self._lock:
                if fact not in self._facts:
                    self._facts.append(fact)
                    self._log("debug", "Fact added", fact=fact)
                    return True
                return False

        except Exception as e:
            self._log("error", "Failed to add fact", error=str(e))
            return False

    async def add_fact_async(self, fact: str) -> bool:
        """
        Add a fact to the state (asynchronous).

        IDEMPOTENT: Duplicate facts are ignored.

        Args:
            fact: Fact statement

        Returns:
            True if added, False if duplicate or error
        """
        try:
            async with self._async_lock:
                if fact not in self._facts:
                    self._facts.append(fact)
                    self._log("debug", "Fact added", fact=fact)
                    return True
                return False

        except Exception as e:
            self._log("error", "Failed to add fact", error=str(e))
            return False

    def add_uncertainty(self, uncertainty: str) -> bool:
        """
        Add an uncertainty to the state (synchronous).

        IDEMPOTENT: Duplicate uncertainties are ignored.

        Args:
            uncertainty: Uncertainty statement

        Returns:
            True if added, False if duplicate or error
        """
        try:
            with self._lock:
                if uncertainty not in self._uncertainties:
                    self._uncertainties.append(uncertainty)
                    self._log("debug", "Uncertainty added", uncertainty=uncertainty)
                    return True
                return False

        except Exception as e:
            self._log("error", "Failed to add uncertainty", error=str(e))
            return False

    async def add_uncertainty_async(self, uncertainty: str) -> bool:
        """
        Add an uncertainty to the state (asynchronous).

        IDEMPOTENT: Duplicate uncertainties are ignored.

        Args:
            uncertainty: Uncertainty statement

        Returns:
            True if added, False if duplicate or error
        """
        try:
            async with self._async_lock:
                if uncertainty not in self._uncertainties:
                    self._uncertainties.append(uncertainty)
                    self._log("debug", "Uncertainty added", uncertainty=uncertainty)
                    return True
                return False

        except Exception as e:
            self._log("error", "Failed to add uncertainty", error=str(e))
            return False

    def _create_snapshot(self, timestamp: Optional[str] = None):
        """Create a snapshot of current state (synchronous, private)."""
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self._version += 1

        snapshot = StateSnapshot(
            timestamp=timestamp,
            triples=deepcopy(self._triples),
            facts=deepcopy(self._facts),
            uncertainties=deepcopy(self._uncertainties),
            version=self._version,
            metadata={
                "query": self.query,
                "correlation_id": self._correlation_id
            }
        )

        self._snapshots.append(snapshot)

    async def _create_snapshot_async(self, timestamp: Optional[str] = None):
        """Create a snapshot of current state (asynchronous, private)."""
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self._version += 1

        snapshot = StateSnapshot(
            timestamp=timestamp,
            triples=deepcopy(self._triples),
            facts=deepcopy(self._facts),
            uncertainties=deepcopy(self._uncertainties),
            version=self._version,
            metadata={
                "query": self.query,
                "correlation_id": self._correlation_id
            }
        )

        self._snapshots.append(snapshot)

    def get_state_at_time(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Get state snapshot at a specific time (synchronous).

        Args:
            timestamp: UTC timestamp to query

        Returns:
            State snapshot dictionary or None if not found
        """
        with self._lock:
            # Find snapshot closest to timestamp (before or equal)
            candidates = [
                s for s in self._snapshots
                if s.timestamp <= timestamp
            ]

            if not candidates:
                return None

            # Get most recent snapshot before timestamp
            snapshot = max(candidates, key=lambda s: s.timestamp)
            return snapshot.to_dict()

    async def get_state_at_time_async(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Get state snapshot at a specific time (asynchronous).

        Args:
            timestamp: UTC timestamp to query

        Returns:
            State snapshot dictionary or None if not found
        """
        async with self._async_lock:
            # Find snapshot closest to timestamp (before or equal)
            candidates = [
                s for s in self._snapshots
                if s.timestamp <= timestamp
            ]

            if not candidates:
                return None

            # Get most recent snapshot before timestamp
            snapshot = max(candidates, key=lambda s: s.timestamp)
            return snapshot.to_dict()

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state (synchronous).

        Returns:
            Current state dictionary
        """
        with self._lock:
            return {
                "query": self.query,
                "triples": [t.to_dict() for t in self._triples],
                "facts": self._facts,
                "uncertainties": self._uncertainties,
                "version": self._version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": self._correlation_id
            }

    async def get_current_state_async(self) -> Dict[str, Any]:
        """
        Get current state (asynchronous).

        Returns:
            Current state dictionary
        """
        async with self._async_lock:
            return {
                "query": self.query,
                "triples": [t.to_dict() for t in self._triples],
                "facts": self._facts,
                "uncertainties": self._uncertainties,
                "version": self._version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": self._correlation_id
            }

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get state history (synchronous).

        Returns:
            List of state snapshots
        """
        with self._lock:
            return [s.to_dict() for s in self._snapshots]

    async def get_history_async(self) -> List[Dict[str, Any]]:
        """
        Get state history (asynchronous).

        Returns:
            List of state snapshots
        """
        async with self._async_lock:
            return [s.to_dict() for s in self._snapshots]

    def search_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search triples by pattern (synchronous).

        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            obj: Filter by object (optional)

        Returns:
            List of matching triple dictionaries
        """
        with self._lock:
            results = []

            for triple in self._triples:
                if subject and triple.subject != subject:
                    continue
                if predicate and triple.predicate != predicate:
                    continue
                if obj and triple.obj != obj:
                    continue

                results.append(triple.to_dict())

            return results

    async def search_triples_async(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search triples by pattern (asynchronous).

        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            obj: Filter by object (optional)

        Returns:
            List of matching triple dictionaries
        """
        async with self._async_lock:
            results = []

            for triple in self._triples:
                if subject and triple.subject != subject:
                    continue
                if predicate and triple.predicate != predicate:
                    continue
                if obj and triple.obj != obj:
                    continue

                results.append(triple.to_dict())

            return results

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to dictionary (synchronous).

        Returns:
            Dictionary representation
        """
        with self._lock:
            return {
                "query": self.query,
                "triples": [t.to_dict() for t in self._triples],
                "facts": self._facts,
                "uncertainties": self._uncertainties,
                "version": self._version,
                "snapshots": [s.to_dict() for s in self._snapshots],
                "correlation_id": self._correlation_id,
                # Backward compatibility fields
                "search_history": self._search_history,
                "candidate_answers": self._candidate_answers,
                "current_understanding": self._current_understanding
            }

    async def to_dict_async(self) -> Dict[str, Any]:
        """
        Serialize state to dictionary (asynchronous).

        Returns:
            Dictionary representation
        """
        async with self._async_lock:
            return {
                "query": self.query,
                "triples": [t.to_dict() for t in self._triples],
                "facts": self._facts,
                "uncertainties": self._uncertainties,
                "version": self._version,
                "snapshots": [s.to_dict() for s in self._snapshots],
                "correlation_id": self._correlation_id,
                # Backward compatibility fields
                "search_history": self._search_history,
                "candidate_answers": self._candidate_answers,
                "current_understanding": self._current_understanding
            }

    def to_json(self) -> str:
        """
        Serialize state to JSON (synchronous).

        Returns:
            JSON string representation
        """
        data = self.to_dict()
        return json.dumps(data, indent=2)

    async def to_json_async(self) -> str:
        """
        Serialize state to JSON (asynchronous).

        Returns:
            JSON string representation
        """
        data = await self.to_dict_async()
        return json.dumps(data, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeState':
        """
        Create state from dictionary (synchronous).

        Args:
            data: Dictionary representation

        Returns:
            KnowledgeState instance
        """
        state = cls(
            query=data['query'],
            correlation_id=data.get('correlation_id')
        )

        # Load triples
        state._triples = [
            KnowledgeTriple.from_dict(t)
            for t in data.get('triples', [])
        ]

        # Load facts and uncertainties
        state._facts = data.get('facts', [])
        state._uncertainties = data.get('uncertainties', [])

        # Load snapshots
        state._snapshots = [
            StateSnapshot.from_dict(s)
            for s in data.get('snapshots', [])
        ]

        # Load version
        state._version = data.get('version', 0)

        # Load backward compatibility fields
        state._search_history = data.get('search_history', [])
        state._candidate_answers = data.get('candidate_answers', [])
        state._current_understanding = data.get('current_understanding', "")

        return state

    @classmethod
    async def from_dict_async(cls, data: Dict[str, Any]) -> 'KnowledgeState':
        """
        Create state from dictionary (asynchronous).

        Args:
            data: Dictionary representation

        Returns:
            KnowledgeState instance
        """
        state = cls(
            query=data['query'],
            correlation_id=data.get('correlation_id')
        )

        # Load triples
        state._triples = [
            KnowledgeTriple.from_dict(t)
            for t in data.get('triples', [])
        ]

        # Load facts and uncertainties
        state._facts = data.get('facts', [])
        state._uncertainties = data.get('uncertainties', [])

        # Load snapshots
        state._snapshots = [
            StateSnapshot.from_dict(s)
            for s in data.get('snapshots', [])
        ]

        # Load version
        state._version = data.get('version', 0)

        # Load backward compatibility fields
        state._search_history = data.get('search_history', [])
        state._candidate_answers = data.get('candidate_answers', [])
        state._current_understanding = data.get('current_understanding', "")

        return state

    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeState':
        """
        Create state from JSON (synchronous).

        Args:
            json_str: JSON string representation

        Returns:
            KnowledgeState instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    async def from_json_async(cls, json_str: str) -> 'KnowledgeState':
        """
        Create state from JSON (asynchronous).

        Args:
            json_str: JSON string representation

        Returns:
            KnowledgeState instance
        """
        data = json.loads(json_str)
        return await cls.from_dict_async(data)

    # Property getters for read-only access to private attributes
    @property
    def facts(self) -> List[str]:
        """Get list of facts (read-only copy)."""
        with self._lock:
            return self._facts.copy()

    @property
    def uncertainties(self) -> List[str]:
        """Get list of uncertainties (read-only copy)."""
        with self._lock:
            return self._uncertainties.copy()

    @property
    def triples(self) -> List[KnowledgeTriple]:
        """Get list of knowledge triples (read-only copy)."""
        with self._lock:
            return self._triples.copy()

    @property
    def snapshots(self) -> List[StateSnapshot]:
        """Get list of state snapshots (read-only copy)."""
        with self._lock:
            return self._snapshots.copy()

    @property
    def version(self) -> int:
        """Get current state version."""
        with self._lock:
            return self._version

    def clear(self):
        """Clear all state (synchronous)."""
        with self._lock:
            self._triples.clear()
            self._facts.clear()
            self._uncertainties.clear()
            self._snapshots.clear()
            self._version = 0

            self._log("info", "State cleared")

    async def clear_async(self):
        """Clear all state (asynchronous)."""
        async with self._async_lock:
            self._triples.clear()
            self._facts.clear()
            self._uncertainties.clear()
            self._snapshots.clear()
            self._version = 0

            self._log("info", "State cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state statistics (synchronous).

        Returns:
            Dictionary with state metrics
        """
        with self._lock:
            return {
                "query": self.query,
                "triple_count": len(self._triples),
                "fact_count": len(self._facts),
                "uncertainty_count": len(self._uncertainties),
                "snapshot_count": len(self._snapshots),
                "version": self._version,
                "correlation_id": self._correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_statistics_async(self) -> Dict[str, Any]:
        """
        Get state statistics (asynchronous).

        Returns:
            Dictionary with state metrics
        """
        async with self._async_lock:
            return {
                "query": self.query,
                "triple_count": len(self._triples),
                "fact_count": len(self._facts),
                "uncertainty_count": len(self._uncertainties),
                "snapshot_count": len(self._snapshots),
                "version": self._version,
                "correlation_id": self._correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
