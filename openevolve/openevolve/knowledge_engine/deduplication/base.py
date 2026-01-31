"""
Base classes for deduplication strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a knowledge entity that can be deduplicated."""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation."""
    canonical_entities: List[Entity]
    duplicate_groups: List[List[Entity]]  # Groups of duplicates found
    stats: Dict[str, Any] = field(default_factory=dict)
    strategy_used: str = ""
    processing_time_ms: float = 0.0

    def __len__(self):
        return len(self.canonical_entities)


class DeduplicationStrategy(ABC):
    """
    Base class for all deduplication strategies.

    Each strategy must implement the deduplicate method which takes
    a list of entities and returns unique canonical entities.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    async def deduplicate(
        self,
        entities: List[Entity],
        context: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Deduplicate entities using this strategy.

        Args:
            entities: List of entities to deduplicate
            context: Optional context information

        Returns:
            DeduplicationResult with canonical entities and duplicate groups
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass

    async def preprocess_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Optional preprocessing step. Override in subclass if needed.
        """
        return entities

    async def postprocess_results(
        self,
        result: DeduplicationResult
    ) -> DeduplicationResult:
        """
        Optional postprocessing step. Override in subclass if needed.
        """
        return result

    def calculate_confidence(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> float:
        """
        Calculate confidence score for entity similarity.
        Override in subclass for strategy-specific logic.

        Returns:
            Float between 0.0 and 1.0
        """
        return 0.0
