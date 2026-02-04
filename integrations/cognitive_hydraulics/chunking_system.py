"""Chunking System - Convert successful resolutions to rules.

When ACT-R/evolutionary guess works, chunk it as new Soar rule.
Prevents future impasses on same problem type.

Key Concepts:
    - Chunk: Learned rule from impasse resolution
    - Generalization: Make chunk more broadly applicable
    - Validation: Test chunk correctness
    - Repository: Storage for learned chunks

Process:
    1. Detect impasse resolution
    2. Create specific chunk from resolution
    3. Validate chunk correctness
    4. Generalize chunk if valid
    5. Add to production memory
"""

import logging
import uuid
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timezone
from enum import Enum, auto
from copy import deepcopy

from .soar_engine import SoarRule, Impasse, ImpasseType, SoarProductionSystem
from .actr_engine import ACTRProduction

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks that can be learned."""
    OPERATOR_SELECTION = auto()  # Learn which operator to select
    CONSTRAINT_RESOLUTION = auto()  # Learn how to resolve constraints
    SUBGOAL_RESOLUTION = auto()  # Learn subgoal handling
    PATTERN_MATCH = auto()  # Learn pattern matching rules


class ChunkQuality(Enum):
    """Quality rating for chunks."""
    UNVALIDATED = auto()
    VALIDATED = auto()
    GENERALIZED = auto()
    PROVEN = auto()


@dataclass
class Chunk:
    """Learned rule from impasse resolution."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    chunk_type: ChunkType = ChunkType.OPERATOR_SELECTION
    quality: ChunkQuality = ChunkQuality.UNVALIDATED
    
    # Rule components
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Source information
    source_impasse_type: ImpasseType = ImpasseType.NO_CHANGE
    source_problem: str = ""
    creation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Usage statistics
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Generalization history
    parent_chunk_id: Optional[str] = None
    generalization_level: int = 0
    
    # Validation
    validation_tests: List[Dict] = field(default_factory=list)
    
    def to_soar_rule(self) -> SoarRule:
        """Convert chunk to Soar production rule."""
        return SoarRule(
            rule_id=self.chunk_id,
            name=self.name,
            conditions=deepcopy(self.conditions),
            actions=deepcopy(self.actions),
            learned=True,
            utility=self._calculate_utility()
        )
    
    def to_actr_production(self) -> ACTRProduction:
        """Convert chunk to ACT-R production."""
        return ACTRProduction(
            production_id=self.chunk_id,
            name=self.name,
            conditions=deepcopy(self.conditions),
            actions=deepcopy(self.actions),
            utility=self._calculate_utility(),
            probability=self.success_rate if self.usage_count > 0 else 0.5
        )
    
    def _calculate_utility(self) -> float:
        """Calculate utility based on success rate and usage."""
        if self.usage_count == 0:
            return 0.5
        
        base_utility = self.success_count / self.usage_count
        
        # Bonus for validation
        if self.quality == ChunkQuality.VALIDATED:
            base_utility += 0.1
        elif self.quality == ChunkQuality.GENERALIZED:
            base_utility += 0.15
        elif self.quality == ChunkQuality.PROVEN:
            base_utility += 0.2
        
        return min(1.0, base_utility)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def record_usage(self, success: bool):
        """Record a usage of this chunk."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "name": self.name,
            "chunk_type": self.chunk_type.name,
            "quality": self.quality.name,
            "conditions": self.conditions,
            "actions": self.actions,
            "source_impasse_type": self.source_impasse_type.name,
            "source_problem": self.source_problem,
            "created_at": self.creation_timestamp.isoformat(),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "parent_chunk_id": self.parent_chunk_id,
            "generalization_level": self.generalization_level,
        }


class ChunkRepository:
    """Storage for learned chunks."""
    
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.index_by_type: Dict[ChunkType, Set[str]] = {}
        self.index_by_impasse: Dict[ImpasseType, Set[str]] = {}
    
    def add(self, chunk: Chunk) -> str:
        """Add a chunk to the repository."""
        self.chunks[chunk.chunk_id] = chunk
        
        # Index by type
        if chunk.chunk_type not in self.index_by_type:
            self.index_by_type[chunk.chunk_type] = set()
        self.index_by_type[chunk.chunk_type].add(chunk.chunk_id)
        
        # Index by impasse type
        if chunk.source_impasse_type not in self.index_by_impasse:
            self.index_by_impasse[chunk.source_impasse_type] = set()
        self.index_by_impasse[chunk.source_impasse_type].add(chunk.chunk_id)
        
        logger.debug(f"Added chunk {chunk.name} (type: {chunk.chunk_type.name})")
        return chunk.chunk_id
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def find_by_type(self, chunk_type: ChunkType) -> List[Chunk]:
        """Find chunks by type."""
        ids = self.index_by_type.get(chunk_type, set())
        return [self.chunks[cid] for cid in ids if cid in self.chunks]
    
    def find_by_impasse(self, impasse_type: ImpasseType) -> List[Chunk]:
        """Find chunks for a specific impasse type."""
        ids = self.index_by_impasse.get(impasse_type, set())
        return [self.chunks[cid] for cid in ids if cid in self.chunks]
    
    def find_matching(self, context: Dict[str, Any]) -> List[Chunk]:
        """Find chunks that match the given context."""
        matches = []
        
        for chunk in self.chunks.values():
            if self._chunk_matches(chunk, context):
                matches.append(chunk)
        
        # Sort by utility
        matches.sort(key=lambda c: c._calculate_utility(), reverse=True)
        return matches
    
    def _chunk_matches(self, chunk: Chunk, context: Dict[str, Any]) -> bool:
        """Check if chunk conditions match context."""
        for condition in chunk.conditions:
            attr = condition.get("attribute") or condition.get("slot")
            value = condition.get("value")
            op = condition.get("operator", "equals")
            
            context_value = context.get(attr)
            
            if op == "equals":
                if context_value != value:
                    return False
            elif op == "exists":
                if context_value is None:
                    return False
            elif op == "in":
                if context_value not in value:
                    return False
        
        return True
    
    def remove(self, chunk_id: str):
        """Remove a chunk from the repository."""
        if chunk_id not in self.chunks:
            return
        
        chunk = self.chunks[chunk_id]
        
        # Remove from indices
        if chunk.chunk_type in self.index_by_type:
            self.index_by_type[chunk.chunk_type].discard(chunk_id)
        
        if chunk.source_impasse_type in self.index_by_impasse:
            self.index_by_impasse[chunk.source_impasse_type].discard(chunk_id)
        
        del self.chunks[chunk_id]
    
    def get_all(self) -> List[Chunk]:
        """Get all chunks."""
        return list(self.chunks.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            "total_chunks": len(self.chunks),
            "by_type": {t.name: len(ids) for t, ids in self.index_by_type.items()},
            "by_impasse": {i.name: len(ids) for i, ids in self.index_by_impasse.items()},
            "avg_success_rate": sum(c.success_rate for c in self.chunks.values()) / len(self.chunks) if self.chunks else 0.0
        }


class Generalizer:
    """Generalize specific solutions to broader rules."""
    
    def generalize(self, chunk: Chunk) -> Chunk:
        """Make chunk more general."""
        generalized_conditions = []
        
        for condition in chunk.conditions:
            generalized = self._generalize_condition(condition)
            if generalized:
                generalized_conditions.append(generalized)
        
        generalized_actions = []
        for action in chunk.actions:
            generalized = self._generalize_action(action)
            if generalized:
                generalized_actions.append(generalized)
        
        generalized = Chunk(
            name=f"generalized_{chunk.name}",
            chunk_type=chunk.chunk_type,
            quality=ChunkQuality.GENERALIZED,
            conditions=generalized_conditions,
            actions=generalized_actions,
            source_impasse_type=chunk.source_impasse_type,
            parent_chunk_id=chunk.chunk_id,
            generalization_level=chunk.generalization_level + 1
        )
        
        logger.debug(f"Generalized chunk {chunk.name} -> {generalized.name}")
        return generalized
    
    def _generalize_condition(self, condition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generalize a single condition."""
        generalized = dict(condition)
        value = condition.get("value")
        
        # Generalize specific values
        if isinstance(value, str):
            # Remove UUIDs
            if len(value) == 36 and "-" in value:
                return None  # Remove UUID conditions
            
            # Generalize timestamps
            if "T" in value and ":" in value:
                # Likely ISO timestamp
                generalized["operator"] = "exists"
                del generalized["value"]
        
        elif isinstance(value, (int, float)):
            # Replace specific numbers with range
            if "range" not in generalized:
                generalized["operator"] = "exists"
                del generalized["value"]
        
        return generalized
    
    def _generalize_action(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generalize a single action."""
        # Actions are usually kept as-is, but specific values may be abstracted
        return action


class ChunkingEngine:
    """
    Main Chunking System.
    
    Learns rules from successful impasse resolutions.
    """
    
    def __init__(
        self,
        production_system: Optional[SoarProductionSystem] = None
    ):
        self.repository = ChunkRepository()
        self.generalizer = Generalizer()
        self.production_system = production_system
        
        # Configuration
        self.validation_threshold = 0.8  # Success rate needed for validation
        self.auto_generalize = True
        
        # Stats
        self.chunks_created = 0
        self.chunks_validated = 0
        self.chunks_generalized = 0
    
    def create_chunk(
        self,
        impasse: Impasse,
        resolution: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Chunk:
        """Create a new chunk from a successful resolution."""
        # Extract conditions from impasse context
        conditions = self._extract_conditions(impasse, context)
        
        # Extract actions from resolution
        actions = self._extract_actions(resolution)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(impasse)
        
        # Create chunk
        chunk = Chunk(
            name=f"chunk_{impasse.impasse_type.name.lower()}_{self.chunks_created}",
            chunk_type=chunk_type,
            conditions=conditions,
            actions=actions,
            source_impasse_type=impasse.impasse_type,
            source_problem=context.get("problem_description", ""),
            quality=ChunkQuality.UNVALIDATED
        )
        
        # Add to repository
        self.repository.add(chunk)
        self.chunks_created += 1
        
        # Add to production system
        if self.production_system:
            self.add_to_production_memory(chunk)
        
        logger.info(f"Created chunk: {chunk.name}")
        return chunk
    
    def _extract_conditions(
        self,
        impasse: Impasse,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract conditions from impasse context."""
        conditions = []
        
        # Add impasse type as condition
        conditions.append({
            "attribute": "impasse_type",
            "operator": "equals",
            "value": impasse.impasse_type.name
        })
        
        # Add relevant context conditions
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append({
                    "attribute": key,
                    "operator": "equals",
                    "value": value
                })
        
        return conditions
    
    def _extract_actions(self, resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actions from resolution."""
        actions = []
        
        if "operator_id" in resolution:
            actions.append({
                "type": "propose",
                "operator_id": resolution["operator_id"]
            })
        
        if "value" in resolution:
            actions.append({
                "type": "add",
                "attribute": "resolution_value",
                "value": resolution["value"]
            })
        
        if "action" in resolution:
            actions.append(resolution["action"])
        
        return actions
    
    def _determine_chunk_type(self, impasse: Impasse) -> ChunkType:
        """Determine the type of chunk based on impasse."""
        if impasse.impasse_type == ImpasseType.TIE:
            return ChunkType.OPERATOR_SELECTION
        elif impasse.impasse_type == ImpasseType.CONSTRAINT_FAILURE:
            return ChunkType.CONSTRAINT_RESOLUTION
        elif impasse.impasse_type == ImpasseType.NO_CHANGE:
            return ChunkType.SUBGOAL_RESOLUTION
        else:
            return ChunkType.PATTERN_MATCH
    
    def generalize_chunk(self, chunk: Chunk) -> Optional[Chunk]:
        """Make chunk more general."""
        if chunk.quality != ChunkQuality.VALIDATED:
            logger.warning(f"Cannot generalize unvalidated chunk: {chunk.name}")
            return None
        
        generalized = self.generalizer.generalize(chunk)
        generalized.quality = ChunkQuality.GENERALIZED
        
        self.repository.add(generalized)
        self.chunks_generalized += 1
        
        # Add generalized version to production system
        if self.production_system:
            self.add_to_production_memory(generalized)
        
        logger.info(f"Generalized chunk: {chunk.name} -> {generalized.name}")
        return generalized
    
    def add_to_production_memory(self, chunk: Chunk):
        """Add chunk to Soar production memory."""
        if not self.production_system:
            return
        
        rule = chunk.to_soar_rule()
        self.production_system.add_rule(rule)
        
        logger.debug(f"Added chunk {chunk.name} to production memory")
    
    def match_chunk(self, state: Dict[str, Any]) -> List[Chunk]:
        """Find applicable chunks for a state."""
        return self.repository.find_matching(state)
    
    def validate_chunk(
        self,
        chunk: Chunk,
        test_cases: List[Dict[str, Any]],
        validator: Optional[Callable[[Chunk, Dict[str, Any]], bool]] = None
    ) -> bool:
        """Test chunk correctness."""
        if not test_cases:
            # Cannot validate without test cases
            return False
        
        passed = 0
        
        for test_case in test_cases:
            if validator:
                success = validator(chunk, test_case)
            else:
                success = self._default_validate(chunk, test_case)
            
            if success:
                passed += 1
            
            chunk.validation_tests.append({
                "test_case": test_case,
                "passed": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        success_rate = passed / len(test_cases)
        
        if success_rate >= self.validation_threshold:
            chunk.quality = ChunkQuality.VALIDATED
            self.chunks_validated += 1
            logger.info(f"Validated chunk {chunk.name}: {success_rate:.2%}")
            
            # Auto-generalize if enabled
            if self.auto_generalize:
                self.generalize_chunk(chunk)
            
            return True
        
        logger.warning(f"Chunk {chunk.name} validation failed: {success_rate:.2%}")
        return False
    
    def _default_validate(self, chunk: Chunk, test_case: Dict[str, Any]) -> bool:
        """Default validation: check if conditions match."""
        for condition in chunk.conditions:
            attr = condition.get("attribute")
            value = condition.get("value")
            op = condition.get("operator", "equals")
            
            test_value = test_case.get(attr)
            
            if op == "equals":
                if test_value != value:
                    return False
            elif op == "exists":
                if test_value is None:
                    return False
        
        return True
    
    def record_chunk_usage(self, chunk_id: str, success: bool):
        """Record usage of a chunk."""
        chunk = self.repository.get(chunk_id)
        if chunk:
            chunk.record_usage(success)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "chunks_created": self.chunks_created,
            "chunks_validated": self.chunks_validated,
            "chunks_generalized": self.chunks_generalized,
            "repository_stats": self.repository.get_stats()
        }
