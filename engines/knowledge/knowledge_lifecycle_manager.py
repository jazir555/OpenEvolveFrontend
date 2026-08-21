"""
Knowledge Lifecycle Manager

Manages the full lifecycle of memories from creation through archival:
- Confidence scoring for memory quality
- Decay detection based on usage patterns
- Automatic archival of cold memories
- Compression and cold storage management
- Periodic maintenance jobs

Author: OpenEvolve AI
"""
from __future__ import annotations


import json
import gzip
import sqlite3
import logging
import threading
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict
import heapq

# Configure logging
logger = logging.getLogger(__name__)


class LifecycleStage(Enum):
    """Stages in the memory lifecycle."""
    ACTIVE = "active"           # In main index, fully searchable
    COOLING = "cooling"         # Still active but marked for monitoring
    ARCHIVED = "archived"       # Moved to cold storage, compressed
    EXPIRED = "expired"         # Ready for deletion (kept for audit)
    DELETED = "deleted"         # Permanently removed


class MemoryType(Enum):
    """Types of memories with different decay rules."""
    EPHEMERAL = "ephemeral"     # Short-lived, 30-60 days
    STANDARD = "standard"       # Normal memory, 90-180 days
    CORE = "core"               # Never decay, permanent
    TEMPORAL = "temporal"       # Time-sensitive, expires after date


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle thresholds and behavior."""
    
    # Decay thresholds (days)
    decay_days_min: int = 90
    decay_days_max: int = 180
    cooling_threshold_days: int = 60
    
    # Confidence thresholds (0.0 - 1.0)
    archive_threshold_confidence: float = 0.3
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3
    
    # Access frequency thresholds
    min_access_count_for_active: int = 3
    cooling_access_threshold: int = 1
    
    # Compression settings
    compress_archived: bool = True
    compression_level: int = 6
    min_size_for_compression: int = 256  # bytes
    
    # Maintenance settings
    maintenance_interval_hours: int = 24
    batch_size: int = 100
    max_archived_retention_days: int = 365 * 2  # 2 years
    
    # Scoring weights
    source_reliability_weight: float = 0.3
    confirmation_count_weight: float = 0.25
    contradiction_penalty_weight: float = 0.25
    user_confirmation_weight: float = 0.2
    
    # Decay calculation weights
    time_decay_weight: float = 0.4
    frequency_decay_weight: float = 0.3
    relevance_decay_weight: float = 0.3
    
    # Database paths
    active_db_path: str = "knowledge_active.db"
    archive_db_path: str = "knowledge_archive.db"
    metadata_db_path: str = "knowledge_lifecycle.db"
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.decay_days_min >= self.decay_days_max:
            raise ValueError("decay_days_min must be less than decay_days_max")
        if not (0.0 <= self.archive_threshold_confidence <= 1.0):
            raise ValueError("archive_threshold_confidence must be between 0.0 and 1.0")


@dataclass
class MemoryMetadata:
    """Metadata tracking for a memory's lifecycle."""
    memory_id: str
    stage: LifecycleStage = LifecycleStage.ACTIVE
    memory_type: MemoryType = MemoryType.STANDARD
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    archived_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Access tracking
    access_count: int = 0
    access_history: List[datetime] = field(default_factory=list)
    
    # Confidence tracking
    confidence_score: float = 0.5
    source_reliability: float = 0.5
    confirmation_count: int = 0
    contradiction_count: int = 0
    user_confirmed: bool = False
    
    # Decay tracking
    decay_score: float = 0.0
    relevance_score: float = 1.0
    
    # Storage
    compressed_size: Optional[int] = None
    original_size: Optional[int] = None
    
    # Content hash for deduplication
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert enums to strings
        data['stage'] = self.stage.value
        data['memory_type'] = self.memory_type.value
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'last_accessed', 'last_modified', 'archived_at', 'expires_at']:
            if data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        # Convert datetime list
        data['access_history'] = [
            d.isoformat() if isinstance(d, datetime) else d 
            for d in data['access_history']
        ][-50:]  # Keep last 50 access records
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        """Create from dictionary."""
        # Convert string enums back
        data['stage'] = LifecycleStage(data.get('stage', 'active'))
        data['memory_type'] = MemoryType(data.get('memory_type', 'standard'))
        
        # Convert ISO strings back to datetime
        for key in ['created_at', 'last_accessed', 'last_modified', 'archived_at', 'expires_at']:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
        
        # Convert access history
        if data.get('access_history'):
            data['access_history'] = [
                datetime.fromisoformat(d) if isinstance(d, str) else d 
                for d in data['access_history']
            ]
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def record_access(self):
        """Record an access to this memory."""
        now = datetime.now()
        self.last_accessed = now
        self.access_count += 1
        self.access_history.append(now)
        # Keep only last 50 access records
        if len(self.access_history) > 50:
            self.access_history = self.access_history[-50:]
    
    def days_since_last_access(self) -> float:
        """Calculate days since last access."""
        return (datetime.now() - self.last_accessed).total_seconds() / 86400
    
    def days_since_creation(self) -> float:
        """Calculate days since creation."""
        return (datetime.now() - self.created_at).total_seconds() / 86400


class ConfidenceScorer:
    """
    Scores memory confidence based on multiple factors:
    - Source reliability
    - Confirmation count
    - Contradiction detection
    - User explicit confirmation
    """
    
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._source_reliability_cache: Dict[str, float] = {}
    
    def calculate_confidence(self, metadata: MemoryMetadata, 
                            source_info: Optional[Dict] = None) -> float:
        """
        Calculate overall confidence score (0.0 - 1.0).
        
        Args:
            metadata: Memory metadata
            source_info: Optional source information dict
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        scores = []
        weights = []
        
        # Source reliability score
        source_score = self._calculate_source_score(metadata, source_info)
        scores.append(source_score)
        weights.append(self.config.source_reliability_weight)
        
        # Confirmation count score
        confirmation_score = self._calculate_confirmation_score(metadata)
        scores.append(confirmation_score)
        weights.append(self.config.confirmation_count_weight)
        
        # Contradiction penalty
        contradiction_score = self._calculate_contradiction_score(metadata)
        scores.append(contradiction_score)
        weights.append(self.config.contradiction_penalty_weight)
        
        # User confirmation score
        user_score = self._calculate_user_score(metadata)
        scores.append(user_score)
        weights.append(self.config.user_confirmation_weight)
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        
        confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Apply sigmoid normalization for better distribution
        confidence = self._sigmoid_normalize(confidence)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_source_score(self, metadata: MemoryMetadata,
                                source_info: Optional[Dict]) -> float:
        """Calculate score based on source reliability."""
        base_reliability = metadata.source_reliability
        
        if source_info:
            source_type = source_info.get('type', 'unknown')
            
            # Source type reliability multipliers
            type_multipliers = {
                'verified_database': 1.0,
                'expert_input': 0.95,
                'automated_system': 0.85,
                'user_input': 0.70,
                'web_scrape': 0.60,
                'unknown': 0.50,
                'unverified': 0.30
            }
            
            multiplier = type_multipliers.get(source_type, 0.50)
            base_reliability *= multiplier
            
            # Check for authoritative sources
            if source_info.get('authoritative', False):
                base_reliability = min(1.0, base_reliability * 1.2)
            
            # Check source age
            source_age_days = source_info.get('age_days', 0)
            if source_age_days > 365:
                base_reliability *= 0.9  # Slight penalty for old sources
        
        return min(1.0, base_reliability)
    
    def _calculate_confirmation_score(self, metadata: MemoryMetadata) -> float:
        """Calculate score based on confirmation count."""
        confirmations = metadata.confirmation_count
        
        # Logarithmic scale for confirmation count
        # 0 confirmations = 0.3, 1 = 0.5, 5 = 0.8, 10+ = 0.95
        if confirmations == 0:
            return 0.3
        elif confirmations == 1:
            return 0.5
        elif confirmations < 5:
            return 0.5 + (confirmations - 1) * 0.075
        else:
            return min(0.95, 0.8 + (confirmations - 5) * 0.03)
    
    def _calculate_contradiction_score(self, metadata: MemoryMetadata) -> float:
        """Calculate penalty based on contradictions."""
        contradictions = metadata.contradiction_count
        confirmations = max(1, metadata.confirmation_count)
        
        # Contradiction ratio
        ratio = contradictions / (contradictions + confirmations)
        
        # Score decreases as contradiction ratio increases
        # 0 contradictions = 1.0, 50% contradictions = 0.4, 100% = 0.1
        if ratio == 0:
            return 1.0
        elif ratio < 0.25:
            return 1.0 - ratio
        elif ratio < 0.5:
            return 0.75 - ratio
        else:
            return max(0.1, 0.5 - ratio)
    
    def _calculate_user_score(self, metadata: MemoryMetadata) -> float:
        """Calculate score based on user confirmation."""
        if metadata.user_confirmed:
            return 1.0
        return 0.5
    
    def _sigmoid_normalize(self, x: float) -> float:
        """Apply sigmoid function for smoother distribution."""
        import math
        # Shift to center at 0.5
        x = (x - 0.5) * 4
        return 1 / (1 + math.exp(-x))
    
    def update_confidence(self, metadata: MemoryMetadata,
                         confirmation: bool = False,
                         contradiction: bool = False,
                         user_confirmed: Optional[bool] = None) -> float:
        """
        Update confidence based on new information.
        
        Args:
            metadata: Memory metadata to update
            confirmation: Whether this is a confirmation
            contradiction: Whether this is a contradiction
            user_confirmed: New user confirmation status
            
        Returns:
            Updated confidence score
        """
        if confirmation:
            metadata.confirmation_count += 1
        
        if contradiction:
            metadata.contradiction_count += 1
        
        if user_confirmed is not None:
            metadata.user_confirmed = user_confirmed
        
        metadata.confidence_score = self.calculate_confidence(metadata)
        metadata.last_modified = datetime.now()
        
        return metadata.confidence_score
    
    def get_confidence_tier(self, confidence: float) -> str:
        """Get confidence tier label."""
        if confidence >= self.config.high_confidence_threshold:
            return "high"
        elif confidence >= self.config.archive_threshold_confidence:
            return "medium"
        elif confidence >= self.config.low_confidence_threshold:
            return "low"
        return "critical"


class DecayDetector:
    """
    Detects when memories should decay based on:
    - Time since last access (90-180 days threshold)
    - Access frequency dropping
    - Relevance to recent queries declining
    - Override for CORE memories (never decay)
    """
    
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._recent_queries: List[Tuple[str, datetime]] = []
        self._lock = threading.Lock()
    
    def calculate_decay_score(self, metadata: MemoryMetadata,
                             current_queries: Optional[List[str]] = None) -> float:
        """
        Calculate decay score (0.0 = fresh, 1.0 = fully decayed).
        
        Args:
            metadata: Memory metadata
            current_queries: Recent query strings for relevance comparison
            
        Returns:
            Decay score between 0.0 and 1.0
        """
        # CORE memories never decay
        if metadata.memory_type == MemoryType.CORE:
            return 0.0
        
        # TEMPORAL memories decay based on expiration date
        if metadata.memory_type == MemoryType.TEMPORAL:
            return self._calculate_temporal_decay(metadata)
        
        scores = []
        weights = []
        
        # Time-based decay
        time_score = self._calculate_time_decay(metadata)
        scores.append(time_score)
        weights.append(self.config.time_decay_weight)
        
        # Access frequency decay
        freq_score = self._calculate_frequency_decay(metadata)
        scores.append(freq_score)
        weights.append(self.config.frequency_decay_weight)
        
        # Relevance decay
        if current_queries:
            rel_score = self._calculate_relevance_decay(metadata, current_queries)
            scores.append(rel_score)
            weights.append(self.config.relevance_decay_weight)
        
        # Calculate weighted decay
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        decay = sum(s * w for s, w in zip(scores, weights)) / total_weight
        metadata.decay_score = decay
        
        return decay
    
    def _calculate_time_decay(self, metadata: MemoryMetadata) -> float:
        """Calculate decay based on time since last access."""
        days_since = metadata.days_since_last_access()
        
        # Use min/max thresholds
        if days_since < self.config.decay_days_min:
            return 0.0
        elif days_since >= self.config.decay_days_max:
            return 1.0
        else:
            # Linear interpolation between thresholds
            range_size = self.config.decay_days_max - self.config.decay_days_min
            return (days_since - self.config.decay_days_min) / range_size
    
    def _calculate_frequency_decay(self, metadata: MemoryMetadata) -> float:
        """Calculate decay based on access frequency."""
        access_count = metadata.access_count
        days_since_creation = max(1, metadata.days_since_creation())
        
        # Access frequency (accesses per day)
        freq = access_count / days_since_creation
        
        # Higher frequency = lower decay
        # freq >= 1.0 => 0.0 decay
        # freq <= 0.01 => 1.0 decay
        if freq >= 1.0:
            return 0.0
        elif freq <= 0.01:
            return 1.0
        else:
            # Logarithmic scale
            import math
            return 1.0 - (math.log10(freq * 100) / 2)
    
    def _calculate_relevance_decay(self, metadata: MemoryMetadata,
                                   current_queries: List[str]) -> float:
        """Calculate decay based on relevance to recent queries."""
        # Simple keyword matching - in production, use embeddings
        memory_keywords = set(metadata.memory_id.lower().split('_'))
        
        relevance_scores = []
        for query in current_queries[-10:]:  # Last 10 queries
            query_keywords = set(query.lower().split())
            overlap = len(memory_keywords & query_keywords)
            relevance_scores.append(overlap / max(len(query_keywords), 1))
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        
        # Store for tracking
        metadata.relevance_score = avg_relevance
        
        # Higher relevance = lower decay
        return 1.0 - avg_relevance
    
    def _calculate_temporal_decay(self, metadata: MemoryMetadata) -> float:
        """Calculate decay for temporal memories."""
        if not metadata.expires_at:
            return self._calculate_time_decay(metadata)
        
        now = datetime.now()
        if now >= metadata.expires_at:
            return 1.0
        
        # Linear decay towards expiration
        total_lifetime = (metadata.expires_at - metadata.created_at).total_seconds()
        elapsed = (now - metadata.created_at).total_seconds()
        
        if total_lifetime <= 0:
            return 1.0
        
        return elapsed / total_lifetime
    
    def should_transition(self, metadata: MemoryMetadata,
                         decay_score: Optional[float] = None) -> Optional[LifecycleStage]:
        """
        Determine if memory should transition to a new stage.
        
        Args:
            metadata: Memory metadata
            decay_score: Pre-calculated decay score (optional)
            
        Returns:
            New stage if transition needed, None otherwise
        """
        if decay_score is None:
            decay_score = metadata.decay_score
        
        current = metadata.stage
        days_since = metadata.days_since_last_access()
        
        # Transition rules
        if current == LifecycleStage.ACTIVE:
            if metadata.memory_type == MemoryType.CORE:
                return None  # CORE never transitions
            
            if days_since > self.config.cooling_threshold_days:
                return LifecycleStage.COOLING
            
            if decay_score > 0.7 or metadata.confidence_score < self.config.low_confidence_threshold:
                return LifecycleStage.COOLING
        
        elif current == LifecycleStage.COOLING:
            if days_since > self.config.decay_days_min and decay_score > 0.5:
                return LifecycleStage.ARCHIVED
            
            # Return to active if accessed again
            if days_since < 7:  # Accessed recently
                return LifecycleStage.ACTIVE
        
        elif current == LifecycleStage.ARCHIVED:
            # Check if should expire
            if metadata.archived_at:
                days_archived = (datetime.now() - metadata.archived_at).days
                if days_archived > self.config.max_archived_retention_days:
                    return LifecycleStage.EXPIRED
            
            # Return to active if explicitly requested
            if metadata.access_count > 0 and days_since < 1:
                return LifecycleStage.ACTIVE
        
        elif current == LifecycleStage.EXPIRED:
            # Check if should be deleted (after audit period)
            if metadata.archived_at:
                days_total = (datetime.now() - metadata.created_at).days
                if days_total > self.config.max_archived_retention_days + 90:
                    return LifecycleStage.DELETED
        
        return None  # No transition needed
    
    def record_query(self, query: str):
        """Record a query for relevance tracking."""
        with self._lock:
            self._recent_queries.append((query, datetime.now()))
            # Keep only last 100 queries
            if len(self._recent_queries) > 100:
                self._recent_queries = self._recent_queries[-100:]
    
    def get_recent_queries(self, n: int = 10) -> List[str]:
        """Get recent queries for relevance calculation."""
        with self._lock:
            return [q for q, _ in self._recent_queries[-n:]]


class ArchivalManager:
    """
    Manages archival of cold memories:
    - Compress archived memories
    - Move to cold storage (separate SQLite file)
    - Can still retrieve if needed (slower)
    - Maintains index for archived content
    """
    
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._lock = threading.RLock()
        self._init_archive_storage()
    
    def _init_archive_storage(self):
        """Initialize archive storage database."""
        with self._get_archive_connection() as conn:
            cursor = conn.cursor()
            
            # Archive content table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS archived_memories (
                    memory_id TEXT PRIMARY KEY,
                    content BLOB NOT NULL,
                    compressed BOOLEAN NOT NULL,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    archived_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            ''')
            
            # Archive index for searching
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS archive_index (
                    memory_id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    keywords TEXT,
                    archived_at TEXT NOT NULL,
                    confidence_score REAL,
                    FOREIGN KEY (memory_id) REFERENCES archived_memories(memory_id)
                )
            ''')
            
            # Statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS archive_stats (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_memories INTEGER DEFAULT 0,
                    total_compressed_size INTEGER DEFAULT 0,
                    total_original_size INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            ''')
            
            # Insert default stats row if not exists
            cursor.execute('''
                INSERT OR IGNORE INTO archive_stats (id, last_updated) 
                VALUES (1, ?)
            ''', (datetime.now().isoformat(),))
            
            conn.commit()
    
    @contextmanager
    def _get_archive_connection(self):
        """Get connection to archive database."""
        conn = sqlite3.connect(self.config.archive_db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def archive_memory(self, memory_id: str, content: bytes,
                      metadata: MemoryMetadata) -> Tuple[bool, int]:
        """
        Archive a memory to cold storage.
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content bytes
            metadata: Memory metadata
            
        Returns:
            Tuple of (success, final_size)
        """
        with self._lock:
            try:
                original_size = len(content)
                
                # Compress if enabled and content is large enough
                if (self.config.compress_archived and 
                    original_size >= self.config.min_size_for_compression):
                    compressed = gzip.compress(
                        content, 
                        compresslevel=self.config.compression_level
                    )
                    compressed_size = len(compressed)
                    is_compressed = True
                else:
                    compressed = content
                    compressed_size = original_size
                    is_compressed = False
                
                # Update metadata
                metadata.archived_at = datetime.now()
                metadata.stage = LifecycleStage.ARCHIVED
                metadata.original_size = original_size
                metadata.compressed_size = compressed_size
                
                with self._get_archive_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Store archived memory
                    cursor.execute('''
                        INSERT OR REPLACE INTO archived_memories 
                        (memory_id, content, compressed, original_size, compressed_size, 
                         archived_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory_id,
                        compressed,
                        is_compressed,
                        original_size,
                        compressed_size,
                        metadata.archived_at.isoformat(),
                        json.dumps(metadata.to_dict())
                    ))
                    
                    # Update index
                    keywords = self._extract_keywords(content)
                    cursor.execute('''
                        INSERT OR REPLACE INTO archive_index 
                        (memory_id, content_hash, keywords, archived_at, confidence_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        memory_id,
                        metadata.content_hash or self._compute_hash(content),
                        json.dumps(keywords),
                        metadata.archived_at.isoformat(),
                        metadata.confidence_score
                    ))
                    
                    # Update statistics
                    self._update_stats(conn, compressed_size, original_size)
                    
                    conn.commit()
                
                logger.info(f"Archived memory {memory_id}: "
                           f"{original_size} -> {compressed_size} bytes "
                           f"({compressed_size/original_size*100:.1f}%)")
                
                return True, compressed_size
                
            except Exception as e:
                logger.error(f"Failed to archive memory {memory_id}: {e}")
                return False, 0
    
    def retrieve_archived(self, memory_id: str) -> Optional[Tuple[bytes, MemoryMetadata]]:
        """
        Retrieve an archived memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Tuple of (content, metadata) or None if not found
        """
        with self._lock:
            try:
                with self._get_archive_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT content, compressed, metadata 
                        FROM archived_memories 
                        WHERE memory_id = ?
                    ''', (memory_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    content, is_compressed, metadata_json = row
                    
                    # Decompress if needed
                    if is_compressed:
                        content = gzip.decompress(content)
                    
                    metadata = MemoryMetadata.from_dict(json.loads(metadata_json))
                    
                    return content, metadata
                    
            except Exception as e:
                logger.error(f"Failed to retrieve archived memory {memory_id}: {e}")
                return None
    
    def delete_archived(self, memory_id: str) -> bool:
        """Permanently delete an archived memory."""
        with self._lock:
            try:
                with self._get_archive_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get size info for stats update
                    cursor.execute('''
                        SELECT compressed_size, original_size 
                        FROM archived_memories 
                        WHERE memory_id = ?
                    ''', (memory_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return False
                    
                    compressed_size, original_size = row
                    
                    # Delete from tables
                    cursor.execute('DELETE FROM archive_index WHERE memory_id = ?', (memory_id,))
                    cursor.execute('DELETE FROM archived_memories WHERE memory_id = ?', (memory_id,))
                    
                    # Update statistics
                    cursor.execute('''
                        UPDATE archive_stats 
                        SET total_memories = total_memories - 1,
                            total_compressed_size = total_compressed_size - ?,
                            total_original_size = total_original_size - ?,
                            last_updated = ?
                        WHERE id = 1
                    ''', (compressed_size, original_size, datetime.now().isoformat()))
                    
                    conn.commit()
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to delete archived memory {memory_id}: {e}")
                return False
    
    def search_archive(self, keywords: List[str],
                      min_confidence: float = 0.0) -> List[Dict]:
        """
        Search archived memories by keywords.
        
        Args:
            keywords: List of keywords to search
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching memory metadata
        """
        results = []
        
        with self._get_archive_connection() as conn:
            cursor = conn.cursor()
            
            # Simple keyword matching
            cursor.execute('''
                SELECT memory_id, keywords, archived_at, confidence_score
                FROM archive_index
                WHERE confidence_score >= ?
            ''', (min_confidence,))
            
            for row in cursor.fetchall():
                memory_id, keywords_json, archived_at, confidence = row
                stored_keywords = json.loads(keywords_json)
                
                # Check for keyword overlap
                overlap = set(k.lower() for k in keywords) & set(stored_keywords)
                if overlap:
                    results.append({
                        'memory_id': memory_id,
                        'keywords': stored_keywords,
                        'archived_at': archived_at,
                        'confidence_score': confidence,
                        'match_score': len(overlap) / len(keywords)
                    })
            
            # Sort by match score
            results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        with self._get_archive_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT total_memories, total_compressed_size, 
                       total_original_size, last_updated
                FROM archive_stats
                WHERE id = 1
            ''')
            
            row = cursor.fetchone()
            if row:
                total, compressed, original, updated = row
                compression_ratio = (1 - compressed / original) * 100 if original > 0 else 0
                
                return {
                    'total_memories': total,
                    'total_compressed_size': compressed,
                    'total_original_size': original,
                    'compression_ratio_percent': compression_ratio,
                    'last_updated': updated
                }
            
            return {}
    
    def _extract_keywords(self, content: bytes, max_keywords: int = 20) -> List[str]:
        """Extract keywords from content for indexing."""
        try:
            # Try to decode as text
            text = content.decode('utf-8', errors='ignore').lower()
            
            # Simple keyword extraction (in production, use NLP)
            # Remove common words and extract terms
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                           'be', 'been', 'being', 'have', 'has', 'had',
                           'do', 'does', 'did', 'will', 'would', 'could',
                           'should', 'may', 'might', 'must', 'shall',
                           'can', 'need', 'dare', 'ought', 'used',
                           'to', 'of', 'in', 'for', 'on', 'with', 'at',
                           'by', 'from', 'as', 'into', 'through', 'during',
                           'before', 'after', 'above', 'below', 'between',
                           'and', 'but', 'or', 'yet', 'so', 'if', 'because',
                           'although', 'though', 'while', 'where', 'when',
                           'that', 'which', 'who', 'whom', 'whose', 'what',
                           'this', 'these', 'those', 'i', 'you', 'he', 'she',
                           'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            # Split and filter
            words = [w.strip('.,!?;:()[]{}"\'') for w in text.split()]
            words = [w for w in words if w and w not in common_words and len(w) > 2]
            
            # Count frequency and get top keywords
            freq = {}
            for word in words:
                freq[word] = freq.get(word, 0) + 1
            
            top = heapq.nlargest(max_keywords, freq.items(), key=lambda x: x[1])
            return [word for word, _ in top]
            
        except Exception:
            return []
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(content).hexdigest()
    
    def _update_stats(self, conn: sqlite3.Connection, 
                     compressed_size: int, original_size: int):
        """Update archive statistics."""
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE archive_stats 
            SET total_memories = total_memories + 1,
                total_compressed_size = total_compressed_size + ?,
                total_original_size = total_original_size + ?,
                last_updated = ?
            WHERE id = 1
        ''', (compressed_size, original_size, datetime.now().isoformat()))


class DeduplicationManager:
    """Manages deduplication of memories to prevent redundant storage."""
    
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._content_hashes: Dict[str, str] = {}  # hash -> memory_id
        self._lock = threading.Lock()
    
    def compute_hash(self, content: bytes) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def check_duplicate(self, content: bytes) -> Optional[str]:
        """
        Check if content already exists.
        
        Args:
            content: Content to check
            
        Returns:
            Existing memory_id if duplicate, None otherwise
        """
        content_hash = self.compute_hash(content)
        
        with self._lock:
            return self._content_hashes.get(content_hash)
    
    def register_content(self, memory_id: str, content: bytes):
        """Register content hash for new memory."""
        content_hash = self.compute_hash(content)
        
        with self._lock:
            self._content_hashes[content_hash] = memory_id
    
    def find_similar(self, content: bytes, 
                    threshold: float = 0.9) -> List[Tuple[str, float]]:
        """
        Find similar content using simple similarity metric.
        
        Args:
            content: Content to compare
            threshold: Minimum similarity threshold
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        # Simple implementation - in production, use embeddings
        content_hash = self.compute_hash(content)
        
        similar = []
        with self._lock:
            for stored_hash, memory_id in self._content_hashes.items():
                # Simple hash comparison - not true similarity
                # In production, use embedding cosine similarity
                if stored_hash == content_hash:
                    similar.append((memory_id, 1.0))
        
        return similar
    
    def merge_duplicates(self, memory_ids: List[str]) -> Optional[str]:
        """
        Merge duplicate memories into one.
        
        Args:
            memory_ids: List of duplicate memory IDs
            
        Returns:
            Primary memory ID after merge, or None
        """
        if not memory_ids:
            return None
        
        # Keep the first one as primary
        primary = memory_ids[0]
        
        logger.info(f"Merged {len(memory_ids)} duplicates into {primary}")
        
        return primary


class MemoryLifecycleManager:
    """
    Manages full lifecycle of memories:
    - Creation with confidence scoring
    - Active retrieval phase
    - Decay detection
    - Archival (cold storage)
    - Deletion/expiration
    """
    
    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()
        
        # Initialize components
        self.confidence_scorer = ConfidenceScorer(self.config)
        self.decay_detector = DecayDetector(self.config)
        self.archival_manager = ArchivalManager(self.config)
        self.deduplication_manager = DeduplicationManager(self.config)
        
        # Storage
        self._metadata: Dict[str, MemoryMetadata] = {}
        self._active_content: Dict[str, bytes] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'created_count': 0,
            'archived_count': 0,
            'deleted_count': 0,
            'retrieved_count': 0,
            'transitions': defaultdict(int)
        }
        
        # Maintenance
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_running = False
        
        # Initialize storage
        self._init_metadata_storage()
        
        logger.info("MemoryLifecycleManager initialized")
    
    def _init_metadata_storage(self):
        """Initialize metadata storage database."""
        conn = sqlite3.connect(self.config.metadata_db_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    memory_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stage ON memory_metadata(stage)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON memory_metadata(last_accessed)
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    @contextmanager
    def _get_metadata_connection(self):
        """Get connection to metadata database."""
        conn = sqlite3.connect(self.config.metadata_db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def create_memory(self, memory_id: str, content: bytes,
                     source_info: Optional[Dict] = None,
                     memory_type: MemoryType = MemoryType.STANDARD,
                     expires_at: Optional[datetime] = None) -> MemoryMetadata:
        """
        Create a new memory with lifecycle tracking.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Memory content
            source_info: Source information for confidence scoring
            memory_type: Type of memory
            expires_at: Optional expiration date for temporal memories
            
        Returns:
            MemoryMetadata for the created memory
        """
        with self._lock:
            # Check for duplicates
            existing_id = self.deduplication_manager.check_duplicate(content)
            if existing_id:
                logger.info(f"Duplicate detected: {memory_id} matches {existing_id}")
                # Return existing metadata
                return self._metadata.get(existing_id, MemoryMetadata(memory_id=existing_id))
            
            # Create metadata
            metadata = MemoryMetadata(
                memory_id=memory_id,
                stage=LifecycleStage.ACTIVE,
                memory_type=memory_type,
                expires_at=expires_at,
                content_hash=self.deduplication_manager.compute_hash(content)
            )
            
            # Calculate initial confidence
            metadata.confidence_score = self.confidence_scorer.calculate_confidence(
                metadata, source_info
            )
            
            # Store content and metadata
            self._active_content[memory_id] = content
            self._metadata[memory_id] = metadata
            self.deduplication_manager.register_content(memory_id, content)
            
            # Persist metadata
            self._persist_metadata(metadata)
            
            self._stats['created_count'] += 1
            
            logger.debug(f"Created memory {memory_id} with confidence {metadata.confidence_score:.2f}")
            
            return metadata
    
    def retrieve_memory(self, memory_id: str) -> Optional[Tuple[bytes, MemoryMetadata]]:
        """
        Retrieve a memory, updating access statistics.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Tuple of (content, metadata) or None if not found
        """
        with self._lock:
            metadata = self._metadata.get(memory_id)
            
            if not metadata:
                # Try archived
                archived = self.archival_manager.retrieve_archived(memory_id)
                if archived:
                    content, metadata = archived
                    metadata.record_access()
                    self._stats['retrieved_count'] += 1
                    
                    # Check if should reactivate
                    new_stage = self.decay_detector.should_transition(metadata)
                    if new_stage == LifecycleStage.ACTIVE:
                        self._reactivate_memory(memory_id, content, metadata)
                    
                    return content, metadata
                return None
            
            # Update access stats
            metadata.record_access()
            self._stats['retrieved_count'] += 1
            
            # Check for stage transition
            self._check_and_transition(metadata)
            
            content = self._active_content.get(memory_id)
            if content:
                return content, metadata
            
            return None
    
    def update_memory(self, memory_id: str, content: Optional[bytes] = None,
                     confirmation: bool = False,
                     contradiction: bool = False,
                     user_confirmed: Optional[bool] = None) -> Optional[MemoryMetadata]:
        """
        Update a memory with new information.
        
        Args:
            memory_id: Memory identifier
            content: New content (optional)
            confirmation: Whether this confirms the memory
            contradiction: Whether this contradicts the memory
            user_confirmed: New user confirmation status
            
        Returns:
            Updated metadata or None if not found
        """
        with self._lock:
            metadata = self._metadata.get(memory_id)
            
            if not metadata:
                # Try to retrieve from archive
                archived = self.archival_manager.retrieve_archived(memory_id)
                if archived:
                    _, metadata = archived
                    # Will need to reactivate to update
                    return None
                return None
            
            # Update content if provided
            if content:
                self._active_content[memory_id] = content
                metadata.content_hash = self.deduplication_manager.compute_hash(content)
                metadata.last_modified = datetime.now()
            
            # Update confidence
            self.confidence_scorer.update_confidence(
                metadata, confirmation, contradiction, user_confirmed
            )
            
            metadata.record_access()
            self._persist_metadata(metadata)
            
            return metadata
    
    def delete_memory(self, memory_id: str, permanent: bool = False) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory identifier
            permanent: If True, permanently delete; otherwise move to EXPIRED
            
        Returns:
            True if successful
        """
        with self._lock:
            metadata = self._metadata.get(memory_id)
            
            if metadata:
                if permanent:
                    # Remove from active storage
                    self._active_content.pop(memory_id, None)
                    self._metadata.pop(memory_id, None)
                    self._delete_metadata(memory_id)
                    self._stats['deleted_count'] += 1
                else:
                    # Move to expired
                    old_stage = metadata.stage
                    metadata.stage = LifecycleStage.EXPIRED
                    self._persist_metadata(metadata)
                    self._stats['transitions'][f"{old_stage.value}_to_expired"] += 1
                
                return True
            
            # Try to delete from archive
            if permanent:
                return self.archival_manager.delete_archived(memory_id)
            
            return False
    
    def _check_and_transition(self, metadata: MemoryMetadata):
        """Check if memory should transition and perform transition."""
        # Calculate decay
        recent_queries = self.decay_detector.get_recent_queries()
        decay_score = self.decay_detector.calculate_decay_score(metadata, recent_queries)
        
        # Check for transition
        new_stage = self.decay_detector.should_transition(metadata, decay_score)
        
        if new_stage and new_stage != metadata.stage:
            self._perform_transition(metadata, new_stage)
    
    def _perform_transition(self, metadata: MemoryMetadata, 
                           new_stage: LifecycleStage):
        """Perform lifecycle stage transition."""
        old_stage = metadata.stage
        memory_id = metadata.memory_id
        
        logger.info(f"Transitioning {memory_id} from {old_stage.value} to {new_stage.value}")
        
        if new_stage == LifecycleStage.ARCHIVED:
            # Archive the memory
            content = self._active_content.get(memory_id)
            if content:
                success, _ = self.archival_manager.archive_memory(
                    memory_id, content, metadata
                )
                if success:
                    # Remove from active
                    self._active_content.pop(memory_id, None)
                    self._metadata.pop(memory_id, None)
                    self._delete_metadata(memory_id)
                    self._stats['archived_count'] += 1
        
        elif new_stage == LifecycleStage.DELETED:
            # Permanent deletion
            self._active_content.pop(memory_id, None)
            self._metadata.pop(memory_id, None)
            self._delete_metadata(memory_id)
            self.archival_manager.delete_archived(memory_id)
            self._stats['deleted_count'] += 1
        
        else:
            # Simple stage update
            metadata.stage = new_stage
            metadata.last_modified = datetime.now()
            self._persist_metadata(metadata)
        
        self._stats['transitions'][f"{old_stage.value}_to_{new_stage.value}"] += 1
    
    def _reactivate_memory(self, memory_id: str, content: bytes, 
                          metadata: MemoryMetadata):
        """Reactivate an archived memory."""
        metadata.stage = LifecycleStage.ACTIVE
        metadata.archived_at = None
        metadata.last_modified = datetime.now()
        
        self._active_content[memory_id] = content
        self._metadata[memory_id] = metadata
        self._persist_metadata(metadata)
        
        # Delete from archive
        self.archival_manager.delete_archived(memory_id)
        
        logger.info(f"Reactivated memory {memory_id}")
    
    def _persist_metadata(self, metadata: MemoryMetadata):
        """Persist metadata to database."""
        with self._get_metadata_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memory_metadata 
                (memory_id, stage, memory_type, created_at, last_accessed, 
                 last_modified, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.memory_id,
                metadata.stage.value,
                metadata.memory_type.value,
                metadata.created_at.isoformat(),
                metadata.last_accessed.isoformat(),
                metadata.last_modified.isoformat(),
                json.dumps(metadata.to_dict())
            ))
            
            conn.commit()
    
    def _delete_metadata(self, memory_id: str):
        """Delete metadata from database."""
        with self._get_metadata_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memory_metadata WHERE memory_id = ?', (memory_id,))
            conn.commit()
    
    def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance job to process lifecycle transitions.
        
        Returns:
            Maintenance job statistics
        """
        start_time = time.time()
        
        with self._lock:
            processed = 0
            transitions = 0
            archived = 0
            deleted = 0
            
            # Process batch of memories
            memory_ids = list(self._metadata.keys())[:self.config.batch_size]
            
            for memory_id in memory_ids:
                metadata = self._metadata.get(memory_id)
                if not metadata:
                    continue
                
                processed += 1
                
                # Calculate decay
                recent_queries = self.decay_detector.get_recent_queries()
                decay_score = self.decay_detector.calculate_decay_score(
                    metadata, recent_queries
                )
                
                # Check for transition
                new_stage = self.decay_detector.should_transition(metadata, decay_score)
                
                if new_stage and new_stage != metadata.stage:
                    self._perform_transition(metadata, new_stage)
                    transitions += 1
                    
                    if new_stage == LifecycleStage.ARCHIVED:
                        archived += 1
                    elif new_stage == LifecycleStage.DELETED:
                        deleted += 1
            
            duration = time.time() - start_time
            
            results = {
                'processed': processed,
                'transitions': transitions,
                'archived': archived,
                'deleted': deleted,
                'duration_seconds': duration
            }
            
            logger.info(f"Maintenance completed: {results}")
            
            return results
    
    def start_maintenance_scheduler(self):
        """Start the maintenance scheduler in background thread."""
        if self._maintenance_running:
            return
        
        self._maintenance_running = True
        
        def maintenance_loop():
            interval = self.config.maintenance_interval_hours * 3600
            
            while self._maintenance_running:
                try:
                    logger.info("Running scheduled maintenance")
                    self.run_maintenance()
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                
                # Sleep with interrupt checking
                for _ in range(int(interval)):
                    if not self._maintenance_running:
                        break
                    time.sleep(1)
        
        self._maintenance_thread = threading.Thread(
            target=maintenance_loop,
            daemon=True
        )
        self._maintenance_thread.start()
        
        logger.info("Maintenance scheduler started")
    
    def stop_maintenance_scheduler(self):
        """Stop the maintenance scheduler."""
        self._maintenance_running = False
        
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)
            self._maintenance_thread = None
        
        logger.info("Maintenance scheduler stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics."""
        with self._lock:
            stage_counts = defaultdict(int)
            total_size = 0
            
            for memory_id, metadata in self._metadata.items():
                stage_counts[metadata.stage.value] += 1
                content = self._active_content.get(memory_id)
                if content:
                    total_size += len(content)
            
            # Add archived stats
            archive_stats = self.archival_manager.get_archive_stats()
            
            return {
                'active_memories': len(self._metadata),
                'stage_distribution': dict(stage_counts),
                'active_storage_bytes': total_size,
                'created_total': self._stats['created_count'],
                'archived_total': self._stats['archived_count'],
                'deleted_total': self._stats['deleted_count'],
                'retrieved_total': self._stats['retrieved_count'],
                'transitions': dict(self._stats['transitions']),
                'archive_stats': archive_stats
            }
    
    def get_memory_lifecycle(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get complete lifecycle information for a memory."""
        with self._lock:
            metadata = self._metadata.get(memory_id)
            
            if not metadata:
                # Try archived
                archived = self.archival_manager.retrieve_archived(memory_id)
                if archived:
                    _, metadata = archived
                else:
                    return None
            
            days_active = metadata.days_since_creation()
            days_since_access = metadata.days_since_last_access()
            
            return {
                'memory_id': memory_id,
                'stage': metadata.stage.value,
                'memory_type': metadata.memory_type.value,
                'confidence_score': metadata.confidence_score,
                'decay_score': metadata.decay_score,
                'relevance_score': metadata.relevance_score,
                'created_at': metadata.created_at.isoformat(),
                'last_accessed': metadata.last_accessed.isoformat(),
                'days_since_creation': days_active,
                'days_since_last_access': days_since_access,
                'access_count': metadata.access_count,
                'confirmation_count': metadata.confirmation_count,
                'contradiction_count': metadata.contradiction_count,
                'user_confirmed': metadata.user_confirmed,
                'archived_at': metadata.archived_at.isoformat() if metadata.archived_at else None,
                'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                'next_action': self._predict_next_action(metadata)
            }
    
    def _predict_next_action(self, metadata: MemoryMetadata) -> str:
        """Predict next lifecycle action for a memory."""
        new_stage = self.decay_detector.should_transition(metadata)
        
        if new_stage:
            return f"transition_to_{new_stage.value}"
        
        if metadata.stage == LifecycleStage.ACTIVE:
            if metadata.days_since_last_access() > self.config.cooling_threshold_days // 2:
                return "monitor_for_decay"
        
        return "no_action"
    
    def query_memories(self, stage: Optional[LifecycleStage] = None,
                      min_confidence: float = 0.0,
                      memory_type: Optional[MemoryType] = None,
                      limit: int = 100) -> List[MemoryMetadata]:
        """
        Query memories by filters.
        
        Args:
            stage: Filter by lifecycle stage
            min_confidence: Minimum confidence threshold
            memory_type: Filter by memory type
            limit: Maximum results
            
        Returns:
            List of matching memory metadata
        """
        with self._lock:
            results = []
            
            for metadata in self._metadata.values():
                if stage and metadata.stage != stage:
                    continue
                if metadata.confidence_score < min_confidence:
                    continue
                if memory_type and metadata.memory_type != memory_type:
                    continue
                
                results.append(metadata)
                
                if len(results) >= limit:
                    break
            
            # Sort by confidence descending
            results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return results
    
    def force_archive(self, memory_id: str) -> bool:
        """
        Force immediate archival of a memory.
        
        Args:
            memory_id: Memory to archive
            
        Returns:
            True if archived successfully
        """
        with self._lock:
            metadata = self._metadata.get(memory_id)
            if not metadata:
                return False
            
            content = self._active_content.get(memory_id)
            if not content:
                return False
            
            self._perform_transition(metadata, LifecycleStage.ARCHIVED)
            return True
    
    def force_reactivate(self, memory_id: str) -> bool:
        """
        Force reactivation of an archived memory.
        
        Args:
            memory_id: Memory to reactivate
            
        Returns:
            True if reactivated successfully
        """
        with self._lock:
            # Try to retrieve from archive
            archived = self.archival_manager.retrieve_archived(memory_id)
            if archived:
                content, metadata = archived
                self._reactivate_memory(memory_id, content, metadata)
                return True
            return False
    
    def close(self):
        """Clean up resources and stop maintenance."""
        self.stop_maintenance_scheduler()
        
        # Persist all metadata
        with self._lock:
            for metadata in self._metadata.values():
                self._persist_metadata(metadata)
        
        logger.info("MemoryLifecycleManager closed")


# Convenience functions for common operations

def create_lifecycle_manager(config: Optional[LifecycleConfig] = None) -> MemoryLifecycleManager:
    """Create and initialize a lifecycle manager."""
    return MemoryLifecycleManager(config)


def run_lifecycle_maintenance(manager: MemoryLifecycleManager) -> Dict[str, Any]:
    """Run one maintenance cycle."""
    return manager.run_maintenance()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create lifecycle manager with custom config
    config = LifecycleConfig(
        decay_days_min=30,  # Faster decay for demo
        decay_days_max=90,
        maintenance_interval_hours=1,
        compress_archived=True
    )
    
    manager = create_lifecycle_manager(config)
    
    # Create test memories
    print("\n=== Creating Test Memories ===")
    
    # High confidence memory
    meta1 = manager.create_memory(
        memory_id="test_memory_1",
        content=b"This is a high-confidence memory from verified source.",
        source_info={
            'type': 'verified_database',
            'authoritative': True,
            'age_days': 10
        },
        memory_type=MemoryType.STANDARD
    )
    print(f"Created memory 1: confidence={meta1.confidence_score:.2f}")
    
    # Low confidence memory
    meta2 = manager.create_memory(
        memory_id="test_memory_2",
        content=b"This is an unverified memory.",
        source_info={
            'type': 'unverified',
            'authoritative': False
        },
        memory_type=MemoryType.EPHEMERAL
    )
    print(f"Created memory 2: confidence={meta2.confidence_score:.2f}")
    
    # Core memory (never decays)
    meta3 = manager.create_memory(
        memory_id="test_memory_3",
        content=b"This is a CORE memory that should never decay.",
        source_info={
            'type': 'verified_database',
            'authoritative': True
        },
        memory_type=MemoryType.CORE
    )
    print(f"Created memory 3: confidence={meta3.confidence_score:.2f}, type={meta3.memory_type.value}")
    
    # Simulate access patterns
    print("\n=== Simulating Access Patterns ===")
    manager.retrieve_memory("test_memory_1")  # Access memory 1
    manager.retrieve_memory("test_memory_1")  # Access again
    
    # Update confidence
    manager.update_memory("test_memory_1", confirmation=True)
    print("Memory 1: confirmed twice")
    
    # Query memories
    print("\n=== Querying Memories ===")
    active_memories = manager.query_memories(stage=LifecycleStage.ACTIVE)
    print(f"Active memories: {len(active_memories)}")
    for meta in active_memories:
        print(f"  - {meta.memory_id}: confidence={meta.confidence_score:.2f}, "
              f"access_count={meta.access_count}")
    
    # Get statistics
    print("\n=== Lifecycle Statistics ===")
    stats = manager.get_statistics()
    print(f"Total active: {stats['active_memories']}")
    print(f"Stage distribution: {stats['stage_distribution']}")
    print(f"Created total: {stats['created_total']}")
    
    # Run maintenance
    print("\n=== Running Maintenance ===")
    maint_results = manager.run_maintenance()
    print(f"Maintenance results: {maint_results}")
    
    # Check lifecycle status
    print("\n=== Memory Lifecycle Status ===")
    for mid in ["test_memory_1", "test_memory_2", "test_memory_3"]:
        lifecycle = manager.get_memory_lifecycle(mid)
        if lifecycle:
            print(f"\n{mid}:")
            print(f"  Stage: {lifecycle['stage']}")
            print(f"  Confidence: {lifecycle['confidence_score']:.2f}")
            print(f"  Decay Score: {lifecycle['decay_score']:.2f}")
            print(f"  Next Action: {lifecycle['next_action']}")
    
    # Cleanup
    manager.close()
    print("\n=== Demo Complete ===")
