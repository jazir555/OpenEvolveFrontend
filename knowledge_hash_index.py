"""
Knowledge Hash Index - Hash-based Deduplication Layer for Memories

This module provides a deduplication layer using multiple hash strategies:
- Exact hash (MD5/SHA256) for identical content detection
- SimHash for near-duplicate detection using Hamming distance
- MinHash for fuzzy matching using Jaccard similarity
- Bloom filter for fast existence checks

Author: OpenEvolve AI
Version: 1.0.0
"""

import hashlib
import json
import sqlite3
import threading
import time
import pickle
import struct
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HashIndexConfig:
    """Configuration for the Hash Index system."""
    
    # Database settings
    db_path: str = "knowledge_hash_index.db"
    
    # Similarity thresholds
    simhash_threshold: int = 3  # Hamming distance threshold for SimHash
    minhash_threshold: float = 0.85  # Jaccard similarity threshold for MinHash
    
    # MinHash parameters
    minhash_num_permutations: int = 128
    minhash_shingle_size: int = 3
    
    # SimHash parameters
    simhash_vector_size: int = 64
    
    # Bloom filter parameters
    bloom_filter_size: int = 1000000  # Number of bits
    bloom_filter_hash_count: int = 7  # Number of hash functions
    
    # Merger settings
    auto_merge_enabled: bool = True
    keep_most_detailed: bool = True
    preserve_all_links: bool = True
    
    # Performance settings
    cache_size: int = 10000
    batch_size: int = 1000
    vacuum_interval: int = 10000  # Operations between vacuum
    
    # Thread safety
    enable_locking: bool = True


# =============================================================================
# Utility Functions
# =============================================================================

def compute_md5_hash(content: Union[str, bytes]) -> str:
    """Compute MD5 hash for exact duplicate detection."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.md5(content).hexdigest()


def compute_sha256_hash(content: Union[str, bytes]) -> str:
    """Compute SHA256 hash for cryptographic exact matching."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def tokenize_text(text: str, shingle_size: int = 3) -> List[str]:
    """Tokenize text into shingles for MinHash computation."""
    # Normalize text
    text = text.lower().strip()
    # Simple word-based tokenization
    words = text.split()
    
    if len(words) < shingle_size:
        return [' '.join(words)]
    
    shingles = []
    for i in range(len(words) - shingle_size + 1):
        shingle = ' '.join(words[i:i + shingle_size])
        shingles.append(shingle)
    
    return shingles


def hamming_distance(hash1: int, hash2: int, num_bits: int = 64) -> int:
    """Calculate Hamming distance between two bit strings."""
    xor = hash1 ^ hash2
    return bin(xor).count('1')


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# Hash Computation Functions
# =============================================================================

def compute_simhash(content: Union[str, bytes], vector_size: int = 64) -> int:
    """
    Compute SimHash for near-duplicate detection using locality-sensitive hashing.
    
    SimHash creates similar hashes for similar content, allowing detection
    of near-duplicates using Hamming distance.
    
    Args:
        content: The content to hash
        vector_size: Size of the hash vector (typically 64 or 128 bits)
        
    Returns:
        Integer representing the SimHash value
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    # Initialize vector
    vector = [0] * vector_size
    
    # Tokenize content into features (words)
    text = content.decode('utf-8', errors='ignore')
    words = text.split()
    
    # Compute hash for each word and update vector
    for word in words:
        word_hash = hashlib.md5(word.encode('utf-8')).digest()
        # Use first vector_size bits
        for i in range(vector_size):
            byte_idx = i // 8
            bit_idx = i % 8
            if word_hash[byte_idx] & (1 << bit_idx):
                vector[i] += 1
            else:
                vector[i] -= 1
    
    # Convert to bit string
    result = 0
    for i, v in enumerate(vector):
        if v > 0:
            result |= (1 << i)
    
    return result


def compute_minhash(
    content: Union[str, bytes],
    num_permutations: int = 128,
    shingle_size: int = 3,
    seed: int = 42
) -> List[int]:
    """
    Compute MinHash signature for Jaccard similarity estimation.
    
    MinHash creates compact signatures that allow estimation of Jaccard
    similarity between documents without comparing full content.
    
    Args:
        content: The content to hash
        num_permutations: Number of hash functions (signature size)
        shingle_size: Size of shingles (n-grams)
        seed: Random seed for reproducibility
        
    Returns:
        List of integers representing the MinHash signature
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='ignore')
    
    # Generate shingles
    shingles = set(tokenize_text(content, shingle_size))
    
    if not shingles:
        return [0] * num_permutations
    
    # Initialize signature with infinity
    signature = [float('inf')] * num_permutations
    
    # Pre-compute random hash parameters
    import random
    rng = random.Random(seed)
    hash_params = [(rng.randint(1, 2**32), rng.randint(0, 2**32)) 
                   for _ in range(num_permutations)]
    
    # Compute MinHash signature
    for shingle in shingles:
        shingle_hash = hash(shingle) & 0xFFFFFFFF
        for i, (a, b) in enumerate(hash_params):
            # Universal hashing: (a * x + b) % p
            hash_val = ((a * shingle_hash + b) % (2**32 - 1))
            signature[i] = min(signature[i], hash_val)
    
    return [int(x) for x in signature]


def compute_minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Compute Jaccard similarity estimate from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have the same length")
    
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def compute_combined_hash(content: Union[str, bytes]) -> Dict[str, Any]:
    """
    Compute all hash types for comprehensive deduplication.
    
    Returns:
        Dictionary containing all hash values
    """
    if isinstance(content, bytes):
        text = content.decode('utf-8', errors='ignore')
    else:
        text = content
        content = content.encode('utf-8')
    
    return {
        'md5': compute_md5_hash(content),
        'sha256': compute_sha256_hash(content),
        'simhash': str(compute_simhash(content)),  # Store as string to avoid overflow
        'minhash': compute_minhash(text),
        'timestamp': time.time()
    }


# =============================================================================
# Bloom Filter
# =============================================================================

class BloomFilter:
    """
    Bloom filter for fast existence checks.
    
    Provides O(1) membership testing with a small probability of false positives
    but no false negatives. Useful for quickly checking if a memory might exist
    before performing expensive database queries.
    """
    
    def __init__(self, size: int = 1000000, hash_count: int = 7):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bytearray((size + 7) // 8)
        self._lock = threading.RLock()
        self._item_count = 0
    
    def _get_hash_positions(self, item: str) -> List[int]:
        """Get bit positions for an item using multiple hash functions."""
        positions = []
        hash1 = hashlib.md5(item.encode()).hexdigest()
        hash2 = hashlib.sha256(item.encode()).hexdigest()
        
        # Double hashing technique
        h1 = int(hash1, 16) % self.size
        h2 = int(hash2, 16) % self.size
        
        for i in range(self.hash_count):
            pos = (h1 + i * h2) % self.size
            positions.append(pos)
        
        return positions
    
    def add(self, item: str) -> None:
        """Add an item to the bloom filter."""
        with self._lock:
            for pos in self._get_hash_positions(item):
                byte_idx = pos // 8
                bit_idx = pos % 8
                self.bit_array[byte_idx] |= (1 << bit_idx)
            self._item_count += 1
    
    def contains(self, item: str) -> bool:
        """Check if an item might be in the set (may have false positives)."""
        with self._lock:
            for pos in self._get_hash_positions(item):
                byte_idx = pos // 8
                bit_idx = pos % 8
                if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                    return False
            return True
    
    def clear(self) -> None:
        """Clear the bloom filter."""
        with self._lock:
            self.bit_array = bytearray((self.size + 7) // 8)
            self._item_count = 0
    
    @property
    def item_count(self) -> int:
        """Get approximate number of items added."""
        return self._item_count
    
    def estimated_false_positive_rate(self) -> float:
        """Estimate the current false positive probability."""
        n = self._item_count
        m = self.size
        k = self.hash_count
        return (1 - (1 - 1/m)**(k*n))**k
    
    def to_bytes(self) -> bytes:
        """Serialize bloom filter to bytes."""
        with self._lock:
            return bytes(self.bit_array)
    
    def from_bytes(self, data: bytes) -> None:
        """Deserialize bloom filter from bytes."""
        with self._lock:
            self.bit_array = bytearray(data)


# =============================================================================
# Hash Entry
# =============================================================================

@dataclass
class HashEntry:
    """
    Represents a hashed memory with collision handling.
    
    Stores multiple hash types for comprehensive deduplication and
    maintains metadata for conflict resolution during merging.
    """
    
    # Unique identifier for the memory
    memory_id: str
    
    # Hash values
    exact_hash: str  # MD5 hash
    sha256_hash: str  # SHA256 hash
    simhash: str  # SimHash value (stored as string to avoid SQLite integer overflow)
    minhash: List[int] = field(default_factory=list)  # MinHash signature
    
    # Metadata
    content_size: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    # Content analysis for merging
    content_depth: int = 0  # Measure of detail/complexity
    num_relationships: int = 0  # Number of linked memories
    quality_score: float = 0.0  # Quality assessment
    
    # Collision handling
    collision_chain: List[str] = field(default_factory=list)  # IDs of similar entries
    is_primary: bool = True  # Primary entry vs. merged duplicate
    merged_from: List[str] = field(default_factory=list)  # IDs merged into this entry
    
    # Additional metadata
    tags: Set[str] = field(default_factory=set)
    source_context: str = ""
    merge_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure minhash is a list."""
        if isinstance(self.minhash, str):
            self.minhash = json.loads(self.minhash)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HashEntry':
        """Create entry from dictionary."""
        # Convert lists back to sets
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        return cls(**data)
    
    def update_access(self) -> None:
        """Update access timestamp and count."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def compute_content_depth(self, content: str) -> int:
        """Compute a measure of content detail/complexity."""
        # Simple heuristic: length + unique words + sentence count
        words = content.split()
        unique_words = len(set(words))
        sentences = content.count('.') + content.count('!') + content.count('?')
        self.content_depth = len(words) + unique_words + sentences
        return self.content_depth
    
    def simhash_distance(self, other: 'HashEntry') -> int:
        """Compute Hamming distance to another entry's SimHash."""
        return hamming_distance(int(self.simhash), int(other.simhash))
    
    def minhash_similarity(self, other: 'HashEntry') -> float:
        """Compute MinHash similarity to another entry."""
        return compute_minhash_similarity(self.minhash, other.minhash)
    
    def is_duplicate_of(self, other: 'HashEntry', 
                        simhash_threshold: int = 3,
                        minhash_threshold: float = 0.85) -> bool:
        """
        Check if this entry is a duplicate of another.
        
        Uses multiple criteria:
        1. Exact hash match (MD5 or SHA256)
        2. SimHash within threshold
        3. MinHash above threshold
        """
        # Exact match
        if self.exact_hash == other.exact_hash:
            return True
        if self.sha256_hash == other.sha256_hash:
            return True
        
        # Near-duplicate detection
        if self.simhash_distance(other) <= simhash_threshold:
            return True
        
        if self.minhash and other.minhash:
            if self.minhash_similarity(other) >= minhash_threshold:
                return True
        
        return False


# =============================================================================
# Duplicate Merger
# =============================================================================

class DuplicateMerger:
    """
    Merges near-duplicate memories with intelligent conflict resolution.
    
    Merging strategies:
    - Keeps the most detailed version (highest content depth)
    - Updates access timestamps to most recent
    - Aggregates metadata (tags, sources, relationships)
    - Preserves all relationship links
    - Maintains merge history for traceability
    """
    
    def __init__(self, config: Optional[HashIndexConfig] = None):
        self.config = config or HashIndexConfig()
        self._merge_stats = {
            'total_merges': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'fields_merged': defaultdict(int)
        }
    
    def merge_entries(self, primary: HashEntry, 
                      duplicates: List[HashEntry]) -> HashEntry:
        """
        Merge duplicate entries into the primary entry.
        
        Args:
            primary: The primary entry to merge into
            duplicates: List of duplicate entries
            
        Returns:
            Updated primary entry
        """
        all_entries = [primary] + duplicates
        
        # Select best primary if configured
        if self.config.keep_most_detailed:
            primary = self._select_best_primary(all_entries)
            duplicates = [e for e in all_entries if e.memory_id != primary.memory_id]
        
        # Record merge operation
        merge_record = {
            'timestamp': time.time(),
            'primary_id': primary.memory_id,
            'merged_ids': [d.memory_id for d in duplicates],
            'strategy': 'content_depth' if self.config.keep_most_detailed else 'first_seen'
        }
        primary.merge_history.append(merge_record)
        primary.merged_from.extend([d.memory_id for d in duplicates])
        
        # Aggregate metadata
        self._aggregate_metadata(primary, duplicates)
        
        # Update statistics
        self._merge_stats['total_merges'] += len(duplicates)
        for dup in duplicates:
            if dup.exact_hash == primary.exact_hash:
                self._merge_stats['exact_duplicates'] += 1
            else:
                self._merge_stats['near_duplicates'] += 1
        
        logger.info(f"Merged {len(duplicates)} duplicates into {primary.memory_id}")
        return primary
    
    def _select_best_primary(self, entries: List[HashEntry]) -> HashEntry:
        """Select the best primary entry based on multiple criteria."""
        def score_entry(entry: HashEntry) -> float:
            """Score an entry for quality (higher is better)."""
            score = 0.0
            
            # Prefer more detailed content
            score += entry.content_depth * 1.0
            
            # Prefer entries with more relationships
            score += entry.num_relationships * 10.0
            
            # Prefer higher quality scores
            score += entry.quality_score * 50.0
            
            # Prefer more recently accessed entries
            score += entry.access_count * 5.0
            
            # Prefer primary entries over already-merged ones
            if entry.is_primary:
                score += 100.0
            
            return score
        
        return max(entries, key=score_entry)
    
    def _aggregate_metadata(self, primary: HashEntry, 
                           duplicates: List[HashEntry]) -> None:
        """Aggregate metadata from duplicates into primary."""
        
        # Aggregate tags
        for dup in duplicates:
            primary.tags.update(dup.tags)
            self._merge_stats['fields_merged']['tags'] += 1
        
        # Keep most recent access time
        all_access_times = [primary.last_accessed] + [d.last_accessed for d in duplicates]
        primary.last_accessed = max(all_access_times)
        
        # Sum access counts
        primary.access_count += sum(d.access_count for d in duplicates)
        self._merge_stats['fields_merged']['access_count'] += len(duplicates)
        
        # Keep oldest creation time
        all_create_times = [primary.created_at] + [d.created_at for d in duplicates]
        primary.created_at = min(all_create_times)
        
        # Update update time
        primary.updated_at = time.time()
        
        # Aggregate collision chains
        for dup in duplicates:
            primary.collision_chain.extend(dup.collision_chain)
            primary.collision_chain.append(dup.memory_id)
        primary.collision_chain = list(set(primary.collision_chain))
        
        # Preserve relationship count (sum all)
        primary.num_relationships += sum(d.num_relationships for d in duplicates)
        
        # Keep highest quality score
        all_quality = [primary.quality_score] + [d.quality_score for d in duplicates]
        primary.quality_score = max(all_quality)
        
        # Aggregate merge histories
        for dup in duplicates:
            primary.merge_history.extend(dup.merge_history)
    
    def create_merged_content(self, contents: List[str]) -> str:
        """
        Create merged content by combining unique information.
        
        This is a placeholder - actual implementation would use
        more sophisticated text merging techniques.
        """
        # For now, return the longest/most detailed content
        return max(contents, key=len)
    
    def get_merge_stats(self) -> Dict[str, Any]:
        """Get statistics about merge operations."""
        return {
            **self._merge_stats,
            'fields_merged': dict(self._merge_stats['fields_merged'])
        }
    
    def reset_stats(self) -> None:
        """Reset merge statistics."""
        self._merge_stats = {
            'total_merges': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'fields_merged': defaultdict(int)
        }


# =============================================================================
# Incremental Hasher
# =============================================================================

class IncrementalHasher:
    """
    Supports incremental hashing for streaming updates.
    
    Allows computing hash values for content that arrives in chunks,
    useful for processing large memories or streaming content.
    """
    
    def __init__(self):
        self.md5_hash = hashlib.md5()
        self.sha256_hash = hashlib.sha256()
        self.chunks: List[bytes] = []
        self.total_size = 0
        self._finalized = False
        self._simhash_words: List[str] = []
        self._shingles: Set[str] = set()
    
    def update(self, chunk: Union[str, bytes]) -> None:
        """Update hashes with a new chunk of content."""
        if self._finalized:
            raise ValueError("Cannot update finalized hasher")
        
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
        
        # Update cryptographic hashes
        self.md5_hash.update(chunk)
        self.sha256_hash.update(chunk)
        
        # Store for other hash types
        self.chunks.append(chunk)
        self.total_size += len(chunk)
        
        # Collect words for SimHash
        text = chunk.decode('utf-8', errors='ignore')
        self._simhash_words.extend(text.split())
        
        # Collect shingles for MinHash
        words = text.split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            self._shingles.add(shingle)
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize and return all hash values."""
        if self._finalized:
            raise ValueError("Already finalized")
        
        # Finalize cryptographic hashes
        exact_hash = self.md5_hash.hexdigest()
        sha256_hash = self.sha256_hash.hexdigest()
        
        # Compute SimHash from collected words
        simhash = self._compute_simhash_from_words()
        
        # Compute MinHash from collected shingles
        minhash = self._compute_minhash_from_shingles()
        
        self._finalized = True
        
        return {
            'md5': exact_hash,
            'sha256': sha256_hash,
            'simhash': simhash,
            'minhash': minhash,
            'size': self.total_size
        }
    
    def _compute_simhash_from_words(self) -> int:
        """Compute SimHash from collected words."""
        vector_size = 64
        vector = [0] * vector_size
        
        for word in self._simhash_words:
            word_hash = hashlib.md5(word.encode('utf-8')).digest()
            for i in range(vector_size):
                byte_idx = i // 8
                bit_idx = i % 8
                if word_hash[byte_idx] & (1 << bit_idx):
                    vector[i] += 1
                else:
                    vector[i] -= 1
        
        result = 0
        for i, v in enumerate(vector):
            if v > 0:
                result |= (1 << i)
        return result
    
    def _compute_minhash_from_shingles(self) -> List[int]:
        """Compute MinHash from collected shingles."""
        num_permutations = 128
        seed = 42
        
        if not self._shingles:
            return [0] * num_permutations
        
        signature = [float('inf')] * num_permutations
        
        import random
        rng = random.Random(seed)
        hash_params = [(rng.randint(1, 2**32), rng.randint(0, 2**32))
                       for _ in range(num_permutations)]
        
        for shingle in self._shingles:
            shingle_hash = hash(shingle) & 0xFFFFFFFF
            for i, (a, b) in enumerate(hash_params):
                hash_val = ((a * shingle_hash + b) % (2**32 - 1))
                signature[i] = min(signature[i], hash_val)
        
        return [int(x) for x in signature]
    
    def reset(self) -> None:
        """Reset the hasher for reuse."""
        self.md5_hash = hashlib.md5()
        self.sha256_hash = hashlib.sha256()
        self.chunks = []
        self.total_size = 0
        self._finalized = False
        self._simhash_words = []
        self._shingles = set()


# =============================================================================
# Hash Index
# =============================================================================

class HashIndex:
    """
    Deduplication layer using multiple hash strategies.
    
    Provides:
    - Exact hash for identical content (MD5/SHA256)
    - SimHash for near-duplicate detection (Hamming distance)
    - MinHash for fuzzy matching (Jaccard similarity)
    - Bloom filter for fast existence checks
    - Automatic duplicate merging
    - Thread-safe operations
    - Persistence to SQLite
    """
    
    def __init__(self, config: Optional[HashIndexConfig] = None):
        self.config = config or HashIndexConfig()
        self.db_path = self.config.db_path
        
        # Initialize bloom filter
        self.bloom_filter = BloomFilter(
            size=self.config.bloom_filter_size,
            hash_count=self.config.bloom_filter_hash_count
        )
        
        # Initialize merger
        self.merger = DuplicateMerger(self.config)
        
        # Thread safety
        self._lock = threading.RLock() if self.config.enable_locking else contextmanager(lambda: (yield))
        
        # Caches
        self._entry_cache: Dict[str, HashEntry] = {}
        self._simhash_index: Dict[int, List[str]] = defaultdict(list)  # simhash -> memory_ids
        
        # Operation counter for maintenance
        self._op_count = 0
        
        # Initialize database
        self._init_database()
        self._load_bloom_filter()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hash_entries (
                memory_id TEXT PRIMARY KEY,
                exact_hash TEXT NOT NULL,
                sha256_hash TEXT NOT NULL,
                simhash TEXT NOT NULL,
                minhash TEXT NOT NULL,
                content_size INTEGER DEFAULT 0,
                created_at REAL DEFAULT 0,
                updated_at REAL DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0,
                content_depth INTEGER DEFAULT 0,
                num_relationships INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0,
                collision_chain TEXT DEFAULT '[]',
                is_primary INTEGER DEFAULT 1,
                merged_from TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                source_context TEXT DEFAULT '',
                merge_history TEXT DEFAULT '[]'
            )
        """)
        
        # Create indexes for fast lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_exact_hash ON hash_entries(exact_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sha256_hash ON hash_entries(sha256_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_simhash ON hash_entries(simhash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_primary ON hash_entries(is_primary)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON hash_entries(last_accessed)")
        
        # Bloom filter storage
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bloom_filter (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data BLOB NOT NULL,
                item_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        conn.close()
    
    def _load_bloom_filter(self) -> None:
        """Load bloom filter from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT data, item_count FROM bloom_filter WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    self.bloom_filter.from_bytes(row[0])
                    self.bloom_filter._item_count = row[1]
        except Exception as e:
            logger.warning(f"Could not load bloom filter: {e}")
    
    def _save_bloom_filter(self) -> None:
        """Save bloom filter to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = self.bloom_filter.to_bytes()
                count = self.bloom_filter.item_count
                conn.execute("""
                    INSERT INTO bloom_filter (id, data, item_count)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET data = ?, item_count = ?
                """, (data, count, data, count))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not save bloom filter: {e}")
    
    @contextmanager
    def _get_db(self):
        """Get database connection with proper locking."""
        if self.config.enable_locking:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    yield conn
                finally:
                    conn.close()
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
    
    def _entry_to_row(self, entry: HashEntry) -> Tuple:
        """Convert HashEntry to database row."""
        return (
            entry.memory_id,
            entry.exact_hash,
            entry.sha256_hash,
            entry.simhash,
            json.dumps(entry.minhash),
            entry.content_size,
            entry.created_at,
            entry.updated_at,
            entry.access_count,
            entry.last_accessed,
            entry.content_depth,
            entry.num_relationships,
            entry.quality_score,
            json.dumps(entry.collision_chain),
            1 if entry.is_primary else 0,
            json.dumps(entry.merged_from),
            json.dumps(list(entry.tags)),
            entry.source_context,
            json.dumps(entry.merge_history)
        )
    
    def _row_to_entry(self, row: Tuple) -> HashEntry:
        """Convert database row to HashEntry."""
        return HashEntry(
            memory_id=row[0],
            exact_hash=row[1],
            sha256_hash=row[2],
            simhash=row[3],
            minhash=json.loads(row[4]),
            content_size=row[5],
            created_at=row[6],
            updated_at=row[7],
            access_count=row[8],
            last_accessed=row[9],
            content_depth=row[10],
            num_relationships=row[11],
            quality_score=row[12],
            collision_chain=json.loads(row[13]),
            is_primary=bool(row[14]),
            merged_from=json.loads(row[15]),
            tags=set(json.loads(row[16])),
            source_context=row[17],
            merge_history=json.loads(row[18])
        )
    
    def _increment_op_counter(self) -> None:
        """Increment operation counter and trigger maintenance if needed."""
        self._op_count += 1
        if self._op_count >= self.config.vacuum_interval:
            self._maintenance()
            self._op_count = 0
    
    def _maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        self._save_bloom_filter()
        # Vacuum database for performance
        try:
            with self._get_db() as conn:
                conn.execute("VACUUM")
        except Exception as e:
            logger.warning(f"Vacuum failed: {e}")
    
    # =========================================================================
    # Core API Methods
    # =========================================================================
    
    def add(self, memory_id: str, content: Union[str, bytes],
            metadata: Optional[Dict] = None) -> Tuple[bool, Optional[HashEntry]]:
        """
        Add a new memory to the hash index.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Memory content to hash
            metadata: Optional metadata (tags, relationships, etc.)
            
        Returns:
            Tuple of (is_duplicate, existing_or_merged_entry)
            - If not duplicate: (False, None)
            - If duplicate: (True, merged_entry)
        """
        metadata = metadata or {}
        
        # Compute all hashes
        hashes = compute_combined_hash(content)
        
        # Check bloom filter first (fast path)
        bloom_hit = self.bloom_filter.contains(hashes['md5'])
        
        # Create entry
        entry = HashEntry(
            memory_id=memory_id,
            exact_hash=hashes['md5'],
            sha256_hash=hashes['sha256'],
            simhash=hashes['simhash'],
            minhash=hashes['minhash'],
            content_size=len(content) if isinstance(content, bytes) else len(content),
            tags=set(metadata.get('tags', [])),
            num_relationships=metadata.get('num_relationships', 0),
            quality_score=metadata.get('quality_score', 0.0),
            source_context=metadata.get('source_context', '')
        )
        entry.compute_content_depth(content.decode('utf-8', errors='ignore') 
                                    if isinstance(content, bytes) else content)
        
        with self._get_db() as conn:
            # Check for exact duplicates
            cursor = conn.execute(
                "SELECT * FROM hash_entries WHERE exact_hash = ? OR sha256_hash = ?",
                (entry.exact_hash, entry.sha256_hash)
            )
            exact_dup = cursor.fetchone()
            
            if exact_dup:
                existing = self._row_to_entry(exact_dup)
                existing.update_access()
                self._update_entry_in_db(conn, existing)
                self._increment_op_counter()
                return True, existing
            
            # If bloom filter hit, check for near-duplicates
            if bloom_hit or not self.config.auto_merge_enabled:
                duplicates = self._find_near_duplicates_internal(
                    conn, entry, 
                    self.config.simhash_threshold,
                    self.config.minhash_threshold
                )
                
                if duplicates:
                    if self.config.auto_merge_enabled:
                        # Merge duplicates
                        merged = self.merger.merge_entries(entry, duplicates)
                        self._save_entry_to_db(conn, merged)
                        
                        # Mark duplicates as non-primary
                        for dup in duplicates:
                            conn.execute(
                                "UPDATE hash_entries SET is_primary = 0 WHERE memory_id = ?",
                                (dup.memory_id,)
                            )
                        
                        conn.commit()
                        self._increment_op_counter()
                        return True, merged
                    else:
                        # Just return the most similar duplicate
                        self._save_entry_to_db(conn, entry)
                        self._increment_op_counter()
                        return True, duplicates[0]
            
            # No duplicates found - add new entry
            self._save_entry_to_db(conn, entry)
            self.bloom_filter.add(entry.exact_hash)
            
            # Update in-memory index
            self._simhash_index[int(entry.simhash)].append(entry.memory_id)
            self._cache_entry(entry)
            
            conn.commit()
            self._increment_op_counter()
            return False, None
    
    def _save_entry_to_db(self, conn: sqlite3.Connection, entry: HashEntry) -> None:
        """Save entry to database."""
        conn.execute("""
            INSERT OR REPLACE INTO hash_entries
            (memory_id, exact_hash, sha256_hash, simhash, minhash, content_size,
             created_at, updated_at, access_count, last_accessed, content_depth,
             num_relationships, quality_score, collision_chain, is_primary,
             merged_from, tags, source_context, merge_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._entry_to_row(entry))
    
    def _update_entry_in_db(self, conn: sqlite3.Connection, entry: HashEntry) -> None:
        """Update entry in database."""
        self._save_entry_to_db(conn, entry)
    
    def _cache_entry(self, entry: HashEntry) -> None:
        """Add entry to LRU cache."""
        if len(self._entry_cache) >= self.config.cache_size:
            # Simple eviction: clear half the cache
            keys = list(self._entry_cache.keys())
            for key in keys[:len(keys)//2]:
                del self._entry_cache[key]
        
        self._entry_cache[entry.memory_id] = entry
    
    def get(self, memory_id: str) -> Optional[HashEntry]:
        """Get a hash entry by memory ID."""
        # Check cache first
        if memory_id in self._entry_cache:
            entry = self._entry_cache[memory_id]
            entry.update_access()
            return entry
        
        # Query database
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM hash_entries WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            if row:
                entry = self._row_to_entry(row)
                self._cache_entry(entry)
                entry.update_access()
                return entry
        
        return None
    
    def find_exact_duplicates(self, content: Union[str, bytes]) -> List[HashEntry]:
        """Find exact duplicates by content hash."""
        hashes = compute_combined_hash(content)
        
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM hash_entries WHERE exact_hash = ? OR sha256_hash = ?",
                (hashes['md5'], hashes['sha256'])
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]
    
    def find_near_duplicates(self, content: Union[str, bytes],
                            simhash_threshold: Optional[int] = None,
                            minhash_threshold: Optional[float] = None) -> List[HashEntry]:
        """
        Find near-duplicate entries for given content.
        
        Uses SimHash Hamming distance and MinHash Jaccard similarity.
        """
        hashes = compute_combined_hash(content)
        
        temp_entry = HashEntry(
            memory_id="temp",
            exact_hash=hashes['md5'],
            sha256_hash=hashes['sha256'],
            simhash=hashes['simhash'],
            minhash=hashes['minhash']
        )
        
        with self._get_db() as conn:
            return self._find_near_duplicates_internal(
                conn, temp_entry,
                simhash_threshold or self.config.simhash_threshold,
                minhash_threshold or self.config.minhash_threshold
            )
    
    def _find_near_duplicates_internal(self, conn: sqlite3.Connection,
                                       entry: HashEntry,
                                       simhash_threshold: int,
                                       minhash_threshold: float) -> List[HashEntry]:
        """Internal method to find near-duplicates within a transaction."""
        duplicates = []
        
        # Query by SimHash range (simplified - in production use Hamming distance index)
        cursor = conn.execute(
            "SELECT * FROM hash_entries WHERE is_primary = 1"
        )
        
        for row in cursor.fetchall():
            other = self._row_to_entry(row)
            
            # Skip self
            if other.memory_id == entry.memory_id:
                continue
            
            # Check SimHash distance
            sim_distance = entry.simhash_distance(other)
            if sim_distance <= simhash_threshold:
                duplicates.append(other)
                continue
            
            # Check MinHash similarity
            if entry.minhash and other.minhash:
                min_sim = entry.minhash_similarity(other)
                if min_sim >= minhash_threshold:
                    duplicates.append(other)
        
        return duplicates
    
    def check_exists(self, content: Union[str, bytes]) -> bool:
        """
        Fast existence check using bloom filter.
        
        May return false positives but never false negatives.
        """
        hashes = compute_combined_hash(content)
        return self.bloom_filter.contains(hashes['md5'])
    
    def delete(self, memory_id: str) -> bool:
        """Delete a hash entry."""
        with self._get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM hash_entries WHERE memory_id = ?",
                (memory_id,)
            )
            conn.commit()
            
            # Remove from cache and index
            if memory_id in self._entry_cache:
                del self._entry_cache[memory_id]
            
            # Note: Cannot remove from bloom filter, but that's okay
            # as bloom filters are designed to handle this
            
            self._increment_op_counter()
            return cursor.rowcount > 0
    
    def update_metadata(self, memory_id: str, 
                       metadata: Dict[str, Any]) -> Optional[HashEntry]:
        """Update metadata for an existing entry."""
        entry = self.get(memory_id)
        if not entry:
            return None
        
        # Update fields
        if 'tags' in metadata:
            entry.tags.update(metadata['tags'])
        if 'num_relationships' in metadata:
            entry.num_relationships = metadata['num_relationships']
        if 'quality_score' in metadata:
            entry.quality_score = metadata['quality_score']
        
        entry.updated_at = time.time()
        
        with self._get_db() as conn:
            self._update_entry_in_db(conn, entry)
            conn.commit()
        
        self._cache_entry(entry)
        return entry
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def add_batch(self, items: List[Tuple[str, Union[str, bytes], Optional[Dict]]]
                 ) -> List[Tuple[bool, Optional[HashEntry]]]:
        """
        Add multiple memories in a batch operation.
        
        Args:
            items: List of (memory_id, content, metadata) tuples
            
        Returns:
            List of results matching the input order
        """
        results = []
        
        for memory_id, content, metadata in items:
            result = self.add(memory_id, content, metadata)
            results.append(result)
            
            # Commit in batches
            if len(results) % self.config.batch_size == 0:
                self._maintenance()
        
        return results
    
    def find_duplicates_batch(self, contents: List[Union[str, bytes]]
                             ) -> List[List[HashEntry]]:
        """Find duplicates for multiple contents efficiently."""
        return [self.find_near_duplicates(content) for content in contents]
    
    # =========================================================================
    # Deduplication Operations
    # =========================================================================
    
    def run_deduplication(self) -> Dict[str, Any]:
        """
        Run full deduplication pass over all entries.
        
        Scans all entries and merges any duplicates found.
        Returns statistics about the operation.
        """
        stats = {
            'scanned': 0,
            'duplicates_found': 0,
            'merged': 0
        }
        
        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM hash_entries WHERE is_primary = 1")
            entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        
        processed_ids = set()
        
        for entry in entries:
            if entry.memory_id in processed_ids:
                continue
            
            stats['scanned'] += 1
            
            # Find duplicates
            with self._get_db() as conn:
                duplicates = self._find_near_duplicates_internal(
                    conn, entry,
                    self.config.simhash_threshold,
                    self.config.minhash_threshold
                )
            
            if duplicates:
                # Filter out already processed
                duplicates = [d for d in duplicates if d.memory_id not in processed_ids]
                
                if duplicates:
                    stats['duplicates_found'] += len(duplicates)
                    
                    # Merge
                    merged = self.merger.merge_entries(entry, duplicates)
                    
                    with self._get_db() as conn:
                        self._save_entry_to_db(conn, merged)
                        
                        # Mark duplicates as non-primary
                        for dup in duplicates:
                            conn.execute(
                                "UPDATE hash_entries SET is_primary = 0 WHERE memory_id = ?",
                                (dup.memory_id,)
                            )
                            processed_ids.add(dup.memory_id)
                        
                        conn.commit()
                    
                    stats['merged'] += 1
            
            processed_ids.add(entry.memory_id)
        
        self._maintenance()
        return stats
    
    def get_duplicate_clusters(self) -> List[List[HashEntry]]:
        """Get clusters of duplicate entries."""
        clusters = []
        processed = set()
        
        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM hash_entries WHERE is_primary = 1")
            primaries = [self._row_to_entry(row) for row in cursor.fetchall()]
        
        for entry in primaries:
            if entry.memory_id in processed:
                continue
            
            cluster = [entry]
            processed.add(entry.memory_id)
            
            # Get all merged entries
            if entry.merged_from:
                with self._get_db() as conn:
                    for merged_id in entry.merged_from:
                        merged_entry = self.get(merged_id)
                        if merged_entry:
                            cluster.append(merged_entry)
                            processed.add(merged_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    # =========================================================================
    # Statistics and Reporting
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hash index."""
        with self._get_db() as conn:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM hash_entries")
            total_entries = cursor.fetchone()[0]
            
            # Primary entries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM hash_entries WHERE is_primary = 1"
            )
            primary_entries = cursor.fetchone()[0]
            
            # Merged entries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM hash_entries WHERE is_primary = 0"
            )
            merged_entries = cursor.fetchone()[0]
            
            # Bloom filter stats
            bloom_stats = {
                'estimated_items': self.bloom_filter.item_count,
                'false_positive_rate': self.bloom_filter.estimated_false_positive_rate()
            }
        
        return {
            'total_entries': total_entries,
            'primary_entries': primary_entries,
            'merged_entries': merged_entries,
            'deduplication_ratio': merged_entries / total_entries if total_entries > 0 else 0,
            'bloom_filter': bloom_stats,
            'merge_stats': self.merger.get_merge_stats(),
            'config': {
                'simhash_threshold': self.config.simhash_threshold,
                'minhash_threshold': self.config.minhash_threshold
            }
        }
    
    def clear(self) -> None:
        """Clear all entries from the hash index."""
        with self._get_db() as conn:
            conn.execute("DELETE FROM hash_entries")
            conn.execute("DELETE FROM bloom_filter")
            conn.commit()
        
        self.bloom_filter.clear()
        self._entry_cache.clear()
        self._simhash_index.clear()
        self.merger.reset_stats()
    
    def close(self) -> None:
        """Close the hash index and save state."""
        self._maintenance()
        self._save_bloom_filter()


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_similarity_hash(content: Union[str, bytes], 
                           algorithm: str = 'simhash') -> Union[int, List[int]]:
    """
    Compute similarity hash for content.
    
    Args:
        content: Content to hash
        algorithm: 'simhash' or 'minhash'
        
    Returns:
        SimHash as int or MinHash as list of ints
    """
    if algorithm == 'simhash':
        return compute_simhash(content)
    elif algorithm == 'minhash':
        return compute_minhash(content)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def find_near_duplicates(index: HashIndex, content: Union[str, bytes],
                        threshold: Optional[float] = None) -> List[HashEntry]:
    """
    Find near-duplicate memories within threshold.
    
    Convenience wrapper around HashIndex.find_near_duplicates().
    """
    return index.find_near_duplicates(content, threshold)


def create_hash_index(db_path: Optional[str] = None,
                     config: Optional[HashIndexConfig] = None) -> HashIndex:
    """
    Factory function to create a configured HashIndex.
    
    Args:
        db_path: Optional database path (overrides config)
        config: Optional configuration object
        
    Returns:
        Configured HashIndex instance
    """
    if config is None:
        config = HashIndexConfig()
    
    if db_path:
        config.db_path = db_path
    
    return HashIndex(config)


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    import tempfile
    import os
    
    # Create temporary database for testing
    temp_db = tempfile.mktemp(suffix='.db')
    
    # Create hash index
    config = HashIndexConfig(
        db_path=temp_db,
        simhash_threshold=3,
        minhash_threshold=0.85,
        auto_merge_enabled=True
    )
    
    index = HashIndex(config)
    
    # Add some memories
    memories = [
        ("mem1", "The quick brown fox jumps over the lazy dog", {"tags": ["animals"]}),
        ("mem2", "The quick brown fox jumped over the lazy dog", {"tags": ["animals"]}),  # Near-duplicate
        ("mem3", "Machine learning is a subset of artificial intelligence", {"tags": ["ai"]}),
        ("mem4", "Machine learning is part of artificial intelligence technology", {"tags": ["ai"]}),  # Near-duplicate
        ("mem5", "The quick brown fox jumps over the lazy dog", {"tags": ["duplicate"]}),  # Exact duplicate
    ]
    
    for mem_id, content, meta in memories:
        is_dup, existing = index.add(mem_id, content, meta)
        if is_dup:
            print(f"'{mem_id}' is duplicate of {existing.memory_id if existing else 'unknown'}")
        else:
            print(f"'{mem_id}' added successfully")
    
    # Get stats
    stats = index.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Primary entries: {stats['primary_entries']}")
    print(f"  Merged entries: {stats['merged_entries']}")
    print(f"  Deduplication ratio: {stats['deduplication_ratio']:.2%}")
    
    # Find near duplicates
    query = "The quick brown fox leaps over the lazy dog"
    near_dups = index.find_near_duplicates(query)
    print(f"\nNear-duplicates for query: '{query}'")
    for dup in near_dups:
        print(f"  - {dup.memory_id} (accessed {dup.access_count} times)")
    
    # Check existence
    exists = index.check_exists("The quick brown fox jumps over the lazy dog")
    print(f"\nExists check: {exists}")
    
    # Run full deduplication
    dedup_stats = index.run_deduplication()
    print(f"\nDeduplication pass: {dedup_stats}")
    
    # Get duplicate clusters
    clusters = index.get_duplicate_clusters()
    print(f"\nDuplicate clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {[e.memory_id for e in cluster]}")
    
    # Get merge stats
    merge_stats = index.merger.get_merge_stats()
    print(f"\nMerge statistics: {merge_stats}")
    
    # Cleanup
    index.close()
    
    # Remove temporary database
    try:
        os.unlink(temp_db)
    except:
        pass
    
    print("\nHash index closed successfully")
