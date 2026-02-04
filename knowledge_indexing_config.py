"""
Knowledge Indexing Configuration Module

Provides unified configuration for the 4-layer indexing system:
- Hierarchical Index: Multi-level memory organization with promotion/demotion
- Graph Index: Relationship-based knowledge representation
- Hash Index: Fast deduplication and exact matching
- Semantic Index: Vector-based similarity search

Plus Context Assembler: Intelligent context composition from all layers.

Usage:
    >>> from knowledge_indexing_config import get_default_config, load_config_from_yaml
    >>> config = get_default_config()
    >>> config = load_config_from_yaml("config.yaml")
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Try to import yaml, provide helpful error if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML functions will be unavailable.")


class IndexingPreset(Enum):
    """Configuration presets for different use cases."""
    MINIMAL = "minimal"           # Low resource usage, basic functionality
    BALANCED = "balanced"         # Default balanced configuration
    COMPREHENSIVE = "comprehensive"  # Maximum features, higher resource usage
    PERFORMANCE = "performance"   # Optimized for speed
    ACCURACY = "accuracy"         # Optimized for retrieval accuracy


@dataclass
class HierarchicalIndexConfig:
    """
    Configuration for hierarchical indexing.
    
    Manages multi-level memory organization with automatic promotion
    and demotion based on access patterns and relevance.
    
    Memory Levels:
        - Core: Critical, frequently accessed knowledge
        - Important: High-value knowledge with moderate access
        - Contextual: Situational, session-specific knowledge
        - Granular: Detailed, rarely accessed knowledge
    """
    # Promotion/Demotion thresholds
    promotion_threshold: float = 0.75
    """Score threshold for promoting memory to higher level (0.0-1.0)."""
    
    demotion_threshold: float = 0.25
    """Score threshold for demoting memory to lower level (0.0-1.0)."""
    
    # Decay configuration
    decay_half_life_days: float = 30.0
    """Number of days for memory score to decay by half."""
    
    decay_rate: float = 0.05
    """Daily decay rate for memory scores (alternative to half-life)."""
    
    # Access scoring weights
    access_frequency_weight: float = 0.4
    """Weight given to access frequency in scoring."""
    
    recency_weight: float = 0.3
    """Weight given to recency in scoring."""
    
    relevance_weight: float = 0.3
    """Weight given to computed relevance in scoring."""
    
    # Memory level limits
    max_core_memories: int = 100
    """Maximum number of memories in core level."""
    
    max_important_memories: int = 500
    """Maximum number of memories in important level."""
    
    max_contextual_memories: int = 1000
    """Maximum number of memories in contextual level."""
    
    max_granular_memories: int = 10000
    """Maximum number of memories in granular level."""
    
    # Auto-organization settings
    enable_auto_promotion: bool = True
    """Enable automatic promotion based on scores."""
    
    enable_auto_demotion: bool = True
    """Enable automatic demotion based on scores."""
    
    prune_on_overflow: bool = True
    """Remove lowest-scoring memories when level is full."""
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not 0.0 <= self.promotion_threshold <= 1.0:
            errors.append(f"promotion_threshold must be in [0, 1], got {self.promotion_threshold}")
        
        if not 0.0 <= self.demotion_threshold <= 1.0:
            errors.append(f"demotion_threshold must be in [0, 1], got {self.demotion_threshold}")
        
        if self.promotion_threshold <= self.demotion_threshold:
            errors.append("promotion_threshold must be greater than demotion_threshold")
        
        if self.decay_half_life_days <= 0:
            errors.append(f"decay_half_life_days must be positive, got {self.decay_half_life_days}")
        
        total_weight = self.access_frequency_weight + self.recency_weight + self.relevance_weight
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        for name, value in [
            ("max_core_memories", self.max_core_memories),
            ("max_important_memories", self.max_important_memories),
            ("max_contextual_memories", self.max_contextual_memories),
            ("max_granular_memories", self.max_granular_memories),
        ]:
            if value < 0:
                errors.append(f"{name} must be non-negative, got {value}")
        
        return errors


@dataclass
class GraphIndexConfig:
    """
    Configuration for graph indexing.
    
    Manages relationship-based knowledge representation with
    support for multiple relationship types and traversal strategies.
    """
    # Traversal settings
    max_traversal_depth: int = 5
    """Maximum depth for graph traversal operations."""
    
    min_edge_weight: float = 0.3
    """Minimum edge weight to consider in traversals."""
    
    default_edge_weight: float = 0.5
    """Default weight for new edges."""
    
    # Relationship extraction
    auto_extract_relationships: bool = True
    """Automatically extract relationships from content."""
    
    relationship_types: List[str] = field(default_factory=lambda: [
        "related_to", "part_of", "depends_on", "causes", "leads_to",
        "similar_to", "contradicts", "supports", "example_of", "instance_of"
    ])
    """List of allowed relationship types."""
    
    extract_semantic_relationships: bool = True
    """Use NLP to extract semantic relationships."""
    
    extract_temporal_relationships: bool = True
    """Extract before/after temporal relationships."""
    
    # Graph maintenance
    prune_disconnected_nodes: bool = False
    """Remove nodes with no connections."""
    
    min_node_connections: int = 1
    """Minimum connections to keep a node (if pruning enabled)."""
    
    edge_decay_enabled: bool = True
    """Enable edge weight decay over time."""
    
    edge_decay_rate: float = 0.01
    """Daily decay rate for edge weights."""
    
    # Query settings
    max_results_per_query: int = 100
    """Maximum nodes to return per graph query."""
    
    include_edge_metadata: bool = True
    """Include metadata (creation time, confidence, etc.) with edges."""
    
    bidirectional_traversal: bool = True
    """Traverse edges in both directions."""
    
    # Community detection
    enable_community_detection: bool = True
    """Enable automatic community/cluster detection."""
    
    community_detection_algorithm: str = "louvain"
    """Algorithm for community detection (louvain, leiden, etc.)."""
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.max_traversal_depth < 1:
            errors.append(f"max_traversal_depth must be >= 1, got {self.max_traversal_depth}")
        
        if not 0.0 <= self.min_edge_weight <= 1.0:
            errors.append(f"min_edge_weight must be in [0, 1], got {self.min_edge_weight}")
        
        if not 0.0 <= self.default_edge_weight <= 1.0:
            errors.append(f"default_edge_weight must be in [0, 1], got {self.default_edge_weight}")
        
        if self.min_edge_weight > self.default_edge_weight:
            errors.append("min_edge_weight should not exceed default_edge_weight")
        
        if not self.relationship_types:
            errors.append("relationship_types cannot be empty")
        
        if self.min_node_connections < 0:
            errors.append(f"min_node_connections must be non-negative, got {self.min_node_connections}")
        
        if not 0.0 <= self.edge_decay_rate <= 1.0:
            errors.append(f"edge_decay_rate must be in [0, 1], got {self.edge_decay_rate}")
        
        valid_algorithms = ["louvain", "leiden", "label_propagation", "walktrap"]
        if self.community_detection_algorithm not in valid_algorithms:
            errors.append(f"Invalid community_detection_algorithm: {self.community_detection_algorithm}")
        
        return errors


@dataclass
class HashIndexConfig:
    """
    Configuration for hash/deduplication indexing.
    
    Provides fast exact matching and near-duplicate detection
    using multiple hash algorithms and similarity measures.
    """
    # Similarity thresholds
    similarity_threshold: float = 0.85
    """Minimum similarity score to consider items as duplicates."""
    
    hamming_threshold: int = 3
    """Maximum Hamming distance for hash similarity."""
    
    exact_match_threshold: float = 1.0
    """Threshold for considering items as exact matches."""
    
    # Hash algorithms
    primary_hash_algorithm: str = "simhash"
    """Primary hash algorithm (simhash, minhash, perceptual)."""
    
    secondary_hash_algorithms: List[str] = field(default_factory=lambda: ["minhash"])
    """Secondary hash algorithms for cross-verification."""
    
    hash_bit_length: int = 64
    """Bit length for hash values (64 or 128 recommended)."""
    
    # Bloom filter settings
    enable_bloom_filter: bool = True
    """Enable Bloom filter for fast negative lookups."""
    
    bloom_filter_size: int = 1000000
    """Size of Bloom filter in bits."""
    
    bloom_filter_hash_count: int = 7
    """Number of hash functions for Bloom filter."""
    
    bloom_filter_false_positive_rate: float = 0.01
    """Target false positive rate for Bloom filter."""
    
    # Deduplication settings
    deduplication_enabled: bool = True
    """Enable automatic deduplication."""
    
    merge_similar_entries: bool = False
    """Merge entries that exceed similarity threshold."""
    
    keep_highest_scored_on_merge: bool = True
    """Keep the highest-scored entry when merging."""
    
    # Content preprocessing
    normalize_whitespace: bool = True
    """Normalize whitespace before hashing."""
    
    lowercase_before_hash: bool = True
    """Convert to lowercase before hashing."""
    
    remove_punctuation: bool = False
    """Remove punctuation before hashing."""
    
    max_content_length: int = 10000
    """Maximum content length to hash (truncate if longer)."""
    
    # Shingling for MinHash
    shingle_size: int = 5
    """Size of shingles (n-grams) for MinHash."""
    
    num_minhash_permutations: int = 128
    """Number of permutations for MinHash."""
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append(f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}")
        
        if self.hamming_threshold < 0:
            errors.append(f"hamming_threshold must be non-negative, got {self.hamming_threshold}")
        
        valid_algorithms = ["simhash", "minhash", "perceptual", "md5", "sha256"]
        if self.primary_hash_algorithm not in valid_algorithms:
            errors.append(f"Invalid primary_hash_algorithm: {self.primary_hash_algorithm}")
        
        for algo in self.secondary_hash_algorithms:
            if algo not in valid_algorithms:
                errors.append(f"Invalid secondary_hash_algorithm: {algo}")
        
        if self.hash_bit_length not in [32, 64, 128, 256]:
            errors.append(f"hash_bit_length should be 32, 64, 128, or 256, got {self.hash_bit_length}")
        
        if self.bloom_filter_size < 1000:
            errors.append(f"bloom_filter_size too small: {self.bloom_filter_size}")
        
        if self.bloom_filter_hash_count < 1:
            errors.append(f"bloom_filter_hash_count must be >= 1, got {self.bloom_filter_hash_count}")
        
        if not 0.0 < self.bloom_filter_false_positive_rate < 1.0:
            errors.append(f"bloom_filter_false_positive_rate must be in (0, 1), got {self.bloom_filter_false_positive_rate}")
        
        if self.shingle_size < 1:
            errors.append(f"shingle_size must be >= 1, got {self.shingle_size}")
        
        if self.num_minhash_permutations < 16:
            errors.append(f"num_minhash_permutations too small: {self.num_minhash_permutations}")
        
        return errors


@dataclass
class SemanticIndexConfig:
    """
    Configuration for semantic indexing.
    
    Manages vector-based similarity search using embeddings
    with support for multiple embedding models and vector stores.
    """
    # Embedding model configuration
    embedding_model: str = "text-embedding-3-small"
    """Name of embedding model to use."""
    
    embedding_dimension: int = 1536
    """Dimension of embedding vectors."""
    
    embedding_provider: str = "openai"
    """Provider for embeddings (openai, local, huggingface, cohere)."""
    
    local_fallback: bool = True
    """Fallback to local model if primary provider fails."""
    
    local_model_name: str = "all-MiniLM-L6-v2"
    """Local model name for fallback."""
    
    # Similarity settings
    similarity_threshold: float = 0.7
    """Minimum cosine similarity for retrieval."""
    
    similarity_metric: str = "cosine"
    """Similarity metric (cosine, euclidean, dot_product)."""
    
    # Vector store configuration
    vector_store_type: str = "qdrant"
    """Vector store backend (qdrant, chroma, pinecone, weaviate)."""
    
    vector_store_url: Optional[str] = None
    """URL for vector store connection."""
    
    vector_store_collection: str = "knowledge_index"
    """Collection name in vector store."""
    
    # Indexing parameters
    batch_size: int = 100
    """Batch size for embedding operations."""
    
    max_text_length: int = 8000
    """Maximum text length to embed (truncate if longer)."""
    
    chunk_overlap: int = 200
    """Overlap between text chunks."""
    
    enable_chunking: bool = True
    """Enable automatic text chunking."""
    
    # Query settings
    top_k_default: int = 10
    """Default number of results to return."""
    
    max_top_k: int = 100
    """Maximum allowed top_k value."""
    
    enable_hybrid_search: bool = True
    """Enable hybrid (sparse + dense) search."""
    
    sparse_weight: float = 0.3
    """Weight for sparse retrieval in hybrid search."""
    
    dense_weight: float = 0.7
    """Weight for dense retrieval in hybrid search."""
    
    # Reranking
    enable_reranking: bool = True
    """Enable result reranking."""
    
    reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """Model for reranking results."""
    
    # Caching
    cache_embeddings: bool = True
    """Cache computed embeddings."""
    
    cache_ttl_hours: float = 24.0
    """Time-to-live for cached embeddings in hours."""
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append(f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}")
        
        valid_metrics = ["cosine", "euclidean", "dot_product", "manhattan"]
        if self.similarity_metric not in valid_metrics:
            errors.append(f"Invalid similarity_metric: {self.similarity_metric}")
        
        valid_providers = ["openai", "local", "huggingface", "cohere", "google", "azure"]
        if self.embedding_provider not in valid_providers:
            errors.append(f"Invalid embedding_provider: {self.embedding_provider}")
        
        valid_stores = ["qdrant", "chroma", "pinecone", "weaviate", "milvus", "faiss"]
        if self.vector_store_type not in valid_stores:
            errors.append(f"Invalid vector_store_type: {self.vector_store_type}")
        
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.top_k_default < 1:
            errors.append(f"top_k_default must be >= 1, got {self.top_k_default}")
        
        if self.max_top_k < self.top_k_default:
            errors.append(f"max_top_k ({self.max_top_k}) must be >= top_k_default ({self.top_k_default})")
        
        if not 0.0 <= self.sparse_weight <= 1.0:
            errors.append(f"sparse_weight must be in [0, 1], got {self.sparse_weight}")
        
        if not 0.0 <= self.dense_weight <= 1.0:
            errors.append(f"dense_weight must be in [0, 1], got {self.dense_weight}")
        
        if abs(self.sparse_weight + self.dense_weight - 1.0) > 0.01:
            errors.append("sparse_weight + dense_weight must equal 1.0")
        
        return errors


@dataclass
class ContextAssemblerConfig:
    """
    Configuration for context assembly pipeline.
    
    Orchestrates the composition of context from all indexing layers
    with intelligent token budgeting and relevance weighting.
    """
    # Token budget configuration
    max_tokens: int = 4000
    """Maximum total tokens for assembled context."""
    
    target_tokens: int = 3500
    """Target token count (leaves buffer for safety)."""
    
    # Memory level allocation ratios (must sum to 1.0)
    core_memory_ratio: float = 0.3
    """Token ratio allocated to core memories."""
    
    important_memory_ratio: float = 0.4
    """Token ratio allocated to important memories."""
    
    contextual_memory_ratio: float = 0.2
    """Token ratio allocated to contextual memories."""
    
    granular_memory_ratio: float = 0.1
    """Token ratio allocated to granular memories."""
    
    # Assembly strategy
    assembly_strategy: str = "weighted_relevance"
    """Strategy for assembling context (weighted_relevance, chronological, hierarchical)."""
    
    deduplicate_across_sources: bool = True
    """Remove duplicates when combining from multiple sources."""
    
    prioritize_recent: bool = True
    """Give higher priority to recent memories."""
    
    recency_boost_hours: float = 24.0
    """Time window for recency boost."""
    
    # Content formatting
    include_metadata: bool = True
    """Include metadata (timestamps, scores, sources) in context."""
    
    include_relevance_scores: bool = False
    """Include relevance scores in assembled context."""
    
    format_as_xml: bool = False
    """Format output as XML structure."""
    
    format_as_json: bool = False
    """Format output as JSON structure."""
    
    separator: str = "\n\n---\n\n"
    """Separator between context sections."""
    
    # Cross-index fusion
    enable_cross_index_fusion: bool = True
    """Enable fusion of results from all indexes."""
    
    hierarchical_weight: float = 0.25
    """Weight for hierarchical index results."""
    
    graph_weight: float = 0.25
    """Weight for graph index results."""
    
    hash_weight: float = 0.15
    """Weight for hash index results."""
    
    semantic_weight: float = 0.35
    """Weight for semantic index results."""
    
    # Adaptive assembly
    enable_adaptive_assembly: bool = True
    """Dynamically adjust ratios based on query."""
    
    query_complexity_threshold: float = 0.5
    """Threshold for triggering complex query handling."""
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.max_tokens < 100:
            errors.append(f"max_tokens too small: {self.max_tokens}")
        
        if self.target_tokens > self.max_tokens:
            errors.append(f"target_tokens ({self.target_tokens}) must be <= max_tokens ({self.max_tokens})")
        
        total_ratio = (
            self.core_memory_ratio + 
            self.important_memory_ratio + 
            self.contextual_memory_ratio + 
            self.granular_memory_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            errors.append(f"Memory ratios must sum to 1.0, got {total_ratio}")
        
        for name, value in [
            ("core_memory_ratio", self.core_memory_ratio),
            ("important_memory_ratio", self.important_memory_ratio),
            ("contextual_memory_ratio", self.contextual_memory_ratio),
            ("granular_memory_ratio", self.granular_memory_ratio),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be in [0, 1], got {value}")
        
        valid_strategies = ["weighted_relevance", "chronological", "hierarchical", "round_robin"]
        if self.assembly_strategy not in valid_strategies:
            errors.append(f"Invalid assembly_strategy: {self.assembly_strategy}")
        
        total_cross_weight = (
            self.hierarchical_weight + 
            self.graph_weight + 
            self.hash_weight + 
            self.semantic_weight
        )
        if abs(total_cross_weight - 1.0) > 0.01:
            errors.append(f"Cross-index weights must sum to 1.0, got {total_cross_weight}")
        
        return errors


@dataclass
class UnifiedIndexingConfig:
    """
    Master configuration combining all 4 indexing layers.
    
    Provides a unified interface for configuring the complete
    knowledge indexing system with global settings and feature flags.
    """
    # Sub-configurations
    hierarchical: HierarchicalIndexConfig = field(default_factory=HierarchicalIndexConfig)
    """Hierarchical indexing configuration."""
    
    graph: GraphIndexConfig = field(default_factory=GraphIndexConfig)
    """Graph indexing configuration."""
    
    hash: HashIndexConfig = field(default_factory=HashIndexConfig)
    """Hash/deduplication indexing configuration."""
    
    semantic: SemanticIndexConfig = field(default_factory=SemanticIndexConfig)
    """Semantic indexing configuration."""
    
    assembler: ContextAssemblerConfig = field(default_factory=ContextAssemblerConfig)
    """Context assembler configuration."""
    
    # Global settings
    config_name: str = "default"
    """Name identifier for this configuration."""
    
    config_version: str = "1.0.0"
    """Version of this configuration."""
    
    # Feature flags
    enable_hierarchical_index: bool = True
    """Enable hierarchical memory indexing."""
    
    enable_graph_index: bool = True
    """Enable graph-based indexing."""
    
    enable_hash_index: bool = True
    """Enable hash-based deduplication."""
    
    enable_semantic_index: bool = True
    """Enable semantic/vector indexing."""
    
    enable_parallel_indexing: bool = True
    """Enable parallel indexing across layers."""
    
    # Performance settings
    indexing_batch_size: int = 100
    """Batch size for indexing operations."""
    
    query_timeout_seconds: float = 5.0
    """Timeout for query operations."""
    
    max_concurrent_queries: int = 10
    """Maximum concurrent queries allowed."""
    
    # Persistence
    auto_save_interval_minutes: float = 5.0
    """Interval for auto-saving indexes."""
    
    persist_to_disk: bool = True
    """Enable persistence to disk."""
    
    persistence_path: str = "./knowledge_indexes"
    """Path for persistence files."""
    
    # Logging and monitoring
    log_index_operations: bool = False
    """Log detailed indexing operations."""
    
    collect_metrics: bool = True
    """Collect performance metrics."""
    
    def validate(self) -> List[str]:
        """Validate entire configuration and return list of errors."""
        errors = []
        
        # Validate sub-configs
        errors.extend([f"hierarchical: {e}" for e in self.hierarchical.validate()])
        errors.extend([f"graph: {e}" for e in self.graph.validate()])
        errors.extend([f"hash: {e}" for e in self.hash.validate()])
        errors.extend([f"semantic: {e}" for e in self.semantic.validate()])
        errors.extend([f"assembler: {e}" for e in self.assembler.validate()])
        
        # Validate global settings
        if self.indexing_batch_size < 1:
            errors.append(f"indexing_batch_size must be >= 1, got {self.indexing_batch_size}")
        
        if self.query_timeout_seconds <= 0:
            errors.append(f"query_timeout_seconds must be positive, got {self.query_timeout_seconds}")
        
        if self.max_concurrent_queries < 1:
            errors.append(f"max_concurrent_queries must be >= 1, got {self.max_concurrent_queries}")
        
        # Check that at least one index is enabled
        if not any([
            self.enable_hierarchical_index,
            self.enable_graph_index,
            self.enable_hash_index,
            self.enable_semantic_index,
        ]):
            errors.append("At least one index type must be enabled")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


def get_default_config() -> UnifiedIndexingConfig:
    """
    Get default configuration with sensible defaults.
    
    Returns:
        UnifiedIndexingConfig: Default balanced configuration
        
    Example:
        >>> config = get_default_config()
        >>> config.hierarchical.promotion_threshold
        0.75
    """
    return UnifiedIndexingConfig()


def get_preset_config(preset: Union[IndexingPreset, str]) -> UnifiedIndexingConfig:
    """
    Get configuration for a specific preset.
    
    Args:
        preset: One of "minimal", "balanced", "comprehensive", 
                "performance", or "accuracy"
    
    Returns:
        UnifiedIndexingConfig: Configuration for the specified preset
        
    Example:
        >>> config = get_preset_config("minimal")
        >>> config.semantic.embedding_model
        'text-embedding-3-small'
    """
    if isinstance(preset, str):
        preset = IndexingPreset(preset.lower())
    
    config = UnifiedIndexingConfig()
    
    if preset == IndexingPreset.MINIMAL:
        # Reduce resource usage
        config.hierarchical.max_core_memories = 50
        config.hierarchical.max_important_memories = 200
        config.hierarchical.max_contextual_memories = 500
        config.hierarchical.max_granular_memories = 2000
        config.graph.max_traversal_depth = 3
        config.hash.enable_bloom_filter = False
        config.semantic.local_fallback = True
        config.semantic.embedding_model = "all-MiniLM-L6-v2"
        config.semantic.embedding_dimension = 384
        config.assembler.max_tokens = 2000
        config.enable_parallel_indexing = False
        
    elif preset == IndexingPreset.COMPREHENSIVE:
        # Maximum features
        config.hierarchical.max_core_memories = 500
        config.hierarchical.max_important_memories = 2000
        config.hierarchical.max_contextual_memories = 5000
        config.hierarchical.max_granular_memories = 50000
        config.graph.max_traversal_depth = 10
        config.graph.enable_community_detection = True
        config.hash.enable_bloom_filter = True
        config.hash.bloom_filter_size = 10000000
        config.semantic.enable_hybrid_search = True
        config.semantic.enable_reranking = True
        config.assembler.max_tokens = 8000
        config.assembler.enable_adaptive_assembly = True
        
    elif preset == IndexingPreset.PERFORMANCE:
        # Optimize for speed
        config.hierarchical.promotion_threshold = 0.8
        config.hierarchical.demotion_threshold = 0.2
        config.graph.max_traversal_depth = 3
        config.graph.max_results_per_query = 50
        config.hash.primary_hash_algorithm = "md5"
        config.semantic.similarity_threshold = 0.6
        config.semantic.batch_size = 500
        config.semantic.top_k_default = 5
        config.semantic.enable_reranking = False
        config.assembler.assembly_strategy = "round_robin"
        config.query_timeout_seconds = 2.0
        
    elif preset == IndexingPreset.ACCURACY:
        # Optimize for retrieval accuracy
        config.hierarchical.promotion_threshold = 0.6
        config.hierarchical.demotion_threshold = 0.4
        config.graph.max_traversal_depth = 7
        config.graph.min_edge_weight = 0.1
        config.hash.similarity_threshold = 0.75
        config.semantic.similarity_threshold = 0.8
        config.semantic.enable_hybrid_search = True
        config.semantic.sparse_weight = 0.4
        config.semantic.dense_weight = 0.6
        config.semantic.enable_reranking = True
        config.assembler.assembly_strategy = "weighted_relevance"
        config.assembler.deduplicate_across_sources = True
        config.query_timeout_seconds = 10.0
    
    # BALANCED uses defaults
    config.config_name = preset.value
    return config


def load_config_from_yaml(path: Union[str, Path]) -> UnifiedIndexingConfig:
    """
    Load configuration from YAML file.
    
    Expected YAML structure:
        knowledge_indexing:
          hierarchical:
            promotion_threshold: 0.75
            ...
          graph:
            max_traversal_depth: 5
            ...
          hash:
            similarity_threshold: 0.85
            ...
          semantic:
            embedding_model: "text-embedding-3-small"
            ...
          assembler:
            max_tokens: 4000
            ...
    
    Args:
        path: Path to YAML configuration file
    
    Returns:
        UnifiedIndexingConfig: Loaded configuration
        
    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If file does not exist
        ValueError: If YAML is malformed
        
    Example:
        >>> config = load_config_from_yaml("config.yaml")
        >>> print(config.hierarchical.promotion_threshold)
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML configuration. "
            "Install with: pip install pyyaml"
        )
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    # Extract knowledge_indexing section or use root
    config_data = data.get('knowledge_indexing', data)
    
    # Create sub-configs from YAML data
    hierarchical = HierarchicalIndexConfig(**config_data.get('hierarchical', {}))
    graph = GraphIndexConfig(**config_data.get('graph', {}))
    hash_config = HashIndexConfig(**config_data.get('hash', {}))
    semantic = SemanticIndexConfig(**config_data.get('semantic', {}))
    assembler = ContextAssemblerConfig(**config_data.get('assembler', {}))
    
    # Create unified config
    config = UnifiedIndexingConfig(
        hierarchical=hierarchical,
        graph=graph,
        hash=hash_config,
        semantic=semantic,
        assembler=assembler,
    )
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    logger.info(f"Loaded configuration from {path}")
    return config


def save_config_to_yaml(config: UnifiedIndexingConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Path for YAML output file
        
    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If configuration is invalid
        
    Example:
        >>> config = get_default_config()
        >>> save_config_to_yaml(config, "config.yaml")
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML configuration. "
            "Install with: pip install pyyaml"
        )
    
    if not config.is_valid():
        errors = config.validate()
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to nested dict structure
    data = {
        'knowledge_indexing': {
            'hierarchical': asdict(config.hierarchical),
            'graph': asdict(config.graph),
            'hash': asdict(config.hash),
            'semantic': asdict(config.semantic),
            'assembler': asdict(config.assembler),
        }
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {path}")


def _apply_env_overrides(config: UnifiedIndexingConfig) -> UnifiedIndexingConfig:
    """
    Apply environment variable overrides to configuration.
    
    Supported environment variables:
        KNOWLEDGE_INDEX_PROMOTION_THRESHOLD
        KNOWLEDGE_INDEX_DEMOTION_THRESHOLD
        KNOWLEDGE_INDEX_DECAY_HALF_LIFE_DAYS
        KNOWLEDGE_INDEX_GRAPH_MAX_DEPTH
        KNOWLEDGE_INDEX_HASH_SIMILARITY_THRESHOLD
        KNOWLEDGE_INDEX_SEMANTIC_MODEL
        KNOWLEDGE_INDEX_SEMANTIC_SIMILARITY_THRESHOLD
        KNOWLEDGE_INDEX_ASSEMBLER_MAX_TOKENS
        KNOWLEDGE_INDEX_ENABLE_HIERARCHICAL
        KNOWLEDGE_INDEX_ENABLE_GRAPH
        KNOWLEDGE_INDEX_ENABLE_HASH
        KNOWLEDGE_INDEX_ENABLE_SEMANTIC
    
    Args:
        config: Configuration to modify
        
    Returns:
        UnifiedIndexingConfig: Modified configuration
    """
    env_mappings = {
        # Hierarchical
        'KNOWLEDGE_INDEX_PROMOTION_THRESHOLD': ('hierarchical', 'promotion_threshold', float),
        'KNOWLEDGE_INDEX_DEMOTION_THRESHOLD': ('hierarchical', 'demotion_threshold', float),
        'KNOWLEDGE_INDEX_DECAY_HALF_LIFE_DAYS': ('hierarchical', 'decay_half_life_days', float),
        
        # Graph
        'KNOWLEDGE_INDEX_GRAPH_MAX_DEPTH': ('graph', 'max_traversal_depth', int),
        'KNOWLEDGE_INDEX_MIN_EDGE_WEIGHT': ('graph', 'min_edge_weight', float),
        
        # Hash
        'KNOWLEDGE_INDEX_HASH_SIMILARITY_THRESHOLD': ('hash', 'similarity_threshold', float),
        'KNOWLEDGE_INDEX_HAMMING_THRESHOLD': ('hash', 'hamming_threshold', int),
        'KNOWLEDGE_INDEX_ENABLE_BLOOM_FILTER': ('hash', 'enable_bloom_filter', bool),
        
        # Semantic
        'KNOWLEDGE_INDEX_SEMANTIC_MODEL': ('semantic', 'embedding_model', str),
        'KNOWLEDGE_INDEX_SEMANTIC_DIMENSION': ('semantic', 'embedding_dimension', int),
        'KNOWLEDGE_INDEX_SEMANTIC_SIMILARITY_THRESHOLD': ('semantic', 'similarity_threshold', float),
        'KNOWLEDGE_INDEX_SEMANTIC_TOP_K': ('semantic', 'top_k_default', int),
        'KNOWLEDGE_INDEX_ENABLE_HYBRID_SEARCH': ('semantic', 'enable_hybrid_search', bool),
        
        # Assembler
        'KNOWLEDGE_INDEX_ASSEMBLER_MAX_TOKENS': ('assembler', 'max_tokens', int),
        'KNOWLEDGE_INDEX_CORE_MEMORY_RATIO': ('assembler', 'core_memory_ratio', float),
        'KNOWLEDGE_INDEX_IMPORTANT_MEMORY_RATIO': ('assembler', 'important_memory_ratio', float),
        
        # Global feature flags
        'KNOWLEDGE_INDEX_ENABLE_HIERARCHICAL': (None, 'enable_hierarchical_index', bool),
        'KNOWLEDGE_INDEX_ENABLE_GRAPH': (None, 'enable_graph_index', bool),
        'KNOWLEDGE_INDEX_ENABLE_HASH': (None, 'enable_hash_index', bool),
        'KNOWLEDGE_INDEX_ENABLE_SEMANTIC': (None, 'enable_semantic_index', bool),
    }
    
    for env_var, (section, key, type_func) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                if type_func == bool:
                    parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    parsed_value = type_func(value)
                
                if section is None:
                    setattr(config, key, parsed_value)
                else:
                    subconfig = getattr(config, section)
                    setattr(subconfig, key, parsed_value)
                
                logger.debug(f"Applied env override: {env_var}={parsed_value}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse {env_var}={value}: {e}")
    
    return config


def create_default_yaml_config(path: Union[str, Path]) -> None:
    """
    Create a default YAML configuration file.
    
    This is a convenience function to generate a template
    configuration file that can be customized.
    
    Args:
        path: Path for output YAML file
        
    Example:
        >>> create_default_yaml_config("knowledge_indexing.yaml")
        >>> # Edit knowledge_indexing.yaml as needed
        >>> config = load_config_from_yaml("knowledge_indexing.yaml")
    """
    default_yaml = """# Knowledge Indexing Configuration
# 4-Layer Indexing System: Hierarchical, Graph, Hash, Semantic

knowledge_indexing:
  # Hierarchical Memory Index Configuration
  hierarchical:
    # Promotion/Demotion thresholds (0.0-1.0)
    promotion_threshold: 0.75
    demotion_threshold: 0.25
    
    # Decay configuration
    decay_half_life_days: 30.0
    decay_rate: 0.05
    
    # Scoring weights (must sum to 1.0)
    access_frequency_weight: 0.4
    recency_weight: 0.3
    relevance_weight: 0.3
    
    # Memory level limits
    max_core_memories: 100
    max_important_memories: 500
    max_contextual_memories: 1000
    max_granular_memories: 10000
    
    # Auto-organization
    enable_auto_promotion: true
    enable_auto_demotion: true
    prune_on_overflow: true

  # Graph Index Configuration
  graph:
    # Traversal settings
    max_traversal_depth: 5
    min_edge_weight: 0.3
    default_edge_weight: 0.5
    
    # Relationship extraction
    auto_extract_relationships: true
    relationship_types:
      - related_to
      - part_of
      - depends_on
      - causes
      - leads_to
      - similar_to
      - contradicts
      - supports
      - example_of
      - instance_of
    extract_semantic_relationships: true
    extract_temporal_relationships: true
    
    # Graph maintenance
    prune_disconnected_nodes: false
    min_node_connections: 1
    edge_decay_enabled: true
    edge_decay_rate: 0.01
    
    # Query settings
    max_results_per_query: 100
    include_edge_metadata: true
    bidirectional_traversal: true
    
    # Community detection
    enable_community_detection: true
    community_detection_algorithm: louvain

  # Hash/Deduplication Index Configuration
  hash:
    # Similarity thresholds
    similarity_threshold: 0.85
    hamming_threshold: 3
    exact_match_threshold: 1.0
    
    # Hash algorithms
    primary_hash_algorithm: simhash
    secondary_hash_algorithms:
      - minhash
    hash_bit_length: 64
    
    # Bloom filter settings
    enable_bloom_filter: true
    bloom_filter_size: 1000000
    bloom_filter_hash_count: 7
    bloom_filter_false_positive_rate: 0.01
    
    # Deduplication
    deduplication_enabled: true
    merge_similar_entries: false
    keep_highest_scored_on_merge: true
    
    # Content preprocessing
    normalize_whitespace: true
    lowercase_before_hash: true
    remove_punctuation: false
    max_content_length: 10000
    
    # Shingling for MinHash
    shingle_size: 5
    num_minhash_permutations: 128

  # Semantic/Vector Index Configuration
  semantic:
    # Embedding model
    embedding_model: text-embedding-3-small
    embedding_dimension: 1536
    embedding_provider: openai
    local_fallback: true
    local_model_name: all-MiniLM-L6-v2
    
    # Similarity settings
    similarity_threshold: 0.7
    similarity_metric: cosine
    
    # Vector store
    vector_store_type: qdrant
    vector_store_url: null
    vector_store_collection: knowledge_index
    
    # Indexing parameters
    batch_size: 100
    max_text_length: 8000
    chunk_overlap: 200
    enable_chunking: true
    
    # Query settings
    top_k_default: 10
    max_top_k: 100
    enable_hybrid_search: true
    sparse_weight: 0.3
    dense_weight: 0.7
    
    # Reranking
    enable_reranking: true
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
    
    # Caching
    cache_embeddings: true
    cache_ttl_hours: 24.0

  # Context Assembler Configuration
  assembler:
    # Token budget
    max_tokens: 4000
    target_tokens: 3500
    
    # Memory allocation ratios (must sum to 1.0)
    core_memory_ratio: 0.3
    important_memory_ratio: 0.4
    contextual_memory_ratio: 0.2
    granular_memory_ratio: 0.1
    
    # Assembly strategy
    assembly_strategy: weighted_relevance
    deduplicate_across_sources: true
    prioritize_recent: true
    recency_boost_hours: 24.0
    
    # Content formatting
    include_metadata: true
    include_relevance_scores: false
    format_as_xml: false
    format_as_json: false
    separator: "\\n\\n---\\n\\n"
    
    # Cross-index fusion
    enable_cross_index_fusion: true
    hierarchical_weight: 0.25
    graph_weight: 0.25
    hash_weight: 0.15
    semantic_weight: 0.35
    
    # Adaptive assembly
    enable_adaptive_assembly: true
    query_complexity_threshold: 0.5
"""
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(default_yaml)
    
    logger.info(f"Created default configuration at {path}")


# Export all public members
__all__ = [
    'IndexingPreset',
    'HierarchicalIndexConfig',
    'GraphIndexConfig',
    'HashIndexConfig',
    'SemanticIndexConfig',
    'ContextAssemblerConfig',
    'UnifiedIndexingConfig',
    'get_default_config',
    'get_preset_config',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'create_default_yaml_config',
]
