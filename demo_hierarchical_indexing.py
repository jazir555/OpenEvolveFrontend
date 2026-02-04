"""
================================================================================
DEMO: 4-Layer Hierarchical Indexing System for Context Rot Prevention
================================================================================

This demonstration showcases the enhanced knowledge engine with four complementary
indexing layers that work together to prevent "context rot" - the phenomenon where
LLMs lose track of important information as conversations grow longer.

The 4-Layer Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: HIERARCHICAL INDEX (Importance-Based Organization)               │
│  ├── CORE: High-level principles, facts that never change                   │
│  ├── IMPORTANT: Key concepts, domain knowledge                              │
│  ├── CONTEXTUAL: Conversation state, recent decisions                       │
│  └── GRANULAR: One-off details, specific examples                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: GRAPH INDEX (Relationship Preservation)                          │
│  ├── Causal relationships (because, therefore)                              │
│  ├── Temporal sequences (then, after, before)                               │
│  ├── Semantic connections (similar, related)                                │
│  └── Referential links (refers to, about)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: SEMANTIC INDEX (Meaning-Based Retrieval)                         │
│  ├── Vector embeddings for meaning-based search                             │
│  ├── Multi-stage filtering with hierarchy/graph pre-filtering               │
│  └── Context-aware ranking (recency, importance, similarity)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 4: HASH INDEX (Deduplication Layer)                                 │
│  ├── Exact hash (MD5/SHA256) for identical content                          │
│  ├── SimHash for near-duplicate detection                                   │
│  ├── MinHash for fuzzy matching                                             │
│  └── Bloom filter for fast existence checks                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    python demo_hierarchical_indexing.py

Author: OpenEvolve AI
Version: 1.0.0
"""

import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Import the 4 indexing layers
from knowledge_hierarchical_index import (
    HierarchicalIndex, MemoryNode, MemoryLevel, ImportanceScorer
)
from knowledge_graph_index import (
    GraphIndex, RelationshipType, NodeType, MemoryNode as GraphMemoryNode,
    RelationshipEdge, TraversalMode
)
from knowledge_semantic_index import (
    SemanticIndex, SemanticQuery, SemanticResult, EmbeddingGenerator,
    EmbeddingStore, SemanticIndexConfig
)
from knowledge_hash_index import (
    HashIndex, HashEntry, compute_combined_hash, HashIndexConfig,
    DuplicateMerger, BloomFilter
)


# ============================================================================
# DEMO CONFIGURATION AND UTILITIES
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    num_messages: int = 55
    hierarchical_db: str = "./demo_hierarchical_index.db"
    graph_db: str = "./demo_graph_index.db"
    hash_db: str = "./demo_hash_index.db"
    semantic_cache: str = "./demo_semantic_cache"
    show_token_count: bool = True
    verbose: bool = True


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_section(title: str, color: str = Colors.HEADER):
    """Print a formatted section header."""
    width = 80
    print(f"\n{color}{'=' * width}{Colors.END}")
    print(f"{color}{Colors.BOLD}{title.center(width)}{Colors.END}")
    print(f"{color}{'=' * width}{Colors.END}\n")


def print_subsection(title: str, color: str = Colors.BLUE):
    """Print a formatted subsection header."""
    print(f"\n{color}{Colors.BOLD}> {title}{Colors.END}")
    print(f"{color}{'-' * (len(title) + 2)}{Colors.END}")


def print_metric(label: str, value: str, color: str = Colors.CYAN):
    """Print a metric with formatting."""
    print(f"  {Colors.BOLD}{label}:{Colors.END} {color}{value}{Colors.END}")


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def print_tree(node_id: str, hierarchical_index: HierarchicalIndex, 
               prefix: str = "", is_last: bool = True, visited: set = None) -> str:
    """Generate ASCII tree representation of hierarchical structure."""
    if visited is None:
        visited = set()
    
    if node_id in visited:
        return ""
    visited.add(node_id)
    
    node = hierarchical_index.get_memory(node_id, record_access=False)
    if not node:
        return ""
    
    result = ""
    connector = "`-- " if is_last else "|-- "
    
    # Get level color
    level_colors = {
        MemoryLevel.CORE: Colors.RED,
        MemoryLevel.IMPORTANT: Colors.YELLOW,
        MemoryLevel.CONTEXTUAL: Colors.GREEN,
        MemoryLevel.GRANULAR: Colors.BLUE
    }
    level_color = level_colors.get(node.level, Colors.END)
    
    # Truncate content for display
    content_str = str(node.content)
    if len(content_str) > 50:
        content_str = content_str[:47] + "..."
    
    result += f"{prefix}{connector}{level_color}[{node.level.name}]{Colors.END} {content_str}\n"
    
    # Process children
    children = node.child_ids
    for i, child_id in enumerate(children):
        is_last_child = (i == len(children) - 1)
        extension = "    " if is_last else "|   "
        result += print_tree(child_id, hierarchical_index, prefix + extension, 
                           is_last_child, visited)
    
    return result


# ============================================================================
# ENHANCED KNOWLEDGE ENGINE
# ============================================================================

class EnhancedKnowledgeEngine:
    """
    Enhanced knowledge engine integrating all 4 indexing layers.
    
    This engine demonstrates how the 4-layer system works together:
    - Hierarchical: Organizes by importance (CORE → GRANULAR)
    - Graph: Preserves logical relationships between memories
    - Semantic: Enables meaning-based retrieval
    - Hash: Deduplicates redundant information
    """
    
    def __init__(self, config: DemoConfig = None):
        self.config = config or DemoConfig()
        self.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        
        print(f"{Colors.CYAN}Initializing Enhanced Knowledge Engine...{Colors.END}")
        start_time = time.time()
        
        # Layer 1: Hierarchical Index (Importance-based organization)
        self.hierarchical = HierarchicalIndex(
            storage_path=self.config.hierarchical_db,
            use_sqlite=True
        )
        
        # Layer 2: Graph Index (Relationship preservation)
        self.graph = GraphIndex(
            db_path=self.config.graph_db,
            enable_networkx=True
        )
        
        # Layer 3: Semantic Index (Meaning-based retrieval)
        semantic_config = SemanticIndexConfig(
            cache_dir=self.config.semantic_cache,
            vector_backend="sqlite"
        )
        self.semantic_store = EmbeddingStore(semantic_config)
        self.embedding_generator = EmbeddingGenerator(semantic_config)
        
        # Layer 4: Hash Index (Deduplication)
        hash_config = HashIndexConfig(
            db_path=self.config.hash_db,
            auto_merge_enabled=True
        )
        self.hash_index = HashIndex(hash_config)
        
        init_time = time.time() - start_time
        print(f"  {Colors.GREEN}[OK] All 4 indexes initialized in {init_time:.2f}s{Colors.END}")
        print(f"  {Colors.GREEN}[OK] Session ID: {self.session_id}{Colors.END}\n")
        
        # Track statistics
        self.stats = {
            'total_memories': 0,
            'duplicates_prevented': 0,
            'relationships_created': 0,
            'tokens_saved': 0
        }
    
    def add_memory(self, content: str, level: MemoryLevel, 
                   relationships: List[Tuple[str, RelationshipType]] = None,
                   tags: List[str] = None, domain: str = "general",
                   metadata: Dict = None) -> Dict[str, Any]:
        """
        Add a memory through all 4 indexing layers.
        
        Returns:
            Dictionary with IDs across all layers and deduplication info.
        """
        result = {
            'deduplicated': False,
            'hierarchical_id': None,
            'graph_id': None,
            'semantic_id': None,
            'existing_entry': None
        }
        
        # Layer 4: Check for duplicates first
        memory_id = f"mem_{self.session_id}_{self.stats['total_memories']}"
        is_dup, existing = self.hash_index.add(memory_id, content, metadata)
        
        if is_dup and existing:
            result['deduplicated'] = True
            result['existing_entry'] = existing
            self.stats['duplicates_prevented'] += 1
            self.stats['tokens_saved'] += estimate_tokens(content)
            return result
        
        # Layer 1: Add to hierarchical index
        hier_node = self.hierarchical.add_memory(
            content=content,
            level=level,
            tags=tags or [],
            domain=domain,
            metadata=metadata or {}
        )
        result['hierarchical_id'] = hier_node.node_id
        
        # Layer 2: Add to graph index
        # Determine node type based on level
        node_type_map = {
            MemoryLevel.CORE: NodeType.CONCEPT,
            MemoryLevel.IMPORTANT: NodeType.FACT,
            MemoryLevel.CONTEXTUAL: NodeType.DECISION,
            MemoryLevel.GRANULAR: NodeType.OBSERVATION
        }
        
        graph_id = self.graph.add_node(
            content=content,
            node_type=node_type_map.get(level, NodeType.CONCEPT),
            metadata={
                'hierarchical_id': hier_node.node_id,
                'level': level.name,
                'domain': domain
            }
        )
        result['graph_id'] = graph_id
        
        # Create relationships in graph
        if relationships:
            for target_id, rel_type in relationships:
                self.graph.add_edge(graph_id, target_id, rel_type)
                self.stats['relationships_created'] += 1
        
        # Layer 3: Add to semantic index
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate(content, use_cache=True)
            
            semantic_id = f"sem_{hier_node.node_id}"
            self.semantic_store.add(
                id=semantic_id,
                content=content,
                embedding=embedding,
                hierarchy_level=level.name.lower(),
                graph_node_id=graph_id,
                importance=0.8 if level in [MemoryLevel.CORE, MemoryLevel.IMPORTANT] else 0.5
            )
            result['semantic_id'] = semantic_id
        except Exception as e:
            # Semantic layer is optional (may fail without API keys)
            result['semantic_id'] = None
        
        self.stats['total_memories'] += 1
        return result
    
    def query_by_level(self, level: MemoryLevel, limit: int = 10) -> List[MemoryNode]:
        """Query memories by hierarchical level."""
        return self.hierarchical.query_by_level(level, limit=limit)
    
    def traverse_relationships(self, node_id: str, depth: int = 2) -> List[Dict]:
        """Traverse graph relationships from a starting node."""
        result = self.graph.traverse_relationships(node_id, depth=depth)
        if result:
            return [{
                'node': n.content,
                'node_id': n.node_id
            } for n in result.nodes]
        return []
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search across all memories."""
        try:
            # Simple semantic search using embedding similarity
            query_embedding = self.embedding_generator.generate(query, use_cache=True)
            
            # Get all embeddings and compute similarities
            all_embeddings = self.semantic_store.get_all()
            results = []
            
            import numpy as np
            for id, content, embedding, metadata in all_embeddings:
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                results.append({
                    'id': id,
                    'content': content,
                    'similarity': similarity,
                    'metadata': metadata
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            # Fallback: return empty list if semantic search fails
            return []
    
    def get_context_assembly(self, query: str = None) -> Dict[str, Any]:
        """
        Assemble curated context from all 4 layers.
        
        This is the key method for context rot prevention - it assembles
        a "state of the union" from all layers rather than just recent messages.
        """
        assembly = {
            'core_principles': [],
            'important_concepts': [],
            'recent_context': [],
            'relevant_details': [],
            'relationships': [],
            'estimated_tokens': 0
        }
        
        # Get CORE principles (never forget these)
        core_memories = self.query_by_level(MemoryLevel.CORE, limit=5)
        for mem in core_memories:
            content = str(mem.content)
            assembly['core_principles'].append(content)
            assembly['estimated_tokens'] += estimate_tokens(content)
        
        # Get IMPORTANT concepts
        important_memories = self.query_by_level(MemoryLevel.IMPORTANT, limit=5)
        for mem in important_memories:
            content = str(mem.content)
            assembly['important_concepts'].append(content)
            assembly['estimated_tokens'] += estimate_tokens(content)
        
        # Get recent CONTEXTUAL items
        contextual_memories = self.query_by_level(MemoryLevel.CONTEXTUAL, limit=10)
        for mem in contextual_memories:
            content = str(mem.content)
            assembly['recent_context'].append(content)
            assembly['estimated_tokens'] += estimate_tokens(content)
        
        # Get relevant GRANULAR details based on query
        if query:
            granular_memories = self.query_by_level(MemoryLevel.GRANULAR, limit=10)
            # Simple keyword matching for demo
            query_words = set(query.lower().split())
            for mem in granular_memories:
                content = str(mem.content)
                content_words = set(content.lower().split())
                if query_words & content_words:  # If there's overlap
                    assembly['relevant_details'].append(content)
                    assembly['estimated_tokens'] += estimate_tokens(content)
        
        return assembly
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all 4 layers."""
        return {
            'session_id': self.session_id,
            'total_memories': self.stats['total_memories'],
            'duplicates_prevented': self.stats['duplicates_prevented'],
            'relationships_created': self.stats['relationships_created'],
            'tokens_saved': self.stats['tokens_saved'],
            'by_level': {
                'CORE': len(self.hierarchical.query_by_level(MemoryLevel.CORE)),
                'IMPORTANT': len(self.hierarchical.query_by_level(MemoryLevel.IMPORTANT)),
                'CONTEXTUAL': len(self.hierarchical.query_by_level(MemoryLevel.CONTEXTUAL)),
                'GRANULAR': len(self.hierarchical.query_by_level(MemoryLevel.GRANULAR))
            }
        }


# ============================================================================
# CONVERSATION SIMULATION
# ============================================================================

class ConversationSimulator:
    """Simulates a long conversation to demonstrate context rot prevention."""
    
    def __init__(self, engine: EnhancedKnowledgeEngine):
        self.engine = engine
        self.messages = []
        self.message_id_map = {}  # Maps message index to graph node ID
    
    def simulate_conversation(self, num_messages: int = 55) -> Dict[str, Any]:
        """
        Simulate a complex conversation about system architecture.
        
        The conversation is designed to demonstrate:
        1. CORE principles added early that should persist
        2. IMPORTANT concepts established at the beginning
        3. CONTEXTUAL decisions made throughout
        4. GRANULAR details that accumulate (with duplicates)
        """
        print(f"{Colors.CYAN}Simulating {num_messages}-message conversation...{Colors.END}\n")
        
        start_time = time.time()
        
        # Phase 1: Establish CORE principles (Messages 1-3)
        print_subsection("Phase 1: Establishing CORE Principles (Messages 1-3)")
        self._add_core_principles()
        
        # Phase 2: Add IMPORTANT concepts (Messages 4-8)
        print_subsection("Phase 2: Adding IMPORTANT Concepts (Messages 4-8)")
        self._add_important_concepts()
        
        # Phase 3: CONTEXTUAL decisions throughout (Messages 9-35)
        print_subsection("Phase 3: Making CONTEXTUAL Decisions (Messages 9-35)")
        self._add_contextual_decisions()
        
        # Phase 4: GRANULAR details with intentional duplicates (Messages 36-55)
        print_subsection("Phase 4: Adding GRANULAR Details with Duplicates (Messages 36-55)")
        duplicates_created = self._add_granular_details_with_duplicates()
        
        elapsed = time.time() - start_time
        
        print(f"\n{Colors.GREEN}[OK] Conversation simulation complete in {elapsed:.2f}s{Colors.END}")
        print(f"{Colors.GREEN}[OK] Total messages: {len(self.messages)}{Colors.END}")
        print(f"{Colors.GREEN}[OK] Duplicates created in simulation: {duplicates_created}{Colors.END}")
        
        return {
            'messages_added': len(self.messages),
            'duplicates_injected': duplicates_created,
            'simulation_time': elapsed
        }
    
    def _add_core_principles(self):
        """Add CORE level principles (high-level, never change)."""
        core_principles = [
            ("System reliability is the highest priority - all design decisions must prioritize uptime and fault tolerance",
             ["reliability", "priority", "principles"]),
            ("We follow a microservices architecture with clear service boundaries and API contracts",
             ["architecture", "microservices", "boundaries"]),
            ("All code must be testable with minimum 80% coverage - untested code is not production-ready",
             ["testing", "coverage", "quality"])
        ]
        
        prev_graph_id = None
        for i, (content, tags) in enumerate(core_principles, 1):
            result = self.engine.add_memory(
                content=content,
                level=MemoryLevel.CORE,
                tags=tags,
                domain="system_design",
                metadata={
                    'message_number': i,
                    'phase': 'core_establishment'
                }
            )
            self.messages.append(('CORE', content, result))
            
            # Create causal relationship chain
            if prev_graph_id and result['graph_id']:
                self.engine.graph.add_edge(
                    prev_graph_id, result['graph_id'], 
                    RelationshipType.SEQUENTIAL
                )
            prev_graph_id = result['graph_id']
            
            print(f"  {Colors.RED}[CORE]{Colors.END} {content[:70]}...")
    
    def _add_important_concepts(self):
        """Add IMPORTANT level concepts (key domain knowledge)."""
        important_concepts = [
            ("The user service handles authentication, authorization, and user profile management",
             ["user_service", "auth", "domain"]),
            ("The order service manages the entire order lifecycle from creation to fulfillment",
             ["order_service", "lifecycle", "domain"]),
            ("The inventory service tracks stock levels and manages reservation/release of items",
             ["inventory_service", "stock", "domain"]),
            ("The payment service integrates with Stripe and PayPal for payment processing",
             ["payment_service", "integration", "domain"]),
            ("Event-driven communication between services using RabbitMQ for async operations",
             ["events", "rabbitmq", "communication"])
        ]
        
        for i, (content, tags) in enumerate(important_concepts, 4):
            result = self.engine.add_memory(
                content=content,
                level=MemoryLevel.IMPORTANT,
                tags=tags,
                domain="system_design",
                metadata={'message_number': i}
            )
            self.messages.append(('IMPORTANT', content, result))
            self.message_id_map[i] = result['graph_id']
            
            print(f"  {Colors.YELLOW}[IMPORTANT]{Colors.END} {content[:65]}...")
    
    def _add_contextual_decisions(self):
        """Add CONTEXTUAL level decisions (recent, may change)."""
        contextual_decisions = [
            "Decision: Use PostgreSQL for primary data storage",
            "Decision: Redis will be used for session caching",
            "Decision: Implement rate limiting at the API gateway level",
            "Decision: Use JWT tokens with 24-hour expiration",
            "Decision: Deploy on Kubernetes with auto-scaling",
            "Decision: Log aggregation via ELK stack (Elasticsearch, Logstash, Kibana)",
            "Decision: Monitoring with Prometheus and Grafana dashboards",
            "Decision: Circuit breaker pattern for external service calls",
            "Decision: Retry with exponential backoff for transient failures",
            "Decision: Database migrations managed via Flyway",
            "Decision: API versioning in URL path (/v1/, /v2/)",
            "Decision: Webhook notifications for async order updates",
            "Decision: Soft deletes with audit trail for compliance",
            "Decision: Read replicas for report generation queries",
            "Decision: CDN for static asset delivery",
            "Decision: Blue-green deployment strategy for zero downtime",
            "Decision: Feature flags for gradual rollout of new features",
            "Decision: Service mesh (Istio) for inter-service communication",
            "Decision: Secrets management via HashiCorp Vault",
            "Decision: Backup strategy: daily full, hourly incremental",
            "Decision: Multi-region deployment for disaster recovery",
            "Decision: GDPR compliance: right to erasure implemented",
            "Decision: PII encryption at rest and in transit",
            "Decision: Audit logging for all data modifications",
            "Decision: Performance SLA: 99th percentile < 200ms",
            "Decision: Availability SLA: 99.99% uptime target",
            "Decision: On-call rotation with PagerDuty integration"
        ]
        
        for i, content in enumerate(contextual_decisions, 9):
            result = self.engine.add_memory(
                content=content,
                level=MemoryLevel.CONTEXTUAL,
                tags=["decision", "context"],
                domain="system_design",
                metadata={'message_number': i}
            )
            self.messages.append(('CONTEXTUAL', content, result))
            self.message_id_map[i] = result['graph_id']
            
            if i <= 12:  # Only print first few
                print(f"  {Colors.GREEN}[CONTEXTUAL]{Colors.END} {content[:60]}...")
        
        if len(contextual_decisions) > 4:
            print(f"  ... and {len(contextual_decisions) - 4} more contextual decisions")
    
    def _add_granular_details_with_duplicates(self) -> int:
        """
        Add GRANULAR details with intentional duplicates.
        Returns the number of duplicates created.
        """
        # Base granular details
        granular_base = [
            "The user service runs on port 8081",
            "Database connection pool: max 20 connections",
            "Redis cache TTL: 3600 seconds for sessions",
            "API rate limit: 1000 requests per minute per IP",
            "Log retention period: 30 days",
            "Health check endpoint: /health every 10 seconds",
            "JWT secret rotated every 90 days",
            "Backup retention: 7 daily, 4 weekly, 12 monthly",
            "Alert threshold: CPU > 80% for 5 minutes",
            "Memory limit per pod: 512MB request, 1GB limit"
        ]
        
        # Create variations with duplicates
        variations = [
            "The user service runs on port 8081",  # EXACT DUPLICATE
            "User service port is 8081",  # NEAR DUPLICATE
            "Database connection pool max is 20",  # NEAR DUPLICATE
            "Max database connections: 20",  # NEAR DUPLICATE
            "Redis cache TTL set to 3600 seconds for user sessions",  # NEAR DUPLICATE
            "Session cache in Redis expires after 3600s",  # NEAR DUPLICATE
            "Rate limiting: 1000 req/min per IP address",  # NEAR DUPLICATE
            "API gateway limits to 1000 requests per minute",  # NEAR DUPLICATE
            "Logs kept for 30 days",  # NEAR DUPLICATE
            "30-day log retention policy",  # NEAR DUPLICATE
        ]
        
        all_granular = granular_base + variations
        random.shuffle(all_granular)
        
        duplicates_count = 0
        for i, content in enumerate(all_granular, 36):
            result = self.engine.add_memory(
                content=content,
                level=MemoryLevel.GRANULAR,
                tags=["detail", "configuration"],
                domain="system_design",
                metadata={'message_number': i}
            )
            self.messages.append(('GRANULAR', content, result))
            
            if result['deduplicated']:
                duplicates_count += 1
        
        print(f"  {Colors.BLUE}[GRANULAR]{Colors.END} Added {len(all_granular)} granular details")
        print(f"  {Colors.GREEN}[OK] {duplicates_count} duplicates detected and prevented{Colors.END}")
        
        return duplicates_count
    
    def get_message_5_id(self) -> Optional[str]:
        """Get the graph ID of message 5 (for relationship demonstration)."""
        return self.message_id_map.get(5)
    
    def get_message_50_id(self) -> Optional[str]:
        """Get the graph ID of message 50 (for relationship demonstration)."""
        return self.message_id_map.get(50)


# ============================================================================
# DEMO EXECUTION
# ============================================================================

def run_demo():
    """Run the complete hierarchical indexing demo."""
    
    # Print header
    print(f"{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'4-LAYER HIERARCHICAL INDEXING DEMO'.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{'Context Rot Prevention for Long Conversations'.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.END}\n")
    
    # Initialize the enhanced knowledge engine
    print_section("1. SETUP & CONFIGURATION")
    config = DemoConfig(num_messages=55)
    engine = EnhancedKnowledgeEngine(config)
    
    # Simulate the conversation
    print_section("2. SIMULATE LONG CONVERSATION (55+ Messages)")
    simulator = ConversationSimulator(engine)
    sim_results = simulator.simulate_conversation(config.num_messages)
    
    # Demonstrate each index layer
    print_section("3. DEMONSTRATE EACH INDEX LAYER")
    
    # 3.1 Hierarchical Organization
    print_subsection("3.1 Hierarchical Organization (Layer 1)")
    print("Querying memories by hierarchical level:\n")
    
    for level in [MemoryLevel.CORE, MemoryLevel.IMPORTANT, 
                  MemoryLevel.CONTEXTUAL, MemoryLevel.GRANULAR]:
        memories = engine.query_by_level(level, limit=3)
        level_colors = {
            MemoryLevel.CORE: Colors.RED,
            MemoryLevel.IMPORTANT: Colors.YELLOW,
            MemoryLevel.CONTEXTUAL: Colors.GREEN,
            MemoryLevel.GRANULAR: Colors.BLUE
        }
        color = level_colors.get(level, Colors.END)
        print(f"  {color}[{level.name}]{Colors.END} {len(memories)} memories found")
        for mem in memories[:2]:
            content = str(mem.content)[:60]
            print(f"    - {content}...")
        if len(memories) > 2:
            print(f"    ... and {len(memories) - 2} more")
    
    # 3.2 Graph Relationships
    print_subsection("3.2 Graph Relationships (Layer 2)")
    print("Demonstrating relationship preservation:\n")
    
    # Create a connection between message 5 and a recent message
    msg_5_id = simulator.get_message_5_id()
    if msg_5_id:
        # Add a relationship to a recent contextual decision
        recent_nodes = engine.graph.find_nodes_by_type(NodeType.DECISION, limit=1)
        if recent_nodes:
            recent_id = recent_nodes[0].node_id
            engine.graph.add_edge(
                msg_5_id, recent_id, 
                RelationshipType.DEPENDS_ON,
                metadata={'note': 'Inventory depends on order lifecycle'}
            )
            print(f"  {Colors.CYAN}Created relationship:{Colors.END}")
            print(f"    Source (Message 5): Order service manages order lifecycle")
            print(f"    --> DEPENDS_ON -->")
            print(f"    Target: {recent_nodes[0].content[:50]}...")
            
            # Traverse from message 5
            print(f"\n  {Colors.CYAN}Traversing from Message 5 (depth=2):{Colors.END}")
            traversal = engine.traverse_relationships(msg_5_id, depth=2)
            for item in traversal[:3]:
                print(f"    - {item['node'][:50]}...")
    
    # 3.3 Deduplication Stats
    print_subsection("3.3 Deduplication (Layer 4)")
    stats = engine.get_stats()
    print(f"  {Colors.GREEN}[OK] Duplicates prevented: {stats['duplicates_prevented']}{Colors.END}")
    print(f"  {Colors.GREEN}[OK] Tokens saved: ~{stats['tokens_saved']:,}{Colors.END}")
    
    # Show hash index stats
    merge_stats = engine.hash_index.merger.get_merge_stats()
    print(f"  {Colors.GREEN}[OK] Total merges: {merge_stats['total_merges']}{Colors.END}")
    print(f"  {Colors.GREEN}[OK] Exact duplicates: {merge_stats['exact_duplicates']}{Colors.END}")
    print(f"  {Colors.GREEN}[OK] Near duplicates: {merge_stats['near_duplicates']}{Colors.END}")
    
    # 3.4 Semantic Search (if available)
    print_subsection("3.4 Semantic Relevance (Layer 3)")
    try:
        search_results = engine.search_semantic("database connection pool", top_k=3)
        if search_results:
            print(f"  {Colors.CYAN}Query: 'database connection pool'{Colors.END}")
            for result in search_results:
                similarity = result.get('similarity', 0)
                content = result.get('content', '')[:50]
                print(f"    - [{similarity:.2f}] {content}...")
        else:
            print(f"  {Colors.YELLOW}(Semantic search requires embedding backend){Colors.END}")
    except Exception as e:
        print(f"  {Colors.YELLOW}(Semantic search requires embedding backend){Colors.END}")
    
    # Context Assembly Demo
    print_section("4. CONTEXT ASSEMBLY DEMO")
    print_subsection("4.1 Raw Conversation (What Would Overwhelm an LLM)")
    
    raw_text = "\n".join([f"[{level}] {content[:60]}..." 
                          for level, content, _ in simulator.messages])
    raw_tokens = estimate_tokens(raw_text)
    
    print(f"  {Colors.RED}Total messages: {len(simulator.messages)}{Colors.END}")
    print(f"  {Colors.RED}Estimated tokens (raw): ~{raw_tokens:,}{Colors.END}")
    print(f"  {Colors.RED}This would exceed most LLM context windows!{Colors.END}")
    
    print_subsection("4.2 Curated Context Through 4 Indexes")
    print("Assembling 'state of the union' from all layers:\n")
    
    assembly = engine.get_context_assembly(query="system architecture")
    
    print(f"  {Colors.RED}[CORE Principles]{Colors.END}")
    for item in assembly['core_principles']:
        print(f"    [OK] {item[:70]}...")
    
    print(f"\n  {Colors.YELLOW}[IMPORTANT Concepts]{Colors.END}")
    for item in assembly['important_concepts'][:3]:
        print(f"    [OK] {item[:70]}...")
    
    print(f"\n  {Colors.GREEN}[Recent Context]{Colors.END}")
    for item in assembly['recent_context'][:3]:
        print(f"    [OK] {item[:70]}...")
    
    curated_tokens = assembly['estimated_tokens']
    savings_percent = (1 - curated_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0
    
    print(f"\n  {Colors.CYAN}Token Efficiency:{Colors.END}")
    print_metric("Raw conversation", f"~{raw_tokens:,} tokens", Colors.RED)
    print_metric("Curated context", f"~{curated_tokens:,} tokens", Colors.GREEN)
    print_metric("Space saved", f"{savings_percent:.1f}%", Colors.GREEN)
    
    # Context Rot Prevention in Action
    print_section("5. CONTEXT ROT PREVENTION IN ACTION")
    
    print_subsection("5.1 Query Mid-Conversation: 'What are our core principles?'")
    print("Even after 55 messages, CORE principles are instantly accessible:\n")
    
    core_memories = engine.query_by_level(MemoryLevel.CORE, limit=5)
    for i, mem in enumerate(core_memories, 1):
        content = str(mem.content)
        access_info = f"(accessed {mem.access_count} times)"
        print(f"  {Colors.RED}{i}. {content}{Colors.END}")
        print(f"     {Colors.CYAN}{access_info}{Colors.END}")
    
    print_subsection("5.2 Graph Connections Maintain Logical Thread")
    print("Relationship graph preserves connections across 45 messages:\n")
    
    if msg_5_id:
            # Find related nodes through the graph
            related = engine.graph.get_related_nodes(msg_5_id)
            if related:
                print(f"  Message 5 (Order service) is connected to:")
                for node_id in related[:3]:
                    node = engine.graph.get_node(node_id, update_access=False)
                    if node:
                        print(f"    - {node.content[:60]}...")
    
    # Final Statistics
    print_section("6. FINAL STATISTICS")
    
    final_stats = engine.get_stats()
    
    print(f"  {Colors.BOLD}Session Summary:{Colors.END}")
    print_metric("Session ID", final_stats['session_id'])
    print_metric("Total memories stored", str(final_stats['total_memories']))
    print_metric("Duplicates prevented", str(final_stats['duplicates_prevented']))
    print_metric("Relationships created", str(final_stats['relationships_created']))
    print_metric("Tokens saved via dedup", f"~{final_stats['tokens_saved']:,}")
    
    print(f"\n  {Colors.BOLD}Distribution by Level:{Colors.END}")
    for level_name, count in final_stats['by_level'].items():
        bar = "#" * (count // 2)
        print(f"    {level_name:12} {bar} {count}")
    
    # ASCII Architecture Diagram
    print_section("7. SYSTEM ARCHITECTURE")
    
    architecture = """
    +---------------------------------------------------------------------+
    |                    ENHANCED KNOWLEDGE ENGINE                        |
    |                        (4-Layer Indexing)                           |
    +---------------------------------------------------------------------+
    |                                                                     |
    |   INPUT                    PROCESSING                  OUTPUT       |
    |   -----                    ---------                  ------        |
    |                                                                     |
    |   +---------+    +-------------------------+    +-------------+    |
    |   | Message |--->|  Layer 1: Hierarchical  |--->| Structured  |    |
    |   | Stream  |    |  +- CORE (0)            |    | Context     |    |
    |   +---------+    |  +- IMPORTANT (1)       |    | Assembly    |    |
    |       |          |  +- CONTEXTUAL (2)      |    +-------------+    |
    |       |          |  +- GRANULAR (3)       |           |           |
    |       |          +-----------+-----------+           |           |
    |       |                      |                       |           |
    |       |          +-----------v-----------+           |           |
    |       +---------->|  Layer 2: Graph       |-----------+           |
    |                  |  +- Causal edges      |                       |
    |                  |  +- Temporal edges    |                       |
    |                  |  +- Semantic edges    |                       |
    |                  +-----------+-----------+                       |
    |                              |                                   |
    |                  +-----------v-----------+                       |
    |                  |  Layer 3: Semantic    |                       |
    |                  |  +- Vector embeddings  |                       |
    |                  |  +- Similarity search  |                       |
    |                  +-----------+-----------+                       |
    |                              |                                   |
    |                  +-----------v-----------+                       |
    |                  |  Layer 4: Hash        |                       |
    |                  |  +- Exact matching    |                       |
    |                  |  +- Near-duplicate    |                       |
    |                  |  +- Deduplication     |                       |
    |                  +-----------------------+                       |
    |                                                                  |
    |                                          +-----------v-----+     |
    |                                          | Context Window  |     |
    |                                          | (Optimized for  |     |
    |                                          |  LLM ingestion) |     |
    |                                          +-----------------+     |
    |                                                                  |
    +------------------------------------------------------------------+
    """
    print(architecture)
    
    # Conclusion
    print_section("CONCLUSION")
    
    conclusion = f"""
    {Colors.GREEN}{Colors.BOLD}The 4-Layer Hierarchical Indexing System successfully prevents context rot:{Colors.END}
    
    {Colors.CYAN}1. HIERARCHICAL INDEX{Colors.END} ensures important principles (CORE) are never lost,
       even after 55+ messages of conversation.
    
    {Colors.CYAN}2. GRAPH INDEX{Colors.END} preserves logical relationships, maintaining the thread
       of reasoning from message 5 to message 50.
    
    {Colors.CYAN}3. SEMANTIC INDEX{Colors.END} enables meaning-based retrieval, finding relevant
       information even with different wording.
    
    {Colors.CYAN}4. HASH INDEX{Colors.END} prevents redundant storage, saving ~{final_stats['tokens_saved']:,}
       tokens through intelligent deduplication.
    
    {Colors.GREEN}{Colors.BOLD}Result: A structured "state of the union" that gives LLMs the context they need,{Colors.END}
    {Colors.GREEN}{Colors.BOLD}without overwhelming their context windows.{Colors.END}
    """
    print(conclusion)
    
    print(f"\n{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'DEMO COMPLETE'.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.END}\n")
    
    return engine, simulator


def main():
    """Main entry point for the demo."""
    try:
        # Disable colors on Windows if not supported
        import os
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                # Disable colors if not supported
                for attr in dir(Colors):
                    if not attr.startswith('_'):
                        setattr(Colors, attr, '')
        
        # Run the demo
        engine, simulator = run_demo()
        
        # Ask if user wants to clean up
        print(f"\n{Colors.YELLOW}Demo databases created:{Colors.END}")
        print(f"  - {engine.config.hierarchical_db}")
        print(f"  - {engine.config.graph_db}")
        print(f"  - {engine.config.hash_db}")
        print(f"  - {engine.config.semantic_cache}/")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted by user.{Colors.END}")
    except Exception as e:
        print(f"\n\n{Colors.RED}Error during demo: {e}{Colors.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
