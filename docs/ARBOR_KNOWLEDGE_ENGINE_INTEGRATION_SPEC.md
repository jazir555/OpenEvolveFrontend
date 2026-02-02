# Arbor-Knowledge Engine Integration Specification

## Executive Summary

This document specifies the integration of **Arbor** (a Rust-based code graph intelligence layer) with the **OpenEvolve Knowledge Engine** (a Python-based comprehensive knowledge management platform). The integration will enable the Knowledge Engine to ingest, query, and reason about code structure graphs, providing AI agents with precise code navigation capabilities beyond traditional embedding-based RAG.

**Version:** 1.0  
**Status:** Draft  
**Date:** 2026-02-01

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Integration Architecture](#integration-architecture)
4. [Component Mapping](#component-mapping)
5. [Data Flow](#data-flow)
6. [API Specifications](#api-specifications)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Phase 1: Foundation](#phase-1-foundation)
9. [Phase 2: Graph Bridge](#phase-2-graph-bridge)
10. [Phase 3: Intelligence Layer](#phase-3-intelligence-layer)
11. [Phase 4: MCP Integration](#phase-4-mcp-integration)
12. [Phase 5: Visualization](#phase-5-visualization)
13. [Testing Strategy](#testing-strategy)
14. [Deployment](#deployment)
15. [Appendix](#appendix)

---

## Overview

### What is Arbor?

Arbor is a graph-native intelligence layer for code that:
- Parses codebases into AST graphs using Tree-sitter (~144ms for 10k lines)
- Creates a living graph where functions, classes, and variables are **nodes**
- Represents imports, calls, and implementations as **edges**
- Provides precise code navigation via graph traversal (not embedding similarity)
- Supports 10 languages: Rust, TypeScript, JavaScript, Python, Go, Java, C, C++, C#, Dart
- Exposes graph queries via WebSocket and MCP (Model Context Protocol)

### What is the Knowledge Engine?

The OpenEvolve Knowledge Engine is a comprehensive platform that:
- Stores multi-modal knowledge (documents, graphs, embeddings)
- Provides hybrid search (semantic + graph + keyword)
- Supports multiple backends (PostgreSQL, Memgraph, Qdrant, Redis)
- Has existing integrations with Graphiti, KG-Gen, NeuralKG, etc.
- Uses MCP for AI agent communication

### Integration Value Proposition

```
Traditional RAG:                    Arbor + Knowledge Engine:
                                    
"auth" → 47 embeddings              "auth" → AuthController (from Arbor)
         (ambiguous)                         ├── validates via → TokenMiddleware
                                            ├── queries → UserRepository
                                            └── emits → AuthEvent
                                                    ↓
                                            [Knowledge Engine]
                                                    ↓
                                            Cross-reference with:
                                            - Documentation
                                            - Bug reports
                                            - Architecture decisions
                                            - Team expertise
```

---

## System Architecture

### Current Arbor Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ARBOR SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  arbor-cli  │  │arbor-server │  │     arbor-mcp       │  │
│  │  (CLI)      │  │(WebSocket)  │  │  (MCP Bridge)       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┴─────────────────────┘             │
│                          │                                   │
│                   ┌──────┴──────┐                           │
│                   │ arbor-graph │                           │
│                   │  (petgraph) │                           │
│                   └──────┬──────┘                           │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐             │
│    │arbor-core│     │arbor-watcher      │   Sled    │             │
│    │(Tree-sitter)   │(file watch)│     │(persist) │             │
│    └─────────┘     └─────────┘     └─────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Current Knowledge Engine Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   API Layer │  │   Graph     │  │   MCP Server        │  │
│  │  (REST/WS)  │  │   Engine    │  │  (Agent Bridge)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┴─────────────────────┘             │
│                          │                                   │
│                   ┌──────┴──────┐                           │
│                   │UnifiedKnowledgePlatform                 │
│                   └──────┬──────┘                           │
│                          │                                   │
│    ┌─────────────────────┼─────────────────────┐             │
│    │                     │                     │             │
│ ┌──┴───┐  ┌─────────┐  ┌┴────────┐  ┌────────┴┐            │
│ │Postgre│  │Memgraph │  │ Qdrant  │  │  Redis  │            │
│ │SQL    │  │(Graph)  │  │(Vectors)│  │ (Cache) │            │
│ └──────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Architecture

### Target Integrated Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED KNOWLEDGE PLATFORM                               │
│                         (Python)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ARBOR INTEGRATION LAYER                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │   │
│  │  │Arbor Client │  │Arbor Graph  │  │   Arbor MCP Bridge          │ │   │
│  │  │ (WebSocket) │  │  Adapter    │  │  (Model Context Protocol)   │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────────┬──────────────┘ │   │
│  │         │                │                        │                │   │
│  │         └────────────────┴────────────────────────┘                │   │
│  │                          │                                         │   │
│  │                   ┌──────┴──────┐                                  │   │
│  │                   │ UnifiedGraph │                                  │   │
│  │                   │  Merger      │                                  │   │
│  │                   └──────┬──────┘                                  │   │
│  └──────────────────────────┼────────────────────────────────────────┘   │
│                             │                                             │
│  ┌──────────────────────────┼────────────────────────────────────────┐   │
│  │                   KNOWLEDGE ENGINE CORE                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐│   │
│  │  │  Semantic   │  │   Graph     │  │   Entity-Relation           ││   │
│  │  │   Search    │  │  Analytics  │  │   Extraction                ││   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘│   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                             │                                             │
│  ┌──────────────────────────┴────────────────────────────────────────┐   │
│  │                     STORAGE LAYER                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │PostgreSQL│  │Memgraph │  │ Qdrant  │  │  Redis  │  │  Sled   │  │   │
│  │  │(Metadata)│  │(Graph)  │  │(Vectors)│  │ (Cache) │  │(Arbor)  │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket
                                    │
┌───────────────────────────────────┴───────────────────────────────────────┐
│                          ARBOR SIDE CAR (Rust)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │arbor-server │  │arbor-graph  │  │arbor-watcher│  │   arbor-core    │   │
│  │            │  │  (petgraph) │  │             │  │ (Tree-sitter)   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                                                                             │
│                    ┌─────────────────┐                                      │
│                    │    Codebase     │                                      │
│                    │   (Your Files)  │                                      │
│                    └─────────────────┘                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Component Mapping

### Arbor → Knowledge Engine Mapping

| Arbor Component | Purpose | Knowledge Engine Equivalent | Integration Strategy |
|-----------------|---------|----------------------------|---------------------|
| `arbor-core` | Tree-sitter parsing | `document_loader`, `deepke` | Use Arbor for code-specific parsing, keep KE for general docs |
| `arbor-graph` | In-memory graph (petgraph) | `core/entity_knowledge_graph.py` | Bridge Arbor graph to KE unified graph |
| `arbor-watcher` | File watching | Real-time collaboration layer | Integrate file change events into KE event bus |
| `arbor-server` | WebSocket server | `api_gateway.py`, `server.py` | Connect via WebSocket client, expose through KE API |
| `arbor-mcp` | MCP bridge | `orchestration/mcp_server.py` | Unified MCP server that routes to Arbor |
| `arbor-cli` | CLI interface | `__main__.py`, CLI tools | Extend KE CLI with Arbor commands |

### Graph Schema Mapping

#### Arbor Node → Knowledge Engine Entity

```python
# Arbor Node Schema
{
  "id": "unique_node_identifier",
  "name": "FunctionName",
  "qualifiedName": "ModuleName.ClassName.FunctionName",
  "kind": "function",  # function, method, class, interface, etc.
  "file": "src/services/user.ts",
  "lineStart": 45,
  "lineEnd": 78,
  "column": 2,
  "signature": "async validateUser(id: string): Promise<User>",
  "visibility": "public",
  "attributes": {"async": true, "static": false},
  "docstring": "Validates a user by their ID.",
  "centrality": 0.75
}

# Maps to Knowledge Engine Entity
{
  "entity_id": "arbor:unique_node_identifier",
  "name": "FunctionName",
  "entity_type": "code_function",  # prefixed to avoid collision
  "properties": {
    "arbor_kind": "function",
    "qualified_name": "ModuleName.ClassName.FunctionName",
    "file_path": "src/services/user.ts",
    "location": {"line_start": 45, "line_end": 78, "column": 2},
    "signature": "async validateUser(id: string): Promise<User>",
    "visibility": "public",
    "attributes": {"async": true, "static": false},
    "docstring": "Validates a user by their ID.",
    "centrality_score": 0.75,
    "source_system": "arbor"
  },
  "metadata": {
    "indexed_at": "2026-02-01T00:00:00Z",
    "language": "typescript"
  }
}
```

#### Arbor Edge → Knowledge Engine Relationship

```python
# Arbor Edge Schema
{
  "from": "source_node_id",
  "to": "target_node_id",
  "kind": "calls",  # calls, imports, extends, implements, etc.
  "location": {"file": "src/services/user.ts", "line": 52, "column": 8}
}

# Maps to Knowledge Engine Relationship
{
  "relationship_id": "arbor:edge_hash",
  "source_id": "arbor:source_node_id",
  "target_id": "arbor:target_node_id",
  "relationship_type": "code_calls",  # prefixed
  "properties": {
    "arbor_kind": "calls",
    "location": {"file": "src/services/user.ts", "line": 52, "column": 8}
  },
  "metadata": {
    "indexed_at": "2026-02-01T00:00:00Z",
    "source_system": "arbor"
  }
}
```

---

## Data Flow

### 1. Initial Codebase Indexing Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  User    │────►│   KE     │────►│  Arbor   │────►│ File     │
│  Request │     │  API     │     │  Server  │     │ System   │
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                        │
                              ┌─────────────────────────┘
                              │ Parse (Tree-sitter)
                              ▼
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  KE      │◄────│  Arbor   │◄────│  Graph   │◄────│  AST     │
│  Graph   │     │  Export  │     │ Builder  │     │  Nodes   │
└────┬─────┘     └──────────┘     └──────────┘     └──────────┘
     │
     │ Transform & Merge
     ▼
┌──────────┐     ┌──────────┐
│  Unified │────►│ Memgraph │
│  Graph   │     │ Qdrant   │
└──────────┘     └──────────┘
```

### 2. Real-time Update Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  File    │────►│  Arbor   │────►│  Arbor   │────►│   KE     │
│  Change  │     │  Watcher │     │  Server  │     │  Bridge  │
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                        │
                              ┌─────────────────────────┘
                              │ WebSocket Event
                              ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│  KE      │────►│  Delta   │────►│  Graph   │
│  Event   │     │  Engine  │     │  Update  │
│  Bus     │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘
```

### 3. Query Flow (AI Agent)

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   AI     │────►│   MCP    │────►│   KE     │────►│  Arbor   │
│  Agent   │     │  Server  │     │  Router  │     │  Client  │
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
     ▲                                                  │
     │                                                  │ ArborQL
     │                                                  ▼
     │                                           ┌──────────┐
     │                                           │  Arbor   │
     │                                           │  Graph   │
     │                                           └────┬─────┘
     │                                                │
     │           ┌────────────────────────────────────┘
     │           │ Results + Cross-references
     │           ▼
     │    ┌──────────┐     ┌──────────┐     ┌──────────┐
     └────│   KE     │◄────│  Related │◄────│  Docs/   │
          │  Context │     │  Nodes   │     │  Issues  │
          └──────────┘     └──────────┘     └──────────┘
```

---

## API Specifications

### 1. Arbor Client API (Python)

```python
# knowledge_engine/integrations/arbor/client.py

class ArborClient:
    """
    WebSocket client for communicating with Arbor server.
    """
    
    def __init__(self, ws_url: str = "ws://localhost:7433"):
        self.ws_url = ws_url
        self._ws = None
        self._graph_cache = None
    
    async def connect(self) -> bool:
        """Establish WebSocket connection to Arbor server."""
        pass
    
    async def index_codebase(self, path: str) -> IndexingResult:
        """Trigger full codebase indexing."""
        pass
    
    async def query_graph(self, query: ArborQLQuery) -> QueryResult:
        """Execute ArborQL query against the graph."""
        pass
    
    async def find_path(self, start: str, end: str) -> List[Node]:
        """Find path between two nodes using A* algorithm."""
        pass
    
    async def analyze_impact(self, node_id: str) -> ImpactAnalysis:
        """Determine blast radius of changing a node."""
        pass
    
    async def get_context(self, node_id: str, depth: int = 2) -> ContextGraph:
        """Get contextual subgraph around a node."""
        pass
    
    async def subscribe_changes(self, callback: Callable) -> None:
        """Subscribe to real-time graph changes."""
        pass
    
    async def export_graph(self) -> Dict:
        """Export full graph as JSON."""
        pass
```

### 2. Arbor Graph Adapter API

```python
# knowledge_engine/integrations/arbor/graph_adapter.py

class ArborGraphAdapter:
    """
    Adapter to convert Arbor graph format to Knowledge Engine unified graph.
    """
    
    def __init__(self, entity_knowledge_graph: EntityKnowledgeGraph):
        self.ekg = entity_knowledge_graph
        self.node_id_mapping = {}
    
    def convert_arbor_node(self, arbor_node: Dict) -> Entity:
        """Convert Arbor node to KE Entity."""
        return Entity(
            entity_id=f"arbor:{arbor_node['id']}",
            name=arbor_node['name'],
            entity_type=f"code_{arbor_node['kind']}",
            properties={
                'arbor_kind': arbor_node['kind'],
                'qualified_name': arbor_node.get('qualifiedName'),
                'file_path': arbor_node['file'],
                'location': {
                    'line_start': arbor_node['lineStart'],
                    'line_end': arbor_node['lineEnd'],
                    'column': arbor_node.get('column')
                },
                'signature': arbor_node.get('signature'),
                'visibility': arbor_node.get('visibility'),
                'attributes': arbor_node.get('attributes', {}),
                'docstring': arbor_node.get('docstring'),
                'centrality_score': arbor_node.get('centrality', 0.0)
            },
            metadata={
                'source_system': 'arbor',
                'indexed_at': datetime.utcnow().isoformat()
            }
        )
    
    def convert_arbor_edge(self, arbor_edge: Dict) -> Relationship:
        """Convert Arbor edge to KE Relationship."""
        return Relationship(
            source_id=f"arbor:{arbor_edge['from']}",
            target_id=f"arbor:{arbor_edge['to']}",
            relationship_type=f"code_{arbor_edge['kind']}",
            properties={
                'arbor_kind': arbor_edge['kind'],
                'location': arbor_edge.get('location')
            },
            metadata={'source_system': 'arbor'}
        )
    
    async def merge_arbor_graph(self, arbor_graph: Dict) -> MergeResult:
        """Merge full Arbor graph into Knowledge Engine."""
        pass
    
    async def apply_delta(self, delta: GraphDelta) -> None:
        """Apply incremental graph changes."""
        pass
```

### 3. MCP Bridge API

```python
# knowledge_engine/integrations/arbor/mcp_bridge.py

class ArborMCPBridge:
    """
    Bridge to expose Arbor capabilities through MCP (Model Context Protocol).
    Integrates with Knowledge Engine's MCP server.
    """
    
    def __init__(self, arbor_client: ArborClient):
        self.arbor = arbor_client
    
    # MCP Tool: Code Navigation
    @mcp_tool("arbor_find_definition")
    async def find_definition(self, symbol: str, file: Optional[str] = None) -> Dict:
        """
        Find the definition of a symbol in the codebase.
        
        Args:
            symbol: Symbol name to find (e.g., "AuthController")
            file: Optional file context for disambiguation
        
        Returns:
            Node information including file, line, and signature
        """
        pass
    
    # MCP Tool: Call Graph
    @mcp_tool("arbor_get_callers")
    async def get_callers(self, function_name: str) -> List[Dict]:
        """
        Get all functions that call the specified function.
        
        Args:
            function_name: Name of the function
        
        Returns:
            List of calling functions with context
        """
        pass
    
    # MCP Tool: Impact Analysis
    @mcp_tool("arbor_analyze_impact")
    async def analyze_impact(self, symbol: str, change_type: str) -> ImpactReport:
        """
        Analyze the impact of modifying a symbol.
        
        Args:
            symbol: Symbol to analyze
            change_type: Type of change (rename, modify_signature, delete)
        
        Returns:
            Impact report with affected files and blast radius
        """
        pass
    
    # MCP Tool: Get Context
    @mcp_tool("arbor_get_context")
    async def get_context(self, symbol: str, depth: int = 2) -> ContextGraph:
        """
        Get relevant code context for a symbol.
        
        Args:
            symbol: Symbol to get context for
            depth: How many hops to include (default: 2)
        
        Returns:
            Subgraph with related code entities
        """
        pass
    
    # MCP Tool: Find Path
    @mcp_tool("arbor_find_path")
    async def find_path(self, start: str, end: str) -> CodePath:
        """
        Find the logic flow between two components.
        
        Args:
            start: Starting symbol
            end: Ending symbol
        
        Returns:
            Path through the code with explanations
        """
        pass
    
    # MCP Tool: Refactor Validation
    @mcp_tool("arbor_validate_refactor")
    async def validate_refactor(self, operations: List[RefactorOp]) -> ValidationReport:
        """
        Validate a refactoring operation before applying.
        
        Args:
            operations: List of refactoring operations
        
        Returns:
            Validation report with potential issues
        """
        pass
```

---

## Implementation Roadmap

### Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| Phase 1 | 2 weeks | Foundation | Client, config, connection |
| Phase 2 | 3 weeks | Graph Bridge | Adapter, mapping, sync |
| Phase 3 | 2 weeks | Intelligence | Query, context, impact |
| Phase 4 | 2 weeks | MCP Integration | Tools, routing, testing |
| Phase 5 | 1 week | Visualization | Arbor visualizer integration |
| **Total** | **10 weeks** | | |

---

## Phase 1: Foundation

### Week 1-2: Arbor Client & Connection

#### Tasks

1. **Create Arbor Client Module**
   - File: `knowledge_engine/integrations/arbor/__init__.py`
   - File: `knowledge_engine/integrations/arbor/client.py`
   - Implement WebSocket connection management
   - Implement reconnection logic
   - Add connection pooling for multiple Arbor instances

2. **Configuration System**
   - File: `knowledge_engine/integrations/arbor/config.py`
   - Add Arbor configuration to KE config system
   - Support multiple Arbor profiles (dev, staging, prod)
   - Environment variable support

3. **Health Checking**
   - File: `knowledge_engine/integrations/arbor/health.py`
   - Implement health probes for Arbor connection
   - Integration with KE health monitoring

4. **Basic Testing**
   - File: `knowledge_engine/tests/integrations/test_arbor_client.py`
   - Unit tests for client
   - Mock Arbor server for testing

#### Deliverables

```python
# Example usage after Phase 1
from knowledge_engine.integrations.arbor import ArborClient

client = ArborClient(ws_url="ws://localhost:7433")
await client.connect()
status = await client.health_check()
```

### Phase 1 Code Structure

```
knowledge_engine/integrations/arbor/
├── __init__.py
├── client.py          # WebSocket client
├── config.py          # Configuration
├── health.py          # Health checks
├── exceptions.py      # Custom exceptions
└── tests/
    ├── __init__.py
    ├── test_client.py
    └── test_config.py
```

---

## Phase 2: Graph Bridge

### Week 3-5: Graph Adapter & Synchronization

#### Tasks

1. **Graph Schema Mapping**
   - File: `knowledge_engine/integrations/arbor/schema_mapping.py`
   - Define Arbor → KE entity mappings
   - Handle language-specific node types
   - Map centrality scores to KE properties

2. **Graph Adapter Implementation**
   - File: `knowledge_engine/integrations/arbor/graph_adapter.py`
   - Convert Arbor nodes to KE Entities
   - Convert Arbor edges to KE Relationships
   - Handle ID namespacing (prefix with `arbor:`)

3. **Full Graph Import**
   - File: `knowledge_engine/integrations/arbor/importer.py`
   - Import full Arbor graph into KE
   - Batch processing for large codebases
   - Progress tracking and resumption

4. **Delta Synchronization**
   - File: `knowledge_engine/integrations/arbor/sync.py`
   - Real-time sync from Arbor watcher
   - Handle file create/update/delete events
   - Conflict resolution strategy

5. **Storage Integration**
   - Integrate with Memgraph for graph storage
   - Integrate with Qdrant for vector search on code
   - Redis caching for hot code paths

#### Deliverables

```python
# Example usage after Phase 2
from knowledge_engine.integrations.arbor import ArborGraphAdapter
from knowledge_engine.core import EntityKnowledgeGraph

graph = EntityKnowledgeGraph()
adapter = ArborGraphAdapter(graph)

# Import full Arbor graph
arbor_export = await arbor_client.export_graph()
result = await adapter.merge_arbor_graph(arbor_export)
print(f"Imported {result.nodes_imported} nodes, {result.edges_imported} edges")

# Real-time sync
await arbor_client.subscribe_changes(adapter.apply_delta)
```

### Phase 2 Code Structure

```
knowledge_engine/integrations/arbor/
├── __init__.py
├── client.py
├── schema_mapping.py   # NEW
├── graph_adapter.py    # NEW
├── importer.py         # NEW
├── sync.py             # NEW
└── tests/
    ├── test_adapter.py
    ├── test_importer.py
    └── test_sync.py
```

---

## Phase 3: Intelligence Layer

### Week 6-7: Query & Context Engine

#### Tasks

1. **Query Bridge**
   - File: `knowledge_engine/integrations/arbor/query.py`
   - Translate KE queries to ArborQL
   - Federated queries (Arbor + KE graph)
   - Result ranking and deduplication

2. **Context Assembly**
   - File: `knowledge_engine/integrations/arbor/context.py`
   - Assemble code context for AI agents
   - Cross-reference with documentation
   - Include bug reports, PRs related to code

3. **Impact Analysis Integration**
   - File: `knowledge_engine/integrations/arbor/impact.py`
   - Extend Arbor's impact analysis with KE data
   - Show business impact (which features affected)
   - Historical change success rates

4. **Hybrid Search**
   - Combine Arbor graph traversal with KE semantic search
   - Vector search on code signatures and docstrings
   - Keyword search on code comments

#### Deliverables

```python
# Example usage after Phase 3
from knowledge_engine.integrations.arbor import ArborContextEngine

context = ArborContextEngine(arbor_client, knowledge_engine)

# Get rich context for AI
result = await context.get_enriched_context(
    symbol="AuthController",
    include_documentation=True,
    include_related_issues=True,
    include_team_expertise=True
)
```

---

## Phase 4: MCP Integration

### Week 8-9: MCP Tools & Routing

#### Tasks

1. **MCP Bridge Implementation**
   - File: `knowledge_engine/integrations/arbor/mcp_bridge.py`
   - Implement MCP tools for Arbor capabilities
   - Integration with KE's MCP server

2. **Tool Registration**
   - Register Arbor tools with KE orchestration
   - Tool discovery and documentation
   - Permission and access control

3. **Prompt Templates**
   - File: `knowledge_engine/integrations/arbor/prompts/`
   - System prompts for code understanding
   - Refactoring assistant prompts
   - Code review assistant prompts

4. **Testing & Validation**
   - End-to-end MCP testing
   - Agent workflow validation
   - Performance benchmarking

#### MCP Tools Delivered

| Tool | Description | Use Case |
|------|-------------|----------|
| `arbor_find_definition` | Find symbol definition | "Where is UserService defined?" |
| `arbor_get_callers` | Get calling functions | "What calls authenticate()?" |
| `arbor_get_callees` | Get called functions | "What does PaymentService call?" |
| `arbor_find_path` | Find logic flow | "How does auth flow to the database?" |
| `arbor_analyze_impact` | Change impact analysis | "What breaks if I rename this?" |
| `arbor_get_context` | Get code context | "Show me relevant code for this task" |
| `arbor_validate_refactor` | Refactor validation | "Is this refactoring safe?" |
| `arbor_search` | Semantic code search | "Find functions that handle JWT" |

---

## Phase 5: Visualization

### Week 10: Visualizer Integration

#### Tasks

1. **Arbor Visualizer Bridge**
   - File: `knowledge_engine/integrations/arbor/visualizer.py`
   - Launch/connect to Arbor visualizer
   - Spotlight protocol integration

2. **KE Visualizer Integration**
   - Extend KE visualization with Arbor graph data
   - Unified graph view (code + knowledge)
   - Interactive node inspection

3. **Spotlight Protocol**
   - Synchronize AI focus between systems
   - When AI examines a node, highlight in visualizer
   - Camera animation to follow AI reasoning

#### Deliverables

```python
# Launch visualizer with current codebase
await knowledge_engine.arbor.launch_visualizer(
    highlight_nodes=["AuthController", "TokenMiddleware"],
    follow_ai_focus=True
)
```

---

## Testing Strategy

### Test Levels

```
┌─────────────────────────────────────────────────────────────┐
│                    TEST PYRAMID                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ▲                                        │
│                   / \   E2E Tests (5%)                      │
│                  /   \  - Full agent workflows              │
│                 /─────\                                     │
│                /       \  Integration Tests (15%)           │
│               /─────────\ - Arbor-KE integration            │
│              /           \- Graph sync                      │
│             /─────────────\                                 │
│            /               \ Unit Tests (80%)               │
│           /─────────────────\- Client, adapter, tools       │
│          /                   \- Schema mapping              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Test Files

```
knowledge_engine/tests/integrations/arbor/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_client.py                 # Unit: WebSocket client
├── test_adapter.py                # Unit: Graph adapter
├── test_schema_mapping.py         # Unit: Schema conversion
├── test_sync.py                   # Unit: Delta sync
├── test_mcp_bridge.py             # Unit: MCP tools
├── test_integration.py            # Integration: Full flow
└── test_e2e.py                    # E2E: Agent workflows
```

### Key Test Scenarios

1. **Connection Resilience**
   - Reconnection on WebSocket drop
   - Graceful degradation when Arbor unavailable

2. **Graph Consistency**
   - Verify node/edge counts match after import
   - Test incremental updates maintain consistency

3. **Query Accuracy**
   - Path finding returns valid paths
   - Impact analysis catches all dependencies

4. **Performance**
   - Import 100k nodes in < 5 minutes
   - Query response time < 100ms

---

## Deployment

### Architecture Options

#### Option A: Sidecar Pattern (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Pod                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │   Knowledge Engine      │  │        Arbor            │   │
│  │   (Python Container)    │◄─┤    (Rust Container)     │   │
│  │                         │  │    - arbor-server       │   │
│  │  - Web API              │  │    - arbor-watcher      │   │
│  │  - MCP Server           │  │    - Sled storage       │   │
│  │  - Graph sync           │  │                         │   │
│  └─────────────────────────┘  └─────────────────────────┘   │
│           │                            │                    │
│           └──────────┬─────────────────┘                    │
│                      │                                      │
│              Shared Volume (code checkout)                  │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Tight integration
- Shared filesystem
- Single deployment unit

**Cons:**
- Resource coupling
- Requires Kubernetes

#### Option B: Service Mesh

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   Knowledge Engine      │◄───────►│        Arbor            │
│   (Python Service)      │  HTTP   │    (Rust Service)       │
└─────────────────────────┘         └─────────────────────────┘
```

**Pros:**
- Independent scaling
- Language-agnostic
- Easier local development

**Cons:**
- Network latency
- More complex deployment

### Configuration

```yaml
# config/arbor.yaml
arbor:
  # Connection settings
  connection:
    ws_url: "ws://localhost:7433"
    reconnect_interval: 5
    max_reconnects: 10
    timeout: 30
  
  # Sync settings
  sync:
    mode: "realtime"  # realtime | batch | manual
    batch_size: 1000
    full_sync_interval: 3600  # seconds
  
  # Storage settings
  storage:
    graph_backend: "memgraph"  # memgraph | neo4j
    vector_backend: "qdrant"   # qdrant | pinecone
    cache_backend: "redis"     # redis | valkey
  
  # Indexing settings
  indexing:
    languages: ["python", "rust", "typescript"]
    exclude_patterns: ["*.test.py", "node_modules/**"]
    max_file_size: 1048576  # 1MB
  
  # MCP settings
  mcp:
    enabled: true
    tools:
      - arbor_find_definition
      - arbor_get_callers
      - arbor_analyze_impact
      - arbor_find_path
```

---

## Appendix

### A. ArborQL Reference

Arbor's query language for graph traversal:

```rust
// Find all functions that call authenticate
FIND function WHERE CALLS("authenticate")

// Find path from API to Database
PATH FROM "AuthController.validate" TO "UserRepository.find"

// Get all implementations of an interface
FIND class WHERE IMPLEMENTS("IAuthenticator")

// Complex query with filtering
FIND function 
  WHERE FILE("src/services/**") 
  AND CALLS("logger.error") 
  AND centrality > 0.5
```

### B. Node Type Mapping

| Arbor Kind | KE Entity Type | Description |
|------------|----------------|-------------|
| `function` | `code_function` | Standalone function |
| `method` | `code_method` | Class method |
| `class` | `code_class` | Class definition |
| `interface` | `code_interface` | Interface/protocol |
| `struct` | `code_struct` | Struct (Rust) |
| `enum` | `code_enum` | Enum definition |
| `variable` | `code_variable` | Module-level variable |
| `import` | `code_import` | Import statement |
| `module` | `code_module` | File/module boundary |

### C. Edge Type Mapping

| Arbor Kind | KE Relationship Type | Description |
|------------|----------------------|-------------|
| `calls` | `code_calls` | Function invocation |
| `imports` | `code_imports` | Import statement |
| `exports` | `code_exports` | Re-export |
| `extends` | `code_extends` | Class inheritance |
| `implements` | `code_implements` | Interface implementation |
| `uses_type` | `code_uses_type` | Type reference |
| `references` | `code_references` | General reference |
| `contains` | `code_contains` | Nesting relationship |

### D. Related Documents

- [Arbor README](../../arbor/arbor/README.md)
- [Arbor Architecture](../../arbor/arbor/docs/ARCHITECTURE.md)
- [Arbor Graph Schema](../../arbor/arbor/docs/GRAPH_SCHEMA.md)
- [Knowledge Engine README](../../knowledge_engine/README.md)
- [MCP Protocol](../../arbor/arbor/docs/PROTOCOL.md)

### E. Glossary

| Term | Definition |
|------|------------|
| **Arbor** | Rust-based code graph intelligence system |
| **ArborQL** | Query language for Arbor graph |
| **AST** | Abstract Syntax Tree |
| **Centrality** | Importance score of a node in graph |
| **KE** | Knowledge Engine (this system) |
| **MCP** | Model Context Protocol for AI agents |
| **Tree-sitter** | Fast incremental parser used by Arbor |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-01 | OpenEvolve | Initial specification |

---

*This document is a living specification. Updates will be tracked in the revision history.*
