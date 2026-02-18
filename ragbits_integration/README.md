# RAGBits Integration for Decomposition Workflow

## Overview

This package integrates RAGBits (Rapid AI Building Blocks) into the Sovereign-Grade Decomposition Workflow, providing:

- **Real-time intermediary storage** during workflow execution
- **Semantic search** over historical solutions and patterns
- **Cross-stage context retrieval** for agents
- **Artifact lifecycle management** with versioning
- **Hybrid knowledge management** bridging RAGBits and existing KB

## Installation

```bash
# Install RAGBits (if not already installed)
pip install ragbits-document-search ragbits-core

# The integration is a local package
# No additional installation needed
```

## Quick Start

```python
from ragbits_integration import (
    IntermediaryStorageManager,
    ContextGatherer,
    RagbitsKnowledgeRetriever
)
from ragbits.document_search import DocumentSearch
from ragbits.core.vector_stores import InMemoryVectorStore
from ragbits.core.embeddings import LiteLLMEmbedder

# Setup
embedder = LiteLLMEmbedder(model_name="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedder=embedder)
document_search = DocumentSearch(vector_store=vector_store)

# Initialize components
storage = IntermediaryStorageManager(document_search)
gatherer = ContextGatherer(storage)
retriever = RagbitsKnowledgeRetriever(document_search)

# Use during workflow
# Store artifacts immediately as they're generated
artifact_id = await storage.store_artifact(
    artifact_type="solution_draft",
    content="Microservices architecture with load balancing...",
    metadata={
        "stage": "stage_3",
        "team": "blue",
        "sub_problem_id": "sub_1"
    }
)

# Retrieve context for agents
context = await gatherer.gather_for_blue_team(
    sub_problem_id="sub_1",
    problem_description="Implement user authentication"
)

# Search for similar solutions from history
similar = await retriever.retrieve_similar_solutions(
    problem_description="User authentication system",
    top_k=5
)
```

## Architecture

```
Decomposition Workflow
    ↓
IntermediaryStorageManager (real-time storage)
    ↓
RAGBits DocumentSearch (vector store)
    ↓
Vector Store (InMemory / Qdrant / PGVector)
```

## Key Components

### 1. IntermediaryStorageManager

Manages real-time storage and retrieval of workflow artifacts.

```python
# Store artifact
artifact_id = await storage.store_artifact(
    artifact_type="solution_draft",
    content="...",
    metadata={"team": "blue", "sub_problem_id": "sub_1"}
)

# Retrieve context
context = await storage.retrieve_context_for_stage(
    stage="stage_3_red_team_critique",
    sub_problem_id="sub_1"
)

# Get artifact chain (linked artifacts)
chain = await storage.get_artifact_chain(artifact_id)
```

### 2. ContextGatherer

High-level API for agents to gather relevant context.

```python
# Blue Team context
blue_context = await gatherer.gather_for_blue_team(
    sub_problem_id="sub_1",
    problem_description="..."
)

# Red Team context
red_context = await gatherer.gather_for_red_team(
    sub_problem_id="sub_1"
)

# Gold Team context
gold_context = await gatherer.gather_for_gold_team(
    sub_problem_id="sub_1"
)
```

### 3. RagbitsKnowledgeRetriever

Semantic search over historical knowledge.

```python
# Find similar solutions
solutions = await retriever.retrieve_similar_solutions(
    problem_description="Build REST API",
    top_k=5,
    min_success_rate=0.8
)

# Find decomposition patterns
patterns = await retriever.retrieve_relevant_decompositions(
    problem_type="distributed_systems",
    complexity=8.5
)
```

### 4. ArtifactLifecycleManager

Manages artifact state transitions.

```python
# Create draft
artifact_id = await lifecycle.create_draft(
    artifact_type="solution_draft",
    content="...",
    metadata={"team": "blue"}
)

# Transition through lifecycle
await lifecycle.transition_to_pending(artifact_id)
await lifecycle.transition_to_verified(artifact_id)
await lifecycle.transition_to_final(artifact_id)
```

### 5. HybridKnowledgeManager

Bridges RAGBits with existing knowledge base.

```python
from ragbits_integration.hybrid_knowledge import HybridKnowledgeManager

hybrid = HybridKnowledgeManager(
    ragbits_store=document_search,
    existing_kb=knowledge_base  # Optional
)

# Store in both systems
result = await hybrid.store_artifact(
    artifact={"content": "...", "metadata": {...}},
    stage="stage_3"
)

# Retrieve from both sources
context = await hybrid.retrieve_context(
    query="authentication patterns",
    filters={"type": "solution"}
)
```

## Configuration

```python
from ragbits_integration.config import (
    RagbitsIntegrationConfig,
    get_default_config,
    get_production_config
)

# Use defaults
config = get_default_config()

# Or production config
config = get_production_config()

# Or from environment
config = RagbitsIntegrationConfig.from_env()

# Or from dict
config = RagbitsIntegrationConfig.from_dict({
    "vector_store": {
        "store_type": "qdrant",
        "qdrant_host": "localhost",
        "qdrant_port": 6333
    },
    "storage": {
        "enable_cache": True,
        "cache_max_size": 10000
    }
})

# Validate
config.validate()
```

## Environment Variables

```bash
# Vector Store
export RAGBITS_VECTOR_STORE_TYPE=qdrant  # in_memory, qdrant, pgvector
export RAGBITS_QDRANT_HOST=localhost
export RAGBITS_QDRANT_PORT=6333
export RAGBITS_QDRANT_API_KEY=your_api_key

# PGVector (if using)
export RAGBITS_PGVECTOR_CONN="postgresql://..."

# Embeddings
export RAGBITS_EMBEDDING_MODEL=text-embedding-3-small

# CrewAI
export RAGBITS_CREWAI_ENDPOINT=http://localhost:8000

# Logging
export RAGBITS_LOG_LEVEL=INFO
```

## Testing

```bash
# Run all tests
python -m pytest ragbits_integration/tests/

# Run specific test file
python -m pytest ragbits_integration/tests/test_storage_manager.py

# Run with coverage
python -m pytest --cov=ragbits_integration ragbits_integration/tests/

# Run integration tests manually
python ragbits_integration/tests/test_integration.py
```

## Artifact Lifecycle

```
draft → pending → verified → final
         ↓
      rejected
```

## Artifact Types

- `content_analysis` - Stage 0 analysis results
- `decomposition_plan` - Stage 1 decomposition plans
- `solution_draft` - Stage 3 solution drafts
- `critique` - Stage 3 critique reports
- `verification` - Stage 3 verification reports
- `assembled_solution` - Stage 4 assembled solutions
- `final_verification` - Stage 5 final verification

## Stage Integration

### Stage 0: Content Analysis
```python
await storage.store_artifact(
    artifact_type="content_analysis",
    content="Problem complexity: High",
    metadata={"stage": "stage_0", "complexity": 8.5}
)
```

### Stage 1: Decomposition
```python
context = await gatherer.gather_for_decomposition(
    problem_description="Build scalable system"
)
```

### Stage 3: Sub-Problem Solving
```python
# Blue Team
blue_context = await gatherer.gather_for_blue_team(
    sub_problem_id="sub_1",
    problem_description="Implement auth"
)

# Red Team
red_context = await gatherer.gather_for_red_team(
    sub_problem_id="sub_1"
)

# Gold Team
gold_context = await gatherer.gather_for_gold_team(
    sub_problem_id="sub_1"
)
```

### Stage 4: Reassembly
```python
context = await gatherer.gather_for_reassembly()
```

### Stage 5: Final Verification
```python
context = await gatherer.gather_for_final_verification(
    assembled_solution="..."
)
```

## Files Structure

```
ragbits_integration/
├── __init__.py
├── config.py                          # Configuration management
├── hybrid_knowledge.py                # Hybrid KB manager
├── intermediary_storage/
│   ├── __init__.py
│   ├── storage_manager.py             # Core storage manager
│   ├── artifact_lifecycle.py          # Lifecycle management
│   └── context_gatherer.py            # Context gathering API
├── document_search/
│   ├── __init__.py
│   └── knowledge_retriever.py         # Semantic search
└── tests/
    ├── __init__.py
    ├── test_storage_manager.py
    ├── test_artifact_lifecycle.py
    ├── test_context_gatherer.py
    ├── test_config.py
    └── test_integration.py
```

## Features

### Phase 1: Storage & Retrieval ✅
- ✅ Real-time artifact storage and indexing
- ✅ Semantic search over all artifacts
- ✅ Cross-stage context retrieval
- ✅ Artifact linking and chaining
- ✅ Versioned history with rollback
- ✅ Lifecycle state management
- ✅ In-memory caching for performance
- ✅ Hybrid knowledge (RAGBits + existing KB)

### Phase 2: Agent Coordination ✅
- ✅ Blue Team agent (solution generation)
- ✅ Red Team agent (critique)
- ✅ Gold Team agent (verification)
- ✅ A2A protocol for inter-agent messaging
- ✅ Agent tools (knowledge search, evaluation, patterns)
- ✅ CrewAI LLM integration
- ✅ Message routing and delivery
- ✅ Request/response tracking
- ✅ Refinement workflow support
- ✅ Comprehensive unit and integration tests

### Phase 3: Evaluation Framework ✅
- ✅ Multi-dimensional metrics collection (8 categories, 20+ metric types)
- ✅ Metrics analyzer with category scoring
- ✅ Enhanced gauntlet validation with multi-dimensional scoring
- ✅ Historical comparison with trend analysis
- ✅ Evaluation dashboard generation
- ✅ HTML report export
- ✅ Percentile ranking and insight generation
- ✅ Comprehensive test coverage

### Phase 4: Enhanced Knowledge Base ✅
- ✅ Automatic knowledge extraction (10 entity types)
- ✅ Knowledge enrichment with context and quality scoring
- ✅ Vector indexing optimization (5 strategies)
- ✅ Advanced RAG engine with hybrid search
- ✅ LLM-based query expansion and reranking
- ✅ Entity deduplication and linking
- ✅ Comprehensive test coverage

### Phase 5: UI/CLI Integration ✅
- ✅ CLI tools with 8 commands (extract, score, compare, dashboard, explore, stats, validate, trend)
- ✅ Review interface with inline commenting, threading, and version comparison
- ✅ Monitoring dashboard with metric collection, alerts, and HTML export
- ✅ Knowledge explorer with multi-strategy search and faceted filtering
- ✅ Comprehensive test coverage

## Implementation Phases

### Phase 1: Document Search & Intermediary Storage ✅ COMPLETE
- Real-time storage and retrieval system
- Semantic search over artifacts
- Cross-stage context gathering
- Artifact lifecycle management
- Hybrid knowledge bridge

**Documentation**: See Phase 1 features above

### Phase 2: Agent Coordination with A2A ✅ COMPLETE
- Base workflow agent with CrewAI
- Blue, Red, and Gold team agents
- Agent tools (search, evaluation, patterns)
- A2A communication protocol
- Message routing and delivery

**Documentation**: [agents/README.md](agents/README.md)

### Phase 3: Evaluation Framework ✅ COMPLETE
- Multi-dimensional metrics collection and analysis
- Enhanced gauntlet validation with 8 dimensions
- Historical comparison and trend analysis
- Evaluation dashboard generation
- HTML report export

**Documentation**: [evaluation/README.md](evaluation/README.md)

### Phase 4: Enhanced Knowledge Base ✅ COMPLETE
- Automatic knowledge extraction with 10 entity types
- Knowledge enrichment and quality scoring
- Vector indexing optimization with 5 strategies
- Advanced RAG engine with hybrid search and reranking
- Query expansion and result deduplication

**Documentation**: [knowledge_base/README.md](knowledge_base/README.md)

### Phase 5: UI/CLI Integration ✅ COMPLETE
- CLI tools for RAGBits operations
- Review interface with collaborative features
- Monitoring dashboards with alerts
- Interactive knowledge exploration

**Documentation**: [ui_cli/README.md](ui_cli/README.md)

## Dependencies

```
ragbits-core
ragbits-document-search
ragbits-agents (for Phase 2)
ragbits-evaluate (for Phase 3)
```

## License

Part of the OpenEvolve Decomposition Workflow project.
