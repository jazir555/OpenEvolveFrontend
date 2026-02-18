# RAGBits + Knowledge Engine + Task Agents Integration

## 🎯 Overview

Complete integration of RAGBits semantic search with the OpenEvolve Knowledge Engine and task agent system, enabling intelligent knowledge retrieval and context-aware agent execution.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Task Agents (Blue/Red/Gold)              │
│  - RAGBitsEnhancedBlueTeamAgent                             │
│  - Enhanced with semantic search capabilities               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Tools
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              RAGBits-Enhanced Agent Tools                    │
│  - RAGBitsKnowledgeSearchTool                               │
│  - RAGBitsContextGathererTool                               │
│  - RAGBitsArtifactIndexerTool                               │
│  - RAGBitsPatternAnalyzerTool                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Knowledge Retrieval
                     ↓
┌─────────────────────────────────────────────────────────────┐
│           RAGBitsEnhancedRetriever                          │
│  - Semantic search via RAGBits                             │
│  - Context-aware retrieval                                 │
│  - Multi-category search                                   │
│  - Artifact indexing                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Vector Search
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  RAGBits Document Search                    │
│  - Vector embeddings (OpenAI, LiteLLM, etc.)               │
│  - Vector stores (Qdrant, PGVector, Chroma)                │
│  - Hybrid search (semantic + keyword)                      │
└─────────────────────────────────────────────────────────────┘
                     │
                     │ Knowledge Base
                     ↓
┌─────────────────────────────────────────────────────────────┐
│            OpenEvolve Knowledge Engine                      │
│  - Document indexing                                       │
│  - Knowledge extraction                                    │
│  - Storage management                                      │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Components Created

### 1. RAGBits-Enhanced Retriever
**File:** `knowledge_engine/ragbits_retriever.py`

**Features:**
- ✅ Semantic vector search via RAGBits
- ✅ Similar solution search
- ✅ Decomposition pattern search
- ✅ Critique pattern search
- ✅ Verification benchmark search
- ✅ Context-aware retrieval
- ✅ Automatic artifact indexing
- ✅ Result caching

**Methods:**
```python
# Search for similar solutions
results = await retriever.search_similar_solutions(
    query="microservices authentication",
    top_k=5,
    filters={"stage": "stage_3"},
    min_success_rate=0.8
)

# Context-aware search
results = await retriever.search_contextual_knowledge(
    query="API design",
    context={"stage": "stage_3", "team": "blue"},
    top_k=5
)

# Index artifact
artifact_id = await retriever.ingest_artifact(
    content="Solution content...",
    metadata={"stage": "stage_3", "team": "blue"},
    artifact_type="solution"
)
```

### 2. RAGBits-Enhanced Agent Tools
**File:** `ragbits_integration/agents/tools/ragbits_enhanced_tools.py`

**Tools Included:**

#### a) RAGBitsKnowledgeSearchTool
Advanced semantic search with multiple search types:
- `similar_solutions` - Find similar solutions from history
- `decomposition_patterns` - Find decomposition strategies
- `critique_patterns` - Find critique and feedback patterns
- `verification_benchmarks` - Find test cases and benchmarks
- `contextual` - Context-aware search

```python
tool = RAGBitsKnowledgeSearchTool()
results = await tool.execute(
    search_type="similar_solutions",
    query="JWT authentication",
    top_k=5,
    filters={"stage": "stage_3"},
    min_success_rate=0.8
)
```

#### b) RAGBitsContextGathererTool
Gather comprehensive context from multiple knowledge sources:

```python
tool = RAGBitsContextGathererTool()
context = await tool.execute(
    query="user authentication system",
    sub_problem_id="sub_1",
    stage="stage_3",
    team="blue"
)

# Returns:
# {
#     "similar_solutions": [...],
#     "decomposition_patterns": [...],
#     "critique_patterns": [...],
#     "verification_benchmarks": [...]
# }
```

#### c) RAGBitsArtifactIndexerTool
Automatically index workflow artifacts:

```python
tool = RAGBitsArtifactIndexerTool()
artifact_id = await tool.execute(
    content="Solution for authentication...",
    metadata={
        "stage": "stage_3",
        "team": "blue",
        "sub_problem_id": "sub_1"
    },
    artifact_type="solution"
)
```

#### d) RAGBitsPatternAnalyzerTool
Analyze patterns in historical data:

```python
tool = RAGBitsPatternAnalyzerTool()
patterns = await tool.execute(
    analysis_type="solutions",
    query="successful authentication patterns",
    filters={"stage": "stage_3"}
)
```

### 3. RAGBits-Enhanced Blue Team Agent
**File:** `ragbits_integration/agents/examples/ragbits_enhanced_blue_team.py`

**Features:**
- ✅ Knowledge-aware solution generation
- ✅ Context gathering from historical data
- ✅ Automatic artifact indexing
- ✅ Pattern analysis capabilities

**Usage:**
```python
agent = RAGBitsEnhancedBlueTeamAgent(
    crewai_client=crewai,
    storage_manager=storage,
    enable_ragbits=True
)

# Generate solution with knowledge
result = await agent.execute(
    task="generate_solution",
    context={
        "sub_problem": sub_problem,
        "stage": "stage_3",
        "sub_problem_id": "sub_1"
    },
    use_knowledge=True
)

# Analyze patterns
analysis = await agent.execute(
    task="analyze_patterns",
    context={
        "query": "authentication patterns",
        "filters": {"stage": "stage_3"}
    }
)
```

## 🔌 Integration Points

### 1. Knowledge Engine Integration

**Integrate with existing KnowledgeEngine:**

```python
from knowledge_engine.engine import KnowledgeEngine
from knowledge_engine.ragbits_retriever import RAGBitsEnhancedRetriever

# Initialize knowledge engine
engine = KnowledgeEngine()

# Initialize RAGBits retriever
ragbits_retriever = RAGBitsEnhancedRetriever()

# Use RAGBits for semantic search
results = await ragbits_retriever.search_similar_solutions(
    query="microservices architecture",
    top_k=5
)
```

### 2. Task Agent Integration

**Enhance existing agents:**

```python
from ragbits_integration.agents.blue_team_agent import BlueTeamAgent
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsKnowledgeSearchTool,
    RAGBitsContextGathererTool
)

# Create standard blue team agent
agent = BlueTeamAgent(
    crewai_client=crewai,
    storage_manager=storage
)

# Add RAGBits tools
agent.tools["ragbits_search"] = RAGBitsKnowledgeSearchTool()
agent.tools["ragbits_context"] = RAGBitsContextGathererTool()

# Use in agent workflow
context = await agent.tools["ragbits_context"].execute(
    query="authentication system",
    sub_problem_id="sub_1",
    stage="stage_3"
)
```

### 3. Workflow Integration

**Integrate into decomposition workflow:**

```python
from decomposition_engine import DecompositionEngine
from knowledge_engine.ragbits_retriever import get_ragbits_retriever

# Initialize decomposition engine
decomposition_engine = DecompositionEngine()

# Get RAGBits retriever
ragbits = get_ragbits_retriever()

# Before blue team generates solution
sub_problem = decomposition_engine.get_sub_problem("sub_1")

# Search for similar solutions
similar = await ragbits.search_similar_solutions(
    query=sub_problem["description"],
    top_k=3,
    filters={"problem_type": sub_problem["type"]}
)

# Pass context to blue team
blue_team_context = {
    "sub_problem": sub_problem,
    "similar_solutions": similar
}

# Blue team generates solution with context
solution = await blue_team.generate(blue_team_context)

# Index the solution for future use
await ragbits.ingest_artifact(
    content=solution["content"],
    metadata={
        "stage": "stage_3",
        "team": "blue",
        "sub_problem_id": "sub_1"
    },
    artifact_type="solution"
)
```

## 🚀 Usage Examples

### Example 1: Blue Team with Knowledge

```python
import asyncio
from ragbits_integration.agents.examples.ragbits_enhanced_blue_team import (
    RAGBitsEnhancedBlueTeamAgent
)

async def main():
    # Create enhanced agent
    agent = RAGBitsEnhancedBlueTeamAgent(
        crewai_client=None,  # Or provide CrewAI client
        enable_ragbits=True
    )

    # Generate solution with knowledge
    result = await agent.execute(
        task="generate_solution",
        context={
            "sub_problem": {
                "description": "Implement JWT authentication",
                "constraints": "Use industry standards",
                "acceptance_criteria": "Tokens can be refreshed"
            },
            "stage": "stage_3",
            "sub_problem_id": "sub_1"
        },
        use_knowledge=True
    )

    print(f"Solution: {result['solution']['content']}")
    print(f"Similar solutions found: {len(result['knowledge_context']['similar_solutions'])}")

asyncio.run(main())
```

### Example 2: Red Team with Critique Patterns

```python
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsKnowledgeSearchTool
)

async def red_team_with_patterns():
    # Initialize tool
    search_tool = RAGBitsKnowledgeSearchTool()

    # Search for critique patterns
    patterns = await search_tool.execute(
        search_type="critique_patterns",
        query="security vulnerabilities in authentication",
        top_k=5,
        critique_type="security"
    )

    # Use patterns to inform critique
    for pattern in patterns:
        print(f"Pattern: {pattern['content']}")
        print(f"Relevance: {pattern['score']}")

asyncio.run(red_team_with_patterns())
```

### Example 3: Gold Team with Verification Benchmarks

```python
from knowledge_engine.ragbits_retriever import get_ragbits_retriever

async def gold_team_verification():
    # Get retriever
    retriever = get_ragbits_retriever()

    # Search for verification benchmarks
    benchmarks = await retriever.search_verification_benchmarks(
        query="authentication testing",
        top_k=5,
        filters={"stage": "stage_4"},
        min_coverage=0.8
    )

    # Use benchmarks to guide verification
    for benchmark in benchmarks:
        print(f"Benchmark: {benchmark['content']}")
        print(f"Coverage: {benchmark.get('coverage', 'N/A')}")

asyncio.run(gold_team_verification())
```

## 📊 Benefits

### 1. **Improved Solution Quality**
- ✅ Agents learn from historical successes
- ✅ Pattern-aware decision making
- ✅ Context-informed generation

### 2. **Faster Iteration**
- ✅ Reuse proven patterns
- ✅ Avoid common pitfalls
- ✅ Accelerated decision making

### 3. **Better Knowledge Management**
- ✅ Automatic artifact indexing
- ✅ Semantic search across all artifacts
- ✅ Cross-stage knowledge transfer

### 4. **Enhanced Collaboration**
- ✅ Share insights across teams
- ✅ Learn from other workflows
- ✅ Build organizational knowledge

## 🔧 Configuration

### RAGBits Configuration

```yaml
# ragbits_config.yaml
ragbits:
  vector_store:
    store_type: "qdrant"  # or "in_memory", "pgvector", "chroma"
    collection_name: "workflow_artifacts"
    embedding_model: "text-embedding-3-small"

  document_search:
    chunk_size: 500
    chunk_overlap: 50
    default_top_k: 5
    similarity_threshold: 0.75

  storage:
    enable_cache: true
    cache_ttl_seconds: 3600
    enable_versioning: true
```

### Agent Configuration

```python
# Initialize with custom config
from ragbits_integration.config import RagbitsIntegrationConfig

config = RagbitsIntegrationConfig(
    vector_store_config={
        "store_type": "qdrant",
        "qdrant_host": "localhost",
        "qdrant_port": 6333
    },
    document_search_config={
        "default_top_k": 10,
        "similarity_threshold": 0.8
    }
)

agent = RAGBitsEnhancedBlueTeamAgent(
    crewai_client=crewai,
    enable_ragbits=True
)

agent.ragbits_search.retriever = RAGBitsEnhancedRetriever(config)
```

## 🧪 Testing

### Test RAGBits Integration

```bash
# Test retriever
cd knowledge_engine
python -c "
import asyncio
from ragbits_retriever import get_ragbits_retriever

async def test():
    retriever = get_ragbits_retriever()
    results = await retriever.search_similar_solutions(
        'authentication system',
        top_k=3
    )
    print(f'Found {len(results)} results')

asyncio.run(test())
"
```

### Test Agent Tools

```bash
# Test knowledge search tool
cd ragbits_integration/agents/tools
python -c "
import asyncio
from ragbits_enhanced_tools import RAGBitsKnowledgeSearchTool

async def test():
    tool = RAGBitsKnowledgeSearchTool()
    results = await tool.execute(
        search_type='similar_solutions',
        query='API design',
        top_k=3
    )
    print(f'Results: {len(results)}')

asyncio.run(test())
"
```

## 📝 Next Steps

1. **Setup RAGBits Server**
   - Install RAGBits dependencies
   - Configure vector store (Qdrant recommended)
   - Start RAGBits API server

2. **Index Historical Data**
   - Import existing solutions
   - Index decomposition patterns
   - Add critique history

3. **Configure Agents**
   - Integrate tools into blue/red/gold teams
   - Set up automatic artifact indexing
   - Configure search parameters

4. **Monitor Performance**
   - Track search relevance
   - Monitor cache effectiveness
   - Measure agent improvement

5. **Iterate and Improve**
   - Fine-tune similarity thresholds
   - Optimize chunking strategies
   - Add custom metadata fields

## 🐛 Troubleshooting

### RAGBits Not Available

```python
# Check if RAGBits is available
from knowledge_engine.ragbits_retriever import get_ragbits_retriever

retriever = get_ragbits_retriever()
stats = await retriever.get_statistics()
print(f"RAGBits available: {stats['ragbits_available']}")
```

### No Search Results

- Check if documents are indexed
- Lower similarity threshold
- Try broader queries
- Verify filter criteria

### Performance Issues

- Enable caching
- Reduce top_k values
- Use hybrid search selectively
- Consider vector store optimization

## 📚 References

- [RAGBits Documentation](https://ragbits.com/docs)
- [Knowledge Engine Guide](./knowledge_engine/README.md)
- [Agent System Overview](./ragbits_integration/agents/README.md)
- [Integration Examples](./ragbits_integration/agents/examples/)

## ✅ Summary

This integration provides:
- ✅ 3 new Python modules (retriever, tools, enhanced agent)
- ✅ 4 new agent tools (search, context, indexer, analyzer)
- ✅ Complete example implementation
- ✅ Full documentation and usage guide

The system is ready for immediate use in task agent workflows!
