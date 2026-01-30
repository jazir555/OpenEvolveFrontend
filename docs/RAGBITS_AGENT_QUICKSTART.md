# RAGBits + Knowledge Engine + Task Agents - Quick Start

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
# Ensure RAGBits is installed
pip install ragbits-document-search ragbits-core

# Or if using local RAGBits
cd ragbits
pip install -e .
```

### 2. Initialize RAGBits Retriever

```python
from knowledge_engine.ragbits_retriever import get_ragbits_retriever

# Get singleton instance (auto-initializes)
retriever = get_ragbits_retriever()

# Check availability
stats = await retriever.get_statistics()
print(f"RAGBits available: {stats['ragbits_available']}")
```

### 3. Use in Task Agents

```python
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsKnowledgeSearchTool,
    RAGBitsContextGathererTool
)
from ragbits_integration.agents.blue_team_agent import BlueTeamAgent

# Create agent
agent = BlueTeamAgent(hephaestus_client=hephaestus)

# Add RAGBits tools
agent.tools["ragbits_search"] = RAGBitsKnowledgeSearchTool()
agent.tools["ragbits_context"] = RAGBitsContextGathererTool()

# Use tools
context = await agent.tools["ragbits_context"].execute(
    query="authentication system",
    sub_problem_id="sub_1",
    stage="stage_3"
)

# Access results
print(f"Similar solutions: {len(context['similar_solutions'])}")
print(f"Patterns: {len(context['decomposition_patterns'])}")
```

### 4. Enhanced Agent (Complete Example)

```python
from ragbits_integration.agents.examples.ragbits_enhanced_blue_team import (
    RAGBitsEnhancedBlueTeamAgent,
    demo_ragbits_blue_team
)

# Run demo
import asyncio
asyncio.run(demo_ragbits_blue_team())
```

## 📋 Common Tasks

### Search Similar Solutions

```python
retriever = get_ragbits_retriever()

results = await retriever.search_similar_solutions(
    query="microservices authentication",
    top_k=5,
    filters={"stage": "stage_3"},
    min_success_rate=0.8
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:200]}...")
```

### Gather Context for Agent

```python
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsContextGathererTool
)

tool = RAGBitsContextGathererTool()

context = await tool.execute(
    query="API rate limiting",
    sub_problem_id="sub_2",
    stage="stage_3",
    team="blue",
    max_results_per_category=3
)

# Use context in agent prompt
similar = context['similar_solutions']
patterns = context['decomposition_patterns']
```

### Index Generated Solutions

```python
retriever = get_ragbits_retriever()

artifact_id = await retriever.ingest_artifact(
    content="Implement JWT authentication with refresh tokens...",
    metadata={
        "stage": "stage_3",
        "team": "blue",
        "sub_problem_id": "sub_1",
        "success_rate": 0.9
    },
    artifact_type="solution"
)

print(f"Indexed: {artifact_id}")
```

### Analyze Patterns

```python
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsPatternAnalyzerTool
)

tool = RAGBitsPatternAnalyzerTool()

analysis = await tool.execute(
    analysis_type="solutions",
    query="successful authentication patterns",
    filters={"stage": "stage_3"}
)

print(f"Patterns found: {len(analysis['patterns'])}")
for insight in analysis['insights']:
    print(f"- {insight}")
```

## 🎯 Integration Patterns

### Pattern 1: Blue Team with Knowledge

```python
async def blue_team_with_knowledge():
    from ragbits_integration.agents.examples.ragbits_enhanced_blue_team import (
        RAGBitsEnhancedBlueTeamAgent
    )

    agent = RAGBitsEnhancedBlueTeamAgent(
        hephaestus_client=hephaestus,
        enable_ragbits=True
    )

    result = await agent.execute(
        task="generate_solution",
        context={
            "sub_problem": {
                "description": "Implement OAuth2 flow",
                "constraints": "Use RFC 6749 standard"
            },
            "stage": "stage_3",
            "sub_problem_id": "sub_1"
        },
        use_knowledge=True  # Enable RAGBits search
    )

    return result['solution']
```

### Pattern 2: Red Team with Critique Patterns

```python
async def red_team_with_patterns():
    from ragbits_integration.agents.red_team_agent import RedTeamAgent
    from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
        RAGBitsKnowledgeSearchTool
    )

    agent = RedTeamAgent(hephaestus_client=hephaestus)
    agent.tools["ragbits_search"] = RAGBitsKnowledgeSearchTool()

    # Search for critique patterns
    patterns = await agent.tools["ragbits_search"].execute(
        search_type="critique_patterns",
        query="security issues in authentication",
        top_k=5
    )

    # Use patterns to inform critique
    critique_prompt = """
    Review this solution for security issues:
    {solution}

    Consider these historical issues:
    {patterns}
    """.format(
        solution=solution_content,
        patterns="\n".join([p['content'] for p in patterns])
    )

    critique = await agent._call_llm(critique_prompt)
    return critique
```

### Pattern 3: Gold Team with Benchmarks

```python
async def gold_team_with_benchmarks():
    from knowledge_engine.ragbits_retriever import get_ragbits_retriever

    retriever = get_ragbits_retriever()

    # Find verification benchmarks
    benchmarks = await retriever.search_verification_benchmarks(
        query="authentication testing",
        top_k=5,
        min_coverage=0.8
    )

    # Use benchmarks to guide verification
    test_cases = []
    for benchmark in benchmarks:
        test_cases.extend(benchmark.get('test_cases', []))

    # Run tests with gold team
    results = await gold_team.verify(
        solution=solution,
        test_cases=test_cases
    )

    return results
```

## 🔧 Configuration

### Basic Config

```python
from knowledge_engine.ragbits_retriever import RAGBitsEnhancedRetriever

# Use defaults
retriever = RAGBitsEnhancedRetriever()
```

### Custom Config

```python
# With custom RAGBits config
ragbits_config = {
    "vector_store": {
        "store_type": "qdrant",
        "qdrant_host": "localhost",
        "qdrant_port": 6333
    },
    "document_search": {
        "default_top_k": 10,
        "similarity_threshold": 0.8
    }
}

retriever = RAGBitsEnhancedRetriever(
    ragbits_config=ragbits_config,
    enable_cache=True,
    cache_ttl=3600
)
```

### Disable RAGBits (Fallback)

```python
# Works without RAGBits - returns mock results
retriever = RAGBitsEnhancedRetriever()

# Check availability
stats = await retriever.get_statistics()
if not stats['ragbits_available']:
    print("⚠️ RAGBits not available, using fallback")
```

## 📊 Performance Tips

### 1. Enable Caching

```python
retriever = RAGBitsEnhancedRetriever(
    enable_cache=True,
    cache_ttl=3600  # 1 hour
)
```

### 2. Use Appropriate top_k Values

```python
# Faster searches with fewer results
quick_results = await retriever.search_similar_solutions(
    query="API design",
    top_k=3  # Instead of 10
)
```

### 3. Filter by Metadata

```python
# Narrow down search space
results = await retriever.search_similar_solutions(
    query="authentication",
    filters={
        "stage": "stage_3",
        "team": "blue"
    }
)
```

### 4. Batch Operations

```python
# Index multiple artifacts at once
artifacts = [
    {"content": "...", "metadata": {...}},
    {"content": "...", "metadata": {...}},
    {"content": "...", "metadata": {...}}
]

for artifact in artifacts:
    await retriever.ingest_artifact(
        content=artifact["content"],
        metadata=artifact["metadata"]
    )
```

## 🧪 Testing

### Test Search

```bash
python -c "
import asyncio
from knowledge_engine.ragbits_retriever import get_ragbits_retriever

async def test():
    retriever = get_ragbits_retriever()
    results = await retriever.search_similar_solutions(
        'authentication',
        top_k=3
    )
    print(f'Found {len(results)} results')

asyncio.run(test())
"
```

### Test Tools

```bash
python -c "
import asyncio
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsContextGathererTool
)

async def test():
    tool = RAGBitsContextGathererTool()
    context = await tool.execute(
        'API design',
        sub_problem_id='sub_1'
    )
    print(f'Solutions: {len(context[\"similar_solutions\"])}')

asyncio.run(test())
"
```

## 🐛 Troubleshooting

### Import Errors

```python
# If you get import errors, check paths
import sys
sys.path.insert(0, '.')  # Add project root

from knowledge_engine.ragbits_retriever import get_ragbits_retriever
```

### RAGBits Not Available

```python
# Check availability
retriever = get_ragbits_retriever()
stats = await retriever.get_statistics()

if not stats['ragbits_available']:
    print("Install RAGBits:")
    print("pip install ragbits-document-search ragbits-core")
```

### No Search Results

```python
# Try with lower threshold
results = await retriever.search_similar_solutions(
    query="broad query",
    top_k=10,
    min_success_rate=0.0  # No threshold
)
```

## 📚 Next Steps

1. ✅ **Setup RAGBits server** - Install and configure vector store
2. ✅ **Index historical data** - Import existing solutions
3. ✅ **Configure agents** - Add tools to your agents
4. ✅ **Run workflow** - Test with decomposition workflow
5. ✅ **Monitor performance** - Track search relevance and cache hits

## 🎓 Learn More

- [Full Integration Guide](./RAGBITS_KNOWLEDGE_ENGINE_INTEGRATION.md)
- [RAGBits Plugin](./bubblelabs-ragbits-plugin/README.md)
- [Knowledge Engine](./knowledge_engine/README.md)
- [Agent System](./ragbits_integration/agents/README.md)

---

**Ready to use!** The integration is complete and ready for immediate deployment in your task agent workflows.
