# Knowledge Engine Integration Quick Start Guide

Get up and running with Knowledge Engine integrations in 5 minutes.

## Prerequisites

```bash
# Python 3.9+
python --version

# Install Knowledge Engine
pip install knowledge-engine
```

## 5-Minute Quick Start

### 1. Choose Your Integration

Pick the integration that matches your needs:

| Need | Use This | Install Command |
|------|----------|-----------------|
| Reasoning & problem-solving | DSPy | `pip install knowledge-engine[dspy]` |
| Knowledge extraction | DeepKE | `pip install knowledge-engine[deepke]` |
| Multi-agent workflows | CrewAI | `pip install knowledge-engine[crewai]` |
| Document search | Ragbits | `pip install knowledge-engine[ragbits]` |
| Complex problem solving | ROMA | Included in core |

### 2. Basic Usage

#### DSPy - Reasoning

```python
from knowledge_engine.integrations import DSPyIntegration

# Initialize
dspy = DSPyIntegration()

# Solve with chain-of-thought
result = await dspy.chain_of_thought(
    query="If I have 5 apples and eat 2, then buy 3 more, how many do I have?"
)

print(f"Answer: {result.output}")
print(f"Reasoning: {result.reasoning}")
```

#### DeepKE - Knowledge Extraction

```python
from knowledge_engine.integrations import DeepKEIntegration

# Initialize
deepke = DeepKEIntegration()

# Extract entities and relations
result = await deepke.extract_entities_relations(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California."
)

print(f"Entities: {result.entities}")
print(f"Relations: {result.relations}")
print(f"Triples: {result.triples}")
```

#### Ragbits - Document Search

```python
from knowledge_engine.integrations import RagbitsIntegration

# Initialize
ragbits = RagbitsIntegration()

# Ingest documents
await ragbits.ingest_documents([
    {"content": "Machine learning is a subset of AI..."},
    {"content": "Deep learning uses neural networks..."}
])

# Search
results = await ragbits.search(
    query="What is machine learning?",
    top_k=5
)

for doc in results:
    print(f"Score: {doc.score}")
    print(f"Content: {doc.content}")
```

#### CrewAI - Multi-Agent

```python
from knowledge_engine.integrations import CrewAIIntegration

# Initialize
crewai = CrewAIIntegration()

# Create agents and tasks
researcher = {
    "role": "Researcher",
    "goal": "Find information",
    "backstory": "Expert researcher"
}

writer = {
    "role": "Writer",
    "goal": "Write content",
    "backstory": "Expert writer"
}

# Create crew
await crewai.create_crew(
    crew_id="my_crew",
    agents=[researcher, writer],
    tasks=[
        {"description": "Research AI", "agent": "researcher"},
        {"description": "Write article", "agent": "writer"}
    ]
)

# Execute
result = await crewai.execute_crew(
    crew_id="my_crew",
    inputs={"topic": "artificial intelligence"}
)
```

#### ROMA - Complex Problem Solving

```python
from knowledge_engine.integrations import ROMAIntegration

# Initialize
roma = ROMAIntegration()

# Solve complex problem
result = await roma.solve(
    problem="Design a scalable microservices architecture",
    domain="software_engineering"
)

print(f"Solution: {result.solution}")
print(f"Decomposition: {result.decomposition}")
```

## Common Patterns

### Pattern 1: RAG (Retrieval-Augmented Generation)

```python
from knowledge_engine.integrations import RagbitsIntegration, DSPyIntegration

# Setup
ragbits = RagbitsIntegration()
dspy = DSPyIntegration()

# Ingest knowledge base
await ragbits.ingest_documents(your_documents)

# Query with RAG
async def ask_question(question: str):
    # 1. Retrieve relevant docs
    docs = await ragbits.search(query=question, top_k=5)

    # 2. Generate answer with context
    answer = await dspy.chain_of_thought(
        query=question,
        context={"documents": docs}
    )

    return answer.output

# Use
result = await ask_question("What is machine learning?")
```

### Pattern 2: Knowledge Extraction Pipeline

```python
from knowledge_engine.integrations import DeepKEIntegration, ROMAEntityExtractor

# Setup
deepke = DeepKEIntegration()
roma = ROMAEntityExtractor()

# Extract and store
async def extract_and_store(text: str):
    # Extract knowledge
    result = await deepke.extract_entities_relations(text)

    # Store in knowledge graph
    for entity in result.entities:
        await roma.add_entity(
            entity_type=entity["type"],
            name=entity["text"]
        )

    for relation in result.relations:
        await roma.add_relation(
            from_entity=relation["head"],
            relation_type=relation["relation"],
            to_entity=relation["tail"]
        )

# Use
await extract_and_store("Apple was founded by Steve Jobs.")
```

### Pattern 3: Multi-Agent Problem Solving

```python
from knowledge_engine.integrations import ROMAIntegration, CrewAIIntegration

# Setup
roma = ROMAIntegration()
crewai = CrewAIIntegration()

# Decompose and solve
async def solve_complex_problem(problem: str):
    # 1. Decompose with ROMA
    decomposition = await roma.decompose(problem)

    # 2. Create agents for subproblems
    agents = []
    tasks = []
    for subproblem in decomposition.subproblems:
        agent = {
            "role": f"Specialist for {subproblem['type']}",
            "goal": f"Solve {subproblem['description']}"
        }
        task = {
            "description": subproblem["description"],
            "agent": agent["role"]
        }
        agents.append(agent)
        tasks.append(task)

    # 3. Coordinate with CrewAI
    await crewai.create_crew(
        crew_id="problem_solvers",
        agents=agents,
        tasks=tasks
    )

    # 4. Execute
    result = await crewai.execute_crew(crew_id="problem_solvers")
    return result

# Use
solution = await solve_complex_problem("Design a distributed system")
```

## Configuration

### Environment Variables

```bash
# Required for most integrations
export OPENAI_API_KEY="your-openai-key"

# Optional: DSPy
export DSPY_MODEL="gpt-4o"

# Optional: Ragbits
export RAGBITS_VECTOR_STORE="qdrant"
export QDRANT_URL="http://localhost:6333"

# Optional: DeepKE
export DEEPKE_DEVICE="cuda"  # or "cpu"
```

### Configuration File

```python
# config.py
DSPY_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 4096
}

DEEPKE_CONFIG = {
    "model_type": "standard",
    "device": "cuda",
    "batch_size": 16
}

RAGBITS_CONFIG = {
    "vector_store": {
        "type": "qdrant",
        "config": {
            "location": "http://localhost:6333"
        }
    }
}

# Use
from knowledge_engine.integrations import DSPyIntegration
dspy = DSPyIntegration(config=DSPY_CONFIG)
```

## Next Steps

### Learn More
- [Full Documentation](./README.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Integration Index](./INTEGRATION_INDEX.md)

### Explore Integrations
- [DSPy Guide](./DSPY_INTEGRATION.md)
- [DeepKE Guide](./DEEPKE_INTEGRATION.md)
- [CrewAI Guide](./CREWAI_INTEGRATION.md)
- [Ragbits Guide](./RAGBITS_INTEGRATION.md)
- [ROMA Guide](./ROMA_INTEGRATION.md)

### Advanced Topics
- Cross-integration patterns
- Performance optimization
- Error handling strategies
- Security best practices

## Common Issues

### Issue: Import Error
```python
# Error: ImportError: No module named 'dspy'
# Solution: Install the integration
pip install knowledge-engine[dspy]
```

### Issue: API Key Missing
```python
# Error: API key not found
# Solution: Set environment variable
export OPENAI_API_KEY="your-key"
```

### Issue: CUDA Out of Memory (DeepKE)
```python
# Solution: Use CPU or reduce batch size
config = {
    "device": "cpu",
    "batch_size": 8
}
```

### Issue: Vector Store Connection (Ragbits)
```python
# Solution: Check vector store is running
# For Qdrant:
docker run -p 6333:6333 qdrant/qdrant
```

## Get Help

- **Documentation**: See full guides above
- **Examples**: Check `examples/` directory
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions

## Cheat Sheet

```python
# DSPy: Reasoning
from knowledge_engine.integrations import DSPyIntegration
result = await DSPyIntegration().chain_of_thought(query)

# DeepKE: Extraction
from knowledge_engine.integrations import DeepKEIntegration
result = await DeepKEIntegration().extract_entities_relations(text)

# Ragbits: Search
from knowledge_engine.integrations import RagbitsIntegration
await RagbitsIntegration().ingest_documents(docs)
results = await RagbitsIntegration().search(query)

# CrewAI: Orchestration
from knowledge_engine.integrations import CrewAIIntegration
await CrewAIIntegration().create_crew(crew_id, agents, tasks)
result = await CrewAIIntegration().execute_crew(crew_id)

# ROMA: Decomposition
from knowledge_engine.integrations import ROMAIntegration
result = await ROMAIntegration().solve(problem, domain)
```

---

**Ready to build?** Start with the integration that matches your use case, and refer to the detailed guides for advanced features!

**Last Updated**: 2025-02-03
