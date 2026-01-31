# 5-Minute Quick Start

Get up and running with the OpenEvolve Knowledge Engine in 5 minutes.

## Prerequisites (1 minute)

```bash
# Check Python version (3.9+)
python --version

# Check Neo4j is running (optional, for graph features)
neo4j status
```

## Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/openevolve/frontend.git
cd Frontend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="your-password"
```

## Your First Knowledge Query (2 minutes)

Create `quickstart.py`:

```python
import asyncio
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine
from datetime import datetime

async def main():
    # Initialize engine
    engine = TemporalKnowledgeEngine()

    # Add some knowledge
    await engine.add_knowledge_temporal(
        content="Python is a programming language created by Guido van Rossum",
        artifact_type="fact",
        valid_at=datetime(2024, 1, 1),
        metadata={"category": "programming"}
    )

    await engine.add_knowledge_temporal(
        content="JavaScript is used for web development",
        artifact_type="fact",
        valid_at=datetime(2024, 1, 1),
        metadata={"category": "programming"}
    )

    # Query knowledge
    results = await engine.search_with_graphiti(
        query="programming languages",
        max_results=5
    )

    # Display results
    for i, artifact in enumerate(results, 1):
        print(f"{i}. {artifact.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python quickstart.py
```

## Next Steps

- Read [Developer Setup Guide](developer_setup.md) for detailed setup
- Explore [Sample Data Tutorial](sample_data_tutorial.md)
- Check [Tutorials](../tutorials/) for in-depth guides

## Troubleshooting

**Import Error**: Make sure you're in the `Frontend` directory
**Neo4j Connection**: Ensure Neo4j is running or disable Graphiti features
**API Key Error**: Set `OPENAI_API_KEY` environment variable
