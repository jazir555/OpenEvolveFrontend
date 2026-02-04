# Research Quest - Curie-GlobalChem Integration

This integration enables Research Quest (the systematic scientific reasoning system) to leverage the combined power of Curie (the AI research experimentation agent) and GlobalChem (the comprehensive chemical knowledge graph) for conducting advanced chemistry-related research.

## Architecture

The integration follows the CLAUDE.md principles:

- **Zero Trust**: All inputs and outputs are validated
- **Anti-Hallucination**: Data integrity is verified through GlobalChem's curated datasets
- **Read-Only State**: Underlying systems' data remains unmodified
- **Idempotency**: Operations are safe to repeat
- **Configuration Explicitness**: All parameters are configurable via environment variables
- **UTC**: All timestamps are stored in UTC

## Components

### 1. ResearchQuestCurieGlobalChemAdapter
The main adapter class that bridges Research Quest with the Curie-GlobalChem integration, providing:

- Chemical search capabilities within research contexts
- Chemistry-focused research execution
- Chemical interaction analysis
- Research proposal generation combining all three systems

### 2. Interface Functions
Provides a clean API for Research Quest to access chemistry knowledge:

- `search`: Find chemicals relevant to research topics
- `research`: Conduct chemistry-focused research
- `interactions`: Analyze potential chemical interactions
- `proposal`: Generate research proposals

## Usage

### Direct Usage
```python
import asyncio
from research_quest_curie_globalchem_adapter import ResearchQuestCurieGlobalChemAdapter, create_research_interface

async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'curie_globalchem_config': {
            'log_level': 'WARNING'
        },
        'research_quest_config': {
            'model': 'openai/gpt-4o',
            'temperature': 0.1
        }
    }
    
    adapter = ResearchQuestCurieGlobalChemAdapter(config=config)
    
    # Create the interface for Research Quest
    research_interface = create_research_interface(adapter)
    
    # Search for chemicals related to a research topic
    chem_results = await research_interface('search', research_topic='aspirin synthesis pathways')
    print("Chemical search results:", chem_results)
    
    # Conduct a chemistry research
    research_results = await research_interface('research', research_question='What are the properties of aspirin and related compounds?')
    print("Research results:", research_results)
    
    # Analyze chemical interactions
    interactions = await research_interface('interactions', chemical_pairs=[('aspirin', 'caffeine')])
    print("Interaction analysis:", interactions)
    
    # Generate a research proposal
    proposal = await research_interface('proposal', topic='Development of new analgesic compounds')
    print("Research proposal:", proposal)

# Run the async function
asyncio.run(main())
```

### Integration with Research Quest
The adapter can be integrated into Research Quest's workflow to enable chemistry-focused research:

```python
import asyncio
from research_quest_curie_globalchem_adapter import ResearchQuestCurieGlobalChemAdapter, create_research_interface

async def enhanced_research_workflow():
    # Initialize the adapter
    adapter = ResearchQuestCurieGlobalChemAdapter()
    research_interface = create_research_interface(adapter)
    
    # Use the interface within Research Quest stages
    # For example, during hypothesis generation, use GlobalChem knowledge
    research_topic = "Investigate novel catalysts for CO2 reduction"
    
    # Search for relevant chemicals
    chem_findings = await research_interface('search', research_topic=research_topic)
    
    # Generate research proposal based on findings
    proposal = await research_interface('proposal', topic=research_topic)
    
    return proposal
```

## Configuration

The adapter supports the following configuration options:

- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `max_results`: Maximum number of results to return
- `timeout_seconds`: Timeout for operations
- `curie_globalchem_config`: Configuration for the Curie-GlobalChem integration
- `research_quest_config`: Configuration for the Research Quest integration

## Testing

Run the test suite:
```bash
python test_integration.py
```

## Deployment

The integration can be deployed as a container using the provided Dockerfile:

```bash
docker build -t research-quest-curie-globalchem-adapter .
docker run -d --name research-integration research-quest-curie-globalchem-adapter
```

## Security

This integration follows security best practices:

- No direct access to underlying systems' internal data structures
- Input validation for all queries
- Read-only access to data sources
- Proper error handling to prevent information disclosure
- Isolated execution environments for each system