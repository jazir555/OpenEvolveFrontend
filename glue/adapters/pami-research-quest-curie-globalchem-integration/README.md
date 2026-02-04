# PAMI - Research Quest - Curie-GlobalChem Integration

This integration enables PAMI (Pattern Analysis and Machine Intelligence) to work seamlessly with the Research Quest - Curie-GlobalChem system for conducting advanced scientific research with comprehensive pattern analysis capabilities.

## Architecture

The integration follows the CLAUDE.md principles:

- **Zero Trust**: All inputs and outputs are validated
- **Anti-Hallucination**: Data integrity is verified through GlobalChem's curated datasets
- **Read-Only State**: Underlying systems' data remains unmodified
- **Idempotency**: Operations are safe to repeat
- **Configuration Explicitness**: All parameters are configurable via environment variables
- **UTC**: All timestamps are stored in UTC

## Components

### 1. PAMIResearchQuestCurieGlobalChemAdapter
The main adapter class that bridges PAMI with the Research Quest - Curie-GlobalChem integration, providing:

- Pattern analysis in research data
- Pattern-enriched research execution
- Chemical knowledge graph pattern analysis
- Pattern-based research proposal generation

### 2. Interface Functions
Provides a clean API for unified access to all systems:

- `pattern_analysis`: Analyze patterns in research data
- `enriched_research`: Conduct research enriched with pattern analysis
- `graph_analysis`: Analyze patterns in knowledge graphs
- `proposal`: Generate pattern-based research proposals

## Usage

### Direct Usage
```python
import asyncio
from pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter, create_unified_interface

async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'min_support': 0.1,
        'min_confidence': 0.5,
        'research_quest_curie_globalchem_config': {
            'log_level': 'WARNING',
            'curie_globalchem_config': {
                'log_level': 'WARNING'
            },
            'research_quest_config': {
                'model': 'openai/gpt-4o',
                'temperature': 0.1
            }
        },
        'pami_config': {}
    }
    
    adapter = PAMIResearchQuestCurieGlobalChemAdapter(config=config)
    
    # Create the unified interface
    unified_interface = create_unified_interface(adapter)
    
    # Analyze patterns in research data
    sample_data = {
        'transactions': [
            ['aspirin', 'pain_relief', 'acetylation'],
            ['ibuprofen', 'pain_relief', 'inflammation'],
            ['acetaminophen', 'pain_relief', 'fever_reduction']
        ]
    }
    
    pattern_results = await unified_interface('pattern_analysis', research_data=sample_data)
    print("Pattern analysis results:", pattern_results)
    
    # Conduct pattern-enriched research
    research_results = await unified_interface('enriched_research', research_question='What are effective pain relief medications?')
    print("Pattern-enriched research results:", research_results)
    
    # Analyze chemical knowledge graph
    sample_graph = {
        'nodes': [
            {'id': 'n1', 'name': 'aspirin', 'type': 'drug'},
            {'id': 'n2', 'name': 'pain', 'type': 'symptom'}
        ],
        'edges': [
            {'source': 'n1', 'target': 'n2', 'type': 'treats'}
        ]
    }
    
    graph_results = await unified_interface('graph_analysis', knowledge_graph=sample_graph)
    print("Graph analysis results:", graph_results)
    
    # Generate pattern-based research proposal
    proposal = await unified_interface('proposal', topic='Novel drug combinations for pain management')
    print("Research proposal:", proposal)

# Run the async function
asyncio.run(main())
```

### Integration with Existing Systems
The adapter can be integrated into existing workflows to enhance them with pattern analysis:

```python
import asyncio
from pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter, create_unified_interface

async def enhanced_research_workflow():
    # Initialize the adapter
    adapter = PAMIResearchQuestCurieGlobalChemAdapter()
    unified_interface = create_unified_interface(adapter)
    
    # Use pattern analysis to enhance research
    research_topic = "Investigate novel catalysts for CO2 reduction"
    
    # Conduct enriched research with pattern analysis
    results = await unified_interface('enriched_research', research_question=research_topic)
    
    # Generate pattern-based insights
    insights = results.get('combined_insights', [])
    
    return {
        'original_research': results,
        'pattern_insights': insights,
        'recommendations': results.get('recommendations', [])
    }
```

## Configuration

The adapter supports the following configuration options:

- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `min_support`: Minimum support threshold for pattern mining
- `min_confidence`: Minimum confidence threshold for association rules
- `timeout_seconds`: Timeout for operations
- `research_quest_curie_globalchem_config`: Configuration for the Research Quest - Curie-GlobalChem integration
- `pami_config`: Configuration for the PAMI integration

## Testing

Run the test suite:
```bash
python test_integration.py
```

## Deployment

The integration can be deployed as a container using the provided Dockerfile:

```bash
docker build -t pami-research-quest-curie-globalchem-adapter .
docker run -d --name pami-integration pami-research-quest-curie-globalchem-adapter
```

## Security

This integration follows security best practices:

- No direct access to underlying systems' internal data structures
- Input validation for all queries
- Read-only access to data sources
- Proper error handling to prevent information disclosure
- Isolated execution environments for each system