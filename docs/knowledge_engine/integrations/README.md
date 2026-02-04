# Knowledge Engine Integrations Documentation

Welcome to the comprehensive documentation for Knowledge Engine integrations. This directory contains detailed guides for all 30+ integrations available in the OpenEvolve Knowledge Engine.

## Quick Navigation

### Core Integrations
- [DSPy Integration](./DSPY_INTEGRATION.md) - Program-of-thought prompting system
- [DeepKE Integration](./DEEPKE_INTEGRATION.md) - Knowledge extraction and NER
- [CrewAI Integration](./CREWAI_INTEGRATION.md) - Multi-agent orchestration
- [Ragbits Integration](./RAGBITS_INTEGRATION.md) - Retrieval-augmented generation
- [Agentic Context Engine (ACE)](./ACE_INTEGRATION.md) - Adaptive learning and reflection

### Graph & Knowledge Systems
- [ROMA Integration](./ROMA_INTEGRATION.md) - Meta-agent orchestration system
- [ROMA Entity Knowledge Graph](./ROMA_EKG_INTEGRATION.md) - Entity extraction and graph management
- [Graphiti Integration](./GRAPHITI_INTEGRATION.md) - Temporal knowledge graphs
- [AIKG Integration](./AIKG_INTEGRATION.md) - AI knowledge graph construction
- [Karate Club Integration](./KARATECLUB_INTEGRATION.md) - Graph embeddings and analytics

### Mathematical & Formal Systems
- [Z3 Prover Integration](./Z3_INTEGRATION.md) - SMT solver and theorem proving
- [LeanAIDE Integration](./LEANAIDE_INTEGRATION.md) - Lean 4 proof assistant
- [Unified Math Bridge](./MATH_BRIDGE_INTEGRATION.md) - Cross-system mathematical knowledge

### Research & Evolution
- [Research Quest Integration](./RESEARCH_QUEST_INTEGRATION.md) - Research automation
- [Unified Evolution Integration](./EVOLUTION_INTEGRATION.md) - Evolutionary algorithms
- [LoongFlow Integration](./LOONGFLOW_INTEGRATION.md) - Prompt engineering system

### Specialized Systems
- [AgentJSON Integration](./AGENTJSON_INTEGRATION.md) - Agent protocol standardization
- [MCP Gateway Integration](./MCP_GATEWAY_INTEGRATION.md) - Model Context Protocol gateway
- [OpenEvolve Integration](./OPENEVOLVE_INTEGRATION.md) - Core OpenEvolve platform
- [NeuralKG Integration](./NEURALKG_INTEGRATION.md) - Neural knowledge graph embedding
- [OneKE Integration](./ONEKE_INTEGRATION.md) - Knowledge extraction toolkit
- [GlobalChem Integration](./GLOBALCHEM_INTEGRATION.md) - Chemical knowledge graphs
- [Neuromancer Integration](./NEUROMANCER_INTEGRATION.md) - Neuromorphic computing
- [PAMI Integration](./PAMI_INTEGRATION.md) - Pattern mining algorithms
- [CausalLearn Integration](./CAUSAL_LEARN_INTEGRATION.md) - Causal discovery
- [Lagrange Mapper Integration](./LAGRANGE_MAPPER_INTEGRATION.md) - Topological data analysis
- [PyGraphistry Integration](./PYGRAPHISTRY_INTEGRATION.md) - Graph visualization

### Cross-Integration Pipelines
- [ROMA-DSPy Integration](./ROMA_DSPY_INTEGRATION.md) - ROMA with DSPy reasoning
- [ROMA-DeepKE Integration](./ROMA_DEEPKE_INTEGRATION.md) - ROMA with knowledge extraction
- [ROMA-Ragbits Integration](./ROMA_RAGBITS_INTEGRATION.md) - ROMA with retrieval
- [LeanAIDE-Ragbits Integration](./LEANAIDE_RAGBITS_INTEGRATION.md) - Lean proofs with RAG

### Knowledge Generation
- [KGGen Integration](./KGGEN_INTEGRATION.md) - Knowledge graph generation
- [KGGen Pipeline](./KGGEN_PIPELINE.md) - End-to-end generation pipelines
- [Unified Knowledge Extraction](./UNIFIED_EXTRACTION.md) - Cross-system extraction

## Getting Started

### Installation

Each integration has its own dependencies. Check individual integration guides for specific requirements.

```bash
# Install core Knowledge Engine
pip install -e .

# Install specific integration dependencies
pip install knowledge-engine[dspy]  # For DSPy
pip install knowledge-engine[deepke]  # For DeepKE
pip install knowledge-engine[crewai]  # For CrewAI
# ... etc
```

### Basic Usage

```python
from knowledge_engine.integrations import DSPyIntegration

# Initialize integration
integration = DSPyIntegration(config={
    "model": "gpt-4o",
    "api_key": "your-api-key"
})

# Use the integration
result = integration.chain_of_thought(
    query="Solve this problem step by step",
    context={}
)
```

### Configuration

All integrations support:
- Environment variables for secrets
- YAML/JSON configuration files
- Runtime configuration override
- Default fallback configurations

See [Configuration Guide](./CONFIGURATION.md) for details.

## Integration Categories

### 1. Reasoning & Problem Solving
Integrations that enhance reasoning capabilities:
- **DSPy**: Program-of-thought prompting
- **CrewAI**: Multi-agent collaboration
- **ACE**: Adaptive learning and reflection
- **Research Quest**: Automated research workflows

### 2. Knowledge Extraction
Integrations for extracting structured knowledge:
- **DeepKE**: Entity and relation extraction
- **OneKE**: Comprehensive extraction toolkit
- **AIKG**: Knowledge graph construction
- **KGGen**: Graph generation pipelines

### 3. Knowledge Representation
Integrations for representing and storing knowledge:
- **Graphiti**: Temporal knowledge graphs
- **ROMA EKG**: Entity knowledge graphs
- **NeuralKG**: Neural graph embeddings
- **Karate Club**: Graph analytics

### 4. Retrieval & Search
Integrations for knowledge retrieval:
- **Ragbits**: Document search and retrieval
- **ROMA-Ragbits**: Enhanced retrieval with ROMA

### 5. Mathematical & Formal
Integrations for formal reasoning:
- **Z3 Prover**: SMT solving
- **LeanAIDE**: Proof assistance
- **Unified Math Bridge**: Cross-system reasoning

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Engine Core                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Knowledge    │  │ Embedding    │  │ Semantic     │      │
│  │ Items        │  │ Service      │  │ Search       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
┌───▼────────┐    ┌─────────▼──────────┐    ┌───────▼────────┐
│ Integrations│    │   Integration      │    │   Integration  │
│  Layer      │    │   Adapters         │    │   Orchestrator│
│ ┌────────┐  │    │ ┌────────────────┐ │    │ ┌────────────┐│
│ │ DSPy   │  │    │ │ROMA-DSPy       │ │    │ │Unified     ││
│ │DeepKE  │  │    │ │ROMA-DeepKE     │ │    │ │Evolution   ││
│ │CrewAI  │  │    │ │ROMA-Ragbits    │ │    │ │Pipeline    ││
│ │...     │  │    │ │...             │ │    │ │...         ││
│ └────────┘  │    │ └────────────────┘ │    │ └────────────┘│
└─────────────┘    └────────────────────┘    └────────────────┘
       │                     │                        │
       └─────────────────────┼────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
┌───▼─────────┐   ┌─────────▼─────────┐   ┌────────▼───────┐
│External     │   │External           │   │External        │
│Systems      │   │Systems            │   │Systems         │
│•DSPy        │   │•ROMA              │   │•Z3 Prover      │
│•DeepKE      │   │•Ragbits           │   │•Lean 4         │
│•CrewAI      │   │•ACE               │   │•Graphiti       │
│...          │   │...                │   │...             │
└──────────────┘   └───────────────────┘   └────────────────┘
```

## Best Practices

### 1. Choose the Right Integration
- **For reasoning**: Use DSPy or CrewAI
- **For extraction**: Use DeepKE or OneKE
- **For retrieval**: Use Ragbits
- **For graphs**: Use Graphiti or ROMA EKG
- **For math**: Use Z3 or LeanAIDE

### 2. Configuration Management
- Use environment variables for secrets
- Store configs in version-controlled files
- Validate configs at startup
- Use sensible defaults

### 3. Error Handling
- Always check `success` field in results
- Handle optional dependencies gracefully
- Implement retry logic for transient failures
- Use fallback integrations when possible

### 4. Performance
- Enable caching for expensive operations
- Use batch processing when available
- Configure appropriate timeouts
- Monitor resource usage

### 5. Testing
- Write integration tests
- Mock external dependencies
- Test error conditions
- Validate output schemas

## Contributing

See [Contributing Guide](../CONTRIBUTING.md) for information on:
- Adding new integrations
- Updating documentation
- Reporting issues
- Submitting PRs

## Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See individual integration guides
- **Examples**: Check `examples/` directory

## License

See project LICENSE file for details.

---

**Last Updated**: 2025-02-03
**Version**: 1.0.0
