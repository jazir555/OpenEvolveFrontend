# OpenEvolve Knowledge Engine - Quick Reference

## 30-Second Overview

```python
from knowledge_engine import UnifiedKGIntegrationHub, UnifiedKGConfig

# 1. Configure
config = UnifiedKGConfig(
    enable_deepke=True,      # Extract from text
    enable_z3=True,          # Verify logic
    enable_neuralkg=True,    # Semantic search
    enable_unified_knowledge_graph=True
)

# 2. Initialize
hub = UnifiedKGIntegrationHub(config)
await hub.initialize()

# 3. Extract knowledge
text = "Alice works at OpenAI and knows Bob"
result = await hub.extract_knowledge(text)

# 4. Query
answer = hub.query("Where does Alice work?")
# → "OpenAI" (with evidence chain)
```

---

## Component Map

```
INPUT                    PROCESSING              OUTPUT
─────────────────────────────────────────────────────────
📄 Text                  → DeepKE/OneKE         → 📊 Structured triples
🌐 Web pages             → Browser Agent        → 🔗 Entity relationships
🗄️  Databases            → Connectors            → 📈 Analytics
📧 Documents             → KG-Gen               → 💡 Insights
─────────────────────────────────────────────────────────
                         ↓
              ┌───────────────────┐
              │  UnifiedKnowledge │  ← Core storage
              │     Graph         │
              └───────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    REASONING LAYER                      │
│  Z3 (logic) • LeanAide (math) • Causal-Learn (causal)  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    LEARNING LAYER                       │
│  NeuralKG (embeddings) • Pattern Mining • Adaptation   │
└─────────────────────────────────────────────────────────┘
                         ↓
📋 Answers • 🎯 Recommendations • ⚠️ Alerts • 📊 Reports
```

---

## Common Tasks

### Extract Knowledge from Text
```python
text = "OpenAI released GPT-4 in March 2023"
result = await hub.extract_knowledge(text)
# Returns: [(OpenAI, released, GPT-4), (GPT-4, released_in, March 2023)]
```

### Verify a Fact
```python
is_valid = await hub.verify_with_z3({
    "premises": ["All humans are mortal", "Socrates is human"],
    "conclusion": "Socrates is mortal"
})
# Returns: {valid: True, explanation: "Modus ponens"}
```

### Find Similar Entities
```python
similar = await hub.analyze_graph(analysis_type="embeddings")
# Returns entities semantically similar to query
```

### Track Changes Over Time
```python
past = await hub.query_temporal(
    "Who was CEO?",
    timestamp="2020-01-01"
)
# Returns answer valid at that time
```

### Mine Patterns
```python
patterns = await hub.mine_patterns(min_support=0.1)
# Returns: "Users who buy X often buy Y within 30 days"
```

---

## When to Use What

| Goal | Use This | Example |
|------|----------|---------|
| Extract from documents | `DeepKE`, `OneKE` | Research papers |
| Verify calculations | `Z3`, `LeanAide` | Financial models |
| Semantic search | `NeuralKG` | Find similar papers |
| Track history | `Graphiti` | Audit trails |
| Find causes | `Causal-Learn` | Why did sales drop? |
| Pattern discovery | `PAMI` | Customer behavior |
| Web research | `Browser Agent` | Competitive analysis |
| Full orchestration | `UnifiedKGIntegrationHub` | All of the above |

---

## Core Classes

### UnifiedKnowledgeGraph
```python
from graph.unified_kg import UnifiedKnowledgeGraph, UnifiedTriple

ukg = UnifiedKnowledgeGraph(backend='memory')
ukg.add_triple(UnifiedTriple('Alice', 'knows', 'Bob', confidence=0.95))
results = ukg.get_triples(subject='Alice')
paths = ukg.find_paths('Alice', 'Charlie', max_length=3)
```

### KnowledgeGraphModels
```python
from graph.kg_models import KnowledgeGraphModels, KnowledgeStatement

kgm = KnowledgeGraphModels()
stmt = kgm.create_statement('DrugX', 'treats', 'DiseaseY', confidence=0.88)
profile = kgm.create_entity_profile('Alice', types=['Person', 'Researcher'])
```

---

## Architecture in 5 Lines

1. **Storage Layer**: `UnifiedKnowledgeGraph` stores triples (S-P-O)
2. **Model Layer**: `KnowledgeGraphModels` adds schema and provenance
3. **Extraction Layer**: DeepKE/OneKE/KG-Gen extract from text
4. **Reasoning Layer**: Z3/LeanAide verify correctness
5. **Learning Layer**: NeuralKG/Adaptation improve over time

---

## Key Concepts

### Triple
```
Subject ──[Predicate]──▶ Object
Alice   ──[knows]─────▶ Bob
```

### Entity Profile
Rich representation of an entity including:
- Properties (name, type, attributes)
- Relationships (connections to other entities)
- Provenance (where info came from)
- Confidence (how certain we are)

### Provenance Chain
```
Conclusion: "DrugX treats DiseaseY"
Evidence:
  1. Paper in Nature (confidence: 0.9)
  2. Clinical trial data (confidence: 0.85)
  3. FDA approval (confidence: 1.0)
Overall confidence: 0.95
```

---

## Benefits Summary

| Before | After |
|--------|-------|
| Information scattered | Unified knowledge graph |
| Manual verification | Automated validation |
| Static facts | Temporal tracking |
| Keyword search | Semantic understanding |
| Fixed rules | Continuous learning |
| Black box | Explainable reasoning |

---

## Next Steps

1. **Quick Start**: Run `examples/basic_workflow.py`
2. **Deep Dive**: Read `SYSTEM_ARCHITECTURE_AND_USAGE_GUIDE.md`
3. **API Reference**: See class docstrings
4. **Examples**: Check `examples/` directory
5. **Tests**: Review `tests/` for usage patterns
