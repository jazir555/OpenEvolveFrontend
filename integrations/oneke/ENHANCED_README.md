# OneKE Enhanced Integration - Phase 4

## Overview

This module provides advanced knowledge extraction capabilities through reflection, quality enhancement, and case-based learning. It extends the base OneKE integration with intelligent self-improvement mechanisms.

## Features

### 1. Reflection Agent
- **Self-consistency checking**: Generate multiple extraction samples and verify agreement
- **Case-based retrieval**: Learn from similar past extractions
- **Quality scoring**: Comprehensive quality metrics (completeness, accuracy, consistency, confidence)
- **Iterative refinement**: Automatically improve extraction quality through multiple reflection cycles

### 2. Quality Enhancement System
- **Multi-strategy enhancement**: Apply reflection, validation, case-learning, and consistency checking
- **Automatic quality improvement**: Iteratively refine extractions
- **Validation framework**: Schema validation with automatic error fixing
- **Quality metrics**: Detailed quality tracking and reporting

### 3. Case Repository
- **Semantic similarity search**: Retrieve similar cases using embeddings
- **Quality tracking**: Track case quality over time
- **Persistent storage**: Automatic saving and loading of cases
- **Export/import**: Share cases between systems

### 4. Enhanced Bridge
- **Unified interface**: Simple API for enhanced extraction
- **Feedback loop**: Learn from human feedback
- **Batch processing**: Extract from multiple texts efficiently
- **Domain optimization**: Domain-specific enhancement strategies

## Installation

```bash
# Install dependencies
pip install sentence-transformers numpy pyyaml

# Optional: For better embeddings
pip install sentence-transformers[all]
```

## Quick Start

### Basic Extraction with Enhancement

```python
from integrations.oneke.enhanced_bridge import create_enhanced_oneke_bridge

async def main():
    # Create bridge
    bridge = await create_enhanced_oneke_bridge()

    # Extract with enhancement
    result = await bridge.extract_with_enhancement(
        text="Python uses async/await for concurrent code...",
        schema="software_engineering",
        domain="software_engineering",
        enable_reflection=True,
        enable_cases=True,
        enable_validation=True
    )

    print(f"Quality: {result.quality_score.overall:.2f}")
    print(f"Improvement: {result.quality_improvement:.2%}")

    await bridge.shutdown()
```

### Extraction with Feedback

```python
# Extract and learn from human feedback
result = await bridge.extract_and_learn(
    text="Quantum entanglement is a phenomenon where...",
    schema="physics",
    domain="physics",
    feedback={
        'correctness': 0.9,
        'completeness': 0.85,
        'comments': 'Good extraction'
    }
)

print(f"Learning occurred: {result.metadata.get('learning_occurred')}")
```

### Batch Processing

```python
texts = [
    "Text 1 about software...",
    "Text 2 about physics...",
    "Text 3 about math..."
]

results = await bridge.batch_extract_with_enhancement(
    texts=texts,
    schema="general",
    domain="general"
)

for i, result in enumerate(results):
    print(f"Text {i}: Quality={result.quality_score.overall:.2f}")
```

## Architecture

### Components

```
EnhancedOneKEBridge
├── OneKEAdapter (base extraction)
├── OneKECaseRepository (case storage & retrieval)
├── OneKEReflectionAgent (quality improvement)
└── OneKEQualityEnhancer (multi-strategy enhancement)
```

### Data Flow

```
Input Text
    ↓
Initial Extraction (OneKE)
    ↓
Quality Scoring
    ↓
Enhancement Strategies
    ├── Reflection → Self-consistency checking
    ├── Validation → Schema validation
    ├── Cases → Case-based learning
    └── Consistency → Multi-sample agreement
    ↓
Quality Re-scoring
    ↓
Store High-Quality Case (if quality >= 0.7)
    ↓
Enhanced Extraction Result
```

## Configuration

Create `config_enhanced.yaml`:

```yaml
reflection:
  enabled: true
  iterations: 3
  num_samples: 3
  temperature: 0.3

quality_enhancement:
  strategies:
    - reflection
    - validation
    - cases
    - consistency
  min_quality_threshold: 0.7

case_repository:
  storage_path: "data/oneke_cases.json"
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  auto_save: true
  save_interval: 100
```

## API Reference

### EnhancedOneKEBridge

#### `extract_with_enhancement()`
Extract knowledge with full enhancement pipeline.

**Parameters:**
- `text` (str): Input text
- `schema` (str): Target schema name
- `domain` (str): Domain label
- `enable_reflection` (bool): Enable reflection strategy
- `enable_cases` (bool): Enable case-based learning
- `enable_validation` (bool): Enable validation strategy
- `enable_consistency` (bool): Enable consistency checking

**Returns:** `EnhancedResult`

#### `extract_and_learn()`
Extract and learn from human feedback.

**Parameters:**
- `text` (str): Input text
- `schema` (str): Target schema
- `domain` (str): Domain label
- `feedback` (dict): Human feedback
  - `correctness` (float): Correctness score (0-1)
  - `completeness` (float): Completeness score (0-1)
  - `comments` (str): Optional comments

**Returns:** `EnhancedResult`

#### `batch_extract_with_enhancement()`
Extract from multiple texts.

**Parameters:**
- `texts` (List[str]): Input texts
- `schema` (str): Target schema
- `domain` (str): Domain label
- `enable_enhancement` (bool): Enable enhancement

**Returns:** `List[EnhancedResult]`

### Quality Scoring

Quality scores range from 0-1 and include:

- **Completeness**: Fraction of required entities extracted
- **Accuracy**: Fraction of entities matching schema
- **Consistency**: Absence of contradictions
- **Confidence**: Average entity confidence
- **Overall**: Weighted average of all metrics

### Enhancement Strategies

#### Reflection
Generates multiple extraction samples and identifies inconsistencies:
- Checks self-consistency across samples
- Identifies potential errors
- Refines extraction based on consensus

#### Validation
Validates extraction against schema:
- Checks required fields
- Validates entity types
- Fixes validation errors automatically

#### Cases
Retrieves and learns from similar cases:
- Semantic similarity search
- Adds missing entities from high-quality cases
- Boosts confidence based on case patterns

#### Consistency
Checks agreement across multiple samples:
- Generates multiple extraction samples
- Computes consensus extraction
- Measures agreement ratio

## Case Repository

### Managing Cases

```python
# Retrieve similar cases
similar = await bridge.case_repository.retrieve_similar_cases(
    query={'input_text': query_text, 'domain': 'physics'},
    top_k=5,
    min_similarity=0.7
)

# Get high-quality cases
good_cases = await bridge.case_repository.get_good_cases(
    domain='physics',
    min_quality=0.8,
    limit=10
)

# Get repository statistics
stats = await bridge.get_repository_statistics()
print(f"Total cases: {stats['total_cases']}")
print(f"Average quality: {stats['average_quality']:.2f}")
```

### Export/Import

```python
# Export repository
await bridge.export_repository("data/backup_cases.json")

# Import repository
await bridge.import_repository("data/backup_cases.json")
```

## Domain-Specific Optimization

Configure domain-specific settings in `config_enhanced.yaml`:

```yaml
domains:
  physics:
    strategies:
      - reflection
      - cases
      - consistency
    min_quality_threshold: 0.75
    case_similarity_threshold: 0.75

  chemistry:
    strategies:
      - reflection
      - validation
      - cases
    min_quality_threshold: 0.75
```

## Performance Considerations

### Embedding Model
- Default: `sentence-transformers/all-mpnet-base-v2`
- Faster option: `sentence-transformers/all-MiniLM-L6-v2`
- Best quality: `sentence-transformers/all-mpnet-base-v2`

### Reflection Iterations
- More iterations = better quality, slower processing
- Recommended: 2-3 iterations
- Set in config: `reflection.iterations`

### Consistency Samples
- More samples = better consistency checking, slower
- Recommended: 3 samples
- Set in config: `reflection.num_samples`

## Testing

Run the test suite:

```bash
# Run all tests
pytest integrations/oneke/test_enhanced.py -v

# Run specific test class
pytest integrations/oneke/test_enhanced.py::TestCaseRepository -v

# Run with coverage
pytest integrations/oneke/test_enhanced.py --cov=integrations.oneke -v
```

## Examples

See `example_enhanced.py` for comprehensive examples:

```bash
python integrations/oneke/example_enhanced.py
```

Examples include:
1. Basic extraction with enhancement
2. Extraction with feedback and learning
3. Batch processing
4. Retrieving similar cases
5. Repository management
6. Quality metrics
7. Domain-specific extraction
8. Quick extraction

## Integration with Knowledge Engine

```python
from knowledge_engine.engine import KnowledgeEngine
from integrations.oneke.enhanced_bridge import EnhancedOneKEBridge

class EnhancedKnowledgeEngine(KnowledgeEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oneke_bridge = EnhancedOneKEBridge()

    async def initialize(self):
        await super().initialize()
        await self.oneke_bridge.initialize()

    async def extract_with_quality(self, text, schema, domain):
        return await self.oneke_bridge.extract_with_enhancement(
            text=text,
            schema=schema,
            domain=domain
        )
```

## Troubleshooting

### Low Quality Scores
- Increase `reflection.iterations`
- Enable more enhancement strategies
- Check schema definition
- Provide more training cases

### Slow Performance
- Reduce `reflection.num_samples`
- Use faster embedding model
- Disable unnecessary strategies
- Use batch processing

### Memory Issues
- Reduce `case_repository.save_interval`
- Limit repository size with `learning.case_limit`
- Export and clear old cases periodically

## Best Practices

1. **Start Simple**: Begin with validation and reflection, add other strategies gradually
2. **Provide Feedback**: Use `extract_and_learn()` with human feedback for best results
3. **Monitor Quality**: Track quality metrics to identify issues early
4. **Domain Optimization**: Configure domain-specific settings for better results
5. **Regular Backups**: Export repository regularly to avoid data loss

## Contributing

When contributing to the enhanced integration:

1. Add tests for new features in `test_enhanced.py`
2. Update examples in `example_enhanced.py`
3. Document configuration options
4. Follow existing code style

## License

This module is part of the OpenEvolve project and follows the same license.

## References

- OneKE: [OneKE Documentation](https://github.com/INKLab/OneKE)
- Sentence Transformers: [Documentation](https://www.sbert.net/)
- Case-Based Reasoning: [Overview](https://en.wikipedia.org/wiki/Case-based_reasoning)
