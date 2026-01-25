# OneKE Enhanced Integration - Quick Start Guide

Get up and running with the enhanced OneKE integration in 5 minutes.

## Installation

```bash
# Install core dependencies
pip install pyyaml numpy

# Install sentence-transformers for semantic similarity (recommended)
pip install sentence-transformers

# Optional: For GPU acceleration
pip install sentence-transformers[all]
```

## Basic Usage

### Option 1: Through Knowledge Engine (Recommended)

```python
from knowledge_engine.engine import KnowledgeEngine

async def main():
    # Initialize engine
    engine = KnowledgeEngine()

    # Extract with quality enhancement
    result = await engine.extract_with_quality(
        text="Python uses async/await for concurrent code execution",
        schema="software_engineering",
        domain="software_engineering"
    )

    # Access results
    print(f"Quality: {result['quality_score']['overall']:.2f}")
    print(f"Entities: {len(result['extraction']['entities'])}")

    for entity in result['extraction']['entities']:
        print(f"  - {entity['text']}: {entity['confidence']:.2f}")

# Run
import asyncio
asyncio.run(main())
```

### Option 2: Direct Bridge Usage

```python
from integrations.oneke.enhanced_bridge import create_enhanced_oneke_bridge

async def main():
    # Create bridge
    bridge = await create_enhanced_oneke_bridge()

    # Extract with enhancement
    result = await bridge.extract_with_enhancement(
        text="Quantum entanglement connects particles across distances",
        schema="physics",
        domain="physics"
    )

    # Access results
    print(f"Quality: {result.quality_score.overall:.2f}")
    print(f"Improvement: {result.quality_improvement:.2%}")

    # Cleanup
    await bridge.shutdown()

import asyncio
asyncio.run(main())
```

### Option 3: Quick Convenience Function

```python
from integrations.oneke.enhanced_bridge import extract_with_quality

async def main():
    # One-line extraction
    result = await extract_with_quality(
        text="All prime numbers greater than 2 are odd",
        schema="mathematics",
        domain="mathematics"
    )

    print(f"Quality: {result['quality_score']['overall']:.2f}")

import asyncio
asyncio.run(main())
```

## With Feedback Loop

```python
from integrations.oneke.enhanced_bridge import create_enhanced_oneke_bridge

async def main():
    bridge = await create_enhanced_oneke_bridge()

    # Extract and learn from feedback
    result = await bridge.extract_and_learn(
        text="Photosynthesis converts CO2 and water into glucose",
        schema="chemistry",
        domain="chemistry",
        feedback={
            'correctness': 0.9,
            'completeness': 0.85,
            'comments': 'Good chemical formula extraction'
        }
    )

    print(f"Learning occurred: {result.metadata['learning_occurred']}")

    # Check repository statistics
    stats = await bridge.get_repository_statistics()
    print(f"Total cases stored: {stats['total_cases']}")

    await bridge.shutdown()

import asyncio
asyncio.run(main())
```

## Batch Processing

```python
from knowledge_engine.engine import KnowledgeEngine

async def main():
    engine = KnowledgeEngine()

    texts = [
        "Python async/await syntax",
        "Quantum entanglement phenomena",
        "Prime number theorem"
    ]

    # Batch extract
    results = await engine.batch_extract_with_quality(
        texts=texts,
        schema="general",
        domain="general"
    )

    for i, result in enumerate(results):
        print(f"Text {i+1}: Quality={result['quality_score']['overall']:.2f}")

import asyncio
asyncio.run(main())
```

## Common Use Cases

### Extract Software Engineering Knowledge

```python
result = await engine.extract_with_quality(
    text="""
    Python 3.5 introduced type hints and async/await syntax.
    The asyncio library provides tools for concurrent programming.
    Type hints improve IDE support and code documentation.
    """,
    schema="software_engineering",
    domain="software_engineering"
)
```

### Extract Physics Concepts

```python
result = await engine.extract_with_quality(
    text="""
    The Schrödinger equation describes quantum state evolution.
    Wave functions represent probability amplitudes.
    Heisenberg uncertainty principle limits measurement precision.
    """,
    schema="physics",
    domain="physics"
)
```

### Extract Mathematical Knowledge

```python
result = await engine.extract_with_quality(
    text="""
    Theorem: All primes greater than 2 are odd.
    Proof by contradiction assumes an even prime exists.
    This leads to a contradiction with primality.
    """,
    schema="mathematics",
    domain="mathematics"
)
```

## Configuration

### Quick Configuration

Create `config_enhanced.yaml`:

```yaml
reflection:
  iterations: 2      # Fewer = faster
  num_samples: 2     # Fewer = faster

quality_enhancement:
  min_quality_threshold: 0.7  # Accept good extractions
  strategies:
    - validation          # Fastest
    - reflection          # Good balance
    # - cases            # Requires repository
    # - consistency      # Slowest

case_repository:
  storage_path: "data/oneke_cases.json"
  auto_save: true
  save_interval: 50
```

### Performance Tuning

**For Speed:**
```yaml
reflection:
  iterations: 1
  num_samples: 2

quality_enhancement:
  strategies:
    - validation
```

**For Quality:**
```yaml
reflection:
  iterations: 3
  num_samples: 5

quality_enhancement:
  strategies:
    - reflection
    - validation
    - cases
    - consistency
```

**For Balance:**
```yaml
reflection:
  iterations: 2
  num_samples: 3

quality_enhancement:
  strategies:
    - reflection
    - validation
    - cases
```

## Quality Metrics

Understanding the quality scores:

```python
result = await engine.extract_with_quality(...)

# Quality breakdown
quality = result['quality_score']
print(f"Completeness: {quality['completeness']:.2f}")  # All entities extracted?
print(f"Accuracy: {quality['accuracy']:.2f}")          # Schema valid?
print(f"Consistency: {quality['consistency']:.2f}")    # No contradictions?
print(f"Confidence: {quality['confidence']:.2f}")      # Average confidence
print(f"Overall: {quality['overall']:.2f}")            # Weighted average
```

## Troubleshooting

### Low Quality Scores

**Problem**: Quality < 0.7

**Solutions**:
1. Enable more enhancement strategies
2. Increase reflection iterations
3. Provide more context in text
4. Use domain-specific schema

### Slow Performance

**Problem**: Takes > 10 seconds

**Solutions**:
1. Reduce reflection iterations to 1-2
2. Disable consistency checking
3. Use faster embedding model
4. Batch process multiple texts

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solutions**:
1. Install dependencies: `pip install sentence-transformers`
2. Check Python path
3. Use fallback mode (no sentence-transformers)

### Empty Repository

**Problem**: No similar cases found

**Solutions**:
1. Extract more texts to build repository
2. Lower similarity threshold
3. Use different domain
4. Provide feedback loop

## Next Steps

1. **Run Examples**: See `example_enhanced.py`
2. **Read Documentation**: See `ENHANCED_README.md`
3. **Run Tests**: `pytest integrations/oneke/test_enhanced.py -v`
4. **Customize Config**: Edit `config_enhanced.yaml`
5. **Integrate**: Add to your workflow

## API Reference

### KnowledgeEngine Methods

```python
# Initialize OneKE bridge
await engine.initialize_oneke_bridge()

# Extract with quality
result = await engine.extract_with_quality(text, schema, domain)

# Extract and learn
result = await engine.extract_and_learn(text, schema, domain, feedback)

# Batch process
results = await engine.batch_extract_with_quality(texts, schema, domain)

# Repository statistics
stats = await engine.get_oneke_repository_statistics()

# Export/import
await engine.export_oneke_repository(path)
await engine.import_oneke_repository(path)
```

### Result Structure

```python
{
    'extraction': {
        'entities': [...],
        'relations': [...],
        'events': [...],
        'triples': [...]
    },
    'quality_score': {
        'completeness': 0.9,
        'accuracy': 0.85,
        'consistency': 0.95,
        'confidence': 0.88,
        'overall': 0.89
    },
    'original_quality': {...},
    'quality_improvement': 0.15,
    'strategies_applied': ['reflection', 'validation', 'cases'],
    'metadata': {...}
}
```

## Tips & Best Practices

### 1. Start Simple
```python
# Start with validation only
result = await engine.extract_with_quality(
    text=text,
    schema=schema,
    domain=domain,
    enable_enhancement=False  # Base extraction
)
```

### 2. Add Strategies Gradually
```python
# Add validation
result = await engine.extract_with_quality(
    text=text,
    schema=schema,
    domain=domain,
    enable_enhancement=True
)
```

### 3. Provide Feedback
```python
# Use feedback for continuous improvement
result = await engine.extract_and_learn(
    text=text,
    schema=schema,
    feedback={'correctness': 0.9, 'completeness': 0.85}
)
```

### 4. Monitor Quality
```python
# Track quality over time
quality = result['quality_score']['overall']
if quality < 0.7:
    print("Low quality - consider manual review")
```

### 5. Use Domain Optimization
```python
# Domain-specific extraction
result = await engine.extract_with_quality(
    text=text,
    schema='physics',
    domain='physics'  # Uses physics-specific settings
)
```

## Support

- Documentation: `ENHANCED_README.md`
- Examples: `example_enhanced.py`
- Tests: `test_enhanced.py`
- Implementation: `PHASE4_IMPLEMENTATION_SUMMARY.md`

## Summary

**Quick Start Checklist:**
- [ ] Install dependencies
- [ ] Create config (optional)
- [ ] Initialize bridge/engine
- [ ] Extract with enhancement
- [ ] Check quality scores
- [ ] Provide feedback (optional)
- [ ] Monitor repository growth

**Time to First Extraction**: < 5 minutes
**Expected Quality Improvement**: 10-25%
**Learning Curve**: Low to Medium

You're ready to extract high-quality knowledge with OneKE! 🚀
