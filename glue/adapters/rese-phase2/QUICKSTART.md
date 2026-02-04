# RESE Phase II: Quick Start Guide

Get started with RESE Phase II: Isomorphic Mapping in 5 minutes.

## Installation

```bash
cd glue/adapters/rese-phase2

# Verify installation
python verify_install.py
```

## Basic Usage

### 1. Execute Phase II

```python
from phase2_adapter import Phase2Adapter

# Create adapter
adapter = Phase2Adapter()

# Execute Phase II
result = adapter.execute_phase2({
    "source_domain": "physics",
    "problem_description": "Energy conservation in closed system",
    "target_domains": ["biology", "economics"],
    "constraints": ["energy is conserved"]
})

# View results
print(f"Mappings: {result['summary']['mapping_count']}")
print(f"Best I_mech: {result['summary']['best_imech_score']:.2f}")
```

### 2. Check for Valid Isomorphisms

```python
if result['best_mapping']:
    mapping = result['best_mapping']
    if mapping['i_mech_score'] > 0.7:
        print(f"Valid isomorphism to: {mapping['target_domain']}")
        print(f"Confidence: {mapping['confidence']:.2f}")
```

### 3. Access Inverted Constraints

```python
for inv in result['inverted_constraints']:
    print(f"Original: {inv['original']}")
    print(f"Inverted: {inv['inverted']}")
    print(f"Reduction: {inv['reduction_factor']}x")
```

### 4. View Cross-Domain Patterns

```python
for pattern in result['cross_domain_patterns']:
    print(f"Pattern: {pattern['name']}")
    print(f"Domains: {', '.join(pattern['domains'])}")
    print(f"Confidence: {pattern['confidence']:.2f}")
```

## Configuration

Set environment variables:

```bash
export PHASE2_IMECH_THRESHOLD=0.7
export PHASE2_TIMEOUT_MS=20000
export PHASE2_MAX_MAPPINGS=50
```

## CLI Usage

```bash
# Show configuration
python src/phase2_adapter.py --config

# Execute Phase II
python src/phase2_adapter.py \
    --source physics \
    --problem "Energy conservation" \
    --targets biology economics

# Check health
python src/phase2_adapter.py --health
```

## Examples

### Example 1: Cross-Domain Transfer

```python
# Find isomorphic mappings from physics to biology
result = adapter.execute_phase2({
    "source_domain": "physics",
    "problem_description": "Wave propagation with damping",
    "target_domains": ["biology"]
})

if result['best_mapping'] and result['best_mapping']['i_mech_score'] > 0.7:
    print("Valid transfer found!")
    # Use this mapping for knowledge transfer
```

### Example 2: Constraint Inversion

```python
# Invert constraints to define solution space
result = adapter.execute_phase2({
    "source_domain": "computer_science",
    "problem_description": "Optimization algorithm",
    "constraints": [
        "algorithm must be O(n log n)",
        "memory usage must be minimal"
    ]
})

# Use inverted constraints
for inv in result['inverted_constraints']:
    # This defines the solution space
    print(inv['inverted'])
```

### Example 3: Pattern Recognition

```python
# Identify patterns across domains
result = adapter.execute_phase2({
    "source_domain": "biology",
    "problem_description": "Population dynamics"
})

# Access recurring patterns
for pattern in result['cross_domain_patterns']:
    if pattern['confidence'] > 0.6:
        print(f"High-confidence pattern: {pattern['name']}")
```

## Troubleshooting

**Low I_mech scores?**
- Increase domain knowledge in KB
- Lower threshold: `PHASE2_IMECH_THRESHOLD=0.6`

**No mappings found?**
- Check target domains
- Increase search depth: `PHASE2_SEARCH_DEPTH=10`

**Timeout errors?**
- Increase timeout: `PHASE2_TIMEOUT_MS=30000`
- Reduce targets: `PHASE2_MAX_TARGET_DOMAINS=5`

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [ADR.md](ADR.md) for architecture decisions
- Run tests: `pytest tests/test_phase2.py -v`
- View examples in [tests/test_integration.py](tests/test_integration.py)

## Support

- Documentation: [README.md](README.md)
- Issues: Check [ADR.md](ADR.md) for design decisions
- Tests: [tests/test_phase2.py](tests/test_phase2.py)
