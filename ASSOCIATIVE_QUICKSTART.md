# Associative Recomposition Quick Start

## 30-Second Overview

```python
from associative_recomposition import AssociativeRecomposer

# Setup
recomposer = AssociativeRecomposer(use_agentjson=True)

# Run
assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions={
        'sol_1': {'solution_content': 'def auth(): ...'},
        'sol_2': {'solution_content': 'class User: ...'}
    },
    conflicts=[],
    problem_statement="Build user system",
    llm_call_fn=lambda p: your_llm_api.call(p)
)

# Result
if assembled:
    print(f"✓ Domain: {metadata['classification']['domain']}")
    print(f"✓ Correct: {metadata['judgment']['is_correct']}")
    print(assembled)
```

## What It Does

1. **LLM Classifies** - Identifies problem domain (not hardcoded)
2. **AgentJSON Parses** - Robust JSON parsing with repair
3. **Algorithmic Verification** - Ensures content preserved
4. **LLM Judges** - Evaluates correctness

## Why Use This?

| Problem | Solution |
|---------|----------|
| Hardcoded domains don't scale | LLM classifies domain itself |
| LLMs output bad JSON | AgentJSON repairs it |
| Content gets mutated | Algorithmic verification catches it |
| Can't judge correctness | LLM evaluates domain-specific quality |

## Installation

No installation needed! Uses:
- Standard library (json, hashlib, logging)
- AgentJSON (already in `agentjson/` folder)
- Your LLM API (OpenAI, Anthropic, etc.)

## Basic Example

```python
import sys
sys.path.insert(0, '.')

from associative_recomposition import AssociativeRecomposer

# Your LLM API call
def call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key="your-key")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Sub-solutions (from decomposition)
solutions = {
    'sol_1': {
        'description': 'Authentication',
        'solution_content': '''
def authenticate(username, password):
    """Authenticate user"""
    return verify(username, password)
''',
        'confidence_score': 0.95
    },
    'sol_2': {
        'description': 'User Profile',
        'solution_content': '''
class UserProfile:
    """User profile model"""
    def __init__(self, user_id):
        self.user_id = user_id
        self.email = None
''',
        'confidence_score': 0.90
    }
}

# Recompose
recomposer = AssociativeRecomposer()
assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=solutions,
    conflicts=[],
    problem_statement="Build authentication and profile system",
    llm_call_fn=call_llm
)

# Check result
if assembled:
    print("✓ Success!")
    print(f"Domain: {metadata['classification']['domain']}")
    print(f"Quality: {metadata['judgment']['quality_score']:.2f}")
    print("\nAssembled:")
    print(assembled)
else:
    print("✗ Failed")
    print(f"Errors: {metadata}")
```

## Understanding the Output

### Classification (LLM-provided)

```python
{
    'domain': 'software_development',      # LLM's classification
    'solution_type': 'code',               # LLM's classification
    'field': 'web authentication',         # LLM's classification
    'complexity': 'medium',                # LLM's classification
    'confidence': 0.92                      # LLM's confidence
}
```

### Judgment (LLM evaluation)

```python
{
    'is_correct': true,                    # Is solution correct?
    'completeness_score': 0.95,           # How complete?
    'quality_score': 0.90,                # How good?
    'verdict': 'good',                     # Overall verdict
    'missing_elements': [],                # What's missing
    'issues': ['Could use error handling'], # What to fix
    'strengths': ['All components present'] # What's good
}
```

### Verification (Algorithmic)

```python
{
    'sol_1': (True, "Content preserved exactly"),
    'sol_2': (True, "Code components verified")
}
```

## Common Patterns

### Pattern 1: Code Assembly

```python
solutions = {
    'sol_1': {'solution_content': 'def func1(): ...'},
    'sol_2': {'solution_content': 'class Class1: ...'},
}

assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=solutions,
    problem_statement="Build Python module",
    llm_call_fn=call_llm
)

# LLM will classify as:
# - domain: software_development
# - solution_type: code
# - Will preserve code exactly
```

### Pattern 2: Documentation

```python
solutions = {
    'sol_1': {'solution_content': '# Authentication Guide\n\n...'},
    'sol_2': {'solution_content': '# API Reference\n\n...'},
}

assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=solutions,
    problem_statement="Create API documentation",
    llm_call_fn=call_llm
)

# LLM will classify as:
# - domain: software_development
# - solution_type: documentation
# - Will merge prose sections smoothly
```

### Pattern 3: Multi-Domain

```python
solutions = {
    'sol_1': {'solution_content': 'Financial model code...'},
    'sol_2': {'solution_content': 'Legal compliance text...'},
}

# LLM will detect mixed domain and handle appropriately
assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=solutions,
    problem_statement="Build compliant financial system",
    llm_call_fn=call_llm
)

# LLM might classify as:
# - domain: finance
# - solution_type: code (with legal requirements)
```

## Error Handling

### Handle Parse Failures

```python
assembled, metadata = recomposer.recompose_with_verification(...)

if not assembled:
    for attempt in metadata.get('attempts', []):
        if 'parse_errors' in attempt:
            print(f"Parse errors: {attempt['parse_errors']}")
```

### Handle Verification Failures

```python
verification = metadata.get('verification_results', {})
for sub_id, (preserved, details) in verification.items():
    if not preserved:
        print(f"Missing: {sub_id} - {details}")
```

### Handle Judgment Failures

```python
judgment = metadata.get('judgment', {})
if not judgment.get('is_correct', False):
    print(f"Issues: {judgment.get('issues', [])}")
    print(f"Missing: {judgment.get('missing_elements', [])}")
```

## Configuration

### Enable AgentJSON (Recommended)

```python
recomposer = AssociativeRecomposer(
    use_agentjson=True,  # Robust JSON parsing
    max_retries=3        # Retry attempts
)
```

### Disable AgentJSON (Fallback)

```python
recomposer = AssociativeRecomposer(
    use_agentjson=False,  # Use standard json.loads
    max_retries=1
)
```

### Custom Ground Truth Store

```python
from ground_truth_store import GroundTruthStore

store = GroundTruthStore(
    storage_path="my_ground_truth.json",
    backend="file"
)

recomposer = AssociativeRecomposer(
    ground_truth_store=store
)
```

## Tips

1. **Always check classification** - Helps understand what LLM thinks the problem is
2. **Review judgment reasoning** - Explains WHY solution is correct/incorrect
3. **Check verification results** - Ensures nothing was lost
4. **Use AgentJSON** - More robust to LLM output errors
5. **Provide clear problem statements** - Helps LLM classify correctly

## Troubleshooting

### "AgentJSON not available"

**Solution:** System will fall back to standard JSON automatically.

### "Parse failed"

**Solution:** Check LLM response format. Ensure prompt is clear about JSON output.

### "Verification failed"

**Solution:** Some content was lost. Check which sub-solutions and retry.

### "Judgment: incorrect"

**Solution:** Review judgment reasoning to understand why and fix issues.

## Next Steps

1. Read full guide: `ASSOCIATIVE_RECOMPOSITION_GUIDE.md`
2. See example: `examples/associative_recomposition_example.py`
3. Integrate into your workflow

## Key Files

- `associative_recomposition.py` - Main system
- `ground_truth_store.py` - Verification layer
- `ASSOCIATIVE_RECOMPOSITION_GUIDE.md` - Full documentation
- `examples/associative_recomposition_example.py` - Working example
