# Enhanced Red Flagging System - Quick Start Guide

Get started with the enhanced red flagging system in 5 minutes.

## Installation

Ensure you have the required dependencies:

```bash
# Core dependencies
pip install lmql guardrails-ai

# Optional: For MDAP integration
pip install reliability-plugin
```

## Basic Usage (30 seconds)

```python
from reliability.enhanced_redflagger import create_enhanced_redflagger

# Create flagger
flagger = create_enhanced_redflagger()

# Validate output
is_flagged, flags = flagger.check_for_red_flags(
    raw_text="Your response text here",
    candidate={"response": "Your response text here"}
)

if is_flagged:
    print(f"🚩 Found {len(flags)} red flags!")
    for flag in flags:
        print(f"  - {flag}")
else:
    print("✅ No red flags detected")
```

## Run Examples

```bash
# Run all examples
cd reliability/examples
python enhanced_redflagging_examples.py

# Run specific example
python -c "from enhanced_redflagging_examples import example_1_basic_redflagging; example_1_basic_redflagging()"
```

## Configuration (5 minutes)

```python
from reliability.enhanced_redflagger import EnhancedRedFlagger, EnhancedRedFlagRules

# Create custom rules
rules = EnhancedRedFlagRules(
    max_tokens=500,
    confidence_threshold=0.7,
    enable_lmql_constraints=True,
    guardrails_validators=[
        "toxic_language",
        "pii_filter",
        "secrets_detection"
    ],
    required_keywords=["solution"],
    forbidden_keywords=["error", "failed"]
)

# Use custom rules
flagger = EnhancedRedFlagger(rules=rules)
```

## Key Features

1. **Pre-Generation Prevention**: Use LMQL to prevent bad content
   ```python
   constraints = flagger.get_lmql_constraints()
   ```

2. **Post-Generation Validation**: Validate with Guardrails
   ```python
   is_flagged, flags = flagger.check_for_red_flags(text, candidate)
   ```

3. **Severity Levels**: Handle flags by severity
   ```python
   critical = [f for f in flags if f.severity == RedFlagSeverity.CRITICAL]
   ```

4. **Statistics**: Monitor performance
   ```python
   stats = flagger.get_statistics()
   print(f"Flag rate: {stats['flag_rate']:.2%}")
   ```

## Next Steps

- Read the full [README.md](./README.md) for detailed documentation
- Explore [enhanced_redflagging_examples.py](./enhanced_redflagging_examples.py) for complete examples
- Check the API reference in the source code

## Common Patterns

### Pattern 1: Strict Security
```python
rules = EnhancedRedFlagRules(
    confidence_threshold=0.9,
    guardrails_validators=["secrets_detection", "malicious_patterns"],
    enable_lmql_constraints=True
)
```

### Pattern 2: Content Quality
```python
rules = EnhancedRedFlagRules(
    required_keywords=["answer", "explanation"],
    required_format="json",
    confidence_threshold=0.7
)
```

### Pattern 3: Cost Optimization
```python
rules = EnhancedRedFlagRules(
    enable_lmql_constraints=True,  # Prevent waste
    lmql_max_retries=1,
    guardrails_validators=[]  # Minimal post-generation checks
)
```

## Support

Need help? Check:
1. [README.md](./README.md) - Full documentation
2. [enhanced_redflagging_examples.py](./enhanced_redflagging_examples.py) - Working examples
3. Troubleshooting section in README.md

---

**Ready to go!** Start with `python enhanced_redflagging_examples.py`
