# Enhanced Red Flagging System - Documentation

Complete guide to the multi-layered red flagging system with LMQL and Guardrails integration.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [When to Use](#when-to-use)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Enhanced Red Flagging System provides a **multi-layered approach** to content validation:

### Layers of Protection

1. **Pre-Generation (LMQL Constraints)**: Prevents flagged content from being generated
   - 70-90% cost reduction
   - Catches issues before token generation
   - Configurable constraints for length, format, keywords

2. **Post-Generation (Guardrails)**: Validates output after generation
   - Comprehensive validation checks
   - Toxic language detection
   - PII/Secrets detection
   - Schema validation

3. **Custom Rules**: Domain-specific validation
   - Keyword requirements
   - Pattern matching
   - Business logic validation

### Key Benefits

- **Proactive Prevention**: Stop bad content before it's generated
- **Cost Efficient**: LMQL constraints reduce waste by 70-90%
- **Flexible**: Configurable rules for different use cases
- **Observable**: Detailed statistics and monitoring
- **Production Ready**: Comprehensive error handling

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User Request                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│     Layer 1: LMQL Pre-Generation Constraints            │
│     • Max tokens/characters                              │
│     • Required format (JSON, etc.)                      │
│     • Forbidden/Required keywords                       │
│     • Confidence thresholds                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            LLM Generation                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│     Layer 2: Post-Generation Validation                 │
│     • Basic validation (length, confidence)             │
│     • Pattern matching (regex)                          │
│     • Guardrails validators (toxicity, PII, secrets)    │
│     • Schema validation                                 │
│     • Custom rules                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Result Processing                          │
│     • Accept (no flags)                                 │
│     • Flag for review (medium/low severity)             │
│     • Reject (critical/high severity)                   │
│     • Remediate (if applicable)                         │
└─────────────────────────────────────────────────────────┘
```

---

## When to Use

### ✅ Ideal Use Cases

1. **Security-Critical Applications**
   - Authentication systems
   - Financial transactions
   - Sensitive data processing

2. **Content Generation**
   - Automated responses
   - Report generation
   - Documentation creation

3. **Multi-Agent Systems**
   - MDAP voting validation
   - Agent orchestration
   - Distributed decision making

4. **API Integration**
   - Request/response validation
   - Third-party integrations
   - Webhook processing

### ❌ Not Recommended For

- Ultra-low latency requirements (< 50ms)
- Simple boolean validations
- Static content without LLM generation

---

## Configuration Options

### EnhancedRedFlagRules

```python
from reliability.enhanced_redflagger import EnhancedRedFlagRules

rules = EnhancedRedFlagRules(
    # Basic Rules
    max_tokens=750,                    # Maximum tokens allowed
    max_characters=6000,               # Maximum character length
    min_confidence=0.2,                # Minimum confidence score
    require_schema_match=True,         # Require schema compliance

    # LMQL Pre-Generation
    enable_lmql_constraints=True,      # Enable pre-generation checks
    lmql_max_retries=3,                # Retry attempts for LMQL

    # Guardrails Validators
    guardrails_validators=[
        "toxic_language",              # Detect toxic content
        "pii_filter",                  # Detect PII
        "secrets_detection",           # Detect API keys, passwords
        "malicious_patterns"           # Detect security threats
    ],

    # Custom Rules
    required_keywords=["solution"],    # Keywords that must appear
    forbidden_keywords=["error"],      # Keywords that must NOT appear
    required_format="json",            # Required output format

    # Thresholds
    toxicity_threshold=0.8,            # Toxicity threshold (0-1)
    pii_detection_strict=True,         # Strict PII detection
    confidence_threshold=0.5           # Confidence threshold
)
```

### Severity Levels

Red flags are categorized by severity:

- **CRITICAL**: Security threats, malicious content
  - Action: Always reject
  - Example: API keys, passwords, injection attempts

- **HIGH**: Violates core requirements
  - Action: Reject or require manual review
  - Example: Toxic language, blocked patterns

- **MEDIUM**: Quality issues, edge cases
  - Action: Flag for review, consider remediation
  - Example: Exceeds length limit, low confidence

- **LOW**: Minor issues, warnings
  - Action: Log for monitoring
  - Example: Minor formatting issues

---

## Usage Examples

### Example 1: Basic Red Flagging

```python
from reliability.enhanced_redflagger import create_enhanced_redflagger

# Create flagger with defaults
flagger = create_enhanced_redflagger()

# Validate output
is_flagged, flags = flagger.check_for_red_flags(
    raw_text="This is the response text",
    candidate={"response": "This is the response text"},
    schema=None
)

if is_flagged:
    for flag in flags:
        print(f"🚩 {flag.category}: {flag.message}")
        print(f"   Severity: {flag.severity.value}")
        print(f"   Remediation: {flag.remediation}")
```

### Example 2: Custom Rules

```python
from reliability.enhanced_redflagger import (
    EnhancedRedFlagger,
    EnhancedRedFlagRules
)

# Define strict rules for financial data
financial_rules = EnhancedRedFlagRules(
    max_tokens=500,
    max_characters=3000,
    confidence_threshold=0.9,
    guardrails_validators=[
        "toxic_language",
        "pii_filter",
        "secrets_detection"
    ],
    forbidden_keywords=["confidential", "internal"],
    required_keywords=["amount", "currency"],
    required_format="json",
    toxicity_threshold=0.95
)

flagger = EnhancedRedFlagger(rules=financial_rules)
```

### Example 3: LMQL Pre-Generation

```python
from reliability.lmql_adapter import get_default_adapter

# Get constraints for pre-generation
constraints = flagger.get_lmql_constraints()

# Use with LMQL adapter
lmql = get_default_adapter()
result = lmql.constrained_generation(
    prompt="Generate a financial report",
    constraints=constraints
)
```

### Example 4: Integration with MDAP

```python
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import (
    solve_with_redflagging
)

# Solve with enhanced validation
result = solve_with_redflagging(
    task="Calculate the ROI for this investment",
    mdap_k_ahead=5,
    use_lmql_constraints=True,
    use_enhanced_validation=True,
    schema={
        "type": "object",
        "required": ["roi", "confidence"]
    }
)

if result['success']:
    print(f"Result: {result['result']}")
else:
    print(f"Red Flags: {result['red_flag_count']}")
```

### Example 5: Severity-Based Handling

```python
is_flagged, flags = flagger.check_for_red_flags(
    raw_text=output,
    candidate=candidate
)

# Handle based on severity
critical = [f for f in flags if f.severity == RedFlagSeverity.CRITICAL]
high = [f for f in flags if f.severity == RedFlagSeverity.HIGH]

if critical or high:
    # Reject output
    return {"status": "rejected", "reason": flags}
else:
    # Accept or flag for review
    return {"status": "accepted", "output": output}
```

### Example 6: Statistics and Monitoring

```python
# Get comprehensive statistics
stats = flagger.get_statistics()

print(f"Total Checks: {stats['total_checks']}")
print(f"Flag Rate: {stats['flag_rate']:.2%}")
print(f"Prevention Rate: {stats['prevention_rate']:.2%}")
print(f"Critical Flags: {stats['critical_flags']}")
print(f"High Flags: {stats['high_flags']}")

# Reset statistics if needed
flagger.reset_statistics()
```

---

## Best Practices

### 1. Layered Defense

Always use multiple layers of validation:

```python
# ✅ GOOD: Multiple layers
rules = EnhancedRedFlagRules(
    enable_lmql_constraints=True,      # Prevent
    guardrails_validators=[...],        # Detect
    required_keywords=[...],            # Ensure
    confidence_threshold=0.7            # Quality
)

# ❌ BAD: Single layer
rules = EnhancedRedFlagRules(
    enable_lmql_constraints=False,
    guardrails_validators=[]
)
```

### 2. Appropriate Severity

Assign correct severity levels:

```python
# ✅ GOOD: Critical for security
if "api_key" in text:
    return RedFlag(
        category="secrets_detection",
        severity=RedFlagSeverity.CRITICAL
    )

# ❌ BAD: Low for security
if "api_key" in text:
    return RedFlag(
        category="secrets_detection",
        severity=RedFlagSeverity.LOW
    )
```

### 3. Schema Validation

Always provide schemas when possible:

```python
# ✅ GOOD: With schema
is_flagged, flags = flagger.check_for_red_flags(
    raw_text=output,
    candidate=json.loads(output),
    schema={
        "type": "object",
        "required": ["answer", "confidence"]
    }
)

# ❌ BAD: No schema
is_flagged, flags = flagger.check_for_red_flags(
    raw_text=output,
    candidate=output,
    schema=None
)
```

### 4. Confidence Thresholds

Set appropriate confidence thresholds:

```python
# For critical systems (financial, medical)
rules = EnhancedRedFlagRules(
    confidence_threshold=0.9  # High confidence required
)

# For creative tasks
rules = EnhancedRedFlagRules(
    confidence_threshold=0.5  # Lower confidence acceptable
)
```

### 5. Monitor Statistics

Regularly review statistics:

```python
# Check weekly
stats = flagger.get_statistics()

if stats['flag_rate'] > 0.3:
    # High flag rate - adjust rules or prompts
    logger.warning(f"High flag rate: {stats['flag_rate']:.2%}")

if stats['prevention_rate'] < 0.5:
    # Low prevention rate - enable more LMQL constraints
    logger.warning(f"Low prevention rate: {stats['prevention_rate']:.2%}")
```

---

## Troubleshooting

### Issue: High False Positive Rate

**Symptoms**: Many legitimate outputs are flagged

**Solutions**:
1. Adjust thresholds to appropriate levels
```python
rules = EnhancedRedFlagRules(
    toxicity_threshold=0.9,  # Increase from 0.8
    confidence_threshold=0.6  # Decrease from 0.7
)
```

2. Remove overly restrictive validators
```python
rules = EnhancedRedFlagRules(
    guardrails_validators=[
        "toxic_language",
        "secrets_detection"
        # Remove "pii_filter" if not needed
    ]
)
```

### Issue: High False Negative Rate

**Symptoms**: Bad content is not being caught

**Solutions**:
1. Enable LMQL constraints
```python
rules = EnhancedRedFlagRules(
    enable_lmql_constraints=True,  # Was False
    lmql_max_retries=3
)
```

2. Add more validators
```python
rules = EnhancedRedFlagRules(
    guardrails_validators=[
        "toxic_language",
        "pii_filter",
        "secrets_detection",
        "malicious_patterns",  # Add this
        "competitor_check"     # Add this
    ]
)
```

3. Lower thresholds
```python
rules = EnhancedRedFlagRules(
    toxicity_threshold=0.7,  # Decrease from 0.8
    confidence_threshold=0.8  # Increase from 0.5
)
```

### Issue: LMQL Not Working

**Symptoms**: LMQL constraints not being applied

**Solutions**:
1. Check availability
```python
stats = flagger.get_statistics()
print(f"LMQL Available: {stats['lmql_available']}")
```

2. Ensure LMQL is installed
```bash
pip install lmql
```

3. Verify adapter initialization
```python
from reliability.lmql_adapter import get_default_adapter
adapter = get_default_adapter()
print(f"Available: {adapter.is_available()}")
```

### Issue: Guardrails Not Working

**Symptoms**: Guardrails validators not catching issues

**Solutions**:
1. Check availability
```python
stats = flagger.get_statistics()
print(f"Guardrails Available: {stats['guardrails_available']}")
```

2. Install Guardrails
```bash
pip install guardrails-ai
```

3. Verify validator names
```python
# Use correct validator names
rules = EnhancedRedFlagRules(
    guardrails_validators=[
        "toxic_language",    # Correct
        "pii_filter",        # Correct
        # "toxicity"         # Incorrect - will fail
    ]
)
```

### Issue: Performance Degradation

**Symptoms**: Validation is too slow

**Solutions**:
1. Enable LMQL for pre-generation prevention
```python
rules = EnhancedRedFlagRules(
    enable_lmql_constraints=True,  # Prevents 70-90% of waste
)
```

2. Reduce number of validators
```python
rules = EnhancedRedFlagRules(
    guardrails_validators=[
        "toxic_language",
        "secrets_detection"
        # Remove non-critical validators
    ]
)
```

3. Adjust LMQL retry count
```python
rules = EnhancedRedFlagRules(
    lmql_max_retries=1  # Reduce from 3
)
```

### Issue: Schema Validation Fails

**Symptoms**: Valid outputs fail schema validation

**Solutions**:
1. Verify schema format
```python
# ✅ GOOD: Correct schema
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}

# ❌ BAD: Incorrect schema
schema = {
    "type": "object",
    "answer": "string",  # Wrong format
    "confidence": 0.5
}
```

2. Disable schema matching if not needed
```python
rules = EnhancedRedFlagRules(
    require_schema_match=False  # Disable strict schema validation
)
```

---

## Additional Resources

- **Examples**: See `enhanced_redflagging_examples.py` for complete working examples
- **API Reference**: See `enhanced_redflagger.py` for full API documentation
- **LMQL Documentation**: https://lmql.ai/
- **Guardrails Documentation**: https://guardrails.ai/

---

## Support

For issues or questions:
1. Check this documentation first
2. Review the examples in `enhanced_redflagging_examples.py`
3. Check logs for detailed error messages
4. Open an issue on the project repository

---

**Last Updated**: 2025-01-10
**Version**: 2.0.0
