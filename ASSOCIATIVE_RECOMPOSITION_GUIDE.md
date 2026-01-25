# Associative Recomposition System

## Overview

Domain-agnostic recomposition system that blends **generative LLM reasoning** with **algorithmic verification** and **LLM judgment** to create reliable, verifiable solution assembly.

## Core Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE PROBLEM                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Hardcoded triggers don't scale across domains               │
│  • LLMs can mutate/corrupt content during assembly             │
│  • Algorithmic checks can't judge correctness                  │
│  • Need both LLM judgment AND algorithmic verification         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    THE SOLUTION                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. LLM as CLASSIFIER - Identifies problem domain itself       │
│  2. LLM as ARCHITECT - Outputs structured JSON plan            │
│  3. Algorithmic LAYER - Ensures content preserved              │
│  4. LLM as JUDGE - Evaluates correctness (domain-specific)     │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 1: GENERATIVE                         │
│                                                                   │
│  Sub-Solutions (Full Content) ──→ LLM Judge                      │
│  ↓                                                                │
│  LLM sees EVERYTHING to make accurate decisions                   │
│  • Classifies problem domain (not hardcoded)                     │
│  • Identifies each component's purpose                           │
│  • Decides assembly strategy                                     │
│  • Outputs structured JSON                                       │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 2: PREDICTIVE                         │
│                                                                   │
│  LLM Response ──→ AgentJSON Parser ──→ AssemblyPlanJSON          │
│  ↓                                                                 │
│  Structured JSON with:                                            │
│  • Domain classification (domain, type, field, complexity)        │
│  • Sub-problem identities (what each component IS)               │
│  • Assembly instructions (keep_verbatim, merge, etc.)            │
│  • Transitions and structure                                     │
│  • Target solution specification                                 │
│  • Success criteria                                               │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 3: ALGORITHMIC                        │
│                                                                   │
│  AssemblyPlanJSON + Sub-Solutions ──→ Algorithmic Assembly        │
│  ↓                                                                 │
│  Algorithmic operations:                                          │
│  • Sort by position                                               │
│  • Insert content VERBATIM                                        │
│  • Add headers and transitions                                   │
│  • Execute instructions (no LLM involved)                        │
│  ↓                                                                 │
│  Ground Truth Verification                                        │
│  • Content hash matching                                         │
│  • Code component verification                                    │
│  • Fingerprint detection                                         │
│  • Ensure NOTHING lost                                           │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 4: JUDGMENT                           │
│                                                                   │
│  Assembled Solution ──→ LLM Judge ──→ Correctness Evaluation      │
│  ↓                                                                 │
│  LLM evaluates:                                                   │
│  • Is the solution correct?                                      │
│  • Are all components present?                                   │
│  • Does it meet success criteria?                                │
│  • Quality assessment                                            │
│  • Domain-specific correctness judgment                          │
│                                                                   │
│  If judgment FAILS → Retry with feedback                         │
└───────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Domain-Agnostic Associative System

**Problem:** Hardcoded triggers don't scale.

**Solution:** LLM classifies the problem itself.

```python
# LLM determines this (not hardcoded!)
{
    "classification": {
        "problem_type": "User authentication and authorization system",
        "domain": "software_development",  # LLM's choice
        "solution_type": "code",           # LLM's choice
        "field": "web security",           # LLM's choice
        "complexity": "medium",            # LLM's choice
        "confidence": 0.92,
        "reasoning": "This is a web security problem..."
    }
}
```

**Benefits:**
- ✅ No hardcoded domain lists
- ✅ Works for ANY problem type
- ✅ LLM reasoning guides process
- ✅ Self-describing system

### 2. AgentJSON Integration

**Problem:** LLMs often output malformed JSON.

**Solution:** AgentJSON with probabilistic repair.

```python
# AgentJSON handles:
- Truncated output
- Malformed JSON
- Missing quotes
- Extra commas
- Incomplete structures
- Partial results accepted

options = RepairOptions(
    mode="probabilistic",
    top_k=3,
    beam_width=32,
    max_repairs=50,
    partial_ok=True  # Accept partial results
)

result = agentjson_parse(llm_response, options)
```

**Benefits:**
- ✅ Robust to LLM errors
- ✅ Handles malformed output
- ✅ Graceful degradation
- ✅ Multiple repair strategies

### 3. Ground Truth Verification

**Problem:** How to ensure content wasn't lost/corrupted?

**Solution:** Ground truth store with algorithmic verification.

```python
# Step 1: Store ground truth
ground_truth_store.store_sub_solution(
    sub_problem_id="sol_1",
    solution_content=original_content,  # Stored with hash
    ...
)

# Step 2: Verify assembly
all_preserved, results = ground_truth_store.verify_all_solutions_preserved(
    assembled_output=assembled,
    sub_problem_ids=["sol_1", "sol_2", "sol_3"]
)

# Algorithmic checks:
- Exact content match
- Hash verification
- Code component detection
- Fingerprint matching
```

**Benefits:**
- ✅ Algorithmic (not LLM-dependent)
- ✅ Deterministic verification
- ✅ Hash-based integrity checking
- ✅ No trust in LLM required

### 4. LLM as Final Judge

**Problem:** Algorithmic checks can't judge correctness.

**Solution:** LLM evaluates domain-specific correctness.

```python
# LLM judges:
{
    "is_correct": true,
    "completeness_score": 0.95,
    "quality_score": 0.90,
    "missing_elements": [],
    "issues": ["Could use error handling examples"],
    "strengths": ["All components present", "Proper JWT usage"],
    "verdict": "good",
    "reasoning": "The reassembled solution correctly includes..."
}
```

**Benefits:**
- ✅ Domain-specific judgment
- ✅ Understands context
- ✅ Can evaluate quality
- ✅ Provides actionable feedback

## Usage Examples

### Basic Usage

```python
from associative_recomposition import AssociativeRecomposer

# Create recomposer
recomposer = AssociativeRecomposer(
    use_agentjson=True,  # Use AgentJSON for robust parsing
    max_retries=3
)

# Sub-solutions from decomposition
sub_solutions = {
    'sol_1': {
        'description': 'Authentication',
        'solution_content': 'def authenticate(): ...',
        'confidence_score': 0.95
    },
    'sol_2': {
        'description': 'User Profile',
        'solution_content': 'class UserProfile: ...',
        'confidence_score': 0.90
    }
}

# LLM call function
def llm_call_fn(prompt: str) -> str:
    # Call your LLM API (OpenAI, Anthropic, etc.)
    response = your_llm_api.call(prompt)
    return response

# Run recomposition
assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=sub_solutions,
    conflicts=[],
    problem_statement="Build user management system",
    llm_call_fn=llm_call_fn
)

if assembled:
    print("✓ Success!")
    print(f"Domain: {metadata['classification']['domain']}")
    print(f"Judgment: {metadata['judgment']['verdict']}")
    print(assembled)
```

### With Custom LLM Provider

```python
import anthropic

def anthropic_llm_call(prompt: str) -> str:
    """Use Anthropic Claude for LLM calls."""
    client = anthropic.Anthropic(api_key="your-key")

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

# Use with recomposer
assembled, metadata = recomposer.recompose_with_verification(
    sub_solutions=sub_solutions,
    conflicts=conflicts,
    problem_statement=problem_statement,
    llm_call_fn=anthropic_llm_call
)
```

### Accessing Classification Results

```python
assembled, metadata = recomposer.recompose_with_verification(...)

# LLM's domain classification
cls = metadata['classification']
print(f"Domain: {cls['domain']}")           # e.g., "software_development"
print(f"Type: {cls['solution_type']}")      # e.g., "code"
print(f"Field: {cls['field']}")             # e.g., "web security"
print(f"Complexity: {cls['complexity']}")   # e.g., "medium"

# Sub-problem identities (LLL-provided)
plan = metadata['attempts'][0]['plan']
for instr in plan['instructions']:
    print(f"{instr['sub_problem_id']}: {instr['sub_problem_identity']}")
    # Output:
    # sol_1: JWT authentication module with token generation
    # sol_2: User profile management with CRUD operations
    # sol_3: Role-based access control middleware
```

### Accessing Judgment Details

```python
judgment = metadata['judgment']

print(f"Correct: {judgment['is_correct']}")
print(f"Completeness: {judgment['completeness_score']:.2f}")
print(f"Quality: {judgment['quality_score']:.2f}")
print(f"Verdict: {judgment['verdict']}")

print("\nStrengths:")
for strength in judgment['strengths']:
    print(f"  ✓ {strength}")

print("\nIssues:")
for issue in judgment['issues']:
    print(f"  ⚠ {issue}")

print(f"\nReasoning:\n{judgment['reasoning']}")
```

## JSON Schema

The LLM must output this exact JSON structure:

```json
{
    "classification": {
        "problem_type": "Free-form description of problem type",
        "domain": "software_development|data_science|machine_learning|devops|security|business|research|education|legal|healthcare|finance|other",
        "solution_type": "code|documentation|configuration|api_spec|data_model|architecture|workflow|analysis|report|tutorial|other",
        "field": "Specific field (e.g., 'web authentication')",
        "complexity": "low|medium|high|expert",
        "confidence": 0.95,
        "reasoning": "Explanation of classification"
    },
    "target_solution_type": "code|documentation|...",
    "target_solution_description": "What final output should be",
    "success_criteria": [
        "Criterion 1",
        "Criterion 2",
        "Criterion 3"
    ],
    "sub_problem_identities": {
        "sol_1": "What this component IS",
        "sol_2": "What this component IS",
        "sol_3": "What this component IS"
    },
    "instructions": [
        {
            "sub_problem_id": "sol_1",
            "sub_problem_identity": "Reminder of what this is",
            "action": "keep_verbatim|merge|reorder|skip",
            "section_header": "Header for this section",
            "position": 0,
            "preserve_integrity": true,
            "merge_with": null,
            "transformations": null,
            "transition_before": null,
            "transition_after": "Transition text",
            "notes": "Additional notes"
        }
    ],
    "intro": "Brief introduction",
    "conclusion": "Brief conclusion",
    "global_notes": "Important notes",
    "confidence_score": 0.95,
    "reasoning": "Assembly strategy explanation",
    "estimated_quality": "low|medium|high|excellent"
}
```

## Verification Results

Algorithmic verification produces detailed results:

```python
verification_results = {
    'sol_1': (True, "Content preserved exactly (hash: a1b2c3d4...)"),
    'sol_2': (True, "Code components verified present"),
    'sol_3': (False, "Content NOT preserved - original content not found")
}

# Check overall
all_preserved = all(preserved for preserved, _ in verification_results.values())
```

## Retry Loop with Feedback

System automatically retries on failure:

```
Attempt 1:
  → LLM creates assembly plan
  → Algorithmic verification: FAIL (sol_3 missing)
  → Feedback: "Content not preserved for: ['sol_3']"
  → Retry with feedback

Attempt 2:
  → LLM creates revised plan (with feedback in prompt)
  → Algorithmic verification: PASS
  → LLM judgment: "is_correct: true"
  → SUCCESS
```

## Error Handling

### AgentJSON Fallback

```python
# If AgentJSON fails, falls back to standard JSON
if AGENTJSON_AVAILABLE:
    plan, errors = recomposer._parse_with_agentjson(llm_response)
else:
    plan, errors = recomposer._parse_with_json(llm_response)
```

### Parse Errors

```python
plan, parse_errors = recomposer.parse_llm_response(llm_response)

if plan is None:
    print(f"Parse failed: {parse_errors}")
    # Retry with feedback
```

### Verification Failures

```python
all_preserved, results = ground_truth_store.verify_all_solutions_preserved(
    assembled, sub_problem_ids
)

if not all_preserved:
    for sub_id, (preserved, details) in results.items():
        if not preserved:
            print(f"✗ {sub_id}: {details}")
    # Don't proceed to judgment if algorithmic check fails
```

## Best Practices

### 1. Always Use Ground Truth Storage

```python
# Ground truth is automatically stored, but you can access it:
ground_truth = ground_truth_store.get_sub_solution('sol_1')
print(f"Original hash: {ground_truth.content_hash}")
```

### 2. Check Classification

```python
# LLM's classification helps understand the problem
cls = metadata['classification']

# Use classification to guide post-processing
if cls['solution_type'] == 'code':
    # Run syntax checker
    pass
elif cls['solution_type'] == 'documentation':
    # Run style checker
    pass
```

### 3. Review LLM Reasoning

```python
# Understanding WHY LLM made decisions helps debug
plan = metadata['attempts'][0]['plan']
print(f"Reasoning: {plan['reasoning']}")

for instr in plan['instructions']:
    print(f"{instr['sub_problem_id']}: {instr['notes']}")
```

### 4. Use Judgment for Quality Control

```python
judgment = metadata['judgment']

# Only accept high-quality assemblies
if judgment['quality_score'] < 0.7:
    # Retry or flag for review
    pass

# Check for missing elements
if judgment['missing_elements']:
    # Critical issue - must fix
    pass
```

### 5. Handle Different Domains

```python
# Classification is domain-agnostic - works for any problem
cls = metadata['classification']

if cls['domain'] == 'healthcare':
    # Apply HIPAA compliance checks
    pass
elif cls['domain'] == 'finance':
    # Apply SOX compliance checks
    pass
```

## Troubleshooting

### AgentJSON Not Available

```
Warning: AgentJSON not available, falling back to json.loads
```

**Solution:** Install AgentJSON or it will use standard JSON.

### Parse Failures

```
Error: Failed to parse LLM response
```

**Solution:**
1. Check LLM response format
2. Ensure prompt is clear about JSON output
3. Use AgentJSON for robustness

### Verification Failures

```
Error: Algorithmic verification FAILED - content missing
```

**Solution:**
1. Check which sub-solutions are missing
2. Review assembly instructions
3. Ensure `keep_verbatim` action is used

### Judgment Failures

```
Warning: LLM judgment FAILED - solution incorrect
```

**Solution:**
1. Review LLM's reasoning
2. Check missing_elements
3. Verify success criteria are met
4. May need manual review

## Comparison: Traditional vs Associative

| Aspect | Traditional | Associative |
|--------|-------------|-------------|
| Domain Classification | Hardcoded triggers | LLM classifies |
| JSON Parsing | Fragile (json.loads) | Robust (AgentJSON) |
| Content Preservation | Trust LLM | Algorithmic verification |
| Correctness | Assumed | LLM judgment |
| Domain Coverage | Limited | Unlimited |
| Error Recovery | Limited | Retry with feedback |
| Transparency | Black box | Full reasoning |

## Files

- `associative_recomposition.py` - Main system
- `ground_truth_store.py` - Ground truth storage
- `examples/associative_recomposition_example.py` - Working example
- `agentjson/` - Robust JSON parsing library

## Summary

The associative recomposition system provides:

✅ **Domain-agnostic** - Works for any problem type
✅ **Robust parsing** - AgentJSON handles malformed JSON
✅ **Algorithmic verification** - Content preservation guaranteed
✅ **LLM judgment** - Domain-specific correctness evaluation
✅ **Self-describing** - LLM classifies and explains
✅ **Retry with feedback** - Automatic recovery from failures

This blends the best of LLM reasoning (classification, judgment) with algorithmic correctness (verification, ground truth).
