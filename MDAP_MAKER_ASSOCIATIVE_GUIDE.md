# MDAP/MAKER + Associative Recomposition Integration Guide

## Overview

Complete integration of three powerful systems:
- **MDAP** (Multi-Agent Debate Protocol) - Multi-agent solution validation
- **MAKER** (Multi-step orchestration) - Structured workflow management
- **Associative Recomposition** - Domain-agnostic LLM + algorithmic verification

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     MAKER WORKFLOW ORCHESTRATION                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Initial Assessment                                       │
│    → Analyze problem, sub-solutions, conflicts                    │
│    → Estimate complexity                                          │
│                                                                     │
│  Step 2: Solution Generation Verification                         │
│    → Verify sub-solutions are valid                              │
│    → Check content exists                                         │
│                                                                     │
│  Step 3: Associative Recomposition                               │
│    → LLM classifies domain (not hardcoded)                       │
│    → LLM creates assembly plan (structured JSON)                 │
│    → AgentJSON parses plan (robust parsing)                      │
│    → Algorithmic assembly (verbatim insertion)                    │
│    → Algorithmic verification (ground truth)                      │
│    → LLM judgment (correctness evaluation)                       │
│                                                                     │
│  Step 4: Algorithmic Verification (Ground Truth)                  │
│    → Hash-based integrity checking                                │
│    → Code component verification                                   │
│    → Fingerprint detection                                        │
│                                                                     │
│  Step 5: MDAP Multi-Agent Validation                             │
│    → Multiple agents evaluate assembled solution                  │
│    → Each agent votes independently                               │
│    → Consensus reached (majority voting)                         │
│    → Aggregate metrics computed                                   │
│                                                                     │
│  Step 6: Complete                                                │
│    → Success if all checks pass                                   │
│    → Final assembled solution returned                            │
│                                                                     │
└───────────────────────────────────────────────────────────────────┘
```

## Component Integration

### 1. Associative Recomposition

**Role:** Domain-agnostic LLM + algorithmic assembly

**Features:**
- LLM sees full content (not metadata)
- LLM classifies problem domain itself
- AgentJSON parses structured output
- Algorithmic assembly (verbatim insertion)
- LLM judges correctness

**Output:**
```python
{
    'classification': {
        'domain': 'software_development',
        'field': 'web security',
        'complexity': 'medium'
    },
    'assembly_plan': {...},
    'judgment': {
        'is_correct': true,
        'quality_score': 0.90
    }
}
```

### 2. MDAP Validation

**Role:** Multi-agent solution validation

**Features:**
- Multiple agents evaluate solution
- Each agent votes independently
- Consensus reached via voting
- Aggregate metrics computed

**Output:**
```python
{
    'num_agents': 5,
    'consensus': {
        'decision': 'approve',
        'votes_for': 5,
        'votes_against': 0
    },
    'agreement_ratio': 1.0,
    'validation_details': {
        'avg_confidence': 0.92,
        'avg_quality': 0.90
    }
}
```

### 3. Ground Truth Store

**Role:** Persistent verification layer

**Features:**
- Content hashing (SHA-256)
- Algorithmic verification
- Multiple storage backends
- Code component detection

**Verification:**
```python
# Algorithmic checks
all_preserved, results = ground_truth_store.verify_all_solutions_preserved(
    assembled_output, sub_problem_ids
)
```

## Usage

### Basic Usage

```python
from mdap_maker_associative_integration import recompose_with_mdap_maker

# Run full workflow
results = recompose_with_mdap_maker(
    problem_statement="Build user management system",
    sub_solutions={
        'sol_1': {'solution_content': 'def auth(): ...'},
        'sol_2': {'solution_content': 'class User: ...'}
    },
    conflicts=[],
    llm_call_fn=lambda p: your_llm_api.call(p),
    mdap_agent_llm_calls=[
        lambda p: agent1.call(p),
        lambda p: agent2.call(p),
        lambda p: agent3.call(p),
        lambda p: agent4.call(p),
        lambda p: agent5.call(p)
    ]
)

# Check results
if results['success']:
    print("✓ Success!")
    print(f"Decision: {results['validation_results']['consensus']['decision']}")
    print(results['final_assembled'])
```

### Advanced Usage

```python
from mdap_maker_associative_integration import MakerRecomposerWorkflow

# Create workflow
workflow = MakerRecomposerWorkflow(
    use_mdap=True,
    use_associative=True,
    num_mdap_agents=7  # More agents
)

# Run with custom configuration
results = workflow.run_full_workflow(
    problem_statement=problem,
    sub_solutions=solutions,
    conflicts=conflicts,
    llm_call_fn=primary_llm,
    mdap_agent_llm_calls=agent_calls
)
```

## Workflow Stages

### Stage 1: Initial Assessment

```python
results['metadata']['initial_assessment'] = {
    'num_sub_solutions': 3,
    'num_conflicts': 1,
    'has_code': True,
    'estimated_complexity': 'medium'
}
```

### Stage 2: Solution Verification

```python
results['metadata']['solution_verification'] = {
    'sol_1': {
        'has_content': True,
        'length': 1234,
        'has_code': True,
        'confidence': 0.95
    },
    ...
}
```

### Stage 3: Associative Recomposition

```python
results['metadata']['associative_recomposition'] = {
    'classification': {
        'domain': 'software_development',
        'solution_type': 'code',
        'field': 'web security'
    },
    'judgment': {
        'is_correct': True,
        'quality_score': 0.90
    }
}
```

### Stage 4: Algorithmic Verification

```python
results['metadata']['algorithmic_verification'] = {
    'all_preserved': True,
    'verification_results': {
        'sol_1': (True, "Content preserved exactly"),
        'sol_2': (True, "Code components verified"),
        'sol_3': (True, "Fingerprint verified")
    }
}
```

### Stage 5: MDAP Validation

```python
results['metadata']['mdap_validation'] = {
    'num_agents': 5,
    'consensus': {
        'decision': 'approve',
        'votes_for': 5,
        'votes_against': 0
    },
    'agreement_ratio': 1.0,
    'validation_details': {
        'avg_confidence': 0.92,
        'avg_quality': 0.90,
        'avg_correctness': 0.91
    }
}
```

## Key Classes

### MDAPRecomposer

Multi-agent validation:

```python
recomposer = MDAPRecomposer(
    num_agents=5,
    voting_strategy="majority"
)

results = recomposer.validate_with_agents(
    assembled_content=assembled,
    plan=assembly_plan,
    sub_solutions=sub_solutions,
    agent_llm_calls=[agent1, agent2, ...]
)
```

### MakerRecomposerWorkflow

Full workflow orchestration:

```python
workflow = MakerRecomposerWorkflow(
    use_mdap=True,
    use_associative=True,
    num_mdap_agents=5
)

results = workflow.run_full_workflow(
    problem_statement=problem,
    sub_solutions=solutions,
    conflicts=conflicts,
    llm_call_fn=llm,
    mdap_agent_llm_calls=agents
)
```

## Decision Making

### Consensus Process

```
Agent 1: APPROVE (confidence: 0.90)
Agent 2: APPROVE (confidence: 0.88)
Agent 3: APPROVE (confidence: 0.95)
Agent 4: APPROVE (confidence: 0.85)
Agent 5: APPROVE (confidence: 0.92)

↓

Consensus: APPROVE
Votes For: 5
Votes Against: 0
Agreement: 100%
```

### Validation Metrics

```python
{
    'avg_confidence': 0.90,      # Average agent confidence
    'avg_quality': 0.89,          # Average quality rating
    'avg_correctness': 0.91,       # Average correctness rating
    'agreement_ratio': 1.0         # How much agents agree
}
```

## Error Handling

### Fallback Behavior

```
IF MDAP unavailable:
    → Use single validation
    → results['num_agents'] = 1

IF Associative unavailable:
    → Use fallback assembly (simple concatenation)
    → Warning logged

IF Ground Truth unavailable:
    → Skip algorithmic verification
    → Warning logged
    → Continue to MDAP validation
```

### Retry Logic

```python
# Associative recomposition has built-in retry
assembled, metadata = associative_recomposer.recompose_with_verification(
    sub_solutions=solutions,
    max_retries=3,
    llm_call_fn=llm
)

# Each attempt:
# - Gets new assembly plan from LLM
# - Verifies algorithmically
# - Evaluates with LLM judgment
# - Provides feedback on failure
```

## Best Practices

### 1. Use All Three Layers

```python
# Full system
results = recompose_with_mdap_maker(
    use_mdap=True,           # Multi-agent validation
    use_associative=True,    # LLM + algorithmic
    ...
)
```

### 2. Tune Agent Count

```python
# More agents = higher confidence but slower
# Fewer agents = faster but less robust

results = recompose_with_mdap_maker(
    num_mdap_agents=7,  # High stakes scenario
    ...
)
```

### 3. Provide Good Prompts

```python
def create_agent_prompt(agent_role: str) -> str:
    """Create role-specific prompts for agents"""
    return f"""You are a {agent_role} expert evaluating...
    Be thorough and honest in your assessment.
    """
```

### 4. Check All Metrics

```python
# Check consensus
if results['validation_results']['consensus']['decision'] == 'reject':
    # Handle rejection
    pass

# Check agreement
if results['validation_results']['agreement_ratio'] < 0.6:
    # Low agreement - manual review needed
    pass

# Check quality
if results['validation_results']['validation_details']['avg_quality'] < 0.7:
    # Low quality - may need revision
    pass
```

## Comparison: With vs Without MDAP/MAKER

| Aspect | Without | With |
|--------|--------|-------|
| Validation | Single LLM | 5+ agents debate |
| Confidence | Single point | Consensus-driven |
| Robustness | Fragile | Multiple perspectives |
| Workflow | Ad-hoc | Structured steps |
| Verification | Trust-based | Algorithmic + LLM |
| Traceability | Limited | Full audit trail |
| Error Recovery | Manual | Automatic retry |

## Files

- `mdap_maker_associative_integration.py` - Main integration
- `associative_recomposition.py` - Domain-agnostic LLM system
- `ground_truth_store.py` - Persistent verification layer
- `examples/mdap_maker_associative_example.py` - Working example
- `ASSOCIATIVE_RECOMPOSITION_GUIDE.md` - Associative system guide
- `ASSOCIATIVE_QUICKSTART.md` - Quick reference

## Summary

The MDAP/MAKER + Associative Recomposition system provides:

✅ **Multi-layer validation** - MDAP + Algorithmic + LLM judgment
✅ **Domain-agnostic** - Works for any problem type
✅ **Robust parsing** - AgentJSON handles malformed JSON
✅ **Algorithmic verification** - Content preservation guaranteed
✅ **Consensus-driven** - Multiple agents validate
✅ **Structured workflow** - MAKER orchestrates steps
✅ **Full traceability** - Every decision recorded

This is the most comprehensive recomposition system, combining the best of all approaches!
