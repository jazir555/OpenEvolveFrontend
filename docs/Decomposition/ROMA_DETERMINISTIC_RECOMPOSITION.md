# ROMA Deterministic Recomposition

## Overview

ROMA recomposition now supports **two modes** to address the critical issue of LLM-induced mutations in sub-solutions:

- **DETERMINISTIC MODE** (default): Sub-solutions remain immutable, ROMA only decides structure
- **CREATIVE MODE**: ROMA may rewrite and integrate sub-solutions

## The Problem: LLM Non-Determinism

When using LLMs for recomposition, a critical issue emerges:

### Creative Mode (Traditional ROMA)

```python
# LLM sees FULL CONTENT
task = """
Recompose these sub-solutions:

[sol_1] Authentication
Implement JWT authentication with refresh tokens...
[500 words of detailed content]

[sol_2] Profile Management
Create user profile with session handling...
[500 words of detailed content]

[sol_3] Authorization
Add role-based access control...
[500 words of detailed content]
"""

# LLM response rewrites everything
response = """
# Complete User Management System

## Authentication and Session Management
[LLM rewrites both sol_1 and sol_2, merging them, changing technical details]

## Role-Based Authorization
[LLM modifies sol_3 to align with its rewritten approach]
"""
```

**Problems:**
- ❌ Original code/functionality may be altered
- ❌ Technical details can be lost or changed
- ❌ Non-deterministic output (different every time)
- ❌ Hard to debug when something breaks
- ❌ Code precision lost in LLM "interpretation"

## The Solution: Deterministic Mode

### How Deterministic Mode Works

```
┌─────────────────────────────────────────────────────────────┐
│              TRADITIONAL (CREATIVE) MODE                     │
├─────────────────────────────────────────────────────────────┤
│  1. Send full sub-solution content to LLM                   │
│  2. LLM rewrites, merges, and modifies content              │
│  3. Return integrated solution                              │
│                                                              │
│  Result: Content mutations, non-deterministic output        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              DETERMINISTIC MODE (NEW!)                       │
├─────────────────────────────────────────────────────────────┤
│  1. Extract METADATA ONLY from sub-solutions                │
│     - Title, type, length, confidence                       │
│     - Dependencies, structure hints                         │
│     - NO full content                                       │
│                                                              │
│  2. Send METADATA to LLM for STRUCTURE DECISION             │
│     LLM Prompt: "Decide assembly order, headers,           │
│                  and transitions based on metadata"         │
│                                                              │
│  3. LLM returns STRUCTURE PLAN                              │
│     - ORDER: [sol_1, sol_2, sol_3]                         │
│     - HEADERS: {sol_1: "Authentication", ...}              │
│     - TRANSITIONS: [(sol_1, sol_2, "Connect text"), ...]   │
│                                                              │
│  4. DETERMINISTIC ASSEMBLY                                  │
│     - Insert sub-solutions VERBATIM in specified order      │
│     - Add ROMA's suggested headers                           │
│     - Insert ROMA's transitions between sections            │
│                                                              │
│  Result: Original content preserved, deterministic output  │
└─────────────────────────────────────────────────────────────┘
```

### Code Trace: Deterministic Mode

```python
# STEP 1: Extract metadata (immutable)
metadata = {
    'sol_1': {
        'title': 'JWT Authentication',
        'type': 'code',
        'confidence': 0.90,
        'length': 523,
        'dependencies': [],
    },
    'sol_2': {
        'title': 'User Profile Management',
        'type': 'code',
        'confidence': 0.85,
        'length': 412,
        'dependencies': [],
    },
    'sol_3': {
        'title': 'Role-Based Access Control',
        'type': 'code',
        'confidence': 0.88,
        'length': 387,
        'dependencies': ['sol_1'],
    },
}

# STEP 2: Build structure planning task
task = """
You are an expert solution architect.

OBJECTIVE: Determine OPTIMAL STRUCTURE for assembling 3 sub-solutions.

SUB-SOLUTION METADATA:
[sol_1] JWT Authentication
  - Type: code
  - Confidence: 0.90
  - Length: 523 chars
  - Dependencies: None

[sol_2] User Profile Management
  - Type: code
  - Confidence: 0.85
  - Length: 412 chars
  - Dependencies: None

[sol_3] Role-Based Access Control
  - Type: code
  - Confidence: 0.88
  - Length: 387 chars
  - Dependencies: sol_1

YOUR TASK - DECIDE STRUCTURE ONLY:
1. ASSEMBLY ORDER: [sol_1, sol_2, sol_3]
2. HEADERS: Suggest heading for each
3. TRANSITIONS: Brief connections between sections

CRITICAL: You decide STRUCTURE ONLY. Sub-solutions will be inserted VERBATIM.
"""

# STEP 3: ROMA returns structure plan
structure_plan = {
    'order': ['sol_1', 'sol_3', 'sol_2'],  # Note: sol_3 before sol_2
    'headers': {
        'sol_1': '## Authentication Foundation',
        'sol_3': '## Authorization Layer',
        'sol_2': '## User Profile System',
    },
    'transitions': [
        ('sol_1', 'sol_3', 'With authentication established, we add role-based access control.'),
        ('sol_3', 'sol_2', 'The profile system integrates with both auth and authorization.'),
    ],
    'intro': 'Complete user management system with authentication, authorization, and profiles.',
    'conclusion': 'All components use JWT tokens for stateless scalability.',
}

# STEP 4: Deterministic assembly
parts = []

# Add intro
parts.append(structure_plan['intro'])
parts.append('')

# Add sol_1 verbatim
parts.append('## Authentication Foundation')
parts.append('')
parts.append(sub_solutions['sol_1'].solution_content)  # VERBATIM INSERTION
parts.append('')

# Add transition
parts.append('With authentication established, we add role-based access control.')
parts.append('')

# Add sol_3 verbatim
parts.append('## Authorization Layer')
parts.append('')
parts.append(sub_solutions['sol_3'].solution_content)  # VERBATIM INSERTION
parts.append('')

# ... and so on

assembled = '\n'.join(parts)
```

**Key Insight:**
- Metadata extraction: **Deterministic** (no LLM)
- Structure planning: **LLM-assisted** (ROMA)
- Content insertion: **Deterministic** (verbatim)

## Usage Examples

### Example 1: Default Deterministic Mode

```python
from problem_recomposition import SolutionAssembler
from roma_recomposition_config import ROMARecompositionPresets

# Deterministic mode is DEFAULT
config = ROMARecompositionPresets.balanced()
# config.deterministic = True  # Already True by default

assembler = SolutionAssembler(enable_roma=True)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)

# Sub-solutions are inserted VERBATIM
# Only structure (order, headers, transitions) from ROMA
```

### Example 2: Explicit Creative Mode

```python
from roma_recomposition_config import ROMARecompositionConfig

# Enable creative mode (LLM may rewrite)
config = ROMARecompositionConfig(
    deterministic=False,  # Creative mode
    enable_roma=True,
)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)

# ROMA may rewrite, merge, and modify sub-solutions
# More integrated but less deterministic
```

### Example 3: Compare Side-by-Side

```python
# Deterministic assembly
result_deterministic = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    roma_deterministic=True,  # Verbatim insertion
)

# Creative assembly
result_creative = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    roma_deterministic=False,  # LLM rewriting allowed
)

# Compare
print(f"Deterministic length: {len(result_deterministic.assembled_content)}")
print(f"Creative length: {len(result_creative.assembled_content)}")

# Check if code is preserved
original_function = "def authenticate_user"
print(f"Code preserved in deterministic: {original_function in result_deterministic.assembled_content}")
print(f"Code preserved in creative: {original_function in result_creative.assembled_content}")
```

## When to Use Each Mode

### Use DETERMINISTIC Mode for:

✅ **Code and Technical Specifications**
```python
# Critical: Preserve exact code
sub_solution = """
def authenticate_user(username, password):
    hash = bcrypt.hashpw(password.encode(), SALT)
    return database.verify(username, hash)
"""
# Deterministic mode ensures this exact code is preserved
```

✅ **API Definitions and Contracts**
```python
# Critical: Preserve exact API signatures
sub_solution = """
POST /api/auth/login
Request: {username: str, password: str}
Response: {token: str, expires: timestamp}
"""
# Deterministic mode preserves contract exactly
```

✅ **Configuration Files**
```python
# Critical: Exact syntax matters
sub_solution = """
database:
  host: localhost
  port: 5432
  ssl: true
"""
# Deterministic mode preserves YAML/JSON exactly
```

✅ **Production Systems**
- Reproducible builds
- Debuggable issues
- Version control friendly
- Code review possible

✅ **Legal and Compliance Documents**
- Exact wording required
- Regulatory compliance
- Audit trails needed

### Use CREATIVE Mode for:

✅ **Documentation and Prose**
```python
# Less critical: Flow and readability more important
sub_solution = """
User authentication is a critical component...

[Multiple paragraphs of explanation]
"""
# Creative mode can improve flow and readability
```

✅ **Summaries and Overviews**
```python
# Less critical: High-level integration more important
sub_solution = """
The system provides secure access control mechanisms...

[High-level description]
"""
# Creative mode can create coherent narrative
```

✅ **Exploratory Drafting**
- Brainstorming sessions
- Early prototyping
- Iterative refinement

✅ **Content Generation**
- Blog posts
- Tutorials
- Explanatory guides

⚠️ **WARNING**: Never use creative mode for:
- Production code
- API contracts
- Configuration files
- Legal documents
- Security-critical components

## Configuration Reference

### ROMARecompositionConfig

```python
from roma_recomposition_config import ROMARecompositionConfig

config = ROMARecompositionConfig(
    # Mode selection
    deterministic=True,  # True=verbatim, False=creative

    # Other parameters
    enable_roma=True,
    max_depth=2,
    temperature=0.7,
)
```

### Direct Parameter

```python
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_deterministic=True,  # Verbatim insertion
)
```

### Presets

All presets default to deterministic mode:

```python
from roma_recomposition_config import ROMARecompositionPresets

# All these have deterministic=True by default
config = ROMARecompositionPresets.fast()        # deterministic=True
config = ROMARecompositionPresets.balanced()    # deterministic=True
config = ROMARecompositionPresets.thorough()    # deterministic=True
config = ROMARecompositionPresets.high_conflict()  # deterministic=True
config = ROMARecompositionPresets.code_focused()  # deterministic=True

# To enable creative mode with a preset:
config = ROMARecompositionPresets.balanced()
config.deterministic = False  # Override to creative
```

## Implementation Details

### Metadata Extraction

Deterministic mode extracts only metadata:

```python
def _extract_solution_metadata(self, sub_solutions):
    """
    Extract immutable metadata - NEVER returns full content.
    """
    metadata = {}
    for sol_id, solution in sub_solutions.items():
        content = solution.solution_content

        metadata[sol_id] = {
            'id': sol_id,
            'title': self._extract_title(content),  # First heading
            'type': self._detect_type(content),    # code/markdown/unknown
            'confidence': solution.confidence_score,
            'length': len(content),
            'line_count': len(content.split('\n')),
            'dependencies': self._extract_dependencies(content),
            'has_code_blocks': '```' in content,
            'has_headings': any(line.startswith('#') for line in content.split('\n')),
        }
    return metadata
```

**CRITICAL**: The `solution_content` is never included in metadata!

### Structure Planning Task

ROMA receives ONLY metadata:

```
OBJECTIVE: Determine OPTIMAL STRUCTURE

SUB-SOLUTION METADATA:
[sol_1] JWT Authentication
  - Type: code
  - Confidence: 0.90
  - Length: 523 chars
  - Dependencies: None

[sol_2] User Profile
  - Type: code
  - Confidence: 0.85
  - Length: 412 chars
  - Dependencies: None

CONFLICTS:
- OVERLAP: Both handle sessions

DECIDE STRUCTURE ONLY:
1. ASSEMBLY ORDER: [id1, id2, ...]
2. HEADERS: {id: "Heading", ...}
3. TRANSITIONS: [(id1, id2, "Text"), ...]

CRITICAL: Sub-solutions will be inserted VERBATIM
```

ROMA cannot rewrite content because it never sees it!

### Verbatim Assembly

```python
def _assemble_from_structure_plan(self, structure_plan, sub_solutions):
    """
    Insert sub-solutions VERBATIM using ROMA's structure.
    """
    parts = []

    for sol_id in structure_plan['order']:
        # Add ROMA's header
        header = structure_plan['headers'].get(sol_id)
        parts.append(f"## {header}")

        # Insert sub-solution VERBATIM (no modification)
        parts.append(sub_solutions[sol_id].solution_content)  # ← VERBATIM

        # Add ROMA's transition
        # ...

    return '\n'.join(parts)
```

## Quality Metrics

Both modes produce quality metrics:

```python
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    roma_deterministic=True,
)

# Access quality
print(f"Overall Quality: {result.quality_metrics.overall_score:.2f}")
print(f"Coherence: {result.quality_metrics.coherence_score:.2f}")
print(f"Integration: {result.quality_metrics.integration_quality:.2f}")
print(f"Consistency: {result.quality_metrics.consistency_score:.2f}")
```

**Note:** Deterministic mode may have slightly lower "integration_quality" because ROMA can't rewrite for perfect flow, but the tradeoff is **preserved correctness**.

## Troubleshooting

### Issue: Structure Parsing Fails

**Symptom:** `Warning: Could not parse ROMA structure plan`

**Solution:**
1. Check if ROMA is available
2. Verify structure task format
3. Falls back to hierarchical assembly automatically

### Issue: Sub-solutions Appear Disconnected

**Symptom:** No transitions between sections

**Solution:**
```python
# Provide more context to help ROMA generate better transitions
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    roma_context="Domain: E-commerce platform with microservices architecture",
    roma_deterministic=True,
)
```

### Issue: Wrong Assembly Order

**Symptom:** Dependencies not respected

**Solution:**
```python
# Ensure metadata includes dependencies
metadata = self._extract_solution_metadata(sub_solutions)
# metadata['sol_3']['dependencies'] = ['sol_1']  # Extracted automatically

# ROMA uses dependencies to determine order
```

## Best Practices

1. **Default to Deterministic Mode**
   ```python
   # Good: Safe default
   config = ROMARecompositionPresets.balanced()  # deterministic=True
   ```

2. **Use Creative Mode Intentionally**
   ```python
   # Explicit creative mode for documentation
   config = ROMARecompositionConfig(
       deterministic=False,  # Intentional creative mode
       temperature=0.8,  # Higher temperature for prose
   )
   ```

3. **Verify Code Preservation**
   ```python
   # For code solutions, verify integrity
   for sol_id, solution in sub_solutions.items():
       assert solution.solution_content in result.assembled_content
   ```

4. **Provide Domain Context**
   ```python
   # Helps ROMA make better structural decisions
   result = assembler.assemble_solution(
       decomposition_plan=plan,
       sub_solutions=sub_solutions,
       roma_context="Domain: Financial services with strict compliance requirements",
       roma_deterministic=True,
   )
   ```

## Summary

| Aspect | Deterministic Mode | Creative Mode |
|--------|-------------------|---------------|
| **Sub-solutions** | Inserted verbatim | May be rewritten |
| **Determinism** | High (reproducible) | Low (LLM variance) |
| **Code Integrity** | Preserved exactly | May be modified |
| **Transitions** | ROMA-generated | ROMA-generated |
| **Best For** | Code, APIs, configs | Documentation, prose |
| **Default** | ✅ Yes | ❌ No |

**Recommendation:** Start with deterministic mode. Only switch to creative mode when you explicitly want LLM rewriting for better prose flow.

## References

- **Problem Recomposition:** `problem_recomposition.py`
- **Configuration Helper:** `roma_recomposition_config.py`
- **Examples:** `examples/roma_recomposition_examples.py` (Example 7)
- **Integration Guide:** `ROMA_RECOMPOSITION_INTEGRATION.md`
