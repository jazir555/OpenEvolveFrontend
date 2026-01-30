# RAGBits Integration - Complete Scoring System Documentation

**Version:** 1.0
**Last Updated:** 2025-12-29
**Phase:** 3 & 4 (Evaluation Framework + Enhanced Knowledge Base)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scoring System Architecture](#scoring-system-architecture)
3. [Scoring Mechanisms](#scoring-mechanisms)
4. [Multi-Dimensional Scoring](#multi-dimensional-scoring)
5. [Metrics Collection and Analysis](#metrics-collection-and-analysis)
6. [Historical Comparison and Trend Analysis](#historical-comparison-and-trend-analysis)
7. [Scoring Algorithms and Formulas](#scoring-algorithms-and-formulas)
8. [Configuration and Customization](#configuration-and-customization)
9. [Examples and Use Cases](#examples-and-use-cases)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)

---

## Executive Summary

### What is the Scoring System?

The RAGBits Integration scoring system is a **multi-dimensional, multi-mechanism evaluation framework** that provides comprehensive quality assessment for workflow artifacts. It is **domain-agnostic** and can evaluate:

- Software code and implementations
- Scientific experimental designs
- Architecture documents
- Process/procedure documentation
- Any structured technical artifact

### Key Principles

1. **Multi-Dimensional**: Scores across 8 independent dimensions
2. **Multi-Mechanism**: Uses 7 different scoring approaches
3. **Weighted**: Different dimensions have different importance
4. **Contextual**: Scores are compared against historical data
5. **Actionable**: Provides specific recommendations, not just scores

### Scoring at a Glance

```
Input: Artifact (code, document, design, etc.)
  ↓
[7 Scoring Mechanisms]
  ↓
[8 Dimensional Scores (0-10)]
  ↓
[Weighted Overall Score (0-10)]
  ↓
[Verdict: EXCELLENT | GOOD | ACCEPTABLE | POOR]
  ↓
[Recommendations for Improvement]
```

---

## Scoring System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCORING SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Gauntlet   │  │   Metrics    │  │  Historical  │    │
│  │   Validator  │  │   Analyzer   │  │  Comparator  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│          │                 │                  │            │
│          └─────────────────┼──────────────────┘            │
│                           │                               │
│                    ┌──────▼──────┐                         │
│                    │  Scoring    │                         │
│                    │  Engine     │                         │
│                    └──────┬──────┘                         │
│                           │                               │
│  ┌──────────────────────────┼──────────────────────────┐  │
│  │                          │                          │  │
│  ▼                          ▼                          ▼  │
│ ┌─────────┐          ┌─────────┐          ┌─────────┐  │
│ │ Report  │          │Dashboard│          │ Alerts  │  │
│ │ Generator│         │ Generator│          │ System  │  │
│ └─────────┘          └─────────┘          └─────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| **Gauntlet Validator** | Runs automated tests | Artifact text | Test results (0-10 each) |
| **Metrics Collector** | Gathers quantitative metrics | Artifact data | Metric values |
| **Metrics Analyzer** | Analyzes metrics for insights | Collected metrics | Category scores (0-1) |
| **Historical Comparator** | Compares with historical data | Current + historical | Percentiles, trends |
| **Scoring Engine** | Aggregates all scores | All component outputs | Final scores, verdicts |
| **Report Generator** | Creates human-readable reports | Scoring data | Comprehensive reports |
| **Dashboard Generator** | Visualizes scoring data | Scoring data | Charts, HTML |

---

## Scoring Mechanisms

### Overview

The scoring system uses **7 distinct mechanisms** to evaluate artifacts:

1. **Explicit Test Functions** - Rule-based automated tests
2. **Keyword Frequency Analysis** - Term counting and frequency
3. **Pattern Matching** - Regex-based extraction
4. **LLM-Based Evaluation** - AI-powered judgment
5. **Comparative Analysis** - Historical comparison
6. **Heuristic Rules** - Experience-based scoring
7. **Statistical Analysis** - Trend detection

Each mechanism is explained in detail below.

---

## 1. Explicit Test Functions

### Purpose

Execute predefined test cases with pass/fail criteria and numerical scoring.

### Mechanism

```python
# Pseudo-code for explicit test function
async def run_explicit_test(artifact_content, test_criteria):
    """
    Executes a specific test with defined criteria.

    Args:
        artifact_content: The text/content to test
        test_criteria: Dictionary defining:
            - required_elements: List of items that must be present
            - thresholds: Pass/warning/fail thresholds
            - scoring_formula: How to calculate score

    Returns:
        {
            "test_name": str,
            "result": "passed" | "warning" | "failed",
            "score": float (0-10),
            "details": dict
        }
    """

    # 1. Check for required elements
    found = 0
    for element in test_criteria['required_elements']:
        if element.lower() in artifact_content.lower():
            found += 1

    # 2. Calculate coverage
    coverage = found / len(test_criteria['required_elements'])

    # 3. Apply scoring formula
    score = coverage * 10  # Convert to 0-10 scale

    # 4. Determine result based on thresholds
    if coverage >= test_criteria['thresholds']['pass']:
        result = "passed"
    elif coverage >= test_criteria['thresholds']['warning']:
        result = "warning"
    else:
        result = "failed"

    # 5. Return structured result
    return {
        "test_name": test_criteria['name'],
        "result": result,
        "score": score,
        "details": {
            "coverage": coverage,
            "found": found,
            "total": len(test_criteria['required_elements'])
        }
    }
```

### Built-in Tests

#### Test 1: Requirements Coverage

**Purpose**: Verify that all specified requirements are addressed

**Algorithm**:
```python
def test_requirements_coverage(content, requirements):
    """
    Scores how well the content covers given requirements.

    Scoring:
    - For each requirement, extract keywords (length > 3 chars)
    - Check if any keyword appears in content (case-insensitive)
    - Coverage = (covered requirements) / (total requirements)
    - Score = coverage × 10

    Thresholds:
    - Passed: coverage ≥ 0.80 (80%)
    - Warning: coverage ≥ 0.60 (60%)
    - Failed: coverage < 0.60
    """

    covered = 0
    for req in requirements:
        # Extract meaningful keywords
        keywords = [
            kw.lower()
            for kw in req.split()
            if len(kw) > 3
        ]

        # Check if any keyword present
        if any(kw in content.lower() for kw in keywords):
            covered += 1

    # Calculate score
    coverage = covered / len(requirements)
    score = coverage * 10

    # Determine result
    if coverage >= 0.80:
        result = "passed"
    elif coverage >= 0.60:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Requirements coverage: {coverage:.1%}",
        "details": {
            "covered": covered,
            "total": len(requirements),
            "coverage_percentage": coverage * 100
        }
    }
```

**Example**:
```python
# Input
requirements = ["JWT authentication", "bcrypt hashing", "rate limiting"]
content = "Implement JWT auth with bcrypt password hashing"

# Execution
# Req 1: "JWT" found → covered
# Req 2: "bcrypt" found → covered
# Req 3: "rate limiting" not found → not covered
# Coverage: 2/3 = 0.667 (66.7%)
# Score: 6.67/10
# Result: "warning" (66.7% ≥ 60% but < 80%)
```

#### Test 2: Edge Case Handling

**Purpose**: Evaluate consideration of edge cases and exceptional scenarios

**Algorithm**:
```python
def test_edge_case_handling(content):
    """
    Scores based on mentions of edge case keywords.

    Scoring:
    - Define list of edge case indicators
    - Count how many appear in content
    - Score = (mentions / total_keywords) × 10
    - Minimum score: 0, Maximum: 10

    Edge Case Keywords:
    - null, none, empty, zero, invalid
    - error, exception, timeout, boundary
    - missing, undefined, nan, infinity

    Thresholds:
    - Passed: ≥ 4 mentions
    - Warning: ≥ 2 mentions
    - Failed: < 2 mentions
    """

    edge_case_keywords = [
        "null", "none", "empty", "zero", "invalid",
        "error", "exception", "timeout", "boundary",
        "missing", "undefined", "nan", "infinity"
    ]

    # Count occurrences
    mentions = sum(
        1 for kw in edge_case_keywords
        if kw in content.lower()
    )

    # Calculate score
    score = min(10.0, (mentions / len(edge_case_keywords)) * 10)

    # Determine result
    if mentions >= 4:
        result = "passed"
    elif mentions >= 2:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Edge cases mentioned: {mentions}",
        "details": {
            "mentions": mentions,
            "total_keywords": len(edge_case_keywords)
        }
    }
```

**Example**:
```python
# Input
content = """
Handle null values and empty strings.
Catch timeout errors.
Check for invalid input.
Validate boundary conditions.
"""

# Execution
# Found: "null", "empty", "timeout", "invalid", "boundary"
# Mentions: 5
# Score: (5/14) × 10 = 3.57/10
# Result: "passed" (5 ≥ 4)
```

#### Test 3: Security Keywords

**Purpose**: Assess security considerations

**Algorithm**:
```python
def test_security_keywords(content):
    """
    Scores based on security-related terminology.

    Scoring:
    - Count security keyword mentions
    - Score increases with mentions (logarithmic scale)
    - Bonus for critical security terms

    Security Keywords:
    Critical (weight 2):
    - authentication, authorization, encryption, injection
    - vulnerability, exploit, breach, attack

    Important (weight 1.5):
    - validate, sanitize, hash, salt, encrypt
    - token, session, cookie, https, tls

    Standard (weight 1):
    - security, secure, protection, defense, firewall

    Calculation:
    score = Σ(weight × mentions) / max_possible × 10

    Thresholds:
    - Passed: weighted score ≥ 2
    - Warning: weighted score ≥ 1
    - Failed: weighted score < 1
    """

    security_terms = {
        "critical": {
            "weight": 2.0,
            "keywords": ["authentication", "authorization", "encryption",
                        "injection", "vulnerability", "exploit",
                        "breach", "attack"]
        },
        "important": {
            "weight": 1.5,
            "keywords": ["validate", "sanitize", "hash", "salt",
                        "encrypt", "token", "session", "cookie",
                        "https", "tls"]
        },
        "standard": {
            "weight": 1.0,
            "keywords": ["security", "secure", "protection",
                        "defense", "firewall"]
        }
    }

    # Calculate weighted score
    weighted_mentions = 0
    max_possible = 0

    for category, config in security_terms.items():
        for keyword in config["keywords"]:
            if keyword in content.lower():
                weighted_mentions += config["weight"]
            max_possible += config["weight"]

    # Normalize to 0-10
    score = (weighted_mentions / max_possible) * 10 if max_possible > 0 else 0

    # Determine result
    if weighted_mentions >= 2:
        result = "passed"
    elif weighted_mentions >= 1:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Security score: {weighted_mentions:.1f}",
        "details": {
            "weighted_mentions": weighted_mentions,
            "max_possible": max_possible
        }
    }
```

#### Test 4: Time Complexity Analysis

**Purpose**: Evaluate consideration of algorithmic efficiency

**Algorithm**:
```python
def test_time_complexity(content):
    """
    Scores based on complexity discussion and analysis.

    Scoring:
    - Check for Big-O notation mentions (O(n), O(log n), etc.)
    - Look for complexity-related keywords
    - Bonus for actual complexity analysis

    Keywords:
    - o(n), o(log n), o(1), o(n²), o(n log n)
    - complexity, efficient, optimize, scale
    - time, space, algorithm, performance

    Scoring:
    - Big-O notation present: +3 points
    - Complexity keyword present: +1 point each (max 5)
    - Optimization mentioned: +2 points

    Thresholds:
    - Passed: score ≥ 6
    - Warning: score ≥ 3
    - Failed: score < 3
    """

    score = 0

    # Check for Big-O notation
    big_o_patterns = [r"o\([^)]+\)", r"O\([^)]+\)", r"Θ\([^)]+\)"]
    for pattern in big_o_patterns:
        if re.search(pattern, content):
            score += 3
            break

    # Check for complexity keywords
    complexity_keywords = ["complexity", "efficient", "optimize",
                          "scale", "time", "space", "algorithm",
                          "performance"]

    mentions = sum(1 for kw in complexity_keywords if kw in content.lower())
    score += min(5, mentions)  # Max 5 points from keywords

    # Check for optimization
    if "optimize" in content.lower() or "optimization" in content.lower():
        score += 2

    # Normalize to 0-10
    score = min(10.0, score)

    # Determine result
    if score >= 6:
        result = "passed"
    elif score >= 3:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Complexity analysis score: {score}/10",
        "details": {}
    }
```

#### Test 5: Input Validation

**Purpose**: Assess input validation considerations

**Algorithm**:
```python
def test_input_validation(content):
    """
    Scores based on input validation practices.

    Scoring:
    - Count validation-related keywords
    - Check for specific validation patterns
    - Score based on completeness

    Keywords:
    - validate, validation, check, verify
    - sanitize, filter, clean, escape
    - input, parameter, argument, user data

    Patterns:
    - "validate.*input"
    - "sanitize.*data"
    - "check.*null"
    - "verify.*format"

    Scoring:
    - Keyword mentions: 0.5 points each (max 3)
    - Pattern matches: 1 point each (max 4)
    - Total: max 7, scaled to 0-10

    Thresholds:
    - Passed: score ≥ 5
    - Warning: score ≥ 2
    - Failed: score < 2
    """

    score = 0

    # Keyword mentions
    validation_keywords = ["validate", "validation", "check", "verify",
                         "sanitize", "filter", "clean", "escape",
                         "input", "parameter", "argument"]

    mentions = sum(1 for kw in validation_keywords if kw in content.lower())
    score += min(3, mentions * 0.5)

    # Pattern matching
    validation_patterns = [
        r"validate.{0,20}\binput\b",
        r"sanitize.{0,20}\bdata\b",
        r"check.{0,10}\bnull\b",
        r"verify.{0,20}\bformat\b"
    ]

    for pattern in validation_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            score += 1

    # Normalize to 0-10
    score = min(10.0, score * 10 / 7)

    # Determine result
    if score >= 5:
        result = "passed"
    elif score >= 2:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Input validation score: {score:.1f}/10",
        "details": {}
    }
```

#### Test 6: Error Handling

**Purpose**: Evaluate error handling and exception management

**Algorithm**:
```python
def test_error_handling(content):
    """
    Scores based on error handling practices.

    Scoring:
    - Count error-handling related keywords
    - Check for try-catch patterns
    - Look for specific error scenarios

    Keywords:
    - try, catch, except, error, exception
    - handle, recover, retry, fallback
    - throw, raise, finally, ensure

    Programming language patterns:
    - try\s*{.*catch
    - try\s*:\s*\n.*except
    - if.*error.*catch

    Scoring:
    - Keywords: 0.3 points each (max 3)
    - Try-catch patterns: 2 points each (max 6)
    - Error scenarios mentioned: 1 point each (max 3)
    - Total: max 12, scaled to 0-10

    Thresholds:
    - Passed: score ≥ 6
    - Warning: score ≥ 3
    - Failed: score < 3
    """

    score = 0

    # Keywords
    error_keywords = ["try", "catch", "except", "error", "exception",
                     "handle", "recover", "retry", "fallback",
                     "throw", "raise", "finally", "ensure"]

    mentions = sum(1 for kw in error_keywords if kw in content.lower())
    score += min(3, mentions * 0.3)

    # Try-catch patterns
    try_catch_patterns = [
        r"try\s*{",
        r"try\s*:\s*\n.*except",
        r"\.catch\(",
        r"catch\s*\("
    ]

    pattern_matches = sum(
        1 for pattern in try_catch_patterns
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
    )
    score += min(6, pattern_matches * 2)

    # Error scenarios
    error_scenarios = ["timeout", "failure", "invalid", "missing", "null"]
    scenario_mentions = sum(1 for es in error_scenarios if es in content.lower())
    score += min(3, scenario_mentions)

    # Normalize to 0-10
    score = min(10.0, score * 10 / 12)

    # Determine result
    if score >= 6:
        result = "passed"
    elif score >= 3:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Error handling score: {score:.1f}/10",
        "details": {}
    }
```

#### Test 7: Fault Tolerance

**Purpose**: Assess fault tolerance and resilience mechanisms

**Algorithm**:
```python
def test_fault_tolerance(content):
    """
    Scores based on fault tolerance considerations.

    Scoring:
    - Check for fault tolerance keywords and patterns
    - Look for redundancy, fallback, recovery mechanisms

    Keywords:
    - fallback, backup, redundant, redundancy
    - recover, recovery, restore, rollback
    - tolerate, tolerance, resilient, resilience
    - failover, replica, cluster, distributed

    Scoring:
    - Each unique keyword: 1 point
    - Bonus for related concepts: +0.5 points
    - Max score: 10

    Thresholds:
    - Passed: score ≥ 5
    - Warning: score ≥ 2
    - Failed: score < 2
    """

    ft_keywords = [
        "fallback", "backup", "redundant", "redundancy",
        "recover", "recovery", "restore", "rollback",
        "tolerate", "tolerance", "resilient", "resilience",
        "failover", "replica", "cluster", "distributed"
    ]

    # Count unique mentions
    unique_mentions = len(set(
        kw for kw in ft_keywords
        if kw in content.lower()
    ))

    # Base score
    score = unique_mentions

    # Bonus for related concepts
    if "redundant" in content.lower() and "backup" in content.lower():
        score += 0.5

    if "fallback" in content.lower() and "recover" in content.lower():
        score += 0.5

    # Normalize to 0-10
    score = min(10.0, score)

    # Determine result
    if score >= 5:
        result = "passed"
    elif score >= 2:
        result = "warning"
    else:
        result = "failed"

    return {
        "result": result,
        "score": score,
        "message": f"Fault tolerance: {unique_mentions} concepts mentioned",
        "details": {
            "unique_concepts": unique_mentions
        }
    }
```

### Summary of Explicit Tests

| Test | Purpose | Scoring Method | Max Score | Pass Threshold |
|------|---------|---------------|-----------|----------------|
| Requirements Coverage | Verify requirement addressing | (covered/total) × 10 | 10 | 80% coverage |
| Edge Case Handling | Check edge case consideration | (mentions/total) × 10 | 10 | 4+ mentions |
| Security Keywords | Assess security awareness | Weighted keyword count | 10 | Score ≥ 2 |
| Time Complexity | Evaluate efficiency analysis | Keyword + pattern count | 10 | Score ≥ 6 |
| Input Validation | Check input validation practices | Keyword + pattern count | 10 | Score ≥ 5 |
| Error Handling | Assess error handling | Pattern + keyword count | 10 | Score ≥ 6 |
| Fault Tolerance | Evaluate fault tolerance | Unique concept count | 10 | Score ≥ 5 |

---

## 2. Keyword Frequency Analysis

### Purpose

Quantitative analysis of term occurrences to measure focus and completeness.

### Mechanism

```python
def analyze_keyword_frequency(content, keyword_categories):
    """
    Analyzes frequency of keywords in different categories.

    Args:
        content: Text to analyze
        keyword_categories: Dict mapping category_name to list of keywords

    Returns:
        {
            category_name: {
                "count": int,
                "frequency": float,  # per 1000 words
                "score": float      # 0-10
            }
        }
    """

    word_count = len(content.split())
    results = {}

    for category, keywords in keyword_categories.items():
        # Count occurrences
        count = sum(
            1 for kw in keywords
            if kw.lower() in content.lower()
        )

        # Calculate frequency per 1000 words
        frequency = (count / word_count) * 1000 if word_count > 0 else 0

        # Score based on expected frequency ranges
        # (e.g., for security: expect 2-5 mentions per 1000 words)
        score = calculate_frequency_score(category, frequency)

        results[category] = {
            "count": count,
            "frequency": frequency,
            "score": score
        }

    return results


def calculate_frequency_score(category, frequency):
    """
    Converts keyword frequency to 0-10 score.

    Different categories have different optimal frequencies.

    Security: High frequency good (5+ per 1000 words → 10/10)
    Performance: Medium frequency good (2-4 per 1000 words → 8-10/10)
    Error handling: Medium-high frequency good (3-5 per 1000 words → 8-10/10)
    """

    # Category-specific scoring curves
    scoring_curves = {
        "security": lambda f: min(10, f * 2),           # Linear: 5/1000 → 10/10
        "performance": lambda f: 10 - abs(f - 3) * 2,   # Peak at 3/1000
        "error_handling": lambda f: min(10, f * 2.5),    # Linear: 4/1000 → 10/10
        "documentation": lambda f: min(10, f * 1.5),    # Linear: 6.7/1000 → 10/10
    }

    scoring_function = scoring_curves.get(category, lambda f: min(10, f))

    return max(0, scoring_function(frequency))
```

### Example

```python
content = """
Implement secure authentication using JWT tokens.
Validate all inputs and sanitize user data.
Handle errors gracefully with try-catch blocks.
Monitor performance metrics and optimize bottlenecks.
Document all functions with clear comments.
"""

# Analysis
word_count = 32

# Security keywords: ["authentication", "jwt", "secure", "validate", "sanitize"]
security_count = 5
security_frequency = (5 / 32) * 1000 = 156.25 per 1000 words
security_score = min(10, 156.25 * 0.05) = 7.81/10  # Scaled down

# Performance keywords: ["performance", "optimize"]
performance_count = 2
performance_frequency = (2 / 32) * 1000 = 62.5 per 1000 words
performance_score = 10 - abs(62.5 - 3) * 2 = Negative → 0/10  # Too infrequent
```

---

## 3. Pattern Matching

### Purpose

Extract structured information using regular expressions and natural language patterns.

### Mechanism

```python
class PatternExtractor:
    """
    Extracts entities using regex patterns and natural language patterns.
    """

    # Predefined patterns for different entity types
    PATTERNS = {
        "best_practice": [
            r"best practice:\s*([^.\n]+)",
            r"recommended:\s*([^.\n]+)",
            r"should\s+(.+?)(?:\.|\n)",
            r"ideally\s+(.+?)(?:\.|\n)"
        ],

        "anti_pattern": [
            r"avoid:\s*([^.\n]+)",
            r"should\s+not\s+(.+?)(?:\.|\n)",
            r"don't\s+(.+?)(?:\.|\n)",
            r"never:\s*([^.\n]+)"
        ],

        "lesson_learned": [
            r"lesson\s+learned:\s*([^.\n]+)",
            r"learned\s+that\s+(.+?)(?:\.|\n)",
            r"found\s+that\s+(.+?)(?:\.|\n)"
        ],

        "requirement": [
            r"requirement:\s*([^.\n]+)",
            r"must\s+(.+?)(?:\.|\n)",
            r"shall\s+(.+?)(?:\.|\n)",
            r"required:\s*([^.\n]+)"
        ]
    }

    def extract_patterns(self, content, min_confidence=0.3):
        """
        Extract entities using all defined patterns.

        Args:
            content: Text to extract from
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of (entity_type, content, confidence) tuples
        """

        extracted = []

        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                # Find all matches
                matches = re.finditer(pattern, content, re.IGNORECASE)

                for match in matches:
                    entity_content = match.group(1).strip()

                    # Filter: Minimum length check
                    if len(entity_content) < 10:
                        continue

                    # Calculate confidence
                    confidence = self.calculate_pattern_confidence(
                        match,
                        content,
                        entity_content
                    )

                    # Filter by confidence threshold
                    if confidence >= min_confidence:
                        extracted.append({
                            "type": entity_type,
                            "content": entity_content,
                            "confidence": confidence
                        })

        # Deduplicate entities
        deduplicated = self.deduplicate(extracted)

        return deduplicated

    def calculate_pattern_confidence(self, match, content, entity_content):
        """
        Calculate confidence score for a pattern match (0-1).

        Confidence factors:
        1. Base confidence: 0.5
        2. Length bonus: +0.1 for >100 chars, +0.1 for >300 chars
        3. Technical terms bonus: +0.15 for technical indicators nearby
        4. Completeness bonus: +0.1 for sentence-ending punctuation
        """

        confidence = 0.5

        # Factor 1: Content length
        if len(entity_content) > 100:
            confidence += 0.1
        if len(entity_content) > 300:
            confidence += 0.1

        # Factor 2: Context check for technical indicators
        context_start = max(0, match.start() - 50)
        context_end = min(len(content), match.end() + 50)
        context = content[context_start:context_end].lower()

        technical_terms = [
            "implement", "architecture", "design", "pattern",
            "system", "component", "service", "api",
            "function", "method", "class", "algorithm",
            "database", "server", "client", "protocol"
        ]

        if any(term in context for term in technical_terms):
            confidence += 0.15

        # Factor 3: Completeness check
        if any(punct in context for punct in [".", "!", ";", ":"]):
            confidence += 0.1

        # Cap at 1.0
        return min(1.0, confidence)

    def deduplicate(self, extracted):
        """
        Remove duplicate entities based on type and content similarity.
        """

        seen = set()
        deduplicated = []

        for entity in extracted:
            # Create signature from type and first 100 chars of content
            signature = (
                entity["type"],
                entity["content"].lower()[:100]
            )

            if signature not in seen:
                seen.add(signature)
                deduplicated.append(entity)

        return deduplicated
```

### Example

```python
content = """
Best practice: Use connection pooling for database access.
Avoid: Creating new connections for each query.
Lesson learned: Connection pools significantly improve performance.
"""

# Extraction
extracted = [
    {
        "type": "best_practice",
        "content": "Use connection pooling for database access",
        "confidence": 0.75  # 0.5 + 0.1(length>50) + 0.15(technical: "database")
    },
    {
        "type": "anti_pattern",
        "content": "Creating new connections for each query",
        "confidence": 0.75  # 0.5 + 0.1(length>50) + 0.15(technical: "connections")
    },
    {
        "type": "lesson_learned",
        "content": "Connection pools significantly improve performance",
        "confidence": 0.85  # 0.5 + 0.1(length>50) + 0.15(technical: "performance") + 0.1(completeness: ".")
    }
]
```

---

## 4. LLM-Based Evaluation

### Purpose

Use AI/LLM to make nuanced judgments about quality, completeness, and other subjective criteria.

### Mechanism

```python
class LLMScorer:
    """
    Uses Large Language Model to evaluate artifacts.
    """

    async def score_with_llm(self, artifact_content, criteria, hephaestus_client):
        """
        Score artifact using LLM evaluation.

        Args:
            artifact_content: The artifact to score
            criteria: Scoring criteria and instructions
            hephaestus_client: LLM client

        Returns:
            {
                "overall_score": float (0-10),
                "dimension_scores": dict,
                "reasoning": str,
                "confidence": float (0-1)
            }
        """

        # Construct evaluation prompt
        prompt = self._build_evaluation_prompt(artifact_content, criteria)

        # Call LLM
        response = await hephaestus_client.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=1000
        )

        # Parse response
        scores = self._parse_llm_response(response["text"])

        return scores

    def _build_evaluation_prompt(self, content, criteria):
        """
        Constructs a detailed prompt for LLM evaluation.
        """

        prompt = f"""Evaluate the following artifact based on the given criteria.

ARTIFACT:
{content[:3000]}  # First 3000 chars

EVALUATION CRITERIA:
{self._format_criteria(criteria)}

SCORING INSTRUCTIONS:
1. Evaluate each dimension on a scale of 0-10
2. Provide an overall score (weighted average of dimensions)
3. Explain your reasoning for each score
4. Indicate your confidence in each score (0-1)

OUTPUT FORMAT (JSON):
{{
    "overall_score": 8.5,
    "dimension_scores": {{
        "functionality": {{"score": 9.0, "reasoning": "...", "confidence": 0.9}},
        "completeness": {{"score": 8.0, "reasoning": "...", "confidence": 0.85}},
        "quality": {{"score": 8.5, "reasoning": "...", "confidence": 0.8}}
    }},
    "overall_reasoning": "The artifact demonstrates...",
    "confidence": 0.85
}}

Please provide your evaluation:"""

        return prompt

    def _format_criteria(self, criteria):
        """
        Format evaluation criteria for the prompt.
        """

        formatted = []

        for dimension, definition in criteria.items():
            formatted.append(f"""
{dimension.upper()}:
- Definition: {definition['description']}
- Scoring Guidelines: {definition['guidelines']}
- What to look for: {definition['indicators']}
""")

        return "\n".join(formatted)

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response to extract scores.

        Handles various response formats and provides fallbacks.
        """

        # Try to extract JSON
        import json

        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response_text)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data
            except json.JSONDecodeError:
                pass

        # Fallback: Parse text response
        return self._parse_text_response(response_text)

    def _parse_text_response(self, text):
        """
        Parse non-JSON text response.
        Extracts scores using regex patterns.
        """

        result = {
            "overall_score": None,
            "dimension_scores": {},
            "overall_reasoning": text[:500]
        }

        # Extract overall score
        score_match = re.search(r'overall\s+score[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            result["overall_score"] = float(score_match.group(1))

        # Extract dimension scores
        dimension_pattern = r'(\w+)\s+score[:\s]+(\d+(?:\.\d+)?)'
        for match in re.finditer(dimension_pattern, text, re.IGNORECASE):
            dimension = match.group(1).lower()
            score = float(match.group(2))
            result["dimension_scores"][dimension] = {
                "score": score,
                "reasoning": "Extracted from text",
                "confidence": 0.7  # Lower confidence for text extraction
            }

        # If no scores found, use sentiment analysis
        if result["overall_score"] is None:
            result["overall_score"] = self._sentiment_to_score(text)

        return result

    def _sentiment_to_score(self, text):
        """
        Convert sentiment of text response to score (0-10).

        Positive words → higher scores
        Negative words → lower scores
        """

        positive_words = ["excellent", "good", "great", "strong", "comprehensive",
                        "thorough", "well", "effective", "complete", "clear"]

        negative_words = ["poor", "weak", "incomplete", "unclear", "missing",
                        "lacking", "inadequate", "insufficient", "problematic"]

        # Count positive and negative indicators
        positive_count = sum(1 for pw in positive_words if pw in text.lower())
        negative_count = sum(1 for nw in negative_words if nw in text.lower())

        # Calculate net score
        net = positive_count - negative_count

        # Convert to 0-10 scale
        # Base: 5.0
        # Each positive: +0.5
        # Each negative: -0.5
        score = 5.0 + (net * 0.5)

        return max(0, min(10, score))
```

### Example

```python
# Input
artifact = {
    "content": "Implement REST API with JWT authentication...",
    "type": "solution"
}

# Criteria
criteria = {
    "functionality": {
        "description": "Does the solution address the problem?",
        "guidelines": "Score based on how well the solution meets requirements",
        "indicators": ["Clear implementation", "Addresses all requirements", "Executable"]
    },
    "security": {
        "description": "Are security concerns addressed?",
        "guidelines": "Score based on security considerations",
        "indicators": ["Input validation", "Encryption", "Authentication", "Authorization"]
    }
}

# LLM Evaluation
result = await llm_scorer.score_with_llm(
    artifact["content"],
    criteria,
    hephaestus_client
)

# Result
{
    "overall_score": 8.2,
    "dimension_scores": {
        "functionality": {
            "score": 8.5,
            "reasoning": "Clear implementation plan with REST API structure. All requirements addressed.",
            "confidence": 0.9
        },
        "security": {
            "score": 8.0,
            "reasoning": "JWT authentication mentioned. Input validation could be more explicit.",
            "confidence": 0.85
        }
    },
    "overall_reasoning": "The artifact demonstrates a solid understanding of REST API design with appropriate security measures...",
    "confidence": 0.87
}
```

---

## 5. Comparative Analysis (Historical Comparison)

### Purpose

Compare artifacts against historical data to determine relative performance and rank.

### Mechanism

```python
class HistoricalComparator:
    """
    Compares artifacts against historical data.
    """

    async def compare_artifact(self, artifact_id, artifact_type, metrics_collector, metrics_analyzer):
        """
        Compare artifact against historical artifacts of same type.

        Returns:
            {
                "current_score": float,
                "historical_scores": list,
                "percentile_rank": float (0-100),
                "above_average": bool,
                "rank": int,
                "total_compared": int,
                "summary": str
            }
        """

        # 1. Get current artifact's score
        current_report = await metrics_analyzer.analyze_artifact(artifact_id)
        current_score = current_report.overall_score

        # 2. Get historical scores
        historical_metrics = await metrics_collector.get_historical_metrics(
            artifact_type=artifact_type,
            limit=100
        )

        historical_scores = []
        for hm in historical_metrics:
            if hm.artifact_id == artifact_id:
                continue  # Skip self

            report = await metrics_analyzer.analyze_artifact(hm.artifact_id)
            if report:
                historical_scores.append(report.overall_score)

        if not historical_scores:
            return {
                "current_score": current_score,
                "historical_scores": [],
                "percentile_rank": None,
                "message": "No historical data available"
            }

        # 3. Calculate percentile rank
        all_scores = sorted(historical_scores + [current_score])
        rank = all_scores.index(current_score)
        percentile = (rank / len(all_scores)) * 100

        # 4. Calculate statistics
        average_historical = sum(historical_scores) / len(historical_scores)
        median_historical = sorted(historical_scores)[len(historical_scores) // 2]
        std_dev = calculate_standard_deviation(historical_scores)

        # 5. Determine standing
        above_average = current_score > average_historical

        # 6. Generate summary
        if percentile >= 90:
            summary = f"Top 10% - Excellent performance"
        elif percentile >= 75:
            summary = f"Top quartile - Above average"
        elif percentile >= 50:
            summary = f"Above median - Good performance"
        elif percentile >= 25:
            summary = f"Below median but above bottom quartile"
        else:
            summary = f"Bottom quartile - Needs improvement"

        return {
            "current_score": current_score,
            "historical_scores": historical_scores,
            "percentile_rank": percentile,
            "rank": rank + 1,  # 1-indexed
            "total_compared": len(all_scores),
            "above_average": above_average,
            "average_historical": average_historical,
            "median_historical": median_historical,
            "std_dev": std_dev,
            "z_score": (current_score - average_historical) / std_dev if std_dev > 0 else 0,
            "summary": summary
        }
```

### Example

```python
# Current artifact
current_score = 0.82

# Historical scores (same artifact type)
historical_scores = [0.65, 0.72, 0.78, 0.80, 0.85, 0.88, 0.90, 0.91, 0.93]

# Analysis
all_scores = sorted([0.65, 0.72, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.93])
#            [0.65, 0.72, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.93]
# Rank:                                  5th (0-indexed)
# Percentile: 5 / 10 = 50th percentile

average_historical = (0.65 + 0.72 + ... + 0.93) / 9 = 0.83
median_historical = 0.85
std_dev = 0.09

# Result
{
    "current_score": 0.82,
    "percentile_rank": 50.0,
    "rank": 5,
    "total_compared": 10,
    "above_average": False,  # 0.82 < 0.83
    "average_historical": 0.83,
    "median_historical": 0.85,
    "std_dev": 0.09,
    "z_score": (0.82 - 0.83) / 0.09 = -0.11,  # Slightly below average
    "summary": "Above median but below average - Typical performance"
}
```

### Z-Score Interpretation

| Z-Score Range | Interpretation | Percentile |
|---------------|----------------|------------|
| z > 2.0 | Far above average | > 97th |
| 1.0 < z ≤ 2.0 | Above average | 84th - 97th |
| 0.0 < z ≤ 1.0 | Slightly above average | 50th - 84th |
| -1.0 ≤ z ≤ 0.0 | Slightly below average | 16th - 50th |
| -2.0 ≤ z < -1.0 | Below average | 3rd - 16th |
| z < -2.0 | Far below average | < 3rd |

---

## 6. Heuristic Rules

### Purpose

Apply domain knowledge and experience-based rules to assess quality.

### Mechanism

```python
class HeuristicScorer:
    """
    Applies heuristic rules to score artifacts.
    """

    def calculate_quality_score(self, entity):
        """
        Calculate base quality score using heuristics.

        Starts with baseline (0.5) and adds bonuses/subtractions.
        Final score clamped to [0, 1].
        """

        score = 0.5  # Baseline

        # Rule 1: Length heuristics
        # Longer content = more detail (up to a point)
        if len(entity.content) > 100:
            score += 0.1  # Decent length
        if len(entity.content) > 300:
            score += 0.1  # Comprehensive
        if len(entity.content) > 1000:
            score += 0.05  # Very detailed

        # Rule 2: Confidence from extraction method
        # Higher confidence extraction = better quality
        score += (entity.confidence - 0.5) * 0.3

        # Rule 3: Tag count
        # More tags = better categorization
        if len(entity.tags) >= 1:
            score += 0.05
        if len(entity.tags) >= 3:
            score += 0.05
        if len(entity.tags) >= 5:
            score += 0.05

        # Rule 4: Related entities
        # Connections = thoroughness
        if len(entity.related_entities) >= 1:
            score += 0.05
        if len(entity.related_entities) >= 3:
            score += 0.05

        # Rule 5: Technical depth
        # Presence of technical terms indicates depth
        technical_terms = [
            "implement", "design", "architecture", "algorithm",
            "system", "component", "service", "interface",
            "database", "api", "protocol", "standard"
        ]

        technical_mention_count = sum(
            1 for term in technical_terms
            if term in entity.content.lower()
        )

        if technical_mention_count >= 1:
            score += 0.05
        if technical_mention_count >= 3:
            score += 0.05

        # Rule 6: Structure indicators
        # Bullet points, numbered lists, sections = better organization
        if re.search(r'^\s*[-*•]\s', entity.content, re.MULTILINE):
            score += 0.05  # Has bullet points

        if re.search(r'^\s*\d+[.\)]\s', entity.content, re.MULTILINE):
            score += 0.05  # Has numbered list

        # Rule 7: Completeness indicators
        # End punctuation, conclusion markers
        if entity.content.strip().endswith(('.', '!', '?')):
            score += 0.05

        if any(marker in entity.content.lower() for marker in
               ['conclusion', 'summary', 'finally', 'complete']):
            score += 0.05

        # Rule 8: Clarity indicators
        # Clear language, avoid ambiguity
        # Check for vague words
        vague_words = ['maybe', 'possibly', 'might', 'could', 'sort of', 'kind of']
        vague_count = sum(1 for vw in vague_words if vw in entity.content.lower())

        if vague_count == 0:
            score += 0.1  # Clear and direct
        elif vague_count <= 2:
            score += 0.05
        # No penalty for vague words (might be intentional)

        # Clamp to valid range
        return max(0.0, min(1.0, score))
```

### Heuristic Rules for Different Domains

#### Software Code

```python
def code_quality_heuristics(code_entity):
    """
    Domain-specific heuristics for code quality.
    """

    score = 0.5

    # Functions/methods present
    if re.search(r'def\s+\w+\s*\(', code_entity.content):
        score += 0.1

    # Comments present
    if re.search(r'#.*', code_entity.content):
        comment_ratio = len(re.findall(r'#.*', code_entity.content)) / max(1, len(code_entity.content.split('\n')))
        if 0.1 <= comment_ratio <= 0.3:  # 10-30% comments
            score += 0.1

    # Error handling
    if any(kw in code_entity.content.lower() for kw in ['try', 'except', 'catch', 'error']):
        score += 0.1

    # Documentation
    if re.search(r'"""[\s\S]*?"""', code_entity.content):  # Docstrings
        score += 0.1

    # Type hints (Python)
    if re.search(r':\s*\w+', code_entity.content):
        score += 0.05

    return min(1.0, score)
```

#### Scientific Writing

```python
def scientific_writing_heuristics(document_entity):
    """
    Domain-specific heuristics for scientific documents.
    """

    score = 0.5

    # Citations/references
    if re.search(r'\[\d+\]', document_entity.content):
        citation_count = len(re.findall(r'\[\d+\]', document_entity.content))
        score += min(0.15, citation_count * 0.01)

    # Quantitative data
    if re.search(r'\d+\.?\d*\s*(?:%|degrees?|units?)', document_entity.content):
        score += 0.1

    # Scientific terminology
    science_terms = [
        'hypothesis', 'experiment', 'methodology', 'results',
        'analysis', 'conclusion', 'significant', 'correlation'
    ]

    term_count = sum(1 for term in science_terms if term in document_entity.content.lower())
    score += min(0.15, term_count * 0.03)

    # Passive voice (common in scientific writing, but excessive is bad)
    passive_indicators = ['was measured', 'were analyzed', 'was observed']
    passive_count = sum(1 for pi in passive_indicators if pi in document_entity.content.lower())

    if 0.02 <= passive_count / max(1, len(document_entity.content.split())) <= 0.15:
        score += 0.05  # Appropriate passive voice usage

    return min(1.0, score)
```

---

## 7. Statistical Analysis

### Purpose

Detect trends, patterns, and anomalies in scoring over time.

### Mechanism

```python
class StatisticalAnalyzer:
    """
    Performs statistical analysis on scoring data.
    """

    def calculate_trend(self, scores, timestamps=None):
        """
        Calculate trend direction and magnitude using linear regression.

        Args:
            scores: List of scores in chronological order
            timestamps: Optional list of timestamps (defaults to indices)

        Returns:
            {
                "direction": "improving" | "declining" | "stable" | "insufficient_data",
                "slope": float,
                "intercept": float,
                "r_squared": float,  # Goodness of fit
                "start_score": float,
                "end_score": float,
                "change": float,
                "change_percent": float
            }
        """

        if len(scores) < 2:
            return {
                "direction": "insufficient_data",
                "message": "Need at least 2 data points"
            }

        # Use indices or timestamps
        if timestamps is None:
            x_values = list(range(len(scores)))
        else:
            # Normalize timestamps to 0-N range
            min_time = min(timestamps)
            max_time = max(timestamps)
            if max_time == min_time:
                x_values = [0] * len(scores)
            else:
                x_values = [
                    (t - min_time) / (max_time - min_time) * len(scores)
                    for t in timestamps
                ]

        y_values = scores

        # Calculate linear regression: y = mx + b
        n = len(x_values)

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x ** 2 for x in x_values)

        # Calculate slope (m) and intercept (b)
        denominator = n * sum_x2 - sum_x ** 2

        if denominator == 0:
            # All x values are the same (horizontal line)
            slope = 0
            intercept = sum(y_values) / n
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

        # Calculate R² (coefficient of determination)
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2
                   for x, y in zip(x_values, y_values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"

        # Calculate change
        start_score = scores[0]
        end_score = scores[-1]
        change = end_score - start_score
        change_percent = (change / start_score * 100) if start_score != 0 else 0

        return {
            "direction": direction,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "start_score": start_score,
            "end_score": end_score,
            "change": change,
            "change_percent": change_percent,
            "data_points": len(scores)
        }

    def detect_anomalies(self, scores, threshold=2.0):
        """
        Detect anomalous scores using z-score method.

        Anomaly = score beyond threshold standard deviations from mean.

        Args:
            scores: List of scores
            threshold: Z-score threshold (default 2.0 = 2 standard deviations)

        Returns:
            {
                "anomalies": [
                    {
                        "index": int,
                        "score": float,
                        "z_score": float,
                        "type": "high" | "low"
                    }
                ],
                "mean": float,
                "std_dev": float
            }
        """

        if len(scores) < 3:
            return {
                "anomalies": [],
                "message": "Need at least 3 data points for anomaly detection"
            }

        # Calculate statistics
        mean = sum(scores) / len(scores)

        # Sample standard deviation
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return {
                "anomalies": [],
                "mean": mean,
                "std_dev": 0,
                "message": "All scores are identical"
            }

        # Find anomalies
        anomalies = []

        for i, score in enumerate(scores):
            z_score = (score - mean) / std_dev

            if abs(z_score) >= threshold:
                anomalies.append({
                    "index": i,
                    "score": score,
                    "z_score": z_score,
                    "type": "high" if z_score > 0 else "low"
                })

        return {
            "anomalies": anomalies,
            "mean": mean,
            "std_dev": std_dev,
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(scores)
        }

    def calculate_moving_average(self, scores, window_size=3):
        """
        Calculate moving average to smooth score trends.

        Args:
            scores: List of scores
            window_size: Number of points to average (default 3)

        Returns:
            List of moving averages (length = len(scores) - window_size + 1)
        """

        if len(scores) < window_size:
            return []

        moving_avgs = []

        for i in range(len(scores) - window_size + 1):
            window = scores[i:i + window_size]
            avg = sum(window) / window_size
            moving_avgs.append(avg)

        return moving_avgs
```

### Example: Trend Analysis

```python
# Scores over time
scores = [0.65, 0.68, 0.72, 0.75, 0.78, 0.82]
timestamps = [1, 2, 3, 4, 5, 6]  # Could be actual Unix timestamps

# Calculate trend
trend = statistical_analyzer.calculate_trend(scores, timestamps)

# Result
{
    "direction": "improving",
    "slope": 0.034,          # Score increases by 0.034 per time unit
    "intercept": 0.617,      # Starting point (y-intercept)
    "r_squared": 0.98,       # Excellent linear fit
    "start_score": 0.65,
    "end_score": 0.82,
    "change": 0.17,
    "change_percent": 26.15,  # 26.15% improvement
    "data_points": 6
}

# Interpretation
# - Direction: "improving" (slope > 0.01)
# - Strength: Strong trend (R² = 0.98 close to 1.0)
# - Magnitude: 26.15% improvement over the period
```

### Example: Anomaly Detection

```python
# Scores with one anomaly
scores = [0.72, 0.75, 0.73, 0.76, 0.35, 0.78, 0.77]
#                                            ↑
#                                         Anomaly (unusually low)

# Detect anomalies
anomalies = statistical_analyzer.detect_anomalies(scores, threshold=2.0)

# Result
{
    "anomalies": [
        {
            "index": 4,
            "score": 0.35,
            "z_score": -3.42,  # 3.42 standard deviations below mean
            "type": "low"
        }
    ],
    "mean": 0.723,
    "std_dev": 0.109,
    "anomaly_count": 1,
    "anomaly_rate": 0.143  # 14.3% of scores are anomalous
}
```

---

## Multi-Dimensional Scoring

### Overview

Artifacts are scored across **8 dimensions** to provide comprehensive evaluation:

| Dimension | Description | Weight | Key Indicators |
|-----------|-------------|--------|----------------|
| **Functionality** | Solves the core problem | 1.2 | Requirements met, working solution |
| **Performance** | Efficient resource usage | 0.9 | Speed, optimization, complexity |
| **Security** | Security considerations | 1.3 | Vulnerability prevention, validation |
| **Reliability** | Dependability and robustness | 1.1 | Error handling, fault tolerance |
| **Completeness** | Coverage of all aspects | 1.0 | Features, edge cases, documentation |
| **Efficiency** | Optimal use of resources | 0.9 | No waste, streamlined approach |
| **Maintainability** | Ease of maintenance | 0.8 | Readability, modularity, documentation |
| **Scalability** | Growth capability | 0.8 | Load handling, extensibility |

### Dimension Score Calculation

Each dimension is calculated independently (0-10 scale):

```python
def calculate_dimension_score(dimension_name, test_results, metrics):
    """
    Calculate score for a single dimension.

    Combines:
    1. Gauntlet test results (if any)
    2. Related metrics
    3. Heuristic rules

    Returns:
        float in range [0, 10]
    """

    # Get relevant tests for this dimension
    dimension_tests = get_tests_for_dimension(dimension_name, test_results)

    # Get relevant metrics for this dimension
    dimension_metrics = get_metrics_for_dimension(dimension_name, metrics)

    # Calculate score from tests
    if dimension_tests:
        test_score = calculate_test_score(dimension_tests)
    else:
        test_score = None

    # Calculate score from metrics
    if dimension_metrics:
        metric_score = calculate_metric_score(dimension_metrics)
    else:
        metric_score = None

    # Combine scores
    if test_score is not None and metric_score is not None:
        # Both available: weighted average
        combined = (test_score * 0.6 + metric_score * 0.4)
    elif test_score is not None:
        combined = test_score
    elif metric_score is not None:
        combined = metric_score
    else:
        # No tests or metrics: use heuristic
        combined = calculate_heuristic_score(dimension_name, test_results, metrics)

    # Apply dimension-specific adjustments
    combined = apply_dimension_adjustments(dimension_name, combined, test_results, metrics)

    # Clamp to [0, 10]
    return max(0.0, min(10.0, combined))


def calculate_test_score(test_results):
    """
    Calculate score from test results.

    Averages test scores, but if any test fails significantly,
    applies a penalty.
    """

    if not test_results:
        return None

    scores = [t["score"] for t in test_results]

    # Calculate average
    avg_score = sum(scores) / len(scores)

    # Check for failures
    failed_tests = [t for t in test_results if t["result"] == "failed"]

    if failed_tests:
        # Penalty for failures
        penalty = len(failed_tests) * 1.0
        avg_score = max(0, avg_score - penalty)

    return avg_score


def calculate_metric_score(metrics):
    """
    Calculate score from metrics.

    Normalizes metric values to [0, 10] range.
    """

    if not metrics:
        return None

    normalized_values = []

    for metric in metrics:
        value = metric["value"]
        min_val = metric.get("min_value")
        max_val = metric.get("max_value")

        # Normalize
        if min_val is not None and max_val is not None and max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        elif value <= 1.0:
            # Already normalized
            normalized = value
        elif value <= 100.0:
            # Percentage
            normalized = value / 100.0
        else:
            # Use sigmoid for unbounded
            normalized = 1 / (1 + math.exp(-0.01 * (value - 50)))

        normalized_values.append(normalized)

    # Average and scale to [0, 10]
    avg_normalized = sum(normalized_values) / len(normalized_values)
    return avg_normalized * 10
```

### Overall Score Calculation

```python
def calculate_overall_score(dimension_scores):
    """
    Calculate weighted overall score from dimensions.

    Args:
        dimension_scores: Dict mapping dimension_name to score (0-10)

    Returns:
        float in range [0, 10]
    """

    # Dimension weights
    weights = {
        "functionality": 1.2,
        "performance": 0.9,
        "security": 1.3,
        "reliability": 1.1,
        "completeness": 1.0,
        "efficiency": 0.9,
        "maintainability": 0.8,
        "scalability": 0.8
    }

    # Calculate weighted sum
    weighted_sum = sum(
        score * weights.get(dim, 1.0)
        for dim, score in dimension_scores.items()
    )

    # Calculate total weight
    total_weight = sum(
        weights.get(dim, 1.0)
        for dim in dimension_scores.keys()
    )

    # Calculate weighted average
    overall = weighted_sum / total_weight if total_weight > 0 else 0

    return overall
```

### Verdict Determination

```python
def determine_verdict(multi_dimensional_score):
    """
    Determine verdict based on dimension scores.

    Verdict criteria:
    - EXCELLENT: Overall ≥ 8.0 AND no critical dimensions
    - GOOD: Overall ≥ 6.5 AND ≤ 1 critical dimension
    - ACCEPTABLE: Overall ≥ 5.0
    - POOR: Overall < 5.0

    Critical dimension: Score < 5.0/10
    """

    overall = multi_dimensional_score.overall_score
    critical_dims = multi_dimensional_score.critical_dimensions

    # Count critical dimensions
    critical_count = len(critical_dims)

    # Determine verdict
    if overall >= 8.0 and critical_count == 0:
        verdict = "EXCELLENT"
    elif overall >= 6.5 and critical_count <= 1:
        verdict = "GOOD"
    elif overall >= 5.0:
        verdict = "ACCEPTABLE"
    else:
        verdict = "POOR"

    return verdict
```

---

## Metrics Collection and Analysis

### Metric Categories

Metrics are organized into **8 categories**:

```python
class MetricCategory(Enum):
    """Categories of evaluation metrics"""

    QUALITY = "quality"              # Solution quality metrics
    PERFORMANCE = "performance"      # Performance metrics
    RELIABILITY = "reliability"      # Reliability metrics
    SECURITY = "security"            # Security metrics
    COMPLETENESS = "completeness"    # Completeness metrics
    EFFICIENCY = "efficiency"        # Efficiency metrics
    MAINTAINABILITY = "maintainability"  # Maintainability metrics
    SCALABILITY = "scalability"      # Scalability metrics
```

### Metric Types

Each category has multiple specific metrics:

```python
class MetricType(Enum):
    """Types of metrics"""

    # Quality metrics
    REQUIREMENTS_COVERAGE = "requirements_coverage"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION_QUALITY = "documentation_quality"

    # Performance metrics
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"

    # Reliability metrics
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    FAULT_TOLERANCE = "fault_tolerance"

    # Security metrics
    VULNERABILITY_COUNT = "vulnerability_count"
    SECURITY_SCORE = "security_score"
    COMPLIANCE_SCORE = "compliance_score"

    # Completeness metrics
    FEATURE_COVERAGE = "feature_coverage"
    EDGE_CASE_HANDLING = "edge_case_handling"
    TEST_COVERAGE = "test_coverage"

    # Efficiency metrics
    TIME_COMPLEXITY = "time_complexity"
    SPACE_COMPLEXITY = "space_complexity"
    OPTIMIZATION_SCORE = "optimization_score"

    # Maintainability metrics
    CODE_READABILITY = "code_readability"
    MODULARITY = "modularity"
    COUPLING = "coupling"

    # Scalability metrics
    HORIZONTAL_SCALABILITY = "horizontal_scalability"
    VERTICAL_SCALABILITY = "vertical_scalability"
    LOAD_HANDLING = "load_handling"
```

### Metric Value Structure

```python
@dataclass
class MetricValue:
    """A single metric value with metadata"""

    metric_type: MetricType
    value: Union[float, int, str]
    category: MetricCategory
    timestamp: float
    metadata: Dict[str, Any]
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Validate metric value"""
        # Ensure numeric values for metrics that need it
        if self.metric_type in [
            MetricType.REQUIREMENTS_COVERAGE,
            MetricType.ERROR_RATE,
            MetricType.SECURITY_SCORE
        ]:
            if not isinstance(self.value, (int, float)):
                raise ValueError(f"{self.metric_type} requires numeric value")
```

### Metrics Collection Process

```python
async def collect_metrics_for_artifact(artifact_id, artifact_content, artifact_type):
    """
    Collect all metrics for an artifact.

    Process:
    1. Extract measurable characteristics
    2. Calculate metric values
    3. Store in metrics collector
    4. Return metric set
    """

    # 1. Extract characteristics
    characteristics = extract_characteristics(artifact_content)

    # 2. Calculate metrics
    metrics = []

    # Requirements coverage
    requirements_coverage = calculate_requirements_coverage(
        artifact_content,
        characteristics.get("requirements", [])
    )
    metrics.append(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=requirements_coverage,
        category=MetricCategory.QUALITY,
        timestamp=datetime.now().timestamp(),
        unit="ratio",
        min_value=0.0,
        max_value=1.0,
        metadata={"requirements_count": len(characteristics.get("requirements", []))}
    ))

    # Code quality
    code_quality = calculate_code_quality(artifact_content)
    metrics.append(MetricValue(
        metric_type=MetricType.CODE_QUALITY,
        value=code_quality,
        category=MetricCategory.QUALITY,
        timestamp=datetime.now().timestamp(),
        unit="score",
        min_value=0.0,
        max_value=10.0,
        metadata={}
    ))

    # Security score
    security_score = calculate_security_score(artifact_content)
    metrics.append(MetricValue(
        metric_type=MetricType.SECURITY_SCORE,
        value=security_score,
        category=MetricCategory.SECURITY,
        timestamp=datetime.now().timestamp(),
        unit="score",
        min_value=0.0,
        max_value=10.0,
        metadata={}
    ))

    # 3. Create metric set
    metric_set = MetricSet(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        metrics={m.metric_type: m for m in metrics},
        timestamp=datetime.now().timestamp()
    )

    # 4. Store
    await metrics_collector.collect_metrics(metric_set)

    return metric_set
```

### Metrics Analysis

```python
async def analyze_metrics(artifact_id, metrics_collector):
    """
    Analyze collected metrics and generate insights.

    Returns:
        {
            "overall_score": float,
            "category_scores": dict,
            "strengths": list,
            "issues": list,
            "recommendations": list
        }
    """

    # Get metrics
    metric_set = await metrics_collector.get_metrics(artifact_id)

    if not metric_set:
        return None

    # Analyze each category
    category_scores = []

    for category in MetricCategory:
        category_metrics = metric_set.get_metrics_by_category(category)

        if not category_metrics:
            continue

        # Calculate category score
        cat_score = calculate_category_score(category_metrics)

        # Identify issues and strengths
        issues, strengths = analyze_category_metrics(category_metrics, cat_score)

        category_scores.append(CategoryScore(
            category=category,
            score=cat_score,
            weight=DEFAULT_CATEGORY_WEIGHTS[category],
            metric_count=len(category_metrics),
            issues=issues,
            strengths=strengths
        ))

    # Calculate overall
    overall = calculate_overall_score(category_scores)

    # Generate recommendations
    recommendations = generate_recommendations(category_scores)

    return AnalysisReport(
        artifact_id=artifact_id,
        overall_score=overall,
        category_scores=category_scores,
        recommendations=recommendations,
        strengths=[s for cs in category_scores for s in cs.strengths],
        critical_issues=[i for cs in category_scores for i in cs.issues if "critical" in i.lower()]
    )
```

---

## Historical Comparison and Trend Analysis

### Percentile Ranking

```python
def calculate_percentile_rank(current_score, historical_scores):
    """
    Calculate percentile rank of current score.

    Percentile = percentage of scores that are ≤ current score.

    Formula:
        percentile = (rank / (N + 1)) × 100

    where:
        rank = position in sorted array (1-indexed)
        N = total number of historical scores

    Example:
        Current score: 0.82
        Historical: [0.65, 0.72, 0.78, 0.80, 0.85, 0.88]
        Sorted: [0.65, 0.72, 0.78, 0.80, 0.82, 0.85, 0.88]
        Rank: 5th (1-indexed)
        Percentile: (5 / 7) × 100 = 71.4th percentile
    """

    if not historical_scores:
        return None

    # Add current score and sort
    all_scores = sorted(historical_scores + [current_score])

    # Find rank (1-indexed)
    rank = all_scores.index(current_score) + 1

    # Calculate percentile
    percentile = (rank / len(all_scores)) * 100

    return percentile
```

### Trend Detection Algorithms

#### Linear Regression Trend

```python
def linear_regression_trend(scores, timestamps):
    """
    Detect linear trend using ordinary least squares regression.

    Formula:
        y = mx + b

        where:
            m (slope) = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²)
            b (intercept) = (Σy - mΣx) / n

    Interpretation:
        slope > 0.01: Improving
        slope < -0.01: Declining
        otherwise: Stable

    Returns:
        {
            "direction": str,
            "slope": float,
            "intercept": float,
            "r_squared": float  # Goodness of fit
        }
    """

    n = len(scores)

    if n < 2:
        return None

    # Normalize timestamps
    if timestamps:
        min_time = min(timestamps)
        max_time = max(timestamps)
        if max_time > min_time:
            x = [(t - min_time) / (max_time - min_time) for t in timestamps]
        else:
            x = list(range(n))
    else:
        x = list(range(n))

    y = scores

    # Calculate sums
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)

    # Calculate slope and intercept
    denominator = n * sum_x2 - sum_x ** 2

    if denominator == 0:
        return {
            "direction": "stable",
            "slope": 0,
            "intercept": sum(y) / n,
            "r_squared": 1.0
        }

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    # Calculate R²
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - y_predi) ** 2 for yi, y_predi in zip(y, y_pred))

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Determine direction
    if slope > 0.01:
        direction = "improving"
    elif slope < -0.01:
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared
    }
```

#### Moving Average Trend

```python
def moving_average_trend(scores, window_size=3):
    """
    Detect trend using moving average crossover.

    Method:
        1. Calculate short-term moving average (MA_short)
        2. Calculate long-term moving average (MA_long)
        3. Compare current values

    Signal:
        - MA_short > MA_long: Uptrend (improving)
        - MA_short < MA_long: Downtrend (declining)
        - Otherwise: Stable

    Returns:
        {
            "direction": str,
            "ma_short": float,
            "ma_long": float,
            "short_scores": list,
            "long_scores": list
        }
    """

    if len(scores) < window_size * 2:
        return None

    # Calculate short-term MA (recent scores)
    short_window = min(window_size, len(scores) // 2)
    ma_short = sum(scores[-short_window:]) / short_window

    # Calculate long-term MA (all scores)
    long_window = min(window_size * 2, len(scores))
    ma_long = sum(scores[-long_window:]) / long_window

    # Determine trend
    if ma_short > ma_long * 1.05:  # 5% above long-term MA
        direction = "improving"
    elif ma_short < ma_long * 0.95:  # 5% below long-term MA
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "ma_short_window": short_window,
        "ma_long_window": long_window
    }
```

#### Mann-Kendall Trend Test

```python
def mann_kendall_trend(scores):
    """
    Non-parametric trend test using Mann-Kendall method.

    More robust than linear regression for non-linear trends.

    Method:
        1. Compare each pair of points (i, j) where i < j
        2. Count:
           - Concordant pairs: score increases (sign = +1)
           - Discordant pairs: score decreases (sign = -1)
           - Ties: scores are equal (sign = 0)

        3. Calculate Kendall's Tau (τ)
           τ = (S - D) / (n(n-1)/2)
           where S = concordant, D = discordant

        4. Test significance

    Returns:
        {
            "direction": str,
            "kendalls_tau": float,  # [-1, 1]
            "p_value": float,
            "is_significant": bool,
            "trend_strength": str  # "weak", "moderate", "strong"
        }
    """

    n = len(scores)

    if n < 3:
        return None

    # Calculate S statistic
    S = 0

    for i in range(n):
        for j in range(i + 1, n):
            if scores[j] > scores[i]:
                S += 1  # Concordant
            elif scores[j] < scores[i]:
                S -= 1  # Discordant
            # Ties: no change

    # Calculate Kendall's Tau
    max_comparisons = n * (n - 1) / 2
    kendalls_tau = S / max_comparisons

    # Interpret strength
    tau_abs = abs(kendalls_tau)

    if tau_abs < 0.1:
        strength = "none"  # No trend
    elif tau_abs < 0.3:
        strength = "weak"
    elif tau_abs < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    # Determine direction
    if kendalls_tau > 0.02:  # Small threshold to avoid noise
        direction = "improving"
    elif kendalls_tau < -0.02:
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "kendalls_tau": kendalls_tau,
        "trend_strength": strength,
        "S_statistic": S,
        "comparisons": int(max_comparisons)
    }
```

### Change Point Detection

```python
def detect_change_points(scores, min_size=3, threshold=2.0):
    """
    Detect points where score trend changes significantly.

    Uses mean shift detection with z-score threshold.

    Algorithm:
        1. Slide window of size min_size through scores
        2. Calculate mean before and after each point
        3. If difference exceeds threshold * std_dev, mark as change point

    Returns:
        {
            "change_points": [
                {
                    "index": int,
                    "before_mean": float,
                    "after_mean": float,
                    "change_magnitude": float
                }
            ],
            "num_changes": int
        }
    """

    if len(scores) < min_size * 2:
        return {
            "change_points": [],
            "num_changes": 0,
            "message": "Insufficient data for change point detection"
        }

    change_points = []

    for i in range(min_size, len(scores) - min_size + 1):
        # Calculate means before and after
        before_scores = scores[max(0, i - min_size):i]
        after_scores = scores[i:i + min_size]

        before_mean = sum(before_scores) / len(before_scores)
        after_mean = sum(after_scores) / len(after_scores)

        # Calculate overall std dev for reference
        overall_std = calculate_std_dev(scores)

        if overall_std == 0:
            continue

        # Calculate z-score of difference
        diff = after_mean - before_mean
        z_score = diff / overall_std

        # Mark as change point if significant
        if abs(z_score) >= threshold:
            change_points.append({
                "index": i,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "change_magnitude": diff,
                "z_score": z_score
            })

    return {
        "change_points": change_points,
        "num_changes": len(change_points)
    }
```

---

## Scoring Algorithms and Formulas

### Complete Scoring Flow

```python
def complete_scoring_pipeline(artifact, context):
    """
    End-to-end scoring pipeline.

    Steps:
        1. Run explicit gauntlet tests
        2. Analyze keyword frequency
        3. Extract patterns
        4. LLM-based evaluation (if available)
        5. Calculate dimension scores
        6. Calculate overall score
        7. Compare with historical data
        8. Generate verdict
        9. Provide recommendations
    """

    # Step 1: Run gauntlet tests
    gauntlet_results = run_all_gauntlet_tests(
        artifact["content"],
        artifact.get("requirements", [])
    )

    # Step 2: Keyword frequency analysis
    keyword_scores = analyze_keyword_frequency(
        artifact["content"],
        get_keyword_categories()
    )

    # Step 3: Pattern extraction
    extracted_entities = extract_patterns(
        artifact["content"],
        min_confidence=0.3
    )

    # Step 4: LLM evaluation (if client available)
    if context.get("hephaestus_client"):
        llm_scores = await evaluate_with_llm(
            artifact["content"],
            context["evaluation_criteria"],
            context["hephaestus_client"]
        )
    else:
        llm_scores = None

    # Step 5: Calculate dimension scores
    dimension_scores = {}

    for dimension in ALL_DIMENSIONS:
        dimension_scores[dimension] = calculate_dimension_score(
            dimension,
            gauntlet_results,
            keyword_scores,
            llm_scores
        )

    # Step 6: Calculate overall score
    overall_score = calculate_overall_score(dimension_scores)

    # Step 7: Historical comparison
    historical_comparison = await compare_with_historical(
        artifact["id"],
        artifact["type"],
        overall_score,
        context["metrics_collector"]
    )

    # Step 8: Determine verdict
    verdict = determine_verdict_from_dimensions(dimension_scores, overall_score)

    # Step 9: Generate recommendations
    recommendations = generate_recommendations(
        dimension_scores,
        gauntlet_results,
        historical_comparison
    )

    # Compile final report
    final_report = {
        "artifact_id": artifact["id"],
        "dimension_scores": dimension_scores,
        "overall_score": overall_score,
        "verdict": verdict,
        "historical_comparison": historical_comparison,
        "gauntlet_results": gauntlet_results,
        "recommendations": recommendations,
        "timestamp": datetime.now().timestamp()
    }

    return final_report
```

### Dimension Score Formula

```python
def calculate_dimension_score_formula(dimension, tests, keywords, llm=None):
    """
    Master formula for calculating dimension score.

    Formula:
        Score = (W_t × S_t + W_k × S_k + W_l × S_l) / (W_t + W_k + W_l)

        where:
            W_t = weight for test scores (default: 0.6)
            W_k = weight for keyword scores (default: 0.3)
            W_l = weight for LLM scores (default: 0.1)
            S_t = test score (0-10)
            S_k = keyword score (0-10)
            S_l = LLM score (0-10)

    If a component is missing, its weight is redistributed proportionally.
    """

    # Weights for each scoring mechanism
    weights = {
        "test": 0.6,
        "keyword": 0.3,
        "llm": 0.1
    }

    # Calculate scores from each mechanism
    test_score = get_test_score_for_dimension(dimension, tests)
    keyword_score = get_keyword_score_for_dimension(dimension, keywords)
    llm_score = get_llm_score_for_dimension(dimension, llm) if llm else None

    # Build list of (score, weight) pairs for available mechanisms
    available = []

    if test_score is not None:
        available.append((test_score, weights["test"]))

    if keyword_score is not None:
        available.append((keyword_score, weights["keyword"]))

    if llm_score is not None:
        available.append((llm_score, weights["llm"]))

    if not available:
        # No scores available: use heuristic
        return calculate_heuristic_score(dimension, tests, keywords)

    # Calculate weighted average
    weighted_sum = sum(score * weight for score, weight in available)
    total_weight = sum(weight for _, weight in available)

    dimension_score = weighted_sum / total_weight if total_weight > 0 else 0

    # Clamp to [0, 10]
    return max(0.0, min(10.0, dimension_score))
```

### Score Normalization

```python
def normalize_score(raw_score, scale_from, scale_to):
    """
    Normalize score from one scale to another.

    Formula:
        normalized = (value - min_from) / (max_from - min_from) × (max_to - min_to) + min_to

    Args:
        raw_score: Value to normalize
        scale_from: (min, max) of original scale
        scale_to: (min, max) of target scale

    Example:
        normalize_score(5, (0, 100), (0, 10))
        = (5 - 0) / (100 - 0) × (10 - 0) + 0
        = 0.05 × 10
        = 0.5
    """

    min_from, max_from = scale_from
    min_to, max_to = scale_to

    # Handle edge case where range is zero
    if max_from == min_from:
        return min_to

    # Normalize
    normalized = (raw_score - min_from) / (max_from - min_from)
    scaled = normalized * (max_to - min_to) + min_to

    return scaled
```

### Sigmoid Normalization

```python
def sigmoid_normalize(value, center=50, scale=10):
    """
    Normalize unbounded values using sigmoid function.

    Formula:
        normalized = 1 / (1 + e^(-(value - center) / scale))

    Properties:
        - Always outputs in (0, 1)
        - center: value that maps to 0.5
        - scale: controls steepness (lower = steeper)

    Args:
        value: Value to normalize (unbounded)
        center: Value that should map to 0.5
        scale: Scale parameter (lower = steeper curve)

    Example:
        sigmoid_normalize(50, center=50, scale=10) → 0.5
        sigmoid_normalize(60, center=50, scale=10) → 0.73
        sigmoid_normalize(40, center=50, scale=10) → 0.27
    """

    import math

    exponent = -(value - center) / scale
    normalized = 1 / (1 + math.exp(exponent))

    return normalized
```

---

## Configuration and Customization

### Category Weights

```python
# Default weights
DEFAULT_CATEGORY_WEIGHTS = {
    "functionality": 1.2,
    "performance": 0.9,
    "security": 1.3,
    "reliability": 1.1,
    "completeness": 1.0,
    "efficiency": 0.9,
    "maintainability": 0.8,
    "scalability": 0.8
}

# Customize for security-critical applications
SECURITY_FOCUSED_WEIGHTS = {
    **DEFAULT_CATEGORY_WEIGHTS,
    "security": 2.0,  # Even higher weight
    "reliability": 1.2,
    "performance": 0.7  # Lower priority
}

# Customize for research/scientific applications
SCIENTIFIC_FOCUSED_WEIGHTS = {
    **DEFAULT_CATEGORY_WEIGHTS,
    "quality": 1.5,  # High importance on quality
    "reliability": 1.3,  # Results must be reproducible
    "scalability": 0.5,  # Less concern for scaling
    "performance": 0.7
}
```

### Score Thresholds

```python
# Default verdict thresholds
DEFAULT_THRESHOLDS = {
    "excellent": 8.0,
    "good": 6.5,
    "acceptable": 5.0,
    "poor": 0.0
}

# Stricter thresholds (high-stakes applications)
STRICT_THRESHOLDS = {
    "excellent": 9.0,
    "good": 7.5,
    "acceptable": 6.0,
    "poor": 0.0
}

# Lenient thresholds (exploratory/prototype phase)
LENIENT_THRESHOLDS = {
    "excellent": 7.0,
    "good": 5.5,
    "acceptable": 4.0,
    "poor": 0.0
}
```

### Custom Scoring Rules

```python
class CustomScoringRules:
    """
    Define custom scoring rules for specific domains.
    """

    @staticmethod
    def medical_domain_rules(content):
        """
        Scoring rules for medical/clinical applications.

        Higher penalties for:
        - Missing safety considerations
        - Unclear procedures
        - Missing validation
        """

        score = 0.5

        # Safety is critical
        if "safety" not in content.lower():
            score -= 0.3  # Large penalty

        if "patient safety" in content.lower():
            score += 0.2

        # Validation is critical
        if "validate" not in content.lower() and "validation" not in content.lower():
            score -= 0.2

        # Clinical trial mentions (positive)
        if any(kw in content.lower() for kw in ["clinical trial", "fda", "protocol"]):
            score += 0.1

        return max(0, min(1, score))

    @staticmethod
    def financial_domain_rules(content):
        """
        Scoring rules for financial applications.

        Higher penalties for:
        - Missing audit trail
        - Lack of precision
        - Missing error handling
        """

        score = 0.5

        # Audit trail is critical
        if "audit" not in content.lower() and "logging" not in content.lower():
            score -= 0.2

        # Precision matters
        if "decimal" in content.lower() or "precision" in content.lower():
            score += 0.1

        # Compliance
        if any(kw in content.lower() for kw in ["gaap", "ifrs", "sox", "compliance"]):
            score += 0.2

        return max(0, min(1, score))
```

---

## Examples and Use Cases

### Example 1: Software Solution

```python
artifact = {
    "id": "solution_jwt_auth",
    "type": "solution",
    "content": """
    Implement JWT authentication for REST API.

    Key features:
    - JWT token generation with HS256 algorithm
    - Token validation middleware
    - Bcrypt password hashing (12 rounds)
    - Rate limiting: 10 requests per minute
    - Refresh token rotation every 24 hours

    Error handling:
    - Catch invalid token exceptions
    - Return 401 for expired tokens
    - Log authentication failures

    Security considerations:
    - Store tokens in httpOnly cookies
    - Implement CSRF protection
    - Validate all input parameters
    """,
    "requirements": [
        "JWT authentication",
        "bcrypt hashing",
        "rate limiting"
    ]
}

# Scoring process
result = await score_artifact(artifact)

# Result
{
    "dimension_scores": {
        "functionality": 9.0,    # All requirements met, clear implementation
        "performance": 7.5,     # Good: rate limiting, bcrypt
        "security": 9.5,        # Excellent: httpOnly, CSRF, validation
        "reliability": 8.5,      # Good error handling
        "completeness": 8.0,     # Comprehensive coverage
        "efficiency": 7.0,       # Some optimizations possible
        "maintainability": 8.5,  # Clear, well-documented
        "scalability": 7.5       # Rate limiting helps, but could be more distributed
    },
    "overall_score": 8.23,
    "verdict": "EXCELLENT",
    "recommendations": [
        "Consider distributed architecture for better scalability"
    ],
    "gauntlet_results": [
        {"test": "requirements_coverage", "score": 10.0, "result": "passed"},
        {"test": "edge_case_handling", "score": 7.0, "result": "passed"},
        {"test": "security_keywords", "score": 9.5, "result": "passed"}
    ]
}
```

### Example 2: Scientific Experiment

```python
artifact = {
    "id": "experiment_pendulum",
    "type": "experiment",
    "content": """
    Experiment: Determine acceleration due to gravity (g) using simple pendulum

    Objective:
    Measure g by analyzing pendulum motion period

    Materials:
    - Pendulum bob (mass = 0.5 kg)
    - String (length = 1.0 m, adjustable)
    - Stopwatch (precision = 0.01 s)
    - Meter stick (precision = 1 mm)
    - Protractor (precision = 1 degree)

    Procedure:
    1. Measure string length L from pivot to bob center
    2. Displace pendulum by small angle (< 10 degrees)
    3. Release and time 20 complete oscillations
    4. Divide total time by 20 to get period T
    5. Repeat 5 times for each length
    6. Use formula: T = 2π√(L/g) to solve for g
    7. Calculate: g = 4π²L/T²

    Data collected:
    - L = 1.0 m, T = 2.01 s → g = 9.78 m/s²
    - L = 0.8 m, T = 1.80 s → g = 9.77 m/s²
    - L = 0.6 m, T = 1.55 s → g = 9.84 m/s²

    Average g = 9.80 m/s²
    Accepted value: 9.81 m/s²
    Percent error: 0.10%

    Sources of error:
    - Air resistance (minimal for small bob)
    - String mass (assumed negligible)
    - Angle measurement (small angle approximation)
    - Human reaction time in timing

    Error analysis:
    Using error propagation, uncertainty in g is ±0.05 m/s²
    Measured value: 9.80 ± 0.05 m/s²
    Accepted value: 9.81 m/s²
    Within uncertainty range ✓
    """,
    "requirements": [
        "measure gravitational acceleration",
        "analyze error sources",
        "compare with accepted value"
    ]
}

# Scoring
result = await score_artifact(artifact)

# Result
{
    "dimension_scores": {
        "functionality": 9.5,    # Clear objective, proven method
        "quality": 9.0,          # Sound experimental design
        "completeness": 9.0,     # All steps detailed
        "reliability": 9.0,      # Multiple trials, error analysis
        "clarity": 9.0,          # Well-documented procedure
        "accuracy": 8.5,         # 0.10% error, very good
        "safety": 7.5,           # Minimal safety concerns mentioned
        "reproducibility": 9.5    # Fully documented, reproducible
    },
    "overall_score": 8.92,
    "verdict": "EXCELLENT"
}
```

### Example 3: Architecture Document

```python
artifact = {
    "id": "architecture_microservices",
    "type": "architecture",
    "content": """
    Microservices Architecture: E-commerce Platform

    System Components:
    1. API Gateway (Kong)
       - Rate limiting per service
       - Authentication & authorization
       - Request routing

    2. Services:
       a. User Service (Node.js)
          - User registration, authentication
          - Profile management
          - Database: PostgreSQL (replicated)

       b. Product Service (Python)
          - Product catalog
          - Search & filtering
          - Database: MongoDB (sharded)

       c. Order Service (Go)
          - Order processing
          - Payment integration
          - Database: PostgreSQL

    3. Data Layer:
       - PostgreSQL: User, Order data
       - MongoDB: Product catalog
       - Redis: Session cache, rate limiting
       - Elasticsearch: Search indexing

    4. Message Queue:
       - RabbitMQ for async communication
       - Event-driven architecture

    Scalability:
    - Horizontal scaling: All services stateless
    - Database: Master-slave replication, sharding for Product DB
    - Caching: Redis cluster with consistent hashing

    Reliability:
    - Circuit breakers between services
    - Retry logic with exponential backoff
    - Health checks and monitoring
    - Graceful degradation

    Security:
    - OAuth 2.0 / JWT for authentication
    - HTTPS/TLS for all communication
    - Input validation at API gateway
    - Secrets management (Vault)
    """,
    "requirements": [
        "microservices architecture",
        "handle 10k concurrent users",
        "99.9% uptime"
    ]
}

# Scoring
result = await score_artifact(artifact)

# Result
{
    "dimension_scores": {
        "functionality": 9.0,    # Comprehensive coverage
        "scalability": 9.0,     # Horizontal scaling, sharding
        "reliability": 9.0,      # Circuit breakers, retries, monitoring
        "security": 8.5,         # OAuth, HTTPS, input validation
        "performance": 8.5,     # Caching, async messaging
        "maintainability": 7.5,  # Service boundaries clear
        "completeness": 9.0,     # All components addressed
        "quality": 8.5           # Well-designed architecture
    },
    "overall_score": 8.56,
    "verdict": "EXCELLENT"
}
```

---

## Best Practices

### For Using the Scoring System

1. **Define Clear Requirements**
   ```python
   # Good: Specific requirements
   requirements = [
       "JWT authentication with 256-bit keys",
       "Response time < 200ms",
       "Handle 1000 concurrent users"
   ]

   # Bad: Vague requirements
   requirements = [
       "Good security",
       "Fast performance",
       "Handle many users"
   ]
   ```

2. **Choose Appropriate Thresholds**
   ```python
   # For critical systems (medical, financial):
   thresholds = STRICT_THRESHOLDS

   # For prototypes/exploration:
   thresholds = LENIENT_THRESHOLDS

   # For general use:
   thresholds = DEFAULT_THRESHOLDS
   ```

3. **Customize Weights for Domain**
   ```python
   # Security-focused: increase security weight
   weights = SECURITY_FOCUSED_WEIGHTS

   # Research-focused: increase quality/reliability weights
   weights = SCIENTIFIC_FOCUSED_WEIGHTS

   # Performance-focused: increase performance weight
   weights = {
       **DEFAULT_CATEGORY_WEIGHTS,
       "performance": 1.5
   }
   ```

4. **Use Multiple Scoring Mechanisms**
   - Don't rely on a single mechanism
   - Combine tests, keywords, LLM, and heuristics
   - Cross-validate results

5. **Review Recommendations**
   - Always review the recommendations, not just scores
   - Low-scoring dimensions indicate improvement areas
   - Historical comparison provides context

### For Interpreting Scores

1. **Look at the Full Picture**
   - Overall score is a summary
   - Check individual dimension scores
   - Identify critical dimensions

2. **Consider Historical Context**
   - A score of 7.5 might be excellent or poor depending on history
   - Use percentile rank for context
   - Check trends (improving vs declining)

3. **Account for Domain**
   - Scientific domains need high reliability
   - Startups may prioritize speed over scalability
   - Adjust expectations accordingly

4. **Track Changes Over Time**
   - Monitor trends in scoring
   - Identify regression early
   - Celebrate improvements

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: All Scores Are Zero

**Possible Causes:**
- Scoring mechanisms not finding matches
- Content too short or malformed
- Tests not configured correctly

**Solutions:**
```python
# Check content length
if len(content) < 10:
    logger.warning("Content too short for meaningful scoring")

# Check test configuration
print("Test criteria:", test_criteria)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Issue 2: Scores Seem Wrong

**Possible Causes:**
- Wrong weights configured
- Thresholds inappropriate for domain
- Tests not matching content type

**Solutions:**
```python
# Verify weights
print("Current weights:", CATEGORY_WEIGHTS)

# Check thresholds
print("Current thresholds:", SCORE_THRESHOLDS)

# Review individual test results
for test in gauntlet_results:
    print(f"{test['name']}: {test['score']} - {test['result']}")

# Adjust if needed
CUSTOM_WEIGHTS = {
    **DEFAULT_CATEGORY_WEIGHTS,
    "your_dimension": 2.0  # Increase weight
}
```

#### Issue 3: Historical Comparison Returns No Data

**Possible Causes:**
- No historical artifacts available
- Artifact type mismatch
- Historical artifacts not scored yet

**Solutions:**
```python
# Check for historical data
historical = await get_historical_metrics(artifact_type)
print(f"Found {len(historical)} historical artifacts")

# Check if they have scores
for hm in historical:
    report = await analyze_artifact(hm.artifact_id)
    if report:
        print(f"{hm.artifact_id}: {report.overall_score}")
    else:
        print(f"{hm.artifact_id}: No score available")

# Score historical artifacts if needed
for hm in historical:
    await score_artifact(hm)
```

#### Issue 4: LLM Scoring Fails

**Possible Causes:**
- Hephaestus client not configured
- API key missing
- Network issues

**Solutions:**
```python
# Check client availability
if hephaestus_client is None:
    logger.warning("LLM client not available, using heuristic scoring")
    # Fallback to non-LLM scoring
    score = calculate_heuristic_score(artifact)

# Test connection
try:
    test_response = await hephaestus_client.generate("Test", max_tokens=10)
    print("LLM connection working:", test_response)
except Exception as e:
    print(f"LLM connection failed: {e}")
    print("Falling back to alternative scoring methods")
```

---

## API Reference

### Core Classes and Methods

#### EnhancedGauntletValidator

```python
class EnhancedGauntletValidator:
    """Enhanced gauntlet validator with multi-dimensional scoring"""

    def __init__(self, metrics_collector, test_registry=None):
        """
        Initialize validator.

        Args:
            metrics_collector: MetricsCollector instance
            test_registry: Optional custom test functions
        """

    async def validate_solution(
        self,
        artifact_id: str,
        solution_text: str,
        test_types: List[GauntletTestType] = None,
        requirements: List[str] = None,
        custom_tests: List[str] = None
    ) -> GauntletValidationResult:
        """
        Run gauntlet validation on solution.

        Args:
            artifact_id: Artifact identifier
            solution_text: Solution content to validate
            test_types: Types of tests to run (default: all)
            requirements: Requirements list for coverage checking
            custom_tests: Optional custom test names

        Returns:
            GauntletValidationResult with multi-dimensional score
        """
```

#### EvaluationMetricsCollector

```python
class EvaluationMetricsCollector:
    """Collects and manages evaluation metrics"""

    def __init__(self, storage_manager=None):
        """
        Initialize metrics collector.

        Args:
            storage_manager: Optional storage manager for persistence
        """

    async def collect_metrics(
        self,
        metric_set: MetricSet,
        persist: bool = True
    ) -> bool:
        """
        Collect metrics for an artifact.

        Args:
            metric_set: MetricSet to collect
            persist: Whether to persist to storage

        Returns:
            True if successful
        """

    async def get_metrics(
        self,
        artifact_id: str
    ) -> Optional[MetricSet]:
        """Get metrics for artifact"""

    async def compare_metrics(
        self,
        artifact_id_1: str,
        artifact_id_2: str
    ) -> Dict[str, Any]:
        """Compare metrics between two artifacts"""
```

#### MetricsAnalyzer

```python
class MetricsAnalyzer:
    """Analyzes metrics and generates reports"""

    def __init__(
        self,
        metrics_collector: EvaluationMetricsCollector,
        category_weights: Dict[MetricCategory, float] = None
    ):
        """
        Initialize analyzer.

        Args:
            metrics_collector: MetricsCollector instance
            category_weights: Optional custom category weights
        """

    async def analyze_artifact(
        self,
        artifact_id: str,
        include_recommendations: bool = True
    ) -> Optional[AnalysisReport]:
        """Analyze artifact metrics"""

    async def compare_artifacts(
        self,
        artifact_id_1: str,
        artifact_id_2: str
    ) -> Optional[Dict[str, Any]]:
        """Compare two artifacts"""
```

#### HistoricalComparator

```python
class HistoricalComparator:
    """Compares current with historical data"""

    async def compare_with_historical(
        self,
        artifact_id: str,
        artifact_type: str,
        lookback_days: int = 30,
        limit: int = 50
    ) -> Optional[ComparisonReport]:
        """Compare artifact with historical data"""

    async def analyze_trends(
        self,
        artifact_type: str,
        metric_category: Optional[MetricCategory] = None,
        window_size: int = 20
    ) -> Dict[str, Any]:
        """Analyze trends over time"""
```

#### AdvancedRAGEngine

```python
class AdvancedRAGEngine:
    """Advanced RAG engine with hybrid search"""

    async def query(
        self,
        query_text: str,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        expand_query: bool = False,
        rerank: bool = True,
        **kwargs
    ) -> RAGResult:
        """
        Execute RAG query.

        Args:
            query_text: Query string
            search_type: Type of search
            top_k: Number of results
            filters: Metadata filters
            expand_query: Enable query expansion
            rerank: Enable result reranking

        Returns:
            RAGResult with retrieved and ranked documents
        """
```

---

## Appendix

### A. Glossary

- **Artifact**: Any document, code, design, or content being scored
- **Dimension**: A specific aspect being evaluated (e.g., security, performance)
- **Metric**: A measurable quantity used for scoring
- **Percentile Rank**: Percentage of historical scores below current score
- **Z-Score**: Number of standard deviations from mean
- **Verdict**: Overall quality assessment (EXCELLENT/GOOD/ACCEPTABLE/POOR)
- **Critical Dimension**: Dimension scoring below 5.0/10
- **RAG**: Retrieval-Augmented Generation
- **HNSW**: Hierarchical Navigable Small World (vector indexing)
- **IVF**: Inverted File Index (vector indexing)

### B. Default Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `min_confidence` | 0.3 | Minimum confidence for pattern extraction |
| `top_k` | 5 | Number of search results |
| `temperature` | 0.3 | LLM temperature (lower = more consistent) |
| `lookback_days` | 30 | Historical comparison window |
| `threshold` | 2.0 | Z-score threshold for anomalies |

### C. Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Gauntlet validation (7 tests) | 100-500ms | Depends on content length |
| Pattern extraction | 50-200ms | Regex-based, very fast |
| LLM evaluation | 1-3 seconds | Depends on LLM speed |
| Historical comparison | 50-150ms | In-memory operations |
| Full scoring pipeline | 2-5 seconds | All components combined |

### D. Domain Mappings

| Domain | Critical Dimensions | Less Critical Dimensions |
|--------|-------------------|----------------------|
| **Software Development** | Functionality, Security | Scalability (initially) |
| **Scientific Research** | Quality, Reliability | Efficiency |
| **Medical/Clinical** | Safety, Reliability | Efficiency |
| **Financial** | Reliability, Security | Performance (within SLA) |
| **Real-time Systems** | Performance, Reliability | Maintainability |
| **Startups/MVPs** | Functionality, Speed | Scalability, Maintainability |

---

**End of Complete Scoring System Documentation**

For questions or issues, refer to the code documentation in:
- `ragbits_integration/evaluation/` - Evaluation framework
- `ragbits_integration/evaluation/metrics/` - Metrics system
- `ragbits_integration/evaluation/gauntlets/` - Gauntlet validation
- `ragbits_integration/evaluation/comparison/` - Historical comparison

**Version History:**
- v1.0 (2025-12-29): Initial comprehensive documentation
