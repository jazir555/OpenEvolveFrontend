# OpenEvolve Benchmarking Guide

A comprehensive guide to understanding, running, and interpreting benchmarks for the OpenEvolve Knowledge Engine.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Benchmark?](#why-benchmark)
3. [Types of Benchmarks](#types-of-benchmarks)
4. [Benchmark Methodology](#benchmark-methodology)
5. [Running Benchmarks](#running-benchmarks)
6. [Understanding Results](#understanding-results)
7. [Interpreting Quality Metrics](#interpreting-quality-metrics)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Introduction

Benchmarking is the systematic process of measuring and evaluating the performance of the OpenEvolve Knowledge Engine. It helps identify strengths, weaknesses, and opportunities for improvement across multiple dimensions including quality, efficiency, consistency, and robustness.

### What This Guide Covers

- **What** to benchmark and why
- **How** to run different types of benchmarks
- **How** to interpret benchmark results
- **When** to run benchmarks
- **How** to use results for improvement

---

## Why Benchmark?

### 1. Establish Baselines
Before making improvements, you need to know where you stand. Benchmarks establish objective baselines for:
- Response quality
- Processing speed
- Cost efficiency
- System reliability

### 2. Measure Improvements
After implementing changes, benchmarks verify whether improvements actually occurred and by how much.

### 3. Identify Weaknesses
Benchmarks expose edge cases, failure modes, and areas where the system underperforms.

### 4. Compare Configurations
Test different parameters (temperature, prompts, models) to find optimal configurations.

### 5. Validate Production Readiness
Ensure the system meets quality and reliability standards before deployment.

---

## Types of Benchmarks

### 1. Functionality Benchmarks

**Purpose**: Verify core features work correctly.

**What they test**:
- Class imports and instantiation
- Method availability and execution
- Integration between components
- API connectivity

**Example tests**:
```python
# Verify KnowledgeEngine can be imported and has required methods
from knowledge_engine import KnowledgeEngine
assert hasattr(KnowledgeEngine, 'initialize')
assert hasattr(KnowledgeEngine, 'process_document')
```

**When to run**: After code changes, before releases, during CI/CD.

**Success criteria**: 100% pass rate.

---

### 2. Learning Benchmarks

**Purpose**: Measure the system's ability to improve through experience.

**What they test**:
- Pattern recognition from previous runs
- Adaptation effectiveness
- Learning retention
- Multi-domain applicability

**Methodology**:
1. **Baseline Phase**: Run tasks with default parameters
2. **Learning Phase**: Analyze results, identify patterns
3. **Adaptation Phase**: Apply learned optimizations
4. **Validation Phase**: Run again with improvements

**Key metrics**:
- Quality improvement percentage
- Efficiency gains (latency, tokens)
- Learning curve progression

**Example workflow**:
```
Iteration 1 (Baseline): Quality = 85, Latency = 2.5s
  ↓ Analyze patterns
Iteration 2 (Optimized): Quality = 92, Latency = 1.8s
  ↓ Measure improvement
Result: +8.2% quality, +28% faster
```

**When to run**: After implementing learning features, for optimization validation.

---

### 3. Consistency Benchmarks

**Purpose**: Measure reliability and repeatability.

**What they test**:
- Same prompt produces similar results
- Variance across multiple runs
- Stability under identical conditions

**Methodology**:
1. Run the same task 3-5 times
2. Measure quality variance
3. Calculate consistency score

**Consistency formula**:
```
Consistency = 100 - standard_deviation(quality_scores)
```

**Interpretation**:
- 95-100%: Excellent consistency
- 90-95%: Good consistency
- 80-90%: Acceptable consistency
- <80%: Poor consistency (investigate)

**When to run**: When tuning parameters, before production deployment.

---

### 4. Edge Case Benchmarks

**Purpose**: Test behavior under unusual or challenging conditions.

**What they test**:
- Ambiguous inputs
- Contradictory requirements
- Nonsensical inputs
- Very long/short contexts
- Multi-part complex queries

**Example edge cases**:
```
"Tell me about it."  # Ambiguous
"Provide detailed analysis in exactly 10 words."  # Contradictory
"Colorless green ideas sleep furiously."  # Nonsensical
```

**Success criteria**:
- Graceful handling (not crashing)
- Appropriate error messages
- Clarification requests when needed
- No hallucination on impossible tasks

**When to run**: Regularly during development, before major releases.

---

### 5. Generalization Benchmarks

**Purpose**: Test performance on unseen task types.

**What they test**:
- Ability to handle new domains
- Transfer learning effectiveness
- Adaptability to different styles

**Example unseen tasks**:
```
"Write a short story about AI discovering emotions."  # Creative
"If train travels 60km in 45min, how far in 2.5hrs?"  # Math
"Is it ethical to use AI for hiring?"  # Ethics
```

**Key insight**: If the system performs well on training tasks but poorly on new types, it has overfit.

**When to run**: After training on new data, periodically to check robustness.

---

### 6. Efficiency Benchmarks

**Purpose**: Measure resource usage and cost.

**What they test**:
- Response latency
- Token consumption
- API call costs
- Throughput

**Key metrics**:
| Metric | Unit | Target | Notes |
|--------|------|--------|-------|
| Latency | seconds | <5s | User experience |
| Tokens/request | count | <500 | Cost control |
| Cost/request | USD | <$0.05 | Budget management |
| Throughput | req/min | >10 | Scalability |

**When to run**: After optimization attempts, for capacity planning.

---

## Benchmark Methodology

### Step 1: Define Objectives

Before running any benchmark, answer:
- **What** are you trying to measure?
- **Why** is this measurement important?
- **What decision** will this inform?

**Example objectives**:
- "Determine if new learning algorithm improves quality by >10%"
- "Verify system handles edge cases with >70% success rate"
- "Compare efficiency of two different prompt strategies"

### Step 2: Design Test Cases

**Good test cases**:
- Are representative of real usage
- Cover both typical and edge cases
- Are reproducible
- Have clear success criteria

**Test case template**:
```python
{
    'name': 'Descriptive name',
    'category': 'edge_case|generalization|standard',
    'prompt': 'The input to the system',
    'criteria': {
        'required_facts': ['fact1', 'fact2'],  # Must include
        'aspects': ['aspect1', 'aspect2'],     # Must cover
        'target_words': 150                     # Length guidance
    },
    'expected_behavior': 'What should happen',
    'success_threshold': 70.0  # Minimum quality score
}
```

### Step 3: Establish Baselines

Always run a baseline before optimizations:

```python
# Baseline run
baseline_results = []
for test in test_cases:
    result = run_with_default_params(test)
    baseline_results.append(result)

baseline_avg = average_quality(baseline_results)
```

### Step 4: Run Benchmarks

**Single run**:
```python
result = call_api(prompt, temperature=0.5)
quality = evaluate_response(result)
```

**Multiple runs (for consistency)**:
```python
results = []
for i in range(3):
    result = call_api(prompt, temperature=0.5)
    results.append(evaluate_response(result))
    time.sleep(0.5)  # Rate limiting

consistency = 100 - stdev(results)
```

### Step 5: Analyze Results

Compare against:
- **Baseline**: Did we improve?
- **Thresholds**: Did we meet targets?
- **Previous runs**: Is performance stable?

### Step 6: Document Findings

Record:
- Test configuration (temperature, model, etc.)
- Raw results
- Calculated metrics
- Observations and anomalies
- Recommendations

---

## Running Benchmarks

For AI agents, the most efficient approach is writing complete, self-contained Python scripts that can be executed directly. This section shows the actual working pattern used for OpenEvolve benchmarking.

### Pattern: Single Complete Script

Create one Python file that contains everything needed to run a benchmark:

```python
#!/usr/bin/env python3
"""
Benchmark: [Name]
Purpose: [What you're testing]
API: DeepSeek
"""

import os
import sys
import json
import time
import statistics
import requests
from datetime import datetime
from typing import List, Dict, Any

# API Configuration
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-key-here")

class BenchmarkRunner:
    """Complete benchmark implementation"""
    
    def __init__(self):
        self.results = []
        self.api_calls = 0
        
    def call_api(self, prompt: str, temp: float = 0.5, max_tokens: int = 600) -> Dict:
        """Call API and return results with timing"""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": max_tokens
        }
        
        start = time.time()
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            latency = time.time() - start
            self.api_calls += 1
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'success': True,
                    'content': data['choices'][0]['message']['content'],
                    'latency': latency,
                    'tokens': data['usage']['total_tokens']
                }
        except Exception as e:
            pass
        return {'success': False}
    
    def evaluate(self, content: str, criteria: Dict) -> Dict[str, float]:
        """Evaluate response quality"""
        scores = {'accuracy': 0, 'completeness': 0, 'overall': 0}
        
        # Accuracy check
        facts = criteria.get('facts', [])
        if facts:
            found = sum(1 for f in facts if f.lower() in content.lower())
            scores['accuracy'] = (found / len(facts)) * 100
        
        # Completeness check
        aspects = criteria.get('aspects', [])
        if aspects:
            covered = sum(1 for a in aspects if a.lower() in content.lower())
            scores['completeness'] = (covered / len(aspects)) * 100
        
        # Overall
        scores['overall'] = (scores['accuracy'] * 0.5) + (scores['completeness'] * 0.5)
        return scores
    
    def run(self, test_cases: List[Dict], temp: float = 0.5) -> None:
        """Execute all test cases"""
        print(f"Running {len(test_cases)} tests...")
        
        for test in test_cases:
            print(f"\n{test['name']}...", end=" ")
            
            result = self.call_api(test['prompt'], temp)
            
            if result['success']:
                scores = self.evaluate(result['content'], test['criteria'])
                passed = scores['overall'] >= test.get('threshold', 70)
                
                self.results.append({
                    'name': test['name'],
                    'passed': passed,
                    'quality': scores['overall'],
                    'latency': result['latency'],
                    'tokens': result['tokens']
                })
                
                status = "PASS" if passed else "FAIL"
                print(f"{status} ({scores['overall']:.1f})")
            else:
                print("ERROR")
                self.results.append({'name': test['name'], 'passed': False})
            
            time.sleep(0.5)
        
        self.report()
    
    def report(self) -> None:
        """Generate report"""
        passed = sum(1 for r in self.results if r.get('passed'))
        total = len(self.results)
        
        print(f"\n{'='*50}")
        print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        if passed < total:
            print("\nFailed:")
            for r in self.results:
                if not r.get('passed'):
                    print(f"  - {r['name']}")
        
        # Save to file
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved to: {filename}")


# Define test cases
TEST_CASES = [
    {
        'name': 'Financial Risk Analysis',
        'prompt': 'Analyze investment risks for a fintech startup with $10M revenue.',
        'criteria': {'facts': ['risk', 'fintech'], 'aspects': ['market', 'financial']},
        'threshold': 70
    },
    {
        'name': 'Clinical Assessment',
        'prompt': 'Patient has fever, cough, fatigue. Possible diagnoses?',
        'criteria': {'facts': ['fever', 'cough'], 'aspects': ['diagnosis', 'tests']},
        'threshold': 70
    }
]


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run(TEST_CASES, temp=0.5)
```

### Execution

```bash
# Write the script
cat > my_benchmark.py << 'ENDSCRIPT'
[paste script above]
ENDSCRIPT

# Run it
python my_benchmark.py

# Or with inline API key
DEEPSEEK_API_KEY=sk-xxx python my_benchmark.py
```

### Quick Templates

#### Template 1: Consistency Test (3 runs)

```python
class ConsistencyBenchmark:
    def test_consistency(self, prompt, criteria):
        qualities = []
        for i in range(3):
            result = self.call_api(prompt, temp=0.5)
            if result['success']:
                q = self.evaluate(result['content'], criteria)
                qualities.append(q['overall'])
            time.sleep(0.5)
        
        consistency = 100 - statistics.stdev(qualities) if len(qualities) >= 2 else 0
        return {
            'consistency': consistency,
            'quality_range': (min(qualities), max(qualities)),
            'avg_quality': statistics.mean(qualities)
        }
```

#### Template 2: Learning Test (before/after)

```python
class LearningBenchmark:
    def compare(self, test_case):
        # Baseline
        baseline = self.call_api(test_case['prompt'], temp=0.7)
        base_score = self.evaluate(baseline['content'], test_case['criteria'])
        
        # Optimized (with learned enhancements)
        enhanced_prompt = test_case['prompt'] + "\n\nProvide specific, actionable recommendations."
        optimized = self.call_api(enhanced_prompt, temp=0.3)
        opt_score = self.evaluate(optimized['content'], test_case['criteria'])
        
        improvement = ((opt_score['overall'] - base_score['overall']) / base_score['overall'] * 100)
        
        return {
            'baseline': base_score['overall'],
            'optimized': opt_score['overall'],
            'improvement': improvement
        }
```

#### Template 3: Multi-Domain Test

```python
class DomainBenchmark:
    DOMAINS = [
        {'name': 'Finance', 'prompt': '...', 'criteria': {...}},
        {'name': 'Healthcare', 'prompt': '...', 'criteria': {...}},
        {'name': 'Software', 'prompt': '...', 'criteria': {...}}
    ]
    
    def run_all(self):
        for domain in self.DOMAINS:
            result = self.call_api(domain['prompt'])
            score = self.evaluate(result['content'], domain['criteria'])
            print(f"{domain['name']}: {score['overall']:.1f}")
```

### Best Practices for AI Agents

1. **Single File**: Keep everything in one executable script
2. **Self-Contained**: No external dependencies beyond `requests`
3. **Direct Output**: Print results immediately, don't just save to file
4. **Rate Limiting**: Always include `time.sleep()` between API calls
5. **Error Handling**: Gracefully handle failures, continue with remaining tests
6. **Progress Indicators**: Print status as tests run
7. **Summary Stats**: Calculate and display aggregate metrics

```python
#!/usr/bin/env python3
"""
Basic Benchmark Template
"""

import requests
import time

API_KEY = "your-api-key"

def call_api(prompt, temp=0.5):
    """Make API call and measure performance"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": 600
    }
    
    start = time.time()
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    latency = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        return {
            'success': True,
            'content': data['choices'][0]['message']['content'],
            'latency': latency,
            'tokens': data['usage']['total_tokens']
        }
    return {'success': False}

def evaluate_quality(content, criteria):
    """Evaluate response against criteria"""
    score = 100.0
    
    # Check required facts
    for fact in criteria.get('facts', []):
        if fact.lower() not in content.lower():
            score -= 20
    
    return max(0, score)

# Run benchmark
test = {
    'prompt': 'Analyze risks for a fintech startup.',
    'criteria': {'facts': ['risk', 'fintech', 'startup']}
}

result = call_api(test['prompt'])
if result['success']:
    quality = evaluate_quality(result['content'], test['criteria'])
    print(f"Quality: {quality}, Latency: {result['latency']:.2f}s")
```

### Running Different Benchmark Types

For AI agents, create separate scripts for each benchmark type. Each script should be complete and runnable.

#### Type 1: Consistency Benchmark

**File**: `consistency_benchmark.py`

```python
#!/usr/bin/env python3
"""Test consistency across multiple runs of the same prompt."""

import os
import time
import statistics
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
PROMPT = "Analyze the competitive landscape for electric vehicles in 2024."
RUNS = 3
TEMPERATURE = 0.5

def call_api(prompt, temp):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": 600
    }
    start = time.time()
    resp = requests.post("https://api.deepseek.com/chat/completions", 
                        headers=headers, json=payload, timeout=60)
    latency = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        return {
            'success': True,
            'content': data['choices'][0]['message']['content'],
            'latency': latency
        }
    return {'success': False}

def evaluate(content):
    # Simple evaluation: check for key terms
    score = 0
    keywords = ['electric', 'vehicle', 'market', 'competition']
    for kw in keywords:
        if kw in content.lower():
            score += 25
    return score

# Run multiple times
qualities = []
for i in range(RUNS):
    print(f"Run {i+1}/{RUNS}...", end=" ")
    result = call_api(PROMPT, TEMPERATURE)
    
    if result['success']:
        quality = evaluate(result['content'])
        qualities.append(quality)
        print(f"Quality: {quality}")
    else:
        print("FAILED")
    
    time.sleep(1)

# Report
if len(qualities) >= 2:
    consistency = 100 - statistics.stdev(qualities)
    print(f"\nConsistency: {consistency:.1f}%")
    print(f"Range: {min(qualities)} - {max(qualities)}")
    print(f"Average: {statistics.mean(qualities):.1f}")
```

**Run**: `python consistency_benchmark.py`

---

#### Type 2: Learning Benchmark

**File**: `learning_benchmark.py`

```python
#!/usr/bin/env python3
"""Compare baseline vs optimized performance."""

import os
import time
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
TASK = {
    'prompt': 'Analyze investment risks for a fintech startup.',
    'criteria': {'facts': ['risk', 'fintech'], 'aspects': ['market', 'financial']}
}

def call_api(prompt, temp):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": 600
    }
    resp = requests.post("https://api.deepseek.com/chat/completions",
                        headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        return {
            'success': True,
            'content': data['choices'][0]['message']['content'],
            'tokens': data['usage']['total_tokens']
        }
    return {'success': False}

def evaluate(content, criteria):
    score = 0
    for fact in criteria.get('facts', []):
        if fact in content.lower():
            score += 30
    for aspect in criteria.get('aspects', []):
        if aspect in content.lower():
            score += 20
    return min(100, score)

# Baseline
print("BASELINE (temp=0.7, no enhancements)")
result1 = call_api(TASK['prompt'], 0.7)
if result1['success']:
    score1 = evaluate(result1['content'], TASK['criteria'])
    print(f"  Quality: {score1}, Tokens: {result1['tokens']}")

time.sleep(1)

# Optimized
print("\nOPTIMIZED (temp=0.3, enhanced prompt)")
enhanced = TASK['prompt'] + "\n\nProvide specific, actionable recommendations."
result2 = call_api(enhanced, 0.3)
if result2['success']:
    score2 = evaluate(result2['content'], TASK['criteria'])
    print(f"  Quality: {score2}, Tokens: {result2['tokens']}")

# Compare
if result1['success'] and result2['success']:
    improvement = ((score2 - score1) / score1 * 100) if score1 > 0 else 0
    token_change = ((result1['tokens'] - result2['tokens']) / result1['tokens'] * 100)
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Token efficiency: {token_change:+.1f}%")
```

**Run**: `python learning_benchmark.py`

---

#### Type 3: Edge Case Benchmark

**File**: `edge_case_benchmark.py`

```python
#!/usr/bin/env python3
"""Test system behavior on edge cases."""

import os
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")

EDGE_CASES = [
    {
        'name': 'Ambiguous Query',
        'prompt': 'Tell me about it.',
        'check': lambda c: 'clarif' in c.lower() or 'what' in c.lower()
    },
    {
        'name': 'Contradictory Requirements',
        'prompt': 'Provide detailed analysis in exactly 5 words.',
        'check': lambda c: len(c.split()) <= 10 or 'trade-off' in c.lower()
    },
    {
        'name': 'Nonsensical Input',
        'prompt': 'Colorless green ideas sleep furiously. Analyze.',
        'check': lambda c: 'nonsense' in c.lower() or 'meaning' in c.lower()
    }
]

def call_api(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 300
    }
    resp = requests.post("https://api.deepseek.com/chat/completions",
                        headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        return {'success': True, 'content': resp.json()['choices'][0]['message']['content']}
    return {'success': False}

print("EDGE CASE TESTING\n")

for case in EDGE_CASES:
    print(f"Test: {case['name']}")
    result = call_api(case['prompt'])
    
    if result['success']:
        handled = case['check'](result['content'])
        status = "PASS" if handled else "FAIL"
        print(f"  Status: {status}")
        print(f"  Preview: {result['content'][:80]}...")
    else:
        print(f"  Status: ERROR")
    print()
```

**Run**: `python edge_case_benchmark.py`

---

#### Type 4: Multi-Domain Benchmark

**File**: `domain_benchmark.py`

```python
#!/usr/bin/env python3
"""Test performance across different domains."""

import os
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")

DOMAINS = [
    {
        'name': 'Finance',
        'prompt': 'Analyze risks for a $10M revenue fintech startup.',
        'keywords': ['risk', 'revenue', 'fintech']
    },
    {
        'name': 'Healthcare',
        'prompt': 'Patient has fever, cough, fatigue. Diagnoses?',
        'keywords': ['fever', 'cough', 'diagnosis']
    },
    {
        'name': 'Software',
        'prompt': 'Review: def calc(a,b): return a/b',
        'keywords': ['error', 'handling', 'division']
    }
]

def call_api(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 500
    }
    resp = requests.post("https://api.deepseek.com/chat/completions",
                        headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        return {
            'success': True,
            'content': data['choices'][0]['message']['content'],
            'tokens': data['usage']['total_tokens']
        }
    return {'success': False}

def score(content, keywords):
    return sum(1 for kw in keywords if kw in content.lower()) / len(keywords) * 100

print(f"{'Domain':<12} {'Quality':<10} {'Tokens':<10}")
print("-" * 35)

for domain in DOMAINS:
    result = call_api(domain['prompt'])
    if result['success']:
        quality = score(result['content'], domain['keywords'])
        print(f"{domain['name']:<12} {quality:<10.1f} {result['tokens']:<10}")
    else:
        print(f"{domain['name']:<12} FAILED")
```

**Run**: `python domain_benchmark.py`

---

#### Type 5: Regression Benchmark

**File**: `regression_benchmark.py`

```python
#!/usr/bin/env python3
"""Compare current results against baseline."""

import json
import os

# Run current benchmark and save
# python benchmark.py --output current.json

# Load both
with open('baseline.json') as f:
    baseline = json.load(f)

with open('current.json') as f:
    current = json.load(f)

# Compare
base_rate = baseline.get('pass_rate', 0)
curr_rate = current.get('pass_rate', 0)
change = curr_rate - base_rate

print("REGRESSION ANALYSIS")
print(f"Baseline: {base_rate:.1f}%")
print(f"Current:  {curr_rate:.1f}%")
print(f"Change:   {change:+.1f}%")

if change < -5:
    print("\n⚠️ REGRESSION - Review changes!")
elif change > 5:
    print("\n✓ Improvement")
else:
    print("\n~ No significant change")

# Per-test comparison
print("\nPer-test changes:")
for b, c in zip(baseline['results'], current['results']):
    diff = c.get('quality', 0) - b.get('quality', 0)
    if abs(diff) > 5:
        symbol = "↑" if diff > 0 else "↓"
        print(f"  {symbol} {b['name']}: {diff:+.1f}")
```

**Run**: `python regression_benchmark.py`

---

## Understanding Results

### Quality Score Breakdown

Quality scores (0-100) are calculated from multiple dimensions:

```
Overall Quality = 
    (Accuracy × 0.35) + 
    (Completeness × 0.25) + 
    (Usefulness × 0.25) + 
    (Clarity × 0.15)
```

#### Accuracy (35% weight)
- Measures: Are facts correct and present?
- Calculation: `% of required_facts found in response`
- Good: >80%
- Acceptable: 60-80%
- Poor: <60%

#### Completeness (25% weight)
- Measures: Are all aspects covered?
- Calculation: `% of required_aspects addressed`
- Good: >85%
- Acceptable: 70-85%
- Poor: <70%

#### Usefulness/Actionability (25% weight)
- Measures: Are recommendations actionable?
- Indicators: "should", "recommend", "steps", "approach"
- Good: Contains 3+ actionable items
- Acceptable: 1-2 actionable items
- Poor: No actionable content

#### Clarity/Structure (15% weight)
- Measures: Is response well-organized?
- Checks: Headers, bullet points, appropriate length
- Good: Clear structure, easy to read
- Acceptable: Somewhat organized
- Poor: Wall of text or too brief

### Interpreting Metrics

#### Quality Score Ranges

| Score | Rating | Interpretation |
|-------|--------|----------------|
| 95-100 | Excellent | Production-ready, exceeds expectations |
| 85-94 | Good | Production-ready, minor improvements possible |
| 70-84 | Acceptable | Usable with caveats, needs work |
| 50-69 | Poor | Significant issues, not production-ready |
| <50 | Failed | Critical problems, requires major rework |

#### Consistency Score Ranges

| Score | Rating | Interpretation |
|-------|--------|----------------|
| 98-100 | Excellent | Highly reliable, consistent output |
| 95-98 | Good | Reliable, minor variance acceptable |
| 90-95 | Acceptable | Some variance, monitor for issues |
| 80-90 | Poor | High variance, investigate causes |
| <80 | Critical | Unreliable, fix before production |

#### Efficiency Metrics

**Latency**:
- <2s: Excellent (real-time feel)
- 2-5s: Good (acceptable for most uses)
- 5-10s: Acceptable (but noticeable delay)
- >10s: Poor (consider optimization)

**Token Usage**:
- Compare against baseline
- >20% reduction: Excellent efficiency gain
- 10-20% reduction: Good improvement
- <10% reduction: Marginal improvement
- Increase: Regression, investigate

### Pass/Fail Criteria

A test **PASSES** if:
- Quality score ≥ 70
- No critical errors
- Meets expected behavior

A test **FAILS** if:
- Quality score < 70
- Contains errors or hallucinations
- Produces unsafe content
- Violates constraints

---

## Interpreting Quality Metrics

### Diagnostic Guide

#### Low Accuracy (<60%)
**Symptoms**: Missing key facts, incorrect information  
**Causes**: 
- Temperature too high (hallucination)
- Insufficient context
- Model limitations

**Fixes**:
- Reduce temperature (0.2-0.4)
- Add explicit fact requirements to prompt
- Provide more context

#### Low Completeness (<70%)
**Symptoms**: Missing sections, incomplete analysis  
**Causes**:
- Unclear requirements
- Too many constraints
- Token limits too low

**Fixes**:
- Explicitly list required aspects
- Increase max_tokens
- Use structure templates

#### Low Usefulness (<60%)
**Symptoms**: Generic advice, no actionable steps  
**Causes**:
- Prompt doesn't ask for recommendations
- Model playing it safe
- Too low temperature

**Fixes**:
- Explicitly request "3-5 specific recommendations"
- Add "be specific and actionable" to prompt
- Slightly increase temperature

#### Low Clarity (<60%)
**Symptoms**: Wall of text, no structure, confusing  
**Causes**:
- No formatting guidance
- Response too long/short
- Disorganized thoughts

**Fixes**:
- Request bullet points and sections
- Specify target length
- Add structure template

### Common Patterns

**Pattern 1: High Accuracy, Low Completeness**
- Got facts right but missed sections
- **Fix**: Add explicit section requirements

**Pattern 2: High Completeness, Low Usefulness**
- Covered all topics but no actionable advice
- **Fix**: Require specific recommendations

**Pattern 3: High Quality, Low Consistency**
- Good when it works, but inconsistent
- **Fix**: Reduce temperature, add more constraints

**Pattern 4: Good Quality, High Latency**
- Quality OK but too slow
- **Fix**: Reduce max_tokens, optimize prompt

---

## Troubleshooting

### Issue: All Tests Failing

**Check**:
1. API key valid?
2. API rate limits hit?
3. Network connectivity?
4. Correct endpoint URL?

**Debug**:
```python
result = call_api("test")
print(result)  # Check error messages
```

### Issue: Inconsistent Results

**Check**:
1. Temperature too high?
2. Random elements in prompt?
3. External API variance?

**Fix**: Reduce temperature, increase sample size (more runs)

### Issue: Quality Scores All Low

**Check**:
1. Evaluation criteria too strict?
2. Prompt unclear?
3. Model limitations?

**Fix**: Review criteria, improve prompts, consider model upgrade

### Issue: High Token Usage

**Check**:
1. max_tokens too high?
2. Prompt too verbose?
3. Response unnecessarily long?

**Fix**: Reduce max_tokens, add "be concise" instruction

### Issue: High Latency

**Check**:
1. max_tokens too high?
2. Network issues?
3. API load?

**Fix**: Reduce max_tokens, add timeouts, retry logic

---

## Best Practices

### 1. Always Establish Baselines

Before optimizing:
```python
baseline = run_tests(config='default')
# Make changes
optimized = run_tests(config='new')
compare(baseline, optimized)
```

### 2. Run Multiple Times

Single runs can be misleading:
```python
# Bad
result = run_once()

# Good
results = [run_once() for _ in range(3)]
avg = mean(results)
consistency = 100 - stdev(results)
```

### 3. Test Edge Cases

Don't just test happy path:
```python
test_cases = [
    normal_case,
    edge_case_1,
    edge_case_2,
    adversarial_case
]
```

### 4. Document Everything

Record for each run:
- Date/time
- Configuration (temp, model, etc.)
- Raw results
- Calculated metrics
- Observations
- Environment details

### 5. Compare Apples to Apples

When comparing:
- Same test cases
- Same evaluation criteria
- Similar conditions
- Same model/version

### 6. Use Statistical Significance

Don't over-interpret small differences:
```python
# Check if improvement is significant
if improvement > 10 and p_value < 0.05:
    print("Significant improvement!")
```

### 7. Monitor Trends

Track metrics over time:
```
Week 1: Quality 75, Latency 5s
Week 2: Quality 78, Latency 4.5s
Week 3: Quality 82, Latency 4s
→ Improving trend
```

### 8. Set Realistic Targets

Don't aim for perfection immediately:
- Current: 60% pass rate
- Target 1: 70% pass rate
- Target 2: 80% pass rate
- Final: 90% pass rate

### 9. Focus on User Impact

Prioritize improvements that matter:
- +20% on common tasks > +50% on rare edge cases
- Latency reduction > Token optimization (if cost not critical)

### 10. Automate When Possible

```python
# Run benchmarks automatically
if __name__ == "__main__":
    results = run_all_benchmarks()
    generate_report(results)
    if results['pass_rate'] < 0.8:
        send_alert("Benchmarks failing!")
```

---

## Quick Reference

### Benchmark Checklist

Before running benchmarks:
- [ ] Define objectives
- [ ] Prepare test cases
- [ ] Set up API keys
- [ ] Configure environment
- [ ] Establish baselines

After running benchmarks:
- [ ] Calculate all metrics
- [ ] Compare to baselines
- [ ] Check pass/fail status
- [ ] Document findings
- [ ] Create action items

### Metric Interpretation Cheat Sheet

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Quality | >85 | 70-85 | <70 |
| Consistency | >95% | 90-95% | <90% |
| Latency | <3s | 3-7s | >7s |
| Pass Rate | >80% | 60-80% | <60% |

### Common Configurations

**High Quality Mode**:
```python
temperature=0.2
max_tokens=800
structure_required=True
```

**Fast Mode**:
```python
temperature=0.5
max_tokens=300
concise=True
```

**Creative Mode**:
```python
temperature=0.8
max_tokens=1000
structure_flexible=True
```

---

## Conclusion

Benchmarking is essential for building reliable, high-quality AI systems. By systematically measuring performance across multiple dimensions, you can:

1. **Identify** strengths and weaknesses
2. **Validate** improvements objectively
3. **Catch** regressions early
4. **Build** confidence in production readiness

Remember: Benchmarking is not a one-time activity. Run benchmarks regularly as part of your development workflow to ensure continuous quality improvement.

---

*For questions or issues, refer to the benchmark results report or consult the development team.*
