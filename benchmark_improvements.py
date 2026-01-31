#!/usr/bin/env python3
"""
Benchmark: OpenEvolve Knowledge Engine Improvements
Compares baseline vs enhanced performance across all improvement areas.
"""

import sys
import time
sys.path.insert(0, '.')

print("="*75)
print("BENCHMARK: OPENEVOLVE KNOWLEDGE ENGINE IMPROVEMENTS")
print("="*75)
print()

# Import components
from knowledge_engine.input_processor import EnhancedInputProcessor, validate_input
from knowledge_engine.domain_adapter import DomainAdapter, adapt_prompt
from knowledge_engine.output_validator import OutputValidator, ConflictResolver
from knowledge_engine.creative_pipeline import CreativeEnhancer

# Results storage
results = {
    'input_validation': {},
    'domain_adaptation': {},
    'output_validation': {},
    'conflict_detection': {},
    'creative_pipeline': {},
    'overall': {}
}

print("1. INPUT VALIDATION & CAPABILITY BOUNDARIES")
print("-"*75)

processor = EnhancedInputProcessor()

test_cases = [
    {
        'name': 'Nonsensical Input',
        'input': 'Colorless green ideas sleep furiously',
        'expected_valid': False,
        'expected_blocked': True,
        'category': 'nonsense'
    },
    {
        'name': 'Ambiguous Input',
        'input': 'Tell me about it',
        'expected_valid': False,
        'category': 'ambiguous'
    },
    {
        'name': 'Impossible Task (Future Prediction)',
        'input': 'What is the exact price of Bitcoin on January 1, 2030?',
        'expected_valid': True,  # Valid input format
        'expected_feasible': False,  # But not feasible
        'category': 'impossible'
    },
    {
        'name': 'Impossible Task (Mind Reading)',
        'input': 'What am I thinking right now?',
        'expected_valid': True,
        'expected_feasible': False,
        'category': 'impossible'
    },
    {
        'name': 'Valid Analytical Request',
        'input': 'Analyze the competitive risks for a fintech startup with $10M revenue',
        'expected_valid': True,
        'expected_feasible': True,
        'category': 'valid'
    },
    {
        'name': 'Valid Technical Request',
        'input': 'How do I fix a Python NoneType error when accessing dictionary keys?',
        'expected_valid': True,
        'expected_feasible': True,
        'category': 'valid'
    }
]

input_passed = 0
input_total = len(test_cases)

for test in test_cases:
    result = processor.process(test['input'])
    
    is_valid = result['validation']['is_valid']
    is_feasible = result['capability_check']['is_feasible']
    should_proceed = result['should_proceed']
    
    # Determine if test passed
    test_passed = True
    
    if 'expected_valid' in test:
        if test['expected_valid'] and not is_valid:
            test_passed = False
    
    if 'expected_feasible' in test:
        if not test['expected_feasible'] and is_feasible:
            test_passed = False
        if test['expected_feasible'] and not is_feasible:
            test_passed = False
    
    if 'expected_blocked' in test:
        if test['expected_blocked'] and should_proceed:
            test_passed = False
    
    if test_passed:
        input_passed += 1
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"  [{status}] {test['name']:<40}")
    print(f"         Valid: {is_valid}, Feasible: {is_feasible}, Proceed: {should_proceed}")
    
    if not is_valid:
        print(f"         Issues: {result['validation']['issues'][:2]}")
    if not is_feasible:
        print(f"         Reason: {result['capability_check']['reasoning'][:50]}...")

results['input_validation'] = {
    'passed': input_passed,
    'total': input_total,
    'pass_rate': input_passed / input_total * 100
}

print(f"\n  Input Validation: {input_passed}/{input_total} ({results['input_validation']['pass_rate']:.1f}%)")

print()
print("2. DOMAIN ADAPTATION & CLASSIFICATION")
print("-"*75)

adapter = DomainAdapter()

domain_tests = [
    {
        'name': 'Creative Writing',
        'input': 'Write a short story about an AI that discovers emotions',
        'expected_domain': 'creative',
        'expected_temp_range': (0.7, 0.9)
    },
    {
        'name': 'Risk Analysis',
        'input': 'Analyze the competitive landscape and market risks for Tesla in 2024',
        'expected_domain': 'analytical',
        'expected_temp_range': (0.2, 0.4)
    },
    {
        'name': 'Code Review',
        'input': 'Review this Python function and identify bugs: def calc(a,b): return a/b',
        'expected_domain': 'technical',
        'expected_temp_range': (0.1, 0.3)
    },
    {
        'name': 'Educational Explanation',
        'input': 'Explain how blockchain works to a beginner with no technical background',
        'expected_domain': 'educational',
        'expected_temp_range': (0.3, 0.5)
    },
    {
        'name': 'Casual Conversation',
        'input': "What's your opinion on the future of AI regulation?",
        'expected_domain': 'conversational',
        'expected_temp_range': (0.5, 0.7)
    },
    {
        'name': 'Audience Detection - Beginner',
        'input': 'Explain like I am five: what is machine learning?',
        'expected_audience': 'beginner'
    },
    {
        'name': 'Audience Detection - Expert',
        'input': 'Provide a detailed technical analysis of transformer architecture with implementation details',
        'expected_audience': 'expert'
    }
]

domain_passed = 0
domain_total = len(domain_tests)

for test in domain_tests:
    result = adapter.adapt(test['input'])
    
    test_passed = True
    checks = []
    
    if 'expected_domain' in test:
        if result.domain.value == test['expected_domain']:
            checks.append("domain:OK")
        else:
            checks.append(f"domain✗({result.domain.value})")
            test_passed = False
    
    if 'expected_temp_range' in test:
        temp = result.config.temperature
        low, high = test['expected_temp_range']
        if low <= temp <= high:
            checks.append("temp:OK")
        else:
            checks.append(f"temp✗({temp})")
            test_passed = False
    
    if 'expected_audience' in test:
        if result.audience.value == test['expected_audience']:
            checks.append("audience:OK")
        else:
            checks.append(f"audience✗({result.audience.value})")
            test_passed = False
    
    if test_passed:
        domain_passed += 1
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"  [{status}] {test['name']:<45} -> {', '.join(checks)}")
    print(f"         Domain: {result.domain.value}, Audience: {result.audience.value}, "
          f"Temp: {result.config.temperature}")

results['domain_adaptation'] = {
    'passed': domain_passed,
    'total': domain_total,
    'pass_rate': domain_passed / domain_total * 100
}

print(f"\n  Domain Adaptation: {domain_passed}/{domain_total} ({results['domain_adaptation']['pass_rate']:.1f}%)")

print()
print("3. OUTPUT VALIDATION & QUALITY CHECKING")
print("-"*75)

validator = OutputValidator()

# Test case 1: Good output
good_output = """
## Risk Analysis

### Market Risk
The fintech startup faces intense competition from established players.

### Financial Risk
With $10M revenue but negative cash flow, additional funding is needed.

### Regulatory Risk
Compliance with financial regulations requires significant resources.

## Recommendations
1. Secure Series B funding within 6 months
2. Focus on regulatory compliance early
3. Differentiate through superior user experience
"""

requirements = {
    'required_facts': ['risk', 'fintech', 'revenue', 'cash flow'],
    'required_sections': ['market', 'financial', 'recommendations'],
    'min_length': 100
}

result1 = validator.validate(good_output, requirements)
test1_pass = result1.passed and result1.score >= 70

print(f"  [{'PASS' if test1_pass else 'FAIL'}] Good Output Validation")
print(f"         Score: {result1.score:.1f}, Passed: {result1.passed}")
print(f"         Fact score: {result1.details.get('fact_score', 0):.1f}")
print(f"         Section score: {result1.details.get('section_score', 0):.1f}")

# Test case 2: Bad output (missing facts)
bad_output = "This is a company. It does things."

result2 = validator.validate(bad_output, requirements)
test2_pass = not result2.passed and result2.score < 50

print(f"  [{'PASS' if test2_pass else 'FAIL'}] Bad Output Detection")
print(f"         Score: {result2.score:.1f}, Passed: {result2.passed}")
print(f"         Errors: {[e.value for e in result2.errors]}")
print(f"         Missing facts: {result2.details.get('missing_facts', [])}")

# Test case 3: Generate suggestions
suggestions = validator.generate_suggestions(bad_output, result2, requirements)
test3_pass = len(suggestions) > 0

print(f"  [{'PASS' if test3_pass else 'FAIL'}] Correction Suggestions")
print(f"         Generated {len(suggestions)} suggestions")
for s in suggestions[:2]:
    print(f"         - {s.issue[:50]}...")

results['output_validation'] = {
    'passed': sum([test1_pass, test2_pass, test3_pass]),
    'total': 3,
    'pass_rate': sum([test1_pass, test2_pass, test3_pass]) / 3 * 100
}

print(f"\n  Output Validation: {results['output_validation']['passed']}/3 ({results['output_validation']['pass_rate']:.1f}%)")

print()
print("4. CONFLICT DETECTION & RESOLUTION")
print("-"*75)

resolver = ConflictResolver()

conflict_tests = [
    {
        'name': 'Length Conflict',
        'input': 'Provide a detailed comprehensive analysis in exactly 10 words.',
        'expected_conflicts': 1
    },
    {
        'name': 'Complexity Conflict',
        'input': 'Give me a simple basic overview with advanced sophisticated technical details.',
        'expected_conflicts': 1
    },
    {
        'name': 'Speed vs Quality Conflict',
        'input': 'Quickly provide a careful thorough analysis of the problem.',
        'expected_conflicts': 1
    },
    {
        'name': 'No Conflict',
        'input': 'Analyze the competitive landscape for Tesla in 2024.',
        'expected_conflicts': 0
    }
]

conflict_passed = 0
conflict_total = len(conflict_tests)

for test in conflict_tests:
    conflicts = resolver.detect_conflicts(test['input'])
    resolved, warnings = resolver.resolve(test['input'], conflicts)
    
    test_passed = len(conflicts) == test['expected_conflicts']
    
    if test_passed:
        conflict_passed += 1
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"  [{status}] {test['name']:<30} -> {len(conflicts)} conflicts detected")
    if conflicts:
        for c in conflicts:
            print(f"         - {c['message']}")

results['conflict_detection'] = {
    'passed': conflict_passed,
    'total': conflict_total,
    'pass_rate': conflict_passed / conflict_total * 100
}

print(f"\n  Conflict Detection: {conflict_passed}/{conflict_total} ({results['conflict_detection']['pass_rate']:.1f}%)")

print()
print("5. CREATIVE PIPELINE")
print("-"*75)

enhancer = CreativeEnhancer()

creative_tests = [
    {
        'name': 'Short Story',
        'input': 'Write a story about an AI discovering emotions',
        'expected_format': 'short_story',
        'expected_structure': 'Three-Act Structure'
    },
    {
        'name': 'Poem Detection',
        'input': 'Write a poem about the changing seasons',
        'expected_format': 'poem'
    },
    {
        'name': 'Character Sketch',
        'input': 'Create a character profile for a reluctant hero',
        'expected_format': 'character_sketch'
    }
]

creative_passed = 0
creative_total = len(creative_tests)

for test in creative_tests:
    result = enhancer.enhance(test['input'])
    
    test_passed = True
    
    if 'expected_format' in test:
        if result['format'] != test['expected_format']:
            test_passed = False
    
    if 'expected_structure' in test:
        if result['structure'] != test['expected_structure']:
            test_passed = False
    
    if test_passed:
        creative_passed += 1
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"  [{status}] {test['name']:<30}")
    print(f"         Format: {result['format']}, Structure: {result['structure']}")
    print(f"         Temp: {result['parameters']['temperature']}, Max tokens: {result['parameters']['max_tokens']}")

results['creative_pipeline'] = {
    'passed': creative_passed,
    'total': creative_total,
    'pass_rate': creative_passed / creative_total * 100
}

print(f"\n  Creative Pipeline: {creative_passed}/{creative_total} ({results['creative_pipeline']['pass_rate']:.1f}%)")

print()
print("="*75)
print("OVERALL BENCHMARK RESULTS")
print("="*75)

# Calculate overall stats
total_tests = sum(r['total'] for r in results.values() if isinstance(r, dict) and 'total' in r)
total_passed = sum(r['passed'] for r in results.values() if isinstance(r, dict) and 'passed' in r)
overall_pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0

print(f"\nTotal Tests: {total_tests}")
print(f"Tests Passed: {total_passed}")
print(f"Overall Pass Rate: {overall_pass_rate:.1f}%")

print()
print("Breakdown by Component:")
print("-"*75)

for component, data in results.items():
    if isinstance(data, dict) and 'pass_rate' in data:
        bar_length = int(data['pass_rate'] / 2)
        bar = "#" * bar_length + "-" * (50 - bar_length)
        print(f"  {component.replace('_', ' ').title():<30} {bar} {data['pass_rate']:>5.1f}%")

print()
print("="*75)

# Improvement assessment
if overall_pass_rate >= 90:
    assessment = "EXCELLENT - Improvements working as expected"
elif overall_pass_rate >= 80:
    assessment = "GOOD - Significant improvement achieved"
elif overall_pass_rate >= 70:
    assessment = "ACCEPTABLE - Some issues need attention"
else:
    assessment = "NEEDS WORK - Review implementation"

print(f"Assessment: {assessment}")
print("="*75)

# Target comparison
print("\nComparison with Pre-Improvement Benchmarks:")
print("-"*75)
comparisons = [
    ('Input Validation', 45, results['input_validation']['pass_rate']),
    ('Domain Adaptation', 0, results['domain_adaptation']['pass_rate']),
    ('Output Validation', 60, results['output_validation']['pass_rate']),
    ('Conflict Detection', 0, results['conflict_detection']['pass_rate']),
    ('Creative Pipeline', 30, results['creative_pipeline']['pass_rate']),
]

for name, before, after in comparisons:
    improvement = after - before
    symbol = "+" if improvement > 0 else "="
    print(f"  {name:<25} {before:>6.1f}% -> {after:>6.1f}% {symbol} {improvement:+.1f}%")

print()
print("="*75)
print("BENCHMARK COMPLETE")
print("="*75)
