#!/usr/bin/env python3
"""Coverage Gap Analysis"""
import re
from pathlib import Path
from collections import defaultdict

htmlcov_path = Path('htmlcov')
index_file = htmlcov_path / 'index.html'
html_content = index_file.read_text()

pattern = r'<tr class="[^"]*">\s*<td class="name"><a href="([^"]+)">([^<]+)</a></td>\s*<td class="spacer">&nbsp;</td>\s*<td>(\d+)</td>\s*<td>(\d+)</td>\s*<td>(\d+)</td>\s*<td class="spacer">&nbsp;</td>\s*<td[^>]* data-ratio="(\d+)\s+(\d+)">(\d+)%</td>'

gaps = []
for match in re.finditer(pattern, html_content):
    html_file = match.group(1)
    module_name = match.group(2).replace('\u2009', '').replace('\\', '/')
    statements = int(match.group(3))
    missing = int(match.group(4))
    excluded = int(match.group(5))
    covered = int(match.group(6))
    total_statements = int(match.group(7))
    coverage_pct = int(match.group(8))

    if '__init__' in html_file or 'test_' in html_file or coverage_pct == 100:
        continue

    module_lower = module_name.lower()
    if coverage_pct < 20 and any(x in module_lower for x in ['security', 'validator', 'independence']):
        priority, category = 1, 'Security & Validation'
    elif coverage_pct == 0 and any(x in module_lower for x in ['performance', 'monitoring', 'config', 'api']):
        priority, category = 1, 'Infrastructure'
    elif coverage_pct == 0:
        priority, category = 2, 'Zero Coverage'
    elif coverage_pct < 50 and any(x in module_lower for x in ['core', 'phase', 'integration', 'stage']):
        priority, category = 2, 'Core Logic'
    elif coverage_pct < 50:
        priority, category = 3, 'Major Gap'
    elif coverage_pct < 80:
        priority, category = 4, 'Moderate Gap'
    else:
        priority, category = 5, 'Minor Gap'

    effort = f'{(missing//30)+1}-{(missing//20)+2} days' if coverage_pct < 50 else f'{(missing//50)+1} days'

    gaps.append({
        'file': module_name,
        'pct': coverage_pct,
        'missing': missing,
        'statements': statements,
        'priority': priority,
        'category': category,
        'effort': effort
    })

gaps.sort(key=lambda g: (g['priority'], g['pct']))

print('='*100)
print('COVERAGE GAP ANALYSIS REPORT - Path to 100% Coverage')
print('='*100)
print('')
print(f'Total files with gaps: {len(gaps)}')
print(f'Total missing lines: {sum(g["missing"] for g in gaps):,}')
print(f'Zero coverage files: {len([g for g in gaps if g["pct"] == 0])}')
print(f'Critical priority (P1): {len([g for g in gaps if g["priority"] == 1])}')
print(f'High priority (P2): {len([g for g in gaps if g["priority"] == 2])}')
print('')

print('='*100)
print('CRITICAL GAPS - PRIORITY 1 (Immediate Action Required)')
print('='*100)
p1 = [g for g in gaps if g['priority'] == 1]
for g in p1:
    print(f"\n{g['file']}")
    print(f"  Coverage: {g['pct']}% | Missing: {g['missing']} lines | Effort: {g['effort']}")
    print(f"  Category: {g['category']}")

print('\n')
print('='*100)
print('HIGH PRIORITY GAPS - PRIORITY 2')
print('='*100)
p2 = [g for g in gaps if g['priority'] == 2][:20]
for g in p2:
    print(f"{g['file']:<60} | {g['pct']:>3}% | {g['missing']:>4} miss | {g['effort']:>15} | {g['category']}")

print('\n')
print('='*100)
print('TOP 50 FILES REQUIRING TESTS')
print('='*100)
print('FILE | COVERAGE | MISSING | EFFORT | CATEGORY')
print('-'*100)
for g in gaps[:50]:
    print(f"{g['file']:<60} | {g['pct']:>3}% | {g['missing']:>4} miss | {g['effort']:>12} | {g['category']}")

print('\n')
print('='*100)
print('RECOMMENDED ACTION PLAN')
print('='*100)
print("""
WEEK 1-2: Critical Security & Infrastructure
  - Focus on P1 gaps (security, validators, infrastructure)
  - Create test infrastructure for zero-coverage modules
  - Target: Add tests for all P1 files

WEEK 3-4: Core Logic Coverage
  - Address P2 gaps (core logic, integrations)
  - Test all public functions in core modules
  - Add integration tests for multi-stage workflows
  - Target: Achieve 70%+ coverage on all core modules

MONTH 2: Fill Remaining Gaps
  - Address P3 gaps (major coverage holes)
  - Focus on error paths and edge cases
  - Add property-based tests
  - Target: Reach 85% overall coverage

MONTH 3+: Polish and Optimize
  - Address P4-P5 gaps (moderate/minor)
  - Focus on hard-to-test scenarios
  - Optimize test performance
  - Target: Achieve 95%+ overall coverage

TESTING STRATEGY:
  1. Start with unit tests for all public APIs
  2. Add integration tests for workflows
  3. Include error injection for exception paths
  4. Use property-based testing for data validation
  5. Add performance benchmarks for optimization code

COVERAGE TARGETS BY MODULE TYPE:
  - Security modules: 100% (non-negotiable)
  - Validation logic: 95%+
  - Core algorithms: 95%+
  - Integration points: 90%+
  - Utility functions: 100%
  - Overall target: 95%+ coverage
""")
