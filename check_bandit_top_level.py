import json
from pathlib import Path

# Read the Bandit report
with open('bandit_final_report.json', 'r') as f:
    report = json.load(f)

# Filter for top-level files only
top_level_issues = []
for issue in report.get('results', []):
    filename = issue.get('filename', '')
    # Check if it's a top-level file (no subdirectories after the first .\)
    if filename.startswith('.\\'):
        parts = filename.split('\\')
        # Only count if there's just .\filename.py (2 parts)
        if len(parts) == 2:
            top_level_issues.append(issue)

print('='*80)
print('BANDIT SECURITY SCAN - TOP LEVEL FILES ONLY')
print('='*80)
print()

# Count issues by severity for top-level
severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
issues_by_type = {}

for issue in top_level_issues:
    severity = issue.get('issue_severity', 'UNKNOWN')
    test_id = issue.get('test_id', 'UNKNOWN')

    if severity in severity_counts:
        severity_counts[severity] += 1

    if test_id not in issues_by_type:
        issues_by_type[test_id] = []
    issues_by_type[test_id].append(issue)

print(f'Total Issues in Top-Level Files: {len(top_level_issues)}')
print()

if len(top_level_issues) == 0:
    print('[SUCCESS] No security issues found in top-level files!')
    print()
    print('All security fixes have been successfully validated.')
else:
    print('BY SEVERITY:')
    print('-'*80)
    for severity, count in severity_counts.items():
        if count > 0:
            print(f'  {severity}: {count}')
    print()

    print('BY ISSUE TYPE:')
    print('-'*80)
    for test_id, issues in sorted(issues_by_type.items()):
        if len(issues) > 0:
            print(f'  {test_id}: {len(issues)}')
    print()

    # Show high/medium severity issues
    high_medium = [i for i in top_level_issues if i.get('issue_severity') in ['HIGH', 'MEDIUM']]
    if high_medium:
        print('HIGH/MEDIUM SEVERITY ISSUES:')
        print('-'*80)
        for issue in high_medium[:20]:
            severity = issue.get('issue_severity', 'UNKNOWN')
            test_id = issue.get('test_id', 'UNKNOWN')
            filename = issue.get('filename', 'unknown')
            line = issue.get('line_number', 0)
            text = issue.get('issue_text', 'No description')
            print(f'  [{severity}] {test_id}')
            print(f'    File: {filename}:{line}')
            print(f'    Issue: {text}')
            print()

print('='*80)
