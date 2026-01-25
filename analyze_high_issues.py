import json
from pathlib import Path

with open('verification_scan.json', 'r') as f:
    report = json.load(f)

# Get HIGH severity issues for top-level files
high_issues = []
for issue in report.get('results', []):
    filename = issue.get('filename', '')
    if filename.startswith('.\\'):
        parts = filename.split('\\')
        if len(parts) == 2:
            if issue.get('issue_severity') == 'HIGH':
                high_issues.append(issue)

print(f'HIGH Severity Issues in Top-Level Files: {len(high_issues)}')
print()

# Group by test ID
by_type = {}
for issue in high_issues:
    test_id = issue.get('test_id', 'UNKNOWN')
    if test_id not in by_type:
        by_type[test_id] = []
    by_type[test_id].append(issue)

print('By Type:')
for test_id, issues in sorted(by_type.items()):
    print(f'  {test_id}: {len(issues)}')

print()
print('Sample issues:')
for issue in high_issues[:15]:
    fname = issue.get('filename', '').replace('.\\', '')
    line = issue.get('line_number', 0)
    test_id = issue.get('test_id', 'UNKNOWN')
    text = issue.get('issue_text', 'No description')
    print(f'  {fname}:{line} [{test_id}]')
    print(f'    {text}')
