import json
from pathlib import Path

with open('verification_scan.json', 'r') as f:
    report = json.load(f)

# Get MEDIUM severity issues for top-level files
medium_issues = []
for issue in report.get('results', []):
    filename = issue.get('filename', '')
    if filename.startswith('.\\'):
        parts = filename.split('\\')
        if len(parts) == 2:
            if issue.get('issue_severity') == 'MEDIUM':
                medium_issues.append(issue)

print(f'MEDIUM Severity Issues in Top-Level Files: {len(medium_issues)}')
print()

# Group by test ID
by_type = {}
for issue in medium_issues:
    test_id = issue.get('test_id', 'UNKNOWN')
    if test_id not in by_type:
        by_type[test_id] = []
    by_type[test_id].append(issue)

print('By Type:')
for test_id, issues in sorted(by_type.items(), key=lambda x: -len(x[1])):
    count = len(issues)
    print(f'  {test_id}: {count}')
    if count <= 5:
        for issue in issues:
            fname = issue.get('filename', '').replace('.\\', '')
            line = issue.get('line_number', 0)
            print(f'    {fname}:{line}')
