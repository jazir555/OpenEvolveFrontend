"""
Analyze remaining non-security issues from Bandit scan
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_issues():
    with open('bandit_final_report.json', 'r') as f:
        report = json.load(f)

    # Filter for top-level files only
    top_level_issues = []
    for issue in report.get('results', []):
        filename = issue.get('filename', '')
        if filename.startswith('.\\'):
            parts = filename.split('\\')
            if len(parts) == 2:
                top_level_issues.append(issue)

    # Group by issue type
    issues_by_type = defaultdict(list)
    for issue in top_level_issues:
        test_id = issue.get('test_id', 'UNKNOWN')
        issues_by_type[test_id].append(issue)

    print("="*80)
    print("REMAINING NON-SECURITY ISSUES ANALYSIS")
    print("="*80)
    print()

    # Show counts
    for test_id in sorted(issues_by_type.keys()):
        issues = issues_by_type[test_id]
        severity = issues[0].get('issue_severity', 'UNKNOWN')
        print(f"{test_id}: {len(issues)} issues ({severity} severity)")

    print()
    print("="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    print()

    # B105: Hardcoded password
    if 'B105' in issues_by_type:
        print("\nB105: Hardcoded Password")
        print("-"*80)
        for issue in issues_by_type['B105'][:10]:
            filename = issue.get('filename', '').replace('.\\', '')
            line = issue.get('line_number', 0)
            text = issue.get('issue_text', 'No description')
            print(f"  {filename}:{line}")
            print(f"    {text}")

    # B106: Hardcoded password function argument
    if 'B106' in issues_by_type:
        print("\nB106: Hardcoded Password Function Argument")
        print("-"*80)
        for issue in issues_by_type['B106'][:10]:
            filename = issue.get('filename', '').replace('.\\', '')
            line = issue.get('line_number', 0)
            text = issue.get('issue_text', 'No description')
            print(f"  {filename}:{line}")
            print(f"    {text}")

    # B603: Subprocess call with shell=True
    if 'B603' in issues_by_type:
        print("\nB603: Subprocess Call With Shell=True")
        print("-"*80)
        files = set()
        for issue in issues_by_type['B603'][:20]:
            filename = issue.get('filename', '').replace('.\\', '')
            files.add(filename)
        for f in sorted(files):
            print(f"  {f}")

    # B311: Non-cryptographic random
    if 'B311' in issues_by_type:
        print("\nB311: Non-Cryptographic Random (PRNG)")
        print("-"*80)
        files = set()
        for issue in issues_by_type['B311'][:20]:
            filename = issue.get('filename', '').replace('.\\', '')
            files.add(filename)
        print(f"  Found in {len(issues_by_type['B311'])} locations")
        print(f"  Affected files: {len(files)}")
        for f in sorted(files)[:15]:
            print(f"    - {f}")

if __name__ == '__main__':
    analyze_issues()
