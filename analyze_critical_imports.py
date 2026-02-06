#!/usr/bin/env python3
"""Analyze critical import issues in the main project."""

import json

with open('import_issues.json') as f:
    issues = json.load(f)

# Filter for main project issues only (exclude core-projects subdirs)
main_project_issues = []
for issue in issues:
    files = issue['files']
    # Check if any file is in the main project (not in core-projects)
    main_files = [f for f in files if not f.startswith('core-projects') and not f.startswith('bubblelabs_nodes')]
    if main_files:
        issue['main_files'] = main_files
        main_project_issues.append(issue)

print(f"=== CRITICAL IMPORT ISSUES ({len(main_project_issues)}) ===\n")

for issue in main_project_issues:
    print(f"Module: {issue['module']}")
    print(f"  Referenced in:")
    for f in issue['main_files'][:5]:
        print(f"    - {f}")
    if len(issue['main_files']) > 5:
        print(f"    ... and {len(issue['main_files']) - 5} more files")
    print()

# Categorize the issues
print("\n=== CATEGORIZATION ===\n")

# Check for common patterns
categories = {
    'z3_': [],
    'roma_': [],
    'gauntlet_': [],
    'solution_': [],
    'sovereign_': [],
    'openevolve_': [],
    'leanaide_': [],
    'unified_': [],
    'other': []
}

for issue in main_project_issues:
    module = issue['module']
    found = False
    for prefix in categories.keys():
        if prefix != 'other' and module.startswith(prefix):
            categories[prefix].append(issue)
            found = True
            break
    if not found:
        categories['other'].append(issue)

for prefix, issues_list in categories.items():
    if issues_list:
        print(f"{prefix}: {len(issues_list)} issues")
        for i in issues_list[:3]:
            print(f"  - {i['module']}")
        print()
