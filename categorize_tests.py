#!/usr/bin/env python3
"""Categorize test files by migration priority"""
import os
import re
from pathlib import Path

# Test file patterns
HIGH_PRIORITY_PATTERNS = [
    r'test_integration',
    r'test_openevolve',
    r'test_.*_complete',
    r'test_.*_comprehensive',
    r'test_.*_end_to_end',
    r'test_.*_e2e',
    r'test_evolution_.*',
    r'test_adversarial_.*',
    r'test_decomposition_.*',
    r'test_leanaide_.*',
    r'test_workflow',
    r'test_team_.*',
]

MEDIUM_PRIORITY_PATTERNS = [
    r'test_.*_config',
    r'test_.*_adapter',
    r'test_.*_engine',
    r'test_.*_functionality',
    r'test_.*_integration',
    r'test_backward_compatibility',
    r'test_imports',
    r'test_.*_bugs',
    r'test_.*_fixes',
    r'test_.*_performance',
    r'test_.*_security',
]

LOWER_PRIORITY_PATTERNS = [
    r'test_.*_simple',
    r'test_.*_demo',
    r'test_.*_legacy',
    r'test_.*_deprecated',
    r'test_reproduction',
    r'test_proofgpt',
]

def categorize_test(filename):
    """Categorize a test file by priority"""
    for pattern in HIGH_PRIORITY_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return 'HIGH'
    
    for pattern in MEDIUM_PRIORITY_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return 'MEDIUM'
    
    for pattern in LOWER_PRIORITY_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return 'LOWER'
    
    return 'MEDIUM'  # Default to medium

def main():
    test_files = list(Path('.').glob('test_*.py'))
    
    categories = {'HIGH': [], 'MEDIUM': [], 'LOWER': []}
    
    for test_file in sorted(test_files):
        priority = categorize_test(test_file.name)
        categories[priority].append(test_file.name)
    
    print("=" * 80)
    print("TEST FILE CATEGORIZATION")
    print("=" * 80)
    
    for priority in ['HIGH', 'MEDIUM', 'LOWER']:
        print(f"\n{priority} PRIORITY ({len(categories[priority])} files):")
        print("-" * 80)
        for filename in categories[priority]:
            print(f"  {filename}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(test_files)} test files")
    print(f"HIGH PRIORITY: {len(categories['HIGH'])}")
    print(f"MEDIUM PRIORITY: {len(categories['MEDIUM'])}")
    print(f"LOWER PRIORITY: {len(categories['LOWER'])}")
    print("=" * 80)
    
    # Save to file
    with open('test_categorization.txt', 'w') as f:
        for priority in ['HIGH', 'MEDIUM', 'LOWER']:
            f.write(f"\n{priority} PRIORITY ({len(categories[priority])} files):\n")
            f.write("-" * 80 + "\n")
            for filename in categories[priority]:
                f.write(f"  {filename}\n")
    
    print("\nSaved to test_categorization.txt")

if __name__ == '__main__':
    main()
