#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze RESE Lean 4 files for verification report.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

LEAN_DIR = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/lean4")

def extract_items(content, item_type):
    """Extract theorems, definitions, etc. from Lean code."""
    patterns = {
        'theorem': r'^theorem\s+(\w+)',
        'example': r'^example\s+(?::)?\s*',
        'def': r'^def\s+(\w+)',
        'structure': r'^structure\s+(\w+)',
        'inductive': r'^inductive\s+(\w+)',
        'abbrev': r'^abbrev\s+(\w+)',
    }
    pattern = patterns.get(item_type, '')
    if not pattern:
        return []

    matches = re.finditer(pattern, content, re.MULTILINE)
    return [(m.group(1) if m.groups() else f"anon_{item_type}", m.start())
            for m in matches]

def count_sorry(content):
    """Count admitted proofs (sorry)."""
    return len(re.findall(r'\bsorry\b', content))

def find_imports(content):
    """Extract all imports."""
    return re.findall(r'^import\s+(.+)$', content, re.MULTILINE)

def analyze_file(filepath):
    """Analyze a single Lean file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    return {
        'theorems': extract_items(content, 'theorem'),
        'examples': extract_items(content, 'example'),
        'defs': extract_items(content, 'def'),
        'structures': extract_items(content, 'structure'),
        'inductives': extract_items(content, 'inductive'),
        'abbrevs': extract_items(content, 'abbrev'),
        'sorry_count': count_sorry(content),
        'imports': find_imports(content),
        'lines': len(content.split('\n')),
    }

def main():
    print("=" * 70)
    print("RESE Lean 4 Verification Analysis")
    print("=" * 70)
    print()

    files = ['Basic.lean', 'Constraint.lean', 'Templates.lean', 'TestCases.lean', 'RESE.lean']
    all_data = {}
    total_stats = defaultdict(int)

    # Section 1: File Inventory
    print("## 1. File Inventory")
    print("-" * 70)
    for fname in files:
        fpath = LEAN_DIR / fname
        if fpath.exists():
            all_data[fname] = analyze_file(fpath)
            print(f"✓ {fname:20} {all_data[fname]['lines']:5} lines")
        else:
            print(f"✗ {fname:20} NOT FOUND")
    print()

    # Section 2: Theorem Catalog
    print("## 2. Complete Theorem Catalog")
    print("-" * 70)
    for fname in files:
        if fname not in all_data:
            continue
        data = all_data[fname]
        if data['theorems']:
            print(f"\n### {fname}")
            for name, _ in data['theorems']:
                print(f"  - theorem {name}")
                total_stats['theorems'] += 1
    print(f"\nTotal theorems: {total_stats['theorems']}")
    print()

    # Section 3: Definitions and Structures
    print("## 3. Definitions and Structures")
    print("-" * 70)
    for fname in files:
        if fname not in all_data:
            continue
        data = all_data[fname]
        print(f"\n### {fname}")
        print(f"  Structures: {len(data['structures'])}")
        for name, _ in data['structures']:
            print(f"    - structure {name}")
        print(f"  Inductives: {len(data['inductives'])}")
        for name, _ in data['inductives']:
            print(f"    - inductive {name}")
        print(f"  Definitions: {len(data['defs'])}")
        total_stats['structures'] += len(data['structures'])
        total_stats['inductives'] += len(data['inductives'])
        total_stats['defs'] += len(data['defs'])
    print()

    # Section 4: Admitted Proofs
    print("## 4. Admitted Proofs (sorry)")
    print("-" * 70)
    has_sorry = False
    for fname in files:
        if fname not in all_data:
            continue
        data = all_data[fname]
        if data['sorry_count'] > 0:
            has_sorry = True
            print(f"⚠ {fname:20} {data['sorry_count']:3} admitted proofs")
            total_stats['sorry'] += data['sorry_count']
    if not has_sorry:
        print("✓ No admitted proofs found")
    print(f"\nTotal admitted proofs: {total_stats['sorry']}")
    print()

    # Section 5: Import Dependencies
    print("## 5. Import Dependencies")
    print("-" * 70)
    all_imports = set()
    for fname in files:
        if fname not in all_data:
            continue
        data = all_data[fname]
        all_imports.update(data['imports'])
    for imp in sorted(all_imports):
        print(f"  - import {imp}")
    print()

    # Section 6: Known Issues
    print("## 6. Known Issues")
    print("-" * 70)
    print("""
1. Basic.lean:38:18 - 'from' is a reserved keyword
   - Fix: Rename parameter to 'fromId' or similar

2. Basic.lean:94:15 - Unknown constant `List.length_eraseDups`
   - Fix: Use `List.length_eraseDups` from Mathlib or prove manually

3. Constraint.lean:158 - Admitted proof in transitive_deps_partial_order

4. Templates.lean:110 - Admitted proof in acyclicity_by_topological_sort

5. TestCases.lean:142 - Admitted proof in cyclic graph detection
6. TestCases.lean:342 - Admitted proof in topological sort
7. TestCases.lean:379 - Admitted proof in integrated system
    """)

    # Section 7: Summary Statistics
    print("## 7. Summary Statistics")
    print("-" * 70)
    print(f"Total files analyzed: {len(all_data)}")
    print(f"Total theorems: {total_stats['theorems']}")
    print(f"Total structures: {total_stats['structures']}")
    print(f"Total inductives: {total_stats['inductives']}")
    print(f"Total definitions: {total_stats['defs']}")
    print(f"Total admitted proofs: {total_stats['sorry']}")
    completion_rate = (1 - total_stats['sorry'] / max(total_stats['theorems'], 1)) * 100
    print(f"Proof completion rate: {completion_rate:.1f}%")
    print()

    # Section 8: Critical Theorems Status
    print("## 8. Critical RESE Theorems Status")
    print("-" * 70)
    critical_theorems = [
        ("main_rese_theorem", "RESE.lean", "Main theorem: transformations preserve validity"),
        ("complexity_reduction_theorem", "RESE.lean", "Complexity reduction proof"),
        ("transitive_deps_partial_order", "Constraint.lean", "Dependency ordering (ADMITTED)"),
        ("acyclic_implies_topological_sort", "Constraint.lean", "Topological sort existence (ADMITTED)"),
        ("contradiction_symmetric", "Constraint.lean", "Contradiction symmetry"),
        ("checking_complexity_polynomial", "Constraint.lean", "Polynomial complexity bound"),
    ]

    for thm, fname, desc in critical_theorems:
        if fname in all_data:
            data = all_data[fname]
            found = any(name == thm for name, _ in data['theorems'])
            status = "✓ FOUND" if found else "✗ MISSING"
            print(f"{status:12} {thm:35} ({fname})")
            print(f"{'':12} {desc}")
    print()

if __name__ == '__main__':
    main()
