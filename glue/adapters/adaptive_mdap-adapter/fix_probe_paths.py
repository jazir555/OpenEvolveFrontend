#!/usr/bin/env python3
"""
Fix probe scripts to use correct path for python -c blocks.
__file__ is not available in python -c context, so we need to use a simpler approach.
"""

import os
import glob

def fix_probe_paths():
    """Fix all probe files to use correct sys.path setup."""

    probe_files = glob.glob('probes/check_*.sh')

    print(f"Fixing sys.path in {len(probe_files)} probe files...")

    for filepath in probe_files:
        print(f"  Checking {os.path.basename(filepath)}...")

        with open(filepath, 'r') as f:
            content = f.read()

        original = content

        # Replace the complex path logic with simple approach
        # Old: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
        # New: sys.path.insert(0, '.')

        # Pattern 1: One-liner with if/else
        pattern1 = r"sys\.path\.insert\(0, os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), '\.\./src'\)\) if os\.path\.exists\('\.\./src'\) else 'src'\)"
        replacement1 = "sys.path.insert(0, '.')"

        content = re.sub(pattern1, replacement1, content)

        # Pattern 2: Multi-line version
        pattern2 = r"sys\.path\.insert\(0,\s*os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\),\s*'\.\./src'\)\)\s*if\s*os\.path\.exists\('\.\./src'\)\s*else\s*'src'\)"
        replacement2 = "sys.path.insert(0, '.')"

        content = re.sub(pattern2, replacement2, content)

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"    [FIXED] {os.path.basename(filepath)}")
        else:
            print(f"    [OK] Already correct")

    print()
    print("Done!")

if __name__ == '__main__':
    import re
    fix_probe_paths()
