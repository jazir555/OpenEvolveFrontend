#!/usr/bin/env python3
"""
Fix probe scripts to use correct absolute path for src.
The probes need to add the ADAPTER ROOT to sys.path, not the current directory.
"""

import os
import glob

def fix_probe_paths_final():
    """Fix all probe files to use correct absolute path."""

    probe_files = glob.glob('probes/check_*.sh')

    print(f"Fixing sys.path in {len(probe_files)} probe files...")
    print()

    for filepath in probe_files:
        filename = os.path.basename(filepath)
        print(f"  {filename}...")

        with open(filepath, 'r') as f:
            content = f.read()

        original = content

        # Replace sys.path.insert(0, '.') with absolute path to adapter root
        # The probes are in probes/ subdirectory, so we need to go up one level
        pattern = r"sys\.path\.insert\(0, '\.'\)"
        replacement = r"sys.path.insert(0, os.path.abspath('..'))"

        # Count replacements
        count_before = len(re.findall(pattern, content))
        content = re.sub(pattern, replacement, content)
        count_after = len(re.findall(replacement, content))

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"    [FIXED] Replaced {count_before} occurrences")
        else:
            print(f"    [OK] No changes needed")

    print()
    print("Done! Probes now use os.path.abspath('..') to reference adapter root.")

if __name__ == '__main__':
    import re
    fix_probe_paths_final()
