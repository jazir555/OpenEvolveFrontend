#!/usr/bin/env python3
"""
Fix all probe scripts to add missing 'import os' statements.
"""

import os
import glob
import re

def fix_probe_file(filepath):
    """Fix a single probe file."""
    print(f"Checking {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Fix: Add 'import os' right after 'python -c "' if it's missing
    # Pattern: Look for TEST_OUTPUT=$(python -c " followed by content that uses os.
    # Add 'import os' after the opening quote if not already present

    # Split by python -c blocks
    lines = content.split('\n')
    fixed_lines = []
    in_python_block = False
    has_import_os = False

    for i, line in enumerate(lines):
        fixed_lines.append(line)

        # Check if we're starting a python -c block
        if 'TEST_OUTPUT=$(python -c "' in line:
            in_python_block = True
            has_import_os = False
            # Look ahead to see if import os exists
            for j in range(i+1, min(i+5, len(lines))):
                if 'import os' in lines[j]:
                    has_import_os = True
                    break
                if lines[j].strip() and not lines[j].strip().startswith('import'):
                    # Non-import line found, stop looking
                    break

            # If no import os but we see os.path usage, add it
            if not has_import_os:
                # Check if os is used in this block
                for j in range(i+1, min(i+20, len(lines))):
                    if '")' in lines[j]:  # End of python block
                        break
                    if 'os.' in lines[j]:
                        # os is used but not imported - add import os
                        fixed_lines.append('import os')
                        print(f"  [FIX] Added 'import os' after line {i+1}")
                        break

        # Reset at end of python block
        if in_python_block and '")' in line:
            in_python_block = False

    fixed_content = '\n'.join(fixed_lines)

    if fixed_content != original:
        with open(filepath, 'w') as f:
            f.write(fixed_content)
        print(f"  [OK] Fixed {filepath}")
        return True
    else:
        print(f"  [SKIP] No changes needed")
        return False

def main():
    """Fix all probe files."""
    probe_files = glob.glob('probes/check_*.sh')

    print(f"Found {len(probe_files)} probe files")
    print()

    fixed_count = 0
    for filepath in probe_files:
        if fix_probe_file(filepath):
            fixed_count += 1
        print()

    print(f"Total files fixed: {fixed_count}/{len(probe_files)}")

if __name__ == '__main__':
    main()
