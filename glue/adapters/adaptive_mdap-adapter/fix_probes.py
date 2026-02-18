#!/usr/bin/env python3
"""
Fix probe scripts to add missing 'import os' statements.
"""

import re

def fix_async_probe():
    """Fix check_async_features.sh to add import os."""
    filepath = 'probes/check_async_features.sh'

    with open(filepath, 'r') as f:
        content = f.read()

    # Fix each python -c block that uses os.path but doesn't import os
    # Pattern: Find python -c blocks and add import os after the opening quote
    fixed = re.sub(
        r'(TEST_OUTPUT=\$\(python -c ")\n(import sys\n)',
        r'\1\nimport os\n\2',
        content
    )

    if fixed != content:
        with open(filepath, 'w') as f:
            f.write(fixed)
        print(f'[OK] Fixed {filepath}')
        return True
    else:
        print(f'[SKIP] No changes needed for {filepath}')
        return False

if __name__ == '__main__':
    fix_async_probe()
