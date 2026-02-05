#!/usr/bin/env python3
"""
Fix logger calls in phase1_executor.py to use kwargs instead of dict syntax.
"""

import re

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find logger calls with dict kwargs
    # self.logger.METHOD("msg", {...})
    pattern = r'(self\.logger\.(info|error|warn|debug)\s*\(\s*"([^"]+)"\s*,)\s*\{([^}]+)\}\s*\)'

    def replacer(match):
        before = match.group(1)  # self.logger.METHOD("msg",
        method = match.group(2)
        msg = match.group(3)
        dict_content = match.group(4)  # key: val, key2: val2

        # Convert dict to kwargs
        kwargs = []
        for line in dict_content.split('\n'):
            line = line.strip()
            if not line or line == ',':
                continue
            # Match 'key': value or "key": value
            m = re.match(r"['\"]([^'\"]+)['\"]:\s*(.+)", line)
            if m:
                key = m.group(1)
                val = m.group(2).rstrip(',')
                kwargs.append(f"{key}={val}")

        return f'{before}\n                {", ".join(kwargs)}\n            )'

    # Apply the replacement
    content_new = re.sub(pattern, replacer, content, flags=re.MULTILINE | re.DOTALL)

    # Also fix single-line dict patterns
    def simple_replacer(m):
        result = m.group(1) + " "
        dict_content = m.group(4)
        # Convert 'key': val to key=val
        dict_content = dict_content.replace(": ", "=").replace("'", "").replace('"', '')
        result += dict_content
        return result

    content_new = re.sub(
        r'(self\.logger\.(info|error|warn|debug)\s*\(\s*"([^"]+)"\s*,)\s*\{([^}]+)\}',
        simple_replacer,
        content_new
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content_new)

    print(f"Fixed {file_path}")

if __name__ == '__main__':
    fix_file('glue/adapters/rese-phase1/src/phase1_executor.py')
