"""Fix domain optimizer imports to use local enums instead of unified.config."""

files_to_fix = [
    "core-projects/openevolve/openevolve/domain/finance_optimizer.py",
    "core-projects/openevolve/openevolve/domain/trading_optimizer.py",
    "core-projects/openevolve/openevolve/domain/engineering_optimizer.py",
    "core-projects/openevolve/openevolve/domain/pharma_optimizer.py",
    "core-projects/openevolve/openevolve/domain/web_design_optimizer.py",
]

for filepath in files_to_fix:
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check if it needs fixing
        if 'from . import EvolutionMode, DomainType' in content:
            print(f"Skipping {filepath} - already fixed")
            continue

        # Find and replace the import block
        old_import = """from ..unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,"""

        new_import = """from ..unified.config import (
    UnifiedEvolutionConfig,"""

        if old_import.split('\n')[0] in content:
            # Replace the import
            content = content.replace(old_import, new_import)

            # Find the closing ) of the from ..unified.config import and add local import after it
            lines = content.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                new_lines.append(lines[i])
                if lines[i].strip().startswith('from ..unified.config import'):
                    # Skip lines until we find the closing )
                    while i < len(lines) and ')' not in lines[i]:
                        i += 1
                        new_lines.append(lines[i])
                    # Add local import after the closing )
                    new_lines.append('# Import enums from local __init__.py to avoid conflicts with glue layer')
                    new_lines.append('from . import EvolutionMode, DomainType')
                i += 1

            content = '\n'.join(new_lines)

            with open(filepath, 'w') as f:
                f.write(content)

            print(f"Fixed {filepath}")
        else:
            print(f"Skipping {filepath} - no matching import pattern")

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

print("Done!")
