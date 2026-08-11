#!/usr/bin/env python3
"""
Automated Test Migration Script - Batch 4

Migrates test files to use:
- Import guards with try/except
- openevolve_imports module
- UnifiedConfiguration instead of ParameterManager
- Pytest skip decorators where appropriate
"""

import re
import ast
from pathlib import Path
from typing import List, Tuple, Set


class TestMigrationTransformer(ast.NodeTransformer):
    """AST transformer for test file migration"""

    def __init__(self):
        self.imports_added = False
        self.has_evolution_import = False
        self.has_adversarial_import = False
        self.has_parameter_manager = False
        self.imports_to_add = set()

    def visit_ImportFrom(self, node):
        # Detect old imports
        if node.module:
            if node.module == 'evolution':
                self.has_evolution_import = True
                self.imports_to_add.add('evolution')
            elif node.module == 'adversarial':
                self.has_adversarial_import = True
                self.imports_to_add.add('adversarial')
            elif node.module == 'parameter_manager':
                self.has_parameter_manager = True
                self.imports_to_add.add('parameter_manager')

        return node


def detect_patterns(content: str) -> dict:
    """Detect migration patterns in test file"""
    patterns = {
        'has_evolution_import': bool(re.search(r'from evolution import', content)),
        'has_adversarial_import': bool(re.search(r'from adversarial import', content)),
        'has_parameter_manager': bool(re.search(r'ParameterManager', content)),
        'has_old_config_usage': bool(re.search(r'\.get_preset\(|\.validate\(|\.get_defaults\(', content)),
        'has_pytest': bool(re.search(r'import pytest|from pytest', content)),
        'has_main_block': bool(re.search(r"if __name__ == ['\"]__main__['\"]", content)),
    }
    return patterns


def generate_import_guards(patterns: dict) -> str:
    """Generate import guard code"""
    guards = []

    if patterns['has_evolution_import']:
        guards.append("""
# Import guards for evolution module
try:
    from openevolve_imports import EvolutionAPI, EVOLUTION_AVAILABLE
except ImportError:
    EVOLUTION_AVAILABLE = False
""")

    if patterns['has_adversarial_import']:
        guards.append("""
# Import guards for adversarial module
try:
    from openevolve_imports import AdversarialAPI, ADVERSARIAL_AVAILABLE
except ImportError:
    ADVERSARIAL_AVAILABLE = False
""")

    if patterns['has_parameter_manager']:
        guards.append("""
# Import guards for unified configuration
try:
    from unified_configuration import create_unified_config, UnifiedConfiguration
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
""")

    return '\n'.join(guards)


def migrate_parameter_manager_usage(content: str) -> Tuple[str, int]:
    """Migrate ParameterManager usage to UnifiedConfiguration"""
    changes = 0

    # Pattern 1: ParameterManager() -> create_unified_config()
    pattern1 = r'ParameterManager\(\)'
    if re.search(pattern1, content):
        content = re.sub(pattern1, 'create_unified_config()', content)
        changes += 1

    # Pattern 2: manager.validate() returning tuple -> ValidationResult
    # This is complex, so we'll skip for now

    # Pattern 3: manager.get_defaults() -> create_unified_config()
    pattern3 = r'(\w+)\.get_defaults\(\)'
    if re.search(pattern3, content):
        content = re.sub(pattern3, r'create_unified_config()', content)
        changes += 1

    # Pattern 4: manager.get_preset(...) -> create_unified_config(preset=...)
    pattern4 = r'(\w+)\.get_preset\(([^)]+)\)'
    if re.search(pattern4, content):
        def replacer(match):
            var = match.group(1)
            preset = match.group(2)
            return f'create_unified_config(preset={preset.strip()})'
        content = re.sub(pattern4, replacer, content)
        changes += 1

    return content, changes


def migrate_test_file(filepath: Path) -> dict:
    """Migrate a single test file"""
    print(f"\n{'='*80}")
    print(f"Migrating: {filepath.name}")
    print(f"{'='*80}")

    # Read file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return {'success': False, 'error': str(e)}

    original_content = content

    # Detect patterns
    patterns = detect_patterns(content)
    print(f"Detected patterns:")
    for key, value in patterns.items():
        if value:
            print(f"  {key}")

    # Check if migration is needed
    needs_migration = any([
        patterns['has_evolution_import'],
        patterns['has_adversarial_import'],
        patterns['has_parameter_manager'],
    ])

    if not needs_migration:
        print("  No migration needed")
        return {'success': True, 'migrated': False, 'reason': 'No old patterns found'}

    # Step 1: Add import guards at the top
    if patterns['has_evolution_import'] or patterns['has_adversarial_import'] or patterns['has_parameter_manager']:
        guards = generate_import_guards(patterns)

        # Find the position after existing imports and docstring
        lines = content.split('\n')
        insert_pos = 0

        # Skip shebang
        if lines and lines[0].startswith('#!'):
            insert_pos = 1

        # Skip encoding declaration
        if insert_pos < len(lines) and 'coding' in lines[insert_pos]:
            insert_pos += 1

        # Skip docstring
        if insert_pos < len(lines) and lines[insert_pos].strip().startswith('"""'):
            insert_pos += 1
            while insert_pos < len(lines) and '"""' not in lines[insert_pos]:
                insert_pos += 1
            if insert_pos < len(lines):
                insert_pos += 1

        # Skip empty lines
        while insert_pos < len(lines) and not lines[insert_pos].strip():
            insert_pos += 1

        # Skip imports
        while insert_pos < len(lines):
            line = lines[insert_pos].strip()
            if line.startswith('import ') or line.startswith('from '):
                insert_pos += 1
            else:
                break

        # Insert guards
        lines.insert(insert_pos, guards)
        lines.insert(insert_pos + 1, '')
        content = '\n'.join(lines)

    # Step 2: Migrate ParameterManager usage
    if patterns['has_parameter_manager']:
        content, pm_changes = migrate_parameter_manager_usage(content)
        print(f"  ParameterManager migrations: {pm_changes}")
    else:
        pm_changes = 0

    # Check if anything changed
    if content == original_content:
        print("  No changes made (already migrated or no patterns)")
        return {'success': True, 'migrated': False, 'reason': 'No changes needed'}

    # Write back
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Successfully migrated")
        return {
            'success': True,
            'migrated': True,
            'changes': pm_changes,
            'patterns': patterns
        }
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return {'success': False, 'error': str(e)}


def main():
    """Main migration function"""
    import sys

    print("="*80)
    print("BATCH 4: AUTOMATED TEST MIGRATION")
    print("="*80)

    # Read categorization
    test_dir = Path('.')

    # Get all test files
    test_files = sorted(test_dir.glob('test_*.py'))

    if not test_files:
        print("No test files found!")
        return

    print(f"\nFound {len(test_files)} test files")
    print(f"\nStarting migration...\n")

    # Statistics
    stats = {
        'total': len(test_files),
        'migrated': 0,
        'skipped': 0,
        'failed': 0,
        'changes': 0,
    }

    # Migrate each file
    for test_file in test_files:
        result = migrate_test_file(test_file)

        if result['success']:
            if result.get('migrated'):
                stats['migrated'] += 1
                stats['changes'] += result.get('changes', 0)
            else:
                stats['skipped'] += 1
        else:
            stats['failed'] += 1
            print(f"  Failed: {result.get('error', 'Unknown error')}")

    # Print summary
    print("\n" + "="*80)
    print("MIGRATION SUMMARY")
    print("="*80)
    print(f"Total files:        {stats['total']}")
    print(f"Successfully migrated: {stats['migrated']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Total changes:     {stats['changes']}")
    print("="*80)

    # Save report
    with open('test_migration_report.txt', 'w') as f:
        f.write("BATCH 4: TEST MIGRATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total files:        {stats['total']}\n")
        f.write(f"Successfully migrated: {stats['migrated']}\n")
        f.write(f"Skipped:           {stats['skipped']}\n")
        f.write(f"Failed:            {stats['failed']}\n")
        f.write(f"Total changes:     {stats['changes']}\n")

    print("\nReport saved to test_migration_report.txt")


if __name__ == '__main__':
    main()
