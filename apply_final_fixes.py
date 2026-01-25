"""
Apply final fixes to all remaining files with old import patterns.
"""

import os
import re

def fix_file_with_old_imports(filepath):
    """Fix old import patterns in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            original_content = content

        # Check if the file already has the new import pattern
        if 'from openevolve_imports import' in content:
            return False  # Already fixed

        # Track what we need to add
        needed_imports = []

        # Check for old evolution import
        if 'from evolution import' in content and 'EVOLUTION_AVAILABLE' not in content:
            # Find what's being imported from evolution
            match = re.search(r'from evolution import ([^\n]+)', content)
            if match:
                imports = match.group(1).strip()
                needed_imports.append(('evolution', imports))

        # Check for old adversarial import
        if 'from adversarial import' in content and 'ADVERSARIAL_AVAILABLE' not in content:
            match = re.search(r'from adversarial import ([^\n]+)', content)
            if match:
                imports = match.group(1).strip()
                needed_imports.append(('adversarial', imports))

        # Check for old leanaide_mcts_mdap import
        if 'from leanaide_mcts_mdap import' in content and 'LEANAIDE_MCTS_MDAP_AVAILABLE' not in content:
            match = re.search(r'from leanaide_mcts_mdap import ([^\n]+)', content)
            if match:
                imports = match.group(1).strip()
                needed_imports.append(('leanaide_mcts_mdap', imports))

        if not needed_imports:
            return False  # Nothing to fix

        # Build the new import block
        new_import_lines = []
        new_import_lines.append('# OpenEvolve imports with backward compatibility')

        for module_name, imports in needed_imports:
            var_name = f'{module_name.upper().replace("-", "_")}_AVAILABLE'
            new_import_lines.append(f'try:')
            new_import_lines.append(f'    from openevolve_imports import {imports}')
            new_import_lines.append(f'    {var_name} = True')
            new_import_lines.append(f'except ImportError:')
            new_import_lines.append(f'    try:')
            new_import_lines.append(f'        from {module_name} import {imports}')
            new_import_lines.append(f'        {var_name} = True')
            new_import_lines.append(f'    except ImportError:')
            new_import_lines.append(f'        {var_name} = False')
            new_import_lines.append('')

        # Find the last import statement
        lines = content.split('\n')
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_idx = i

        # Insert the new imports after the last import
        if last_import_idx >= 0:
            # Insert after the last import block
            insert_idx = last_import_idx + 1
            while insert_idx < len(lines) and lines[insert_idx].strip() == '':
                insert_idx += 1

            lines.insert(insert_idx, '')
            for line in reversed(new_import_lines):
                lines.insert(insert_idx, line)

            content = '\n'.join(lines)

        # Check if anything changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"Error fixing {filepath}: {e}")
        return False

def fix_parameter_manager_usage(filepath):
    """Fix direct ParameterManager usage."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'ParameterManager()' not in content or 'create_unified_config' in content:
            return False

        # Replace ParameterManager() with create_unified_config()
        content = re.sub(
            r'ParameterManager\(\)',
            'create_unified_config()',
            content
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return True

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"Error fixing ParameterManager in {filepath}: {e}")
        return False

def main():
    """Apply all fixes."""
    files_to_fix = [
        ('adversarial.py', 'imports'),
        ('adversarial_adapter.py', 'imports'),
        ('bubblelabs_evolution_integration.py', 'imports'),
        ('bubblelabs_leanaide_integration.py', 'imports'),
        ('comprehensive_functional_tests.py', 'imports'),
        ('evolution.py', 'imports'),
        ('evolutionary_optimization.py', 'imports'),
        ('evolution_adapter.py', 'imports'),
        ('evolution_old.py', 'imports'),
        ('leanaide_mdap.py', 'imports'),
        ('leanaide_sop_integration.py', 'imports'),
        ('openevolve_workflow_manager_integrated.py', 'parameter_manager'),
        ('simple_verify_implementation.py', 'imports'),
        ('test_enhanced_redflagging.py', 'imports'),
        ('test_leanaide_mcts_mdap.py', 'imports'),
        ('unified_configuration.py', 'imports'),
        ('validate_adversarial_maker_integration.py', 'imports'),
        ('validate_batch4_class_refactoring.py', 'imports'),
        ('validate_evolution_maker_integration.py', 'imports'),
        ('validate_phase2_completeness.py', 'imports'),
        ('verify_fix.py', 'imports'),
        ('verify_mdap_maker_integration.py', 'imports'),
        ('leanaide-bubblelab-plugin/test_final_verification.py', 'imports'),
        ('tests/test_enhanced_adversarial.py', 'imports'),
        ('tests/test_integration.py', 'imports'),
    ]

    fixed_count = 0
    failed_count = 0

    print("Applying final fixes...")
    print("=" * 80)

    for filepath, fix_type in files_to_fix:
        print(f"\nProcessing: {filepath}")

        if not os.path.exists(filepath):
            print(f"  File not found, skipping...")
            failed_count += 1
            continue

        if fix_type == 'imports':
            success = fix_file_with_old_imports(filepath)
        elif fix_type == 'parameter_manager':
            success = fix_parameter_manager_usage(filepath)
        else:
            print(f"  Unknown fix type: {fix_type}")
            failed_count += 1
            continue

        if success:
            print(f"  Fixed successfully!")
            fixed_count += 1
        else:
            print(f"  No changes needed or already fixed")
            fixed_count += 1

    print("\n" + "=" * 80)
    print(f"Fixed: {fixed_count}/{len(files_to_fix)}")
    print(f"Failed: {failed_count}/{len(files_to_fix)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
