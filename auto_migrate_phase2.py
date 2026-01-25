#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Migration Script for Phase 2 Files

Applies Phase 1 migration patterns to all remaining files.
"""

import os
import re
import sys
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


FILES_TO_MIGRATE = [
    "test_adversarial_comprehensive.py",
    "test_adversarial_evolution_complete.py",
    "test_adversarial_simple.py",
    "test_backward_compatibility.py",
    "test_critical_blockers_resolved.py",
    "test_error_handling.py",
    "test_evolution_adversarial_basic.py",
    "test_evolution_comprehensive.py",
    "test_integration_openevolve.py",
    "test_leanaide_evolution_mdap.py",
    "test_missing_dependencies.py",
    "test_openevolve_integration.py",
    "test_phase1_team_integration.py",
    "test_session_state_removal.py",
    "test_sidebar_parameter_integration.py",
    "test_team_system_working.py",
    "test_unified_config_functionality_clean.py",
    "test_unified_config_integration.py",
    "compare_parameter_managers.py",
    "compare_parameter_managers_simple.py",
    "compare_simple_ascii.py",
    "demo_adversarial_maker.py",
    "demo_evolution_maker.py",
]


def migrate_file(filepath: str) -> dict:
    """Migrate a single file"""
    if not os.path.exists(filepath):
        return {"success": False, "error": "File not found", "changes": 0}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = 0

        # Pattern 1: Replace "from evolution import" with centralized imports
        if re.search(r'from evolution import', content):
            # Add import guard at top if not present
            if 'from openevolve_imports import' not in content:
                # Find first import line
                import_match = re.search(r'\nimport ', content)
                if import_match:
                    insert_pos = import_match.start()
                    guard_block = '''# OpenEvolve Import Centralizer (PHASE 2 MIGRATION)
try:
    from openevolve_imports import EvolutionAPI, EVOLUTION_AVAILABLE
except ImportError:
    EVOLUTION_AVAILABLE = False
    EvolutionAPI = None

'''
                    content = content[:insert_pos] + guard_block + content[insert_pos:]
                    changes += 1

            # Replace old imports
            old_evolution_imports = re.findall(r'from evolution import[^\n]+\n(?:\s+[^\n]+\n)*', content)
            for old_import in old_evolution_imports:
                # Extract function names
                functions = re.findall(r'(\w+)', old_import)
                if functions:
                    # Create new import
                    new_import_lines = [f"    {func} = EvolutionAPI.{func}" for func in functions[2:]]  # Skip 'from', 'evolution'
                    if new_import_lines:
                        new_import_block = "if EVOLUTION_AVAILABLE:\n" + "\n".join(new_import_lines) + "\nelse:\n    # Fallback placeholders\n" + "\n".join([f"    {func} = None" for func in functions[2:]]) + "\n"
                        content = content.replace(old_import, new_import_block)
                        changes += 1

        # Pattern 2: Replace "from adversarial import" with centralized imports
        if re.search(r'from adversarial import', content):
            if 'from openevolve_imports import AdversarialAPI' not in content:
                import_match = re.search(r'\nimport ', content)
                if import_match:
                    insert_pos = import_match.start()
                    guard_block = '''# OpenEvolve Import Centralizer (PHASE 2 MIGRATION)
try:
    from openevolve_imports import AdversarialAPI, ADVERSARIAL_AVAILABLE
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    AdversarialAPI = None

'''
                    content = content[:insert_pos] + guard_block + content[insert_pos:]
                    changes += 1

            old_adversarial_imports = re.findall(r'from adversarial import[^\n]+\n(?:\s+[^\n]+\n)*', content)
            for old_import in old_adversarial_imports:
                functions = re.findall(r'(\w+)', old_import)
                if functions:
                    new_import_lines = [f"    {func} = AdversarialAPI.{func}" for func in functions[2:]]
                    if new_import_lines:
                        new_import_block = "if ADVERSARIAL_AVAILABLE:\n" + "\n".join(new_import_lines) + "\nelse:\n    # Fallback placeholders\n" + "\n".join([f"    {func} = None" for func in functions[2:]]) + "\n"
                        content = content.replace(old_import, new_import_block)
                        changes += 1

        # Pattern 3: Replace ParameterManager usage
        if 'from parameter_manager import ParameterManager' in content:
            content = content.replace(
                'from parameter_manager import ParameterManager',
                '# PHASE 2 MIGRATION: ParameterManager deprecated\n# from parameter_manager import ParameterManager  # OLD PATTERN\nParameterManager = None  # Placeholder'
            )
            changes += 1

        # Write back if changed
        if content != original_content:
            # Create backup
            backup_path = filepath + ".phase2_backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Write migrated content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            return {"success": True, "changes": changes, "backup": backup_path}
        else:
            return {"success": True, "changes": 0, "backup": None}

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return {"success": False, "error": str(e), "changes": 0}


def main():
    """Migrate all files"""
    print("=" * 80)
    print("PHASE 2: AUTOMATED MIGRATION")
    print("=" * 80)

    total_files = len(FILES_TO_MIGRATE)
    success_count = 0
    total_changes = 0

    for i, filename in enumerate(FILES_TO_MIGRATE, 1):
        print(f"\n[{i}/{total_files}] Migrating {filename}...")

        result = migrate_file(filename)

        if result["success"]:
            if result["changes"] > 0:
                print(f"  [OK] Migrated with {result['changes']} changes")
                if result.get("backup"):
                    print(f"  [INFO] Backup: {result['backup']}")
                success_count += 1
                total_changes += result["changes"]
            else:
                print(f"  [SKIP] No changes needed")
        else:
            print(f"  [ERROR] {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)
    print(f"Total files: {total_files}")
    print(f"Successfully migrated: {success_count}")
    print(f"Total changes: {total_changes}")
    print("\n[OK] Migration complete!")


if __name__ == "__main__":
    main()
