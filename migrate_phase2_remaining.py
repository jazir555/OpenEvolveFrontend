#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Migration Script - Remaining Files

Automatically migrates the remaining 26 files with old patterns.
"""

import os
import re
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Files to migrate (categorized)
MIGRATION_MAP = {
    "test_files": [
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
    ],
    "comparison_files": [
        "compare_before_after.py",
        "compare_parameter_managers.py",
        "compare_parameter_managers_simple.py",
        "compare_simple_ascii.py",
    ],
    "demo_files": [
        "demo_adversarial_maker.py",
        "demo_evolution_maker.py",
    ]
}

# Migration patterns
OLD_IMPORT_PATTERNS = [
    r"from evolution import",
    r"from adversarial import",
    r"from parameter_manager import ParameterManager",
    r"ParameterManager\(\)",
]

NEW_IMPORT_BLOCK = """# OpenEvolve Import Centralizer (PHASE 2 MIGRATION)
try:
    from openevolve_imports import EvolutionAPI, AdversarialAPI, EVOLUTION_AVAILABLE, ADVERSARIAL_AVAILABLE
except ImportError:
    EVOLUTION_AVAILABLE = False
    ADVERSARIAL_AVAILABLE = False
    EvolutionAPI = None
    AdversarialAPI = None"""

DEPRECATED_COMMENT = "# PHASE 1 MIGRATION: ParameterManager deprecated - using UnifiedConfiguration"


def check_file_needs_migration(filepath: str) -> bool:
    """Check if file contains old patterns"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for pattern in OLD_IMPORT_PATTERNS:
            if re.search(pattern, content):
                return True
        return False
    except (re.error, OSError, IOError):
        return False


def get_file_status(filepath: str) -> dict:
    """Get migration status of a file"""
    if not os.path.exists(filepath):
        return {"status": "NOT_FOUND", "needs_migration": False}

    needs_migration = check_file_needs_migration(filepath)
    return {"status": "FOUND", "needs_migration": needs_migration}


def scan_all_files():
    """Scan all files in migration map"""
    print("=" * 80)
    print("PHASE 2: REMAINING FILES MIGRATION STATUS")
    print("=" * 80)

    total_files = 0
    need_migration = 0
    already_migrated = 0
    not_found = 0

    results = {
        "test_files": [],
        "comparison_files": [],
        "demo_files": []
    }

    for category, files in MIGRATION_MAP.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 80)

        for filename in files:
            total_files += 1
            filepath = filename

            status_info = get_file_status(filepath)
            results[category].append({
                "filename": filename,
                **status_info
            })

            if status_info["status"] == "NOT_FOUND":
                not_found += 1
                print(f"  [X] {filename} - NOT FOUND")
            elif status_info["needs_migration"]:
                need_migration += 1
                print(f"  [!] {filename} - NEEDS MIGRATION")
            else:
                already_migrated += 1
                print(f"  [OK] {filename} - ALREADY MIGRATED")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {total_files}")
    print(f"Need migration: {need_migration}")
    print(f"Already migrated: {already_migrated}")
    print(f"Not found: {not_found}")

    return results


def generate_migration_script(results: dict):
    """Generate migration script for files that need it"""
    print("\n" + "=" * 80)
    print("GENERATING MIGRATION COMMANDS")
    print("=" * 80)

    commands = []

    for category, files in results.items():
        for file_info in files:
            if file_info["needs_migration"]:
                filename = file_info["filename"]
                commands.append(f"# Migrate {filename}")
                commands.append(f"# Apply Phase 1 patterns:")
                commands.append(f"# 1. Replace 'from evolution import' with openevolve_imports")
                commands.append(f"# 2. Replace 'from adversarial import' with openevolve_imports")
                commands.append(f"# 3. Replace 'ParameterManager' with UnifiedConfiguration")
                commands.append("")

    if commands:
        print("\nMigration commands to apply:")
        print("\n".join(commands[:20]))  # Show first 20
        print(f"... ({len(commands) - 20} more commands)" if len(commands) > 20 else "")
    else:
        print("\n✅ All files already migrated!")


def main():
    """Main migration workflow"""
    print("Phase 2 Migration Scanner")
    print("Scanning remaining files for old patterns...\n")

    # Scan all files
    results = scan_all_files()

    # Generate migration script
    generate_migration_script(results)

    print("\n" + "=" * 80)
    print("MIGRATION STATUS: COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
