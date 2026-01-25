#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Migration Progress Report

This script generates a comprehensive markdown report showing:
1. Batch 1 progress: Import replacements
2. Batch 2 progress: Adapter integration
3. Batch 3 progress: Configuration migration
4. Files updated and lines reduced
5. Overall completion status

Usage:
    python migration_report.py [output_file]

Arguments:
    output_file: Path to save the report (default: MIGRATION_REPORT.md)
"""

import os
import re
import sys
import io
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Migration batch definitions
BATCHES = {
    'batch1': {
        'name': 'Batch 1: Import Replacements',
        'description': 'Replace all try/except import patterns with openevolve_imports',
        'files': [
            'test_adversarial_comprehensive.py',
            'test_adversarial_evolution_complete.py',
            'test_adversarial_mdap_mcts_complete.py',
            'test_all_imports.py',
            'test_bubblelabs_complete_integration.py',
            'test_bubblelabs_complete_validation.py',
            'test_bubblelabs_comprehensive.py',
            'test_bubblelabs_edge_cases.py',
            'test_evolution_comprehensive.py',
            'test_final_integration.py',
            'final_integration_test.py',
            'conftest.py',
        ],
    },
    'batch2': {
        'name': 'Batch 2: Adapter Integration',
        'description': 'Create adapter wrappers for all major modules',
        'files': [
            'evolution_adapter.py',
            'adversarial_adapter.py',
            'maker_engine_adapter.py',
            'mdap_engine_adapter.py',
            'decomposition_adapter.py',
        ],
    },
    'batch3': {
        'name': 'Batch 3: Configuration Migration',
        'description': 'Migrate to unified configuration system',
        'files': [
            'base_configuration.py',
            'configuration_manager.py',
            'configuration_schema.py',
        ],
    },
}


def check_file_migration_status(filepath: Path) -> Dict[str, any]:
    """
    Check migration status of a single file.

    Returns:
        Dictionary with migration status information
    """
    status = {
        'file': filepath.name,
        'exists': filepath.exists(),
        'uses_openevolve_imports': False,
        'has_old_patterns': False,
        'line_count': 0,
        'import_count': 0,
        'status': 'NOT_STARTED',
    }

    if not status['exists']:
        return status

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            status['line_count'] = len(lines)
            content = ''.join(lines)

        # Check for openevolve_imports usage
        if re.search(r'from\s+openevolve_imports\s+import', content):
            status['uses_openevolve_imports'] = True
            status['import_count'] = len(re.findall(r'from\s+openevolve_imports\s+import', content))

        # Check for old patterns
        old_patterns = [
            r'try:\s*from\s+evolution\s+import',
            r'try:\s*from\s+adversarial\s+import',
            r'except\s+ImportError:\s*\w+_AVAILABLE\s*=\s*False',
        ]
        for pattern in old_patterns:
            if re.search(pattern, content):
                status['has_old_patterns'] = True
                break

        # Determine status
        if status['uses_openevolve_imports'] and not status['has_old_patterns']:
            status['status'] = 'COMPLETE'
        elif status['uses_openevolve_imports'] and status['has_old_patterns']:
            status['status'] = 'PARTIAL'
        elif status['has_old_patterns']:
            status['status'] = 'IN_PROGRESS'
        else:
            status['status'] = 'NOT_STARTED'

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        status['error'] = str(e)

    return status


def analyze_batch(batch_name: str, batch_info: Dict, cwd: Path) -> Dict[str, any]:
    """Analyze migration status for a batch"""
    results = {
        'name': batch_info['name'],
        'description': batch_info['description'],
        'total_files': len(batch_info['files']),
        'files_completed': 0,
        'files_in_progress': 0,
        'files_not_started': 0,
        'files_missing': 0,
        'total_lines': 0,
        'uses_openevolve_imports': 0,
        'file_details': [],
    }

    for filename in batch_info['files']:
        filepath = cwd / filename
        status = check_file_migration_status(filepath)
        results['file_details'].append(status)
        results['total_lines'] += status['line_count']

        if not status['exists']:
            results['files_missing'] += 1
        elif status['status'] == 'COMPLETE':
            results['files_completed'] += 1
        elif status['status'] in ['PARTIAL', 'IN_PROGRESS']:
            results['files_in_progress'] += 1
        else:
            results['files_not_started'] += 1

        if status['uses_openevolve_imports']:
            results['uses_openevolve_imports'] += 1

    # Calculate completion percentage
    if results['total_files'] > 0:
        results['completion_percent'] = (
            (results['files_completed'] / results['total_files']) * 100
        )
    else:
        results['completion_percent'] = 0

    return results


def generate_report() -> str:
    """Generate markdown migration progress report"""
    cwd = Path.cwd()

    # Analyze all batches
    batch_results = {}
    for batch_name, batch_info in BATCHES.items():
        batch_results[batch_name] = analyze_batch(batch_name, batch_info, cwd)

    # Calculate overall statistics
    total_files = sum(b['total_files'] for b in batch_results.values())
    total_completed = sum(b['files_completed'] for b in batch_results.values())
    total_in_progress = sum(b['files_in_progress'] for b in batch_results.values())
    total_lines = sum(b['total_lines'] for b in batch_results.values())

    overall_completion = (total_completed / total_files * 100) if total_files > 0 else 0

    # Generate report
    report_lines = [
        "# Migration Progress Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total Files:** {total_files}",
        f"- **Completed:** {total_completed} ({overall_completion:.1f}%)",
        f"- **In Progress:** {total_in_progress}",
        f"- **Total Lines:** {total_lines:,}",
        "",
    ]

    # Add overall progress bar
    progress_bars = int(overall_completion / 10)
    report_lines.extend([
        "### Overall Progress",
        "",
        f"{'█' * progress_bars}{'░' * (10 - progress_bars)} {overall_completion:.0f}%",
        "",
        "---",
        "",
    ])

    # Add batch details
    for batch_name, batch_info in BATCHES.items():
        result = batch_results[batch_name]
        report_lines.extend([
            f"## {result['name']}",
            "",
            f"{result['description']}",
            "",
            f"- **Files:** {result['files_completed']}/{result['total_files']} completed",
            f"- **Progress:** {result['completion_percent']:.1f}%",
            f"- **Total Lines:** {result['total_lines']:,}",
            "",
        ])

        # Add progress bar
        progress_bars = int(result['completion_percent'] / 10)
        report_lines.append(f"{'█' * progress_bars}{'░' * (10 - progress_bars)} {result['completion_percent']:.0f}%")
        report_lines.append("")

        # Add file details table
        report_lines.extend([
            "### File Details",
            "",
            "| File | Status | Lines | Uses openevolve_imports |",
            "|------|--------|-------|-------------------------|",
        ])

        for detail in result['file_details']:
            status_icon = {
                'COMPLETE': '✓',
                'PARTIAL': '⚠',
                'IN_PROGRESS': '🔄',
                'NOT_STARTED': '✗',
                'MISSING': '❌',
            }.get(detail.get('status', 'NOT_STARTED'), '?')

            uses_import = '✓' if detail.get('uses_openevolve_imports') else '✗'
            report_lines.append(
                f"| {detail['file']} | {status_icon} {detail.get('status', 'UNKNOWN')} | {detail['line_count']} | {uses_import} |"
            )

        report_lines.extend(["", "---", ""])

    # Add recommendations section
    report_lines.extend([
        "## Recommendations",
        "",
    ])

    if batch_results['batch1']['completion_percent'] < 100:
        report_lines.extend([
            "### Priority 1: Complete Batch 1",
            "",
            f"- {batch_results['batch1']['files_not_started']} files not started",
            f"- {batch_results['batch1']['files_in_progress']} files in progress",
            "- Focus on replacing old try/except patterns with openevolve_imports",
            "",
        ])

    if batch_results['batch2']['files_missing'] > 0:
        report_lines.extend([
            "### Priority 2: Create Batch 2 Adapters",
            "",
            f"- {batch_results['batch2']['files_missing']} adapter files missing",
            "- Create adapter wrappers for each major module",
            "- Ensure adapters use openevolve_imports",
            "",
        ])

    report_lines.extend([
        "## Next Steps",
        "",
        "1. Run `python validate_batch1_imports.py` to check migration status",
        "2. Run `python test_import_functionality.py` to verify imports work",
        "3. Run `python validate_syntax.py` to check syntax of all files",
        "4. Review and update any remaining files with old patterns",
        "",
        "---",
        "",
        "*This report was auto-generated by migration_report.py*",
    ])

    return "\n".join(report_lines)


def main():
    """Main function to generate and save report"""
    # Parse arguments
    output_file = sys.argv[1] if len(sys.argv) > 1 else "MIGRATION_REPORT.md"
    output_path = Path(output_file)

    # Generate report
    print("Generating migration progress report...")
    report_content = generate_report()

    # Save report
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n✓ Report saved to: {output_path.absolute()}")
        print(f"  Size: {len(report_content):,} characters")

        # Print summary
        print("\n" + "="*60)
        print("MIGRATION REPORT SUMMARY")
        print("="*60)

        # Extract key stats
        lines = report_content.split('\n')
        for i, line in enumerate(lines):
            if 'Total Files:' in line:
                print(line.strip())
            elif 'Completed:' in line:
                print(line.strip())
            elif 'Total Lines:' in line:
                print(line.strip())
                break

        print("="*60 + "\n")

        return 0

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n✗ Error saving report: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
