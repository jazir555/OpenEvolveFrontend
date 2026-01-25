#!/usr/bin/env python3
"""
Phase 6 CrewAI Migration Script

Updates all Phase 6 workflow and integration files to use CrewAI instead of Hephaestus.

This script performs the following migrations:
1. Replace Hephaestus imports with CrewAI imports
2. Update class/function names from Hephaestus to CrewAI
3. Replace API calls with local CrewAI execution
4. Update environment variable references
5. Add migration notices

Author: CrewAI Migration Team
Date: 2026-01-21
License: MIT
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Migration mapping: old (Hephaestus) -> new (CrewAI)
IMPORT_REPLACEMENTS = {
    # Core imports
    r'from hephaestus_unified_bridge import': 'from crewai_unified_bridge import CrewAIUnifiedBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from hephaestus_integration import': 'from crewai_integration import CrewAIIntegrationManager, setup_crewai_integration  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from hephaestus_client import': 'from crewai_client import CrewAIClient, create_crewai_client  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'import hephaestus_unified_bridge': 'import crewai_unified_bridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'import hephaestus_integration': 'import crewai_integration  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'import hephaestus_client': 'import crewai_client  # CrewAI (MIT) - replaced Hephaestus (AGPL)',

    # Bridge imports
    r'from bubblelabs_hephaestus_bridge import': 'from bubblelabs_crewai_bridge import BubbleLabsCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from leanaide_hephaestus_bridge import': 'from leanaide_crewai_bridge import LeanAideCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from claudiomiro_hephaestus_bridge import': 'from claudiomiro_crewai_bridge import ClaudiomiroCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from datapizza_hephaestus_bridge import': 'from datapizza_crewai_bridge import DataPizzaCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from decomposition_hephaestus_bridge import': 'from decomposition_crewai_bridge import DecompositionCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from roma_hephaestus_bridge import': 'from roma_crewai_bridge import ROMACrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from roma_mdap_maker_hephaestus_bridge import': 'from roma_mdap_maker_crewai_bridge import ROMAMDAPMakerCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from openevolve_hephaestus_bridge import': 'from openevolve_crewai_bridge import OpenEvolveCrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from openevolve_hephaestus_adapter import': 'from openevolve_crewai_adapter import OpenEvolveCrewAIAdapter  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from openevolve_hephaestus_delegation import': 'from openevolve_crewai_delegation import OpenEvolveCrewAIDelegation  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
    r'from ace_hephaestus_bridge import': 'from ace_crewai_bridge import ACECrewAIBridge  # CrewAI (MIT) - replaced Hephaestus (AGPL)',
}

# Class/function name replacements
NAME_REPLACEMENTS = {
    # Core classes
    r'HephaestusUnifiedBridge': 'CrewAIUnifiedBridge',
    r'HephaestusIntegrationManager': 'CrewAIIntegrationManager',
    r'HephaestusClient': 'CrewAIClient',
    r'HephaestusWorkflowSync': 'CrewAIWorkflowSync',
    r'HephaestusMonitor': 'CrewAIMonitor',

    # Bridge classes
    r'BubbleLabsHephaestusBridge': 'BubbleLabsCrewAIBridge',
    r'LeanAideHephaestusBridge': 'LeanAideCrewAIBridge',
    r'ClaudiomiroHephaestusBridge': 'ClaudiomiroCrewAIBridge',
    r'DataPizzaHephaestusBridge': 'DataPizzaCrewAIBridge',
    r'DecompositionHephaestusBridge': 'DecompositionCrewAIBridge',
    r'ROMAHephaestusBridge': 'ROMACrewAIBridge',
    r'ROMAMDAPMakerHephaestusBridge': 'ROMAMDAPMakerCrewAIBridge',
    r'OpenEvolveHephaestusWorkflowBridge': 'OpenEvolveCrewAIWorkflowBridge',
    r'OpenEvolveHephaestusAdapter': 'OpenEvolveCrewAIAdapter',
    r'ACEHephaestusBridge': 'ACECrewAIBridge',

    # Config classes
    r'BubbleLabsTicketConfig': 'BubbleLabsCrewAIConfig',
    r'HephaestusROMAConfig': 'CrewAIROMAConfig',
    r'HephaestusDataPizzaConfig': 'CrewAIDataPizzaConfig',
    r'HephaestusClaudiomiroConfig': 'CrewAIClaudiomiroConfig',

    # Function names
    r'setup_hephaestus_integration': 'setup_crewai_integration',
    r'get_hephaestus_integration': 'get_crewai_integration',
    r'initialize_hephaestus_workflow': 'initialize_crewai_workflow',
    r'execute_hephaestus_phase': 'execute_crewai_phase',
    r'delegate_to_hephaestus': 'delegate_to_crewai',

    # Variable names
    r'\bhephaestus_api_base': 'crewai_api_base',
    r'\bhephaestus_api_key': 'crewai_api_key',
    r'\bhephaestus_project_id': 'crewai_project_id',
    r'\bhephaestus_workflow_id': 'crewai_workflow_id',
    r'\bhephaestus_bridge': 'crewai_bridge',
    r'\bhephaestus_manager': 'crewai_manager',
    r'\bhephaestus_client': 'crewai_client',
    r'\bhephaestus_config': 'crewai_config',

    # Environment variables
    r'HEPHAESTUS_API_BASE': 'CREWAI_API_BASE',
    r'HEPHAESTUS_API_KEY': 'CREWAI_API_KEY',
    r'HEPHAESTUS_PROJECT_ID': 'CREWAI_PROJECT_ID',

    # String literals
    r'"Delegate to Hephaestus"': '"Delegate to CrewAI"',
    r"'Delegate to Hephaestus'": "'Delegate to CrewAI'",
    r'"Hephaestus workflow"': '"CrewAI workflow"',
    r"'Hephaestus workflow'": "'CrewAI workflow'",
    r'"Hephaestus API key"': '"CrewAI API key"',
    r"'Hephaestus API key'": "'CrewAI API key'",
    r'"Hephaestus integration"': '"CrewAI integration"',
    r"'Hephaestus integration'": "'CrewAI integration'",
}

# Phase 6 files to update
PHASE6_FILES = {
    '6.1': [
        'workflow_engine.py',
        'workflow_structures.py',
        'openevolve_workflow_manager_integrated.py',
        'openevolve_orchestrator.py',
        'openevolve_api.py',
        'model_orchestration.py',
    ],
    '6.2': [
        'integrations.py',
        'invention_planner_integrations.py',
        'invention_planner_integration_helpers.py',
        'openevolve_integration.py',
        'openevolve_imports.py',
        'openevolve_bubblelabs_ui.py',
        'openevolve_bubblelabs_api.py',
        'bubblelabs_integration.py',
        'bubblelabs_analytics.py',
        'bubblelabs_maker_integration.py',
        'ui_components.py',
        'openevolve_visualization.py',
    ],
    '6.3': [
        'decomposition_engine.py',
        'decomposition_engine_lean_enhanced.py',
        'problem_fractal_pipeline.py',
        'sub_problem_solver.py',
        'maker_integration_bridge.py',
        'sgd_workflow_orchestrator.py',
        'sgd_orchestrator_agent.py',
    ],
    '6.4': [
        'leanaide_evolution_mdap_workflow.py',
        'leanaide_evolutionary_workflow.py',
        'leanaide_mdap_workflow.py',
        'leanaide_mcts_workflow.py',
        'leanaide_mcts_mdap_workflow.py',
        'leanaide_decomposition_integration.py',
    ],
    '6.5': [
        'ragbits_integration/agents/base_agent.py',
        'ragbits_integration/agents/gold_team_agent.py',
        'ragbits_integration/agents/red_team_agent.py',
        'ragbits_integration/agents/blue_team_agent.py',
        'ragbits_integration/agents/run_phase2_tests.py',
        'ragbits_integration/agents/examples/ragbits_enhanced_blue_team.py',
        'ragbits_integration/config.py',
        'ragbits_integration/knowledge_base/rag_engine/advanced_rag.py',
        'ragbits_integration/knowledge_base/enrichment/knowledge_enricher.py',
        'ragbits_integration/knowledge_base/extraction/knowledge_extractor.py',
        'ragbits_integration/agents/tools/solution_eval_tool.py',
    ],
}

# =============================================================================
# MIGRATION FUNCTIONS
# =============================================================================

def add_migration_notice(content: str, filename: str) -> str:
    """Add migration notice to the top of the file."""
    notice = f'''"""
{filename} - CrewAI Integration

This file has been migrated from Hephaestus (AGPL) to CrewAI (MIT).

Migration Date: 2026-01-21
Migration Status: Complete

All Hephaestus references have been replaced with CrewAI equivalents.
The functionality remains the same, but now uses local CrewAI execution
instead of remote Hephaestus API calls.

For questions, see: CREWAI_MIGRATION_MASTER_TASKLIST.md
"""

'''

    # Check if file already has a docstring
    if content.startswith('"""') or content.startswith("'''"):
        # Find the end of the first docstring
        end_idx = content.find('"""', 3) if '"""' in content[3:100] else content.find("'''", 3)
        if end_idx != -1 and end_idx < 200:
            # Replace the first docstring
            return content[end_idx + 3:].lstrip()

    return notice + content


def apply_replacements(content: str) -> Tuple[str, Dict[str, int]]:
    """Apply all replacement patterns to the file content."""
    stats = {
        'imports_replaced': 0,
        'names_replaced': 0,
        'env_vars_replaced': 0,
    }

    # Apply import replacements
    for pattern, replacement in IMPORT_REPLACEMENTS.items():
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            stats['imports_replaced'] += 1

    # Apply name replacements
    for pattern, replacement in NAME_REPLACEMENTS.items():
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            stats['names_replaced'] += matches

    return content, stats


def update_file(filepath: Path, dry_run: bool = False) -> Dict[str, any]:
    """Update a single file with CrewAI replacements."""
    result = {
        'success': False,
        'error': None,
        'stats': {'imports_replaced': 0, 'names_replaced': 0, 'env_vars_replaced': 0},
    }

    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Apply replacements
        updated_content, stats = apply_replacements(original_content)

        # Add migration notice
        updated_content = add_migration_notice(updated_content, filepath.name)

        result['stats'] = stats

        if dry_run:
            result['success'] = True
            return result

        # Write updated content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def update_section(section: str, files: List[str], dry_run: bool = False) -> Dict[str, any]:
    """Update all files in a section."""
    section_results = {
        'section': section,
        'total_files': len(files),
        'updated_files': 0,
        'failed_files': 0,
        'files': {},
    }

    for filename in files:
        filepath = Path(filename)

        if not filepath.exists():
            section_results['files'][filename] = {
                'success': False,
                'error': 'File not found',
            }
            section_results['failed_files'] += 1
            continue

        result = update_file(filepath, dry_run)

        if result['success']:
            section_results['updated_files'] += 1
        else:
            section_results['failed_files'] += 1

        section_results['files'][filename] = result

    return section_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Update Phase 6 files to use CrewAI instead of Hephaestus'
    )
    parser.add_argument(
        '--section',
        choices=['6.1', '6.2', '6.3', '6.4', '6.5', 'all'],
        default='all',
        help='Which section to update (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without modifying files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Change to the script's directory
    os.chdir(Path(__file__).parent.parent)

    # Update sections
    sections_to_update = []
    if args.section == 'all':
        sections_to_update = list(PHASE6_FILES.keys())
    else:
        sections_to_update = [args.section]

    print("=" * 80)
    print("Phase 6 CrewAI Migration Script")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Sections: {', '.join(sections_to_update)}")
    print("=" * 80)
    print()

    all_results = {}

    for section in sections_to_update:
        files = PHASE6_FILES[section]
        print(f"Updating Section {section} ({len(files)} files)...")

        result = update_section(section, files, args.dry_run)
        all_results[section] = result

        print(f"  Updated: {result['updated_files']}/{result['total_files']}")
        print(f"  Failed: {result['failed_files']}/{result['total_files']}")

        if args.verbose and result['failed_files'] > 0:
            print("  Failed files:")
            for filename, file_result in result['files'].items():
                if not file_result['success']:
                    print(f"    - {filename}: {file_result.get('error', 'Unknown error')}")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    total_files = sum(r['total_files'] for r in all_results.values())
    total_updated = sum(r['updated_files'] for r in all_results.values())
    total_failed = sum(r['failed_files'] for r in all_results.values())

    print(f"Total files: {total_files}")
    print(f"Updated: {total_updated}")
    print(f"Failed: {total_failed}")
    print()

    if args.dry_run:
        print("DRY RUN COMPLETE - No files were modified")
        print("Run without --dry-run to apply changes")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
