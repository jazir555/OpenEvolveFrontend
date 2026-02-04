"""
Capability Reporting for Knowledge Engine

This module provides functions for reporting which integrations and capabilities
are available in the current environment.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def get_capabilities() -> Dict[str, Any]:
    """
    Get a comprehensive report of available capabilities.

    Returns:
        Dictionary with:
        - 'available': List of available capability names
        - 'unavailable': List of unavailable capability details with reasons
        - 'integrations': Dict of integration availability
        - 'features': Dict of feature availability
    """
    capabilities = {
        'available': [],
        'unavailable': [],
        'integrations': {},
        'features': {}
    }

    # Core capabilities
    core_checks = [
        ('knowledge_graph', 'Entity Knowledge Graph', 'knowledge_engine.core.entity_knowledge_graph'),
        ('storage_backends', 'Storage Backends', 'knowledge_engine.core.backends'),
        ('strategy_recommender', 'Strategy Recommender', 'knowledge_engine.core.strategy_recommender_complete'),
        ('embedding_service', 'Embedding Service', 'knowledge_engine.embedding_service'),
        ('confidence_scorer', 'Confidence Scorer', 'knowledge_engine.confidence_scorer'),
    ]

    for name, display_name, module_path in core_checks:
        try:
            __import__(module_path)
            capabilities['available'].append(display_name)
            capabilities['features'][name] = True
        except ImportError:
            capabilities['unavailable'].append({
                'name': display_name,
                'type': 'feature',
                'reason': f'Module {module_path} not available'
            })
            capabilities['features'][name] = False

    # Integration capabilities
    try:
        from knowledge_engine.integrations import (
            Z3_INTEGRATION_AVAILABLE,
            LEANAIDE_KE_AVAILABLE,
            LEANAIDE_PROOF_AVAILABLE,
            UNIFIED_BRIDGE_AVAILABLE,
            LOONGFLOW_INTEGRATION_AVAILABLE,
            UNIFIED_EVOLUTION_AVAILABLE,
            ROMA_INTEGRATION_AVAILABLE,
            ROMA_EKG_INTEGRATION_AVAILABLE,
            CAUSAL_LEARN_AVAILABLE,
            DEEPKE_INTEGRATION_AVAILABLE,
            DSPY_INTEGRATION_AVAILABLE,
            RAGBITS_INTEGRATION_AVAILABLE,
            ACE_INTEGRATION_AVAILABLE,
            AGENTJSON_INTEGRATION_AVAILABLE,
            RESEARCH_QUEST_INTEGRATION_AVAILABLE,
            MCP_GATEWAY_INTEGRATION_AVAILABLE,
            OPENEVOLVE_INTEGRATION_AVAILABLE,
        )

        integrations = {
            'z3_knowledge': (Z3_INTEGRATION_AVAILABLE, 'Z3 Knowledge Integration', 'pip install z3-solver'),
            'leanaide_ke': (LEANAIDE_KE_AVAILABLE, 'LeanAIDE Knowledge Extraction', 'See LEANAIDE documentation'),
            'leanaide_proof': (LEANAIDE_PROOF_AVAILABLE, 'LeanAIDE Proof Integration', 'See LEANAIDE documentation'),
            'unified_bridge': (UNIFIED_BRIDGE_AVAILABLE, 'Unified Math Knowledge Bridge', 'Core feature'),
            'loongflow': (LOONGFLOW_INTEGRATION_AVAILABLE, 'LoongFlow Integration', 'See LoongFlow documentation'),
            'unified_evolution': (UNIFIED_EVOLUTION_AVAILABLE, 'Unified Evolution Integration', 'Core feature'),
            'roma': (ROMA_INTEGRATION_AVAILABLE, 'ROMA Integration', 'See ROMA documentation'),
            'roma_ekg': (ROMA_EKG_INTEGRATION_AVAILABLE, 'ROMA Entity Knowledge Graph', 'See ROMA documentation'),
            'causal_learn': (CAUSAL_LEARN_AVAILABLE, 'Causal-Learn Integration', 'pip install causal-learn'),
            'deepke': (DEEPKE_INTEGRATION_AVAILABLE, 'DeepKE Integration', 'pip install deepke'),
            'dspy': (DSPY_INTEGRATION_AVAILABLE, 'DSPy Integration', 'pip install dspy-ai'),
            'ragbits': (RAGBITS_INTEGRATION_AVAILABLE, 'Ragbits Integration', 'pip install ragbits'),
            'ace': (ACE_INTEGRATION_AVAILABLE, 'Agentic Context Engine', 'See ACE documentation'),
            'agentjson': (AGENTJSON_INTEGRATION_AVAILABLE, 'AgentJSON Integration', 'pip install agentjson'),
            'research_quest': (RESEARCH_QUEST_INTEGRATION_AVAILABLE, 'Research-Quest Integration', 'See Research-Quest documentation'),
            'mcp_gateway': (MCP_GATEWAY_INTEGRATION_AVAILABLE, 'MCP Gateway Integration', 'See MCP documentation'),
            'openevolve': (OPENEVOLVE_INTEGRATION_AVAILABLE, 'OpenEvolve Integration Library', 'Core feature'),
        }

        for key, (available, display_name, install_hint) in integrations.items():
            capabilities['integrations'][key] = available
            if available:
                capabilities['available'].append(display_name)
            else:
                capabilities['unavailable'].append({
                    'name': display_name,
                    'type': 'integration',
                    'reason': f'Integration not available',
                    'install_hint': install_hint
                })

    except ImportError as e:
        logger.warning(f"Could not import integration flags: {e}")
        capabilities['unavailable'].append({
            'name': 'Integration Flags',
            'type': 'system',
            'reason': f'Could not import integration availability flags: {e}'
        })

    # Optional dependencies
    try:
        from .optional_imports import OPTIONAL_DEPENDENCIES, is_available

        for module_name, info in OPTIONAL_DEPENDENCIES.items():
            available = is_available(module_name.replace('.', '_'))
            if available:
                capabilities['available'].append(f"{info['package']} (optional)")
            else:
                capabilities['unavailable'].append({
                    'name': info['package'],
                    'type': 'optional_dependency',
                    'reason': f"Optional dependency for {info['feature']}",
                    'install_hint': info['install']
                })

    except ImportError:
        pass

    return capabilities


def print_capability_report():
    """Print a human-readable capability report."""
    capabilities = get_capabilities()

    print("\n" + "="*80)
    print("KNOWLEDGE ENGINE CAPABILITY REPORT")
    print("="*80)

    print("\nAvailable Capabilities:")
    if capabilities['available']:
        for cap in capabilities['available']:
            print(f"  [+] {cap}")
    else:
        print("  (none)")

    print("\nUnavailable Capabilities:")
    if capabilities['unavailable']:
        for cap in capabilities['unavailable']:
            print(f"  [-] {cap['name']}")
            if cap.get('reason'):
                print(f"      Reason: {cap['reason']}")
            if cap.get('install_hint'):
                print(f"      Install: {cap['install_hint']}")
    else:
        print("  (none)")

    print("\nIntegration Status:")
    for integration, available in capabilities.get('integrations', {}).items():
        status = "[+]" if available else "[-]"
        print(f"  {status} {integration}")

    print("\nFeature Status:")
    for feature, available in capabilities.get('features', {}).items():
        status = "[+]" if available else "[-]"
        print(f"  {status} {feature}")

    print("\n" + "="*80)


def get_integration_summary() -> Dict[str, Any]:
    """
    Get a summary of integration availability.

    Returns:
        Dictionary with integration availability summary
    """
    capabilities = get_capabilities()

    integrations = capabilities.get('integrations', {})
    available_count = sum(1 for v in integrations.values() if v)
    total_count = len(integrations)

    return {
        'total': total_count,
        'available': available_count,
        'unavailable': total_count - available_count,
        'availability_percentage': (available_count / total_count * 100) if total_count > 0 else 0,
        'integrations': integrations
    }


__all__ = [
    'get_capabilities',
    'print_capability_report',
    'get_integration_summary',
]
