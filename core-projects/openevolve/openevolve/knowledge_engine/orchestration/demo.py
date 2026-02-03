"""
Knowledge Engine Orchestration Demo

Demonstrates the orchestration system with domain-specific presets,
component filtering, and MCP server capabilities.
"""

import json
from datetime import datetime, timezone


def demo_domain_presets():
    """Demo different domain-specific orchestrators"""
    print("=" * 70)
    print("KNOWLEDGE ENGINE ORCHESTRATION - DOMAIN PRESETS DEMO")
    print("=" * 70)
    
    # Import after handling potential import errors
    try:
        from . import (
            create_finance_orchestrator,
            create_chemistry_orchestrator,
            create_research_orchestrator,
            create_minimal_orchestrator,
        )
        from .knowledge_orchestrator import ComponentType
    except ImportError:
        print("Note: Running in demo mode (integrations not available)")
        print("The orchestrator will work with available components only.")
        return
    
    print("\n" + "-" * 70)
    print("1. FINANCE ORCHESTRATOR")
    print("-" * 70)
    print("Features:")
    print("  - DeepKE for entity extraction")
    print("  - Karate Club for graph analysis")
    print("  - Causal-Learn for market causality analysis")
    print("  - PAMI for pattern mining")
    print("  - DISABLED: GlobalChem, Neuromancer")
    print()
    
    try:
        finance_orch = create_finance_orchestrator()
        status = finance_orch.get_status()
        print(f"Status: {json.dumps(status, indent=2, default=str)}")
        print()
    except Exception as e:
        print(f"Note: Finance orchestrator demo skipped: {e}")
    
    print("\n" + "-" * 70)
    print("2. CHEMISTRY ORCHESTRATOR")
    print("-" * 70)
    print("Features:")
    print("  - GlobalChem for chemical entity recognition")
    print("  - Neuromancer for molecular dynamics modeling")
    print("  - DeepKE for general entity extraction")
    print("  - NeuralKG for knowledge graph embeddings")
    print("  - DISABLED: None (chemistry-specific)")
    print()
    
    try:
        chem_orch = create_chemistry_orchestrator()
        status = chem_orch.get_status()
        print(f"Status: {json.dumps(status, indent=2, default=str)}")
        print()
    except Exception as e:
        print(f"Note: Chemistry orchestrator demo skipped: {e}")
    
    print("\n" + "-" * 70)
    print("3. RESEARCH ORCHESTRATOR (COMPREHENSIVE)")
    print("-" * 70)
    print("Features:")
    print("  - ALL components enabled")
    print("  - DeepKE, Karate Club, KG-Gen")
    print("  - PAMI, NeuralKG, Causal-Learn")
    print("  - Lagrange-Mapper, GlobalChem, Neuromancer")
    print()
    
    try:
        research_orch = create_research_orchestrator()
        status = research_orch.get_status()
        print(f"Status: {json.dumps(status, indent=2, default=str)}")
        print()
    except Exception as e:
        print(f"Note: Research orchestrator demo skipped: {e}")
    
    print("\n" + "-" * 70)
    print("4. MINIMAL ORCHESTRATOR")
    print("-" * 70)
    print("Features:")
    print("  - DeepKE for extraction")
    print("  - KG-Gen for graph building")
    print("  - DISABLED: Everything else")
    print()
    
    try:
        minimal_orch = create_minimal_orchestrator()
        status = minimal_orch.get_status()
        print(f"Status: {json.dumps(status, indent=2, default=str)}")
        print()
    except Exception as e:
        print(f"Note: Minimal orchestrator demo skipped: {e}")


def demo_component_management():
    """Demo component enable/disable functionality"""
    print("\n" + "=" * 70)
    print("COMPONENT MANAGEMENT DEMO")
    print("=" * 70)
    
    try:
        from . import create_research_orchestrator
        from .knowledge_orchestrator import ComponentType
        
        print("\n1. Starting with research orchestrator (all components)")
        orch = create_research_orchestrator()
        
        status = orch.get_status()
        print(f"   Initial components: {len(status['initialized_components'])}")
        print(f"   Configured components: {len(status['configured_components'])}")
        
        print("\n2. Disabling Causal-Learn component")
        orch.config.disable_component(ComponentType.CAUSAL_LEARN)
        print(f"   Causal-Learn now disabled in config")
        
        print("\n3. Disabling GlobalChem component")
        orch.config.disable_component(ComponentType.GLOBAL_CHEM)
        print(f"   GlobalChem now disabled in config")
        
        print("\n4. Re-initializing with new configuration")
        orch._initialize_components()
        
        new_status = orch.get_status()
        print(f"   Final components: {len(new_status['initialized_components'])}")
        
    except Exception as e:
        print(f"Note: Component management demo skipped: {e}")


def demo_processing():
    """Demo data processing with orchestrator"""
    print("\n" + "=" * 70)
    print("PROCESSING DEMO")
    print("=" * 70)
    
    try:
        from . import create_minimal_orchestrator
        
        print("\n1. Creating minimal orchestrator for demo")
        orch = create_minimal_orchestrator()
        
        print("\n2. Processing sample text")
        sample_data = {
            'text': 'Apple Inc. is a technology company headquartered in Cupertino, California.',
            'data_type': 'company_description'
        }
        
        result = orch.process(sample_data)
        
        print(f"\n   Result Summary:")
        print(f"   - Status: {result['status']}")
        print(f"   - Domain: {result['domain']}")
        print(f"   - Stages executed: {result['execution']['stages_executed']}")
        print(f"   - Duration: {result['execution']['duration_ms']:.2f} ms")
        
        if result.get('results'):
            print(f"\n   Results Keys: {list(result['results'].keys())}")
        
        if result.get('skipped_stages'):
            print(f"\n   Skipped Stages: {[s['name'] for s in result['skipped_stages']]}")
        
    except Exception as e:
        print(f"Note: Processing demo skipped: {e}")


def demo_mcp_server():
    """Demo MCP server functionality"""
    print("\n" + "=" * 70)
    print("MCP SERVER DEMO")
    print("=" * 70)
    
    try:
        from . import create_mcp_server
        
        print("\n1. Creating MCP server handler")
        handler = create_mcp_server()
        
        print("\n2. Getting available methods")
        methods_response = handler.handle({
            "jsonrpc": "2.0",
            "method": "knowledge.get_available_methods",
            "params": {},
            "id": 1
        })
        
        if "result" in methods_response:
            methods = methods_response["result"]["methods"]
            print(f"   Total available methods: {len(methods)}")
            print("\n   Method Categories:")
            
            categories = {}
            for m in methods:
                category = m["name"].split(".")[1].split("_")[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(m["name"])
            
            for cat, meths in categories.items():
                print(f"     - {cat.capitalize()}: {len(meths)} methods")
        
        print("\n3. Creating finance orchestrator via MCP")
        create_response = handler.handle({
            "jsonrpc": "2.0",
            "method": "knowledge.create_finance_orchestrator",
            "params": {
                "orchestrator_id": "mcp_finance_demo"
            },
            "id": 2
        })
        
        if "result" in create_response:
            result = create_response["result"]
            print(f"   Created: {result['orchestrator_id']}")
            print(f"   Domain: {result['domain']}")
            print(f"   Components: {len(result['components'])}")
        
        print("\n4. Getting component status via MCP")
        status_response = handler.handle({
            "jsonrpc": "2.0",
            "method": "knowledge.get_component_status",
            "params": {
                "orchestrator_id": "mcp_finance_demo"
            },
            "id": 3
        })
        
        if "result" in status_response:
            result = status_response["result"]
            print(f"   Total configured: {result['total_configured']}")
            print(f"   Total available: {result['total_available']}")
            print(f"   Active components: {len(result['active_components'])}")
        
        print("\n5. Performing health check")
        health_response = handler.handle({
            "jsonrpc": "2.0",
            "method": "knowledge.health_check",
            "params": {},
            "id": 4
        })
        
        if "result" in health_response:
            result = health_response["result"]
            print(f"   Overall status: {result['overall_status']}")
            print(f"   Orchestrators: {len(result['orchestrators'])}")
            print(f"   Timestamp: {result['timestamp']}")
        
    except Exception as e:
        print(f"Note: MCP server demo skipped: {e}")


def demo_config_persistence():
    """Demo configuration save/load"""
    print("\n" + "=" * 70)
    print("CONFIGURATION PERSISTENCE DEMO")
    print("=" * 70)
    
    try:
        from . import create_finance_orchestrator
        import tempfile
        import os
        
        print("\n1. Creating finance orchestrator")
        orch = create_finance_orchestrator()
        
        print("\n2. Saving configuration to temporary file")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        orch.save_config(temp_path)
        print(f"   Saved to: {temp_path}")
        
        print("\n3. Loading configuration")
        loaded_orch = create_finance_orchestrator.__wrapped__(
            KnowledgeOrchestrator.load_config(temp_path)
        ) if hasattr(create_finance_orchestrator, '__wrapped__') else KnowledgeOrchestrator.load_config(temp_path)
        
        loaded_status = loaded_orch.get_status()
        print(f"   Loaded orchestrator: {loaded_status['name']}")
        print(f"   Domain: {loaded_status['domain']}")
        
        # Cleanup
        os.unlink(temp_path)
        print(f"   Cleaned up temporary file")
        
    except Exception as e:
        print(f"Note: Config persistence demo skipped: {e}")


def run_all_demos():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("KNOWLEDGE ENGINE ORCHESTRATION SYSTEM")
    print("Complete Demonstration")
    print("=" * 70)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    
    demo_domain_presets()
    demo_component_management()
    demo_processing()
    demo_mcp_server()
    demo_config_persistence()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Key Takeaways:")
    print("  1. Domain presets automatically configure components")
    print("  2. Finance: disables chemistry components")
    print("  3. Chemistry: enables chemical analysis tools")
    print("  4. Research: enables all components")
    print("  5. Minimal: only essential components")
    print("  6. Components can be enabled/disabled at runtime")
    print("  7. MCP server provides standardized API access")
    print("  8. Configuration can be saved and loaded")


if __name__ == "__main__":
    run_all_demos()
