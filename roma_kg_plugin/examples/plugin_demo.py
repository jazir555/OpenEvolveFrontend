"""
ROMA Knowledge Graph Plugin - Usage Examples

This demonstrates how to use the ROMA Knowledge Graph Plugin
following the Air Gap principle.
"""

import asyncio
from typing import Any, Dict


async def demo_plugin_initialization():
    """Demo: Plugin initialization and setup."""
    print("\n=== Demo 1: Plugin Initialization ===\n")

    from roma_kg_plugin import create_plugin

    # Create plugin instance
    plugin = create_plugin()

    print(f"Plugin created: {plugin.name} v{plugin.version}")
    print(f"Description: {plugin.description}")

    # Get plugin info
    info = plugin.get_info()
    print(f"\nPlugin Info:")
    print(f"  Features: {', '.join(info['features'][:3])}...")
    print(f"  Components: {list(info['components'].keys())}")


async def demo_plugin_with_roma():
    """Demo: Using plugin with ROMA client."""
    print("\n=== Demo 2: Plugin with ROMA Client ===\n")

    from roma_kg_plugin import create_plugin

    # Mock ROMA client (in real use, this would be actual ROMA client)
    class MockROMAClient:
        def __init__(self):
            self.connected = True

        async def query(self, query: str):
            return {"results": []}

    # Mock knowledge engine
    class MockKGEngine:
        def __init__(self):
            self.graph = {"nodes": [], "edges": []}

        async def search(self, query: str):
            return {"nodes": [], "edges": []}

    # Create and initialize plugin
    plugin = create_plugin()

    # Initialize with dependencies
    await plugin.initialize(
        roma_client=MockROMAClient(),
        knowledge_engine=MockKGEngine(),
        config={
            "visualization": {
                "max_nodes": 1000,
                "layout_algorithm": "spring"
            }
        }
    )

    print(f"Plugin initialized: {plugin._initialized}")
    print(f"Plugin enabled: {plugin._enabled}")


async def demo_knowledge_graph_commands():
    """Demo: Using knowledge graph commands."""
    print("\n=== Demo 3: Knowledge Graph Commands ===\n")

    from roma_kg_plugin.commands import KnowledgeGraphCommands

    # Mock dependencies
    class MockROMAClient:
        pass

    class MockKGEngine:
        async def search(self, query: str):
            return {"nodes": [{"id": "1", "label": "Python"}], "edges": []}

    # Create command handler
    commands = KnowledgeGraphCommands(
        roma_client=MockROMAClient(),
        kg_engine=MockKGEngine()
    )

    # Get available commands
    available = commands.get_available_commands()

    print("Available Commands:")
    for cmd in available:
        print(f"  {cmd['command']}")
        print(f"    {cmd['description']}")


async def demo_knowledge_graph_panel():
    """Demo: Using knowledge graph panel."""
    print("\n=== Demo 4: Knowledge Graph Panel ===\n")

    from roma_kg_plugin.panels import KnowledgeGraphPanel

    # Mock dependencies
    class MockROMAClient:
        pass

    class MockKGEngine:
        pass

    # Create panel
    panel = KnowledgeGraphPanel(
        roma_client=MockROMAClient(),
        kg_manager=MockKGEngine()
    )

    print(f"Panel created: {panel.__class__.__name__}")
    print(f"Current graph: {panel.current_graph}")

    # Update filter
    panel.update_filter("node_types", ["entity", "concept"])
    print(f"Filters updated: {panel.filter_state}")


async def demo_analytics_dashboard():
    """Demo: Using analytics dashboard."""
    print("\n=== Demo 5: Analytics Dashboard ===\n")

    from roma_kg_plugin.panels import AnalyticsDashboard

    # Mock knowledge engine
    class MockKGEngine:
        pass

    # Create dashboard
    dashboard = AnalyticsDashboard(
        kg_manager=MockKGEngine()
    )

    print(f"Dashboard created: {dashboard.__class__.__name__}")
    print(f"Current metrics: {dashboard.current_metrics}")


async def demo_ascii_graph_renderer():
    """Demo: Using ASCII graph renderer."""
    print("\n=== Demo 6: ASCII Graph Renderer ===\n")

    from roma_kg_plugin.visualization import AsciiGraphRenderer

    # Create renderer
    renderer = AsciiGraphRenderer()

    # Render sample graph
    graph_data = {
        "nodes": [
            {"id": "A", "label": "Node A", "type": "entity"},
            {"id": "B", "label": "Node B", "type": "concept"},
            {"id": "C", "label": "Node C", "type": "entity"}
        ],
        "edges": [
            {"source": "A", "target": "B", "type": "related_to"},
            {"source": "B", "target": "C", "type": "part_of"}
        ]
    }

    try:
        ascii_graph = renderer.render_graph(graph_data, width=40, height=10)
        print("ASCII Graph:")
        print(ascii_graph)
    except Exception as e:
        print(f"Render error (expected in demo): {e}")


async def demo_interactive_exploration():
    """Demo: Interactive graph exploration."""
    print("\n=== Demo 7: Interactive Exploration ===\n")

    from roma_kg_plugin.interactive import InteractiveGraphExplorer

    # Mock knowledge engine
    class MockKGEngine:
        async def get_graph(self):
            import networkx as nx
            G = nx.Graph()
            G.add_edge("A", "B")
            G.add_edge("B", "C")
            return G

    # Create explorer
    explorer = InteractiveGraphExplorer(kg_manager=MockKGEngine())

    print(f"Explorer created: {explorer.__class__.__name__}")
    print(f"Exploration stack: {explorer.exploration_stack}")
    print(f"Current focus: {explorer.current_focus}")


async def demo_export_functionality():
    """Demo: Export functionality."""
    print("\n=== Demo 8: Export Functionality ===\n")

    from roma_kg_plugin.panels import KnowledgeGraphPanel

    # Mock dependencies
    class MockROMAClient:
        pass

    class MockKGEngine:
        pass

    # Create panel
    panel = KnowledgeGraphPanel(
        roma_client=MockROMAClient(),
        kg_manager=MockKGEngine()
    )

    # Export formats
    formats = ["json", "gexf", "csv", "graphml"]

    print("Supported export formats:")
    for fmt in formats:
        print(f"  - {fmt}")


async def demo_configuration():
    """Demo: Plugin configuration."""
    print("\n=== Demo 9: Plugin Configuration ===\n")

    import yaml

    # Load config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        print("Configuration loaded:")
        print(f"  Plugin name: {config['plugin']['name']}")
        print(f"  Version: {config['plugin']['version']}")
        print(f"  Enabled: {config['plugin']['enabled']}")
        print(f"  Priority: {config['plugin']['priority']}")

        print("\nVisualization settings:")
        viz_config = config.get("visualization", {})
        print(f"  Max nodes: {viz_config.get('max_nodes')}")
        print(f"  Layout: {viz_config.get('layout_algorithm')}")
        print(f"  Dimensions: {viz_config.get('default_width')}x{viz_config.get('default_height')}")

    except FileNotFoundError:
        print("Config file not found (this is okay for demo)")


async def demo_complete_workflow():
    """Demo: Complete workflow from start to finish."""
    print("\n=== Demo 10: Complete Workflow ===\n")

    from roma_kg_plugin import create_plugin

    # 1. Create plugin
    plugin = create_plugin()
    print("1. Plugin created")

    # 2. Initialize with dependencies
    class MockROMAClient:
        pass

    class MockKGEngine:
        pass

    await plugin.initialize(
        roma_client=MockROMAClient(),
        knowledge_engine=MockKGEngine(),
        config={}
    )
    print("2. Plugin initialized")

    # 3. Register components
    mock_registry = {}
    result = await plugin.register_panels(mock_registry)
    print(f"3. Panels registered: {result}")

    # 4. Get plugin info
    info = plugin.get_info()
    print(f"4. Plugin features: {len(info['features'])}")

    # 5. Shutdown
    await plugin.shutdown()
    print("5. Plugin shutdown complete")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("ROMA Knowledge Graph Plugin - Usage Examples")
    print("="*60)

    demos = [
        demo_plugin_initialization,
        demo_plugin_with_roma,
        demo_knowledge_graph_commands,
        demo_knowledge_graph_panel,
        demo_analytics_dashboard,
        demo_ascii_graph_renderer,
        demo_interactive_exploration,
        demo_export_functionality,
        demo_configuration,
        demo_complete_workflow,
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"\nDemo failed: {e}")

    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
