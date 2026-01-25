"""
ROMA Knowledge Graph Plugin - Test Suite

Tests plugin functionality following CLAUDE.md principles.
"""

import pytest
from datetime import datetime, timezone


class TestROMAKnowledgeGraphPlugin:
    """Test plugin initialization and core functionality."""

    def test_plugin_creation(self):
        """Test plugin factory function."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()
        assert plugin is not None
        assert plugin.name == "roma_kg_plugin"
        assert plugin.version == "1.0.0"
        assert not plugin._initialized

    def test_plugin_info(self):
        """Test plugin info method."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()
        info = plugin.get_info()

        assert "name" in info
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert isinstance(info["features"], list)

    @pytest.mark.asyncio
    async def test_plugin_initialization(self):
        """Test plugin initialization with dependencies."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()

        # Mock dependencies
        mock_client = MockROMAClient()
        mock_kg_engine = MockKGEngine()

        # Initialize plugin
        result = await plugin.initialize(
            roma_client=mock_client,
            knowledge_engine=mock_kg_engine,
            config={}
        )

        assert result is True
        assert plugin._initialized is True
        assert plugin.roma_client is mock_client
        assert plugin.kg_engine is mock_kg_engine

    @pytest.mark.asyncio
    async def test_plugin_command_registration(self):
        """Test command registration."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()
        await plugin.initialize(
            roma_client=MockROMAClient(),
            knowledge_engine=MockKGEngine()
        )

        # Mock command registry
        registry = MockCommandRegistry()

        # Register commands
        result = await plugin.register_commands(registry)

        assert result is True
        assert len(registry.commands) == 8  # 8 kg commands

    @pytest.mark.asyncio
    async def test_plugin_panel_registration(self):
        """Test panel registration."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()
        await plugin.initialize(
            roma_client=MockROMAClient(),
            knowledge_engine=MockKGEngine()
        )

        # Mock panel registry
        registry = MockPanelRegistry()

        # Register panels
        result = await plugin.register_panels(registry)

        assert result is True
        assert len(registry.panels) == 2  # knowledge_graph, analytics

    @pytest.mark.asyncio
    async def test_plugin_shutdown(self):
        """Test plugin cleanup."""
        from roma_kg_plugin import create_plugin

        plugin = create_plugin()
        await plugin.initialize(
            roma_client=MockROMAClient(),
            knowledge_engine=MockKGEngine()
        )

        # Shutdown plugin
        await plugin.shutdown()

        assert plugin._initialized is False
        assert plugin._enabled is False
        assert len(plugin.panels) == 0
        assert len(plugin.commands) == 0


class TestKnowledgeGraphCommands:
    """Test command handlers."""

    def test_commands_initialization(self):
        """Test command handler initialization."""
        from roma_kg_plugin.commands import KnowledgeGraphCommands

        commands = KnowledgeGraphCommands(
            roma_client=MockROMAClient(),
            kg_engine=MockKGEngine()
        )

        assert commands.roma_client is not None
        assert commands.kg_engine is not None
        assert len(commands.command_history) == 0

    def test_available_commands(self):
        """Test list of available commands."""
        from roma_kg_plugin.commands import KnowledgeGraphCommands

        commands = KnowledgeGraphCommands()
        available = commands.get_available_commands()

        assert len(available) == 8
        assert any(cmd['command'].startswith('/kg search') for cmd in available)


class TestKnowledgeGraphPanel:
    """Test knowledge graph panel."""

    def test_panel_initialization(self):
        """Test panel initialization."""
        from roma_kg_plugin.panels import KnowledgeGraphPanel

        panel = KnowledgeGraphPanel(
            roma_client=MockROMAClient(),
            kg_manager=MockKGEngine()
        )

        assert panel.client is not None
        assert panel.kg is not None
        assert panel.current_graph is None

    def test_panel_filter_update(self):
        """Test filter state updates."""
        from roma_kg_plugin.panels import KnowledgeGraphPanel

        panel = KnowledgeGraphPanel(
            roma_client=MockROMAClient(),
            kg_manager=MockKGEngine()
        )

        # Update filter
        panel.update_filter("node_types", ["entity", "concept"])

        assert panel.filter_state["node_types"] == ["entity", "concept"]


class TestAnalyticsDashboard:
    """Test analytics dashboard."""

    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        from roma_kg_plugin.panels import AnalyticsDashboard

        dashboard = AnalyticsDashboard(
            kg_manager=MockKGEngine()
        )

        assert dashboard.kg is not None
        assert dashboard.current_metrics == {}


class TestAsciiGraphRenderer:
    """Test ASCII graph renderer."""

    def test_renderer_initialization(self):
        """Test renderer initialization."""
        from roma_kg_plugin.visualization import AsciiGraphRenderer

        renderer = AsciiGraphRenderer()

        assert renderer.node_symbols is not None
        assert renderer.edge_symbols is not None

    def test_render_empty_graph(self):
        """Test rendering empty graph."""
        from roma_kg_plugin.visualization import AsciiGraphRenderer

        renderer = AsciiGraphRenderer()
        result = renderer.render_graph({
            "nodes": [],
            "edges": []
        })

        assert result == "Empty graph"


class TestInteractiveExplorer:
    """Test interactive graph explorer."""

    @pytest.mark.asyncio
    async def test_explorer_initialization(self):
        """Test explorer initialization."""
        from roma_kg_plugin.interactive import InteractiveGraphExplorer

        explorer = InteractiveGraphExplorer(kg_manager=MockKGEngine())

        assert explorer.kg is not None
        assert len(explorer.exploration_stack) == 0
        assert explorer.current_focus is None


# Mock classes for testing

class MockROMAClient:
    """Mock ROMA client for testing."""

    def __init__(self):
        self.connected = True

    def log(self, level, message, context):
        """Mock log method."""
        pass


class MockKGEngine:
    """Mock knowledge graph engine for testing."""

    def __init__(self):
        self.connected = True

    async def query(self, query):
        """Mock query method."""
        return {"nodes": [], "edges": []}

    async def get_graph(self):
        """Mock get_graph method."""
        import networkx as nx
        return nx.Graph()


class MockCommandRegistry:
    """Mock command registry for testing."""

    def __init__(self):
        self.commands = {}

    def register_command(self, name, handler, help_text, usage=""):
        """Mock register_command method."""
        self.commands[name] = {
            "handler": handler,
            "help_text": help_text,
            "usage": usage
        }


class MockPanelRegistry:
    """Mock panel registry for testing."""

    def __init__(self):
        self.panels = {}

    def register_panel(self, name, panel_class, title, description=""):
        """Mock register_panel method."""
        self.panels[name] = {
            "class": panel_class,
            "title": title,
            "description": description
        }


class MockMenuRegistry:
    """Mock menu registry for testing."""

    def __init__(self):
        self.menus = {}

    def register_menu(self, name, menu, title, description=""):
        """Mock register_menu method."""
        self.menus[name] = {
            "menu": menu,
            "title": title,
            "description": description
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
