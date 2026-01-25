"""
ROMA Knowledge Graph Plugin

A plugin for ROMA that adds knowledge graph visualization,
analytics, and exploration capabilities without modifying
ROMA core files (follows CLAUDE.md Air Gap principle).

This plugin uses dependency injection and registration hooks
to extend ROMA's functionality.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING, List
import asyncio
from datetime import datetime, timezone

if TYPE_CHECKING:
    from roma_dspy.tui.core.client import ROMAClient

class ROMAKnowledgeGraphPlugin:
    """
    ROMA Knowledge Graph Plugin.

    This plugin extends ROMA with knowledge graph capabilities
    through registration hooks, without modifying core ROMA code.

    Architecture:
    - Panels: Knowledge graph and analytics visualizations
    - Commands: 8 custom commands for graph operations
    - Menus: Hierarchical menu system
    - Integration: Connection to knowledge engine

    All dependencies are injected - no direct coupling to ROMA internals.
    """

    def __init__(self):
        self.name = "roma_kg_plugin"
        self.version = "1.0.0"
        self.description = "Knowledge Graph Integration for ROMA"
        self.author = "OpenEvolve"

        # Dependency injection
        self.roma_client: Optional["ROMAClient"] = None
        self.kg_engine: Optional[Any] = None
        self.config: Dict[str, Any] = {}

        # Plugin components
        self.panels: Dict[str, Any] = {}
        self.commands: Dict[str, Any] = {}
        self.menus: Dict[str, Any] = {}

        # State
        self._initialized = False
        self._enabled = False

        # Logging
        self._log_context = {
            "plugin": self.name,
            "version": self.version
        }

    async def initialize(
        self,
        roma_client: "ROMAClient",
        knowledge_engine=None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialize plugin with ROMA client and knowledge engine.

        This is called by ROMA's plugin system at startup.
        Uses dependency injection - no direct imports of ROMA modules.

        Args:
            roma_client: ROMA client instance (injected)
            knowledge_engine: Knowledge engine instance (injected)
            config: Plugin configuration dict

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Store injected dependencies
            self.roma_client = roma_client
            self.kg_engine = knowledge_engine
            self.config = config or {}

            # Validate configuration
            self._validate_config()

            # Initialize components
            await self._initialize_panels()
            await self._initialize_commands()
            await self._initialize_menus()

            self._initialized = True
            self._enabled = True

            # Log initialization
            self._log("info", "Plugin initialized successfully", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": self.config
            })

            return True

        except Exception as e:
            self._log("error", f"Initialization failed: {e}", {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False

    async def register_commands(self, command_registry) -> bool:
        """
        Register plugin commands with ROMA's command registry.

        ROMA will call this to register our custom commands.
        All 8 KG commands are registered here.

        Args:
            command_registry: ROMA's command registry (injected)

        Returns:
            True if registration successful
        """
        try:
            # Import command handler
            from .commands import KnowledgeGraphCommands

            # Create command handler instance with injected dependencies
            kg_commands = KnowledgeGraphCommands(
                roma_client=self.roma_client,
                kg_engine=self.kg_engine
            )

            # Register all 8 commands
            commands_to_register = [
                {
                    "name": "kg_search",
                    "handler": kg_commands.handle_search_command,
                    "help_text": "Search knowledge graph for entities",
                    "usage": "/kg search <query>"
                },
                {
                    "name": "kg_explore",
                    "handler": kg_commands.handle_explore_command,
                    "help_text": "Explore knowledge graph neighborhood",
                    "usage": "/kg explore <entity_id>"
                },
                {
                    "name": "kg_path",
                    "handler": kg_commands.handle_path_command,
                    "help_text": "Find shortest path between entities",
                    "usage": "/kg path <from_entity> <to_entity>"
                },
                {
                    "name": "kg_communities",
                    "handler": kg_commands.handle_communities_command,
                    "help_text": "List graph communities",
                    "usage": "/kg communities [limit]"
                },
                {
                    "name": "kg_stats",
                    "handler": kg_commands.handle_stats_command,
                    "help_text": "Show knowledge graph statistics",
                    "usage": "/kg stats"
                },
                {
                    "name": "kg_export",
                    "handler": kg_commands.handle_export_command,
                    "help_text": "Export knowledge graph data",
                    "usage": "/kg export <format> [output_path]"
                },
                {
                    "name": "kg_timeline",
                    "handler": kg_commands.handle_timeline_command,
                    "help_text": "Show entity timeline",
                    "usage": "/kg timeline <entity_id>"
                },
                {
                    "name": "kg_analyze",
                    "handler": kg_commands.handle_analyze_command,
                    "help_text": "Run graph analysis",
                    "usage": "/kg analyze <analysis_type>"
                }
            ]

            for cmd in commands_to_register:
                command_registry.register_command(**cmd)
                self.commands[cmd["name"]] = cmd

            self._log("info", f"Registered {len(commands_to_register)} commands")

            return True

        except Exception as e:
            self._log("error", f"Command registration failed: {e}")
            return False

    async def register_panels(self, panel_registry) -> bool:
        """
        Register plugin panels with ROMA's panel registry.

        Args:
            panel_registry: ROMA's panel registry (injected)

        Returns:
            True if registration successful
        """
        try:
            from .panels import KnowledgeGraphPanel, AnalyticsDashboard

            # Register Knowledge Graph panel
            panel_registry.register_panel(
                name="knowledge_graph",
                panel_class=KnowledgeGraphPanel,
                title="Knowledge Graph",
                description="Interactive knowledge graph visualization"
            )

            # Register Analytics Dashboard panel
            panel_registry.register_panel(
                name="analytics",
                panel_class=AnalyticsDashboard,
                title="Analytics Dashboard",
                description="Knowledge graph metrics and statistics"
            )

            # Store references
            self.panels["knowledge_graph"] = KnowledgeGraphPanel
            self.panels["analytics"] = AnalyticsDashboard

            self._log("info", "Registered 2 panels: knowledge_graph, analytics")

            return True

        except Exception as e:
            self._log("error", f"Panel registration failed: {e}")
            return False

    async def register_menus(self, menu_registry) -> bool:
        """
        Register plugin menu with ROMA's menu registry.

        Args:
            menu_registry: ROMA's menu registry (injected)

        Returns:
            True if registration successful
        """
        try:
            from .menus import KnowledgeGraphMenu

            # Create menu instance with injected dependencies
            kg_menu = KnowledgeGraphMenu(
                roma_client=self.roma_client,
                kg_engine=self.kg_engine
            )

            # Register menu
            menu_registry.register_menu(
                name="knowledge_graph",
                menu=kg_menu,
                title="Knowledge Graph",
                description="Knowledge graph operations and visualizations"
            )

            # Store reference
            self.menus["knowledge_graph"] = kg_menu

            self._log("info", "Registered knowledge_graph menu")

            return True

        except Exception as e:
            self._log("error", f"Menu registration failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Return plugin information.

        Returns:
            Dict with plugin metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "initialized": self._initialized,
            "enabled": self._enabled,
            "features": [
                "Knowledge graph visualization",
                "Analytics dashboard",
                "Interactive exploration",
                "8 custom commands",
                "Hierarchical menu system",
                "Dependency injection",
                "Air gap compliant"
            ],
            "components": {
                "panels": list(self.panels.keys()),
                "commands": list(self.commands.keys()),
                "menus": list(self.menus.keys())
            }
        }

    async def shutdown(self):
        """
        Cleanup plugin resources.

        Called by ROMA when shutting down.
        """
        self._log("info", "Shutting down plugin")

        # Clear references
        self.panels.clear()
        self.commands.clear()
        self.menus.clear()

        self._initialized = False
        self._enabled = False

    def _validate_config(self):
        """Validate plugin configuration."""
        required_keys = []

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    async def _initialize_panels(self):
        """Initialize panel components."""
        from .panels import KnowledgeGraphPanel, AnalyticsDashboard

        # Panel classes are stored for registration
        self.panels = {
            "knowledge_graph": KnowledgeGraphPanel,
            "analytics": AnalyticsDashboard
        }

    async def _initialize_commands(self):
        """Initialize command components."""
        from .commands import KnowledgeGraphCommands

        # Commands are initialized during registration
        pass

    async def _initialize_menus(self):
        """Initialize menu components."""
        from .menus import KnowledgeGraphMenu

        # Menus are initialized during registration
        pass

    def _log(self, level: str, message: str, context: Optional[Dict] = None):
        """
        Log message with context.

        Args:
            level: Log level (info, warning, error)
            message: Log message
            context: Additional context dict
        """
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._log_context,
            **(context or {})
        }

        # If ROMA client has logging, use it
        if self.roma_client and hasattr(self.roma_client, "log"):
            self.roma_client.log(level, message, log_entry)
        else:
            # Fallback to print
            print(f"[{level.upper()}] {message}: {log_entry}")
