"""
Configuration for Arbor Integration

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: No magic defaults
- TYPE SAFETY: Use dataclasses with validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class ArborConnectionConfig:
    """Configuration for Arbor server connection."""
    
    ws_url: str = field(default="ws://localhost:7433")
    """WebSocket URL for Arbor server."""
    
    reconnect_interval: float = field(default=5.0)
    """Seconds between reconnection attempts."""
    
    max_reconnects: int = field(default=10)
    """Maximum reconnection attempts before giving up."""
    
    connection_timeout: float = field(default=30.0)
    """Timeout for establishing connection."""
    
    request_timeout: float = field(default=60.0)
    """Timeout for individual requests."""
    
    heartbeat_interval: float = field(default=30.0)
    """Seconds between heartbeat pings."""
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.ws_url.startswith(("ws://", "wss://")):
            raise ValueError(f"ws_url must start with ws:// or wss://: {self.ws_url}")
        if self.reconnect_interval < 0:
            raise ValueError("reconnect_interval must be non-negative")
        if self.max_reconnects < 0:
            raise ValueError("max_reconnects must be non-negative")


@dataclass
class ArborSyncConfig:
    """Configuration for graph synchronization."""
    
    mode: str = field(default="realtime")
    """Sync mode: realtime | batch | manual."""
    
    batch_size: int = field(default=1000)
    """Number of nodes to process in a batch."""
    
    full_sync_interval: int = field(default=3600)
    """Seconds between full syncs (0 = disable)."""
    
    incremental_sync: bool = field(default=True)
    """Enable incremental updates from file watcher."""
    
    debounce_ms: int = field(default=50)
    """Milliseconds to debounce file change events."""
    
    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ["realtime", "batch", "manual"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}: {self.mode}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass
class ArborIndexingConfig:
    """Configuration for codebase indexing."""
    
    languages: List[str] = field(default_factory=lambda: [
        "python", "rust", "typescript", "javascript"
    ])
    """Languages to index."""
    
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.test.py",
        "*_test.py",
        "node_modules/**",
        ".git/**",
        "target/**",
        "dist/**",
        "build/**",
        "__pycache__/**",
        "*.min.js",
        "*.min.css"
    ])
    """Glob patterns to exclude from indexing."""
    
    max_file_size: int = field(default=1_048_576)  # 1MB
    """Maximum file size in bytes."""
    
    follow_symlinks: bool = field(default=False)
    """Follow symbolic links."""
    
    respect_gitignore: bool = field(default=True)
    """Respect .gitignore files."""


@dataclass
class ArborMCPConfig:
    """Configuration for MCP (Model Context Protocol) integration."""
    
    enabled: bool = field(default=True)
    """Enable MCP tools."""
    
    tools: List[str] = field(default_factory=lambda: [
        "arbor_find_definition",
        "arbor_get_callers",
        "arbor_get_callees",
        "arbor_find_path",
        "arbor_analyze_impact",
        "arbor_get_context",
        "arbor_search"
    ])
    """List of enabled MCP tools."""
    
    max_context_depth: int = field(default=3)
    """Maximum depth for context queries."""
    
    max_results: int = field(default=50)
    """Maximum results to return from queries."""


@dataclass
class ArborConfig:
    """Main configuration for Arbor integration."""
    
    enabled: bool = field(default=True)
    """Enable Arbor integration."""
    
    connection: ArborConnectionConfig = field(
        default_factory=ArborConnectionConfig
    )
    """Connection configuration."""
    
    sync: ArborSyncConfig = field(
        default_factory=ArborSyncConfig
    )
    """Synchronization configuration."""
    
    indexing: ArborIndexingConfig = field(
        default_factory=ArborIndexingConfig
    )
    """Indexing configuration."""
    
    mcp: ArborMCPConfig = field(
        default_factory=ArborMCPConfig
    )
    """MCP configuration."""
    
    storage_prefix: str = field(default="arbor")
    """Prefix for stored entities/relationships."""
    
    debug: bool = field(default=False)
    """Enable debug logging."""
    
    @classmethod
    def from_env(cls) -> "ArborConfig":
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("ARBOR_ENABLED", "true").lower() == "true",
            connection=ArborConnectionConfig(
                ws_url=os.getenv("ARBOR_WS_URL", "ws://localhost:7433"),
                reconnect_interval=float(os.getenv("ARBOR_RECONNECT_INTERVAL", "5.0")),
                max_reconnects=int(os.getenv("ARBOR_MAX_RECONNECTS", "10")),
                connection_timeout=float(os.getenv("ARBOR_CONNECTION_TIMEOUT", "30.0")),
                request_timeout=float(os.getenv("ARBOR_REQUEST_TIMEOUT", "60.0"))
            ),
            sync=ArborSyncConfig(
                mode=os.getenv("ARBOR_SYNC_MODE", "realtime"),
                batch_size=int(os.getenv("ARBOR_BATCH_SIZE", "1000")),
                full_sync_interval=int(os.getenv("ARBOR_FULL_SYNC_INTERVAL", "3600"))
            ),
            debug=os.getenv("ARBOR_DEBUG", "false").lower() == "true"
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArborConfig":
        """Create configuration from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            connection=ArborConnectionConfig(**data.get("connection", {})),
            sync=ArborSyncConfig(**data.get("sync", {})),
            indexing=ArborIndexingConfig(**data.get("indexing", {})),
            mcp=ArborMCPConfig(**data.get("mcp", {})),
            storage_prefix=data.get("storage_prefix", "arbor"),
            debug=data.get("debug", False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "connection": {
                "ws_url": self.connection.ws_url,
                "reconnect_interval": self.connection.reconnect_interval,
                "max_reconnects": self.connection.max_reconnects,
                "connection_timeout": self.connection.connection_timeout,
                "request_timeout": self.connection.request_timeout,
                "heartbeat_interval": self.connection.heartbeat_interval
            },
            "sync": {
                "mode": self.sync.mode,
                "batch_size": self.sync.batch_size,
                "full_sync_interval": self.sync.full_sync_interval,
                "incremental_sync": self.sync.incremental_sync,
                "debounce_ms": self.sync.debounce_ms
            },
            "indexing": {
                "languages": self.indexing.languages,
                "exclude_patterns": self.indexing.exclude_patterns,
                "max_file_size": self.indexing.max_file_size,
                "follow_symlinks": self.indexing.follow_symlinks,
                "respect_gitignore": self.indexing.respect_gitignore
            },
            "mcp": {
                "enabled": self.mcp.enabled,
                "tools": self.mcp.tools,
                "max_context_depth": self.mcp.max_context_depth,
                "max_results": self.mcp.max_results
            },
            "storage_prefix": self.storage_prefix,
            "debug": self.debug
        }
