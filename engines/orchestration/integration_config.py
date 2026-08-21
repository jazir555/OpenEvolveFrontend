"""
Integration Configuration System - License: Apache 2.0

Centralized configuration management for all integrations.
Supports environment variables, config files, and validation.

Dependencies:
- pydantic: MIT License
- pyyaml: MIT License

Author: OpenEvolve
Date: 2026-02-02
"""
from __future__ import annotations


import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

# Pydantic - MIT License
from pydantic import BaseModel, Field, validator

# YAML - MIT License
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ValkeyConfig(BaseModel):
    """Valkey connection configuration."""
    host: str = Field(default="localhost", description="Valkey server host")
    port: int = Field(default=6379, description="Valkey server port")
    db: int = Field(default=0, description="Valkey database number")
    password: Optional[str] = Field(default=None, description="Valkey password")
    ssl: bool = Field(default=False, description="Use SSL connection")
    connect_timeout: float = Field(default=5.0, description="Connection timeout")
    
    class Config:
        env_prefix = "VALKEY_"


class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry configuration."""
    enabled: bool = Field(default=True, description="Enable telemetry")
    service_name: str = Field(default="openevolve", description="Service name")
    service_version: str = Field(default="1.0.0", description="Service version")
    otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OTLP collector endpoint (e.g., http://localhost:4317)"
    )
    console_export: bool = Field(
        default=False,
        description="Export to console (for development)"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Tracing sample rate"
    )
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    enable_tracing: bool = Field(default=True, description="Enable tracing")
    
    class Config:
        env_prefix = "OTEL_"


class MCPConfig(BaseModel):
    """MCP Server configuration."""
    enabled: bool = Field(default=True, description="Enable MCP server")
    name: str = Field(default="openevolve-unified", description="Server name")
    enable_decomposition_tools: bool = Field(default=True)
    enable_knowledge_tools: bool = Field(default=True)
    enable_z3_tools: bool = Field(default=True)
    enable_leanaide_tools: bool = Field(default=True)
    enable_workflow_tools: bool = Field(default=True)
    
    class Config:
        env_prefix = "MCP_"


class GraphQLConfig(BaseModel):
    """GraphQL API configuration."""
    enabled: bool = Field(default=True, description="Enable GraphQL API")
    port: int = Field(default=8001, description="GraphQL server port")
    host: str = Field(default="0.0.0.0", description="GraphQL server host")
    ide: str = Field(
        default="apollo-sandbox",
        description="GraphQL IDE (apollo-sandbox or graphiql)"
    )
    introspection: bool = Field(
        default=True,
        description="Enable schema introspection"
    )
    max_query_depth: int = Field(
        default=10,
        description="Maximum query nesting depth"
    )
    
    class Config:
        env_prefix = "GRAPHQL_"


class RESTAPIConfig(BaseModel):
    """REST API configuration."""
    enabled: bool = Field(default=True, description="Enable REST API")
    port: int = Field(default=8000, description="REST API server port")
    host: str = Field(default="0.0.0.0", description="REST API server host")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins"
    )
    api_key_required: bool = Field(
        default=False,
        description="Require API key authentication"
    )
    rate_limit: int = Field(
        default=100,
        description="Requests per minute per client"
    )
    
    class Config:
        env_prefix = "REST_"


class EventBusConfig(BaseModel):
    """Event Bus configuration."""
    enabled: bool = Field(default=True, description="Enable event bus")
    valkey: ValkeyConfig = Field(default_factory=ValkeyConfig)
    enable_persistence: bool = Field(default=True)
    max_history: int = Field(default=10000, description="Max events to keep in memory")
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for event notifications"
    )
    
    class Config:
        env_prefix = "EVENT_BUS_"


class DecompositionConfig(BaseModel):
    """Decomposition engine configuration."""
    default_strategy: str = Field(default="hybrid")
    enable_z3_validation: bool = Field(default=True)
    max_subproblems: int = Field(default=10)
    min_subproblems: int = Field(default=2)
    enable_adaptive_selection: bool = Field(default=True)
    
    class Config:
        env_prefix = "DECOMP_"


class KnowledgeEngineConfig(BaseModel):
    """Knowledge engine configuration."""
    enabled_extractors: List[str] = Field(
        default_factory=lambda: ["deepke", "oneke", "kg_gen"]
    )
    default_backend: str = Field(default="memgraph")
    enable_neuralkg: bool = Field(default=True)
    enable_graphiti: bool = Field(default=True)
    
    class Config:
        env_prefix = "KNOWLEDGE_"


class IntegrationConfig(BaseModel):
    """
    Master configuration for all integrations.
    
    Loads from:
    1. Default values
    2. Config file (JSON/YAML)
    3. Environment variables
    """
    
    # Service enablement
    services: Dict[str, bool] = Field(
        default_factory=lambda: {
            "rest_api": True,
            "graphql_api": True,
            "mcp_server": True,
            "event_bus": True,
            "telemetry": True
        }
    )
    
    # Individual configs
    rest_api: RESTAPIConfig = Field(default_factory=RESTAPIConfig)
    graphql: GraphQLConfig = Field(default_factory=GraphQLConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    telemetry: OpenTelemetryConfig = Field(default_factory=OpenTelemetryConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    knowledge: KnowledgeEngineConfig = Field(default_factory=KnowledgeEngineConfig)
    
    # Orchestrator
    orchestrator_port: int = Field(default=8080)
    log_level: str = Field(default="INFO")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'log_level must be one of {allowed}')
        return v.upper()
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "IntegrationConfig":
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        content = path.read_text()
        
        if path.suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
            data = yaml.safe_load(content)
        elif path.suffix == '.json':
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        """Load configuration from environment variables."""
        # Pydantic automatically reads from env with env_prefix
        return cls()
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "IntegrationConfig":
        """
        Load configuration with priority:
        1. Config file (if provided)
        2. Environment variables
        3. Default values
        """
        if config_path:
            return cls.from_file(config_path)
        
        # Check for default config files
        for filename in ['openevolve.yaml', 'openevolve.yml', 'openevolve.json']:
            if Path(filename).exists():
                return cls.from_file(filename)
        
        # Fall back to environment variables
        return cls.from_env()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        data = self.dict()
        
        if path.suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
            path.write_text(yaml.safe_dump(data, default_flow_style=False))
        elif path.suffix == '.json':
            path.write_text(json.dumps(data, indent=2))
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict()
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("OpenEvolve Integration Configuration")
        print("=" * 50)
        print(f"Services:")
        for name, enabled in self.services.items():
            status = "[OK]" if enabled else "[FAIL]"
            print(f"  {status} {name}")
        
        print(f"\nREST API: http://{self.rest_api.host}:{self.rest_api.port}")
        print(f"GraphQL:  http://{self.graphql.host}:{self.graphql.port}/graphql")
        print(f"Telemetry: {'enabled' if self.telemetry.enabled else 'disabled'}")
        print(f"Event Bus: {'enabled' if self.event_bus.enabled else 'disabled'}")


# Default configuration instance
default_config = IntegrationConfig()


def get_config(config_path: Optional[str] = None) -> IntegrationConfig:
    """Get configuration instance."""
    return IntegrationConfig.load(config_path)


# Example configuration file
default_config_yaml = """
# OpenEvolve Integration Configuration
# Copy this to openevolve.yaml and customize

services:
  rest_api: true
  graphql_api: true
  mcp_server: true
  event_bus: true
  telemetry: true

rest_api:
  port: 8000
  host: 0.0.0.0
  cors_origins:
    - "*"
  api_key_required: false
  rate_limit: 100

graphql:
  port: 8001
  host: 0.0.0.0
  ide: apollo-sandbox
  introspection: true
  max_query_depth: 10

mcp:
  enabled: true
  name: openevolve-unified
  enable_decomposition_tools: true
  enable_knowledge_tools: true
  enable_z3_tools: true
  enable_leanaide_tools: true
  enable_workflow_tools: true

event_bus:
  enabled: true
  enable_persistence: true
  max_history: 10000
  valkey:
    host: localhost
    port: 6379
    db: 0
    ssl: false

telemetry:
  enabled: true
  service_name: openevolve
  service_version: 1.0.0
  otlp_endpoint: null
  console_export: false
  sample_rate: 1.0
  enable_metrics: true
  enable_tracing: true

decomposition:
  default_strategy: hybrid
  enable_z3_validation: true
  max_subproblems: 10
  min_subproblems: 2
  enable_adaptive_selection: true

knowledge:
  enabled_extractors:
    - deepke
    - oneke
    - kg_gen
  default_backend: memgraph
  enable_neuralkg: true
  enable_graphiti: true

orchestrator_port: 8080
log_level: INFO
"""


if __name__ == "__main__":
    # Print default configuration
    print(default_config_yaml)
    
    # Print current config
    print("\n\nCurrent Configuration (from env/defaults):")
    print("=" * 50)
    config = get_config()
    config.print_config()
