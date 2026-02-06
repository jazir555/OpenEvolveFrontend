"""
MCP Gateway Integration for OpenEvolve Knowledge Engine

This module provides integration with the Model Context Protocol (MCP) Gateway,
enabling standardized tool orchestration and coordination across multiple systems.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class MCPResult:
    """Result of an MCP operation."""
    success: bool
    output: Any
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'output': self.output,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class MCPGatewayIntegration:
    """
    Integration with MCP Gateway for tool orchestration.
    
    Provides methods for:
    - Tool discovery and registration
    - Tool execution and orchestration
    - Multi-server coordination
    - Standardized tool calling
    - Response aggregation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MCP Gateway integration.
        
        Args:
            config: Configuration for MCP Gateway components
        """
        self.config = config or self._get_default_config()
        
        # Initialize MCP Gateway components
        self.unified_gateway = None
        self.tool_registry = None
        self.tool_router = None
        
        # Initialize based on configuration
        self._initialize_components()
        
        logger.info({
            "msg": "MCPGatewayIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for MCP Gateway integration."""
        return {
            "gateway_url": "http://localhost:8080",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "reset_timeout": 30000,
                "success_threshold": 2
            },
            "load_balancing": "round_robin",  # round_robin, random, weighted
            "fallback_enabled": True,
            "metrics_enabled": True,
            "cache_enabled": True,
            "cache_ttl": 300,  # seconds
            "supported_namespaces": [
                "kggen",      # Knowledge graph generation
                "graphiti",   # Temporal knowledge graphs
                "openevolve", # Evolution and adaptation
                "crewai",     # Multi-agent coordination
                "deepke",     # Knowledge extraction
                "ragbits",    # Retrieval-augmented generation
                "oneke",      # Bilingual extraction
                "aikg",       # AI knowledge graphs
                "researchquest", # Research automation
                "agentic",    # Agentic systems
                "agentjson",  # Structured data
                "dspy",       # Program-of-thought prompting
                "leanaide",   # Formal verification
                "openevolve_integration"  # OpenEvolve integration library
            ],
            "default_namespace": "openevolve",
            "tool_call_timeout": 120  # seconds
        }
    
    def _initialize_components(self):
        """Initialize MCP Gateway components based on configuration."""
        try:
            # Import MCP Gateway components
            from mcp.gateway.unified_mcp_gateway import UnifiedMCPGateway
            from mcp.gateway.models import ToolDefinition, ServerConfig, ToolCategory
            
            # Initialize the unified gateway
            self.unified_gateway = UnifiedMCPGateway(config_path=self.config.get("config_path"))
            
            # Initialize components
            self.tool_registry = self.unified_gateway.tool_registry
            self.tool_router = self.unified_gateway.tool_router
            
            # Initialize the gateway
            asyncio.run(self.unified_gateway.initialize())
            
            logger.info({
                "msg": "MCP Gateway components initialized successfully",
                "gateway_url": self.config.get("gateway_url", "unknown"),
                "supported_namespaces": self.config.get("supported_namespaces", []),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError as e:
            logger.warning({
                "msg": f"MCP Gateway not available, using mock implementation: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize MCP Gateway components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def _initialize_mock_components(self):
        """Initialize mock components when MCP Gateway is not available."""
        logger.warning({
            "msg": "MCP Gateway not available - components will fail on use",
            "install": "pip install mcp-gateway",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create failing mock implementations
        from ..optional_imports import create_failing_mock
        
        MockToolRegistry = create_failing_mock(
            package_name='mcp-gateway',
            feature_name='MCP Tool Registry',
            install_command='pip install mcp-gateway'
        )
        
        MockToolRouter = create_failing_mock(
            package_name='mcp-gateway',
            feature_name='MCP Tool Router',
            install_command='pip install mcp-gateway'
        )
        
        MockUnifiedGateway = create_failing_mock(
            package_name='mcp-gateway',
            feature_name='MCP Unified Gateway',
            install_command='pip install mcp-gateway'
        )
        
        self._mock_classes = {
            'tool_registry': MockToolRegistry,
            'tool_router': MockToolRouter,
            'unified_gateway': MockUnifiedGateway
        }
        self.tool_registry = None
        self.tool_router = None
        self.unified_gateway = None
    
    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        namespace: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> MCPResult:
        """
        Call a tool through the MCP Gateway.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters for the tool
            namespace: Namespace for the tool (optional)
            correlation_id: Correlation ID for tracking
            
        Returns:
            MCPResult with tool execution results
        """
        correlation_id = correlation_id or f"mcp_call_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway tool call",
            "tool_name": tool_name,
            "namespace": namespace,
            "params_count": len(params),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Format tool name with namespace if provided
            full_tool_name = tool_name
            if namespace:
                full_tool_name = f"{namespace}/{tool_name}"
            
            # Call the tool through the gateway
            result = await self.unified_gateway.call_tool(full_tool_name, params)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            mcp_result = MCPResult(
                success=result.success,
                output=result.result if hasattr(result, 'result') else result,
                metadata={
                    "tool_name": tool_name,
                    "namespace": namespace or "default",
                    "server_name": getattr(result, 'server_name', 'unknown'),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "MCP Gateway tool call completed",
                "correlation_id": correlation_id,
                "tool_name": tool_name,
                "namespace": namespace,
                "success": result.success,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return mcp_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway tool call failed",
                "correlation_id": correlation_id,
                "tool_name": tool_name,
                "namespace": namespace,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return MCPResult(
                success=False,
                output=None,
                metadata={
                    "tool_name": tool_name,
                    "namespace": namespace or "default",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def discover_tools(
        self,
        namespace: Optional[str] = None,
        category: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> MCPResult:
        """
        Discover available tools through the MCP Gateway.
        
        Args:
            namespace: Optional namespace to filter tools
            category: Optional category to filter tools
            correlation_id: Correlation ID for tracking
            
        Returns:
            MCPResult with list of available tools
        """
        correlation_id = correlation_id or f"mcp_discover_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway tool discovery",
            "namespace": namespace,
            "category": category,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Discover tools
            tools = await self.unified_gateway.list_tools(
                namespace=namespace or "",
                category=category
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            mcp_result = MCPResult(
                success=True,
                output=tools,
                metadata={
                    "discovered_count": len(tools),
                    "namespace_filter": namespace,
                    "category_filter": category,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "MCP Gateway tool discovery completed",
                "correlation_id": correlation_id,
                "tools_count": len(tools),
                "namespace_filter": namespace,
                "category_filter": category,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return mcp_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway tool discovery failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "namespace_filter": namespace,
                "category_filter": category,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return MCPResult(
                success=False,
                output=[],
                metadata={
                    "discovered_count": 0,
                    "namespace_filter": namespace,
                    "category_filter": category,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def execute_knowledge_extraction_workflow(
        self,
        text: str,
        extraction_types: List[str] = None,
        correlation_id: Optional[str] = None
    ) -> MCPResult:
        """
        Execute a knowledge extraction workflow using multiple MCP tools.
        
        Args:
            text: Text to extract knowledge from
            extraction_types: Types of extraction to perform
            correlation_id: Correlation ID for tracking
            
        Returns:
            MCPResult with workflow results
        """
        correlation_id = correlation_id or f"mcp_kg_wf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway knowledge extraction workflow",
            "text_length": len(text),
            "extraction_types": extraction_types or ["entities", "relations", "triples"],
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Default extraction types
            if extraction_types is None:
                extraction_types = ["entities", "relations", "triples"]
            
            # Initialize results
            workflow_results = {}
            
            # Extract entities using appropriate tool
            if "entities" in extraction_types:
                entity_result = await self.call_tool(
                    tool_name="extract_entities",
                    params={"text": text},
                    namespace="openevolve_integration",  # Using the integration library namespace
                    correlation_id=f"{correlation_id}_entities"
                )
                workflow_results["entities"] = entity_result
                
                if not entity_result.success:
                    logger.warning({
                        "msg": "Entity extraction failed in workflow",
                        "correlation_id": f"{correlation_id}_entities",
                        "error": entity_result.error
                    })
            
            # Extract relations using appropriate tool
            if "relations" in extraction_types:
                relation_result = await self.call_tool(
                    tool_name="extract_relations",
                    params={"text": text},
                    namespace="deepke",  # Using DeepKE namespace
                    correlation_id=f"{correlation_id}_relations"
                )
                workflow_results["relations"] = relation_result
                
                if not relation_result.success:
                    logger.warning({
                        "msg": "Relation extraction failed in workflow",
                        "correlation_id": f"{correlation_id}_relations",
                        "error": relation_result.error
                    })
            
            # Extract triples using appropriate tool
            if "triples" in extraction_types:
                triple_result = await self.call_tool(
                    tool_name="extract_triples",
                    params={"text": text},
                    namespace="aikg",  # Using AIKG namespace
                    correlation_id=f"{correlation_id}_triples"
                )
                workflow_results["triples"] = triple_result
                
                if not triple_result.success:
                    logger.warning({
                        "msg": "Triple extraction failed in workflow",
                        "correlation_id": f"{correlation_id}_triples",
                        "error": triple_result.error
                    })
            
            # Extract knowledge graph using appropriate tool
            if "graph" in extraction_types:
                graph_result = await self.call_tool(
                    tool_name="build_knowledge_graph",
                    params={"text": text},
                    namespace="graphiti",  # Using Graphiti namespace
                    correlation_id=f"{correlation_id}_graph"
                )
                workflow_results["graph"] = graph_result
                
                if not graph_result.success:
                    logger.warning({
                        "msg": "Knowledge graph extraction failed in workflow",
                        "correlation_id": f"{correlation_id}_graph",
                        "error": graph_result.error
                    })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Aggregate results
            aggregated_output = {}
            success_count = 0
            errors = []

            for ext_type, result in workflow_results.items():
                if result.success:
                    aggregated_output[ext_type] = result.output
                    success_count += 1
                else:
                    aggregated_output[ext_type] = None
                    if result.error:
                        errors.append(f"{ext_type}: {result.error}")

            # If all extractions failed, include error message
            error_msg = "; ".join(errors) if errors else None

            mcp_result = MCPResult(
                success=success_count > 0,  # Success if at least one extraction succeeded
                output=aggregated_output,
                metadata={
                    "extraction_types": extraction_types,
                    "successful_extractions": success_count,
                    "total_extractions": len(workflow_results),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=error_msg if success_count == 0 else None
            )
            
            logger.info({
                "msg": "MCP Gateway knowledge extraction workflow completed",
                "correlation_id": correlation_id,
                "successful_extractions": success_count,
                "total_extractions": len(workflow_results),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return mcp_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway knowledge extraction workflow failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return MCPResult(
                success=False,
                output={},
                metadata={
                    "extraction_types": extraction_types or ["entities", "relations", "triples"],
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def execute_multi_agent_coordination(
        self,
        task_description: str,
        agent_preferences: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> MCPResult:
        """
        Execute multi-agent coordination using MCP tools.
        
        Args:
            task_description: Description of task to coordinate
            agent_preferences: Preferences for agent selection
            correlation_id: Correlation ID for tracking
            
        Returns:
            MCPResult with coordination results
        """
        correlation_id = correlation_id or f"mcp_coord_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway multi-agent coordination",
            "task_description_length": len(task_description),
            "agent_preferences_count": len(agent_preferences) if agent_preferences else 0,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Use CrewAI integration through MCP
            coordination_result = await self.call_tool(
                tool_name="coordinate_agents",
                params={
                    "task": task_description,
                    "preferences": agent_preferences or {},
                    "max_agents": 5
                },
                namespace="crewai",
                correlation_id=f"{correlation_id}_coord"
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            mcp_result = MCPResult(
                success=coordination_result.success,
                output=coordination_result.output,
                metadata={
                    "task_description_length": len(task_description),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=coordination_result.error if not coordination_result.success else None
            )
            
            logger.info({
                "msg": "MCP Gateway multi-agent coordination completed",
                "correlation_id": correlation_id,
                "success": coordination_result.success,
                "output_length": len(str(coordination_result.output)) if coordination_result.output else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return mcp_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway multi-agent coordination failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return MCPResult(
                success=False,
                output=None,
                metadata={
                    "task_description_length": len(task_description),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def execute_formal_verification(
        self,
        theorem: str,
        proof: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> MCPResult:
        """
        Execute formal verification using MCP tools.
        
        Args:
            theorem: Theorem to verify
            proof: Optional proof to verify
            correlation_id: Correlation ID for tracking
            
        Returns:
            MCPResult with verification results
        """
        correlation_id = correlation_id or f"mcp_verify_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway formal verification",
            "theorem_length": len(theorem),
            "proof_provided": proof is not None,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Use LeanAide integration through MCP
            verification_params = {
                "theorem": theorem
            }
            if proof:
                verification_params["proof"] = proof
            
            verification_result = await self.call_tool(
                tool_name="verify_theorem",
                params=verification_params,
                namespace="leanaide",
                correlation_id=f"{correlation_id}_verify"
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            mcp_result = MCPResult(
                success=verification_result.success,
                output=verification_result.output,
                metadata={
                    "theorem_length": len(theorem),
                    "proof_provided": proof is not None,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=verification_result.error if not verification_result.success else None
            )
            
            logger.info({
                "msg": "MCP Gateway formal verification completed",
                "correlation_id": correlation_id,
                "success": verification_result.success,
                "verified": verification_result.output.get("verified", False) if isinstance(verification_result.output, dict) else False,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return mcp_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway formal verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return MCPResult(
                success=False,
                output=None,
                metadata={
                    "theorem_length": len(theorem),
                    "proof_provided": proof is not None,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def batch_execute(
        self,
        tool_calls: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> List[MCPResult]:
        """
        Execute multiple tool calls in batch through MCP Gateway.
        
        Args:
            tool_calls: List of tool call specifications
                Each dict should have: {
                    "tool_name": str,
                    "params": dict,
                    "namespace": str (optional)
                }
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of MCPResult objects
        """
        correlation_id = correlation_id or f"mcp_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting MCP Gateway batch execution",
            "tool_calls_count": len(tool_calls),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.unified_gateway:
                raise RuntimeError("MCP Gateway not initialized")
            
            # Execute all tool calls in parallel
            tasks = []
            for i, call_spec in enumerate(tool_calls):
                task = self.call_tool(
                    tool_name=call_spec["tool_name"],
                    params=call_spec.get("params", {}),
                    namespace=call_spec.get("namespace"),
                    correlation_id=f"{correlation_id}_call_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch call {i} failed",
                        "correlation_id": f"{correlation_id}_call_{i}",
                        "error": str(result)
                    })
                    processed_results.append(MCPResult(
                        success=False,
                        output=None,
                        metadata={"batch_index": i, "error": str(result)},
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            successful_count = sum(1 for r in processed_results if r.success)
            
            logger.info({
                "msg": "MCP Gateway batch execution completed",
                "correlation_id": correlation_id,
                "tool_calls_count": len(tool_calls),
                "successful_count": successful_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "MCP Gateway batch execution failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all calls
            error_results = []
            for i in range(len(tool_calls)):
                error_results.append(MCPResult(
                    success=False,
                    output=None,
                    metadata={"batch_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(tool_calls) if tool_calls else 0.0,
                    error=str(e)
                ))
            
            return error_results
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get the status of the MCP Gateway integration.
        
        Returns:
            Dictionary with integration status
        """
        try:
            if self.unified_gateway and hasattr(self.unified_gateway, 'get_health_status'):
                status = asyncio.run(self.unified_gateway.get_health_status())
                return {
                    "available": True,
                    "initialized": self.unified_gateway.is_initialized,
                    "running": self.unified_gateway.is_running,
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                # Return mock status
                return {
                    "available": self.unified_gateway is not None,
                    "initialized": getattr(self.unified_gateway, 'is_initialized', False) if self.unified_gateway else False,
                    "running": getattr(self.unified_gateway, 'is_running', False) if self.unified_gateway else False,
                    "status": {
                        "gateway": {
                            "status": "running" if (self.unified_gateway and getattr(self.unified_gateway, 'is_running', False)) else "not_initialized",
                            "initialized": getattr(self.unified_gateway, 'is_initialized', False) if self.unified_gateway else False,
                            "uptime_seconds": 0
                        },
                        "servers": {},
                        "tools": len(self.tool_registry.tools) if self.tool_registry else 0
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.error({
                "msg": "Failed to get MCP Gateway status",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {
                "available": False,
                "initialized": False,
                "running": False,
                "status": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing MCP Gateway integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Shutdown the unified gateway if available
        if self.unified_gateway and hasattr(self.unified_gateway, 'shutdown'):
            try:
                await self.unified_gateway.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down unified gateway: {e}")
        
        logger.info({
            "msg": "MCP Gateway integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# Availability flag
try:
    import mcp_gateway
    MCP_GATEWAY_INTEGRATION_AVAILABLE = True
except ImportError:
    MCP_GATEWAY_INTEGRATION_AVAILABLE = False
