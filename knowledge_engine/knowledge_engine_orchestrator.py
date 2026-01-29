"""
OpenEvolve Knowledge Engine - Main Orchestrator

This module provides the main orchestrator for all knowledge engine integrations,
coordinating between different systems and providing unified access to all
knowledge processing capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid

try:
    from .integrations.graphiti_integration import GraphitiIntegration
except ImportError:
    from integrations.graphiti_integration import GraphitiIntegration

try:
    from .integrations.kggen_integration import KGGenIntegration
except ImportError:
    from integrations.kggen_integration import KGGenIntegration

try:
    from .integrations.oneke_integration import OneKEIntegration
except ImportError:
    from integrations.oneke_integration import OneKEIntegration

try:
    from .integrations.aikg_integration import AIKGIntegration
except ImportError:
    from integrations.aikg_integration import AIKGIntegration

try:
    from .integrations.ragbits_integration import RagbitsIntegration
except ImportError:
    from integrations.ragbits_integration import RagbitsIntegration

try:
    from .integrations.crewai_integration import CrewAIIntegration
except ImportError:
    from integrations.crewai_integration import CrewAIIntegration

try:
    from .integrations.deepke_integration import DeepKEIntegration
except ImportError:
    from integrations.deepke_integration import DeepKEIntegration

try:
    from .integrations.research_quest_integration import ResearchQuestIntegration
except ImportError:
    from integrations.research_quest_integration import ResearchQuestIntegration

try:
    from .integrations.agentic_context_integration import AgenticContextEngine
except ImportError:
    from integrations.agentic_context_integration import AgenticContextEngine

try:
    from .integrations.agentjson_integration import AgentJSONIntegration
except ImportError:
    from integrations.agentjson_integration import AgentJSONIntegration

try:
    from .integrations.dspy_integration import DSPyIntegration
except ImportError:
    from integrations.dspy_integration import DSPyIntegration

try:
    from .integrations.leanaide_integration import LeanAideIntegration
except ImportError:
    from integrations.leanaide_integration import LeanAideIntegration

try:
    from .integrations.openevolve_integration_library import OpenEvolveIntegrationLibrary
except ImportError:
    from integrations.openevolve_integration_library import OpenEvolveIntegrationLibrary

try:
    from .integrations.mcp_gateway_integration import MCPGatewayIntegration
except ImportError:
    from integrations.mcp_gateway_integration import MCPGatewayIntegration


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEngineResult:
    """Result from a knowledge engine operation."""
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


class KnowledgeEngineOrchestrator:
    """
    Main orchestrator for the OpenEvolve Knowledge Engine.
    
    Coordinates all integrated systems:
    - Graphiti temporal knowledge graphs
    - KG-Gen knowledge extraction
    - OneKE bilingual extraction
    - AI-Knowledge-Graph processing
    - Ragbits retrieval-augmented generation
    - CrewAI multi-agent framework
    - DeepKE knowledge extraction
    - Research-Quest research automation
    - Agentic Context Engine
    - AgentJSON structured data
    - DSPy program-of-thought prompting
    - LeanAide formal verification
    - OpenEvolve Integration Library
    - MCP Gateway tool orchestration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Knowledge Engine orchestrator.
        
        Args:
            config: Configuration for all integrated components
        """
        self.config = config or self._get_default_config()
        
        # Initialize all integrated components
        self.graphiti = GraphitiIntegration(
            config=self.config.get("graphiti", {})
        )
        self.kggen = KGGenIntegration(
            config=self.config.get("kggen", {})
        )
        self.oneke = OneKEIntegration(
            config=self.config.get("oneke", {})
        )
        self.aikg = AIKGIntegration(
            config=self.config.get("aikg", {})
        )
        self.ragbits = RagbitsIntegration(
            config=self.config.get("ragbits", {})
        )
        self.crewai = CrewAIIntegration(
            config=self.config.get("crewai", {})
        )
        self.deepke = DeepKEIntegration(
            config=self.config.get("deepke", {})
        )
        self.research_quest = ResearchQuestIntegration(
            config=self.config.get("research_quest", {})
        )
        self.agentic_context = AgenticContextEngine(
            config=self.config.get("agentic_context", {})
        )
        self.agentjson = AgentJSONIntegration(
            config=self.config.get("agentjson", {})
        )
        self.dspy = DSPyIntegration(
            config=self.config.get("dspy", {})
        )
        self.leanaide = LeanAideIntegration(
            config=self.config.get("leanaide", {})
        )
        self.openevolve_lib = OpenEvolveIntegrationLibrary(
            config=self.config.get("openevolve_lib", {})
        )
        self.mcp_gateway = MCPGatewayIntegration(
            config=self.config.get("mcp_gateway", {})
        )
        
        # Component registry for dynamic access
        self.components = {
            'graphiti': self.graphiti,
            'kggen': self.kggen,
            'oneke': self.oneke,
            'aikg': self.aikg,
            'ragbits': self.ragbits,
            'crewai': self.crewai,
            'deepke': self.deepke,
            'research_quest': self.research_quest,
            'agentic_context': self.agentic_context,
            'agentjson': self.agentjson,
            'dspy': self.dspy,
            'leanaide': self.leanaide,
            'openevolve_lib': self.openevolve_lib,
            'mcp_gateway': self.mcp_gateway
        }
        
        logger.info({
            "msg": "KnowledgeEngineOrchestrator initialized",
            "components_count": len(self.components),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all components."""
        return {
            "graphiti": {
                "api_key": None,
                "base_url": "http://localhost:8000",
                "timeout": 30
            },
            "kggen": {
                "model": "gpt-4o",
                "api_key": None,
                "max_tokens": 4096
            },
            "oneke": {
                "model": "gpt-4o",
                "api_key": None,
                "language": "en"
            },
            "aikg": {
                "model": "gpt-4o",
                "api_key": None,
                "embedding_model": "text-embedding-ada-002"
            },
            "ragbits": {
                "model": "gpt-4o",
                "api_key": None,
                "vector_store": "qdrant"
            },
            "crewai": {
                "model": "gpt-4o",
                "api_key": None,
                "max_rpm": 100
            },
            "deepke": {
                "model": "gpt-4o",
                "api_key": None,
                "task_type": "relation_extraction"
            },
            "research_quest": {
                "model": "gpt-4o",
                "api_key": None,
                "max_workers": 4
            },
            "agentic_context": {
                "model": "gpt-4o",
                "api_key": None,
                "max_reflection_rounds": 3
            },
            "agentjson": {
                "model": "gpt-4o",
                "api_key": None,
                "top_k": 5
            },
            "dspy": {
                "model": "gpt-4o",
                "api_key": None,
                "max_hops": 3
            },
            "leanaide": {
                "model": "gpt-4o",
                "api_key": None,
                "max_proof_depth": 10
            },
            "openevolve_lib": {
                "api_key": None,
                "base_url": "http://localhost:8000"
            },
            "mcp_gateway": {
                "gateway_url": "http://localhost:8080",
                "timeout": 30
            }
        }
    
    async def process_knowledge_request(
        self,
        query: str,
        components: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> KnowledgeEngineResult:
        """
        Process a knowledge request using multiple integrated components.
        
        Args:
            query: Knowledge query to process
            components: List of component names to use (if None, use all)
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeEngineResult with combined results
        """
        correlation_id = correlation_id or f"ke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge engine processing",
            "query_length": len(query),
            "components_requested": components or list(self.components.keys()),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use all components if none specified
            if components is None:
                components = list(self.components.keys())
            
            # Filter to only valid components
            valid_components = [c for c in components if c in self.components]
            invalid_components = [c for c in components if c not in self.components]
            
            if invalid_components:
                logger.warning({
                    "msg": "Invalid components requested",
                    "invalid_components": invalid_components,
                    "correlation_id": correlation_id
                })
            
            # Execute requests in parallel across selected components
            tasks = []
            for comp_name in valid_components:
                component = self.components[comp_name]
                
                # Different components may have different methods for processing queries
                if comp_name == 'graphiti':
                    task = component.search_with_temporal_filters(
                        query=query,
                        correlation_id=f"{correlation_id}_graphiti"
                    )
                elif comp_name == 'kggen':
                    task = component.extract_knowledge_graph(
                        text=query,
                        correlation_id=f"{correlation_id}_kggen"
                    )
                elif comp_name == 'oneke':
                    task = component.extract_knowledge_bilingual(
                        text=query,
                        correlation_id=f"{correlation_id}_oneke"
                    )
                elif comp_name == 'aikg':
                    task = component.process_with_ai_kg(
                        text=query,
                        correlation_id=f"{correlation_id}_aikg"
                    )
                elif comp_name == 'ragbits':
                    task = component.search_documents(
                        query=query,
                        correlation_id=f"{correlation_id}_ragbits"
                    )
                elif comp_name == 'crewai':
                    task = component.execute_analysis(
                        text=query,
                        correlation_id=f"{correlation_id}_crewai"
                    )
                elif comp_name == 'deepke':
                    task = component.extract_knowledge(
                        text=query,
                        correlation_id=f"{correlation_id}_deepke"
                    )
                elif comp_name == 'research_quest':
                    task = component.extract_knowledge(
                        text=query,
                        correlation_id=f"{correlation_id}_research_quest"
                    )
                elif comp_name == 'agentic_context':
                    task = component.process_with_adaptive_learning(
                        text=query,
                        correlation_id=f"{correlation_id}_agentic_context"
                    )
                elif comp_name == 'agentjson':
                    task = component.parse_json(
                        text=query,
                        correlation_id=f"{correlation_id}_agentjson"
                    )
                elif comp_name == 'dspy':
                    task = component.chain_of_thought(
                        question=query,
                        correlation_id=f"{correlation_id}_dspy"
                    )
                elif comp_name == 'leanaide':
                    task = component.verify_theorem(
                        theorem=query,
                        correlation_id=f"{correlation_id}_leanaide"
                    )
                elif comp_name == 'openevolve_lib':
                    task = component.execute_integration(
                        integration_name="knowledge",
                        operation="extract",
                        input_data={"text": query},
                        correlation_id=f"{correlation_id}_openevolve_lib"
                    )
                elif comp_name == 'mcp_gateway':
                    task = component.call_tool(
                        tool_name="knowledge_extraction",
                        params={"query": query},
                        correlation_id=f"{correlation_id}_mcp_gateway"
                    )
                else:
                    # Default to a generic call if component doesn't have specific method
                    if hasattr(component, 'process'):
                        task = component.process(query, correlation_id=f"{correlation_id}_{comp_name}")
                    elif hasattr(component, 'search'):
                        task = component.search(query, correlation_id=f"{correlation_id}_{comp_name}")
                    else:
                        # Skip component if no appropriate method found
                        continue
                
                tasks.append(asyncio.create_task(
                    self._execute_component_task(comp_name, task, f"{correlation_id}_{comp_name}")
                ))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            success_count = 0
            
            for i, result in enumerate(results):
                comp_name = valid_components[i]
                
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Component {comp_name} failed",
                        "correlation_id": f"{correlation_id}_{comp_name}",
                        "error": str(result)
                    })
                    processed_results[comp_name] = {
                        "success": False,
                        "error": str(result),
                        "output": None
                    }
                else:
                    processed_results[comp_name] = result
                    if result.get("success", False):
                        success_count += 1
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Create combined result
            combined_result = KnowledgeEngineResult(
                success=success_count > 0,  # Success if at least one component succeeded
                output=processed_results,
                metadata={
                    "components_requested": components,
                    "components_valid": valid_components,
                    "components_invalid": invalid_components,
                    "successful_components": success_count,
                    "total_components": len(valid_components),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Knowledge engine processing completed",
                "correlation_id": correlation_id,
                "successful_components": success_count,
                "total_components": len(valid_components),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return combined_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge engine processing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=False,
                output={},
                metadata={
                    "components_requested": components or list(self.components.keys()),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def _execute_component_task(self, comp_name: str, task, correlation_id: str):
        """Execute a component task and return formatted result."""
        try:
            result = await task
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            else:
                return {
                    "success": True,
                    "output": result,
                    "metadata": {"component": comp_name}
                }
        except Exception as e:
            logger.error({
                "msg": f"Component {comp_name} execution failed",
                "correlation_id": correlation_id,
                "error": str(e)
            })
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "metadata": {"component": comp_name}
            }
    
    async def run_comprehensive_analysis(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> KnowledgeEngineResult:
        """
        Run comprehensive knowledge analysis using multiple components.
        
        Args:
            text: Text to analyze
            analysis_types: Types of analysis to perform
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeEngineResult with analysis results
        """
        correlation_id = correlation_id or f"ke_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting comprehensive knowledge analysis",
            "text_length": len(text),
            "analysis_types": analysis_types or ["entities", "relations", "patterns", "insights"],
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Define which components to use for different analysis types
            analysis_mapping = {
                "entities": ["oneke", "deepke", "aikg"],
                "relations": ["deepke", "aikg", "kggen"],
                "patterns": ["dspy", "research_quest", "crewai"],
                "insights": ["dspy", "research_quest", "agentic_context"],
                "verification": ["leanaide", "research_quest"],
                "context": ["agentic_context", "graphiti"]
            }
            
            # Determine which components to use
            if analysis_types is None:
                analysis_types = list(analysis_mapping.keys())
            
            components_to_use = set()
            for analysis_type in analysis_types:
                if analysis_type in analysis_mapping:
                    components_to_use.update(analysis_mapping[analysis_type])
            
            # Execute analysis tasks in parallel
            tasks = []
            for comp_name in components_to_use:
                if comp_name in self.components:
                    component = self.components[comp_name]
                    
                    # Execute appropriate analysis based on component
                    if comp_name == 'oneke':
                        task = component.extract_knowledge_bilingual(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'deepke':
                        task = component.extract_knowledge(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'aikg':
                        task = component.process_with_ai_kg(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'kggen':
                        task = component.extract_knowledge_graph(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'dspy':
                        task = component.chain_of_thought(
                            question=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'research_quest':
                        task = component.extract_knowledge(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'agentic_context':
                        task = component.process_with_adaptive_learning(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'graphiti':
                        task = component.search_with_temporal_filters(
                            query=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'crewai':
                        task = component.execute_analysis(
                            text=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    elif comp_name == 'leanaide':
                        task = component.verify_theorem(
                            theorem=text,
                            correlation_id=f"{correlation_id}_{comp_name}"
                        )
                    else:
                        # Default processing
                        if hasattr(component, 'analyze'):
                            task = component.analyze(text, correlation_id=f"{correlation_id}_{comp_name}")
                        elif hasattr(component, 'process'):
                            task = component.process(text, correlation_id=f"{correlation_id}_{comp_name}")
                        else:
                            continue  # Skip if no appropriate method
                    
                    tasks.append(asyncio.create_task(
                        self._execute_component_task(comp_name, task, f"{correlation_id}_{comp_name}")
                    ))
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            success_count = 0
            
            for i, result in enumerate(list(components_to_use)):
                if i < len(results):
                    comp_result = results[i]
                    if isinstance(comp_result, Exception):
                        logger.error({
                            "msg": f"Analysis component failed",
                            "component": result,  # This is actually the component name from components_to_use
                            "correlation_id": f"{correlation_id}_{result}",
                            "error": str(comp_result)
                        })
                        processed_results[result] = {
                            "success": False,
                            "error": str(comp_result),
                            "output": None
                        }
                    else:
                        processed_results[result] = comp_result
                        if comp_result.get("success", False):
                            success_count += 1
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Aggregate results by analysis type
            aggregated_results = {}
            for analysis_type in analysis_types:
                if analysis_type in analysis_mapping:
                    type_results = {}
                    for comp_name in analysis_mapping[analysis_type]:
                        if comp_name in processed_results:
                            type_results[comp_name] = processed_results[comp_name]
                    aggregated_results[analysis_type] = type_results
            
            analysis_result = KnowledgeEngineResult(
                success=success_count > 0,
                output=aggregated_results,
                metadata={
                    "analysis_types": analysis_types,
                    "components_used": list(components_to_use),
                    "successful_components": success_count,
                    "total_components": len(components_to_use),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Comprehensive knowledge analysis completed",
                "correlation_id": correlation_id,
                "analysis_types_count": len(analysis_types),
                "successful_components": success_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return analysis_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Comprehensive knowledge analysis failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=False,
                output={},
                metadata={
                    "analysis_types": analysis_types or [],
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of all integrated systems.
        
        Returns:
            Dictionary with status information for all components
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Getting knowledge engine system status",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Get status from all components in parallel
            status_tasks = []
            for name, component in self.components.items():
                if hasattr(component, 'get_status'):
                    task = component.get_status()
                elif hasattr(component, 'health_check'):
                    task = component.health_check()
                else:
                    # Default status for components without status method
                    task = asyncio.sleep(0)  # Immediate completion
                    task = {"available": hasattr(component, '__class__'), "initialized": True}
                
                if asyncio.iscoroutine(task):
                    status_tasks.append(asyncio.create_task(
                        self._get_component_status(name, task)
                    ))
                else:
                    # Handle non-coroutine status methods
                    try:
                        status = await asyncio.get_event_loop().run_in_executor(None, lambda: task)
                        status_tasks.append(asyncio.sleep(0))  # Just to have a coroutine
                    except:
                        status_tasks.append(asyncio.sleep(0))
            
            # Actually run the status checks properly
            statuses = {}
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'get_status'):
                        status = await component.get_status()
                    elif hasattr(component, 'health_check'):
                        status = await component.health_check()
                    else:
                        status = {"available": True, "initialized": True}
                    
                    statuses[name] = status
                except Exception as e:
                    statuses[name] = {"available": False, "error": str(e)}
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Overall system status
            available_count = sum(1 for s in statuses.values() if s.get("available", False))
            total_count = len(statuses)
            
            system_status = {
                "overall_status": "healthy" if available_count == total_count else 
                                 "degraded" if available_count > 0 else "unhealthy",
                "available_components": available_count,
                "total_components": total_count,
                "components": statuses,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "Knowledge engine system status retrieved",
                "available_components": available_count,
                "total_components": total_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return system_status
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to get knowledge engine system status",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "overall_status": "error",
                "available_components": 0,
                "total_components": len(self.components),
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_component_status(self, name: str, status_task) -> Dict[str, Any]:
        """Get status of a single component."""
        try:
            if asyncio.iscoroutine(status_task):
                status = await status_task
            else:
                status = status_task
            return {name: status}
        except Exception as e:
            return {name: {"available": False, "error": str(e)}}
    
    async def close(self):
        """Close all integrated components and clean up resources."""
        logger.info({
            "msg": "Closing Knowledge Engine Orchestrator resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close each component
        close_tasks = []
        for name, component in self.components.items():
            if hasattr(component, 'close'):
                try:
                    close_task = component.close()
                    if asyncio.iscoroutine(close_task):
                        close_tasks.append(close_task)
                except Exception as e:
                    logger.error(f"Error closing {name}: {e}")
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        logger.info({
            "msg": "Knowledge Engine Orchestrator resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })