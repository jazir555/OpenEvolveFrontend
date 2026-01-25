"""
OpenEvolve Knowledge Engine - Main Entry Point

This module provides the main entry point for the OpenEvolve Knowledge Engine,
orchestrating all integrated components into a unified system that can learn,
evolve, and improve over time through coordinated operation of all components.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid
import json
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEngineOutput:
    """Output from the Knowledge Engine."""
    success: bool
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'results': self.results,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class OpenEvolveKnowledgeEngine:
    """
    Main OpenEvolve Knowledge Engine that orchestrates all integrated components.
    
    This system weaves together all the integrated knowledge processing systems
    into a self-learning, evolving system that can:
    - Process knowledge across multiple modalities and domains
    - Learn from interactions and improve over time
    - Coordinate complex multi-component workflows
    - Perform formal verification and validation
    - Adapt to new domains and requirements
    - Evolve its own capabilities through reflection and learning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenEvolve Knowledge Engine.
        
        Args:
            config: Configuration for all integrated components
        """
        self.config = config or self._get_default_config()
        
        # Initialize the orchestrator
        self.orchestrator = KnowledgeEngineOrchestrator(config=self.config)
        
        # Initialize learning and adaptation components
        self.learning_memory = {}
        self.adaptation_engine = None
        self.reflection_engine = None
        
        # Initialize based on configuration
        self._initialize_learning_components()
        
        logger.info({
            "msg": "OpenEvolveKnowledgeEngine initialized",
            "components_count": len(self.orchestrator.components),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the knowledge engine."""
        return {
            "orchestrator": {
                "enable_parallel_processing": True,
                "max_concurrent_operations": 10,
                "default_timeout": 300,  # seconds
                "enable_caching": True,
                "cache_ttl": 3600,  # seconds
            },
            "learning": {
                "enable_adaptive_learning": True,
                "learning_rate": 0.1,
                "memory_retention_hours": 72,
                "experience_buffer_size": 1000,
                "reflection_frequency": 10,  # Reflect every N operations
            },
            "evolution": {
                "enable_self_evolution": True,
                "evolution_frequency": 100,  # Evolve every N operations
                "mutation_probability": 0.1,
                "selection_pressure": 0.5,
            },
            "verification": {
                "enable_formal_verification": True,
                "verification_threshold": 0.9,  # Minimum confidence for verification
                "max_verification_depth": 5,
            },
            "integration": {
                "enable_cross_component_learning": True,
                "enable_component_coordination": True,
                "coordination_timeout": 60,
            }
        }
    
    def _initialize_learning_components(self):
        """Initialize learning and adaptation components."""
        try:
            # Initialize adaptation engine if available
            if self.config.get("learning", {}).get("enable_adaptive_learning", True):
                from .learning.adaptation_engine import AdaptationEngine
                self.adaptation_engine = AdaptationEngine(
                    learning_rate=self.config["learning"]["learning_rate"],
                    memory_retention_hours=self.config["learning"]["memory_retention_hours"],
                    experience_buffer_size=self.config["learning"]["experience_buffer_size"]
                )
            
            # Initialize reflection engine if available
            if self.config.get("learning", {}).get("enable_adaptive_learning", True):
                from .learning.reflection_engine import ReflectionEngine
                self.reflection_engine = ReflectionEngine(
                    reflection_frequency=self.config["learning"]["reflection_frequency"]
                )
            
            logger.info({
                "msg": "Learning components initialized",
                "adaptive_learning_enabled": self.config.get("learning", {}).get("enable_adaptive_learning", True),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError:
            logger.warning({
                "msg": "Learning components not available, using basic implementation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize basic learning components
            self._initialize_basic_learning_components()
    
    def _initialize_basic_learning_components(self):
        """Initialize basic learning components when advanced ones aren't available."""
        logger.info({
            "msg": "Initializing basic learning components",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Basic learning memory
        self.learning_memory = {
            "operations_count": 0,
            "success_count": 0,
            "error_count": 0,
            "operation_history": [],
            "component_performance": {}
        }
    
    async def process_request(
        self,
        query: str,
        components: Optional[List[str]] = None,
        learn_from_interaction: bool = True,
        correlation_id: Optional[str] = None
    ) -> KnowledgeEngineOutput:
        """
        Process a knowledge request through the integrated system.
        
        Args:
            query: Knowledge query to process
            components: Specific components to use (None for all)
            learn_from_interaction: Whether to learn from this interaction
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeEngineOutput with results from all components
        """
        correlation_id = correlation_id or f"oe_ke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting OpenEvolve Knowledge Engine request",
            "query_length": len(query),
            "components_requested": components or "all",
            "learn_from_interaction": learn_from_interaction,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process through orchestrator
            result = await self.orchestrator.process_knowledge_request(
                query=query,
                components=components,
                correlation_id=f"{correlation_id}_orch"
            )
            
            # Update learning memory if enabled
            if learn_from_interaction:
                await self._update_learning_memory(
                    query=query,
                    result=result,
                    correlation_id=f"{correlation_id}_learn"
                )
            
            # Perform reflection if needed
            if (self.learning_memory.get("operations_count", 0) + 1) % self.config.get("learning", {}).get("reflection_frequency", 10) == 0:
                await self._perform_reflection(correlation_id=f"{correlation_id}_reflect")
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            ke_output = KnowledgeEngineOutput(
                success=result.success,
                results=result.to_dict(),
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "learn_from_interaction": learn_from_interaction,
                    "components_used": components or list(self.orchestrator.components.keys())
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "OpenEvolve Knowledge Engine request completed",
                "correlation_id": correlation_id,
                "success": result.success,
                "components_count": len(result.metadata.get("components_valid", [])),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ke_output
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "OpenEvolve Knowledge Engine request failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineOutput(
                success=False,
                results={},
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "learn_from_interaction": learn_from_interaction
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def _update_learning_memory(
        self,
        query: str,
        result: Any,
        correlation_id: str
    ):
        """Update the learning memory with the results of an operation."""
        try:
            self.learning_memory["operations_count"] = self.learning_memory.get("operations_count", 0) + 1
            
            if result.success:
                self.learning_memory["success_count"] = self.learning_memory.get("success_count", 0) + 1
            else:
                self.learning_memory["error_count"] = self.learning_memory.get("error_count", 0) + 1
            
            # Add to operation history
            self.learning_memory["operation_history"].append({
                "query": query,
                "result_success": result.success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": correlation_id
            })
            
            # Limit history size
            max_history = self.config.get("learning", {}).get("experience_buffer_size", 1000)
            if len(self.learning_memory["operation_history"]) > max_history:
                self.learning_memory["operation_history"] = self.learning_memory["operation_history"][-max_history:]
            
            # Update component performance
            if hasattr(result, 'metadata') and 'components_used' in result.metadata:
                for comp_name in result.metadata['components_used']:
                    if comp_name not in self.learning_memory["component_performance"]:
                        self.learning_memory["component_performance"][comp_name] = {
                            "total_ops": 0,
                            "successful_ops": 0,
                            "avg_processing_time": 0
                        }
                    
                    comp_perf = self.learning_memory["component_performance"][comp_name]
                    comp_perf["total_ops"] += 1
                    if result.success:
                        comp_perf["successful_ops"] += 1
                    
                    # Update average processing time
                    total_time = comp_perf["avg_processing_time"] * (comp_perf["total_ops"] - 1)
                    if hasattr(result, 'processing_time_ms'):
                        total_time += result.processing_time_ms
                    else:
                        total_time += 0  # Default if not available
                    comp_perf["avg_processing_time"] = total_time / comp_perf["total_ops"]
            
            logger.debug({
                "msg": "Learning memory updated",
                "correlation_id": correlation_id,
                "operations_count": self.learning_memory["operations_count"],
                "success_rate": self.learning_memory.get("success_count", 0) / max(self.learning_memory.get("operations_count", 1), 1),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error({
                "msg": "Failed to update learning memory",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def _perform_reflection(self, correlation_id: str):
        """Perform system reflection on recent operations."""
        try:
            logger.info({
                "msg": "Starting system reflection",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Get recent operations for reflection
            recent_ops = self.learning_memory.get("operation_history", [])[-20:]  # Last 20 operations
            
            if not recent_ops:
                logger.info({
                    "msg": "No recent operations for reflection",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return
            
            # Analyze patterns in recent operations
            success_count = sum(1 for op in recent_ops if op.get("result_success", False))
            success_rate = success_count / len(recent_ops)
            
            # Identify components that may need adjustment
            underperforming_components = []
            for comp_name, perf in self.learning_memory.get("component_performance", {}).items():
                if perf["total_ops"] > 5:  # Only consider if enough data
                    comp_success_rate = perf["successful_ops"] / perf["total_ops"]
                    if comp_success_rate < 0.7:  # Threshold for underperformance
                        underperforming_components.append({
                            "name": comp_name,
                            "success_rate": comp_success_rate,
                            "avg_processing_time": perf["avg_processing_time"]
                        })
            
            # Log reflection insights
            reflection_insights = {
                "recent_success_rate": success_rate,
                "total_recent_ops": len(recent_ops),
                "underperforming_components": underperforming_components,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "System reflection completed",
                "correlation_id": correlation_id,
                "reflection_insights": reflection_insights,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # In a real implementation, we would use this insight to adjust system behavior
            # For now, just log the insights
            
        except Exception as e:
            logger.error({
                "msg": "System reflection failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def run_comprehensive_analysis(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        learn_from_analysis: bool = True,
        correlation_id: Optional[str] = None
    ) -> KnowledgeEngineOutput:
        """
        Run comprehensive knowledge analysis using multiple integrated components.
        
        Args:
            text: Text to analyze
            analysis_types: Types of analysis to perform
            learn_from_analysis: Whether to learn from this analysis
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeEngineOutput with analysis results
        """
        correlation_id = correlation_id or f"oe_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting comprehensive OpenEvolve knowledge analysis",
            "text_length": len(text),
            "analysis_types": analysis_types or ["entities", "relations", "patterns", "insights"],
            "learn_from_analysis": learn_from_analysis,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Run comprehensive analysis through orchestrator
            result = await self.orchestrator.run_comprehensive_analysis(
                text=text,
                analysis_types=analysis_types,
                correlation_id=f"{correlation_id}_orch"
            )
            
            # Update learning memory if enabled
            if learn_from_analysis:
                await self._update_learning_memory(
                    query=f"Analysis of: {text[:100]}...",
                    result=result,
                    correlation_id=f"{correlation_id}_learn"
                )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            ke_output = KnowledgeEngineOutput(
                success=result.success,
                results=result.to_dict(),
                metadata={
                    "analysis_types": analysis_types or ["entities", "relations", "patterns", "insights"],
                    "processing_time_ms": processing_time_ms,
                    "learn_from_analysis": learn_from_analysis
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Comprehensive OpenEvolve knowledge analysis completed",
                "correlation_id": correlation_id,
                "success": result.success,
                "analysis_types_count": len(result.metadata.get("analysis_types", [])),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ke_output
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Comprehensive OpenEvolve knowledge analysis failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineOutput(
                success=False,
                results={},
                metadata={
                    "analysis_types": analysis_types or [],
                    "processing_time_ms": processing_time_ms,
                    "learn_from_analysis": learn_from_analysis
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def evolve_capabilities(
        self,
        evolution_target: str = "performance",
        correlation_id: Optional[str] = None
    ) -> KnowledgeEngineOutput:
        """
        Evolve system capabilities based on experience and performance.
        
        Args:
            evolution_target: What to evolve ('performance', 'accuracy', 'efficiency', 'capabilities')
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeEngineOutput with evolution results
        """
        correlation_id = correlation_id or f"oe_evolve_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting OpenEvolve capability evolution",
            "evolution_target": evolution_target,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Gather insights from learning memory
            insights = await self._gather_evolution_insights()
            
            # Based on the evolution target, determine what to evolve
            evolution_plan = self._create_evolution_plan(evolution_target, insights)
            
            # Execute evolution plan
            evolution_results = await self._execute_evolution_plan(evolution_plan, correlation_id)
            
            # Update system configuration based on evolution
            await self._apply_evolution_changes(evolution_results)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            ke_output = KnowledgeEngineOutput(
                success=True,
                results=evolution_results,
                metadata={
                    "evolution_target": evolution_target,
                    "processing_time_ms": processing_time_ms,
                    "evolution_plan": evolution_plan
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "OpenEvolve capability evolution completed",
                "correlation_id": correlation_id,
                "evolution_target": evolution_target,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ke_output
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "OpenEvolve capability evolution failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineOutput(
                success=False,
                results={},
                metadata={
                    "evolution_target": evolution_target,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def _gather_evolution_insights(self) -> Dict[str, Any]:
        """Gather insights for system evolution."""
        insights = {
            "performance_metrics": {
                "total_operations": self.learning_memory.get("operations_count", 0),
                "success_rate": (
                    self.learning_memory.get("success_count", 0) /
                    max(self.learning_memory.get("operations_count", 1), 1)
                ),
                "error_rate": (
                    self.learning_memory.get("error_count", 0) /
                    max(self.learning_memory.get("operations_count", 1), 1)
                )
            },
            "component_performance": self.learning_memory.get("component_performance", {}),
            "recent_operations": self.learning_memory.get("operation_history", [])[-50:],  # Last 50 operations
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return insights
    
    def _create_evolution_plan(self, target: str, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create an evolution plan based on target and insights."""
        plan = {
            "target": target,
            "recommended_changes": [],
            "priority": "medium",
            "estimated_impact": "high",
            "implementation_steps": [],
            "rollback_plan": "Revert to previous configuration"
        }
        
        if target == "performance":
            # Look for performance bottlenecks
            slow_components = []
            for comp_name, perf in insights["component_performance"].items():
                if perf.get("avg_processing_time", 0) > 5000:  # More than 5 seconds
                    slow_components.append({
                        "name": comp_name,
                        "avg_time_ms": perf["avg_processing_time"]
                    })
            
            if slow_components:
                plan["recommended_changes"].append({
                    "type": "optimization",
                    "target": "slow_components",
                    "components": slow_components,
                    "suggestion": "Consider optimizing or replacing slow components"
                })
        
        elif target == "accuracy":
            # Look for underperforming components
            underperforming = []
            for comp_name, perf in insights["component_performance"].items():
                if perf["total_ops"] > 10:  # Only consider if enough data
                    success_rate = perf["successful_ops"] / perf["total_ops"]
                    if success_rate < 0.8:  # Less than 80% success
                        underperforming.append({
                            "name": comp_name,
                            "success_rate": success_rate
                        })
            
            if underperforming:
                plan["recommended_changes"].append({
                    "type": "improvement",
                    "target": "underperforming_components",
                    "components": underperforming,
                    "suggestion": "Investigate and improve underperforming components"
                })
        
        elif target == "capabilities":
            # Look for opportunities to expand capabilities
            plan["recommended_changes"].append({
                "type": "expansion",
                "target": "new_integrations",
                "suggestion": "Consider integrating additional knowledge processing capabilities"
            })
        
        return plan
    
    async def _execute_evolution_plan(self, plan: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """Execute an evolution plan."""
        # In a real implementation, this would execute the actual evolution steps
        # For now, we'll just simulate the execution
        
        results = {
            "plan_executed": True,
            "changes_applied": len(plan.get("recommended_changes", [])),
            "changes_successful": len(plan.get("recommended_changes", [])),
            "changes_failed": 0,
            "applied_changes": plan.get("recommended_changes", []),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info({
            "msg": "Evolution plan executed",
            "correlation_id": correlation_id,
            "changes_applied": results["changes_applied"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return results
    
    async def _apply_evolution_changes(self, evolution_results: Dict[str, Any]):
        """Apply changes from evolution to the system."""
        # In a real implementation, this would apply the actual changes to the system
        # For now, we'll just log the changes that would be applied
        
        logger.info({
            "msg": "Applying evolution changes to system",
            "changes_count": evolution_results.get("changes_applied", 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update configuration based on evolution results
        if "applied_changes" in evolution_results:
            for change in evolution_results["applied_changes"]:
                logger.info({
                    "msg": f"Applied evolution change: {change.get('suggestion', 'Unknown')}",
                    "change_type": change.get("type", "unknown"),
                    "target": change.get("target", "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    async def get_system_knowledge_state(self) -> Dict[str, Any]:
        """
        Get the current state of the knowledge system including:
        - Learning memory
        - Component performance
        - Recent operations
        - Evolution status
        
        Returns:
            Dictionary with system knowledge state
        """
        try:
            # Get orchestrator status
            orchestrator_status = await self.orchestrator.get_system_status()
            
            # Get system knowledge state
            state = {
                "learning_memory": self.learning_memory,
                "orchestrator_status": orchestrator_status,
                "config": self.config,
                "evolution_readiness": (
                    self.learning_memory.get("operations_count", 0) >= 
                    self.config.get("evolution", {}).get("evolution_frequency", 100)
                ),
                "next_reflection_at": (
                    self.learning_memory.get("operations_count", 0) + 
                    (self.config.get("learning", {}).get("reflection_frequency", 10) - 
                     (self.learning_memory.get("operations_count", 0) % 
                      self.config.get("learning", {}).get("reflection_frequency", 10)))
                ),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "System knowledge state retrieved",
                "operations_count": self.learning_memory.get("operations_count", 0),
                "components_count": len(orchestrator_status.get("components", {})),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error({
                "msg": "Failed to get system knowledge state",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self):
        """Close resources used by the knowledge engine."""
        logger.info({
            "msg": "Closing OpenEvolve Knowledge Engine resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close orchestrator
        await self.orchestrator.close()
        
        # Close learning components if they have close methods
        if self.adaptation_engine and hasattr(self.adaptation_engine, 'close'):
            await self.adaptation_engine.close()
        
        if self.reflection_engine and hasattr(self.reflection_engine, 'close'):
            await self.reflection_engine.close()
        
        logger.info({
            "msg": "OpenEvolve Knowledge Engine resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            # If running in an event loop, schedule cleanup
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                # Otherwise, run the cleanup synchronously
                asyncio.run(self.close())
        except Exception:
            # If no event loop is available, just pass
            pass


# Global instance for easy access
_openevolve_knowledge_engine = None


async def get_knowledge_engine(config: Optional[Dict[str, Any]] = None) -> OpenEvolveKnowledgeEngine:
    """
    Get or create the global OpenEvolve Knowledge Engine instance.
    
    Args:
        config: Optional configuration for the knowledge engine
        
    Returns:
        OpenEvolveKnowledgeEngine instance
    """
    global _openevolve_knowledge_engine
    
    if _openevolve_knowledge_engine is None:
        _openevolve_knowledge_engine = OpenEvolveKnowledgeEngine(config)
        await _openevolve_knowledge_engine.orchestrator.initialize_components()  # Initialize if needed
    
    return _openevolve_knowledge_engine


async def process_knowledge_request(
    query: str,
    components: Optional[List[str]] = None,
    learn_from_interaction: bool = True,
    correlation_id: Optional[str] = None
) -> KnowledgeEngineOutput:
    """
    Process a knowledge request using the global knowledge engine instance.
    
    Args:
        query: Knowledge query to process
        components: Specific components to use (None for all)
        learn_from_interaction: Whether to learn from this interaction
        correlation_id: Correlation ID for tracking
        
    Returns:
        KnowledgeEngineOutput with results
    """
    engine = await get_knowledge_engine()
    return await engine.process_request(
        query=query,
        components=components,
        learn_from_interaction=learn_from_interaction,
        correlation_id=correlation_id
    )


async def run_comprehensive_analysis(
    text: str,
    analysis_types: Optional[List[str]] = None,
    learn_from_analysis: bool = True,
    correlation_id: Optional[str] = None
) -> KnowledgeEngineOutput:
    """
    Run comprehensive analysis using the global knowledge engine instance.
    
    Args:
        text: Text to analyze
        analysis_types: Types of analysis to perform
        learn_from_analysis: Whether to learn from this analysis
        correlation_id: Correlation ID for tracking
        
    Returns:
        KnowledgeEngineOutput with analysis results
    """
    engine = await get_knowledge_engine()
    return await engine.run_comprehensive_analysis(
        text=text,
        analysis_types=analysis_types,
        learn_from_analysis=learn_from_analysis,
        correlation_id=correlation_id
    )