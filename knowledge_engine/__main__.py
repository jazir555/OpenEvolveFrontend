"""
Final Knowledge Engine Implementation for OpenEvolve

This module ties together all the components of the OpenEvolve Knowledge Engine
into a cohesive system with all phases implemented.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import uuid


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEngineResult:
    """Result of a knowledge engine operation."""
    success: bool
    data: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class FinalKnowledgeEngine:
    """
    Final Knowledge Engine implementation combining all phases.
    
    Provides a unified interface to:
    - Phase 1: Basic knowledge extraction and storage
    - Phase 2: Enhanced features with ML and personalization  
    - Phase 3: Production-ready multi-database integration
    - Integration with Graphiti, KG-Gen, OneKE, and AIKG
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the final knowledge engine.
        
        Args:
            config: Configuration for the knowledge engine
        """
        self.config = config or self._get_default_config()
        
        # Initialize all components
        self._initialize_components()
        
        logger.info({
            "msg": "FinalKnowledgeEngine initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "phase1_enabled": True,
            "phase2_enabled": True,
            "phase3_enabled": True,
            "graphiti_enabled": True,
            "kggen_enabled": True,
            "oneke_enabled": True,
            "aikg_enabled": True,
            "default_timeout_ms": 30000,
            "enable_caching": True,
            "cache_ttl": 300,
            "max_retries": 3
        }
    
    def _initialize_components(self):
        """Initialize all knowledge engine components."""
        # Import components
        from knowledge_engine.knowledge_extractor import KnowledgeExtractor
        from knowledge_engine.knowledge_storage import KnowledgeStorage
        from knowledge_engine.knowledge_retriever import KnowledgeRetriever
        from knowledge_engine.enhanced_storage import EnhancedKnowledgeStorage
        from knowledge_engine.enhanced_retriever import EnhancedKnowledgeRetriever
        from knowledge_engine.real_database_integration import RealDatabaseIntegrator
        from knowledge_engine.embedding_generator import EmbeddingGenerator
        
        # Initialize Phase 1 components
        if self.config.get("phase1_enabled", True):
            self.extractor = KnowledgeExtractor(self.config)
            self.storage = KnowledgeStorage(self.config)
            self.retriever = KnowledgeRetriever(self.storage, self.config)
        
        # Initialize Phase 2 components
        if self.config.get("phase2_enabled", True):
            self.enhanced_storage = EnhancedKnowledgeStorage(self.config)
            self.enhanced_retriever = EnhancedKnowledgeRetriever(self.enhanced_storage, self.config)
        
        # Initialize Phase 3 components
        if self.config.get("phase3_enabled", True):
            self.database_integrator = RealDatabaseIntegrator(self.config)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(self.config)
        
        # Initialize integration components
        self.integration_components = {}
        
        # Initialize Graphiti if enabled
        if self.config.get("graphiti_enabled", True):
            try:
                from knowledge_engine.integrations.graphiti import GraphitiTemporalBridge
                self.integration_components["graphiti"] = GraphitiTemporalBridge(
                    uri=self.config.get("graphiti_uri", "bolt://localhost:7687"),
                    user=self.config.get("graphiti_user", "neo4j"),
                    password=self.config.get("graphiti_password", "password")
                )
            except ImportError:
                logger.warning("Graphiti integration not available")
        
        # Initialize KG-Gen if enabled
        if self.config.get("kggen_enabled", True):
            try:
                from knowledge_engine.integrations.kggen import KGGenPipelineIntegration
                self.integration_components["kggen"] = KGGenPipelineIntegration()
            except ImportError:
                logger.warning("KG-Gen integration not available")
        
        # Initialize OneKE if enabled
        if self.config.get("oneke_enabled", True):
            try:
                from knowledge_engine.integrations.oneke import EnhancedOneKEBridge
                self.integration_components["oneke"] = EnhancedOneKEBridge()
            except ImportError:
                logger.warning("OneKE integration not available")
        
        # Initialize AIKG if enabled
        if self.config.get("aikg_enabled", True):
            try:
                from knowledge_engine.integrations.aikg_integration import AIKGIntegration
                self.integration_components["aikg"] = AIKGIntegration()
            except ImportError:
                logger.warning("AIKG integration not available")
    
    async def process_workflow_execution(
        self,
        workflow_data: Dict[str, Any],
        generate_embeddings: bool = True,
        run_phase2_enhancements: bool = True,
        run_phase3_integration: bool = True
    ) -> KnowledgeEngineResult:
        """
        Process workflow execution data through all phases.
        
        Args:
            workflow_data: Workflow execution data
            generate_embeddings: Whether to generate embeddings
            run_phase2_enhancements: Whether to run Phase 2 enhancements
            run_phase3_integration: Whether to run Phase 3 integration
            
        Returns:
            KnowledgeEngineResult with processing results
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Processing workflow execution through all phases",
            "workflow_id": workflow_data.get('workflow_id'),
            "generate_embeddings": generate_embeddings,
            "run_phase2_enhancements": run_phase2_enhancements,
            "run_phase3_integration": run_phase3_integration,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Phase 1: Extract knowledge artifacts
            artifacts = self.extractor.extract_from_workflow(workflow_data)
            
            # Store artifacts with embeddings if requested
            stored_artifact_ids = []
            for artifact in artifacts:
                artifact_dict = {
                    'type': artifact.artifact_type,
                    'source': artifact.source,
                    'content': artifact.content,
                    'context': artifact.context,
                    'metadata': artifact.metadata
                }
                
                if generate_embeddings:
                    embedding = self.embedding_generator.generate_knowledge_artifact_embedding(artifact_dict)
                    artifact_dict['embedding'] = embedding
                
                # Store using basic storage
                artifact_id = self.storage.store_knowledge_artifact(artifact_dict, generate_embedding=False)
                stored_artifact_ids.append(artifact_id)
            
            phase1_result = {
                "artifacts_extracted": len(artifacts),
                "artifacts_stored": len(stored_artifact_ids),
                "stored_artifact_ids": stored_artifact_ids
            }
            
            # Phase 2: Enhanced processing
            phase2_result = {}
            if run_phase2_enhancements and self.config.get("phase2_enabled", True):
                # Store in enhanced storage
                enhanced_artifact_ids = []
                for artifact in artifacts:
                    artifact_dict = {
                        'type': artifact.artifact_type,
                        'source': artifact.source,
                        'content': artifact.content,
                        'context': artifact.context,
                        'metadata': artifact.metadata,
                        'artifact_id': str(uuid.uuid4())
                    }
                    
                    result = self.enhanced_storage.store_knowledge_artifact(
                        artifact=artifact_dict,
                        generate_embedding=generate_embeddings
                    )
                    
                    if result.success:
                        enhanced_artifact_ids.append(result.artifact_id)
                
                phase2_result = {
                    "enhanced_artifacts_stored": len(enhanced_artifact_ids),
                    "enhanced_artifact_ids": enhanced_artifact_ids
                }
            
            # Phase 3: Real database integration
            phase3_result = {}
            if run_phase3_integration and self.config.get("phase3_enabled", True):
                # Integrate with real databases
                db_integration_results = []
                
                # Example: Execute a query to verify integration
                if self.database_integrator.is_production_ready():
                    query_result = self.database_integrator.execute_query(
                        query="COUNT(*) FROM knowledge_artifacts",
                        database_type=None  # Use default
                    )
                    db_integration_results.append(query_result)
                
                phase3_result = {
                    "database_integration_results": [r.__dict__ for r in db_integration_results],
                    "production_ready": self.database_integrator.is_production_ready()
                }
            
            # Integration with external systems
            integration_results = {}
            
            # Graphiti integration
            if "graphiti" in self.integration_components:
                try:
                    graphiti_bridge = self.integration_components["graphiti"]
                    await graphiti_bridge.initialize()
                    
                    # Add artifacts to temporal knowledge graph
                    for artifact in artifacts:
                        await graphiti_bridge.add_entity(
                            name=artifact.content[:50],  # Use first 50 chars as name
                            entity_type=artifact.artifact_type,
                            metadata=artifact.metadata,
                            correlation_id=workflow_data.get('workflow_id', 'unknown')
                        )
                    
                    integration_results["graphiti"] = {
                        "entities_added": len(artifacts),
                        "success": True
                    }
                except Exception as e:
                    integration_results["graphiti"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # KG-Gen integration
            if "kggen" in self.integration_components:
                try:
                    kggen_pipeline = self.integration_components["kggen"]
                    
                    # Extract knowledge graph from workflow content
                    content = " ".join([a.content for a in artifacts])
                    if content:
                        graph = await kggen_pipeline.extract_knowledge_graph(
                            text=content,
                            context=workflow_data.get('domain', 'general')
                        )
                        
                        integration_results["kggen"] = {
                            "entities_extracted": len(graph.entities),
                            "relations_extracted": len(graph.relations),
                            "success": True
                        }
                    else:
                        integration_results["kggen"] = {
                            "success": False,
                            "error": "No content to extract from"
                        }
                except Exception as e:
                    integration_results["kggen"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Compile final result
            final_result = {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "integrations": integration_results,
                "workflow_processed": True,
                "processing_complete": True
            }
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Workflow execution processing completed through all phases",
                "workflow_id": workflow_data.get('workflow_id'),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=True,
                data=final_result,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Workflow execution processing failed",
                "workflow_id": workflow_data.get('workflow_id'),
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=False,
                data={},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def query_knowledge(
        self,
        query: str,
        query_type: str = "hybrid",
        use_enhanced: bool = True,
        include_integrations: bool = True
    ) -> KnowledgeEngineResult:
        """
        Query knowledge through all available systems.
        
        Args:
            query: Query string
            query_type: Type of query ('keyword', 'semantic', 'hybrid', etc.)
            use_enhanced: Whether to use enhanced retrieval
            include_integrations: Whether to include integration results
            
        Returns:
            KnowledgeEngineResult with query results
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Querying knowledge through all systems",
            "query": query,
            "query_type": query_type,
            "use_enhanced": use_enhanced,
            "include_integrations": include_integrations,
            "timestamp": start_time.isoformat()
        })
        
        try:
            results = {}
            
            # Basic retrieval
            if use_enhanced and self.config.get("phase2_enabled", True):
                retrieval_result = self.enhanced_retriever.search_knowledge(
                    query=query,
                    query_type=query_type,
                    limit=10
                )
                results["enhanced_retrieval"] = retrieval_result
            else:
                retrieval_result = self.retriever.search_knowledge(
                    query=query,
                    query_type=query_type,
                    limit=10
                )
                results["basic_retrieval"] = retrieval_result
            
            # Integration queries
            if include_integrations:
                integration_query_results = {}
                
                # Graphiti temporal query
                if "graphiti" in self.integration_components:
                    try:
                        graphiti_bridge = self.integration_components["graphiti"]
                        temporal_results = await graphiti_bridge.query_at_point_in_time(
                            query=query,
                            timestamp=datetime.now(timezone.utc),
                            max_results=10
                        )
                        integration_query_results["graphiti_temporal"] = [r.to_dict() for r in temporal_results]
                    except Exception as e:
                        integration_query_results["graphiti_temporal"] = {"error": str(e)}
                
                # KG-Gen query
                if "kggen" in self.integration_components:
                    try:
                        # This would involve more complex logic in a real implementation
                        integration_query_results["kggen"] = {"message": "KG-Gen query interface would be implemented here"}
                    except Exception as e:
                        integration_query_results["kggen"] = {"error": str(e)}
                
                results["integration_queries"] = integration_query_results
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Knowledge query completed",
                "query": query,
                "result_count": len(results),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=True,
                data={
                    "query_results": results,
                    "query": query,
                    "query_type": query_type,
                    "use_enhanced": use_enhanced
                },
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge query failed",
                "query": query,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeEngineResult(
                success=False,
                data={},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status across all components.
        
        Returns:
            Dictionary with system status
        """
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phases_enabled": {
                "phase1": self.config.get("phase1_enabled", True),
                "phase2": self.config.get("phase2_enabled", True),
                "phase3": self.config.get("phase3_enabled", True)
            },
            "components": {},
            "production_ready": False
        }
        
        # Check Phase 1 components
        status["components"]["phase1"] = {
            "extractor": hasattr(self, 'extractor'),
            "storage": hasattr(self, 'storage'),
            "retriever": hasattr(self, 'retriever')
        }
        
        # Check Phase 2 components
        if self.config.get("phase2_enabled", True):
            status["components"]["phase2"] = {
                "enhanced_storage": hasattr(self, 'enhanced_storage'),
                "enhanced_retriever": hasattr(self, 'enhanced_retriever')
            }
        
        # Check Phase 3 components
        if self.config.get("phase3_enabled", True):
            status["components"]["phase3"] = {
                "database_integrator": hasattr(self, 'database_integrator'),
                "production_ready": self.database_integrator.is_production_ready() if hasattr(self, 'database_integrator') else False
            }
            status["production_ready"] = status["components"]["phase3"]["production_ready"]
        
        # Check integration components
        status["components"]["integrations"] = {
            "graphiti": "graphiti" in self.integration_components,
            "kggen": "kggen" in self.integration_components,
            "oneke": "oneke" in self.integration_components,
            "aikg": "aikg" in self.integration_components
        }
        
        # Get detailed status from individual components
        if hasattr(self, 'database_integrator'):
            status["database_health"] = self.database_integrator.get_health_status()
        
        if hasattr(self, 'enhanced_retriever'):
            status["retrieval_insights"] = self.enhanced_retriever.get_retrieval_insights()
        
        return status
    
    async def close(self):
        """Close all connections and clean up resources."""
        logger.info({
            "msg": "Closing FinalKnowledgeEngine and cleaning up resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close database connections
        if hasattr(self, 'database_integrator'):
            self.database_integrator.close_connections()
        
        # Close enhanced storage connections
        if hasattr(self, 'enhanced_storage'):
            self.enhanced_storage.close_connections()
        
        # Close basic storage connections
        if hasattr(self, 'storage'):
            self.storage.close_connections()
        
        # Close Graphiti connections
        if "graphiti" in self.integration_components:
            try:
                await self.integration_components["graphiti"].close()
            except Exception as e:
                logger.error(f"Error closing Graphiti: {e}")
        
        logger.info({
            "msg": "FinalKnowledgeEngine closed successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


# Convenience function for easy initialization
async def create_final_knowledge_engine(config: Optional[Dict[str, Any]] = None) -> FinalKnowledgeEngine:
    """
    Create and initialize a FinalKnowledgeEngine instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized FinalKnowledgeEngine ready to use
    """
    engine = FinalKnowledgeEngine(config)
    return engine


# Example usage
async def main():
    """Example usage of the Final Knowledge Engine."""
    print("🚀 Initializing Final Knowledge Engine...")
    
    # Create engine
    engine = await create_final_knowledge_engine()
    
    # Check system status
    status = engine.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)[:500]}...")
    
    # Example workflow data
    workflow_data = {
        "workflow_id": "example_workflow_001",
        "domain": "software_engineering",
        "complexity": "high",
        "team_size": 5,
        "success": True,
        "execution_time": 3600,
        "solution_patterns": [
            {
                "pattern": "modular_decomposition",
                "effectiveness": 0.95,
                "context": "complex_software_design"
            }
        ],
        "critique_patterns": [
            {
                "pattern": "tight_coupling",
                "issue": "suboptimal_modularity",
                "severity": "high"
            }
        ],
        "team_performance": {
            "efficiency": 0.87,
            "collaboration": 0.92,
            "adaptability": 0.85
        },
        "gauntlet_effectiveness": {
            "completion_rate": 0.90,
            "quality_score": 0.88,
            "iteration_count": 3
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Process workflow
    print("\n1. Processing workflow execution...")
    result = await engine.process_workflow_execution(workflow_data)
    print(f"[OK] Processing result: {result.success}")
    print(f"[OK] Data keys: {list(result.data.keys())}")
    
    # Query knowledge
    print("\n2. Querying knowledge...")
    query_result = await engine.query_knowledge("modular decomposition")
    print(f"[OK] Query result: {query_result.success}")
    print(f"[OK] Query data keys: {list(query_result.data.keys())}")
    
    # Close engine
    await engine.close()
    print("\n[OK] Final Knowledge Engine closed successfully!")


if __name__ == "__main__":
    asyncio.run(main())