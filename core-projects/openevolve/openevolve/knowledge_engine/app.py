"""
OpenEvolve Knowledge Engine - Main Application

This is the main application module that initializes and coordinates all components
of the knowledge engine system into a unified, production-ready application.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml
import argparse
from contextlib import asynccontextmanager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import all the integration components
from knowledge_engine.config.config_manager import ConfigManager
from knowledge_engine.data.storage import KnowledgeStorageEngine
from knowledge_engine.integrations.main_orchestrator import KnowledgeEngineOrchestrator
from knowledge_engine.server import app, lifespan


class OpenEvolveKnowledgeEngine:
    """
    Main OpenEvolve Knowledge Engine application class.
    
    This class coordinates all integrated components into a unified system
    that can learn, evolve, and improve over time through coordinated operation
    of all knowledge processing components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OpenEvolve Knowledge Engine.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config_manager = None
        self.storage_engine = None
        self.orchestrator = None
        self.server = None
        
        # Initialize configuration
        self._initialize_config()
        
        logger.info({
            "msg": "OpenEvolve Knowledge Engine initialized",
            "config_path": config_path,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _initialize_config(self):
        """Initialize configuration manager."""
        try:
            self.config_manager = ConfigManager(config_path=self.config_path)
            logger.info({
                "msg": "Configuration manager initialized",
                "config_path": self.config_path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize configuration: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    async def initialize_components(self):
        """Initialize all knowledge engine components."""
        logger.info({
            "msg": "Starting OpenEvolve Knowledge Engine component initialization",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        try:
            # Initialize storage engine
            storage_config = self.config_manager.get_component_config("storage") or {}
            self.storage_engine = KnowledgeStorageEngine(storage_config)
            await self.storage_engine.initialize()
            
            # Initialize orchestrator with all integrated components
            orchestrator_config = self.config_manager.get_component_config("orchestrator") or {}
            self.orchestrator = KnowledgeEngineOrchestrator(orchestrator_config)
            await self.orchestrator.initialize_components()
            
            logger.info({
                "msg": "All OpenEvolve Knowledge Engine components initialized successfully",
                "storage_initialized": self.storage_engine is not None,
                "orchestrator_initialized": self.orchestrator is not None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize OpenEvolve Knowledge Engine components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    async def process_request(
        self,
        query: str,
        components: Optional[list[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Process a knowledge request through the integrated system.
        
        Args:
            query: Knowledge query to process
            components: Specific components to use (None for all)
            context: Context information for the request
            correlation_id: Correlation ID for tracking
            
        Returns:
            Processed result from the knowledge engine
        """
        if not self.orchestrator:
            raise RuntimeError("Knowledge engine orchestrator not initialized")
        
        return await self.orchestrator.process_knowledge_request(
            query=query,
            components=components,
            context=context or {},
            correlation_id=correlation_id
        )
    
    async def run_comprehensive_analysis(
        self,
        text: str,
        analysis_types: Optional[list[str]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Run comprehensive knowledge analysis using multiple integrated components.
        
        Args:
            text: Text to analyze
            analysis_types: Types of analysis to perform
            correlation_id: Correlation ID for tracking
            
        Returns:
            Analysis results from multiple components
        """
        if not self.orchestrator:
            raise RuntimeError("Knowledge engine orchestrator not initialized")
        
        return await self.orchestrator.run_comprehensive_analysis(
            text=text,
            analysis_types=analysis_types,
            correlation_id=correlation_id
        )
    
    async def store_knowledge_artifact(
        self,
        content: str,
        artifact_type: str,
        source: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a knowledge artifact in the integrated storage system.
        
        Args:
            content: Content of the knowledge artifact
            artifact_type: Type of artifact ('entity', 'relation', 'triple', etc.)
            source: Source of the knowledge
            embedding: Optional embedding vector
            metadata: Optional metadata
            
        Returns:
            Artifact ID of the stored artifact
        """
        if not self.storage_engine:
            raise RuntimeError("Knowledge storage engine not initialized")
        
        return await self.storage_engine.store_knowledge_artifact(
            content=content,
            artifact_type=artifact_type,
            source=source,
            embedding=embedding,
            metadata=metadata
        )
    
    async def search_knowledge(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        top_k: int = 10,
        correlation_id: Optional[str] = None
    ):
        """
        Search the knowledge base using integrated search capabilities.
        
        Args:
            query: Search query
            artifact_type: Optional type filter
            top_k: Number of results to return
            correlation_id: Correlation ID for tracking
            
        Returns:
            Search results from the knowledge base
        """
        if not self.storage_engine:
            raise RuntimeError("Knowledge storage engine not initialized")
        
        return await self.storage_engine.search_knowledge_artifacts(
            query=query,
            artifact_type=artifact_type,
            top_k=top_k
        )
    
    async def evolve_capabilities(
        self,
        evolution_target: str = "performance",
        correlation_id: Optional[str] = None
    ):
        """
        Evolve system capabilities based on experience and performance.
        
        Args:
            evolution_target: What to evolve ('performance', 'accuracy', 'efficiency', 'capabilities')
            correlation_id: Correlation ID for tracking
            
        Returns:
            Evolution results and changes applied
        """
        if not self.orchestrator:
            raise RuntimeError("Knowledge engine orchestrator not initialized")
        
        return await self.orchestrator.evolve_capabilities(
            evolution_target=evolution_target,
            correlation_id=correlation_id
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of all integrated components.
        
        Returns:
            Dictionary with status information for all components
        """
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "config_manager": {
                    "initialized": self.config_manager is not None,
                    "config_path": getattr(self.config_manager, 'config_path', 'unknown')
                },
                "storage_engine": {
                    "initialized": self.storage_engine is not None,
                    "status": await self.storage_engine.get_status() if self.storage_engine else "not_initialized"
                },
                "orchestrator": {
                    "initialized": self.orchestrator is not None,
                    "status": await self.orchestrator.get_status() if self.orchestrator else "not_initialized"
                }
            },
            "integrations": {}
        }
        
        # Add integration status if orchestrator is available
        if self.orchestrator:
            status["integrations"] = await self.orchestrator.get_integration_status()
        
        return status
    
    async def close(self):
        """Close all resources used by the knowledge engine."""
        logger.info({
            "msg": "Closing OpenEvolve Knowledge Engine resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close orchestrator
        if self.orchestrator:
            await self.orchestrator.close()
        
        # Close storage engine
        if self.storage_engine:
            await self.storage_engine.close()
        
        # Close config manager
        if self.config_manager:
            await self.config_manager.close()
        
        logger.info({
            "msg": "OpenEvolve Knowledge Engine resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


# Global application instance
_openevolve_app: Optional[OpenEvolveKnowledgeEngine] = None


async def get_openevolve_app(config_path: Optional[str] = None) -> OpenEvolveKnowledgeEngine:
    """
    Get or create the global OpenEvolve Knowledge Engine instance.
    
    Args:
        config_path: Optional configuration file path
        
    Returns:
        OpenEvolveKnowledgeEngine instance
    """
    global _openevolve_app
    
    if _openevolve_app is None:
        _openevolve_app = OpenEvolveKnowledgeEngine(config_path)
        await _openevolve_app.initialize_components()
    
    return _openevolve_app


async def run_openevolve_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: Optional[str] = None
):
    """
    Run the OpenEvolve Knowledge Engine server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to configuration file
    """
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine server",
        "host": host,
        "port": port,
        "config_path": config_path,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Initialize the application
    app = await get_openevolve_app(config_path)
    
    # Run the server using uvicorn
    import uvicorn
    
    # Configure uvicorn to use our lifespan
    config = uvicorn.Config(
        "knowledge_engine.server:app",  # Use the app from server module
        host=host,
        port=port,
        log_level="info",
        lifespan="on"
    )
    
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await app.close()


async def run_batch_processing(
    input_file: str,
    output_file: str,
    config_path: Optional[str] = None
):
    """
    Run batch processing on a file of queries.
    
    Args:
        input_file: Path to input file with queries (JSONL format)
        output_file: Path to output file for results
        config_path: Path to configuration file
    """
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine batch processing",
        "input_file": input_file,
        "output_file": output_file,
        "config_path": config_path,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Initialize the application
    app = await get_openevolve_app(config_path)
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line.strip()) for line in f if line.strip()]
        
        logger.info({
            "msg": "Processing batch queries",
            "query_count": len(queries),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Process each query
        results = []
        for i, query_data in enumerate(queries):
            try:
                query_text = query_data.get('query', '') if isinstance(query_data, dict) else query_data
                components = query_data.get('components') if isinstance(query_data, dict) else None
                context = query_data.get('context') if isinstance(query_data, dict) else {}
                
                result = await app.process_request(
                    query=query_text,
                    components=components,
                    context=context,
                    correlation_id=f"batch_{i}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                )
                
                results.append({
                    "input": query_data,
                    "result": result.to_dict() if hasattr(result, 'to_dict') else result,
                    "index": i,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                logger.info({
                    "msg": f"Processed batch query {i+1}/{len(queries)}",
                    "success": result.success if hasattr(result, 'success') else True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error({
                    "msg": f"Failed to process batch query {i}",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                results.append({
                    "input": query_data,
                    "error": str(e),
                    "index": i,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Write results to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info({
            "msg": "Batch processing completed",
            "input_file": input_file,
            "output_file": output_file,
            "processed_count": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    finally:
        await app.close()


def create_default_config(config_path: str = "config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to create the configuration file
    """
    default_config = {
        "name": "OpenEvolve Knowledge Engine",
        "version": "1.0.0",
        "environment": "production",
        "database": {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "username": "openevolve",
            "password": "",
            "database": "openevolve_kg",
            "connection_pool_size": 20,
            "ssl_enabled": False
        },
        "vector_store": {
            "type": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection_name": "knowledge_artifacts",
            "distance_metric": "cosine",
            "vector_size": 1536,
            "recreate_collection": False
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "ttl_seconds": 3600,
            "max_items": 10000
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "timeout": 300,
            "max_connections": 1000,
            "cors_enabled": True,
            "cors_origins": ["*"],
            "ssl_enabled": False
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "",
            "base_url": None,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "request_timeout": 120,
            "max_retries": 3,
            "retry_delay": 1.0
        },
        "integrations": {
            "enable_graphiti": True,
            "enable_kggen": True,
            "enable_oneke": True,
            "enable_aikg": True,
            "enable_ragbits": True,
            "enable_crewai": True,
            "enable_deepke": True,
            "enable_researchquest": True,
            "enable_agentic_context": True,
            "enable_agentjson": True,
            "enable_dspy": True,
            "enable_leanaide": True,
            "enable_openevolve_lib": True,
            "enable_mcp_gateway": True
        },
        "features": {
            "enable_temporal_reasoning": True,
            "enable_bilingual_extraction": True,
            "enable_multi_agent_collaboration": True,
            "enable_formal_verification": True,
            "enable_retrieval_augmentation": True,
            "enable_program_of_thought": True,
            "enable_self_evolution": True
        },
        "performance": {
            "max_concurrent_requests": 100,
            "request_queue_size": 1000,
            "response_cache_ttl": 300,
            "enable_request_logging": True,
            "enable_response_caching": True
        },
        "security": {
            "enable_authentication": True,
            "enable_authorization": True,
            "jwt_secret": "",
            "rate_limit_requests": 100,
            "rate_limit_window": 60
        },
        "monitoring": {
            "log_level": "INFO",
            "log_format": "json",
            "enable_metrics": True,
            "metrics_export_port": 9090,
            "enable_tracing": False,
            "tracing_endpoint": None
        }
    }
    
    # Create directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration file
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration created at {config_path}")


async def main():
    """Main entry point for the OpenEvolve Knowledge Engine."""
    parser = argparse.ArgumentParser(description="OpenEvolve Knowledge Engine")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["server", "batch", "interactive"], 
                       default="interactive", help="Execution mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for server mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for server mode")
    parser.add_argument("--input-file", type=str, help="Input file for batch mode")
    parser.add_argument("--output-file", type=str, help="Output file for batch mode")
    
    args = parser.parse_args()
    
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine",
        "mode": args.mode,
        "config": args.config,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        if args.mode == "server":
            await run_openevolve_server(
                host=args.host,
                port=args.port,
                config_path=args.config
            )
        elif args.mode == "batch":
            if not args.input_file or not args.output_file:
                raise ValueError("Input and output files required for batch mode")
            await run_batch_processing(
                input_file=args.input_file,
                output_file=args.output_file,
                config_path=args.config
            )
        elif args.mode == "interactive":
            # Initialize the application
            app = await get_openevolve_app(args.config)
            
            print("OpenEvolve Knowledge Engine - Interactive Mode")
            print("Type 'help' for commands, 'quit' to exit")
            
            while True:
                try:
                    query = input("\nEnter query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if query.lower() == 'help':
                        print("\nCommands:")
                        print("  quit/exit/q - Exit the application")
                        print("  status - Get system status")
                        print("  evolve - Evolve system capabilities")
                        print("  <query> - Process a knowledge query")
                        continue
                    if query.lower() == 'status':
                        status = await app.get_system_status()
                        print(f"System Status: {json.dumps(status, indent=2)}")
                        continue
                    if query.lower() == 'evolve':
                        result = await app.evolve_capabilities()
                        print(f"Evolution Result: {json.dumps(result, indent=2)}")
                        continue
                    
                    if query:
                        result = await app.process_request(query=query)
                        print(f"Result: {json.dumps(result, indent=2)}")
                        
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error processing query: {e}")
            
            await app.close()
        
    except Exception as e:
        logger.error(f"Error running OpenEvolve Knowledge Engine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())