#!/usr/bin/env python3
"""
OpenEvolve Knowledge Engine - Production Entry Point

This is the main entry point for the production OpenEvolve Knowledge Engine system.
It initializes all integrated components and provides a unified interface for
knowledge processing, reasoning, and automation.
"""

import asyncio
import logging
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import yaml
from typing import Dict, Any, Optional


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_knowledge_engine(config_path: Optional[str] = None):
    """
    Initialize the complete OpenEvolve Knowledge Engine with all integrations.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized knowledge engine instance
    """
    logger.info({
        "msg": "Initializing OpenEvolve Knowledge Engine with all integrations",
        "config_path": config_path,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        # Import the main application
        from knowledge_engine.app import OpenEvolveKnowledgeEngine
        
        # Initialize the knowledge engine
        engine = OpenEvolveKnowledgeEngine(config_path=config_path)
        await engine.initialize_components()
        
        logger.info({
            "msg": "OpenEvolve Knowledge Engine initialized successfully with all integrations",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return engine
        
    except Exception as e:
        logger.error({
            "msg": f"Failed to initialize OpenEvolve Knowledge Engine: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise


async def run_server(host: str = "0.0.0.0", port: int = 8000, config_path: Optional[str] = None):
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
    
    try:
        # Import and run the server
        from knowledge_engine.server import run_server as run_knowledge_server
        await run_knowledge_server(
            host=host,
            port=port,
            config_path=config_path
        )
    except ImportError:
        # Fallback to uvicorn if direct import fails
        import uvicorn
        from knowledge_engine.server import app
        
        logger.info("Running server with uvicorn")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )


async def run_batch_processing(input_file: str, output_file: str, config_path: Optional[str] = None):
    """
    Run batch processing on a set of inputs.
    
    Args:
        input_file: Path to input file with queries/data
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
    
    try:
        from knowledge_engine.app import run_batch_processing as engine_batch_process
        await engine_batch_process(
            input_file=input_file,
            output_file=output_file,
            config_path=config_path
        )
    except Exception as e:
        logger.error({
            "msg": f"Batch processing failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise


async def run_interactive_mode(config_path: Optional[str] = None):
    """
    Run the knowledge engine in interactive mode.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine in interactive mode",
        "config_path": config_path,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        from knowledge_engine.app import OpenEvolveKnowledgeEngine
        
        # Initialize engine
        engine = await initialize_knowledge_engine(config_path)
        
        print("OpenEvolve Knowledge Engine - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("  process <query> - Process a knowledge query")
        print("  status - Get system status")
        print("  evolve - Evolve system capabilities")
        print("  help - Show this help")
        print("  quit/exit - Exit the system")
        print("")
        
        while True:
            try:
                command = input("OpenEvolve> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Shutting down OpenEvolve Knowledge Engine...")
                    break
                elif command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  process <query> - Process a knowledge query")
                    print("  status - Get system status")
                    print("  evolve - Evolve system capabilities")
                    print("  help - Show this help")
                    print("  quit/exit - Exit the system\n")
                elif command.lower().startswith('process '):
                    query = command[8:].strip()  # Remove 'process ' prefix
                    if query:
                        result = await engine.process_request(query=query)
                        print(f"Result: {json.dumps(result.to_dict(), indent=2)}")
                    else:
                        print("Please provide a query to process")
                elif command.lower() == 'status':
                    status = await engine.get_system_status()
                    print(f"Status: {json.dumps(status, indent=2)}")
                elif command.lower() == 'evolve':
                    result = await engine.evolve_capabilities()
                    print(f"Evolution result: {json.dumps(result.to_dict(), indent=2)}")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nShutting down OpenEvolve Knowledge Engine...")
                break
            except Exception as e:
                print(f"Error processing command: {e}")
        
        # Close the engine
        await engine.close()
        
    except Exception as e:
        logger.error({
            "msg": f"Interactive mode failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise


def create_default_config(config_path: str = "config.yaml"):
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_path: Path where to create the configuration file
    """
    config_path = Path(config_path)
    
    if config_path.exists():
        logger.info(f"Configuration file already exists: {config_path}")
        return
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
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
            "recreate_collection": false
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
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration created at {config_path}")


async def main():
    """Main entry point for the OpenEvolve Knowledge Engine."""
    parser = argparse.ArgumentParser(description="OpenEvolve Knowledge Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["server", "batch", "interactive"], 
                       default="interactive", help="Execution mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for server mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for server mode")
    parser.add_argument("--input-file", type=str, help="Input file for batch mode")
    parser.add_argument("--output-file", type=str, help="Output file for batch mode")
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    if not Path(args.config).exists():
        create_default_config(args.config)
    
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine",
        "mode": args.mode,
        "config": args.config,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        if args.mode == "server":
            await run_server(host=args.host, port=args.port, config_path=args.config)
        elif args.mode == "batch":
            if not args.input_file or not args.output_file:
                raise ValueError("Input and output files required for batch mode")
            await run_batch_processing(
                input_file=args.input_file,
                output_file=args.output_file,
                config_path=args.config
            )
        elif args.mode == "interactive":
            await run_interactive_mode(config_path=args.config)
    except Exception as e:
        logger.error({
            "msg": f"OpenEvolve Knowledge Engine failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())