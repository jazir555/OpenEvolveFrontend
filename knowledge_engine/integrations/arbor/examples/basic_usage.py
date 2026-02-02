"""
Basic Usage Example for Arbor Integration

This example demonstrates:
1. Connecting to Arbor server
2. Indexing a codebase
3. Querying the code graph
4. Integrating with Knowledge Engine
5. Finding code paths and impact analysis

Prerequisites:
- Arbor server running (cargo run --release in arbor/crates)
- Knowledge Engine initialized
- websockets package installed
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Arbor integration
from knowledge_engine.integrations.arbor import (
    ArborClient,
    ArborConfig,
    ArborGraphAdapter,
    ArborSchemaMapper
)
from knowledge_engine.core import EntityKnowledgeGraph


async def example_1_basic_connection():
    """Example 1: Basic connection to Arbor server."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Connection")
    print("=" * 60)
    
    # Create configuration
    config = ArborConfig(
        connection__ws_url="ws://localhost:7433",
        connection__connection_timeout=10.0
    )
    
    # Create client
    client = ArborClient(config)
    
    try:
        # Connect to Arbor
        connected = await client.connect()
        print(f"Connected to Arbor: {connected}")
        
        # Get server stats
        stats = await client.get_stats()
        print(f"Server stats: {stats}")
        
    except Exception as e:
        print(f"Connection failed: {e}")
        print("  Make sure Arbor server is running on ws://localhost:7433")
    
    finally:
        await client.disconnect()
        print("Disconnected")


async def example_7_schema_mapping():
    """Example 7: Direct schema mapping."""
    print("\n" + "=" * 60)
    print("Example 7: Schema Mapping")
    print("=" * 60)
    
    # Example Arbor node
    arbor_node = {
        "id": "func_123",
        "name": "authenticate",
        "kind": "function",
        "qualifiedName": "auth.authenticate",
        "file": "src/auth.py",
        "lineStart": 45,
        "lineEnd": 78,
        "signature": "def authenticate(username: str, password: str) -> bool",
        "visibility": "public",
        "attributes": {"async": False, "static": False},
        "docstring": "Authenticate a user.",
        "centrality": 0.85
    }
    
    print("\nConverting Arbor node to Knowledge Engine Entity...")
    
    # Use mapper directly
    mapper = ArborSchemaMapper(storage_prefix="arbor")
    entity = mapper.convert_arbor_node(arbor_node)
    
    print(f"Conversion complete")
    print(f"  Entity ID: {entity.entity_id}")
    print(f"  Name: {entity.name}")
    print(f"  Type: {entity.entity_type}")
    print(f"  Properties:")
    for key, value in entity.properties.items():
        print(f"    {key}: {value}")
    
    # Example Arbor edge
    arbor_edge = {
        "from": "func_123",
        "to": "func_456",
        "kind": "calls",
        "location": {"file": "src/auth.py", "line": 52, "column": 12}
    }
    
    print("\nConverting Arbor edge to Relationship...")
    relationship = mapper.convert_arbor_edge(arbor_edge)
    
    print(f"Conversion complete")
    print(f"  Source: {relationship.source_id}")
    print(f"  Target: {relationship.target_id}")
    print(f"  Type: {relationship.relationship_type}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Arbor Integration Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the Arbor-Knowledge Engine integration.")
    print("Make sure Arbor server is running: cargo run --release (in arbor/crates)")
    
    # Run examples
    await example_1_basic_connection()
    await example_7_schema_mapping()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
