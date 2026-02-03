#!/bin/bash
# Probe script for KG-Gen MCP server
# LAW OF RUNTIME TRUTH: Verify MCP server works before using it

set -e

echo "=== KG-Gen MCP Server Probe ==="

# Check if mcp_server module exists
echo "Checking mcp_server module..."
python3 -c "from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer; print('✓ KGGenMCPServer import successful')"

# Check configuration validation
echo "Testing configuration validation..."
python3 -c "
from knowledge_engine.integrations.kggen.mcp_server import MemoryStoreConfig
config = MemoryStoreConfig()
config.validate()
print('✓ Configuration validation successful')
"

# Test memory addition
echo "Testing memory addition..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer, MemoryType

async def test():
    server = KGGenMCPServer()

    memory = await server.memory_manager.add_memory(
        content='Test memory',
        memory_type=MemoryType.FACT,
        session_id='test-session',
        correlation_id='test-correlation'
    )

    assert memory.memory_id, 'Memory ID not generated'
    assert memory.content == 'Test memory', 'Memory content incorrect'
    print(f'✓ Memory addition successful: {memory.memory_id}')
    await server.close()

asyncio.run(test())
"

# Test memory retrieval
echo "Testing memory retrieval..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer, MemoryType, MemoryQuery

async def test():
    server = KGGenMCPServer()

    # Add test memory
    await server.memory_manager.add_memory(
        content='Apple is a tech company',
        memory_type=MemoryType.FACT,
        session_id='test-session'
    )

    # Retrieve
    query = MemoryQuery(query_text='Apple', session_id='test-session', max_results=10)
    memories = await server.memory_manager.retrieve_relevant_memories(query)

    assert len(memories) > 0, 'No memories retrieved'
    print(f'✓ Memory retrieval successful: {len(memories)} memories retrieved')
    await server.close()

asyncio.run(test())
"

# Test add_memories MCP tool
echo "Testing add_memories MCP tool..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer, MemoryType

async def test():
    server = KGGenMCPServer()

    memories_data = [
        {'content': 'Fact 1', 'memory_type': 'fact'},
        {'content': 'Fact 2', 'memory_type': 'fact'}
    ]

    result = await server.add_memories(
        memories=memories_data,
        session_id='test-session'
    )

    assert result['success'] == True, 'add_memories failed'
    assert result['count'] == 2, f'Expected 2 memories, got {result["count"]}'

    print(f'✓ add_memories tool successful: {result["count"]} memories added')
    await server.close()

asyncio.run(test())
"

# Test retrieve_relevant_memories MCP tool
echo "Testing retrieve_relevant_memories MCP tool..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer, MemoryType

async def test():
    server = KGGenMCPServer()

    # Add test memory
    await server.memory_manager.add_memory(
        content='Python is a programming language',
        memory_type=MemoryType.FACT,
        session_id='test-session'
    )

    # Retrieve using MCP tool
    result = await server.retrieve_relevant_memories(
        query_text='programming',
        session_id='test-session',
        max_results=10
    )

    assert result['success'] == True, 'retrieve_relevant_memories failed'
    print(f'✓ retrieve_relevant_memories tool successful: {result["count"]} memories retrieved')
    await server.close()

asyncio.run(test())
"

# Test visualize_memories MCP tool
echo "Testing visualize_memories MCP tool..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.mcp_server import KGGenMCPServer, MemoryType

async def test():
    server = KGGenMCPServer()

    # Add test memories
    for i in range(3):
        await server.memory_manager.add_memory(
            content=f'Test memory {i}',
            memory_type=MemoryType.FACT,
            session_id='test-session'
        )

    # Visualize
    result = await server.visualize_memories(session_id='test-session')

    assert result['success'] == True, 'visualize_memories failed'
    assert result['statistics']['total_memories'] == 3, 'Incorrect memory count'

    print(f'✓ visualize_memories tool successful: {result["statistics"]["total_memories"]} memories visualized')
    await server.close()

asyncio.run(test())
"

echo "=== All MCP Server Probes Passed ==="
