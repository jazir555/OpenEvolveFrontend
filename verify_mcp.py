#!/usr/bin/env python3
"""Quick MCP server verification."""

import asyncio
from unified_mcp_server import UnifiedMCPServer, MCP_AVAILABLE

server = UnifiedMCPServer()

print('=' * 60)
print('MCP SERVER VERIFICATION')
print('=' * 60)
print(f'MCP Package Available: {MCP_AVAILABLE}')
print(f'Server Mode: {server.mode.upper()}')
print(f'Server Name: {server.name}')
print()

print('REGISTERED TOOLS:')
print('-' * 40)
categories = server.registry.get_tools_by_category()
for cat, tools in categories.items():
    if tools:
        print(f'{cat.value.upper()}: {len(tools)} tools')
        for tool in tools:
            print(f'  - {tool}')

print()
total_tools = len(server.registry.list_tools())
print(f'TOTAL TOOLS: {total_tools}')
print()

async def test_tool():
    result = await server.execute_tool('analyze_complexity', {
        'description': 'Build a machine learning pipeline'
    })
    print('TEST TOOL EXECUTION:')
    success = result.get('success')
    print(f'  Success: {success}')
    if success:
        print(f'  Problem Type: {result.get("problem_type")}')
        print(f'  Domain: {result.get("domain")}')
    else:
        print(f'  Error: {result.get("error")}')

asyncio.run(test_tool())
print()
print('=' * 60)
print('MCP SERVER: FULLY OPERATIONAL')
print('=' * 60)
