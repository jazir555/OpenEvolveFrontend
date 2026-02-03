#!/usr/bin/env python3
"""Test the comprehensive MCP server."""

import asyncio
from unified_mcp_server_comprehensive import UnifiedMCPServer, MCP_AVAILABLE, ToolCategory

server = UnifiedMCPServer()

categories = server.registry.get_tools_by_category()
total = len(server.registry.list_tools())

print('=' * 70)
print('COMPREHENSIVE MCP SERVER - VERIFICATION')
print('=' * 70)
print(f'Mode: {server.mode.upper()}')
print(f'MCP Available: {MCP_AVAILABLE}')
print()

for cat, tools in sorted(categories.items(), key=lambda x: x[0].value):
    if tools:
        print(f'{cat.value.upper():25s}: {len(tools):2d} tools')
        for t in tools[:3]:  # Show first 3
            print(f'  - {t}')
        if len(tools) > 3:
            print(f'  ... and {len(tools)-3} more')
        print()

print('-' * 70)
print(f'TOTAL: {total} tools')
print('=' * 70)

# Test a few tools
async def test_tools():
    print('\nTesting tools:')
    
    # Test 1: LeanAide
    result = await server.execute_tool('leanaide_translate_theorem', {
        'theorem_statement': 'The sum of two even numbers is even'
    })
    print(f'  leanaide_translate_theorem: {"OK" if result.get("success") else "FAIL"}')
    
    # Test 2: Decomposition
    result = await server.execute_tool('analyze_problem_for_decomposition', {
        'problem_text': 'Build a web application'
    })
    print(f'  analyze_problem_for_decomposition: {"OK" if result.get("success") else "FAIL"}')
    
    # Test 3: Steer
    result = await server.execute_tool('verify_json_output', {
        'output': '{"key": "value"}'
    })
    print(f'  verify_json_output: {"OK" if result.get("success") else "FAIL"}')
    
    # Test 4: ROMA
    result = await server.execute_tool('get_roma_status', {})
    print(f'  get_roma_status: {"OK" if result.get("success") else "FAIL"}')
    
    print('\nAll tests completed!')

asyncio.run(test_tools())
