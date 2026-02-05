"""
Arbor MCP Bridge Demo: AI Agent Code Intelligence

This demo shows how AI agents can use Arbor's MCP tools to:
1. Find code definitions
2. Navigate call graphs
3. Analyze impact of changes
4. Get contextual code information

Usage:
    python -m knowledge_engine.integrations.arbor.examples.mcp_demo

Requirements:
    - Arbor server running (cargo run in arbor/)
    - Python dependencies: asyncio, websockets
"""

import asyncio
import logging
import sys

# Configure logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def demo_find_definition(mcp):
    """Demo: Finding code definitions."""
    print("\n" + "="*60)
    print("DEMO 1: Finding Code Definitions")
    print("="*60)
    
    examples = [
        {"symbol": "ArborClient", "file": None},
        {"symbol": "main", "file": None, "kind": "function"},
    ]
    
    for params in examples:
        print(f"\nFinding definition: {params}")
        result = await mcp.execute_tool("arbor_find_definition", params)
        
        if result.success:
            data = result.data
            print(f"  [OK] Found {data['kind']} '{data['name']}'")
            print(f"    File: {data['file']}")
            print(f"    Lines: {data['location']['line_start']}-{data['location']['line_end']}")
            if data.get('signature'):
                print(f"    Signature: {data['signature']}")
        else:
            print(f"  [FAIL] {result.message}")


async def demo_call_graph(mcp):
    """Demo: Navigating call graphs."""
    print("\n" + "="*60)
    print("DEMO 2: Navigating Call Graphs")
    print("="*60)
    
    # Get callers (who calls this function?)
    print("\nFinding callers of 'initialize'...")
    result = await mcp.execute_tool(
        "arbor_get_callers",
        {"function_name": "initialize"}
    )
    
    if result.success:
        callers = result.data["callers"]
        print(f"  [OK] Found {result.data['total_count']} callers")
        for caller in callers[:3]:  # Show first 3
            print(f"    - {caller['name']} ({caller['kind']}) at {caller.get('file', '?')}:{caller.get('line', '?')}")
    else:
        print(f"  [FAIL] {result.message}")
    
    # Get callees (what does this function call?)
    print("\nFinding functions called by 'main'...")
    result = await mcp.execute_tool(
        "arbor_get_callees",
        {"function_name": "main"}
    )
    
    if result.success:
        callees = result.data["callees"]
        print(f"  [OK] Found {result.data['total_count']} callees")
        for callee in callees[:3]:  # Show first 3
            print(f"    - {callee['name']} ({callee['kind']})")
    else:
        print(f"  [FAIL] {result.message}")


async def demo_find_path(mcp):
    """Demo: Finding logic flow between components."""
    print("\n" + "="*60)
    print("DEMO 3: Finding Logic Flow")
    print("="*60)
    
    print("\nFinding path from 'main' to 'initialize'...")
    result = await mcp.execute_tool(
        "arbor_find_path",
        {"start": "main", "end": "initialize"}
    )
    
    if result.success:
        path = result.data["path"]
        print(f"  [OK] Found path ({len(path)} steps):")
        path_str = " -> ".join(p["name"] for p in path)
        print(f"    {path_str}")
    else:
        print(f"  [FAIL] {result.message}")


async def demo_impact_analysis(mcp):
    """Demo: Analyzing impact of changes."""
    print("\n" + "="*60)
    print("DEMO 4: Impact Analysis")
    print("="*60)
    
    scenarios = [
        {"symbol": "ArborConfig", "change_type": "modify"},
        {"symbol": "connect", "change_type": "rename"},
    ]
    
    for params in scenarios:
        print(f"\nAnalyzing impact of {params['change_type']} on '{params['symbol']}'...")
        result = await mcp.execute_tool("arbor_analyze_impact", params)
        
        if result.success:
            data = result.data
            print(f"  [OK] Analysis complete:")
            print(f"    Direct impacts: {len(data['direct_impacts'])}")
            print(f"    Transitive impacts: {data['transitive_impacts_count']}")
            print(f"    Files affected: {len(data['files_to_modify'])}")
            if data['direct_impacts']:
                print("    Affected components:")
                for impact in data['direct_impacts'][:3]:
                    print(f"      - {impact['name']} ({impact['kind']})")
        else:
            print(f"  [FAIL] {result.message}")


async def demo_get_context(mcp):
    """Demo: Getting contextual code information."""
    print("\n" + "="*60)
    print("DEMO 5: Getting Code Context")
    print("="*60)
    
    print("\nGetting context for 'ArborClient' (depth=2)...")
    result = await mcp.execute_tool(
        "arbor_get_context",
        {"symbol": "ArborClient", "depth": 2}
    )
    
    if result.success:
        data = result.data
        print(f"  [OK] Context for '{data['symbol']}' ({data['kind']}):")
        if data.get('signature'):
            print(f"    Signature: {data['signature']}")
        print(f"    Related components: {data['total_related']}")
        if data['related_components']:
            print("    Related:")
            for rel in data['related_components'][:5]:
                print(f"      - {rel['name']} ({rel['kind']})")
    else:
        print(f"  [FAIL] {result.message}")


async def demo_search(mcp):
    """Demo: Searching code."""
    print("\n" + "="*60)
    print("DEMO 6: Code Search")
    print("="*60)
    
    queries = [
        {"query": "connect", "max_results": 5},
        {"query": "Config", "kind": "class", "max_results": 5},
    ]
    
    for params in queries:
        kind_filter = f" [{params['kind']}]" if params.get('kind') else ""
        print(f"\nSearching for '{params['query']}'{kind_filter}...")
        result = await mcp.execute_tool("arbor_search", params)
        
        if result.success:
            matches = result.data["matches"]
            print(f"  [OK] Found {result.data['total_count']} matches")
            for match in matches[:3]:
                print(f"    - {match['name']} ({match['kind']})")
                if match.get('file'):
                    print(f"      at {match['file']}:{match.get('line', '?')}")
        else:
            print(f"  [FAIL] {result.message}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Arbor MCP Bridge Demo")
    print("="*60)
    print("\nThis demo shows how AI agents can use Arbor's code intelligence")
    print("tools to navigate and analyze codebases.")
    
    # Check for Arbor server
    print("\nChecking Arbor server availability...")
    
    try:
        from knowledge_engine.integrations.arbor import ArborClient, ArborConfig, ArborMCPBridge
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        print("\nMake sure you're running from the project root.")
        return 1
    
    config = ArborConfig.from_env()
    client = ArborClient(config)
    
    # Try to connect with retries
    connected = False
    for attempt in range(3):
        try:
            connected = await client.connect()
            if connected:
                break
        except Exception as e:
            print(f"  Attempt {attempt + 1}/3: {e}")
            await asyncio.sleep(1)
    
    if not connected:
        print("\n[FAIL] Could not connect to Arbor server.")
        print("\nPlease start Arbor server first:")
        print("  cd arbor/ && cargo run --release")
        print("\nOr update ARBOR_WS_URL to point to your Arbor server.")
        return 1
    
    print("[OK] Connected to Arbor server")
    
    try:
        # Create MCP bridge
        mcp = ArborMCPBridge(client, config.mcp)
        print(f"[OK] MCP Bridge initialized with {len(mcp._tools)} tools")
        print(f"  Available tools: {', '.join(mcp._tools.keys())}")
        
        # Run demos
        await demo_find_definition(mcp)
        await demo_call_graph(mcp)
        await demo_find_path(mcp)
        await demo_impact_analysis(mcp)
        await demo_get_context(mcp)
        await demo_search(mcp)
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nThe MCP tools above can be integrated into AI agents to provide:")
        print("  * Precise code navigation")
        print("  * Automated refactoring analysis")
        print("  * Context-aware code understanding")
        print("  * Impact assessment for changes")
        
    finally:
        await client.disconnect()
        print("\n[OK] Disconnected from Arbor server")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
