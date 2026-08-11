"""
MCP Import Fix - License: Apache 2.0

Fixes MCP import by working around local directory shadowing.
"""

import sys
from pathlib import Path

def fix_mcp_import():
    """Fix MCP import by temporarily removing local mcp from path."""
    # Remove local mcp directory from path
    local_mcp = str(Path(__file__).parent / 'mcp')
    
    # Save original path
    original_path = sys.path.copy()
    
    # Remove local mcp if present
    if local_mcp in sys.path:
        sys.path.remove(local_mcp)
    
    # Also remove empty string and current dir if needed
    paths_to_remove = [p for p in sys.path if 'mcp' in p.lower() and Path(p).name == 'mcp']
    for p in paths_to_remove:
        sys.path.remove(p)
    
    try:
        # Now import the real mcp package
        import importlib
        
        # Force reload if already imported
        if 'mcp' in sys.modules:
            del sys.modules['mcp']
        if 'mcp.server' in sys.modules:
            del sys.modules['mcp.server']
        
        # Import from site-packages
        import mcp as real_mcp
        
        return real_mcp
    except ImportError as e:
        print(f"Could not import real MCP: {e}")
        return None
    finally:
        # Restore original path
        sys.path = original_path


# Alternative: Create a unified MCP server that doesn't depend on external mcp package
class UnifiedMCPServerNative:
    """
    Native MCP server implementation that doesn't require external mcp package.
    
    Implements MCP protocol directly using JSON-RPC over stdio.
    """
    
    def __init__(self, name: str = "openevolve-unified"):
        self.name = name
        self.tools = {}
        self.version = "1.0.0"
    
    def register_tool(self, name: str, handler, schema: dict):
        """Register a tool."""
        self.tools[name] = {
            'handler': handler,
            'schema': schema
        }
    
    def get_tools(self) -> list:
        """Get list of registered tools."""
        return [
            {
                'name': name,
                'description': info['schema'].get('description', ''),
                'inputSchema': info['schema']
            }
            for name, info in self.tools.items()
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a registered tool."""
        if name not in self.tools:
            return {
                'content': [{'type': 'text', 'text': f'Tool {name} not found'}],
                'isError': True
            }
        
        try:
            handler = self.tools[name]['handler']
            result = await handler(arguments)
            return {
                'content': [{'type': 'text', 'text': str(result)}],
                'isError': False
            }
        except Exception as e:
            return {
                'content': [{'type': 'text', 'text': f'Error: {str(e)}'}],
                'isError': True
            }
    
    async def run(self):
        """Run the MCP server."""
        import json
        
        print(f"Starting {self.name} MCP server...", file=sys.stderr)
        
        while True:
            try:
                line = input()
                if not line:
                    continue
                
                message = json.loads(line)
                method = message.get('method')
                request_id = message.get('id')
                
                if method == 'initialize':
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'protocolVersion': '2024-11-05',
                            'serverInfo': {
                                'name': self.name,
                                'version': self.version
                            },
                            'capabilities': {
                                'tools': {}
                            }
                        }
                    }
                    print(json.dumps(response), flush=True)
                
                elif method == 'tools/list':
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'tools': self.get_tools()
                        }
                    }
                    print(json.dumps(response), flush=True)
                
                elif method == 'tools/call':
                    params = message.get('params', {})
                    name = params.get('name')
                    arguments = params.get('arguments', {})
                    
                    result = await self.call_tool(name, arguments)
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': result
                    }
                    print(json.dumps(response), flush=True)
                
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)


# Export
__all__ = ['fix_mcp_import', 'UnifiedMCPServerNative']
