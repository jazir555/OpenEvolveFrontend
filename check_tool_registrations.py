"""Check MCP tool registrations in all MCP tool files"""

import os
import re

mcp_files = [
    ('roma_mcp_tools.py', 'ROMA'),
    ('roma_mdap_maker_mcp_tools.py', 'ROMA-MDAP-MAKER'),
    ('decomposition_mcp_tools.py', 'Decomposition'),
    ('bubblelabs_mcp_tools.py', 'BubbleLabs'),
    ('openevolve_mcp_tools.py', 'OpenEvolve'),
    ('leanaide_mcp_tools.py', 'LeanAide'),
    ('claudiomiro_mcp_tools.py', 'Claudiomiro'),
    ('datapizza_mcp_tools.py', 'DataPizza'),
    ('steer_mcp_tools.py', 'Steer'),
    ('ace_mcp_tools.py', 'ACE'),
    ('guardrails_mcp_tools.py', 'Guardrails'),
    ('c2c_mcp_tools.py', 'C2C'),
    ('lmql_mcp_tools.py', 'LMQL')
]

print('=' * 100)
print('MCP TOOL REGISTRATION VERIFICATION')
print('=' * 100)

for filename, system in mcp_files:
    filepath = f'C:/Users/mmeadow/Documents/OpenEvolve/Frontend/{filename}'

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        print(f'\n{system} ({filename}):')
        print('  [ERROR] Could not read file')
        continue

    # Look for function definitions that might be MCP tools
    tool_functions = []
    for line in lines:
        # Look for def statements
        if re.match(r'^\s*def\s+', line):
            # Check if it looks like a tool function
            if any(keyword in line for keyword in ['solve', 'verify', 'analyze', 'create', 'get', 'run', 'execute']):
                func_name = re.search(r'def\s+(\w+)\s*\(', line)
                if func_name:
                    tool_functions.append(func_name.group(1))

    # Look for registration calls
    has_register = any('register' in line.lower() for line in lines)
    has_mcp_dict = any('_mcp_tools' in line.lower() or 'mcp_tools' in line.lower() for line in lines)
    has_decorator = any('@mcp_tool' in line or '@tool' in line for line in lines)

    print(f'\n{system} ({filename}):')
    print(f'  Potential tool functions: {len(tool_functions)}')
    if tool_functions:
        for func in tool_functions[:5]:
            print(f'    - {func}')
        if len(tool_functions) > 5:
            print(f'    ... and {len(tool_functions) - 5} more')
    print(f'  Has registration calls: {has_register}')
    print(f'  Has MCP dict: {has_mcp_dict}')
    print(f'  Has @mcp_tool decorator: {has_decorator}')

print('\n' + '=' * 100)
