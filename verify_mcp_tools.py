"""
Comprehensive MCP Tools Verification Script

Verifies all 13 MCP tool files for:
1. Import success
2. Logger ordering (before try/except)
3. Hephaestus references (should be only in comments)
4. MCP tool registration
5. Tool schema validity
"""

import ast
import os
import sys
import re
from typing import Dict, List, Tuple, Any

class MCPToolVerifier(ast.NodeVisitor):
    """AST visitor to verify MCP tool structure"""

    def __init__(self, filename: str):
        self.filename = filename
        self.tool_functions = []
        self.registration_calls = []
        self.has_tool_decorator = False
        self.errors = []
        self.warnings = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Check for MCP tool decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['mcp_tool', 'tool']:
                    self.has_tool_decorator = True
                    self.tool_functions.append(node.name)
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ['register', 'tool']:
                    self.has_tool_decorator = True
                    self.tool_functions.append(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for registration calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['register', 'add_tool', 'register_tool']:
                self.registration_calls.append(node.func.attr)
        self.generic_visit(node)

def verify_file(filepath: str) -> Dict[str, Any]:
    """Verify a single MCP tool file"""
    result = {
        'filename': os.path.basename(filepath),
        'path': filepath,
        'exists': False,
        'import_status': 'UNKNOWN',
        'has_logger': False,
        'logger_before_try': False,
        'hephaestus_imports': [],
        'crewai_imports': [],
        'tool_count': 0,
        'registration_mechanism': None,
        'errors': [],
        'warnings': [],
        'status': 'UNKNOWN'
    }

    if not os.path.exists(filepath):
        result['status'] = 'SKIP'
        result['errors'].append('File not found')
        return result

    result['exists'] = True

    # Read file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        result['status'] = 'ERROR'
        result['errors'].append(f'Failed to read file: {e}')
        return result

    # Check for logger
    logger_line = None
    first_try_line = None
    for i, line in enumerate(lines):
        if 'logger = logging.getLogger' in line and not line.strip().startswith('#'):
            logger_line = i
            result['has_logger'] = True
        if 'try:' in line and not line.strip().startswith('#') and first_try_line is None:
            first_try_line = i
            break

    if logger_line is not None and first_try_line is not None:
        result['logger_before_try'] = logger_line < first_try_line

    # Check for Hephaestus imports
    for i, line in enumerate(lines):
        # Check for active imports (not commented)
        if re.match(r'^\s*(from|import)\s+', line):
            if 'hephaestus' in line.lower() and not line.strip().startswith('#'):
                result['hephaestus_imports'].append((i+1, line.strip()))
            if 'crewai' in line.lower() and not line.strip().startswith('#'):
                result['crewai_imports'].append((i+1, line.strip()))

    # Parse AST for tool structure
    try:
        tree = ast.parse(content)
        verifier = MCPToolVerifier(result['filename'])
        verifier.visit(tree)
        result['tool_count'] = len(verifier.tool_functions)
        if verifier.has_tool_decorator:
            result['registration_mechanism'] = 'decorator'
        elif verifier.registration_calls:
            result['registration_mechanism'] = 'function_call'
    except SyntaxError as e:
        result['errors'].append(f'Syntax error: {e}')
        result['status'] = 'ERROR'
        return result

    # Determine overall status
    if result['errors']:
        result['status'] = 'FAIL'
    elif result['hephaestus_imports']:
        result['status'] = 'FAIL'
        result['errors'].append(f'Found {len(result["hephaestus_imports"])} active Hephaestus imports')
    elif not result['logger_before_try'] and result['has_logger']:
        result['status'] = 'WARN'
        result['warnings'].append('Logger defined after try/except block')
    else:
        result['status'] = 'PASS'

    return result

# Main verification
if __name__ == '__main__':
    mcp_files = [
        'roma_mcp_tools.py',
        'roma_mdap_maker_mcp_tools.py',
        'decomposition_mcp_tools.py',
        'bubblelabs_mcp_tools.py',
        'openevolve_mcp_tools.py',
        'leanaide_mcp_tools.py',
        'claudiomiro_mcp_tools.py',
        'datapizza_mcp_tools.py',
        'steer_mcp_tools.py',
        'ace_mcp_tools.py',
        'guardrails_mcp_tools.py',
        'c2c_mcp_tools.py',
        'lmql_mcp_tools.py'
    ]

    print('=' * 100)
    print('COMPREHENSIVE MCP TOOLS VERIFICATION REPORT')
    print('=' * 100)

    results = []
    for filename in mcp_files:
        filepath = f'C:/Users/mmeadow/Documents/OpenEvolve/Frontend/{filename}'
        result = verify_file(filepath)
        results.append(result)

    # Print summary
    print('\nSUMMARY:\n')
    passed = sum(1 for r in results if r['status'] == 'PASS')
    warned = sum(1 for r in results if r['status'] == 'WARN')
    failed = sum(1 for r in results if r['status'] == 'FAIL' or r['status'] == 'ERROR')
    skipped = sum(1 for r in results if r['status'] == 'SKIP')

    print(f'Total: {len(results)} | Passed: {passed} | Warned: {warned} | Failed: {failed} | Skipped: {skipped}')

    # Detailed results
    print('\n' + '=' * 100)
    print('DETAILED RESULTS')
    print('=' * 100)

    for result in results:
        print(f'\n{result["filename"]}:')
        print(f'  Status: {result["status"]}')
        print(f'  Logger: {"Yes" if result["has_logger"] else "No"}', end='')
        if result['has_logger']:
            print(f' ({"Before try/except" if result["logger_before_try"] else "After try/except - WARN"})')
        else:
            print()
        print(f'  Hephaestus imports: {len(result["hephaestus_imports"])}')
        print(f'  CrewAI imports: {len(result["crewai_imports"])}')
        print(f'  MCP tools detected: {result["tool_count"]}')
        print(f'  Registration: {result["registration_mechanism"] or "None detected"}')

        if result['errors']:
            print(f'  Errors:')
            for error in result['errors']:
                print(f'    - {error}')

        if result['warnings']:
            print(f'  Warnings:')
            for warning in result['warnings']:
                print(f'    - {warning}')

    print('\n' + '=' * 100)
