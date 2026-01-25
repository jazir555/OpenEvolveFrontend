"""
Bug Check and Code Analysis

This script performs static analysis on the AI integration code
to identify potential bugs, issues, or areas for improvement.
"""

import sys
import os
import ast
import re
from typing import Dict, Any, List

def check_file_for_issues(file_path: str) -> Dict[str, Any]:
    """Check a single file for potential issues."""
    
    results = {
        'file': file_path,
        'issues': [],
        'warnings': [],
        'status': 'ok'
    }
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common issues
        issues = []
        warnings = []
        
        # 1. Check for TODO comments (potential incomplete code)
        todo_count = len(re.findall(r'#\s*TODO', content, re.IGNORECASE))
        if todo_count > 0:
            warnings.append(f'Found {todo_count} TODO comments - check for incomplete features')
        
        # 2. Check for FIXME comments
        fixme_count = len(re.findall(r'#\s*FIXME', content, re.IGNORECASE))
        if fixme_count > 0:
            warnings.append(f'Found {fixme_count} FIXME comments - check for known issues')
        
        # 3. Check for HACK comments
        hack_count = len(re.findall(r'#\s*HACK', content, re.IGNORECASE))
        if hack_count > 0:
            warnings.append(f'Found {hack_count} HACK comments - check for suboptimal solutions')
        
        # 4. Check for XXX comments
        xxx_count = len(re.findall(r'#\s*XXX', content, re.IGNORECASE))
        if xxx_count > 0:
            warnings.append(f'Found {xxx_count} XXX comments - check for problematic code')
        
        # 5. Check for bare except clauses
        bare_except_count = len(re.findall(r'except:\s*(?:#|$)', content))
        if bare_except_count > 0:
            warnings.append(f'Found {bare_except_count} bare except clauses - should specify exception types')
        
        # 6. Check for print statements (should use logging in production)
        print_count = len(re.findall(r'print\(', content))
        if print_count > 5:  # More than 5 print statements might indicate debugging code
            warnings.append(f'Found {print_count} print statements - consider using logging for production')
        
        # 7. Check for hardcoded paths or credentials
        hardcoded_paths = re.findall(r'["\'](/[a-zA-Z0-9_/\.\-]+)["\']', content)
        if len(hardcoded_paths) > 3:
            warnings.append(f'Found {len(hardcoded_paths)} potential hardcoded paths - check for config issues')
        
        # 8. Check for long functions (more than 100 lines)
        lines = content.split('\n')
        in_function = False
        function_start = 0
        long_functions = []
        
        for i, line in enumerate(lines):
            if re.match(r'\s*def\s+\w+\s*\(', line):
                in_function = True
                function_start = i
            elif in_function and line.strip() and not line.strip().startswith('#'):
                # Check if this is the start of a new function
                if re.match(r'\s*def\s+\w+\s*\(', line):
                    function_length = i - function_start
                    if function_length > 100:
                        function_name = lines[function_start].split('def')[1].split('(')[0].strip()
                        long_functions.append(function_name)
                    function_start = i
            elif in_function and (line.strip() == '' or line.strip().startswith('#')):
                # End of function
                function_length = i - function_start
                if function_length > 100:
                    function_name = lines[function_start].split('def')[1].split('(')[0].strip()
                    long_functions.append(function_name)
                in_function = False
        
        if long_functions:
            warnings.append(f'Found long functions: {", ".join(long_functions)} - consider refactoring')