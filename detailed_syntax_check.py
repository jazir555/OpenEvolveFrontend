#!/usr/bin/env python3
"""Detailed syntax and import error checking."""

import ast
import glob
import json
import re
import warnings

def check_file(filepath):
    """Check a single file for various issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for invalid escape sequences
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compile(content, filepath, 'exec')
            for warning in w:
                if 'invalid escape sequence' in str(warning.message):
                    issues.append({
                        'type': 'invalid_escape',
                        'message': str(warning.message),
                        'line': warning.lineno if hasattr(warning, 'lineno') else None
                    })
        
        # Parse AST
        tree = ast.parse(content)
        
        # Check for f-string issues
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                # Check f-string content
                pass
            
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['__builtin__', 'cPickle', 'urllib2']:
                        issues.append({
                            'type': 'python2_import',
                            'message': f'Python 2 import: {alias.name}',
                            'line': node.lineno
                        })
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('.'):
                    # Relative import - might cause issues
                    pass
        
        # Check for f-string with backslash (Python 3.12+ only, errors before)
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for f"...\..." patterns
            if re.search(r'f["\'].*\\.*["\']', line):
                # This might be an issue
                pass
            
    except SyntaxError as e:
        issues.append({
            'type': 'syntax_error',
            'message': str(e),
            'line': e.lineno,
            'text': e.text
        })
    except Exception as e:
        issues.append({
            'type': 'other_error',
            'message': str(e)
        })
    
    return issues

def main():
    patterns = [
        'workflow*.py',
        'decomposition*.py',
        '*engine*.py',
        'evolution*.py',
        'knowledge*.py',
        'gauntlet*.py'
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    
    all_issues = []
    files_with_issues = []
    
    for filepath in files:
        issues = check_file(filepath)
        if issues:
            files_with_issues.append(filepath)
            for issue in issues:
                all_issues.append({
                    'file': filepath,
                    **issue
                })
    
    # Save report
    report = {
        'files_scanned': len(files),
        'files_with_issues': files_with_issues,
        'issues_found': len(all_issues),
        'issues': all_issues
    }
    
    with open('temp_detailed_issues.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Files scanned: {len(files)}")
    print(f"Files with issues: {len(files_with_issues)}")
    print(f"Total issues: {len(all_issues)}")
    
    for issue in all_issues:
        print(f"\n{issue['file']}:{issue.get('line', '?')}")
        print(f"  [{issue['type']}] {issue['message']}")

if __name__ == '__main__':
    main()
