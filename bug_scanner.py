#!/usr/bin/env python3
"""Comprehensive bug scanner for OpenEvolve Frontend Python files."""

__all__ = ['scan_file', 'scan_all_files', 'generate_report', 'bugs']

import ast
import os
import sys
import re
from collections import defaultdict

bugs = []
py_files = sorted([f for f in os.listdir('.') if f.endswith('.py')])

def scan_file(filename):
    """Scan a single Python file for bugs."""
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')

        # Syntax check
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError as e:
            if 'test' not in filename.lower():
                bugs.append({
                    'file': filename,
                    'line': e.lineno,
                    'category': 'SYNTAX_ERROR',
                    'severity': 'CRITICAL',
                    'description': f'Syntax error: {e.msg}',
                    'evidence': lines[e.lineno-1][:100] if e.lineno <= len(lines) else ''
                })
            return

        # Line-by-line analysis
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Security: Shell injection
            if 'shell=True' in line and ('subprocess' in line or 'Popen' in line):
                bugs.append({
                    'file': filename,
                    'line': line_num,
                    'category': 'SECURITY_SHELL_INJECTION',
                    'severity': 'HIGH',
                    'description': 'shell=True in subprocess allows shell injection',
                    'evidence': stripped[:100]
                })

            # Security: os.system
            if 'os.system(' in line:
                if 'test' not in filename.lower() and 'example' not in filename.lower():
                    bugs.append({
                        'file': filename,
                        'line': line_num,
                        'category': 'SECURITY_SHELL_INJECTION',
                        'severity': 'HIGH',
                        'description': 'os.system() allows shell injection',
                        'evidence': stripped[:100]
                    })

            # Security: eval/exec
            if re.search(r'\b(eval|exec)\s*\(', line):
                if 'test' not in filename.lower() and 'example' not in filename.lower():
                    func = 'eval()' if 'eval(' in line else 'exec()'
                    bugs.append({
                        'file': filename,
                        'line': line_num,
                        'category': 'SECURITY_CODE_INJECTION',
                        'severity': 'HIGH',
                        'description': f'{func} allows arbitrary code execution',
                        'evidence': stripped[:100]
                    })

            # Security: Hardcoded credentials
            pwd_match = re.search(
                r'(password|api_key|secret|pwd)\s*=\s*["\'][^"\']{3,}["\']',
                line, re.IGNORECASE
            )
            if pwd_match and 'test' not in filename.lower():
                if 'your_' not in line.lower() and 'placeholder' not in line.lower():
                    bugs.append({
                        'file': filename,
                        'line': line_num,
                        'category': 'SECURITY_HARDCODED_CREDENTIALS',
                        'severity': 'HIGH',
                        'description': 'Hardcoded credential detected',
                        'evidence': stripped[:100]
                    })

            # Code quality: Bare except
            if re.search(r'except\s*:\s*$', line):
                bugs.append({
                    'file': filename,
                    'line': line_num,
                    'category': 'CODE_QUALITY_BARE_EXCEPT',
                    'severity': 'MEDIUM',
                    'description': 'Bare except catches all exceptions including SystemExit',
                    'evidence': stripped[:100]
                })

            # Code quality: Broad exception
            if re.search(r'except\s+Exception\s*:', line):
                bugs.append({
                    'file': filename,
                    'line': line_num,
                    'category': 'CODE_QUALITY_BROAD_EXCEPT',
                    'severity': 'MEDIUM',
                    'description': 'Overly broad exception handler (catches Exception)',
                    'evidence': stripped[:100]
                })

            # Code quality: Mutable default args
            if re.search(r'def\s+\w+\([^)]*=\s*\[', line):
                bugs.append({
                    'file': filename,
                    'line': line_num,
                    'category': 'CODE_QUALITY_MUTABLE_DEFAULT',
                    'severity': 'MEDIUM',
                    'description': 'Mutable default argument (list) can cause bugs',
                    'evidence': stripped[:100]
                })

            # Style: None comparison
            if ' == None' in line or '!= None' in line:
                bugs.append({
                    'file': filename,
                    'line': line_num,
                    'category': 'CODE_STYLE',
                    'severity': 'LOW',
                    'description': 'Use "is None" instead of "== None"',
                    'evidence': stripped[:100]
                })

        # AST checks
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        bugs.append({
                            'file': filename,
                            'line': node.lineno,
                            'category': 'SECURITY_CODE_INJECTION',
                            'severity': 'HIGH',
                            'description': 'eval() allows arbitrary code execution',
                            'evidence': 'eval()'
                        })
                    elif node.func.id == 'exec':
                        bugs.append({
                            'file': filename,
                            'line': node.lineno,
                            'category': 'SECURITY_CODE_INJECTION',
                            'severity': 'HIGH',
                            'description': 'exec() allows arbitrary code execution',
                            'evidence': 'exec()'
                        })

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        bugs.append({
            'file': filename,
            'line': 0,
            'category': 'SCAN_ERROR',
            'severity': 'LOW',
            'description': f'Scan error: {str(e)[:60]}',
            'evidence': ''
        })

print('Scanning for bugs...', file=sys.stderr)
for f in py_files:
    scan_file(f)

# Output results
print(f'Total bugs found: {len(bugs)}')
for bug in bugs:
    print(f'{bug["severity"]}|{bug["category"]}|{bug["file"]}|{bug["line"]}|{bug["description"]}|{bug["evidence"]}')
