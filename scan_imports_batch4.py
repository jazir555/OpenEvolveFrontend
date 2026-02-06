#!/usr/bin/env python3
"""
Scan Python files for import errors in batch 4.
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_with_ast(file_path: str) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Parse a Python file with AST to check for syntax errors.
    Returns (success, error_info)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        if not source.strip():
            # Empty file - not an error but worth noting
            return True, None
            
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, {
            "type": "syntax_error",
            "line_number": e.lineno if e.lineno else 0,
            "message": str(e),
            "text": e.text if e.text else ""
        }
    except UnicodeDecodeError as e:
        return False, {
            "type": "unicode_error",
            "line_number": 0,
            "message": f"Unicode decode error: {e}",
            "text": ""
        }
    except Exception as e:
        return False, {
            "type": "parse_error",
            "line_number": 0,
            "message": f"Parse error: {e}",
            "text": ""
        }


def compile_with_py_compile(file_path: str) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Compile a Python file with py_compile to catch import/syntax errors.
    Returns (success, error_info)
    """
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        # Extract line number from the exception if possible
        line_number = 0
        msg = str(e)
        
        # Try to extract line number from error message
        if "line" in msg.lower():
            parts = msg.split()
            for i, part in enumerate(parts):
                if part.lower() == "line" and i + 1 < len(parts):
                    try:
                        line_number = int(parts[i + 1].rstrip(':,.'))
                        break
                    except ValueError:
                        continue
        
        return False, {
            "type": "compile_error",
            "line_number": line_number,
            "message": msg,
            "text": ""
        }
    except Exception as e:
        return False, {
            "type": "compile_error",
            "line_number": 0,
            "message": str(e),
            "text": ""
        }


def analyze_imports(file_path: str) -> List[Dict[str, Any]]:
    """
    Analyze imports in a Python file for potential issues.
    Returns a list of potential import issues.
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        if not source.strip():
            return issues
            
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check for suspicious imports
                    if alias.name in ['StringIO', 'cStringIO', 'urllib2', 'urlparse']:
                        issues.append({
                            "type": "deprecated_import",
                            "line_number": node.lineno,
                            "module": alias.name,
                            "message": f"Potentially deprecated module: {alias.name}",
                            "suggested_fix": f"Consider using a Python 3 compatible alternative for {alias.name}"
                        })
                    
                    # Check for imports that might indicate Python 2 code
                    if alias.name in ['__future__']:
                        # Check for Python 2 specific imports
                        pass  # __future__ is valid in Python 3 too
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Check for relative imports that might be problematic
                    if node.level > 0:  # Relative import
                        # Check if this is in the root of the project
                        file_dir = os.path.dirname(file_path)
                        # If in root and using relative imports, might be an issue
                        rel_path = os.path.relpath(file_path, os.getcwd())
                        if '/' not in rel_path.replace('\\', '/') and node.level > 0:
                            issues.append({
                                "type": "relative_import_in_root",
                                "line_number": node.lineno,
                                "module": node.module,
                                "message": f"Relative import in root-level file: from {'.' * node.level}{node.module}",
                                "suggested_fix": "Use absolute imports for root-level files"
                            })
                    
                    # Check for circular import patterns (simplified check)
                    if node.module.startswith('.'):
                        issues.append({
                            "type": "circular_import_risk",
                            "line_number": node.lineno,
                            "module": node.module,
                            "message": f"Potential circular import: {node.module}",
                            "suggested_fix": "Review import structure to avoid circular dependencies"
                        })
                        
    except Exception:
        # If we can't parse, skip import analysis
        pass
    
    return issues


def check_file_exists_for_import(file_path: str, import_module: str) -> bool:
    """
    Check if the module being imported exists relative to the file.
    This is a simplified check.
    """
    file_dir = os.path.dirname(file_path)
    parts = import_module.split('.')
    
    # Check for the module file
    for ext in ['.py', '/__init__.py', '']:
        possible_path = os.path.join(file_dir, *parts) + ext
        if os.path.exists(possible_path):
            return True
    
    return False


def scan_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for import and syntax errors.
    Returns a list of errors found.
    """
    errors = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        return [{
            "file": file_path,
            "error_type": "file_not_found",
            "line_number": 0,
            "message": "File does not exist",
            "suggested_fix": "Check file path"
        }]
    
    # Check if file is empty
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            # Empty file - not an error but skip further checks
            return []
    except Exception as e:
        return [{
            "file": file_path,
            "error_type": "read_error",
            "line_number": 0,
            "message": f"Cannot read file: {e}",
            "suggested_fix": "Check file permissions"
        }]
    
    # Try AST parsing
    ast_success, ast_error = parse_with_ast(file_path)
    if not ast_success:
        errors.append({
            "file": file_path,
            "error_type": "syntax_error",
            "line_number": ast_error.get("line_number", 0) if ast_error else 0,
            "message": ast_error.get("message", "Unknown syntax error") if ast_error else "Unknown syntax error",
            "suggested_fix": "Fix syntax error in file"
        })
    
    # Try py_compile
    compile_success, compile_error = compile_with_py_compile(file_path)
    if not compile_success:
        # Avoid duplicate errors
        error_msg = compile_error.get("message", "") if compile_error else ""
        if not any(e.get("message") == error_msg for e in errors):
            errors.append({
                "file": file_path,
                "error_type": "compile_error",
                "line_number": compile_error.get("line_number", 0) if compile_error else 0,
                "message": error_msg or "Compilation error",
                "suggested_fix": "Fix compilation error"
            })
    
    # Analyze imports
    import_issues = analyze_imports(file_path)
    for issue in import_issues:
        errors.append({
            "file": file_path,
            "error_type": issue.get("type", "import_warning"),
            "line_number": issue.get("line_number", 0),
            "message": issue.get("message", ""),
            "suggested_fix": issue.get("suggested_fix", "")
        })
    
    return errors


def main():
    """
    Main function to scan batch 4 files.
    """
    batch_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_4.txt"
    output_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_4.json"
    
    # Read the batch file
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading batch file: {e}")
        sys.exit(1)
    
    # Filter only Python files
    python_files = [f for f in files if f.endswith('.py')]
    print(f"Found {len(python_files)} Python files to scan in batch 4")
    
    # Scan each file
    all_errors = []
    total_files = len(python_files)
    
    for i, file_path in enumerate(python_files, 1):
        if i % 50 == 0:
            print(f"  Scanned {i}/{total_files} files...")
        
        errors = scan_file(file_path)
        all_errors.extend(errors)
    
    # Generate report
    report = {
        "total_files": total_files,
        "errors_found": len(all_errors),
        "errors": all_errors
    }
    
    # Write JSON report
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to: {output_file}")
        print(f"Total files scanned: {total_files}")
        print(f"Total errors found: {len(all_errors)}")
        
        # Print summary by error type
        error_types = {}
        for error in all_errors:
            etype = error.get("error_type", "unknown")
            error_types[etype] = error_types.get(etype, 0) + 1
        
        if error_types:
            print("\nErrors by type:")
            for etype, count in sorted(error_types.items()):
                print(f"  {etype}: {count}")
        
    except Exception as e:
        print(f"Error writing report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
