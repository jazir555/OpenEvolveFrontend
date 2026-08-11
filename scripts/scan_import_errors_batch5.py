#!/usr/bin/env python3
"""
Scan Python files for import errors - Batch 5
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Any


def scan_file(file_path: str) -> list[dict[str, Any]]:
    """Scan a single Python file for syntax and import errors."""
    errors = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        errors.append({
            "file": file_path,
            "error_type": "other",
            "line_number": 0,
            "message": f"File not found: {file_path}",
            "suggested_fix": "Check if file path is correct"
        })
        return errors
    
    # Try to read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                source = f.read()
        except Exception as e:
            errors.append({
                "file": file_path,
                "error_type": "other",
                "line_number": 0,
                "message": f"Cannot read file: {str(e)}",
                "suggested_fix": "Check file encoding"
            })
            return errors
    except Exception as e:
        errors.append({
            "file": file_path,
            "error_type": "other",
            "line_number": 0,
            "message": f"Cannot read file: {str(e)}",
            "suggested_fix": "Check file permissions"
        })
        return errors
    
    # Try to parse with AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append({
            "file": file_path,
            "error_type": "syntax_error",
            "line_number": e.lineno if e.lineno else 0,
            "message": f"Syntax error: {e.msg}",
            "suggested_fix": f"Check syntax around line {e.lineno}"
        })
        return errors
    except Exception as e:
        errors.append({
            "file": file_path,
            "error_type": "syntax_error",
            "line_number": 0,
            "message": f"AST parse error: {str(e)}",
            "suggested_fix": "Check Python syntax"
        })
        return errors
    
    # Try to compile with py_compile
    try:
        py_compile.compile(file_path, doraise=True)
    except py_compile.PyCompileError as e:
        error_msg = str(e)
        line_no = 0
        
        # Try to extract line number from error message
        if "line" in error_msg.lower():
            parts = error_msg.lower().split("line")
            if len(parts) > 1:
                try:
                    line_no = int(parts[1].split()[0].strip())
                except:
                    pass
        
        # Classify the error
        if "syntax" in error_msg.lower():
            error_type = "syntax_error"
        elif "import" in error_msg.lower():
            error_type = "import_error"
        else:
            error_type = "other"
        
        errors.append({
            "file": file_path,
            "error_type": error_type,
            "line_number": line_no,
            "message": f"Compilation error: {error_msg}",
            "suggested_fix": "Check file syntax and imports"
        })
        return errors
    
    # Analyze imports in the AST
    import_errors = analyze_imports(tree, file_path, source)
    errors.extend(import_errors)
    
    return errors


def analyze_imports(tree: ast.AST, file_path: str, source: str) -> list[dict[str, Any]]:
    """Analyze imports for potential issues."""
    errors = []
    file_dir = os.path.dirname(file_path)
    
    # Track imports and from imports
    imports = []
    from_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'name': alias.name,
                    'lineno': node.lineno,
                    'asname': alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            from_imports.append({
                'module': module,
                'names': [alias.name for alias in node.names],
                'lineno': node.lineno,
                'level': node.level  # For relative imports
            })
    
    # Check for circular import patterns (files importing each other)
    # This is a basic check - real circular imports happen at runtime
    
    # Check for relative imports that might not resolve
    for imp in from_imports:
        if imp['level'] > 0:  # Relative import
            # Check if the relative import can resolve
            if not check_relative_import(file_dir, imp['module'], imp['level']):
                errors.append({
                    "file": file_path,
                    "error_type": "import_error",
                    "line_number": imp['lineno'],
                    "message": f"Relative import may not resolve: from {'.' * imp['level']}{imp['module']} import {', '.join(imp['names'])}",
                    "suggested_fix": f"Check if module exists at relative path level {imp['level']}"
                })
    
    # Check for common typo patterns in imports
    common_typos = {
        'typping': 'typing',
        'typng': 'typing',
        'tying': 'typing',
        'jsonn': 'json',
        'os.pathh': 'os.path',
        'sysy': 'sys',
        'pathlibb': 'pathlib',
        'collecttions': 'collections',
        'dataclasss': 'dataclasses',
        'abcx': 'abc',
        'functoolss': 'functools',
    }
    
    for imp in imports:
        base_module = imp['name'].split('.')[0]
        if base_module in common_typos:
            errors.append({
                "file": file_path,
                "error_type": "import_error",
                "line_number": imp['lineno'],
                "message": f"Possible typo in import: '{imp['name']}' (did you mean '{common_typos[base_module]}'?)",
                "suggested_fix": f"Change to '{common_typos[base_module]}'"
            })
    
    for imp in from_imports:
        if imp['module']:
            base_module = imp['module'].split('.')[0]
            if base_module in common_typos:
                errors.append({
                    "file": file_path,
                    "error_type": "import_error",
                    "line_number": imp['lineno'],
                    "message": f"Possible typo in import: '{imp['module']}' (did you mean '{common_typos[base_module]}'?)",
                    "suggested_fix": f"Change to '{common_typos[base_module]}'"
                })
    
    return errors


def check_relative_import(file_dir: str, module: str, level: int) -> bool:
    """Check if a relative import can potentially resolve."""
    # Navigate up the directory tree based on level
    base_dir = file_dir
    for _ in range(level - 1):
        parent = os.path.dirname(base_dir)
        if parent == base_dir:  # Hit root
            return False
        base_dir = parent
    
    # Check if the module path exists
    if module:
        module_parts = module.split('.')
        check_path = os.path.join(base_dir, *module_parts)
        
        # Check as package
        if os.path.isdir(check_path) and os.path.exists(os.path.join(check_path, '__init__.py')):
            return True
        # Check as module
        if os.path.exists(check_path + '.py'):
            return True
    else:
        # Just checking parent exists
        parent = os.path.dirname(base_dir)
        return parent != base_dir or os.path.exists(os.path.join(base_dir, '__init__.py'))
    
    return False


def main():
    """Main function to scan batch 5 files."""
    batch_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_5.txt"
    output_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_5.json"
    
    # Read file list
    with open(batch_file, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Scanning {len(files)} files from batch 5...")
    
    all_errors = []
    processed = 0
    
    for file_path in files:
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(files)} files...")
        
        errors = scan_file(file_path)
        all_errors.extend(errors)
    
    # Generate report
    report = {
        "total_files": len(files),
        "errors_found": len(all_errors),
        "errors": all_errors
    }
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nScan complete!")
    print(f"Total files scanned: {len(files)}")
    print(f"Errors found: {len(all_errors)}")
    print(f"Report saved to: {output_file}")
    
    # Print summary of errors
    if all_errors:
        print("\nError summary by type:")
        error_types = {}
        for err in all_errors:
            err_type = err['error_type']
            error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items()):
            print(f"  {err_type}: {count}")
        
        print("\nFirst 10 errors:")
        for err in all_errors[:10]:
            print(f"  - {err['file']}: {err['message'][:80]}...")


if __name__ == "__main__":
    main()
