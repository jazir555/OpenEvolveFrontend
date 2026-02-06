#!/usr/bin/env python3
"""
Scan Python files for import errors - Batch 6
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import List, Dict, Any


def scan_file_for_imports(filepath: str) -> List[Dict[str, Any]]:
    """Scan a single Python file for import-related issues."""
    errors = []
    
    if not os.path.exists(filepath):
        errors.append({
            "file": filepath,
            "error_type": "other",
            "line_number": 0,
            "message": f"File does not exist: {filepath}",
            "suggested_fix": "Check file path"
        })
        return errors
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                source = f.read()
        except Exception as e:
            errors.append({
                "file": filepath,
                "error_type": "other",
                "line_number": 0,
                "message": f"Cannot read file: {str(e)}",
                "suggested_fix": "Check file encoding"
            })
            return errors
    except Exception as e:
        errors.append({
            "file": filepath,
            "error_type": "other",
            "line_number": 0,
            "message": f"Cannot read file: {str(e)}",
            "suggested_fix": "Check file permissions"
        })
        return errors
    
    # Check 1: Parse with AST for syntax errors
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append({
            "file": filepath,
            "error_type": "syntax_error",
            "line_number": e.lineno or 0,
            "message": f"Syntax error: {e.msg}",
            "suggested_fix": f"Check line {e.lineno}: {e.text.strip() if e.text else 'syntax error'}"
        })
        return errors
    except Exception as e:
        errors.append({
            "file": filepath,
            "error_type": "syntax_error",
            "line_number": 0,
            "message": f"AST parse error: {str(e)}",
            "suggested_fix": "Check for invalid Python syntax"
        })
        return errors
    
    # Check 2: Compile with py_compile
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append({
            "file": filepath,
            "error_type": "syntax_error",
            "line_number": getattr(e, 'lineno', 0) or 0,
            "message": f"Compilation error: {str(e)}",
            "suggested_fix": "Fix syntax error"
        })
        return errors
    
    # Check 3: Analyze imports for common issues
    imported_modules = set()
    imported_names = set()
    
    for node in ast.walk(tree):
        # Check for Import nodes
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                # Check for suspicious module names (potential typos)
                parts = alias.name.split('.')
                for part in parts:
                    if part.startswith('creawai') or part.startswith('creaw'):
                        errors.append({
                            "file": filepath,
                            "error_type": "import_error",
                            "line_number": node.lineno,
                            "message": f"Potential typo in import: '{alias.name}' - did you mean 'crewai'?",
                            "suggested_fix": f"Change '{alias.name}' to 'crewai'"
                        })
        
        # Check for ImportFrom nodes
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)
                
                # Check for relative imports that might not resolve
                if node.level > 0:
                    # Relative import - check if we're at package root
                    parts = filepath.replace('\\', '/').split('/')
                    if node.level >= len(parts) - 1:
                        errors.append({
                            "file": filepath,
                            "error_type": "import_error",
                            "line_number": node.lineno,
                            "message": f"Relative import level {node.level} may not resolve from this file location",
                            "suggested_fix": "Check relative import path or use absolute import"
                        })
                
                # Check for potential typos in crewai imports
                if 'creawai' in node.module or 'creaw' in node.module:
                    errors.append({
                        "file": filepath,
                        "error_type": "import_error",
                        "line_number": node.lineno,
                        "message": f"Potential typo in import: '{node.module}' - did you mean 'crewai'?",
                        "suggested_fix": f"Change '{node.module}' to 'crewai'"
                    })
            
            # Track imported names for circular import detection
            for alias in node.names:
                imported_names.add(alias.name)
    
    # Check 4: Try to actually import the module (optional - might fail for many legitimate reasons)
    # Only check if it's a module file (not a standalone script)
    if filepath.endswith('__init__.py'):
        # Try to construct the module path
        rel_path = filepath.replace('\\', '/')
        if 'src/' in rel_path:
            module_parts = []
            parts = rel_path.split('/')
            found_src = False
            for part in parts:
                if found_src:
                    if part.endswith('.py'):
                        if part != '__init__.py':
                            module_parts.append(part[:-3])
                    else:
                        module_parts.append(part)
                if part == 'src':
                    found_src = True
            
            if module_parts:
                module_name = '.'.join(module_parts)
                try:
                    # Don't actually import as it may have side effects
                    pass
                except Exception as e:
                    errors.append({
                        "file": filepath,
                        "error_type": "import_error",
                        "line_number": 0,
                        "message": f"Module import issue: {str(e)}",
                        "suggested_fix": "Check module dependencies"
                    })
    
    return errors


def main():
    batch_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_6.txt"
    output_file = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_6.json"
    
    # Read the batch file
    with open(batch_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Scanning {len(files)} files for import errors...")
    
    all_errors = []
    total_scanned = 0
    files_with_errors = 0
    
    for i, filepath in enumerate(files, 1):
        if not filepath.endswith('.py'):
            continue
        
        total_scanned += 1
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(files)} files scanned...")
        
        errors = scan_file_for_imports(filepath)
        if errors:
            files_with_errors += 1
            all_errors.extend(errors)
    
    # Create the report
    report = {
        "total_files": total_scanned,
        "errors_found": len(all_errors),
        "files_with_errors": files_with_errors,
        "errors": all_errors
    }
    
    # Write the JSON report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nScan complete!")
    print(f"  Total files scanned: {total_scanned}")
    print(f"  Files with errors: {files_with_errors}")
    print(f"  Total errors found: {len(all_errors)}")
    print(f"  Report saved to: {output_file}")


if __name__ == "__main__":
    main()
