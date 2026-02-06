#!/usr/bin/env python3
"""
Scan Python files in batches 11-16 for import errors.
Generates a consolidated JSON report.
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Batch files to process
BATCH_FILES = [
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_11.txt",
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_12.txt",
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_13.txt",
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_14.txt",
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_15.txt",
    r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_16.txt",
]

OUTPUT_FILE = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_11_16.json"


def get_all_python_files() -> List[str]:
    """Read all Python file paths from batch files."""
    all_files = []
    for batch_file in BATCH_FILES:
        if os.path.exists(batch_file):
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.endswith('.py'):
                        all_files.append(line)
    return all_files


def check_syntax_ast(file_path: str) -> Optional[Dict[str, Any]]:
    """Check file for syntax errors using AST parser."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        ast.parse(source)
        return None
    except SyntaxError as e:
        return {
            "error_type": "syntax_error",
            "line_number": e.lineno if e.lineno else 0,
            "message": str(e),
            "suggested_fix": "Check for invalid Python syntax, missing colons, or mismatched brackets"
        }
    except Exception as e:
        return {
            "error_type": "other",
            "line_number": 0,
            "message": f"AST parsing failed: {str(e)}",
            "suggested_fix": "Check file encoding and content"
        }


def check_compile(file_path: str) -> Optional[Dict[str, Any]]:
    """Check file for compilation errors using py_compile."""
    try:
        py_compile.compile(file_path, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return {
            "error_type": "syntax_error",
            "line_number": 0,
            "message": str(e),
            "suggested_fix": "Fix syntax errors identified by the Python compiler"
        }


def extract_imports(file_path: str) -> List[Dict[str, Any]]:
    """Extract all imports from a Python file using AST."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "lineno": node.lineno,
                        "level": 0
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                level = node.level if node.level else 0
                imports.append({
                    "type": "from_import",
                    "module": module,
                    "lineno": node.lineno,
                    "level": level
                })
    except:
        pass
    return imports


def check_import_issues(file_path: str, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check for common import issues like relative imports and potential typos."""
    issues = []
    file_dir = os.path.dirname(file_path)
    
    for imp in imports:
        # Check for relative imports that may not resolve
        if imp["type"] == "from_import" and imp["level"] > 0:
            relative_path = file_dir
            for _ in range(imp["level"]):
                relative_path = os.path.dirname(relative_path)
            
            if imp["module"]:
                module_parts = imp["module"].split('.')
                potential_path = os.path.join(relative_path, *module_parts)
                if not os.path.exists(potential_path + ".py") and not os.path.exists(os.path.join(potential_path, "__init__.py")):
                    issues.append({
                        "error_type": "import_error",
                        "line_number": imp["lineno"],
                        "message": f"Relative import '{'.' * imp['level']}{imp['module']}' may not resolve",
                        "suggested_fix": f"Verify the module exists at the expected relative path"
                    })
        
        # Check for common typo patterns in module names
        module = imp.get("module", "")
        if module:
            common_typos = [
                ("openevovle", "openevolve"),
                ("opnevolve", "openevolve"),
                ("leanaide", "leanaide"),
                ("bubblelab", "bubblelab"),
                ("guardrails", "guardrails"),
            ]
            for typo, correct in common_typos:
                if typo in module.lower() and typo != correct:
                    issues.append({
                        "error_type": "import_error",
                        "line_number": imp["lineno"],
                        "message": f"Potential typo in module name: '{module}'",
                        "suggested_fix": f"Check if you meant '{correct}' instead of '{typo}'"
                    })
    
    return issues


def scan_file(file_path: str) -> List[Dict[str, Any]]:
    """Scan a single file for all types of errors."""
    errors = []
    
    # Skip files that don't exist
    if not os.path.exists(file_path):
        return [{
            "file": file_path,
            "error_type": "other",
            "line_number": 0,
            "message": "File does not exist",
            "suggested_fix": "Verify the file path"
        }]
    
    # Check syntax with AST
    ast_error = check_syntax_ast(file_path)
    if ast_error:
        ast_error["file"] = file_path
        errors.append(ast_error)
        # If syntax error, don't continue with other checks
        return errors
    
    # Check compilation
    compile_error = check_compile(file_path)
    if compile_error:
        compile_error["file"] = file_path
        errors.append(compile_error)
    
    # Extract and check imports
    imports = extract_imports(file_path)
    import_issues = check_import_issues(file_path, imports)
    for issue in import_issues:
        issue["file"] = file_path
        errors.append(issue)
    
    return errors


def main():
    print("Starting import error scan for batches 11-16...")
    
    # Get all Python files
    all_files = get_all_python_files()
    total_files = len(all_files)
    print(f"Found {total_files} Python files to scan")
    
    # Scan each file
    all_errors = []
    errors_found = 0
    
    for i, file_path in enumerate(all_files, 1):
        if i % 100 == 0:
            print(f"Scanned {i}/{total_files} files...")
        
        errors = scan_file(file_path)
        all_errors.extend(errors)
        if errors:
            errors_found += len(errors)
    
    # Create report
    report = {
        "total_files": total_files,
        "errors_found": len(all_errors),
        "errors": all_errors
    }
    
    # Write report to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nScan complete!")
    print(f"Total files scanned: {total_files}")
    print(f"Errors found: {len(all_errors)}")
    print(f"Report saved to: {OUTPUT_FILE}")
    
    # Print summary by error type
    if all_errors:
        error_types = {}
        for err in all_errors:
            err_type = err.get("error_type", "unknown")
            error_types[err_type] = error_types.get(err_type, 0) + 1
        
        print("\nErrors by type:")
        for err_type, count in sorted(error_types.items()):
            print(f"  {err_type}: {count}")


if __name__ == "__main__":
    main()
