#!/usr/bin/env python3
"""
Scan Python files for import errors - Batch 7
"""

import ast
import json
import py_compile
import sys
from pathlib import Path
from typing import Optional


def get_imports_from_ast(file_path: str, content: str) -> tuple:
    """Extract imports from AST."""
    imports = []
    from_imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                from_imports.append((module, names))
    except SyntaxError:
        pass
    return imports, from_imports


def check_syntax_with_ast(file_path: str, content: str) -> Optional[dict]:
    """Check syntax using AST parsing."""
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        return {
            "error_type": "syntax_error",
            "line_number": e.lineno if e.lineno else 0,
            "message": f"Syntax error: {str(e)}",
            "suggested_fix": f"Check syntax around line {e.lineno if e.lineno else 'unknown'}"
        }


def check_compile(file_path: str) -> Optional[dict]:
    """Check if file compiles with py_compile."""
    try:
        py_compile.compile(file_path, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return {
            "error_type": "syntax_error",
            "line_number": 0,
            "message": f"Compilation error: {str(e)}",
            "suggested_fix": "Fix syntax errors before checking imports"
        }


def analyze_imports(file_path: str, content: str, imports: list, from_imports: list) -> list:
    """Analyze imports for potential issues."""
    errors = []
    file_dir = Path(file_path).parent
    
    # Check for relative imports
    for module, names in from_imports:
        # Check for relative imports that might not resolve
        if module.startswith('.'):
            # Relative import - check if it resolves
            try:
                # Find the package root
                parts = module.split('.')
                current = file_dir
                for part in parts:
                    if part == '':
                        current = current.parent
                    else:
                        current = current / part
                
                # Check if the module exists
                if not (current.exists() or (current.parent / f"{current.name}.py").exists()):
                    errors.append({
                        "error_type": "import_error",
                        "line_number": 0,
                        "message": f"Relative import '{module}' may not resolve",
                        "suggested_fix": f"Ensure module exists at expected path"
                    })
            except Exception:
                pass
    
    # Check for circular imports (basic heuristic)
    for module in imports:
        # If file imports itself or a module that likely imports this file
        module_parts = module.split('.')
        if len(module_parts) > 1:
            # Check if any part matches current file name
            current_file_name = Path(file_path).stem
            if current_file_name in module_parts:
                errors.append({
                    "error_type": "circular_import",
                    "line_number": 0,
                    "message": f"Potential circular import with '{module}'",
                    "suggested_fix": "Consider restructuring imports to avoid circular dependencies"
                })
    
    return errors


def scan_file(file_path: str) -> Optional[dict]:
    """Scan a single Python file for import errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            return {
                "file": file_path,
                "error_type": "other",
                "line_number": 0,
                "message": f"Cannot read file: {str(e)}",
                "suggested_fix": "Check file encoding"
            }
    except Exception as e:
        return {
            "file": file_path,
            "error_type": "other",
            "line_number": 0,
            "message": f"Cannot read file: {str(e)}",
            "suggested_fix": "Check file permissions"
        }
    
    errors = []
    
    # Check syntax with AST
    syntax_error = check_syntax_with_ast(file_path, content)
    if syntax_error:
        syntax_error["file"] = file_path
        errors.append(syntax_error)
        # If syntax error, skip further analysis
        return errors[0] if errors else None
    
    # Check compile
    compile_error = check_compile(file_path)
    if compile_error:
        compile_error["file"] = file_path
        errors.append(compile_error)
        return errors[0] if errors else None
    
    # Get imports
    imports, from_imports = get_imports_from_ast(file_path, content)
    
    # Analyze imports
    import_errors = analyze_imports(file_path, content, imports, from_imports)
    for err in import_errors:
        err["file"] = file_path
        errors.append(err)
    
    return errors[0] if errors else None


def main():
    batch_file = "c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\batch_7.txt"
    output_file = "c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\import_errors_batch_7.json"
    
    # Read file list
    with open(batch_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Scanning {len(files)} files...")
    
    errors_found = []
    total_files = len(files)
    
    for i, file_path in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{total_files} files scanned...")
        
        if not file_path.endswith('.py'):
            continue
        
        if not Path(file_path).exists():
            errors_found.append({
                "file": file_path,
                "error_type": "other",
                "line_number": 0,
                "message": "File does not exist",
                "suggested_fix": "Remove from batch or create file"
            })
            continue
        
        error = scan_file(file_path)
        if error:
            errors_found.append(error)
    
    # Create report
    report = {
        "total_files": total_files,
        "errors_found": len(errors_found),
        "errors": errors_found
    }
    
    # Write report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nScan complete!")
    print(f"Total files: {total_files}")
    print(f"Errors found: {len(errors_found)}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
