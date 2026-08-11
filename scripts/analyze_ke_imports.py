#!/usr/bin/env python3
"""
Comprehensive import analyzer for knowledge_engine directory.
Finds broken imports, non-existent modules, and circular dependencies.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ImportIssue:
    file_path: str
    line_number: int
    import_statement: str
    issue_type: str  # 'module_not_found', 'class_not_found', 'circular_import', etc.
    details: str

@dataclass
class ModuleInfo:
    path: str
    exports: Set[str] = field(default_factory=set)  # Classes, functions, variables defined
    imports: List[Tuple[int, str, str]] = field(default_factory=list)  # (line, import_stmt, type)

class ImportAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.modules: Dict[str, ModuleInfo] = {}
        self.issues: List[ImportIssue] = []
        self.knowledge_engine_dir = self.root_dir / "knowledge_engine"
        
    def scan_all_modules(self):
        """Scan all Python files and build module info."""
        print(f"Scanning {self.knowledge_engine_dir}...")
        
        for py_file in self.knowledge_engine_dir.rglob("*.py"):
            if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
                continue  # Skip test files as requested
            if "tests" in py_file.parts:
                continue  # Skip tests directories
                
            rel_path = py_file.relative_to(self.root_dir)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                exports = self._extract_exports(tree)
                imports = self._extract_imports(tree)
                
                self.modules[module_name] = ModuleInfo(
                    path=str(rel_path),
                    exports=exports,
                    imports=imports
                )
            except SyntaxError as e:
                print(f"  Syntax error in {rel_path}: {e}")
            except Exception as e:
                print(f"  Error parsing {rel_path}: {e}")
        
        print(f"Found {len(self.modules)} modules")
    
    def _extract_exports(self, tree: ast.AST) -> Set[str]:
        """Extract all classes, functions, and variables defined at module level."""
        exports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                exports.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                exports.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        exports.add(target.id)
        
        # Check __all__ if present
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports.add(elt.value)
        
        return exports
    
    def _extract_imports(self, tree: ast.AST) -> List[Tuple[int, str, str]]:
        """Extract all import statements with line numbers."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((node.lineno, f"import {alias.name}", "import"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                level = "." * node.level
                imports.append((node.lineno, f"from {level}{module} import {names}", "from"))
        
        return imports
    
    def check_all_imports(self):
        """Check all imports for validity."""
        print("\nChecking imports...")
        
        for module_name, module_info in self.modules.items():
            for line_no, import_stmt, import_type in module_info.imports:
                self._validate_import(module_name, line_no, import_stmt, import_type)
    
    def _validate_import(self, source_module: str, line_no: int, import_stmt: str, import_type: str):
        """Validate a single import statement."""
        # Parse the import to get module and names
        if import_stmt.startswith("from "):
            # from X import Y
            parts = import_stmt[5:].split(" import ")
            if len(parts) != 2:
                return
            
            module_path = parts[0].strip()
            imported_names = [n.strip() for n in parts[1].split(",")]
            
            # Handle relative imports
            if module_path.startswith("."):
                self._check_relative_import(source_module, line_no, import_stmt, module_path, imported_names)
            else:
                self._check_absolute_import(source_module, line_no, import_stmt, module_path, imported_names)
        
        elif import_stmt.startswith("import "):
            # import X or import X.Y
            module_path = import_stmt[7:].strip()
            self._check_module_import(source_module, line_no, import_stmt, module_path)
    
    def _get_module_path_parts(self, source_module: str) -> List[str]:
        """Get the path parts of a module."""
        return source_module.split(".")
    
    def _check_relative_import(self, source_module: str, line_no: int, import_stmt: str, 
                                module_path: str, imported_names: List[str]):
        """Check relative imports (from .X import Y)."""
        # Count dots
        dots = 0
        for c in module_path:
            if c == ".":
                dots += 1
            else:
                break
        
        rel_path = module_path[dots:]
        source_parts = self._get_module_path_parts(source_module)
        
        # Calculate target module
        if dots > len(source_parts):
            self.issues.append(ImportIssue(
                file_path=source_module,
                line_number=line_no,
                import_statement=import_stmt,
                issue_type="invalid_relative_import",
                details=f"Too many dots ({dots}) for module depth {len(source_parts)}"
            ))
            return
        
        # Build target module name
        base_parts = source_parts[:-dots] if dots > 0 else source_parts
        if rel_path:
            target_parts = base_parts + rel_path.split(".")
        else:
            target_parts = base_parts
        
        target_module = ".".join(target_parts)
        
        # Check if target exists
        if target_module not in self.modules:
            # Also check if it's a package
            package_init = target_module + ".__init__"
            if package_init not in self.modules:
                self.issues.append(ImportIssue(
                    file_path=source_module,
                    line_number=line_no,
                    import_statement=import_stmt,
                    issue_type="module_not_found",
                    details=f"Relative import target '{target_module}' not found"
                ))
                return
            else:
                target_module = package_init
        
        # Check if imported names exist in target
        target_exports = self.modules.get(target_module, ModuleInfo("")).exports
        for name in imported_names:
            if name not in target_exports:
                # Could be a submodule
                submodule = f"{target_module}.{name}"
                if submodule not in self.modules:
                    self.issues.append(ImportIssue(
                        file_path=source_module,
                        line_number=line_no,
                        import_statement=import_stmt,
                        issue_type="class_not_found",
                        details=f"'{name}' not found in '{target_module}'"
                    ))
    
    def _check_absolute_import(self, source_module: str, line_no: int, import_stmt: str,
                                module_path: str, imported_names: List[str]):
        """Check absolute imports from knowledge_engine or other modules."""
        # Check if it's importing from knowledge_engine
        if module_path.startswith("knowledge_engine"):
            if module_path not in self.modules:
                # Check if it's a package
                package_init = module_path + ".__init__"
                if package_init in self.modules:
                    target_module = package_init
                else:
                    self.issues.append(ImportIssue(
                        file_path=source_module,
                        line_number=line_no,
                        import_statement=import_stmt,
                        issue_type="module_not_found",
                        details=f"Module '{module_path}' not found in knowledge_engine"
                    ))
                    return
            else:
                target_module = module_path
            
            # Check if imported names exist
            target_exports = self.modules.get(target_module, ModuleInfo("")).exports
            for name in imported_names:
                if name == "*":
                    continue  # Wildcard imports are hard to validate
                if name not in target_exports:
                    # Could be a submodule
                    submodule = f"{target_module}.{name}"
                    if submodule not in self.modules:
                        self.issues.append(ImportIssue(
                            file_path=source_module,
                            line_number=line_no,
                            import_statement=import_stmt,
                            issue_type="class_not_found",
                            details=f"'{name}' not exported from '{target_module}'"
                        ))
        
        # For non-knowledge_engine imports, we can't easily validate without running
    
    def _check_module_import(self, source_module: str, line_no: int, import_stmt: str, module_path: str):
        """Check 'import X' or 'import X.Y' statements."""
        if module_path.startswith("knowledge_engine"):
            if module_path not in self.modules:
                # Check for package
                parts = module_path.split(".")
                for i in range(len(parts), 0, -1):
                    partial = ".".join(parts[:i])
                    if partial in self.modules:
                        return
                    if f"{partial}.__init__" in self.modules:
                        return
                
                self.issues.append(ImportIssue(
                    file_path=source_module,
                    line_number=line_no,
                    import_statement=import_stmt,
                    issue_type="module_not_found",
                    details=f"Module '{module_path}' not found"
                ))
    
    def check_circular_imports(self):
        """Detect circular import dependencies."""
        print("Checking for circular imports...")
        
        # Build import graph
        graph = defaultdict(set)
        
        for module_name, module_info in self.modules.items():
            for line_no, import_stmt, import_type in module_info.imports:
                if import_stmt.startswith("from "):
                    parts = import_stmt[5:].split(" import ")
                    if len(parts) == 2:
                        module_path = parts[0].strip()
                        if module_path.startswith("."):
                            # Resolve relative to absolute
                            dots = len(module_path) - len(module_path.lstrip("."))
                            source_parts = module_name.split(".")
                            rel = module_path.lstrip(".")
                            if dots <= len(source_parts):
                                base = source_parts[:-dots] if dots > 0 else source_parts
                                if rel:
                                    target = ".".join(base + rel.split("."))
                                else:
                                    target = ".".join(base)
                                graph[module_name].add(target)
                        elif module_path.startswith("knowledge_engine"):
                            graph[module_name].add(module_path)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle back to current path
                if node in path:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    return cycle
                return None
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                result = dfs(neighbor, path)
                if result:
                    return result
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        for module in graph:
            if module not in visited:
                cycle = dfs(module, [])
                if cycle:
                    cycles.append(cycle)
                    self.issues.append(ImportIssue(
                        file_path=cycle[0],
                        line_number=0,
                        import_statement="",
                        issue_type="circular_import",
                        details=f"Circular import detected: {' -> '.join(cycle)}"
                    ))
        
        # Reset for additional cycles
        visited.clear()
        rec_stack.clear()
        for module in list(graph.keys()):
            if module not in visited:
                dfs(module, [])
    
    def generate_report(self) -> str:
        """Generate a detailed report of all issues."""
        report = []
        report.append("=" * 80)
        report.append("KNOWLEDGE_ENGINE IMPORT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total modules scanned: {len(self.modules)}")
        report.append(f"Total issues found: {len(self.issues)}")
        report.append("")
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append(issue)
        
        for issue_type, issues in sorted(issues_by_type.items()):
            report.append("-" * 80)
            report.append(f"{issue_type.upper().replace('_', ' ')} ({len(issues)} issues)")
            report.append("-" * 80)
            
            # Sort by file path
            for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
                report.append(f"\nFile: {issue.file_path}")
                report.append(f"Line: {issue.line_number}")
                if issue.import_statement:
                    report.append(f"Import: {issue.import_statement}")
                report.append(f"Issue: {issue.details}")
            report.append("")
        
        return "\n".join(report)


def main():
    analyzer = ImportAnalyzer(os.getcwd())
    analyzer.scan_all_modules()
    analyzer.check_all_imports()
    analyzer.check_circular_imports()
    
    report = analyzer.generate_report()
    print(report)
    
    # Save report to file
    output_file = "knowledge_engine_import_issues.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_file}")
    
    return len(analyzer.issues)


if __name__ == "__main__":
    sys.exit(main())
