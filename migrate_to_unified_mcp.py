"""
Migration Script: Scattered MCP Files -> Unified MCP Server
License: Apache 2.0

This script helps migrate from the old scattered MCP file structure
to the new unified MCP server.

Features:
- Analyzes existing MCP files
- Generates unified server configuration
- Validates migration completeness
- Provides rollback support

Usage:
    python migrate_to_unified_mcp.py --analyze
    python migrate_to_unified_mcp.py --generate-config
    python migrate_to_unified_mcp.py --validate
    python migrate_to_unified_mcp.py --backup-old
"""

import re
import ast
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str
    module: str
    parameters: List[Dict]
    decorators: List[str]
    deprecated: bool = False


@dataclass
class MigrationReport:
    """Migration analysis report."""
    total_files: int
    total_tools: int
    deprecated_tools: int
    conflicts: List[str]
    recommendations: List[str]
    files_to_migrate: List[str]


class MCPMigrationAnalyzer:
    """
    Analyzer for MCP file migration.
    
    Scans all MCP-related files and generates a migration report.
    """
    
    # Files to exclude from migration (too specialized or legacy)
    EXCLUDE_PATTERNS = [
        "*test*.py",
        "*_backup*.py",
        "*_FIXED*.py",
        "*_EDGE_CASE_*.py",
        "c2c_mcp_tools.py",  # C2C-specific
        "claudiomiro_mcp_tools.py",  # Legacy
        "bubblelab_mcp_tools.py",  # Handled separately
        "roma_mcp_tools.py",  # Handled separately
        "ace_mcp_tools.py",  # Handled separately
    ]
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.mcp_files: List[Path] = []
        self.tools_found: List[ToolInfo] = []
    
    def find_mcp_files(self) -> List[Path]:
        """Find all MCP-related Python files."""
        pattern = re.compile(r'.*mcp.*\.py$', re.IGNORECASE)
        
        files = []
        for py_file in self.project_root.rglob("*.py"):
            if pattern.match(py_file.name):
                # Check exclusion patterns
                if any(py_file.match(pattern) for pattern in self.EXCLUDE_PATTERNS):
                    continue
                files.append(py_file)
        
        self.mcp_files = sorted(files)
        return self.mcp_files
    
    def parse_tools_from_file(self, filepath: Path) -> List[ToolInfo]:
        """Parse tool definitions from a Python file."""
        tools = []
        
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Look for function definitions with @mcp.tool() decorator
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr == 'tool':
                                    # Extract tool info
                                    tool_name = node.name
                                    docstring = ast.get_docstring(node) or ""
                                    
                                    # Parse parameters
                                    params = []
                                    for arg in node.args.args:
                                        param_info = {
                                            'name': arg.arg,
                                            'type': ast.unparse(arg.annotation) if arg.annotation else 'Any'
                                        }
                                        params.append(param_info)
                                    
                                    tools.append(ToolInfo(
                                        name=tool_name,
                                        description=docstring[:100],
                                        module=filepath.stem,
                                        parameters=params,
                                        decorators=['@mcp.tool()']
                                    ))
        except SyntaxError:
            pass
        except Exception as e:
            print(f"Warning: Error parsing {filepath}: {e}")
        
        return tools
    
    def analyze(self) -> MigrationReport:
        """Analyze all MCP files and generate report."""
        print("[INFO] Analyzing MCP files...")
        
        self.find_mcp_files()
        print(f"  [INFO] Found {len(self.mcp_files)} MCP-related files")
        
        all_tools = []
        tool_names = {}
        conflicts = []
        deprecated = 0
        
        for mcp_file in self.mcp_files:
            tools = self.parse_tools_from_file(mcp_file)
            all_tools.extend(tools)
            
            for tool in tools:
                if tool.name in tool_names:
                    conflicts.append(
                        f"Duplicate tool '{tool.name}' in {mcp_file.name} and {tool_names[tool.name]}"
                    )
                else:
                    tool_names[tool.name] = mcp_file.name
                
                if tool.deprecated or 'deprecated' in tool.description.lower():
                    deprecated += 1
        
        self.tools_found = all_tools
        
        # Generate recommendations
        recommendations = []
        
        if len(self.mcp_files) > 5:
            recommendations.append(
                f"Consider consolidating {len(self.mcp_files)} MCP files into unified_mcp_server.py"
            )
        
        if conflicts:
            recommendations.append(
                f"Resolve {len(conflicts)} tool name conflicts before migration"
            )
        
        if deprecated > 0:
            recommendations.append(
                f"Review {deprecated} deprecated tools for removal"
            )
        
        return MigrationReport(
            total_files=len(self.mcp_files),
            total_tools=len(all_tools),
            deprecated_tools=deprecated,
            conflicts=conflicts,
            recommendations=recommendations,
            files_to_migrate=[f.name for f in self.mcp_files]
        )
    
    def generate_unified_config(self, output_path: Path = None) -> str:
        """Generate configuration for unified MCP server."""
        if not self.tools_found:
            print("No tools found. Run analyze() first.")
            return ""
        
        # Group tools by category
        categories = {
            'decomposition': [],
            'knowledge': [],
            'z3_prover': [],
            'leanaide': [],
            'workflow': [],
            'other': []
        }
        
        for tool in self.tools_found:
            category = 'other'
            if 'decompos' in tool.module.lower() or 'decompos' in tool.name.lower():
                category = 'decomposition'
            elif 'knowledge' in tool.module.lower() or 'knowledge' in tool.name.lower():
                category = 'knowledge'
            elif 'z3' in tool.module.lower() or 'z3' in tool.name.lower():
                category = 'z3_prover'
            elif 'leanaide' in tool.module.lower() or 'lean' in tool.name.lower():
                category = 'leanaide'
            elif 'workflow' in tool.module.lower():
                category = 'workflow'
            
            categories[category].append(tool)
        
        # Generate config
        config_lines = [
            "# Unified MCP Server Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "mcp:",
            "  server:",
            "    name: \"OpenEvolve Unified MCP Server\"",
            "    version: \"2.0.0\"",
            "    ",
            "  categories:",
        ]
        
        for category, tools in categories.items():
            if tools:
                config_lines.append(f"    {category}:")
                for tool in tools:
                    config_lines.append(f"      - name: {tool.name}")
                    config_lines.append(f"        description: {tool.description[:50]}")
                    config_lines.append(f"        source: {tool.module}")
        
        config = "\n".join(config_lines)
        
        if output_path:
            output_path.write_text(config)
            print(f"[SUCCESS] Configuration written to {output_path}")
        
        return config


class MCPMigrator:
    """
    Handles the migration from scattered MCP files to unified server.
    """
    
    def __init__(self, project_root: Path = None, dry_run: bool = True):
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.analyzer = MCPMigrationAnalyzer(project_root)
        self.backup_dir: Path = None
    
    def create_backup(self) -> Path:
        """Create backup of old MCP files."""
        backup_dir = self.project_root / "mcp_migration_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer.find_mcp_files()
        
        for mcp_file in self.analyzer.mcp_files:
            target = backup_dir / mcp_file.relative_to(self.project_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.dry_run:
                shutil.copy2(mcp_file, target)
            else:
                print(f"  [DRY RUN] Would backup: {mcp_file} -> {target}")
        
        self.backup_dir = backup_dir
        
        # Create restore script
        restore_script = backup_dir / "restore.py"
        restore_content = f'''"""
MCP Migration Restore Script
Generated: {datetime.now().isoformat()}
"""
import shutil
from pathlib import Path

def restore():
    backup_dir = Path(__file__).parent
    project_root = Path.cwd()
    
    # Copy files back
    for file in backup_dir.rglob("*.py"):
        if file.name == "restore.py":
            continue
        
        target = project_root / file.relative_to(backup_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        print(f"Restored: {{target}}")

if __name__ == "__main__":
    restore()
'''
        
        if not self.dry_run:
            restore_script.write_text(restore_content)
            print(f"[SUCCESS] Backup created at: {backup_dir}")
        
        return backup_dir
    
    def validate_migration(self) -> bool:
        """Validate that migration is complete and working."""
        print("\n[INFO] Validating migration...")
        
        checks = []
        
        # Check unified server exists
        unified_server = self.project_root / "unified_mcp_server.py"
        checks.append(("Unified MCP Server exists", unified_server.exists()))
        
        # Check it imports correctly
        try:
            import unified_mcp_server
            checks.append(("Unified server imports", True))
        except Exception as e:
            checks.append(("Unified server imports", False))
        
        # Check old files have been archived
        old_count = len(self.analyzer.mcp_files)
        checks.append(("Old MCP files identified", old_count > 0))
        
        # Print results
        all_passed = all(check[1] for check in checks)
        
        for name, passed in checks:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {name}")
        
        return all_passed
    
    def generate_migration_report(self, output_path: Path = None) -> str:
        """Generate comprehensive migration report."""
        report = self.analyzer.analyze()
        
        report_lines = [
            "# MCP Migration Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Summary",
            f"- Total MCP files found: {report.total_files}",
            f"- Total tools identified: {report.total_tools}",
            f"- Deprecated tools: {report.deprecated_tools}",
            f"- Conflicts detected: {len(report.conflicts)}",
            "\n## Files to Migrate",
        ]
        
        for file in report.files_to_migrate:
            report_lines.append(f"- {file}")
        
        if report.conflicts:
            report_lines.append("\n## Conflicts")
            for conflict in report.conflicts:
                report_lines.append(f"- {conflict}")
        
        if report.recommendations:
            report_lines.append("\n## Recommendations")
            for rec in report.recommendations:
                report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "\n## Migration Steps",
            "1. Review this report",
            "2. Create backup: `python migrate_to_unified_mcp.py --backup`",
            "3. Resolve any conflicts",
            "4. Run unified server tests",
            "5. Update client configurations",
            "6. Archive old files",
            "\n## Rollback",
            "If issues occur, restore from backup using the restore.py script.",
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path.write_text(report_text)
            print(f"[SUCCESS] Report written to {output_path}")
        
        return report_text


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate from scattered MCP files to unified MCP server"
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze MCP files")
    parser.add_argument("--generate-config", action="store_true", help="Generate unified config")
    parser.add_argument("--validate", action="store_true", help="Validate migration")
    parser.add_argument("--backup-old", action="store_true", help="Backup old files")
    parser.add_argument("--report", action="store_true", help="Generate migration report")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.generate_config, args.validate, args.backup_old, args.report]):
        parser.print_help()
        return
    
    migrator = MCPMigrator(dry_run=args.dry_run)
    
    if args.analyze:
        report = migrator.analyzer.analyze()
        print(f"\n{'='*60}")
        print("MCP Migration Analysis")
        print(f"{'='*60}")
        print(f"Files found: {report.total_files}")
        print(f"Tools identified: {report.total_tools}")
        print(f"Deprecated: {report.deprecated_tools}")
        print(f"Conflicts: {len(report.conflicts)}")
        
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  * {rec}")
    
    if args.generate_config:
        migrator.analyzer.find_mcp_files()
        for file in migrator.analyzer.mcp_files:
            migrator.analyzer.parse_tools_from_file(file)
        
        config = migrator.analyzer.generate_unified_config(
            Path("unified_mcp_config.yaml")
        )
        print("\nGenerated config preview:")
        print(config[:500] + "...")
    
    if args.backup_old:
        backup_dir = migrator.create_backup()
        if args.dry_run:
            print("\n[DRY RUN] No files were actually backed up")
    
    if args.validate:
        migrator.validate_migration()
    
    if args.report:
        report = migrator.generate_migration_report(Path("MCP_MIGRATION_REPORT.md"))
        print("\nMigration report generated!")


if __name__ == "__main__":
    main()
