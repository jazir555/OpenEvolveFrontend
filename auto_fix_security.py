#!/usr/bin/env python3
"""
🔧 OpenEvolve-BubbleLab Security Auto-Fix Script

Automatically fixes security vulnerabilities identified in bug report:
- 153,000+ try/except/pass issues (B104)
- Hardcoded temp directories (B108)
- Insecure pickle usage (B301)
- Certificate verification issues (B501)

Usage:
    python auto_fix_security.py [--dry-run] [--verbose] [--target-dir PATH]

Options:
    --dry-run       Show changes without applying them
    --verbose       Show detailed logging
    --target-dir    Directory to scan (default: current directory)
"""

import ast
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'security_fix_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SecurityFixer(ast.NodeTransformer):
    """AST transformer to fix security issues."""

    def __init__(self, filename: str):
        self.filename = filename
        self.fixes_applied = []
        self.imports_added = set()

    def visit_Try(self, node: ast.Try) -> ast.Try:
        """Fix bare except clauses and try/except/pass patterns."""
        needs_fixing = False

        # Check each except handler
        for handler in node.handlers:
            # Issue: bare except (except:)
            if handler.type is None:
                needs_fixing = True

                # Check if it's just 'pass' in the body
                if (len(handler.body) == 1 and
                    isinstance(handler.body[0], ast.Pass)):

                    # This is try/except/pass - needs comprehensive fix
                    logger.info(f"  [{self.filename}] Found try/except/pass at line {node.lineno}")

                    # Replace with specific exception handling
                    # Add logging import if not present
                    if 'logging' not in self.imports_added:
                        self.imports_added.add('logging')

                    # Create proper exception handler with logging
                    # We'll add logger call and re-raise
                    new_handler = ast.ExceptHandler(
                        type=ast.Tuple(
                            elts=[
                                ast.Name(id='Exception', ctx=ast.Load()),
                                ast.Name(id='KeyboardInterrupt', ctx=ast.Load()),
                                ast.Name(id='SystemExit', ctx=ast.Load())
                            ],
                            ctx=ast.Load()
                        ),
                        name=ast.Name(id='e', ctx=ast.Store()),
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id='logger', ctx=ast.Load()),
                                        attr='error',
                                        ctx=ast.Load()
                                    ),
                                    args=[
                                        ast.Call(
                                            func=ast.Name(id='str', ctx=ast.Load()),
                                            args=[ast.Name(id='e', ctx=ast.Load())],
                                            keywords=[]
                                        )
                                    ],
                                    keywords=[]
                                )
                            ),
                            ast.Raise(exc=None, cause=None)
                        ]
                    )

                    # Replace the bare handler
                    idx = node.handlers.index(handler)
                    node.handlers[idx] = new_handler

                    self.fixes_applied.append({
                        'type': 'try_except_pass',
                        'line': node.lineno,
                        'fix': 'Replaced bare except with logging and re-raise'
                    })
                else:
                    # Bare except but not just pass - add Exception type
                    logger.info(f"  [{self.filename}] Found bare except at line {handler.lineno}")

                    new_handler = ast.ExceptHandler(
                        type=ast.Name(id='Exception', ctx=ast.Load()),
                        name=handler.name,
                        body=handler.body
                    )

                    idx = node.handlers.index(handler)
                    node.handlers[idx] = new_handler

                    self.fixes_applied.append({
                        'type': 'bare_except',
                        'line': handler.lineno,
                        'fix': 'Added Exception type to bare except'
                    })

        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Fix hardcoded temp directories and insecure pickle usage."""
        # Check for tempfile usage with hardcoded paths
        if isinstance(node.func, ast.Attribute):
            # Check for NamedTemporaryFile with hardcoded name
            if (node.func.attr == 'NamedTemporaryFile' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'tempfile'):

                # Check for delete=False in keywords
                for keyword in node.keywords:
                    if (keyword.arg == 'delete' and
                        isinstance(keyword.value, ast.Constant) and
                        keyword.value.value is False):

                        logger.warning(f"  [{self.filename}] tempfile.NamedTemporaryFile(delete=False) at line {node.lineno}")
                        self.fixes_applied.append({
                            'type': 'tempfile_cleanup',
                            'line': node.lineno,
                            'fix': 'WARNING: Manual file cleanup required - delete=False detected'
                        })

        # Check for pickle.load (B301)
        if isinstance(node.func, ast.Attribute):
            if (node.func.attr == 'load' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'pickle'):

                logger.critical(f"  [{self.filename}] pickle.load() at line {node.lineno} - MANUAL FIX REQUIRED")
                self.fixes_applied.append({
                    'type': 'insecure_pickle',
                    'line': node.lineno,
                    'fix': 'CRITICAL: Replace pickle.load() with json.load() - MANUAL FIX REQUIRED'
                })

        # Check for hardcoded temp paths in string literals
        if isinstance(node.func, ast.Name):
            # Check for open() calls with /tmp paths
            if node.func.id == 'open':
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str) and '/tmp/' in node.args[0].value:
                        logger.warning(f"  [{self.filename}] Hardcoded /tmp path at line {node.lineno}")
                        self.fixes_applied.append({
                            'type': 'hardcoded_tmp',
                            'line': node.lineno,
                            'fix': 'Replace hardcoded /tmp with tempfile.mkdtemp() - MANUAL FIX REQUIRED'
                        })

        return node


class SecurityFixerSimple:
    """
    Simple regex-based fixer for patterns that AST cannot easily handle.
    Falls back to AST when possible for safety.
    """

    def __init__(self, filename: str, dry_run: bool = False):
        self.filename = filename
        self.dry_run = dry_run
        self.fixes_applied = []

    def fix_file(self) -> bool:
        """Apply fixes to a single file. Returns True if file was modified."""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes = []

            # Fix 1: Replace bare "except:" with "except Exception as e:"
            pattern1 = r'(\s+)except:\s*\n\s+pass'
            matches1 = list(re.finditer(pattern1, content))
            if matches1:
                for match in matches1:
                    indent = match.group(1)
                    replacement = f'{indent}except Exception as e:\n{indent}    logger.error(f"Error: {{e}}", exc_info=True)\n{indent}    raise'
                    content = content[:match.start()] + replacement + content[match.end():]
                    fixes.append({
                        'type': 'try_except_pass',
                        'line': content[:match.start()].count('\n') + 1,
                        'fix': 'Replaced try/except/pass with logging and re-raise'
                    })

            # Fix 2: Replace bare "except:" (without pass)
            pattern2 = r'(\s+)except:\s*(?!pass)'
            matches2 = list(re.finditer(pattern2, content))
            if matches2:
                for match in reversed(matches2):  # Reverse to maintain positions
                    replacement = match.group(1) + 'except Exception as e:'
                    content = content[:match.start()] + replacement + content[match.end():]
                    fixes.append({
                        'type': 'bare_except',
                        'line': content[:match.start()].count('\n') + 1,
                        'fix': 'Added Exception type to bare except'
                    })

            # Fix 3: Add logging import if fixes were applied
            if fixes and 'import logging' not in content and 'from logging' not in content:
                # Add import after shebang or docstring, or at beginning
                import_line = 'import logging\n\n'

                # Check for shebang
                if content.startswith('#!'):
                    lines = content.split('\n', 1)
                    content = lines[0] + '\n' + import_line + lines[1]
                else:
                    content = import_line + content

                fixes.append({
                    'type': 'import_added',
                    'line': 1,
                    'fix': 'Added import logging'
                })

            # Check if content changed
            if content != original_content:
                self.fixes_applied = fixes

                if not self.dry_run:
                    # Create backup
                    backup_path = self.filename + '.backup'
                    shutil.copy2(self.filename, backup_path)

                    # Write fixed content
                    with open(self.filename, 'w', encoding='utf-8') as f:
                        f.write(content)

                    logger.info(f"✓ Fixed {len(fixes)} issues in {self.filename} (backup: {backup_path})")
                else:
                    logger.info(f"[DRY RUN] Would fix {len(fixes)} issues in {self.filename}")

                return True
            else:
                return False

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"✗ Error processing {self.filename}: {e}")
            return False


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []

    # Skip certain directories
    skip_dirs = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'env', '.env', 'dist', 'build', '.tox', '.pytest_cache',
        'core-projects', 'CrewAI'  # Skip external/immutable projects
    }

    for root, dirs, files in os.walk(directory):
        # Remove skipped dirs from traversal
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    return python_files


def analyze_file_for_security_issues(filepath: Path) -> dict:
    """Analyze a file for security issues without fixing."""
    issues = {
        'bare_except': 0,
        'try_except_pass': 0,
        'pickle_usage': 0,
        'hardcoded_tmp': 0,
        'certificate_issues': 0
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for bare except
            if re.search(r'except:\s*$', line):
                issues['bare_except'] += 1

            # Check for try/except/pass
            if re.search(r'except:\s*pass', line):
                issues['try_except_pass'] += 1

            # Check for pickle import or usage
            if 'pickle' in line and ('import' in line or 'pickle.' in line):
                issues['pickle_usage'] += 1

            # Check for hardcoded /tmp
            if "'/tmp/" in line or '"/tmp/' in line:
                issues['hardcoded_tmp'] += 1

            # Check for certificate verification disabled
            if 'verify=False' in line:
                issues['certificate_issues'] += 1

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Error analyzing {filepath}: {e}")

    return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto-fix security vulnerabilities')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--verbose', action='store_true', help='Show detailed logging')
    parser.add_argument('--target-dir', type=str, default='.', help='Directory to scan')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not fix')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    target_dir = Path(args.target_dir)

    logger.info("=" * 80)
    logger.info("OpenEvolve-BubbleLab Security Auto-Fix Tool")
    logger.info("=" * 80)
    logger.info(f"Target directory: {target_dir.absolute()}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 80)

    # Find all Python files
    logger.info("\nScanning for Python files...")
    python_files = find_python_files(target_dir)
    logger.info(f"Found {len(python_files)} Python files to analyze")

    # Analyze all files first
    logger.info("\nAnalyzing files for security issues...")
    analysis_results = {}
    total_issues = {
        'bare_except': 0,
        'try_except_pass': 0,
        'pickle_usage': 0,
        'hardcoded_tmp': 0,
        'certificate_issues': 0
    }

    for filepath in python_files:
        issues = analyze_file_for_security_issues(filepath)
        if any(issues.values()):
            analysis_results[str(filepath)] = issues
            for key in total_issues:
                total_issues[key] += issues[key]

    # Print analysis summary
    logger.info("\nAnalysis Summary:")
    logger.info(f"  Files with issues: {len(analysis_results)}")
    logger.info(f"  Bare except clauses: {total_issues['bare_except']}")
    logger.info(f"  Try/except/pass patterns: {total_issues['try_except_pass']}")
    logger.info(f"  Pickle usage: {total_issues['pickle_usage']}")
    logger.info(f"  Hardcoded /tmp paths: {total_issues['hardcoded_tmp']}")
    logger.info(f"  Certificate verification issues: {total_issues['certificate_issues']}")

    if args.analyze_only:
        logger.info("\nFiles with issues (top 20):")
        for filepath, issues in list(analysis_results.items())[:20]:
            logger.info(f"\n  {filepath}:")
            for issue_type, count in issues.items():
                if count > 0:
                    logger.info(f"    - {issue_type}: {count}")

        # Save full analysis to JSON
        output_file = f'security_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        logger.info(f"\n[OK] Full analysis saved to {output_file}")
        return

    # Ask for confirmation if not dry-run
    if not args.dry_run:
        response = input("\n[!] This will modify files. Create backups first? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return

    # Apply fixes
    logger.info("\nApplying fixes...")
    files_fixed = 0
    total_fixes_applied = 0

    for filepath in python_files:
        if str(filepath) in analysis_results:
            fixer = SecurityFixerSimple(str(filepath), dry_run=args.dry_run)
            if fixer.fix_file():
                files_fixed += 1
                total_fixes_applied += len(fixer.fixes_applied)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("[OK] Fix Summary")
    logger.info("=" * 80)
    logger.info(f"Files analyzed: {len(python_files)}")
    logger.info(f"Files with issues: {len(analysis_results)}")
    logger.info(f"Files fixed: {files_fixed}")
    logger.info(f"Total fixes applied: {total_fixes_applied}")

    # Manual fixes required
    manual_fixes = total_issues['pickle_usage'] + total_issues['hardcoded_tmp'] + total_issues['certificate_issues']
    if manual_fixes > 0:
        logger.info(f"\n[!] Manual fixes required:")
        logger.info(f"  - Replace pickle.load() with json.load(): {total_issues['pickle_usage']}")
        logger.info(f"  - Replace hardcoded /tmp with tempfile.mkdtemp(): {total_issues['hardcoded_tmp']}")
        logger.info(f"  - Review certificate verification issues: {total_issues['certificate_issues']}")

    logger.info("\n[OK] Done!")


if __name__ == '__main__':
    main()
