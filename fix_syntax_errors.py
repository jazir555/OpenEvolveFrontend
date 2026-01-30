#!/usr/bin/env python3
"""
Syntax Error Fixer - Top Level Files

Fixes the 12 syntax errors identified in top-level Python files.

Usage:
    python fix_syntax_errors.py [--dry-run]
"""

__all__ = ['SYNTAX_ERROR_FILES', 'SyntaxErrorFixer', 'fix_all_syntax_errors']

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'syntax_fix_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Files with syntax errors and their specific issues
SYNTAX_ERROR_FILES = [
    'ace_mcp_tools_FIXED.py',
    'adversarial_adapter.py',
    'adversarial_error_handling.py',
    'bubblelabs_evolution_integration.py',
    'demo_mcts_mdap.py',
    'hybrid_error_handling.py',
    'leanaide_mdap_demo.py',
    'leanaide_sop_integration.py',
    'openevolve_leanaide_bridge.py',
    'simple_verify_implementation.py',
    'sovereign_gauntlets.py',
    'workflow_stage_functions.py'
]


class SyntaxErrorFixer:
    """Fix syntax errors in Python files."""

    def __init__(self, filepath: Path, dry_run: bool = False):
        self.filepath = filepath
        self.dry_run = dry_run
        self.fixes_applied = []

    def fix(self) -> bool:
        """Attempt to fix syntax errors."""
        try:
            content = self._read_source_file()
        except (IOError, OSError) as e:
            logger.error(f"  [ERROR] Cannot read {self.filepath.name}: {e}")
            return False

        original_content = content
        filename = self.filepath.name
        logger.info(f"\n[*] Processing: {filename}")

        # Check if file has syntax errors
        syntax_error = self._check_syntax_error(content)
        if not syntax_error:
            logger.info(f"  [OK] No syntax errors found")
            return False

        logger.info(f"  [ERROR] Syntax error at line {syntax_error.lineno}: {syntax_error.msg}")
        logger.info(f"     Text: {syntax_error.text.strip() if syntax_error.text else 'N/A'}")

        # Apply specific fix based on filename
        content = self._apply_fix_by_filename(filename, content)

        # Verify the fix worked
        if self._check_syntax_error(content):
            logger.info(f"  [FAIL] STILL BROKEN: {syntax_error.msg} at line {syntax_error.lineno}")
            return False

        logger.info(f"  [FIXED] File now compiles successfully")
        self.fixes_applied.append("Syntax error fixed")

        if content == original_content:
            logger.info(f"  [INFO] No changes made")
            return False

        return self._save_fixed_content(content)

    def _read_source_file(self) -> str:
        """Read the source file content."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _check_syntax_error(self, content: str) -> Optional[SyntaxError]:
        """Check if content has syntax errors. Returns the error or None."""
        try:
            compile(content, str(self.filepath), 'exec')
            return None
        except SyntaxError as e:
            return e

    def _apply_fix_by_filename(self, filename: str, content: str) -> str:
        """Apply the appropriate fix based on filename."""
        fix_methods = {
            'demo_mcts_mdap.py': self._fix_demo_mcts_mdap,
            'leanaide_mdap_demo.py': self._fix_leanaide_mdap_demo,
            'workflow_stage_functions.py': self._fix_workflow_stage_functions,
            'adversarial_adapter.py': self._fix_adversarial_adapter,
            'bubblelabs_evolution_integration.py': self._fix_bubblelabs_evolution_integration,
            'simple_verify_implementation.py': self._fix_simple_verify_implementation,
            'adversarial_error_handling.py': self._fix_adversarial_error_handling,
            'hybrid_error_handling.py': self._fix_hybrid_error_handling,
            'sovereign_gauntlets.py': self._fix_sovereign_gauntlets,
        }

        if filename in fix_methods:
            return fix_methods[filename](content)

        if filename in ['ace_mcp_tools_FIXED.py', 'leanaide_sop_integration.py',
                        'openevolve_leanaide_bridge.py']:
            return self._fix_generic_invalid_syntax(content)

        return content

    def _save_fixed_content(self, content: str) -> bool:
        """Save the fixed content with backup handling."""
        backup_path = str(self.filepath) + '.backup'

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would save changes")
            return True

        try:
            shutil.copy2(self.filepath, backup_path)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"  [SAVED] (backup: {backup_path})")
            return True
        except (IOError, OSError, shutil.Error) as e:
            self._cleanup_backup(backup_path)
            logger.error(f"  [ERROR] Failed to save: {e}")
            return False

    def _cleanup_backup(self, backup_path: str) -> None:
        """Clean up backup file if it exists."""
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
                logger.info(f"  [INFO] Cleaned up backup file after error")
            except OSError:
                pass

    def _fix_demo_mcts_mdap(self, content: str) -> str:
        """Fix f-string with backslash in demo_mcts_mdap.py line 604."""
        logger.info("  Fixing: f-string with backslash")

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'f"' in line and '\n' in line and 't' in line:
                # f-string with \n needs to use double backslash or different approach
                # Pattern: f"text\n" should be f"text\\n" or use separate string
                if '\n' in line and line.count('\n') % 2 == 1:
                    # Escape the backslash
                    lines[i] = line.replace('\n', r'\n')
                    logger.info(f"    Line {i+1}: Escaped backslash in f-string")
                    self.fixes_applied.append(f"Line {i+1}: Fixed f-string backslash")

        return '\n'.join(lines)

    def _fix_leanaide_mdap_demo(self, content: str) -> str:
        """Fix unterminated string literal in leanaide_mdap_demo.py line 44."""
        logger.info("  Fixing: Unterminated string literal")

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for line with unterminated string around line 44
            stripped = line.strip()
            if i < 60:  # Search in first 60 lines
                # Check for odd number of quotes
                if stripped.count('"') % 2 != 0 or stripped.count("'") % 2 != 0:
                    # Add missing quote at end
                    if stripped.startswith('"') and not stripped.endswith('"'):
                        lines[i] = line + '"'
                        logger.info(f"    Line {i+1}: Added closing double quote")
                        self.fixes_applied.append(f"Line {i+1}: Fixed unterminated string")
                    elif stripped.startswith("'") and not stripped.endswith("'"):
                        lines[i] = line + "'"
                        logger.info(f"    Line {i+1}: Added closing single quote")
                        self.fixes_applied.append(f"Line {i+1}: Fixed unterminated string")

        return '\n'.join(lines)

    def _fix_workflow_stage_functions(self, content: str) -> str:
        """Fix unterminated string literal in workflow_stage_functions.py line 90."""
        logger.info("  Fixing: Unterminated string literal")

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if i < 120:  # Search in first 120 lines
                # Check for triple-quoted strings or multiline strings
                if '"""' in line or "'''" in line:
                    # Count quotes - should be even for complete strings
                    triple_dbl = line.count('"""')
                    triple_sgl = line.count("'''")
                    if triple_dbl % 2 != 0 or triple_sgl % 2 != 0:
                        # Look ahead to see if we can find the closing quotes
                        found_close = False
                        for j in range(i+1, min(i+10, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                found_close = True
                                break
                        if not found_close:
                            # Add closing quotes
                            if '"""' in line:
                                lines[i] = line + '"""'
                            else:
                                lines[i] = line + "'''"
                            logger.info(f"    Line {i+1}: Added closing triple quotes")
                            self.fixes_applied.append(f"Line {i+1}: Fixed unterminated string")

        return '\n'.join(lines)

    def _fix_adversarial_adapter(self, content: str) -> str:
        """Fix missing except/finally block in adversarial_adapter.py line 355."""
        logger.info("  Fixing: Missing except/finally block")

        lines = content.split('\n')
        in_try = False
        try_indent = 0
        try_line = 0

        for i, line in enumerate(lines):
            if re.match(r'^(\s*)try:\s*$', line):
                in_try = True
                try_indent = len(line) - len(line.lstrip())
                try_line = i
            elif in_try:
                # Check if next non-empty line has less or equal indentation
                if line.strip() and not line.startswith('#'):
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= try_indent:
                        # Block ended without except - add it
                        indent = ' ' * try_indent
                        lines.insert(i, indent + 'except Exception as e:')
                        lines.insert(i + 1, indent + '    ' + f'# TODO: Handle exception properly')
                        lines.insert(i + 2, indent + '    raise')
                        logger.info(f"    Line {try_line+1}: Added missing except block at line {i}")
                        self.fixes_applied.append(f"Line {try_line+1}: Added except block")
                        in_try = False
                        break
                    elif 'except' in line or 'finally' in line:
                        in_try = False

        return '\n'.join(lines)

    def _fix_bubblelabs_evolution_integration(self, content: str) -> str:
        """Fix missing except/finally block in bubblelabs_evolution_integration.py line 449."""
        logger.info("  Fixing: Missing except/finally block")

        return self._fix_missing_except_block(content)

    def _fix_simple_verify_implementation(self, content: str) -> str:
        """Fix missing except/finally block in simple_verify_implementation.py line 77."""
        logger.info("  Fixing: Missing except/finally block")

        return self._fix_missing_except_block(content)

    def _fix_missing_except_block(self, content: str) -> str:
        """Generic fix for missing except/finally blocks."""
        lines = content.split('\n')

        for i in range(len(lines)):
            line = lines[i]
            if re.match(r'^(\s*)try:\s*$', line):
                try_indent = len(line) - len(line.lstrip())

                # Look ahead to see if except/finally exists
                found_handler = False
                for j in range(i + 1, min(i + 30, len(lines))):
                    next_line = lines[j]
                    if next_line.strip() == '' or next_line.strip().startswith('#'):
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())

                    # If we're back at try level or less, check for handler
                    if next_indent <= try_indent:
                        if 'except' in next_line or 'finally' in next_line:
                            found_handler = True
                        break

                # If no handler found, add one
                if not found_handler:
                    # Find where to insert (end of try block)
                    insert_pos = i + 1
                    for j in range(i + 1, min(i + 30, len(lines))):
                        line_indent = len(lines[j]) - len(lines[j].lstrip())
                        if line_indent <= try_indent and lines[j].strip():
                            insert_pos = j
                            break

                    indent = ' ' * try_indent
                    lines.insert(insert_pos, indent + 'except Exception as e:')
                    lines.insert(insert_pos + 1, indent + '    ' + 'import logging')
                    lines.insert(insert_pos + 2, indent + '    ' + 'logger = logging.getLogger(__name__)')
                    lines.insert(insert_pos + 3, indent + '    ' + f'logger.error(f"Error: {{e}}", exc_info=True)')
                    lines.insert(insert_pos + 4, indent + '    ' + 'raise')
                    logger.info(f"    Added missing except block at line {insert_pos+1}")
                    self.fixes_applied.append(f"Line {i+1}: Added except block")
                    break

        return '\n'.join(lines)

    def _fix_adversarial_error_handling(self, content: str) -> str:
        """Fix 'await' outside async function in adversarial_error_handling.py line 778."""
        logger.info("  Fixing: await outside async function")

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for await statement
            if re.search(r'^\s*await\s+', line):
                # Check if we're in an async function
                in_async = False
                for j in range(max(0, i - 100), i):
                    if 'async def' in lines[j]:
                        in_async = True
                        break

                if not in_async:
                    # Remove await
                    lines[i] = re.sub(r'^(\s*)await\s+', r'\1', line)
                    logger.info(f"    Line {i+1}: Removed await from non-async context")
                    self.fixes_applied.append(f"Line {i+1}: Removed await")

        return '\n'.join(lines)

    def _fix_hybrid_error_handling(self, content: str) -> str:
        """Fix 'await' outside async function in hybrid_error_handling.py line 297."""
        logger.info("  Fixing: await outside async function")

        return self._fix_await_outside_async(content)

    def _fix_await_outside_async(self, content: str) -> str:
        """Generic fix for await outside async function."""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if re.search(r'^\s*await\s+', line):
                # Check if in async function
                in_async = False
                for j in range(max(0, i - 50), i):
                    if 'async def' in lines[j]:
                        in_async = True
                        break

                if not in_async:
                    # Option 1: Make function async
                    # Option 2: Remove await
                    # We'll remove await for now
                    lines[i] = re.sub(r'^(\s*)await\s+', r'\1', line)
                    logger.info(f"    Line {i+1}: Removed await from non-async function")
                    self.fixes_applied.append(f"Line {i+1}: Removed await")

        return '\n'.join(lines)

    def _fix_sovereign_gauntlets(self, content: str) -> str:
        """Fix expected indented block in sovereign_gauntlets.py line 451."""
        logger.info("  Fixing: Missing indented block after except")

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for except statement without indented block
            if re.match(r'^(\s*)except.*:\s*$', line):
                except_indent = len(line) - len(line.lstrip())

                # Check if next line has proper indentation
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() == '' or next_line.strip().startswith('#'):
                        # Empty line or comment - need to add pass or raise
                        indent = ' ' * (except_indent + 4)
                        lines.insert(i + 1, indent + 'raise  # TODO: Add proper exception handling')
                        logger.info(f"    Line {i+1}: Added raise statement to empty except block")
                        self.fixes_applied.append(f"Line {i+1}: Added except block body")

        return '\n'.join(lines)

    def _fix_generic_invalid_syntax(self, content: str) -> str:
        """Generic fix for invalid syntax."""
        logger.info("  Fixing: Generic invalid syntax")

        # Try common fixes
        lines = content.split('\n')

        # Fix 1: Unmatched parentheses
        for i, line in enumerate(lines):
            # Count parens, brackets, braces
            open_parens = line.count('(') - line.count(')')
            open_brackets = line.count('[') - line.count(']')
            open_braces = line.count('{') - line.count('}')

            if open_parens > 0:
                lines[i] = line + ')' * open_parens
                logger.info(f"    Line {i+1}: Added {open_parens} closing paren(s)")
                self.fixes_applied.append(f"Line {i+1}: Fixed parentheses")
            elif open_brackets > 0:
                lines[i] = line + ']' * open_brackets
                logger.info(f"    Line {i+1}: Added {open_brackets} closing bracket(s)")
                self.fixes_applied.append(f"Line {i+1}: Fixed brackets")
            elif open_braces > 0:
                lines[i] = line + '}' * open_braces
                logger.info(f"    Line {i+1}: Added {open_braces} closing brace(s)")
                self.fixes_applied.append(f"Line {i+1}: Fixed braces")

        return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fix syntax errors in top-level files')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--target-dir', type=str, default='.', help='Directory to scan')

    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()

    logger.info("=" * 80)
    logger.info("Syntax Error Fixer - Top Level Files")
    logger.info("=" * 80)
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 80)

    # Find files with syntax errors
    files_to_fix = []
    for filename in SYNTAX_ERROR_FILES:
        filepath = target_dir / filename
        if filepath.exists():
            files_to_fix.append(filepath)
        else:
            logger.warning(f"[!] File not found: {filename}")

    if not files_to_fix:
        logger.info("\n[!] No files to fix")
        return

    logger.info(f"\n[*] Found {len(files_to_fix)} files with syntax errors")
    logger.info("\nFiles to fix:")
    for f in files_to_fix:
        logger.info(f"  - {f.name}")

    # Fix each file
    logger.info("\n[*] Fixing syntax errors...")
    fixed_count = 0
    failed_count = 0

    for filepath in files_to_fix:
        fixer = SyntaxErrorFixer(filepath, dry_run=args.dry_run)
        if fixer.fix():
            fixed_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("[OK] Fix Complete")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Files processed: {len(files_to_fix)}")
    logger.info(f"  Fixed: {fixed_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Success rate: {fixed_count / len(files_to_fix) * 100:.1f}%")

    if failed_count > 0:
        logger.info("\n[!] Some files could not be fixed automatically")
        logger.info("    These will require manual review")

    logger.info("\n[OK] Done!")


if __name__ == '__main__':
    main()
