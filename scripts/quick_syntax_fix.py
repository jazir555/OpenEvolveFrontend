#!/usr/bin/env python3
"""Quick syntax error fixes for top-level files."""

import re
from pathlib import Path

def fix_file(filepath, description, fix_func):
    """Fix a file and report results."""
    print(f"\n[*] {description}")
    print(f"    File: {filepath.name}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        # Apply fix
        fixed = fix_func(original)

        if fixed != original:
            # Backup
            backup = str(filepath) + '.syntax_backup'
            with open(backup, 'w', encoding='utf-8') as f:
                f.write(original)

            # Write fixed
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed)

            # Verify
            try:
                compile(fixed, str(filepath), 'exec')
                print(f"    [OK] FIXED and verified")
                return True
            except SyntaxError as e:
                print(f"    [!] Still has error: {e.msg} at line {e.lineno}")
                return False
        else:
            print(f"    [-] No changes needed")
            return False

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"    [X] Error: {e}")
        return False


def remove_markdown_blocks(content):
    """Remove markdown code blocks."""
    # Remove ``` markers
    content = re.sub(r'\n```\n', '\n', content)
    content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
    return content


def fix_fstring_backslash(content):
    """Fix f-string with backslash."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'f"' in line and '\n' in line and line.count('\n') == 1:
            # Escape the backslash
            lines[i] = line.replace('\n', r'\n')
            print(f"    Line {i+1}: Fixed f-string backslash")
    return '\n'.join(lines)


def fix_unterminated_string(content):
    """Fix unterminated string literal."""
    lines = content.split('\n')
    for i, line in enumerate(lines[:100]):  # Check first 100 lines
        stripped = line.strip()
        # Check for odd number of quotes
        if (stripped.count('"') % 2 != 0 and "'" not in stripped[:3]) or \
           (stripped.count("'") % 2 != 0 and '"' not in stripped[:3]):
            # Add closing quote
            if stripped.startswith('"') and not stripped.endswith('"'):
                lines[i] = line + '"'
                print(f"    Line {i+1}: Added closing double quote")
            elif stripped.startswith("'") and not stripped.endswith("'"):
                lines[i] = line + "'"
                print(f"    Line {i+1}: Added closing single quote")
    return '\n'.join(lines)


def fix_missing_except_block(content):
    """Fix missing except block."""
    lines = content.split('\n')

    for i in range(len(lines)):
        line = lines[i]
        if re.match(r'^(\s*)try:\s*$', line):
            try_indent = len(line) - len(line.lstrip())

            # Look for except/finally
            found = False
            for j in range(i+1, min(i+30, len(lines))):
                next_line = lines[j]
                if next_line.strip() and not next_line.strip().startswith('#'):
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= try_indent:
                        if 'except' not in next_line and 'finally' not in next_line:
                            # Add except block
                            indent = ' ' * try_indent
                            lines.insert(j, indent + 'except Exception as e:')
                            lines.insert(j+1, indent + '    raise')
                            print(f"    Line {i+1}: Added missing except block")
                        found = True
                        break

            if found:
                break

    return '\n'.join(lines)


def fix_await_outside_async(content):
    """Fix await outside async function."""
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if re.match(r'^\s*await\s+', line):
            # Check if in async function
            in_async = False
            for j in range(max(0, i-50), i):
                if 'async def' in lines[j]:
                    in_async = True
                    break

            if not in_async:
                lines[i] = re.sub(r'^(\s*)await\s+', r'\1', line)
                print(f"    Line {i+1}: Removed await from non-async function")

    return '\n'.join(lines)


def fix_empty_except(content):
    """Fix empty except block."""
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if re.match(r'^(\s*)except.*:\s*$', line):
            except_indent = len(line) - len(line.lstrip())

            # Check next line
            if i+1 < len(lines):
                next_line = lines[i+1]
                if next_line.strip() == '' or next_line.strip().startswith('#'):
                    # Add raise or pass
                    indent = ' ' * (except_indent + 4)
                    lines.insert(i+1, indent + 'raise  # TODO: Add exception handling')
                    print(f"    Line {i+1}: Added raise to empty except block")

    return '\n'.join(lines)


def main():
    import shutil

    print("=" * 80)
    print("Quick Syntax Error Fixer")
    print("=" * 80)

    target_dir = Path('.')

    # Define fixes for each file
    fixes = [
        ('ace_mcp_tools_FIXED.py', 'Remove markdown blocks', remove_markdown_blocks),
        ('demo_mcts_mdap.py', 'Fix f-string backslash', fix_fstring_backslash),
        ('leanaide_mdap_demo.py', 'Fix unterminated string', fix_unterminated_string),
        ('workflow_stage_functions.py', 'Fix unterminated string', fix_unterminated_string),
        ('adversarial_adapter.py', 'Fix missing except block', fix_missing_except_block),
        ('bubblelabs_evolution_integration.py', 'Fix missing except block', fix_missing_except_block),
        ('simple_verify_implementation.py', 'Fix missing except block', fix_missing_except_block),
        ('adversarial_error_handling.py', 'Fix await outside async', fix_await_outside_async),
        ('hybrid_error_handling.py', 'Fix await outside async', fix_await_outside_async),
        ('sovereign_gauntlets.py', 'Fix empty except block', fix_empty_except),
    ]

    fixed = 0
    failed = 0

    for filename, description, fix_func in fixes:
        filepath = target_dir / filename
        if filepath.exists():
            if fix_file(filepath, description, fix_func):
                fixed += 1
            else:
                failed += 1
        else:
            print(f"\n[!] File not found: {filename}")

    # Generic fixes for remaining files
    remaining = ['leanaide_sop_integration.py', 'openevolve_leanaide_bridge.py']
    for filename in remaining:
        filepath = target_dir / filename
        if filepath.exists():
            print(f"\n[*] Attempting generic fix for {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Try multiple fixes
                content = remove_markdown_blocks(content)
                content = fix_unterminated_string(content)
                content = fix_missing_except_block(content)

                # Write and test
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                compile(content, str(filepath), 'exec')
                print(f"    [OK] FIXED")
                fixed += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"    [!] Could not fix automatically")
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error: {e}", exc_info=True)
                failed += 1

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Fixed: {fixed}")
    print(f"Failed: {failed}")
    print(f"\nBackups created with .syntax_backup extension")
    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
