"""
Comprehensive fixer for non-security code quality issues
Fixes: B101 (assert), B110 (try/except/pass), B112 (try/except/continue), B113 (requests timeout)
"""

import ast
import re
from pathlib import Path
from datetime import datetime


def fix_assert_statements(content, filename):
    """
    Fix B101: Assert used
    Replace assert statements with proper if checks and ValueError
    """
    # Skip test files
    if 'test' in filename.lower():
        return content, 0

    changes = 0
    lines = content.split('\n')
    result = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for assert statement
        if re.match(r'^\s*assert\s+', line):
            # Extract the assert condition
            assert_match = re.match(r'(\s*)assert\s+(.+?)(?:,\s*["\'](.+?)["\'])?\s*$', stripped)
            if assert_match:
                indent = assert_match.group(1)
                condition = assert_match.group(2)
                message = assert_match.group(3) or f"Assertion failed: {condition}"

                # Replace with if statement
                result.append(f'{indent}if not ({condition}):')
                result.append(f'{indent}    raise ValueError("{message}")')
                changes += 1
            else:
                result.append(line)
        else:
            result.append(line)

        i += 1

    return '\n'.join(result), changes


def fix_try_except_pass(content, filename):
    """
    Fix B110: Try, Except, Pass detected
    Replace with proper logging
    """
    changes = 0
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for pass in except block
        if re.search(r'except\s*.*:\s*$', line):
            # Look ahead for pass statement
            if i + 1 < len(lines) and lines[i + 1].strip() == 'pass':
                indent = len(line) - len(line.lstrip())
                base_indent = ' ' * indent
                log_indent = ' ' * (indent + 4)

                # Add logging instead of pass
                result.append(line.rstrip())
                result.append(f'{log_indent}import logging')
                result.append(f'{log_indent}logger = logging.getLogger(__name__)')
                result.append(f'{log_indent}logger.error(f"Error in {{__name__}}", exc_info=True)')
                result.append(f'{log_indent}raise  # Re-raise the exception')
                i += 2  # Skip the pass statement
                changes += 1
                continue

        result.append(line)
        i += 1

    return '\n'.join(result), changes


def fix_try_except_continue(content, filename):
    """
    Fix B112: Try, Except, Continue detected
    Add logging before continue
    """
    changes = 0
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for continue in except block
        if re.search(r'except\s*.*:\s*$', line):
            # Look ahead for continue statement
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1

            if j < len(lines) and lines[j].strip() == 'continue':
                indent = len(line) - len(line.lstrip())
                log_indent = ' ' * (indent + 4)

                # Add logging before continue
                result.append(line.rstrip())
                result.append(f'{log_indent}import logging')
                result.append(f'{log_indent}logger = logging.getLogger(__name__)')
                result.append(f'{log_indent}logger.warning(f"Continuing after error", exc_info=True)')
                result.append(lines[j])  # The continue statement
                i = j + 1
                changes += 1
                continue

        result.append(line)
        i += 1

    return '\n'.join(result), changes


def fix_requests_timeout(content, filename):
    """
    Fix B113: Calls to requests without timeout
    Add timeout parameter to requests calls
    """
    changes = 0
    lines = content.split('\n')
    result = []

    for line in lines:
        new_line = line

        # Find requests.get/post/put/delete/etc without timeout
        # Match: requests.method(...) without timeout parameter
        if 'requests.' in line and 'timeout' not in line:
            # Add timeout before closing parenthesis
            # Look for requests.get/post/put/delete/patch
            pattern = r'(requests\.(?:get|post|put|delete|patch|head|options))\s*\('

            def add_timeout(match):
                # Find the closing parenthesis
                start = match.end()
                rest_of_line = line[start:]
                # Check if there's already a timeout
                if 'timeout=' in rest_of_line:
                    return match.group(0)

                # Check if there are any parameters
                if ')' in rest_of_line:
                    # Find where to insert timeout
                    paren_pos = rest_of_line.rindex(')')
                    before = rest_of_line[:paren_pos]
                    after = rest_of_line[paren_pos:]

                    # Check if there are arguments
                    if before.strip() and before.strip() != '':
                        # Have arguments, add timeout before closing paren
                        return f"{match.group(1)}({before}, timeout=30{after}"
                    else:
                        # No arguments, just add timeout
                        return f"{match.group(1)}(timeout=30{after}"

                return match.group(0)

            new_line = re.sub(pattern, add_timeout, line)

        if new_line != line:
            changes += 1

        result.append(new_line)

    return '\n'.join(result), changes


def main():
    print("="*80)
    print("Non-Security Issue Fixer")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    python_files = sorted([f for f in Path('.').iterdir() if f.suffix == '.py'])

    # Skip test files and scripts
    skip_files = {
        'fix_non_security_issues.py', 'validate_all_fixes.py',
        'check_bandit_top_level.py', 'verify_our_fixes.py',
        'scan_top_level_only.py'
    }

    stats = {
        'assert_fixed': 0,
        'try_except_pass_fixed': 0,
        'try_except_continue_fixed': 0,
        'requests_timeout_fixed': 0,
        'files_modified': 0
    }

    for filepath in python_files:
        if filepath.name in skip_files:
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()

            content = original_content
            filename = filepath.name

            # Apply fixes
            content, assert_changes = fix_assert_statements(content, filename)
            content, pass_changes = fix_try_except_pass(content, filename)
            content, continue_changes = fix_try_except_continue(content, filename)
            content, timeout_changes = fix_requests_timeout(content, filename)

            total_changes = assert_changes + pass_changes + continue_changes + timeout_changes

            if total_changes > 0:
                # Create backup
                backup_path = filepath.with_suffix('.py.nonsec_backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                stats['assert_fixed'] += assert_changes
                stats['try_except_pass_fixed'] += pass_changes
                stats['try_except_continue_fixed'] += continue_changes
                stats['requests_timeout_fixed'] += timeout_changes
                stats['files_modified'] += 1

                print(f"[*] Fixed {filename}:")
                print(f"    - Assert statements: {assert_changes}")
                print(f"    - Try/Except/Pass: {pass_changes}")
                print(f"    - Try/Except/Continue: {continue_changes}")
                print(f"    - Requests timeout: {timeout_changes}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"[ERROR] Failed to process {filename}: {e}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files modified: {stats['files_modified']}")
    print(f"Assert statements fixed: {stats['assert_fixed']}")
    print(f"Try/Except/Pass fixed: {stats['try_except_pass_fixed']}")
    print(f"Try/Except/Continue fixed: {stats['try_except_continue_fixed']}")
    print(f"Requests timeout added: {stats['requests_timeout_fixed']}")
    print(f"Total fixes: {sum(stats.values()) - stats['files_modified']}")
    print("="*80)


if __name__ == '__main__':
    main()
