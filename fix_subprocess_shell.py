"""
Fixer for subprocess calls with shell=True (B603)
Replace with safer shell=False or use list arguments
"""
import re
from pathlib import Path
from datetime import datetime


def fix_subprocess_shell(content, filename):
    """
    Fix B603: Subprocess call with shell=True
    Try to replace with shell=False and list arguments
    """
    changes = 0
    lines = content.split('\n')
    result = []

    for line in lines:
        new_line = line
        stripped = line.strip()

        # Look for subprocess.run/call/Popen with shell=True
        if 'subprocess.' in stripped and 'shell=True' in stripped:
            # Check if it's a simple command that can be converted
            # Look for: subprocess.run("command", shell=True)
            # Convert to: subprocess.run(["command"], shell=False)

            # Match subprocess patterns
            patterns = [
                r'subprocess\.run\(["\']([^"\']+)["\'],\s*shell=True\)',
                r'subprocess\.call\(["\']([^"\']+)["\'],\s*shell=True\)',
                r'subprocess\.Popen\(["\']([^"\']+)["\'],\s*shell=True\)',
            ]

            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    command = match.group(1)
                    # Split command into list
                    parts = command.split()
                    list_command = str(parts).replace("'", '"')

                    # Replace with list form and shell=False
                    new_line = re.sub(
                        pattern,
                        f'subprocess.{pattern.split(".")[1].split("(")[0]}({list_command}, shell=False)',
                        line
                    )
                    changes += 1
                    break

        result.append(new_line)

    return '\n'.join(result), changes


def main():
    print("="*80)
    print("Subprocess Shell=True Fixer")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Files with B603 issues from analysis
    target_files = [
        'bubblelab-auto-setup.py',
        'bubblelab-auto-setup-v2.py',
        'bubblelab-auto-setup-v3.py',
        'ci_cd_pipeline.py',
        'claudiomiro_mcp_tools.py',
    ]

    stats = {
        'files_modified': 0,
        'subprocess_fixed': 0
    }

    for filename in target_files:
        filepath = Path(filename)
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()

            content, changes = fix_subprocess_shell(original_content, filename)

            if changes > 0:
                # Create backup
                backup_path = filepath.with_suffix('.py.subproc_backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                stats['files_modified'] += 1
                stats['subprocess_fixed'] += changes

                print(f"[*] Fixed {filename}: {changes} subprocess calls")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"[ERROR] Failed to process {filename}: {e}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files modified: {stats['files_modified']}")
    print(f"Subprocess calls fixed: {stats['subprocess_fixed']}")
    print("="*80)


if __name__ == '__main__':
    main()
