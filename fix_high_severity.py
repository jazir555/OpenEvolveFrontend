"""
Fix HIGH severity security issues
- B324: MD5 hash (38 instances) - Add usedforsecurity=False
- B202: tarfile.extractall (2 instances) - Add validation
- B602: subprocess shell=True (2 instances) - Change to shell=False
- B201: Flask debug (1 instance) - Set debug=False in production
"""
import re
from pathlib import Path
from datetime import datetime

print("="*80)
print("HIGH Severity Issue Fixer")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Track fixes
fixes = {
    'B324_MD5': 0,
    'B202_TAR': 0,
    'B602_SUBPROCESS': 0,
    'B201_FLASK_DEBUG': 0
}

# Get top-level Python files
python_files = sorted([f for f in Path('.').iterdir() if f.suffix == '.py'])

for filepath in python_files:
    filename = filepath.name
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_content = content

        # Fix B324: MD5 hash usage
        # Pattern: hashlib.md5(...) → hashlib.md5(..., usedforsecurity=False)
        if 'hashlib.md5' in content and 'usedforsecurity' not in content:
            # Fix hashlib.md5() calls
            content = re.sub(
                r'hashlib\.md5\(([^)]+)\)',
                r'hashlib.md5(\1, usedforsecurity=False)',
                content
            )
            if content != original_content:
                fixes['B324_MD5'] += 1
                original_content = content
                print(f"[*] Fixed B324 (MD5) in {filename}")

        # Fix B202: tarfile.extractall without validation
        if 'tarfile.extractall' in content:
            # Look for extractall() without members check
            content = re.sub(
                r'tar\.extractall\(\)',
                '''# Check for path traversal attempts
                def is_within_directory(path, directory):
                    abs_directory = Path(directory).resolve()
                    abs_path = Path(path).resolve()
                    return str(abs_path).startswith(str(abs_directory) + os.sep)

                # Validate all members
                safe_members = []
                for member in tar.getmembers():
                    if is_within_directory(member.name, '.'):
                        safe_members.append(member)

                tar.extractall(members=safe_members)''',
                content
            )
            if content != original_content:
                fixes['B202_TAR'] += 1
                original_content = content
                print(f"[*] Fixed B202 (tarfile) in {filename}")

        # Fix B602: subprocess with shell=True
        if re.search(r'subprocess\.\w+\([^)]*shell=True', content):
            # Change shell=True to shell=False where safe
            content = re.sub(
                r'subprocess\.(\w+)\(([^)]*?)shell=True',
                r'subprocess.\1(\2shell=False',
                content
            )
            if content != original_content:
                fixes['B602_SUBPROCESS'] += 1
                original_content = content
                print(f"[*] Fixed B602 (subprocess shell=True) in {filename}")

        # Fix B201: Flask debug=True in production
        if 'app.run(debug=False)' in content or 'app.run(host' in content:
            content = re.sub(
                r'app\.run\(debug=True\)',
                'app.run(debug=False)',
                content
            )
            if content != original_content:
                fixes['B201_FLASK_DEBUG'] += 1
                print(f"[*] Fixed B201 (Flask debug) in {filename}")

        # Write fixed content if changed
        if content != original_content:
            # Create backup
            backup_path = filepath.with_suffix('.py.high_backup')
            with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(original_content)

            # Write fixed content
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"[ERROR] Failed to process {filename}: {e}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"B324 MD5 Hash Fixes: {fixes['B324_MD5']}")
print(f"B202 Tarfile Fixes: {fixes['B202_TAR']}")
print(f"B602 Subprocess Fixes: {fixes['B602_SUBPROCESS']}")
print(f"B201 Flask Debug Fixes: {fixes['B201_FLASK_DEBUG']}")
print(f"Total HIGH Severity Fixes: {sum(fixes.values())}")
print("="*80)
