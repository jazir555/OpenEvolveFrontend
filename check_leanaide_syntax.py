import os
import py_compile
import sys

def check_syntax():
    count = 0
    errors = []
    for f in os.listdir('.'):
        if f.startswith('leanaide_') and f.endswith('.py'):
            count += 1
            try:
                py_compile.compile(f, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append((f, str(e)))
                print(f"Error in {f}: {e}")
    
    print(f"Checked {count} files.")
    if errors:
        print(f"Found {len(errors)} files with syntax errors.")
        sys.exit(1)
    else:
        print("No syntax errors found.")

if __name__ == "__main__":
    check_syntax()
