import os
import py_compile
import sys

def check_syntax():
    with open('leanaide_files.txt', 'r') as f:
        files = [line.strip() for line in f if line.strip().endswith('.py')]
    
    count = 0
    errors = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: File {f} listed in leanaide_files.txt does not exist.")
            continue
            
        count += 1
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((f, str(e)))
            print(f"Error in {f}: {e}")
    
    print(f"Checked {count} files.")
    if errors:
        print(f"Found {len(errors)} files with syntax errors.")
        for f, err in errors:
            print(f"--- {f} ---")
            print(err)
        sys.exit(1)
    else:
        print("No syntax errors found.")

if __name__ == "__main__":
    check_syntax()
