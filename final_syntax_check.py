import os
import py_compile
import sys

def check_syntax():
    frontend_dir = os.getcwd()
    files = [f for f in os.listdir(frontend_dir) if f.startswith('leanaide_') and f.endswith('.py')]
    print(f"Found {len(files)} leanaide_*.py files in {frontend_dir}")
    
    error_count = 0
    for f in sorted(files):
        try:
            py_compile.compile(os.path.join(frontend_dir, f), doraise=True)
            print(f"OK: {f}")
        except py_compile.PyCompileError as e:
            print(f"ERROR in {f}:")
            print(e)
            error_count += 1
        except Exception as e:
            print(f"Unexpected error checking {f}: {e}")
            error_count += 1
            
    if error_count > 0:
        print(f"\nFound {error_count} syntax errors.")
        sys.exit(1)
    else:
        print("\nAll files passed syntax check.")
        sys.exit(0)

if __name__ == "__main__":
    check_syntax()
