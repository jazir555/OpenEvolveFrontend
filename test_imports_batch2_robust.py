#!/usr/bin/env python3
"""
Robust import testing for all Python files in core-projects/ directory.
Generates a JSON report with success/failure statistics.
"""

import os
import sys
import json
import importlib.util
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings and configure environment to avoid initialization issues
warnings.filterwarnings('ignore')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Configuration
CORE_PROJECTS_DIR = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/core-projects")
REPORT_PATH = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/import_test_batch2.json")

# Add paths for imports
PROJECT_ROOT = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_PROJECTS_DIR))

# Results tracking
results = {
    "total_files": 0,
    "successful_imports": 0,
    "failed_imports": 0,
    "success_rate": "0%",
    "successful": [],
    "failed": [],
    "error_breakdown": {
        "ImportError": 0,
        "SyntaxError": 0,
        "ModuleNotFoundError": 0,
        "AttributeError": 0,
        "TypeError": 0,
        "ValueError": 0,
        "RuntimeError": 0,
        "KeyError": 0,
        "OSError": 0,
        "Other": 0
    },
    "timestamp": datetime.now().isoformat(),
    "directory_tested": str(CORE_PROJECTS_DIR)
}

# Directories to skip
SKIP_DIRS = {
    '__pycache__', '.venv', 'venv', 'env', '.git', 
    '.pytest_cache', 'node_modules', '.mypy_cache',
    'dist', 'build', 'egg-info', '.tox', '.idea',
    '.vscode', 'htmlcov', '.coverage', 'site-packages',
    'Lib', 'Scripts', 'include', 'share'
}


def get_all_python_files(directory):
    """Recursively find all Python files in directory."""
    python_files = []
    try:
        for root, dirs, files in os.walk(directory):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    python_files.append(full_path)
    except Exception as e:
        print(f"Error scanning directory: {e}")
    
    return python_files


def test_import_file(file_path):
    """Test importing a single Python file using compile/exec approach."""
    try:
        # Read and compile the file to catch syntax errors
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        
        # Try to compile first (catches SyntaxError without executing)
        compiled = compile(source, str(file_path), 'exec')
        
        # Create isolated namespace for execution
        namespace = {
            '__file__': str(file_path),
            '__name__': '__test_module__',
        }
        
        # Add parent directory to path for relative imports
        parent_dir = str(file_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try to execute
        exec(compiled, namespace)
        
        return True, None
        
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
        return False, error_msg
    except ImportError as e:
        error_msg = f"ImportError: {str(e)}"
        return False, error_msg
    except ModuleNotFoundError as e:
        error_msg = f"ModuleNotFoundError: {str(e)}"
        return False, error_msg
    except AttributeError as e:
        error_msg = f"AttributeError: {str(e)}"
        return False, error_msg
    except TypeError as e:
        error_msg = f"TypeError: {str(e)}"
        return False, error_msg
    except ValueError as e:
        error_msg = f"ValueError: {str(e)}"
        return False, error_msg
    except RuntimeError as e:
        error_msg = f"RuntimeError: {str(e)}"
        return False, error_msg
    except KeyError as e:
        error_msg = f"KeyError: {str(e)}"
        return False, error_msg
    except OSError as e:
        error_msg = f"OSError: {str(e)}"
        return False, error_msg
    except MemoryError as e:
        error_msg = f"MemoryError: {str(e)}"
        return False, error_msg
    except RecursionError as e:
        error_msg = f"RecursionError: {str(e)}"
        return False, error_msg
    except SystemExit as e:
        error_msg = f"SystemExit: code {e.code}"
        return False, error_msg
    except Exception as e:
        error_class = type(e).__name__
        error_msg = f"{error_class}: {str(e)[:150]}"
        return False, error_msg


def categorize_error(error_msg):
    """Extract error type from error message."""
    for error_type in ["SyntaxError", "ImportError", "ModuleNotFoundError", 
                       "AttributeError", "TypeError", "ValueError", 
                       "RuntimeError", "KeyError", "OSError", 
                       "MemoryError", "RecursionError", "SystemExit"]:
        if error_msg.startswith(error_type):
            return error_type
    return "Other"


def main():
    """Main function to test all imports."""
    print("=" * 80)
    print("IMPORT TESTING FOR core-projects/")
    print("=" * 80)
    print(f"Scanning directory: {CORE_PROJECTS_DIR}")
    print()
    
    # Get all Python files
    print("Finding Python files...")
    python_files = get_all_python_files(CORE_PROJECTS_DIR)
    results["total_files"] = len(python_files)
    print(f"Found {len(python_files)} Python files")
    print()
    
    # Test each file
    print("Testing imports (this may take a few minutes)...")
    print("-" * 80)
    
    batch_size = 500
    for i, file_path in enumerate(python_files, 1):
        try:
            relative_path = str(file_path.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_path = str(file_path)
        
        # Progress indicator
        if i % 100 == 0 or i == 1:
            print(f"Progress: {i}/{len(python_files)} files tested ({100*i//len(python_files)}%)...")
        
        success, error = test_import_file(file_path)
        
        if success:
            results["successful_imports"] += 1
            results["successful"].append(relative_path)
        else:
            results["failed_imports"] += 1
            error_type = categorize_error(error)
            if error_type in results["error_breakdown"]:
                results["error_breakdown"][error_type] += 1
            else:
                results["error_breakdown"]["Other"] += 1
            
            # Store failure details
            error_summary = error[:250] + "..." if len(error) > 250 else error
            results["failed"].append({
                "file": relative_path,
                "error": error_summary
            })
            
            # Print failures periodically
            if i % batch_size == 0:
                recent_failures = [f for f in results["failed"][-10:]]
                if recent_failures:
                    print(f"  Recent failures (batch ending at {i}):")
                    for fail in recent_failures[-5:]:
                        print(f"    - {fail['file']}: {fail['error'][:80]}")
    
    # Calculate success rate
    if results["total_files"] > 0:
        success_rate = (results["successful_imports"] / results["total_files"]) * 100
        results["success_rate"] = f"{success_rate:.2f}%"
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files:      {results['total_files']}")
    print(f"Successful:       {results['successful_imports']}")
    print(f"Failed:           {results['failed_imports']}")
    print(f"Success rate:     {results['success_rate']}")
    print()
    print("Error Breakdown:")
    for error_type, count in sorted(results["error_breakdown"].items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / results['total_files']) * 100 if results['total_files'] > 0 else 0
            print(f"  {error_type}: {count} ({pct:.1f}%)")
    print()
    
    # Write JSON report
    report_data = results.copy()
    
    # Keep only first 2000 successful imports in report to avoid huge file
    if len(report_data["successful"]) > 2000:
        report_data["successful_note"] = f"Showing first 2000 of {len(report_data['successful'])} successful imports"
        report_data["successful"] = report_data["successful"][:2000]
    
    # Keep all failures
    
    try:
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Report written to: {REPORT_PATH}")
    except Exception as e:
        print(f"Error writing report: {e}")
    
    print()
    
    return results


if __name__ == "__main__":
    main()
