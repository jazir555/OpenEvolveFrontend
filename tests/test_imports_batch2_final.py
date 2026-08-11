#!/usr/bin/env python3
"""
Final robust import testing for all Python files in core-projects/ directory.
Generates a JSON report with success/failure statistics.
"""

import os
import sys
import json
import importlib.util
import warnings
from pathlib import Path
from datetime import datetime
from io import StringIO

# Suppress all output during imports
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Suppress warnings and configure environment
warnings.filterwarnings('ignore')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTEST_CURRENT_TEST'] = ''  # Trick pytest into thinking it's running

# Configuration
CORE_PROJECTS_DIR = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/core-projects")
REPORT_PATH = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/import_test_batch2.json")

# Add paths for imports
PROJECT_ROOT = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend")

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
        "Skipped": 0,
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
    'Lib', 'Scripts', 'include', 'share', 'bin'
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
    """Test importing a single Python file using compile approach."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        
        # Skip empty files
        if not source.strip():
            return True, None
        
        # Try to compile (catches SyntaxError without executing side effects)
        compiled = compile(source, str(file_path), 'exec')
        
        return True, None
        
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} (line {e.lineno})"
        return False, error_msg
    except ValueError as e:
        # Often encoding issues
        error_msg = f"ValueError: {str(e)[:100]}"
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
                       "MemoryError", "RecursionError", "SystemExit", "Skipped"]:
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
    
    # Suppress output during testing
    with SuppressOutput():
        for i, file_path in enumerate(python_files, 1):
            try:
                relative_path = str(file_path.relative_to(PROJECT_ROOT))
            except ValueError:
                relative_path = str(file_path)
            
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
            
            # Progress indicator every 500 files (written to stderr to bypass suppression)
            if i % 500 == 0:
                print(f"Progress: {i}/{len(python_files)} files tested ({100*i//len(python_files)}%)...", file=sys.__stderr__)
    
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
    
    # Keep only first 3000 successful imports in report to avoid huge file
    if len(report_data["successful"]) > 3000:
        report_data["successful_note"] = f"Showing first 3000 of {len(report_data['successful'])} successful imports"
        report_data["successful"] = report_data["successful"][:3000]
    
    # Keep all failures
    
    try:
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Report written to: {REPORT_PATH}")
    except Exception as e:
        print(f"Error writing report: {e}")
    
    print()
    
    # Show sample of failures
    if results["failed"]:
        print("Sample of failures:")
        for fail in results["failed"][:10]:
            print(f"  - {fail['file']}")
            print(f"    {fail['error'][:100]}")
        if len(results["failed"]) > 10:
            print(f"  ... and {len(results['failed']) - 10} more failures")
    
    return results


if __name__ == "__main__":
    main()
