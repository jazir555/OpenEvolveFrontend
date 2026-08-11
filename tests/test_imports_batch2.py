#!/usr/bin/env python3
"""
Systematic import testing for all Python files in core-projects/ directory.
Generates a JSON report with success/failure statistics.
"""

import os
import sys
import json
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime

# Configuration
CORE_PROJECTS_DIR = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/core-projects")
REPORT_PATH = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend/import_test_batch2.json")

# Add project root and core-projects to sys.path for proper imports
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


def get_all_python_files(directory):
    """Recursively find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.venv', 'venv', 'env', '.git', 
            '.pytest_cache', 'node_modules', '.mypy_cache',
            'dist', 'build', 'egg-info', '.tox', '.idea',
            '.vscode', 'htmlcov', '.coverage'
        }]
        
        for file in files:
            if file.endswith('.py'):
                full_path = Path(root) / file
                python_files.append(full_path)
    
    return python_files


def get_module_name(file_path, base_dir):
    """Convert file path to module name for import."""
    try:
        relative_path = file_path.relative_to(base_dir)
        parts = list(relative_path.parts)
        
        # Remove .py extension from last part
        parts[-1] = parts[-1][:-3]
        
        # Join with dots
        module_name = '.'.join(parts)
        return module_name
    except Exception:
        return None


def test_import_file(file_path, base_dir):
    """Test importing a single Python file."""
    relative_path = str(file_path.relative_to(PROJECT_ROOT))
    
    try:
        # Method 1: Try using importlib with spec
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Try to execute the module
        spec.loader.exec_module(module)
        
        return True, None
        
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} (line {e.lineno})"
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
    except Exception as e:
        error_class = type(e).__name__
        error_msg = f"{error_class}: {str(e)}"
        return False, error_msg


def categorize_error(error_msg):
    """Extract error type from error message."""
    if error_msg.startswith("ImportError"):
        return "ImportError"
    elif error_msg.startswith("SyntaxError"):
        return "SyntaxError"
    elif error_msg.startswith("ModuleNotFoundError"):
        return "ModuleNotFoundError"
    elif error_msg.startswith("AttributeError"):
        return "AttributeError"
    elif error_msg.startswith("TypeError"):
        return "TypeError"
    elif error_msg.startswith("ValueError"):
        return "ValueError"
    elif error_msg.startswith("RuntimeError"):
        return "RuntimeError"
    elif error_msg.startswith("KeyError"):
        return "KeyError"
    elif error_msg.startswith("OSError"):
        return "OSError"
    else:
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
    
    for i, file_path in enumerate(python_files, 1):
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        
        # Progress indicator every 100 files
        if i % 100 == 0 or i == 1:
            print(f"Progress: {i}/{len(python_files)} files tested...")
        
        success, error = test_import_file(file_path, CORE_PROJECTS_DIR)
        
        if success:
            results["successful_imports"] += 1
            results["successful"].append(relative_path)
        else:
            results["failed_imports"] += 1
            error_type = categorize_error(error)
            results["error_breakdown"][error_type] += 1
            
            # Store failure details (limit error message length)
            error_summary = error[:200] + "..." if len(error) > 200 else error
            results["failed"].append({
                "file": relative_path,
                "error": error_summary
            })
            
            # Print failures in real-time for visibility
            print(f"  FAIL: {relative_path}")
            print(f"        {error_summary}")
    
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
    for error_type, count in results["error_breakdown"].items():
        if count > 0:
            print(f"  {error_type}: {count}")
    print()
    
    # Write JSON report
    # Remove successful list if too large for readability
    report_data = results.copy()
    
    # Keep only first 1000 successful imports in report to avoid huge file
    if len(report_data["successful"]) > 1000:
        report_data["successful_note"] = f"Showing first 1000 of {len(report_data['successful'])} successful imports"
        report_data["successful"] = report_data["successful"][:1000]
    
    # Keep all failures
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"Report written to: {REPORT_PATH}")
    print()
    
    return results


if __name__ == "__main__":
    main()
