"""
Simple Bug Check - Basic Code Analysis

This is a simplified version that performs basic checks
without the complexity that caused issues.
"""

import sys
import os
import re
from typing import Dict, Any, List

def check_file_basics(file_path: str) -> Dict[str, Any]:
    """Perform basic checks on a file."""
    
    results = {
        'file': file_path,
        'warnings': [],
        'status': 'ok'
    }
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic checks
        warnings = []
        
        # 1. Check for TODO/FIXME comments
        todo_count = len(re.findall(r'#\s*TODO', content, re.IGNORECASE))
        fixme_count = len(re.findall(r'#\s*FIXME', content, re.IGNORECASE))
        
        if todo_count > 0:
            warnings.append(f'TODO comments: {todo_count}')
        if fixme_count > 0:
            warnings.append(f'FIXME comments: {fixme_count}')
        
        # 2. Check for bare except clauses
        bare_except_count = len(re.findall(r'except:\s*(?:#|$)', content))
        if bare_except_count > 0:
            warnings.append(f'Bare except clauses: {bare_except_count}')
        
        # 3. Check for excessive print statements
        print_count = len(re.findall(r'print\(', content))
        if print_count > 10:
            warnings.append(f'Print statements: {print_count} (consider logging)')
        
        # 4. Check file size
        lines = content.split('\n')
        if len(lines) > 500:
            warnings.append(f'Large file: {len(lines)} lines')
        
        # 5. Check for hardcoded paths
        hardcoded_paths = re.findall(r'["\'](/[^"\']+)["\']', content)
        if len(hardcoded_paths) > 2:
            warnings.append(f'Hardcoded paths: {len(hardcoded_paths)}')
        
        if warnings:
            results['warnings'] = warnings
            results['status'] = 'warnings_found'
        
        return results
        
    except Exception as e:
        results['warnings'] = [f'Error reading file: {str(e)}']
        results['status'] = 'error'
        return results

def analyze_integration_files():
    """Analyze all integration files."""
    
    print("Simple Bug Check - Basic Analysis")
    print("=" * 50)
    
    # List of files to analyze
    knowledge_engine_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    integration_files = [
        'integrations/__init__.py',
        'integrations/deepke_integration.py',
        'integrations/karateclub_integration.py',
        'integrations/kg_gen_integration.py',
        'ai_enhanced_integration.py'
    ]
    
    all_results = []
    
    for file_path in integration_files:
        full_path = os.path.join(knowledge_engine_path, file_path)
        
        if os.path.exists(full_path):
            print(f"\nChecking {file_path}...")
            results = check_file_basics(full_path)
            all_results.append(results)
            
            # Print summary
            if results['status'] == 'ok':
                print(f"   [OK] No issues found")
            elif results['status'] == 'warnings_found':
                print(f"   [!!] {len(results['warnings'])} warnings")
                for warning in results['warnings']:
                    print(f"      - {warning}")
            else:
                print(f"   [ER] Error: {results['warnings'][0]}")
        else:
            print(f"   [ER] File not found: {full_path}")
    
    return all_results

def main():
    """Main function."""
    
    print("Simple Bug Check for AI Integration Code")
    print("=" * 50)
    print("Performing basic analysis of integration files...")
    print("=" * 50)
    
    # Analyze files
    results = analyze_integration_files()
    
    # Generate summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Count results
    files_checked = len(results)
    files_with_warnings = sum(1 for r in results if r['status'] == 'warnings_found')
    files_ok = sum(1 for r in results if r['status'] == 'ok')
    
    print(f"\nFiles Checked: {files_checked}")
    print(f"Clean: {files_ok}")
    print(f"With Warnings: {files_with_warnings}")
    
    if files_with_warnings == 0:
        print(f"\n[OK] No major issues found in basic analysis")
        print(f"   Code appears to be in good shape")
    else:
        print(f"\n[!!] Some files have warnings - see above for details")
    
    print("\n" + "=" * 50)
    print("Basic Analysis Complete")
    print("=" * 50)

if __name__ == "__main__":
    main()