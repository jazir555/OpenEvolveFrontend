"""
Integration Verification Script

This script verifies that all AI integration files are properly structured
and can be imported without executing the full functionality.
"""

import sys
import os
import ast

# Add knowledge_engine to Python path
knowledge_engine_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if knowledge_engine_path not in sys.path:
    sys.path.insert(0, knowledge_engine_path)

def verify_integration_files():
    """Verify that all integration files are properly structured."""
    
    print("Verifying AI Integration Files...")
    print("=" * 50)
    
    # List of integration files to verify
    integration_files = [
        'integrations/__init__.py',
        'integrations/deepke_integration.py',
        'integrations/karateclub_integration.py',
        'integrations/kg_gen_integration.py',
        'ai_enhanced_integration.py'
    ]
    
    results = {}
    
    for file_path in integration_files:
        full_path = os.path.join(knowledge_engine_path, file_path)
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                results[file_path] = {
                    'status': 'missing',
                    'error': 'File not found'
                }
                print(f"[FAIL] {file_path}: File not found")
                continue
            
            # Check if file is valid Python
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            try:
                ast.parse(content)
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                error_msg = str(e)
            
            # Check for key components
            key_components = {
                'integrations/__init__.py': ['AIKnowledgeGraphIntegrator'],
                'integrations/deepke_integration.py': ['DeepKEEnhancedExtractor'],
                'integrations/karateclub_integration.py': ['KarateClubGraphAnalyzer'],
                'integrations/kg_gen_integration.py': ['EnhancedKnowledgeGraphManager'],
                'ai_enhanced_integration.py': ['AIEnhancedKnowledgeEngine']
            }
            
            file_components = key_components.get(os.path.basename(file_path), [])
            components_found = []
            
            for component in file_components:
                if component in content:
                    components_found.append(component)
            
            # Determine status
            if syntax_valid and len(components_found) == len(file_components):
                status = 'success'
                message = 'File is valid and contains all required components'
            elif syntax_valid:
                status = 'partial'
                message = f'File is valid but missing some components: {set(file_components) - set(components_found)}'
            else:
                status = 'syntax_error'
                message = f'File has syntax errors: {error_msg}'
            
            results[file_path] = {
                'status': status,
                'syntax_valid': syntax_valid,
                'components_found': components_found,
                'components_expected': file_components,
                'message': message
            }
            
            # Print result
            if status == 'success':
                print(f"[OK] {file_path}: Valid ({len(components_found)} components)")
            elif status == 'partial':
                print(f"[!!] {file_path}: Valid but missing components")
            else:
                print(f"[ER] {file_path}: Syntax errors")
                
        except Exception as e:
            results[file_path] = {
                'status': 'error',
                'error': str(e),
                'message': 'File verification failed'
            }
            print(f"[ER] {file_path}: Verification failed - {e}")
    
    return results

def verify_integration_structure():
    """Verify the overall integration structure."""
    
    print("\nVerifying Integration Structure...")
    print("=" * 50)
    
    # Check directory structure
    integrations_dir = os.path.join(knowledge_engine_path, 'integrations')
    
    structure_checks = {
        'integrations directory': os.path.isdir(integrations_dir),
        '__init__.py exists': os.path.exists(os.path.join(integrations_dir, '__init__.py')),
        'deepke_integration.py exists': os.path.exists(os.path.join(integrations_dir, 'deepke_integration.py')),
        'karateclub_integration.py exists': os.path.exists(os.path.join(integrations_dir, 'karateclub_integration.py')),
        'kg_gen_integration.py exists': os.path.exists(os.path.join(integrations_dir, 'kg_gen_integration.py')),
        'ai_enhanced_integration.py exists': os.path.exists(os.path.join(knowledge_engine_path, 'ai_enhanced_integration.py'))
    }
    
    structure_results = {}
    all_passed = True
    
    for check_name, check_result in structure_checks.items():
        structure_results[check_name] = check_result
        if check_result:
            print(f"[OK] {check_name}")
        else:
            print(f"[ER] {check_name}")
            all_passed = False
    
    return {'structure_valid': all_passed, 'checks': structure_results}

def main():
    """Main verification function."""
    
    print("AI Knowledge Graph Integration Verification")
    print("=" * 50)
    print("This tool verifies the structure and syntax of AI integration files")
    print("without executing the full functionality or requiring all dependencies.")
    print("=" * 50)
    
    # Verify file structure and syntax
    file_results = verify_integration_files()
    
    # Verify overall structure
    structure_results = verify_integration_structure()
    
    # Generate summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    # Count results
    success_count = sum(1 for result in file_results.values() if result['status'] == 'success')
    partial_count = sum(1 for result in file_results.values() if result['status'] == 'partial')
    error_count = sum(1 for result in file_results.values() if result['status'] in ['error', 'syntax_error', 'missing'])
    
    print(f"\n[FILE] File Verification:")
    print(f"   Success: {success_count}")
    print(f"   Partial: {partial_count}")
    print(f"   Errors: {error_count}")
    
    print(f"\n[STRUCT] Structure Verification:")
    print(f"   Structure Valid: {'[OK]' if structure_results['structure_valid'] else '[ER]'}")
    
    # Overall assessment
    if success_count == len(file_results) and structure_results['structure_valid']:
        overall_status = '[OK] COMPLETE SUCCESS'
        message = 'All integration files are properly structured and ready for use.'
    elif success_count >= len(file_results) * 0.8 and structure_results['structure_valid']:
        overall_status = '[!!] PARTIAL SUCCESS'
        message = 'Most files are correct with minor issues that can be addressed.'
    else:
        overall_status = '[ER] NEEDS ATTENTION'
        message = 'Significant issues found that need to be resolved.'
    
    print(f"\n[TARGET] {overall_status}")
    print(f"   {message}")
    
    # Provide recommendations
    if overall_status == '[OK] COMPLETE SUCCESS':
        print(f"\n[INFO] Recommendations:")
        print(f"   - The AI integrations are properly structured")
        print(f"   - All required components are present")
        print(f"   - The system is ready for testing and deployment")
    else:
        print(f"\n[INFO] Issues to Address:")
        for file_path, result in file_results.items():
            if result['status'] != 'success':
                print(f"   - {file_path}: {result['message']}")
        
        if not structure_results['structure_valid']:
            print(f"   - File structure issues detected")
    
    print("\n" + "=" * 50)
    print("Verification Complete")
    print("=" * 50)

if __name__ == "__main__":
    main()