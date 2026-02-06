#!/usr/bin/env python3
"""Create all missing stub modules."""

import os
import json

# Load the scan results
with open('deep_import_scan_results.json') as f:
    data = json.load(f)

missing = data['missing_modules']

# Define content for common stub patterns
STUBS = {}

# Create basic stubs based on module names
for item in missing:
    module = item['module']
    
    # Skip if already created
    if os.path.exists(module + '.py') or os.path.exists(module.replace('.', '/') + '.py'):
        continue
    
    # Skip sub-modules (parent should handle them)
    if '.' in module:
        parent = module.split('.')[0]
        if parent in ['types', 'models', 'utils_ee', 'verification_result']:
            continue
    
    # Create class name from module name
    class_name = ''.join(word.capitalize() for word in module.split('_'))
    if '.' in module:
        class_name = ''.join(word.capitalize() for word in module.split('.')[-1].split('_'))
    
    content = f'''"""{module} module stub."""

class {class_name}:
    """Stub class for {module}."""
    pass

# Additional common exports
'''
    
    # Add specific imports based on what's used
    for file_info in item['files']:
        if 'Config' in file_info or 'config' in module:
            content += f'''
class {class_name}Config:
    """Configuration for {class_name}."""
    pass
'''
        if 'Error' in file_info:
            content += f'''
class {class_name}Error(Exception):
    """Error for {class_name}."""
    pass
'''
    
    STUBS[module] = content

def create_module(module_name, content):
    """Create a module file or package."""
    if '.' in module_name:
        # It's a sub-module, create package structure
        parts = module_name.split('.')
        dir_path = os.path.join(*parts[:-1]) if len(parts) > 1 else ''
        filename = parts[-1] + '.py'
        
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            # Create __init__.py for package
            init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'"""{dir_path} package."""\n')
        
        filepath = os.path.join(dir_path, filename) if dir_path else filename
    else:
        filepath = module_name + '.py'
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    return None

def main():
    print(f"Creating {len(STUBS)} stub modules...\n")
    
    created = 0
    for module, content in STUBS.items():
        result = create_module(module, content)
        if result:
            print(f"  Created: {result}")
            created += 1
    
    print(f"\nCreated {created} new stub modules")

if __name__ == "__main__":
    main()
