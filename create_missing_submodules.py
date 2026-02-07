#!/usr/bin/env python3
"""Create missing submodules for paths that have the top module existing."""

import os
import json
from pathlib import Path

# Load the missing imports
with open('main_project_missing.json') as f:
    missing = json.load(f)

# Filter for ones where top module exists
top_module_exists = {}
for mod, files in missing.items():
    top = mod.split('.')[0]
    if Path(f"{top}.py").exists() or Path(top).exists():
        top_module_exists[mod] = files

print(f"Missing submodules where top exists: {len(top_module_exists)}")

# Create the missing submodules
for mod, files in top_module_exists.items():
    parts = mod.split('.')
    
    # Skip external libraries
    if parts[0] in {'dspy', 'crewai', 'langchain_openai', 'cryptography', 'rich', 
                     'PIL', 'neo4j', 'loguru', 'jinja2', 'reportlab', 'psycopg2',
                     'starlette', 'pygraphistry', 'datapizza'}:
        continue
    
    # Create the file path
    filepath = Path(*parts).with_suffix('.py')
    
    if filepath.exists():
        continue
    
    # Check if it's importing from a real file
    if len(files) > 0:
        # Get what names are being imported
        import_names = set()
        for file in files[:5]:  # Check first few files
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find the import statement
                for line in content.split('\n'):
                    if f'from {mod}' in line or f'import {mod}' in line:
                        # Extract what's being imported
                        if 'import' in line:
                            parts_line = line.split('import')
                            if len(parts_line) > 1:
                                names = parts_line[1].strip()
                                for name in names.split(','):
                                    import_names.add(name.strip().split()[0])
            except:
                pass
        
        # Create the module with the imported names
        dir_path = filepath.parent
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Generate content
        class_name = ''.join(p.capitalize() for p in parts[-1].split('_'))
        
        content_lines = [f'"""{mod} module."""', '']
        
        # Add the imported names
        for name in import_names:
            if name and name not in ('*', ''):
                content_lines.append(f'class {name}:')
                content_lines.append('    pass')
                content_lines.append('')
        
        # Always add a main class
        content_lines.append(f'class {class_name}:')
        content_lines.append('    """Main class."""')
        content_lines.append('    pass')
        content_lines.append('')
        
        content = '\n'.join(content_lines)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created: {filepath}")
        except Exception as e:
            print(f"Error creating {filepath}: {e}")

print("\nDone creating missing submodules")
