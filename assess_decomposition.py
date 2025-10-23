"""
Honest assessment of problem_decomposition.py implementation
"""

import ast
import inspect
from problem_decomposition import ProblemDecomposer

# Get the class
cls = ProblemDecomposer

# Get all methods
all_methods = {}
for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
    if not name.startswith('__'):
        source = inspect.getsource(method)
        lines = len(source.split('\n'))
        # Check if it's a stub/placeholder
        is_stub = 'pass' in source and lines < 10
        has_todo = 'TODO' in source or 'FIXME' in source
        has_placeholder = 'placeholder' in source.lower() or 'not implemented' in source.lower()
        has_real_logic = lines > 15 and not is_stub
        
        all_methods[name] = {
            'lines': lines,
            'is_stub': is_stub,
            'has_todo': has_todo,
            'has_placeholder': has_placeholder,
            'has_real_logic': has_real_logic
        }

print('=== HONEST ASSESSMENT OF ProblemDecomposer ===\n')
print(f'Total methods: {len(all_methods)}\n')

# Categorize methods
public_methods = {k: v for k, v in all_methods.items() if not k.startswith('_')}
private_methods = {k: v for k, v in all_methods.items() if k.startswith('_')}

print(f'Public methods: {len(public_methods)}')
for name, info in sorted(public_methods.items()):
    status = '⚠️ STUB' if info['is_stub'] else ('✅ REAL' if info['has_real_logic'] else '⚡ SIMPLE')
    print(f'  {status} {name} ({info["lines"]} lines)')

print(f'\nPrivate/helper methods: {len(private_methods)}')
for name, info in sorted(private_methods.items()):
    status = '⚠️ STUB' if info['is_stub'] else ('✅ REAL' if info['has_real_logic'] else '⚡ SIMPLE')
    print(f'  {status} {name} ({info["lines"]} lines)')

# Check for stubs
stubs = [k for k, v in all_methods.items() if v['is_stub']]
todos = [k for k, v in all_methods.items() if v['has_todo']]
placeholders = [k for k, v in all_methods.items() if v['has_placeholder']]
real_implementations = [k for k, v in all_methods.items() if v['has_real_logic']]

print(f'\n=== QUALITY ASSESSMENT ===')
print(f'✅ Real implementations: {len(real_implementations)}/{len(all_methods)} ({len(real_implementations)*100//len(all_methods)}%)')
print(f'⚠️  Stub methods: {len(stubs)}')
if stubs:
    print(f'    {stubs}')
print(f'⚠️  TODO markers: {len(todos)}')
if todos:
    print(f'    {todos}')
print(f'⚠️  Placeholder methods: {len(placeholders)}')
if placeholders:
    print(f'    {placeholders}')

# Test actual functionality
print(f'\n=== FUNCTIONAL TESTING ===')
try:
    decomposer = ProblemDecomposer()
    
    test_content = """
# Test Header
This is test content with multiple sections.

## Subsection
More content here with details.

def test_function():
    return "test"
"""
    
    # Test each strategy
    from problem_decomposition import DecompositionStrategy
    strategies = [
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
        DecompositionStrategy.SEMANTIC,
        DecompositionStrategy.STRUCTURAL,
        DecompositionStrategy.DEPENDENCY_BASED,
        DecompositionStrategy.COMPLEXITY_BASED
    ]
    
    for strategy in strategies:
        try:
            result = decomposer.decompose_content(
                test_content,
                strategy=strategy,
                max_components=5,
                min_component_size=10
            )
            print(f'✅ {strategy.value}: {len(result.components)} components, quality {result.quality_score:.2f}')
        except Exception as e:
            print(f'❌ {strategy.value}: FAILED - {str(e)[:50]}')
    
    # Test reassembly
    try:
        result = decomposer.decompose_content(test_content)
        reassembly = decomposer.reassemble_components(
            result.components,
            result.reassembly_instructions
        )
        print(f'✅ Reassembly: quality {reassembly.quality_score:.2f}')
    except Exception as e:
        print(f'❌ Reassembly: FAILED - {str(e)[:50]}')
    
except Exception as e:
    print(f'❌ CRITICAL ERROR: {e}')
    import traceback
    traceback.print_exc()

print('\n=== FINAL VERDICT ===')
if len(real_implementations) > len(all_methods) * 0.7:
    print('✅ MOSTLY IMPLEMENTED - Good production quality')
elif len(real_implementations) > len(all_methods) * 0.4:
    print('⚡ PARTIALLY IMPLEMENTED - Has basic functionality')
else:
    print('❌ MOSTLY STUBS - Needs significant work')
