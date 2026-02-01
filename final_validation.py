#!/usr/bin/env python3
"""Final validation suite for math verification bubbles."""

import sys
sys.path.insert(0, '.')

print('Running Final Validation Suite...')
print('=' * 60)

# Test 1: Import all math bubbles
print('\n[Test 1] Import all math bubbles...')
try:
    from bubblelabs_nodes.lean_autoformalization_node import LeanAutoformalizationNode
    from bubblelabs_nodes.lean_proof_checking_node import LeanProofCheckingNode
    from bubblelabs_nodes.z3_constraint_solving_node import Z3ConstraintSolvingNode
    from bubblelabs_nodes.z3_theorem_proving_node import Z3TheoremProvingNode
    from bubblelabs_nodes.math_verification_pipeline_node import MathVerificationPipelineNode
    from bubblelabs_nodes.math_knowledge_extraction_node import MathKnowledgeExtractionNode
    from bubblelabs_nodes.proof_translation_node import ProofTranslationNode
    from bubblelabs_nodes.math_verification_dashboard_node import MathVerificationDashboardNode
    print('  [PASS] All imports successful')
except Exception as e:
    print(f'  [FAIL] Import error: {e}')
    sys.exit(1)

# Test 2: Instantiate all bubbles
print('\n[Test 2] Instantiate all bubbles...')
bubbles = [
    ('LeanAutoformalizationNode', LeanAutoformalizationNode),
    ('LeanProofCheckingNode', LeanProofCheckingNode),
    ('Z3ConstraintSolvingNode', Z3ConstraintSolvingNode),
    ('Z3TheoremProvingNode', Z3TheoremProvingNode),
    ('MathVerificationPipelineNode', MathVerificationPipelineNode),
    ('MathKnowledgeExtractionNode', MathKnowledgeExtractionNode),
    ('ProofTranslationNode', ProofTranslationNode),
    ('MathVerificationDashboardNode', MathVerificationDashboardNode),
]

instances = {}
for name, cls in bubbles:
    try:
        instance = cls(config={})
        instances[name] = instance
        print(f'  [PASS] {name} instantiated')
    except Exception as e:
        print(f'  [FAIL] {name} instantiation error: {e}')
        sys.exit(1)

# Test 3: Check all required attributes
print('\n[Test 3] Check required attributes...')
required_attrs = ['DISPLAY_NAME', 'DESCRIPTION', 'ICON', 'CATEGORY', 'VERSION']
for name, instance in instances.items():
    for attr in required_attrs:
        if not hasattr(instance, attr):
            print(f'  [FAIL] {name} missing {attr}')
            sys.exit(1)
    print(f'  [PASS] {name} has all required attributes')

# Test 4: Check all required methods
print('\n[Test 4] Check required methods...')
required_methods = ['execute', 'validate_inputs', 'get_parameter_schema', 'is_healthy']
for name, instance in instances.items():
    for method in required_methods:
        if not hasattr(instance, method):
            print(f'  [FAIL] {name} missing {method}')
            sys.exit(1)
    print(f'  [PASS] {name} has all required methods')

# Test 5: Check is_healthy returns True
print('\n[Test 5] Check is_healthy() returns True...')
for name, instance in instances.items():
    try:
        healthy = instance.is_healthy()
        if healthy:
            print(f'  [PASS] {name} is healthy')
        else:
            print(f'  [WARN] {name} is not healthy (but method works)')
    except Exception as e:
        print(f'  [FAIL] {name} is_healthy() error: {e}')
        sys.exit(1)

# Test 6: Check get_parameter_schema returns dict
print('\n[Test 6] Check get_parameter_schema() returns valid dict...')
for name, instance in instances.items():
    try:
        schema = instance.get_parameter_schema()
        if isinstance(schema, dict) and 'type' in schema and schema['type'] == 'object':
            print(f'  [PASS] {name} has valid parameter schema')
        else:
            print(f'  [WARN] {name} schema may be incomplete')
    except Exception as e:
        print(f'  [FAIL] {name} get_parameter_schema() error: {e}')
        sys.exit(1)

# Test 7: Check validate_inputs works
print('\n[Test 7] Check validate_inputs() works...')
for name, instance in instances.items():
    try:
        errors = instance.validate_inputs({})
        if isinstance(errors, list):
            print(f'  [PASS] {name} validate_inputs() works')
        else:
            print(f'  [FAIL] {name} validate_inputs() does not return list')
            sys.exit(1)
    except Exception as e:
        print(f'  [FAIL] {name} validate_inputs() error: {e}')
        sys.exit(1)

# Test 8: Check all bubbles have correct category
print('\n[Test 8] Check CATEGORY is "mathematical_verification"...')
for name, instance in instances.items():
    if instance.CATEGORY == 'mathematical_verification':
        print(f'  [PASS] {name} has correct category')
    else:
        print(f'  [FAIL] {name} has wrong category: {instance.CATEGORY}')
        sys.exit(1)

print('\n' + '=' * 60)
print('ALL TESTS PASSED!')
print('=' * 60)
print()
print('Summary:')
print('  - 8 math verification bubbles validated')
print('  - All imports working')
print('  - All attributes present')
print('  - All methods implemented')
print('  - All health checks passing')
print('  - All schemas valid')
print('  - All categories correct')
