"""
Quick verification script for Phase II adapter
"""
import os
import sys

# Setup paths - use absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
# Go up to adapters, then up to glue, then to schemas
schemas_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'schemas'))

print(f"Adding to sys.path:")
print(f"  src: {src_dir}")
print(f"  schemas: {schemas_dir}")

sys.path.insert(0, src_dir)
sys.path.insert(0, schemas_dir)

# Set env vars BEFORE imports
os.environ['PHASE2_MAX_TARGET_DOMAINS'] = '10'
os.environ['PHASE2_IMECH_THRESHOLD'] = '0.7'
os.environ['PHASE2_PATTERN_THRESHOLD'] = '0.6'
os.environ['PHASE2_TIMEOUT_MS'] = '20000'
os.environ['PHASE2_MAX_MAPPINGS'] = '50'
os.environ['PHASE2_ENABLE_CONSTRAINT_INVERSION'] = 'true'
os.environ['PHASE2_SEARCH_DEPTH'] = '5'

print("Phase II Adapter Verification")
print("=" * 60)

# Test 1: Import schemas
print("\n1. Testing schema imports...")
try:
    from rese_schemas import (
        Phase2Config,
        IsomorphicMapping,
        FunctionalDependencyGraph,
        InvertedConstraint,
        IsomorphismType,
    )
    print("   OK - Schemas imported")
except Exception as e:
    print(f"   FAILED - {e}")
    sys.exit(1)

# Test 2: Import executor
print("\n2. Testing executor imports...")
try:
    from phase2_executor import (
        IsomorphicMappingExecutor,
        create_executor,
        StructureIdentifier,
        CrossDomainMapper,
    )
    print("   OK - Executor imported")
except Exception as e:
    print(f"   FAILED - {e}")
    sys.exit(1)

# Test 3: Create config
print("\n3. Testing configuration...")
try:
    config = Phase2Config.from_env()
    print(f"   OK - Config loaded")
    print(f"      - Max targets: {config.max_target_domains}")
    print(f"      - I_mech threshold: {config.i_mech_threshold}")
except Exception as e:
    print(f"   FAILED - {e}")
    sys.exit(1)

# Test 4: Create executor
print("\n4. Testing executor creation...")
try:
    executor = create_executor(config)
    print("   OK - Executor created")
except Exception as e:
    print(f"   FAILED - {e}")
    sys.exit(1)

# Test 5: Execute Phase II
print("\n5. Testing Phase II execution...")
try:
    result = executor.execute_phase2(
        source_domain='physics',
        problem_description='Energy conservation in closed system',
        target_domains=['biology', 'economics'],
        constraints=['energy is conserved']
    )

    print("   OK - Execution successful")
    print(f"      - Source: {result.source_domain}")
    print(f"      - Targets: {len(result.target_domains)}")
    print(f"      - Mappings: {len(result.mappings_found)}")
    print(f"      - Patterns: {len(result.cross_domain_patterns)}")
    print(f"      - Inverted: {len(result.inverted_constraints)}")
    print(f"      - Time: {result.execution_time_ms:.2f}ms")

    if result.best_mapping:
        print(f"      - Best I_mech: {result.best_mapping.i_mech_score:.2f}")

except Exception as e:
    print(f"   FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL VERIFICATION TESTS PASSED!")
print("=" * 60)
