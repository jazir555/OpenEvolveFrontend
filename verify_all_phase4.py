import py_compile
import sys

phase4_files = [
    'workflow_structures.py',
    'workflow_engine.py',
    'knowledge_manager.py',
    'analytics_dashboard.py',
    'knowledge_base_ui.py',
    'dependency_visualizer.py',
    'resource_manager.py',
    'auto_approval.py',
    'batch_operations.py',
    'llm_cache.py',
    'performance_utils.py',
    'dynamic_gauntlet_adaptation.py',
    'process_optimization.py',
    'test_workflow_engine.py',
    'test_integration.py'
]

print("="*60)
print("PHASE 4 IMPLEMENTATION VERIFICATION")
print("="*60)
print()

all_pass = True
for filename in phase4_files:
    try:
        py_compile.compile(filename, doraise=True)
        print(f"✅ {filename}")
    except Exception as e:
        print(f"❌ {filename}: {e}")
        all_pass = False

print()
print("="*60)
if all_pass:
    print("✅ ALL PHASE 4 FILES COMPILE SUCCESSFULLY!")
    print("="*60)
    print()
    print("Summary:")
    print(f"  Total Files: {len(phase4_files)}")
    print(f"  All Passing: YES")
    print()
    print("Run tests with:")
    print("  python -m pytest test_workflow_engine.py test_integration.py -v")
else:
    print("❌ SOME FILES HAVE ERRORS")
    print("="*60)
    sys.exit(1)
