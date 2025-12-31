import py_compile
import sys

files_to_check = [
    'workflow_structures.py',
    'knowledge_manager.py',
    'analytics_dashboard.py',
    'knowledge_base_ui.py',
    'dependency_visualizer.py',
    'resource_manager.py',
    'auto_approval.py',
    'batch_operations.py',
    'test_workflow_engine.py'
]

print("Verifying all Phase 4 files compile...\n")

all_pass = True
for filename in files_to_check:
    try:
        py_compile.compile(filename, doraise=True)
        print(f"✅ {filename}")
    except Exception as e:
        print(f"❌ {filename}: {e}")
        all_pass = False

print("\n" + "="*50)
if all_pass:
    print("✅ ALL FILES COMPILE SUCCESSFULLY!")
else:
    print("❌ SOME FILES HAVE ERRORS")
    sys.exit(1)
