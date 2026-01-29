<<<<<<< HEAD
"""
BubbleLabs Plugin System - Verification Script

This script verifies that the plugin system is correctly implemented
and all components are working as expected.

Run this script to verify your installation:
    python verify_plugin_system.py

Author: OpenEvolve Integration Team
Created: 2026-01-03
"""

import sys
import os
from pathlib import Path

# Add frontend directory to path
frontend_dir = Path(__file__).parent
sys.path.insert(0, str(frontend_dir))

def check_imports():
    """Check that all modules can be imported."""
    print("\n" + "=" * 80)
    print("CHECK 1: Module Imports")
    print("=" * 80)

    errors = []

    try:
        import bubblelabs_plugin_system
        print("✓ bubblelabs_plugin_system imported successfully")
    except ImportError as e:
        errors.append(f"bubblelabs_plugin_system: {e}")
        print(f"✗ bubblelabs_plugin_system failed: {e}")

    try:
        import openevolve_bubblelabs_plugin
        print("✓ openevolve_bubblelabs_plugin imported successfully")
    except ImportError as e:
        errors.append(f"openevolve_bubblelabs_plugin: {e}")
        print(f"✗ openevolve_bubblelabs_plugin failed: {e}")

    try:
        from bubblelabs_plugin_system import (
            BubbleLabsPlugin,
            PluginMetadata,
            PluginPriority,
            PluginState,
            PluginEvent,
            PluginRegistry,
            EventBus,
            get_plugin_registry,
            register_plugin,
        )
        print("✓ All core classes imported successfully")
    except ImportError as e:
        errors.append(f"Core classes: {e}")
        print(f"✗ Core classes failed: {e}")

    return len(errors) == 0, errors


def check_plugin_base_class():
    """Check that base plugin class has all required methods."""
    print("\n" + "=" * 80)
    print("CHECK 2: Base Plugin Class")
    print("=" * 80)

    from bubblelabs_plugin_system import BubbleLabsPlugin
    import inspect

    required_methods = [
        "get_metadata",
        "initialize",
        "start",
        "stop",
        "cleanup",
    ]

    optional_methods = [
        "register_hooks",
        "health_check",
        "get_status",
        "get_config",
        "update_config",
    ]

    errors = []

    for method in required_methods:
        if hasattr(BubbleLabsPlugin, method):
            print(f"✓ Required method: {method}")
        else:
            errors.append(f"Missing required method: {method}")
            print(f"✗ Missing required method: {method}")

    for method in optional_methods:
        if hasattr(BubbleLabsPlugin, method):
            print(f"✓ Optional method: {method}")
        else:
            print(f"⚠ Missing optional method: {method}")

    return len(errors) == 0, errors


def check_openevolve_plugin():
    """Check that OpenEvolve plugin is properly implemented."""
    print("\n" + "=" * 80)
    print("CHECK 3: OpenEvolve Plugin")
    print("=" * 80)

    from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin
    from bubblelabs_plugin_system import BubbleLabsPlugin

    errors = []

    # Check inheritance
    if issubclass(OpenEvolveBubbleLabsPlugin, BubbleLabsPlugin):
        print("✓ OpenEvolveBubbleLabsPlugin inherits from BubbleLabsPlugin")
    else:
        errors.append("OpenEvolveBubbleLabsPlugin doesn't inherit from BubbleLabsPlugin")
        print("✗ Invalid inheritance")

    # Check metadata
    try:
        metadata = OpenEvolveBubbleLabsPlugin.get_metadata()
        print(f"✓ Metadata: {metadata.name} v{metadata.version}")
        print(f"  Author: {metadata.author}")
        print(f"  Description: {metadata.description}")
    except Exception as e:
        errors.append(f"Metadata error: {e}")
        print(f"✗ Metadata error: {e}")

    return len(errors) == 0, errors


def check_backward_compatibility():
    """Check backward compatibility wrapper."""
    print("\n" + "=" * 80)
    print("CHECK 4: Backward Compatibility")
    print("=" * 80)

    from openevolve_bubblelabs_plugin import bubblelabs_integration

    errors = []

    # Check wrapper exists
    if bubblelabs_integration is not None:
        print("✓ Backward compatibility wrapper exists")
    else:
        errors.append("Backward compatibility wrapper not found")
        print("✗ Backward compatibility wrapper not found")

    # Check wrapper has required methods
    required_methods = [
        "create_workflow_definition_from_openevolve",
        "control_workflow_local",
    ]

    for method in required_methods:
        if hasattr(bubblelabs_integration, method):
            print(f"✓ Wrapper method: {method}")
        else:
            errors.append(f"Missing wrapper method: {method}")
            print(f"✗ Missing wrapper method: {method}")

    return len(errors) == 0, errors


def check_documentation():
    """Check that all documentation files exist."""
    print("\n" + "=" * 80)
    print("CHECK 5: Documentation Files")
    print("=" * 80)

    doc_files = [
        "BUBBLELABS_PLUGIN_SYSTEM_README.md",
        "BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md",
        "BUBBLELABS_PLUGIN_QUICK_REFERENCE.md",
        "BUBBLELABS_PLUGIN_REFACTORING_COMPLETE.md",
        "BUBBLELABS_PLUGIN_INDEX.md",
    ]

    errors = []

    for doc_file in doc_files:
        path = frontend_dir / doc_file
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {doc_file} ({size} bytes)")
        else:
            errors.append(f"Missing: {doc_file}")
            print(f"✗ Missing: {doc_file}")

    return len(errors) == 0, errors


def check_examples():
    """Check that example file exists."""
    print("\n" + "=" * 80)
    print("CHECK 6: Example File")
    print("=" * 80)

    example_file = frontend_dir / "examples" / "bubblelabs_plugin_examples.py"
    errors = []

    if example_file.exists():
        size = example_file.stat().st_size
        print(f"✓ bubblelabs_plugin_examples.py ({size} bytes)")
    else:
        errors.append("Missing example file")
        print("✗ Missing: examples/bubblelabs_plugin_examples.py")

    return len(errors) == 0, errors


def check_async_support():
    """Check async/await support."""
    print("\n" + "=" * 80)
    print("CHECK 7: Async/Await Support")
    print("=" * 80)

    import inspect
    from bubblelabs_plugin_system import BubbleLabsPlugin

    errors = []

    # Check that methods are async
    async_methods = ["initialize", "start", "stop", "cleanup"]

    for method in async_methods:
        if hasattr(BubbleLabsPlugin, method):
            method_obj = getattr(BubbleLabsPlugin, method)
            if inspect.iscoroutinefunction(method_obj):
                print(f"✓ {method} is async")
            else:
                errors.append(f"{method} is not async")
                print(f"✗ {method} is not async")
        else:
            errors.append(f"Missing method: {method}")
            print(f"✗ Missing method: {method}")

    return len(errors) == 0, errors


def check_thread_safety():
    """Check thread safety features."""
    print("\n" + "=" * 80)
    print("CHECK 8: Thread Safety")
    print("=" * 80)

    from bubblelabs_plugin_system import PluginRegistry
    import inspect

    errors = []

    # Check that registry uses locks
    if hasattr(PluginRegistry, "_lock"):
        print("✓ PluginRegistry has _lock for thread safety")
    else:
        errors.append("PluginRegistry missing _lock")
        print("✗ PluginRegistry missing _lock")

    # Check that EventBus uses locks
    from bubblelabs_plugin_system import EventBus
    if hasattr(EventBus, "_lock"):
        print("✓ EventBus has _lock for thread safety")
    else:
        errors.append("EventBus missing _lock")
        print("✗ EventBus missing _lock")

    return len(errors) == 0, errors


def check_event_system():
    """Check event system."""
    print("\n" + "=" * 80)
    print("CHECK 9: Event System")
    print("=" * 80)

    from bubblelabs_plugin_system import PluginEvent

    errors = []

    # Check required events exist
    required_events = [
        "BEFORE_LOAD",
        "AFTER_LOAD",
        "BEFORE_INIT",
        "AFTER_INIT",
        "BEFORE_START",
        "AFTER_START",
        "BEFORE_STOP",
        "AFTER_STOP",
        "BEFORE_UNLOAD",
        "AFTER_UNLOAD",
        "ON_ERROR",
        "ON_CONFIG_CHANGE",
    ]

    for event in required_events:
        if hasattr(PluginEvent, event):
            print(f"✓ Event: {event}")
        else:
            errors.append(f"Missing event: {event}")
            print(f"✗ Missing event: {event}")

    return len(errors) == 0, errors


def check_type_hints():
    """Check type hints."""
    print("\n" + "=" * 80)
    print("CHECK 10: Type Hints")
    print("=" * 80)

    import inspect
    from bubblelabs_plugin_system import PluginRegistry

    errors = []

    # Check that methods have type hints
    methods_to_check = [
        "register_plugin",
        "load_plugin",
        "start_plugin",
        "stop_plugin",
        "unload_plugin",
    ]

    for method_name in methods_to_check:
        if hasattr(PluginRegistry, method_name):
            method = getattr(PluginRegistry, method_name)
            signature = inspect.signature(method)
            if signature.return_annotation != inspect.Signature.empty:
                print(f"✓ {method_name} has type hints")
            else:
                print(f"⚠ {method_name} missing return type hint")
        else:
            errors.append(f"Missing method: {method_name}")
            print(f"✗ Missing method: {method_name}")

    return len(errors) == 0, errors


def main():
    """Run all verification checks."""
    print("\n")
    print("=" * 80)
    print(" BUBBLELABS PLUGIN SYSTEM - VERIFICATION")
    print("=" * 80)
    print("\nRunning verification checks...\n")

    checks = [
        ("Module Imports", check_imports),
        ("Base Plugin Class", check_plugin_base_class),
        ("OpenEvolve Plugin", check_openevolve_plugin),
        ("Backward Compatibility", check_backward_compatibility),
        ("Documentation Files", check_documentation),
        ("Example File", check_examples),
        ("Async/Await Support", check_async_support),
        ("Thread Safety", check_thread_safety),
        ("Event System", check_event_system),
        ("Type Hints", check_type_hints),
    ]

    results = []
    all_errors = []

    for check_name, check_func in checks:
        try:
            passed, errors = check_func()
            results.append((check_name, passed))
            all_errors.extend(errors)
        except Exception as e:
            results.append((check_name, False))
            all_errors.append(f"{check_name}: {e}")
            print(f"\n✗ {check_name} failed with exception: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80 + "\n")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for check_name, passed_result in results:
        status = "✓ PASS" if passed_result else "✗ FAIL"
        print(f"{status}: {check_name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if all_errors:
        print(f"\nErrors found: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")

    print("\n" + "=" * 80)
    if passed == total:
        print("✓ ALL CHECKS PASSED - Plugin system is ready!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("=" * 80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
=======
"""
BubbleLabs Plugin System - Verification Script

This script verifies that the plugin system is correctly implemented
and all components are working as expected.

Run this script to verify your installation:
    python verify_plugin_system.py

Author: OpenEvolve Integration Team
Created: 2026-01-03
"""

import sys
import os
from pathlib import Path

# Add frontend directory to path
frontend_dir = Path(__file__).parent
sys.path.insert(0, str(frontend_dir))

def check_imports():
    """Check that all modules can be imported."""
    print("\n" + "=" * 80)
    print("CHECK 1: Module Imports")
    print("=" * 80)

    errors = []

    try:
        import bubblelabs_plugin_system
        print("✓ bubblelabs_plugin_system imported successfully")
    except ImportError as e:
        errors.append(f"bubblelabs_plugin_system: {e}")
        print(f"✗ bubblelabs_plugin_system failed: {e}")

    try:
        import openevolve_bubblelabs_plugin
        print("✓ openevolve_bubblelabs_plugin imported successfully")
    except ImportError as e:
        errors.append(f"openevolve_bubblelabs_plugin: {e}")
        print(f"✗ openevolve_bubblelabs_plugin failed: {e}")

    try:
        from bubblelabs_plugin_system import (
            BubbleLabsPlugin,
            PluginMetadata,
            PluginPriority,
            PluginState,
            PluginEvent,
            PluginRegistry,
            EventBus,
            get_plugin_registry,
            register_plugin,
        )
        print("✓ All core classes imported successfully")
    except ImportError as e:
        errors.append(f"Core classes: {e}")
        print(f"✗ Core classes failed: {e}")

    return len(errors) == 0, errors


def check_plugin_base_class():
    """Check that base plugin class has all required methods."""
    print("\n" + "=" * 80)
    print("CHECK 2: Base Plugin Class")
    print("=" * 80)

    from bubblelabs_plugin_system import BubbleLabsPlugin
    import inspect

    required_methods = [
        "get_metadata",
        "initialize",
        "start",
        "stop",
        "cleanup",
    ]

    optional_methods = [
        "register_hooks",
        "health_check",
        "get_status",
        "get_config",
        "update_config",
    ]

    errors = []

    for method in required_methods:
        if hasattr(BubbleLabsPlugin, method):
            print(f"✓ Required method: {method}")
        else:
            errors.append(f"Missing required method: {method}")
            print(f"✗ Missing required method: {method}")

    for method in optional_methods:
        if hasattr(BubbleLabsPlugin, method):
            print(f"✓ Optional method: {method}")
        else:
            print(f"⚠ Missing optional method: {method}")

    return len(errors) == 0, errors


def check_openevolve_plugin():
    """Check that OpenEvolve plugin is properly implemented."""
    print("\n" + "=" * 80)
    print("CHECK 3: OpenEvolve Plugin")
    print("=" * 80)

    from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin
    from bubblelabs_plugin_system import BubbleLabsPlugin

    errors = []

    # Check inheritance
    if issubclass(OpenEvolveBubbleLabsPlugin, BubbleLabsPlugin):
        print("✓ OpenEvolveBubbleLabsPlugin inherits from BubbleLabsPlugin")
    else:
        errors.append("OpenEvolveBubbleLabsPlugin doesn't inherit from BubbleLabsPlugin")
        print("✗ Invalid inheritance")

    # Check metadata
    try:
        metadata = OpenEvolveBubbleLabsPlugin.get_metadata()
        print(f"✓ Metadata: {metadata.name} v{metadata.version}")
        print(f"  Author: {metadata.author}")
        print(f"  Description: {metadata.description}")
    except Exception as e:
        errors.append(f"Metadata error: {e}")
        print(f"✗ Metadata error: {e}")

    return len(errors) == 0, errors


def check_backward_compatibility():
    """Check backward compatibility wrapper."""
    print("\n" + "=" * 80)
    print("CHECK 4: Backward Compatibility")
    print("=" * 80)

    from openevolve_bubblelabs_plugin import bubblelabs_integration

    errors = []

    # Check wrapper exists
    if bubblelabs_integration is not None:
        print("✓ Backward compatibility wrapper exists")
    else:
        errors.append("Backward compatibility wrapper not found")
        print("✗ Backward compatibility wrapper not found")

    # Check wrapper has required methods
    required_methods = [
        "create_workflow_definition_from_openevolve",
        "control_workflow_local",
    ]

    for method in required_methods:
        if hasattr(bubblelabs_integration, method):
            print(f"✓ Wrapper method: {method}")
        else:
            errors.append(f"Missing wrapper method: {method}")
            print(f"✗ Missing wrapper method: {method}")

    return len(errors) == 0, errors


def check_documentation():
    """Check that all documentation files exist."""
    print("\n" + "=" * 80)
    print("CHECK 5: Documentation Files")
    print("=" * 80)

    doc_files = [
        "BUBBLELABS_PLUGIN_SYSTEM_README.md",
        "BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md",
        "BUBBLELABS_PLUGIN_QUICK_REFERENCE.md",
        "BUBBLELABS_PLUGIN_REFACTORING_COMPLETE.md",
        "BUBBLELABS_PLUGIN_INDEX.md",
    ]

    errors = []

    for doc_file in doc_files:
        path = frontend_dir / doc_file
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {doc_file} ({size} bytes)")
        else:
            errors.append(f"Missing: {doc_file}")
            print(f"✗ Missing: {doc_file}")

    return len(errors) == 0, errors


def check_examples():
    """Check that example file exists."""
    print("\n" + "=" * 80)
    print("CHECK 6: Example File")
    print("=" * 80)

    example_file = frontend_dir / "examples" / "bubblelabs_plugin_examples.py"
    errors = []

    if example_file.exists():
        size = example_file.stat().st_size
        print(f"✓ bubblelabs_plugin_examples.py ({size} bytes)")
    else:
        errors.append("Missing example file")
        print("✗ Missing: examples/bubblelabs_plugin_examples.py")

    return len(errors) == 0, errors


def check_async_support():
    """Check async/await support."""
    print("\n" + "=" * 80)
    print("CHECK 7: Async/Await Support")
    print("=" * 80)

    import inspect
    from bubblelabs_plugin_system import BubbleLabsPlugin

    errors = []

    # Check that methods are async
    async_methods = ["initialize", "start", "stop", "cleanup"]

    for method in async_methods:
        if hasattr(BubbleLabsPlugin, method):
            method_obj = getattr(BubbleLabsPlugin, method)
            if inspect.iscoroutinefunction(method_obj):
                print(f"✓ {method} is async")
            else:
                errors.append(f"{method} is not async")
                print(f"✗ {method} is not async")
        else:
            errors.append(f"Missing method: {method}")
            print(f"✗ Missing method: {method}")

    return len(errors) == 0, errors


def check_thread_safety():
    """Check thread safety features."""
    print("\n" + "=" * 80)
    print("CHECK 8: Thread Safety")
    print("=" * 80)

    from bubblelabs_plugin_system import PluginRegistry
    import inspect

    errors = []

    # Check that registry uses locks
    if hasattr(PluginRegistry, "_lock"):
        print("✓ PluginRegistry has _lock for thread safety")
    else:
        errors.append("PluginRegistry missing _lock")
        print("✗ PluginRegistry missing _lock")

    # Check that EventBus uses locks
    from bubblelabs_plugin_system import EventBus
    if hasattr(EventBus, "_lock"):
        print("✓ EventBus has _lock for thread safety")
    else:
        errors.append("EventBus missing _lock")
        print("✗ EventBus missing _lock")

    return len(errors) == 0, errors


def check_event_system():
    """Check event system."""
    print("\n" + "=" * 80)
    print("CHECK 9: Event System")
    print("=" * 80)

    from bubblelabs_plugin_system import PluginEvent

    errors = []

    # Check required events exist
    required_events = [
        "BEFORE_LOAD",
        "AFTER_LOAD",
        "BEFORE_INIT",
        "AFTER_INIT",
        "BEFORE_START",
        "AFTER_START",
        "BEFORE_STOP",
        "AFTER_STOP",
        "BEFORE_UNLOAD",
        "AFTER_UNLOAD",
        "ON_ERROR",
        "ON_CONFIG_CHANGE",
    ]

    for event in required_events:
        if hasattr(PluginEvent, event):
            print(f"✓ Event: {event}")
        else:
            errors.append(f"Missing event: {event}")
            print(f"✗ Missing event: {event}")

    return len(errors) == 0, errors


def check_type_hints():
    """Check type hints."""
    print("\n" + "=" * 80)
    print("CHECK 10: Type Hints")
    print("=" * 80)

    import inspect
    from bubblelabs_plugin_system import PluginRegistry

    errors = []

    # Check that methods have type hints
    methods_to_check = [
        "register_plugin",
        "load_plugin",
        "start_plugin",
        "stop_plugin",
        "unload_plugin",
    ]

    for method_name in methods_to_check:
        if hasattr(PluginRegistry, method_name):
            method = getattr(PluginRegistry, method_name)
            signature = inspect.signature(method)
            if signature.return_annotation != inspect.Signature.empty:
                print(f"✓ {method_name} has type hints")
            else:
                print(f"⚠ {method_name} missing return type hint")
        else:
            errors.append(f"Missing method: {method_name}")
            print(f"✗ Missing method: {method_name}")

    return len(errors) == 0, errors


def main():
    """Run all verification checks."""
    print("\n")
    print("=" * 80)
    print(" BUBBLELABS PLUGIN SYSTEM - VERIFICATION")
    print("=" * 80)
    print("\nRunning verification checks...\n")

    checks = [
        ("Module Imports", check_imports),
        ("Base Plugin Class", check_plugin_base_class),
        ("OpenEvolve Plugin", check_openevolve_plugin),
        ("Backward Compatibility", check_backward_compatibility),
        ("Documentation Files", check_documentation),
        ("Example File", check_examples),
        ("Async/Await Support", check_async_support),
        ("Thread Safety", check_thread_safety),
        ("Event System", check_event_system),
        ("Type Hints", check_type_hints),
    ]

    results = []
    all_errors = []

    for check_name, check_func in checks:
        try:
            passed, errors = check_func()
            results.append((check_name, passed))
            all_errors.extend(errors)
        except Exception as e:
            results.append((check_name, False))
            all_errors.append(f"{check_name}: {e}")
            print(f"\n✗ {check_name} failed with exception: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80 + "\n")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for check_name, passed_result in results:
        status = "✓ PASS" if passed_result else "✗ FAIL"
        print(f"{status}: {check_name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if all_errors:
        print(f"\nErrors found: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")

    print("\n" + "=" * 80)
    if passed == total:
        print("✓ ALL CHECKS PASSED - Plugin system is ready!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("=" * 80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
>>>>>>> 1cb9c5e35 (update)
