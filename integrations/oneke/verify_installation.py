"""
OneKE Enhanced Integration - Installation Verification

This script verifies that all components are properly installed
and working correctly.
"""

import asyncio
import sys
from pathlib import Path

print("="*80)
print("OneKE Enhanced Integration - Installation Verification")
print("="*80)

# Track results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def check(condition, name, description):
    """Check a condition and record result."""
    if condition:
        results['passed'].append((name, description))
        print(f"[OK] {name}: {description}")
    else:
        results['failed'].append((name, description))
        print(f"[FAIL] {name}: {description}")

def warn(condition, name, description):
    """Check a condition and record warning."""
    if not condition:
        results['warnings'].append((name, description))
        print(f"[WARN] {name}: {description}")
    else:
        print(f"[OK] {name}: {description}")

# 1. Check Python version
print("\n1. Python Version Check")
print("-" * 80)
version = sys.version_info
check(
    version >= (3, 8),
    "Python Version",
    f"Python {version.major}.{version.minor}.{version.micro} (>= 3.8 required)"
)

# 2. Check core dependencies
print("\n2. Core Dependencies")
print("-" * 80)
try:
    import yaml
    check(True, "PyYAML", f"PyYAML {yaml.__version__}")
except ImportError:
    check(False, "PyYAML", "Not installed - run: pip install pyyaml")

try:
    import numpy
    check(True, "NumPy", f"NumPy {numpy.__version__}")
except ImportError:
    check(False, "NumPy", "Not installed - run: pip install numpy")

# 3. Check optional dependencies
print("\n3. Optional Dependencies")
print("-" * 80)
try:
    import sentence_transformers
    check(True, "Sentence Transformers", f"sentence-transformers {sentence_transformers.__version__}")
except ImportError:
    warn(False, "Sentence Transformers", "Not installed - fallback to keyword similarity")

try:
    import torch
    check(True, "PyTorch", f"PyTorch {torch.__version__}")
except ImportError:
    warn(False, "PyTorch", "Not installed - CPU mode for embeddings")

# 4. Check module imports
print("\n4. Module Imports")
print("-" * 80)
try:
    from integrations.oneke.case import (
        Case, CaseSimilarity, QualityScore,
        ReflectionResult, ConsistencyResult, EnhancedResult
    )
    check(True, "Case Data Structures", "All data structures imported")
except ImportError as e:
    check(False, "Case Data Structures", f"Import failed: {e}")

try:
    from integrations.oneke.case_repository import OneKECaseRepository
    check(True, "Case Repository", "Repository class imported")
except ImportError as e:
    check(False, "Case Repository", f"Import failed: {e}")

try:
    from integrations.oneke.reflection_agent import OneKEReflectionAgent
    check(True, "Reflection Agent", "Reflection agent imported")
except ImportError as e:
    check(False, "Reflection Agent", f"Import failed: {e}")

try:
    from integrations.oneke.quality_enhancement import OneKEQualityEnhancer
    check(True, "Quality Enhancer", "Quality enhancer imported")
except ImportError as e:
    check(False, "Quality Enhancer", f"Import failed: {e}")

try:
    from integrations.oneke.enhanced_bridge import EnhancedOneKEBridge
    check(True, "Enhanced Bridge", "Enhanced bridge imported")
except ImportError as e:
    check(False, "Enhanced Bridge", f"Import failed: {e}")

# 5. Check Knowledge Engine integration
print("\n5. Knowledge Engine Integration")
print("-" * 80)
try:
    from knowledge_engine.engine import KnowledgeEngine

    # Check if methods exist
    engine_methods = [
        'initialize_oneke_bridge',
        'extract_with_quality',
        'extract_and_learn',
        'batch_extract_with_quality'
    ]

    all_methods = all(hasattr(KnowledgeEngine, method) for method in engine_methods)
    check(
        all_methods,
        "KnowledgeEngine Methods",
        f"All {len(engine_methods)} OneKE methods present"
    )
except ImportError as e:
    check(False, "KnowledgeEngine", f"Import failed: {e}")

# 6. Check configuration files
print("\n6. Configuration Files")
print("-" * 80)
config_path = Path("integrations/oneke/config_enhanced.yaml")
if config_path.exists():
    check(True, "Enhanced Config", f"Config file found: {config_path}")
else:
    warn(False, "Enhanced Config", "Config file not found - using defaults")

# 7. Check data directory
print("\n7. Data Directory")
print("-" * 80)
data_dir = Path("data")
if data_dir.exists():
    check(True, "Data Directory", f"Data directory exists: {data_dir}")
else:
    warn(False, "Data Directory", "Data directory not found - will be created")

# 8. Verify data structure creation
print("\n8. Data Structure Creation")
print("-" * 80)
try:
    from integrations.oneke.case import Case

    # Create a test case
    test_case = Case.create(
        input_text="Test text",
        extracted_data={'entities': [], 'relations': []},
        schema="test",
        domain="test",
        quality_score=0.8
    )

    # Test serialization
    case_dict = test_case.to_dict()
    restored_case = Case.from_dict(case_dict)

    check(
        restored_case.case_id == test_case.case_id,
        "Case Serialization",
        "Case to_dict/from_dict working"
    )
except Exception as e:
    check(False, "Case Serialization", f"Failed: {e}")

# 9. Verify quality scoring
print("\n9. Quality Scoring")
print("-" * 80)
try:
    from integrations.oneke.case import QualityScore

    # Create test score
    score = QualityScore(
        completeness=0.9,
        accuracy=0.85,
        consistency=0.95,
        confidence=0.88,
        overall=0.89
    )

    # Verify ranges
    in_range = (
        0.0 <= score.completeness <= 1.0 and
        0.0 <= score.accuracy <= 1.0 and
        0.0 <= score.consistency <= 1.0 and
        0.0 <= score.confidence <= 1.0 and
        0.0 <= score.overall <= 1.0
    )

    check(in_range, "Quality Score Ranges", "All scores in valid range [0, 1]")
except Exception as e:
    check(False, "Quality Score Ranges", f"Failed: {e}")

# 10. Async verification (basic test)
print("\n10. Async Functionality")
print("-" * 80)
async def test_async():
    """Test async components."""
    try:
        # Create a simple case repository
        from integrations.oneke.case_repository import OneKECaseRepository
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = OneKECaseRepository(
                storage_path=str(Path(tmpdir) / "test.json")
            )

            # Initialize
            await repo.initialize()

            # Add a test case
            from integrations.oneke.case import Case
            test_case = Case.create(
                input_text="Test",
                extracted_data={},
                schema="test",
                domain="test",
                quality_score=0.8
            )

            await repo.add_case(test_case)

            # Get statistics
            stats = await repo.get_statistics()

            # Verify
            assert stats.total_cases == 1

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

try:
    success = asyncio.run(test_async())
    check(success, "Async Operations", "Async repository operations working")
except Exception as e:
    check(False, "Async Operations", f"Failed: {e}")

# 11. Documentation check
print("\n11. Documentation")
print("-" * 80)
docs_files = [
    ("README", "integrations/oneke/ENHANCED_README.md"),
    ("Quickstart", "integrations/oneke/QUICKSTART.md"),
    ("Implementation Summary", "integrations/oneke/PHASE4_IMPLEMENTATION_SUMMARY.md"),
    ("Examples", "integrations/oneke/example_enhanced.py"),
    ("Tests", "integrations/oneke/test_enhanced.py")
]

for name, path in docs_files:
    exists = Path(path).exists()
    if exists:
        check(True, f"Documentation: {name}", f"Found: {path}")
    else:
        warn(False, f"Documentation: {name}", f"Not found: {path}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print(f"\n[OK] Passed: {len(results['passed'])}")
print(f"[FAIL] Failed: {len(results['failed'])}")
print(f"[WARN] Warnings: {len(results['warnings'])}")

if results['failed']:
    print("\nFailed Checks:")
    for name, desc in results['failed']:
        print(f"  [FAIL] {name}: {desc}")

if results['warnings']:
    print("\nWarnings:")
    for name, desc in results['warnings']:
        print(f"  [WARN] {name}: {desc}")

# Overall result
if not results['failed']:
    print("\n" + "="*80)
    print("[OK] ALL CRITICAL CHECKS PASSED!")
    print("="*80)
    print("\nThe OneKE Enhanced Integration is ready to use.")
    print("\nNext Steps:")
    print("  1. Read QUICKSTART.md for usage examples")
    print("  2. Review ENHANCED_README.md for detailed documentation")
    print("  3. Run example_enhanced.py to see examples in action")
    print("  4. Run pytest integrations/oneke/test_enhanced.py for testing")
    sys.exit(0)
else:
    print("\n" + "="*80)
    print("[FAIL] SOME CHECKS FAILED")
    print("="*80)
    print("\nPlease resolve the failed checks above before using the integration.")
    sys.exit(1)
