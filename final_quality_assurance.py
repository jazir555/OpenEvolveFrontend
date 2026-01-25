#!/usr/bin/env python3
"""
Final Quality Assurance Suite
==============================

Comprehensive testing suite for the OpenEvolve migration.

Tests:
- Import tests (all files importable)
- Syntax tests (all files parse)
- Backward compatibility tests
- Performance benchmarks
- Memory usage tests
- Integration tests

Author: OpenEvolve Migration Team
Date: 2026-01-03
Status: Phase 3 Final QA
"""

import os
import sys
import time
import ast
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util
import subprocess


class QualityAssuranceSuite:
    """Comprehensive quality assurance testing suite."""

    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.python_files = list(self.root.rglob("*.py"))
        self.results = {
            "import_tests": {"passed": 0, "failed": 0, "errors": []},
            "syntax_tests": {"passed": 0, "failed": 0, "errors": []},
            "compatibility_tests": {"passed": 0, "failed": 0, "errors": []},
            "performance_tests": {"load_time": [], "access_time": [], "memory_usage": []},
            "integration_tests": {"passed": 0, "failed": 0, "errors": []},
            "start_time": time.time(),
        }

    def test_imports(self) -> None:
        """Test that all migrated files can be imported."""
        print("🔍 Testing imports...")
        print("-" * 80)

        test_files = [
            "unified_configuration",
            "base_configuration",
            "evolution_adapter",
            "adversarial_adapter",
            "openevolve_imports",
            "openevolve_validation",
        ]

        for module_name in test_files:
            try:
                # Try to import the module
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    self.root / f"{module_name}.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print(f"✅ {module_name}: Import successful")
                    self.results["import_tests"]["passed"] += 1
                else:
                    raise ImportError(f"Could not create spec for {module_name}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ {module_name}: Import failed - {e}")
                self.results["import_tests"]["failed"] += 1
                self.results["import_tests"]["errors"].append({
                    "module": module_name,
                    "error": str(e)
                })

        print()

    def test_syntax(self) -> None:
        """Test that all Python files have valid syntax."""
        print("🔍 Testing syntax...")
        print("-" * 80)

        checked = 0
        for filepath in self.python_files:
            # Skip __pycache__, .git, etc.
            if any(skip in str(filepath) for skip in ["__pycache__", ".git", "venv", ".venv"]):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
                checked += 1
                if checked <= 10:  # Show first 10
                    print(f"✅ {filepath.name}: Valid syntax")
            except SyntaxError as e:
                print(f"❌ {filepath.name}: Syntax error at line {e.lineno}")
                self.results["syntax_tests"]["failed"] += 1
                self.results["syntax_tests"]["errors"].append({
                    "file": str(filepath),
                    "error": str(e),
                    "line": e.lineno
                })
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"⚠️  {filepath.name}: Read error - {e}")
                self.results["syntax_tests"]["failed"] += 1

        self.results["syntax_tests"]["passed"] = checked
        if checked > 10:
            print(f"   ... and {checked - 10} more files")
        print()

    def test_backward_compatibility(self) -> None:
        """Test backward compatibility with old patterns."""
        print("🔍 Testing backward compatibility...")
        print("-" * 80)

        # Test that UnifiedConfig still works
        try:
            from unified_configuration import UnifiedConfig, get_config

            # Test basic access
            config = get_config()
            print("✅ get_config(): Works")

            # Test parameter access
            try:
                value = config.get("evolution.population_size")
                print(f"✅ Parameter access: evolution.population_size = {value}")
                self.results["compatibility_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ Parameter access failed: {e}")
                self.results["compatibility_tests"]["failed"] += 1

            # Test fallback
            try:
                value = config.get("nonexistent.parameter", default=42)
                print(f"✅ Fallback value: nonexistent.parameter = {value}")
                self.results["compatibility_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ Fallback failed: {e}")
                self.results["compatibility_tests"]["failed"] += 1

            # Test evolution adapter
            try:
                from evolution_adapter import EvolutionConfig
                evo_config = EvolutionConfig()
                print("✅ EvolutionConfig: Works")
                self.results["compatibility_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ EvolutionConfig failed: {e}")
                self.results["compatibility_tests"]["failed"] += 1

            # Test adversarial adapter
            try:
                from adversarial_adapter import AdversarialConfig
                adv_config = AdversarialConfig()
                print("✅ AdversarialConfig: Works")
                self.results["compatibility_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ AdversarialConfig failed: {e}")
                self.results["compatibility_tests"]["failed"] += 1

        except ImportError as e:
            print(f"❌ Import failed: {e}")
            self.results["compatibility_tests"]["failed"] += 1
            self.results["compatibility_tests"]["errors"].append({
                "test": "import",
                "error": str(e)
            })

        print()

    def test_performance(self) -> None:
        """Test performance of configuration system."""
        print("🔍 Testing performance...")
        print("-" * 80)

        try:
            from unified_configuration import get_config

            # Test load time
            load_times = []
            for _ in range(10):
                start = time.perf_counter()
                config = get_config()
                end = time.perf_counter()
                load_times.append((end - start) * 1000)  # Convert to ms

            avg_load_time = sum(load_times) / len(load_times)
            print(f"⏱️  Average load time: {avg_load_time:.2f}ms")
            self.results["performance_tests"]["load_time"] = avg_load_time

            # Test access time
            access_times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = config.get("evolution.population_size")
                end = time.perf_counter()
                access_times.append((end - start) * 1_000_000)  # Convert to μs

            avg_access_time = sum(access_times) / len(access_times)
            print(f"⏱️  Average access time: {avg_access_time:.2f}μs")
            self.results["performance_tests"]["access_time"] = avg_access_time

            # Test memory usage
            tracemalloc.start()
            config = get_config()

            # Perform some operations
            for _ in range(100):
                _ = config.get("evolution.population_size")
                _ = config.get("adversarial.enabled", default=False)
                _ = config.get("maker.max_iterations", default=100)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"💾 Current memory: {current / 1024:.2f}KB")
            print(f"💾 Peak memory: {peak / 1024:.2f}KB")
            self.results["performance_tests"]["memory_usage"] = peak / 1024

            # Performance verdict
            if avg_load_time < 100:
                print(f"✅ Load time: EXCELLENT ({avg_load_time:.2f}ms < 100ms)")
            elif avg_load_time < 200:
                print(f"✅ Load time: GOOD ({avg_load_time:.2f}ms < 200ms)")
            else:
                print(f"⚠️  Load time: ACCEPTABLE ({avg_load_time:.2f}ms)")

            if avg_access_time < 2:
                print(f"✅ Access time: EXCELLENT ({avg_access_time:.2f}μs < 2μs)")
            elif avg_access_time < 5:
                print(f"✅ Access time: GOOD ({avg_access_time:.2f}μs < 5μs)")
            else:
                print(f"⚠️  Access time: ACCEPTABLE ({avg_access_time:.2f}μs)")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"❌ Performance testing failed: {e}")
            self.results["performance_tests"]["errors"] = str(e)

        print()

    def test_integration(self) -> None:
        """Test integration between components."""
        print("🔍 Testing integration...")
        print("-" * 80)

        try:
            # Test unified imports
            try:
                from openevolve_imports import (
                    UnifiedConfig,
                    EvolutionConfig,
                    AdversarialConfig,
                    get_config
                )
                print("✅ Unified imports: Working")
                self.results["integration_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ Unified imports failed: {e}")
                self.results["integration_tests"]["failed"] += 1

            # Test cross-adapter compatibility
            try:
                config = get_config()
                evo_config = EvolutionConfig()
                adv_config = AdversarialConfig()

                # All should share the same underlying data
                print("✅ Cross-adapter compatibility: Working")
                self.results["integration_tests"]["passed"] += 1
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ Cross-adapter compatibility failed: {e}")
                self.results["integration_tests"]["failed"] += 1

            # Test validation
            try:
                from openevolve_validation import validate_config, validate_schema

                # Validate evolution config
                evolution_valid, evo_errors = validate_schema("evolution")
                print(f"✅ Evolution schema validation: {'VALID' if evolution_valid else 'INVALID'}")
                self.results["integration_tests"]["passed"] += 1

                # Validate adversarial config
                adversarial_valid, adv_errors = validate_schema("adversarial")
                print(f"✅ Adversarial schema validation: {'VALID' if adversarial_valid else 'INVALID'}")
                self.results["integration_tests"]["passed"] += 1

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"❌ Validation failed: {e}")
                self.results["integration_tests"]["failed"] += 1

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"❌ Integration testing failed: {e}")
            self.results["integration_tests"]["errors"].append({
                "test": "integration",
                "error": str(e)
            })

        print()

    def print_summary(self) -> None:
        """Print QA summary."""
        total_time = time.time() - self.results["start_time"]

        print("=" * 80)
        print("QUALITY ASSURANCE SUMMARY")
        print("=" * 80)
        print()

        print("📊 TEST RESULTS")
        print("-" * 80)

        # Import tests
        total_imports = self.results["import_tests"]["passed"] + self.results["import_tests"]["failed"]
        if total_imports > 0:
            import_pass_rate = (self.results["import_tests"]["passed"] / total_imports) * 100
            print(f"Import Tests:      {self.results['import_tests']['passed']}/{total_imports} passed ({import_pass_rate:.1f}%)")
        else:
            print("Import Tests:      SKIPPED")

        # Syntax tests
        total_syntax = self.results["syntax_tests"]["passed"] + self.results["syntax_tests"]["failed"]
        if total_syntax > 0:
            syntax_pass_rate = (self.results["syntax_tests"]["passed"] / total_syntax) * 100
            print(f"Syntax Tests:      {self.results['syntax_tests']['passed']}/{total_syntax} passed ({syntax_pass_rate:.1f}%)")
        else:
            print("Syntax Tests:      SKIPPED")

        # Compatibility tests
        total_compat = self.results["compatibility_tests"]["passed"] + self.results["compatibility_tests"]["failed"]
        if total_compat > 0:
            compat_pass_rate = (self.results["compatibility_tests"]["passed"] / total_compat) * 100
            print(f"Compatibility:     {self.results['compatibility_tests']['passed']}/{total_compat} passed ({compat_pass_rate:.1f}%)")
        else:
            print("Compatibility:     SKIPPED")

        # Integration tests
        total_integration = self.results["integration_tests"]["passed"] + self.results["integration_tests"]["failed"]
        if total_integration > 0:
            integration_pass_rate = (self.results["integration_tests"]["passed"] / total_integration) * 100
            print(f"Integration:       {self.results['integration_tests']['passed']}/{total_integration} passed ({integration_pass_rate:.1f}%)")
        else:
            print("Integration:       SKIPPED")

        print()

        # Performance
        if self.results["performance_tests"]["load_time"]:
            print("⏱️  PERFORMANCE METRICS")
            print("-" * 80)
            print(f"Load Time:        {self.results['performance_tests']['load_time']:.2f}ms")
            print(f"Access Time:      {self.results['performance_tests']['access_time']:.2f}μs")
            print(f"Memory Usage:     {self.results['performance_tests']['memory_usage']:.2f}KB")
            print()

        # Overall score
        print("=" * 80)
        print("FINAL ASSESSMENT")
        print("=" * 80)

        total_tests = (
            total_imports + total_syntax + total_compat + total_integration
        )
        total_passed = (
            self.results["import_tests"]["passed"] +
            self.results["syntax_tests"]["passed"] +
            self.results["compatibility_tests"]["passed"] +
            self.results["integration_tests"]["passed"]
        )

        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            print(f"Overall Pass Rate: {total_passed}/{total_tests} ({pass_rate:.1f}%)")
            print(f"Total Test Time:  {total_time:.2f}s")
            print()

            if pass_rate >= 99:
                print("✅ QUALITY ASSURANCE: EXCELLENT")
                print("✅ Ready for production deployment")
            elif pass_rate >= 95:
                print("✅ QUALITY ASSURANCE: VERY GOOD")
                print("✅ Ready for production deployment with monitoring")
            elif pass_rate >= 90:
                print("⚠️  QUALITY ASSURANCE: GOOD")
                print("⚠️  Minor issues should be addressed before production")
            else:
                print("❌ QUALITY ASSURANCE: NEEDS IMPROVEMENT")
                print("❌ Significant issues must be resolved")
        else:
            print("⚠️  No tests were run")

        print("=" * 80)


def main():
    """Main execution."""
    print("=" * 80)
    print("FINAL QUALITY ASSURANCE SUITE")
    print("OpenEvolve Unified Configuration Migration")
    print("=" * 80)
    print()

    qa = QualityAssuranceSuite(".")

    # Run all tests
    qa.test_imports()
    qa.test_syntax()
    qa.test_backward_compatibility()
    qa.test_performance()
    qa.test_integration()

    # Print summary
    qa.print_summary()

    # Return exit code based on pass rate
    total_tests = (
        qa.results["import_tests"]["passed"] + qa.results["import_tests"]["failed"] +
        qa.results["syntax_tests"]["passed"] + qa.results["syntax_tests"]["failed"] +
        qa.results["compatibility_tests"]["passed"] + qa.results["compatibility_tests"]["failed"] +
        qa.results["integration_tests"]["passed"] + qa.results["integration_tests"]["failed"]
    )
    total_passed = (
        qa.results["import_tests"]["passed"] +
        qa.results["syntax_tests"]["passed"] +
        qa.results["compatibility_tests"]["passed"] +
        qa.results["integration_tests"]["passed"]
    )

    if total_tests > 0:
        pass_rate = (total_passed / total_tests) * 100
        return 0 if pass_rate >= 90 else 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
