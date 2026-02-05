#!/usr/bin/env python3
"""
PES Enhanced Integration Test Suite

Tests how well the PES Enhanced system integrates with REAL existing OpenEvolve systems.

Test Coverage:
1. openevolve_agnostic_pes integration
2. Adaptive MDAP integration  
3. Workflow Engine integration
4. API Server integration
5. Breaking changes detection
"""

import sys
import asyncio
import inspect
import importlib
import traceback
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Test result tracking
@dataclass
class IntegrationTestResult:
    component: str
    test_name: str
    status: str  # WORKS, PARTIAL, BROKEN, UNKNOWN
    details: str
    code_snippet: str = ""
    recommendation: str = ""

class PESIntegrationTester:
    """Comprehensive PES Enhanced Integration Tester"""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.issues_found: List[Dict] = []
    
    def log_result(self, component: str, test_name: str, status: str, details: str, 
                   code_snippet: str = "", recommendation: str = ""):
        """Log a test result."""
        result = IntegrationTestResult(
            component=component,
            test_name=test_name,
            status=status,
            details=details,
            code_snippet=code_snippet,
            recommendation=recommendation
        )
        self.results.append(result)
        print(f"\n[{status}] {component} - {test_name}")
        print(f"  Details: {details}")
        if code_snippet:
            print(f"  Code: {code_snippet[:200]}...")
        if recommendation:
            print(f"  Fix: {recommendation}")
    
    # =========================================================================
    # TEST 1: openevolve_agnostic_pes Integration
    # =========================================================================
    
    def test_agnostic_pes_integration(self):
        """Test integration with AgnosticPESEngine."""
        print("\n" + "="*70)
        print("TEST 1: openevolve_agnostic_pes Integration")
        print("="*70)
        
        try:
            # Import the module
            import openevolve_agnostic_pes as agnostic_pes
            self.log_result(
                "openevolve_agnostic_pes",
                "Module Import",
                "WORKS",
                "Module imports successfully"
            )
        except ImportError as e:
            self.log_result(
                "openevolve_agnostic_pes",
                "Module Import",
                "BROKEN",
                f"Failed to import: {e}",
                recommendation="Ensure openevolve_agnostic_pes.py exists and has no syntax errors"
            )
            return
        
        # Test 1a: Check AgnosticPESEngine exists
        try:
            engine_class = agnostic_pes.AgnosticPESEngine
            self.log_result(
                "openevolve_agnostic_pes",
                "AgnosticPESEngine Class",
                "WORKS",
                f"AgnosticPESEngine found: {engine_class}"
            )
        except AttributeError as e:
            self.log_result(
                "openevolve_agnostic_pes",
                "AgnosticPESEngine Class",
                "BROKEN",
                f"AgnosticPESEngine not found: {e}",
                recommendation="Add AgnosticPESEngine class to openevolve_agnostic_pes.py"
            )
            return
        
        # Test 1b: Check evolve method signature
        try:
            evolve_method = engine_class.evolve
            sig = inspect.signature(evolve_method)
            params = list(sig.parameters.keys())
            
            expected_params = ['self', 'code', 'tests', 'problem_type']
            missing = [p for p in expected_params if p not in params]
            
            if missing:
                self.log_result(
                    "openevolve_agnostic_pes",
                    "evolve() Signature",
                    "PARTIAL",
                    f"Missing parameters: {missing}",
                    code_snippet=str(sig),
                    recommendation=f"Add missing parameters: {missing}"
                )
            else:
                self.log_result(
                    "openevolve_agnostic_pes",
                    "evolve() Signature",
                    "WORKS",
                    f"All expected parameters found: {params}",
                    code_snippet=str(sig)
                )
        except Exception as e:
            self.log_result(
                "openevolve_agnostic_pes",
                "evolve() Signature",
                "BROKEN",
                f"Error checking signature: {e}"
            )
        
        # Test 1c: Check EvolutionResult dataclass
        try:
            result_class = agnostic_pes.EvolutionResult
            expected_fields = ['original_code', 'evolved_code', 'iterations', 
                             'fixes_applied', 'improvement', 'final_score',
                             'tests_passed', 'tests_total']
            
            # Check if it's a dataclass
            if hasattr(result_class, '__dataclass_fields__'):
                actual_fields = list(result_class.__dataclass_fields__.keys())
                missing_fields = [f for f in expected_fields if f not in actual_fields]
                
                if missing_fields:
                    self.log_result(
                        "openevolve_agnostic_pes",
                        "EvolutionResult Fields",
                        "PARTIAL",
                        f"Missing fields: {missing_fields}",
                        code_snippet=str(actual_fields),
                        recommendation=f"Add missing fields to EvolutionResult: {missing_fields}"
                    )
                else:
                    self.log_result(
                        "openevolve_agnostic_pes",
                        "EvolutionResult Fields",
                        "WORKS",
                        f"All expected fields present: {actual_fields}"
                    )
            else:
                self.log_result(
                    "openevolve_agnostic_pes",
                    "EvolutionResult Type",
                    "PARTIAL",
                    "EvolutionResult is not a dataclass",
                    recommendation="Convert EvolutionResult to @dataclass"
                )
        except Exception as e:
            self.log_result(
                "openevolve_agnostic_pes",
                "EvolutionResult",
                "BROKEN",
                f"Error checking EvolutionResult: {e}"
            )
        
        # Test 1d: Check convenience functions
        try:
            evolve_code_func = agnostic_pes.evolve_code
            quick_evolve_func = agnostic_pes.quick_evolve
            
            self.log_result(
                "openevolve_agnostic_pes",
                "Convenience Functions",
                "WORKS",
                "evolve_code and quick_evolve functions available"
            )
        except AttributeError as e:
            self.log_result(
                "openevolve_agnostic_pes",
                "Convenience Functions",
                "PARTIAL",
                f"Missing convenience functions: {e}",
                recommendation="Add evolve_code() and quick_evolve() functions"
            )
    
    # =========================================================================
    # TEST 2: Adaptive MDAP Integration
    # =========================================================================
    
    def test_adaptive_mdap_integration(self):
        """Test integration with Adaptive MDAP system."""
        print("\n" + "="*70)
        print("TEST 2: Adaptive MDAP Integration")
        print("="*70)
        
        # Test 2a: Check adaptive_strategy_integration
        try:
            import adaptive_strategy_integration as asi
            self.log_result(
                "adaptive_strategy_integration",
                "Module Import",
                "WORKS",
                "Module imports successfully"
            )
        except ImportError as e:
            self.log_result(
                "adaptive_strategy_integration",
                "Module Import",
                "BROKEN",
                f"Failed to import: {e}"
            )
            asi = None
        
        # Test 2b: Check AdaptiveIntegrationManager
        if asi:
            try:
                manager_class = asi.AdaptiveIntegrationManager
                self.log_result(
                    "adaptive_strategy_integration",
                    "AdaptiveIntegrationManager",
                    "WORKS",
                    f"Manager class found: {manager_class}"
                )
                
                # Check for required methods
                required_methods = ['record_performance', 'select_strategy', 
                                  'get_recommended_strategies', 'get_performance_summary']
                for method in required_methods:
                    if hasattr(manager_class, method):
                        self.log_result(
                            "adaptive_strategy_integration",
                            f"Method: {method}()",
                            "WORKS",
                            f"Method {method}() exists"
                        )
                    else:
                        self.log_result(
                            "adaptive_strategy_integration",
                            f"Method: {method}()",
                            "BROKEN",
                            f"Method {method}() missing",
                            recommendation=f"Add {method}() method to AdaptiveIntegrationManager"
                        )
                        
            except AttributeError as e:
                self.log_result(
                    "adaptive_strategy_integration",
                    "AdaptiveIntegrationManager",
                    "BROKEN",
                    f"Manager not found: {e}"
                )
        
        # Test 2c: Check adaptive_mdap_pes_integration
        try:
            import adaptive_mdap_pes_integration as amp_integration
            self.log_result(
                "adaptive_mdap_pes_integration",
                "Module Import",
                "WORKS",
                "Module imports successfully"
            )
            
            # Check for key classes
            key_classes = ['AdaptivePESCoordinator', 'AdaptivePESConfig', 
                          'UnifiedBudgetTracker', 'ComplexityPESBridge']
            for cls_name in key_classes:
                if hasattr(amp_integration, cls_name):
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        f"Class: {cls_name}",
                        "WORKS",
                        f"Class {cls_name} found"
                    )
                else:
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        f"Class: {cls_name}",
                        "BROKEN" if cls_name == 'AdaptivePESCoordinator' else "PARTIAL",
                        f"Class {cls_name} missing",
                        recommendation=f"Add {cls_name} class to module"
                    )
                    
        except ImportError as e:
            self.log_result(
                "adaptive_mdap_pes_integration",
                "Module Import",
                "BROKEN",
                f"Failed to import: {e}"
            )
            amp_integration = None
        
        # Test 2d: Check complexity classifier integration
        if amp_integration:
            try:
                # Check if it tries to import TaskComplexityClassifier
                import inspect
                source = inspect.getsource(amp_integration)
                
                if 'TaskComplexityClassifier' in source:
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        "TaskComplexityClassifier Import",
                        "WORKS",
                        "Module attempts to import TaskComplexityClassifier"
                    )
                else:
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        "TaskComplexityClassifier Import",
                        "BROKEN",
                        "Module does not import TaskComplexityClassifier",
                        recommendation="Add TaskComplexityClassifier import with graceful fallback"
                    )
                    
                if 'AdaptiveMDAPAllocator' in source:
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        "AdaptiveMDAPAllocator Import",
                        "WORKS",
                        "Module attempts to import AdaptiveMDAPAllocator"
                    )
                else:
                    self.log_result(
                        "adaptive_mdap_pes_integration",
                        "AdaptiveMDAPAllocator Import",
                        "BROKEN",
                        "Module does not import AdaptiveMDAPAllocator",
                        recommendation="Add AdaptiveMDAPAllocator import with graceful fallback"
                    )
                    
            except Exception as e:
                self.log_result(
                    "adaptive_mdap_pes_integration",
                    "Source Inspection",
                    "UNKNOWN",
                    f"Could not inspect source: {e}"
                )
    
    # =========================================================================
    # TEST 3: Workflow Engine Integration
    # =========================================================================
    
    def test_workflow_engine_integration(self):
        """Test integration with Workflow Engine."""
        print("\n" + "="*70)
        print("TEST 3: Workflow Engine Integration")
        print("="*70)
        
        # Test 3a: Check workflow_engine imports
        try:
            import workflow_engine as we
            self.log_result(
                "workflow_engine",
                "Module Import",
                "WORKS",
                "Module imports successfully"
            )
        except ImportError as e:
            self.log_result(
                "workflow_engine",
                "Module Import",
                "BROKEN",
                f"Failed to import: {e}"
            )
            return
        
        # Test 3b: Check WorkflowState class
        try:
            from workflow_structures import WorkflowState
            self.log_result(
                "workflow_engine",
                "WorkflowState Import",
                "WORKS",
                "WorkflowState can be imported from workflow_structures"
            )
        except ImportError as e:
            self.log_result(
                "workflow_engine",
                "WorkflowState Import",
                "BROKEN",
                f"Cannot import WorkflowState: {e}",
                recommendation="Ensure workflow_structures.py exports WorkflowState"
            )
            WorkflowState = None
        
        # Test 3c: Check if workflow_engine uses PES
        try:
            import inspect
            we_source = inspect.getsource(we)
            
            pes_related = ['PES', 'agnostic_pes', 'pes_enhanced', 'evolve_code']
            found_pes = [term for term in pes_related if term.lower() in we_source.lower()]
            
            if found_pes:
                self.log_result(
                    "workflow_engine",
                    "PES Integration Check",
                    "WORKS",
                    f"Workflow engine contains PES references: {found_pes}"
                )
            else:
                self.log_result(
                    "workflow_engine",
                    "PES Integration Check",
                    "PARTIAL",
                    "Workflow engine does not appear to integrate with PES",
                    recommendation="Consider adding PES integration hooks to workflow_engine.py"
                )
                
        except Exception as e:
            self.log_result(
                "workflow_engine",
                "Source Inspection",
                "UNKNOWN",
                f"Could not inspect workflow_engine: {e}"
            )
        
        # Test 3d: Check ResourceManager integration
        try:
            from resource_manager import ResourceManager
            self.log_result(
                "workflow_engine",
                "ResourceManager Import",
                "WORKS",
                "ResourceManager available for cost tracking"
            )
        except ImportError as e:
            self.log_result(
                "workflow_engine",
                "ResourceManager Import",
                "PARTIAL",
                f"ResourceManager not available: {e}"
            )
        
        # Test 3e: Check monitoring_system integration
        try:
            from monitoring_system import add_metric, trace_operation, MetricType
            self.log_result(
                "workflow_engine",
                "Monitoring System Import",
                "WORKS",
                "Monitoring system available for cost tracking"
            )
        except ImportError as e:
            self.log_result(
                "workflow_engine",
                "Monitoring System Import",
                "PARTIAL",
                f"Monitoring system not available: {e}"
            )
    
    # =========================================================================
    # TEST 4: API Server Integration
    # =========================================================================
    
    def test_api_server_integration(self):
        """Test integration with API Server."""
        print("\n" + "="*70)
        print("TEST 4: API Server Integration")
        print("="*70)
        
        # Test 4a: Check api_server imports
        try:
            import api_server
            self.log_result(
                "api_server",
                "Module Import",
                "WORKS",
                "Module imports successfully"
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.log_result(
                "api_server",
                "Module Import",
                "BROKEN",
                f"Failed to import: {str(e)[:100]}",
                code_snippet=tb[:500],
                recommendation="Fix import error in api_server.py or its dependencies"
            )
            return
        
        # Test 4b: Check FastAPI app
        try:
            app = api_server.app
            self.log_result(
                "api_server",
                "FastAPI App",
                "WORKS",
                f"FastAPI app found: {type(app)}"
            )
        except AttributeError as e:
            self.log_result(
                "api_server",
                "FastAPI App",
                "BROKEN",
                f"FastAPI app not found: {e}",
                recommendation="Ensure api_server exports 'app' variable"
            )
            return
        
        # Test 4c: Check for PES-related routes
        try:
            routes = app.routes
            route_paths = [r.path for r in routes if hasattr(r, 'path')]
            
            pes_routes = [p for p in route_paths if 'pes' in p.lower()]
            if pes_routes:
                self.log_result(
                    "api_server",
                    "PES Routes",
                    "WORKS",
                    f"PES-related routes found: {pes_routes}"
                )
            else:
                self.log_result(
                    "api_server",
                    "PES Routes",
                    "PARTIAL",
                    "No PES-specific routes found",
                    recommendation="Consider adding /pes/enhance or similar endpoints"
                )
                
        except Exception as e:
            self.log_result(
                "api_server",
                "Route Inspection",
                "UNKNOWN",
                f"Could not inspect routes: {e}"
            )
        
        # Test 4d: Check Pydantic models
        try:
            from pydantic import BaseModel
            self.log_result(
                "api_server",
                "Pydantic BaseModel",
                "WORKS",
                "Pydantic models available for request/response validation"
            )
        except ImportError:
            self.log_result(
                "api_server",
                "Pydantic BaseModel",
                "BROKEN",
                "Pydantic not available",
                recommendation="Install pydantic: pip install pydantic"
            )
        
        # Test 4e: Check dependencies
        try:
            from fastapi import Depends, HTTPException, status
            self.log_result(
                "api_server",
                "FastAPI Dependencies",
                "WORKS",
                "FastAPI dependencies available"
            )
        except ImportError as e:
            self.log_result(
                "api_server",
                "FastAPI Dependencies",
                "BROKEN",
                f"FastAPI dependencies not available: {e}",
                recommendation="Install fastapi: pip install fastapi"
            )
    
    # =========================================================================
    # TEST 5: Breaking Changes Detection
    # =========================================================================
    
    def test_breaking_changes(self):
        """Check for breaking changes, naming conflicts, and import conflicts."""
        print("\n" + "="*70)
        print("TEST 5: Breaking Changes Detection")
        print("="*70)
        
        # Test 5a: Check for naming conflicts
        modules_to_check = [
            'openevolve_agnostic_pes',
            'openevolve_pes_integration',
            'adaptive_mdap_pes_integration',
            'adaptive_strategy_integration',
        ]
        
        all_exports = {}
        for module_name in modules_to_check:
            try:
                module = importlib.import_module(module_name)
                exports = [name for name in dir(module) if not name.startswith('_')]
                all_exports[module_name] = exports
            except ImportError:
                all_exports[module_name] = []
        
        # Check for duplicate exports
        export_counts = {}
        for module, exports in all_exports.items():
            for export in exports:
                if export not in export_counts:
                    export_counts[export] = []
                export_counts[export].append(module)
        
        duplicates = {k: v for k, v in export_counts.items() if len(v) > 1}
        
        # Filter out common duplicates (like dataclass, asyncio, etc.)
        common_names = {'dataclass', 'field', 'asyncio', 'List', 'Dict', 'Any', 'Optional',
                       'logging', 'logger', 'time', 'datetime', 'json', 'os', 'sys'}
        duplicates = {k: v for k, v in duplicates.items() if k not in common_names}
        
        if duplicates:
            for name, modules in duplicates.items():
                self.log_result(
                    "Breaking Changes",
                    f"Naming Conflict: {name}",
                    "PARTIAL",
                    f"'{name}' exported by multiple modules: {modules}",
                    recommendation=f"Consider namespacing {name} or using __all__ to control exports"
                )
        else:
            self.log_result(
                "Breaking Changes",
                "Naming Conflicts",
                "WORKS",
                "No significant naming conflicts detected"
            )
        
        # Test 5b: Check import cycles
        try:
            # Clear modules and re-import to check for cycles
            for module_name in list(sys.modules.keys()):
                if 'pes' in module_name.lower() or 'adaptive' in module_name.lower():
                    if module_name in sys.modules:
                        del sys.modules[module_name]
            
            # Re-import and check
            import openevolve_pes_integration
            self.log_result(
                "Breaking Changes",
                "Import Cycles",
                "WORKS",
                "No import cycles detected in openevolve_pes_integration"
            )
        except ImportError as e:
            self.log_result(
                "Breaking Changes",
                "Import Cycles",
                "BROKEN",
                f"Import cycle or error detected: {e}",
                recommendation="Check for circular imports between PES modules"
            )
        
        # Test 5c: Check for missing dependencies
        required_modules = [
            'openevolve_agnostic_pes',
            'openevolve_pes_integration',
        ]
        
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
                self.log_result(
                    "Breaking Changes",
                    f"Dependency: {module_name}",
                    "WORKS",
                    f"Module {module_name} available"
                )
            except ImportError as e:
                self.log_result(
                    "Breaking Changes",
                    f"Dependency: {module_name}",
                    "BROKEN",
                    f"Required module {module_name} not available: {e}",
                    recommendation=f"Ensure {module_name}.py exists and all its dependencies are available"
                )
        
        # Test 5d: Check for required third-party packages
        required_packages = [
            ('fastapi', 'FastAPI'),
            ('pydantic', 'Pydantic'),
        ]
        
        for package, display_name in required_packages:
            try:
                importlib.import_module(package)
                self.log_result(
                    "Breaking Changes",
                    f"Package: {display_name}",
                    "WORKS",
                    f"Package {display_name} available"
                )
            except ImportError:
                self.log_result(
                    "Breaking Changes",
                    f"Package: {display_name}",
                    "PARTIAL",
                    f"Package {display_name} not installed",
                    recommendation=f"Install with: pip install {package}"
                )
    
    # =========================================================================
    # TEST 6: Functional Integration Test
    # =========================================================================
    
    async def test_functional_integration(self):
        """Run functional tests to verify actual integration works."""
        print("\n" + "="*70)
        print("TEST 6: Functional Integration Test")
        print("="*70)
        
        # Test 6a: Test AgnosticPESEngine can be instantiated
        try:
            from openevolve_agnostic_pes import AgnosticPESEngine
            engine = AgnosticPESEngine(max_iterations=3)
            self.log_result(
                "Functional Test",
                "AgnosticPESEngine Instantiation",
                "WORKS",
                f"Engine created: {engine}"
            )
        except Exception as e:
            self.log_result(
                "Functional Test",
                "AgnosticPESEngine Instantiation",
                "BROKEN",
                f"Failed to create engine: {e}",
                recommendation="Check AgnosticPESEngine.__init__ for required parameters"
            )
            engine = None
        
        # Test 6b: Test OpenEvolvePESEnhancer
        try:
            from openevolve_pes_integration import OpenEvolvePESEnhancer
            enhancer = OpenEvolvePESEnhancer(max_iterations=3)
            self.log_result(
                "Functional Test",
                "OpenEvolvePESEnhancer Instantiation",
                "WORKS",
                f"Enhancer created: {enhancer}"
            )
        except Exception as e:
            self.log_result(
                "Functional Test",
                "OpenEvolvePESEnhancer Instantiation",
                "BROKEN",
                f"Failed to create enhancer: {e}",
                recommendation="Check OpenEvolvePESEnhancer.__init__ for required parameters"
            )
            enhancer = None
        
        # Test 6c: Test code evolution with simple example
        if engine:
            try:
                code = '''def add(a, b):
    return a + b
'''
                tests = [
                    {"name": "test_add", "input": {"a": 1, "b": 2}, "expected": 3, "function": "add"}
                ]
                
                result = await engine.evolve(code, tests, problem_type="general")
                
                self.log_result(
                    "Functional Test",
                    "Code Evolution",
                    "WORKS",
                    f"Evolution completed: score={result.final_score:.2%}, fixes={len(result.fixes_applied)}"
                )
            except Exception as e:
                self.log_result(
                    "Functional Test",
                    "Code Evolution",
                    "BROKEN",
                    f"Evolution failed: {e}",
                    recommendation="Check UniversalTestRunner and UniversalFixGenerator implementations"
                )
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    def generate_summary(self):
        """Generate test summary report."""
        print("\n" + "="*70)
        print("PES ENHANCED INTEGRATION TEST SUMMARY")
        print("="*70)
        
        status_counts = {"WORKS": 0, "PARTIAL": 0, "BROKEN": 0, "UNKNOWN": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        print(f"\nTotal Tests: {len(self.results)}")
        print(f"  [OK] WORKS:    {status_counts['WORKS']}")
        print(f"  [WARN] PARTIAL:  {status_counts['PARTIAL']}")
        print(f"  [FAIL] BROKEN:   {status_counts['BROKEN']}")
        print(f"  ? UNKNOWN:  {status_counts['UNKNOWN']}")
        
        # Critical issues
        critical_issues = [r for r in self.results if r.status == "BROKEN"]
        if critical_issues:
            print("\n" + "="*70)
            print("CRITICAL ISSUES (BROKEN)")
            print("="*70)
            for issue in critical_issues:
                print(f"\n* {issue.component} - {issue.test_name}")
                print(f"  {issue.details}")
                if issue.recommendation:
                    print(f"  -> Fix: {issue.recommendation}")
        
        # Partial issues
        partial_issues = [r for r in self.results if r.status == "PARTIAL"]
        if partial_issues:
            print("\n" + "="*70)
            print("PARTIAL ISSUES (NEED ATTENTION)")
            print("="*70)
            for issue in partial_issues:
                print(f"\n* {issue.component} - {issue.test_name}")
                print(f"  {issue.details}")
                if issue.recommendation:
                    print(f"  -> Fix: {issue.recommendation}")
        
        # Overall assessment
        print("\n" + "="*70)
        print("OVERALL ASSESSMENT")
        print("="*70)
        
        broken_pct = status_counts['BROKEN'] / len(self.results) * 100 if self.results else 0
        
        if broken_pct == 0:
            print("[OK] EXCELLENT: No critical issues found. Integration is solid.")
        elif broken_pct < 10:
            print("[OK] GOOD: Minor issues found but core functionality works.")
        elif broken_pct < 25:
            print("[WARN] MODERATE: Several issues need attention before production use.")
        else:
            print("[FAIL] POOR: Significant issues found. Integration needs major work.")
        
        return {
            'total_tests': len(self.results),
            'status_counts': status_counts,
            'broken_pct': broken_pct,
            'critical_issues': len(critical_issues),
            'partial_issues': len(partial_issues)
        }


async def main():
    """Run all integration tests."""
    tester = PESIntegrationTester()
    
    # Run all tests
    tester.test_agnostic_pes_integration()
    tester.test_adaptive_mdap_integration()
    tester.test_workflow_engine_integration()
    tester.test_api_server_integration()
    tester.test_breaking_changes()
    await tester.test_functional_integration()
    
    # Generate summary
    summary = tester.generate_summary()
    
    return summary


if __name__ == "__main__":
    summary = asyncio.run(main())
    
    # Exit with appropriate code
    if summary['broken_pct'] > 25:
        sys.exit(1)  # Major issues
    elif summary['broken_pct'] > 0:
        sys.exit(2)  # Minor issues
    else:
        sys.exit(0)  # All good
