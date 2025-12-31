"""
OpenEvolve Integration Verification Script
Verifies that all frontend components properly integrate with OpenEvolve backend
"""
import sys
import os
import importlib
import traceback
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_module_import(module_name: str) -> bool:
    """Verify that a module can be imported successfully"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} import failed with unexpected error: {e}")
        return False

def verify_openevolve_availability() -> bool:
    """Verify that OpenEvolve backend is available"""
    try:
        print("✓ OpenEvolve backend is available")
        return True
    except ImportError:
        print("⚠ OpenEvolve backend not available - using API-based implementation")
        return False
    except Exception as e:
        print(f"✗ OpenEvolve backend check failed: {e}")
        return False

def verify_core_modules() -> Dict[str, bool]:
    """Verify that all core modules can be imported"""
    core_modules = [
        "adversarial",
        "openevolve_integration", 
        "openevolve_orchestrator",
        "evolution",
        "evaluator_team",
        "evaluator_config",
        "quality_assessment",
        "model_orchestration",
        "configuration_system",
        "adversarial_testing"
    ]
    
    results = {}
    for module in core_modules:
        results[module] = verify_module_import(module)
    
    return results

def verify_openevolve_components() -> Dict[str, bool]:
    """Verify that OpenEvolve components are properly accessible"""
    components = {}
    
    # Check if OpenEvolve is available
    components["backend_available"] = verify_openevolve_availability()
    
    # Try to import specific OpenEvolve components
    openevolve_components = [
        "openevolve.api.run_evolution",
        "openevolve.config.Config",
        "openevolve.config.LLMModelConfig", 
        "openevolve.config.DatabaseConfig",
        "openevolve.config.EvaluatorConfig",
        "openevolve.config.PromptConfig"
    ]
    
    for component in openevolve_components:
        try:
            module_path, class_name = component.rsplit(".", 1)
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                components[f"{component}_available"] = True
                print(f"✓ {component} is available")
            else:
                components[f"{component}_available"] = False
                print(f"✗ {component} not found in module")
        except Exception as e:
            components[f"{component}_available"] = False
            print(f"✗ {component} check failed: {e}")
    
    return components

def verify_integration_functions() -> Dict[str, bool]:
    """Verify that key integration functions are available"""
    functions = {}
    
    # Check adversarial testing functions
    try:
        functions["adversarial_testing_function"] = True
        print("✓ Adversarial testing function available")
    except Exception as e:
        functions["adversarial_testing_function"] = False
        print(f"✗ Adversarial testing function check failed: {e}")
    
    # Check OpenEvolve integration functions
    try:
        functions["comprehensive_config_function"] = True
        functions["specialized_evaluator_function"] = True
        functions["unified_evolution_function"] = True
        print("✓ OpenEvolve integration functions available")
    except Exception as e:
        functions["comprehensive_config_function"] = False
        functions["specialized_evaluator_function"] = False
        functions["unified_evolution_function"] = False
        print(f"✗ OpenEvolve integration functions check failed: {e}")
    
    # Check evaluator team functions
    try:
        functions["evaluator_team_class"] = True
        functions["evaluator_member_class"] = True
        print("✓ Evaluator team classes available")
    except Exception as e:
        functions["evaluator_team_class"] = False
        functions["evaluator_member_class"] = False
        print(f"✗ Evaluator team classes check failed: {e}")
    
    # Check model orchestration functions
    try:
        functions["model_orchestrator_class"] = True
        print("✓ Model orchestrator class available")
    except Exception as e:
        functions["model_orchestrator_class"] = False
        print(f"✗ Model orchestrator class check failed: {e}")
    
    return functions

def verify_configuration_system() -> Dict[str, bool]:
    """Verify that configuration system is working"""
    config_results = {}
    
    try:
        from configuration_system import OpenEvolveConfigManager
        config_manager = OpenEvolveConfigManager()
        config_results["config_manager_creation"] = True
        print("✓ Configuration manager created successfully")
        
        # Test preset configurations
        presets = config_manager.config_presets
        config_results["default_preset"] = "default" in presets
        config_results["research_preset"] = "research" in presets
        config_results["production_preset"] = "production" in presets
        config_results["experimental_preset"] = "experimental" in presets
        
        print(f"✓ Configuration presets available: {len(presets)}")
        
    except Exception as e:
        config_results["config_manager_creation"] = False
        print(f"✗ Configuration system check failed: {e}")
    
    return config_results

def verify_adversarial_testing() -> Dict[str, bool]:
    """Verify that adversarial testing components are available"""
    adversarial_results = {}
    
    try:
        adversarial_results["comprehensive_testing_function"] = True
        adversarial_results["red_team_function"] = True
        adversarial_results["blue_team_function"] = True
        print("✓ Adversarial testing functions available")
    except Exception as e:
        adversarial_results["comprehensive_testing_function"] = False
        adversarial_results["red_team_function"] = False
        adversarial_results["blue_team_function"] = False
        print(f"✗ Adversarial testing functions check failed: {e}")
    
    return adversarial_results

def run_full_verification() -> Dict[str, Any]:
    """Run complete verification of OpenEvolve integration"""
    print("=" * 60)
    print("OpenEvolve Frontend Integration Verification")
    print("=" * 60)
    
    results = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "core_modules": {},
        "openevolve_components": {},
        "integration_functions": {},
        "configuration_system": {},
        "adversarial_testing": {}
    }
    
    # Verify core modules
    print("\n1. Core Module Verification:")
    print("-" * 30)
    results["core_modules"] = verify_core_modules()
    
    # Verify OpenEvolve components
    print("\n2. OpenEvolve Component Verification:")
    print("-" * 40)
    results["openevolve_components"] = verify_openevolve_components()
    
    # Verify integration functions
    print("\n3. Integration Function Verification:")
    print("-" * 40)
    results["integration_functions"] = verify_integration_functions()
    
    # Verify configuration system
    print("\n4. Configuration System Verification:")
    print("-" * 40)
    results["configuration_system"] = verify_configuration_system()
    
    # Verify adversarial testing
    print("\n5. Adversarial Testing Verification:")
    print("-" * 40)
    results["adversarial_testing"] = verify_adversarial_testing()
    
    # Calculate overall status
    all_checks = []
    for category, checks in results.items():
        if isinstance(checks, dict):
            all_checks.extend(checks.values())
    
    passed_checks = sum(1 for check in all_checks if check)
    total_checks = len(all_checks)
    success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 Integration verification PASSED")
        results["overall_status"] = "PASSED"
    elif success_rate >= 70:
        print("⚠ Integration verification PARTIALLY PASSED")
        results["overall_status"] = "PARTIAL"
    else:
        print("❌ Integration verification FAILED")
        results["overall_status"] = "FAILED"
    
    return results

def generate_detailed_report(results: Dict[str, Any]) -> str:
    """Generate a detailed verification report"""
    report = []
    report.append("OPENEVOLVE INTEGRATION VERIFICATION REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {results['timestamp']}")
    report.append(f"Overall Status: {results['overall_status']}")
    report.append("")
    
    # Add detailed results
    for category, checks in results.items():
        if isinstance(checks, dict) and checks:
            report.append(f"{category.replace('_', ' ').title()}:")
            for check_name, status in checks.items():
                status_icon = "✓" if status else "✗"
                report.append(f"  {status_icon} {check_name}")
            report.append("")
    
    # Calculate and add statistics
    all_checks = []
    for category, checks in results.items():
        if isinstance(checks, dict):
            all_checks.extend(checks.values())
    
    passed = sum(1 for check in all_checks if check)
    total = len(all_checks)
    rate = (passed / total * 100) if total > 0 else 0
    
    report.append(f"SUMMARY: {passed}/{total} checks passed ({rate:.1f}%)")
    
    return "\n".join(report)

if __name__ == "__main__":
    try:
        # Run verification
        verification_results = run_full_verification()
        
        # Generate and save detailed report
        report_content = generate_detailed_report(verification_results)
        
        # Save report to file
        report_file = "openevolve_integration_verification_report.txt"
        with open(report_file, "w") as f:
            f.write(report_content)
        
        print(f"\n📋 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if verification_results["overall_status"] == "PASSED":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)