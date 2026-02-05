"""
TRUE 100% Fixes Verification Test

Tests all the fixes applied to reach TRUE 100%:
1. Z3 Prover: ParetoOptimizer pareto_optimize method
2. Physics Validator: Modal analysis eigenvalue handling
3. SOP Generator: generate_sop method with correct signature
4. OneKE Adapter: _call_oneke method
5. DeepKE Setup: Isolated environment setup script
"""

import asyncio
import sys
from pathlib import Path


def test_z3_prover():
    """Test Z3 Prover ParetoOptimizer fixes."""
    print("\n" + "=" * 60)
    print("TEST 1: Z3 Prover ParetoOptimizer")
    print("=" * 60)
    
    try:
        from z3prover_advanced import ParetoOptimizer, MultiObjectiveOptimizer, OptimizationObjective
        
        # Test ParetoOptimizer has pareto_optimize method
        po = ParetoOptimizer()
        assert hasattr(po, 'pareto_optimize'), "ParetoOptimizer missing pareto_optimize method"
        print("[OK] ParetoOptimizer has pareto_optimize method")
        
        # Test ParetoOptimizer has optimize_multi_objective method
        assert hasattr(po, 'optimize_multi_objective'), "ParetoOptimizer missing optimize_multi_objective method"
        print("[OK] ParetoOptimizer has optimize_multi_objective method")
        
        # Test MultiObjectiveOptimizer inherits from ParetoOptimizer
        mo = MultiObjectiveOptimizer()
        assert hasattr(mo, 'pareto_optimize'), "MultiObjectiveOptimizer missing pareto_optimize method"
        print("[OK] MultiObjectiveOptimizer has pareto_optimize method")
        
        print("\n[PASS] Z3 Prover: PASS")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Z3 Prover: FAIL - {e}")
        return False


def test_physics_validator():
    """Test Physics Validator modal analysis fix."""
    print("\n" + "=" * 60)
    print("TEST 2: Physics Validator Modal Analysis")
    print("=" * 60)
    
    try:
        from physics_validator_real import RealFiniteElementAnalysis, MeshGenerator
        import numpy as np
        
        # Test RealFiniteElementAnalysis has modal_analysis method
        rfea = RealFiniteElementAnalysis()
        assert hasattr(rfea, 'modal_analysis'), "RealFiniteElementAnalysis missing modal_analysis method"
        print("[OK] RealFiniteElementAnalysis has modal_analysis method")
        
        # Test MeshGenerator is available
        assert hasattr(MeshGenerator, 'generate_2d_rectangular_mesh'), "MeshGenerator missing methods"
        print("[OK] MeshGenerator has generate_2d_rectangular_mesh method")
        
        # Test modal analysis with simple mesh (eigenvalue handling fix)
        mesh = MeshGenerator.generate_2d_rectangular_mesh(1.0, 1.0, 2, 2)
        result = rfea.modal_analysis(
            mesh=mesh,
            E=200e9,
            rho=7850,
            nu=0.3,
            thickness=0.01,
            fixed_nodes=[0],
            n_modes=3
        )
        
        assert 'natural_frequencies' in result, "Modal analysis missing natural_frequencies"
        assert isinstance(result['natural_frequencies'], list), "natural_frequencies should be a list"
        print(f"[OK] Modal analysis returns natural_frequencies: {len(result['natural_frequencies'])} modes")
        
        print("\n[PASS] Physics Validator: PASS")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Physics Validator: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sop_generator():
    """Test SOP Generator generate_sop fix."""
    print("\n" + "=" * 60)
    print("TEST 3: SOP Generator generate_sop")
    print("=" * 60)
    
    try:
        from sop_generator_real import RealSOPGenerator
        
        # Test RealSOPGenerator has generate_sop method
        gen = RealSOPGenerator()
        assert hasattr(gen, 'generate_sop'), "RealSOPGenerator missing generate_sop method"
        print("[OK] RealSOPGenerator has generate_sop method")
        
        # Test format methods exist
        assert hasattr(gen, '_format_markdown'), "RealSOPGenerator missing _format_markdown method"
        print("[OK] RealSOPGenerator has _format_markdown method")
        
        assert hasattr(gen, '_format_html'), "RealSOPGenerator missing _format_html method"
        print("[OK] RealSOPGenerator has _format_html method")
        
        # Test async generate_sop works
        async def test_async():
            invention_spec = {
                'name': 'Test Invention',
                'manufacturing': {
                    'material': 'steel',
                    'features': ['hole', 'slot'],
                    'tolerances': {'length': 0.1}
                },
                'equipment': ['CNC Mill', 'Lathe']
            }
            
            result = await gen.generate_sop(invention_spec, format='markdown')
            assert isinstance(result, str), "generate_sop should return string"
            assert len(result) > 0, "generate_sop should return non-empty string"
            return result
        
        result = asyncio.get_event_loop().run_until_complete(test_async())
        print(f"[OK] generate_sop returns string of length {len(result)}")
        
        print("\n[PASS] SOP Generator: PASS")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] SOP Generator: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_oneke_adapter():
    """Test OneKE Adapter _call_oneke fix."""
    print("\n" + "=" * 60)
    print("TEST 4: OneKE Adapter _call_oneke")
    print("=" * 60)
    
    try:
        from integrations.oneke.adapter import OneKEAdapter
        
        # Test OneKEAdapter has _call_oneke method
        adapter = OneKEAdapter(allow_fallback=True)
        assert hasattr(adapter, '_call_oneke'), "OneKEAdapter missing _call_oneke method"
        print("[OK] OneKEAdapter has _call_oneke method")
        
        # Test _call_oneke is callable
        assert callable(getattr(adapter, '_call_oneke')), "_call_oneke should be callable"
        print("[OK] _call_oneke is callable")
        
        print("\n[PASS] OneKE Adapter: PASS")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] OneKE Adapter: FAIL - {e}")
        return False


def test_deepke_setup():
    """Test DeepKE Setup script exists."""
    print("\n" + "=" * 60)
    print("TEST 5: DeepKE Setup Script")
    print("=" * 60)
    
    try:
        setup_script = Path("setup_deepke_fixed.py")
        assert setup_script.exists(), "setup_deepke_fixed.py not found"
        print("[OK] setup_deepke_fixed.py exists")
        
        # Test script has required functions
        import setup_deepke_fixed
        assert hasattr(setup_deepke_fixed, 'install_deepke_isolated'), "Missing install_deepke_isolated function"
        print("[OK] install_deepke_isolated function exists")
        
        assert hasattr(setup_deepke_fixed, 'activate_deepke'), "Missing activate_deepke function"
        print("[OK] activate_deepke function exists")
        
        assert hasattr(setup_deepke_fixed, 'verify_deepke'), "Missing verify_deepke function"
        print("[OK] verify_deepke function exists")
        
        print("\n[PASS] DeepKE Setup: PASS")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] DeepKE Setup: FAIL - {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRUE 100% FIXES VERIFICATION")
    print("=" * 60)
    print("\nVerifying all fixes for TRUE 100% completion...")
    
    results = []
    
    results.append(("Z3 Prover", test_z3_prover()))
    results.append(("Physics Validator", test_physics_validator()))
    results.append(("SOP Generator", test_sop_generator()))
    results.append(("OneKE Adapter", test_oneke_adapter()))
    results.append(("DeepKE Setup", test_deepke_setup()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name:<25} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n*** ALL TESTS PASSED - TRUE 100% ACHIEVED! ***")
        return 0
    else:
        print(f"\n*** {total - passed} test(s) failed ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
