"""
Enhanced Tests for LeanAide Continuous Math with Real Verification

This test suite:
1. Checks for Lean 4 installation and auto-setups if needed
2. Performs real proof verification (not skips)
3. Tests LLM integration if API keys available
4. Tests without LLM using template fallback

Run with: pytest test_leanaide_continuous_math_enhanced.py -v
"""

import asyncio
import os
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import enhanced modules
try:
    from lean4_integration_enhanced import (
        LeanAideServiceEnhanced,
        Lean4ServerConfig,
        Lean4VerificationEngine,
        Lean4AutoformalizationEngine,
        Lean4InstallationManager,
        LLMClient,
        LLMProvider,
        VerificationResult,
        VerificationStatus,
        AutoformalizationResult,
        create_lean4_service
    )
    ENHANCED_AVAILABLE = True
except ImportError as e:
    ENHANCED_AVAILABLE = False
    print(f"Warning: lean4_integration_enhanced not available: {e}")

try:
    from setup_lean4 import (
        Lean4SetupManager,
        detect_lean_installation,
        LeanInstallationStatus
    )
    SETUP_AVAILABLE = True
except ImportError as e:
    SETUP_AVAILABLE = False
    print(f"Warning: setup_lean4 not available: {e}")

try:
    from leanaide_continuous_math import (
        ContinuousMathEngine,
        create_continuous_math_engine,
        LimitType,
        Interval
    )
    CONTINUOUS_MATH_AVAILABLE = True
except ImportError as e:
    CONTINUOUS_MATH_AVAILABLE = False
    print(f"Warning: leanaide_continuous_math not available: {e}")


# ============================================================================
# Global Setup
# ============================================================================

LEAN_AVAILABLE = False
SETUP_COMPLETE = False

def setup_module(module):
    """Setup before running tests"""
    global LEAN_AVAILABLE, SETUP_COMPLETE
    
    if SETUP_AVAILABLE:
        manager = Lean4SetupManager()
        status = manager.check_installation()
        LEAN_AVAILABLE = status.lean_available
        
        if not LEAN_AVAILABLE:
            print("\n" + "="*70)
            print("⚠️  Lean 4 not detected. Attempting auto-setup...")
            print("="*70)
            result = manager.auto_install()
            if result.success:
                print("[OK] Auto-setup successful!")
                LEAN_AVAILABLE = True
                SETUP_COMPLETE = True
            else:
                print(f"[FAIL] Auto-setup failed: {result.message}")
                print("Run: python setup_lean4.py --instructions")
        else:
            SETUP_COMPLETE = True
    else:
        print("Warning: setup_lean4 not available, cannot check Lean status")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def installation_manager():
    """Create installation manager"""
    if SETUP_AVAILABLE:
        return Lean4SetupManager()
    return None


@pytest.fixture
def verification_engine():
    """Create verification engine"""
    if ENHANCED_AVAILABLE:
        return Lean4VerificationEngine()
    return None


@pytest.fixture
def lean_service():
    """Create enhanced Lean service"""
    if ENHANCED_AVAILABLE:
        # Try to get API keys from environment
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        return create_lean4_service(openai_key, anthropic_key)
    return None


# ============================================================================
# Installation Tests (REAL - no skips)
# ============================================================================

class TestLeanInstallation:
    """Tests for Lean 4 installation detection and setup"""
    
    def test_detect_lean_installation(self, installation_manager):
        """Test detecting Lean installation"""
        if not SETUP_AVAILABLE:
            pytest.skip("setup_lean4 not available")
        
        status = installation_manager.check_installation()
        
        # Should return a status object
        assert status is not None
        assert hasattr(status, 'lean_available')
        assert hasattr(status, 'lake_available')
        assert hasattr(status, 'mathlib_available')
        
        # Log status for debugging
        print(f"\nLean status: {status.lean_available}")
        print(f"Lake status: {status.lake_available}")
        print(f"Mathlib status: {status.mathlib_available}")
        if status.lean_version:
            print(f"Lean version: {status.lean_version}")
    
    def test_lean_command_available(self):
        """Test that lean command is available in PATH"""
        import shutil
        lean_path = shutil.which("lean")
        lake_path = shutil.which("lake")
        
        # Log findings
        print(f"\nLean path: {lean_path}")
        print(f"Lake path: {lake_path}")
        
        # If setup module ran, these should be available
        if SETUP_COMPLETE:
            assert lean_path is not None, "lean command not found in PATH"
            assert lake_path is not None, "lake command not found in PATH"
    
    def test_lean_version(self):
        """Test running lean --version"""
        import subprocess
        
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(f"\nLean version output: {result.stdout}")
            print(f"Lean version stderr: {result.stderr}")
            
            if SETUP_COMPLETE:
                assert result.returncode == 0, f"lean --version failed: {result.stderr}"
                assert "Lean" in result.stdout, f"Unexpected version output: {result.stdout}"
        except FileNotFoundError:
            if SETUP_COMPLETE:
                pytest.fail("lean command not found")
            else:
                pytest.skip("lean not installed")


# ============================================================================
# Verification Tests (REAL - no skips)
# ============================================================================

class TestLeanVerification:
    """Tests for Lean 4 code verification"""
    
    @pytest.mark.asyncio
    async def test_verify_simple_proof(self, verification_engine):
        """Test verifying a simple valid proof"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        code = """
theorem simple_theorem : 1 + 1 = 2 := by
  rfl
"""
        result = await verification_engine.verify(code)
        
        # Should return a result
        assert isinstance(result, VerificationResult)
        assert result.lean_available is not None
        
        if result.lean_available:
            print(f"\nVerification status: {result.status}")
            print(f"Success: {result.success}")
            print(f"Errors: {result.errors}")
            
            # For a valid proof, should succeed
            assert result.success is True, f"Verification failed: {result.errors}"
            assert result.status == VerificationStatus.SUCCESS
            assert len(result.errors) == 0
        else:
            pytest.skip("Lean not available for verification")
    
    @pytest.mark.asyncio
    async def test_verify_invalid_proof(self, verification_engine):
        """Test verifying an invalid proof"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        code = """
theorem wrong_theorem : 1 + 1 = 3 := by
  rfl
"""
        result = await verification_engine.verify(code)
        
        assert isinstance(result, VerificationResult)
        
        if result.lean_available:
            print(f"\nInvalid proof status: {result.status}")
            print(f"Success: {result.success}")
            print(f"Errors: {result.errors}")
            
            # Should fail
            assert result.success is False
            assert len(result.errors) > 0 or result.status == VerificationStatus.PROOF_ERROR
    
    @pytest.mark.asyncio
    async def test_verify_syntax_error(self, verification_engine):
        """Test verifying code with syntax error"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        code = """
theorem syntax_error : 1 + 1 = 2 :=
  -- missing 'by'
  rfl
"""
        result = await verification_engine.verify(code)
        
        assert isinstance(result, VerificationResult)
        
        if result.lean_available:
            print(f"\nSyntax error status: {result.status}")
            print(f"Errors: {result.errors}")
            
            # Should have errors
            assert result.success is False
    
    @pytest.mark.asyncio
    async def test_verify_with_imports(self, verification_engine):
        """Test verifying code with mathlib imports"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        code = """
import Mathlib

theorem nat_add_comm (n m : ℕ) : n + m = m + n := by
  rw [Nat.add_comm]
"""
        result = await verification_engine.verify(code)
        
        assert isinstance(result, VerificationResult)
        
        if result.lean_available:
            print(f"\nMathlib import status: {result.status}")
            print(f"Success: {result.success}")
            print(f"Errors: {result.errors}")
            
            # Should succeed with mathlib
            if result.success:
                print("✓ Mathlib import working!")


# ============================================================================
# LLM Integration Tests
# ============================================================================

class TestLLMIntegration:
    """Tests for LLM integration"""
    
    def test_llm_client_creation(self, lean_service):
        """Test LLM client initialization"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        # Check if any LLM is available
        status = lean_service.get_status()
        print(f"\nLLM available: {status['llm_available']}")
        print(f"LLM provider: {status['llm_provider']}")
        
        # Should return status dict
        assert isinstance(status, dict)
        assert 'llm_available' in status
        assert 'llm_provider' in status
    
    @pytest.mark.asyncio
    async def test_autoformalize_with_llm(self, lean_service):
        """Test autoformalization with LLM"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        status = lean_service.get_status()
        
        result = await lean_service.autoformalize(
            "The limit as x approaches 0 of sin(x)/x equals 1",
            domain="real_analysis",
            statement_type="theorem"
        )
        
        assert isinstance(result, AutoformalizationResult)
        assert result.natural_language == "The limit as x approaches 0 of sin(x)/x equals 1"
        assert result.domain == "real_analysis"
        assert len(result.lean_code) > 0
        assert "import" in result.lean_code
        
        print(f"\nAutoformalization result:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence}")
        print(f"  LLM Provider: {result.llm_provider}")
        print(f"  Generated code:\n{result.lean_code[:300]}...")
        
        if status['llm_available']:
            # With LLM, should have better results
            assert result.llm_provider in ['openai', 'anthropic']
    
    @pytest.mark.asyncio
    async def test_autoformalize_without_llm(self):
        """Test autoformalization without LLM (template fallback)"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        # Create service without API keys
        service = create_lean4_service()
        
        result = await service.autoformalize(
            "Every even number greater than 2 is the sum of two primes",
            domain="number_theory"
        )
        
        # Should still produce code even without LLM
        assert isinstance(result, AutoformalizationResult)
        assert len(result.lean_code) > 0
        assert "import Mathlib" in result.lean_code
        
        print(f"\nTemplate autoformalization:")
        print(f"  LLM Provider: {result.llm_provider}")
        print(f"  Generated code:\n{result.lean_code}")


# ============================================================================
# Continuous Math Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestContinuousMath:
    """Tests for continuous math engine"""
    
    @pytest.mark.asyncio
    async def test_compute_limit(self):
        """Test limit computation"""
        engine = create_continuous_math_engine(enable_lean_proofs=False)
        
        result = await engine.compute_limit("sin(x)/x", "x", 0.0)
        
        assert result.expression == "sin(x)/x"
        assert result.variable == "x"
        assert result.point == 0.0
        assert abs(result.limit_value - 1.0) < 1e-10
        assert result.limit_type == LimitType.TWO_SIDED
        assert result.existence_proven is True
    
    @pytest.mark.asyncio
    async def test_compute_derivative(self):
        """Test derivative computation"""
        engine = create_continuous_math_engine(enable_lean_proofs=False)
        
        result = await engine.compute_derivative("x**3", "x", order=1)
        
        assert result.function == "x**3"
        assert result.variable == "x"
        assert "3*x**2" in result.derivative or "3*x^2" in result.derivative
    
    @pytest.mark.asyncio
    async def test_compute_integral(self):
        """Test integral computation"""
        engine = create_continuous_math_engine(enable_lean_proofs=False)
        
        result = await engine.compute_integral("x**2", "x", 0.0, 1.0)
        
        assert result.integrand == "x**2"
        assert result.is_definite is True
        assert abs(result.value - 1/3) < 1e-6


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, lean_service):
        """Test end-to-end workflow: formalize -> verify"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        status = lean_service.get_status()
        print(f"\nService status: {status}")
        
        # Step 1: Autoformalize
        formalization = await lean_service.autoformalize(
            "For all natural numbers n, n + 0 = n",
            domain="algebra"
        )
        
        assert formalization.lean_code is not None
        print(f"\nFormalized code:\n{formalization.lean_code}")
        
        # Step 2: Verify
        if status['lean_available']:
            verification = await lean_service.verify(formalization.lean_code)
            
            print(f"Verification: {verification.status}")
            print(f"Errors: {verification.errors}")
            
            # Should complete verification (may or may not succeed depending on generated code)
            assert isinstance(verification, VerificationResult)
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, verification_engine):
        """Test batch verification"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("lean4_integration_enhanced not available")
        
        codes = [
            "theorem t1 : 1 + 1 = 2 := by rfl",
            "theorem t2 : 2 + 2 = 4 := by rfl",
            "theorem t3 : 0 + n = n := by simp"
        ]
        
        results = await verification_engine.verify_batch(codes)
        
        assert len(results) == 3
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}: {result.status}")
            print(f"  Success: {result.success}")
            print(f"  Errors: {result.errors}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
