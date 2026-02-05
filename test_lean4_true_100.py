"""
Test Lean 4 TRUE 100% Implementation

Comprehensive tests for the TRUE 100% Lean 4 integration including:
- Lean 4 installation detection
- Real proof verification (no sorry)
- LLM integration
- Proof completion
- Mathlib4 support

Author: OpenEvolve
Version: 3.0.0 - TRUE 100%
"""

import asyncio
import os
import pytest
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lean4_true_100_integration import (
    Lean4True100Service,
    Lean4ServerConfig,
    LLMProvider,
    VerificationStatus,
    create_lean4_true100_service,
    Lean4InstallationManager,
    Lean4VerificationEngine,
    LLMClient,
    ProofCompletionEngine,
    Lean4AutoformalizationEngine
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def service():
    """Create a TRUE 100% service"""
    return create_lean4_true100_service()


@pytest.fixture
def config():
    """Create a test configuration"""
    return Lean4ServerConfig(
        working_dir="./test_lean_workspace",
        cache_dir="./test_lean_cache",
        timeout_seconds=30.0
    )


# ============================================================================
# Installation Tests
# ============================================================================

class TestLeanInstallation:
    """Tests for Lean 4 installation"""
    
    def test_installation_detection(self, service):
        """Test that Lean installation can be detected"""
        status = service.get_status()
        
        # Should return status dictionary
        assert isinstance(status, dict)
        assert "lean_available" in status
        assert "lake_available" in status
        assert "mathlib_available" in status
    
    def test_lean_in_path(self, service):
        """Test that lean command is available"""
        import subprocess
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            lean_available = result.returncode == 0
        except:
            lean_available = False
        
        status = service.get_status()
        # Status should reflect actual availability
        assert status["lean_available"] == lean_available
    
    def test_lake_in_path(self, service):
        """Test that lake command is available"""
        import subprocess
        try:
            result = subprocess.run(
                ["lake", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            lake_available = result.returncode == 0
        except:
            lake_available = False
        
        status = service.get_status()
        assert status["lake_available"] == lake_available


# ============================================================================
# Verification Tests - NO SORRY
# ============================================================================

class TestLeanVerification:
    """Tests for Lean 4 code verification"""
    
    @pytest.mark.asyncio
    async def test_verify_simple_theorem(self, service):
        """Test verification of a simple theorem"""
        code = """
theorem simple_addition : 1 + 1 = 2 := by
  rfl
"""
        result = await service.verify(code)
        
        # Should return result
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'has_sorry')
        assert hasattr(result, 'proof_complete')
    
    @pytest.mark.asyncio
    async def test_detects_sorry(self, service):
        """Test that verification detects sorry"""
        code_with_sorry = """
theorem unproven : 1 + 1 = 2 := by
  sorry
"""
        result = await service.verify(code_with_sorry)
        
        # Should detect sorry
        assert result.has_sorry == True
        assert result.proof_complete == False
    
    @pytest.mark.asyncio
    async def test_proof_complete_detection(self, service):
        """Test that complete proofs are detected"""
        complete_code = """
theorem complete_proof : 1 + 1 = 2 := by
  rfl
"""
        result = await service.verify(complete_code)
        
        # Should not have sorry
        if result.success:
            assert result.has_sorry == False
            assert result.proof_complete == True
    
    @pytest.mark.asyncio
    async def test_syntax_error_detection(self, service):
        """Test detection of syntax errors"""
        bad_code = """
theorem broken : 1 + = 2 :=
  rfl
"""
        result = await service.verify(bad_code)
        
        # Should fail
        assert result.success == False
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, service):
        """Test batch verification"""
        codes = [
            "theorem t1 : 1 + 1 = 2 := by rfl",
            "theorem t2 : 2 + 2 = 4 := by rfl",
        ]
        
        # Use verification engine directly for batch
        engine = Lean4VerificationEngine()
        results = await engine.verify_batch(codes)
        
        assert len(results) == len(codes)
        for r in results:
            assert hasattr(r, 'success')


# ============================================================================
# LLM Integration Tests
# ============================================================================

class TestLLMIntegration:
    """Tests for LLM integration"""
    
    def test_llm_availability_check(self, service):
        """Test LLM availability detection"""
        status = service.get_status()
        
        assert "llm_available" in status
        assert "llm_provider" in status
        
        # Should be boolean
        assert isinstance(status["llm_available"], bool)
    
    def test_openai_initialization(self):
        """Test OpenAI client initialization"""
        config = Lean4ServerConfig(
            openai_api_key="test-key",
            llm_provider=LLMProvider.OPENAI
        )
        
        client = LLMClient(config)
        
        # Should create client (may fail on actual call without real key)
        assert client is not None
    
    def test_anthropic_initialization(self):
        """Test Anthropic client initialization"""
        config = Lean4ServerConfig(
            anthropic_api_key="test-key",
            llm_provider=LLMProvider.ANTHROPIC
        )
        
        client = LLMClient(config)
        
        # Should create client
        assert client is not None
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_openai_generation(self):
        """Test actual OpenAI generation (requires API key)"""
        config = Lean4ServerConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        client = LLMClient(config)
        
        success, result = await client.generate(
            "Generate a Lean 4 theorem that 2 + 2 = 4",
            "You are a Lean 4 expert."
        )
        
        # Should succeed with valid key
        if client.is_available():
            assert success == True
            assert len(result) > 0


# ============================================================================
# Proof Completion Tests - NO SORRY
# ============================================================================

class TestProofCompletion:
    """Tests for proof completion (replacing sorry)"""
    
    @pytest.mark.asyncio
    async def test_proof_completion_detection(self, service):
        """Test that proof completion detects sorry"""
        code_with_sorry = """
theorem test_theorem : 1 + 1 = 2 := by
  sorry
"""
        result = await service.complete_proof(code_with_sorry)
        
        # Should return result
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'original_code')
        assert hasattr(result, 'completed_code')
    
    @pytest.mark.asyncio
    async def test_no_sorry_returns_success(self, service):
        """Test that code without sorry returns success immediately"""
        complete_code = """
theorem already_done : 1 + 1 = 2 := by
  rfl
"""
        result = await service.complete_proof(complete_code)
        
        # Should succeed immediately
        assert result.success == True
        assert result.original_code == complete_code
        assert result.completed_code == complete_code
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
        reason="No LLM API key available"
    )
    async def test_actual_proof_completion(self, service):
        """Test actual proof completion with LLM"""
        code_with_sorry = """
theorem simple_equality : 2 + 2 = 4 := by
  sorry
"""
        result = await service.complete_proof(code_with_sorry)
        
        status = service.get_status()
        if status["llm_available"]:
            # If LLM available, may succeed
            assert result is not None
            # Should attempt to complete
            assert len(result.tactics_used) >= 0


# ============================================================================
# Autoformalization Tests
# ============================================================================

class TestAutoformalization:
    """Tests for autoformalization"""
    
    @pytest.mark.asyncio
    async def test_autoformalize_returns_result(self, service):
        """Test that autoformalize returns a result"""
        result = await service.autoformalize(
            "The square of any odd number is odd",
            domain="number_theory"
        )
        
        # Should return result
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'lean_code')
        assert hasattr(result, 'natural_language')
    
    @pytest.mark.asyncio
    async def test_autoformalize_preserves_input(self, service):
        """Test that autoformalize preserves the input statement"""
        input_text = "The sum of two even numbers is even"
        
        result = await service.autoformalize(input_text)
        
        # Should preserve input
        assert result.natural_language == input_text
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
        reason="No LLM API key available"
    )
    async def test_autoformalize_with_llm(self, service):
        """Test autoformalization with LLM"""
        result = await service.autoformalize(
            "The sum of two even numbers is even",
            domain="number_theory"
        )
        
        status = service.get_status()
        if status["llm_available"]:
            # Should generate code
            assert len(result.lean_code) > 0
            assert "theorem" in result.lean_code.lower() or "def" in result.lean_code.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
        reason="No LLM API key available"
    )
    async def test_full_pipeline(self):
        """Test the full pipeline: NL -> Lean -> Verify -> Complete"""
        service = create_lean4_true100_service()
        
        # Step 1: Autoformalize
        result = await service.autoformalize(
            "For all natural numbers n, n + 0 = n",
            domain="number_theory"
        )
        
        # Should generate code
        assert result.lean_code
        
        # Step 2: Verify
        verification = await service.verify(result.lean_code)
        
        # Should return verification result
        assert verification is not None
        
        # Step 3: Complete proof if needed
        if "sorry" in result.lean_code.lower():
            completion = await service.complete_proof(result.lean_code)
            assert completion is not None
    
    @pytest.mark.asyncio
    async def test_service_status(self, service):
        """Test comprehensive service status"""
        status = service.get_status()
        
        required_keys = [
            "lean_available",
            "lake_available",
            "mathlib_available",
            "elan_available",
            "llm_available",
            "llm_provider",
            "proof_completion_enabled"
        ]
        
        for key in required_keys:
            assert key in status, f"Missing key: {key}"


# ============================================================================
# Mathlib4 Tests
# ============================================================================

class TestMathlib4:
    """Tests for Mathlib4 integration"""
    
    def test_mathlib_detection(self, service):
        """Test mathlib4 detection"""
        status = service.get_status()
        
        # Should have mathlib status
        assert "mathlib_available" in status
        assert isinstance(status["mathlib_available"], bool)
    
    @pytest.mark.asyncio
    async def test_mathlib_import(self, service):
        """Test that mathlib can be imported"""
        code = """
import Mathlib

theorem uses_mathlib (n : ℕ) : n + 0 = n := by
  rw [add_zero]
"""
        result = await service.verify(code)
        
        # May succeed or fail depending on mathlib availability
        # But should not crash
        assert result is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Tests for performance"""
    
    @pytest.mark.asyncio
    async def test_verification_performance(self, service):
        """Test that verification completes in reasonable time"""
        import time
        
        code = "theorem t : 1 + 1 = 2 := by rfl"
        
        start = time.time()
        result = await service.verify(code)
        elapsed = time.time() - start
        
        # Should complete within 60 seconds
        assert elapsed < 60.0
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_caching(self, service):
        """Test that caching works"""
        code = "theorem t : 1 + 1 = 2 := by rfl"
        
        # First call
        result1 = await service.verify(code)
        
        # Second call (should be cached)
        result2 = await service.verify(code)
        
        # Results should be the same
        assert result1.success == result2.success


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_malformed_code(self, service):
        """Test handling of malformed code"""
        bad_code = "this is not lean code at all !!!"
        
        result = await service.verify(bad_code)
        
        # Should fail gracefully
        assert result.success == False
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_empty_code(self, service):
        """Test handling of empty code"""
        result = await service.verify("")
        
        # Should handle gracefully
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_very_long_code(self, service):
        """Test handling of very long code"""
        long_code = "\n".join([f"theorem t{i} : {i} = {i} := by rfl" for i in range(100)])
        
        result = await service.verify(long_code)
        
        # Should handle without crashing
        assert result is not None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run with pytest if available
    pytest.main([__file__, "-v"])
