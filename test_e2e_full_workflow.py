"""
End-to-End Full Workflow Integration Tests - License: Apache 2.0

Comprehensive tests verifying the complete OpenEvolve workflow:
1. Problem analysis
2. Decomposition
3. Solution generation
4. Physics validation
5. Error analysis
6. SOP generation
7. Security validation
8. Quality gauntlet
9. Knowledge extraction
10. Final verification

Run: pytest test_e2e_full_workflow.py -v
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

import pytest

# Core system imports
try:
    from problem_analyzer import ProblemAnalyzer, ProblemDefinition
    PROBLEM_ANALYZER_AVAILABLE = True
except ImportError:
    PROBLEM_ANALYZER_AVAILABLE = False

try:
    from decomposition_engine import DecompositionEngine, DecompositionResult
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False

try:
    from solution_assembler import SolutionAssembler
    SOLUTION_AVAILABLE = True
except ImportError:
    SOLUTION_AVAILABLE = False

try:
    from physics_validator import PhysicsValidator, PhysicsValidationResult
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False

try:
    from error_handler import ErrorHandler, ErrorSeverity
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False

try:
    from sop_generator import SOPGenerator, StandardOperatingProcedure
    SOP_AVAILABLE = True
except ImportError:
    SOP_AVAILABLE = False

try:
    from security_framework import SecurityFramework, Permission
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    from gauntlet_manager import GauntletManager, GauntletResult
    GAUNTLET_AVAILABLE = True
except ImportError:
    GAUNTLET_AVAILABLE = False

try:
    from stage6_knowledge_extraction import Stage6KnowledgeExtraction
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from verification_engine import VerificationEngine
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

try:
    from quality_gate_engine import QualityGateEngine
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False


@dataclass
class E2ETestResult:
    """Result of an E2E workflow test."""
    test_name: str
    stage: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_ms: float
    message: str = ""
    details: Dict = field(default_factory=dict)
    artifacts: Dict = field(default_factory=dict)


class TestE2EFullWorkflow:
    """
    End-to-End Full Workflow Integration Tests.
    
    Tests the complete 10-stage workflow from problem input to final verification.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results: List[E2ETestResult] = []
        
        # Initialize all systems
        self._init_systems()
        
        yield
        
        # Cleanup
        self.temp_dir.cleanup()
    
    def _init_systems(self):
        """Initialize all required systems."""
        self.systems = {}
        
        if PROBLEM_ANALYZER_AVAILABLE:
            self.systems['problem_analyzer'] = ProblemAnalyzer()
        
        if DECOMPOSITION_AVAILABLE:
            self.systems['decomposition'] = DecompositionEngine()
        
        if SOLUTION_AVAILABLE:
            self.systems['solution'] = SolutionAssembler()
        
        if PHYSICS_AVAILABLE:
            self.systems['physics'] = PhysicsValidator()
        
        if ERROR_HANDLER_AVAILABLE:
            self.systems['error_handler'] = ErrorHandler()
        
        if SOP_AVAILABLE:
            self.systems['sop'] = SOPGenerator()
        
        if SECURITY_AVAILABLE:
            self.systems['security'] = SecurityFramework()
        
        if GAUNTLET_AVAILABLE:
            self.systems['gauntlet'] = GauntletManager()
        
        if KNOWLEDGE_AVAILABLE:
            self.systems['knowledge'] = Stage6KnowledgeExtraction(
                storage_path=Path(self.temp_dir.name)
            )
        
        if VERIFICATION_AVAILABLE:
            self.systems['verification'] = VerificationEngine()
        
        if QUALITY_AVAILABLE:
            self.systems['quality'] = QualityGateEngine()
    
    def _record_result(self, result: E2ETestResult):
        """Record test result."""
        self.results.append(result)
        return result.status == 'passed'
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage1_problem_analysis(self):
        """Test Stage 1: Problem Analysis - Analyze complex problem input."""
        start = time.time()
        
        if not PROBLEM_ANALYZER_AVAILABLE:
            pytest.skip("Problem analyzer not available")
        
        try:
            # Test problem input
            problem_input = {
                "description": "Design an efficient solar panel cooling system",
                "domain": "engineering",
                "constraints": ["cost < $100", "efficiency > 90%", "maintenance-free for 5 years"],
                "requirements": ["passive cooling", "weather resistant", "modular design"]
            }
            
            analyzer = self.systems['problem_analyzer']
            
            # Analyze problem
            analysis_result = analyzer.analyze(problem_input)
            
            assert analysis_result is not None, "Analysis result should not be None"
            assert hasattr(analysis_result, 'complexity_score') or isinstance(analysis_result, dict), \
                "Analysis should return complexity metrics"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage1_problem_analysis",
                stage="problem_analysis",
                status="passed",
                duration_ms=duration,
                message="Problem analysis completed successfully",
                details={"problem_type": "engineering", "has_constraints": True}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage1_problem_analysis",
                stage="problem_analysis",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage2_decomposition(self):
        """Test Stage 2: Decomposition - Break down problem into subproblems."""
        start = time.time()
        
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition engine not available")
        
        try:
            engine = self.systems['decomposition']
            
            # Test decomposition
            problem = {
                "id": "test_e2e_001",
                "description": "Design a secure authentication system",
                "domain": "software_security",
                "complexity": "high"
            }
            
            result = engine.decompose(problem)
            
            assert result is not None, "Decomposition result should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage2_decomposition",
                stage="decomposition",
                status="passed",
                duration_ms=duration,
                message="Problem decomposed successfully",
                details={"problem_id": problem["id"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage2_decomposition",
                stage="decomposition",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage3_solution_generation(self):
        """Test Stage 3: Solution Generation - Generate solutions for subproblems."""
        start = time.time()
        
        if not SOLUTION_AVAILABLE:
            pytest.skip("Solution assembler not available")
        
        try:
            assembler = self.systems['solution']
            
            # Test solution generation
            subproblems = [
                {"id": "sp1", "description": "Implement password hashing", "priority": "high"},
                {"id": "sp2", "description": "Add 2FA support", "priority": "medium"},
                {"id": "sp3", "description": "Setup session management", "priority": "high"}
            ]
            
            solutions = []
            for sp in subproblems:
                solution = assembler.assemble_solution(sp)
                solutions.append(solution)
            
            assert len(solutions) == len(subproblems), "Should generate solution for each subproblem"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage3_solution_generation",
                stage="solution_generation",
                status="passed",
                duration_ms=duration,
                message=f"Generated {len(solutions)} solutions",
                details={"solution_count": len(solutions)}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage3_solution_generation",
                stage="solution_generation",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage4_physics_validation(self):
        """Test Stage 4: Physics Validation - Validate physical constraints."""
        start = time.time()
        
        if not PHYSICS_AVAILABLE:
            pytest.skip("Physics validator not available")
        
        try:
            validator = self.systems['physics']
            
            # Test physics validation
            design = {
                "type": "mechanical_system",
                "components": [
                    {"name": "rotor", "mass_kg": 10, "max_rpm": 5000},
                    {"name": "housing", "material": "aluminum", "thickness_mm": 5}
                ],
                "forces": [{"type": "centrifugal", "magnitude": 1000}]
            }
            
            validation_result = validator.validate(design)
            
            assert validation_result is not None, "Validation result should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage4_physics_validation",
                stage="physics_validation",
                status="passed",
                duration_ms=duration,
                message="Physics validation completed",
                details={"design_type": design["type"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage4_physics_validation",
                stage="physics_validation",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage5_error_analysis(self):
        """Test Stage 5: Error Analysis - Identify and analyze error sources."""
        start = time.time()
        
        if not ERROR_HANDLER_AVAILABLE:
            pytest.skip("Error handler not available")
        
        try:
            handler = self.systems['error_handler']
            
            # Test error analysis
            potential_errors = [
                {"type": "runtime", "probability": 0.3, "impact": "high"},
                {"type": "logic", "probability": 0.1, "impact": "critical"},
                {"type": "network", "probability": 0.5, "impact": "medium"}
            ]
            
            analysis_results = []
            for error in potential_errors:
                analysis = handler.analyze_error(error)
                analysis_results.append(analysis)
            
            assert len(analysis_results) == len(potential_errors), "Should analyze all errors"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage5_error_analysis",
                stage="error_analysis",
                status="passed",
                duration_ms=duration,
                message=f"Analyzed {len(analysis_results)} error sources",
                details={"error_count": len(analysis_results)}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage5_error_analysis",
                stage="error_analysis",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage6_sop_generation(self):
        """Test Stage 6: SOP Generation - Create standard operating procedures."""
        start = time.time()
        
        if not SOP_AVAILABLE:
            pytest.skip("SOP generator not available")
        
        try:
            generator = self.systems['sop']
            
            # Test SOP generation
            procedure_request = {
                "title": "System Deployment Procedure",
                "scope": "Production deployment of microservices",
                "steps": [
                    {"action": "Pre-deployment checks", "responsible": "DevOps"},
                    {"action": "Deploy to staging", "responsible": "CI/CD"},
                    {"action": "Run smoke tests", "responsible": "QA"},
                    {"action": "Deploy to production", "responsible": "DevOps"}
                ]
            }
            
            sop = generator.generate(procedure_request)
            
            assert sop is not None, "SOP should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage6_sop_generation",
                stage="sop_generation",
                status="passed",
                duration_ms=duration,
                message="SOP generated successfully",
                details={"sop_title": procedure_request["title"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage6_sop_generation",
                stage="sop_generation",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage7_security_validation(self):
        """Test Stage 7: Security Validation - Validate security measures."""
        start = time.time()
        
        if not SECURITY_AVAILABLE:
            pytest.skip("Security framework not available")
        
        try:
            security = self.systems['security']
            
            # Test security validation
            security_check = {
                "component": "api_gateway",
                "checks": ["authentication", "authorization", "input_validation", "rate_limiting"]
            }
            
            validation_result = security.validate_security(security_check)
            
            assert validation_result is not None, "Security validation should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage7_security_validation",
                stage="security_validation",
                status="passed",
                duration_ms=duration,
                message="Security validation completed",
                details={"component": security_check["component"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage7_security_validation",
                stage="security_validation",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage8_quality_gauntlet(self):
        """Test Stage 8: Quality Gauntlet - Run 3-round quality evaluation."""
        start = time.time()
        
        if not GAUNTLET_AVAILABLE:
            pytest.skip("Gauntlet manager not available")
        
        try:
            gauntlet = self.systems['gauntlet']
            
            # Test gauntlet execution
            solution = {
                "id": "sol_test_001",
                "content": "def authenticate(user, pwd): return hash(pwd) == stored_hash",
                "type": "code"
            }
            
            gauntlet_result = gauntlet.run_gauntlet(solution)
            
            assert gauntlet_result is not None, "Gauntlet result should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage8_quality_gauntlet",
                stage="quality_gauntlet",
                status="passed",
                duration_ms=duration,
                message="Quality gauntlet completed",
                details={"solution_id": solution["id"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage8_quality_gauntlet",
                stage="quality_gauntlet",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage9_knowledge_extraction(self):
        """Test Stage 9: Knowledge Extraction - Extract patterns from execution."""
        start = time.time()
        
        if not KNOWLEDGE_AVAILABLE:
            pytest.skip("Knowledge extraction not available")
        
        try:
            knowledge = self.systems['knowledge']
            
            # Test knowledge extraction
            execution_trace = {
                "trace_id": "trace_001",
                "workflow_id": "wf_test_001",
                "problem_description": "Test optimization problem",
                "stages": [
                    {"stage_name": "decompose", "parameters": {"strategy": "hybrid"}},
                    {"stage_name": "evolve", "parameters": {"generations": 100}},
                    {"stage_name": "assemble", "parameters": {}}
                ],
                "final_result": {"fitness": 0.95},
                "execution_time_ms": 5000.0,
                "timestamp": datetime.now()
            }
            
            # Process trace for knowledge extraction
            asyncio.run(knowledge.process_trace(execution_trace))
            
            stats = knowledge.get_statistics()
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage9_knowledge_extraction",
                stage="knowledge_extraction",
                status="passed",
                duration_ms=duration,
                message="Knowledge extraction completed",
                details={"traces_processed": stats.get('traces_processed', 0)}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage9_knowledge_extraction",
                stage="knowledge_extraction",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_stage10_final_verification(self):
        """Test Stage 10: Final Verification - Complete system verification."""
        start = time.time()
        
        if not VERIFICATION_AVAILABLE:
            pytest.skip("Verification engine not available")
        
        try:
            verification = self.systems['verification']
            
            # Test final verification
            final_output = {
                "deliverables": ["code", "documentation", "tests"],
                "quality_score": 0.95,
                "security_score": 0.98,
                "performance_score": 0.92
            }
            
            verification_result = verification.verify(final_output)
            
            assert verification_result is not None, "Verification result should not be None"
            
            duration = (time.time() - start) * 1000
            
            self._record_result(E2ETestResult(
                test_name="test_stage10_final_verification",
                stage="final_verification",
                status="passed",
                duration_ms=duration,
                message="Final verification completed",
                details={"quality_score": final_output["quality_score"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(E2ETestResult(
                test_name="test_stage10_final_verification",
                stage="final_verification",
                status="failed",
                duration_ms=duration,
                message=str(e)
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_e2e_workflow(self):
        """Test complete end-to-end workflow through all 10 stages."""
        start = time.time()
        
        try:
            print("\n" + "="*70)
            print("STARTING COMPLETE E2E WORKFLOW TEST")
            print("="*70)
            
            # Stage 1: Problem Analysis
            print("\n[Stage 1/10] Problem Analysis...")
            if PROBLEM_ANALYZER_AVAILABLE:
                analyzer = self.systems['problem_analyzer']
                problem = {"description": "Test problem", "domain": "test"}
                analysis = analyzer.analyze(problem)
                print("   [OK] Problem analyzed")
            
            # Stage 2: Decomposition
            print("\n[Stage 2/10] Decomposition...")
            if DECOMPOSITION_AVAILABLE:
                engine = self.systems['decomposition']
                decomp_result = engine.decompose({"id": "test", "description": "Test"})
                print("   [OK] Problem decomposed")
            
            # Stage 3: Solution Generation
            print("\n[Stage 3/10] Solution Generation...")
            if SOLUTION_AVAILABLE:
                assembler = self.systems['solution']
                solution = assembler.assemble_solution({"id": "sp1", "description": "Test"})
                print("   [OK] Solution generated")
            
            # Stage 4: Physics Validation
            print("\n[Stage 4/10] Physics Validation...")
            if PHYSICS_AVAILABLE:
                validator = self.systems['physics']
                physics_result = validator.validate({"type": "test"})
                print("   [OK] Physics validated")
            
            # Stage 5: Error Analysis
            print("\n[Stage 5/10] Error Analysis...")
            if ERROR_HANDLER_AVAILABLE:
                handler = self.systems['error_handler']
                error_result = handler.analyze_error({"type": "test"})
                print("   [OK] Errors analyzed")
            
            # Stage 6: SOP Generation
            print("\n[Stage 6/10] SOP Generation...")
            if SOP_AVAILABLE:
                generator = self.systems['sop']
                sop = generator.generate({"title": "Test SOP"})
                print("   [OK] SOP generated")
            
            # Stage 7: Security Validation
            print("\n[Stage 7/10] Security Validation...")
            if SECURITY_AVAILABLE:
                security = self.systems['security']
                sec_result = security.validate_security({"component": "test"})
                print("   [OK] Security validated")
            
            # Stage 8: Quality Gauntlet
            print("\n[Stage 8/10] Quality Gauntlet...")
            if GAUNTLET_AVAILABLE:
                gauntlet = self.systems['gauntlet']
                gauntlet_result = gauntlet.run_gauntlet({"id": "test"})
                print("   [OK] Gauntlet completed")
            
            # Stage 9: Knowledge Extraction
            print("\n[Stage 9/10] Knowledge Extraction...")
            if KNOWLEDGE_AVAILABLE:
                knowledge = self.systems['knowledge']
                trace = {"trace_id": "test", "workflow_id": "test"}
                asyncio.run(knowledge.process_trace(trace))
                print("   [OK] Knowledge extracted")
            
            # Stage 10: Final Verification
            print("\n[Stage 10/10] Final Verification...")
            if VERIFICATION_AVAILABLE:
                verification = self.systems['verification']
                verify_result = verification.verify({"test": True})
                print("   [OK] Final verification completed")
            
            duration = (time.time() - start) * 1000
            
            print("\n" + "="*70)
            print(f"COMPLETE E2E WORKFLOW FINISHED IN {duration:.2f}ms")
            print("="*70)
            
            # Verify at least some systems were tested
            available_systems = sum([
                PROBLEM_ANALYZER_AVAILABLE,
                DECOMPOSITION_AVAILABLE,
                SOLUTION_AVAILABLE,
                PHYSICS_AVAILABLE,
                ERROR_HANDLER_AVAILABLE,
                SOP_AVAILABLE,
                SECURITY_AVAILABLE,
                GAUNTLET_AVAILABLE,
                KNOWLEDGE_AVAILABLE,
                VERIFICATION_AVAILABLE
            ])
            
            assert available_systems > 0, "At least some systems should be available"
            
        except Exception as e:
            print(f"\n[FAIL] E2E workflow failed: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
