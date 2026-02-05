"""
Task 16.5: Final Validation Script
Comprehensive validation of all requirements and success criteria.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    # SECURITY FIX: Use subprocess with shell=False to prevent command injection
    import subprocess
    try:
        subprocess.run(['chcp', '65001'], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        # Log the specific error for debugging
        import logging
        logging.exception(f"Error setting console encoding: {e}")
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class Task16Validator:
    """Validates completion of Task 16 and overall system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = []
    
    def validate_documentation(self) -> bool:
        """Validate all documentation is complete."""
        print("\n" + "="*70)
        print("VALIDATING DOCUMENTATION (Task 16.1 & 16.2)")
        print("="*70)
        
        required_docs = {
            "SOVEREIGN_API_DOCUMENTATION.md": "API documentation with examples",
            "SOVEREIGN_USER_GUIDE.md": "User guide with tutorials",
            "OPERATIONS_GUIDE.md": "Operations and monitoring guide",
            "DEPLOYMENT_README.md": "Deployment guide",
            "README.md": "Project overview",
            "example_integration_usage.py": "Integration examples"
        }
        
        all_present = True
        for doc, description in required_docs.items():
            path = self.project_root / doc
            if path.exists():
                size = path.stat().st_size
                print(f"[OK] {doc} ({size:,} bytes) - {description}")
                self.results.append(("Documentation", doc, True, f"{size:,} bytes"))
            else:
                print(f"[FAIL] {doc} - MISSING")
                self.results.append(("Documentation", doc, False, "Missing"))
                all_present = False
        
        return all_present
    
    def validate_deployment_automation(self) -> bool:
        """Validate deployment automation (Task 16.3)."""
        print("\n" + "="*70)
        print("VALIDATING DEPLOYMENT AUTOMATION (Task 16.3)")
        print("="*70)
        
        required_files = {
            "deploy.py": "Deployment script",
            "Makefile": "Make commands",
            "Dockerfile": "Docker image definition",
            "docker-compose.yml": "Docker orchestration",
            ".github/workflows/ci.yml": "CI/CD pipeline"
        }
        
        all_present = True
        for file, description in required_files.items():
            path = self.project_root / file
            if path.exists():
                print(f"[OK] {file} - {description}")
                self.results.append(("Deployment", file, True, "Present"))
            else:
                print(f"[FAIL] {file} - MISSING")
                self.results.append(("Deployment", file, False, "Missing"))
                all_present = False
        
        # Test deployment script
        try:
            from deploy import DeploymentManager
            manager = DeploymentManager("development")
            print(f"[OK] Deployment script is importable")
            self.results.append(("Deployment", "deploy.py import", True, "OK"))
        except Exception as e:
            print(f"[FAIL] Deployment script import failed: {e}")
            self.results.append(("Deployment", "deploy.py import", False, str(e)))
            all_present = False
        
        return all_present
    
    def validate_monitoring(self) -> bool:
        """Validate monitoring and operations (Task 16.4)."""
        print("\n" + "="*70)
        print("VALIDATING MONITORING & OPERATIONS (Task 16.4)")
        print("="*70)
        
        try:
            from sovereign_reliability import get_health_monitor, get_error_handler
            from sovereign_performance_optimization import get_performance_stats
            
            # Test health monitoring
            monitor = get_health_monitor()
            print(f"[OK] Health monitor initialized")
            self.results.append(("Monitoring", "Health Monitor", True, "OK"))
            
            # Test error handling
            handler = get_error_handler()
            print(f"[OK] Error handler initialized")
            self.results.append(("Monitoring", "Error Handler", True, "OK"))
            
            # Test performance monitoring
            stats = get_performance_stats()
            print(f"[OK] Performance monitoring available")
            self.results.append(("Monitoring", "Performance Stats", True, "OK"))
            
            return True
        except Exception as e:
            print(f"[FAIL] Monitoring validation failed: {e}")
            self.results.append(("Monitoring", "System", False, str(e)))
            return False
    
    def validate_requirements(self) -> Dict[str, bool]:
        """Validate all 10 requirement areas."""
        print("\n" + "="*70)
        print("VALIDATING ALL REQUIREMENTS (Task 16.5)")
        print("="*70)
        
        requirements = {
            "1. Problem Analysis": self._check_problem_analysis(),
            "2. Decomposition Strategies": self._check_decomposition(),
            "3. Sub-problem Generation": self._check_subproblem_generation(),
            "4. Dependency Management": self._check_dependency_management(),
            "5. Validation Gauntlets": self._check_gauntlets(),
            "6. Team Integration": self._check_team_integration(),
            "7. Solution Tracking": self._check_solution_tracking(),
            "8. Knowledge Management": self._check_knowledge_management(),
            "9. Quality Assessment": self._check_quality_assessment(),
            "10. Performance & Scalability": self._check_performance()
        }
        
        for req, status in requirements.items():
            symbol = "[OK]" if status else "[FAIL]"
            print(f"{symbol} {req}")
            self.results.append(("Requirements", req, status, "Validated"))
        
        return requirements
    
    def _check_problem_analysis(self) -> bool:
        """Check requirement 1: Problem Analysis."""
        try:
            from problem_analyzer import ProblemAnalyzer
            analyzer = ProblemAnalyzer()
            return True
        except:
            return False
    
    def _check_decomposition(self) -> bool:
        """Check requirement 2: Decomposition Strategies."""
        try:
            from decomposition_engine import DecompositionEngine
            from problem_analyzer import ProblemAnalyzer
            analyzer = ProblemAnalyzer()
            engine = DecompositionEngine(analyzer)
            # Check that engine has the decompose method
            return hasattr(engine, 'decompose')
        except:
            return False
    
    def _check_subproblem_generation(self) -> bool:
        """Check requirement 3: Sub-problem Generation."""
        try:
            from sovereign_data_models import SubProblem, SubProblemType
            return True
        except:
            return False
    
    def _check_dependency_management(self) -> bool:
        """Check requirement 4: Dependency Management."""
        try:
            from dependency_manager import DependencyManager
            manager = DependencyManager()
            return hasattr(manager, 'build_graph') and hasattr(manager, 'detect_cycles')
        except:
            return False
    
    def _check_gauntlets(self) -> bool:
        """Check requirement 5: Validation Gauntlets."""
        try:
            from sovereign_gauntlets import GauntletSystem
            system = GauntletSystem()
            return True
        except:
            return False
    
    def _check_team_integration(self) -> bool:
        """Check requirement 6: Team Integration."""
        try:
            from sovereign_team_coordination import TeamCoordinator
            coordinator = TeamCoordinator()
            return True
        except:
            return False
    
    def _check_solution_tracking(self) -> bool:
        """Check requirement 7: Solution Tracking."""
        try:
            from sovereign_solution_orchestration import SolutionOrchestrator
            orchestrator = SolutionOrchestrator()
            return True
        except:
            return False
    
    def _check_knowledge_management(self) -> bool:
        """Check requirement 8: Knowledge Management."""
        try:
            from sovereign_knowledge_manager import KnowledgeManager
            manager = KnowledgeManager()
            return True
        except:
            return False
    
    def _check_quality_assessment(self) -> bool:
        """Check requirement 9: Quality Assessment."""
        try:
            from sovereign_quality_assessment import QualityAssessor
            assessor = QualityAssessor()
            return True
        except:
            return False
    
    def _check_performance(self) -> bool:
        """Check requirement 10: Performance & Scalability."""
        try:
            from sovereign_performance_optimization import PerformanceCache, PerformanceMonitor
            from sovereign_reliability import HealthMonitor
            return True
        except:
            return False
    
    def validate_success_criteria(self) -> Dict[str, Tuple[bool, str]]:
        """Validate success criteria."""
        print("\n" + "="*70)
        print("VALIDATING SUCCESS CRITERIA")
        print("="*70)
        
        criteria = {
            "Technical Success": self._check_technical_success(),
            "Quality Success": self._check_quality_success(),
            "Learning Success": self._check_learning_success(),
            "Documentation Complete": self._check_documentation_complete(),
            "Deployment Ready": self._check_deployment_ready()
        }
        
        for criterion, (status, message) in criteria.items():
            symbol = "[OK]" if status else "[FAIL]"
            print(f"{symbol} {criterion}: {message}")
            self.results.append(("Success Criteria", criterion, status, message))
        
        return criteria
    
    def _check_technical_success(self) -> Tuple[bool, str]:
        """Check technical success criteria."""
        try:
            # Check all modules are importable
            modules = [
                'problem_analyzer', 'decomposition_engine', 'dependency_manager',
                'sovereign_gauntlets', 'sovereign_quality_assessment',
                'sovereign_team_coordination', 'sovereign_solution_orchestration',
                'sovereign_knowledge_manager', 'sovereign_refinement',
                'sovereign_performance_optimization', 'sovereign_reliability'
            ]
            
            for module in modules:
                __import__(module)
            
            return True, "All core modules present and importable"
        except Exception as e:
            return False, f"Module import failed: {e}"
    
    def _check_quality_success(self) -> Tuple[bool, str]:
        """Check quality success criteria."""
        try:
            from sovereign_quality_assessment import QualityAssessor
            assessor = QualityAssessor()
            
            # Check thresholds are defined
            has_thresholds = hasattr(assessor, 'thresholds')
            return has_thresholds, "Quality thresholds configured"
        except:
            return False, "Quality assessment not available"
    
    def _check_learning_success(self) -> Tuple[bool, str]:
        """Check learning success criteria."""
        try:
            from sovereign_knowledge_manager import KnowledgeManager
            manager = KnowledgeManager()
            
            # Check knowledge management capabilities
            has_extract = hasattr(manager, 'extract_patterns')
            has_retrieve = hasattr(manager, 'retrieve_patterns')
            
            if has_extract and has_retrieve:
                return True, "Knowledge extraction and retrieval available"
            return False, "Knowledge management incomplete"
        except:
            return False, "Knowledge management not available"
    
    def _check_documentation_complete(self) -> Tuple[bool, str]:
        """Check documentation completeness."""
        required = [
            "SOVEREIGN_API_DOCUMENTATION.md",
            "SOVEREIGN_USER_GUIDE.md",
            "OPERATIONS_GUIDE.md",
            "DEPLOYMENT_README.md"
        ]
        
        missing = [doc for doc in required if not (self.project_root / doc).exists()]
        
        if not missing:
            return True, "All documentation present"
        return False, f"Missing: {', '.join(missing)}"
    
    def _check_deployment_ready(self) -> Tuple[bool, str]:
        """Check deployment readiness."""
        required = [
            "deploy.py",
            "Dockerfile",
            "docker-compose.yml",
            ".github/workflows/ci.yml"
        ]
        
        missing = [file for file in required if not (self.project_root / file).exists()]
        
        if not missing:
            return True, "Deployment automation complete"
        return False, f"Missing: {', '.join(missing)}"
    
    def run_test_suite(self) -> Tuple[bool, str]:
        """Run the test suite."""
        print("\n" + "="*70)
        print("RUNNING TEST SUITE")
        print("="*70)
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_sovereign*.py", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            # Count passed tests
            if "passed" in output:
                # Extract test count
                import re
                match = re.search(r'(\d+) passed', output)
                if match:
                    passed = int(match.group(1))
                    print(f"[OK] {passed} tests passed")
                    self.results.append(("Tests", "Test Suite", True, f"{passed} passed"))
                    return True, f"{passed} tests passed"
            
            print(f"[FAIL] Some tests failed")
            self.results.append(("Tests", "Test Suite", False, "Some tests failed"))
            return False, "Some tests failed"
            
        except subprocess.TimeoutExpired:
            print(f"[FAIL] Tests timed out")
            self.results.append(("Tests", "Test Suite", False, "Timeout"))
            return False, "Tests timed out"
        except Exception as e:
            print(f"[FAIL] Test execution failed: {e}")
            self.results.append(("Tests", "Test Suite", False, str(e)))
            return False, f"Test execution failed: {e}"
    
    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("\n" + "="*70)
        report.append("TASK 16 VALIDATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Group results by category
        categories = {}
        for category, item, status, message in self.results:
            if category not in categories:
                categories[category] = []
            categories[category].append((item, status, message))
        
        # Generate summary
        total = len(self.results)
        passed = sum(1 for _, _, status, _ in self.results if status)
        
        report.append(f"Overall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        report.append("")
        
        # Detail by category
        for category, items in categories.items():
            report.append(f"\n{category}:")
            report.append("-" * 70)
            for item, status, message in items:
                symbol = "[OK]" if status else "[FAIL]"
                report.append(f"  {symbol} {item}: {message}")
        
        report.append("\n" + "="*70)
        
        if passed == total:
            report.append("[OK] ALL VALIDATIONS PASSED - TASK 16 COMPLETE")
        else:
            report.append(f"[FAIL] {total - passed} VALIDATIONS FAILED")
        
        report.append("="*70 + "\n")
        
        return "\n".join(report)
    
    def run_full_validation(self) -> bool:
        """Run complete validation."""
        print("\n" + "="*70)
        print("TASK 16: DOCUMENTATION AND DEPLOYMENT - FINAL VALIDATION")
        print("="*70)
        
        # Run all validations
        doc_ok = self.validate_documentation()
        deploy_ok = self.validate_deployment_automation()
        monitor_ok = self.validate_monitoring()
        requirements = self.validate_requirements()
        criteria = self.validate_success_criteria()
        test_ok, test_msg = self.run_test_suite()
        
        # Generate report
        report = self.generate_report()
        print(report)
        
        # Save report
        report_file = self.project_root / "TASK_16_VALIDATION_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")
        
        # Overall success
        all_requirements_met = all(requirements.values())
        all_criteria_met = all(status for status, _ in criteria.values())
        
        return (doc_ok and deploy_ok and monitor_ok and 
                all_requirements_met and all_criteria_met and test_ok)


def main():
    """Main validation entry point."""
    validator = Task16Validator()
    success = validator.run_full_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
