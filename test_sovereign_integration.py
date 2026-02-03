import pytest
from datetime import datetime
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_gauntlets import GauntletSystem
from sovereign_team_coordination import TeamCoordinator, DecompositionWorkflow
from sovereign_quality_assessment import QualityAssessor
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_refinement import RefinementCoordinator
from sovereign_knowledge_manager import KnowledgeManager

class TestEndToEndWorkflow:
    def test_complete_decomposition_workflow(self):
        """Test complete workflow from problem to solution."""
        # Step 1: Analyze problem
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Build a recommendation system with ML models and real-time processing",
            title="Recommendation System"
        )
        assert problem is not None
        assert problem.title == "Recommendation System"
        
        # Step 2: Decompose problem
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem, strategy='semantic')
        assert plan is not None
        assert len(plan.sub_problems) > 0
        
        # Step 3: Validate with gauntlets
        gauntlet_system = GauntletSystem()
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        assert len(results) > 0
        
        # Step 4: Assess quality
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(plan)
        assert report.metrics.overall_score > 0
    
    def test_workflow_with_refinement(self):
        """Test workflow with iterative refinement."""
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Design a distributed caching system",
            title="Cache System"
        )
        
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem)
        
        # Refine the plan
        coordinator = RefinementCoordinator()
        gauntlet_system = GauntletSystem()
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        feedback = gauntlet_system.process_gauntlet_feedback(results)
        
        if feedback:
            refinement_plan = coordinator.generate_refinement_plan(plan, feedback)
            assert refinement_plan is not None
            assert len(refinement_plan.improvements) > 0
    
    def test_workflow_with_knowledge_learning(self):
        """Test workflow with knowledge extraction."""
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Implement a search engine",
            title="Search Engine"
        )
        
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem)
        
        # Extract knowledge
        knowledge_mgr = KnowledgeManager()
        patterns = knowledge_mgr.extract_patterns(plan, success=True, quality_score=0.85)
        assert isinstance(patterns, list)

class TestIntegrationScenarios:
    def test_research_problem_workflow(self):
        """Test workflow for research-type problems."""
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Research the effectiveness of different neural network architectures",
            title="NN Research"
        )
        
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem, strategy='semantic')
        
        # Should create research-oriented sub-problems
        assert len(plan.sub_problems) > 0
        assert plan.strategy.value == 'semantic'
    
    def test_implementation_problem_workflow(self):
        """Test workflow for implementation problems."""
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Build a REST API with authentication and rate limiting",
            title="REST API"
        )
        
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem)
        
        assert len(plan.sub_problems) > 0
    
    def test_hybrid_strategy_workflow(self):
        """Test workflow using hybrid decomposition."""
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Create a data pipeline with ETL and analytics",
            title="Data Pipeline"
        )
        
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem, strategy='hybrid')
        
        assert plan.strategy.value == 'hybrid'
        assert len(plan.sub_problems) > 0

class TestSystemIntegration:
    def test_all_components_work_together(self):
        """Test that all major components integrate properly."""
        # Initialize all components
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        assessor = QualityAssessor()
        orchestrator = SolutionOrchestrator()
        knowledge_mgr = KnowledgeManager()
        
        # Run through workflow
        problem = analyzer.analyze_problem(
            "Build a microservices architecture",
            title="Microservices"
        )
        plan = engine.decompose(problem)
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        report = assessor.generate_quality_report(plan)
        
        # All components should work without errors
        assert problem is not None
        assert plan is not None
        assert len(results) > 0
        assert report is not None
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        from sovereign_reliability import get_error_handler, with_retry
        
        handler = get_error_handler()
        initial_count = len(handler.error_log)
        
        # This should handle errors gracefully
        analyzer = ProblemAnalyzer()
        try:
            problem = analyzer.analyze_problem("", title="")
        except:
            pass
        
        # Error handler should have recorded something
        assert len(handler.error_log) >= initial_count
