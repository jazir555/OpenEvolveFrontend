"""
Production-Ready Test Suite for Sovereign Decomposition System
Tests the upgraded LLM-powered components with real-world scenarios
"""

import pytest
import logging
from datetime import datetime

from sovereign_data_models import (
    ProblemDefinition, ProblemType, DomainContext, ComplexityScore,
    Constraint, SuccessCriterion, generate_id
)
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_gauntlets import GauntletSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProductionReadyProblemAnalyzer:
    """Test production-ready Problem Analyzer with LLM integration."""
    
    def test_llm_domain_extraction(self):
        """Test LLM-based domain extraction with real problem."""
        analyzer = ProblemAnalyzer()
        
        problem_text = """
        Build a distributed machine learning training system that can scale to 
        handle millions of data points across multiple GPU clusters. The system 
        must support real-time model updates, fault tolerance, and efficient 
        resource allocation. Integration with existing Kubernetes infrastructure 
        is required.
        """
        
        problem = analyzer.analyze_problem(problem_text, "Distributed ML Training System")
        
        # Verify domain extraction
        assert problem.domain_context.domain is not None
        assert problem.domain_context.domain != "general"
        logger.info(f"Extracted domain: {problem.domain_context.domain}")
        logger.info(f"Subdomain: {problem.domain_context.subdomain}")
        logger.info(f"Related domains: {problem.domain_context.related_domains}")
        
        # Should identify ML/distributed systems domain
        assert any(term in problem.domain_context.domain.lower() 
                  for term in ['machine_learning', 'distributed', 'systems', 'engineering'])
    
    def test_llm_complexity_assessment(self):
        """Test LLM-based complexity assessment."""
        analyzer = ProblemAnalyzer()
        
        # Simple problem
        simple_problem = analyzer.analyze_problem(
            "Create a simple TODO list application with add and delete functionality",
            "Simple TODO App"
        )
        
        # Complex problem
        complex_problem = analyzer.analyze_problem(
            """Design and implement a fault-tolerant distributed consensus algorithm 
            that maintains consistency across geographically distributed data centers 
            while handling network partitions, Byzantine failures, and achieving 
            sub-second latency for 99.99% of operations.""",
            "Distributed Consensus System"
        )
        
        logger.info(f"Simple problem complexity: {simple_problem.complexity_score.overall_complexity}")
        logger.info(f"Complex problem complexity: {complex_problem.complexity_score.overall_complexity}")
        
        # Complex problem should have higher complexity
        assert complex_problem.complexity_score.overall_complexity > simple_problem.complexity_score.overall_complexity
        assert complex_problem.complexity_score.overall_complexity >= 7.0
    
    def test_llm_constraint_identification(self):
        """Test LLM-based constraint identification."""
        analyzer = ProblemAnalyzer()
        
        problem_text = """
        Develop a healthcare data analytics platform that must:
        - Comply with HIPAA regulations
        - Complete within 6 months
        - Work within a $200,000 budget
        - Achieve 99.9% uptime
        - Integrate with existing Epic EHR system
        - Support 10,000 concurrent users
        """
        
        problem = analyzer.analyze_problem(problem_text, "Healthcare Analytics Platform")
        
        logger.info(f"Identified {len(problem.constraints)} constraints:")
        for c in problem.constraints:
            logger.info(f"  - {c.type}: {c.description} ({c.severity})")
        
        # Should identify multiple constraint types
        assert len(problem.constraints) >= 3
        
        # Should identify different constraint types
        constraint_types = {c.type for c in problem.constraints}
        assert len(constraint_types) >= 2  # At least 2 different types
    
    def test_success_criteria_generation(self):
        """Test success criteria generation."""
        analyzer = ProblemAnalyzer()
        
        problem = analyzer.analyze_problem(
            "Optimize database query performance to reduce average response time by 50%",
            "Database Optimization"
        )
        
        assert len(problem.success_criteria) > 0
        logger.info(f"Generated {len(problem.success_criteria)} success criteria:")
        for sc in problem.success_criteria:
            logger.info(f"  - {sc.description}")


class TestProductionReadyDecompositionEngine:
    """Test production-ready Decomposition Engine with LLM strategies."""
    
    def test_llm_semantic_decomposition(self):
        """Test LLM-based semantic decomposition."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem_text = """
        Build a comprehensive e-commerce platform with user authentication, 
        product catalog management, shopping cart, payment processing, 
        order tracking, and customer support chat. The system must handle 
        high traffic, ensure data security, and provide excellent user experience.
        """
        
        problem = analyzer.analyze_problem(problem_text, "E-Commerce Platform")
        plan = engine.decompose(problem, strategy='semantic')
        
        logger.info(f"Semantic decomposition created {len(plan.sub_problems)} sub-problems:")
        for i, sp in enumerate(plan.sub_problems, 1):
            logger.info(f"{i}. {sp.title} ({sp.type.value})")
            logger.info(f"   Priority: {sp.priority}, Effort: {sp.estimated_effort}h")
        
        # Should create multiple meaningful sub-problems
        assert len(plan.sub_problems) >= 3
        assert len(plan.sub_problems) <= 8
        
        # Sub-problems should have distinct titles
        titles = [sp.title for sp in plan.sub_problems]
        assert len(titles) == len(set(titles))  # All unique
        
        # Should have success criteria
        assert all(len(sp.success_criteria) > 0 for sp in plan.sub_problems)
    
    def test_llm_strategy_selection(self):
        """Test LLM-based strategy selection."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        # Simple problem - should select semantic
        simple_problem = analyzer.analyze_problem(
            "Create a blog website with posts and comments",
            "Simple Blog"
        )
        
        # Complex problem - should select complexity or hybrid
        complex_problem = analyzer.analyze_problem(
            """Design a real-time distributed stream processing system handling 
            millions of events per second with exactly-once semantics, 
            state management, and complex event processing across multiple 
            data centers with sub-100ms latency requirements.""",
            "Stream Processing System"
        )
        
        simple_strategy = engine.select_strategy(simple_problem)
        complex_strategy = engine.select_strategy(complex_problem)
        
        logger.info(f"Simple problem strategy: {simple_strategy}")
        logger.info(f"Complex problem strategy: {complex_strategy}")
        
        # Strategies should be valid
        assert simple_strategy in ['semantic', 'dependency', 'complexity', 'hybrid']
        assert complex_strategy in ['semantic', 'dependency', 'complexity', 'hybrid']
    
    def test_dependency_graph_construction(self):
        """Test dependency graph construction."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Implement a CI/CD pipeline with testing, building, and deployment stages",
            "CI/CD Pipeline"
        )
        
        plan = engine.decompose(problem)
        
        # Should have dependency graph
        assert plan.dependency_graph is not None
        assert plan.dependency_graph.nodes is not None
        assert plan.dependency_graph.execution_order is not None
        
        logger.info(f"Execution order: {len(plan.dependency_graph.execution_order)} steps")
        
        # Execution order should include all sub-problems
        assert len(plan.dependency_graph.execution_order) == len(plan.sub_problems)


class TestProductionReadyGauntlets:
    """Test production-ready Gauntlets with LLM validation."""
    
    def test_llm_coherence_validation(self):
        """Test LLM-based coherence validation."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        
        problem = analyzer.analyze_problem(
            "Build a mobile app for fitness tracking with workout logging and progress visualization",
            "Fitness Tracker App"
        )
        
        plan = engine.decompose(problem)
        results = gauntlet_system.run_decomposition_gauntlets(plan, ['coherence'])
        
        assert 'coherence' in results
        result = results['coherence']
        
        logger.info(f"Coherence validation: {'PASS' if result.passed else 'FAIL'}")
        logger.info(f"Score: {result.score:.2f}")
        logger.info(f"Feedback: {result.feedback}")
        
        # Should have meaningful feedback
        assert result.feedback is not None
        assert len(result.feedback) > 10
    
    def test_llm_completeness_validation(self):
        """Test LLM-based completeness validation."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        
        problem = analyzer.analyze_problem(
            """Create a customer relationship management (CRM) system with 
            contact management, sales pipeline tracking, email integration, 
            reporting dashboards, and mobile access.""",
            "CRM System"
        )
        
        plan = engine.decompose(problem)
        results = gauntlet_system.run_decomposition_gauntlets(plan, ['completeness'])
        
        assert 'completeness' in results
        result = results['completeness']
        
        logger.info(f"Completeness validation: {'PASS' if result.passed else 'FAIL'}")
        logger.info(f"Score: {result.score:.2f}")
        logger.info(f"Improvements: {result.improvements}")
        
        # Should provide improvements if not perfect
        if result.score < 1.0:
            assert len(result.improvements) > 0
    
    def test_all_gauntlets_integration(self):
        """Test all gauntlets working together."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        
        problem = analyzer.analyze_problem(
            "Develop a real-time collaborative document editing system like Google Docs",
            "Collaborative Editor"
        )
        
        plan = engine.decompose(problem)
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        
        logger.info("\n=== All Gauntlets Results ===")
        for name, result in results.items():
            logger.info(f"{name}: {'PASS' if result.passed else 'FAIL'} ({result.score:.2f})")
        
        # Should run all gauntlets
        assert len(results) == 4  # coherence, completeness, feasibility, dependency
        
        # Calculate overall quality
        overall_quality = gauntlet_system.get_overall_quality(results)
        logger.info(f"\nOverall Quality: {overall_quality:.2f}")
        
        assert 0.0 <= overall_quality <= 1.0


class TestEndToEndProductionWorkflow:
    """Test complete end-to-end production workflow."""
    
    def test_complete_decomposition_workflow(self):
        """Test complete workflow from problem to validated decomposition."""
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION-READY END-TO-END WORKFLOW TEST")
        logger.info("="*60)
        
        # Step 1: Analyze problem
        analyzer = ProblemAnalyzer()
        problem_text = """
        Design and implement a microservices-based inventory management system 
        for a large retail chain. The system must:
        - Track inventory across 500+ stores in real-time
        - Handle 10,000+ transactions per second
        - Provide predictive analytics for stock optimization
        - Integrate with existing POS systems
        - Support mobile apps for store managers
        - Ensure 99.99% uptime
        - Complete within 9 months with a team of 8 developers
        - Comply with data privacy regulations
        """
        
        logger.info("\nStep 1: Analyzing problem...")
        problem = analyzer.analyze_problem(problem_text, "Retail Inventory Management System")
        
        logger.info(f"  Domain: {problem.domain_context.domain}")
        logger.info(f"  Complexity: {problem.complexity_score.overall_complexity}/10")
        logger.info(f"  Constraints: {len(problem.constraints)}")
        
        # Step 2: Decompose problem
        logger.info("\nStep 2: Decomposing problem...")
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(problem)
        
        logger.info(f"  Strategy: {plan.strategy.value}")
        logger.info(f"  Sub-problems: {len(plan.sub_problems)}")
        
        for i, sp in enumerate(plan.sub_problems, 1):
            logger.info(f"    {i}. {sp.title}")
            logger.info(f"       Type: {sp.type.value}, Priority: {sp.priority}, Effort: {sp.estimated_effort}h")
        
        # Step 3: Validate with gauntlets
        logger.info("\nStep 3: Running validation gauntlets...")
        gauntlet_system = GauntletSystem()
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        
        for name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            logger.info(f"  {name}: {status} ({result.score:.2f})")
        
        overall_quality = gauntlet_system.get_overall_quality(results)
        all_passed = gauntlet_system.all_passed(results)
        
        logger.info(f"\n  Overall Quality: {overall_quality:.2f}")
        logger.info(f"  All Gauntlets Passed: {all_passed}")
        
        # Step 4: Generate feedback
        logger.info("\nStep 4: Processing feedback...")
        feedback_list = gauntlet_system.process_gauntlet_feedback(results)
        
        for feedback in feedback_list:
            logger.info(f"  [{feedback.severity}] {feedback.content}")
        
        logger.info("\n" + "="*60)
        logger.info("WORKFLOW COMPLETE")
        logger.info("="*60)
        
        # Assertions
        assert problem is not None
        assert plan is not None
        assert len(plan.sub_problems) >= 3
        assert len(results) == 4
        assert 0.0 <= overall_quality <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
