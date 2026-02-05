"""
Comprehensive Unit Tests for Decomposition Engine

Tests the problem decomposition engine including:
- Problem analysis
- Sub-problem extraction
- Dependency mapping
- Strategy selection
- Decomposition validation

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProblemAnalysis:
    """Test problem analysis functionality"""

    def test_problem_analyzer_creation(self):
        """Test ProblemAnalyzer initialization"""
        from decomposition_engine import ProblemAnalyzer
        
        analyzer = ProblemAnalyzer()
        assert analyzer is not None

    def test_analyze_problem(self):
        """Test problem analysis"""
        from decomposition_engine import ProblemAnalyzer
        
        analyzer = ProblemAnalyzer()
        
        result = analyzer.analyze(
            problem_text="Optimize a portfolio of stocks with constraints on risk and return"
        )
        
        assert result is not None
        assert hasattr(result, 'complexity')
        assert hasattr(result, 'domain')

    def test_complexity_assessment(self):
        """Test complexity scoring"""
        from decomposition_engine import ComplexityAssessor
        
        assessor = ComplexityAssessor()
        
        # Simple problem
        simple_score = assessor.assess("Calculate the sum of two numbers")
        assert simple_score < 0.5
        
        # Complex problem
        complex_score = assessor.assess(
            "Design and implement a distributed machine learning system "
            "with real-time processing, auto-scaling, and fault tolerance"
        )
        assert complex_score > 0.5


class TestSubProblemExtraction:
    """Test sub-problem extraction"""

    def test_sub_problem_creation(self):
        """Test SubProblem dataclass"""
        from decomposition_engine import SubProblem
        
        problem = SubProblem(
            id="sub_001",
            description="Optimize the objective function",
            complexity=0.7,
            dependencies=[]
        )
        
        assert problem.id == "sub_001"
        assert problem.complexity == 0.7

    def test_extract_sub_problems(self):
        """Test extracting sub-problems from main problem"""
        from decomposition_engine import SubProblemExtractor
        
        extractor = SubProblemExtractor()
        
        main_problem = (
            "Build a web application with user authentication, "
            "database integration, and real-time notifications"
        )
        
        sub_problems = extractor.extract(main_problem)
        
        assert isinstance(sub_problems, list)
        assert len(sub_problems) > 0


class TestDependencyMapping:
    """Test dependency mapping"""

    def test_dependency_graph_creation(self):
        """Test DependencyGraph initialization"""
        from decomposition_engine import DependencyGraph
        
        graph = DependencyGraph()
        assert graph is not None

    def test_add_dependency(self):
        """Test adding dependency between sub-problems"""
        from decomposition_engine import DependencyGraph
        
        graph = DependencyGraph()
        
        graph.add_node("problem_1", complexity=0.5)
        graph.add_node("problem_2", complexity=0.7)
        graph.add_edge("problem_1", "problem_2")
        
        assert graph.has_edge("problem_1", "problem_2")

    def test_detect_cycles(self):
        """Test cycle detection in dependencies"""
        from decomposition_engine import DependencyGraph
        
        graph = DependencyGraph()
        
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        # No cycle yet
        assert graph.has_cycle() == False
        
        # Add cycle
        graph.add_edge("C", "A")
        assert graph.has_cycle() == True

    def test_topological_sort(self):
        """Test topological sorting of dependencies"""
        from decomposition_engine import DependencyGraph
        
        graph = DependencyGraph()
        
        graph.add_node("init")
        graph.add_node("process")
        graph.add_node("finalize")
        
        graph.add_edge("init", "process")
        graph.add_edge("process", "finalize")
        
        sorted_nodes = graph.topological_sort()
        
        assert sorted_nodes.index("init") < sorted_nodes.index("process")
        assert sorted_nodes.index("process") < sorted_nodes.index("finalize")


class TestStrategySelection:
    """Test decomposition strategy selection"""

    def test_strategy_selector(self):
        """Test StrategySelector"""
        from decomposition_engine import StrategySelector
        
        selector = StrategySelector()
        
        # Should select appropriate strategy
        strategy = selector.select(
            problem_complexity=0.3,
            domain="optimization"
        )
        
        assert strategy is not None

    def test_semantic_decomposition(self):
        """Test semantic decomposition strategy"""
        from decomposition_engine import SemanticDecomposition
        
        strategy = SemanticDecomposition()
        
        result = strategy.decompose(
            problem_text="Solve the traveling salesman problem with constraints"
        )
        
        assert result is not None

    def test_structural_decomposition(self):
        """Test structural decomposition strategy"""
        from decomposition_engine import StructuralDecomposition
        
        strategy = StructuralDecomposition()
        
        result = strategy.decompose(
            problem_text="Build a multi-tier architecture with API, database, and frontend"
        )
        
        assert result is not None

    def test_hierarchical_decomposition(self):
        """Test hierarchical decomposition strategy"""
        from decomposition_engine import HierarchicalDecomposition
        
        strategy = HierarchicalDecomposition()
        
        result = strategy.decompose(
            problem_text="Design a large-scale system from high-level to implementation"
        )
        
        assert result is not None


class TestDecompositionValidation:
    """Test decomposition validation"""

    def test_validate_decomposition(self):
        """Test validating decomposition results"""
        from decomposition_engine import DecompositionValidator
        
        validator = DecompositionValidator()
        
        decomposition = {
            "sub_problems": [
                {"id": "1", "description": "Part 1"},
                {"id": "2", "description": "Part 2"}
            ],
            "dependencies": []
        }
        
        result = validator.validate(decomposition)
        
        assert isinstance(result, bool)

    def test_check_completeness(self):
        """Test completeness checking"""
        from decomposition_engine import CompletenessChecker
        
        checker = CompletenessChecker()
        
        # Complete decomposition
        complete = {
            "sub_problems": [
                {"id": "1", "description": "A", "solved": False},
                {"id": "2", "description": "B", "solved": False}
            ],
            "all_solved": False
        }
        
        assert checker.is_complete(complete) == False


class TestDecompositionResult:
    """Test decomposition result"""

    def test_decomposition_result_creation(self):
        """Test DecompositionResult dataclass"""
        from decomposition_engine import DecompositionResult
        
        result = DecompositionResult(
            success=True,
            sub_problems=[],
            execution_order=[],
            total_complexity=0.5
        )
        
        assert result.success == True
        assert result.total_complexity == 0.5

    def test_result_serialization(self):
        """Test result serialization"""
        from decomposition_engine import DecompositionResult
        
        result = DecompositionResult(
            success=True,
            sub_problems=[{"id": "1"}],
            execution_order=["1"],
            total_complexity=0.7
        )
        
        serialized = result.to_dict()
        
        assert isinstance(serialized, dict)
        assert serialized["success"] == True


class TestDecompositionConfig:
    """Test decomposition configuration"""

    def test_config_creation(self):
        """Test DecompositionConfig"""
        from decomposition_engine import DecompositionConfig
        
        config = DecompositionConfig(
            max_sub_problems=10,
            min_sub_problems=2,
            complexity_threshold=0.8,
            enable_parallel=True
        )
        
        assert config.max_sub_problems == 10
        assert config.enable_parallel == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
