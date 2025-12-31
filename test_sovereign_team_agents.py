"""
Test Suite for Production-Ready Team Agents
"""

import pytest
import logging

from sovereign_data_models import ProblemDefinition, ProblemType, generate_id
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_agents import (
    RedTeamAgent, BlueTeamAgent, GoldTeamAgent, TeamCoordinator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRedTeamAgent:
    """Test Red Team adversarial critique."""
    
    def test_red_team_critique(self):
        """Test Red Team identifies issues."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Build a social media platform with user profiles, posts, and messaging",
            "Social Media Platform"
        )
        
        plan = engine.decompose(problem)
        
        red_team = RedTeamAgent()
        feedback = red_team.analyze(plan)
        
        logger.info(f"\nRed Team found {len(feedback)} issues:")
        for f in feedback:
            logger.info(f"  [{f.severity}] {f.content}")
        
        assert len(feedback) > 0
        assert all(f.source == "red_team" for f in feedback)


class TestBlueTeamAgent:
    """Test Blue Team constructive refinement."""
    
    def test_blue_team_refinement(self):
        """Test Blue Team provides improvements."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Create an API gateway for microservices with authentication and rate limiting",
            "API Gateway"
        )
        
        plan = engine.decompose(problem)
        
        blue_team = BlueTeamAgent()
        feedback = blue_team.analyze(plan)
        
        logger.info(f"\nBlue Team provided {len(feedback)} suggestions:")
        for f in feedback:
            logger.info(f"  [{f.severity}] {f.content}")
        
        assert len(feedback) > 0
        assert all(f.source == "blue_team" for f in feedback)


class TestGoldTeamAgent:
    """Test Gold Team quality evaluation."""
    
    def test_gold_team_evaluation(self):
        """Test Gold Team provides quality assessment."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Implement a caching layer for database queries with TTL and invalidation",
            "Database Cache"
        )
        
        plan = engine.decompose(problem)
        
        gold_team = GoldTeamAgent()
        feedback = gold_team.analyze(plan)
        
        logger.info(f"\nGold Team evaluation:")
        for f in feedback:
            logger.info(f"  {f.content}")
            if f.metadata:
                logger.info(f"  Metadata: {f.metadata}")
        
        assert len(feedback) > 0
        assert all(f.source == "gold_team" for f in feedback)


class TestTeamCoordinator:
    """Test team coordination."""
    
    def test_full_team_review(self):
        """Test coordinated review by all teams."""
        logger.info("\n" + "="*60)
        logger.info("FULL TEAM COORDINATION TEST")
        logger.info("="*60)
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            """Build a real-time analytics dashboard that processes streaming data,
            performs aggregations, and displays visualizations with sub-second latency.
            Must handle 100K events/second and support 1000 concurrent users.""",
            "Real-time Analytics Dashboard"
        )
        
        plan = engine.decompose(problem)
        
        coordinator = TeamCoordinator()
        team_results = coordinator.coordinate_review(plan)
        
        logger.info("\n=== Team Review Results ===")
        for team, feedback_list in team_results.items():
            logger.info(f"\n{team.upper()}: {len(feedback_list)} items")
            for f in feedback_list[:3]:  # Show first 3
                logger.info(f"  [{f.severity}] {f.content[:100]}...")
        
        # Get consolidated feedback
        all_feedback = coordinator.get_consolidated_feedback(team_results)
        logger.info(f"\nTotal feedback items: {len(all_feedback)}")
        
        # Get recommendation
        recommendation = coordinator.get_recommendation(team_results)
        logger.info(f"Overall recommendation: {recommendation.upper()}")
        
        logger.info("="*60)
        
        assert 'red_team' in team_results
        assert 'blue_team' in team_results
        assert 'gold_team' in team_results
        assert len(all_feedback) > 0
        assert recommendation in ['approve', 'revise', 'reject']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
