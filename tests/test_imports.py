"""
Test file to verify imports work correctly
"""
import sys
import os
from pathlib import Path
import pytest

# Add parent directory (root) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work correctly"""
    from content_analyzer import ContentAnalyzer
    from red_team import RedTeam
    from blue_team import BlueTeam
    from evaluator_team import EvaluatorTeam
    from quality_assessment import QualityAssessmentEngine

    # Test instantiation
    ContentAnalyzer()
    RedTeam()
    BlueTeam()
    EvaluatorTeam()
    QualityAssessmentEngine()