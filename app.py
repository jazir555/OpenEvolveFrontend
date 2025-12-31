"""
Main application demonstrating OpenEvolve functionality
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main application function"""
    print("OpenEvolve Demo Application")
    print("=" * 40)
    
    # Import components
    from content_analyzer import ContentAnalyzer
    from red_team import RedTeam
    from blue_team import BlueTeam
    from evaluator_team import EvaluatorTeam
    from quality_assessment import QualityAssessmentEngine
    
    # Sample content to analyze and improve
    sample_content = """
    # Security Authentication Protocol
    
    This document describes a security protocol for user authentication.
    
    ## Authentication Process
    1. User enters username and password
    2. System checks credentials
    3. If valid, grant access
    4. If invalid, deny access
    
    ## Issues
    - Passwords are stored in plain text
    - No password complexity requirements
    - No account lockout mechanism
    - No multi-factor authentication
    - No session management
    - No input validation
    """
    
    print("\\n1. Content Analysis")
    print("-" * 20)
    
    # Analyze content
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_content(sample_content)
    print(f"Content Type: {analysis.semantic_understanding.get('content_type', 'unknown')}")
    print(f"Overall Score: {analysis.overall_score:.2f}/100")
    print(f"Word Count: {analysis.input_parsing.get('word_count', 0)}")
    
    print("\\n2. Quality Assessment")
    print("-" * 20)
    
    # Assess quality
    qa_engine = QualityAssessmentEngine()
    quality_result = qa_engine.assess_quality(sample_content, "document")
    print(f"Quality Score: {quality_result.composite_score:.2f}/100")
    print(f"Issues Found: {len(quality_result.issues)}")
    print(f"Recommendations: {len(quality_result.recommendations)}")
    
    print("\\n3. Red Team Assessment")
    print("-" * 20)
    
    # Red team assessment
    red_team = RedTeam()
    red_result = red_team.assess_content(sample_content, "document")
    print(f"Issues Identified: {len(red_result.findings)}")
    if red_result.findings:
        print("Top Issues:")
        for i, finding in enumerate(red_result.findings[:3]):
            print(f"  {i+1}. {finding.severity.value.upper()}: {finding.title}")
    
    print("\\n4. Blue Team Fixes")
    print("-" * 20)
    
    # Blue team fixes
    blue_team = BlueTeam()
    blue_result = blue_team.apply_fixes(sample_content, red_result.findings, "document")
    print(f"Fixes Applied: {len(blue_result.applied_fixes)}")
    print(f"Improvement Score: {blue_result.overall_improvement_score:.2f}/100")
    
    print("\\n5. Evaluator Team Assessment")
    print("-" * 20)
    
    # Evaluator team assessment
    evaluator_team = EvaluatorTeam()
    eval_result = evaluator_team.evaluate_content(blue_result.fixed_content, "document")
    print(f"Consensus Reached: {eval_result.consensus_reached}")
    print(f"Final Verdict: {eval_result.final_verdict}")
    print(f"Composite Score: {eval_result.consensus_score:.2f}/100")
    
    print("\\n6. Before/After Comparison")
    print("-" * 20)
    print("ORIGINAL CONTENT:")
    print(sample_content[:200] + "..." if len(sample_content) > 200 else sample_content)
    print("\\nIMPROVED CONTENT:")
    print(blue_result.fixed_content[:200] + "..." if len(blue_result.fixed_content) > 200 else blue_result.fixed_content)
    
    print("\\n\\n[DONE] OpenEvolve demo completed successfully!")
    print("All components are working together to analyze, critique, fix, and evaluate content.")

if __name__ == "__main__":
    main()