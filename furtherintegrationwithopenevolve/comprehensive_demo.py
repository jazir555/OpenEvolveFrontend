"""
Comprehensive demo showcasing all OpenEvolve functionality
"""
import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def comprehensive_demo():
    """Comprehensive demonstration of all OpenEvolve functionality"""
    print("OpenEvolve Comprehensive Demonstration")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Import all components
    from content_analyzer import ContentAnalyzer
    from prompt_engineering import PromptEngineeringSystem
    from model_orchestration import ModelOrchestrator
    from quality_assessment import QualityAssessmentEngine
    from red_team import RedTeam
    from blue_team import BlueTeam
    from evaluator_team import EvaluatorTeam
    from evolutionary_optimization import EvolutionaryOptimizer, EvolutionConfiguration
    from configuration_system import ConfigurationManager
    from quality_assurance import QualityAssuranceOrchestrator
    from performance_optimization import PerformanceOptimizationOrchestrator
    
    print("\\n[OK] All components imported successfully")
    
    # Initialize components
    content_analyzer = ContentAnalyzer()
    prompt_engineering = PromptEngineeringSystem()
    model_orchestrator = ModelOrchestrator()
    quality_assessment = QualityAssessmentEngine()
    red_team = RedTeam()
    blue_team = BlueTeam()
    evaluator_team = EvaluatorTeam()
    config_manager = ConfigurationManager()
    qa_orchestrator = QualityAssuranceOrchestrator()
    perf_optimizer = PerformanceOptimizationOrchestrator()
    
    print("[OK] All components instantiated successfully")
    
    # Create complex sample content
    complex_content = """
    # Enterprise Security Framework Implementation
    
    ## Overview
    This document outlines the implementation of an enterprise-grade security framework 
    for protecting sensitive corporate data and ensuring compliance with regulatory requirements.
    
    ## Security Architecture
    The framework consists of multiple layers:
    1. Network Security Layer
    2. Application Security Layer
    3. Data Security Layer
    4. Identity and Access Management Layer
    5. Monitoring and Incident Response Layer
    
    ## Implementation Steps
    ### Step 1: Network Security
    - Deploy firewalls at all network boundaries
    - Implement intrusion detection systems
    - Configure VPN for remote access
    - Set up network segmentation
    
    ### Step 2: Application Security
    - Conduct security code reviews
    - Implement input validation
    - Use secure coding practices
    - Deploy web application firewalls
    
    ### Step 3: Data Security
    - Encrypt data at rest and in transit
    - Implement access controls
    - Establish data backup procedures
    - Deploy data loss prevention tools
    
    ### Step 4: Identity and Access Management
    - Implement multi-factor authentication
    - Establish role-based access control
    - Deploy identity federation
    - Set up privileged access management
    
    ### Step 5: Monitoring and Incident Response
    - Deploy security information and event management (SIEM)
    - Establish security operations center (SOC)
    - Implement incident response procedures
    - Conduct regular security assessments
    
    ## Compliance Requirements
    - GDPR data protection requirements
    - HIPAA healthcare data security
    - PCI DSS payment card industry standards
    - SOX financial reporting controls
    
    ## Risk Assessment
    ### High-Risk Areas
    1. Legacy system integration
    2. Third-party vendor access
    3. Mobile device security
    4. Cloud service security
    
    ### Mitigation Strategies
    - Regular security audits
    - Vendor security assessments
    - Mobile device management solutions
    - Cloud security posture management
    
    ## Testing and Validation
    ### Security Testing
    - Penetration testing
    - Vulnerability scanning
    - Code review assessments
    - Security configuration reviews
    
    ### Performance Testing
    - Load testing
    - Stress testing
    - Scalability testing
    - Availability testing
    
    ## Training and Awareness
    ### Security Training Programs
    - Annual security awareness training
    - Role-specific security training
    - Incident response training
    - Secure coding training
    
    ### Ongoing Awareness
    - Monthly security newsletters
    - Security tip campaigns
    - Phishing simulation exercises
    - Security champion programs
    
    ## Maintenance and Updates
    ### Patch Management
    - Automated patch deployment
    - Patch testing procedures
    - Emergency patch processes
    - Third-party patch management
    
    ### Framework Updates
    - Quarterly framework reviews
    - Annual comprehensive assessments
    - Regulatory compliance updates
    - Emerging threat adaptation
    
    ## Conclusion
    This enterprise security framework provides a comprehensive approach to protecting 
    corporate assets while ensuring regulatory compliance. Regular reviews and updates 
    will ensure the framework remains effective against evolving threats.
    """
    
    print("\\n[OK] Sample content created")
    print(f"  Content length: {len(complex_content)} characters")
    
    # 1. Content Analysis Phase
    print("\\n1. Content Analysis Phase")
    print("-" * 30)
    
    analysis_start = time.time()
    analysis_result = content_analyzer.analyze_content(complex_content)
    analysis_time = time.time() - analysis_start
    
    print(f"  ✓ Content analyzed in {analysis_time:.2f} seconds")
    print(f"  Content type: {analysis_result.semantic_understanding.get('content_type', 'unknown')}")
    print(f"  Overall score: {analysis_result.overall_score:.2f}/100")
    print(f"  Word count: {analysis_result.input_parsing.get('word_count', 0)}")
    print(f"  Sentence count: {analysis_result.input_parsing.get('sentence_count', 0)}")
    
    # 2. Prompt Engineering Phase
    print("\\n2. Prompt Engineering Phase")
    print("-" * 30)
    
    prompt_start = time.time()
    variables = {
        'content': complex_content[:500] + "...",  # Truncate for demo
        'content_type': analysis_result.semantic_understanding.get('content_type', 'document').value,
        'compliance_requirements': 'GDPR, HIPAA, PCI DSS, SOX'
    }
    
    critique_prompt = prompt_engineering.prompt_manager.instantiate_prompt(
        'red_team_critique', variables
    )
    
    patch_prompt = prompt_engineering.prompt_manager.instantiate_prompt(
        'blue_team_patch', variables
    )
    
    prompt_time = time.time() - prompt_start
    print(f"  ✓ Prompts generated in {prompt_time:.2f} seconds")
    print(f"  Critique prompt length: {len(critique_prompt.rendered_prompt)} characters")
    print(f"  Patch prompt length: {len(patch_prompt.rendered_prompt)} characters")
    
    # 3. Quality Assessment Phase
    print("\\n3. Quality Assessment Phase")
    print("-" * 30)
    
    qa_start = time.time()
    qa_result = quality_assessment.assess_quality(complex_content, "document")
    qa_time = time.time() - qa_start
    
    print(f"  ✓ Quality assessed in {qa_time:.2f} seconds")
    print(f"  Composite score: {qa_result.composite_score:.2f}/100")
    print(f"  Issues found: {len(qa_result.issues)}")
    print(f"  Recommendations: {len(qa_result.recommendations)}")
    
    # 4. Red Team Assessment Phase
    print("\\n4. Red Team Assessment Phase")
    print("-" * 30)
    
    red_start = time.time()
    red_result = red_team.assess_content(complex_content, "document")
    red_time = time.time() - red_start
    
    print(f"  ✓ Red team assessment in {red_time:.2f} seconds")
    print(f"  Issues identified: {len(red_result.findings)}")
    if red_result.findings:
        severity_counts = {}
        for finding in red_result.findings:
            severity = finding.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        print(f"  Severity breakdown: {severity_counts}")
    
    # 5. Blue Team Fixing Phase
    print("\\n5. Blue Team Fixing Phase")
    print("-" * 30)
    
    blue_start = time.time()
    blue_result = blue_team.apply_fixes(complex_content, red_result.findings, "document")
    blue_time = time.time() - blue_start
    
    print(f"  ✓ Blue team fixes applied in {blue_time:.2f} seconds")
    print(f"  Fixes applied: {len(blue_result.applied_fixes)}")
    print(f"  Improvement score: {blue_result.overall_improvement_score:.2f}/100")
    print(f"  Fixed content length: {len(blue_result.fixed_content)} characters")
    
    # 6. Evaluator Team Assessment Phase
    print("\\n6. Evaluator Team Assessment Phase")
    print("-" * 30)
    
    eval_start = time.time()
    eval_result = evaluator_team.evaluate_content(blue_result.fixed_content, "document")
    eval_time = time.time() - eval_start
    
    print(f"  ✓ Evaluator team assessment in {eval_time:.2f} seconds")
    print(f"  Consensus reached: {eval_result.consensus_reached}")
    print(f"  Consensus score: {eval_result.consensus_score:.2f}/100")
    print(f"  Final verdict: {eval_result.final_verdict}")
    
    # 7. Configuration System
    print("\\n7. Configuration System")
    print("-" * 30)
    
    config_params = config_manager.list_parameters()
    print(f"  ✓ Configuration system ready")
    print(f"  Parameters available: {len(config_params)}")
    
    # 8. Quality Assurance
    print("\\n8. Quality Assurance")
    print("-" * 30)
    
    qa_gates = len(qa_orchestrator.gates)
    print(f"  ✓ Quality assurance system ready")
    print(f"  Quality gates: {qa_gates}")
    
    # 9. Performance Optimization
    print("\\n9. Performance Optimization")
    print("-" * 30)
    
    perf_optimizers = len(perf_optimizer.optimizers)
    print(f"  ✓ Performance optimization system ready")
    print(f"  Optimizers available: {perf_optimizers}")
    
    # Summary
    print("\\n" + "=" * 50)
    print("COMPREHENSIVE DEMO SUMMARY")
    print("=" * 50)
    
    total_time = time.time() - start_time
    print(f"Demo completed in: {total_time:.2f} seconds")
    
    # Performance summary
    phase_times = {
        "Analysis": analysis_time,
        "Prompt Engineering": prompt_time,
        "Quality Assessment": qa_time,
        "Red Team": red_time,
        "Blue Team": blue_time,
        "Evaluator Team": eval_time
    }
    
    print("\\nPhase Performance:")
    for phase, duration in phase_times.items():
        print(f"  {phase}: {duration:.2f}s")
    
    # Results summary
    print("\\nResults Summary:")
    print(f"  Initial content score: {analysis_result.overall_score:.2f}/100")
    print(f"  Quality assessment score: {qa_result.composite_score:.2f}/100")
    print(f"  Red team issues: {len(red_result.findings)}")
    print(f"  Blue team fixes: {len(blue_result.applied_fixes)}")
    print(f"  Improvement score: {blue_result.overall_improvement_score:.2f}/100")
    print(f"  Final evaluator score: {eval_result.consensus_score:.2f}/100")
    print(f"  Evaluator consensus: {eval_result.consensus_reached}")
    
    # Improvement calculation
    initial_score = analysis_result.overall_score
    final_score = eval_result.consensus_score
    improvement = final_score - initial_score
    improvement_percentage = (improvement / initial_score) * 100 if initial_score > 0 else 0
    
    print(f"\\nOverall Improvement:")
    print(f"  Score change: {initial_score:.2f} → {final_score:.2f} ({improvement:+.2f})")
    print(f"  Percentage improvement: {improvement_percentage:+.2f}%")
    
    print("\\n\\n[SUCCESS] Comprehensive demo completed successfully!")
    print("All OpenEvolve components are working together seamlessly.")
    print("=" * 50)

if __name__ == "__main__":
    comprehensive_demo()