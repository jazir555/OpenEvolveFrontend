"""
Demo Application for OpenEvolve
A simple demonstration of the complete OpenEvolve system functionality.
"""
import streamlit as st
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import OpenEvolve components
from content_analyzer import ContentAnalyzer

from quality_assessment import QualityAssessmentEngine
from red_team import RedTeam
from blue_team import BlueTeam
from evaluator_team import EvaluatorTeam
from evolutionary_optimization import EvolutionaryOptimizer, EvolutionConfiguration
from configuration_system import ConfigurationManager
from quality_assurance import QualityAssuranceOrchestrator
from performance_optimization import (
    CachingOptimizer, 
    ParallelizationOptimizer, 
    MemoryManagementOptimizer
)



# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4a6fa5, #6b8cbc);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        background-color: #e0f2fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo application"""
    # Header
    st.markdown('<div class="main-header"><h1>🧬 OpenEvolve Demo</h1><p>AI-Powered Content Evolution & Testing Platform</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Content type selection
    content_type = st.sidebar.selectbox(
        "Content Type",
        ["general", "code_python", "code_js", "code_java", "code_cpp", "legal", "medical", "technical"],
        index=0
    )
    
    # Demo mode selection
    demo_mode = st.sidebar.radio(
        "Demo Mode",
        ["Simple Analysis", "Full Adversarial Testing", "Evolutionary Optimization"],
        index=0
    )
    
    # Sample content templates
    templates = {
        "Security Policy": """# Security Policy Template

## Overview
This document describes a security policy for protecting sensitive information.

## Scope
Applies to all employees and contractors with system access.

## Policy Statements
1. All users must use strong passwords
2. Multi-factor authentication is required for sensitive systems
3. Regular security training is mandatory
4. Incident reporting must occur within 24 hours

## Compliance
Violations result in disciplinary action.""",
        
        "Medical Protocol": """# Patient Care Protocol

## Overview
This protocol outlines care procedures for patients with chronic conditions.

## Assessment
1. Initial patient interview
2. Vital signs measurement
3. Symptom evaluation
4. Risk factor identification

## Treatment Plan
1. Medication management
2. Lifestyle modification recommendations
3. Follow-up appointment scheduling
4. Emergency contact procedures

## Documentation
All patient interactions must be documented in the electronic health record.""",
        
        "Code Review": """def authenticate_user(username, password):
    # This function has security vulnerabilities
    if username == "admin" and password == "password123":
        return True
    return False

def process_data(data):
    # Process data without proper validation
    result = eval(data)  # Dangerous!
    return result"""
    }
    
    # Template selection
    selected_template = st.sidebar.selectbox(
        "Sample Templates",
        ["", "Security Policy", "Medical Protocol", "Code Review"],
        index=0
    )
    
    # Load template if selected
    initial_content = ""
    if selected_template and selected_template in templates:
        initial_content = templates[selected_template]
    
    # Main content area
    st.subheader("📝 Content Input")
    
    # Text area for content
    content = st.text_area(
        "Enter or paste your content:",
        value=initial_content,
        height=300,
        placeholder="Paste your draft content here..."
    )
    
    # Run button
    if st.button("🚀 Run OpenEvolve Process", type="primary", use_container_width=True):
        if not content.strip():
            st.warning("Please enter some content to analyze.")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize components
            status_text.text("Initializing OpenEvolve components...")
            progress_bar.progress(10)
            
            content_analyzer = ContentAnalyzer()

            quality_assessment = QualityAssessmentEngine()
            red_team = RedTeam()
            blue_team = BlueTeam()
            evaluator_team = EvaluatorTeam()
            config_manager = ConfigurationManager()
            qa_orchestrator = QualityAssuranceOrchestrator()
            
            # Performance optimizers
            caching_optimizer = CachingOptimizer()
            parallelization_optimizer = ParallelizationOptimizer()

            memory_optimizer = MemoryManagementOptimizer()
            
            progress_bar.progress(20)
            
            # 1. Content Analysis
            status_text.text("Analyzing content...")
            analysis_result = content_analyzer.analyze_content(content)
            progress_bar.progress(30)
            
            # 2. Quality Assessment
            status_text.text("Assessing quality...")
            quality_result = quality_assessment.assess_quality(content, content_type)
            progress_bar.progress(40)
            
            if demo_mode == "Simple Analysis":
                # Show simple analysis results
                st.success("✅ Analysis completed successfully!")
                progress_bar.progress(100)
                
                # Display results in tabs
                result_tabs = st.tabs(["Content Analysis", "Quality Assessment"])
                
                with result_tabs[0]:
                    st.markdown('<div class="section-header"><h3>Content Analysis Results</h3></div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Overall Score", f"{analysis_result.overall_score:.1f}/100")
                    col2.metric("Word Count", analysis_result.input_parsing.get('word_count', 0))
                    col3.metric("Sentence Count", analysis_result.input_parsing.get('sentence_count', 0))
                    
                    st.markdown("**Semantic Understanding:**")
                    st.json(analysis_result.semantic_understanding)
                    
                    st.markdown("**Pattern Recognition:**")
                    st.json(analysis_result.pattern_recognition)
                
                with result_tabs[1]:
                    st.markdown('<div class="section-header"><h3>Quality Assessment Results</h3></div>', unsafe_allow_html=True)
                    st.metric("Composite Score", f"{quality_result.composite_score:.1f}/100")
                    
                    st.markdown("**Quality Scores by Dimension:**")
                    score_data = []
                    for dimension, score in quality_result.scores.items():
                        score_data.append({"Dimension": dimension.value, "Score": score})
                    
                    import pandas as pd
                    df = pd.DataFrame(score_data)
                    st.bar_chart(df.set_index("Dimension"))
                    
                    if quality_result.issues:
                        st.markdown("**Issues Found:**")
                        for issue in quality_result.issues[:5]:  # Show first 5
                            st.markdown(f"- **{issue.title}** ({issue.severity.value}): {issue.description}")
                    
                    if quality_result.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in quality_result.recommendations[:5]:  # Show first 5
                            st.markdown(f"- {rec}")
            
            elif demo_mode == "Full Adversarial Testing":
                # Full adversarial testing
                status_text.text("Running red team assessment...")
                red_result = red_team.assess_content(content, content_type)
                progress_bar.progress(50)
                
                status_text.text("Running blue team fixes...")
                blue_result = blue_team.apply_fixes(content, red_result.findings, content_type)
                progress_bar.progress(70)
                
                status_text.text("Running evaluator team assessment...")
                eval_result = evaluator_team.evaluate_content(blue_result.fixed_content, content_type)
                progress_bar.progress(90)
                
                # Show adversarial testing results
                st.success("✅ Adversarial testing completed successfully!")
                progress_bar.progress(100)
                
                # Display results in tabs
                result_tabs = st.tabs(["Red Team", "Blue Team", "Evaluator Team", "Comparison"])
                
                with result_tabs[0]:
                    st.markdown('<div class="section-header"><h3>Red Team (Critics) Results</h3></div>', unsafe_allow_html=True)
                    st.metric("Issues Identified", len(red_result.findings))
                    
                    if red_result.findings:
                        st.markdown("**Critical Issues:**")
                        critical_issues = [f for f in red_result.findings if f.severity.value == "critical"]
                        for issue in critical_issues[:3]:
                            st.markdown(f"- 🔴 **{issue.title}**: {issue.description}")
                        
                        st.markdown("**All Issues:**")
                        for i, issue in enumerate(red_result.findings[:10]):
                            st.markdown(f"{i+1}. **{issue.title}** ({issue.severity.value}): {issue.description}")
                
                with result_tabs[1]:
                    st.markdown('<div class="section-header"><h3>Blue Team (Fixers) Results</h3></div>', unsafe_allow_html=True)
                    st.metric("Fixes Applied", len(blue_result.applied_fixes))
                    st.metric("Improvement Score", f"{blue_result.overall_improvement_score:.1f}/100")
                    
                    if blue_result.applied_fixes:
                        st.markdown("**Applied Fixes:**")
                        for fix in blue_result.applied_fixes[:5]:
                            st.markdown(f"- 🔵 **{fix.title}**: {fix.description}")
                
                with result_tabs[2]:
                    st.markdown('<div class="section-header"><h3>Evaluator Team (Judges) Results</h3></div>', unsafe_allow_html=True)
                    st.metric("Consensus Reached", "Yes" if eval_result.consensus_reached else "No")
                    st.metric("Consensus Score", f"{eval_result.consensus_score:.1f}/100")
                    st.markdown(f"**Final Verdict**: {eval_result.final_verdict}")
                
                with result_tabs[3]:
                    st.markdown('<div class="section-header"><h3>Before/After Comparison</h3></div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Content:**")
                        st.text_area("", value=content[:500] + "..." if len(content) > 500 else content, height=200, key="original_content")
                    
                    with col2:
                        st.markdown("**Improved Content:**")
                        st.text_area("", value=blue_result.fixed_content[:500] + "..." if len(blue_result.fixed_content) > 500 else blue_result.fixed_content, height=200, key="improved_content")
            
            elif demo_mode == "Evolutionary Optimization":
                # Evolutionary optimization
                status_text.text("Setting up evolutionary optimization...")
                
                # Mock fitness function for demo
                class MockFitnessFunction:
                    def evaluate(self, individual, content_type="general"):
                        # Simple fitness based on content characteristics
                        content = individual.genome
                        word_count = len(content.split())
                        sentence_count = len(content.split('.')) + len(content.split('!')) + len(content.split('?'))
                        
                        # Base score
                        score = 50.0
                        
                        # Adjust based on content characteristics
                        if word_count > 100:
                            score += 20
                        if sentence_count > 5:
                            score += 15
                        if 'introduction' in content.lower() or 'conclusion' in content.lower():
                            score += 15
                        
                        return min(100.0, score)
                
                # Create evolution configuration
                evolution_config = EvolutionConfiguration(
                    population_size=20,
                    num_generations=5,
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    elitism_rate=0.1
                )
                
                # Create evolutionary optimizer
                evolutionary_optimizer = EvolutionaryOptimizer(evolution_config, MockFitnessFunction())
                
                status_text.text("Running evolutionary optimization...")
                evolution_result = evolutionary_optimizer.evolve(content, content_type, max_generations=3)
                progress_bar.progress(90)
                
                # Show evolutionary results
                st.success("✅ Evolutionary optimization completed successfully!")
                progress_bar.progress(100)
                
                # Display results in tabs
                result_tabs = st.tabs(["Evolution Process", "Best Solution", "Performance"])
                
                with result_tabs[0]:
                    st.markdown('<div class="section-header"><h3>Evolution Process</h3></div>', unsafe_allow_html=True)
                    st.metric("Generations Completed", len(evolution_result.evolution_history))
                    st.metric("Best Fitness Score", f"{evolution_result.best_individual.fitness:.2f}/100")
                    
                    if evolution_result.evolution_history:
                        st.markdown("**Generation History:**")
                        gen_data = []
                        for i, gen in enumerate(evolution_result.evolution_history):
                            gen_data.append({
                                "Generation": i+1,
                                "Best Fitness": max([ind.fitness for ind in gen.individuals]),
                                "Avg Fitness": sum([ind.fitness for ind in gen.individuals]) / len(gen.individuals)
                            })
                        
                        import pandas as pd
                        df = pd.DataFrame(gen_data)
                        st.line_chart(df.set_index("Generation"))
                
                with result_tabs[1]:
                    st.markdown('<div class="section-header"><h3>Best Solution Found</h3></div>', unsafe_allow_html=True)
                    st.text_area("Best Content", value=evolution_result.best_individual.genome, height=300)
                    
                    # Show improvement metrics
                    original_fitness = MockFitnessFunction().evaluate(
                        type('obj', (object,), {'genome': content})(), content_type
                    )
                    improvement = evolution_result.best_individual.fitness - original_fitness
                    st.metric("Improvement", f"{improvement:+.2f} points")
                
                with result_tabs[2]:
                    st.markdown('<div class="section-header"><h3>Performance Metrics</h3></div>', unsafe_allow_html=True)
                    
                    # Show performance optimization stats
                    st.markdown("**Caching Performance:**")
                    cache_info = caching_optimizer.get_cache_info()
                    st.json(cache_info)
                    
                    st.markdown("**Parallelization Stats:**")
                    parallel_stats = parallelization_optimizer.parallelization_stats
                    st.json(parallel_stats)
                    
                    st.markdown("**Memory Management:**")
                    memory_stats = memory_optimizer.memory_stats
                    st.json(memory_stats)
            
            # Show configuration info
            st.markdown('<div class="section-header"><h3>Configuration</h3></div>', unsafe_allow_html=True)
            config_params = config_manager.list_parameters()
            st.info(f"Configuration system has {len(config_params)} parameters")
            
            # Show quality assurance info
            st.markdown('<div class="section-header"><h3>Quality Assurance</h3></div>', unsafe_allow_html=True)
            qa_gates = len(qa_orchestrator.gates)
            st.info(f"Quality assurance system has {qa_gates} validation gates")
            
        except Exception as e:
            st.error(f"❌ Error running OpenEvolve process: {e}")
            import traceback
            st.code(traceback.format_exc())
            progress_bar.progress(100)
    
    # Footer
    st.markdown("---")
    st.markdown("🧬 **OpenEvolve** - AI-Powered Content Evolution & Testing Platform")
    st.markdown(f"🕒 Demo last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()