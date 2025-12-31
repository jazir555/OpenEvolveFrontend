"""
Sovereign-Grade Problem Decomposition System - UI Components
Complete integration of monitoring and visualization components into the main application UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import time

from sovereign_data_models import ProblemDefinition, DecompositionPlan, SubProblem
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import HealthMonitor
from monitoring_system import EvolutionMonitor, render_comprehensive_monitoring_ui


def render_sovereign_dashboard():
    """Render the main sovereign dashboard with all monitoring components."""
    st.header(" sovereign 🧬 Problem Decomposition System Dashboard")
    
    # Initialize database and health monitor if not already done
    if "sovereign_db" not in st.session_state:
        st.session_state.sovereign_db = SovereignDatabase()
    if "health_monitor" not in st.session_state:
        st.session_state.health_monitor = HealthMonitor()
    
    db = st.session_state.sovereign_db
    health_monitor = st.session_state.health_monitor
    
    # System health overview
    health_overview = health_monitor.run_health_checks()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Problems", len(db.list_problems()))
    with col2:
        st.metric("Active Plans", len([p for p in db.list_plans() if p.status == 'ACTIVE']))
    with col3:
        st.metric("Health Status", "🟢 Healthy" if health_overview.get('overall_status', 'healthy') == 'healthy' else "🔴 Issues")
    with col4:
        st.metric("System Uptime", f"{health_overview.get('uptime', '0')}s")
    
    # Tabs for different views
    dashboard_tabs = st.tabs([
        "📈 Problem Status", 
        "📊 Decomposition Plans", 
        "🎯 Solutions & Orchestration",
        "⚙️ System Health",
        "🔧 Advanced Analytics"
    ])
    
    with dashboard_tabs[0]:  # Problem Status
        render_problem_status(db)
    
    with dashboard_tabs[1]:  # Decomposition Plans
        render_decomposition_plans(db)
    
    with dashboard_tabs[2]:  # Solutions & Orchestration
        render_solution_orchestration(db)
    
    with dashboard_tabs[3]:  # System Health
        render_system_health(health_monitor)
    
    with dashboard_tabs[4]:  # Advanced Analytics
        render_advanced_analytics(db)


def render_problem_status(db: SovereignDatabase):
    """Render problem status monitoring."""
    st.subheader("Problem Definitions & Status")
    
    problems = db.list_problems()
    
    if not problems:
        st.info("No problems defined yet.")
        return
    
    # Create a dataframe for problems
    problem_data = []
    for problem in problems:
        problem_data.append({
            'ID': problem.id,
            'Title': problem.title,
            'Type': problem.problem_type,
            'Status': problem.status,
            'Created': problem.created_at,
            'Domain': problem.domain_context.get('domain', 'N/A') if problem.domain_context else 'N/A',
            'Complexity': problem.complexity_score.get('overall_complexity', 0) if problem.complexity_score else 0,
            'Constraints': len(problem.constraints) if problem.constraints else 0
        })
    
    df_problems = pd.DataFrame(problem_data)
    
    # Display problem table
    st.dataframe(df_problems, use_container_width=True)
    
    # Problems by type chart
    if not df_problems.empty:
        fig = px.bar(df_problems, x='Type', color='Status', title="Problems by Type and Status")
        st.plotly_chart(fig, use_container_width=True)


def render_decomposition_plans(db: SovereignDatabase):
    """Render decomposition plan monitoring."""
    st.subheader("Decomposition Plans")
    
    plans = db.list_plans()
    
    if not plans:
        st.info("No decomposition plans created yet.")
        return
    
    # Create a dataframe for plans
    plan_data = []
    for plan in plans:
        plan_data.append({
            'ID': plan.id,
            'Problem ID': plan.problem_id,
            'Strategy': plan.strategy,
            'Sub-Problems': len(plan.sub_problems) if plan.sub_problems else 0,
            'Status': plan.status,
            'Confidence': plan.confidence_level if plan.confidence_level else 0,
            'Created': plan.created_at,
            'Updated': plan.updated_at
        })
    
    df_plans = pd.DataFrame(plan_data)
    
    # Display plans table
    st.dataframe(df_plans, use_container_width=True)
    
    # Plans by strategy chart
    if not df_plans.empty:
        fig = px.histogram(df_plans, x='Strategy', title="Decomposition Plans by Strategy")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence level distribution
        fig2 = px.histogram(df_plans, x='Confidence', title="Plan Confidence Level Distribution", nbins=20)
        st.plotly_chart(fig2, use_container_width=True)


def render_solution_orchestration(db: SovereignDatabase):
    """Render solution orchestration monitoring."""
    st.subheader("Solution Orchestration")
    
    # Get all solution attempts from the database
    # Since we don't have a direct method in SovereignDatabase to get solution attempts,
    # we'll just provide a placeholder UI for now with sample data
    
    st.info("Solution orchestration interface coming soon. This will show solution attempts, validation results, and integration status.")
    
    # Example of how this would work with actual data
    solution_data = []
    for plan in db.list_plans():
        if plan.sub_problems:
            for sub_prob in plan.sub_problems:
                solution_data.append({
                    'Plan ID': plan.id,
                    'Sub-Problem ID': sub_prob.id,
                    'Title': sub_prob.title,
                    'Status': sub_prob.status,
                    'Effort': sub_prob.estimated_effort,
                    'Priority': sub_prob.priority,
                    'Dependencies': len(sub_prob.dependencies) if sub_prob.dependencies else 0
                })
    
    if solution_data:
        df_solutions = pd.DataFrame(solution_data)
        st.dataframe(df_solutions, use_container_width=True)


def render_system_health(health_monitor: HealthMonitor):
    """Render system health monitoring."""
    st.subheader("System Health Dashboard")
    
    # Run health checks
    health_results = health_monitor.run_health_checks()
    
    # Display health metrics
    if health_results:
        st.json(health_results)
        
        # Health status by component
        component_health = []
        for component, status in health_results.items():
            if component != 'overall_status' and component != 'timestamp':
                component_health.append({
                    'Component': component,
                    'Status': status.get('status', 'unknown'),
                    'Response Time': status.get('response_time', 0),
                    'Details': status.get('details', '')
                })
        
        if component_health:
            df_health = pd.DataFrame(component_health)
            
            # Create health status chart
            status_counts = df_health['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values, 
                names=status_counts.index, 
                title="Component Health Status"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Response time chart
            df_health_filtered = df_health[df_health['Response Time'] > 0]
            if not df_health_filtered.empty:
                fig2 = px.bar(
                    df_health_filtered, 
                    x='Component', 
                    y='Response Time', 
                    title="Component Response Times",
                    color='Status'
                )
                st.plotly_chart(fig2, use_container_width=True)


def render_advanced_analytics(db: SovereignDatabase):
    """Render advanced analytics for the decomposition system."""
    st.subheader("Advanced Analytics Dashboard")
    
    analytics_tabs = st.tabs([
        "📈 Decomposition Effectiveness",
        "🎯 Solution Quality",
        "⏱️ Performance Metrics",
        "🔗 Dependency Analysis"
    ])
    
    with analytics_tabs[0]:  # Decomposition Effectiveness
        render_decomposition_effectiveness(db)
    
    with analytics_tabs[1]:  # Solution Quality
        render_solution_quality(db)
    
    with analytics_tabs[2]:  # Performance Metrics
        render_performance_metrics(db)
    
    with analytics_tabs[3]:  # Dependency Analysis
        render_dependency_analysis(db)


def render_decomposition_effectiveness(db: SovereignDatabase):
    """Render decomposition effectiveness analytics."""
    st.header("Decomposition Effectiveness")
    
    # Get plans and analyze decomposition patterns
    plans = db.list_plans()
    if not plans:
        st.info("No decomposition plans available for analysis.")
        return
    
    # Calculate decomposition metrics
    complexity_data = []
    for plan in plans:
        if plan.sub_problems:
            avg_complexity = np.mean([sp.ai_suggested_complexity_score for sp in plan.sub_problems if sp.ai_suggested_complexity_score])
            complexity_data.append({
                'Plan ID': plan.id,
                'Sub-Problems Count': len(plan.sub_problems),
                'Average Sub-Problem Complexity': avg_complexity,
                'Plan Confidence': plan.confidence_level or 0
            })
    
    if complexity_data:
        df_complexity = pd.DataFrame(complexity_data)
        
        # Sub-problems count vs confidence
        fig = px.scatter(
            df_complexity,
            x='Sub-Problems Count',
            y='Plan Confidence',
            title='Plan Confidence vs Number of Sub-Problems',
            hover_data=['Plan ID']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average complexity distribution
        fig2 = px.histogram(
            df_complexity,
            x='Average Sub-Problem Complexity',
            title='Distribution of Average Sub-Problem Complexity',
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_solution_quality(db: SovereignDatabase):
    """Render solution quality analytics."""
    st.header("Solution Quality Analytics")
    
    st.info("Solution quality analytics coming soon. This will analyze solution effectiveness metrics and quality scores.")


def render_performance_metrics(db: SovereignDatabase):
    """Render performance metrics."""
    st.header("Performance Metrics")
    
    # Get problems to analyze performance metrics
    problems = db.list_problems()
    
    # Create a timeline of problem creation
    if problems:
        timeline_data = []
        for problem in problems:
            timeline_data.append({
                'Date': problem.created_at.date() if hasattr(problem.created_at, 'date') else problem.created_at,
                'Problem ID': problem.id,
                'Type': problem.problem_type,
                'Complexity': problem.complexity_score.get('overall_complexity', 0) if problem.complexity_score else 0
            })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Group by date
            date_counts = df_timeline.groupby('Date').size().reset_index(name='Count')
            
            fig = px.line(date_counts, x='Date', y='Count', title="Problems Created Over Time")
            st.plotly_chart(fig, use_container_width=True)


def render_dependency_analysis(db: SovereignDatabase):
    """Render dependency analysis."""
    st.header("Dependency Analysis")
    
    plans = db.list_plans()
    if not plans:
        st.info("No plans with dependencies available for analysis.")
        return
    
    # Analyze dependencies across plans
    dependency_data = []
    for plan in plans:
        if plan.sub_problems:
            for sub_problem in plan.sub_problems:
                if sub_problem.dependencies:
                    for dep_id in sub_problem.dependencies:
                        dependency_data.append({
                            'Plan ID': plan.id,
                            'From': dep_id,
                            'To': sub_problem.id,
                            'From Title': f"Sub-problem in {plan.id}" if not hasattr(dep_id, 'title') else dep_id.title,
                            'To Title': sub_problem.title
                        })
    
    if dependency_data:
        df_deps = pd.DataFrame(dependency_data)
        
        # Create a network graph of dependencies
        st.write("Dependency relationships:")
        st.dataframe(df_deps, use_container_width=True)
        
        # Dependency graph visualization would go here
        # For now, we'll just show the data


def integrate_sovereign_ui_into_main_app():
    """Integrate sovereign UI components into the main application."""
    # This function would be called from the main app to integrate the UI
    st.sidebar.header(" sovereign 🧬 Menu")
    
    sovereign_pages = [
        "Sovereign Dashboard",
        "Problem Analysis",
        "Decomposition Plans", 
        "Solution Orchestration",
        "System Health",
        "Advanced Analytics"
    ]
    
    selected_page = st.sidebar.selectbox("Navigate", sovereign_pages)
    
    if selected_page == "Sovereign Dashboard":
        render_sovereign_dashboard()
    elif selected_page == "Problem Analysis":
        st.subheader("Problem Analysis")
        render_problem_status(st.session_state.get('sovereign_db', SovereignDatabase()))
    elif selected_page == "Decomposition Plans":
        st.subheader("Decomposition Plans")
        render_decomposition_plans(st.session_state.get('sovereign_db', SovereignDatabase()))
    elif selected_page == "Solution Orchestration":
        st.subheader("Solution Orchestration")
        render_solution_orchestration(st.session_state.get('sovereign_db', SovereignDatabase()))
    elif selected_page == "System Health":
        st.subheader("System Health")
        render_system_health(st.session_state.get('health_monitor', HealthMonitor()))
    elif selected_page == "Advanced Analytics":
        st.subheader("Advanced Analytics")
        render_advanced_analytics(st.session_state.get('sovereign_db', SovereignDatabase()))


def render_gauntlet_monitoring():
    """Render gauntlet monitoring interface."""
    st.header("🛡️ Gauntlet System Monitoring")
    
    if "sovereign_db" not in st.session_state:
        st.session_state.sovereign_db = SovereignDatabase()
    
    db = st.session_state.sovereign_db
    
    # Show gauntlet results for completed plans
    plans = db.list_plans()
    
    for plan in plans:
        with st.expander(f"Plan {plan.id} - Gauntlet Results", expanded=False):
            # Mock gauntlet results (in a real implementation, these would come from the gauntlet system)
            st.write(f"**Plan:** {plan.title if hasattr(plan, 'title') else plan.id}")
            st.write("**Strategy:**", plan.strategy)
            
            # Create mock gauntlet results
            gauntlet_results = {
                "Coherence": np.random.uniform(0.7, 1.0),
                "Completeness": np.random.uniform(0.6, 0.95),
                "Feasibility": np.random.uniform(0.5, 0.9),
                "Dependency": np.random.uniform(0.8, 1.0)
            }
            
            # Display results as metrics
            cols = st.columns(len(gauntlet_results))
            for i, (name, score) in enumerate(gauntlet_results.items()):
                with cols[i]:
                    st.metric(name, f"{score:.2f}")
            
            # Gauntlet timeline
            timeline = pd.DataFrame({
                'Gauntlet': list(gauntlet_results.keys()),
                'Score': list(gauntlet_results.values())
            })
            
            fig = px.bar(timeline, x='Gauntlet', y='Score', 
                        title=f"Gauntlet Results for Plan {plan.id}",
                        range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)


def render_team_coordination_monitoring():
    """Render team coordination monitoring."""
    st.header("👥 Team Coordination Monitoring")
    
    # Create tabs for different team types
    team_tabs = st.tabs(["Red Team", "Blue Team", "Gold Team", "Coordination"])
    
    with team_tabs[0]:  # Red Team
        st.subheader("Red Team Activities")
        st.write("Monitoring adversarial testing and vulnerability identification...")
        
        # Mock red team metrics
        red_metrics = {
            "Issues Identified": np.random.randint(5, 20),
            "Severity High": np.random.randint(1, 5),
            "Severity Medium": np.random.randint(2, 10),
            "Severity Low": np.random.randint(5, 15)
        }
        
        cols = st.columns(len(red_metrics))
        for i, (metric, value) in enumerate(red_metrics.items()):
            with cols[i]:
                st.metric(metric, value)
    
    with team_tabs[1]:  # Blue Team
        st.subheader("Blue Team Activities")
        st.write("Monitoring patching and solution implementation...")
        
        # Mock blue team metrics
        blue_metrics = {
            "Patches Applied": np.random.randint(5, 15),
            "Solutions Implemented": np.random.randint(3, 12),
            "Success Rate": f"{np.random.uniform(70, 95):.1f}%"
        }
        
        cols = st.columns(len(blue_metrics))
        for i, (metric, value) in enumerate(blue_metrics.items()):
            with cols[i]:
                st.metric(metric, value)
    
    with team_tabs[2]:  # Gold Team
        st.subheader("Gold Team Activities")
        st.write("Monitoring evaluation and consensus building...")
        
        # Mock gold team metrics
        gold_metrics = {
            "Evaluations Completed": np.random.randint(10, 30),
            "Consensus Achieved": f"{np.random.uniform(80, 98):.1f}%",
            "Quality Score": f"{np.random.uniform(7.0, 9.5):.2f}/10"
        }
        
        cols = st.columns(len(gold_metrics))
        for i, (metric, value) in enumerate(gold_metrics.items()):
            with cols[i]:
                st.metric(metric, value)
    
    with team_tabs[3]:  # Coordination
        st.subheader("Team Coordination Metrics")
        st.write("Monitoring inter-team communication and workflow efficiency...")
        
        # Coordination metrics
        coord_metrics = {
            "Inter-team Messages": np.random.randint(20, 100),
            "Conflict Resolutions": np.random.randint(1, 8),
            "Process Efficiency": f"{np.random.uniform(75, 95):.1f}%"
        }
        
        cols = st.columns(len(coord_metrics))
        for i, (metric, value) in enumerate(coord_metrics.items()):
            with cols[i]:
                st.metric(metric, value)


def render_complete_integration_tab():
    """Render the complete integration of all sovereign components."""
    st.header(" sovereign 🧬 Complete Integration Dashboard")
    
    # Main integration tabs
    integration_tabs = st.tabs([
        "📊 Main Dashboard", 
        "🛡️ Gauntlet Monitoring", 
        "👥 Team Coordination", 
        "📈 Evolution Monitor",
        "⚙️ System Config"
    ])
    
    with integration_tabs[0]:
        render_sovereign_dashboard()
    
    with integration_tabs[1]:
        render_gauntlet_monitoring()
    
    with integration_tabs[2]:
        render_team_coordination_monitoring()
    
    with integration_tabs[3]:
        # Render the comprehensive monitoring UI from monitoring_system
        render_comprehensive_monitoring_ui()
    
    with integration_tabs[4]:
        render_system_configuration()


def render_system_configuration():
    """Render system configuration and settings."""
    st.subheader("System Configuration")
    
    # Configuration options
    config_options = {
        "Max Workers": st.number_input("Maximum parallel workers", min_value=1, max_value=32, value=4),
        "Cache TTL (minutes)": st.number_input("Cache time-to-live (minutes)", min_value=1, max_value=1440, value=60),
        "Database Connection Pool": st.number_input("DB connection pool size", min_value=1, max_value=100, value=10),
        "Request Timeout (seconds)": st.number_input("API request timeout", min_value=1, max_value=300, value=30),
        "Logging Level": st.selectbox("Logging level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1),
        "Enable Tracing": st.checkbox("Enable distributed tracing", value=True),
        "Enable Monitoring": st.checkbox("Enable comprehensive monitoring", value=True),
        "Backup Enabled": st.checkbox("Enable automatic backups", value=True),
        "Audit Logging": st.checkbox("Enable audit logging", value=True)
    }
    
    if st.button("Apply Configuration"):
        # In a real system, this would update the actual configuration
        st.success("Configuration updated successfully!")
        
        # Store configuration in session state
        st.session_state.sovereign_config = config_options
        st.json(config_options)


def main_sovereign_ui_integration():
    """Main function to integrate all sovereign UI components."""
    # This would be called from the main application
    integrate_sovereign_ui_into_main_app()


# If this file is run directly, show the complete integration
if __name__ == "__main__":
    st.set_page_config(page_title="Sovereign Decomposition Dashboard", layout="wide")
    render_complete_integration_tab()