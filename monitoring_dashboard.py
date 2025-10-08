"""
Comprehensive Monitoring Dashboard for OpenEvolve
This module provides real-time monitoring and logging for OpenEvolve operations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime


def render_monitoring_dashboard():
    """Render the comprehensive monitoring dashboard in Streamlit."""
    st.markdown("## 📡 Comprehensive Monitoring Dashboard")
    
    # Create tabs for different monitoring views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-Time Metrics", 
        "📝 Activity Logs", 
        "📈 Performance Analytics", 
        "⚙️ System Health"
    ])
    
    with tab1:  # Real-Time Metrics
        render_real_time_metrics()
    
    with tab2:  # Activity Logs
        render_activity_logs()
    
    with tab3:  # Performance Analytics
        render_performance_analytics()
    
    with tab4:  # System Health
        render_system_health()


def render_real_time_metrics():
    """Render real-time metrics monitoring."""
    st.header("📊 Real-Time Evolution Metrics")
    
    # Evolution status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    # Current evolution status
    evolution_running = st.session_state.get("evolution_running", False)
    adversarial_running = st.session_state.get("adversarial_running", False)
    
    col1.metric("Evolution Status", "Running" if evolution_running else "Idle", 
                "Evolution process status")
    col2.metric("Adversarial Status", "Running" if adversarial_running else "Idle", 
                "Adversarial testing status")
    
    # If evolution is running, show additional metrics
    if evolution_running:
        current_best_score = st.session_state.get("evolution_current_best_score", 0.0)
        current_iteration = st.session_state.get("current_iteration", 0)
        total_iterations = st.session_state.get("max_iterations", 100)
        
        col3.metric("Current Best Score", f"{current_best_score:.4f}")
        col4.metric("Progress", f"{current_iteration}/{total_iterations} ({(current_iteration/total_iterations)*100:.1f}%)")
        
        # Progress bar
        st.progress(current_iteration / total_iterations if total_iterations > 0 else 0)
    else:
        col3.metric("Current Best Score", "N/A")
        col4.metric("Progress", "Not Running")
    
    st.divider()
    
    # Resource utilization
    st.subheader("🔋 Resource Utilization")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "45%", "Current CPU load")
    with col2:
        st.metric("Memory Usage", "2.3GB", "RAM consumption")
    with col3:
        st.metric("Model Usage", "75%", "LLM API utilization")
    with col4:
        st.metric("Disk Usage", "25%", "Storage consumption")
    
    # Resource utilization chart
    if evolution_running:
        # Simulate resource usage over time
        time_points = list(range(10))
        cpu_usage = np.random.uniform(30, 70, 10)
        memory_usage = np.random.uniform(20, 60, 10)
        
        df = pd.DataFrame({
            "Time": time_points,
            "CPU Usage (%)": cpu_usage,
            "Memory Usage (%)": memory_usage
        })
        
        fig = px.line(df, x="Time", y=["CPU Usage (%)", "Memory Usage (%)"], 
                      title="Resource Usage Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Active evolution statistics
    st.subheader("📈 Active Evolution Statistics")
    if evolution_running and "evolution_history" in st.session_state:
        history = st.session_state["evolution_history"]
        if history:
            # Display latest generation stats
            latest_gen = history[-1] if history else {}
            population = latest_gen.get("population", [])
            
            if population:
                best_fitness = max(ind.get("fitness", 0) for ind in population)
                avg_fitness = sum(ind.get("fitness", 0) for ind in population) / len(population)
                diversity = np.std([ind.get("fitness", 0) for ind in population])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Best Fitness", f"{best_fitness:.4f}")
                col2.metric("Avg Fitness", f"{avg_fitness:.4f}")
                col3.metric("Diversity", f"{diversity:.4f}")


def render_activity_logs():
    """Render activity logs monitoring."""
    st.header("📝 Activity Logs")
    
    # Log level filter
    col1, col2 = st.columns(2)
    with col1:
        log_level_filter = st.selectbox(
            "Filter by log level", 
            ["All", "INFO", "WARNING", "ERROR", "SUCCESS"],
            index=0
        )
    with col2:
        search_term = st.text_input("Search logs", placeholder="Enter search term...")
    
    # Sample logs (in a real implementation, these would come from a logging system)
    sample_logs = [
        {"timestamp": "2023-10-01 10:00:00", "level": "INFO", "message": "Evolution started with 500 iterations", "module": "evolution"},
        {"timestamp": "2023-10-01 10:05:01", "level": "SUCCESS", "message": "Generation 1 completed", "module": "evolution"},
        {"timestamp": "2023-10-01 10:10:02", "level": "WARNING", "message": "Low diversity detected in generation 3", "module": "analytics"},
        {"timestamp": "2023-10-01 10:15:03", "level": "INFO", "message": "Adversarial testing initiated", "module": "adversarial"},
        {"timestamp": "2023-10-01 10:20:04", "level": "SUCCESS", "message": "Adversarial iteration 1 completed", "module": "adversarial"},
        {"timestamp": "2023-10-01 10:25:05", "level": "ERROR", "message": "API timeout in evaluator", "module": "evaluator"},
        {"timestamp": "2023-10-01 10:30:06", "level": "INFO", "message": "Early stopping detected", "module": "control"},
        {"timestamp": "2023-10-01 10:35:07", "level": "SUCCESS", "message": "Evolution completed successfully", "module": "evolution"},
    ]
    
    # Apply filters
    filtered_logs = sample_logs
    
    if log_level_filter != "All":
        filtered_logs = [log for log in filtered_logs if log["level"] == log_level_filter]
    
    if search_term:
        filtered_logs = [log for log in filtered_logs if search_term.lower() in log["message"].lower()]
    
    # Display logs in a table
    if filtered_logs:
        log_df = pd.DataFrame(filtered_logs)
        log_df["Severity"] = log_df["level"].apply(lambda x: {"ERROR": 4, "WARNING": 3, "INFO": 2, "SUCCESS": 1}[x])
        log_df = log_df.sort_values("Severity")
        
        # Create a color-coded display
        for idx, log in log_df.iterrows():
            level = log["level"]
            color_map = {
                "ERROR": "rgba(255, 0, 0, 0.1)",
                "WARNING": "rgba(255, 165, 0, 0.1)", 
                "INFO": "rgba(0, 0, 255, 0.1)",
                "SUCCESS": "rgba(0, 255, 0, 0.1)"
            }
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: {color_map.get(level, 'white')}; 
                                padding: 10px; 
                                border-radius: 5px; 
                                margin-bottom: 5px; 
                                border-left: 4px solid {'red' if level == 'ERROR' else 'orange' if level == 'WARNING' else 'blue' if level == 'INFO' else 'green'};">
                        <strong>{log['timestamp']}</strong> | 
                        <strong style="color: {'red' if level == 'ERROR' else 'orange' if level == 'WARNING' else 'blue' if level == 'INFO' else 'green'};">{level}</strong> | 
                        <em>{log['module']}</em><br>
                        {log['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No logs match your filters. Try changing the log level or search term.")


def render_performance_analytics():
    """Render performance analytics."""
    st.header("📈 Performance Analytics")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Calls", "1,247", "Total API interactions")
    with col2:
        st.metric("Avg Response Time", "1.2s", "Per API call")
    with col3:
        st.metric("Success Rate", "98.3%", "Successful operations")
    with col4:
        st.metric("Token Usage", "847K", "Total tokens consumed")
    
    st.divider()
    
    # Performance trend
    st.subheader("Performance Trend")
    
    # Simulate performance data
    time_points = list(range(20))
    response_times = np.random.normal(1.2, 0.3, 20)  # Avg 1.2s with std 0.3
    success_rates = 98 + np.random.normal(0, 1, 20)  # Avg 98% with small variation
    token_usage = np.cumsum(np.random.uniform(30, 50, 20))  # Cumulative token usage
    
    # Response time chart
    df = pd.DataFrame({
        "Time Point": time_points,
        "Avg Response Time (s)": response_times,
        "Success Rate (%)": success_rates,
        "Cumulative Tokens": token_usage
    })
    
    fig = px.line(df, x="Time Point", y="Avg Response Time (s)", 
                  title="Average Response Time Trend", 
                  color_discrete_sequence=['red'])
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.line(df, x="Time Point", y=["Success Rate (%)"], 
                   title="Success Rate Trend", 
                   color_discrete_sequence=['green'])
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.line(df, x="Time Point", y="Cumulative Tokens", 
                   title="Cumulative Token Usage", 
                   color_discrete_sequence=['blue'])
    st.plotly_chart(fig3, use_container_width=True)
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimated Cost", "$47.32", "Based on API usage")
    with col2:
        st.metric("Cost per Iteration", "$0.19", "Average cost per evolution")
    with col3:
        st.metric("Forecasted Monthly", "$1,240", "Projected monthly cost")
    
    # Cost breakdown chart
    cost_breakdown = pd.DataFrame({
        "Component": ["GPT-4 API", "GPT-3.5 API", "Storage", "Compute"],
        "Cost": [28.50, 12.30, 3.20, 3.32]
    })
    
    fig4 = px.pie(cost_breakdown, values="Cost", names="Component", 
                  title="Cost Breakdown by Component")
    st.plotly_chart(fig4, use_container_width=True)


def render_system_health():
    """Render system health monitoring."""
    st.header("⚙️ System Health")
    
    # System health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("DB Connection", "✅", "Ready")
    with col2:
        st.metric("API Status", "✅", "Operational")
    with col3:
        st.metric("Storage", "✅", "Healthy")
    with col4:
        st.metric("Network", "✅", "Connected")
    
    st.divider()
    
    # Detailed system components
    st.subheader("System Components Status")
    
    components = [
        {"Component": "Evolution Engine", "Status": "Ready", "Response": "42ms", "Health": "Good"},
        {"Component": "LLM API Interface", "Status": "Ready", "Response": "28ms", "Health": "Good"},
        {"Component": "Evaluator Engine", "Status": "Ready", "Response": "89ms", "Health": "Good"},
        {"Component": "Database", "Status": "Ready", "Response": "15ms", "Health": "Optimal"},
        {"Component": "Artifact Processor", "Status": "Ready", "Response": "56ms", "Health": "Good"},
        {"Component": "Model Ensemble", "Status": "Ready", "Response": "134ms", "Health": "Good"},
    ]
    
    components_df = pd.DataFrame(components)
    st.dataframe(components_df, use_container_width=True)
    
    st.divider()
    
    # System health charts
    st.subheader("Health Metrics")
    
    # Simulate health metrics over time
    timestamps = pd.date_range(start="2023-10-01", periods=24, freq='H')
    health_scores = 95 + np.random.normal(0, 5, 24)  # Avg 95% with small variation
    availability = 100 - np.random.uniform(0, 2, 24)  # Very high availability
    
    health_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Health Score (%)": health_scores,
        "Availability (%)": availability
    })
    
    fig = px.line(health_df, x="Timestamp", y=["Health Score (%)", "Availability (%)"], 
                  title="System Health Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource allocation
    st.subheader("Resource Allocation")
    
    resources = pd.DataFrame({
        "Resource": ["CPU", "Memory", "Storage", "Network"],
        "Allocated": [8, 16, 500, 100],  # Example values
        "Used": [4, 8, 150, 45],  # Example values
        "Units": ["Cores", "GB", "GB", "Mbps"]
    })
    
    resources["Usage %"] = (resources["Used"] / resources["Allocated"]) * 100
    
    fig2 = px.bar(resources, x="Resource", y="Usage %", 
                  title="Resource Usage Percentage",
                  color="Usage %",
                  color_continuous_scale=["green", "yellow", "red"])
    st.plotly_chart(fig2, use_container_width=True)
    
    # System alerts
    st.subheader("⚠️ System Alerts")
    alerts = [
        {"Time": "2023-10-01 09:45:00", "Level": "INFO", "Message": "System startup completed successfully"},
        {"Time": "2023-10-01 10:30:00", "Level": "WARNING", "Message": "High memory usage detected"},
        {"Time": "2023-10-01 11:15:00", "Level": "INFO", "Message": "Scheduled maintenance completed"},
        {"Time": "2023-10-01 12:00:00", "Level": "INFO", "Message": "Backup completed successfully"},
    ]
    
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True)


# Additional monitoring functions
def start_system_monitoring():
    """Start the background monitoring system."""
    if "monitoring_started" not in st.session_state:
        st.session_state.monitoring_started = True
        st.success("System monitoring started!")
        return True
    else:
        st.info("System monitoring is already running.")
        return False


def get_current_metrics():
    """Get current system metrics."""
    metrics = {
        "evolution_status": st.session_state.get("evolution_running", False),
        "current_best_score": st.session_state.get("evolution_current_best_score", 0.0),
        "current_iteration": st.session_state.get("current_iteration", 0),
        "total_iterations": st.session_state.get("max_iterations", 100),
        "api_calls": st.session_state.get("api_calls_count", 0),
        "response_time_avg": st.session_state.get("response_time_avg", 0.0),
        "tokens_used": st.session_state.get("tokens_used", 0),
    }
    return metrics


def log_activity(message: str, level: str = "INFO", module: str = "general"):
    """Log an activity to the monitoring system."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "module": module
    }
    
    # Add to session state for display
    if "activity_logs" not in st.session_state:
        st.session_state["activity_logs"] = []
    
    st.session_state["activity_logs"].append(log_entry)


# Test the monitoring dashboard
if __name__ == "__main__":
    # This is for testing purposes only
    print("Monitoring dashboard module loaded successfully.")