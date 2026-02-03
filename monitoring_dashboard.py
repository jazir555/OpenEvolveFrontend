"""
Monitoring Dashboard - License: Apache 2.0

Web-based monitoring dashboard for OpenEvolve services.
Provides real-time metrics, logs, and health status.

Dependencies (all permissive licenses):
- streamlit: Apache 2.0
- plotly: MIT License
- pandas: BSD License

Author: OpenEvolve
Date: 2026-02-02
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import streamlit as st

# Plotly - MIT License
import plotly.graph_objects as go
import plotly.express as px

# Pandas - BSD License
import pandas as pd

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ServiceMetrics:
    """Metrics for a single service."""
    name: str
    status: str
    uptime_seconds: float
    requests_total: int
    requests_per_second: float
    error_rate: float
    response_time_ms: float
    cpu_percent: float
    memory_mb: float
    last_check: datetime


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    timestamp: datetime
    services: List[ServiceMetrics]
    total_requests: int
    total_errors: int
    avg_response_time: float
    throughput_rps: float


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_header():
    """Render dashboard header."""
    st.set_page_config(
        page_title="OpenEvolve Monitoring",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 OpenEvolve Monitoring Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Auto-refresh
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh",
            value=st.session_state.get('auto_refresh', True)
        )
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)",
                min_value=1,
                max_value=60,
                value=5
            )
            st.session_state.refresh_interval = refresh_interval
        
        # Time range
        st.subheader("Time Range")
        time_range = st.selectbox(
            "Select range",
            ["Last 5 minutes", "Last 15 minutes", "Last 1 hour", "Last 6 hours", "Last 24 hours"]
        )
        st.session_state.time_range = time_range
        
        # Service filter
        st.subheader("Services")
        services = ["All", "REST API", "GraphQL", "Event Bus", "MCP Server", "Telemetry"]
        selected_service = st.selectbox("Filter by service", services)
        st.session_state.selected_service = selected_service
        
        # Actions
        st.subheader("Actions")
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        if st.button("📊 Export Metrics"):
            export_metrics()
        
        if st.button("🔔 Configure Alerts"):
            st.info("Alert configuration would open here")


def render_overview_cards(metrics: SystemMetrics):
    """Render overview metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🟢 Services Up",
            value=sum(1 for s in metrics.services if s.status == "healthy"),
            delta=f"of {len(metrics.services)} total"
        )
    
    with col2:
        st.metric(
            label="📈 Throughput",
            value=f"{metrics.throughput_rps:.1f}",
            delta="requests/sec"
        )
    
    with col3:
        st.metric(
            label="⚡ Avg Response",
            value=f"{metrics.avg_response_time:.0f}ms",
            delta=f"{metrics.total_errors} errors",
            delta_color="inverse"
        )
    
    with col4:
        total_uptime = sum(s.uptime_seconds for s in metrics.services) / max(len(metrics.services), 1)
        hours = int(total_uptime // 3600)
        mins = int((total_uptime % 3600) // 60)
        st.metric(
            label="⏱️ Avg Uptime",
            value=f"{hours}h {mins}m"
        )


def render_service_status(services: List[ServiceMetrics]):
    """Render service status table."""
    st.subheader("📊 Service Status")
    
    # Create DataFrame
    data = []
    for s in services:
        data.append({
            "Service": s.name,
            "Status": s.status,
            "Requests/sec": f"{s.requests_per_second:.2f}",
            "Error Rate": f"{s.error_rate*100:.1f}%",
            "Response Time": f"{s.response_time_ms:.0f}ms",
            "CPU": f"{s.cpu_percent:.1f}%",
            "Memory": f"{s.memory_mb:.0f}MB",
            "Uptime": format_duration(s.uptime_seconds)
        })
    
    df = pd.DataFrame(data)
    
    # Style status column
    def color_status(val):
        if val == "healthy":
            return "background-color: #90EE90; color: black"
        elif val == "degraded":
            return "background-color: #FFE4B5; color: black"
        else:
            return "background-color: #FFB6C1; color: black"
    
    styled_df = df.style.map(color_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True)


def render_charts(services: List[ServiceMetrics]):
    """Render performance charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Request Rate by Service")
        
        fig = go.Figure()
        for s in services:
            fig.add_trace(go.Bar(
                name=s.name,
                x=[s.name],
                y=[s.requests_per_second],
                marker_color=get_status_color(s.status)
            ))
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Service",
            yaxis_title="Requests/sec",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Response Time Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[s.name for s in services],
            y=[s.response_time_ms for s in services],
            mode='markers+lines',
            marker=dict(
                size=[s.requests_total/100 for s in services],
                color=[s.error_rate*100 for s in services],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Error %")
            )
        ))
        
        fig.update_layout(
            xaxis_title="Service",
            yaxis_title="Response Time (ms)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def render_resource_usage(services: List[ServiceMetrics]):
    """Render resource usage charts."""
    st.subheader("💾 Resource Usage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU usage pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[s.name for s in services],
            values=[s.cpu_percent for s in services],
            hole=0.4
        )])
        fig.update_layout(title="CPU Usage by Service", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory usage bar chart
        fig = go.Figure(data=[go.Bar(
            x=[s.name for s in services],
            y=[s.memory_mb for s in services],
            marker_color='lightblue'
        )])
        fig.update_layout(
            title="Memory Usage (MB)",
            xaxis_title="Service",
            yaxis_title="Memory (MB)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def render_logs_section():
    """Render logs viewer."""
    st.subheader("📝 Recent Logs")
    
    # Log level filter
    log_levels = st.multiselect(
        "Log levels",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=["INFO", "WARNING", "ERROR"]
    )
    
    # Simulated logs (would come from actual log aggregation)
    logs = [
        {"time": "2026-02-02 13:10:05", "level": "INFO", "service": "REST API", "message": "Request completed in 45ms"},
        {"time": "2026-02-02 13:10:03", "level": "WARNING", "service": "GraphQL", "message": "Slow query detected (2.3s)"},
        {"time": "2026-02-02 13:09:58", "level": "ERROR", "service": "Event Bus", "message": "Connection timeout to Valkey"},
        {"time": "2026-02-02 13:09:55", "level": "INFO", "service": "MCP Server", "message": "Tool executed: decompose_problem"},
    ]
    
    log_df = pd.DataFrame(logs)
    st.dataframe(log_df, use_container_width=True)
    
    # Log file download
    if st.button("📥 Download Full Logs"):
        st.info("This would download log files")


def render_alerts_section():
    """Render active alerts."""
    st.subheader("🚨 Active Alerts")
    
    # Simulated alerts
    alerts = [
        {"severity": "high", "service": "Event Bus", "message": "Connection failures", "since": "5 min ago"},
        {"severity": "medium", "service": "GraphQL", "message": "High response time", "since": "15 min ago"},
        {"severity": "low", "service": "Telemetry", "message": "Metric buffer at 80%", "since": "1 hour ago"},
    ]
    
    for alert in alerts:
        severity_colors = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🔵"
        }
        
        with st.expander(f"{severity_colors[alert['severity']]} {alert['service']}: {alert['message']}"):
            st.write(f"Severity: {alert['severity']}")
            st.write(f"Since: {alert['since']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Acknowledge", key=f"ack_{alert['service']}"):
                    st.success("Alert acknowledged")
            with col2:
                if st.button("📝 View Details", key=f"det_{alert['service']}"):
                    st.info("Detailed alert information would appear here")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_status_color(status: str) -> str:
    """Get color for status."""
    return {
        "healthy": "#90EE90",
        "degraded": "#FFE4B5",
        "error": "#FFB6C1",
        "stopped": "#D3D3D3"
    }.get(status, "#D3D3D3")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m"
    elif seconds < 86400:
        return f"{int(seconds // 3600)}h"
    else:
        return f"{int(seconds // 86400)}d"


def fetch_metrics() -> SystemMetrics:
    """Fetch metrics from services."""
    # In production, this would query actual services
    # For demo, generate simulated metrics
    
    services = [
        ServiceMetrics(
            name="REST API",
            status="healthy",
            uptime_seconds=86400 + 3600,
            requests_total=15420,
            requests_per_second=45.3,
            error_rate=0.001,
            response_time_ms=45,
            cpu_percent=12.5,
            memory_mb=128.0,
            last_check=datetime.now()
        ),
        ServiceMetrics(
            name="GraphQL",
            status="degraded",
            uptime_seconds=86400,
            requests_total=8320,
            requests_per_second=23.1,
            error_rate=0.03,
            response_time_ms=120,
            cpu_percent=15.0,
            memory_mb=96.0,
            last_check=datetime.now()
        ),
        ServiceMetrics(
            name="Event Bus",
            status="error",
            uptime_seconds=1800,
            requests_total=45000,
            requests_per_second=125.0,
            error_rate=0.15,
            response_time_ms=250,
            cpu_percent=8.0,
            memory_mb=64.0,
            last_check=datetime.now()
        ),
        ServiceMetrics(
            name="MCP Server",
            status="healthy",
            uptime_seconds=86400 + 7200,
            requests_total=3210,
            requests_per_second=8.9,
            error_rate=0.0,
            response_time_ms=85,
            cpu_percent=5.0,
            memory_mb=48.0,
            last_check=datetime.now()
        ),
        ServiceMetrics(
            name="Telemetry",
            status="healthy",
            uptime_seconds=86400 + 3600,
            requests_total=150000,
            requests_per_second=416.7,
            error_rate=0.0001,
            response_time_ms=5,
            cpu_percent=3.0,
            memory_mb=32.0,
            last_check=datetime.now()
        )
    ]
    
    return SystemMetrics(
        timestamp=datetime.now(),
        services=services,
        total_requests=sum(s.requests_total for s in services),
        total_errors=sum(int(s.requests_total * s.error_rate) for s in services),
        avg_response_time=sum(s.response_time_ms for s in services) / len(services),
        throughput_rps=sum(s.requests_per_second for s in services)
    )


def export_metrics():
    """Export metrics to file."""
    metrics = fetch_metrics()
    
    # Convert to dict
    data = {
        "timestamp": metrics.timestamp.isoformat(),
        "total_requests": metrics.total_requests,
        "total_errors": metrics.total_errors,
        "avg_response_time": metrics.avg_response_time,
        "throughput_rps": metrics.throughput_rps,
        "services": [asdict(s) for s in metrics.services]
    }
    
    # Save
    output_path = Path(f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.write_text(json.dumps(data, indent=2, default=str))
    
    st.success(f"Metrics exported to {output_path}")


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main dashboard entry point."""
    render_header()
    render_sidebar()
    
    # Fetch metrics
    metrics = fetch_metrics()
    
    # Overview section
    render_overview_cards(metrics)
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Status", "📈 Metrics", "📝 Logs", "🚨 Alerts"])
    
    with tab1:
        render_service_status(metrics.services)
        render_resource_usage(metrics.services)
    
    with tab2:
        render_charts(metrics.services)
        
        # Historical metrics (placeholder)
        st.subheader("📊 Historical Performance")
        st.info("Historical metrics visualization would appear here")
        
        # Custom time range selector
        st.select_slider(
            "Time Range",
            options=["1h", "6h", "24h", "7d", "30d"],
            value="24h"
        )
    
    with tab3:
        render_logs_section()
    
    with tab4:
        render_alerts_section()
    
    # Auto-refresh
    if st.session_state.get('auto_refresh', False):
        interval = st.session_state.get('refresh_interval', 5)
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
