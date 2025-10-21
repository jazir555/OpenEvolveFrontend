"""
Resource Manager Module

This module provides resource tracking, management, and optimization for the
decomposition workflow, including API call limits, token usage, and cost tracking.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ResourceUsage:
    """Tracks resource usage for a workflow or component."""
    api_calls: int = 0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    """Defines resource limits for a workflow."""
    max_api_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_execution_time_seconds: Optional[float] = None
    max_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_api_calls': self.max_api_calls,
            'max_tokens': self.max_tokens,
            'max_cost': self.max_cost,
            'max_execution_time_seconds': self.max_execution_time_seconds,
            'max_memory_mb': self.max_memory_mb
        }


class ResourceManager:
    """Manages resource tracking and enforcement for workflows."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize the resource manager.
        
        Args:
            limits: Optional resource limits to enforce
        """
        self.limits = limits or ResourceLimits()
        self.usage = ResourceUsage()
        self.component_usage: Dict[str, ResourceUsage] = {}
        self.start_time = time.time()
        
        # Cost per token for different models (approximate)
        self.cost_per_token = {
            'gpt-4': 0.00003,
            'gpt-4-turbo': 0.00001,
            'gpt-3.5-turbo': 0.000002,
            'claude-3-opus': 0.000015,
            'claude-3-sonnet': 0.000003,
            'claude-3-haiku': 0.00000025,
            'default': 0.00001
        }
    
    def track_api_call(
        self,
        component: str,
        model: str,
        tokens: int,
        execution_time: float = 0.0
    ):
        """
        Track an API call and its resource usage.
        
        Args:
            component: Name of the component making the call
            model: Model being used
            tokens: Number of tokens used
            execution_time: Time taken for the call in seconds
        """
        # Update global usage
        self.usage.api_calls += 1
        self.usage.tokens_used += tokens
        self.usage.execution_time_seconds += execution_time
        
        # Calculate cost
        cost_rate = self._get_cost_rate(model)
        cost = tokens * cost_rate
        self.usage.estimated_cost += cost
        
        # Update component usage
        if component not in self.component_usage:
            self.component_usage[component] = ResourceUsage()
        
        comp_usage = self.component_usage[component]
        comp_usage.api_calls += 1
        comp_usage.tokens_used += tokens
        comp_usage.execution_time_seconds += execution_time
        comp_usage.estimated_cost += cost
        comp_usage.details[model] = comp_usage.details.get(model, 0) + tokens
    
    def _get_cost_rate(self, model: str) -> float:
        """Get cost per token for a model."""
        # Try exact match
        if model in self.cost_per_token:
            return self.cost_per_token[model]
        
        # Try partial match
        for model_key in self.cost_per_token:
            if model_key in model.lower():
                return self.cost_per_token[model_key]
        
        return self.cost_per_token['default']
    
    def check_limits(self) -> tuple[bool, List[str]]:
        """
        Check if current usage exceeds any limits.
        
        Returns:
            Tuple of (within_limits, violations)
        """
        violations = []
        
        # Check API calls
        if self.limits.max_api_calls and self.usage.api_calls >= self.limits.max_api_calls:
            violations.append(
                f"API call limit exceeded: {self.usage.api_calls}/{self.limits.max_api_calls}"
            )
        
        # Check tokens
        if self.limits.max_tokens and self.usage.tokens_used >= self.limits.max_tokens:
            violations.append(
                f"Token limit exceeded: {self.usage.tokens_used}/{self.limits.max_tokens}"
            )
        
        # Check cost
        if self.limits.max_cost and self.usage.estimated_cost >= self.limits.max_cost:
            violations.append(
                f"Cost limit exceeded: ${self.usage.estimated_cost:.2f}/${self.limits.max_cost:.2f}"
            )
        
        # Check execution time
        elapsed_time = time.time() - self.start_time
        if self.limits.max_execution_time_seconds and elapsed_time >= self.limits.max_execution_time_seconds:
            violations.append(
                f"Time limit exceeded: {elapsed_time:.1f}s/{self.limits.max_execution_time_seconds:.1f}s"
            )
        
        # Check memory (if available)
        if self.limits.max_memory_mb and self.usage.memory_usage_mb >= self.limits.max_memory_mb:
            violations.append(
                f"Memory limit exceeded: {self.usage.memory_usage_mb:.1f}MB/{self.limits.max_memory_mb:.1f}MB"
            )
        
        return len(violations) == 0, violations
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'api_calls': self.usage.api_calls,
            'tokens_used': self.usage.tokens_used,
            'estimated_cost': self.usage.estimated_cost,
            'execution_time_seconds': elapsed_time,
            'memory_usage_mb': self.usage.memory_usage_mb,
            'limits': self.limits.to_dict(),
            'component_breakdown': {
                comp: {
                    'api_calls': usage.api_calls,
                    'tokens_used': usage.tokens_used,
                    'estimated_cost': usage.estimated_cost,
                    'execution_time_seconds': usage.execution_time_seconds
                }
                for comp, usage in self.component_usage.items()
            }
        }
    
    def optimize_resource_allocation(self, sub_problems: List[Any]) -> Dict[str, Any]:
        """
        Suggest optimal resource allocation for sub-problems.
        
        Args:
            sub_problems: List of sub-problems to allocate resources for
            
        Returns:
            Dictionary of optimization suggestions
        """
        suggestions = {
            'priority_order': [],
            'resource_allocation': {},
            'estimated_total_cost': 0.0,
            'estimated_total_time': 0.0
        }
        
        # Sort by complexity (higher complexity = more resources)
        sorted_problems = sorted(
            sub_problems,
            key=lambda sp: sp.ai_suggested_complexity_score,
            reverse=True
        )
        
        # Allocate resources proportionally
        total_complexity = sum(sp.ai_suggested_complexity_score for sp in sub_problems)
        
        for sp in sorted_problems:
            # Calculate resource allocation
            complexity_ratio = sp.ai_suggested_complexity_score / total_complexity if total_complexity > 0 else 0
            
            # Estimate resources needed
            estimated_tokens = int(complexity_ratio * 10000)  # Base estimate
            estimated_cost = estimated_tokens * 0.00001  # Average cost
            estimated_time = complexity_ratio * 60  # Base time in seconds
            
            suggestions['priority_order'].append(sp.id)
            suggestions['resource_allocation'][sp.id] = {
                'estimated_tokens': estimated_tokens,
                'estimated_cost': estimated_cost,
                'estimated_time_seconds': estimated_time,
                'priority': len(suggestions['priority_order'])
            }
            
            suggestions['estimated_total_cost'] += estimated_cost
            suggestions['estimated_total_time'] += estimated_time
        
        return suggestions
    
    def export_usage_report(self, filepath: str):
        """Export usage report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_usage_summary(),
            'limits': self.limits.to_dict(),
            'within_limits': self.check_limits()[0]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def get_remaining_resources(self) -> Dict[str, Any]:
        """Get remaining resources based on limits."""
        remaining = {}
        
        if self.limits.max_api_calls:
            remaining['api_calls'] = max(0, self.limits.max_api_calls - self.usage.api_calls)
        
        if self.limits.max_tokens:
            remaining['tokens'] = max(0, self.limits.max_tokens - self.usage.tokens_used)
        
        if self.limits.max_cost:
            remaining['cost'] = max(0, self.limits.max_cost - self.usage.estimated_cost)
        
        if self.limits.max_execution_time_seconds:
            elapsed = time.time() - self.start_time
            remaining['time_seconds'] = max(0, self.limits.max_execution_time_seconds - elapsed)
        
        return remaining


def render_resource_dashboard(resource_manager: ResourceManager):
    """
    Render resource usage dashboard in Streamlit.
    
    Args:
        resource_manager: ResourceManager instance to display
    """
    import streamlit as st
    import plotly.graph_objects as go
    
    st.subheader("📊 Resource Usage Dashboard")
    
    summary = resource_manager.get_usage_summary()
    within_limits, violations = resource_manager.check_limits()
    
    # Status indicator
    if within_limits:
        st.success("✓ All resource usage within limits")
    else:
        st.error("⚠️ Resource limits exceeded!")
        for violation in violations:
            st.write(f"- {violation}")
    
    # Current usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "API Calls",
            summary['api_calls'],
            delta=f"Limit: {summary['limits']['max_api_calls']}" if summary['limits']['max_api_calls'] else "No limit"
        )
    
    with col2:
        st.metric(
            "Tokens Used",
            f"{summary['tokens_used']:,}",
            delta=f"Limit: {summary['limits']['max_tokens']:,}" if summary['limits']['max_tokens'] else "No limit"
        )
    
    with col3:
        st.metric(
            "Estimated Cost",
            f"${summary['estimated_cost']:.4f}",
            delta=f"Limit: ${summary['limits']['max_cost']:.2f}" if summary['limits']['max_cost'] else "No limit"
        )
    
    with col4:
        st.metric(
            "Execution Time",
            f"{summary['execution_time_seconds']:.1f}s",
            delta=f"Limit: {summary['limits']['max_execution_time_seconds']:.1f}s" if summary['limits']['max_execution_time_seconds'] else "No limit"
        )
    
    # Component breakdown
    if summary['component_breakdown']:
        st.subheader("Component Breakdown")
        
        # Prepare data for visualization
        components = list(summary['component_breakdown'].keys())
        api_calls = [summary['component_breakdown'][c]['api_calls'] for c in components]
        tokens = [summary['component_breakdown'][c]['tokens_used'] for c in components]
        costs = [summary['component_breakdown'][c]['estimated_cost'] for c in components]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # API calls by component
            fig = go.Figure(data=[go.Bar(x=components, y=api_calls)])
            fig.update_layout(
                title='API Calls by Component',
                xaxis_title='Component',
                yaxis_title='API Calls',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost by component
            fig = go.Figure(data=[go.Bar(x=components, y=costs)])
            fig.update_layout(
                title='Cost by Component',
                xaxis_title='Component',
                yaxis_title='Cost ($)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Remaining resources
    remaining = resource_manager.get_remaining_resources()
    if remaining:
        st.subheader("Remaining Resources")
        
        cols = st.columns(len(remaining))
        for i, (resource, amount) in enumerate(remaining.items()):
            with cols[i]:
                if resource == 'cost':
                    st.metric(resource.replace('_', ' ').title(), f"${amount:.4f}")
                elif resource == 'time_seconds':
                    st.metric("Time", f"{amount:.1f}s")
                else:
                    st.metric(resource.replace('_', ' ').title(), f"{amount:,}")


def create_resource_limits_from_config(config: Dict[str, Any]) -> ResourceLimits:
    """
    Create ResourceLimits from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResourceLimits instance
    """
    return ResourceLimits(
        max_api_calls=config.get('max_api_calls'),
        max_tokens=config.get('max_tokens'),
        max_cost=config.get('max_cost'),
        max_execution_time_seconds=config.get('max_execution_time_seconds'),
        max_memory_mb=config.get('max_memory_mb')
    )


def get_default_resource_limits() -> ResourceLimits:
    """Get default resource limits."""
    return ResourceLimits(
        max_api_calls=1000,
        max_tokens=1000000,
        max_cost=10.0,
        max_execution_time_seconds=3600,  # 1 hour
        max_memory_mb=2048  # 2 GB
    )
