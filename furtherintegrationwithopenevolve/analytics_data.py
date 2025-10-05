"""
Analytics Data Module for OpenEvolve - Data generation and processing
This module provides functions for generating analytics data for visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import time


def get_population_diversity(evolution_id: str) -> pd.DataFrame:
    """Get population diversity data for visualization.
    
    Args:
        evolution_id (str): ID of the evolution run
        
    Returns:
        pd.DataFrame: DataFrame with diversity data
    """
    # In a real implementation, this would fetch data from the backend
    # For now, we'll generate synthetic data
    data = {
        "iteration": np.arange(50),
        "diversity": np.random.rand(50) * 0.5 + 0.2
    }
    return pd.DataFrame(data)


def get_code_complexity(evolution_id: str) -> pd.DataFrame:
    """Get code complexity data for visualization.
    
    Args:
        evolution_id (str): ID of the evolution run
        
    Returns:
        pd.DataFrame: DataFrame with complexity data
    """
    # In a real implementation, this would fetch data from the backend
    # For now, we'll generate synthetic data
    data = {
        "iteration": np.arange(50),
        "complexity": np.random.randint(10, 30, 50)
    }
    return pd.DataFrame(data)


def get_linter_scores(evolution_id: str) -> pd.DataFrame:
    """Get linter scores for visualization.
    
    Args:
        evolution_id (str): ID of the evolution run
        
    Returns:
        pd.DataFrame: DataFrame with linter scores
    """
    # In a real implementation, this would fetch data from the backend
    # For now, we'll generate synthetic data
    data = {
        "iteration": np.arange(50),
        "linter_score": np.random.rand(50) * 4 + 6
    }
    return pd.DataFrame(data)


def get_evolution_performance_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get evolution performance data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with performance data
    """
    # Extract data from session state
    history = session_state.get("evolution_history", [])
    
    if not history:
        # Return empty DataFrame if no history
        return pd.DataFrame()
    
    # Process evolution history
    data = []
    for generation in history:
        gen_idx = generation.get("generation", 0)
        population = generation.get("population", [])
        
        for individual in population:
            data.append({
                "generation": gen_idx,
                "fitness": individual.get("fitness", 0),
                "complexity": individual.get("complexity", 0),
                "diversity": individual.get("diversity", 0),
                "code": individual.get("code", "")[:100] + "..." if len(individual.get("code", "")) > 100 else individual.get("code", "")
            })
    
    return pd.DataFrame(data)


def get_adversarial_performance_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get adversarial performance data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with adversarial performance data
    """
    # Extract data from session state
    results = session_state.get("adversarial_results", {})
    iterations = results.get("iterations", [])
    
    if not iterations:
        # Return empty DataFrame if no iterations
        return pd.DataFrame()
    
    # Process adversarial iterations
    data = []
    for iteration in iterations:
        iter_idx = iteration.get("iteration", 0)
        approval_check = iteration.get("approval_check", {})
        approval_rate = approval_check.get("approval_rate", 0)
        
        # Count issues from critiques
        total_issues = 0
        critiques = iteration.get("critiques", [])
        for critique in critiques:
            critique_json = critique.get("critique_json", {})
            if critique_json:
                issues = critique_json.get("issues", [])
                total_issues += len(issues)
        
        data.append({
            "iteration": iter_idx,
            "approval_rate": approval_rate,
            "total_issues": total_issues,
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_model_performance_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get model performance data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with model performance data
    """
    # Extract data from session state
    model_performance = session_state.get("adversarial_model_performance", {})
    
    if not model_performance:
        # Return empty DataFrame if no model performance data
        return pd.DataFrame()
    
    # Process model performance data
    data = []
    for model_id, perf_data in model_performance.items():
        data.append({
            "model": model_id,
            "score": perf_data.get("score", 0),
            "issues_found": perf_data.get("issues_found", 0),
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_cost_and_resource_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get cost and resource usage data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with cost and resource data
    """
    # Extract data from session state
    evolution_history = session_state.get("evolution_history", [])
    adversarial_results = session_state.get("adversarial_results", {})
    
    # Process data
    data = []
    
    # Evolution costs
    for generation in evolution_history:
        gen_idx = generation.get("generation", 0)
        cost_estimate = generation.get("cost_estimate_usd", 0)
        prompt_tokens = generation.get("tokens", {}).get("prompt", 0)
        completion_tokens = generation.get("tokens", {}).get("completion", 0)
        
        data.append({
            "type": "evolution",
            "iteration": gen_idx,
            "cost_usd": cost_estimate,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": time.time()
        })
    
    # Adversarial costs
    adversarial_iterations = adversarial_results.get("iterations", [])
    for iteration in adversarial_iterations:
        iter_idx = iteration.get("iteration", 0)
        cost_estimate = iteration.get("cost_estimate_usd", 0)
        prompt_tokens = iteration.get("tokens", {}).get("prompt", 0)
        completion_tokens = iteration.get("tokens", {}).get("completion", 0)
        
        data.append({
            "type": "adversarial",
            "iteration": iter_idx,
            "cost_usd": cost_estimate,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_content_quality_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get content quality metrics data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with content quality data
    """
    # Extract data from session state
    protocol_text = session_state.get("protocol_text", "")
    evolution_history = session_state.get("evolution_history", [])
    adversarial_results = session_state.get("adversarial_results", {})
    
    # Process data
    data = []
    
    # Initial content quality
    if protocol_text:
        data.append({
            "stage": "initial",
            "quality_score": len(protocol_text) / 1000,  # Simplified quality score
            "word_count": len(protocol_text.split()),
            "character_count": len(protocol_text),
            "timestamp": time.time()
        })
    
    # Evolution quality improvements
    for generation in evolution_history:
        gen_idx = generation.get("generation", 0)
        population = generation.get("population", [])
        if population:
            best_individual = max(population, key=lambda x: x.get("fitness", 0))
            best_code = best_individual.get("code", "")
            data.append({
                "stage": f"evolution_{gen_idx}",
                "quality_score": best_individual.get("fitness", 0) * 100,  # Scale to 0-100
                "word_count": len(best_code.split()),
                "character_count": len(best_code),
                "timestamp": time.time()
            })
    
    # Final adversarial quality
    if adversarial_results:
        final_sop = adversarial_results.get("final_sop", "")
        final_approval_rate = adversarial_results.get("final_approval_rate", 0)
        data.append({
            "stage": "final",
            "quality_score": final_approval_rate,
            "word_count": len(final_sop.split()),
            "character_count": len(final_sop),
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_issue_resolution_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get issue resolution data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with issue resolution data
    """
    # Extract data from session state
    adversarial_results = session_state.get("adversarial_results", {})
    iterations = adversarial_results.get("iterations", [])
    
    if not iterations:
        # Return empty DataFrame if no iterations
        return pd.DataFrame()
    
    # Process issue resolution data
    data = []
    for iteration in iterations:
        iter_idx = iteration.get("iteration", 0)
        approval_check = iteration.get("approval_check", {})
        approval_rate = approval_check.get("approval_rate", 0)
        
        # Count issues and resolutions
        total_issues = 0
        resolved_issues = 0
        critiques = iteration.get("critiques", [])
        patches = iteration.get("patches", [])
        
        # Count total issues
        for critique in critiques:
            critique_json = critique.get("critique_json", {})
            if critique_json:
                issues = critique_json.get("issues", [])
                total_issues += len(issues)
        
        # Count resolved issues
        for patch in patches:
            patch_json = patch.get("patch_json", {})
            if patch_json:
                mitigation_matrix = patch_json.get("mitigation_matrix", [])
                resolved_issues += len([m for m in mitigation_matrix if m.get("status", "").lower() in ["resolved", "mitigated"]])
        
        data.append({
            "iteration": iter_idx,
            "total_issues": total_issues,
            "resolved_issues": resolved_issues,
            "unresolved_issues": max(0, total_issues - resolved_issues),
            "resolution_rate": (resolved_issues / max(1, total_issues)) * 100 if total_issues > 0 else 0,
            "approval_rate": approval_rate,
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_compliance_analysis_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get compliance analysis data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with compliance analysis data
    """
    # Extract data from session state
    compliance_requirements = session_state.get("compliance_requirements", "")
    adversarial_results = session_state.get("adversarial_results", {})
    
    # Process compliance data
    data = []
    
    # Initial compliance check
    if compliance_requirements:
        data.append({
            "stage": "initial",
            "compliance_checks": len(compliance_requirements.split(",")),
            "compliance_met": 0,  # Will be updated during adversarial testing
            "compliance_score": 0,  # Will be updated during adversarial testing
            "timestamp": time.time()
        })
    
    # Final compliance check
    if adversarial_results and compliance_requirements:
        final_compliance_check = adversarial_results.get("final_compliance_check", {})
        compliance_met = final_compliance_check.get("compliance_met", 0)
        total_checks = final_compliance_check.get("total_checks", 1)
        compliance_score = (compliance_met / max(1, total_checks)) * 100
        
        data.append({
            "stage": "final",
            "compliance_checks": total_checks,
            "compliance_met": compliance_met,
            "compliance_score": compliance_score,
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)


def get_performance_trends_data(session_state: Dict[str, Any]) -> pd.DataFrame:
    """Get performance trends data for visualization.
    
    Args:
        session_state (Dict[str, Any]): Streamlit session state
        
    Returns:
        pd.DataFrame: DataFrame with performance trends data
    """
    # Extract data from session state
    evolution_history = session_state.get("evolution_history", [])
    adversarial_results = session_state.get("adversarial_results", {})
    
    # Process performance trends data
    data = []
    
    # Evolution trends
    for generation in evolution_history:
        gen_idx = generation.get("generation", 0)
        population = generation.get("population", [])
        if population:
            fitness_values = [ind.get("fitness", 0) for ind in population]
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_fitness = max(fitness_values)
            worst_fitness = min(fitness_values)
            
            data.append({
                "phase": "evolution",
                "iteration": gen_idx,
                "avg_fitness": avg_fitness,
                "best_fitness": best_fitness,
                "worst_fitness": worst_fitness,
                "timestamp": time.time()
            })
    
    # Adversarial trends
    adversarial_iterations = adversarial_results.get("iterations", [])
    for iteration in adversarial_iterations:
        iter_idx = iteration.get("iteration", 0)
        approval_check = iteration.get("approval_check", {})
        approval_rate = approval_check.get("approval_rate", 0)
        
        data.append({
            "phase": "adversarial",
            "iteration": iter_idx,
            "approval_rate": approval_rate,
            "timestamp": time.time()
        })
    
    return pd.DataFrame(data)