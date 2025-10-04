"""
Evaluator Team Configuration and Utilities
This module provides configuration and utility functions for the evaluator team functionality.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Union
import json


# Evaluator team presets for different evaluation scenarios
EVALUATOR_PRESETS = {
    "Quality Assurance": {
        "name": "✅ Quality Assurance",
        "description": "Focus on content quality, accuracy, and completeness.",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro"
        ],
        "threshold": 90.0,
        "consecutive_rounds": 1,
        "sample_size": 3,
        "system_prompt": """You are a quality assurance expert evaluating content for accuracy, completeness, and clarity.
Your role is to assess the content based on these criteria:
1. Accuracy: Is the information factually correct?
2. Completeness: Are all necessary topics covered?
3. Clarity: Is the content easy to understand?
4. Consistency: Is the content internally consistent?
5. Professionalism: Is the tone and style appropriate?

Provide a detailed score from 0-100 and specific feedback for improvement.""",
        "weight_factors": {
            "accuracy": 0.3,
            "completeness": 0.25,
            "clarity": 0.2,
            "consistency": 0.15,
            "professionalism": 0.1
        }
    },
    "Security Review": {
        "name": "🔐 Security Review",
        "description": "Evaluate content for security vulnerabilities and best practices.",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "meta-llama/llama-3-70b-instruct"
        ],
        "threshold": 95.0,
        "consecutive_rounds": 2,
        "sample_size": 3,
        "system_prompt": """You are a security expert reviewing content for potential security vulnerabilities.
Focus on identifying:
1. Security vulnerabilities or risks
2. Compliance with security best practices
3. Potential for misuse or abuse
4. Data privacy and protection concerns
5. Authentication and authorization issues

Provide a detailed security assessment with a score from 0-100 and specific recommendations for mitigation.""",
        "weight_factors": {
            "vulnerabilities": 0.4,
            "compliance": 0.3,
            "misuse_potential": 0.15,
            "privacy": 0.1,
            "auth_issues": 0.05
        }
    },
    "Legal Compliance": {
        "name": "⚖️ Legal Compliance",
        "description": "Ensure content meets legal and regulatory requirements.",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro"
        ],
        "threshold": 98.0,
        "consecutive_rounds": 2,
        "sample_size": 3,
        "system_prompt": """You are a legal expert reviewing content for compliance with applicable laws and regulations.
Focus on:
1. Regulatory compliance (GDPR, CCPA, HIPAA, etc.)
2. Contractual obligations
3. Intellectual property considerations
4. Liability and risk management
5. Industry-specific legal requirements

Provide a comprehensive legal assessment with a score from 0-100 and specific compliance recommendations.""",
        "weight_factors": {
            "regulatory_compliance": 0.4,
            "contractual_obligations": 0.25,
            "ip_considerations": 0.15,
            "liability_management": 0.1,
            "industry_requirements": 0.1
        }
    },
    "Technical Review": {
        "name": "💻 Technical Review",
        "description": "Assess technical accuracy and implementation feasibility.",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "codellama/codellama-70b-instruct"
        ],
        "threshold": 92.0,
        "consecutive_rounds": 1,
        "sample_size": 3,
        "system_prompt": """You are a technical expert reviewing content for technical accuracy and implementation feasibility.
Evaluate:
1. Technical accuracy and correctness
2. Implementation feasibility
3. Performance considerations
4. Scalability and reliability
5. Integration requirements

Provide a detailed technical assessment with a score from 0-100 and specific technical recommendations.""",
        "weight_factors": {
            "technical_accuracy": 0.35,
            "implementation_feasibility": 0.25,
            "performance": 0.2,
            "scalability": 0.1,
            "integration": 0.1
        }
    },
    "User Experience": {
        "name": "😊 User Experience",
        "description": "Evaluate content from a user experience perspective.",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro"
        ],
        "threshold": 88.0,
        "consecutive_rounds": 1,
        "sample_size": 3,
        "system_prompt": """You are a user experience expert evaluating content from the end-user perspective.
Focus on:
1. Usability and accessibility
2. Clarity and ease of understanding
3. User engagement and satisfaction
4. Visual design and layout (if applicable)
5. User journey and flow

Provide a comprehensive UX assessment with a score from 0-100 and specific UX recommendations.""",
        "weight_factors": {
            "usability": 0.3,
            "clarity": 0.25,
            "engagement": 0.2,
            "design": 0.15,
            "flow": 0.1
        }
    }
}


def get_evaluator_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get available evaluator team presets.
    
    Returns:
        Dictionary of evaluator presets
    """
    return EVALUATOR_PRESETS


def get_default_evaluator_config() -> Dict[str, Any]:
    """
    Get the default evaluator team configuration.
    
    Returns:
        Dictionary with default evaluator configuration
    """
    return {
        "models": ["openai/gpt-4o"],
        "threshold": 90.0,
        "consecutive_rounds": 1,
        "sample_size": 1,
        "system_prompt": "You are an expert evaluator assessing the quality of content.",
        "weight_factors": {
            "overall_quality": 1.0
        }
    }


def validate_evaluator_config(config: Dict[str, Any]) -> bool:
    """
    Validate an evaluator team configuration.
    
    Args:
        config: Evaluator configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["models", "threshold", "consecutive_rounds", "sample_size", "system_prompt"]
    
    for field in required_fields:
        if field not in config:
            st.error(f"Missing required field in evaluator config: {field}")
            return False
    
    if not isinstance(config["models"], list) or len(config["models"]) == 0:
        st.error("Evaluator config must have at least one model")
        return False
    
    if not isinstance(config["threshold"], (int, float)) or not (0 <= config["threshold"] <= 100):
        st.error("Threshold must be a number between 0 and 100")
        return False
    
    if not isinstance(config["consecutive_rounds"], int) or config["consecutive_rounds"] < 1:
        st.error("Consecutive rounds must be a positive integer")
        return False
    
    if not isinstance(config["sample_size"], int) or config["sample_size"] < 1:
        st.error("Sample size must be a positive integer")
        return False
    
    if not isinstance(config["system_prompt"], str) or len(config["system_prompt"].strip()) == 0:
        st.error("System prompt must be a non-empty string")
        return False
    
    return True


def load_evaluator_config(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Load an evaluator configuration by name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        Evaluator configuration or None if not found
    """
    if config_name in EVALUATOR_PRESETS:
        return EVALUATOR_PRESETS[config_name].copy()
    
    # Check if it's a custom configuration stored in session state
    if "custom_evaluator_configs" in st.session_state:
        if config_name in st.session_state.custom_evaluator_configs:
            return st.session_state.custom_evaluator_configs[config_name].copy()
    
    return None


def save_custom_evaluator_config(config_name: str, config: Dict[str, Any]) -> bool:
    """
    Save a custom evaluator configuration.
    
    Args:
        config_name: Name for the configuration
        config: Configuration to save
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if not validate_evaluator_config(config):
            return False
            
        if "custom_evaluator_configs" not in st.session_state:
            st.session_state.custom_evaluator_configs = {}
            
        st.session_state.custom_evaluator_configs[config_name] = config
        return True
    except Exception as e:
        st.error(f"Error saving custom evaluator config: {e}")
        return False


def delete_custom_evaluator_config(config_name: str) -> bool:
    """
    Delete a custom evaluator configuration.
    
    Args:
        config_name: Name of the configuration to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        if "custom_evaluator_configs" in st.session_state:
            if config_name in st.session_state.custom_evaluator_configs:
                del st.session_state.custom_evaluator_configs[config_name]
                return True
        return False
    except Exception as e:
        st.error(f"Error deleting custom evaluator config: {e}")
        return False


def list_custom_evaluator_configs() -> List[str]:
    """
    List all custom evaluator configurations.
    
    Returns:
        List of custom evaluator configuration names
    """
    if "custom_evaluator_configs" in st.session_state:
        return list(st.session_state.custom_evaluator_configs.keys())
    return []


def calculate_weighted_score(scores: Dict[str, float], weight_factors: Dict[str, float]) -> float:
    """
    Calculate a weighted score based on individual scores and weight factors.
    
    Args:
        scores: Dictionary of individual scores
        weight_factors: Dictionary of weight factors for each score
        
    Returns:
        Weighted average score
    """
    total_weight = sum(weight_factors.values())
    if total_weight == 0:
        return sum(scores.values()) / len(scores) if scores else 0.0
    
    weighted_sum = 0.0
    for factor, weight in weight_factors.items():
        if factor in scores:
            weighted_sum += scores[factor] * weight
    
    return weighted_sum / total_weight


def format_evaluator_results(results: Dict[str, Any]) -> str:
    """
    Format evaluator results for display.
    
    Args:
        results: Evaluator results to format
        
    Returns:
        Formatted results as a string
    """
    try:
        formatted = f"## Evaluator Results\n\n"
        formatted += f"**Score:** {results.get('score', 0.0):.1f}%\n"
        formatted += f"**Threshold Met:** {'✅ Yes' if results.get('threshold_met', False) else '❌ No'}\n"
        formatted += f"**Consecutive Rounds:** {results.get('consecutive_rounds', 0)}/{results.get('required_consecutive_rounds', 1)}\n\n"
        
        if "feedback" in results:
            formatted += "### Feedback\n"
            for item in results["feedback"]:
                formatted += f"- {item}\n"
        
        if "detailed_scores" in results:
            formatted += "\n### Detailed Scores\n"
            for category, score in results["detailed_scores"].items():
                formatted += f"- {category.capitalize()}: {score:.1f}%\n"
        
        return formatted
    except Exception as e:
        return f"Error formatting evaluator results: {e}"