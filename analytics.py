"""
Analytics Module for OpenEvolve - Data analysis and insights generation
This module provides functions for analyzing content quality, evolution performance, and generating insights
"""

import re
import time
import numpy as np
from typing import Dict, Any, List
import streamlit as st



def analyze_plan_quality(plan_text: str) -> Dict[str, Any]:
    """Analyze plan quality metrics.

    Args:
        plan_text (str): The plan to analyze

    Returns:
        Dict[str, Any]: Plan quality metrics
    """
    if not plan_text:
        return {
            "sections": 0,
            "objectives": 0,
            "milestones": 0,
            "resources": 0,
            "risks": 0,
            "dependencies": 0,
            "timeline_elements": 0,
            "quality_score": 0,
        }

    # Count sections (headers)
    sections = len(
        re.findall(r"^#{1,6}\\s+|.*\\n[=]{3,}|.*\\n[-]{3,}", plan_text, re.MULTILINE)
    )

    # Count objectives (look for objective-related terms)
    objective_patterns = [r"\\bobjectives?\\b", r"\\bgoals?\\b", r"\\bpurpose\\b", r"\\baim\\b"]
    objectives = 0
    for pattern in objective_patterns:
        objectives += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count milestones (look for milestone-related terms)
    milestone_patterns = [
        r"\\bmilestones?\\b",
        r"\\bdeadlines?\\b",
        r"\\btimelines?\\b",
        r"\\bschedule\\b",
    ]
    milestones = 0
    for pattern in milestone_patterns:
        milestones += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count resources (look for resource-related terms)
    resource_patterns = [
        r"\\bresources?\\b",
        r"\\bbudget\\b",
        r"\\bcosts?\\b",
        r"\\bmaterials?\\b",
    ]
    resources = 0
    for pattern in resource_patterns:
        resources += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count risks (look for risk-related terms)
    risk_patterns = [
        r"\\brisks?\\b",
        r"\\bthreats?\\b",
        r"\\bvulnerabilit(?:y|ies)\\b",
        r"\\bhazards?\\b",
    ]
    risks = 0
    for pattern in risk_patterns:
        risks += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count dependencies (look for dependency-related terms)
    dependency_patterns = [
        r"\\bdependenc(?:y|ies)\\b",
        r"\\bprerequisites?\\b",
        r"\\brequires?\\b",
        r"\\bneeds?\\b",
    ]
    dependencies = 0
    for pattern in dependency_patterns:
        dependencies += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count timeline elements (dates, time-related terms)
    timeline_patterns = [
        r"\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}",
        r"\\d{4}[/-]\\d{1,2}[/-]\\d{1,2}",
        r"\\bweeks?\\b",
        r"\\bmonths?\\b",
        r"\\byears?\\b",
        r"\\bdays?\\b",
    ]
    timeline_elements = 0
    for pattern in timeline_patterns:
        timeline_elements += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Calculate quality score (simplified)
    quality_score = 50  # Start with baseline

    # Add points for completeness
    quality_score += min(20, sections * 2)  # Up to 20 points for sections
    quality_score += min(15, objectives * 3)  # Up to 15 points for objectives
    quality_score += min(10, milestones * 2)  # Up to 10 points for milestones
    quality_score += min(10, resources * 2)  # Up to 10 points for resources
    quality_score += min(10, risks * 2)  # Up to 10 points for risks
    quality_score += min(10, dependencies * 2)  # Up to 10 points for dependencies
    quality_score += min(10, timeline_elements * 2)  # Up to 10 points for timeline

    quality_score = max(0, min(100, quality_score))  # Clamp to 0-100

    return {
        "sections": sections,
        "objectives": objectives,
        "milestones": milestones,
        "resources": resources,
        "risks": risks,
        "dependencies": dependencies,
        "timeline_elements": timeline_elements,
        "quality_score": round(quality_score, 2),
    }


def analyze_code_quality(code_text: str) -> Dict[str, Any]:
    """Analyze code quality metrics.

    Args:
        code_text (str): The code to analyze

    Returns:
        Dict[str, Any]: Code quality metrics
    """
    if not code_text:
        return {
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0,
            "comments": 0,
            "comment_ratio": 0.0,
            "complexity": 0,
            "readability_score": 0,
        }

    # Count lines of code (non-empty lines)
    lines = [line for line in code_text.split('\\n') if line.strip()]
    lines_of_code = len(lines)

    # Count functions (language agnostic)
    function_patterns = [
        r"\\bdef\\s+\\w+",  # Python
        r"\\bfunction\\s+\\w+",  # JavaScript
        r"\\b\\w+\\s*\\([^)]*\\)\\s*{",  # Java/C-like
    ]
    functions = 0
    for pattern in function_patterns:
        functions += len(re.findall(pattern, code_text))

    # Count classes
    class_patterns = [
        r"\\bclass\\s+\\w+",  # Python/Java/C++
        r"\\bstruct\\s+\\w+",  # C/C++
    ]
    classes = 0
    for pattern in class_patterns:
        classes += len(re.findall(pattern, code_text))

    # Count comments
    comment_patterns = [
        r"//.*",  # Single-line comments (C++, Java, JavaScript)
        r"#.*",  # Python comments
        r"/\\*.*?\\*/",  # Multi-line comments (C-style)
        r"<!--.*?-->",  # HTML comments
    ]
    comments = 0
    for pattern in comment_patterns:
        comments += len(re.findall(pattern, code_text, re.DOTALL))

    # Calculate comment ratio
    comment_ratio = comments / max(1, lines_of_code)

    # Estimate complexity (based on nesting, conditionals, loops)
    complexity_patterns = [
        r"\\b(if|elif|else)\\b",
        r"\\b(for|while)\\b",
        r"\\bswitch\\b",
        r"\\btry\\b",
        r"\\bcatch\\b",
    ]
    complexity = 0
    for pattern in complexity_patterns:
        complexity += len(re.findall(pattern, code_text))

    # Calculate readability score (simplified)
    avg_line_length = np.mean([len(line) for line in lines]) if lines else 0
    readability_score = max(0, 100 - (avg_line_length / 2))  # Higher avg length = lower readability

    return {
        "lines_of_code": lines_of_code,
        "functions": functions,
        "classes": classes,
        "comments": comments,
        "comment_ratio": round(comment_ratio, 2),
        "complexity": complexity,
        "readability_score": round(readability_score, 1),
    }


def analyze_document_quality(document_text: str) -> Dict[str, Any]:
    """Analyze document quality metrics.

    Args:
        document_text (str): The document to analyze

    Returns:
        Dict[str, Any]: Document quality metrics
    """
    if not document_text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_sentence_length": 0,
            "avg_paragraph_length": 0,
            "readability_score": 0,
            "tone_consistency": 0,
        }

    # Word count
    words = document_text.split()
    word_count = len(words)

    # Sentence count
    sentences = re.split(r'[.!?]+', document_text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Paragraph count
    paragraphs = re.split(r'\\n\\s*\\n', document_text)
    paragraphs = [p for p in paragraphs if p.strip()]
    paragraph_count = len(paragraphs)

    # Average sentence length
    avg_sentence_length = word_count / max(1, sentence_count)

    # Average paragraph length
    avg_paragraph_length = sentence_count / max(1, paragraph_count)

    # Simplified readability score (Flesch-like approximation)
    syllables = sum(word.count(vowel) for word in words for vowel in 'aeiouAEIOU')
    readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllables / max(1, word_count)))

    # Tone consistency (simplified)
    # Count positive/negative words for consistency
    positive_words = ['good', 'excellent', 'great', 'amazing', 'fantastic', 'wonderful', 'outstanding']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst']
    
    pos_count = sum(document_text.lower().count(word) for word in positive_words)
    neg_count = sum(document_text.lower().count(word) for word in negative_words)
    
    # Tone consistency is higher when one sentiment dominates
    total_sentiment = pos_count + neg_count
    if total_sentiment > 0:
        tone_consistency = max(pos_count, neg_count) / total_sentiment * 100
    else:
        tone_consistency = 50  # Neutral

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "avg_paragraph_length": round(avg_paragraph_length, 1),
        "readability_score": round(max(0, min(100, readability_score)), 1),
        "tone_consistency": round(tone_consistency, 1),
    }


def analyze_evolution_performance(evolution_history: List[Dict]) -> Dict[str, Any]:
    """Analyze evolution performance metrics.

    Args:
        evolution_history (List[Dict]): History of evolution iterations

    Returns:
        Dict[str, Any]: Evolution performance metrics
    """
    if not evolution_history:
        return {
            "total_generations": 0,
            "best_fitness": 0,
            "avg_improvement_rate": 0,
            "convergence_speed": 0,
            "diversity_score": 0,
        }

    total_generations = len(evolution_history)
    
    # Extract fitness scores
    fitness_scores = []
    for generation in evolution_history:
        if 'population' in generation:
            for individual in generation['population']:
                if 'fitness' in individual:
                    fitness_scores.append(individual['fitness'])
    
    best_fitness = max(fitness_scores) if fitness_scores else 0
    
    # Calculate average improvement rate
    if len(fitness_scores) > 1:
        improvements = [fitness_scores[i] - fitness_scores[i-1] for i in range(1, len(fitness_scores))]
        avg_improvement_rate = sum(improvements) / len(improvements)
    else:
        avg_improvement_rate = 0
    
    # Convergence speed (how quickly it reaches 90% of best fitness)
    if best_fitness > 0 and fitness_scores:
        target_fitness = best_fitness * 0.9
        convergence_generation = None
        for i, score in enumerate(fitness_scores):
            if score >= target_fitness:
                convergence_generation = i
                break
        convergence_speed = convergence_generation / total_generations if convergence_generation is not None else 1.0
    else:
        convergence_speed = 0
    
    # Diversity score (based on variance of fitness scores in recent generations)
    if len(fitness_scores) >= 5:
        recent_scores = fitness_scores[-5:]
        diversity_score = np.std(recent_scores) if recent_scores else 0
    else:
        diversity_score = 0

    return {
        "total_generations": total_generations,
        "best_fitness": round(best_fitness, 4),
        "avg_improvement_rate": round(avg_improvement_rate, 4),
        "convergence_speed": round(convergence_speed, 4),
        "diversity_score": round(diversity_score, 4),
    }


def analyze_adversarial_performance(adversarial_results: Dict) -> Dict[str, Any]:
    """Analyze adversarial testing performance metrics.

    Args:
        adversarial_results (Dict): Results from adversarial testing

    Returns:
        Dict[str, Any]: Adversarial performance metrics
    """
    if not adversarial_results:
        return {
            "total_iterations": 0,
            "final_approval_rate": 0,
            "total_issues_found": 0,
            "issues_fixed": 0,
            "avg_issue_severity": 0,
        }

    iterations = adversarial_results.get('iterations', [])
    total_iterations = len(iterations)
    
    final_approval_rate = adversarial_results.get('final_approval_rate', 0)
    
    # Count issues
    total_issues_found = 0
    issues_fixed = 0
    severity_scores = []
    
    for iteration in iterations:
        critiques = iteration.get('critiques', [])
        for critique in critiques:
            critique_json = critique.get('critique_json', {})
            if critique_json and isinstance(critique_json, dict):
                issues = critique_json.get('issues', [])
                if isinstance(issues, list):
                    total_issues_found += len(issues)
                    for issue in issues:
                        if isinstance(issue, dict):
                            severity = issue.get('severity', 'low').lower()
                            severity_map = {'low': 1, 'medium': 3, 'high': 6, 'critical': 12}
                            severity_scores.append(severity_map.get(severity, 1))
                            # Assume issues are fixed if we're past the first iteration
                            if iteration != iterations[0]:
                                issues_fixed += 1

    avg_issue_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

    return {
        "total_iterations": total_iterations,
        "final_approval_rate": round(final_approval_rate, 1),
        "total_issues_found": total_issues_found,
        "issues_fixed": issues_fixed,
        "avg_issue_severity": round(avg_issue_severity, 2),
    }


def analyze_model_performance(model_performance: Dict) -> Dict[str, Any]:
    """Analyze model performance metrics.

    Args:
        model_performance (Dict): Model performance data

    Returns:
        Dict[str, Any]: Model performance metrics
    """
    if not model_performance:
        return {
            "total_models": 0,
            "avg_model_score": 0,
            "best_model": "",
            "best_model_score": 0,
        }

    total_models = len(model_performance)
    
    scores = []
    best_model = ""
    best_model_score = 0
    
    for model_id, perf_data in model_performance.items():
        score = perf_data.get('score', 0)
        scores.append(score)
        
        if score > best_model_score:
            best_model_score = score
            best_model = model_id

    avg_model_score = sum(scores) / len(scores) if scores else 0

    return {
        "total_models": total_models,
        "avg_model_score": round(avg_model_score, 2),
        "best_model": best_model,
        "best_model_score": round(best_model_score, 2),
    }


def generate_comprehensive_report(
    protocol_text: str,
    evolution_history: List[Dict],
    adversarial_results: Dict,
    model_performance: Dict,
    content_type: str = "general"
) -> Dict[str, Any]:
    """Generate a comprehensive analytics report.

    Args:
        protocol_text (str): The content being analyzed
        evolution_history (List[Dict]): Evolution history
        adversarial_results (Dict): Adversarial testing results
        model_performance (Dict): Model performance data
        content_type (str): Type of content

    Returns:
        Dict[str, Any]: Comprehensive analytics report
    """
    # Analyze content based on type
    if content_type.startswith("code_"):
        content_metrics = analyze_code_quality(protocol_text)
    elif content_type in ["legal", "medical", "technical"]:
        content_metrics = analyze_document_quality(protocol_text)
    else:
        content_metrics = analyze_plan_quality(protocol_text)
    
    # Analyze performance metrics
    evolution_metrics = analyze_evolution_performance(evolution_history)
    adversarial_metrics = analyze_adversarial_performance(adversarial_results)
    model_metrics = analyze_model_performance(model_performance)
    
    # Generate overall score
    overall_score = (
        content_metrics.get('quality_score', 0) * 0.3 +
        evolution_metrics.get('best_fitness', 0) * 100 * 0.3 +  # Scale fitness to 0-100
        adversarial_metrics.get('final_approval_rate', 0) * 0.2 +
        model_metrics.get('avg_model_score', 0) * 0.2
    )
    
    return {
        "timestamp": time.time(),
        "content_type": content_type,
        "content_metrics": content_metrics,
        "evolution_metrics": evolution_metrics,
        "adversarial_metrics": adversarial_metrics,
        "model_metrics": model_metrics,
        "overall_score": round(overall_score, 2),
        "recommendations": generate_recommendations(
            content_metrics, evolution_metrics, adversarial_metrics, model_metrics
        )
    }


def generate_recommendations(
    content_metrics: Dict,
    evolution_metrics: Dict,
    adversarial_metrics: Dict,
    model_metrics: Dict
) -> List[str]:
    """Generate recommendations based on analytics.

    Args:
        content_metrics (Dict): Content quality metrics
        evolution_metrics (Dict): Evolution performance metrics
        adversarial_metrics (Dict): Adversarial testing metrics
        model_metrics (Dict): Model performance metrics

    Returns:
        List[str]: List of recommendations
    """
    recommendations = []
    
    # Content quality recommendations
    if content_metrics.get('quality_score', 0) < 50:
        recommendations.append("Content quality is below average. Consider restructuring and adding more detail.")
    elif content_metrics.get('quality_score', 0) < 75:
        recommendations.append("Content quality is good but can be improved with more structure and detail.")
    
    # Evolution recommendations
    if evolution_metrics.get('total_generations', 0) < 10:
        recommendations.append("Consider increasing evolution iterations for better optimization.")
    
    if evolution_metrics.get('avg_improvement_rate', 0) < 0.01:
        recommendations.append("Evolution improvement rate is low. Try adjusting parameters or using different models.")
    
    # Adversarial testing recommendations
    if adversarial_metrics.get('final_approval_rate', 0) < 70:
        recommendations.append("Approval rate is low. Consider more adversarial iterations or different models.")
    
    if adversarial_metrics.get('avg_issue_severity', 0) > 2.5:
        recommendations.append("High severity issues detected. Address critical issues before deployment.")
    
    # Model recommendations
    if model_metrics.get('total_models', 0) < 3:
        recommendations.append("Using more diverse models could improve results.")
    
    if model_metrics.get('avg_model_score', 0) < 50:
        recommendations.append("Model performance is below average. Consider using higher quality models.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Content quality is high. Continue with current approach.")
        recommendations.append("Evolution and adversarial testing are performing well.")
        recommendations.append("Model performance is satisfactory.")
    
    return recommendations


def generate_performance_report(
    evolution_history: List[Dict],
    adversarial_results: Dict,
    model_performance: Dict
) -> Dict[str, Any]:
    """Generate a performance-focused report.

    Args:
        evolution_history (List[Dict]): Evolution history
        adversarial_results (Dict): Adversarial testing results
        model_performance (Dict): Model performance data

    Returns:
        Dict[str, Any]: Performance-focused report
    """
    # Analyze performance metrics
    evolution_metrics = analyze_evolution_performance(evolution_history)
    adversarial_metrics = analyze_adversarial_performance(adversarial_results)
    model_metrics = analyze_model_performance(model_performance)
    
    # Calculate performance score
    performance_score = (
        evolution_metrics.get('best_fitness', 0) * 100 * 0.4 +  # Scale fitness to 0-100
        adversarial_metrics.get('final_approval_rate', 0) * 0.3 +
        model_metrics.get('avg_model_score', 0) * 0.3
    )
    
    return {
        "timestamp": time.time(),
        "performance_score": round(performance_score, 2),
        "evolution_metrics": evolution_metrics,
        "adversarial_metrics": adversarial_metrics,
        "model_metrics": model_metrics,
        "performance_recommendations": generate_performance_recommendations(
            evolution_metrics, adversarial_metrics, model_metrics
        )
    }


def generate_performance_recommendations(
    evolution_metrics: Dict,
    adversarial_metrics: Dict,
    model_metrics: Dict
) -> List[str]:
    """Generate performance-focused recommendations.

    Args:
        evolution_metrics (Dict): Evolution performance metrics
        adversarial_metrics (Dict): Adversarial testing metrics
        model_metrics (Dict): Model performance metrics

    Returns:
        List[str]: List of performance recommendations
    """
    recommendations = []
    
    # Evolution performance recommendations
    if evolution_metrics.get('total_generations', 0) < 10:
        recommendations.append("Increase evolution iterations for better optimization.")
    
    if evolution_metrics.get('avg_improvement_rate', 0) < 0.01:
        recommendations.append("Adjust evolution parameters or use more diverse models to improve convergence.")
    
    if evolution_metrics.get('diversity_score', 0) < 0.1:
        recommendations.append("Increase population diversity by using more varied models or adjusting parameters.")
    
    # Adversarial testing performance recommendations
    if adversarial_metrics.get('final_approval_rate', 0) < 70:
        recommendations.append("Improve adversarial testing by using better models or adjusting iteration parameters.")
    
    if adversarial_metrics.get('avg_issue_severity', 0) > 2.5:
        recommendations.append("Focus on resolving high-severity issues to improve overall performance.")
    
    # Model performance recommendations
    if model_metrics.get('total_models', 0) < 3:
        recommendations.append("Use more diverse models to improve overall performance.")
    
    if model_metrics.get('avg_model_score', 0) < 50:
        recommendations.append("Upgrade to higher-quality models to improve performance.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Performance is excellent. Continue with current approach.")
        recommendations.append("Evolution and adversarial testing are performing optimally.")
        recommendations.append("Model performance is outstanding.")
    
    return recommendations


def generate_quality_report(
    protocol_text: str,
    content_type: str,
    evolution_history: List[Dict],
    adversarial_results: Dict
) -> Dict[str, Any]:
    """Generate a quality-focused report.

    Args:
        protocol_text (str): The content being analyzed
        content_type (str): Type of content
        evolution_history (List[Dict]): Evolution history
        adversarial_results (Dict): Adversarial testing results

    Returns:
        Dict[str, Any]: Quality-focused report
    """
    # Analyze content based on type
    if content_type.startswith("code_"):
        content_metrics = analyze_code_quality(protocol_text)
    elif content_type in ["legal", "medical", "technical"]:
        content_metrics = analyze_document_quality(protocol_text)
    else:
        content_metrics = analyze_plan_quality(protocol_text)
    
    # Analyze performance metrics
    evolution_metrics = analyze_evolution_performance(evolution_history)
    adversarial_metrics = analyze_adversarial_performance(adversarial_results)
    
    # Calculate quality score
    quality_score = (
        content_metrics.get('quality_score', 0) * 0.4 +
        evolution_metrics.get('best_fitness', 0) * 100 * 0.3 +  # Scale fitness to 0-100
        adversarial_metrics.get('final_approval_rate', 0) * 0.3
    )
    
    return {
        "timestamp": time.time(),
        "quality_score": round(quality_score, 2),
        "content_metrics": content_metrics,
        "evolution_metrics": evolution_metrics,
        "adversarial_metrics": adversarial_metrics,
        "quality_recommendations": generate_quality_recommendations(
            content_metrics, evolution_metrics, adversarial_metrics
        )
    }


def generate_quality_recommendations(
    content_metrics: Dict,
    evolution_metrics: Dict,
    adversarial_metrics: Dict
) -> List[str]:
    """Generate quality-focused recommendations.

    Args:
        content_metrics (Dict): Content quality metrics
        evolution_metrics (Dict): Evolution performance metrics
        adversarial_metrics (Dict): Adversarial testing metrics

    Returns:
        List[str]: List of quality recommendations
    """
    recommendations = []
    
    # Content quality recommendations
    if content_metrics.get('quality_score', 0) < 50:
        recommendations.append("Significantly improve content structure and completeness.")
    elif content_metrics.get('quality_score', 0) < 75:
        recommendations.append("Enhance content quality with more detail and better organization.")
    
    # Evolution quality recommendations
    if evolution_metrics.get('total_generations', 0) < 10:
        recommendations.append("Run more evolution iterations to improve content quality.")
    
    if evolution_metrics.get('avg_improvement_rate', 0) < 0.01:
        recommendations.append("Adjust evolution parameters to improve content quality improvement rate.")
    
    # Adversarial testing quality recommendations
    if adversarial_metrics.get('final_approval_rate', 0) < 70:
        recommendations.append("Improve adversarial testing to achieve higher content approval rates.")
    
    if adversarial_metrics.get('avg_issue_severity', 0) > 2.5:
        recommendations.append("Address high-severity issues to improve overall content quality.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Content quality is excellent. Continue with current approach.")
        recommendations.append("Evolution and adversarial testing are producing high-quality content.")
        recommendations.append("No significant quality improvements needed at this time.")
    
    return recommendations


def generate_efficiency_report(
    evolution_history: List[Dict],
    adversarial_results: Dict,
    model_performance: Dict
) -> Dict[str, Any]:
    """Generate an efficiency-focused report.

    Args:
        evolution_history (List[Dict]): Evolution history
        adversarial_results (Dict): Adversarial testing results
        model_performance (Dict): Model performance data

    Returns:
        Dict[str, Any]: Efficiency-focused report
    """
    # Analyze performance metrics
    evolution_metrics = analyze_evolution_performance(evolution_history)
    adversarial_metrics = analyze_adversarial_performance(adversarial_results)
    model_metrics = analyze_model_performance(model_performance)
    
    # Calculate efficiency score
    efficiency_score = (
        evolution_metrics.get('convergence_speed', 0) * 100 * 0.4 +  # Scale convergence to 0-100
        adversarial_metrics.get('final_approval_rate', 0) * 0.3 +
        model_metrics.get('avg_model_score', 0) * 0.3
    )
    
    return {
        "timestamp": time.time(),
        "efficiency_score": round(efficiency_score, 2),
        "evolution_metrics": evolution_metrics,
        "adversarial_metrics": adversarial_metrics,
        "model_metrics": model_metrics,
        "efficiency_recommendations": generate_efficiency_recommendations(
            evolution_metrics, adversarial_metrics, model_metrics
        )
    }


def generate_efficiency_recommendations(
    evolution_metrics: Dict,
    adversarial_metrics: Dict,
    model_metrics: Dict
) -> List[str]:
    """Generate efficiency-focused recommendations.

    Args:
        evolution_metrics (Dict): Evolution performance metrics
        adversarial_metrics (Dict): Adversarial testing metrics
        model_metrics (Dict): Model performance metrics

    Returns:
        List[str]: List of efficiency recommendations
    """
    recommendations = []
    
    # Evolution efficiency recommendations
    if evolution_metrics.get('convergence_speed', 0) < 0.5:
        recommendations.append("Improve evolution convergence speed by adjusting parameters or using better models.")
    
    if evolution_metrics.get('diversity_score', 0) < 0.1:
        recommendations.append("Increase population diversity to improve convergence efficiency.")
    
    # Adversarial testing efficiency recommendations
    if adversarial_metrics.get('final_approval_rate', 0) < 70:
        recommendations.append("Improve adversarial testing efficiency by using more effective models.")
    
    # Model efficiency recommendations
    if model_metrics.get('avg_model_score', 0) < 50:
        recommendations.append("Use higher-quality models to improve overall process efficiency.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Process efficiency is excellent. Continue with current approach.")
        recommendations.append("Evolution and adversarial testing are running efficiently.")
        recommendations.append("Model performance is optimal for efficiency.")
    
    return recommendations


def generate_compliance_report(
    protocol_text: str,
    compliance_requirements: str,
    content_type: str
) -> Dict[str, Any]:
    """Generate a compliance-focused report.

    Args:
        protocol_text (str): The content being analyzed
        compliance_requirements (str): Compliance requirements to check
        content_type (str): Type of content

    Returns:
        Dict[str, Any]: Compliance-focused report
    """
    # Analyze content based on type
    if content_type.startswith("code_"):
        content_metrics = analyze_code_quality(protocol_text)
    elif content_type in ["legal", "medical", "technical"]:
        content_metrics = analyze_document_quality(protocol_text)
    else:
        content_metrics = analyze_plan_quality(protocol_text)
    
    # Check compliance
    compliance_matches = 0
    total_requirements = 0
    
    if compliance_requirements:
        requirements = compliance_requirements.split(",")
        total_requirements = len(requirements)
        
        for requirement in requirements:
            requirement = requirement.strip().lower()
            if requirement in protocol_text.lower():
                compliance_matches += 1
    
    compliance_score = (compliance_matches / max(1, total_requirements)) * 100 if total_requirements > 0 else 100
    
    return {
        "timestamp": time.time(),
        "compliance_score": round(compliance_score, 2),
        "content_metrics": content_metrics,
        "compliance_matches": compliance_matches,
        "total_requirements": total_requirements,
        "compliance_recommendations": generate_compliance_recommendations(
            content_metrics, compliance_matches, total_requirements
        )
    }


def generate_compliance_recommendations(
    content_metrics: Dict,
    compliance_matches: int,
    total_requirements: int
) -> List[str]:
    """Generate compliance-focused recommendations.

    Args:
        content_metrics (Dict): Content quality metrics
        compliance_matches (int): Number of compliance requirements met
        total_requirements (int): Total number of compliance requirements

    Returns:
        List[str]: List of compliance recommendations
    """
    recommendations = []
    
    # Compliance recommendations
    if compliance_matches < total_requirements:
        recommendations.append(f"Address {total_requirements - compliance_matches} unmet compliance requirements.")
    else:
        recommendations.append("All compliance requirements have been met.")
    
    # Content quality recommendations for compliance
    if content_metrics.get('quality_score', 0) < 50:
        recommendations.append("Improve content quality to better meet compliance requirements.")
    elif content_metrics.get('quality_score', 0) < 75:
        recommendations.append("Enhance content quality for better compliance adherence.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Compliance is excellent. Continue with current approach.")
        recommendations.append("Content meets all compliance requirements.")
        recommendations.append("No compliance improvements needed at this time.")
    
    return recommendations


def generate_security_report(
    protocol_text: str,
    content_type: str,
    adversarial_results: Dict
) -> Dict[str, Any]:
    """Generate a security-focused report.

    Args:
        protocol_text (str): The content being analyzed
        content_type (str): Type of content
        adversarial_results (Dict): Adversarial testing results

    Returns:
        Dict[str, Any]: Security-focused report
    """
    # Analyze content based on type
    if content_type.startswith("code_"):
        content_metrics = analyze_code_quality(protocol_text)
    elif content_type in ["legal", "medical", "technical"]:
        content_metrics = analyze_document_quality(protocol_text)
    else:
        content_metrics = analyze_plan_quality(protocol_text)
    
    # Analyze adversarial results for security
    adversarial_metrics = analyze_adversarial_performance(adversarial_results)
    
    # Calculate security score
    security_score = (
        content_metrics.get('quality_score', 0) * 0.3 +
        (100 - adversarial_metrics.get('avg_issue_severity', 0) * 20) * 0.4 +  # Scale severity to 0-100
        adversarial_metrics.get('final_approval_rate', 0) * 0.3
    )
    
    return {
        "timestamp": time.time(),
        "security_score": round(security_score, 2),
        "content_metrics": content_metrics,
        "adversarial_metrics": adversarial_metrics,
        "security_recommendations": generate_security_recommendations(
            content_metrics, adversarial_metrics
        )
    }


def generate_security_recommendations(
    content_metrics: Dict,
    adversarial_metrics: Dict
) -> List[str]:
    """Generate security-focused recommendations.

    Args:
        content_metrics (Dict): Content quality metrics
        adversarial_metrics (Dict): Adversarial testing metrics

    Returns:
        List[str]: List of security recommendations
    """
    recommendations = []
    
    # Content security recommendations
    if content_metrics.get('quality_score', 0) < 50:
        recommendations.append("Significantly improve content security by addressing structural weaknesses.")
    elif content_metrics.get('quality_score', 0) < 75:
        recommendations.append("Enhance content security with better organization and detail.")
    
    # Adversarial testing security recommendations
    if adversarial_metrics.get('final_approval_rate', 0) < 70:
        recommendations.append("Improve security by strengthening adversarial testing processes.")
    
    if adversarial_metrics.get('avg_issue_severity', 0) > 2.5:
        recommendations.append("Address high-severity security issues immediately.")
    
    # Generic recommendations if no specific issues
    if not recommendations:
        recommendations.append("Security posture is strong. Continue with current approach.")
        recommendations.append("Content passes security adversarial testing effectively.")
        recommendations.append("No significant security improvements needed at this time.")
    
    return recommendations


def generate_advanced_analytics(results: Dict) -> Dict:
    """Generate advanced analytics from adversarial testing results.

    Args:
        results (Dict): Adversarial testing results

    Returns:
        Dict: Advanced analytics data
    """
    analytics = {
        "total_iterations": len(results.get("iterations", [])),
        "final_approval_rate": results.get("final_approval_rate", 0),
        "total_cost_usd": results.get("cost_estimate_usd", 0),
        "total_tokens": results.get("tokens", {}).get("prompt", 0)
        + results.get("tokens", {}).get("completion", 0),
        "confidence_trend": [],
        "issue_resolution_rate": 0,
        "model_performance": {},
        "efficiency_score": 0,
        "security_strength": 0,
        "compliance_coverage": 0,
        "clarity_score": 0,
        "completeness_score": 0,
    }

    # Calculate confidence trend
    if results.get("iterations"):
        confidence_history = [
            iter.get("approval_check", {}).get("approval_rate", 0)
            for iter in results.get("iterations", [])
        ]
        analytics["confidence_trend"] = confidence_history

    # Calculate issue resolution rate
    if results.get("iterations"):
        total_issues_found = 0
        total_issues_resolved = 0
        severity_weights = {"low": 1, "medium": 3, "high": 6, "critical": 12}

        for iteration in results.get("iterations", []):
            critiques = iteration.get("critiques", [])
            for critique in critiques:
                critique_json = critique.get("critique_json", {})
                issues = critique_json.get("issues", [])
                total_issues_found += len(issues)

                # Count resolved issues with severity weighting
                patches = iteration.get("patches", [])
                resolved_weighted = 0
                total_weighted = 0

                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    weight = severity_weights.get(severity, 1)
                    total_weighted += weight

                for patch in patches:
                    patch_json = patch.get("patch_json", {})
                    mitigation_matrix = patch_json.get("mitigation_matrix", [])
                    for mitigation in mitigation_matrix:
                        if mitigation.get("issue") == issue.get("title") and str(
                            mitigation.get("status", "")
                        ).lower() in [
                            "resolved", "mitigated"
                        ]:
                            resolved_weighted += 3  # Average weight
                            break

                total_issues_resolved += min(len(issues), len(patches))

        if total_issues_found > 0:
            analytics["issue_resolution_rate"] = (
                total_issues_resolved / total_issues_found
            ) * 100

    # Calculate efficiency score
    efficiency = 100
    if analytics["total_cost_usd"] > 0:
        # Lower cost = higher efficiency
        efficiency -= min(50, analytics["total_cost_usd"] * 10)
    if analytics["total_iterations"] > 10:
        # More iterations = lower efficiency
        efficiency -= min(30, (analytics["total_iterations"] - 10) * 2)
    analytics["efficiency_score"] = max(0, efficiency)

    # Calculate security strength based on resolved critical/high issues
    if results.get("iterations"):
        critical_high_resolved = 0
        total_critical_high = 0

        for iteration in results.get("iterations", []):
            critiques = iteration.get("critiques", [])
            for critique in critiques:
                critique_json = critique.get("critique_json", {})
                issues = critique_json.get("issues", [])
                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    if severity in ["critical", "high"]:
                        total_critical_high += 1
                        # Check if this issue was resolved in any patch
                        for patch in iteration.get("patches", []):
                            patch_json = patch.get("patch_json", {})
                            mitigation_matrix = patch_json.get("mitigation_matrix", [])
                            for mitigation in mitigation_matrix:
                                if mitigation.get("issue") == issue.get(
                                    "title"
                                ) and str(mitigation.get("status", "")).lower() in [
                                    "resolved", "mitigated"
                                ]:
                                    critical_high_resolved += 1
                                    break

        if total_critical_high > 0:
            analytics["security_strength"] = (
                critical_high_resolved / total_critical_high
            ) * 100

    # Calculate compliance coverage
    if results.get("compliance_requirements"):
        # Simple check for compliance mentions in final protocol
        final_sop = results.get("final_sop", "")
        compliance_reqs = results.get("compliance_requirements", "")

        # Count how many compliance requirements are addressed
        reqs_addressed = 0
        total_reqs = 0

        for req in compliance_reqs.split(","):
            req = req.strip().lower()
            if req:
                total_reqs += 1
                if req in final_sop.lower():
                    reqs_addressed += 1

        if total_reqs > 0:
            analytics["compliance_coverage"] = (reqs_addressed / total_reqs) * 100

    # Calculate clarity score based on protocol structure
    final_sop = results.get("final_sop", "")
    if final_sop:
        # Placeholder calls for now, will be replaced with actual functions
        structure = {
            "has_headers": True,
            "has_numbered_steps": True,
            "has_preconditions": True,
            "has_postconditions": True,
            "has_error_handling": True,
            "section_count": 5,
        }
        complexity = {"word_count": 100, "unique_words": 50, "avg_sentence_length": 15}

        # Clarity score based on structure elements
        clarity_score = 0
        if structure["has_headers"]:
            clarity_score += 25
        if structure["has_numbered_steps"] or structure["has_bullet_points"]:
            clarity_score += 25
        if structure["has_preconditions"]:
            clarity_score += 15
        if structure["has_postconditions"]:
            clarity_score += 15
        if structure["has_error_handling"]:
            clarity_score += 20

        analytics["clarity_score"] = clarity_score

        # Completeness score based on structure and complexity
        completeness_score = min(
            100,
            (
                structure["section_count"] * 5  # Sections contribute to completeness
                + complexity["unique_words"]
                / max(1, complexity["word_count"])
                * 100
                * 0.3  # Vocabulary diversity
                + (1 - complexity["avg_sentence_length"] / 50)
                * 100
                * 0.7  # Sentence complexity balance
            ),
        )
        analytics["completeness_score"] = completeness_score

    return analytics




def render_analytics_settings():
    """
    Renders the analytics settings section in the Streamlit UI.
    Allows users to configure what data to collect, how to display it, etc.
    """
    st.header("📊 Analytics Settings")
    
    st.info("Configure your analytics preferences to track and visualize application usage.")
    
    # Data collection settings
    with st.expander("📈 Data Collection Settings", expanded=True):
        st.subheader("Data Collection Preferences")
        
        # Simulate data collection settings
        collect_usage_data = st.checkbox("Enable anonymous usage data collection", value=True)
        collect_performance_data = st.checkbox("Enable performance metrics collection", value=True)
        collect_error_data = st.checkbox("Enable error reporting", value=True)
        
        if st.button("Save Analytics Settings"):
            # In a real implementation, this would save to user preferences
            st.session_state.analytics_settings = {
                "collect_usage_data": collect_usage_data,
                "collect_performance_data": collect_performance_data,
                "collect_error_data": collect_error_data
            }
            st.success("Analytics settings saved successfully!")
    
    # Report format settings
    with st.expander("📄 Report Preferences", expanded=True):
        st.subheader("Default Report Format")
        report_format = st.selectbox("Default report format", ["Markdown", "JSON", "PDF", "CSV", "Excel"])
        
        if st.button("Set Default Format"):
            st.session_state.default_report_format = report_format
            st.success(f"Default report format set to {report_format}")
    
    # Data retention settings
    with st.expander("🗂️ Data Retention", expanded=True):
        st.subheader("Data Retention Settings")
        retention_period = st.select_slider("Data retention period", 
                                          options=[7, 30, 60, 90, 180, 365], 
                                          format_func=lambda x: f"{x} days",
                                          value=90)
        
        st.write(f"Analytics data will be retained for **{retention_period} days**")
        
        if st.button("Apply Retention Policy"):
            st.session_state.data_retention_days = retention_period
            st.success(f"Data retention policy updated to {retention_period} days")
    
    # Privacy settings
    with st.expander("🔒 Privacy Settings", expanded=True):
        st.subheader("Privacy Controls")
        include_personal_info = st.checkbox("Include personal information in analytics", value=False)

        
        st.info("Personal information is never shared without your explicit consent.")
    
    # Current settings summary
    with st.expander("📋 Current Configuration", expanded=True):
        st.subheader("Current Analytics Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Usage Data Collection:** {'✅ Enabled' if collect_usage_data else '❌ Disabled'}")
            st.write(f"**Performance Data Collection:** {'✅ Enabled' if collect_performance_data else '❌ Disabled'}")
            st.write(f"**Error Reporting:** {'✅ Enabled' if collect_error_data else '❌ Disabled'}")
        with col2:
            st.write(f"**Default Report Format:** {st.session_state.get('default_report_format', 'Markdown')}")
            st.write(f"**Data Retention:** {st.session_state.get('data_retention_days', 90)} days")
            st.write(f"**Personal Info Included:** {'Yes' if include_personal_info else 'No'}")
    
    st.info("💡 Analytics help improve the application. You can change these settings anytime in this panel.")