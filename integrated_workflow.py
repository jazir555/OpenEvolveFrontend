"""
Integrated Adversarial Testing + Evolution Workflow
This module provides the core functionality that combines adversarial testing and evolution
into a single, powerful workflow that enhances content quality through AI-driven critique and refinement.
"""

import streamlit as st
import json
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


from review_utils import determine_review_type, get_appropriate_prompts

from evolution import (
    ContentEvaluator,
    _update_evolution_log_and_status
)
from logging_util import _update_adv_log_and_status

from session_utils import _compose_messages, _safe_list
from openevolve_integration import (
    create_language_specific_evaluator,
    create_specialized_evaluator
)

# Check if OpenEvolve is available for deeper integration
try:

    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using API-based evolution only")


def run_fully_integrated_adversarial_evolution(
    current_content: str,
    content_type: str,
    api_key: str,
    base_url: str,
    red_team_models: List[str],
    blue_team_models: List[str],
    evaluator_models: List[str],
    max_iterations: int,
    adversarial_iterations: int,
    evolution_iterations: int,
    evaluation_iterations: int,
    system_prompt: str,
    evaluator_system_prompt: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    rotation_strategy: str,
    red_team_sample_size: int,
    blue_team_sample_size: int,
    evaluator_sample_size: int,
    confidence_threshold: float,
    evaluator_threshold: float = 90.0,
    evaluator_consecutive_rounds: int = 1,
    compliance_requirements: str = "",
    enable_data_augmentation: bool = False,
    augmentation_model_id: str = None,
    augmentation_temperature: float = 0.7,
    enable_human_feedback: bool = False,
    multi_objective_optimization: bool = False,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    archive_size: int = 100,
    checkpoint_interval: int = 10,
    keyword_analysis_enabled: bool = True,
    keywords_to_target: List[str] = None,
    keyword_penalty_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Run fully integrated adversarial testing, evolution optimization, and evaluation phases.
    
    This function creates a comprehensive workflow that combines adversarial testing (red team),
    evolution optimization (blue team), and evaluation (evaluator team) in a single, optimized 
    process that can adapt and improve content through multiple AI-driven phases with configurable
    acceptance thresholds.
    """
    print(f"Starting fully integrated adversarial-evolution-evaluation process with {adversarial_iterations} adversarial, {evolution_iterations} evolution, and {evaluation_iterations} evaluation iterations")
    _update_adv_log_and_status(f"🚀 Starting fully integrated adversarial-evolution-evaluation with {adversarial_iterations} adversarial, {evolution_iterations} evolution, and {evaluation_iterations} evaluation iterations")
    
    # Initialize results
    integrated_results = {
        "initial_content": current_content,
        "final_content": current_content,
        "adversarial_results": {},
        "evolution_results": {},
        "evaluation_results": {},
        "integrated_score": 0.0,
        "total_cost_usd": 0.0,
        "total_tokens": {"prompt": 0, "completion": 0},
        "process_log": [],
        "success": True,
        "metrics": {},
        "keyword_analysis": {}
    }
    
    # Phase 1: Perform adversarial testing to identify weaknesses and vulnerabilities
    _update_adv_log_and_status("🔍 Phase 1: Starting adversarial testing to identify weaknesses...")
    
    # Perform data augmentation if enabled
    working_content = current_content
    if enable_data_augmentation and augmentation_model_id:
        _update_adv_log_and_status(f"🧪 Augmenting content using {augmentation_model_id} for adversarial testing...")
        working_content = generate_adversarial_data_augmentation(
            content=working_content,
            content_type=content_type,
            api_key=api_key,
            model_id=augmentation_model_id,
            temperature=augmentation_temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        _update_adv_log_and_status("✅ Content augmentation for adversarial testing complete.")
    
    # Run adversarial testing using our enhanced function
    try:
        # Determine appropriate prompts based on content type
        review_type = determine_review_type(working_content)
        red_team_prompt, blue_team_prompt = get_appropriate_prompts(review_type)
        
        # Set up dynamic prompt enhancements based on content type
        red_team_prompt_enhancement = ""
        blue_team_prompt_enhancement = ""
        
        if content_type.startswith("code_"):
            red_team_prompt_enhancement += "\nFocus on security vulnerabilities, code quality, and performance issues in the code. Look for potential bugs, inefficient algorithms, and non-idiomatic code."
            blue_team_prompt_enhancement += "\nFocus on fixing all identified issues, improving code robustness, and optimizing performance. Ensure the code is clean, secure, and follows best practices."
        elif content_type == "document_legal":
            red_team_prompt_enhancement += "\nScrutinize the legal document for ambiguities, loopholes, non-compliance with legal standards (e.g., GDPR, CCPA), and potential liabilities."
            blue_team_prompt_enhancement += "\nRefine the legal document for clarity, enforce compliance with relevant regulations, and mitigate all identified legal risks."
        elif content_type == "document_medical":
            red_team_prompt_enhancement += "\nAnalyze the medical document for factual inaccuracies, patient privacy violations (e.g., HIPAA), ethical concerns, and clarity for medical professionals."
            blue_team_prompt_enhancement += "\nEnsure the medical document is factually accurate, compliant with patient privacy laws, ethically sound, and clearly understandable by medical staff."
        elif content_type == "document_technical":
            red_team_prompt_enhancement += "\nReview the technical document for technical inaccuracies, outdated information, unclear instructions, and potential security implications of described systems."
            blue_team_prompt_enhancement += "\nUpdate the technical document for accuracy, clarity, and completeness. Ensure all technical details are correct and instructions are easy to follow."
        elif review_type == "plan":
            red_team_prompt_enhancement += "\nCritique the plan for feasibility, resource allocation, risk assessment, and alignment with strategic objectives. Identify any hidden dependencies or unrealistic timelines."
            blue_team_prompt_enhancement += "\nRefine the plan to address all identified risks, optimize resource allocation, and ensure feasibility and strategic alignment."
        else:  # General SOP
            red_team_prompt_enhancement += "\nIdentify any weaknesses, inefficiencies, or potential misinterpretations in the general SOP. Focus on clarity, completeness, and robustness."
            blue_team_prompt_enhancement += "\nImprove the general SOP by enhancing clarity, ensuring completeness, and making it more robust against misinterpretation."
        
        # Add compliance requirements if specified
        if compliance_requirements:
            red_team_prompt_enhancement += f"\nAlso, specifically check for compliance with the following requirements: {compliance_requirements}"
            blue_team_prompt_enhancement += f"\nEnsure the final output strictly adheres to the following compliance requirements: {compliance_requirements}"
        
        # Run adversarial testing with our new function
        adversarial_results = run_enhanced_adversarial_loop(
            current_content=working_content,
            api_key=api_key,
            base_url=base_url,
            red_team_models=red_team_models,
            blue_team_models=blue_team_models,
            max_iterations=adversarial_iterations,
            population_size=len(red_team_models + blue_team_models),
            red_team_prompt=red_team_prompt,
            blue_team_prompt=blue_team_prompt,
            approval_prompt=st.session_state.get("adversarial_custom_approval_prompt", "You are an evaluator assessing the quality of a Standard Operating Procedure (SOP). Please evaluate the provided SOP according to these criteria..."),
            extra_headers={},
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed,
            rotation_strategy=rotation_strategy,
            red_team_sample_size=red_team_sample_size,
            blue_team_sample_size=blue_team_sample_size,
            compliance_requirements=compliance_requirements,
            critique_depth=st.session_state.get("adversarial_critique_depth", 5),
            patch_quality=st.session_state.get("adversarial_patch_quality", 5),
            review_type=review_type,
            red_team_prompt_enhancement=red_team_prompt_enhancement,
            blue_team_prompt_enhancement=blue_team_prompt_enhancement,
        )
        
        # Get the adversarially improved content
        adversarially_improved = adversarial_results.get("final_content", working_content)
        _update_adv_log_and_status(f"✅ Adversarial testing complete. Content improved from {len(working_content)} to {len(adversarially_improved)} chars.")
        
        # Update the content to evolve
        working_content = adversarially_improved
        
        # Store adversarial results
        integrated_results["adversarial_results"] = adversarial_results
        
    except Exception as e:
        _update_adv_log_and_status(f"⚠️ Adversarial testing failed: {e}. Continuing with original content.")
        # Continue with evolution even if adversarial testing fails
        import traceback
        traceback.print_exc()
    
    # Phase 2: Run evolution optimization on the adversarially improved content
    _update_adv_log_and_status("🔄 Phase 2: Starting evolution optimization on adversarially improved content...")
    
    # Determine evaluator based on content type
    if content_type.startswith("code_"):
        evaluator_func = create_specialized_evaluator(content_type, compliance_requirements)
    else:
        evaluator_func = create_language_specific_evaluator(content_type, compliance_requirements)
    
    # Prepare extra headers
    extra_headers = json.loads(st.session_state.get("extra_headers", "{}"))
    
    # Run evolution loop
    try:
        evolution_final_content = run_enhanced_evolution_loop(
            current_content=working_content,
            api_key=api_key,
            base_url=base_url,
            model=red_team_models[0] if red_team_models else "gpt-4o",  # Use first red team model or default
            max_iterations=evolution_iterations,
            population_size=len(blue_team_models) if blue_team_models else 5,  # Use blue team models as population
            system_prompt=system_prompt,
            evaluator=evaluator_func,
            extra_headers=extra_headers,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed,
            multi_objective_optimization=multi_objective_optimization,
            feature_dimensions=feature_dimensions,
            feature_bins=feature_bins,
            elite_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            archive_size=archive_size,
            checkpoint_interval=checkpoint_interval
        )
        
        _update_adv_log_and_status(f"✅ Evolution optimization complete. Final content length: {len(evolution_final_content)} chars.")
        
        # Update content for evaluation phase
        working_content = evolution_final_content
        
        # Store evolution results
        integrated_results["evolution_results"]["final_content"] = evolution_final_content
        integrated_results["evolution_results"]["process_stage"] = "evolution_completed"
        
    except Exception as e:
        _update_adv_log_and_status(f"❌ Evolution optimization failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue with evaluation using the best available content
        working_content = adversarially_improved
    
    # Phase 3: Run evaluation by evaluator team with configurable acceptance thresholds
    _update_adv_log_and_status("⚖️ Phase 3: Starting evaluation by evaluator team...")
    
    # Perform keyword analysis if enabled
    if keyword_analysis_enabled:
        keywords_to_target = keywords_to_target or []
        _update_adv_log_and_status(f"🔍 Performing keyword analysis for: {', '.join(keywords_to_target)}")
        keyword_analysis_result = analyze_content_keywords(working_content, keywords_to_target)
        integrated_results["keyword_analysis"] = keyword_analysis_result
        
        # Enhance prompts with keyword guidance
        if keywords_to_target:
            keyword_guidance = "\n\nAdditional requirement: The content should appropriately incorporate these keywords: " + ", ".join(keywords_to_target)
            system_prompt += keyword_guidance
            evaluator_system_prompt += keyword_guidance
    
    # Run evaluation loop
    try:
        evaluation_results = run_evaluator_loop(
            current_content=working_content,
            api_key=api_key,
            base_url=base_url,
            evaluator_models=evaluator_models,
            max_iterations=evaluation_iterations,
            evaluator_sample_size=evaluator_sample_size,
            evaluator_threshold=evaluator_threshold,
            evaluator_consecutive_rounds=evaluator_consecutive_rounds,
            evaluator_system_prompt=evaluator_system_prompt,
            extra_headers=extra_headers,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed,
            rotation_strategy=rotation_strategy
        )
        
        _update_adv_log_and_status(f"✅ Evaluation complete. Final content length: {len(evaluation_results.get('final_content', working_content))} chars.")
        
        # Update final results
        integrated_results["final_content"] = evaluation_results.get("final_content", working_content)
        integrated_results["evaluation_results"] = evaluation_results
        integrated_results["evaluation_results"]["process_stage"] = "evaluation_completed"
        
        # Calculate integrated score based on all processes
        integrated_score = calculate_integrated_score(
            adversarial_results,
            evolution_final_content if 'evolution_final_content' in locals() else working_content,
            evaluation_results.get('final_content', working_content)
        )
        integrated_results["integrated_score"] = integrated_score
        
        return integrated_results
        
    except Exception as e:
        _update_adv_log_and_status(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        integrated_results["success"] = False
        integrated_results["error"] = str(e)
        integrated_results["final_content"] = working_content  # Return best available content
        return integrated_results


def run_evaluator_loop(
    current_content: str,
    api_key: str,
    base_url: str,
    evaluator_models: List[str],
    max_iterations: int,
    evaluator_sample_size: int,
    evaluator_threshold: float,
    evaluator_consecutive_rounds: int,
    evaluator_system_prompt: str,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    rotation_strategy: str = "Round Robin"
) -> Dict[str, Any]:
    """
    Run the evaluator loop with configurable acceptance thresholds.
    
    Args:
        current_content: The content to evaluate
        api_key: API key for the models
        base_url: Base URL for the API
        evaluator_models: List of evaluator models
        max_iterations: Maximum number of evaluation iterations
        evaluator_sample_size: Number of evaluators to use per iteration
        evaluator_threshold: Minimum score threshold for acceptance (e.g., 90.0 for 9/10)
        evaluator_consecutive_rounds: Number of consecutive rounds required for acceptance
        evaluator_system_prompt: System prompt for evaluators
        extra_headers: Extra headers for the API
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        max_tokens: Maximum tokens to generate
        seed: Random seed
        rotation_strategy: Strategy for rotating evaluators
    """
    _update_adv_log_and_status(f"⚖️ Starting evaluator loop with threshold {evaluator_threshold}% for {evaluator_consecutive_rounds} consecutive rounds")
    
    iteration_results = []
    current_content_text = current_content
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    
    # Track consecutive rounds meeting threshold
    consecutive_acceptable_rounds = 0
    
    # Initialize rotation state
    rotation_state = 0
    
    for iteration in range(max_iterations):
        _update_adv_log_and_status(f"--- Evaluation Iteration {iteration + 1}/{max_iterations} ---")

        # Select evaluator models based on rotation strategy
        selected_evaluators = []
        if rotation_strategy == "Round Robin":
            for _ in range(evaluator_sample_size):
                selected_evaluators.append(evaluator_models[rotation_state % len(evaluator_models)])
                rotation_state += 1
        elif rotation_strategy == "Random Sampling":
            import random
            selected_evaluators = random.sample(
                evaluator_models, 
                min(evaluator_sample_size, len(evaluator_models))
            )
        elif rotation_strategy == "Arbitrary Rotation":
            # For arbitrary rotation, allow custom configurations
            # This could be implemented based on specific configuration needs
            selected_evaluators = evaluator_models[:evaluator_sample_size]
        else:  # Default to using all models
            selected_evaluators = evaluator_models

        if not selected_evaluators:
            _update_adv_log_and_status("❌ Error: No evaluator models selected for current iteration. Stopping.")
            break

        _update_adv_log_and_status(f"Evaluator Team: {', '.join(selected_evaluators)}")

        # Run evaluation with all selected evaluators
        evaluation_scores = []
        model_configs = _collect_model_configs(selected_evaluators, max_tokens)
        
        for model_id in selected_evaluators:
            try:
                result = analyze_with_model(
                    api_key,
                    model_id,
                    current_content_text,
                    model_configs.get(model_id, {}),
                    evaluator_system_prompt,
                    force_json=True,
                    seed=seed,
                    extra_headers=extra_headers
                )
                
                if result.get("ok"):
                    evaluation_json = result.get("json", {})
                    score = evaluation_json.get("score", 0)
                    evaluation_scores.append(score)
                    _update_adv_log_and_status(f"⚖️ {model_id} gave score: {score}")
                    
                    # Add prompt and completion tokens to totals
                    total_prompt_tokens += result.get("ptoks", 0)
                    total_completion_tokens += result.get("ctoks", 0)
                    total_cost += result.get("cost", 0.0)
                    
                else:
                    _update_adv_log_and_status(f"⚖️ {model_id} evaluation failed: {result.get('text')}")
                    evaluation_scores.append(0)  # Default score
                    
            except Exception as e:
                _update_adv_log_and_status(f"⚖️ {model_id} evaluation error: {e}")
                evaluation_scores.append(0)  # Default score

        # Calculate whether all evaluators met the threshold
        all_meet_threshold = all(score >= evaluator_threshold for score in evaluation_scores)
        avg_score = sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0
        
        # Update consecutive rounds counter
        if all_meet_threshold:
            consecutive_acceptable_rounds += 1
        else:
            consecutive_acceptable_rounds = 0  # Reset counter if threshold not met
        
        _update_adv_log_and_status(f"📊 Avg Score: {avg_score:.1f}%, All evaluators >= {evaluator_threshold}%: {all_meet_threshold}, Consecutive rounds: {consecutive_acceptable_rounds}/{evaluator_consecutive_rounds}")

        # Add this iteration's results
        iteration_results.append({
            "iteration": iteration + 1,
            "evaluator_models": selected_evaluators,
            "scores": evaluation_scores,
            "average_score": avg_score,
            "all_meet_threshold": all_meet_threshold,
            "consecutive_acceptable_rounds": consecutive_acceptable_rounds
        })

        # Check if content meets acceptance criteria
        if consecutive_acceptable_rounds >= evaluator_consecutive_rounds:
            _update_adv_log_and_status(f"✅ Content accepted after {consecutive_acceptable_rounds} consecutive rounds meeting threshold!")
            break

        # Content doesn't meet threshold, continue to next iteration
        if iteration + 1 < max_iterations:
            # Optionally modify content based on evaluator feedback before next iteration
            # This could involve using suggestions from evaluators to improve content
            pass

    # Return final results
    return {
        "final_content": current_content_text,
        "iterations": iteration_results,
        "final_score": avg_score,
        "all_evaluators_met_threshold": all_meet_threshold,
        "consecutive_rounds_met": consecutive_acceptable_rounds >= evaluator_consecutive_rounds,
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens
        },
        "cost": total_cost,
        "success": consecutive_acceptable_rounds >= evaluator_consecutive_rounds
    }


def analyze_content_keywords(content: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Analyze content for keyword presence and relevance.
    
    Args:
        content: The content to analyze
        keywords: List of keywords to search for
    
    Returns:
        Dictionary with keyword analysis results
    """
    analysis = {
        "keywords_found": [],
        "keyword_density": {},
        "keyword_positions": {},
        "relevance_score": 0.0
    }
    
    content_lower = content.lower()
    total_words = len(content.split())
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = content_lower.count(keyword_lower)
        
        if count > 0:
            analysis["keywords_found"].append(keyword)
            analysis["keyword_density"][keyword] = count / max(1, total_words) * 100
            
            # Find positions of the keyword
            positions = []
            start = 0
            while True:
                pos = content_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            analysis["keyword_positions"][keyword] = positions
    
    # Calculate relevance score (how many keywords were found)
    if keywords:
        analysis["relevance_score"] = len(analysis["keywords_found"]) / len(keywords)
    
    return analysis


def run_enhanced_adversarial_loop(
    current_content: str,
    api_key: str,
    base_url: str,
    red_team_models: List[str],
    blue_team_models: List[str],
    max_iterations: int,
    population_size: int,
    red_team_prompt: str,
    blue_team_prompt: str,
    approval_prompt: str,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    rotation_strategy: str,
    red_team_sample_size: int,
    blue_team_sample_size: int,
    compliance_requirements: str = "",
    critique_depth: int = 5,
    patch_quality: int = 5,
    review_type: str = "general",
    red_team_prompt_enhancement: str = "",
    blue_team_prompt_enhancement: str = "",
) -> Dict[str, Any]:
    """
    Enhanced adversarial loop with better integration capabilities for the evolution phase.
    """
    print(f"Starting enhanced adversarial loop with {max_iterations} iterations")
    _update_adv_log_and_status(f"🚀 Starting enhanced adversarial loop with {max_iterations} iterations")
    
    iteration_results = []
    current_content_text = current_content
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    final_approval_rate = 0.0
    
    # Initialize model rotation state
    rotation_state = {"red_idx": 0, "blue_idx": 0}
    
    for iteration in range(max_iterations):
        with st.session_state.thread_lock:
            if st.session_state.get("adversarial_stop_flag", False):
                _update_adv_log_and_status("⏹️ Adversarial evolution stopped by user.")
                break

        _update_adv_log_and_status(f"--- Adversarial Iteration {iteration + 1}/{max_iterations} ---")

        # --- Model Selection based on Rotation Strategy ---
        red_team_models_selected = []
        blue_team_models_selected = []

        if rotation_strategy == "Round Robin":
            for _ in range(red_team_sample_size):
                red_team_models_selected.append(red_team_models[rotation_state["red_idx"]])
                rotation_state["red_idx"] = (rotation_state["red_idx"] + 1) % len(red_team_models)
            for _ in range(blue_team_sample_size):
                blue_team_models_selected.append(blue_team_models[rotation_state["blue_idx"]])
                rotation_state["blue_idx"] = (rotation_state["blue_idx"] + 1) % len(blue_team_models)
        elif rotation_strategy == "Random Sampling":
            red_team_models_selected = red_team_models[:red_team_sample_size]
            blue_team_models_selected = blue_team_models[:blue_team_sample_size]
        else:  # Default to using all models
            red_team_models_selected = red_team_models
            blue_team_models_selected = blue_team_models

        if not red_team_models_selected or not blue_team_models_selected:
            _update_adv_log_and_status("❌ Error: No models selected for current iteration. Stopping.")
            break

        _update_adv_log_and_status(f"Red Team: {', '.join(red_team_models_selected)} | Blue Team: {', '.join(blue_team_models_selected)}")

        # --- Red Team Critiques ---
        critiques = []
        model_configs = _collect_model_configs(red_team_models_selected + blue_team_models_selected, max_tokens)
        
        with ThreadPoolExecutor(max_workers=population_size) as ex:
            future_to_critique = {
                ex.submit(
                    analyze_with_model,
                    api_key,
                    model_id,
                    current_content_text,
                    model_configs.get(model_id, {}),
                    red_team_prompt,
                    user_suffix=f"Critique Depth: {critique_depth}",
                    force_json=True,
                    seed=seed,
                    compliance_requirements=compliance_requirements,
                    prompt_enhancement=red_team_prompt_enhancement,
                ): model_id
                for model_id in red_team_models_selected
            }
            for future in as_completed(future_to_critique):
                model_id = future_to_critique[future]
                try:
                    res = future.result()
                    total_prompt_tokens += res["ptoks"]
                    total_completion_tokens += res["ctoks"]
                    total_cost += res["cost"]
                    if res["ok"]:
                        critiques.append(
                            {"model": model_id, "critique_json": res["json"], "raw_text": res["text"]}
                        )
                        _update_adv_log_and_status(f"🔴 {model_id} critiqued content.")
                    else:
                        _update_adv_log_and_status(f"🔴 {model_id} critique failed: {res['text']}")
                except Exception as exc:
                    _update_adv_log_and_status(f"🔴 {model_id} generated an exception: {exc}")

        if not critiques:
            _update_adv_log_and_status("⚠️ No valid critiques generated. Stopping.")
            break

        # --- Blue Team Patches ---
        blue_patches = []
        with ThreadPoolExecutor(max_workers=population_size) as ex:
            future_to_patch = {
                ex.submit(
                    analyze_with_model,
                    api_key,
                    model_id,
                    current_content_text,
                    model_configs.get(model_id, {}),
                    blue_team_prompt,
                    user_suffix=f"""Critiques:
{json.dumps([c['critique_json'] for c in critiques if c['critique_json']], indent=2)}

Patch Quality: {patch_quality}""",
                    force_json=True,
                    seed=seed,
                    compliance_requirements=compliance_requirements,
                    prompt_enhancement=blue_team_prompt_enhancement,
                ): model_id
                for model_id in blue_team_models_selected
            }
            for future in as_completed(future_to_patch):
                model_id = future_to_patch[future]
                try:
                    res = future.result()
                    total_prompt_tokens += res["ptoks"]
                    total_completion_tokens += res["ctoks"]
                    total_cost += res["cost"]
                    if res["ok"]:
                        blue_patches.append(
                            {"model": model_id, "patch_json": res["json"], "raw_text": res["text"]}
                        )
                        _update_adv_log_and_status(f"🔵 {model_id} patched content.")
                    else:
                        _update_adv_log_and_status(f"🔵 {model_id} patch failed: {res['text']}")
                except Exception as exc:
                    _update_adv_log_and_status(f"🔵 {model_id} generated an exception: {exc}")

        # --- Consensus Merging ---
        new_content_text, merge_diagnostics = _merge_consensus_sop(
            current_content_text, blue_patches, critiques
        )
        _update_adv_log_and_status(f"✨ Content merged. Reason: {merge_diagnostics.get('reason')}, Score: {merge_diagnostics.get('score')}")
        
        # Prepare detailed diagnostics for the evolution phase
        detailed_diagnostics = {
            "merge_diagnostics": merge_diagnostics,
            "critiques_summary": {
                "total_critiques": len(critiques),
                "issues_found": sum(len(_safe_list(c.get("critique_json", {}), "issues")) for c in critiques),
                "average_severity": calculate_average_severity(critiques)
            },
            "patches_summary": {
                "total_patches": len(blue_patches),
                "mitigation_count": count_mitigations(blue_patches)
            }
        }

        # --- Approval Check ---
        approval_check = check_approval_rate(
            api_key,
            red_team_models_selected,
            new_content_text,
            model_configs,
            seed,
            population_size,
            approval_prompt,
        )
        total_prompt_tokens += approval_check["prompt_tokens"]
        total_completion_tokens += approval_check["completion_tokens"]
        total_cost += approval_check["cost"]
        final_approval_rate = approval_check["approval_rate"]
        _update_adv_log_and_status(f"✅ Approval Rate: {final_approval_rate:.1f}%")

        # --- Update State ---
        iteration_results.append(
            {
                "iteration": iteration + 1,
                "content_before_patch": current_content_text,
                "critiques": critiques,
                "patches": blue_patches,
                "content_after_patch": new_content_text,
                "merge_diagnostics": merge_diagnostics,
                "detailed_diagnostics": detailed_diagnostics,
                "approval_check": approval_check,
                "agg_risk": _aggregate_red_risk(critiques),
            }
        )
        current_content_text = new_content_text

        # Update session state with progress information for the evolution phase
        with st.session_state.thread_lock:
            if "integrated_adversarial_history" not in st.session_state:
                st.session_state.integrated_adversarial_history = []
            st.session_state.integrated_adversarial_history.append({
                "iteration": iteration + 1,
                "approval_rate": final_approval_rate,
                "issues_found": detailed_diagnostics["critiques_summary"]["issues_found"],
                "mitigations": detailed_diagnostics["patches_summary"]["mitigation_count"],
                "content_length": len(new_content_text)
            })

    _update_adv_log_and_status("✅ Enhanced adversarial evolution complete.")
    
    return {
        "final_content": current_content_text,
        "final_approval_rate": final_approval_rate,
        "iterations": iteration_results,
        "cost_estimate_usd": total_cost,
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
        },
        "log": st.session_state.get("adversarial_log", []),
        "detailed_diagnostics": detailed_diagnostics
    }


def run_enhanced_evolution_loop(
    current_content: str,
    api_key: str,
    base_url: str,
    model: str,
    max_iterations: int,
    population_size: int,
    system_prompt: str,
    evaluator: ContentEvaluator,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    multi_objective_optimization: bool = False,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    archive_size: int = 100,
    checkpoint_interval: int = 10
):
    """
    Enhanced evolution loop that can use adversarial testing results for better optimization.
    This function now prefers OpenEvolve when available for more sophisticated evolution.
    """
    try:
        # If OpenEvolve is available, use it for the more sophisticated evolution process
        if OPENEVOLVE_AVAILABLE:
            _update_evolution_log_and_status("🔄 Using OpenEvolve backend for enhanced evolution...")
            
            # Create OpenEvolve configuration
            from openevolve.config import Config, LLMModelConfig
            from openevolve.api import run_evolution
            
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model,
                api_key=api_key,
                api_base=base_url if base_url else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                seed=seed,
            )
            
            config.llm.models = [llm_config]
            config.max_iterations = max_iterations
            config.database.population_size = population_size
            config.database.archive_size = archive_size
            config.checkpoint_interval = checkpoint_interval
            config.database.num_islands = st.session_state.get("num_islands", 1)  # Add island model for better exploration
            
            # Configure database settings for multi-objective evolution if needed
            if feature_dimensions is not None:
                config.database.feature_dimensions = feature_dimensions
            if feature_bins is not None:
                config.database.feature_bins = feature_bins
            else:
                # Set default feature bins if none provided
                config.database.feature_bins = 10
            
            # Configure ratios
            config.database.elite_selection_ratio = elite_ratio
            config.database.exploration_ratio = exploration_ratio
            config.database.exploitation_ratio = exploitation_ratio
            
            # Configure evaluator settings for better integration
            config.evaluator.timeout = 300
            config.evaluator.max_retries = 3
            config.evaluator.cascade_evaluation = True
            config.evaluator.cascade_thresholds = [0.5, 0.75, 0.9]
            config.evaluator.parallel_evaluations = os.cpu_count() or 4
            
            # Use adversarial diagnostics to inform the evolution process
            adversarial_history = st.session_state.get("integrated_adversarial_history", [])
            if adversarial_history and multi_objective_optimization:
                # If using multi-objective optimization and have adversarial history, 
                # add relevant feature dimensions
                if config.database.feature_dimensions is not None:
                    # Add adversarial-relevant dimensions if they're not already present
                    if "issues_resolved" not in config.database.feature_dimensions:
                        config.database.feature_dimensions.append("issues_resolved")
                    if "mitigation_effectiveness" not in config.database.feature_dimensions:
                        config.database.feature_dimensions.append("mitigation_effectiveness")

            # Create a temporary file for the content
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
                temp_file.write(current_content)
                temp_file_path = temp_file.name
            
            try:
                # Use OpenEvolve API with the evaluator
                result = run_evolution(
                    initial_program=temp_file_path,
                    evaluator=evaluator.evaluate,
                    config=config,
                    iterations=max_iterations,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                if result.best_program and result.best_code:
                    final_content = result.best_code
                    with st.session_state.thread_lock:
                        st.session_state.evolution_current_best = final_content
                    _update_evolution_log_and_status(
                        f"🏆 OpenEvolve enhanced evolution completed. Best score: {result.best_score:.4f}"
                    )
                    return final_content
                else:
                    _update_evolution_log_and_status(
                        "🤔 OpenEvolve enhanced evolution completed with no improvement."
                    )
                    return current_content
                    
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            # Fallback to the custom evolution loop when OpenEvolve is not available
            _update_evolution_log_and_status("⚠️ OpenEvolve not available, using API-based enhanced evolution as fallback...")
            for i in range(max_iterations):
                # Check if we have adversarial stop flag
                if st.session_state.get("adversarial_stop_flag", False):
                    _update_evolution_log_and_status("⏹️ Evolution stopped due to adversarial stop flag.")
                    break

                _update_evolution_log_and_status(f"🔄 Evolution Iteration {i + 1}/{max_iterations}")
                _update_evolution_log_and_status("🧬 Generating new population...")

                if i % checkpoint_interval == 0:
                    _update_evolution_log_and_status(f"💾 Saving checkpoint at evolution iteration {i}")

                # Generate new population
                with ThreadPoolExecutor(max_workers=population_size) as executor:
                    futures = [
                        executor.submit(
                            _request_openai_compatible_chat,
                            api_key,
                            base_url,
                            model,
                            _compose_messages(system_prompt, current_content),
                            extra_headers,
                            temperature,
                            top_p,
                            frequency_penalty,
                            presence_penalty,
                            max_tokens,
                            seed + i if seed is not None else None,
                        )
                        for i in range(population_size)
                    ]

                    new_population = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            new_population.append(result)
                        except Exception as e:
                            _update_evolution_log_and_status(f"❌ Error generating candidate: {e}")
                            # Add a placeholder if generation fails
                            new_population.append(current_content)

                _update_evolution_log_and_status("🔍 Evaluating new population...")

                best_candidate = ""
                best_score = -1

                # Use adversarial diagnostics to inform the evolution process
                adversarial_history = st.session_state.get("integrated_adversarial_history", [])
                if adversarial_history:
                    # Get the latest adversarial diagnostics to inform evolution
                    latest_diag = adversarial_history[-1]
                    # If there were many issues found in adversarial testing, 
                    # we should focus more on fixing those issues in evolution
                    if latest_diag.get("issues_found", 0) > latest_diag.get("mitigations", 0):
                        # Adjust system prompt to focus on fixing issues
                        improvement_system_prompt = f"{system_prompt}\n\nBased on adversarial testing results, there are still more issues than have been mitigated. Focus on addressing remaining issues and vulnerabilities identified during adversarial testing."
                        
                        # Generate alternative candidates specifically focused on addressing issues
                        with ThreadPoolExecutor(max_workers=max(1, population_size//2)) as executor:
                            improvement_futures = [
                                executor.submit(
                                    _request_openai_compatible_chat,
                                    api_key,
                                    base_url,
                                    model,
                                    _compose_messages(improvement_system_prompt, current_content),
                                    extra_headers,
                                    temperature,
                                    top_p,
                                    frequency_penalty,
                                    presence_penalty,
                                    max_tokens,
                                    seed + population_size + i if seed is not None else None,
                                )
                                for i in range(max(1, population_size//2))
                            ]
                            
                            for future in as_completed(improvement_futures):
                                try:
                                    result = future.result()
                                    new_population.append(result)
                                except Exception as e:
                                    _update_evolution_log_and_status(f"❌ Error generating improvement candidate: {e}")

                # Evaluate new population
                with ThreadPoolExecutor(max_workers=population_size) as executor:
                    futures = {
                        executor.submit(
                            _evaluate_candidate,
                            candidate,
                            api_key,
                            base_url,
                            model,
                            evaluator,
                            extra_headers,
                            temperature,
                            top_p,
                            frequency_penalty,
                            presence_penalty,
                            max_tokens,
                            seed,
                        ): candidate
                        for candidate in new_population
                    }

                    for future in as_completed(futures):
                        try:
                            candidate = futures[future]
                            score = future.result()

                            if score > best_score:
                                best_score = score
                                best_candidate = candidate
                        except Exception as e:
                            _update_evolution_log_and_status(f"❌ Error evaluating candidate: {e}")
                            continue

                if best_candidate and best_score > 0:
                    current_content = best_candidate
                    with st.session_state.thread_lock:
                        st.session_state.evolution_current_best = current_content
                    _update_evolution_log_and_status(
                        f"🏆 New best candidate found with score: {best_score:.2f}"
                    )
                else:
                    _update_evolution_log_and_status("🤔 No improvement in this iteration.")

            _update_evolution_log_and_status("🏁 Enhanced evolution finished.")
            return current_content
    except Exception as e:
        _update_evolution_log_and_status(f"💥 Enhanced evolution loop failed: {e}")
        import traceback
        traceback.print_exc()
        return current_content


def calculate_integrated_score(
    adversarial_results: Dict, 
    evolution_final_content: str, 
    initial_content: str
) -> float:
    """
    Calculate a comprehensive score that takes into account both adversarial testing and evolution results.
    """
    # Base score components
    adversarial_score = adversarial_results.get("final_approval_rate", 0.0) / 100.0  # Convert percentage to 0-1
    
    # Content improvement score (how much content improved in length and quality)
    initial_length = len(initial_content)
    final_length = len(evolution_final_content)
    length_improvement = max(0, min(1, (final_length - initial_length) / max(1, initial_length))) if initial_length > 0 else 0
    
    # Calculate issue resolution rate from adversarial testing
    total_issues = 0
    resolved_issues = 0
    
    for iteration in adversarial_results.get("iterations", []):
        for critique in iteration.get("critiques", []):
            critique_json = critique.get("critique_json", {})
            if critique_json:
                total_issues += len(_safe_list(critique_json, "issues"))
        
        for patch in iteration.get("patches", []):
            patch_json = patch.get("patch_json", {})
            if patch_json:
                mitigation_matrix = _safe_list(patch_json, "mitigation_matrix")
                resolved_issues += len([m for m in mitigation_matrix if m.get("status", "").lower() in ["resolved", "mitigated"]])
    
    resolution_rate = resolved_issues / max(1, total_issues) if total_issues > 0 else 1.0
    
    # Combine scores with weights
    final_score = (
        0.4 * adversarial_score +  # 40% from adversarial approval rate
        0.3 * resolution_rate +    # 30% from issue resolution rate
        0.3 * min(1.0, length_improvement * 2)  # 30% from content improvement (capped)
    )
    
    return final_score


def generate_adversarial_data_augmentation(
    content: str,
    content_type: str,
    api_key: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    seed: Optional[int] = None,
    augmentation_strategy: str = "rephrase",
) -> str:
    """
    Generates an augmented version of the content using an LLM for adversarial testing purposes.
    """
    try:
        system_prompt = f"You are an expert content rephraser. Your task is to rephrase the provided {content_type} content to make it more complex, ambiguous, or subtly flawed, without changing its core functionality or intent. Focus on introducing nuances that might challenge an AI reviewer."
        user_prompt = f"Rephrase the following {content_type} content:\n\n---\n\n{content}\n\n---\n\nRephrased content:"

        # Use our internal request function
        augmented_content, _, _, _ = _request_openai_compatible_chat(
            api_key=api_key,
            base_url=st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
            model=model_id,
            messages=_compose_messages(system_prompt, user_prompt),
            extra_headers={},
            temperature=temperature,
            top_p=1.0,  # Keep top_p high for more diverse rephrasing
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        return augmented_content
    except Exception as e:
        _update_adv_log_and_status(f"Error during adversarial data augmentation: {e}")
        return content  # Return original content on error


def capture_integrated_human_feedback(
    content_example: Dict[str, Any], 
    human_score: float, 
    human_comments: str
):
    """
    Captures human feedback on integrated content improvement.
    """
    feedback_entry = {
        "timestamp": time.time(),
        "content_example_id": content_example.get("id"),
        "human_score": human_score,
        "human_comments": human_comments,
        "content": content_example.get("content"),  # Store content for context
        "content_type": content_example.get("content_type"),
        "iteration": content_example.get("iteration"),
        "process_stage": content_example.get("process_stage", "unknown"),
    }


    all_feedback = []
    if st.session_state.get("human_feedback_log"):
        all_feedback = st.session_state["human_feedback_log"]

    all_feedback.append(feedback_entry)

    try:
        # Update session state with new feedback
        st.session_state["human_feedback_log"] = all_feedback
        _update_adv_log_and_status(
            f"📝 Captured integrated human feedback for content example {content_example.get('id')}"
        )
    except Exception as e:
        _update_adv_log_and_status(f"❌ Failed to save integrated human feedback: {e}")


def calculate_average_severity(critiques: List[Dict]) -> float:
    """Calculate the average severity of issues found in critiques."""
    severity_map = {"low": 1, "medium": 3, "high": 6, "critical": 10}
    total_severity = 0
    issue_count = 0
    
    for critique in critiques:
        critique_json = critique.get("critique_json", {})
        if critique_json:
            issues = _safe_list(critique_json, "issues")
            for issue in issues:
                severity = issue.get("severity", "low").lower()
                total_severity += severity_map.get(severity, 1)
                issue_count += 1
    
    return total_severity / max(1, issue_count) if issue_count > 0 else 0


def count_mitigations(patches: List[Dict]) -> int:
    """Count the number of mitigations in patches."""
    mitigation_count = 0
    for patch in patches:
        patch_json = patch.get("patch_json", {})
        if patch_json:
            mitigation_matrix = _safe_list(patch_json, "mitigation_matrix")
            for mitigation in mitigation_matrix:
                if mitigation.get("status", "").lower() in ["resolved", "mitigated"]:
                    mitigation_count += 1
    return mitigation_count


def _collect_model_configs(
    model_ids: List[str], max_tokens: int
) -> Dict[str, Dict[str, Any]]:
    """Collect model configurations."""
    return {
        model_id: {
            "temperature": st.session_state.get(f"temp_{model_id}", 0.7),
            "top_p": st.session_state.get(f"topp_{model_id}", 1.0),
            "frequency_penalty": st.session_state.get(f"freqpen_{model_id}", 0.0),
            "presence_penalty": st.session_state.get(f"prespen_{model_id}", 0.0),
            "max_tokens": max_tokens,
        }
        for model_id in model_ids
    }


def _hash_text(text: str) -> str:
    """Create a hash of the text."""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()


@st.cache_data(ttl=3600) # Cache for 1 hour
def _request_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    req_timeout: int = 60,
    max_retries: int = 5,
    provider: str = "OpenAI",
) -> str:
    """Make a request to an OpenAI-compatible API."""
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers,
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": min(max(temperature, 0.0), 2.0),
        "max_tokens": max_tokens,
        "top_p": min(max(top_p, 0.0), 1.0),
        "frequency_penalty": min(max(frequency_penalty, -2.0), 2.0),
        "presence_penalty": min(max(presence_penalty, -2.0), 2.0),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code in {429, 500, 502, 503, 504}:
                import time
                sleep_s = (2**attempt) + (time.time() % 0.1)  # Add jitter
                time.sleep(sleep_s)
                last_err = Exception(f"Transient error {r.status_code}: Retrying...")
                continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
            else:
                content = ""
            return content or ""
        except Exception as e:
            last_err = e
            sleep_s = (2**attempt) + (time.time() % 0.1)  # Add jitter
            time.sleep(sleep_s)
    raise RuntimeError(
        f"Request failed for {model} after {max_retries} attempts: {last_err}"
    )


def _evaluate_candidate(
    candidate: str,
    api_key: str,
    base_url: str,
    model: str,
    evaluator: ContentEvaluator,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
) -> float:
    """
    Evaluate a single candidate using the evaluator.
    """
    try:
        # If the evaluator has an evaluate method that works with file paths,
        # we need to create a temporary file for the candidate
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(candidate)
            temp_file_path = temp_file.name
        
        try:
            # Call the evaluator
            evaluation_result = evaluator.evaluate(temp_file_path)
            score = evaluation_result.get("score", 0.0)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        return score
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return 0.0  # Return zero score if evaluation fails