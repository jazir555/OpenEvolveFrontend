import dataclasses # Added for dataclasses.is_dataclass
import streamlit as st
import time
import json
import uuid
import threading # Added for parallel execution in gauntlets
import os # Added for path manipulation in OpenEvolve integration
import re # Added for regex parsing in targeted feedback
from typing import Any, Dict, List, Literal, Optional

import streamlit as st
from ui_components import render_manual_review_panel # Import for Stage 2 UI


from workflow_structures import (
    CritiqueReport, DecompositionPlan, GauntletDefinition, GauntletRoundRule,
    ModelConfig, SolutionAttempt, SubProblem, Team, VerificationReport,
    WorkflowState
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config

# Initialize managers (assuming they are initialized in ui_components or main app)
# These managers are used to retrieve Team and Gauntlet definitions.
team_manager = TeamManager()
gauntlet_manager = GauntletManager()

from llm_utils import _request_openai_compatible_chat


def _compose_messages(system_message: str, user_message: str) -> List[Dict[str, str]]:
    """Helper function to compose messages in the format expected by OpenAI-compatible chat APIs.

    Args:
        system_message (str): The system message to set the context or role of the AI.
        user_message (str): The user's message or prompt.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries.
    """
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

# --- Core Workflow Engine Functions ---

def run_content_analysis(problem_statement: str, team: Team) -> Dict[str, Any]:
    """
    Executes Stage 0: Content Analysis. A Blue Team analyzes the problem statement and extracts structured context.
    All members of the team contribute to the analysis, and their outputs are combined.

    Args:
        problem_statement (str): The raw, high-level problem description from the user.
        team (Team): The Blue Team (role: `Content Analyzer`) responsible for the analysis.

    Returns:
        Dict[str, Any]: An `AnalyzedContext` object (dictionary) containing structured information
                        extracted from the problem statement, or an error message if analysis fails.
    """
    if not team.members:
        st.error(f"Content Analysis Team '{team.name}' has no members. Please configure the team in the Team Manager.")
        return {"error": "No team members"}

    analyses = []
    threads = []

    system_prompt = team.content_analysis_system_prompt if team.content_analysis_system_prompt else "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format."
    user_prompt_template = team.content_analysis_user_prompt_template if team.content_analysis_user_prompt_template else f"""Analyze the following problem statement and extract:
    - `domain`: (e.g., "Software Development", "Physics", "Legal")
    - `keywords`: List of important terms.
    - `estimated_complexity`: (1-10)
    - `potential_challenges`: List of anticipated difficulties.
    - `required_expertise`: List of expertise areas needed.
    - `summary`: A brief, concise summary of the problem.

    Problem Statement:
    ---
    {{problem_statement}}
    ---
    """

    def _analyze_with_model(model_config: ModelConfig):
        # Replace the {{problem_statement}} placeholder in the user prompt template
        formatted_user_prompt = user_prompt_template.replace("{{problem_statement}}", problem_statement)
        response = _request_openai_compatible_chat(
            api_key=model_config.api_key,
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(system_prompt, formatted_user_prompt),
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            frequency_penalty=model_config.frequency_penalty,
            presence_penalty=model_config.presence_penalty,
            max_tokens=model_config.max_tokens,
            seed=model_config.seed,
            stop_sequences=model_config.stop_sequences,
            logprobs=model_config.logprobs,
            top_logprobs=model_config.top_logprobs,
            response_format=model_config.response_format,
            stream=model_config.stream,
            user=model_config.user,
            reasoning_effort=model_config.reasoning_effort,
            max_retries=model_config.max_retries,
            timeout=model_config.timeout,
            organization=model_config.organization,
            response_model=model_config.response_model,
            tools=model_config.tools,
            tool_choice=model_config.tool_choice,
            system_fingerprint=model_config.system_fingerprint,
            deployment_id=model_config.deployment_id,
            encoding_format=model_config.encoding_format,
            max_input_tokens=model_config.max_input_tokens,
            stop_token=model_config.stop_token,
            best_of=model_config.best_of,
            logprobs_offset=model_config.logprobs_offset,
            suffix=model_config.suffix,
            presence_penalty_range=model_config.presence_penalty_range,
            frequency_penalty_range=model_config.frequency_penalty_range,
            stop_token_id=model_config.stop_token_id,
            response_json_format=model_config.response_json_format,
            max_output_tokens=model_config.max_output_tokens,
            stream_options=model_config.stream_options,
            logprobs_type=model_config.logprobs_type,
            top_k=model_config.top_k,
            repetition_penalty=model_config.repetition_penalty,
            length_penalty=model_config.length_penalty,
            early_stopping=model_config.early_stopping,
            num_beams=model_config.num_beams,
            do_sample=model_config.do_sample,
            temperature_fallback=model_config.temperature_fallback,
            top_p_fallback=model_config.top_p_fallback,
            max_time=model_config.max_time,
            return_full_text=model_config.return_full_text,
            tokenizer_config=model_config.tokenizer_config,
            model_kwargs=model_config.model_kwargs
        )
        if response:
            try:
                analyses.append(json.loads(response))
            except json.JSONDecodeError:
                st.warning(f"Content Analysis response from {model_config.model_id} was not valid JSON: {response[:200]}...")
                analyses.append({"summary": response}) # Fallback

    for member in team.members:
        thread = threading.Thread(target=_analyze_with_model, args=(member,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if not analyses:
        return {"summary": "Failed to analyze content with any team member."}

    # Combine analyses from multiple team members using ensemble aggregation:
    # - Concatenate summaries to capture all perspectives
    # - Union of keywords, challenges, and expertise to ensure comprehensive coverage
    # - Average complexity scores for balanced assessment
    # - Majority voting for domain classification
    combined_summary = " ".join([a.get("summary", "") for a in analyses if a.get("summary")])
    combined_keywords = list(set(kw for a in analyses for kw in a.get("keywords", [])))
    combined_challenges = list(set(c for a in analyses for c in a.get("potential_challenges", [])))
    combined_expertise = list(set(e for a in analyses for e in a.get("required_expertise", [])))
    
    # Average complexity with validation, default to 5 if no valid complexities found
    complexities = [a.get("estimated_complexity", 0) for a in analyses if isinstance(a.get("estimated_complexity"), int) and 1 <= a.get("estimated_complexity") <= 10]
    avg_complexity = int(sum(complexities) / len(complexities)) if complexities else 5

    # Use majority voting for domain classification
    domains = [a.get("domain") for a in analyses if a.get("domain")]
    from collections import Counter
    most_common_domain = Counter(domains).most_common(1)[0][0] if domains else "General"

    return {
        "domain": most_common_domain,
        "keywords": combined_keywords,
        "estimated_complexity": avg_complexity,
        "potential_challenges": combined_challenges,
        "required_expertise": combined_expertise,
        "summary": combined_summary
    }

def run_ai_decomposition(problem_statement: str, analyzed_context: Dict[str, Any], team: Team) -> DecompositionPlan:
    """
    Executes Stage 1: AI-Assisted Decomposition. A Blue Team (Planners) generates a decomposition plan
    by breaking down the complex problem into manageable sub-problems. All members of the team
    contribute to generating decomposition plans, and the first valid plan is selected.

    Args:
        problem_statement (str): The original problem statement.
        analyzed_context (Dict[str, Any]): The structured context obtained from content analysis.
        team (Team): The Blue Team (role: `Planner`) responsible for generating the decomposition plan.

    Returns:
        DecompositionPlan: An object containing the AI-generated sub-problems and their configurations.
    """
    if not team.members:
        st.error(f"Decomposition Team '{team.name}' has no members. Please configure the team in the Team Manager.")
        return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=[])

    plans = []
    threads = []

    system_prompt = team.decomposition_system_prompt if team.decomposition_system_prompt else "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects."
    user_prompt_template = team.decomposition_user_prompt_template if team.decomposition_user_prompt_template else f"""Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
    - `id`: A unique identifier (e.g., "sub_1.1")
    - `description`: A clear statement of the sub-problem.
    - `dependencies`: A list of `id`s of other sub-problems this one depends on.
    - `ai_suggested_evolution_mode`: Suggested evolution mode (e.g., "standard", "adversarial", "quality_diversity").
    - `ai_suggested_complexity_score`: An integer from 1 to 10.
    - `ai_suggested_evaluation_prompt`: A specific prompt for a Gold Team to evaluate this sub-problem's solution.

    Problem Statement:
    ---
    {{problem_statement}}
    ---

    Analyzed Context:
    ---
    {{analyzed_context}}
    ---

    Provide the output as a JSON array of sub-problem objects.
    """

    def _decompose_with_model(model_config: ModelConfig):
        # Replace placeholders in the user prompt template
        formatted_user_prompt = user_prompt_template.replace("{{problem_statement}}", problem_statement)
        formatted_user_prompt = formatted_user_prompt.replace("{{analyzed_context}}", json.dumps(analyzed_context, indent=2))
        response = _request_openai_compatible_chat(
            api_key=model_config.api_key,
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(system_prompt, formatted_user_prompt),
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            frequency_penalty=model_config.frequency_penalty,
            presence_penalty=model_config.presence_penalty,
            max_tokens=model_config.max_tokens,
            seed=model_config.seed,
            stop_sequences=model_config.stop_sequences,
            logprobs=model_config.logprobs,
            top_logprobs=model_config.top_logprobs,
            response_format=model_config.response_format,
            stream=model_config.stream,
            user=model_config.user,
            reasoning_effort=model_config.reasoning_effort,
            max_retries=model_config.max_retries,
            timeout=model_config.timeout,
            organization=model_config.organization,
            response_model=model_config.response_model,
            tools=model_config.tools,
            tool_choice=model_config.tool_choice,
            system_fingerprint=model_config.system_fingerprint,
            deployment_id=model_config.deployment_id,
            encoding_format=model_config.encoding_format,
            max_input_tokens=model_config.max_input_tokens,
            stop_token=model_config.stop_token,
            best_of=model_config.best_of,
            logprobs_offset=model_config.logprobs_offset,
            suffix=model_config.suffix,
            presence_penalty_range=model_config.presence_penalty_range,
            frequency_penalty_range=model_config.frequency_penalty_range,
            stop_token_id=model_config.stop_token_id,
            response_json_format=model_config.response_json_format,
            max_output_tokens=model_config.max_output_tokens,
            stream_options=model_config.stream_options,
            logprobs_type=model_config.logprobs_type,
            top_k=model_config.top_k,
            repetition_penalty=model_config.repetition_penalty,
            length_penalty=model_config.length_penalty,
            early_stopping=model_config.early_stopping,
            num_beams=model_config.num_beams,
            do_sample=model_config.do_sample,
            temperature_fallback=model_config.temperature_fallback,
            top_p_fallback=model_config.top_p_fallback,
            max_time=model_config.max_time,
            return_full_text=model_config.return_full_text,
            tokenizer_config=model_config.tokenizer_config,
            model_kwargs=model_config.model_kwargs
        )
        if response:
            try:
                sub_problems_data = json.loads(response)
                sub_problems = [SubProblem(**sp) for sp in sub_problems_data]
                plans.append(DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=sub_problems))
            except json.JSONDecodeError:
                st.warning(f"AI Decomposition response from {model_config.model_id} was not valid JSON. Response: {response[:500]}...")

    for member in team.members:
        thread = threading.Thread(target=_decompose_with_model, args=(member,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if not plans:
        st.error("Failed to get AI decomposition plan from any team member. Please check the LLM configuration and the problem statement.")
        return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=[])
    
    return plans[0]

import statistics # Need to import this for variance calculation

def run_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any] # Additional context for LLM prompts, e.g., sub_problem details
) -> Dict[str, Any]:
    """
    Executes a Gauntlet with a given Team to critique or verify a piece of content.
    This function supports evaluation by Blue, Red, and Gold Teams, applying programmable rules
    for each round and generating detailed reports.
    
    Supports multiple gauntlet types:
    - standard: Fixed rules for all rounds
    - adaptive: Rules adapt based on content being evaluated
    - hierarchical: Multiple tiers with increasingly strict criteria
    - competitive: Multiple solutions compete against each other
    - collaborative: Models work together to improve solution

    Args:
        solution_content (str): The content (e.g., solution, critique) to be evaluated by the gauntlet.
        gauntlet_def (GauntletDefinition): The definition of the gauntlet to run, including its rules and rounds.
        team (Team): The Team (Blue, Red, or Gold) that will execute the gauntlet.
        context (Dict[str, Any]): Additional contextual information for LLM prompts, e.g., sub_problem details.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        - 'is_approved' (bool): True if the content passed the gauntlet, False otherwise.
                        - 'report_summary' (str): A summary of the gauntlet's outcome.
                        - 'critique_report' (CritiqueReport) or 'verification_report' (VerificationReport):
                          A detailed report based on the team's role.
                          The 'targeted_feedback' within these reports is expected to be a JSON array of strings
                          (sub-problem IDs) if applicable.
    """
    st.info(f"Running {gauntlet_def.gauntlet_type.upper()} Gauntlet '{gauntlet_def.name}' with Team '{team.name}'...")
    
    # Route to appropriate gauntlet type handler
    if gauntlet_def.gauntlet_type == "adaptive":
        return _run_adaptive_gauntlet(solution_content, gauntlet_def, team, context)
    elif gauntlet_def.gauntlet_type == "hierarchical":
        return _run_hierarchical_gauntlet(solution_content, gauntlet_def, team, context)
    elif gauntlet_def.gauntlet_type == "competitive":
        return _run_competitive_gauntlet(solution_content, gauntlet_def, team, context)
    elif gauntlet_def.gauntlet_type == "collaborative":
        return _run_collaborative_gauntlet(solution_content, gauntlet_def, team, context)
    else:  # standard
        return _run_standard_gauntlet(solution_content, gauntlet_def, team, context)


def _run_standard_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a standard gauntlet with fixed rules."""
    st.info(f"Running Standard Gauntlet '{gauntlet_def.name}' with Team '{team.name}'...")
    
    all_judge_reports = []
    overall_gauntlet_approved = True
    
    # Track successful rounds per judge for per-judge approval counts
    # Track successful rounds per judge for per-judge approval counts, initialized to 0 for each member.
    successful_rounds_per_judge = {member.model_id: 0 for member in team.members}

    for round_rule in gauntlet_def.rounds:
        st.subheader(f"Gauntlet: {gauntlet_def.name} - Round {round_rule.round_number}")
        round_approved_count = 0
        current_round_scores = []
        current_round_judge_reports = []

        # Prepare prompt for judges based on team role and gauntlet type
        system_prompt = ""
        user_prompt_template = ""

        # Convert sub_problem and final_solution in context to dict for JSON serialization
        serializable_context = context.copy()
        if "sub_problem" in serializable_context and dataclasses.is_dataclass(serializable_context["sub_problem"]):
            serializable_context["sub_problem"] = dataclasses.asdict(serializable_context["sub_problem"])
        if "final_solution" in serializable_context and dataclasses.is_dataclass(serializable_context["final_solution"]):
            serializable_context["final_solution"] = dataclasses.asdict(serializable_context["final_solution"])

        if team.role == "Red":
            system_prompt = "You are a Red Team AI. Your goal is to find flaws, vulnerabilities, and weaknesses in the provided content. If you find a flaw, explain it clearly. If not, state that the content appears robust. Provide your response as a JSON object with 'score' (0.0-1.0 for robustness), 'justification' (string), and 'targeted_feedback' (JSON array of strings, listing specific sub-problem IDs like ['sub_1.2', 'sub_2.1'] that are faulty)."
            user_prompt_template = f"""Critique the following content for flaws and vulnerabilities.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            Attack Modes: {', '.join(gauntlet_def.attack_modes) if gauntlet_def.attack_modes else 'General Vulnerability Scan'}
            
            Provide your critique as a JSON object with 'score', 'justification', and 'targeted_feedback'.
            If the failure is traceable to specific sub-problems, list their IDs in the 'targeted_feedback' field as a JSON array of strings, e.g., ['sub_1.2', 'sub_2.1'].
            """
        elif team.role == "Gold":
            system_prompt = "You are a Gold Team AI. Your goal is to impartially evaluate the provided content for correctness, quality, and adherence to requirements. Provide your response as a JSON object with 'score' (0.0-1.0), 'justification' (string), and 'targeted_feedback' (JSON array of strings, listing specific sub-problem IDs like ['sub_1.2', 'sub_2.1'] that are faulty)."
            user_prompt_template = f"""Evaluate the following content for correctness and quality.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            Evaluation Prompt: {context.get('evaluation_prompt', 'Evaluate for overall quality and correctness.')}
            
            Provide your evaluation as a JSON object with 'score', 'justification', and 'targeted_feedback'.
            If the evaluation fails and you can trace it to specific sub-problems, list their IDs in the 'targeted_feedback' field as a JSON array of strings, e.g., ['sub_1.2', 'sub_2.1'].
            """
        elif team.role == "Blue": # For Blue Team Gauntlets (e.g., internal quality check, peer review)
            system_prompt = team.gold_team_system_prompt if team.gold_team_system_prompt else "You are a Blue Team AI acting as an internal quality assurance or peer reviewer. Your goal is to critically evaluate the provided content for its quality, correctness, and adherence to specified criteria. Provide your response as a JSON object with 'score' (0.0-1.0 for quality), 'justification', and 'targeted_feedback' (JSON array of strings, listing specific sub-problem IDs like ['sub_1.2', 'sub_2.1'] that are faulty)."
            user_prompt_template = team.gold_team_user_prompt_template if team.gold_team_user_prompt_template else f"""Evaluate the following content. This content was generated internally by a Blue Team for a sub-problem.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            
            Based on your evaluation, provide a JSON object with a 'score' (0.0-1.0) for the content's quality and a 'justification' for your score.
            If the evaluation fails and you can trace it to specific sub-problems, list their IDs in the 'targeted_feedback' field as a JSON array of strings, e.g., ['sub_1.2', 'sub_2.1'].
            """

        # If collaboration mode is enabled, judges in later rounds see previous feedback
        # This logic aggregates feedback from the *previous* round to inform the current round's judges.
        if round_rule.collaboration_mode == "share_previous_feedback" and all_judge_reports:
            # Filter for reports from the immediately preceding round
            previous_round_reports = [r for r in all_judge_reports if r.get('round_number') == round_rule.round_number - 1]
            if previous_round_reports:
                previous_feedback = "\n".join([f"Model {r['model_id']}: {r['justification']} (Score: {r['score']})" for r in previous_round_reports])
                user_prompt_template += f"\n\nPrevious round's feedback:\n---\n{previous_feedback}\n---"

        # Invoke each member of the team in parallel using threading to speed up evaluation.
        member_results = []
        threads = []

        def _evaluate_member(member_idx, member, system_prompt, user_prompt_template, solution_content, min_score_for_judge):
            messages = _compose_messages(system_prompt, user_prompt_template.replace("{content}", solution_content))
            response_content = _request_openai_compatible_chat(
                api_key=member.api_key,
                base_url=member.api_base,
                model=member.model_id,
                messages=_compose_messages(system_prompt, user_prompt_template.replace("{content}", solution_content)),
                temperature=member.temperature,
                top_p=member.top_p,
                frequency_penalty=member.frequency_penalty,
                presence_penalty=member.presence_penalty,
                max_tokens=member.max_tokens,
                seed=member.seed,
                n=member.n,
                logit_bias=member.logit_bias,
                reasoning_effort=member.reasoning_effort,
                stop_sequences=member.stop_sequences,
                logprobs=member.logprobs,
                top_logprobs=member.top_logprobs,
                response_format=member.response_format,
                stream=member.stream,
                user=member.user,
                max_retries=member.max_retries,
                timeout=member.timeout,
                organization=member.organization,
                response_model=member.response_model,
                tools=member.tools,
                tool_choice=member.tool_choice,
                system_fingerprint=member.system_fingerprint,
                deployment_id=member.deployment_id,
                encoding_format=member.encoding_format,
                max_input_tokens=member.max_input_tokens,
                stop_token=member.stop_token,
                best_of=member.best_of,
                logprobs_offset=member.logprobs_offset,
                suffix=member.suffix,
                presence_penalty_range=member.presence_penalty_range,
                frequency_penalty_range=member.frequency_penalty_range,
                stop_token_id=member.stop_token_id,
                response_json_format=member.response_json_format,
                max_output_tokens=member.max_output_tokens,
                stream_options=member.stream_options,
                logprobs_type=member.logprobs_type,
                top_k=member.top_k,
                repetition_penalty=member.repetition_penalty,
                length_penalty=member.length_penalty,
                early_stopping=member.early_stopping,
                num_beams=member.num_beams,
                do_sample=member.do_sample,
                temperature_fallback=member.temperature_fallback,
                top_p_fallback=member.top_p_fallback,
                max_time=member.max_time,
                return_full_text=member.return_full_text,
                tokenizer_config=member.tokenizer_config,
                model_kwargs=member.model_kwargs
            )

            judge_score = 0.0
            justification = "No response or invalid format."
            targeted_feedback = ""
            
            if response_content:
                try:
                    parsed_response = json.loads(response_content)
                    judge_score = parsed_response.get("score", 0.0)
                    justification = parsed_response.get("justification", response_content)
                    targeted_feedback = parsed_response.get("targeted_feedback", "")
                    # Validate score range: ensure score is between 0.0 and 1.0
                    if not (0.0 <= judge_score <= 1.0):
                        st.warning(f"LLM {member.model_id} returned an out-of-range score: {judge_score}. Clamping to 0.0-1.0.")
                        judge_score = max(0.0, min(1.0, judge_score))
                except json.JSONDecodeError:
                    st.warning(f"LLM {member.model_id} did not return valid JSON. Attempting regex fallback for score. Response: {response_content[:200]}...")
                    # Regex to find a score in the response if JSON parsing fails
                    score_match = re.search(r"score:\s*(\d+\.?\d*)", response_content, re.IGNORECASE)
                    if score_match:
                        judge_score = float(score_match.group(1))
                        if judge_score > 1.0: judge_score /= 100.0 # Assume percentage if > 1.0 (e.g., 90 instead of 0.9)
                        # Validate score range after regex
                        if not (0.0 <= judge_score <= 1.0):
                            st.warning(f"LLM {member.model_id} returned an out-of-range score via regex: {judge_score}. Clamping to 0.0-1.0.")
                            judge_score = max(0.0, min(1.0, judge_score))
                    justification = response_content
                    targeted_feedback = "" # Cannot reliably extract targeted feedback without JSON
            
            judge_passed_this_round = False
            # Determine if the judge passed this round based on their score against the minimum required score.
            # For Red Team, approval means the solution is robust enough (score >= min_score_for_judge).
            # For Gold/Blue Team, approval means the solution meets quality/correctness criteria (score >= min_score_for_judge).
            if judge_score >= min_score_for_judge:
                judge_passed_this_round = True
            
            member_results.append({
                "member_idx": member_idx,
                "member": member,
                "judge_score": judge_score,
                "justification": justification,
                "targeted_feedback": targeted_feedback,
                "judge_passed_this_round": judge_passed_this_round
            })

        for member_idx, member in enumerate(team.members):
            # Determine the minimum score required for this specific judge in this round.
            # It prioritizes per-judge requirements if specified, otherwise falls back to the round's overall minimum confidence.
            per_judge_req = round_rule.per_judge_requirements.get(member.model_id, {})
            min_score_for_judge = per_judge_req.get('min_score', round_rule.min_overall_confidence)

            thread = threading.Thread(target=_evaluate_member, args=(member_idx, member, system_prompt, user_prompt_template, solution_content, min_score_for_judge))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Process results from parallel evaluations
        # Sort results by member_idx to maintain original order if needed, though not strictly necessary for correctness.
        member_results.sort(key=lambda x: x["member_idx"])

        for result in member_results:
            member = result["member"]
            judge_score = result["judge_score"]
            justification = result["justification"]
            targeted_feedback = result["targeted_feedback"]
            judge_passed_this_round = result["judge_passed_this_round"]
            min_score_for_judge = round_rule.per_judge_requirements.get(member.model_id, {}).get('min_score', round_rule.min_overall_confidence)

            st.write(f"    - {member.model_id} Score: {judge_score:.2f} (Required: {min_score_for_judge:.2f})")
            st.caption(f"      Justification: {justification[:100]}...")
            
            if judge_passed_this_round:
                round_approved_count += 1
                successful_rounds_per_judge[member.model_id] += 1
            
            current_round_scores.append(judge_score)
            current_round_judge_reports.append({
                "model_id": member.model_id,
                "score": judge_score,
                "justification": justification,
                "targeted_feedback": targeted_feedback,
                "passed_round": judge_passed_this_round,
                "round_number": round_rule.round_number # Add round number to report
            })
        
        # --- Evaluate Round Success ---
        round_passed = True
        
        # 1. Check Quorum: Ensure enough judges approved the content in this round.
        if round_rule.quorum_required_approvals > round_approved_count:
            st.warning(f"  - Round {round_rule.round_number} failed: Quorum not met ({round_approved_count}/{round_rule.quorum_required_approvals} approvals).")
            round_passed = False
        
        # 2. Check Minimum Overall Confidence: Ensure the average score across all judges meets the threshold.
        if current_round_scores and statistics.mean(current_round_scores) < round_rule.min_overall_confidence:
            st.warning(f"  - Round {round_rule.round_number} failed: Average score ({statistics.mean(current_round_scores):.2f}) below minimum overall confidence ({round_rule.min_overall_confidence:.2f}).")
            round_passed = False
            
        # 3. Check Max Score Variance: Ensure judges have sufficient consensus (scores are not too spread out).
        if round_rule.max_score_variance is not None and len(current_round_scores) > 1:
            current_variance = statistics.variance(current_round_scores)
            if current_variance > round_rule.max_score_variance:
                st.warning(f"  - Round {round_rule.round_number} failed: Score variance ({current_variance:.2f}) above maximum allowed ({round_rule.max_score_variance:.2f}).")
                round_passed = False
        
        # Always collect judge reports for the current round, regardless of pass/fail.
        all_judge_reports.extend(current_round_judge_reports)

        if not round_passed:
            overall_gauntlet_approved = False
            break # Gauntlet failed, no need to continue to next rounds
        else:
            st.success(f"  - Round {round_rule.round_number} passed.")

    # --- Final Gauntlet Approval Check (Per-Judge Approval Counts) ---
    # After all rounds, perform a final check based on per-judge requirements across all rounds.
    if overall_gauntlet_approved:
        for member in team.members:
            # Aggregate required_successful_rounds from all round_rules for this member.
            # A judge must meet the highest `required_successful_rounds` specified for them across any round.
            required_successful_rounds_for_member = 0
            for round_rule in gauntlet_def.rounds:
                per_judge_req = round_rule.per_judge_requirements.get(member.model_id, {})
                if 'required_successful_rounds' in per_judge_req:
                    required_successful_rounds_for_member = max(required_successful_rounds_for_member, per_judge_req['required_successful_rounds'])
            
            # If a judge has a specific requirement for successful rounds, check if it was met.
            if required_successful_rounds_for_member > 0 and successful_rounds_per_judge[member.model_id] < required_successful_rounds_for_member:
                st.warning(f"Gauntlet '{gauntlet_def.name}' failed: Model {member.model_id} did not meet its required successful rounds ({successful_rounds_per_judge[member.model_id]}/{required_successful_rounds_for_member}).")
                overall_gauntlet_approved = False
                break

    report_summary = f"Gauntlet '{gauntlet_def.name}' {'APPROVED' if overall_gauntlet_approved else 'REJECTED'} by Team '{team.name}'."
    st.markdown(f"**{report_summary}**")

    # Return appropriate report type
    if team.role == "Red":
        return {
            "is_approved": overall_gauntlet_approved,
            "report_summary": report_summary,
            "critique_report": CritiqueReport(
                solution_attempt_id=context.get('solution_id', 'unknown'),
                gauntlet_name=gauntlet_def.name,
                is_approved=overall_gauntlet_approved,
                reports_by_judge=all_judge_reports,
                summary=report_summary
            )
        }
    else: # Gold or Blue
        return {
            "is_approved": overall_gauntlet_approved,
            "report_summary": report_summary,
            "verification_report": VerificationReport(
                solution_attempt_id=context.get('solution_id', 'unknown'),
                gauntlet_name=gauntlet_def.name,
                is_approved=overall_gauntlet_approved,
                reports_by_judge=all_judge_reports,
                average_score=statistics.mean(current_round_scores) if current_round_scores else 0.0,
                score_variance=statistics.variance(current_round_scores) if len(current_round_scores) > 1 else 0.0,
                summary=report_summary
            )
        }


# Main orchestrator function (will be expanded significantly)
def run_sovereign_workflow(
    workflow_state: WorkflowState,
    content_analyzer_team: Team,
    planner_team: Team,
    solver_team: Team,
    patcher_team: Team, # New: for fixing rejected solutions
    assembler_team: Team,
    
    # Gauntlets for sub-problems
    sub_problem_red_gauntlet: GauntletDefinition,
    sub_problem_gold_gauntlet: GauntletDefinition,
    
    # Gauntlets for final solution
    final_red_gauntlet: GauntletDefinition,
    final_gold_gauntlet: GauntletDefinition,
    
    solver_generation_gauntlet: GauntletDefinition,
    max_refinement_loops: int = 3
):
    """
    Orchestrates the end-to-end Sovereign-Grade Decomposition Workflow.

    This function manages the state transitions between different stages of the workflow,
    invoking appropriate teams and gauntlets for content analysis, decomposition,
    sub-problem solving, reassembly, and final verification. It also implements
    the self-healing loop for refinement.

    Args:
        workflow_state: The current state object of the workflow.
        content_analyzer_team: The Blue Team responsible for initial content analysis.
        planner_team: The Blue Team responsible for generating the decomposition plan.
        solver_team: The Blue Team responsible for generating solutions for sub-problems.
        patcher_team: The Blue Team responsible for fixing rejected solutions.
        assembler_team: The Blue Team responsible for reassembling the final solution.
        sub_problem_red_gauntlet: The Red Team Gauntlet for critiquing sub-problem solutions.
        sub_problem_gold_gauntlet: The Gold Team Gauntlet for verifying sub-problem solutions.
        final_red_gauntlet: The Red Team Gauntlet for critiquing the final assembled solution.
        final_gold_gauntlet: The Gold Team Gauntlet for verifying the final assembled solution.
        solver_generation_gauntlet: The Blue Team Gauntlet used by the solver/patcher for internal generation/peer review.
        max_refinement_loops: The maximum number of self-healing loops allowed for the final solution.
    """
    st.info(f"Starting Sovereign-Grade Workflow: {workflow_state.workflow_id}")
    workflow_state.status = "running"
    
    # Initial validation: Ensure all required teams and gauntlets are provided and valid.
    if not all([content_analyzer_team, planner_team, solver_team, patcher_team, assembler_team,
                sub_problem_red_gauntlet, sub_problem_gold_gauntlet, final_red_gauntlet, final_gold_gauntlet,
                solver_generation_gauntlet]):
        st.error("One or more required teams or gauntlets are missing or invalid. Workflow cannot proceed.")
        workflow_state.status = "failed"
        return

    # --- Stage 0: Content Analysis ---
    # The workflow starts here, or returns here if re-initialized.
    if workflow_state.current_stage == "INITIALIZING" or workflow_state.current_stage == "Content Analysis":
        workflow_state.current_stage = "Content Analysis"
        st.info(f"[{workflow_state.current_stage}] Analyzing problem statement...")
        analyzed_context = run_content_analysis(workflow_state.problem_statement, content_analyzer_team)
        # Store the analyzed context and initial plan structure in the workflow state.
        workflow_state.decomposition_plan = DecompositionPlan(
            problem_statement=workflow_state.problem_statement,
            analyzed_context=analyzed_context,
            sub_problems=[], # Will be filled in next stage
            max_refinement_loops=max_refinement_loops,
            assembler_team_name=assembler_team.name,
            final_red_team_gauntlet_name=final_red_gauntlet.name,
            final_gold_team_gauntlet_name=final_gold_gauntlet.name
        )
        st.success(f"[{workflow_state.current_stage}] Analysis complete.")
        workflow_state.current_stage = "AI-Assisted Decomposition" # Transition to the next stage.
        workflow_state.progress = 0.2 # Update overall progress.

    # --- Stage 1: AI-Assisted Decomposition ---
    # AI breaks down the problem into sub-problems.
    if workflow_state.current_stage == "AI-Assisted Decomposition":
        st.info(f"[{workflow_state.current_stage}] Generating decomposition plan...")
        decomposition_plan = run_ai_decomposition(
            workflow_state.problem_statement,
            workflow_state.decomposition_plan.analyzed_context,
            planner_team
        )
        workflow_state.decomposition_plan.sub_problems = decomposition_plan.sub_problems
        st.success(f"[{workflow_state.current_stage}] Decomposition plan generated.")
        workflow_state.current_stage = "Manual Review & Override" # Transition to human-in-the-loop stage.
        workflow_state.progress = 0.4 # Update overall progress.

    # --- Stage 2: Manual Review & Override ---
    # This stage is a human-in-the-loop step where the user reviews and potentially modifies
    # the AI-generated decomposition plan. The workflow pauses here awaiting user input.
    if workflow_state.current_stage == "Manual Review & Override":
        st.info("Awaiting user review and approval of the decomposition plan in the UI.")
        workflow_state.status = "awaiting_user_input"
        
        # Render the manual review panel and wait for user action
        # The render_manual_review_panel function will return "approved", "rejected", or "pending"
        # along with the (potentially modified) decomposition plan.
        review_status, approved_plan = render_manual_review_panel(workflow_state.decomposition_plan)

        if review_status == "approved":
            workflow_state.decomposition_plan = approved_plan
            st.success("[Manual Review & Override] Decomposition plan approved by user.")
            workflow_state.current_stage = "Sub-Problem Solving Loop" # Transition to the next stage.
            workflow_state.status = "running" # Resume workflow execution.
            workflow_state.progress = 0.5 # Update overall progress.
            st.rerun() # Rerun to continue the workflow immediately.
        elif review_status == "rejected":
            st.error("[Manual Review & Override] Decomposition plan rejected by user. Workflow terminated.")
            workflow_state.status = "failed"
            return # Terminate workflow.
        else: # review_status == "pending"
            # If the plan is still pending review, we need to stop execution here
            # and wait for the next Streamlit rerun triggered by user interaction in the UI.
            return

    # --- Stage 3: Sub-Problem Solving Loop ---
    # Iteratively generates, critiques, and verifies solutions for each sub-problem,
    # respecting dependencies and applying self-healing mechanisms.
    if workflow_state.current_stage == "Sub-Problem Solving Loop":
        st.info(f"[{workflow_state.current_stage}] Starting sub-problem solving...")
        
        if not workflow_state.decomposition_plan or not workflow_state.decomposition_plan.sub_problems:
            st.error("Decomposition plan is missing or empty. Cannot proceed with sub-problem solving. Workflow failed.")
            workflow_state.status = "failed"
            return

        sub_problems_by_id = {sp.id: sp for sp in workflow_state.decomposition_plan.sub_problems}

        # Initialize data structures for topological sort:
        # `in_degree` tracks the number of unmet dependencies for each sub-problem.
        # `adj` stores a list of sub-problems that depend on a given sub-problem.
        in_degree = {sp_id: 0 for sp_id in sub_problems_by_id}
        adj = {sp_id: [] for sp_id in sub_problems_by_id}
        
        # Populate `in_degree` and `adj` based on sub-problem dependencies.
        for sp_id, sp in sub_problems_by_id.items():
            for dep_id in sp.dependencies:
                if dep_id in sub_problems_by_id:
                    adj[dep_id].append(sp_id)
                    in_degree[sp_id] += 1
                else:
                    st.error(f"Sub-problem '{sp_id}' has an invalid dependency: '{dep_id}'. Workflow failed.")
                    workflow_state.status = "failed"
                    return
        
        # Initialize the queue with sub-problems that have no unmet dependencies (in-degree of 0).
        # Only add sub-problems that have not been solved yet to avoid re-processing.
        queue = [sp_id for sp_id, degree in in_degree.items() if degree == 0 and sp_id not in workflow_state.solved_sub_problem_ids]
        
        # Add any previously rejected sub-problems back to the queue for re-processing.
        # This ensures that sub-problems flagged for rework are re-evaluated once their dependencies are met.
        for rejected_sp_id in workflow_state.rejected_sub_problems.keys():
            if rejected_sp_id not in queue and rejected_sp_id not in workflow_state.solved_sub_problem_ids:
                # Ensure all dependencies for the rejected sub-problem are met before re-adding to queue.
                if in_degree[rejected_sp_id] == 0:
                    queue.append(rejected_sp_id)
                else:
                    st.warning(f"Rejected sub-problem {rejected_sp_id} has unmet dependencies. Will be re-added when dependencies are met.")

        # Check for initial unsolvable state (e.g., circular dependencies or no starting points).
        if not queue and len(workflow_state.solved_sub_problem_ids) < len(workflow_state.decomposition_plan.sub_problems):
            st.error("Circular dependency detected or no solvable sub-problems initially. Workflow failed.")
            workflow_state.status = "failed"
            return

        # Process sub-problems in topological order (i.e., only after all their dependencies are met).
        processed_this_iteration = set() # Initialize set to track sub-problems processed in this iteration
        while queue:
            current_sp_id = queue.pop(0) # Get the next solvable sub-problem from the queue.
            current_sub_problem = sub_problems_by_id.get(current_sp_id)
            
            if not current_sub_problem:
                st.error(f"Sub-problem {current_sp_id} not found in decomposition plan. Skipping.")
                continue

            if current_sp_id in workflow_state.solved_sub_problem_ids:
                continue # Skip if already solved (e.g., if re-added to queue but solved in a previous iteration).

            workflow_state.current_sub_problem_id = current_sp_id
            st.info(f"[{workflow_state.current_stage}] Solving sub-problem: {current_sp_id} - {current_sub_problem.description[:50]}...")
            
            generated_content = ""
            # If a solution for this sub-problem already exists (e.g., from a previous refinement loop), use it as a base.
            if current_sp_id in workflow_state.sub_problem_solutions:
                generated_content = workflow_state.sub_problem_solutions[current_sp_id].content

            # Determine the actual solver_generation_gauntlet for this sub-problem.
            # It can be specified per sub-problem or fall back to the global one.
            actual_solver_generation_gauntlet = None
            if current_sub_problem.solver_generation_gauntlet_name:
                actual_solver_generation_gauntlet = gauntlet_manager.get_gauntlet(current_sub_problem.solver_generation_gauntlet_name)
            else:
                actual_solver_generation_gauntlet = solver_generation_gauntlet # Fallback to global if not specified for sub-problem

            # If a solution exists and was rejected, use the Patcher Team to fix it.
            if current_sp_id in workflow_state.rejected_sub_problems:
                st.info(f"  - Invoking Patcher Team for {current_sp_id} based on previous rejection.")
                last_report = workflow_state.rejected_sub_problems[current_sp_id]
                generated_content = generate_solution_for_sub_problem(
                    sub_problem=current_sub_problem,
                    team=patcher_team,
                    context={"current_solution": generated_content, "feedback_report": last_report},
                    workflow_state=workflow_state,
                    solver_generation_gauntlet=actual_solver_generation_gauntlet,
                )
                del workflow_state.rejected_sub_problems[current_sp_id] # Clear rejection status after attempting to patch.
            else:
                # Otherwise, use the Solver Team to generate a new solution.
                actual_solver_team = team_manager.get_team(current_sub_problem.solver_team_name) if current_sub_problem.solver_team_name else solver_team
                generated_content = generate_solution_for_sub_problem(
                    sub_problem=current_sub_problem,
                    team=actual_solver_team,
                    context={"current_solution": generated_content},
                    workflow_state=workflow_state,
                    solver_generation_gauntlet=actual_solver_generation_gauntlet,
                )
            
            if generated_content.startswith("Failed to generate solution:"):
                st.error(f"Failed to generate solution for sub-problem {current_sp_id}. Workflow failed.")
                workflow_state.status = "failed"
                return # Halt workflow execution if solution generation fails.

            solution_attempt = SolutionAttempt(
                sub_problem_id=current_sp_id,
                content=generated_content,
                generated_by_model=solver_team.members[0].model_id, # Assuming first member of the solver team generated it.
                timestamp=time.time()
            )
            
            # --- Step B: Red Team Gauntlet (Critique) ---
            # Determine the actual red gauntlet for this sub-problem (can be specified per sub-problem or global).
            actual_red_gauntlet = None
            if current_sub_problem.red_team_gauntlet_name:
                actual_red_gauntlet = gauntlet_manager.get_gauntlet(current_sub_problem.red_team_gauntlet_name)
            else:
                actual_red_gauntlet = sub_problem_red_gauntlet # Fallback to global if not specified for sub-problem

            if actual_red_gauntlet:
                workflow_state.current_gauntlet_name = actual_red_gauntlet.name
                st.info(f"  - Running Red Team Gauntlet for {current_sp_id}...")
                red_gauntlet_result = run_gauntlet(
                    solution_attempt.content,
                    actual_red_gauntlet,
                    team_manager.get_team(actual_red_gauntlet.team_name), # Use the assigned Red Team.
                    {"sub_problem": current_sub_problem, "solution_id": solution_attempt.sub_problem_id}
                )
                workflow_state.all_critique_reports.append(red_gauntlet_result['critique_report'])
                st.info("INFO: Red team gauntlet finished.")
                
                if not red_gauntlet_result['is_approved']:
                    st.warning(f"  - Red Team rejected solution for {current_sp_id}. Marking for rework.")
                    workflow_state.rejected_sub_problems[current_sp_id] = red_gauntlet_result['critique_report']
                    # Re-add to queue to be re-processed after patching in a subsequent iteration.
                    queue.append(current_sp_id) 
                    continue # Skip Gold Team and next dependencies for this sub-problem; it needs rework.
            else:
                st.info(f"  - No Red Team Gauntlet configured for {current_sp_id}. Skipping Red Team evaluation.")

            st.info("INFO: About to run gold team gauntlet.")
            # --- Step C: Gold Team Gauntlet (Verification) ---
            # Determine the actual gold gauntlet for this sub-problem (can be specified per sub-problem or global).
            actual_gold_gauntlet = None
            if current_sub_problem.gold_team_gauntlet_name:
                actual_gold_gauntlet = gauntlet_manager.get_gauntlet(current_sub_problem.gold_team_gauntlet_name)
            else:
                actual_gold_gauntlet = sub_problem_gold_gauntlet # Fallback to global if not specified for sub-problem

            if actual_gold_gauntlet:
                workflow_state.current_gauntlet_name = actual_gold_gauntlet.name
                st.info(f"  - Running Gold Team Gauntlet for {current_sp_id}...")
                gold_gauntlet_result = run_gauntlet(
                    solution_attempt.content,
                    actual_gold_gauntlet,
                    team_manager.get_team(actual_gold_gauntlet.team_name), # Use the assigned Gold Team.
                    {"sub_problem": current_sub_problem, "solution_id": solution_attempt.sub_problem_id}
                )
                workflow_state.all_verification_reports.append(gold_gauntlet_result['verification_report'])

                if not gold_gauntlet_result['is_approved']:
                    st.warning(f"  - Gold Team rejected solution for {current_sp_id}. Marking for rework.")
                    workflow_state.rejected_sub_problems[current_sp_id] = gold_gauntlet_result['verification_report']
                    # Re-add to queue to be re-processed after patching in a subsequent iteration.
                    queue.append(current_sp_id) 
                    continue # Skip next dependencies for this sub-problem; it needs rework.
            else:
                st.info(f"  - No Gold Team Gauntlet configured for {current_sp_id}. Skipping Gold Team evaluation.")
            
            # If both Red and Gold Gauntlets pass (or are skipped), the sub-problem is considered solved.
            workflow_state.sub_problem_solutions[current_sp_id] = solution_attempt
            workflow_state.solved_sub_problem_ids.add(current_sp_id)
            processed_this_iteration.add(current_sp_id)
            st.success(f"[{workflow_state.current_stage}] Sub-problem {current_sp_id} solved and verified.")

            # Update overall progress based on solved sub-problems.
            # Stage 3 (Sub-Problem Solving) accounts for 30% of total progress (0.4 to 0.7).
            workflow_state.progress = 0.4 + (0.3 * (len(workflow_state.solved_sub_problem_ids) / len(workflow_state.decomposition_plan.sub_problems))) 

            # Update in-degrees of dependent sub-problems.
            # Decrement in-degree for all sub-problems that depend on the currently solved one.
            # If an in-degree becomes 0, it means all its dependencies are met, so add it to the queue.
            for dependent_sp_id in adj[current_sp_id]:
                in_degree[dependent_sp_id] -= 1
                if in_degree[dependent_sp_id] == 0 and dependent_sp_id not in workflow_state.solved_sub_problem_ids:
                    queue.append(dependent_sp_id)
            
        # After the queue is empty, check if all sub-problems were solved.
        if len(workflow_state.solved_sub_problem_ids) < len(workflow_state.decomposition_plan.sub_problems):
            st.error("Could not solve all sub-problems. Possible circular dependency or unsolvable problem. Workflow failed.")
            workflow_state.status = "failed"
            return

        st.success(f"[{workflow_state.current_stage}] All sub-problems solved.")
        workflow_state.current_stage = "Configurable Reassembly" # Move to next stage
        workflow_state.progress = 0.7 # Update progress after Stage 3

    # --- Stage 4: Configurable Reassembly ---
    # Integrates all individually verified sub-problem solutions into a single, coherent final product.
    if workflow_state.current_stage == "Configurable Reassembly":
        st.info(f"[{workflow_state.current_stage}] Reassembling final solution using {assembler_team.name} via OpenEvolve...")
        
        # Collect all verified sub-problem solutions to provide as context for the assembler team.
        verified_solutions_content = []
        for sp in workflow_state.decomposition_plan.sub_problems:
            if sp.id in workflow_state.sub_problem_solutions:
                verified_solutions_content.append(f"### Sub-Problem {sp.id}\n{{workflow_state.sub_problem_solutions[sp.id].content}}")
        
        combined_solutions_input = "\n\n".join(verified_solutions_content)

        if not assembler_team.members:
            st.error(f"Assembler Team '{assembler_team.name}' has no members. Please configure the team in the Team Manager.")
            workflow_state.status = "failed"
            return

        model_config = assembler_team.members[0] # Use the first model in the team for reassembly.

        # Construct prompt for the Assembler Team to guide the reassembly process.
        assembler_system_message = assembler_team.assembler_system_prompt if assembler_team.assembler_system_prompt else f"You are an expert Assembler AI. Your task is to integrate multiple verified sub-problem solutions into a single, coherent, and high-quality final product. Ensure all dependencies are respected and the final output addresses the original problem statement."
        assembler_user_message_template = assembler_team.assembler_user_prompt_template if assembler_team.assembler_user_prompt_template else f"""Integrate the following verified sub-problem solutions into a single, coherent final product. The original problem statement was: {{problem_statement}}

        Verified Sub-Problem Solutions:
        ---
        {{combined_solutions_input}}
        ---

        Provide the complete, integrated final solution.
        """
        
        assembler_user_message = assembler_user_message_template.replace("{{problem_statement}}", workflow_state.problem_statement)
        assembler_user_message = assembler_user_message.replace("{{combined_solutions_input}}", combined_solutions_input)

        # Construct arguments for OpenEvolve reassembly, leveraging its unified evolution capabilities.
        evolution_args = {
            "content": assembler_user_message,
            "content_type": "text_general",
            "evolution_mode": "standard", # Reassembly is typically a standard generation task.
            "model_configs": [{"name": model_config.model_id, "weight": 1.0}],
            "api_key": model_config.api_key,
            "api_base": model_config.api_base,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "max_tokens": model_config.max_tokens,
            "frequency_penalty": model_config.frequency_penalty,
            "presence_penalty": model_config.presence_penalty,
            "seed": model_config.seed,
            "stop_sequences": model_config.stop_sequences,
            "logprobs": model_config.logprobs,
            "top_logprobs": model_config.top_logprobs,
            "response_format": model_config.response_format,
            "stream": model_config.stream,
            "user": model_config.user,
            "system_message": assembler_system_message,
        }

        try:
            result = run_unified_evolution(**evolution_args)
            if result and result.get("success") and result.get("best_solution"): 
                final_solution_content = result["best_solution"]
            else:
                st.error(f"OpenEvolve failed to reassemble the final solution. Result: {result}")
                workflow_state.status = "failed"
                return
        except Exception as e:
            st.error(f"Error running OpenEvolve for reassembly: {e}")
            workflow_state.status = "failed"
            return
        
        # Store the final assembled solution attempt in the workflow state.
        final_solution_attempt = SolutionAttempt(
            sub_problem_id="final_solution",
            content=final_solution_content,
            generated_by_model=assembler_team.members[0].model_id,
            timestamp=time.time()
        )
        workflow_state.final_solution = final_solution_attempt
        st.success(f"[{workflow_state.current_stage}] Final solution reassembled.")
        workflow_state.current_stage = "Final Verification & Self-Healing Loop" # Transition to final verification.
        workflow_state.progress = 0.9 # Update overall progress.


    # --- Stage 5: Final Verification & Self-Healing Loop ---
    # Rigorously verifies the final assembled solution and, if necessary, triggers targeted self-correction.
    if workflow_state.current_stage == "Final Verification & Self-Healing Loop":
        st.info(f"[{workflow_state.current_stage}] Starting final verification...")
        
        # The self-healing loop continues until the solution is approved or max refinement loops are reached.
        while workflow_state.refinement_loop_count <= workflow_state.max_refinement_loops:
            st.subheader(f"Final Verification Loop: {workflow_state.refinement_loop_count + 1}/{workflow_state.max_refinement_loops + 1}")
            
            # Final Red Team Gauntlet: Critiques the assembled solution for integration errors or new vulnerabilities.
            workflow_state.current_gauntlet_name = final_red_gauntlet.name
            st.info(f"  - Running Final Red Team Gauntlet...")
            final_red_gauntlet_result = run_gauntlet(
                workflow_state.final_solution.content,
                final_red_gauntlet,
                team_manager.get_team(final_red_gauntlet.team_name), # Use the assigned Red Team
                {"final_solution": workflow_state.final_solution} # Provide the final solution as context.
            )
            workflow_state.all_critique_reports.append(final_red_gauntlet_result['critique_report'])

            if not final_red_gauntlet_result['is_approved']:
                st.warning(f"  - Final Red Team rejected solution. Initiating self-healing.")
                # Parse feedback to identify specific sub-problems that caused the failure.
                problematic_sub_problem_ids = parse_targeted_feedback(final_red_gauntlet_result['critique_report'])
                if not problematic_sub_problem_ids:
                    st.error("  - Red Team rejected, but no specific problematic sub-problems identified. Cannot self-heal. Please review the Red Team's LLM output or prompt for actionable feedback.")
                    workflow_state.status = "failed"
                    return

                st.info(f"  - Problematic sub-problems identified: {', '.join(problematic_sub_problem_ids)}. Re-queuing for re-solve.")
                # Clear solutions for problematic sub-problems to force re-solve in Stage 3.
                for sp_id in problematic_sub_problem_ids:
                    if sp_id in workflow_state.sub_problem_solutions:
                        del workflow_state.sub_problem_solutions[sp_id]
                        workflow_state.rejected_sub_problems[sp_id] = final_red_gauntlet_result['critique_report'] # Store report for patcher to use.
                
                workflow_state.refinement_loop_count += 1
                # Check if max refinement loops have been reached.
                if workflow_state.refinement_loop_count > workflow_state.max_refinement_loops:
                    st.error("Max refinement loops reached for final solution. Manual intervention required.")
                    workflow_state.status = "failed"
                    return
                
                workflow_state.current_stage = "Sub-Problem Solving Loop" # Go back to solve problematic sub-problems.
                return # Exit current run, Streamlit will re-run and and continue from Stage 3.

            # Final Gold Team Gauntlet: Holistically evaluates the assembled solution against original requirements.
            workflow_state.current_gauntlet_name = final_gold_gauntlet.name
            st.info(f"  - Running Final Gold Team Gauntlet...")
            final_gold_gauntlet_result = run_gauntlet(
                workflow_state.final_solution.content,
                final_gold_gauntlet,
                team_manager.get_team(final_gold_gauntlet.team_name), # Use the assigned Gold Team
                {"final_solution": workflow_state.final_solution} # Provide the final solution as context.
            )
            workflow_state.all_verification_reports.append(final_gold_gauntlet_result['verification_report'])

            if not final_gold_gauntlet_result['is_approved']:
                st.warning(f"  - Final Gold Team rejected solution. Initiating self-healing.")
                # Parse feedback to identify specific sub-problems that caused the failure.
                problematic_sub_problem_ids = parse_targeted_feedback(final_gold_gauntlet_result['verification_report'])
                if not problematic_sub_problem_ids:
                    st.error("  - Gold Team rejected, but no specific problematic sub-problems identified. Cannot self-heal. Please review the Gold Team's LLM output or prompt for actionable feedback.")
                    workflow_state.status = "failed"
                    return

                st.info(f"  - Problematic sub-problems identified: {', '.join(problematic_sub_problem_ids)}. Re-queuing for re-solve.")
                # Clear solutions for problematic sub-problems to force re-solve in Stage 3.
                for sp_id in problematic_sub_problem_ids:
                    if sp_id in workflow_state.sub_problem_solutions:
                        del workflow_state.sub_problem_solutions[sp_id]
                        workflow_state.rejected_sub_problems[sp_id] = final_gold_gauntlet_result['verification_report'] # Store report for patcher to use.
                
                workflow_state.refinement_loop_count += 1
                # Check if max refinement loops have been reached.
                if workflow_state.refinement_loop_count > workflow_state.max_refinement_loops:
                    st.error("Max refinement loops reached for final solution. Manual intervention required.")
                    workflow_state.status = "failed"
                    return
                
                workflow_state.current_stage = "Sub-Problem Solving Loop" # Go back to solve problematic sub-problems.
                return # Exit current run, Streamlit will re-run and continue from Stage 3.
            
            # If both final gauntlets pass, the workflow is completed successfully.
            st.success(f"[{workflow_state.current_stage}] Final solution verified. Workflow completed successfully!")
            workflow_state.status = "completed"
            workflow_state.end_time = time.time()
            workflow_state.progress = 1.0
            
            # --- Stage 6: Knowledge Extraction & Learning ---
            st.info("[Knowledge Extraction] Extracting knowledge from workflow execution...")
            try:
                from knowledge_manager import KnowledgeManager
                km = KnowledgeManager()
                
                # Extract knowledge artifacts from the completed workflow
                artifacts = _extract_workflow_knowledge(workflow_state, km)
                
                if artifacts:
                    st.success(f"[Knowledge Extraction] Extracted {len(artifacts)} knowledge artifacts.")
                    workflow_state.knowledge_artifacts = artifacts
                else:
                    st.info("[Knowledge Extraction] No knowledge artifacts extracted.")
            except Exception as e:
                st.warning(f"[Knowledge Extraction] Failed to extract knowledge: {e}")
            
            st.info("INFO: Workflow completed.")
            return # Workflow completed.

        st.error("Max refinement loops reached for final solution. Manual intervention required.")
        workflow_state.status = "failed"
        workflow_state.end_time = time.time()
        st.info("INFO: Workflow failed.")

def parse_targeted_feedback(report: Any) -> List[str]:
    """
    Parses a critique or verification report to identify problematic sub-problem IDs mentioned in the feedback.
    It expects `targeted_feedback` within the judge reports to be a JSON array of strings (sub-problem IDs).
    It attempts to parse JSON feedback first, falling back to regular expression matching if JSON parsing fails or is not an array.

    Args:
        report (Any): The critique or verification report object (CritiqueReport or VerificationReport).

    Returns:
        List[str]: A list of unique sub-problem IDs identified as problematic in the feedback.
    """
    problematic_ids = []
    
    # Convert report to dict if it's a dataclass
    if dataclasses.is_dataclass(report):
        report = dataclasses.asdict(report)

    for judge_report in report['reports_by_judge']:
        feedback = judge_report.get('targeted_feedback', '')
        
        # If feedback is already a list, use it directly
        if isinstance(feedback, list):
            problematic_ids.extend(feedback)
            continue
        
        # Attempt to parse as JSON first if it's a string
        if isinstance(feedback, str):
            try:
                json_feedback = json.loads(feedback)
                # If the feedback is directly a list of strings (sub-problem IDs)
                if isinstance(json_feedback, list) and all(isinstance(item, str) for item in json_feedback):
                    problematic_ids.extend(json_feedback)
                # If it's a dictionary that might contain a list of problematic_sub_problems
                elif isinstance(json_feedback, dict) and "problematic_sub_problems" in json_feedback and isinstance(json_feedback["problematic_sub_problems"], list):
                    problematic_ids.extend(json_feedback["problematic_sub_problems"])
                else:
                    st.warning(f"Targeted feedback JSON from LLM was not in expected format (list of strings or dict with 'problematic_sub_problems' list): {feedback[:200]}...")
            except json.JSONDecodeError:
                # Fallback to regex if not JSON or if JSON parsing fails
                found_ids = re.findall(r'(sub_\d+\.\d+)', feedback)
                if found_ids:
                    st.info(f"Extracted sub-problem IDs via regex: {found_ids}")
                    problematic_ids.extend(found_ids)
                else:
                    st.warning(f"Could not parse targeted feedback as JSON or extract sub-problem IDs via regex: {feedback[:200]}...")
            
    return list(set(problematic_ids)) # Return unique IDs

from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config

def generate_solution_for_sub_problem(sub_problem: SubProblem, team: Team, context: Dict[str, Any], workflow_state: WorkflowState, solver_generation_gauntlet: Optional[GauntletDefinition] = None) -> str:
    """    Generates a solution for a given sub-problem using the assigned solver team and OpenEvolve.
    This function supports different generation modes based on the `solver_generation_gauntlet`:
    - `single_candidate`: A single model directly generates the solution.
    - `multi_candidate_peer_review`: Multiple models generate candidates, and then a peer review process
      selects or synthesizes the best solution.
    If `sub_problem.evolution_params` are provided, it leverages `run_unified_evolution` for advanced evolutionary generation.

    Args:
        sub_problem (SubProblem): The sub-problem for which to generate a solution.
        team (Team): The Blue Team (Solver or Patcher) responsible for generating the solution.
        context (Dict[str, Any]): Additional context, such as previous solution attempts or feedback reports.
        workflow_state (WorkflowState): The current state of the overall workflow, used to access global parameters.
        solver_generation_gauntlet (Optional[GauntletDefinition]): The Blue Team Gauntlet defining the generation mode.

    Returns:
        str: The generated solution content, or an error message if generation fails.
    """
    st.info(f"Generating solution for {sub_problem.id} using {team.name} via OpenEvolve...")
    
    # Enrich context with external knowledge if available
    try:
        from external_knowledge_integration import get_knowledge_integration_manager
        knowledge_manager = get_knowledge_integration_manager()
        
        # Query external knowledge based on sub-problem
        knowledge_context = {
            "query": sub_problem.description,
            "domain": workflow_state.decomposition_plan.analyzed_context.get("domain", ""),
            "keywords": workflow_state.decomposition_plan.analyzed_context.get("keywords", []),
            "limit": 5
        }
        
        external_knowledge = knowledge_manager.query_all_connectors(knowledge_context)
        if external_knowledge:
            context["external_knowledge"] = external_knowledge
            st.info(f"  - Enriched context with external knowledge from {len(external_knowledge)} sources")
    except Exception as e:
        # Continue without external knowledge if it fails
        st.warning(f"  - Could not retrieve external knowledge: {e}")

    if not team.members:
        st.error(f"Solver Team '{team.name}' has no members. Please configure the team in the Team Manager.")
        return "Failed to generate solution: No team members."

    model_config = team.members[0] # Use the first model in the team for generation

    # Construct OpenEvolve configuration
    system_message = team.solver_system_prompt if team.solver_system_prompt else f"You are an expert AI assistant tasked with solving sub-problem {sub_problem.id}. Generate a high-quality solution based on the description and context."
    user_prompt_template = team.solver_user_prompt_template if team.solver_user_prompt_template else f"""Solve the following sub-problem:
    ---
    {{sub_problem_description}}
    ---
    
    Context from overall problem:
    ---
    {{analyzed_context}}
    ---
    
    {{existing_solution_to_refine}}
    
    Provide the solution directly.
    """
    
    # Format the user prompt template with actual values
    formatted_user_prompt = user_prompt_template.replace("{{sub_problem_description}}", sub_problem.description)
    formatted_user_prompt = formatted_user_prompt.replace("{{analyzed_context}}", json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2))
    
    existing_solution_text = ""
    if "current_solution" in context and context["current_solution"]:
        existing_solution_text = f"Existing solution to refine:\n---\n{context['current_solution']}\n---"
    formatted_user_prompt = formatted_user_prompt.replace("{{existing_solution_to_refine}}", existing_solution_text)
    
    # Add external knowledge to prompt if available
    if "external_knowledge" in context and context["external_knowledge"]:
        external_knowledge_text = "\n\nRelevant External Knowledge:\n"
        for source_name, items in context["external_knowledge"].items():
            if items:
                external_knowledge_text += f"\nFrom {source_name}:\n"
                for item in items[:3]:  # Limit to top 3 items per source
                    if hasattr(item, 'content'):
                        external_knowledge_text += f"- {item.content[:200]}...\n"
                    elif isinstance(item, dict):
                        external_knowledge_text += f"- {str(item)[:200]}...\n"
        formatted_user_prompt += external_knowledge_text

    if "feedback_report" in context:
        feedback_report_obj = context["feedback_report"]
        if dataclasses.is_dataclass(feedback_report_obj):
            feedback_report_obj = dataclasses.asdict(feedback_report_obj)
        feedback_json = json.dumps(feedback_report_obj, indent=2)
        system_message += f"\n\nPrevious feedback for this sub-problem:\n---\n{feedback_json}\n---"
        user_prompt_template += f"\n\nAddress the issues raised in this feedback to improve the solution."
    
    if solver_generation_gauntlet and solver_generation_gauntlet.generation_mode == "single_candidate":
        st.info(f"  - Using single_candidate generation mode for {sub_problem.id}...")
        
        # If an evolutionary mode is suggested, use run_unified_evolution
        if sub_problem.ai_suggested_evolution_mode != "standard":
            st.info(f"  - Using OpenEvolve for {sub_problem.ai_suggested_evolution_mode} evolution for {sub_problem.id}...")
            
            # Construct OpenEvolve configuration from workflow_state and sub_problem.evolution_params
            # Prioritize sub_problem.evolution_params for overrides
            openevolve_config_params = {
                "content": formatted_user_prompt,
                "content_type": sub_problem.content_type, # Use sub_problem's content_type
                "evolution_mode": sub_problem.ai_suggested_evolution_mode,
                "model_configs": [model_config], # Pass the specific model config
                "api_key": model_config.api_key,
                "api_base": model_config.api_base,
                "system_message": system_message,
                "evaluator_system_message": team.gold_team_system_prompt, # Use Gold Team system prompt for evaluator

                # Global OpenEvolve parameters from workflow_state
                "max_iterations": workflow_state.max_iterations,
                "population_size": workflow_state.population_size,
                "num_islands": workflow_state.num_islands,
                "migration_interval": workflow_state.migration_interval,
                "migration_rate": workflow_state.migration_rate,
                "archive_size": workflow_state.archive_size,
                "elite_ratio": workflow_state.elite_ratio,
                "exploration_ratio": workflow_state.exploration_ratio,
                "exploitation_ratio": workflow_state.exploitation_ratio,
                "checkpoint_interval": workflow_state.checkpoint_interval,
                "feature_dimensions": workflow_state.feature_dimensions,
                "feature_bins": workflow_state.feature_bins,
                "diversity_metric": workflow_state.diversity_metric,
                "enable_artifacts": workflow_state.enable_artifacts,
                "cascade_evaluation": workflow_state.cascade_evaluation,
                "cascade_thresholds": workflow_state.cascade_thresholds,
                "use_llm_feedback": workflow_state.use_llm_feedback,
                "llm_feedback_weight": workflow_state.llm_feedback_weight,
                "parallel_evaluations": workflow_state.parallel_evaluations,
                "distributed": workflow_state.distributed,
                "template_dir": workflow_state.template_dir,
                "num_top_programs": workflow_state.num_top_programs,
                "num_diverse_programs": workflow_state.num_diverse_programs,
                "use_template_stochasticity": workflow_state.use_template_stochasticity,
                "template_variations": workflow_state.template_variations,
                "use_meta_prompting": workflow_state.use_meta_prompting,
                "meta_prompt_weight": workflow_state.meta_prompt_weight,
                "include_artifacts": workflow_state.include_artifacts,
                "max_artifact_bytes": workflow_state.max_artifact_bytes,
                "artifact_security_filter": workflow_state.artifact_security_filter,
                "early_stopping_patience": workflow_state.early_stopping_patience,
                "convergence_threshold": workflow_state.convergence_threshold,
                "early_stopping_metric": workflow_state.early_stopping_metric,
                "memory_limit_mb": workflow_state.memory_limit_mb,
                "cpu_limit": workflow_state.cpu_limit,
                "random_seed": workflow_state.random_seed,
                "db_path": workflow_state.db_path,
                "in_memory": workflow_state.in_memory,
                "diff_based_evolution": workflow_state.diff_based_evolution,
                "max_code_length": workflow_state.max_code_length,
                "evolution_trace_enabled": workflow_state.evolution_trace_enabled,
                "evolution_trace_format": workflow_state.evolution_trace_format,
                "evolution_trace_include_code": workflow_state.evolution_trace_include_code,
                "evolution_trace_include_prompts": workflow_state.evolution_trace_include_prompts,
                "evolution_trace_output_path": workflow_state.evolution_trace_output_path,
                "evolution_trace_buffer_size": workflow_state.evolution_trace_buffer_size,
                "evolution_trace_compress": workflow_state.evolution_trace_compress,
                "log_level": workflow_state.log_level,
                "log_dir": workflow_state.log_dir,
                "api_timeout": workflow_state.api_timeout,
                "api_retries": workflow_state.api_retries,
                "api_retry_delay": workflow_state.api_retry_delay,
                "artifact_size_threshold": workflow_state.artifact_size_threshold,
                "cleanup_old_artifacts": workflow_state.cleanup_old_artifacts,
                "artifact_retention_days": workflow_state.artifact_retention_days,
                "diversity_reference_size": workflow_state.diversity_reference_size,
                "max_retries_eval": workflow_state.max_retries_eval,
                "evaluator_timeout": workflow_state.evaluator_timeout,
                "double_selection": workflow_state.double_selection,
                "adaptive_feature_dimensions": workflow_state.adaptive_feature_dimensions,
                "test_time_compute": workflow_state.test_time_compute,
                "optillm_integration": workflow_state.optillm_integration,
                "plugin_system": workflow_state.plugin_system,
                "hardware_optimization": workflow_state.hardware_optimization,
                "multi_strategy_sampling": workflow_state.multi_strategy_sampling,
                "ring_topology": workflow_state.ring_topology,
                "controlled_gene_flow": workflow_state.controlled_gene_flow,
                "auto_diff": workflow_state.auto_diff,
                "symbolic_execution": workflow_state.symbolic_execution,
                "coevolutionary_approach": workflow_state.coevolutionary_approach
            }
            
            # Override with sub_problem-specific evolution_params
            openevolve_config_params.update(sub_problem.evolution_params)

            # Create a comprehensive config object for OpenEvolve
            openevolve_config = create_comprehensive_openevolve_config(**openevolve_config_params)

            try:
                result = run_unified_evolution(**openevolve_config)
                if result and result.get("success") and result.get("best_solution"):
                    generated_solution_content = result["best_solution"]
                    st.success(f"Solution generated for {sub_problem.id} using OpenEvolve ({sub_problem.ai_suggested_evolution_mode}).")
                else:
                    st.error(f"OpenEvolve failed to generate solution for {sub_problem.id}. Result: {result}")
                    return "Failed to generate solution: OpenEvolve failed."
            except Exception as e:
                st.error(f"Error running OpenEvolve for sub-problem {sub_problem.id}: {e}")
                return "Failed to generate solution: OpenEvolve error."
        else:
            # Fallback to direct LLM call for "standard" evolution mode or if no specific mode is suggested.
            st.info(f"  - Using direct LLM call for {sub_problem.id} (standard generation)...")
            response = _request_openai_compatible_chat(
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                model=model_config.model_id,
                messages=_compose_messages(system_message, formatted_user_prompt),
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                max_tokens=model_config.max_tokens,
                frequency_penalty=model_config.frequency_penalty,
                presence_penalty=model_config.presence_penalty,
                seed=model_config.seed,
                n=model_config.n,
                logit_bias=model_config.logit_bias,
                reasoning_effort=model_config.reasoning_effort,
                stop_sequences=model_config.stop_sequences,
                logprobs=model_config.logprobs,
                top_logprobs=model_config.top_logprobs,
                response_format=model_config.response_format,
                stream=model_config.stream,
                user=model_config.user,
                max_retries=model_config.max_retries,
                timeout=model_config.timeout,
                organization=model_config.organization,
                response_model=model_config.response_model,
                tools=model_config.tools,
                tool_choice=model_config.tool_choice,
                system_fingerprint=model_config.system_fingerprint,
                deployment_id=model_config.deployment_id,
                encoding_format=model_config.encoding_format,
                max_input_tokens=model_config.max_input_tokens,
                stop_token=model_config.stop_token,
                best_of=model_config.best_of,
                logprobs_offset=model_config.logprobs_offset,
                suffix=model_config.suffix,
                presence_penalty_range=model_config.presence_penalty_range,
                frequency_penalty_range=model_config.frequency_penalty_range,
                stop_token_id=model_config.stop_token_id,
                response_json_format=model_config.response_json_format,
                max_output_tokens=model_config.max_output_tokens,
                stream_options=model_config.stream_options,
                logprobs_type=model_config.logprobs_type,
                top_k=model_config.top_k,
                repetition_penalty=model_config.repetition_penalty,
                length_penalty=model_config.length_penalty,
                early_stopping=model_config.early_stopping,
                num_beams=model_config.num_beams,
                do_sample=model_config.do_sample,
                temperature_fallback=model_config.temperature_fallback,
                top_p_fallback=model_config.top_p_fallback,
                max_time=model_config.max_time,
                return_full_text=model_config.return_full_text,
                tokenizer_config=model_config.tokenizer_config,
                model_kwargs=model_config.model_kwargs
            )
            
            if response:
                generated_solution_content = response
                st.success(f"Solution generated for {sub_problem.id} by {model_config.model_id}.")
            else:
                st.error(f"Failed to generate solution for {sub_problem.id} in single_candidate mode.")
                return "Failed to generate solution: LLM call failed."

    # Multi-Candidate Peer Review Generation: Multiple models generate candidates, then one synthesizes/reviews.
    elif solver_generation_gauntlet.generation_mode == "multi_candidate_peer_review":
        st.info(f"  - Using multi_candidate_peer_review generation mode for {sub_problem.id}...")
        candidates = []
        
        # Step 1: Generate multiple candidate solutions from team members.
        for i, member in enumerate(team.members):
            candidate_system_message = team.solver_system_prompt if team.solver_system_prompt else f"You are an AI assistant tasked with generating a candidate solution for sub-problem {sub_problem.id}. Your goal is to provide a unique and high-quality approach."
            candidate_user_prompt_template = team.solver_user_prompt_template if team.solver_user_prompt_template else f"""Generate a candidate solution for the following sub-problem:
            ---
            {{sub_problem_description}}
            ---
            
            Context from overall problem:
            ---
            {{analyzed_context}}
            ---
            
            {{existing_solution_to_refine}}
            
            Provide the candidate solution directly.
            """
            
            # Add feedback to system message if available
            feedback_json = None
            if "feedback_report" in context:
                feedback_report_obj = context["feedback_report"]
                if dataclasses.is_dataclass(feedback_report_obj):
                    feedback_report_obj = dataclasses.asdict(feedback_report_obj)
                feedback_json = json.dumps(feedback_report_obj, indent=2)
                candidate_system_message += f"\n\nPrevious feedback for this sub-problem:\n---\n{feedback_json}\n---\nAddress the issues raised in this feedback to improve the solution."
                candidate_user_prompt_template += f"\n\nAddress the issues raised in this feedback to improve the solution."

            # Format the user prompt template with actual values
            formatted_candidate_user_prompt = candidate_user_prompt_template.replace("{{sub_problem_description}}", sub_problem.description)
            formatted_candidate_user_prompt = formatted_candidate_user_prompt.replace("{{analyzed_context}}", json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2))
            
            existing_solution_text = ""
            if "current_solution" in context and context["current_solution"]:
                existing_solution_text = f"Existing solution to refine:\n---\n{context['current_solution']}\n---"
            formatted_candidate_user_prompt = formatted_candidate_user_prompt.replace("{{existing_solution_to_refine}}", existing_solution_text)

            candidate_response = _request_openai_compatible_chat(
                            api_key=member.api_key,
                            base_url=member.api_base,
                            model=member.model_id,
                            messages=_compose_messages(candidate_system_message, formatted_candidate_user_prompt),
                            temperature=member.temperature + (i * 0.1), # Slightly vary temperature for diversity in candidates.
                            top_p=member.top_p,
                            max_tokens=member.max_tokens,
                            frequency_penalty=member.frequency_penalty,
                            presence_penalty=member.presence_penalty,
                            seed=member.seed,
                            n=member.n,
                            logit_bias=member.logit_bias,
                            reasoning_effort=member.reasoning_effort,
                            stop_sequences=member.stop_sequences,
                            logprobs=member.logprobs,
                            top_logprobs=member.top_logprobs,
                            response_format=member.response_format,
                            stream=member.stream,
                            user=member.user,
                            max_retries=member.max_retries,
                            timeout=member.timeout,
                            organization=member.organization,
                            response_model=member.response_model,
                            tools=member.tools,
                            tool_choice=member.tool_choice,
                            system_fingerprint=member.system_fingerprint,
                            deployment_id=member.deployment_id,
                            encoding_format=member.encoding_format,
                            max_input_tokens=member.max_input_tokens,
                            stop_token=member.stop_token,
                            best_of=member.best_of,
                            logprobs_offset=member.logprobs_offset,
                            suffix=member.suffix,
                            presence_penalty_range=member.presence_penalty_range,
                            frequency_penalty_range=member.frequency_penalty_range,
                            stop_token_id=member.stop_token_id,
                            response_json_format=member.response_json_format,
                            max_output_tokens=member.max_output_tokens,
                            stream_options=member.stream_options,
                            logprobs_type=member.logprobs_type,
                            top_k=member.top_k,
                            repetition_penalty=member.repetition_penalty,
                            length_penalty=member.length_penalty,
                            early_stopping=member.early_stopping,
                            num_beams=member.num_beams,
                            do_sample=member.do_sample,
                            temperature_fallback=member.temperature_fallback,
                            top_p_fallback=member.top_p_fallback,
                            max_time=member.max_time,
                            return_full_text=member.return_full_text,
                            tokenizer_config=member.tokenizer_config,
                            model_kwargs=member.model_kwargs
                        )
            
            if candidate_response:
                candidates.append({"model_id": member.model_id, "content": candidate_response})
                st.info(f"    - Candidate {i+1} generated by {member.model_id}.")
            else:
                st.warning(f"    - Failed to generate candidate {i+1} by {member.model_id}.")

        if not candidates:
            st.error(f"No candidates generated for sub-problem {sub_problem.id} in multi_candidate_peer_review mode.")
            return "Failed to generate solution: No candidates produced."

        # Step 2: Peer review and synthesize the best candidate from the generated options.
        review_system_message = team.solver_system_prompt if team.solver_system_prompt else f"You are an expert AI peer reviewer and synthesizer. Your task is to review multiple candidate solutions for sub-problem {sub_problem.id} and synthesize the best possible solution, incorporating the strengths of each and addressing any weaknesses. If a single candidate is clearly superior, you may select it. Otherwise, combine and refine."
        review_user_prompt_template = team.solver_user_prompt_template if team.solver_user_prompt_template else f"""Review the following candidate solutions for sub-problem {sub_problem.id} and synthesize the best possible solution.
        
        Sub-problem Description:
        ---
        {{sub_problem_description}}
        ---
        
        Context from overall problem:
        ---
        {{analyzed_context}}
        ---
        
        {{existing_solution_to_refine}}
        
        Candidate Solutions:
        ---
        {{candidate_solutions}}
        ---
        
        Provide the synthesized best solution directly.
        """
        
        # Format the user prompt template with actual values
        formatted_review_user_prompt = review_user_prompt_template.replace("{{sub_problem_description}}", sub_problem.description)
        formatted_review_user_prompt = formatted_review_user_prompt.replace("{{analyzed_context}}", json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2))
        formatted_review_user_prompt = formatted_review_user_prompt.replace("{{candidate_solutions}}", json.dumps(candidates, indent=2))
        
        existing_solution_text = ""
        if "current_solution" in context and context["current_solution"]:
            existing_solution_text = f"Existing solution to refine:\n---\n{context['current_solution']}\n---"
        formatted_review_user_prompt = formatted_review_user_prompt.replace("{{existing_solution_to_refine}}", existing_solution_text)

        synthesized_response = _request_openai_compatible_chat(
            api_key=model_config.api_key, # Use the primary model for synthesis.
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(review_system_message, formatted_review_user_prompt),
            temperature=0.5, # Lower temperature for more deterministic synthesis.
            top_p=model_config.top_p,
            max_tokens=model_config.max_tokens,
            frequency_penalty=model_config.frequency_penalty,
            presence_penalty=model_config.presence_penalty,
            seed=model_config.seed,
            n=model_config.n,
            logit_bias=model_config.logit_bias,
            reasoning_effort=model_config.reasoning_effort,
            stop_sequences=model_config.stop_sequences,
            logprobs=model_config.logprobs,
            top_logprobs=model_config.top_logprobs,
            response_format=model_config.response_format,
            stream=model_config.stream,
            user=model_config.user,
            max_retries=model_config.max_retries,
            timeout=model_config.timeout,
            organization=model_config.organization,
            response_model=model_config.response_model,
            tools=model_config.tools,
            tool_choice=model_config.tool_choice,
            system_fingerprint=model_config.system_fingerprint,
            deployment_id=model_config.deployment_id,
            encoding_format=model_config.encoding_format,
            max_input_tokens=model_config.max_input_tokens,
            stop_token=model_config.stop_token,
            best_of=model_config.best_of,
            logprobs_offset=model_config.logprobs_offset,
            suffix=model_config.suffix,
            presence_penalty_range=model_config.presence_penalty_range,
            frequency_penalty_range=model_config.frequency_penalty_range,
            stop_token_id=model_config.stop_token_id,
            response_json_format=model_config.response_json_format,
            max_output_tokens=model_config.max_output_tokens,
            stream_options=model_config.stream_options,
            logprobs_type=model_config.logprobs_type,
            top_k=model_config.top_k,
            repetition_penalty=model_config.repetition_penalty,
            length_penalty=model_config.length_penalty,
            early_stopping=model_config.early_stopping,
            num_beams=model_config.num_beams,
            do_sample=model_config.do_sample,
            temperature_fallback=model_config.temperature_fallback,
            top_p_fallback=model_config.top_p_fallback,
            max_time=model_config.max_time,
            return_full_text=model_config.return_full_text,
            tokenizer_config=model_config.tokenizer_config,
            model_kwargs=model_config.model_kwargs
        )
        
        if synthesized_response:
            generated_solution_content = synthesized_response
            st.success(f"Solution synthesized for {sub_problem.id} by {model_config.model_id}.")
        else:
            st.error(f"Failed to synthesize solution for {sub_problem.id} in multi_candidate_peer_review mode.")
            return "Failed to generate solution: Synthesis failed."
    else:
        st.error(f"No valid generation method specified for sub-problem {sub_problem.id}. Neither evolution_params nor solver_generation_gauntlet provided.")
        return "Failed to generate solution: No generation method specified."

    return generated_solution_content



# --- Advanced Gauntlet Type Implementations ---

def _run_adaptive_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute an adaptive gauntlet that adjusts its rules based on the content being evaluated.
    
    Adaptive behavior:
    - Analyzes content complexity and adjusts thresholds
    - Increases scrutiny for complex or critical solutions
    - Reduces requirements for simple, low-risk solutions
    """
    st.info("Adaptive Gauntlet: Analyzing content to adjust evaluation criteria...")
    
    # Analyze content to determine complexity
    content_complexity = _analyze_content_complexity(solution_content, context)
    st.write(f"Content complexity score: {content_complexity:.2f}")
    
    # Create adapted gauntlet definition
    adapted_gauntlet = GauntletDefinition(
        name=f"{gauntlet_def.name} (Adapted)",
        team_name=gauntlet_def.team_name,
        rounds=[],
        description=f"Adaptive version of {gauntlet_def.name}",
        attack_modes=gauntlet_def.attack_modes,
        generation_mode=gauntlet_def.generation_mode,
        gauntlet_type="standard"  # Run as standard after adaptation
    )
    
    # Adapt each round based on complexity
    for round_rule in gauntlet_def.rounds:
        adapted_round = GauntletRoundRule(
            round_number=round_rule.round_number,
            quorum_required_approvals=round_rule.quorum_required_approvals,
            quorum_from_panel_size=round_rule.quorum_from_panel_size,
            min_overall_confidence=_adapt_threshold(
                round_rule.min_overall_confidence,
                content_complexity
            ),
            max_score_variance=_adapt_variance(
                round_rule.max_score_variance,
                content_complexity
            ),
            per_judge_requirements=round_rule.per_judge_requirements,
            collaboration_mode=round_rule.collaboration_mode
        )
        adapted_gauntlet.rounds.append(adapted_round)
    
    st.success(f"Adapted gauntlet criteria based on complexity score {content_complexity:.2f}")
    
    # Run the adapted gauntlet as a standard gauntlet
    return _run_standard_gauntlet(solution_content, adapted_gauntlet, team, context)


def _analyze_content_complexity(solution_content: str, context: Dict[str, Any]) -> float:
    """
    Analyze content to determine its complexity score (0.0-1.0).
    Higher scores indicate more complex content requiring stricter evaluation.
    """
    complexity_score = 0.5  # Base score
    
    # Length-based complexity
    content_length = len(solution_content)
    if content_length > 5000:
        complexity_score += 0.2
    elif content_length > 2000:
        complexity_score += 0.1
    
    # Code complexity indicators
    if "def " in solution_content or "class " in solution_content:
        complexity_score += 0.1
    if solution_content.count("if ") > 5:
        complexity_score += 0.1
    if solution_content.count("for ") > 3 or solution_content.count("while ") > 2:
        complexity_score += 0.1
    
    # Context-based complexity
    if context.get("sub_problem"):
        sub_problem = context["sub_problem"]
        if isinstance(sub_problem, dict):
            ai_complexity = sub_problem.get("ai_suggested_complexity_score", 5)
            complexity_score += (ai_complexity / 10) * 0.2
    
    return min(1.0, complexity_score)


def _adapt_threshold(original_threshold: float, complexity: float) -> float:
    """Adapt confidence threshold based on complexity."""
    # Higher complexity = higher threshold
    adaptation_factor = 0.5 + (complexity * 0.5)  # 0.5 to 1.0
    adapted = original_threshold * adaptation_factor
    return min(0.95, max(0.3, adapted))  # Clamp between 0.3 and 0.95


def _adapt_variance(original_variance: Optional[float], complexity: float) -> Optional[float]:
    """Adapt variance threshold based on complexity."""
    if original_variance is None:
        return None
    # Higher complexity = lower allowed variance (stricter consensus)
    adaptation_factor = 1.5 - (complexity * 0.5)  # 1.5 to 1.0
    adapted = original_variance * adaptation_factor
    return max(0.05, adapted)  # Minimum variance of 0.05


def _run_hierarchical_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a hierarchical gauntlet with multiple tiers of evaluation.
    
    Each tier has increasingly strict criteria. Solutions must pass all tiers.
    """
    st.info("Hierarchical Gauntlet: Evaluating through multiple tiers...")
    
    # Divide rounds into tiers (every 2 rounds = 1 tier)
    tier_size = max(1, len(gauntlet_def.rounds) // 3)  # 3 tiers
    tiers = []
    for i in range(0, len(gauntlet_def.rounds), tier_size):
        tiers.append(gauntlet_def.rounds[i:i + tier_size])
    
    all_judge_reports = []
    current_tier = 1
    
    for tier_rounds in tiers:
        st.subheader(f"Tier {current_tier}/{len(tiers)}")
        
        # Create a gauntlet for this tier
        tier_gauntlet = GauntletDefinition(
            name=f"{gauntlet_def.name} - Tier {current_tier}",
            team_name=gauntlet_def.team_name,
            rounds=tier_rounds,
            description=f"Tier {current_tier} of hierarchical gauntlet",
            attack_modes=gauntlet_def.attack_modes,
            generation_mode=gauntlet_def.generation_mode,
            gauntlet_type="standard"
        )
        
        # Run this tier
        tier_result = _run_standard_gauntlet(solution_content, tier_gauntlet, team, context)
        
        # Collect reports
        if "critique_report" in tier_result:
            all_judge_reports.extend(tier_result["critique_report"].reports_by_judge)
        elif "verification_report" in tier_result:
            all_judge_reports.extend(tier_result["verification_report"].reports_by_judge)
        
        # If tier failed, stop evaluation
        if not tier_result["is_approved"]:
            st.warning(f"Failed at Tier {current_tier}. Hierarchical gauntlet rejected.")
            
            # Return failure report
            if team.role == "Red":
                return {
                    "is_approved": False,
                    "report_summary": f"Hierarchical Gauntlet '{gauntlet_def.name}' REJECTED at Tier {current_tier}",
                    "critique_report": CritiqueReport(
                        solution_attempt_id=context.get('solution_id', 'unknown'),
                        gauntlet_name=gauntlet_def.name,
                        is_approved=False,
                        reports_by_judge=all_judge_reports,
                        summary=f"Failed at Tier {current_tier}/{len(tiers)}"
                    )
                }
            else:
                return {
                    "is_approved": False,
                    "report_summary": f"Hierarchical Gauntlet '{gauntlet_def.name}' REJECTED at Tier {current_tier}",
                    "verification_report": VerificationReport(
                        solution_attempt_id=context.get('solution_id', 'unknown'),
                        gauntlet_name=gauntlet_def.name,
                        is_approved=False,
                        reports_by_judge=all_judge_reports,
                        summary=f"Failed at Tier {current_tier}/{len(tiers)}"
                    )
                }
        
        st.success(f"Passed Tier {current_tier}")
        current_tier += 1
    
    # All tiers passed
    st.success(f"Passed all {len(tiers)} tiers!")
    
    if team.role == "Red":
        return {
            "is_approved": True,
            "report_summary": f"Hierarchical Gauntlet '{gauntlet_def.name}' APPROVED (passed all {len(tiers)} tiers)",
            "critique_report": CritiqueReport(
                solution_attempt_id=context.get('solution_id', 'unknown'),
                gauntlet_name=gauntlet_def.name,
                is_approved=True,
                reports_by_judge=all_judge_reports,
                summary=f"Passed all {len(tiers)} tiers"
            )
        }
    else:
        return {
            "is_approved": True,
            "report_summary": f"Hierarchical Gauntlet '{gauntlet_def.name}' APPROVED (passed all {len(tiers)} tiers)",
            "verification_report": VerificationReport(
                solution_attempt_id=context.get('solution_id', 'unknown'),
                gauntlet_name=gauntlet_def.name,
                is_approved=True,
                reports_by_judge=all_judge_reports,
                summary=f"Passed all {len(tiers)} tiers"
            )
        }


def _run_competitive_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a competitive gauntlet where multiple solutions compete.
    
    Note: This requires multiple solutions in context. If only one solution is provided,
    it falls back to standard evaluation.
    """
    st.info("Competitive Gauntlet: Comparing multiple solutions...")
    
    # Check if we have multiple solutions to compare
    competing_solutions = context.get("competing_solutions", [solution_content])
    
    if len(competing_solutions) <= 1:
        st.warning("Only one solution provided. Running as standard gauntlet.")
        return _run_standard_gauntlet(solution_content, gauntlet_def, team, context)
    
    st.write(f"Comparing {len(competing_solutions)} solutions...")
    
    # Evaluate each solution
    solution_scores = []
    for idx, solution in enumerate(competing_solutions):
        st.subheader(f"Evaluating Solution {idx + 1}/{len(competing_solutions)}")
        
        # Run standard gauntlet for this solution
        result = _run_standard_gauntlet(solution, gauntlet_def, team, context)
        
        # Extract average score
        if "verification_report" in result:
            avg_score = result["verification_report"].average_score
        elif "critique_report" in result:
            # For critique, use inverse of flaw count as score
            avg_score = 1.0 if result["is_approved"] else 0.5
        else:
            avg_score = 0.5
        
        solution_scores.append({
            "solution_idx": idx,
            "solution": solution,
            "score": avg_score,
            "result": result
        })
    
    # Sort by score (highest first)
    solution_scores.sort(key=lambda x: x["score"], reverse=True)
    
    # The best solution wins
    best_solution = solution_scores[0]
    st.success(f"Solution {best_solution['solution_idx'] + 1} wins with score {best_solution['score']:.2f}")
    
    # Return result for the best solution
    return best_solution["result"]


def _run_collaborative_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a collaborative gauntlet where models work together to improve the solution.
    
    Instead of just evaluating, models suggest improvements and the solution is iteratively refined.
    """
    st.info("Collaborative Gauntlet: Models working together to improve solution...")
    
    current_solution = solution_content
    improvement_iterations = gauntlet_def.gauntlet_config.get("max_iterations", 3) if gauntlet_def.gauntlet_config else 3
    all_judge_reports = []
    
    for iteration in range(improvement_iterations):
        st.subheader(f"Collaboration Iteration {iteration + 1}/{improvement_iterations}")
        
        # Evaluate current solution
        result = _run_standard_gauntlet(current_solution, gauntlet_def, team, context)
        
        # Collect reports
        if "critique_report" in result:
            all_judge_reports.extend(result["critique_report"].reports_by_judge)
        elif "verification_report" in result:
            all_judge_reports.extend(result["verification_report"].reports_by_judge)
        
        # If approved, we're done
        if result["is_approved"]:
            st.success(f"Solution approved after {iteration + 1} iterations!")
            return result
        
        # Otherwise, collect improvement suggestions
        st.info("Collecting improvement suggestions from team...")
        improvements = []
        
        for judge_report in (result.get("critique_report") or result.get("verification_report")).reports_by_judge:
            if judge_report.get("justification"):
                improvements.append(judge_report["justification"])
        
        if not improvements:
            st.warning("No improvement suggestions available. Stopping collaboration.")
            return result
        
        # Ask a team member to improve the solution
        improvement_prompt = f"""The following solution needs improvement based on team feedback:

Original Solution:
---
{current_solution}
---

Team Feedback:
{chr(10).join(f"- {imp}" for imp in improvements)}

Please provide an improved version of the solution that addresses the feedback."""
        
        # Use first team member to generate improvement
        if team.members:
            member = team.members[0]
            improved_solution = _request_openai_compatible_chat(
                api_key=member.api_key,
                base_url=member.api_base,
                model=member.model_id,
                messages=_compose_messages(
                    "You are an AI assistant helping to improve solutions based on team feedback.",
                    improvement_prompt
                ),
                temperature=member.temperature,
                max_tokens=member.max_tokens
            )
            
            if improved_solution:
                current_solution = improved_solution
                st.success("Solution improved. Re-evaluating...")
            else:
                st.warning("Failed to generate improvement. Stopping collaboration.")
                return result
        else:
            st.warning("No team members available for improvement. Stopping collaboration.")
            return result
    
    # Max iterations reached
    st.warning(f"Max iterations ({improvement_iterations}) reached. Returning final result.")
    final_result = _run_standard_gauntlet(current_solution, gauntlet_def, team, context)
    
    # Update solution in context if it was improved
    if current_solution != solution_content:
        context["improved_solution"] = current_solution
    
    return final_result



# --- Knowledge Extraction Helper ---

def _extract_workflow_knowledge(workflow_state: WorkflowState, km: 'KnowledgeManager') -> List['KnowledgeArtifact']:
    """
    Extract knowledge artifacts from a completed workflow execution.
    
    Args:
        workflow_state: The completed workflow state
        km: KnowledgeManager instance
        
    Returns:
        List of extracted knowledge artifacts
    """
    from workflow_structures import KnowledgeArtifact
    import hashlib
    from datetime import datetime
    
    artifacts = []
    
    # 1. Extract solution patterns from successful sub-problems
    for sp_id, solution in workflow_state.sub_problem_solutions.items():
        if sp_id in workflow_state.solved_sub_problem_ids:
            artifact_id = hashlib.md5(
                f"{workflow_state.workflow_id}_solution_{sp_id}_{datetime.now().isoformat()}".encode()
            ).hexdigest()
            
            # Find the sub-problem
            sub_problem = next(
                (sp for sp in workflow_state.decomposition_plan.sub_problems if sp.id == sp_id),
                None
            )
            
            if sub_problem:
                artifact = KnowledgeArtifact(
                    id=artifact_id,
                    artifact_type="solution_pattern",
                    content={
                        "sub_problem_id": sp_id,
                        "sub_problem_description": sub_problem.description,
                        "solution_content": solution.content,
                        "solution_approach": solution.solution_approach,
                        "quality_metrics": solution.quality_metrics,
                        "generated_by_model": solution.generated_by_model,
                        "evolution_mode": sub_problem.ai_suggested_evolution_mode,
                        "complexity_score": sub_problem.ai_suggested_complexity_score
                    },
                    source_workflow_id=workflow_state.workflow_id,
                    domain=workflow_state.decomposition_plan.analyzed_context.get("domain"),
                    problem_type=workflow_state.decomposition_plan.analyzed_context.get("problem_type", "general")
                )
                km.store_knowledge_artifact(artifact)
                artifacts.append(artifact)
    
    # 2. Extract problem-solution mapping
    if workflow_state.final_solution and workflow_state.decomposition_plan:
        artifact_id = hashlib.md5(
            f"{workflow_state.workflow_id}_mapping_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        artifact = KnowledgeArtifact(
            id=artifact_id,
            artifact_type="problem_solution_mapping",
            content={
                "problem_statement": workflow_state.problem_statement,
                "analyzed_context": workflow_state.decomposition_plan.analyzed_context,
                "num_sub_problems": len(workflow_state.decomposition_plan.sub_problems),
                "decomposition_strategy": {
                    "sub_problem_types": [sp.ai_suggested_evolution_mode 
                                         for sp in workflow_state.decomposition_plan.sub_problems],
                    "complexity_scores": [sp.ai_suggested_complexity_score 
                                         for sp in workflow_state.decomposition_plan.sub_problems]
                },
                "final_solution": workflow_state.final_solution.content,
                "success_metrics": {
                    "total_time": workflow_state.end_time - workflow_state.start_time if workflow_state.end_time else 0,
                    "refinement_loops": workflow_state.refinement_loop_count,
                    "solved_sub_problems": len(workflow_state.solved_sub_problem_ids)
                }
            },
            source_workflow_id=workflow_state.workflow_id,
            domain=workflow_state.decomposition_plan.analyzed_context.get("domain"),
            problem_type=workflow_state.decomposition_plan.analyzed_context.get("problem_type", "general")
        )
        km.store_knowledge_artifact(artifact)
        artifacts.append(artifact)
    
    # 3. Extract critique insights from failed attempts
    for critique_report in workflow_state.all_critique_reports:
        if not critique_report.is_approved and critique_report.identified_flaws:
            artifact_id = hashlib.md5(
                f"{workflow_state.workflow_id}_critique_{critique_report.solution_attempt_id}_{datetime.now().isoformat()}".encode()
            ).hexdigest()
            
            artifact = KnowledgeArtifact(
                id=artifact_id,
                artifact_type="critique_insight",
                content={
                    "solution_attempt_id": critique_report.solution_attempt_id,
                    "identified_flaws": critique_report.identified_flaws,
                    "flaw_severity_scores": critique_report.flaw_severity_scores,
                    "suggested_improvements": critique_report.suggested_improvements,
                    "gauntlet_name": critique_report.gauntlet_name
                },
                source_workflow_id=workflow_state.workflow_id,
                domain=workflow_state.decomposition_plan.analyzed_context.get("domain") if workflow_state.decomposition_plan else None,
                problem_type=workflow_state.decomposition_plan.analyzed_context.get("problem_type", "general") if workflow_state.decomposition_plan else "general"
            )
            km.store_knowledge_artifact(artifact)
            artifacts.append(artifact)
    
    # 4. Extract team performance metrics
    teams_used = {}
    
    # Track content analyzer
    if workflow_state.content_analyzer_team:
        teams_used[workflow_state.content_analyzer_team.name] = {
            "role": "content_analyzer",
            "success": True
        }
    
    # Track planner
    if workflow_state.planner_team:
        teams_used[workflow_state.planner_team.name] = {
            "role": "planner",
            "success": True
        }
    
    # Track solver/patcher teams
    if workflow_state.decomposition_plan:
        for sp in workflow_state.decomposition_plan.sub_problems:
            if sp.solver_team_name:
                if sp.solver_team_name not in teams_used:
                    teams_used[sp.solver_team_name] = {
                        "role": "solver",
                        "successes": 0,
                        "failures": 0
                    }
                
                if sp.id in workflow_state.solved_sub_problem_ids:
                    teams_used[sp.solver_team_name]["successes"] += 1
                else:
                    teams_used[sp.solver_team_name]["failures"] += 1
    
    # Create team performance artifacts
    for team_name, performance_data in teams_used.items():
        artifact_id = hashlib.md5(
            f"{workflow_state.workflow_id}_team_{team_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        artifact = KnowledgeArtifact(
            id=artifact_id,
            artifact_type="team_performance",
            content={
                "team_name": team_name,
                "performance_data": performance_data,
                "workflow_id": workflow_state.workflow_id
            },
            source_workflow_id=workflow_state.workflow_id,
            domain=workflow_state.decomposition_plan.analyzed_context.get("domain") if workflow_state.decomposition_plan else None,
            problem_type=workflow_state.decomposition_plan.analyzed_context.get("problem_type", "general") if workflow_state.decomposition_plan else "general"
        )
        km.store_knowledge_artifact(artifact)
        artifacts.append(artifact)
    
    # 5. Extract gauntlet effectiveness
    gauntlets_used = {}
    
    # Track red team gauntlets
    for critique_report in workflow_state.all_critique_reports:
        if critique_report.gauntlet_name not in gauntlets_used:
            gauntlets_used[critique_report.gauntlet_name] = {
                "type": "red_team",
                "total_runs": 0,
                "flaws_found": 0,
                "approved": 0,
                "rejected": 0
            }
        
        gauntlets_used[critique_report.gauntlet_name]["total_runs"] += 1
        if critique_report.is_approved:
            gauntlets_used[critique_report.gauntlet_name]["approved"] += 1
        else:
            gauntlets_used[critique_report.gauntlet_name]["rejected"] += 1
            gauntlets_used[critique_report.gauntlet_name]["flaws_found"] += len(critique_report.identified_flaws)
    
    # Track gold team gauntlets
    for verification_report in workflow_state.all_verification_reports:
        if verification_report.gauntlet_name not in gauntlets_used:
            gauntlets_used[verification_report.gauntlet_name] = {
                "type": "gold_team",
                "total_runs": 0,
                "approved": 0,
                "rejected": 0,
                "average_scores": []
            }
        
        gauntlets_used[verification_report.gauntlet_name]["total_runs"] += 1
        if verification_report.is_approved:
            gauntlets_used[verification_report.gauntlet_name]["approved"] += 1
        else:
            gauntlets_used[verification_report.gauntlet_name]["rejected"] += 1
        
        gauntlets_used[verification_report.gauntlet_name]["average_scores"].append(
            verification_report.average_score
        )
    
    # Create gauntlet effectiveness artifacts
    for gauntlet_name, effectiveness_data in gauntlets_used.items():
        artifact_id = hashlib.md5(
            f"{workflow_state.workflow_id}_gauntlet_{gauntlet_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        artifact = KnowledgeArtifact(
            id=artifact_id,
            artifact_type="gauntlet_effectiveness",
            content={
                "gauntlet_name": gauntlet_name,
                "effectiveness_data": effectiveness_data,
                "workflow_id": workflow_state.workflow_id
            },
            source_workflow_id=workflow_state.workflow_id,
            domain=workflow_state.decomposition_plan.analyzed_context.get("domain") if workflow_state.decomposition_plan else None,
            problem_type=workflow_state.decomposition_plan.analyzed_context.get("problem_type", "general") if workflow_state.decomposition_plan else "general"
        )
        km.store_knowledge_artifact(artifact)
        artifacts.append(artifact)
    
    return artifacts


# --- OpenEvolve Integration Functions ---

def run_content_analysis_with_openevolve(
    problem_statement: str,
    team: Team,
    api_key: str,
    model_name: str = "gpt-4o",
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Run content analysis using OpenEvolve evolution
    
    Args:
        problem_statement: Problem to analyze
        team: Team to use for analysis
        api_key: API key for OpenEvolve
        model_name: Model to use
        max_iterations: Number of evolution iterations
        
    Returns:
        Analyzed context dictionary with OpenEvolve metrics
    """
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        
        # Run evolution to generate analysis
        result = client.evolve(
            content=problem_statement,
            evolution_mode="standard",
            max_iterations=max_iterations,
            population_size=10,
            temperature=0.7,
            model_name=model_name,
            content_type="text_general"
        )
        
        # Parse best analysis
        best_analysis = result.get('best_code', '')
        
        try:
            analyzed_context = json.loads(best_analysis)
        except json.JSONDecodeError:
            # Fallback to basic structure
            analyzed_context = {
                'domain': 'general',
                'keywords': [],
                'estimated_complexity': 5,
                'potential_challenges': [],
                'required_expertise': [],
                'summary': best_analysis[:200]
            }
        
        # Add OpenEvolve metrics
        analyzed_context['openevolve_metrics'] = result.get('metrics', {})
        analyzed_context['openevolve_used'] = True
        
        return analyzed_context
        
    except Exception as e:
        st.error(f"Error using OpenEvolve for content analysis: {e}")
        # Fallback to standard analysis
        return run_content_analysis(problem_statement, team)


def run_decomposition_with_openevolve(
    problem_statement: str,
    analyzed_context: Dict[str, Any],
    team: Team,
    api_key: str,
    model_name: str = "gpt-4o",
    max_iterations: int = 10
) -> Any:
    """
    Run decomposition using OpenEvolve evolution
    
    Args:
        problem_statement: Problem to decompose
        analyzed_context: Context from analysis
        team: Team to use for decomposition
        api_key: API key for OpenEvolve
        model_name: Model to use
        max_iterations: Number of evolution iterations
        
    Returns:
        DecompositionPlan with OpenEvolve metrics
    """
    try:
        from openevolve_client import OpenEvolveClient
        from workflow_structures import DecompositionPlan, SubProblem
        
        client = OpenEvolveClient(api_key=api_key)
        
        # Create prompt for decomposition
        decomposition_prompt = f"""Decompose the following problem into sub-problems:

Problem: {problem_statement}

Context: {json.dumps(analyzed_context, indent=2)}

Provide a JSON array of sub-problems, each with:
- id: unique identifier
- description: clear description
- dependencies: list of dependent sub-problem ids
- estimated_complexity: 1-10 scale
"""
        
        # Run evolution
        result = client.evolve(
            content=decomposition_prompt,
            evolution_mode="standard",
            max_iterations=max_iterations,
            population_size=15,
            temperature=0.8,
            model_name=model_name,
            content_type="text_general"
        )
        
        # Parse decomposition
        best_decomposition = result.get('best_code', '')
        
        try:
            sub_problems_data = json.loads(best_decomposition)
            if not isinstance(sub_problems_data, list):
                sub_problems_data = [sub_problems_data]
        except json.JSONDecodeError:
            # Fallback to single sub-problem
            sub_problems_data = [{
                'id': 'sp_1',
                'description': problem_statement,
                'dependencies': [],
                'estimated_complexity': 5
            }]
        
        # Create SubProblem objects
        sub_problems = []
        for sp_data in sub_problems_data:
            sub_problem = SubProblem(
                id=sp_data.get('id', f'sp_{len(sub_problems)+1}'),
                description=sp_data.get('description', ''),
                dependencies=sp_data.get('dependencies', []),
                ai_suggested_complexity_score=sp_data.get('estimated_complexity', 5)
            )
            sub_problems.append(sub_problem)
        
        # Create decomposition plan
        plan = DecompositionPlan(
            problem_statement=problem_statement,
            analyzed_context=analyzed_context,
            sub_problems=sub_problems
        )
        
        # Add OpenEvolve metrics to analyzed_context
        if 'openevolve_metrics' not in analyzed_context:
            analyzed_context['openevolve_metrics'] = {}
        analyzed_context['openevolve_metrics']['decomposition'] = result.get('metrics', {})
        
        return plan
        
    except Exception as e:
        st.error(f"Error using OpenEvolve for decomposition: {e}")
        # Fallback to standard decomposition
        return run_ai_decomposition(problem_statement, analyzed_context, team)


def run_assembly_with_openevolve(
    sub_problem_solutions: Dict[str, str],
    problem_statement: str,
    team: Team,
    api_key: str,
    model_name: str = "gpt-4o",
    max_iterations: int = 5
) -> str:
    """
    Run solution assembly using OpenEvolve evolution
    
    Args:
        sub_problem_solutions: Dictionary of sub-problem solutions
        problem_statement: Original problem statement
        team: Team to use for assembly
        api_key: API key for OpenEvolve
        model_name: Model to use
        max_iterations: Number of evolution iterations
        
    Returns:
        Assembled final solution
    """
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        
        # Create assembly prompt
        solutions_text = "\n\n".join([
            f"Sub-problem {sp_id}:\n{solution}"
            for sp_id, solution in sub_problem_solutions.items()
        ])
        
        assembly_prompt = f"""Assemble the following sub-problem solutions into a coherent final solution:

Original Problem: {problem_statement}

Sub-problem Solutions:
{solutions_text}

Provide a unified, coherent solution that integrates all sub-solutions.
"""
        
        # Run evolution
        result = client.evolve(
            content=assembly_prompt,
            evolution_mode="standard",
            max_iterations=max_iterations,
            population_size=10,
            temperature=0.7,
            model_name=model_name,
            content_type="text_general"
        )
        
        # Get best assembled solution
        final_solution = result.get('best_code', '')
        
        return final_solution
        
    except Exception as e:
        st.error(f"Error using OpenEvolve for assembly: {e}")
        # Fallback to standard assembly
        return run_assembly(sub_problem_solutions, problem_statement, team)
