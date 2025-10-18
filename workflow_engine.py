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

from ui_components import render_manual_review_panel
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

def _request_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None,
    stream: Optional[bool] = None,
    user: Optional[str] = None
) -> Optional[str]:
    """
    Makes a request to an OpenAI-compatible API endpoint for chat completions.

    Args:
        api_key (str): The API key for authentication.
        base_url (str): The base URL of the API endpoint.
        model (str): The model identifier to use for the completion.
        messages (List[Dict[str, str]]): A list of message dictionaries, typically in OpenAI chat format.
        extra_headers (Optional[Dict[str]]): Additional headers to include in the request.
        temperature (float): Controls randomness in model outputs.
        top_p (float): Nucleus sampling parameter.
        frequency_penalty (float): Penalizes new tokens based on their existing frequency.
        presence_penalty (float): Penalizes new tokens based on whether they appear in the text.
        max_tokens (int): Maximum number of tokens to generate.
        seed (Optional[int]): Seed for reproducible sampling.
        stop_sequences (Optional[List[str]]): Up to 4 sequences where the API will stop generating further tokens.
        logprobs (Optional[bool]): Whether to return log probabilities of the output tokens or not.
        top_logprobs (Optional[int]): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.
        response_format (Optional[Dict[str, str]]): An object specifying the format that the model must output.
        stream (Optional[bool]): If set, partial message deltas will be sent, like in ChatGPT.
        user (Optional[str]): A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.

    Returns:
        Optional[str]: The content of the generated message, or None if an error occurred.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop_sequences,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "stream": stream,
            "user": user
        }
        # Filter out None values to avoid sending them to the API if not specified
        completion_params = {k: v for k, v in completion_params.items() if v is not None}

        response = client.chat.completions.create(**completion_params)
        
        return response.choices[0].message.content
        
    except ImportError:
        # If openai package is not available, try using requests
        import requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop_sequences,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "stream": stream,
            "user": user
        }
        # Filter out None values
        data = {k: v for k, v in data.items() if v is not None}
            
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        st.error(f"Error making API request: {e}. Please check your API key, base URL, and network connection.")
        return None

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

    system_prompt = "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format."
    user_prompt_template = f"""Analyze the following problem statement and extract:
    - `domain`: (e.g., "Software Development", "Physics", "Legal")
    - `keywords`: List of important terms.
    - `estimated_complexity`: (1-10)
    - `potential_challenges`: List of anticipated difficulties.
    - `required_expertise`: List of expertise areas needed.
    - `summary`: A brief, concise summary of the problem.

    Problem Statement:
    ---
    {problem_statement}
    ---
    """

    def _analyze_with_model(model_config: ModelConfig):
        response = _request_openai_compatible_chat(
            api_key=model_config.api_key,
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(system_prompt, user_prompt_template),
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
            user=model_config.user
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

    # Combine analyses: For simplicity, concatenate summaries and keywords, take average complexity.
    combined_summary = " ".join([a.get("summary", "") for a in analyses if a.get("summary")])
    combined_keywords = list(set(kw for a in analyses for kw in a.get("keywords", [])))
    combined_challenges = list(set(c for a in analyses for c in a.get("potential_challenges", [])))
    combined_expertise = list(set(e for a in analyses for e in a.get("required_expertise", [])))
    
    # Average complexity, default to 5 if no valid complexities found
    complexities = [a.get("estimated_complexity", 0) for a in analyses if isinstance(a.get("estimated_complexity"), int) and 1 <= a.get("estimated_complexity") <= 10]
    avg_complexity = int(sum(complexities) / len(complexities)) if complexities else 5

    # Take the most frequent domain, or the first one if no clear majority
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

    system_prompt = "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects."
    user_prompt_template = f"""Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
    - `id`: A unique identifier (e.g., "sub_1.1")
    - `description`: A clear statement of the sub-problem.
    - `dependencies`: A list of `id`s of other sub-problems this one depends on.
    - `ai_suggested_evolution_mode`: Suggested evolution mode (e.g., "standard", "adversarial", "quality_diversity").
    - `ai_suggested_complexity_score`: An integer from 1 to 10.
    - `ai_suggested_evaluation_prompt`: A specific prompt for a Gold Team to evaluate this sub-problem's solution.

    Problem Statement:
    ---
    {problem_statement}
    ---

    Analyzed Context:
    ---
    {json.dumps(analyzed_context, indent=2)}
    ---

    Provide the output as a JSON array of sub-problem objects.
    """

    def _decompose_with_model(model_config: ModelConfig):
        response = _request_openai_compatible_chat(
            api_key=model_config.api_key,
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(system_prompt, user_prompt_template),
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
            user=model_config.user
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
    
    # For simplicity, return the first valid plan. In a more sophisticated implementation,
    # a synthesis or selection mechanism (e.g., a Blue Team Gauntlet for plan evaluation)
    # would be used to combine or choose the best plan from multiple candidates.
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
    st.info(f"Running Gauntlet '{gauntlet_def.name}' with Team '{team.name}'...")
    
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
            system_prompt = "You are a Blue Team AI acting as an internal quality assurance or peer reviewer. Your goal is to critically evaluate the provided content for its quality, correctness, and adherence to specified criteria. Provide your response as a JSON object with 'score' (0.0-1.0 for quality) and 'justification'."
            user_prompt_template = f"""Evaluate the following content. This content was generated internally by a Blue Team for a sub-problem.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            
            Based on your evaluation, provide a JSON object with a 'score' (0.0-1.0) for the content's quality and a 'justification' for your score.
            Consider the sub-problem's description and any relevant evolution parameters.
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
                messages=messages,
                temperature=member.temperature,
                top_p=member.top_p,
                frequency_penalty=member.frequency_penalty,
                presence_penalty=member.presence_penalty,
                max_tokens=member.max_tokens,
                seed=member.seed,
                stop_sequences=member.stop_sequences,
                logprobs=member.logprobs,
                top_logprobs=member.top_logprobs,
                response_format=member.response_format,
                stream=member.stream,
                user=member.user
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
            # st.info(f"  - Model: {member.model_id} evaluating...") # Moved outside for performance
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
        st.info("Awaiting user review and approval of the decomposition plan.")
        # Dynamically render the manual review panel and pause execution.
        # render_manual_review_panel returns the approved plan and a boolean indicating approval status.
        approved_plan, approved = render_manual_review_panel(workflow_state.decomposition_plan)

        if approved:
            # If the user approves, update the decomposition plan and transition to the next stage.
            workflow_state.decomposition_plan = approved_plan
            workflow_state.status = "running"
            workflow_state.current_stage = "Sub-Problem Solving Loop"
            st.rerun() # Rerun to immediately proceed to the next stage.
        elif approved is False: # Explicitly check for False to differentiate from None (not yet acted upon)
            # If the user explicitly rejects, mark the workflow as failed.
            workflow_state.status = "failed"
            st.error("Decomposition plan rejected by user. Workflow terminated.")
            # No return here, as openevolve_orchestrator.py handles the termination and rerun.
        else:
            # If the user has not yet approved or rejected (i.e., `approved` is None),
            # keep the status as `awaiting_user_input`.
            # This allows Streamlit to re-render the UI and wait for user interaction.
            workflow_state.status = "awaiting_user_input"
            # No return here, as openevolve_orchestrator.py handles the rerun.

    # If the workflow resumes after manual review, the plan should be approved and updated.
    # This block handles the transition if Streamlit reruns after user approval.
    if workflow_state.current_stage == "Sub-Problem Solving Loop" and workflow_state.decomposition_plan.sub_problems and workflow_state.status != "running":
        st.success(f"[Manual Review & Override] Decomposition plan approved. Resuming workflow.")
        workflow_state.status = "running" # Set status back to running to continue execution.

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
        assembler_system_message = f"You are an expert Assembler AI. Your task is to integrate multiple verified sub-problem solutions into a single, coherent, and high-quality final product. Ensure all dependencies are respected and the final output addresses the original problem statement."
        assembler_user_message = f"""Integrate the following verified sub-problem solutions into a single, coherent final product. The original problem statement was: {workflow_state.problem_statement}

        Verified Sub-Problem Solutions:
        ---
        {combined_solutions_input}
        ---

        Provide the complete, integrated final solution.
        """

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
                if isinstance(json_feedback, list):
                    problematic_ids.extend(json_feedback)
                # If it's a dictionary that might contain a list of problematic_sub_problems (for backward compatibility or alternative formats)
                elif isinstance(json_feedback, dict) and "problematic_sub_problems" in json_feedback:
                    problematic_ids.extend(json_feedback["problematic_sub_problems"])
                # Add other JSON structures if anticipated
            except json.JSONDecodeError:
                # Fallback to regex if not JSON or if JSON is not in expected list/dict format
                found_ids = re.findall(r'(sub_\d+\.\d+)', feedback)
                problematic_ids.extend(found_ids)
            
    return list(set(problematic_ids)) # Return unique IDs

from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config

def generate_solution_for_sub_problem(sub_problem: SubProblem, team: Team, context: Dict[str, Any], workflow_state: WorkflowState, solver_generation_gauntlet: Optional[GauntletDefinition] = None) -> str:
    """
    Generates a solution for a given sub-problem using the assigned solver team and OpenEvolve.
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

    if not team.members:
        st.error(f"Solver Team '{team.name}' has no members. Please configure the team in the Team Manager.")
        return "Failed to generate solution: No team members."

    model_config = team.members[0] # Use the first model in the team for generation

    # Construct OpenEvolve configuration
    system_message = f"You are an expert AI assistant tasked with solving sub-problem {sub_problem.id}. Generate a high-quality solution based on the description and context."
    feedback_json = None # Initialize feedback_json to None
    if "feedback_report" in context:
        feedback_report_obj = context["feedback_report"]
        if dataclasses.is_dataclass(feedback_report_obj):
            feedback_report_obj = dataclasses.asdict(feedback_report_obj)
        feedback_json = json.dumps(feedback_report_obj, indent=2)
        system_message += f"\n\nPrevious feedback for this sub-problem:\n---\n{feedback_json}\n---\nAddress the issues raised in this feedback to improve the solution."

    generated_solution_content = "" # Initialize here
    
    # If evolution_params are provided, always use the full OpenEvolve unified evolution process.
    if sub_problem.evolution_params:
            st.info(f"  - Using OpenEvolve's `run_unified_evolution` for {sub_problem.id} due to provided `evolution_params`...")
            
            # Get a base configuration from the workflow_state's parameters
            base_evolution_args = create_comprehensive_openevolve_config(
                content=sub_problem.description,
                content_type=workflow_state.decomposition_plan.analyzed_context.get("content_type", "text_general"), # Use content_type from analyzed_context or default
                evolution_mode=sub_problem.ai_suggested_evolution_mode,
                model_configs=[{"name": model_config.model_id, "weight": 1.0}],
                api_key=model_config.api_key,
                api_base=model_config.api_base,
                temperature=model_config.temperature,
                top_p=model_config.top_p, # Use model_config's top_p
                max_tokens=model_config.max_tokens,
                frequency_penalty=model_config.frequency_penalty,
                presence_penalty=model_config.presence_penalty,
                seed=model_config.seed,
                stop_sequences=model_config.stop_sequences, # Pass new ModelConfig parameter
                logprobs=model_config.logprobs, # Pass new ModelConfig parameter
                top_logprobs=model_config.top_logprobs, # Pass new ModelConfig parameter
                response_format=model_config.response_format, # Pass new ModelConfig parameter
                stream=model_config.stream, # Pass new ModelConfig parameter
                user=model_config.user, # Pass new ModelConfig parameter
                system_message=system_message,
                evaluator_system_message=sub_problem.ai_suggested_evaluation_prompt, # Use sub-problem's evaluation prompt as evaluator system message
                
                # Parameters from workflow_state (or its decomposition_plan)
                max_iterations=workflow_state.decomposition_plan.max_refinement_loops * 2, # Example: scale with refinement loops
                population_size=100, # Default, can be overridden by evolution_params
                num_islands=5, # Default
                migration_interval=50, # Default
                migration_rate=0.1, # Default
                archive_size=100, # Default
                elite_ratio=0.1, # Default
                exploration_ratio=0.2, # Default
                exploitation_ratio=0.7, # Default
                checkpoint_interval=100, # Default
                feature_dimensions=workflow_state.decomposition_plan.analyzed_context.get("feature_dimensions", ["complexity", "diversity"]),
                feature_bins=10, # Default
                diversity_metric="edit_distance", # Default
                
                enable_artifacts=True, # Default
                cascade_evaluation=True, # Default
                cascade_thresholds=[0.5, 0.75, 0.9], # Default
                use_llm_feedback=True, # Default
                llm_feedback_weight=0.1, # Default
                parallel_evaluations=4, # Default
                distributed=False, # Default
                template_dir=None, # Default
                num_top_programs=3, # Default
                num_diverse_programs=2, # Default
                use_template_stochasticity=True, # Default
                template_variations=None, # Default
                use_meta_prompting=False, # Default
                meta_prompt_weight=0.1, # Default
                include_artifacts=True, # Default
                max_artifact_bytes=20 * 1024, # Default
                artifact_security_filter=True, # Default
                early_stopping_patience=None, # Default
                convergence_threshold=0.001, # Default
                early_stopping_metric="combined_score", # Default
                memory_limit_mb=2048, # Default
                cpu_limit=4.0, # Default
                random_seed=42, # Default
                db_path=None, # Default
                in_memory=True, # Default
                
                diff_based_evolution=True, # Default
                max_code_length=10000, # Default
                evolution_trace_enabled=False, # Default
                evolution_trace_format="jsonl", # Default
                evolution_trace_include_code=False, # Default
                evolution_trace_include_prompts=True, # Default
                evolution_trace_output_path=None, # Default
                evolution_trace_buffer_size=10, # Default
                evolution_trace_compress=False, # Default
                log_level="INFO", # Default
                log_dir=None, # Default
                api_timeout=60, # Default
                api_retries=3, # Default
                api_retry_delay=5, # Default
                artifact_size_threshold=32 * 1024, # Default
                cleanup_old_artifacts=True, # Default
                artifact_retention_days=30, # Default
                diversity_reference_size=20, # Default
                max_retries_eval=3, # Default
                evaluator_timeout=300, # Default
                evaluator_models=None, # Default
                
                double_selection=True, # Default
                adaptive_feature_dimensions=True, # Default
                test_time_compute=False, # Default
                optillm_integration=False, # Default
                plugin_system=False, # Default
                hardware_optimization=False, # Default
                multi_strategy_sampling=True, # Default
                ring_topology=True, # Default
                controlled_gene_flow=True, # Default
                auto_diff=True, # Default
                symbolic_execution=False, # Default
                coevolutionary_approach=False, # Default
            )
    
            # Override base configuration with sub_problem.evolution_params
            # Ensure that the output_dir is handled correctly, as it's a required parameter for run_unified_evolution
            output_dir = os.path.join(os.getcwd(), "openevolve_checkpoints", workflow_state.workflow_id, sub_problem.id)
            os.makedirs(output_dir, exist_ok=True)
            
            evolution_args = base_evolution_args.copy() if base_evolution_args else {}
            evolution_args.update(sub_problem.evolution_params)
            evolution_args["output_dir"] = output_dir # Ensure output_dir is set

            try:
                result = run_unified_evolution(**evolution_args)
                if result and result.get("success") and result.get("best_solution"):
                    st.success(f"Solution generated for {sub_problem.id} by OpenEvolve.")
                    generated_solution_content = result["best_solution"]
                else:
                    st.error(f"OpenEvolve failed to generate a solution for {sub_problem.id}. Result: {result}")
                    return "Failed to generate solution: OpenEvolve did not return a valid solution."
            except Exception as e:
                st.error(f"Error running OpenEvolve for sub-problem {sub_problem.id}: {e}")
                return f"Failed to generate solution: OpenEvolve execution error: {e}"
    # Handle generation via Blue Team Gauntlet modes (multi-candidate or single direct call).
    elif solver_generation_gauntlet and solver_generation_gauntlet.generation_mode: # Added check for None and generation_mode
        # Single Candidate Generation: A single model directly generates the solution.
        if solver_generation_gauntlet.generation_mode == "single_candidate":
            st.info(f"  - Using single_candidate generation mode for {sub_problem.id}...")
            user_prompt = f"""Solve the following sub-problem:
            ---
            {sub_problem.description}
            ---
            
            Context from overall problem:
            ---
            {json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2)}
            ---
            
            {"Existing solution to refine:\n---\n" + context["current_solution"] + "\n---" if "current_solution" in context and context["current_solution"] else ""}
            
            Provide the solution directly.
            """
            
            response = _request_openai_compatible_chat(
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                model=model_config.model_id,
                messages=_compose_messages(system_message, user_prompt),
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
                user=model_config.user
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
                candidate_system_message = f"You are an AI assistant tasked with generating a candidate solution for sub-problem {sub_problem.id}. Your goal is to provide a unique and high-quality approach."
                if "feedback_report" in context:
                    candidate_system_message += f"\n\nPrevious feedback for this sub-problem:\n---\n{feedback_json}\n---\nAddress the issues raised in this feedback to improve the solution."

                candidate_user_prompt = f"""Generate a candidate solution for the following sub-problem:
                ---
                {sub_problem.description}
                ---
                
                Context from overall problem:
                ---
                {json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2)}
                ---
                
                {"Existing solution to refine:\n---\n" + context["current_solution"] + "\n---" if "current_solution" in context and context["current_solution"] else ""}
                
                Provide the candidate solution directly.
                """
                
                candidate_response = _request_openai_compatible_chat(
                    api_key=member.api_key,
                    base_url=member.api_base,
                    model=member.model_id,
                    messages=_compose_messages(candidate_system_message, candidate_user_prompt),
                    temperature=member.temperature + (i * 0.1), # Slightly vary temperature for diversity in candidates.
                    top_p=member.top_p,
                    frequency_penalty=member.frequency_penalty,
                    presence_penalty=member.presence_penalty,
                    max_tokens=member.max_tokens,
                    seed=member.seed,
                    stop_sequences=member.stop_sequences,
                    logprobs=member.logprobs,
                    top_logprobs=member.top_logprobs,
                    response_format=member.response_format,
                    stream=member.stream,
                    user=member.user
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
            review_system_message = f"You are an expert AI peer reviewer and synthesizer. Your task is to review multiple candidate solutions for sub-problem {sub_problem.id} and synthesize the best possible solution, incorporating the strengths of each and addressing any weaknesses. If a single candidate is clearly superior, you may select it. Otherwise, combine and refine."
            review_user_prompt = f"""Review the following candidate solutions for sub-problem {sub_problem.id} and synthesize the best possible solution.
            
            Sub-problem Description:
            ---
            {sub_problem.description}
            ---
            
            Context from overall problem:
            ---
            {json.dumps(workflow_state.decomposition_plan.analyzed_context, indent=2)}
            ---
            
            {"Existing solution to refine:\n---\n" + context["current_solution"] + "\n---" if "current_solution" in context and context["current_solution"] else ""}
            
            Candidate Solutions:
            ---
            {json.dumps(candidates, indent=2)}
            ---
            
            Provide the synthesized best solution directly.
            """
            
            synthesized_response = _request_openai_compatible_chat(
                api_key=model_config.api_key, # Use the primary model for synthesis.
                base_url=model_config.api_base,
                model=model_config.model_id,
                messages=_compose_messages(review_system_message, review_user_prompt),
                temperature=0.5, # Lower temperature for more deterministic synthesis.
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
                user=model_config.user
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
