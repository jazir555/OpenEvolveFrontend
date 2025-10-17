import streamlit as st
import time
import json
import os
import re
import dataclasses # Added this line
from typing import List, Dict, Any, Optional, Callable
from workflow_structures import (
    ModelConfig, Team, GauntletRoundRule, GauntletDefinition,
    SubProblem, DecompositionPlan, SolutionAttempt, CritiqueReport,
    VerificationReport, WorkflowState
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config

# Initialize managers (assuming they are initialized in ui_components or main app)
# For standalone testing, uncomment these:
# team_manager = TeamManager()
# gauntlet_manager = GauntletManager()

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
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Make a request to an OpenAI-compatible API.
    This function is a copy from evolution.py, adapted for workflow_engine.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed
        )
        
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
            "max_tokens": max_tokens
        }
        
        if seed is not None:
            data["seed"] = seed
            
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        st.error(f"Error making API request: {e}")
        return None

def _compose_messages(system_message: str, user_message: str) -> List[Dict[str, str]]:
    """Helper to compose messages for LLM API calls."""
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

# --- Core Workflow Engine Functions ---

def run_content_analysis(problem_statement: str, team: Team) -> Dict[str, Any]:
    """
    Stage 0: Content Analysis.
    Uses a Blue Team model to analyze the problem statement and extract context.
    """
    if not team.members:
        st.error(f"Content Analysis Team '{team.name}' has no members.")
        return {"error": "No team members"}

    # For simplicity, use the first model in the team
    model_config = team.members[0]
    
    system_prompt = "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format."
    user_prompt = f"""Analyze the following problem statement and extract:
    - `domain`: (e.g., "Software Development", "Physics", "Legal")
    - `keywords`: List of important terms.
    - `estimated_complexity`: (1-10)
    - `potential_challenges`: List of anticipated difficulties.
    - `required_expertise`: List of expertise areas needed.
    - `summary`: A brief summary of the problem.

    Problem Statement:
    ---
    {problem_statement}
    ---
    """
    
    response = _request_openai_compatible_chat(
        api_key=model_config.api_key,
        base_url=model_config.api_base,
        model=model_config.model_id,
        messages=_compose_messages(system_prompt, user_prompt),
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens
    )
    
    if response:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            st.warning(f"Content Analysis response was not valid JSON: {response[:200]}...")
            return {"summary": response} # Fallback
    return {"summary": "Failed to analyze content."}

def run_ai_decomposition(problem_statement: str, analyzed_context: Dict[str, Any], team: Team) -> DecompositionPlan:
    """
    Stage 1: AI-Assisted Decomposition.
    Uses a Blue Team (Planners) to generate a decomposition plan.
    """
    if not team.members:
        st.error(f"Decomposition Team '{team.name}' has no members.")
        return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=[])

    model_config = team.members[0] # Use the first model for now

    system_prompt = "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects."
    user_prompt = f"""Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
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

    response = _request_openai_compatible_chat(
        api_key=model_config.api_key,
        base_url=model_config.api_base,
        model=model_config.model_id,
        messages=_compose_messages(system_prompt, user_prompt),
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens
    )

    if response:
        try:
            sub_problems_data = json.loads(response)
            sub_problems = [SubProblem(**sp) for sp in sub_problems_data]
            return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=sub_problems)
        except json.JSONDecodeError:
            st.error(f"AI Decomposition response was not valid JSON: {response[:500]}...")
            return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=[])
    
    st.error("Failed to get AI decomposition plan.")
    return DecompositionPlan(problem_statement=problem_statement, analyzed_context=analyzed_context, sub_problems=[])

import statistics # Need to import this for variance calculation

def run_gauntlet(
    solution_content: str,
    gauntlet_def: GauntletDefinition,
    team: Team,
    context: Dict[str, Any] # Additional context for LLM prompts, e.g., sub_problem details
) -> Dict[str, Any]:
    """
    Executes a Gauntlet with a given Team.
    Returns a dictionary with 'is_approved', 'report_summary', 'critique_report' or 'verification_report'.
    """
    st.info(f"Running Gauntlet '{gauntlet_def.name}' with Team '{team.name}'...")
    
    all_judge_reports = []
    overall_gauntlet_approved = True
    
    # Track successful rounds per judge for per-judge approval counts
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
            system_prompt = "You are a Red Team AI. Your goal is to find flaws, vulnerabilities, and weaknesses in the provided content. If you find a flaw, explain it clearly. If not, state that the content appears robust. Provide your response as a JSON object with 'score' (0.0-1.0 for robustness), 'justification' (string), and 'targeted_feedback' (string, if applicable, mentioning specific sub-problem IDs like 'sub_1.2' that are faulty)."
            user_prompt_template = f"""Critique the following content for flaws and vulnerabilities.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            Attack Modes: {', '.join(gauntlet_def.attack_modes) if gauntlet_def.attack_modes else 'General Vulnerability Scan'}
            
            Provide your critique as a JSON object with 'score', 'justification', and 'targeted_feedback'.
            If the failure is traceable to specific sub-problems, list their IDs in the 'targeted_feedback' field, e.g., 'Fault in sub_1.2 and sub_2.1'.
            """
        elif team.role == "Gold":
            system_prompt = "You are a Gold Team AI. Your goal is to impartially evaluate the provided content for correctness, quality, and adherence to requirements. Provide your response as a JSON object with 'score' (0.0-1.0), 'justification' (string), and 'targeted_feedback' (string, if applicable, mentioning specific sub-problem IDs like 'sub_1.2' that are faulty)."
            user_prompt_template = f"""Evaluate the following content for correctness and quality.
            Context: {json.dumps(serializable_context, indent=2)}
            Content:
            ---
            {{content}}
            ---
            Evaluation Prompt: {context.get('evaluation_prompt', 'Evaluate for overall quality and correctness.')}
            
            Provide your evaluation as a JSON object with 'score', 'justification', and 'targeted_feedback'.
            If the evaluation fails and you can trace it to specific sub-problems, list their IDs in the 'targeted_feedback' field, e.g., 'Integration error between sub_1.1 and sub_1.3'.
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
        if round_rule.collaboration_mode == "share_previous_feedback" and all_judge_reports:
            previous_feedback = "\n".join([f"Model {r['model_id']}: {r['justification']} (Score: {r['score']})" for r in all_judge_reports[-len(team.members):]]) # Last round's feedback
            user_prompt_template += f"\n\nPrevious round's feedback:\n---\n{previous_feedback}\n---"

        # Invoke each member of the team
        for member in team.members:
            st.info(f"  - Model: {member.model_id} evaluating...")
            
            # Get per-judge requirements for this round
            per_judge_req = round_rule.per_judge_requirements.get(member.model_id, {})
            min_score_for_judge = per_judge_req.get('min_score', round_rule.min_overall_confidence)

            messages = _compose_messages(system_prompt, user_prompt_template.replace("{content}", solution_content))
            
            response_content = _request_openai_compatible_chat(
                api_key=member.api_key,
                base_url=member.api_base,
                model=member.model_id,
                messages=messages,
                temperature=member.temperature,
                max_tokens=member.max_tokens
            )

            judge_score = 0.0
            justification = "No response or invalid format."
            targeted_feedback = ""
            
            if response_content:
                # Attempt to parse JSON response for structured feedback
                try:
                    parsed_response = json.loads(response_content)
                    judge_score = parsed_response.get("score", 0.0)
                    justification = parsed_response.get("justification", response_content)
                    targeted_feedback = parsed_response.get("targeted_feedback", "")
                except json.JSONDecodeError:
                    # Fallback to regex if not JSON
                    score_match = re.search(r"score:\s*(\d+\.?\d*)", response_content, re.IGNORECASE)
                    if score_match:
                        judge_score = float(score_match.group(1))
                        if judge_score > 1.0: judge_score /= 100.0 # Normalize if out of 100
                    justification = response_content
            
            st.write(f"    - {member.model_id} Score: {judge_score:.2f} (Required: {min_score_for_judge:.2f})")
            st.caption(f"      Justification: {justification[:100]}...")

            judge_passed_this_round = False
            if team.role == "Red":
                # For Red Team, a low score (finding a flaw) means it FAILED the solution, but PASSED its role as a Red Team member.
                # The gauntlet is approved if the Red Team *fails* to find a flaw (i.e., score is high).
                if judge_score >= min_score_for_judge: # Red Team found no significant flaw
                    judge_passed_this_round = True
            else: # Blue or Gold Team
                if judge_score >= min_score_for_judge:
                    judge_passed_this_round = True
            
            if judge_passed_this_round:
                round_approved_count += 1
                successful_rounds_per_judge[member.model_id] += 1
            
            current_round_scores.append(judge_score)
            current_round_judge_reports.append({
                "model_id": member.model_id,
                "score": judge_score,
                "justification": justification,
                "targeted_feedback": targeted_feedback,
                "passed_round": judge_passed_this_round
            })
        
        # --- Evaluate Round Success ---
        round_passed = True
        
        # 1. Check Quorum
        if round_rule.quorum_required_approvals > round_approved_count:
            st.warning(f"  - Round {round_rule.round_number} failed: Quorum not met ({round_approved_count}/{round_rule.quorum_required_approvals} approvals).")
            round_passed = False
        
        # 2. Check Minimum Overall Confidence
        if current_round_scores and statistics.mean(current_round_scores) < round_rule.min_overall_confidence:
            st.warning(f"  - Round {round_rule.round_number} failed: Average score ({statistics.mean(current_round_scores):.2f}) below minimum overall confidence ({round_rule.min_overall_confidence:.2f}).")
            round_passed = False
            
        # 3. Check Max Score Variance (if configured)
        if round_rule.max_score_variance is not None and len(current_round_scores) > 1:
            current_variance = statistics.variance(current_round_scores)
            if current_variance > round_rule.max_score_variance:
                st.warning(f"  - Round {round_rule.round_number} failed: Score variance ({current_variance:.2f}) above maximum allowed ({round_rule.max_score_variance:.2f}).")
                round_passed = False
        
        if not round_passed:
            overall_gauntlet_approved = False
            break # Gauntlet failed, no need to continue to next rounds
        else:
            st.success(f"  - Round {round_rule.round_number} passed.")
            all_judge_reports.extend(current_round_judge_reports) # Collect reports from successful rounds

    # --- Final Gauntlet Approval Check (Per-Judge Approval Counts) ---
    if overall_gauntlet_approved:
        for member in team.members:
            per_judge_req = {} # Need to get this from GauntletDefinition if it's a global setting
            # For now, assume per-judge approval counts are handled within round_rule or not implemented yet
            # This part needs more thought on how per-judge approval counts across rounds are defined in GauntletDefinition
            
            # Placeholder for per-judge approval counts across rounds
            # if successful_rounds_per_judge[member.model_id] < per_judge_req.get('required_successful_rounds', len(gauntlet_def.rounds)):
            #     st.warning(f"Gauntlet '{gauntlet_def.name}' failed: Model {member.model_id} did not meet its required successful rounds.")
            #     overall_gauntlet_approved = False
            #     break
            pass

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
    
    max_refinement_loops: int = 3
):
    st.info(f"Starting Sovereign-Grade Workflow: {workflow_state.workflow_id}")
    workflow_state.status = "running"
    
    team_manager = TeamManager() # Initialize team_manager

    # --- Stage 0: Content Analysis ---
    if workflow_state.current_stage == "INITIALIZING" or workflow_state.current_stage == "Content Analysis":
        workflow_state.current_stage = "Content Analysis"
        st.info(f"[{workflow_state.current_stage}] Analyzing problem statement...")
        analyzed_context = run_content_analysis(workflow_state.problem_statement, content_analyzer_team)
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
        workflow_state.current_stage = "AI-Assisted Decomposition" # Move to next stage
        workflow_state.progress = 0.2 # Update progress after Stage 0

    # --- Stage 1: AI-Assisted Decomposition ---
    if workflow_state.current_stage == "AI-Assisted Decomposition":
        st.info(f"[{workflow_state.current_stage}] Generating decomposition plan...")
        decomposition_plan = run_ai_decomposition(
            workflow_state.problem_statement,
            workflow_state.decomposition_plan.analyzed_context,
            planner_team
        )
        workflow_state.decomposition_plan.sub_problems = decomposition_plan.sub_problems
        st.success(f"[{workflow_state.current_stage}] Decomposition plan generated.")
        workflow_state.current_stage = "Manual Review & Override" # Move to next stage
        workflow_state.progress = 0.4 # Update progress after Stage 1

    # --- Stage 2: Manual Review & Override ---
    if workflow_state.current_stage == "Manual Review & Override":
        st.info(f"[{workflow_state.current_stage}] Awaiting user approval for decomposition plan...")
        workflow_state.status = "awaiting_user_input"
        return # Pause execution and wait for user input via UI

    # If the workflow resumes after manual review, the plan should be approved and updated
    if workflow_state.current_stage == "Sub-Problem Solving Loop" and workflow_state.decomposition_plan.sub_problems and workflow_state.status != "running":
        # This means the user has approved the plan in the UI and the workflow is resuming
        st.success(f"[Manual Review & Override] Decomposition plan approved. Resuming workflow.")
        workflow_state.status = "running" # Set status back to running to continue execution

    # --- Stage 3: Sub-Problem Solving Loop ---
    if workflow_state.current_stage == "Sub-Problem Solving Loop":
        st.info(f"[{workflow_state.current_stage}] Starting sub-problem solving...")
        
        # Correctly implement topological sort for sub-problems
        sub_problems_by_id = {sp.id: sp for sp in workflow_state.decomposition_plan.sub_problems}
        
        # Validate dependencies and calculate in-degrees
        in_degree = {sp_id: 0 for sp_id in sub_problems_by_id}
        adj = {sp_id: [] for sp_id in sub_problems_by_id}
        
        for sp_id, sp in sub_problems_by_id.items():
            for dep_id in sp.dependencies:
                if dep_id in sub_problems_by_id:
                    adj[dep_id].append(sp_id)
                    in_degree[sp_id] += 1
                else:
                    st.error(f"Sub-problem '{sp_id}' has an invalid dependency: '{dep_id}'. Workflow failed.")
                    workflow_state.status = "failed"
                    return
        
        # Initialize queue with sub-problems that have no dependencies and are not yet solved
        queue = [sp_id for sp_id, degree in in_degree.items() if degree == 0 and sp_id not in workflow_state.solved_sub_problem_ids]
        
        if not queue and len(workflow_state.solved_sub_problem_ids) < len(workflow_state.decomposition_plan.sub_problems):
            st.error("Circular dependency detected or no solvable sub-problems initially. Workflow failed.")
            workflow_state.status = "failed"
            return

        processed_this_iteration = set()

        while queue:
            current_sp_id = queue.pop(0)
            current_sub_problem = sub_problems_by_id.get(current_sp_id)
            
            if not current_sub_problem:
                st.error(f"Sub-problem {current_sp_id} not found in decomposition plan. Skipping.")
                continue

            if current_sp_id in workflow_state.solved_sub_problem_ids:
                continue # Already solved

            workflow_state.current_sub_problem_id = current_sp_id
            st.info(f"[{workflow_state.current_stage}] Solving sub-problem: {current_sp_id} - {current_sub_problem.description[:50]}...")
            
            generated_content = ""
            if current_sp_id in workflow_state.sub_problem_solutions:
                generated_content = workflow_state.sub_problem_solutions[current_sp_id].content

            # If a solution exists and was rejected, use the Patcher Team
            if current_sp_id in workflow_state.rejected_sub_problems:
                st.info(f"  - Invoking Patcher Team for {current_sp_id} based on previous rejection.")
                last_report = workflow_state.rejected_sub_problems[current_sp_id]
                generated_content = generate_solution_for_sub_problem(
                    current_sub_problem, 
                    patcher_team, 
                    {"current_solution": generated_content, "feedback_report": last_report},
                    workflow_state.solver_generation_gauntlet
                )
                del workflow_state.rejected_sub_problems[current_sp_id] # Clear rejection status
            else:
                generated_content = generate_solution_for_sub_problem(current_sub_problem, solver_team, {"current_solution": generated_content}, workflow_state.solver_generation_gauntlet)
            
            if generated_content.startswith("Failed to generate solution:"):
                st.error(f"Failed to generate solution for sub-problem {current_sp_id}. Workflow failed.")
                workflow_state.status = "failed"
                return # Halt workflow execution

            solution_attempt = SolutionAttempt(
                sub_problem_id=current_sp_id,
                content=generated_content,
                generated_by_model=solver_team.members[0].model_id, # Assuming first member
                timestamp=time.time()
            )
            
            # --- Step B: Red Team Gauntlet ---
            workflow_state.current_gauntlet_name = sub_problem_red_gauntlet.name
            st.info(f"  - Running Red Team Gauntlet for {current_sp_id}...")
            red_gauntlet_result = run_gauntlet(
                solution_attempt.content,
                sub_problem_red_gauntlet,
                team_manager.get_team(sub_problem_red_gauntlet.team_name), # Use the assigned Red Team
                {"sub_problem": current_sub_problem, "solution_id": solution_attempt.sub_problem_id}
            )
            workflow_state.all_critique_reports.append(red_gauntlet_result['critique_report'])
            st.info("INFO: Red team gauntlet finished.")
            
            if not red_gauntlet_result['is_approved']:
                st.warning(f"  - Red Team rejected solution for {current_sp_id}. Marking for rework.")
                workflow_state.rejected_sub_problems[current_sp_id] = red_gauntlet_result['critique_report']
                # Re-add to queue if it needs to be re-processed after patching
                queue.append(current_sp_id) # This will cause it to be re-evaluated
                continue # Skip Gold Team and next dependencies for this sub-problem

            st.info("INFO: About to run gold team gauntlet.")
            # --- Step C: Gold Team Gauntlet ---
            workflow_state.current_gauntlet_name = sub_problem_gold_gauntlet.name
            st.info(f"  - Running Gold Team Gauntlet for {current_sp_id}...")
            gold_gauntlet_result = run_gauntlet(
                solution_attempt.content,
                sub_problem_gold_gauntlet,
                team_manager.get_team(sub_problem_gold_gauntlet.team_name), # Use the assigned Gold Team
                {"sub_problem": current_sub_problem, "solution_id": solution_attempt.sub_problem_id}
            )
            workflow_state.all_verification_reports.append(gold_gauntlet_result['verification_report'])

            if not gold_gauntlet_result['is_approved']:
                st.warning(f"  - Gold Team rejected solution for {current_sp_id}. Marking for rework.")
                workflow_state.rejected_sub_problems[current_sp_id] = gold_gauntlet_result['verification_report']
                # Re-add to queue if it needs to be re-processed after patching
                queue.append(current_sp_id) # This will cause it to be re-evaluated
                continue # Skip next dependencies for this sub-problem
            
            workflow_state.sub_problem_solutions[current_sp_id] = solution_attempt
            workflow_state.solved_sub_problem_ids.add(current_sp_id)
            processed_this_iteration.add(current_sp_id)
            st.success(f"[{workflow_state.current_stage}] Sub-problem {current_sp_id} solved and verified.")

            # Update overall progress based on solved sub-problems
            workflow_state.progress = 0.4 + (0.3 * (len(workflow_state.solved_sub_problem_ids) / len(workflow_state.decomposition_plan.sub_problems))) # Stage 3 is 40-70%

            # Update in-degrees of dependent sub-problems correctly
            for dependent_sp_id in adj[current_sp_id]:
                in_degree[dependent_sp_id] -= 1
                if in_degree[dependent_sp_id] == 0 and dependent_sp_id not in workflow_state.solved_sub_problem_ids:
                    queue.append(dependent_sp_id)
            
        if len(workflow_state.solved_sub_problem_ids) < len(workflow_state.decomposition_plan.sub_problems):
            st.error("Could not solve all sub-problems. Possible circular dependency or unsolvable problem. Workflow failed.")
            workflow_state.status = "failed"
            return

        st.success(f"[{workflow_state.current_stage}] All sub-problems solved.")
        workflow_state.current_stage = "Configurable Reassembly" # Move to next stage
        workflow_state.progress = 0.7 # Update progress after Stage 3

    # --- Stage 4: Configurable Reassembly ---
    if workflow_state.current_stage == "Configurable Reassembly":
        st.info(f"[{workflow_state.current_stage}] Reassembling final solution using {assembler_team.name} via OpenEvolve...")
        
        # Collect all verified sub-problem solutions
        verified_solutions_content = []
        for sp in workflow_state.decomposition_plan.sub_problems:
            if sp.id in workflow_state.sub_problem_solutions:
                verified_solutions_content.append(f"### Sub-Problem {sp.id}\n{workflow_state.sub_problem_solutions[sp.id].content}")
        
        combined_solutions_input = "\n\n".join(verified_solutions_content)

        if not assembler_team.members:
            st.error(f"Assembler Team '{assembler_team.name}' has no members.")
            workflow_state.status = "failed"
            return

        model_config = assembler_team.members[0] # Use the first model in the team

        # Construct prompt for the Assembler Team
        assembler_system_message = f"You are an expert Assembler AI. Your task is to integrate multiple verified sub-problem solutions into a single, coherent, and high-quality final product. Ensure all dependencies are respected and the final output addresses the original problem statement."
        assembler_user_message = f"""Integrate the following verified sub-problem solutions into a single, coherent final product. The original problem statement was: {workflow_state.problem_statement}

        Verified Sub-Problem Solutions:
        ---
        {combined_solutions_input}
        ---

        Provide the complete, integrated final solution.
        """

        # Construct arguments for OpenEvolve reassembly
        evolution_args = {
            "content": assembler_user_message,
            "content_type": "text_general",
            "evolution_mode": "standard",
            "model_configs": [{"name": model_config.model_id, "weight": 1.0}],
            "api_key": model_config.api_key,
            "api_base": model_config.api_base,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
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
        
        final_solution_attempt = SolutionAttempt(
            sub_problem_id="final_solution",
            content=final_solution_content,
            generated_by_model=assembler_team.members[0].model_id,
            timestamp=time.time()
        )
        workflow_state.final_solution = final_solution_attempt
        st.success(f"[{workflow_state.current_stage}] Final solution reassembled.")
        workflow_state.current_stage = "Final Verification & Self-Healing Loop"
        workflow_state.progress = 0.9


    # --- Stage 5: Final Verification & Self-Healing Loop ---
    if workflow_state.current_stage == "Final Verification & Self-Healing Loop":
        st.info(f"[{workflow_state.current_stage}] Starting final verification...")
        
        while workflow_state.refinement_loop_count <= workflow_state.max_refinement_loops:
            st.subheader(f"Final Verification Loop: {workflow_state.refinement_loop_count + 1}/{workflow_state.max_refinement_loops + 1}")
            
            # Final Red Team Gauntlet
            workflow_state.current_gauntlet_name = final_red_gauntlet.name
            st.info(f"  - Running Final Red Team Gauntlet...")
            final_red_gauntlet_result = run_gauntlet(
                workflow_state.final_solution.content,
                final_red_gauntlet,
                team_manager.get_team(final_red_gauntlet.team_name), # Use the assigned Red Team
                {"final_solution": workflow_state.final_solution}
            )
            workflow_state.all_critique_reports.append(final_red_gauntlet_result['critique_report'])

            if not final_red_gauntlet_result['is_approved']:
                st.warning(f"  - Final Red Team rejected solution. Initiating self-healing.")
                # Parse feedback and identify problematic sub-problems
                problematic_sub_problem_ids = parse_targeted_feedback(final_red_gauntlet_result['critique_report'])
                if not problematic_sub_problem_ids:
                    st.error("  - Red Team rejected, but no specific problematic sub-problems identified. Cannot self-heal.")
                    workflow_state.status = "failed"
                    return

                st.info(f"  - Problematic sub-problems identified: {', '.join(problematic_sub_problem_ids)}. Re-queuing for re-solve.")
                # Clear solutions for problematic sub-problems to force re-solve
                for sp_id in problematic_sub_problem_ids:
                    if sp_id in workflow_state.sub_problem_solutions:
                        del workflow_state.sub_problem_solutions[sp_id]
                        workflow_state.rejected_sub_problems[sp_id] = final_red_gauntlet_result['critique_report'] # Store report for patcher
                
                workflow_state.refinement_loop_count += 1
                if workflow_state.refinement_loop_count > workflow_state.max_refinement_loops:
                    st.error("Max refinement loops reached for final solution. Manual intervention required.")
                    workflow_state.status = "failed"
                    return
                
                workflow_state.current_stage = "Sub-Problem Solving Loop" # Go back to solve problematic sub-problems
                return # Exit current run, Streamlit will re-run and continue from Stage 3

            # Final Gold Team Gauntlet
            workflow_state.current_gauntlet_name = final_gold_gauntlet.name
            st.info(f"  - Running Final Gold Team Gauntlet...")
            final_gold_gauntlet_result = run_gauntlet(
                workflow_state.final_solution.content,
                final_gold_gauntlet,
                team_manager.get_team(final_gold_gauntlet.team_name), # Use the assigned Gold Team
                {"final_solution": workflow_state.final_solution}
            )
            workflow_state.all_verification_reports.append(final_gold_gauntlet_result['verification_report'])

            if not final_gold_gauntlet_result['is_approved']:
                st.warning(f"  - Final Gold Team rejected solution. Initiating self-healing.")
                problematic_sub_problem_ids = parse_targeted_feedback(final_gold_gauntlet_result['verification_report'])
                if not problematic_sub_problem_ids:
                    st.error("  - Gold Team rejected, but no specific problematic sub-problems identified. Cannot self-heal.")
                    workflow_state.status = "failed"
                    return

                st.info(f"  - Problematic sub-problems identified: {', '.join(problematic_sub_problem_ids)}. Re-queuing for re-solve.")
                for sp_id in problematic_sub_problem_ids:
                    if sp_id in workflow_state.sub_problem_solutions:
                        del workflow_state.sub_problem_solutions[sp_id]
                        workflow_state.rejected_sub_problems[sp_id] = final_gold_gauntlet_result['verification_report'] # Store report for patcher
                
                workflow_state.refinement_loop_count += 1
                if workflow_state.refinement_loop_count > workflow_state.max_refinement_loops:
                    st.error("Max refinement loops reached for final solution. Manual intervention required.")
                    workflow_state.status = "failed"
                    return
                
                workflow_state.current_stage = "Sub-Problem Solving Loop" # Go back to solve problematic sub-problems
                return # Exit current run, Streamlit will re-run and continue from Stage 3
            
            # If both final gauntlets pass
            st.success(f"[{workflow_state.current_stage}] Final solution verified. Workflow completed successfully!")
            workflow_state.status = "completed"
            workflow_state.end_time = time.time()
            workflow_state.progress = 1.0
            st.info("INFO: Workflow completed.")
            return # Workflow completed

        st.error("Max refinement loops reached for final solution. Manual intervention required.")
        workflow_state.status = "failed"
        workflow_state.end_time = time.time()
        st.info("INFO: Workflow failed.")

def parse_targeted_feedback(report: Any) -> List[str]: # Changed type hint to Any
    """
    Parses a critique or verification report to identify problematic sub-problem IDs.
    Prioritizes JSON parsing, falls back to regex.
    """
    problematic_ids = []
    
    # Convert report to dict if it's a dataclass
    if dataclasses.is_dataclass(report):
        report = dataclasses.asdict(report)

    for judge_report in report['reports_by_judge']:
        feedback = judge_report.get('targeted_feedback', '')
        
        # Attempt to parse as JSON first
        try:
            json_feedback = json.loads(feedback)
            # Assuming JSON feedback might contain a list of problematic_sub_problems or similar
            if isinstance(json_feedback, dict) and "problematic_sub_problems" in json_feedback:
                problematic_ids.extend(json_feedback["problematic_sub_problems"])
            # Add other JSON structures if anticipated
        except json.JSONDecodeError:
            # Fallback to regex if not JSON
            found_ids = re.findall(r'(sub_\d+\.\d+)', feedback)
            problematic_ids.extend(found_ids)
            
    return list(set(problematic_ids)) # Return unique IDs

from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config

def generate_solution_for_sub_problem(sub_problem: SubProblem, team: Team, context: Dict[str, Any], solver_generation_gauntlet: Optional[GauntletDefinition] = None) -> str:
    """
    Generates a solution for a sub-problem using the assigned solver team and OpenEvolve.
    Supports single-candidate generation or multi-candidate peer review based on the provided
    solver_generation_gauntlet's generation_mode.
    """
    st.info(f"Generating solution for {sub_problem.id} using {team.name} via OpenEvolve...")

    if not team.members:
        st.error(f"Solver Team '{team.name}' has no members.")
        return "Failed to generate solution: No team members."

    model_config = team.members[0] # Use the first model in the team for generation

    # Construct OpenEvolve configuration
    system_message = f"You are an expert AI assistant tasked with solving sub-problem {sub_problem.id}. Generate a high-quality solution based on the description and context."
    if "feedback_report" in context:
        feedback_report_obj = context["feedback_report"]
        if dataclasses.is_dataclass(feedback_report_obj):
            feedback_report_obj = dataclasses.asdict(feedback_report_obj)
        feedback_json = json.dumps(feedback_report_obj, indent=2)
        system_message += f"\n\nPrevious feedback for this sub-problem:\n---\n{feedback_json}\n---\nAddress the issues raised in this feedback to improve the solution."

    generated_solution_content = ""

    # If evolution_params are provided, always use the full OpenEvolve unified evolution process.
    if sub_problem.evolution_params:
        st.info(f"  - Using OpenEvolve's `run_unified_evolution` for {sub_problem.id} due to provided `evolution_params`...")
        
        evolution_args = {
            "content": sub_problem.description,
            "content_type": "text_general",
            "evolution_mode": sub_problem.ai_suggested_evolution_mode,
            "model_configs": [{"name": model_config.model_id, "weight": 1.0}],
            "api_key": model_config.api_key,
            "api_base": model_config.api_base,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "system_message": system_message,
        }
        evolution_args.update(sub_problem.evolution_params)

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

    # Handle generation via Blue Team Gauntlet modes (multi-candidate or single direct call)
    elif solver_generation_gauntlet and solver_generation_gauntlet.generation_mode == "multi_candidate_peer_review":
        st.info(f"  - Multi-candidate peer review generation mode for {sub_problem.id}...")
        candidates = []
        generation_system_prompt = system_message
        generation_user_prompt = sub_problem.description # Use sub-problem description as base for generation

        for i, member in enumerate(team.members):
            st.info(f"    - Generating candidate {i+1}/{len(team.members)} using {member.model_id}...")
            candidate_content = _request_openai_compatible_chat(
                api_key=member.api_key,
                base_url=member.api_base,
                model=member.model_id,
                messages=_compose_messages(generation_system_prompt, generation_user_prompt),
                temperature=member.temperature,
                max_tokens=member.max_tokens
            )
            if candidate_content:
                candidates.append({"model_id": member.model_id, "content": candidate_content})
            else:
                st.warning(f"    - Failed to generate candidate from {member.model_id}. Skipping.")

        if not candidates:
            st.error("    - Failed to generate any candidates for peer review.")
            return "Failed to generate solution: No candidates for peer review."

        st.info("    - Performing peer review to select best candidate...")
        peer_review_system_prompt = "You are an expert peer reviewer. Evaluate the provided candidate solutions and select the best one, or synthesize a superior solution from them. Provide your response as a JSON object with 'score' (0.0-1.0), 'justification', and 'selected_solution'."
        peer_review_user_prompt = f"Sub-problem Description: {sub_problem.description}\n\nCandidates for review:\n---\n"
        for i, cand in enumerate(candidates):
            peer_review_user_prompt += f"Candidate {i+1} (from {cand['model_id']}):\n{cand['content']}\n---\n"
        peer_review_user_prompt += "\nSelect the best candidate or synthesize a new one. Output in JSON format: {\"score\": float, \"justification\": \"string\", \"selected_solution\": \"string\"}"

        # Use a dedicated peer reviewer (e.g., the first member of the team)
        reviewer_model_config = team.members[0]
        peer_review_response = _request_openai_compatible_chat(
            api_key=reviewer_model_config.api_key,
            base_url=reviewer_model_config.api_base,
            model=reviewer_model_config.model_id,
            messages=_compose_messages(peer_review_system_prompt, peer_review_user_prompt),
            temperature=reviewer_model_config.temperature,
            max_tokens=reviewer_model_config.max_tokens
        )

        if peer_review_response:
            try:
                parsed_review = json.loads(peer_review_response)
                generated_solution_content = parsed_review.get("selected_solution")
                if generated_solution_content:
                    st.success("    - Best candidate selected/synthesized via peer review.")
                else:
                    st.error("    - Peer review failed to select/synthesize a solution.")
                    return "Failed to generate solution: Peer review did not yield a solution."
            except json.JSONDecodeError:
                st.error(f"    - Peer review response was not valid JSON: {peer_review_response[:200]}...")
                return "Failed to generate solution: Peer review response invalid."
        else:
            st.error("    - Peer review failed to get a response.")
            return "Failed to generate solution: Peer review failed to get a response."

    else: # Default to single_candidate direct LLM call
        st.info(f"  - Single candidate direct generation mode for {sub_problem.id}...")
        generation_user_prompt = sub_problem.description
        generated_solution_content = _request_openai_compatible_chat(
            api_key=model_config.api_key,
            base_url=model_config.api_base,
            model=model_config.model_id,
            messages=_compose_messages(system_message, generation_user_prompt),
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )
        if not generated_solution_content:
            st.error(f"Direct generation failed for sub-problem {sub_problem.id}.")
            return f"Failed to generate solution: Direct generation failed for {sub_problem.id}."
    
    return generated_solution_content
