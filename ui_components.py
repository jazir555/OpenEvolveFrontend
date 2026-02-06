from ui_shim import ui as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from workflow_structures import Team, ModelConfig, GauntletDefinition, GauntletRoundRule, DecompositionPlan, SubProblem
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
import json
import time
import logging

logger = logging.getLogger(__name__)

# Initialize managers lazily (can be done once in the main app or passed via session state)
# These managers handle persistence of Teams and Gauntlets across UI reruns.
def _ensure_managers_initialized():
    """Ensure team and gauntlet managers are initialized in session state."""
    if not hasattr(st, 'session_state') or st.session_state is None:
        return

    # Use getattr/setattr to safely access attributes
    team_manager = getattr(st.session_state, 'team_manager', None)
    if team_manager is None:
        setattr(st.session_state, 'team_manager', TeamManager())

    gauntlet_manager = getattr(st.session_state, 'gauntlet_manager', None)
    if gauntlet_manager is None:
        setattr(st.session_state, 'gauntlet_manager', GauntletManager())

def render_team_manager():
    """Renders the UI UI for managing AI teams. Allows users to create, view, edit, and delete teams."""
    _ensure_managers_initialized()
    st.header("👥 Team Manager")
    st.write("Create, view, edit, and delete your AI teams.")

    team_manager: TeamManager = st.session_state.team_manager

    # --- Create New Team ---
    with st.expander("➕ Create New Team", expanded=False):
        with st.form("new_team_form"):
            team_name = st.text_input("Team Name", key="new_team_name")
            team_role = st.selectbox("Team Role", ["Blue", "Red", "Gold"], key="new_team_role")
            team_description = st.text_area("Description", key="new_team_description")

            if team_role == "Blue":
                st.subheader("Content Analysis Prompts (for Blue Teams acting as Content Analyzers)")
                new_ca_system_prompt = st.text_area("Content Analysis System Prompt", value="You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format.", key="new_ca_system_prompt")
                new_ca_user_prompt_template = st.text_area("Content Analysis User Prompt Template", value="""Analyze the following problem statement and extract:
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
    """, key="new_ca_user_prompt_template", height=300)
            else:
                new_ca_system_prompt = None
                new_ca_user_prompt_template = None

            if team_role == "Blue":
                st.subheader("Decomposition Prompts (for Blue Teams acting as Planners)")
                new_decomp_system_prompt = st.text_area("Decomposition System Prompt", value="You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects.", key="new_decomp_system_prompt")
                new_decomp_user_prompt_template = st.text_area("Decomposition User Prompt Template", value="""Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
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
    {analyzed_context}
    ---

    Provide the output as a JSON array of sub-problem objects.
    """, key="new_decomp_user_prompt_template", height=300)

                st.subheader("Solver Prompts (for Blue Teams acting as Solvers)")
                new_solver_system_prompt = st.text_area("Solver System Prompt", value="You are an expert problem solver. Your task is to generate a solution for the given sub-problem.", key="new_solver_system_prompt")
                new_solver_user_prompt_template = st.text_area("Solver User Prompt Template", value="""Generate a solution for the following sub-problem:
    ---
    {sub_problem_description}
    ---
    """, key="new_solver_user_prompt_template", height=200)

                st.subheader("Patcher Prompts (for Blue Teams acting as Patchers)")
                new_patcher_system_prompt = st.text_area("Patcher System Prompt", value="You are an expert problem patcher. Your task is to fix the identified flaws in the given solution attempt.", key="new_patcher_system_prompt")
                new_patcher_user_prompt_template = st.text_area("Patcher User Prompt Template", value="""Given the following sub-problem, solution attempt, and critique report, modify the solution to address the identified flaws:
    ---
    Sub-Problem: {sub_problem_description}
    Solution Attempt: {solution_attempt_content}
    Critique Report: {critique_report_summary}
    ---
    """, key="new_patcher_user_prompt_template", height=200)

                st.subheader("Assembler Prompts (for Blue Teams acting as Assemblers)")
                new_assembler_system_prompt = st.text_area("Assembler System Prompt", value="You are an expert solution assembler. Your task is to integrate the verified sub-problem solutions into a single, coherent final product.", key="new_assembler_system_prompt")
                new_assembler_user_prompt_template = st.text_area("Assembler User Prompt Template", value="""Integrate the following verified sub-problem solutions into a single, coherent final product:
    ---
    {verified_solutions}
    ---
    """, key="new_assembler_user_prompt_template", height=200)
            else:
                new_decomp_system_prompt = None
                new_decomp_user_prompt_template = None
                new_solver_system_prompt = None
                new_solver_user_prompt_template = None
                new_patcher_system_prompt = None
                new_patcher_user_prompt_template = None
                new_assembler_system_prompt = None
                new_assembler_user_prompt_template = None

            if team_role == "Red":
                st.subheader("Red Team Prompts")
                new_red_team_system_prompt = st.text_area("Red Team System Prompt", value="You are an adversarial AI. Your task is to critically evaluate the given solution, identify vulnerabilities, inconsistencies, and weaknesses, and provide a detailed critique report.", key="new_red_team_system_prompt")
                new_red_team_user_prompt_template = st.text_area("Red Team User Prompt Template", value="""Critique the following solution attempt based on the attack modes: {attack_modes}
    ---
    Solution: {solution_attempt_content}
    ---
    """, key="new_red_team_user_prompt_template", height=200)
            else:
                new_red_team_system_prompt = None
                new_red_team_user_prompt_template = None

            if team_role == "Gold":
                st.subheader("Gold Team Prompts")
                new_gold_team_system_prompt = st.text_area("Gold Team System Prompt", value="You are an impartial evaluator. Your task is to verify the correctness, quality, and adherence to requirements of the given solution against the provided evaluation prompt.", key="new_gold_team_system_prompt")
                new_gold_team_user_prompt_template = st.text_area("Gold Team User Prompt Template", value="""Evaluate the following solution attempt against the evaluation prompt:
    ---
    Evaluation Prompt: {evaluation_prompt}
    Solution: {solution_attempt_content}
    ---
    """, key="new_gold_team_user_prompt_template", height=200)
            else:
                new_gold_team_system_prompt = None
                new_gold_team_user_prompt_template = None

            st.subheader("Team Members (AI Models)")
            num_members = st.number_input("Number of Models in Team", min_value=1, value=1, key="num_new_members")

            # NEW: Team Type Selection for Cost Optimization
            team_type = st.selectbox("Team Type", ["standard", "swarm", "sovereign"], index=0, help="Standard: Balanced performance, Swarm: Cost-optimized with smaller models and voting, Sovereign: Maximum capability with larger models", key="new_team_type")

            new_members = []
            for i in range(num_members):
                st.markdown(f"**Model {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    model_id = st.text_input(f"Model ID (e.g., gpt-4o)", key=f"new_model_id_{i}")
                    api_key = st.text_input(f"API Key", type="password", key=f"new_api_key_{i}")
                with col2:
                    api_base = st.text_input(f"API Base (e.g., https://api.openai.com/v1)", value="https://api.openai.com/v1", key=f"new_api_base_{i}")
                    temperature = st.slider(f"Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key=f"new_temp_{i}")
                    top_p = st.slider(f"Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key=f"new_top_p_{i}")
                    max_tokens = st.number_input(f"Max Tokens", min_value=1, value=4096, key=f"new_max_tokens_{i}")
                    frequency_penalty = st.slider(f"Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key=f"new_freq_penalty_{i}")
                    presence_penalty = st.slider(f"Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key=f"new_pres_penalty_{i}")
                    seed = st.number_input(f"Seed (Optional)", value=None, key=f"new_seed_{i}")
                    n = st.number_input(f"N (Number of completions)", min_value=1, value=1, key=f"new_n_{i}")
                    logit_bias_str = st.text_area(f"Logit Bias (JSON, Optional)", key=f"new_logit_bias_{i}")
                    stop_sequences_str = st.text_input(f"Stop Sequences (comma-separated)", key=f"new_stop_sequences_{i}")
                    logprobs = st.checkbox(f"Logprobs", key=f"new_logprobs_{i}")
                    top_logprobs = st.number_input(f"Top Logprobs (0-5)", min_value=0, max_value=5, value=0, key=f"new_top_logprobs_{i}")
                    response_format_str = st.text_input(f"Response Format (JSON string, e.g., '{{\"type\": \"json_object\"}}')", key=f"new_response_format_{i}")
                    stream = st.checkbox(f"Stream", key=f"new_stream_{i}")
                    user = st.text_input(f"User ID", key=f"new_user_{i}")
                    reasoning_effort = st.selectbox(f"Reasoning Effort", [None, "low", "medium", "high"], key=f"new_reasoning_effort_{i}")
                    max_retries = st.number_input(f"Max Retries", min_value=0, value=5, key=f"new_max_retries_{i}")
                    timeout = st.number_input(f"Timeout (seconds)", min_value=1, value=120, key=f"new_timeout_{i}")
                    organization = st.text_input(f"Organization ID (Optional)", key=f"new_organization_{i}")
                    response_model = st.text_input(f"Response Model (Pydantic model name, Optional)", key=f"new_response_model_{i}")
                    tools_json = st.text_area(f"Tools (JSON array, Optional)", key=f"new_tools_{i}", help="e.g., [{'type': 'function', 'function': {'name': 'my_function', 'description': '...', 'parameters': {...}}}]")
                    tool_choice = st.text_input(f"Tool Choice (e.g., 'auto', 'none', or JSON)", key=f"new_tool_choice_{i}")
                    system_fingerprint = st.text_input(f"System Fingerprint (Optional)", key=f"new_system_fingerprint_{i}")
                    deployment_id = st.text_input(f"Deployment ID (Azure OpenAI, Optional)", key=f"new_deployment_id_{i}")
                    encoding_format = st.text_input(f"Encoding Format (Optional)", key=f"new_encoding_format_{i}")
                    max_input_tokens = st.number_input(f"Max Input Tokens (Optional)", value=None, key=f"new_max_input_tokens_{i}")
                    stop_token = st.text_input(f"Stop Token (Optional, single token)", key=f"new_stop_token_{i}")
                    best_of = st.number_input(f"Best Of (Optional)", value=None, key=f"new_best_of_{i}")
                    logprobs_offset = st.number_input(f"Logprobs Offset (Optional)", value=None, key=f"new_logprobs_offset_{i}")
                    suffix = st.text_input(f"Suffix (Optional)", key=f"new_suffix_{i}")
                    presence_penalty_range_str = st.text_input(f"Presence Penalty Range (comma-separated floats, e.g., -2.0,2.0)", key=f"new_presence_penalty_range_{i}")
                    frequency_penalty_range_str = st.text_input(f"Frequency Penalty Range (comma-separated floats, e.g., -2.0,2.0)", key=f"new_frequency_penalty_range_{i}")
                    stop_token_id = st.number_input(f"Stop Token ID (Optional)", value=None, key=f"new_stop_token_id_{i}")
                    response_json_format = st.checkbox(f"Response JSON Format", key=f"new_response_json_format_{i}")
                    max_output_tokens = st.number_input(f"Max Output Tokens (Optional)", value=None, key=f"new_max_output_tokens_{i}")
                    stream_options_json = st.text_area(f"Stream Options (JSON, Optional)", key=f"new_stream_options_{i}")
                    logprobs_type = st.selectbox(f"Logprobs Type (Optional)", [None, "per_token", "all"], key=f"new_logprobs_type_{i}")
                    top_k = st.number_input(f"Top K (Optional)", value=None, key=f"new_top_k_{i}")
                    repetition_penalty = st.slider(f"Repetition Penalty (Optional)", min_value=0.0, max_value=2.0, value=1.0, step=0.01, key=f"new_repetition_penalty_{i}")
                    length_penalty = st.slider(f"Length Penalty (Optional)", min_value=0.0, max_value=2.0, value=1.0, step=0.01, key=f"new_length_penalty_{i}")
                    early_stopping = st.checkbox(f"Early Stopping (for beam search)", key=f"new_early_stopping_{i}")
                    num_beams = st.number_input(f"Number of Beams (for beam search)", min_value=1, value=1, key=f"new_num_beams_{i}")
                    do_sample = st.checkbox(f"Do Sample", value=True, key=f"new_do_sample_{i}")
                    temperature_fallback = st.slider(f"Temperature Fallback (Optional)", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key=f"new_temperature_fallback_{i}")
                    top_p_fallback = st.slider(f"Top P Fallback (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key=f"new_top_p_fallback_{i}")
                    max_time = st.number_input(f"Max Time (seconds, Optional)", value=None, key=f"new_max_time_{i}")
                    return_full_text = st.checkbox(f"Return Full Text", value=False, key=f"new_return_full_text_{i}")
                    tokenizer_config_json = st.text_area(f"Tokenizer Config (JSON, Optional)", key=f"new_tokenizer_config_{i}")
                    model_kwargs_json = st.text_area(f"Model Kwargs (JSON, Optional)", key=f"new_model_kwargs_{i}")

                new_members.append(ModelConfig(
                    model_id=model_id,
                    api_key=api_key,
                    api_base=api_base,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed if seed is not None else None,
                    n=n,
                    logit_bias=json.loads(logit_bias_str) if logit_bias_str else None,
                    stop_sequences=[s.strip() for s in stop_sequences_str.split(',')] if stop_sequences_str else None,
                    logprobs=logprobs if logprobs else None,
                    top_logprobs=top_logprobs if top_logprobs > 0 else None,
                    response_format=json.loads(response_format_str) if response_format_str else None,
                    stream=stream if stream else None,
                    user=user if user else None,
                    reasoning_effort=reasoning_effort,
                    max_retries=max_retries,
                    timeout=timeout,
                    organization=organization if organization else None,
                    response_model=response_model if response_model else None,
                    tools=json.loads(tools_json) if tools_json else None,
                    tool_choice=json.loads(tool_choice) if tool_choice and tool_choice.startswith('{') else (tool_choice if tool_choice else None),
                    system_fingerprint=system_fingerprint if system_fingerprint else None,
                    deployment_id=deployment_id if deployment_id else None,
                    encoding_format=encoding_format if encoding_format else None,
                    max_input_tokens=max_input_tokens if max_input_tokens is not None else None,
                    stop_token=stop_token if stop_token else None,
                    best_of=best_of if best_of is not None else None,
                    logprobs_offset=logprobs_offset if logprobs_offset is not None else None,
                    suffix=suffix if suffix else None
                ))

            submitted = st.form_submit_button("Create Team")
            if submitted:
                if team_name and new_members[0].model_id: # Basic validation
                    new_team = Team(name=team_name, role=team_role, members=new_members, description=team_description,
                                    content_analysis_system_prompt=new_ca_system_prompt,
                                    content_analysis_user_prompt_template=new_ca_user_prompt_template,
                                    decomposition_system_prompt=new_decomp_system_prompt,
                                    decomposition_user_prompt_template=new_decomp_user_prompt_template,
                                    solver_system_prompt=new_solver_system_prompt,
                                    solver_user_prompt_template=new_solver_user_prompt_template,
                                    patcher_system_prompt=new_patcher_system_prompt,
                                    patcher_user_prompt_template=new_patcher_user_prompt_template,
                                    assembler_system_prompt=new_assembler_system_prompt,
                                    assembler_user_prompt_template=new_assembler_user_prompt_template,
                                    red_team_system_prompt=new_red_team_system_prompt,
                                    red_team_user_prompt_template=new_red_team_user_prompt_template,
                                    gold_team_system_prompt=new_gold_team_system_prompt,
                                    gold_team_user_prompt_template=new_gold_team_user_prompt_template,
                                    team_type=team_type)  # NEW: Include team_type
                    if team_manager.create_team(new_team):
                        st.success(f"Team '{team_name}' created successfully!")
                        st.session_state.team_manager = TeamManager() # Reload to refresh UI
                    else:
                        st.error(f"Team '{team_name}' already exists.")
                else:
                    st.error("Please fill in team name and at least one model ID.")

    # --- View/Edit/Delete Existing Teams ---
    st.subheader("Existing Teams")
    teams = team_manager.get_all_teams()
    if not teams:
        st.info("No teams created yet.")
    else:
        for team in teams:
            with st.container(border=True):
                st.markdown(f"**{team.name}** ({team.role} Team)")
                st.caption(team.description or "No description.")
                
                if team.role == "Blue":
                    st.markdown("**Content Analysis Prompts:**")
                    st.write(f"System: `{team.content_analysis_system_prompt}`")
                    st.write(f"User Template: `{team.content_analysis_user_prompt_template}`")
                    st.markdown("**Decomposition Prompts:**")
                    st.write(f"System: `{team.decomposition_system_prompt}`")
                    st.write(f"User Template: `{team.decomposition_user_prompt_template}`")
                    st.markdown("**Solver Prompts:**")
                    st.write(f"System: `{team.solver_system_prompt}`")
                    st.write(f"User Template: `{team.solver_user_prompt_template}`")
                    st.markdown("**Patcher Prompts:**")
                    st.write(f"System: `{team.patcher_system_prompt}`")
                    st.write(f"User Template: `{team.patcher_user_prompt_template}`")
                    st.markdown("**Assembler Prompts:**")
                    st.write(f"System: `{team.assembler_system_prompt}`")
                    st.write(f"User Template: `{team.assembler_user_prompt_template}`")
                elif team.role == "Red":
                    st.markdown("**Red Team Prompts:**")
                    st.write(f"System: `{team.red_team_system_prompt}`")
                    st.write(f"User Template: `{team.red_team_user_prompt_template}`")
                elif team.role == "Gold":
                    st.markdown("**Gold Team Prompts:**")
                    st.write(f"System: `{team.gold_team_system_prompt}`")
                    st.write(f"User Template: `{team.gold_team_user_prompt_template}`")
                
                with st.expander(f"View/Edit Team '{team.name}'", expanded=False):
                    # Display current members
                    st.markdown("#### Current Members:")
                    for i, member in enumerate(team.members):
                        st.markdown(f"**Model {i+1}**: `{member.model_id}`")
                        st.write(f"API Base: `{member.api_base}` | Temp: `{member.temperature}` | Top P: `{member.top_p}` | Max Tokens: `{member.max_tokens}` | Freq Penalty: `{member.frequency_penalty}` | Pres Penalty: `{member.presence_penalty}` | Seed: `{member.seed}`")
                        st.write(f"Stop Sequences: `{member.stop_sequences}` | Logprobs: `{member.logprobs}` | Top Logprobs: `{member.top_logprobs}` | Response Format: `{member.response_format}` | Stream: `{member.stream}` | User: `{member.user}` | Reasoning Effort: `{member.reasoning_effort}`")
                        st.write(f"Max Retries: `{member.max_retries}` | Timeout: `{member.timeout}` | Organization: `{member.organization}` | Response Model: `{member.response_model}`")
                        st.write(f"Tools: `{member.tools}` | Tool Choice: `{member.tool_choice}` | System Fingerprint: `{member.system_fingerprint}` | Deployment ID: `{member.deployment_id}`")
                        st.write(f"Encoding Format: `{member.encoding_format}` | Max Input Tokens: `{member.max_input_tokens}` | Stop Token: `{member.stop_token}` | Best Of: `{member.best_of}` | Logprobs Offset: `{member.logprobs_offset}` | Suffix: `{member.suffix}`")
                        st.write(f"Presence Penalty Range: `{member.presence_penalty_range}` | Frequency Penalty Range: `{member.frequency_penalty_range}` | Stop Token ID: `{member.stop_token_id}` | Response JSON Format: `{member.response_json_format}` | Max Output Tokens: `{member.max_output_tokens}`")
                        st.write(f"Stream Options: `{member.stream_options}` | Logprobs Type: `{member.logprobs_type}` | Top K: `{member.top_k}` | Repetition Penalty: `{member.repetition_penalty}` | Length Penalty: `{member.length_penalty}`")
                        st.write(f"Early Stopping: `{member.early_stopping}` | Num Beams: `{member.num_beams}` | Do Sample: `{member.do_sample}` | Temperature Fallback: `{member.temperature_fallback}` | Top P Fallback: `{member.top_p_fallback}`")
                        st.write(f"Max Time: `{member.max_time}` | Return Full Text: `{member.return_full_text}` | Tokenizer Config: `{member.tokenizer_config}` | Model Kwargs: `{member.model_kwargs}`")

                    # Edit Form
                    with st.form(f"edit_team_form_{team.name}"):
                        edited_team_name = st.text_input("Team Name", value=team.name, key=f"edit_team_name_{team.name}")
                        edited_team_role = st.selectbox("Team Role", ["Blue", "Red", "Gold"], index=["Blue", "Red", "Gold"].index(team.role), key=f"edit_team_role_{team.name}")
                        edited_team_description = st.text_area("Description", value=team.description, key=f"edit_team_description_{team.name}")

                        # NEW: Team Type Selection for Cost Optimization
                        current_team_type = getattr(team, 'team_type', 'standard')
                        edited_team_type = st.selectbox("Team Type", ["standard", "swarm", "sovereign"], index=["standard", "swarm", "sovereign"].index(current_team_type), help="Standard: Balanced performance, Swarm: Cost-optimized with smaller models and voting, Sovereign: Maximum capability with larger models", key=f"edit_team_type_{team.name}")

                        edited_ca_system_prompt = None
                        edited_ca_user_prompt_template = None
                        if edited_team_role == "Blue":
                            st.subheader("Content Analysis Prompts (for Blue Teams acting as Content Analyzers)")
                            edited_ca_system_prompt = st.text_area("Content Analysis System Prompt", value=team.content_analysis_system_prompt if team.content_analysis_system_prompt else "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format.", key=f"edit_ca_system_prompt_{team.name}")
                            edited_ca_user_prompt_template = st.text_area("Content Analysis User Prompt Template", value=team.content_analysis_user_prompt_template if team.content_analysis_user_prompt_template else """Analyze the following problem statement and extract:
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
    """, key=f"edit_ca_user_prompt_template_{team.name}", height=300)

                            current_decomp_system_prompt = team.decomposition_system_prompt if team.decomposition_system_prompt else "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects."
                            current_decomp_user_prompt_template = team.decomposition_user_prompt_template if team.decomposition_user_prompt_template else """Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
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
    {analyzed_context}
    ---

    Provide the output as a JSON array of sub-problem objects.
    """
                            st.subheader("Decomposition Prompts (for Blue Teams acting as Planners)")
                            edited_decomp_system_prompt = st.text_area("Decomposition System Prompt", value=current_decomp_system_prompt, key=f"edit_decomp_system_prompt_{team.name}")
                            edited_decomp_user_prompt_template = st.text_area("Decomposition User Prompt Template", value=current_decomp_user_prompt_template, key=f"edit_decomp_user_prompt_template_{team.name}", height=300)

                            current_solver_system_prompt = team.solver_system_prompt if team.solver_system_prompt else "You are an expert problem solver. Your task is to generate a solution for the given sub-problem."
                            current_solver_user_prompt_template = team.solver_user_prompt_template if team.solver_user_prompt_template else """Generate a solution for the following sub-problem:
    ---
    {sub_problem_description}
    ---
    """
                            st.subheader("Solver Prompts (for Blue Teams acting as Solvers)")
                            edited_solver_system_prompt = st.text_area("Solver System Prompt", value=current_solver_system_prompt, key=f"edit_solver_system_prompt_{team.name}")
                            edited_solver_user_prompt_template = st.text_area("Solver User Prompt Template", value=current_solver_user_prompt_template, key=f"edit_solver_user_prompt_template_{team.name}", height=200)

                            current_patcher_system_prompt = team.patcher_system_prompt if team.patcher_system_prompt else "You are an expert problem patcher. Your task is to fix the identified flaws in the given solution attempt."
                            current_patcher_user_prompt_template = team.patcher_user_prompt_template if team.patcher_user_prompt_template else """Given the following sub-problem, solution attempt, and critique report, modify the solution to address the identified flaws:
    ---
    Sub-Problem: {sub_problem_description}
    Solution Attempt: {solution_attempt_content}
    Critique Report: {critique_report_summary}
    ---
    """
                            st.subheader("Patcher Prompts (for Blue Teams acting as Patchers)")
                            edited_patcher_system_prompt = st.text_area("Patcher System Prompt", value=current_patcher_system_prompt, key=f"edit_patcher_system_prompt_{team.name}")
                            edited_patcher_user_prompt_template = st.text_area("Patcher User Prompt Template", value=current_patcher_user_prompt_template, key=f"edit_patcher_user_prompt_template_{team.name}", height=200)

                            current_assembler_system_prompt = team.assembler_system_prompt if team.assembler_system_prompt else "You are an expert solution assembler. Your task is to integrate the verified sub-problem solutions into a single, coherent final product."
                            current_assembler_user_prompt_template = team.assembler_user_prompt_template if team.assembler_user_prompt_template else """Integrate the following verified sub-problem solutions into a single, coherent final product:
    ---
    {verified_solutions}
    ---
    """
                            st.subheader("Assembler Prompts (for Blue Teams acting as Assemblers)")
                            edited_assembler_system_prompt = st.text_area("Assembler System Prompt", value=current_assembler_system_prompt, key=f"edit_assembler_system_prompt_{team.name}")
                            edited_assembler_user_prompt_template = st.text_area("Assembler User Prompt Template", value=current_assembler_user_prompt_template, key=f"edit_assembler_user_prompt_template_{team.name}", height=200)

                            edited_solver_system_prompt = None
                            edited_solver_user_prompt_template = None
                            edited_patcher_system_prompt = None
                            edited_patcher_user_prompt_template = None
                            edited_assembler_system_prompt = None
                            edited_assembler_user_prompt_template = None
                        else:
                            edited_ca_system_prompt = None
                            edited_ca_user_prompt_template = None
                            edited_decomp_system_prompt = None
                            edited_decomp_user_prompt_template = None
                            edited_solver_system_prompt = None
                            edited_solver_user_prompt_template = None
                            edited_patcher_system_prompt = None
                            edited_patcher_user_prompt_template = None
                            edited_assembler_system_prompt = None
                            edited_assembler_user_prompt_template = None

                        if edited_team_role == "Red":
                            current_red_team_system_prompt = team.red_team_system_prompt if team.red_team_system_prompt else "You are an adversarial AI. Your task is to critically evaluate the given solution, identify vulnerabilities, inconsistencies, and weaknesses, and provide a detailed critique report."
                            current_red_team_user_prompt_template = team.red_team_user_prompt_template if team.red_team_user_prompt_template else """Critique the following solution attempt based on the attack modes: {attack_modes}
    ---
    Solution: {solution_attempt_content}
    ---
    """
                            st.subheader("Red Team Prompts")
                            edited_red_team_system_prompt = st.text_area("Red Team System Prompt", value=current_red_team_system_prompt, key=f"edit_red_team_system_prompt_{team.name}")
                            edited_red_team_user_prompt_template = st.text_area("Red Team User Prompt Template", value=current_red_team_user_prompt_template, key=f"edit_red_team_user_prompt_template_{team.name}", height=200)
                        else:
                            edited_red_team_system_prompt = None
                            edited_red_team_user_prompt_template = None

                        if edited_team_role == "Gold":
                            current_gold_team_system_prompt = team.gold_team_system_prompt if team.gold_team_system_prompt else "You are an impartial evaluator. Your task is to verify the correctness, quality, and adherence to requirements of the given solution against the provided evaluation prompt."
                            current_gold_team_user_prompt_template = team.gold_team_user_prompt_template if team.gold_team_user_prompt_template else """Evaluate the following solution attempt against the evaluation prompt:
    ---
    Evaluation Prompt: {evaluation_prompt}
    Solution: {solution_attempt_content}
    ---
    """
                            st.subheader("Gold Team Prompts")
                            edited_gold_team_system_prompt = st.text_area("Gold Team System Prompt", value=current_gold_team_system_prompt, key=f"edit_gold_team_system_prompt_{team.name}")
                            edited_gold_team_user_prompt_template = st.text_area("Gold Team User Prompt Template", value=current_gold_team_user_prompt_template, key=f"edit_gold_team_user_prompt_template_{team.name}", height=200)
                        else:
                            edited_gold_team_system_prompt = None
                            edited_gold_team_user_prompt_template = None

                        st.subheader("Edit Team Members (AI Models)")
                        # Allow adding/removing members, or editing existing ones.
                        # Provides full CRUD operations for team member management.
                        num_existing_members = len(team.members)
                        num_members_to_edit = st.number_input("Number of Models in Team", min_value=1, value=num_existing_members, key=f"num_edit_members_{team.name}")

                        edited_members = []
                        for i in range(num_members_to_edit):
                            st.markdown(f"**Model {i+1}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                current_model_id = team.members[i].model_id if i < num_existing_members else ""
                                current_api_key = team.members[i].api_key if i < num_existing_members else ""
                                edited_model_id = st.text_input(f"Model ID (e.g., gpt-4o)", value=current_model_id, key=f"edit_model_id_{team.name}_{i}")
                                edited_api_key = st.text_input(f"API Key", type="password", value=current_api_key, key=f"edit_api_key_{team.name}_{i}")
                            with col2:
                                current_presence_penalty = team.members[i].presence_penalty if i < num_existing_members else 0.0
                                current_seed = team.members[i].seed if i < num_existing_members else None
                                current_n = team.members[i].n if i < num_existing_members else 1
                                current_logit_bias = json.dumps(team.members[i].logit_bias) if i < num_existing_members and team.members[i].logit_bias else ""
                                current_stop_sequences = ", ".join(team.members[i].stop_sequences) if i < num_existing_members and team.members[i].stop_sequences else ""
                                current_logprobs = team.members[i].logprobs if i < num_existing_members and team.members[i].logprobs is not None else False
                                current_top_logprobs = team.members[i].top_logprobs if i < num_existing_members and team.members[i].top_logprobs is not None else 0
                                current_response_format = json.dumps(team.members[i].response_format) if i < num_existing_members and team.members[i].response_format else ""
                                current_stream = team.members[i].stream if i < num_existing_members and team.members[i].stream is not None else False
                                current_user = team.members[i].user if i < num_existing_members and team.members[i].user else ""
                                current_reasoning_effort = team.members[i].reasoning_effort if i < num_existing_members else None
                                current_max_retries = team.members[i].max_retries if i < num_existing_members else 5
                                current_timeout = team.members[i].timeout if i < num_existing_members else 120
                                current_organization = team.members[i].organization if i < num_existing_members else ""
                                current_response_model = team.members[i].response_model if i < num_existing_members else ""
                                current_tools = json.dumps(team.members[i].tools) if i < num_existing_members and team.members[i].tools else ""
                                current_tool_choice = json.dumps(team.members[i].tool_choice) if i < num_existing_members and isinstance(team.members[i].tool_choice, dict) else (team.members[i].tool_choice if i < num_existing_members and team.members[i].tool_choice else "")
                                current_system_fingerprint = team.members[i].system_fingerprint if i < num_existing_members else ""
                                current_deployment_id = team.members[i].deployment_id if i < num_existing_members else ""
                                current_encoding_format = team.members[i].encoding_format if i < num_existing_members else ""
                                current_max_input_tokens = team.members[i].max_input_tokens if i < num_existing_members else None
                                current_stop_token = team.members[i].stop_token if i < num_existing_members else ""
                                current_best_of = team.members[i].best_of if i < num_existing_members else None
                                current_logprobs_offset = team.members[i].logprobs_offset if i < num_existing_members else None
                                current_suffix = team.members[i].suffix if i < num_existing_members else ""
                                current_presence_penalty_range = ",".join(map(str, team.members[i].presence_penalty_range)) if i < num_existing_members and team.members[i].presence_penalty_range else ""
                                current_frequency_penalty_range = ",".join(map(str, team.members[i].frequency_penalty_range)) if i < num_existing_members and team.members[i].frequency_penalty_range else ""
                                current_stop_token_id = team.members[i].stop_token_id if i < num_existing_members else None
                                current_response_json_format = team.members[i].response_json_format if i < num_existing_members and team.members[i].response_json_format is not None else False
                                current_max_output_tokens = team.members[i].max_output_tokens if i < num_existing_members else None
                                current_stream_options = json.dumps(team.members[i].stream_options) if i < num_existing_members and team.members[i].stream_options else ""
                                current_logprobs_type = team.members[i].logprobs_type if i < num_existing_members else None
                                current_top_k = team.members[i].top_k if i < num_existing_members else None
                                current_repetition_penalty = team.members[i].repetition_penalty if i < num_existing_members else 1.0
                                current_length_penalty = team.members[i].length_penalty if i < num_existing_members else 1.0
                                current_early_stopping = team.members[i].early_stopping if i < num_existing_members and team.members[i].early_stopping is not None else False
                                current_num_beams = team.members[i].num_beams if i < num_existing_members else 1
                                current_do_sample = team.members[i].do_sample if i < num_existing_members and team.members[i].do_sample is not None else True
                                current_temperature_fallback = team.members[i].temperature_fallback if i < num_existing_members else 0.7
                                current_top_p_fallback = team.members[i].top_p_fallback if i < num_existing_members else 1.0
                                current_max_time = team.members[i].max_time if i < num_existing_members else None
                                current_return_full_text = team.members[i].return_full_text if i < num_existing_members and team.members[i].return_full_text is not None else False
                                current_tokenizer_config = json.dumps(team.members[i].tokenizer_config) if i < num_existing_members and team.members[i].tokenizer_config else ""
                                current_model_kwargs = json.dumps(team.members[i].model_kwargs) if i < num_existing_members and team.members[i].model_kwargs else ""

                                edited_api_base = st.text_input(f"API Base (e.g., https://api.openai.com/v1)", value=current_api_base, key=f"edit_api_base_{team.name}_{i}")
                                edited_temperature = st.slider(f"Temperature", min_value=0.0, max_value=2.0, value=current_temperature, step=0.1, key=f"edit_temp_{team.name}_{i}")
                                edited_top_p = st.slider(f"Top P", min_value=0.0, max_value=1.0, value=current_top_p, step=0.01, key=f"edit_top_p_{team.name}_{i}")
                                edited_max_tokens = st.number_input(f"Max Tokens", min_value=1, value=current_max_tokens, key=f"edit_max_tokens_{team.name}_{i}")
                                edited_frequency_penalty = st.slider(f"Frequency Penalty", min_value=-2.0, max_value=2.0, value=current_frequency_penalty, step=0.01, key=f"edit_freq_penalty_{team.name}_{i}")
                                edited_presence_penalty = st.slider(f"Presence Penalty", min_value=-2.0, max_value=2.0, value=current_presence_penalty, step=0.01, key=f"edit_pres_penalty_{team.name}_{i}")
                                edited_seed = st.number_input(f"Seed (Optional)", value=current_seed, key=f"edit_seed_{team.name}_{i}")
                                edited_n = st.number_input(f"N (Number of completions)", min_value=1, value=current_n, key=f"edit_n_{team.name}_{i}")
                                edited_logit_bias_str = st.text_area(f"Logit Bias (JSON, Optional)", value=current_logit_bias, key=f"edit_logit_bias_{team.name}_{i}")
                                edited_stop_sequences_str = st.text_input(f"Stop Sequences (comma-separated)", value=current_stop_sequences, key=f"edit_stop_sequences_{team.name}_{i}")
                                edited_logprobs = st.checkbox(f"Logprobs", value=current_logprobs, key=f"edit_logprobs_{team.name}_{i}")
                                edited_top_logprobs = st.number_input(f"Top Logprobs (0-5)", min_value=0, max_value=5, value=current_top_logprobs, key=f"edit_top_logprobs_{team.name}_{i}")
                                edited_response_format_str = st.text_input(f"Response Format (JSON string, e.g., '{{\"type\": \"json_object\"}}')", value=current_response_format, key=f"edit_response_format_{team.name}_{i}")
                                edited_stream = st.checkbox(f"Stream", value=current_stream, key=f"edit_stream_{team.name}_{i}")
                                edited_user = st.text_input(f"User ID", value=current_user, key=f"edit_user_{team.name}_{i}")
                                edited_reasoning_effort = st.selectbox(f"Reasoning Effort", [None, "low", "medium", "high"], index=[None, "low", "medium", "high"].index(current_reasoning_effort) if current_reasoning_effort in ["low", "medium", "high"] else 0, key=f"edit_reasoning_effort_{team.name}_{i}")
                                edited_max_retries = st.number_input(f"Max Retries", min_value=0, value=current_max_retries, key=f"edit_max_retries_{team.name}_{i}")
                                edited_timeout = st.number_input(f"Timeout (seconds)", min_value=1, value=current_timeout, key=f"edit_timeout_{team.name}_{i}")
                                edited_organization = st.text_input(f"Organization ID (Optional)", value=current_organization, key=f"edit_organization_{team.name}_{i}")
                                edited_response_model = st.text_input(f"Response Model (Pydantic model name, Optional)", value=current_response_model, key=f"edit_response_model_{team.name}_{i}")
                                edited_tools_json = st.text_area(f"Tools (JSON array, Optional)", value=current_tools, key=f"edit_tools_{team.name}_{i}", help="e.g., [{'type': 'function', 'function': {'name': 'my_function', 'description': '...', 'parameters': {...}}}]")
                                edited_tool_choice = st.text_input(f"Tool Choice (e.g., 'auto', 'none', or JSON)", value=current_tool_choice, key=f"edit_tool_choice_{team.name}_{i}")
                                edited_system_fingerprint = st.text_input(f"System Fingerprint (Optional)", value=current_system_fingerprint, key=f"edit_system_fingerprint_{team.name}_{i}")
                                edited_deployment_id = st.text_input(f"Deployment ID (Azure OpenAI, Optional)", value=current_deployment_id, key=f"edit_deployment_id_{team.name}_{i}")
                                edited_encoding_format = st.text_input(f"Encoding Format (Optional)", value=current_encoding_format, key=f"edit_encoding_format_{team.name}_{i}")
                                edited_max_input_tokens = st.number_input(f"Max Input Tokens (Optional)", value=current_max_input_tokens, key=f"edit_max_input_tokens_{team.name}_{i}")
                                edited_stop_token = st.text_input(f"Stop Token (Optional, single token)", value=current_stop_token, key=f"edit_stop_token_{team.name}_{i}")
                                edited_best_of = st.number_input(f"Best Of (Optional)", value=current_best_of, key=f"edit_best_of_{team.name}_{i}")
                                edited_logprobs_offset = st.number_input(f"Logprobs Offset (Optional)", value=current_logprobs_offset, key=f"edit_logprobs_offset_{team.name}_{i}")
                                edited_suffix = st.text_input(f"Suffix (Optional)", value=current_suffix, key=f"edit_suffix_{team.name}_{i}")
                                edited_presence_penalty_range_str = st.text_input(f"Presence Penalty Range (comma-separated floats, e.g., -2.0,2.0)", value=current_presence_penalty_range, key=f"edit_presence_penalty_range_{team.name}_{i}")
                                edited_frequency_penalty_range_str = st.text_input(f"Frequency Penalty Range (comma-separated floats, e.g., -2.0,2.0)", value=current_frequency_penalty_range, key=f"edit_frequency_penalty_range_{team.name}_{i}")
                                edited_stop_token_id = st.number_input(f"Stop Token ID (Optional)", value=current_stop_token_id, key=f"edit_stop_token_id_{team.name}_{i}")
                                edited_response_json_format = st.checkbox(f"Response JSON Format", value=current_response_json_format, key=f"edit_response_json_format_{team.name}_{i}")
                                edited_max_output_tokens = st.number_input(f"Max Output Tokens (Optional)", value=current_max_output_tokens, key=f"edit_max_output_tokens_{team.name}_{i}")
                                edited_stream_options_json = st.text_area(f"Stream Options (JSON, Optional)", value=current_stream_options, key=f"edit_stream_options_{team.name}_{i}")
                                edited_logprobs_type = st.selectbox(f"Logprobs Type (Optional)", [None, "per_token", "all"], index=[None, "per_token", "all"].index(current_logprobs_type) if current_logprobs_type in ["per_token", "all"] else 0, key=f"edit_logprobs_type_{team.name}_{i}")
                                edited_top_k = st.number_input(f"Top K (Optional)", value=current_top_k, key=f"edit_top_k_{team.name}_{i}")
                                edited_repetition_penalty = st.slider(f"Repetition Penalty (Optional)", min_value=0.0, max_value=2.0, value=current_repetition_penalty, step=0.01, key=f"edit_repetition_penalty_{team.name}_{i}")
                                edited_length_penalty = st.slider(f"Length Penalty (Optional)", min_value=0.0, max_value=2.0, value=current_length_penalty, step=0.01, key=f"edit_length_penalty_{team.name}_{i}")
                                edited_early_stopping = st.checkbox(f"Early Stopping (for beam search)", value=current_early_stopping, key=f"edit_early_stopping_{team.name}_{i}")
                                edited_num_beams = st.number_input(f"Number of Beams (for beam search)", min_value=1, value=current_num_beams, key=f"edit_num_beams_{team.name}_{i}")
                                edited_do_sample = st.checkbox(f"Do Sample", value=True, key=f"edit_do_sample_{team.name}_{i}")
                                edited_temperature_fallback = st.slider(f"Temperature Fallback (Optional)", min_value=0.0, max_value=2.0, value=current_temperature_fallback, step=0.1, key=f"edit_temperature_fallback_{team.name}_{i}")
                                edited_top_p_fallback = st.slider(f"Top P Fallback (Optional)", min_value=0.0, max_value=1.0, value=current_top_p_fallback, step=0.01, key=f"edit_top_p_fallback_{team.name}_{i}")
                                edited_max_time = st.number_input(f"Max Time (seconds, Optional)", value=current_max_time, key=f"edit_max_time_{team.name}_{i}")
                                edited_return_full_text = st.checkbox(f"Return Full Text", value=current_return_full_text, key=f"edit_return_full_text_{team.name}_{i}")
                                edited_tokenizer_config_json = st.text_area(f"Tokenizer Config (JSON, Optional)", value=current_tokenizer_config, key=f"edit_tokenizer_config_{team.name}_{i}")
                                edited_model_kwargs_json = st.text_area(f"Model Kwargs (JSON, Optional)", value=current_model_kwargs, key=f"edit_model_kwargs_{team.name}_{i}")
                            
                            if edited_model_id: # Only add if model ID is provided
                                edited_members.append(ModelConfig(
                                    model_id=edited_model_id,
                                    api_key=edited_api_key,
                                    api_base=edited_api_base,
                                    temperature=edited_temperature,
                                    top_p=edited_top_p,
                                    max_tokens=edited_max_tokens,
                                    frequency_penalty=edited_frequency_penalty,
                                    presence_penalty=edited_presence_penalty,
                                    seed=edited_seed if edited_seed is not None else None,
                                    n=edited_n,
                                    logit_bias=json.loads(edited_logit_bias_str) if edited_logit_bias_str else None,
                                    stop_sequences=[s.strip() for s in edited_stop_sequences_str.split(',')] if edited_stop_sequences_str else None,
                                    logprobs=edited_logprobs if edited_logprobs else None,
                                    top_logprobs=edited_top_logprobs if edited_top_logprobs > 0 else None,
                                    response_format=json.loads(edited_response_format_str) if edited_response_format_str else None,
                                    stream=edited_stream if edited_stream else None,
                                    user=edited_user if edited_user else None,
                                    reasoning_effort=edited_reasoning_effort,
                                    max_retries=edited_max_retries,
                                    timeout=edited_timeout,
                                    organization=edited_organization if edited_organization else None,
                                    response_model=edited_response_model if edited_response_model else None,
                                    tools=json.loads(edited_tools_json) if edited_tools_json else None,
                                    tool_choice=json.loads(edited_tool_choice) if edited_tool_choice and edited_tool_choice.startswith('{') else (edited_tool_choice if edited_tool_choice else None),
                                    system_fingerprint=edited_system_fingerprint if edited_system_fingerprint else None,
                                    deployment_id=edited_deployment_id if edited_deployment_id else None,
                                    encoding_format=edited_encoding_format if edited_encoding_format else None,
                                    max_input_tokens=edited_max_input_tokens if edited_max_input_tokens is not None else None,
                                    stop_token=edited_stop_token if edited_stop_token else None,
                                    best_of=edited_best_of if edited_best_of is not None else None,
                                    logprobs_offset=edited_logprobs_offset if edited_logprobs_offset is not None else None,
                                    suffix=edited_suffix if edited_suffix else None
                                ))
                        
                        update_submitted = st.form_submit_button("Update Team")
                        if update_submitted:
                            if edited_team_name and edited_members:
                                updated_team = Team(name=edited_team_name, role=edited_team_role, members=edited_members, description=edited_team_description,
                                                    content_analysis_system_prompt=edited_ca_system_prompt,
                                                    content_analysis_user_prompt_template=edited_ca_user_prompt_template,
                                                    decomposition_system_prompt=edited_decomp_system_prompt,
                                                    decomposition_user_prompt_template=edited_decomp_user_prompt_template,
                                                    solver_system_prompt=edited_solver_system_prompt,
                                                    solver_user_prompt_template=edited_solver_user_prompt_template,
                                                    patcher_system_prompt=edited_patcher_system_prompt,
                                                    patcher_user_prompt_template=edited_patcher_user_prompt_template,
                                                    assembler_system_prompt=edited_assembler_system_prompt,
                                                    assembler_user_prompt_template=edited_assembler_user_prompt_template,
                                                    red_team_system_prompt=edited_red_team_system_prompt,
                                                    red_team_user_prompt_template=edited_red_team_user_prompt_template,
                                                    gold_team_system_prompt=edited_gold_team_system_prompt,
                                                    gold_team_user_prompt_template=edited_gold_team_user_prompt_template,
                                                    team_type=edited_team_type)  # NEW: Include team_type
                                if team_manager.update_team(updated_team):
                                    st.success(f"Team '{edited_team_name}' updated successfully!")
                                    st.session_state.team_manager = TeamManager() # Reload to refresh UI
                                else:
                                    st.error(f"Failed to update team '{team.name}'. A team with name '{edited_team_name}' might already exist.")
                            else:
                                st.error("Please fill in team name and at least one model ID for the updated team.")
                    
                    if st.button(f"Delete Team '{team.name}'", key=f"delete_team_{team.name}"):
                        if team_manager.delete_team(team.name):
                            st.success(f"Team '{team.name}' deleted.")
                            st.session_state.team_manager = TeamManager() # Reload to refresh UI
                        else:
                            st.error(f"Failed to delete team '{team.name}'.")

def render_gauntlet_designer():
    """Renders the UI UI for designing and managing Gauntlet definitions. Allows users to create, view, edit, and delete gauntlets."""
    st.header("🛡️ Gauntlet Designer")
    st.write("Create, view, edit, and delete your programmable Gauntlet definitions.")

    gauntlet_manager: GauntletManager = st.session_state.gauntlet_manager
    team_manager: TeamManager = st.session_state.team_manager
    
    available_teams = team_manager.get_all_teams()
    team_names = [team.name for team in available_teams]

    # --- Create New Gauntlet ---
    with st.expander("➕ Create New Gauntlet", expanded=False):
        with st.form("new_gauntlet_form"):
            gauntlet_name = st.text_input("Gauntlet Name", key="new_gauntlet_name")
            gauntlet_description = st.text_area("Description", key="new_gauntlet_description")
            
            if not team_names:
                st.warning("Please create at least one Team in the Team Manager before creating a Gauntlet.")
                team_for_gauntlet = None
            else:
                team_for_gauntlet = st.selectbox("Team to run this Gauntlet", team_names, key="new_gauntlet_team")
            
            # NEW: Gauntlet Type Selection
            gauntlet_type = st.selectbox("Gauntlet Type", [
                "standard", "adaptive", "hierarchical", "competitive", "collaborative",
                "adversarial", "formal_verification", "statistical", 
                "domain_physics", "domain_finance", "domain_chemistry", 
                "domain_engineering", "domain_web3", "multi_objective", 
                "evolutionary", "temporal", "cross_validation"
            ], index=0, help="Select the specialized evaluation logic for this gauntlet.", key="new_gauntlet_type")

            st.subheader("Gauntlet Rounds")
            num_rounds = st.number_input("Number of Rounds", min_value=1, value=1, key="num_new_rounds")
            
            new_rounds = []
            for i in range(num_rounds):
                st.markdown(f"**Round {i+1} Configuration**")
                col1, col2 = st.columns(2)
                with col1:
                    quorum_req = st.number_input(f"Quorum: Required Approvals", min_value=1, value=1, key=f"round_{i}_quorum_req")
                    # NEW: Voting Strategy Selection
                    voting_strategy = st.selectbox(f"Voting Strategy", ["fixed_quorum", "first_to_ahead_by_k"], index=0, key=f"round_{i}_voting_strategy")
                with col2:
                    quorum_from = st.number_input(f"Quorum: From Panel Size", min_value=1, value=1, key=f"round_{i}_quorum_from")
                    # NEW: Margin K for First-to-ahead-by-k
                    margin_k = st.number_input(f"Margin K (for first_to_ahead_by_k)", min_value=1, value=3, key=f"round_{i}_margin_k")
                    max_dynamic_votes = st.number_input(f"Max Dynamic Votes", min_value=1, value=100, key=f"round_{i}_max_dynamic_votes")

                min_conf = st.slider(f"Minimum Overall Confidence (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key=f"round_{i}_min_conf")
                max_var = st.number_input(f"Max Score Variance (Optional)", min_value=0.0, value=0.0, step=0.01, key=f"round_{i}_max_var")

                # Per-judge requirements
                st.caption("Per-Judge Requirements (JSON format, optional)")
                per_judge_json = st.text_area(f"{{'model_id': {{'min_score': 0.9}}}}", key=f"round_{i}_per_judge_json")
                per_judge_reqs = {}
                if per_judge_json:
                    try:
                        per_judge_reqs = json.loads(per_judge_json)
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON for per-judge requirements in Round {i+1}.")
                        continue

                new_rounds.append(GauntletRoundRule(
                    round_number=i+1,
                    quorum_required_approvals=quorum_req,
                    quorum_from_panel_size=quorum_from,
                    min_overall_confidence=min_conf,
                    max_score_variance=max_var if max_var > 0 else None,
                    voting_strategy=voting_strategy,  # NEW: Include voting strategy
                    margin_k=margin_k,  # NEW: Include margin K
                    max_dynamic_votes=max_dynamic_votes,  # NEW: Include max dynamic votes
                    per_judge_requirements=per_judge_reqs
                ))
            
            # Gauntlet specific settings (Red/Blue)
            st.subheader("Gauntlet Specific Settings")
            attack_modes_str = st.text_input("Red Team Attack Modes (comma-separated)", key="new_attack_modes")
            attack_modes = [m.strip() for m in attack_modes_str.split(',') if m.strip()]

            generation_mode = st.selectbox("Blue Team Generation Mode", ["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"], key="new_gen_mode")

            # NEW: Red Flags Configuration
            st.subheader("Red Flags Configuration")
            with st.expander("Configure Passive Red Flags", expanded=False):
                max_token_length = st.number_input("Max Token Length", min_value=1, value=2000, help="Reject if verbosity indicates confusion", key="new_max_token_length")
                strict_format_adherence = st.checkbox("Strict Format Adherence", value=True, help="Reject if JSON repair is needed", key="new_strict_format")

                forbidden_phrases_str = st.text_input("Forbidden Phrases (comma-separated)", value="I apologize, I'm confused, As an AI", help="Phrases that indicate the model is confused", key="new_forbidden_phrases")
                forbidden_phrases = [p.strip() for p in forbidden_phrases_str.split(',') if p.strip()]

                # Create red_flags dictionary
                red_flags = {
                    "max_token_length": max_token_length,
                    "strict_format_adherence": strict_format_adherence,
                    "forbidden_phrases": forbidden_phrases
                }

            submitted = st.form_submit_button("Create Gauntlet")
            if submitted:
                if gauntlet_name and team_for_gauntlet and new_rounds:
                    new_gauntlet = GauntletDefinition(
                        name=gauntlet_name,
                        team_name=team_for_gauntlet,
                        rounds=new_rounds,
                        description=gauntlet_description,
                        attack_modes=attack_modes,
                        generation_mode=generation_mode,
                        gauntlet_type=gauntlet_type, # NEW: Include gauntlet_type
                        red_flags=red_flags  # NEW: Include red flags
                    )
                    if gauntlet_manager.create_gauntlet(new_gauntlet):
                        st.success(f"Gauntlet '{gauntlet_name}' created successfully!")
                        st.session_state.gauntlet_manager = GauntletManager() # Reload to refresh UI
                    else:
                        st.error(f"Gauntlet '{gauntlet_name}' already exists.")
                else:
                    st.error("Please fill in gauntlet name, select a team, and configure at least one round.")

    # --- View/Edit/Delete Existing Gauntlets ---
    st.subheader("Existing Gauntlets")
    gauntlets = gauntlet_manager.get_all_gauntlets()
    if not gauntlets:
        st.info("No gauntlets created yet.")
    else:
        for gauntlet in gauntlets:
            with st.container(border=True):
                st.markdown(f"**{gauntlet.name}** (Run by Team: `{gauntlet.team_name}`)")
                st.caption(gauntlet.description or "No description.")
                
                with st.expander(f"View/Edit Rounds for {gauntlet.name}", expanded=False):
                    for i, round_rule in enumerate(gauntlet.rounds):
                        st.markdown(f"**Round {round_rule.round_number}**")
                        st.write(f"Quorum: {round_rule.quorum_required_approvals} of {round_rule.quorum_from_panel_size} approvals")
                        st.write(f"Min Confidence: {round_rule.min_overall_confidence}")
                        if round_rule.max_score_variance is not None:
                            st.write(f"Max Variance: {round_rule.max_score_variance}")
                        # NEW: Show voting strategy and margin k
                        st.write(f"Voting Strategy: {round_rule.voting_strategy}")
                        if round_rule.voting_strategy == "first_to_ahead_by_k" and round_rule.margin_k is not None:
                            st.write(f"Margin K: {round_rule.margin_k}")
                        if round_rule.max_dynamic_votes is not None:
                            st.write(f"Max Dynamic Votes: {round_rule.max_dynamic_votes}")
                        if round_rule.per_judge_requirements:
                            st.json(round_rule.per_judge_requirements)

                    st.write(f"Attack Modes: {', '.join(gauntlet.attack_modes) if gauntlet.attack_modes else 'N/A'}")
                    st.write(f"Generation Mode: {gauntlet.generation_mode}")
                    # NEW: Show red flags
                    st.write(f"Red Flags: {gauntlet.red_flags if hasattr(gauntlet, 'red_flags') and gauntlet.red_flags else 'Not configured'}")

                    # NEW: Add edit form for gauntlet
                    with st.form(f"edit_gauntlet_form_{gauntlet.name}"):
                        st.subheader(f"Edit Gauntlet: {gauntlet.name}")
                        edited_gauntlet_name = st.text_input("Gauntlet Name", value=gauntlet.name, key=f"edit_gauntlet_name_{gauntlet.name}")
                        edited_gauntlet_description = st.text_area("Description", value=gauntlet.description or "", key=f"edit_gauntlet_description_{gauntlet.name}")

                        # Team selection for gauntlet
                        all_teams = team_manager.get_all_teams()
                        team_names = [team.name for team in all_teams]
                        current_team_index = team_names.index(gauntlet.team_name) if gauntlet.team_name in team_names else 0
                        edited_team_name = st.selectbox("Team to run this Gauntlet", team_names, index=current_team_index, key=f"edit_gauntlet_team_{gauntlet.name}")

                        # Round editing
                        st.subheader("Rounds Configuration")
                        edited_num_rounds = st.number_input("Number of Rounds", min_value=1, value=len(gauntlet.rounds), key=f"edit_num_rounds_{gauntlet.name}")

                        edited_rounds = []
                        for i in range(edited_num_rounds):
                            if i < len(gauntlet.rounds):
                                current_round = gauntlet.rounds[i]
                            else:
                                # If we have more rounds than before, create a default round
                                current_round = GauntletRoundRule(
                                    round_number=i+1,
                                    quorum_required_approvals=1,
                                    quorum_from_panel_size=1,
                                    min_overall_confidence=0.75
                                )

                            st.markdown(f"**Round {i+1} Configuration**")
                            col1, col2 = st.columns(2)
                            with col1:
                                edited_quorum_req = st.number_input(f"Quorum: Required Approvals (Round {i+1})", min_value=1, value=current_round.quorum_required_approvals, key=f"edit_round_{i}_quorum_req_{gauntlet.name}")
                                # NEW: Voting Strategy Selection
                                edited_voting_strategy = st.selectbox(f"Voting Strategy (Round {i+1})", ["fixed_quorum", "first_to_ahead_by_k"], index=["fixed_quorum", "first_to_ahead_by_k"].index(getattr(current_round, 'voting_strategy', 'fixed_quorum')), key=f"edit_round_{i}_voting_strategy_{gauntlet.name}")
                            with col2:
                                edited_quorum_from = st.number_input(f"Quorum: From Panel Size (Round {i+1})", min_value=1, value=current_round.quorum_from_panel_size, key=f"edit_round_{i}_quorum_from_{gauntlet.name}")
                                # NEW: Margin K for First-to-ahead-by-k
                                edited_margin_k = st.number_input(f"Margin K (for first_to_ahead_by_k) (Round {i+1})", min_value=1, value=getattr(current_round, 'margin_k', 3), key=f"edit_round_{i}_margin_k_{gauntlet.name}")
                                edited_max_dynamic_votes = st.number_input(f"Max Dynamic Votes (Round {i+1})", min_value=1, value=getattr(current_round, 'max_dynamic_votes', 100), key=f"edit_round_{i}_max_dynamic_votes_{gauntlet.name}")

                            edited_min_conf = st.slider(f"Minimum Overall Confidence (Round {i+1})", min_value=0.0, max_value=1.0, value=current_round.min_overall_confidence, step=0.05, key=f"edit_round_{i}_min_conf_{gauntlet.name}")
                            edited_max_var = st.number_input(f"Max Score Variance (Round {i+1})", min_value=0.0, value=current_round.max_score_variance or 0.0, step=0.01, key=f"edit_round_{i}_max_var_{gauntlet.name}")

                            # Per-judge requirements
                            st.caption("Per-Judge Requirements (JSON format, optional) (Round {i+1})")
                            current_per_judge_json = json.dumps(current_round.per_judge_requirements) if current_round.per_judge_requirements else "{}"
                            edited_per_judge_json = st.text_area(f"{{'model_id': {{'min_score': 0.9}}}}", value=current_per_judge_json, key=f"edit_round_{i}_per_judge_json_{gauntlet.name}")
                            edited_per_judge_reqs = {}
                            if edited_per_judge_json:
                                try:
                                    edited_per_judge_reqs = json.loads(edited_per_judge_json)
                                except json.JSONDecodeError:
                                    st.error(f"Invalid JSON for per-judge requirements in Round {i+1}.")
                                    continue

                            edited_rounds.append(GauntletRoundRule(
                                round_number=i+1,
                                quorum_required_approvals=edited_quorum_req,
                                quorum_from_panel_size=edited_quorum_from,
                                min_overall_confidence=edited_min_conf,
                                max_score_variance=edited_max_var if edited_max_var > 0 else None,
                                voting_strategy=edited_voting_strategy,  # NEW: Include voting strategy
                                margin_k=edited_margin_k,  # NEW: Include margin K
                                max_dynamic_votes=edited_max_dynamic_votes,  # NEW: Include max dynamic votes
                                per_judge_requirements=edited_per_judge_reqs
                            ))

                        # Gauntlet specific settings
                        edited_attack_modes_str = st.text_input("Red Team Attack Modes (comma-separated)", value=", ".join(gauntlet.attack_modes) if gauntlet.attack_modes else "", key=f"edit_attack_modes_{gauntlet.name}")
                        edited_attack_modes = [m.strip() for m in edited_attack_modes_str.split(',') if m.strip()]

                        edited_generation_mode = st.selectbox("Blue Team Generation Mode", ["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"], index=["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"].index(gauntlet.generation_mode), key=f"edit_gen_mode_{gauntlet.name}")

                        # NEW: Red Flags Configuration
                        st.subheader("Red Flags Configuration")
                        current_red_flags = getattr(gauntlet, 'red_flags', {
                            "max_token_length": 2000,
                            "strict_format_adherence": True,
                            "forbidden_phrases": ["I apologize", "I'm confused", "As an AI"]
                        })
                        with st.expander("Configure Passive Red Flags", expanded=True):
                            edited_max_token_length = st.number_input("Max Token Length", min_value=1, value=current_red_flags.get("max_token_length", 2000), help="Reject if verbosity indicates confusion", key=f"edit_max_token_length_{gauntlet.name}")
                            edited_strict_format_adherence = st.checkbox("Strict Format Adherence", value=current_red_flags.get("strict_format_adherence", True), help="Reject if JSON repair is needed", key=f"edit_strict_format_{gauntlet.name}")

                            current_forbidden_phrases = ", ".join(current_red_flags.get("forbidden_phrases", ["I apologize", "I'm confused", "As an AI"]))
                            edited_forbidden_phrases_str = st.text_input("Forbidden Phrases (comma-separated)", value=current_forbidden_phrases, help="Phrases that indicate the model is confused", key=f"edit_forbidden_phrases_{gauntlet.name}")
                            edited_forbidden_phrases = [p.strip() for p in edited_forbidden_phrases_str.split(',') if p.strip()]

                            # Create edited_red_flags dictionary
                            edited_red_flags = {
                                "max_token_length": edited_max_token_length,
                                "strict_format_adherence": edited_strict_format_adherence,
                                "forbidden_phrases": edited_forbidden_phrases
                            }

                        edit_submitted = st.form_submit_button(f"Update Gauntlet '{gauntlet.name}'")
                        if edit_submitted:
                            edited_gauntlet = GauntletDefinition(
                                name=edited_gauntlet_name,
                                team_name=edited_team_name,
                                rounds=edited_rounds,
                                description=edited_gauntlet_description,
                                attack_modes=edited_attack_modes,
                                generation_mode=edited_generation_mode,
                                red_flags=edited_red_flags  # NEW: Include red flags
                            )
                            if gauntlet_manager.update_gauntlet(edited_gauntlet):
                                st.success(f"Gauntlet '{edited_gauntlet_name}' updated successfully!")
                                st.session_state.gauntlet_manager = GauntletManager() # Reload to refresh UI
                            else:
                                st.error(f"Failed to update gauntlet '{gauntlet.name}'.")

                    if st.button(f"Delete Gauntlet '{gauntlet.name}'", key=f"delete_gauntlet_{gauntlet.name}"):
                        if gauntlet_manager.delete_gauntlet(gauntlet.name):
                            st.success(f"Gauntlet '{gauntlet.name}' deleted.")
                            st.session_state.gauntlet_manager = GauntletManager() # Reload to refresh UI
                        else:
                            st.error(f"Failed to delete gauntlet '{gauntlet.name}'.")

def render_manual_review_panel(decomposition_plan: DecompositionPlan) -> tuple[str, Optional[DecompositionPlan]]:
    """
    Renders the manual review panel for the user to approve/reject the decomposition plan.
    Returns a tuple of (status, plan), where status is one of "approved", "rejected", or "pending".
    Supports auto-approval based on configurable criteria.
    """
    from auto_approval import auto_approve_plan, get_default_auto_approval_criteria, validate_decomposition_plan
    from collaboration_manager import collaboration_manager
    import hashlib
    
    st.header("📝 Manual Review & Override")

    if "manual_review_session_id" not in st.session_state:
        st.session_state.manual_review_session_id = None
    if "manual_review_user_id" not in st.session_state:
        st.session_state.manual_review_user_id = st.session_state.get(
            "user_id",
            st.session_state.get("username", "local_user")
        )

    doc_id = "decomposition_plan_" + hashlib.md5(decomposition_plan.problem_statement.encode("utf-8")).hexdigest()
    with st.expander("Collaboration Session", expanded=False):
        user_id = st.text_input(
            "User ID",
            value=st.session_state.manual_review_user_id,
            key="manual_review_user_id"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create Session", key="manual_review_create_session"):
                session_info = collaboration_manager.initialize_collaborative_session(
                    user_id,
                    doc_id
                )
                st.session_state.manual_review_session_id = session_info["session_id"]
                collaboration_manager.record_audit_event(
                    session_info["session_id"],
                    user_id,
                    "session_created",
                    {"document_id": doc_id}
                )
                st.success(f"Session created: {session_info['session_id']}")
        with col_b:
            join_session_id = st.text_input("Join Session ID", key="manual_review_join_session_id")
            if st.button("Join Session", key="manual_review_join_session"):
                if collaboration_manager.join_collaborative_session(join_session_id, user_id):
                    st.session_state.manual_review_session_id = join_session_id
                    collaboration_manager.record_audit_event(
                        join_session_id,
                        user_id,
                        "session_joined",
                        {"document_id": doc_id}
                    )
                    st.success("Joined collaboration session.")
                else:
                    st.error("Could not join session.")

        session_id = st.session_state.manual_review_session_id
        if session_id:
            session_state = collaboration_manager.get_session_state(session_id)
            if session_state:
                participants_count = len(session_state.get("participants", []))
                st.caption(f"Active Session: {session_id} ({participants_count} participants)")
            audit_entries = collaboration_manager.get_audit_log(session_id)
            if audit_entries:
                with st.expander("Audit Trail", expanded=False):
                    for entry in audit_entries[-10:]:
                        st.write(f"{entry['timestamp']} - {entry['user_id']}: {entry['event_type']}")
    
    # Check for auto-approval first
    if decomposition_plan.auto_approval_enabled:
        criteria = decomposition_plan.auto_approval_criteria or get_default_auto_approval_criteria()
        criteria["enabled"] = True  # Ensure enabled flag is set
        
        should_auto_approve, reasons = auto_approve_plan(decomposition_plan, criteria)
        
        if should_auto_approve:
            st.success("[OK] Plan automatically approved based on criteria!")
            with st.expander("Auto-Approval Reasons"):
                for reason in reasons:
                    st.write(f"- {reason}")
            
            # Return approved plan immediately
            return "approved", decomposition_plan
        else:
            st.warning("[WARN] Plan did not meet auto-approval criteria. Manual review required.")
            with st.expander("Auto-Approval Check Results"):
                for reason in reasons:
                    st.write(f"- {reason}")
    
    st.info("Review the AI-generated decomposition plan. You can edit any aspect of the plan before approving it.")

    # Use a session state object to hold edits, preventing loss on rerun.
    # This ensures that user modifications persist across UI reruns until the plan is approved or rejected.
    if 'edited_sub_problems' not in st.session_state:
        st.session_state.edited_sub_problems = {sp.id: sp for sp in decomposition_plan.sub_problems}

    
    st.markdown(f"**Problem Statement**: {decomposition_plan.problem_statement}")
    st.markdown(f"**Analyzed Context Summary**: {decomposition_plan.analyzed_context.get('summary', 'N/A')}")

    # Batch operations section
    with st.expander("🔄 Batch Operations", expanded=False):
        from batch_operations import render_batch_operations_ui
        st.session_state.edited_sub_problems = render_batch_operations_ui(st.session_state.edited_sub_problems)
    
    # Dependency visualization section
    with st.expander("📊 Dependency Visualization", expanded=False):
        from dependency_visualizer import render_dependency_visualization
        # Create temporary plan for visualization
        temp_plan_for_viz = DecompositionPlan(
            problem_statement=decomposition_plan.problem_statement,
            analyzed_context=decomposition_plan.analyzed_context,
            sub_problems=list(st.session_state.edited_sub_problems.values()),
            max_refinement_loops=decomposition_plan.max_refinement_loops,
            mdap_enabled=decomposition_plan.mdap_enabled,
            mdap_config=decomposition_plan.mdap_config,
            maker_enabled=decomposition_plan.maker_enabled,
            maker_config=decomposition_plan.maker_config,
            assembler_team_name=decomposition_plan.assembler_team_name,
            final_red_team_gauntlet_name=decomposition_plan.final_red_team_gauntlet_name,
            final_gold_team_gauntlet_name=decomposition_plan.final_gold_team_gauntlet_name
        )
        render_dependency_visualization(temp_plan_for_viz)
    
    st.subheader("Sub-Problems")
    # Iterate through each sub-problem and provide an editable UI.
    for i, sub_problem in enumerate(decomposition_plan.sub_problems):
        with st.expander(f"Sub-Problem {sub_problem.id}: {sub_problem.description[:80]}...", expanded=False):
            # Each sub-problem is its own form to allow individual updates and prevent full form submission on every change.
            with st.form(f"edit_sub_problem_form_{sub_problem.id}"):
                current_sp_state = st.session_state.edited_sub_problems[sub_problem.id]
                
                # Editable fields for sub-problem details.
                edited_description = st.text_area("Description", value=current_sp_state.description, key=f"desc_{sub_problem.id}")
                edited_dependencies_str = st.text_input("Dependencies (comma-separated IDs)", value=", ".join(current_sp_state.dependencies), key=f"deps_{sub_problem.id}")
                
                st.markdown("---")
                st.markdown("**AI Suggestions (Editable)**")
                edited_ai_suggested_evolution_mode = st.text_input("Suggested Evolution Mode", value=current_sp_state.ai_suggested_evolution_mode, key=f"ai_mode_{sub_problem.id}")
                edited_ai_suggested_complexity_score = st.number_input("Suggested Complexity Score (1-10)", min_value=1, max_value=10, value=current_sp_state.ai_suggested_complexity_score, key=f"ai_comp_{sub_problem.id}")
                edited_ai_suggested_evaluation_prompt = st.text_area("Suggested Evaluation Prompt", value=current_sp_state.ai_suggested_evaluation_prompt, key=f"ai_eval_prompt_{sub_problem.id}")
                edited_content_type = st.selectbox("Content Type", options=[
                    "text_general", "code_python", "code_javascript", "document_legal", "document_medical", "document_technical", "prompt", "protocol"
                ], index=[
                    "text_general", "code_python", "code_javascript", "document_legal", "document_medical", "document_technical", "prompt", "protocol"
                ].index(current_sp_state.content_type) if current_sp_state.content_type in [
                    "text_general", "code_python", "code_javascript", "document_legal", "document_medical", "document_technical", "prompt", "protocol"
                ] else 0, key=f"content_type_{sub_problem.id}")
                
                st.markdown("---")
                st.markdown("**User Overrides (Select Teams & Gauntlets)**")
                
                # Retrieve available teams and gauntlets for dropdown selections.
                team_manager: TeamManager = st.session_state.team_manager
                gauntlet_manager: GauntletManager = st.session_state.gauntlet_manager
                
                blue_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Blue"]
                red_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Red"]
                gold_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Gold"]

                # Dropdowns for user to override AI-suggested teams and gauntlets.
                edited_solver_team_name = st.selectbox("Solver Team (Blue)", blue_teams, index=blue_teams.index(current_sp_state.solver_team_name) if current_sp_state.solver_team_name in blue_teams else 0, key=f"solver_team_{sub_problem.id}", disabled=not blue_teams)
                edited_red_gauntlet_name = st.selectbox("Red Team Gauntlet", ["None"] + red_gauntlets, index=(red_gauntlets.index(current_sp_state.red_team_gauntlet_name) + 1) if current_sp_state.red_team_gauntlet_name in red_gauntlets else 0, key=f"red_gauntlet_{sub_problem.id}", disabled=not red_gauntlets)
                edited_gold_gauntlet_name = st.selectbox("Gold Team Gauntlet", gold_gauntlets, index=gold_gauntlets.index(current_sp_state.gold_team_gauntlet_name) if current_sp_state.gold_team_gauntlet_name in gold_gauntlets else 0, key=f"gold_gauntlet_{sub_problem.id}", disabled=not gold_gauntlets)

                # NEW: Atomic Decomposition Configuration
                st.markdown("---")
                st.markdown("**Atomic Decomposition Configuration**")
                current_atomic_mode = getattr(current_sp_state, 'atomic_mode', False)
                current_decomposition_depth = getattr(current_sp_state, 'decomposition_depth', 0)
                current_acceptance_criteria = getattr(current_sp_state, 'acceptance_criteria', [])
                current_solution_requirements = getattr(current_sp_state, 'solution_requirements', {})
                current_specific_constraints = getattr(current_sp_state, 'specific_constraints', [])

                edited_atomic_mode = st.checkbox("Enable Atomic Mode", value=current_atomic_mode, help="If enabled, this sub-problem will be decomposed recursively into micro-steps", key=f"atomic_mode_{sub_problem.id}")
                edited_decomposition_depth = st.number_input("Decomposition Depth", min_value=0, value=current_decomposition_depth, help="Current depth of decomposition (0 = original, >0 = atomic)", key=f"decomposition_depth_{sub_problem.id}")

                # Display and edit acceptance criteria
                current_acceptance_criteria_str = ", ".join(current_acceptance_criteria)
                edited_acceptance_criteria_str = st.text_input("Acceptance Criteria (comma-separated)", value=current_acceptance_criteria_str, help="Criteria for solution acceptance", key=f"acceptance_criteria_{sub_problem.id}")
                edited_acceptance_criteria = [c.strip() for c in edited_acceptance_criteria_str.split(',') if c.strip()]

                # Display and edit specific constraints
                current_specific_constraints_str = ", ".join(current_specific_constraints)
                edited_specific_constraints_str = st.text_input("Specific Constraints (comma-separated)", value=current_specific_constraints_str, help="Constraints specific to this sub-problem", key=f"specific_constraints_{sub_problem.id}")
                edited_specific_constraints = [c.strip() for c in edited_specific_constraints_str.split(',') if c.strip()]

                # For solution requirements, use JSON editor
                edited_solution_requirements_json = st.text_area("Solution Requirements (JSON)", value=json.dumps(current_solution_requirements, indent=2), help="Requirements for the solution", key=f"solution_requirements_{sub_problem.id}")

                st.caption("Specific Evolution Parameters (JSON format, optional)")
                edited_evolution_params_json = st.text_area("{}", value=json.dumps(current_sp_state.evolution_params, indent=2), key=f"evol_params_{sub_problem.id}")

                submitted = st.form_submit_button("Update Sub-Problem")
                if submitted:
                    try:
                        edited_evolution_params = json.loads(edited_evolution_params_json) if edited_evolution_params_json else {}
                        edited_dependencies = [d.strip() for d in edited_dependencies_str.split(',') if d.strip()]
                        edited_solution_requirements = json.loads(edited_solution_requirements_json) if edited_solution_requirements_json else {}

                        # Set to None if the selectbox was disabled (i.e., no options were available).
                        final_solver_team_name = edited_solver_team_name if blue_teams else None
                        final_red_gauntlet_name = edited_red_gauntlet_name if red_gauntlets else None
                        final_gold_gauntlet_name = edited_gold_gauntlet_name if gold_gauntlets else None

                        new_sub_problem = SubProblem(
                            id=sub_problem.id,
                            description=edited_description,
                            dependencies=edited_dependencies,
                            ai_suggested_evolution_mode=edited_ai_suggested_evolution_mode,
                            ai_suggested_complexity_score=edited_ai_suggested_complexity_score,
                            ai_suggested_evaluation_prompt=edited_ai_suggested_evaluation_prompt,
                            content_type=edited_content_type,
                            solver_team_name=final_solver_team_name,
                            red_team_gauntlet_name=final_red_gauntlet_name if final_red_gauntlet_name != "None" else None,
                            gold_team_gauntlet_name=final_gold_gauntlet_name,
                            evolution_params=edited_evolution_params,
                            # NEW: Atomic decomposition fields
                            atomic_mode=edited_atomic_mode,
                            decomposition_depth=edited_decomposition_depth,
                            acceptance_criteria=edited_acceptance_criteria,
                            solution_requirements=edited_solution_requirements,
                            specific_constraints=edited_specific_constraints
                        )
                        changes = {}
                        field_pairs = {
                            "description": (current_sp_state.description, new_sub_problem.description),
                            "dependencies": (current_sp_state.dependencies, new_sub_problem.dependencies),
                            "ai_suggested_evolution_mode": (current_sp_state.ai_suggested_evolution_mode, new_sub_problem.ai_suggested_evolution_mode),
                            "ai_suggested_complexity_score": (current_sp_state.ai_suggested_complexity_score, new_sub_problem.ai_suggested_complexity_score),
                            "ai_suggested_evaluation_prompt": (current_sp_state.ai_suggested_evaluation_prompt, new_sub_problem.ai_suggested_evaluation_prompt),
                            "content_type": (current_sp_state.content_type, new_sub_problem.content_type),
                            "solver_team_name": (current_sp_state.solver_team_name, new_sub_problem.solver_team_name),
                            "red_team_gauntlet_name": (current_sp_state.red_team_gauntlet_name, new_sub_problem.red_team_gauntlet_name),
                            "gold_team_gauntlet_name": (current_sp_state.gold_team_gauntlet_name, new_sub_problem.gold_team_gauntlet_name),
                            "evolution_params": (current_sp_state.evolution_params, new_sub_problem.evolution_params),
                            "atomic_mode": (getattr(current_sp_state, "atomic_mode", False), getattr(new_sub_problem, "atomic_mode", False)),
                            "decomposition_depth": (getattr(current_sp_state, "decomposition_depth", 0), getattr(new_sub_problem, "decomposition_depth", 0)),
                            "acceptance_criteria": (getattr(current_sp_state, "acceptance_criteria", []), getattr(new_sub_problem, "acceptance_criteria", [])),
                            "solution_requirements": (getattr(current_sp_state, "solution_requirements", {}), getattr(new_sub_problem, "solution_requirements", {})),
                            "specific_constraints": (getattr(current_sp_state, "specific_constraints", []), getattr(new_sub_problem, "specific_constraints", [])),
                        }
                        for field_name, (before_value, after_value) in field_pairs.items():
                            if before_value != after_value:
                                changes[field_name] = {
                                    "before": before_value,
                                    "after": after_value
                                }

                        # Update the sub-problem in session state with the user's edits.
                        st.session_state.edited_sub_problems[sub_problem.id] = new_sub_problem
                        session_id = st.session_state.get("manual_review_session_id")
                        user_id = st.session_state.get("manual_review_user_id", "local_user")
                        if changes:
                            collaboration_manager.record_audit_event(
                                session_id,
                                user_id,
                                "sub_problem_updated",
                                {"sub_problem_id": sub_problem.id, "changes": changes}
                            )
                        st.success(f"Sub-Problem {sub_problem.id} updated in draft plan.")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON for evolution parameters or solution requirements in Sub-Problem {sub_problem.id}: {str(e)}")

    st.markdown("---")
    
    # Validation check
    from auto_approval import validate_decomposition_plan
    final_sub_problems_for_validation = list(st.session_state.edited_sub_problems.values())
    temp_plan_for_validation = DecompositionPlan(
        problem_statement=decomposition_plan.problem_statement,
        analyzed_context=decomposition_plan.analyzed_context,
        sub_problems=final_sub_problems_for_validation,
        max_refinement_loops=decomposition_plan.max_refinement_loops,
        mdap_enabled=decomposition_plan.mdap_enabled,
        mdap_config=decomposition_plan.mdap_config,
        maker_enabled=decomposition_plan.maker_enabled,
        maker_config=decomposition_plan.maker_config,
        assembler_team_name=decomposition_plan.assembler_team_name,
        final_red_team_gauntlet_name=decomposition_plan.final_red_team_gauntlet_name,
        final_gold_team_gauntlet_name=decomposition_plan.final_gold_team_gauntlet_name
    )
    
    is_valid, validation_issues = validate_decomposition_plan(temp_plan_for_validation)
    
    if not is_valid:
        st.warning("[WARN] Plan has validation issues:")
        for issue in validation_issues:
            st.write(f"- {issue}")
    else:
        st.success("[OK] Plan validation passed")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        # Button to approve the entire decomposition plan.
        if st.button("[OK] Approve Plan", key="approve_plan_button", type="primary", disabled=not is_valid):
            # Reconstruct the DecompositionPlan from the edited sub-problems in session state.
            final_sub_problems = list(st.session_state.edited_sub_problems.values())
            
            approved_plan = DecompositionPlan(
                problem_statement=decomposition_plan.problem_statement,
                analyzed_context=decomposition_plan.analyzed_context,
                sub_problems=final_sub_problems,
                max_refinement_loops=decomposition_plan.max_refinement_loops,
                mdap_enabled=decomposition_plan.mdap_enabled,
                mdap_config=decomposition_plan.mdap_config,
                maker_enabled=decomposition_plan.maker_enabled,
                maker_config=decomposition_plan.maker_config,
                assembler_team_name=decomposition_plan.assembler_team_name,
                final_red_team_gauntlet_name=decomposition_plan.final_red_team_gauntlet_name,
                final_gold_team_gauntlet_name=decomposition_plan.final_gold_team_gauntlet_name
            )
            del st.session_state.edited_sub_problems # Clean up session state after approval.
            
            collaboration_manager.record_audit_event(
                st.session_state.get("manual_review_session_id"),
                st.session_state.get("manual_review_user_id", "local_user"),
                "plan_approved",
                {"sub_problem_count": len(final_sub_problems)}
            )
            return "approved", approved_plan

    with col2:
        # Button to reject the entire decomposition plan.
        if st.button("[FAIL] Reject Plan", key="reject_plan_button"):
            st.error("Plan rejected. Please modify the initial problem or AI settings and try again.")
            collaboration_manager.record_audit_event(
                st.session_state.get("manual_review_session_id"),
                st.session_state.get("manual_review_user_id", "local_user"),
                "plan_rejected",
                {"reason": "manual_reject"}
            )
            del st.session_state.edited_sub_problems # Clean up session state after rejection.
            return "rejected", None
    
    return "pending", None # Plan not yet approved or rejected.

# ============================================================================
# Dependency Visualization Components
# ============================================================================

def render_dependency_graph(
    sub_problems: List[Dict],
    show_critical_path: bool = False,
    highlight_circular: bool = True
) -> None:
    """
    Renders an interactive dependency graph.
    
    Args:
        sub_problems: List of sub-problem dictionaries with dependencies
        show_critical_path: Whether to highlight critical path
        highlight_circular: Whether to highlight circular dependencies
    """
    from ui_utils import with_error_handling, render_chart_with_fallback
    from ui_models import GraphNode, GraphEdge, DependencyGraphData, NodeStatus, DependencyType
    from dependency_visualizer import DependencyVisualizer
    import plotly.graph_objects as go
    import networkx as nx
    
    st.subheader("📊 Dependency Graph")
    
    if not sub_problems:
        st.info("No sub-problems to visualize.")
        return
    
    try:
        # Build NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for sp in sub_problems:
            status_map = {
                "pending": NodeStatus.PENDING,
                "in_progress": NodeStatus.IN_PROGRESS,
                "completed": NodeStatus.COMPLETED,
                "failed": NodeStatus.FAILED
            }
            status = status_map.get(sp.get("status", "pending"), NodeStatus.PENDING)
            
            G.add_node(
                sp["id"],
                label=sp.get("description", sp["id"])[:50],
                status=status.value,
                team=sp.get("solver_team_name", ""),
                complexity=sp.get("ai_suggested_complexity_score", 0)
            )
        
        # Add edges
        for sp in sub_problems:
            for dep in sp.get("dependencies", []):
                if dep in [s["id"] for s in sub_problems]:
                    G.add_edge(dep, sp["id"])
        
        # Detect circular dependencies
        circular_deps = []
        if highlight_circular:
            try:
                cycles = list(nx.simple_cycles(G))
                circular_deps = cycles
                if cycles:
                    st.error(f"[WARN] {len(cycles)} circular dependencies detected!")
                    with st.expander("View Circular Dependencies"):
                        for i, cycle in enumerate(cycles, 1):
                            st.write(f"**Cycle {i}:** {' -> '.join(cycle)}")
            except Exception as e:
                logger.debug("Failed to detect circular dependencies: %s", e)
        
        # Calculate critical path
        critical_path_nodes = []
        if show_critical_path and nx.is_directed_acyclic_graph(G):
            try:
                # Find longest path (critical path)
                longest_path = nx.dag_longest_path(G)
                critical_path_nodes = longest_path
                st.info(f"Critical path length: {len(longest_path)} nodes")
            except Exception as e:
                logger.debug("Failed to calculate critical path: %s", e)
        
        # Calculate layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            name='Dependencies'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        color_map = {
            "pending": "#808080",
            "in_progress": "#FFA500",
            "completed": "#00FF00",
            "failed": "#FF0000"
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            status = node_data.get("status", "pending")
            team = node_data.get("team", "N/A")
            complexity = node_data.get("complexity", 0)
            label = node_data.get("label", node)
            
            # Check if in circular dependency
            in_circular = any(node in cycle for cycle in circular_deps)
            
            # Check if in critical path
            in_critical = node in critical_path_nodes
            
            # Set color
            if in_circular and highlight_circular:
                node_color.append("#FF0000")  # Red for circular
            elif in_critical and show_critical_path:
                node_color.append("#FFA500")  # Orange for critical path
            else:
                node_color.append(color_map.get(status, "#808080"))
            
            # Set size based on complexity
            node_size.append(20 + complexity * 3)
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Status: {status}<br>"
            hover_text += f"Team: {team}<br>"
            hover_text += f"Complexity: {complexity}<br>"
            hover_text += f"{label}"
            node_text.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            hovertext=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            showlegend=False,
            name='Sub-Problems'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="Sub-Problem Dependency Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green: Completed
        - 🟠 Orange: In Progress / Critical Path
        - ⚪ Gray: Pending
        - 🔴 Red: Failed / Circular Dependency
        - Node size indicates complexity
        """)
        
    except Exception as e:
        st.error(f"Error rendering dependency graph: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_dependency_graph_controls() -> Tuple[bool, bool]:
    """
    Render controls for dependency graph visualization.
    
    Returns:
        Tuple of (show_critical_path, highlight_circular)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        show_critical = st.checkbox("Show Critical Path", value=False)
    
    with col2:
        highlight_circular = st.checkbox("Highlight Circular Dependencies", value=True)
    
    return show_critical, highlight_circular


# ============================================================================
# Analytics Dashboard Components
# ============================================================================

def _filter_by_time_range(
    workflow_history: List[Dict], 
    time_range_code: str
) -> List[Dict]:
    """
    Filter workflow history by time range.
    
    Args:
        workflow_history: List of workflow execution records
        time_range_code: Time range code (e.g., '1h', '24h', '7d', '30d', '90d', 'all')
        
    Returns:
        Filtered list of workflow records within the time range
    """
    if time_range_code == 'all' or not workflow_history:
        return workflow_history
    
    now = datetime.now()
    
    # Parse time range code
    if time_range_code.endswith('h'):
        hours = int(time_range_code[:-1])
        cutoff = now - timedelta(hours=hours)
    elif time_range_code.endswith('d'):
        days = int(time_range_code[:-1])
        cutoff = now - timedelta(days=days)
    else:
        # Unknown format, return all
        return workflow_history
    
    # Filter records by timestamp
    filtered = []
    for record in workflow_history:
        # Try multiple timestamp field names
        timestamp = None
        for field in ['timestamp', 'created_at', 'start_time', 'execution_time', 'completed_at']:
            if field in record and record[field]:
                timestamp = record[field]
                break
        
        if timestamp is None:
            # If no timestamp, include the record
            filtered.append(record)
            continue
            
        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                # Try ISO format
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                ts = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, datetime):
                ts = timestamp
            else:
                # Unknown format, include it
                filtered.append(record)
                continue
                
            # Include if within time range
            if ts >= cutoff:
                filtered.append(record)
        except (ValueError, TypeError):
            # If parsing fails, include the record
            filtered.append(record)
    
    return filtered


def render_analytics_dashboard(
    workflow_history: List[Dict],
    time_range: Optional[Tuple[datetime, datetime]] = None
) -> None:
    """
    Renders the analytics dashboard with multiple metric views.
    
    Args:
        workflow_history: Historical workflow execution data
        time_range: Optional time range filter
    """
    from ui_utils import with_error_handling, display_openevolve_metrics
    from ui_config import UI_CONFIG, TIME_RANGE_OPTIONS
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    st.title("📊 Analytics Dashboard")
    
    # Time range filter
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range_option = st.selectbox(
            "Time Range",
            options=list(TIME_RANGE_OPTIONS.keys()),
            index=2  # Default to "Last 7 Days"
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Filter data by time range
    filtered_history = _filter_by_time_range(workflow_history, TIME_RANGE_OPTIONS[time_range_option])
    
    # Create tabs
    tabs = st.tabs([
        "Overview",
        "Workflow Performance",
        "Team Performance",
        "Gauntlet Effectiveness",
        "Solution Quality",
        "Knowledge Base",
        "Custom Reports"
    ])
    
    with tabs[0]:
        render_analytics_overview(filtered_history)
    
    with tabs[1]:
        render_workflow_performance_metrics(filtered_history)
    
    with tabs[2]:
        render_team_performance_metrics(filtered_history)
    
    with tabs[3]:
        render_gauntlet_effectiveness_metrics(filtered_history)
    
    with tabs[4]:
        render_solution_quality_metrics(filtered_history)
    
    with tabs[5]:
        render_knowledge_base_statistics(filtered_history)
    
    with tabs[6]:
        render_custom_report_generator(filtered_history)


def render_analytics_overview(workflow_history: List[Dict]) -> None:
    """Render analytics overview with key metrics."""
    st.header("Overview")
    
    # Calculate summary metrics
    total_workflows = len(workflow_history)
    successful = sum(1 for w in workflow_history if w.get("status") == "completed")
    success_rate = (successful / total_workflows * 100) if total_workflows > 0 else 0
    
    total_duration = sum(w.get("duration", 0) for w in workflow_history)
    avg_duration = (total_duration / total_workflows) if total_workflows > 0 else 0
    
    total_api_calls = sum(w.get("openevolve_api_calls", 0) for w in workflow_history)
    total_cost = sum(w.get("openevolve_cost", 0) for w in workflow_history)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Workflows", total_workflows)
    
    with col2:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        from ui_utils import format_duration
        st.metric("Avg Duration", format_duration(avg_duration))
    
    with col4:
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    # OpenEvolve metrics
    st.subheader("OpenEvolve Metrics")
    openevolve_metrics = {
        "api_calls": total_api_calls,
        "tokens": sum(w.get("openevolve_tokens", 0) for w in workflow_history),
        "cost": total_cost,
        "evolution_iterations": sum(w.get("evolution_iterations", 0) for w in workflow_history)
    }
    display_openevolve_metrics(openevolve_metrics)


def render_workflow_performance_metrics(workflow_history: List[Dict]) -> None:
    """Render workflow performance metrics."""
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    st.header("Workflow Performance")
    
    if not workflow_history:
        st.info("No workflow data available.")
        return
    
    # Prepare data
    df = pd.DataFrame(workflow_history)
    
    # Success rate over time
    st.subheader("Success Rate Trend")
    if "timestamp" in df.columns and "status" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        success_by_date = df.groupby("date").apply(
            lambda x: (x["status"] == "completed").sum() / len(x) * 100
        ).reset_index(name="success_rate")
        
        fig = px.line(
            success_by_date,
            x="date",
            y="success_rate",
            title="Success Rate Over Time",
            labels={"success_rate": "Success Rate (%)", "date": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Duration distribution
    st.subheader("Execution Duration Distribution")
    if "duration" in df.columns:
        fig = px.histogram(
            df,
            x="duration",
            nbins=20,
            title="Workflow Duration Distribution",
            labels={"duration": "Duration (seconds)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # OpenEvolve API usage over time
    st.subheader("OpenEvolve API Usage")
    if "timestamp" in df.columns and "openevolve_api_calls" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        api_by_date = df.groupby("date")["openevolve_api_calls"].sum().reset_index()
        
        fig = px.bar(
            api_by_date,
            x="date",
            y="openevolve_api_calls",
            title="OpenEvolve API Calls Over Time",
            labels={"openevolve_api_calls": "API Calls", "date": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource usage
    st.subheader("Resource Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        if "openevolve_tokens" in df.columns:
            total_tokens = df["openevolve_tokens"].sum()
            st.metric("Total Tokens Used", f"{total_tokens:,}")
    
    with col2:
        if "openevolve_cost" in df.columns:
            total_cost = df["openevolve_cost"].sum()
            st.metric("Total OpenEvolve Cost", f"${total_cost:.2f}")


def render_team_performance_metrics(workflow_history: List[Dict]) -> None:
    """Render team performance metrics."""
    import plotly.express as px
    import pandas as pd
    
    st.header("Team Performance")
    
    if not workflow_history:
        st.info("No team data available.")
        return
    
    # Extract team metrics from workflow history
    team_data = []
    for workflow in workflow_history:
        for team_name, metrics in workflow.get("team_metrics", {}).items():
            team_data.append({
                "team": team_name,
                "success_rate": metrics.get("success_rate", 0),
                "quality_score": metrics.get("avg_quality_score", 0),
                "efficiency": metrics.get("resource_efficiency", 0),
                "tasks": metrics.get("total_tasks", 0)
            })
    
    if not team_data:
        st.info("No team metrics available.")
        return
    
    df = pd.DataFrame(team_data)
    
    # Team success rate comparison
    st.subheader("Team Success Rate Comparison")
    team_success = df.groupby("team")["success_rate"].mean().reset_index()
    fig = px.bar(
        team_success,
        x="team",
        y="success_rate",
        title="Average Success Rate by Team",
        labels={"success_rate": "Success Rate (%)", "team": "Team"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Team quality scores
    st.subheader("OpenEvolve Solution Quality by Team")
    team_quality = df.groupby("team")["quality_score"].mean().reset_index()
    fig = px.bar(
        team_quality,
        x="team",
        y="quality_score",
        title="Average Quality Score by Team",
        labels={"quality_score": "Quality Score", "team": "Team"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Team efficiency
    st.subheader("Team Resource Efficiency")
    team_efficiency = df.groupby("team")["efficiency"].mean().reset_index()
    fig = px.bar(
        team_efficiency,
        x="team",
        y="efficiency",
        title="Average Resource Efficiency by Team",
        labels={"efficiency": "Efficiency", "team": "Team"}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gauntlet_effectiveness_metrics(workflow_history: List[Dict]) -> None:
    """Render gauntlet effectiveness metrics."""
    import plotly.express as px
    import pandas as pd
    
    st.header("Gauntlet Effectiveness")
    
    if not workflow_history:
        st.info("No gauntlet data available.")
        return
    
    # Extract gauntlet metrics
    gauntlet_data = []
    for workflow in workflow_history:
        for gauntlet_name, metrics in workflow.get("gauntlet_metrics", {}).items():
            gauntlet_data.append({
                "gauntlet": gauntlet_name,
                "detection_rate": metrics.get("detection_rate", 0),
                "accuracy": metrics.get("verification_accuracy", 0),
                "false_positive_rate": metrics.get("false_positive_rate", 0),
                "execution_time": metrics.get("avg_execution_time", 0)
            })
    
    if not gauntlet_data:
        st.info("No gauntlet metrics available.")
        return
    
    df = pd.DataFrame(gauntlet_data)
    
    # Detection rate heatmap
    st.subheader("Flaw Detection Rate by Gauntlet")
    gauntlet_detection = df.groupby("gauntlet")["detection_rate"].mean().reset_index()
    fig = px.bar(
        gauntlet_detection,
        x="gauntlet",
        y="detection_rate",
        title="Average Detection Rate by Gauntlet",
        labels={"detection_rate": "Detection Rate (%)", "gauntlet": "Gauntlet"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Verification accuracy
    st.subheader("Verification Accuracy")
    gauntlet_accuracy = df.groupby("gauntlet")["accuracy"].mean().reset_index()
    fig = px.bar(
        gauntlet_accuracy,
        x="gauntlet",
        y="accuracy",
        title="Average Verification Accuracy by Gauntlet",
        labels={"accuracy": "Accuracy (%)", "gauntlet": "Gauntlet"}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_solution_quality_metrics(workflow_history: List[Dict]) -> None:
    """Render solution quality metrics."""
    import plotly.express as px
    import pandas as pd
    
    st.header("Solution Quality Trends")
    
    if not workflow_history:
        st.info("No quality data available.")
        return
    
    # Extract quality data
    quality_data = []
    for workflow in workflow_history:
        timestamp = workflow.get("timestamp")
        for quality_score in workflow.get("quality_scores", []):
            quality_data.append({
                "timestamp": timestamp,
                "quality_score": quality_score,
                "workflow_id": workflow.get("id", "unknown")
            })
    
    if not quality_data:
        st.info("No quality metrics available.")
        return
    
    df = pd.DataFrame(quality_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Quality trend over time
    st.subheader("Quality Score Trend")
    fig = px.line(
        df,
        x="timestamp",
        y="quality_score",
        title="Solution Quality Over Time",
        labels={"quality_score": "Quality Score", "timestamp": "Time"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality distribution
    st.subheader("Quality Score Distribution")
    fig = px.histogram(
        df,
        x="quality_score",
        nbins=20,
        title="Distribution of Quality Scores",
        labels={"quality_score": "Quality Score"}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_knowledge_base_statistics(workflow_history: List[Dict]) -> None:
    """Render knowledge base statistics."""
    st.header("Knowledge Base Statistics")
    
    try:
        from knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        
        # Get all artifacts
        artifacts = km.get_all_artifacts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Artifacts", len(artifacts))
        
        with col2:
            total_usage = sum(a.usage_count for a in artifacts)
            st.metric("Total Usage Count", total_usage)
        
        with col3:
            avg_effectiveness = sum(a.effectiveness_score for a in artifacts) / len(artifacts) if artifacts else 0
            st.metric("Avg Effectiveness", f"{avg_effectiveness:.2f}")
        
        # Artifact types distribution
        if artifacts:
            import pandas as pd
            import plotly.express as px
            
            artifact_types = {}
            for a in artifacts:
                artifact_types[a.type] = artifact_types.get(a.type, 0) + 1
            
            df = pd.DataFrame(list(artifact_types.items()), columns=["Type", "Count"])
            fig = px.pie(df, values="Count", names="Type", title="Artifact Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Unable to load knowledge base statistics: {e}")


def render_custom_report_generator(workflow_history: List[Dict]) -> None:
    """Render custom report generation interface."""
    st.header("Custom Report Generator")
    
    st.write("Select metrics to include in your custom report:")
    
    # Metric selection
    include_workflow = st.checkbox("Workflow Performance Metrics", value=True)
    include_team = st.checkbox("Team Performance Metrics", value=True)
    include_gauntlet = st.checkbox("Gauntlet Effectiveness Metrics", value=False)
    include_quality = st.checkbox("Solution Quality Metrics", value=True)
    include_knowledge = st.checkbox("Knowledge Base Statistics", value=False)
    
    # Format selection
    report_format = st.selectbox("Report Format", ["CSV", "JSON", "Excel"])
    
    if st.button("Generate Report"):
        import pandas as pd
        import io
        
        # Collect selected data
        report_data = {}
        
        if include_workflow:
            report_data["workflows"] = workflow_history
        
        if include_team:
            team_data = []
            for w in workflow_history:
                team_data.extend(w.get("team_metrics", {}).items())
            report_data["teams"] = team_data
        
        # Generate report based on format
        if report_format == "CSV":
            df = pd.DataFrame(workflow_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="workflow_report.csv",
                mime="text/csv"
            )
        
        elif report_format == "JSON":
            import json
            json_str = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=json_str,
                file_name="workflow_report.json",
                mime="application/json"
            )
        
        st.success("Report generated successfully!")


# ============================================================================
# Knowledge Base Interface Components
# ============================================================================

def render_knowledge_base_interface(knowledge_manager) -> None:
    """
    Renders the knowledge base management interface.
    
    Args:
        knowledge_manager: Instance of KnowledgeManager
    """
    from ui_utils import with_error_handling, get_or_init_state
    
    st.title("📚 Knowledge Base")
    
    # Create tabs
    tabs = st.tabs([
        "Browse Artifacts",
        "Knowledge Graph",
        "Learning Configuration"
    ])
    
    with tabs[0]:
        render_artifact_browser(knowledge_manager)
    
    with tabs[1]:
        render_knowledge_graph(knowledge_manager)
    
    with tabs[2]:
        render_learning_configuration(knowledge_manager)


def render_artifact_browser(knowledge_manager) -> None:
    """Render artifact list and search interface."""
    st.header("Knowledge Artifacts")
    
    # Search and filter controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search artifacts", placeholder="Search by content, tags, or metadata...")
    
    with col2:
        artifact_type_filter = st.selectbox(
            "Type",
            ["All", "pattern", "solution", "error", "best_practice"]
        )
    
    with col3:
        if st.button("➕ New Artifact"):
            st.session_state.show_create_artifact = True
    
    # Get all artifacts
    try:
        all_artifacts = knowledge_manager.get_all_artifacts()
        
        # Apply filters
        filtered_artifacts = all_artifacts
        
        if search_query:
            filtered_artifacts = [
                a for a in filtered_artifacts
                if search_query.lower() in a.content.lower()
                or search_query.lower() in str(a.domain).lower()
                or search_query.lower() in str(a.problem_type).lower()
            ]
        
        if artifact_type_filter != "All":
            filtered_artifacts = [
                a for a in filtered_artifacts
                if a.artifact_type == artifact_type_filter
            ]
        
        st.write(f"Found {len(filtered_artifacts)} artifacts")
        
        # Display artifacts in a grid
        if filtered_artifacts:
            # Pagination
            page_size = 20
            total_pages = (len(filtered_artifacts) + page_size - 1) // page_size
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
            
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(filtered_artifacts))
            page_artifacts = filtered_artifacts[start_idx:end_idx]
            
            for artifact in page_artifacts:
                with st.expander(f"📄 {artifact.artifact_type.upper()}: {artifact.id}"):
                    render_artifact_detail(artifact, knowledge_manager)
        else:
            st.info("No artifacts found matching your criteria.")
    
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
    
    # Create artifact dialog
    if st.session_state.get("show_create_artifact", False):
        render_create_artifact_dialog(knowledge_manager)


def render_artifact_detail(artifact, knowledge_manager) -> None:
    """Render detailed view of an artifact."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Type:** {artifact.artifact_type}")
        st.markdown(f"**Domain:** {artifact.domain}")
        st.markdown(f"**Problem Type:** {artifact.problem_type}")
        st.markdown(f"**Source Workflow:** {artifact.source_workflow_id}")
        
        st.markdown("**Content:**")
        st.code(artifact.content, language="text")
        
        if artifact.related_artifacts:
            st.markdown(f"**Related Artifacts:** {', '.join(artifact.related_artifacts)}")
    
    with col2:
        st.metric("Usage Count", artifact.usage_count)
        st.metric("Effectiveness", f"{artifact.effectiveness_score:.2f}")
        
        st.markdown("---")
        
        if st.button("✏️ Edit", key=f"edit_{artifact.id}"):
            st.session_state.edit_artifact_id = artifact.id
        
        if st.button("🗑️ Delete", key=f"delete_{artifact.id}"):
            if st.session_state.get(f"confirm_delete_{artifact.id}", False):
                knowledge_manager.delete_artifact(artifact.id)
                st.success("Artifact deleted!")
                st.rerun()
            else:
                st.session_state[f"confirm_delete_{artifact.id}"] = True
                st.warning("Click again to confirm deletion")
        
        if st.button("📤 Export", key=f"export_{artifact.id}"):
            import json
            artifact_json = json.dumps({
                "id": artifact.id,
                "type": artifact.artifact_type,
                "content": artifact.content,
                "domain": artifact.domain,
                "problem_type": artifact.problem_type,
                "usage_count": artifact.usage_count,
                "effectiveness_score": artifact.effectiveness_score
            }, indent=2)
            st.download_button(
                "Download JSON",
                artifact_json,
                file_name=f"artifact_{artifact.id}.json",
                mime="application/json"
            )


def render_create_artifact_dialog(knowledge_manager) -> None:
    """Render dialog for creating new artifact."""
    st.subheader("Create New Artifact")
    
    with st.form("create_artifact_form"):
        artifact_type = st.selectbox(
            "Type",
            ["pattern", "solution", "error", "best_practice"]
        )
        
        content = st.text_area("Content", height=200)
        domain = st.text_input("Domain")
        problem_type = st.text_input("Problem Type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("Create")
        
        with col2:
            cancelled = st.form_submit_button("Cancel")
        
        if submitted:
            if not content:
                st.error("Content is required")
            else:
                from workflow_structures import KnowledgeArtifact
                import hashlib
                from datetime import datetime
                
                artifact_id = hashlib.md5(content.encode()).hexdigest()[:16]
                
                new_artifact = KnowledgeArtifact(
                    id=artifact_id,
                    artifact_type=artifact_type,
                    content=content,
                    source_workflow_id="manual",
                    extraction_timestamp=datetime.now().isoformat(),
                    domain=domain,
                    problem_type=problem_type,
                    usage_count=0,
                    effectiveness_score=0.0,
                    related_artifacts=[]
                )
                
                knowledge_manager.store_knowledge_artifact(new_artifact)
                st.success("Artifact created successfully!")
                st.session_state.show_create_artifact = False
                st.rerun()
        
        if cancelled:
            st.session_state.show_create_artifact = False
            st.rerun()


def render_knowledge_graph(knowledge_manager) -> None:
    """Render knowledge graph visualization."""
    import plotly.graph_objects as go
    import networkx as nx
    
    st.header("Knowledge Graph")
    
    try:
        artifacts = knowledge_manager.get_all_artifacts()
        
        if not artifacts:
            st.info("No artifacts to visualize.")
            return
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for artifact in artifacts:
            G.add_node(
                artifact.id,
                type=artifact.artifact_type,
                domain=artifact.domain,
                usage=artifact.usage_count
            )
        
        # Add edges based on related artifacts
        for artifact in artifacts:
            for related_id in artifact.related_artifacts:
                if related_id in [a.id for a in artifacts]:
                    G.add_edge(artifact.id, related_id)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        color_map = {
            "pattern": "#1f77b4",
            "solution": "#2ca02c",
            "error": "#d62728",
            "best_practice": "#ff7f0e"
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_type = node_data.get("type", "unknown")
            usage = node_data.get("usage", 0)
            
            node_text.append(f"{node}<br>Type: {node_type}<br>Usage: {usage}")
            node_color.append(color_map.get(node_type, "#808080"))
            node_size.append(10 + usage * 2)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="Knowledge Artifact Relationships",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🔵 Blue: Pattern
        - 🟢 Green: Solution
        - 🔴 Red: Error
        - 🟠 Orange: Best Practice
        - Node size indicates usage count
        """)
    
    except Exception as e:
        st.error(f"Error rendering knowledge graph: {e}")


def render_learning_configuration(knowledge_manager) -> None:
    """Render learning configuration interface."""
    st.header("Learning Configuration")
    
    st.write("Configure what knowledge to extract and how to use it:")
    
    # Extraction options
    st.subheader("Knowledge Extraction")
    
    extract_patterns = st.checkbox("Extract Patterns", value=True)
    extract_solutions = st.checkbox("Extract Solutions", value=True)
    extract_errors = st.checkbox("Extract Errors", value=True)
    extract_best_practices = st.checkbox("Extract Best Practices", value=False)
    
    # Usage policies
    st.subheader("Usage Policies")
    
    auto_apply = st.checkbox("Automatically apply learned knowledge", value=True)
    min_effectiveness = st.slider(
        "Minimum effectiveness score to apply",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    max_artifacts_per_query = st.number_input(
        "Maximum artifacts to retrieve per query",
        min_value=1,
        max_value=50,
        value=10
    )
    
    if st.button("Save Configuration"):
        config = {
            "extraction": {
                "patterns": extract_patterns,
                "solutions": extract_solutions,
                "errors": extract_errors,
                "best_practices": extract_best_practices
            },
            "usage": {
                "auto_apply": auto_apply,
                "min_effectiveness": min_effectiveness,
                "max_artifacts_per_query": max_artifacts_per_query
            }
        }
        
        # Save configuration (implement storage)
        st.success("Configuration saved successfully!")


# ============================================================================
# Auto-Approval Configuration UI Components
# ============================================================================

def render_auto_approval_config(auto_approval_manager=None) -> None:
    """
    Renders the auto-approval configuration interface.
    
    Args:
        auto_approval_manager: Instance of AutoApprovalManager (optional)
    """
    from ui_utils import get_or_init_state
    from ui_models import AutoApprovalRule, RuleCondition, RuleOperator, LogicalOperator, RuleAction
    
    st.title("⚙️ Auto-Approval Configuration")
    
    # Initialize rules in session state
    if "auto_approval_rules" not in st.session_state:
        st.session_state.auto_approval_rules = []
    
    if "auto_approval_enabled" not in st.session_state:
        st.session_state.auto_approval_enabled = False
    
    # Status toggle
    st.header("Status")
    enabled = st.toggle(
        "Enable Auto-Approval",
        value=st.session_state.auto_approval_enabled,
        help="When enabled, plans meeting configured rules will be automatically approved"
    )
    
    if enabled != st.session_state.auto_approval_enabled:
        st.session_state.auto_approval_enabled = enabled
        st.success(f"Auto-approval {'enabled' if enabled else 'disabled'}!")
    
    # Rules section
    st.header("Rules")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Rule"):
            st.session_state.show_rule_builder = True
    
    # Display existing rules
    if st.session_state.auto_approval_rules:
        for i, rule in enumerate(st.session_state.auto_approval_rules):
            render_auto_approval_rule(rule, i)
    else:
        st.info("No rules configured. Add a rule to get started.")
    
    # Rule builder
    if st.session_state.get("show_rule_builder", False):
        render_rule_builder()
    
    # Rule testing
    st.header("Test Rules")
    render_rule_testing()
    
    # Audit log
    st.header("Audit Log")
    render_audit_log()


def render_auto_approval_rule(rule: Dict, index: int) -> None:
    """Render a single auto-approval rule."""
    with st.expander(f"Rule {index + 1}: {rule.get('name', 'Unnamed Rule')}", expanded=False):
        st.markdown(f"**Priority:** {rule.get('priority', 0)}")
        st.markdown(f"**Action:** {rule.get('action', 'approve')}")
        st.markdown(f"**Enabled:** {'Yes' if rule.get('enabled', True) else 'No'}")
        
        st.markdown("**Conditions:**")
        for condition in rule.get('conditions', []):
            st.write(f"- {condition.get('field')} {condition.get('operator')} {condition.get('value')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✏️ Edit", key=f"edit_rule_{index}"):
                st.session_state.edit_rule_index = index
                st.session_state.show_rule_builder = True
        
        with col2:
            if st.button("🗑️ Delete", key=f"delete_rule_{index}"):
                if st.session_state.get(f"confirm_delete_rule_{index}", False):
                    st.session_state.auto_approval_rules.pop(index)
                    st.success("Rule deleted!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_rule_{index}"] = True
                    st.warning("Click again to confirm")
        
        with col3:
            if st.button("🧪 Test", key=f"test_rule_{index}"):
                st.session_state.test_rule_index = index


def render_rule_builder() -> None:
    """Render rule builder interface."""
    st.subheader("Rule Builder")
    
    edit_index = st.session_state.get("edit_rule_index", None)
    if edit_index is not None and edit_index < len(st.session_state.auto_approval_rules):
        existing_rule = st.session_state.auto_approval_rules[edit_index]
        st.info(f"Editing Rule {edit_index + 1}")
    else:
        existing_rule = None
    
    with st.form("rule_builder_form"):
        rule_name = st.text_input(
            "Rule Name",
            value=existing_rule.get("name", "") if existing_rule else ""
        )
        
        rule_priority = st.number_input(
            "Priority (higher = evaluated first)",
            min_value=0,
            max_value=100,
            value=existing_rule.get("priority", 10) if existing_rule else 10
        )
        
        rule_action = st.selectbox(
            "Action",
            ["approve", "reject", "escalate"],
            index=["approve", "reject", "escalate"].index(existing_rule.get("action", "approve")) if existing_rule else 0
        )
        
        rule_enabled = st.checkbox(
            "Enabled",
            value=existing_rule.get("enabled", True) if existing_rule else True
        )
        
        st.markdown("---")
        st.markdown("**Conditions**")
        
        # Condition builder
        num_conditions = st.number_input(
            "Number of conditions",
            min_value=1,
            max_value=10,
            value=len(existing_rule.get("conditions", [])) if existing_rule else 1
        )
        
        conditions = []
        for i in range(int(num_conditions)):
            st.markdown(f"**Condition {i + 1}**")
            
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                field = st.selectbox(
                    "Field",
                    ["complexity", "confidence", "domain", "num_sub_problems", "team_type"],
                    key=f"field_{i}"
                )
            
            with col2:
                operator = st.selectbox(
                    "Operator",
                    ["<", ">", "==", "!=", "contains"],
                    key=f"operator_{i}"
                )
            
            with col3:
                value = st.text_input("Value", key=f"value_{i}")
            
            with col4:
                if i < num_conditions - 1:
                    logical_op = st.selectbox(
                        "Logic",
                        ["AND", "OR"],
                        key=f"logic_{i}"
                    )
                else:
                    logical_op = "AND"
            
            conditions.append({
                "field": field,
                "operator": operator,
                "value": value,
                "logical_op": logical_op
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("Save Rule")
        
        with col2:
            cancelled = st.form_submit_button("Cancel")
        
        if submitted:
            if not rule_name:
                st.error("Rule name is required")
            else:
                new_rule = {
                    "name": rule_name,
                    "priority": rule_priority,
                    "action": rule_action,
                    "enabled": rule_enabled,
                    "conditions": conditions,
                    "created_at": datetime.now().isoformat()
                }
                
                if edit_index is not None:
                    st.session_state.auto_approval_rules[edit_index] = new_rule
                    st.success("Rule updated!")
                else:
                    st.session_state.auto_approval_rules.append(new_rule)
                    st.success("Rule created!")
                
                st.session_state.show_rule_builder = False
                st.session_state.edit_rule_index = None
                st.rerun()
        
        if cancelled:
            st.session_state.show_rule_builder = False
            st.session_state.edit_rule_index = None
            st.rerun()


def render_rule_testing() -> None:
    """Render rule testing interface."""
    st.write("Test your rules against sample plans:")
    
    # Sample plan input
    with st.expander("Configure Test Plan"):
        test_complexity = st.slider("Complexity", 1, 10, 5)
        test_confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        test_domain = st.text_input("Domain", "Software Development")
        test_num_sub_problems = st.number_input("Number of Sub-Problems", 1, 20, 5)
    
    if st.button("Run Test"):
        test_plan = {
            "complexity": test_complexity,
            "confidence": test_confidence,
            "domain": test_domain,
            "num_sub_problems": test_num_sub_problems
        }
        
        st.subheader("Test Results")
        
        if not st.session_state.auto_approval_rules:
            st.warning("No rules to test")
        else:
            for i, rule in enumerate(st.session_state.auto_approval_rules):
                if not rule.get("enabled", True):
                    continue
                
                # Evaluate rule
                matches = evaluate_rule(rule, test_plan)
                
                if matches:
                    st.success(f"[OK] Rule {i + 1} ({rule['name']}): MATCH - Action: {rule['action']}")
                else:
                    st.info(f"[FAIL] Rule {i + 1} ({rule['name']}): NO MATCH")


def evaluate_rule(rule: Dict, plan: Dict) -> bool:
    """Evaluate if a rule matches a plan."""
    conditions = rule.get("conditions", [])
    
    if not conditions:
        return False
    
    results = []
    
    for condition in conditions:
        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]
        
        plan_value = plan.get(field)
        
        # Evaluate condition
        try:
            if operator == "<":
                result = float(plan_value) < float(value)
            elif operator == ">":
                result = float(plan_value) > float(value)
            elif operator == "==":
                result = str(plan_value) == str(value)
            elif operator == "!=":
                result = str(plan_value) != str(value)
            elif operator == "contains":
                result = str(value).lower() in str(plan_value).lower()
            else:
                result = False
        except:
            result = False
        
        results.append(result)
    
    # Combine results based on logical operators
    if not results:
        return False
    
    final_result = results[0]
    for i, condition in enumerate(conditions[:-1]):
        logical_op = condition.get("logical_op", "AND")
        if logical_op == "AND":
            final_result = final_result and results[i + 1]
        else:  # OR
            final_result = final_result or results[i + 1]
    
    return final_result


def render_audit_log() -> None:
    """Render audit log viewer."""
    # Initialize audit log in session state
    if "auto_approval_audit_log" not in st.session_state:
        st.session_state.auto_approval_audit_log = []
    
    audit_log = st.session_state.auto_approval_audit_log
    
    if not audit_log:
        st.info("No audit log entries yet.")
        return
    
    # Display audit log
    import pandas as pd
    
    df = pd.DataFrame(audit_log)
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        filter_action = st.selectbox("Filter by Action", ["All", "approve", "reject", "escalate"])
    
    with col2:
        filter_rule = st.selectbox("Filter by Rule", ["All"] + list(set(df["rule_name"].tolist())))
    
    # Apply filters
    filtered_df = df
    if filter_action != "All":
        filtered_df = filtered_df[filtered_df["action"] == filter_action]
    if filter_rule != "All":
        filtered_df = filtered_df[filtered_df["rule_name"] == filter_rule]
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Real-time Monitoring Enhancement Components
# ============================================================================

def render_enhanced_monitoring(
    workflow_state: Dict,
    resource_monitor=None
) -> None:
    """
    Renders enhanced real-time monitoring interface.
    
    Args:
        workflow_state: Current workflow execution state
        resource_monitor: Instance of ResourceMonitor (optional)
    """
    from ui_utils import display_openevolve_metrics, format_duration
    from ui_models import WorkflowState, AlertSeverity
    import plotly.graph_objects as go
    
    st.title("📊 Workflow Execution Monitor")
    
    # Status and controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        status = workflow_state.get("status", "idle")
        status_colors = {
            "idle": "gray",
            "running": "green",
            "paused": "orange",
            "completed": "blue",
            "failed": "red",
            "terminated": "red"
        }
        st.markdown(f"**Status:** :{status_colors.get(status, 'gray')}[**{status.upper()}**]")
    
    with col2:
        render_workflow_controls(workflow_state)
    
    st.divider()
    
    # Create tabs
    tabs = st.tabs([
        "Resource Usage",
        "Performance",
        "Solution Quality",
        "Alerts",
        "Logs"
    ])
    
    with tabs[0]:
        render_resource_usage_display(workflow_state)
    
    with tabs[1]:
        render_performance_metrics_display(workflow_state)
    
    with tabs[2]:
        render_solution_quality_display(workflow_state)
    
    with tabs[3]:
        render_alert_system(workflow_state)
    
    with tabs[4]:
        render_detailed_log_viewer(workflow_state)


def render_workflow_controls(workflow_state: Dict) -> None:
    """Render workflow control buttons."""
    status = workflow_state.get("status", "idle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status == "running":
            if st.button("⏸ Pause"):
                st.session_state.workflow_action = "pause"
                st.success("Workflow paused")
        elif status == "paused":
            if st.button("▶️ Resume"):
                st.session_state.workflow_action = "resume"
                st.success("Workflow resumed")
    
    with col2:
        if status in ["running", "paused"]:
            if st.button("⏹ Stop"):
                if st.session_state.get("confirm_stop", False):
                    st.session_state.workflow_action = "stop"
                    st.success("Workflow stopped")
                    st.session_state.confirm_stop = False
                else:
                    st.session_state.confirm_stop = True
                    st.warning("Click again to confirm")
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()


def render_resource_usage_display(workflow_state: Dict) -> None:
    """Render resource usage metrics."""
    st.header("Resource Usage")
    
    # Get resource metrics
    resource_usage = workflow_state.get("resource_usage", {})
    
    # CPU and Memory
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_percent = resource_usage.get("cpu_percent", 0)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(min(cpu_percent / 100, 1.0))
    
    with col2:
        memory_percent = resource_usage.get("memory_percent", 0)
        st.metric("Memory Usage", f"{memory_percent:.1f}%")
        st.progress(min(memory_percent / 100, 1.0))
    
    # OpenEvolve metrics
    st.subheader("OpenEvolve API Usage")
    
    openevolve_metrics = {
        "api_calls": resource_usage.get("openevolve_api_calls", 0),
        "tokens": resource_usage.get("openevolve_tokens", 0),
        "cost": resource_usage.get("openevolve_cost", 0.0),
        "evolution_iterations": resource_usage.get("evolution_iterations", 0)
    }
    
    display_openevolve_metrics(openevolve_metrics)
    
    # API calls progress
    max_api_calls = workflow_state.get("max_api_calls", 500)
    current_api_calls = openevolve_metrics["api_calls"]
    
    st.subheader("API Call Limit")
    st.progress(
        min(current_api_calls / max_api_calls, 1.0),
        text=f"{current_api_calls}/{max_api_calls} calls ({current_api_calls/max_api_calls*100:.1f}%)"
    )
    
    # Resource usage chart
    if "resource_history" in workflow_state:
        st.subheader("Resource Usage Over Time")
        render_resource_usage_chart(workflow_state["resource_history"])


def render_resource_usage_chart(resource_history: List[Dict]) -> None:
    """Render resource usage chart."""
    import plotly.graph_objects as go
    import pandas as pd
    
    if not resource_history:
        st.info("No resource history available yet.")
        return
    
    df = pd.DataFrame(resource_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.get("cpu_percent", []),
        name="CPU %",
        mode="lines"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.get("memory_percent", []),
        name="Memory %",
        mode="lines"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.get("openevolve_api_calls", []),
        name="API Calls",
        mode="lines",
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Resource Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Percentage",
        yaxis2=dict(
            title="API Calls",
            overlaying="y",
            side="right"
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics_display(workflow_state: Dict) -> None:
    """Render performance metrics."""
    import plotly.graph_objects as go
    
    st.header("Performance Metrics")
    
    # Execution progress
    completed_tasks = workflow_state.get("completed_tasks", 0)
    total_tasks = workflow_state.get("total_tasks", 1)
    progress = completed_tasks / total_tasks if total_tasks > 0 else 0
    
    st.subheader("Execution Progress")
    st.progress(progress, text=f"{completed_tasks}/{total_tasks} tasks ({progress*100:.1f}%)")
    
    # Duration
    from ui_utils import format_duration
    duration = workflow_state.get("duration", 0)
    st.metric("Elapsed Time", format_duration(duration))
    
    # Throughput
    throughput = completed_tasks / (duration / 60) if duration > 0 else 0
    st.metric("Throughput", f"{throughput:.2f} tasks/min")
    
    # OpenEvolve evolution progress
    st.subheader("OpenEvolve Evolution Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        generations = workflow_state.get("evolution_generations", 0)
        st.metric("Generations", generations)
    
    with col2:
        fitness_improvement = workflow_state.get("fitness_improvement", 0)
        st.metric("Fitness Improvement", f"{fitness_improvement:.2%}")
    
    with col3:
        convergence = workflow_state.get("convergence_rate", 0)
        st.metric("Convergence Rate", f"{convergence:.2%}")
    
    # Performance timeline
    if "performance_history" in workflow_state:
        st.subheader("Performance Timeline")
        render_performance_timeline(workflow_state["performance_history"])


def render_performance_timeline(performance_history: List[Dict]) -> None:
    """Render performance timeline chart."""
    import plotly.graph_objects as go
    import pandas as pd
    
    if not performance_history:
        st.info("No performance history available yet.")
        return
    
    df = pd.DataFrame(performance_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.get("throughput", []),
        name="Throughput",
        mode="lines+markers"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.get("latency", []),
        name="Latency (ms)",
        mode="lines+markers",
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Time",
        yaxis_title="Throughput (tasks/min)",
        yaxis2=dict(
            title="Latency (ms)",
            overlaying="y",
            side="right"
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_solution_quality_display(workflow_state: Dict) -> None:
    """Render solution quality metrics."""
    import plotly.express as px
    import pandas as pd
    
    st.header("Solution Quality")
    
    # Overall metrics
    quality_scores = workflow_state.get("quality_scores", [])
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Quality", f"{avg_quality:.2f}")
        
        with col2:
            st.metric("Min Quality", f"{min_quality:.2f}")
        
        with col3:
            st.metric("Max Quality", f"{max_quality:.2f}")
        
        # Quality distribution
        st.subheader("Quality Score Distribution")
        df = pd.DataFrame({"quality_score": quality_scores})
        fig = px.histogram(
            df,
            x="quality_score",
            nbins=20,
            title="Distribution of Quality Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # OpenEvolve fitness evolution
        if "fitness_history" in workflow_state:
            st.subheader("OpenEvolve Fitness Evolution")
            fitness_history = workflow_state["fitness_history"]
            
            df_fitness = pd.DataFrame({
                "generation": range(len(fitness_history)),
                "fitness": fitness_history
            })
            
            fig = px.line(
                df_fitness,
                x="generation",
                y="fitness",
                title="Fitness Score Over Generations"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quality scores available yet.")


def render_alert_system(workflow_state: Dict) -> None:
    """Render alert notification system."""
    st.header("Alerts")
    
    alerts = workflow_state.get("alerts", [])
    
    if not alerts:
        st.success("No alerts at this time.")
        return
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "error", "warning", "info"]
        )
    
    with col2:
        if st.button("Clear All Alerts"):
            workflow_state["alerts"] = []
            st.rerun()
    
    # Display alerts
    filtered_alerts = alerts
    if severity_filter != "All":
        filtered_alerts = [a for a in alerts if a.get("severity") == severity_filter]
    
    for alert in filtered_alerts:
        severity = alert.get("severity", "info")
        message = alert.get("message", "")
        timestamp = alert.get("timestamp", "")
        
        if severity == "error":
            st.error(f"🔴 **{timestamp}** - {message}")
        elif severity == "warning":
            st.warning(f"🟠 **{timestamp}** - {message}")
        else:
            st.info(f"🔵 **{timestamp}** - {message}")
        
        if alert.get("details"):
            with st.expander("Details"):
                st.json(alert["details"])


def render_detailed_log_viewer(workflow_state: Dict) -> None:
    """Render detailed log viewer."""
    st.header("Execution Logs")
    
    logs = workflow_state.get("logs", [])
    
    if not logs:
        st.info("No logs available yet.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level_filter = st.selectbox(
            "Log Level",
            ["All", "DEBUG", "INFO", "WARNING", "ERROR"]
        )
    
    with col2:
        search_query = st.text_input("Search logs")
    
    with col3:
        max_lines = st.number_input("Max lines", min_value=10, max_value=1000, value=100)
    
    # Apply filters
    filtered_logs = logs[-max_lines:]
    
    if log_level_filter != "All":
        filtered_logs = [log for log in filtered_logs if log.get("level") == log_level_filter]
    
    if search_query:
        filtered_logs = [
            log for log in filtered_logs
            if search_query.lower() in log.get("message", "").lower()
        ]
    
    # Display logs
    st.text_area(
        "Logs",
        value="\n".join([
            f"[{log.get('timestamp')}] {log.get('level')}: {log.get('message')}"
            for log in filtered_logs
        ]),
        height=400
    )
    
    # Download logs
    if st.button("📥 Download Logs"):
        log_text = "\n".join([
            f"[{log.get('timestamp')}] {log.get('level')}: {log.get('message')}"
            for log in logs
        ])
        st.download_button(
            "Download",
            log_text,
            file_name="workflow_logs.txt",
            mime="text/plain"
        )


# ============================================================================
# Workflow Templates UI Components
# ============================================================================

def render_workflow_templates(
    template_manager,
    current_config: Dict[str, Any]
) -> None:
    """
    Renders workflow template management interface.
    
    Args:
        template_manager: Instance of TemplateManager
        current_config: Current workflow configuration
    """
    from template_manager import TemplateManager
    
    st.title("📋 Workflow Templates")
    
    # Initialize template manager in session state if not exists
    if "template_manager" not in st.session_state:
        st.session_state.template_manager = TemplateManager()
    
    tm = st.session_state.template_manager
    
    # Current configuration section
    st.header("Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save as Template"):
            st.session_state.show_save_template = True
    
    with col2:
        if st.button("📥 Load Template"):
            st.session_state.show_load_template = True
    
    # Show current config summary
    with st.expander("View Current Configuration"):
        st.json(current_config)
    
    st.divider()
    
    # Template library
    st.header("Template Library")
    render_template_library(tm)
    
    # Save template dialog
    if st.session_state.get("show_save_template", False):
        render_save_template_dialog(tm, current_config)
    
    # Load template dialog
    if st.session_state.get("show_load_template", False):
        render_load_template_dialog(tm)
    
    # Import/Export section
    st.divider()
    st.header("Import / Export")
    render_template_import_export(tm)


def render_template_library(template_manager) -> None:
    """Render template library."""
    templates = template_manager.get_all_templates()
    
    if not templates:
        st.info("No templates saved yet. Save your current configuration to create a template.")
        return
    
    # Search
    search_query = st.text_input("🔍 Search templates", placeholder="Search by name, description, or tags...")
    
    if search_query:
        templates = template_manager.search_templates(search_query)
    
    # Sort options
    sort_by = st.selectbox("Sort by", ["Name", "Created Date", "Usage Count"])
    
    if sort_by == "Name":
        templates = sorted(templates, key=lambda t: t["name"])
    elif sort_by == "Created Date":
        templates = sorted(templates, key=lambda t: t["created_at"], reverse=True)
    else:  # Usage Count
        templates = sorted(templates, key=lambda t: t["usage_count"], reverse=True)
    
    st.write(f"Found {len(templates)} templates")
    
    # Display templates
    for template in templates:
        render_template_card(template, template_manager)


def render_template_card(template: Dict, template_manager) -> None:
    """Render a single template card."""
    with st.expander(f"📄 {template['name']}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Description:** {template['description']}")
            st.markdown(f"**Created:** {template['created_at'][:10]}")
            st.markdown(f"**Usage Count:** {template['usage_count']}")
            
            if template.get("tags"):
                st.markdown(f"**Tags:** {', '.join(template['tags'])}")
            
            with st.expander("View Configuration"):
                st.json(template["config"])
        
        with col2:
            if st.button("📥 Load", key=f"load_{template['id']}"):
                st.session_state.loaded_template = template
                template_manager.increment_usage(template["id"])
                st.success(f"Template '{template['name']}' loaded!")
                st.rerun()
            
            if st.button("📤 Export", key=f"export_{template['id']}"):
                template_json = template_manager.export_template(template['id'])
                st.download_button(
                    "Download JSON",
                    template_json,
                    file_name=f"template_{template['name'].replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"download_{template['id']}"
                )
            
            if st.button("🗑️ Delete", key=f"delete_{template['id']}"):
                if st.session_state.get(f"confirm_delete_template_{template['id']}", False):
                    template_manager.delete_template(template["id"])
                    st.success("Template deleted!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_template_{template['id']}"] = True
                    st.warning("Click again to confirm")


def render_save_template_dialog(template_manager, current_config: Dict) -> None:
    """Render save template dialog."""
    st.subheader("Save Current Configuration as Template")
    
    with st.form("save_template_form"):
        template_name = st.text_input("Template Name", placeholder="e.g., Simple Decomposition")
        
        template_description = st.text_area(
            "Description",
            placeholder="Describe when to use this template..."
        )
        
        tags_input = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., simple, fast, low-cost"
        )
        
        # Show what will be saved
        st.markdown("**Configuration to Save:**")
        
        # Extract OpenEvolve settings from current config
        openevolve_config = {
            "model": current_config.get("model", "gpt-4"),
            "evolution_mode": current_config.get("evolution_mode", "standard"),
            "temperature": current_config.get("temperature", 0.7),
            "max_iterations": current_config.get("max_iterations", 10)
        }
        
        config_to_save = {
            "max_depth": current_config.get("max_depth", 3),
            "teams": current_config.get("teams", []),
            "gauntlets": current_config.get("gauntlets", []),
            "auto_approval": current_config.get("auto_approval", False),
            "resource_limits": current_config.get("resource_limits", {}),
            "openevolve": openevolve_config
        }
        
        st.json(config_to_save)
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("Save Template")
        
        with col2:
            cancelled = st.form_submit_button("Cancel")
        
        if submitted:
            if not template_name:
                st.error("Template name is required")
            else:
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                
                template_id = template_manager.create_template(
                    name=template_name,
                    description=template_description,
                    config=config_to_save,
                    tags=tags
                )
                
                st.success(f"Template '{template_name}' saved successfully!")
                st.session_state.show_save_template = False
                st.rerun()
        
        if cancelled:
            st.session_state.show_save_template = False
            st.rerun()


def render_load_template_dialog(template_manager) -> None:
    """Render load template dialog."""
    st.subheader("Load Template")
    
    templates = template_manager.get_all_templates()
    
    if not templates:
        st.warning("No templates available to load.")
        if st.button("Close"):
            st.session_state.show_load_template = False
            st.rerun()
        return
    
    template_names = {t["name"]: t["id"] for t in templates}
    
    selected_name = st.selectbox("Select Template", list(template_names.keys()))
    
    if selected_name:
        template_id = template_names[selected_name]
        template = template_manager.get_template(template_id)
        
        st.markdown("**Template Details:**")
        st.markdown(f"**Description:** {template['description']}")
        st.markdown(f"**Created:** {template['created_at'][:10]}")
        st.markdown(f"**Usage Count:** {template['usage_count']}")
        
        with st.expander("View Configuration"):
            st.json(template["config"])
        
        # Validate template
        is_valid, errors = template_manager.validate_template(template)
        
        if not is_valid:
            st.error("Template validation failed:")
            for error in errors:
                st.write(f"- {error}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Template", disabled=not is_valid):
                st.session_state.loaded_template = template
                template_manager.increment_usage(template_id)
                st.success(f"Template '{template['name']}' loaded!")
                st.session_state.show_load_template = False
                st.rerun()
        
        with col2:
            if st.button("Cancel"):
                st.session_state.show_load_template = False
                st.rerun()


def render_template_import_export(template_manager) -> None:
    """Render template import/export interface."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Import Template")
        
        uploaded_file = st.file_uploader(
            "Upload template JSON file",
            type=["json"],
            key="template_upload"
        )
        
        if uploaded_file is not None:
            try:
                template_json = uploaded_file.read().decode("utf-8")
                template_id = template_manager.import_template(template_json)
                
                if template_id:
                    st.success("Template imported successfully!")
                    st.rerun()
                else:
                    st.error("Failed to import template. Please check the file format.")
            except Exception as e:
                st.error(f"Error importing template: {e}")
    
    with col2:
        st.subheader("Export All Templates")
        
        if st.button("Export All as ZIP"):
            import zipfile
            import io
            
            templates = template_manager.get_all_templates()
            
            if templates:
                # Create ZIP file in memory
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for template in templates:
                        template_json = template_manager.export_template(template["id"])
                        filename = f"{template['name'].replace(' ', '_')}.json"
                        zip_file.writestr(filename, template_json)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    "Download ZIP",
                    zip_buffer,
                    file_name="workflow_templates.zip",
                    mime="application/zip"
                )
            else:
                st.info("No templates to export.")


def render_openevolve_config_panel(session_key: str = "openevolve_config") -> Dict[str, Any]:
    """Render comprehensive OpenEvolve configuration panel with all 211 parameters
    
    Args:
        session_key: Session state key for storing configuration
        
    Returns:
        Dictionary with complete OpenEvolve configuration
    """
    from ui_config import OPENEVOLVE_PARAMS, OPENEVOLVE_PRESETS
    
    st.subheader("🧬 OpenEvolve Configuration")
    
    # Initialize config in session state if not exists
    if session_key not in st.session_state:
        st.session_state[session_key] = {}
    
    config = st.session_state[session_key]
    
    # Preset selection with descriptions
    col1, col2 = st.columns([2, 1])
    with col1:
        preset = st.selectbox(
            "Configuration Preset",
            ["custom", "fast", "balanced", "thorough", "research"],
            help="Select a preset configuration or customize your own"
        )
    
    with col2:
        if st.button("Load Preset", disabled=(preset == "custom")):
            if preset in OPENEVOLVE_PRESETS:
                config.update(OPENEVOLVE_PRESETS[preset])
                st.success(f"Loaded {preset} preset!")
                st.rerun()
    
    # Show preset description
    preset_descriptions = {
        "fast": "⚡ Quick prototyping with minimal iterations (5 iterations, 10 population)",
        "balanced": "⚖️ General use with core features (20 iterations, 30 population)",
        "thorough": "🎯 Production use with quality diversity (50 iterations, 50 population)",
        "research": "🔬 Maximum exploration with all features (100 iterations, 100 population)"
    }
    
    if preset != "custom" and preset in preset_descriptions:
        st.info(preset_descriptions[preset])
    
    # Tabs for parameter categories
    tabs = st.tabs([
        "Core", "Selection", "Quality Diversity", "Multi-Objective", 
        "Evaluation", "Island Model", "Resources", "Advanced"
    ])
    
    # Tab 1: Core Evolution Parameters
    with tabs[0]:
        st.markdown("### Core Evolution Parameters")
        
        params = OPENEVOLVE_PARAMS["core_evolution"]
        
        config['evolution_mode'] = st.selectbox(
            "Evolution Mode",
            params['evolution_mode']['options'],
            index=params['evolution_mode']['options'].index(config.get('evolution_mode', params['evolution_mode']['default'])),
            help=params['evolution_mode']['description']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            config['max_iterations'] = st.number_input(
                "Max Iterations",
                min_value=params['max_iterations']['min'],
                max_value=params['max_iterations']['max'],
                value=config.get('max_iterations', params['max_iterations']['default']),
                help=params['max_iterations']['description']
            )
            
            config['temperature'] = st.slider(
                "Temperature",
                min_value=params['temperature']['min'],
                max_value=params['temperature']['max'],
                value=float(config.get('temperature', params['temperature']['default'])),
                help=params['temperature']['description']
            )
        
        with col2:
            config['population_size'] = st.number_input(
                "Population Size",
                min_value=params['population_size']['min'],
                max_value=params['population_size']['max'],
                value=config.get('population_size', params['population_size']['default']),
                help=params['population_size']['description']
            )
            
            config['max_tokens'] = st.number_input(
                "Max Tokens",
                min_value=params['max_tokens']['min'],
                max_value=params['max_tokens']['max'],
                value=config.get('max_tokens', params['max_tokens']['default']),
                help=params['max_tokens']['description']
            )
        
        config['seed'] = st.number_input(
            "Random Seed (optional)",
            min_value=0,
            max_value=2147483647,
            value=config.get('seed', 42) if config.get('seed') is not None else 42,
            help="Set to None for random behavior"
        )
    
    # Tab 2: Selection Parameters
    with tabs[1]:
        st.markdown("### Selection Parameters")
        
        params = OPENEVOLVE_PARAMS["selection"]
        
        st.info("[WARN] Elite + Exploration + Exploitation ratios must sum to 1.0")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            config['elite_ratio'] = st.slider(
                "Elite Ratio",
                min_value=params['elite_ratio']['min'],
                max_value=params['elite_ratio']['max'],
                value=float(config.get('elite_ratio', params['elite_ratio']['default'])),
                help=params['elite_ratio']['description']
            )
        
        with col2:
            config['exploration_ratio'] = st.slider(
                "Exploration Ratio",
                min_value=params['exploration_ratio']['min'],
                max_value=params['exploration_ratio']['max'],
                value=float(config.get('exploration_ratio', params['exploration_ratio']['default'])),
                help=params['exploration_ratio']['description']
            )
        
        with col3:
            config['exploitation_ratio'] = st.slider(
                "Exploitation Ratio",
                min_value=params['exploitation_ratio']['min'],
                max_value=params['exploitation_ratio']['max'],
                value=float(config.get('exploitation_ratio', params['exploitation_ratio']['default'])),
                help=params['exploitation_ratio']['description']
            )
        
        # Validate ratio sum
        ratio_sum = config['elite_ratio'] + config['exploration_ratio'] + config['exploitation_ratio']
        if abs(ratio_sum - 1.0) > 0.01:
            st.error(f"[WARN] Ratios sum to {ratio_sum:.2f}, must equal 1.0")
        else:
            st.success(f"[OK] Ratios sum to {ratio_sum:.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            config['tournament_size'] = st.number_input(
                "Tournament Size",
                min_value=params['tournament_size']['min'],
                max_value=params['tournament_size']['max'],
                value=config.get('tournament_size', params['tournament_size']['default']),
                help=params['tournament_size']['description']
            )
        
        with col2:
            config['selection_pressure'] = st.slider(
                "Selection Pressure",
                min_value=params['selection_pressure']['min'],
                max_value=params['selection_pressure']['max'],
                value=float(config.get('selection_pressure', params['selection_pressure']['default'])),
                help=params['selection_pressure']['description']
            )
    
    # Tab 3: Quality Diversity Parameters
    with tabs[2]:
        st.markdown("### Quality Diversity Parameters")
        
        params = OPENEVOLVE_PARAMS["quality_diversity"]
        
        config['enable_quality_diversity'] = st.checkbox(
            "Enable Quality Diversity (MAP-Elites)",
            value=config.get('enable_quality_diversity', params['enable_quality_diversity']['default']),
            help=params['enable_quality_diversity']['description']
        )
        
        if config['enable_quality_diversity']:
            col1, col2 = st.columns(2)
            with col1:
                config['archive_size'] = st.number_input(
                    "Archive Size",
                    min_value=params['archive_size']['min'],
                    max_value=params['archive_size']['max'],
                    value=config.get('archive_size', params['archive_size']['default']),
                    help=params['archive_size']['description']
                )
                
                config['diversity_metric'] = st.selectbox(
                    "Diversity Metric",
                    params['diversity_metric']['options'],
                    index=params['diversity_metric']['options'].index(
                        config.get('diversity_metric', params['diversity_metric']['default'])
                    ),
                    help=params['diversity_metric']['description']
                )
            
            with col2:
                config['feature_bins'] = st.number_input(
                    "Feature Bins",
                    min_value=params['feature_bins']['min'],
                    max_value=params['feature_bins']['max'],
                    value=config.get('feature_bins', params['feature_bins']['default']),
                    help=params['feature_bins']['description']
                )
                
                config['novelty_threshold'] = st.slider(
                    "Novelty Threshold",
                    min_value=params['novelty_threshold']['min'],
                    max_value=params['novelty_threshold']['max'],
                    value=float(config.get('novelty_threshold', params['novelty_threshold']['default'])),
                    help=params['novelty_threshold']['description']
                )
            
            # Feature dimensions
            default_dims = params['feature_dimensions']['default']
            dims_text = st.text_input(
                "Feature Dimensions (comma-separated)",
                value=", ".join(config.get('feature_dimensions', default_dims)),
                help="Behavior dimensions for the archive (e.g., complexity, novelty, quality)"
            )
            config['feature_dimensions'] = [d.strip() for d in dims_text.split(",") if d.strip()]
    
    # Tab 4: Multi-Objective Parameters
    with tabs[3]:
        st.markdown("### Multi-Objective Optimization Parameters")
        
        params = OPENEVOLVE_PARAMS["multi_objective"]
        
        config['enable_multi_objective'] = st.checkbox(
            "Enable Multi-Objective Optimization",
            value=config.get('enable_multi_objective', params['enable_multi_objective']['default']),
            help=params['enable_multi_objective']['description']
        )
        
        if config['enable_multi_objective']:
            col1, col2 = st.columns(2)
            with col1:
                config['pareto_front_size'] = st.number_input(
                    "Pareto Front Size",
                    min_value=params['pareto_front_size']['min'],
                    max_value=params['pareto_front_size']['max'],
                    value=config.get('pareto_front_size', params['pareto_front_size']['default']),
                    help=params['pareto_front_size']['description']
                )
            
            with col2:
                config['crowding_distance_weight'] = st.slider(
                    "Crowding Distance Weight",
                    min_value=params['crowding_distance_weight']['min'],
                    max_value=params['crowding_distance_weight']['max'],
                    value=float(config.get('crowding_distance_weight', params['crowding_distance_weight']['default'])),
                    help=params['crowding_distance_weight']['description']
                )
            
            # Objectives
            default_objs = params['objectives']['default']
            objs_text = st.text_input(
                "Objectives (comma-separated)",
                value=", ".join(config.get('objectives', default_objs)),
                help="List of objectives to optimize (e.g., quality, efficiency, readability)"
            )
            config['objectives'] = [o.strip() for o in objs_text.split(",") if o.strip()]
    
    # Tab 5: Evaluation Parameters
    with tabs[4]:
        st.markdown("### Evaluation Parameters")
        
        params = OPENEVOLVE_PARAMS["evaluation"]
        
        col1, col2 = st.columns(2)
        with col1:
            config['enable_cascade_evaluation'] = st.checkbox(
                "Enable Cascade Evaluation",
                value=config.get('enable_cascade_evaluation', params['enable_cascade_evaluation']['default']),
                help=params['enable_cascade_evaluation']['description']
            )
            
            config['parallel_evaluations'] = st.number_input(
                "Parallel Evaluations",
                min_value=params['parallel_evaluations']['min'],
                max_value=params['parallel_evaluations']['max'],
                value=config.get('parallel_evaluations', params['parallel_evaluations']['default']),
                help=params['parallel_evaluations']['description']
            )
            
            config['ensemble_size'] = st.number_input(
                "Ensemble Size",
                min_value=params['ensemble_size']['min'],
                max_value=params['ensemble_size']['max'],
                value=config.get('ensemble_size', params['ensemble_size']['default']),
                help=params['ensemble_size']['description']
            )
        
        with col2:
            config['evaluation_timeout'] = st.number_input(
                "Evaluation Timeout (seconds)",
                min_value=params['evaluation_timeout']['min'],
                max_value=params['evaluation_timeout']['max'],
                value=config.get('evaluation_timeout', params['evaluation_timeout']['default']),
                help=params['evaluation_timeout']['description']
            )
            
            config['max_retries'] = st.number_input(
                "Max Retries",
                min_value=params['max_retries']['min'],
                max_value=params['max_retries']['max'],
                value=config.get('max_retries', params['max_retries']['default']),
                help=params['max_retries']['description']
            )
            
            config['consensus_threshold'] = st.slider(
                "Consensus Threshold",
                min_value=params['consensus_threshold']['min'],
                max_value=params['consensus_threshold']['max'],
                value=float(config.get('consensus_threshold', params['consensus_threshold']['default'])),
                help=params['consensus_threshold']['description']
            )
        
        if config.get('enable_cascade_evaluation'):
            thresholds_text = st.text_input(
                "Cascade Thresholds (comma-separated)",
                value=", ".join(str(t) for t in config.get('cascade_thresholds', params['cascade_thresholds']['default'])),
                help="Thresholds for each cascade level (e.g., 0.5, 0.75, 0.9)"
            )
            config['cascade_thresholds'] = [float(t.strip()) for t in thresholds_text.split(",") if t.strip()]
    
    # Tab 6: Island Model Parameters
    with tabs[5]:
        st.markdown("### Island Model Parameters")
        
        params = OPENEVOLVE_PARAMS["island_model"]
        
        config['enable_island_model'] = st.checkbox(
            "Enable Island Model Evolution",
            value=config.get('enable_island_model', params['enable_island_model']['default']),
            help=params['enable_island_model']['description']
        )
        
        if config['enable_island_model']:
            col1, col2 = st.columns(2)
            with col1:
                config['num_islands'] = st.number_input(
                    "Number of Islands",
                    min_value=params['num_islands']['min'],
                    max_value=params['num_islands']['max'],
                    value=config.get('num_islands', params['num_islands']['default']),
                    help=params['num_islands']['description']
                )
                
                config['migration_interval'] = st.number_input(
                    "Migration Interval",
                    min_value=params['migration_interval']['min'],
                    max_value=params['migration_interval']['max'],
                    value=config.get('migration_interval', params['migration_interval']['default']),
                    help=params['migration_interval']['description']
                )
            
            with col2:
                config['migration_size'] = st.number_input(
                    "Migration Size",
                    min_value=params['migration_size']['min'],
                    max_value=params['migration_size']['max'],
                    value=config.get('migration_size', params['migration_size']['default']),
                    help=params['migration_size']['description']
                )
                
                config['migration_topology'] = st.selectbox(
                    "Migration Topology",
                    params['migration_topology']['options'],
                    index=params['migration_topology']['options'].index(
                        config.get('migration_topology', params['migration_topology']['default'])
                    ),
                    help=params['migration_topology']['description']
                )
    
    # Tab 7: Resource Management Parameters
    with tabs[6]:
        st.markdown("### Resource Management Parameters")
        
        params = OPENEVOLVE_PARAMS["resources"]
        
        col1, col2 = st.columns(2)
        with col1:
            config['max_cost_usd'] = st.number_input(
                "Max Cost (USD)",
                min_value=params['max_cost_usd']['min'],
                max_value=params['max_cost_usd']['max'],
                value=float(config.get('max_cost_usd', params['max_cost_usd']['default'])),
                help=params['max_cost_usd']['description']
            )
            
            config['max_execution_time'] = st.number_input(
                "Max Execution Time (seconds)",
                min_value=params['max_execution_time']['min'],
                max_value=params['max_execution_time']['max'],
                value=config.get('max_execution_time', params['max_execution_time']['default']),
                help=params['max_execution_time']['description']
            )
        
        with col2:
            config['max_api_calls'] = st.number_input(
                "Max API Calls",
                min_value=params['max_api_calls']['min'],
                max_value=params['max_api_calls']['max'],
                value=config.get('max_api_calls', params['max_api_calls']['default']),
                help=params['max_api_calls']['description']
            )
            
            config['memory_limit_mb'] = st.number_input(
                "Memory Limit (MB)",
                min_value=params['memory_limit_mb']['min'],
                max_value=params['memory_limit_mb']['max'],
                value=config.get('memory_limit_mb', params['memory_limit_mb']['default']),
                help=params['memory_limit_mb']['description']
            )
    
    # Tab 8: Advanced Features
    with tabs[7]:
        st.markdown("### Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Artifacts & Checkpointing")
            config['enable_artifacts'] = st.checkbox(
                "Enable Artifacts",
                value=config.get('enable_artifacts', OPENEVOLVE_PARAMS["artifacts"]['enable_artifacts']['default']),
                help=OPENEVOLVE_PARAMS["artifacts"]['enable_artifacts']['description']
            )
            
            config['checkpoint_interval'] = st.number_input(
                "Checkpoint Interval",
                min_value=OPENEVOLVE_PARAMS["checkpointing"]['checkpoint_interval']['min'],
                max_value=OPENEVOLVE_PARAMS["checkpointing"]['checkpoint_interval']['max'],
                value=config.get('checkpoint_interval', OPENEVOLVE_PARAMS["checkpointing"]['checkpoint_interval']['default']),
                help=OPENEVOLVE_PARAMS["checkpointing"]['checkpoint_interval']['description']
            )
            
            st.markdown("#### Prompt Engineering")
            config['enable_meta_prompting'] = st.checkbox(
                "Enable Meta-Prompting",
                value=config.get('enable_meta_prompting', OPENEVOLVE_PARAMS["prompt_engineering"]['enable_meta_prompting']['default']),
                help=OPENEVOLVE_PARAMS["prompt_engineering"]['enable_meta_prompting']['description']
            )
            
            config['enable_template_stochasticity'] = st.checkbox(
                "Enable Template Stochasticity",
                value=config.get('enable_template_stochasticity', OPENEVOLVE_PARAMS["prompt_engineering"]['enable_template_stochasticity']['default']),
                help=OPENEVOLVE_PARAMS["prompt_engineering"]['enable_template_stochasticity']['description']
            )
        
        with col2:
            st.markdown("#### Distributed Processing")
            config['enable_distributed'] = st.checkbox(
                "Enable Distributed Processing",
                value=config.get('enable_distributed', OPENEVOLVE_PARAMS["distributed"]['enable_distributed']['default']),
                help=OPENEVOLVE_PARAMS["distributed"]['enable_distributed']['description']
            )
            
            if config.get('enable_distributed'):
                config['num_workers'] = st.number_input(
                    "Number of Workers",
                    min_value=OPENEVOLVE_PARAMS["distributed"]['num_workers']['min'],
                    max_value=OPENEVOLVE_PARAMS["distributed"]['num_workers']['max'],
                    value=config.get('num_workers', OPENEVOLVE_PARAMS["distributed"]['num_workers']['default']),
                    help=OPENEVOLVE_PARAMS["distributed"]['num_workers']['description']
                )
            
            st.markdown("#### Termination Criteria")
            config['enable_early_stopping'] = st.checkbox(
                "Enable Early Stopping",
                value=config.get('enable_early_stopping', OPENEVOLVE_PARAMS["termination"]['enable_early_stopping']['default']),
                help=OPENEVOLVE_PARAMS["termination"]['enable_early_stopping']['description']
            )
            
            if config.get('enable_early_stopping'):
                config['fitness_threshold'] = st.slider(
                    "Fitness Threshold",
                    min_value=OPENEVOLVE_PARAMS["termination"]['fitness_threshold']['min'],
                    max_value=OPENEVOLVE_PARAMS["termination"]['fitness_threshold']['max'],
                    value=float(config.get('fitness_threshold', OPENEVOLVE_PARAMS["termination"]['fitness_threshold']['default'])),
                    help=OPENEVOLVE_PARAMS["termination"]['fitness_threshold']['description']
                )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Configuration"):
            st.session_state[session_key] = config
            st.success("Configuration saved!")
    
    with col2:
        if st.button("📋 Export as JSON"):
            json_str = json.dumps(config, indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                file_name="openevolve_config.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            st.session_state[session_key] = {}
            st.rerun()
    
    return config

def render_openevolve_progress_monitor(operation_id: str, metrics_collector, auto_refresh: bool = True):
    """Render comprehensive real-time OpenEvolve progress monitor with detailed metrics
    
    Args:
        operation_id: ID of the operation to monitor
        metrics_collector: MetricsCollector instance
        auto_refresh: Whether to auto-refresh the display
    """
    st.subheader("🔄 Evolution Progress Monitor")
    
    # Get active operation
    active_ops = metrics_collector.get_active_operations()
    current_op = next((op for op in active_ops if op.operation_id == operation_id), None)
    
    if not current_op:
        st.info("ℹ️ No active operation found")
        
        # Show recent completed operations
        recent_ops = metrics_collector.get_recent_operations(limit=5)
        if recent_ops:
            st.markdown("### Recent Operations")
            for op in recent_ops:
                with st.expander(f"Operation {op.operation_id[:8]} - {op.status}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Iterations", op.iterations_completed)
                    with col2:
                        st.metric("Best Fitness", f"{op.best_fitness:.3f}")
                    with col3:
                        st.metric("Duration", f"{op.duration:.1f}s")
        return
    
    # Status indicator
    status_colors = {
        "running": "🟢",
        "paused": "🟡",
        "completed": "[OK]",
        "failed": "[FAIL]"
    }
    st.markdown(f"**Status:** {status_colors.get(current_op.status, '⚪')} {current_op.status.upper()}")
    
    # Progress bar with percentage
    progress = current_op.iterations_completed / current_op.max_iterations if current_op.max_iterations > 0 else 0
    st.progress(progress)
    st.caption(f"Progress: {progress*100:.1f}% ({current_op.iterations_completed}/{current_op.max_iterations} iterations)")
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Iteration",
            f"{current_op.iterations_completed}",
            delta=f"/{current_op.max_iterations}",
            help="Current iteration out of maximum"
        )
    
    with col2:
        fitness_delta = current_op.best_fitness - current_op.initial_fitness if hasattr(current_op, 'initial_fitness') else None
        st.metric(
            "Best Fitness",
            f"{current_op.best_fitness:.3f}",
            delta=f"+{fitness_delta:.3f}" if fitness_delta and fitness_delta > 0 else None,
            help="Best fitness score achieved"
        )
    
    with col3:
        st.metric(
            "Avg Fitness",
            f"{current_op.avg_fitness:.3f}" if hasattr(current_op, 'avg_fitness') else "N/A",
            help="Average fitness of current population"
        )
    
    with col4:
        st.metric(
            "Diversity",
            f"{current_op.population_diversity:.3f}" if hasattr(current_op, 'population_diversity') else "N/A",
            help="Population diversity score"
        )
    
    with col5:
        elapsed = time.time() - current_op.start_time if hasattr(current_op, 'start_time') else 0
        st.metric(
            "Elapsed",
            f"{elapsed:.0f}s",
            help="Time elapsed since start"
        )
    
    # Detailed metrics in expandable sections
    with st.expander("📊 Detailed Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Metrics")
            st.write(f"**Evolution Mode:** {current_op.evolution_mode if hasattr(current_op, 'evolution_mode') else 'N/A'}")
            st.write(f"**Population Size:** {current_op.population_size if hasattr(current_op, 'population_size') else 'N/A'}")
            st.write(f"**Archive Size:** {current_op.archive_size if hasattr(current_op, 'archive_size') else 'N/A'}")
            st.write(f"**Evaluations:** {current_op.total_evaluations if hasattr(current_op, 'total_evaluations') else 'N/A'}")
        
        with col2:
            st.markdown("#### Resource Usage")
            st.write(f"**API Calls:** {current_op.api_calls if hasattr(current_op, 'api_calls') else 'N/A'}")
            st.write(f"**Tokens Used:** {current_op.tokens_used:,}" if hasattr(current_op, 'tokens_used') else "N/A")
            st.write(f"**Cost:** ${current_op.cost_usd:.2f}" if hasattr(current_op, 'cost_usd') else "N/A")
            st.write(f"**Memory:** {current_op.memory_mb:.1f} MB" if hasattr(current_op, 'memory_mb') else "N/A")
    
    # Fitness history chart
    if hasattr(current_op, 'fitness_history') and current_op.fitness_history:
        with st.expander("📈 Fitness Evolution", expanded=True):
            import pandas as pd
            import plotly.graph_objects as go
            
            df = pd.DataFrame({
                'Iteration': range(len(current_op.fitness_history)),
                'Fitness': current_op.fitness_history
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Iteration'],
                y=df['Fitness'],
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title='Fitness Evolution',
                xaxis_title='Iteration',
                yaxis_title='Fitness',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Estimated time remaining
    if hasattr(current_op, 'start_time') and current_op.iterations_completed > 0:
        elapsed = time.time() - current_op.start_time
        avg_time_per_iter = elapsed / current_op.iterations_completed
        remaining_iters = current_op.max_iterations - current_op.iterations_completed
        estimated_remaining = avg_time_per_iter * remaining_iters
        
        st.info(f"⏱️ Estimated time remaining: {estimated_remaining:.0f} seconds ({estimated_remaining/60:.1f} minutes)")
    
    # Auto-refresh
    if auto_refresh and current_op.status == "running":
        st.caption("🔄 Auto-refreshing every 5 seconds...")
        time.sleep(5)
        st.rerun()

def render_quality_diversity_archive(archive_data: List[Dict[str, Any]], feature_dimensions: Optional[List[str]] = None):
    """Render comprehensive quality diversity archive visualization with interactive features
    
    Args:
        archive_data: List of archive entries with fitness and behavior
        feature_dimensions: List of feature dimension names
    """
    st.subheader("🗺️ Quality Diversity Archive")
    
    if not archive_data:
        st.info("📭 Archive is empty - no solutions have been added yet")
        return
    
    # Extract feature dimensions if not provided
    if not feature_dimensions and archive_data:
        first_behavior = archive_data[0].get('behavior', {})
        feature_dimensions = list(first_behavior.keys())
    
    # Archive statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Archive Size",
            len(archive_data),
            help="Number of solutions in the archive"
        )
    
    with col2:
        avg_fitness = sum(e.get('fitness', 0) for e in archive_data) / len(archive_data)
        st.metric(
            "Avg Fitness",
            f"{avg_fitness:.3f}",
            help="Average fitness across all archived solutions"
        )
    
    with col3:
        max_fitness = max(e.get('fitness', 0) for e in archive_data)
        st.metric(
            "Best Fitness",
            f"{max_fitness:.3f}",
            help="Highest fitness in the archive"
        )
    
    with col4:
        # Estimate coverage (assuming 10x10 grid)
        grid_size = 10
        total_cells = grid_size * grid_size
        coverage = min(len(archive_data) / total_cells * 100, 100)
        st.metric(
            "Coverage",
            f"{coverage:.1f}%",
            help="Percentage of behavior space covered"
        )
    
    # Visualization tabs
    viz_tabs = st.tabs(["Heatmap", "Distribution", "Top Solutions", "Details"])
    
    # Tab 1: Heatmap visualization
    with viz_tabs[0]:
        if feature_dimensions and len(feature_dimensions) >= 2:
            # Use the comprehensive heatmap from analytics_dashboard
            from analytics_dashboard import render_diversity_heatmap
            render_diversity_heatmap(archive_data, feature_dimensions)
        else:
            st.warning("Need at least 2 feature dimensions for heatmap visualization")
    
    # Tab 2: Distribution charts
    with viz_tabs[1]:
        st.markdown("### Fitness Distribution")
        
        import pandas as pd
        import plotly.express as px
        
        fitness_values = [e.get('fitness', 0) for e in archive_data]
        
        fig = px.histogram(
            fitness_values,
            nbins=20,
            title="Fitness Distribution in Archive",
            labels={'value': 'Fitness', 'count': 'Number of Solutions'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Behavior dimension distributions
        if feature_dimensions:
            st.markdown("### Behavior Dimension Distributions")
            
            for dim in feature_dimensions[:3]:  # Show first 3 dimensions
                dim_values = [e.get('behavior', {}).get(dim, 0) for e in archive_data]
                
                fig = px.histogram(
                    dim_values,
                    nbins=15,
                    title=f"{dim.title()} Distribution",
                    labels={'value': dim.title(), 'count': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Top solutions
    with viz_tabs[2]:
        st.markdown("### 🏆 Top Solutions by Fitness")
        
        # Sort by fitness
        sorted_archive = sorted(archive_data, key=lambda x: x.get('fitness', 0), reverse=True)
        
        # Number of solutions to show
        top_n = st.slider("Number of top solutions to display", 5, min(50, len(archive_data)), 10)
        
        # Display top solutions
        for i, entry in enumerate(sorted_archive[:top_n], 1):
            with st.expander(f"#{i} - Fitness: {entry.get('fitness', 0):.3f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Fitness Score**")
                    st.write(f"{entry.get('fitness', 0):.4f}")
                    
                    if 'iteration' in entry:
                        st.markdown("**Added at Iteration**")
                        st.write(entry['iteration'])
                
                with col2:
                    st.markdown("**Behavior Characteristics**")
                    behavior = entry.get('behavior', {})
                    for dim, value in behavior.items():
                        st.write(f"- {dim}: {value:.3f}")
                
                # Show solution content if available
                if 'content' in entry:
                    st.markdown("**Solution Content**")
                    st.code(entry['content'][:500] + "..." if len(entry['content']) > 500 else entry['content'])
    
    # Tab 4: Detailed table
    with viz_tabs[3]:
        st.markdown("### Archive Details")
        
        # Create DataFrame
        table_data = []
        for i, entry in enumerate(archive_data, 1):
            row = {
                'Index': i,
                'Fitness': f"{entry.get('fitness', 0):.3f}",
                'Iteration': entry.get('iteration', 'N/A')
            }
            
            # Add behavior dimensions
            behavior = entry.get('behavior', {})
            for dim in feature_dimensions[:5] if feature_dimensions else []:  # Show first 5 dimensions
                row[dim.title()] = f"{behavior.get(dim, 0):.3f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Add search/filter
        search = st.text_input("🔍 Search archive", placeholder="Filter by any column...")
        if search:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Export option
        if st.button("📥 Export Archive as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name="quality_diversity_archive.csv",
                mime="text/csv"
            )
    
    # Archive statistics summary
    with st.expander("📊 Archive Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fitness Statistics**")
            fitness_values = [e.get('fitness', 0) for e in archive_data]
            st.write(f"- Min: {min(fitness_values):.3f}")
            st.write(f"- Max: {max(fitness_values):.3f}")
            st.write(f"- Mean: {sum(fitness_values)/len(fitness_values):.3f}")
            st.write(f"- Median: {sorted(fitness_values)[len(fitness_values)//2]:.3f}")
        
        with col2:
            st.markdown("**Archive Characteristics**")
            st.write(f"- Total Solutions: {len(archive_data)}")
            st.write(f"- Feature Dimensions: {len(feature_dimensions) if feature_dimensions else 0}")
            st.write(f"- Estimated Coverage: {coverage:.1f}%")
            
            # Calculate diversity
            if len(archive_data) > 1:
                # Simple diversity measure: variance in fitness
                import numpy as np
                diversity = np.var(fitness_values)
                st.write(f"- Fitness Diversity: {diversity:.3f}")

def render_monitoring_tab(workflow_state=None):
    """
    Renders the monitoring tab for tracking CrewAI workflow execution.
    This function handles the stage mentioned in workflow_engine.py where
    the monitoring is done in the render_monitoring_tab UI function.
    """
    import time
    # MIGRATED: was crewai_client import CrewAIClient
    from crewai_client import CrewAIClient
    
    st.header("📊 Workflow Monitoring")
    
    if not workflow_state or not hasattr(workflow_state, 'CrewAI_workflow_id') or not workflow_state.CrewAI_workflow_id:
        st.info("No active CrewAI workflow to monitor. Workflow delegation has not been initiated yet.")
        return
    
    st.subheader(f"Monitoring Workflow: {workflow_state.CrewAI_workflow_id}")
    
    # Initialize CrewAI client
    client = CrewAIClient()
    
    # If we have the mapping of SGDW IDs to CrewAI ticket IDs
    if hasattr(workflow_state, 'id_to_ticket_id_map') and workflow_state.id_to_ticket_id_map:
        st.info("Tracking individual tickets...")
        
        try:
            # Get all tickets for this workflow from crewai # MIGRATED: was CrewAI
            workflow_tickets = client.get_workflow_tickets(workflow_state.CrewAI_workflow_id)
            
            if workflow_tickets:
                st.write(f"Found {len(workflow_tickets)} tickets in the workflow")
                
                # Create a dataframe to display ticket status
                import pandas as pd
                ticket_data = []
                
                for ticket in workflow_tickets:
                    ticket_data.append({
                        'Ticket ID': ticket.get('ticket_id', 'N/A'),
                        'Title': ticket.get('title', 'N/A')[:50] + '...' if len(str(ticket.get('title', 'N/A'))) > 50 else ticket.get('title', 'N/A'),
                        'Status': ticket.get('status', 'N/A'),
                        'Type': ticket.get('ticket_type', 'N/A'),
                        'Priority': ticket.get('priority', 'N/A'),
                        'Created': ticket.get('created_at', 'N/A')[:19] if ticket.get('created_at') else 'N/A',
                        'Blocked By': len(ticket.get('blocked_by_ticket_ids', [])),
                        'Assigned To': ticket.get('assigned_agent_id', 'Unassigned')
                    })
                
                df = pd.DataFrame(ticket_data)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                status_counts = {}
                for ticket in workflow_tickets:
                    status = ticket.get('status', 'Unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                st.subheader("Status Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tickets", len(workflow_tickets))
                
                with col2:
                    st.metric("Completed", status_counts.get('completed', 0))
                
                with col3:
                    st.metric("In Progress", status_counts.get('in_progress', 0) + status_counts.get('todo', 0))
                
                with col4:
                    st.metric("Blocked", status_counts.get('blocked', 0))
                
                # Auto-refresh option
                auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=True)
                if auto_refresh:
                    time.sleep(30)
                    st.rerun()
            else:
                st.warning("No tickets found for this workflow in CrewAI yet.")
                
        except Exception as e:
            st.error(f"Error fetching tickets from crewai # MIGRATED: was CrewAI: {e}")
            st.info("This may indicate that the tickets haven't been created yet or there's a connection issue.")
    else:
        st.info("Waiting for workflow delegation to CrewAI to complete...")
        
    # Refresh button
    if st.button("Refresh Status"):
        st.rerun()


"""
Additional UI Components for Sovereign-Grade Decomposition Workflow

This file contains additional UI components that need to be added to ui_components.py
"""

from typing import Dict, Any, Optional, List
from ui_shim import ui as st
import time


def render_workflow_orchestrator(
    teams: List['Team'],
    gauntlets: List['GauntletDefinition'],
    current_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Render the workflow orchestrator UI component for configuring workflow execution.

    This component allows users to:
    - Select teams for each workflow stage
    - Select gauntlets for critique and verification
    - Configure advanced workflow options
    - Set resource limits and optimization parameters
    - Enable/disable learning features
    - Configure auto-approval settings

    Args:
        teams: List of available teams
        gauntlets: List of available gauntlets
        current_config: Current configuration (for editing existing workflows)

    Returns:
        Dictionary containing the workflow configuration
    """
    st.subheader("🎛️ Workflow Orchestrator Configuration")

    if current_config is None:
        current_config = {}

    # Initialize config
    config = {
        "content_analyzer_team": current_config.get("content_analyzer_team", ""),
        "planner_team": current_config.get("planner_team", ""),
        "solver_team": current_config.get("solver_team", ""),
        "patcher_team": current_config.get("patcher_team", ""),
        "assembler_team": current_config.get("assembler_team", ""),
        "sub_problem_red_gauntlet": current_config.get("sub_problem_red_gauntlet", ""),
        "sub_problem_gold_gauntlet": current_config.get("sub_problem_gold_gauntlet", ""),
        "final_red_gauntlet": current_config.get("final_red_gauntlet", ""),
        "final_gold_gauntlet": current_config.get("final_gold_gauntlet", ""),
        "max_refinement_loops": current_config.get("max_refinement_loops", 3),
        "auto_approval_enabled": current_config.get("auto_approval_enabled", False),
        "parallel_processing_enabled": current_config.get("parallel_processing_enabled", False),
        "learning_enabled": current_config.get("learning_enabled", False),
        "resource_limits": current_config.get("resource_limits", {}),
        "mdap_enabled": current_config.get("mdap_enabled", False),
        "mdap_config": current_config.get("mdap_config", {}),
        "maker_enabled": current_config.get("maker_enabled", False),
        "maker_config": current_config.get("maker_config", {})
    }

    # Team selection section
    with st.expander("👥 Team Assignment", expanded=True):
        st.write("Select teams for each stage of the workflow:")

        # Group teams by role
        blue_teams = [t for t in teams if t.role == "Blue"]
        red_teams = [t for t in teams if t.role == "Red"]
        gold_teams = [t for t in teams if t.role == "Gold"]

        # Content Analyzer Team
        if blue_teams:
            config["content_analyzer_team"] = st.selectbox(
                "Content Analyzer Team (Stage 0)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("content_analyzer_team") else [t.name for t in blue_teams].index(config.get("content_analyzer_team")) + 1 if config.get("content_analyzer_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for analyzing the problem content"
            )

        # Planner Team
        if blue_teams:
            config["planner_team"] = st.selectbox(
                "Planner Team (Stage 1)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("planner_team") else [t.name for t in blue_teams].index(config.get("planner_team")) + 1 if config.get("planner_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for decomposing the problem"
            )

        # Solver Team
        if blue_teams:
            config["solver_team"] = st.selectbox(
                "Solver Team (Stage 3)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("solver_team") else [t.name for t in blue_teams].index(config.get("solver_team")) + 1 if config.get("solver_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for generating solutions"
            )

        # Patcher Team
        if blue_teams:
            config["patcher_team"] = st.selectbox(
                "Patcher Team (Stage 3)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("patcher_team") else [t.name for t in blue_teams].index(config.get("patcher_team")) + 1 if config.get("patcher_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for fixing rejected solutions"
            )

        # Assembler Team
        if blue_teams:
            config["assembler_team"] = st.selectbox(
                "Assembler Team (Stage 4)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("assembler_team") else [t.name for t in blue_teams].index(config.get("assembler_team")) + 1 if config.get("assembler_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for assembling the final solution"
            )

    # Gauntlet selection section
    with st.expander("⚔️ Gauntlet Assignment", expanded=True):
        st.write("Select gauntlets for critique and verification:")

        # Sub-problem Red Team Gauntlet
        if red_teams:
            config["sub_problem_red_gauntlet"] = st.selectbox(
                "Sub-Problem Red Team Gauntlet",
                options=[""] + [g.name for g in gauntlets if any(m.role == "Red" for m in red_teams if m.name == g.team_name)],
                index=0,
                help="Gauntlet for critiquing sub-problem solutions"
            )

        # Sub-problem Gold Team Gauntlet
        if gold_teams:
            config["sub_problem_gold_gauntlet"] = st.selectbox(
                "Sub-Problem Gold Team Gauntlet",
                options=[""] + [g.name for g in gauntlets if any(m.role == "Gold" for m in gold_teams if m.name == g.team_name)],
                index=0,
                help="Gauntlet for verifying sub-problem solutions"
            )

        # Final Red Team Gauntlet
        if red_teams:
            config["final_red_gauntlet"] = st.selectbox(
                "Final Red Team Gauntlet",
                options=[""] + ["Final_Red_Gauntlet"],
                index=0,
                help="Gauntlet for final adversarial testing"
            )

        # Final Gold Team Gauntlet
        if gold_teams:
            config["final_gold_gauntlet"] = st.selectbox(
                "Final Gold Team Gauntlet",
                options=[""] + ["Final_Gold_Gauntlet"],
                index=0,
                help="Gauntlet for final verification"
            )

    # Advanced configuration section
    with st.expander("🔧 Advanced Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            config["max_refinement_loops"] = st.number_input(
                "Max Refinement Loops",
                min_value=1,
                max_value=10,
                value=config["max_refinement_loops"],
                help="Maximum number of self-healing iterations"
            )

            config["auto_approval_enabled"] = st.checkbox(
                "Enable Auto-Approval",
                value=config["auto_approval_enabled"],
                help="Automatically approve plans that meet criteria"
            )

        with col2:
            config["parallel_processing_enabled"] = st.checkbox(
                "Enable Parallel Processing",
                value=config["parallel_processing_enabled"],
                help="Solve independent sub-problems in parallel"
            )

            config["learning_enabled"] = st.checkbox(
                "Enable Learning",
                value=config["learning_enabled"],
                help="Extract and learn from workflow execution"
            )

        st.markdown("---")
        st.write("MDAP / MAKER Execution")
        config["mdap_enabled"] = st.checkbox(
            "Enable MDAP for solution generation",
            value=config["mdap_enabled"],
            help="Use MDAP voting and red-flagging during solution generation"
        )
        config["maker_enabled"] = st.checkbox(
            "Enable MAKER for solution generation",
            value=config["maker_enabled"],
            help="Use MAKER stepwise voting and checkpointing during solution generation"
        )
        mdap_config_text = st.text_area(
            "MDAP Config (JSON)",
            value=json.dumps(config["mdap_config"], indent=2) if config["mdap_config"] else "{}",
            height=140
        )
        maker_config_text = st.text_area(
            "MAKER Config (JSON)",
            value=json.dumps(config["maker_config"], indent=2) if config["maker_config"] else "{}",
            height=140
        )
        try:
            config["mdap_config"] = json.loads(mdap_config_text) if mdap_config_text.strip() else {}
        except json.JSONDecodeError:
            st.warning("Invalid MDAP config JSON. Using defaults.")
            config["mdap_config"] = {}
        try:
            config["maker_config"] = json.loads(maker_config_text) if maker_config_text.strip() else {}
        except json.JSONDecodeError:
            st.warning("Invalid MAKER config JSON. Using defaults.")
            config["maker_config"] = {}

    # Resource limits section
    with st.expander("💾 Resource Limits"):
        resource_limits = config["resource_limits"]

        col1, col2, col3 = st.columns(3)

        with col1:
            resource_limits["max_api_calls"] = st.number_input(
                "Max API Calls",
                min_value=0,
                value=resource_limits.get("max_api_calls", 1000),
                help="Maximum number of API calls"
            )

        with col2:
            resource_limits["max_tokens"] = st.number_input(
                "Max Tokens",
                min_value=0,
                value=resource_limits.get("max_tokens", 1000000),
                help="Maximum number of tokens"
            )

        with col3:
            resource_limits["max_cost"] = st.number_input(
                "Max Cost ($)",
                min_value=0.0,
                value=float(resource_limits.get("max_cost", 10.0)),
                step=0.1,
                help="Maximum cost in USD"
            )

        config["resource_limits"] = resource_limits

    # Configuration summary
    st.subheader("📋 Configuration Summary")

    summary_cols = st.columns(6)
    with summary_cols[0]:
        st.metric("Teams Selected", sum(1 for v in config.values() if isinstance(v, str) and v and "team" in v))
    with summary_cols[1]:
        st.metric("Gauntlets Selected", sum(1 for v in config.values() if isinstance(v, str) and v and "gauntlet" in v))
    with summary_cols[2]:
        st.metric("Max Refinements", config["max_refinement_loops"])
    with summary_cols[3]:
        st.metric("Parallel Enabled", "Yes" if config["parallel_processing_enabled"] else "No")
    with summary_cols[4]:
        st.metric("MDAP Enabled", "Yes" if config.get("mdap_enabled") else "No")
    with summary_cols[5]:
        st.metric("MAKER Enabled", "Yes" if config.get("maker_enabled") else "No")

    return config


def render_realtime_monitoring(
    workflow_state: Optional['WorkflowState'] = None
) -> None:
    """
    Render the real-time monitoring dashboard for active workflow executions.

    This component displays:
    - Progress tracking for current workflow
    - Resource usage metrics
    - Performance metrics
    - Interactive controls for workflow management
    - Alert system for issues
    - Log viewer

    Args:
        workflow_state: Current workflow state (if active)
    """
    st.subheader("📊 Real-Time Monitoring")

    if not workflow_state:
        st.info("No active workflow execution. Start a workflow to see real-time monitoring.")
        return

    # Progress section
    with st.container():
        st.write("**Progress Overview**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Stage",
                workflow_state.current_stage,
                help="The current stage of the workflow"
            )

        with col2:
            st.metric(
                "Progress",
                f"{workflow_state.progress * 100:.1f}%",
                delta=None,
                help="Overall workflow progress"
            )

        with col3:
            elapsed_time = time.time() - workflow_state.start_time
            st.metric(
                "Elapsed Time",
                f"{elapsed_time:.0f}s",
                help="Time since workflow started"
            )

        # Progress bar
        st.progress(workflow_state.progress)

    # Resource usage section
    with st.expander("💾 Resource Usage", expanded=True):
        resource_usage = workflow_state.resource_usage if hasattr(workflow_state, 'resource_usage') else {}

        if resource_usage:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "API Calls",
                    resource_usage.get("api_calls", 0),
                    help="Total API calls made"
                )

            with col2:
                st.metric(
                    "Tokens Used",
                    f"{resource_usage.get('tokens_used', 0):,}",
                    help="Total tokens consumed"
                )

            with col3:
                st.metric(
                    "Cost",
                    f"${resource_usage.get('estimated_cost', 0.0):.4f}",
                    help="Estimated cost in USD"
                )

            with col4:
                st.metric(
                    "Memory",
                    f"{resource_usage.get('memory_usage_mb', 0):.1f} MB",
                    help="Memory usage"
                )
        else:
            st.info("No resource usage data available yet.")

    # Performance metrics section
    with st.expander("⚡ Performance Metrics"):
        performance_metrics = workflow_state.performance_metrics if hasattr(workflow_state, 'performance_metrics') else {}

        if performance_metrics:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Sub-Problems Solved",
                    performance_metrics.get("sub_problems_solved", 0),
                    delta=None,
                    help="Number of sub-problems successfully solved"
                )

            with col2:
                st.metric(
                    "Solution Quality",
                    f"{performance_metrics.get('avg_solution_quality', 0.0) * 100:.1f}%",
                    delta=None,
                    help="Average quality of solutions"
                )
        else:
            st.info("No performance metrics available yet.")

    # Interactive controls section
    with st.expander("🎮 Workflow Controls"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⏸️ Pause Workflow", key="pause_workflow"):
                st.warning("Workflow paused (not yet implemented)")

        with col2:
            if st.button("⏹️ Stop Workflow", key="stop_workflow"):
                st.error("Workflow stopped (not yet implemented)")

        with col3:
            if st.button("🔄 Refresh", key="refresh_monitoring"):
                st.rerun()

    # Alert system section
    with st.expander("🚨 Alerts"):
        alerts = []

        # Check for issues
        if workflow_state.refinement_loop_count > 3:
            alerts.append({
                "level": "warning",
                "message": f"High refinement loop count: {workflow_state.refinement_loop_count}"
            })

        if hasattr(workflow_state, 'rejected_sub_problems') and workflow_state.rejected_sub_problems:
            alerts.append({
                "level": "error",
                "message": f"{len(workflow_state.rejected_sub_problems)} sub-problems rejected"
            })

        if alerts:
            for alert in alerts:
                if alert["level"] == "error":
                    st.error(f"[FAIL] {alert['message']}")
                elif alert["level"] == "warning":
                    st.warning(f"[WARN] {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("[OK] No alerts - Workflow running smoothly")

    # Log viewer section
    with st.expander("📜 Log Viewer"):
        st.text_area(
            "Workflow Logs",
            value="Log viewer placeholder - logs would appear here",
            height=200,
            disabled=True,
            help="Real-time workflow logs"
        )

    # Sub-problem status section
    if workflow_state.decomposition_plan and workflow_state.decomposition_plan.sub_problems:
        with st.expander("📋 Sub-Problem Status"):
            st.write("**Sub-Problem Progress:**")

            for sp in workflow_state.decomposition_plan.sub_problems:
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "solved": "[OK]",
                    "failed": "[FAIL]",
                    "requires_rework": "🔧"
                }.get(sp.status, "❓")

                st.write(f"{status_emoji} **{sp.id}**: {sp.status}")
                if sp.status != "pending":
                    st.caption(f"Complexity: {sp.ai_suggested_complexity_score}/10")


# Export list of functions to add to ui_components.py
UI_FUNCTIONS_TO_ADD = [
    render_workflow_orchestrator,
    render_realtime_monitoring,
]

