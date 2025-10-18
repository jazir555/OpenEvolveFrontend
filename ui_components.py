import streamlit as st
from typing import List, Dict, Any, Optional
from workflow_structures import Team, ModelConfig, GauntletDefinition, GauntletRoundRule, DecompositionPlan, SubProblem
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
import json
import time

# Initialize managers (can be done once in the main app or passed via session state)
# These managers handle persistence of Teams and Gauntlets across Streamlit reruns.
if 'team_manager' not in st.session_state:
    st.session_state.team_manager = TeamManager()
if 'gauntlet_manager' not in st.session_state:
    st.session_state.gauntlet_manager = GauntletManager()

def render_team_manager():
    """Renders the Streamlit UI for managing AI teams. Allows users to create, view, edit, and delete teams."""
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
            else:
                new_decomp_system_prompt = None
                new_decomp_user_prompt_template = None

            st.subheader("Team Members (AI Models)")
            num_members = st.number_input("Number of Models in Team", min_value=1, value=1, key="num_new_members")
            
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
                    new_team = Team(name=team_name, role=team_role, members=new_members, description=team_description, content_analysis_system_prompt=new_ca_system_prompt, content_analysis_user_prompt_template=new_ca_user_prompt_template, decomposition_system_prompt=new_decomp_system_prompt, decomposition_user_prompt_template=new_decomp_user_prompt_template)
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

                    # Edit Form
                    with st.form(f"edit_team_form_{team.name}"):
                        edited_team_name = st.text_input("Team Name", value=team.name, key=f"edit_team_name_{team.name}")
                        edited_team_role = st.selectbox("Team Role", ["Blue", "Red", "Gold"], index=["Blue", "Red", "Gold"].index(team.role), key=f"edit_team_role_{team.name}")
                        edited_team_description = st.text_area("Description", value=team.description, key=f"edit_team_description_{team.name}")

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
                        else:
                            edited_ca_system_prompt = None
                            edited_ca_user_prompt_template = None
                            edited_decomp_system_prompt = None
                            edited_decomp_user_prompt_template = None

                        st.subheader("Edit Team Members (AI Models)")
                        # Allow adding/removing members, or editing existing ones.
                        # For simplicity, we'll allow editing existing members and adding new ones.
                        # A more complex UI would allow reordering and more granular control.
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
                                current_api_base = team.members[i].api_base if i < num_existing_members else "https://api.openai.com/v1"
                                current_temperature = team.members[i].temperature if i < num_existing_members else 0.7
                                current_top_p = team.members[i].top_p if i < num_existing_members else 1.0
                                current_max_tokens = team.members[i].max_tokens if i < num_existing_members else 4096
                                current_frequency_penalty = team.members[i].frequency_penalty if i < num_existing_members else 0.0
                                current_presence_penalty = team.members[i].presence_penalty if i < num_existing_members else 0.0
                                current_seed = team.members[i].seed if i < num_existing_members else None
                                current_stop_sequences = ", ".join(team.members[i].stop_sequences) if i < num_existing_members and team.members[i].stop_sequences else ""
                                current_logprobs = team.members[i].logprobs if i < num_existing_members and team.members[i].logprobs is not None else False
                                current_top_logprobs = team.members[i].top_logprobs if i < num_existing_members and team.members[i].top_logprobs is not None else 0
                                current_response_format = json.dumps(team.members[i].response_format) if i < num_existing_members and team.members[i].response_format else ""
                                current_stream = team.members[i].stream if i < num_existing_members and team.members[i].stream is not None else False
                                current_user = team.members[i].user if i < num_existing_members and team.members[i].user else ""

                                edited_api_base = st.text_input(f"API Base (e.g., https://api.openai.com/v1)", value=current_api_base, key=f"edit_api_base_{team.name}_{i}")
                                edited_temperature = st.slider(f"Temperature", min_value=0.0, max_value=2.0, value=current_temperature, step=0.1, key=f"edit_temp_{team.name}_{i}")
                                edited_top_p = st.slider(f"Top P", min_value=0.0, max_value=1.0, value=current_top_p, step=0.01, key=f"edit_top_p_{team.name}_{i}")
                                edited_max_tokens = st.number_input(f"Max Tokens", min_value=1, value=current_max_tokens, key=f"edit_max_tokens_{team.name}_{i}")
                                edited_frequency_penalty = st.slider(f"Frequency Penalty", min_value=-2.0, max_value=2.0, value=current_frequency_penalty, step=0.01, key=f"edit_freq_penalty_{team.name}_{i}")
                                edited_presence_penalty = st.slider(f"Presence Penalty", min_value=-2.0, max_value=2.0, value=current_presence_penalty, step=0.01, key=f"edit_pres_penalty_{team.name}_{i}")
                                edited_seed = st.number_input(f"Seed (Optional)", value=current_seed, key=f"edit_seed_{team.name}_{i}")
                                edited_stop_sequences_str = st.text_input(f"Stop Sequences (comma-separated)", value=current_stop_sequences, key=f"edit_stop_sequences_{team.name}_{i}")
                                edited_logprobs = st.checkbox(f"Logprobs", value=current_logprobs, key=f"edit_logprobs_{team.name}_{i}")
                                edited_top_logprobs = st.number_input(f"Top Logprobs (0-5)", min_value=0, max_value=5, value=current_top_logprobs, key=f"edit_top_logprobs_{team.name}_{i}")
                                edited_response_format_str = st.text_input(f"Response Format (JSON string, e.g., '{{\"type\": \"json_object\"}}')", value=current_response_format, key=f"edit_response_format_{team.name}_{i}")
                                edited_stream = st.checkbox(f"Stream", value=current_stream, key=f"edit_stream_{team.name}_{i}")
                                edited_user = st.text_input(f"User ID", value=current_user, key=f"edit_user_{team.name}_{i}")
                                current_reasoning_effort = team.members[i].reasoning_effort if i < num_existing_members else None
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
                                updated_team = Team(name=edited_team_name, role=edited_team_role, members=edited_members, description=edited_team_description, content_analysis_system_prompt=edited_ca_system_prompt, content_analysis_user_prompt_template=edited_ca_user_prompt_template, decomposition_system_prompt=edited_decomp_system_prompt, decomposition_user_prompt_template=edited_decomp_user_prompt_template)
                                if team_manager.update_team(team.name, updated_team):
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
    """Renders the Streamlit UI for designing and managing Gauntlet definitions. Allows users to create, view, edit, and delete gauntlets."""
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
            
            st.subheader("Gauntlet Rounds")
            num_rounds = st.number_input("Number of Rounds", min_value=1, value=1, key="num_new_rounds")
            
            new_rounds = []
            for i in range(num_rounds):
                st.markdown(f"**Round {i+1} Configuration**")
                col1, col2 = st.columns(2)
                with col1:
                    quorum_req = st.number_input(f"Quorum: Required Approvals", min_value=1, value=1, key=f"round_{i}_quorum_req")
                with col2:
                    quorum_from = st.number_input(f"Quorum: From Panel Size", min_value=1, value=1, key=f"round_{i}_quorum_from")
                
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
                    per_judge_requirements=per_judge_reqs
                ))
            
            # Gauntlet specific settings (Red/Blue)
            st.subheader("Gauntlet Specific Settings")
            attack_modes_str = st.text_input("Red Team Attack Modes (comma-separated)", key="new_attack_modes")
            attack_modes = [m.strip() for m in attack_modes_str.split(',') if m.strip()]
            
            generation_mode = st.selectbox("Blue Team Generation Mode", ["single_candidate", "multi_candidate_peer_review"], key="new_gen_mode")

            submitted = st.form_submit_button("Create Gauntlet")
            if submitted:
                if gauntlet_name and team_for_gauntlet and new_rounds:
                    new_gauntlet = GauntletDefinition(
                        name=gauntlet_name,
                        team_name=team_for_gauntlet,
                        rounds=new_rounds,
                        description=gauntlet_description,
                        attack_modes=attack_modes,
                        generation_mode=generation_mode
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
                        if round_rule.per_judge_requirements:
                            st.json(round_rule.per_judge_requirements)
                    
                    st.write(f"Attack Modes: {', '.join(gauntlet.attack_modes) if gauntlet.attack_modes else 'N/A'}")
                    st.write(f"Generation Mode: {gauntlet.generation_mode}")

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
    """
    st.header("📝 Manual Review & Override")
    st.info("Review the AI-generated decomposition plan. You can edit any aspect of the plan before approving it.")

    # Use a session state object to hold edits, preventing loss on rerun.
    # This ensures that user modifications persist across Streamlit reruns until the plan is approved or rejected.
    if 'edited_sub_problems' not in st.session_state:
        st.session_state.edited_sub_problems = {sp.id: sp for sp in decomposition_plan.sub_problems}

    
    st.markdown(f"**Problem Statement**: {decomposition_plan.problem_statement}")
    st.markdown(f"**Analyzed Context Summary**: {decomposition_plan.analyzed_context.get('summary', 'N/A')}")

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

                st.caption("Specific Evolution Parameters (JSON format, optional)")
                edited_evolution_params_json = st.text_area("{}", value=json.dumps(current_sp_state.evolution_params, indent=2), key=f"evol_params_{sub_problem.id}")

                submitted = st.form_submit_button("Update Sub-Problem")
                if submitted:
                    try:
                        edited_evolution_params = json.loads(edited_evolution_params_json) if edited_evolution_params_json else {}
                        edited_dependencies = [d.strip() for d in edited_dependencies_str.split(',') if d.strip()]
                        
                        # Set to None if the selectbox was disabled (i.e., no options were available).
                        final_solver_team_name = edited_solver_team_name if blue_teams else None
                        final_red_gauntlet_name = edited_red_gauntlet_name if red_gauntlets else None
                        final_gold_gauntlet_name = edited_gold_gauntlet_name if gold_gauntlets else None

                        # Update the sub-problem in session state with the user's edits.
                        st.session_state.edited_sub_problems[sub_problem.id] = SubProblem(
                            id=sub_problem.id,
                            description=edited_description,
                            dependencies=edited_dependencies,
                            ai_suggested_evolution_mode=edited_ai_suggested_evolution_mode,
                            ai_suggested_complexity_score=edited_ai_suggested_complexity_score,
                            ai_suggested_evaluation_prompt=edited_ai_suggested_evaluation_prompt,
                            solver_team_name=final_solver_team_name,
                            red_team_gauntlet_name=final_red_gauntlet_name if final_red_gauntlet_name != "None" else None,
                            gold_team_gauntlet_name=final_gold_gauntlet_name,
                            evolution_params=edited_evolution_params
                        )
                        st.success(f"Sub-Problem {sub_problem.id} updated in draft plan.")
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON for evolution parameters in Sub-Problem {sub_problem.id}.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        # Button to approve the entire decomposition plan.
        if st.button("✅ Approve Plan", key="approve_plan_button", type="primary"):
            # Reconstruct the DecompositionPlan from the edited sub-problems in session state.
            final_sub_problems = list(st.session_state.edited_sub_problems.values())
            
            approved_plan = DecompositionPlan(
                problem_statement=decomposition_plan.problem_statement,
                analyzed_context=decomposition_plan.analyzed_context,
                sub_problems=final_sub_problems,
                max_refinement_loops=decomposition_plan.max_refinement_loops,
                assembler_team_name=decomposition_plan.assembler_team_name,
                final_red_team_gauntlet_name=decomposition_plan.final_red_team_gauntlet_name,
                final_gold_team_gauntlet_name=decomposition_plan.final_gold_team_gauntlet_name
            )
            del st.session_state.edited_sub_problems # Clean up session state after approval.
            return "approved", approved_plan
    with col2:
        # Button to reject the entire decomposition plan.
        if st.button("❌ Reject Plan", key="reject_plan_button"):
            st.error("Plan rejected. Please modify the initial problem or AI settings and try again.")
            del st.session_state.edited_sub_problems # Clean up session state after rejection.
            return "rejected", None
    
    return "pending", None # Plan not yet approved or rejected.