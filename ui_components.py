import streamlit as st
from typing import List, Dict, Any, Optional
from workflow_structures import Team, ModelConfig, GauntletDefinition, GauntletRoundRule, DecompositionPlan, SubProblem
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
import json
import time

# Initialize managers (can be done once in the main app or passed via session state)
if 'team_manager' not in st.session_state:
    st.session_state.team_manager = TeamManager()
if 'gauntlet_manager' not in st.session_state:
    st.session_state.gauntlet_manager = GauntletManager()

def render_team_manager():
    st.header("👥 Team Manager")
    st.write("Create, view, edit, and delete your AI teams.")

    team_manager: TeamManager = st.session_state.team_manager

    # --- Create New Team ---
    with st.expander("➕ Create New Team", expanded=False):
        with st.form("new_team_form"):
            team_name = st.text_input("Team Name", key="new_team_name")
            team_role = st.selectbox("Team Role", ["Blue", "Red", "Gold"], key="new_team_role")
            team_description = st.text_area("Description", key="new_team_description")

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
                
                # Add more model parameters as needed
                new_members.append(ModelConfig(model_id=model_id, api_key=api_key, api_base=api_base, temperature=temperature))
            
            submitted = st.form_submit_button("Create Team")
            if submitted:
                if team_name and new_members[0].model_id: # Basic validation
                    new_team = Team(name=team_name, role=team_role, members=new_members, description=team_description)
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
                
                with st.expander(f"View/Edit Members for {team.name}", expanded=False):
                    for i, member in enumerate(team.members):
                        st.markdown(f"**Model {i+1}**: `{member.model_id}`")
                        st.write(f"API Base: `{member.api_base}` | Temp: `{member.temperature}`")
                        # Display other params as needed
                    
                    # Edit functionality (simplified for now, full edit would be more complex)
                    if st.button(f"Delete Team '{team.name}'", key=f"delete_team_{team.name}"):
                        if team_manager.delete_team(team.name):
                            st.success(f"Team '{team.name}' deleted.")
                            st.session_state.team_manager = TeamManager() # Reload to refresh UI
                        else:
                            st.error(f"Failed to delete team '{team.name}'.")

def render_gauntlet_designer():
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
                
                # Per-judge requirements (simplified UI for now)
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

    # Use a session state object to hold edits, preventing loss on rerun
    if 'edited_sub_problems' not in st.session_state:
        st.session_state.edited_sub_problems = {sp.id: sp for sp in decomposition_plan.sub_problems}

    
    st.markdown(f"**Problem Statement**: {decomposition_plan.problem_statement}")
    st.markdown(f"**Analyzed Context Summary**: {decomposition_plan.analyzed_context.get('summary', 'N/A')}")

    st.subheader("Sub-Problems")
    for i, sub_problem in enumerate(decomposition_plan.sub_problems):
        with st.expander(f"Sub-Problem {sub_problem.id}: {sub_problem.description[:80]}...", expanded=False):
            # Each sub-problem is its own form to allow individual updates
            with st.form(f"edit_sub_problem_form_{sub_problem.id}"):
                current_sp_state = st.session_state.edited_sub_problems[sub_problem.id]
                
                edited_description = st.text_area("Description", value=current_sp_state.description, key=f"desc_{sub_problem.id}")
                edited_dependencies_str = st.text_input("Dependencies (comma-separated IDs)", value=", ".join(current_sp_state.dependencies), key=f"deps_{sub_problem.id}")
                
                st.markdown("---")
                st.markdown("**AI Suggestions (Editable)**")
                edited_ai_suggested_evolution_mode = st.text_input("Suggested Evolution Mode", value=current_sp_state.ai_suggested_evolution_mode, key=f"ai_mode_{sub_problem.id}")
                edited_ai_suggested_complexity_score = st.number_input("Suggested Complexity Score (1-10)", min_value=1, max_value=10, value=current_sp_state.ai_suggested_complexity_score, key=f"ai_comp_{sub_problem.id}")
                edited_ai_suggested_evaluation_prompt = st.text_area("Suggested Evaluation Prompt", value=current_sp_state.ai_suggested_evaluation_prompt, key=f"ai_eval_prompt_{sub_problem.id}")
                
                st.markdown("---")
                st.markdown("**User Overrides (Select Teams & Gauntlets)**")
                
                team_manager: TeamManager = st.session_state.team_manager
                gauntlet_manager: GauntletManager = st.session_state.gauntlet_manager
                
                blue_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Blue"]
                red_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Red"]
                gold_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Gold"]

                edited_solver_team_name = st.selectbox("Solver Team (Blue)", blue_teams, index=blue_teams.index(current_sp_state.solver_team_name) if current_sp_state.solver_team_name in blue_teams else 0, key=f"solver_team_{sub_problem.id}")
                edited_red_gauntlet_name = st.selectbox("Red Team Gauntlet", ["None"] + red_gauntlets, index=(red_gauntlets.index(current_sp_state.red_team_gauntlet_name) + 1) if current_sp_state.red_team_gauntlet_name in red_gauntlets else 0, key=f"red_gauntlet_{sub_problem.id}")
                edited_gold_gauntlet_name = st.selectbox("Gold Team Gauntlet", gold_gauntlets, index=gold_gauntlets.index(current_sp_state.gold_team_gauntlet_name) if current_sp_state.gold_team_gauntlet_name in gold_gauntlets else 0, key=f"gold_gauntlet_{sub_problem.id}")

                st.caption("Specific Evolution Parameters (JSON format, optional)")
                edited_evolution_params_json = st.text_area("{}", value=json.dumps(current_sp_state.evolution_params, indent=2), key=f"evol_params_{sub_problem.id}")

                submitted = st.form_submit_button("Update Sub-Problem")
                if submitted:
                    try:
                        edited_evolution_params = json.loads(edited_evolution_params_json) if edited_evolution_params_json else {}
                        edited_dependencies = [d.strip() for d in edited_dependencies_str.split(',') if d.strip()]
                        
                        # Update the sub-problem in session state
                        st.session_state.edited_sub_problems[sub_problem.id] = SubProblem(
                            id=sub_problem.id,
                            description=edited_description,
                            dependencies=edited_dependencies,
                            ai_suggested_evolution_mode=edited_ai_suggested_evolution_mode,
                            ai_suggested_complexity_score=edited_ai_suggested_complexity_score,
                            ai_suggested_evaluation_prompt=edited_ai_suggested_evaluation_prompt,
                            solver_team_name=edited_solver_team_name,
                            red_team_gauntlet_name=edited_red_gauntlet_name if edited_red_gauntlet_name != "None" else None,
                            gold_team_gauntlet_name=edited_gold_gauntlet_name,
                            evolution_params=edited_evolution_params
                        )
                        st.success(f"Sub-Problem {sub_problem.id} updated in draft plan.")
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON for evolution parameters in Sub-Problem {sub_problem.id}.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Plan", key="approve_plan_button", type="primary"):
            # Reconstruct the DecompositionPlan from the edited sub-problems in session state
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
            del st.session_state.edited_sub_problems # Clean up session state
            return "approved", approved_plan
    with col2:
        if st.button("❌ Reject Plan", key="reject_plan_button"):
            st.error("Plan rejected. Please modify the initial problem or AI settings and try again.")
            del st.session_state.edited_sub_problems # Clean up session state
            return "rejected", None
    
    return "pending", None # Plan not yet approved or rejected