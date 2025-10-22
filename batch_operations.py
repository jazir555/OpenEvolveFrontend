"""
Batch Operations Module

This module provides batch operation functionality for managing multiple sub-problems
simultaneously in the decomposition workflow.
"""

from typing import List, Dict, Any, Optional
from workflow_structures import DecompositionPlan, SubProblem


class BatchOperations:
    """Handles batch operations on sub-problems."""
    
    @staticmethod
    def batch_assign_team(
        sub_problems: List[SubProblem],
        team_name: str,
        team_type: str = "solver"
    ) -> List[SubProblem]:
        """
        Assign a team to multiple sub-problems at once.
        
        Args:
            sub_problems: List of sub-problems to update
            team_name: Name of the team to assign
            team_type: Type of team ("solver" or "patcher")
            
        Returns:
            Updated list of sub-problems
        """
        updated_sub_problems = []
        
        for sp in sub_problems:
            if team_type == "solver":
                sp.solver_team_name = team_name
            elif team_type == "patcher":
                sp.patcher_team_name = team_name
            
            updated_sub_problems.append(sp)
        
        return updated_sub_problems
    
    @staticmethod
    def batch_assign_gauntlet(
        sub_problems: List[SubProblem],
        gauntlet_name: str,
        gauntlet_type: str = "gold"
    ) -> List[SubProblem]:
        """
        Assign a gauntlet to multiple sub-problems at once.
        
        Args:
            sub_problems: List of sub-problems to update
            gauntlet_name: Name of the gauntlet to assign
            gauntlet_type: Type of gauntlet ("red" or "gold")
            
        Returns:
            Updated list of sub-problems
        """
        updated_sub_problems = []
        
        for sp in sub_problems:
            if gauntlet_type == "red":
                sp.red_team_gauntlet_name = gauntlet_name
            elif gauntlet_type == "gold":
                sp.gold_team_gauntlet_name = gauntlet_name
            
            updated_sub_problems.append(sp)
        
        return updated_sub_problems
    
    @staticmethod
    def batch_update_parameters(
        sub_problems: List[SubProblem],
        parameters: Dict[str, Any]
    ) -> List[SubProblem]:
        """
        Update parameters for multiple sub-problems at once.
        
        Args:
            sub_problems: List of sub-problems to update
            parameters: Dictionary of parameters to update
                       Can include: evolution_mode, complexity_score, content_type, etc.
            
        Returns:
            Updated list of sub-problems
        """
        updated_sub_problems = []
        
        for sp in sub_problems:
            # Update evolution mode if specified
            if "evolution_mode" in parameters:
                sp.ai_suggested_evolution_mode = parameters["evolution_mode"]
            
            # Update complexity score if specified
            if "complexity_score" in parameters:
                sp.ai_suggested_complexity_score = parameters["complexity_score"]
            
            # Update content type if specified
            if "content_type" in parameters:
                sp.content_type = parameters["content_type"]
            
            # Update evolution parameters if specified
            if "evolution_params" in parameters:
                sp.evolution_params.update(parameters["evolution_params"])
            
            updated_sub_problems.append(sp)
        
        return updated_sub_problems
    
    @staticmethod
    def batch_add_dependency(
        sub_problems: List[SubProblem],
        dependency_id: str
    ) -> List[SubProblem]:
        """
        Add a dependency to multiple sub-problems at once.
        
        Args:
            sub_problems: List of sub-problems to update
            dependency_id: ID of the dependency to add
            
        Returns:
            Updated list of sub-problems
        """
        updated_sub_problems = []
        
        for sp in sub_problems:
            if dependency_id not in sp.dependencies:
                sp.dependencies.append(dependency_id)
            
            updated_sub_problems.append(sp)
        
        return updated_sub_problems
    
    @staticmethod
    def batch_remove_dependency(
        sub_problems: List[SubProblem],
        dependency_id: str
    ) -> List[SubProblem]:
        """
        Remove a dependency from multiple sub-problems at once.
        
        Args:
            sub_problems: List of sub-problems to update
            dependency_id: ID of the dependency to remove
            
        Returns:
            Updated list of sub-problems
        """
        updated_sub_problems = []
        
        for sp in sub_problems:
            if dependency_id in sp.dependencies:
                sp.dependencies.remove(dependency_id)
            
            updated_sub_problems.append(sp)
        
        return updated_sub_problems
    
    @staticmethod
    def filter_sub_problems(
        sub_problems: List[SubProblem],
        filters: Dict[str, Any]
    ) -> List[SubProblem]:
        """
        Filter sub-problems based on criteria.
        
        Args:
            sub_problems: List of sub-problems to filter
            filters: Dictionary of filter criteria
                    Can include: complexity_range, evolution_mode, has_team, has_gauntlet, etc.
            
        Returns:
            Filtered list of sub-problems
        """
        filtered = sub_problems.copy()
        
        # Filter by complexity range
        if "min_complexity" in filters:
            filtered = [sp for sp in filtered if sp.ai_suggested_complexity_score >= filters["min_complexity"]]
        
        if "max_complexity" in filters:
            filtered = [sp for sp in filtered if sp.ai_suggested_complexity_score <= filters["max_complexity"]]
        
        # Filter by evolution mode
        if "evolution_mode" in filters:
            filtered = [sp for sp in filtered if sp.ai_suggested_evolution_mode == filters["evolution_mode"]]
        
        # Filter by team assignment status
        if "has_solver_team" in filters:
            if filters["has_solver_team"]:
                filtered = [sp for sp in filtered if sp.solver_team_name]
            else:
                filtered = [sp for sp in filtered if not sp.solver_team_name]
        
        # Filter by gauntlet assignment status
        if "has_gold_gauntlet" in filters:
            if filters["has_gold_gauntlet"]:
                filtered = [sp for sp in filtered if sp.gold_team_gauntlet_name]
            else:
                filtered = [sp for sp in filtered if not sp.gold_team_gauntlet_name]
        
        # Filter by content type
        if "content_type" in filters:
            filtered = [sp for sp in filtered if sp.content_type == filters["content_type"]]
        
        # Filter by status
        if "status" in filters:
            filtered = [sp for sp in filtered if sp.status == filters["status"]]
        
        return filtered
    
    @staticmethod
    def get_batch_statistics(sub_problems: List[SubProblem]) -> Dict[str, Any]:
        """
        Get statistics for a batch of sub-problems.
        
        Args:
            sub_problems: List of sub-problems to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not sub_problems:
            return {
                "count": 0,
                "avg_complexity": 0,
                "evolution_modes": {},
                "content_types": {},
                "with_solver_team": 0,
                "with_red_gauntlet": 0,
                "with_gold_gauntlet": 0,
                "with_dependencies": 0
            }
        
        # Calculate statistics
        avg_complexity = sum(sp.ai_suggested_complexity_score for sp in sub_problems) / len(sub_problems)
        
        evolution_modes = {}
        content_types = {}
        
        for sp in sub_problems:
            evolution_modes[sp.ai_suggested_evolution_mode] = evolution_modes.get(sp.ai_suggested_evolution_mode, 0) + 1
            content_types[sp.content_type] = content_types.get(sp.content_type, 0) + 1
        
        with_solver_team = sum(1 for sp in sub_problems if sp.solver_team_name)
        with_red_gauntlet = sum(1 for sp in sub_problems if sp.red_team_gauntlet_name)
        with_gold_gauntlet = sum(1 for sp in sub_problems if sp.gold_team_gauntlet_name)
        with_dependencies = sum(1 for sp in sub_problems if sp.dependencies)
        
        return {
            "count": len(sub_problems),
            "avg_complexity": avg_complexity,
            "evolution_modes": evolution_modes,
            "content_types": content_types,
            "with_solver_team": with_solver_team,
            "with_red_gauntlet": with_red_gauntlet,
            "with_gold_gauntlet": with_gold_gauntlet,
            "with_dependencies": with_dependencies
        }


def render_batch_operations_ui(sub_problems: Dict[str, SubProblem]) -> Dict[str, SubProblem]:
    """
    Render batch operations UI in Streamlit.
    
    Args:
        sub_problems: Dictionary of sub-problem ID to SubProblem
        
    Returns:
        Updated dictionary of sub-problems
    """
    import streamlit as st
    from team_manager import TeamManager
    from gauntlet_manager import GauntletManager
    
    st.subheader("🔄 Batch Operations")
    st.write("Apply changes to multiple sub-problems at once.")
    
    # Get managers
    team_manager: TeamManager = st.session_state.team_manager
    gauntlet_manager: GauntletManager = st.session_state.gauntlet_manager
    
    # Selection interface
    st.write("**Select Sub-Problems:**")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_by_complexity = st.checkbox("Filter by Complexity")
        if filter_by_complexity:
            min_complexity = st.slider("Min Complexity", 1, 10, 1, key="batch_min_complexity")
            max_complexity = st.slider("Max Complexity", 1, 10, 10, key="batch_max_complexity")
    
    with col2:
        filter_by_team = st.checkbox("Filter by Team Status")
        if filter_by_team:
            team_filter = st.selectbox(
                "Team Status",
                ["Has Solver Team", "Missing Solver Team"],
                key="batch_team_filter"
            )
    
    with col3:
        filter_by_gauntlet = st.checkbox("Filter by Gauntlet Status")
        if filter_by_gauntlet:
            gauntlet_filter = st.selectbox(
                "Gauntlet Status",
                ["Has Gold Gauntlet", "Missing Gold Gauntlet"],
                key="batch_gauntlet_filter"
            )
    
    # Apply filters
    filtered_sub_problems = list(sub_problems.values())
    
    if filter_by_complexity:
        filters = {"min_complexity": min_complexity, "max_complexity": max_complexity}
        filtered_sub_problems = BatchOperations.filter_sub_problems(filtered_sub_problems, filters)
    
    if filter_by_team:
        filters = {"has_solver_team": team_filter == "Has Solver Team"}
        filtered_sub_problems = BatchOperations.filter_sub_problems(filtered_sub_problems, filters)
    
    if filter_by_gauntlet:
        filters = {"has_gold_gauntlet": gauntlet_filter == "Has Gold Gauntlet"}
        filtered_sub_problems = BatchOperations.filter_sub_problems(filtered_sub_problems, filters)
    
    # Show filtered count
    st.write(f"Filtered: **{len(filtered_sub_problems)}** sub-problems")
    
    # Manual selection
    if filtered_sub_problems:
        selected_ids = st.multiselect(
            "Select specific sub-problems (or leave empty to apply to all filtered)",
            [sp.id for sp in filtered_sub_problems],
            key="batch_selected_ids"
        )
        
        # Determine which sub-problems to operate on
        if selected_ids:
            target_sub_problems = [sp for sp in filtered_sub_problems if sp.id in selected_ids]
        else:
            target_sub_problems = filtered_sub_problems
        
        st.write(f"Target: **{len(target_sub_problems)}** sub-problems")
        
        # Show statistics
        with st.expander("Selection Statistics"):
            stats = BatchOperations.get_batch_statistics(target_sub_problems)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Count", stats["count"])
                st.metric("Avg Complexity", f"{stats['avg_complexity']:.1f}")
            
            with col2:
                st.write("**Evolution Modes:**")
                for mode, count in stats["evolution_modes"].items():
                    st.write(f"- {mode}: {count}")
            
            with col3:
                st.write("**Assignments:**")
                st.write(f"- With Solver Team: {stats['with_solver_team']}")
                st.write(f"- With Red Gauntlet: {stats['with_red_gauntlet']}")
                st.write(f"- With Gold Gauntlet: {stats['with_gold_gauntlet']}")
        
        st.divider()
        
        # Batch operations
        st.write("**Batch Operations:**")
        
        operation = st.selectbox(
            "Select Operation",
            [
                "Assign Solver Team",
                "Assign Red Gauntlet",
                "Assign Gold Gauntlet",
                "Update Evolution Mode",
                "Update Complexity Score",
                "Update Content Type"
            ],
            key="batch_operation"
        )
        
        # Operation-specific inputs
        if operation == "Assign Solver Team":
            blue_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Blue"]
            if blue_teams:
                team_name = st.selectbox("Select Team", blue_teams, key="batch_team_name")
                
                if st.button("Apply to Selected", key="batch_apply_team"):
                    updated = BatchOperations.batch_assign_team(target_sub_problems, team_name, "solver")
                    for sp in updated:
                        sub_problems[sp.id] = sp
                    st.success(f"Assigned team '{team_name}' to {len(updated)} sub-problems!")
                    st.rerun()
            else:
                st.warning("No Blue teams available. Create teams first.")
        
        elif operation == "Assign Red Gauntlet":
            red_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() 
                           if gauntlet_manager.get_gauntlet(g.name) and 
                           team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and 
                           team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Red"]
            if red_gauntlets:
                gauntlet_name = st.selectbox("Select Gauntlet", red_gauntlets, key="batch_red_gauntlet_name")
                
                if st.button("Apply to Selected", key="batch_apply_red_gauntlet"):
                    updated = BatchOperations.batch_assign_gauntlet(target_sub_problems, gauntlet_name, "red")
                    for sp in updated:
                        sub_problems[sp.id] = sp
                    st.success(f"Assigned gauntlet '{gauntlet_name}' to {len(updated)} sub-problems!")
                    st.rerun()
            else:
                st.warning("No Red gauntlets available. Create gauntlets first.")
        
        elif operation == "Assign Gold Gauntlet":
            gold_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() 
                            if gauntlet_manager.get_gauntlet(g.name) and 
                            team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and 
                            team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Gold"]
            if gold_gauntlets:
                gauntlet_name = st.selectbox("Select Gauntlet", gold_gauntlets, key="batch_gold_gauntlet_name")
                
                if st.button("Apply to Selected", key="batch_apply_gold_gauntlet"):
                    updated = BatchOperations.batch_assign_gauntlet(target_sub_problems, gauntlet_name, "gold")
                    for sp in updated:
                        sub_problems[sp.id] = sp
                    st.success(f"Assigned gauntlet '{gauntlet_name}' to {len(updated)} sub-problems!")
                    st.rerun()
            else:
                st.warning("No Gold gauntlets available. Create gauntlets first.")
        
        elif operation == "Update Evolution Mode":
            evolution_mode = st.selectbox(
                "Select Evolution Mode",
                ["standard", "adversarial", "quality_diversity", "multi_objective"],
                key="batch_evolution_mode"
            )
            
            if st.button("Apply to Selected", key="batch_apply_evolution_mode"):
                updated = BatchOperations.batch_update_parameters(
                    target_sub_problems,
                    {"evolution_mode": evolution_mode}
                )
                for sp in updated:
                    sub_problems[sp.id] = sp
                st.success(f"Updated evolution mode to '{evolution_mode}' for {len(updated)} sub-problems!")
                st.rerun()
        
        elif operation == "Update Complexity Score":
            complexity_score = st.slider("Complexity Score", 1, 10, 5, key="batch_complexity_score")
            
            if st.button("Apply to Selected", key="batch_apply_complexity"):
                updated = BatchOperations.batch_update_parameters(
                    target_sub_problems,
                    {"complexity_score": complexity_score}
                )
                for sp in updated:
                    sub_problems[sp.id] = sp
                st.success(f"Updated complexity score to {complexity_score} for {len(updated)} sub-problems!")
                st.rerun()
        
        elif operation == "Update Content Type":
            content_type = st.selectbox(
                "Select Content Type",
                ["text_general", "code_python", "code_javascript", "document_legal", 
                 "document_medical", "document_technical", "prompt", "protocol"],
                key="batch_content_type"
            )
            
            if st.button("Apply to Selected", key="batch_apply_content_type"):
                updated = BatchOperations.batch_update_parameters(
                    target_sub_problems,
                    {"content_type": content_type}
                )
                for sp in updated:
                    sub_problems[sp.id] = sp
                st.success(f"Updated content type to '{content_type}' for {len(updated)} sub-problems!")
                st.rerun()
    else:
        st.info("No sub-problems match the current filters.")
    
    return sub_problems


def batch_evolve_solutions(problems: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Batch evolve solutions for multiple problems"""
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        results = []
        
        for problem in problems:
            result = client.evolve(
                content=problem,
                evolution_mode="standard",
                max_iterations=10,
                population_size=20
            )
            results.append(result)
        
        return results
    except Exception as e:
        return [{'error': str(e)} for _ in problems]
