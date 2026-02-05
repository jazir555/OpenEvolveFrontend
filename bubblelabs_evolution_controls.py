"""
Evolution Control UI Components for BubbleLabs

Provides specialized UI components for:
- Evolution parameter controls
- Population visualization
- Fitness landscape plotting
- Real-time evolution monitoring

Author: OpenEvolve Frontend Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EvolutionControlState:
    """State for evolution controls"""
    # Population parameters
    population_size: int = 20
    selection_method: str = "tournament"
    tournament_size: int = 3
    elitism_count: int = 2

    # Genetic operators
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    mutation_strength: float = 0.2

    # Evolution parameters
    max_generations: int = 100
    convergence_threshold: float = 0.001
    diversity_threshold: float = 0.3

    # Advanced features
    enable_maker_voting: bool = False
    voting_threshold: int = 3
    enable_mdap_decomposition: bool = False
    decomposition_depth: int = 3


class EvolutionControlPanel:
    """
    Streamlit component for evolution parameter controls.

    Provides organized controls for all evolution parameters with
    real-time validation and preset configurations.
    """

    def __init__(self):
        self.presets = {
            "default": EvolutionControlState(),
            "fast": EvolutionControlState(
                population_size=10,
                max_generations=50,
                mutation_rate=0.15,
                convergence_threshold=0.01
            ),
            "thorough": EvolutionControlState(
                population_size=50,
                max_generations=200,
                tournament_size=5,
                elitism_count=5,
                convergence_threshold=0.0001
            ),
            "maker_voting": EvolutionControlState(
                population_size=20,
                enable_maker_voting=True,
                voting_threshold=3,
                enable_mdap_decomposition=False
            ),
            "mdap_decomposition": EvolutionControlState(
                population_size=30,
                enable_mdap_decomposition=True,
                decomposition_depth=5,
                max_generations=150
            ),
            "hybrid": EvolutionControlState(
                population_size=25,
                enable_maker_voting=True,
                voting_threshold=3,
                enable_mdap_decomposition=True,
                decomposition_depth=3,
                max_generations=120
            )
        }

    def render(self, key_prefix: str = "evo_ctrl") -> EvolutionControlState:
        """
        Render the evolution control panel.

        Args:
            key_prefix: Prefix for Streamlit keys to avoid conflicts

        Returns:
            EvolutionControlState with current parameter values
        """
        st.markdown("### 🎛️ Evolution Controls")

        # Preset selection
        preset_options = list(self.presets.keys())
        preset_labels = {
            "default": "Default",
            "fast": "Fast (Quick Exploration)",
            "thorough": "Thorough (Deep Search)",
            "maker_voting": "MAKER Voting",
            "mdap_decomposition": "MDAP Decomposition",
            "hybrid": "Hybrid (MAKER + MDAP)"
        }

        selected_preset = st.selectbox(
            "Configuration Preset",
            options=preset_options,
            format_func=lambda x: preset_labels.get(x, x),
            key=f"{key_prefix}_preset"
        )

        # Load preset
        state = EvolutionControlState(**asdict(self.presets[selected_preset]))

        # Create tabs for different parameter categories
        tabs = st.tabs([
            "Population",
            "Genetic Operators",
            "Evolution",
            "Advanced"
        ])

        with tabs[0]:
            state = self._render_population_controls(state, key_prefix)

        with tabs[1]:
            state = self._render_genetic_operators(state, key_prefix)

        with tabs[2]:
            state = self._render_evolution_parameters(state, key_prefix)

        with tabs[3]:
            state = self._render_advanced_features(state, key_prefix)

        # Validation
        self._validate_parameters(state)

        return state

    def _render_population_controls(
        self,
        state: EvolutionControlState,
        key_prefix: str
    ) -> EvolutionControlState:
        """Render population-related controls"""
        st.markdown("#### 👥 Population Parameters")

        col1, col2 = st.columns(2)

        with col1:
            state.population_size = st.number_input(
                "Population Size",
                min_value=2,
                max_value=500,
                value=state.population_size,
                help="Number of individuals in each generation",
                key=f"{key_prefix}_pop_size"
            )

            state.selection_method = st.selectbox(
                "Selection Method",
                options=["tournament", "roulette", "rank", "steady_state"],
                index=0,
                help="Method for selecting parents",
                key=f"{key_prefix}_selection"
            )

        with col2:
            if state.selection_method == "tournament":
                state.tournament_size = st.number_input(
                    "Tournament Size",
                    min_value=2,
                    max_value=20,
                    value=state.tournament_size,
                    help="Number of individuals in each tournament",
                    key=f"{key_prefix}_tournament"
                )

            state.elitism_count = st.number_input(
                "Elitism Count",
                min_value=0,
                max_value=20,
                value=state.elitism_count,
                help="Number of top individuals to preserve",
                key=f"{key_prefix}_elitism"
            )

        return state

    def _render_genetic_operators(
        self,
        state: EvolutionControlState,
        key_prefix: str
    ) -> EvolutionControlState:
        """Render genetic operator controls"""
        st.markdown("#### 🧬 Genetic Operators")

        col1, col2 = st.columns(2)

        with col1:
            state.mutation_rate = st.slider(
                "Mutation Rate",
                min_value=0.0,
                max_value=1.0,
                value=state.mutation_rate,
                step=0.01,
                help="Probability of mutation for each individual",
                key=f"{key_prefix}_mutation_rate"
            )

            state.mutation_strength = st.slider(
                "Mutation Strength",
                min_value=0.0,
                max_value=1.0,
                value=state.mutation_strength,
                step=0.05,
                help="Magnitude of mutations when applied",
                key=f"{key_prefix}_mutation_strength"
            )

        with col2:
            state.crossover_rate = st.slider(
                "Crossover Rate",
                min_value=0.0,
                max_value=1.0,
                value=state.crossover_rate,
                step=0.01,
                help="Probability of crossover between parents",
                key=f"{key_prefix}_crossover_rate"
            )

            # Mutation type
            mutation_type = st.selectbox(
                "Mutation Type",
                options=["point", "gaussian", "uniform", "adaptive"],
                index=0,
                key=f"{key_prefix}_mutation_type"
            )

        return state

    def _render_evolution_parameters(
        self,
        state: EvolutionControlState,
        key_prefix: str
    ) -> EvolutionControlState:
        """Render evolution parameter controls"""
        st.markdown("#### 🔄 Evolution Parameters")

        col1, col2 = st.columns(2)

        with col1:
            state.max_generations = st.number_input(
                "Max Generations",
                min_value=1,
                max_value=1000,
                value=state.max_generations,
                help="Maximum number of generations to run",
                key=f"{key_prefix}_max_gen"
            )

            state.convergence_threshold = st.number_input(
                "Convergence Threshold",
                min_value=0.0,
                max_value=0.1,
                value=state.convergence_threshold,
                step=0.0001,
                format="%.4f",
                help="Stop when fitness improvement is below this threshold",
                key=f"{key_prefix}_convergence"
            )

        with col2:
            state.diversity_threshold = st.slider(
                "Diversity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=state.diversity_threshold,
                step=0.05,
                help="Minimum population diversity to maintain",
                key=f"{key_prefix}_diversity"
            )

            # Stopping conditions
            stopping_condition = st.selectbox(
                "Stopping Condition",
                options=["generations", "convergence", "both"],
                index=2,
                help="When to stop evolution",
                key=f"{key_prefix}_stopping"
            )

        return state

    def _render_advanced_features(
        self,
        state: EvolutionControlState,
        key_prefix: str
    ) -> EvolutionControlState:
        """Render advanced feature controls"""
        st.markdown("#### 🚀 Advanced Features")

        # MAKER Voting
        st.markdown("**MAKER Voting**")
        state.enable_maker_voting = st.checkbox(
            "Enable MAKER Voting",
            value=state.enable_maker_voting,
            help="Use first-to-ahead-by-k voting for selection",
            key=f"{key_prefix}_maker_voting"
        )

        if state.enable_maker_voting:
            col1, col2 = st.columns(2)
            with col1:
                state.voting_threshold = st.number_input(
                    "Voting Threshold (k)",
                    min_value=1,
                    max_value=10,
                    value=state.voting_threshold,
                    help="Consensus threshold for voting",
                    key=f"{key_prefix}_voting_k"
                )

            with col2:
                adaptive_voting = st.checkbox(
                    "Adaptive Voting",
                    value=True,
                    help="Adjust threshold based on diversity",
                    key=f"{key_prefix}_adaptive_voting"
                )

        # MDAP Decomposition
        st.markdown("**MDAP Decomposition**")
        state.enable_mdap_decomposition = st.checkbox(
            "Enable MDAP Decomposition",
            value=state.enable_mdap_decomposition,
            help="Decompose evolution task into subtasks",
            key=f"{key_prefix}_mdap_decomp"
        )

        if state.enable_mdap_decomposition:
            col1, col2 = st.columns(2)
            with col1:
                state.decomposition_depth = st.number_input(
                    "Decomposition Depth",
                    min_value=1,
                    max_value=10,
                    value=state.decomposition_depth,
                    help="Maximum depth for task decomposition",
                    key=f"{key_prefix}_decomp_depth"
                )

            with col2:
                max_subtasks = st.number_input(
                    "Max Subtasks",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Maximum number of subtasks to create",
                    key=f"{key_prefix}_max_subtasks"
                )

        return state

    def _validate_parameters(self, state: EvolutionControlState):
        """Validate evolution parameters and show warnings"""
        warnings = []

        # Check population size
        if state.population_size < 4:
            warnings.append("[WARN] Small population size may lead to premature convergence")

        # Check elitism
        if state.elitism_count >= state.population_size // 2:
            warnings.append("[WARN] High elitism count may reduce diversity")

        # Check mutation rate
        if state.mutation_rate < 0.01:
            warnings.append("[WARN] Very low mutation rate may limit exploration")
        elif state.mutation_rate > 0.5:
            warnings.append("[WARN] Very high mutation rate may disrupt good solutions")

        # Check voting threshold
        if state.enable_maker_voting and state.voting_threshold > state.population_size // 3:
            warnings.append("[WARN] Voting threshold may be too high for population size")

        if warnings:
            st.warning("\n".join(warnings))


class PopulationVisualizer:
    """
    Component for visualizing population state and diversity.
    """

    def render_population_overview(
        self,
        population_size: int,
        current_generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float
    ):
        """Render population overview metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Population Size", population_size)

        with col2:
            st.metric("Generation", current_generation)

        with col3:
            st.metric("Best Fitness", f"{best_fitness:.4f}")

        with col4:
            st.metric("Diversity", f"{diversity:.3f}")

    def render_fitness_distribution(
        self,
        fitness_values: List[float],
        generation: int
    ):
        """Render histogram of fitness values"""
        if not fitness_values:
            st.info("No fitness data available")
            return

        fig = go.Figure(data=[go.Histogram(
            x=fitness_values,
            nbinsx=20,
            marker_color='lightblue',
            name='Fitness Distribution'
        )])

        fig.update_layout(
            title=f"Fitness Distribution - Generation {generation}",
            xaxis_title="Fitness",
            yaxis_title="Count",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_diversity_heatmap(
        self,
        population_data: List[str],
        generation: int
    ):
        """Render heatmap showing pairwise distances between individuals"""
        if len(population_data) < 2:
            st.info("Need at least 2 individuals for diversity visualization")
            return

        # Calculate pairwise distances
        n = len(population_data)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Simple edit distance
                s1, s2 = population_data[i], population_data[j]
                max_len = max(len(s1), len(s2))
                if max_len == 0:
                    dist = 0.0
                else:
                    dist = sum(c1 != c2 for c1, c2 in zip(s1, s2)) / max_len
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=[f"Ind {i}" for i in range(n)],
            y=[f"Ind {i}" for i in range(n)],
            colorscale='Viridis',
            colorbar=dict(title="Distance")
        ))

        fig.update_layout(
            title=f"Population Diversity - Generation {generation}",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_fitness_landscape_3d(
        self,
        generations: List[int],
        best_fitness: List[float],
        avg_fitness: List[float]
    ):
        """Render 3D surface plot of fitness landscape"""
        if len(generations) < 3:
            st.info("Need at least 3 generations for landscape visualization")
            return

        # Create meshgrid for 3D plot
        gen_array = np.array(generations)
        fitness_array = np.array(best_fitness)

        # Create 3D surface
        fig = go.Figure(data=[go.Scatter3d(
            x=gen_array,
            y=fitness_array,
            z=avg_fitness,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=gen_array,
                colorscale='Viridis'
            ),
            line=dict(color='darkblue', width=4)
        )])

        fig.update_layout(
            title="Fitness Landscape Evolution",
            scene=dict(
                xaxis_title="Generation",
                yaxis_title="Best Fitness",
                zaxis_title="Average Fitness"
            ),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)


class AdversarialControlPanel:
    """
    Control panel for adversarial testing parameters.
    """

    def __init__(self):
        self.presets = {
            "light": {
                "rounds": 3,
                "red_team_size": 2,
                "blue_team_size": 2,
                "attack_strength": 0.3
            },
            "standard": {
                "rounds": 5,
                "red_team_size": 3,
                "blue_team_size": 3,
                "attack_strength": 0.5
            },
            "thorough": {
                "rounds": 10,
                "red_team_size": 5,
                "blue_team_size": 5,
                "attack_strength": 0.7
            },
            "maker_enhanced": {
                "rounds": 5,
                "red_team_size": 3,
                "blue_team_size": 3,
                "attack_strength": 0.5,
                "enable_maker_voting": True,
                "voting_threshold": 3,
                "enable_mdap_defense": True
            }
        }

    def render(self, key_prefix: str = "adv_ctrl") -> Dict[str, Any]:
        """Render adversarial control panel"""
        st.markdown("### ⚔️ Adversarial Testing Controls")

        # Preset selection
        preset = st.selectbox(
            "Configuration Preset",
            options=list(self.presets.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
            key=f"{key_prefix}_preset"
        )

        config = self.presets[preset].copy()

        # Custom configuration
        with st.expander("Custom Configuration", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                config["rounds"] = st.number_input(
                    "Adversarial Rounds",
                    min_value=1,
                    max_value=20,
                    value=config["rounds"],
                    key=f"{key_prefix}_rounds"
                )

                config["red_team_size"] = st.number_input(
                    "Red Team Size",
                    min_value=1,
                    max_value=10,
                    value=config["red_team_size"],
                    key=f"{key_prefix}_red_size"
                )

            with col2:
                config["blue_team_size"] = st.number_input(
                    "Blue Team Size",
                    min_value=1,
                    max_value=10,
                    value=config["blue_team_size"],
                    key=f"{key_prefix}_blue_size"
                )

                config["attack_strength"] = st.slider(
                    "Attack Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=config["attack_strength"],
                    step=0.1,
                    key=f"{key_prefix}_strength"
                )

            # Advanced options
            config["coevolution"] = st.checkbox(
                "Enable Coevolution",
                value=False,
                help="Attack and defense evolve together",
                key=f"{key_prefix}_coevolution"
            )

            config["enable_maker_voting"] = st.checkbox(
                "Enable MAKER Red Team Voting",
                value=config.get("enable_maker_voting", False),
                key=f"{key_prefix}_maker_voting"
            )

            config["enable_mdap_defense"] = st.checkbox(
                "Enable MDAP Blue Team Decomposition",
                value=config.get("enable_mdap_defense", False),
                key=f"{key_prefix}_mdap_defense"
            )

        return config

    def render_adversarial_results(
        self,
        vulnerabilities: List[Dict[str, Any]],
        fixes: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ):
        """Render adversarial testing results"""
        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Vulnerabilities Found", len(vulnerabilities))

        with col2:
            st.metric("Fixes Applied", len(fixes))

        with col3:
            st.metric("Attack Success Rate", f"{metrics.get('attack_success_rate', 0) * 100:.1f}%")

        with col4:
            st.metric("Defense Success Rate", f"{metrics.get('defense_success_rate', 0) * 100:.1f}%")

        # Vulnerability details
        if vulnerabilities:
            st.markdown("### 🔴 Vulnerabilities Found")

            for i, vuln in enumerate(vulnerabilities[:10]):  # Show top 10
                with st.expander(f"{i+1}. {vuln.get('title', 'Untitled')}", expanded=False):
                    st.markdown(f"**Severity:** {vuln.get('severity', 'N/A')}")
                    st.markdown(f"**Category:** {vuln.get('category', 'N/A')}")
                    st.markdown(f"**Description:** {vuln.get('description', 'N/A')}")
                    if vuln.get('recommendation'):
                        st.markdown(f"**Recommendation:** {vuln['recommendation']}")

        # Fix details
        if fixes:
            st.markdown("### 🔵 Fixes Applied")

            for i, fix in enumerate(fixes[:10]):  # Show top 10
                with st.expander(f"{i+1}. {fix.get('title', 'Untitled')}", expanded=False):
                    st.markdown(f"**Type:** {fix.get('type', 'N/A')}")
                    st.markdown(f"**Description:** {fix.get('description', 'N/A')}")
                    if fix.get('code_changes'):
                        st.code(fix['code_changes'], language="python")
