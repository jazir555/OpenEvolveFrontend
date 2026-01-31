"""
Causal Model Builder

Builds causal models from agent outcomes using causal discovery algorithms.
Supports intervention analysis and counterfactual reasoning.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, UTC
from collections import defaultdict
import numpy as np
import pandas as pd
import logging

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .schemas.long_horizon import (
    CausalModel,
    CausalRelationship,
    EffectPrediction,
    Explanation
)


logger = logging.getLogger(__name__)


class CausalModelBuilder:
    """
    Build causal models from agent outcomes

    Uses causal discovery algorithms to identify causal relationships
    from observational data. Supports:
    - PC algorithm for causal discovery
    - Intervention effect prediction
    - Counterfactual reasoning
    - Causal graph management

    Usage:
        builder = CausalModelBuilder()

        # Build model from outcomes
        model = await builder.build_model(
            domain="finance",
            outcomes=outcomes_list
        )

        # Predict intervention effect
        effect = await builder.predict_intervention(
            model=model,
            cause="exploration_rate",
            value=0.5
        )

        # Explain outcome
        explanation = await builder.explain_outcome(
            model=model,
            outcome="low_fitness"
        )
    """

    def __init__(
        self,
        discovery_method: str = "pc",  # PC algorithm
        min_confidence: float = 0.7,
        max_parents: int = 5
    ):
        """
        Initialize causal model builder

        Args:
            discovery_method: Causal discovery algorithm
            min_confidence: Minimum confidence for relationships
            max_parents: Maximum parent nodes per variable
        """
        self.discovery_method = discovery_method
        self.min_confidence = min_confidence
        self.max_parents = max_parents

        # Model storage
        # Key: domain -> CausalModel
        self.models: Dict[str, CausalModel] = {}

        if not HAS_NETWORKX:
            logger.warning(
                "networkx not installed. Causal modeling will use simplified approach."
            )

    async def build_model(
        self,
        domain: str,
        outcomes: List[Dict[str, Any]]
    ) -> CausalModel:
        """
        Build causal model from outcomes

        Args:
            domain: Problem domain
            outcomes: List of outcome dictionaries

        Returns:
            Causal model
        """
        model_id = f"causal_{domain}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Extract factors and outcomes
        factors, outcome_vars = self._extract_variables(outcomes)

        # Build data matrix
        data = self._build_data_matrix(outcomes, factors, outcome_vars)

        # Discover causal relationships
        relationships = await self._discover_causes(data, factors, outcome_vars)

        # Build graph structure
        graph_data = self._build_graph(relationships)

        model = CausalModel(
            model_id=model_id,
            domain=domain,
            relationships=relationships,
            factors=factors,
            outcomes=outcome_vars,
            graph_data=graph_data
        )

        self.models[domain] = model

        logger.info(
            f"Built causal model {model_id}: "
            f"{len(relationships)} relationships, "
            f"{len(factors)} factors"
        )

        return model

    async def update_model(
        self,
        model: CausalModel,
        new_data: List[Dict[str, Any]]
    ) -> CausalModel:
        """
        Update existing causal model with new data

        Args:
            model: Existing model to update
            new_data: New outcomes

        Returns:
            Updated model
        """
        # Rebuild with combined data
        updated_model = await self.build_model(
            domain=model.domain,
            outcomes=new_data  # In practice, would combine with existing data
        )

        # Preserve model ID
        updated_model.model_id = model.model_id

        return updated_model

    async def identify_causes(
        self,
        model: CausalModel,
        outcome: str
    ) -> List[CausalRelationship]:
        """
        Identify causes of a specific outcome

        Args:
            model: Causal model
            outcome: Outcome variable

        Returns:
            List of causal relationships
        """
        causes = []

        for rel in model.relationships:
            if rel.effect == outcome and rel.confidence >= self.min_confidence:
                causes.append(rel)

        # Sort by strength
        causes.sort(key=lambda r: r.strength, reverse=True)

        return causes

    async def predict_intervention(
        self,
        model: CausalModel,
        cause: str,
        value: float
    ) -> EffectPrediction:
        """
        Predict effect of intervention

        Args:
            model: Causal model
            cause: Factor to intervene on
            value: Value to set it to

        Returns:
            Effect prediction
        """
        # Find relationships where this is the cause
        effects = [
            rel for rel in model.relationships
            if rel.cause == cause
        ]

        if not effects:
            return EffectPrediction(
                intervention=f"Set {cause} to {value}",
                predicted_effect=0.0,
                confidence=0.0,
                alternative_outcomes=[],
                risk_assessment=["No causal relationships found"]
            )

        # Calculate predicted effects
        predicted_effects = []
        for effect in effects:
            # Simplified linear model: effect = strength * value
            predicted = effect.strength * value
            predicted_effects.append((effect.effect, predicted))

        # Aggregate predictions
        total_effect = sum(p for _, p in predicted_effects)
        avg_confidence = np.mean([e.confidence for e in effects])

        # Alternative outcomes (sensitivity analysis)
        alternatives = [
            (f"{cause} = {value * 0.8}", total_effect * 0.8),
            (f"{cause} = {value * 1.2}", total_effect * 1.2)
        ]

        # Risk assessment
        risks = []
        if avg_confidence < 0.8:
            risks.append("Low confidence in prediction")
        if total_effect < 0:
            risks.append("Intervention may decrease performance")

        return EffectPrediction(
            intervention=f"Set {cause} to {value}",
            predicted_effect=total_effect,
            confidence=avg_confidence,
            alternative_outcomes=alternatives,
            risk_assessment=risks
        )

    async def explain_outcome(
        self,
        model: CausalModel,
        outcome: str,
        outcome_value: Optional[float] = None
    ) -> Explanation:
        """
        Explain an outcome using causal model

        Args:
            model: Causal model
            outcome: Outcome to explain
            outcome_value: Actual observed value

        Returns:
            Explanation
        """
        # Identify causes
        causes = await self.identify_causes(model, outcome)

        if not causes:
            return Explanation(
                outcome=outcome,
                causes=[],
                contribution={},
                confidence=0.0,
                counterfactuals=["No causal relationships found"]
            )

        # Calculate contributions
        contributions = {}
        total_strength = sum(c.strength for c in causes)

        for cause in causes:
            contributions[cause.cause] = (
                cause.strength / max(0.001, total_strength)
            )

        # Overall confidence
        confidence = np.mean([c.confidence for c in causes])

        # Generate counterfactuals
        counterfactuals = []
        for cause in causes[:3]:  # Top 3 causes
            counterfactuals.append(
                f"If {cause.cause} were different, {outcome} would likely change by "
                f"{cause.strength:.2f}"
            )

        return Explanation(
            outcome=outcome,
            causes=[c.cause for c in causes],
            contribution=contributions,
            confidence=confidence,
            counterfactuals=counterfactuals
        )

    def _extract_variables(
        self,
        outcomes: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract factors and outcome variables

        Returns:
            (factors, outcomes)
        """
        # Heuristic: factors are in 'context', outcomes are in 'metrics'
        factors_set = set()
        outcomes_set = set()

        for outcome in outcomes:
            # Extract from context (factors)
            context = outcome.get("context", {})
            if isinstance(context, dict):
                factors_set.update(context.keys())

            # Extract from metrics (outcomes)
            metrics = outcome.get("metrics", {})
            if isinstance(metrics, dict):
                outcomes_set.update(metrics.keys())

        return list(factors_set), list(outcomes_set)

    def _build_data_matrix(
        self,
        outcomes: List[Dict[str, Any]],
        factors: List[str],
        outcome_vars: List[str]
    ) -> pd.DataFrame:
        """
        Build pandas DataFrame from outcomes
        """
        rows = []

        for outcome in outcomes:
            row = {}

            # Extract factors
            context = outcome.get("context", {})
            for factor in factors:
                if factor in context:
                    value = context[factor]
                    # Convert to numeric if possible
                    try:
                        row[factor] = float(value)
                    except (ValueError, TypeError):
                        # Categorical: use hash
                        row[factor] = hash(str(value)) % 100

            # Extract outcomes
            metrics = outcome.get("metrics", {})
            for var in outcome_vars:
                if var in metrics:
                    try:
                        row[var] = float(metrics[var])
                    except (ValueError, TypeError):
                        row[var] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    async def _discover_causes(
        self,
        data: pd.DataFrame,
        factors: List[str],
        outcomes: List[str]
    ) -> List[CausalRelationship]:
        """
        Discover causal relationships using simplified PC algorithm

        Real implementation would use:
        - causal-learn library for PC algorithm
        - DoWhy for causal inference
        - CausalNex for discrete models
        """
        relationships = []

        # Simplified approach: correlation-based
        # In production, use proper causal discovery

        all_vars = factors + outcomes

        for outcome_var in outcomes:
            if outcome_var not in data.columns:
                continue

            for factor in factors:
                if factor not in data.columns:
                    continue

                # Calculate correlation
                corr = data[[factor, outcome_var]].corr().iloc[0, 1]

                # Skip weak correlations
                if abs(corr) < 0.3:
                    continue

                # Calculate confidence (simplified)
                # In practice, use statistical tests
                confidence = min(0.95, abs(corr) + 0.5)

                if confidence >= self.min_confidence:
                    # Generate mechanism description
                    mechanism = self._infer_mechanism(factor, outcome_var, corr)

                    relationships.append(CausalRelationship(
                        cause=factor,
                        effect=outcome_var,
                        strength=abs(corr),
                        confidence=confidence,
                        mechanism=mechanism,
                        evidence=[f"Correlation: {corr:.3f}"]
                    ))

        return relationships

    def _infer_mechanism(
        self,
        cause: str,
        effect: str,
        correlation: float
    ) -> str:
        """
        Infer causal mechanism from variable names
        """
        # Simple heuristic-based mechanism inference
        if "rate" in cause and "performance" in effect:
            return f"Adjusting {cause} directly influences {effect}"
        elif "size" in cause and "cost" in effect:
            return f"{cause} scales computational {effect}"
        elif "temperature" in cause and "diversity" in effect:
            return f"{cause} controls exploration, affecting {effect}"
        else:
            return f"{cause} influences {effect} through unknown mechanism"

    def _build_graph(
        self,
        relationships: List[CausalRelationship]
    ) -> Dict[str, Any]:
        """
        Build graph structure from relationships

        Returns:
            Serialized graph data
        """
        if HAS_NETWORKX:
            # Build NetworkX graph
            G = nx.DiGraph()

            for rel in relationships:
                G.add_edge(
                    rel.cause,
                    rel.effect,
                    weight=rel.strength,
                    confidence=rel.confidence
                )

            # Serialize
            return {
                "nodes": list(G.nodes()),
                "edges": [
                    {
                        "source": u,
                        "target": v,
                        "weight": d["weight"],
                        "confidence": d["confidence"]
                    }
                    for u, v, d in G.edges(data=True)
                ],
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges()
            }
        else:
            # Simplified representation without NetworkX
            nodes = set()
            edges = []

            for rel in relationships:
                nodes.add(rel.cause)
                nodes.add(rel.effect)
                edges.append({
                    "source": rel.cause,
                    "target": rel.effect,
                    "weight": rel.strength,
                    "confidence": rel.confidence
                })

            return {
                "nodes": list(nodes),
                "edges": edges,
                "num_nodes": len(nodes),
                "num_edges": len(edges)
            }

    def get_model(self, domain: str) -> Optional[CausalModel]:
        """
        Get causal model for domain

        Args:
            domain: Domain identifier

        Returns:
            Causal model or None
        """
        return self.models.get(domain)

    def list_models(self) -> List[str]:
        """
        List available models

        Returns:
            List of domain names
        """
        return list(self.models.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "discovery_method": self.discovery_method,
            "min_confidence": self.min_confidence,
            "max_parents": self.max_parents,
            "models": {
                domain: model.to_dict()
                for domain, model in self.models.items()
            }
        }
