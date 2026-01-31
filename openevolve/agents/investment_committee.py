#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Investment Committee Agent

A sophisticated long-horizon workflow that performs weekly portfolio reviews,
continuously learns from outcomes, adapts investment strategies over time,
and uses LoongFlow for multi-stage reasoning.

This agent orchestrates the investment decision-making process through:
- Portfolio review and analysis
- Hypothesis generation and testing
- Adversarial challenge of recommendations
- Mathematical verification of decisions
- Continuous learning from outcomes
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

# LoongFlow imports
try:
    from loongflow.framework.pes.pes_agent import PESAgent
    from loongflow.framework.pes.context import EvolveChainConfig
    from loongflow.framework.pes.database.database import EvolveDatabase
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False
    logging.warning("LoongFlow not available. Investment Committee will run in standalone mode.")

from openevolve.agents.investment.rlm_decomposer import RLMDecomposer
from openevolve.agents.investment.roma_tester import ROMATester
from openevolve.agents.investment.adversarial_tester import AdversarialTester
from openevolve.agents.investment.math_verifier import MathVerifier
from openevolve.agents.investment.knowledge_integrator import KnowledgeIntegrator


class PortfolioState:
    """Maintains the current state of the investment portfolio."""

    def __init__(
        self,
        holdings: Dict[str, float],
        cash: float,
        last_rebalance: Optional[datetime] = None,
        total_value: Optional[float] = None
    ):
        self.holdings = holdings  # ticker -> shares
        self.cash = cash
        self.last_rebalance = last_rebalance or datetime.utcnow()
        self.total_value = total_value or cash

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio state to dictionary for serialization."""
        return {
            "holdings": self.holdings,
            "cash": self.cash,
            "last_rebalance": self.last_rebalance.isoformat(),
            "total_value": self.total_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioState":
        """Create PortfolioState from dictionary."""
        return cls(
            holdings=data["holdings"],
            cash=data["cash"],
            last_rebalance=datetime.fromisoformat(data["last_rebalance"]),
            total_value=data.get("total_value")
        )


class InvestmentDecision:
    """Represents a single investment decision with its reasoning and outcomes."""

    def __init__(
        self,
        decision_id: str,
        timestamp: datetime,
        decision_type: str,  # "rebalance", "hold", "analyze"
        actions: List[Dict[str, Any]],
        reasoning: str,
        confidence: float,
        expected_outcome: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.decision_id = decision_id
        self.timestamp = timestamp
        self.decision_type = decision_type
        self.actions = actions  # List of {"ticker": str, "action": "buy"/"sell", "shares": float}
        self.reasoning = reasoning
        self.confidence = confidence  # 0.0 to 1.0
        self.expected_outcome = expected_outcome
        self.metadata = metadata or {}

        # Outcome tracking (filled in later)
        self.actual_outcome: Optional[str] = None
        self.outcome_timestamp: Optional[datetime] = None
        self.performance_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type,
            "actions": self.actions,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata,
            "actual_outcome": self.actual_outcome,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "performance_metrics": self.performance_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestmentDecision":
        """Create InvestmentDecision from dictionary."""
        decision = cls(
            decision_id=data["decision_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            decision_type=data["decision_type"],
            actions=data["actions"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            expected_outcome=data["expected_outcome"],
            metadata=data.get("metadata")
        )
        decision.actual_outcome = data.get("actual_outcome")
        decision.outcome_timestamp = datetime.fromisoformat(data["outcome_timestamp"]) if data.get("outcome_timestamp") else None
        decision.performance_metrics = data.get("performance_metrics")
        return decision


class InvestmentCommitteeAgent:
    """
    Autonomous Investment Committee Agent

    Orchestrates weekly portfolio review cycle, maintains portfolio state over time,
    tracks decisions and outcomes, learns from performance feedback, and generates
    investment recommendations using multi-stage reasoning.
    """

    def __init__(
        self,
        portfolio_state: PortfolioState,
        market_data_provider: Any,  # Provider for market data
        database_path: Optional[Path] = None,
        loongflow_config: Optional[EvolveChainConfig] = None,
        risk_tolerance: float = 0.15,  # Max portfolio volatility target
        max_position_size: float = 0.20,  # Max % of portfolio in single position
        rebalance_threshold: float = 0.05,  # Trigger rebalance if drift > 5%
        review_frequency_days: int = 7,  # Weekly reviews
        enable_loongflow: bool = True
    ):
        """
        Initialize the Investment Committee Agent.

        Args:
            portfolio_state: Initial portfolio state
            market_data_provider: Provider for historical and current market data
            database_path: Path to persistent storage for decisions and learnings
            loongflow_config: Optional LoongFlow configuration for advanced reasoning
            risk_tolerance: Target maximum portfolio volatility
            max_position_size: Maximum allocation to single position
            rebalance_threshold: Deviation threshold to trigger rebalancing
            review_frequency_days: Days between portfolio reviews
            enable_loongflow: Whether to use LoongFlow for multi-stage reasoning
        """
        self.portfolio = portfolio_state
        self.market_data = market_data_provider
        self.database_path = database_path or Path("./investment_committee_db")
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold
        self.review_frequency = timedelta(days=review_frequency_days)
        self.enable_loongflow = enable_loongflow and LOONGFLOW_AVAILABLE

        # Initialize components
        self.rlm_decomposer = RLMDecomposer()
        self.roma_tester = ROMATester(market_data_provider)
        self.adversarial_tester = AdversarialTester()
        self.math_verifier = MathVerifier()
        self.knowledge_integrator = KnowledgeIntegrator(self.database_path)

        # Decision history
        self.decisions: List[InvestmentDecision] = []
        self.last_review = self.portfolio.last_rebalance

        # LoongFlow agent (if enabled)
        self.loongflow_agent: Optional[PESAgent] = None
        if self.enable_loongflow and loongflow_config:
            self._init_loongflow(loongflow_config)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load previous decisions if database exists
        self._load_state()

    def _init_loongflow(self, config: EvolveChainConfig):
        """Initialize LoongFlow PES agent for multi-stage reasoning."""
        try:
            self.loongflow_agent = PESAgent(config=config)
            self.logger.info("LoongFlow agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LoongFlow: {e}")
            self.enable_loongflow = False

    def _load_state(self):
        """Load previous decisions and knowledge from database."""
        try:
            decisions_file = self.database_path / "decisions.json"
            if decisions_file.exists():
                with open(decisions_file, "r") as f:
                    decisions_data = json.load(f)
                    self.decisions = [
                        InvestmentDecision.from_dict(d) for d in decisions_data
                    ]
                self.logger.info(f"Loaded {len(self.decisions)} previous decisions")

            # Load knowledge graph
            self.knowledge_integrator.load_knowledge()

        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def _save_state(self):
        """Save current state to database."""
        try:
            self.database_path.mkdir(parents=True, exist_ok=True)

            # Save decisions
            decisions_file = self.database_path / "decisions.json"
            with open(decisions_file, "w") as f:
                json.dump([d.to_dict() for d in self.decisions], f, indent=2)

            # Save knowledge
            self.knowledge_integrator.save_knowledge()

        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    def should_review(self) -> bool:
        """Check if it's time for a portfolio review."""
        time_since_review = datetime.utcnow() - self.last_review
        return time_since_review >= self.review_frequency

    async def weekly_review_cycle(self) -> InvestmentDecision:
        """
        Execute a complete weekly portfolio review cycle.

        This is the main workflow that orchestrates all components:

        1. Review Phase: Gather data and identify changes
        2. Analysis Phase: RLM decomposition, ROMA testing, adversarial challenge
        3. Decision Phase: Synthesize findings into recommendations
        4. Learning Phase: Track outcomes and update knowledge

        Returns:
            InvestmentDecision: The final investment decision with reasoning
        """
        self.logger.info(f"Starting weekly review cycle for {datetime.utcnow().isoformat()}")

        try:
            # Phase 1: Review - Gather portfolio and market data
            review_data = await self._review_phase()

            # Phase 2: Analysis - Deep dive with multiple reasoning methods
            analysis_results = await self._analysis_phase(review_data)

            # Phase 3: Decision - Synthesize into actionable recommendation
            decision = await self._decision_phase(review_data, analysis_results)

            # Phase 4: Learning - Extract knowledge for future improvement
            await self._learning_phase(decision)

            # Update state
            self.decisions.append(decision)
            self.last_review = datetime.utcnow()
            self._save_state()

            return decision

        except Exception as e:
            self.logger.error(f"Error in weekly review cycle: {e}", exc_info=True)
            raise

    async def _review_phase(self) -> Dict[str, Any]:
        """
        Review Phase: Gather portfolio state, market data, and identify changes.

        Returns:
            Dictionary containing portfolio state, market data, and changes
        """
        self.logger.info("Review Phase: Gathering data")

        # Get current portfolio value and composition
        portfolio_value = self.portfolio.total_value
        current_allocations = self._calculate_allocations()

        # Get market data for holdings
        market_context = await self.market_data.get_current_state(list(self.portfolio.holdings.keys()))

        # Identify significant changes since last review
        changes = self._identify_changes(market_context)

        # Retrieve relevant historical context
        historical_context = await self.knowledge_integrator.retrieve_similar_scenarios(changes)

        return {
            "portfolio_value": portfolio_value,
            "allocations": current_allocations,
            "market_context": market_context,
            "changes": changes,
            "historical_context": historical_context,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _analysis_phase(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysis Phase: Apply multiple reasoning methods to analyze the situation.

        Args:
            review_data: Data gathered from review phase

        Returns:
            Dictionary containing analysis results from all methods
        """
        self.logger.info("Analysis Phase: Multi-method analysis")

        results = {}

        # RLM Decomposition: Break down the problem
        self.logger.info("  - RLM Decomposition")
        results["rlm_decomposition"] = await self.rlm_decomposer.decompose(
            portfolio_state=review_data["allocations"],
            market_context=review_data["market_context"],
            changes=review_data["changes"]
        )

        # ROMA Testing: Test investment hypotheses
        self.logger.info("  - ROMA Hypothesis Testing")
        results["roma_tests"] = await self.roma_tester.test_hypotheses(
            hypotheses=results["rlm_decomposition"]["hypotheses"],
            historical_data=await self.market_data.get_historical_data(
                list(self.portfolio.holdings.keys()),
                period="1y"
            )
        )

        # Adversarial Challenge: Stress test the recommendations
        self.logger.info("  - Adversarial Testing")
        results["adversarial_analysis"] = await self.adversarial_tester.challenge_recommendations(
            recommendations=results["roma_tests"]["recommendations"],
            portfolio_state=review_data["allocations"]
        )

        # Mathematical Verification: Validate the math
        self.logger.info("  - Mathematical Verification")
        results["math_verification"] = await self.math_verifier.verify_decision(
            recommendations=results["roma_tests"]["recommendations"],
            current_portfolio=review_data["allocations"],
            constraints={
                "max_position_size": self.max_position_size,
                "risk_tolerance": self.risk_tolerance
            }
        )

        # Use LoongFlow for advanced synthesis if enabled
        if self.enable_loongflow and self.loongflow_agent:
            self.logger.info("  - LoongFlow Synthesis")
            results["loongflow_synthesis"] = await self._loongflow_synthesis(results)

        return results

    async def _decision_phase(
        self,
        review_data: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        Decision Phase: Synthesize all analysis into a final investment decision.

        Args:
            review_data: Data from review phase
            analysis_results: Results from all analysis methods

        Returns:
            InvestmentDecision: Final decision with reasoning and confidence
        """
        self.logger.info("Decision Phase: Synthesizing recommendations")

        # Check if rebalancing is needed
        needs_rebalance = self._check_rebalance_needed(
            review_data["allocations"],
            analysis_results.get("math_verification", {})
        )

        if not needs_rebalance:
            # Decision to hold
            return InvestmentDecision(
                decision_id=f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.utcnow(),
                decision_type="hold",
                actions=[],
                reasoning="No rebalancing needed. Portfolio within target allocation bands.",
                confidence=0.85,
                expected_outcome="Portfolio continues to track target allocations",
                metadata={
                    "review_data": review_data,
                    "analysis_summary": self._summarize_analysis(analysis_results)
                }
            )

        # Synthesize recommendations from all sources
        if self.enable_loongflow and "loongflow_synthesis" in analysis_results:
            # Use LoongFlow synthesis
            synthesis = analysis_results["loongflow_synthesis"]
            actions = synthesis["recommended_actions"]
            reasoning = synthesis["reasoning"]
            confidence = synthesis["confidence"]

        else:
            # Use default synthesis logic
            actions = self._synthesis_default(analysis_results)
            reasoning = self._generate_reasoning(actions, analysis_results)
            confidence = self._calculate_confidence(actions, analysis_results)

        # Create decision
        decision = InvestmentDecision(
            decision_id=f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            decision_type="rebalance",
            actions=actions,
            reasoning=reasoning,
            confidence=confidence,
            expected_outcome=self._predict_outcome(actions, review_data),
            metadata={
                "review_data": review_data,
                "analysis_results": analysis_results
            }
        )

        return decision

    async def _learning_phase(self, decision: InvestmentDecision):
        """
        Learning Phase: Extract knowledge from the decision cycle.

        Args:
            decision: The decision that was made
        """
        self.logger.info("Learning Phase: Extracting knowledge")

        # Extract causal factors from the decision context
        await self.knowledge_integrator.extract_causal_knowledge(decision)

        # Update prediction heuristics based on historical patterns
        await self.knowledge_integrator.update_heuristics(self.decisions)

        # Identify what factors actually predict outcomes
        await self.knowledge_integrator.analyze_predictive_factors(self.decisions)

    async def _loongflow_synthesis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LoongFlow to synthesize recommendations from multiple analysis methods.

        Args:
            analysis_results: Results from all analysis methods

        Returns:
            Synthesis with recommended actions, reasoning, and confidence
        """
        # This is a placeholder. In production, you'd configure LoongFlow
        # with proper prompts and tasks for investment synthesis.
        # For now, return default synthesis.
        return {
            "recommended_actions": self._synthesis_default(analysis_results),
            "reasoning": "Synthesis based on RLM decomposition, ROMA testing, and adversarial analysis",
            "confidence": 0.75
        }

    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate current portfolio allocations as percentages."""
        if self.portfolio.total_value == 0:
            return {}

        allocations = {}
        for ticker, shares in self.portfolio.holdings.items():
            # Get current price (would need market data provider)
            # For now, assume equal weight
            allocations[ticker] = (shares * 100) / self.portfolio.total_value

        return allocations

    def _identify_changes(self, market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify significant changes since last review."""
        changes = []

        # Check for price movements beyond threshold
        # Check for changes in market conditions
        # Check for corporate actions

        return changes

    def _check_rebalance_needed(
        self,
        allocations: Dict[str, float],
        math_verification: Dict[str, Any]
    ) -> bool:
        """Check if portfolio needs rebalancing."""
        # Check if any position has drifted beyond threshold
        for ticker, allocation in allocations.items():
            target = 0.10  # Example target allocation
            if abs(allocation - target) > self.rebalance_threshold:
                return True

        return False

    def _synthesis_default(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default synthesis logic when LoongFlow is not available."""
        actions = []

        # Start with ROMA recommendations
        if "roma_tests" in analysis_results:
            for rec in analysis_results["roma_tests"].get("recommendations", []):
                actions.append({
                    "ticker": rec["ticker"],
                    "action": rec.get("action", "hold"),
                    "shares": rec.get("shares", 0),
                    "rationale": rec.get("rationale", "")
                })

        return actions

    def _generate_reasoning(
        self,
        actions: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate reasoning text for the decision."""
        reasoning_parts = []

        # RLM insights
        if "rlm_decomposition" in analysis_results:
            rlm = analysis_results["rlm_decomposition"]
            reasoning_parts.append(f"Key Factors: {', '.join(rlm.get('key_factors', []))}")

        # ROMA test results
        if "roma_tests" in analysis_results:
            roma = analysis_results["roma_tests"]
            reasoning_parts.append(f"Hypothesis Confidence: {roma.get('avg_confidence', 0):.2%}")

        # Adversarial concerns
        if "adversarial_analysis" in analysis_results:
            adversarial = analysis_results["adversarial_analysis"]
            if adversarial.get("concerns"):
                reasoning_parts.append(f"Key Concerns: {'; '.join(adversarial['concerns'][:3])}")

        return " | ".join(reasoning_parts)

    def _calculate_confidence(
        self,
        actions: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the decision."""
        confidences = []

        # ROMA confidence
        if "roma_tests" in analysis_results:
            confidences.append(analysis_results["roma_tests"].get("avg_confidence", 0.5))

        # Math verification pass rate
        if "math_verification" in analysis_results:
            math_verif = analysis_results["math_verification"]
            if math_verif.get("all_passed"):
                confidences.append(0.9)
            else:
                confidences.append(0.6)

        # Adversarial severity (inverse)
        if "adversarial_analysis" in analysis_results:
            adversarial = analysis_results["adversarial_analysis"]
            severity = adversarial.get("severity_score", 0.5)
            confidences.append(1.0 - severity)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _predict_outcome(
        self,
        actions: List[Dict[str, Any]],
        review_data: Dict[str, Any]
    ) -> str:
        """Predict the expected outcome of this decision."""
        if not actions:
            return "Portfolio maintains current allocations"

        action_types = [a.get("action", "hold") for a in actions]
        if "buy" in action_types or "sell" in action_types:
            return f"Rebalance to target allocations with {len(actions)} trades"

        return "Portfolio adjustments implemented"

    def _summarize_analysis(self, analysis_results: Dict[str, Any]) -> str:
        """Create a brief summary of the analysis results."""
        summary_parts = []

        if "rlm_decomposition" in analysis_results:
            rlm = analysis_results["rlm_decomposition"]
            summary_parts.append(f"RLM identified {len(rlm.get('key_factors', []))} key factors")

        if "roma_tests" in analysis_results:
            roma = analysis_results["roma_tests"]
            summary_parts.append(f"ROMA tested {len(roma.get('hypotheses_tested', []))} hypotheses")

        if "adversarial_analysis" in analysis_results:
            adversarial = analysis_results["adversarial_analysis"]
            summary_parts.append(f"Adversarial found {len(adversarial.get('concerns', []))} concerns")

        return "; ".join(summary_parts)

    async def record_outcome(
        self,
        decision_id: str,
        actual_outcome: str,
        performance_metrics: Dict[str, float]
    ):
        """
        Record the actual outcome of a previous decision.

        Args:
            decision_id: ID of the decision to update
            actual_outcome: Description of what actually happened
            performance_metrics: Metrics like return, volatility, etc.
        """
        # Find the decision
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)

        if decision:
            decision.actual_outcome = actual_outcome
            decision.outcome_timestamp = datetime.utcnow()
            decision.performance_metrics = performance_metrics

            # Learn from this outcome
            await self.knowledge_integrator.learn_from_outcome(decision)

            # Save updated state
            self._save_state()

        else:
            self.logger.warning(f"Decision {decision_id} not found")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent's performance over time."""
        if not self.decisions:
            return {"total_decisions": 0}

        # Calculate metrics
        total_decisions = len(self.decisions)
        decisions_with_outcomes = [d for d in self.decisions if d.actual_outcome]

        avg_confidence = sum(d.confidence for d in self.decisions) / total_decisions

        # Calculate accuracy if we have outcomes
        accuracy = None
        if decisions_with_outcomes:
            correct = sum(
                1 for d in decisions_with_outcomes
                if "positive" in d.actual_outcome.lower() or "gain" in d.actual_outcome.lower()
            )
            accuracy = correct / len(decisions_with_outcomes)

        return {
            "total_decisions": total_decisions,
            "decisions_with_outcomes": len(decisions_with_outcomes),
            "average_confidence": avg_confidence,
            "accuracy": accuracy,
            "last_review": self.last_review.isoformat() if self.last_review else None
        }
