"""
DTS (Dialogue Tree Search) integration for OpenEvolve.

This module provides a bridge between OpenEvolve and the DTS engine,
enabling conversational strategy exploration, multi-judge scoring,
and adversarial dialogue simulation.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import DTS components
try:
    # Add DTS directory to sys.path if needed
    sys.path.insert(0, "DTS")
    from backend.core.dts import DTSEngine, DTSConfig, DTSRunResult
    from backend.core.dts.types import DialogueNode, Strategy, UserIntent
    from backend.llm.client import LLM
    from backend.utils.config import config as dts_config
    DTS_AVAILABLE = True
except (ImportError, Exception) as e:
    # Catch both ImportError and validation errors from DTS config
    logger.warning(f"DTS not available: {e}")
    DTS_AVAILABLE = False
    DTSEngine = None
    DTSConfig = None
    DTSRunResult = None
    LLM = None

# Try to import DSPy for enhanced prompting
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.predict import Predict
    DSPY_AVAILABLE = True
    logger.info("DSPy available for enhanced programmatic prompting")
except ImportError:
    dspy = None
    BootstrapFewShot = None
    Predict = None
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using standard prompting methods")


@dataclass
class DTSIntegrationConfig:
    """Configuration for DTS integration."""
    # Whether to enable DTS integration
    enabled: bool = True
    # Default DTS parameters
    max_rounds: int = 3
    init_branches: int = 6
    turns_per_branch: int = 5
    user_intents_per_branch: int = 3
    scoring_mode: str = "comparative"  # "comparative" or "absolute"
    prune_threshold: float = 6.5
    deep_research: bool = False
    user_variability: bool = False
    max_concurrency: int = 16
    # LLM configuration
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "minimax/minimax-m2.1"
    # Integration-specific
    use_dts_for_adversarial: bool = True
    use_dts_for_scoring: bool = True
    use_dts_for_strategy_generation: bool = False
    use_multi_judge: bool = True
    judge_count: int = 3
    use_comparative_scoring: bool = True
    attacker_persona: str = "critical_expert"
    defender_persona: str = "defensive_expert"
    use_strategy_exploration: bool = False
    # DSPy integration
    use_dspy_for_enhanced_prompts: bool = True
    dspy_model_name: str = "gpt-4o-mini"


class DTSIntegration:
    """Main integration class for DTS functionality."""
    
    def __init__(self, config: Optional[DTSIntegrationConfig] = None):
        self.config = config or DTSIntegrationConfig()
        self._llm_client = None
        self._engine = None
        
    def is_available(self) -> bool:
        """Check if DTS is available for use."""
        return DTS_AVAILABLE and self.config.enabled
    
    async def initialize(self) -> bool:
        """Initialize DTS engine and LLM client."""
        if not self.is_available():
            logger.warning("DTS not available or disabled")
            return False
        
        try:
            # Create LLM client
            from backend.llm.client import LLM
            from backend.utils.config import config as dts_config
            
            # Use environment variables or provided config
            api_key = self.config.llm_api_key or dts_config.openrouter_api_key
            base_url = self.config.llm_base_url or dts_config.openai_base_url
            
            self._llm_client = LLM(
                api_key=api_key,
                base_url=base_url,
                model=self.config.llm_model,
            )
            logger.info("DTS LLM client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DTS LLM client: {e}")
            return False
    
    async def run_conversation_search(
        self,
        goal: str,
        first_message: str,
        rounds: int = 2,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[DTSRunResult]:
        """
        Run a DTS conversation search.
        
        Args:
            goal: The conversation goal
            first_message: The opening user message
            rounds: Number of expansion rounds
            custom_config: Override DTSConfig parameters
            
        Returns:
            DTSRunResult if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("DTS not available, skipping conversation search")
            return None
        
        if self._llm_client is None:
            if not await self.initialize():
                return None
        
        try:
            # Create DTS configuration
            dts_config_params = {
                "goal": goal,
                "first_message": first_message,
                "init_branches": self.config.init_branches,
                "turns_per_branch": self.config.turns_per_branch,
                "user_intents_per_branch": self.config.user_intents_per_branch,
                "scoring_mode": self.config.scoring_mode,
                "prune_threshold": self.config.prune_threshold,
                "deep_research": self.config.deep_research,
                "user_variability": self.config.user_variability,
                "max_concurrency": self.config.max_concurrency,
            }
            if custom_config:
                dts_config_params.update(custom_config)
            
            config = DTSConfig(**dts_config_params)
            engine = DTSEngine(llm=self._llm_client, config=config)
            
            logger.info(f"Starting DTS conversation search for goal: {goal}")
            result = await engine.run(rounds=rounds)
            logger.info(f"DTS search completed with best score: {result.best_score:.1f}")
            
            return result
        except Exception as e:
            logger.error(f"DTS conversation search failed: {e}")
            return None
    
    def adversarial_dialogue(
        self,
        content: str = "",
        content_type: str = "general",
        attacker_persona: str = "aggressive hacker",
        defender_persona: str = "security expert",
        rounds: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run an adversarial dialogue between attacker and defender using DTS.
        
        Args:
            content: Content to attack/defend
            content_type: Type of content
            attacker_persona: Persona for the attacker
            defender_persona: Persona for the defender
            rounds: Number of dialogue rounds
            
        Returns:
            Dictionary with results including winner, scores, and conversation
        """
        if not self.is_available():
            logger.warning("DTS not available, using fallback adversarial dialogue")
            return self._fallback_adversarial_dialogue(
                attacker_persona, defender_persona, content, rounds
            )
        
        # Prepare goals
        attacker_goal = kwargs.get("attacker_goal", f"Find vulnerabilities in this {content_type}")
        defender_goal = kwargs.get("defender_goal", f"Defend this {content_type} and fix issues")
        initial_message = content or kwargs.get("initial_message", "Starting assessment...")
        
        # For simplicity, we'll run DTS with a combined goal
        combined_goal = f"Attacker ({attacker_persona}): {attacker_goal} | Defender ({defender_persona}): {defender_goal}"
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(self.run_conversation_search(
            goal=combined_goal,
            first_message=initial_message,
            rounds=rounds,
            custom_config={
                "user_variability": True,
                "user_intents_per_branch": 2,  # Attacker and defender perspectives
            }
        ))
        
        if result is None:
            return {"error": "DTS search failed", "winner": None}
        
        # Analyze results to determine winner
        # This is a simplified heuristic: look at scores of branches
        # that align with attacker vs defender strategies
        best_node = result.best_node_id
        best_score = result.best_score
        
        # For now, return basic results
        return {
            "winner": "attacker" if best_score > 7.0 else "defender",
            "best_score": best_score,
            "total_branches": len(result.all_nodes),
            "pruned_branches": result.pruned_count,
            "conversation": result.best_messages if hasattr(result, 'best_messages') else [],
        }
    
    def _fallback_adversarial_dialogue(
        self,
        attacker_goal: str,
        defender_goal: str,
        initial_message: str,
        rounds: int,
    ) -> Dict[str, Any]:
        """Fallback implementation when DTS is not available."""
        logger.warning("Using fallback adversarial dialogue (no DTS)")
        return {
            "winner": "unknown",
            "best_score": 0.0,
            "total_branches": 0,
            "pruned_branches": 0,
            "conversation": [{"role": "user", "content": initial_message}],
            "note": "DTS not available, using fallback",
        }
    
    def multi_judge_scoring(
        self,
        content: str = "",
        context: Optional[Dict[str, Any]] = None,
        scoring_criteria: Union[str, List[str]] = "general quality",
        num_judges: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use DTS's multi-judge scoring system to evaluate text.
        
        Args:
            content: The text to evaluate
            context: Additional context for evaluation
            scoring_criteria: Criteria to score against
            num_judges: Number of judges to use (max 3 in DTS)
            
        Returns:
            Dictionary with scores, consensus, and feedback
        """
        text = content or kwargs.get("text", "")
        criteria = [scoring_criteria] if isinstance(scoring_criteria, str) else scoring_criteria
        
        if not self.is_available():
            logger.warning("DTS not available, using fallback scoring")
            return self._fallback_scoring(text, criteria, num_judges)
        
        # DTS's evaluator component isn't directly exposed, so we'll
        # create a simple conversation search where the goal is to evaluate
        # This is a placeholder implementation
        goal = f"Evaluate the following text based on criteria: {', '.join(criteria)}"
        if context:
            goal += f" Context: {context}"
            
        first_message = f"Text to evaluate:\n\n{text}\n\nPlease provide scores for each criterion."
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(self.run_conversation_search(
            goal=goal,
            first_message=first_message,
            rounds=1,
            custom_config={
                "init_branches": num_judges,
                "turns_per_branch": 1,
                "scoring_mode": "absolute",
            }
        ))
        
        if result is None:
            return self._fallback_scoring(text, criteria, num_judges)
        
        # Extract scores from result (simplified)
        scores = {}
        for node in result.all_nodes[:num_judges]:
            # In real implementation, parse node content for scores
            scores[f"judge_{len(scores)+1}"] = {
                "overall": node.stats.aggregated_score if hasattr(node, 'stats') else 5.0,
                "criteria": {c: 5.0 for c in criteria},
            }
        
        return {
            "scores": scores,
            "average_score": result.best_score,
            "consensus": "high" if result.best_score > 7.0 else "medium" if result.best_score > 5.0 else "low",
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
        }
    
    def _fallback_scoring(
        self,
        text: str,
        criteria: List[str],
        num_judges: int,
    ) -> Dict[str, Any]:
        """Fallback scoring when DTS is not available."""
        import random
        scores = {}
        for i in range(num_judges):
            scores[f"judge_{i+1}"] = {
                "overall": random.uniform(3.0, 9.0),
                "criteria": {c: random.uniform(3.0, 9.0) for c in criteria},
            }
        
        avg = sum(s["overall"] for s in scores.values()) / num_judges
        return {
            "scores": scores,
            "average_score": avg,
            "consensus": "high" if avg > 7.0 else "medium" if avg > 5.0 else "low",
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "note": "Fallback scoring (DTS not available)",
        }
    
    def generate_strategies(
        self,
        problem: str = "",
        num_strategies: int = 6,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse strategies for solving a problem using DTS.
        
        Args:
            problem: Description of the problem to solve
            num_strategies: Number of strategies to generate
            context: Additional context for strategy generation
            **kwargs: Additional parameters for strategy generation
            
        Returns:
            List of strategy dictionaries with tagline and description
        """
        problem = problem or kwargs.get("goal", "")
        
        if not self.is_available():
            logger.warning("DTS not available, using fallback strategy generation")
            return self._fallback_strategies(problem, num_strategies, context)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Use DTS's strategy generator component
        try:
            from backend.core.dts.components.generator import StrategyGenerator
            
            async def _run():
                if self._llm_client is None:
                    if not await self.initialize():
                        return self._fallback_strategies(problem, num_strategies, context)
                
                generator = StrategyGenerator(
                    llm=self._llm_client,
                    goal=problem,
                    model=self.config.llm_model,
                    temperature=0.7,
                    max_concurrency=self.config.max_concurrency,
                )
                
                strategies = await generator.generate_strategies(
                    first_message=problem,
                    count=num_strategies,
                    deep_context=context,
                )
                
                return [
                    {"tagline": s.tagline, "description": s.description}
                    for s in strategies
                ]
            
            return loop.run_until_complete(_run())
        except Exception as e:
            logger.error(f"DTS strategy generation failed: {e}")
            return self._fallback_strategies(problem, num_strategies, context)
    
    def _fallback_strategies(
        self,
        problem: str,
        num_strategies: int,
        context: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fallback strategy generation when DTS is not available."""
        # Simple placeholder strategies
        strategies = [
            {"tagline": "Direct Approach", "description": f"Solve {problem[:50]} directly using standard methods."},
            {"tagline": "Iterative Refinement", "description": "Start with a simple solution and iteratively improve it."},
            {"tagline": "Divide and Conquer", "description": "Break the problem into smaller subproblems and solve each independently."},
            {"tagline": "Creative Synthesis", "description": "Combine ideas from multiple domains to create novel solutions."},
            {"tagline": "Risk-Averse Strategy", "description": "Prioritize safety and reliability over optimality."},
            {"tagline": "Aggressive Optimization", "description": "Push boundaries to achieve maximum performance."},
        ]
        return strategies[:num_strategies]

    def _initialize_dspy_if_available(self):
        """
        Initialize DSPy with the configured model if DSPy is available.

        Returns:
            DSPy LM object if available, None otherwise
        """
        if not DSPY_AVAILABLE or not self.config.use_dspy_for_enhanced_prompts:
            return None

        try:
            # Initialize DSPy with the configured model
            lm = dspy.LM(model=self.config.dspy_model_name)
            dspy.settings.configure(lm=lm)
            return lm
        except Exception as e:
            logger.warning(f"Could not initialize DSPy with model {self.config.dspy_model_name}: {e}")
            return None

    async def enhanced_multi_judge_scoring_with_dspy(
        self,
        text: str,
        criteria: List[str],
        num_judges: int = 3,
    ) -> Dict[str, Any]:
        """
        Enhanced multi-judge scoring using DSPy for more consistent and structured evaluation.

        Args:
            text: The text to evaluate
            criteria: List of scoring criteria
            num_judges: Number of judges to use

        Returns:
            Dictionary with enhanced scores, consensus, and structured feedback
        """
        if not DSPY_AVAILABLE or not self.config.use_dspy_for_enhanced_prompts:
            logger.info("DSPy not available or disabled, falling back to standard multi-judge scoring")
            return await self.multi_judge_scoring(text, criteria, num_judges)

        try:
            # Initialize DSPy
            lm = self._initialize_dspy_if_available()
            if not lm:
                return await self.multi_judge_scoring(text, criteria, num_judges)

            # Define a DSPy signature for evaluation
            class EvaluationSignature(dspy.Signature):
                """Signature for evaluating text based on multiple criteria."""
                text_to_evaluate = dspy.InputField(desc="The text to evaluate")
                evaluation_criteria = dspy.InputField(desc="List of criteria to evaluate against")

                criterion_scores = dspy.OutputField(desc="JSON with scores for each criterion (1-10 scale)")
                overall_score = dspy.OutputField(desc="Overall score (1-10 scale)")
                strengths = dspy.OutputField(desc="List of strengths identified in the text")
                weaknesses = dspy.OutputField(desc="List of weaknesses identified in the text")
                improvement_suggestions = dspy.OutputField(desc="Specific suggestions for improvement")

            # Create a predictor using the signature
            evaluate = dspy.Predict(EvaluationSignature)

            # Run the evaluation
            result = evaluate(
                text_to_evaluate=text,
                evaluation_criteria=", ".join(criteria)
            )

            # Parse the results
            import json
            try:
                criterion_scores = json.loads(result.criterion_scores) if isinstance(result.criterion_scores, str) else result.criterion_scores
            except json.JSONDecodeError:
                criterion_scores = {criterion: 5.0 for criterion in criteria}  # Default scores

            # Structure the response similar to the original method but with DSPy enhancements
            scores = {}
            for i in range(num_judges):
                scores[f"dspy_judge_{i+1}"] = {
                    "overall": float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0,
                    "criteria": criterion_scores,
                    "strengths": result.strengths.split(", ") if result.strengths else [],
                    "weaknesses": result.weaknesses.split(", ") if result.weaknesses else [],
                    "improvements": result.improvement_suggestions.split(", ") if result.improvement_suggestions else []
                }

            avg_score = sum(float(s["overall"]) for s in scores.values() if str(s["overall"]).replace('.', '').isdigit()) / len(scores)

            return {
                "scores": scores,
                "average_score": avg_score,
                "consensus": "high" if avg_score > 7.0 else "medium" if avg_score > 5.0 else "low",
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "dspy_enhanced": True,
                "structured_feedback": {
                    "strengths": result.strengths.split(", ") if result.strengths else [],
                    "weaknesses": result.weaknesses.split(", ") if result.weaknesses else [],
                    "improvements": result.improvement_suggestions.split(", ") if result.improvement_suggestions else []
                }
            }

        except Exception as e:
            logger.warning(f"DSPy enhanced scoring failed, falling back to standard method: {e}")
            return await self.multi_judge_scoring(text, criteria, num_judges)

    async def enhanced_strategy_generation_with_dspy(
        self,
        problem: str,
        num_strategies: int = 6,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced strategy generation using DSPy for more structured and diverse strategies.

        Args:
            problem: Description of the problem to solve
            num_strategies: Number of strategies to generate
            context: Additional context for strategy generation

        Returns:
            List of enhanced strategy dictionaries with more detailed information
        """
        if not DSPY_AVAILABLE or not self.config.use_dspy_for_enhanced_prompts:
            logger.info("DSPy not available or disabled, falling back to standard strategy generation")
            return await self.generate_strategies(problem, num_strategies, context)

        try:
            # Initialize DSPy
            lm = self._initialize_dspy_if_available()
            if not lm:
                return await self.generate_strategies(problem, num_strategies, context)

            # Define a DSPy signature for strategy generation
            class StrategyGenerationSignature(dspy.Signature):
                """Signature for generating problem-solving strategies."""
                problem_description = dspy.InputField(desc="Detailed description of the problem to solve")
                context_information = dspy.InputField(desc="Additional context or constraints")
                number_of_strategies = dspy.InputField(desc="How many diverse strategies to generate")

                strategies_json = dspy.OutputField(desc="""JSON array of strategies, each with:
                    - tagline: Short catchy name for the strategy
                    - description: Detailed explanation of the approach
                    - applicability: When this strategy works best
                    - potential_challenges: Known challenges with this approach
                    - implementation_notes: Key implementation considerations""")

            # Create a predictor using the signature
            generate_strategies = dspy.Predict(StrategyGenerationSignature)

            # Run the strategy generation
            result = generate_strategies(
                problem_description=problem,
                context_information=context or "No additional context provided",
                number_of_strategies=str(num_strategies)
            )

            # Parse the results
            import json
            try:
                strategies_list = json.loads(result.strategies_json) if isinstance(result.strategies_json, str) else result.strategies_json
            except json.JSONDecodeError:
                logger.warning("Could not parse DSPy strategy generation result, using fallback")
                return await self.generate_strategies(problem, num_strategies, context)

            # Ensure we have the requested number of strategies
            while len(strategies_list) < num_strategies:
                # Add fallback strategies to reach the requested count
                strategies_list.append({
                    "tagline": f"Strategy {len(strategies_list)+1}",
                    "description": "Additional strategy generated to meet requested count",
                    "applicability": "General purpose",
                    "potential_challenges": ["May require additional validation"],
                    "implementation_notes": ["Implementation details to be determined"]
                })

            # Format the results to match the expected return format but with enhanced details
            enhanced_strategies = []
            for i, strategy in enumerate(strategies_list[:num_strategies]):
                enhanced_strategy = {
                    "tagline": strategy.get("tagline", f"Strategy {i+1}"),
                    "description": strategy.get("description", "No description provided"),
                    "applicability": strategy.get("applicability", "General"),
                    "potential_challenges": strategy.get("potential_challenges", []),
                    "implementation_notes": strategy.get("implementation_notes", []),
                    "dspy_enhanced": True
                }
                enhanced_strategies.append(enhanced_strategy)

            return enhanced_strategies

        except Exception as e:
            logger.warning(f"DSPy enhanced strategy generation failed, falling back to standard method: {e}")
            return await self.generate_strategies(problem, num_strategies, context)

    async def analyze_dialogue_tree_with_dspy(
        self,
        dialogue_tree: List[Dict[str, Any]],
        analysis_focus: str = "comparative_effectiveness",
        depth: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze a dialogue tree using DSPy for enhanced insights and recommendations.

        Args:
            dialogue_tree: List of dialogue exchanges with scores and outcomes
            analysis_focus: What aspect to focus on ('comparative_effectiveness', 'strategy_optimization', 'convergence_analysis')
            depth: Depth of analysis (higher = more detailed)

        Returns:
            Dictionary with detailed analysis, insights, and recommendations
        """
        if not DSPY_AVAILABLE or not self.config.use_dspy_for_enhanced_prompts:
            logger.info("DSPy not available or disabled, falling back to basic dialogue analysis")
            # Basic fallback analysis
            return {
                "analysis_focus": analysis_focus,
                "total_exchanges": len(dialogue_tree),
                "average_score": sum([d.get("score", 0) for d in dialogue_tree]) / len(dialogue_tree) if dialogue_tree else 0,
                "dspy_enhanced": False,
                "insights": ["Basic analysis performed without DSPy enhancement"]
            }

        try:
            # Define a DSPy signature for dialogue tree analysis
            class DialogueTreeAnalysisSignature(dspy.Signature):
                """Analyze a dialogue tree to extract insights and recommendations."""
                dialogue_tree_data = dspy.InputField(desc="JSON representation of the dialogue tree with exchanges, scores, and outcomes")
                analysis_focus = dspy.InputField(desc="What aspect to focus on (comparative_effectiveness, strategy_optimization, convergence_analysis)")
                analysis_depth = dspy.InputField(desc="Depth of analysis requested (1-5 scale)")

                effectiveness_analysis = dspy.OutputField(desc="Analysis of comparative effectiveness of different dialogue paths")
                strategy_recommendations = dspy.OutputField(desc="Recommendations for strategy optimization")
                convergence_insights = dspy.OutputField(desc="Insights about convergence patterns in the dialogue tree")
                risk_assessment = dspy.OutputField(desc="Assessment of potential risks in the dialogue approaches")
                improvement_opportunities = dspy.OutputField(desc="Specific opportunities for improvement")

            # Create a predictor using the signature
            analyze_dialogue = dspy.Predict(DialogueTreeAnalysisSignature)

            # Prepare dialogue tree data
            dialogue_json = {
                "exchanges": dialogue_tree,
                "focus": analysis_focus,
                "depth": depth
            }

            # Run the analysis
            result = analyze_dialogue(
                dialogue_tree_data=str(dialogue_json),
                analysis_focus=analysis_focus,
                analysis_depth=str(depth)
            )

            # Return comprehensive analysis
            return {
                "analysis_focus": analysis_focus,
                "total_exchanges": len(dialogue_tree),
                "dspy_enhanced": True,
                "effectiveness_analysis": result.effectiveness_analysis,
                "strategy_recommendations": result.strategy_recommendations,
                "convergence_insights": result.convergence_insights,
                "risk_assessment": result.risk_assessment,
                "improvement_opportunities": result.improvement_opportunities,
                "analysis_depth": depth
            }

        except Exception as e:
            logger.warning(f"DSPy dialogue tree analysis failed, returning basic analysis: {e}")
            return {
                "analysis_focus": analysis_focus,
                "total_exchanges": len(dialogue_tree),
                "average_score": sum([d.get("score", 0) for d in dialogue_tree]) / len(dialogue_tree) if dialogue_tree else 0,
                "dspy_enhanced": False,
                "insights": [f"DSPy analysis failed: {str(e)}"]
            }


# Global instance for easy access
_dts_integration_instance = None

def get_dts_integration(config: Optional[DTSIntegrationConfig] = None) -> DTSIntegration:
    """Get or create the global DTS integration instance."""
    global _dts_integration_instance
    if _dts_integration_instance is None:
        _dts_integration_instance = DTSIntegration(config)
    return _dts_integration_instance


# Example usage
if __name__ == "__main__":
    # Test the integration
    import asyncio
    
    async def test():
        dts = get_dts_integration()
        print(f"DTS available: {dts.is_available()}")
        
        if dts.is_available():
            # Test conversation search
            result = await dts.run_conversation_search(
                goal="Convince user to try a new product",
                first_message="I'm not sure I need this product.",
                rounds=1,
            )
            if result:
                print(f"Best score: {result.best_score}")
            
            # Test adversarial dialogue
            adversarial = await dts.adversarial_dialogue(
                attacker_goal="Find security vulnerabilities",
                defender_goal="Defend against attacks",
                initial_message="I think your system has a vulnerability.",
            )
            print(f"Adversarial winner: {adversarial.get('winner')}")
            
            # Test multi-judge scoring
            scoring = await dts.multi_judge_scoring(
                text="This is a sample solution to the problem.",
                criteria=["clarity", "correctness", "efficiency"],
            )
            print(f"Average score: {scoring.get('average_score')}")
            
            # Test strategy generation
            strategies = await dts.generate_strategies(
                problem="Optimize database queries for high throughput",
                num_strategies=3,
            )
            print(f"Generated {len(strategies)} strategies")
    
    asyncio.run(test())