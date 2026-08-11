"""
Sovereign-Grade Problem Decomposition System - Knowledge Management
Extracts, stores, and applies learned decomposition patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from sovereign_data_models import (
    DecompositionPlan, Pattern, ProblemType, DecompositionStrategy, generate_id, ProblemDefinition
)
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import with_retry # Import with_retry

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """Manages knowledge extraction and application for decomposition with LLM-powered learning."""
    
    def __init__(self, database: Optional[SovereignDatabase] = None, openevolve_client=None):
        """
        Initialize knowledge manager.
        
        Args:
            database: Optional database instance
            openevolve_client: Optional OpenEvolve client for LLM
        """
        self.database = database or SovereignDatabase()
        self.openevolve_client = openevolve_client
        self.logger = logging.getLogger(__name__)
        
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for knowledge management")
        
        # Track strategy performance
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Pattern embeddings cache (for similarity)
        self.pattern_embeddings: Dict[str, List[float]] = {}
    
    def extract_patterns(
        self,
        plan: DecompositionPlan,
        success: bool,
        quality_score: float
    ) -> List[Pattern]:
        """
        Extracts patterns from successful/failed decompositions using LLM analysis.
        
        Args:
            plan: The decomposition plan
            success: Whether the decomposition was successful
            quality_score: Quality score achieved
            
        Returns:
            List of extracted Pattern objects
        """
        self.logger.info(f"Extracting patterns from plan {plan.id} (success={success})")
        
        if not success or quality_score < 0.7: # Increased threshold
            # Don't extract patterns from poor decompositions
            self.logger.info("Skipping pattern extraction for low-quality decomposition.")
            return []
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for pattern extraction.")

        try:
            patterns = self._extract_patterns_with_llm(plan, quality_score)
            if not patterns:
                self.logger.warning("LLM pattern extraction returned no patterns.")
                return []

            self.logger.info(f"LLM extracted {len(patterns)} patterns")
            
            # Store patterns
            for pattern in patterns:
                self.store_pattern(pattern)
            
            return patterns
        except Exception as e:
            self.logger.error(f"LLM-based pattern extraction failed: {e}")
            raise RuntimeError(f"Failed to extract patterns using LLM: {e}") from e
    
    @with_retry(max_attempts=3, retry_on=(RuntimeError, ValueError))
    def _extract_patterns_with_llm(self, plan: DecompositionPlan, quality_score: float) -> List[Pattern]:
        """Use LLM to extract deep patterns from successful decomposition."""
        # Get problem from database
        problem = self.database.get_problem(plan.problem_id)
        if not problem:
            return []
        
        # Build decomposition summary
        sp_summary = "\n".join([
            f"{i+1}. {sp.title} ({sp.type.value}) - Priority: {sp.priority}, Effort: {sp.estimated_effort}h"
            for i, sp in enumerate(plan.sub_problems[:8])
        ])
        
        prompt = f"""You are an expert at extracting reusable patterns from successful problem decompositions. Analyze this decomposition and identify transferable patterns.

PROBLEM:
Type: {problem.problem_type.value}
Domain: {problem.domain_context.domain}
Complexity: {problem.complexity_score.overall_complexity}/10
Description: {problem.description[:200]}...

SUCCESSFUL DECOMPOSITION:
Strategy: {plan.strategy.value}
Quality Score: {quality_score:.2f}
Sub-problems:
{sp_summary}

PATTERN EXTRACTION:
Identify 2-4 reusable patterns that made this decomposition successful:

For each pattern provide:
1. Pattern Name: Clear, descriptive name
2. Pattern Type: (strategy/structural/workflow/domain-specific)
3. Key Characteristics: What makes this pattern effective
4. Applicability: When to use this pattern
5. Success Indicators: What suggests this pattern will work

Format EXACTLY as:
---
PATTERN 1
Name: <name>
Type: <type>
Characteristics: <char1> | <char2> | <char3>
Applicability: <when to use>
SuccessIndicators: <indicator1> | <indicator2>
---

Provide 2-4 patterns. Be specific and actionable."""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.4,
            max_tokens=1200
        )
        
        if result.success and result.best_code:
            return self._parse_pattern_response(result.best_code, problem, plan, quality_score)
        
        return []
    
    def _parse_pattern_response(self, response: str, problem, plan: DecompositionPlan, quality_score: float) -> List[Pattern]:
        """Parse LLM pattern extraction response."""
        patterns = []
        sections = response.split('---')
        
        for section in sections:
            section = section.strip()
            if not section or 'PATTERN' not in section:
                continue
            
            try:
                # Parse fields
                name = self._extract_pattern_field(section, 'Name:')
                pattern_type = self._extract_pattern_field(section, 'Type:')
                characteristics = self._extract_pattern_field(section, 'Characteristics:')
                applicability = self._extract_pattern_field(section, 'Applicability:')
                success_indicators = self._extract_pattern_field(section, 'SuccessIndicators:')
                
                if not name:
                    continue
                
                # Create pattern
                pattern = Pattern(
                    id=generate_id("pattern"),
                    name=name,
                    problem_type=problem.problem_type,
                    strategy=plan.strategy,
                    pattern_data={
                        'type': pattern_type,
                        'characteristics': [c.strip() for c in characteristics.split('|') if c.strip()],
                        'applicability': applicability,
                        'success_indicators': [s.strip() for s in success_indicators.split('|') if s.strip()],
                        'sub_problem_count': len(plan.sub_problems),
                        'avg_complexity': sum(sp.complexity_score.overall_complexity for sp in plan.sub_problems) / len(plan.sub_problems),
                        'extraction_method': 'llm'
                    },
                    success_rate=1.0,  # Initial
                    usage_count=1,
                    avg_quality_score=quality_score,
                    applicable_domains=[problem.domain_context.domain],
                    discovered_at=datetime.now(),
                    last_used=datetime.now()
                )
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse pattern section: {e}")
                continue
        
        return patterns
    
    def _extract_pattern_field(self, text: str, field_name: str) -> str:
        """Extract field value from pattern text."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(field_name):
                return line.split(':', 1)[1].strip()
        return ""
    
    def store_pattern(self, pattern: Pattern) -> bool:
        """
        Stores pattern in knowledge base.
        
        Args:
            pattern: The pattern to store
            
        Returns:
            True if successful
        """
        self.logger.info(f"Storing pattern {pattern.id}")
        
        # Check if similar pattern exists
        existing = self.database.get_patterns_by_type(pattern.problem_type.value)
        
        for existing_pattern in existing:
            if self._patterns_similar(pattern, existing_pattern):
                # Update existing pattern
                existing_pattern.usage_count += 1
                existing_pattern.avg_quality_score = (
                    (existing_pattern.avg_quality_score * (existing_pattern.usage_count - 1) +
                     pattern.avg_quality_score) / existing_pattern.usage_count
                )
                existing_pattern.last_used = datetime.now()
                self.database.update_pattern(existing_pattern)
                return True
        
        # Store new pattern
        return self.database.create_pattern(pattern)
    
    def retrieve_patterns(
        self,
        problem_type: ProblemType,
        domain: Optional[str] = None,
        min_success_rate: float = 0.7
    ) -> List[Pattern]:
        """
        Retrieves relevant patterns for problem.
        
        Args:
            problem_type: Type of problem
            domain: Optional domain filter
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of relevant Pattern objects
        """
        self.logger.info(f"Retrieving patterns for {problem_type.value}")
        
        # Get patterns by type
        patterns = self.database.get_patterns_by_type(problem_type.value)
        
        # Filter by success rate
        patterns = [p for p in patterns if p.success_rate >= min_success_rate]
        
        # Filter by domain if specified
        if domain:
            patterns = [p for p in patterns 
                       if domain in p.applicable_domains or not p.applicable_domains]
        
        # Sort by effectiveness
        patterns.sort(key=lambda p: (p.success_rate, p.avg_quality_score), reverse=True)
        
        return patterns
    
    def apply_pattern(
        self,
        pattern: Pattern,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Applies learned pattern to new problem.
        
        Args:
            pattern: The pattern to apply
            problem_description: Description of the new problem
            
        Returns:
            Dictionary with application guidance
        """
        self.logger.info(f"Applying pattern {pattern.id}")
        
        # Update pattern usage
        pattern.usage_count += 1
        pattern.last_used = datetime.now()
        self.database.update_pattern(pattern)
        
        # Generate application guidance
        guidance = {
            'pattern_id': pattern.id,
            'strategy': pattern.strategy.value,
            'description': pattern.pattern_description,
            'success_rate': pattern.success_rate,
            'avg_quality': pattern.avg_quality_score,
            'recommendations': self._generate_recommendations(pattern),
            'applicable': True
        }
        
        return guidance
    
    def track_strategy_performance(
        self,
        strategy: DecompositionStrategy,
        quality_score: float
    ) -> None:
        """
        Tracks strategy effectiveness over time.
        
        Args:
            strategy: The decomposition strategy used
            quality_score: Quality score achieved
        """
        self.logger.info(f"Tracking performance for {strategy.value}: {quality_score:.2f}")
        
        self.strategy_performance[strategy.value].append(quality_score)
    
    def get_strategy_performance(
        self,
        strategy: DecompositionStrategy
    ) -> Dict[str, float]:
        """
        Gets performance metrics for a strategy.
        
        Args:
            strategy: The decomposition strategy
            
        Returns:
            Dictionary with performance metrics
        """
        scores = self.strategy_performance.get(strategy.value, [])
        
        if not scores:
            return {
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'usage_count': 0
            }
        
        return {
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'usage_count': len(scores)
        }
    
    def adapt_strategies(self) -> Dict[str, Any]:
        """
        Adapts strategies based on performance data using LLM analysis.
        
        Returns:
            Dictionary with adaptation recommendations
        """
        self.logger.info("Adapting strategies based on performance using LLM.")

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for strategy adaptation.")

        performance_data = "Historical performance data:\n"
        for strategy_name, scores in self.strategy_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                recent_avg = sum(scores[-5:]) / min(5, len(scores))
                performance_data += f"- {strategy_name}: Avg Score={avg_score:.2f}, Recent Avg={recent_avg:.2f}, Usage={len(scores)}\n"

        prompt = f"""You are an expert in strategic analysis. Analyze the performance of these decomposition strategies and provide adaptation recommendations.

PERFORMANCE DATA:
{performance_data}

ANALYSIS TASK:
For each strategy, analyze its performance trend (improving, declining, stable) and recommend an action.
Actions can be: 'increase_usage', 'maintain', 'review_and_adjust', 'deprecate'.

Provide your analysis in this EXACT format:
[strategy_name]: <trend> | <action> | <confidence>
[strategy_name]: <trend> | <action> | <confidence>
...
"""
        try:
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.3,
                max_tokens=500
            )

            if result.success and result.best_code:
                recommendations = {}
                lines = result.best_code.strip().split('\n')
                for line in lines:
                    if ':' in line and '|' in line:
                        strategy_name, rest = line.split(':', 1)
                        parts = [p.strip() for p in rest.split('|')]
                        if len(parts) == 3:
                            recommendations[strategy_name.strip()] = {
                                'trend': parts[0],
                                'action': parts[1],
                                'confidence': parts[2]
                            }
                return recommendations
        except Exception as e:
            self.logger.error(f"LLM-based strategy adaptation failed: {e}")
            raise RuntimeError(f"Failed to adapt strategies using LLM: {e}") from e
        
        return {}
    
    def get_best_strategy(
        self,
        problem: 'ProblemDefinition'
    ) -> Optional[DecompositionStrategy]:
        """
        Recommends best strategy based on historical performance using LLM analysis.
        
        Args:
            problem: The problem definition
            
        Returns:
            Recommended DecompositionStrategy or None
        """
        self.logger.info(f"Finding best strategy for {problem.problem_type.value} using LLM.")

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for strategy recommendation.")

        # Get performance data
        performance_data = "Historical performance data:\n"
        for strategy_name in ['semantic', 'dependency', 'complexity', 'hybrid']:
            perf = self.get_strategy_performance(DecompositionStrategy(strategy_name))
            performance_data += f"- {strategy_name}: Avg Score={perf['avg_score']:.2f}, Usage={perf['usage_count']}\n"

        prompt = f"""You are an expert in selecting problem decomposition strategies. Given the problem and historical performance data, select the BEST strategy.

PROBLEM:
Title: {problem.title}
Description: {problem.description}
Domain: {problem.domain_context.domain}
Type: {problem.problem_type.value}
Complexity: {problem.complexity_score.overall_complexity}/10

{performance_data}

AVAILABLE STRATEGIES:
1. SEMANTIC: Decomposes based on semantic concepts. Best for problems with clear conceptual structure.
2. DEPENDENCY: Decomposes based on prerequisite relationships. Best for problems with strong sequential dependencies.
3. COMPLEXITY: Decomposes to balance cognitive load. Best for very complex problems.
4. HYBRID: Combines multiple strategies. Best for complex problems needing multiple perspectives.

Based on the problem and the historical data, which strategy is most likely to succeed?
Respond with ONLY ONE WORD - the strategy name:
semantic OR dependency OR complexity OR hybrid
"""
        try:
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.2,
                max_tokens=50
            )

            if result.success and result.best_code:
                strategy_name = result.best_code.strip().lower()
                if strategy_name in ['semantic', 'dependency', 'complexity', 'hybrid']:
                    return DecompositionStrategy(strategy_name)
        except Exception as e:
            self.logger.error(f"LLM-based strategy recommendation failed: {e}")
            raise RuntimeError(f"Failed to get best strategy using LLM: {e}") from e
        
        return None
    
    # Helper methods
    

    
    def _patterns_similar(self, p1: Pattern, p2: Pattern) -> bool:
        """Check if two patterns are similar using semantic analysis."""
        # Basic checks first
        if p1.problem_type != p2.problem_type:
            return False
        
        if p1.strategy != p2.strategy:
            return False
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for pattern similarity.")

        try:
            similarity = self._calculate_pattern_similarity(p1, p2)
            return similarity > 0.85 # Increased threshold for better matching
        except Exception as e:
            self.logger.error(f"Semantic similarity check failed: {e}")
            raise RuntimeError(f"Failed to check pattern similarity using LLM: {e}") from e
    
    def _calculate_pattern_similarity(self, p1: Pattern, p2: Pattern) -> float:
        """Calculate semantic similarity between patterns."""
        # Get embeddings
        emb1 = self._get_pattern_embedding(p1)
        emb2 = self._get_pattern_embedding(p2)
        
        if not emb1 or not emb2:
            return 0.0
        
        # Cosine similarity
        import math
        dot = sum(a * b for a, b in zip(emb1, emb2))
        mag1 = math.sqrt(sum(a * a for a in emb1))
        mag2 = math.sqrt(sum(b * b for b in emb2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    def _get_pattern_embedding(self, pattern: Pattern) -> Optional[List[float]]:
        """Get or generate embedding for pattern using LLM."""
        if pattern.id in self.pattern_embeddings:
            return self.pattern_embeddings[pattern.id]

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for generating embeddings.")

        pattern_text = f"Pattern Name: {pattern.name}\n"
        pattern_text += f"Problem Type: {pattern.problem_type.value}\n"
        pattern_text += f"Strategy: {pattern.strategy.value}\n"
        pattern_text += f"Applicability: {pattern.pattern_data.get('applicability', '')}"

        prompt = f"""Generate a 128-dimensional embedding vector for the following decomposition pattern. The embedding should capture the semantic meaning of the pattern.
Output the embedding as a comma-separated list of 128 floating-point numbers.

Pattern:
{pattern_text}

Embedding:
"""
        try:
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.1,
                max_tokens=1024 
            )

            if result.success and result.best_code:
                embedding_str = result.best_code.strip()
                embedding = [float(x.strip()) for x in embedding_str.split(',')]
                if len(embedding) == 128:
                    self.pattern_embeddings[pattern.id] = embedding
                    return embedding
                else:
                    self.logger.warning(f"LLM returned an embedding of length {len(embedding)}, expected 128.")
        except Exception as e:
            self.logger.error(f"LLM embedding generation failed: {e}")

        return None
    
    def _generate_recommendations(self, pattern: Pattern) -> List[str]:
        """Generate recommendations based on pattern."""
        recommendations = []
        
        recommendations.append(f"Use {pattern.strategy.value} decomposition strategy")
        
        if pattern.success_rate >= 0.9:
            recommendations.append("This pattern has high success rate")
        
        if pattern.usage_count > 10:
            recommendations.append("This pattern is well-tested")
        
        if 'sub_problem_count' in pattern.metadata:
            count = pattern.metadata['sub_problem_count']
            recommendations.append(f"Consider decomposing into ~{count} sub-problems")
        
        return recommendations
