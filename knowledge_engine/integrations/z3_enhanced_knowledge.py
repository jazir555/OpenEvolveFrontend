"""
Enhanced Z3 Knowledge Integration with ML-Powered Pattern Learning

Advanced features:
- Machine learning-based pattern matching
- Automated strategy optimization
- Cross-domain knowledge transfer
- Real-time performance analytics
- Adaptive learning rates
- CAV-NLP integration for formalization and verification

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import numpy as np

# CAV-NLP Integration
CAV_NLP_AVAILABLE = False
UnifiedMathService = None
try:
    from openevolve.unified_math_service import UnifiedMathService as _UnifiedMathService
    UnifiedMathService = _UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    try:
        from unified_math_service import UnifiedMathService as _UnifiedMathService
        UnifiedMathService = _UnifiedMathService
        CAV_NLP_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)
if CAV_NLP_AVAILABLE:
    logger.info("CAV-NLP UnifiedMathService available for enhanced Z3 knowledge integration")

# Configure logging
logger = logging.getLogger(__name__)

# ML imports (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using fallback similarity")

# Import base integration
try:
    from knowledge_engine.integrations.z3_knowledge_integration import (
        Z3KnowledgeIntegration,
        Z3KnowledgeEntry
    )
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

# Import Z3 extraction
try:
    from z3_knowledge_extraction import (
        Z3KnowledgeExtractor,
        ProofPattern,
        ConstraintPattern,
        SolutionStrategy,
        get_z3_knowledge_extractor
    )
    Z3_KE_AVAILABLE = True
except ImportError:
    Z3_KE_AVAILABLE = False


@dataclass
class PatternEmbedding:
    """Embedding representation of a pattern for ML operations."""
    pattern_id: str
    pattern_type: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    total_uses: int = 0
    successful_uses: int = 0
    avg_solving_time: float = 0.0
    avg_memory_usage: float = 0.0
    problem_types: List[str] = field(default_factory=list)
    effectiveness_trend: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.successful_uses / self.total_uses if self.total_uses > 0 else 0.0
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score."""
        if not self.effectiveness_trend:
            return 0.5
        # Weight recent performance more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.effectiveness_trend)))
        weighted_avg = np.average(self.effectiveness_trend, weights=weights)
        return float(weighted_avg)


class MLPoweredPatternMatcher:
    """Machine learning powered pattern matching system."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.pattern_embeddings: Dict[str, PatternEmbedding] = {}
        self.vectorizer = None
        self.clustering_model = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize the text vectorizer."""
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=self.embedding_dim,
                ngram_range=(1, 3),
                min_df=1
            )
    
    def create_embedding(
        self,
        pattern_id: str,
        pattern_content: str,
        pattern_type: str,
        metadata: Optional[Dict] = None
    ) -> PatternEmbedding:
        """Create embedding for a pattern."""
        if SKLEARN_AVAILABLE and self.vectorizer:
            # Fit and transform if needed
            try:
                vector = self.vectorizer.fit_transform([pattern_content]).toarray()[0]
            except:
                # Fallback to simple hashing
                vector = self._simple_hash_embedding(pattern_content)
        else:
            vector = self._simple_hash_embedding(pattern_content)
        
        embedding = PatternEmbedding(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
            metadata=metadata or {}
        )
        
        self.pattern_embeddings[pattern_id] = embedding
        return embedding
    
    def _simple_hash_embedding(self, content: str) -> List[float]:
        """Create simple hash-based embedding."""
        # Use multiple hash functions for better distribution
        hash_values = []
        for i in range(self.embedding_dim):
            hash_input = f"{content}_{i}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            # Normalize to [-1, 1]
            normalized = (hash_val % 10000) / 5000 - 1
            hash_values.append(normalized)
        return hash_values
    
    def find_similar_patterns(
        self,
        query_pattern: str,
        pattern_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find patterns similar to query using ML similarity."""
        if not self.pattern_embeddings:
            return []
        
        # Create query embedding
        query_embedding = self.create_embedding(
            pattern_id="_query_",
            pattern_content=query_pattern,
            pattern_type="query"
        )
        
        # Calculate similarities
        similarities = []
        for pattern_id, embedding in self.pattern_embeddings.items():
            if pattern_id == "_query_":
                continue
            if pattern_type and embedding.pattern_type != pattern_type:
                continue
            
            similarity = self._calculate_similarity(
                query_embedding.vector,
                embedding.vector
            )
            
            if similarity >= min_similarity:
                similarities.append((pattern_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if SKLEARN_AVAILABLE:
            try:
                v1 = np.array(vec1).reshape(1, -1)
                v2 = np.array(vec2).reshape(1, -1)
                return float(cosine_similarity(v1, v2)[0][0])
            except:
                pass
        
        # Fallback to manual cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def cluster_patterns(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster patterns using K-means."""
        if not SKLEARN_AVAILABLE or len(self.pattern_embeddings) < n_clusters:
            return {}
        
        # Prepare data
        pattern_ids = list(self.pattern_embeddings.keys())
        embeddings = [self.pattern_embeddings[pid].vector for pid in pattern_ids]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for pid, label in zip(pattern_ids, labels):
            clusters[int(label)].append(pid)
        
        return dict(clusters)


class AdaptiveStrategyOptimizer:
    """Adaptive optimizer for solution strategies."""
    
    def __init__(self):
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.problem_type_mapping: Dict[str, List[str]] = defaultdict(list)
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
    
    def record_strategy_use(
        self,
        strategy_id: str,
        success: bool,
        solving_time: float,
        memory_usage: float,
        problem_type: str
    ):
        """Record performance of a strategy use."""
        if strategy_id not in self.strategy_performances:
            self.strategy_performances[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id
            )
        
        perf = self.strategy_performances[strategy_id]
        perf.total_uses += 1
        if success:
            perf.successful_uses += 1
        
        # Update running averages
        perf.avg_solving_time = (
            (perf.avg_solving_time * (perf.total_uses - 1) + solving_time)
            / perf.total_uses
        )
        perf.avg_memory_usage = (
            (perf.avg_memory_usage * (perf.total_uses - 1) + memory_usage)
            / perf.total_uses
        )
        
        if problem_type not in perf.problem_types:
            perf.problem_types.append(problem_type)
        
        # Update effectiveness trend
        effectiveness = 1.0 if success else 0.0
        perf.effectiveness_trend.append(effectiveness)
        
        # Keep only last 100 entries
        if len(perf.effectiveness_trend) > 100:
            perf.effectiveness_trend = perf.effectiveness_trend[-100:]
        
        # Update problem type mapping
        if strategy_id not in self.problem_type_mapping[problem_type]:
            self.problem_type_mapping[problem_type].append(strategy_id)
    
    def get_optimal_strategy(
        self,
        problem_features: Dict[str, Any],
        problem_type: str,
        available_strategies: List[str]
    ) -> Tuple[str, float]:
        """
        Get optimal strategy using multi-armed bandit approach.
        
        Returns:
            Tuple of (strategy_id, confidence)
        """
        # Filter strategies for this problem type
        relevant_strategies = [
            sid for sid in available_strategies
            if sid in self.problem_type_mapping.get(problem_type, [])
            or sid in self.strategy_performances
        ]
        
        if not relevant_strategies:
            return (available_strategies[0] if available_strategies else "", 0.5)
        
        # Calculate UCB (Upper Confidence Bound) scores
        strategy_scores = []
        for sid in relevant_strategies:
            if sid in self.strategy_performances:
                perf = self.strategy_performances[sid]
                exploitation = perf.effectiveness_score
                exploration = self.exploration_rate * (
                    (2 * np.log(max(perf.total_uses, 1))) / max(perf.total_uses, 1)
                ) ** 0.5
                score = exploitation + exploration
            else:
                score = 0.5  # Default score for untried strategies
            
            strategy_scores.append((sid, score))
        
        # Select best strategy
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        best_strategy, best_score = strategy_scores[0]
        
        # Calculate confidence based on number of uses
        total_uses = self.strategy_performances.get(
            best_strategy, StrategyPerformance(best_strategy)
        ).total_uses
        confidence = min(0.95, 0.5 + (total_uses / 100) * 0.45)
        
        return best_strategy, confidence
    
    def get_strategy_ranking(self, problem_type: str) -> List[Tuple[str, float]]:
        """Get ranked list of strategies for a problem type."""
        strategies = self.problem_type_mapping.get(problem_type, [])
        
        ranked = []
        for sid in strategies:
            if sid in self.strategy_performances:
                perf = self.strategy_performances[sid]
                score = perf.effectiveness_score * perf.success_rate
                ranked.append((sid, score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def suggest_strategy_improvements(self, strategy_id: str) -> List[str]:
        """Suggest improvements for a strategy based on performance."""
        if strategy_id not in self.strategy_performances:
            return []
        
        perf = self.strategy_performances[strategy_id]
        suggestions = []
        
        if perf.success_rate < 0.7:
            suggestions.append(f"Low success rate ({perf.success_rate:.1%}). Consider adding fallback tactics.")
        
        if perf.avg_solving_time > 10.0:
            suggestions.append(f"High avg solving time ({perf.avg_solving_time:.1f}s). Consider optimizing tactic order.")
        
        if len(perf.effectiveness_trend) >= 10:
            recent_trend = np.mean(perf.effectiveness_trend[-10:])
            older_trend = np.mean(perf.effectiveness_trend[-20:-10]) if len(perf.effectiveness_trend) >= 20 else 0.5
            
            if recent_trend < older_trend - 0.1:
                suggestions.append("Performance declining. Strategy may need adaptation.")
        
        return suggestions


class CrossDomainKnowledgeTransfer:
    """Transfer knowledge between different problem domains."""
    
    def __init__(self):
        self.domain_mappings: Dict[Tuple[str, str], float] = {}
        self.transfer_history: List[Dict[str, Any]] = []
    
    def calculate_domain_similarity(
        self,
        domain1: str,
        domain2: str,
        patterns1: List[Dict],
        patterns2: List[Dict]
    ) -> float:
        """Calculate similarity between two domains."""
        # Simple heuristic: compare pattern type distributions
        types1 = defaultdict(int)
        types2 = defaultdict(int)
        
        for p in patterns1:
            types1[p.get('type', 'unknown')] += 1
        for p in patterns2:
            types2[p.get('type', 'unknown')] += 1
        
        # Calculate Jaccard similarity
        all_types = set(types1.keys()) | set(types2.keys())
        if not all_types:
            return 0.0
        
        intersection = sum(min(types1[t], types2[t]) for t in all_types)
        union = sum(max(types1[t], types2[t]) for t in all_types)
        
        similarity = intersection / union if union > 0 else 0.0
        self.domain_mappings[(domain1, domain2)] = similarity
        
        return similarity
    
    def suggest_transfers(
        self,
        source_domain: str,
        target_domain: str,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Suggest knowledge transfers between domains."""
        similarity = self.domain_mappings.get((source_domain, target_domain), 0.0)
        
        if similarity < min_similarity:
            return []
        
        # In a real implementation, this would analyze specific patterns
        # that could be transferred
        suggestions = [
            {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "similarity": similarity,
                "suggested_patterns": [],
                "confidence": similarity
            }
        ]
        
        return suggestions


class EnhancedZ3KnowledgeIntegration:
    """
    Enhanced Z3 Knowledge Integration with ML capabilities.
    
    Extends base Z3KnowledgeIntegration with:
    - ML-powered pattern matching
    - Adaptive strategy optimization
    - Cross-domain knowledge transfer
    - Real-time analytics
    - CAV-NLP formalization and hybrid verification
    """
    
    def __init__(self, storage_engine: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced integration."""
        self.config = config or {}
        self.base_integration = None
        if BASE_AVAILABLE:
            from knowledge_engine.integrations.z3_knowledge_integration import Z3KnowledgeIntegration
            self.base_integration = Z3KnowledgeIntegration(storage_engine, config)
        
        self.z3_extractor = get_z3_knowledge_extractor() if Z3_KE_AVAILABLE else None
        self.pattern_matcher = MLPoweredPatternMatcher()
        self.strategy_optimizer = AdaptiveStrategyOptimizer()
        self.knowledge_transfer = CrossDomainKnowledgeTransfer()
        
        # CAV-NLP Integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", True)
        self.math_service = None
        if self.use_cav_nlp and CAV_NLP_AVAILABLE and UnifiedMathService:
            try:
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP math service initialized for enhanced integration")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP math service: {e}")
                self.use_cav_nlp = False
        
        # Analytics
        self.analytics = {
            "extractions": 0,
            "successful_matches": 0,
            "strategy_optimizations": 0,
            "domain_transfers": 0,
            "cav_nlp_formalizations": 0,
            "cav_nlp_verifications": 0,
            "cav_nlp_canonicalizations": 0
        }
        
        logger.info("Enhanced Z3 Knowledge Integration initialized")
    
    async def initialize(self):
        """Initialize all components."""
        if self.base_integration:
            await self.base_integration.initialize()
    
    async def extract_with_ml_enhancement(
        self,
        result: Any,
        problem_statement: str,
        problem_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract knowledge with ML enhancement.
        
        Args:
            result: Z3 solver result
            problem_statement: Original problem
            problem_type: Problem classification
            
        Returns:
            Enhanced extraction results with ML insights
        """
        # Base extraction
        base_result = await self._base_extract(result, problem_statement, problem_type)
        
        # ML enhancements
        ml_insights = {}
        
        # Find similar patterns
        if self.pattern_matcher and self.pattern_matcher.pattern_embeddings:
            similar = self.pattern_matcher.find_similar_patterns(
                problem_statement,
                pattern_type=problem_type,
                top_k=3
            )
            ml_insights["similar_patterns"] = similar
        
        # Get optimal strategy recommendation
        if result and hasattr(result, 'constraints'):
            features = {
                "type": problem_type,
                "constraint_count": len(result.constraints),
                "var_count": len(getattr(result, 'model', {}).assignments or {})
            }
            
            available_strategies = list(
                self.z3_extractor.strategies.keys()
            ) if self.z3_extractor else []
            
            if available_strategies:
                optimal_strategy, confidence = self.strategy_optimizer.get_optimal_strategy(
                    features, problem_type, available_strategies
                )
                ml_insights["recommended_strategy"] = {
                    "strategy_id": optimal_strategy,
                    "confidence": confidence
                }
        
        # Check for domain transfer opportunities
        # (In production, would compare with other domains)
        
        self.analytics["extractions"] += 1
        
        return {
            "base_extraction": base_result,
            "ml_insights": ml_insights,
            "problem_type": problem_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _base_extract(
        self,
        result: Any,
        problem_statement: str,
        problem_type: str
    ) -> Dict[str, Any]:
        """Perform base knowledge extraction."""
        if self.base_integration:
            return await self.base_integration.extract_from_solver_result(
                result, problem_statement, problem_type
            )
        
        # Fallback to direct extractor
        if self.z3_extractor:
            insights = self.z3_extractor.extract_insights(result, problem_statement)
            patterns = self.z3_extractor.analyze_constraints(
                getattr(result, 'constraints', []),
                getattr(result, 'solving_time', 0.0),
                getattr(result, 'success', False)
            )
            
            return {
                "insights": [i.to_dict() for i in insights],
                "patterns": [p.to_dict() for p in patterns]
            }
        
        return {"error": "No extraction engine available"}
    
    def create_pattern_embeddings(self, patterns: List[Dict[str, Any]]):
        """Create embeddings for patterns."""
        for pattern in patterns:
            pattern_id = pattern.get('id', pattern.get('pattern_id', 'unknown'))
            content = json.dumps(pattern)
            pattern_type = pattern.get('type', 'unknown')
            
            self.pattern_matcher.create_embedding(
                pattern_id=pattern_id,
                pattern_content=content,
                pattern_type=pattern_type,
                metadata=pattern
            )
        
        logger.info(f"Created {len(patterns)} pattern embeddings")
    
    def optimize_strategies(self) -> Dict[str, Any]:
        """Run strategy optimization."""
        optimizations = []
        
        for strategy_id, perf in self.strategy_optimizer.strategy_performances.items():
            suggestions = self.strategy_optimizer.suggest_strategy_improvements(strategy_id)
            if suggestions:
                optimizations.append({
                    "strategy_id": strategy_id,
                    "performance": {
                        "success_rate": perf.success_rate,
                        "avg_time": perf.avg_solving_time,
                        "total_uses": perf.total_uses
                    },
                    "suggestions": suggestions
                })
        
        self.analytics["strategy_optimizations"] += len(optimizations)
        
        return {
            "optimizations": optimizations,
            "total_strategies_analyzed": len(self.strategy_optimizer.strategy_performances)
        }
    
    async def formalize_to_z3_knowledge(self, natural_language: str, domain: str = "general") -> Optional[Dict[str, Any]]:
        """
        Formalize natural language to Z3-compatible knowledge using CAV-NLP.
        
        Args:
            natural_language: Natural language description
            domain: Problem domain
            
        Returns:
            Formalized knowledge structure
        """
        if not self.use_cav_nlp or not self.math_service:
            return None
        
        try:
            result = await self.math_service.formalize(natural_language, domain_hint=domain)
            self.analytics["cav_nlp_formalizations"] += 1
            
            knowledge = {
                "source_text": natural_language,
                "formal_code": result.code if hasattr(result, 'code') else str(result),
                "lean_code": result.lean_code if hasattr(result, 'lean_code') else None,
                "z3_constraints": result.z3_constraints if hasattr(result, 'z3_constraints') else None,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.5,
                "domain": domain,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create pattern embedding for the formalized knowledge
            self.pattern_matcher.create_embedding(
                pattern_id=f"cav_nlp_{hash(natural_language) % 1000000}",
                pattern_content=json.dumps(knowledge),
                pattern_type=domain,
                metadata=knowledge
            )
            
            logger.info({
                "msg": "Knowledge formalized with CAV-NLP",
                "domain": domain,
                "confidence": knowledge["confidence"]
            })
            
            return knowledge
            
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed: {e}")
            return None
    
    async def verify_knowledge_hybrid(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify knowledge using hybrid Z3 + Lean approach.
        
        Args:
            knowledge: Knowledge structure to verify
            
        Returns:
            Verification result
        """
        if not self.use_cav_nlp or not self.math_service:
            return {"verified": False, "error": "CAV-NLP not available"}
        
        try:
            formal_code = knowledge.get("lean_code") or knowledge.get("formal_code")
            if not formal_code:
                return {"verified": False, "error": "No formal code available"}
            
            result = await self.math_service.verify(formal_code)
            self.analytics["cav_nlp_verifications"] += 1
            
            verification = {
                "verified": result.success if hasattr(result, 'success') else False,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                "proof": result.proof if hasattr(result, 'proof') else None,
                "z3_result": result.z3_result if hasattr(result, 'z3_result') else None,
                "lean_result": result.lean_result if hasattr(result, 'lean_result') else None,
                "method": "hybrid_z3_lean",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "Knowledge verified with hybrid approach",
                "verified": verification["verified"],
                "confidence": verification["confidence"]
            })
            
            return verification
            
        except Exception as e:
            logger.error(f"Hybrid verification failed: {e}")
            return {"verified": False, "error": str(e)}
    
    async def canonicalize_knowledge_representation(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonicalize knowledge representation to standard form.
        
        Args:
            knowledge: Knowledge structure to canonicalize
            
        Returns:
            Canonicalized knowledge
        """
        if not self.use_cav_nlp or not self.math_service:
            return knowledge
        
        try:
            formal_code = knowledge.get("formal_code") or knowledge.get("lean_code")
            if not formal_code:
                return knowledge
            
            result = await self.math_service.canonicalize(formal_code)
            self.analytics["cav_nlp_canonicalizations"] += 1
            
            canonicalized = knowledge.copy()
            canonicalized.update({
                "canonical_form": result.code if hasattr(result, 'code') else str(result),
                "simplified": result.simplified if hasattr(result, 'simplified') else False,
                "normalized": True,
                "canonicalized_at": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info("Knowledge canonicalized with CAV-NLP")
            return canonicalized
            
        except Exception as e:
            logger.warning(f"Canonicalization failed: {e}")
            return knowledge
    
    async def export_knowledge_to_lean(self, knowledge: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Export knowledge to Lean 4 format.
        
        Args:
            knowledge: Knowledge structure
            output_path: Optional output file path
            
        Returns:
            Lean code string
        """
        # Try CAV-NLP first
        if self.use_cav_nlp and self.math_service:
            try:
                formal_code = knowledge.get("formal_code") or knowledge.get("lean_code")
                if formal_code:
                    result = await self.math_service.export_to_lean(formal_code)
                    lean_code = result.lean_code if hasattr(result, 'lean_code') else str(result)
                    
                    if output_path:
                        with open(output_path, 'w') as f:
                            f.write(lean_code)
                        logger.info(f"Knowledge exported to Lean: {output_path}")
                    
                    return lean_code
            except Exception as e:
                logger.warning(f"CAV-NLP export failed, using fallback: {e}")
        
        # Fallback: generate basic Lean code
        lean_code = self._generate_lean_export(knowledge)
        if output_path and lean_code:
            with open(output_path, 'w') as f:
                f.write(lean_code)
        
        return lean_code
    
    def _generate_lean_export(self, knowledge: Dict[str, Any]) -> str:
        """Generate basic Lean code from knowledge."""
        lean_code = knowledge.get("lean_code", "")
        if lean_code:
            return lean_code
        
        source = knowledge.get("source_text", "Extracted Knowledge")
        formal = knowledge.get("formal_code", "")
        
        lean_code = f"""-- Auto-generated Lean 4 code from Z3 Knowledge
-- Source: {source[:100]}...
-- Generated: {datetime.now(timezone.utc).isoformat()}

import Mathlib

-- Formalized Knowledge
"""
        if formal:
            lean_code += f"\n-- Original formal code:\n-- {formal[:500]}\n"
        
        lean_code += """
-- Theorem placeholder
theorem extracted_knowledge : True := by
  trivial
"""
        return lean_code
    
    async def extract_with_cav_nlp_enhancement(
        self,
        result: Any,
        problem_statement: str,
        problem_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract knowledge with CAV-NLP enhancement.
        
        Combines ML-enhanced extraction with CAV-NLP formalization.
        """
        # Get base ML-enhanced extraction
        base_result = await self.extract_with_ml_enhancement(result, problem_statement, problem_type)
        
        # Add CAV-NLP enhancement
        cav_nlp_result = None
        if self.use_cav_nlp and self.math_service:
            try:
                # Formalize the problem statement
                cav_nlp_result = await self.formalize_to_z3_knowledge(problem_statement, problem_type)
                
                # If we have formalized knowledge, verify it
                if cav_nlp_result:
                    verification = await self.verify_knowledge_hybrid(cav_nlp_result)
                    cav_nlp_result["verification"] = verification
                    
                    # Canonicalize
                    cav_nlp_result = await self.canonicalize_knowledge_representation(cav_nlp_result)
                    
            except Exception as e:
                logger.warning(f"CAV-NLP enhancement failed: {e}")
        
        return {
            **base_result,
            "cav_nlp_enhancement": cav_nlp_result,
            "enhanced": cav_nlp_result is not None
        }
    
    def get_cav_nlp_analytics(self) -> Dict[str, Any]:
        """Get CAV-NLP specific analytics."""
        return {
            "available": CAV_NLP_AVAILABLE,
            "enabled": self.use_cav_nlp,
            "initialized": self.math_service is not None,
            "formalizations": self.analytics["cav_nlp_formalizations"],
            "verifications": self.analytics["cav_nlp_verifications"],
            "canonicalizations": self.analytics["cav_nlp_canonicalizations"]
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get integration analytics."""
        return {
            **self.analytics,
            "pattern_embeddings": len(self.pattern_matcher.pattern_embeddings),
            "strategies_tracked": len(self.strategy_optimizer.strategy_performances),
            "domain_mappings": len(self.knowledge_transfer.domain_mappings),
            "cav_nlp": self.get_cav_nlp_analytics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global instance
_enhanced_integration: Optional[EnhancedZ3KnowledgeIntegration] = None


async def get_enhanced_z3_integration() -> EnhancedZ3KnowledgeIntegration:
    """Get global enhanced integration instance."""
    global _enhanced_integration
    if _enhanced_integration is None:
        _enhanced_integration = EnhancedZ3KnowledgeIntegration()
        await _enhanced_integration.initialize()
    return _enhanced_integration


# Example usage
async def example_enhanced():
    """Example: Enhanced integration usage."""
    print("Enhanced Z3 Knowledge Integration Example")
    print("=" * 50)
    
    integration = await get_enhanced_z3_integration()
    
    # Create mock result
    class MockResult:
        success = True
        model = type('Model', (), {'assignments': {'x': 5, 'y': 10}})()
        constraints = ["(> x 0)", "(< x 10)", "(= y (+ x 5))"]
        solving_time = 1.5
    
    # Extract with ML enhancement
    result = await integration.extract_with_ml_enhancement(
        result=MockResult(),
        problem_statement="Linear constraints",
        problem_type="linear"
    )
    
    print(f"\nExtraction completed:")
    print(f"  ML insights: {list(result['ml_insights'].keys())}")
    
    # Get analytics
    analytics = integration.get_analytics()
    print(f"\nAnalytics:")
    print(f"  Pattern embeddings: {analytics['pattern_embeddings']}")
    print(f"  Strategies tracked: {analytics['strategies_tracked']}")


if __name__ == "__main__":
    asyncio.run(example_enhanced())
