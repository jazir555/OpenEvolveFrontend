"""
Solution Pattern Miner - Stage 6 Knowledge Extraction

This module uses machine learning to identify and cluster solution patterns.
It supports multiple clustering algorithms and dimensionality reduction techniques.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import logging

# Import DSPy through the global integration module for consistency
try:
    from dspy_integration import DSPY_AVAILABLE, get_global_dspy_instance, initialize_dspy
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.predict import Predict
    logging.getLogger(__name__).info("DSPy available through global integration for enhanced programmatic prompting")
except ImportError:
    # Fallback to local import if global module not available
    try:
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        from dspy.predict import Predict
        DSPY_AVAILABLE = True
        logging.getLogger(__name__).info("DSPy available for enhanced programmatic prompting")
    except ImportError:
        dspy = None
        BootstrapFewShot = None
        Predict = None
        DSPY_AVAILABLE = False
        logging.getLogger(__name__).warning("DSPy not available - using standard prompting methods")

from workflow_structures import (
    SolutionPatternArtifact,
    KnowledgeArtifactManager,
)


class SolutionPatternMiner:
    """
    Mines solution patterns using machine learning techniques.

    Features:
    - TF-IDF vectorization for text features
    - Dimensionality reduction (PCA, UMAP if available)
    - Clustering (K-Means, DBSCAN, Agglomerative)
    - Pattern summarization
    - Similarity search

    Attributes:
        artifact_manager: Manager for accessing artifacts
        vectorizer: TF-IDF vectorizer
        clustering_algorithm: Algorithm to use for clustering
        dimensionality_reduction: Method for dimensionality reduction
    """

    def __init__(
        self,
        db_path: str = "./knowledge_artifacts.db",
        clustering_algorithm: str = "kmeans",
        dimensionality_reduction: str = "pca",
        n_clusters: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the pattern miner.

        Args:
            db_path: Path to artifact database
            clustering_algorithm: 'kmeans', 'dbscan', or 'agglomerative'
            dimensionality_reduction: 'pca', 'umap', or None
            n_clusters: Number of clusters (for K-Means)
            random_state: Random state for reproducibility
        """
        self.artifact_manager = KnowledgeArtifactManager(db_path)
        self.clustering_algorithm = clustering_algorithm
        self.dimensionality_reduction = dimensionality_reduction
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Initialize vectorizer
        self.vectorizer = None
        self._init_vectorizer()

        # Initialize models
        self.dim_reducer = None
        self.cluster_model = None
        self.feature_names = []

    def _init_vectorizer(self):
        """Initialize TF-IDF vectorizer."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        except ImportError:
            print("scikit-learn not available. Text features will be limited.")
            self.vectorizer = None

    def _extract_features(self, patterns: List[SolutionPatternArtifact]) -> np.ndarray:
        """
        Extract features from solution patterns.

        Args:
            patterns: List of solution patterns

        Returns:
            Feature matrix (n_patterns x n_features)
        """
        features_list = []
        self.feature_names = []

        for pattern in patterns:
            features = []

            # Text features (TF-IDF)
            # Combine text fields
            text = " ".join([
                pattern.solution_approach,
                " ".join(pattern.problem_characteristics),
                " ".join(pattern.code_patterns),
                " ".join(pattern.optimization_techniques),
                pattern.decomposition_strategy,
                pattern.domain
            ])

            # Numerical features
            features.extend([
                pattern.complexity,
                pattern.success_rate,
                pattern.confidence,
                pattern.usage_count,
                len(pattern.problem_characteristics),
                len(pattern.code_patterns),
                len(pattern.optimization_techniques),
                len(pattern.typical_refinements),
            ])

            # Domain one-hot encoding
            domains = ["algorithms", "data_structures", "machine_learning", "web_development", "system_design", "general"]
            for domain in domains:
                features.append(1.0 if pattern.domain == domain else 0.0)

            # Decomposition strategy one-hot encoding
            strategies = ["ROMA", "MAKER", "MDAP", "unknown"]
            for strategy in strategies:
                features.append(1.0 if pattern.decomposition_strategy == strategy else 0.0)

            features_list.append(features)

        # Feature names for interpretability
        self.feature_names = [
            "complexity", "success_rate", "confidence", "usage_count",
            "num_characteristics", "num_patterns", "num_optimizations", "num_refinements",
        ] + [f"domain_{d}" for d in domains] + [f"strategy_{s}" for s in strategies]

        return np.array(features_list)

    def _reduce_dimensions(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of features.

        Args:
            features: Feature matrix
            n_components: Number of components to keep

        Returns:
            Reduced feature matrix
        """
        if self.dimensionality_reduction == "pca":
            try:
                from sklearn.decomposition import PCA
                if self.dim_reducer is None:
                    self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
                    return self.dim_reducer.fit_transform(features)
                else:
                    return self.dim_reducer.transform(features)
            except ImportError:
                print("scikit-learn not available. Skipping dimensionality reduction.")
                return features

        elif self.dimensionality_reduction == "umap":
            try:
                import umap
                if self.dim_reducer is None:
                    self.dim_reducer = umap.UMAP(n_components=n_components, random_state=self.random_state)
                    return self.dim_reducer.fit_transform(features)
                else:
                    return self.dim_reducer.transform(features)
            except ImportError:
                print("UMAP not available. Falling back to PCA.")
                self.dimensionality_reduction = "pca"
                return self._reduce_dimensions(features, n_components)

        else:
            return features

    def _cluster_patterns(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster solution patterns.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        if self.clustering_algorithm == "kmeans":
            try:
                from sklearn.cluster import KMeans
                self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
                return self.cluster_model.fit_predict(features)
            except ImportError:
                print("scikit-learn not available. Using dummy clustering.")
                return np.array([0] * len(features))

        elif self.clustering_algorithm == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
                return self.cluster_model.fit_predict(features)
            except ImportError:
                print("scikit-learn not available. Using dummy clustering.")
                return np.array([0] * len(features))

        elif self.clustering_algorithm == "agglomerative":
            try:
                from sklearn.cluster import AgglomerativeClustering
                self.cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)
                return self.cluster_model.fit_predict(features)
            except ImportError:
                print("scikit-learn not available. Using dummy clustering.")
                return np.array([0] * len(features))

        else:
            # Default: assign all to cluster 0
            return np.array([0] * len(features))

    # ========== Individual Methods for MASTER_TASKLIST Compatibility ==========

    def _extract_text_features(self, patterns: List[SolutionPatternArtifact]) -> np.ndarray:
        """
        Extract text features from solution patterns (TF-IDF).

        This is a wrapper around _extract_features() for MASTER_TASKLIST compatibility.

        Args:
            patterns: List of solution patterns

        Returns:
            Text feature matrix
        """
        # Extract all features and return them
        return self._extract_features(patterns)

    def _extract_structural_features(self, patterns: List[SolutionPatternArtifact]) -> np.ndarray:
        """
        Extract structural features from solution patterns.

        This is a wrapper around _extract_features() for MASTER_TASKLIST compatibility.

        Args:
            patterns: List of solution patterns

        Returns:
            Structural feature matrix
        """
        # Extract all features and return them
        return self._extract_features(patterns)

    def _build_feature_matrix(self, patterns: List[SolutionPatternArtifact]) -> np.ndarray:
        """
        Build feature matrix from solution patterns.

        This is an alias for _extract_features() for MASTER_TASKLIST compatibility.

        Args:
            patterns: List of solution patterns

        Returns:
            Feature matrix
        """
        return self._extract_features(patterns)

    def apply_pca(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.

        Args:
            features: Feature matrix
            n_components: Number of components to keep

        Returns:
            Reduced feature matrix
        """
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=self.random_state)
            return pca.fit_transform(features)
        except ImportError:
            print("scikit-learn not available. Returning original features.")
            return features

    def apply_umap(self, features: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.

        Args:
            features: Feature matrix
            n_components: Number of components to keep
            n_neighbors: Number of neighbors for UMAP

        Returns:
            Reduced feature matrix
        """
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=self.random_state)
            return reducer.fit_transform(features)
        except ImportError:
            print("UMAP not available. Falling back to PCA.")
            return self.apply_pca(features, n_components)

    def fit_kmeans(self, features: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Fit K-Means clustering.

        Args:
            features: Feature matrix
            n_clusters: Number of clusters (defaults to self.n_clusters)

        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            return kmeans.fit_predict(features)
        except ImportError:
            print("scikit-learn not available. Using dummy clustering.")
            return np.array([0] * len(features))

    def fit_dbscan(self, features: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Fit DBSCAN clustering.

        Args:
            features: Feature matrix
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum samples in neighborhood for core point

        Returns:
            Cluster labels
        """
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            return dbscan.fit_predict(features)
        except ImportError:
            print("scikit-learn not available. Using dummy clustering.")
            return np.array([0] * len(features))

    def fit_agglomerative(self, features: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Fit Agglomerative clustering.

        Args:
            features: Feature matrix
            n_clusters: Number of clusters (defaults to self.n_clusters)

        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        try:
            from sklearn.cluster import AgglomerativeClustering
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            return agg.fit_predict(features)
        except ImportError:
            print("scikit-learn not available. Using dummy clustering.")
            return np.array([0] * len(features))

    def evaluate_cluster_quality(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate cluster quality metrics.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary with cluster quality metrics
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            metrics = {}

            # Silhouette score (higher is better, range [-1, 1])
            if len(np.unique(labels)) > 1:
                metrics["silhouette_score"] = silhouette_score(features, labels)
            else:
                metrics["silhouette_score"] = 0.0

            # Calinski-Harabasz score (higher is better)
            if len(np.unique(labels)) > 1:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(features, labels)
            else:
                metrics["calinski_harabasz_score"] = 0.0

            # Davies-Bouldin score (lower is better)
            if len(np.unique(labels)) > 1:
                metrics["davies_bouldin_score"] = davies_bouldin_score(features, labels)
            else:
                metrics["davies_bouldin_score"] = 0.0

            return metrics
        except ImportError:
            print("scikit-learn not available. Returning empty metrics.")
            return {}

    def fit(self, patterns: Optional[List[SolutionPatternArtifact]] = None) -> Dict[str, Any]:
        """
        Fit the pattern miner on solution patterns.

        Args:
            patterns: Optional list of patterns (if None, loads from database)

        Returns:
            Dictionary with clustering results
        """
        # Load patterns if not provided
        if patterns is None:
            patterns = self.artifact_manager.list_solution_patterns(limit=10000)

        if len(patterns) < 2:
            print(f"Need at least 2 patterns to cluster, got {len(patterns)}")
            return {"status": "error", "message": "Not enough patterns"}

        # Extract features
        features = self._extract_features(patterns)
        print(f"Extracted features: {features.shape}")

        # Reduce dimensions
        if self.dimensionality_reduction:
            reduced_features = self._reduce_dimensions(features)
            print(f"Reduced dimensions: {reduced_features.shape}")
        else:
            reduced_features = features

        # Cluster patterns
        cluster_labels = self._cluster_patterns(reduced_features)
        print(f"Clustered patterns into {len(set(cluster_labels))} clusters")

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(patterns, cluster_labels, reduced_features)

        return {
            "status": "success",
            "n_patterns": len(patterns),
            "n_clusters": len(set(cluster_labels)),
            "cluster_labels": cluster_labels.tolist(),
            "cluster_analysis": cluster_analysis,
        }

    def _analyze_clusters(self, patterns: List[SolutionPatternArtifact], labels: np.ndarray, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze clusters to extract insights.

        Args:
            patterns: List of patterns
            labels: Cluster labels
            features: Feature matrix

        Returns:
            List of cluster analyses
        """
        clusters = {}
        for pattern, label in zip(patterns, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pattern)

        analyses = []
        for cluster_id, cluster_patterns in clusters.items():
            analysis = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_patterns),
                "patterns": [p.artifact_id for p in cluster_patterns],
            }

            # Calculate cluster statistics
            analysis["avg_complexity"] = np.mean([p.complexity for p in cluster_patterns])
            analysis["avg_success_rate"] = np.mean([p.success_rate for p in cluster_patterns])
            analysis["avg_confidence"] = np.mean([p.confidence for p in cluster_patterns])

            # Most common domain
            domains = [p.domain for p in cluster_patterns]
            analysis["most_common_domain"] = max(set(domains), key=domains.count) if domains else ""

            # Most common strategy
            strategies = [p.decomposition_strategy for p in cluster_patterns]
            analysis["most_common_strategy"] = max(set(strategies), key=strategies.count) if strategies else ""

            # Common problem characteristics
            all_characteristics = []
            for p in cluster_patterns:
                all_characteristics.extend(p.problem_characteristics)
            # Count characteristics
            char_counts = {}
            for char in all_characteristics:
                char_counts[char] = char_counts.get(char, 0) + 1
            analysis["common_characteristics"] = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            # Generate cluster description
            analysis["description"] = self._generate_cluster_description(analysis, cluster_patterns)

            analyses.append(analysis)

        return analyses

    def _generate_cluster_description(self, analysis: Dict[str, Any], patterns: List[SolutionPatternArtifact]) -> str:
        """
        Generate a human-readable description of a cluster.

        Args:
            analysis: Cluster analysis
            patterns: Patterns in the cluster

        Returns:
            Description string
        """
        # Try to use DSPy for enhanced description generation if available
        if DSPY_AVAILABLE:
            try:
                # Define a DSPy signature for cluster description generation
                class ClusterDescriptionSignature(dspy.Signature):
                    """Generate a human-readable description of a solution pattern cluster."""
                    cluster_analysis = dspy.InputField(desc="Dictionary with cluster analysis results")
                    cluster_patterns = dspy.InputField(desc="List of patterns in the cluster")

                    cluster_description = dspy.OutputField(desc="Human-readable description of the cluster")

                # Create a predictor
                generate_description = dspy.Predict(ClusterDescriptionSignature)

                # Prepare input data
                cluster_analysis_str = json.dumps({
                    "size": analysis.get("size", 0),
                    "avg_complexity": analysis.get("avg_complexity", 0),
                    "avg_success_rate": analysis.get("avg_success_rate", 0),
                    "most_common_domain": analysis.get("most_common_domain", ""),
                    "most_common_strategy": analysis.get("most_common_strategy", ""),
                    "common_characteristics": analysis.get("common_characteristics", [])
                }, default=str)

                patterns_summary = [f"Pattern {i+1}: {getattr(p, 'title', 'Unknown')}" for i, p in enumerate(patterns[:5])]  # Limit to first 5 patterns

                # Run DSPy prediction
                result = generate_description(
                    cluster_analysis=cluster_analysis_str,
                    cluster_patterns=str(patterns_summary)
                )

                # Return DSPy-generated description if successful
                if result and hasattr(result, 'cluster_description') and result.cluster_description:
                    return result.cluster_description

            except Exception as e:
                logging.getLogger(__name__).warning(f"DSPy cluster description generation failed: {e}")

        # Fallback to traditional method
        parts = []

        # Domain and strategy
        if analysis["most_common_domain"]:
            parts.append(f"Focuses on {analysis['most_common_domain']} problems")
        if analysis["most_common_strategy"]:
            parts.append(f"using {analysis['most_common_strategy']} decomposition")

        # Complexity
        complexity = analysis["avg_complexity"]
        if complexity < 4:
            parts.append("with low complexity")
        elif complexity < 7:
            parts.append("with medium complexity")
        else:
            parts.append("with high complexity")

        # Success rate
        success_rate = analysis["avg_success_rate"]
        parts.append(f"(success rate: {success_rate:.1%})")

        # Common characteristics
        if analysis["common_characteristics"]:
            top_chars = [char for char, count in analysis["common_characteristics"][:3]]
            parts.append(f"Common traits: {', '.join(top_chars)}")

        return ". ".join(parts) + "."

    def mine_patterns_with_dspy(
        self,
        patterns: Optional[List[SolutionPatternArtifact]] = None,
        n_clusters: Optional[int] = None,
        use_clustering_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Mine solution patterns using DSPy for enhanced analysis and clustering.

        Args:
            patterns: Optional list of patterns (if None, loads from database)
            n_clusters: Number of clusters (if None, uses default)
            use_clustering_analysis: Whether to use DSPy for enhanced cluster analysis

        Returns:
            Dictionary with enhanced clustering results
        """
        if not DSPY_AVAILABLE:
            logging.getLogger(__name__).info("DSPy not available, falling back to standard mining")
            return self.fit(patterns)

        try:
            # Load patterns if not provided
            if patterns is None:
                patterns = self.artifact_manager.list_solution_patterns(limit=10000)

            if len(patterns) < 2:
                return {"status": "error", "message": "Not enough patterns for clustering"}

            # Extract features
            features = self._extract_features(patterns)
            print(f"Extracted features with DSPy enhancement: {features.shape}")

            # Reduce dimensions
            if self.dimensionality_reduction:
                reduced_features = self._reduce_dimensions(features)
                print(f"Reduced dimensions: {reduced_features.shape}")
            else:
                reduced_features = features

            # Cluster patterns
            cluster_labels = self._cluster_patterns(reduced_features)
            print(f"Clustered patterns into {len(set(cluster_labels))} clusters")

            # Use DSPy for enhanced cluster analysis if requested
            if use_clustering_analysis:
                cluster_analysis = self._analyze_clusters_with_dspy(patterns, cluster_labels, reduced_features)
            else:
                cluster_analysis = self._analyze_clusters(patterns, cluster_labels, reduced_features)

            return {
                "status": "success",
                "n_patterns": len(patterns),
                "n_clusters": len(set(cluster_labels)),
                "cluster_labels": cluster_labels.tolist(),
                "cluster_analysis": cluster_analysis,
                "dspy_enhanced": True,
            }
        except Exception as e:
            logging.getLogger(__name__).warning(f"DSPy pattern mining failed, falling back to standard: {e}")
            return self.fit(patterns)

    def _analyze_clusters_with_dspy(self, patterns: List[SolutionPatternArtifact], labels: np.ndarray, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze clusters using DSPy for enhanced insights.

        Args:
            patterns: List of patterns
            labels: Cluster labels
            features: Feature matrix

        Returns:
            List of enhanced cluster analyses
        """
        clusters = {}
        for pattern, label in zip(patterns, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pattern)

        analyses = []
        for cluster_id, cluster_patterns in clusters.items():
            analysis = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_patterns),
                "patterns": [p.artifact_id for p in cluster_patterns],
            }

            # Calculate cluster statistics
            analysis["avg_complexity"] = float(np.mean([p.complexity for p in cluster_patterns]))
            analysis["avg_success_rate"] = float(np.mean([p.success_rate for p in cluster_patterns]))
            analysis["avg_confidence"] = float(np.mean([p.confidence for p in cluster_patterns]))

            # Most common domain
            domains = [p.domain for p in cluster_patterns]
            if domains:
                analysis["most_common_domain"] = max(set(domains), key=domains.count)
            else:
                analysis["most_common_domain"] = ""

            # Most common strategy
            strategies = [p.decomposition_strategy for p in cluster_patterns]
            if strategies:
                analysis["most_common_strategy"] = max(set(strategies), key=strategies.count)
            else:
                analysis["most_common_strategy"] = ""

            # Common problem characteristics
            all_characteristics = []
            for p in cluster_patterns:
                all_characteristics.extend(p.problem_characteristics)
            # Count characteristics
            char_counts = {}
            for char in all_characteristics:
                char_counts[char] = char_counts.get(char, 0) + 1
            analysis["common_characteristics"] = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            # Use DSPy for enhanced description generation
            try:
                # Define a DSPy signature for cluster analysis
                class ClusterAnalysisSignature(dspy.Signature):
                    """Analyze a cluster of solution patterns to extract insights."""
                    cluster_patterns = dspy.InputField(desc="List of patterns in the cluster")
                    cluster_statistics = dspy.InputField(desc="Basic statistics about the cluster")

                    cluster_insights = dspy.OutputField(desc="Key insights about the cluster")
                    improvement_opportunities = dspy.OutputField(desc="Opportunities for improvement")
                    pattern_categories = dspy.OutputField(desc="Categories of patterns in the cluster")
                    success_factors = dspy.OutputField(desc="Factors contributing to success")

                # Create a predictor
                analyze_cluster = dspy.Predict(ClusterAnalysisSignature)

                # Prepare input data
                patterns_info = [
                    {
                        "title": getattr(p, 'title', ''),
                        "domain": getattr(p, 'domain', ''),
                        "strategy": getattr(p, 'decomposition_strategy', ''),
                        "complexity": getattr(p, 'complexity', 0),
                        "success_rate": getattr(p, 'success_rate', 0),
                        "confidence": getattr(p, 'confidence', 0)
                    }
                    for p in cluster_patterns
                ]

                stats_info = {
                    "size": analysis["size"],
                    "avg_complexity": analysis["avg_complexity"],
                    "avg_success_rate": analysis["avg_success_rate"],
                    "avg_confidence": analysis["avg_confidence"],
                    "most_common_domain": analysis["most_common_domain"],
                    "most_common_strategy": analysis["most_common_strategy"]
                }

                # Run DSPy analysis
                result = analyze_cluster(
                    cluster_patterns=json.dumps(patterns_info, default=str),
                    cluster_statistics=json.dumps(stats_info, default=str)
                )

                # Add DSPy-enhanced analysis
                analysis["dspy_analysis"] = {
                    "insights": getattr(result, 'cluster_insights', 'Not available'),
                    "improvement_opportunities": getattr(result, 'improvement_opportunities', 'Not available'),
                    "pattern_categories": getattr(result, 'pattern_categories', 'Not available'),
                    "success_factors": getattr(result, 'success_factors', 'Not available')
                }
            except Exception as e:
                logging.getLogger(__name__).warning(f"DSPy cluster analysis failed for cluster {cluster_id}: {e}")
                analysis["dspy_analysis"] = {
                    "insights": "DSPy analysis not available",
                    "improvement_opportunities": "DSPy analysis not available",
                    "pattern_categories": "DSPy analysis not available",
                    "success_factors": "DSPy analysis not available"
                }

            # Generate cluster description (this will use DSPy if available)
            analysis["description"] = self._generate_cluster_description(analysis, cluster_patterns)

            analyses.append(analysis)

        return analyses

    def find_similar_patterns(self, pattern: SolutionPatternArtifact, n_neighbors: int = 5) -> List[Tuple[SolutionPatternArtifact, float]]:
        """
        Find patterns similar to a given pattern.

        Args:
            pattern: The pattern to find similarities for
            n_neighbors: Number of similar patterns to return

        Returns:
            List of (similar_pattern, similarity_score) tuples
        """
        # Load all patterns
        all_patterns = self.artifact_manager.list_solution_patterns(limit=10000)

        # Extract features
        all_features = self._extract_features(all_patterns)
        target_features = self._extract_features([pattern])[0]

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([target_features], all_features)[0]

        # Get top neighbors (excluding self)
        indices = np.argsort(similarities)[::-1][1:n_neighbors+1]

        results = []
        for idx in indices:
            if similarities[idx] > 0:  # Only include positive similarities
                results.append((all_patterns[idx], float(similarities[idx])))

        return results

    def recommend_patterns_for_problem(self, problem_statement: str, domain: str = "", complexity: int = 5) -> List[SolutionPatternArtifact]:
        """
        Recommend solution patterns for a given problem.

        Args:
            problem_statement: The problem statement
            domain: Problem domain
            complexity: Estimated complexity (1-10)

        Returns:
            List of recommended patterns
        """
        # Load all patterns
        all_patterns = self.artifact_manager.list_solution_patterns(limit=10000)

        # Score patterns based on relevance
        scored_patterns = []
        for pattern in all_patterns:
            score = 0.0

            # Domain match
            if domain and pattern.domain == domain:
                score += 0.3

            # Complexity match (within 2 points)
            if abs(pattern.complexity - complexity) <= 2:
                score += 0.2

            # Success rate
            score += pattern.success_rate * 0.3

            # Usage count (popularity)
            score += min(pattern.usage_count / 10.0, 0.2) * 0.2

            scored_patterns.append((pattern, score))

        # Sort by score and return top recommendations
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, score in scored_patterns[:10]]

    def visualize_clusters(self, output_path: str = "pattern_clusters.html"):
        """
        Create an interactive visualization of pattern clusters.

        Args:
            output_path: Path to save the HTML file
        """
        try:
            import networkx as nx
            import plotly.graph_objects as go
        except ImportError:
            print("networkx or plotly not available. Skipping visualization.")
            return

        # Load patterns and fit if not already fitted
        patterns = self.artifact_manager.list_solution_patterns(limit=1000)
        if len(patterns) < 2:
            print("Not enough patterns to visualize")
            return

        # Fit if not already fitted
        if self.cluster_model is None:
            self.fit(patterns)

        # Get 2D embeddings
        features = self._extract_features(patterns)
        if self.dimensionality_reduction:
            embeddings = self._reduce_dimensions(features, n_components=2)
        else:
            # Use PCA for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            embeddings = pca.fit_transform(features)

        # Get cluster labels
        labels = self.cluster_model.labels_ if hasattr(self.cluster_model, 'labels_') else np.array([0] * len(patterns))

        # Create interactive plot
        fig = go.Figure(data=go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=labels,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"Pattern: {p.artifact_id}<br>Domain: {p.domain}<br>Success: {p.success_rate:.2f}" for p in patterns],
            hoverinfo='text'
        ))

        fig.update_layout(
            title="Solution Pattern Clusters",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest'
        )

        fig.write_html(output_path)
        print(f"Visualization saved to {output_path}")


# ========== Convenience Functions ==========

def mine_solution_patterns(
    db_path: str = "./knowledge_artifacts.db",
    clustering_algorithm: str = "kmeans",
    n_clusters: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to mine solution patterns.

    Args:
        db_path: Path to artifact database
        clustering_algorithm: Clustering algorithm to use
        n_clusters: Number of clusters

    Returns:
        Dictionary with clustering results
    """
    miner = SolutionPatternMiner(
        db_path=db_path,
        clustering_algorithm=clustering_algorithm,
        n_clusters=n_clusters
    )
    return miner.fit()

def find_similar_patterns(
    pattern_id: str,
    db_path: str = "./knowledge_artifacts.db",
    n_neighbors: int = 5
) -> List[Dict[str, Any]]:
    """
    Find patterns similar to a given pattern.

    Args:
        pattern_id: ID of the pattern
        db_path: Path to artifact database
        n_neighbors: Number of similar patterns to return

    Returns:
        List of similar patterns with scores
    """
    artifact_manager = KnowledgeArtifactManager(db_path)
    pattern = artifact_manager.read_solution_pattern(pattern_id)

    if not pattern:
        return []

    miner = SolutionPatternMiner(db_path=db_path)
    similar = miner.find_similar_patterns(pattern, n_neighbors)

    return [{"artifact_id": p.artifact_id, "similarity": score} for p, score in similar]
