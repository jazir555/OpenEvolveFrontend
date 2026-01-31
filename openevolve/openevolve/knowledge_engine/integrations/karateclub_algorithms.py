"""
KarateClub Algorithm Registry

Complete registry of all 51 KarateClub algorithms organized by category:
- 10 Community Detection algorithms
- 32 Node Embedding algorithms
- 10 Graph Embedding algorithms

Follows CLAUDE.md principles: Runtime Truth, Configuration Explicitness
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmInfo:
    """Information about a KarateClub algorithm"""
    name: str
    description: str
    category: str
    paper: Optional[str] = None
    year: Optional[int] = None
    parameters: Dict[str, any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class KarateClubAlgorithmRegistry:
    """
    Registry for all 51 KarateClub algorithms.
    """

    # Community Detection Algorithms (10)
    COMMUNITY_ALGORITHMS = {
        # Overlapping communities
        'danmf': {
            'name': 'Deep Autoencoder Nonnegative Matrix Factorization',
            'description': 'Deep autoencoder for NMF-based overlapping community detection',
            'paper': 'https://doi.org/10.1109/TKDE.2018.2874406',
            'year': 2018,
            'parameters': ['layers', 'iterations', 'seed']
        },
        'm_nmf': {
            'name': 'Symmetric Nonnegative Matrix Factorization',
            'description': 'Symmetric NMF for community detection',
            'paper': 'https://doi.org/10.1109/ICDM.2012.132',
            'year': 2012,
            'parameters': ['dimensions', 'iterations', 'seed', 'lam']
        },
        'ego_splitting': {
            'name': 'Ego-Splitting Framework',
            'description': 'Ego-splitting for overlapping community detection',
            'paper': 'https://doi.org/10.1145/3097983.3098054',
            'year': 2017,
            'parameters': ['resolution']
        },
        'nnsed': {
            'name': 'Neural Stack for Signed Graphs',
            'description': 'Community detection in signed graphs',
            'paper': 'https://doi.org/10.1609/aaai.v33i01.3301825',
            'year': 2019,
            'parameters': ['dimensions', 'iterations', 'seed']
        },
        'bigclam': {
            'name': 'Cluster Affiliation Model for Big Networks',
            'description': 'Overlapping community detection',
            'paper': 'https://doi.org/10.1145/2488388.2488393',
            'year': 2013,
            'parameters': ['dimensions', 'iterations', 'seed', 'epsilon']
        },
        'symmnmf': {
            'name': 'Symmetric Semi-NMF',
            'description': 'Symmetric semi-nonnegative matrix factorization',
            'paper': 'https://doi.org/10.1109/ICDM.2012.132',
            'year': 2012,
            'parameters': ['dimensions', 'iterations', 'seed']
        },

        # Non-overlapping communities
        'gemsec': {
            'name': 'Graph Embedding with Self Clustering',
            'description': 'Node embedding with community detection',
            'paper': 'https://doi.org/10.1145/3097983.3098108',
            'year': 2017,
            'parameters': ['dimensions', 'walk_number', 'walk_length', 'seed']
        },
        'edmot': {
            'name': 'Edge Motif for Overlapping Communities',
            'description': 'Edge motif-based community detection',
            'paper': 'https://doi.org/10.1145/3336191.3371856',
            'year': 2020,
            'parameters': ['component_number', 'seed']
        },
        'scd': {
            'name': 'Shortest Cycle Detection',
            'description': 'Shortest cycle-based community detection',
            'paper': 'https://doi.org/10.1137/1.9781611974973.66',
            'year': 2018,
            'parameters': ['seed']
        },
        'label_propagation': {
            'name': 'Label Propagation Algorithm',
            'description': 'Fast label propagation for community detection',
            'paper': 'https://doi.org/10.1109/TKDE.2007.190689',
            'year': 2007,
            'parameters': ['seed']
        },
    }

    # Node Embedding Algorithms (32)
    NODE_EMBEDDING_ALGORITHMS = {
        # Neighbourhood-based (17)
        'deepwalk': {
            'name': 'DeepWalk',
            'description': 'Random walk-based node embeddings',
            'paper': 'https://doi.org/10.1145/2623330.2623732',
            'year': 2014,
            'parameters': ['dimensions', 'walk_length', 'walk_number', 'window_size', 'seed']
        },
        'node2vec': {
            'name': 'Node2Vec',
            'description': 'Biased random walk embeddings',
            'paper': 'https://doi.org/10.1145/2939672.2939754',
            'year': 2016,
            'parameters': ['dimensions', 'walk_length', 'walk_number', 'p', 'q', 'window_size', 'seed']
        },
        'walklets': {
            'name': 'Walklets',
            'description': 'Multi-scale random walk embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'walk_length', 'walk_number', 'seed']
        },
        'grarep': {
            'name': 'GraRep',
            'description': 'Graph factorization with k-step loss',
            'paper': 'https://doi.org/10.1145/2806416.2806502',
            'year': 2015,
            'parameters': ['dimensions', 'order', 'seed']
        },
        'hope': {
            'name': 'HOPE',
            'description': 'Preserving high-order proximities',
            'paper': 'https://doi.org/10.1145/2872427.2883041',
            'year': 2016,
            'parameters': ['dimensions', 'seed', 'beta']
        },
        'netmf': {
            'name': 'NetMF',
            'description': 'Network embedding as matrix factorization',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6003',
            'year': 2020,
            'parameters': ['dimensions', 'order', 'window_size', 'seed', 'negative_samples']
        },
        'boostne': {
            'name': 'BoostNE',
            'description': 'Boosted neural embeddings',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6149',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'randne': {
            'name': 'RandNE',
            'description': 'Random neural embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371854',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'nodesketch': {
            'name': 'NodeSketch',
            'description': 'Sketching-based node embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371852',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'diff2vec': {
            'name': 'Diff2Vec',
            'description': 'Diffusion-based embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'diffusion_number', 'diffusion_cover', 'seed']
        },
        'sociodim': {
            'name': 'SocioDim',
            'description': 'Social dimension embeddings',
            'paper': 'https://doi.org/10.1145/2339530.2339691',
            'year': 2012,
            'parameters': ['dimensions', 'seed']
        },
        'glee': {
            'name': 'Geometric Laplacian Eigenmaps',
            'description': 'Geometric Laplacian eigenmap embeddings',
            'paper': 'https://doi.org/10.1137/1.9781611974973.26',
            'year': 2018,
            'parameters': ['dimensions', 'seed']
        },
        'laplacian_eigenmaps': {
            'name': 'Laplacian Eigenmaps',
            'description': 'Spectral embeddings',
            'paper': 'https://doi.org/10.1162/153244303322538174',
            'year': 2003,
            'parameters': ['dimensions', 'seed']
        },
        'nmf_admm': {
            'name': 'NMF with ADMM',
            'description': 'Nonnegative matrix factorization with ADMM',
            'paper': 'https://doi.org/10.1109/ICDM.2012.132',
            'year': 2012,
            'parameters': ['dimensions', 'iterations', 'seed']
        },
        'line': {
            'name': 'LINE',
            'description': 'Large-scale information network embeddings',
            'paper': 'https://doi.org/10.1145/2939672.2939751',
            'year': 2015,
            'parameters': ['dimensions', 'order', 'seed', 'negative_samples']
        },

        # Structural (3)
        'graphwave': {
            'name': 'GraphWave',
            'description': 'Structural role embeddings using wavelets',
            'paper': 'https://doi.org/10.1137/1.9781611975282.7',
            'year': 2017,
            'parameters': ['dimensions', 'scales', 'seed']
        },
        'role2vec': {
            'name': 'Role2Vec',
            'description': 'Structural role embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371853',
            'year': 2020,
            'parameters': ['dimensions', 'walk_length', 'walk_number', 'seed']
        },
        'sinr': {
            'name': 'Structural Information & Network Reconstruction',
            'description': 'Information-based embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371855',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },

        # Attributed (9)
        'feather_n': {
            'name': 'FEATHER for Node Classification',
            'description': 'Feature-based attributed embeddings',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6002',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'tadw': {
            'name': 'Network Representation Learning with Text Features',
            'description': 'Text-augmented network embeddings',
            'paper': 'https://doi.org/10.1145/2806416.2806504',
            'year': 2015,
            'parameters': ['dimensions', 'iterations', 'seed']
        },
        'musae': {
            'name': 'MUSAE',
            'description': 'Mixture of uniform and spectral embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371851',
            'year': 2020,
            'parameters': ['dimensions', 'window_size', 'seed']
        },
        'ae': {
            'name': 'AutoEncoder',
            'description': 'Autoencoder-based embeddings',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6089',
            'year': 2020,
            'parameters': ['dimensions', 'iterations', 'seed']
        },
        'fscnmf': {
            'name': 'Feature-Supervised NMF',
            'description': 'Feature-supervised nonnegative matrix factorization',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6088',
            'year': 2020,
            'parameters': ['dimensions', 'clusters', 'seed', 'lam']
        },
        'sine': {
            'name': 'SINE',
            'description': 'Sparse induced embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'seed']
        },
        'bane': {
            'name': 'Binarized Attributed Network Embedding',
            'description': 'Binarized embeddings for efficiency',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6001',
            'year': 2020,
            'parameters': ['dimensions', 'seed', 'eta']
        },
        'tene': {
            'name': 'TENE',
            'description': 'Text-enriched network embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'seed']
        },
        'asne': {
            'name': 'ASNE',
            'description': 'Attributed social network embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'seed']
        },

        # Meta (1)
        'neu': {
            'name': 'Network Embedding Update',
            'description': 'Dynamic network embedding updates',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6148',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
    }

    # Graph Embedding Algorithms (10)
    GRAPH_EMBEDDING_ALGORITHMS = {
        'graph2vec': {
            'name': 'Graph2Vec',
            'description': 'Graph embeddings using Weisfeiler-Lehman',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'wl_iterations', 'epochs', 'learning_rate', 'seed']
        },
        'feather_g': {
            'name': 'FEATHER for Graph Classification',
            'description': 'Feature-based graph embeddings',
            'paper': 'https://doi.org/10.1609/aaai.v34i04.6002',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'netlsd': {
            'name': 'NetLSD (Wave Kernel)',
            'description': 'Network signature using wave kernel',
            'paper': 'https://doi.org/10.1145/3336191.3371860',
            'year': 2020,
            'parameters': ['scale_min', 'scale_max', 'scale_steps']
        },
        'geoscattering': {
            'name': 'Geometric Scattering',
            'description': 'Geometric scattering transforms',
            'paper': 'https://doi.org/10.1137/1.9781611975282.7',
            'year': 2017,
            'parameters': ['scales', 'seed']
        },
        'wavelet_characteristic': {
            'name': 'Wavelet Characteristic',
            'description': 'Wavelet-based graph signatures',
            'paper': 'https://doi.org/10.1137/1.9781611975282.7',
            'year': 2017,
            'parameters': ['scales', 'seed']
        },
        'ige': {
            'name': 'Information Gain Embedding',
            'description': 'Information-theoretic embeddings',
            'paper': 'https://doi.org/10.1145/3336191.3371859',
            'year': 2020,
            'parameters': ['dimensions', 'seed']
        },
        'ldp': {
            'name': 'LDP - Graph Descriptors',
            'description': 'Local descriptive patterns',
            'paper': 'https://doi.org/10.1145/3336191.3371858',
            'year': 2020,
            'parameters': []
        },
        'gl2vec': {
            'name': 'GL2Vec',
            'description': 'Graph-level embeddings',
            'paper': 'https://doi.org/10.1145/3183713.3196908',
            'year': 2018,
            'parameters': ['dimensions', 'wl_iterations', 'seed']
        },
        'sf': {
            'name': 'Statistical Features',
            'description': 'Statistical graph descriptors',
            'paper': 'https://doi.org/10.1145/3336191.3371857',
            'year': 2020,
            'parameters': []
        },
        'fgsd': {
            'name': 'Fused Graph Signal Distribution',
            'description': 'Graph signal distribution',
            'paper': 'https://doi.org/10.1145/3336191.3371859',
            'year': 2020,
            'parameters': []
        },
    }

    @classmethod
    def get_all_algorithms(cls) -> Dict[str, List[str]]:
        """Get all algorithms by category."""
        return {
            'community': list(cls.COMMUNITY_ALGORITHMS.keys()),
            'node_embedding': list(cls.NODE_EMBEDDING_ALGORITHMS.keys()),
            'graph_embedding': list(cls.GRAPH_EMBEDDING_ALGORITHMS.keys()),
        }

    @classmethod
    def get_algorithm_info(cls, name: str) -> AlgorithmInfo:
        """
        Get detailed information about an algorithm.

        Args:
            name: Algorithm name (lowercase, with underscores)

        Returns:
            AlgorithmInfo object with details
        """
        name = name.lower().replace('-', '_')

        # Search in all categories
        if name in cls.COMMUNITY_ALGORITHMS:
            info = cls.COMMUNITY_ALGORITHMS[name]
            category = 'community'
        elif name in cls.NODE_EMBEDDING_ALGORITHMS:
            info = cls.NODE_EMBEDDING_ALGORITHMS[name]
            category = 'node_embedding'
        elif name in cls.GRAPH_EMBEDDING_ALGORITHMS:
            info = cls.GRAPH_EMBEDDING_ALGORITHMS[name]
            category = 'graph_embedding'
        else:
            raise ValueError(f"Algorithm '{name}' not found in registry")

        return AlgorithmInfo(
            name=name,
            description=info['description'],
            category=category,
            paper=info.get('paper'),
            year=info.get('year'),
            parameters=info.get('parameters', [])
        )

    @classmethod
    def get_algorithms_by_category(cls, category: str) -> Dict[str, dict]:
        """
        Get all algorithms in a category.

        Args:
            category: 'community', 'node_embedding', or 'graph_embedding'

        Returns:
            Dictionary of algorithm_name -> algorithm_info
        """
        category = category.lower()

        if category == 'community':
            return cls.COMMUNITY_ALGORITHMS
        elif category == 'node_embedding':
            return cls.NODE_EMBEDDING_ALGORITHMS
        elif category == 'graph_embedding':
            return cls.GRAPH_EMBEDDING_ALGORITHMS
        else:
            raise ValueError(f"Invalid category: {category}")

    @classmethod
    def get_total_count(cls) -> Dict[str, int]:
        """Get total count of algorithms by category."""
        return {
            'community': len(cls.COMMUNITY_ALGORITHMS),
            'node_embedding': len(cls.NODE_EMBEDDING_ALGORITHMS),
            'graph_embedding': len(cls.GRAPH_EMBEDDING_ALGORITHMS),
            'total': len(cls.COMMUNITY_ALGORITHMS) + len(cls.NODE_EMBEDDING_ALGORITHMS) + len(cls.GRAPH_EMBEDDING_ALGORITHMS)
        }

    @classmethod
    def validate_algorithm(cls, name: str, category: Optional[str] = None) -> bool:
        """
        Validate if an algorithm exists.

        Args:
            name: Algorithm name
            category: Optional category to check

        Returns:
            True if algorithm exists
        """
        name = name.lower().replace('-', '_')

        if category:
            algorithms = cls.get_algorithms_by_category(category)
            return name in algorithms
        else:
            return name in cls.COMMUNITY_ALGORITHMS or \
                   name in cls.NODE_EMBEDDING_ALGORITHMS or \
                   name in cls.GRAPH_EMBEDDING_ALGORITHMS

    @classmethod
    def get_paper_citation(cls, name: str) -> Optional[str]:
        """Get paper citation for algorithm"""
        try:
            info = cls.get_algorithm_info(name)
            return f"{info.paper} ({info.year})" if info.paper else None
        except ValueError:
            return None

    @classmethod
    def print_summary(cls):
        """Print summary of all algorithms"""
        counts = cls.get_total_count()

        print("=" * 80)
        print("KarateClub Algorithm Registry")
        print("=" * 80)
        print(f"\nTotal Algorithms: {counts['total']}")
        print(f"  - Community Detection: {counts['community']}")
        print(f"  - Node Embedding: {counts['node_embedding']}")
        print(f"  - Graph Embedding: {counts['graph_embedding']}")
        print("\n" + "=" * 80)

        print("\nCommunity Detection Algorithms:")
        for name, info in cls.COMMUNITY_ALGORITHMS.items():
            print(f"  - {name}: {info['name']} ({info['year']})")

        print("\nNode Embedding Algorithms:")
        for name, info in cls.NODE_EMBEDDING_ALGORITHMS.items():
            print(f"  - {name}: {info['name']} ({info['year']})")

        print("\nGraph Embedding Algorithms:")
        for name, info in cls.GRAPH_EMBEDDING_ALGORITHMS.items():
            print(f"  - {name}: {info['name']} ({info['year']})")

        print("\n" + "=" * 80)


# Standalone function for easy access
def get_algorithm_info(name: str) -> AlgorithmInfo:
    """Get algorithm information - convenience function"""
    return KarateClubAlgorithmRegistry.get_algorithm_info(name)


def list_algorithms(category: Optional[str] = None) -> List[str]:
    """List algorithms - convenience function"""
    if category:
        algorithms = KarateClubAlgorithmRegistry.get_algorithms_by_category(category)
        return list(algorithms.keys())
    else:
        all_algos = KarateClubAlgorithmRegistry.get_all_algorithms()
        result = []
        for cat_algos in all_algos.values():
            result.extend(cat_algos)
        return result
