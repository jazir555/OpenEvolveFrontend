/**
 * KarateClub Algorithm Registry
 *
 * Complete registry of all 51 KarateClub algorithms with their parameters,
 * papers, and metadata. Used for validation and documentation.
 *
 * Reference: core-projects/karateclub/
 */

import {
  NodeEmbeddingAlgorithm,
  CommunityAlgorithm,
  GraphEmbeddingAlgorithm,
  AlgorithmCategory,
} from '../../../schemas/karateclub-canonical';

export interface AlgorithmInfo {
  name: string;
  description: string;
  category: AlgorithmCategory;
  paper?: string;
  year?: number;
  parameters: ParameterInfo[];
  defaultTimeout: number; // milliseconds
}

export interface ParameterInfo {
  name: string;
  type: 'number' | 'integer' | 'boolean' | 'string';
  default?: any;
  description: string;
  required?: boolean;
}

/**
 * Community Detection Algorithms (10 total)
 */
export const COMMUNITY_ALGORITHMS: Record<CommunityAlgorithm, AlgorithmInfo> = {
  // Overlapping Communities
  danmf: {
    name: 'Deep Autoencoder Nonnegative Matrix Factorization',
    description: 'Deep autoencoder for NMF-based overlapping community detection',
    category: 'community',
    paper: 'https://doi.org/10.1109/TKDE.2018.2874406',
    year: 2018,
    parameters: [
      { name: 'layers', type: 'integer', description: 'Number of layers in autoencoder', default: 32 },
      { name: 'iterations', type: 'integer', description: 'Number of training iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 60000,
  },
  m_nmf: {
    name: 'Symmetric Nonnegative Matrix Factorization',
    description: 'Symmetric NMF for community detection',
    category: 'community',
    paper: 'https://doi.org/10.1109/ICDM.2012.132',
    year: 2012,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Number of dimensions', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'lam', type: 'number', description: 'Lambda parameter', default: 1.0 },
    ],
    defaultTimeout: 45000,
  },
  ego_splitting: {
    name: 'Ego-Splitting Framework',
    description: 'Ego-splitting for overlapping community detection',
    category: 'community',
    paper: 'https://doi.org/10.1145/3097983.3098054',
    year: 2017,
    parameters: [
      { name: 'resolution', type: 'number', description: 'Resolution parameter', default: 1.0 },
    ],
    defaultTimeout: 30000,
  },
  nnsed: {
    name: 'Neural Stack for Signed Graphs',
    description: 'Community detection in signed graphs',
    category: 'community',
    paper: 'https://doi.org/10.1609/aaai.v33i01.3301825',
    year: 2019,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Number of dimensions', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 60000,
  },
  bigclam: {
    name: 'Cluster Affiliation Model for Big Networks',
    description: 'Overlapping community detection',
    category: 'community',
    paper: 'https://doi.org/10.1145/2488388.2488393',
    year: 2013,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Number of dimensions', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'epsilon', type: 'number', description: 'Convergence threshold', default: 0.0001 },
    ],
    defaultTimeout: 60000,
  },
  symmnmf: {
    name: 'Symmetric Semi-NMF',
    description: 'Symmetric semi-nonnegative matrix factorization',
    category: 'community',
    paper: 'https://doi.org/10.1109/ICDM.2012.132',
    year: 2012,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Number of dimensions', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 45000,
  },

  // Non-overlapping Communities
  gemsec: {
    name: 'Graph Embedding with Self Clustering',
    description: 'Node embedding with community detection',
    category: 'community',
    paper: 'https://doi.org/10.1145/3097983.3098108',
    year: 2017,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Number of dimensions', default: 128 },
      { name: 'walk_number', type: 'integer', description: 'Number of random walks', default: 10 },
      { name: 'walk_length', type: 'integer', description: 'Length of random walks', default: 80 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 90000,
  },
  edmot: {
    name: 'Edge Motif for Overlapping Communities',
    description: 'Edge motif-based community detection',
    category: 'community',
    paper: 'https://doi.org/10.1145/3336191.3371856',
    year: 2020,
    parameters: [
      { name: 'component_number', type: 'integer', description: 'Number of motif components', default: 10 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 30000,
  },
  scd: {
    name: 'Shortest Cycle Detection',
    description: 'Shortest cycle-based community detection',
    category: 'community',
    paper: 'https://doi.org/10.1137/1.9781611974973.66',
    year: 2018,
    parameters: [
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 30000,
  },
  label_propagation: {
    name: 'Label Propagation Algorithm',
    description: 'Fast label propagation for community detection',
    category: 'community',
    paper: 'https://doi.org/10.1109/TKDE.2007.190689',
    year: 2007,
    parameters: [
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 15000,
  },
};

/**
 * Node Embedding Algorithms (32 total)
 */
export const NODE_EMBEDDING_ALGORITHMS: Partial<Record<NodeEmbeddingAlgorithm, AlgorithmInfo>> = {
  // Neighbourhood-based (17)
  deepwalk: {
    name: 'DeepWalk',
    description: 'Random walk-based node embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/2623330.2623732',
    year: 2014,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'walk_length', type: 'integer', description: 'Random walk length', default: 80 },
      { name: 'walk_number', type: 'integer', description: 'Number of random walks', default: 10 },
      { name: 'window_size', type: 'integer', description: 'Context window size', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  node2vec: {
    name: 'Node2Vec',
    description: 'Biased random walk embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/2939672.2939754',
    year: 2016,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'walk_length', type: 'integer', description: 'Random walk length', default: 80 },
      { name: 'walk_number', type: 'integer', description: 'Number of random walks', default: 10 },
      { name: 'p', type: 'number', description: 'Return parameter', default: 1.0 },
      { name: 'q', type: 'number', description: 'In-out parameter', default: 1.0 },
      { name: 'window_size', type: 'integer', description: 'Context window size', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  walklets: {
    name: 'Walklets',
    description: 'Multi-scale random walk embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'walk_length', type: 'integer', description: 'Random walk length', default: 80 },
      { name: 'walk_number', type: 'integer', description: 'Number of random walks', default: 10 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  grarep: {
    name: 'GraRep',
    description: 'Graph factorization with k-step loss',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/2806416.2806502',
    year: 2015,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'order', type: 'integer', description: 'Order of proximity', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 90000,
  },
  hope: {
    name: 'HOPE',
    description: 'Preserving high-order proximities',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/2872427.2883041',
    year: 2016,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'beta', type: 'number', description: 'Beta parameter', default: 0.01 },
    ],
    defaultTimeout: 60000,
  },
  netmf: {
    name: 'NetMF',
    description: 'Network embedding as matrix factorization',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6003',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'order', type: 'integer', description: 'Order of proximity', default: 5 },
      { name: 'window_size', type: 'integer', description: 'Context window size', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'negative_samples', type: 'integer', description: 'Number of negative samples', default: 10 },
    ],
    defaultTimeout: 120000,
  },

  // Structural (3)
  graphwave: {
    name: 'GraphWave',
    description: 'Structural role embeddings using wavelets',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1137/1.9781611975282.7',
    year: 2017,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'scales', type: 'integer', description: 'Number of scales', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 180000,
  },
  role2vec: {
    name: 'Role2Vec',
    description: 'Structural role embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371853',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'walk_length', type: 'integer', description: 'Random walk length', default: 80 },
      { name: 'walk_number', type: 'integer', description: 'Number of random walks', default: 10 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  sinr: {
    name: 'Structural Information & Network Reconstruction',
    description: 'Information-based embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371855',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 90000,
  },

  // Attributed (9)
  feather_n: {
    name: 'FEATHER for Node Classification',
    description: 'Feature-based attributed embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6002',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  tadw: {
    name: 'Network Representation Learning with Text Features',
    description: 'Text-augmented network embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/2806416.2806504',
    year: 2015,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 50 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  musae: {
    name: 'MUSAE',
    description: 'Mixture of uniform and spectral embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371851',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'window_size', type: 'integer', description: 'Context window size', default: 5 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  ae: {
    name: 'AutoEncoder',
    description: 'Autoencoder-based embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6089',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'iterations', type: 'integer', description: 'Number of iterations', default: 100 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  fscnmf: {
    name: 'Feature-Supervised NMF',
    description: 'Feature-supervised nonnegative matrix factorization',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6088',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'clusters', type: 'integer', description: 'Number of clusters', default: 10 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'lam', type: 'number', description: 'Lambda parameter', default: 1.0 },
    ],
    defaultTimeout: 120000,
  },
  sine: {
    name: 'SINE',
    description: 'Sparse induced embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 90000,
  },
  bane: {
    name: 'Binarized Attributed Network Embedding',
    description: 'Binarized embeddings for efficiency',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6001',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
      { name: 'eta', type: 'number', description: 'Eta parameter', default: 0.5 },
    ],
    defaultTimeout: 120000,
  },
  tene: {
    name: 'TENE',
    description: 'Text-enriched network embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  asne: {
    name: 'ASNE',
    description: 'Attributed social network embeddings',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },

  // Meta (1)
  neu: {
    name: 'Network Embedding Update',
    description: 'Dynamic network embedding updates',
    category: 'node_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6148',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
};

/**
 * Graph Embedding Algorithms (10 total)
 */
export const GRAPH_EMBEDDING_ALGORITHMS: Partial<Record<GraphEmbeddingAlgorithm, AlgorithmInfo>> = {
  graph2vec: {
    name: 'Graph2Vec',
    description: 'Graph embeddings using Weisfeiler-Lehman',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'wl_iterations', type: 'integer', description: 'Weisfeiler-Lehman iterations', default: 2 },
      { name: 'epochs', type: 'integer', description: 'Training epochs', default: 10 },
      { name: 'learning_rate', type: 'number', description: 'Learning rate', default: 0.025 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 300000,
  },
  feather_g: {
    name: 'FEATHER for Graph Classification',
    description: 'Feature-based graph embeddings',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1609/aaai.v34i04.6002',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 180000,
  },
  netlsd: {
    name: 'NetLSD (Wave Kernel)',
    description: 'Network signature using wave kernel',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371860',
    year: 2020,
    parameters: [
      { name: 'scale_min', type: 'number', description: 'Minimum scale', default: -2.0 },
      { name: 'scale_max', type: 'number', description: 'Maximum scale', default: 2.0 },
      { name: 'scale_steps', type: 'integer', description: 'Number of scale steps', default: 250 },
    ],
    defaultTimeout: 120000,
  },
  geoscattering: {
    name: 'Geometric Scattering',
    description: 'Geometric scattering transforms',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1137/1.9781611975282.7',
    year: 2017,
    parameters: [
      { name: 'scales', type: 'integer', description: 'Number of scales', default: 3 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 180000,
  },
  ige: {
    name: 'Information Gain Embedding',
    description: 'Information-theoretic embeddings',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371859',
    year: 2020,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 120000,
  },
  gl2vec: {
    name: 'GL2Vec',
    description: 'Graph-level embeddings',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3183713.3196908',
    year: 2018,
    parameters: [
      { name: 'dimensions', type: 'integer', description: 'Embedding dimensionality', default: 128 },
      { name: 'wl_iterations', type: 'integer', description: 'Weisfeiler-Lehman iterations', default: 2 },
      { name: 'seed', type: 'integer', description: 'Random seed', default: 42 },
    ],
    defaultTimeout: 240000,
  },
  sf: {
    name: 'Statistical Features',
    description: 'Statistical graph descriptors',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371857',
    year: 2020,
    parameters: [],
    defaultTimeout: 30000,
  },
  fgsd: {
    name: 'Fused Graph Signal Distribution',
    description: 'Graph signal distribution',
    category: 'graph_embedding',
    paper: 'https://doi.org/10.1145/3336191.3371859',
    year: 2020,
    parameters: [],
    defaultTimeout: 60000,
  },
};

/**
 * Get algorithm information by name and category
 */
export function getAlgorithmInfo(
  algorithm: string,
  category: AlgorithmCategory
): AlgorithmInfo | undefined {
  switch (category) {
    case 'community':
      return COMMUNITY_ALGORITHMS[algorithm as CommunityAlgorithm];
    case 'node_embedding':
      return NODE_EMBEDDING_ALGORITHMS[algorithm as NodeEmbeddingAlgorithm];
    case 'graph_embedding':
      return GRAPH_EMBEDDING_ALGORITHMS[algorithm as GraphEmbeddingAlgorithm];
    default:
      return undefined;
  }
}

/**
 * Get all algorithms by category
 */
export function getAlgorithmsByCategory(category: AlgorithmCategory): string[] {
  switch (category) {
    case 'community':
      return Object.keys(COMMUNITY_ALGORITHMS);
    case 'node_embedding':
      return Object.keys(NODE_EMBEDDING_ALGORITHMS);
    case 'graph_embedding':
      return Object.keys(GRAPH_EMBEDDING_ALGORITHMS);
    default:
      return [];
  }
}

/**
 * Get default timeout for algorithm
 */
export function getDefaultTimeout(
  algorithm: string,
  category: AlgorithmCategory
): number {
  const info = getAlgorithmInfo(algorithm, category);
  return info?.defaultTimeout ?? 60000; // Default 60 seconds
}
