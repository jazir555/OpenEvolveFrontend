/**
 * KarateClub Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for KarateClub graph ML operations.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for KarateClub data in the glue layer.
 * Do not pass raw KarateClub API responses between services.
 *
 * KarateClub provides 51 algorithms across 3 categories:
 * - Community Detection (10 algorithms)
 * - Node Embedding (32 algorithms)
 * - Graph Embedding (10 algorithms)
 */
import { z } from 'zod';
/**
 * Algorithm Categories
 */
export declare const AlgorithmCategory: z.ZodEnum<["community", "node_embedding", "graph_embedding"]>;
export type AlgorithmCategory = z.infer<typeof AlgorithmCategory>;
/**
 * Supported Node Embedding Algorithms (32)
 */
export declare const NodeEmbeddingAlgorithm: z.ZodEnum<["deepwalk", "node2vec", "walklets", "grarep", "hope", "netmf", "boostne", "randne", "nodesketch", "diff2vec", "sociodim", "glee", "laplacian_eigenmaps", "nmf_admm", "line", "graphwave", "role2vec", "sinr", "feather_n", "tadw", "musae", "ae", "fscnmf", "sine", "bane", "tene", "asne", "neu"]>;
export type NodeEmbeddingAlgorithm = z.infer<typeof NodeEmbeddingAlgorithm>;
/**
 * Supported Community Detection Algorithms (10)
 */
export declare const CommunityAlgorithm: z.ZodEnum<["danmf", "m_nmf", "ego_splitting", "nnsed", "bigclam", "symmnmf", "gemsec", "edmot", "scd", "label_propagation"]>;
export type CommunityAlgorithm = z.infer<typeof CommunityAlgorithm>;
/**
 * Supported Graph Embedding Algorithms (10)
 */
export declare const GraphEmbeddingAlgorithm: z.ZodEnum<["graph2vec", "feather_g", "netlsd", "geoscattering", "wavelet_characteristic", "ige", "ldp", "gl2vec", "sf", "fgsd"]>;
export type GraphEmbeddingAlgorithm = z.infer<typeof GraphEmbeddingAlgorithm>;
/**
 * Graph Structure Schema
 */
export declare const GraphStructure: z.ZodObject<{
    nodes: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        features: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        label: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        metadata?: Record<string, any> | undefined;
        label?: string | undefined;
        features?: number[] | undefined;
    }, {
        id: string;
        metadata?: Record<string, any> | undefined;
        label?: string | undefined;
        features?: number[] | undefined;
    }>, "many">;
    edges: z.ZodArray<z.ZodObject<{
        source: z.ZodString;
        target: z.ZodString;
        weight: z.ZodOptional<z.ZodNumber>;
        attributes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        source: string;
        target: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
    }, {
        source: string;
        target: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
    }>, "many">;
    directed: z.ZodDefault<z.ZodBoolean>;
    weighted: z.ZodDefault<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    edges: {
        source: string;
        target: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
    }[];
    nodes: {
        id: string;
        metadata?: Record<string, any> | undefined;
        label?: string | undefined;
        features?: number[] | undefined;
    }[];
    weighted: boolean;
    directed: boolean;
}, {
    edges: {
        source: string;
        target: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
    }[];
    nodes: {
        id: string;
        metadata?: Record<string, any> | undefined;
        label?: string | undefined;
        features?: number[] | undefined;
    }[];
    weighted?: boolean | undefined;
    directed?: boolean | undefined;
}>;
export type GraphStructure = z.infer<typeof GraphStructure>;
/**
 * Node Embedding Request Schema
 */
export declare const NodeEmbeddingRequest: z.ZodObject<{
    algorithm: z.ZodEnum<["deepwalk", "node2vec", "walklets", "grarep", "hope", "netmf", "boostne", "randne", "nodesketch", "diff2vec", "sociodim", "glee", "laplacian_eigenmaps", "nmf_admm", "line", "graphwave", "role2vec", "sinr", "feather_n", "tadw", "musae", "ae", "fscnmf", "sine", "bane", "tene", "asne", "neu"]>;
    graph: z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            features: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            label: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            attributes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }>, "many">;
        directed: z.ZodDefault<z.ZodBoolean>;
        weighted: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }>;
    parameters: z.ZodOptional<z.ZodObject<{
        dimensions: z.ZodDefault<z.ZodNumber>;
        walk_length: z.ZodOptional<z.ZodNumber>;
        walk_number: z.ZodOptional<z.ZodNumber>;
        window_size: z.ZodOptional<z.ZodNumber>;
        p: z.ZodOptional<z.ZodNumber>;
        q: z.ZodOptional<z.ZodNumber>;
        epochs: z.ZodOptional<z.ZodNumber>;
        seed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        dimensions: number;
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    }, {
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        dimensions?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    parameters?: {
        dimensions: number;
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    parameters?: {
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        dimensions?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    } | undefined;
}>;
export type NodeEmbeddingRequest = z.infer<typeof NodeEmbeddingRequest>;
/**
 * Node Embedding Response Schema
 */
export declare const NodeEmbeddingResponse: z.ZodObject<{
    success: z.ZodBoolean;
    embeddings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNumber, "many">>>;
    dimensions: z.ZodNumber;
    algorithm: z.ZodEnum<["deepwalk", "node2vec", "walklets", "grarep", "hope", "netmf", "boostne", "randne", "nodesketch", "diff2vec", "sociodim", "glee", "laplacian_eigenmaps", "nmf_admm", "line", "graphwave", "role2vec", "sinr", "feather_n", "tadw", "musae", "ae", "fscnmf", "sine", "bane", "tene", "asne", "neu"]>;
    metadata: z.ZodObject<{
        num_nodes: z.ZodNumber;
        training_time_ms: z.ZodNumber;
        convergence: z.ZodOptional<z.ZodBoolean>;
        epochs_completed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        num_nodes: number;
        training_time_ms: number;
        convergence?: boolean | undefined;
        epochs_completed?: number | undefined;
    }, {
        num_nodes: number;
        training_time_ms: number;
        convergence?: boolean | undefined;
        epochs_completed?: number | undefined;
    }>;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    metadata: {
        num_nodes: number;
        training_time_ms: number;
        convergence?: boolean | undefined;
        epochs_completed?: number | undefined;
    };
    success: boolean;
    dimensions: number;
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    error?: string | undefined;
    embeddings?: Record<string, number[]> | undefined;
}, {
    timestamp: string;
    metadata: {
        num_nodes: number;
        training_time_ms: number;
        convergence?: boolean | undefined;
        epochs_completed?: number | undefined;
    };
    success: boolean;
    dimensions: number;
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    error?: string | undefined;
    embeddings?: Record<string, number[]> | undefined;
}>;
export type NodeEmbeddingResponse = z.infer<typeof NodeEmbeddingResponse>;
/**
 * Community Detection Request Schema
 */
export declare const CommunityDetectionRequest: z.ZodObject<{
    algorithm: z.ZodEnum<["danmf", "m_nmf", "ego_splitting", "nnsed", "bigclam", "symmnmf", "gemsec", "edmot", "scd", "label_propagation"]>;
    graph: z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            features: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            label: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            attributes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }>, "many">;
        directed: z.ZodDefault<z.ZodBoolean>;
        weighted: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }>;
    parameters: z.ZodOptional<z.ZodObject<{
        resolution: z.ZodOptional<z.ZodNumber>;
        iterations: z.ZodOptional<z.ZodNumber>;
        seed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    }, {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    parameters?: {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    parameters?: {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    } | undefined;
}>;
export type CommunityDetectionRequest = z.infer<typeof CommunityDetectionRequest>;
/**
 * Community Detection Response Schema
 */
export declare const CommunityDetectionResponse: z.ZodObject<{
    success: z.ZodBoolean;
    memberships: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    overlapping_memberships: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNumber, "many">>>;
    num_communities: z.ZodOptional<z.ZodNumber>;
    community_sizes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    algorithm: z.ZodEnum<["danmf", "m_nmf", "ego_splitting", "nnsed", "bigclam", "symmnmf", "gemsec", "edmot", "scd", "label_propagation"]>;
    metadata: z.ZodObject<{
        detection_time_ms: z.ZodNumber;
        modularity: z.ZodOptional<z.ZodNumber>;
        coverage: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        detection_time_ms: number;
        modularity?: number | undefined;
        coverage?: number | undefined;
    }, {
        detection_time_ms: number;
        modularity?: number | undefined;
        coverage?: number | undefined;
    }>;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    metadata: {
        detection_time_ms: number;
        modularity?: number | undefined;
        coverage?: number | undefined;
    };
    success: boolean;
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    error?: string | undefined;
    num_communities?: number | undefined;
    memberships?: Record<string, number> | undefined;
    overlapping_memberships?: Record<string, number[]> | undefined;
    community_sizes?: Record<string, number> | undefined;
}, {
    timestamp: string;
    metadata: {
        detection_time_ms: number;
        modularity?: number | undefined;
        coverage?: number | undefined;
    };
    success: boolean;
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    error?: string | undefined;
    num_communities?: number | undefined;
    memberships?: Record<string, number> | undefined;
    overlapping_memberships?: Record<string, number[]> | undefined;
    community_sizes?: Record<string, number> | undefined;
}>;
export type CommunityDetectionResponse = z.infer<typeof CommunityDetectionResponse>;
/**
 * Graph Embedding Request Schema
 */
export declare const GraphEmbeddingRequest: z.ZodObject<{
    algorithm: z.ZodEnum<["graph2vec", "feather_g", "netlsd", "geoscattering", "wavelet_characteristic", "ige", "ldp", "gl2vec", "sf", "fgsd"]>;
    graphs: z.ZodArray<z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            features: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            label: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            attributes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }>, "many">;
        directed: z.ZodDefault<z.ZodBoolean>;
        weighted: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }>, "many">;
    parameters: z.ZodOptional<z.ZodObject<{
        dimensions: z.ZodDefault<z.ZodNumber>;
        wl_iterations: z.ZodOptional<z.ZodNumber>;
        epochs: z.ZodOptional<z.ZodNumber>;
        learning_rate: z.ZodOptional<z.ZodNumber>;
        seed: z.ZodOptional<z.ZodNumber>;
        scales: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        dimensions: number;
        seed?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    }, {
        seed?: number | undefined;
        dimensions?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    graphs: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }[];
    correlation_id?: string | undefined;
    parameters?: {
        dimensions: number;
        seed?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    graphs: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }[];
    correlation_id?: string | undefined;
    parameters?: {
        seed?: number | undefined;
        dimensions?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    } | undefined;
}>;
export type GraphEmbeddingRequest = z.infer<typeof GraphEmbeddingRequest>;
/**
 * Graph Embedding Response Schema
 */
export declare const GraphEmbeddingResponse: z.ZodObject<{
    success: z.ZodBoolean;
    embeddings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNumber, "many">>>;
    dimensions: z.ZodNumber;
    algorithm: z.ZodEnum<["graph2vec", "feather_g", "netlsd", "geoscattering", "wavelet_characteristic", "ige", "ldp", "gl2vec", "sf", "fgsd"]>;
    metadata: z.ZodObject<{
        num_graphs: z.ZodNumber;
        training_time_ms: z.ZodNumber;
        epochs_completed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        training_time_ms: number;
        num_graphs: number;
        epochs_completed?: number | undefined;
    }, {
        training_time_ms: number;
        num_graphs: number;
        epochs_completed?: number | undefined;
    }>;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    metadata: {
        training_time_ms: number;
        num_graphs: number;
        epochs_completed?: number | undefined;
    };
    success: boolean;
    dimensions: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    correlation_id?: string | undefined;
    error?: string | undefined;
    embeddings?: Record<string, number[]> | undefined;
}, {
    timestamp: string;
    metadata: {
        training_time_ms: number;
        num_graphs: number;
        epochs_completed?: number | undefined;
    };
    success: boolean;
    dimensions: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    correlation_id?: string | undefined;
    error?: string | undefined;
    embeddings?: Record<string, number[]> | undefined;
}>;
export type GraphEmbeddingResponse = z.infer<typeof GraphEmbeddingResponse>;
/**
 * Combined Graph Analysis Request
 */
export declare const GraphAnalysisRequest: z.ZodObject<{
    graph: z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            features: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            label: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }, {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            attributes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }, {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }>, "many">;
        directed: z.ZodDefault<z.ZodBoolean>;
        weighted: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }, {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }>;
    analyses: z.ZodArray<z.ZodEnum<["node_embeddings", "community_detection", "graph_statistics", "centrality"]>, "many">;
    node_embedding_algorithm: z.ZodOptional<z.ZodEnum<["deepwalk", "node2vec", "walklets", "grarep", "hope", "netmf", "boostne", "randne", "nodesketch", "diff2vec", "sociodim", "glee", "laplacian_eigenmaps", "nmf_admm", "line", "graphwave", "role2vec", "sinr", "feather_n", "tadw", "musae", "ae", "fscnmf", "sine", "bane", "tene", "asne", "neu"]>>;
    community_algorithm: z.ZodOptional<z.ZodEnum<["danmf", "m_nmf", "ego_splitting", "nnsed", "bigclam", "symmnmf", "gemsec", "edmot", "scd", "label_propagation"]>>;
    parameters: z.ZodOptional<z.ZodObject<{
        embedding_dimensions: z.ZodDefault<z.ZodNumber>;
        top_k_nodes: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        embedding_dimensions: number;
        top_k_nodes: number;
    }, {
        embedding_dimensions?: number | undefined;
        top_k_nodes?: number | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    analyses: ("community_detection" | "node_embeddings" | "graph_statistics" | "centrality")[];
    correlation_id?: string | undefined;
    parameters?: {
        embedding_dimensions: number;
        top_k_nodes: number;
    } | undefined;
    node_embedding_algorithm?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
    community_algorithm?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    analyses: ("community_detection" | "node_embeddings" | "graph_statistics" | "centrality")[];
    correlation_id?: string | undefined;
    parameters?: {
        embedding_dimensions?: number | undefined;
        top_k_nodes?: number | undefined;
    } | undefined;
    node_embedding_algorithm?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
    community_algorithm?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
}>;
export type GraphAnalysisRequest = z.infer<typeof GraphAnalysisRequest>;
/**
 * Combined Graph Analysis Response
 */
export declare const GraphAnalysisResponse: z.ZodObject<{
    success: z.ZodBoolean;
    results: z.ZodOptional<z.ZodObject<{
        node_embeddings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNumber, "many">>>;
        communities: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
        graph_statistics: z.ZodOptional<z.ZodObject<{
            num_nodes: z.ZodNumber;
            num_edges: z.ZodNumber;
            density: z.ZodNumber;
            is_connected: z.ZodBoolean;
            avg_degree: z.ZodOptional<z.ZodNumber>;
            avg_clustering: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        }, {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        }>>;
        centrality: z.ZodOptional<z.ZodObject<{
            top_degree: z.ZodOptional<z.ZodArray<z.ZodObject<{
                node_id: z.ZodString;
                score: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                score: number;
                node_id: string;
            }, {
                score: number;
                node_id: string;
            }>, "many">>;
            top_betweenness: z.ZodOptional<z.ZodArray<z.ZodObject<{
                node_id: z.ZodString;
                score: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                score: number;
                node_id: string;
            }, {
                score: number;
                node_id: string;
            }>, "many">>;
            top_pagerank: z.ZodOptional<z.ZodArray<z.ZodObject<{
                node_id: z.ZodString;
                score: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                score: number;
                node_id: string;
            }, {
                score: number;
                node_id: string;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        }, {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        node_embeddings?: Record<string, number[]> | undefined;
        graph_statistics?: {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        } | undefined;
        centrality?: {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        } | undefined;
        communities?: Record<string, number> | undefined;
    }, {
        node_embeddings?: Record<string, number[]> | undefined;
        graph_statistics?: {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        } | undefined;
        centrality?: {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        } | undefined;
        communities?: Record<string, number> | undefined;
    }>>;
    algorithms_used: z.ZodOptional<z.ZodObject<{
        node_embedding: z.ZodOptional<z.ZodEnum<["deepwalk", "node2vec", "walklets", "grarep", "hope", "netmf", "boostne", "randne", "nodesketch", "diff2vec", "sociodim", "glee", "laplacian_eigenmaps", "nmf_admm", "line", "graphwave", "role2vec", "sinr", "feather_n", "tadw", "musae", "ae", "fscnmf", "sine", "bane", "tene", "asne", "neu"]>>;
        community_detection: z.ZodOptional<z.ZodEnum<["danmf", "m_nmf", "ego_splitting", "nnsed", "bigclam", "symmnmf", "gemsec", "edmot", "scd", "label_propagation"]>>;
    }, "strip", z.ZodTypeAny, {
        node_embedding?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
        community_detection?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
    }, {
        node_embedding?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
        community_detection?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
    }>>;
    execution_time_ms: z.ZodNumber;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    execution_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    results?: {
        node_embeddings?: Record<string, number[]> | undefined;
        graph_statistics?: {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        } | undefined;
        centrality?: {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        } | undefined;
        communities?: Record<string, number> | undefined;
    } | undefined;
    algorithms_used?: {
        node_embedding?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
        community_detection?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
    } | undefined;
}, {
    timestamp: string;
    success: boolean;
    execution_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    results?: {
        node_embeddings?: Record<string, number[]> | undefined;
        graph_statistics?: {
            num_nodes: number;
            num_edges: number;
            density: number;
            is_connected: boolean;
            avg_degree?: number | undefined;
            avg_clustering?: number | undefined;
        } | undefined;
        centrality?: {
            top_degree?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_betweenness?: {
                score: number;
                node_id: string;
            }[] | undefined;
            top_pagerank?: {
                score: number;
                node_id: string;
            }[] | undefined;
        } | undefined;
        communities?: Record<string, number> | undefined;
    } | undefined;
    algorithms_used?: {
        node_embedding?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
        community_detection?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
    } | undefined;
}>;
export type GraphAnalysisResponse = z.infer<typeof GraphAnalysisResponse>;
/**
 * Validation helper functions
 */
export declare function validateNodeEmbeddingRequest(data: unknown): z.SafeParseReturnType<{
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    parameters?: {
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        dimensions?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    algorithm: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm";
    correlation_id?: string | undefined;
    parameters?: {
        dimensions: number;
        p?: number | undefined;
        q?: number | undefined;
        seed?: number | undefined;
        walk_number?: number | undefined;
        walk_length?: number | undefined;
        window_size?: number | undefined;
        epochs?: number | undefined;
    } | undefined;
}>;
export declare function validateCommunityDetectionRequest(data: unknown): z.SafeParseReturnType<{
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    parameters?: {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    algorithm: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation";
    correlation_id?: string | undefined;
    parameters?: {
        iterations?: number | undefined;
        seed?: number | undefined;
        resolution?: number | undefined;
    } | undefined;
}>;
export declare function validateGraphEmbeddingRequest(data: unknown): z.SafeParseReturnType<{
    timeout_ms: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    graphs: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    }[];
    correlation_id?: string | undefined;
    parameters?: {
        seed?: number | undefined;
        dimensions?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    algorithm: "graph2vec" | "feather_g" | "netlsd" | "geoscattering" | "ige" | "gl2vec" | "sf" | "fgsd" | "wavelet_characteristic" | "ldp";
    graphs: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    }[];
    correlation_id?: string | undefined;
    parameters?: {
        dimensions: number;
        seed?: number | undefined;
        scales?: number | undefined;
        wl_iterations?: number | undefined;
        epochs?: number | undefined;
        learning_rate?: number | undefined;
    } | undefined;
}>;
export declare function validateGraphAnalysisRequest(data: unknown): z.SafeParseReturnType<{
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted?: boolean | undefined;
        directed?: boolean | undefined;
    };
    analyses: ("community_detection" | "node_embeddings" | "graph_statistics" | "centrality")[];
    correlation_id?: string | undefined;
    parameters?: {
        embedding_dimensions?: number | undefined;
        top_k_nodes?: number | undefined;
    } | undefined;
    node_embedding_algorithm?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
    community_algorithm?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
}, {
    timeout_ms: number;
    graph: {
        edges: {
            source: string;
            target: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        nodes: {
            id: string;
            metadata?: Record<string, any> | undefined;
            label?: string | undefined;
            features?: number[] | undefined;
        }[];
        weighted: boolean;
        directed: boolean;
    };
    analyses: ("community_detection" | "node_embeddings" | "graph_statistics" | "centrality")[];
    correlation_id?: string | undefined;
    parameters?: {
        embedding_dimensions: number;
        top_k_nodes: number;
    } | undefined;
    node_embedding_algorithm?: "line" | "deepwalk" | "node2vec" | "walklets" | "grarep" | "hope" | "netmf" | "graphwave" | "role2vec" | "sinr" | "feather_n" | "tadw" | "musae" | "ae" | "fscnmf" | "sine" | "bane" | "tene" | "asne" | "neu" | "boostne" | "randne" | "nodesketch" | "diff2vec" | "sociodim" | "glee" | "laplacian_eigenmaps" | "nmf_admm" | undefined;
    community_algorithm?: "danmf" | "m_nmf" | "ego_splitting" | "nnsed" | "bigclam" | "symmnmf" | "gemsec" | "edmot" | "scd" | "label_propagation" | undefined;
}>;
//# sourceMappingURL=karateclub-canonical.d.ts.map