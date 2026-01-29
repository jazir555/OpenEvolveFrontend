import { default as React } from 'react';
export interface DecompositionConfig {
    strategy: 'hierarchical' | 'semantic' | 'dependency_based' | 'temporal' | 'hybrid';
    maxDepth: number;
    maxSubProblems: number;
    granularity: 'fine' | 'medium' | 'coarse';
    minSubProblemSize: number;
    maxSubProblemSize: number;
    targetSubProblemSize: number;
    parallelDecomposition: boolean;
    maxParallelTasks: number;
    asyncDecomposition: boolean;
    pruningEnabled: boolean;
    pruningThreshold: number;
    similarityThreshold: number;
    mergeThreshold: number;
    semanticSimilarity: 'cosine' | 'jaccard' | 'euclidean' | 'manhattan';
    embeddingModel: string;
    clusteringAlgorithm: 'kmeans' | 'dbscan' | 'hierarchical' | 'spectral';
    dependencyDetection: 'static' | 'dynamic' | 'hybrid';
    circularDependencyHandling: 'break' | 'merge' | 'error';
    dependencyVisualization: boolean;
    timeHorizon: number;
    timeGranularity: 'seconds' | 'minutes' | 'hours' | 'days';
    temporalDependencies: boolean;
    cohesivenessTarget: number;
    couplingLimit: number;
    complexityThreshold: number;
    validateDecomposition: boolean;
    testDecomposition: boolean;
    feedbackLoop: boolean;
    adaptiveDecomposition: boolean;
    learningEnabled: boolean;
    historicalData: boolean;
}
interface DecompositionConfigPanelProps {
    config: DecompositionConfig;
    onConfigChange: (config: DecompositionConfig) => void;
}
export declare const DecompositionConfigPanel: React.FC<DecompositionConfigPanelProps>;
export default DecompositionConfigPanel;
