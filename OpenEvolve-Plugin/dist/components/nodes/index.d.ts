/**
 * OpenEvolve React Flow Nodes
 *
 * Export all node components and types for easy integration
 */
export { OpenEvolveNode } from './OpenEvolveNode';
export { DecompositionNodeComponent } from './DecompositionNodeComponent';
export { SolutionNodeComponent } from './SolutionNodeComponent';
export { VerificationNodeComponent } from './VerificationNodeComponent';
export type { OpenEvolveNodeData, NodeStatus, NodeType, NodeConfig, NodeResult, OpenEvolveFlowNode, BaseOpenEvolveNodeProps, SubProblem, DependencyInfo, AlternativeSolution, SolutionMetrics, QualityMetrics, Requirement, } from '../types/nodeTypes';
export { createOpenEvolveNode, createFlowNode, NODE_PRESETS, } from '../types/nodeTypes';
/**
 * Node Type Registry for React Flow
 *
 * Use this to register all OpenEvolve node types with React Flow
 */
export declare const OPENEVOLVE_NODE_TYPES: {
    readonly openevolve: "OpenEvolveNode";
    readonly decomposition: "DecompositionNodeComponent";
    readonly solution: "SolutionNodeComponent";
    readonly verification: "VerificationNodeComponent";
};
/**
 * Node Components Map
 *
 * Pass this to React Flow's nodeTypes prop
 */
export declare const openEvolveNodeComponents: {
    OpenEvolveNode: import('react').LazyExoticComponent<import('react').MemoExoticComponent<(props: import('@xyflow/react').NodeProps<import('.').OpenEvolveNodeData>) => import("react/jsx-runtime").JSX.Element>>;
    DecompositionNodeComponent: import('react').LazyExoticComponent<import('react').MemoExoticComponent<(props: import('@xyflow/react').NodeProps<import('../..').DecompositionNodeData>) => import("react/jsx-runtime").JSX.Element>>;
    SolutionNodeComponent: import('react').LazyExoticComponent<import('react').MemoExoticComponent<(props: import('@xyflow/react').NodeProps) => import("react/jsx-runtime").JSX.Element>>;
    VerificationNodeComponent: import('react').LazyExoticComponent<import('react').MemoExoticComponent<(props: import('@xyflow/react').NodeProps) => import("react/jsx-runtime").JSX.Element>>;
};
