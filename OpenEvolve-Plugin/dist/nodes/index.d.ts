import { OpenEvolveBaseNode } from './BaseNode';
import { DecompositionNode } from './DecompositionNode';
import { SolutionNode } from './SolutionNode';
import { VerificationNode } from './VerificationNode';
export { OpenEvolveBaseNode, NodeExecutionError, NodeStatus, } from './BaseNode';
export type { NodeConfig, NodeInputs, NodeResult, ExecutionContext, ParameterSchema, ValidationError, NodeMetrics, ErrorDetails, } from './BaseNode';
export { OpenEvolveBaseNode as OpenEvolveNode, } from './OpenEvolveBaseNode';
export { NodeRegistry, registerNodes, createNodeFromConfig, type NodeClass, type NodeMetadata, type NodeCreationConfig, type ValidationResult, type RegistrationOptions, type RegistryStats, } from './registry';
export { DecompositionNode, type DecompositionStrategy, type SubProblem, type DependencyGraph, type DecompositionNodeConfig, type DecompositionResult, } from './DecompositionNode';
export { SolutionNode, type SolutionStrategy, type Solution, type ConvergenceMetrics, type SolutionNodeConfig, type SolutionResult, } from './SolutionNode';
export { VerificationNode, type VerificationCheck, type CheckResult, type VerificationReport, type VerificationNodeConfig, } from './VerificationNode';
export { EvolutionNode, type EvolutionMode, type EvolutionResult, type EvolutionNodeConfig, } from './EvolutionNode';
export { AdversarialNode, type AttackMode, type AdversarialTestResult, type AdversarialNodeConfig, } from './AdversarialNode';
export { KnowledgeQueryNode, type QueryType, type KnowledgeQueryResult, type KnowledgeQueryNodeConfig, } from './KnowledgeQueryNode';
export { LeanAideNode, type VerificationResult, type LeanAideNodeConfig, } from './LeanAIDENode';
export { CrewAINode, type CrewAITaskType, type CrewAIResult, type CrewAINodeConfig, } from './CrewAINode';
export { MDAPNode, type MDAPStrategy, type MDAPNodeConfig, } from './MDAPNode';
export { MAKERNode, type ContentType, type MAKERResult, type MAKERNodeConfig, } from './MAKERNode';
export { ROMANode, type ROMAMode, type AgentRole, type ROMAResult, type ReasoningStep, type AgentVote, type ROMASubtask, type ROMANodeConfig, } from './ROMANode';
export { InventionNode, type InventionDomain, type PlanningStage, type DetailLevel, type InventionResult, type InventionPlan, type PriorArtAnalysis, type FeasibilityAnalysis, type InventionNodeConfig, } from './InventionNode';
/**
 * Default export - Registry and all nodes
 *
 * @example
 * ```typescript
 * import * as OpenEvolveNodes from '@openevolve/bubblelab-plugin/nodes';
 *
 * // Access registry
 * const registry = OpenEvolveNodes.NodeRegistry;
 *
 * // Create a node
 * const node = registry.create('Decomposition', 'node-1');
 * ```
 */
declare const _default: {
    NodeRegistry: any;
    OpenEvolveBaseNode: typeof OpenEvolveBaseNode;
    DecompositionNode: typeof DecompositionNode;
    SolutionNode: typeof SolutionNode;
    VerificationNode: typeof VerificationNode;
};
export default _default;
/**
 * Registry auto-export
 *
 * Allows direct import of registry without specifying class name.
 *
 * @example
 * ```typescript
 * import { Registry } from '@openevolve/bubblelab-plugin/nodes';
 * const node = Registry.create('Decomposition', 'node-1');
 * ```
 */
export declare const Registry: any;
/**
 * Get all registered node types
 *
 * Convenience function to list all available nodes.
 *
 * @returns Array of node metadata
 *
 * @example
 * ```typescript
 * import { getAllNodeTypes } from '@openevolve/bubblelab-plugin/nodes';
 *
 * const nodes = getAllNodeTypes();
 * nodes.forEach(node => {
 *   console.log(`${node.displayName}: ${node.description}`);
 * });
 * ```
 */
export declare function getAllNodeTypes(): Array<{
    type: string;
    metadata: import('./registry').NodeMetadata;
}>;
/**
 * Get node by type
 *
 * Convenience function to create a node by type with metadata lookup.
 *
 * @param type - Node type
 * @param id - Node ID (optional, auto-generated if not provided)
 * @param config - Node configuration (optional)
 * @returns Node instance or null
 *
 * @example
 * ```typescript
 * import { getNode } from '@openevolve/bubblelab-plugin/nodes';
 *
 * const node = getNode('Decomposition', 'my-node', {
 *   config: { strategy: 'semantic' }
 * });
 * ```
 */
export declare function getNode(type: string, id?: string, config?: import('./registry').NodeCreationConfig): import('./BaseNode').OpenEvolveBaseNode | null;
/**
 * Search for nodes by query
 *
 * @param query - Search query
 * @returns Array of matching nodes
 */
export declare function searchNodes(query: string): Array<{
    type: string;
    metadata: import('./registry').NodeMetadata;
}>;
/**
 * Get nodes by category
 *
 * @param category - Category name
 * @returns Array of nodes in category
 */
export declare function getNodesByCategory(category: string): Array<{
    type: string;
    metadata: import('./registry').NodeMetadata;
}>;
/**
 * Get all categories
 *
 * @returns Array of category names
 */
export declare function getCategories(): string[];
/**
 * Validate node configuration
 *
 * @param type - Node type
 * @param config - Configuration to validate
 * @returns Validation result
 */
export declare function validateNodeConfig(type: string, config: Record<string, any>): import('./registry').ValidationResult;
/**
 * Get registry statistics
 *
 * @returns Registry stats
 */
export declare function getRegistryStats(): import('./registry').RegistryStats;
