// @ts-nocheck
/**
 * Nodes Module - Central Export Point
 *
 * Exports all node classes, base classes, registry, and related utilities
 * for the OpenEvolve BubbleLab plugin.
 *
 * @module nodes
 * @version 1.0.0
 */

// Auto-register all core nodes
import './init';

import { NodeRegistry } from './registry';
import { OpenEvolveBaseNode } from './BaseNode';
import { DecompositionNode } from './DecompositionNode';
import { SolutionNode } from './SolutionNode';
import { VerificationNode } from './VerificationNode';
import { SubProblemNode } from './SubProblemNode';
import { GauntletNode } from './GauntletNode';
import { AssemblyNode } from './AssemblyNode';
import { OutputNode } from './OutputNode';
import { KnowledgeExtractionNode } from './KnowledgeExtractionNode';
import { ResearchQuestNode } from './ResearchQuestNode';

// ==========================================================================
// Export Base Classes
// ==========================================================================

export {
  OpenEvolveBaseNode,
  NodeExecutionError,
  NodeStatus,
} from './BaseNode';

export type {
  NodeConfig,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ParameterSchema,
  ValidationError,
  NodeMetrics,
  ErrorDetails,
} from './BaseNode';

// Re-export from OpenEvolveBaseNode for backward compatibility
export {
  OpenEvolveBaseNode as OpenEvolveNode,
} from './OpenEvolveBaseNode';

// ==========================================================================
// Export Node Registry
// ==========================================================================

export {
  NodeRegistry,
  registerNodes,
  createNodeFromConfig,
  type NodeClass,
  type NodeMetadata,
  type NodeCreationConfig,
  type ValidationResult,
  type RegistrationOptions,
  type RegistryStats,
} from './registry';

// ==========================================================================
// Export Node Implementations
// ==========================================================================

// Decomposition Node
export {
  DecompositionNode,
  type DecompositionStrategy,
  type SubProblem,
  type DependencyGraph,
  type DecompositionNodeConfig,
  type DecompositionResult,
} from './DecompositionNode';

// Solution Node
export {
  SolutionNode,
  type SolutionStrategy,
  type Solution,
  type ConvergenceMetrics,
  type SolutionNodeConfig,
  type SolutionResult,
} from './SolutionNode';

// Verification Node
export {
  VerificationNode,
  type VerificationCheck,
  type CheckResult,
  type VerificationReport,
  type VerificationNodeConfig,
} from './VerificationNode';

// Evolution Node
export {
  EvolutionNode,
  type EvolutionMode,
  type EvolutionResult,
  type EvolutionNodeConfig,
} from './EvolutionNode';

// Adversarial Node
export {
  AdversarialNode,
  type AttackMode,
  type AdversarialTestResult,
  type AdversarialNodeConfig,
} from './AdversarialNode';

// Knowledge Query Node
export {
  KnowledgeQueryNode,
  type QueryType,
  type KnowledgeQueryResult,
  type KnowledgeQueryNodeConfig,
} from './KnowledgeQueryNode';

// LeanAIDE Node
export {
  LeanAideNode,
  type VerificationResult,
  type LeanAideNodeConfig,
} from './LeanAIDENode';

// CrewAI Node
export {
  CrewAINode,
  type CrewAITaskType,
  type CrewAIResult,
  type CrewAINodeConfig,
} from './CrewAINode';

// MDAP Node
export {
  MDAPNode,
  type MDAPStrategy,
  type MDAPNodeConfig,
} from './MDAPNode';

// MAKER Node
export {
  MAKERNode,
  type ContentType,
  type MAKERResult,
  type MAKERNodeConfig,
} from './MAKERNode';

// ROMA Node
export {
  ROMANode,
  type ROMAMode,
  type AgentRole,
  type ROMAResult,
  type ReasoningStep,
  type AgentVote,
  type ROMASubtask,
  type ROMANodeConfig,
} from './ROMANode';

// Invention Node
export {
  InventionNode,
  type InventionDomain,
  type PlanningStage,
  type DetailLevel,
  type InventionResult,
  type InventionPlan,
  type PriorArtAnalysis,
  type FeasibilityAnalysis,
  type InventionNodeConfig,
} from './InventionNode';

// Sub-Problem Node
export {
  SubProblemNode,
  type SubProblemNodeConfig,
  type SubProblemPlanStep,
  type SubProblemResult,
} from './SubProblemNode';

// Gauntlet Node
export {
  GauntletNode,
  type GauntletNodeConfig,
  type GauntletStageConfig,
  type CriteriaResult,
  type StageResult,
  type GauntletResult,
} from './GauntletNode';

// Assembly Node
export {
  AssemblyNode,
  type AssemblyStrategy,
  type IntegrationMethod,
  type ConflictResolutionStrategy,
  type AssemblyNodeConfig,
  type AssemblyResult,
} from './AssemblyNode';

// Output Node
export {
  OutputNode,
  type OutputFormat,
  type OutputDestination,
  type OutputNodeConfig,
  type OutputResult,
} from './OutputNode';

// Knowledge Extraction Node
export {
  KnowledgeExtractionNode,
  type KnowledgeSourceType,
  type ExtractionMethod,
  type KnowledgeSource,
  type KnowledgeEntity,
  type KnowledgeRelationship,
  type KnowledgeExtractionResult,
  type KnowledgeExtractionNodeConfig,
} from './KnowledgeExtractionNode';

// Research Quest Node
export {
  ResearchQuestNode,
  type ResearchQuestStage,
  type ResearchQuestParameters,
  type ResearchQuestNodeConfig,
  type ResearchQuestResult,
} from './ResearchQuestNode';

// PyGraphistry Node
export {
  PyGraphistryNode,
  type PyGraphistryLayout,
  type PyGraphistryClusteringMethod,
  type GraphNode,
  type GraphEdge,
  type PyGraphistryNodeConfig,
  type PyGraphistryResult,
} from './PyGraphistryNode';

// ==========================================================================
// Re-export for Convenience
// ==========================================================================

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
export default {
  NodeRegistry,
  OpenEvolveBaseNode,
  DecompositionNode,
  SolutionNode,
  VerificationNode,
  SubProblemNode,
  GauntletNode,
  AssemblyNode,
  OutputNode,
  KnowledgeExtractionNode,
  ResearchQuestNode,
  PyGraphistryNode,
};

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
export const Registry = NodeRegistry;

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
export function getAllNodeTypes(): Array<{ type: string; metadata: import('./registry').NodeMetadata }> {
  return NodeRegistry.listAll();
}

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
export function getNode(
  type: string,
  id?: string,
  config?: import('./registry').NodeCreationConfig
): import('./BaseNode').OpenEvolveBaseNode | null {
  return NodeRegistry.create(type, id, config);
}

/**
 * Search for nodes by query
 *
 * @param query - Search query
 * @returns Array of matching nodes
 */
export function searchNodes(
  query: string
): Array<{ type: string; metadata: import('./registry').NodeMetadata }> {
  return NodeRegistry.search(query);
}

/**
 * Get nodes by category
 *
 * @param category - Category name
 * @returns Array of nodes in category
 */
export function getNodesByCategory(
  category: string
): Array<{ type: string; metadata: import('./registry').NodeMetadata }> {
  return NodeRegistry.getByCategory(category);
}

/**
 * Get all categories
 *
 * @returns Array of category names
 */
export function getCategories(): string[] {
  return NodeRegistry.getCategories();
}

/**
 * Validate node configuration
 *
 * @param type - Node type
 * @param config - Configuration to validate
 * @returns Validation result
 */
export function validateNodeConfig(
  type: string,
  config: Record<string, any>
): import('./registry').ValidationResult {
  return NodeRegistry.validateConfig(type, config);
}

/**
 * Get registry statistics
 *
 * @returns Registry stats
 */
export function getRegistryStats(): import('./registry').RegistryStats {
  return NodeRegistry.getStats();
}
