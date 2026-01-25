/**
 * OpenEvolve React Flow Nodes
 *
 * Export all node components and types for easy integration
 */

// Base node component
export { OpenEvolveNode } from './OpenEvolveNode';

// Specialized node components
export { DecompositionNodeComponent } from './DecompositionNodeComponent';
export { SolutionNodeComponent } from './SolutionNodeComponent';
export { VerificationNodeComponent } from './VerificationNodeComponent';
export { ResearchQuestNodeComponent } from './ResearchQuestNodeComponent';
export { PyGraphistryNodeComponent } from './PyGraphistryNodeComponent';

// Export all types
export type {
  OpenEvolveNodeData,
  NodeStatus,
  NodeType,
  NodeConfig,
  NodeResult,
  OpenEvolveFlowNode,
  BaseOpenEvolveNodeProps,
  SubProblem,
  DependencyInfo,
  AlternativeSolution,
  SolutionMetrics,
  QualityMetrics,
  Requirement,
} from '../types/nodeTypes';

// Export helper functions
export {
  createOpenEvolveNode,
  createFlowNode,
  NODE_PRESETS,
} from '../types/nodeTypes';

/**
 * Node Type Registry for React Flow
 *
 * Use this to register all OpenEvolve node types with React Flow
 */
export const OPENEVOLVE_NODE_TYPES = {
  openevolve: 'OpenEvolveNode',
  decomposition: 'DecompositionNodeComponent',
  solution: 'SolutionNodeComponent',
  verification: 'VerificationNodeComponent',
  researchQuest: 'ResearchQuestNodeComponent',
  pyGraphistry: 'PyGraphistryNodeComponent',
} as const;

/**
 * Node Components Map
 *
 * Pass this to React Flow's nodeTypes prop
 */
export const openEvolveNodeComponents = {
  OpenEvolveNode: lazy(() => import('./OpenEvolveNode').then(m => ({ default: m.OpenEvolveNode }))),
  DecompositionNodeComponent: lazy(() => import('./DecompositionNodeComponent').then(m => ({ default: m.DecompositionNodeComponent }))),
  SolutionNodeComponent: lazy(() => import('./SolutionNodeComponent').then(m => ({ default: m.SolutionNodeComponent }))),
  VerificationNodeComponent: lazy(() => import('./VerificationNodeComponent').then(m => ({ default: m.VerificationNodeComponent }))),
  ResearchQuestNodeComponent: lazy(() => import('./ResearchQuestNodeComponent').then(m => ({ default: m.ResearchQuestNodeComponent }))),
  PyGraphistryNodeComponent: lazy(() => import('./PyGraphistryNodeComponent').then(m => ({ default: m.PyGraphistryNodeComponent }))),
};

/**
 * Convenience import for React.lazy (if needed)
 */
import { lazy } from 'react';
