/**
 * Decomposition Engine Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const decompositionParameters: ParameterSchema[] = [
  {
    name: 'problem',
    type: 'textarea',
    label: 'Problem Statement',
    description: 'The complex problem to decompose',
    required: true,
    multiline: true,
    placeholder: 'Describe the complex problem...',
  },
  {
    name: 'decompositionMethod',
    type: 'select',
    label: 'Decomposition Method',
    description: 'Approach to breaking down the problem',
    required: true,
    defaultValue: 'hierarchical',
    options: [
      { value: 'hierarchical', label: 'Hierarchical Decomposition' },
      { value: 'temporal', label: 'Temporal/Sequential' },
      { value: 'functional', label: 'Functional Decomposition' },
      { value: 'object-oriented', label: 'Object-Oriented' },
      { value: 'data-flow', label: 'Data-Flow Decomposition' },
    ],
  },
  {
    name: 'granularity',
    type: 'select',
    label: 'Granularity Level',
    description: 'How detailed should the decomposition be',
    defaultValue: 'medium',
    options: [
      { value: 'coarse', label: 'Coarse (high-level tasks)' },
      { value: 'medium', label: 'Medium (balanced)' },
      { value: 'fine', label: 'Fine (detailed subtasks)' },
      { value: 'atomic', label: 'Atomic (smallest executable units)' },
    ],
  },
  {
    name: 'maxDepth',
    type: 'number',
    label: 'Maximum Depth',
    description: 'Maximum depth of decomposition hierarchy',
    defaultValue: 3,
    min: 1,
    max: 5,
  },
  {
    name: 'recursionDepthLimit',
    type: 'number',
    label: 'Recursion Depth Limit',
    description: '0 = unlimited recursion depth',
    defaultValue: 1,
    min: 0,
    max: 10,
  },
  {
    name: 'maxSubProblems',
    type: 'number',
    label: 'Max Sub-Problems',
    description: '0 = unlimited sub-problems',
    defaultValue: 3,
    min: 0,
    max: 100,
  },
  {
    name: 'dependencies',
    type: 'boolean',
    label: 'Include Dependencies',
    description: 'Analyze and show task dependencies',
    defaultValue: true,
  },
  {
    name: 'prioritization',
    type: 'select',
    label: 'Prioritization Method',
    options: [
      { value: 'critical-path', label: 'Critical Path Method' },
      { value: 'moscow', label: 'MoSCoW Prioritization' },
      { value: 'effort-impact', label: 'Effort vs Impact' },
      { value: 'risk-based', label: 'Risk-Based Prioritization' },
    ],
  },
  {
    name: 'outputFormat',
    type: 'select',
    label: 'Output Format',
    options: [
      { value: 'tree', label: 'Tree Structure' },
      { value: 'dag', label: 'Directed Acyclic Graph' },
      { value: 'wbs', label: 'Work Breakdown Structure' },
      { value: 'mindmap', label: 'Mind Map Format' },
    ],
  },
];
