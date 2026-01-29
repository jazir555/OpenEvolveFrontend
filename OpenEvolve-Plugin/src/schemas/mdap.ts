/**
 * MDAP (Multi-Domain Agent Planner) Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const mdapParameters: ParameterSchema[] = [
  {
    name: 'objective',
    type: 'textarea',
    label: 'Planning Objective',
    description: 'What you want to achieve with multi-agent planning',
    required: true,
    multiline: true,
    placeholder: 'Describe your planning objective...',
  },
  {
    name: 'domains',
    type: 'multiselect',
    label: 'Domains',
    description: 'Select domains to involve in planning',
    required: true,
    options: [
      { value: 'research', label: 'Research' },
      { value: 'analysis', label: 'Analysis' },
      { value: 'planning', label: 'Planning' },
      { value: 'execution', label: 'Execution' },
      { value: 'evaluation', label: 'Evaluation' },
      { value: 'optimization', label: 'Optimization' },
    ],
  },
  {
    name: 'agents',
    type: 'number',
    label: 'Number of Agents',
    description: 'Number of agents to deploy',
    defaultValue: 5,
    min: 2,
    max: 10,
  },
  {
    name: 'planningHorizon',
    type: 'select',
    label: 'Planning Horizon',
    description: 'Time horizon for planning',
    defaultValue: 'medium',
    options: [
      { value: 'short', label: 'Short-term (immediate actions)' },
      { value: 'medium', label: 'Medium-term (strategic planning)' },
      { value: 'long', label: 'Long-term (visionary planning)' },
    ],
  },
  {
    name: 'coordinationStrategy',
    type: 'select',
    label: 'Coordination Strategy',
    description: 'How agents should coordinate',
    defaultValue: 'hierarchical',
    options: [
      { value: 'hierarchical', label: 'Hierarchical' },
      { value: 'flat', label: 'Flat/Peer-to-Peer' },
      { value: 'dynamic', label: 'Dynamic/Adaptive' },
    ],
  },
  {
    name: 'iterations',
    type: 'number',
    label: 'Planning Iterations',
    defaultValue: 3,
    min: 1,
    max: 10,
  },
  {
    name: 'enableCollaboration',
    type: 'boolean',
    label: 'Enable Agent Collaboration',
    defaultValue: true,
  },
];
