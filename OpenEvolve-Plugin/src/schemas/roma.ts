/**
 * ROMA (Reasoning and Multi-Agent) Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const romaParameters: ParameterSchema[] = [
  {
    name: 'task',
    type: 'textarea',
    label: 'Task Description',
    description: 'The complex reasoning task',
    required: true,
    multiline: true,
    placeholder: 'Describe the reasoning task...',
  },
  {
    name: 'reasoningMode',
    type: 'select',
    label: 'Reasoning Mode',
    description: 'Type of reasoning to apply',
    required: true,
    defaultValue: 'collaborative',
    options: [
      { value: 'collaborative', label: 'Collaborative Reasoning' },
      { value: 'adversarial', label: 'Adversarial Reasoning' },
      { value: 'debate', label: 'Debate-Based' },
      { value: 'consensus', label: 'Consensus Building' },
      { value: 'hierarchical', label: 'Hierarchical Reasoning' },
    ],
  },
  {
    name: 'agentCount',
    type: 'number',
    label: 'Number of Agents',
    defaultValue: 3,
    min: 2,
    max: 7,
  },
  {
    name: 'agentRoles',
    type: 'multiselect',
    label: 'Agent Roles',
    description: 'Roles for reasoning agents',
    options: [
      { value: 'analyst', label: 'Analyst' },
      { value: 'critic', label: 'Critic' },
      { value: 'synthesizer', label: 'Synthesizer' },
      { value: 'validator', label: 'Validator' },
      { value: 'explorer', label: 'Explorer' },
      { value: 'integrator', label: 'Integrator' },
    ],
  },
  {
    name: 'rounds',
    type: 'number',
    label: 'Reasoning Rounds',
    description: 'Number of reasoning iterations',
    defaultValue: 3,
    min: 1,
    max: 10,
  },
  {
    name: 'confidenceThreshold',
    type: 'slider',
    label: 'Confidence Threshold',
    description: 'Minimum confidence for consensus (0.0 - 1.0)',
    defaultValue: 0.7,
    min: 0,
    max: 1,
    step: 0.1,
  },
  {
    name: 'includeReasoningTrace',
    type: 'boolean',
    label: 'Include Reasoning Trace',
    description: 'Show detailed reasoning process',
    defaultValue: true,
  },
  {
    name: 'enableVoting',
    type: 'boolean',
    label: 'Enable Agent Voting',
    defaultValue: true,
  },
];
