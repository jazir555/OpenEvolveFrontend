/**
 * Invention Planner Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const inventionParameters: ParameterSchema[] = [
  {
    name: 'goal',
    type: 'textarea',
    label: 'Invention Goal',
    description: 'What do you want to invent?',
    required: true,
    multiline: true,
    placeholder: 'Describe your invention goal...',
  },
  {
    name: 'domain',
    type: 'select',
    label: 'Domain',
    description: 'Primary domain of invention',
    required: true,
    options: [
      { value: 'technology', label: 'Technology/Software' },
      { value: 'hardware', label: 'Hardware/Physical' },
      { value: 'business', label: 'Business Model' },
      { value: 'process', label: 'Process/System' },
      { value: 'scientific', label: 'Scientific Discovery' },
      { value: 'creative', label: 'Creative/Artistic' },
    ],
  },
  {
    name: 'innovativeness',
    type: 'slider',
    label: 'Innovativeness Level',
    description: 'How radical should the invention be? (0.0 = incremental, 1.0 = disruptive)',
    defaultValue: 0.7,
    min: 0,
    max: 1,
    step: 0.1,
  },
  {
    name: 'planningStages',
    type: 'multiselect',
    label: 'Planning Stages',
    description: 'Stages to include in invention plan',
    options: [
      { value: 'research', label: 'Research Phase' },
      { value: 'ideation', label: 'Ideation' },
      { value: 'prototyping', label: 'Prototyping' },
      { value: 'testing', label: 'Testing' },
      { value: 'validation', label: 'Validation' },
      { value: 'scaling', label: 'Scaling' },
      { value: 'commercialization', label: 'Commercialization' },
    ],
  },
  {
    name: 'constraints',
    type: 'textarea',
    label: 'Constraints',
    description: 'Any constraints or limitations',
    multiline: true,
    placeholder: 'Budget, time, resources, technical constraints...',
  },
  {
    name: 'targetAudience',
    type: 'text',
    label: 'Target Audience',
    placeholder: 'Who is this invention for?',
  },
  {
    name: 'includePriorArt',
    type: 'boolean',
    label: 'Include Prior Art Analysis',
    description: 'Analyze existing solutions and patents',
    defaultValue: true,
  },
  {
    name: 'includeFeasibility',
    type: 'boolean',
    label: 'Include Feasibility Analysis',
    defaultValue: true,
  },
  {
    name: 'includeRoadmap',
    type: 'boolean',
    label: 'Include Implementation Roadmap',
    defaultValue: true,
  },
  {
    name: 'detailLevel',
    type: 'select',
    label: 'Detail Level',
    options: [
      { value: 'overview', label: 'High-Level Overview' },
      { value: 'detailed', label: 'Detailed Plan' },
      { value: 'comprehensive', label: 'Comprehensive (All Details)' },
    ],
  },
];
