/**
 * Maker Engine Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const makerParameters: ParameterSchema[] = [
  {
    name: 'prompt',
    type: 'textarea',
    label: 'Generation Prompt',
    description: 'Describe what you want to create',
    required: true,
    multiline: true,
    placeholder: 'Describe what you want to generate...',
  },
  {
    name: 'generationMode',
    type: 'select',
    label: 'Generation Mode',
    description: 'Type of content to generate',
    required: true,
    defaultValue: 'creative',
    options: [
      { value: 'creative', label: 'Creative Writing' },
      { value: 'technical', label: 'Technical Content' },
      { value: 'code', label: 'Code Generation' },
      { value: 'design', label: 'Design Concepts' },
      { value: 'business', label: 'Business Documents' },
      { value: 'research', label: 'Research Synthesis' },
    ],
  },
  {
    name: 'creativity',
    type: 'slider',
    label: 'Creativity Level',
    description: 'Creativity and diversity (0.0 - 1.0)',
    defaultValue: 0.7,
    min: 0,
    max: 1,
    step: 0.1,
  },
  {
    name: 'iterations',
    type: 'number',
    label: 'Generation Iterations',
    description: 'Number of refinement iterations',
    defaultValue: 3,
    min: 1,
    max: 10,
  },
  {
    name: 'style',
    type: 'select',
    label: 'Output Style',
    description: 'Style of the generated content',
    options: [
      { value: 'formal', label: 'Formal' },
      { value: 'casual', label: 'Casual' },
      { value: 'technical', label: 'Technical' },
      { value: 'creative', label: 'Creative' },
      { value: 'academic', label: 'Academic' },
    ],
  },
  {
    name: 'provider',
    type: 'select',
    label: 'AI Provider',
    defaultValue: 'anthropic',
    options: [
      { value: 'openai', label: 'OpenAI' },
      { value: 'anthropic', label: 'Anthropic' },
      { value: 'google', label: 'Google' },
    ],
  },
  {
    name: 'model',
    type: 'select',
    label: 'Model',
    defaultValue: 'claude-3-sonnet',
    options: [
      { value: 'gpt-4', label: 'GPT-4' },
      { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
      { value: 'claude-3-opus', label: 'Claude 3 Opus' },
      { value: 'gemini-pro', label: 'Gemini Pro' },
    ],
  },
  {
    name: 'length',
    type: 'select',
    label: 'Output Length',
    description: 'Target length of generated content',
    options: [
      { value: 'short', label: 'Short (100-300 words)' },
      { value: 'medium', label: 'Medium (300-700 words)' },
      { value: 'long', label: 'Long (700-1500 words)' },
      { value: 'comprehensive', label: 'Comprehensive (1500+ words)' },
    ],
  },
  {
    name: 'includeVariations',
    type: 'boolean',
    label: 'Generate Variations',
    description: 'Generate multiple variations',
    defaultValue: false,
  },
  {
    name: 'variationCount',
    type: 'number',
    label: 'Variation Count',
    description: 'Number of variations to generate',
    defaultValue: 3,
    min: 2,
    max: 5,
    condition: (params) => params.includeVariations === true,
  },
];
