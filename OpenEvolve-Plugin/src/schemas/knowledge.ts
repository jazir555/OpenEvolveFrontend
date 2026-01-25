/**
 * Knowledge Engine Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const knowledgeParameters: ParameterSchema[] = [
  {
    name: 'query',
    type: 'text',
    label: 'Knowledge Query',
    description: 'Search or query the knowledge base',
    required: true,
    placeholder: 'Enter your query...',
  },
  {
    name: 'operation',
    type: 'select',
    label: 'Operation',
    description: 'Knowledge operation to perform',
    required: true,
    defaultValue: 'search',
    options: [
      { value: 'search', label: 'Search' },
      { value: 'add', label: 'Add Knowledge' },
      { value: 'update', label: 'Update Knowledge' },
      { value: 'delete', label: 'Delete Knowledge' },
      { value: 'reason', label: 'Reason/Infer' },
      { value: 'explore', label: 'Explore Graph' },
    ],
  },
  {
    name: 'knowledgeType',
    type: 'select',
    label: 'Knowledge Type',
    description: 'Type of knowledge entity',
    options: [
      { value: 'concept', label: 'Concept' },
      { value: 'relationship', label: 'Relationship' },
      { value: 'fact', label: 'Fact' },
      { value: 'rule', label: 'Rule' },
      { value: 'pattern', label: 'Pattern' },
    ],
  },
  {
    name: 'content',
    type: 'textarea',
    label: 'Content',
    description: 'Knowledge content (for add/update operations)',
    multiline: true,
    condition: (params) => ['add', 'update'].includes(params.operation),
  },
  {
    name: 'searchDepth',
    type: 'number',
    label: 'Search Depth',
    description: 'Depth of graph traversal',
    defaultValue: 2,
    min: 1,
    max: 5,
    condition: (params) => ['search', 'explore', 'reason'].includes(params.operation),
  },
  {
    name: 'maxResults',
    type: 'number',
    label: 'Maximum Results',
    defaultValue: 10,
    min: 1,
    max: 50,
  },
  {
    name: 'includeReasoning',
    type: 'boolean',
    label: 'Include Reasoning Trace',
    defaultValue: false,
  },
];
