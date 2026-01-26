/**
 * Hephaestus Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const hephaestusParameters: ParameterSchema[] = [
  {
    name: 'requirement',
    type: 'textarea',
    label: 'Code Requirement',
    description: 'Describe the code to generate',
    required: true,
    multiline: true,
    placeholder: 'Describe the code you want to generate...',
  },
  {
    name: 'language',
    type: 'select',
    label: 'Programming Language',
    required: true,
    defaultValue: 'python',
    options: [
      { value: 'python', label: 'Python' },
      { value: 'javascript', label: 'JavaScript' },
      { value: 'typescript', label: 'TypeScript' },
      { value: 'java', label: 'Java' },
      { value: 'cpp', label: 'C++' },
      { value: 'rust', label: 'Rust' },
      { value: 'go', label: 'Go' },
      { value: 'ruby', label: 'Ruby' },
      { value: 'csharp', label: 'C#' },
    ],
  },
  {
    name: 'codeType',
    type: 'select',
    label: 'Code Type',
    description: 'Type of code to generate',
    defaultValue: 'function',
    options: [
      { value: 'function', label: 'Function/Method' },
      { value: 'class', label: 'Class/Object' },
      { value: 'module', label: 'Module/Package' },
      { value: 'api', label: 'API Endpoint' },
      { value: 'script', label: 'Full Script' },
      { value: 'test', label: 'Test Code' },
    ],
  },
  {
    name: 'framework',
    type: 'text',
    label: 'Framework (optional)',
    description: 'Specific framework or library',
    placeholder: 'e.g., React, FastAPI, Spring...',
  },
  {
    name: 'includeTests',
    type: 'boolean',
    label: 'Include Unit Tests',
    defaultValue: true,
  },
  {
    name: 'includeDocs',
    type: 'boolean',
    label: 'Include Documentation',
    defaultValue: true,
  },
  {
    name: 'includeExamples',
    type: 'boolean',
    label: 'Include Usage Examples',
    defaultValue: false,
  },
  {
    name: 'codeStyle',
    type: 'select',
    label: 'Code Style',
    options: [
      { value: 'clean', label: 'Clean Code' },
      { value: 'functional', label: 'Functional' },
      { value: 'oop', label: 'Object-Oriented' },
      { value: 'concise', label: 'Concise' },
      { value: 'verbose', label: 'Verbose/Explicit' },
    ],
  },
  {
    name: 'errorHandling',
    type: 'select',
    label: 'Error Handling',
    options: [
      { value: 'basic', label: 'Basic Try-Catch' },
      { value: 'comprehensive', label: 'Comprehensive Error Handling' },
      { value: 'custom', label: 'Custom Error Types' },
    ],
  },
  {
    name: 'targetSystem',
    type: 'text',
    label: 'Target System',
    description: 'Where will this code run?',
    placeholder: 'e.g., AWS Lambda, Docker, local server...',
  },
];
