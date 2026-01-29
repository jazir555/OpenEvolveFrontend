/**
 * Adversarial Testing Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const adversarialParameters: ParameterSchema[] = [
  {
    name: 'targetContent',
    type: 'textarea',
    label: 'Target Content',
    description: 'Content to test adversarially',
    required: true,
    multiline: true,
    placeholder: 'Enter content to test...',
  },
  {
    name: 'attackMode',
    type: 'select',
    label: 'Attack Mode',
    description: 'Type of adversarial attack simulation',
    required: true,
    defaultValue: 'prompt-injection',
    options: [
      { value: 'prompt-injection', label: 'Prompt Injection' },
      { value: 'jailbreak', label: 'Jailbreak Attempts' },
      { value: 'adversarial-example', label: 'Adversarial Examples' },
      { value: 'model-extraction', label: 'Model Extraction' },
      { value: 'data-poisoning', label: 'Data Poisoning' },
      { value: 'comprehensive', label: 'Comprehensive Testing' },
    ],
  },
  {
    name: 'redTeamProvider',
    type: 'select',
    label: 'Red Team Provider',
    description: 'AI provider for red team (attacker)',
    required: true,
    defaultValue: 'anthropic',
    options: [
      { value: 'openai', label: 'OpenAI' },
      { value: 'anthropic', label: 'Anthropic' },
      { value: 'google', label: 'Google' },
    ],
  },
  {
    name: 'blueTeamProvider',
    type: 'select',
    label: 'Blue Team Provider',
    description: 'AI provider for blue team (defender)',
    required: true,
    defaultValue: 'openai',
    options: [
      { value: 'openai', label: 'OpenAI' },
      { value: 'anthropic', label: 'Anthropic' },
      { value: 'google', label: 'Google' },
    ],
  },
  {
    name: 'rounds',
    type: 'number',
    label: 'Battle Rounds',
    description: 'Number of attack/defense rounds',
    defaultValue: 3,
    min: 1,
    max: 10,
    step: 1,
  },
  {
    name: 'intensity',
    type: 'slider',
    label: 'Attack Intensity',
    description: 'Intensity of adversarial attacks (0.0 - 1.0)',
    defaultValue: 0.5,
    min: 0,
    max: 1,
    step: 0.1,
  },
  {
    name: 'enableReporting',
    type: 'boolean',
    label: 'Enable Detailed Reporting',
    description: 'Generate comprehensive security report',
    defaultValue: true,
  },
];
