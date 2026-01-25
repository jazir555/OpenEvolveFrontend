import type { ParameterSchema, ParameterValue } from '@/types/evolution';

export const evolutionParameters: ParameterSchema[] = [
  {
    name: 'content',
    type: 'textarea',
    label: 'Content to Evolve',
    description: 'The content that will be evolved through genetic algorithms',
    required: true,
    multiline: true,
    defaultValue: '',
    placeholder: 'Enter content to evolve...',
  },
  {
    name: 'iterations',
    type: 'number',
    label: 'Iterations',
    description: 'Number of evolution iterations to perform',
    required: true,
    defaultValue: 10,
    min: 1,
    max: 100,
    step: 1,
  },
  {
    name: 'temperature',
    type: 'slider',
    label: 'Temperature',
    description:
      'Creativity temperature (0.0 - 2.0). Higher values increase randomness.',
    required: true,
    defaultValue: 0.7,
    min: 0,
    max: 2,
    step: 0.1,
  },
  {
    name: 'populationSize',
    type: 'number',
    label: 'Population Size',
    description: 'Size of population per generation',
    required: true,
    defaultValue: 10,
    min: 10,
    max: 500,
    step: 1,
  },
  {
    name: 'maxBudget',
    type: 'number',
    label: 'Max Budget (USD)',
    description: 'Maximum budget allowed for this evolution run',
    required: false,
    defaultValue: 25,
    min: 0,
    max: 10000,
    step: 1,
  },
  {
    name: 'mutationRate',
    type: 'slider',
    label: 'Mutation Rate',
    description: 'Rate of genetic mutation (0.0 - 1.0)',
    required: true,
    defaultValue: 0.1,
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    name: 'crossoverRate',
    type: 'slider',
    label: 'Crossover Rate',
    description: 'Rate of genetic crossover (0.0 - 1.0)',
    required: true,
    defaultValue: 0.7,
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    name: 'branchingMode',
    type: 'select',
    label: 'Branching Mode',
    description:
      'Choose whether descendants branch off the latest survivors or the original seed',
    required: true,
    defaultValue: 'lineage',
    options: [
      { value: 'lineage', label: 'Lineage Branching' },
      { value: 'root', label: 'Root Variations' },
    ],
  },
  {
    name: 'childrenPerParent',
    type: 'number',
    label: 'Children per Parent',
    description: 'How many descendants each survivor spawns per generation',
    required: true,
    defaultValue: 3,
    min: 1,
    max: 50,
    step: 1,
  },
  {
    name: 'survivalThreshold',
    type: 'slider',
    label: 'Survival Threshold',
    description:
      'Fitness cutoff used by the judge to keep descendants alive (0.0 - 1.0)',
    required: true,
    defaultValue: 0.6,
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    name: 'selectionMethod',
    type: 'select',
    label: 'Selection Method',
    description: 'Method for selecting parents for next generation',
    required: true,
    defaultValue: 'tournament',
    options: [
      { value: 'tournament', label: 'Tournament Selection' },
      { value: 'roulette', label: 'Roulette Wheel' },
      { value: 'rank', label: 'Rank-Based Selection' },
      { value: 'steady', label: 'Steady State' },
    ],
  },
  {
    name: 'provider',
    type: 'select',
    label: 'AI Provider',
    description: 'The AI provider to use for evolution',
    required: true,
    defaultValue: 'anthropic',
    options: [
      { value: 'openai', label: 'OpenAI' },
      { value: 'anthropic', label: 'Anthropic' },
      { value: 'google', label: 'Google' },
      { value: 'openrouter', label: 'OpenRouter' },
      { value: 'deepseek', label: 'DeepSeek' },
    ],
  },
  {
    name: 'model',
    type: 'select',
    label: 'Model',
    description: 'The AI model to use',
    required: true,
    defaultValue: 'claude-3-sonnet',
    options: [
      { value: 'gpt-4', label: 'GPT-4' },
      { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
      { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
      { value: 'claude-3-opus', label: 'Claude 3 Opus' },
      { value: 'claude-3-haiku', label: 'Claude 3 Haiku' },
      { value: 'gemini-pro', label: 'Gemini Pro' },
      { value: 'deepseek-chat', label: 'DeepSeek Chat' },
      { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner' },
    ],
  },
  {
    name: 'apiKey',
    type: 'text',
    label: 'Model API Key',
    description: 'API key for the selected provider (required for execution)',
    required: true,
    defaultValue: '',
    placeholder: 'sk-... or provider API key',
  },
  {
    name: 'objective',
    type: 'textarea',
    label: 'Objective Function',
    description: 'Define the fitness objective (optional custom objective)',
    required: false,
    multiline: true,
    placeholder: 'Define custom fitness criteria...',
  },
  {
    name: 'preserveElite',
    type: 'boolean',
    label: 'Preserve Elite',
    description: 'Preserve best individuals from each generation',
    required: false,
    defaultValue: true,
  },
  {
    name: 'enableLogging',
    type: 'boolean',
    label: 'Enable Logging',
    description: 'Enable detailed logging of evolution process',
    required: false,
    defaultValue: true,
  },
];

export const evolutionConstraintParameters: ParameterSchema[] = [
  {
    name: 'seedImageUrl',
    type: 'text',
    label: 'Seed Image URL',
    description: 'Optional starting image to guide the evolution',
    placeholder: 'https://...',
  },
  {
    name: 'brandGuidelines',
    type: 'textarea',
    label: 'Brand Guidelines',
    description: 'Voice, tone, do/don\'t rules, and overall direction',
    multiline: true,
    placeholder: 'e.g. Minimal, confident, premium; avoid heavy gradients...',
  },
  {
    name: 'colorPalette',
    type: 'text',
    label: 'Color Palette',
    description: 'Comma-separated colors or tokens to prioritize',
    placeholder: '#0B3D91, #F2F2F2, #FF9F1C',
  },
  {
    name: 'typography',
    type: 'text',
    label: 'Typography',
    description: 'Preferred fonts or typographic notes',
    placeholder: 'Space Grotesk for headings, Inter for body',
  },
  {
    name: 'imageryNotes',
    type: 'textarea',
    label: 'Imagery Notes',
    description: 'Image style, subject matter, and constraints',
    multiline: true,
    placeholder: 'e.g. abstract product renders, no people, high contrast',
  },
  {
    name: 'layoutConstraints',
    type: 'textarea',
    label: 'Layout Constraints',
    description: 'Specific layout rules or sections to include/avoid',
    multiline: true,
    placeholder: 'Hero + 3 feature cards + testimonial band, no sticky nav',
  },
];

export const evolutionPreviewParameters: ParameterSchema[] = [
  {
    name: 'cachePreviews',
    type: 'boolean',
    label: 'Cache Preview Images',
    description:
      'Store per-node preview thumbnails locally for faster browsing (disable to reduce disk usage).',
    defaultValue: true,
  },
];

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

const defaultForParameter = (parameter: ParameterSchema): ParameterValue => {
  if (parameter.defaultValue !== undefined) {
    return parameter.defaultValue;
  }

  switch (parameter.type) {
    case 'boolean':
      return false;
    case 'number':
    case 'slider':
      return parameter.min ?? 0;
    default:
      return '';
  }
};

export const buildDefaultValues = (
  parameters: ParameterSchema[]
): Record<string, ParameterValue> =>
  parameters.reduce<Record<string, ParameterValue>>((acc, parameter) => {
    acc[parameter.name] = defaultForParameter(parameter);
    return acc;
  }, {});
