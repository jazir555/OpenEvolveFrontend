/**
 * Test realistic flow decomposition scenario
 */

import { generateDisplayedBubbleParameters } from '../src/services/bubble-flow-parser.js';
import { BubbleParameterType } from '@bubblelab/shared-schemas';
import type { ParsedBubble } from '@bubblelab/shared-schemas';

// Create a test flow similar to what the API would process
const testFlow: Record<string, ParsedBubble> = {
  'postgres': {
    variableName: 'postgres',
    bubbleName: 'postgres',
    className: 'PostgresBubble',
    parameters: [
      {
        name: 'connectionString',
        value: 'process.env.DATABASE_URL',
        type: BubbleParameterType.ENV,
      },
      {
        name: 'query',
        value: 'SELECT * FROM users WHERE active = true',
        type: BubbleParameterType.STRING,
      },
    ],
    hasAwait: true,
    hasActionCall: false,
  },
  'aiAgent': {
    variableName: 'aiAgent',
    bubbleName: 'ai-agent',
    className: 'AIAgentBubble',
    parameters: [
      {
        name: 'model',
        value: 'gpt-4',
        type: BubbleParameterType.STRING,
      },
      {
        name: 'prompt',
        value: 'Analyze the user engagement data',
        type: BubbleParameterType.STRING,
      },
      {
        name: 'tools',
        value: '[{"name": "web-search-tool"}]',
        type: BubbleParameterType.ARRAY,
      },
    ],
    hasAwait: true,
    hasActionCall: false,
  },
  'slack': {
    variableName: 'slack',
    bubbleName: 'slack',
    className: 'SlackBubble',
    parameters: [
      {
        name: 'channel',
        value: '#analytics',
        type: BubbleParameterType.STRING,
      },
      {
        name: 'message',
        value: 'aiAgent.responseText',
        type: BubbleParameterType.STRING,
      },
    ],
    hasAwait: false,
    hasActionCall: false,
  },
};

console.log('Testing flow decomposition with realistic data...');
console.log('====================================\n');

const result = generateDisplayedBubbleParameters(testFlow);

console.log('DECOMPOSITION RESULT:');
console.log('====================================');
console.log(`Displayed Parameters: ${result.displayedParameters.length}`);
result.displayedParameters.forEach((param, i) => {
  console.log(`  ${i + 1}. ${param.name}`);
  console.log(`     Display: ${param.displayName}`);
  console.log(`     Type: ${param.type}`);
  console.log(`     Source: ${param.source}`);
  console.log(`     Required: ${param.isRequired}`);
  console.log(`     Configurable: ${param.isConfigurable}`);
  if (param.dependencies && param.dependencies.length > 0) {
    console.log(`     Dependencies: ${param.dependencies.join(', ')}`);
  }
  console.log('');
});

console.log('Dependencies:');
console.log(`  Nodes: ${result.dependencies.nodes.length}`);
console.log(`  Edges: ${result.dependencies.edges.length}`);

console.log('\nValidation Rules:', result.validationRules.length);
result.validationRules.slice(0, 5).forEach((rule, i) => {
  console.log(`  ${i + 1}. [${rule.type}] ${rule.message} (severity: ${rule.severity})`);
});

console.log('\nMetadata:');
console.log(`  Total Parameters: ${result.metadata.totalParameters}`);
console.log(`  Required: ${result.metadata.requiredParameters}`);
console.log(`  Configurable: ${result.metadata.configurableParameters}`);
console.log(`  Environment: ${result.metadata.environmentParameters}`);
console.log(`  Nested: ${result.metadata.nestedParameterCount}`);
console.log(`  Circular Dependencies: ${result.metadata.hasCircularDependencies}`);
console.log(`  Complexity: ${result.metadata.estimatedComplexity}`);
console.log(`  Groups: ${result.metadata.groups.length}`);

result.metadata.groups.forEach((group) => {
  console.log(`    - ${group.name}: ${group.parameters.length} parameters`);
});

console.log('\n✅ Flow decomposition test completed successfully!');
