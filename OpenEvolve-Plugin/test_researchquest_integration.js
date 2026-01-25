// Test script to verify ResearchQuestNode integration
import { NodeRegistry } from './src/nodes/registry';
import { ResearchQuestNode } from './src/nodes/ResearchQuestNode';

console.log('Testing ResearchQuestNode integration...');

// Check if ResearchQuestNode is registered
const registeredNode = NodeRegistry.get('ResearchQuest');
if (registeredNode) {
  console.log('✅ ResearchQuestNode is registered in the registry');
  
  // Check static properties
  console.log('Display Name:', ResearchQuestNode.DISPLAY_NAME);
  console.log('Description:', ResearchQuestNode.DESCRIPTION);
  console.log('Icon:', ResearchQuestNode.ICON);
  console.log('Category:', ResearchQuestNode.CATEGORY);
  console.log('Version:', ResearchQuestNode.VERSION);
} else {
  console.log('❌ ResearchQuestNode is NOT registered in the registry');
}

// List all registered nodes to see if ResearchQuest is there
console.log('\nAll registered nodes:');
const allNodes = NodeRegistry.listAll();
allNodes.forEach(({ type, metadata }) => {
  console.log(`- ${type}: ${metadata.displayName} (${metadata.category})`);
});

// Check if ResearchQuest appears in the list
const researchQuestExists = allNodes.some(node => node.type === 'ResearchQuest');
if (researchQuestExists) {
  console.log('\n✅ ResearchQuestNode found in registry listing');
} else {
  console.log('\n❌ ResearchQuestNode NOT found in registry listing');
}

// Try to create an instance
try {
  const nodeInstance = NodeRegistry.create('ResearchQuest', 'test-node-id');
  if (nodeInstance) {
    console.log('\n✅ Successfully created ResearchQuestNode instance');
    console.log('Node ID:', nodeInstance.id);
  } else {
    console.log('\n❌ Failed to create ResearchQuestNode instance');
  }
} catch (error) {
  console.log('\n❌ Error creating ResearchQuestNode instance:', error.message);
}

console.log('\nIntegration test completed!');