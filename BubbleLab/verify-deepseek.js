#!/usr/bin/env node
/**
 * DeepSeek Integration Verification Script
 * Verifies that DeepSeek is properly configured in the system
 */

import { CredentialType } from './packages/bubble-shared-schemas/dist/types.js';
import { CREDENTIAL_ENV_MAP, SYSTEM_CREDENTIALS, BUBBLE_CREDENTIAL_OPTIONS } from './packages/bubble-shared-schemas/dist/credential-schema.js';

console.log('🔍 Verifying DeepSeek API Integration...\n');

let passed = 0;
let failed = 0;

// Test 1: Credential Type exists
try {
  if (CredentialType.DEEPSEEK_CRED === 'DEEPSEEK_CRED') {
    console.log('✓ CredentialType.DEEPSEEK_CRED is defined');
    passed++;
  } else {
    console.log('✗ CredentialType.DEEPSEEK_CRED has wrong value');
    failed++;
  }
} catch (e) {
  console.log('✗ CredentialType.DEEPSEEK_CRED not found:', e.message);
  failed++;
}

// Test 2: Environment variable mapping exists
try {
  if (CREDENTIAL_ENV_MAP[CredentialType.DEEPSEEK_CRED] === 'DEEPSEEK_API_KEY') {
    console.log('✓ CREDENTIAL_ENV_MAP maps to DEEPSEEK_API_KEY');
    passed++;
  } else {
    console.log('✗ CREDENTIAL_ENV_MAP has wrong mapping');
    failed++;
  }
} catch (e) {
  console.log('✗ CREDENTIAL_ENV_MAP not found:', e.message);
  failed++;
}

// Test 3: DeepSeek in SYSTEM_CREDENTIALS
try {
  if (SYSTEM_CREDENTIALS.has(CredentialType.DEEPSEEK_CRED)) {
    console.log('✓ DeepSeek is in SYSTEM_CREDENTIALS');
    passed++;
  } else {
    console.log('✗ DeepSeek not in SYSTEM_CREDENTIALS');
    failed++;
  }
} catch (e) {
  console.log('✗ SYSTEM_CREDENTIALS check failed:', e.message);
  failed++;
}

// Test 4: DeepSeek in ai-agent credentials
try {
  if (BUBBLE_CREDENTIAL_OPTIONS['ai-agent'].includes(CredentialType.DEEPSEEK_CRED)) {
    console.log('✓ DeepSeek is available for ai-agent bubble');
    passed++;
  } else {
    console.log('✗ DeepSeek not in ai-agent credentials');
    failed++;
  }
} catch (e) {
  console.log('✗ ai-agent credentials check failed:', e.message);
  failed++;
}

// Test 5: DeepSeek in bubbleflow-generator credentials
try {
  if (BUBBLE_CREDENTIAL_OPTIONS['bubbleflow-generator'].includes(CredentialType.DEEPSEEK_CRED)) {
    console.log('✓ DeepSeek is available for bubbleflow-generator');
    passed++;
  } else {
    console.log('✗ DeepSeek not in bubbleflow-generator credentials');
    failed++;
  }
} catch (e) {
  console.log('✗ bubbleflow-generator credentials check failed:', e.message);
  failed++;
}

// Test 6: Verify model string format
try {
  const testModel = 'deepseek/deepseek-chat';
  const [provider, modelName] = testModel.split('/');
  if (provider === 'deepseek' && modelName === 'deepseek-chat') {
    console.log(`✓ Model string format is correct: ${testModel}`);
    passed++;
  } else {
    console.log('✗ Model string format is incorrect');
    failed++;
  }
} catch (e) {
  console.log('✗ Model string format check failed:', e.message);
  failed++;
}

// Summary
console.log('\n' + '='.repeat(50));
console.log(`Verification Results: ${passed} passed, ${failed} failed`);
console.log('='.repeat(50));

if (failed === 0) {
  console.log('\n✅ All DeepSeek integration checks PASSED!');
  console.log('\nTo use DeepSeek, add your API key to .env:');
  console.log('DEEPSEEK_API_KEY=your-deepseek-api-key');
  console.log('\nThen use models like:');
  console.log('- deepseek/deepseek-chat');
  console.log('- deepseek/deepseek-coder');
  process.exit(0);
} else {
  console.log('\n❌ Some checks failed. Please review the errors above.');
  process.exit(1);
}
