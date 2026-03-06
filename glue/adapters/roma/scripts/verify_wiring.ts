#!/usr/bin/env tsx
/**
 * ROMA Integration Wiring Verification Script
 *
 * This script verifies that all ROMA components are properly connected:
 * 1. Canonical schema exports
 * 2. Adapter exports
 * 3. Python bridge imports
 * 4. Workflow templates
 * 5. Test files exist
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface WiringCheck {
  name: string;
  passed: boolean;
  details: string;
  file?: string;
}

const checks: WiringCheck[] = [];

function check(name: string, passed: boolean, details: string, file?: string) {
  checks.push({ name, passed, details, file });
}

console.log('ROMA Integration Wiring Verification\n');
console.log('='.repeat(60));

// ============================================================================
// 1. Check Canonical Schema
// ============================================================================
console.log('\n1. Checking Canonical Schema...');

const schemaPath = join(__dirname, '../../../schemas/roma-canonical.ts');
check(
  'Schema file exists',
  existsSync(schemaPath),
  existsSync(schemaPath) ? 'Found at glue/schemas/roma-canonical.ts' : 'File not found',
  schemaPath
);

const schemaContent = readFileSync(schemaPath, 'utf-8');
const schemaExports = [
  'RomaExecutionRequest',
  'RomaExecutionResponse',
  'RomaExecutionStatistics',
  'RomaTaskNode',
  'RomaCheckpoint',
  'transformRomaResponseToCanonical',
  'transformCanonicalToRomaRequest',
  'validateRomaExecutionRequest',
  'validateRomaExecutionResponse',
];

schemaExports.forEach(exp => {
  check(
    `Schema exports ${exp}`,
    schemaContent.includes(`export ${exp}`) || schemaContent.includes(`export { ${exp}`) || schemaContent.includes(`export {${exp}`) || schemaContent.includes(`export interface ${exp}`) || schemaContent.includes(`export type ${exp}`) || schemaContent.includes(`export function ${exp}`),
    schemaContent.includes(`export ${exp}`) || schemaContent.includes(`export { ${exp}`) || schemaContent.includes(`export {${exp}`) || schemaContent.includes(`export interface ${exp}`) || schemaContent.includes(`export type ${exp}`) || schemaContent.includes(`export function ${exp}`)
      ? 'Export found'
      : 'Export missing'
  );
});

// ============================================================================
// 2. Check Schema Index
// ============================================================================
console.log('\n2. Checking Schema Index...');

const indexPath = join(__dirname, '../../../schemas/index.ts');
check(
  'Schema index exists',
  existsSync(indexPath),
  existsSync(indexPath) ? 'Found at glue/schemas/index.ts' : 'File not found',
  indexPath
);

const indexContent = readFileSync(indexPath, 'utf-8');

check(
  'ROMA schemas exported from index',
  indexContent.includes("from './roma-canonical'"),
  indexContent.includes("from './roma-canonical'") ? 'ROMA exports found' : 'ROMA exports missing'
);

check(
  'ROMA in SchemaRegistry',
  indexContent.includes("roma: {"),
  indexContent.includes("roma: {") ? 'ROMA registered in SchemaRegistry' : 'ROMA not in registry'
);

// ============================================================================
// 3. Check Canonical Adapter
// ============================================================================
console.log('\n3. Checking Canonical Adapter...');

const adapterPath = join(__dirname, '../../../adapters/roma-adapter/src/adapter.ts');
check(
  'Adapter file exists',
  existsSync(adapterPath),
  existsSync(adapterPath) ? 'Found at glue/adapters/roma-adapter/src/adapter.ts' : 'File not found',
  adapterPath
);

const adapterContent = readFileSync(adapterPath, 'utf-8');

const adapterExports = [
  'RomaAdapterConfig',
  'AdapterExecutionContext',
  'RomaCanonicalAdapter',
  'createRomaAdapter',
];

adapterExports.forEach(exp => {
  check(
    `Adapter exports ${exp}`,
    adapterContent.includes(`export ${exp}`) || adapterContent.includes(`export interface ${exp}`) || adapterContent.includes(`export class ${exp}`) || adapterContent.includes(`export function ${exp}`),
    adapterContent.includes(`export ${exp}`) || adapterContent.includes(`export interface ${exp}`) || adapterContent.includes(`export class ${exp}`) || adapterContent.includes(`export function ${exp}`)
      ? 'Export found'
      : 'Export missing'
  );
});

check(
  'Adapter uses canonical schema',
  adapterContent.includes("from '../../../schemas/roma-canonical'") ||
  adapterContent.includes("from '../../../schemas") ||
  adapterContent.includes("from '../../schemas/roma-canonical'") ||
  adapterContent.includes("from '../../schemas"),
  adapterContent.includes("from '../../../schemas") || adapterContent.includes("from '../../schemas")
    ? 'Schema imports found'
    : 'Schema imports missing'
);

check(
  'Adapter has EventEmitter',
  adapterContent.includes('extends EventEmitter'),
  adapterContent.includes('extends EventEmitter') ? 'EventEmitter integration found' : 'EventEmitter missing'
);

// ============================================================================
// 4. Check Python Bridge
// ============================================================================
console.log('\n4. Checking Python Bridge...');

const bridgePath = join(__dirname, '../roma-bridge.py');
check(
  'Python bridge exists',
  existsSync(bridgePath),
  existsSync(bridgePath) ? 'Found at glue/adapters/roma/roma-bridge.py' : 'File not found',
  bridgePath
);

const bridgeContent = readFileSync(bridgePath, 'utf-8');

const bridgeExports = [
  'RomaCanonicalBridge',
  'get_roma_bridge',
  'solve_with_roma',
  'recursive_solve',
];

bridgeExports.forEach(exp => {
  check(
    `Bridge exports ${exp}`,
    bridgeContent.includes(exp) && bridgeContent.includes('__all__'),
    bridgeContent.includes(exp) ? 'Export found' : 'Export missing'
  );
});

check(
  'Bridge has async methods',
  bridgeContent.includes('async def execute_task'),
  bridgeContent.includes('async def execute_task') ? 'Async methods found' : 'Async methods missing'
);

// ============================================================================
// 5. Check Workflow Templates
// ============================================================================
console.log('\n5. Checking Workflow Templates...');

const workflowPath = join(__dirname, '../../../orchestration/workflow-system/roma-workflow-templates.ts');
check(
  'Workflow templates exist',
  existsSync(workflowPath),
  existsSync(workflowPath) ? 'Found at glue/orchestration/workflow-system/roma-workflow-templates.ts' : 'File not found',
  workflowPath
);

const workflowContent = readFileSync(workflowPath, 'utf-8');

const workflowTemplates = [
  'ROMA_DECOMPOSITION_WORKFLOW',
  'ROMA_MDAP_MAKER_WORKFLOW',
  'ROMA_MULTI_AGENT_WORKFLOW',
  'ROMA_HYBRID_WORKFLOW',
];

workflowTemplates.forEach(tpl => {
  check(
    `Workflow template ${tpl}`,
    workflowContent.includes(`export const ${tpl}`),
    workflowContent.includes(`export const ${tpl}`) ? 'Template found' : 'Template missing'
  );
});

check(
  'Workflow registry exists',
  workflowContent.includes('ROMA_WORKFLOW_TEMPLATES'),
  workflowContent.includes('ROMA_WORKFLOW_TEMPLATES') ? 'Registry found' : 'Registry missing'
);

// ============================================================================
// 6. Check Contract Tests
// ============================================================================
console.log('\n6. Checking Contract Tests...');

const testDir = join(__dirname, '../roma-bubblelab-plugin/src/tests/contract');

const clientTestPath = join(testDir, 'roma-client.test.ts');
check(
  'Client contract tests exist',
  existsSync(clientTestPath),
  existsSync(clientTestPath) ? 'Found at roma-client.test.ts' : 'File not found',
  clientTestPath
);

if (existsSync(clientTestPath)) {
  const clientTestContent = readFileSync(clientTestPath, 'utf-8');
  check(
    'Client tests use Jest',
    clientTestContent.includes('describe') && clientTestContent.includes('it('),
    'Jest test format found'
  );
}

const serviceTestPath = join(testDir, 'roma-service.test.ts');
check(
  'Service contract tests exist',
  existsSync(serviceTestPath),
  existsSync(serviceTestPath) ? 'Found at roma-service.test.ts' : 'File not found',
  serviceTestPath
);

// ============================================================================
// 7. Check Probe Scripts
// ============================================================================
console.log('\n7. Checking Probe Scripts...');

const probesDir = join(__dirname, '../probes');

const probeScripts = [
  'check_api.sh',
  'probe_execution.sh',
  'probe_storage.sh',
];

probeScripts.forEach(script => {
  const scriptPath = join(probesDir, script);
  check(
    `Probe script ${script}`,
    existsSync(scriptPath),
    existsSync(scriptPath) ? `Found at ${script}` : 'Script not found',
    scriptPath
  );
});

// ============================================================================
// 8. Check Documentation
// ============================================================================
console.log('\n8. Checking Documentation...');

const docs: [string, string][] = [
  ['ROMA_UNIFICATION_GUIDE.md', 'docs/ROMA_UNIFICATION_GUIDE.md'],
  ['ROMA_AIR_GAP_COMPLIANCE_REPORT.md', 'docs/ROMA_AIR_GAP_COMPLIANCE_REPORT.md'],
  ['ROMA_REFACTORING_GUIDE.md', 'docs/ROMA_REFACTORING_GUIDE.md'],
];

docs.forEach(([doc, path]) => {
  const docPath = join(__dirname, '../../../..', path);
  check(
    `Documentation ${doc}`,
    existsSync(docPath),
    existsSync(docPath) ? `Found at ${path}` : 'Documentation not found',
    docPath
  );
});

// ============================================================================
// RESULTS
// ============================================================================
console.log('\n' + '='.repeat(60));
console.log('\nRESULTS:\n');

const passed = checks.filter(c => c.passed).length;
const failed = checks.filter(c => !c.passed).length;

checks.forEach(check => {
  const icon = check.passed ? '✓' : '✗';
  const status = check.passed ? 'PASS' : 'FAIL';
  console.log(`  ${icon} [${status}] ${check.name}`);
  if (!check.passed && check.file) {
    console.log(`           File: ${check.file}`);
    console.log(`           Details: ${check.details}`);
  }
});

console.log('\n' + '-'.repeat(60));
console.log(`Total: ${checks.length} checks`);
console.log(`Passed: ${passed} (${Math.round(passed / checks.length * 100)}%)`);
console.log(`Failed: ${failed} (${Math.round(failed / checks.length * 100)}%)`);

if (failed === 0) {
  console.log('\n✓ All ROMA integration wiring checks passed!');
  console.log('\nThe ROMA integration is properly wired and ready for use.');
  process.exit(0);
} else {
  console.log('\n✗ Some wiring checks failed. Please review the issues above.');
  process.exit(1);
}
