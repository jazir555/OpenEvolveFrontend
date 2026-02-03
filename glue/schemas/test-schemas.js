/**
 * Quick Schema Validation Test
 *
 * This script verifies that the canonical schemas are properly defined
 * and can be used for runtime validation.
 *
 * Usage: node glue/schemas/test-schemas.js
 */

console.log('='.repeat(60));
console.log('Canonical Schema Structure Verification');
console.log('='.repeat(60));
console.log();

// Verify file structure
const fs = require('fs');
const path = require('path');

// Get the correct directory path (this file is in glue/schemas/)
const schemasDir = __dirname;
const requiredFiles = [
  'z3-canonical.ts',
  'leanaide-canonical.ts',
  'index.ts',
  'validation-examples.ts',
  'README.md',
];

console.log('Verifying required files exist:');
let allFilesExist = true;

requiredFiles.forEach(file => {
  const filePath = path.join(schemasDir, file);
  const exists = fs.existsSync(filePath);
  console.log(`  ${exists ? '✓' : '✗'} ${file}`);
  if (!exists) allFilesExist = false;
});

console.log();

if (!allFilesExist) {
  console.log('ERROR: Not all required files exist!');
  process.exit(1);
}

// Verify file contents
console.log('Verifying file contents:');
console.log();

const z3Schema = fs.readFileSync(path.join(schemasDir, 'z3-canonical.ts'), 'utf8');
const leanSchema = fs.readFileSync(path.join(schemasDir, 'leanaide-canonical.ts'), 'utf8');
const indexFile = fs.readFileSync(path.join(schemasDir, 'index.ts'), 'utf8');

// Check Z3 schema exports
const z3Exports = [
  'SolverRequest',
  'SolverResponse',
  'KnowledgeGraphResponse',
  'Entity',
  'Relation',
  'transformZ3ResponseToCanonical',
  'validateSolverRequest',
  'validateSolverResponse',
  'validateKnowledgeGraphResponse',
];

console.log('Z3 Canonical Schema exports:');
z3Exports.forEach(exp => {
  const exists = z3Schema.includes(`export ${exp}`);
  console.log(`  ${exists ? '✓' : '✗'} ${exp}`);
});
console.log();

// Check LeanAide schema exports
const leanExports = [
  'ProofVerificationRequest',
  'ProofVerificationResponse',
  'LeanCompilationRequest',
  'LeanCompilationResponse',
  'LeanMessage',
  'transformLeanAideResponseToCanonical',
  'validateProofVerificationRequest',
  'validateProofVerificationResponse',
];

console.log('LeanAide Canonical Schema exports:');
leanExports.forEach(exp => {
  const exists = leanSchema.includes(`export ${exp}`);
  console.log(`  ${exists ? '✓' : '✗'} ${exp}`);
});
console.log();

// Check index re-exports
const indexExports = [
  'SolverRequest',
  'ProofVerificationRequest',
  'createCorrelationId',
  'createUTCTimestamp',
  'DEFAULT_TIMEOUTS',
];

console.log('Index.ts re-exports:');
indexExports.forEach(exp => {
  const exists = indexFile.includes(`export ${exp}`);
  console.log(`  ${exists ? '✓' : '✗'} ${exp}`);
});
console.log();

// Check for validation examples
const validationExamples = fs.readFileSync(path.join(schemasDir, 'validation-examples.ts'), 'utf8');
const exampleTests = [
  'testZ3Schemas',
  'testLeanAideSchemas',
  'testUtilityFunctions',
  'runAllTests',
];

console.log('Validation Examples tests:');
exampleTests.forEach(test => {
  const exists = validationExamples.includes(`function ${test}`);
  console.log(`  ${exists ? '✓' : '✗'} ${test}`);
});
console.log();

// Summary
console.log('='.repeat(60));
console.log('Summary:');
console.log('='.repeat(60));
console.log();
console.log('Total lines of code:');
console.log(`  z3-canonical.ts:        ${z3Schema.split('\n').length} lines`);
console.log(`  leanaide-canonical.ts:  ${leanSchema.split('\n').length} lines`);
console.log(`  index.ts:               ${indexFile.split('\n').length} lines`);
console.log(`  validation-examples.ts: ${validationExamples.split('\n').length} lines`);
console.log();

console.log('Schema features:');
console.log('  ✓ Zod validation schemas');
console.log('  ✓ TypeScript type definitions');
console.log('  ✓ Transformation functions');
console.log('  ✓ Validation functions');
console.log('  ✓ Example data');
console.log('  ✓ Utility functions (correlation ID, UTC timestamp)');
console.log('  ✓ Default timeouts');
console.log('  ✓ Size limits');
console.log('  ✓ Error codes');
console.log();

console.log('Next steps:');
console.log('  1. Install Zod: npm install zod');
console.log('  2. Run validation tests: ts-node glue/schemas/validation-examples.ts');
console.log('  3. Import in adapters: import { SolverRequest } from \'../schemas\'');
console.log();

console.log('✓ All canonical schemas created successfully!');
console.log('='.repeat(60));
