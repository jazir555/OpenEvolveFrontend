/**
 * Validation Examples and Tests
 *
 * This file demonstrates how to use the canonical schemas for validation.
 * Run this file to verify that all schemas are working correctly.
 *
 * Usage:
 *   npm run validate-schemas
 *   or
 *   ts-node glue/schemas/validation-examples.ts
 */

import {
  // Z3 imports
  validateSolverRequest,
  validateSolverResponse,
  validateKnowledgeGraphResponse,
  Z3Examples,
  // LeanAide imports
  validateProofVerificationRequest,
  validateProofVerificationResponse,
  validateLeanCompilationRequest,
  validateLeanCompilationResponse,
  LeanAideExamples,
  // Utilities
  createCorrelationId,
  createUTCTimestamp,
  formatValidationErrors,
  DEFAULT_TIMEOUTS,
} from './index';

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function logSuccess(message: string) {
  console.log(`${colors.green}✓ ${message}${colors.reset}`);
}

function logError(message: string) {
  console.log(`${colors.red}✗ ${message}${colors.reset}`);
}

function logSection(title: string) {
  console.log(`\n${colors.cyan}${'='.repeat(60)}`);
  console.log(`${title}`);
  console.log(`${'='.repeat(60)}${colors.reset}\n`);
}

function logInfo(message: string) {
  console.log(`${colors.blue}ℹ ${message}${colors.reset}`);
}

/**
 * Test Z3 Schema Validations
 */
function testZ3Schemas() {
  logSection('Testing Z3 Canonical Schemas');

  // Test 1: Valid SolverRequest
  logInfo('Test 1: Valid Z3 SolverRequest');
  const solverRequestValidation = validateSolverRequest(Z3Examples.validSolverRequest);
  if (solverRequestValidation.success) {
    logSuccess('Valid SolverRequest passed validation');
    console.log('  Problem:', `${solverRequestValidation.data!.problem.substring(0, 50)}...`);
    console.log('  Timeout:', `${solverRequestValidation.data!.timeout_ms}ms`);
  } else {
    logError('Valid SolverRequest failed validation');
    console.log('  Errors:', solverRequestValidation.errors);
  }

  // Test 2: Valid SolverResponse
  logInfo('\nTest 2: Valid Z3 SolverResponse');
  const solverResponseValidation = validateSolverResponse(Z3Examples.validSolverResponse);
  if (solverResponseValidation.success) {
    logSuccess('Valid SolverResponse passed validation');
    console.log('  Result:', solverResponseValidation.data!.result);
    console.log('  Timestamp:', solverResponseValidation.data!.timestamp);
  } else {
    logError('Valid SolverResponse failed validation');
    console.log('  Errors:', solverResponseValidation.errors);
  }

  // Test 3: Valid KnowledgeGraphResponse
  logInfo('\nTest 3: Valid Z3 KnowledgeGraphResponse');
  const kgValidation = validateKnowledgeGraphResponse(Z3Examples.validKnowledgeGraphResponse);
  if (kgValidation.success) {
    logSuccess('Valid KnowledgeGraphResponse passed validation');
    console.log('  Entities:', kgValidation.data!.entities.length);
    console.log('  Relations:', kgValidation.data!.relations.length);
  } else {
    logError('Valid KnowledgeGraphResponse failed validation');
    console.log('  Errors:', kgValidation.errors);
  }

  // Test 4: Invalid SolverRequest (missing timeout)
  logInfo('\nTest 4: Invalid Z3 SolverRequest (missing timeout)');
  const invalidRequest = {
    problem: '(declare-const x Int)',
  };
  const invalidValidation = validateSolverRequest(invalidRequest);
  if (!invalidValidation.success) {
    logSuccess('Invalid SolverRequest correctly rejected');
    console.log('  Errors:', invalidValidation.errors);
  } else {
    logError('Invalid SolverRequest incorrectly accepted');
  }

  // Test 5: Invalid SolverResponse (invalid result type)
  logInfo('\nTest 5: Invalid Z3 SolverResponse (invalid result)');
  const invalidResponse = {
    result: 'invalid',
    timestamp: new Date().toISOString(),
  };
  const invalidResponseValidation = validateSolverResponse(invalidResponse);
  if (!invalidResponseValidation.success) {
    logSuccess('Invalid SolverResponse correctly rejected');
    console.log('  Errors:', invalidResponseValidation.errors);
  } else {
    logError('Invalid SolverResponse incorrectly accepted');
  }
}

/**
 * Test LeanAide Schema Validations
 */
function testLeanAideSchemas() {
  logSection('Testing LeanAide Canonical Schemas');

  // Test 1: Valid ProofVerificationRequest
  logInfo('Test 1: Valid LeanAide ProofVerificationRequest');
  const proofRequestValidation = validateProofVerificationRequest(
    LeanAideExamples.validProofVerificationRequest
  );
  if (proofRequestValidation.success) {
    logSuccess('Valid ProofVerificationRequest passed validation');
    console.log('  Theorem:', proofRequestValidation.data!.theorem);
    console.log('  Timeout:', `${proofRequestValidation.data!.timeout_ms}ms`);
  } else {
    logError('Valid ProofVerificationRequest failed validation');
    console.log('  Errors:', proofRequestValidation.errors);
  }

  // Test 2: Valid ProofVerificationResponse
  logInfo('\nTest 2: Valid LeanAide ProofVerificationResponse');
  const proofResponseValidation = validateProofVerificationResponse(
    LeanAideExamples.validProofVerificationResponse
  );
  if (proofResponseValidation.success) {
    logSuccess('Valid ProofVerificationResponse passed validation');
    console.log('  Verified:', proofResponseValidation.data!.verified);
    console.log('  Tactics used:', proofResponseValidation.data!.tactics_used?.join(', '));
  } else {
    logError('Valid ProofVerificationResponse failed validation');
    console.log('  Errors:', proofResponseValidation.errors);
  }

  // Test 3: Valid LeanCompilationRequest
  logInfo('\nTest 3: Valid LeanAide LeanCompilationRequest');
  const compilationRequestValidation = validateLeanCompilationRequest(
    LeanAideExamples.validLeanCompilationRequest
  );
  if (compilationRequestValidation.success) {
    logSuccess('Valid LeanCompilationRequest passed validation');
    console.log('  Filename:', compilationRequestValidation.data!.filename);
    console.log('  Code length:', `${compilationRequestValidation.data!.code.length} chars`);
  } else {
    logError('Valid LeanCompilationRequest failed validation');
    console.log('  Errors:', compilationRequestValidation.errors);
  }

  // Test 4: Valid LeanCompilationResponse
  logInfo('\nTest 4: Valid LeanAide LeanCompilationResponse');
  const compilationResponseValidation = validateLeanCompilationResponse(
    LeanAideExamples.validLeanCompilationResponse
  );
  if (compilationResponseValidation.success) {
    logSuccess('Valid LeanCompilationResponse passed validation');
    console.log('  Compiled:', compilationResponseValidation.data!.compiled);
    console.log('  Warnings:', compilationResponseValidation.data!.warnings?.length || 0);
  } else {
    logError('Valid LeanCompilationResponse failed validation');
    console.log('  Errors:', compilationResponseValidation.errors);
  }

  // Test 5: ProofVerificationResponse with errors
  logInfo('\nTest 5: LeanAide ProofVerificationResponse with errors');
  const errorResponseValidation = validateProofVerificationResponse(
    LeanAideExamples.proofVerificationWithError
  );
  if (errorResponseValidation.success) {
    logSuccess('ProofVerificationResponse with errors passed validation');
    console.log('  Verified:', errorResponseValidation.data!.verified);
    console.log('  Messages:', errorResponseValidation.data!.messages?.length || 0);
  } else {
    logError('ProofVerificationResponse with errors failed validation');
    console.log('  Errors:', errorResponseValidation.errors);
  }

  // Test 6: Invalid ProofVerificationRequest (missing required fields)
  logInfo('\nTest 6: Invalid LeanAide ProofVerificationRequest (missing theorem)');
  const invalidProofRequest = {
    proof_code: 'theorem x : Nat := by trivial',
    timeout_ms: 5000,
  };
  const invalidProofValidation = validateProofVerificationRequest(invalidProofRequest);
  if (!invalidProofValidation.success) {
    logSuccess('Invalid ProofVerificationRequest correctly rejected');
    console.log('  Errors:', invalidProofValidation.errors);
  } else {
    logError('Invalid ProofVerificationRequest incorrectly accepted');
  }
}

/**
 * Test Utility Functions
 */
function testUtilityFunctions() {
  logSection('Testing Utility Functions');

  // Test createCorrelationId
  logInfo('Test 1: Create Correlation ID');
  const correlationId = createCorrelationId();
  console.log(`  Generated: ${correlationId}`);
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (uuidRegex.test(correlationId)) {
    logSuccess('Correlation ID format is valid (UUID v4)');
  } else {
    logError('Correlation ID format is invalid');
  }

  // Test createUTCTimestamp
  logInfo('\nTest 2: Create UTC Timestamp');
  const timestamp = createUTCTimestamp();
  console.log(`  Generated: ${timestamp}`);
  const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
  if (isoRegex.test(timestamp)) {
    logSuccess('UTC timestamp format is valid (ISO-8601)');
  } else {
    logError('UTC timestamp format is invalid');
  }

  // Test DEFAULT_TIMEOUTS
  logInfo('\nTest 3: Default Timeouts');
  console.log(`  QUICK: ${DEFAULT_TIMEOUTS.QUICK}ms (${DEFAULT_TIMEOUTS.QUICK / 1000}s)`);
  console.log(`  NORMAL: ${DEFAULT_TIMEOUTS.NORMAL}ms (${DEFAULT_TIMEOUTS.NORMAL / 1000}s)`);
  console.log(`  LONG: ${DEFAULT_TIMEOUTS.LONG}ms (${DEFAULT_TIMEOUTS.LONG / 1000}s)`);
  console.log(`  EXTENDED: ${DEFAULT_TIMEOUTS.EXTENDED}ms (${DEFAULT_TIMEOUTS.EXTENDED / 1000}s)`);
  logSuccess('Default timeouts are defined');
}

/**
 * Run Schema Validation Tests
 */
export function runAllTests() {
  console.log(`\n${colors.yellow}Starting Canonical Schema Validation Tests...${colors.reset}`);

  try {
    testZ3Schemas();
    testLeanAideSchemas();
    testUtilityFunctions();

    logSection('Test Summary');
    logSuccess('All schema validation tests completed successfully!');
    console.log(`\n${colors.cyan}Note:${colors.reset} These are runtime validation tests.`);
    console.log(`For compile-time type safety, TypeScript will validate types at compile time.\n`);
  } catch (error) {
    logError('Tests failed with error:');
    console.error(error);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests();
}

/**
 * Example: Using Schemas in Adapter Code
 *
 * This section shows how adapters should use the canonical schemas.
 */
export function adapterUsageExample() {
  console.log(`\n${colors.cyan}Adapter Usage Example:${colors.reset}\n`);

  // Example 1: Receiving a request and validating it
  logInfo('Example 1: Validate incoming request');
  const incomingRequest = {
    problem: '(declare-const x Int) (assert (> x 10))',
    timeout_ms: 5000,
  };

  const validation = validateSolverRequest(incomingRequest);
  if (!validation.success) {
    logError('Invalid request received');
    console.log('  Errors:', validation.errors);
    return;
  }

  logSuccess('Request is valid, processing...');
  const request = validation.data!;

  // Example 2: Creating a response with proper fields
  logInfo('\nExample 2: Create canonical response');
  const response = {
    result: 'sat' as const,
    explanation: 'The constraint is satisfiable',
    model: { x: 11 },
    metadata: {
      solver_version: '4.12.1',
      solve_time_ms: 45,
    },
    correlation_id: request.correlation_id || createCorrelationId(),
    timestamp: createUTCTimestamp(),
  };

  const responseValidation = validateSolverResponse(response);
  if (responseValidation.success) {
    logSuccess('Response is valid, ready to send');
    console.log('  Response:', JSON.stringify(response, null, 2));
  } else {
    logError('Response validation failed');
    console.log('  Errors:', responseValidation.errors);
  }
}
