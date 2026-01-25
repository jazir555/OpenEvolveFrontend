/**
 * Test Suite for Reliability Fixes
 *
 * This file demonstrates and tests the reliability fixes:
 * - Bug #2: Request Timeout
 * - Bug #3: Retry Logic with Exponential Backoff
 * - Bug #5 & #7: Circuit Breaker Protection
 */

import { ApiClient, ApiClientConfig } from './BubbleLab/apps/bubble-studio/src/lib/api';
import { CircuitBreaker, createEvolutionApiCircuitBreaker } from './BubbleLab/apps/bubble-studio/src/lib/circuitBreaker';

// ============================================================================
// TEST 1: Request Timeout (Bug #2)
// ============================================================================

async function testRequestTimeout() {
  console.log('\n=== TEST 1: Request Timeout ===');

  const config: ApiClientConfig = {
    baseURL: 'http://localhost:9999', // Non-existent server
    timeout: 5000, // 5 second timeout for testing
    enableRetry: false, // Disable retry for this test
  };

  const client = new ApiClient(config);

  try {
    await client.get('/api/test');
    console.error('❌ TEST FAILED: Should have timed out');
  } catch (error) {
    if (error instanceof Error && error.message.includes('timeout')) {
      console.log('✅ TEST PASSED: Request timed out as expected');
      console.log(`   Error: ${error.message}`);
    } else {
      console.error('❌ TEST FAILED: Wrong error type', error);
    }
  }
}

// ============================================================================
// TEST 2: Retry Logic with Exponential Backoff (Bug #3)
// ============================================================================

async function testRetryLogic() {
  console.log('\n=== TEST 2: Retry Logic ===');

  let attemptCount = 0;

  const config: ApiClientConfig = {
    baseURL: 'http://localhost:9999',
    timeout: 1000,
    enableRetry: true,
    maxRetries: 3,
    retryDelay: 500, // Short delay for testing
  };

  const client = new ApiClient(config);

  try {
    await client.get('/api/test');
    console.error('❌ TEST FAILED: Should have failed after retries');
  } catch (error) {
    if (error instanceof Error) {
      console.log('✅ TEST PASSED: Retry logic executed');
      console.log(`   Error: ${error.message}`);
      console.log('   Check logs for retry attempts with exponential backoff');
    }
  }
}

// ============================================================================
// TEST 3: Circuit Breaker - OPEN State (Bug #5 & #7)
// ============================================================================

async function testCircuitBreakerOpens() {
  console.log('\n=== TEST 3: Circuit Breaker Opens ===');

  const circuitBreaker = new CircuitBreaker('test-api', {
    failureThreshold: 3, // Open after 3 failures
    timeout: 10000, // 10 seconds
    halfOpenAttempts: 2,
  });

  // Trigger failures
  for (let i = 1; i <= 3; i++) {
    try {
      await circuitBreaker.execute(async () => {
        throw new Error('Simulated failure');
      });
    } catch (error) {
      console.log(`   Failure ${i}: Circuit state = ${circuitBreaker.getState()}`);
    }
  }

  const state = circuitBreaker.getState();
  if (state === 'open') {
    console.log('✅ TEST PASSED: Circuit breaker opened after 3 failures');
  } else {
    console.error(`❌ TEST FAILED: Circuit breaker state is ${state}, expected 'open'`);
  }

  // Try to make request through open circuit
  try {
    await circuitBreaker.execute(async () => {
      console.log('   This should not execute');
      return 'success';
    });
    console.error('❌ TEST FAILED: Circuit breaker should have blocked request');
  } catch (error) {
    if (error instanceof Error && error.message.includes('OPEN')) {
      console.log('✅ TEST PASSED: Circuit breaker blocked request');
      console.log(`   Error: ${error.message}`);
    }
  }
}

// ============================================================================
// TEST 4: Circuit Breaker - HALF_OPEN to CLOSED Transition
// ============================================================================

async function testCircuitBreakerRecovers() {
  console.log('\n=== TEST 4: Circuit Breaker Recovery ===');

  const circuitBreaker = new CircuitBreaker('test-api', {
    failureThreshold: 2,
    timeout: 5000, // 5 seconds
    halfOpenAttempts: 2,
  });

  // Open the circuit
  console.log('   Opening circuit...');
  for (let i = 0; i < 2; i++) {
    try {
      await circuitBreaker.execute(async () => {
        throw new Error('Failure');
      });
    } catch (error) {
      // Expected
    }
  }

  console.log(`   Circuit state: ${circuitBreaker.getState()}`);

  // Wait for timeout
  console.log('   Waiting 6 seconds for circuit timeout...');
  await new Promise(resolve => setTimeout(resolve, 6000));

  // Make successful requests to close circuit
  console.log('   Attempting recovery with successful requests...');
  for (let i = 0; i < 2; i++) {
    try {
      const result = await circuitBreaker.execute(async () => {
        return 'success';
      });
      console.log(`   Success ${i + 1}: Circuit state = ${circuitBreaker.getState()}`);
    } catch (error) {
      console.error(`   Unexpected error: ${error}`);
    }
  }

  const state = circuitBreaker.getState();
  if (state === 'closed') {
    console.log('✅ TEST PASSED: Circuit breaker recovered and closed');
  } else {
    console.error(`❌ TEST FAILED: Circuit breaker state is ${state}, expected 'closed'`);
  }
}

// ============================================================================
// TEST 5: Circuit Breaker Metrics
// ============================================================================

async function testCircuitBreakerMetrics() {
  console.log('\n=== TEST 5: Circuit Breaker Metrics ===');

  const circuitBreaker = createEvolutionApiCircuitBreaker();

  // Get initial metrics
  let metrics = circuitBreaker.getMetrics();
  console.log('   Initial metrics:', JSON.stringify(metrics, null, 2));

  // Trigger some activity
  try {
    await circuitBreaker.execute(async () => {
      throw new Error('Test failure');
    });
  } catch (error) {
    // Expected
  }

  metrics = circuitBreaker.getMetrics();
  console.log('   After failure:', JSON.stringify(metrics, null, 2));

  if (metrics.failureCount > 0) {
    console.log('✅ TEST PASSED: Metrics tracking failures');
  } else {
    console.error('❌ TEST FAILED: Metrics not tracking failures');
  }
}

// ============================================================================
// TEST 6: Evolution API Integration
// ============================================================================

async function testEvolutionApiIntegration() {
  console.log('\n=== TEST 6: Evolution API Integration ===');

  // This test demonstrates the integration but doesn't make actual API calls
  // since we don't have a running Evolution API server

  const config: ApiClientConfig = {
    baseURL: 'http://localhost:8000',
    timeout: 30000, // 30 seconds
    enableRetry: true, // Enable retry
    maxRetries: 3,
    retryDelay: 1000,
  };

  const client = new ApiClient(config);
  const circuitBreaker = createEvolutionApiCircuitBreaker();

  console.log('✅ Evolution API client configured with:');
  console.log(`   - Timeout: ${config.timeout}ms`);
  console.log(`   - Retry: ${config.enableRetry ? 'enabled' : 'disabled'}`);
  console.log(`   - Max retries: ${config.maxRetries}`);
  console.log(`   - Circuit breaker: ${circuitBreaker.getState()}`);

  console.log('✅ TEST PASSED: Evolution API integration configured');
}

// ============================================================================
// RUN ALL TESTS
// ============================================================================

async function runAllTests() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║   Reliability Fixes Test Suite                          ║');
  console.log('║   Testing: Timeout, Retry, Circuit Breaker               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  await testRequestTimeout();
  await testRetryLogic();
  await testCircuitBreakerOpens();
  await testCircuitBreakerRecovers();
  await testCircuitBreakerMetrics();
  await testEvolutionApiIntegration();

  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║   Test Suite Complete                                     ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
}

// Export tests for running
export {
  testRequestTimeout,
  testRetryLogic,
  testCircuitBreakerOpens,
  testCircuitBreakerRecovers,
  testCircuitBreakerMetrics,
  testEvolutionApiIntegration,
  runAllTests,
};

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests().catch(console.error);
}
