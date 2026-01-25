/**
 * Test Runner for Reliability Tests
 *
 * This script runs the reliability tests for timeout, retry logic, and circuit breaker.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';

// Import test modules
import './timeout.test';
import './retry.test';
import './circuit-breaker.test';
import './integration.test';

describe('Reliability Test Suite', () => {
  beforeAll(() => {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║   Reliability Fixes Test Suite                          ║');
    console.log('║   Testing: Timeout, Retry, Circuit Breaker               ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
  });

  afterAll(() => {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║   Test Suite Complete                                     ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
  });
});
