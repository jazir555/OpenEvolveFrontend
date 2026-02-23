/**
 * Comprehensive Functionality Test
 * Tests actual runtime behavior of all core components
 */

import { z } from 'zod';

// Test 1: Import Verification
console.log('\n=== TEST 1: Import Verification ===');

try {
  // Test lib imports
  const {
    Logger,
    retry,
    CircuitBreaker,
    validateEnv
  } = require('./glue/lib/index.js');

  // Test schemas imports
  const {
    LoongFlowSolutionSchema,
    BubbleLabSolutionSchema,
    RESESearchResultSchema,
    HybridPESStateSchema
  } = require('./glue/schemas/index.js');

  // Test orchestration imports
  const {
    InMemoryEventBus,
    DeadLetterQueue
  } = require('./glue/orchestration/event-bus.js');

  // Test adapter imports
  let LoongFlowAdapter;
  try {
    LoongFlowAdapter = require('./glue/adapters/loongflow-adapter/dist/index.js').LoongFlowAdapter;
    console.log('✓ LoongFlow Adapter imports successfully');
  } catch (e) {
    console.log('⚠️  LoongFlow Adapter import:', e.message);
  }

  console.log('✓ All core imports successful');
} catch (error) {
  console.log('✗ Import failed:', error.message);
  process.exit(1);
}

// Test 2: Schema Validation
console.log('\n=== TEST 2: Schema Validation ===');

try {
  const {
    LoongFlowSolutionSchema,
    BubbleLabSolutionSchema,
    RESESearchResultSchema,
    HybridPESStateSchema
  } = require('./glue/schemas/index.js');

  // Test LoongFlow schema
  const loongflowTest = {
    solution: "x^2 + 2x + 1 = 0",
    solution_id: "test-loongflow-123",
    score: 0.95,
    metadata: {
      timestamp: new Date().toISOString(),
      provenance: "loongflow"
    }
  };

  const loongflowResult = LoongFlowSolutionSchema.safeParse(loongflowTest);
  console.log(`  LoongFlow Schema: ${loongflowResult.success ? '✓ Pass' : '✗ Fail'}`);
  if (!loongflowResult.success) {
    console.log('    Error:', loongflowResult.error.errors[0].message);
  }

  // Test BubbleLab schema
  const bubblelabTest = {
    solution: "test solution",
    solution_id: "test-bubblelab-456",
    score: 0.87,
    metadata: {
      timestamp: new Date().toISOString(),
      provenance: "bubblelab"
    }
  };

  const bubblelabResult = BubbleLabSolutionSchema.safeParse(bubblelabTest);
  console.log(`  BubbleLab Schema: ${bubblelabResult.success ? '✓ Pass' : '✗ Fail'}`);
  if (!bubblelabResult.success) {
    console.log('    Error:', bubblelabResult.error.errors[0].message);
  }

  // Test RESE schema
  const reseTest = {
    search_results: [
      {
        content: "test content",
        score: 0.92,
        metadata: {
          source: "test",
          timestamp: new Date().toISOString()
        }
      }
    ],
    query: "test query",
    total_results: 1,
    metadata: {
      timestamp: new Date().toISOString(),
      search_strategy: "hybrid"
    }
  };

  const reseResult = RESESearchResultSchema.safeParse(reseTest);
  console.log(`  RESE Schema: ${reseResult.success ? '✓ Pass' : '✗ Fail'}`);
  if (!reseResult.success) {
    console.log('    Error:', reseResult.error.errors[0].message);
  }

  // Test Hybrid PES schema
  const pesTest = {
    current_stage: "exploration",
    solutions: [],
    evolution_history: [],
    metadata: {
      timestamp: new Date().toISOString(),
      workflow_id: "test-workflow-123"
    }
  };

  const pesResult = HybridPESStateSchema.safeParse(pesTest);
  console.log(`  Hybrid PES Schema: ${pesResult.success ? '✓ Pass' : '✗ Fail'}`);
  if (!pesResult.success) {
    console.log('    Error:', pesResult.error.errors[0].message);
  }

  console.log('✓ Schema validation complete');
} catch (error) {
  console.log('✗ Schema validation failed:', error.message);
}

// Test 3: Event Bus Functionality
console.log('\n=== TEST 3: Event Bus Functionality ===');

(async () => {
  try {
    const { InMemoryEventBus, DeadLetterQueue } = require('./glue/orchestration/event-bus.js');

    const bus = new InMemoryEventBus();

    // Subscribe to test events
    let eventReceived = false;
    bus.subscribe('test', async (event) => {
      eventReceived = true;
      console.log(`  ✓ Received event: ${event.type}`);
    });

    // Publish test event
    await bus.publish({
      type: 'test',
      correlationId: 'test-123',
      timestamp: new Date(),
      data: { test: true }
    });

    // Wait a bit for async processing
    await new Promise(resolve => setTimeout(resolve, 100));

    if (eventReceived) {
      console.log('✓ Event bus publish/subscribe works');
    } else {
      console.log('✗ Event bus publish/subscribe failed - no event received');
    }

    // Test DLQ
    const dlq = new DeadLetterQueue();
    await dlq.add({
      type: 'failed-event',
      correlationId: 'dlq-test-123',
      timestamp: new Date(),
      data: { error: 'test error' },
      error: new Error('Test error')
    });

    const failedEvents = await dlq.getFailedEvents();
    console.log(`✓ Dead Letter Queue works (${failedEvents.length} events)`);

  } catch (error) {
    console.log('✗ Event bus test failed:', error.message);
    console.log('   Stack:', error.stack);
  }
})();

// Test 4: Logger Functionality
console.log('\n=== TEST 4: Logger Functionality ===');

try {
  const { Logger } = require('./glue/lib/index.js');

  const logger = new Logger('test-component');

  logger.info({ msg: 'Test info log', correlation_id: 'test-123' });
  logger.warn({ msg: 'Test warning', correlation_id: 'test-456' });
  logger.error({ msg: 'Test error', correlation_id: 'test-789', error: 'Test error message' });

  console.log('✓ Logger works (check above for log output)');
} catch (error) {
  console.log('✗ Logger test failed:', error.message);
}

// Test 5: Retry Logic
console.log('\n=== TEST 5: Retry Logic ===');

(async () => {
  try {
    const { retry } = require('./glue/lib/index.js');

    let attempts = 0;
    const result = await retry({
      maxAttempts: 3,
      initialDelay: 100,
      operation: async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Not yet');
        }
        return 'success';
      }
    });

    console.log(`✓ Retry logic works (succeeded after ${attempts} attempts)`);
  } catch (error) {
    console.log('✗ Retry test failed:', error.message);
  }
})();

// Test 6: Circuit Breaker
console.log('\n=== TEST 6: Circuit Breaker ===');

try {
  const { CircuitBreaker } = require('./glue/lib/index.js');

  const breaker = new CircuitBreaker('test-service', {
    failureThreshold: 2,
    resetTimeout: 1000
  });

  // Test successful call
  breaker.execute(async () => {
    return 'success';
  }).then(result => {
    console.log(`✓ Circuit breaker allows successful calls`);
  });

  // Test failures
  (async () => {
    for (let i = 0; i < 3; i++) {
      try {
        await breaker.execute(async () => {
          throw new Error('Service failure');
        });
      } catch (e) {
        // Expected failures
      }
    }

    // Circuit should be open now
    try {
      await breaker.execute(async () => {
        return 'should not execute';
      });
      console.log('⚠️  Circuit breaker did not open as expected');
    } catch (e) {
      if (e.message.includes('Circuit breaker is OPEN')) {
        console.log('✓ Circuit breaker opens after threshold');
      } else {
        console.log('⚠️  Unexpected error:', e.message);
      }
    }
  })();

} catch (error) {
  console.log('✗ Circuit breaker test failed:', error.message);
}

// Test 7: Environment Validation
console.log('\n=== TEST 7: Environment Validation ===');

try {
  const { validateEnv } = require('./glue/lib/index.js');

  // Test with missing env vars
  const originalEnv = process.env.TEST_VAR;
  delete process.env.TEST_VAR;

  try {
    validateEnv({
      TEST_VAR: 'required test variable'
    });
    console.log('⚠️  Env validation did not catch missing variable');
  } catch (e) {
    console.log('✓ Environment validation catches missing vars');
  }

  // Test with present env var
  process.env.TEST_VAR = 'test-value';
  try {
    validateEnv({
      TEST_VAR: 'required test variable'
    });
    console.log('✓ Environment validation passes with valid vars');
  } catch (e) {
    console.log('✗ Environment validation failed unexpectedly:', e.message);
  }

  // Restore original
  if (originalEnv !== undefined) {
    process.env.TEST_VAR = originalEnv;
  } else {
    delete process.env.TEST_VAR;
  }

} catch (error) {
  console.log('✗ Environment validation test failed:', error.message);
}

// Test 8: LoongFlow Adapter (if available)
console.log('\n=== TEST 8: LoongFlow Adapter ===');

(async () => {
  try {
    const { LoongFlowAdapter } = require('./glue/adapters/loongflow-adapter/dist/index.js');

    const adapter = new LoongFlowAdapter({
      baseUrl: 'http://localhost:8000',
      timeout: 5000
    });

    console.log('✓ LoongFlow Adapter instantiates successfully');
    console.log('  Note: API calls will fail without running container');

  } catch (error) {
    console.log('⚠️  LoongFlow Adapter test:', error.message);
  }
})();

// Give async tests time to complete
setTimeout(() => {
  console.log('\n=== All Tests Complete ===');
}, 1500);
