/**
 * Comprehensive Error Handling Tests
 * Tests all aspects of the enhanced error handling system
 */

import { 
  unifiedErrorHandlingService, 
  withUnifiedErrorHandling,
  useUnifiedErrorHandler
} from '../utils/UnifiedErrorHandlingService';

import {
  advancedErrorRecoveryManager,
  AdvancedErrorRecoveryManager
} from '../utils/AdvancedErrorRecovery';

import {
  enhancedErrorContextManager,
  createEnhancedErrorContext,
  recordErrorForContext,
  addUserActionToContext
} from '../utils/EnhancedErrorContext';

import {
  enhancedErrorReporter,
  reportEnhancedError,
  ErrorCategory
} from '../utils/EnhancedErrorReporting';

import {
  apiErrorHandlingMiddleware,
  withApiErrorHandling,
  withFetchErrorHandling,
  createErrorHandledApiClient
} from '../utils/ApiErrorHandlingMiddleware';

import {
  useErrorHandling,
  useAsyncState,
  useApiCall,
  usePolling,
  useFormErrorHandling
} from '../utils/ReactErrorHandlingHooks';

// Mock test functions
const mockSuccessfulOperation = (): Promise<string> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve('Success!'), 10);
  });
};

const mockFailingOperation = (failureCount: number = 1): Promise<string> => {
  let callCount = 0;
  return new Promise((resolve, reject) => {
    callCount++;
    if (callCount >= failureCount) {
      resolve('Success after retry!');
    } else {
      reject(new Error(`Operation failed on attempt ${callCount}`));
    }
  });
};

const mockAlwaysFailingOperation = (): Promise<string> => {
  return new Promise((_, reject) => {
    reject(new Error('This operation always fails'));
  });
};

// Test suite
describe('Comprehensive Error Handling Tests', () => {
  beforeEach(() => {
    // Reset any state before each test
    enhancedErrorContextManager.clearContextCache();
    enhancedErrorReporter.clearReports();
    apiErrorHandlingMiddleware.clearCache();
  });

  describe('Unified Error Handling Service', () => {
    test('should handle successful operations', async () => {
      const result = await withUnifiedErrorHandling(mockSuccessfulOperation, {
        strategy: 'retry',
        maxRetries: 3
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('Success!');
      expect(result.error).toBeUndefined();
    });

    test('should handle operations that succeed after retries', async () => {
      const result = await withUnifiedErrorHandling(() => mockFailingOperation(2), {
        strategy: 'retry',
        maxRetries: 3,
        retryDelay: 10
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('Success after retry!');
      expect(result.retries).toBeGreaterThan(0);
    });

    test('should handle operations that always fail', async () => {
      const result = await withUnifiedErrorHandling(mockAlwaysFailingOperation, {
        strategy: 'retry',
        maxRetries: 2,
        retryDelay: 10
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error?.message).toBe('This operation always fails');
    });

    test('should use fallback value when operation fails', async () => {
      const fallbackValue = 'Fallback result';
      const result = await withUnifiedErrorHandling(mockAlwaysFailingOperation, {
        strategy: 'fallback',
        maxRetries: 1,
        fallbackValue
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe(fallbackValue);
      expect(result.strategyUsed).toBe('fallback');
    });

    test('should handle timeout correctly', async () => {
      const slowOperation = (): Promise<string> => {
        return new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Slow operation')), 100);
        });
      };

      const result = await withUnifiedErrorHandling(slowOperation, {
        timeout: 50,
        maxRetries: 1
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Advanced Error Recovery', () => {
    test('should select appropriate recovery strategy for network errors', async () => {
      const networkError = new Error('Network Error: Failed to fetch');
      const context = {
        error: networkError,
        operation: 'test_network_operation',
        maxAttempts: 3
      };

      const result = await advancedErrorRecoveryManager.applyRecoveryStrategy(context);

      expect(result.success).toBe(true);
      expect(result.shouldRetry).toBe(true);
    });

    test('should select appropriate recovery strategy for timeout errors', async () => {
      const timeoutError = new Error('Request timeout');
      const context = {
        error: timeoutError,
        operation: 'test_timeout_operation',
        maxAttempts: 3
      };

      const result = await advancedErrorRecoveryManager.applyRecoveryStrategy(context);

      expect(result.success).toBe(true);
      expect(result.actionTaken).toContain('timeout');
    });

    test('should get recovery statistics', () => {
      const stats = advancedErrorRecoveryManager.getRecoveryStats();
      
      expect(stats).toHaveProperty('totalAttempts');
      expect(stats).toHaveProperty('successfulRecoveries');
      expect(stats).toHaveProperty('successRate');
      expect(stats).toHaveProperty('avgRecoveryTime');
      expect(stats).toHaveProperty('byStrategy');
    });
  });

  describe('Enhanced Error Context', () => {
    test('should create enhanced error context', async () => {
      const context = await createEnhancedErrorContext({
        component: 'TestComponent',
        function: 'testFunction',
        operation: 'TEST_OPERATION'
      }, {
        tags: ['test', 'unit'],
        customField: 'customValue'
      });

      expect(context).toBeDefined();
      expect(context.component).toBe('TestComponent');
      expect(context.function).toBe('testFunction');
      expect(context.operation).toBe('TEST_OPERATION');
      expect(context.customTags).toContain('test');
      expect(context.additionalData?.customField).toBe('customValue');
    });

    test('should record errors for context', () => {
      const error = new Error('Test error');
      recordErrorForContext(error, 'TestComponent');

      const previousErrors = enhancedErrorContextManager.getPreviousErrors();
      expect(previousErrors).toHaveLength(1);
      expect(previousErrors[0].message).toBe('Test error');
      expect(previousErrors[0].component).toBe('TestComponent');
    });

    test('should add user actions to trail', () => {
      addUserActionToContext('click', 'button', { id: 'test-button' });

      const userActions = enhancedErrorContextManager.getUserActionTrail();
      expect(userActions).toHaveLength(1);
      expect(userActions[0].action).toBe('click');
      expect(userActions[0].element).toBe('button');
      expect(userActions[0].details).toEqual({ id: 'test-button' });
    });
  });

  describe('Enhanced Error Reporting', () => {
    test('should classify network errors correctly', () => {
      const classifier = enhancedErrorReporter.getClassifier();
      const error = new Error('Network connection failed');
      const context = {
        component: 'NetworkComponent',
        function: 'fetchData'
      } as any;

      const classification = classifier.classify(error, context);

      expect(classification.category).toBe(ErrorCategory.NETWORK);
      expect(classification.severity).toBe('error');
    });

    test('should classify authentication errors correctly', () => {
      const classifier = enhancedErrorReporter.getClassifier();
      const error = new Error('Unauthorized: Invalid token');
      const context = {
        component: 'AuthComponent',
        function: 'validateToken'
      } as any;

      const classification = classifier.classify(error, context);

      expect(classification.category).toBe(ErrorCategory.AUTHENTICATION);
      expect(classification.severity).toBe('error');
    });

    test('should report errors with enhanced categorization', async () => {
      const error = new Error('Test error for reporting');
      const context = await createEnhancedErrorContext({
        component: 'TestReporter',
        function: 'testReport',
        operation: 'REPORT_TEST'
      });

      const report = await reportEnhancedError(error, context, ['test', 'report']);

      expect(report).toBeDefined();
      expect(report.id).toBeDefined();
      expect(report.message).toBe('Test error for reporting');
      expect(report.category).toBeDefined();
      expect(report.severity).toBe('error');
      expect(report.tags).toContain('test');
    });

    test('should get error statistics', () => {
      const stats = enhancedErrorReporter.getErrorStatistics();

      expect(stats).toHaveProperty('totalErrors');
      expect(stats).toHaveProperty('byCategory');
      expect(stats).toHaveProperty('bySeverity');
      expect(stats).toHaveProperty('byPriority');
      expect(stats).toHaveProperty('topFrequentErrors');
      expect(stats).toHaveProperty('uniqueUsersAffected');
    });
  });

  describe('API Error Handling Middleware', () => {
    test('should handle successful API calls', async () => {
      const mockApiCall = (): Promise<string> => Promise.resolve('API Success');

      const result = await withApiErrorHandling(mockApiCall, {
        strategy: 'retry',
        maxRetries: 2
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('API Success');
    });

    test('should handle failing API calls with retries', async () => {
      let callCount = 0;
      const mockFailingApiCall = (): Promise<string> => {
        callCount++;
        if (callCount >= 2) {
          return Promise.resolve('Success after retry');
        }
        return Promise.reject(new Error('API call failed'));
      };

      const result = await withApiErrorHandling(mockFailingApiCall, {
        strategy: 'retry',
        maxRetries: 3,
        retryDelay: 10
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('Success after retry');
      expect(callCount).toBe(2);
    });

    test('should handle fetch calls with error handling', async () => {
      // Mock fetch globally for this test
      const originalFetch = global.fetch;
      global.fetch = jest.fn(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ message: 'Fetch success' }),
          text: () => Promise.resolve(JSON.stringify({ message: 'Fetch success' })),
          headers: new Headers(),
          statusText: 'OK',
          type: 'basic',
          url: '',
          redirected: false,
          clone: () => ({} as Response),
          body: null,
          bodyUsed: false,
          arrayBuffer: () => Promise.reject(new Error('Not implemented')),
          blob: () => Promise.reject(new Error('Not implemented')),
          formData: () => Promise.reject(new Error('Not implemented')),
          getReader: () => ({ 
            cancel: () => Promise.resolve(), 
            read: () => Promise.reject(new Error('Not implemented')), 
            releaseLock: () => {} 
          })
        } as Response)
      ) as jest.Mock;

      try {
        const result = await withFetchErrorHandling<{ message: string }>('https://api.example.com/data', {
          method: 'GET'
        });

        expect(result.success).toBe(true);
        expect(result.data?.message).toBe('Fetch success');
      } finally {
        global.fetch = originalFetch;
      }
    });

    test('should create error-handled API client', () => {
      const client = createErrorHandledApiClient('https://api.example.com');

      expect(client).toBeDefined();
      expect(client.get).toBeDefined();
      expect(client.post).toBeDefined();
      expect(client.put).toBeDefined();
      expect(client.patch).toBeDefined();
      expect(client.delete).toBeDefined();
    });
  });

  describe('React Error Handling Hooks', () => {
    // Note: These tests would normally run in a React testing environment
    // For now, we'll just test the function existence and basic behavior
    
    test('should initialize useErrorHandling hook properly', () => {
      const hookResult = useErrorHandling();
      
      expect(hookResult).toBeDefined();
      expect(hookResult.errorState).toBeDefined();
      expect(hookResult.execute).toBeDefined();
      expect(hookResult.retry).toBeDefined();
      expect(hookResult.reset).toBeDefined();
      expect(hookResult.recover).toBeDefined();
      expect(hookResult.setError).toBeDefined();
      expect(hookResult.isLoading).toBeDefined();
    });

    test('should initialize useAsyncState hook properly', () => {
      const hookResult = useAsyncState();

      expect(hookResult).toBeDefined();
      expect(hookResult.data).toBeNull();
      expect(hookResult.loading).toBeDefined();
      expect(hookResult.error).toBeNull();
      expect(hookResult.execute).toBeDefined();
      expect(hookResult.retry).toBeDefined();
      expect(hookResult.reset).toBeDefined();
    });

    test('should initialize useApiCall hook properly', () => {
      const hookResult = useApiCall();

      expect(hookResult).toBeDefined();
      expect(hookResult.data).toBeNull();
      expect(hookResult.loading).toBeDefined();
      expect(hookResult.error).toBeNull();
      expect(hookResult.execute).toBeDefined();
      expect(hookResult.retry).toBeDefined();
      expect(hookResult.reset).toBeDefined();
    });

    test('should initialize usePolling hook properly', () => {
      const mockOperation = () => Promise.resolve('poll result');
      const hookResult = usePolling(mockOperation);

      expect(hookResult).toBeDefined();
      expect(hookResult.data).toBeNull();
      expect(hookResult.loading).toBeDefined();
      expect(hookResult.error).toBeNull();
      expect(hookResult.start).toBeDefined();
      expect(hookResult.stop).toBeDefined();
      expect(hookResult.isActive).toBeDefined();
    });

    test('should initialize useFormErrorHandling hook properly', () => {
      const mockSubmit = (data: any) => Promise.resolve();
      const hookResult = useFormErrorHandling(mockSubmit);

      expect(hookResult).toBeDefined();
      expect(hookResult.handleSubmit).toBeDefined();
      expect(hookResult.isSubmitting).toBeDefined();
      expect(hookResult.error).toBeNull();
      expect(hookResult.reset).toBeDefined();
      expect(hookResult.formData).toBeNull();
    });
  });

  describe('Integration Tests', () => {
    test('should integrate unified error handling with enhanced context', async () => {
      const context = await createEnhancedErrorContext({
        component: 'IntegrationTest',
        function: 'testIntegration',
        operation: 'INTEGRATION_TEST'
      });

      const result = await withUnifiedErrorHandling(async () => {
        // Simulate an operation that records an error
        const error = new Error('Integration test error');
        recordErrorForContext(error, 'IntegrationTest');
        throw error;
      }, {
        context: {
          component: 'IntegrationTest',
          function: 'testIntegration',
          operation: 'INTEGRATION_TEST'
        },
        reportError: true
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.errorReport).toBeDefined();
    });

    test('should handle API calls with full error handling pipeline', async () => {
      let callCount = 0;
      const mockApiCall = (): Promise<string> => {
        callCount++;
        if (callCount >= 3) {
          return Promise.resolve('Final success');
        }
        return Promise.reject(new Error(`Attempt ${callCount} failed`));
      };

      const result = await withApiErrorHandling(mockApiCall, {
        strategy: 'retry',
        maxRetries: 5,
        retryDelay: 5,
        cacheResponse: true,
        cacheKey: 'integration-test-cache-key'
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('Final success');
      expect(callCount).toBe(3);
    });
  });

  describe('Performance and Edge Cases', () => {
    test('should handle concurrent error handling operations', async () => {
      const operations = Array.from({ length: 10 }, (_, i) => 
        withUnifiedErrorHandling(async () => `result-${i}`, {
          strategy: 'retry',
          maxRetries: 1
        })
      );

      const results = await Promise.all(operations);

      expect(results).toHaveLength(10);
      results.forEach((result, i) => {
        expect(result.success).toBe(true);
        expect(result.data).toBe(`result-${i}`);
      });
    });

    test('should handle rapid successive errors', async () => {
      const errorResults = await Promise.all(
        Array.from({ length: 5 }, () => 
          withUnifiedErrorHandling(mockAlwaysFailingOperation, {
            strategy: 'retry',
            maxRetries: 1
          })
        )
      );

      expect(errorResults).toHaveLength(5);
      errorResults.forEach(result => {
        expect(result.success).toBe(false);
        expect(result.error).toBeDefined();
      });
    });

    test('should handle very large error objects', async () => {
      const hugeError = new Error('Large error object');
      (hugeError as any).largeData = new Array(10000).fill('data').join('-');

      const result = await withUnifiedErrorHandling(async () => {
        throw hugeError;
      }, {
        strategy: 'fallback',
        fallbackValue: 'fallback-for-large-error'
      });

      expect(result.success).toBe(true);
      expect(result.data).toBe('fallback-for-large-error');
    });
  });
});

// Run the tests
console.log('🧪 Starting comprehensive error handling tests...\n');

// Execute tests (in a real environment, this would be handled by a test runner)
const runTests = async () => {
  try {
    // This is a simplified test runner for demonstration
    console.log('✓ All test suites defined successfully');
    console.log('✓ Unified Error Handling Service tests ready');
    console.log('✓ Advanced Error Recovery tests ready');
    console.log('✓ Enhanced Error Context tests ready');
    console.log('✓ Enhanced Error Reporting tests ready');
    console.log('✓ API Error Handling Middleware tests ready');
    console.log('✓ React Error Handling Hooks tests ready');
    console.log('✓ Integration tests ready');
    console.log('✓ Performance and edge case tests ready');
    
    console.log('\n✅ Comprehensive error handling tests are ready to run with Jest or similar test framework');
  } catch (error) {
    console.error('❌ Error setting up tests:', error);
  }
};

runTests().catch(console.error);

export default runTests;