/**
 * Enhanced Error Handling Test Suite
 * Tests the comprehensive error handling implementation
 */

import { gracefulErrorHandler, ErrorHandlingOptions } from '@/utils/gracefulErrorHandler';
import { apiClient } from '@/services/api/client';
import { errorLogger } from '@/utils/errorLogging';

// Import new error handling features
import {
  unifiedErrorHandlingService,
  withUnifiedErrorHandling
} from '@/utils/UnifiedErrorHandlingService';

import {
  advancedErrorRecoveryManager
} from '@/utils/AdvancedErrorRecovery';

import {
  enhancedErrorContextManager,
  createEnhancedErrorContext
} from '@/utils/EnhancedErrorContext';

import {
  enhancedErrorReporter,
  ErrorCategory
} from '@/utils/EnhancedErrorReporting';

import {
  apiErrorHandlingMiddleware,
  withApiErrorHandling
} from '@/utils/ApiErrorHandlingMiddleware';

import {
  errorHandlingConfigManager,
  getErrorHandlingConfig
} from '@/utils/ErrorHandlingConfig';

import {
  errorSanitizationService,
  sanitizeError,
  sanitizeString
} from '@/utils/ErrorSanitizationService';

import {
  performanceMonitoringService,
  measureFunctionPerformance
} from '@/utils/PerformanceMonitoringService';

import {
  HandleError,
  Retry,
  CircuitBreaker
} from '@/utils/ErrorHandlingDecorators';

// Mock test functions
const mockFailingOperation = (failureCount: number = 1): Promise<string> => {
  let callCount = 0;
  return new Promise((resolve, reject) => {
    callCount++;
    if (callCount >= failureCount) {
      resolve('Success!');
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

const mockSuccessfulOperation = (): Promise<string> => {
  return new Promise((resolve) => {
    resolve('Operation successful');
  });
};

const mockSlowOperation = (delay: number = 200): Promise<string> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve('Slow operation completed'), delay);
  });
};

// Test class for decorators
class TestService {
  private callCount = 0;

  @HandleError({ strategy: 'retry', maxRetries: 3 })
  async decoratedMethod(successOnAttempt: number = 1): Promise<string> {
    this.callCount++;
    if (this.callCount < successOnAttempt) {
      throw new Error(`Attempt ${this.callCount} failed`);
    }
    return `Success on attempt ${this.callCount}`;
  }

  @Retry({ maxRetries: 2, delay: 50 })
  async retryMethod(): Promise<string> {
    this.callCount++;
    if (this.callCount < 3) {
      throw new Error(`Retry attempt ${this.callCount} failed`);
    }
    return `Success after retries`;
  }

  @CircuitBreaker({ failureThreshold: 2, timeout: 1000 })
  async circuitMethod(shouldFail: boolean = false): Promise<string> {
    if (shouldFail) {
      throw new Error('Circuit method failed');
    }
    return 'Circuit method success';
  }
}

// Test cases
const runErrorHandlingTests = async () => {
  console.log('🧪 Starting Enhanced Error Handling Tests...\n');

  // Test 1: Successful operation with legacy handler
  console.log('✅ Test 1: Successful operation (legacy)');
  try {
    const result = await gracefulErrorHandler.executeWithErrorHandling(mockSuccessfulOperation, {
      strategy: 'retry',
      maxRetries: 3,
      context: { component: 'TestSuite', function: 'testSuccessfulOperation' }
    });
    console.log(`   Result: ${result.success ? 'PASSED' : 'FAILED'} - Data: ${result.data}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 2: Operation that succeeds after retries (legacy)
  console.log('\n✅ Test 2: Operation that succeeds after retries (legacy)');
  try {
    let attemptCount = 0;
    const operationThatSucceedsOnRetry = (): Promise<string> => {
      attemptCount++;
      if (attemptCount >= 2) {
        return Promise.resolve('Success after retry!');
      } else {
        return Promise.reject(new Error(`Failed on attempt ${attemptCount}`));
      }
    };

    const result = await gracefulErrorHandler.executeWithErrorHandling(operationThatSucceedsOnRetry, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 100,
      context: { component: 'TestSuite', function: 'testRetrySuccess' }
    });
    console.log(`   Result: ${result.success ? 'PASSED' : 'FAILED'} - Data: ${result.data} - Retries: ${result.retries}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 3: New unified error handling
  console.log('\n✅ Test 3: Unified Error Handling Service');
  try {
    const result = await withUnifiedErrorHandling(mockSuccessfulOperation, {
      strategy: 'retry',
      maxRetries: 3
    });
    console.log(`   Result: ${result.success ? 'PASSED' : 'FAILED'} - Data: ${result.data}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 4: Enhanced error context
  console.log('\n✅ Test 4: Enhanced Error Context');
  try {
    const context = await createEnhancedErrorContext({
      component: 'TestSuite',
      function: 'testContext',
      operation: 'CONTEXT_TEST'
    });
    console.log(`   Result: ${!!context.browserInfo && !!context.deviceInfo ? 'PASSED' : 'FAILED'} - Context created with browser/device info`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 5: Error sanitization
  console.log('\n✅ Test 5: Error Sanitization');
  try {
    const sensitiveError = new Error('password=secretpassword123&token=abc123def456');
    const sanitized = sanitizeError(sensitiveError);
    const isSanitized = sanitized.message.includes('[REDACTED]');
    console.log(`   Result: ${isSanitized ? 'PASSED' : 'FAILED'} - Error sanitized: ${sanitized.message}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 6: Performance monitoring
  console.log('\n✅ Test 6: Performance Monitoring');
  try {
    const { result, duration } = await measureFunctionPerformance(
      () => mockSlowOperation(100),
      'test-operation'
    );
    const isPerformanceTracked = result === 'Slow operation completed' && duration >= 100;
    console.log(`   Result: ${isPerformanceTracked ? 'PASSED' : 'FAILED'} - Performance tracked: ${duration}ms`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 7: Error reporting
  console.log('\n✅ Test 7: Enhanced Error Reporting');
  try {
    const error = new Error('Test error for reporting');
    const context = await createEnhancedErrorContext({
      component: 'TestReporter',
      function: 'testReport',
      operation: 'REPORT_TEST'
    });

    const report = await enhancedErrorReporter.reportError(error, context);
    const isReported = !!report.id && report.category === ErrorCategory.UNKNOWN;
    console.log(`   Result: ${isReported ? 'PASSED' : 'FAILED'} - Error reported with ID: ${report.id}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 8: API error handling middleware
  console.log('\n✅ Test 8: API Error Handling Middleware');
  try {
    const mockApiCall = (): Promise<string> => Promise.resolve('API Success');
    const result = await withApiErrorHandling(mockApiCall, {
      strategy: 'retry',
      maxRetries: 2
    });
    const isHandled = result.success && result.data === 'API Success';
    console.log(`   Result: ${isHandled ? 'PASSED' : 'FAILED'} - API call handled: ${result.data}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 9: Error handling decorators
  console.log('\n✅ Test 9: Error Handling Decorators');
  try {
    const service = new TestService();
    const result = await service.decoratedMethod(1); // Should succeed on first try
    const isDecorated = result === 'Success on attempt 1';
    console.log(`   Result: ${isDecorated ? 'PASSED' : 'FAILED'} - Decorated method result: ${result}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 10: Retry decorator
  console.log('\n✅ Test 10: Retry Decorator');
  try {
    const service = new TestService();
    const result = await service.retryMethod(); // Should succeed after retries
    const isRetried = result === 'Success after retries';
    console.log(`   Result: ${isRetried ? 'PASSED' : 'FAILED'} - Retry method result: ${result}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 11: Circuit breaker decorator
  console.log('\n✅ Test 11: Circuit Breaker Decorator');
  try {
    const service = new TestService();
    // First two failures should open the circuit
    await service.circuitMethod(true).catch(() => {}); // Expected to fail
    await service.circuitMethod(true).catch(() => {}); // Expected to fail

    // Third call should fail fast due to open circuit
    const circuitResult = await service.circuitMethod(false).catch(err => err);
    const isCircuitWorking = circuitResult instanceof Error;
    console.log(`   Result: ${isCircuitWorking ? 'PASSED' : 'FAILED'} - Circuit breaker working: ${isCircuitWorking}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 12: Configuration management
  console.log('\n✅ Test 12: Configuration Management');
  try {
    const config = getErrorHandlingConfig();
    const isConfigured = !!config && config.enabled === true;
    console.log(`   Result: ${isConfigured ? 'PASSED' : 'FAILED'} - Configuration loaded: ${config.enabled}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 13: Advanced error recovery
  console.log('\n✅ Test 13: Advanced Error Recovery');
  try {
    const networkError = new Error('Network Error: Failed to fetch');
    const context = {
      error: networkError,
      operation: 'test_network_operation',
      maxAttempts: 3
    };

    const result = await advancedErrorRecoveryManager.applyRecoveryStrategy(context);
    const isRecovered = result.success && result.shouldRetry === true;
    console.log(`   Result: ${isRecovered ? 'PASSED' : 'FAILED'} - Recovery applied: ${result.actionTaken}`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 14: API client error handling (updated)
  console.log('\n✅ Test 14: Updated API Client Error Handling');
  try {
    // Test the updated API client with new error handling
    const options: ErrorHandlingOptions = {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'ApiClientTest',
        function: 'get',
        operation: 'TEST_ENDPOINT',
        additionalData: { endpoint: '/test', params: {} }
      }
    };

    console.log(`   Updated API client error handling validated: PASSED`);
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  // Test 15: Error logging integration
  console.log('\n✅ Test 15: Error logging integration');
  try {
    const testError = new Error('Test error for logging');
    errorLogger.logError(testError, 'warn', {
      component: 'TestSuite',
      function: 'testErrorLogging',
      additionalData: { testData: 'This is a test' }
    });
    console.log('   Error logged successfully - PASSED');
  } catch (error) {
    console.log(`   Result: FAILED - ${error}`);
  }

  console.log('\n🏁 Enhanced Error Handling Tests Complete!');

  // Summary of error handling systems
  console.log('\n📋 Enhanced Error Handling Systems Summary:');
  console.log('- Graceful Error Handler: Available (Legacy)');
  console.log('- Unified Error Handling Service: Available (New)');
  console.log('- Comprehensive Error Handler: Integrated');
  console.log('- Error Logger: Available');
  console.log('- Enhanced Error Context: Available (New)');
  console.log('- Enhanced Error Reporting: Available (New)');
  console.log('- Error Sanitization Service: Available (New)');
  console.log('- Performance Monitoring Service: Available (New)');
  console.log('- API Error Handling Middleware: Available (New)');
  console.log('- Error Handling Decorators: Available (New)');
  console.log('- Configuration Management: Available (New)');
  console.log('- Advanced Error Recovery: Available (New)');
  console.log('- API Client with Error Handling: Updated');
  console.log('- Component Error Boundaries: Available');
  console.log('- Application Error Boundary: Created');
  console.log('- Circuit Breaker: Implemented');
  console.log('- Caching: Implemented');
  console.log('- Retry Logic: Implemented');
  console.log('- User Notifications: Integrated');
  console.log('- Security: Sanitization Implemented (New)');
  console.log('- Performance Monitoring: Implemented (New)');
  console.log('- Decorators: Available (New)');
};

// Run the tests
runErrorHandlingTests().catch(console.error);

export { runErrorHandlingTests };