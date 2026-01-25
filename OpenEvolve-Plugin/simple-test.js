// Simple test to validate the error handling implementation
import { gracefulErrorHandler } from './src/utils/gracefulErrorHandler';

console.log('Testing graceful error handler import...');
console.log('Graceful error handler available:', !!gracefulErrorHandler);

// Test basic functionality
async function testBasicFunctionality() {
  try {
    const result = await gracefulErrorHandler.executeWithErrorHandling(
      async () => {
        return 'Success!';
      },
      {
        strategy: 'retry',
        maxRetries: 1,
        context: { component: 'Test', function: 'basicTest' }
      }
    );
    
    console.log('Basic test result:', result);
    console.log('✅ Basic functionality test passed');
  } catch (error) {
    console.log('❌ Basic functionality test failed:', error);
  }
}

testBasicFunctionality();