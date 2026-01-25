/**
 * Test Script for Error Handling Implementation
 */

import { runErrorHandlingTests } from './utils/testErrorHandling';

// Run the tests
console.log('Starting error handling tests...\n');
runErrorHandlingTests()
  .then(() => {
    console.log('\nAll tests completed!');
  })
  .catch((error) => {
    console.error('Test suite failed:', error);
  });