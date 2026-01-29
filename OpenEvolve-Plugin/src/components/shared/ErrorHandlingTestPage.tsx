/**
 * Error Handling Test Suite
 * Tests the error handling capabilities of the OpenEvolve plugin
 */

import React, { useState } from 'react';
import { EnhancedErrorBoundary, AsyncOperationErrorBoundary, NetworkErrorBoundary } from '@/components/shared/EnhancedErrorBoundary';
import { safeAsyncOperation, safeSyncOperation, safeGet, safeFetch } from '@/utils/safeOperations';
import { HandleError, HandleNetworkOperation } from '@/utils/errorHandlingDecorators';
import { apiClient } from '@/utils/ApiErrorHandlingMiddleware';
import { toast } from 'react-toastify';

// Test component that will throw an error
const ErrorTestComponent: React.FC = () => {
  const [shouldThrow, setShouldThrow] = useState(false);
  
  if (shouldThrow) {
    throw new Error('Test error from ErrorTestComponent');
  }
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">Error Test Component</h3>
      <p>This component works normally until you click the button below.</p>
      <button 
        onClick={() => setShouldThrow(true)}
        className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
      >
        Trigger Error
      </button>
    </div>
  );
};

// Test component for async operations
const AsyncTestComponent: React.FC = () => {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  
  const failingAsyncOperation = async (): Promise<string> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    throw new Error('Simulated async operation failure');
  };
  
  const successfulAsyncOperation = async (): Promise<string> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return 'Success!';
  };
  
  const handleFailingOperation = async () => {
    setLoading(true);
    try {
      const result = await safeAsyncOperation(failingAsyncOperation, {
        fallbackValue: 'Fallback value after error',
        retries: 2,
        retryDelay: 500,
        errorContext: 'AsyncTestComponent.failingOperation'
      });
      
      if (result.success) {
        setResult(result.data);
      } else {
        setResult(`Operation failed: ${result.error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const handleSuccessfulOperation = async () => {
    setLoading(true);
    try {
      const result = await safeAsyncOperation(successfulAsyncOperation, {
        errorContext: 'AsyncTestComponent.successfulOperation'
      });
      
      if (result.success) {
        setResult(result.data);
      } else {
        setResult(`Operation failed: ${result.error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">Async Operation Test</h3>
      <div className="flex space-x-3 mb-3">
        <button 
          onClick={handleFailingOperation}
          disabled={loading}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Failing Async Op'}
        </button>
        <button 
          onClick={handleSuccessfulOperation}
          disabled={loading}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Successful Async Op'}
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 bg-gray-100 rounded">
          <p className="font-medium">Result:</p>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

// Test component for sync operations
const SyncTestComponent: React.FC = () => {
  const [result, setResult] = useState<string | null>(null);
  
  const failingSyncOperation = (): string => {
    throw new Error('Simulated sync operation failure');
  };
  
  const successfulSyncOperation = (): string => {
    return 'Success!';
  };
  
  const handleFailingOperation = () => {
    const result = safeSyncOperation(failingSyncOperation, {
      fallbackValue: 'Fallback after sync error',
      errorContext: 'SyncTestComponent.failingOperation'
    });
    
    if (result.success) {
      setResult(result.data);
    } else {
      setResult(`Operation failed: ${result.error.message}`);
    }
  };
  
  const handleSuccessfulOperation = () => {
    const result = safeSyncOperation(successfulSyncOperation, {
      errorContext: 'SyncTestComponent.successfulOperation'
    });
    
    if (result.success) {
      setResult(result.data);
    } else {
      setResult(`Operation failed: ${result.error.message}`);
    }
  };
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">Sync Operation Test</h3>
      <div className="flex space-x-3 mb-3">
        <button 
          onClick={handleFailingOperation}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Failing Sync Op
        </button>
        <button 
          onClick={handleSuccessfulOperation}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          Successful Sync Op
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 bg-gray-100 rounded">
          <p className="font-medium">Result:</p>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

// Test component for safe property access
const SafeAccessTestComponent: React.FC = () => {
  const [result, setResult] = useState<string | null>(null);
  
  const testData = {
    user: {
      profile: {
        name: 'John Doe',
        settings: {
          theme: 'dark',
        }
      }
    }
  };
  
  const handleSafeAccess = () => {
    // This should work
    const name = safeGet(testData, 'user.profile.name', 'Unknown User');
    
    // This should return fallback because path doesn't exist
    const missingValue = safeGet(testData, 'user.profile.avatar', 'default-avatar.png');
    
    // This should return fallback because path is deeply nested and doesn't exist
    const deepMissing = safeGet(testData, 'user.profile.settings.language', 'en');
    
    setResult(`Name: ${name}, Missing: ${missingValue}, Deep Missing: ${deepMissing}`);
  };
  
  const handleUnsafeAccess = () => {
    try {
      // This would cause an error if we accessed it directly
      // @ts-ignore - intentionally unsafe access for testing
      const value = testData.user.profile.missing.deeply.nested.property;
      setResult(`This should not appear: ${value}`);
    } catch (error) {
      setResult(`Caught error from unsafe access: ${(error as Error).message}`);
    }
  };
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">Safe Property Access Test</h3>
      <div className="flex space-x-3 mb-3">
        <button 
          onClick={handleSafeAccess}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Safe Access
        </button>
        <button 
          onClick={handleUnsafeAccess}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Unsafe Access (try/catch)
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 bg-gray-100 rounded">
          <p className="font-medium">Result:</p>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

// Test component for API operations
const ApiTestComponent: React.FC = () => {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  
  const handleSuccessfulApiCall = async () => {
    setLoading(true);
    try {
      // Using the apiClient with built-in error handling
      const response = await apiClient.get('https://jsonplaceholder.typicode.com/posts/1');
      
      if (response.success) {
        setResult(`API Success: ${response.data?.title || 'No title'}`);
      } else {
        setResult(`API Failed: ${response.error?.message || 'Unknown error'}`);
      }
    } catch (error) {
      setResult(`Unexpected error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleFailingApiCall = async () => {
    setLoading(true);
    try {
      // This should fail because the URL doesn't exist
      const response = await apiClient.get('https://invalid-url-for-testing.com/api/data');
      
      if (response.success) {
        setResult(`API Success: ${JSON.stringify(response.data)}`);
      } else {
        setResult(`API Failed as expected: ${response.error?.message || 'Unknown error'}`);
      }
    } catch (error) {
      setResult(`Unexpected error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">API Operation Test</h3>
      <div className="flex space-x-3 mb-3">
        <button 
          onClick={handleSuccessfulApiCall}
          disabled={loading}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Successful API Call'}
        </button>
        <button 
          onClick={handleFailingApiCall}
          disabled={loading}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Failing API Call'}
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 bg-gray-100 rounded">
          <p className="font-medium">Result:</p>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

// Test class with decorated methods
class TestService {
  @HandleError({ retries: 2, errorContext: 'TestService.failingMethod' })
  async failingMethod(): Promise<string> {
    throw new Error('Decorated method error');
  }
  
  @HandleNetworkOperation({ retries: 3, errorContext: 'TestService.networkMethod' })
  async networkMethod(): Promise<string> {
    throw new Error('Network operation error');
  }
  
  @HandleError({ fallbackValue: 'Handled gracefully', errorContext: 'TestService.handledMethod' })
  async handledMethod(): Promise<string> {
    throw new Error('This error is handled gracefully');
  }
}

const DecoratorTestComponent: React.FC = () => {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const testService = new TestService();
  
  const handleFailingMethod = async () => {
    setLoading(true);
    try {
      const result = await testService.failingMethod();
      setResult(`Result: ${result}`);
    } catch (error) {
      setResult(`Caught error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleNetworkMethod = async () => {
    setLoading(true);
    try {
      const result = await testService.networkMethod();
      setResult(`Result: ${result}`);
    } catch (error) {
      setResult(`Caught error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleHandledMethod = async () => {
    setLoading(true);
    try {
      const result = await testService.handledMethod();
      setResult(`Result: ${result}`);
    } catch (error) {
      setResult(`Caught error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">Decorator Test</h3>
      <div className="flex flex-wrap gap-3 mb-3">
        <button 
          onClick={handleFailingMethod}
          disabled={loading}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Failing Method'}
        </button>
        <button 
          onClick={handleNetworkMethod}
          disabled={loading}
          className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Network Method'}
        </button>
        <button 
          onClick={handleHandledMethod}
          disabled={loading}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Handled Method'}
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 bg-gray-100 rounded">
          <p className="font-medium">Result:</p>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

// Main test page component
const ErrorHandlingTestPage: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Error Handling Test Suite</h1>
      
      <p className="text-gray-600 mb-6">
        This page tests various error handling mechanisms in the OpenEvolve plugin.
        Each section demonstrates different aspects of the error handling system.
      </p>
      
      {/* Error Boundary Tests */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-3">Error Boundary Tests</h2>
          
          <EnhancedErrorBoundary
            customErrorTitle="Component Error Detected"
            customErrorMessage="A component in this section encountered an error"
            showDetailedError={true}
          >
            <ErrorTestComponent />
          </EnhancedErrorBoundary>
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-3">Async Operation Tests</h2>
          
          <AsyncOperationErrorBoundary>
            <AsyncTestComponent />
          </AsyncOperationErrorBoundary>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-3">Network Error Tests</h2>
          
          <NetworkErrorBoundary>
            <ApiTestComponent />
          </NetworkErrorBoundary>
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-3">Decorator Tests</h2>
          
          <EnhancedErrorBoundary>
            <DecoratorTestComponent />
          </EnhancedErrorBoundary>
        </div>
      </div>
      
      {/* Safe Operation Tests */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-3">Sync Operation Tests</h2>
          <SyncTestComponent />
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-3">Safe Access Tests</h2>
          <SafeAccessTestComponent />
        </div>
      </div>
      
      {/* Instructions */}
      <div className="mt-8 p-6 bg-blue-50 rounded-lg border border-blue-200">
        <h2 className="text-xl font-semibold mb-3 text-blue-800">Testing Instructions</h2>
        <ul className="list-disc pl-5 space-y-2 text-blue-700">
          <li>Click "Trigger Error" in the Error Test Component to see the error boundary in action</li>
          <li>Try the async operations to see retry logic and fallback values</li>
          <li>Test the API operations to see network error handling</li>
          <li>Use the decorator test buttons to see how method decorators handle errors</li>
          <li>Compare safe vs unsafe property access to see the difference</li>
        </ul>
      </div>
    </div>
  );
};

export default ErrorHandlingTestPage;