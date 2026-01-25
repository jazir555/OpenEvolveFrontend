/**
 * React Error Handling Hooks
 * Provides React hooks for managing errors in functional components
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  UnifiedErrorHandlingOptions, 
  UnifiedErrorResult, 
  withUnifiedErrorHandling 
} from './UnifiedErrorHandlingService';
import { 
  ExtendedErrorContext, 
  createEnhancedErrorContext,
  recordErrorForContext,
  addUserActionToContext
} from './EnhancedErrorContext';
import { 
  reportEnhancedError, 
  EnhancedErrorReport 
} from './EnhancedErrorReporting';
import { advancedErrorRecoveryManager } from './AdvancedErrorRecovery';
import { toast } from 'react-toastify';

// Define error state
export interface ErrorState {
  error: Error | null;
  hasError: boolean;
  errorReport?: EnhancedErrorReport;
  recoveryAttempts: number;
  isRecovering: boolean;
  retryCount: number;
}

// Define error handling hook options
export interface UseErrorHandlingOptions extends UnifiedErrorHandlingOptions {
  onError?: (error: Error, context: ExtendedErrorContext) => void;
  onSuccess?: () => void;
  onRecoveryAttempt?: (strategy: string) => void;
  autoReset?: boolean; // Whether to auto-reset error state after successful operation
  resetKeys?: any[]; // Keys that trigger error state reset
}

// Define error handling hook result
export interface UseErrorHandlingResult<T = any> {
  errorState: ErrorState;
  execute: (operation: () => Promise<T>, options?: Partial<UseErrorHandlingOptions>) => Promise<UnifiedErrorResult<T>>;
  retry: () => void;
  reset: () => void;
  recover: () => Promise<boolean>;
  setError: (error: Error) => void;
  isLoading: boolean;
}

/**
 * Custom hook for handling errors in React components
 */
export function useErrorHandling<T = any>(options: UseErrorHandlingOptions = {}): UseErrorHandlingResult<T> {
  const [errorState, setErrorState] = useState<ErrorState>({
    error: null,
    hasError: false,
    recoveryAttempts: 0,
    isRecovering: false,
    retryCount: 0
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const mountedRef = useRef(true);
  const operationRef = useRef<(() => Promise<T>) | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Reset error state when resetKeys change
  useEffect(() => {
    if (options.autoReset && options.resetKeys) {
      reset();
    }
  }, [options.resetKeys]);

  /**
   * Execute an operation with error handling
   */
  const execute = useCallback(async (
    operation: () => Promise<T>,
    overrideOptions: Partial<UseErrorHandlingOptions> = {}
  ): Promise<UnifiedErrorResult<T>> => {
    const mergedOptions = { ...options, ...overrideOptions };
    
    setIsLoading(true);
    operationRef.current = operation;

    try {
      // Create enhanced context
      const context = await createEnhancedErrorContext(
        {
          component: mergedOptions.context?.component || 'ReactHook',
          function: mergedOptions.function || 'useErrorHandling',
          operation: mergedOptions.context?.operation || 'HOOK_OPERATION',
          additionalData: {
            ...mergedOptions.context?.additionalData,
            hookId: Math.random().toString(36).substr(2, 9)
          }
        },
        { 
          tags: ['react', 'hook', 'ui'],
          ...mergedOptions.metadata
        }
      );

      // Execute with unified error handling
      const result = await withUnifiedErrorHandling<T>(operation, mergedOptions);

      if (result.success) {
        // Success case
        if (mountedRef.current) {
          setErrorState(prev => ({
            ...prev,
            error: null,
            hasError: false,
            retryCount: 0
          }));
        }

        if (mergedOptions.onSuccess) {
          try {
            mergedOptions.onSuccess();
          } catch (callbackError) {
            console.error('Success callback failed:', callbackError);
          }
        }
      } else if (result.error) {
        // Error case
        const error = result.error;
        
        // Record error for context
        recordErrorForContext(error, context.component);

        // Create enhanced error report
        let errorReport: EnhancedErrorReport | undefined;
        if (mergedOptions.reportError !== false) {
          try {
            errorReport = await reportEnhancedError(error, context, [
              'react_error',
              'hook_error',
              ...(mergedOptions.context?.additionalData?.tags || [])
            ]);
          } catch (reportError) {
            console.error('Error reporting failed:', reportError);
          }
        }

        if (mountedRef.current) {
          setErrorState(prev => ({
            ...prev,
            error,
            hasError: true,
            errorReport,
            retryCount: prev.retryCount + 1
          }));
        }

        // Call error callback
        if (mergedOptions.onError) {
          try {
            mergedOptions.onError(error, context);
          } catch (callbackError) {
            console.error('Error callback failed:', callbackError);
          }
        }
      }

      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      
      // Record error for context
      recordErrorForContext(err, options.context?.component || 'ReactHook');

      // Create enhanced error report
      let errorReport: EnhancedErrorReport | undefined;
      if (options.reportError !== false) {
        const context = await createEnhancedErrorContext(
          {
            component: options.context?.component || 'ReactHook',
            function: options.function || 'useErrorHandling',
            operation: options.context?.operation || 'HOOK_OPERATION',
            additionalData: {
              ...options.context?.additionalData,
              hookId: Math.random().toString(36).substr(2, 9)
            }
          },
          { 
            tags: ['react', 'hook', 'ui'],
            ...options.metadata
          }
        );

        try {
          errorReport = await reportEnhancedError(err, context, [
            'react_error',
            'hook_error',
            'unexpected_error'
          ]);
        } catch (reportError) {
          console.error('Error reporting failed:', reportError);
        }
      }

      if (mountedRef.current) {
        setErrorState(prev => ({
          ...prev,
          error: err,
          hasError: true,
          errorReport,
          retryCount: prev.retryCount + 1
        }));
      }

      // Call error callback
      if (options.onError) {
        try {
          const context = await createEnhancedErrorContext(
            {
              component: options.context?.component || 'ReactHook',
              function: options.function || 'useErrorHandling',
              operation: options.context?.operation || 'HOOK_OPERATION',
              additionalData: {
                ...options.context?.additionalData,
                hookId: Math.random().toString(36).substr(2, 9)
              }
            },
            { 
              tags: ['react', 'hook', 'ui'],
              ...options.metadata
            }
          );
          
          options.onError(err, context);
        } catch (callbackError) {
          console.error('Error callback failed:', callbackError);
        }
      }

      return {
        success: false,
        error: err
      };
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [options]);

  /**
   * Retry the last operation
   */
  const retry = useCallback(() => {
    if (operationRef.current) {
      execute(operationRef.current);
    }
  }, [execute]);

  /**
   * Reset error state
   */
  const reset = useCallback(() => {
    setErrorState({
      error: null,
      hasError: false,
      recoveryAttempts: 0,
      isRecovering: false,
      retryCount: 0
    });
  }, []);

  /**
   * Attempt to recover from error
   */
  const recover = useCallback(async (): Promise<boolean> => {
    if (!errorState.error) {
      return true; // Nothing to recover from
    }

    setErrorState(prev => ({ ...prev, isRecovering: true }));

    try {
      const recoveryContext = {
        error: errorState.error,
        operation: options.context?.operation || 'HOOK_RECOVERY',
        params: options.context?.additionalData,
        previousAttempts: errorState.recoveryAttempts,
        maxAttempts: options.maxRetries || 3,
        fallbackData: options.fallbackValue,
        recoveryOptions: {
          failureThreshold: options.maxRetries || 3
        },
        metadata: options.metadata
      };

      const recoveryResult = await advancedErrorRecoveryManager.applyRecoveryStrategy(recoveryContext);

      if (recoveryResult.success) {
        if (options.onRecoveryAttempt) {
          try {
            options.onRecoveryAttempt(recoveryResult.actionTaken);
          } catch (callbackError) {
            console.error('Recovery callback failed:', callbackError);
          }
        }

        if (recoveryResult.shouldRetry && operationRef.current) {
          // Retry the operation
          await execute(operationRef.current);
        } else if (recoveryResult.newData !== undefined) {
          // Update state with recovered data
          setErrorState(prev => ({
            ...prev,
            error: null,
            hasError: false,
            recoveryAttempts: prev.recoveryAttempts + 1,
            isRecovering: false
          }));
          return true;
        }

        setErrorState(prev => ({
          ...prev,
          recoveryAttempts: prev.recoveryAttempts + 1,
          isRecovering: false
        }));

        return true;
      } else {
        setErrorState(prev => ({
          ...prev,
          recoveryAttempts: prev.recoveryAttempts + 1,
          isRecovering: false
        }));
        return false;
      }
    } catch (recoveryError) {
      console.error('Recovery failed:', recoveryError);
      
      setErrorState(prev => ({
        ...prev,
        isRecovering: false
      }));
      
      return false;
    }
  }, [errorState.error, errorState.recoveryAttempts, execute, options]);

  /**
   * Set error state manually
   */
  const setError = useCallback((error: Error) => {
    setErrorState(prev => ({
      ...prev,
      error,
      hasError: true,
      retryCount: prev.retryCount + 1
    }));
  }, []);

  return {
    errorState,
    execute,
    retry,
    reset,
    recover,
    setError,
    isLoading
  };
}

// Define async state hook result
export interface UseAsyncStateResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  execute: (operation: () => Promise<T>) => Promise<void>;
  retry: () => void;
  reset: () => void;
}

/**
 * Hook for managing async operations with loading and error states
 */
export function useAsyncState<T = any>(
  initialValue: T | null = null,
  options: UseErrorHandlingOptions = {}
): UseAsyncStateResult<T> {
  const [data, setData] = useState<T | null>(initialValue);
  const { errorState, execute, retry, reset, isLoading } = useErrorHandling<T>(options);
  
  const wrappedExecute = useCallback(async (operation: () => Promise<T>) => {
    const result = await execute(operation);
    if (result.success && result.data !== undefined) {
      setData(result.data);
    }
  }, [execute]);

  return {
    data,
    loading: isLoading,
    error: errorState.error,
    execute: wrappedExecute,
    retry,
    reset
  };
}

// Define API call hook result
export interface UseApiCallResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  execute: (url: string, init?: RequestInit) => Promise<Response | null>;
  retry: () => void;
  reset: () => void;
}

/**
 * Hook for making API calls with built-in error handling
 */
export function useApiCall<T = any>(
  options: UseErrorHandlingOptions & { 
    baseUrl?: string; 
    defaultHeaders?: Record<string, string> 
  } = {}
): UseApiCallResult<T> {
  const [data, setData] = useState<T | null>(null);
  const { errorState, execute, retry, reset, isLoading } = useErrorHandling<T>(options);
  
  const wrappedExecute = useCallback(async (url: string, init?: RequestInit): Promise<Response | null> => {
    // Combine default headers with request headers
    const headers = {
      ...(options.defaultHeaders || {}),
      ...(init?.headers || {})
    };

    const operation = async (): Promise<T> => {
      const fullUrl = options.baseUrl ? `${options.baseUrl}${url}` : url;
      const response = await fetch(fullUrl, {
        ...init,
        headers
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      let result: T;

      if (contentType && contentType.includes('application/json')) {
        result = await response.json();
      } else {
        result = await response.text() as unknown as T;
      }

      return result;
    };

    const result = await execute(operation);
    if (result.success && result.data !== undefined) {
      setData(result.data);
      return responseFromData(result.data); // This is a simplification
    }

    return null;
  }, [execute, options.baseUrl, options.defaultHeaders]);

  // Helper function to create a mock response from data (simplified)
  const responseFromData = (data: any): Response => {
    // Convert data to string once for reuse
    const dataString = typeof data === 'string' ? data : JSON.stringify(data);
    const dataBytes = new TextEncoder().encode(dataString);

    return {
      ok: true,
      status: 200,
      json: () => Promise.resolve(data),
      text: () => Promise.resolve(dataString),
      headers: new Headers(),
      statusText: 'OK',
      type: 'basic' as ResponseType,
      url: '',
      redirected: false,
      clone: () => responseFromData(data),
      body: null,
      bodyUsed: false,
      arrayBuffer: () => Promise.resolve(dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength)),
      blob: () => Promise.resolve(new Blob([dataBytes], { type: 'application/json' })),
      formData: () => {
        // Parse JSON and attempt to create FormData
        return new Promise<FormData>((resolve, reject) => {
          try {
            const formData = new FormData();
            if (typeof data === 'object' && data !== null) {
              Object.entries(data).forEach(([key, value]) => {
                if (value instanceof File) {
                  formData.append(key, value);
                } else if (value !== null && value !== undefined) {
                  formData.append(key, String(value));
                }
              });
            }
            resolve(formData);
          } catch (error) {
            reject(new Error('Failed to convert response to FormData'));
          }
        });
      }
    } as unknown as Response;
  };

  return {
    data,
    loading: isLoading,
    error: errorState.error,
    execute: wrappedExecute,
    retry,
    reset
  };
}

// Define polling hook options
export interface UsePollingOptions<T> {
  interval?: number;
  maxAttempts?: number;
  onError?: (error: Error) => void;
  onSuccess?: (data: T) => void;
  onMaxAttemptsReached?: () => void;
}

// Define polling hook result
export interface UsePollingResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  start: () => void;
  stop: () => void;
  isActive: boolean;
}

/**
 * Hook for polling with error handling
 */
export function usePolling<T = any>(
  operation: () => Promise<T>,
  options: UsePollingOptions<T> & UseErrorHandlingOptions = {}
): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isActive, setIsActive] = useState(false);
  const { errorState, execute } = useErrorHandling<T>(options);
  const intervalId = useRef<NodeJS.Timeout | null>(null);
  const attemptCount = useRef(0);
  
  const maxAttempts = options.maxAttempts || Infinity;
  const interval = options.interval || 5000; // Default 5 seconds

  const start = useCallback(() => {
    if (isActive) return; // Already running

    setIsActive(true);
    attemptCount.current = 0;
    
    const poll = async () => {
      if (!isActive) return;

      try {
        attemptCount.current++;
        const result = await execute(operation);
        
        if (result.success && result.data !== undefined) {
          setData(result.data);
          if (options.onSuccess) {
            options.onSuccess(result.data);
          }
          attemptCount.current = 0; // Reset on success
        }
      } catch (error) {
        console.error('Polling error:', error);
        if (options.onError) {
          options.onError(error instanceof Error ? error : new Error(String(error)));
        }
      }

      // Check if we've reached max attempts
      if (attemptCount.current >= maxAttempts) {
        stop();
        if (options.onMaxAttemptsReached) {
          options.onMaxAttemptsReached();
        }
        return;
      }

      // Schedule next poll
      if (isActive) {
        intervalId.current = setTimeout(poll, interval);
      }
    };

    // Start polling immediately
    poll();
  }, [isActive, execute, interval, maxAttempts, operation, options]);

  const stop = useCallback(() => {
    setIsActive(false);
    if (intervalId.current) {
      clearTimeout(intervalId.current);
      intervalId.current = null;
    }
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    data,
    loading: isActive,
    error: errorState.error,
    start,
    stop,
    isActive
  };
}

/**
 * Hook for handling form submission errors
 */
export function useFormErrorHandling<T = any>(
  onSubmit: (data: T) => Promise<any>,
  options: UseErrorHandlingOptions = {}
) {
  const [formData, setFormData] = useState<T | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { errorState, execute, reset } = useErrorHandling(options);

  const handleSubmit = useCallback(async (data: T) => {
    setFormData(data);
    setIsSubmitting(true);
    
    try {
      await execute(() => onSubmit(data));
    } finally {
      setIsSubmitting(false);
    }
  }, [execute, onSubmit]);

  return {
    handleSubmit,
    isSubmitting,
    error: errorState.error,
    reset,
    formData
  };
}