/**
 * Enhanced Error Boundary Component
 * Provides sophisticated error handling with multiple recovery strategies and user-friendly fallbacks
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { errorLogger } from '@/utils/errorLogging';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { toast } from 'react-toastify';

// Define error boundary props
interface EnhancedErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void; retry: () => void; errorInfo?: ErrorInfo }>;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onRecover?: (strategyUsed: string) => void;
  recoveryStrategies?: ('retry' | 'fallback' | 'cache' | 'graceful_degradation')[];
  enableLogging?: boolean;
  enableUserNotification?: boolean;
  enableAutoRecovery?: boolean;
  enableErrorReporting?: boolean;
  maxRecoveryAttempts?: number;
  retryDelay?: number;
  showDetailedError?: boolean;
  customErrorTitle?: string;
  customErrorMessage?: string;
  resetKeys?: any[]; // Keys that trigger a reset when changed
  errorContext?: string; // Context for error reporting
}

// Define error boundary state
interface EnhancedErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  recoveryAttempts: number;
  isRecovering: boolean;
  recoveryStrategyUsed?: string;
  errorId?: string;
}

/**
 * Enhanced Error Boundary Component
 * Provides sophisticated error handling with multiple recovery strategies
 */
export class EnhancedErrorBoundary extends Component<EnhancedErrorBoundaryProps, EnhancedErrorBoundaryState> {
  private timeoutId: NodeJS.Timeout | null = null;

  constructor(props: EnhancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): EnhancedErrorBoundaryState {
    return {
      hasError: true,
      error,
      errorInfo: null, // Will be set in componentDidCatch
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });

    // Generate error ID for tracking
    const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.setState({ errorId });

    // Log the error
    if (this.props.enableLogging !== false) {
      errorLogger.logError(error, 'error', {
        component: 'EnhancedErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          componentStack: errorInfo.componentStack,
          recoveryAttempts: this.state.recoveryAttempts,
          errorId,
          context: this.props.errorContext || 'unknown',
        }
      });
    }

    // Call custom error handler
    if (this.props.onError) {
      try {
        this.props.onError(error, errorInfo);
      } catch (customErrorHandlerError) {
        errorLogger.logError(customErrorHandlerError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Custom error handler failed' } });
      }
    }

    // Show user notification
    if (this.props.enableUserNotification !== false) {
      toast.error(`Interface error: ${error.message}`);
    }
  }

  componentDidUpdate(prevProps: EnhancedErrorBoundaryProps) {
    // Reset error boundary if reset keys change
    if (this.props.resetKeys &&
        JSON.stringify(prevProps.resetKeys) !== JSON.stringify(this.props.resetKeys)) {
      this.resetError();
    }
  }

  componentWillUnmount() {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
    }
  }

  handleReset = () => {
    this.resetError();
  };

  resetError = () => {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
      recoveryStrategyUsed: undefined,
      errorId: undefined,
    });
  };

  handleRetry = async () => {
    const maxRecoveryAttempts = this.props.maxRecoveryAttempts ?? 3;

    if (this.state.recoveryAttempts >= maxRecoveryAttempts) {
      this.resetError();
      return;
    }

    this.setState({ isRecovering: true });

    try {
      // Determine which recovery strategies to try
      const strategies = this.props.recoveryStrategies || ['retry', 'graceful_degradation'];
      
      for (const strategy of strategies) {
        if (strategy === 'retry') {
          // Simple retry by resetting the error boundary
          this.timeoutId = setTimeout(() => {
            this.resetError();
          }, this.props.retryDelay ?? 500);
          
          this.setState({
            recoveryAttempts: this.state.recoveryAttempts + 1,
            isRecovering: false,
            recoveryStrategyUsed: 'retry',
          });
          
          if (this.props.onRecover) {
            this.props.onRecover('retry');
          }
          return;
        } else if (strategy === 'graceful_degradation') {
          // For graceful degradation, we'll reset the error boundary
          // but potentially render a simplified version of the component
          this.setState({
            recoveryAttempts: this.state.recoveryAttempts + 1,
            isRecovering: false,
            recoveryStrategyUsed: 'graceful_degradation',
          });
          
          if (this.props.onRecover) {
            this.props.onRecover('graceful_degradation');
          }
          
          // Reset the error boundary after a short delay
          this.timeoutId = setTimeout(() => {
            this.resetError();
          }, 100);
          return;
        }
      }

      // If no strategies worked, just reset
      this.resetError();
    } catch (recoveryError) {
      errorLogger.logError(recoveryError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Recovery failed' } });
      this.setState({
        recoveryAttempts: this.state.recoveryAttempts + 1,
        isRecovering: false,
      });

      toast.error(`Recovery failed: ${recoveryError instanceof Error ? recoveryError.message : String(recoveryError)}`);
    }
  };

  render() {
    if (this.state.hasError && this.state.error) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback;
        return (
          <FallbackComponent
            error={this.state.error}
            resetError={this.handleReset}
            retry={this.handleRetry}
            errorInfo={this.state.errorInfo}
          />
        );
      }

      // Default fallback UI
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-gradient-to-br from-red-50 to-red-100 border border-red-200 rounded-xl shadow-sm">
          <div className="flex items-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-red-500 mr-3" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <h2 className="text-xl font-bold text-red-800">
              {this.props.customErrorTitle || 'Something went wrong'}
            </h2>
          </div>

          <p className="text-red-600 mb-4 text-center max-w-md">
            {this.props.customErrorMessage || this.state.error.message}
          </p>

          {this.state.errorId && (
            <p className="text-red-500 text-sm mb-2">
              Error ID: {this.state.errorId}
            </p>
          )}

          {this.state.recoveryStrategyUsed && (
            <p className="text-red-500 text-sm mb-4">
              Last recovery attempt: {this.state.recoveryStrategyUsed}
            </p>
          )}

          {this.props.showDetailedError && this.state.errorInfo && (
            <details className="w-full mb-4">
              <summary className="text-left text-red-700 cursor-pointer font-medium">Error Details</summary>
              <pre className="mt-2 p-4 bg-red-100 text-red-900 text-xs rounded-lg overflow-auto max-h-40 font-mono">
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}

          <div className="flex flex-wrap gap-3 justify-center">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering || this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3)}
              className="px-5 py-2.5 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg hover:from-red-700 hover:to-red-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg flex items-center"
            >
              {this.state.isRecovering ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Recovering...
                </>
              ) : 'Try Again'}
            </button>

            <button
              type="button"
              onClick={this.handleReset}
              className="px-5 py-2.5 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all shadow-md hover:shadow-lg"
            >
              Reset Component
            </button>
          </div>

          {this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3) && (
            <p className="mt-4 text-red-700 text-sm text-center max-w-md">
              Max recovery attempts reached. Please refresh the page or contact support if the problem persists.
            </p>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Safe Component Wrapper
 * A higher-order component that wraps any component with error handling
 */
export function withSafeComponent<P extends Record<string, any>>(
  Component: React.ComponentType<P>,
  componentName?: string,
  boundaryProps?: Omit<EnhancedErrorBoundaryProps, 'children'>
) {
  const displayName = componentName || Component.name || 'Component';
  
  const ComponentWithErrorBoundary = (props: P) => {
    return (
      <EnhancedErrorBoundary
        {...boundaryProps}
        errorContext={`${displayName} component`}
        customErrorTitle={`${displayName} Unavailable`}
        customErrorMessage={`The ${displayName} component encountered an error and couldn't be displayed.`}
      >
        <Component {...props} />
      </EnhancedErrorBoundary>
    );
  };

  ComponentWithErrorBoundary.displayName = `withSafeComponent(${displayName})`;
  
  return ComponentWithErrorBoundary;
}

/**
 * Async Operation Error Boundary
 * Specifically handles errors in async operations
 */
export class AsyncOperationErrorBoundary extends Component<EnhancedErrorBoundaryProps, EnhancedErrorBoundaryState> {
  constructor(props: EnhancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): EnhancedErrorBoundaryState {
    return {
      hasError: true,
      error,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });

    if (this.props.enableLogging !== false) {
      errorLogger.logError(error, 'error', {
        component: 'AsyncOperationErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          componentStack: errorInfo.componentStack,
          recoveryAttempts: this.state.recoveryAttempts,
          context: this.props.errorContext || 'async_operation',
        }
      });
    }

    if (this.props.onError) {
      try {
        this.props.onError(error, errorInfo);
      } catch (customErrorHandlerError) {
        errorLogger.logError(customErrorHandlerError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Custom error handler failed' } });
      }
    }

    if (this.props.enableUserNotification !== false) {
      toast.error(`Operation failed: ${error.message}`);
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    });
  };

  handleRetry = async () => {
    if (this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3)) {
      this.handleReset();
      return;
    }

    this.setState({ isRecovering: true });

    try {
      // For async operations, we'll use the graceful error handler
      const result = await gracefulErrorHandler.executeWithErrorHandling(
        async () => {
          // Simulate a small delay before reset
          await new Promise(resolve => setTimeout(resolve, 500));
          return true;
        },
        {
          strategy: 'retry',
          maxRetries: 1,
          retryDelay: 1000,
          showUserNotification: false,
          logError: false,
          context: {
            component: 'AsyncOperationErrorBoundary',
            function: 'handleRetry',
            operation: 'ASYNC_RETRY',
            additionalData: {
              recoveryAttempts: this.state.recoveryAttempts + 1,
            }
          }
        }
      );

      if (result.success) {
        this.setState(prevState => ({
          recoveryAttempts: prevState.recoveryAttempts + 1,
          isRecovering: false,
          recoveryStrategyUsed: 'async_retry',
        }));
        
        if (this.props.onRecover) {
          this.props.onRecover('async_retry');
        }
        
        // Reset after successful retry
        setTimeout(() => {
          this.handleReset();
        }, 100);
      } else {
        this.setState(prevState => ({
          recoveryAttempts: prevState.recoveryAttempts + 1,
          isRecovering: false,
        }));
      }
    } catch (error) {
      this.setState(prevState => ({
        recoveryAttempts: prevState.recoveryAttempts + 1,
        isRecovering: false,
      }));
    }
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[150px] p-5 bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl">
          <div className="flex items-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
            </svg>
            <h2 className="text-lg font-semibold text-blue-800">Operation Failed</h2>
          </div>

          <p className="text-blue-600 mb-3 text-center">
            {this.state.error.message || 'An operation failed unexpectedly.'}
          </p>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering}
              className="px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {this.state.isRecovering ? 'Retrying...' : 'Retry Operation'}
            </button>

            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all"
            >
              Dismiss
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Network Error Boundary
 * Specifically handles network-related errors
 */
export class NetworkErrorBoundary extends Component<EnhancedErrorBoundaryProps, EnhancedErrorBoundaryState> {
  constructor(props: EnhancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): EnhancedErrorBoundaryState {
    // Only catch network errors
    const isNetworkError =
      error.message.toLowerCase().includes('network') ||
      error.message.toLowerCase().includes('fetch') ||
      error.message.toLowerCase().includes('http') ||
      error.message.includes('502') ||
      error.message.includes('503') ||
      error.message.includes('504') ||
      error.message.includes('408') ||
      error.name === 'TypeError' ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('Network Error');

    if (isNetworkError) {
      return {
        hasError: true,
        error,
        errorInfo: null,
        recoveryAttempts: 0,
        isRecovering: false,
      };
    }

    // If it's not a network error, don't change state
    return {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Only process network errors
    const isNetworkError =
      error.message.toLowerCase().includes('network') ||
      error.message.toLowerCase().includes('fetch') ||
      error.message.toLowerCase().includes('http') ||
      error.message.includes('502') ||
      error.message.includes('503') ||
      error.message.includes('504') ||
      error.message.includes('408') ||
      error.name === 'TypeError' ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('Network Error');

    if (isNetworkError) {
      this.setState({ errorInfo });

      if (this.props.enableLogging !== false) {
        errorLogger.logError(error, 'error', {
          component: 'NetworkErrorBoundary',
          function: 'componentDidCatch',
          additionalData: {
            componentStack: errorInfo.componentStack,
            errorType: 'NETWORK_ERROR',
            recoveryAttempts: this.state.recoveryAttempts,
            context: this.props.errorContext || 'network',
          }
        });
      }

      if (this.props.onError) {
        try {
          this.props.onError(error, errorInfo);
        } catch (customErrorHandlerError) {
          errorLogger.logError(customErrorHandlerError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Custom error handler failed' } });
        }
      }

      if (this.props.enableUserNotification !== false) {
        toast.error(`Network error: ${error.message}. Please check your connection.`);
      }
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    });
  };

  handleRetry = async () => {
    if (this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3)) {
      this.handleReset();
      return;
    }

    this.setState({ isRecovering: true });

    try {
      // For network errors, we'll try to reconnect
      const result = await gracefulErrorHandler.executeWithErrorHandling(
        async () => {
          // Simulate checking network connectivity
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Check if we're online (browser API)
          if (typeof navigator !== 'undefined' && navigator.onLine) {
            return true;
          }
          throw new Error('Still offline');
        },
        {
          strategy: 'retry',
          maxRetries: 1,
          retryDelay: 1000,
          showUserNotification: false,
          logError: false,
          context: {
            component: 'NetworkErrorBoundary',
            function: 'handleRetry',
            operation: 'NETWORK_RETRY',
            additionalData: {
              recoveryAttempts: this.state.recoveryAttempts + 1,
            }
          }
        }
      );

      if (result.success) {
        this.setState(prevState => ({
          recoveryAttempts: prevState.recoveryAttempts + 1,
          isRecovering: false,
          recoveryStrategyUsed: 'network_retry',
        }));
        
        if (this.props.onRecover) {
          this.props.onRecover('network_retry');
        }
        
        // Reset after successful retry
        setTimeout(() => {
          this.handleReset();
        }, 100);
      } else {
        this.setState(prevState => ({
          recoveryAttempts: prevState.recoveryAttempts + 1,
          isRecovering: false,
        }));
      }
    } catch (error) {
      this.setState(prevState => ({
        recoveryAttempts: prevState.recoveryAttempts + 1,
        isRecovering: false,
      }));
    }
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[150px] p-5 bg-gradient-to-br from-indigo-50 to-indigo-100 border border-indigo-200 rounded-xl">
          <div className="flex items-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <h2 className="text-lg font-semibold text-indigo-800">Network Issue</h2>
          </div>

          <p className="text-indigo-600 mb-3 text-center">
            {this.state.error.message || 'A network error occurred. Please check your connection.'}
          </p>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering}
              className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-lg hover:from-indigo-700 hover:to-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {this.state.isRecovering ? 'Checking...' : 'Retry Connection'}
            </button>

            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all"
            >
              Dismiss
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}