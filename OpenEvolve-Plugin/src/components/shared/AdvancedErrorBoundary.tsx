/**
 * Advanced Error Boundary Components
 * Provides sophisticated error boundary components with multiple recovery strategies
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { errorLogger } from '@/utils/errorLogging';
import { unifiedErrorHandlingService } from '@/utils/UnifiedErrorHandlingService';
import { enhancedErrorContextManager } from '@/utils/EnhancedErrorContext';
import { enhancedErrorReporter } from '@/utils/EnhancedErrorReporting';
import { advancedErrorRecoveryManager } from '@/utils/AdvancedErrorRecovery';
import { toast } from 'react-toastify';

// Define error boundary props
interface AdvancedErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void; retry: () => void; errorInfo?: ErrorInfo }>;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onRecover?: (strategyUsed: string) => void;
  recoveryStrategies?: string[]; // Specific strategies to try
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
}

// Define error boundary state
interface AdvancedErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  recoveryAttempts: number;
  isRecovering: boolean;
  recoveryStrategyUsed?: string;
}

/**
 * Advanced Error Boundary Component
 * Provides sophisticated error handling with multiple recovery strategies
 */
export class AdvancedErrorBoundary extends Component<AdvancedErrorBoundaryProps, AdvancedErrorBoundaryState> {
  constructor(props: AdvancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): AdvancedErrorBoundaryState {
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

    // Log the error
    if (this.props.enableLogging !== false) {
      errorLogger.logError(error, 'error', {
        component: 'AdvancedErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          componentStack: errorInfo.componentStack,
          recoveryAttempts: this.state.recoveryAttempts,
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

    // Report error
    if (this.props.enableErrorReporting !== false) {
      enhancedErrorContextManager.createEnhancedContext({
        component: 'AdvancedErrorBoundary',
        function: 'componentDidCatch',
        operation: 'RENDER_ERROR',
        additionalData: {
          componentStack: errorInfo.componentStack,
          recoveryAttempts: this.state.recoveryAttempts,
        }
      }).then(context => {
        enhancedErrorReporter.reportError(error, context, ['ui_error', 'render_error']);
      }).catch(reportError => {
        errorLogger.logError(reportError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error reporting failed' } });
      });
    }

    // Show user notification
    if (this.props.enableUserNotification !== false) {
      toast.error(`Interface error: ${error.message}`);
    }
  }

  componentDidUpdate(prevProps: AdvancedErrorBoundaryProps) {
    // Reset error boundary if reset keys change
    if (this.props.resetKeys && 
        JSON.stringify(prevProps.resetKeys) !== JSON.stringify(this.props.resetKeys)) {
      this.resetError();
    }
  }

  handleReset = () => {
    this.resetError();
  };

  resetError = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
      recoveryStrategyUsed: undefined,
    });
  };

  handleRetry = async () => {
    if (this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3)) {
      this.resetError();
      return;
    }

    this.setState({ isRecovering: true });

    try {
      // Create recovery context
      const recoveryContext = {
        error: this.state.error!,
        operation: 'UI_RECOVERY',
        params: {
          componentStack: this.state.errorInfo?.componentStack,
        },
        previousAttempts: this.state.recoveryAttempts,
        maxAttempts: this.props.maxRecoveryAttempts ?? 3,
        fallbackData: null,
        recoveryOptions: {
          failureThreshold: this.props.maxRecoveryAttempts ?? 3
        },
        metadata: {}
      };

      // Attempt recovery
      const recoveryResult = await advancedErrorRecoveryManager.applyRecoveryStrategy(recoveryContext);

      if (recoveryResult.success) {
        if (this.props.onRecover) {
          this.props.onRecover(recoveryResult.actionTaken);
        }

        this.setState({
          recoveryAttempts: this.state.recoveryAttempts + 1,
          isRecovering: false,
          recoveryStrategyUsed: recoveryResult.actionTaken,
        });

        if (recoveryResult.shouldRetry) {
          // Wait a bit before attempting to re-render
          await new Promise(resolve => setTimeout(resolve, 500));
          this.resetError();
        } else if (recoveryResult.newData !== undefined) {
          // If recovery provided new data, we could potentially update state
          // For now, just reset the error boundary
          this.resetError();
        } else {
          // Recovery successful but no specific action needed
          this.resetError();
        }
      } else {
        // Recovery failed, increment attempts
        this.setState({
          recoveryAttempts: this.state.recoveryAttempts + 1,
          isRecovering: false,
        });

        if (this.state.recoveryAttempts + 1 >= (this.props.maxRecoveryAttempts ?? 3)) {
          toast.error('Max recovery attempts reached. Please refresh the page.');
        } else {
          toast.info(`Recovery attempt ${this.state.recoveryAttempts + 1} failed. Retrying...`);
        }
      }
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
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <h2 className="text-xl font-semibold text-red-800">
              {this.props.customErrorTitle || 'Something went wrong'}
            </h2>
          </div>
          
          <p className="text-red-600 mb-4 text-center">
            {this.props.customErrorMessage || this.state.error.message}
          </p>
          
          {this.state.recoveryStrategyUsed && (
            <p className="text-red-500 text-sm mb-4">
              Last recovery attempt: {this.state.recoveryStrategyUsed}
            </p>
          )}
          
          {this.props.showDetailedError && this.state.errorInfo && (
            <details className="w-full mb-4">
              <summary className="text-left text-red-700 cursor-pointer">Error Details</summary>
              <pre className="mt-2 p-4 bg-red-100 text-red-900 text-xs rounded overflow-auto max-h-40">
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
          
          <div className="flex gap-3">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering || this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3)}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {this.state.isRecovering ? 'Recovering...' : 'Try Again'}
            </button>
            
            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            >
              Reset Component
            </button>
          </div>
          
          {this.state.recoveryAttempts >= (this.props.maxRecoveryAttempts ?? 3) && (
            <p className="mt-4 text-red-700 text-sm">
              Max recovery attempts reached. Please refresh the page or contact support.
            </p>
          )}
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
export class NetworkErrorBoundary extends Component<AdvancedErrorBoundaryProps, AdvancedErrorBoundaryState> {
  constructor(props: AdvancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): AdvancedErrorBoundaryState {
    // Only catch network errors
    const isNetworkError = 
      error.message.includes('Network Error') ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('NetworkError') ||
      error.message.includes('TypeError: Failed to fetch') ||
      error.message.includes('502') ||
      error.message.includes('503') ||
      error.message.includes('504');

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
      error.message.includes('Network Error') ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('NetworkError') ||
      error.message.includes('TypeError: Failed to fetch') ||
      error.message.includes('502') ||
      error.message.includes('503') ||
      error.message.includes('504');

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
      // For network errors, we'll just try to reset
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network check
      this.handleReset();
    } catch (error) {
      this.setState({
        recoveryAttempts: this.state.recoveryAttempts + 1,
        isRecovering: false,
      });

      if (this.state.recoveryAttempts + 1 >= (this.props.maxRecoveryAttempts ?? 3)) {
        toast.error('Max network recovery attempts reached.');
      } else {
        toast.info(`Network retry ${this.state.recoveryAttempts + 1} failed. Retrying...`);
      }
    }
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
            </svg>
            <h2 className="text-xl font-semibold text-blue-800">Network Issue</h2>
          </div>
          
          <p className="text-blue-600 mb-4 text-center">
            {this.state.error.message || 'A network error occurred. Please check your connection.'}
          </p>
          
          <div className="flex gap-3">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {this.state.isRecovering ? 'Checking...' : 'Retry Connection'}
            </button>
            
            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Suspense Error Boundary
 * Handles errors that occur during rendering of lazy-loaded components
 */
export class SuspenseErrorBoundary extends Component<AdvancedErrorBoundaryProps, AdvancedErrorBoundaryState> {
  constructor(props: AdvancedErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      recoveryAttempts: 0,
      isRecovering: false,
    };
  }

  static getDerivedStateFromError(error: Error): AdvancedErrorBoundaryState {
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
        component: 'SuspenseErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          componentStack: errorInfo.componentStack,
          recoveryAttempts: this.state.recoveryAttempts,
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
      toast.error(`Component loading error: ${error.message}`);
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
      // Try to reload the component by resetting
      await new Promise(resolve => setTimeout(resolve, 1000));
      this.handleReset();
    } catch (error) {
      this.setState({
        recoveryAttempts: this.state.recoveryAttempts + 1,
        isRecovering: false,
      });

      toast.error(`Loading retry failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-purple-50 border border-purple-200 rounded-lg">
          <div className="flex items-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-purple-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4 2a2 2 0 00-2 2v11a3 3 0 106 0V4a2 2 0 00-2-2H4zm1 14a1 1 0 100-2 1 1 0 000 2zm5-1.757l4.9-4.9a2 2 0 000-2.828L13.485 5.1a2 2 0 00-2.828 0L10 5.757v8.486zM16 18H9.071l6-6H16a2 2 0 012 2v2a2 2 0 01-2 2z" clipRule="evenodd" />
            </svg>
            <h2 className="text-xl font-semibold text-purple-800">Component Loading Error</h2>
          </div>
          
          <p className="text-purple-600 mb-4 text-center">
            Failed to load component: {this.state.error.message}
          </p>
          
          <div className="flex gap-3">
            <button
              type="button"
              onClick={this.handleRetry}
              disabled={this.state.isRecovering}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {this.state.isRecovering ? 'Reloading...' : 'Reload Component'}
            </button>
            
            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Higher-order component that wraps a component with AdvancedErrorBoundary
 */
export function withAdvancedErrorBoundary<P extends Record<string, any>>(
  Component: React.ComponentType<P>,
  boundaryProps?: Omit<AdvancedErrorBoundaryProps, 'children'>
) {
  return function ComponentWithErrorBoundary(props: P) {
    return (
      <AdvancedErrorBoundary {...boundaryProps}>
        <Component {...props} />
      </AdvancedErrorBoundary>
    );
  };
}

/**
 * Hook to use error boundary functionality in functional components
 */
export function useErrorBoundary() {
  const [error, setError] = React.useState<Error | null>(null);
  const [errorInfo, setErrorInfo] = React.useState<ErrorInfo | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
    setErrorInfo(null);
  }, []);

  const ErrorBoundaryFallback = React.useCallback(({ error, resetError: reset }: { error: Error; resetError: () => void }) => {
    return (
      <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          <h2 className="text-xl font-semibold text-red-800">Something went wrong</h2>
        </div>
        
        <p className="text-red-600 mb-4">{error.message}</p>
        
        <button
          type="button"
          onClick={reset}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }, []);

  return {
    error,
    errorInfo,
    resetError,
    ErrorBoundaryFallback,
    setError: (error: Error, errorInfo?: ErrorInfo) => {
      setError(error);
      setErrorInfo(errorInfo || null);
    },
    hasError: error !== null
  };
}