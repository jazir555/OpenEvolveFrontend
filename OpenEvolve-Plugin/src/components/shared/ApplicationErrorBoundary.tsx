/**
 * Application-Level Error Boundary
 * Provides comprehensive error handling with graceful degradation and user-friendly messaging
 */

import React from 'react';
import { errorLogger } from '@/utils/errorLogging';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { toast } from 'react-toastify';

interface ApplicationErrorBoundaryProps {
  label: string;
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void; retry: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  enableLogging?: boolean;
  enableUserNotification?: boolean;
  enableAutoRetry?: boolean;
  maxRetries?: number;
  retryDelay?: number;
}

interface ApplicationErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  retryCount: number;
}

export class ApplicationErrorBoundary extends React.Component<
  ApplicationErrorBoundaryProps,
  ApplicationErrorBoundaryState
> {
  state: ApplicationErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
    retryCount: 0,
  };

  static getDerivedStateFromError(error: Error): ApplicationErrorBoundaryState {
    try {
      return {
        hasError: true,
        error,
        errorInfo: null, // Will be set in componentDidCatch
        retryCount: 0,
      };
    } catch (stateError) {
      errorLogger.logError(stateError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in getDerivedStateFromError' } });
      return {
        hasError: true,
        error: stateError instanceof Error ? stateError : new Error(String(stateError)),
        errorInfo: null,
        retryCount: 0,
      };
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    try {
      // Update errorInfo in state
      this.setState({ errorInfo });

      errorLogger.logError(error, 'critical', {
        component: `ApplicationErrorBoundary-${this.props.label}`,
        function: 'componentDidCatch',
        additionalData: { errorInfo, label: this.props.label }
      });

      // Call custom error handler if provided
      if (this.props.onError) {
        try {
          this.props.onError(error, errorInfo);
        } catch (customErrorHandlerError) {
          errorLogger.logError(customErrorHandlerError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Custom error handler failed' } });
        }
      }

      // Log the error using our centralized error logging system
      if (this.props.enableLogging !== false) {
        errorLogger.logError(error, 'error', {
          component: 'ApplicationErrorBoundary',
          function: 'componentDidCatch',
          additionalData: {
            label: this.props.label,
            componentStack: errorInfo.componentStack,
            retryCount: this.state.retryCount,
          }
        });
      }

      // Show user notification if enabled
      if (this.props.enableUserNotification !== false) {
        toast.error(`Application error in ${this.props.label}: ${error.message}`);
      }
    } catch (loggingError) {
      errorLogger.logError(loggingError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in componentDidCatch error logging' } });
      // Even if logging fails, we still want to maintain the error state
      if (!this.state.error) {
        this.setState({
          error: loggingError instanceof Error ? loggingError : new Error(String(loggingError)),
          errorInfo,
          retryCount: this.state.retryCount,
        });
      }
    }
  }

  handleReset = () => {
    try {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: 0,
      });
    } catch (resetError) {
      errorLogger.logError(resetError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in handleReset' } });
      errorLogger.logError(
        resetError instanceof Error ? resetError : new Error(String(resetError)),
        'error',
        { 
          component: 'ApplicationErrorBoundary', 
          function: 'handleReset', 
          additionalData: { 
            label: this.props.label,
            retryCount: this.state.retryCount
          } 
        }
      );

      // Force a state reset even if the setState fails
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: 0,
      });
    }
  };

  handleRetry = async () => {
    const maxRetries = this.props.maxRetries ?? 3;
    
    if (this.state.retryCount >= maxRetries) {
      // Max retries reached, just reset
      this.handleReset();
      return;
    }

    try {
      // Attempt graceful recovery
      const result = await gracefulErrorHandler.executeWithErrorHandling(
        async () => {
          // Try to reset the error state
          this.setState(prevState => ({
            ...prevState,
            hasError: false,
            error: null,
            errorInfo: null,
            retryCount: prevState.retryCount + 1,
          }));
          
          // Wait a bit before attempting to re-render
          await new Promise(resolve => setTimeout(resolve, 500));
          
          return true;
        },
        {
          strategy: 'retry',
          maxRetries: 1,
          retryDelay: this.props.retryDelay ?? 1000,
          showUserNotification: false, // We'll handle notifications ourselves
          logError: false, // We'll handle logging ourselves
          context: {
            component: 'ApplicationErrorBoundary',
            function: 'handleRetry',
            operation: `RETRY_RENDER_${this.props.label}`,
            additionalData: {
              retryCount: this.state.retryCount + 1,
              maxRetries,
            }
          }
        }
      );

      if (!result.success) {
        // If graceful recovery failed, show notification and increment retry count
        this.setState(prevState => ({
          ...prevState,
          retryCount: prevState.retryCount + 1,
        }));

        if (this.state.retryCount + 1 >= maxRetries) {
          toast.error(`Max retries reached for ${this.props.label}. Component remains unavailable.`);
        } else {
          toast.info(`Retrying to restore ${this.props.label}... (Attempt ${this.state.retryCount + 1}/${maxRetries})`);
        }
      } else {
        // Recovery successful
        toast.success(`${this.props.label} recovered successfully!`);
      }
    } catch (retryError) {
      errorLogger.logError(retryError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in handleRetry' } });
      this.setState(prevState => ({
        ...prevState,
        retryCount: prevState.retryCount + 1,
      }));
      
      toast.error(`Retry failed: ${retryError instanceof Error ? retryError.message : String(retryError)}`);
    }
  };

  render() {
    try {
      if (this.state.hasError && this.state.error) {
        // Use custom fallback if provided
        if (this.props.fallback) {
          const FallbackComponent = this.props.fallback;
          return (
            <FallbackComponent 
              error={this.state.error} 
              resetError={this.handleReset} 
              retry={this.handleRetry} 
            />
          );
        }

        // Default fallback UI
        return (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <div className="font-semibold flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {this.props.label} unavailable
            </div>
            <div className="mt-1 text-xs">{this.state.error.message}</div>
            {this.state.errorInfo && (
              <details className="mt-2 text-[10px] text-red-600">
                <summary>Error details</summary>
                <pre className="mt-1 whitespace-pre-wrap break-words font-mono">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={this.handleRetry}
                disabled={this.state.retryCount >= (this.props.maxRetries ?? 3)}
                className="rounded-md border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {this.state.retryCount > 0 
                  ? `Retry (${this.state.retryCount}/${this.props.maxRetries ?? 3})` 
                  : 'Retry'}
              </button>
              <button
                type="button"
                onClick={this.handleReset}
                className="rounded-md border border-gray-300 bg-white px-2.5 py-1 text-xs font-medium text-gray-700 hover:bg-gray-100"
              >
                Reset
              </button>
            </div>
          </div>
        );
      }

      return this.props.children;
    } catch (renderError) {
      errorLogger.logError(renderError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in ApplicationErrorBoundary render' } });
      errorLogger.logError(
        renderError instanceof Error ? renderError : new Error(String(renderError)),
        'error',
        { 
          component: 'ApplicationErrorBoundary', 
          function: 'render', 
          additionalData: { 
            label: this.props.label,
            retryCount: this.state.retryCount
          } 
        }
      );

      // Absolute fallback UI in case of render error
      return (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <div className="font-semibold flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            {this.props.label} unavailable
          </div>
          <div className="mt-1 text-xs">An unexpected error occurred while rendering this component.</div>
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-3 rounded-md border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100"
          >
            Reset Component
          </button>
        </div>
      );
    }
  }
}

/**
 * Higher-order component that wraps a component with ApplicationErrorBoundary
 */
export function withApplicationErrorBoundary<P>(
  Component: React.ComponentType<P>,
  label: string,
  options?: Omit<ApplicationErrorBoundaryProps, 'children' | 'label'>
) {
  try {
    const Wrapped = (props: P) => (
      <ApplicationErrorBoundary
        label={label}
        enableLogging={options?.enableLogging ?? true}
        enableUserNotification={options?.enableUserNotification ?? true}
        enableAutoRetry={options?.enableAutoRetry ?? true}
        maxRetries={options?.maxRetries ?? 3}
        retryDelay={options?.retryDelay ?? 1000}
        fallback={options?.fallback}
        onError={options?.onError}
      >
        <Component {...props} />
      </ApplicationErrorBoundary>
    );
    Wrapped.displayName = `WithApplicationErrorBoundary(${label})`;
    return Wrapped;
  } catch (boundaryError) {
    errorLogger.logError(boundaryError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating application error boundary' } });
    errorLogger.logError(
      boundaryError instanceof Error ? boundaryError : new Error(String(boundaryError)),
      'error',
      { 
        component: 'ApplicationErrorBoundary', 
        function: 'withApplicationErrorBoundary', 
        additionalData: { label } 
      }
    );

    // Return the original component if boundary creation fails
    return Component as React.ComponentType<any>;
  }
}