import React from 'react';
import { errorLogger } from '@/utils';

interface ComponentErrorBoundaryProps {
  label: string;
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ComponentErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class ComponentErrorBoundary extends React.Component<
  ComponentErrorBoundaryProps,
  ComponentErrorBoundaryState
> {
  state: ComponentErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  static getDerivedStateFromError(error: Error): ComponentErrorBoundaryState {
    try {
      return {
        hasError: true,
        error,
        errorInfo: null, // Will be set in componentDidCatch
      };
    } catch (stateError) {
      errorLogger.logError(stateError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in getDerivedStateFromError' } });
      return {
        hasError: true,
        error: stateError instanceof Error ? stateError : new Error(String(stateError)),
        errorInfo: null,
      };
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    try {
      // Update errorInfo in state
      this.setState({ errorInfo });

      console.error(`[ComponentErrorBoundary] ${this.props.label} failed`, error, errorInfo);

      // Call custom error handler if provided
      if (this.props.onError) {
        try {
          this.props.onError(error, errorInfo);
        } catch (customErrorHandlerError) {
          errorLogger.logError(customErrorHandlerError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Custom error handler failed' } });
        }
      }

      // Log the error using our centralized error logging system
      errorLogger.logError(error, 'error', {
        component: 'ComponentErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          label: this.props.label,
          componentStack: errorInfo.componentStack,
        }
      });
    } catch (loggingError) {
      errorLogger.logError(loggingError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in componentDidCatch error logging' } });
      // Even if logging fails, we still want to maintain the error state
      if (!this.state.error) {
        this.setState({
          error: loggingError instanceof Error ? loggingError : new Error(String(loggingError)),
          errorInfo,
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
      });
    } catch (resetError) {
      errorLogger.logError(resetError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in handleReset' } });
      errorLogger.logError(
        resetError instanceof Error ? resetError : new Error(String(resetError)),
        'error',
        { component: 'ComponentErrorBoundary', function: 'handleReset', additionalData: { label: this.props.label } }
      );

      // Force a state reset even if the setState fails
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
      });
    }
  };

  render() {
    try {
      if (this.state.hasError && this.state.error) {
        // Use custom fallback if provided
        if (this.props.fallback) {
          const FallbackComponent = this.props.fallback;
          return <FallbackComponent error={this.state.error} resetError={this.handleReset} />;
        }

        // Default fallback UI
        return (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
            <div className="font-semibold">{this.props.label} unavailable</div>
            <div className="mt-1">{this.state.error.message}</div>
            {this.state.errorInfo && (
              <details className="mt-2 text-[10px] text-rose-600">
                <summary>Error details</summary>
                <pre className="mt-1 whitespace-pre-wrap break-words">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
            <button
              type="button"
              onClick={this.handleReset}
              className="mt-2 rounded-md border border-rose-200 bg-white px-2 py-1 text-[10px] font-semibold text-rose-700 hover:bg-rose-100"
            >
              Reset Component
            </button>
          </div>
        );
      }

      return this.props.children;
    } catch (renderError) {
      errorLogger.logError(renderError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in ComponentErrorBoundary render' } });
      errorLogger.logError(
        renderError instanceof Error ? renderError : new Error(String(renderError)),
        'error',
        { component: 'ComponentErrorBoundary', function: 'render', additionalData: { label: this.props.label } }
      );

      // Absolute fallback UI in case of render error
      return (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
          <div className="font-semibold">{this.props.label} unavailable</div>
          <div className="mt-1">An unexpected error occurred while rendering this component.</div>
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-2 rounded-md border border-rose-200 bg-white px-2 py-1 text-[10px] font-semibold text-rose-700 hover:bg-rose-100"
          >
            Reset Component
          </button>
        </div>
      );
    }
  }
}

export function withComponentBoundary<P>(
  Component: React.ComponentType<P>,
  label: string,
  options?: {
    fallback?: React.ComponentType<{ error: Error; resetError: () => void }>;
    onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  }
) {
  try {
    const Wrapped = (props: P) => (
      <ComponentErrorBoundary
        label={label}
        fallback={options?.fallback}
        onError={options?.onError}
      >
        <Component {...props} />
      </ComponentErrorBoundary>
    );
    Wrapped.displayName = `WithComponentBoundary(${label})`;
    return Wrapped;
  } catch (boundaryError) {
    errorLogger.logError(boundaryError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating component boundary' } });
    errorLogger.logError(
      boundaryError instanceof Error ? boundaryError : new Error(String(boundaryError)),
      'error',
      { component: 'ComponentErrorBoundary', function: 'withComponentBoundary', additionalData: { label } }
    );

    // Return the original component if boundary creation fails
    return Component as React.ComponentType<any>;
  }
}

/**
 * Higher-order component that adds error boundary to all child components
 */
export function withErrorBoundary<T extends Record<string, any>>(
  Component: React.ComponentType<T>,
  boundaryProps?: Omit<ComponentErrorBoundaryProps, 'children'>
) {
  return function ComponentWithErrorBoundary(props: T) {
    return (
      <ComponentErrorBoundary {...boundaryProps} label={boundaryProps?.label || Component.name || 'Unknown'}>
        <Component {...props} />
      </ComponentErrorBoundary>
    );
  };
}

/**
 * Error boundary component that only catches JavaScript errors
 */
export class JSOnlyErrorBoundary extends React.Component<
  ComponentErrorBoundaryProps,
  ComponentErrorBoundaryState
> {
  state: ComponentErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  static getDerivedStateFromError(error: Error): ComponentErrorBoundaryState {
    // Only catch JavaScript errors, not network or other errors
    if (error instanceof Error) {
      return { hasError: true, error, errorInfo: null };
    }
    return { hasError: false, error: null, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    try {
      this.setState({ errorInfo });

      console.error(`[JSOnlyErrorBoundary] ${this.props.label} failed`, error, errorInfo);

      // Log the error
      errorLogger.logError(error, 'error', {
        component: 'JSOnlyErrorBoundary',
        function: 'componentDidCatch',
        additionalData: {
          label: this.props.label,
          componentStack: errorInfo.componentStack,
        }
      });
    } catch (loggingError) {
      errorLogger.logError(loggingError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in JSOnlyErrorBoundary componentDidCatch' } });
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          <div className="font-semibold">{this.props.label} encountered a JavaScript error</div>
          <div className="mt-1">{this.state.error.message}</div>
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-2 rounded-md border border-amber-200 bg-white px-2 py-1 text-[10px] font-semibold text-amber-700 hover:bg-amber-100"
          >
            Reset Component
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Network error boundary - catches network-related errors
 */
export class NetworkErrorBoundary extends React.Component<
  ComponentErrorBoundaryProps,
  ComponentErrorBoundaryState
> {
  state: ComponentErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  static getDerivedStateFromError(error: Error): ComponentErrorBoundaryState {
    // Check if this is a network error
    const isNetworkError =
      error.message.includes('Network Error') ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('NetworkError') ||
      error.message.includes('TypeError: Failed to fetch');

    if (isNetworkError) {
      return { hasError: true, error, errorInfo: null };
    }
    return { hasError: false, error: null, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    try {
      const isNetworkError =
        error.message.includes('Network Error') ||
        error.message.includes('Failed to fetch') ||
        error.message.includes('NetworkError') ||
        error.message.includes('TypeError: Failed to fetch');

      if (isNetworkError) {
        this.setState({ errorInfo });

        console.error(`[NetworkErrorBoundary] ${this.props.label} network error`, error, errorInfo);

        // Log the network error
        errorLogger.logError(error, 'error', {
          component: 'NetworkErrorBoundary',
          function: 'componentDidCatch',
          additionalData: {
            label: this.props.label,
            componentStack: errorInfo.componentStack,
            errorType: 'NETWORK_ERROR',
          }
        });
      }
    } catch (loggingError) {
      errorLogger.logError(loggingError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in NetworkErrorBoundary componentDidCatch' } });
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-700">
          <div className="font-semibold">{this.props.label} network issue</div>
          <div className="mt-1">There was a network issue loading this component.</div>
          <button
            type="button"
            onClick={this.handleRetry}
            className="mt-2 rounded-md border border-blue-200 bg-white px-2 py-1 text-[10px] font-semibold text-blue-700 hover:bg-blue-100"
          >
            Retry Connection
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
