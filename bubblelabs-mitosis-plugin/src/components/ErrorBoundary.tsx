import React from 'react';
import { logger } from '../utils/logger';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class MitosisErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    try {
      return { hasError: true, error };
    } catch (getStateError) {
      logger.error('Error in getDerivedStateFromError:', getStateError);
      return { hasError: true };
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    try {
      logger.error('Mitosis plugin error caught by boundary:', error, errorInfo);
    } catch (loggingError) {
      // If logging fails, at least try to console.error
      try {
        console.error('Mitosis plugin error caught by boundary:', error, errorInfo);
      } catch (consoleError) {
        // If all logging fails, silently continue
      }
    }
  }

  render() {
    try {
      if (this.state.hasError) {
        // You can render any custom fallback UI
        return (
          <div style={{
            padding: '1rem',
            backgroundColor: '#fee',
            border: '1px solid #ecc',
            borderRadius: '4px',
            color: '#900'
          }}>
            <h3>Animation temporarily unavailable</h3>
            <p>The mitosis animation component encountered an error and has been disabled.</p>
            <button
              onClick={() => {
                try {
                  this.setState({ hasError: false });
                } catch (resetError) {
                  logger.error('Error resetting error boundary:', resetError);
                }
              }}
              style={{
                marginTop: '0.5rem',
                padding: '0.25rem 0.5rem',
                backgroundColor: '#4F46E5',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Retry Animation
            </button>
          </div>
        );
      }
    } catch (renderError) {
      logger.error('Error rendering error boundary fallback:', renderError);
      // Fallback to simple error message
      return (
        <div style={{
          padding: '1rem',
          backgroundColor: '#fee',
          border: '1px solid #ecc',
          borderRadius: '4px',
          color: '#900'
        }}>
          <p>Animation component unavailable due to an error.</p>
        </div>
      );
    }

    return this.props.children;
  }
}