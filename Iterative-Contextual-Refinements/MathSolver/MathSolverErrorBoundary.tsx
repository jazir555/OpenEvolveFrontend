/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver Error Boundary
 * 
 * Catches errors in MathSolver UI components and displays a fallback UI.
 * Prevents the entire application from crashing due to MathSolver errors.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children: ReactNode;
    onReset?: () => void;
}

interface State {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

export class MathSolverErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
        errorInfo: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error, errorInfo: null };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        console.error('[MathSolverErrorBoundary] Uncaught error:', error);
        console.error('[MathSolverErrorBoundary] Component stack:', errorInfo.componentStack);
        this.setState({ errorInfo });
        
        // Report to analytics if available
        if (typeof window !== 'undefined' && (window as any).analytics) {
            (window as any).analytics.track('mathsolver_error_boundary_triggered', {
                error_message: error.message,
                error_stack: error.stack,
                component_stack: errorInfo.componentStack
            });
        }
    }

    private handleReset = (): void => {
        this.setState({ hasError: false, error: null, errorInfo: null });
        this.props.onReset?.();
    };

    private handleReload = (): void => {
        window.location.reload();
    };

    public render(): ReactNode {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '24px',
                    backgroundColor: '#ffebee',
                    border: '1px solid #ef5350',
                    borderRadius: '8px',
                    maxWidth: '600px',
                    margin: '20px auto'
                }}>
                    <h2 style={{ color: '#c62828', marginTop: 0 }}>
                        MathSolver Error
                    </h2>
                    <p style={{ color: '#37474f' }}>
                        The MathSolver encountered an unexpected error. This could be due to:
                    </p>
                    <ul style={{ color: '#37474f' }}>
                        <li>A problem with the backend connection</li>
                        <li>An incompatible browser or environment</li>
                        <li>A bug in the MathSolver component</li>
                    </ul>
                    
                    {process.env.NODE_ENV === 'development' && this.state.error && (
                        <details style={{ marginTop: '16px', marginBottom: '16px' }}>
                            <summary style={{ cursor: 'pointer', color: '#37474f' }}>
                                Error Details (Development Only)
                            </summary>
                            <pre style={{
                                backgroundColor: '#fff',
                                padding: '12px',
                                borderRadius: '4px',
                                overflow: 'auto',
                                fontSize: '12px',
                                marginTop: '8px'
                            }}>
                                {this.state.error.toString()}
                                {this.state.errorInfo?.componentStack}
                            </pre>
                        </details>
                    )}
                    
                    <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                        <button
                            onClick={this.handleReset}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: '#1976d2',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Try Again
                        </button>
                        <button
                            onClick={this.handleReload}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: '#757575',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Reload Page
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default MathSolverErrorBoundary;
