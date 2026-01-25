import React from 'react';

interface VizErrorBoundaryProps {
  label: string;
  children: React.ReactNode;
}

interface VizErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class VizErrorBoundary extends React.Component<VizErrorBoundaryProps, VizErrorBoundaryState> {
  state: VizErrorBoundaryState = {
    hasError: false,
    message: '',
  };

  static getDerivedStateFromError(error: Error): VizErrorBoundaryState {
    return {
      hasError: true,
      message: error.message || 'Visualization failed to load.',
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[VizErrorBoundary] ${this.props.label} failed`, error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, message: '' });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          <div className="font-semibold">{this.props.label} unavailable</div>
          <div className="mt-1">{this.state.message}</div>
          <button
            type="button"
            onClick={this.handleRetry}
            className="mt-3 rounded-md border border-rose-200 bg-white px-3 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-100"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
