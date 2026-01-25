import React from 'react';

interface PageErrorBoundaryProps {
  label: string;
  children: React.ReactNode;
}

interface PageErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class PageErrorBoundary extends React.Component<PageErrorBoundaryProps, PageErrorBoundaryState> {
  state: PageErrorBoundaryState = {
    hasError: false,
    message: '',
  };

  static getDerivedStateFromError(error: Error): PageErrorBoundaryState {
    return {
      hasError: true,
      message: error.message || 'An unexpected error occurred.',
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[PageErrorBoundary] ${this.props.label} failed`, error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, message: '' });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 p-6">
          <div className="mx-auto max-w-3xl rounded-xl border border-rose-200 bg-rose-50 px-6 py-5 text-rose-700">
            <div className="text-lg font-semibold">{this.props.label} unavailable</div>
            <p className="mt-2 text-sm">{this.state.message}</p>
            <button
              type="button"
              onClick={this.handleRetry}
              className="mt-4 rounded-md border border-rose-200 bg-white px-3 py-1.5 text-xs font-semibold text-rose-700 hover:bg-rose-100"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
