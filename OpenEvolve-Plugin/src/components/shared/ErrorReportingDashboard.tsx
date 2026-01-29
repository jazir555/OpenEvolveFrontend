/**
 * Error Reporting Dashboard
 * Provides a comprehensive view of application errors with filtering and analytics
 */

import React, { useState, useEffect, useMemo } from 'react';
import { errorLogger } from '@/utils/errorLogging';
import { toast } from 'react-toastify';

// Define error report types
interface ErrorReport {
  id: string;
  message: string;
  stack?: string;
  severity: 'debug' | 'info' | 'warn' | 'error' | 'critical';
  context: {
    component?: string;
    function?: string;
    url?: string;
    userAgent?: string;
    timestamp?: string;
    additionalData?: Record<string, any>;
  };
  timestamp: string;
  handled: boolean;
}

// Define filter options
interface ErrorFilter {
  severity?: ('debug' | 'info' | 'warn' | 'error' | 'critical')[];
  component?: string;
  dateRange?: { start: string; end: string };
  search?: string;
}

// Define dashboard props
interface ErrorReportingDashboardProps {
  maxReports?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const ErrorReportingDashboard: React.FC<ErrorReportingDashboardProps> = ({
  maxReports = 100,
  autoRefresh = false,
  refreshInterval = 30000,
}) => {
  const [reports, setReports] = useState<ErrorReport[]>([]);
  const [filters, setFilters] = useState<ErrorFilter>({});
  const [isLoading, setIsLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<ErrorReport | null>(null);
  const [expandedReport, setExpandedReport] = useState<string | null>(null);

  // Load error reports
  useEffect(() => {
    loadErrorReports();
    
    if (autoRefresh) {
      const interval = setInterval(loadErrorReports, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // Load error reports from logger
  const loadErrorReports = () => {
    setIsLoading(true);
    try {
      // Get recent error reports from the logger
      const recentReports = errorLogger.getRecentReports(maxReports);
      setReports(recentReports.map(report => ({
        ...report,
        timestamp: report.timestamp.toISOString(),
        context: {
          ...report.context,
          timestamp: report.context.timestamp?.toISOString(),
        }
      })));
    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Failed to load error reports' } });
      toast.error('Failed to load error reports');
    } finally {
      setIsLoading(false);
    }
  };

  // Apply filters to reports
  const filteredReports = useMemo(() => {
    return reports.filter(report => {
      // Filter by severity
      if (filters.severity && filters.severity.length > 0) {
        if (!filters.severity.includes(report.severity)) {
          return false;
        }
      }

      // Filter by component
      if (filters.component && filters.component.trim()) {
        const componentMatch = report.context.component?.toLowerCase().includes(filters.component.toLowerCase());
        const functionMatch = report.context.function?.toLowerCase().includes(filters.component.toLowerCase());
        if (!componentMatch && !functionMatch) {
          return false;
        }
      }

      // Filter by date range
      if (filters.dateRange) {
        const reportDate = new Date(report.timestamp);
        const startDate = new Date(filters.dateRange.start);
        const endDate = new Date(filters.dateRange.end);
        
        if (reportDate < startDate || reportDate > endDate) {
          return false;
        }
      }

      // Filter by search term
      if (filters.search && filters.search.trim()) {
        const searchTerm = filters.search.toLowerCase();
        const messageMatch = report.message.toLowerCase().includes(searchTerm);
        const componentMatch = report.context.component?.toLowerCase().includes(searchTerm);
        const functionMatch = report.context.function?.toLowerCase().includes(searchTerm);
        
        if (!messageMatch && !componentMatch && !functionMatch) {
          return false;
        }
      }

      return true;
    });
  }, [reports, filters]);

  // Get error statistics
  const errorStats = useMemo(() => {
    const stats = {
      total: filteredReports.length,
      bySeverity: {
        debug: 0,
        info: 0,
        warn: 0,
        error: 0,
        critical: 0,
      },
      byComponent: {} as Record<string, number>,
    };

    filteredReports.forEach(report => {
      stats.bySeverity[report.severity]++;
      
      const component = report.context.component || 'Unknown';
      stats.byComponent[component] = (stats.byComponent[component] || 0) + 1;
    });

    return stats;
  }, [filteredReports]);

  // Handle filter changes
  const handleFilterChange = (filterType: keyof ErrorFilter, value: any) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value,
    }));
  };

  // Clear all filters
  const clearFilters = () => {
    setFilters({});
  };

  // Toggle report expansion
  const toggleReportExpansion = (id: string) => {
    setExpandedReport(expandedReport === id ? null : id);
  };

  // Copy error details to clipboard
  const copyErrorDetails = (report: ErrorReport) => {
    const details = `
Error ID: ${report.id}
Message: ${report.message}
Severity: ${report.severity}
Component: ${report.context.component || 'Unknown'}
Function: ${report.context.function || 'Unknown'}
Timestamp: ${report.timestamp}
Stack: ${report.stack || 'N/A'}
Additional Data: ${JSON.stringify(report.context.additionalData, null, 2)}
    `.trim();

    navigator.clipboard.writeText(details);
    toast.success('Error details copied to clipboard');
  };

  // Export error reports
  const exportReports = () => {
    const dataStr = JSON.stringify(filteredReports, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportFileDefaultName = `error-reports-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success('Error reports exported successfully');
  };

  // Severity badge component
  const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
    const severityClasses = {
      debug: 'bg-gray-100 text-gray-800',
      info: 'bg-blue-100 text-blue-800',
      warn: 'bg-yellow-100 text-yellow-800',
      error: 'bg-red-100 text-red-800',
      critical: 'bg-purple-100 text-purple-800',
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${severityClasses[severity as keyof typeof severityClasses] || severityClasses.error}`}>
        {severity.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Error Reporting Dashboard</h1>
        <p className="text-gray-600">Monitor and analyze application errors with detailed insights</p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Total Errors</h3>
          <p className="text-2xl font-bold text-gray-900">{errorStats.total}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Critical</h3>
          <p className="text-2xl font-bold text-red-600">{errorStats.bySeverity.critical}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">High</h3>
          <p className="text-2xl font-bold text-orange-600">{errorStats.bySeverity.error}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Medium</h3>
          <p className="text-2xl font-bold text-yellow-600">{errorStats.bySeverity.warn}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500">Low</h3>
          <p className="text-2xl font-bold text-blue-600">{errorStats.bySeverity.info + errorStats.bySeverity.debug}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
            <select
              multiple
              value={filters.severity || []}
              onChange={(e) => handleFilterChange('severity', Array.from(e.target.selectedOptions, option => option.value))}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            >
              <option value="debug">Debug</option>
              <option value="info">Info</option>
              <option value="warn">Warning</option>
              <option value="error">Error</option>
              <option value="critical">Critical</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Component</label>
            <input
              type="text"
              value={filters.component || ''}
              onChange={(e) => handleFilterChange('component', e.target.value)}
              placeholder="Filter by component"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
            <input
              type="text"
              value={filters.search || ''}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              placeholder="Search in messages"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
          
          <div className="flex items-end space-x-2">
            <button
              onClick={clearFilters}
              className="w-full px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 transition-colors"
            >
              Clear Filters
            </button>
            <button
              onClick={exportReports}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
            >
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Error List */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">
            Error Reports ({filteredReports.length})
          </h2>
        </div>
        
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading error reports...</p>
          </div>
        ) : filteredReports.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-gray-600">No error reports found with current filters.</p>
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {filteredReports.map((report) => (
              <li key={report.id} className="hover:bg-gray-50 transition-colors">
                <div 
                  className="px-6 py-4 cursor-pointer"
                  onClick={() => toggleReportExpansion(report.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <SeverityBadge severity={report.severity} />
                      <div>
                        <p className="text-sm font-medium text-gray-900 truncate max-w-md">
                          {report.message}
                        </p>
                        <p className="text-sm text-gray-500">
                          {report.context.component || 'Unknown Component'} • {new Date(report.timestamp).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          copyErrorDetails(report);
                        }}
                        className="text-gray-400 hover:text-gray-600"
                        title="Copy details"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                          <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
                        </svg>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedReport(report);
                        }}
                        className="text-gray-400 hover:text-gray-600"
                        title="View details"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                          <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                        </svg>
                      </button>
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        className={`h-5 w-5 text-gray-400 transform transition-transform ${expandedReport === report.id ? 'rotate-180' : ''}`}
                        viewBox="0 0 20 20" 
                        fill="currentColor"
                      >
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                  
                  {expandedReport === report.id && (
                    <div className="mt-4 pl-8 pr-4 py-3 bg-gray-50 rounded-lg text-sm">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium text-gray-900 mb-1">Details</h4>
                          <p className="text-gray-700 break-words">{report.message}</p>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 mb-1">Context</h4>
                          <p className="text-gray-700">
                            Component: {report.context.component || 'Unknown'}<br />
                            Function: {report.context.function || 'Unknown'}<br />
                            URL: {report.context.url || 'Unknown'}<br />
                            Time: {new Date(report.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      
                      {report.stack && (
                        <div className="mt-3">
                          <h4 className="font-medium text-gray-900 mb-1">Stack Trace</h4>
                          <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded overflow-x-auto max-h-40">
                            {report.stack}
                          </pre>
                        </div>
                      )}
                      
                      {report.context.additionalData && (
                        <div className="mt-3">
                          <h4 className="font-medium text-gray-900 mb-1">Additional Data</h4>
                          <pre className="text-xs bg-gray-100 p-3 rounded overflow-x-auto">
                            {JSON.stringify(report.context.additionalData, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Error Detail Modal */}
      {selectedReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900">Error Details</h2>
                  <p className="text-gray-600">{selectedReport.id}</p>
                </div>
                <button
                  onClick={() => setSelectedReport(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-900 mb-1">Message</h3>
                  <p className="text-gray-700">{selectedReport.message}</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Severity</h3>
                    <SeverityBadge severity={selectedReport.severity} />
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Timestamp</h3>
                    <p className="text-gray-700">{new Date(selectedReport.timestamp).toLocaleString()}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Component</h3>
                    <p className="text-gray-700">{selectedReport.context.component || 'Unknown'}</p>
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Function</h3>
                    <p className="text-gray-700">{selectedReport.context.function || 'Unknown'}</p>
                  </div>
                </div>
                
                {selectedReport.stack && (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Stack Trace</h3>
                    <pre className="text-xs bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto max-h-60">
                      {selectedReport.stack}
                    </pre>
                  </div>
                )}
                
                {selectedReport.context.additionalData && (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">Additional Data</h3>
                    <pre className="text-xs bg-gray-100 p-4 rounded overflow-x-auto">
                      {JSON.stringify(selectedReport.context.additionalData, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-end space-x-3">
                <button
                  onClick={() => copyErrorDetails(selectedReport)}
                  className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 transition-colors"
                >
                  Copy Details
                </button>
                <button
                  onClick={() => setSelectedReport(null)}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ErrorReportingDashboard;