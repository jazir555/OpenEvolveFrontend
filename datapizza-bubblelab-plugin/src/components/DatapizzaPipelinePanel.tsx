// Datapizza Pipeline Panel Component
// React component for running data pipelines

import React, { useState, useEffect } from 'react';
import { X, Play, Pause, Stop, Check, AlertTriangle, Info, Database, Pipeline, Clock, BarChart2 } from 'lucide-react';
import { DatapizzaPipelineResult, DATAPIZZA_PIPELINE_TYPES } from '../types/plugin-types';

export interface DatapizzaPipelinePanelProps {
  /** Data source to process */
  dataSource: string;
  
  /** Optional initial pipeline type */
  initialPipelineType?: string;
  
  /** Callback with pipeline result */
  onResult: (result: DatapizzaPipelineResult) => void;
  
  /** Callback when panel is closed */
  onClose: () => void;
  
  /** Show debug information */
  showDebug?: boolean;
}

export function DatapizzaPipelinePanel({
  dataSource,
  initialPipelineType,
  onResult,
  onClose,
  showDebug = false
}: DatapizzaPipelinePanelProps) {
  const [pipelineType, setPipelineType] = useState(initialPipelineType || 'standard');
  const [status, setStatus] = useState<'idle' | 'running' | 'paused' | 'completed' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DatapizzaPipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [endTime, setEndTime] = useState<Date | null>(null);

  useEffect(() => {
    if (initialPipelineType) {
      setPipelineType(initialPipelineType);
    }
  }, [initialPipelineType]);

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `${new Date().toISOString()} - ${message}`]);
  };

  const handleRunPipeline = async () => {
    try {
      setStatus('running');
      setProgress(0);
      setResult(null);
      setError(null);
      setLogs([]);
      setStartTime(new Date());
      setEndTime(null);

      addLog(`Starting pipeline: ${pipelineType}`);
      addLog(`Data source: ${dataSource.substring(0, 100)}...`);

      // Simulate pipeline execution with progress updates
      for (let i = 10; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setProgress(i);
        addLog(`Progress: ${i}%`);
      }

      // Simulate successful result
      const mockResult: DatapizzaPipelineResult = {
        success: true,
        pipelineId: `pipeline_${Date.now()}`,
        dataSource,
        processedData: {
          recordsProcessed: 1000,
          chunksCreated: 100,
          embeddingsGenerated: 100,
          vectorStoreUpdated: true
        },
        confidenceScore: 0.95,
        pipelineType,
        dataDomain: 'structured',
        errors: [],
        warnings: ['Some data fields were empty and were skipped'],
        executionTime: 15000,
        metadata: {
          timestamp: new Date().toISOString(),
          processingSteps: ['validation', 'chunking', 'embedding', 'vector_storage']
        },
        timestamp: new Date()
      };

      setResult(mockResult);
      setStatus('completed');
      setEndTime(new Date());
      addLog('Pipeline completed successfully');

      // Call the onResult callback
      onResult(mockResult);

    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unknown error');
      addLog(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setEndTime(new Date());
    }
  };

  const handlePausePipeline = () => {
    setStatus('paused');
    addLog('Pipeline paused');
  };

  const handleStopPipeline = () => {
    setStatus('idle');
    setProgress(0);
    addLog('Pipeline stopped');
  };

  const getExecutionTime = () => {
    if (!startTime) return '0s';
    if (endTime) {
      const diff = endTime.getTime() - startTime.getTime();
      return `${(diff / 1000).toFixed(1)}s`;
    }
    const diff = Date.now() - startTime.getTime();
    return `${(diff / 1000).toFixed(1)}s`;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <Pipeline className="h-5 w-5 text-blue-500" />
            <h3 className="font-semibold text-lg">Datapizza Pipeline</h3>
            {status === 'running' && (
              <span className="flex items-center gap-1 text-sm text-blue-600">
                <Clock className="h-3 w-3" />
                <span>{getExecutionTime()}</span>
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            aria-label="Close pipeline panel"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden p-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full">
            {/* Left Column - Configuration */}
            <div className="lg:col-span-1 space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                  <Database className="h-4 w-4" />
                  Data Source
                </h4>
                <div className="text-sm text-gray-600 dark:text-gray-300 break-words">
                  {dataSource}
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                  <Pipeline className="h-4 w-4" />
                  Pipeline Configuration
                </h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Pipeline Type
                    </label>
                    <select
                      value={pipelineType}
                      onChange={(e) => setPipelineType(e.target.value)}
                      disabled={status === 'running'}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white disabled:opacity-50"
                    >
                      {DATAPIZZA_PIPELINE_TYPES.map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Description
                    </label>
                    <div className="text-sm text-gray-600 dark:text-gray-300">
                      {DATAPIZZA_PIPELINE_TYPES.find(t => t.value === pipelineType)?.description}
                    </div>
                  </div>
                </div>
              </div>

              {/* Pipeline Controls */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                  <Play className="h-4 w-4" />
                  Pipeline Controls
                </h4>
                <div className="flex gap-2">
                  <button
                    onClick={handleRunPipeline}
                    disabled={status === 'running'}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:bg-blue-400"
                  >
                    <Play className="h-4 w-4" />
                    Run Pipeline
                  </button>
                  <button
                    onClick={handlePausePipeline}
                    disabled={status !== 'running'}
                    className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    <Pause className="h-4 w-4" />
                  </button>
                  <button
                    onClick={handleStopPipeline}
                    disabled={status === 'idle' || status === 'completed' || status === 'error'}
                    className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    <Stop className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Center Column - Progress and Status */}
            <div className="lg:col-span-1 space-y-4">
              {/* Status Indicator */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                  <Info className="h-4 w-4" />
                  Pipeline Status
                </h4>
                <div className="flex items-center gap-2">
                  {status === 'idle' && (
                    <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                      <Clock className="h-4 w-4" />
                      <span>Ready to start</span>
                    </div>
                  )}
                  {status === 'running' && (
                    <div className="flex items-center gap-2 text-blue-600">
                      <Clock className="h-4 w-4 animate-spin" />
                      <span>Running...</span>
                    </div>
                  )}
                  {status === 'paused' && (
                    <div className="flex items-center gap-2 text-yellow-600">
                      <Pause className="h-4 w-4" />
                      <span>Paused</span>
                    </div>
                  )}
                  {status === 'completed' && (
                    <div className="flex items-center gap-2 text-green-600">
                      <Check className="h-4 w-4" />
                      <span>Completed successfully</span>
                    </div>
                  )}
                  {status === 'error' && (
                    <div className="flex items-center gap-2 text-red-600">
                      <AlertTriangle className="h-4 w-4" />
                      <span>Error occurred</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Progress */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                  <BarChart2 className="h-4 w-4" />
                  Progress
                </h4>
                <div className="space-y-2">
                  <div className="text-sm text-gray-600 dark:text-gray-300">
                    {progress}% complete
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                    <div
                      className={`bg-blue-600 h-2.5 rounded-full transition-all duration-300 ${status === 'error' ? 'bg-red-600' : ''} ${status === 'completed' ? 'bg-green-600' : ''}`}
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Statistics */}
              {result && (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2 mb-3">
                    <BarChart2 className="h-4 w-4" />
                    Pipeline Statistics
                  </h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Confidence:</span>
                      <span className="font-medium">{(result.confidenceScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Execution Time:</span>
                      <span className="font-medium">{result.executionTime}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Records Processed:</span>
                      <span className="font-medium">{(result.processedData as any)?.recordsProcessed || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Chunks Created:</span>
                      <span className="font-medium">{(result.processedData as any)?.chunksCreated || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Logs */}
            <div className="lg:col-span-1 space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-full flex flex-col">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Pipeline Logs
                  </h4>
                  <button
                    onClick={() => setLogs([])}
                    disabled={logs.length === 0}
                    className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50"
                  >
                    Clear Logs
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto text-xs font-mono text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-600">
                  {logs.length === 0 ? (
                    <div className="text-gray-400 dark:text-gray-500">No logs yet. Start the pipeline to see activity.</div>
                  ) : (
                    logs.map((log, index) => (
                      <div key={index} className="mb-1 break-words">
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0" />
              <div>
                <h5 className="font-medium text-red-800 dark:text-red-400 mb-1">Pipeline Error</h5>
                <p className="text-sm text-red-600 dark:text-red-300 break-words">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Debug Information */}
        {showDebug && result && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Debug Information</h5>
              <pre className="text-xs overflow-x-auto bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-600">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}