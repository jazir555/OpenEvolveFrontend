import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { WorkflowMonitorResult } from '../types/plugin-types';

interface Props {
  workflowId: string;
}

export const WorkflowMonitorViz: React.FC<Props> = ({ 
  workflowId 
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.workflowMonitorEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<WorkflowMonitorResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMonitoringData = async () => {
    if (!isEnabled || !workflowId) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/openevolve/workflow/${workflowId}/monitor`);
      if (!response.ok) {
        throw new Error('Failed to fetch workflow monitoring data');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled && workflowId) {
      fetchMonitoringData();
      const interval = setInterval(fetchMonitoringData, 5000); // 5s refresh
      return () => clearInterval(interval);
    }
  }, [isEnabled, workflowId]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Workflow Execution Monitor is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">W</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Workflow execution monitor</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase">ID: {workflowId}</span>
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase">Status</p>
              <p className="text-sm font-bold text-blue-600 mt-1 uppercase">{result.status}</p>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase">Progress</p>
              <p className="text-lg font-bold text-slate-800 mt-1">{(result.progress * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase">Runtime</p>
              <p className="text-lg font-bold text-slate-800 mt-1">{result.execution_time.toFixed(1)}s</p>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase">Current Stage</p>
              <p className="text-sm font-bold text-slate-700 mt-1">{result.current_stage}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Performance Metrics</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 border rounded-lg bg-white shadow-sm">
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Best Fitness</p>
                  <p className="text-xl font-mono font-bold text-emerald-600">{result.metrics.best_fitness.toFixed(4)}</p>
                </div>
                <div className="p-3 border rounded-lg bg-white shadow-sm">
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Avg Fitness</p>
                  <p className="text-xl font-mono font-bold text-slate-700">{result.metrics.avg_fitness.toFixed(4)}</p>
                </div>
                <div className="p-3 border rounded-lg bg-white shadow-sm">
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Diversity</p>
                  <p className="text-xl font-mono font-bold text-indigo-600">{result.metrics.diversity.toFixed(4)}</p>
                </div>
                <div className="p-3 border rounded-lg bg-white shadow-sm">
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Population</p>
                  <p className="text-xl font-mono font-bold text-slate-800">{result.metrics.population_size}</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Resource Utilization</h4>
              <div className="space-y-3 p-4 bg-slate-50 rounded-xl border border-slate-100">
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-500 uppercase">Memory Usage</span>
                    <span className="text-slate-700">{result.resource_usage.memory_mb} MB</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div className="bg-blue-500 h-full" style={{ width: `${Math.min(100, (result.resource_usage.memory_mb / 4096) * 100)}%` }} />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-500 uppercase">CPU Load</span>
                    <span className="text-slate-700">{(result.resource_usage.cpu_cores * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div className="bg-indigo-500 h-full" style={{ width: `${Math.min(100, result.resource_usage.cpu_cores * 100)}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Live Event Log</h4>
            <div className="bg-slate-900 rounded-lg p-3 space-y-2 max-h-[150px] overflow-y-auto">
              {result.events.map((event, i) => (
                <div key={i} className="flex gap-3 text-[11px] font-mono">
                  <span className="text-slate-500">[{event.timestamp}]</span>
                  <span className={`font-bold ${event.status === 'error' ? 'text-rose-400' : 'text-emerald-400'}`}>{event.status.toUpperCase()}</span>
                  <span className="text-indigo-200">{event.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
