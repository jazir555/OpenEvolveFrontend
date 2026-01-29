import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { SGDResult } from '../types/plugin-types';

export const SGDViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.sgdEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<SGDResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSGDMetrics = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/sgd/monitoring');
      if (!response.ok) {
        throw new Error('Failed to fetch SGD metrics');
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
    if (isEnabled) {
      fetchSGDMetrics();
      const interval = setInterval(fetchSGDMetrics, 10000); // Auto-refresh every 10s
      return () => clearInterval(interval);
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">SGD Workflow Monitoring is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white font-bold text-xs shadow-md">S</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Sovereign-Grade Workflow Monitor</h3>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] font-bold text-emerald-600 uppercase">Live Metrics</span>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active Workflows</p>
              <p className="text-2xl font-bold text-blue-600 mt-1">{result.active_workflows}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Success Rate</p>
              <p className="text-2xl font-bold text-emerald-600 mt-1">{(result.success_rate * 100).toFixed(1)}%</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Open Tickets</p>
              <p className="text-2xl font-bold text-indigo-600 mt-1">{result.active_tickets}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Gauntlet Runs</p>
              <p className="text-2xl font-bold text-slate-800 mt-1">{result.total_gauntlet_runs}</p>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Pipeline Throughput</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3 border rounded-lg bg-slate-50/50">
                <div className="flex justify-between text-xs font-medium mb-2">
                  <span className="text-slate-500">Completed vs Failed Workflows</span>
                  <span className="text-slate-700 font-bold">{result.completed_workflows} / {result.failed_workflows}</span>
                </div>
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden flex">
                  <div className="bg-emerald-500 h-full" style={{ width: `${(result.completed_workflows / (result.completed_workflows + result.failed_workflows + 0.1)) * 100}%` }} />
                  <div className="bg-rose-500 h-full" style={{ width: `${(result.failed_workflows / (result.completed_workflows + result.failed_workflows + 0.1)) * 100}%` }} />
                </div>
              </div>
              <div className="p-3 border rounded-lg bg-slate-50/50">
                <div className="flex justify-between text-xs font-medium mb-2">
                  <span className="text-slate-500">Gauntlet Pass Rate</span>
                  <span className="text-slate-700 font-bold">{result.successful_gauntlet_runs} successful</span>
                </div>
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div className="bg-indigo-500 h-full" style={{ width: `${(result.successful_gauntlet_runs / result.total_gauntlet_runs) * 100}%` }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
