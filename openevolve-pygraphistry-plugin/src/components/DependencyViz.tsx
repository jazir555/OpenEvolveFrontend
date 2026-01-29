import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { DependencyResult } from '../types/plugin-types';

interface Props {
  subProblems: any[];
}

export const DependencyViz: React.FC<Props> = ({ 
  subProblems 
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.dependencyEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DependencyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/dependencies/graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sub_problems: subProblems })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch dependency graph');
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
    if (isEnabled && subProblems.length > 0) {
      fetchGraph();
    }
  }, [isEnabled, subProblems]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Dependency Mapping visualization is currently disabled in settings.</p>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'solved': return 'bg-emerald-500';
      case 'in_progress': return 'bg-blue-500';
      case 'pending': return 'bg-amber-500';
      case 'failed': return 'bg-rose-500';
      default: return 'bg-slate-400';
    }
  };

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Workflow Dependency Map</h3>
          <p className="text-xs text-slate-500">Visualization of sub-problem execution order and complexity.</p>
        </div>
        <button 
          onClick={fetchGraph}
          disabled={loading}
          className="text-xs font-bold text-indigo-600 hover:underline"
        >
          {loading ? 'Refreshing...' : 'Refresh Map'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          {/* Simple relative layout visualization */}
          <div className="p-8 bg-slate-50 rounded-xl border border-slate-100 overflow-x-auto">
            <div className="flex items-center gap-8 min-w-max">
              {result.nodes.map((node, i) => (
                <React.Fragment key={node.id}>
                  <div className="flex flex-col items-center gap-2">
                    <div className={`w-16 h-16 rounded-2xl shadow-md flex items-center justify-center text-white font-bold text-xs border-4 border-white ${getStatusColor(node.status)}`}>
                      {node.id}
                    </div>
                    <span className="text-[10px] font-bold text-slate-500 uppercase">{node.status}</span>
                    <span className="text-[8px] px-1.5 py-0.5 bg-white border rounded text-slate-400">Comp: {node.complexity}</span>
                  </div>
                  {i < result.nodes.length - 1 && (
                    <div className="w-12 h-0.5 bg-slate-200 relative">
                      <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 rotate-45 border-t-2 border-r-2 border-slate-300" />
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Logic Dependencies</h4>
              <div className="space-y-1">
                {result.edges.map((edge, i) => (
                  <div key={i} className="p-2 border rounded bg-slate-50/50 flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-600">{edge.source}</span>
                    <span className="text-slate-300">→</span>
                    <span className="font-bold text-slate-800">{edge.target}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="p-4 bg-indigo-50/30 rounded-lg border border-indigo-100">
              <h4 className="text-[10px] font-bold text-indigo-600 uppercase tracking-widest mb-2">Analysis Note</h4>
              <p className="text-[11px] text-indigo-800/80 leading-relaxed italic">
                This DAG (Directed Acyclic Graph) ensures that prerequisites are satisfied before higher-level integration tasks begin. 
                Nodes are sized by relative complexity and colored by execution status.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
