import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ArtifactResult } from '../types/plugin-types';

export const ArtifactViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.artifactGraphEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ArtifactResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/artifacts/graph');
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch artifact graph');
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
      fetchGraph();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Knowledge Artifact Mapping visualization is currently disabled in settings.</p>
      </div>
    );
  }

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'solution_pattern': return 'bg-amber-100 text-amber-700 border-amber-200';
      case 'team_performance': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'gauntlet_effectiveness': return 'bg-rose-100 text-rose-700 border-rose-200';
      default: return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Knowledge Artifact Graph</h3>
          <p className="text-xs text-slate-500">Relationships between solution patterns, teams, and effectiveness.</p>
        </div>
        <button 
          onClick={fetchGraph}
          disabled={loading}
          className="text-xs font-bold text-amber-600 hover:underline"
        >
          {loading ? 'Analyzing...' : 'Refresh Graph'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-100 min-h-[300px] flex flex-col items-center justify-center text-center">
              {/* Abstract visualization of the graph nodes */}
              <div className="flex flex-wrap justify-center gap-4 max-w-md">
                {result.nodes.map((node) => (
                  <div key={node.id} className={`px-3 py-2 rounded-lg border shadow-sm ${getTypeColor(node.type)}`}>
                    <p className="text-[10px] font-bold uppercase opacity-60 tracking-wider">{node.type.replace('_', ' ')}</p>
                    <p className="text-xs font-bold">{node.label}</p>
                  </div>
                ))}
              </div>
              <div className="mt-8">
                <span className="text-[10px] text-slate-400 uppercase font-bold tracking-widest">Graph Link Summary</span>
                <div className="flex gap-4 mt-2">
                  <span className="text-xs text-slate-500">Nodes: <strong>{result.nodes.length}</strong></span>
                  <span className="text-xs text-slate-500">Relationships: <strong>{result.edges.length}</strong></span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Extracted Facts</h4>
              <div className="space-y-2 overflow-y-auto max-h-[300px] pr-2">
                {result.edges.map((edge, i) => (
                  <div key={i} className="p-2 border rounded bg-white flex flex-col gap-1 shadow-sm">
                    <div className="flex justify-between items-center text-[10px] text-slate-400 font-bold uppercase">
                      <span>{edge.source}</span>
                      <span>{edge.label}</span>
                    </div>
                    <p className="text-xs text-slate-700 font-medium text-center">{edge.target}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
