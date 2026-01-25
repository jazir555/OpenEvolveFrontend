import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { KGResult } from '../types/plugin-types';

interface Props {
  initialText?: string;
}

export const KGViz: React.FC<Props> = ({ 
  initialText = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.kgEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<KGResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState(initialText);

  const runKGGeneration = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/kg/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'KG generation failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400">
        <p className="font-medium italic">Knowledge Graph Generation is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-emerald-700 flex items-center justify-center text-white font-bold text-xs shadow-sm">KG</div>
          <h3 className="text-lg font-bold text-slate-800">Knowledge Graph Generation (KG-GEN)</h3>
        </div>
        <p className="text-xs text-slate-500">Transform unstructured text into a structured knowledge graph.</p>
        <textarea 
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to build a graph from..."
          className="w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-emerald-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runKGGeneration}
            disabled={loading || !text}
            className="px-6 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Building Graph...' : 'Generate Graph'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-emerald-700 uppercase tracking-widest bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100">
              Graph Entities & Relations
            </span>
            <span className="text-[10px] text-slate-400 font-mono">{result.nodes.length} Nodes, {result.edges.length} Edges</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Entities</h4>
              <div className="flex flex-wrap gap-2">
                {result.nodes.map((node, i) => (
                  <span key={i} className="px-2 py-1 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded text-xs font-medium">
                    {node.label}
                  </span>
                ))}
              </div>
            </div>
            
            <div className="space-y-2">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Relations</h4>
              <div className="space-y-1">
                {result.edges.map((edge, i) => (
                  <div key={i} className="p-2 bg-slate-50 border rounded text-xs flex justify-between items-center">
                    <span className="font-bold text-slate-700">{edge.source}</span>
                    <span className="text-[10px] font-mono text-emerald-600 font-bold px-2">{edge.label}</span>
                    <span className="font-bold text-slate-700">{edge.target}</span>
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
