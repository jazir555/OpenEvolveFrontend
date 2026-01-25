import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { DeepKEResult } from '../types/plugin-types';

interface Props {
  initialText?: string;
}

export const DeepKEViz: React.FC<Props> = ({ 
  initialText = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.deepkeEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DeepKEResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState(initialText);

  const runExtraction = async () => {
    if (!isEnabled || !text) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/knowledge/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error('DeepKE extraction failed');
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
        <p className="font-medium italic">Knowledge Extraction (DeepKE) visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-emerald-700 flex items-center justify-center text-white font-bold text-xs shadow-sm">K</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Structured Extraction (DeepKE)</h3>
        </div>
        <div className="flex flex-col gap-2">
          <textarea 
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text for entity/relation extraction..."
            className="w-full p-2 border rounded-md focus:ring-2 focus:ring-emerald-500 outline-none text-sm min-h-[80px]"
          />
          <div className="flex justify-end">
            <button
              onClick={runExtraction}
              disabled={loading || !text}
              className="px-4 py-2 bg-emerald-700 text-white rounded hover:bg-emerald-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
            >
              {loading ? 'Extracting...' : 'Run DeepKE'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="space-y-3">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Discovered Entities</h4>
            <div className="flex flex-wrap gap-2">
              {result.entities.map((e, i) => (
                <div key={i} className="flex flex-col items-center p-2 bg-slate-50 border rounded-lg shadow-sm min-w-[80px]">
                  <span className="text-[8px] font-bold text-emerald-600 uppercase tracking-tighter mb-1">{e.type}</span>
                  <span className="text-xs font-bold text-slate-800">{e.text}</span>
                  <div className="mt-1 w-full bg-slate-200 h-0.5 rounded-full overflow-hidden">
                    <div className="bg-emerald-500 h-full" style={{ width: `${e.confidence * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Semantic Relations</h4>
            <div className="space-y-2">
              {result.relations.map((r, i) => (
                <div key={i} className="p-2 border rounded bg-white flex items-center justify-between text-xs shadow-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-slate-700">{r.head}</span>
                    <span className="text-[10px] font-mono font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100 uppercase">
                      {r.relation}
                    </span>
                    <span className="font-bold text-slate-700">{r.tail}</span>
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">{(r.confidence * 100).toFixed(0)}% Match</span>
                </div>
              ))}
            </div>
          </div>

          {result.events.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Detected Events</h4>
              <div className="space-y-2">
                {result.events.map((ev, i) => (
                  <div key={i} className="p-3 border-l-4 border-l-amber-400 bg-amber-50/20 rounded-r-lg">
                    <p className="text-xs font-bold text-slate-800 capitalize">Trigger: {ev.trigger}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {ev.arguments.map(arg => (
                        <span key={arg} className="text-[9px] bg-white border border-amber-200 text-amber-700 px-1.5 rounded-full font-medium">
                          {arg}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
