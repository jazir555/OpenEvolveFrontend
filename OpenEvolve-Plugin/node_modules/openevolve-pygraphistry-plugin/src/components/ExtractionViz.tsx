import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ExtractionResult as IExtractionResult } from '../types/plugin-types';

interface Props {
  initialText?: string;
}

export const ExtractionViz: React.FC<Props> = ({ 
  initialText = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.onekeEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<IExtractionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState(initialText);

  const runExtraction = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/oneke/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Extraction failed');
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
      <div className="p-6 text-center border-2 border-dashed border-orange-100 rounded-lg bg-orange-50/30 text-orange-400">
        <p className="font-medium italic">Knowledge Extraction is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-800">Schema-Guided Knowledge Extraction</h3>
        <textarea 
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste text to extract structured knowledge..."
          className="w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-orange-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runExtraction}
            disabled={loading || !text}
            className="px-6 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 disabled:opacity-50 transition-colors font-medium"
          >
            {loading ? 'Extracting...' : 'Extract Entities & Relations'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2">
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Detected Entities</h4>
            <div className="flex flex-wrap gap-2">
              {result.entities.map((entity, i) => (
                <div key={i} className="px-3 py-1.5 bg-orange-50 border border-orange-100 rounded-full flex items-center gap-2 shadow-sm">
                  <span className="text-sm font-semibold text-orange-800">{entity.text}</span>
                  <span className="text-[10px] bg-orange-200 text-orange-700 px-1.5 rounded-md font-bold uppercase">{entity.type}</span>
                </div>
              ))}
              {result.entities.length === 0 && <p className="text-sm text-slate-400 italic">No entities found.</p>}
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Extracted Relations</h4>
            <div className="space-y-2">
              {result.relations.map((rel, i) => (
                <div key={i} className="p-3 bg-slate-50 border rounded-lg flex items-center justify-between group hover:bg-white hover:border-orange-200 transition-all shadow-sm">
                  <span className="text-sm font-bold text-slate-700">{rel.subject}</span>
                  <span className="text-[10px] font-mono text-orange-500 font-bold uppercase bg-white border px-2 py-0.5 rounded-full mx-2">{rel.predicate}</span>
                  <span className="text-sm font-bold text-slate-700">{rel.object}</span>
                </div>
              ))}
              {result.relations.length === 0 && <p className="text-sm text-slate-400 italic">No relations found.</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
