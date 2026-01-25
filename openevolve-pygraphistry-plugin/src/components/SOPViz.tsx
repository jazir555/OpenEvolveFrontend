import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { SOP } from '../types/plugin-types';

interface Props {
  initialRequirement?: string;
  domain?: string;
}

export const SOPViz: React.FC<Props> = ({ 
  initialRequirement = '',
  domain = 'general'
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.sopEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [sop, setSop] = useState<SOP | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requirement, setRequirement] = useState(initialRequirement);

  const generateSOP = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/sop/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ requirement, domain })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'SOP generation failed');
      }

      const data = await response.json();
      setSop(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">SOP Generation is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-800">SOP Generator & Refiner</h3>
        <p className="text-xs text-slate-500">Generate turnkey-ready operating procedures from high-level goals.</p>
        <textarea 
          value={requirement}
          onChange={(e) => setRequirement(e.target.value)}
          placeholder="Describe the process or requirement (e.g., 'Protocol for high-speed centrifugation of plasma samples')..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-slate-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={generateSOP}
            disabled={loading || !requirement}
            className="px-6 py-2 bg-slate-800 text-white rounded hover:bg-slate-900 disabled:opacity-50 transition-colors font-medium shadow-sm"
          >
            {loading ? 'Generating SOP...' : 'Generate Protocol'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {sop && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-xl font-bold text-slate-900">{sop.title}</h2>
              <div className="flex gap-2 mt-1">
                <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded border font-bold uppercase tracking-wider text-slate-600">v{sop.version}</span>
                <span className="text-[10px] bg-blue-50 px-2 py-0.5 rounded border border-blue-100 font-bold uppercase tracking-wider text-blue-600">{sop.status}</span>
              </div>
            </div>
          </div>

          <p className="text-sm text-slate-600 leading-relaxed italic border-l-4 border-slate-200 pl-3">
            {sop.description}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Required Equipment</h4>
              <div className="flex flex-wrap gap-2">
                {sop.equipment.map((eq, i) => (
                  <span key={i} className="px-2 py-1 bg-slate-50 border rounded text-xs font-medium text-slate-700">
                    🛠️ {typeof eq === 'string' ? eq : (eq.name || 'Unknown Device')}
                  </span>
                ))}
              </div>
            </div>
            
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Execution Steps</h4>
              <div className="space-y-2">
                {sop.protocols.map((step) => (
                  <div key={step.step_number} className="flex gap-3 items-start p-2 bg-slate-50/50 rounded border border-slate-100">
                    <span className="flex-none w-5 h-5 rounded-full bg-slate-800 text-white flex items-center justify-center text-[10px] font-bold">
                      {step.step_number}
                    </span>
                    <p className="text-xs text-slate-700 font-medium">{step.action}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {(sop.safety_protocols || sop.quality_control) && (
            <div className="p-4 bg-amber-50/50 rounded-lg border border-amber-100">
              <h4 className="text-xs font-bold text-amber-800 uppercase tracking-widest mb-2">Safety & Quality Assurance</h4>
              <ul className="list-disc list-inside space-y-1">
                {(sop.safety_protocols || []).map((s, i) => (
                  <li key={`s-${i}`} className="text-xs text-amber-900/80">{s}</li>
                ))}
                {(sop.quality_control || []).map((q, i) => (
                  <li key={`q-${i}`} className="text-xs text-slate-700">{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
