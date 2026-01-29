import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ExperimentProtocol as IExperimentProtocol } from '../types/plugin-types';

interface Props {
  initialHypothesis?: string;
  domain?: string;
}

export const ExperimentViz: React.FC<Props> = ({ 
  initialHypothesis = '',
  domain = 'physics'
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.curieEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [protocol, setProtocol] = useState<IExperimentProtocol | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hypothesis, setHypothesis] = useState(initialHypothesis);

  const designExperiment = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/curie/design', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis,
          domain
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Experiment design failed');
      }

      const data = await response.json();
      setProtocol(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-amber-100 rounded-lg bg-amber-50/30 text-amber-400">
        <p className="font-medium italic">Curie Experimentation is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-800">Scientific Experiment Designer</h3>
        <textarea 
          value={hypothesis}
          onChange={(e) => setHypothesis(e.target.value)}
          placeholder="Enter your scientific hypothesis here..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-amber-500 outline-none text-sm"
        />
        <div className="flex justify-end">
          <button
            onClick={designExperiment}
            disabled={loading || !hypothesis}
            className="px-6 py-2 bg-amber-600 text-white rounded hover:bg-amber-700 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Designing Protocol...' : 'Design Experiment'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {protocol && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
          <div className="p-4 bg-amber-50 rounded-lg border border-amber-100">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] font-bold text-amber-700 uppercase tracking-widest">Protocol Generated</span>
              <span className="text-xs font-mono text-amber-600">{protocol.protocol_id}</span>
            </div>
            <h4 className="font-bold text-slate-800">Hypothesis Components</h4>
            <div className="grid grid-cols-2 gap-4 mt-2">
              <div>
                <p className="text-[10px] text-slate-400 uppercase font-bold">Independent</p>
                <p className="text-sm text-slate-700">{protocol.hypothesis.independent_variables.join(', ') || 'N/A'}</p>
              </div>
              <div>
                <p className="text-[10px] text-slate-400 uppercase font-bold">Dependent</p>
                <p className="text-sm text-slate-700">{protocol.hypothesis.dependent_variables.join(', ') || 'N/A'}</p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-bold text-slate-600 uppercase tracking-tight">Execution Workflow</h4>
            <div className="space-y-2">
              {protocol.steps.map((step, i) => (
                <div key={i} className="flex gap-3 items-start">
                  <div className="flex-none w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-xs font-bold text-slate-500 border">
                    {i + 1}
                  </div>
                  <div className="flex-1 p-2 bg-white border rounded shadow-sm text-sm text-slate-700">
                    {step.description}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {protocol.equipment.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {protocol.equipment.map((item) => (
                <span key={item} className="px-2 py-1 bg-slate-100 text-slate-600 rounded text-[10px] font-medium border">
                  🛠️ {item}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
