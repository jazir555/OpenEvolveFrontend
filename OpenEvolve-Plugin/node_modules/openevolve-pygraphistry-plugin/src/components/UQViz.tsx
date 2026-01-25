import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { UQResult as IUQResult } from '../types/plugin-types';

interface Props {
  testFunction: string;
  nSamples?: number;
}

export const UQViz: React.FC<Props> = ({ 
  testFunction,
  nSamples = 500
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.uqEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<IUQResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runUQAnalysis = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/uq/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          function_name: testFunction.toLowerCase(),
          n_samples: nSamples
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'UQ analysis failed');
      }

      const data_result = await response.json();
      setResult(data_result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-blue-100 rounded-lg bg-blue-50/30 text-blue-400">
        <p className="font-medium italic">Uncertainty Quantification is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">UQ Analysis: {testFunction}</h3>
          <p className="text-xs text-gray-500">{nSamples} Monte Carlo samples</p>
        </div>
        <button
          onClick={runUQAnalysis}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Analyzing...' : 'Run UQ Pipeline'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Mean', value: result.statistics.mean },
              { label: 'Std Dev', value: result.statistics.std },
              { label: 'Min', value: result.statistics.min },
              { label: 'Max', value: result.statistics.max },
            ].map((stat) => (
              <div key={stat.label} className="p-3 bg-slate-50 rounded-md border text-center">
                <p className="text-[10px] uppercase text-slate-400 font-bold">{stat.label}</p>
                <p className="text-lg font-mono text-slate-700">{stat.value.toFixed(4)}</p>
              </div>
            ))}
          </div>

          {result.sensitivity && (
            <div className="border rounded-lg p-4 bg-gray-50">
              <h4 className="text-sm font-bold text-slate-600 mb-3 uppercase tracking-tight">Sobol Sensitivity Indices</h4>
              <div className="space-y-3">
                {result.sensitivity.first_order.map((val, i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span>Input X{i+1}</span>
                      <span>{(val * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="bg-blue-500 h-1.5 rounded-full" 
                        style={{ width: `${val * 100}%` }}
                      />
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
