import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { CausalDiscoveryResult } from '../types/plugin-types';

interface Props {
  data: number[][];
  variableNames: string[];
  height?: string | number;
}

export const CausalDiscoveryViz: React.FC<Props> = ({
  data,
  variableNames,
  height = 500
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.causalDiscoveryEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CausalDiscoveryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const discoverCausalStructure = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/causal/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data,
          variable_names: variableNames,
          method: 'pc'
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Causal discovery failed');
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
      <div className="p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400">
        <p className="font-medium italic">Causal Discovery visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-800">Causal Discovery Analysis</h3>
        <button
          onClick={discoverCausalStructure}
          disabled={loading}
          className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Analyzing...' : 'Run Causal Discovery'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded">
          {error}
        </div>
      )}

      {!result && !loading && (
        <div className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-gray-200 rounded-lg text-gray-400">
          <p>Click "Run Causal Discovery" to analyze variables:</p>
          <div className="flex flex-wrap gap-2 mt-2">
            {variableNames.map(name => (
              <span key={name} className="px-2 py-1 bg-gray-100 rounded text-xs">{name}</span>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded p-4 bg-gray-50">
            <h4 className="font-medium mb-2">Discovered Relationships</h4>
            <ul className="space-y-1">
              {result.edges.map((edge: any, i: number) => (
                <li key={i} className="text-sm flex items-center">
                  <span className="font-bold text-indigo-600">{result.nodes[edge[0]]}</span>
                  <span className="mx-2">→</span>
                  <span className="font-bold text-teal-600">{result.nodes[edge[1]]}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="border rounded p-4 bg-gray-50">
            <h4 className="font-medium mb-2">Algorithm Insights</h4>
            <p className="text-sm text-gray-600">
              Method: <span className="font-mono bg-white px-1 border rounded">{result.algorithm}</span>
            </p>
            <p className="text-sm text-gray-600 mt-2">
              Detected {result.edges.length} causal pathways across {result.nodes.length} variables.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
