import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ChemicalEntry } from '../types/plugin-types';

interface Props {
  initialList?: string;
}

export const ChemicalViz: React.FC<Props> = ({ 
  initialList = 'fda_approved'
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.globalChemEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [chemicals, setChemicals] = useState<ChemicalEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');

  const searchChemicals = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/chem/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          list_name: initialList
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Chemical search failed');
      }

      const data = await response.json();
      setChemicals(data.chemicals);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-teal-100 rounded-lg bg-teal-50/30 text-teal-400">
        <p className="font-medium italic">Chemical Knowledge Explorer is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex flex-col md:flex-row md:items-end gap-4">
        <div className="flex-1 space-y-1">
          <label className="text-xs font-bold text-slate-400 uppercase">Search Molecules</label>
          <input 
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && searchChemicals()}
            placeholder="Search by name or SMILES..."
            className="w-full p-2 border rounded-md focus:ring-2 focus:ring-teal-500 outline-none"
          />
        </div>
        <button
          onClick={searchChemicals}
          disabled={loading}
          className="px-6 py-2 bg-teal-600 text-white rounded hover:bg-teal-700 disabled:opacity-50 transition-colors h-[42px]"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {chemicals.map((chem, i) => (
          <div key={i} className="p-4 border rounded-lg hover:border-teal-300 transition-colors bg-slate-50 group">
            <div className="flex justify-between items-start mb-2">
              <h4 className="font-bold text-slate-800 truncate" title={chem.name}>{chem.name}</h4>
              <span className="text-[10px] px-1.5 py-0.5 bg-teal-100 text-teal-700 rounded font-medium">
                {chem.list || 'GlobalChem'}
              </span>
            </div>
            <div className="bg-white p-2 rounded border mb-2 overflow-hidden h-32 flex items-center justify-center">
              {/* Placeholder for SMILES rendering - in real app would use a canvas component */}
              <p className="text-[10px] font-mono text-slate-400 break-all leading-tight">
                {chem.smiles}
              </p>
            </div>
            <div className="flex justify-between items-center mt-auto">
              <button className="text-[10px] text-teal-600 font-bold uppercase group-hover:underline">
                View Details
              </button>
              <span className="text-[10px] text-slate-400 font-mono">
                {chem.molecular_weight ? `${chem.molecular_weight.toFixed(1)} g/mol` : ''}
              </span>
            </div>
          </div>
        ))}
        
        {!loading && chemicals.length === 0 && !error && (
          <div className="col-span-full py-12 text-center text-slate-400 border-2 border-dashed rounded-lg">
            <p>No chemical data loaded. Try searching for "Aspirin" or "Caffeine".</p>
          </div>
        )}
      </div>
    </div>
  );
};
