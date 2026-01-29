import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { HephaestusResult } from '../types/plugin-types';

export const HephaestusViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.hephaestusEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<HephaestusResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/hephaestus/summary');
      if (!response.ok) {
        throw new Error('Failed to fetch Hephaestus summary');
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
      fetchSummary();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Project Management visualization is currently disabled in settings.</p>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status.toUpperCase()) {
      case 'DONE': return 'bg-emerald-100 text-emerald-700 border-emerald-200';
      case 'IN_PROGRESS': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'TODO': return 'bg-slate-100 text-slate-700 border-slate-200';
      case 'CANCELLED': return 'bg-rose-100 text-rose-700 border-rose-200';
      default: return 'bg-slate-50 text-slate-500';
    }
  };

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-white font-bold text-xs shadow-sm">H</div>
          <h3 className="text-lg font-bold text-slate-800">Hephaestus Project Tracking</h3>
        </div>
        <button 
          onClick={fetchSummary}
          disabled={loading}
          className="text-xs font-bold text-blue-600 hover:text-blue-700 transition-colors"
        >
          {loading ? 'Syncing...' : 'Sync Now'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-top-2">
          {/* Status Overview */}
          <div className="lg:col-span-1 space-y-4">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Ticket Overview</h4>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-4">
              <div className="text-center pb-4 border-b">
                <p className="text-3xl font-bold text-slate-800">{result.total_tickets}</p>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Active Mappings</p>
              </div>
              <div className="space-y-3">
                {Object.entries(result.status_distribution).map(([status, count]) => (
                  <div key={status} className="flex justify-between items-center">
                    <span className="text-xs font-medium text-slate-600">{status}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${getStatusColor(status)}`}>
                      {count}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Activity Feed */}
          <div className="lg:col-span-2 space-y-4">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Recent Activity</h4>
            <div className="space-y-3">
              {result.recent_activity.map((ticket) => (
                <div key={ticket.id} className="p-3 border rounded-lg bg-white hover:shadow-md transition-all flex justify-between items-center group">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono font-bold text-slate-400 group-hover:text-blue-600 transition-colors">{ticket.id}</span>
                    <p className="text-sm font-medium text-slate-700">{ticket.task}</p>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-1 rounded border ${getStatusColor(ticket.status)}`}>
                    {ticket.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
