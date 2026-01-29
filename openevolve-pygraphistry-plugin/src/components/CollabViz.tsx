import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { CollabSession } from '../types/plugin-types';

export const CollabViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.collaborationEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [sessions, setSessions] = useState<CollabSession[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/collaboration/sessions');
      if (!response.ok) {
        throw new Error('Failed to fetch collaboration sessions');
      }
      const data = await response.json();
      setSessions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchSessions();
      const interval = setInterval(fetchSessions, 5000); // 5s refresh
      return () => clearInterval(interval);
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-sky-100 rounded-lg bg-sky-50/30 text-sky-400">
        <p className="font-medium italic">Multi-Agent Collaboration visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-sky-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">H</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Collaboration Hub</h3>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] font-bold text-emerald-600 uppercase">Live Sync</span>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="space-y-3 animate-in fade-in slide-in-from-top-2">
        {sessions.map((sess) => (
          <div key={sess.session_id} className="p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all">
            <div className="flex justify-between items-start mb-3">
              <div>
                <h4 className="text-sm font-bold text-slate-800">{sess.name}</h4>
                <p className="text-[10px] text-slate-400 font-mono">{sess.session_id}</p>
              </div>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                sess.status === 'active' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-amber-50 text-amber-600 border-amber-100'
              }`}>
                {sess.status}
              </span>
            </div>

            <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-1">
              {sess.participants.map(p => (
                <div key={p} className="flex-none px-2 py-1 bg-white border rounded text-[10px] font-bold text-slate-600 flex items-center gap-1.5 shadow-sm">
                  <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
                  {p}
                </div>
              ))}
            </div>

            <div className="flex justify-between items-center pt-3 border-t border-slate-200/50">
              <div className="flex gap-4">
                <div className="text-center">
                  <p className="text-[8px] font-bold text-slate-400 uppercase">Conflicts</p>
                  <p className={`text-xs font-bold ${sess.conflict_count > 0 ? 'text-rose-500' : 'text-slate-600'}`}>{sess.conflict_count}</p>
                </div>
                <div className="text-center">
                  <p className="text-[8px] font-bold text-slate-400 uppercase">Last Activity</p>
                  <p className="text-xs font-bold text-slate-600">{sess.last_edit}</p>
                </div>
              </div>
              <button className="px-3 py-1 bg-sky-600 text-white text-[10px] font-bold rounded hover:bg-sky-700 transition-colors shadow-sm">
                JOIN SESSION
              </button>
            </div>
          </div>
        ))}

        {!loading && sessions.length === 0 && !error && (
          <div className="py-12 text-center text-slate-400 border-2 border-dashed rounded-lg">
            No active collaboration sessions found.
          </div>
        )}
      </div>
    </div>
  );
};
