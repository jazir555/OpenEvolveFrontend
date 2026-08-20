/**
 * KnowledgeBase
 * Top-level Knowledge Base interface for the BubbleLab React app.
 *
 * Browses and searches knowledge artifacts extracted by the OpenEvolve
 * workflow, shows a single artifact's detail, renders the extracted knowledge
 * graph, and surfaces aggregate statistics.
 *
 * Data is fetched through the `openevolveApi` knowledge methods:
 *  - `listKnowledgeArtifacts` / `searchKnowledge` for the artifact list
 *  - `getKnowledgeArtifact` for a single artifact's detail
 *  - `getKnowledgeGraph` for the graph view
 *  - `getKnowledgeStats` for the stats summary
 */

import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  KnowledgeArtifact,
  KnowledgeGraph,
  KnowledgeStats,
} from '@/types/openevolve';
import { ArtifactList } from './ArtifactList';
import { ArtifactDetail } from './ArtifactDetail';
import { KnowledgeGraphView } from './KnowledgeGraphView';
import { KnowledgeStatsView } from './KnowledgeStatsView';

type TabId = 'artifacts' | 'graph' | 'stats';

const TABS: { id: TabId; label: string }[] = [
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'graph', label: 'Graph' },
  { id: 'stats', label: 'Stats' },
];

export function KnowledgeBase() {
  const [activeTab, setActiveTab] = useState<TabId>('artifacts');
  const [query, setQuery] = useState('');

  const [artifacts, setArtifacts] = useState<KnowledgeArtifact[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [selected, setSelected] = useState<KnowledgeArtifact | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  const runSearch = useCallback(async (searchQuery: string) => {
    setListLoading(true);
    setListError(null);
    setSelected(null);
    try {
      const trimmed = searchQuery.trim();
      if (trimmed.length === 0) {
        const res = await openevolveApi.listKnowledgeArtifacts();
        setArtifacts(res.artifacts ?? []);
      } else {
        const res = await openevolveApi.searchKnowledge({ query: trimmed });
        setArtifacts(res.results ?? []);
      }
    } catch (err) {
      setListError(err instanceof Error ? err.message : 'Failed to load artifacts.');
      setArtifacts([]);
    } finally {
      setListLoading(false);
    }
  }, []);

  // Initial load of the artifact list.
  useEffect(() => {
    void runSearch('');
  }, [runSearch]);

  const handleSelect = useCallback(async (artifact: KnowledgeArtifact) => {
    setSelected(artifact);
    setDetailLoading(true);
    setDetailError(null);
    try {
      const detail = await openevolveApi.getKnowledgeArtifact(artifact.id);
      setSelected(detail);
    } catch (err) {
      setDetailError(
        err instanceof Error ? err.message : 'Failed to load artifact detail.'
      );
    } finally {
      setDetailLoading(false);
    }
  }, []);

  // Lazy load the graph when its tab is first opened.
  useEffect(() => {
    if (activeTab !== 'graph' || graph || graphLoading) return;
    let cancelled = false;
    setGraphLoading(true);
    setGraphError(null);
    openevolveApi
      .getKnowledgeGraph()
      .then((res) => {
        if (!cancelled) setGraph(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setGraphError(
            err instanceof Error ? err.message : 'Failed to load knowledge graph.'
          );
        }
      })
      .finally(() => {
        if (!cancelled) setGraphLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, graph, graphLoading]);

  // Lazy load stats when its tab is first opened.
  useEffect(() => {
    if (activeTab !== 'stats' || stats || statsLoading) return;
    let cancelled = false;
    setStatsLoading(true);
    setStatsError(null);
    openevolveApi
      .getKnowledgeStats()
      .then((res) => {
        if (!cancelled) setStats(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setStatsError(
            err instanceof Error ? err.message : 'Failed to load knowledge stats.'
          );
        }
      })
      .finally(() => {
        if (!cancelled) setStatsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, stats, statsLoading]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void runSearch(query);
  };

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Knowledge Base
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Browse, search and explore knowledge artifacts extracted by OpenEvolve.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <div className="relative flex-1">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
            <svg
              className="h-4 w-4 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search knowledge (title, type, domain)…"
            className="w-full rounded-md border border-gray-300 bg-white py-2 pl-10 pr-3 text-sm text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          />
        </div>
        <button
          type="submit"
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Search
        </button>
        <button
          type="button"
          onClick={() => {
            setQuery('');
            void runSearch('');
          }}
          className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
        >
          Reset
        </button>
      </form>

      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8" aria-label="Knowledge tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`whitespace-nowrap border-b-2 py-3 px-1 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'artifacts' && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="thin-scrollbar max-h-[70vh] overflow-auto rounded-lg border border-gray-200 bg-white p-2 shadow-sm dark:border-gray-700 dark:bg-gray-800">
            {listError ? (
              <div className="m-2 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
                {listError}
              </div>
            ) : (
              <ArtifactList
                artifacts={artifacts}
                selectedId={selected?.id ?? null}
                onSelect={handleSelect}
                loading={listLoading}
              />
            )}
          </div>
          <div className="thin-scrollbar max-h-[70vh] overflow-auto rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
            <ArtifactDetail
              artifact={selected}
              loading={detailLoading}
              error={detailError}
            />
          </div>
        </div>
      )}

      {activeTab === 'graph' && (
        <div>
          {graphError && (
            <div className="mb-3 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
              {graphError}
            </div>
          )}
          {graphLoading && !graph ? (
            <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
              Loading knowledge graph…
            </div>
          ) : graph ? (
            <KnowledgeGraphView graph={graph} />
          ) : null}
        </div>
      )}

      {activeTab === 'stats' && (
        <div>
          {statsError && (
            <div className="mb-3 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
              {statsError}
            </div>
          )}
          {statsLoading && !stats ? (
            <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
              Loading statistics…
            </div>
          ) : stats ? (
            <KnowledgeStatsView stats={stats} />
          ) : null}
        </div>
      )}
    </div>
  );
}
