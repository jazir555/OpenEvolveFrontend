/**
 * KnowledgeStatsView
 * Summary of the knowledge base extracted by the OpenEvolve workflow.
 *
 * Renders the aggregate statistics returned by `getKnowledgeStats` plus a
 * per-type distribution so users can see what kind of artifacts have been
 * extracted.
 */

import type { KnowledgeStats } from '@/types/openevolve';

interface KnowledgeStatsViewProps {
  stats: KnowledgeStats;
}

const TYPE_COLORS = [
  'bg-blue-600',
  'bg-emerald-600',
  'bg-violet-600',
  'bg-amber-500',
  'bg-rose-600',
  'bg-cyan-600',
  'bg-fuchsia-600',
];

export function KnowledgeStatsView({ stats }: KnowledgeStatsViewProps) {
  const typeEntries = Object.entries(stats.by_type ?? {}).sort(
    (a, b) => b[1] - a[1]
  );
  const maxType = Math.max(1, ...typeEntries.map(([, v]) => v));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Total Artifacts"
          value={stats.total_artifacts}
        />
        <StatCard label="Total Usage" value={stats.total_usage} />
        <StatCard
          label="Avg. Effectiveness"
          value={stats.average_effectiveness.toFixed(2)}
        />
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h3 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
          Artifacts by Type
        </h3>
        {typeEntries.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No artifacts recorded yet.
          </p>
        ) : (
          <div className="space-y-3">
            {typeEntries.map(([type, count], index) => (
              <div key={type} className="flex items-center gap-3">
                <span className="w-40 truncate text-sm text-gray-600 dark:text-gray-300">
                  {type}
                </span>
                <div className="h-4 flex-1 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className={`h-full transition-all duration-500 ${
                      TYPE_COLORS[index % TYPE_COLORS.length]
                    }`}
                    style={{ width: `${(count / maxType) * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right text-sm font-medium text-gray-900 dark:text-white">
                  {count}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-gray-900 dark:text-white">
        {value}
      </p>
    </div>
  );
}
