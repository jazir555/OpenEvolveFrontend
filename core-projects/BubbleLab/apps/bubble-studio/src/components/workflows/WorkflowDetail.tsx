/**
 * WorkflowDetail
 *
 * Fetches and renders the full detail of a decomposition workflow plus its
 * results. Consumes the canonical OpenEvolve client surface
 * (`getWorkflowDetail`, `getWorkflowResults`, `pauseWorkflow`,
 * `resumeWorkflow`, `deleteWorkflow`).
 */

import { useCallback, useEffect, useState } from 'react';
import { openevolveApi } from '@/services/openevolveApi';
import type { WorkflowDetail as WorkflowDetailData, WorkflowResults } from '@/types/openevolve';
import { StatusBadge } from '../common/StatusBadge';

interface WorkflowDetailPanelProps {
  workflowId: string;
  onClose: () => void;
  onStatusChanged: () => void;
  onDeleted: () => void;
}

function normalizeTs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const ms = value > 1e12 ? value : value * 1000;
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString();
}

function ProgressBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round((Number.isFinite(value) ? value : 0) * 100)));
  return (
    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
      <div
        className="bg-blue-600 h-2.5 rounded-full transition-all"
        style={{ width: `${pct}%` }}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}

function SolutionBlock({
  title,
  content,
  generatedBy,
  timestamp,
}: {
  title: string;
  content: string;
  generatedBy?: string;
  timestamp?: string;
}) {
  return (
    <div className="bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h4>
        {generatedBy && (
          <span className="text-xs text-gray-500 dark:text-gray-400">{generatedBy}</span>
        )}
      </div>
      <pre className="whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300 font-mono leading-relaxed">
        {content}
      </pre>
      {timestamp && (
        <p className="mt-2 text-xs text-gray-400 dark:text-gray-500">{timestamp}</p>
      )}
    </div>
  );
}

export function WorkflowDetailPanel({
  workflowId,
  onClose,
  onStatusChanged,
  onDeleted,
}: WorkflowDetailPanelProps) {
  const [detail, setDetail] = useState<WorkflowDetailData | null>(null);
  const [results, setResults] = useState<WorkflowResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [detailRes, resultsRes] = await Promise.all([
        openevolveApi.getWorkflowDetail(workflowId),
        openevolveApi.getWorkflowResults(workflowId),
      ]);
      setDetail(detailRes);
      setResults(resultsRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workflow detail');
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    void load();
  }, [load]);

  const runAction = useCallback(
    async (label: string, fn: () => Promise<unknown>) => {
      setActionLoading(label);
      setError(null);
      try {
        await fn();
        onStatusChanged();
        await load();
      } catch (err) {
        setError(err instanceof Error ? err.message : `Failed to ${label}`);
      } finally {
        setActionLoading(null);
      }
    },
    [load, onStatusChanged]
  );

  const handlePause = () => runAction('pause', () => openevolveApi.pauseWorkflow(workflowId));
  const handleResume = () => runAction('resume', () => openevolveApi.resumeWorkflow(workflowId));
  const handleDelete = () =>
    runAction('delete', async () => {
      await openevolveApi.deleteWorkflow(workflowId);
      onDeleted();
    });

  const isRunning = detail?.status === 'running';
  const isPaused = detail?.status === 'paused';

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-gray-900 dark:text-white truncate">
            Workflow Detail
          </h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 font-mono truncate">
            {workflowId}
          </p>
        </div>
        <button
          onClick={onClose}
          className="px-3 py-1 text-sm font-medium text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
          aria-label="Close detail panel"
        >
          Close
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {loading && (
          <p className="text-sm text-gray-500 dark:text-gray-400">Loading workflow…</p>
        )}

        {error && (
          <div className="rounded-lg border border-red-300 bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-700 dark:text-red-300">
            {error}
          </div>
        )}

        {!loading && !error && detail && (
          <>
            <section>
              <div className="flex items-center gap-3 mb-2">
                <StatusBadge status={detail.status} />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Stage: {detail.current_stage || '—'}
                </span>
              </div>
              <div className="mb-3">
                <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>Progress</span>
                  <span>{Math.round(detail.progress * 100)}%</span>
                </div>
                <ProgressBar value={detail.progress} />
              </div>
            </section>

            <section className="space-y-2">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Problem</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {detail.problem_statement || '—'}
              </p>
            </section>

            <section className="grid grid-cols-2 gap-3 text-sm">
              <Metric label="Started" value={normalizeTs(detail.start_time)} />
              <Metric label="Ended" value={normalizeTs(detail.end_time)} />
              <Metric label="Refinement loops" value={String(detail.refinement_loop_count)} />
              <Metric
                label="Sub-problems"
                value={`${detail.solved_sub_problems} / ${detail.total_sub_problems}`}
              />
            </section>

            <section className="flex flex-wrap gap-2">
              {isRunning && (
                <button
                  onClick={handlePause}
                  disabled={actionLoading !== null}
                  className="px-3 py-1.5 text-sm font-medium rounded-md bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50"
                >
                  {actionLoading === 'pause' ? 'Pausing…' : 'Pause'}
                </button>
              )}
              {isPaused && (
                <button
                  onClick={handleResume}
                  disabled={actionLoading !== null}
                  className="px-3 py-1.5 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
                >
                  {actionLoading === 'resume' ? 'Resuming…' : 'Resume'}
                </button>
              )}
              <button
                onClick={handleDelete}
                disabled={actionLoading !== null}
                className="px-3 py-1.5 text-sm font-medium rounded-md bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
              >
                {actionLoading === 'delete' ? 'Deleting…' : 'Delete'}
              </button>
            </section>

            {results && (
              <section className="space-y-3">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Results</h4>

                {results.final_solution ? (
                  <SolutionBlock
                    title="Final Solution"
                    content={results.final_solution.content}
                    generatedBy={results.final_solution.generated_by}
                    timestamp={results.final_solution.timestamp}
                  />
                ) : (
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    No final solution available yet.
                  </p>
                )}

                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                    {Object.keys(results.sub_problem_solutions).length} sub-problem solution(s)
                    {typeof results.execution_time === 'number'
                      ? ` • ${Math.round(results.execution_time * 100) / 100}s execution`
                      : ''}
                    {` • ${results.refinement_loops} refinement loop(s)`}
                  </p>
                  <div className="space-y-2">
                    {Object.entries(results.sub_problem_solutions).map(([key, sol]) => (
                      <SolutionBlock
                        key={key}
                        title={key}
                        content={sol.content}
                        generatedBy={sol.generated_by}
                        timestamp={sol.timestamp}
                      />
                    ))}
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
      <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
      <p className="text-sm font-medium text-gray-900 dark:text-white mt-0.5 break-words">
        {value}
      </p>
    </div>
  );
}
