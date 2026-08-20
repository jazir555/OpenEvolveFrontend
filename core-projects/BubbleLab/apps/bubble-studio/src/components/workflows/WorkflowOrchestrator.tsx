/**
 * WorkflowOrchestrator
 *
 * Main view for the OpenEvolve decomposition Workflow Orchestrator. Lists
 * workflows, allows creating a new decomposition workflow, pausing/resuming
 * running workflows, deleting them, and opening a detail/results panel.
 *
 * Uses the canonical OpenEvolve client surface from `@/services/openevolveApi`.
 */

import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { openevolveApi } from '@/services/openevolveApi';
import type { WorkflowSummary, WorkflowCreateRequest } from '@/types/openevolve';
import { StatusBadge } from '../common/StatusBadge';
import { WorkflowDetailPanel } from './WorkflowDetail';

const DEFAULT_TEAM = 'default';
const DEFAULT_GAUNTLET = 'default';

function canPause(status: string): boolean {
  return status === 'running';
}

function canResume(status: string): boolean {
  return status === 'paused';
}

function CreateWorkflowForm({
  onCreated,
  onCancel,
}: {
  onCreated: (id: string) => void;
  onCancel: () => void;
}) {
  const [problemStatement, setProblemStatement] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [contentAnalyzerTeam, setContentAnalyzerTeam] = useState(DEFAULT_TEAM);
  const [plannerTeam, setPlannerTeam] = useState(DEFAULT_TEAM);
  const [solverTeam, setSolverTeam] = useState(DEFAULT_TEAM);
  const [patcherTeam, setPatcherTeam] = useState(DEFAULT_TEAM);
  const [assemblerTeam, setAssemblerTeam] = useState(DEFAULT_TEAM);
  const [subProblemRedGauntlet, setSubProblemRedGauntlet] = useState(DEFAULT_GAUNTLET);
  const [subProblemGoldGauntlet, setSubProblemGoldGauntlet] = useState(DEFAULT_GAUNTLET);
  const [finalRedGauntlet, setFinalRedGauntlet] = useState(DEFAULT_GAUNTLET);
  const [finalGoldGauntlet, setFinalGoldGauntlet] = useState(DEFAULT_GAUNTLET);
  const [solverGenerationGauntlet, setSolverGenerationGauntlet] = useState(DEFAULT_GAUNTLET);
  const [maxRefinementLoops, setMaxRefinementLoops] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!problemStatement.trim()) {
      setError('A problem statement is required.');
      return;
    }
    setSubmitting(true);
    setError(null);

    const payload: WorkflowCreateRequest = {
      problem_statement: problemStatement.trim(),
      content_analyzer_team: contentAnalyzerTeam.trim() || DEFAULT_TEAM,
      planner_team: plannerTeam.trim() || DEFAULT_TEAM,
      solver_team: solverTeam.trim() || DEFAULT_TEAM,
      patcher_team: patcherTeam.trim() || DEFAULT_TEAM,
      assembler_team: assemblerTeam.trim() || DEFAULT_TEAM,
      sub_problem_red_gauntlet: subProblemRedGauntlet.trim() || DEFAULT_GAUNTLET,
      sub_problem_gold_gauntlet: subProblemGoldGauntlet.trim() || DEFAULT_GAUNTLET,
      final_red_gauntlet: finalRedGauntlet.trim() || DEFAULT_GAUNTLET,
      final_gold_gauntlet: finalGoldGauntlet.trim() || DEFAULT_GAUNTLET,
      solver_generation_gauntlet: solverGenerationGauntlet.trim() || DEFAULT_GAUNTLET,
    };

    const loops = Number(maxRefinementLoops);
    if (maxRefinementLoops.trim() !== '' && Number.isFinite(loops) && loops > 0) {
      payload.max_refinement_loops = Math.floor(loops);
    }

    try {
      const res = await openevolveApi.createDecompositionWorkflow(payload);
      onCreated(res.workflow_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workflow');
    } finally {
      setSubmitting(false);
    }
  };

  const inputClass =
    'w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500';

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-4 space-y-4"
    >
      <h3 className="text-base font-semibold text-gray-900 dark:text-white">
        Create Decomposition Workflow
      </h3>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Problem Statement <span className="text-red-500">*</span>
        </label>
        <textarea
          value={problemStatement}
          onChange={(e) => setProblemStatement(e.target.value)}
          rows={3}
          required
          placeholder="Describe the problem to decompose and solve…"
          className={inputClass}
        />
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced((v) => !v)}
        className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
      >
        {showAdvanced ? 'Hide' : 'Show'} teams &amp; gauntlets
      </button>

      {showAdvanced && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Field label="Content Analyzer Team" value={contentAnalyzerTeam} onChange={setContentAnalyzerTeam} />
          <Field label="Planner Team" value={plannerTeam} onChange={setPlannerTeam} />
          <Field label="Solver Team" value={solverTeam} onChange={setSolverTeam} />
          <Field label="Patcher Team" value={patcherTeam} onChange={setPatcherTeam} />
          <Field label="Assembler Team" value={assemblerTeam} onChange={setAssemblerTeam} />
          <Field label="Sub-problem Red Gauntlet" value={subProblemRedGauntlet} onChange={setSubProblemRedGauntlet} />
          <Field label="Sub-problem Gold Gauntlet" value={subProblemGoldGauntlet} onChange={setSubProblemGoldGauntlet} />
          <Field label="Final Red Gauntlet" value={finalRedGauntlet} onChange={setFinalRedGauntlet} />
          <Field label="Final Gold Gauntlet" value={finalGoldGauntlet} onChange={setFinalGoldGauntlet} />
          <Field label="Solver Generation Gauntlet" value={solverGenerationGauntlet} onChange={setSolverGenerationGauntlet} />
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Max Refinement Loops (optional)
            </label>
            <input
              type="number"
              min={1}
              value={maxRefinementLoops}
              onChange={(e) => setMaxRefinementLoops(e.target.value)}
              className={inputClass}
            />
          </div>
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="submit"
          disabled={submitting}
          className="px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {submitting ? 'Creating…' : 'Create Workflow'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={submitting}
          className="px-4 py-2 text-sm font-medium rounded-md border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
        {label}
      </label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
}

export function WorkflowOrchestrator() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await openevolveApi.listWorkflowSummaries();
      setWorkflows(res.workflows);
      setTotal(res.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workflows');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const runRowAction = useCallback(
    async (id: string, label: 'pause' | 'resume' | 'delete', fn: () => Promise<unknown>) => {
      setBusyId(id);
      setError(null);
      try {
        await fn();
        await load();
        if (selectedId === id) {
          // Detail panel refreshes itself via onStatusChanged from its buttons;
          // for row actions we just close to reflect the list state.
          if (label === 'delete') setSelectedId(null);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : `Failed to ${label} workflow`);
      } finally {
        setBusyId(null);
      }
    },
    [load, selectedId]
  );

  const handleCreated = useCallback(
    async (id: string) => {
      setShowCreate(false);
      await load();
      setSelectedId(id);
    },
    [load]
  );

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Workflow Orchestrator
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {total} decomposition workflow{total === 1 ? '' : 's'}
          </p>
        </div>
        <button
          onClick={() => setShowCreate((v) => !v)}
          className="px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700"
        >
          {showCreate ? 'Close Form' : 'Create Workflow'}
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-red-300 bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {showCreate && (
        <div className="mb-6">
          <CreateWorkflowForm onCreated={handleCreated} onCancel={() => setShowCreate(false)} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {loading ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">Loading workflows…</p>
          ) : workflows.length === 0 ? (
            <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center text-sm text-gray-500 dark:text-gray-400">
              No workflows yet. Click “Create Workflow” to start one.
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-900/40 text-gray-500 dark:text-gray-400">
                  <tr>
                    <th className="text-left font-medium px-4 py-3">Workflow ID</th>
                    <th className="text-left font-medium px-4 py-3">Status</th>
                    <th className="text-left font-medium px-4 py-3">Stage</th>
                    <th className="text-left font-medium px-4 py-3">Progress</th>
                    <th className="text-right font-medium px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {workflows.map((wf) => (
                    <tr
                      key={wf.workflow_id}
                      className={selectedId === wf.workflow_id ? 'bg-blue-50 dark:bg-blue-900/20' : ''}
                    >
                      <td className="px-4 py-3 font-mono text-xs text-gray-700 dark:text-gray-300">
                        {wf.workflow_id}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={wf.status} />
                      </td>
                      <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                        {wf.current_stage || '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                        {Math.round(wf.progress * 100)}%
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => setSelectedId(wf.workflow_id)}
                            className="px-2.5 py-1 text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                          >
                            View
                          </button>
                          <button
                            onClick={() => runRowAction(wf.workflow_id, 'pause', () => openevolveApi.pauseWorkflow(wf.workflow_id))}
                            disabled={!canPause(wf.status) || busyId === wf.workflow_id}
                            className="px-2.5 py-1 text-xs font-medium rounded-md border border-orange-300 text-orange-700 hover:bg-orange-50 dark:border-orange-700 dark:text-orange-400 dark:hover:bg-orange-900/20 disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            Pause
                          </button>
                          <button
                            onClick={() => runRowAction(wf.workflow_id, 'resume', () => openevolveApi.resumeWorkflow(wf.workflow_id))}
                            disabled={!canResume(wf.status) || busyId === wf.workflow_id}
                            className="px-2.5 py-1 text-xs font-medium rounded-md border border-green-300 text-green-700 hover:bg-green-50 dark:border-green-700 dark:text-green-400 dark:hover:bg-green-900/20 disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            Resume
                          </button>
                          <button
                            onClick={() => runRowAction(wf.workflow_id, 'delete', () => openevolveApi.deleteWorkflow(wf.workflow_id))}
                            disabled={busyId === wf.workflow_id}
                            className="px-2.5 py-1 text-xs font-medium rounded-md border border-red-300 text-red-700 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20 disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="lg:col-span-1">
          {selectedId ? (
            <WorkflowDetailPanel
              key={selectedId}
              workflowId={selectedId}
              onClose={() => setSelectedId(null)}
              onStatusChanged={() => void load()}
              onDeleted={() => {
                setSelectedId(null);
                void load();
              }}
            />
          ) : (
            <div className="bg-white dark:bg-gray-800 border border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center text-sm text-gray-500 dark:text-gray-400">
              Select “View” on a workflow to see its detail and results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default WorkflowOrchestrator;
