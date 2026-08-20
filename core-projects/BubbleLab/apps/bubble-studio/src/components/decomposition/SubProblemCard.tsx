import { useMemo } from 'react';
import type { WorkflowSubProblem } from '@/types/openevolve';

export const SUB_PROBLEM_STATUS_OPTIONS = [
  'not_started',
  'pending',
  'in_progress',
  'blocked',
  'completed',
  'review',
] as const;

export interface SubProblemCardProps {
  subProblem: WorkflowSubProblem;
  index: number;
  allIds: string[];
  canRemove: boolean;
  onChange: (updated: WorkflowSubProblem) => void;
  onRemove: () => void;
}

/**
 * One editable sub-problem row/card inside the Decomposition / Manual Review panel.
 *
 * The canonical `WorkflowSubProblem` has no `title` field — its human-facing label
 * is `description`. `assigned_team` maps to the optional `solver_team_name`. These
 * are surfaced as inline-editable controls alongside `status` and `dependencies`.
 */
export function SubProblemCard({
  subProblem,
  index,
  allIds,
  canRemove,
  onChange,
  onRemove,
}: SubProblemCardProps) {
  const otherIds = useMemo(
    () => allIds.filter((id) => id !== subProblem.id),
    [allIds, subProblem.id]
  );

  const update = (patch: Partial<WorkflowSubProblem>) =>
    onChange({ ...subProblem, ...patch });

  const toggleDependency = (depId: string) => {
    const current = subProblem.dependencies ?? [];
    const next = current.includes(depId)
      ? current.filter((id) => id !== depId)
      : [...current, depId];
    update({ dependencies: next });
  };

  return (
    <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="rounded bg-[#1c1c1c] px-2 py-0.5 font-mono text-xs text-gray-400">
            #{index + 1}
          </span>
          <span className="font-mono text-xs text-amber-300/80">{subProblem.id}</span>
        </div>
        {canRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="rounded border border-[#3a3a3a] px-2 py-1 text-xs text-red-300 hover:bg-[#1a1a1a] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Remove
          </button>
        )}
      </div>

      <label className="mt-3 block">
        <span className="mb-1 block text-xs text-gray-400">Description</span>
        <textarea
          value={subProblem.description}
          onChange={(event) => update({ description: event.target.value })}
          rows={2}
          spellCheck={false}
          className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
        />
      </label>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1 block text-xs text-gray-400">Status</span>
          <select
            value={subProblem.status ?? ''}
            onChange={(event) =>
              update({ status: event.target.value || undefined })
            }
            className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
          >
            <option value="">(none)</option>
            {SUB_PROBLEM_STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-xs text-gray-400">Assigned Team (solver)</span>
          <input
            value={subProblem.solver_team_name ?? ''}
            onChange={(event) => update({ solver_team_name: event.target.value })}
            placeholder="e.g. solver_team"
            className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
          />
        </label>
      </div>

      <div className="mt-3">
        <span className="mb-1 block text-xs text-gray-400">
          Dependencies{' '}
          <span className="text-gray-600">(sub-problems this one depends on)</span>
        </span>
        {otherIds.length === 0 ? (
          <p className="text-xs text-gray-600">No other sub-problems available.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {otherIds.map((id) => {
              const checked = (subProblem.dependencies ?? []).includes(id);
              return (
                <label
                  key={id}
                  className="flex cursor-pointer items-center gap-1.5 rounded border border-[#2e2e2e] bg-[#111] px-2 py-1 text-xs text-gray-300 hover:border-[#3a3a3a]"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleDependency(id)}
                    className="h-3 w-3 accent-amber-500"
                  />
                  <span className="font-mono">{id}</span>
                </label>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
