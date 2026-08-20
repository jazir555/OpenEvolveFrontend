/**
 * Gauntlet Form
 *
 * Create / edit a single {@link GauntletDefinition}. Mirrors the canonical
 * OpenEvolve gauntlet contract: a name, an owning team, an optional
 * description, and a repeatable list of {@link GauntletRoundRule} rows.
 *
 * This component is intentionally stateless about persistence: it calls the
 * `onSubmit` callback (which the parent wires to `openevolveApi`) and lets the
 * parent own loading / saving / error state.
 */

import { useState } from 'react';
import {
  type GauntletDefinition,
  type GauntletRoundRule,
  type TeamSummary,
  createDefaultGauntlet,
  createDefaultGauntletRound,
} from '@/types/openevolve';

/** Numeric round fields the designer is allowed to edit directly. */
const NUMERIC_ROUND_FIELDS = [
  'round_number',
  'quorum_required_approvals',
  'quorum_from_panel_size',
  'min_overall_confidence',
] as const;

type NumericRoundField = (typeof NUMERIC_ROUND_FIELDS)[number];

interface GauntletFormProps {
  /** When provided, the form edits this definition; otherwise it creates one. */
  initial?: GauntletDefinition | null;
  /** Available teams, used to populate the `team_name` select. */
  teams: TeamSummary[];
  /** Called with the assembled definition when the form is submitted. */
  onSubmit: (definition: GauntletDefinition) => Promise<void>;
  /** Called when the user dismisses the form without saving. */
  onCancel: () => void;
  /** True while the parent is persisting; disables the submit button. */
  isSubmitting?: boolean;
}

/**
 * Re-number rounds sequentially (1, 2, 3, ...) so `round_number` always mirrors
 * the row position regardless of how rows were added or removed.
 */
const reindexRounds = (rounds: GauntletRoundRule[]): GauntletRoundRule[] =>
  rounds.map((round, index) => ({ ...round, round_number: index + 1 }));

const inputClass =
  'mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm ' +
  'focus:border-blue-500 focus:outline-none focus:ring-blue-500 ' +
  'dark:border-gray-600 dark:bg-gray-700 dark:text-white';

const labelClass =
  'block text-xs font-medium text-gray-700 dark:text-gray-300';

export function GauntletForm({
  initial,
  teams,
  onSubmit,
  onCancel,
  isSubmitting = false,
}: GauntletFormProps) {
  const [definition, setDefinition] = useState<GauntletDefinition>(
    () => initial ?? createDefaultGauntlet()
  );
  const [error, setError] = useState<string | null>(null);

  const isEditing = Boolean(initial);

  const updateField = <K extends keyof GauntletDefinition>(
    field: K,
    value: GauntletDefinition[K]
  ) => {
    setDefinition((prev) => ({ ...prev, [field]: value }));
  };

  const updateRoundField = (
    index: number,
    field: NumericRoundField,
    value: number
  ) => {
    setDefinition((prev) => ({
      ...prev,
      rounds: prev.rounds.map((round, i) =>
        i === index ? ({ ...round, [field]: value } as GauntletRoundRule) : round
      ),
    }));
  };

  const addRound = () => {
    setDefinition((prev) => ({
      ...prev,
      rounds: reindexRounds([
        ...prev.rounds,
        createDefaultGauntletRound(prev.rounds.length + 1),
      ]),
    }));
  };

  const removeRound = (index: number) => {
    setDefinition((prev) => ({
      ...prev,
      rounds: reindexRounds(prev.rounds.filter((_, i) => i !== index)),
    }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    const name = definition.name.trim();
    if (!name) {
      setError('A gauntlet name is required.');
      return;
    }
    if (!definition.team_name) {
      setError('A team must be selected.');
      return;
    }
    if (definition.rounds.length === 0) {
      setError('At least one round is required.');
      return;
    }

    const payload: GauntletDefinition = {
      ...definition,
      name,
      rounds: reindexRounds(definition.rounds),
    };

    try {
      await onSubmit(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save gauntlet.');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Name */}
      <div>
        <label
          htmlFor="gauntlet-name"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Gauntlet Name *
        </label>
        <input
          id="gauntlet-name"
          type="text"
          value={definition.name}
          onChange={(e) => updateField('name', e.target.value)}
          required
          disabled={isEditing}
          className={inputClass}
          placeholder="My Gauntlet"
        />
        {isEditing && (
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            The name is the gauntlet&apos;s identifier and cannot be changed.
          </p>
        )}
      </div>

      {/* Team */}
      <div>
        <label
          htmlFor="gauntlet-team"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Team *
        </label>
        <select
          id="gauntlet-team"
          value={definition.team_name}
          onChange={(e) => updateField('team_name', e.target.value)}
          required
          className={inputClass}
        >
          <option value="">Select a team…</option>
          {teams.map((team) => (
            <option key={team.name} value={team.name}>
              {team.name}
            </option>
          ))}
        </select>
      </div>

      {/* Description */}
      <div>
        <label
          htmlFor="gauntlet-description"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Description
        </label>
        <textarea
          id="gauntlet-description"
          value={definition.description ?? ''}
          onChange={(e) => updateField('description', e.target.value)}
          rows={2}
          className={inputClass}
          placeholder="Optional description…"
        />
      </div>

      {/* Rounds */}
      <div>
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Rounds *
          </label>
          <button
            type="button"
            onClick={addRound}
            className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
          >
            + Add Round
          </button>
        </div>

        <div className="mt-2 space-y-4">
          {definition.rounds.map((round, index) => (
            <div
              key={index}
              className="rounded-lg border border-gray-200 p-4 dark:border-gray-700"
            >
              <div className="mb-3 flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  Round {round.round_number}
                </h4>
                {definition.rounds.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeRound(index)}
                    className="text-sm text-red-600 hover:text-red-700 dark:text-red-400"
                  >
                    Remove
                  </button>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className={labelClass}>Round Number</label>
                  <input
                    type="number"
                    min={1}
                    value={round.round_number}
                    onChange={(e) =>
                      updateRoundField(
                        index,
                        'round_number',
                        e.target.valueAsNumber || index + 1
                      )
                    }
                    className={inputClass}
                  />
                </div>

                <div>
                  <label className={labelClass}>
                    Quorum Required Approvals
                  </label>
                  <input
                    type="number"
                    min={0}
                    value={round.quorum_required_approvals}
                    onChange={(e) =>
                      updateRoundField(
                        index,
                        'quorum_required_approvals',
                        e.target.valueAsNumber
                      )
                    }
                    className={inputClass}
                  />
                </div>

                <div>
                  <label className={labelClass}>Quorum From Panel Size</label>
                  <input
                    type="number"
                    min={0}
                    value={round.quorum_from_panel_size}
                    onChange={(e) =>
                      updateRoundField(
                        index,
                        'quorum_from_panel_size',
                        e.target.valueAsNumber
                      )
                    }
                    className={inputClass}
                  />
                </div>

                <div>
                  <label className={labelClass}>Min Overall Confidence</label>
                  <input
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={round.min_overall_confidence ?? 0}
                    onChange={(e) =>
                      updateRoundField(
                        index,
                        'min_overall_confidence',
                        e.target.valueAsNumber
                      )
                    }
                    className={inputClass}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-3 pt-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={isSubmitting}
          className="inline-flex justify-center rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isSubmitting ? 'Saving…' : isEditing ? 'Update Gauntlet' : 'Create Gauntlet'}
        </button>
      </div>
    </form>
  );
}

export default GauntletForm;
