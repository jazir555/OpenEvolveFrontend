/**
 * Gauntlet Designer
 *
 * Top-level view for managing OpenEvolve gauntlets. Lists existing gauntlets
 * (as {@link GauntletSummary}), exposes "Create Gauntlet", and per-row
 * Edit / Delete actions. Editing opens a modal {@link GauntletForm}; the form's
 * submit is wired to `openevolveApi.createGauntletDefinition` /
 * `openevolveApi.updateGauntlet`.
 *
 * Team names are sourced from `openevolveApi.listTeamSummaries()` and used to
 * populate the `team_name` select in the form.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  type GauntletDefinition,
  type GauntletSummary,
  type TeamSummary,
  createDefaultGauntlet,
} from '@/types/openevolve';
import { openevolveApi } from '@/services/openevolveApi';
import { GauntletForm } from './GauntletForm';

type EditorState =
  | { mode: 'closed' }
  | { mode: 'create' }
  | { mode: 'edit'; definition: GauntletDefinition; id: string };

export function GauntletDesigner() {
  const [gauntlets, setGauntlets] = useState<GauntletSummary[]>([]);
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState>({ mode: 'closed' });
  const [saving, setSaving] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [gauntletResult, teamResult] = await Promise.all([
        openevolveApi.listGauntletSummaries(),
        openevolveApi.listTeamSummaries(),
      ]);
      setGauntlets(gauntletResult.gauntlets);
      setTeams(teamResult.teams);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load gauntlets.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const handleCreate = () => {
    setEditor({ mode: 'create' });
  };

  const handleEdit = async (summary: GauntletSummary) => {
    setError(null);
    try {
      const definition = await openevolveApi.getGauntletDefinition(summary.id);
      setEditor({ mode: 'edit', definition, id: summary.id });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load gauntlet.');
    }
  };

  const handleDelete = async (summary: GauntletSummary) => {
    if (!window.confirm(`Delete gauntlet "${summary.name}"? This cannot be undone.`)) {
      return;
    }
    setError(null);
    try {
      await openevolveApi.deleteGauntlet(summary.id);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete gauntlet.');
    }
  };

  const handleSubmit = async (definition: GauntletDefinition) => {
    setSaving(true);
    setError(null);
    try {
      if (editor.mode === 'edit') {
        await openevolveApi.updateGauntlet(editor.id, definition);
      } else {
        await openevolveApi.createGauntletDefinition({
          ...createDefaultGauntlet(),
          ...definition,
        });
      }
      setEditor({ mode: 'closed' });
      await loadData();
    } finally {
      setSaving(false);
    }
  };

  const closeEditor = () => setEditor({ mode: 'closed' });

  const isEditorOpen = editor.mode !== 'closed';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Gauntlet Designer
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Configure multi-round evaluation runs and their quorum rules.
          </p>
        </div>
        <button
          type="button"
          onClick={handleCreate}
          className="inline-flex items-center rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700"
        >
          Create Gauntlet
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700"
            />
          ))}
        </div>
      ) : gauntlets.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-600">
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No gauntlets
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Get started by creating a new gauntlet.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Team
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Rounds
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Description
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-800">
              {gauntlets.map((gauntlet) => (
                <tr key={gauntlet.name}>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">
                    {gauntlet.name}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {gauntlet.team_name}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {gauntlet.round_count}
                  </td>
                  <td className="max-w-xs truncate px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                    {gauntlet.description || '—'}
                  </td>
                  <td className="px-4 py-3 text-right text-sm">
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => void handleEdit(gauntlet)}
                        className="inline-flex items-center rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                          onClick={() => void handleDelete(gauntlet)}
                        className="inline-flex items-center rounded-md border border-red-300 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-900/20"
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

      {isEditorOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div
              className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
              onClick={closeEditor}
            />
            <span className="hidden sm:inline-block sm:h-screen sm:align-middle">
              &#8203;
            </span>

            <div className="inline-block max-h-[85vh] transform overflow-y-auto rounded-lg bg-white text-left align-bottom shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-4xl sm:align-middle dark:bg-gray-800">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4 dark:bg-gray-800">
                <h3 className="mb-4 text-lg font-medium text-gray-900 dark:text-white">
                  {editor.mode === 'edit' ? 'Edit Gauntlet' : 'Create Gauntlet'}
                </h3>
                <GauntletForm
                  initial={editor.mode === 'edit' ? editor.definition : null}
                  teams={teams}
                  onSubmit={handleSubmit}
                  onCancel={closeEditor}
                  isSubmitting={saving}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default GauntletDesigner;
