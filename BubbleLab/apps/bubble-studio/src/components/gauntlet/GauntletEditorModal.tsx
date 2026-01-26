/**
 * Gauntlet Editor Modal
 * Create or edit a gauntlet
 */

import { useState } from 'react';
import { Gauntlet, GauntletRound, CreateGauntletRequest } from '../../types/api';

interface GauntletEditorModalProps {
  gauntlet?: Gauntlet;
  isOpen: boolean;
  onClose: () => void;
  onSave: (gauntlet: CreateGauntletRequest) => void;
  isSaving?: boolean;
}

export function GauntletEditorModal({ gauntlet, isOpen, onClose, onSave, isSaving = false }: GauntletEditorModalProps) {
  const [name, setName] = useState(gauntlet?.name || '');
  const [description, setDescription] = useState(gauntlet?.description || '');
  const [rounds, setRounds] = useState<Omit<GauntletRound, 'id'>[]>(
    gauntlet?.rounds.map(({ id, ...round }) => round) || [
      {
        name: 'Round 1',
        quorum_threshold: 0.7,
        confidence_threshold: 0.8,
        evaluation_type: 'majority_vote',
        required_consensus: true,
        max_iterations: 3,
      },
    ]
  );

  if (!isOpen) return null;

  const handleAddRound = () => {
    setRounds([
      ...rounds,
      {
        name: `Round ${rounds.length + 1}`,
        quorum_threshold: 0.7,
        confidence_threshold: 0.8,
        evaluation_type: 'majority_vote',
        required_consensus: true,
        max_iterations: 3,
      },
    ]);
  };

  const handleRemoveRound = (index: number) => {
    setRounds(rounds.filter((_, i) => i !== index));
  };

  const handleRoundChange = (index: number, field: keyof Omit<GauntletRound, 'id'>, value: any) => {
    const updatedRounds = [...rounds];
    updatedRounds[index] = { ...updatedRounds[index], [field]: value };
    setRounds(updatedRounds);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      name,
      description,
      rounds,
    });
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
          onClick={onClose}
        />
        <span className="hidden sm:inline-block sm:h-screen sm:align-middle">&#8203;</span>

        <div className="inline-block transform overflow-hidden rounded-lg bg-white text-left align-bottom shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-4xl sm:align-middle dark:bg-gray-800">
          <form onSubmit={handleSubmit}>
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4 dark:bg-gray-800">
              <div className="sm:flex sm:items-start">
                <div className="w-full">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    {gauntlet ? 'Edit Gauntlet' : 'Create Gauntlet'}
                  </h3>

                  <div className="mt-4 space-y-4">
                    {/* Gauntlet Name */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Gauntlet Name *
                      </label>
                      <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                        className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        placeholder="My Gauntlet"
                      />
                    </div>

                    {/* Description */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Description
                      </label>
                      <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        rows={2}
                        className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        placeholder="Optional description..."
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
                          onClick={handleAddRound}
                          className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
                        >
                          + Add Round
                        </button>
                      </div>

                      <div className="mt-2 space-y-4">
                        {rounds.map((round, index) => (
                          <div
                            key={index}
                            className="rounded-lg border border-gray-200 p-4 dark:border-gray-700"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                                {round.name}
                              </h4>
                              {rounds.length > 1 && (
                                <button
                                  type="button"
                                  onClick={() => handleRemoveRound(index)}
                                  className="text-sm text-red-600 hover:text-red-700 dark:text-red-400"
                                >
                                  Remove
                                </button>
                              )}
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Round Name
                                </label>
                                <input
                                  type="text"
                                  value={round.name}
                                  onChange={(e) => handleRoundChange(index, 'name', e.target.value)}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Quorum Threshold
                                </label>
                                <input
                                  type="number"
                                  step="0.1"
                                  min="0"
                                  max="1"
                                  value={round.quorum_threshold}
                                  onChange={(e) => handleRoundChange(index, 'quorum_threshold', parseFloat(e.target.value))}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Confidence Threshold
                                </label>
                                <input
                                  type="number"
                                  step="0.1"
                                  min="0"
                                  max="1"
                                  value={round.confidence_threshold}
                                  onChange={(e) => handleRoundChange(index, 'confidence_threshold', parseFloat(e.target.value))}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Max Iterations
                                </label>
                                <input
                                  type="number"
                                  min="1"
                                  value={round.max_iterations}
                                  onChange={(e) => handleRoundChange(index, 'max_iterations', parseInt(e.target.value))}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div className="col-span-2">
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Evaluation Type
                                </label>
                                <select
                                  value={round.evaluation_type}
                                  onChange={(e) => handleRoundChange(index, 'evaluation_type', e.target.value)}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                >
                                  <option value="majority_vote">Majority Vote</option>
                                  <option value="consensus">Consensus</option>
                                  <option value="unanimous">Unanimous</option>
                                  <option value="weighted">Weighted</option>
                                </select>
                              </div>

                              <div className="col-span-2 flex items-center">
                                <input
                                  type="checkbox"
                                  id={`consensus-${index}`}
                                  checked={round.required_consensus}
                                  onChange={(e) => handleRoundChange(index, 'required_consensus', e.target.checked)}
                                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                <label htmlFor={`consensus-${index}`} className="ml-2 block text-sm text-gray-900 dark:text-white">
                                  Require consensus
                                </label>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 dark:bg-gray-700">
                <button
                  type="submit"
                  disabled={isSaving}
                  className="inline-flex w-full justify-center rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 sm:ml-3 sm:w-auto disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save'}
                </button>
                <button
                  type="button"
                  onClick={onClose}
                  className="mt-3 inline-flex w-full justify-center rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 sm:mt-0 sm:ml-3 sm:w-auto dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
