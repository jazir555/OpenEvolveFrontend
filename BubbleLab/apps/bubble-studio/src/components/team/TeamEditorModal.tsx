/**
 * Team Editor Modal
 * Create or edit a team
 */

import { useState } from 'react';
import { Team, TeamMember, CreateTeamRequest } from '../../types/api';

interface TeamEditorModalProps {
  team?: Team;
  isOpen: boolean;
  onClose: () => void;
  onSave: (team: CreateTeamRequest) => void;
  isSaving?: boolean;
}

export function TeamEditorModal({ team, isOpen, onClose, onSave, isSaving = false }: TeamEditorModalProps) {
  const [name, setName] = useState(team?.name || '');
  const [description, setDescription] = useState(team?.description || '');
  const [members, setMembers] = useState<Omit<TeamMember, 'id'>[]>(
    team?.members.map(({ id, ...member }) => member) || [
      {
        name: 'Member 1',
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        max_iterations: 5,
        role: 'analyst',
      },
    ]
  );

  if (!isOpen) return null;

  const handleAddMember = () => {
    setMembers([
      ...members,
      {
        name: `Member ${members.length + 1}`,
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        max_iterations: 5,
        role: 'analyst',
      },
    ]);
  };

  const handleRemoveMember = (index: number) => {
    setMembers(members.filter((_, i) => i !== index));
  };

  const handleMemberChange = (index: number, field: keyof Omit<TeamMember, 'id'>, value: any) => {
    const updatedMembers = [...members];
    updatedMembers[index] = { ...updatedMembers[index], [field]: value };
    setMembers(updatedMembers);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      name,
      description,
      members,
    });
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
          onClick={onClose}
        />

        <span className="hidden sm:inline-block sm:h-screen sm:align-middle">&#8203;</span>

        {/* Modal panel */}
        <div className="inline-block transform overflow-hidden rounded-lg bg-white text-left align-bottom shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-4xl sm:align-middle dark:bg-gray-800">
          <form onSubmit={handleSubmit}>
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4 dark:bg-gray-800">
              <div className="sm:flex sm:items-start">
                <div className="w-full">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    {team ? 'Edit Team' : 'Create Team'}
                  </h3>

                  <div className="mt-4 space-y-4">
                    {/* Team Name */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Team Name *
                      </label>
                      <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                        className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        placeholder="My Team"
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

                    {/* Members */}
                    <div>
                      <div className="flex items-center justify-between">
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                          Members *
                        </label>
                        <button
                          type="button"
                          onClick={handleAddMember}
                          className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
                        >
                          + Add Member
                        </button>
                      </div>

                      <div className="mt-2 space-y-4">
                        {members.map((member, index) => (
                          <div
                            key={index}
                            className="rounded-lg border border-gray-200 p-4 dark:border-gray-700"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                                Member {index + 1}
                              </h4>
                              {members.length > 1 && (
                                <button
                                  type="button"
                                  onClick={() => handleRemoveMember(index)}
                                  className="text-sm text-red-600 hover:text-red-700 dark:text-red-400"
                                >
                                  Remove
                                </button>
                              )}
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Name
                                </label>
                                <input
                                  type="text"
                                  value={member.name}
                                  onChange={(e) => handleMemberChange(index, 'name', e.target.value)}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Model
                                </label>
                                <select
                                  value={member.model}
                                  onChange={(e) => handleMemberChange(index, 'model', e.target.value)}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                >
                                  <option value="gpt-4">GPT-4</option>
                                  <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                                  <option value="claude-3-opus">Claude 3 Opus</option>
                                  <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                                </select>
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Temperature
                                </label>
                                <input
                                  type="number"
                                  step="0.1"
                                  min="0"
                                  max="2"
                                  value={member.temperature}
                                  onChange={(e) => handleMemberChange(index, 'temperature', parseFloat(e.target.value))}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>

                              <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                  Max Tokens
                                </label>
                                <input
                                  type="number"
                                  min="1"
                                  value={member.max_tokens}
                                  onChange={(e) => handleMemberChange(index, 'max_tokens', parseInt(e.target.value))}
                                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
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
          </form>
        </div>
      </div>
    </div>
  );
}
