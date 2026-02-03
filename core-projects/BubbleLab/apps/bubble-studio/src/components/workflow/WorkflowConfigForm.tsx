/**
 * Workflow Configuration Form
 * Multi-step form for creating new workflows
 */

import { useState } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { useCreateWorkflow } from '../../hooks/use-workflows-api';
import { useTeams } from '../../hooks/use-teams-api';
import { useGauntlets } from '../../hooks/use-gauntlets-api';
import { CreateWorkflowRequest, WorkflowMetadata } from '../../types/api';

type FormStep = 'problem' | 'teams' | 'gauntlets' | 'advanced';

export function WorkflowConfigForm() {
  const navigate = useNavigate();
  const createWorkflow = useCreateWorkflow();
  const { data: teams } = useTeams();
  const { data: gauntlets } = useGauntlets();

  const [currentStep, setCurrentStep] = useState<FormStep>('problem');
  const [formData, setFormData] = useState<Partial<CreateWorkflowRequest>>({
    name: '',
    description: '',
    problem_statement: '',
    content_type: 'text',
    teams: [],
    gauntlets: [],
    metadata: {},
  });

  const steps: FormStep[] = ['problem', 'teams', 'gauntlets', 'advanced'];
  const currentStepIndex = steps.indexOf(currentStep);

  const updateFormData = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleNext = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStep(steps[currentStepIndex + 1]);
    }
  };

  const handleBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStep(steps[currentStepIndex - 1]);
    }
  };

  const handleSubmit = async () => {
    try {
      const workflow = await createWorkflow.mutateAsync(formData as CreateWorkflowRequest);
      navigate({ to: `/oe-workflows/${workflow.id}` });
    } catch (error) {
      console.error('Failed to create workflow:', error);
    }
  };

  return (
    <div className="mx-auto max-w-4xl">
      {/* Progress Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={step} className="flex items-center">
              <div
                className={`
                  flex h-10 w-10 items-center justify-center rounded-full border-2 font-medium
                  ${
                    index <= currentStepIndex
                      ? 'border-blue-600 bg-blue-600 text-white'
                      : 'border-gray-300 bg-white text-gray-500 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-400'
                  }
                `}
              >
                {index + 1}
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`h-0.5 w-24 ${
                    index < currentStepIndex ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                />
              )}
            </div>
          ))}
        </div>
        <div className="mt-2 flex justify-between text-xs text-gray-600 dark:text-gray-400">
          <span>Problem</span>
          <span>Teams</span>
          <span>Gauntlets</span>
          <span>Advanced</span>
        </div>
      </div>

      {/* Form Content */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        {currentStep === 'problem' && (
          <ProblemStatementStep formData={formData} updateFormData={updateFormData} />
        )}
        {currentStep === 'teams' && (
          <TeamSelectionStep formData={formData} updateFormData={updateFormData} teams={teams || []} />
        )}
        {currentStep === 'gauntlets' && (
          <GauntletSelectionStep formData={formData} updateFormData={updateFormData} gauntlets={gauntlets || []} />
        )}
        {currentStep === 'advanced' && (
          <AdvancedSettingsStep formData={formData} updateFormData={updateFormData} />
        )}
      </div>

      {/* Navigation Buttons */}
      <div className="mt-6 flex justify-between">
        <button
          onClick={handleBack}
          disabled={currentStepIndex === 0}
          className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700"
        >
          Back
        </button>
        {currentStepIndex < steps.length - 1 ? (
          <button
            onClick={handleNext}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={createWorkflow.isPending}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {createWorkflow.isPending ? 'Creating...' : 'Create Workflow'}
          </button>
        )}
      </div>
    </div>
  );
}

function ProblemStatementStep({
  formData,
  updateFormData,
}: {
  formData: Partial<CreateWorkflowRequest>;
  updateFormData: (field: string, value: any) => void;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Problem Statement
        </h2>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Define the problem you want to solve
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Workflow Name *
        </label>
        <input
          type="text"
          value={formData.name || ''}
          onChange={(e) => updateFormData('name', e.target.value)}
          className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          placeholder="My Awesome Workflow"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Description
        </label>
        <textarea
          value={formData.description || ''}
          onChange={(e) => updateFormData('description', e.target.value)}
          rows={3}
          className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          placeholder="Optional description..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Problem Statement *
        </label>
        <textarea
          value={formData.problem_statement || ''}
          onChange={(e) => updateFormData('problem_statement', e.target.value)}
          rows={5}
          required
          className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          placeholder="Describe the problem you want to solve..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Content Type
        </label>
        <select
          value={formData.content_type || 'text'}
          onChange={(e) => updateFormData('content_type', e.target.value)}
          className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
        >
          <option value="text">Text</option>
          <option value="code">Code</option>
          <option value="document">Document</option>
          <option value="theorem">Theorem</option>
        </select>
      </div>
    </div>
  );
}

function TeamSelectionStep({
  formData,
  updateFormData,
  teams,
}: {
  formData: Partial<CreateWorkflowRequest>;
  updateFormData: (field: string, value: any) => void;
  teams: any[];
}) {
  const toggleTeam = (teamId: string) => {
    const currentTeams = formData.teams || [];
    const updatedTeams = currentTeams.includes(teamId)
      ? currentTeams.filter((id) => id !== teamId)
      : [...currentTeams, teamId];
    updateFormData('teams', updatedTeams);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Select Teams
        </h2>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Choose which AI teams to work on this problem
        </p>
      </div>

      {teams.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center dark:border-gray-600">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            No teams available. Create a team first.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {teams.map((team) => (
            <label
              key={team.id}
              className="flex cursor-pointer items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-700"
            >
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">
                  {team.name}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {team.description || `${team.members?.length || 0} members`}
                </p>
              </div>
              <input
                type="checkbox"
                checked={(formData.teams || []).includes(team.id)}
                onChange={() => toggleTeam(team.id)}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function GauntletSelectionStep({
  formData,
  updateFormData,
  gauntlets,
}: {
  formData: Partial<CreateWorkflowRequest>;
  updateFormData: (field: string, value: any) => void;
  gauntlets: any[];
}) {
  const toggleGauntlet = (gauntletId: string) => {
    const currentGauntlets = formData.gauntlets || [];
    const updatedGauntlets = currentGauntlets.includes(gauntletId)
      ? currentGauntlets.filter((id) => id !== gauntletId)
      : [...currentGauntlets, gauntletId];
    updateFormData('gauntlets', updatedGauntlets);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Select Gauntlets
        </h2>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Choose validation gauntlets for quality assurance
        </p>
      </div>

      {gauntlets.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center dark:border-gray-600">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            No gauntlets available. Create a gauntlet first.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {gauntlets.map((gauntlet) => (
            <label
              key={gauntlet.id}
              className="flex cursor-pointer items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-700"
            >
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">
                  {gauntlet.name}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {gauntlet.description || `${gauntlet.rounds?.length || 0} rounds`}
                </p>
              </div>
              <input
                type="checkbox"
                checked={(formData.gauntlets || []).includes(gauntlet.id)}
                onChange={() => toggleGauntlet(gauntlet.id)}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function AdvancedSettingsStep({
  formData,
  updateFormData,
}: {
  formData: Partial<CreateWorkflowRequest>;
  updateFormData: (field: string, value: any) => void;
}) {
  const metadata = formData.metadata || ({} as WorkflowMetadata);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Advanced Settings
        </h2>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Configure optional advanced features
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium text-gray-900 dark:text-white">
              Enable MDAP
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Multi-Directional Anthropic Progression
            </p>
          </div>
          <input
            type="checkbox"
            checked={metadata.mdap_enabled || false}
            onChange={(e) =>
              updateFormData('metadata', {
                ...metadata,
                mdap_enabled: e.target.checked,
              })
            }
            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium text-gray-900 dark:text-white">
              Enable MAKER
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Multi-Agent Knowledge Evolution & Refinement
            </p>
          </div>
          <input
            type="checkbox"
            checked={metadata.maker_enabled || false}
            onChange={(e) =>
              updateFormData('metadata', {
                ...metadata,
                maker_enabled: e.target.checked,
              })
            }
            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
        </div>
      </div>
    </div>
  );
}
