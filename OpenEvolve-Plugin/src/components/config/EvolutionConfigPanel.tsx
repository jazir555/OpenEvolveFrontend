/**
 * EvolutionConfigPanel.tsx
 *
 * Configuration panel for genetic algorithm and evolutionary optimization settings
 * in the OpenEvolve plugin.
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { IconWrapper } from '../icons/IconWrapper';
import {
  BubbleBadge,
  BubbleButton,
  BubbleCard,
  BubbleField,
  BubbleInput,
  BubbleSelect,
  BubbleToggle,
} from '@/components/bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import {
  Dna,
  Brain,
  Zap,
  TrendingUp,
  Settings,
} from 'lucide-react';

export interface EvolutionConfig {
  // Population parameters
  populationSize: number;
  generations: number;
  elitismCount: number;

  // Genetic operators
  mutationRate: number;
  crossoverRate: number;
  mutationStrength: number;

  // Selection methods
  selectionMethod: 'tournament' | 'roulette' | 'rank' | 'steady_state';
  tournamentSize: number;
  selectionPressure: number;

  // Convergence and termination
  convergenceThreshold: number;
  maxIterations: number;
  stagnationGenerations: number;

  // Diversity management
  diversityThreshold: number;
  nichingEnabled: boolean;
  crowdingDistance: number;

  // Advanced settings
  adaptiveMutation: boolean;
  adaptiveCrossover: boolean;
  multiObjectiveOptimization: boolean;
  paretoFrontSize: number;

  // Constraint handling
  penaltyFactor: number;
  constraintHandling: 'death' | 'penalty' | 'repair';

  // Parallelization
  parallelEvaluation: boolean;
  evaluationBatchSize: number;
  asyncEvaluation: boolean;
}

interface EvolutionConfigPanelProps {
  config?: EvolutionConfig;
  onConfigChange?: (config: EvolutionConfig) => void;
}

const DEFAULT_CONFIG: EvolutionConfig = {
  populationSize: 100,
  generations: 50,
  elitismCount: 2,
  mutationRate: 0.1,
  crossoverRate: 0.8,
  mutationStrength: 0.5,
  selectionMethod: 'tournament',
  tournamentSize: 3,
  selectionPressure: 2.0,
  convergenceThreshold: 0.001,
  maxIterations: 1000,
  stagnationGenerations: 10,
  diversityThreshold: 0.2,
  nichingEnabled: false,
  crowdingDistance: 0.5,
  adaptiveMutation: true,
  adaptiveCrossover: true,
  multiObjectiveOptimization: false,
  paretoFrontSize: 50,
  penaltyFactor: 1000,
  constraintHandling: 'penalty',
  parallelEvaluation: true,
  evaluationBatchSize: 10,
  asyncEvaluation: false,
};

const EvolutionConfigPanelBase: React.FC<EvolutionConfigPanelProps> = ({
  config = DEFAULT_CONFIG,
  onConfigChange = () => {},
}) => {
  const [localConfig, setLocalConfig] = useState<EvolutionConfig>(config);
  const [activeSection, setActiveSection] = useState<
    'population' | 'operators' | 'selection' | 'convergence' | 'diversity' | 'advanced'
  >('population');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setLocalConfig(config);
    setHasChanges(false);
  }, [config]);

  const handleFieldChange = <K extends keyof EvolutionConfig>(
    field: K,
    value: EvolutionConfig[K]
  ) => {
    const newConfig = { ...localConfig, [field]: value };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      onConfigChange(localConfig);
      setHasChanges(false);
      toast.success('Evolution configuration saved successfully');
      return true;
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'EvolutionConfigPanel',
        function: 'handleSave',
        operation: 'SAVE_EVOLUTION_CONFIG',
        additionalData: { configSize: Object.keys(localConfig).length }
      }
    });

    if (!result.success) {
      toast.error(`Failed to save configuration: ${result.error?.message || 'Unknown error'}`);
    }
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset to default configuration?')) {
      setLocalConfig(DEFAULT_CONFIG);
      setHasChanges(true);
      toast.info('Configuration reset to defaults. Click Save to apply.');
    }
  };

  const handleDiscard = () => {
    setLocalConfig(config);
    setHasChanges(false);
    toast.info('Changes discarded');
  };

  const sections = [
    { id: 'population', label: 'Population', icon: <Dna className="w-4 h-4" /> },
    { id: 'operators', label: 'Genetic Operators', icon: <Zap className="w-4 h-4" /> },
    { id: 'selection', label: 'Selection', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'convergence', label: 'Convergence', icon: <Settings className="w-4 h-4" /> },
    { id: 'diversity', label: 'Diversity', icon: <Brain className="w-4 h-4" /> },
    { id: 'advanced', label: 'Advanced', icon: <Settings className="w-4 h-4" /> },
  ] as const;

  return (
    <div className="evolution-config-panel rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="border-b border-slate-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Dna className="mr-3 text-2xl" />
            <h2 className="text-xl font-bold text-slate-900">Evolution Configuration</h2>
          </div>
          <div className="flex items-center gap-2">
            {hasChanges && <BubbleBadge tone="warning">Unsaved Changes</BubbleBadge>}
            <BubbleButton onClick={handleSave} disabled={!hasChanges}>
              Save
            </BubbleButton>
            <BubbleButton onClick={handleDiscard} disabled={!hasChanges} variant="secondary">
              Discard
            </BubbleButton>
            <BubbleButton onClick={handleReset} variant="secondary">
              Reset to Defaults
            </BubbleButton>
          </div>
        </div>
      </div>

      <div className="flex">
        <aside className="w-64 border-r border-slate-200 bg-slate-50">
          <nav className="p-4 space-y-2">
            {sections.map((section) => (
              <BubbleButton
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                variant={activeSection === section.id ? 'primary' : 'secondary'}
                className="w-full justify-start gap-3"
              >
                <span>{section.icon}</span>
                <span>{section.label}</span>
              </BubbleButton>
            ))}
          </nav>
        </aside>

        <div className="flex-1 p-6">
          {activeSection === 'population' && (
            <BubbleCard
              title="Population Parameters"
              description="Population size and generation limits for the evolutionary run."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Population Size" hint="Range: 10-1000">
                  <BubbleInput
                    type="number"
                    min={10}
                    max={1000}
                    value={localConfig.populationSize}
                    onChange={(e) =>
                      handleFieldChange('populationSize', parseInt(e.target.value, 10) || 10)
                    }
                  />
                </BubbleField>
                <BubbleField label="Max Generations" hint="Range: 1-500">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={500}
                    value={localConfig.generations}
                    onChange={(e) =>
                      handleFieldChange('generations', parseInt(e.target.value, 10) || 1)
                    }
                  />
                </BubbleField>
                <BubbleField label="Elitism Count" hint="Range: 0-20">
                  <BubbleInput
                    type="number"
                    min={0}
                    max={20}
                    value={localConfig.elitismCount}
                    onChange={(e) =>
                      handleFieldChange('elitismCount', parseInt(e.target.value, 10) || 0)
                    }
                  />
                </BubbleField>
                <BubbleField label="Max Total Iterations" hint="Range: 1-10000">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={10000}
                    value={localConfig.maxIterations}
                    onChange={(e) =>
                      handleFieldChange('maxIterations', parseInt(e.target.value, 10) || 1)
                    }
                  />
                </BubbleField>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'operators' && (
            <BubbleCard
              title="Genetic Operators"
              description="Mutation and crossover controls for evolution."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Mutation Rate" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={localConfig.mutationRate}
                    onChange={(e) =>
                      handleFieldChange('mutationRate', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <BubbleField label="Crossover Rate" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={localConfig.crossoverRate}
                    onChange={(e) =>
                      handleFieldChange('crossoverRate', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <BubbleField label="Mutation Strength" hint="Range: 0.0-2.0">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min={0}
                    max={2}
                    value={localConfig.mutationStrength}
                    onChange={(e) =>
                      handleFieldChange('mutationStrength', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <div className="space-y-3">
                  <BubbleToggle
                    checked={localConfig.adaptiveMutation}
                    onChange={(checked) => handleFieldChange('adaptiveMutation', checked)}
                    label="Adaptive Mutation"
                  />
                  <BubbleToggle
                    checked={localConfig.adaptiveCrossover}
                    onChange={(checked) => handleFieldChange('adaptiveCrossover', checked)}
                    label="Adaptive Crossover"
                  />
                </div>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'selection' && (
            <BubbleCard
              title="Selection Methods"
              description="Choose how individuals are selected for reproduction."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Selection Method">
                  <BubbleSelect
                    value={localConfig.selectionMethod}
                    onChange={(e) =>
                      handleFieldChange(
                        'selectionMethod',
                        e.target.value as EvolutionConfig['selectionMethod']
                      )
                    }
                  >
                    <option value="tournament">Tournament Selection</option>
                    <option value="roulette">Roulette Wheel</option>
                    <option value="rank">Rank-Based Selection</option>
                    <option value="steady_state">Steady-State</option>
                  </BubbleSelect>
                </BubbleField>

                {localConfig.selectionMethod === 'tournament' && (
                  <BubbleField label="Tournament Size" hint="Range: 2-10">
                    <BubbleInput
                      type="number"
                      min={2}
                      max={10}
                      value={localConfig.tournamentSize}
                      onChange={(e) =>
                        handleFieldChange('tournamentSize', parseInt(e.target.value, 10) || 2)
                      }
                    />
                  </BubbleField>
                )}

                <BubbleField label="Selection Pressure" hint="Range: 1.0-5.0">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min={1}
                    max={5}
                    value={localConfig.selectionPressure}
                    onChange={(e) =>
                      handleFieldChange('selectionPressure', parseFloat(e.target.value) || 1)
                    }
                  />
                </BubbleField>
              </div>

              <div className="mt-6 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-600">
                <div className="font-semibold text-slate-700">Selection Method Guide</div>
                <ul className="mt-2 space-y-1">
                  <li>- Tournament: fast, good for parallelization.</li>
                  <li>- Roulette: proportional to fitness.</li>
                  <li>- Rank: stable selection pressure.</li>
                  <li>- Steady-State: replaces worst individuals.</li>
                </ul>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'convergence' && (
            <BubbleCard
              title="Convergence & Termination"
              description="Stop conditions and constraint handling."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Convergence Threshold" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.0001"
                    min={0}
                    max={1}
                    value={localConfig.convergenceThreshold}
                    onChange={(e) =>
                      handleFieldChange('convergenceThreshold', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <BubbleField label="Stagnation Limit" hint="Range: 1-100">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={100}
                    value={localConfig.stagnationGenerations}
                    onChange={(e) =>
                      handleFieldChange(
                        'stagnationGenerations',
                        parseInt(e.target.value, 10) || 1
                      )
                    }
                  />
                </BubbleField>
                <BubbleField label="Constraint Handling">
                  <BubbleSelect
                    value={localConfig.constraintHandling}
                    onChange={(e) =>
                      handleFieldChange(
                        'constraintHandling',
                        e.target.value as EvolutionConfig['constraintHandling']
                      )
                    }
                  >
                    <option value="penalty">Penalty Method</option>
                    <option value="death">Death Penalty</option>
                    <option value="repair">Repair Method</option>
                  </BubbleSelect>
                </BubbleField>
                {localConfig.constraintHandling === 'penalty' && (
                  <BubbleField label="Penalty Factor" hint="Range: 1-10000">
                    <BubbleInput
                      type="number"
                      min={1}
                      max={10000}
                      value={localConfig.penaltyFactor}
                      onChange={(e) =>
                        handleFieldChange('penaltyFactor', parseInt(e.target.value, 10) || 1)
                      }
                    />
                  </BubbleField>
                )}
              </div>
            </BubbleCard>
          )}

          {activeSection === 'diversity' && (
            <BubbleCard
              title="Diversity Management"
              description="Maintain population diversity to avoid premature convergence."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Diversity Threshold" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.diversityThreshold}
                    onChange={(e) =>
                      handleFieldChange('diversityThreshold', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <BubbleField label="Crowding Distance" hint="Range: 0.0-2.0">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min={0}
                    max={2}
                    value={localConfig.crowdingDistance}
                    onChange={(e) =>
                      handleFieldChange('crowdingDistance', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>
                <BubbleToggle
                  checked={localConfig.nichingEnabled}
                  onChange={(checked) => handleFieldChange('nichingEnabled', checked)}
                  label="Enable Niching"
                />
              </div>
            </BubbleCard>
          )}

          {activeSection === 'advanced' && (
            <BubbleCard
              title="Advanced Settings"
              description="Multi-objective optimization and parallelization options."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.multiObjectiveOptimization}
                  onChange={(checked) => handleFieldChange('multiObjectiveOptimization', checked)}
                  label="Multi-Objective Optimization"
                />
                {localConfig.multiObjectiveOptimization && (
                  <BubbleField label="Pareto Front Size" hint="Range: 10-200">
                    <BubbleInput
                      type="number"
                      min={10}
                      max={200}
                      value={localConfig.paretoFrontSize}
                      onChange={(e) =>
                        handleFieldChange('paretoFrontSize', parseInt(e.target.value, 10) || 10)
                      }
                    />
                  </BubbleField>
                )}
                <BubbleToggle
                  checked={localConfig.parallelEvaluation}
                  onChange={(checked) => handleFieldChange('parallelEvaluation', checked)}
                  label="Parallel Evaluation"
                />
                {localConfig.parallelEvaluation && (
                  <BubbleField label="Evaluation Batch Size" hint="Range: 1-100">
                    <BubbleInput
                      type="number"
                      min={1}
                      max={100}
                      value={localConfig.evaluationBatchSize}
                      onChange={(e) =>
                        handleFieldChange(
                          'evaluationBatchSize',
                          parseInt(e.target.value, 10) || 1
                        )
                      }
                    />
                  </BubbleField>
                )}
                <BubbleToggle
                  checked={localConfig.asyncEvaluation}
                  onChange={(checked) => handleFieldChange('asyncEvaluation', checked)}
                  label="Async Evaluation"
                />
              </div>
            </BubbleCard>
          )}
        </div>
      </div>
    </div>
  );
};

export const EvolutionConfigPanel = withComponentBoundary(
  EvolutionConfigPanelBase,
  'EvolutionConfigPanel'
);

export default EvolutionConfigPanel;
