/**
 * AdversarialConfigPanel.tsx
 *
 * Configuration panel for adversarial testing and red teaming capabilities
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
import {
  Shield,
  Sword,
  Target,
  AlertTriangle,
  Settings,
  Lock,
} from 'lucide-react';

export interface AdversarialConfig {
  // Attack strategies
  attackStrategy: 'pgd' | 'fgsm' | 'cw' | 'deepfool' | 'boundary' | 'genetic';
  attackStrength: number;
  stepSize: number;
  numSteps: number;

  // Target configuration
  targetedAttack: boolean;
  targetConfidence: number;
  randomTargets: boolean;

  // Red team configuration
  redTeamSize: number;
  redTeamStrategy: 'coordinated' | 'independent' | 'competitive';
  redTeamCommunication: boolean;

  // Blue team configuration
  blueTeamSize: number;
  blueTeamStrategy: 'static' | 'adaptive' | 'proactive';
  blueTeamLearning: boolean;

  // Defense mechanisms
  adversarialTraining: boolean;
  inputSanitization: boolean;
  outputValidation: boolean;
  anomalyDetection: boolean;
  defenseDiversity: number;

  // Tournament settings
  maxRounds: number;
  roundTimeout: number;
  victoryCondition: 'score' | 'survival' | 'objective';
  victoryThreshold: number;

  // Evaluation metrics
  successRateThreshold: number;
  robustnessScore: number;
  coverageTarget: number;

  // Advanced settings
  transferAttack: boolean;
  ensembleAttack: boolean;
  queryEfficiency: boolean;
  maxQueries: number;

  // Content type
  contentType: 'code' | 'text' | 'design' | 'strategy' | 'all';
}

interface AdversarialConfigPanelProps {
  config: AdversarialConfig;
  onConfigChange: (config: AdversarialConfig) => void;
}

const DEFAULT_CONFIG: AdversarialConfig = {
  attackStrategy: 'pgd',
  attackStrength: 0.1,
  stepSize: 0.01,
  numSteps: 40,
  targetedAttack: false,
  targetConfidence: 0.9,
  randomTargets: false,
  redTeamSize: 5,
  redTeamStrategy: 'coordinated',
  redTeamCommunication: true,
  blueTeamSize: 3,
  blueTeamStrategy: 'adaptive',
  blueTeamLearning: true,
  adversarialTraining: true,
  inputSanitization: true,
  outputValidation: true,
  anomalyDetection: true,
  defenseDiversity: 3,
  maxRounds: 10,
  roundTimeout: 300,
  victoryCondition: 'score',
  victoryThreshold: 0.8,
  successRateThreshold: 0.7,
  robustnessScore: 0.6,
  coverageTarget: 0.9,
  transferAttack: false,
  ensembleAttack: false,
  queryEfficiency: false,
  maxQueries: 1000,
  contentType: 'all',
};

const AdversarialConfigPanelBase: React.FC<AdversarialConfigPanelProps> = ({
  config,
  onConfigChange,
}) => {
  const [localConfig, setLocalConfig] = useState<AdversarialConfig>(config);
  const [activeSection, setActiveSection] = useState<
    'attacks' | 'redteam' | 'blueteam' | 'defenses' | 'tournament' | 'advanced'
  >('attacks');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setLocalConfig(config);
    setHasChanges(false);
  }, [config]);

  const handleFieldChange = <K extends keyof AdversarialConfig>(
    field: K,
    value: AdversarialConfig[K]
  ) => {
    const newConfig = { ...localConfig, [field]: value };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = () => {
    try {
      onConfigChange(localConfig);
      setHasChanges(false);
      toast.success('Adversarial configuration saved successfully');
    } catch (error) {
      toast.error(`Failed to save configuration: ${error instanceof Error ? error.message : String(error)}`);
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
    { id: 'attacks', label: 'Attack Strategies', icon: <Sword className="w-4 h-4" /> },
    { id: 'redteam', label: 'Red Team', icon: <Target className="w-4 h-4" /> },
    { id: 'blueteam', label: 'Blue Team', icon: <Shield className="w-4 h-4" /> },
    { id: 'defenses', label: 'Defenses', icon: <Lock className="w-4 h-4" /> },
    { id: 'tournament', label: 'Tournament', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'advanced', label: 'Advanced', icon: <Settings className="w-4 h-4" /> },
  ] as const;

  return (
    <div className="adversarial-config-panel rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="border-b border-slate-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Shield className="mr-3 text-2xl" />
            <h2 className="text-xl font-bold text-slate-900">Adversarial Configuration</h2>
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
          {activeSection === 'attacks' && (
            <BubbleCard title="Attack Configuration" description="Configure adversarial strategies and target settings.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Attack Strategy">
                  <BubbleSelect
                    value={localConfig.attackStrategy}
                    onChange={(e) =>
                      handleFieldChange('attackStrategy', e.target.value as AdversarialConfig['attackStrategy'])
                    }
                  >
                    <option value="pgd">Projected Gradient Descent (PGD)</option>
                    <option value="fgsm">Fast Gradient Sign Method (FGSM)</option>
                    <option value="cw">Carlini and Wagner (C&W)</option>
                    <option value="deepfool">DeepFool</option>
                    <option value="boundary">Boundary Attack</option>
                    <option value="genetic">Genetic Algorithm Attack</option>
                  </BubbleSelect>
                </BubbleField>

                <BubbleField label="Content Type">
                  <BubbleSelect
                    value={localConfig.contentType}
                    onChange={(e) =>
                      handleFieldChange('contentType', e.target.value as AdversarialConfig['contentType'])
                    }
                  >
                    <option value="all">All Types</option>
                    <option value="code">Code</option>
                    <option value="text">Text</option>
                    <option value="design">Design</option>
                    <option value="strategy">Strategy</option>
                  </BubbleSelect>
                </BubbleField>

                <BubbleField label="Attack Strength" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={localConfig.attackStrength}
                    onChange={(e) =>
                      handleFieldChange('attackStrength', parseFloat(e.target.value) || 0)
                    }
                  />
                </BubbleField>

                <BubbleField label="Step Size" hint="Range: 0.001-0.1">
                  <BubbleInput
                    type="number"
                    step="0.001"
                    min={0.001}
                    max={0.1}
                    value={localConfig.stepSize}
                    onChange={(e) => handleFieldChange('stepSize', parseFloat(e.target.value) || 0.001)}
                  />
                </BubbleField>

                <BubbleField label="Number of Steps" hint="Range: 1-200">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={200}
                    value={localConfig.numSteps}
                    onChange={(e) => handleFieldChange('numSteps', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>

                <div className="space-y-3">
                  <BubbleToggle
                    checked={localConfig.targetedAttack}
                    onChange={(checked) => handleFieldChange('targetedAttack', checked)}
                    label="Targeted Attack"
                  />
                  {localConfig.targetedAttack && (
                    <>
                      <BubbleField label="Target Confidence" hint="Range: 0.0-1.0">
                        <BubbleInput
                          type="number"
                          step="0.05"
                          min={0}
                          max={1}
                          value={localConfig.targetConfidence}
                          onChange={(e) =>
                            handleFieldChange('targetConfidence', parseFloat(e.target.value) || 0)
                          }
                        />
                      </BubbleField>
                      <BubbleToggle
                        checked={localConfig.randomTargets}
                        onChange={(checked) => handleFieldChange('randomTargets', checked)}
                        label="Random Targets"
                      />
                    </>
                  )}
                </div>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'redteam' && (
            <BubbleCard title="Red Team" description="Configure offensive team composition and coordination.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Red Team Size" hint="Range: 1-20">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={20}
                    value={localConfig.redTeamSize}
                    onChange={(e) => handleFieldChange('redTeamSize', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleField label="Red Team Strategy">
                  <BubbleSelect
                    value={localConfig.redTeamStrategy}
                    onChange={(e) =>
                      handleFieldChange('redTeamStrategy', e.target.value as AdversarialConfig['redTeamStrategy'])
                    }
                  >
                    <option value="coordinated">Coordinated</option>
                    <option value="independent">Independent</option>
                    <option value="competitive">Competitive</option>
                  </BubbleSelect>
                </BubbleField>
                <BubbleToggle
                  checked={localConfig.redTeamCommunication}
                  onChange={(checked) => handleFieldChange('redTeamCommunication', checked)}
                  label="Red Team Communication"
                />
              </div>
            </BubbleCard>
          )}

          {activeSection === 'blueteam' && (
            <BubbleCard title="Blue Team" description="Configure defensive team strategy and learning.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Blue Team Size" hint="Range: 1-20">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={20}
                    value={localConfig.blueTeamSize}
                    onChange={(e) => handleFieldChange('blueTeamSize', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleField label="Blue Team Strategy">
                  <BubbleSelect
                    value={localConfig.blueTeamStrategy}
                    onChange={(e) =>
                      handleFieldChange('blueTeamStrategy', e.target.value as AdversarialConfig['blueTeamStrategy'])
                    }
                  >
                    <option value="static">Static</option>
                    <option value="adaptive">Adaptive</option>
                    <option value="proactive">Proactive</option>
                  </BubbleSelect>
                </BubbleField>
                <BubbleToggle
                  checked={localConfig.blueTeamLearning}
                  onChange={(checked) => handleFieldChange('blueTeamLearning', checked)}
                  label="Blue Team Learning"
                />
              </div>
            </BubbleCard>
          )}

          {activeSection === 'defenses' && (
            <BubbleCard title="Defense Mechanisms" description="Enable and tune defense layers.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.adversarialTraining}
                  onChange={(checked) => handleFieldChange('adversarialTraining', checked)}
                  label="Adversarial Training"
                />
                <BubbleToggle
                  checked={localConfig.inputSanitization}
                  onChange={(checked) => handleFieldChange('inputSanitization', checked)}
                  label="Input Sanitization"
                />
                <BubbleToggle
                  checked={localConfig.outputValidation}
                  onChange={(checked) => handleFieldChange('outputValidation', checked)}
                  label="Output Validation"
                />
                <BubbleToggle
                  checked={localConfig.anomalyDetection}
                  onChange={(checked) => handleFieldChange('anomalyDetection', checked)}
                  label="Anomaly Detection"
                />
                <BubbleField label="Defense Diversity" hint="Range: 1-10">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={10}
                    value={localConfig.defenseDiversity}
                    onChange={(e) => handleFieldChange('defenseDiversity', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'tournament' && (
            <BubbleCard title="Tournament Settings" description="Configure red/blue team evaluation rounds.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Max Rounds" hint="Range: 1-100">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={100}
                    value={localConfig.maxRounds}
                    onChange={(e) => handleFieldChange('maxRounds', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleField label="Round Timeout (seconds)" hint="Range: 60-3600">
                  <BubbleInput
                    type="number"
                    min={60}
                    max={3600}
                    value={localConfig.roundTimeout}
                    onChange={(e) => handleFieldChange('roundTimeout', parseInt(e.target.value, 10) || 60)}
                  />
                </BubbleField>
                <BubbleField label="Victory Condition">
                  <BubbleSelect
                    value={localConfig.victoryCondition}
                    onChange={(e) =>
                      handleFieldChange('victoryCondition', e.target.value as AdversarialConfig['victoryCondition'])
                    }
                  >
                    <option value="score">Score-Based</option>
                    <option value="survival">Survival</option>
                    <option value="objective">Objective Completion</option>
                  </BubbleSelect>
                </BubbleField>
                <BubbleField label="Victory Threshold" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.victoryThreshold}
                    onChange={(e) => handleFieldChange('victoryThreshold', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleField label="Success Rate Threshold" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.successRateThreshold}
                    onChange={(e) => handleFieldChange('successRateThreshold', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleField label="Robustness Score" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.robustnessScore}
                    onChange={(e) => handleFieldChange('robustnessScore', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleField label="Coverage Target" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.coverageTarget}
                    onChange={(e) => handleFieldChange('coverageTarget', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'advanced' && (
            <BubbleCard title="Advanced Settings" description="Optimization and efficiency options.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.transferAttack}
                  onChange={(checked) => handleFieldChange('transferAttack', checked)}
                  label="Transfer Attack"
                />
                <BubbleToggle
                  checked={localConfig.ensembleAttack}
                  onChange={(checked) => handleFieldChange('ensembleAttack', checked)}
                  label="Ensemble Attack"
                />
                <BubbleToggle
                  checked={localConfig.queryEfficiency}
                  onChange={(checked) => handleFieldChange('queryEfficiency', checked)}
                  label="Query Efficiency Mode"
                />
                {localConfig.queryEfficiency && (
                  <BubbleField label="Max Queries" hint="Range: 1-10000">
                    <BubbleInput
                      type="number"
                      min={1}
                      max={10000}
                      value={localConfig.maxQueries}
                      onChange={(e) => handleFieldChange('maxQueries', parseInt(e.target.value, 10) || 1)}
                    />
                  </BubbleField>
                )}
              </div>
            </BubbleCard>
          )}
        </div>
      </div>
    </div>
  );
};

export const AdversarialConfigPanel = withComponentBoundary(
  AdversarialConfigPanelBase,
  'AdversarialConfigPanel'
);

export default AdversarialConfigPanel;
