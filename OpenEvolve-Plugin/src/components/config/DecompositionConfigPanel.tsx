/**
 * DecompositionConfigPanel.tsx
 *
 * Configuration panel for problem decomposition strategies
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
  PuzzlePiece,
  TreePine,
  Layers,
  Zap,
  Settings,
  Filter,
} from 'lucide-react';

export interface DecompositionConfig {
  // Decomposition strategy
  strategy: 'hierarchical' | 'semantic' | 'dependency_based' | 'temporal' | 'hybrid';
  maxDepth: number;
  recursionDepthLimit: number;
  maxSubProblems: number;
  granularity: 'fine' | 'medium' | 'coarse';

  // Size constraints
  minSubProblemSize: number;
  maxSubProblemSize: number;
  targetSubProblemSize: number;

  // Parallel processing
  parallelDecomposition: boolean;
  maxParallelTasks: number;
  asyncDecomposition: boolean;

  // Pruning and optimization
  pruningEnabled: boolean;
  pruningThreshold: number;
  similarityThreshold: number;
  mergeThreshold: number;

  // Semantic analysis (for semantic strategy)
  semanticSimilarity: 'cosine' | 'jaccard' | 'euclidean' | 'manhattan';
  embeddingModel: string;
  clusteringAlgorithm: 'kmeans' | 'dbscan' | 'hierarchical' | 'spectral';

  // Dependency analysis (for dependency-based strategy)
  dependencyDetection: 'static' | 'dynamic' | 'hybrid';
  circularDependencyHandling: 'break' | 'merge' | 'error';
  dependencyVisualization: boolean;

  // Temporal analysis (for temporal strategy)
  timeHorizon: number;
  timeGranularity: 'seconds' | 'minutes' | 'hours' | 'days';
  temporalDependencies: boolean;

  // Quality metrics
  cohesivenessTarget: number;
  couplingLimit: number;
  complexityThreshold: number;

  // Validation and verification
  validateDecomposition: boolean;
  testDecomposition: boolean;
  feedbackLoop: boolean;

  // Advanced settings
  adaptiveDecomposition: boolean;
  learningEnabled: boolean;
  historicalData: boolean;
}

interface DecompositionConfigPanelProps {
  config: DecompositionConfig;
  onConfigChange: (config: DecompositionConfig) => void;
}

const DEFAULT_CONFIG: DecompositionConfig = {
  strategy: 'hierarchical',
  maxDepth: 3,
  recursionDepthLimit: 1,
  maxSubProblems: 3,
  granularity: 'medium',
  minSubProblemSize: 50,
  maxSubProblemSize: 500,
  targetSubProblemSize: 200,
  parallelDecomposition: true,
  maxParallelTasks: 5,
  asyncDecomposition: false,
  pruningEnabled: true,
  pruningThreshold: 0.1,
  similarityThreshold: 0.8,
  mergeThreshold: 0.9,
  semanticSimilarity: 'cosine',
  embeddingModel: 'text-embedding-ada-002',
  clusteringAlgorithm: 'hierarchical',
  dependencyDetection: 'hybrid',
  circularDependencyHandling: 'merge',
  dependencyVisualization: false,
  timeHorizon: 24,
  timeGranularity: 'hours',
  temporalDependencies: true,
  cohesivenessTarget: 0.7,
  couplingLimit: 0.3,
  complexityThreshold: 0.8,
  validateDecomposition: true,
  testDecomposition: false,
  feedbackLoop: true,
  adaptiveDecomposition: false,
  learningEnabled: false,
  historicalData: false,
};

const DecompositionConfigPanelBase: React.FC<DecompositionConfigPanelProps> = ({
  config,
  onConfigChange,
}) => {
  const [localConfig, setLocalConfig] = useState<DecompositionConfig>(config);
  const [lastRecursionDepthLimit, setLastRecursionDepthLimit] = useState(
    config.recursionDepthLimit > 0 ? config.recursionDepthLimit : 1
  );
  const [lastMaxSubProblems, setLastMaxSubProblems] = useState(
    config.maxSubProblems > 0 ? config.maxSubProblems : 3
  );
  const [activeSection, setActiveSection] = useState<
    'strategy' | 'constraints' | 'parallel' | 'pruning' | 'quality' | 'advanced'
  >('strategy');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setLocalConfig(config);
    if (config.recursionDepthLimit > 0) {
      setLastRecursionDepthLimit(config.recursionDepthLimit);
    }
    if (config.maxSubProblems > 0) {
      setLastMaxSubProblems(config.maxSubProblems);
    }
    setHasChanges(false);
  }, [config]);

  const handleFieldChange = <K extends keyof DecompositionConfig>(
    field: K,
    value: DecompositionConfig[K]
  ) => {
    const newConfig = { ...localConfig, [field]: value };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = () => {
    try {
      onConfigChange(localConfig);
      setHasChanges(false);
      toast.success('Decomposition configuration saved successfully');
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
    { id: 'strategy', label: 'Strategy', icon: <PuzzlePiece className="w-4 h-4" /> },
    { id: 'constraints', label: 'Constraints', icon: <Layers className="w-4 h-4" /> },
    { id: 'parallel', label: 'Parallel', icon: <Zap className="w-4 h-4" /> },
    { id: 'pruning', label: 'Optimization', icon: <Filter className="w-4 h-4" /> },
    { id: 'quality', label: 'Quality', icon: <Settings className="w-4 h-4" /> },
    { id: 'advanced', label: 'Advanced', icon: <TreePine className="w-4 h-4" /> },
  ] as const;

  return (
    <div className="decomposition-config-panel rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="border-b border-slate-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <PuzzlePiece className="mr-3 text-2xl" />
            <h2 className="text-xl font-bold text-slate-900">Decomposition Configuration</h2>
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
          {activeSection === 'strategy' && (
            <BubbleCard title="Strategy" description="Choose the primary decomposition approach.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Decomposition Strategy">
                  <BubbleSelect
                    value={localConfig.strategy}
                    onChange={(e) => handleFieldChange('strategy', e.target.value as DecompositionConfig['strategy'])}
                  >
                    <option value="hierarchical">Hierarchical</option>
                    <option value="semantic">Semantic</option>
                    <option value="dependency_based">Dependency-Based</option>
                    <option value="temporal">Temporal</option>
                    <option value="hybrid">Hybrid</option>
                  </BubbleSelect>
                </BubbleField>

                <BubbleField label="Granularity">
                  <BubbleSelect
                    value={localConfig.granularity}
                    onChange={(e) => handleFieldChange('granularity', e.target.value as DecompositionConfig['granularity'])}
                  >
                    <option value="fine">Fine</option>
                    <option value="medium">Medium</option>
                    <option value="coarse">Coarse</option>
                  </BubbleSelect>
                </BubbleField>

                <BubbleField label="Max Depth" hint="Range: 1-20">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={20}
                    value={localConfig.maxDepth}
                    onChange={(e) => handleFieldChange('maxDepth', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>

                <BubbleField label="Recursion Depth Limit" hint="0 = unlimited">
                  <BubbleInput
                    type="number"
                    min={0}
                    max={20}
                    value={localConfig.recursionDepthLimit}
                    disabled={localConfig.recursionDepthLimit === 0}
                    onChange={(e) => {
                      const nextValue = parseInt(e.target.value, 10) || 0;
                      handleFieldChange('recursionDepthLimit', nextValue);
                      if (nextValue > 0) {
                        setLastRecursionDepthLimit(nextValue);
                      }
                    }}
                  />
                </BubbleField>

                <BubbleToggle
                  checked={localConfig.recursionDepthLimit === 0}
                  onChange={(checked) =>
                    handleFieldChange(
                      'recursionDepthLimit',
                      checked ? 0 : lastRecursionDepthLimit || 1
                    )
                  }
                  label="Unlimited Recursion Depth"
                />

                <BubbleField label="Max Sub-Problems" hint="0 = unlimited">
                  <BubbleInput
                    type="number"
                    min={0}
                    max={100}
                    value={localConfig.maxSubProblems}
                    disabled={localConfig.maxSubProblems === 0}
                    onChange={(e) => {
                      const nextValue = parseInt(e.target.value, 10) || 0;
                      handleFieldChange('maxSubProblems', nextValue);
                      if (nextValue > 0) {
                        setLastMaxSubProblems(nextValue);
                      }
                    }}
                  />
                </BubbleField>

                <BubbleToggle
                  checked={localConfig.maxSubProblems === 0}
                  onChange={(checked) =>
                    handleFieldChange(
                      'maxSubProblems',
                      checked ? 0 : lastMaxSubProblems || 3
                    )
                  }
                  label="Unlimited Sub-Problems"
                />
              </div>

              {localConfig.strategy === 'semantic' && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <BubbleField label="Similarity Metric">
                    <BubbleSelect
                      value={localConfig.semanticSimilarity}
                      onChange={(e) => handleFieldChange('semanticSimilarity', e.target.value as DecompositionConfig['semanticSimilarity'])}
                    >
                      <option value="cosine">Cosine</option>
                      <option value="jaccard">Jaccard</option>
                      <option value="euclidean">Euclidean</option>
                      <option value="manhattan">Manhattan</option>
                    </BubbleSelect>
                  </BubbleField>
                  <BubbleField label="Embedding Model">
                    <BubbleInput
                      type="text"
                      value={localConfig.embeddingModel}
                      onChange={(e) => handleFieldChange('embeddingModel', e.target.value)}
                    />
                  </BubbleField>
                  <BubbleField label="Clustering Algorithm">
                    <BubbleSelect
                      value={localConfig.clusteringAlgorithm}
                      onChange={(e) => handleFieldChange('clusteringAlgorithm', e.target.value as DecompositionConfig['clusteringAlgorithm'])}
                    >
                      <option value="kmeans">K-Means</option>
                      <option value="dbscan">DBSCAN</option>
                      <option value="hierarchical">Hierarchical</option>
                      <option value="spectral">Spectral</option>
                    </BubbleSelect>
                  </BubbleField>
                </div>
              )}

              {localConfig.strategy === 'dependency_based' && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <BubbleField label="Dependency Detection">
                    <BubbleSelect
                      value={localConfig.dependencyDetection}
                      onChange={(e) => handleFieldChange('dependencyDetection', e.target.value as DecompositionConfig['dependencyDetection'])}
                    >
                      <option value="static">Static</option>
                      <option value="dynamic">Dynamic</option>
                      <option value="hybrid">Hybrid</option>
                    </BubbleSelect>
                  </BubbleField>
                  <BubbleField label="Circular Dependencies">
                    <BubbleSelect
                      value={localConfig.circularDependencyHandling}
                      onChange={(e) => handleFieldChange('circularDependencyHandling', e.target.value as DecompositionConfig['circularDependencyHandling'])}
                    >
                      <option value="break">Break</option>
                      <option value="merge">Merge</option>
                      <option value="error">Error</option>
                    </BubbleSelect>
                  </BubbleField>
                  <BubbleToggle
                    checked={localConfig.dependencyVisualization}
                    onChange={(checked) => handleFieldChange('dependencyVisualization', checked)}
                    label="Dependency Visualization"
                  />
                </div>
              )}

              {localConfig.strategy === 'temporal' && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <BubbleField label="Time Horizon" hint="Hours or units based on granularity">
                    <BubbleInput
                      type="number"
                      min={1}
                      value={localConfig.timeHorizon}
                      onChange={(e) => handleFieldChange('timeHorizon', parseInt(e.target.value, 10) || 1)}
                    />
                  </BubbleField>
                  <BubbleField label="Time Granularity">
                    <BubbleSelect
                      value={localConfig.timeGranularity}
                      onChange={(e) => handleFieldChange('timeGranularity', e.target.value as DecompositionConfig['timeGranularity'])}
                    >
                      <option value="seconds">Seconds</option>
                      <option value="minutes">Minutes</option>
                      <option value="hours">Hours</option>
                      <option value="days">Days</option>
                    </BubbleSelect>
                  </BubbleField>
                  <BubbleToggle
                    checked={localConfig.temporalDependencies}
                    onChange={(checked) => handleFieldChange('temporalDependencies', checked)}
                    label="Temporal Dependencies"
                  />
                </div>
              )}
            </BubbleCard>
          )}

          {activeSection === 'constraints' && (
            <BubbleCard title="Size Constraints" description="Define size limits for sub-problems.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Min Sub-Problem Size">
                  <BubbleInput
                    type="number"
                    min={1}
                    value={localConfig.minSubProblemSize}
                    onChange={(e) => handleFieldChange('minSubProblemSize', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleField label="Max Sub-Problem Size">
                  <BubbleInput
                    type="number"
                    min={1}
                    value={localConfig.maxSubProblemSize}
                    onChange={(e) => handleFieldChange('maxSubProblemSize', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleField label="Target Sub-Problem Size">
                  <BubbleInput
                    type="number"
                    min={1}
                    value={localConfig.targetSubProblemSize}
                    onChange={(e) => handleFieldChange('targetSubProblemSize', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
              </div>
            </BubbleCard>
          )}

          {activeSection === 'parallel' && (
            <BubbleCard title="Parallel Processing" description="Scale decomposition across workers.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.parallelDecomposition}
                  onChange={(checked) => handleFieldChange('parallelDecomposition', checked)}
                  label="Parallel Decomposition"
                />
                <BubbleField label="Max Parallel Tasks" hint="Range: 1-20">
                  <BubbleInput
                    type="number"
                    min={1}
                    max={20}
                    value={localConfig.maxParallelTasks}
                    onChange={(e) => handleFieldChange('maxParallelTasks', parseInt(e.target.value, 10) || 1)}
                  />
                </BubbleField>
                <BubbleToggle
                  checked={localConfig.asyncDecomposition}
                  onChange={(checked) => handleFieldChange('asyncDecomposition', checked)}
                  label="Async Decomposition"
                />
              </div>
            </BubbleCard>
          )}

          {activeSection === 'pruning' && (
            <BubbleCard title="Optimization & Pruning" description="Reduce complexity with pruning rules.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.pruningEnabled}
                  onChange={(checked) => handleFieldChange('pruningEnabled', checked)}
                  label="Enable Pruning"
                />
                {localConfig.pruningEnabled && (
                  <>
                    <BubbleField label="Pruning Threshold" hint="Range: 0.0-1.0">
                      <BubbleInput
                        type="number"
                        step="0.05"
                        min={0}
                        max={1}
                        value={localConfig.pruningThreshold}
                        onChange={(e) => handleFieldChange('pruningThreshold', parseFloat(e.target.value) || 0)}
                      />
                    </BubbleField>
                    <BubbleField label="Similarity Threshold" hint="Range: 0.0-1.0">
                      <BubbleInput
                        type="number"
                        step="0.05"
                        min={0}
                        max={1}
                        value={localConfig.similarityThreshold}
                        onChange={(e) => handleFieldChange('similarityThreshold', parseFloat(e.target.value) || 0)}
                      />
                    </BubbleField>
                    <BubbleField label="Merge Threshold" hint="Range: 0.0-1.0">
                      <BubbleInput
                        type="number"
                        step="0.05"
                        min={0}
                        max={1}
                        value={localConfig.mergeThreshold}
                        onChange={(e) => handleFieldChange('mergeThreshold', parseFloat(e.target.value) || 0)}
                      />
                    </BubbleField>
                  </>
                )}
              </div>
            </BubbleCard>
          )}

          {activeSection === 'quality' && (
            <BubbleCard title="Quality Metrics" description="Set targets and validation options.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleField label="Cohesiveness Target" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.cohesivenessTarget}
                    onChange={(e) => handleFieldChange('cohesivenessTarget', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleField label="Coupling Limit" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.couplingLimit}
                    onChange={(e) => handleFieldChange('couplingLimit', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleField label="Complexity Threshold" hint="Range: 0.0-1.0">
                  <BubbleInput
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={localConfig.complexityThreshold}
                    onChange={(e) => handleFieldChange('complexityThreshold', parseFloat(e.target.value) || 0)}
                  />
                </BubbleField>
                <BubbleToggle
                  checked={localConfig.validateDecomposition}
                  onChange={(checked) => handleFieldChange('validateDecomposition', checked)}
                  label="Validate Decomposition"
                />
                <BubbleToggle
                  checked={localConfig.testDecomposition}
                  onChange={(checked) => handleFieldChange('testDecomposition', checked)}
                  label="Test Decomposition"
                />
                <BubbleToggle
                  checked={localConfig.feedbackLoop}
                  onChange={(checked) => handleFieldChange('feedbackLoop', checked)}
                  label="Feedback Loop"
                />
              </div>
            </BubbleCard>
          )}

          {activeSection === 'advanced' && (
            <BubbleCard title="Advanced Features" description="Experimental and learning-based options.">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <BubbleToggle
                  checked={localConfig.adaptiveDecomposition}
                  onChange={(checked) => handleFieldChange('adaptiveDecomposition', checked)}
                  label="Adaptive Decomposition"
                />
                <BubbleToggle
                  checked={localConfig.learningEnabled}
                  onChange={(checked) => handleFieldChange('learningEnabled', checked)}
                  label="Machine Learning"
                />
                <BubbleToggle
                  checked={localConfig.historicalData}
                  onChange={(checked) => handleFieldChange('historicalData', checked)}
                  label="Use Historical Data"
                />
              </div>
            </BubbleCard>
          )}
        </div>
      </div>
    </div>
  );
};

export const DecompositionConfigPanel = withComponentBoundary(
  DecompositionConfigPanelBase,
  'DecompositionConfigPanel'
);

export default DecompositionConfigPanel;
