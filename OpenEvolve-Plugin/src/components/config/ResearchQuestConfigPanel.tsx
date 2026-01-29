/**
 * Research Quest Configuration Panel
 *
 * Configuration panel for Research Quest node settings
 */

import React, { useState, useEffect } from 'react';
import { ResearchQuestNodeConfig } from '../../nodes/ResearchQuestNode';

interface ResearchQuestConfigPanelProps {
  config: ResearchQuestNodeConfig;
  onUpdate: (config: ResearchQuestNodeConfig) => void;
}

const ResearchQuestConfigPanel: React.FC<ResearchQuestConfigPanelProps> = ({ config, onUpdate }) => {
  const [localConfig, setLocalConfig] = useState<ResearchQuestNodeConfig>({
    enableMultiLayer: true,
    maxHypotheses: 5,
    enableCausalInference: true,
    enableTemporalAnalysis: true,
    enableBiasAssessment: true,
    enableFalsificationChecks: true,
    enableImpactScoring: true,
    enableInterdisciplinaryBridges: true,
    enableKnowledgeGapDetection: true,
    enableProbabilisticConfidence: true,
    enableGraphRestructuring: true,
    enableTopologyAnalysis: true,
    enableInformationTheoryMetrics: true,
    enableAttributionTracking: true,
    enableStatisticalPowerAnalysis: true,
    enableMultiScaleAnalysis: true,
    enableCostEstimation: true,
    enableSelfAudit: true,
    enableEvidenceIntegration: true,
    enablePruningMerging: true,
    enableSubgraphExtraction: true,
    enableReflection: true,
    enableComposition: true,
    enableHypothesisGeneration: true,
    enableTaskDecomposition: true,
    enableInitialization: true,
    enableBackendExecution: true,
    backendUrl: 'http://localhost:8000',
    parameters: {
      P1_0: true,
      P1_1: true,
      P1_2: true,
      P1_3: true,
      P1_4: true,
      P1_5: true,
      P1_6: true,
      P1_7: true,
      P1_8: true,
      P1_9: true,
      P1_10: true,
      P1_11: true,
      P1_12: true,
      P1_13: true,
      P1_14: true,
      P1_15: true,
      P1_16: true,
      P1_17: true,
      P1_18: true,
      P1_19: true,
      P1_20: true,
      P1_21: true,
      P1_22: true,
      P1_23: true,
      P1_24: true,
      P1_25: true,
      P1_26: true,
      P1_27: true,
      P1_28: true,
      P1_29: true,
    },
    ...config
  });

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setLocalConfig(prev => ({
        ...prev,
        ...config
      }));
    } catch (effectError) {
      setError('Error loading configuration');
      errorLogger.logError(effectError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'ResearchQuestConfigPanel effect error' } });
    }
  }, [config]);

  const handleChange = (field: keyof ResearchQuestNodeConfig, value: any) => {
    try {
      const updatedConfig = { ...localConfig, [field]: value };
      setLocalConfig(updatedConfig);
      onUpdate(updatedConfig);
    } catch (updateError) {
      setError('Error updating configuration');
      errorLogger.logError(updateError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Configuration update error' } });
    }
  };

  const handleParameterChange = (param: string, value: boolean) => {
    try {
      const updatedParameters = {
        ...(localConfig.parameters || {}),
        [param]: value
      };
      const updatedConfig = { ...localConfig, parameters: updatedParameters };
      setLocalConfig(updatedConfig);
      onUpdate(updatedConfig);
    } catch (paramError) {
      setError('Error updating parameter');
      errorLogger.logError(paramError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Parameter update error' } });
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Research Quest Configuration</h3>

      {/* Error display */}
      {error && (
        <div className="p-3 bg-red-100 text-red-700 text-sm rounded-md">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Max Hypotheses
          </label>
          <input
            type="number"
            min="3"
            max="5"
            value={localConfig.maxHypotheses}
            onChange={(e) => {
              try {
                const value = parseInt(e.target.value);
                if (!isNaN(value)) {
                  handleChange('maxHypotheses', value);
                }
              } catch (inputError) {
                setError('Error updating max hypotheses');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Max hypotheses input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Backend URL
          </label>
          <input
            type="text"
            value={localConfig.backendUrl}
            onChange={(e) => {
              try {
                handleChange('backendUrl', e.target.value);
              } catch (inputError) {
                setError('Error updating backend URL');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend URL input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableMultiLayer"
            checked={localConfig.enableMultiLayer}
            onChange={(e) => {
              try {
                handleChange('enableMultiLayer', e.target.checked);
              } catch (inputError) {
                setError('Error updating multi-layer setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Multi-layer checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableMultiLayer" className="ml-2 block text-sm text-gray-900">
            Multi-Layer
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableCausalInference"
            checked={localConfig.enableCausalInference}
            onChange={(e) => {
              try {
                handleChange('enableCausalInference', e.target.checked);
              } catch (inputError) {
                setError('Error updating causal inference setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Causal inference checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableCausalInference" className="ml-2 block text-sm text-gray-900">
            Causal Inference
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableTemporalAnalysis"
            checked={localConfig.enableTemporalAnalysis}
            onChange={(e) => {
              try {
                handleChange('enableTemporalAnalysis', e.target.checked);
              } catch (inputError) {
                setError('Error updating temporal analysis setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Temporal analysis checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableTemporalAnalysis" className="ml-2 block text-sm text-gray-900">
            Temporal Analysis
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableBiasAssessment"
            checked={localConfig.enableBiasAssessment}
            onChange={(e) => {
              try {
                handleChange('enableBiasAssessment', e.target.checked);
              } catch (inputError) {
                setError('Error updating bias assessment setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Bias assessment checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableBiasAssessment" className="ml-2 block text-sm text-gray-900">
            Bias Assessment
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableFalsificationChecks"
            checked={localConfig.enableFalsificationChecks}
            onChange={(e) => {
              try {
                handleChange('enableFalsificationChecks', e.target.checked);
              } catch (inputError) {
                setError('Error updating falsification checks setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Falsification checks checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableFalsificationChecks" className="ml-2 block text-sm text-gray-900">
            Falsification Checks
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableImpactScoring"
            checked={localConfig.enableImpactScoring}
            onChange={(e) => {
              try {
                handleChange('enableImpactScoring', e.target.checked);
              } catch (inputError) {
                setError('Error updating impact scoring setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Impact scoring checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableImpactScoring" className="ml-2 block text-sm text-gray-900">
            Impact Scoring
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableInterdisciplinaryBridges"
            checked={localConfig.enableInterdisciplinaryBridges}
            onChange={(e) => {
              try {
                handleChange('enableInterdisciplinaryBridges', e.target.checked);
              } catch (inputError) {
                setError('Error updating interdisciplinary bridges setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Interdisciplinary bridges checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableInterdisciplinaryBridges" className="ml-2 block text-sm text-gray-900">
            Interdisciplinary Bridges
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableKnowledgeGapDetection"
            checked={localConfig.enableKnowledgeGapDetection}
            onChange={(e) => {
              try {
                handleChange('enableKnowledgeGapDetection', e.target.checked);
              } catch (inputError) {
                setError('Error updating knowledge gap detection setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Knowledge gap detection checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableKnowledgeGapDetection" className="ml-2 block text-sm text-gray-900">
            Knowledge Gap Detection
          </label>
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableBackendExecution"
            checked={localConfig.enableBackendExecution}
            onChange={(e) => {
              try {
                handleChange('enableBackendExecution', e.target.checked);
              } catch (inputError) {
                setError('Error updating backend execution setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend execution checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableBackendExecution" className="ml-2 block text-sm text-gray-900">
            Backend Execution
          </label>
        </div>
      </div>

      <div className="border-t border-gray-200 pt-4">
        <h4 className="text-md font-medium text-gray-800 mb-3">Research Parameters (P1.0-P1.29)</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {Object.entries(localConfig.parameters || {}).map(([param, value]) => (
            <div key={param} className="flex items-center">
              <input
                type="checkbox"
                id={param}
                checked={value}
                onChange={(e) => {
                  try {
                    handleParameterChange(param, e.target.checked);
                  } catch (inputError) {
                    setError('Error updating parameter');
                    console.error(`Parameter ${param} checkbox error:`, inputError);
                  }
                }}
                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
              />
              <label htmlFor={param} className="ml-2 block text-xs text-gray-700">
                {param.replace('_', '.')}
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResearchQuestConfigPanel;