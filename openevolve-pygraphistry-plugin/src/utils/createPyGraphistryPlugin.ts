import { PyGraphistryPlugin, PyGraphistryPluginState, PyGraphistryConfig, PyGraphistryVizOptions, CausalDiscoveryResult } from '../types/plugin-types';

let globalState: PyGraphistryPluginState = {
  config: {},
  lastVizUrl: null,
  isInitializing: false,
  error: null,
  features: {
    pygraphistryEnabled: true,
    causalDiscoveryEnabled: true,
    optimizationEnabled: true,
    uqEnabled: true,
    globalChemEnabled: true,
    curieEnabled: true,
    temporalGraphEnabled: true,
    onekeEnabled: true,
    leanAideEnabled: true,
    sopEnabled: true,
    adversarialEnabled: true,
    pamiEnabled: true,
    aceEnabled: true,
    romaEnabled: true,
    datapizzaEnabled: true,
    crewaiEnabled: true,
    claudiomiroEnabled: true,
    steerEnabled: true,
    researchQuestEnabled: true,
    kgEnabled: true,
    sgdEnabled: true,
    globalAnalyticsEnabled: true,
    mapElitesEnabled: true,
    verificationEnabled: true,
    problemAnalysisEnabled: true,
    dependencyEnabled: true,
    artifactGraphEnabled: true,
    sceEnabled: false,
    staticAnalysisEnabled: false,
    lltlEnabled: false,
    collaborationEnabled: false,
    workflowMonitorEnabled: false,
    lineageEnabled: false,
    gauntletEnabled: false,
    patternMiningEnabled: false,
    adaptationEnabled: false,
    ditoEnabled: false,
    crewaiEnabled: false,
    ragEnabled: false,
    deepkeEnabled: false,
    lean4Enabled: false,
    makerEnabled: false,
    mdapEnabled: false,
    mctsEnabled: false,
    hybridMCTSEnabled: false,
    e2ePlannerEnabled: false,
    evaluatorTeamEnabled: false,
    redTeamEnabled: false,
    blueTeamEnabled: false,
    qaSuiteEnabled: false,
    reseEnabled: false,
    materialKGEnabled: false,
    gnomeEnabled: false,
    physicsNemoEnabled: false,
    autogptEnabled: false,
    autogenEnabled: false,
    metagptEnabled: false,
    llm4iasEnabled: false,
    claraverseEnabled: false,
    aiScientistEnabled: false,
    uncertainpyEnabled: false,
    riskAnalyzerEnabled: false,
    karateclubEnabled: false,
    neuralKGEnabled: false,
    pylabrobotEnabled: false,
    pinnsEnabled: false
  }
};

interface CausalDiscoveryResponse extends CausalDiscoveryResult {
  url?: string;
}

class PyGraphistryService {
  async fetchVisualizationUrl(options: PyGraphistryVizOptions): Promise<string> {
    if (!globalState.features.pygraphistryEnabled) {
      throw new Error('PyGraphistry visualization is currently disabled.');
    }
    
    try {
      const response = await fetch('/api/openevolve/visualize/pygraphistry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...options,
          config: globalState.config
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate PyGraphistry visualization');
      }
      
      const data: { url: string } = await response.json();
      return data.url;
    } catch (err) {
      throw err;
    }
  }
}

export function createPyGraphistryPlugin(): PyGraphistryPlugin {
  const service = new PyGraphistryService();

  return {
    initialize: async (config: PyGraphistryConfig) => {
      globalState.isInitializing = true;
      globalState.config = config;
      // Initialize connection to backend
      globalState.isInitializing = false;
    },

    generateVisualization: async (options: PyGraphistryVizOptions) => {
      try {
        const url = await service.fetchVisualizationUrl(options);
        globalState.lastVizUrl = url;
        return url;
      } catch (err) {
        globalState.error = err instanceof Error ? err.message : String(err);
        return null;
      }
    },

    updateFeatures: (features: Partial<PyGraphistryPluginState['features']>) => {
      globalState.features = { ...globalState.features, ...features };
    },

    updateConfig: (config: Partial<PyGraphistryConfig>) => {
      globalState.config = { ...globalState.config, ...config };
    },

    getState: () => ({ ...globalState })
  };
}

export const pygraphistryPlugin = createPyGraphistryPlugin();
