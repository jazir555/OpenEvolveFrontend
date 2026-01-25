/**
 * Mitosis Plugin for BubbleLab
 *
 * Integrates the mitosis bubble splitting animation as a plugin within BubbleLab's visualization system.
 */

import { MitosisAnimation, MitosisSettings, MitosisDemo } from './components';
import { mitosisPlugin } from './utils/createMitosisPlugin';
import { EvolutionEventManager, connectToOpenEvolve, disconnectFromOpenEvolve, processOpenEvolveResult } from './openevolve-evolution-integration';
import type { OpenEvolvePlugin } from '@openevolve/plugin';

// Define the BubbleLab plugin interface
export interface BubbleLabPlugin {
  id: string;
  name: string;
  version: string;
  description: string;
  icon?: string;
  author?: string;
  website?: string;
  documentation?: string;
  capabilities: Record<string, boolean>;
  components: Record<string, React.ComponentType<any>>;
  settingsComponent?: React.ComponentType<any>;
  routes?: Array<{ path: string; component: string; title: string }>;
  services?: string[];
  apiEndpoints?: Record<string, string>;
  configSchema?: any;
  initialize?: (config?: any) => void;
  destroy?: () => void;
  hooks?: {
    onBeforeExecute?: (serviceId: string, params: any) => void;
    onAfterExecute?: (serviceId: string, result: any) => void;
    onError?: (serviceId: string, error: Error) => void;
  };
}

// Create the BubbleLab-compatible plugin
export const MitosisPlugin: BubbleLabPlugin = {
  id: 'mitosis-animation',
  name: 'Mitosis Bubble Splitting',
  version: '1.0.0',
  description: 'Adds cell-division-like animations to visualize OpenEvolve evolutions',
  icon: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z', // SVG path for a bubble/bubble-split icon
  author: 'OpenEvolve',
  website: 'https://openevolve.ai',
  documentation: 'https://docs.openevolve.ai/mitosis-plugin',

  // Capabilities this plugin provides
  capabilities: {
    visualization: true,
    animation: true,
    evolution: true,
    ui: true,
  },

  // UI components provided by this plugin
  components: {
    MitosisAnimation,
    MitosisSettings,
    MitosisDemo,
  },

  settingsComponent: MitosisSettings,

  // Routes provided by this plugin (if any)
  routes: [
    // No dedicated routes, integrates into existing visualization areas
  ],

  // Services provided by this plugin
  services: [
    'mitosis-animation',
    'bubble-splitting',
    'evolution-visualization',
  ],

  // API endpoints (if any)
  apiEndpoints: {
    // No API endpoints, operates client-side
  },

  // Configuration schema
  configSchema: {
    // Configuration handled through the settings component
  },

  // Initialization function
  initialize: (config?: any) => {
    try {
      // Initialize the mitosis plugin with default or provided configuration
      const defaultConfig = {
        enabled: false,
        animationDuration: 1500,
        bounceIntensity: 0.3,
        splitDelay: 300,
        colorVariation: 0.1,
        rotationIntensity: 0.2,
        opacityEffect: true,
        trailEffect: false,
        easingFunction: 'cubic-bezier(0.25, 0.1, 0.25, 1)',
        particleEffects: false
      };

      const finalConfig = config ? { ...defaultConfig, ...config } : defaultConfig;
      mitosisPlugin.initialize(finalConfig);

      console.log('[MitosisPlugin] Initialized successfully');
    } catch (error) {
      console.error('[MitosisPlugin] Error during initialization:', error);
    }
  },

  // Destroy/cleanup function
  destroy: () => {
    try {
      // Disconnect from OpenEvolve events if connected
      disconnectFromOpenEvolve();

      // Clean up mitosis plugin resources
      mitosisPlugin.cleanup();

      console.log('[MitosisPlugin] Destroyed successfully');
    } catch (error) {
      console.error('[MitosisPlugin] Error during destruction:', error);
    }
  },

  // Lifecycle hooks
  hooks: {
    onBeforeExecute: async (serviceId: string, params: any) => {
      console.log(`[MitosisPlugin] Preparing ${serviceId} execution`, params);
    },
    onAfterExecute: async (serviceId: string, result: any) => {
      console.log(`[MitosisPlugin] Completed ${serviceId} execution`, result);
    },
    onError: async (serviceId: string, error: Error) => {
      console.error(`[MitosisPlugin] Error in ${serviceId}`, error);
    },
  },
};

// Function to connect to OpenEvolve evolution events
export const connectToOpenEvolveEvolution = async (plugin: OpenEvolvePlugin) => {
  await connectToOpenEvolve(plugin);
};

// Function to disconnect from OpenEvolve evolution events
export const disconnectFromOpenEvolveEvolution = () => {
  disconnectFromOpenEvolve();
};

// Function to process OpenEvolve execution results
export const processOpenEvolveExecutionResult = async (result: any) => {
  await processOpenEvolveResult(result);
};

// Export the evolution event manager for advanced usage
export { EvolutionEventManager };

export default MitosisPlugin;