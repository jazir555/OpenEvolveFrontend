/**
 * Integration file for BubbleLab Mitosis Plugin
 *
 * This file demonstrates how to integrate the mitosis bubble splitting plugin
 * with the existing BubbleLab UI components.
 */

import { MitosisAnimation, MitosisSettings, mitosisPlugin, resetMitosisPluginState } from '../../bubblelabs-mitosis-plugin/src/index';

// Initialize the mitosis plugin with default configuration
try {
  // Validate configuration before initialization
  const config = {
    enabled: false, // Disabled by default
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

  // Sanitize config values
  const sanitizedConfig = {
    enabled: typeof config.enabled === 'boolean' ? config.enabled : false,
    animationDuration: typeof config.animationDuration === 'number' && isFinite(config.animationDuration)
      ? Math.max(100, Math.min(10000, config.animationDuration)) : 1500,
    bounceIntensity: typeof config.bounceIntensity === 'number' && isFinite(config.bounceIntensity)
      ? Math.max(0, Math.min(1, config.bounceIntensity)) : 0.3,
    splitDelay: typeof config.splitDelay === 'number' && isFinite(config.splitDelay)
      ? Math.max(0, Math.min(5000, config.splitDelay)) : 300,
    colorVariation: typeof config.colorVariation === 'number' && isFinite(config.colorVariation)
      ? Math.max(0, Math.min(1, config.colorVariation)) : 0.1,
    rotationIntensity: typeof config.rotationIntensity === 'number' && isFinite(config.rotationIntensity)
      ? Math.max(0, Math.min(1, config.rotationIntensity)) : 0.2,
    opacityEffect: typeof config.opacityEffect === 'boolean' ? config.opacityEffect : true,
    trailEffect: typeof config.trailEffect === 'boolean' ? config.trailEffect : false,
    easingFunction: typeof config.easingFunction === 'string' ? config.easingFunction : 'cubic-bezier(0.25, 0.1, 0.25, 1)',
    particleEffects: typeof config.particleEffects === 'boolean' ? config.particleEffects : false
  };

  mitosisPlugin.initialize(sanitizedConfig);
} catch (error) {
  console.warn('Mitosis plugin: error during initialization:', error);
}

// Function to register the mitosis plugin with BubbleLab
export function registerMitosisPlugin() {
  try {
    // Add mitosis plugin to the BubbleLab plugin registry
    // This would typically be added to the plugins array in the main plugin registry

    // For now, we'll just return the plugin components for manual integration
    return {
      MitosisAnimation,
      MitosisSettings,
      mitosisPlugin
    };
  } catch (error) {
    console.warn('Mitosis plugin: error registering plugin:', error);
    // Return safe defaults
    return {
      MitosisAnimation: () => null,
      MitosisSettings: () => null,
      mitosisPlugin: {
        initialize: () => {},
        triggerMitosisSplit: async () => {},
        triggerBatchMitosis: async () => {},
        updateConfig: () => {},
        getState: () => ({ enabled: false, config: {}, isAnimating: false, lastAnimationTime: null }),
        toggleEnabled: () => {},
        isEnabled: () => false,
        applyPreset: () => {},
        getPerformanceMetrics: () => ({ avgDuration: 0, activeAnimations: 0, queuedAnimations: 0 })
      }
    };
  }
}

// Example function to trigger a mitosis animation in an existing visualization
export function triggerMitosisForEvolution(
  parentNode: { id: string; x: number; y: number; radius: number; color: string; label?: string },
  childNodes: Array<{ id: string; x: number; y: number; radius: number; color: string; label?: string }>
) {
  try {
    // This function would be called when an OpenEvolve evolution occurs
    // It triggers the mitosis animation to visualize the split

    if (mitosisPlugin.isEnabled()) {
      // Find the container where the animation should occur
      const container = document.getElementById('visualization-container') || document.body;
      const containerRef = { current: container };

      // Prepare the animation parameters
      const params = {
        parentNode,
        childNodes,
        containerRef
      };

      // Trigger the animation
      mitosisPlugin.triggerMitosisSplit(params);
    }
  } catch (error) {
    console.warn('Mitosis plugin: error triggering evolution animation:', error);
  }
}

// Example function to trigger batch mitosis animations in an existing visualization
export function triggerBatchMitosisForEvolution(
  parentNodes: Array<{ id: string; x: number; y: number; radius: number; color: string; label?: string }>,
  childNodeGroups: Array<Array<{ id: string; x: number; y: number; radius: number; color: string; label?: string }>>
) {
  try {
    // This function would be called when multiple OpenEvolve evolutions occur
    // It triggers multiple mitosis animations to visualize the splits

    if (mitosisPlugin.isEnabled()) {
      // Find the container where the animation should occur
      const container = document.getElementById('visualization-container') || document.body;
      const containerRef = { current: container };

      // Prepare the animation parameters
      const params = {
        parentNodes,
        childNodeGroups,
        containerRef
      };

      // Trigger the batch animation
      mitosisPlugin.triggerBatchMitosis(params);
    }
  } catch (error) {
    console.warn('Mitosis plugin: error triggering batch evolution animation:', error);
  }
}

// Function to cleanup the plugin resources
export function cleanupMitosisPlugin() {
  try {
    resetMitosisPluginState();
  } catch (error) {
    console.warn('Mitosis plugin: error during cleanup:', error);
  }
}

// Export the plugin for integration with BubbleLab
export default {
  id: 'mitosis-animation',
  name: 'Mitosis Bubble Splitting',
  description: 'Adds cell-division-like animations to visualize OpenEvolve evolutions',
  initialize: (config: any) => {
    try {
      mitosisPlugin.initialize(config);
    } catch (error) {
      console.warn('Mitosis plugin: error during initialization:', error);
    }
  },
  cleanup: cleanupMitosisPlugin,
  triggerMitosisForEvolution,
  triggerBatchMitosisForEvolution,
  components: {
    MitosisAnimation,
    MitosisSettings
  },
  settingsComponent: MitosisSettings,
  applyPreset: (preset: string) => {
    try {
      mitosisPlugin.applyPreset(preset as any);
    } catch (error) {
      console.warn('Mitosis plugin: error applying preset:', error);
    }
  },
  getPerformanceMetrics: () => {
    try {
      return mitosisPlugin.getPerformanceMetrics();
    } catch (error) {
      console.warn('Mitosis plugin: error getting performance metrics:', error);
      return { avgDuration: 0, activeAnimations: 0, queuedAnimations: 0 };
    }
  }
};