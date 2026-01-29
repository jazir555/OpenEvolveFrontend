/**
 * Plugin Registration Module for BubbleLab
 * 
 * This module provides the necessary functions and metadata for BubbleLab
 * to register and integrate the mitosis plugin.
 */

import { mitosisPlugin } from './utils/createMitosisPlugin';
import { MitosisAnimation, MitosisSettings, MitosisDemo } from './components';
import type { BubbleNode, SplitAnimationParams, BatchAnimationParams, AnimationPreset, PerformanceMetrics } from './types/plugin-types';

// Define the plugin interface that BubbleLab expects
export interface BubbleLabPlugin {
  id: string;
  name: string;
  description: string;
  version: string;
  initialize: (config?: any) => void;
  destroy: () => void;
  components: Record<string, React.ComponentType<any>>;
  settingsComponent?: React.ComponentType<any>;
  actions: Record<string, (...args: any[]) => any>;
  selectors: Record<string, (...args: any[]) => any>;
}

// Create the plugin registration object
const MitosisBubbleLabPlugin: BubbleLabPlugin = {
  id: 'mitosis-animation',
  name: 'Mitosis Bubble Splitting',
  description: 'Adds cell-division-like animations to visualize OpenEvolve evolutions',
  version: '1.0.0',
  
  initialize: (config?: any) => {
    try {
      // Initialize the mitosis plugin with provided configuration or defaults
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
    } catch (error) {
      console.error('Error initializing mitosis plugin:', error);
    }
  },
  
  destroy: () => {
    try {
      // Clean up plugin resources
      mitosisPlugin.cleanup();
    } catch (error) {
      console.error('Error destroying mitosis plugin:', error);
    }
  },
  
  components: {
    MitosisAnimation,
    MitosisSettings,
    MitosisDemo
  },
  
  settingsComponent: MitosisSettings,
  
  actions: {
    /**
     * Trigger a mitosis animation for a single parent-child split
     */
    triggerMitosisSplit: async (params: SplitAnimationParams) => {
      try {
        return await mitosisPlugin.triggerMitosisSplit(params);
      } catch (error) {
        console.error('Error in triggerMitosisSplit action:', error);
        throw error;
      }
    },
    
    /**
     * Trigger multiple mitosis animations in sequence
     */
    triggerBatchMitosis: async (params: BatchAnimationParams) => {
      try {
        return await mitosisPlugin.triggerBatchMitosis(params);
      } catch (error) {
        console.error('Error in triggerBatchMitosis action:', error);
        throw error;
      }
    },

    /**
     * Trigger an evolution animation with survival-of-fittest mechanics
     */
    triggerEvolutionSplit: async (params: EvolutionAnimationParams) => {
      try {
        return await mitosisPlugin.triggerEvolutionSplit(params);
      } catch (error) {
        console.error('Error in triggerEvolutionSplit action:', error);
        throw error;
      }
    },
    
    /**
     * Update plugin configuration
     */
    updateConfig: (config: Partial<any>) => {
      try {
        return mitosisPlugin.updateConfig(config);
      } catch (error) {
        console.error('Error in updateConfig action:', error);
        throw error;
      }
    },
    
    /**
     * Toggle plugin enabled state
     */
    toggleEnabled: () => {
      try {
        return mitosisPlugin.toggleEnabled();
      } catch (error) {
        console.error('Error in toggleEnabled action:', error);
        throw error;
      }
    },
    
    /**
     * Apply an animation preset
     */
    applyPreset: (preset: AnimationPreset) => {
      try {
        return mitosisPlugin.applyPreset(preset);
      } catch (error) {
        console.error('Error in applyPreset action:', error);
        throw error;
      }
    },
    
    /**
     * Reset plugin to initial state
     */
    reset: () => {
      try {
        // Reset to default configuration
        mitosisPlugin.initialize({
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
        });
      } catch (error) {
        console.error('Error in reset action:', error);
        throw error;
      }
    }
  },
  
  selectors: {
    /**
     * Get current plugin state
     */
    getState: () => {
      try {
        return mitosisPlugin.getState();
      } catch (error) {
        console.error('Error in getState selector:', error);
        return { enabled: false, config: {}, isAnimating: false, lastAnimationTime: null };
      }
    },
    
    /**
     * Check if plugin is enabled
     */
    isEnabled: () => {
      try {
        return mitosisPlugin.isEnabled();
      } catch (error) {
        console.error('Error in isEnabled selector:', error);
        return false;
      }
    },
    
    /**
     * Get performance metrics
     */
    getPerformanceMetrics: (): PerformanceMetrics => {
      try {
        return mitosisPlugin.getPerformanceMetrics();
      } catch (error) {
        console.error('Error in getPerformanceMetrics selector:', error);
        return { avgDuration: 0, activeAnimations: 0, queuedAnimations: 0 };
      }
    }
  }
};

/**
 * Function to register the mitosis plugin with BubbleLab
 */
export const registerMitosisPlugin = (): BubbleLabPlugin => {
  return MitosisBubbleLabPlugin;
};

export default MitosisBubbleLabPlugin;