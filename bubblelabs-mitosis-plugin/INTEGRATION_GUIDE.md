/**
 * Integration Guide: Adding Mitosis Plugin to BubbleLab
 *
 * This guide explains how to integrate the Mitosis Bubble Splitting plugin
 * into the BubbleLab ecosystem alongside the OpenEvolve plugin.
 */

// 1. Install the plugin
//    npm install @openevolve/bubblelab-mitosis-plugin

// 2. Update the BubbleLab plugin registry at:
//    BubbleLab/apps/bubble-studio/src/plugins/index.ts

// Import the MitosisPlugin
import { OpenEvolvePlugin } from '@openevolve/plugin';
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Add the MitosisPlugin to the plugins array
export const plugins = [
  OpenEvolvePlugin,    // Main OpenEvolve functionality
  MitosisPlugin,       // Mitosis bubble splitting animations
  // Add more plugins here as needed
];

// 3. The MitosisPlugin is now available throughout BubbleLab and integrates
//    with OpenEvolve's evolution visualization capabilities

// 4. Usage in visualization components:
//    The MitosisAnimation component can be used to visualize evolution splits
//    The MitosisSettings component provides user controls for animation parameters

// 5. The plugin follows the same interface as other BubbleLab plugins,
//    ensuring compatibility with the plugin ecosystem