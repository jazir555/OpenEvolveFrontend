import { MitosisAnimation } from './components/MitosisAnimation';
import { MitosisSettings } from './components/MitosisSettings';
import { MitosisDemo } from './components/MitosisDemo';
import { MitosisErrorBoundary } from './components/ErrorBoundary';
import { createMitosisPlugin, mitosisPlugin } from './utils/createMitosisPlugin';
import { MitosisConfig, MitosisPlugin, MitosisPluginState, BubbleNode, SplitAnimationParams, AnimationPreset } from './types/plugin-types';

// Import the plugin's CSS for styling and animations
import './styles/mitosis-animations.css';

// Export the plugin registration module
export * from './register-plugin';

// Export the BubbleLab-compatible plugin
export * from './bubblelab-plugin';

// Export OpenEvolve integration utilities
export * from './openevolve-integration';

// Export OpenEvolve evolution integration
export * from './openevolve-evolution-integration';

// Global error handler for the plugin
const globalErrorHandler = (error: Error, source?: string) => {
  try {
    console.warn(`Mitosis plugin global error handler: ${source || 'Unknown source'} - ${error.message}`);
    // In a production environment, you might want to send this to an error reporting service
  } catch (handlerError) {
    // If error handling itself fails, try to log to console
    try {
      console.error('Global error handler failed:', handlerError);
    } catch (consoleError) {
      // If all fails, silently continue
    }
  }
};

// Add error handling for the window if running in browser
if (typeof window !== 'undefined' && window.addEventListener) {
  window.addEventListener('error', (event) => {
    globalErrorHandler(event.error, 'window.error');
  });

  window.addEventListener('unhandledrejection', (event) => {
    globalErrorHandler(event.reason, 'unhandledrejection');
  });

  // Add additional error handling for common issues
  window.addEventListener('load', () => {
    try {
      // Check if required APIs are available
      if (typeof Promise === 'undefined') {
        console.warn('Mitosis plugin: Promise API not available');
      }

      if (typeof JSON === 'undefined') {
        console.warn('Mitosis plugin: JSON API not available');
      }

      if (typeof document !== 'undefined' && !document.createElement) {
        console.warn('Mitosis plugin: DOM API not available');
      }
    } catch (error) {
      console.warn('Mitosis plugin: error during environment check:', error);
    }
  });
}

export {
  MitosisAnimation,
  MitosisSettings,
  MitosisDemo,
  MitosisErrorBoundary,
  createMitosisPlugin,
  mitosisPlugin,
  // Types
  type MitosisConfig,
  type MitosisPlugin,
  type MitosisPluginState,
  type BubbleNode,
  type SplitAnimationParams
};

// Also export as default
export default mitosisPlugin;