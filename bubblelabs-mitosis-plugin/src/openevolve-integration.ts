/**
 * OpenEvolve Integration Module for Mitosis Plugin
 *
 * This module provides specific integration points between the
 * Mitosis Bubble Splitting plugin and the OpenEvolve evolution engine.
 */

import { mitosisPlugin } from './src/utils/createMitosisPlugin';
import type { BubbleNode } from './src/types/plugin-types';

// Interface for OpenEvolve evolution events
export interface EvolutionEvent {
  parentId: string;
  parentPosition: { x: number; y: number };
  parentSize: number;
  parentColor: string;
  childIds: string[];
  childPositions: Array<{ x: number; y: number }>;
  childSizes: number[];
  childColors: string[];
  timestamp: number;
  metadata?: Record<string, any>;
}

// Interface for evolution visualization parameters
export interface EvolutionVisualizationParams {
  parentNode: BubbleNode;
  childNodes: BubbleNode[];
  containerId?: string;
  animationType?: 'mitosis' | 'morph' | 'fade' | 'custom';
  duration?: number;
  callback?: () => void;
}

/**
 * Process an OpenEvolve evolution event and trigger the appropriate visualization
 */
export const processEvolutionEvent = (event: EvolutionEvent): Promise<void> => {
  return new Promise((resolve, reject) => {
    try {
      // Check if mitosis plugin is enabled
      if (!mitosisPlugin.isEnabled()) {
        resolve();
        return;
      }

      // Create parent node from event data
      const parentNode: BubbleNode = {
        id: event.parentId,
        x: event.parentPosition.x,
        y: event.parentPosition.y,
        radius: event.parentSize,
        color: event.parentColor,
        label: `Parent ${event.parentId.substring(0, 4)}`
      };

      // Create child nodes from event data
      const childNodes: BubbleNode[] = event.childIds.map((id, index) => ({
        id,
        x: event.childPositions[index]?.x ?? 0,
        y: event.childPositions[index]?.y ?? 0,
        radius: event.childSizes[index] ?? 15,
        color: event.childColors[index] ?? '#60A5FA',
        label: `Child ${id.substring(0, 4)}`
      }));

      // Find the visualization container
      const containerId = event.metadata?.containerId || 'visualization-container';
      const container = document.getElementById(containerId);
      
      if (!container) {
        console.warn(`Mitosis plugin: Container with id '${containerId}' not found`);
        resolve();
        return;
      }

      const containerRef = { current: container };

      // Check if this is a survival-of-fittest evolution
      if (event.metadata?.evolutionType === 'survival-of-fittest' || event.metadata?.evolutionType === 'selection') {
        // For survival-of-fittest, we'll determine which children survive
        const survivorIndices = event.metadata?.survivorIndices || [0]; // Default to first child surviving

        // Trigger the evolution animation with survival logic
        mitosisPlugin.triggerEvolutionSplit({
          parentNode,
          childNodes,
          containerRef,
          evolutionType: 'survival-of-fittest',
          survivorIndices
        }).then(() => {
          resolve();
        }).catch(error => {
          console.error('Mitosis plugin: Error triggering evolution animation:', error);
          // Fallback to standard animation if evolution fails
          try {
            mitosisPlugin.triggerMitosisSplit({
              parentNode,
              childNodes,
              containerRef
            }).then(() => {
              resolve();
            }).catch(fallbackError => {
              console.error('Mitosis plugin: Fallback animation also failed:', fallbackError);
              reject(fallbackError);
            });
          } catch (fallbackError) {
            console.error('Mitosis plugin: Fallback animation setup failed:', fallbackError);
            reject(fallbackError);
          }
        });
      } else {
        // For standard evolution, just do a regular split
        mitosisPlugin.triggerMitosisSplit({
          parentNode,
          childNodes,
          containerRef
        }).then(() => {
          resolve();
        }).catch(error => {
          console.error('Mitosis plugin: Error triggering evolution animation:', error);
          reject(error);
        });
      }
    } catch (error) {
      console.error('Mitosis plugin: Error processing evolution event:', error);
      reject(error);
    }
  });
};

/**
 * Process multiple evolution events simultaneously
 */
export const processBatchEvolutionEvents = async (events: EvolutionEvent[]): Promise<void> => {
  try {
    // Check if mitosis plugin is enabled
    if (!mitosisPlugin.isEnabled()) {
      return;
    }

    // Convert events to batch animation format
    const parentNodes: BubbleNode[] = [];
    const childNodeGroups: BubbleNode[][] = [];

    for (const event of events) {
      // Create parent node from event data
      const parentNode: BubbleNode = {
        id: event.parentId,
        x: event.parentPosition.x,
        y: event.parentPosition.y,
        radius: event.parentSize,
        color: event.parentColor,
        label: `Parent ${event.parentId.substring(0, 4)}`
      };

      parentNodes.push(parentNode);

      // Create child nodes from event data
      const childNodes: BubbleNode[] = event.childIds.map((id, index) => ({
        id,
        x: event.childPositions[index]?.x ?? 0,
        y: event.childPositions[index]?.y ?? 0,
        radius: event.childSizes[index] ?? 15,
        color: event.childColors[index] ?? '#60A5FA',
        label: `Child ${id.substring(0, 4)}`
      }));

      childNodeGroups.push(childNodes);
    }

    // Find the visualization container
    const containerId = events[0]?.metadata?.containerId || 'visualization-container';
    const container = document.getElementById(containerId);
    
    if (!container) {
      console.warn(`Mitosis plugin: Container with id '${containerId}' not found`);
      return;
    }

    const containerRef = { current: container };

    // Trigger the batch mitosis animation
    await mitosisPlugin.triggerBatchMitosis({
      parentNodes,
      childNodeGroups,
      containerRef
    });
  } catch (error) {
    console.error('Mitosis plugin: Error processing batch evolution events:', error);
    throw error;
  }
};

/**
 * Configure the mitosis plugin for optimal OpenEvolve integration
 */
export const configureForOpenEvolve = (customConfig?: Partial<Record<string, any>>) => {
  try {
    const defaultConfig = {
      enabled: true,
      animationDuration: 1200,        // Slightly faster for evolution sequences
      bounceIntensity: 0.2,         // Moderate bounce for visual interest
      splitDelay: 200,              // Quick delay before bounce
      colorVariation: 0.15,         // Small color variation for offspring
      rotationIntensity: 0.1,       // Gentle rotation
      opacityEffect: true,          // Enable opacity transitions
      trailEffect: false,           // Disable trails for cleaner evolution view
      easingFunction: 'ease-out',   // Clean ending to animations
      particleEffects: false        // Disable particles for performance
    };

    const config = { ...defaultConfig, ...customConfig };
    mitosisPlugin.initialize(config);
  } catch (error) {
    console.error('Mitosis plugin: Error configuring for OpenEvolve:', error);
  }
};

/**
 * Subscribe to OpenEvolve evolution events
 */
export const subscribeToOpenEvolveEvents = (callback: (event: EvolutionEvent) => void) => {
  // This would typically connect to OpenEvolve's event system
  // For now, we'll simulate by exposing the callback interface
  console.log('Mitosis plugin: Subscribed to OpenEvolve events');
  
  // In a real implementation, this would listen to OpenEvolve's
  // evolution events and call the callback when events occur
  return {
    unsubscribe: () => {
      console.log('Mitosis plugin: Unsubscribed from OpenEvolve events');
    }
  };
};

// Export the integration functions
export default {
  processEvolutionEvent,
  processBatchEvolutionEvents,
  configureForOpenEvolve,
  subscribeToOpenEvolveEvents
};