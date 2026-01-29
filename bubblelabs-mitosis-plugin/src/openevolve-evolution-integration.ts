/**
 * BubbleLab Mitosis Plugin - OpenEvolve Evolution Integration
 *
 * This module provides deep integration between the mitosis plugin and OpenEvolve's evolution engine,
 * allowing automatic visualization of evolution events as bubble splitting animations.
 */

import { mitosisPlugin } from './utils/createMitosisPlugin';
import type { BubbleNode } from './types/plugin-types';
import type {
  OpenEvolvePlugin,
  OpenEvolveExecutionResult
} from '@openevolve/plugin';

// Interface for evolution visualization parameters
export interface EvolutionVisualizationParams {
  parentNode: BubbleNode;
  childNodes: BubbleNode[];
  containerId?: string;
  containerRef?: React.RefObject<HTMLDivElement>;
  animationType?: 'mitosis' | 'morph' | 'fade' | 'custom';
  duration?: number;
  callback?: () => void;
}

// Interface for evolution event data
export interface EvolutionEventData {
  id: string;
  parent: {
    id: string;
    position: { x: number; y: number };
    size: number;
    color: string;
    metadata?: Record<string, any>;
  };
  children: Array<{
    id: string;
    position: { x: number; y: number };
    size: number;
    color: string;
    metadata?: Record<string, any>;
  }>;
  timestamp: number;
  evolutionType?: 'mutation' | 'crossover' | 'selection' | 'speciation';
  fitnessChange?: number;
  metadata?: Record<string, any>;
}

// Interface for evolution event listener
export interface EvolutionEventListener {
  onEvolutionStart: (event: EvolutionEventData) => void;
  onEvolutionComplete: (event: EvolutionEventData) => void;
  onError: (error: Error, event?: EvolutionEventData) => void;
}

/**
 * Evolution Event Manager
 *
 * Manages listening to OpenEvolve evolution events and triggering mitosis animations
 */
export class EvolutionEventManager {
  private openevolvePlugin: OpenEvolvePlugin | null = null;
  private eventListeners: EvolutionEventListener[] = [];
  private isListening: boolean = false;
  private evolutionSubscription: (() => void) | null = null;

  /**
   * Initialize the event manager with the OpenEvolve plugin
   */
  public initialize(plugin: OpenEvolvePlugin): void {
    this.openevolvePlugin = plugin;
  }

  /**
   * Add an event listener
   */
  public addListener(listener: EvolutionEventListener): void {
    this.eventListeners.push(listener);
  }

  /**
   * Remove an event listener
   */
  public removeListener(listener: EvolutionEventListener): void {
    const index = this.eventListeners.indexOf(listener);
    if (index > -1) {
      this.eventListeners.splice(index, 1);
    }
  }

  /**
   * Start listening to evolution events
   */
  public async startListening(): Promise<void> {
    if (this.isListening) {
      return;
    }

    if (!this.openevolvePlugin) {
      throw new Error('OpenEvolve plugin not initialized. Call initialize() first.');
    }

    this.isListening = true;

    try {
      // In a real implementation, this would subscribe to OpenEvolve's evolution events
      // For now, we'll log that we're listening
      console.log('Evolution event manager started listening to OpenEvolve events');
    } catch (error) {
      console.error('Failed to start listening to evolution events:', error);
      this.isListening = false;
      throw error;
    }
  }

  /**
   * Stop listening to evolution events
   */
  public stopListening(): void {
    if (!this.isListening) {
      return;
    }

    if (this.evolutionSubscription) {
      // Actual cleanup would depend on OpenEvolve's event system
      this.evolutionSubscription = null;
    }

    this.isListening = false;
    console.log('Evolution event manager stopped listening');
  }

  /**
   * Process an evolution event and trigger mitosis animation
   */
  public async processEvolutionEvent(event: EvolutionEventData): Promise<void> {
    // Notify listeners
    this.eventListeners.forEach(listener => {
      try {
        listener.onEvolutionStart(event);
      } catch (error) {
        console.error('Error in onEvolutionStart listener:', error);
        listener.onError(error as Error, event);
      }
    });

    try {
      // Check if mitosis plugin is enabled
      if (!mitosisPlugin.isEnabled()) {
        console.log('Mitosis plugin is disabled, skipping animation');
        return;
      }

      // Create parent node from event data
      const parentNode: BubbleNode = {
        id: event.parent.id,
        x: event.parent.position.x,
        y: event.parent.position.y,
        radius: event.parent.size,
        color: event.parent.color,
        label: `Gen ${event.metadata?.generation || 'N/A'}`
      };

      // Create child nodes from event data
      const childNodes: BubbleNode[] = event.children.map(child => ({
        id: child.id,
        x: child.position.x,
        y: child.position.y,
        radius: child.size,
        color: child.color,
        label: `Offspring ${child.id.substring(0, 4)}`
      }));

      // Find the visualization container
      const containerId = event.metadata?.containerId || 'visualization-container';
      const container = document.getElementById(containerId);

      if (!container) {
        console.warn(`Mitosis plugin: Container with id '${containerId}' not found`);
        // Try to find a default container
        const defaultContainer = document.querySelector('.evolution-visualization, .bubble-container, [id*="visual"]');
        if (defaultContainer) {
          console.log('Using default container for animation');
        } else {
          console.warn('No visualization container found, animation skipped');
          return;
        }
      }

      // Create a temporary container ref if needed
      const containerRef = { current: container || document.body };

      // Check if this is a survival-of-fittest evolution
      if (event.evolutionType === 'survival-of-fittest' || event.evolutionType === 'selection') {
        // For survival-of-fittest, we'll determine which children survive
        // For this example, let's say the best performer survives (the one with highest fitness)
        const survivorIndices = event.metadata?.survivorIndices || [0]; // Default to first child surviving

        try {
          // Trigger the evolution animation with survival logic
          await mitosisPlugin.triggerEvolutionSplit({
            parentNode,
            childNodes,
            containerRef,
            evolutionType: 'survival-of-fittest',
            survivorIndices
          });
        } catch (evolutionError) {
          logger.error('Error in evolution split animation:', evolutionError);
          // Fallback to standard animation if evolution fails
          try {
            await mitosisPlugin.triggerMitosisSplit({
              parentNode,
              childNodes,
              containerRef
            });
          } catch (fallbackError) {
            logger.error('Fallback mitosis animation also failed:', fallbackError);
            throw fallbackError; // Re-throw to maintain error propagation
          }
        }
      } else {
        // For standard evolution, just do a regular split
        await mitosisPlugin.triggerMitosisSplit({
          parentNode,
          childNodes,
          containerRef
        });
      }

      // Notify listeners of completion
      this.eventListeners.forEach(listener => {
        try {
          listener.onEvolutionComplete(event);
        } catch (error) {
          console.error('Error in onEvolutionComplete listener:', error);
          listener.onError(error as Error, event);
        }
      });

    } catch (error) {
      console.error('Error processing evolution event:', error);

      // Notify listeners of error
      this.eventListeners.forEach(listener => {
        try {
          listener.onError(error as Error, event);
        } catch (listenerError) {
          console.error('Error in onError listener:', listenerError);
        }
      });
    }
  }

  /**
   * Process batch evolution events
   */
  public async processBatchEvolutionEvents(events: EvolutionEventData[]): Promise<void> {
    if (!mitosisPlugin.isEnabled()) {
      console.log('Mitosis plugin is disabled, skipping batch animation');
      return;
    }

    // Convert events to batch animation format
    const parentNodes: BubbleNode[] = [];
    const childNodeGroups: BubbleNode[][] = [];

    for (const event of events) {
      // Create parent node from event data
      const parentNode: BubbleNode = {
        id: event.parent.id,
        x: event.parent.position.x,
        y: event.parent.position.y,
        radius: event.parent.size,
        color: event.parent.color,
        label: `Gen ${event.metadata?.generation || 'N/A'}`
      };

      parentNodes.push(parentNode);

      // Create child nodes from event data
      const childNodes: BubbleNode[] = event.children.map(child => ({
        id: child.id,
        x: child.position.x,
        y: child.position.y,
        radius: child.size,
        color: child.color,
        label: `Offspring ${child.id.substring(0, 4)}`
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
  }

  /**
   * Process OpenEvolve execution results to trigger mitosis animations
   */
  public async processOpenEvolveResult(result: any): Promise<void> {
    try {
      // Check if mitosis plugin is enabled
      if (!mitosisPlugin.isEnabled()) {
        console.log('Mitosis plugin is disabled, skipping animation');
        return;
      }

      // Extract evolution data from the result
      if (result.module === 'evolution' && result.output) {
        // Process evolution results to create visualization events
        const evolutionData = result.output;

        // Create visualization event based on evolution result
        const visualizationEvent: EvolutionEventData = {
          id: result.executionId,
          parent: {
            id: `parent-${result.executionId}`,
            position: { x: 100, y: 100 }, // Default position, could be based on actual data
            size: 30,
            color: '#4F46E5',
            metadata: {
              fitness: evolutionData.fitnessScores?.[0] || 0,
              generation: evolutionData.generations || 0
            }
          },
          children: evolutionData.population?.slice(0, 3).map((solution: any, index: number) => ({
            id: `child-${result.executionId}-${index}`,
            position: {
              x: 70 + (index * 30),
              y: 130 + (index * 10)
            },
            size: 20,
            color: ['#60A5FA', '#34D399', '#FBBF24'][index] || '#9CA3AF',
            metadata: {
              fitness: evolutionData.fitnessScores?.[index + 1] || 0,
              solution: solution
            }
          })) || [],
          timestamp: Date.now(),
          evolutionType: 'mutation',
          fitnessChange: evolutionData.convergence,
          metadata: {
            containerId: 'visualization-container'
          }
        };

        // Process the visualization event
        await this.processEvolutionEvent(visualizationEvent);
      } else if (result.module === 'decomposition' && result.output) {
        // Handle decomposition results - create visualization for problem decomposition
        const decompositionData = result.output;

        const visualizationEvent: EvolutionEventData = {
          id: result.executionId,
          parent: {
            id: `problem-${result.executionId}`,
            position: { x: 150, y: 150 },
            size: 35,
            color: '#7C3AED',
            metadata: {
              type: 'original_problem',
              complexity: decompositionData.complexityAnalysis?.overall || 'medium'
            }
          },
          children: decompositionData.subProblems?.slice(0, 4).map((subProblem: any, index: number) => ({
            id: `sub-${result.executionId}-${index}`,
            position: {
              x: 100 + (index % 2) * 80,
              y: 100 + Math.floor(index / 2) * 80
            },
            size: 20,
            color: ['#F59E0B', '#EF4444', '#10B981', '#8B5CF6'][index] || '#6B7280',
            metadata: {
              type: 'sub_problem',
              description: subProblem.description?.substring(0, 20) || 'Sub-problem',
              complexity: subProblem.complexity || 'low'
            }
          })) || [],
          timestamp: Date.now(),
          evolutionType: 'decomposition',
          fitnessChange: 0, // Decomposition doesn't have fitness in the traditional sense
          metadata: {
            containerId: 'visualization-container'
          }
        };

        // Process the visualization event
        await this.processEvolutionEvent(visualizationEvent);
      } else if (result.module === 'adversarial' && result.output) {
        // Handle adversarial results - create visualization for adversarial testing
        const adversarialData = result.output;

        const visualizationEvent: EvolutionEventData = {
          id: result.executionId,
          parent: {
            id: `original-${result.executionId}`,
            position: { x: 150, y: 150 },
            size: 30,
            color: '#DC2626', // Red for original content that needs improvement
            metadata: {
              type: 'original_content',
              quality: adversarialData.evaluatorAssessment?.originalScore || 0.5
            }
          },
          children: [
            {
              id: `improved-${result.executionId}`,
              position: { x: 100, y: 100 },
              size: 25,
              color: '#10B981', // Green for improved content
              metadata: {
                type: 'improved_content',
                quality: adversarialData.evaluatorAssessment?.improvedScore || 0.8
              }
            },
            {
              id: `insight-${result.executionId}`,
              position: { x: 200, y: 100 },
              size: 20,
              color: '#F59E0B', // Yellow for insights
              metadata: {
                type: 'insight',
                count: adversarialData.redTeamCritiques?.length || 0
              }
            }
          ],
          timestamp: Date.now(),
          evolutionType: 'adversarial_improvement',
          fitnessChange: adversarialData.evaluatorAssessment?.improvementPercentage || 0,
          metadata: {
            containerId: 'visualization-container'
          }
        };

        // Process the visualization event
        await this.processEvolutionEvent(visualizationEvent);
      }
    } catch (error) {
      console.error('Error processing OpenEvolve result:', error);
    }
  }

  /**
   * Get current listening status
   */
  public getListeningStatus(): boolean {
    return this.isListening;
  }
}

/**
 * Default evolution event manager instance
 */
export const evolutionEventManager = new EvolutionEventManager();

/**
 * Convenience function to process a single evolution event
 */
export const processEvolutionEvent = async (event: EvolutionEventData): Promise<void> => {
  return evolutionEventManager.processEvolutionEvent(event);
};

/**
 * Convenience function to process batch evolution events
 */
export const processBatchEvolutionEvents = async (events: EvolutionEventData[]): Promise<void> => {
  return evolutionEventManager.processBatchEvolutionEvents(events);
};

/**
 * Process OpenEvolve execution result to trigger mitosis animation
 */
export const processOpenEvolveResult = async (result: OpenEvolveExecutionResult): Promise<void> => {
  return evolutionEventManager.processOpenEvolveResult(result);
};

/**
 * Configure the mitosis plugin for optimal OpenEvolve integration
 */
export const configureForOpenEvolve = (customConfig?: Partial<Record<string, any>>): void => {
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
 * Connect to OpenEvolve evolution events
 */
export const connectToOpenEvolve = async (plugin: OpenEvolvePlugin, listener?: EvolutionEventListener): Promise<void> => {
  try {
    // Initialize the event manager
    evolutionEventManager.initialize(plugin);

    // Add the provided listener if available
    if (listener) {
      evolutionEventManager.addListener(listener);
    }

    // Start listening to events
    await evolutionEventManager.startListening();
  } catch (error) {
    console.error('Failed to connect to OpenEvolve evolution events:', error);
    throw error;
  }
};

/**
 * Disconnect from OpenEvolve evolution events
 */
export const disconnectFromOpenEvolve = (): void => {
  evolutionEventManager.stopListening();
};

// Export the integration functions
export default {
  evolutionEventManager,
  processEvolutionEvent,
  processBatchEvolutionEvents,
  processOpenEvolveResult,
  configureForOpenEvolve,
  connectToOpenEvolve,
  disconnectFromOpenEvolve
};