import { MitosisConfig, MitosisPlugin, MitosisPluginState, SplitAnimationParams, BubbleNode } from '../types/plugin-types';
import { logger } from './logger';

// Global state for the mitosis plugin
let globalState: MitosisPluginState = {
  config: {
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
  },
  isAnimating: false,
  enabled: false,
  lastAnimationTime: null
};

// Lock to prevent concurrent state updates
let stateUpdateLock = false;

// Maximum attempts to acquire state update lock
const MAX_STATE_UPDATE_ATTEMPTS = 10;

// Safely update global state with error handling and locking
function updateGlobalState(updates: Partial<MitosisPluginState>): void {
  // Attempt to acquire the lock with retry mechanism
  let attempts = 0;
  while (stateUpdateLock && attempts < MAX_STATE_UPDATE_ATTEMPTS) {
    // Brief pause to avoid busy waiting
    try {
      // Use a small timeout to yield control
      setTimeout(() => {}, 1);
    } catch (pauseError) {
      // If timeout fails, continue without pause
    }
    attempts++;
  }

  if (attempts >= MAX_STATE_UPDATE_ATTEMPTS) {
    logger.warn('Failed to acquire state update lock after maximum attempts');
    return;
  }

  // Acquire the lock
  stateUpdateLock = true;

  try {
    logger.debug('Updating global state with:', updates);

    // Create a deep clone to prevent external mutations
    let clonedState, clonedUpdates;

    try {
      clonedState = JSON.parse(JSON.stringify(globalState));
    } catch (cloneError) {
      logger.warn('Error cloning global state, using defaults:', cloneError);
      clonedState = {
        config: {
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
        },
        isAnimating: false,
        enabled: false,
        lastAnimationTime: null
      };
    }

    try {
      clonedUpdates = JSON.parse(JSON.stringify(updates));
    } catch (cloneError) {
      logger.warn('Error cloning updates, using empty object:', cloneError);
      clonedUpdates = {};
    }

    // Validate updates before applying
    const safeUpdates: Partial<MitosisPluginState> = {};

    if ('config' in clonedUpdates && clonedUpdates.config && typeof clonedUpdates.config === 'object') {
      const config = clonedUpdates.config;
      safeUpdates.config = {
        enabled: 'enabled' in config ? Boolean(config.enabled) : clonedState.config.enabled,
        animationDuration: 'animationDuration' in config && typeof config.animationDuration === 'number' && isFinite(config.animationDuration)
          ? Math.max(100, Math.min(10000, config.animationDuration)) : clonedState.config.animationDuration,
        bounceIntensity: 'bounceIntensity' in config && typeof config.bounceIntensity === 'number' && isFinite(config.bounceIntensity)
          ? Math.max(0, Math.min(1, config.bounceIntensity)) : clonedState.config.bounceIntensity,
        splitDelay: 'splitDelay' in config && typeof config.splitDelay === 'number' && isFinite(config.splitDelay)
          ? Math.max(0, Math.min(5000, config.splitDelay)) : clonedState.config.splitDelay,
        colorVariation: 'colorVariation' in config && typeof config.colorVariation === 'number' && isFinite(config.colorVariation)
          ? Math.max(0, Math.min(1, config.colorVariation)) : clonedState.config.colorVariation,
        rotationIntensity: 'rotationIntensity' in config && typeof config.rotationIntensity === 'number' && isFinite(config.rotationIntensity)
          ? Math.max(0, Math.min(1, config.rotationIntensity)) : clonedState.config.rotationIntensity,
        opacityEffect: 'opacityEffect' in config ? Boolean(config.opacityEffect) : clonedState.config.opacityEffect,
        trailEffect: 'trailEffect' in config ? Boolean(config.trailEffect) : clonedState.config.trailEffect,
        easingFunction: 'easingFunction' in config && typeof config.easingFunction === 'string'
          ? config.easingFunction : clonedState.config.easingFunction,
        particleEffects: 'particleEffects' in config ? Boolean(config.particleEffects) : clonedState.config.particleEffects
      };
    } else if ('config' in clonedState) {
      // Preserve existing config if no new config provided
      safeUpdates.config = clonedState.config;
    }

    if ('isAnimating' in clonedUpdates) {
      safeUpdates.isAnimating = Boolean(clonedUpdates.isAnimating);
    } else if ('isAnimating' in clonedState) {
      // Preserve existing value if not in updates
      safeUpdates.isAnimating = clonedState.isAnimating;
    }

    if ('enabled' in clonedUpdates) {
      safeUpdates.enabled = Boolean(clonedUpdates.enabled);
    } else if ('enabled' in clonedState) {
      // Preserve existing value if not in updates
      safeUpdates.enabled = clonedState.enabled;
    }

    if ('lastAnimationTime' in clonedUpdates) {
      safeUpdates.lastAnimationTime = clonedUpdates.lastAnimationTime;
    } else if ('lastAnimationTime' in clonedState) {
      // Preserve existing value if not in updates
      safeUpdates.lastAnimationTime = clonedState.lastAnimationTime;
    }

    globalState = { ...clonedState, ...safeUpdates };
    logger.debug('Global state updated successfully');
  } catch (error) {
    logger.error('Error updating global state, resetting to defaults:', error);
    // Reset to safe defaults if state becomes corrupted
    try {
      globalState = {
        config: {
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
        },
        isAnimating: false,
        enabled: false,
        lastAnimationTime: null
      };
    } catch (resetError) {
      logger.error('Critical error: Could not reset global state:', resetError);
      // Last resort: try to preserve basic functionality
      try {
        globalState.isAnimating = false;
        globalState.enabled = false;
      } catch (lastResortError) {
        logger.error('Critical error: Could not recover global state:', lastResortError);
      }
    }
  } finally {
    // Always release the lock
    stateUpdateLock = false;
  }
}

class MitosisAnimationService {
  // Track active animations to allow for cleanup
  private activeAnimations: Map<HTMLElement, Animation | undefined> = new Map();

  // Throttle animations to prevent too many running simultaneously
  private animationQueue: Array<() => void> = [];
  private isProcessingQueue: boolean = false;
  private maxConcurrentAnimations: number = 5; // Limit concurrent animations for performance

  // Performance tracking
  private animationPerformance: {
    totalAnimations: number;
    avgDuration: number;
    lastFrameTime: number;
  } = {
    totalAnimations: 0,
    avgDuration: 0,
    lastFrameTime: 0
  };

  async executeSplitAnimation(params: SplitAnimationParams): Promise<void> {
    return new Promise((resolve) => {
      // Add to queue to throttle animations
      const animationTask = () => {
        // Start performance tracking
        const startTime = performance.now();
        this.performAnimation(params, () => {
          // Update performance stats
          const endTime = performance.now();
          this.updatePerformanceStats(endTime - startTime);
          resolve();
        });
      };

      this.animationQueue.push(animationTask);
      this.processQueue();
    });
  }

  private updatePerformanceStats(duration: number): void {
    try {
      this.animationPerformance.totalAnimations++;
      // Calculate moving average of animation durations
      this.animationPerformance.avgDuration =
        (this.animationPerformance.avgDuration * (this.animationPerformance.totalAnimations - 1) + duration) /
        this.animationPerformance.totalAnimations;
    } catch (error) {
      logger.warn('Error updating performance stats:', error);
    }
  }

  // Method to get performance metrics
  getPerformanceMetrics(): { avgDuration: number; activeAnimations: number; queuedAnimations: number } {
    return {
      avgDuration: this.animationPerformance.avgDuration,
      activeAnimations: this.activeAnimations.size,
      queuedAnimations: this.animationQueue.length
    };
  }

  private async performAnimation(params: SplitAnimationParams, resolve: () => void): Promise<void> {
    logger.info('Starting animation execution');
    try {
      // Validate parameters first
      if (!params || !params.containerRef || !params.containerRef.current) {
        logger.warn('Invalid parameters or container ref');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      if (!params.parentNode || !Array.isArray(params.childNodes) || params.childNodes.length === 0) {
        logger.warn('Invalid parent or child nodes provided');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      const { parentNode, childNodes, containerRef } = params;
      let container: HTMLElement | null = null;

      try {
        container = containerRef.current;
      } catch (containerError) {
        logger.warn('Error accessing container ref:', containerError);
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      logger.debug('Validating animation parameters', { parentNode, childNodesCount: childNodes.length });

      // Validate container element
      if (!container || !(container instanceof HTMLElement)) {
        logger.warn('Invalid container element');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Validate parent node properties
      if (typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
          typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string') {
        logger.warn('Invalid parent node properties');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Validate child nodes
      let validChildNodes: BubbleNode[] = [];
      try {
        validChildNodes = childNodes.filter(node =>
          node &&
          typeof node.x === 'number' &&
          typeof node.y === 'number' &&
          typeof node.radius === 'number' &&
          typeof node.color === 'string'
        );
      } catch (filterError) {
        logger.warn('Error filtering child nodes:', filterError);
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      if (validChildNodes.length === 0) {
        logger.warn('No valid child nodes to animate');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      logger.info(`Processing animation with ${validChildNodes.length} child nodes`);

      // Create parent bubble element
      let parentBubble: HTMLElement | null = null;
      try {
        parentBubble = this.createBubbleElement(parentNode);
      } catch (createError) {
        logger.warn('Error creating parent bubble element:', createError);
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      if (!parentBubble) {
        logger.warn('Could not create parent bubble element');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Add parent bubble to container
      try {
        if (container && container.appendChild) {
          container.appendChild(parentBubble);
          logger.debug('Parent bubble added to container');
        } else {
          logger.warn('Container does not support appendChild');
          resolve();
          this.dequeueAndProcessNext();
          return;
        }
      } catch (error) {
        logger.warn('Error appending parent bubble to container:', error);
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Wait for the split delay
      setTimeout(() => {
        logger.debug('Starting split phase');
        try {
          // Remove parent bubble
          try {
            if (parentBubble && parentBubble.parentNode === container) {
              parentBubble.remove();
              logger.debug('Parent bubble removed');
            }
          } catch (error) {
            logger.warn('Error removing parent bubble:', error);
          }

          // Create child bubbles at the same position initially using a document fragment for better performance
          const childBubbles: HTMLElement[] = [];
          let fragment: DocumentFragment | null = null;

          try {
            fragment = document.createDocumentFragment();
          } catch (fragmentError) {
            logger.warn('Error creating document fragment:', fragmentError);
            // Fallback: add bubbles directly to container
            // Continue with individual additions
          }

          validChildNodes.forEach(childNode => {
            try {
              const childBubble = this.createBubbleElement({
                ...childNode,
                x: parentNode.x, // Start at parent position
                y: parentNode.y  // Start at parent position
              });

              if (childBubble) {
                try {
                  if (fragment) {
                    fragment.appendChild(childBubble);
                  } else {
                    // Fallback: add directly to container
                    container?.appendChild(childBubble);
                  }
                  childBubbles.push(childBubble);
                  logger.debug('Child bubble added to fragment');
                } catch (error) {
                  logger.warn('Error appending child bubble to fragment:', error);
                }
              }
            } catch (error) {
              logger.warn('Error creating child bubble:', error);
            }
          });

          // Append all bubbles to the container in a single operation if fragment was created
          if (fragment && childBubbles.length > 0) {
            try {
              if (container && container.appendChild) {
                container.appendChild(fragment);
                logger.debug(`Appended ${childBubbles.length} child bubbles to container`);
              }
            } catch (error) {
              logger.warn('Error appending fragment to container:', error);
            }
          }

          logger.info(`Created ${childBubbles.length} child bubbles for animation`);

          // Animate each child bubble to its final position with bounce effect
          childBubbles.forEach((bubble, index) => {
            try {
              const targetNode = validChildNodes[index];

              // Calculate animation properties
              const dx = targetNode.x - parentNode.x;
              const dy = targetNode.y - parentNode.y;

              // Validate animation deltas
              if (!isFinite(dx) || !isFinite(dy)) {
                logger.warn('Invalid animation deltas');
                return;
              }

              logger.debug(`Animating bubble ${index} with dx: ${dx}, dy: ${dy}`);

              // Apply mitosis animation with bounce
              let animation: Animation | undefined;
              try {
                animation = this.animateBubbleSplit(bubble, dx, dy, globalState.config);
              } catch (animateError) {
                logger.warn('Error in bubble animation:', animateError);
              }

              // Store the animation reference for potential cleanup
              if (animation && bubble) {
                try {
                  this.activeAnimations.set(bubble, animation);
                  logger.debug(`Animation registered for bubble ${index}`);
                } catch (setAnimationError) {
                  logger.warn('Error setting animation reference:', setAnimationError);
                }
              }
            } catch (error) {
              logger.warn('Error animating bubble:', error);
            }
          });

          // Resolve after animation duration
          setTimeout(() => {
            logger.info('Animation completed, cleaning up');
            try {
              updateGlobalState({
                isAnimating: false,
                lastAnimationTime: Date.now()
              });

              // Clean up animation references after completion
              childBubbles.forEach(bubble => {
                if (bubble) {
                  try {
                    this.activeAnimations.delete(bubble);
                  } catch (deleteError) {
                    logger.warn('Error deleting animation reference:', deleteError);
                  }
                }
              });
              logger.debug('Animation cleanup completed');
            } catch (error) {
              logger.warn('Error updating state:', error);
            } finally {
              resolve();
              this.dequeueAndProcessNext();
            }
          }, globalState.config.animationDuration!);
        } catch (error) {
          logger.error('Error during split phase:', error);
          try {
            updateGlobalState({ isAnimating: false });

            // Clean up any remaining animation references
            childBubbles.forEach(bubble => {
              if (bubble) {
                try {
                  this.activeAnimations.delete(bubble);
                } catch (deleteError) {
                  logger.warn('Error deleting animation reference during cleanup:', deleteError);
                }
              }
            });
          } catch (stateError) {
            logger.warn('Error updating state after split phase error:', stateError);
          }
          resolve();
          this.dequeueAndProcessNext();
        }
      }, globalState.config.splitDelay);
    } catch (error) {
      logger.error('Error starting animation:', error);
      try {
        updateGlobalState({ isAnimating: false });
      } catch (stateError) {
        logger.warn('Error updating state after start error:', stateError);
      }
      resolve();
      this.dequeueAndProcessNext();
    }
  }

  private dequeueAndProcessNext(): void {
    // Remove the completed task
    this.animationQueue.shift();

    // Process next if available and under limit
    this.processQueue();
  }

  private processQueue(): void {
    // Process as many tasks as allowed by our concurrency limit
    while (!this.isProcessingQueue &&
           this.animationQueue.length > 0 &&
           this.activeAnimations.size < this.maxConcurrentAnimations) {

      this.isProcessingQueue = true;
      const nextTask = this.animationQueue.shift();
      if (nextTask) {
        // Execute the next animation task
        setTimeout(() => {
          try {
            nextTask();
          } catch (taskError) {
            console.warn('Mitosis animation: error executing queued animation task:', taskError);
            // Ensure we reset the processing flag even if the task fails
            this.isProcessingQueue = false;
            // Process the next item in queue
            this.processQueue();
          }
        }, 0); // Use setTimeout to prevent blocking
      } else {
        this.isProcessingQueue = false;
      }
    }
  }

  // Method to cancel all active animations (for cleanup purposes)
  cancelAllAnimations(): void {
    try {
      this.activeAnimations.forEach((animation, element) => {
        if (animation && typeof animation.cancel === 'function') {
          try {
            animation.cancel();
          } catch (cancelError) {
            console.warn('Mitosis animation: error cancelling animation:', cancelError);
          }
        }

        // Remove the element from DOM as well
        try {
          if (element.parentNode) {
            element.remove();
          }
        } catch (removeError) {
          console.warn('Mitosis animation: error removing element from DOM:', removeError);
        }
      });

      this.activeAnimations.clear();

      // Clear the animation queue as well
      this.animationQueue = [];
      this.isProcessingQueue = false;
    } catch (cleanupError) {
      console.warn('Mitosis animation: error during animation cleanup:', cleanupError);
    }
  }

  async executeEvolutionSplit(params: EvolutionAnimationParams): Promise<void> {
    return new Promise(async (resolve) => {
      try {
        // Add to queue to throttle animations
        const animationTask = () => {
          try {
            // Start performance tracking
            const startTime = performance.now();
            this.performEvolutionAnimation(params, () => {
              try {
                // Update performance stats
                const endTime = performance.now();
                this.updatePerformanceStats(endTime - startTime);
                resolve();
              } catch (perfError) {
                logger.warn('Error updating performance stats:', perfError);
                resolve(); // Still resolve to prevent hanging
              }
            });
          } catch (taskError) {
            logger.warn('Error in animation task:', taskError);
            resolve(); // Resolve to prevent hanging
          }
        };

        this.animationQueue.push(animationTask);
        this.processQueue();
      } catch (queueError) {
        logger.warn('Error queuing evolution animation:', queueError);
        resolve(); // Resolve to prevent hanging
      }
    });
  }

  private async performEvolutionAnimation(params: EvolutionAnimationParams, resolve: () => void): Promise<void> {
    logger.info('Starting evolution animation execution');
    try {
      // Validate parameters first
      if (!params || !params.containerRef || !params.containerRef.current) {
        logger.warn('Invalid parameters or container ref');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      if (!params.parentNode || !Array.isArray(params.childNodes) || params.childNodes.length === 0) {
        logger.warn('Invalid parent or child nodes provided');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      const { parentNode, childNodes, containerRef, survivorIndices } = params;
      const container = containerRef.current;

      logger.debug('Validating evolution animation parameters', { parentNode, childNodesCount: childNodes.length });

      // Validate container element
      if (!container || !(container instanceof HTMLElement)) {
        logger.warn('Invalid container element');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Validate parent node properties
      if (typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
          typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string') {
        logger.warn('Invalid parent node properties');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Validate child nodes
      const validChildNodes = childNodes.filter(node =>
        node &&
        typeof node.x === 'number' &&
        typeof node.y === 'number' &&
        typeof node.radius === 'number' &&
        typeof node.color === 'string' &&
        isFinite(node.x) &&
        isFinite(node.y) &&
        isFinite(node.radius)
      );

      if (validChildNodes.length === 0) {
        logger.warn('No valid child nodes to animate');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      logger.info(`Processing evolution animation with ${validChildNodes.length} child nodes`);

      // Create parent bubble element
      const parentBubble = this.createBubbleElement(parentNode);
      if (!parentBubble) {
        logger.warn('Could not create parent bubble element');
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Add parent bubble to container
      try {
        container.appendChild(parentBubble);
        logger.debug('Parent bubble added to container');
      } catch (error) {
        logger.warn('Error appending parent bubble to container:', error);
        resolve();
        this.dequeueAndProcessNext();
        return;
      }

      // Wait for the split delay
      setTimeout(() => {
        logger.debug('Starting evolution split phase');
        try {
          // Remove parent bubble
          try {
            if (parentBubble.parentNode === container) {
              parentBubble.remove();
              logger.debug('Parent bubble removed');
            }
          } catch (error) {
            logger.warn('Error removing parent bubble:', error);
          }

          // Create child bubbles at the same position initially using a document fragment for better performance
          const childBubbles: HTMLElement[] = [];
          const fragment = document.createDocumentFragment();

          validChildNodes.forEach(childNode => {
            try {
              const childBubble = this.createBubbleElement({
                ...childNode,
                x: parentNode.x, // Start at parent position
                y: parentNode.y  // Start at parent position
              });

              if (childBubble) {
                try {
                  fragment.appendChild(childBubble);
                  childBubbles.push(childBubble);
                  logger.debug('Child bubble added to fragment');
                } catch (error) {
                  logger.warn('Error appending child bubble to fragment:', error);
                }
              }
            } catch (error) {
              logger.warn('Error creating child bubble:', error);
            }
          });

          // Append all bubbles to the container in a single operation
          try {
            container.appendChild(fragment);
            logger.debug(`Appended ${childBubbles.length} child bubbles to container`);
          } catch (error) {
            logger.warn('Error appending fragment to container:', error);
          }

          logger.info(`Created ${childBubbles.length} child bubbles for evolution animation`);

          // Animate each child bubble to its final position with bounce effect
          childBubbles.forEach((bubble, index) => {
            try {
              const targetNode = validChildNodes[index];

              // Calculate animation properties
              const dx = targetNode.x - parentNode.x;
              const dy = targetNode.y - parentNode.y;

              // Validate animation deltas
              if (!isFinite(dx) || !isFinite(dy)) {
                logger.warn('Invalid animation deltas');
                return;
              }

              logger.debug(`Animating bubble ${index} with dx: ${dx}, dy: ${dy}`);

              // Apply mitosis animation with bounce
              const animation = this.animateBubbleSplit(bubble, dx, dy, globalState.config);

              // Store the animation reference for potential cleanup
              if (animation) {
                this.activeAnimations.set(bubble, animation);
                logger.debug(`Animation registered for bubble ${index}`);
              }
            } catch (error) {
              logger.warn('Error animating bubble:', error);
            }
          });

          // After the split animation completes, apply evolution logic
          setTimeout(() => {
            try {
              logger.debug('Applying evolution survival logic');

              // Apply survival-of-fittest logic
              if (Array.isArray(survivorIndices) && survivorIndices.length > 0) {
                // Color the bubbles based on survival status
                childBubbles.forEach((bubble, index) => {
                  try {
                    if (survivorIndices.includes(index)) {
                      // Survivor - turn green
                      if (bubble && bubble.style) {
                        bubble.style.backgroundColor = '#10B981'; // Green
                      }

                      // Update label if it exists
                      if (bubble) {
                        const labelElement = bubble.querySelector('.mitosis-bubble-label');
                        if (labelElement) {
                          labelElement.textContent = `✓ ${validChildNodes[index]?.label || `Survivor ${index}`}`;
                        }
                      }
                    } else {
                      // Died - turn red
                      if (bubble && bubble.style) {
                        bubble.style.backgroundColor = '#EF4444'; // Red
                      }

                      // Update label if it exists
                      if (bubble) {
                        const labelElement = bubble.querySelector('.mitosis-bubble-label');
                        if (labelElement) {
                          labelElement.textContent = `✗ ${validChildNodes[index]?.label || `Failed ${index}`}`;
                        }
                      }
                    }
                  } catch (bubbleError) {
                    logger.warn('Error updating bubble appearance:', bubbleError);
                    // Continue with other bubbles even if one fails
                  }
                });

                // If there's a next evolution, trigger it for survivors
                if (params.nextEvolution) {
                  try {
                    // Find the surviving bubbles and trigger the next evolution
                    const survivorBubbles = childBubbles.filter((_, index) => survivorIndices.includes(index));

                    if (survivorBubbles.length > 0) {
                      // Use the first survivor for the next evolution (or pick randomly)
                      const firstSurvivorIndex = survivorIndices[0];
                      const firstSurvivorNode = validChildNodes[firstSurvivorIndex];

                      // Wait a bit before the next evolution
                      setTimeout(() => {
                        try {
                          this.performEvolutionAnimation(params.nextEvolution!, () => {
                            try {
                              // Complete the entire evolution chain
                              updateGlobalState({
                                isAnimating: false,
                                lastAnimationTime: Date.now()
                              });

                              // Clean up animation references after completion
                              childBubbles.forEach(bubble => {
                                if (bubble) {
                                  this.activeAnimations.delete(bubble);
                                }
                              });
                              logger.debug('Evolution animation cleanup completed');
                              resolve();
                            } catch (chainCleanupError) {
                              logger.warn('Error in chained evolution cleanup:', chainCleanupError);
                              resolve();
                            }
                          });
                        } catch (chainError) {
                          logger.warn('Error in chained evolution:', chainError);
                          // Continue with normal resolution
                          setTimeout(() => {
                            try {
                              updateGlobalState({
                                isAnimating: false,
                                lastAnimationTime: Date.now()
                              });
                              childBubbles.forEach(bubble => {
                                if (bubble) {
                                  this.activeAnimations.delete(bubble);
                                }
                              });
                              resolve();
                              this.dequeueAndProcessNext();
                            } catch (fallbackError) {
                              logger.warn('Error in fallback resolution:', fallbackError);
                              resolve();
                              this.dequeueAndProcessNext();
                            }
                          }, 2000);
                        }
                      }, 1500); // Wait 1.5 seconds before next evolution

                      return; // Return early since we're chaining evolutions
                    }
                  } catch (nextEvolutionError) {
                    logger.warn('Error preparing next evolution:', nextEvolutionError);
                    // Continue with normal resolution
                  }
                }
              }
            } catch (evolutionLogicError) {
              logger.warn('Error in evolution logic application:', evolutionLogicError);
              // Continue with normal resolution despite evolution logic error
            }

            // Resolve after evolution logic is applied
            setTimeout(() => {
              try {
                logger.info('Evolution animation completed, cleaning up');
                try {
                  updateGlobalState({
                    isAnimating: false,
                    lastAnimationTime: Date.now()
                  });

                  // Clean up animation references after completion
                  childBubbles.forEach(bubble => {
                    if (bubble) {
                      this.activeAnimations.delete(bubble);
                    }
                  });
                  logger.debug('Evolution animation cleanup completed');
                } catch (stateError) {
                  logger.warn('Error updating state:', stateError);
                } finally {
                  resolve();
                  this.dequeueAndProcessNext();
                }
              } catch (resolutionError) {
                logger.warn('Error in resolution:', resolutionError);
                resolve();
                this.dequeueAndProcessNext();
              }
            }, 2000); // Keep the evolution results visible for 2 seconds
          }, globalState.config.animationDuration!);
        } catch (error) {
          logger.error('Error during evolution split phase:', error);
          try {
            updateGlobalState({ isAnimating: false });

            // Clean up any remaining animation references
            childBubbles.forEach(bubble => {
              this.activeAnimations.delete(bubble);
            });
          } catch (stateError) {
            logger.warn('Error updating state after split phase error:', stateError);
          }
          resolve();
          this.dequeueAndProcessNext();
        }
      }, globalState.config.splitDelay);
    } catch (error) {
      logger.error('Error starting evolution animation:', error);
      try {
        updateGlobalState({ isAnimating: false });
      } catch (stateError) {
        logger.warn('Error updating state after start error:', stateError);
      }
      resolve();
      this.dequeueAndProcessNext();
    }
  }

  private createBubbleElement(node: BubbleNode): HTMLElement | null {
    try {
      // Validate node properties
      if (!node || typeof node !== 'object') {
        logger.warn('Mitosis animation: invalid node object for bubble creation');
        return null;
      }

      if (typeof node.x !== 'number' || typeof node.y !== 'number' ||
          typeof node.radius !== 'number' || typeof node.color !== 'string') {
        logger.warn('Mitosis animation: invalid node properties for bubble creation');
        return null;
      }

      // Validate values are finite numbers
      if (!isFinite(node.x) || !isFinite(node.y) || !isFinite(node.radius)) {
        logger.warn('Mitosis animation: non-finite node values for bubble creation');
        return null;
      }

      // Sanitize values to prevent invalid CSS
      const sanitizedX = Math.max(-10000, Math.min(10000, node.x));
      const sanitizedY = Math.max(-10000, Math.min(10000, node.y));
      const sanitizedRadius = Math.max(1, Math.min(1000, node.radius)); // Prevent negative or extremely large radii
      const sanitizedColor = typeof node.color === 'string' && node.color.trim() !== '' ? node.color : '#4F46E5';
      const sanitizedId = typeof node.id === 'string' && node.id.trim() !== '' ? node.id : `bubble-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      let bubble: HTMLElement | null = null;

      try {
        // Check if document is available before creating elements
        if (typeof document !== 'undefined' && document.createElement) {
          bubble = document.createElement('div');
        } else {
          logger.warn('Mitosis animation: document not available for bubble creation');
          return null;
        }
      } catch (createElementError) {
        logger.warn('Mitosis animation: error creating div element:', createElementError);
        return null;
      }

      if (!bubble) {
        logger.warn('Mitosis animation: failed to create bubble element');
        return null;
      }

      // Set class name safely
      try {
        bubble.className = 'mitosis-bubble';
        // Add data attribute for identification
        bubble.setAttribute('data-bubble-id', sanitizedId);
      } catch (classNameError) {
        logger.warn('Error setting class name:', classNameError);
        // Continue anyway since this is not critical
      }

      // Set accessibility attributes
      try {
        bubble.setAttribute('role', 'img');
        bubble.setAttribute('aria-label', node.label || `Bubble at position ${sanitizedX}, ${sanitizedY}`);
      } catch (ariaError) {
        logger.warn('Error setting accessibility attributes:', ariaError);
        // Continue anyway since this is not critical
      }

      // Set position and size styles safely (these need to be inline as they're dynamic)
      try {
        bubble.style.left = `${sanitizedX - sanitizedRadius}px`;
        bubble.style.top = `${sanitizedY - sanitizedRadius}px`;
        bubble.style.width = `${sanitizedRadius * 2}px`;
        bubble.style.height = `${sanitizedRadius * 2}px`;
        bubble.style.backgroundColor = sanitizedColor;
        // Ensure the bubble is visible and properly positioned
        bubble.style.position = 'absolute';
        bubble.style.borderRadius = '50%';
        bubble.style.boxSizing = 'border-box';
        bubble.style.border = '2px solid rgba(255, 255, 255, 0.5)';
      } catch (styleError) {
        logger.warn('Error setting position styles:', styleError);
        // Continue anyway since we have a valid element
      }

      // Set label content safely
      if (node.label && typeof node.label === 'string') {
        try {
          // Add label class for styling
          const labelDiv = document.createElement('div');
          labelDiv.className = 'mitosis-bubble-label';

          // More robust sanitization to prevent XSS
          // Remove HTML tags and limit length
          let sanitizedLabel = node.label.replace(/<[^>]*>?/gm, '').substring(0, 100);
          // Remove potential script tags and other dangerous content
          sanitizedLabel = sanitizedLabel
            .replace(/javascript:/gi, '')
            .replace(/vbscript:/gi, '')
            .replace(/on\w+\s*=/gi, '')
            .replace(/script/gi, '');

          labelDiv.textContent = sanitizedLabel;
          bubble.appendChild(labelDiv);
        } catch (labelError) {
          logger.warn('Error setting label content:', labelError);
          // Continue anyway since label is optional
        }
      }

      return bubble;
    } catch (error) {
      logger.warn('Mitosis animation: error creating bubble element:', error);
      return null;
    }
  }

  private createMotionTrails(
    bubble: HTMLElement,
    dx: number,
    dy: number,
    duration: number,
    container: HTMLElement
  ): void {
    try {
      // Validate inputs
      if (!bubble || !container || typeof dx !== 'number' || typeof dy !== 'number' || typeof duration !== 'number') {
        logger.warn('Invalid parameters for motion trails');
        return;
      }

      // Create multiple trail elements along the path
      const trailCount = 5;
      let bubbleRect, containerRect, bubbleComputedStyle;

      try {
        bubbleRect = bubble.getBoundingClientRect();
        containerRect = container.getBoundingClientRect();
        bubbleComputedStyle = window.getComputedStyle(bubble);
      } catch (measureError) {
        logger.warn('Error measuring elements for motion trails:', measureError);
        return;
      }

      // Use a document fragment for better performance
      const fragment = document.createDocumentFragment();
      const trails: HTMLElement[] = [];

      for (let i = 1; i <= trailCount; i++) {
        const progress = i / (trailCount + 1); // Don't place at start or end
        const trailX = dx * progress;
        const trailY = dy * progress;

        let trail: HTMLElement;
        try {
          trail = document.createElement('div');
          trail.className = 'mitosis-bubble-trail';
        } catch (createElementError) {
          logger.warn('Error creating trail element:', createElementError);
          continue; // Skip this iteration
        }

        // Copy bubble properties to trail
        try {
          trail.style.width = bubble.style.width || bubbleComputedStyle.width;
          trail.style.height = bubble.style.height || bubbleComputedStyle.height;
          trail.style.backgroundColor = bubbleComputedStyle.backgroundColor;
          trail.style.border = bubbleComputedStyle.border;
          trail.style.borderRadius = bubbleComputedStyle.borderRadius;

          // Position the trail
          trail.style.left = `${(bubbleRect.left - containerRect.left) + trailX}px`;
          trail.style.top = `${(bubbleRect.top - containerRect.top) + trailY}px`;
        } catch (styleError) {
          logger.warn('Error setting trail styles:', styleError);
          continue; // Skip this iteration
        }

        // Add to fragment instead of container directly
        try {
          fragment.appendChild(trail);
          trails.push(trail);
        } catch (appendError) {
          logger.warn('Error appending trail to fragment:', appendError);
          continue; // Skip this iteration
        }
      }

      // Append all trails to the container in a single operation
      try {
        container.appendChild(fragment);
      } catch (appendError) {
        logger.warn('Error appending trails fragment to container:', appendError);
        return;
      }

      // Set up removal timeouts for all trails
      trails.forEach(trail => {
        // Remove trail after animation completes
        setTimeout(() => {
          try {
            if (trail && trail.parentNode === container) {
              trail.remove();
            }
          } catch (removeError) {
            logger.warn('Error removing trail element:', removeError);
          }
        }, duration);
      });
    } catch (error) {
      logger.warn('Error creating motion trails:', error);
    }
  }

  private createParticleEffects(
    bubble: HTMLElement,
    dx: number,
    dy: number,
    container: HTMLElement
  ): void {
    try {
      // Validate inputs
      if (!bubble || !container || typeof dx !== 'number' || typeof dy !== 'number') {
        logger.warn('Invalid parameters for particle effects');
        return;
      }

      // Create particle effects at the split point
      const particleCount = 8;
      let bubbleRect, containerRect, bubbleComputedStyle, bubbleColor;

      try {
        bubbleRect = bubble.getBoundingClientRect();
        containerRect = container.getBoundingClientRect();
        bubbleComputedStyle = window.getComputedStyle(bubble);
        bubbleColor = bubbleComputedStyle.backgroundColor || '#4F46E5'; // Default color if not available
      } catch (measureError) {
        logger.warn('Error measuring elements for particle effects:', measureError);
        return;
      }

      // Use a document fragment for better performance
      const fragment = document.createDocumentFragment();
      const particles: HTMLElement[] = [];

      for (let i = 0; i < particleCount; i++) {
        let particle: HTMLElement;
        try {
          particle = document.createElement('div');
          particle.className = 'mitosis-particle';
        } catch (createElementError) {
          logger.warn('Error creating particle element:', createElementError);
          continue; // Skip this iteration
        }

        // Set particle properties
        try {
          const size = Math.floor(Math.random() * 6) + 2; // Random size between 2-8px
          particle.style.width = `${size}px`;
          particle.style.height = `${size}px`;
          particle.style.backgroundColor = bubbleColor;

          // Position at bubble's center
          const centerX = bubbleRect.left - containerRect.left + bubbleRect.width / 2;
          const centerY = bubbleRect.top - containerRect.top + bubbleRect.height / 2;
          particle.style.left = `${centerX}px`;
          particle.style.top = `${centerY}px`;
        } catch (propertyError) {
          logger.warn('Error setting particle properties:', propertyError);
          continue; // Skip this iteration
        }

        // Add to fragment instead of container directly
        try {
          fragment.appendChild(particle);
          particles.push(particle);
        } catch (appendError) {
          logger.warn('Error appending particle to fragment:', appendError);
          continue; // Skip this iteration
        }
      }

      // Append all particles to the container in a single operation
      try {
        container.appendChild(fragment);
      } catch (appendError) {
        logger.warn('Error appending particles fragment to container:', appendError);
        return;
      }

      // Animate each particle
      particles.forEach((particle, i) => {
        // Animate particle if browser supports Web Animations API
        try {
          if (typeof particle.animate === 'function') {
            const angle = (i * (360 / particleCount)) * (Math.PI / 180); // Evenly distribute angles
            const distance = Math.floor(Math.random() * 30) + 20; // Random distance
            const duration = Math.floor(Math.random() * 500) + 300; // Random duration

            const animation = particle.animate([
              {
                transform: 'translate(0, 0)',
                opacity: 1
              },
              {
                transform: `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px)`,
                opacity: 0
              }
            ], {
              duration: duration,
              easing: 'ease-out',
              fill: 'forwards'
            });

            // Remove particle after animation completes
            if (animation.onfinish) {
              animation.onfinish = () => {
                try {
                  if (particle && particle.parentNode === container) {
                    particle.remove();
                  }
                } catch (removeError) {
                  logger.warn('Error removing particle element:', removeError);
                }
              };
            } else {
              // Fallback for browsers that don't support onfinish
              setTimeout(() => {
                try {
                  if (particle && particle.parentNode === container) {
                    particle.remove();
                  }
                } catch (removeError) {
                  logger.warn('Error removing particle element (timeout):', removeError);
                }
              }, duration);
            }
          } else {
            // Fallback for browsers that don't support Web Animations API
            setTimeout(() => {
              try {
                if (particle && particle.parentNode === container) {
                  particle.remove();
                }
              } catch (removeError) {
                logger.warn('Error removing particle element (fallback):', removeError);
              }
            }, 1000); // Remove after 1 second if no animation support
          }
        } catch (animationError) {
          logger.warn('Error animating particle:', animationError);
          // Still remove the particle after a timeout even if animation fails
          setTimeout(() => {
            try {
              if (particle && particle.parentNode === container) {
                particle.remove();
              }
            } catch (removeError) {
              logger.warn('Error removing particle element (animation error):', removeError);
            }
          }, 1000);
        }
      });
    } catch (error) {
      logger.warn('Error creating particle effects:', error);
    }
  }

  private animateBubbleSplit(
    bubble: HTMLElement,
    dx: number,
    dy: number,
    config: MitosisConfig
  ): Animation | undefined {
    try {
      // Validate inputs
      if (!bubble || !(bubble instanceof HTMLElement)) {
        console.warn('Mitosis animation: invalid bubble element for animation');
        return undefined;
      }

      if (typeof dx !== 'number' || typeof dy !== 'number' || !isFinite(dx) || !isFinite(dy)) {
        console.warn('Mitosis animation: invalid animation parameters');
        return undefined;
      }

      // Validate config
      if (!config || typeof config !== 'object') {
        console.warn('Mitosis animation: invalid config provided for animation');
        return undefined;
      }

      // Sanitize values to prevent extremely large translations
      const sanitizedDx = Math.max(-10000, Math.min(10000, dx));
      const sanitizedDy = Math.max(-10000, Math.min(10000, dy));
      const sanitizedDuration = Math.max(100, Math.min(10000,
        typeof config.animationDuration === 'number' && isFinite(config.animationDuration) ? config.animationDuration : 1500)); // Between 100ms and 10s
      const sanitizedBounce = Math.max(0, Math.min(1,
        typeof config.bounceIntensity === 'number' && isFinite(config.bounceIntensity) ? config.bounceIntensity : 0.3)); // Between 0 and 1
      const rotationIntensity = Math.max(0, Math.min(1,
        typeof config.rotationIntensity === 'number' && isFinite(config.rotationIntensity) ? config.rotationIntensity : 0)); // Between 0 and 1
      const rotationDegrees = rotationIntensity * 360; // Max 360 degrees of rotation
      const opacityEffect = config.opacityEffect !== undefined ? config.opacityEffect : true;
      const particleEffects = config.particleEffects !== undefined ? config.particleEffects : false;
      const easingFunction = typeof config.easingFunction === 'string' && config.easingFunction ? config.easingFunction : 'cubic-bezier(0.25, 0.1, 0.25, 1)';
      const trailEffect = config.trailEffect !== undefined ? config.trailEffect : false;

      // Get container from bubble's parent
      const container = bubble.parentElement;

      // Check if browser supports Web Animations API
      if (typeof bubble.animate === 'function') {
        // Create motion trails if enabled
        if (trailEffect && container) {
          this.createMotionTrails(bubble, sanitizedDx, sanitizedDy, sanitizedDuration, container);
        }

        // Create particle effects if enabled
        if (particleEffects && container) {
          this.createParticleEffects(bubble, sanitizedDx, sanitizedDy, container);
        }

        // Use CSS animations for smooth performance
        let keyframes;
        try {
          if (opacityEffect) {
            keyframes = [
              { transform: 'translate(0, 0) scale(1)', opacity: 1 },
              { transform: `translate(${sanitizedDx * 0.3}px, ${sanitizedDy * 0.3}px) scale(0.8) rotate(${rotationDegrees * 0.3}deg)`, opacity: 0.9 },
              { transform: `translate(${sanitizedDx * 0.7}px, ${sanitizedDy * 0.7}px) scale(0.9) rotate(${rotationDegrees * 0.7}deg)`, opacity: 0.85 },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`,
                opacity: 0.8,
                offset: 0.8
              },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(${1 + sanitizedBounce * 0.2}) rotate(${rotationDegrees}deg)`,
                opacity: 1,
                offset: 0.9
              },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`,
                opacity: 1,
                offset: 1.0
              }
            ];
          } else {
            keyframes = [
              { transform: 'translate(0, 0) scale(1)' },
              { transform: `translate(${sanitizedDx * 0.3}px, ${sanitizedDy * 0.3}px) scale(0.8) rotate(${rotationDegrees * 0.3}deg)` },
              { transform: `translate(${sanitizedDx * 0.7}px, ${sanitizedDy * 0.7}px) scale(0.9) rotate(${rotationDegrees * 0.7}deg)` },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`,
                offset: 0.8
              },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(${1 + sanitizedBounce * 0.2}) rotate(${rotationDegrees}deg)`,
                offset: 0.9
              },
              {
                transform: `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`,
                offset: 1.0
              }
            ];
          }
        } catch (keyframeError) {
          console.warn('Mitosis animation: error creating keyframes:', keyframeError);
          return undefined;
        }

        let options: KeyframeAnimationOptions;
        try {
          options = {
            duration: sanitizedDuration,
            easing: easingFunction,
            fill: 'forwards'
          };
        } catch (optionsError) {
          console.warn('Mitosis animation: error creating animation options:', optionsError);
          return undefined;
        }

        // Check if bubble element is still attached to DOM
        try {
          if (typeof document !== 'undefined' && document.contains && !document.contains(bubble)) {
            console.warn('Mitosis animation: bubble element not in DOM, skipping animation');
            return undefined;
          }
        } catch (domError) {
          console.warn('Mitosis animation: error checking DOM containment:', domError);
          // Continue with animation since this is not critical
        }

        try {
          const animation = bubble.animate(keyframes, options);

          // Add error handling for the animation itself
          if (animation && typeof animation.addEventListener === 'function') {
            animation.addEventListener('error', (event) => {
              logger.warn('Animation error event:', event);
              // Apply fallback style if animation fails
              try {
                bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1)`;
              } catch (fallbackError) {
                logger.warn('Fallback style application failed:', fallbackError);
              }
            });

            // Add completion handler to clean up when animation finishes
            if (typeof animation.onfinish === 'function') {
              animation.onfinish = () => {
                try {
                  // Remove transition class if it was added
                  bubble.classList.remove('mitosis-bubble-transition');
                } catch (finishError) {
                  logger.warn('Error in animation finish handler:', finishError);
                }
              };
            } else {
              // Fallback for browsers that don't support onfinish
              setTimeout(() => {
                try {
                  bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1)`;
                  bubble.classList.remove('mitosis-bubble-transition');
                } catch (timeoutError) {
                  logger.warn('Timeout fallback failed:', timeoutError);
                }
              }, sanitizedDuration);
            }
          } else {
            // Fallback for browsers that don't support addEventListener on animations
            setTimeout(() => {
              try {
                bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1)`;
                bubble.classList.remove('mitosis-bubble-transition');
              } catch (timeoutError) {
                logger.warn('Timeout fallback failed:', timeoutError);
              }
            }, sanitizedDuration);
          }

          return animation;
        } catch (animationError) {
          logger.warn('Animation API error, using fallback:', animationError);
          // Fallback: apply final styles directly
          try {
            bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1)`;
          } catch (fallbackError) {
            logger.warn('Fallback style application failed:', fallbackError);
          }
        }
      } else {
        // Fallback for browsers that don't support Web Animations API
        // Use CSS transitions with a simplified animation
        try {
          // Create motion trails if enabled
          if (trailEffect && container) {
            this.createMotionTrails(bubble, sanitizedDx, sanitizedDy, sanitizedDuration, container);
          }

          // Create particle effects if enabled
          if (particleEffects && container) {
            this.createParticleEffects(bubble, sanitizedDx, sanitizedDy, container);
          }

          // Add transition class for styling
          bubble.classList.add('mitosis-bubble-transition');

          // Set CSS variables for the animation parameters
          bubble.style.setProperty('--animation-duration', `${sanitizedDuration}ms`);
          bubble.style.setProperty('--easing-function', easingFunction);

          // Set initial transform state
          bubble.style.transform = 'translate(0, 0) scale(1)';

          // Apply opacity if enabled
          if (opacityEffect) {
            bubble.style.opacity = '1';
          }

          // Force reflow to ensure the initial state is applied
          void bubble.offsetWidth;

          // Apply the final transform with rotation and scale
          // We'll implement a simplified version of the bounce effect using a sequence of transitions
          bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`;

          // For a more complex animation, we'll need to chain transitions
          // First, do the main movement
          setTimeout(() => {
            try {
              // Apply bounce effect after the main movement
              if (sanitizedBounce > 0) {
                // Add a small bounce effect by scaling up and back down
                const bounceDuration = Math.min(sanitizedDuration * 0.1, 200);
                bubble.style.transition = `transform ${bounceDuration}ms ease-out`;
                bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(${1 + sanitizedBounce * 0.2}) rotate(${rotationDegrees}deg)`;

                // After the bounce, return to normal size
                setTimeout(() => {
                  if (bubble && bubble.style) {
                    const returnDuration = Math.min(sanitizedDuration * 0.1, 200);
                    bubble.style.transition = `transform ${returnDuration}ms ease-out`;
                    bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`;
                  }
                }, bounceDuration);
              }
            } catch (bounceError) {
              logger.warn('Bounce effect application failed:', bounceError);
              // Continue with basic transition even if bounce fails
            }
          }, sanitizedDuration * 0.8); // Apply bounce after main animation is nearly complete

          // Set a timeout to remove the transition after completion to prevent unintended animations
          setTimeout(() => {
            try {
              if (bubble && bubble.style) {
                bubble.style.transition = '';
                bubble.classList.remove('mitosis-bubble-transition');
              }
            } catch (cleanupError) {
              logger.warn('Transition cleanup failed:', cleanupError);
            }
          }, sanitizedDuration);
        } catch (transitionError) {
          logger.warn('CSS transition fallback failed:', transitionError);
          // Final fallback: apply styles directly
          try {
            bubble.style.transform = `translate(${sanitizedDx}px, ${sanitizedDy}px) scale(1) rotate(${rotationDegrees}deg)`;
          } catch (directError) {
            logger.warn('Direct style application failed:', directError);
          }
        }
      }

      return undefined;
    } catch (error) {
      console.warn('Mitosis animation: error during bubble animation:', error);
      // Ensure bubble ends in final position even if animation fails
      try {
        // Use safe values for fallback
        const safeDx = typeof dx === 'number' && isFinite(dx) ? dx : 0;
        const safeDy = typeof dy === 'number' && isFinite(dy) ? dy : 0;

        // Apply fallback transformation
        if (bubble && bubble.style) {
          bubble.style.transform = `translate(${safeDx}px, ${safeDy}px) scale(1)`;
        }
      } catch (fallbackError) {
        console.warn('Mitosis animation: fallback style application failed:', fallbackError);
      }

      return undefined;
    }
  }
}

export function createMitosisPlugin(): MitosisPlugin {
  let service: MitosisAnimationService;

  try {
    service = new MitosisAnimationService();
  } catch (error) {
    logger.warn('Mitosis plugin: error creating animation service, using fallback:', error);
    // Create a minimal service that just logs errors
    service = {
      executeSplitAnimation: async (_params: SplitAnimationParams): Promise<void> => {
        logger.warn('Mitosis animation service not available');
        return Promise.resolve();
      },
      executeEvolutionSplit: async (_params: EvolutionAnimationParams): Promise<void> => {
        logger.warn('Mitosis evolution animation service not available');
        return Promise.resolve();
      },
      cancelAllAnimations: () => {} // Add the missing method
    } as MitosisAnimationService;
  }

  return {
    initialize: (config: MitosisConfig) => {
      try {
        // Validate config before applying
        if (!config || typeof config !== 'object') {
          logger.warn('Mitosis plugin: invalid config provided to initialize');
          return;
        }

        // Validate each property individually to prevent partial failures
        const validatedConfig: Partial<MitosisConfig> = {};

        if ('enabled' in config) {
          try {
            validatedConfig.enabled = Boolean(config.enabled);
          } catch (enabledError) {
            logger.warn('Mitosis plugin: error validating enabled config property:', enabledError);
          }
        }

        if ('animationDuration' in config) {
          try {
            if (typeof config.animationDuration === 'number' && isFinite(config.animationDuration)) {
              validatedConfig.animationDuration = Math.max(100, Math.min(10000, config.animationDuration));
            } else {
              logger.warn('Mitosis plugin: invalid animationDuration, using default');
              validatedConfig.animationDuration = 1500;
            }
          } catch (durationError) {
            logger.warn('Mitosis plugin: error validating animationDuration config property:', durationError);
            validatedConfig.animationDuration = 1500;
          }
        }

        if ('bounceIntensity' in config) {
          try {
            if (typeof config.bounceIntensity === 'number' && isFinite(config.bounceIntensity)) {
              validatedConfig.bounceIntensity = Math.max(0, Math.min(1, config.bounceIntensity));
            } else {
              logger.warn('Mitosis plugin: invalid bounceIntensity, using default');
              validatedConfig.bounceIntensity = 0.3;
            }
          } catch (bounceError) {
            logger.warn('Mitosis plugin: error validating bounceIntensity config property:', bounceError);
            validatedConfig.bounceIntensity = 0.3;
          }
        }

        if ('splitDelay' in config) {
          try {
            if (typeof config.splitDelay === 'number' && isFinite(config.splitDelay)) {
              validatedConfig.splitDelay = Math.max(0, Math.min(5000, config.splitDelay));
            } else {
              logger.warn('Mitosis plugin: invalid splitDelay, using default');
              validatedConfig.splitDelay = 300;
            }
          } catch (delayError) {
            logger.warn('Mitosis plugin: error validating splitDelay config property:', delayError);
            validatedConfig.splitDelay = 300;
          }
        }

        if ('colorVariation' in config) {
          try {
            if (typeof config.colorVariation === 'number' && isFinite(config.colorVariation)) {
              validatedConfig.colorVariation = Math.max(0, Math.min(1, config.colorVariation));
            } else {
              logger.warn('Mitosis plugin: invalid colorVariation, using default');
              validatedConfig.colorVariation = 0.1;
            }
          } catch (colorError) {
            logger.warn('Mitosis plugin: error validating colorVariation config property:', colorError);
            validatedConfig.colorVariation = 0.1;
          }
        }

        if ('rotationIntensity' in config) {
          try {
            if (typeof config.rotationIntensity === 'number' && isFinite(config.rotationIntensity)) {
              validatedConfig.rotationIntensity = Math.max(0, Math.min(1, config.rotationIntensity));
            } else {
              logger.warn('Mitosis plugin: invalid rotationIntensity, using default');
              validatedConfig.rotationIntensity = 0.2;
            }
          } catch (rotationError) {
            logger.warn('Mitosis plugin: error validating rotationIntensity config property:', rotationError);
            validatedConfig.rotationIntensity = 0.2;
          }
        }

        if ('opacityEffect' in config) {
          try {
            validatedConfig.opacityEffect = Boolean(config.opacityEffect);
          } catch (opacityError) {
            logger.warn('Mitosis plugin: error validating opacityEffect config property:', opacityError);
            validatedConfig.opacityEffect = true;
          }
        }

        if ('trailEffect' in config) {
          try {
            validatedConfig.trailEffect = Boolean(config.trailEffect);
          } catch (trailError) {
            logger.warn('Mitosis plugin: error validating trailEffect config property:', trailError);
            validatedConfig.trailEffect = false;
          }
        }

        if ('easingFunction' in config) {
          try {
            if (typeof config.easingFunction === 'string') {
              validatedConfig.easingFunction = config.easingFunction;
            } else {
              logger.warn('Mitosis plugin: invalid easingFunction, using default');
              validatedConfig.easingFunction = 'cubic-bezier(0.25, 0.1, 0.25, 1)';
            }
          } catch (easingError) {
            logger.warn('Mitosis plugin: error validating easingFunction config property:', easingError);
            validatedConfig.easingFunction = 'cubic-bezier(0.25, 0.1, 0.25, 1)';
          }
        }

        if ('particleEffects' in config) {
          try {
            validatedConfig.particleEffects = Boolean(config.particleEffects);
          } catch (particleError) {
            logger.warn('Mitosis plugin: error validating particleEffects config property:', particleError);
            validatedConfig.particleEffects = false;
          }
        }

        // Merge with existing config to preserve values not in the new config
        const mergedConfig = {
          ...globalState.config,
          ...validatedConfig
        };

        updateGlobalState({
          config: mergedConfig,
          enabled: validatedConfig.enabled ?? globalState.enabled
        });
      } catch (error) {
        logger.warn('Mitosis plugin: error during initialization:', error);
        // Set default values to ensure plugin remains functional
        try {
          updateGlobalState({
            config: {
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
            },
            enabled: false
          });
        } catch (fallbackError) {
          logger.warn('Mitosis plugin: error setting fallback config:', fallbackError);
        }
      }
    },

    triggerMitosisSplit: async (params: SplitAnimationParams) => {
      try {
        // Validate parameters
        if (!params || typeof params !== 'object') {
          console.warn('Mitosis plugin: invalid parameters object provided to triggerMitosisSplit');
          return;
        }

        if (!params.parentNode || typeof params.parentNode !== 'object') {
          console.warn('Mitosis plugin: invalid parent node provided to triggerMitosisSplit');
          return;
        }

        if (!Array.isArray(params.childNodes) || params.childNodes.length === 0) {
          console.warn('Mitosis plugin: invalid or empty child nodes array provided to triggerMitosisSplit');
          return;
        }

        // Validate container ref
        if (!params.containerRef || typeof params.containerRef !== 'object') {
          console.warn('Mitosis plugin: invalid container ref provided to triggerMitosisSplit');
          return;
        }

        // Validate parent node properties
        const parentNode = params.parentNode;
        if (typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
            typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string' ||
            !isFinite(parentNode.x) || !isFinite(parentNode.y) || !isFinite(parentNode.radius)) {
          console.warn('Mitosis plugin: invalid parent node properties');
          return;
        }

        // Validate child nodes
        const validChildNodes = params.childNodes.filter(node =>
          node &&
          typeof node === 'object' &&
          typeof node.x === 'number' &&
          typeof node.y === 'number' &&
          typeof node.radius === 'number' &&
          typeof node.color === 'string' &&
          isFinite(node.x) &&
          isFinite(node.y) &&
          isFinite(node.radius)
        );

        if (validChildNodes.length === 0) {
          console.warn('Mitosis plugin: no valid child nodes after validation');
          return;
        }

        // Check if plugin is enabled and not already animating
        let currentState;
        try {
          currentState = { ...globalState }; // Create a snapshot to avoid race conditions
        } catch (stateSnapshotError) {
          logger.warn('Mitosis plugin: error getting state snapshot:', stateSnapshotError);
          return;
        }

        if (!currentState.enabled || currentState.isAnimating) {
          return;
        }

        try {
          updateGlobalState({ isAnimating: true });
        } catch (updateStateError) {
          logger.warn('Mitosis plugin: error updating animation state:', updateStateError);
          return;
        }

        try {
          await service.executeSplitAnimation(params);
        } catch (executionError) {
          logger.warn('Mitosis plugin: error executing split animation:', executionError);
          // Still reset the animation state
          try {
            updateGlobalState({ isAnimating: false });
          } catch (resetError) {
            logger.warn('Mitosis plugin: error resetting animation state after execution error:', resetError);
          }
          return;
        }
      } catch (error) {
        logger.warn('Mitosis plugin: error triggering split animation:', error);
        // Reset animation state to prevent lockup
        try {
          updateGlobalState({ isAnimating: false });
        } catch (stateError) {
          logger.warn('Mitosis plugin: error resetting animation state after trigger error:', stateError);
        }
      }
    },

    updateConfig: (config: Partial<MitosisConfig>) => {
      try {
        if (!config || typeof config !== 'object') {
          console.warn('Mitosis plugin: invalid config provided to updateConfig');
          return;
        }

        // Create a safe config object with only valid properties
        const safeUpdates: Partial<MitosisConfig> = {};

        if ('enabled' in config) {
          try {
            safeUpdates.enabled = Boolean(config.enabled);
          } catch (error) {
            console.warn('Mitosis plugin: error validating enabled config property:', error);
          }
        }

        if ('animationDuration' in config) {
          try {
            if (typeof config.animationDuration === 'number' && isFinite(config.animationDuration)) {
              safeUpdates.animationDuration = Math.max(100, Math.min(10000, config.animationDuration));
            } else {
              console.warn('Mitosis plugin: invalid animationDuration value, skipping update');
            }
          } catch (error) {
            console.warn('Mitosis plugin: error validating animationDuration config property:', error);
          }
        }

        if ('bounceIntensity' in config) {
          try {
            if (typeof config.bounceIntensity === 'number' && isFinite(config.bounceIntensity)) {
              safeUpdates.bounceIntensity = Math.max(0, Math.min(1, config.bounceIntensity));
            } else {
              console.warn('Mitosis plugin: invalid bounceIntensity value, skipping update');
            }
          } catch (error) {
            console.warn('Mitosis plugin: error validating bounceIntensity config property:', error);
          }
        }

        if ('splitDelay' in config) {
          try {
            if (typeof config.splitDelay === 'number' && isFinite(config.splitDelay)) {
              safeUpdates.splitDelay = Math.max(0, Math.min(5000, config.splitDelay));
            } else {
              console.warn('Mitosis plugin: invalid splitDelay value, skipping update');
            }
          } catch (error) {
            console.warn('Mitosis plugin: error validating splitDelay config property:', error);
          }
        }

        if ('colorVariation' in config) {
          try {
            if (typeof config.colorVariation === 'number' && isFinite(config.colorVariation)) {
              safeUpdates.colorVariation = Math.max(0, Math.min(1, config.colorVariation));
            } else {
              console.warn('Mitosis plugin: invalid colorVariation value, skipping update');
            }
          } catch (error) {
            console.warn('Mitosis plugin: error validating colorVariation config property:', error);
          }
        }

        // Update the global state with validated properties
        try {
          updateGlobalState({
            config: { ...globalState.config, ...safeUpdates }
          });
        } catch (stateError) {
          console.warn('Mitosis plugin: error updating global state with config:', stateError);
        }
      } catch (error) {
        console.warn('Mitosis plugin: error updating configuration:', error);
      }
    },

    getState: () => {
      try {
        // Create a deep copy to prevent external mutations
        let stateCopy;
        try {
          stateCopy = JSON.parse(JSON.stringify(globalState));
        } catch (copyError) {
          logger.warn('Mitosis plugin: error creating state copy, using defaults:', copyError);
          // Return a safe default state if cloning fails
          return {
            config: {
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
            },
            isAnimating: false,
            enabled: false,
            lastAnimationTime: null
          };
        }

        // Validate the copied state before returning
        if (!stateCopy || typeof stateCopy !== 'object') {
          logger.warn('Mitosis plugin: invalid state copy, using defaults');
          return {
            config: {
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
            },
            isAnimating: false,
            enabled: false,
            lastAnimationTime: null
          };
        }

        // Validate each property individually
        const validatedState: MitosisPluginState = {
          config: {
            enabled: typeof stateCopy.config?.enabled === 'boolean' ? stateCopy.config.enabled : false,
            animationDuration: typeof stateCopy.config?.animationDuration === 'number' && isFinite(stateCopy.config.animationDuration)
              ? Math.max(100, Math.min(10000, stateCopy.config.animationDuration)) : 1500,
            bounceIntensity: typeof stateCopy.config?.bounceIntensity === 'number' && isFinite(stateCopy.config.bounceIntensity)
              ? Math.max(0, Math.min(1, stateCopy.config.bounceIntensity)) : 0.3,
            splitDelay: typeof stateCopy.config?.splitDelay === 'number' && isFinite(stateCopy.config.splitDelay)
              ? Math.max(0, Math.min(5000, stateCopy.config.splitDelay)) : 300,
            colorVariation: typeof stateCopy.config?.colorVariation === 'number' && isFinite(stateCopy.config.colorVariation)
              ? Math.max(0, Math.min(1, stateCopy.config.colorVariation)) : 0.1,
            rotationIntensity: typeof stateCopy.config?.rotationIntensity === 'number' && isFinite(stateCopy.config.rotationIntensity)
              ? Math.max(0, Math.min(1, stateCopy.config.rotationIntensity)) : 0.2,
            opacityEffect: typeof stateCopy.config?.opacityEffect === 'boolean' ? stateCopy.config.opacityEffect : true,
            trailEffect: typeof stateCopy.config?.trailEffect === 'boolean' ? stateCopy.config.trailEffect : false,
            easingFunction: typeof stateCopy.config?.easingFunction === 'string' ? stateCopy.config.easingFunction : 'cubic-bezier(0.25, 0.1, 0.25, 1)',
            particleEffects: typeof stateCopy.config?.particleEffects === 'boolean' ? stateCopy.config.particleEffects : false
          },
          isAnimating: typeof stateCopy.isAnimating === 'boolean' ? stateCopy.isAnimating : false,
          enabled: typeof stateCopy.enabled === 'boolean' ? stateCopy.enabled : false,
          lastAnimationTime: stateCopy.lastAnimationTime
        };

        return validatedState;
      } catch (error) {
        logger.warn('Mitosis plugin: error getting state:', error);
        // Return a safe default state
        return {
          config: {
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
          },
          isAnimating: false,
          enabled: false,
          lastAnimationTime: null
        };
      }
    },

    toggleEnabled: () => {
      try {
        updateGlobalState({ enabled: !globalState.enabled });
      } catch (error) {
        console.warn('Mitosis plugin: error toggling enabled state:', error);
      }
    },

    isEnabled: () => {
      try {
        return globalState.enabled;
      } catch (error) {
        console.warn('Mitosis plugin: error checking enabled state:', error);
        return false; // Safe default
      }
    },

    cleanup: () => {
      try {
        // Cancel all active animations
        if (service && typeof service.cancelAllAnimations === 'function') {
          try {
            service.cancelAllAnimations();
          } catch (cancelError) {
            logger.warn('Mitosis plugin: error canceling animations:', cancelError);
          }
        }

        // Reset the global state
        try {
          updateGlobalState({
            config: {
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
            },
            isAnimating: false,
            enabled: false,
            lastAnimationTime: null
          });
        } catch (stateError) {
          logger.warn('Mitosis plugin: error updating state during cleanup:', stateError);
          // Fallback to direct state reset if updateGlobalState fails
          try {
            globalState = {
              config: {
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
              },
              isAnimating: false,
              enabled: false,
              lastAnimationTime: null
            };
          } catch (directResetError) {
            logger.warn('Mitosis plugin: error performing direct state reset:', directResetError);
          }
        }
      } catch (error) {
        logger.warn('Mitosis plugin: error during cleanup:', error);
      }
    },

    getPerformanceMetrics: () => {
      try {
        if (service && typeof service.getPerformanceMetrics === 'function') {
          return service.getPerformanceMetrics();
        }
        // Return default values if service is not available
        return {
          avgDuration: 0,
          activeAnimations: 0,
          queuedAnimations: 0
        };
      } catch (error) {
        console.warn('Mitosis plugin: error getting performance metrics:', error);
        return {
          avgDuration: 0,
          activeAnimations: 0,
          queuedAnimations: 0
        };
      }
    },

    applyPreset: (preset: AnimationPreset) => {
      try {
        let newConfig: Partial<MitosisConfig>;

        switch (preset) {
          case 'smooth':
            newConfig = {
              animationDuration: 2000,
              bounceIntensity: 0.1,
              rotationIntensity: 0.1,
              opacityEffect: true,
              trailEffect: false,
              easingFunction: 'cubic-bezier(0.23, 1, 0.32, 1)'
            };
            break;
          case 'dramatic':
            newConfig = {
              animationDuration: 1800,
              bounceIntensity: 0.6,
              rotationIntensity: 0.7,
              opacityEffect: true,
              trailEffect: true,
              particleEffects: true,
              easingFunction: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)'
            };
            break;
          case 'subtle':
            newConfig = {
              animationDuration: 1200,
              bounceIntensity: 0.1,
              rotationIntensity: 0.05,
              opacityEffect: false,
              trailEffect: false,
              particleEffects: false,
              easingFunction: 'ease-out'
            };
            break;
          case 'fast':
            newConfig = {
              animationDuration: 800,
              bounceIntensity: 0.2,
              rotationIntensity: 0.3,
              opacityEffect: true,
              trailEffect: false,
              particleEffects: false,
              easingFunction: 'ease-in-out'
            };
            break;
          case 'default':
          case 'custom':
          default:
            newConfig = {
              animationDuration: 1500,
              bounceIntensity: 0.3,
              rotationIntensity: 0.2,
              opacityEffect: true,
              trailEffect: false,
              particleEffects: false,
              easingFunction: 'cubic-bezier(0.25, 0.1, 0.25, 1)'
            };
            break;
        }

        // Update the configuration
        mitosisPlugin.updateConfig(newConfig);
      } catch (error) {
        console.warn('Mitosis plugin: error applying preset:', error);
      }
    },

    triggerEvolutionSplit: async (params: EvolutionAnimationParams) => {
      try {
        // Validate parameters
        if (!params || typeof params !== 'object') {
          console.warn('Mitosis plugin: invalid parameters object provided to triggerEvolutionSplit');
          return;
        }

        if (!params.parentNode || !Array.isArray(params.childNodes) || params.childNodes.length === 0) {
          console.warn('Mitosis plugin: invalid parent or child nodes provided to triggerEvolutionSplit');
          return;
        }

        if (!params.containerRef || typeof params.containerRef !== 'object') {
          console.warn('Mitosis plugin: invalid container ref provided to triggerEvolutionSplit');
          return;
        }

        // Validate parent node properties
        const parentNode = params.parentNode;
        if (typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
            typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string' ||
            !isFinite(parentNode.x) || !isFinite(parentNode.y) || !isFinite(parentNode.radius)) {
          console.warn('Mitosis plugin: invalid parent node properties');
          return;
        }

        // Validate child nodes
        const validChildNodes = params.childNodes.filter(node =>
          node &&
          typeof node === 'object' &&
          typeof node.x === 'number' &&
          typeof node.y === 'number' &&
          typeof node.radius === 'number' &&
          typeof node.color === 'string' &&
          isFinite(node.x) &&
          isFinite(node.y) &&
          isFinite(node.radius)
        );

        if (validChildNodes.length === 0) {
          console.warn('Mitosis plugin: no valid child nodes after validation');
          return;
        }

        // Check if plugin is enabled and not already animating
        let currentState;
        try {
          currentState = { ...globalState }; // Create a snapshot to avoid race conditions
        } catch (stateSnapshotError) {
          logger.warn('Mitosis plugin: error getting state snapshot:', stateSnapshotError);
          return;
        }

        if (!currentState.enabled || currentState.isAnimating) {
          return;
        }

        try {
          updateGlobalState({ isAnimating: true });
        } catch (updateStateError) {
          logger.warn('Mitosis plugin: error updating animation state:', updateStateError);
          return;
        }

        // Determine evolution type
        const evolutionType = params.evolutionType || 'standard';

        if (evolutionType === 'survival-of-fittest') {
          // For survival-of-fittest, we'll animate the split first, then apply the survival logic
          try {
            await service.executeEvolutionSplit(params);
          } catch (evolutionError) {
            logger.warn('Mitosis plugin: error executing evolution split:', evolutionError);
            // Fallback to standard animation if evolution fails
            try {
              await service.executeSplitAnimation({
                parentNode: params.parentNode,
                childNodes: params.childNodes,
                containerRef: params.containerRef
              });
            } catch (fallbackError) {
              logger.warn('Mitosis plugin: fallback to standard animation also failed:', fallbackError);
            }
          }
        } else {
          // For standard evolution, just do a regular split
          try {
            await service.executeSplitAnimation({
              parentNode: params.parentNode,
              childNodes: params.childNodes,
              containerRef: params.containerRef
            });
          } catch (standardError) {
            logger.warn('Mitosis plugin: standard split animation failed:', standardError);
            // Try to reset state and continue
            try {
              updateGlobalState({ isAnimating: false });
            } catch (resetError) {
              logger.warn('Mitosis plugin: error resetting animation state after standard error:', resetError);
            }
          }
        }
      } catch (error) {
        logger.warn('Mitosis plugin: error triggering evolution split:', error);
        // Reset animation state to prevent lockup
        try {
          updateGlobalState({ isAnimating: false });
        } catch (stateError) {
          logger.warn('Mitosis plugin: error resetting animation state after trigger error:', stateError);
        }

        // Try fallback to standard animation
        try {
          await service.executeSplitAnimation({
            parentNode: params.parentNode,
            childNodes: params.childNodes,
            containerRef: params.containerRef
          });
        } catch (fallbackError) {
          logger.warn('Mitosis plugin: fallback to standard animation also failed:', fallbackError);
        }
      }
    },

    triggerBatchMitosis: async (params: BatchAnimationParams) => {
      try {
        // Validate parameters
        if (!params || typeof params !== 'object') {
          logger.warn('Mitosis plugin: invalid parameters object provided to triggerBatchMitosis');
          return;
        }

        if (!Array.isArray(params.parentNodes) || !Array.isArray(params.childNodeGroups)) {
          logger.warn('Mitosis plugin: invalid parentNodes or childNodeGroups arrays provided to triggerBatchMitosis');
          return;
        }

        if (params.parentNodes.length !== params.childNodeGroups.length) {
          logger.warn('Mitosis plugin: parentNodes and childNodeGroups arrays must have the same length');
          return;
        }

        if (!params.containerRef || typeof params.containerRef !== 'object') {
          logger.warn('Mitosis plugin: invalid container ref provided to triggerBatchMitosis');
          return;
        }

        if (!params.containerRef.current) {
          logger.warn('Mitosis plugin: container ref current value is null');
          return;
        }

        // Process each parent-child group with a small delay between them
        for (let i = 0; i < params.parentNodes.length; i++) {
          try {
            const parentNode = params.parentNodes[i];
            const childNodes = params.childNodeGroups[i];

            // Validate parent node
            if (!parentNode || typeof parentNode !== 'object') {
              logger.warn(`Mitosis plugin: invalid parent node at index ${i}`);
              continue;
            }

            // Validate child nodes
            if (!Array.isArray(childNodes) || childNodes.length === 0) {
              logger.warn(`Mitosis plugin: invalid or empty child nodes array at index ${i}`);
              continue;
            }

            // Validate parent node properties
            if (typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
                typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string' ||
                !isFinite(parentNode.x) || !isFinite(parentNode.y) || !isFinite(parentNode.radius)) {
              logger.warn(`Mitosis plugin: invalid parent node properties at index ${i}`);
              continue;
            }

            // Validate child nodes
            let validChildNodes: BubbleNode[] = [];
            try {
              validChildNodes = childNodes.filter(node =>
                node &&
                typeof node === 'object' &&
                typeof node.x === 'number' &&
                typeof node.y === 'number' &&
                typeof node.radius === 'number' &&
                typeof node.color === 'string' &&
                isFinite(node.x) &&
                isFinite(node.y) &&
                isFinite(node.radius)
              );
            } catch (filterError) {
              logger.warn(`Mitosis plugin: error filtering child nodes at index ${i}:`, filterError);
              continue;
            }

            if (validChildNodes.length === 0) {
              logger.warn(`Mitosis plugin: no valid child nodes after validation at index ${i}`);
              continue;
            }

            // Trigger the animation for this parent-child group
            const splitParams: SplitAnimationParams = {
              parentNode,
              childNodes: validChildNodes,
              containerRef: params.containerRef
            };

            try {
              await mitosisPlugin.triggerMitosisSplit(splitParams);
            } catch (splitError) {
              logger.warn(`Mitosis plugin: error triggering split animation at index ${i}:`, splitError);
              // Continue with other animations
              continue;
            }

            // Add a small delay between animations to prevent overwhelming the system
            try {
              await new Promise(resolve => setTimeout(resolve, 100));
            } catch (delayError) {
              logger.warn(`Mitosis plugin: error in delay at index ${i}:`, delayError);
              // Continue with next animation
            }
          } catch (itemError) {
            logger.warn(`Mitosis plugin: error processing item at index ${i}:`, itemError);
            // Continue with next item
            continue;
          }
        }
      } catch (error) {
        logger.warn('Mitosis plugin: error triggering batch mitosis:', error);
      }
    }
  };
}

// Export the plugin instance
export const mitosisPlugin = createMitosisPlugin();

// Export a cleanup function to reset the global state
export function resetMitosisPluginState(): void {
  try {
    // Acquire lock before resetting state
    let attempts = 0;
    while (stateUpdateLock && attempts < MAX_STATE_UPDATE_ATTEMPTS) {
      attempts++;
      // Brief pause to avoid busy waiting
      try {
        setTimeout(() => {}, 1);
      } catch (pauseError) {
        // If timeout fails, continue without pause
      }
    }

    if (attempts >= MAX_STATE_UPDATE_ATTEMPTS) {
      logger.warn('Failed to acquire state update lock for reset after maximum attempts');
      return;
    }

    // Acquire the lock
    stateUpdateLock = true;

    globalState = {
      config: {
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
      },
      isAnimating: false,
      enabled: false,
      lastAnimationTime: null
    };
  } catch (error) {
    logger.warn('Mitosis plugin: error resetting global state:', error);
  } finally {
    // Always release the lock
    stateUpdateLock = false;
  }
}