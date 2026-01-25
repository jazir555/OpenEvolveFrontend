import React, { useEffect, useRef, useState } from 'react';
import { SplitAnimationParams, BubbleNode } from '../types/plugin-types';
import { mitosisPlugin } from '../utils/createMitosisPlugin';
import { logger } from '../utils/logger';

interface MitosisAnimationProps {
  parentNode?: BubbleNode;
  childNodes?: BubbleNode[];
  containerRef?: React.RefObject<HTMLDivElement>;
  enabled?: boolean;
  evolutionType?: 'survival-of-fittest' | 'standard' | 'speciation';
  survivorIndices?: number[]; // Indices of child nodes that survive (for survival-of-fittest)
  nextEvolution?: EvolutionAnimationParams; // For chaining evolutions
}

export const MitosisAnimation: React.FC<MitosisAnimationProps> = ({
  parentNode,
  childNodes = [],
  containerRef: externalContainerRef,
  enabled = true,
  evolutionType,
  survivorIndices,
  nextEvolution
}) => {
  const internalContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = externalContainerRef || internalContainerRef;
  const [error, setError] = useState<string | null>(null);
  const animationId = useRef<number>(0); // Track animation instances

  useEffect(() => {
    logger.info('MitosisAnimation component effect triggered');

    // Validate inputs
    if (!enabled || !parentNode || !Array.isArray(childNodes) || childNodes.length === 0) {
      logger.debug('Skipping animation - inputs are invalid', { enabled, hasParentNode: !!parentNode, childNodesLength: childNodes?.length });
      // Don't trigger animation if inputs are invalid
      return;
    }

    // Increment animation ID to track this instance
    const currentAnimationId = ++animationId.current;
    logger.debug(`Starting animation with ID: ${currentAnimationId}`);

    const triggerAnimation = async () => {
      try {
        // Double-check container ref exists
        const effectiveContainerRef = containerRef || internalContainerRef;
        if (!effectiveContainerRef.current) {
          logger.warn('Container ref not available');
          return;
        }

        // Validate that all required properties exist on nodes
        if (!parentNode ||
            !parentNode.id ||
            typeof parentNode.x !== 'number' ||
            typeof parentNode.y !== 'number' ||
            typeof parentNode.radius !== 'number' ||
            typeof parentNode.color !== 'string' ||
            !isFinite(parentNode.x) ||
            !isFinite(parentNode.y) ||
            !isFinite(parentNode.radius)) {
          logger.warn('Invalid parent node properties', { parentNode });
          return;
        }

        // Validate child nodes
        const validChildNodes = childNodes.filter(node =>
          node &&
          node.id &&
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
          return;
        }

        logger.info(`Validated ${validChildNodes.length} valid child nodes for animation`);

        // Check if this animation is still the current one
        if (currentAnimationId !== animationId.current) {
          logger.debug('Skipping animation - another animation has started');
          return; // Another animation has started, skip this one
        }

        // Check if this is an evolution animation
        if (evolutionType === 'survival-of-fittest') {
          try {
            const params: EvolutionAnimationParams = {
              parentNode,
              childNodes: validChildNodes,
              containerRef: effectiveContainerRef,
              evolutionType,
              survivorIndices,
              nextEvolution
            };

            logger.info('Triggering evolution split animation');
            await mitosisPlugin.triggerEvolutionSplit(params);
            logger.info('Evolution split animation completed');
          } catch (evolutionError) {
            logger.error('Error in evolution split animation:', evolutionError);
            // Fallback to standard animation if evolution fails
            const params: SplitAnimationParams = {
              parentNode,
              childNodes: validChildNodes,
              containerRef: effectiveContainerRef
            };

            logger.info('Falling back to standard mitosis split animation');
            await mitosisPlugin.triggerMitosisSplit(params);
            logger.info('Fallback mitosis split animation completed');
          }
        } else {
          const params: SplitAnimationParams = {
            parentNode,
            childNodes: validChildNodes,
            containerRef: effectiveContainerRef
          };

          logger.info('Triggering mitosis split animation');
          await mitosisPlugin.triggerMitosisSplit(params);
          logger.info('Mitosis split animation completed');
        }
      } catch (err) {
        logger.error('Error triggering animation:', err);
        setError('Animation failed to start');
        // Don't rethrow - we want the component to continue working
      }
    };

    // Trigger the animation when component mounts or when childNodes change
    triggerAnimation();
  }, [parentNode, childNodes, enabled, containerRef, evolutionType, survivorIndices, nextEvolution]);

  // Cleanup function to clear any ongoing animations
  useEffect(() => {
    return () => {
      logger.info('Cleaning up MitosisAnimation component');
      // Increment animation ID to cancel any ongoing animations
      animationId.current++;

      // Also call the plugin's cleanup method to ensure all animations are cancelled
      mitosisPlugin.cleanup();
      logger.info('MitosisAnimation component cleanup completed');
    };
  }, []);

  if (error) {
    // Render nothing or a fallback UI if there was an error
    return null;
  }

  return (
    <div
      ref={containerRef || internalContainerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'visible'
      }}
      aria-label="Mitosis animation container"
      role="region"
    >
      {/* The animation elements are added dynamically to this container */}
    </div>
  );
};