/**
 * Example: Modified BubbleLab Visualization Component
 *
 * This file shows how the existing BubbleLab visualization component
 * would be enhanced with the mitosis plugin.
 */

import React, { useState, useRef, useEffect } from 'react';
import { mitosisPlugin, MitosisAnimation, MitosisSettings } from '../bubblelabs-mitosis-plugin/src/index';

// Example of how to enhance an existing visualization component
const EnhancedVisualizationComponent = ({ nodes = [], edges = [], onEvolution }) => {
  const [showMitosisControls, setShowMitosisControls] = useState(false);
  const [mitosisEnabled, setMitosisEnabled] = useState(false);
  const containerRef = useRef(null);
  const intervalId = useRef<NodeJS.Timeout | null>(null);

  // Initialize mitosis plugin when component mounts
  useEffect(() => {
    try {
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

      // Update local state when plugin state changes
      const updateState = () => {
        try {
          const state = mitosisPlugin.getState();
          setMitosisEnabled(state.enabled);
        } catch (error) {
          console.warn('Mitosis component: error getting plugin state:', error);
        }
      };

      // Set up periodic state updates
      updateState();
      intervalId.current = setInterval(updateState, 1000); // Update every second
    } catch (error) {
      console.warn('Mitosis component: error initializing plugin:', error);
    }

    // Cleanup function
    return () => {
      if (intervalId.current) {
        clearInterval(intervalId.current);
        intervalId.current = null;
      }

      // Clean up mitosis plugin resources
      mitosisPlugin.cleanup();
    };
  }, []);

  // Handle evolution events to trigger mitosis animation
  const handleEvolution = (parentNodeId, childNodeIds) => {
    try {
      if (onEvolution) {
        onEvolution(parentNodeId, childNodeIds);

        // If mitosis is enabled, trigger the animation
        if (mitosisEnabled) {
          // Validate inputs
          if (!parentNodeId || !Array.isArray(childNodeIds) || childNodeIds.length === 0) {
            console.warn('Mitosis component: invalid evolution parameters');
            return;
          }

          const parentNode = nodes.find(n => n && n.id === parentNodeId);
          const childNodes = childNodeIds.map(id => nodes.find(n => n && n.id === id)).filter(Boolean);

          if (!parentNode || childNodes.length === 0) {
            console.warn('Mitosis component: parent or child nodes not found');
            return;
          }

          // Validate that all required properties exist on nodes
          if (!parentNode.id || typeof parentNode.x !== 'number' || typeof parentNode.y !== 'number' ||
              typeof parentNode.radius !== 'number' || typeof parentNode.color !== 'string') {
            console.warn('Mitosis component: invalid parent node properties');
            return;
          }

          // Validate child nodes
          const validChildNodes = childNodes.filter(node =>
            node && node.id && typeof node.x === 'number' && typeof node.y === 'number' &&
            typeof node.radius === 'number' && typeof node.color === 'string'
          );

          if (validChildNodes.length === 0) {
            console.warn('Mitosis component: no valid child nodes to animate');
            return;
          }

          // Convert to the format expected by the mitosis plugin
          const formattedParent = {
            id: parentNode.id,
            x: typeof parentNode.x === 'number' && isFinite(parentNode.x) ? parentNode.x : 100, // default position if not specified
            y: typeof parentNode.y === 'number' && isFinite(parentNode.y) ? parentNode.y : 100,
            radius: typeof parentNode.radius === 'number' && isFinite(parentNode.radius) ? parentNode.radius : 20,
            color: typeof parentNode.color === 'string' ? parentNode.color : '#4F46E5',
            label: typeof parentNode.label === 'string' ? parentNode.label : ''
          };

          const formattedChildren = validChildNodes.map(node => ({
            id: node.id,
            x: typeof node.x === 'number' && isFinite(node.x) ? node.x : 150, // default position if not specified
            y: typeof node.y === 'number' && isFinite(node.y) ? node.y : 150,
            radius: typeof node.radius === 'number' && isFinite(node.radius) ? node.radius : 15,
            color: typeof node.color === 'string' ? node.color : '#60A5FA',
            label: typeof node.label === 'string' ? node.label : ''
          }));

          // Validate container ref before triggering animation
          if (!containerRef || !containerRef.current) {
            console.warn('Mitosis component: container ref not available for animation');
            return;
          }

          // Trigger the mitosis animation
          try {
            mitosisPlugin.triggerMitosisSplit({
              parentNode: formattedParent,
              childNodes: formattedChildren,
              containerRef
            });
          } catch (animationError) {
            console.warn('Mitosis component: error triggering animation:', animationError);
          }
        }
      }
    } catch (error) {
      console.warn('Mitosis component: error handling evolution:', error);
    }
  };

  return (
    <div className="visualization-container">
      <div className="visualization-header">
        <h2>OpenEvolve Evolution Visualization</h2>
        <button
          onClick={() => {
            try {
              setShowMitosisControls(prev => !prev);
            } catch (error) {
              console.warn('Mitosis component: error toggling controls:', error);
            }
          }}
          className="mitosis-toggle-btn"
        >
          {showMitosisControls ? 'Hide' : 'Show'} Mitosis Controls
        </button>
      </div>

      {showMitosisControls && (
        <div className="mitosis-controls-panel">
          <MitosisSettings
            onToggle={(enabled) => {
              try {
                setMitosisEnabled(enabled);
              } catch (error) {
                console.warn('Mitosis component: error updating mitosis enabled state:', error);
              }
            }}
          />
        </div>
      )}

      <div
        ref={containerRef}
        className="visualization-area"
        style={{
          position: 'relative',
          width: '100%',
          height: '600px',
          border: '1px solid #ddd',
          borderRadius: '4px'
        }}
      >
        {/* Existing visualization content would go here */}
        {Array.isArray(nodes) && nodes.map(node => {
          if (!node) return null;
          return (
            <div
              key={node.id || Math.random()}
              className="node"
              style={{
                position: 'absolute',
                left: `${typeof node.x === 'number' ? node.x : 100}px`,
                top: `${typeof node.y === 'number' ? node.y : 100}px`,
                width: `${(typeof node.radius === 'number' ? node.radius : 20) * 2}px`,
                height: `${(typeof node.radius === 'number' ? node.radius : 20) * 2}px`,
                backgroundColor: typeof node.color === 'string' ? node.color : '#4F46E5',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '12px',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
              onClick={() => {
                try {
                  handleEvolution(node.id, []);
                } catch (error) {
                  console.warn('Mitosis component: error on node click:', error);
                }
              }}
            >
              {node.label || node.id || 'Node'}
            </div>
          );
        })}

        {/* Render the mitosis animation component */}
        <MitosisAnimation
          parentNode={{ id: '', x: 0, y: 0, radius: 0, color: '' }}
          childNodes={[]}
          containerRef={containerRef}
          enabled={mitosisEnabled}
        />
      </div>
    </div>
  );
};

export default EnhancedVisualizationComponent;