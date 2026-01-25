import React, { useState, useRef, useEffect } from 'react';
import { mitosisPlugin } from '../utils/createMitosisPlugin';
import { BubbleNode } from '../types/plugin-types';

interface MitosisDemoProps {
  containerRef?: React.RefObject<HTMLDivElement>;
  enabled?: boolean;
  onDemoComplete?: () => void;
}

export const MitosisDemo: React.FC<MitosisDemoProps> = ({
  containerRef: externalContainerRef,
  enabled = true,
  onDemoComplete
}) => {
  const internalContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = externalContainerRef || internalContainerRef;
  const [demoStage, setDemoStage] = useState<'idle' | 'firstSplit' | 'selection' | 'secondSplit'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [showNarrative, setShowNarrative] = useState(true);

  // Function to start the demo
  const startDemo = async () => {
    if (!enabled) return;

    setDemoStage('firstSplit');

    try {
      // Get container element
      const container = containerRef.current;
      if (!container) {
        setError('Container not available');
        return;
      }

      // Clear any existing bubbles
      try {
        container.innerHTML = '';
      } catch (clearError) {
        console.warn('Error clearing container:', clearError);
        // Continue anyway
      }

      // Create parent bubble "Draft Email"
      const parentBubble: BubbleNode = {
        id: 'draft-email-parent',
        x: 200,
        y: 150,
        radius: 30,
        color: '#4F46E5', // Blue color
        label: 'Draft Email'
      };

      // Create 5 child bubbles in a circular pattern around the parent
      const childBubbles: BubbleNode[] = [];
      const centerX = 200;
      const centerY = 150;
      const radius = 100; // Distance from center
      const childRadius = 20;

      for (let i = 0; i < 5; i++) {
        const angle = (i * (2 * Math.PI)) / 5; // Divide circle into 5 equal parts
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);

        childBubbles.push({
          id: `child-${i}`,
          x,
          y,
          radius: childRadius,
          color: '#9CA3AF', // Gray color initially
          label: `Strategy ${i + 1}`
        });
      }

      // Trigger the first split animation
      try {
        await mitosisPlugin.triggerEvolutionSplit({
          parentNode: parentBubble,
          childNodes: childBubbles,
          containerRef,
          evolutionType: 'survival-of-fittest',
          survivorIndices: [4], // Index 4 (last bubble) survives
          nextEvolution: {
            // Define the second evolution for the survivor
            parentNode: {
              ...childBubbles[4], // The survivor becomes parent
              color: '#10B981', // Green color for winner
              label: `Winner - ${childBubbles[4].label}`
            },
            childNodes: (() => {
              // Create 3 new child bubbles from the winner
              const secondGenerationBubbles: BubbleNode[] = [];
              const winnerCenterX = childBubbles[4].x;
              const winnerCenterY = childBubbles[4].y;
              const secondGenRadius = 60; // Smaller radius for second gen
              const secondGenChildRadius = 15;

              for (let i = 0; i < 3; i++) {
                const angle = (i * (2 * Math.PI)) / 3; // Divide circle into 3 equal parts
                const x = winnerCenterX + secondGenRadius * Math.cos(angle);
                const y = winnerCenterY + secondGenRadius * Math.sin(angle);

                secondGenerationBubbles.push({
                  id: `second-gen-${i}`,
                  x,
                  y,
                  radius: secondGenChildRadius,
                  color: '#8B5CF6', // Purple color for evolved bubbles
                  label: `Evolved ${i + 1}`
                });
              }
              return secondGenerationBubbles;
            })(),
            containerRef,
            evolutionType: 'standard'
          }
        });

        // Demo complete after all evolutions
        setTimeout(() => {
          setDemoStage('idle');
          onDemoComplete?.();
        }, 4000); // Wait for all animations to complete
      } catch (evolutionError) {
        console.error('Evolution animation failed, falling back to standard animation:', evolutionError);

        // Fallback to standard animation
        try {
          await mitosisPlugin.triggerMitosisSplit({
            parentNode: parentBubble,
            childNodes: childBubbles,
            containerRef
          });

          // After first split, wait a moment then show selection manually
          setTimeout(() => {
            setDemoStage('selection');

            // Change colors: 4 red (failed), 1 green (winner)
            childBubbles.forEach((bubble, index) => {
              try {
                const bubbleElement = container.querySelector(`[data-bubble-id="${bubble.id}"]`) as HTMLElement;
                if (bubbleElement) {
                  if (index < 4) {
                    // Failed bubbles turn red
                    bubbleElement.style.backgroundColor = '#EF4444'; // Red

                    // Update label if it exists
                    const labelElement = bubbleElement.querySelector('.mitosis-bubble-label');
                    if (labelElement) {
                      labelElement.textContent = `Failed - Strategy ${index + 1}`;
                    }
                  } else {
                    // Winner bubble turns green
                    bubbleElement.style.backgroundColor = '#10B981'; // Green

                    // Update label if it exists
                    const labelElement = bubbleElement.querySelector('.mitosis-bubble-label');
                    if (labelElement) {
                      labelElement.textContent = `Winner - Strategy ${index + 1}`;
                    }
                  }
                }
              } catch (updateError) {
                console.warn('Error updating bubble appearance:', updateError);
                // Continue with other bubbles
              }
            });

            // After showing selection, wait and then split the winner
            setTimeout(() => {
              setDemoStage('secondSplit');

              // Create the second split from the winner bubble
              const winnerIndex = 4; // The last bubble (green one)
              const winnerBubble = childBubbles[winnerIndex];

              // Create 3 new child bubbles from the winner
              const secondGenerationBubbles: BubbleNode[] = [];
              const winnerCenterX = winnerBubble.x;
              const winnerCenterY = winnerBubble.y;
              const secondGenRadius = 60; // Smaller radius for second gen
              const secondGenChildRadius = 15;

              for (let i = 0; i < 3; i++) {
                const angle = (i * (2 * Math.PI)) / 3; // Divide circle into 3 equal parts
                const x = winnerCenterX + secondGenRadius * Math.cos(angle);
                const y = winnerCenterY + secondGenRadius * Math.sin(angle);

                secondGenerationBubbles.push({
                  id: `second-gen-${i}`,
                  x,
                  y,
                  radius: secondGenChildRadius,
                  color: '#8B5CF6', // Purple color for evolved bubbles
                  label: `Evolved ${i + 1}`
                });
              }

              // Trigger the second split animation
              mitosisPlugin.triggerMitosisSplit({
                parentNode: winnerBubble,
                childNodes: secondGenerationBubbles,
                containerRef
              }).then(() => {
                // Demo complete
                setDemoStage('idle');
                onDemoComplete?.();
              }).catch(secondSplitError => {
                console.error('Second split animation failed:', secondSplitError);
                setDemoStage('idle');
                onDemoComplete?.();
              });
            }, 2000); // Wait 2 seconds before second split
          }, 1500); // Wait 1.5 seconds after first split
        } catch (fallbackError) {
          setError('Demo animation failed');
          console.error('Mitosis demo fallback error:', fallbackError);
          setDemoStage('idle');
        }
      }
    } catch (err) {
      setError('Demo setup failed');
      console.error('Mitosis demo setup error:', err);
      setDemoStage('idle');
    }
  };

  // Start the demo when component mounts
  useEffect(() => {
    if (enabled) {
      startDemo();
    }
  }, [enabled]);

  return (
    <div
      ref={containerRef || internalContainerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '500px',
        overflow: 'visible',
        backgroundColor: '#f9fafb',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        margin: '20px 0'
      }}
      aria-label="Mitosis demo animation container"
      role="region"
    >
      {showNarrative && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          right: '10px',
          backgroundColor: 'rgba(255,255,255,0.9)',
          padding: '10px',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          zIndex: 10001,
          fontSize: '14px',
          lineHeight: '1.4'
        }}>
          <strong>The "Mitosis" Visual:</strong> Watch the system evolve! A single bubble labeled "Draft Email" splits into 5 different strategies. Then, 4 bubbles turn red (Failed/Killed) while 1 turns green (Winner). The green winner splits again, demonstrating survival-of-the-fittest evolution.
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'red',
          fontWeight: 'bold',
          zIndex: 10002
        }}>
          {error}
        </div>
      )}

      {demoStage === 'idle' && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          color: '#6b7280',
          zIndex: 10002
        }}>
          <p>Demo complete. Ready for next demonstration.</p>
          <button
            onClick={startDemo}
            style={{
              marginTop: '10px',
              padding: '8px 16px',
              backgroundColor: '#4F46E5',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Restart Demo
          </button>
          <div style={{ marginTop: '10px' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={showNarrative}
                onChange={(e) => setShowNarrative(e.target.checked)}
                style={{ marginRight: '5px' }}
              />
              Show Narrative
            </label>
          </div>
        </div>
      )}

      {demoStage !== 'idle' && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '5px 10px',
          borderRadius: '4px',
          fontSize: '12px',
          zIndex: 10000
        }}>
          Stage: {demoStage}
        </div>
      )}
    </div>
  );
};