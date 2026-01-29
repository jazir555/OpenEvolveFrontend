# Complete Integration Guide: Mitosis Plugin with BubbleLab and OpenEvolve

This guide explains how to fully integrate the Mitosis Bubble Splitting plugin with both the BubbleLab platform and the OpenEvolve evolution engine.

## 1. Installation

```bash
npm install @openevolve/bubblelab-mitosis-plugin
```

## 2. BubbleLab Plugin Registration

Update the BubbleLab plugin registry at `BubbleLab/apps/bubble-studio/src/plugins/index.ts`:

```typescript
import { OpenEvolvePlugin } from '@openevolve/plugin';
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

export const plugins = [
  OpenEvolvePlugin,    // Main OpenEvolve functionality
  MitosisPlugin,       // Mitosis bubble splitting animations
  // Add more plugins here as needed
];

// Helper functions for plugin management
export function getPluginById(id: string) {
  return plugins.find((plugin) => plugin.id === id);
}

export function getEnabledPlugins() {
  return plugins.filter((plugin) => plugin.capabilities);
}
```

## 3. OpenEvolve Event Integration

The mitosis plugin provides special integration points for OpenEvolve evolution events:

### Configure for OpenEvolve

```typescript
import { configureForOpenEvolve } from '@openevolve/bubblelab-mitosis-plugin';

// Configure the plugin for optimal OpenEvolve integration
configureForOpenEvolve({
  animationDuration: 1200,        // Slightly faster for evolution sequences
  bounceIntensity: 0.2,         // Moderate bounce for visual interest
  splitDelay: 200,              // Quick delay before bounce
  colorVariation: 0.15,         // Small color variation for offspring
  rotationIntensity: 0.1,       // Gentle rotation
  opacityEffect: true,          // Enable opacity transitions
  trailEffect: false,           // Disable trails for cleaner evolution view
  easingFunction: 'ease-out',   // Clean ending to animations
  particleEffects: false        // Disable particles for performance
});
```

### Process Evolution Events

```typescript
import { processEvolutionEvent, processBatchEvolutionEvents } from '@openevolve/bubblelab-mitosis-plugin';

// Process a single evolution event from OpenEvolve
const evolutionEvent = {
  parentId: 'parent-1',
  parentPosition: { x: 100, y: 100 },
  parentSize: 30,
  parentColor: '#4F46E5',
  childIds: ['child-1', 'child-2'],
  childPositions: [{ x: 70, y: 130 }, { x: 130, y: 130 }],
  childSizes: [20, 20],
  childColors: ['#60A5FA', '#34D399'],
  timestamp: Date.now(),
  metadata: {
    containerId: 'evolution-visualization'  // Optional: specify container
  }
};

await processEvolutionEvent(evolutionEvent);

// Process multiple evolution events simultaneously
await processBatchEvolutionEvents([evolutionEvent1, evolutionEvent2]);
```

## 4. Visualization Component Integration

Integrate the mitosis visualization into your evolution visualization components:

```tsx
import React, { useEffect } from 'react';
import { MitosisAnimation, MitosisSettings } from '@openevolve/bubblelab-mitosis-plugin';

const EvolutionVisualization = ({ evolutionEvents }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Process evolution events when they occur
    evolutionEvents.forEach(event => {
      // The mitosis plugin will handle the visualization automatically
      // when processEvolutionEvent is called
    });
  }, [evolutionEvents]);

  return (
    <div>
      <div className="controls">
        <MitosisSettings />
      </div>
      
      <div 
        ref={containerRef}
        id="evolution-visualization"
        className="evolution-container"
        style={{ 
          position: 'relative', 
          width: '100%', 
          height: '600px',
          border: '1px solid #ccc'
        }}
      >
        {/* Your existing visualization elements */}
        <MitosisAnimation 
          enabled={true}
          containerRef={containerRef}
        />
      </div>
    </div>
  );
};
```

## 5. Connecting to OpenEvolve Event System

To connect the mitosis plugin to OpenEvolve's event system, you can create a bridge:

```typescript
import { processEvolutionEvent } from '@openevolve/bubblelab-mitosis-plugin';

// Example of connecting to OpenEvolve's event system
class OpenEvolveMitosisBridge {
  constructor() {
    this.subscribeToEvolutionEvents();
  }

  private subscribeToEvolutionEvents() {
    // Subscribe to OpenEvolve's evolution events
    // This is pseudocode - actual implementation depends on OpenEvolve's event system
    OpenEvolve.events.on('evolution.created', (evolutionData) => {
      this.handleEvolutionEvent(evolutionData);
    });
  }

  private async handleEvolutionEvent(evolutionData: any) {
    // Convert OpenEvolve event data to mitosis format
    const mitosisEvent = {
      parentId: evolutionData.parentId,
      parentPosition: {
        x: evolutionData.parentPosition.x,
        y: evolutionData.parentPosition.y
      },
      parentSize: evolutionData.parentSize || 30,
      parentColor: evolutionData.parentColor || '#4F46E5',
      childIds: evolutionData.children.map(child => child.id),
      childPositions: evolutionData.children.map(child => ({
        x: child.position.x,
        y: child.position.y
      })),
      childSizes: evolutionData.children.map(child => child.size || 20),
      childColors: evolutionData.children.map(child => child.color || '#60A5FA'),
      timestamp: Date.now(),
      metadata: {
        containerId: 'evolution-visualization',
        evolutionType: evolutionData.type
      }
    };

    // Process the event with the mitosis plugin
    await processEvolutionEvent(mitosisEvent);
  }
}

// Initialize the bridge
const bridge = new OpenEvolveMitosisBridge();
```

## 6. Performance Optimization

The mitosis plugin includes built-in performance optimizations:

- Animation throttling (max 5 concurrent animations)
- Efficient DOM operations using document fragments
- Cleanup of animation resources
- Performance metrics tracking

You can monitor performance using:

```typescript
import { mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Get performance metrics
const metrics = mitosisPlugin.getPerformanceMetrics();
console.log(`Avg. Duration: ${metrics.avgDuration}ms`);
console.log(`Active Animations: ${metrics.activeAnimations}`);
console.log(`Queued Animations: ${metrics.queuedAnimations}`);
```

## 7. Preset Configurations

The plugin includes several animation presets optimized for different scenarios:

```typescript
import { mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Apply a preset optimized for evolution visualization
mitosisPlugin.applyPreset('smooth');  // For gentle, continuous evolution
mitosisPlugin.applyPreset('fast');    // For rapid succession of events
mitosisPlugin.applyPreset('dramatic'); // For emphasis on major evolutionary jumps
```

This integration ensures that the mitosis plugin works seamlessly with both BubbleLab's plugin architecture and OpenEvolve's evolution engine, providing rich visualization capabilities for evolutionary processes.