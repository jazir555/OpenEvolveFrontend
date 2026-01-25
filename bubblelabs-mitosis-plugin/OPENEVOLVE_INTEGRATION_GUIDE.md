# Complete Integration Guide: Mitosis Plugin with OpenEvolve Plugin

This guide explains how to fully integrate the Mitosis Bubble Splitting plugin with the OpenEvolve plugin in BubbleLab.

## 1. Understanding the Architecture

The OpenEvolve plugin is a comprehensive BubbleLab plugin that handles evolution, adversarial, decomposition, and other AI capabilities. The mitosis plugin integrates with this system to provide visualizations of evolution events.

```
BubbleLab Studio
├── OpenEvolve Plugin (handles evolution, adversarial, decomposition)
├── Mitosis Plugin (visualizes evolution events as bubble splitting)
└── Other plugins...
```

## 2. Installation

First, install both plugins:

```bash
npm install @openevolve/plugin
npm install @openevolve/bubblelab-mitosis-plugin
```

## 3. Registering Both Plugins

Update the BubbleLab plugin registry at `BubbleLab/apps/bubble-studio/src/plugins/index.ts`:

```typescript
import { OpenEvolvePlugin } from '@openevolve/plugin';
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Initialize plugins with their default configurations
OpenEvolvePlugin.initialize({
  // OpenEvolve plugin configuration
  evolutionConfig: {
    maxIterations: 100,
    populationSize: 50,
    mutationRate: 0.1
  },
  adversarialConfig: {
    redTeamSize: 3,
    blueTeamSize: 3,
    maxRounds: 5
  },
  decompositionConfig: {
    maxSubProblems: 5,
    strategy: 'hierarchical'
  }
});

MitosisPlugin.initialize({
  enabled: false, // Disabled by default
  animationDuration: 1200,
  bounceIntensity: 0.2,
  trailEffect: false,
  particleEffects: false
});

export const plugins = [
  OpenEvolvePlugin,    // Main OpenEvolve functionality
  MitosisPlugin,       // Mitosis bubble splitting animations
  // Add more plugins here as needed
];

export default plugins;
```

## 4. Connecting the Plugins

To connect the mitosis plugin to OpenEvolve's evolution events:

```typescript
import { openevolvePlugin } from '@openevolve/plugin';
import { 
  connectToOpenEvolveEvolution, 
  disconnectFromOpenEvolveEvolution,
  processOpenEvolveExecutionResult,
  configureForOpenEvolve
} from '@openevolve/bubblelab-mitosis-plugin';

// Configure the mitosis plugin for optimal OpenEvolve integration
configureForOpenEvolve({
  animationDuration: 1200,
  bounceIntensity: 0.2,
  trailEffect: false,
  particleEffects: false
});

// Connect to OpenEvolve evolution events
await connectToOpenEvolveEvolution(openevolvePlugin);

// Example: Execute an evolution and visualize the result
try {
  const evolutionResult = await openevolvePlugin.executeEvolution(
    "Optimize neural network architecture for MNIST classification",
    {
      evolutionConfig: {
        maxIterations: 50,
        populationSize: 30
      }
    }
  );

  // Process the result to trigger mitosis animation
  await processOpenEvolveExecutionResult(evolutionResult);
} catch (error) {
  console.error('Evolution execution failed:', error);
}

// Clean up when your application unmounts
const cleanup = () => {
  disconnectFromOpenEvolveEvolution();
};
```

## 5. Using in Visualization Components

The mitosis plugin can be used in BubbleLab visualization components to show evolution events:

```tsx
import React, { useEffect, useRef } from 'react';
import { openevolvePlugin } from '@openevolve/plugin';
import { 
  MitosisAnimation, 
  MitosisSettings,
  connectToOpenEvolveEvolution,
  processOpenEvolveExecutionResult
} from '@openevolve/bubblelab-mitosis-plugin';

const EvolutionVisualization = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [evolutionResults, setEvolutionResults] = React.useState<any[]>([]);

  useEffect(() => {
    const setupEvolutionIntegration = async () => {
      // Connect to OpenEvolve evolution events
      await connectToOpenEvolveEvolution(openevolvePlugin);
    };

    setupEvolutionIntegration();

    // Clean up on unmount
    return () => {
      // Cleanup would happen elsewhere
    };
  }, []);

  // Function to run an evolution and visualize it
  const runEvolution = async () => {
    try {
      const result = await openevolvePlugin.executeEvolution(
        "Optimize algorithm performance",
        { evolutionConfig: { maxIterations: 20 } }
      );

      // Add to results list
      setEvolutionResults(prev => [...prev, result]);

      // Process the result to trigger mitosis animation
      await processOpenEvolveExecutionResult(result);
    } catch (error) {
      console.error('Evolution failed:', error);
    }
  };

  return (
    <div className="evolution-visualization">
      <div className="controls">
        <button onClick={runEvolution}>Run Evolution</button>
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

## 6. Processing Different OpenEvolve Execution Types

The mitosis plugin can visualize different types of OpenEvolve executions:

```typescript
import { 
  processOpenEvolveExecutionResult,
  processEvolutionEvent,
  processBatchEvolutionEvents
} from '@openevolve/bubblelab-mitosis-plugin';

// Process evolution results
const evolutionResult = await openevolvePlugin.executeEvolution(goal, config);
await processOpenEvolveExecutionResult(evolutionResult);

// Process adversarial results
const adversarialResult = await openevolvePlugin.executeAdversarial(content, config);
await processOpenEvolveExecutionResult(adversarialResult);

// Process decomposition results
const decompositionResult = await openevolvePlugin.executeDecomposition(problem, config);
await processOpenEvolveExecutionResult(decompositionResult);

// Process integrated results
const integratedResult = await openevolvePlugin.executeIntegrated(goal, config);
await processOpenEvolveExecutionResult(integratedResult);
```

## 7. Configuration Options

The mitosis plugin can be configured with various options optimized for different visualization needs:

```typescript
import { configureForOpenEvolve } from '@openevolve/bubblelab-mitosis-plugin';

// Smooth preset - for continuous evolution visualization
configureForOpenEvolve({
  animationDuration: 2000,
  bounceIntensity: 0.1,
  rotationIntensity: 0.1,
  opacityEffect: true,
  trailEffect: false,
  easingFunction: 'cubic-bezier(0.23, 1, 0.32, 1)'
});

// Dramatic preset - for highlighting major evolutionary jumps
configureForOpenEvolve({
  animationDuration: 1800,
  bounceIntensity: 0.6,
  rotationIntensity: 0.7,
  opacityEffect: true,
  trailEffect: true,
  particleEffects: true,
  easingFunction: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)'
});

// Fast preset - for rapid succession of events
configureForOpenEvolve({
  animationDuration: 800,
  bounceIntensity: 0.2,
  rotationIntensity: 0.3,
  opacityEffect: true,
  trailEffect: false,
  particleEffects: false,
  easingFunction: 'ease-in-out'
});
```

## 8. Performance Considerations

The mitosis plugin is optimized for performance when integrated with OpenEvolve:

- Animation throttling (max 5 concurrent animations)
- Efficient DOM operations using document fragments
- Cleanup of animation resources
- Performance metrics tracking

Monitor performance using:

```typescript
import { mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Get performance metrics
const metrics = mitosisPlugin.getPerformanceMetrics();
console.log(`Avg. Duration: ${metrics.avgDuration}ms`);
console.log(`Active Animations: ${metrics.activeAnimations}`);
console.log(`Queued Animations: ${metrics.queuedAnimations}`);
```

## 9. Error Handling

Both plugins include comprehensive error handling:

```typescript
// Set up error handling for OpenEvolve plugin
openevolvePlugin.hooks = {
  onError: (serviceId, error) => {
    console.error(`OpenEvolve error in ${serviceId}:`, error);
  }
};

// The mitosis plugin also has built-in error handling
```

This integration ensures that the mitosis plugin works seamlessly with the OpenEvolve plugin, providing rich visualization capabilities for OpenEvolve's evolutionary processes within the BubbleLab platform.