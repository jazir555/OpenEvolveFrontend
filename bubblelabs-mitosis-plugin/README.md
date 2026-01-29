# BubbleLab Mitosis Plugin

A plugin that adds mitosis-style bubble splitting animations to BubbleLab, simulating cell division for visualizing OpenEvolve evolutions. When enabled, nodes will split into child nodes with a smooth animation that includes a bounce effect after the split.

## Integration with BubbleLab and OpenEvolve

The plugin is designed to integrate seamlessly with both the BubbleLab platform and the OpenEvolve evolution engine. It can be toggled on/off through the settings panel and works as an enhancement to the existing visualization without replacing core functionality.

### Architecture Overview

The integration follows this architecture:

```
BubbleLab Studio
├── OpenEvolve Plugin (handles evolution, adversarial, decomposition)
│   ├── Evolution Engine
│   ├── Adversarial Testing
│   ├── Decomposition Engine
│   └── Other AI capabilities
└── Mitosis Plugin (visualizes events as bubble splitting)
    ├── Mitosis Animations
    ├── Evolution Event Processor
    └── Visualization Components
```

When OpenEvolve completes an evolution, decomposition, or adversarial task, the mitosis plugin automatically creates a bubble splitting animation to visualize the results.

### BubbleLab Plugin Architecture

The mitosis plugin conforms to BubbleLab's plugin interface and can be registered in the plugin registry:

```typescript
import { OpenEvolvePlugin } from '@openevolve/plugin';
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Initialize plugins
OpenEvolvePlugin.initialize();
MitosisPlugin.initialize({
  enabled: false, // Disabled by default
  animationDuration: 1200,
  bounceIntensity: 0.2
});

export const plugins = [
  OpenEvolvePlugin,
  MitosisPlugin, // Add the mitosis plugin
  // Other plugins...
];
```

### OpenEvolve Evolution Integration

The plugin includes special integration points for OpenEvolve evolution events through the OpenEvolve plugin:

```typescript
import {
  processEvolutionEvent,
  processBatchEvolutionEvents,
  configureForOpenEvolve,
  connectToOpenEvolveEvolution,
  disconnectFromOpenEvolveEvolution,
  processOpenEvolveExecutionResult
} from '@openevolve/bubblelab-mitosis-plugin';
import { openevolvePlugin } from '@openevolve/plugin';

// Configure for optimal OpenEvolve integration
configureForOpenEvolve({
  animationDuration: 1200,
  bounceIntensity: 0.2,
  trailEffect: false,
  particleEffects: false
});

// Connect to OpenEvolve evolution events
await connectToOpenEvolveEvolution(openevolvePlugin);

// Process a single evolution event
const evolutionEvent = {
  id: 'evolution-1',
  parent: {
    id: 'parent-1',
    position: { x: 100, y: 100 },
    size: 30,
    color: '#4F46E5'
  },
  children: [{
    id: 'child-1',
    position: { x: 70, y: 130 },
    size: 20,
    color: '#60A5FA'
  }, {
    id: 'child-2',
    position: { x: 130, y: 130 },
    size: 20,
    color: '#34D399'
  }],
  timestamp: Date.now(),
  evolutionType: 'mutation'
};

await processEvolutionEvent(evolutionEvent);

// Process a survival-of-fittest evolution event
const survivalOfFittestEvent = {
  id: 'survival-1',
  parent: {
    id: 'draft-email',
    position: { x: 200, y: 150 },
    size: 30,
    color: '#4F46E5',
    label: 'Draft Email'
  },
  children: [
    { id: 'strategy-1', position: { x: 100, y: 100 }, size: 20, color: '#9CA3AF', label: 'Strategy 1' },
    { id: 'strategy-2', position: { x: 150, y: 80 }, size: 20, color: '#9CA3AF', label: 'Strategy 2' },
    { id: 'strategy-3', position: { x: 200, y: 100 }, size: 20, color: '#9CA3AF', label: 'Strategy 3' },
    { id: 'strategy-4', position: { x: 125, y: 150 }, size: 20, color: '#9CA3AF', label: 'Strategy 4' },
    { id: 'strategy-5', position: { x: 175, y: 150 }, size: 20, color: '#9CA3AF', label: 'Strategy 5' }
  ],
  timestamp: Date.now(),
  evolutionType: 'survival-of-fittest',
  metadata: {
    survivorIndices: [4] // Only the 5th strategy survives
  },
  nextEvolution: {
    // The surviving strategy splits again
    parentNode: { id: 'strategy-5', x: 175, y: 150, radius: 20, color: '#10B981', label: 'Winning Strategy' },
    childNodes: [
      { id: 'evolved-1', x: 150, y: 120, radius: 15, color: '#8B5CF6', label: 'Evolved 1' },
      { id: 'evolved-2', x: 175, y: 120, radius: 15, color: '#8B5CF6', label: 'Evolved 2' },
      { id: 'evolved-3', x: 200, y: 120, radius: 15, color: '#8B5CF6', label: 'Evolved 3' }
    ],
    containerRef: { current: document.getElementById('visualization-container') || document.body },
    evolutionType: 'standard'
  }
};

await processEvolutionEvent(survivalOfFittestEvent);

// Process multiple evolution events simultaneously
await processBatchEvolutionEvents([evolutionEvent1, evolutionEvent2]);

// Process OpenEvolve execution results to trigger mitosis animations
const executionResult = {
  executionId: 'exec-123',
  module: 'evolution',
  output: {
    bestSolution: 'Optimized solution',
    population: ['sol-1', 'sol-2', 'sol-3'],
    fitnessScores: [0.95, 0.87, 0.82],
    generations: 10,
    convergence: 0.98
  }
};

await processOpenEvolveExecutionResult(executionResult);

// Disconnect when done
disconnectFromOpenEvolveEvolution();
```

## Features

- **Mitosis Animation**: Nodes split into child nodes with a cell-division-like animation
- **Bounce Effect**: Child bubbles bounce slightly after splitting for a natural feel
- **Configurable**: Animation duration, bounce intensity, and split delay are configurable
- **Toggleable**: Feature can be enabled/disabled via settings panel
- **Non-intrusive**: Works as an addon without modifying existing BubbleLab UI
- **Robust Error Handling**: Comprehensive error handling throughout the plugin with graceful degradation
- **Input Validation**: All inputs are validated to prevent crashes from malformed data
- **Browser Compatibility**: Fallbacks for older browsers without Web Animations API
- **Sanitized Values**: All values are sanitized to prevent invalid CSS or extreme values
- **Survival-of-the-Fittest Demo**: Special "Mitosis" visual demonstration showing a single bubble splitting into 5, with 4 turning red (failed) and 1 turning green (winner), followed by the winner splitting again

## Installation

```bash
npm install @openevolve/bubblelab-mitosis-plugin
```

## Usage

### Basic Usage

```tsx
import { MitosisAnimation, mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';
// Import the plugin's CSS for proper styling and animations
import '@openevolve/bubblelab-mitosis-plugin/dist/mitosis-animations.css';

// Initialize the plugin
mitosisPlugin.initialize({
  enabled: true,
  animationDuration: 1500,
  bounceIntensity: 0.3
});

// Use the animation component for standard mitosis
<MitosisAnimation
  parentNode={parentNode}
  childNodes={childNodes}
  enabled={true}
/>

// Use the animation component for survival-of-fittest evolution
<MitosisAnimation
  parentNode={parentNode}
  childNodes={childNodes}
  enabled={true}
  evolutionType="survival-of-fittest"
  survivorIndices={[2]} // Index 2 survives
  nextEvolution={{
    // Define what happens to the survivor
    parentNode: survivorNode,
    childNodes: nextGenerationNodes,
    containerRef: containerRef,
    evolutionType: 'standard'
  }}
/>

// Or use the demo component to showcase the survival-of-the-fittest evolution
<MitosisDemo
  enabled={true}
  onDemoComplete={() => console.log('Demo completed!')}
/>
```

### With Settings Panel

```tsx
import { MitosisSettings } from '@openevolve/bubblelab-mitosis-plugin';

<MitosisSettings onToggle={(enabled) => console.log('Mitosis enabled:', enabled)} />
```

## API

### Components

- `MitosisAnimation`: The main animation component that triggers the bubble splitting effect
- `MitosisSettings`: A comprehensive settings panel with sliders and toggles to configure and customize the animation
- `MitosisDemo`: A special demonstration component showcasing the "Mitosis" visual with survival-of-the-fittest evolution (single bubble splits into 5, 4 turn red/failed and 1 turns green/winner, then winner splits again)

### Plugin Functions

- `mitosisPlugin.initialize(config)`: Initialize the plugin with configuration
- `mitosisPlugin.triggerMitosisSplit(params)`: Manually trigger a mitosis animation
- `mitosisPlugin.triggerEvolutionSplit(params)`: Trigger an evolution animation with survival-of-fittest mechanics
- `mitosisPlugin.triggerBatchMitosis(params)`: Trigger multiple mitosis animations in sequence
- `mitosisPlugin.toggleEnabled()`: Toggle the plugin on/off
- `mitosisPlugin.updateConfig(config)`: Update plugin configuration
- `mitosisPlugin.getState()`: Get current plugin state
- `mitosisPlugin.applyPreset(preset)`: Apply a predefined animation preset
- `mitosisPlugin.getPerformanceMetrics()`: Get performance metrics for the plugin

## Configuration Options

- `enabled`: Whether the animation is enabled (default: false)
- `animationDuration`: Duration of the split animation in milliseconds (default: 1500)
- `bounceIntensity`: Intensity of the bounce effect after split (0-1 scale, default: 0.3)
- `splitDelay`: Delay before the bounce effect starts in milliseconds (default: 300)
- `colorVariation`: Amount of color variation for child bubbles (0-1 scale, default: 0.1)

## Integration with BubbleLab and OpenEvolve

The plugin is designed to integrate seamlessly with both BubbleLab UI components and the OpenEvolve evolution engine. It can be toggled on/off through the settings panel and works as an enhancement to the existing visualization without replacing core functionality.

### OpenEvolve-Specific Integration

The plugin includes special integration points for OpenEvolve evolution events:

```typescript
import {
  processEvolutionEvent,
  processBatchEvolutionEvents,
  configureForOpenEvolve,
  subscribeToOpenEvolveEvents
} from '@openevolve/bubblelab-mitosis-plugin';

// Configure for optimal OpenEvolve integration
configureForOpenEvolve({
  animationDuration: 1200,
  bounceIntensity: 0.2,
  trailEffect: false,
  particleEffects: false
});

// Process a single evolution event
const evolutionEvent = {
  parentId: 'parent-1',
  parentPosition: { x: 100, y: 100 },
  parentSize: 30,
  parentColor: '#4F46E5',
  childIds: ['child-1', 'child-2'],
  childPositions: [{ x: 70, y: 130 }, { x: 130, y: 130 }],
  childSizes: [20, 20],
  childColors: ['#60A5FA', '#34D399'],
  timestamp: Date.now()
};

await processEvolutionEvent(evolutionEvent);

// Process multiple evolution events simultaneously
await processBatchEvolutionEvents([evolutionEvent1, evolutionEvent2]);
```

### Integration Steps

1. **Install the plugin**:
   ```bash
   npm install @openevolve/bubblelab-mitosis-plugin
   ```

2. **Register the plugin** in your BubbleLab application using either the registration utility or the direct plugin export:

   Option A - Using the registration utility:
   ```typescript
   import { registerMitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

   // Register the plugin with BubbleLab
   const mitosisPlugin = registerMitosisPlugin();

   // Add to your plugin registry
   export const plugins = [
     OpenEvolvePlugin,
     mitosisPlugin, // Add the registered mitosis plugin
     // Other plugins...
   ];
   ```

   Option B - Using the direct BubbleLab-compatible plugin:
   ```typescript
   import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';
   import { OpenEvolvePlugin } from '@openevolve/plugin';

   // Add to your plugin registry
   export const plugins = [
     OpenEvolvePlugin,
     MitosisPlugin, // Add the BubbleLab-compatible mitosis plugin
     // Other plugins...
   ];
   ```

3. **Initialize the plugin** when your application starts:
   ```typescript
   import { mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

   // Initialize with default or custom configuration
   mitosisPlugin.initialize({
     enabled: false,
     animationDuration: 1500,
     bounceIntensity: 0.3,
     splitDelay: 300,
     rotationIntensity: 0.2,
     opacityEffect: true,
     trailEffect: false,
     particleEffects: false
   });
   ```

4. **Enhance existing visualization components** by adding the MitosisAnimation component and MitosisSettings panel as shown in the example files.

5. **Trigger animations** when evolution events occur by calling the appropriate methods:
   - `mitosisPlugin.triggerMitosisSplit()` for single splits
   - `mitosisPlugin.triggerBatchMitosis()` for multiple simultaneous splits
   - `mitosisPlugin.applyPreset()` for predefined animation styles

### Integration Files

The following files demonstrate how to integrate the mitosis plugin with existing BubbleLab components:

- `integration/bubblelab-integration.ts` - Integration utilities
- `integration/enhanced-visualization-component.tsx` - Example enhanced visualization component

### BubbleLab Plugin Interface

The plugin implements the standard BubbleLab plugin interface with:

- **Actions**: Methods to trigger animations, update config, apply presets
- **Selectors**: Methods to get state and performance metrics
- **Components**: React components for animation and settings
- **Lifecycle**: Initialize and destroy methods for proper resource management

### Modifying Existing UI

To add the mitosis effect to existing BubbleLab visualizations:

1. Add the MitosisSettings component to your visualization controls for user customization
2. Add the MitosisAnimation component to your visualization area
3. Call `triggerMitosisForEvolution()` when evolution events occur
4. Use the comprehensive settings panel to allow users to customize animation parameters
5. Monitor performance metrics through the settings panel

The plugin is designed as a non-intrusive enhancement that preserves all existing functionality while adding the optional mitosis animation effect with full customization capabilities.