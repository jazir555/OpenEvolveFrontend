# BubbleLab Plugin Integration Guide

This guide explains how to properly integrate the Mitosis Bubble Splitting plugin with the BubbleLab platform.

## 1. Understanding BubbleLab's Plugin Architecture

BubbleLab uses a modular plugin system where each plugin must conform to the `BubbleLabPlugin` interface. The system loads plugins from the `plugins/index.ts` file in the BubbleLab studio application.

## 2. Installing the Mitosis Plugin

First, install the plugin in your BubbleLab environment:

```bash
npm install @openevolve/bubblelab-mitosis-plugin
```

## 3. Registering the Plugin with BubbleLab

Update the BubbleLab plugin registry at `BubbleLab/apps/bubble-studio/src/plugins/index.ts`:

```typescript
import { OpenEvolvePlugin } from '@openevolve/plugin';
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

// Initialize plugins with their default configurations
OpenEvolvePlugin.initialize();
MitosisPlugin.initialize({
  enabled: false, // Disabled by default
  animationDuration: 1200,
  bounceIntensity: 0.2,
  splitDelay: 200
});

/**
 * Plugin registry
 *
 * Add new plugins to this array to make them available in the application.
 * Each plugin must implement the BubbleLabPlugin interface.
 */
export const plugins = [
  OpenEvolvePlugin,    // Main OpenEvolve functionality
  MitosisPlugin,       // Mitosis bubble splitting animations
  // Add more plugins here as needed
];

/**
 * Get plugin by ID
 */
export function getPluginById(id: string) {
  return plugins.find((plugin) => plugin.id === id);
}

/**
 * Get all enabled plugins
 */
export function getEnabledPlugins() {
  return plugins.filter((plugin) => {
    // Check if plugin has an enabled capability
    return plugin.capabilities?.enabled !== false;
  });
}

/**
 * Get plugins by capability
 */
export function getPluginsByCapability(capability: string) {
  return plugins.filter((plugin) =>
    plugin.capabilities?.[capability]
  );
}

export default plugins;
```

## 4. Using the Plugin in BubbleLab Components

The mitosis plugin provides React components that can be used in BubbleLab UI:

### In Visualization Components

```tsx
import React, { useEffect, useRef } from 'react';
import { MitosisAnimation, MitosisSettings } from '@openevolve/bubblelab-mitosis-plugin';

const BubbleLabVisualization = ({ evolutionData }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Process evolution events to trigger mitosis animations
  useEffect(() => {
    if (evolutionData && evolutionData.length > 0) {
      // Process each evolution event
      evolutionData.forEach(event => {
        // The mitosis plugin will automatically handle visualization
        // when evolution events occur
      });
    }
  }, [evolutionData]);

  return (
    <div className="bubblelab-visualization">
      <div className="visualization-controls">
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

## 5. Connecting to OpenEvolve Evolution Events

To connect the mitosis plugin to OpenEvolve's evolution engine, use the evolution integration:

```typescript
import { 
  connectToOpenEvolveEvolution, 
  disconnectFromOpenEvolveEvolution,
  MitosisPlugin
} from '@openevolve/bubblelab-mitosis-plugin';
import { createClient } from '@openevolve/integration-library';

// Initialize OpenEvolve client
const openEvolveClient = createClient({
  baseUrl: process.env.REACT_APP_OPENEVOLVE_API_URL || 'http://localhost:8000'
});

// Connect to evolution events
const setupEvolutionIntegration = async () => {
  try {
    // Initialize the mitosis plugin
    MitosisPlugin.initialize({
      enabled: true,
      animationDuration: 1200,
      bounceIntensity: 0.2,
      trailEffect: false,
      particleEffects: false
    });

    // Connect to OpenEvolve evolution events
    await connectToOpenEvolveEvolution(openEvolveClient);

    console.log('Successfully connected to OpenEvolve evolution events');
  } catch (error) {
    console.error('Failed to connect to OpenEvolve evolution events:', error);
  }
};

// Call this when your application initializes
setupEvolutionIntegration();

// Clean up when your application unmounts
const cleanup = () => {
  disconnectFromOpenEvolveEvolution();
};

// Example of how to process evolution data
const processEvolutionEvent = async (evolutionEvent) => {
  // The integration will automatically trigger mitosis animations
  // when evolution events occur through the OpenEvolve client
};
```

## 6. BubbleLab UI Integration

To integrate the mitosis plugin into BubbleLab's UI system, you can add it to existing visualization components:

### Adding to BubbleLab's Main Dashboard

```tsx
// In BubbleLab/apps/bubble-studio/src/pages/Dashboard.tsx
import React from 'react';
import { MitosisSettings } from '@openevolve/bubblelab-mitosis-plugin';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>BubbleLab Dashboard</h1>
        <div className="plugin-controls">
          <MitosisSettings />
        </div>
      </header>
      
      <main className="dashboard-main">
        {/* Your existing dashboard content */}
      </main>
    </div>
  );
};

export default Dashboard;
```

## 7. Configuration and Customization

The mitosis plugin can be customized through its configuration options:

```typescript
// Configure the plugin with custom settings
MitosisPlugin.initialize({
  enabled: true,                    // Enable the plugin
  animationDuration: 1500,          // Duration of split animation in ms
  bounceIntensity: 0.3,             // Intensity of bounce effect (0-1)
  splitDelay: 300,                  // Delay before bounce in ms
  colorVariation: 0.1,              // Color variation for child bubbles (0-1)
  rotationIntensity: 0.2,           // Rotation effect intensity (0-1)
  opacityEffect: true,              // Enable opacity transitions
  trailEffect: false,               // Show motion trails
  easingFunction: 'ease-out',       // CSS easing function
  particleEffects: false            // Show particle effects
});
```

## 8. Lifecycle Management

Properly manage the plugin lifecycle in your BubbleLab application:

```typescript
// In your main application component
import { MitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

const App = () => {
  useEffect(() => {
    // Initialize the plugin when app mounts
    MitosisPlugin.initialize({
      enabled: true
    });

    // Clean up when app unmounts
    return () => {
      MitosisPlugin.destroy();
    };
  }, []);

  return (
    <div className="app">
      {/* Your app content */}
    </div>
  );
};
```

## 9. Error Handling and Debugging

The plugin includes comprehensive error handling:

```typescript
// Listen for plugin errors
MitosisPlugin.hooks = {
  ...MitosisPlugin.hooks,
  onError: async (serviceId, error) => {
    console.error(`[MitosisPlugin] Error in ${serviceId}:`, error);
    // Handle errors appropriately in your application
  }
};
```

## 10. Performance Considerations

The mitosis plugin is optimized for performance:

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

This integration ensures that the mitosis plugin works seamlessly within the BubbleLab ecosystem while providing rich visualization capabilities for OpenEvolve's evolutionary processes.