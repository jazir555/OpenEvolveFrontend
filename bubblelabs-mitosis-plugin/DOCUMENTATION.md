# BubbleLab Mitosis Plugin Documentation

## Overview

The BubbleLab Mitosis Plugin adds mitosis-style bubble splitting animations to BubbleLab, simulating cell division for visualizing OpenEvolve evolutions. When enabled, nodes will split into child nodes with a smooth animation that includes a bounce effect after the split.

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
- **Performance Optimized**: Throttled animations to prevent performance issues
- **Accessible**: Proper ARIA attributes and keyboard navigation support

## Installation

```bash
npm install @openevolve/bubblelab-mitosis-plugin
```

## Basic Usage

### Import and Initialize

```tsx
import { MitosisAnimation, MitosisSettings, mitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';
// Import the plugin's CSS for proper styling and animations
import '@openevolve/bubblelab-mitosis-plugin/dist/mitosis-animations.css';

// Initialize the plugin with configuration
mitosisPlugin.initialize({
  enabled: true,
  animationDuration: 1500,
  bounceIntensity: 0.3,
  splitDelay: 300
});
```

### Using the Animation Component

```tsx
import React from 'react';
import { MitosisAnimation } from '@openevolve/bubblelab-mitosis-plugin';

const MyVisualization = () => {
  const parentNode = {
    id: 'parent-1',
    x: 100,
    y: 100,
    radius: 30,
    color: '#4F46E5',
    label: 'Parent'
  };

  const childNodes = [
    {
      id: 'child-1',
      x: 70,
      y: 130,
      radius: 20,
      color: '#60A5FA',
      label: 'Child 1'
    },
    {
      id: 'child-2',
      x: 130,
      y: 130,
      radius: 20,
      color: '#34D399',
      label: 'Child 2'
    }
  ];

  return (
    <div style={{ width: '100%', height: '500px', position: 'relative' }}>
      <MitosisAnimation
        parentNode={parentNode}
        childNodes={childNodes}
        enabled={true}
      />
    </div>
  );
};
```

### Using the Settings Panel

```tsx
import React from 'react';
import { MitosisSettings } from '@openevolve/bubblelab-mitosis-plugin';

const MySettingsPanel = () => {
  return (
    <MitosisSettings 
      onToggle={(enabled) => console.log('Mitosis enabled:', enabled)} 
    />
  );
};
```

## API Reference

### Components

#### `MitosisAnimation`

The main animation component that triggers the bubble splitting effect.

**Props:**
- `parentNode` (optional): The parent node that will split
  - `id`: Unique identifier
  - `x`, `y`: Position coordinates
  - `radius`: Size of the bubble
  - `color`: Color of the bubble
  - `label` (optional): Label text
- `childNodes` (optional): Array of child nodes to create after split
- `containerRef` (optional): React ref to the container element
- `enabled` (optional): Whether the animation is enabled (default: true)

#### `MitosisSettings`

A comprehensive settings panel with sliders and toggles to configure and customize the animation.

**Props:**
- `onToggle` (optional): Callback when the enabled state changes

The settings panel includes:
- Master toggle to enable/disable the entire animation system
- Sliders for adjusting animation duration, bounce intensity, rotation intensity, and split delay
- Checkboxes for enabling/disabling individual visual effects (opacity, motion trails, particle effects)
- Dropdown for selecting different easing functions
- Preset buttons for quick configuration of common animation styles
- Real-time performance metrics display

### Plugin Functions

#### `mitosisPlugin.initialize(config)`

Initialize the plugin with configuration.

**Parameters:**
- `config`: Configuration object
  - `enabled`: Whether the animation is enabled (default: false)
  - `animationDuration`: Duration of the split animation in milliseconds (default: 1500)
  - `bounceIntensity`: Intensity of the bounce effect after split (0-1 scale, default: 0.3)
  - `splitDelay`: Delay before the bounce effect starts in milliseconds (default: 300)
  - `colorVariation`: Amount of color variation for child bubbles (0-1 scale, default: 0.1)
  - `rotationIntensity`: Intensity of rotation effect during split (0-1 scale, default: 0.2)
  - `opacityEffect`: Whether to include opacity changes during animation (default: true)
  - `trailEffect`: Whether to show motion trails during animation (default: false)
  - `easingFunction`: CSS easing function for the animation (default: 'cubic-bezier(0.25, 0.1, 0.25, 1)')
  - `particleEffects`: Whether to show particle effects during split (default: false)

#### `mitosisPlugin.triggerMitosisSplit(params)`

Manually trigger a mitosis animation.

**Parameters:**
- `params`: Object with parameters
  - `parentNode`: The parent node to split
  - `childNodes`: Array of child nodes to create
  - `containerRef`: React ref to the container element

#### `mitosisPlugin.triggerBatchMitosis(params)`

Trigger multiple mitosis animations in sequence.

**Parameters:**
- `params`: Object with parameters
  - `parentNodes`: Array of parent nodes to split
  - `childNodeGroups`: Array of arrays of child nodes to create for each parent
  - `containerRef`: React ref to the container element

#### `mitosisPlugin.updateConfig(config)`

Update plugin configuration.

**Parameters:**
- `config`: Partial configuration object with values to update

#### `mitosisPlugin.getState()`

Get current plugin state.

**Returns:**
- State object with current configuration and status

#### `mitosisPlugin.toggleEnabled()`

Toggle the plugin on/off.

#### `mitosisPlugin.isEnabled()`

Check if the plugin is currently enabled.

#### `mitosisPlugin.applyPreset(preset)`

Apply a predefined animation preset.

**Parameters:**
- `preset`: One of 'default', 'smooth', 'dramatic', 'subtle', 'fast', or 'custom'

#### `mitosisPlugin.getPerformanceMetrics()`

Get performance metrics for the plugin.

**Returns:**
- Object with performance metrics including average duration, active animations, and queued animations

#### `mitosisPlugin.cleanup()`

Clean up all active animations and reset the plugin state.

## Integration with BubbleLab

The plugin is designed to integrate seamlessly with existing BubbleLab UI components. It can be toggled on/off through the settings panel and works as an enhancement to the existing visualization without replacing core functionality.

### Integration Steps

1. **Install the plugin**:
   ```bash
   npm install @openevolve/bubblelab-mitosis-plugin
   ```

2. **Update the plugin registry** in your BubbleLab application:

   Option A - Using the registration utility:
   ```typescript
   import { registerMitosisPlugin } from '@openevolve/bubblelab-mitosis-plugin';

   // Register the plugin with BubbleLab
   const mitosisPlugin = registerMitosisPlugin();

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

   export const plugins = [
     OpenEvolvePlugin,
     MitosisPlugin, // Add the BubbleLab-compatible mitosis plugin
     // Other plugins...
   ];
   ```

3. **Enhance existing visualization components** by adding the MitosisAnimation component and MitosisSettings panel as shown in the example files.

4. **Trigger animations** when evolution events occur by calling `mitosisPlugin.triggerMitosisSplit()` with the appropriate parameters.

## Performance Considerations

- The plugin implements throttling to limit concurrent animations (maximum 5 at a time)
- Animations are queued to prevent performance degradation
- Hardware acceleration is used for smoother animations
- DOM elements are properly cleaned up after animations complete

## Accessibility

- Proper ARIA labels and roles are implemented
- Keyboard navigation is supported
- Screen reader-friendly markup is used
- Sufficient color contrast is maintained

## Browser Support

- Modern browsers with Web Animations API (Chrome, Firefox, Safari, Edge)
- Fallbacks for browsers without Web Animations API using CSS transitions
- IE 11+ support with reduced functionality

## Troubleshooting

### Common Issues

1. **Animations not appearing**: Check that the container element has `position: relative` or similar positioning context
2. **Performance issues**: The plugin limits concurrent animations to 5; adjust this by modifying the `maxConcurrentAnimations` property in the source
3. **Elements not cleaning up**: Ensure the `MitosisAnimation` component is properly unmounted when no longer needed

### Debugging

Enable debug logging by setting the log level:

```tsx
import { logger } from '@openevolve/bubblelab-mitosis-plugin';

logger.setLevel('debug');
```

## Examples

### Complete Example with Error Boundary

```tsx
import React from 'react';
import { MitosisAnimation, MitosisSettings, MitosisErrorBoundary } from '@openevolve/bubblelab-mitosis-plugin';

const CompleteExample = () => {
  const [evolutionData, setEvolutionData] = React.useState(null);

  // Simulate evolution event
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setEvolutionData({
        parentNode: {
          id: 'parent-1',
          x: 200,
          y: 200,
          radius: 30,
          color: '#4F46E5',
          label: 'Parent'
        },
        childNodes: [
          {
            id: 'child-1',
            x: 150,
            y: 250,
            radius: 20,
            color: '#60A5FA',
            label: 'Child 1'
          },
          {
            id: 'child-2',
            x: 250,
            y: 250,
            radius: 20,
            color: '#34D399',
            label: 'Child 2'
          }
        ]
      });
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div>
      <MitosisSettings />
      
      <MitosisErrorBoundary>
        <div style={{ width: '100%', height: '500px', position: 'relative' }}>
          {evolutionData && (
            <MitosisAnimation
              parentNode={evolutionData.parentNode}
              childNodes={evolutionData.childNodes}
              enabled={true}
            />
          )}
        </div>
      </MitosisErrorBoundary>
    </div>
  );
};
```