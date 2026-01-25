import React, { useState, useEffect } from 'react';
import { mitosisPlugin } from '../utils/createMitosisPlugin';
import { logger } from '../utils/logger';

interface MitosisSettingsProps {
  onToggle?: (enabled: boolean) => void;
}

export const MitosisSettings: React.FC<MitosisSettingsProps> = ({ onToggle }) => {
  const [isEnabled, setIsEnabled] = useState(() => {
    try {
      logger.debug('Getting initial plugin state');
      const state = mitosisPlugin.getState();
      const result = typeof state.enabled === 'boolean' ? state.enabled : false;
      logger.debug('Initial plugin state retrieved', { enabled: result });
      return result;
    } catch (error) {
      logger.error('Error getting initial state:', error);
      return false; // Default to disabled if there's an error
    }
  });

  const [config, setConfig] = useState(() => {
    try {
      logger.debug('Getting initial config');
      const state = mitosisPlugin.getState();
      const stateConfig = state.config || {};
      // Validate and sanitize config values
      const configResult = {
        enabled: typeof stateConfig.enabled === 'boolean' ? stateConfig.enabled : false,
        animationDuration: typeof stateConfig.animationDuration === 'number' && isFinite(stateConfig.animationDuration)
          ? Math.max(100, Math.min(10000, stateConfig.animationDuration)) : 1500,
        bounceIntensity: typeof stateConfig.bounceIntensity === 'number' && isFinite(stateConfig.bounceIntensity)
          ? Math.max(0, Math.min(1, stateConfig.bounceIntensity)) : 0.3,
        splitDelay: typeof stateConfig.splitDelay === 'number' && isFinite(stateConfig.splitDelay)
          ? Math.max(0, Math.min(5000, stateConfig.splitDelay)) : 300,
        colorVariation: typeof stateConfig.colorVariation === 'number' && isFinite(stateConfig.colorVariation)
          ? Math.max(0, Math.min(1, stateConfig.colorVariation)) : 0.1,
        rotationIntensity: typeof stateConfig.rotationIntensity === 'number' && isFinite(stateConfig.rotationIntensity)
          ? Math.max(0, Math.min(1, stateConfig.rotationIntensity)) : 0.2,
        opacityEffect: typeof stateConfig.opacityEffect === 'boolean' ? stateConfig.opacityEffect : true,
        trailEffect: typeof stateConfig.trailEffect === 'boolean' ? stateConfig.trailEffect : false,
        easingFunction: typeof stateConfig.easingFunction === 'string' ? stateConfig.easingFunction : 'cubic-bezier(0.25, 0.1, 0.25, 1)',
        particleEffects: typeof stateConfig.particleEffects === 'boolean' ? stateConfig.particleEffects : false
      };
      logger.debug('Initial config retrieved', configResult);
      return configResult;
    } catch (error) {
      logger.error('Error getting initial config:', error);
      // Return default config if there's an error
      const defaultConfig = {
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
      };
      logger.debug('Using default config', defaultConfig);
      return defaultConfig;
    }
  });

  const [performanceMetrics, setPerformanceMetrics] = useState(() => {
    try {
      return mitosisPlugin.getPerformanceMetrics();
    } catch (error) {
      logger.error('Error getting initial performance metrics:', error);
      return {
        avgDuration: 0,
        activeAnimations: 0,
        queuedAnimations: 0
      };
    }
  });

  useEffect(() => {
    logger.info('MitosisSettings component mounted, setting up state sync');
    try {
      const state = mitosisPlugin.getState();
      if (typeof state.enabled === 'boolean') {
        setIsEnabled(state.enabled);
      }
      if (state.config) {
        const stateConfig = state.config;
        // Validate and sanitize config values
        const sanitizedConfig = {
          enabled: typeof stateConfig.enabled === 'boolean' ? stateConfig.enabled : config.enabled,
          animationDuration: typeof stateConfig.animationDuration === 'number' && isFinite(stateConfig.animationDuration)
            ? Math.max(100, Math.min(10000, stateConfig.animationDuration)) : config.animationDuration,
          bounceIntensity: typeof stateConfig.bounceIntensity === 'number' && isFinite(stateConfig.bounceIntensity)
            ? Math.max(0, Math.min(1, stateConfig.bounceIntensity)) : config.bounceIntensity,
          splitDelay: typeof stateConfig.splitDelay === 'number' && isFinite(stateConfig.splitDelay)
            ? Math.max(0, Math.min(5000, stateConfig.splitDelay)) : config.splitDelay,
          colorVariation: typeof stateConfig.colorVariation === 'number' && isFinite(stateConfig.colorVariation)
            ? Math.max(0, Math.min(1, stateConfig.colorVariation)) : config.colorVariation,
          rotationIntensity: typeof stateConfig.rotationIntensity === 'number' && isFinite(stateConfig.rotationIntensity)
            ? Math.max(0, Math.min(1, stateConfig.rotationIntensity)) : config.rotationIntensity,
          opacityEffect: typeof stateConfig.opacityEffect === 'boolean' ? stateConfig.opacityEffect : config.opacityEffect,
          trailEffect: typeof stateConfig.trailEffect === 'boolean' ? stateConfig.trailEffect : config.trailEffect,
          easingFunction: typeof stateConfig.easingFunction === 'string' ? stateConfig.easingFunction : config.easingFunction,
          particleEffects: typeof stateConfig.particleEffects === 'boolean' ? stateConfig.particleEffects : config.particleEffects
        };
        setConfig(sanitizedConfig);
        logger.debug('Settings state updated', sanitizedConfig);
      }
    } catch (error) {
      logger.error('Error updating state:', error);
    }
  }, []);

  // Update performance metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      try {
        const metrics = mitosisPlugin.getPerformanceMetrics();
        setPerformanceMetrics(metrics);
      } catch (error) {
        logger.error('Error updating performance metrics:', error);
      }
    }, 1000); // Update every second

    // Cleanup interval on unmount
    return () => {
      clearInterval(interval);
    };
  }, []);

  const handleToggle = () => {
    logger.info('Toggling mitosis plugin enabled state');
    try {
      mitosisPlugin.toggleEnabled();
      const newState = mitosisPlugin.getState();
      if (typeof newState.enabled === 'boolean') {
        setIsEnabled(newState.enabled);
        logger.info(`Plugin toggled to: ${newState.enabled}`);
        onToggle?.(newState.enabled);
      }
    } catch (error) {
      logger.error('Error toggling plugin:', error);
    }
  };

  const handleConfigChange = (key: keyof typeof config, value: any) => {
    logger.debug(`Config change for ${key}: ${value}`);
    try {
      // Validate the value before applying
      let validatedValue = value;

      if (key === 'animationDuration' || key === 'splitDelay') {
        const parsedValue = parseInt(value, 10);
        validatedValue = Number.isFinite(parsedValue) ? parsedValue : config[key];
        // Sanitize to safe range
        if (key === 'animationDuration') {
          validatedValue = Math.max(100, Math.min(10000, validatedValue));
        } else if (key === 'splitDelay') {
          validatedValue = Math.max(0, Math.min(5000, validatedValue));
        }
      } else if (key === 'bounceIntensity' || key === 'colorVariation' || key === 'rotationIntensity') {
        const parsedValue = parseFloat(value);
        validatedValue = Number.isFinite(parsedValue) ? parsedValue : config[key];
        // Sanitize to 0-1 range
        validatedValue = Math.max(0, Math.min(1, validatedValue));
      } else if (key === 'enabled' || key === 'opacityEffect' || key === 'trailEffect' || key === 'particleEffects') {
        validatedValue = Boolean(value);
      } else if (key === 'easingFunction') {
        validatedValue = typeof value === 'string' ? value : config[key];
      }

      const newConfig = { ...config, [key]: validatedValue };
      setConfig(newConfig);
      mitosisPlugin.updateConfig({ [key]: validatedValue });
      logger.info(`Config updated for ${key}: ${validatedValue}`);
    } catch (error) {
      logger.error('Error updating config:', error);
    }
  };

  return (
    <div
      className="mitosis-settings-panel"
      style={{
        padding: '1rem',
        border: '1px solid #ccc',
        borderRadius: '4px',
        backgroundColor: '#f9f9f9',
        maxWidth: '500px',
        fontFamily: 'Arial, sans-serif'
      }}
      role="form"
      aria-label="Mitosis Animation Settings"
    >
      <h3 style={{ marginBottom: '1rem', color: '#333' }}>Mitosis Animation Settings</h3>

      <div style={{ marginBottom: '1rem' }}>
        <label
          htmlFor="mitosis-toggle"
          style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
        >
          <input
            id="mitosis-toggle"
            type="checkbox"
            checked={isEnabled}
            onChange={handleToggle}
            style={{ marginRight: '0.5rem', width: '18px', height: '18px' }}
            aria-label={isEnabled ? "Disable mitosis animation" : "Enable mitosis animation"}
          />
          <strong>Enable Mitosis Bubble Splitting</strong>
        </label>
      </div>

      {isEnabled && (
        <div style={{ paddingLeft: '0.5rem' }} aria-live="polite">
          {/* Animation Duration Slider */}
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="animation-duration" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Animation Duration: {config.animationDuration}ms
            </label>
            <input
              id="animation-duration"
              type="range"
              min="100"
              max="3000"
              value={config.animationDuration}
              onChange={(e) => handleConfigChange('animationDuration', e.target.value)}
              disabled={!isEnabled}
              style={{ width: '100%', padding: '0.2rem' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8em', color: '#666' }}>
              <span>Fast (100ms)</span>
              <span>Slow (3000ms)</span>
            </div>
          </div>

          {/* Bounce Intensity Slider */}
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="bounce-intensity" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Bounce Intensity: {(config.bounceIntensity * 100).toFixed(0)}%
            </label>
            <input
              id="bounce-intensity"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config.bounceIntensity}
              onChange={(e) => handleConfigChange('bounceIntensity', e.target.value)}
              disabled={!isEnabled}
              style={{ width: '100%', padding: '0.2rem' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8em', color: '#666' }}>
              <span>No Bounce</span>
              <span>High Bounce</span>
            </div>
          </div>

          {/* Rotation Intensity Slider */}
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="rotation-intensity" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Rotation Intensity: {(config.rotationIntensity * 100).toFixed(0)}%
            </label>
            <input
              id="rotation-intensity"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config.rotationIntensity}
              onChange={(e) => handleConfigChange('rotationIntensity', e.target.value)}
              disabled={!isEnabled}
              style={{ width: '100%', padding: '0.2rem' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8em', color: '#666' }}>
              <span>No Rotation</span>
              <span>Full Rotation</span>
            </div>
          </div>

          {/* Split Delay Slider */}
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="split-delay" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Split Delay: {config.splitDelay}ms
            </label>
            <input
              id="split-delay"
              type="range"
              min="0"
              max="1000"
              value={config.splitDelay}
              onChange={(e) => handleConfigChange('splitDelay', e.target.value)}
              disabled={!isEnabled}
              style={{ width: '100%', padding: '0.2rem' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8em', color: '#666' }}>
              <span>No Delay</span>
              <span>1 Second</span>
            </div>
          </div>

          {/* Effect Toggles */}
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ marginBottom: '0.5rem', color: '#333' }}>Visual Effects</h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={config.opacityEffect}
                  onChange={(e) => handleConfigChange('opacityEffect', e.target.checked)}
                  disabled={!isEnabled}
                  style={{ marginRight: '0.5rem', width: '16px', height: '16px' }}
                />
                Enable Opacity Effects
              </label>

              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={config.trailEffect}
                  onChange={(e) => handleConfigChange('trailEffect', e.target.checked)}
                  disabled={!isEnabled}
                  style={{ marginRight: '0.5rem', width: '16px', height: '16px' }}
                />
                Enable Motion Trails
              </label>

              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={config.particleEffects}
                  onChange={(e) => handleConfigChange('particleEffects', e.target.checked)}
                  disabled={!isEnabled}
                  style={{ marginRight: '0.5rem', width: '16px', height: '16px' }}
                />
                Enable Particle Effects
              </label>
            </div>
          </div>

          {/* Easing Function Selector */}
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="easing-function" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Easing Function
            </label>
            <select
              id="easing-function"
              value={config.easingFunction}
              onChange={(e) => handleConfigChange('easingFunction', e.target.value)}
              disabled={!isEnabled}
              style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
            >
              <option value="ease">ease</option>
              <option value="ease-in">ease-in</option>
              <option value="ease-out">ease-out</option>
              <option value="ease-in-out">ease-in-out</option>
              <option value="linear">linear</option>
              <option value="cubic-bezier(0.25, 0.1, 0.25, 1)">Standard (default)</option>
              <option value="cubic-bezier(0.42, 0, 0.58, 1)">Ease-in-out Sine</option>
              <option value="cubic-bezier(0.68, -0.55, 0.265, 1.55)">Custom Elastic</option>
            </select>
          </div>

          {/* Preset Buttons */}
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ marginBottom: '0.5rem', color: '#333' }}>Animation Presets</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
              <button
                type="button"
                onClick={() => mitosisPlugin.applyPreset('smooth')}
                style={{
                  padding: '0.5rem',
                  backgroundColor: '#e0e7ff',
                  border: '1px solid #93c5fd',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Smooth
              </button>
              <button
                type="button"
                onClick={() => mitosisPlugin.applyPreset('dramatic')}
                style={{
                  padding: '0.5rem',
                  backgroundColor: '#fde68a',
                  border: '1px solid #fbbf24',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Dramatic
              </button>
              <button
                type="button"
                onClick={() => mitosisPlugin.applyPreset('subtle')}
                style={{
                  padding: '0.5rem',
                  backgroundColor: '#d1fae5',
                  border: '1px solid #6ee7b7',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Subtle
              </button>
              <button
                type="button"
                onClick={() => mitosisPlugin.applyPreset('fast')}
                style={{
                  padding: '0.5rem',
                  backgroundColor: '#fecaca',
                  border: '1px solid #f87171',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Fast
              </button>
            </div>
          </div>

          {/* Performance Metrics */}
          <div style={{
            padding: '0.75rem',
            backgroundColor: '#f0f9ff',
            border: '1px solid #bae6fd',
            borderRadius: '4px',
            marginBottom: '1rem'
          }}>
            <h4 style={{ marginBottom: '0.5rem', color: '#333' }}>Performance Metrics</h4>
            <div style={{ fontSize: '0.9em' }}>
              <div>Avg. Duration: {performanceMetrics.avgDuration.toFixed(2)}ms</div>
              <div>Active Animations: {performanceMetrics.activeAnimations}</div>
              <div>Queued Animations: {performanceMetrics.queuedAnimations}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};